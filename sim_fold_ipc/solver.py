"""Incremental-potential (implicit-Euler) solver with IPC contact.

Each timestep minimises

    E(x, phi) = inertia + E_stretch_shear + E_bend_twist + E_barrier

over node positions ``x`` and rotation-vector increments ``phi`` (orientations
are reconstructed as ``exp(phi) * q_prev``).  Minimisation is an L-BFGS descent
(the plan's "Option B") with a backtracking line search that (a) rejects steps
leaving the barrier's feasible region and (b) is capped by a conservative CCD
bound so no node travels more than half its current wall clearance per step —
which is what guarantees zero tunnelling.

Gradients come from Warp's reverse-mode autodiff over the energy kernels.
"""
from __future__ import annotations

import numpy as np
import warp as wp

from . import elasticity as el
from . import barrier as bar
from .rod import RodState


# --------------------------------------------------------------------------- #
# host-side quaternion helpers (x, y, z, w)
# --------------------------------------------------------------------------- #
def _quat_exp_np(phi: np.ndarray) -> np.ndarray:
    th = np.linalg.norm(phi, axis=1)
    out = np.zeros((len(phi), 4), dtype=np.float64)
    small = th < 1e-6
    out[small, 3] = 1.0
    out[small, :3] = 0.5 * phi[small]
    ns = ~small
    half = 0.5 * th[ns]
    s = np.sin(half) / th[ns]
    out[ns, 0] = s * phi[ns, 0]
    out[ns, 1] = s * phi[ns, 1]
    out[ns, 2] = s * phi[ns, 2]
    out[ns, 3] = np.cos(half)
    return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-30)


def _quat_mul_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx, by, bz, bw = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], axis=1)


class Solver:
    def __init__(
        self,
        rod: RodState,
        dt: float = 5.0e-3,
        gravity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        self.rod = rod
        self.dt = float(dt)
        self.gravity = np.asarray(gravity, dtype=np.float64)
        self.device = rod.device
        n = rod.n

        # optimisation variables (autodiff)
        self.x_var = wp.zeros(n, dtype=wp.vec3, device=self.device, requires_grad=True)
        self.phi_var = wp.zeros(n, dtype=wp.vec3, device=self.device, requires_grad=True)
        self.x_pred = wp.zeros(n, dtype=wp.vec3, device=self.device)
        self.phi_pred = wp.zeros(n, dtype=wp.vec3, device=self.device)
        self.E = wp.zeros(1, dtype=wp.float64, device=self.device, requires_grad=True)
        self.gap = wp.zeros(n, dtype=wp.float32, device=self.device)

        # boundary conditions: per-node fixed flag + prescribed x / phi
        self.fixed = np.zeros(n, dtype=bool)
        self.x_bc = rod.positions().astype(np.float64)
        self.phi_bc = np.zeros((n, 3), dtype=np.float64)

        # vessel / barrier (optional)
        self.mesh: wp.Mesh | None = None
        self.rod_radius = rod.params.radius
        self.d_hat = 5.0e-4
        self.kappa = 1.0e3
        self.weight = wp.array(np.ones(n, dtype=np.float32), dtype=wp.float32,
                               device=self.device)

        self.precond = self._build_precond()

    # ------------------------------------------------------------------ #
    def _build_precond(self) -> np.ndarray:
        """Diagonal Hessian approximation (6N,) used as the L-BFGS H0.

        Well-scaled preconditioning is what lets the descent handle stiff
        elasticity: without it a unit gradient step is meters-large and
        backtracking collapses to no progress (the earlier free-fall bug).
        """
        rod = self.rod
        n = rod.n
        p = rod.params
        l0 = rod.l0.numpy().astype(np.float64)
        mass = rod.mass.numpy().astype(np.float64)
        rot = rod.rot_inertia.numpy().astype(np.float64)
        inv_dt2 = 1.0 / self.dt ** 2

        inv_l_sum = np.zeros(n)          # sum of 1/l0 over adjacent segments
        l_sum = np.zeros(n)
        inv_l_sum[:-1] += 1.0 / l0
        inv_l_sum[1:] += 1.0 / l0
        l_sum[:-1] += l0
        l_sum[1:] += l0

        p_pos = mass * inv_dt2 + (p.k_stretch + p.k_shear) * inv_l_sum
        p_phi = (rot * inv_dt2
                 + (p.k_bend + p.k_twist) * inv_l_sum
                 + p.k_shear * l_sum)
        p_pos = np.maximum(p_pos, 1e-8)
        p_phi = np.maximum(p_phi, 1e-8)

        precond = np.zeros((n, 6))
        precond[:, 0:3] = p_pos[:, None]
        precond[:, 3:6] = p_phi[:, None]
        return precond.reshape(-1)

    # ------------------------------------------------------------------ #
    def set_vessel(self, mesh: wp.Mesh, d_hat: float = 5e-4, kappa: float = 1e3):
        self.mesh = mesh
        self.d_hat = float(d_hat)
        self.kappa = float(kappa)

    def set_fixed(self, indices, positions=None, phis=None):
        for k, i in enumerate(indices):
            self.fixed[i] = True
            if positions is not None:
                self.x_bc[i] = positions[k]
            if phis is not None:
                self.phi_bc[i] = phis[k]

    # ------------------------------------------------------------------ #
    def _free_mask(self) -> np.ndarray:
        """(6N,) boolean mask, True where DOF is free to move."""
        n = self.rod.n
        m = np.ones((n, 6), dtype=bool)
        m[self.fixed, :] = False
        return m.reshape(-1)

    def _energy(self, apply_barrier: bool = True) -> float:
        self.E.zero_()
        wp.launch(el.inertia_energy, dim=self.rod.n,
                  inputs=[self.x_var, self.phi_var, self.x_pred, self.phi_pred,
                          self.rod.mass, self.rod.rot_inertia,
                          wp.float32(1.0 / self.dt ** 2), self.E],
                  device=self.device)
        wp.launch(el.stretch_shear_energy, dim=self.rod.n - 1,
                  inputs=[self.x_var, self.phi_var, self.rod.q, self.rod.l0,
                          wp.float32(self.rod.params.k_stretch),
                          wp.float32(self.rod.params.k_shear), self.E],
                  device=self.device)
        wp.launch(el.bend_twist_energy, dim=self.rod.n - 1,
                  inputs=[self.phi_var, self.rod.q, self.rod.l0,
                          wp.float32(self.rod.params.k_bend),
                          wp.float32(self.rod.params.k_twist), self.E],
                  device=self.device)
        if apply_barrier and self.mesh is not None:
            wp.launch(bar.barrier_energy, dim=self.rod.n,
                      inputs=[self.x_var, self.mesh.id,
                              wp.float32(self.rod_radius), wp.float32(self.d_hat),
                              wp.float32(self.kappa), self.weight, self.E],
                      device=self.device)
        return float(self.E.numpy()[0])

    def _min_gap(self, xz: np.ndarray) -> float:
        if self.mesh is None:
            return 1e30
        self.x_var.assign(xz.reshape(-1, 3).astype(np.float32))
        wp.launch(bar.node_gaps, dim=self.rod.n,
                  inputs=[self.x_var, self.mesh.id, wp.float32(self.rod_radius),
                          wp.float32(self.d_hat + self.rod_radius + 1e-3), self.gap],
                  device=self.device)
        return float(self.gap.numpy().min())

    def _eval(self, z: np.ndarray):
        """Return (energy, gradient) of the incremental potential at z."""
        n = self.rod.n
        x = z[: 3 * n].reshape(n, 3).astype(np.float32)
        phi = z[3 * n:].reshape(n, 3).astype(np.float32)
        self.x_var.assign(x)
        self.phi_var.assign(phi)
        self.x_var.grad.zero_()
        self.phi_var.grad.zero_()

        tape = wp.Tape()
        with tape:
            e = self._energy()
        tape.backward(loss=self.E)
        gx = self.x_var.grad.numpy().reshape(-1)
        gp = self.phi_var.grad.numpy().reshape(-1)
        g = np.concatenate([gx, gp]).astype(np.float64)
        g[~self._free_mask()] = 0.0
        return e, g

    # ------------------------------------------------------------------ #
    def step(self, max_iter: int = 40, tol: float = 1e-6, history: int = 8,
             verbose: bool = False):
        n = self.rod.n
        dt = self.dt
        xp = self.rod.positions().astype(np.float64)
        vp = self.rod.v.numpy().astype(np.float64)
        op = self.rod.omega.numpy().astype(np.float64)

        # predicted (free-flight) state
        x_pred = xp + dt * vp + dt * dt * self.gravity[None, :]
        phi_pred = dt * op
        # honour Dirichlet targets in the prediction
        x_pred[self.fixed] = self.x_bc[self.fixed]
        phi_pred[self.fixed] = self.phi_bc[self.fixed]
        self.x_pred.assign(x_pred.astype(np.float32))
        self.phi_pred.assign(phi_pred.astype(np.float32))

        # initial guess = prediction (feasible if previous state was feasible)
        z0 = np.concatenate([x_pred.reshape(-1), phi_pred.reshape(-1)])
        free = self._free_mask()

        # ---- solve via dense regularised Newton -------------------------
        # The stretch/shear modes are ~A/I ~ 1e7-1e8 stiffer than bending, so
        # the Hessian is severely ill-conditioned and strongly couples
        # positions and orientations.  First-order methods stall on this; a
        # Newton step on the true (small, 6N ~ 150) Hessian handles it.  The
        # Hessian is built by forward-differencing the autodiff gradient over
        # the free DOFs, LM-regularised to SPD, and factored densely.
        free_idx = np.where(free)[0]
        nf = free_idx.size
        h_fd = 1.0e-6 / np.sqrt(self.precond)      # per-DOF FD step
        last_alpha = 0.0

        z = z0.copy()
        e, g = self._eval(z)
        g0n = np.linalg.norm(g[free]) + 1e-30

        for outer in range(max_iter):
            gn = np.linalg.norm(g[free])
            if gn < max(1e-10, 1e-6 * g0n):
                break

            H = np.empty((nf, nf))
            gf = g[free]
            for a, j in enumerate(free_idx):
                zj = z.copy()
                zj[j] += h_fd[j]
                gp = self._eval(zj)[1]
                H[:, a] = (gp[free] - gf) / h_fd[j]
            H = 0.5 * (H + H.T)
            H += (1e-7 * (np.trace(H) / nf) + 1e-14) * np.eye(nf)  # LM -> SPD

            try:
                dz_free = np.linalg.solve(H, -gf)
            except np.linalg.LinAlgError:
                dz_free = -gf / np.clip(np.diag(H), 1e-12, None)
            if gf @ dz_free > 0.0:                 # guarantee descent
                dz_free = -gf / np.clip(np.diag(H), 1e-12, None)

            dz = np.zeros_like(z)
            dz[free] = dz_free

            # CCD cap: no node advances past half its wall clearance
            alpha = 1.0
            if self.mesh is not None:
                mg0 = max(self._min_gap(z[: 3 * n]), 0.0)
                max_disp = np.linalg.norm(
                    dz[: 3 * n].reshape(n, 3), axis=1).max() + 1e-30
                alpha = min(1.0, 0.5 * mg0 / max_disp)

            gTd = g @ dz
            accepted = False
            for _ls in range(40):
                z_try = z + alpha * dz
                if self.mesh is not None and self._min_gap(z_try[: 3 * n]) <= 0.0:
                    alpha *= 0.5
                    continue
                e_try, g_try = self._eval(z_try)
                if e_try <= e + 1e-4 * alpha * gTd:
                    z, e, g = z_try, e_try, g_try
                    accepted = True
                    break
                alpha *= 0.5
            last_alpha = alpha
            if not accepted:
                break

        if verbose:
            print(f"    newton it={outer + 1} E={e:.6e} "
                  f"|g|={np.linalg.norm(g[free]):.2e} alpha={last_alpha:.3f}")

        # ---- commit new state -------------------------------------------
        x_new = z[: 3 * n].reshape(n, 3)
        phi_new = z[3 * n:].reshape(n, 3)
        x_new[self.fixed] = self.x_bc[self.fixed]
        phi_new[self.fixed] = self.phi_bc[self.fixed]

        q_prev = self.rod.quaternions().astype(np.float64)
        q_new = _quat_mul_np(_quat_exp_np(phi_new), q_prev)
        q_new /= np.linalg.norm(q_new, axis=1, keepdims=True) + 1e-30

        v_new = (x_new - xp) / dt
        omega_new = phi_new / dt

        self.rod.x.assign(x_new.astype(np.float32))
        self.rod.q.assign(q_new.astype(np.float32))
        self.rod.v.assign(v_new.astype(np.float32))
        self.rod.omega.assign(omega_new.astype(np.float32))
        return {"energy": e, "grad_norm": float(np.linalg.norm(g[free])),
                "nit": int(outer + 1), "ccd_alpha": last_alpha}

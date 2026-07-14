"""Overdamped incremental-potential solver for the positions-primary rod.

Each step minimises

    E(x) = (1/2h^2) sum m_i |x_i - x_i^prev|^2   (proximal / overdamped)
         + E_stretch(x) + E_bend(x) + kappa * B(x)

over node positions with a dense regularised Newton step (the system is small
and, with a moderate stretch stiffness, well-conditioned).  The overdamped
prediction (no momentum carry-over) makes the relaxation unconditionally stable
and monotone — the correct regime for quasi-static endovascular navigation.

Non-penetration is guaranteed by a conservative CCD cap on the committed step
(no node advances more than half its current wall clearance).
"""
from __future__ import annotations

import numpy as np
import warp as wp

from . import barrier as bar
from . import elastic_pos as ep
from .rod import RodState


def curvature_binormal(x: np.ndarray) -> np.ndarray:
    """Discrete curvature binormal at interior nodes (N-2, 3)."""
    e_prev = x[1:-1] - x[:-2]
    e_next = x[2:] - x[1:-1]
    denom = (np.linalg.norm(e_prev, axis=1) * np.linalg.norm(e_next, axis=1)
             + np.einsum("ij,ij->i", e_prev, e_next))
    return 2.0 * np.cross(e_prev, e_next) / (denom[:, None] + 1e-12)


class PosSolver:
    def __init__(self, rod: RodState, dt: float = 0.4,
                 k_bend: float | None = None, stretch_ratio: float = 200.0):
        self.rod = rod
        self.dt = float(dt)
        self.device = rod.device
        n = rod.n

        l0 = rod.l0.numpy().astype(np.float64)
        lmean = float(l0.mean())
        self.k_bend = float(k_bend if k_bend is not None else rod.params.k_bend)
        self.k_stretch = float(stretch_ratio * self.k_bend / lmean ** 2)

        # rest voronoi length at each interior node
        l_bar = 0.5 * (l0[:-1] + l0[1:])
        self.l_bar = wp.array(l_bar.astype(np.float32), dtype=wp.float32,
                              device=self.device)
        # rest curvature = 0 (straight rest shape)
        self.kappa_rest = wp.zeros(n - 2, dtype=wp.vec3, device=self.device)

        self.x_var = wp.zeros(n, dtype=wp.vec3, device=self.device,
                              requires_grad=True)
        self.x_pred = wp.zeros(n, dtype=wp.vec3, device=self.device)
        self.E = wp.zeros(1, dtype=wp.float64, device=self.device,
                          requires_grad=True)
        self.gap = wp.zeros(n, dtype=wp.float32, device=self.device)

        self.fixed = np.zeros(n, dtype=bool)
        self.x_bc = rod.positions().astype(np.float64)

        self.mesh: wp.Mesh | None = None
        self.rod_radius = rod.params.radius
        self.d_hat = 4.0e-4
        self.kappa = 1.0e2

        self.weight = wp.array(np.ones(n, dtype=np.float32), dtype=wp.float32,
                               device=self.device)
        self.precond = self._build_precond()

    # ------------------------------------------------------------------ #
    def set_vessel(self, mesh: wp.Mesh, d_hat: float = 4e-4, kappa: float = 1e2):
        self.mesh = mesh
        self.d_hat = float(d_hat)
        self.kappa = float(kappa)

    def set_fixed(self, indices, positions=None):
        for k, i in enumerate(indices):
            self.fixed[i] = True
            if positions is not None:
                self.x_bc[i] = positions[k]

    def _build_precond(self) -> np.ndarray:
        n = self.rod.n
        l0 = self.rod.l0.numpy().astype(np.float64)
        mass = self.rod.mass.numpy().astype(np.float64)
        inv_l = np.zeros(n)
        inv_l[:-1] += 1.0 / l0
        inv_l[1:] += 1.0 / l0
        p = (mass / self.dt ** 2
             + self.k_stretch * inv_l
             + 4.0 * self.k_bend * inv_l / (l0.mean() ** 2))
        return np.maximum(p, 1e-9).repeat(3)

    # ------------------------------------------------------------------ #
    def _energy(self) -> float:
        self.E.zero_()
        n = self.rod.n
        wp.launch(ep.inertia_energy_pos, dim=n,
                  inputs=[self.x_var, self.x_pred, self.rod.mass,
                          wp.float32(1.0 / self.dt ** 2), self.E],
                  device=self.device)
        wp.launch(ep.stretch_energy_pos, dim=n - 1,
                  inputs=[self.x_var, self.rod.l0,
                          wp.float32(self.k_stretch), self.E],
                  device=self.device)
        wp.launch(ep.bend_energy_pos, dim=n - 2,
                  inputs=[self.x_var, self.l_bar, self.kappa_rest,
                          wp.float32(self.k_bend), self.E],
                  device=self.device)
        if self.mesh is not None:
            wp.launch(bar.barrier_energy, dim=n,
                      inputs=[self.x_var, self.mesh.id,
                              wp.float32(self.rod_radius), wp.float32(self.d_hat),
                              wp.float32(self.kappa), self.weight, self.E],
                      device=self.device)
        return float(self.E.numpy()[0])

    def _eval(self, xz: np.ndarray):
        n = self.rod.n
        self.x_var.assign(xz.reshape(n, 3).astype(np.float32))
        self.x_var.grad.zero_()
        tape = wp.Tape()
        with tape:
            e = self._energy()
        tape.backward(loss=self.E)
        g = self.x_var.grad.numpy().reshape(-1).astype(np.float64)
        g[np.repeat(self.fixed, 3)] = 0.0
        return e, g

    def _min_gap(self, xz: np.ndarray) -> float:
        if self.mesh is None:
            return 1e30
        self.x_var.assign(xz.reshape(-1, 3).astype(np.float32))
        wp.launch(bar.node_gaps, dim=self.rod.n,
                  inputs=[self.x_var, self.mesh.id, wp.float32(self.rod_radius),
                          wp.float32(self.d_hat + self.rod_radius + 1e-3), self.gap],
                  device=self.device)
        return float(self.gap.numpy().min())

    # ------------------------------------------------------------------ #
    def step(self, max_iter: int = 12, verbose: bool = False):
        n = self.rod.n
        free = ~np.repeat(self.fixed, 3)
        free_idx = np.where(free)[0]
        nf = free_idx.size

        xp = self.rod.positions().astype(np.float64)
        x_pred = xp.copy()
        x_pred[self.fixed] = self.x_bc[self.fixed]      # overdamped + Dirichlet
        self.x_pred.assign(x_pred.astype(np.float32))

        z = x_pred.reshape(-1).copy()
        # FD step in metres: large enough that the *soft bending* Hessian
        # entries sit well above the float32 gradient-noise floor (a step tied
        # to the stiff stretch precond is far too small and loses bending).
        h_fd = np.full_like(z, 1.0e-7)
        e, g = self._eval(z)
        g0n = np.linalg.norm(g[free]) + 1e-30
        last_alpha = 0.0

        def build_H(z_at, g_at):
            # FD Hessian on free DOFs (modified Newton: reused across iters)
            Hm = np.empty((nf, nf))
            gf_at = g_at[free]
            for a, j in enumerate(free_idx):
                zj = z_at.copy()
                zj[j] += h_fd[j]
                Hm[:, a] = (self._eval(zj)[1][free] - gf_at) / h_fd[j]
            Hm = 0.5 * (Hm + Hm.T)
            Hm += (1e-7 * (np.trace(Hm) / nf) + 1e-14) * np.eye(nf)
            return Hm

        H = build_H(z, g)                 # built once, refreshed every few iters
        for outer in range(max_iter):
            gn = np.linalg.norm(g[free])
            if gn < max(1e-11, 1e-6 * g0n):
                break
            if outer > 0 and outer % 5 == 0:   # occasional refresh
                H = build_H(z, g)
            gf = g[free]
            try:
                dzf = np.linalg.solve(H, -gf)
            except np.linalg.LinAlgError:
                dzf = -gf / np.clip(np.diag(H), 1e-12, None)
            if gf @ dzf > 0.0:
                dzf = -gf / np.clip(np.diag(H), 1e-12, None)
            dz = np.zeros_like(z)
            dz[free] = dzf

            alpha = 1.0
            if self.mesh is not None:
                mg0 = max(self._min_gap(z), 0.0)
                max_disp = np.linalg.norm(dz.reshape(n, 3), axis=1).max() + 1e-30
                alpha = min(1.0, 0.5 * mg0 / max_disp)

            gTd = g @ dz
            accepted = False
            for _ls in range(40):
                z_try = z + alpha * dz
                if self.mesh is not None and self._min_gap(z_try) <= 0.0:
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

        x_new = z.reshape(n, 3)
        x_new[self.fixed] = self.x_bc[self.fixed]
        self.rod.x.assign(x_new.astype(np.float32))
        return {"energy": e, "grad_norm": float(np.linalg.norm(g[free])),
                "nit": int(outer + 1), "ccd_alpha": last_alpha,
                "min_gap": self._min_gap(z) if self.mesh is not None else None}

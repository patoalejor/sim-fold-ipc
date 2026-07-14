"""Rail actuation + IPC navigation driver.

The proximal part of the rod is snapped to a straight **rail** aligned with the
vessel inlet and fed in at a **constant insertion speed** (Dirichlet BC).  A node
stays pinned to the rail (sliding forward, no lateral deviation) until it has
been fed a short distance past the inlet — i.e. safely inside the straight entry
section — at which point it is released and thereafter kept inside the vessel
purely by the IPC log-barrier + conservative CCD.

Pinning the not-yet-entered portion to the rail is what prevents the classic
"you can't push a string" buckling: every node is either rail-constrained
(straight) or wall-confined (inside the vessel), so there is no unsupported span.

Twist actuation (constant rotation rate) is a no-op for the positions-primary
rod (round cross-section, no material frame) and is deferred to the full
Cosserat model; the hook is left here for that upgrade.
"""
from __future__ import annotations

import numpy as np

from .rod import init_rod_on_centerline
from .solver_pos import PosSolver


def make_straight_rod(tip, direction, length, n, params, device):
    """Straight rod of ``length`` lying on the rail, tip at ``tip`` and body
    extending back along ``-direction``.  Nodes are ordered proximal -> distal."""
    direction = np.asarray(direction, float)
    direction /= np.linalg.norm(direction)
    s = np.linspace(0.0, length, n)
    cl = np.asarray(tip, float)[None, :] - s[:, None] * direction[None, :]
    cl = cl[::-1].copy()                      # proximal (tail) first, tip last
    return init_rod_on_centerline(cl, params, device), direction


def _arclen(centerline):
    seg = np.linalg.norm(np.diff(centerline, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def _interp_centerline(centerline, s_arc, s):
    s = np.clip(s, 0.0, s_arc[-1])
    return np.stack([np.interp(s, s_arc, centerline[:, d]) for d in range(3)])


def run_navigation_guided(mesh, rod, centerline, *, dt=0.3, d_hat=4e-4,
                          kappa=1e2, tip_speed=6e-4, n_steps=90,
                          record_every=2, max_iter=20, verbose=True):
    """Tip-guided navigation: the distal tip tracks the vessel centreline at
    constant ``tip_speed`` (a steerable / pre-shaped tip); the body follows and
    is kept inside the vessel purely by the IPC barrier + CCD.  Returns
    (frames, min_gaps).  Guarantees forward progress (no push-buckling)."""
    solver = PosSolver(rod, dt=dt)
    solver.set_vessel(mesh, d_hat=d_hat, kappa=kappa)

    s_arc = _arclen(centerline)
    tip = rod.n - 1
    solver.fixed[:] = False
    solver.fixed[tip] = True

    frames = [rod.positions().copy()]
    min_gaps = []
    for step in range(n_steps):
        s = (step + 1) * tip_speed
        solver.x_bc[tip] = _interp_centerline(centerline, s_arc, s)
        info = solver.step(max_iter=max_iter)
        min_gaps.append(info["min_gap"])
        if step % record_every == 0:
            frames.append(rod.positions().copy())
        if verbose and (step % 10 == 0 or step == n_steps - 1):
            print(f"step {step:3d}  tip_s={s*1e3:5.1f}mm  E={info['energy']:.2e}  "
                  f"min_gap={info['min_gap']*1e3:+.3f}mm  nit={info['nit']}")
    frames.append(rod.positions().copy())
    return frames, np.array(min_gaps)


def run_navigation(mesh, rod, rail_origin, rail_dir, *, dt=0.3, d_hat=4e-4,
                   kappa=1e2, insert_speed=5e-4, n_steps=90, t_release=6e-3,
                   record_every=2, max_iter=20, verbose=True):
    """Feed the rod along the rail into ``mesh``; return (frames, min_gaps).

    ``t_release`` is how far past the inlet (metres, along the rail) a node is
    fed before being released into free contact — keep it inside the straight
    entry section so a node is never pinned onto a curved region.
    """
    solver = PosSolver(rod, dt=dt)
    solver.set_vessel(mesh, d_hat=d_hat, kappa=kappa)

    rail_origin = np.asarray(rail_origin, float)
    rail_dir = np.asarray(rail_dir, float)
    rail_dir /= np.linalg.norm(rail_dir)
    x0 = rod.positions()
    t0 = (x0 - rail_origin[None, :]) @ rail_dir       # per-node rail coordinate

    frames = [x0.copy()]
    min_gaps = []
    for step in range(n_steps):
        d = (step + 1) * insert_speed
        t = t0 + d
        pinned = np.where(t < t_release)[0]
        solver.fixed[:] = False
        solver.fixed[pinned] = True
        solver.x_bc[pinned] = rail_origin[None, :] + t[pinned, None] * rail_dir[None, :]

        info = solver.step(max_iter=max_iter)
        min_gaps.append(info["min_gap"])
        if step % record_every == 0:
            frames.append(rod.positions().copy())
        if verbose and (step % 10 == 0 or step == n_steps - 1):
            print(f"step {step:3d}  fed={d*1e3:5.1f}mm  n_free={rod.n-pinned.size:2d}  "
                  f"E={info['energy']:.2e}  min_gap={info['min_gap']*1e3:+.3f}mm  "
                  f"nit={info['nit']}  ccd={info['ccd_alpha']:.2f}")
    frames.append(rod.positions().copy())
    return frames, np.array(min_gaps)

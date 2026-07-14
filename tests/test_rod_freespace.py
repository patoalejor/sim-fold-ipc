"""Week-1 validation: free-space rod kinematics (no gravity, no contact).

Endovascular navigation is gravity-free and quasi-static; the rod is validated
by a *release-and-relax* test using the positions-primary (DER) model:

  * Pre-bend the rod into a 90-degree arc (stores bending energy).
  * Clamp the base (position + initial direction) and release, overdamped.
  * The rod must relax monotonically to its straight rest shape: total turning
    angle -> 0, elastic energy decays by orders of magnitude, and the rod stays
    inextensible (no stretch blow-up).
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim_fold_ipc.device import get_device                       # noqa: E402
from sim_fold_ipc.rod import RodParams, init_rod_on_centerline   # noqa: E402
from sim_fold_ipc.solver_pos import PosSolver                    # noqa: E402


def turning(rod):
    p = rod.positions()
    t = np.diff(p, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-30
    d = np.clip(np.einsum("ij,ij->i", t[:-1], t[1:]), -1, 1)
    ang = np.degrees(np.arccos(d))
    return ang.sum(), ang.max()


def main():
    device = get_device()
    print(f"device = {device}")

    N = 25
    R = 0.03
    theta = np.linspace(0.0, np.deg2rad(90.0), N)
    cl = np.stack([R * np.sin(theta), np.zeros(N), R * (1 - np.cos(theta))], axis=1)

    params = RodParams(radius=4.5e-4, youngs=60e9)   # SolitaireX nitinol
    rod = init_rod_on_centerline(cl, params, device)
    solver = PosSolver(rod, dt=0.3)
    solver.set_fixed([0, 1])                         # clamp base + direction

    tot0, mx0 = turning(rod)
    e0 = None
    energies, turns = [], [tot0]
    print(f"initial: total_turn = {tot0:.2f} deg, max/seg = {mx0:.2f} deg")

    for step in range(60):
        info = solver.step(max_iter=12)
        if e0 is None:
            e0 = info["energy"]
        energies.append(info["energy"])
        tot, mx = turning(rod)
        turns.append(tot)
        if step % 10 == 0 or step == 59:
            print(f"step {step:3d}  E = {info['energy']:.3e}  "
                  f"total_turn = {tot:7.3f} deg")

    energies = np.array(energies)
    tot_f, _ = turning(rod)
    seg = np.linalg.norm(np.diff(rod.positions(), axis=0), axis=1)
    rest = rod.l0.numpy()
    max_stretch = float(np.abs(seg / rest - 1).max())
    print(f"\nfinal: E = {energies[-1]:.3e}  total_turn = {tot_f:.3f} deg  "
          f"max_stretch = {100 * max_stretch:.3f} %")

    # 1) relaxed to straight
    assert tot_f < 2.0, f"rod did not straighten: {tot_f:.2f} deg total turn"
    # 2) large energy decay
    assert energies[-1] < 1e-3 * e0, f"energy did not decay: {energies[-1]:.2e}"
    # 3) monotone (overdamped) -- allow tiny numerical noise
    assert np.all(np.diff(energies) <= 1e-9 + 1e-6 * np.abs(energies[:-1])), \
        "energy increased -> not monotone/overdamped"
    # 4) turning angle decreased overall
    assert turns[-1] < 0.1 * turns[0], "turning did not decrease enough"
    # 5) inextensible
    assert max_stretch < 0.02, f"excessive stretch: {100*max_stretch:.2f} %"
    # 6) base clamped
    assert np.linalg.norm(rod.positions()[0] - cl[0]) < 1e-9, "base moved"

    print("\nWeek-1 free-space kinematics validation PASSED.")


if __name__ == "__main__":
    main()

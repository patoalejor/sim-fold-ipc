"""Render the Week-1 release-and-relax validation: a pre-bent rod relaxing to
its straight rest shape (no gravity, no contact)."""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim_fold_ipc.device import get_device                       # noqa: E402
from sim_fold_ipc.rod import RodParams, init_rod_on_centerline   # noqa: E402
from sim_fold_ipc.solver_pos import PosSolver                    # noqa: E402
from sim_fold_ipc import render                                  # noqa: E402

_BG, _FG = "#0e1117", "#e6e6e6"


def main():
    os.makedirs("images", exist_ok=True)
    dev = get_device()

    N = 25
    R = 0.03
    theta = np.linspace(0.0, np.deg2rad(90.0), N)
    cl = np.stack([R * np.sin(theta), np.zeros(N), R * (1 - np.cos(theta))], axis=1)
    rod = init_rod_on_centerline(cl, RodParams(radius=4.5e-4, youngs=60e9), dev)
    solver = PosSolver(rod, dt=0.3)
    solver.set_fixed([0, 1])

    frames = [rod.positions().copy()]
    for _ in range(40):
        solver.step(max_iter=12)
        frames.append(rod.positions().copy())

    # static overlay: initial (faint) vs final (bright)
    fig = plt.figure(figsize=(6.5, 6), dpi=140, facecolor=_BG)
    ax = fig.add_subplot(111, projection="3d")
    init, final = frames[0], frames[-1]
    ax.add_collection3d(Line3DCollection(
        np.stack([init[:-1], init[1:]], axis=1), colors="#ff6b4a",
        linewidths=2.0, alpha=0.35))
    ax.add_collection3d(Line3DCollection(
        np.stack([final[:-1], final[1:]], axis=1), colors="#4c9be8",
        linewidths=3.5))
    allp = np.concatenate([init, final])
    c = 0.5 * (allp.min(0) + allp.max(0)); r = 0.5 * (allp.max(0) - allp.min(0)).max() + 1e-3
    ax.set_xlim(c[0]-r, c[0]+r); ax.set_ylim(c[1]-r, c[1]+r); ax.set_zlim(c[2]-r, c[2]+r)
    ax.set_box_aspect((1, 1, 1))
    ax.set_facecolor(_BG); ax.grid(False)
    ax.set_title("Week 1: pre-bent rod (orange) relaxes to straight (blue)",
                 color=_FG, fontsize=11)
    ax.tick_params(colors=_FG, labelsize=7)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.set_pane_color((1, 1, 1, 0.02))
    ax.view_init(elev=20, azim=-60)
    fig.tight_layout(); fig.savefig("images/week1_relax.png", facecolor=_BG)
    plt.close(fig)

    render.render_sequence("images/week1_relax.mp4", "images/week1_relax.gif",
                           "Week 1: release-and-relax", None, frames,
                           elev=20, azim=-60, fps=15)
    print("wrote images/week1_relax.png/.mp4/.gif")


if __name__ == "__main__":
    main()

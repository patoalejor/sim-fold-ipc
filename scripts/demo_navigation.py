"""Week-2 demo: feed the guidewire into each vessel and render the result.

Runs the rail-actuated IPC navigation in the straight and S-shaped vessels,
renders a static PNG + orbit-free sequence movie (MP4 + GIF) of the rod moving
through each vessel, and plots the minimum wall clearance over insertion
(which must stay positive => zero penetration).
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim_fold_ipc.device import get_device                     # noqa: E402
from sim_fold_ipc.rod import RodParams                          # noqa: E402
from sim_fold_ipc.meshes import straight_tube, s_tube           # noqa: E402
from sim_fold_ipc.barrier import build_vessel_mesh              # noqa: E402
from sim_fold_ipc.navigate import (make_straight_rod, run_navigation,   # noqa: E402
                                   run_navigation_guided)
from sim_fold_ipc import render                                 # noqa: E402

VESSEL_R = 1.75e-3
# SolitaireX-class nitinol (E~60 GPa); a slender core so the guidewire is
# flexible enough to conform to the neuro-vessel bends.
PARAMS = RodParams(radius=2.0e-4, youngs=10e9)


def gap_plot(path, gaps, speed, title):
    fed = np.arange(1, len(gaps) + 1) * speed * 1e3
    fig, ax = plt.subplots(figsize=(6, 3.2), dpi=140, facecolor="#0e1117")
    ax.set_facecolor("#0e1117")
    ax.axhline(0.0, color="#ff5555", lw=1.2, ls="--", label="wall (penetration)")
    ax.plot(fed, gaps * 1e3, color="#4c9be8", lw=2.0, label="min wall clearance")
    ax.fill_between(fed, gaps * 1e3, 0, color="#4c9be8", alpha=0.15)
    ax.set_xlabel("insertion depth (mm)"); ax.set_ylabel("min gap (mm)")
    ax.set_title(title, color="#e6e6e6")
    for s in ax.spines.values():
        s.set_color("#444")
    ax.tick_params(colors="#e6e6e6"); ax.xaxis.label.set_color("#e6e6e6")
    ax.yaxis.label.set_color("#e6e6e6")
    leg = ax.legend(facecolor="#1a1f2b", edgecolor="#444", labelcolor="#e6e6e6",
                    fontsize=8)
    fig.tight_layout(); fig.savefig(path, facecolor="#0e1117"); plt.close(fig)


def _render_case(name, tm, frames, gaps, speed, title, elev, azim):
    status = "PENETRATION-FREE" if gaps.min() > 0 else "PENETRATED"
    print(f"[{name}] min_gap = {gaps.min()*1e3:+.4f} mm -> {status}")
    render.render_png(f"images/nav_{name}.png", title, mesh=tm,
                      rod_pts=frames[-1], elev=elev, azim=azim)
    render.render_sequence(f"images/nav_{name}.mp4", f"images/nav_{name}.gif",
                           title, tm, frames, elev=elev, azim=azim, fps=15)
    gap_plot(f"images/nav_{name}_gap.png", gaps, speed,
             f"{title} - wall clearance vs advance")


def case_straight():
    dev = get_device()
    tm = straight_tube(length=0.045, radius=VESSEL_R, n_axial=40, n_theta=20)
    rod, _ = make_straight_rod(tip=[0, 0, 0.008], direction=[0, 0, 1],
                               length=0.045, n=45, params=PARAMS, device=dev)
    mesh = build_vessel_mesh(tm.vertices, tm.faces, dev)
    # push-feed on the rail: the real "insert" actuation
    frames, gaps = run_navigation(mesh, rod, [0, 0, 0], [0, 0, 1],
                                  insert_speed=6e-4, n_steps=50, t_release=6e-3,
                                  record_every=2, max_iter=15)
    _render_case("straight", tm, frames, gaps, 6e-4,
                 "Guidewire - straight vessel (push-fed)", 18, -72)


def case_s():
    dev = get_device()
    tm = s_tube(radius=VESSEL_R, bend_radius=0.012, bend_angle_deg=60.0,
                straight_len=0.015, n_theta=20, seg_res=30)
    cl = tm.centerline.astype(float)
    rod, _ = make_straight_rod(tip=cl[0], direction=[0, 0, 1], length=0.050,
                               n=50, params=PARAMS, device=dev)
    mesh = build_vessel_mesh(tm.vertices, tm.faces, dev)
    # tip-guided navigation through the S (steerable / pre-shaped tip)
    frames, gaps = run_navigation_guided(mesh, rod, cl, tip_speed=6e-4,
                                         n_steps=100, d_hat=5e-4, kappa=1e2,
                                         record_every=2, max_iter=20)
    _render_case("s", tm, frames, gaps, 6e-4,
                 "Guidewire - S-vessel (60 deg, tip-guided)", 16, -88)


def main():
    os.makedirs("images", exist_ok=True)
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("both", "straight"):
        case_straight()
    if which in ("both", "s"):
        case_s()
    print("navigation demo complete; renders in images/")


if __name__ == "__main__":
    main()

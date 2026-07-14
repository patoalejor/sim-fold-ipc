"""Week-2 validation: IPC navigation is penetration-free in both vessels.

Runs a shortened rail-fed insertion (straight vessel) and a tip-guided
traversal (S vessel) and asserts the minimum wall clearance stays strictly
positive throughout -- i.e. the log-barrier + conservative CCD never let the
guidewire cross the vessel wall.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim_fold_ipc.device import get_device                       # noqa: E402
from sim_fold_ipc.rod import RodParams                           # noqa: E402
from sim_fold_ipc.meshes import straight_tube, s_tube            # noqa: E402
from sim_fold_ipc.barrier import build_vessel_mesh               # noqa: E402
from sim_fold_ipc.navigate import (make_straight_rod,            # noqa: E402
                                   run_navigation,
                                   run_navigation_guided)

PARAMS = RodParams(radius=2.0e-4, youngs=10e9)
VESSEL_R = 1.75e-3


def test_straight():
    dev = get_device()
    tm = straight_tube(length=0.045, radius=VESSEL_R, n_axial=30, n_theta=18)
    rod, _ = make_straight_rod(tip=[0, 0, 0.008], direction=[0, 0, 1],
                               length=0.035, n=30, params=PARAMS, device=dev)
    mesh = build_vessel_mesh(tm.vertices, tm.faces, dev)
    _, gaps = run_navigation(mesh, rod, [0, 0, 0], [0, 0, 1], insert_speed=8e-4,
                             n_steps=30, t_release=6e-3, max_iter=12,
                             verbose=False)
    print(f"[straight] min_gap = {gaps.min()*1e3:+.4f} mm")
    assert gaps.min() > 0.0, "penetration in straight vessel"


def test_s():
    dev = get_device()
    tm = s_tube(radius=VESSEL_R, bend_radius=0.012, bend_angle_deg=60.0,
                straight_len=0.015, n_theta=18, seg_res=24)
    cl = tm.centerline.astype(float)
    rod, _ = make_straight_rod(tip=cl[0], direction=[0, 0, 1], length=0.040,
                               n=34, params=PARAMS, device=dev)
    mesh = build_vessel_mesh(tm.vertices, tm.faces, dev)
    _, gaps = run_navigation_guided(mesh, rod, cl, tip_speed=1.0e-3, n_steps=55,
                                    d_hat=5e-4, kappa=1e2, max_iter=18,
                                    verbose=False)
    print(f"[s-vessel] min_gap = {gaps.min()*1e3:+.4f} mm")
    assert gaps.min() > 0.0, "penetration in S vessel"


def main():
    print(f"device = {get_device()}")
    test_straight()
    test_s()
    print("\nWeek-2 IPC navigation validation PASSED (both vessels penetration-free).")


if __name__ == "__main__":
    main()

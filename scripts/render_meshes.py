"""Render the two vessel meshes (neural-vessel scale) to images/.

Neural vessels are ~2-5 mm in diameter; we use a 3.5 mm inner diameter
(1.75 mm radius) tube for both the straight and S-shaped cases.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim_fold_ipc.meshes import straight_tube, s_tube      # noqa: E402
from sim_fold_ipc import render                            # noqa: E402

VESSEL_R = 1.75e-3   # 3.5 mm inner diameter (neural-vessel range 2-5 mm)


def main():
    os.makedirs("images", exist_ok=True)

    st = straight_tube(length=0.045, radius=VESSEL_R, n_axial=40, n_theta=20)
    ss = s_tube(radius=VESSEL_R, bend_radius=0.012, bend_angle_deg=60.0,
                straight_len=0.015, n_theta=20, seg_res=30)
    print(f"straight: {st.n_vertices} verts / {st.n_faces} faces")
    print(f"s-tube  : {ss.n_vertices} verts / {ss.n_faces} faces")

    render.render_png("images/straight_tube.png",
                      "Straight vessel  (3.5 mm ID)", mesh=st)
    render.render_png("images/s_tube.png",
                      "S-vessel  (60 deg bends, R=12 mm)", mesh=ss)
    print("wrote static PNGs")

    render.render_orbit("images/straight_tube_orbit.mp4",
                        "images/straight_tube_orbit.gif",
                        "Straight vessel", mesh=st, nframes=40)
    render.render_orbit("images/s_tube_orbit.mp4",
                        "images/s_tube_orbit.gif",
                        "S-vessel  (60 deg bends)", mesh=ss, nframes=40)
    print("wrote orbit MP4 + GIF")


if __name__ == "__main__":
    main()

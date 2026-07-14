"""Validation for procedural vessel meshes. Runnable with plain `python`."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim_fold_ipc.meshes import straight_tube, s_tube  # noqa: E402


def _face_normals(mesh):
    v = mesh.vertices
    tris = v[mesh.faces]
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-30
    centroids = tris.mean(axis=1)
    return n, centroids


def _nearest_centerline_pt(mesh, pts):
    cl = mesh.centerline
    d = np.linalg.norm(pts[:, None, :] - cl[None, :, :], axis=2)
    idx = d.argmin(axis=1)
    return cl[idx]


def check_radius(mesh, radius, tol=0.15):
    cl_pts = _nearest_centerline_pt(mesh, mesh.vertices)
    dist = np.linalg.norm(mesh.vertices - cl_pts, axis=1)
    err = np.abs(dist - radius) / radius
    assert err.max() < tol, f"radius error too large: max {err.max():.3f}"
    return dist.mean()


def check_outward_normals(mesh):
    n, centroids = _face_normals(mesh)
    cl_pts = _nearest_centerline_pt(mesh, centroids)
    outward = centroids - cl_pts
    outward /= np.linalg.norm(outward, axis=1, keepdims=True) + 1e-30
    dot = np.einsum("ij,ij->i", n, outward)
    frac_out = (dot > 0).mean()
    assert frac_out > 0.95, f"only {frac_out:.2%} faces point outward"
    return frac_out


def main():
    os.makedirs("assets", exist_ok=True)

    st = straight_tube(length=0.20, radius=0.004, n_axial=60, n_theta=24)
    assert st.n_vertices == 61 * 24, st.n_vertices
    assert st.n_faces == 60 * 24 * 2, st.n_faces
    r_mean = check_radius(st, 0.004)
    frac = check_outward_normals(st)
    st.save_obj("assets/straight_tube.obj")
    print(f"[straight] verts={st.n_vertices} faces={st.n_faces} "
          f"r_mean={r_mean*1e3:.3f}mm outward={frac:.1%}")

    ss = s_tube(radius=0.004, bend_radius=0.03, bend_angle_deg=60.0,
                straight_len=0.04, n_theta=24, seg_res=40)
    r_mean = check_radius(ss, 0.004)
    frac = check_outward_normals(ss)
    ss.save_obj("assets/s_tube.obj")
    # geometry sanity: net turn is +60 then -60 -> final tangent ~ initial (+z)
    cl = ss.centerline
    t_start = cl[1] - cl[0]
    t_end = cl[-1] - cl[-2]
    t_start /= np.linalg.norm(t_start)
    t_end /= np.linalg.norm(t_end)
    align = float(t_start @ t_end)
    print(f"[s-tube ] verts={ss.n_vertices} faces={ss.n_faces} "
          f"r_mean={r_mean*1e3:.3f}mm outward={frac:.1%} "
          f"start·end tangent={align:.3f}")
    assert align > 0.99, f"S-tube endpoints not parallel: {align:.3f}"

    print("\nAll mesh checks passed. OBJs written to assets/.")


if __name__ == "__main__":
    main()

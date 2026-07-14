"""IPC log-barrier contact against a static vessel mesh (Warp).

Broad-phase + narrow-phase distance queries use Warp's built-in ``wp.Mesh``
BVH (``wp.mesh_query_point``). For each rod node we take the unsigned distance
to the closest point on the vessel wall, subtract the rod radius to get the
gap ``g``, and apply the C2 log-barrier

    B(g) = -(g - d_hat)^2 * ln(g / d_hat)   for 0 < g < d_hat,   else 0.

The closest face/barycentric coords are treated as constants inside the kernel
(the correct IPC subgradient), so autodiff yields dB/dx = barrier force.
"""
from __future__ import annotations

import numpy as np
import warp as wp


def build_vessel_mesh(vertices: np.ndarray, faces: np.ndarray, device: str) -> wp.Mesh:
    v = wp.array(vertices.astype(np.float32), dtype=wp.vec3, device=device)
    f = wp.array(faces.reshape(-1).astype(np.int32), dtype=wp.int32, device=device)
    return wp.Mesh(points=v, indices=f)


@wp.kernel
def barrier_energy(
    x: wp.array(dtype=wp.vec3),
    mesh_id: wp.uint64,
    rod_radius: wp.float32,
    d_hat: wp.float32,
    kappa: wp.float32,
    weight: wp.array(dtype=wp.float32),
    E: wp.array(dtype=wp.float64),
):
    i = wp.tid()
    p = x[i]
    max_d = d_hat + rod_radius
    query = wp.mesh_query_point(mesh_id, p, max_d)
    if not query.result:
        return
    cp = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
    d = wp.length(p - cp)
    g = d - rod_radius
    if g > 0.0 and g < d_hat:
        r = g - d_hat
        b = -(r * r) * wp.log(g / d_hat)
        wp.atomic_add(E, 0, wp.float64(kappa * weight[i] * b))


@wp.kernel
def node_gaps(
    x: wp.array(dtype=wp.vec3),
    mesh_id: wp.uint64,
    rod_radius: wp.float32,
    max_query: wp.float32,
    gap_out: wp.array(dtype=wp.float32),
):
    """Signed gap (distance-to-wall minus rod radius) per node.

    Large positive when far from the wall; <= 0 means penetration.
    """
    i = wp.tid()
    p = x[i]
    query = wp.mesh_query_point(mesh_id, p, max_query)
    if not query.result:
        gap_out[i] = max_query
        return
    cp = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
    gap_out[i] = wp.length(p - cp) - rod_radius

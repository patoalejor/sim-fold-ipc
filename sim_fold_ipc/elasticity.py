"""Differentiable Cosserat energy kernels (Warp).

The optimiser variables are node positions ``x`` and per-node rotation-vector
increments ``phi``; the current orientation is reconstructed inside the kernels
as ``q_i = normalize(exp(phi_i) * q_prev_i)`` so that Warp's autodiff can
differentiate the whole incremental potential w.r.t. both ``x`` and ``phi``.

All energy kernels accumulate into a single scalar ``E[0]`` via atomic add.
"""
from __future__ import annotations

import warp as wp

VEC_Z = wp.constant(wp.vec3(0.0, 0.0, 1.0))


@wp.func
def quat_exp(phi: wp.vec3) -> wp.quat:
    """Exponential map: rotation vector -> unit quaternion (x,y,z,w).

    The ``+ eps`` under the sqrt keeps the reverse-mode adjoint finite at
    ``phi = 0`` (``d/dx sqrt(x)`` is otherwise infinite there), which would
    poison every gradient since the predicted increment starts at zero.
    """
    theta = wp.sqrt(wp.dot(phi, phi) + 1.0e-12)
    half = 0.5 * theta
    s = wp.sin(half) / theta
    return wp.quat(s * phi[0], s * phi[1], s * phi[2], wp.cos(half))


@wp.func
def current_q(phi: wp.vec3, q_prev: wp.quat) -> wp.quat:
    return wp.normalize(quat_exp(phi) * q_prev)


@wp.kernel
def inertia_energy(
    x: wp.array(dtype=wp.vec3),
    phi: wp.array(dtype=wp.vec3),
    x_pred: wp.array(dtype=wp.vec3),
    phi_pred: wp.array(dtype=wp.vec3),
    mass: wp.array(dtype=wp.float32),
    rot_inertia: wp.array(dtype=wp.float32),
    inv_dt2: wp.float32,
    E: wp.array(dtype=wp.float64),
):
    i = wp.tid()
    dxp = x[i] - x_pred[i]
    dphi = phi[i] - phi_pred[i]
    e = 0.5 * inv_dt2 * (mass[i] * wp.dot(dxp, dxp)
                         + rot_inertia[i] * wp.dot(dphi, dphi))
    wp.atomic_add(E, 0, wp.float64(e))


@wp.kernel
def stretch_shear_energy(
    x: wp.array(dtype=wp.vec3),
    phi: wp.array(dtype=wp.vec3),
    q_prev: wp.array(dtype=wp.quat),
    l0: wp.array(dtype=wp.float32),
    k_stretch: wp.float32,
    k_shear: wp.float32,
    E: wp.array(dtype=wp.float64),
):
    i = wp.tid()  # segment i in [0, N-2]
    qi = current_q(phi[i], q_prev[i])
    dx = x[i + 1] - x[i]
    # strain in the material frame: R^T (dx/l0) - e_z
    gamma = wp.quat_rotate_inv(qi, dx / l0[i]) - VEC_Z
    e = 0.5 * l0[i] * (k_shear * (gamma[0] * gamma[0] + gamma[1] * gamma[1])
                       + k_stretch * gamma[2] * gamma[2])
    wp.atomic_add(E, 0, wp.float64(e))


@wp.kernel
def bend_twist_energy(
    phi: wp.array(dtype=wp.vec3),
    q_prev: wp.array(dtype=wp.quat),
    l0: wp.array(dtype=wp.float32),
    k_bend: wp.float32,
    k_twist: wp.float32,
    E: wp.array(dtype=wp.float64),
):
    i = wp.tid()  # junction between segment i and i+1
    qi = current_q(phi[i], q_prev[i])
    qj = current_q(phi[i + 1], q_prev[i + 1])
    qrel = wp.quat_inverse(qi) * qj
    vx = qrel[0]
    vy = qrel[1]
    vz = qrel[2]
    if qrel[3] < 0.0:  # shortest arc (quaternion double cover)
        vx = -vx
        vy = -vy
        vz = -vz
    inv_l = 1.0 / l0[i]
    kx = 2.0 * vx * inv_l
    ky = 2.0 * vy * inv_l
    kz = 2.0 * vz * inv_l
    e = 0.5 * l0[i] * (k_bend * (kx * kx + ky * ky) + k_twist * kz * kz)
    wp.atomic_add(E, 0, wp.float64(e))

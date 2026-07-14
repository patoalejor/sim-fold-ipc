"""Positions-primary elastic rod energies (Discrete Elastic Rods style).

Unlike the full quaternion-Cosserat model (see :mod:`elasticity`), the bending
energy here is a function of the *centreline positions* directly, via the
discrete curvature binormal.  This removes the stiff frame<->position coupling
that makes the Cosserat Hessian ~1e8-conditioned, so the incremental potential
solves robustly.  A moderate stretch stiffness keeps the rod effectively
inextensible while staying well-conditioned.

DOFs are node positions only (3 per node).  Energies accumulate into a scalar
``E[0]`` (float64) for Warp autodiff.
"""
from __future__ import annotations

import warp as wp


@wp.kernel
def inertia_energy_pos(
    x: wp.array(dtype=wp.vec3),
    x_pred: wp.array(dtype=wp.vec3),
    mass: wp.array(dtype=wp.float32),
    inv_dt2: wp.float32,
    E: wp.array(dtype=wp.float64),
):
    i = wp.tid()
    d = x[i] - x_pred[i]
    wp.atomic_add(E, 0, wp.float64(0.5 * inv_dt2 * mass[i] * wp.dot(d, d)))


@wp.kernel
def stretch_energy_pos(
    x: wp.array(dtype=wp.vec3),
    l0: wp.array(dtype=wp.float32),
    k_stretch: wp.float32,
    E: wp.array(dtype=wp.float64),
):
    i = wp.tid()  # segment i in [0, N-2]
    e = x[i + 1] - x[i]
    length = wp.length(e)
    d = length - l0[i]
    wp.atomic_add(E, 0, wp.float64(0.5 * k_stretch * d * d / l0[i]))


@wp.kernel
def bend_energy_pos(
    x: wp.array(dtype=wp.vec3),
    l_bar: wp.array(dtype=wp.float32),     # rest voronoi length at interior node
    kappa_rest: wp.array(dtype=wp.vec3),   # rest curvature binormal (0 for straight)
    k_bend: wp.float32,
    E: wp.array(dtype=wp.float64),
):
    j = wp.tid()               # interior node index in [0, N-3]
    i = j + 1                  # actual node (1 .. N-2)
    e_prev = x[i] - x[i - 1]
    e_next = x[i + 1] - x[i]
    denom = wp.length(e_prev) * wp.length(e_next) + wp.dot(e_prev, e_next)
    kb = (2.0 / (denom + 1.0e-12)) * wp.cross(e_prev, e_next)
    dk = kb - kappa_rest[j]
    wp.atomic_add(E, 0, wp.float64(k_bend * wp.dot(dk, dk) / l_bar[j]))

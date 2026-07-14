"""Cosserat rod state and material parameters.

Discretisation (CoRdE / Kugelstadt-style):

* ``N`` nodes with positions ``x[i]`` and orientation quaternions ``q[i]``.
* ``N-1`` segments; segment ``i`` connects node ``i`` and ``i+1`` and uses the
  material frame of node ``i``.
* The tangent director is the quaternion-rotated local +Z axis.

All quantities are SI (metres, kilograms, seconds).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import warp as wp


@dataclass
class RodParams:
    radius: float = 1.8e-4          # guidewire cross-section radius (m)
    density: float = 6500.0         # kg/m^3 (nitinol-ish)
    youngs: float = 5.0e6           # Pa (softened for a stable CPU demo)
    poisson: float = 0.3
    # relative compliance of the (near-rigid) stretch/shear modes.  A real
    # guidewire is ~inextensible, so these modes are ~A/I ~ 1e7-1e8 stiffer
    # than bending, which wrecks the Hessian conditioning.  Softening them
    # keeps the model navigable while remaining effectively inextensible at
    # the loads seen in contact.  1.0 = physically exact.
    stretch_scale: float = 1.0
    shear_scale: float = 1.0

    @property
    def area(self) -> float:
        return np.pi * self.radius ** 2

    @property
    def inertia(self) -> float:              # second moment of area
        return 0.25 * np.pi * self.radius ** 4

    @property
    def shear_modulus(self) -> float:
        return self.youngs / (2.0 * (1.0 + self.poisson))

    # stiffnesses ---------------------------------------------------------
    @property
    def k_stretch(self) -> float:            # axial  (E*A)
        return self.youngs * self.area * self.stretch_scale

    @property
    def k_shear(self) -> float:              # shear  (G*A)
        return self.shear_modulus * self.area * self.shear_scale

    @property
    def k_bend(self) -> float:               # bending (E*I)
        return self.youngs * self.inertia

    @property
    def k_twist(self) -> float:              # torsion (G*J), J = 2I
        return self.shear_modulus * 2.0 * self.inertia


@dataclass
class RodState:
    """Warp-array-backed state of the rod (host mirrors kept in numpy)."""

    n: int
    device: str
    params: RodParams

    x: wp.array = field(default=None)        # vec3  positions
    q: wp.array = field(default=None)        # quat  orientations
    v: wp.array = field(default=None)        # vec3  linear velocity
    omega: wp.array = field(default=None)    # vec3  angular velocity (world)

    l0: wp.array = field(default=None)       # per-segment rest length
    mass: wp.array = field(default=None)     # per-node lumped mass
    rot_inertia: wp.array = field(default=None)  # per-node rotational inertia

    def positions(self) -> np.ndarray:
        return self.x.numpy()

    def quaternions(self) -> np.ndarray:
        return self.q.numpy()


def init_rod_on_centerline(
    centerline: np.ndarray,
    params: RodParams,
    device: str,
    n: int | None = None,
) -> RodState:
    """Create a straight/curved rod whose nodes lie on ``centerline``.

    The rod is resampled to ``n`` nodes (default: len(centerline)). Orientation
    quaternions align the local +Z director with the local tangent.
    """
    cl = np.asarray(centerline, dtype=np.float64)
    if n is not None and n != len(cl):
        # arc-length resample
        seg = np.linalg.norm(np.diff(cl, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        snew = np.linspace(0.0, s[-1], n)
        cl = np.stack([np.interp(snew, s, cl[:, d]) for d in range(3)], axis=1)
    n = len(cl)

    # tangents
    t = np.zeros_like(cl)
    t[:-1] = cl[1:] - cl[:-1]
    t[-1] = t[-2]
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-30

    # quaternion rotating +Z onto each tangent
    quats = np.zeros((n, 4), dtype=np.float32)  # (x,y,z,w)
    zc = np.array([0.0, 0.0, 1.0])
    for i in range(n):
        quats[i] = _quat_from_z_to(zc, t[i])

    seglen = np.linalg.norm(np.diff(cl, axis=0), axis=1).astype(np.float32)
    # lumped node mass from adjacent half-segments
    node_len = np.zeros(n, dtype=np.float64)
    node_len[:-1] += 0.5 * seglen
    node_len[1:] += 0.5 * seglen
    node_len[0] += 0.5 * seglen[0]
    node_len[-1] += 0.5 * seglen[-1]
    mass = (params.density * params.area * node_len).astype(np.float32)
    # rotational inertia proxy: rho * I_polar * length
    rot_in = (params.density * 2.0 * params.inertia * node_len).astype(np.float32)
    rot_in = np.maximum(rot_in, 1e-12).astype(np.float32)

    return RodState(
        n=n,
        device=device,
        params=params,
        x=wp.array(cl.astype(np.float32), dtype=wp.vec3, device=device),
        q=wp.array(quats, dtype=wp.quat, device=device),
        v=wp.zeros(n, dtype=wp.vec3, device=device),
        omega=wp.zeros(n, dtype=wp.vec3, device=device),
        l0=wp.array(seglen, dtype=wp.float32, device=device),
        mass=wp.array(mass, dtype=wp.float32, device=device),
        rot_inertia=wp.array(rot_in, dtype=wp.float32, device=device),
    )


def _quat_from_z_to(z: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Shortest-arc quaternion rotating unit ``z`` onto unit ``target``."""
    z = z / np.linalg.norm(z)
    target = target / np.linalg.norm(target)
    d = float(np.dot(z, target))
    if d > 1.0 - 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    if d < -1.0 + 1e-8:
        # 180 deg: rotate about any axis perpendicular to z
        axis = np.cross(z, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(z, np.array([0.0, 1.0, 0.0]))
        axis /= np.linalg.norm(axis)
        return np.array([axis[0], axis[1], axis[2], 0.0], dtype=np.float32)
    axis = np.cross(z, target)
    w = 1.0 + d
    q = np.array([axis[0], axis[1], axis[2], w])
    q /= np.linalg.norm(q)
    return q.astype(np.float32)

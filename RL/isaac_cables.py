from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from spiral import Array, Config


def cable_kinematics(
    poses: Array,
    velocities: Array,
    centers: Array,
    bodies: NDArray[np.intp],
    sites: Array,
) -> tuple[Array, Array]:
    """Evaluate routed cable lengths and rates from Newton rigid-body state.

    Args:
        poses: Body positions and XYZW quaternions, shape (bodies, 7).
        velocities: World COM linear then angular velocities, shape (bodies, 6).
        centers: Body-local centers of mass, shape (bodies, 3).
        bodies: Body indices for each cable guide, shape (cables, guides).
        sites: Body-local guide coordinates, shape (cables, guides, 3).

    Returns:
        Cable lengths in metres and length rates in metres per second.
    """
    rotation = Rotation.from_quat(poses[:, 3:]).as_matrix()
    offsets = np.einsum("...ij,...j->...i", rotation[bodies], sites)
    points = poses[bodies, :3] + offsets
    com_offsets = np.einsum("bij,bj->bi", rotation, centers)
    guide_velocity = velocities[bodies, :3] + np.cross(
        velocities[bodies, 3:], offsets - com_offsets[bodies]
    )
    segments = np.diff(points, axis=1)
    lengths = np.linalg.norm(segments, axis=2)
    if np.any(lengths <= 0):
        raise ValueError("Cable route contains coincident consecutive guides")
    rates = np.sum(segments * np.diff(guide_velocity, axis=1), axis=2) / lengths
    return lengths.sum(axis=1), rates.sum(axis=1)


class CableController:
    def __init__(self, cfg: Config, rest_lengths: Array) -> None:
        if cfg.max_retraction_m >= float(np.min(rest_lengths)):
            raise ValueError("Maximum cable pull must be less than the initial length")
        self.cfg = cfg
        self.rest_lengths = rest_lengths.copy()
        self.retraction: Array = np.zeros(3)
        self.velocity: Array = np.zeros(3)
        self.tension: Array = np.zeros(3)
        self.target: Array = np.zeros(3)
        self.requested_velocity: Array = np.zeros(3)
        self.steps = 0

    def set_target(self, pulls: Array) -> None:
        if pulls.shape != (3,) or not np.all(np.isfinite(pulls)):
            raise ValueError("Enter three finite cable pulls")
        if np.any(pulls < -self.cfg.max_payout_m) or np.any(
            pulls > self.cfg.max_retraction_m
        ):
            raise ValueError("Cable pulls are outside the configured stroke limits")
        self.target[:] = pulls

    def step(self, lengths: Array, rates: Array) -> Array:
        cfg = self.cfg
        dt = cfg.physics_dt
        acceleration = cfg.max_acceleration_m_s2
        if self.steps % round(cfg.control_dt / dt) == 0:
            self.requested_velocity[:] = np.clip(
                (self.target - self.retraction) / cfg.action_response_seconds,
                -cfg.max_speed_m_s,
                cfg.max_speed_m_s,
            )
        lower = np.sqrt(
            2 * acceleration * (self.retraction + cfg.max_payout_m)
            + (acceleration * dt) ** 2
        )
        upper = np.sqrt(
            2 * acceleration * (cfg.max_retraction_m - self.retraction)
            + (acceleration * dt) ** 2
        )
        desired = np.clip(
            self.requested_velocity,
            -lower + acceleration * dt,
            upper - acceleration * dt,
        )
        self.velocity += np.clip(
            desired - self.velocity, -acceleration * dt, acceleration * dt
        )
        self.retraction[:] = np.clip(
            self.retraction + self.velocity * dt,
            -cfg.max_payout_m,
            cfg.max_retraction_m,
        )
        extension = lengths - (self.rest_lengths - self.retraction)
        self.tension[:] = np.clip(
            np.where(
                extension > 0,
                cfg.cable_stiffness_n_m * extension
                + cfg.cable_damping_n_s_m * (rates + self.velocity),
                0.0,
            ),
            0,
            cfg.max_tension_n,
        )
        self.steps += 1
        return -self.tension.copy()

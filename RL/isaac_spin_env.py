from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnvIndices,
    VecEnvStepReturn,
)

from isaac_spin_physics import SpinPhysics
from spiral import Array, Config
from spiral_spin import SpinConfig


def spin_reward(
    cfg: Config,
    task: SpinConfig,
    tips: Array,
    previous_tips: Array,
    goals: Array,
    phase: Array,
    direction: Array,
    radius: Array,
    upright_height: float,
    actions: Array,
    previous_actions: Array,
) -> tuple[Array, NDArray[np.bool_], Array, Array]:
    dt = cfg.control_dt
    omega = direction * 2 * np.pi / task.orbit_seconds
    velocity = (tips - previous_tips) / dt
    desired = (radius * omega)[:, None] * np.column_stack(
        (-np.sin(phase), np.cos(phase), np.zeros(len(phase)))
    )
    error = np.linalg.norm(tips - goals, axis=1)
    height = tips[:, 2] / upright_height
    in_band = (height >= task.height_min) & (height <= task.height_max)
    delta = np.arctan2(tips[:, 1], tips[:, 0]) - np.arctan2(
        previous_tips[:, 1], previous_tips[:, 0]
    )
    angle = np.arctan2(np.sin(delta), np.cos(delta))
    turns = np.where(
        in_band & (np.linalg.norm(tips[:, :2], axis=1) >= 0.5 * radius),
        direction * angle / (2 * np.pi),
        0,
    )
    reward = dt * (
        task.tracking_weight * np.exp(-((error / task.tracking_scale_m) ** 2))
        - ((height - task.height_fraction) / 0.05) ** 2
        - task.velocity_weight
        * np.mean(((velocity - desired) / cfg.max_speed_m_s) ** 2, axis=1)
        - task.action_change_weight * np.mean((actions - previous_actions) ** 2, axis=1)
    )
    if task.dense_tracking:
        radial = np.linalg.norm(tips[:, :2], axis=1)
        progress = direction * angle / (2 * np.pi / task.orbit_seconds * dt)
        reward = dt * (
            -task.tracking_weight * (error / radius) ** 2
            - 0.2 * ((height - task.height_fraction) / 0.05) ** 2
            - ((radial - radius) / radius) ** 2
            + np.minimum(1.0, radial / radius) * np.clip(progress, -1.0, 1.0)
            - task.velocity_weight
            * np.mean(((velocity - desired) / cfg.max_speed_m_s) ** 2, axis=1)
            - task.action_change_weight
            * np.mean((actions - previous_actions) ** 2, axis=1)
        )
    return reward, in_band, turns, error


class IsaacSpinEnv(VecEnv):
    def __init__(
        self,
        cfg: Config,
        task: SpinConfig,
        run: Path,
        device: str,
        visible: bool,
        start_cache: Path | None = None,
    ) -> None:
        self.cfg, self.task = cfg, task
        self.physics = SpinPhysics(cfg, run, device, visible)
        self.render_mode = "human" if visible else None
        self.num_envs = cfg.training_envs
        self.rng = [
            np.random.default_rng(cfg.seed + index) for index in range(self.num_envs)
        ]
        self.upright_height = float(self.physics.tip()[0, 2])
        self.phase: Array = np.zeros(self.num_envs)
        self.direction: Array = np.ones(self.num_envs)
        self.radius: Array = np.zeros(self.num_envs)
        self.turns: Array = np.zeros(self.num_envs)
        self.band_steps = np.zeros(self.num_envs, dtype=np.int64)
        self.error_sum: Array = np.zeros(self.num_envs)
        self.elapsed = np.zeros(self.num_envs, dtype=np.int64)
        self.previous_action: Array = np.zeros((self.num_envs, 3))
        self.tip_velocity: Array = np.zeros((self.num_envs, 3))
        self.pending: Array | None = None
        self.prepared: dict[str, Array] = {}
        super().__init__(
            self.num_envs,
            gym.spaces.Box(
                -np.inf,
                np.inf,
                (self.physics.reference.nq + self.physics.reference.nv + 20,),
                dtype=np.float32,
            ),
            gym.spaces.Box(-1, 1, (3,), dtype=np.float32),
        )
        if start_cache is None:
            self._prepare_starts()
        else:
            with np.load(start_cache, allow_pickle=False) as cache:
                self.prepared = {key: cache[key].copy() for key in cache.files}
            for key, shape in {
                "q": (3, self.physics.reference.nq),
                "qd": (3, self.physics.reference.nv),
                "pulls": (3, 3),
                "speeds": (3, 3),
                "tip_velocity": (3, 3),
            }.items():
                if (
                    key not in self.prepared
                    or self.prepared[key].shape != shape
                    or not np.all(np.isfinite(self.prepared[key]))
                ):
                    raise ValueError(f"Invalid starting-state cache field: {key}")
        with (run / "start_states.npz").open("xb") as output:
            np.savez(
                output,
                q=self.prepared["q"],
                qd=self.prepared["qd"],
                pulls=self.prepared["pulls"],
                speeds=self.prepared["speeds"],
                tip_velocity=self.prepared["tip_velocity"],
            )

    def _prepare_starts(self) -> None:
        physics = self.physics
        samples: dict[str, list[Array]] = {
            key: [] for key in ("q", "qd", "pulls", "speeds", "tip_velocity")
        }
        for first in range(0, 3, self.num_envs):
            count = min(self.num_envs, 3 - first)
            ids = np.arange(self.num_envs, dtype=np.int64)
            physics.set_state(
                ids,
                physics.initial_q,
                physics.initial_qd,
                np.zeros((self.num_envs, 3)),
                np.zeros((self.num_envs, 3)),
            )
            actions = np.full((self.num_envs, 3), -0.5)
            for index in range(self.num_envs):
                actions[index, (first + index) % 3] = 1.0
            reached = np.zeros(count, dtype=bool)
            batch: dict[str, list[Array | None]] = {
                key: [None] * count for key in samples
            }
            print(
                f"Preparing native starting curls {first + 1}–{first + count}",
                flush=True,
            )
            for step in range(
                round(self.task.deep_preparation_seconds / self.cfg.control_dt)
            ):
                before = physics.tip()
                physics.step(actions)
                tips = physics.tip()
                if not np.all(np.isfinite(tips)):
                    raise RuntimeError(
                        "Non-finite physics while preparing starting curls"
                    )
                # Keep Kit responsive throughout preparation; render() already
                # limits the viewport refresh rate and skips headless runs.
                physics.render(tips)
                newly_reached = np.flatnonzero(
                    ~reached
                    & (
                        tips[:count, 2]
                        <= self.upright_height * self.task.height_fraction
                    )
                )
                if newly_reached.size:
                    values = {
                        "q": physics.state.joint_q.numpy().reshape(self.num_envs, -1),
                        "qd": physics.state.joint_qd.numpy().reshape(self.num_envs, -1),
                        "pulls": physics.pulls.numpy().reshape(self.num_envs, 3),
                        "speeds": physics.speeds.numpy().reshape(self.num_envs, 3),
                        "tip_velocity": (tips - before) / self.cfg.control_dt,
                    }
                    for index in newly_reached.tolist():
                        if np.linalg.norm(tips[index, :2]) < 0.01:
                            raise RuntimeError(
                                "Starting curl has too little radial offset"
                            )
                        for key in samples:
                            batch[key][index] = values[key][index].copy()
                        reached[index] = True
                if np.all(reached):
                    break
                if step % 250 == 0:
                    print(
                        f"Preparation {step * self.cfg.control_dt:.1f}s, heights={np.round(tips[:, 2] / self.upright_height, 3)}",
                        flush=True,
                    )
            if not np.all(reached):
                raise RuntimeError("The spin starting height was not reached in Isaac")
            for key in samples:
                samples[key].extend(value for value in batch[key] if value is not None)
        self.prepared = {key: np.array(values) for key, values in samples.items()}

    def goals(self) -> Array:
        return np.column_stack(
            (
                self.radius * np.cos(self.phase),
                self.radius * np.sin(self.phase),
                np.full(self.num_envs, self.upright_height * self.task.height_fraction),
            )
        )

    def observation(self) -> NDArray[np.float32]:
        physics = self.physics
        # Newton stores ball-joint quaternions XYZW; the original task uses WXYZ.
        q = physics.state.joint_q.numpy().reshape(self.num_envs, -1, 4)
        q = np.roll(q, 1, axis=2).reshape(self.num_envs, -1)
        result = np.concatenate(
            (
                q,
                physics.state.joint_qd.numpy().reshape(self.num_envs, -1),
                physics.pulls.numpy().reshape(self.num_envs, 3)
                / self.cfg.max_retraction_m,
                physics.speeds.numpy().reshape(self.num_envs, 3)
                / self.cfg.max_speed_m_s,
                self.previous_action,
                self.tip_velocity,
                (self.goals() - physics.tip()) / self.task.tip_tolerance_m,
                (self.elapsed * self.cfg.control_dt / self.task.episode_seconds)[
                    :, None
                ],
                np.column_stack(
                    (
                        np.sin(self.phase),
                        np.cos(self.phase),
                        self.direction,
                        self.radius / self.upright_height,
                    )
                ),
            ),
            axis=1,
        ).astype(np.float32)
        if not np.all(np.isfinite(result)):
            raise RuntimeError("Non-finite spin observation")
        return result

    def _reset_indices(self, ids: NDArray[np.int64]) -> None:
        choices = np.array([self.rng[index].integers(3) for index in ids])
        state = self.prepared
        self.physics.set_state(
            ids,
            state["q"][choices],
            state["qd"][choices],
            state["pulls"][choices],
            state["speeds"][choices],
        )
        tips = self.physics.tip()
        self.radius[ids] = np.linalg.norm(tips[ids, :2], axis=1)
        self.phase[ids] = np.arctan2(tips[ids, 1], tips[ids, 0])
        self.direction[ids] = [self.rng[index].choice([-1, 1]) for index in ids]
        self.previous_action[ids] = -0.5
        self.previous_action[ids, choices] = 1.0
        self.tip_velocity[ids] = state["tip_velocity"][choices]
        for values in (self.elapsed, self.turns, self.band_steps, self.error_sum):
            values[ids] = 0
        for index in ids:
            self.reset_infos[index] = {
                "requested_height_fraction": self.task.height_fraction,
                "start_height_reached": True,
                "start_height_fraction": float(tips[index, 2] / self.upright_height),
            }

    def reset(self) -> NDArray[np.float32]:
        for index, seed in enumerate(self._seeds):
            if seed is not None:
                self.rng[index] = np.random.default_rng(seed)
        self.pending = None
        self._reset_indices(np.arange(self.num_envs, dtype=np.int64))
        self._reset_seeds()
        self._reset_options()
        return self.observation()

    def step_async(self, actions: Array) -> None:
        actions = np.asarray(actions, dtype=np.float64)
        if actions.shape != (self.num_envs, 3) or not np.all(np.isfinite(actions)):
            raise ValueError("Expected three finite cable velocities per environment")
        self.pending = np.clip(actions, -1, 1)

    def step_wait(self) -> VecEnvStepReturn:
        if self.pending is None:
            raise RuntimeError("Call step_async before step_wait")
        actions = self.pending
        self.pending = None
        if self.task.residual_scale:
            angles = np.array([np.pi / 2, 7 * np.pi / 6, -np.pi / 6])
            amplitude = min(0.12, self.cfg.max_retraction_m, 2 * self.cfg.max_payout_m)
            target = amplitude * np.cos(self.phase[:, None] - angles)
            pulls = self.physics.pulls.numpy().reshape(self.num_envs, 3)
            baseline = np.clip(
                (target - pulls)
                / (self.cfg.action_response_seconds * self.cfg.max_speed_m_s),
                -1,
                1,
            )
            actions = np.clip(baseline + self.task.residual_scale * actions, -1, 1)
        before = self.physics.tip()
        self.physics.step(actions)
        tips = self.physics.tip()
        self.phase += (
            self.direction * 2 * np.pi / self.task.orbit_seconds * self.cfg.control_dt
        )
        rewards, in_band, turns, error = spin_reward(
            self.cfg,
            self.task,
            tips,
            before,
            self.goals(),
            self.phase,
            self.direction,
            self.radius,
            self.upright_height,
            actions,
            self.previous_action,
        )
        self.tip_velocity = (tips - before) / self.cfg.control_dt
        self.previous_action[:] = actions
        self.elapsed += 1
        self.band_steps += in_band
        self.turns += turns
        self.error_sum += error
        dones = self.elapsed * self.cfg.control_dt >= self.task.episode_seconds
        observations = self.observation()
        infos = []
        for index in range(self.num_envs):
            band = float(self.band_steps[index] / self.elapsed[index])
            mean_error = float(self.error_sum[index] / self.elapsed[index])
            info: dict[str, Any] = {
                "band_fraction": band,
                "turns": float(self.turns[index]),
                "mean_tracking_error_m": mean_error,
                "is_success": bool(
                    self.turns[index] >= 1
                    and band >= 0.8
                    and mean_error <= self.task.tracking_scale_m
                ),
                "TimeLimit.truncated": bool(dones[index]),
            }
            if dones[index]:
                info["terminal_observation"] = observations[index].copy()
            infos.append(info)
        self.physics.render(self.goals())
        if np.any(dones):
            self._reset_indices(np.flatnonzero(dones))
            observations = self.observation()
        return observations, rewards.astype(np.float32), dones, infos

    def close(self) -> None:
        self.physics.sim.stop()

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> list[Any]:
        value = getattr(self, attr_name)
        return [
            value[index]
            if isinstance(value, np.ndarray) and value.shape[0] == self.num_envs
            else value
            for index in self._get_indices(indices)
        ]

    def set_attr(
        self, attr_name: str, value: Any, indices: VecEnvIndices = None
    ) -> None:
        target = getattr(self, attr_name)
        if not isinstance(target, np.ndarray) or target.shape[0] != self.num_envs:
            raise AttributeError(f"{attr_name} is not a per-environment array")
        target[list(self._get_indices(indices))] = value

    def env_method(
        self,
        method_name: str,
        *method_args: Any,
        indices: VecEnvIndices = None,
        **method_kwargs: Any,
    ) -> list[Any]:
        raise NotImplementedError(
            f"Per-environment method is not exposed: {method_name}"
        )

    def env_is_wrapped(
        self, wrapper_class: type[gym.Wrapper], indices: VecEnvIndices = None
    ) -> list[bool]:
        return [False for _ in self._get_indices(indices)]

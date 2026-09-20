from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import TYPE_CHECKING
import xml.etree.ElementTree as ET

import gymnasium as gym
import mujoco
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

if TYPE_CHECKING:
    from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "runs" / "spiral"
PPO_ROLLOUT_STEPS = 1024
PPO_BATCH_SIZE = 64
PPO_LEARNING_RATE = 1e-4
PPO_DEVICE = "cuda"
PPO_GAMMA = 0.999
PPO_GAE_LAMBDA = 0.98
PPO_SDE_SAMPLE_FREQ = 64
PPO_TARGET_KL = 0.02
PPO_WARM_START_EPOCHS = 300
PPO_WARM_START_STATES = 8
PPO_WARM_START_LEARNING_RATE = 0.001
CHECKPOINT_STEPS = 25000
EVALUATION_TARGETS = 8
DEMO_HOLD_SECONDS = 3.0
DEMO_REST_SECONDS = 1.0
Array = NDArray[np.float64]


@dataclass
class Config:
    physics_dt: float
    control_dt: float
    gravity: list[float]
    max_retraction_m: float
    max_speed_m_s: float
    max_acceleration_m_s2: float
    cable_stiffness_n_m: float
    cable_damping_n_s_m: float
    max_tension_n: float
    episode_seconds: float
    settle_seconds: float
    settle_speed_rad_s: float
    workspace_samples: int
    tracking_scale_m: float
    action_change_weight: float
    acceleration_weight: float
    tip_acceleration_weight: float
    tip_acceleration_scale_m_s2: float
    seed: int
    training_steps: int
    spool_radius_m: float | None
    motor_steps_per_revolution: int | None
    microsteps: int | None
    max_payout_m: float = 0.0
    joint_stiffness_scale: float = 1.0
    joint_damping_scale: float = 1.0
    robot_mass_kg: float | None = None
    bend_depths: tuple[float, ...] = (0.25, 0.5, 1.0)
    sweep_directions: int = 12
    waypoint_seconds: float = 24.0
    action_mode: str = "velocity"
    action_response_seconds: float = 0.5
    advance_on_reach: bool = False
    reach_tolerance_m: float = 0.01
    reach_speed_m_s: float = 0.02
    reach_hold_seconds: float = 0.3
    time_penalty: float = 0.05
    training_seconds: float = 1800.0
    training_envs: int = 4

    def __post_init__(self) -> None:
        for name in (
            "physics_dt",
            "control_dt",
            "max_retraction_m",
            "max_speed_m_s",
            "max_acceleration_m_s2",
            "cable_stiffness_n_m",
            "max_tension_n",
            "episode_seconds",
            "settle_seconds",
            "settle_speed_rad_s",
            "tracking_scale_m",
            "tip_acceleration_scale_m_s2",
            "joint_stiffness_scale",
            "waypoint_seconds",
            "action_response_seconds",
            "reach_tolerance_m",
            "reach_speed_m_s",
            "reach_hold_seconds",
            "training_seconds",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in (
            "joint_damping_scale",
            "cable_damping_n_s_m",
            "action_change_weight",
            "acceleration_weight",
            "tip_acceleration_weight",
            "max_payout_m",
            "time_penalty",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.training_envs < 1:
            raise ValueError("training_envs must be positive")
        ratio = self.control_dt / self.physics_dt
        if ratio < 1 or not np.isclose(ratio, round(ratio)):
            raise ValueError("control_dt must be an integer multiple of physics_dt")
        if len(self.gravity) != 3 or not np.all(np.isfinite(self.gravity)):
            raise ValueError("gravity must contain three finite values")
        if self.workspace_samples < 4 or self.training_steps < 1:
            raise ValueError("workspace_samples >= 4 and training_steps >= 1 required")
        if self.action_mode not in ("velocity", "position"):
            raise ValueError("action_mode must be velocity or position")
        if self.sweep_directions < 6 or self.sweep_directions % 6:
            raise ValueError("sweep_directions must be a positive multiple of six")
        if not self.bend_depths or any(
            not np.isfinite(depth) or not 0 < depth <= 1 for depth in self.bend_depths
        ):
            raise ValueError("bend_depths must contain fractions in (0, 1]")
        for name in (
            "spool_radius_m",
            "motor_steps_per_revolution",
            "microsteps",
            "robot_mass_kg",
        ):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be positive when supplied")


def load_config(path: Path) -> Config:
    return Config(**json.loads(path.read_text()))


class Robot:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        root = ET.parse(ROOT / "robot.xml").getroot()
        for mesh in root.findall("asset/mesh"):
            mesh.set("file", str(ROOT / mesh.attrib["file"]))
        tip_body = root.find(".//body[@name='link30']")
        assert tip_body is not None
        # Center of the final three upper cable guides defines the controlled tip.
        guides = [tip_body.find(f"site[@name='s30_{i}']") for i in (2, 4, 6)]
        tip_position = np.mean(
            [np.fromstring(g.attrib["pos"], sep=" ") for g in guides if g is not None],
            axis=0,
        )
        ET.SubElement(
            tip_body,
            "site",
            name="tip",
            pos=" ".join(map(str, tip_position)),
            size="0.002",
            rgba="1 1 0 1",
        )
        world = root.find("worldbody")
        assert world is not None
        ET.SubElement(
            world,
            "site",
            name="target",
            pos="0 0 0.33",
            size="0.004",
            rgba="0.2 1 0.2 0.6",
        )
        for motor in root.findall("actuator/motor"):
            motor.set("gear", "1")
            motor.set("ctrlrange", f"{-cfg.max_tension_n} 0")
        self.model = mujoco.MjModel.from_xml_string(
            ET.tostring(root, encoding="unicode")
        )
        self.model.opt.timestep = cfg.physics_dt
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.model.opt.gravity[:] = cfg.gravity
        self.model.jnt_stiffness[:] *= cfg.joint_stiffness_scale
        self.model.dof_damping[:] *= cfg.joint_damping_scale
        self.data = mujoco.MjData(self.model)
        if cfg.robot_mass_kg is not None:
            segment_ids = [self.model.body(f"link{i}").id for i in range(1, 31)]
            mass_scale = cfg.robot_mass_kg / float(
                np.sum(self.model.body_mass[segment_ids])
            )
            self.model.body_mass[segment_ids] *= mass_scale
            self.model.body_inertia[segment_ids] *= mass_scale
            mujoco.mj_setConst(self.model, self.data)
        self.tip_id = self.model.site("tip").id
        self.target_id = self.model.site("target").id
        mujoco.mj_forward(self.model, self.data)
        self.rest_lengths = self.data.ten_length.copy()
        if cfg.max_retraction_m >= float(np.min(self.rest_lengths)):
            raise ValueError(
                f"max_retraction_m={cfg.max_retraction_m:g} m must be less than "
                f"the shortest initial cable length ({np.min(self.rest_lengths):.6f} m). "
                "Larger pulls produce a nonpositive cable rest length. "
                "Config distances are metres: 50 mm is 0.05 m."
            )
        self.retraction = np.zeros(3)
        self.velocity = np.zeros(3)
        self.reset()

    @property
    def tip(self) -> Array:
        return self.data.site_xpos[self.tip_id].copy()

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.retraction[:] = 0
        self.velocity[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def physics_step(self) -> None:
        extension = self.data.ten_length - (self.rest_lengths - self.retraction)
        tension = np.where(
            extension > 0,
            self.cfg.cable_stiffness_n_m * extension
            + self.cfg.cable_damping_n_s_m * (self.data.ten_velocity + self.velocity),
            0.0,
        )
        self.data.ctrl[:] = -np.clip(tension, 0, self.cfg.max_tension_n)
        mujoco.mj_step(self.model, self.data)
        if np.any(self.data.warning.number) or not np.all(np.isfinite(self.data.qpos)):
            raise RuntimeError(
                "MuJoCo reported unstable physics; inspect model/settings"
            )

    def advance(self, action: Array) -> None:
        if action.shape != (3,) or not np.all(np.isfinite(action)):
            raise ValueError("Expected three finite normalized cable velocities")
        cfg = self.cfg
        dt = cfg.physics_dt
        acceleration = cfg.max_acceleration_m_s2
        requested = np.clip(action, -1, 1) * cfg.max_speed_m_s
        for _ in range(round(cfg.control_dt / dt)):
            # Reserve braking distance before reaching either stroke limit.
            lower = np.sqrt(
                2 * acceleration * (self.retraction + cfg.max_payout_m)
                + (acceleration * dt) ** 2
            )
            upper = np.sqrt(
                2 * acceleration * (cfg.max_retraction_m - self.retraction)
                + (acceleration * dt) ** 2
            )
            desired = np.clip(
                requested, -lower + acceleration * dt, upper - acceleration * dt
            )
            self.velocity += np.clip(
                desired - self.velocity, -acceleration * dt, acceleration * dt
            )
            self.retraction += self.velocity * dt
            if np.any(self.retraction < -cfg.max_payout_m - 1e-10) or np.any(
                self.retraction > cfg.max_retraction_m + 1e-10
            ):
                raise RuntimeError("Cable trajectory exceeded stroke limits")
            self.retraction[:] = np.clip(
                self.retraction, -cfg.max_payout_m, cfg.max_retraction_m
            )
            self.physics_step()
        mujoco.mj_forward(self.model, self.data)

    def equilibrium(self, retraction: Array) -> Array:
        if retraction.shape != (3,) or not np.all(np.isfinite(retraction)):
            raise ValueError("Expected three finite cable retractions in metres")
        if np.any(retraction < -self.cfg.max_payout_m) or np.any(
            retraction > self.cfg.max_retraction_m
        ):
            raise ValueError("Cable retraction is outside configured stroke limits")
        self.reset()
        self.retraction[:] = retraction
        stable_steps = 0
        for _ in range(round(self.cfg.settle_seconds / self.cfg.physics_dt)):
            self.physics_step()
            if np.max(np.abs(self.data.qvel)) < self.cfg.settle_speed_rad_s:
                stable_steps += 1
            else:
                stable_steps = 0
            if stable_steps * self.cfg.physics_dt >= 0.5:
                mujoco.mj_forward(self.model, self.data)
                return self.tip
        raise RuntimeError(
            "Robot did not settle; increase settle_seconds or inspect physics"
        )

    def control(self, action: Array) -> None:
        if action.shape != (3,) or not np.all(np.isfinite(action)):
            raise ValueError("Expected three finite normalized cable commands")
        if self.cfg.action_mode == "position":
            stroke = self.cfg.max_retraction_m + self.cfg.max_payout_m
            target = (np.clip(action, -1, 1) + 1) * stroke / 2 - self.cfg.max_payout_m
            action = (target - self.retraction) / (
                self.cfg.action_response_seconds * self.cfg.max_speed_m_s
            )
        self.advance(np.clip(action, -1, 1))

    def position_action(self, retraction: Array) -> Array:
        stroke = self.cfg.max_retraction_m + self.cfg.max_payout_m
        return np.clip(2 * (retraction + self.cfg.max_payout_m) / stroke - 1, -1, 1)


def training_patterns(cfg: Config) -> tuple[Array, list[list[int]]]:
    """Build cable-space coverage and ordered position-target sequences.

    Each sequence ends at the neutral target (index zero). The cable patterns
    generate tip targets; they are not actuator commands supplied to the policy.
    Sweep angles describe cable differentials, not guaranteed Cartesian angles.
    """
    pulls = [np.zeros(3)]
    sequences: list[list[int]] = []
    for depth in cfg.bend_depths:
        subset_indices = [0]
        for mask in range(1, 8):
            active = np.array([bool(mask & (1 << cable)) for cable in range(3)])
            subset_indices.append(len(pulls))
            pulls.append(
                depth * np.where(active, cfg.max_retraction_m, -cfg.max_payout_m)
            )
        for mask in range(1, 8):
            first = subset_indices[mask]
            opposite = subset_indices[7 ^ mask]
            sequences.append([first, opposite, first, 0])

        ring: list[int] = []
        for angle in np.linspace(0, 2 * np.pi, cfg.sweep_directions, endpoint=False):
            differential = np.cos(angle - np.arange(3) * 2 * np.pi / 3)
            ring.append(len(pulls))
            pulls.append(
                depth
                * np.where(
                    differential >= 0,
                    cfg.max_retraction_m * differential,
                    cfg.max_payout_m * differential,
                )
            )
        # Complete a full turn, reverse through the same targets, then neutral.
        loop = ring + [ring[0]]
        sequences.append(loop + loop[-2::-1] + [0])

    rng = np.random.default_rng(cfg.seed)
    random_start = len(pulls)
    pulls.extend(
        rng.uniform(-cfg.max_payout_m, cfg.max_retraction_m, (cfg.workspace_samples, 3))
    )
    for index in range(random_start, len(pulls)):
        other = random_start + (index - random_start + 1) % cfg.workspace_samples
        sequences.append([index, other, index, 0])
    return np.array(pulls), sequences


def workspace(robot: Robot, pulls: Array | None = None) -> tuple[Array, Array]:
    rng = np.random.default_rng(robot.cfg.seed)
    if pulls is None:
        pulls = rng.uniform(
            -robot.cfg.max_payout_m,
            robot.cfg.max_retraction_m,
            (robot.cfg.workspace_samples, 3),
        )
        pulls[0] = 0
        for i in range(3):
            pulls[i + 1] = -robot.cfg.max_payout_m
            pulls[i + 1, i] = robot.cfg.max_retraction_m
    tips = []
    for index, pull in enumerate(pulls):
        try:
            tips.append(robot.equilibrium(pull))
        except (mujoco.FatalError, RuntimeError) as error:
            raise RuntimeError(
                f"Target {index + 1}/{len(pulls)} failed for cable pulls "
                f"{np.round(pull * 1000, 3).tolist()} mm at simulation time "
                f"{robot.data.time:.3f} s: {error}"
            ) from error
        if (index + 1) % 12 == 0:
            print(f"Settled targets: {index + 1}/{len(pulls)}", flush=True)
    robot.reset()
    return pulls, np.array(tips)


class SpiralEnv(gym.Env):
    metadata: dict[str, list[str]] = {"render_modes": []}

    def __init__(
        self, cfg: Config, goals: Array, sequences: list[list[int]] | None = None
    ) -> None:
        self.robot = Robot(cfg)
        self.cfg = cfg
        self.goals = goals.copy()
        self.sequences = sequences
        if sequences is not None and (
            not sequences
            or any(
                not sequence
                or any(index < 0 or index >= len(goals) for index in sequence)
                for sequence in sequences
            )
        ):
            raise ValueError("Sequences must contain valid goal indices")
        self.sequence_order: list[int] = []
        self.active_sequence: list[int] = []
        self.sequence_index = -1
        self.waypoint_index = 0
        # Allow enough time to traverse the full cable stroke and stop.
        minimum_seconds = (cfg.max_retraction_m + cfg.max_payout_m) / cfg.max_speed_m_s
        minimum_seconds += 2 * cfg.max_speed_m_s / cfg.max_acceleration_m_s2
        self.waypoint_steps = max(
            1, round(max(cfg.waypoint_seconds, minimum_seconds) / cfg.control_dt)
        )
        self.action_space = gym.spaces.Box(-1.0, 1.0, (3,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            -np.inf,
            np.inf,
            (
                self.robot.model.nq
                + self.robot.model.nv
                + 18
                + 2 * cfg.advance_on_reach,
            ),
            dtype=np.float32,
        )
        self.goal = goals[0].copy()
        self.previous_action = np.zeros(3)
        self.tip_velocity = np.zeros(3)
        self.elapsed = 0
        self.waypoint_elapsed = 0
        self.reach_steps = 0

    def observation(self) -> NDArray[np.float32]:
        robot = self.robot
        return np.concatenate(
            (
                robot.data.qpos,
                robot.data.qvel,
                (self.goal - robot.tip) / self.cfg.tracking_scale_m,
                robot.retraction / self.cfg.max_retraction_m,
                robot.velocity / self.cfg.max_speed_m_s,
                robot.data.ten_length / robot.rest_lengths,
                self.previous_action,
                self.tip_velocity / self.cfg.max_speed_m_s,
                [
                    self.waypoint_elapsed / self.waypoint_steps,
                    self.reach_steps
                    * self.cfg.control_dt
                    / self.cfg.reach_hold_seconds,
                ]
                if self.cfg.advance_on_reach
                else [],
            )
        ).astype(np.float32)

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[NDArray[np.float32], dict]:
        super().reset(seed=seed)
        self.robot.reset()
        if seed is not None:
            self.sequence_order = []
        self.waypoint_index = 0
        if self.sequences is None:
            self.goal = self.goals[self.np_random.integers(len(self.goals))].copy()
        else:
            if not self.sequence_order:
                self.sequence_order = self.np_random.permutation(
                    len(self.sequences)
                ).tolist()
            self.sequence_index = self.sequence_order.pop()
            self.active_sequence = self.sequences[self.sequence_index]
            self.goal = self.goals[self.active_sequence[0]].copy()
        self.robot.model.site_pos[self.robot.target_id] = self.goal
        self.previous_action[:] = 0
        self.tip_velocity[:] = 0
        self.elapsed = 0
        self.waypoint_elapsed = 0
        self.reach_steps = 0
        return self.observation(), {}

    def step(
        self, action: Array
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict]:
        action = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        previous_tip = self.robot.tip
        previous_velocity = self.robot.velocity.copy()
        self.robot.control(action)
        velocity = (self.robot.tip - previous_tip) / self.cfg.control_dt
        tip_acceleration = (velocity - self.tip_velocity) / self.cfg.control_dt
        acceleration = (self.robot.velocity - previous_velocity) / self.cfg.control_dt
        error = float(np.linalg.norm(self.robot.tip - self.goal))
        reward = -(
            (error / self.cfg.tracking_scale_m) ** 2
            + self.cfg.action_change_weight
            * np.mean((action - self.previous_action) ** 2)
            + self.cfg.acceleration_weight
            * np.mean((acceleration / self.cfg.max_acceleration_m_s2) ** 2)
            + self.cfg.tip_acceleration_weight
            * np.mean((tip_acceleration / self.cfg.tip_acceleration_scale_m_s2) ** 2)
        )
        self.previous_action = action.copy()
        self.tip_velocity = velocity
        self.elapsed += 1
        self.waypoint_elapsed += 1
        settled = (
            error <= self.cfg.reach_tolerance_m
            and np.linalg.norm(velocity) <= self.cfg.reach_speed_m_s
        )
        self.reach_steps = self.reach_steps + 1 if settled else 0
        reached = self.reach_steps * self.cfg.control_dt >= self.cfg.reach_hold_seconds
        terminated = False
        info = {
            "tip_error_m": error,
            "retraction_m": self.robot.retraction.copy(),
            "sequence_index": self.sequence_index,
            "waypoint_index": self.waypoint_index,
            "waypoint_reached": bool(reached),
            "waypoint_seconds": self.waypoint_elapsed * self.cfg.control_dt,
            "is_success": False,
        }
        if self.cfg.advance_on_reach:
            reward = self.cfg.control_dt * (
                reward - self.cfg.time_penalty * float(not settled)
            )
            reward += float(reached)
            truncated = self.waypoint_elapsed >= self.waypoint_steps and not reached
            if reached:
                terminated = self.sequences is None or self.waypoint_index + 1 >= len(
                    self.active_sequence
                )
                info["is_success"] = terminated
                if not terminated:
                    self.waypoint_index += 1
                    self.goal = self.goals[
                        self.active_sequence[self.waypoint_index]
                    ].copy()
                    self.robot.model.site_pos[self.robot.target_id] = self.goal
                    self.waypoint_elapsed = 0
                    self.reach_steps = 0
        elif self.sequences is None:
            truncated = self.elapsed >= round(
                self.cfg.episode_seconds / self.cfg.control_dt
            )
        else:
            truncated = self.elapsed >= self.waypoint_steps * len(self.active_sequence)
            if not truncated and self.elapsed % self.waypoint_steps == 0:
                self.waypoint_index += 1
                self.goal = self.goals[self.active_sequence[self.waypoint_index]].copy()
                self.robot.model.site_pos[self.robot.target_id] = self.goal
        return (
            self.observation(),
            float(reward),
            terminated,
            truncated,
            info,
        )


def demo_target(cfg: Config, seconds: float) -> Array:
    # The quintic ramp has peak normalized speed 1.875 and zero end acceleration.
    travel = max(cfg.max_retraction_m, cfg.max_payout_m)
    ramp_seconds = max(
        2 * travel / cfg.max_speed_m_s, np.sqrt(6 * travel / cfg.max_acceleration_m_s2)
    )
    cycle_seconds = 2 * ramp_seconds + DEMO_HOLD_SECONDS + DEMO_REST_SECONDS
    cable = int(seconds // cycle_seconds) % 3
    phase = seconds % cycle_seconds
    if phase < ramp_seconds:
        fraction = phase / ramp_seconds
    elif phase < ramp_seconds + DEMO_HOLD_SECONDS:
        fraction = 1.0
    else:
        fraction = max(
            0.0, 1 - (phase - ramp_seconds - DEMO_HOLD_SECONDS) / ramp_seconds
        )
    ramp = fraction**3 * (10 - 15 * fraction + 6 * fraction**2)
    desired = np.full(3, -cfg.max_payout_m)
    desired[cable] = cfg.max_retraction_m
    return desired * ramp


def sweep_target(cfg: Config, seconds: float) -> Array:
    amplitude = (cfg.max_retraction_m + cfg.max_payout_m) / 2
    offset = (cfg.max_retraction_m - cfg.max_payout_m) / 2
    period = max(40.0, 4 * np.pi * amplitude / cfg.max_speed_m_s)
    ramp_seconds = 2 * max(cfg.max_retraction_m, cfg.max_payout_m) / cfg.max_speed_m_s
    fraction = np.clip(seconds / ramp_seconds, 0, 1)
    ramp = fraction**3 * (10 - 15 * fraction + 6 * fraction**2)
    # Actual guide angles in robot.xml; continuously interpolate all azimuths.
    cable_angles = np.array([np.pi / 2, 7 * np.pi / 6, -np.pi / 6])
    angle = 2 * np.pi * seconds / period
    return ramp * (offset + amplitude * np.cos(angle - cable_angles))


def demo(
    cfg: Config, policy_path: Path | None, seconds: float, sweep: bool = False
) -> None:
    import mujoco.viewer

    robot = Robot(cfg)
    env = None
    policy = None
    if policy_path is not None:
        from stable_baselines3 import PPO

        with np.load(policy_path.parent / "workspace.npz") as saved:
            goals = saved["tips"]
            sequences = (
                [row[row >= 0].tolist() for row in saved["sequences"]]
                if "sequences" in saved
                else None
            )
        env = SpiralEnv(cfg, goals, sequences)
        env.reset(seed=cfg.seed)
        robot = env.robot
        policy = PPO.load(policy_path, env=env, device=PPO_DEVICE)
    else:
        robot.model.site_rgba[robot.target_id, 3] = 0
    print("Close the viewer to stop. Positive travel pulls; negative travel pays out.")
    if sweep:
        print("Continuous 360-degree bending-direction sweep; not axial twisting.")
    print(f"Joint stiffness scale: {cfg.joint_stiffness_scale:g} relative to robot.xml")
    if policy is not None:
        print(
            f"Saved policy action mode: {cfg.action_mode}; PPO steps: {policy.num_timesteps}"
        )
    with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
        viewer.cam.lookat[:] = [0, 0, 0.17]
        viewer.cam.distance = 0.7
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -15
        start = time.monotonic()
        step = 0
        while viewer.is_running() and (seconds <= 0 or step * cfg.control_dt < seconds):
            if policy is not None and env is not None:
                action, _ = policy.predict(env.observation(), deterministic=True)
                _, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    env.reset()
            else:
                desired = (
                    sweep_target(cfg, step * cfg.control_dt)
                    if sweep
                    else demo_target(cfg, step * cfg.control_dt)
                )
                robot.advance(
                    np.clip(
                        (desired - robot.retraction) / (0.5 * cfg.max_speed_m_s), -1, 1
                    )
                )
            if step % round(1 / cfg.control_dt) == 0:
                print(
                    f"tip mm={np.round(robot.tip * 1000, 2)}  pull mm={np.round(robot.retraction * 1000, 2)}"
                )
                if policy is not None and env is not None:
                    print(
                        f"action={np.round(action, 3)} target error={np.linalg.norm(robot.tip - env.goal) * 1000:.1f} mm"
                    )
            viewer.sync()
            step += 1
            time.sleep(max(0, start + step * cfg.control_dt - time.monotonic()))


def warm_start_policy(model: PPO, env: SpiralEnv, pulls: Array) -> None:
    """Initialize position commands from the simulated inverse-mapping samples.

    Only the actor is fitted. PPO subsequently trains the critic and fine-tunes
    the actor using tracking and smoothness rewards. This is not grasp training.
    """
    import torch

    if env.cfg.action_mode != "position":
        return
    observations = []
    targets = []
    indices = np.linspace(1, len(pulls) - 1, PPO_WARM_START_STATES - 1, dtype=int)
    for state in np.vstack((np.zeros(3), pulls[indices])):
        env.robot.equilibrium(state)
        env.previous_action = env.robot.position_action(state)
        env.tip_velocity[:] = 0
        for goal, pull in zip(env.goals, pulls, strict=True):
            env.goal = goal.copy()
            observations.append(env.observation())
            targets.append(env.robot.position_action(pull))
    inputs = torch.as_tensor(np.array(observations), device=model.device)
    commands = torch.as_tensor(
        np.array(targets), dtype=torch.float32, device=model.device
    )
    parameters = list(model.policy.mlp_extractor.policy_net.parameters()) + list(
        model.policy.action_net.parameters()
    )
    optimizer = torch.optim.Adam(parameters, lr=PPO_WARM_START_LEARNING_RATE)
    for epoch in range(PPO_WARM_START_EPOCHS):
        for batch in torch.randperm(len(inputs), device=model.device).split(
            PPO_BATCH_SIZE
        ):
            predicted = model.policy._predict(inputs[batch], deterministic=True)
            loss = (predicted - commands[batch]).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(
                f"Actor initialization epoch {epoch + 1}/{PPO_WARM_START_EPOCHS}, command MSE {loss.item():.4f}",
                flush=True,
            )
    env.reset(seed=env.cfg.seed)


def train(cfg: Config, run: Path = RUN) -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import (
        BaseCallback,
        CheckpointCallback,
        EvalCallback,
    )
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv

    deadline = time.monotonic() + cfg.training_seconds
    print(
        f"Budget: {cfg.training_seconds / 60:g} minutes including preparation; {cfg.training_envs} simulations, PPO device {PPO_DEVICE}. Final evaluation/saving may extend this slightly.",
        flush=True,
    )

    class TimeBudgetCallback(BaseCallback):
        def _on_step(self) -> bool:
            return time.monotonic() < deadline

    run.mkdir(parents=True, exist_ok=True)
    if (run / "config.json").exists():
        raise FileExistsError(
            f"Training run already exists at {run}; choose a new --run directory"
        )
    print("Sampling settled, reachable targets from the cable model...", flush=True)
    pulls, sequences = training_patterns(cfg)
    pulls, tips = workspace(Robot(cfg), pulls)
    print(
        f"Training on {len(tips)} targets in {len(sequences)} motion sequences.",
        flush=True,
    )
    padded = np.full((len(sequences), max(map(len, sequences))), -1, dtype=np.int64)
    for index, sequence in enumerate(sequences):
        padded[index, : len(sequence)] = sequence
    np.savez(run / "workspace.npz", pulls=pulls, tips=tips, sequences=padded)
    (run / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    env = SpiralEnv(cfg, tips, sequences)
    check_env(env)
    monitored_env = make_vec_env(
        lambda: SpiralEnv(cfg, tips, sequences),
        n_envs=cfg.training_envs,
        seed=cfg.seed,
        monitor_dir=str(run),
        monitor_kwargs={"info_keywords": ("is_success", "tip_error_m")},
        vec_env_cls=SubprocVecEnv if cfg.training_envs > 1 else None,
    )
    model = PPO(
        "MlpPolicy",
        monitored_env,
        seed=cfg.seed,
        device=PPO_DEVICE,
        verbose=1,
        n_steps=PPO_ROLLOUT_STEPS,
        batch_size=PPO_BATCH_SIZE,
        learning_rate=PPO_LEARNING_RATE,
        gamma=PPO_GAMMA,
        gae_lambda=PPO_GAE_LAMBDA,
        use_sde=True,
        sde_sample_freq=PPO_SDE_SAMPLE_FREQ,
        target_kl=PPO_TARGET_KL,
        policy_kwargs={"squash_output": True, "log_std_init": -1.0},
    )
    eval_env: Monitor | None = None
    try:
        warm_start_policy(model, env, pulls)
        model.save(run / "warm_start_policy")
        # Keep an independently sampled validation set for model selection.
        rng = np.random.default_rng(cfg.seed + 1)
        validation_pulls = rng.uniform(
            -cfg.max_payout_m, cfg.max_retraction_m, (EVALUATION_TARGETS, 3)
        )
        _, validation_goals = workspace(Robot(cfg), validation_pulls)
        eval_env = Monitor(SpiralEnv(cfg, validation_goals))
        eval_env.reset(seed=cfg.seed + 1)
        initial_reward, _ = evaluate_policy(
            model, eval_env, n_eval_episodes=EVALUATION_TARGETS
        )
        initial_reward = float(np.mean(initial_reward))
        model.save(run / "best_model")
        evaluator = EvalCallback(
            eval_env,
            best_model_save_path=str(run),
            log_path=str(run),
            eval_freq=max(1, CHECKPOINT_STEPS // cfg.training_envs),
            n_eval_episodes=EVALUATION_TARGETS,
        )
        evaluator.best_mean_reward = float(initial_reward)
        print(f"Initial held-out mean reward: {initial_reward:.3f}", flush=True)
        model.learn(
            cfg.training_steps,
            callback=[
                CheckpointCallback(
                    save_freq=max(1, CHECKPOINT_STEPS // cfg.training_envs),
                    save_path=str(run),
                    name_prefix="policy",
                ),
                evaluator,
                TimeBudgetCallback(),
            ],
        )
        model.save(run / "last_policy")
        eval_env.reset(seed=cfg.seed + 1)
        final_reward, _ = evaluate_policy(
            model, eval_env, n_eval_episodes=EVALUATION_TARGETS
        )
        if float(np.mean(final_reward)) > evaluator.best_mean_reward:
            model.save(run / "best_model")
        best_model = PPO.load(run / "best_model.zip", device=PPO_DEVICE)
        best_model.save(run / "policy")
        print(
            "policy.zip contains the best evaluated policy; last_policy.zip contains the final update."
        )
    finally:
        monitored_env.close()
        env.close()
        if eval_env is not None:
            eval_env.close()


def evaluate(cfg: Config, policy_path: Path) -> None:
    from stable_baselines3 import PPO

    rng = np.random.default_rng(cfg.seed + 10000)
    robot = Robot(cfg)
    goals = np.array(
        [
            robot.equilibrium(pull)
            for pull in rng.uniform(
                -cfg.max_payout_m, cfg.max_retraction_m, (EVALUATION_TARGETS, 3)
            )
        ]
    )
    env = SpiralEnv(cfg, goals)
    policy = PPO.load(policy_path, env=env, device=PPO_DEVICE)
    for label in ("neutral command", "trained policy"):
        errors: list[float] = []
        accelerations: list[float] = []
        successes = 0
        arrival_times: list[float] = []
        for goal in goals:
            env.reset(seed=cfg.seed)
            env.goal = goal.copy()
            for _ in range(round(cfg.episode_seconds / cfg.control_dt)):
                previous_velocity = env.tip_velocity.copy()
                action = (
                    (
                        env.robot.position_action(np.zeros(3))
                        if cfg.action_mode == "position"
                        else np.zeros(3)
                    )
                    if label == "neutral command"
                    else policy.predict(env.observation(), deterministic=True)[0]
                )
                _, _, terminated, truncated, info = env.step(action)
                accelerations.append(
                    float(
                        np.linalg.norm(
                            (env.tip_velocity - previous_velocity) / cfg.control_dt
                        )
                    )
                )
                if terminated or truncated:
                    if info["is_success"]:
                        successes += 1
                        arrival_times.append(env.elapsed * cfg.control_dt)
                    break
            errors.append(float(np.linalg.norm(env.robot.tip - goal)))
        print(
            f"{label}: held-out final error mean={np.mean(errors) * 1000:.3f} mm, "
            f"max={np.max(errors) * 1000:.3f} mm; "
            f"tip acceleration RMS={np.sqrt(np.mean(np.square(accelerations))):.4f} m/s^2"
        )
        if cfg.advance_on_reach:
            print(
                f"Settled targets: {successes}/{len(goals)}; arrival times (s): {np.round(arrival_times, 2)}"
            )
    env.close()


def mapping(
    cfg: Config, pull_mm: list[float] | None, target_mm: list[float] | None
) -> None:
    robot = Robot(cfg)
    if pull_mm is not None:
        pull = np.asarray(pull_mm) / 1000
        position = robot.equilibrium(pull)
    else:
        assert target_mm is not None
        target = np.asarray(target_mm) / 1000
        if not np.all(np.isfinite(target)):
            raise ValueError("Target coordinates must be finite")
        pulls, tips = workspace(robot)
        initial = pulls[np.argmin(np.linalg.norm(tips - target, axis=1))]
        stroke = cfg.max_retraction_m + cfg.max_payout_m
        # Use normalized stroke variables so numerical differences remain resolvable.
        result = least_squares(
            lambda x: (
                (robot.equilibrium(x * stroke - cfg.max_payout_m) - target)
                / cfg.tracking_scale_m
            ),
            np.clip((initial + cfg.max_payout_m) / stroke, 0.001, 0.999),
            bounds=(0, 1),
            diff_step=0.01,
            max_nfev=60,
        )
        pull = result.x * stroke - cfg.max_payout_m
        position = robot.equilibrium(pull)
        print(
            f"Target residual: {np.linalg.norm(position - target) * 1000:.3f} mm; solver success: {result.success}"
        )
        print(
            "A nonzero residual can indicate an unreachable target or a local solution."
        )
    print(f"Tip XYZ mm: {position * 1000}")
    print(f"Cable take-up / direct linear actuator travel mm: {pull * 1000}")
    if cfg.spool_radius_m is not None:
        revolutions = pull / (2 * np.pi * cfg.spool_radius_m)
        print(f"Constant-radius spool revolutions: {revolutions}")
        if cfg.motor_steps_per_revolution is not None and cfg.microsteps is not None:
            print(
                f"Microsteps from zero: {np.rint(revolutions * cfg.motor_steps_per_revolution * cfg.microsteps).astype(np.int64)}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Three-cable spiral simulation and smooth RL control"
    )
    parser.add_argument("mode", choices=("demo", "train", "map", "evaluate"))
    parser.add_argument("--config", type=Path, default=ROOT / "spiral_config.json")
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Continuous direction sweep for a no-policy demo",
    )
    parser.add_argument(
        "--seconds", type=float, default=0, help="Demo duration; zero runs until closed"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--pull-mm", nargs=3, type=float)
    group.add_argument("--target-mm", nargs=3, type=float)
    args = parser.parse_args()
    if args.sweep and (args.mode != "demo" or args.policy is not None):
        parser.error("--sweep requires demo without --policy")
    if args.mode == "evaluate" and args.policy is None:
        parser.error("evaluate requires --policy")
    cfg = load_config(
        args.policy.parent / "config.json" if args.policy else args.config
    )
    if args.mode == "demo":
        demo(cfg, args.policy, args.seconds, args.sweep)
    elif args.mode == "train":
        train(cfg, args.run)
    elif args.mode == "evaluate":
        evaluate(cfg, args.policy)
    elif args.pull_mm is not None or args.target_mm is not None:
        mapping(cfg, args.pull_mm, args.target_mm)
    else:
        parser.error("map requires --pull-mm or --target-mm")


if __name__ == "__main__":
    main()

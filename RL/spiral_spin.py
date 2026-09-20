from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import time

import numpy as np
from numpy.typing import NDArray

from spiral import Array, Config, ROOT, load_config, Robot
from spiral_upright import UprightConfig, UprightEnv


@dataclass
class SpinConfig(UprightConfig):
    episode_seconds: float = 120.0
    orbit_seconds: float = 90.0
    height_fraction: float = 0.45
    height_min: float = 0.40
    height_max: float = 0.50
    tracking_scale_m: float = 0.04
    tracking_weight: float = 2.0
    velocity_weight: float = 0.1
    residual_scale: float = 0.0
    dense_tracking: bool = False


class SpinEnv(UprightEnv):
    def __init__(self, cfg: Config, task: SpinConfig) -> None:
        import gymnasium as gym

        super().__init__(cfg, task)
        self.spin = task
        self.upright_goal = self.goal.copy()
        self.phase = 0.0
        self.direction = 1.0
        self.radius = 0.1
        self.turns = 0.0
        self.band_steps = 0
        self.error_sum = 0.0
        self.observation_space = gym.spaces.Box(
            -np.inf,
            np.inf,
            (self.robot.model.nq + self.robot.model.nv + 20,),
            dtype=np.float32,
        )

    def observation(self) -> NDArray[np.float32]:
        return np.concatenate(
            (
                super().observation(),
                [
                    np.sin(self.phase),
                    np.cos(self.phase),
                    self.direction,
                    self.radius / self.upright_goal[2],
                ],
            )
        ).astype(np.float32)

    def orbit_target(self) -> Array:
        return np.array(
            [
                self.radius * np.cos(self.phase),
                self.radius * np.sin(self.phase),
                self.upright_goal[2] * self.spin.height_fraction,
            ]
        )

    def display_state(self) -> tuple:  # The display also follows the moving target.
        return (*super().display_state(), self.goal.copy())

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[NDArray[np.float32], dict]:
        self.goal = self.upright_goal.copy()
        _, info = super().reset(
            seed=seed, options={"height_fraction": self.spin.height_fraction}
        )
        if not info["start_height_reached"]:
            raise RuntimeError(
                "The spin starting height is not reachable with this configuration."
            )
        self.radius = float(np.linalg.norm(self.robot.tip[:2]))
        if self.radius < 0.01:
            raise RuntimeError(
                "Starting curl has too little radial offset for a circular task."
            )
        self.phase = float(np.arctan2(self.robot.tip[1], self.robot.tip[0]))
        self.direction = float(self.np_random.choice([-1, 1]))
        self.turns = 0.0
        self.band_steps = 0
        self.error_sum = 0.0
        self.goal = self.orbit_target()
        self.robot.model.site_pos[self.robot.target_id] = self.goal
        return self.observation(), info

    def step(
        self, action: Array
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict]:
        action = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        if self.spin.residual_scale:
            action = np.clip(
                self.cable_sweep_action() + self.spin.residual_scale * action, -1, 1
            )
        previous_tip = self.robot.tip
        self.robot.control(action)
        dt = self.cfg.control_dt
        omega = self.direction * 2 * np.pi / self.spin.orbit_seconds
        self.phase += omega * dt
        self.goal = self.orbit_target()
        self.robot.model.site_pos[self.robot.target_id] = self.goal
        self.tip_velocity = (self.robot.tip - previous_tip) / dt
        desired_velocity = (
            self.radius * omega * np.array([-np.sin(self.phase), np.cos(self.phase), 0])
        )
        error = float(np.linalg.norm(self.robot.tip - self.goal))
        height = self.robot.tip[2] / self.upright_goal[2]
        in_band = self.spin.height_min <= height <= self.spin.height_max
        before = np.arctan2(previous_tip[1], previous_tip[0])
        after = np.arctan2(self.robot.tip[1], self.robot.tip[0])
        angle = np.arctan2(np.sin(after - before), np.cos(after - before))
        # Do not count movement through the axis or outside the requested band.
        if in_band and np.linalg.norm(self.robot.tip[:2]) >= 0.5 * self.radius:
            self.turns += self.direction * float(angle) / (2 * np.pi)
        reward = dt * (
            self.spin.tracking_weight
            * np.exp(-((error / self.spin.tracking_scale_m) ** 2))
            - ((height - self.spin.height_fraction) / 0.05) ** 2
            - self.spin.velocity_weight
            * float(
                np.mean(
                    ((self.tip_velocity - desired_velocity) / self.cfg.max_speed_m_s)
                    ** 2
                )
            )
            - self.spin.action_change_weight
            * float(np.mean((action - self.previous_action) ** 2))
        )
        if self.spin.dense_tracking:
            radial_error = (
                np.linalg.norm(self.robot.tip[:2]) - self.radius
            ) / self.radius
            progress = (
                self.direction
                * float(angle)
                / (2 * np.pi / self.spin.orbit_seconds * dt)
            )
            radial_gate = min(1.0, np.linalg.norm(self.robot.tip[:2]) / self.radius)
            reward = dt * (
                -self.spin.tracking_weight * (error / self.radius) ** 2
                - 0.2 * ((height - self.spin.height_fraction) / 0.05) ** 2
                - radial_error**2
                + radial_gate * np.clip(progress, -1.0, 1.0)
                - self.spin.velocity_weight
                * float(
                    np.mean(
                        (
                            (self.tip_velocity - desired_velocity)
                            / self.cfg.max_speed_m_s
                        )
                        ** 2
                    )
                )
                - self.spin.action_change_weight
                * float(np.mean((action - self.previous_action) ** 2))
            )
        self.previous_action = action.copy()
        self.elapsed += 1
        self.band_steps += int(in_band)
        self.error_sum += error
        info = {
            "band_fraction": self.band_steps / self.elapsed,
            "turns": self.turns,
            "mean_tracking_error_m": self.error_sum / self.elapsed,
            "is_success": bool(
                self.turns >= 1
                and self.band_steps / self.elapsed >= 0.8
                and self.error_sum / self.elapsed <= self.spin.tracking_scale_m
            ),
        }
        return (
            self.observation(),
            float(reward),
            False,
            self.elapsed * dt >= self.spin.episode_seconds,
            info,
        )

    def baseline_action(self) -> Array:
        return np.zeros(3) if self.spin.residual_scale else self.cable_sweep_action()

    def cable_sweep_action(self) -> Array:
        # Geometric cable sweep is a baseline, not an inverse-kinematics solution.
        angles = np.array([np.pi / 2, 7 * np.pi / 6, -np.pi / 6])
        amplitude = min(0.12, self.cfg.max_retraction_m, 2 * self.cfg.max_payout_m)
        target = amplitude * np.cos(self.phase - angles)
        return np.clip(
            (target - self.robot.retraction)
            / (self.cfg.action_response_seconds * self.cfg.max_speed_m_s),
            -1,
            1,
        )


def train(cfg: Config, task: SpinConfig, run: Path, headless: bool = False) -> None:
    import mujoco.viewer
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv

    deadline = time.monotonic() + cfg.training_seconds
    if run.exists():
        raise FileExistsError(f"Choose a new run directory: {run}")
    # Validate before writing a run or launching subprocesses.
    probe = SpinEnv(cfg, task)
    probe.reset(seed=cfg.seed)
    probe.close()
    run.mkdir(parents=True)
    (run / "robot_config.json").write_text(json.dumps(asdict(cfg), indent=2))
    (run / "spin_config.json").write_text(json.dumps(asdict(task), indent=2))

    display_robot = None
    viewer = None
    last_display = 0.0

    class BudgetCallback(BaseCallback):
        def _on_step(self) -> bool:
            nonlocal last_display
            now = time.monotonic()
            if (
                viewer is not None
                and display_robot is not None
                and viewer.is_running()
                and now - last_display >= 1 / 20
            ):
                qpos, qvel, sim_time, goal = self.training_env.env_method(
                    "display_state", indices=0
                )[0]
                with viewer.lock():
                    display_robot.data.qpos[:] = qpos
                    display_robot.data.qvel[:] = qvel
                    display_robot.data.time = sim_time
                    display_robot.model.site_pos[display_robot.target_id] = goal
                    mujoco.mj_forward(display_robot.model, display_robot.data)
                viewer.sync()
                last_display = now
            return time.monotonic() < deadline

    env = make_vec_env(
        lambda: SpinEnv(cfg, task),
        n_envs=cfg.training_envs,
        seed=cfg.seed,
        monitor_dir=str(run),
        monitor_kwargs={"info_keywords": ("is_success", "band_fraction", "turns")},
        vec_env_cls=SubprocVecEnv if cfg.training_envs > 1 else None,
    )
    try:
        if not headless:
            display_robot = Robot(replace(cfg, gravity=[0.0, 0.0, -9.81]))
            display_robot.model.site_pos[display_robot.target_id] = display_robot.tip
            viewer = mujoco.viewer.launch_passive(
                display_robot.model, display_robot.data
            )
            viewer.cam.lookat[:] = [0, 0, 0.16]
            viewer.cam.distance = 0.7
            print(
                "Live view: training environment 1. Closing the view keeps training running.",
                flush=True,
            )
        model = PPO(
            "MlpPolicy",
            env,
            device="cuda",
            seed=cfg.seed,
            verbose=1,
            learning_rate=task.learning_rate,
            n_steps=task.rollout_steps,
            batch_size=task.batch_size,
            n_epochs=task.epochs,
            target_kl=task.target_kl,
            gamma=0.999,
            gae_lambda=0.98,
            use_sde=True,
            sde_sample_freq=64,
            policy_kwargs={"squash_output": True, "log_std_init": -1.0},
        )
        print(
            f"Circular tracking: {cfg.training_seconds / 60:g} minute budget, CUDA; +Z upright.",
            flush=True,
        )
        model.learn(
            cfg.training_steps,
            callback=[
                BudgetCallback(),
                CheckpointCallback(
                    max(1, 25000 // cfg.training_envs), str(run), name_prefix="spin"
                ),
            ],
        )
        model.save(run / "policy")
    finally:
        if viewer is not None:
            viewer.close()
        env.close()


def demo(cfg: Config, task: SpinConfig, policy_path: Path | None) -> None:
    import mujoco.viewer
    from stable_baselines3 import PPO

    env = SpinEnv(cfg, task)
    policy = PPO.load(policy_path, env=env, device="cuda") if policy_path else None
    obs, _ = env.reset(seed=cfg.seed)
    print(
        "Green target circles at 45% height. "
        + (
            "PPO controller."
            if policy
            else "Cable-sweep baseline; tracking is not guaranteed."
        )
    )
    with mujoco.viewer.launch_passive(env.robot.model, env.robot.data) as viewer:
        viewer.cam.lookat[:] = [0, 0, 0.15]
        viewer.cam.distance = 0.7
        while viewer.is_running():
            start = time.monotonic()
            action = (
                policy.predict(obs, deterministic=True)[0]
                if policy
                else env.baseline_action()
            )
            obs, _, _, done, info = env.step(action)
            viewer.sync()
            if done:
                print(info)
                obs, _ = env.reset()
            time.sleep(max(0, cfg.control_dt - time.monotonic() + start))
    env.close()


def evaluate(cfg: Config, task: SpinConfig, policy_path: Path) -> None:
    from stable_baselines3 import PPO

    env = SpinEnv(cfg, task)
    policy = PPO.load(policy_path, env=env, device="cuda")
    for controller in ("cable sweep", "PPO"):
        results = []
        for episode in range(task.evaluation_episodes):
            obs, _ = env.reset(seed=cfg.seed + 10000 + episode)
            done = False
            while not done:
                action = (
                    policy.predict(obs, deterministic=True)[0]
                    if controller == "PPO"
                    else env.baseline_action()
                )
                obs, _, _, done, info = env.step(action)
            results.append(info)
        print(
            controller,
            {
                key: float(np.mean([r[key] for r in results]))
                for key in (
                    "band_fraction",
                    "turns",
                    "mean_tracking_error_m",
                    "is_success",
                )
            },
        )
    env.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PPO circular tip tracking at 40?50% upright height"
    )
    parser.add_argument("mode", choices=("train", "demo", "evaluate"))
    parser.add_argument("--config", type=Path, default=ROOT / "spiral_config.json")
    parser.add_argument("--run", type=Path, default=ROOT / "runs" / "spiral_spin")
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    if args.mode == "evaluate" and args.policy is None:
        parser.error("evaluate requires --policy")
    if args.mode == "train" and args.policy is not None:
        parser.error("Training starts fresh; --policy is for demo/evaluate")
    cfg = load_config(
        args.policy.parent / "robot_config.json" if args.policy else args.config
    )
    task = (
        SpinConfig(**json.loads((args.policy.parent / "spin_config.json").read_text()))
        if args.policy
        else SpinConfig(residual_scale=0.15, dense_tracking=True)
    )
    if args.mode == "train":
        train(cfg, task, args.run, args.headless)
    elif args.mode == "evaluate":
        evaluate(cfg, task, args.policy)
    else:
        demo(cfg, task, args.policy)


if __name__ == "__main__":
    main()

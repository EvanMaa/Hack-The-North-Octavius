from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import time
import warnings

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from spiral import Array, Config, ROOT, Robot, load_config


@dataclass
class UprightConfig:
    episode_seconds: float = 60.0
    preparation_seconds: float = 6.0
    deep_start_probability: float = 0.5
    deep_preparation_seconds: float = 60.0
    deep_height_fraction: float = 0.45
    start_height_min: float = 0.20
    start_height_max: float = 0.80
    start_height_bias: float = 2.0
    tip_tolerance_m: float = 0.015
    angle_tolerance_degrees: float = 10.0
    tip_speed_m_s: float = 0.02
    joint_speed_rad_s: float = 0.1
    success_hold_seconds: float = 5.0
    alignment_weight: float = 1.0
    position_weight: float = 1.0
    hold_weight: float = 2.0
    action_change_weight: float = 0.02
    speed_weight: float = 0.02
    learning_rate: float = 0.0001
    rollout_steps: int = 1024
    batch_size: int = 128
    epochs: int = 5
    target_kl: float = 0.02
    evaluation_episodes: int = 8


class UprightEnv(gym.Env):
    def display_state(self) -> tuple[Array, Array, float]:
        return (
            self.robot.data.qpos.copy(),
            self.robot.data.qvel.copy(),
            float(self.robot.data.time),
        )

    def __init__(self, cfg: Config, task: UprightConfig) -> None:
        # The straight model points along +Z; downward gravity makes it upright.
        self.cfg = replace(cfg, gravity=[0.0, 0.0, -9.81], action_mode="velocity")
        self.task = task
        self.robot = Robot(self.cfg)
        self.goal = self.robot.tip
        self.robot.model.site_pos[self.robot.target_id] = self.goal
        self.links = [self.robot.model.body(f"link{i}").id for i in range(2, 31)]
        self.action_space = gym.spaces.Box(-1, 1, (3,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            -np.inf,
            np.inf,
            (self.robot.model.nq + self.robot.model.nv + 16,),
            dtype=np.float32,
        )
        self.previous_action = np.zeros(3)
        self.tip_velocity = np.zeros(3)
        self.elapsed = 0
        self.hold_seconds = 0.0
        self.best_hold_seconds = 0.0
        self.upright_seconds = 0.0

    def observation(self) -> NDArray[np.float32]:
        return np.concatenate(
            (
                self.robot.data.qpos,
                self.robot.data.qvel,
                self.robot.retraction / self.cfg.max_retraction_m,
                self.robot.velocity / self.cfg.max_speed_m_s,
                self.previous_action,
                self.tip_velocity,
                (self.goal - self.robot.tip) / self.task.tip_tolerance_m,
                [self.elapsed * self.cfg.control_dt / self.task.episode_seconds],
            )
        ).astype(np.float32)

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[NDArray[np.float32], dict]:
        super().reset(seed=seed)
        self.robot.reset()
        # Reach varied shapes dynamically; never teleport joints or cable travel.
        target_height = (
            float(options["height_fraction"])
            if options is not None and "height_fraction" in options
            else self.task.start_height_min
            + (self.task.start_height_max - self.task.start_height_min)
            * self.np_random.random() ** self.task.start_height_bias
        )
        if not 0 < target_height < 1:
            raise ValueError("Starting height must be a fraction between zero and one.")
        seconds = self.task.deep_preparation_seconds
        action = np.full(3, -0.5)
        action[self.np_random.integers(3)] = 1.0
        for _ in range(round(seconds / self.cfg.control_dt)):
            previous_tip = self.robot.tip
            self.robot.control(action)
            self.tip_velocity = (self.robot.tip - previous_tip) / self.cfg.control_dt
            if self.robot.tip[2] <= self.goal[2] * target_height:
                break
        actual_height = float(self.robot.tip[2] / self.goal[2])
        if actual_height > target_height + 0.01:
            warnings.warn(
                f"Requested start height {target_height:.1%} was not reached; "
                f"using actual {actual_height:.1%} after {seconds:g}s of cable motion.",
                RuntimeWarning,
                stacklevel=2,
            )
        self.previous_action = action.copy()
        self.elapsed = 0
        self.hold_seconds = self.best_hold_seconds = self.upright_seconds = 0.0
        return self.observation(), {
            "requested_height_fraction": target_height,
            "start_height_reached": actual_height <= target_height + 0.01,
            "start_height_fraction": float(self.robot.tip[2] / self.goal[2]),
        }

    def step(
        self, action: Array
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict]:
        action = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        previous_tip = self.robot.tip
        self.robot.control(action)
        dt = self.cfg.control_dt
        self.tip_velocity = (self.robot.tip - previous_tip) / dt
        # Each body's local Z axis is its segment direction in world coordinates.
        alignment = self.robot.data.xmat[self.links, 8]
        error = float(np.linalg.norm(self.robot.tip - self.goal))
        upright = bool(
            np.min(alignment) >= np.cos(np.deg2rad(self.task.angle_tolerance_degrees))
            and error <= self.task.tip_tolerance_m
            and np.linalg.norm(self.tip_velocity) <= self.task.tip_speed_m_s
            and np.max(np.abs(self.robot.data.qvel)) <= self.task.joint_speed_rad_s
        )
        self.elapsed += 1
        self.hold_seconds = self.hold_seconds + dt if upright else 0.0
        self.best_hold_seconds = max(self.best_hold_seconds, self.hold_seconds)
        self.upright_seconds += dt * upright
        reward = dt * (
            self.task.alignment_weight * float(np.mean(alignment))
            - self.task.position_weight * (error / np.linalg.norm(self.goal)) ** 2
            + self.task.hold_weight * upright
            - self.task.action_change_weight
            * float(np.mean((action - self.previous_action) ** 2))
            - self.task.speed_weight * float(np.mean(self.robot.data.qvel**2))
        )
        self.previous_action = action.copy()
        info = {
            "is_success": self.best_hold_seconds >= self.task.success_hold_seconds,
            "upright_seconds": self.upright_seconds,
            "best_hold_seconds": self.best_hold_seconds,
            "tip_error_m": error,
        }
        # Keep rewarding sustained balance until the full episode ends.
        return (
            self.observation(),
            float(reward),
            False,
            (self.elapsed * dt >= self.task.episode_seconds),
            info,
        )


def train(cfg: Config, task: UprightConfig, run: Path, headless: bool = False) -> None:
    import mujoco.viewer
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv

    deadline = time.monotonic() + cfg.training_seconds
    if run.exists():
        raise FileExistsError(f"Choose a new run directory: {run}")
    # Validate before writing a run or launching subprocesses.
    probe = UprightEnv(cfg, task)
    probe.reset(seed=cfg.seed)
    probe.close()
    run.mkdir(parents=True)
    (run / "robot_config.json").write_text(json.dumps(asdict(cfg), indent=2))
    (run / "upright_config.json").write_text(json.dumps(asdict(task), indent=2))

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
                qpos, qvel, sim_time = self.training_env.env_method(
                    "display_state", indices=0
                )[0]
                with viewer.lock():
                    display_robot.data.qpos[:] = qpos
                    display_robot.data.qvel[:] = qvel
                    display_robot.data.time = sim_time
                    mujoco.mj_forward(display_robot.model, display_robot.data)
                viewer.sync()
                last_display = now
            return time.monotonic() < deadline

    env = make_vec_env(
        lambda: UprightEnv(cfg, task),
        n_envs=cfg.training_envs,
        seed=cfg.seed,
        monitor_dir=str(run),
        monitor_kwargs={
            "info_keywords": ("is_success", "upright_seconds", "best_hold_seconds")
        },
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
            f"Upright recovery: {cfg.training_seconds / 60:g} minute budget, CUDA; +Z upright.",
            flush=True,
        )
        model.learn(
            cfg.training_steps,
            callback=[
                BudgetCallback(),
                CheckpointCallback(
                    max(1, 25000 // cfg.training_envs), str(run), name_prefix="upright"
                ),
            ],
        )
        model.save(run / "policy")
    finally:
        if viewer is not None:
            viewer.close()
        env.close()


def evaluate(cfg: Config, task: UprightConfig, policy_path: Path) -> None:
    from stable_baselines3 import PPO

    env = UprightEnv(cfg, task)
    policy = PPO.load(policy_path, env=env, device="cuda")
    for controller in ("release to zero", "PPO"):
        results = []
        for episode in range(task.evaluation_episodes):
            obs, _ = env.reset(seed=cfg.seed + 10000 + episode)
            done = False
            while not done:
                if controller == "PPO":
                    action, _ = policy.predict(obs, deterministic=True)
                else:
                    action = -env.robot.retraction / (
                        cfg.action_response_seconds * cfg.max_speed_m_s
                    )
                obs, _, _, done, info = env.step(action)
            results.append(info)
        print(
            f"{controller}: mean upright time {np.mean([r['upright_seconds'] for r in results]):.2f}s; "
            f"mean longest hold {np.mean([r['best_hold_seconds'] for r in results]):.2f}s; "
            f"5-second holds {sum(r['is_success'] for r in results)}/{len(results)}"
        )
    env.close()


def demo(cfg: Config, task: UprightConfig, policy_path: Path | None) -> None:
    import mujoco.viewer
    from stable_baselines3 import PPO

    env = UprightEnv(cfg, task)
    policy = PPO.load(policy_path, env=env, device="cuda") if policy_path else None
    obs, start_info = env.reset(seed=cfg.seed)
    print(
        f"Starting tip at {start_info['start_height_fraction']:.1%} of upright height."
    )
    print(
        "+Z is upright. "
        + (
            "Learned policy."
            if policy
            else "Baseline: release cables to zero; no policy."
        )
    )
    with mujoco.viewer.launch_passive(env.robot.model, env.robot.data) as viewer:
        viewer.cam.lookat[:] = [0, 0, 0.16]
        viewer.cam.distance = 0.7
        preview_until = time.monotonic() + 2.0
        while viewer.is_running():
            started = time.monotonic()
            if started < preview_until:
                viewer.sync()
                time.sleep(cfg.control_dt)
                continue
            if policy:
                action, _ = policy.predict(obs, deterministic=True)
            else:
                action = -env.robot.retraction / (
                    cfg.action_response_seconds * cfg.max_speed_m_s
                )
            obs, _, _, done, info = env.step(action)
            viewer.sync()
            if done:
                print(info)
                obs, _ = env.reset()
                preview_until = time.monotonic() + 2.0
            time.sleep(max(0, cfg.control_dt - time.monotonic() + started))
    env.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Separate upright recovery and holding PPO task"
    )
    parser.add_argument("mode", choices=("train", "demo", "evaluate"))
    parser.add_argument("--config", type=Path, default=ROOT / "spiral_config.json")
    parser.add_argument("--run", type=Path, default=ROOT / "runs" / "spiral_upright")
    parser.add_argument("--policy", type=Path)
    parser.add_argument(
        "--headless", action="store_true", help="Train without the live MuJoCo view"
    )
    args = parser.parse_args()
    if args.mode == "evaluate" and args.policy is None:
        parser.error("evaluate requires --policy")
    if args.mode == "train" and args.policy is not None:
        parser.error("Training starts fresh; --policy is for demo/evaluate")
    cfg = load_config(
        args.policy.parent / "robot_config.json" if args.policy else args.config
    )
    task = (
        UprightConfig(
            **json.loads((args.policy.parent / "upright_config.json").read_text())
        )
        if args.policy
        else UprightConfig()
    )
    if args.mode == "train":
        train(cfg, task, args.run, headless=args.headless)
    elif args.mode == "evaluate":
        evaluate(cfg, task, args.policy)
    else:
        demo(cfg, task, args.policy)


if __name__ == "__main__":
    main()

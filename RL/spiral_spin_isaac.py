from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime
import json
from pathlib import Path
import time
import traceback

from spiral import ROOT, load_config
from spiral_spin import SpinConfig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a separate SpiRob spin policy in Isaac Lab / Newton MuJoCo Warp."
    )
    parser.add_argument("mode", choices=("train", "demo"))
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "isaac_spin_robot.json"
    )
    parser.add_argument(
        "--task-config", type=Path, default=ROOT / "configs" / "isaac_spin_task.json"
    )
    parser.add_argument("--run", type=Path)
    parser.add_argument("--policy", type=Path)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one small PPO update and test checkpoint loading.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Number of demo control steps; zero runs until the GUI closes.",
    )
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.mode == "demo" and args.policy is None:
        parser.error("demo requires --policy")
    if args.mode == "train" and args.policy is not None:
        parser.error("Training starts fresh; --policy is for demo")
    if args.mode != "train" and args.smoke:
        parser.error("--smoke is only valid for train")
    if args.steps < 0:
        parser.error("--steps must be nonnegative")
    if args.visualizer is None:
        args.visualizer = ["none"] if args.mode == "train" or args.headless else ["kit"]
    visible = "kit" in args.visualizer
    if args.mode == "demo" and args.steps == 0 and not visible:
        parser.error("A headless demo requires a positive --steps count")
    if not args.device.startswith("cuda"):
        parser.error("This trainer uses the CUDA MuJoCo Warp backend")
    source = args.policy.parent if args.policy else None
    cfg = load_config(source / "robot_config.json" if source else args.config)
    cfg = replace(cfg, gravity=[0.0, 0.0, -9.81], action_mode="velocity")
    task = SpinConfig(
        **json.loads(
            (source / "spin_config.json" if source else args.task_config).read_text()
        )
    )
    if (
        task.orbit_seconds <= 0
        or task.episode_seconds <= 0
        or not 0 < task.height_min <= task.height_fraction <= task.height_max < 1
    ):
        raise ValueError("Invalid spin period or height band")
    if args.smoke:
        cfg = replace(cfg, training_steps=128, training_seconds=300.0)
        task = replace(
            task, episode_seconds=0.08, rollout_steps=32, batch_size=64, epochs=1
        )
    if args.mode == "demo":
        cfg = replace(cfg, training_envs=1)
    if task.rollout_steps * cfg.training_envs < 2 or task.batch_size < 2:
        raise ValueError(
            "PPO requires at least two rollout samples and a batch size of two"
        )
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run = args.run or ROOT / "runs" / (
        f"spiral_spin_isaac_{args.mode}_{stamp}"
        if args.smoke or args.mode == "demo"
        else "spiral_spin_isaac"
    )
    # Reserve a new directory before launching Kit; never replace prior policies.
    run.mkdir(parents=True, exist_ok=False)
    for name, value in (("robot_config.json", cfg), ("spin_config.json", task)):
        with (run / name).open("x", encoding="utf-8") as output:
            json.dump(asdict(value), output, indent=2)

    app = AppLauncher(args).app
    try:
        import numpy as np
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
        from stable_baselines3.common.vec_env import VecMonitor

        from isaac_spin_env import IsaacSpinEnv

        env = IsaacSpinEnv(
            cfg,
            task,
            run,
            args.device,
            visible,
            source / "start_states.npz" if source else None,
        )
        env.seed(cfg.seed)
        monitored = VecMonitor(
            env,
            str(run / "monitor.csv"),
            info_keywords=(
                "is_success",
                "band_fraction",
                "turns",
                "mean_tracking_error_m",
            ),
        )
        try:
            if args.mode == "train":
                if args.smoke:
                    env.reset()
                    before = (
                        env.physics.state.joint_q.numpy()
                        .reshape(env.num_envs, -1)
                        .copy()
                    )
                    pulls = env.physics.pulls.numpy().reshape(env.num_envs, 3).copy()
                    env._reset_indices(np.array([0], dtype=np.int64))
                    np.testing.assert_array_equal(
                        env.physics.state.joint_q.numpy().reshape(env.num_envs, -1)[1:],
                        before[1:],
                    )
                    np.testing.assert_array_equal(
                        env.physics.pulls.numpy().reshape(env.num_envs, 3)[1:],
                        pulls[1:],
                    )
                    print("Independent reset check passed", flush=True)
                    import mujoco

                    observation = env.observation()[0]
                    reference = env.physics.reference
                    reference_data = mujoco.MjData(reference)
                    reference_data.qpos[:] = observation[: reference.nq]
                    reference_data.qvel[:] = observation[
                        reference.nq : reference.nq + reference.nv
                    ]
                    mujoco.mj_forward(reference, reference_data)
                    expected_tip = np.mean(
                        [
                            reference_data.site_xpos[reference.site(f"s30_{guide}").id]
                            for guide in (2, 4, 6)
                        ],
                        axis=0,
                    )
                    np.testing.assert_allclose(
                        env.physics.tip()[0], expected_tip, atol=1e-5
                    )
                    for _ in range(round(task.episode_seconds / cfg.control_dt)):
                        observations, _, done, infos = env.step(
                            np.zeros((env.num_envs, 3))
                        )
                    if not np.all(done) or not all(
                        info["TimeLimit.truncated"] for info in infos
                    ):
                        raise RuntimeError("Episode timeout did not trigger correctly")
                    for info in infos:
                        if info["terminal_observation"].shape != observations[0].shape:
                            raise RuntimeError(
                                "Missing terminal observation for PPO bootstrapping"
                            )
                    print(
                        "Observation layout and timeout/reset checks passed", flush=True
                    )
                model = PPO(
                    "MlpPolicy",
                    monitored,
                    device=args.device,
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
                deadline = time.monotonic() + cfg.training_seconds

                class BudgetCallback(BaseCallback):
                    def _on_step(self) -> bool:
                        return app.is_running() and time.monotonic() < deadline

                print(
                    f"Isaac spin training: {cfg.training_envs} GPU environments, {cfg.training_seconds / 60:g} minute budget; output={run}",
                    flush=True,
                )
                model.learn(
                    cfg.training_steps,
                    callback=[
                        BudgetCallback(),
                        CheckpointCallback(
                            max(1, 25000 // cfg.training_envs),
                            str(run),
                            name_prefix="spin_isaac",
                        ),
                    ],
                )
                model.save(run / "policy")
                if args.smoke:
                    loaded = PPO.load(
                        run / "policy.zip", env=monitored, device=args.device
                    )
                    observation = monitored.reset()
                    action, _ = loaded.predict(observation, deterministic=True)
                    observation, reward, _, _ = monitored.step(action)
                    if not np.all(np.isfinite(observation)) or not np.all(
                        np.isfinite(reward)
                    ):
                        raise RuntimeError(
                            "Reloaded policy produced non-finite results"
                        )
                    if model._n_updates < 1:
                        raise RuntimeError("Smoke test did not complete a PPO update")
                    print(
                        f"ISAAC_SPIN_SMOKE_OK updates={model._n_updates} timesteps={model.num_timesteps}",
                        flush=True,
                    )
                print(f"Saved policy: {run / 'policy.zip'}", flush=True)
            else:
                model = PPO.load(args.policy, env=monitored, device=args.device)
                observation = monitored.reset()
                count = 0
                while app.is_running() and (args.steps == 0 or count < args.steps):
                    started = time.monotonic()
                    action, _ = model.predict(observation, deterministic=True)
                    observation, _, done, infos = monitored.step(action)
                    count += 1
                    if done[0]:
                        print(
                            {
                                key: value
                                for key, value in infos[0].items()
                                if key != "terminal_observation"
                            },
                            flush=True,
                        )
                    if visible:
                        time.sleep(max(0, cfg.control_dt - time.monotonic() + started))
                print(f"ISAAC_SPIN_DEMO_DONE steps={count}", flush=True)
        finally:
            monitored.close()
    except Exception:
        traceback.print_exc()
        app.close(exit_code=1)
        raise
    else:
        app.close()


if __name__ == "__main__":
    main()

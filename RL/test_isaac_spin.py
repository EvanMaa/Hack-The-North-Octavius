from dataclasses import replace
import unittest

import mujoco
import numpy as np
import warp as wp

from isaac_spin_env import spin_reward
from isaac_spin_physics import cable_step
from spiral import ROOT, Robot, load_config
from spiral_spin import SpinConfig, SpinEnv


class IsaacSpinTest(unittest.TestCase):
    def test_residual_zero_preserves_cable_sweep(self) -> None:
        cfg = load_config(ROOT / "configs" / "isaac_spin_robot.json")
        direct = SpinEnv(cfg, SpinConfig(dense_tracking=True))
        residual = SpinEnv(cfg, SpinConfig(dense_tracking=True, residual_scale=0.15))
        for env in (direct, residual):
            env.phase = 1.2
            env.radius = 0.14
        for _ in range(20):
            direct.step(direct.baseline_action())
            residual.step(np.zeros(3))
        np.testing.assert_allclose(
            direct.robot.data.qpos, residual.robot.data.qpos, atol=1e-12
        )
        np.testing.assert_allclose(
            direct.robot.retraction, residual.robot.retraction, atol=1e-12
        )

    def test_dense_reward_prefers_tracking_to_hovering(self) -> None:
        cfg = load_config(ROOT / "configs" / "isaac_spin_robot.json")
        task = SpinConfig(dense_tracking=True)
        radius = 0.14
        omega = 2 * np.pi / task.orbit_seconds
        phase = np.array([1.0, 1.0])
        goal = np.array([radius * np.cos(1), radius * np.sin(1), 0.15])
        tips = np.array([goal, [radius, 0, 0.15]])
        before = tips.copy()
        before[0, :2] = radius * np.array(
            [np.cos(1 - omega * cfg.control_dt), np.sin(1 - omega * cfg.control_dt)]
        )
        reward, _, _, _ = spin_reward(
            cfg,
            task,
            tips,
            before,
            np.tile(goal, (2, 1)),
            phase,
            np.ones(2),
            np.full(2, radius),
            0.15 / 0.45,
            np.zeros((2, 3)),
            np.zeros((2, 3)),
        )
        self.assertGreater(reward[0], reward[1])

    def test_dense_vector_reward_matches_residual_task(self) -> None:
        cfg = load_config(ROOT / "configs" / "isaac_spin_robot.json")
        task = SpinConfig(dense_tracking=True, residual_scale=0.15)
        env = SpinEnv(cfg, task)
        env.radius = 0.14
        env.phase = 1.0
        before = env.robot.tip
        previous_action = env.previous_action.copy()
        _, expected, _, _, _ = env.step(np.array([0.2, -0.3, 0.1]))
        reward, _, _, _ = spin_reward(
            cfg,
            task,
            env.robot.tip[None],
            before[None],
            env.goal[None],
            np.array([env.phase]),
            np.array([env.direction]),
            np.array([env.radius]),
            float(env.upright_goal[2]),
            env.previous_action[None],
            previous_action[None],
        )
        self.assertAlmostEqual(float(reward[0]), expected, places=10)

    def test_vector_reward_matches_original_task(self) -> None:
        cfg = load_config(ROOT / "configs" / "isaac_spin_robot.json")
        task = SpinConfig()
        env = SpinEnv(cfg, task)
        for direction in (-1.0, 1.0):
            env.direction = direction
            env.phase = 3.14
            env.radius = 0.1
            env.turns = 0
            before = env.robot.tip
            previous_action = env.previous_action.copy()
            action = np.array([0.1, -0.2, 0.3])
            _, expected_reward, _, _, info = env.step(action)
            reward, _, turns, error = spin_reward(
                env.cfg,
                task,
                env.robot.tip[None],
                before[None],
                env.goal[None],
                np.array([env.phase]),
                np.array([direction]),
                np.array([env.radius]),
                float(env.upright_goal[2]),
                action[None],
                previous_action[None],
            )
            self.assertAlmostEqual(float(reward[0]), expected_reward, places=10)
            self.assertAlmostEqual(float(turns[0]), info["turns"], places=10)
            self.assertAlmostEqual(
                float(error[0]), np.linalg.norm(env.robot.tip - env.goal), places=10
            )
        env.close()

    def test_gpu_cables_match_original_velocity_controller(self) -> None:
        cfg = load_config(ROOT / "configs" / "isaac_spin_robot.json")
        cfg = replace(cfg, control_dt=cfg.physics_dt)
        robot = Robot(cfg)
        model, data = robot.model, robot.data
        routes = np.array(
            [
                [
                    model.site(f"s{link}_{guide}").id
                    for link in range(1, 31)
                    for guide in pair
                ]
                for pair in [(1, 2), (3, 4), (5, 6)]
            ]
        )
        for action in (np.array([1.0, -0.5, -0.5]), np.array([-1.0, 1.0, 0.0])):
            robot.retraction[:] = [0.005, 0.0, -0.01]
            robot.velocity[:] = [0.002, -0.001, 0.0]
            mujoco.mj_forward(model, data)
            poses = np.column_stack((data.xpos, np.roll(data.xquat, -1, axis=1)))
            velocities = np.zeros((model.nbody, 6))
            for body in range(model.nbody):
                linear, angular = np.zeros((3, model.nv)), np.zeros((3, model.nv))
                mujoco.mj_jacBodyCom(model, data, linear, angular, body)
                velocities[body] = np.concatenate(
                    (linear @ data.qvel, angular @ data.qvel)
                )
            device = "cuda:0"
            pulls: wp.array = wp.array(robot.retraction, dtype=float, device=device)
            speeds: wp.array = wp.array(robot.velocity, dtype=float, device=device)
            controls = wp.zeros(3, dtype=float, device=device)
            inputs = [
                wp.array(poses, dtype=wp.transform, device=device),
                wp.array(velocities, dtype=wp.spatial_vector, device=device),
                wp.array(model.body_ipos, dtype=wp.vec3, device=device),
                wp.array(model.site_bodyid[routes], dtype=wp.int32, device=device),
                wp.array(model.site_pos[routes], dtype=wp.vec3, device=device),
                wp.array(robot.rest_lengths, dtype=float, device=device),
                wp.array(action, dtype=float, device=device),
                wp.array(
                    [
                        cfg.physics_dt,
                        cfg.max_speed_m_s,
                        cfg.max_acceleration_m_s2,
                        cfg.max_retraction_m,
                        cfg.max_payout_m,
                        cfg.cable_stiffness_n_m,
                        cfg.cable_damping_n_s_m,
                        cfg.max_tension_n,
                    ],
                    dtype=float,
                    device=device,
                ),
                pulls,
                speeds,
                controls,
            ]
            wp.launch(cable_step, dim=3, inputs=inputs, device=device)
            robot.control(action)
            np.testing.assert_allclose(pulls.numpy(), robot.retraction, atol=1e-8)
            np.testing.assert_allclose(speeds.numpy(), robot.velocity, atol=1e-8)
            np.testing.assert_allclose(controls.numpy(), data.ctrl, atol=0.0002)


if __name__ == "__main__":
    unittest.main()

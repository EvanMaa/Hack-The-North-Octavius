from dataclasses import replace
import unittest

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from spiral import (
    Config,
    ROOT,
    Robot,
    SpiralEnv,
    demo_target,
    load_config,
    training_patterns,
)


class SpiralTest(unittest.TestCase):
    def test_reachable_target_requires_settled_arrival(self) -> None:
        cfg = replace(self.cfg, advance_on_reach=True)
        robot = Robot(cfg)
        pull = np.array([0.01, 0.0, 0.0])
        goal = robot.equilibrium(pull)
        env = SpiralEnv(cfg, np.array([goal]))
        env.reset(seed=7)
        command = env.robot.position_action(pull)
        for _ in range(env.waypoint_steps):
            _, _, terminated, truncated, info = env.step(command)
            if terminated or truncated:
                break
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["is_success"])
        self.assertLessEqual(info["tip_error_m"], cfg.reach_tolerance_m)
        self.assertLessEqual(np.linalg.norm(env.tip_velocity), cfg.reach_speed_m_s)
        self.assertGreaterEqual(
            env.reach_steps * cfg.control_dt, cfg.reach_hold_seconds
        )
        env.close()

    def setUp(self) -> None:
        self.cfg = load_config(ROOT / "spiral_config.json")

    def test_settled_sweep_advances_without_reset(self) -> None:
        cfg = replace(self.cfg, advance_on_reach=True)
        robot = Robot(cfg)
        goal = robot.equilibrium(np.zeros(3))
        env = SpiralEnv(cfg, np.array([goal]), [[0, 0]])
        env.reset(seed=7)
        command = env.robot.position_action(np.zeros(3))
        arrivals = 0
        for _ in range(2 * env.waypoint_steps):
            before = env.robot.data.time
            _, _, terminated, truncated, info = env.step(command)
            self.assertGreater(env.robot.data.time, before)
            if info["waypoint_reached"]:
                arrivals += 1
                if arrivals == 1:
                    self.assertEqual(env.waypoint_index, 1)
                    self.assertEqual(env.reach_steps, 0)
                    self.assertFalse(terminated)
            if terminated or truncated:
                break
        self.assertEqual(arrivals, 2)
        self.assertTrue(info["is_success"])
        self.assertFalse(truncated)
        env.close()

    def test_stroke_speed_acceleration_and_tension_limits(self) -> None:
        robot = Robot(self.cfg)
        for sign in (1.0, -1.0, 1.0):
            for _ in range(600):
                previous_velocity = robot.velocity.copy()
                robot.advance(np.full(3, sign))
                self.assertTrue(np.all(robot.retraction >= -self.cfg.max_payout_m))
                self.assertTrue(np.all(robot.retraction <= self.cfg.max_retraction_m))
                self.assertLessEqual(
                    np.max(np.abs(robot.velocity)), self.cfg.max_speed_m_s + 1e-12
                )
                self.assertLessEqual(
                    np.max(np.abs(robot.velocity - previous_velocity))
                    / self.cfg.control_dt,
                    self.cfg.max_acceleration_m_s2 + 1e-10,
                )
                self.assertTrue(np.all(robot.data.ctrl <= 0))
                self.assertTrue(np.all(robot.data.ctrl >= -self.cfg.max_tension_n))

    def test_forward_mapping_and_invalid_inputs(self) -> None:
        robot = Robot(self.cfg)
        resting = robot.equilibrium(np.zeros(3))
        pulled = robot.equilibrium(np.array([0.006, 0, 0]))
        self.assertGreater(np.linalg.norm(pulled - resting), 0.005)
        np.testing.assert_allclose(
            pulled, robot.equilibrium(np.array([0.006, 0, 0])), atol=1e-10
        )
        for invalid in (
            np.array([-self.cfg.max_payout_m - 0.001, 0, 0]),
            np.array([np.nan, 0, 0]),
        ):
            with self.assertRaises(ValueError):
                robot.equilibrium(invalid)

    def test_position_action_moves_to_target_and_stops(self) -> None:
        robot = Robot(self.cfg)
        self.assertEqual(self.cfg.action_mode, "position")
        target = np.array([0.006, -0.003, -0.003])
        command = robot.position_action(target)
        initial_tip = robot.tip
        for _ in range(400):
            robot.control(command)
        np.testing.assert_allclose(robot.retraction, target, atol=1e-6)
        self.assertLess(np.linalg.norm(robot.velocity), 1e-5)
        self.assertGreater(np.linalg.norm(robot.tip - initial_tip), 0.005)

    def test_legacy_config_keeps_velocity_actions(self) -> None:
        values = vars(self.cfg).copy()
        del values["action_mode"]
        legacy = Config(**values)
        self.assertEqual(legacy.action_mode, "velocity")
        robot = Robot(legacy)
        other = Robot(legacy)
        for _ in range(10):
            robot.control(np.array([-1.0, 0, 1.0]))
            other.advance(np.array([-1.0, 0, 1.0]))
        np.testing.assert_array_equal(robot.retraction, other.retraction)
        np.testing.assert_array_equal(robot.data.qpos, other.data.qpos)

    def test_curl_demo_reverses_tip_direction(self) -> None:
        cfg = self.cfg
        robot = Robot(cfg)
        for step in range(round(17 / cfg.control_dt)):
            desired = demo_target(cfg, step * cfg.control_dt)
            robot.advance(
                np.clip((desired - robot.retraction) / (0.5 * cfg.max_speed_m_s), -1, 1)
            )
        self.assertLess(robot.data.body("link30").xmat.reshape(3, 3)[2, 2], -0.2)
        self.assertGreater(robot.retraction[0], 0.045)
        self.assertTrue(np.all(robot.retraction[1:] < -0.025))
        self.assertTrue(np.all(-robot.data.ctrl <= cfg.max_tension_n))

    def test_environment_and_short_training(self) -> None:
        robot = Robot(self.cfg)
        goals = np.array([robot.equilibrium(np.array([0.006, 0, 0]))])
        env = SpiralEnv(replace(self.cfg, episode_seconds=0.2), goals)
        check_env(env)
        first, _ = env.reset(seed=7)
        env.step(np.ones(3))
        second, _ = env.reset(seed=7)
        np.testing.assert_array_equal(first, second)
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=32,
            batch_size=16,
            n_epochs=1,
            seed=7,
            device="cpu",
        )
        model.learn(64)
        action, _ = model.predict(second, deterministic=True)
        observation, reward, _, _, _ = env.step(action)
        self.assertTrue(np.all(np.isfinite(observation)))
        self.assertTrue(np.isfinite(reward))
        env.close()

    def test_training_pattern_coverage(self) -> None:
        pulls, sequences = training_patterns(self.cfg)
        np.testing.assert_array_equal(pulls[0], np.zeros(3))
        self.assertTrue(np.all(pulls >= -self.cfg.max_payout_m))
        self.assertTrue(np.all(pulls <= self.cfg.max_retraction_m))
        for depth in self.cfg.bend_depths:
            for mask in range(1, 8):
                expected = depth * np.array(
                    [
                        self.cfg.max_retraction_m
                        if mask & (1 << cable)
                        else -self.cfg.max_payout_m
                        for cable in range(3)
                    ]
                )
                self.assertTrue(np.any(np.all(np.isclose(pulls, expected), axis=1)))
        rings = [sequence for sequence in sequences if len(sequence) > 4]
        self.assertEqual(len(rings), len(self.cfg.bend_depths))
        for sequence in rings:
            n = self.cfg.sweep_directions
            self.assertEqual(len(set(sequence[:n])), n)
            self.assertEqual(sequence[n], sequence[0])
            self.assertEqual(sequence[n : 2 * n + 1], sequence[: n + 1][::-1])
        self.assertTrue(all(sequence[-1] == 0 for sequence in sequences))
        self.assertEqual(
            set(range(len(pulls))),
            {index for sequence in sequences for index in sequence},
        )
        again, repeated = training_patterns(self.cfg)
        np.testing.assert_array_equal(pulls, again)
        self.assertEqual(sequences, repeated)

    def test_sequence_changes_without_reset_and_round_coverage(self) -> None:
        cfg = replace(
            self.cfg,
            advance_on_reach=False,
            max_speed_m_s=1.0,
            max_acceleration_m_s2=100.0,
            waypoint_seconds=0.08,
        )
        goals = np.array([[0, 0, 0.33], [0, 0.01, 0.33], [0.01, 0, 0.33]])
        sequences = [[1, 2, 1, 0], [2, 1, 2, 0]]
        env = SpiralEnv(cfg, goals, sequences)
        check_env(env)
        env.reset(seed=7)
        first_index = env.sequence_index
        first_goal = env.goal.copy()
        for step in range(env.waypoint_steps * 4):
            previous_time = env.robot.data.time
            _, _, _, truncated, info = env.step(np.zeros(3))
            self.assertGreater(env.robot.data.time, previous_time)
            self.assertEqual(truncated, step == env.waypoint_steps * 4 - 1)
            if step == env.waypoint_steps - 1:
                self.assertEqual(info["waypoint_index"], 0)
                self.assertFalse(np.array_equal(env.goal, first_goal))
                self.assertEqual(env.elapsed, env.waypoint_steps)
        np.testing.assert_array_equal(env.goal, goals[0])
        env.reset()
        self.assertNotEqual(env.sequence_index, first_index)
        env.reset(seed=7)
        self.assertEqual(env.sequence_index, first_index)
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=32,
            batch_size=16,
            n_epochs=1,
            seed=7,
            device="cpu",
        )
        model.learn(64)
        env.close()


if __name__ == "__main__":
    unittest.main()

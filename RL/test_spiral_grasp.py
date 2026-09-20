from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from spiral import ROOT, load_config
from spiral_grasp import (
    Contacts,
    GraspEnv,
    centroid_surface_distance,
    load_grasp_config,
    scripted_action,
    secure_grasp,
)


class GraspTest(unittest.TestCase):
    def test_fixed_placement_is_independent_of_seed(self) -> None:
        cfg = replace(
            self.cfg,
            fixed_object_shape="cylinder",
            fixed_object_position_m=[0.0, 0.08, 0.21],
        )
        env = GraspEnv(self.robot_cfg, cfg)
        first, _ = env.reset(seed=1)
        second, info = env.reset(seed=200)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_allclose(info["spawn_m"], [0.0, 0.08, 0.21])
        self.assertEqual(info["shape"], "cylinder")
        env.close()

    def test_centroid_approach_stops_at_surface(self) -> None:
        for shape in ("box", "cylinder"):
            size = np.array([0.02, 0.04, 0.06])
            far = centroid_surface_distance(np.array([0.1, 0, 0]), shape, size)
            near = centroid_surface_distance(np.array([0.03, 0, 0]), shape, size)
            self.assertGreater(far, near)
            self.assertAlmostEqual(near, 0.01)
            self.assertAlmostEqual(
                centroid_surface_distance(np.array([0.02, 0, 0]), shape, size), 0
            )
            self.assertGreater(centroid_surface_distance(np.zeros(3), shape, size), 0)
        self.assertAlmostEqual(
            centroid_surface_distance(np.array([0, 0, 0.05]), "cylinder", size),
            0.01,
        )
        self.assertAlmostEqual(
            centroid_surface_distance(np.array([0.04, 0.08, 0]), "box", size),
            np.linalg.norm([0.02, 0.04]),
        )

    def setUp(self) -> None:
        self.cfg = load_grasp_config(ROOT / "spiral_grasp.json")
        self.robot_cfg = load_config(ROOT / self.cfg.robot_config)

    def test_approach_reward_hands_off_to_contact(self) -> None:
        env = GraspEnv(self.robot_cfg, self.cfg)
        for links in (0, 1, self.cfg.min_contact_links):
            rewards = []
            for weight in (0.0, 1.0):
                env.cfg = replace(self.cfg, approach_reward=weight)
                env.reset(seed=19, options={"shape": "box"})
                with patch(
                    "spiral_grasp.grasp_contacts",
                    return_value=Contacts(links, 0, 1, False),
                ):
                    _, reward, _, _, info = env.step(
                        env.robot.position_action(np.zeros(3))
                    )
                rewards.append(reward)
            expected = (
                -env.robot_cfg.control_dt
                * (1 - links / self.cfg.min_contact_links)
                * info["approach_distance_m"]
                / self.cfg.distance_scale_m
            )
            self.assertGreater(info["approach_distance_m"], 0)
            self.assertAlmostEqual(rewards[1] - rewards[0], expected)
        env.close()

    def test_randomized_placement_reproducibility_and_shape_balance(self) -> None:
        env = GraspEnv(self.robot_cfg, self.cfg)
        np.testing.assert_array_equal(
            env.robot.model.geom_margin[env.robot.robot_geoms], 0
        )
        self.assertEqual(env.robot.model.geom_priority[env.robot.object_geom], 1)
        first, first_info = env.reset(seed=7)
        second, second_info = env.reset(seed=7)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first_info["shape"], second_info["shape"])
        shapes = [second_info["shape"]]
        positions = [env.spawn.copy()]
        for _ in range(15):
            _, info = env.reset()
            shapes.append(info["shape"])
            positions.append(env.spawn.copy())
            self.assertGreater(env.robot.object_distance(env.robot.robot_geoms), 0)
            self.assertTrue(env.observation_space.contains(env.observation()))
        self.assertEqual(shapes.count("cylinder"), 8)
        self.assertEqual(shapes.count("box"), 8)
        self.assertGreater(np.ptp(np.array(positions)[:, 2]), 0.02)
        self.assertGreater(np.ptp(np.array(positions)[:, 0]), 0.08)
        env.close()

    def test_fixture_release_and_drop_are_not_success(self) -> None:
        cfg = replace(
            self.cfg, presentation_seconds=0.1, hold_seconds=0.1, episode_seconds=1.0
        )
        env = GraspEnv(self.robot_cfg, cfg)
        for shape in ("cylinder", "box"):
            env.reset(seed=7, options={"shape": shape})
            for _ in range(5):
                _, _, _, _, info = env.step(np.zeros(3))
                self.assertFalse(info["released"])
                self.assertFalse(info["is_success"])
                self.assertLess(info["object_drift_m"], 0.001)
            done = False
            while not done:
                _, _, terminated, truncated, info = env.step(np.zeros(3))
                done = terminated or truncated
            self.assertTrue(info["released"])
            self.assertTrue(info["dropped"])
            self.assertFalse(info["is_success"])
            self.assertEqual(info["hold_seconds"], 0)
        env.close()

    def test_success_requires_unsupported_opposing_stable_contacts(self) -> None:
        contacts = Contacts(3, 140.0, 1.0, False)
        self.assertTrue(secure_grasp(contacts, True, 0, 0, 0, self.cfg))
        self.assertFalse(secure_grasp(contacts, False, 0, 0, 0, self.cfg))
        for bad in (
            replace(contacts, links=1),
            replace(contacts, span_deg=30),
            replace(contacts, floor=True),
            replace(contacts, penetration_m=self.cfg.max_penetration_m * 2),
        ):
            self.assertFalse(secure_grasp(bad, True, 0, 0, 0, self.cfg))
        for drift, speed, spin in ((0.1, 0, 0), (0, 0.2, 0), (0, 0, 2)):
            self.assertFalse(secure_grasp(contacts, True, drift, speed, spin, self.cfg))

    def test_both_shapes_generate_robot_contact(self) -> None:
        env = GraspEnv(self.robot_cfg, self.cfg)
        for shape in ("cylinder", "box"):
            env.reset(seed=19, options={"shape": shape})
            max_links = 0
            for _ in range(round(4 / self.robot_cfg.control_dt)):
                _, reward, _, _, info = env.step(scripted_action(env))
                self.assertTrue(np.isfinite(reward))
                max_links = max(max_links, info["contact_links"])
            # Hard contacts need not involve the extra links that previously
            # contacted only because the meshes penetrated the object.
            self.assertGreaterEqual(max_links, 1)
            self.assertFalse(info["is_success"])
        env.close()

    def test_gym_training_and_policy_round_trip(self) -> None:
        cfg = replace(
            self.cfg, presentation_seconds=0.1, hold_seconds=0.1, episode_seconds=0.6
        )
        env = GraspEnv(self.robot_cfg, cfg)
        check_env(env)
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
        observation, _ = env.reset(seed=19)
        before, _ = model.predict(observation, deterministic=True)
        with TemporaryDirectory(prefix="spiral_grasp_test_", dir=ROOT) as directory:
            path = Path(directory) / "policy.zip"
            model.save(path)
            restored = PPO.load(path, env=env, device="cpu")
            after, _ = restored.predict(observation, deterministic=True)
            np.testing.assert_array_equal(before, after)
            observation, reward, _, _, _ = env.step(after)
            self.assertTrue(env.observation_space.contains(observation))
            self.assertTrue(np.isfinite(reward))
        env.close()

    def test_deep_box_contact_remains_stable_when_fixture_releases(self) -> None:
        env = GraspEnv(self.robot_cfg, self.cfg)
        env.reset(seed=7, options={"shape": "box"})
        env.spawn = np.array([0, 0.085, 0.25])
        env.robot.place_object(env.spawn, np.array([np.sqrt(0.5), 0, -np.sqrt(0.5), 0]))
        done = False
        peak_span = 0.0
        while not done:
            observation, reward, terminated, truncated, info = env.step(
                scripted_action(env)
            )
            self.assertTrue(np.all(np.isfinite(observation)))
            self.assertTrue(np.isfinite(reward))
            self.assertLessEqual(info["penetration_m"], self.cfg.max_penetration_m)
            peak_span = max(peak_span, info["contact_span_deg"])
            done = terminated or truncated
        self.assertTrue(info["released"])
        self.assertGreater(peak_span, 90)
        self.assertFalse(np.any(env.robot.data.warning.number))
        env.close()

    def test_more_segments_rewarded_but_penetration_is_failure(self) -> None:
        env = GraspEnv(self.robot_cfg, self.cfg)
        rewards = []
        for contacts in (
            Contacts(3, 140, 1, False),
            Contacts(8, 140, 1, False),
            Contacts(8, 140, 1, False, self.cfg.max_penetration_m * 2),
        ):
            env.reset(seed=19, options={"shape": "box"})
            with patch("spiral_grasp.grasp_contacts", return_value=contacts):
                _, reward, terminated, _, info = env.step(
                    env.robot.position_action(np.zeros(3))
                )
            rewards.append(reward)
        self.assertGreater(rewards[1], rewards[0])
        self.assertLess(rewards[2], rewards[0])
        self.assertTrue(terminated)
        self.assertTrue(info["collision_failure"])
        self.assertFalse(info["is_success"])
        env.close()

    def test_reported_box_placements_do_not_phase_through(self) -> None:
        env = GraspEnv(self.robot_cfg, self.cfg)
        for seed in (19, 7):
            env.reset(seed=seed, options={"shape": "box"})
            done = False
            while not done:
                _, _, terminated, truncated, info = env.step(scripted_action(env))
                self.assertLessEqual(info["penetration_m"], self.cfg.max_penetration_m)
                done = terminated or truncated
            self.assertTrue(info["released"])
            self.assertFalse(info["is_success"])
        env.close()


if __name__ == "__main__":
    unittest.main()

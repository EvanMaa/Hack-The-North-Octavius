from dataclasses import replace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from isaac_cables import CableController, cable_kinematics
from spiral import ROOT, Robot, load_config


class IsaacCableTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = replace(
            load_config(ROOT / "spiral_config.json"), action_mode="position"
        )
        self.robot = Robot(self.cfg)

    def test_lengths_and_rates_match_mujoco_in_bent_poses(self) -> None:
        model, data = self.robot.model, self.robot.data
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
        rng = np.random.default_rng(7)
        for _ in range(5):
            mujoco.mj_resetData(model, data)
            mujoco.mj_integratePos(model, data.qpos, rng.normal(0, 0.1, model.nv), 1.0)
            data.qvel[:] = rng.normal(0, 0.1, model.nv)
            mujoco.mj_forward(model, data)
            poses = np.column_stack((data.xpos, np.roll(data.xquat, -1, axis=1)))
            velocities = np.zeros((model.nbody, 6))
            for body in range(model.nbody):
                linear = np.zeros((3, model.nv))
                angular = np.zeros((3, model.nv))
                mujoco.mj_jacBodyCom(model, data, linear, angular, body)
                velocities[body, :3] = linear @ data.qvel
                velocities[body, 3:] = angular @ data.qvel
            lengths, rates = cable_kinematics(
                poses,
                velocities,
                model.body_ipos,
                model.site_bodyid[routes],
                model.site_pos[routes],
            )
            np.testing.assert_allclose(lengths, data.ten_length, atol=1e-12)
            np.testing.assert_allclose(rates, data.ten_velocity, atol=1e-12)

    def test_motor_motion_matches_original_controller(self) -> None:
        controller = CableController(self.cfg, self.robot.rest_lengths)
        for target in [np.array([0.3, 0.0, -0.25]), np.array([-0.25, 0.3, 0.0])]:
            controller.set_target(target)
            with patch.object(self.robot, "physics_step"):
                for _ in range(100):
                    self.robot.control(self.robot.position_action(target))
                    for _ in range(round(self.cfg.control_dt / self.cfg.physics_dt)):
                        controller.step(controller.rest_lengths, np.zeros(3))
                    np.testing.assert_allclose(
                        controller.retraction, self.robot.retraction, atol=1e-12
                    )
                    np.testing.assert_allclose(
                        controller.velocity, self.robot.velocity, atol=1e-12
                    )

    def test_slack_and_tension_limit(self) -> None:
        controller = CableController(self.cfg, self.robot.rest_lengths)
        command = controller.step(controller.rest_lengths - 0.01, np.ones(3))
        np.testing.assert_array_equal(command, 0)
        command = controller.step(controller.rest_lengths + 10, np.zeros(3))
        np.testing.assert_array_equal(command, -self.cfg.max_tension_n)

    def test_invalid_pulls(self) -> None:
        controller = CableController(self.cfg, self.robot.rest_lengths)
        for target in ([0, 0], [float("nan"), 0, 0], [0.31, 0, 0], [-0.26, 0, 0]):
            with self.assertRaises(ValueError):
                controller.set_target(np.array(target))


if __name__ == "__main__":
    unittest.main()

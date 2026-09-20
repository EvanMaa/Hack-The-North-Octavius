from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import xml.etree.ElementTree as ET

import gymnasium as gym
import mujoco
import numpy as np
from numpy.typing import NDArray

from spiral import (
    Array,
    CHECKPOINT_STEPS,
    Config,
    PPO_BATCH_SIZE,
    PPO_DEVICE,
    PPO_GAMMA,
    PPO_GAE_LAMBDA,
    PPO_LEARNING_RATE,
    PPO_ROLLOUT_STEPS,
    PPO_SDE_SAMPLE_FREQ,
    PPO_TARGET_KL,
    ROOT,
    Robot,
    load_config,
)

RUN = ROOT / "runs" / "spiral_grasp"
SHAPES = ("cylinder", "box")
FIRST_GRASP_LINK = 5
FLOOR_X_M = -0.15
PLACEMENT_ATTEMPTS = 100
PLACEMENT_CLEARANCE_M = 0.002
DROP_DISTANCE_M = 0.15
DEMO_SEED = 19


@dataclass
class GraspConfig:
    robot_config: str
    object_mass_kg: float
    cylinder_radius_m: float
    cylinder_length_m: float
    box_size_m: list[float]
    radial_range_m: list[float]
    axial_range_m: list[float]
    friction: float
    presentation_seconds: float
    episode_seconds: float
    hold_seconds: float
    max_drift_m: float
    max_hold_speed_m_s: float
    max_hold_spin_rad_s: float
    min_contact_force_n: float
    min_contact_links: int
    min_contact_span_deg: float
    distance_scale_m: float
    contact_reward: float
    span_reward: float
    hold_reward: float
    success_reward: float
    drop_penalty: float
    training_steps: int
    evaluation_episodes: int
    seed: int
    physics_dt: float = 0.0005
    contact_time_constant_s: float = 0.001
    max_penetration_m: float = 0.001
    contact_saturation_links: int = 10
    collision_margin_m: float = 0.001
    approach_reward: float = 1.0
    training_seconds: float = 1800.0
    training_envs: int = 4
    fixed_object_position_m: list[float] | None = None
    fixed_object_shape: str | None = None

    def __post_init__(self) -> None:
        if (
            self.fixed_object_shape is not None
            and self.fixed_object_shape not in SHAPES
        ):
            raise ValueError(f"fixed_object_shape must be one of {SHAPES}")
        if self.fixed_object_position_m is not None:
            if len(self.fixed_object_position_m) != 3 or not np.all(
                np.isfinite(self.fixed_object_position_m)
            ):
                raise ValueError(
                    "fixed_object_position_m must contain three finite coordinates"
                )
        for name in (
            "object_mass_kg",
            "cylinder_radius_m",
            "cylinder_length_m",
            "friction",
            "presentation_seconds",
            "episode_seconds",
            "hold_seconds",
            "max_drift_m",
            "max_hold_speed_m_s",
            "max_hold_spin_rad_s",
            "min_contact_force_n",
            "distance_scale_m",
            "physics_dt",
            "contact_time_constant_s",
            "max_penetration_m",
            "collision_margin_m",
            "training_seconds",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in (
            "approach_reward",
            "contact_reward",
            "span_reward",
            "hold_reward",
            "success_reward",
            "drop_penalty",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if (
            len(self.box_size_m) != 3
            or not np.all(np.isfinite(self.box_size_m))
            or min(self.box_size_m) <= 0
        ):
            raise ValueError("box_size_m must have three positive full dimensions")
        for name in ("radial_range_m", "axial_range_m"):
            bounds = getattr(self, name)
            if (
                len(bounds) != 2
                or not np.all(np.isfinite(bounds))
                or not 0 < bounds[0] <= bounds[1]
            ):
                raise ValueError(f"{name} must be an increasing positive interval")
        if self.episode_seconds <= self.presentation_seconds + self.hold_seconds:
            raise ValueError(
                "episode_seconds must allow a hold after the fixture releases"
            )
        if self.contact_time_constant_s < 2 * self.physics_dt:
            raise ValueError(
                "contact_time_constant_s must be at least twice physics_dt"
            )
        if not 2 <= self.min_contact_links <= 30 - FIRST_GRASP_LINK + 1:
            raise ValueError("min_contact_links is outside the available grasp links")
        if (
            not self.min_contact_links
            <= self.contact_saturation_links
            <= 30 - FIRST_GRASP_LINK + 1
        ):
            raise ValueError(
                "contact_saturation_links must be between min_contact_links and the available grasp links"
            )
        if not 90 <= self.min_contact_span_deg <= 180:
            raise ValueError("min_contact_span_deg must be between 90 and 180")
        if self.training_steps < 1 or self.evaluation_episodes < 2:
            raise ValueError(
                "training_steps >= 1 and evaluation_episodes >= 2 required"
            )
        if self.training_envs < 1:
            raise ValueError("training_envs must be positive")


def load_grasp_config(path: Path) -> GraspConfig:
    return GraspConfig(**json.loads(path.read_text()))


class GraspRobot(Robot):
    def __init__(self, robot_cfg: Config, cfg: GraspConfig, shape: str) -> None:
        super().__init__(robot_cfg)
        self.robot_nq = self.model.nq
        self.robot_nv = self.model.nv
        # Export only to a temporary file; the source XML and controller stay intact.
        with TemporaryDirectory(prefix="spiral_grasp_model_") as directory:
            path = Path(directory) / "scene.xml"
            mujoco.mj_saveLastXML(str(path), self.model)
            root = ET.parse(path).getroot()
        # Exported mesh geoms otherwise infer their original masses on recompile.
        if robot_cfg.robot_mass_kg is not None:
            for index in range(1, 31):
                body = root.find(f".//body[@name='link{index}']")
                assert body is not None
                body_id = self.model.body(f"link{index}").id
                inertial = body.find("inertial")
                if inertial is None:
                    inertial = ET.SubElement(body, "inertial")
                inertial.attrib.clear()
                inertial.set("mass", str(self.model.body_mass[body_id]))
                for name, values in (
                    ("pos", self.model.body_ipos[body_id]),
                    ("quat", self.model.body_iquat[body_id]),
                    ("diaginertia", self.model.body_inertia[body_id]),
                ):
                    inertial.set(name, " ".join(map(str, values)))
        world = root.find("worldbody")
        assert world is not None
        ground = world.find("geom[@name='ground']")
        assert ground is not None
        # The robot's local +Z is horizontal; gravity points along local -X.
        ground.set("pos", f"{FLOOR_X_M} 0 0")
        ground.set("quat", "0.7071067811865476 0 0.7071067811865476 0")
        for geom in root.findall(".//geom"):
            geom.set("friction", f"{cfg.friction} 0.005 0.0001")
        fixture = ET.SubElement(
            world, "body", name="fixture", mocap="true", pos="0 0.08 0.24"
        )
        ET.SubElement(
            fixture, "site", name="fixture_marker", size="0.003", rgba="1 0.3 0.1 0.8"
        )
        body = ET.SubElement(world, "body", name="object", pos="0 0.08 0.24")
        ET.SubElement(body, "freejoint", name="object_free")
        if shape == "cylinder":
            size = [cfg.cylinder_radius_m, cfg.cylinder_length_m / 2]
        elif shape == "box":
            size = [dimension / 2 for dimension in cfg.box_size_m]
        else:
            raise ValueError(f"Unknown shape: {shape}")
        ET.SubElement(
            body,
            "geom",
            name="object_geom",
            type=shape,
            size=" ".join(map(str, size)),
            mass=str(cfg.object_mass_kg),
            friction=f"{cfg.friction} 0.005 0.0001",
            condim="4",
            rgba="0.95 0.55 0.15 1",
            solref=f"{cfg.contact_time_constant_s} 1",
            solimp="0.9999 0.9999 0.001",
            margin=str(cfg.collision_margin_m),
            priority="1",
        )
        equality = ET.SubElement(root, "equality")
        ET.SubElement(
            equality,
            "weld",
            name="presentation_fixture",
            body1="object",
            body2="fixture",
            relpose="0 0 0 1 0 0 0",
        )
        self.model = mujoco.MjModel.from_xml_string(
            ET.tostring(root, encoding="unicode")
        )
        # Use dense Newton with elliptic friction for the stiff contact model.
        self.model.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
        self.model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        self.model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        self.data = mujoco.MjData(self.model)
        self.tip_id = self.model.site("tip").id
        self.target_id = self.model.site("target").id
        self.model.site_rgba[self.target_id, 3] = 0
        self.object_id = self.model.body("object").id
        self.object_geom = self.model.geom("object_geom").id
        self.floor_geom = self.model.geom("ground").id
        self.fixture_id = self.model.equality("presentation_fixture").id
        self.fixture_marker = self.model.site("fixture_marker").id
        self.object_qpos = self.model.jnt_qposadr[self.model.joint("object_free").id]
        self.object_dof = self.model.jnt_dofadr[self.model.joint("object_free").id]
        self.grasp_geoms = [
            geom
            for geom in range(self.model.ngeom)
            if self.model.geom_bodyid[geom]
            in {
                self.model.body(f"link{link}").id
                for link in range(FIRST_GRASP_LINK, 31)
            }
        ]
        self.robot_geoms = [
            geom
            for geom in range(self.model.ngeom)
            if self.model.geom_bodyid[geom]
            in {self.model.body(f"link{link}").id for link in range(1, 31)}
        ]
        self.reset()
        self.max_object_penetration = 0.0

    def physics_step(self) -> None:
        super().physics_step()
        for contact in self.data.contact:
            if self.object_geom in (contact.geom1, contact.geom2):
                self.max_object_penetration = max(
                    self.max_object_penetration, -float(contact.dist)
                )

    @property
    def object_position(self) -> Array:
        return self.data.xpos[self.object_id].copy()

    def place_object(self, position: Array, quaternion: Array) -> None:
        self.data.qpos[self.object_qpos : self.object_qpos + 3] = position
        self.data.qpos[self.object_qpos + 3 : self.object_qpos + 7] = quaternion
        self.data.qvel[self.object_dof : self.object_dof + 6] = 0
        self.data.mocap_pos[0] = position
        self.data.mocap_quat[0] = quaternion
        mujoco.mj_forward(self.model, self.data)

    def object_distance(self, geoms: list[int]) -> float:
        return min(
            float(
                mujoco.mj_geomDistance(
                    self.model, self.data, geom, self.object_geom, 1.0, None
                )
            )
            for geom in geoms
        )


@dataclass
class Contacts:
    links: int
    span_deg: float
    normal_force_n: float
    floor: bool
    penetration_m: float = 0.0


def grasp_contacts(robot: GraspRobot, cfg: GraspConfig) -> Contacts:
    links: set[int] = set()
    radial_vectors: list[Array] = []
    normal_force = 0.0
    floor = False
    axis = robot.data.xmat[robot.object_id].reshape(3, 3)[:, 2]
    force = np.zeros(6)
    for index in range(robot.data.ncon):
        contact = robot.data.contact[index]
        if robot.object_geom not in (contact.geom1, contact.geom2):
            continue
        other = contact.geom2 if contact.geom1 == robot.object_geom else contact.geom1
        mujoco.mj_contactForce(robot.model, robot.data, index, force)
        if other == robot.floor_geom and (contact.dist <= 0 or force[0] > 0):
            floor = True
        if other not in robot.grasp_geoms or force[0] < cfg.min_contact_force_n:
            continue
        links.add(int(robot.model.geom_bodyid[other]))
        normal_force += float(force[0])
        radial = contact.pos - robot.object_position
        radial -= np.dot(radial, axis) * axis
        length = np.linalg.norm(radial)
        if length > 1e-8:
            radial_vectors.append(radial / length)
    span = 0.0
    if len(radial_vectors) >= 2:
        vectors = np.array(radial_vectors)
        span = float(np.degrees(np.arccos(np.clip(np.min(vectors @ vectors.T), -1, 1))))
    return Contacts(len(links), span, normal_force, floor, robot.max_object_penetration)


def secure_grasp(
    contacts: Contacts,
    released: bool,
    drift: float,
    speed: float,
    spin: float,
    cfg: GraspConfig,
) -> bool:
    return (
        released
        and not contacts.floor
        and contacts.links >= cfg.min_contact_links
        and contacts.span_deg >= cfg.min_contact_span_deg
        and drift <= cfg.max_drift_m
        and speed <= cfg.max_hold_speed_m_s
        and spin <= cfg.max_hold_spin_rad_s
        and contacts.penetration_m <= cfg.max_penetration_m
    )


class GraspEnv(gym.Env):
    metadata: dict[str, list[str]] = {"render_modes": []}

    def __init__(self, robot_cfg: Config, cfg: GraspConfig) -> None:
        robot_cfg = replace(robot_cfg, physics_dt=cfg.physics_dt)
        if not np.allclose(robot_cfg.gravity, [-9.81, 0, 0]):
            raise ValueError(
                "This horizontal grasp scene requires gravity [-9.81, 0, 0]"
            )
        self.robot_cfg = robot_cfg
        self.cfg = cfg
        self.robots = {shape: GraspRobot(robot_cfg, cfg, shape) for shape in SHAPES}
        self.shape = SHAPES[0]
        self.robot = self.robots[self.shape]
        self.shape_order: list[str] = []
        self.action_space = gym.spaces.Box(-1.0, 1.0, (3,), dtype=np.float32)
        observation_size = self.robot.robot_nq + self.robot.robot_nv + 42
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (observation_size,), dtype=np.float32
        )
        self.spawn = np.zeros(3)
        self.previous_action = np.zeros(3)
        self.tip_velocity = np.zeros(3)
        self.elapsed = 0
        self.hold_steps = 0
        self.last_contacts = Contacts(0, 0, 0, False)

    @property
    def released(self) -> bool:
        return not bool(self.robot.data.eq_active[self.robot.fixture_id])

    def observation(self) -> NDArray[np.float32]:
        robot = self.robot
        sizes = (
            [2 * self.cfg.cylinder_radius_m] * 2 + [self.cfg.cylinder_length_m]
            if self.shape == "cylinder"
            else self.cfg.box_size_m
        )
        remaining = max(
            0, self.cfg.presentation_seconds - self.elapsed * self.robot_cfg.control_dt
        )
        return np.concatenate(
            (
                robot.data.qpos[: robot.robot_nq],
                robot.data.qvel[: robot.robot_nv],
                robot.retraction / self.robot_cfg.max_retraction_m,
                robot.velocity / self.robot_cfg.max_speed_m_s,
                self.previous_action,
                self.tip_velocity,
                robot.object_position / self.cfg.distance_scale_m,
                robot.data.xquat[robot.object_id],
                robot.data.qvel[robot.object_dof : robot.object_dof + 6],
                (self.spawn - robot.object_position) / self.cfg.distance_scale_m,
                np.array(sizes) / self.cfg.distance_scale_m,
                [float(self.shape == shape) for shape in SHAPES],
                [
                    float(self.released),
                    remaining / self.cfg.presentation_seconds,
                    self.hold_steps * self.robot_cfg.control_dt / self.cfg.hold_seconds,
                ],
                [
                    self.last_contacts.links / 26,
                    self.last_contacts.span_deg / 180,
                    self.last_contacts.normal_force_n / self.robot_cfg.max_tension_n,
                ],
                (robot.tip - robot.object_position) / self.cfg.distance_scale_m,
            )
        ).astype(np.float32)

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[NDArray[np.float32], dict]:
        super().reset(seed=seed)
        if seed is not None:
            self.shape_order = []
        if options and "shape" in options:
            self.shape = options["shape"]
            if self.shape not in SHAPES:
                raise ValueError(f"shape must be one of {SHAPES}")
        elif self.cfg.fixed_object_shape is not None:
            self.shape = self.cfg.fixed_object_shape
        else:
            if not self.shape_order:
                self.shape_order = self.np_random.permutation(SHAPES).tolist()
            self.shape = self.shape_order.pop()
        self.robot = self.robots[self.shape]
        self.robot.reset()
        self.robot.max_object_penetration = 0.0
        self.robot.data.eq_active[self.robot.fixture_id] = True
        self.robot.model.site_rgba[self.robot.fixture_marker, 3] = 0.8
        for _ in range(PLACEMENT_ATTEMPTS):
            angle = self.np_random.uniform(0, 2 * np.pi)
            radius = self.np_random.uniform(
                self.cfg.radial_range_m[0], self.cfg.radial_range_m[1]
            )
            self.spawn = np.array(
                [
                    radius * np.cos(angle),
                    radius * np.sin(angle),
                    self.np_random.uniform(
                        self.cfg.axial_range_m[0], self.cfg.axial_range_m[1]
                    ),
                ]
            )
            # Cylinder/prism long axis is tangent to the bending direction.
            if self.cfg.fixed_object_position_m is not None:
                self.spawn = np.array(self.cfg.fixed_object_position_m)
                angle = np.arctan2(self.spawn[1], self.spawn[0])
            radial = np.array([np.cos(angle), np.sin(angle), 0])
            tangent = np.array([-np.sin(angle), np.cos(angle), 0])
            rotation = np.column_stack((radial, np.array([0, 0, -1]), tangent))
            quaternion = np.zeros(4)
            mujoco.mju_mat2Quat(quaternion, rotation.ravel())
            self.robot.place_object(self.spawn, quaternion)
            if (
                self.robot.object_distance(self.robot.robot_geoms)
                > PLACEMENT_CLEARANCE_M
            ):
                break
            if self.cfg.fixed_object_position_m is not None:
                raise ValueError(
                    "Fixed object placement overlaps the robot or lacks clearance"
                )
        else:
            raise ValueError(
                "Could not place object without overlap; adjust size/location ranges"
            )
        self.previous_action[:] = 0
        self.tip_velocity[:] = 0
        self.elapsed = 0
        self.hold_steps = 0
        self.last_contacts = Contacts(0, 0, 0, False)
        return self.observation(), {"shape": self.shape, "spawn_m": self.spawn.copy()}

    def step(
        self, action: Array
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict]:
        action = np.asarray(action, dtype=np.float64)
        robot = self.robot
        dt = self.robot_cfg.control_dt
        if not self.released and self.elapsed * dt >= self.cfg.presentation_seconds:
            robot.data.eq_active[robot.fixture_id] = False
            robot.model.site_rgba[robot.fixture_marker, 3] = 0
            robot.data.qacc_warmstart[:] = 0
            mujoco.mj_forward(robot.model, robot.data)
        previous_tip = robot.tip
        previous_velocity = robot.velocity.copy()
        robot.max_object_penetration = 0.0
        robot.control(action)
        velocity = (robot.tip - previous_tip) / dt
        acceleration = (robot.velocity - previous_velocity) / dt
        tip_acceleration = (velocity - self.tip_velocity) / dt
        contacts = grasp_contacts(robot, self.cfg)
        drift = float(np.linalg.norm(robot.object_position - self.spawn))
        object_velocity = robot.data.qvel[robot.object_dof : robot.object_dof + 6]
        speed = float(np.linalg.norm(object_velocity[:3]))
        spin = float(np.linalg.norm(object_velocity[3:]))
        secure = secure_grasp(contacts, self.released, drift, speed, spin, self.cfg)
        self.hold_steps = self.hold_steps + 1 if secure else 0
        success = self.hold_steps * dt >= self.cfg.hold_seconds
        dropped = self.released and (contacts.floor or drift > DROP_DISTANCE_M)
        collision_failure = contacts.penetration_m > self.cfg.max_penetration_m
        distance = max(0, robot.object_distance(robot.grasp_geoms))
        rotation = robot.data.geom_xmat[robot.object_geom].reshape(3, 3)
        approach_distance = centroid_surface_distance(
            rotation.T @ (robot.tip - robot.data.geom_xpos[robot.object_geom]),
            self.shape,
            robot.model.geom_size[robot.object_geom],
        )
        # Hand off from tip approach to distributed wrapping contact.
        approach_weight = 1 - min(contacts.links / self.cfg.min_contact_links, 1)
        dense_reward = (
            -distance / self.cfg.distance_scale_m
            - self.cfg.approach_reward
            * approach_weight
            * approach_distance
            / self.cfg.distance_scale_m
            + self.cfg.contact_reward
            * min(contacts.links / self.cfg.contact_saturation_links, 1)
            * float(not collision_failure)
            + self.cfg.span_reward
            * contacts.span_deg
            / 180
            * float(not collision_failure)
            + self.cfg.hold_reward * float(secure)
            - self.robot_cfg.action_change_weight
            * float(np.mean((action - self.previous_action) ** 2))
            - self.robot_cfg.acceleration_weight
            * float(np.mean((acceleration / self.robot_cfg.max_acceleration_m_s2) ** 2))
            - self.robot_cfg.tip_acceleration_weight
            * float(
                np.mean(
                    (tip_acceleration / self.robot_cfg.tip_acceleration_scale_m_s2) ** 2
                )
            )
        )
        # Contact is a reward rate; otherwise collecting it at 50 Hz can dwarf
        # the terminal bonus for actually holding the released object.
        reward = (
            dt * dense_reward
            + self.cfg.success_reward * float(success)
            - self.cfg.drop_penalty * float(dropped or collision_failure)
        )
        self.previous_action = np.clip(action, -1, 1)
        self.tip_velocity = velocity
        self.last_contacts = contacts
        self.elapsed += 1
        info = {
            "is_success": success,
            "dropped": dropped,
            "shape": self.shape,
            "released": self.released,
            "contact_links": contacts.links,
            "contact_span_deg": contacts.span_deg,
            "object_drift_m": drift,
            "object_speed_m_s": speed,
            "hold_seconds": self.hold_steps * dt,
            "penetration_m": contacts.penetration_m,
            "collision_failure": collision_failure,
            "approach_distance_m": approach_distance,
            "approach_weight": approach_weight,
        }
        return (
            self.observation(),
            float(reward),
            success or dropped or collision_failure,
            self.elapsed * dt >= self.cfg.episode_seconds,
            info,
        )


def centroid_surface_distance(tip_local: Array, shape: str, size: Array) -> float:
    """Measure tip distance to the surface along a ray through the centroid.

    Coordinates are in the object's frame. Size uses MuJoCo conventions:
    box half extents, or cylinder radius and half length. Interior points
    retain a positive error instead of rewarding penetration toward the centre.
    """
    distance = float(np.linalg.norm(tip_local))
    if distance == 0:
        return float(min(size[:2] if shape == "cylinder" else size))
    direction = tip_local / distance
    if shape == "box":
        nonzero = np.abs(direction) > 0
        radius = float(np.min(size[nonzero] / np.abs(direction[nonzero])))
    elif shape == "cylinder":
        radial = float(np.linalg.norm(direction[:2]))
        radius = min(
            float(size[0]) / radial if radial > 0 else np.inf,
            float(size[1]) / abs(direction[2]) if direction[2] != 0 else np.inf,
        )
    else:
        raise ValueError(f"Unknown shape: {shape}")
    return abs(distance - radius)


def scripted_action(env: GraspEnv) -> Array:
    # This only demonstrates closing around a presented object, not a trained grasp.
    angle = np.arctan2(env.spawn[1], env.spawn[0])
    cable_angles = np.array([np.pi / 2, 7 * np.pi / 6, -np.pi / 6])
    differential = np.cos(angle - cable_angles)
    target = np.where(
        differential >= 0,
        differential * env.robot_cfg.max_retraction_m,
        differential * env.robot_cfg.max_payout_m,
    )
    if env.robot_cfg.action_mode == "position":
        return env.robot.position_action(target)
    return np.clip(
        (target - env.robot.retraction) / (0.5 * env.robot_cfg.max_speed_m_s), -1, 1
    )


def demo(
    robot_cfg: Config, cfg: GraspConfig, policy_path: Path | None, seconds: float
) -> None:
    import mujoco.viewer
    from stable_baselines3 import PPO

    env = GraspEnv(robot_cfg, cfg)
    policy = PPO.load(policy_path, env=env, device=PPO_DEVICE) if policy_path else None
    start = time.monotonic()
    episode = 0
    print(
        "Scripted closing demo; no learned grasp policy."
        if policy is None
        else "Trained grasp policy."
    )
    print(
        f"Cylinder diameter={2 * cfg.cylinder_radius_m * 1000:g} mm; "
        f"sliding friction={cfg.friction:g}; object mass={cfg.object_mass_kg * 1000:g} g"
    )
    while seconds <= 0 or time.monotonic() - start < seconds:
        observation, info = env.reset(seed=DEMO_SEED if episode == 0 else None)
        print(
            f"{info['shape']} at {np.round(env.spawn * 1000, 1)} mm; fixture releases after {cfg.presentation_seconds:g}s"
        )
        robot = env.robot
        with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
            viewer.cam.lookat[:] = [0, 0, 0.22]
            viewer.cam.distance = 0.65
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -15
            deadline = time.monotonic()
            done = False
            while viewer.is_running() and not done:
                if seconds > 0 and time.monotonic() - start >= seconds:
                    env.close()
                    return
                action = (
                    scripted_action(env)
                    if policy is None
                    else policy.predict(observation, deterministic=True)[0]
                )
                observation, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                if env.elapsed % round(1 / robot_cfg.control_dt) == 0 or done:
                    print(
                        f"t={env.elapsed * robot_cfg.control_dt:.1f}s released={info['released']} "
                        f"links={info['contact_links']} span={info['contact_span_deg']:.0f}deg "
                        f"hold={info['hold_seconds']:.2f}s success={info['is_success']}"
                        f" penetration={info['penetration_m'] * 1000:.3f}mm"
                        f" collision_failure={info['collision_failure']}"
                    )
                if done:
                    reason = (
                        "penetration limit exceeded"
                        if info["collision_failure"]
                        else "successful hold"
                        if info["is_success"]
                        else "object dropped"
                        if info["dropped"]
                        else "episode time limit"
                    )
                    print(f"Episode ended: {reason}.", flush=True)
                viewer.sync()
                deadline += robot_cfg.control_dt
                time.sleep(max(0, deadline - time.monotonic()))
            if not viewer.is_running():
                break
        episode += 1
    env.close()


def train(robot_cfg: Config, cfg: GraspConfig, run: Path) -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv

    class TimeBudgetCallback(BaseCallback):
        def _on_training_start(self) -> None:
            self.deadline = time.monotonic() + cfg.training_seconds

        def _on_step(self) -> bool:
            return time.monotonic() < self.deadline

    if run.exists() and any(run.iterdir()):
        raise FileExistsError(
            f"Run directory is not empty: {run}; choose another --run"
        )
    env = GraspEnv(robot_cfg, cfg)
    check_env(env)
    run.mkdir(parents=True, exist_ok=True)
    (run / "robot_config.json").write_text(json.dumps(asdict(env.robot_cfg), indent=2))
    (run / "grasp_config.json").write_text(json.dumps(asdict(cfg), indent=2))
    env.close()
    monitored = make_vec_env(
        lambda: GraspEnv(robot_cfg, cfg),
        n_envs=cfg.training_envs,
        seed=cfg.seed,
        monitor_dir=str(run),
        monitor_kwargs={"info_keywords": ("is_success", "shape", "dropped")},
        vec_env_cls=SubprocVecEnv if cfg.training_envs > 1 else None,
    )
    print(
        f"Training with {cfg.training_envs} simulations and {PPO_DEVICE} PPO; "
        f"budget {cfg.training_seconds / 60:g} minutes or {cfg.training_steps} steps."
    )
    try:
        model = PPO(
            "MlpPolicy",
            monitored,
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
        model.learn(
            cfg.training_steps,
            callback=[
                CheckpointCallback(
                    save_freq=max(1, CHECKPOINT_STEPS // cfg.training_envs),
                    save_path=str(run),
                    name_prefix="grasp_policy",
                ),
                TimeBudgetCallback(),
            ],
        )
        model.save(run / "grasp_policy")
    finally:
        monitored.close()


def evaluate(robot_cfg: Config, cfg: GraspConfig, policy_path: Path) -> None:
    from stable_baselines3 import PPO

    env = GraspEnv(robot_cfg, cfg)
    policy = PPO.load(policy_path, env=env, device=PPO_DEVICE)
    results: list[dict] = []
    for episode in range(cfg.evaluation_episodes):
        shape = cfg.fixed_object_shape or SHAPES[episode % len(SHAPES)]
        observation, _ = env.reset(
            seed=cfg.seed + 10000 + episode, options={"shape": shape}
        )
        done = False
        while not done:
            action, _ = policy.predict(observation, deterministic=True)
            observation, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        results.append(info)
    for shape in SHAPES:
        subset = [result for result in results if result["shape"] == shape]
        if not subset:
            continue
        print(
            f"{shape}: {sum(result['is_success'] for result in subset)}/{len(subset)} successful holds; "
            f"{sum(result['dropped'] for result in subset)} drops; "
            f"{sum(result['collision_failure'] for result in subset)} penetration failures"
        )
    env.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Separate PPO task for spiral grasping"
    )
    parser.add_argument("mode", choices=("demo", "train", "evaluate"))
    parser.add_argument("--config", type=Path, default=ROOT / "spiral_grasp.json")
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--seconds", type=float, default=0)
    parser.add_argument(
        "--diameter-mm", type=float, help="Demo cylinder diameter; mass stays fixed"
    )
    parser.add_argument(
        "--friction", type=float, help="Demo sliding friction coefficient"
    )
    args = parser.parse_args()
    if args.mode != "demo" and (
        args.diameter_mm is not None or args.friction is not None
    ):
        parser.error(
            "--diameter-mm and --friction are demo overrides; use configuration files for training/evaluation"
        )
    if args.mode == "evaluate" and args.policy is None:
        parser.error("evaluate requires --policy")
    if args.mode == "train" and args.policy is not None:
        parser.error(
            "--policy is for demo/evaluate; grasp training starts a separate policy"
        )
    if args.policy:
        cfg = load_grasp_config(args.policy.parent / "grasp_config.json")
        robot_cfg = load_config(args.policy.parent / "robot_config.json")
    else:
        cfg = load_grasp_config(args.config)
        robot_cfg = load_config(args.config.resolve().parent / cfg.robot_config)
    if args.diameter_mm is not None:
        cfg = replace(cfg, cylinder_radius_m=args.diameter_mm / 2000)
    if args.friction is not None:
        cfg = replace(cfg, friction=args.friction)
    if args.mode == "train":
        train(robot_cfg, cfg, args.run)
    elif args.mode == "evaluate":
        evaluate(robot_cfg, cfg, args.policy)
    else:
        demo(robot_cfg, cfg, args.policy, args.seconds)


if __name__ == "__main__":
    main()

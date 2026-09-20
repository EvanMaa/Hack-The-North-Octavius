from __future__ import annotations

from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco
import newton
import numpy as np
from numpy.typing import NDArray
import warp as wp

from spiral import Array, Config

ENV_SPACING_M = 0.7
DISPLAY_FPS = 30.0


@wp.func
def guide_state(
    body: int,
    site: wp.vec3,
    poses: wp.array[wp.transform],
    velocities: wp.array[wp.spatial_vector],
    centers: wp.array[wp.vec3],
) -> tuple[wp.vec3, wp.vec3]:
    pose = poses[body]
    offset = wp.quat_rotate(wp.transform_get_rotation(pose), site - centers[body])
    velocity = wp.vec3(wp.spatial_top(velocities[body])) + wp.cross(
        wp.spatial_bottom(velocities[body]), offset
    )
    return wp.transform_point(pose, site), velocity


@wp.kernel
def cable_step(
    poses: wp.array[wp.transform],
    velocities: wp.array[wp.spatial_vector],
    centers: wp.array[wp.vec3],
    body_ids: wp.array2d[wp.int32],
    sites: wp.array2d[wp.vec3],
    rest: wp.array[float],
    actions: wp.array[float],
    parameters: wp.array[float],
    pulls: wp.array[float],
    speeds: wp.array[float],
    controls: wp.array[float],
) -> None:
    cable = wp.tid()
    dt, speed_max, acceleration = parameters[0], parameters[1], parameters[2]
    pull_max, payout_max = parameters[3], parameters[4]
    pull = float(pulls[cable])
    speed = float(speeds[cable])
    brake = acceleration * dt
    lower = wp.sqrt(
        wp.max(0.0, 2.0 * acceleration * (pull + payout_max) + brake * brake)
    )
    upper = wp.sqrt(wp.max(0.0, 2.0 * acceleration * (pull_max - pull) + brake * brake))
    desired = wp.clamp(actions[cable] * speed_max, -lower + brake, upper - brake)
    speed += wp.clamp(desired - speed, -brake, brake)
    pull = wp.clamp(pull + speed * dt, -payout_max, pull_max)
    length = float(0.0)
    rate = float(0.0)
    previous, previous_velocity = guide_state(
        body_ids[cable, 0], sites[cable, 0], poses, velocities, centers
    )
    for guide in range(1, body_ids.shape[1]):
        point, velocity = guide_state(
            body_ids[cable, guide], sites[cable, guide], poses, velocities, centers
        )
        segment = point - previous
        distance = wp.length(segment)
        length += distance
        rate += wp.dot(segment, velocity - previous_velocity) / wp.max(
            distance, 1.0e-12
        )
        previous, previous_velocity = point, velocity
    extension = length - (rest[cable] - pull)
    tension = float(0.0)
    if extension > 0.0:
        tension = wp.clamp(
            parameters[5] * extension + parameters[6] * (rate + speed),
            0.0,
            parameters[7],
        )
    pulls[cable] = pull
    speeds[cable] = speed
    controls[cable] = -tension


class SpinPhysics:
    def __init__(self, cfg: Config, run: Path, device: str, visible: bool) -> None:
        from newton.solvers import SolverMuJoCo
        from pxr import Gf, UsdGeom
        from scipy.spatial.transform import Rotation

        import isaaclab.sim as sim_utils
        from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonManager

        from prepare_isaac import prepare

        self.cfg = cfg
        self.visible = visible
        self.last_render_time = 0.0
        if visible:
            import omni.kit.app

            self.kit_app = omni.kit.app.get_app()
        self.count = cfg.training_envs
        self.manager = NewtonManager
        asset = prepare(run / "robot_config.json", run / "asset")
        self.reference = mujoco.MjModel.from_xml_path(str(asset))
        reference = self.reference
        data = mujoco.MjData(reference)
        mujoco.mj_forward(reference, data)
        tree = ET.parse(asset)
        for body in tree.findall(".//body"):
            index = reference.body(body.attrib["name"]).id
            inertial = ET.SubElement(body, "inertial")
            for key, values in (
                ("pos", reference.body_ipos[index]),
                ("quat", reference.body_iquat[index]),
                ("diaginertia", reference.body_inertia[index]),
            ):
                inertial.set(key, " ".join(map(str, values)))
            inertial.set("mass", str(reference.body_mass[index]))
        tree.write(asset, encoding="utf-8", xml_declaration=True)
        self.sim = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(
                dt=cfg.physics_dt,
                device=device,
                gravity=tuple(cfg.gravity),
                physics=NewtonCfg(
                    solver_cfg=MJWarpSolverCfg(integrator="implicitfast", njmax=2048),
                    # Capture cable control and solver together below.
                    use_cuda_graph=False,
                ),
            )
        )
        prototype = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(prototype)
        prototype.add_mjcf(str(asset), ctrl_direct=True)
        names = [label.rsplit("/", 1)[-1] for label in prototype.body_label]
        prototype.body_label[:] = names
        builder = NewtonManager.create_builder()
        print(f"Building {self.count} physics environments...", flush=True)
        columns = int(np.ceil(np.sqrt(self.count)))
        self.origins = np.array(
            [
                [
                    (index % columns) * ENV_SPACING_M,
                    (index // columns) * ENV_SPACING_M,
                    0.0,
                ]
                for index in range(self.count)
            ]
        )
        for index, origin in enumerate(self.origins):
            builder.add_world(
                prototype,
                xform=wp.transform(wp.vec3(*origin), wp.quat(0.0, 0.0, 0.0, 1.0)),
                label_prefix=f"/World/envs/env_{index}/Robot",
            )
            if visible and index % 4 == 0:
                self.kit_app.update()
        NewtonManager.set_builder(builder)
        if visible:
            stage = sim_utils.get_current_stage()
            print("Creating shared display geometry...", flush=True)
            library = UsdGeom.Scope.Define(stage, "/__SpiralGeometry")
            # Hide the library's originals. Referenced children do not inherit
            # this ancestor's visibility, so their instances remain visible.
            UsdGeom.Imageable(library.GetPrim()).MakeInvisible()
            for name in names:
                body = reference.body(name).id
                path = f"/__SpiralGeometry/{name}"
                UsdGeom.Xform.Define(stage, path)
                geom = int(np.flatnonzero(reference.geom_bodyid == body)[0])
                mesh_id = reference.geom_dataid[geom]
                start, size = (
                    reference.mesh_vertadr[mesh_id],
                    reference.mesh_vertnum[mesh_id],
                )
                vertices = reference.mesh_vert[start : start + size]
                points = (
                    Rotation.from_quat(np.roll(reference.geom_quat[geom], -1)).apply(
                        vertices
                    )
                    + reference.geom_pos[geom]
                )
                start, size = (
                    reference.mesh_faceadr[mesh_id],
                    reference.mesh_facenum[mesh_id],
                )
                mesh = UsdGeom.Mesh.Define(stage, path + "/mesh")
                mesh.CreatePointsAttr(points.tolist())
                mesh.CreateFaceVertexCountsAttr([3] * size)
                mesh.CreateFaceVertexIndicesAttr(
                    reference.mesh_face[start : start + size].ravel().tolist()
                )
                mesh.CreateSubdivisionSchemeAttr("none")
                mesh.CreateDisplayColorAttr([(0.25, 0.55, 0.8)])
                self.kit_app.update()
            for index, path in enumerate(builder.body_label):
                name = names[index % 30]
                body = reference.body(name).id
                prim = UsdGeom.Xform.Define(stage, path)
                prim.AddTranslateOp().Set(
                    Gf.Vec3d(*(data.xpos[body] + self.origins[index // 30]))
                )
                quat = data.xquat[body]
                prim.AddOrientOp().Set(Gf.Quatf(float(quat[0]), Gf.Vec3f(*quat[1:])))
                visual = stage.DefinePrim(path + "/visual", "Xform")
                visual.GetReferences().AddInternalReference(f"/__SpiralGeometry/{name}")
                visual.SetInstanceable(True)
                if (index + 1) % 30 == 0:
                    self.kit_app.update()
                    if (index + 1) % (30 * 16) == 0:
                        print(
                            f"Display instances: {(index + 1) // 30}/{self.count}",
                            flush=True,
                        )
            light = sim_utils.DomeLightCfg(intensity=2000.0)
            light.func("/World/Light", light)
            self.targets = [
                sim_utils.create_prim(
                    f"/World/target_{index}", "Sphere", attributes={"radius": 0.004}
                )
                for index in range(self.count)
            ]
            self.target_xforms = [
                prim.GetAttribute("xformOp:translate") for prim in self.targets
            ]
            for prim in self.targets:
                UsdGeom.Sphere(prim).CreateDisplayColorAttr([(0.2, 1.0, 0.2)])
            center = (self.origins.min(axis=0) + self.origins.max(axis=0)) / 2
            span = max(ENV_SPACING_M, float(np.ptp(self.origins[:, :2], axis=0).max()))
            self.sim.set_camera_view(
                (center + [span, -span, 0.8 * span]).tolist(),
                (center + [0, 0, 0.15]).tolist(),
            )
        print(
            "Initializing the GPU solver (first launch may compile kernels)...",
            flush=True,
        )
        self.sim.reset()
        self.model = NewtonManager.get_model()
        self.state = NewtonManager.get_state_0()
        self.control = NewtonManager.get_control()
        if (
            self.model.body_count != 30 * self.count
            or self.control.mujoco.ctrl.shape != (3 * self.count,)
        ):
            raise RuntimeError(
                "Expected 30 bodies and three tendon actuators per environment"
            )
        self.initial_q = self.state.joint_q.numpy().reshape(self.count, -1).copy()
        self.initial_qd = self.state.joint_qd.numpy().reshape(self.count, -1).copy()
        if (
            self.initial_q.shape[1] != reference.nq
            or self.initial_qd.shape[1] != reference.nv
        ):
            raise RuntimeError(
                "Imported joint coordinates do not match the spin observation layout"
            )
        routes = np.array(
            [
                [
                    reference.site(site.attrib["site"]).id
                    for site in route.findall("site")
                ]
                for route in tree.findall("tendon/spatial")
            ]
        )
        mapping = {name: index for index, name in enumerate(names)}
        bodies = np.array(
            [
                [
                    mapping[reference.body(int(reference.site_bodyid[site])).name]
                    for site in route
                ]
                for route in routes
            ],
            dtype=np.int32,
        )
        self.body_ids = wp.array(
            np.concatenate([bodies + 30 * index for index in range(self.count)]),
            dtype=wp.int32,
            device=device,
        )
        self.sites = wp.array(
            np.tile(reference.site_pos[routes], (self.count, 1, 1)),
            dtype=wp.vec3,
            device=device,
        )
        self.rest = wp.array(
            np.tile(data.ten_length, self.count), dtype=float, device=device
        )
        if cfg.max_retraction_m >= np.min(data.ten_length):
            raise ValueError("Maximum retraction exceeds initial cable length")
        self.parameters = wp.array(
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
        )
        self.actions = wp.zeros(3 * self.count, dtype=float, device=device)
        self.pulls = wp.zeros_like(self.actions)
        self.speeds = wp.zeros_like(self.actions)
        self.tip_body = mapping["link30"]
        self.tip_local = np.mean(
            [reference.site(f"s30_{guide}").pos for guide in (2, 4, 6)], axis=0
        )
        self.graph = None
        if not visible:
            wp.load_module(device=device)
            # Compile lazy solver kernels before CUDA capture, then restore rest.
            self._advance()
            self.set_state(
                np.arange(self.count, dtype=np.int64),
                self.initial_q,
                self.initial_qd,
                np.zeros((self.count, 3)),
                np.zeros((self.count, 3)),
            )
            with wp.ScopedCapture(device=device) as capture:
                self._advance()
            self.graph = capture.graph

    def _advance(self) -> None:
        for _ in range(round(self.cfg.control_dt / self.cfg.physics_dt)):
            wp.launch(
                cable_step,
                dim=3 * self.count,
                inputs=[
                    self.state.body_q,
                    self.state.body_qd,
                    self.model.body_com,
                    self.body_ids,
                    self.sites,
                    self.rest,
                    self.actions,
                    self.parameters,
                    self.pulls,
                    self.speeds,
                    self.control.mujoco.ctrl,
                ],
                device=self.model.device,
            )
            self.sim.step(render=False)

    def step(self, actions: Array) -> None:
        self.actions.assign(np.asarray(actions, dtype=np.float32).ravel())
        if self.graph is None:
            self._advance()
        else:
            wp.capture_launch(self.graph)

    def tip(self) -> Array:
        from scipy.spatial.transform import Rotation

        poses = self.state.body_q.numpy().reshape(self.count, 30, 7)[:, self.tip_body]
        return (
            poses[:, :3]
            + Rotation.from_quat(poses[:, 3:]).apply(self.tip_local)
            - self.origins
        )

    def set_state(
        self, ids: NDArray[np.int64], q: Array, qd: Array, pulls: Array, speeds: Array
    ) -> None:
        for target, values in (
            (self.state.joint_q, q),
            (self.state.joint_qd, qd),
            (self.pulls, pulls),
            (self.speeds, speeds),
        ):
            current = target.numpy().reshape(self.count, -1)
            current[ids] = values
            target.assign(current.ravel())
        self.state.clear_forces()
        newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)

    def render(self, goals: Array) -> None:
        if self.visible:
            now = time.monotonic()
            if now - self.last_render_time < 1 / DISPLAY_FPS:
                return
            self.last_render_time = now
            from pxr import Gf

            for index, transform in enumerate(self.target_xforms):
                transform.Set(Gf.Vec3d(*(goals[index] + self.origins[index])))
            self.sim.render()

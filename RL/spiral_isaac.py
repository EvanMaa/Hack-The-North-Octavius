from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import traceback
from typing import TYPE_CHECKING, TypedDict
import xml.etree.ElementTree as ET

if TYPE_CHECKING:
    from isaacsim import SimulationApp

ROOT = Path(__file__).resolve().parent


class Settings(TypedDict):
    robot_config: str
    pulls_mm: list[float]
    render_interval: int
    telemetry_interval: int


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SpiRob in Isaac Lab 3 / Isaac Sim 6 using Newton MuJoCo Warp."
    )
    parser.add_argument("--config", type=Path, default=ROOT / "spiral_isaac.json")
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="Stop after N physics steps; 0 runs until closed.",
    )
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.visualizer is None:
        args.visualizer = ["none"] if args.headless or args.steps else ["kit"]
    args.headless = "kit" not in args.visualizer
    if args.steps < 0 or (args.headless and args.steps == 0):
        parser.error("Use a positive --steps count for a headless run")
    settings: Settings = json.loads(args.config.read_text(encoding="utf-8"))
    render_interval = int(settings["render_interval"])
    telemetry_interval = int(settings["telemetry_interval"])
    if render_interval < 1 or telemetry_interval < 1:
        raise ValueError("Render and telemetry intervals must be positive")

    app = AppLauncher(args).app
    try:
        run(args, settings, render_interval, telemetry_interval, app)
    except Exception:
        # Kit may terminate the process in close(); report errors before cleanup.
        traceback.print_exc()
        app.close(exit_code=1)
        raise
    else:
        app.close()


def run(
    args: argparse.Namespace,
    settings: Settings,
    render_interval: int,
    telemetry_interval: int,
    app: SimulationApp,
) -> None:
    # Kit must start before importing simulation or USD modules.
    import mujoco
    import numpy as np
    from newton.solvers import SolverMuJoCo
    from pxr import Gf, UsdGeom
    from scipy.spatial.transform import Rotation

    import isaaclab.sim as sim_utils
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonManager

    from isaac_cables import CableController, cable_kinematics
    from prepare_isaac import prepare
    from spiral import load_config

    config_path = args.config.resolve().parent / settings["robot_config"]
    cfg = load_config(config_path)
    output_root = ROOT / "runs" / "isaac"
    output_root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="spiral_", dir=output_root))
    asset = prepare(config_path, output / "asset")
    reference = mujoco.MjModel.from_xml_path(str(asset))
    reference_data = mujoco.MjData(reference)
    mujoco.mj_forward(reference, reference_data)

    # Explicit inertias avoid differing mesh-inertia inference between importers.
    tree = ET.parse(asset)
    for body in tree.findall(".//body"):
        index = reference.body(body.attrib["name"]).id
        inertial = body.find("inertial")
        if inertial is None:
            inertial = ET.SubElement(body, "inertial")
        for name, values in (
            ("pos", reference.body_ipos[index]),
            ("quat", reference.body_iquat[index]),
            ("diaginertia", reference.body_inertia[index]),
        ):
            inertial.set(name, " ".join(map(str, values)))
        inertial.set("mass", str(reference.body_mass[index]))
    tree.write(asset, encoding="utf-8", xml_declaration=True)

    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(
            dt=cfg.physics_dt,
            device=args.device,
            gravity=tuple(cfg.gravity),
            physics=NewtonCfg(
                solver_cfg=MJWarpSolverCfg(integrator="implicitfast", njmax=2048),
                use_cuda_graph=True,
            ),
        )
    )
    builder = NewtonManager.create_builder()
    SolverMuJoCo.register_custom_attributes(builder)
    builder.add_mjcf(str(asset), ctrl_direct=True)
    body_names = [label.rsplit("/", 1)[-1] for label in builder.body_label]
    builder.body_label[:] = [f"/World/Robot/{name}" for name in body_names]
    NewtonManager.set_builder(builder)

    # Physics comes from the Newton builder; these meshes provide the Kit view.
    stage = sim_utils.get_current_stage()
    UsdGeom.Xform.Define(stage, "/World/Robot")
    for name in body_names:
        body_id = reference.body(name).id
        path = f"/World/Robot/{name}"
        body_prim = UsdGeom.Xform.Define(stage, path)
        body_prim.AddTranslateOp().Set(Gf.Vec3d(*reference_data.xpos[body_id]))
        quat = reference_data.xquat[body_id]
        body_prim.AddOrientOp().Set(Gf.Quatf(float(quat[0]), Gf.Vec3f(*quat[1:])))
        for geom_id in np.flatnonzero(reference.geom_bodyid == body_id):
            mesh_id = reference.geom_dataid[geom_id]
            vertex_start = reference.mesh_vertadr[mesh_id]
            vertices = reference.mesh_vert[
                vertex_start : vertex_start + reference.mesh_vertnum[mesh_id]
            ]
            rotation = Rotation.from_quat(np.roll(reference.geom_quat[geom_id], -1))
            points = rotation.apply(vertices) + reference.geom_pos[geom_id]
            face_start = reference.mesh_faceadr[mesh_id]
            faces = reference.mesh_face[
                face_start : face_start + reference.mesh_facenum[mesh_id]
            ]
            mesh = UsdGeom.Mesh.Define(stage, f"{path}/mesh")
            mesh.CreatePointsAttr(points.tolist())
            mesh.CreateFaceVertexCountsAttr([3] * len(faces))
            mesh.CreateFaceVertexIndicesAttr(faces.ravel().tolist())
            mesh.CreateSubdivisionSchemeAttr("none")
            mesh.CreateDisplayColorAttr([(0.25, 0.55, 0.8)])
    light = sim_utils.DomeLightCfg(intensity=2000.0)
    light.func("/World/Light", light)
    ground = UsdGeom.Mesh.Define(stage, "/World/Ground")
    ground.CreatePointsAttr([(-10, -10, 0), (10, -10, 0), (10, 10, 0), (-10, 10, 0)])
    ground.CreateFaceVertexCountsAttr([4])
    ground.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    ground.CreateDisplayColorAttr([(0.15, 0.18, 0.22)])
    sim.set_camera_view([0.65, 0.65, 0.45], [0.0, 0.0, 0.17])
    sim.reset()

    model = NewtonManager.get_model()
    state = NewtonManager.get_state_0()
    control = NewtonManager.get_control()
    indices = {name: index for index, name in enumerate(body_names)}
    routes = tree.findall("tendon/spatial")
    site_ids = np.array(
        [
            [reference.site(site.attrib["site"]).id for site in route.findall("site")]
            for route in routes
        ]
    )
    bodies = np.array(
        [
            [
                indices[reference.body(int(reference.site_bodyid[site])).name]
                for site in route
            ]
            for route in site_ids
        ]
    )
    sites = reference.site_pos[site_ids]
    centers = model.body_com.numpy().astype(np.float64)
    poses = state.body_q.numpy().astype(np.float64)
    initial_positions = poses[:, :3].copy()
    velocities = state.body_qd.numpy().astype(np.float64)
    lengths, _ = cable_kinematics(poses, velocities, centers, bodies, sites)
    np.testing.assert_allclose(lengths, reference_data.ten_length, atol=1e-6)
    if control.mujoco.ctrl.shape != (3,):
        raise RuntimeError("Expected exactly three native tendon actuators")
    controller = CableController(cfg, lengths)
    controller.set_target(np.array(settings["pulls_mm"], dtype=float) / 1000)
    ui_window = None
    telemetry = None
    if not args.headless:
        import omni.ui as ui

        ui_window = ui.Window("SpiRob cable pulls", width=460, height=290)
        with ui_window.frame:
            with ui.VStack():
                ui.Label("Absolute pulls (mm): positive pulls, negative pays out")
                ui.Label(
                    f"Range: {-cfg.max_payout_m * 1000:g} to {cfg.max_retraction_m * 1000:g} mm"
                )
                fields = []
                for index in range(3):
                    with ui.HStack():
                        ui.Label(f"Cable {index + 1}", width=90)
                        fields.append(ui.FloatField())
                for index, field in enumerate(fields):
                    field.model.set_value(float(controller.target[index] * 1000))
                status = ui.Label("Ready")

                def apply() -> None:
                    try:
                        controller.set_target(
                            np.array(
                                [field.model.get_value_as_float() for field in fields]
                            )
                            / 1000
                        )
                        status.text = "Command accepted"
                    except ValueError as error:
                        status.text = str(error)

                ui.Button("Apply", clicked_fn=apply)
                telemetry = ui.Label("")
    print(f"ISAAC_READY output={output} bodies={model.body_count} cables=3", flush=True)
    count = 0
    while app.is_running() and (args.steps == 0 or count < args.steps):
        if not sim.is_playing():
            sim.render()
            continue
        state = NewtonManager.get_state_0()
        poses = state.body_q.numpy().astype(np.float64)
        velocities = state.body_qd.numpy().astype(np.float64)
        if not np.all(np.isfinite(poses)) or not np.all(np.isfinite(velocities)):
            raise RuntimeError("Isaac simulation produced non-finite body state")
        lengths, rates = cable_kinematics(poses, velocities, centers, bodies, sites)
        control.mujoco.ctrl.assign(controller.step(lengths, rates).astype(np.float32))
        sim.step(render=not args.headless and count % render_interval == 0)
        count += 1
        if telemetry is not None and count % render_interval == 0:
            telemetry.text = (
                f"Pulls mm: {np.round(controller.retraction * 1000, 2)}\n"
                f"Tensions N: {np.round(controller.tension, 3)}"
            )
        if count % telemetry_interval == 0 or count == args.steps:
            print(
                f"step={count} pulls_mm={controller.retraction * 1000} tension_N={controller.tension}",
                flush=True,
            )
    if ui_window is not None:
        ui_window.destroy()
    final_poses = NewtonManager.get_state_0().body_q.numpy()
    if not np.all(np.isfinite(final_poses)):
        raise RuntimeError("Isaac simulation ended with non-finite body poses")
    summary = {
        "steps": count,
        "simulation_seconds": count * cfg.physics_dt,
        "pulls_mm": (controller.retraction * 1000).tolist(),
        "tension_n": controller.tension.tolist(),
        "body_names": body_names,
        "body_positions_m": final_poses[:, :3].tolist(),
        "maximum_body_displacement_m": float(
            np.max(np.linalg.norm(final_poses[:, :3] - initial_positions, axis=1))
        ),
    }
    with (output / "result.json").open("x", encoding="utf-8") as result:
        json.dump(summary, result, indent=2)
    sim.stop()
    print(f"ISAAC_DONE steps={count}", flush=True)


if __name__ == "__main__":
    main()

from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import datetime
import json

import mujoco
import numpy as np

from spiral import ROOT, Robot, load_config
from spiral_grasp import GraspEnv, GraspRobot, load_grasp_config, scripted_action

SECONDS = 16.0
CASES = (
    "neutral",
    "pull1",
    "pull2",
    "pull3",
    "all",
    "fixed",
    "box",
    "cylinder",
    "deep_box",
)


def contact_support(robot: GraspRobot) -> dict:
    upward = -robot.model.opt.gravity / np.linalg.norm(robot.model.opt.gravity)
    force = np.zeros(6)
    support = 0.0
    normal = 0.0
    links: set[str] = set()
    for index, contact in enumerate(robot.data.contact):
        if robot.object_geom not in (contact.geom1, contact.geom2):
            continue
        other = contact.geom1 if contact.geom2 == robot.object_geom else contact.geom2
        if other not in robot.robot_geoms:
            continue
        mujoco.mj_contactForce(robot.model, robot.data, index, force)
        sign = 1 if contact.geom2 == robot.object_geom else -1
        world_force = sign * contact.frame.reshape(3, 3).T @ force[:3]
        support += float(world_force @ upward)
        normal += float(force[0])
        if force[0] > 0.02:
            links.add(robot.model.body(int(robot.model.geom_bodyid[other])).name)
    return {"upward_force_n": support, "normal_force_n": normal, "links": sorted(links)}


def bending(robot: Robot) -> dict:
    angles = []
    for link in range(2, 31):
        body = robot.model.body(f"link{link}").id
        joint = robot.model.body_jntadr[body]
        address = robot.model.jnt_qposadr[joint]
        quat = robot.data.qpos[address : address + 4]
        angles.append(
            float(np.degrees(2 * np.arctan2(np.linalg.norm(quat[1:]), abs(quat[0]))))
        )
    return {
        "joint_rotation_deg_link2_to30": angles,
        "base_mean_deg": float(np.mean(angles[:9])),
        "middle_mean_deg": float(np.mean(angles[9:19])),
        "tip_mean_deg": float(np.mean(angles[19:])),
        "tension_n": (-robot.data.ctrl).tolist(),
        "retraction_m": robot.retraction.tolist(),
        "tip_m": robot.tip.tolist(),
    }


def diagnose(case: str) -> dict:
    cfg = load_grasp_config(ROOT / "spiral_grasp.json")
    robot_cfg = replace(load_config(ROOT / cfg.robot_config), physics_dt=cfg.physics_dt)
    if case in CASES[:5]:
        robot = Robot(robot_cfg)
        pulls = np.full(3, -robot_cfg.max_payout_m)
        if case == "neutral":
            pulls[:] = 0
        elif case == "all":
            pulls[:] = robot_cfg.max_retraction_m
        else:
            pulls[int(case[-1]) - 1] = robot_cfg.max_retraction_m
        action = robot.position_action(pulls)
        for _ in range(round(SECONDS / robot_cfg.control_dt)):
            robot.control(action)
        return {"case": case, **bending(robot)}
    if case == "fixed":
        cfg = replace(
            cfg, fixed_object_shape="cylinder", fixed_object_position_m=[0, 0.05, 0.26]
        )
    env = GraspEnv(robot_cfg, cfg)
    env.reset(
        seed=19, options={"shape": "box" if case in ("box", "deep_box") else "cylinder"}
    )
    if case == "deep_box":
        env.spawn = np.array([0.0, 0.085, 0.25])
        env.robot.place_object(env.spawn, np.array([np.sqrt(0.5), 0, -np.sqrt(0.5), 0]))
        if env.robot.object_distance(env.robot.robot_geoms) <= 0.002:
            raise ValueError("Diagnostic object placement lacks initial clearance")
    trace = []
    peak_penetration = 0.0
    peak_span = 0.0
    peak_links = 0
    while True:
        _, _, terminated, truncated, info = env.step(scripted_action(env))
        peak_penetration = max(peak_penetration, info["penetration_m"])
        peak_span = max(peak_span, info["contact_span_deg"])
        peak_links = max(peak_links, info["contact_links"])
        time_s = env.elapsed * robot_cfg.control_dt
        if env.elapsed % 50 == 0 or time_s >= cfg.presentation_seconds - 0.2:
            trace.append(
                {
                    "time_s": time_s,
                    "released": bool(info["released"]),
                    "object_position_m": env.robot.object_position.tolist(),
                    "object_speed_m_s": info["object_speed_m_s"],
                    "span_deg": info["contact_span_deg"],
                    "penetration_m": info["penetration_m"],
                    **contact_support(env.robot),
                    **bending(env.robot),
                }
            )
        if terminated or truncated:
            break
    result = {
        "case": case,
        "weight_n": cfg.object_mass_kg * 9.81,
        "peak_penetration_m": peak_penetration,
        "peak_span_deg": peak_span,
        "peak_links": peak_links,
        "success": bool(info["is_success"]),
        "dropped": bool(info["dropped"]),
        "trace": trace,
    }
    env.close()
    return result


if __name__ == "__main__":
    output = ROOT / "runs" / f"spiral_diagnostics_{datetime.now():%Y%m%d_%H%M%S}.json"
    results = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for result in pool.map(diagnose, CASES):
            results.append(result)
            print(
                json.dumps(
                    {key: value for key, value in result.items() if key != "trace"}
                ),
                flush=True,
            )
    output.write_text(json.dumps(results, indent=2))
    print(f"Saved {output}")

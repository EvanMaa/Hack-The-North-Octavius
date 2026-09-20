from concurrent.futures import ProcessPoolExecutor
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np

from spiral import ROOT, load_config
from spiral_grasp import GraspEnv, load_grasp_config, scripted_action

CONTACT_HOLD_LINKS = 3
CONTACT_HOLD_SPAN_DEG = 60.0


def contact_hold(render: bool, config_path: Path) -> None:
    cfg = load_grasp_config(config_path)
    if cfg.fixed_object_position_m is None:
        cfg = replace(
            cfg, fixed_object_shape="cylinder", fixed_object_position_m=[0, 0.05, 0.26]
        )
    env = GraspEnv(load_config(config_path.resolve().parent / cfg.robot_config), cfg)
    env.reset(seed=19)
    target = None
    latched = False
    viewer = None
    if render:
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(env.robot.model, env.robot.data)
        viewer.cam.lookat[:] = [0, 0.05, 0.24]
        viewer.cam.distance = 0.6
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -15
    print(
        "Experimental contact-triggered cable hold; no policy and no guaranteed grasp."
    )
    try:
        while viewer is None or viewer.is_running():
            start = time.monotonic()
            contacts = env.last_contacts
            if (
                target is None
                and contacts.links >= CONTACT_HOLD_LINKS
                and contacts.span_deg >= CONTACT_HOLD_SPAN_DEG
            ):
                target = env.robot.position_action(env.robot.retraction.copy())
                latched = True
                print(
                    f"Holding cable targets at {env.elapsed * env.robot_cfg.control_dt:.2f}s",
                    flush=True,
                )
            action = scripted_action(env) if target is None else target
            _, _, terminated, truncated, info = env.step(action)
            if env.elapsed % 50 == 0 or terminated or truncated:
                print(
                    json.dumps(
                        {
                            "time_s": env.elapsed * env.robot_cfg.control_dt,
                            "contact_hold_triggered": latched,
                            "released": bool(info["released"]),
                            "links": info["contact_links"],
                            "span_deg": info["contact_span_deg"],
                            "hold_seconds": info["hold_seconds"],
                            "success": bool(info["is_success"]),
                            "dropped": bool(info["dropped"]),
                            "penetration_mm": info["penetration_m"] * 1000,
                        }
                    ),
                    flush=True,
                )
            if viewer is not None:
                viewer.sync()
                time.sleep(
                    max(0, env.robot_cfg.control_dt - (time.monotonic() - start))
                )
            if terminated or truncated:
                break
    finally:
        if viewer is not None:
            viewer.close()
        env.close()


def check_placement(
    placement: tuple[float, float, float],
    pulls: tuple[float, float, float] | None = None,
) -> dict:
    cfg = load_grasp_config(ROOT / "spiral_grasp.json")
    cfg = replace(
        cfg, fixed_object_shape="cylinder", fixed_object_position_m=list(placement)
    )
    env = GraspEnv(load_config(ROOT / cfg.robot_config), cfg)
    env.reset(seed=19, options={"shape": "cylinder"})
    if env.robot.object_distance(env.robot.robot_geoms) <= 0.002:
        env.close()
        return {"placement": placement, "overlap": True}
    peak_links = 0
    peak_span = 0.0
    peak_hold = 0.0
    peak_penetration = 0.0
    while True:
        action = (
            scripted_action(env)
            if pulls is None
            else env.robot.position_action(np.array(pulls))
        )
        _, _, terminated, truncated, info = env.step(action)
        peak_links = max(peak_links, info["contact_links"])
        peak_span = max(peak_span, info["contact_span_deg"])
        peak_hold = max(peak_hold, info["hold_seconds"])
        peak_penetration = max(peak_penetration, info["penetration_m"])
        if terminated or truncated:
            break
    env.close()
    return {
        "placement": placement,
        "pulls_m": pulls,
        "success": bool(info["is_success"]),
        "peak_links": peak_links,
        "peak_span": peak_span,
        "hold_seconds": peak_hold,
        "penetration_m": peak_penetration,
        "dropped": bool(info["dropped"]),
        "collision_failure": bool(info["collision_failure"]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fixed-cylinder grasp feasibility experiments"
    )
    parser.add_argument(
        "mode", nargs="?", choices=("sweep", "contact-hold"), default="sweep"
    )
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--config", type=Path, default=ROOT / "spiral_grasp.json")
    args = parser.parse_args()
    if args.render and args.mode != "contact-hold":
        parser.error("--render is supported for contact-hold only")
    if args.mode == "contact-hold":
        contact_hold(args.render, args.config)
        raise SystemExit(0)
    pulls = [
        (first, other, other)
        for first in (0.025, 0.05)
        for other in (-0.03, 0.0, 0.02, 0.04)
    ]
    placements = [(0.0, 0.05, 0.26)] * len(pulls)
    with ProcessPoolExecutor(max_workers=4) as pool:
        for result in pool.map(check_placement, placements, pulls):
            print(json.dumps(result), flush=True)

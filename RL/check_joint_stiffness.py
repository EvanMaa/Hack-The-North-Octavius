from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import datetime
import json

import numpy as np

from diagnose_spiral import bending
from spiral import ROOT, Robot, load_config

CASES = ((0.3, 1.0), (0.1, 1.0), (0.03, 1.0), (0.1, 0.2))
SAMPLE_SECONDS = (8, 16, 32)


def compare_scales(scales: tuple[float, float]) -> dict:
    stiffness, damping = scales
    cfg = replace(
        load_config(ROOT / "spiral_config.json"),
        joint_stiffness_scale=stiffness,
        joint_damping_scale=damping,
        physics_dt=0.0005,
    )
    robot = Robot(cfg)
    action = robot.position_action(
        np.array([cfg.max_retraction_m, -cfg.max_payout_m, -cfg.max_payout_m])
    )
    result: dict = {
        "stiffness_scale": stiffness,
        "damping_scale": damping,
        "samples": [],
    }
    try:
        for step in range(1, round(max(SAMPLE_SECONDS) / cfg.control_dt) + 1):
            robot.control(action)
            if step in [round(seconds / cfg.control_dt) for seconds in SAMPLE_SECONDS]:
                result["samples"].append(
                    {
                        "seconds": step * cfg.control_dt,
                        "max_joint_speed_rad_s": float(np.max(np.abs(robot.data.qvel))),
                        **bending(robot),
                    }
                )
    except RuntimeError as error:
        result["physics_error"] = str(error)
    return result


if __name__ == "__main__":
    output = ROOT / "runs" / f"joint_stiffness_{datetime.now():%Y%m%d_%H%M%S}.json"
    results = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for result in pool.map(compare_scales, CASES):
            results.append(result)
            print(json.dumps(result), flush=True)
    output.write_text(json.dumps(results, indent=2))
    print(f"Saved {output}")

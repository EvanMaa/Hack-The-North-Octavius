from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import time
import tkinter as tk
from tkinter import ttk

import mujoco.viewer
import numpy as np

from spiral import Array, Config, ROOT, Robot, load_config


def parse_pulls(text: str, cfg: Config) -> Array:
    """Parse absolute motor travel in millimetres, rejecting invalid commands.

    Accepts bracketed arrays or three comma/space-separated numbers. Positive
    travel retracts a cable relative to its initial length; negative pays out.
    """
    values = text.strip()
    if values.startswith("[") and values.endswith("]"):
        values = values[1:-1]
    try:
        pulls = np.array([float(value) for value in values.replace(",", " ").split()])
    except ValueError as error:
        raise ValueError("Enter three numbers, for example [20, 0, 0].") from error
    if pulls.shape != (3,) or not np.all(np.isfinite(pulls)):
        raise ValueError("Enter exactly three finite numbers.")
    lower = -cfg.max_payout_m * 1000
    upper = cfg.max_retraction_m * 1000
    if np.any(pulls < lower) or np.any(pulls > upper):
        raise ValueError(f"Each pull must be between {lower:g} and {upper:g} mm.")
    return pulls / 1000


def run(cfg: Config) -> None:
    robot = Robot(replace(cfg, action_mode="position"))
    robot.model.site_rgba[robot.target_id, 3] = 0
    desired = np.zeros(3)
    reference_tip = robot.tip
    paused = False

    window = tk.Tk()
    window.title("SpiRob — manual cable control")
    panel = ttk.Frame(window, padding=16)
    panel.grid(sticky="nsew")
    ttk.Label(
        panel, text="Absolute motor pulls [cable 1, cable 2, cable 3] in mm"
    ).grid(row=0, column=0, columnspan=4, sticky="w")
    command = tk.StringVar(value="[0, 0, 0]")
    entry = ttk.Entry(panel, textvariable=command, width=48)
    entry.grid(row=1, column=0, columnspan=4, sticky="ew", pady=8)
    ttk.Label(
        panel,
        text=(
            f"Positive = pull; negative = payout. Range: "
            f"{-cfg.max_payout_m * 1000:g} to {cfg.max_retraction_m * 1000:g} mm.\n"
            "Commands are absolute, not increments. Apply moves all three together."
        ),
    ).grid(row=2, column=0, columnspan=4, sticky="w")
    status = tk.StringVar(
        value="Ready. Gravity is active; zero pull is not a rigid hold."
    )
    telemetry = tk.StringVar()

    def apply() -> None:
        nonlocal desired, reference_tip
        try:
            requested = parse_pulls(command.get(), cfg)
        except ValueError as error:
            status.set(str(error))
            return
        desired = requested
        reference_tip = robot.tip
        status.set("Command accepted." + (" Resume to move." if paused else ""))

    def zero() -> None:
        command.set("[0, 0, 0]")
        apply()

    def reset() -> None:
        nonlocal desired, reference_tip
        robot.reset()
        desired = np.zeros(3)
        reference_tip = robot.tip
        command.set("[0, 0, 0]")
        status.set("Simulation reset to the initial shape and zero motor travel.")

    def toggle_pause() -> None:
        nonlocal paused
        paused = not paused
        pause_button.configure(text="Resume" if paused else "Pause")
        status.set("Simulation paused." if paused else "Simulation running.")

    ttk.Button(panel, text="Apply", command=apply).grid(row=3, column=0, pady=12)
    ttk.Button(panel, text="Zero pulls", command=zero).grid(row=3, column=1)
    ttk.Button(panel, text="Reset simulation", command=reset).grid(row=3, column=2)
    pause_button = ttk.Button(panel, text="Pause", command=toggle_pause)
    pause_button.grid(row=3, column=3)
    entry.bind("<Return>", lambda event: apply())
    ttk.Label(panel, textvariable=telemetry, font="TkFixedFont").grid(
        row=4, column=0, columnspan=4, sticky="w", pady=8
    )
    ttk.Label(panel, textvariable=status, wraplength=540).grid(
        row=5, column=0, columnspan=4, sticky="w"
    )
    ttk.Label(
        panel,
        text="Tip XYZ uses the fixed-base/world axes. Prediction uses spiral_config.json.",
    ).grid(row=6, column=0, columnspan=4, sticky="w", pady=8)
    entry.focus_set()

    with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
        viewer.cam.lookat[:] = [0, 0, 0.17]
        viewer.cam.distance = 0.7
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -15

        def tick() -> None:
            nonlocal paused
            if not viewer.is_running():
                window.destroy()
                return
            started = time.monotonic()
            if not paused:
                try:
                    with viewer.lock():
                        robot.control(robot.position_action(desired))
                except RuntimeError as error:
                    paused = True
                    pause_button.configure(text="Resume")
                    status.set(f"Physics stopped: {error}. Reset before resuming.")
            telemetry.set(
                f"Simulation time: {robot.data.time:8.2f} s\n"
                f"Requested mm:    {np.round(desired * 1000, 2)}\n"
                f"Actual pull mm:  {np.round(robot.retraction * 1000, 2)}\n"
                f"Cable tension N: {np.round(-robot.data.ctrl, 2)}\n"
                f"Tip XYZ mm:      {np.round(robot.tip * 1000, 2)}\n"
                f"Tip change mm:   {np.round((robot.tip - reference_tip) * 1000, 2)}"
            )
            viewer.sync()
            delay = max(1, round(1000 * (cfg.control_dt - time.monotonic() + started)))
            window.after(delay, tick)

        window.after(0, tick)
        window.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manually control three SpiRob cables."
    )
    parser.add_argument("--config", type=Path, default=ROOT / "spiral_config.json")
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()

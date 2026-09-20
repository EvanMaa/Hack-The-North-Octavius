from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent


def prepare(config_path: Path, output: Path) -> Path:
    """Copy the configured MuJoCo asset into a new Isaac import directory.

    This prepares an MJCF input, not a native Isaac controller. The spatial
    tendons and actuators are preserved, but the Python cable force law in
    spiral.py must still be implemented in the target simulation runtime.

    Args:
        config_path: Robot configuration used by the MuJoCo scripts.
        output: New directory for the copied XML and mesh assets.

    Returns:
        Path to the prepared MJCF file.

    Raises:
        FileExistsError: The output directory already exists.
        ValueError: A required physics setting is invalid.
    """
    config = json.loads(config_path.read_text(encoding="utf-8"))
    timestep = float(config["physics_dt"])
    stiffness = float(config.get("joint_stiffness_scale", 1.0))
    damping = float(config.get("joint_damping_scale", 1.0))
    tension = float(config["max_tension_n"])
    gravity = [float(value) for value in config["gravity"]]
    for name, value in (
        ("physics_dt", timestep),
        ("joint_stiffness_scale", stiffness),
        ("max_tension_n", tension),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(damping) or damping < 0:
        raise ValueError("joint_damping_scale must be finite and nonnegative")
    if len(gravity) != 3 or not all(math.isfinite(value) for value in gravity):
        raise ValueError("gravity must contain three finite values")

    tree = ET.parse(ROOT / "robot.xml")
    root = tree.getroot()
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", str(timestep))
    option.set("gravity", " ".join(map(str, gravity)))
    for index, joint in enumerate(root.findall(".//joint")):
        if "name" not in joint.attrib:
            joint.set("name", f"spiral_joint_{index + 1}")
        joint.set("stiffness", str(float(joint.get("stiffness", "0")) * stiffness))
        joint.set("damping", str(float(joint.get("damping", "0")) * damping))
    for motor in root.findall("actuator/motor"):
        motor.set("gear", "1")
        motor.set("ctrlrange", f"{-tension} 0")

    meshes = {ROOT / mesh.attrib["file"] for mesh in root.findall("asset/mesh")}
    for mesh_path in meshes:
        if not mesh_path.is_file():
            raise FileNotFoundError(mesh_path)
    # mkdir is exclusive: never replace a prior export or existing project files.
    output.mkdir(parents=True, exist_ok=False)
    for mesh_path in meshes:
        shutil.copyfile(mesh_path, output / mesh_path.name)
    for mesh in root.findall("asset/mesh"):
        mesh.set("file", Path(mesh.attrib["file"]).name)
    shutil.copyfile(config_path, output / "robot_config.json")
    destination = output / "spiral.xml"
    tree.write(destination, encoding="utf-8", xml_declaration=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a separate MJCF asset for Isaac Sim import. "
            "This does not port the Python cable controller."
        )
    )
    parser.add_argument("--config", type=Path, default=ROOT / "spiral_config.json")
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "isaac_import")
    args = parser.parse_args()
    destination = prepare(args.config, args.output)
    print(f"Prepared MJCF: {destination}")
    print("Cable control still requires a native Isaac runtime implementation.")


if __name__ == "__main__":
    main()

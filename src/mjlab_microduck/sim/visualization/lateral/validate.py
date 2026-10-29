"""Gate a candidate on pure lateral commands in the authoritative CPU physics.

No network, rendering, heading correction, or model-cache mutation is involved.
An ONNX file passing shape checks is not evidence that it can move sideways.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

from mjlab_microduck.sim.body_server import (
    HOME_POSE,
    JOINT_NAMES,
    Body,
    World,
)
from mjlab_microduck.sim.visualization.scenes import VELSTAND_SCENE

SEQUENCE = (
    ("stand", 0.0, 2),
    ("left", 0.3, 10),
    ("stop", 0.0, 3),
    ("right", -0.3, 10),
    ("stop_again", 0.0, 3),
    ("left_after_right", 0.3, 10),
)
SLOTS = [index for index, name in enumerate(JOINT_NAMES) if name != "mouth"]


def yaw(quat) -> float:
    w, x, y, z = quat
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def metrics(before, trajectory, vy: float) -> dict:
    """Measure in the duck's starting axes, retaining full turns rather than wrapping them."""
    positions = np.asarray(trajectory)
    heading = before[3]
    offsets = positions[:, :2] - before[:2]
    dx, dy = positions[-1, :2] - before[:2]
    angles = np.unwrap(np.r_[heading, positions[:, 3]]) - heading
    return {
        "forward_m": float(math.cos(heading) * dx + math.sin(heading) * dy),
        "lateral_m": float(-math.sin(heading) * dx + math.cos(heading) * dy),
        "max_translation_m": float(np.linalg.norm(offsets, axis=1).max()),
        "final_yaw_deg": float(np.rad2deg(angles[-1])),
        "max_yaw_deg": float(np.rad2deg(np.abs(angles).max())),
        "min_trunk_z": float(positions[:, 2].min()),
        "max_gravity_z": float(positions[:, 4].max()),
        "requested": [0, vy, 0],
    }


def passes_gate(result: dict, vy: float, args) -> bool:
    if result["min_trunk_z"] < 0.085 or result["max_gravity_z"] > -0.85:
        return False
    if not vy:
        return (
            result["max_translation_m"] <= 0.05
            and result["max_yaw_deg"] <= args.max_yaw
        )
    progress = result["lateral_m"] * math.copysign(1, vy)
    return bool(
        progress >= args.min_progress
        and result["max_yaw_deg"] <= args.max_yaw
        and abs(result["forward_m"]) <= args.max_forward_ratio * progress
    )


def run(policy: Path, seed: int, args) -> dict:
    world = World(VELSTAND_SCENE, actuator_model="bam")
    body = Body(world, 0)
    body.place(None, 0.125, 0)
    world.bodies.append(body)
    body.bam.rng = np.random.default_rng(seed)
    body.set_torque(True)
    mujoco.mj_forward(world.model, world.data)
    session = ort.InferenceSession(str(policy), providers=["CPUExecutionProvider"])
    if session.get_inputs()[0].shape != [1, 61] or session.get_outputs()[0].shape != [
        1,
        14,
    ]:
        raise ValueError("expected a Microduck 61-observation / 14-action policy")
    input_name = session.get_inputs()[0].name
    last_action = np.zeros(14)
    home = np.asarray(HOME_POSE)[SLOTS]
    stages = []
    for name, vy, seconds in SEQUENCE:
        sensors = body.sensors()
        before = [*sensors["trunk"], yaw(sensors["imu"]["quat"])]
        trajectory = []
        for _ in range(seconds * 50):
            sensors = body.sensors()
            observation = np.r_[
                sensors["imu"]["gyro"],
                sensors["imu"]["gravity"],
                np.asarray(sensors["positions"])[SLOTS] - home,
                np.asarray(sensors["velocities"])[SLOTS],
                last_action,
                [0, vy, 0],
                np.zeros(10),
            ].astype(np.float32)
            action = session.run(None, {input_name: observation[None]})[0][0]
            if not np.isfinite(action).all():
                raise ValueError("policy emitted a non-finite action")
            last_action = action.copy()
            targets = np.asarray(HOME_POSE).copy()
            targets[SLOTS] += action
            body.set_targets(targets.tolist())
            world.step(4)
            sensors = body.sensors()
            trajectory.append(
                [
                    *sensors["trunk"],
                    yaw(sensors["imu"]["quat"]),
                    sensors["imu"]["gravity"][2],
                ]
            )
        result = {
            "seed": seed,
            "stage": name,
            "seconds": seconds,
            **metrics(before, trajectory, vy),
        }
        result["passed"] = passes_gate(result, vy, args)
        stages.append(result)
        print(json.dumps(result), flush=True)
    return {
        "seed": seed,
        "passed": all(stage["passed"] for stage in stages),
        "stages": stages,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--min-progress", type=float, default=0.2, help="metres per 10s strafe"
    )
    parser.add_argument(
        "--max-yaw", type=float, default=10, help="degrees over an entire segment"
    )
    parser.add_argument("--max-forward-ratio", type=float, default=0.25)
    args = parser.parse_args()
    policy = args.policy.resolve(strict=True)
    trials = [run(policy, seed, args) for seed in args.seeds]
    report = {
        "policy": str(policy),
        "sha256": hashlib.sha256(policy.read_bytes()).hexdigest(),
        "validation_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "mujoco_version": mujoco.__version__,
        "numpy_version": np.__version__,
        "physics": "BAM M6/backlash/200Hz physics/50Hz policy; no yaw compensation",
        "limits": {
            "min_progress_m": args.min_progress,
            "max_yaw_deg": args.max_yaw,
            "max_forward_ratio": args.max_forward_ratio,
            "max_stop_translation_m": 0.05,
        },
        "passed": all(trial["passed"] for trial in trials),
        "trials": trials,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()

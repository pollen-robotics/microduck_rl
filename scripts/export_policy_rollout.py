"""Export a headless ONNX policy rollout as a validated Blender motion archive."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from mjlab_microduck.blender_motion import validate_motion
from mjlab_microduck.policy_rollout import (
    PolicyRolloutConfig,
    PolicyRolloutError,
    export_policy_rollout,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy", type=Path, help="ONNX policy exported by scripts/export.py")
    parser.add_argument("--output", type=Path, required=True, help="output .npz motion archive")
    parser.add_argument("--duration", type=float, default=4.0, help="rollout duration in seconds")
    parser.add_argument("--lin-vel-x", type=float, default=0.30, help="forward command in m/s")
    parser.add_argument("--lin-vel-y", type=float, default=0.0, help="lateral command in m/s")
    parser.add_argument("--ang-vel-z", type=float, default=0.0, help="yaw command in rad/s")
    parser.add_argument("--seed", type=int, default=0, help="deterministic rollout seed")
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    try:
        output_path = export_policy_rollout(
            PolicyRolloutConfig(
                policy_path=args.policy,
                output_path=args.output,
                duration_s=args.duration,
                command=(args.lin_vel_x, args.lin_vel_y, args.ang_vel_z),
                seed=args.seed,
            )
        )
        validation = validate_motion(output_path)
        with np.load(output_path, allow_pickle=False) as archive:
            frames = int(archive["joint_pos"].shape[0])
            policy_hash = json.loads(str(archive["source_hashes_json"][0]))[
                "policy_sha256"
            ]
    except (PolicyRolloutError, OSError, ValueError) as exc:
        print(f"Policy rollout failed: {exc}", file=sys.stderr)
        return 2

    print(f"Output: {output_path}")
    print(f"Frames: {frames}")
    print(f"Command: ({args.lin_vel_x}, {args.lin_vel_y}, {args.ang_vel_z})")
    print(f"Policy SHA-256: {policy_hash}")
    print(
        "Validator errors: "
        f"position={validation.max_position_error_m:.3g} m, "
        f"orientation={validation.max_orientation_error_rad:.3g} rad"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

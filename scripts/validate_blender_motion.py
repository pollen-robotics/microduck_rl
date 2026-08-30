#!/usr/bin/env python3
"""Validate a motion exported by Open Duck Blender."""

from __future__ import annotations

import argparse

from mjlab_microduck.blender_motion import MotionValidationError, validate_motion


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("motion", help="Blender-exported .npz archive")
    args = parser.parse_args()
    try:
        result = validate_motion(args.motion)
    except (MotionValidationError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(
        f"Validated {result.frames} frames; max position error "
        f"{result.max_position_error_m:.3g} m, orientation error "
        f"{result.max_orientation_error_rad:.3g} rad"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

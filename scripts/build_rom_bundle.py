"""Create an immutable MicroDuck policy bundle from exported ONNX artifacts."""

from __future__ import annotations

import argparse
import subprocess
from datetime import datetime
from pathlib import Path

from mjlab_microduck.robot.microduck_constants import MICRODUCK_WALK_XML
from mjlab_microduck.rom.bundle import BundleBuildRequest, build_bundle


def _artifact(value: str) -> tuple[str, Path]:
    action_code, separator, path = value.partition("=")
    if not separator or not action_code or not path:
        raise argparse.ArgumentTypeError("artifact must be ACTION_CODE=PATH")
    return action_code, Path(path)


def _created_at(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError("created-at must include a UTC offset")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True)
    parser.add_argument("--artifact", action="append", default=[], type=_artifact)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--model", type=Path, default=MICRODUCK_WALK_XML.with_name("scene_walk.xml")
    )
    parser.add_argument("--terrain", choices=("flat", "ramp", "slope"))
    parser.add_argument("--scenario-profile", choices=("SEEDED_SERVO_RESET_V1",))
    parser.add_argument("--source-repository", default="microduck-rl")
    parser.add_argument("--source-commit")
    parser.add_argument(
        "--created-at", type=_created_at, default=datetime.now().astimezone()
    )
    parser.add_argument("--checkpoint")
    parser.add_argument("--experiment-ref")
    parser.add_argument("--qualification-file", action="append", default=[], type=Path)
    parser.add_argument("--software-license-id", required=True)
    parser.add_argument(
        "--software-license-file", action="append", required=True, type=Path
    )
    parser.add_argument("--model-license-id", required=True)
    parser.add_argument(
        "--model-license-status",
        required=True,
        choices=("DEVELOPMENT_ONLY", "DISTRIBUTION_CLEARED"),
    )
    parser.add_argument("--model-license-file", action="append", required=True, type=Path)
    arguments = parser.parse_args()
    source_commit = (
        arguments.source_commit
        or subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=Path(__file__).parents[1]
        ).strip()
    )
    artifacts = dict(arguments.artifact)
    if len(artifacts) != len(arguments.artifact):
        parser.error("each action code may be supplied only once")
    built = build_bundle(
        BundleBuildRequest(
            release=arguments.release,
            output_zip=arguments.output,
            artifacts=artifacts,
            model_path=arguments.model,
            source_repository=arguments.source_repository,
            source_commit=source_commit,
            created_at=arguments.created_at,
            model_terrain=arguments.terrain,
            scenario_profile=arguments.scenario_profile,
            checkpoint=arguments.checkpoint,
            experiment_ref=arguments.experiment_ref,
            qualification_files=tuple(arguments.qualification_file),
            software_license_id=arguments.software_license_id,
            software_license_files=tuple(arguments.software_license_file),
            model_license_id=arguments.model_license_id,
            model_license_status=arguments.model_license_status,
            model_license_files=tuple(arguments.model_license_file),
        )
    )
    print(f"Written {built.output_zip} ({built.manifest.bundleDigest})")


if __name__ == "__main__":
    main()

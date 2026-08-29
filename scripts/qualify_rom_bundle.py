"""Qualify a verified MicroDuck bundle and emit a new immutable release ZIP."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from mjlab_microduck.rom.qualification import (
    QualificationFailed,
    ReleaseConfiguration,
    ReleaseConfigurationError,
    qualify_and_promote,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle-dir",
        required=True,
        type=Path,
        help="extracted, verified candidate bundle directory",
    )
    parser.add_argument(
        "--release-config",
        required=True,
        type=Path,
        help="MICRODUCK_ROM_RELEASE_V1 JSON policy",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--protected-source-root",
        action="append",
        default=[],
        type=Path,
        help="additional source tree beneath which promotion output is forbidden",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        configuration = ReleaseConfiguration.model_validate_json(
            arguments.release_config.read_bytes()
        )
    except (OSError, ValidationError, ValueError, json.JSONDecodeError):
        print(
            "qualification failed: release configuration is invalid",
            file=sys.stderr,
        )
        return 2
    try:
        repository_root = Path(__file__).resolve().parents[1]
        protected_roots = (
            repository_root / "src/mjlab_microduck/robot/microduck",
            *(path.resolve() for path in arguments.protected_source_root),
        )
        promoted = qualify_and_promote(
            arguments.bundle_dir,
            arguments.output,
            configuration,
            protected_source_roots=protected_roots,
        )
    except FileExistsError:
        print("qualification failed: output already exists", file=sys.stderr)
        return 2
    except ReleaseConfigurationError as error:
        print(f"qualification failed: {error}", file=sys.stderr)
        return 2
    except QualificationFailed as error:
        print(f"qualification failed: {error}", file=sys.stderr)
        return 3
    except ValueError:
        print(
            "qualification failed: candidate bundle verification or runtime preflight failed",
            file=sys.stderr,
        )
        return 4
    except Exception:  # noqa: BLE001 - stable CLI boundary hides internal paths/data.
        print(
            "qualification failed: governed runtime execution failed",
            file=sys.stderr,
        )
        return 5
    print(
        f"Written {promoted.output_zip} "
        f"({promoted.manifest.bundleDigest}; subject {promoted.report.subjectBundleDigest})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

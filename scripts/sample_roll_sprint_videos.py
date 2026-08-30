#!/usr/bin/env python3
"""Audit and record the newest roll-sprint checkpoint on a 150-second cadence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
RECORDER = REPO_ROOT / "scripts" / "record_roll_sprint_policy.py"
EVALUATOR = REPO_ROOT / "scripts" / "evaluate_roll_sprint_checkpoint.py"
DEFAULT_CHECKPOINT_ROOT = REPO_ROOT / "logs" / "rsl_rl" / "microduck_roll_sprint"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "training" / "roll-sprint-samples"
DEFAULT_TASK_ID = "Mjlab-Roll-Sprint-Flat-MicroDuck"
DEFAULT_INTERVAL_SECONDS = 150.0
RECORDING_STEPS = 2000
RECORDING_FRAME_STRIDE = 7
EVALUATION_ENVS = 4
EVALUATION_DURATION = 40.0
STATE_FILENAME = ".sample-roll-sprint-videos.json"
CHECKPOINT_PATTERN = re.compile(r"model_(\d+)\.pt$")


@dataclass(frozen=True)
class CheckpointIdentity:
    path: str
    iteration: int
    size: int
    mtime_ns: int
    sha256: str


def _log(message: str) -> None:
    timestamp = datetime.now(UTC).isoformat(timespec="seconds")
    print(f"[roll-video-sampler {timestamp}] {message}", flush=True)


def _checkpoint_iteration(path: Path) -> int | None:
    match = CHECKPOINT_PATTERN.fullmatch(path.name)
    return int(match.group(1)) if match else None


def find_newest_checkpoint(root: Path) -> Path | None:
    if not root.is_dir():
        return None
    candidates: list[tuple[int, int, str, Path]] = []
    for path in root.rglob("model_*.pt"):
        iteration = _checkpoint_iteration(path)
        if iteration is None or not path.is_file():
            continue
        stat = path.stat()
        candidates.append((stat.st_mtime_ns, iteration, str(path), path))
    return max(candidates)[-1] if candidates else None


def checkpoint_identity(path: Path) -> CheckpointIdentity:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"checkpoint changed while hashing: {path}")
    iteration = _checkpoint_iteration(path)
    if iteration is None:
        raise ValueError(f"unsupported checkpoint filename: {path.name}")
    return CheckpointIdentity(
        path=str(path.resolve()),
        iteration=iteration,
        size=after.st_size,
        mtime_ns=after.st_mtime_ns,
        sha256=digest.hexdigest(),
    )


def _load_state(state_file: Path) -> dict[str, Any]:
    if not state_file.exists():
        return {}
    try:
        state = json.loads(state_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"cannot read sampler state {state_file}: {error}"
        ) from error
    if not isinstance(state, dict):
        raise TypeError(f"sampler state must be a JSON object: {state_file}")
    return state


def _write_state(state_file: Path, state: dict[str, Any]) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = state_file.with_name(f".{state_file.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, state_file)


def _is_duplicate(identity: CheckpointIdentity, state: dict[str, Any]) -> bool:
    previous = state.get("last_checkpoint")
    if not isinstance(previous, dict):
        return False
    return (
        previous.get("path") == identity.path
        or previous.get("sha256") == identity.sha256
    )


def _output_path(output_dir: Path, identity: CheckpointIdentity) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    return output_dir / f"{timestamp}-checkpoint-{identity.iteration:06d}.mp4"


def _evaluation_path(output_dir: Path, identity: CheckpointIdentity) -> Path:
    return (
        output_dir
        / "evaluations"
        / f"checkpoint-{identity.iteration:06d}-{identity.sha256[:12]}.json"
    )


def _evaluator_command(*, checkpoint: Path, output: Path, device: str) -> list[str]:
    return [
        sys.executable,
        str(EVALUATOR),
        str(checkpoint),
        "--num-envs",
        str(EVALUATION_ENVS),
        "--duration",
        f"{EVALUATION_DURATION:g}",
        "--device",
        device,
        "--output",
        str(output),
    ]


def _recorder_command(
    *, checkpoint: Path, output: Path, task_id: str, device: str
) -> list[str]:
    return [
        sys.executable,
        str(RECORDER),
        str(checkpoint),
        str(output),
        "--task-id",
        task_id,
        "--steps",
        str(RECORDING_STEPS),
        "--frame-stride",
        str(RECORDING_FRAME_STRIDE),
        "--device",
        device,
    ]


def sample_once(args: argparse.Namespace) -> bool:
    checkpoint = find_newest_checkpoint(args.checkpoint_root)
    if checkpoint is None:
        _log(f"no model_*.pt checkpoint found under {args.checkpoint_root}")
        return False
    try:
        identity = checkpoint_identity(checkpoint)
    except (OSError, RuntimeError, ValueError) as error:
        _log(f"checkpoint not ready, skipping this interval: {error}")
        return False
    state = _load_state(args.state_file)
    if not args.allow_repeats and _is_duplicate(identity, state):
        _log(f"unchanged checkpoint, no recording: {checkpoint}")
        return False

    evaluation = _evaluation_path(args.output_dir, identity)
    if not args.video_only and not evaluation.is_file():
        evaluation.parent.mkdir(parents=True, exist_ok=True)
        evaluation_command = _evaluator_command(
            checkpoint=checkpoint,
            output=evaluation,
            device=args.device,
        )
        _log(
            f"auditing checkpoint iteration {identity.iteration} on "
            f"{EVALUATION_ENVS} standing starts"
        )
        evaluation_result = subprocess.run(
            evaluation_command, cwd=REPO_ROOT, check=False
        )
        if evaluation_result.returncode != 0:
            _log(
                f"evaluator failed with exit code {evaluation_result.returncode}; "
                "state unchanged"
            )
            return False
        if not evaluation.is_file():
            _log("evaluator returned success without JSON; state unchanged")
            return False
    elif not args.video_only:
        _log(f"reusing completed audit: {evaluation}")

    if args.audit_only:
        return True

    output = _output_path(args.output_dir, identity)
    command = _recorder_command(
        checkpoint=checkpoint,
        output=output,
        task_id=args.task_id,
        device=args.device,
    )
    _log(
        f"recording checkpoint iteration {identity.iteration} for "
        f"{RECORDING_STEPS / 50:g}s to {output}"
    )
    result = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if result.returncode != 0:
        _log(f"recorder failed with exit code {result.returncode}; state unchanged")
        return False
    if not output.is_file():
        _log("recorder returned success without an output video; state unchanged")
        return False
    _write_state(
        args.state_file,
        {
            "version": 2,
            "last_checkpoint": asdict(identity),
            "last_evaluation": (
                str(evaluation.resolve()) if evaluation.is_file() else None
            ),
            "last_video": str(output.resolve()),
            "sampled_at_utc": datetime.now(UTC).isoformat(),
        },
    )
    _log(f"sample complete: {output}")
    return True


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--state-file", type=Path)
    parser.add_argument(
        "--interval-seconds", type=float, default=DEFAULT_INTERVAL_SECONDS
    )
    parser.add_argument("--task-id", default=DEFAULT_TASK_ID)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-repeats", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--video-only",
        action="store_true",
        help="Keep the video cadence independent from long checkpoint audits.",
    )
    mode.add_argument(
        "--audit-only",
        action="store_true",
        help="Audit each unique checkpoint without recording a video.",
    )
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    args.checkpoint_root = args.checkpoint_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.state_file = (
        args.state_file.expanduser().resolve()
        if args.state_file is not None
        else args.output_dir / STATE_FILENAME
    )
    if args.interval_seconds <= 0:
        parser.error("--interval-seconds must be greater than zero")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.audit_only and not RECORDER.is_file():
        raise SystemExit(f"Recorder not found: {RECORDER}")
    if not args.video_only and not EVALUATOR.is_file():
        raise SystemExit(f"Evaluator not found: {EVALUATOR}")
    if not args.checkpoint_root.is_dir():
        raise SystemExit(f"Checkpoint root not found: {args.checkpoint_root}")
    _log(
        f"watching {args.checkpoint_root} every {args.interval_seconds:g}s; "
        f"samples stay in {args.output_dir}"
    )
    while True:
        started = time.monotonic()
        sample_once(args)
        if args.once:
            return 0
        remaining = args.interval_seconds - (time.monotonic() - started)
        if remaining > 0:
            time.sleep(remaining)
        else:
            _log(
                "sampling exceeded the interval; continuing sequentially without overlap"
            )


if __name__ == "__main__":
    raise SystemExit(main())

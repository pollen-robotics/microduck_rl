#!/usr/bin/env python3
"""Periodically record the newest stair-training checkpoint without overlap."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
RECORDER = REPO_ROOT / "scripts" / "record_stair_policy.py"
DEFAULT_CHECKPOINT_ROOT = REPO_ROOT / "logs"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "training" / "stair-policy-samples"
DEFAULT_WALKER_CHECKPOINT = (
    REPO_ROOT
    / "logs"
    / "rsl_rl"
    / "velocity"
    / "2026-08-29_08-50-53_base_walk"
    / "model_500.pt"
)
DEFAULT_TASK_ID = "Mjlab-Stairs-Route-MicroDuck"
DEFAULT_INTERVAL_SECONDS = 150.0
RECORDING_STEPS = 1000
STATE_FILENAME = ".sample-stair-training-videos.json"
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
    print(f"[stair-video-sampler {timestamp}] {message}", flush=True)


def _checkpoint_iteration(path: Path) -> int | None:
    match = CHECKPOINT_PATTERN.fullmatch(path.name)
    return int(match.group(1)) if match else None


def find_newest_checkpoint(root: Path) -> Path | None:
    """Return the most recently modified complete checkpoint candidate."""
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
    """Hash a stable checkpoint, rejecting a file that changes during hashing."""
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
        raise RuntimeError(f"cannot read sampler state {state_file}: {error}") from error
    if not isinstance(state, dict):
        raise RuntimeError(f"sampler state must be a JSON object: {state_file}")
    return state


def _write_state(state_file: Path, state: dict[str, Any]) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = state_file.with_name(f".{state_file.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
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


def _recorder_command(
    *,
    checkpoint: Path,
    output: Path,
    task_id: str,
    walker_checkpoint: Path | None,
    device: str,
) -> list[str]:
    command = [
        sys.executable,
        str(RECORDER),
        str(checkpoint),
        str(output),
        "--task-id",
        task_id,
        "--steps",
        str(RECORDING_STEPS),
        "--device",
        device,
    ]
    if walker_checkpoint is not None:
        command.extend(["--walker-checkpoint", str(walker_checkpoint)])
    return command


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

    output = _output_path(args.output_dir, identity)
    command = _recorder_command(
        checkpoint=checkpoint,
        output=output,
        task_id=args.task_id,
        walker_checkpoint=args.walker_checkpoint,
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

    state = {
        "version": 1,
        "last_checkpoint": asdict(identity),
        "last_video": str(output.resolve()),
        "sampled_at_utc": datetime.now(UTC).isoformat(),
    }
    _write_state(args.state_file, state)
    _log(f"sample complete: {output}")
    return True


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=DEFAULT_CHECKPOINT_ROOT,
        help="Run or log directory searched recursively for model_*.pt files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory that retains every successful 20-second sample.",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        help="Restart state path. Defaults to a JSON file in --output-dir.",
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=DEFAULT_INTERVAL_SECONDS,
        help="Seconds between sampling opportunities (default: 150).",
    )
    parser.add_argument("--task-id", default=DEFAULT_TASK_ID)
    parser.add_argument(
        "--walker-checkpoint",
        type=Path,
        default=DEFAULT_WALKER_CHECKPOINT,
        help="Manufacturer walking checkpoint passed to the route recorder.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Recorder device. CPU is the resource-bounded default.",
    )
    parser.add_argument(
        "--allow-repeats",
        action="store_true",
        help="Record again even when checkpoint path or content is unchanged.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Perform one sampling opportunity and exit.",
    )
    args = parser.parse_args(argv)
    args.checkpoint_root = args.checkpoint_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.state_file = (
        args.state_file.expanduser().resolve()
        if args.state_file is not None
        else args.output_dir / STATE_FILENAME
    )
    args.walker_checkpoint = (
        args.walker_checkpoint.expanduser().resolve()
        if args.walker_checkpoint is not None
        else None
    )
    if args.interval_seconds <= 0:
        parser.error("--interval-seconds must be greater than zero")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if not RECORDER.is_file():
        raise SystemExit(f"Recorder not found: {RECORDER}")
    if not args.checkpoint_root.is_dir():
        raise SystemExit(f"Checkpoint root not found: {args.checkpoint_root}")
    if args.walker_checkpoint is not None and not args.walker_checkpoint.is_file():
        raise SystemExit(f"Walker checkpoint not found: {args.walker_checkpoint}")

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
            _log("sampling exceeded the interval; continuing sequentially without overlap")


if __name__ == "__main__":
    raise SystemExit(main())

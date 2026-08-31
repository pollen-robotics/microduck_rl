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
SELECTOR = REPO_ROOT / "scripts" / "select_best_roll_sprint_checkpoint.py"
DEFAULT_CHECKPOINT_ROOT = REPO_ROOT / "logs" / "rsl_rl" / "microduck_roll_sprint"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "training" / "roll-sprint-samples"
DEFAULT_TASK_ID = "Mjlab-Roll-Sprint-Flat-MicroDuck"
DEFAULT_INTERVAL_SECONDS = 150.0
RECORDING_STEPS = 2000
RECOVERY_RECORDING_STEPS = 600
SIMULATION_HZ = 50
# Compress the full 40 s race horizon into a readable 20 s clip while leaving
# enough wall-clock margin to start a fresh clip every 150 s.
OUTPUT_FPS = 10.0
OUTPUT_VIDEO_SECONDS = 20.0
RECORDING_FRAME_STRIDE = round(
    RECORDING_STEPS / (OUTPUT_FPS * OUTPUT_VIDEO_SECONDS)
)
EVALUATION_ENVS = 4
EVALUATION_DURATION = 40.0
EVALUATION_SCHEMA_VERSION = 8
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
        / (
            f"checkpoint-{identity.iteration:06d}-{identity.sha256[:12]}"
            f"-race-{EVALUATION_DURATION:g}s-v{EVALUATION_SCHEMA_VERSION}.json"
        )
    )


def _recovery_montage_path(
    output_dir: Path, identity: CheckpointIdentity
) -> Path:
    return (
        output_dir
        / "recovery-montages"
        / f"checkpoint-{identity.iteration:06d}-{identity.sha256[:12]}.mp4"
    )


def _evaluator_command(
    *,
    checkpoint: Path,
    output: Path,
    device: str,
    parent_frontier_m: float | None = None,
) -> list[str]:
    command = [
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
    if parent_frontier_m is not None:
        command.extend(("--parent-frontier-m", f"{parent_frontier_m:g}"))
    return command


def _recorder_command(
    *,
    checkpoint: Path,
    output: Path,
    task_id: str,
    device: str,
    recovery_montage: bool = False,
) -> list[str]:
    command = [
        sys.executable,
        str(RECORDER),
        str(checkpoint),
        str(output),
        "--task-id",
        task_id,
        "--steps",
        str(RECOVERY_RECORDING_STEPS if recovery_montage else RECORDING_STEPS),
        "--frame-stride",
        str(RECORDING_FRAME_STRIDE),
        "--device",
        device,
    ]
    if recovery_montage:
        command.append("--recovery-montage")
    return command


def _select_champion(*, output_dir: Path, champion_dir: Path) -> bool:
    result = subprocess.run(
        [
            sys.executable,
            str(SELECTOR),
            "--evaluation-dir",
            str(output_dir / "evaluations"),
            "--champion-dir",
            str(champion_dir),
        ],
        cwd=REPO_ROOT,
        check=False,
    )
    if result.returncode != 0:
        _log(f"champion selection failed with exit code {result.returncode}")
        return False
    return True


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
    skip_evaluation = args.video_only or args.montage_only
    if not skip_evaluation and not evaluation.is_file():
        evaluation.parent.mkdir(parents=True, exist_ok=True)
        evaluation_command = _evaluator_command(
            checkpoint=checkpoint,
            output=evaluation,
            device=args.device,
            parent_frontier_m=args.parent_frontier_m,
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
    elif not skip_evaluation:
        _log(f"reusing completed audit: {evaluation}")

    if (
        not skip_evaluation
        and args.champion_dir is not None
        and not _select_champion(
            output_dir=args.output_dir,
            champion_dir=args.champion_dir,
        )
    ):
        return False

    if args.audit_only:
        _write_state(
            args.state_file,
            {
                "version": 4,
                "last_checkpoint": asdict(identity),
                "last_evaluation": str(evaluation.resolve()),
                "last_video": None,
                "last_recovery_montage": None,
                "sampled_at_utc": datetime.now(UTC).isoformat(),
            },
        )
        return True

    recovery_montage = _recovery_montage_path(args.output_dir, identity)
    if args.montage_only:
        if not recovery_montage.is_file():
            recovery_montage.parent.mkdir(parents=True, exist_ok=True)
            recovery_command = _recorder_command(
                checkpoint=checkpoint,
                output=recovery_montage,
                task_id=args.task_id,
                device=args.device,
                recovery_montage=True,
            )
            _log(
                "recording one recovery montage for checkpoint iteration "
                f"{identity.iteration}: {recovery_montage}"
            )
            recovery_result = subprocess.run(
                recovery_command, cwd=REPO_ROOT, check=False
            )
            if recovery_result.returncode != 0:
                _log(
                    "recovery montage failed with exit code "
                    f"{recovery_result.returncode}; state unchanged"
                )
                return False
            if not recovery_montage.is_file():
                _log("recovery recorder returned success without video; state unchanged")
                return False
        else:
            _log(f"reusing unique recovery montage: {recovery_montage}")
        _write_state(
            args.state_file,
            {
                "version": 3,
                "last_checkpoint": asdict(identity),
                "last_evaluation": None,
                "last_video": None,
                "last_recovery_montage": str(recovery_montage.resolve()),
                "sampled_at_utc": datetime.now(UTC).isoformat(),
            },
        )
        _log(f"recovery montage complete: {recovery_montage}")
        return True

    output = _output_path(args.output_dir, identity)
    command = _recorder_command(
        checkpoint=checkpoint,
        output=output,
        task_id=args.task_id,
        device=args.device,
    )
    _log(
        f"recording checkpoint iteration {identity.iteration}: "
        f"{RECORDING_STEPS / SIMULATION_HZ:g}s simulation as a "
        f"{OUTPUT_VIDEO_SECONDS:g}s video to {output}"
    )
    result = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if result.returncode != 0:
        _log(f"recorder failed with exit code {result.returncode}; state unchanged")
        return False
    if not output.is_file():
        _log("recorder returned success without an output video; state unchanged")
        return False

    if not args.video_only and not recovery_montage.is_file():
        recovery_montage.parent.mkdir(parents=True, exist_ok=True)
        recovery_command = _recorder_command(
            checkpoint=checkpoint,
            output=recovery_montage,
            task_id=args.task_id,
            device=args.device,
            recovery_montage=True,
        )
        _log(
            "recording one recovery montage for checkpoint iteration "
            f"{identity.iteration}: {recovery_montage}"
        )
        recovery_result = subprocess.run(
            recovery_command, cwd=REPO_ROOT, check=False
        )
        if recovery_result.returncode != 0:
            _log(
                "recovery montage failed with exit code "
                f"{recovery_result.returncode}; state unchanged"
            )
            return False
        if not recovery_montage.is_file():
            _log("recovery recorder returned success without video; state unchanged")
            return False
    elif not args.video_only:
        _log(f"reusing unique recovery montage: {recovery_montage}")
    _write_state(
        args.state_file,
        {
            "version": 3,
            "last_checkpoint": asdict(identity),
            "last_evaluation": (
                str(evaluation.resolve()) if evaluation.is_file() else None
            ),
            "last_video": str(output.resolve()),
            "last_recovery_montage": (
                str(recovery_montage.resolve())
                if recovery_montage.is_file()
                else None
            ),
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
        "--champion-dir",
        type=Path,
        help="Keep exactly one best eligible audited checkpoint in this directory.",
    )
    parser.add_argument(
        "--interval-seconds", type=float, default=DEFAULT_INTERVAL_SECONDS
    )
    parser.add_argument("--task-id", default=DEFAULT_TASK_ID)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--parent-frontier-m",
        type=float,
        help="Selected-parent frontier used for the evaluator's 90%% retention gate.",
    )
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
    mode.add_argument(
        "--montage-only",
        action="store_true",
        help="Record one recovery montage per unique checkpoint on its own cadence.",
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
    args.champion_dir = (
        args.champion_dir.expanduser().resolve()
        if args.champion_dir is not None
        else None
    )
    if args.interval_seconds <= 0:
        parser.error("--interval-seconds must be greater than zero")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.audit_only and not RECORDER.is_file():
        raise SystemExit(f"Recorder not found: {RECORDER}")
    if not args.video_only and not args.montage_only and not EVALUATOR.is_file():
        raise SystemExit(f"Evaluator not found: {EVALUATOR}")
    if args.champion_dir is not None and not SELECTOR.is_file():
        raise SystemExit(f"Champion selector not found: {SELECTOR}")
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

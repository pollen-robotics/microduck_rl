#!/usr/bin/env python3
"""Independently audit and record each new grounded-backroll checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = REPO_ROOT / "scripts" / "evaluate_backroll_checkpoint.py"
RECORDER = REPO_ROOT / "scripts" / "record_backroll_policy.py"
CHECKPOINT_PREFIX = "model_"
STATE_FILENAME = ".sample-backroll-videos.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=600.0)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def _log(message: str) -> None:
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    print(f"[backroll-video-sampler {stamp}] {message}", flush=True)


def _iteration(path: Path) -> int | None:
    if path.suffix != ".pt" or not path.stem.startswith(CHECKPOINT_PREFIX):
        return None
    try:
        return int(path.stem.removeprefix(CHECKPOINT_PREFIX))
    except ValueError:
        return None


def _newest_checkpoint(run_dir: Path) -> Path | None:
    candidates = [
        (iteration, path.stat().st_mtime_ns, path)
        for path in run_dir.glob("model_*.pt")
        if (iteration := _iteration(path)) is not None and path.is_file()
    ]
    return max(candidates)[2] if candidates else None


def _sha256(path: Path) -> str:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"checkpoint changed while hashing: {path}")
    return digest.hexdigest()


def _load_state(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid sampler state {path}: {error}") from error
    if not isinstance(payload, dict):
        raise TypeError(f"sampler state must be a JSON object: {path}")
    return payload


def _write_state(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _output_paths(
    output_dir: Path,
    checkpoint: Path,
    digest: str,
) -> tuple[Path, Path]:
    iteration = _iteration(checkpoint)
    if iteration is None:
        raise ValueError(f"unsupported checkpoint name: {checkpoint.name}")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    stem = f"{stamp}-checkpoint-{iteration:06d}-{digest[:12]}"
    return output_dir / f"{stem}.mp4", output_dir / "evaluations" / f"{stem}.json"


def _run(command: list[str]) -> None:
    _log("running " + " ".join(command[1:4]))
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def _sample(
    checkpoint: Path,
    output_dir: Path,
    device: str,
    duration: float,
) -> dict[str, str | int]:
    digest = _sha256(checkpoint)
    video, evaluation = _output_paths(output_dir, checkpoint, digest)
    _run(
        [
            sys.executable,
            str(EVALUATOR),
            str(checkpoint),
            "--output",
            str(evaluation),
            "--device",
            device,
        ]
    )
    _run(
        [
            sys.executable,
            str(RECORDER),
            str(checkpoint),
            str(video),
            "--device",
            device,
            "--duration",
            f"{duration:g}",
            "--allow-incomplete-diagnostic",
        ]
    )
    return {
        "checkpoint": str(checkpoint.resolve()),
        "iteration": _iteration(checkpoint) or 0,
        "sha256": digest,
        "evaluation": str(evaluation.resolve()),
        "video": str(video.resolve()),
    }


def main() -> int:
    args = _parse_args()
    if args.interval_seconds <= 0.0 or args.duration <= 0.0:
        raise SystemExit("--interval-seconds and --duration must be positive")
    run_dir = args.run_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    state_path = output_dir / STATE_FILENAME
    _log(f"watching {run_dir}")
    while True:
        try:
            checkpoint = _newest_checkpoint(run_dir)
            state = _load_state(state_path)
            if checkpoint is not None:
                digest = _sha256(checkpoint)
                previous = state.get("last_checkpoint")
                if not isinstance(previous, dict) or previous.get("sha256") != digest:
                    result = _sample(checkpoint, output_dir, args.device, args.duration)
                    _write_state(state_path, {"last_checkpoint": result})
                    _log(f"published {result['video']}")
        except Exception as error:  # noqa: BLE001 - watchdog must survive one bad checkpoint.
            _log(f"checkpoint sample failed: {error}")
        if args.once:
            return 0
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())

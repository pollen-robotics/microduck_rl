#!/usr/bin/env python3
"""Retain exactly one independently audited roll-race champion checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

SCHEMA_VERSION = 8
TARGET_DISTANCE_M = 10.0


@dataclass(frozen=True)
class Candidate:
    evaluation: Path
    checkpoint: Path
    checkpoint_sha256: str
    report: dict[str, object]
    rank: tuple[float, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finish_times(report: dict[str, object]) -> list[float]:
    robots = report.get("per_robot")
    if not isinstance(robots, list):
        return []
    times: list[float] = []
    for robot in robots:
        if not isinstance(robot, dict) or not robot.get("target_10m_pass"):
            continue
        value = robot.get("time_to_valid_10m_s")
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            times.append(float(value))
    return times


def checkpoint_rank(report: dict[str, object]) -> tuple[float, ...]:
    """Rank completion first, then distance, then speed after all four finish."""

    finish_count = int(report.get("target_distance_reach_count", 0))
    mean_credit = float(report.get("mean_credited_forward_frontier_m", 0.0))
    recovery = report.get("recovery_battery")
    recovery_rate = (
        float(recovery.get("self_right_then_reroll_rate", 0.0))
        if isinstance(recovery, dict)
        else 0.0
    )
    heading_deviation = float(
        report.get("maximum_heading_yaw_deviation_deg", math.inf)
    )
    times = _finish_times(report)
    mean_time = sum(times) / len(times) if times else math.inf
    slowest_time = max(times, default=math.inf)
    if finish_count >= 4:
        return (
            2.0,
            -slowest_time,
            -mean_time,
            mean_credit,
            recovery_rate,
            -heading_deviation,
        )
    return (
        1.0 if finish_count > 0 else 0.0,
        float(finish_count),
        mean_credit,
        -mean_time,
        recovery_rate,
        -heading_deviation,
    )


def _eligible(report: dict[str, object]) -> bool:
    alignment = report.get("canonical_race_alignment")
    return bool(
        int(report.get("schema_version", 0)) == SCHEMA_VERSION
        and int(report.get("num_envs", 0)) == 4
        and math.isclose(float(report.get("target_distance_m", 0.0)), TARGET_DISTANCE_M)
        and isinstance(alignment, dict)
        and alignment.get("alignment_pass") is True
        and int(report.get("nan_env_count", 0)) == 0
        and int(report.get("out_of_bounds_env_count", 0)) == 0
        and int(report.get("road_exit_env_count", 0)) == 0
        and float(report.get("maximum_road_boundary_overshoot_m", math.inf)) <= 0.0
    )


def load_candidate(evaluation: Path) -> Candidate | None:
    try:
        report = json.loads(evaluation.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(report, dict) or not _eligible(report):
        return None
    checkpoint_value = report.get("checkpoint")
    expected_sha = report.get("checkpoint_sha256")
    if not isinstance(checkpoint_value, str) or not isinstance(expected_sha, str):
        return None
    checkpoint = Path(checkpoint_value).expanduser().resolve()
    if not checkpoint.is_file() or _sha256(checkpoint) != expected_sha:
        return None
    return Candidate(
        evaluation=evaluation.resolve(),
        checkpoint=checkpoint,
        checkpoint_sha256=expected_sha,
        report=report,
        rank=checkpoint_rank(report),
    )


def select_best(evaluation_dir: Path) -> Candidate | None:
    candidates = (
        candidate
        for path in evaluation_dir.glob("*.json")
        if (candidate := load_candidate(path)) is not None
    )
    return max(candidates, key=lambda candidate: candidate.rank, default=None)


def retain(candidate: Candidate, champion_dir: Path) -> tuple[Path, Path]:
    champion_dir.mkdir(parents=True, exist_ok=True)
    iteration = int(candidate.report.get("checkpoint_iteration", 0))
    destination = champion_dir / f"model_{iteration}.pt"
    temporary = champion_dir / f".{destination.name}.{os.getpid()}.tmp"
    shutil.copy2(candidate.checkpoint, temporary)
    if _sha256(temporary) != candidate.checkpoint_sha256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("retained checkpoint hash does not match audited source")
    os.replace(temporary, destination)
    for old_checkpoint in champion_dir.glob("model_*.pt"):
        if old_checkpoint != destination:
            old_checkpoint.unlink()

    manifest = champion_dir / "champion.json"
    payload = {
        "schema_version": 1,
        "evaluation_schema_version": SCHEMA_VERSION,
        "source_checkpoint": str(candidate.checkpoint),
        "retained_checkpoint": str(destination.resolve()),
        "checkpoint_sha256": candidate.checkpoint_sha256,
        "evaluation": str(candidate.evaluation),
        "rank": list(candidate.rank),
        "target_distance_reach_count": int(
            candidate.report.get("target_distance_reach_count", 0)
        ),
        "mean_credited_forward_frontier_m": float(
            candidate.report.get("mean_credited_forward_frontier_m", 0.0)
        ),
        "mean_time_to_valid_10m_s": candidate.report.get(
            "mean_time_to_valid_10m_s"
        ),
        "slowest_time_to_valid_10m_s": candidate.report.get(
            "slowest_time_to_valid_10m_s"
        ),
    }
    with NamedTemporaryFile(
        "w", encoding="utf-8", suffix=".tmp", dir=champion_dir, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary_manifest = Path(handle.name)
    os.replace(temporary_manifest, manifest)
    return destination, manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--champion-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    evaluation_dir = args.evaluation_dir.expanduser().resolve()
    champion_dir = args.champion_dir.expanduser().resolve()
    if not evaluation_dir.is_dir():
        raise SystemExit(f"Evaluation directory not found: {evaluation_dir}")
    candidate = select_best(evaluation_dir)
    if candidate is None:
        raise SystemExit("No eligible schema-v8 checkpoint audit found")
    destination, manifest = retain(candidate, champion_dir)
    print(f"[roll-champion] retained {destination}")
    print(f"[roll-champion] manifest {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

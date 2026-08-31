#!/usr/bin/env python3
"""Audit every checkpoint in one completed roll-sprint run, in numeric order."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = REPO_ROOT / "scripts" / "evaluate_roll_sprint_checkpoint.py"
SELECTOR = REPO_ROOT / "scripts" / "select_best_roll_sprint_checkpoint.py"
SCHEMA_VERSION = 8
NUM_ENVS = 4
RACE_DURATION_S = 40.0
TARGET_DISTANCE_M = 10.0
TASK_ID = "Mjlab-Roll-Sprint-Flat-MicroDuck"
CHECKPOINT_PATTERN = re.compile(r"model_(\d+)\.pt$")
NO_ELIGIBLE_CHAMPION = "No eligible schema-v8 checkpoint audit found"


class AuditRunError(RuntimeError):
    """A checkpoint series could not be audited safely."""


@dataclass(frozen=True)
class CheckpointIdentity:
    path: Path
    iteration: int
    sha256: str


@dataclass(frozen=True)
class AuditResult:
    checkpoint: CheckpointIdentity
    evaluation: Path
    reused: bool
    report: dict[str, Any]


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


def _checkpoint_iteration(path: Path) -> int | None:
    match = CHECKPOINT_PATTERN.fullmatch(path.name)
    return int(match.group(1)) if match else None


def find_checkpoints(run_dir: Path) -> list[Path]:
    """Return direct run checkpoints in numeric iteration order."""

    checkpoints = []
    for path in run_dir.glob("model_*.pt"):
        iteration = _checkpoint_iteration(path)
        if iteration is not None and path.is_file():
            checkpoints.append((iteration, path.name, path.resolve()))
    checkpoints.sort()
    return [path for _, _, path in checkpoints]


def checkpoint_identity(path: Path) -> CheckpointIdentity:
    """Hash a stable checkpoint, rejecting a file that changes while read."""

    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise AuditRunError(f"checkpoint changed while hashing: {path}")
    iteration = _checkpoint_iteration(path)
    if iteration is None:
        raise AuditRunError(f"unsupported checkpoint filename: {path.name}")
    return CheckpointIdentity(path.resolve(), iteration, digest.hexdigest())


def _same_path(left: str, right: Path) -> bool:
    try:
        resolved_left = Path(left).expanduser().resolve()
    except (OSError, RuntimeError):
        return False
    return os.path.normcase(str(resolved_left)) == os.path.normcase(str(right.resolve()))


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _valid_robot_result(robot: object) -> bool:
    if not isinstance(robot, dict) or not isinstance(robot.get("target_10m_pass"), bool):
        return False
    finish_time = robot.get("time_to_valid_10m_s")
    if robot["target_10m_pass"]:
        return _finite_number(finish_time) and 0.0 <= float(finish_time) <= RACE_DURATION_S
    return finish_time is None


def valid_audit_report(
    report: object, identity: CheckpointIdentity
) -> dict[str, Any] | None:
    """Validate a complete canonical v8 report for one exact checkpoint.

    Physics failures remain valid completed audits. Champion eligibility is the
    selector's responsibility and must not cause an expensive audit to repeat.
    """

    if not isinstance(report, dict):
        return None
    try:
        schema_version = int(report.get("schema_version", 0))
        checkpoint_iteration = int(report.get("checkpoint_iteration", -1))
        num_envs = int(report.get("num_envs", 0))
        finish_count = int(report.get("target_distance_reach_count", -1))
    except (TypeError, ValueError):
        return None
    checkpoint = report.get("checkpoint")
    per_robot = report.get("per_robot")
    alignment = report.get("canonical_race_alignment")
    recovery = report.get("recovery_battery")
    if not (
        schema_version == SCHEMA_VERSION
        and report.get("task") == TASK_ID
        and checkpoint_iteration == identity.iteration
        and num_envs == NUM_ENVS
        and 0 <= finish_count <= NUM_ENVS
        and isinstance(checkpoint, str)
        and _same_path(checkpoint, identity.path)
        and report.get("checkpoint_sha256") == identity.sha256
        and _finite_number(report.get("duration_s"))
        and math.isclose(float(report["duration_s"]), RACE_DURATION_S)
        and _finite_number(report.get("target_distance_m"))
        and math.isclose(float(report["target_distance_m"]), TARGET_DISTANCE_M)
        and _finite_number(report.get("mean_credited_forward_frontier_m"))
        and isinstance(per_robot, list)
        and len(per_robot) == NUM_ENVS
        and all(_valid_robot_result(robot) for robot in per_robot)
        and sum(robot.get("target_10m_pass") is True for robot in per_robot)
        == finish_count
        and isinstance(alignment, dict)
        and isinstance(alignment.get("alignment_pass"), bool)
        and isinstance(recovery, dict)
        and isinstance(report.get("promotion_pass"), bool)
    ):
        return None
    return report


def load_matching_audit(
    evaluation_dir: Path, identity: CheckpointIdentity
) -> tuple[Path, dict[str, Any]] | None:
    """Find a valid matching report regardless of its filename."""

    if not evaluation_dir.is_dir():
        return None
    for path in sorted(evaluation_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        report = valid_audit_report(payload, identity)
        if report is not None:
            return path.resolve(), report
    return None


def _safe_run_name(run_dir: Path) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", run_dir.name).strip("-._")
    slug = (slug or "run")[:48]
    path_hash = hashlib.sha256(str(run_dir.resolve()).encode()).hexdigest()[:8]
    return f"{slug}-{path_hash}"


def evaluation_path(
    evaluation_dir: Path, run_dir: Path, identity: CheckpointIdentity
) -> Path:
    return evaluation_dir / (
        f"{_safe_run_name(run_dir)}-checkpoint-{identity.iteration:06d}-"
        f"{identity.sha256[:12]}-race-{RACE_DURATION_S:g}s-v{SCHEMA_VERSION}.json"
    )


def evaluator_command(
    identity: CheckpointIdentity,
    output: Path,
    *,
    device: str,
    parent_frontier_m: float | None,
) -> list[str]:
    command = [
        sys.executable,
        str(EVALUATOR),
        str(identity.path),
        "--num-envs",
        str(NUM_ENVS),
        "--duration",
        f"{RACE_DURATION_S:g}",
        "--device",
        device,
        "--output",
        str(output),
    ]
    if parent_frontier_m is not None:
        command.extend(("--parent-frontier-m", f"{parent_frontier_m:g}"))
    return command


def selector_command(evaluation_dir: Path, champion_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(SELECTOR),
        "--evaluation-dir",
        str(evaluation_dir),
        "--champion-dir",
        str(champion_dir),
    ]


def _run_captured(
    command: list[str], *, runner: CommandRunner
) -> subprocess.CompletedProcess[str]:
    return runner(
        command,
        cwd=REPO_ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _audit_checkpoint(
    *,
    run_dir: Path,
    evaluation_dir: Path,
    identity: CheckpointIdentity,
    device: str,
    parent_frontier_m: float | None,
    runner: CommandRunner,
) -> AuditResult:
    existing = load_matching_audit(evaluation_dir, identity)
    if existing is not None:
        path, report = existing
        return AuditResult(identity, path, True, report)

    output = evaluation_path(evaluation_dir, run_dir, identity)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = _run_captured(
        evaluator_command(
            identity,
            output,
            device=device,
            parent_frontier_m=parent_frontier_m,
        ),
        runner=runner,
    )
    if result.returncode != 0:
        detail = result.stdout.strip()
        if detail:
            print(detail, file=sys.stderr)
        raise AuditRunError(
            f"evaluator failed for {identity.path.name} with exit code "
            f"{result.returncode}"
        )
    try:
        payload = json.loads(output.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AuditRunError(
            f"evaluator did not write readable JSON for {identity.path.name}: {output}"
        ) from error
    report = valid_audit_report(payload, identity)
    if report is None:
        raise AuditRunError(
            f"evaluator wrote an invalid schema-v{SCHEMA_VERSION} report: {output}"
        )
    return AuditResult(identity, output.resolve(), False, report)


def useful_score(report: dict[str, Any]) -> tuple[str, float, str]:
    """Return the selector-relevant scalar that is easiest to interpret."""

    finish_count = int(report["target_distance_reach_count"])
    if finish_count == NUM_ENVS and _finite_number(
        report.get("slowest_time_to_valid_10m_s")
    ):
        return (
            "slowest_time_to_valid_10m_s",
            float(report["slowest_time_to_valid_10m_s"]),
            "lower is better",
        )
    return (
        "mean_credited_forward_frontier_m",
        float(report["mean_credited_forward_frontier_m"]),
        "higher is better",
    )


def print_audit_summary(result: AuditResult) -> None:
    report = result.report
    finish_count = int(report["target_distance_reach_count"])
    score_name, score, direction = useful_score(report)
    disposition = "reused" if result.reused else "audited"
    print(
        f"[roll-audit-series] iteration={result.checkpoint.iteration} "
        f"status={disposition} 10m_finishers={finish_count}/{NUM_ENVS} "
        f"useful_score={score:.6g} ({score_name}; {direction})",
        flush=True,
    )


def _select_champion(
    evaluation_dir: Path,
    champion_dir: Path,
    *,
    runner: CommandRunner,
) -> bool:
    champion_dir.mkdir(parents=True, exist_ok=True)
    result = _run_captured(
        selector_command(evaluation_dir, champion_dir),
        runner=runner,
    )
    if result.returncode == 0:
        return True
    detail = result.stdout.strip()
    if NO_ELIGIBLE_CHAMPION in detail:
        print(
            "[roll-audit-series] selector found no eligible champion yet",
            flush=True,
        )
        return False
    if detail:
        print(detail, file=sys.stderr)
    raise AuditRunError(
        f"champion selector failed with exit code {result.returncode}"
    )


def _champion_finish_count(champion_dir: Path) -> int | None:
    manifest = champion_dir / "champion.json"
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    try:
        schema = int(payload.get("evaluation_schema_version", 0))
        finish_count = int(payload.get("target_distance_reach_count", -1))
    except (TypeError, ValueError):
        return None
    if schema != SCHEMA_VERSION or not 0 <= finish_count <= NUM_ENVS:
        return None
    return finish_count


def audit_run(
    *,
    run_dir: Path,
    evaluation_dir: Path,
    champion_dir: Path,
    device: str,
    parent_frontier_m: float | None,
    stop_on_promoted_four_of_four: bool,
    runner: CommandRunner = subprocess.run,
) -> list[AuditResult]:
    checkpoints = find_checkpoints(run_dir)
    if not checkpoints:
        raise AuditRunError(f"no model_*.pt checkpoints found directly in {run_dir}")
    print(
        f"[roll-audit-series] checkpoints={len(checkpoints)} device={device} "
        f"run={run_dir}",
        flush=True,
    )
    results: list[AuditResult] = []
    for index, checkpoint in enumerate(checkpoints, start=1):
        identity = checkpoint_identity(checkpoint)
        print(
            f"[roll-audit-series] [{index}/{len(checkpoints)}] "
            f"iteration={identity.iteration} checkpoint={identity.path.name}",
            flush=True,
        )
        result = _audit_checkpoint(
            run_dir=run_dir,
            evaluation_dir=evaluation_dir,
            identity=identity,
            device=device,
            parent_frontier_m=parent_frontier_m,
            runner=runner,
        )
        results.append(result)
        print_audit_summary(result)
        promoted = _select_champion(
            evaluation_dir,
            champion_dir,
            runner=runner,
        )
        champion_finish_count = (
            _champion_finish_count(champion_dir) if promoted else None
        )
        if champion_finish_count is not None:
            print(
                "[roll-audit-series] selected champion "
                f"10m_finishers={champion_finish_count}/{NUM_ENVS}",
                flush=True,
            )
        if (
            stop_on_promoted_four_of_four
            and promoted
            and champion_finish_count == NUM_ENVS
        ):
            print(
                "[roll-audit-series] stopping after promoted 4/4 finisher",
                flush=True,
            )
            break
    return results


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        help="Audit JSON directory (default: RUN_DIR/checkpoint-audits).",
    )
    parser.add_argument(
        "--champion-dir",
        type=Path,
        help="Retained champion directory (default: RUN_DIR/checkpoint-champion).",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--parent-frontier-m",
        type=float,
        help="Selected-parent frontier passed to each deterministic evaluator.",
    )
    parser.add_argument(
        "--stop-on-promoted-4-of-4",
        dest="stop_on_promoted_four_of_four",
        action="store_true",
        help="Stop only after selector metadata reports a retained 4/4 finisher.",
    )
    args = parser.parse_args(argv)
    args.run_dir = args.run_dir.expanduser().resolve()
    args.evaluation_dir = (
        args.evaluation_dir.expanduser().resolve()
        if args.evaluation_dir is not None
        else args.run_dir / "checkpoint-audits"
    )
    args.champion_dir = (
        args.champion_dir.expanduser().resolve()
        if args.champion_dir is not None
        else args.run_dir / "checkpoint-champion"
    )
    if not args.run_dir.is_dir():
        parser.error(f"run directory not found: {args.run_dir}")
    if not args.device.strip():
        parser.error("--device must not be empty")
    if args.parent_frontier_m is not None and not math.isfinite(
        args.parent_frontier_m
    ):
        parser.error("--parent-frontier-m must be finite")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if not EVALUATOR.is_file():
        raise SystemExit(f"Evaluator not found: {EVALUATOR}")
    if not SELECTOR.is_file():
        raise SystemExit(f"Champion selector not found: {SELECTOR}")
    try:
        audit_run(
            run_dir=args.run_dir,
            evaluation_dir=args.evaluation_dir,
            champion_dir=args.champion_dir,
            device=args.device,
            parent_frontier_m=args.parent_frontier_m,
            stop_on_promoted_four_of_four=args.stop_on_promoted_four_of_four,
        )
    except (AuditRunError, OSError) as error:
        print(f"[roll-audit-series] error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Select a bounded, evidence-backed checkpoint for stair preview.

The selector reads TensorBoard scalars from one completed or active run and
scores only checkpoint files that currently exist in that run directory. A
checkpoint can replace ``model_0.pt`` only when its hard-task score is both
positive and lexicographically better than the baseline. Mean return is only a
tie-breaker between checkpoints that have already cleared that safety gate.

This script does not import or start MuJoCo, mjlab, Torch, training, or a GUI.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
from bisect import bisect_right
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

CHECKPOINT_RE = re.compile(r"model_(\d+)\.pt")
DEFAULT_MAX_CHECKPOINTS = 500
DEFAULT_MAX_EVENT_BYTES = 256 * 1024 * 1024
DEFAULT_MAX_SCALAR_POINTS = 250_000

METRIC_TAGS: dict[str, tuple[str, ...]] = {
    "full_stair_success": (
        "Episode_Reward/stair_top_goal",
        "Episode_Metrics/stair_top_goal",
        "Episode_Metrics/full_stair_success",
    ),
    "first_riser_clearance": (
        "Episode_Reward/stair_first_riser_clearance",
        "Episode_Metrics/stair_first_riser_clearance",
    ),
    "top_approach": (
        "Episode_Reward/stair_top_approach",
        "Episode_Metrics/stair_top_approach",
    ),
    "terrain_max": (
        "Curriculum/terrain_levels/max",
        "Curriculum/terrain_level/max",
    ),
    "terrain_mean": (
        "Curriculum/terrain_levels/mean",
        "Curriculum/terrain_level/mean",
    ),
    "goal_progress": (
        "Episode_Reward/stair_goal_progress",
        "Episode_Metrics/stair_goal_progress",
    ),
    "mean_return": ("Train/mean_reward",),
}

HARD_SCORE_FIELDS = (
    "full_stair_success",
    "first_riser_clearance",
    "top_approach",
    "terrain_max",
    "terrain_mean",
    "goal_progress",
)
SCORE_FIELDS = (*HARD_SCORE_FIELDS, "mean_return")


class RankingError(RuntimeError):
    """Raised when a run cannot be ranked without weakening the contract."""


@dataclass(frozen=True, order=True)
class ScalarPoint:
    """One scalar observation at a training step."""

    step: int
    value: float


@dataclass(frozen=True)
class MetricValue:
    """Nearest scalar evidence used for one metric at one checkpoint."""

    value: float
    step: int | None
    tag: str | None


@dataclass(frozen=True)
class RankedCheckpoint:
    """Checkpoint plus its resolved scalar evidence."""

    path: Path
    iteration: int
    metrics: Mapping[str, MetricValue]

    @property
    def hard_score(self) -> tuple[float, ...]:
        return tuple(self.metrics[field].value for field in HARD_SCORE_FIELDS)

    @property
    def score(self) -> tuple[float, ...]:
        return tuple(self.metrics[field].value for field in SCORE_FIELDS)


@dataclass(frozen=True)
class RankingResult:
    """A baseline-safe ranking result and its explanation."""

    selected: RankedCheckpoint
    baseline: RankedCheckpoint
    candidates: tuple[RankedCheckpoint, ...]
    reason: str


def checkpoint_iteration(path: Path) -> int | None:
    """Return N for an exact ``model_N.pt`` filename, otherwise ``None``."""

    match = CHECKPOINT_RE.fullmatch(path.name)
    return int(match.group(1)) if match else None


def discover_checkpoints(
    run_dir: Path, *, max_checkpoints: int = DEFAULT_MAX_CHECKPOINTS
) -> list[Path]:
    """Find existing checkpoint files and refuse an unexpectedly large run."""

    if max_checkpoints < 1:
        raise ValueError("max_checkpoints must be positive")
    checkpoints = [
        path.resolve()
        for path in run_dir.glob("model_*.pt")
        if path.is_file() and checkpoint_iteration(path) is not None
    ]
    checkpoints.sort(key=lambda path: checkpoint_iteration(path) or 0)
    if len(checkpoints) > max_checkpoints:
        raise RankingError(
            f"run has {len(checkpoints)} checkpoints, exceeding the configured bound of {max_checkpoints}"
        )
    return checkpoints


def normalize_scalar_series(
    points: Iterable[ScalarPoint | tuple[int, float]],
) -> tuple[ScalarPoint, ...]:
    """Validate, sort, and de-duplicate synthetic or parsed scalar points."""

    latest_by_step: dict[int, ScalarPoint] = {}
    for point in points:
        item = (
            point
            if isinstance(point, ScalarPoint)
            else ScalarPoint(int(point[0]), float(point[1]))
        )
        if item.step < 0:
            raise ValueError(f"scalar step must be non-negative, got {item.step}")
        if not math.isfinite(item.value):
            raise ValueError(f"scalar at step {item.step} is not finite")
        latest_by_step[item.step] = item
    return tuple(latest_by_step[step] for step in sorted(latest_by_step))


def nearest_scalar_at_or_before(
    points: Sequence[ScalarPoint], step: int
) -> ScalarPoint | None:
    """Return the nearest scalar whose step is not newer than ``step``."""

    if not points:
        return None
    steps = [point.step for point in points]
    index = bisect_right(steps, step) - 1
    return points[index] if index >= 0 else None


def prepare_scalar_data(
    raw_scalars: Mapping[str, Iterable[ScalarPoint | tuple[int, float]]],
) -> dict[str, tuple[ScalarPoint, ...]]:
    """Keep recognized TensorBoard tags and normalize their scalar series."""

    recognized = {tag for aliases in METRIC_TAGS.values() for tag in aliases}
    return {
        tag: normalize_scalar_series(points)
        for tag, points in raw_scalars.items()
        if tag in recognized
    }


def load_tensorboard_scalars(
    run_dir: Path,
    *,
    max_event_bytes: int = DEFAULT_MAX_EVENT_BYTES,
    max_scalar_points: int = DEFAULT_MAX_SCALAR_POINTS,
) -> dict[str, tuple[ScalarPoint, ...]]:
    """Load relevant TensorBoard scalars under explicit input-size bounds."""

    event_files = sorted(
        path for path in run_dir.glob("events.out.tfevents.*") if path.is_file()
    )
    if not event_files:
        raise RankingError(f"no TensorBoard event files found in {run_dir}")
    if max_event_bytes < 1 or max_scalar_points < 1:
        raise ValueError("TensorBoard bounds must be positive")

    event_bytes = sum(path.stat().st_size for path in event_files)
    if event_bytes > max_event_bytes:
        raise RankingError(
            f"TensorBoard logs total {event_bytes} bytes, exceeding the configured bound of {max_event_bytes}"
        )

    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError as exc:
        raise RankingError("TensorBoard is required to parse run scalars") from exc

    accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    scalar_tags = set(accumulator.Tags().get("scalars", ()))
    recognized = {tag for aliases in METRIC_TAGS.values() for tag in aliases}

    raw: dict[str, list[tuple[int, float]]] = {}
    point_count = 0
    for tag in sorted(scalar_tags & recognized):
        events = accumulator.Scalars(tag)
        point_count += len(events)
        if point_count > max_scalar_points:
            raise RankingError(
                f"relevant scalar count exceeds the configured bound of {max_scalar_points}"
            )
        raw[tag] = [(int(event.step), float(event.value)) for event in events]
    return prepare_scalar_data(raw)


def _resolve_metric(
    scalar_data: Mapping[str, Sequence[ScalarPoint]], metric: str, iteration: int
) -> MetricValue:
    best: MetricValue | None = None
    for tag in METRIC_TAGS[metric]:
        point = nearest_scalar_at_or_before(scalar_data.get(tag, ()), iteration)
        if point is None:
            continue
        candidate = MetricValue(value=point.value, step=point.step, tag=tag)
        if best is None or (candidate.step, candidate.value) > (best.step, best.value):
            best = candidate
    return best or MetricValue(value=0.0, step=None, tag=None)


def score_checkpoint(
    checkpoint: Path,
    scalar_data: Mapping[str, Sequence[ScalarPoint]],
) -> RankedCheckpoint:
    """Resolve nearest-at-or-before evidence and score one checkpoint."""

    iteration = checkpoint_iteration(checkpoint)
    if iteration is None:
        raise ValueError(f"not a model_N.pt checkpoint: {checkpoint}")
    metrics = {
        field: _resolve_metric(scalar_data, field, iteration) for field in SCORE_FIELDS
    }
    return RankedCheckpoint(
        path=checkpoint.resolve(), iteration=iteration, metrics=metrics
    )


def rank_checkpoints(
    checkpoints: Sequence[Path],
    scalar_data: Mapping[str, Sequence[ScalarPoint]],
) -> RankingResult:
    """Rank checkpoints while requiring hard-task improvement over model 0."""

    ranked = tuple(score_checkpoint(path, scalar_data) for path in checkpoints)
    baseline = next((item for item in ranked if item.iteration == 0), None)
    if baseline is None:
        raise RankingError("model_0.pt must exist as the preview ranking baseline")

    eligible = [
        item
        for item in ranked
        if item.iteration != 0
        and item.hard_score > baseline.hard_score
        and any(value > 0.0 for value in item.hard_score)
    ]
    if not eligible:
        return RankingResult(
            selected=baseline,
            baseline=baseline,
            candidates=ranked,
            reason="Kept model_0.pt because no checkpoint has positive hard-task evidence that improves baseline.",
        )

    selected = max(eligible, key=lambda item: item.score)
    first_improvement = next(
        field
        for field, selected_value, baseline_value in zip(
            HARD_SCORE_FIELDS, selected.hard_score, baseline.hard_score, strict=True
        )
        if selected_value != baseline_value
    )
    return RankingResult(
        selected=selected,
        baseline=baseline,
        candidates=ranked,
        reason=(
            f"Selected model_{selected.iteration}.pt because {first_improvement} is the first "
            "lexicographic hard-task improvement over model_0.pt."
        ),
    )


def _metric_json(metric: MetricValue) -> dict[str, object]:
    return {"value": metric.value, "source_step": metric.step, "source_tag": metric.tag}


def build_manifest(result: RankingResult, run_dir: Path) -> dict[str, object]:
    """Build the stable JSON document written for the preview process."""

    def checkpoint_json(item: RankedCheckpoint) -> dict[str, object]:
        return {
            "checkpoint": str(item.path),
            "iteration": item.iteration,
            "hard_score": list(item.hard_score),
            "score": list(item.score),
            "metrics": {
                name: _metric_json(item.metrics[name]) for name in SCORE_FIELDS
            },
        }

    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "run_directory": str(run_dir.resolve()),
        "score_order": list(SCORE_FIELDS),
        "selection_policy": (
            "A non-baseline checkpoint must have a positive hard-task score and a lexicographic "
            "hard-task improvement over model_0.pt; return is only a later tie-breaker."
        ),
        "selected_checkpoint": str(result.selected.path),
        "selected_iteration": result.selected.iteration,
        "reason": result.reason,
        "baseline": checkpoint_json(result.baseline),
        "candidates": [checkpoint_json(item) for item in result.candidates],
    }


def write_json_atomic(path: Path, document: Mapping[str, object]) -> None:
    """Atomically replace ``path`` with a formatted JSON document."""

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(document, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def select_best_checkpoint(
    run_dir: Path,
    *,
    max_checkpoints: int = DEFAULT_MAX_CHECKPOINTS,
    max_event_bytes: int = DEFAULT_MAX_EVENT_BYTES,
    max_scalar_points: int = DEFAULT_MAX_SCALAR_POINTS,
) -> RankingResult:
    """Discover, parse, and rank one run without starting simulation work."""

    run_dir = run_dir.resolve()
    if not run_dir.is_dir():
        raise RankingError(f"run directory does not exist: {run_dir}")
    checkpoints = discover_checkpoints(run_dir, max_checkpoints=max_checkpoints)
    if not checkpoints:
        raise RankingError(f"no model_N.pt checkpoints found in {run_dir}")
    scalars = load_tensorboard_scalars(
        run_dir,
        max_event_bytes=max_event_bytes,
        max_scalar_points=max_scalar_points,
    )
    return rank_checkpoints(checkpoints, scalars)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir",
        type=Path,
        help="rsl_rl run directory containing events and model_N.pt files",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="output JSON path (default: RUN_DIR/best_stair_preview.json)",
    )
    parser.add_argument("--max-checkpoints", type=int, default=DEFAULT_MAX_CHECKPOINTS)
    parser.add_argument("--max-event-bytes", type=int, default=DEFAULT_MAX_EVENT_BYTES)
    parser.add_argument(
        "--max-scalar-points", type=int, default=DEFAULT_MAX_SCALAR_POINTS
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    manifest_path = (args.manifest or run_dir / "best_stair_preview.json").resolve()
    result = select_best_checkpoint(
        run_dir,
        max_checkpoints=args.max_checkpoints,
        max_event_bytes=args.max_event_bytes,
        max_scalar_points=args.max_scalar_points,
    )
    write_json_atomic(manifest_path, build_manifest(result, run_dir))
    print(manifest_path)
    print(result.reason)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

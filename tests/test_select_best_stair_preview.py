from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.select_best_stair_preview import (
    HARD_SCORE_FIELDS,
    RankingError,
    ScalarPoint,
    build_manifest,
    checkpoint_iteration,
    discover_checkpoints,
    nearest_scalar_at_or_before,
    normalize_scalar_series,
    prepare_scalar_data,
    rank_checkpoints,
    write_json_atomic,
)


def _checkpoints(tmp_path: Path, *iterations: int) -> list[Path]:
    paths = []
    for iteration in iterations:
        path = tmp_path / f"model_{iteration}.pt"
        path.touch()
        paths.append(path)
    return paths


def _scalars(**series: list[tuple[int, float]]):
    tags = {
        "success": "Episode_Reward/stair_top_goal",
        "approach": "Episode_Reward/stair_top_approach",
        "terrain_max": "Curriculum/terrain_levels/max",
        "terrain_mean": "Curriculum/terrain_levels/mean",
        "progress": "Episode_Reward/stair_goal_progress",
        "return_": "Train/mean_reward",
    }
    return prepare_scalar_data({tags[name]: values for name, values in series.items()})


def test_checkpoint_parsing_and_discovery_only_accept_existing_exact_files(
    tmp_path: Path,
):
    expected = _checkpoints(tmp_path, 100, 0, 50)
    (tmp_path / "model_75.pt").mkdir()
    (tmp_path / "model_latest.pt").touch()
    (tmp_path / "prefix_model_25.pt").touch()

    discovered = discover_checkpoints(tmp_path)

    assert [checkpoint_iteration(path) for path in discovered] == [0, 50, 100]
    assert set(discovered) == {path.resolve() for path in expected}
    assert checkpoint_iteration(Path("model_9.pt")) == 9
    assert checkpoint_iteration(Path("model_latest.pt")) is None


def test_checkpoint_discovery_refuses_to_silently_truncate(tmp_path: Path):
    _checkpoints(tmp_path, 0, 1)

    with pytest.raises(RankingError, match="exceeding the configured bound"):
        discover_checkpoints(tmp_path, max_checkpoints=1)


def test_scalar_helpers_sort_deduplicate_and_use_nearest_past_value():
    points = normalize_scalar_series([(50, 0.4), (0, 0.0), (50, 0.6), (100, 1.0)])

    assert points == (ScalarPoint(0, 0.0), ScalarPoint(50, 0.6), ScalarPoint(100, 1.0))
    assert nearest_scalar_at_or_before(points, 49) == ScalarPoint(0, 0.0)
    assert nearest_scalar_at_or_before(points, 50) == ScalarPoint(50, 0.6)
    assert nearest_scalar_at_or_before(points, -1) is None


def test_high_return_without_hard_task_improvement_cannot_replace_baseline(
    tmp_path: Path,
):
    checkpoints = _checkpoints(tmp_path, 0, 50, 100)
    scalars = _scalars(
        success=[(0, 0.0), (50, 0.0), (100, 0.0)],
        approach=[(0, 0.0), (50, 0.0), (100, 0.0)],
        terrain_max=[(0, 15.0), (50, 0.0), (100, 0.0)],
        progress=[(0, 0.0), (50, 0.0), (100, 0.0)],
        return_=[(0, 1.0), (50, 500.0), (100, 1000.0)],
    )

    result = rank_checkpoints(checkpoints, scalars)

    assert result.selected.iteration == 0
    assert "no checkpoint has positive hard-task evidence" in result.reason


def test_full_stair_success_outranks_all_lower_priority_metrics(tmp_path: Path):
    checkpoints = _checkpoints(tmp_path, 0, 50, 100)
    scalars = _scalars(
        success=[(0, 0.0), (50, 0.01), (100, 0.0)],
        approach=[(0, 0.0), (50, 0.1), (100, 100.0)],
        terrain_max=[(0, 0.0), (50, 1.0), (100, 15.0)],
        terrain_mean=[(0, 0.0), (50, 1.0), (100, 15.0)],
        progress=[(0, 0.0), (50, 0.1), (100, 100.0)],
        return_=[(0, 0.0), (50, -10.0), (100, 1000.0)],
    )

    result = rank_checkpoints(checkpoints, scalars)

    assert result.selected.iteration == 50
    assert result.selected.metrics["full_stair_success"].value == 0.01
    assert "full_stair_success" in result.reason


def test_approach_then_terrain_then_progress_define_lexicographic_order(tmp_path: Path):
    checkpoints = _checkpoints(tmp_path, 0, 50, 100, 150)
    scalars = _scalars(
        approach=[(0, 0.0), (50, 0.2), (100, 0.2), (150, 0.1)],
        terrain_max=[(0, 0.0), (50, 2.0), (100, 3.0), (150, 15.0)],
        terrain_mean=[(0, 0.0), (50, 1.0), (100, 1.0), (150, 15.0)],
        progress=[(0, 0.0), (50, 10.0), (100, 1.0), (150, 100.0)],
        return_=[(0, 0.0), (50, 1000.0), (100, -1.0), (150, 5000.0)],
    )

    result = rank_checkpoints(checkpoints, scalars)

    assert result.selected.iteration == 100
    assert tuple(
        result.selected.metrics[field].value for field in HARD_SCORE_FIELDS
    ) == (
        0.0,
        0.2,
        3.0,
        1.0,
        1.0,
    )


def test_atomic_manifest_contains_selected_checkpoint_and_evidence(tmp_path: Path):
    checkpoints = _checkpoints(tmp_path, 0, 50)
    scalars = _scalars(
        success=[(0, 0.0), (45, 0.25)],
        approach=[(45, 0.5)],
        return_=[(50, 2.0)],
    )
    result = rank_checkpoints(checkpoints, scalars)
    manifest_path = tmp_path / "preview" / "best.json"

    write_json_atomic(manifest_path, build_manifest(result, tmp_path))
    document = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert document["selected_iteration"] == 50
    assert document["selected_checkpoint"] == str(checkpoints[1].resolve())
    assert document["baseline"]["iteration"] == 0
    assert document["candidates"][1]["metrics"]["full_stair_success"] == {
        "source_step": 45,
        "source_tag": "Episode_Reward/stair_top_goal",
        "value": 0.25,
    }
    assert not list(manifest_path.parent.glob(".*.tmp"))

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "select_best_roll_sprint_checkpoint.py"
)
SPEC = importlib.util.spec_from_file_location(
    "select_best_roll_sprint_checkpoint", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_candidate(
    root: Path,
    *,
    iteration: int,
    mean_credit: float,
    finish_times: list[float],
    heading: float = 10.0,
    road_exits: int = 0,
) -> Path:
    checkpoint = root / f"model_{iteration}.pt"
    checkpoint.write_bytes(f"checkpoint-{iteration}".encode())
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    per_robot = [
        {
            "robot_index": index,
            "target_10m_pass": index < len(finish_times),
            "time_to_valid_10m_s": (
                finish_times[index] if index < len(finish_times) else None
            ),
        }
        for index in range(4)
    ]
    report = {
        "schema_version": 8,
        "num_envs": 4,
        "target_distance_m": 10.0,
        "canonical_race_alignment": {"alignment_pass": True},
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest,
        "checkpoint_iteration": iteration,
        "nan_env_count": 0,
        "out_of_bounds_env_count": 0,
        "road_exit_env_count": road_exits,
        "maximum_road_boundary_overshoot_m": 1.0 if road_exits else 0.0,
        "maximum_heading_yaw_deviation_deg": heading,
        "target_distance_reach_count": len(finish_times),
        "mean_credited_forward_frontier_m": mean_credit,
        "mean_time_to_valid_10m_s": (
            sum(finish_times) / len(finish_times) if finish_times else None
        ),
        "slowest_time_to_valid_10m_s": max(finish_times, default=None),
        "per_robot": per_robot,
        "recovery_battery": {"self_right_then_reroll_rate": 0.75},
    }
    evaluation = root / f"checkpoint-{iteration:06d}.json"
    evaluation.write_text(json.dumps(report), encoding="utf-8")
    return checkpoint


def test_selector_uses_distance_until_all_robots_finish(tmp_path: Path) -> None:
    _write_candidate(tmp_path, iteration=100, mean_credit=5.0, finish_times=[])
    expected = _write_candidate(
        tmp_path,
        iteration=200,
        mean_credit=6.0,
        finish_times=[],
    )

    candidate = MODULE.select_best(tmp_path)

    assert candidate is not None
    assert candidate.checkpoint == expected.resolve()


def test_selector_prioritizes_more_finishers_before_speed_or_mean_distance(
    tmp_path: Path,
) -> None:
    _write_candidate(
        tmp_path,
        iteration=100,
        mean_credit=14.0,
        finish_times=[18.0, 19.0],
    )
    expected = _write_candidate(
        tmp_path,
        iteration=200,
        mean_credit=8.5,
        finish_times=[35.0, 36.0, 37.0],
    )

    candidate = MODULE.select_best(tmp_path)

    assert candidate is not None
    assert candidate.checkpoint == expected.resolve()


def test_selector_prefers_fastest_four_robot_finisher_and_keeps_one_file(
    tmp_path: Path,
) -> None:
    evaluation_dir = tmp_path / "evaluations"
    champion_dir = tmp_path / "champion"
    evaluation_dir.mkdir()
    _write_candidate(
        evaluation_dir,
        iteration=100,
        mean_credit=12.0,
        finish_times=[30.0, 31.0, 32.0, 33.0],
    )
    expected = _write_candidate(
        evaluation_dir,
        iteration=200,
        mean_credit=10.1,
        finish_times=[20.0, 21.0, 22.0, 23.0],
    )
    _write_candidate(
        evaluation_dir,
        iteration=300,
        mean_credit=14.0,
        finish_times=[10.0, 11.0, 12.0, 13.0],
        heading=5.0,
        road_exits=1,
    )

    candidate = MODULE.select_best(evaluation_dir)
    assert candidate is not None
    retained, manifest = MODULE.retain(candidate, champion_dir)

    assert candidate.checkpoint == expected.resolve()
    assert retained.read_bytes() == expected.read_bytes()
    assert list(champion_dir.glob("model_*.pt")) == [retained]
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["target_distance_reach_count"] == 4
    assert payload["slowest_time_to_valid_10m_s"] == 23.0

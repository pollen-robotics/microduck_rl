from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
SEARCH_PATH = SCRIPT_DIR / "search_stair_action_sequence.py"
SEARCH_SPEC = importlib.util.spec_from_file_location(
    "search_stair_action_sequence", SEARCH_PATH
)
assert SEARCH_SPEC is not None and SEARCH_SPEC.loader is not None
SEARCH = importlib.util.module_from_spec(SEARCH_SPEC)
sys.modules[SEARCH_SPEC.name] = SEARCH
SEARCH_SPEC.loader.exec_module(SEARCH)

SCRIPT_PATH = SCRIPT_DIR / "record_stair_action_sequence.py"
SPEC = importlib.util.spec_from_file_location("record_stair_action_sequence", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
RECORD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RECORD
SPEC.loader.exec_module(RECORD)


def test_load_residual_sequence_uses_exact_best_expanded(tmp_path: Path) -> None:
    expected = np.arange(56, dtype=np.float32).reshape(4, 14) / 100.0
    path = tmp_path / "sequence.npz"
    np.savez(path, best_expanded=expected, unrelated=np.ones(2))

    actual = RECORD.load_residual_sequence(path)

    np.testing.assert_array_equal(actual, expected)


def test_load_residual_sequence_rejects_wrong_or_nonfinite_shape(
    tmp_path: Path,
) -> None:
    wrong = tmp_path / "wrong.npz"
    np.savez(wrong, best_expanded=np.zeros((5, 7), dtype=np.float32))
    with pytest.raises(ValueError, match="shape"):
        RECORD.load_residual_sequence(wrong)

    nonfinite = tmp_path / "nonfinite.npz"
    values = np.zeros((2, 14), dtype=np.float32)
    values[0, 0] = np.nan
    np.savez(nonfinite, best_expanded=values)
    with pytest.raises(ValueError, match="finite"):
        RECORD.load_residual_sequence(nonfinite)


def test_video_frame_count_is_exact_and_validated() -> None:
    assert RECORD.video_frame_count(20.0, 50.0) == 1000
    assert RECORD.video_frame_count(0.1, 30.0) == 3
    with pytest.raises(ValueError, match="duration_seconds"):
        RECORD.video_frame_count(0.0, 50.0)
    with pytest.raises(ValueError, match="fps"):
        RECORD.video_frame_count(20.0, float("nan"))


def test_recorder_default_action_limit_matches_a13_search() -> None:
    assert RECORD.DEFAULT_ACTION_LIMIT == SEARCH.DEFAULT_ACTION_LIMIT == 10.0


def test_recorder_and_search_share_the_default_task() -> None:
    assert RECORD.TASK_ID == SEARCH.TASK_ID


def test_attempt_step_indices_repeat_the_frozen_sequence() -> None:
    np.testing.assert_array_equal(
        RECORD.attempt_step_indices(10, 4),
        np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1]),
    )
    with pytest.raises(ValueError, match="positive"):
        RECORD.attempt_step_indices(5, 0)

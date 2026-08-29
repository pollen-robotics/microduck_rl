from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts/search_stair_action_sequence.py"
SPEC = importlib.util.spec_from_file_location("search_stair_action_sequence", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
SEARCH = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SEARCH
SPEC.loader.exec_module(SEARCH)


def _shell_corners(min_x: float, min_z: float, steps: int) -> np.ndarray:
    corners = np.zeros((steps, 8, 3), dtype=np.float64)
    corners[:, :, 0] = min_x
    corners[:, :, 2] = min_z
    return corners


def test_expand_action_knots_is_balanced_and_piecewise_constant() -> None:
    knots = np.array([[[1.0], [2.0], [3.0]]])

    expanded = SEARCH.expand_action_knots(knots, horizon_steps=8)

    np.testing.assert_array_equal(
        expanded[0, :, 0], np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0])
    )
    assert expanded.shape == (1, 8, 1)
    with pytest.raises(ValueError, match="horizon_steps"):
        SEARCH.expand_action_knots(knots, horizon_steps=0)


def test_load_initial_knots_preserves_prefix_and_zero_extends(tmp_path: Path) -> None:
    source = np.arange(12, dtype=np.float32).reshape(3, 4)
    path = tmp_path / "warm.npz"
    np.savez(path, best_knots=source)

    mean = SEARCH.load_initial_knots(path, target_knots=5, action_dim=4)

    np.testing.assert_array_equal(mean[:3], source)
    np.testing.assert_array_equal(mean[3:], np.zeros((2, 4)))
    with pytest.raises(ValueError, match="target has only"):
        SEARCH.load_initial_knots(path, target_knots=2, action_dim=4)


def test_sagittal_action_projection_matches_microduck_mirror_convention() -> None:
    compact = np.arange(14, dtype=np.float64).reshape(2, 7) / 10.0

    full = SEARCH.expand_sagittal_actions(compact)

    np.testing.assert_array_equal(full[:, :5], compact[:, :5])
    np.testing.assert_array_equal(full[:, 5:7], compact[:, 5:7])
    np.testing.assert_array_equal(full[:, 7:9], np.zeros((2, 2)))
    np.testing.assert_array_equal(full[:, 9:14], -compact[:, :5])
    np.testing.assert_allclose(SEARCH.project_sagittal_actions(full), compact)


def test_box_corners_world_uses_exact_oriented_box_geometry() -> None:
    center = np.array([[1.0, 2.0, 3.0]])
    rotation = np.array([[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])

    corners = SEARCH.box_corners_world(center, rotation, np.array([0.1, 0.2, 0.3]))

    assert corners.shape == (1, 8, 3)
    np.testing.assert_allclose(corners[0].min(axis=0), [0.8, 1.9, 2.7])
    np.testing.assert_allclose(corners[0].max(axis=0), [1.2, 2.1, 3.3])


def test_strict_score_never_combines_independent_trajectory_maxima() -> None:
    roots = np.array(
        [
            [0.71, 0.0, 0.15],
            [0.62, 0.0, 0.205],
            [0.69, 0.0, 0.19],
        ]
    )
    corners = _shell_corners(0.64, 0.16, steps=3)
    corners[1, :, 0] = 0.67
    corners[1, :, 2] = 0.18
    secured = np.array([False, False, True])

    split = SEARCH.score_strict_trajectory(roots, corners, secured)

    assert not split.success
    assert not split.root_and_shell_crossing
    assert not split.secured_tread

    roots[2] = [0.71, 0.0, 0.205]
    corners[2, :, 0] = 0.67
    corners[2, :, 2] = 0.18
    same_state = SEARCH.score_strict_trajectory(roots, corners, secured)
    assert same_state.success
    assert same_state.root_and_shell_crossing
    assert same_state.exact_shell_clearance
    assert same_state.score >= SEARCH.StrictScoreConfig().secured_tread_bonus


def test_one_low_shell_corner_blocks_exact_clearance_and_success() -> None:
    roots = np.array([[0.71, 0.0, 0.205]])
    corners = _shell_corners(0.67, 0.18, steps=1)
    corners[0, 0, 2] = 0.169

    result = SEARCH.score_strict_trajectory(roots, corners, np.array([True]))

    assert not result.exact_shell_clearance
    assert not result.root_and_shell_crossing
    assert not result.success


def test_side_bypass_invalidates_otherwise_secured_state() -> None:
    roots = np.array([[0.71, 0.21, 0.205]])
    corners = _shell_corners(0.67, 0.18, steps=1)

    result = SEARCH.score_strict_trajectory(roots, corners, np.array([True]))

    assert result.side_bypass
    assert not result.success
    assert not result.secured_tread
    assert result.score < -900_000.0

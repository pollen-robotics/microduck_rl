from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

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


def test_cem_population_preserves_exact_frozen_prefix() -> None:
    rng = np.random.default_rng(4)
    mean = np.arange(12, dtype=np.float64).reshape(3, 4) / 20.0
    std = np.full_like(mean, 0.25)

    population = SEARCH.sample_cem_population(
        rng,
        mean,
        std,
        candidates=8,
        residual_limit=1.0,
        frozen_prefix_knots=2,
    )

    assert population.shape == (8, 3, 4)
    np.testing.assert_array_equal(
        population[:, :2], np.broadcast_to(mean[:2], (8, 2, 4))
    )
    np.testing.assert_array_equal(population[0], mean)
    assert np.any(population[1:, 2] != mean[2])


def test_forced_head_lever_reset_is_deterministic() -> None:
    params = {
        "lip_release_fraction": 0.1,
        "shell_brace_fraction": 0.2,
        "tread_recovery_fraction": 0.0,
        "real_handoff_fraction": 0.7,
        "shell_local_x_range": (0.54, 0.59),
        "shell_pitch_deg_range": (8.0, 30.0),
        "lip_vertical_speed_range": (0.45, 0.90),
    }
    cfg = SimpleNamespace(
        events={"route_state_curriculum": SimpleNamespace(params=params)}
    )

    SEARCH.force_assisted_reset_mode(cfg, "head-lever")

    assert params["lip_release_fraction"] == 0.0
    assert params["shell_brace_fraction"] == 1.0
    assert params["real_handoff_fraction"] == 0.0
    assert params["shell_local_x_range"] == (0.565, 0.565)
    assert params["shell_pitch_deg_range"] == (19.0, 19.0)
    assert params["lip_vertical_speed_range"] == (0.45, 0.90)


def test_bank_reset_position_can_be_pinned_without_changing_other_params() -> None:
    params = {"bank_path": "bank.pt", "min_root_height": 0.08}
    cfg = SimpleNamespace(events={"walker_state_bank": SimpleNamespace(params=params)})

    SEARCH.pin_bank_reset_position(cfg, 0.602, 0.0)

    assert params["local_x_range"] == (0.602, 0.602)
    assert params["local_y_range"] == (0.0, 0.0)
    assert params["min_root_height"] == 0.08


def test_forward_bank_auto_resolution_and_family_forcing() -> None:
    family = SimpleNamespace(params={"family_weights": (1, 1, 1, 3, 6)})
    cfg = SimpleNamespace(
        events={
            "state_bank_family": family,
            "walker_state_bank": SimpleNamespace(params={}),
            "stage2_forward_state_bank": SimpleNamespace(params={}),
        }
    )

    assert SEARCH.resolve_bank_event_name(cfg) == "stage2_forward_state_bank"
    SEARCH.force_state_bank_family(cfg, 4)

    assert family.params["forced_family"] == 4


def test_cem_disables_only_resetting_a37_terminations() -> None:
    keep = object()
    cfg = SimpleNamespace(
        terminations={
            "time_out": keep,
            "nan_state": keep,
            "secured_tread_success": object(),
            "stair_progress_stalled": object(),
        }
    )

    removed = SEARCH.disable_search_only_terminations(cfg)

    assert removed == ("secured_tread_success", "stair_progress_stalled")
    assert cfg.terminations == {"time_out": keep, "nan_state": keep}


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


def test_contact_sequence_requires_work_release_and_later_tread_support() -> None:
    contact = np.array([False, True, True, False, False, True])
    power = np.array([0.0, 0.04, 0.03, 0.0, 0.0, 0.0])
    support = np.array([False, False, False, False, False, True])

    result = SEARCH.score_contact_sequence(
        contact,
        power,
        support,
        np.ones(6, dtype=bool),
    )

    assert result.sequence
    assert result.anchor_contact_step == 1
    assert result.release_step == 3
    assert result.support_contact_step == 5
    np.testing.assert_allclose(result.positive_pitch_work_j, 0.0014)


def test_contact_sequence_does_not_credit_support_before_release() -> None:
    contact = np.array([True, True, False, False])
    power = np.array([0.05, 0.05, 0.0, 0.0])
    support = np.array([True, False, False, False])

    result = SEARCH.score_contact_sequence(
        contact,
        power,
        support,
        np.ones(4, dtype=bool),
    )

    assert not result.sequence
    assert result.release_step == 2
    assert result.support_contact_step == -1


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


def test_valid_step_mask_can_exclude_a_frozen_prefix_peak() -> None:
    roots = np.array([[0.71, 0.0, 0.205], [0.61, 0.0, 0.13]])
    corners = _shell_corners(0.67, 0.18, steps=2)
    corners[1, :, 0] = 0.59
    corners[1, :, 2] = 0.11
    secured = np.array([True, False])

    all_steps = SEARCH.score_strict_trajectory(roots, corners, secured)
    tail_only = SEARCH.score_strict_trajectory(
        roots, corners, secured, valid_steps=np.array([False, True])
    )

    assert all_steps.success
    assert not tail_only.success
    assert tail_only.best_step == 1
    assert tail_only.score < all_steps.score


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

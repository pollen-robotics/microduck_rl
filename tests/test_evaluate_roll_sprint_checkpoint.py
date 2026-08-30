from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "evaluate_roll_sprint_checkpoint.py"
)
SPEC = importlib.util.spec_from_file_location(
    "evaluate_roll_sprint_checkpoint", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _auditor() -> object:
    return MODULE.RollCycleAuditor(
        initial_position_xy=torch.zeros(1, 2),
        initial_root_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        initial_vertical_velocity=torch.zeros(1),
        step_dt=0.02,
    )


def _observe(
    auditor,
    *,
    omega: float,
    forward_velocity: float = 0.0,
    support: bool = True,
    head_contact: bool = False,
    head_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    root_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> None:
    auditor.observe(
        position_xy=torch.zeros(1, 2),
        root_quat=torch.tensor([root_quat]),
        head_quat=torch.tensor([head_quat]),
        linear_velocity_w=torch.tensor([[forward_velocity, 0.0, 0.0]]),
        angular_velocity_b=torch.tensor([[0.0, omega, 0.0]]),
        support=torch.tensor([support]),
        head_contact=torch.tensor([head_contact]),
    )


def test_auditor_backward_rocking_cannot_complete_or_repay_angle() -> None:
    auditor = _auditor()

    for _ in range(80):
        _observe(auditor, omega=5.0)
        _observe(auditor, omega=-5.0)

    assert auditor.accum.item() == pytest.approx(0.0, abs=1.0e-6)
    assert auditor.valid_count.item() == 0
    assert auditor.invalid_count.item() == 0


def test_auditor_missing_head_contact_marks_full_cycle_invalid() -> None:
    auditor = _auditor()
    auditor.accum[:] = MODULE.TARGET_ANGLE - 0.01

    _observe(auditor, omega=1.0)

    assert auditor.valid_count.item() == 0
    assert auditor.invalid_count.item() == 1
    assert auditor.linked_distance.item() == 0.0


def test_auditor_releases_only_rotation_capped_distance_on_valid_cycle() -> None:
    auditor = _auditor()
    auditor.accum[:] = 1.0
    auditor.head_latch[:] = True
    auditor.accum[:] = MODULE.TARGET_ANGLE - 0.02

    _observe(auditor, omega=2.0, forward_velocity=100.0)

    expected_cap = MODULE.MAX_DISTANCE_PER_RAD * 2.0 * 0.02
    assert auditor.valid_count.item() == 1
    assert auditor.linked_distance.item() == pytest.approx(expected_cap)


def test_heading_uses_lateral_axis_at_vertical_pitch() -> None:
    half = math.sqrt(0.5)
    heading = MODULE.heading_from_quat(
        torch.tensor([[half, 0.0, half, 0.0]], dtype=torch.float32)
    )

    assert heading[0, 0].item() == pytest.approx(1.0, abs=1.0e-6)
    assert heading[0, 1].item() == pytest.approx(0.0, abs=1.0e-6)


def test_promotion_requires_every_gate() -> None:
    report = {
        **MODULE.PROMOTION,
        "nan_env_count": 0,
        "out_of_bounds_env_count": 0,
    }
    assert MODULE.promotion_pass(report)

    for key in MODULE.PROMOTION:
        failing = dict(report)
        if key in {
            "p95_lateral_drift_m",
            "mean_uncredited_positive_displacement_m",
        }:
            failing[key] = float(report[key]) + 0.001
        else:
            failing[key] = float(report[key]) - 0.001
        assert not MODULE.promotion_pass(failing), key

    for key in ("nan_env_count", "out_of_bounds_env_count"):
        failing = dict(report)
        failing[key] = 1
        assert not MODULE.promotion_pass(failing), key

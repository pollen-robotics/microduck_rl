from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "record_backroll_policy.py"
SPEC = importlib.util.spec_from_file_location("record_backroll_policy", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
RECORD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RECORD
SPEC.loader.exec_module(RECORD)


def _detector():
    return RECORD.DiagnosticStuckDetector(min_steps=100, patience_steps=50)


def test_static_diagnostic_cuts_only_after_minimum_and_patience() -> None:
    detector = _detector()
    assert not detector.update(
        step=99,
        frontier_rad=0.0,
        cycle_count=0,
        angular_speed=0.0,
        vertical_speed=0.0,
    )
    assert detector.update(
        step=100,
        frontier_rad=0.0,
        cycle_count=0,
        angular_speed=0.0,
        vertical_speed=0.0,
    )


def test_default_static_diagnostic_is_six_seconds_at_50_hz() -> None:
    detector = RECORD.DiagnosticStuckDetector(
        min_steps=round(RECORD.DIAGNOSTIC_MIN_SECONDS / 0.02),
        patience_steps=round(RECORD.DIAGNOSTIC_STUCK_SECONDS / 0.02),
    )
    assert not detector.update(
        step=299,
        frontier_rad=0.0,
        cycle_count=0,
        angular_speed=0.0,
        vertical_speed=0.0,
    )
    assert detector.update(
        step=300,
        frontier_rad=0.0,
        cycle_count=0,
        angular_speed=0.0,
        vertical_speed=0.0,
    )


def test_roll_frontier_progress_delays_stuck_cut() -> None:
    detector = _detector()
    assert not detector.update(
        step=90,
        frontier_rad=0.5,
        cycle_count=0,
        angular_speed=0.0,
        vertical_speed=0.0,
    )
    assert not detector.update(
        step=139,
        frontier_rad=0.5,
        cycle_count=0,
        angular_speed=0.0,
        vertical_speed=0.0,
    )
    assert detector.update(
        step=140,
        frontier_rad=0.5,
        cycle_count=0,
        angular_speed=0.0,
        vertical_speed=0.0,
    )


def test_recovery_motion_keeps_incomplete_video_alive() -> None:
    detector = _detector()
    assert not detector.update(
        step=120,
        frontier_rad=0.0,
        cycle_count=0,
        angular_speed=1.0,
        vertical_speed=0.0,
    )
    assert not detector.update(
        step=160,
        frontier_rad=0.0,
        cycle_count=0,
        angular_speed=0.0,
        vertical_speed=0.04,
    )
    assert not detector.update(
        step=209,
        frontier_rad=0.0,
        cycle_count=0,
        angular_speed=0.0,
        vertical_speed=0.0,
    )
    assert detector.update(
        step=210,
        frontier_rad=0.0,
        cycle_count=0,
        angular_speed=0.0,
        vertical_speed=0.0,
    )


def test_completed_cycle_rearms_frontier_tracking_for_retry() -> None:
    detector = _detector()
    detector.update(
        step=80,
        frontier_rad=6.2,
        cycle_count=0,
        angular_speed=0.0,
        vertical_speed=0.0,
    )
    assert not detector.update(
        step=120,
        frontier_rad=0.0,
        cycle_count=1,
        angular_speed=0.0,
        vertical_speed=0.0,
    )
    assert not detector.update(
        step=140,
        frontier_rad=0.2,
        cycle_count=1,
        angular_speed=0.0,
        vertical_speed=0.0,
    )


def test_full_duration_diagnostic_requires_incomplete_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "record_backroll_policy.py",
            "checkpoint.pt",
            "output.mp4",
            "--full-duration-diagnostic",
        ],
    )
    args = RECORD._parse_args()
    assert args.full_duration_diagnostic
    assert not args.allow_incomplete_diagnostic
    with pytest.raises(SystemExit, match="requires --allow-incomplete-diagnostic"):
        RECORD._validate_args(args)

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "evaluate_roll_sprint_single.py"
)
SPEC = importlib.util.spec_from_file_location("evaluate_roll_sprint_single", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
sys.path.insert(0, str(SCRIPT_PATH.parent))
SPEC.loader.exec_module(MODULE)


def test_single_robot_cli_defaults_to_full_race(monkeypatch, tmp_path: Path) -> None:
    checkpoint = tmp_path / "model_0.pt"
    onnx = tmp_path / "policy.onnx"
    output = tmp_path / "result.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT_PATH),
            str(checkpoint),
            "--onnx",
            str(onnx),
            "--output",
            str(output),
        ],
    )

    args = MODULE._parse_args()

    assert args.duration == 40.0
    assert args.onnx == onnx
    assert args.output == output


@pytest.mark.parametrize(
    ("input_dim", "output_dim", "finite", "error", "expected"),
    [
        (61, 14, True, 1.0e-6, True),
        (60, 14, True, 1.0e-6, False),
        (61, 13, True, 1.0e-6, False),
        (61, 14, False, 1.0e-6, False),
        (61, 14, True, 1.0e-3, False),
    ],
)
def test_deployment_contract_requires_exact_shape_and_action_parity(
    input_dim: int,
    output_dim: int,
    finite: bool,
    error: float,
    expected: bool,
) -> None:
    assert (
        MODULE.deployment_contract_pass(
            input_dim=input_dim,
            output_dim=output_dim,
            actions_finite=finite,
            max_abs_action_error=error,
        )
        is expected
    )


def test_single_robot_race_gate_requires_distance_road_rolls_and_health() -> None:
    robot = {"target_10m_pass": True, "road_corridor_pass": True}

    assert MODULE.single_robot_race_pass(
        robot,
        valid_roll_count=14,
        recovered_reroll_count=13,
        nan_count=0,
        out_of_bounds_count=0,
    )
    assert not MODULE.single_robot_race_pass(
        robot,
        valid_roll_count=13,
        recovered_reroll_count=13,
        nan_count=0,
        out_of_bounds_count=0,
    )
    assert not MODULE.single_robot_race_pass(
        {**robot, "road_corridor_pass": False},
        valid_roll_count=14,
        recovered_reroll_count=13,
        nan_count=0,
        out_of_bounds_count=0,
    )

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "evaluate_assisted_stair_checkpoint.py"
)
SPEC = importlib.util.spec_from_file_location(
    "evaluate_assisted_stair_checkpoint", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
A12TrajectoryMetrics = MODULE.A12TrajectoryMetrics


def _observe(
    metrics: A12TrajectoryMetrics,
    position: tuple[float, float, float],
    episode_step: int,
) -> dict[str, bool]:
    events = metrics.observe(
        torch.tensor([position], dtype=torch.float32),
        torch.tensor([episode_step], dtype=torch.long),
    )
    return {name: bool(mask.item()) for name, mask in events.items()}


def test_a12_clearance_thresholds_are_conjunctive_and_consecutive() -> None:
    metrics = A12TrajectoryMetrics(1, "cpu")

    assert not _observe(metrics, (0.665, 0.20, 0.174), 3)[
        "root_center_over_lip"
    ]
    assert not _observe(metrics, (0.664, 0.20, 0.175), 4)[
        "root_center_over_lip"
    ]
    assert not _observe(metrics, (0.665, 0.20, 0.175), 5)[
        "root_center_over_lip"
    ]
    assert _observe(metrics, (0.665, 0.20, 0.175), 6)[
        "root_center_over_lip"
    ]

    assert not _observe(metrics, (0.700, 0.20, 0.198), 7)["full_shell_clear"]
    assert not _observe(metrics, (0.700, 0.20, 0.198), 8)["full_shell_clear"]
    assert not _observe(metrics, (0.700, 0.20, 0.198), 9)["full_shell_clear"]
    assert _observe(metrics, (0.700, 0.20, 0.198), 10)["full_shell_clear"]


def test_a12_ignores_first_three_steps_and_resets_trajectory_state() -> None:
    metrics = A12TrajectoryMetrics(1, "cpu")
    qualifying = (0.700, 0.0, 0.198)

    for episode_step in range(3):
        assert not any(_observe(metrics, qualifying, episode_step).values())
    assert not _observe(metrics, qualifying, 3)["root_center_over_lip"]

    metrics.reset(torch.tensor([True]))
    assert not any(_observe(metrics, qualifying, 0).values())
    assert not _observe(metrics, qualifying, 3)["root_center_over_lip"]
    assert _observe(metrics, qualifying, 4)["root_center_over_lip"]


def test_a12_side_bypass_is_once_per_trajectory_and_only_before_clear() -> None:
    metrics = A12TrajectoryMetrics(1, "cpu")

    assert _observe(metrics, (0.660, 0.361, 0.10), 3)["side_bypass"]
    assert not _observe(metrics, (0.800, -0.50, 0.30), 4)["side_bypass"]

    metrics.reset(torch.tensor([True]))
    for episode_step in range(3, 7):
        events = _observe(metrics, (0.700, 0.0, 0.198), episode_step)
    assert events["full_shell_clear"]
    assert not _observe(metrics, (0.800, 0.50, 0.30), 7)["side_bypass"]

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
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
ContactTrajectoryMetrics = MODULE.ContactTrajectoryMetrics
JointFrontierTrajectoryMetrics = MODULE.JointFrontierTrajectoryMetrics
TerminalPositionTrajectoryMetrics = MODULE.TerminalPositionTrajectoryMetrics


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


def test_contact_metrics_ignore_reset_contact_but_count_policy_recontact() -> None:
    metrics = ContactTrajectoryMetrics(1, "cpu")

    assert not metrics.observe(torch.tensor([True]), torch.tensor([1])).item()
    assert not metrics.observe(torch.tensor([True]), torch.tensor([3])).item()
    assert not metrics.observe(torch.tensor([False]), torch.tensor([4])).item()
    assert metrics.observe(torch.tensor([True]), torch.tensor([5])).item()
    assert not metrics.observe(torch.tensor([False]), torch.tensor([6])).item()
    assert not metrics.observe(torch.tensor([True]), torch.tensor([7])).item()


def test_joint_frontier_requires_x_and_z_in_the_same_frame() -> None:
    metrics = JointFrontierTrajectoryMetrics(1, "cpu")

    metrics.observe(
        torch.tensor([[0.665, 0.0, 0.120]]), torch.tensor([3])
    )
    metrics.observe(
        torch.tensor([[0.560, 0.0, 0.190]]), torch.tensor([4])
    )
    progress, best_x, best_z, milestones = metrics.complete(torch.tensor([True]))

    assert progress[0] < 0.30
    assert best_x[0] == pytest.approx(0.665)
    assert best_z[0] == pytest.approx(0.120)
    assert milestones[-2] == 0


def test_joint_frontier_reports_conjunctive_milestones_and_resets() -> None:
    metrics = JointFrontierTrajectoryMetrics(1, "cpu")

    metrics.observe(
        torch.tensor([[0.660, 0.0, 0.175]]), torch.tensor([3])
    )
    progress, best_x, best_z, milestones = metrics.complete(torch.tensor([True]))

    assert 0.95 <= progress[0] < 1.0
    assert best_x[0] == pytest.approx(0.660)
    assert best_z[0] == pytest.approx(0.175)
    assert milestones[4] == 1
    assert milestones[5] == 0

    progress, _, _, milestones = metrics.complete(torch.tensor([True]))
    assert progress == [0.0]
    assert milestones == [0] * len(MODULE.JOINT_FRONTIER_MILESTONES)


def test_terminal_position_metric_integrates_exactly_one_and_resets() -> None:
    metrics = TerminalPositionTrajectoryMetrics(
        num_envs=1,
        device="cpu",
        max_episode_length=150,
        step_dt=0.02,
    )
    target = torch.tensor([[0.720, 0.0, 0.205]], dtype=torch.float32)

    for episode_step in range(151):
        metrics.observe(target, torch.tensor([episode_step]))
    completed = metrics.complete(torch.tensor([True]))
    assert completed == pytest.approx([1.0])

    assert metrics.complete(torch.tensor([True])) == pytest.approx([0.0])


def test_terminal_position_metric_rejects_endpoint_and_lateral_bypass() -> None:
    metrics = TerminalPositionTrajectoryMetrics(
        num_envs=1,
        device="cpu",
        max_episode_length=150,
        step_dt=0.02,
    )
    outside = torch.tensor([[0.720, 0.201, 0.205]], dtype=torch.float32)
    target = torch.tensor([[0.720, 0.0, 0.205]], dtype=torch.float32)

    metrics.observe(outside, torch.tensor([149]))
    metrics.observe(target, torch.tensor([150]))

    assert metrics.complete(torch.tensor([True])) == pytest.approx([0.0])

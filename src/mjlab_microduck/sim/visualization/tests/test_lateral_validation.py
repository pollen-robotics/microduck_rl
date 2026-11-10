"""Acceptance must reject stationary heading holds and completed circles."""

import math
from types import SimpleNamespace

import pytest

from mjlab_microduck.sim.visualization.lateral import validate as module

LIMITS = SimpleNamespace(min_progress=0.2, max_yaw=10, max_forward_ratio=0.25)


def test_displacement_is_measured_along_the_starting_ducks_axes():
    result = module.metrics(
        [0, 0, 0.12, math.pi / 2], [[-0.5, 0.1, 0.12, math.pi / 2, -1]], 0.3
    )
    assert result["lateral_m"] == pytest.approx(0.5)
    assert result["forward_m"] == pytest.approx(0.1)
    assert module.passes_gate(result, 0.3, LIMITS)


def test_yaw_unwrap_retains_a_full_circle_and_handles_the_pi_boundary():
    rows = [
        [0, 0, 0.12, angle, -1] for angle in [0, math.pi / 2, math.pi, -math.pi / 2, 0]
    ]
    circle = module.metrics([0, 0, 0.12, 0], rows, 0.3)
    assert circle["max_yaw_deg"] == pytest.approx(360)
    crossing = module.metrics(
        [0, 0, 0.12, math.radians(179)], [[0, 0.5, 0.12, math.radians(-179), -1]], 0.3
    )
    assert crossing["max_yaw_deg"] == pytest.approx(2)


def test_gate_requires_actual_correctly_signed_movement_and_a_physical_stop():
    stationary = module.metrics([0, 0, 0.12, 0], [[0, 0, 0.12, 0, -1]], 0.3)
    assert not module.passes_gate(stationary, 0.3, LIMITS)
    assert module.passes_gate(stationary, 0, LIMITS)
    right = module.metrics([0, 0, 0.12, 0], [[0, -0.5, 0.12, 0, -1]], -0.3)
    assert module.passes_gate(right, -0.3, LIMITS)
    assert not module.passes_gate(right, 0.3, LIMITS)
    assert not module.passes_gate(right, 0, LIMITS)
    assert not module.passes_gate({**right, "forward_m": 0.2}, -0.3, LIMITS)
    assert not module.passes_gate({**right, "max_yaw_deg": 10.1}, -0.3, LIMITS)


def test_stop_rejects_rotation_or_travel_that_returns_to_the_start():
    loop = module.metrics(
        [0, 0, 0.12, 0],
        [[0.1, 0, 0.12, 0, -1], [0, 0, 0.12, 0, -1]],
        0,
    )
    assert loop["forward_m"] == 0 and loop["lateral_m"] == 0
    assert loop["max_translation_m"] == pytest.approx(0.1)
    assert not module.passes_gate(loop, 0, LIMITS)
    rotation = module.metrics([0, 0, 0.12, 0], [[0, 0, 0.12, math.radians(11), -1]], 0)
    assert not module.passes_gate(rotation, 0, LIMITS)

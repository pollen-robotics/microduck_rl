"""Crowd routes, footstep queues and the follow controller (CPU, no MuJoCo).

Covers the properties the behavior actually depends on: five people who all
keep moving independently, a queue that trails a fixed arc length behind the
selected person, and a controller that emits exactly zero outside FOLLOW.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "scripts" / "behavior_demos")
)

from follow_me_among_others.crowd import (
    COLORS,
    CTRL_HZ,
    DISTRACTOR_COLORS,
    SELECTABLE_COLORS,
    TRAIL_DISTANCE,
    CrowdFollowController,
    FootstepTrail,
    PersonState,
    TrailState,
    crowd_trajectory,
    wrap,
)

CTRL_DT = 1.0 / CTRL_HZ
ROLLOUT_SECONDS = 60.0


def times(seconds=ROLLOUT_SECONDS, dt=CTRL_DT):
    return [step * dt for step in range(int(seconds / dt))]


# --- crowd -----------------------------------------------------------------


def test_exactly_five_people_three_selectable_two_distractors():
    crowd = crowd_trajectory(0.0)
    assert set(crowd) == set(COLORS)
    assert len(crowd) == 5
    assert len(SELECTABLE_COLORS) == 3
    assert len(DISTRACTOR_COLORS) == 2


def test_trail_distance_is_055_m():
    assert TRAIL_DISTANCE == pytest.approx(0.55)


def test_trajectory_is_deterministic():
    assert all(
        np.allclose(crowd_trajectory(7.5)[c].pos, crowd_trajectory(7.5)[c].pos)
        for c in COLORS
    )


def test_every_person_keeps_moving_for_the_whole_rollout():
    # Nobody freezes, waits or teleports to make acquisition easier: over every
    # one-second window, each person covers real ground.
    samples = {c: [] for c in COLORS}
    for t in times():
        crowd = crowd_trajectory(t)
        for color in COLORS:
            samples[color].append(crowd[color].pos.copy())
    for color in COLORS:
        track = np.asarray(samples[color])
        steps = np.linalg.norm(np.diff(track, axis=0), axis=1)
        assert steps.min() > 0.0, f"{color} stopped"
        whole = (len(steps) // int(CTRL_HZ)) * int(CTRL_HZ)
        per_second = steps[:whole].reshape(-1, int(CTRL_HZ)).sum(axis=1)
        assert per_second.min() > 0.01, f"{color} stalled for a second"
        # No teleports: a single control step never jumps.
        assert steps.max() < 0.05, f"{color} teleported"


def test_people_move_independently_not_as_a_rigid_formation():
    # Pairwise distances must vary: a crowd that keeps its shape is one object.
    spans = []
    for t in times(seconds=40.0, dt=0.2):
        crowd = crowd_trajectory(t)
        spans.append(
            [
                float(np.linalg.norm(crowd[a].pos - crowd[b].pos))
                for i, a in enumerate(COLORS)
                for b in COLORS[i + 1 :]
            ]
        )
    variation = np.asarray(spans).std(axis=0)
    assert variation.min() > 0.02, "crowd moves as a rigid formation"


def test_people_travel_in_different_directions():
    # Mixed lap directions, so the crowd is not a carousel.
    crowd = crowd_trajectory(3.0)
    headings = [crowd[c].yaw for c in COLORS]
    assert len(set(np.round(headings, 3))) == len(COLORS)


def test_yaw_follows_the_velocity_direction():
    for t in (0.0, 4.2, 19.7):
        for person in crowd_trajectory(t).values():
            expected = math.atan2(float(person.velocity[1]), float(person.velocity[0]))
            assert wrap(person.yaw - expected) == pytest.approx(0.0, abs=1e-9)


def test_people_stay_within_a_bounded_arena():
    for t in times(seconds=ROLLOUT_SECONDS, dt=0.1):
        for person in crowd_trajectory(t).values():
            assert np.all(np.abs(person.pos) < 3.0)


# --- footstep trail --------------------------------------------------------


def test_trail_lags_by_the_configured_arc_length():
    color = "BLUE"
    trail = FootstepTrail(crowd_trajectory(0.0)[color])
    state = None
    for t in times(seconds=30.0):
        state = trail.update(crowd_trajectory(t)[color])
    assert isinstance(state, TrailState)
    # Once the person has walked further than the gap, the queue is exactly
    # one gap behind in ARC LENGTH (not straight-line distance).
    assert state.leader_path_s > TRAIL_DISTANCE
    assert state.leader_path_s - state.path_s == pytest.approx(TRAIL_DISTANCE)


def test_trail_point_is_behind_the_person_not_on_top_of_them():
    color = "GREEN"
    trail = FootstepTrail(crowd_trajectory(0.0)[color])
    for t in times(seconds=30.0):
        person = crowd_trajectory(t)[color]
        state = trail.update(person)
    separation = float(np.linalg.norm(person.pos - state.pos))
    # Straight-line separation is at most the arc gap, and clearly non-zero.
    assert 0.05 < separation <= TRAIL_DISTANCE + 1e-9


def test_trail_extrapolates_backwards_before_the_gap_is_walked():
    # At t=0 nobody has walked anywhere; the queue must still yield a sensible
    # point behind the person rather than their own position.
    person = crowd_trajectory(0.0)["RED"]
    trail = FootstepTrail(person)
    state = trail.update(person)
    assert float(np.linalg.norm(state.pos - person.pos)) == pytest.approx(
        TRAIL_DISTANCE, abs=1e-6
    )
    assert state.path_s == pytest.approx(-TRAIL_DISTANCE)


def test_trail_arc_length_is_monotonic():
    color = "PURPLE"
    trail = FootstepTrail(crowd_trajectory(0.0)[color])
    previous = -math.inf
    for t in times(seconds=20.0):
        state = trail.update(crowd_trajectory(t)[color])
        assert state.leader_path_s >= previous
        previous = state.leader_path_s


def test_each_person_has_an_independent_queue():
    trails = {c: FootstepTrail(crowd_trajectory(0.0)[c]) for c in COLORS}
    for t in times(seconds=20.0):
        crowd = crowd_trajectory(t)
        states = {c: trails[c].update(crowd[c]) for c in COLORS}
    lengths = [round(states[c].leader_path_s, 6) for c in COLORS]
    assert len(set(lengths)) == len(COLORS), "queues are coupled"


# --- controller ------------------------------------------------------------


def straight_trail(x=1.0, y=0.0, yaw=0.0):
    return TrailState(np.array([x, y]), yaw, 0.0, TRAIL_DISTANCE)


def test_command_is_exactly_zero_when_inactive():
    controller = CrowdFollowController()
    for distance in (0.05, 0.5, 3.0):
        command, _ = controller.update(
            False, straight_trail(x=distance), np.zeros(3), 0.0
        )
        assert np.array_equal(command, np.zeros(3, dtype=np.float32))
        assert float(np.linalg.norm(command)) == 0.0


def test_zero_command_survives_a_previously_active_step():
    # No filter state may leak a residual command into a stationary state.
    controller = CrowdFollowController()
    controller.update(True, straight_trail(), np.zeros(3), 0.0)
    command, _ = controller.update(False, straight_trail(), np.zeros(3), 0.0)
    assert float(np.linalg.norm(command)) == 0.0


def test_walks_forward_at_the_measured_gait_onset_command():
    # Below this vx the stock policy emits an action but does not visibly walk.
    controller = CrowdFollowController()
    command, info = controller.update(True, straight_trail(), np.zeros(3), 0.0)
    assert command[0] == pytest.approx(CrowdFollowController.WALK_VX)
    assert command[1] == 0.0
    assert info["error"] == pytest.approx(1.0)


def test_keeps_walking_while_turning():
    # The RED hand-off regression: slowing down to turn stalled the gait.
    controller = CrowdFollowController()
    for yaw_error_deg in (30.0, 90.0, 150.0, -30.0, -90.0, -150.0):
        command, _ = controller.update(
            True,
            straight_trail(
                x=math.cos(math.radians(yaw_error_deg)),
                y=math.sin(math.radians(yaw_error_deg)),
            ),
            np.zeros(3),
            0.0,
        )
        assert command[0] == pytest.approx(CrowdFollowController.WALK_VX), (
            f"gait stalled while turning {yaw_error_deg} deg"
        )
        assert abs(command[2]) > 0.0


def test_stops_pushing_forward_once_on_the_footprint():
    controller = CrowdFollowController()
    command, _ = controller.update(True, straight_trail(x=0.05), np.zeros(3), 0.0)
    assert command[0] == 0.0


def test_turn_direction_matches_the_bearing_error():
    controller = CrowdFollowController()
    left, _ = controller.update(True, straight_trail(x=0.0, y=1.0), np.zeros(3), 0.0)
    right, _ = controller.update(True, straight_trail(x=0.0, y=-1.0), np.zeros(3), 0.0)
    assert left[2] > 0.0
    assert right[2] < 0.0


def test_no_yaw_command_inside_the_heading_deadband():
    controller = CrowdFollowController()
    command, _ = controller.update(True, straight_trail(), np.zeros(3), 0.0)
    assert command[2] == 0.0


def test_yaw_command_stays_within_configured_bounds():
    controller = CrowdFollowController()
    for yaw_error_deg in range(-180, 181, 5):
        angle = math.radians(yaw_error_deg)
        command, _ = controller.update(
            True,
            straight_trail(x=math.cos(angle), y=math.sin(angle)),
            np.zeros(3),
            0.0,
        )
        wz = float(command[2])
        if wz > 0.0:
            assert (
                CrowdFollowController.LEFT_WZ_MIN
                <= wz
                <= CrowdFollowController.LEFT_WZ_MAX
            )
        elif wz < 0.0:
            assert (
                CrowdFollowController.RIGHT_WZ_MIN
                <= -wz
                <= CrowdFollowController.RIGHT_WZ_MAX
            )


def test_lateral_command_is_never_used():
    # The walking policy is driven in vx/wz only; vy stays zero-padded.
    controller = CrowdFollowController()
    for angle in np.linspace(-math.pi, math.pi, 37):
        command, _ = controller.update(
            True,
            straight_trail(x=float(math.cos(angle)), y=float(math.sin(angle))),
            np.zeros(3),
            0.0,
        )
        assert command[1] == 0.0


def test_command_is_float32_three_vector():
    controller = CrowdFollowController()
    command, _ = controller.update(True, straight_trail(), np.zeros(3), 0.0)
    assert command.shape == (3,)
    assert command.dtype == np.float32


def test_returned_command_is_a_copy():
    # The caller stores this in a record; a shared buffer would rewrite history.
    controller = CrowdFollowController()
    first, _ = controller.update(True, straight_trail(), np.zeros(3), 0.0)
    snapshot = first.copy()
    controller.update(True, straight_trail(x=0.0, y=1.0), np.zeros(3), 0.0)
    assert np.array_equal(first, snapshot)


def test_person_state_carries_position_and_heading():
    person = crowd_trajectory(1.0)["YELLOW"]
    assert isinstance(person, PersonState)
    assert person.pos.shape == (2,)
    assert person.moving

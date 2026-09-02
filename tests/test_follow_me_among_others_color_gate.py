"""Color-gating tests: the demo must lock onto the requested shirt color only.

These drive the state machine with synthetic camera reports in which the wrong
colors are visible, centered and closer than the target. Nothing here needs a
model or a policy — the gate is a pure decision on the camera report, so it is
tested as one.
"""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "scripts" / "behavior_demos")
)

from follow_me_among_others.crowd import (
    DISTRACTOR_COLORS,
    SELECTABLE_COLORS,
    TARGET_SEQUENCE,
    SearchFollowStateMachine,
)

CTRL_DT = 1.0 / 50.0
CONE = SearchFollowStateMachine.FOUND_CONE


def report(visible, off_axis_deg, others=()):
    """One camera report. ``others`` never influences the gate by design."""
    return {
        "target_visible": visible,
        "target_off_axis": math.radians(off_axis_deg),
        "visible_colors": list(others) + (["TARGET"] if visible else []),
    }


def test_found_cone_is_eight_degrees():
    assert CONE == pytest.approx(math.radians(8.0))


def test_distractors_are_not_selectable_targets():
    # The two distractor colors must never appear in the requested sequence.
    assert set(DISTRACTOR_COLORS).isdisjoint(TARGET_SEQUENCE)
    assert set(TARGET_SEQUENCE) <= set(SELECTABLE_COLORS)
    assert set(SELECTABLE_COLORS).isdisjoint(DISTRACTOR_COLORS)


def test_visible_distractors_alone_never_trigger_found():
    # Every distractor is visible and dead-centered; the target is not visible.
    machine = SearchFollowStateMachine(("BLUE",))
    t = 0.0
    while t < SearchFollowStateMachine.MAX_SEARCH_SECONDS - 0.1:
        machine.update(t, report(False, 180.0, others=DISTRACTOR_COLORS))
        assert machine.state == "SEARCH"
        t += CTRL_DT
    assert machine.cycles == []


def test_target_visible_but_off_axis_does_not_trigger_found():
    # Seen, but outside the acquisition cone: looking in roughly the right
    # direction is not the same as having acquired the person.
    machine = SearchFollowStateMachine(("GREEN",))
    t = 0.0
    while t < 5.0:
        machine.update(t, report(True, 12.0))
        assert machine.state == "SEARCH"
        t += CTRL_DT


def test_target_centered_and_visible_triggers_found():
    machine = SearchFollowStateMachine(("RED",))
    t = 0.0
    while machine.state == "SEARCH" and t < 5.0:
        machine.update(t, report(True, 1.5))
        t += CTRL_DT
    assert machine.state == "FOUND"
    assert machine.current["found_s"] == pytest.approx(
        SearchFollowStateMachine.MIN_SEARCH_SECONDS, abs=2 * CTRL_DT
    )


def test_acquisition_is_at_the_cone_boundary():
    # Just inside acquires; just outside does not. Locks the gate value itself.
    inside = SearchFollowStateMachine(("BLUE",))
    outside = SearchFollowStateMachine(("BLUE",))
    t = 0.0
    while t < 3.0:
        inside.update(t, report(True, 7.9))
        outside.update(t, report(True, 8.1))
        t += CTRL_DT
    assert inside.state != "SEARCH"
    assert outside.state == "SEARCH"


def test_wrong_color_locks_are_impossible_by_construction():
    # Whatever the crowd does, a completed cycle records the REQUESTED color:
    # the machine never adopts a target from what it happens to see.
    machine = SearchFollowStateMachine()
    t = 0.0
    while not machine.done and t < 120.0:
        machine.update(t, report(True, 0.5, others=("YELLOW", "PURPLE", "RED")))
        t += CTRL_DT
    assert machine.done
    completed = [c["target"] for c in machine.cycles]
    assert completed == list(TARGET_SEQUENCE)
    wrong_color_locks = sum(
        cycle["target"] != TARGET_SEQUENCE[i] for i, cycle in enumerate(machine.cycles)
    )
    assert wrong_color_locks == 0


def test_target_switches_only_after_a_completed_cycle():
    # The active target must stay fixed for the whole SEARCH->STOP pipeline,
    # so a mid-follow distraction cannot swap targets.
    machine = SearchFollowStateMachine()
    t = 0.0
    seen_per_cycle = {}
    while not machine.done and t < 120.0:
        completed = len(machine.cycles)
        seen_per_cycle.setdefault(completed, set()).add(machine.target)
        machine.update(t, report(True, 0.5, others=DISTRACTOR_COLORS))
        t += CTRL_DT
    for completed, targets in seen_per_cycle.items():
        assert len(targets) == 1, f"target changed mid-cycle {completed}: {targets}"


def test_second_blue_selection_requires_a_fresh_search():
    # Selection 4 repeats BLUE. It must run its own SEARCH, not reuse the lock
    # from selection 1.
    machine = SearchFollowStateMachine()
    t = 0.0
    while not machine.done and t < 120.0:
        machine.update(t, report(True, 0.5))
        t += CTRL_DT
    first_blue, last_blue = machine.cycles[0], machine.cycles[3]
    assert first_blue["target"] == last_blue["target"] == "BLUE"
    assert last_blue["search_start_s"] > first_blue["cycle_end_s"]
    assert last_blue["search_duration_s"] >= SearchFollowStateMachine.MIN_SEARCH_SECONDS

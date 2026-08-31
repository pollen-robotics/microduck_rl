"""Pure state-machine tests for the follow-me-among-others behavior demo.

CPU only: no model, no policy, no renderer. The machine is fed synthetic camera
reports, which is exactly the point — the sequencing logic must be correct
independently of whether a rollout happens to work.
"""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "scripts" / "behavior_demos")
)

from follow_me_among_others.crowd import (
    SELECTABLE_COLORS,
    TARGET_SEQUENCE,
    SearchFollowStateMachine,
    wrap,
)

CTRL_DT = 1.0 / 50.0


def seen(off_axis_deg=0.0):
    return {"target_visible": True, "target_off_axis": math.radians(off_axis_deg)}


NOT_SEEN = {"target_visible": False, "target_off_axis": math.pi}


def drive(machine, camera_for, seconds=90.0, dt=CTRL_DT):
    """Run the machine forward, recording (t, state, target) per step."""
    log = []
    steps = int(seconds / dt)
    for step in range(steps):
        t = step * dt
        state, target = machine.state, machine.target
        log.append((t, "DONE" if machine.done else state, target))
        machine.update(t, camera_for(t, machine))
        if machine.done:
            break
    return log


def test_requested_sequence_is_blue_green_red_blue():
    # The repeated BLUE is deliberate: it forces a second full search instead
    # of letting the controller keep a latched target.
    assert TARGET_SEQUENCE == ("BLUE", "GREEN", "RED", "BLUE")
    assert TARGET_SEQUENCE[0] == TARGET_SEQUENCE[3]


def test_completes_four_cycles_in_requested_order():
    machine = SearchFollowStateMachine()
    drive(machine, lambda t, m: seen())
    assert machine.done
    assert len(machine.cycles) == 4
    assert [c["target"] for c in machine.cycles] == list(TARGET_SEQUENCE)


def test_every_cycle_visits_search_found_follow_stop_in_order():
    machine = SearchFollowStateMachine()
    log = drive(machine, lambda t, m: seen())
    for selection, target in enumerate(TARGET_SEQUENCE, start=1):
        cycle = machine.cycles[selection - 1]
        assert cycle["target"] == target
        # Timestamps must be strictly ordered through the whole pipeline.
        assert (
            cycle["search_start_s"]
            < cycle["found_s"]
            < cycle["follow_start_s"]
            < cycle["stop_s"]
            < cycle["cycle_end_s"]
        )
    states = [entry[1] for entry in log]
    # The observed state string sequence, deduplicated, is the pattern repeated.
    collapsed = [s for i, s in enumerate(states) if i == 0 or s != states[i - 1]]
    assert collapsed[:4] == ["SEARCH", "FOUND", "FOLLOW", "STOP"]
    assert collapsed.count("FOLLOW") == 4


def test_state_durations_match_configuration():
    machine = SearchFollowStateMachine()
    drive(machine, lambda t, m: seen())
    for cycle in machine.cycles:
        assert cycle["follow_start_s"] - cycle["found_s"] == pytest.approx(
            SearchFollowStateMachine.FOUND_SECONDS, abs=2 * CTRL_DT
        )
        assert cycle["stop_s"] - cycle["follow_start_s"] == pytest.approx(
            SearchFollowStateMachine.FOLLOW_SECONDS, abs=2 * CTRL_DT
        )
        assert cycle["cycle_end_s"] - cycle["stop_s"] == pytest.approx(
            SearchFollowStateMachine.STOP_SECONDS, abs=2 * CTRL_DT
        )


def test_follows_only_in_follow_state():
    machine = SearchFollowStateMachine()
    t = 0.0
    while machine.state != "FOLLOW" and t < 10.0:
        assert not machine.follows_now
        machine.update(t, seen())
        t += CTRL_DT
    assert machine.state == "FOLLOW"
    assert machine.follows_now


def test_minimum_search_dwell_is_enforced():
    # Even with the target visible and centered from t=0, the machine must not
    # leave SEARCH before MIN_SEARCH_SECONDS.
    machine = SearchFollowStateMachine()
    t = 0.0
    while t < SearchFollowStateMachine.MIN_SEARCH_SECONDS - CTRL_DT:
        machine.update(t, seen())
        assert machine.state == "SEARCH"
        t += CTRL_DT


def test_search_timeout_raises_rather_than_standing_still():
    # "Never acquired" must be a loud failure, not a silent zero-command
    # rollout that still writes a metrics file.
    machine = SearchFollowStateMachine()
    with pytest.raises(RuntimeError, match="failed to find BLUE"):
        drive(machine, lambda t, m: NOT_SEEN, seconds=20.0)


def test_done_is_terminal_and_idempotent():
    machine = SearchFollowStateMachine()
    drive(machine, lambda t, m: seen())
    assert machine.done
    for step in range(10):
        state, target, changed = machine.update(1000.0 + step, seen())
        assert (state, target, changed) == ("DONE", TARGET_SEQUENCE[-1], False)
    assert len(machine.cycles) == 4


def test_rejects_empty_or_unselectable_sequences():
    with pytest.raises(ValueError):
        SearchFollowStateMachine(())
    # Distractor colors are never valid targets.
    with pytest.raises(ValueError, match="selectable"):
        SearchFollowStateMachine(("YELLOW",))
    with pytest.raises(ValueError):
        SearchFollowStateMachine(("BLUE", "PURPLE"))
    for color in SELECTABLE_COLORS:
        assert SearchFollowStateMachine((color,)).target == color


def test_custom_sequence_is_honored():
    machine = SearchFollowStateMachine(("RED", "RED"))
    drive(machine, lambda t, m: seen())
    assert [c["target"] for c in machine.cycles] == ["RED", "RED"]


def test_wrap_maps_angles_into_pi_interval():
    # wrap() normalizes to the half-open interval [-pi, pi).
    assert wrap(0.0) == pytest.approx(0.0)
    assert wrap(3.0 * math.pi) == pytest.approx(-math.pi)
    assert wrap(math.radians(370.0)) == pytest.approx(math.radians(10.0))
    assert wrap(math.radians(-350.0)) == pytest.approx(math.radians(10.0))
    for angle in (-10.0, -1.0, 0.3, 2.0, 7.5, 100.0):
        assert -math.pi <= wrap(angle) < math.pi

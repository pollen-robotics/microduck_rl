from __future__ import annotations

import pytest

from mjlab_microduck.rom.supervisor_state import (
    SupervisorEffect,
    SupervisorEvent,
    SupervisorState,
    transition,
)


@pytest.mark.parametrize(
    ("state", "event", "next_state", "releases_slot"),
    [
        ("NO_CHILD", "SPAWN_REQUESTED", "SPAWNING", False),
        ("SPAWNING", "READY_RECEIVED", "IDLE", True),
        ("IDLE", "START_SENT", "STARTING", False),
        ("STARTING", "START_ACK", "RUNNING", False),
        ("RUNNING", "STOP_CLAIMED", "STOPPING", False),
        ("STOPPING", "TERMINAL_ACK", "IDLE", True),
        ("STARTING", "OPERATION_TIMEOUT", "QUARANTINED", False),
        ("QUARANTINED", "SIGTERM_SENT", "TERMINATING", False),
        ("TERMINATING", "TERM_TIMEOUT", "KILLING", False),
        ("KILLING", "SIGKILL_SENT", "REAPING", False),
        ("REAPING", "CHILD_REAPED", "NO_CHILD", True),
    ],
)
def test_supervisor_transition_table(
    state: str, event: str, next_state: str, releases_slot: bool
) -> None:
    """Changing a lifecycle edge can lose exclusive motion-slot ownership."""
    result = transition(SupervisorState(state), SupervisorEvent(event))

    assert result.next_state == SupervisorState(next_state)
    assert result.releases_slot is releases_slot


def test_ready_releases_availability_without_releasing_an_owned_task() -> None:
    """Treating READY as task completion would make a future task release the wrong slot."""
    result = transition(SupervisorState.SPAWNING, SupervisorEvent.READY_RECEIVED)

    assert result.effects == (SupervisorEffect.RELEASE_SLOT,)
    assert result.releases_slot is True


def test_every_state_event_pair_is_explicit_or_quarantines() -> None:
    """An unhandled lifecycle pair must isolate the child rather than silently continuing."""
    expected_edges = {
        (SupervisorState.NO_CHILD, SupervisorEvent.SPAWN_REQUESTED),
        (SupervisorState.SPAWNING, SupervisorEvent.READY_RECEIVED),
        (SupervisorState.IDLE, SupervisorEvent.START_SENT),
        (SupervisorState.STARTING, SupervisorEvent.START_ACK),
        (SupervisorState.RUNNING, SupervisorEvent.STOP_CLAIMED),
        (SupervisorState.STOPPING, SupervisorEvent.TERMINAL_ACK),
        (SupervisorState.STARTING, SupervisorEvent.OPERATION_TIMEOUT),
        (SupervisorState.QUARANTINED, SupervisorEvent.SIGTERM_SENT),
        (SupervisorState.TERMINATING, SupervisorEvent.TERM_TIMEOUT),
        (SupervisorState.KILLING, SupervisorEvent.SIGKILL_SENT),
        (SupervisorState.REAPING, SupervisorEvent.CHILD_REAPED),
    }

    for state in SupervisorState:
        for event in SupervisorEvent:
            result = transition(state, event)
            if (state, event) in expected_edges:
                assert result.next_state is not None
            else:
                assert result.next_state is SupervisorState.QUARANTINED
                assert result.effects == (SupervisorEffect.QUARANTINE,)

from __future__ import annotations

import pytest

import mjlab_microduck.rom.supervisor_state as supervisor_state_module
from mjlab_microduck.rom.supervisor_state import (
    SupervisorEffect,
    SupervisorEvent,
    SupervisorState,
    transition,
)


@pytest.mark.parametrize(
    ("state", "event", "next_state", "effects", "releases_slot"),
    [
        ("NO_CHILD", "SPAWN_REQUESTED", "SPAWNING", ("SPAWN_CHILD",), False),
        ("SPAWNING", "READY_RECEIVED", "IDLE", ("RELEASE_SLOT",), True),
        ("IDLE", "START_SENT", "STARTING", ("SEND_START",), False),
        ("STARTING", "START_ACK", "RUNNING", ("PERSIST_RUNNING",), False),
        ("RUNNING", "STOP_CLAIMED", "STOPPING", ("SEND_STOP",), False),
        ("STOPPING", "TERMINAL_ACK", "IDLE", ("RELEASE_SLOT",), True),
        ("STARTING", "OPERATION_TIMEOUT", "QUARANTINED", ("QUARANTINE",), False),
        ("IDLE", "TERMINATION_CLAIMED", "TERMINATING", ("SEND_SIGTERM",), False),
        ("RUNNING", "TERMINATION_CLAIMED", "TERMINATING", ("SEND_SIGTERM",), False),
        ("STARTING", "TERMINATION_CLAIMED", "TERMINATING", ("SEND_SIGTERM",), False),
        ("STOPPING", "TERMINATION_CLAIMED", "TERMINATING", ("SEND_SIGTERM",), False),
        ("QUARANTINED", "TERMINATION_CLAIMED", "TERMINATING", ("SEND_SIGTERM",), False),
        ("TERMINATING", "CHILD_EXITED", "REAPING", ("REAP",), False),
        ("TERMINATING", "TERM_TIMEOUT", "KILLING", ("SEND_SIGKILL",), False),
        ("KILLING", "SIGKILL_SENT", "REAPING", ("REAP",), False),
        ("REAPING", "CHILD_REAPED", "NO_CHILD", ("RELEASE_SLOT",), True),
    ],
)
def test_supervisor_transition_table(
    state: str,
    event: str,
    next_state: str,
    effects: tuple[str, ...],
    releases_slot: bool,
) -> None:
    """Changing a lifecycle edge can lose exclusive motion-slot ownership."""
    result = transition(SupervisorState(state), SupervisorEvent(event))

    assert result.next_state == SupervisorState(next_state)
    assert result.effects == tuple(SupervisorEffect(effect) for effect in effects)
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
        *(
            (state, SupervisorEvent.TERMINATION_CLAIMED)
            for state in (
                SupervisorState.IDLE,
                SupervisorState.RUNNING,
                SupervisorState.STARTING,
                SupervisorState.STOPPING,
                SupervisorState.QUARANTINED,
            )
        ),
        (SupervisorState.TERMINATING, SupervisorEvent.CHILD_EXITED),
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


def test_transition_table_cannot_be_mutated_by_an_importer() -> None:
    """Mutable lifecycle data would let unrelated code disable process quarantine."""
    key = (SupervisorState.RUNNING, SupervisorEvent.STOP_CLAIMED)

    with pytest.raises(TypeError):
        supervisor_state_module._TRANSITIONS[key] = supervisor_state_module.SupervisorTransition(
            SupervisorState.IDLE
        )

    assert transition(*key).effects == (SupervisorEffect.SEND_STOP,)

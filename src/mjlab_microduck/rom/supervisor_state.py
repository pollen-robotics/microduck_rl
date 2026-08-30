"""Pure, total lifecycle state machine for one isolated runtime child."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType


class SupervisorState(str, Enum):
    NO_CHILD = "NO_CHILD"
    SPAWNING = "SPAWNING"
    IDLE = "IDLE"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    QUARANTINED = "QUARANTINED"
    TERMINATING = "TERMINATING"
    KILLING = "KILLING"
    REAPING = "REAPING"


class SupervisorEvent(str, Enum):
    SPAWN_REQUESTED = "SPAWN_REQUESTED"
    READY_RECEIVED = "READY_RECEIVED"
    START_SENT = "START_SENT"
    START_ACK = "START_ACK"
    STOP_CLAIMED = "STOP_CLAIMED"
    TERMINAL_ACK = "TERMINAL_ACK"
    OPERATION_TIMEOUT = "OPERATION_TIMEOUT"
    SIGTERM_SENT = "SIGTERM_SENT"
    TERM_TIMEOUT = "TERM_TIMEOUT"
    SIGKILL_SENT = "SIGKILL_SENT"
    CHILD_REAPED = "CHILD_REAPED"


class SupervisorEffect(str, Enum):
    SPAWN_CHILD = "SPAWN_CHILD"
    SEND_START = "SEND_START"
    PERSIST_RUNNING = "PERSIST_RUNNING"
    SEND_STOP = "SEND_STOP"
    SEND_SIGTERM = "SEND_SIGTERM"
    SEND_SIGKILL = "SEND_SIGKILL"
    REAP = "REAP"
    RELEASE_SLOT = "RELEASE_SLOT"
    QUARANTINE = "QUARANTINE"


@dataclass(frozen=True, slots=True)
class SupervisorTransition:
    """An immutable state change and its declarative supervisor side effects.

    ``releases_slot`` means that the motion slot is now available.  In the
    ``SPAWNING -> IDLE`` READY edge, no task was previously owned; the effect
    announces runtime readiness, rather than releasing a completed task.
    """

    next_state: SupervisorState
    effects: tuple[SupervisorEffect, ...] = ()
    releases_slot: bool = False


_TRANSITIONS = MappingProxyType({
    (SupervisorState.NO_CHILD, SupervisorEvent.SPAWN_REQUESTED): SupervisorTransition(
        SupervisorState.SPAWNING, (SupervisorEffect.SPAWN_CHILD,)
    ),
    (SupervisorState.SPAWNING, SupervisorEvent.READY_RECEIVED): SupervisorTransition(
        SupervisorState.IDLE, (SupervisorEffect.RELEASE_SLOT,), releases_slot=True
    ),
    (SupervisorState.IDLE, SupervisorEvent.START_SENT): SupervisorTransition(
        SupervisorState.STARTING, (SupervisorEffect.SEND_START,)
    ),
    (SupervisorState.STARTING, SupervisorEvent.START_ACK): SupervisorTransition(
        SupervisorState.RUNNING, (SupervisorEffect.PERSIST_RUNNING,)
    ),
    (SupervisorState.RUNNING, SupervisorEvent.STOP_CLAIMED): SupervisorTransition(
        SupervisorState.STOPPING, (SupervisorEffect.SEND_STOP,)
    ),
    (SupervisorState.STOPPING, SupervisorEvent.TERMINAL_ACK): SupervisorTransition(
        SupervisorState.IDLE, (SupervisorEffect.RELEASE_SLOT,), releases_slot=True
    ),
    (SupervisorState.STARTING, SupervisorEvent.OPERATION_TIMEOUT): SupervisorTransition(
        SupervisorState.QUARANTINED, (SupervisorEffect.QUARANTINE,)
    ),
    (SupervisorState.QUARANTINED, SupervisorEvent.SIGTERM_SENT): SupervisorTransition(
        SupervisorState.TERMINATING, (SupervisorEffect.SEND_SIGTERM,)
    ),
    (SupervisorState.TERMINATING, SupervisorEvent.TERM_TIMEOUT): SupervisorTransition(
        SupervisorState.KILLING, (SupervisorEffect.SEND_SIGKILL,)
    ),
    (SupervisorState.KILLING, SupervisorEvent.SIGKILL_SENT): SupervisorTransition(
        SupervisorState.REAPING, (SupervisorEffect.REAP,)
    ),
    (SupervisorState.REAPING, SupervisorEvent.CHILD_REAPED): SupervisorTransition(
        SupervisorState.NO_CHILD, (SupervisorEffect.RELEASE_SLOT,), releases_slot=True
    ),
})


def transition(state: SupervisorState, event: SupervisorEvent) -> SupervisorTransition:
    """Return an explicit transition; every unsupported pair quarantines the child."""
    return _TRANSITIONS.get(
        (state, event),
        SupervisorTransition(SupervisorState.QUARANTINED, (SupervisorEffect.QUARANTINE,)),
    )

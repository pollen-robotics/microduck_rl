from __future__ import annotations

import array
import os
import socket
from pathlib import Path

import pytest
from pydantic import ValidationError

from mjlab_microduck.rom.contracts import canonical_json
from mjlab_microduck.rom.process_protocol import (
    PACKET_MAX_BYTES,
    AckPayload,
    CommandPayload,
    ProtocolViolation,
    RuntimeMessage,
    StartPayload,
    decode_packet,
    encode_packet,
    receive_packet,
)


def test_terminal_event_has_distinct_bounded_unsolicited_correlation() -> None:
    from mjlab_microduck.rom.contracts import TaskEvidence
    from mjlab_microduck.rom.process_protocol import (
        TerminalEventPayload,
        TerminalPayload,
    )

    message = RuntimeMessage(
        kind="TERMINAL_EVENT", generation=7, operationSequence=0, taskId="1" * 32,
        payload=TerminalEventPayload(
            eventSequence=1,
            terminal=TerminalPayload(
                outcome="SUCCEEDED",
                evidence=TaskEvidence(
                    bundleDigest="sha256:" + "a" * 64,
                    policyDigest="sha256:" + "b" * 64,
                    modelDigest="sha256:" + "c" * 64,
                    metrics={"upright": True}, stopReason="TASK_COMPLETE",
                ),
            ),
        ),
    )
    assert decode_packet(encode_packet(message)) == message


def test_terminal_event_requires_zero_operation_sequence() -> None:
    from mjlab_microduck.rom.contracts import TaskEvidence
    from mjlab_microduck.rom.process_protocol import (
        TerminalEventPayload,
        TerminalPayload,
    )

    with pytest.raises(ValueError, match="operationSequence"):
        RuntimeMessage(
            kind="TERMINAL_EVENT", generation=7, operationSequence=9,
            taskId="1" * 32,
            payload=TerminalEventPayload(
                eventSequence=1,
                terminal=TerminalPayload(
                    outcome="FAILED",
                    evidence=TaskEvidence(
                        bundleDigest="sha256:" + "a" * 64,
                        policyDigest="sha256:" + "b" * 64,
                        modelDigest="sha256:" + "c" * 64,
                        stopReason="FALLEN",
                    ),
                ),
            ),
        )


def _start_message() -> RuntimeMessage:
    return RuntimeMessage.start(
        generation=7,
        operationSequence=11,
        taskId="0" * 32,
        payload=StartPayload(
            actionCode="WALK_VELOCITY",
            bundleDigest="sha256:" + "1" * 64,
            parameters={"vxMps": 0.1, "vyMps": 0.0, "yawRateRadps": 0.0},
            scenario={"terrain": "flat", "seed": 7},
            leaseMs=500,
        ),
    )


def test_packet_is_canonical_bounded_and_round_trips() -> None:
    """Unsorted or reformatted packets must not become valid IPC messages."""
    message = _start_message()

    packet = encode_packet(message)

    assert len(packet) <= PACKET_MAX_BYTES
    assert packet == canonical_json(message)
    assert decode_packet(packet) == message


def test_receive_packet_accepts_only_canonical_packet_bytes() -> None:
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    try:
        packet = encode_packet(_start_message())
        parent.sendall(packet)
        assert receive_packet(child) == packet
    finally:
        parent.close()
        child.close()


@pytest.mark.parametrize("descriptor_count", [1, 32])
def test_receive_packet_rejects_control_data_without_leaking_received_fds(
    descriptor_count: int,
) -> None:
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    source_fd = os.open("/dev/null", os.O_RDONLY)
    before = {item.name for item in Path("/proc/self/fd").iterdir()}
    try:
        rights = array.array("i", [source_fd] * descriptor_count)
        parent.sendmsg(
            [encode_packet(_start_message())],
            [(socket.SOL_SOCKET, socket.SCM_RIGHTS, rights)],
        )
        with pytest.raises(ProtocolViolation, match="control data"):
            receive_packet(child)
        after = {item.name for item in Path("/proc/self/fd").iterdir()}
        assert after == before
    finally:
        os.close(source_fd)
        parent.close()
        child.close()


@pytest.mark.parametrize(
    "packet",
    [
        b"{" + b" " * PACKET_MAX_BYTES + b"}",
        b'{"protocol":"WRONG"}',
        b'{"protocol":"MICRODUCK_RUNTIME_IPC_V1","unknown":1}',
    ],
)
def test_packet_rejects_oversize_wrong_version_and_unknown_fields(packet: bytes) -> None:
    """Weak packet validation would let a peer bypass the private IPC contract."""
    with pytest.raises(ProtocolViolation):
        decode_packet(packet)


def test_decode_rejects_valid_but_noncanonical_json() -> None:
    """Dropping byte-for-byte canonical verification would make signed IPC semantics ambiguous."""
    packet = encode_packet(_start_message())

    with pytest.raises(ProtocolViolation, match="canonical"):
        decode_packet(packet.replace(b",", b", ", 1))


@pytest.mark.parametrize(
    "field,value",
    [
        ("generation", -1),
        ("generation", 2**64),
        ("operationSequence", True),
        ("taskId", "A" * 32),
    ],
)
def test_message_rejects_noncanonical_identity_values(field: str, value: object) -> None:
    """Weak generations or task identities could cross process generations incorrectly."""
    raw = _start_message().model_dump(mode="python", by_alias=True)
    raw[field] = value

    with pytest.raises(ValidationError):
        RuntimeMessage.model_validate(raw)


def test_command_reuses_the_public_strict_parameter_contract() -> None:
    """Allowing nested or raw control intent over IPC would bypass the V1 command boundary."""
    with pytest.raises(ValidationError, match="raw control key|JSON nesting"):
        CommandPayload(parameters={"jointTargets": 0}, leaseMs=500)


def test_ack_packet_decodes_its_kind_specific_payload() -> None:
    """A generic payload parser would let acknowledgements lose their command binding."""
    message = RuntimeMessage(
        kind="ACK",
        generation=7,
        operationSequence=12,
        taskId="0" * 32,
        payload=AckPayload(acknowledgedKind="START"),
    )

    decoded = decode_packet(encode_packet(message))

    assert isinstance(decoded.payload, AckPayload)
    assert decoded.payload.acknowledgedKind.value == "START"


@pytest.mark.parametrize(
    ("kind", "payload"),
    [
        ("HELLO", {"runtimeRevision": "mjlab-microduck@0.1.0"}),
        ("LOAD", {"bundleDigest": "sha256:" + "2" * 64}),
        (
            "READY",
            {
                "runtimeRevision": "mjlab-microduck@0.1.0",
                "bundleDigest": "sha256:" + "2" * 64,
            },
        ),
        ("SHUTDOWN", {"reason": "SUPERVISOR_SHUTDOWN"}),
    ],
)
def test_lifecycle_messages_require_a_null_task_id(kind: str, payload: dict[str, object]) -> None:
    """Giving a lifecycle packet a task ID would bind process setup to the wrong task."""
    message = RuntimeMessage(
        kind=kind,
        generation=7,
        operationSequence=1,
        taskId=None,
        payload=payload,
    )

    assert decode_packet(encode_packet(message)).taskId is None


@pytest.mark.parametrize(
    ("kind", "task_id", "payload"),
    [
        ("HELLO", "0" * 32, {"runtimeRevision": "mjlab-microduck@0.1.0"}),
        ("SHUTDOWN", "0" * 32, {"reason": "SUPERVISOR_SHUTDOWN"}),
        ("START", None, _start_message().payload),
        ("COMMAND", None, {"parameters": {"vxMps": 0.0}, "leaseMs": 500}),
    ],
)
def test_message_rejects_task_id_in_the_wrong_lifecycle_scope(
    kind: str, task_id: str | None, payload: object
) -> None:
    """Weak task scoping could apply a task operation to a process lifecycle packet."""
    with pytest.raises(ValidationError, match="taskId"):
        RuntimeMessage(
            kind=kind,
            generation=7,
            operationSequence=1,
            taskId=task_id,
            payload=payload,
        )


def test_status_request_and_response_have_unambiguous_wire_forms() -> None:
    """Conflating a status poll with a snapshot would force a parent to invent robot state."""
    request = RuntimeMessage(
        kind="STATUS",
        generation=7,
        operationSequence=2,
        taskId="0" * 32,
        payload={},
    )
    response = RuntimeMessage(
        kind="STATUS",
        generation=7,
        operationSequence=2,
        taskId="0" * 32,
        payload={"status": _robot_status_wire()},
    )

    assert decode_packet(encode_packet(request)).payload.model_dump() == {}
    assert decode_packet(encode_packet(response)).payload.model_dump(
        mode="json", by_alias=True, exclude_none=True
    ) == {
        "status": _robot_status_wire()
    }


def test_ready_echoes_only_the_verified_runtime_and_bundle_identity() -> None:
    """Dropping the bundle digest would let an unqualified child claim readiness."""
    message = RuntimeMessage(
        kind="READY",
        generation=7,
        operationSequence=3,
        taskId=None,
        payload={
            "runtimeRevision": "mjlab-microduck@0.1.0",
            "bundleDigest": "sha256:" + "2" * 64,
        },
    )

    assert decode_packet(encode_packet(message)).payload.model_dump() == {
        "runtimeRevision": "mjlab-microduck@0.1.0",
        "bundleDigest": "sha256:" + "2" * 64,
    }
    with pytest.raises(ValidationError):
        RuntimeMessage(
            kind="READY",
            generation=7,
            operationSequence=3,
            taskId=None,
            payload={"runtimeRevision": "mjlab-microduck@0.1.0"},
        )


def test_error_payload_allows_only_code_owned_sanitized_detail() -> None:
    """Permitting free-form native error text could leak host internals across IPC."""
    message = RuntimeMessage(
        kind="ERROR",
        generation=7,
        operationSequence=4,
        taskId="0" * 32,
        payload={
            "operationKind": "START",
            "code": "OPERATION_FAILED",
            "detail": {"retryable": False},
        },
    )

    assert decode_packet(encode_packet(message)).payload.model_dump() == {
        "operationKind": "START",
        "code": "OPERATION_FAILED",
        "detail": {"retryable": False},
    }
    with pytest.raises(ValidationError):
        RuntimeMessage(
            kind="ERROR",
            generation=7,
            operationSequence=4,
            taskId="0" * 32,
            payload={
                "operationKind": "START",
                "code": "OPERATION_FAILED",
                "detail": {"retryable": False, "message": "/proc/123/cmdline"},
            },
        )


@pytest.mark.parametrize(
    ("kind", "payload"),
    [
        ("ACK", {"acknowledgedKind": "SHUTDOWN"}),
        (
            "ERROR",
            {
                "operationKind": "LOAD",
                "code": "BUNDLE_REJECTED",
                "detail": {"retryable": False},
            },
        ),
    ],
)
def test_contextual_response_to_a_lifecycle_operation_requires_null_task_id(
    kind: str, payload: dict[str, object]
) -> None:
    """Binding a handshake response to a task would misroute a pre-task failure."""
    message = RuntimeMessage(
        kind=kind,
        generation=7,
        operationSequence=5,
        taskId=None,
        payload=payload,
    )

    assert decode_packet(encode_packet(message)).taskId is None


@pytest.mark.parametrize(
    ("kind", "task_id", "payload"),
    [
        ("ACK", "0" * 32, {"acknowledgedKind": "SHUTDOWN"}),
        ("ACK", None, {"acknowledgedKind": "START"}),
        (
            "ERROR",
            "0" * 32,
            {
                "operationKind": "LOAD",
                "code": "BUNDLE_REJECTED",
                "detail": {"retryable": False},
            },
        ),
        (
            "ERROR",
            None,
            {
                "operationKind": "START",
                "code": "OPERATION_FAILED",
                "detail": {"retryable": False},
            },
        ),
    ],
)
def test_contextual_response_rejects_a_task_id_from_the_wrong_operation_scope(
    kind: str, task_id: str | None, payload: dict[str, object]
) -> None:
    """Optional response task IDs would weaken correlation across process contexts."""
    with pytest.raises(ValidationError, match="taskId"):
        RuntimeMessage(
            kind=kind,
            generation=7,
            operationSequence=5,
            taskId=task_id,
            payload=payload,
        )


@pytest.mark.parametrize(
    ("kind", "payload"),
    [
        ("ACK", {"acknowledgedKind": "READY"}),
        ("ACK", {"acknowledgedKind": "ACK"}),
        ("ACK", {"acknowledgedKind": "ERROR"}),
        (
            "ERROR",
            {
                "operationKind": "TERMINAL",
                "code": "OPERATION_FAILED",
                "detail": {"retryable": False},
            },
        ),
        (
            "ERROR",
            {
                "operationKind": "ERROR",
                "code": "OPERATION_FAILED",
                "detail": {"retryable": False},
            },
        ),
    ],
)
def test_response_kinds_cannot_be_used_as_ack_or_error_operations(
    kind: str, payload: dict[str, object]
) -> None:
    """A response context would make response task scoping recursive and ambiguous."""
    with pytest.raises(ValidationError):
        RuntimeMessage(
            kind=kind,
            generation=7,
            operationSequence=6,
            taskId="0" * 32,
            payload=payload,
        )


def _robot_status_wire() -> dict[str, object]:
    return {
        "schema": "BIPED_POSE_V1",
        "timestamp": "2026-08-29T00:00:00Z",
        "basePositionM": [0.0, 0.0, 0.0],
        "baseOrientationXyzw": [0.0, 0.0, 0.0, 1.0],
        "baseLinearVelocityMps": [0.0, 0.0, 0.0],
        "baseAngularVelocityRadps": [0.0, 0.0, 0.0],
        "jointPositionsRad": [0.0] * 14,
        "jointVelocitiesRadps": [0.0] * 14,
        "policyTarget": {},
        "requestedMotion": {},
        "appliedMotion": {},
        "simulationTimeS": 0.0,
        "loopFrequencyHz": 50.0,
        "fallen": False,
        "limp": False,
        "health": {},
    }

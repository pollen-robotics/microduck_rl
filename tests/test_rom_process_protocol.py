from __future__ import annotations

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

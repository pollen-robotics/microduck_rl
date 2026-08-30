"""Strict private IPC records for the isolated MicroDuck runtime process.

This module is the only owner of IPC message names, envelope fields, payload
parsing, canonical encoding, and the packet-size limit.  It intentionally
depends only on the stable ROM contracts, never on MuJoCo, ONNX, or runtime
objects.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any, Literal

from pydantic import Field, ValidationError, field_validator, model_validator

from .contracts import (
    BoundedDescription,
    BoundedIdentifier,
    BoundedPath,
    ContractModel,
    ParameterObject,
    RobotStatus,
    Scenario,
    TaskEvidence,
    canonical_json,
)

PROTOCOL = "MICRODUCK_RUNTIME_IPC_V1"
PACKET_MAX_BYTES = 65_536
_TASK_ID_PATTERN = r"^[0-9a-f]{32}$"
_DIGEST_PATTERN = r"^sha256:[0-9a-f]{64}$"
_UINT64_MAX = 2**64 - 1


class ProtocolViolation(ValueError):
    """A peer supplied a packet outside the private runtime IPC contract."""


class RuntimeMessageKind(str, Enum):
    HELLO = "HELLO"
    LOAD = "LOAD"
    START = "START"
    COMMAND = "COMMAND"
    STATUS = "STATUS"
    ZERO_AND_STOP = "ZERO_AND_STOP"
    SHUTDOWN = "SHUTDOWN"
    READY = "READY"
    ACK = "ACK"
    TERMINAL = "TERMINAL"
    ERROR = "ERROR"


class HelloPayload(ContractModel):
    runtimeRevision: BoundedIdentifier


class LoadPayload(ContractModel):
    bundleDigest: str = Field(pattern=_DIGEST_PATTERN)
    bundleRoot: BoundedPath | None = None


class StartPayload(ContractModel):
    actionCode: BoundedIdentifier
    bundleDigest: str = Field(pattern=_DIGEST_PATTERN)
    parameters: ParameterObject
    scenario: Scenario
    leaseMs: int = Field(strict=True, gt=0, le=60_000)


class CommandPayload(ContractModel):
    parameters: ParameterObject
    leaseMs: int = Field(strict=True, gt=0, le=60_000)


class StatusPayload(ContractModel):
    status: RobotStatus


class ZeroAndStopPayload(ContractModel):
    reason: BoundedIdentifier


class ShutdownPayload(ContractModel):
    reason: BoundedIdentifier


class ReadyPayload(ContractModel):
    runtimeRevision: BoundedIdentifier


class AckPayload(ContractModel):
    acknowledgedKind: RuntimeMessageKind

    @field_validator("acknowledgedKind", mode="before")
    @classmethod
    def parse_acknowledged_kind(cls, value: Any) -> RuntimeMessageKind | Any:
        try:
            return RuntimeMessageKind(value)
        except (TypeError, ValueError):
            return value


class TerminalPayload(ContractModel):
    outcome: Literal["SUCCEEDED", "FAILED", "CANCELLED", "TIMED_OUT"]
    evidence: TaskEvidence


class ErrorPayload(ContractModel):
    code: BoundedIdentifier
    message: BoundedDescription


type RuntimePayload = (
    HelloPayload
    | LoadPayload
    | StartPayload
    | CommandPayload
    | StatusPayload
    | ZeroAndStopPayload
    | ShutdownPayload
    | ReadyPayload
    | AckPayload
    | TerminalPayload
    | ErrorPayload
)

_PAYLOAD_TYPES: dict[RuntimeMessageKind, type[RuntimePayload]] = {
    RuntimeMessageKind.HELLO: HelloPayload,
    RuntimeMessageKind.LOAD: LoadPayload,
    RuntimeMessageKind.START: StartPayload,
    RuntimeMessageKind.COMMAND: CommandPayload,
    RuntimeMessageKind.STATUS: StatusPayload,
    RuntimeMessageKind.ZERO_AND_STOP: ZeroAndStopPayload,
    RuntimeMessageKind.SHUTDOWN: ShutdownPayload,
    RuntimeMessageKind.READY: ReadyPayload,
    RuntimeMessageKind.ACK: AckPayload,
    RuntimeMessageKind.TERMINAL: TerminalPayload,
    RuntimeMessageKind.ERROR: ErrorPayload,
}


class RuntimeMessage(ContractModel):
    """Canonical envelope shared by the supervisor and child process."""

    protocol: Literal["MICRODUCK_RUNTIME_IPC_V1"] = PROTOCOL
    kind: RuntimeMessageKind
    generation: int = Field(strict=True, ge=0, le=_UINT64_MAX)
    operationSequence: int = Field(strict=True, ge=0, le=_UINT64_MAX)
    taskId: str = Field(pattern=_TASK_ID_PATTERN)
    payload: RuntimePayload

    @model_validator(mode="before")
    @classmethod
    def parse_discriminated_payload(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        kind = value.get("kind")
        try:
            parsed_kind = RuntimeMessageKind(kind)
        except (TypeError, ValueError):
            return value
        payload_type = _PAYLOAD_TYPES[parsed_kind]
        payload = value.get("payload")
        value = value.copy()
        value["kind"] = parsed_kind
        if not isinstance(payload, payload_type):
            value["payload"] = payload_type.model_validate(payload)
        return value

    @model_validator(mode="after")
    def payload_matches_kind(self) -> RuntimeMessage:
        if not isinstance(self.payload, _PAYLOAD_TYPES[self.kind]):
            raise TypeError("IPC payload does not match message kind")
        return self

    @classmethod
    def start(
        cls,
        *,
        generation: int,
        operationSequence: int,
        taskId: str,
        payload: StartPayload,
    ) -> RuntimeMessage:
        return cls(
            kind=RuntimeMessageKind.START,
            generation=generation,
            operationSequence=operationSequence,
            taskId=taskId,
            payload=payload,
        )


def encode_packet(message: RuntimeMessage) -> bytes:
    """Encode one bounded, canonical IPC packet."""
    if not isinstance(message, RuntimeMessage):
        raise TypeError("IPC packets require a RuntimeMessage")
    packet = canonical_json(message)
    if len(packet) > PACKET_MAX_BYTES:
        raise ProtocolViolation("IPC packet exceeds the 65,536-byte limit")
    return packet


def decode_packet(packet: bytes) -> RuntimeMessage:
    """Validate a bounded canonical UTF-8 packet and parse its strict envelope."""
    if not isinstance(packet, bytes):
        raise ProtocolViolation("IPC packet must be bytes")
    if len(packet) > PACKET_MAX_BYTES:
        raise ProtocolViolation("IPC packet exceeds the 65,536-byte limit")
    try:
        raw = json.loads(packet.decode("utf-8"))
        message = RuntimeMessage.model_validate(raw)
    except (UnicodeDecodeError, json.JSONDecodeError, ValidationError, ValueError) as exc:
        raise ProtocolViolation("invalid IPC packet") from exc
    if encode_packet(message) != packet:
        raise ProtocolViolation("IPC packet must use canonical JSON")
    return message

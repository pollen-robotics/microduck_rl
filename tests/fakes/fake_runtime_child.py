"""Deterministic exact-protocol runtime child used by process-supervisor tests."""

from __future__ import annotations

import argparse
import signal
import time
from datetime import UTC, datetime

from mjlab_microduck.rom.contracts import RobotStatus, TaskEvidence
from mjlab_microduck.rom.parent_death import verify_seqpacket_socket
from mjlab_microduck.rom.process_protocol import (
    AckPayload,
    ReadyPayload,
    RuntimeMessage,
    RuntimeMessageKind,
    RuntimeOperationKind,
    StatusPayload,
    TerminalPayload,
    decode_packet,
    encode_packet,
)

MODES = (
    "normal",
    "block-load",
    "block-start",
    "block-command",
    "block-status",
    "block-stop",
    "ignore-sigterm",
    "malformed-response",
    "late-response",
    "exit-before-ack",
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket-fd", required=True, type=int)
    parser.add_argument("--test-socket-fd", required=True, type=int)
    parser.add_argument("--mode", choices=MODES, required=True)
    return parser.parse_args()


def _status() -> RobotStatus:
    return RobotStatus(
        schema="BIPED_POSE_V1",
        timestamp=datetime(2026, 8, 29, tzinfo=UTC),
        basePositionM=(0.0, 0.0, 0.25),
        baseOrientationXyzw=(0.0, 0.0, 0.0, 1.0),
        baseLinearVelocityMps=(0.0, 0.0, 0.0),
        baseAngularVelocityRadps=(0.0, 0.0, 0.0),
        jointPositionsRad=(0.0,) * 14,
        jointVelocitiesRadps=(0.0,) * 14,
        policyTarget={},
        requestedMotion={},
        appliedMotion={},
        simulationTimeS=0.0,
        loopFrequencyHz=50.0,
        fallen=False,
        limp=False,
        health={"ready": True, "healthy": True},
    )


def _reply(request: RuntimeMessage, revision: str) -> RuntimeMessage:
    if request.kind is RuntimeMessageKind.LOAD:
        return RuntimeMessage(
            kind="READY",
            generation=request.generation,
            operationSequence=request.operationSequence,
            taskId=None,
            payload=ReadyPayload(
                runtimeRevision=revision,
                bundleDigest=request.payload.bundleDigest,  # type: ignore[union-attr]
            ),
        )
    if request.kind is RuntimeMessageKind.STATUS:
        return RuntimeMessage(
            kind="STATUS",
            generation=request.generation,
            operationSequence=request.operationSequence,
            taskId=request.taskId,
            payload=StatusPayload(status=_status()),
        )
    if request.kind is RuntimeMessageKind.ZERO_AND_STOP:
        return RuntimeMessage(
            kind="TERMINAL",
            generation=request.generation,
            operationSequence=request.operationSequence,
            taskId=request.taskId,
            payload=TerminalPayload(
                outcome="CANCELLED",
                evidence=TaskEvidence(
                    bundleDigest="sha256:" + "a" * 64,
                    policyDigest="sha256:" + "b" * 64,
                    modelDigest="sha256:" + "c" * 64,
                    stopReason=request.payload.reason,  # type: ignore[union-attr]
                ),
            ),
        )
    return RuntimeMessage(
        kind="ACK",
        generation=request.generation,
        operationSequence=request.operationSequence,
        taskId=request.taskId,
        payload=AckPayload(acknowledgedKind=RuntimeOperationKind(request.kind.value)),
    )


def main() -> int:
    args = _args()
    control = verify_seqpacket_socket(args.socket_fd)
    test_control = verify_seqpacket_socket(args.test_socket_fd)
    if args.mode == "ignore-sigterm":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    revision = "fake-runtime-v1"
    while True:
        packet = control.recv(65_537)
        if not packet:
            return 0
        request = decode_packet(packet)
        if request.kind is RuntimeMessageKind.HELLO:
            revision = request.payload.runtimeRevision  # type: ignore[union-attr]
        operation = request.kind.value.lower().replace("zero_and_stop", "stop")
        if args.mode == "exit-before-ack":
            return 17
        if args.mode == "malformed-response":
            control.sendall(b"{}")
            continue
        if args.mode in {f"block-{operation}", "late-response"}:
            test_control.sendall(request.kind.value.encode("ascii"))
            if not test_control.recv(1):
                return 0
            if args.mode == "late-response":
                time.sleep(0.05)
        control.sendall(encode_packet(_reply(request, revision)))


if __name__ == "__main__":
    raise SystemExit(main())

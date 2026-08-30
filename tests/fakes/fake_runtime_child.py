"""Deterministic exact-protocol runtime child used by process-supervisor tests."""

from __future__ import annotations

import argparse
import signal
from datetime import UTC, datetime

from mjlab_microduck.rom.contracts import RobotStatus, TaskEvidence
from mjlab_microduck.rom.parent_death import (
    close_unrelated_fds,
    install_parent_death_signal,
    verify_seqpacket_socket,
)
from mjlab_microduck.rom.process_protocol import (
    AckPayload,
    ReadyPayload,
    RuntimeMessage,
    RuntimeMessageKind,
    RuntimeOperationKind,
    StatusPayload,
    TerminalEventPayload,
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
    "terminal-event",
    "event-before-status",
    "event-before-command",
    "duplicate-event",
    "stale-event",
    "malformed-event",
    "lease-cleanup-failure",
)

PROOF_MODES = (
    "gate-malformed",
    "gate-exit",
    "stale-generation",
    "wrong-hello-ack",
    "wrong-start-ack",
    "wrong-command-ack",
    "wrong-shutdown-ack",
)

_late_control = None
_late_test_control = None
_late_request = None
_late_revision = "fake-runtime-v1"


def _release_late_response(_signum: int, _frame: object) -> None:
    """Release a response only after the parent's post-deadline SIGTERM."""
    if _late_control is None or _late_test_control is None or _late_request is None:
        return
    _late_control.sendall(encode_packet(_reply(_late_request, _late_revision)))
    _late_test_control.sendall(b"LATE_SENT")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket-fd", required=True, type=int)
    parser.add_argument("--test-socket-fd", required=True, type=int)
    parser.add_argument("--mode", choices=MODES + PROOF_MODES, required=True)
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


def _reply(
    request: RuntimeMessage,
    revision: str,
    *,
    generation: int | None = None,
    acknowledged_kind: RuntimeOperationKind | None = None,
) -> RuntimeMessage:
    if request.kind is RuntimeMessageKind.LOAD:
        return RuntimeMessage(
            kind="READY",
            generation=request.generation if generation is None else generation,
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
            generation=request.generation if generation is None else generation,
            operationSequence=request.operationSequence,
            taskId=request.taskId,
            payload=StatusPayload(status=_status()),
        )
    if request.kind is RuntimeMessageKind.ZERO_AND_STOP:
        return RuntimeMessage(
            kind="TERMINAL",
            generation=request.generation if generation is None else generation,
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
        generation=request.generation if generation is None else generation,
        operationSequence=request.operationSequence,
        taskId=request.taskId,
        payload=AckPayload(
            acknowledgedKind=acknowledged_kind
            or RuntimeOperationKind(request.kind.value)
        ),
    )


def _terminal_event(request: RuntimeMessage, *, generation: int | None = None) -> RuntimeMessage:
    return RuntimeMessage(
        kind="TERMINAL_EVENT", generation=request.generation if generation is None else generation,
        operationSequence=0, taskId=request.taskId,
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


def main() -> int:
    global _late_control, _late_request, _late_revision, _late_test_control
    args = _args()
    control = verify_seqpacket_socket(args.socket_fd)
    test_control = verify_seqpacket_socket(args.test_socket_fd)
    install_parent_death_signal()
    close_unrelated_fds({args.socket_fd, args.test_socket_fd})
    if args.mode == "ignore-sigterm":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    elif args.mode == "late-response":
        signal.signal(signal.SIGTERM, _release_late_response)
    revision = "fake-runtime-v1"
    while True:
        packet = control.recv(65_537)
        if not packet:
            return 0
        request = decode_packet(packet)
        if request.kind is RuntimeMessageKind.HELLO:
            revision = request.payload.runtimeRevision  # type: ignore[union-attr]
        operation = request.kind.value.lower().replace("zero_and_stop", "stop")
        if (
            request.kind is RuntimeMessageKind.STATUS and args.mode == "event-before-status"
        ) or (
            request.kind is RuntimeMessageKind.COMMAND and args.mode == "event-before-command"
        ):
            control.sendall(encode_packet(_terminal_event(request)))
        if args.mode in {"gate-malformed", "gate-exit"}:
            test_control.sendall(request.kind.value.encode("ascii"))
            if not test_control.recv(1):
                return 0
            if args.mode == "gate-exit":
                return 17
            control.sendall(b"{}")
            continue
        if args.mode == "exit-before-ack":
            return 17
        if args.mode == "malformed-response":
            control.sendall(b"{}")
            continue
        if args.mode == "late-response":
            _late_control = control
            _late_test_control = test_control
            _late_request = request
            _late_revision = revision
            test_control.sendall(request.kind.value.encode("ascii"))
            # The parent releases this receive by sending SIGTERM after its deadline.
            if not test_control.recv(1):
                return 0
            continue
        if args.mode == f"block-{operation}":
            test_control.sendall(request.kind.value.encode("ascii"))
            if not test_control.recv(1):
                return 0
        wrong_for = {
            "wrong-hello-ack": RuntimeMessageKind.HELLO,
            "wrong-start-ack": RuntimeMessageKind.START,
            "wrong-command-ack": RuntimeMessageKind.COMMAND,
            "wrong-shutdown-ack": RuntimeMessageKind.SHUTDOWN,
        }
        wrong_kind = wrong_for.get(args.mode)
        acknowledged_kind = None
        if request.kind is wrong_kind:
            acknowledged_kind = (
                RuntimeOperationKind.COMMAND
                if request.kind is not RuntimeMessageKind.COMMAND
                else RuntimeOperationKind.START
            )
        generation = request.generation
        if args.mode == "stale-generation" and request.kind is RuntimeMessageKind.START:
            test_control.sendall(b"START")
            if not test_control.recv(1):
                return 0
            generation -= 1
        control.sendall(
            encode_packet(
                _reply(
                    request,
                    revision,
                    generation=generation,
                    acknowledged_kind=acknowledged_kind,
                )
            )
        )
        if request.kind is RuntimeMessageKind.START and args.mode in {
            "terminal-event",
            "duplicate-event",
            "stale-event",
            "malformed-event",
            "lease-cleanup-failure",
        }:
            test_control.sendall(b"STARTED")
            if args.mode == "lease-cleanup-failure":
                if test_control.recv(4) == b"EMIT":
                    return 0
            elif args.mode == "terminal-event":
                if test_control.recv(4) == b"EMIT":
                    control.sendall(encode_packet(_terminal_event(request)))
            elif args.mode == "malformed-event":
                control.sendall(b'{"kind":"TERMINAL_EVENT"}')
            else:
                event = _terminal_event(request, generation=request.generation - 1 if args.mode == "stale-event" else None)
                control.sendall(encode_packet(event))
                if args.mode == "duplicate-event":
                    control.sendall(encode_packet(event))


if __name__ == "__main__":
    raise SystemExit(main())

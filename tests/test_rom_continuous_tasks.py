from __future__ import annotations

import sqlite3
import threading
import time
from datetime import UTC, datetime
from math import nan
from threading import Event, Thread

import pytest
from fastapi.testclient import TestClient

from mjlab_microduck.rom.action_catalog import (
    CODE_OWNED_ACTION_CODES,
    code_owned_action_definition,
)
from mjlab_microduck.rom.api import create_app
from mjlab_microduck.rom.contracts import (
    CONTROLLED_SERVO_JOINTS,
    OBSERVATION_FIELDS,
    ActionContract,
    ModelArtifact,
    ObservationContract,
    PolicyArtifact,
    PolicyBundle,
    TaskCommandRequest,
    TaskCreateRequest,
)
from mjlab_microduck.rom.runtime import RuntimeHandle
from mjlab_microduck.rom.service import (
    CommandSequenceConflict,
    InvalidParameters,
    NotReady,
    RobotBusy,
    SimulatorTaskService,
    StaleCommand,
)
from mjlab_microduck.rom.store import SqliteTaskStore
from tests.fakes.fake_microduck_runtime import FakeMicroduckRuntime, robot_status


class ControllableClock:
    """Monotonic test clock so lease tests never depend on wall-clock timing."""

    def __init__(self) -> None:
        self._now = 100.0

    def __call__(self) -> float:
        return self._now

    def advance_ms(self, milliseconds: int) -> None:
        self._now += milliseconds / 1_000


def command(
    *, sequence: int, vx: float = 0.0, lease_ms: int = 500
) -> TaskCommandRequest:
    return TaskCommandRequest(
        commandSequence=sequence,
        parameters={"vxMps": vx, "vyMps": 0.0, "yawRateRadps": 0.0},
        leaseMs=lease_ms,
    )


def _invoke_daemon(function):
    completed = Event()
    outcome: dict[str, object] = {}

    def invoke() -> None:
        try:
            outcome["result"] = function()
        except BaseException as exc:  # noqa: BLE001 - the caller inspects the outcome.
            outcome["error"] = exc
        finally:
            completed.set()

    thread = Thread(target=invoke, daemon=True)
    thread.start()
    return completed, outcome, thread


def _wait_for_terminal(service: SimulatorTaskService, task_id: str):
    deadline = time.monotonic() + 0.75
    while time.monotonic() < deadline:
        snapshot = service.get_task(task_id)
        if snapshot.state in {"FAILED", "CANCELLED", "TIMED_OUT"}:
            return snapshot
        time.sleep(0.005)
    raise AssertionError(f"task {task_id} did not terminalize")


def _assert_unresponsive_diagnostics(
    service: SimulatorTaskService,
    walk_request: TaskCreateRequest,
    runtime: FakeMicroduckRuntime,
) -> None:
    terminal = service.get_task(walk_request.taskId)
    assert terminal.state == "FAILED"
    assert terminal.stopReason == "RUNTIME_UNRESPONSIVE"
    assert terminal.evidence is not None
    assert terminal.evidence.metrics["safetyFailure"] == "RUNTIME_UNRESPONSIVE"
    ready, reasons = service.motion_readiness()
    assert ready is False
    assert "RUNTIME_UNRESPONSIVE" in reasons

    calls = [
        lambda: service.get_task(walk_request.taskId),
        lambda: service.events_after(walk_request.taskId, -1),
        service.robot_status,
        lambda: service.cancel_task(walk_request.taskId),
    ]
    for call in calls:
        completed, outcome, _ = _invoke_daemon(call)
        assert completed.wait(timeout=0.2)
        assert "error" not in outcome
    assert runtime.emergency_stop_calls == ["RUNTIME_UNRESPONSIVE"]

    token = "runtime-stall-token"
    with TestClient(create_app(service, token)) as client:
        auth = {"Authorization": f"Bearer {token}"}
        task_id = walk_request.taskId
        ready = client.get("/v1/ready", headers=auth)
        status = client.get("/v1/robot/status", headers=auth)
        task = client.get(f"/v1/tasks/{task_id}", headers=auth)
        events = client.get(f"/v1/tasks/{task_id}/events", headers=auth)
        cancel = client.post(f"/v1/tasks/{task_id}/cancel", headers=auth)
        create = client.post(
            "/v1/tasks",
            headers=auth,
            json=walk_request.model_copy(update={"taskId": "f" * 32}).model_dump(
                mode="json", by_alias=True
            ),
        )
        renew = client.put(
            f"/v1/tasks/{task_id}/command",
            headers=auth,
            json=command(sequence=99).model_dump(mode="json", by_alias=True),
        )

    assert ready.status_code == 200
    assert ready.json()["ready"] is False
    assert "RUNTIME_UNRESPONSIVE" in ready.json()["reasonCodes"]
    assert status.status_code == task.status_code == events.status_code == 200
    assert cancel.status_code == 200
    assert create.status_code == renew.status_code == 503
    assert create.json()["code"] == renew.json()["code"] == "NOT_READY"


def test_blocked_sample_fails_closed_without_holding_service_ownership(
    service, walk_request, runtime
):
    """A stalled sample must not defeat the lease watchdog or block diagnostics."""
    service._runtime_call_timeout_s = 0.05
    service.create_task(walk_request)
    runtime.sample_release.clear()
    tick_done, _, tick_thread = _invoke_daemon(service.tick)
    assert runtime.sample_started.wait(timeout=0.2)

    try:
        _wait_for_terminal(service, walk_request.taskId)
        _assert_unresponsive_diagnostics(service, walk_request, runtime)
        assert tick_done.wait(timeout=0.2)
        assert runtime.safe_stop_calls == []
    finally:
        runtime.sample_release.set()
        tick_thread.join(timeout=0.2)

    assert service.get_task(walk_request.taskId).state == "FAILED"
    assert service._active is None


def test_blocked_command_fails_closed_and_late_return_cannot_renew_ownership(
    service, walk_request, runtime
):
    """A hung command must terminalize once and its late return must be ignored."""
    service._runtime_call_timeout_s = 0.05
    service.create_task(walk_request)
    runtime.command_release.clear()
    command_done, _, command_thread = _invoke_daemon(
        lambda: service.command(
            walk_request.taskId, command(sequence=1, vx=0.2, lease_ms=500)
        )
    )
    assert runtime.command_started.wait(timeout=0.2)

    try:
        _wait_for_terminal(service, walk_request.taskId)
        _assert_unresponsive_diagnostics(service, walk_request, runtime)
        assert command_done.wait(timeout=0.2)
        assert runtime.safe_stop_calls == []
    finally:
        runtime.command_release.set()
        command_thread.join(timeout=0.2)

    terminal = service.get_task(walk_request.taskId)
    assert terminal.state == "FAILED"
    assert terminal.stopReason == "RUNTIME_UNRESPONSIVE"
    assert service._active is None


def test_blocked_safe_stop_becomes_unresponsive_without_duplicate_stop_attempts(
    service, walk_request, runtime
):
    """A hung safe stop must not retain the slot or trigger repeated stop attempts."""
    service._runtime_call_timeout_s = 0.05
    service.create_task(walk_request)
    runtime.safe_stop_release.clear()
    cancel_done, outcome, cancel_thread = _invoke_daemon(
        lambda: service.cancel_task(walk_request.taskId)
    )
    assert runtime.safe_stop_started.wait(timeout=0.2)

    try:
        _wait_for_terminal(service, walk_request.taskId)
        _assert_unresponsive_diagnostics(service, walk_request, runtime)
        assert cancel_done.wait(timeout=0.2)
        assert "error" not in outcome
        assert len(runtime.safe_stop_calls) == 1
    finally:
        runtime.safe_stop_release.set()
        cancel_thread.join(timeout=0.2)

    service.watchdog_failed()
    assert len(runtime.safe_stop_calls) == 1
    assert runtime.emergency_stop_calls == ["RUNTIME_UNRESPONSIVE"]


def test_newer_accepted_command_skips_older_delayed_publication(
    service, walk_request, runtime, monkeypatch
):
    """Concurrent callers may persist in order but must never publish in reverse."""
    service.create_task(walk_request)
    original_invoke = service._invoke_runtime
    older_accepted = Event()
    release_older = Event()

    def delay_older(operation, function, active, **kwargs):
        if (
            operation == "command"
            and threading.current_thread().name == "older-command"
        ):
            older_accepted.set()
            assert release_older.wait(timeout=0.5)
        return original_invoke(operation, function, active, **kwargs)

    monkeypatch.setattr(service, "_invoke_runtime", delay_older)
    outcome: dict[str, object] = {}

    def publish_older() -> None:
        try:
            outcome["result"] = service.command(
                walk_request.taskId, command(sequence=1, vx=0.1)
            )
        except BaseException as exc:  # noqa: BLE001 - assert the concurrent outcome.
            outcome["error"] = exc

    older_thread = Thread(target=publish_older, name="older-command", daemon=True)
    older_thread.start()
    assert older_accepted.wait(timeout=0.2)

    service.command(walk_request.taskId, command(sequence=2, vx=0.2))
    release_older.set()
    older_thread.join(timeout=0.2)

    assert not older_thread.is_alive()
    assert "error" not in outcome
    assert runtime.command_calls == [{"vxMps": 0.2, "vyMps": 0.0, "yawRateRadps": 0.0}]


def test_stop_claim_invalidates_queued_commands_before_zero_publication(
    service, walk_request, runtime, db_path, monkeypatch
):
    """Once stop is claimed, accepted commands cannot publish after zero intent."""
    service.create_task(walk_request)
    monkeypatch.setattr(service, "_require_motion_ready", lambda: None)
    runtime.command_release.clear()
    original_invoke = service._invoke_runtime
    zero_reached = Event()
    release_zero = Event()

    def delay_zero(operation, function, active, **kwargs):
        if operation == "zero_command":
            zero_reached.set()
            assert release_zero.wait(timeout=0.5)
        return original_invoke(operation, function, active, **kwargs)

    monkeypatch.setattr(service, "_invoke_runtime", delay_zero)
    first_done, _, first_thread = _invoke_daemon(
        lambda: service.command(walk_request.taskId, command(sequence=1, vx=0.1))
    )
    assert runtime.command_started.wait(timeout=0.2)
    second_done, second_outcome, second_thread = _invoke_daemon(
        lambda: service.command(walk_request.taskId, command(sequence=2, vx=0.2))
    )
    deadline = time.monotonic() + 0.2
    while time.monotonic() < deadline:
        with sqlite3.connect(db_path) as connection:
            sequence = connection.execute(
                "SELECT command_sequence FROM task WHERE task_id = ?",
                (walk_request.taskId,),
            ).fetchone()[0]
        if sequence == 2:
            break
        time.sleep(0.002)
    assert sequence == 2

    cancel_done, cancel_outcome, cancel_thread = _invoke_daemon(
        lambda: service.cancel_task(walk_request.taskId)
    )
    assert zero_reached.wait(timeout=0.2)
    runtime.command_release.set()
    assert first_done.wait(timeout=0.2)
    release_zero.set()
    assert second_done.wait(timeout=0.2)
    assert cancel_done.wait(timeout=0.2)
    first_thread.join(timeout=0.2)
    second_thread.join(timeout=0.2)
    cancel_thread.join(timeout=0.2)

    assert "error" not in second_outcome
    assert "error" not in cancel_outcome
    assert runtime.operation_log == [
        ("command", {"vxMps": 0.1, "vyMps": 0.0, "yawRateRadps": 0.0}),
        ("command", {"vxMps": 0.0, "vyMps": 0.0, "yawRateRadps": 0.0}),
        ("safe_stop", "CANCELLED"),
    ]


def test_runtime_supervisor_bounds_twenty_four_stalled_callers(service, runtime):
    """One wedged native call must not accumulate one thread per API caller."""
    service._runtime_call_timeout_s = 0.05
    runtime.status_started.clear()
    runtime.status_release.clear()
    worker_identities_before = {
        thread.ident
        for thread in threading.enumerate()
        if thread.name.startswith("microduck-runtime-")
    }
    callers = [_invoke_daemon(service.robot_status) for _ in range(24)]
    assert runtime.status_started.wait(timeout=0.2)

    try:
        for completed, outcome, _ in callers:
            assert completed.wait(timeout=0.2)
            assert "error" not in outcome
        added_workers = [
            thread
            for thread in threading.enumerate()
            if thread.name.startswith("microduck-runtime-")
            and thread.ident not in worker_identities_before
        ]
        assert runtime.status_call_count == 2
        assert added_workers == []
        assert service._runtime_dispatcher.worker.is_alive()
    finally:
        runtime.status_release.set()
        for _, _, thread in callers:
            thread.join(timeout=0.2)


def test_constructor_status_stall_is_bounded_and_fail_closed(bundle, store):
    """Application composition cannot hang forever on the initial status probe."""
    runtime = FakeMicroduckRuntime()
    runtime.status_release.clear()
    completed, outcome, constructor_thread = _invoke_daemon(
        lambda: SimulatorTaskService(bundle, store, runtime, runtimeCallTimeoutS=0.05)
    )

    try:
        assert runtime.status_started.wait(timeout=0.2)
        assert completed.wait(timeout=0.2)
        assert "error" not in outcome
        constructed = outcome["result"]
        assert isinstance(constructed, SimulatorTaskService)
        assert constructed.motion_readiness() == (
            False,
            ("RUNTIME_UNRESPONSIVE",),
        )
        assert constructed.robot_status().health["ready"] is False
        assert runtime.status_call_count == 1
    finally:
        runtime.status_release.set()
        constructor_thread.join(timeout=0.2)


def test_cancel_during_start_repeatedly_stops_the_returned_handle(
    service, walk_request, runtime
):
    """Capturing a pre-start None handle strands every late runtime owner."""
    for index in range(3):
        task_id = f"{index + 3:x}" * 32
        request = walk_request.model_copy(update={"taskId": task_id})
        runtime.started.clear()
        runtime.start_release.clear()
        runtime.emergency_stopped.clear()
        runtime.safe_stopped.clear()
        create_done, create_outcome, create_thread = _invoke_daemon(
            lambda request=request: service.create_task(request)
        )
        assert runtime.started.wait(timeout=0.2)
        cancel_done, cancel_outcome, cancel_thread = _invoke_daemon(
            lambda task_id=task_id: service.cancel_task(task_id)
        )

        deadline = time.monotonic() + 0.2
        while time.monotonic() < deadline:
            with service._lock:
                active = service._active
                stop_claimed = active is not None and active.stop_claimed
            if stop_claimed:
                break
            time.sleep(0.002)
        assert stop_claimed
        runtime.start_release.set()
        assert create_done.wait(timeout=0.2)
        assert cancel_done.wait(timeout=0.2)
        create_thread.join(timeout=0.2)
        cancel_thread.join(timeout=0.2)

        assert "error" not in create_outcome
        assert "error" not in cancel_outcome
        terminal = service.get_task(task_id)
        assert terminal.state == "CANCELLED"
        assert runtime.emergency_stop_calls[-1] == "CANCELLED"
        assert len(runtime.emergency_stop_calls) == index + 1
        assert runtime.safe_stop_calls[-1] == (
            RuntimeHandle(taskId=task_id),
            "CANCELLED",
        )
        assert len(runtime.safe_stop_calls) == index + 1
        assert runtime.active_handle is None


def test_watchdog_during_start_publishes_emergency_before_fifo_cleanup(
    service, walk_request, runtime
):
    """A watchdog stop cannot wait for a blocked start before publishing safety intent."""
    runtime.start_release.clear()
    create_done, _, create_thread = _invoke_daemon(
        lambda: service.create_task(walk_request)
    )
    assert runtime.started.wait(timeout=0.2)
    watchdog_done, watchdog_outcome, watchdog_thread = _invoke_daemon(
        service.watchdog_failed
    )

    try:
        assert runtime.emergency_stopped.wait(timeout=0.2)
    finally:
        runtime.start_release.set()
        create_done.wait(timeout=0.2)
        watchdog_done.wait(timeout=0.2)
        create_thread.join(timeout=0.2)
        watchdog_thread.join(timeout=0.2)

    assert "error" not in watchdog_outcome
    terminal = service.get_task(walk_request.taskId)
    assert terminal.state == "FAILED"
    assert terminal.stopReason == "WATCHDOG_FAILURE"
    assert runtime.emergency_stop_calls == ["WATCHDOG_FAILURE"]
    assert runtime.safe_stop_calls[-1] == (
        RuntimeHandle(taskId=walk_request.taskId),
        "WATCHDOG_FAILURE",
    )
    assert len(runtime.safe_stop_calls) == 1
    assert runtime.active_handle is None


def test_start_timeout_quarantines_slot_until_late_handle_cleanup_finishes(
    bundle, store, runtime, clock, walk_request
):
    """Clearing timed-out start ownership early lets a replacement race its handle."""
    service = SimulatorTaskService(
        bundle,
        store,
        runtime,
        monotonic_clock=clock,
        runtimeCallTimeoutS=0.05,
    )
    runtime.start_release.clear()
    runtime.safe_stop_release.clear()
    create_done, create_outcome, create_thread = _invoke_daemon(
        lambda: service.create_task(walk_request)
    )
    assert runtime.started.wait(timeout=0.2)
    assert create_done.wait(timeout=0.2)
    terminal = _wait_for_terminal(service, walk_request.taskId)
    assert terminal.stopReason == "RUNTIME_UNRESPONSIVE"
    try:
        with service._lock:
            timed_out_owner = service._active
            assert timed_out_owner is not None
            assert timed_out_owner.request.taskId == walk_request.taskId
            assert timed_out_owner.start_pending is True
            assert timed_out_owner.cleanup_pending is True
        blocked_replacement = walk_request.model_copy(update={"taskId": "6" * 32})
        with pytest.raises((NotReady, RobotBusy)):
            service.create_task(blocked_replacement)
        assert store.get(blocked_replacement.taskId) is None
        assert service.get_task(walk_request.taskId).state == "FAILED"
        assert service.events_after(walk_request.taskId, -1)[-1].eventType == (
            "TASK_FAILED"
        )
        assert service.cancel_task(walk_request.taskId).state == "FAILED"

        runtime.start_release.set()
        assert runtime.safe_stop_started.wait(timeout=0.2)
        with service._lock:
            cleanup_owner = service._active
            assert cleanup_owner is timed_out_owner
            assert cleanup_owner.handle == RuntimeHandle(taskId=walk_request.taskId)
            assert cleanup_owner.cleanup_pending is True
        assert runtime.emergency_stop_calls == [
            "RUNTIME_UNRESPONSIVE",
            "RUNTIME_UNRESPONSIVE",
        ]
        assert runtime.safe_stop_calls[-1] == (
            RuntimeHandle(taskId=walk_request.taskId),
            "RUNTIME_UNRESPONSIVE",
        )
        runtime.safe_stop_release.set()
        assert runtime.safe_stopped.wait(timeout=0.2)
    finally:
        runtime.start_release.set()
        runtime.safe_stop_release.set()
        create_thread.join(timeout=0.2)

    assert "error" in create_outcome
    assert runtime.safe_stop_calls[-1] == (
        RuntimeHandle(taskId=walk_request.taskId),
        "RUNTIME_UNRESPONSIVE",
    )
    assert runtime.active_handle is None
    deadline = time.monotonic() + 0.2
    while time.monotonic() < deadline:
        with service._lock:
            if service._active is None:
                break
        time.sleep(0.002)
    assert service._active is None

    replacement = SimulatorTaskService(bundle, store, runtime, monotonic_clock=clock)
    replacement_request = walk_request.model_copy(update={"taskId": "7" * 32})
    running = replacement.create_task(replacement_request)
    assert running.state == "RUNNING"
    assert runtime.active_handle == RuntimeHandle(taskId=replacement_request.taskId)
    replacement.cancel_task(replacement_request.taskId)


def test_start_timeout_after_handle_registration_keeps_cleanup_quarantine(
    bundle, store, runtime, clock, walk_request, monkeypatch
):
    """Publishing the handle before operation completion must not release its slot."""
    service = SimulatorTaskService(
        bundle,
        store,
        runtime,
        monotonic_clock=clock,
        runtimeCallTimeoutS=0.5,
    )
    completion_reached = Event()
    completion_release = Event()
    emergency_started = Event()
    emergency_release = Event()
    original_submit = service._runtime_dispatcher.submit
    original_emergency_stop = runtime.emergency_stop
    emergency_call_count = 0

    def block_first_emergency(reason: str) -> None:
        nonlocal emergency_call_count
        emergency_call_count += 1
        if emergency_call_count == 1:
            emergency_started.set()
            assert emergency_release.wait(timeout=1.0)
        original_emergency_stop(reason)

    def intercept_start_completion(operation) -> None:
        if operation.name == "start":
            service._runtime_call_timeout_s = 0.05
            original_completion_set = operation.completed.set

            def block_completion() -> None:
                completion_reached.set()
                assert runtime.active_handle == RuntimeHandle(
                    taskId=walk_request.taskId
                )
                assert service._active is not None
                assert service._active.handle == RuntimeHandle(
                    taskId=walk_request.taskId
                )
                assert service._active.start_pending is True
                assert completion_release.wait(timeout=1.0)
                original_completion_set()

            operation.completed.set = block_completion
        original_submit(operation)

    monkeypatch.setattr(
        service._runtime_dispatcher, "submit", intercept_start_completion
    )
    monkeypatch.setattr(runtime, "emergency_stop", block_first_emergency)
    runtime.safe_stop_release.clear()
    create_done, create_outcome, create_thread = _invoke_daemon(
        lambda: service.create_task(walk_request)
    )
    assert completion_reached.wait(timeout=0.5)
    assert emergency_started.wait(timeout=0.2)

    try:
        with service._lock:
            cleanup_owner = service._active
            assert cleanup_owner is not None
            assert cleanup_owner.handle == RuntimeHandle(taskId=walk_request.taskId)
        replacement = walk_request.model_copy(update={"taskId": "8" * 32})
        with pytest.raises((NotReady, RobotBusy)):
            service.create_task(replacement)
        assert store.get(replacement.taskId) is None

        completion_release.set()
        assert runtime.safe_stop_started.wait(timeout=0.2)
        with service._lock:
            assert service._active is cleanup_owner
        assert runtime.safe_stop_calls[-1] == (
            RuntimeHandle(taskId=walk_request.taskId),
            "RUNTIME_UNRESPONSIVE",
        )
        runtime.safe_stop_release.set()
        assert runtime.safe_stopped.wait(timeout=0.2)
        with service._lock:
            assert service._active is cleanup_owner

        emergency_release.set()
        assert create_done.wait(timeout=0.2)
        terminal = _wait_for_terminal(service, walk_request.taskId)
        assert terminal.stopReason == "RUNTIME_UNRESPONSIVE"
    finally:
        completion_release.set()
        emergency_release.set()
        runtime.safe_stop_release.set()
        create_thread.join(timeout=0.2)

    assert "error" in create_outcome
    deadline = time.monotonic() + 0.2
    while time.monotonic() < deadline:
        with service._lock:
            if service._active is None:
                break
        time.sleep(0.002)
    assert service._active is None


def test_start_timeout_retains_service_owner_until_emergency_attempt_finishes(
    bundle, store, runtime, clock, walk_request, monkeypatch
):
    """Timeout cleanup cannot clear ownership before publishing emergency intent."""
    service = SimulatorTaskService(
        bundle,
        store,
        runtime,
        monotonic_clock=clock,
        runtimeCallTimeoutS=0.05,
    )
    runtime.start_release.clear()
    emergency_started = Event()
    emergency_release = Event()
    original_emergency_stop = runtime.emergency_stop

    def blocked_emergency_stop(reason: str) -> None:
        emergency_started.set()
        emergency_release.wait()
        original_emergency_stop(reason)

    monkeypatch.setattr(runtime, "emergency_stop", blocked_emergency_stop)
    create_done, _, create_thread = _invoke_daemon(
        lambda: service.create_task(walk_request)
    )
    assert runtime.started.wait(timeout=0.2)
    assert emergency_started.wait(timeout=0.2)

    try:
        with service._lock:
            assert service._active is not None
            assert service._active.request.taskId == walk_request.taskId
        assert service.get_task(walk_request.taskId).state == "VALIDATING"
    finally:
        emergency_release.set()
        runtime.start_release.set()
        create_done.wait(timeout=0.2)
        create_thread.join(timeout=0.2)

    terminal = _wait_for_terminal(service, walk_request.taskId)
    assert terminal.stopReason == "RUNTIME_UNRESPONSIVE"
    assert service._active is None


def test_expired_lease_zeros_velocity_stops_and_times_out(
    service, walk_request, clock, runtime
):
    """Removing target-side expiry would leave the last nonzero velocity active indefinitely."""
    service.create_task(walk_request)
    service.command(walk_request.taskId, command(sequence=1, vx=0.2, lease_ms=500))

    clock.advance_ms(501)
    service.tick()

    assert runtime.last_command == {"vxMps": 0.0, "vyMps": 0.0, "yawRateRadps": 0.0}
    assert runtime.operation_log == [
        ("command", {"vxMps": 0.2, "vyMps": 0.0, "yawRateRadps": 0.0}),
        ("command", {"vxMps": 0.0, "vyMps": 0.0, "yawRateRadps": 0.0}),
        ("safe_stop", "LEASE_EXPIRED"),
    ]
    assert service.get_task(walk_request.taskId).state == "TIMED_OUT"


def test_watchdog_stops_terminal_runtime_fault_before_the_lease_deadline(
    service, walk_request, runtime
):
    """Polling only the lease would leave a fatally stopped control loop durably RUNNING."""
    service.create_task(walk_request)
    service.command(walk_request.taskId, command(sequence=1, vx=0.2, lease_ms=500))
    runtime.complete_next(
        state="FAILED",
        metrics={"loopOverruns": 3, "fallen": False},
        stop_reason="CONTROL_LOOP_OVERRUN",
    )

    service.tick()

    terminal = service.get_task(walk_request.taskId)
    assert terminal.state == "FAILED"
    assert terminal.stopReason == "CONTROL_LOOP_OVERRUN"
    assert terminal.evidence is not None
    assert terminal.evidence.stopReason == "CONTROL_LOOP_OVERRUN"
    assert terminal.evidence.metrics["loopOverruns"] == 3
    assert runtime.last_command == {
        "vxMps": 0.0,
        "vyMps": 0.0,
        "yawRateRadps": 0.0,
    }
    assert runtime.operation_log[-2:] == [
        ("command", {"vxMps": 0.0, "vyMps": 0.0, "yawRateRadps": 0.0}),
        ("safe_stop", "CONTROL_LOOP_OVERRUN"),
    ]


def test_continuous_create_requires_initial_lease(service, walk_request):
    invalid = walk_request.model_copy(update={"leaseMs": None})

    with pytest.raises(InvalidParameters):
        service.create_task(invalid)


def test_continuous_create_requires_typed_initial_command(service, walk_request):
    invalid = walk_request.model_copy(update={"parameters": {}})

    with pytest.raises(InvalidParameters):
        service.create_task(invalid)


@pytest.mark.parametrize(
    ("parameters", "lease_ms"),
    [
        ({"vxMps": 0.400001, "vyMps": 0.0, "yawRateRadps": 0.0}, 500),
        ({"vxMps": 0.0, "vyMps": 0.0, "yawRateRadps": 0.0}, 5_001),
    ],
)
def test_service_enforces_code_owned_command_and_lease_bounds_even_if_manifest_is_widened(
    bundle, store, runtime, clock, walk_request, parameters, lease_ms
) -> None:
    """A widened in-memory manifest must not widen the service execution boundary."""
    walk = bundle.actions[0]
    assert walk.lease is not None
    widened = walk.model_copy(
        update={
            "parameterSchema": {
                **walk.parameterSchema,
                "properties": {
                    **walk.parameterSchema["properties"],
                    "vxMps": {"type": "number", "minimum": -1_000, "maximum": 1_000},
                },
            },
            "lease": walk.lease.model_copy(update={"maxLeaseMs": 1_000_000}),
        }
    )
    unsafe_bundle = bundle.model_copy(
        update={"actions": [widened, *bundle.actions[1:]]}
    )
    service = SimulatorTaskService(bundle, store, runtime, monotonic_clock=clock)
    service._bundle = unsafe_bundle
    request = walk_request.model_copy(
        update={"parameters": parameters, "leaseMs": lease_ms}
    )

    with pytest.raises(InvalidParameters):
        service.create_task(request)


def test_service_constructor_rejects_a_partial_action_catalog(
    bundle, store, runtime
) -> None:
    """Direct service composition must not bypass the complete catalog trust boundary."""
    partial = bundle.model_copy(update={"actions": bundle.actions[:1]})

    with pytest.raises(ValueError, match="complete code-owned V1 action catalog"):
        SimulatorTaskService(partial, store, runtime)


def test_continuous_create_persists_initial_deadline_before_return(
    service, walk_request, db_path
):
    running = service.create_task(walk_request)

    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            "SELECT state, deadline_at FROM task WHERE task_id = ?",
            (walk_request.taskId,),
        ).fetchone()
    assert running.state == "RUNNING"
    assert row == ("RUNNING", "100.500000000")


def test_app_watchdog_expires_initial_lease_without_http_traffic(
    service, walk_request, clock, runtime
):
    """The target deadman is owned by application lifecycle, not request traffic."""
    app = create_app(service, "watchdog-token")
    with TestClient(app) as client:
        response = client.post(
            "/v1/tasks",
            headers={"Authorization": "Bearer watchdog-token"},
            json=walk_request.model_dump(mode="json"),
        )
        assert response.status_code == 202
        clock.advance_ms(501)
        assert runtime.safe_stopped.wait(timeout=1.0)
        terminal = _wait_for_terminal(service, walk_request.taskId)
        assert terminal.state == "TIMED_OUT"
        assert runtime.operation_log == [
            ("command", {"vxMps": 0.0, "vyMps": 0.0, "yawRateRadps": 0.0}),
            ("safe_stop", "LEASE_EXPIRED"),
        ]
    assert app.state.watchdog_thread is None


def test_app_watchdog_observes_runtime_fault_before_lease_without_http_traffic(
    service, walk_request, runtime
):
    """The lifecycle watchdog must poll runtime safety, not only the monotonic lease."""
    app = create_app(service, "watchdog-token")
    with TestClient(app) as client:
        response = client.post(
            "/v1/tasks",
            headers={"Authorization": "Bearer watchdog-token"},
            json=walk_request.model_dump(mode="json"),
        )
        assert response.status_code == 202
        runtime.complete_next(
            state="FAILED",
            metrics={"loopOverruns": 3},
            stop_reason="CONTROL_LOOP_OVERRUN",
        )
        assert runtime.safe_stopped.wait(timeout=1.0)
        terminal = _wait_for_terminal(service, walk_request.taskId)
        assert terminal.state == "FAILED"
        assert terminal.stopReason == "CONTROL_LOOP_OVERRUN"
        assert runtime.last_command == {
            "vxMps": 0.0,
            "vyMps": 0.0,
            "yawRateRadps": 0.0,
        }


def test_watchdog_exception_stops_active_motion_and_gates_only_new_motion(
    service, walk_request, runtime
):
    """A dead watchdog must fail closed without disabling stop or reconciliation reads."""
    service.create_task(walk_request)
    service.command(walk_request.taskId, command(sequence=1, vx=0.2, lease_ms=500))

    def fail_watchdog_tick():
        raise RuntimeError("injected watchdog failure")

    service.tick = fail_watchdog_tick
    app = create_app(service, "watchdog-token")
    auth = {"Authorization": "Bearer watchdog-token"}
    with TestClient(app) as client:
        assert runtime.safe_stopped.wait(timeout=1.0)
        terminal = _wait_for_terminal(service, walk_request.taskId)
        ready = client.get("/v1/ready", headers=auth)
        catalog = client.get("/v1/catalog", headers=auth)
        create = client.post(
            "/v1/tasks",
            headers=auth,
            json=walk_request.model_copy(update={"taskId": "4" * 32}).model_dump(
                mode="json"
            ),
        )
        renew = client.put(
            f"/v1/tasks/{walk_request.taskId}/command",
            headers=auth,
            json=command(sequence=2, vx=0.1).model_dump(mode="json"),
        )
        task = client.get(f"/v1/tasks/{walk_request.taskId}", headers=auth)
        events = client.get(f"/v1/tasks/{walk_request.taskId}/events", headers=auth)
        status = client.get("/v1/robot/status", headers=auth)
        cancel = client.post(f"/v1/tasks/{walk_request.taskId}/cancel", headers=auth)

    assert terminal.state == "FAILED"
    assert terminal.stopReason == "WATCHDOG_FAILURE"
    assert terminal.evidence is not None
    assert terminal.evidence.metrics["safetyFailure"] == "WATCHDOG_FAILURE"
    assert runtime.operation_log[-2:] == [
        ("command", {"vxMps": 0.0, "vyMps": 0.0, "yawRateRadps": 0.0}),
        ("safe_stop", "WATCHDOG_FAILURE"),
    ]
    assert ready.status_code == 200
    assert ready.json()["ready"] is False
    assert "WATCHDOG_UNHEALTHY" in ready.json()["reasonCodes"]
    walk = catalog.json()["actions"][0]
    assert walk["availability"] == "UNAVAILABLE"
    assert walk["unavailableReason"] == "WATCHDOG_UNHEALTHY"
    for response in (create, renew):
        assert response.status_code == 503
        assert response.json()["code"] == "NOT_READY"
    assert task.status_code == events.status_code == status.status_code == 200
    assert cancel.status_code == 200


def test_stale_command_does_not_renew_lease(service, walk_request, clock, runtime):
    """Accepting a lower sequence would let delayed network traffic keep motion alive."""
    service.create_task(walk_request)
    service.command(walk_request.taskId, command(sequence=2, lease_ms=100))
    clock.advance_ms(99)

    with pytest.raises(StaleCommand) as error:
        service.command(walk_request.taskId, command(sequence=1, vx=0.1, lease_ms=500))

    assert error.value.code == "STALE_COMMAND"
    clock.advance_ms(2)
    service.tick()
    assert service.get_task(walk_request.taskId).state == "TIMED_OUT"
    assert runtime.operation_log[-2:] == [
        ("command", {"vxMps": 0.0, "vyMps": 0.0, "yawRateRadps": 0.0}),
        ("safe_stop", "LEASE_EXPIRED"),
    ]


def test_identical_command_sequence_is_idempotent_without_renewing_lease(
    service, walk_request, clock, runtime
):
    """Renewing an equal command would turn retry traffic into an unintended keepalive."""
    service.create_task(walk_request)
    first = command(sequence=5, vx=0.2, lease_ms=100)
    service.command(walk_request.taskId, first)
    clock.advance_ms(99)

    service.command(walk_request.taskId, first)
    clock.advance_ms(2)
    service.tick()

    assert service.get_task(walk_request.taskId).state == "TIMED_OUT"
    assert runtime.operation_log == [
        ("command", {"vxMps": 0.2, "vyMps": 0.0, "yawRateRadps": 0.0}),
        ("command", {"vxMps": 0.0, "vyMps": 0.0, "yawRateRadps": 0.0}),
        ("safe_stop", "LEASE_EXPIRED"),
    ]


def test_reused_command_sequence_with_different_content_is_a_command_conflict(
    service, walk_request
):
    """Treating command reuse as a task-ID collision would expose the wrong recovery contract."""
    service.create_task(walk_request)
    service.command(walk_request.taskId, command(sequence=4, vx=0.1))

    with pytest.raises(CommandSequenceConflict) as error:
        service.command(walk_request.taskId, command(sequence=4, vx=0.2))

    assert error.value.code == "COMMAND_SEQUENCE_CONFLICT"


@pytest.mark.parametrize(
    "invalid",
    [
        command(sequence=1, vx=0.401),
        command(sequence=1, lease_ms=99),
        command(sequence=1, lease_ms=5_001),
        TaskCommandRequest.model_construct(
            commandSequence=1,
            parameters={"vxMps": nan, "vyMps": 0.0, "yawRateRadps": 0.0},
            leaseMs=500,
        ),
        TaskCommandRequest(
            commandSequence=1, parameters={"vxMps": 0.0, "vyMps": 0.0}, leaseMs=100
        ),
    ],
)
def test_command_rejects_out_of_manifest_parameter_or_lease_bounds(
    service, walk_request, invalid
):
    """Clamping or accepting partial commands would make ROM intent differ from the manifest."""
    service.create_task(walk_request)

    with pytest.raises(InvalidParameters) as error:
        service.command(walk_request.taskId, invalid)

    assert error.value.code == "PARAMETER_INVALID"


def test_higher_sequence_replaces_the_lease_deadline(service, walk_request, clock):
    """Ignoring a newer command would make a valid active controller time out on its old lease."""
    service.create_task(walk_request)
    service.command(walk_request.taskId, command(sequence=1, lease_ms=100))
    clock.advance_ms(99)
    service.command(walk_request.taskId, command(sequence=2, vx=0.1, lease_ms=500))
    clock.advance_ms(101)

    service.tick()

    assert service.get_task(walk_request.taskId).state == "RUNNING"


def test_late_higher_sequence_expires_before_command_persistence(
    service, walk_request, clock, runtime, db_path
):
    """Renewing after a missed deadline would allow a late controller to resurrect motion."""
    service.create_task(walk_request)
    service.command(walk_request.taskId, command(sequence=1, vx=0.2, lease_ms=100))
    with sqlite3.connect(db_path) as connection:
        before = connection.execute(
            "SELECT command_sequence, deadline_at FROM task WHERE task_id = ?",
            (walk_request.taskId,),
        ).fetchone()
    clock.advance_ms(101)

    with pytest.raises(InvalidParameters) as error:
        service.command(walk_request.taskId, command(sequence=2, vx=0.1, lease_ms=500))

    with sqlite3.connect(db_path) as connection:
        after = connection.execute(
            "SELECT command_sequence, deadline_at FROM task WHERE task_id = ?",
            (walk_request.taskId,),
        ).fetchone()
    assert error.value.code == "PARAMETER_INVALID"
    assert before == after == (1, "100.100000000")
    assert service.get_task(walk_request.taskId).state == "TIMED_OUT"
    assert runtime.operation_log == [
        ("command", {"vxMps": 0.2, "vyMps": 0.0, "yawRateRadps": 0.0}),
        ("command", {"vxMps": 0.0, "vyMps": 0.0, "yawRateRadps": 0.0}),
        ("safe_stop", "LEASE_EXPIRED"),
    ]


@pytest.mark.parametrize(
    ("failure", "safety_code", "terminal_state", "reason"),
    [
        ("zero_command_error", "ZERO_COMMAND_FAILED", "TIMED_OUT", "LEASE_EXPIRED"),
        ("safe_stop_error", "SAFE_STOP_FAILED", "CANCELLED", "CANCELLED"),
    ],
)
def test_safety_operation_failure_persists_requested_terminal_and_releases_slot(
    service, walk_request, clock, runtime, failure, safety_code, terminal_state, reason
):
    """Letting either safety failure escape would strand task ownership without a durable terminal result."""
    service.create_task(walk_request)
    service.command(walk_request.taskId, command(sequence=1, vx=0.2, lease_ms=100))
    setattr(runtime, failure, RuntimeError(failure))
    if terminal_state == "TIMED_OUT":
        clock.advance_ms(101)
        service.tick()
    else:
        service.cancel_task(walk_request.taskId)
    next_task = service.create_task(
        walk_request.model_copy(update={"taskId": "3" * 32})
    )

    terminal = service.get_task(walk_request.taskId)
    assert terminal.state == terminal_state
    assert next_task.state == "RUNNING"
    assert runtime.operation_log[:3] == [
        ("command", {"vxMps": 0.2, "vyMps": 0.0, "yawRateRadps": 0.0}),
        ("command", {"vxMps": 0.0, "vyMps": 0.0, "yawRateRadps": 0.0}),
        ("safe_stop", reason),
    ]
    assert service.events_after(walk_request.taskId, -1)[-1].payload == {
        "code": reason,
        "safetyCode": safety_code,
    }


def test_cancel_zeros_then_stops_when_runtime_health_is_degraded(
    service, walk_request, runtime
):
    """Health-gating cancel would prevent the safety stop exactly when runtime health is bad."""
    service.create_task(walk_request)
    service.command(walk_request.taskId, command(sequence=1, vx=0.2))
    runtime.status_value = robot_status(healthy=False)

    cancelled = service.cancel_task(walk_request.taskId)

    assert cancelled.state == "CANCELLED"
    assert runtime.operation_log == [
        ("command", {"vxMps": 0.2, "vyMps": 0.0, "yawRateRadps": 0.0}),
        ("command", {"vxMps": 0.0, "vyMps": 0.0, "yawRateRadps": 0.0}),
        ("safe_stop", "CANCELLED"),
    ]


def test_command_and_deadline_are_durable_with_the_accepted_command_event(
    service, walk_request, db_path
):
    """Separating command state from its event could recover a lease with no corresponding audit record."""
    service.create_task(walk_request)
    service.command(walk_request.taskId, command(sequence=3, vx=0.2, lease_ms=500))

    with sqlite3.connect(db_path) as connection:
        task = connection.execute(
            "SELECT command_sequence, command_canonical_json, command_hash, lease_expires_at, deadline_at "
            "FROM task WHERE task_id = ?",
            (walk_request.taskId,),
        ).fetchone()
        event = connection.execute(
            "SELECT event_type FROM task_event WHERE task_id = ? ORDER BY sequence DESC LIMIT 1",
            (walk_request.taskId,),
        ).fetchone()

    assert task[0] == 3
    assert (
        task[1]
        == '{"commandSequence":3,"leaseMs":500,"parameters":{"vxMps":0.2,"vyMps":0.0,"yawRateRadps":0.0}}'
    )
    assert task[2].startswith("sha256:")
    assert task[3] == task[4]
    assert event == ("TASK_COMMAND_ACCEPTED",)


@pytest.fixture
def clock() -> ControllableClock:
    return ControllableClock()


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "simulator.sqlite3"


@pytest.fixture
def store(db_path) -> SqliteTaskStore:
    return SqliteTaskStore(db_path)


@pytest.fixture
def runtime() -> FakeMicroduckRuntime:
    return FakeMicroduckRuntime()


@pytest.fixture
def service(bundle, store, runtime, clock) -> SimulatorTaskService:
    return SimulatorTaskService(bundle, store, runtime, monotonic_clock=clock)


@pytest.fixture
def walk_request() -> TaskCreateRequest:
    return TaskCreateRequest(
        schema="MICRODUCK_SIM_TASK_V1",
        taskId="2" * 32,
        actionCode="WALK_VELOCITY",
        bundleVersion="1.0.0",
        bundleDigest="sha256:" + "a" * 64,
        parameters={"vxMps": 0.0, "vyMps": 0.0, "yawRateRadps": 0.0},
        scenario={"terrain": "flat", "seed": 1},
        leaseMs=500,
        requestedBy="test-continuous",
    )


@pytest.fixture
def bundle() -> PolicyBundle:
    actions = [
        code_owned_action_definition(
            code,
            availability="AVAILABLE" if code == "WALK_VELOCITY" else "UNAVAILABLE",
            policy_ref="walk" if code == "WALK_VELOCITY" else None,
            unavailable_reason=(
                None if code == "WALK_VELOCITY" else "POLICY_ARTIFACT_MISSING"
            ),
        )
        for code in CODE_OWNED_ACTION_CODES
    ]
    return PolicyBundle(
        schema="MICRODUCK_POLICY_BUNDLE_V1",
        bundleId="microduck-test",
        bundleVersion="1.0.0",
        bundleDigest="sha256:" + "a" * 64,
        createdAt=datetime(2026, 8, 29, tzinfo=UTC),
        sourceRepository="microduck-rl",
        sourceCommit="c" * 40,
        robotModel="MICRODUCK",
        observationContract=ObservationContract(
            identifier="MICRODUCK_OBS_61_V1",
            dimension=61,
            fields=list(OBSERVATION_FIELDS),
            units={},
            normalization="BAKED_IN_ONNX",
        ),
        actionContract=ActionContract(
            identifier="MICRODUCK_ACTION_14_V1",
            dimension=14,
            joints=list(CONTROLLED_SERVO_JOINTS),
            units="rad",
            scaling={},
            clipping={},
        ),
        model=ModelArtifact(path="models/robot.xml", digest="sha256:" + "c" * 64),
        policies=[
            PolicyArtifact(
                policyRef="walk",
                path="policies/walk.onnx",
                digest="sha256:" + "b" * 64,
                taskId="Mjlab-Velocity-Flat-MicroDuck",
            )
        ],
        actions=actions,
        qualification={
            "modelTerrain": "flat",
            "scenarioProfile": "SEEDED_SERVO_RESET_V1",
        },
        license={},
    )

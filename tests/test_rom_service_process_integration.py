"""Structural integration gates for process-owned ROM task execution."""

from __future__ import annotations

import ast
import inspect
import sqlite3
import threading
import time
from pathlib import Path

import pytest

from mjlab_microduck.rom.contracts import TaskCommandRequest
from mjlab_microduck.rom.process_service import SimulatorTaskService
from mjlab_microduck.rom.process_supervisor import CorrelatedTerminalDelivery
from mjlab_microduck.rom.service import RuntimeException
from mjlab_microduck.rom.store import SqliteTaskStore
from tests.test_rom_continuous_tasks import bundle as walk_bundle_fixture
from tests.test_rom_continuous_tasks import walk_request as walk_request_fixture
from tests.test_rom_discrete_tasks import bundle as stand_bundle_fixture
from tests.test_rom_discrete_tasks import stand_request as stand_request_fixture
from tests.test_rom_process_supervisor import _supervisor

ROM_ROOT = Path(__file__).parents[1] / "src" / "mjlab_microduck" / "rom"

REPLACED_BEHAVIOR_COVERAGE = {
    "blocked sample / concrete runtime fault": "test_continuous_runtime_fault_is_durable_and_correlated",
    "blocked command and duplicate": "test_blocked_command_never_publishes_or_renews_and_duplicate_shares_failure",
    "blocked safe stop": "test_blocked_stop_persists_failure_only_after_containment",
    "cancel/watchdog during START": "test_cancel_queued_during_blocked_start_stops_once_after_start_ack",
    "START timeout and retained quarantine": "test_blocked_start_holds_slot_until_exact_reap_and_reads_stay_responsive",
    "fresh generation after reap": "test_create_rejected_until_reap_then_fresh_generation_succeeds",
    "direct runtime fault persistence": "test_continuous_runtime_fault_is_durable_and_correlated",
    "late/stale callback association": "test_stale_delivery_and_cached_old_terminal_cannot_mutate_fresh_task",
    "caller/process ownership bounds": "tests/test_rom_process_supervisor.py",
}


def test_public_service_has_no_parent_runtime_ownership_symbols():
    source = (ROM_ROOT / "service.py").read_text()
    implementation = (ROM_ROOT / "process_service.py").read_text()

    assert "_RuntimeDispatcher" not in source + implementation
    assert "_RuntimeOperation" not in source + implementation
    assert "_StartLifecycle" not in source + implementation
    assert "RuntimeHandle" not in source + implementation
    assert "SimulationRuntime" not in source + implementation


def test_replaced_behavior_skip_mapping_is_explicit_and_process_backed():
    assert len(REPLACED_BEHAVIOR_COVERAGE) == 9
    assert all(REPLACED_BEHAVIOR_COVERAGE.values())


def test_parent_rom_modules_do_not_import_native_runtime_libraries():
    violations = []
    for name in (
        "service.py",
        "process_service.py",
        "api.py",
        "main.py",
        "process_supervisor.py",
    ):
        path = ROM_ROOT / name
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = {alias.name.split(".", 1)[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                names = {(node.module or "").split(".", 1)[0]}
            else:
                continue
            if names & {"mujoco", "onnxruntime"}:
                violations.append(path.name)
    assert violations == []


def test_service_requires_a_supervisor_factory_not_a_runtime_handle():
    signature = inspect.signature(SimulatorTaskService)

    assert "supervisor_factory" in signature.parameters
    assert "runtime" not in signature.parameters


def test_main_composes_the_process_supervisor_without_native_runtime_imports():
    source = (ROM_ROOT / "main.py").read_text()

    assert "RuntimeProcessSupervisor" in source
    assert "MicroduckMujocoRuntime" not in source
    assert "from .runtime import" not in source


def _process_service(
    tmp_path, mode, *, discrete=False, timeout=0.75, monotonic_clock=time.monotonic
):
    holder = {}

    def factory(callback):
        supervisor, launch = _supervisor(
            mode, operation_timeout_s=timeout, terminal_callback=callback
        )
        holder.update(supervisor=supervisor, launch=launch)
        return supervisor

    bundle = (
        stand_bundle_fixture.__wrapped__()
        if discrete
        else walk_bundle_fixture.__wrapped__()
    )
    service = SimulatorTaskService(
        bundle,
        SqliteTaskStore(tmp_path / f"{mode}.sqlite3"),
        factory,
        runtimeCallTimeoutS=timeout,
        monotonic_clock=monotonic_clock,
    )
    request = (
        stand_request_fixture.__wrapped__()
        if discrete
        else walk_request_fixture.__wrapped__()
    )
    return service, request, holder["supervisor"], holder["launch"]


def _emit_terminal(launch):
    peer = launch.test_peer
    assert peer is not None
    peer.settimeout(1)
    assert peer.recv(64) == b"STARTED"
    peer.sendall(b"EMIT")


def _wait_state(service, task_id, state):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        snapshot = service.get_task(task_id)
        if snapshot.state == state:
            return snapshot
        time.sleep(0.005)
    raise AssertionError(f"{task_id} did not reach {state}")


def test_process_discrete_stand_success_and_event_paging(tmp_path):
    service, request, supervisor, launch = _process_service(
        tmp_path, "terminal-event", discrete=True
    )
    try:
        created = service.create_task(request)
        assert created.state == "ACCEPTED"
        _emit_terminal(launch)
        terminal = _wait_state(service, request.taskId, "SUCCEEDED")
        assert terminal.stopReason == "TASK_COMPLETE"
        first = service.events_after(request.taskId, -1, page_size=2)
        assert first
        assert service.create_task(request).state == "SUCCEEDED"
    finally:
        supervisor.close()


def test_failed_start_never_persists_running_or_started_event(tmp_path):
    service, request, supervisor, _launch = _process_service(
        tmp_path, "wrong-start-ack"
    )
    try:
        with pytest.raises(RuntimeException):
            service.create_task(request)
        terminal = service.get_task(request.taskId)
        events = service.events_after(request.taskId, -1)
        assert terminal.state == "FAILED"
        assert [event.eventType for event in events] == [
            "TASK_VALIDATING",
            "TASK_FAILED",
        ]
        assert supervisor.snapshot().slot_releasable is True
    finally:
        supervisor.close()


def test_blocked_start_holds_slot_until_exact_reap_and_reads_stay_responsive(tmp_path):
    service, request, supervisor, launch = _process_service(
        tmp_path, "block-start", timeout=0.75
    )
    outcome = {}
    creator = threading.Thread(
        target=lambda: _capture(outcome, lambda: service.create_task(request)),
        daemon=True,
    )
    creator.start()
    peer = launch.test_peer
    assert peer is not None
    peer.settimeout(1)
    assert peer.recv(64) == b"START"
    assert supervisor.snapshot().slot_releasable is False
    assert service.get_task(request.taskId).state == "VALIDATING"
    assert service.events_after(request.taskId, -1)
    assert service.robot_status().schema_ == "BIPED_POSE_V1"
    assert service.motion_readiness()[0] is False
    creator.join(timeout=2)
    assert isinstance(outcome.get("error"), RuntimeException)
    assert supervisor.snapshot().slot_releasable is True
    assert service.get_task(request.taskId).state == "FAILED"
    supervisor.close()


def _capture(target, function):
    try:
        target["result"] = function()
    except BaseException as exc:  # noqa: BLE001 - test captures exact public result.
        target["error"] = exc


def test_blocked_command_never_publishes_or_renews_and_duplicate_shares_failure(
    tmp_path,
):
    service, request, supervisor, launch = _process_service(
        tmp_path, "block-command", timeout=0.75
    )
    service.create_task(request)
    command = TaskCommandRequest(
        commandSequence=1,
        parameters={"vxMps": 0.1, "vyMps": 0.0, "yawRateRadps": 0.0},
        leaseMs=500,
    )
    outcomes = [{}, {}]
    first = threading.Thread(
        target=lambda: _capture(
            outcomes[0], lambda: service.command(request.taskId, command)
        ),
        daemon=True,
    )
    first.start()
    peer = launch.test_peer
    assert peer is not None
    peer.settimeout(1)
    assert peer.recv(64) == b"COMMAND"
    second = threading.Thread(
        target=lambda: _capture(
            outcomes[1], lambda: service.command(request.taskId, command)
        ),
        daemon=True,
    )
    second.start()
    with sqlite3.connect(tmp_path / "block-command.sqlite3") as connection:
        row = connection.execute(
            "SELECT command_sequence, deadline_at FROM task WHERE task_id = ?",
            (request.taskId,),
        ).fetchone()
    assert row == (None, row[1])
    first.join(timeout=2)
    second.join(timeout=2)
    assert all(isinstance(item.get("error"), RuntimeException) for item in outcomes)
    assert not any(
        event.eventType == "TASK_COMMAND_ACCEPTED"
        for event in service.events_after(request.taskId, -1)
    )
    supervisor.close()


def test_command_is_persisted_and_renews_only_after_exact_ack(tmp_path):
    service, request, supervisor, launch = _process_service(tmp_path, "block-command")
    service.create_task(request)
    command = TaskCommandRequest(
        commandSequence=7,
        parameters={"vxMps": 0.1, "vyMps": 0.0, "yawRateRadps": 0.0},
        leaseMs=500,
    )
    outcome = {}
    caller = threading.Thread(
        target=lambda: _capture(
            outcome, lambda: service.command(request.taskId, command)
        ),
        daemon=True,
    )
    caller.start()
    peer = launch.test_peer
    assert peer is not None
    peer.settimeout(1)
    assert peer.recv(64) == b"COMMAND"
    with sqlite3.connect(tmp_path / "block-command.sqlite3") as connection:
        before = connection.execute(
            "SELECT command_sequence FROM task WHERE task_id = ?",
            (request.taskId,),
        ).fetchone()[0]
    assert before is None
    peer.sendall(b"R")
    caller.join(timeout=2)
    assert "error" not in outcome
    with sqlite3.connect(tmp_path / "block-command.sqlite3") as connection:
        after = connection.execute(
            "SELECT command_sequence FROM task WHERE task_id = ?",
            (request.taskId,),
        ).fetchone()[0]
    assert after == 7
    service.cancel_task(request.taskId)
    supervisor.close()


@pytest.mark.parametrize(
    ("mode", "reason"),
    [
        ("terminal-fallen", "FALLEN"),
        ("terminal-overrun", "CONTROL_LOOP_OVERRUN"),
        ("terminal-nonfinite", "NON_FINITE_STATE"),
        ("terminal-runtime-exception", "RUNTIME_EXCEPTION"),
    ],
)
def test_continuous_runtime_fault_is_durable_and_correlated(tmp_path, mode, reason):
    service, request, supervisor, launch = _process_service(tmp_path, mode)
    try:
        service.create_task(request)
        _emit_terminal(launch)
        terminal = _wait_state(service, request.taskId, "FAILED")
        assert terminal.stopReason == reason
        assert terminal.evidence.stopReason == reason
    finally:
        supervisor.close()


def test_stale_delivery_and_cached_old_terminal_cannot_mutate_fresh_task(tmp_path):
    service, request, supervisor, launch = _process_service(tmp_path, "terminal-event")
    try:
        service.create_task(request)
        _emit_terminal(launch)
        old = _wait_state(service, request.taskId, "SUCCEEDED")
        deadline = time.monotonic() + 1
        while not supervisor.readiness() and time.monotonic() < deadline:
            time.sleep(0.005)
        fresh = request.model_copy(update={"taskId": "3" * 32})
        cached = supervisor.snapshot().cached_terminal
        assert cached is not None
        original_start = supervisor.start
        entered, release = threading.Event(), threading.Event()

        def delayed_start(candidate):
            entered.set()
            assert release.wait(timeout=1)
            return original_start(candidate)

        supervisor.start = delayed_start  # type: ignore[method-assign]
        creator = threading.Thread(
            target=lambda: service.create_task(fresh), daemon=True
        )
        creator.start()
        assert entered.wait(timeout=1)
        service.tick()
        assert service.get_task(fresh.taskId).state == "VALIDATING"
        release.set()
        creator.join(timeout=1)
        assert service.get_task(fresh.taskId).state == "RUNNING"
        stale = CorrelatedTerminalDelivery(
            cached.generation, request.taskId, cached.event_sequence, cached.terminal
        )
        with pytest.raises(RuntimeError, match="does not match"):
            service._terminal_callback(stale)
        assert service.get_task(fresh.taskId).state == "RUNNING"
        assert old.state == "SUCCEEDED"
    finally:
        supervisor.close()


def test_walk_start_renew_and_child_acknowledged_lease_timeout(tmp_path):
    now = [100.0]
    service, request, supervisor, _launch = _process_service(
        tmp_path, "normal", monotonic_clock=lambda: now[0]
    )
    try:
        running = service.create_task(request)
        assert running.state == "RUNNING"
        accepted = service.command(
            request.taskId,
            TaskCommandRequest(
                commandSequence=1,
                parameters={"vxMps": 0.1, "vyMps": 0.0, "yawRateRadps": 0.0},
                leaseMs=500,
            ),
        )
        assert accepted.state == "RUNNING"
        now[0] += 0.501
        service.tick()
        terminal = service.get_task(request.taskId)
        assert terminal.state == "TIMED_OUT"
        assert terminal.stopReason == "LEASE_EXPIRED"
    finally:
        supervisor.close()


def test_cancel_queued_during_blocked_start_stops_once_after_start_ack(tmp_path):
    service, request, supervisor, launch = _process_service(tmp_path, "block-start")
    create_outcome, cancel_outcome = {}, {}
    creator = threading.Thread(
        target=lambda: _capture(create_outcome, lambda: service.create_task(request)),
        daemon=True,
    )
    creator.start()
    peer = launch.test_peer
    assert peer is not None
    peer.settimeout(1)
    assert peer.recv(64) == b"START"
    canceller = threading.Thread(
        target=lambda: _capture(
            cancel_outcome, lambda: service.cancel_task(request.taskId)
        ),
        daemon=True,
    )
    canceller.start()
    peer.sendall(b"R")
    creator.join(timeout=2)
    canceller.join(timeout=2)
    assert "error" not in create_outcome | cancel_outcome
    assert service.get_task(request.taskId).state == "CANCELLED"
    assert (
        sum(
            event.eventType == "TASK_CANCEL_REQUESTED"
            for event in service.events_after(request.taskId, -1)
        )
        == 1
    )
    assert service.cancel_task(request.taskId).state == "CANCELLED"
    supervisor.close()


def test_blocked_stop_persists_failure_only_after_containment(tmp_path):
    service, request, supervisor, launch = _process_service(tmp_path, "block-stop")
    service.create_task(request)
    outcome = {}
    canceller = threading.Thread(
        target=lambda: _capture(outcome, lambda: service.cancel_task(request.taskId)),
        daemon=True,
    )
    canceller.start()
    peer = launch.test_peer
    assert peer is not None
    peer.settimeout(1)
    assert peer.recv(64) == b"ZERO_AND_STOP"
    assert supervisor.snapshot().slot_releasable is False
    assert service.get_task(request.taskId).state == "RUNNING"
    canceller.join(timeout=2)
    assert supervisor.snapshot().slot_releasable is True
    assert service.get_task(request.taskId).state == "FAILED"
    supervisor.close()


@pytest.mark.parametrize("mode", ["wrong-start-ack", "exit-start"])
def test_start_protocol_failure_or_crash_is_truthful_without_started_event(
    tmp_path, mode
):
    service, request, supervisor, _launch = _process_service(tmp_path, mode)
    try:
        with pytest.raises(RuntimeException):
            service.create_task(request)
        assert service.get_task(request.taskId).state == "FAILED"
        assert "TASK_STARTED" not in {
            event.eventType for event in service.events_after(request.taskId, -1)
        }
        assert supervisor.snapshot().slot_releasable is True
    finally:
        supervisor.close()


def test_create_rejected_until_reap_then_fresh_generation_succeeds(tmp_path):
    service, request, supervisor, launch = _process_service(tmp_path, "block-start")
    outcome = {}
    creator = threading.Thread(
        target=lambda: _capture(outcome, lambda: service.create_task(request)),
        daemon=True,
    )
    creator.start()
    peer = launch.test_peer
    assert peer is not None
    peer.settimeout(1)
    assert peer.recv(64) == b"START"
    replacement = request.model_copy(update={"taskId": "4" * 32})
    from mjlab_microduck.rom.service import RobotBusy

    with pytest.raises(RobotBusy):
        service.create_task(replacement)
    creator.join(timeout=2)
    assert supervisor.snapshot().slot_releasable is True
    launch.mode = "normal"
    next_task = service.create_task(replacement)
    assert next_task.state == "RUNNING"
    assert supervisor.snapshot().generation >= 2
    service.cancel_task(replacement.taskId)
    supervisor.close()

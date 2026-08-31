"""Structural integration gates for process-owned ROM task execution."""

from __future__ import annotations

import ast
import importlib
import inspect
import json
import os
import select
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import mjlab_microduck.rom.main as rom_main
from mjlab_microduck.rom.api import create_app
from mjlab_microduck.rom.contracts import TaskCommandRequest, TaskEvidence
from mjlab_microduck.rom.process_protocol import AckPayload, TerminalPayload
from mjlab_microduck.rom.process_service import SimulatorTaskService
from mjlab_microduck.rom.process_supervisor import (
    CorrelatedTerminalDelivery,
    ReapReceipt,
    StartAcknowledgement,
    SupervisorSnapshot,
    SupervisorUnavailable,
)
from mjlab_microduck.rom.service import RuntimeException
from mjlab_microduck.rom.store import SqliteTaskStore
from mjlab_microduck.rom.supervisor_state import SupervisorState
from tests.test_rom_continuous_tasks import bundle as walk_bundle_fixture
from tests.test_rom_continuous_tasks import walk_request as walk_request_fixture
from tests.test_rom_discrete_tasks import bundle as stand_bundle_fixture
from tests.test_rom_discrete_tasks import stand_request as stand_request_fixture
from tests.test_rom_process_supervisor import _gate_next_unsolicited_poll, _supervisor

ROM_ROOT = Path(__file__).parents[1] / "src" / "mjlab_microduck" / "rom"

REPLACED_BEHAVIOR_COVERAGE = {
    "test_blocked_sample_fails_closed_without_holding_service_ownership": (
        "tests.test_rom_runtime_child::test_blocked_continuous_monitor_on_lease_expiry_retires_without_event",
        "tests.test_rom_service_process_integration::test_malformed_unsolicited_packet_is_durable_after_exact_child_reap",
    ),
    "test_blocked_command_fails_closed_and_late_return_cannot_renew_ownership": (
        "tests.test_rom_service_process_integration::test_blocked_command_never_publishes_or_renews_and_duplicate_shares_failure",
    ),
    "test_blocked_safe_stop_becomes_unresponsive_without_duplicate_stop_attempts": (
        "tests.test_rom_service_process_integration::test_blocked_stop_persists_failure_only_after_containment",
        "tests.test_rom_runtime_child::test_normal_stop_unresponsive_evidence_withholds_correlated_terminal",
    ),
    "test_newer_accepted_command_skips_older_delayed_publication": (
        "tests.test_rom_service_process_integration::test_command_ack_and_terminal_callback_share_one_atomic_durable_result",
    ),
    "test_stop_claim_invalidates_queued_commands_before_zero_publication": (
        "tests.test_rom_service_process_integration::test_terminal_winning_after_command_ack_gives_duplicates_same_error",
    ),
    "test_runtime_supervisor_bounds_twenty_four_stalled_callers": (
        "tests.test_rom_process_supervisor::test_24_callers_share_one_owner_thread_and_one_child",
    ),
    "test_constructor_status_stall_is_bounded_and_fail_closed": (
        "tests.test_rom_process_supervisor::test_readiness_failure_reaps_exact_child_before_releasing_slot",
    ),
    "test_cancel_during_start_repeatedly_stops_the_returned_handle": (
        "tests.test_rom_service_process_integration::test_cancel_queued_during_blocked_start_stops_once_after_start_ack",
        "tests.test_rom_runtime_child::test_blocked_start_cannot_defeat_local_emergency_zero",
    ),
    "test_watchdog_during_start_publishes_emergency_before_fifo_cleanup": (
        "tests.test_rom_service_process_integration::test_watchdog_queued_during_blocked_start_fails_after_ack",
        "tests.test_rom_runtime_child::test_blocked_start_cannot_defeat_local_emergency_zero",
    ),
    "test_start_timeout_quarantines_slot_until_late_handle_cleanup_finishes": (
        "tests.test_rom_service_process_integration::test_blocked_start_holds_slot_until_exact_reap_and_reads_stay_responsive",
    ),
    "test_start_timeout_after_handle_registration_keeps_cleanup_quarantine": (
        "tests.test_rom_service_process_integration::test_create_rejected_until_reap_then_fresh_generation_succeeds",
    ),
    "test_start_timeout_retains_service_owner_until_emergency_attempt_finishes": (
        "tests.test_rom_process_supervisor::test_start_failure_is_quarantined_and_reaped",
    ),
    "test_safety_operation_failure_persists_requested_terminal_and_releases_slot": (
        "tests.test_rom_service_process_integration::test_blocked_stop_persists_failure_only_after_containment",
    ),
    "test_realtime_stop_during_blocked_start_leaves_no_runtime_owner_or_control": (
        "tests.test_rom_runtime_child::test_blocked_start_cannot_defeat_local_emergency_zero",
        "tests.test_rom_service_process_integration::test_cancel_queued_during_blocked_start_stops_once_after_start_ack",
    ),
    "test_realtime_emergency_after_final_start_check_revokes_publication": (
        "tests.test_rom_runtime_child::test_blocked_start_cannot_defeat_local_emergency_zero",
        "tests.test_rom_service_process_integration::test_watchdog_queued_during_blocked_start_fails_after_ack",
    ),
    "test_realtime_stop_after_runtime_start_return_uses_retained_cleanup_handle": (
        "tests.test_rom_service_process_integration::test_immediate_post_start_terminal_is_durable_before_callback_retries_expire",
        "tests.test_rom_runtime_child::test_blocked_start_cannot_defeat_local_emergency_zero",
    ),
    "test_service_tick_observes_concrete_runtime_fault_and_zeros_applied_motion": (
        "tests.test_rom_service_process_integration::test_continuous_runtime_fault_is_durable_and_correlated",
        "tests.test_rom_mujoco_runtime::test_runtime_fail_safe_stops_when_state_becomes_non_finite",
        "tests.test_rom_mujoco_runtime::test_qualified_stand_api_runs_from_accepted_to_succeeded",
    ),
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
    continuous_module = importlib.import_module("tests.test_rom_continuous_tasks")
    runtime_module = importlib.import_module("tests.test_rom_mujoco_runtime")
    assert set(REPLACED_BEHAVIOR_COVERAGE) == {
        *continuous_module._REPLACED_THREAD_LIFECYCLE_TESTS,
        *runtime_module._REPLACED_DIRECT_SERVICE_TESTS,
    }
    for replacements in REPLACED_BEHAVIOR_COVERAGE.values():
        for replacement in replacements:
            module_name, function_name = replacement.split("::", 1)
            assert callable(getattr(importlib.import_module(module_name), function_name))


def test_parent_rom_modules_do_not_import_native_runtime_libraries():
    violations = []
    for name in (
        "service.py",
        "process_service.py",
        "api.py",
        "main.py",
        "process_supervisor.py",
        "qualification.py",
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


def test_production_parent_import_closure_does_not_load_native_runtime_modules():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import mjlab_microduck.rom.main; "
                "import mjlab_microduck.rom.qualification; "
                "assert 'mujoco' not in sys.modules; "
                "assert 'onnxruntime' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_lifespan_surfaces_exact_reap_failure_instead_of_reporting_clean_shutdown():
    class CloseFailureService:
        shutdown_reap_receipt = ReapReceipt(
            generation=1, pid=23, ownership_identity=41
        )

        def tick(self) -> None:
            return None

        def watchdog_failed(self) -> None:
            return None

        def close(self) -> None:
            raise RuntimeError("sensitive containment detail")

    app = create_app(CloseFailureService(), "shutdown-token")  # type: ignore[arg-type]

    with (
        pytest.raises(RuntimeError, match="shutdown containment failed"),
        TestClient(app) as client,
    ):
        assert client.get("/v1/health").status_code == 200

    assert isinstance(app.state.shutdown_failure, RuntimeError)
    assert app.state.shutdown_reap_receipt is None


@pytest.mark.parametrize(
    "receipt",
    [
        ReapReceipt(generation=1, pid=23, ownership_identity=200),
        ReapReceipt(generation=2, pid=23, ownership_identity=199),
        ReapReceipt(generation=2, pid=24, ownership_identity=200),
    ],
    ids=("same-pid-old-generation", "same-pid-old-owner", "different-pid"),
)
def test_service_rejects_reap_receipt_not_bound_to_preclose_owner(
    tmp_path: Path, receipt: ReapReceipt
) -> None:
    bundle = stand_bundle_fixture.__wrapped__()

    class ReceiptSupervisor:
        def snapshot(self) -> SupervisorSnapshot:
            return SupervisorSnapshot(
                SupervisorState.IDLE,
                2,
                True,
                None,
                None,
                True,
                pid=23,
                ownership_identity=200,
            )

        def ensure_ready(self) -> SupervisorSnapshot:
            return self.snapshot()

        def close(self) -> ReapReceipt:
            return receipt

    supervisor = ReceiptSupervisor()
    service = SimulatorTaskService(
        bundle,
        SqliteTaskStore(tmp_path / "receipt.sqlite3"),
        lambda _callback: supervisor,  # type: ignore[arg-type,return-value]
    )

    service.close()

    assert service.shutdown_reap_receipt is None


def test_service_does_not_publish_receipt_when_close_raises(tmp_path: Path) -> None:
    bundle = stand_bundle_fixture.__wrapped__()
    exact = ReapReceipt(generation=2, pid=23, ownership_identity=200)

    class RaisingSupervisor:
        # A stale attribute must not rescue a failed close.
        reap_receipt = exact

        def snapshot(self) -> SupervisorSnapshot:
            return SupervisorSnapshot(
                SupervisorState.IDLE,
                2,
                True,
                None,
                None,
                True,
                pid=23,
                ownership_identity=200,
            )

        def ensure_ready(self) -> SupervisorSnapshot:
            return self.snapshot()

        def close(self) -> ReapReceipt:
            raise RuntimeError("exact reap unproven")

    supervisor = RaisingSupervisor()
    service = SimulatorTaskService(
        bundle,
        SqliteTaskStore(tmp_path / "raising.sqlite3"),
        lambda _callback: supervisor,  # type: ignore[arg-type,return-value]
    )

    with pytest.raises(RuntimeError, match="exact reap unproven"):
        service.close()

    assert service.shutdown_reap_receipt is None


def test_service_and_api_publish_only_exact_successful_close_receipt(
    tmp_path: Path,
) -> None:
    bundle = stand_bundle_fixture.__wrapped__()
    exact = ReapReceipt(generation=2, pid=23, ownership_identity=200)

    class ReceiptSupervisor:
        def snapshot(self) -> SupervisorSnapshot:
            return SupervisorSnapshot(
                SupervisorState.IDLE,
                2,
                True,
                None,
                None,
                True,
                pid=23,
                ownership_identity=200,
            )

        def ensure_ready(self) -> SupervisorSnapshot:
            return self.snapshot()

        def close(self) -> ReapReceipt:
            return exact

    supervisor = ReceiptSupervisor()
    service = SimulatorTaskService(
        bundle,
        SqliteTaskStore(tmp_path / "success.sqlite3"),
        lambda _callback: supervisor,  # type: ignore[arg-type,return-value]
    )
    app = create_app(service, "shutdown-token")

    with TestClient(app) as client:
        assert client.get("/v1/health").status_code == 200

    assert app.state.shutdown_failure is None
    assert service.shutdown_reap_receipt == exact
    assert app.state.shutdown_reap_receipt == exact


def test_pid1_server_exits_nonzero_when_lifespan_containment_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    app = SimpleNamespace(
        state=SimpleNamespace(shutdown_failure=None, shutdown_reap_receipt=None)
    )
    configured = SimpleNamespace(
        host="127.0.0.1", port=8000, state_db=tmp_path / "tasks.sqlite3"
    )
    captured: dict[str, object] = {}

    class FakeServer:
        def __init__(self, config) -> None:
            captured["config"] = config
            self.lifespan = SimpleNamespace(shutdown_failed=True)

        def run(self) -> None:
            factory, _kwargs = captured["config"]
            assert callable(factory)
            captured["app_created_before_run"] = False
            created_app = factory()
            assert created_app is app
            captured["app_created_during_run"] = True
            app.state.shutdown_failure = RuntimeError("exact reap unproven")

    monkeypatch.setattr(rom_main, "read_configuration", lambda: configured)
    app_created = False

    def create_after_signal_ownership():
        nonlocal app_created
        app_created = True
        return app

    monkeypatch.setattr(rom_main, "create_configured_app", create_after_signal_ownership)
    monkeypatch.setattr(
        rom_main.uvicorn,
        "Config",
        lambda application, **kwargs: (application, kwargs),
    )
    monkeypatch.setattr(rom_main.uvicorn, "Server", FakeServer)

    with pytest.raises(SystemExit) as failure:
        assert app_created is False
        rom_main.main()

    assert failure.value.code == 70
    factory, options = captured["config"]
    assert callable(factory)
    assert options == {
        "host": "127.0.0.1",
        "port": 8000,
        "timeout_graceful_shutdown": 1.0,
        "factory": True,
    }
    assert captured["app_created_before_run"] is False
    assert captured["app_created_during_run"] is True
    assert not (tmp_path / "tasks.sqlite3.shutdown-v1.json").exists()


def test_pid1_shutdown_evidence_records_exact_reap_before_exit(
    tmp_path: Path,
) -> None:
    """The durable marker must identify one exact child and an ordered PID1 exit."""
    state_db = tmp_path / "tasks.sqlite3"

    evidence_path = rom_main._write_shutdown_evidence(
        state_db,
        reap_receipt=ReapReceipt(
            generation=7, pid=23, ownership_identity=101
        ),
        exit_code=0,
    )

    assert evidence_path == tmp_path / "tasks.sqlite3.shutdown-v1.json"
    assert json.loads(evidence_path.read_text()) == {
        "events": [
            {"event": "CHILD_REAPED", "sequence": 0},
            {"event": "PID1_EXITING", "sequence": 1},
        ],
        "exitCode": 0,
        "exactReapConfirmed": True,
        "childGeneration": 7,
        "ownershipIdentity": 101,
        "pid1Pid": os.getpid(),
        "reapedChildPid": 23,
        "schema": "MICRODUCK_ROM_PID1_SHUTDOWN_V1",
    }


def test_pid1_main_writes_only_generation_bound_success_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    receipt = ReapReceipt(generation=4, pid=29, ownership_identity=303)
    app = SimpleNamespace(
        state=SimpleNamespace(
            shutdown_failure=None,
            shutdown_reap_receipt=receipt,
        )
    )
    configured = SimpleNamespace(
        host="127.0.0.1", port=8000, state_db=tmp_path / "tasks.sqlite3"
    )
    captured: dict[str, object] = {}

    class FakeServer:
        def __init__(self, config) -> None:
            captured["config"] = config
            self.lifespan = SimpleNamespace(shutdown_failed=False)

        def run(self) -> None:
            factory, _kwargs = captured["config"]
            assert factory() is app

    monkeypatch.setattr(rom_main, "read_configuration", lambda: configured)
    monkeypatch.setattr(rom_main, "create_configured_app", lambda: app)
    monkeypatch.setattr(
        rom_main.uvicorn,
        "Config",
        lambda application, **kwargs: (application, kwargs),
    )
    monkeypatch.setattr(rom_main.uvicorn, "Server", FakeServer)

    rom_main.main()

    evidence = json.loads(
        (tmp_path / "tasks.sqlite3.shutdown-v1.json").read_text()
    )
    assert evidence["reapedChildPid"] == receipt.pid
    assert evidence["childGeneration"] == receipt.generation
    assert evidence["ownershipIdentity"] == receipt.ownership_identity
    assert evidence["exitCode"] == 0


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


def _failed_terminal(request):
    return TerminalPayload(
        outcome="FAILED",
        evidence=TaskEvidence(
            bundleDigest=request.bundleDigest,
            policyDigest="sha256:" + "b" * 64,
            modelDigest="sha256:" + "c" * 64,
            stopReason="FALLEN",
        ),
    )


def _wait_state(service, task_id, state):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        snapshot = service.get_task(task_id)
        if snapshot.state == state:
            return snapshot
        time.sleep(0.005)
    raise AssertionError(f"{task_id} did not reach {state}")


def test_watchdog_durably_fails_post_ack_child_exit_without_public_operation(
    tmp_path,
):
    """A dead child must not remain cached RUNNING until a caller touches the API."""
    service, request, supervisor, launch = _process_service(
        tmp_path, "exit-after-start-ack"
    )
    try:
        service.create_task(request)
        peer = launch.test_peer
        assert peer is not None
        peer.settimeout(2)
        assert peer.recv(64) == b"STARTED"
        pid = supervisor.snapshot().pid
        assert pid is not None
        pidfd = os.pidfd_open(pid)
        release_poll = _gate_next_unsolicited_poll(supervisor)
        peer.sendall(b"EXIT")
        assert select.select([pidfd], [], [], 2)[0] == [pidfd]
        release_poll.set()

        deadline = time.monotonic() + 2
        while supervisor.snapshot().pid is not None and time.monotonic() < deadline:
            time.sleep(0.005)
        try:
            assert select.select([pidfd], [], [], 0)[0] == [pidfd]
        finally:
            os.close(pidfd)
        service.tick()
        terminal = service._store.get(request.taskId)
        assert terminal is not None
        assert terminal.state == "FAILED"
        assert terminal.stopReason == "RUNTIME_UNRESPONSIVE"
        assert supervisor.snapshot().state.value == "NO_CHILD"
    finally:
        service.close()


def test_terminal_then_immediate_child_exit_preserves_durable_success(tmp_path):
    """A queued truthful terminal must win over the child's subsequent EOF."""
    service, request, supervisor, launch = _process_service(
        tmp_path, "terminal-event-exit", discrete=True
    )
    try:
        service.create_task(request)
        peer = launch.test_peer
        assert peer is not None
        peer.settimeout(2)
        assert peer.recv(64) == b"STARTED"
        pid = supervisor.snapshot().pid
        assert pid is not None
        pidfd = os.pidfd_open(pid)
        release_poll = _gate_next_unsolicited_poll(supervisor)
        peer.sendall(b"EMIT")
        assert select.select([pidfd], [], [], 2)[0] == [pidfd]
        os.close(pidfd)
        release_poll.set()
        deadline = time.monotonic() + 2
        terminal = None
        while time.monotonic() < deadline:
            terminal = service._store.get(request.taskId)
            if (
                terminal is not None
                and terminal.state == "SUCCEEDED"
                and supervisor.snapshot().pid is None
            ):
                break
            time.sleep(0.005)
        assert terminal is not None
        assert terminal.state == "SUCCEEDED"
        assert terminal.stopReason == "TASK_COMPLETE"
        assert supervisor.snapshot().state.value == "NO_CHILD"
    finally:
        service.close()


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
        second = service.events_after(
            request.taskId, first[-1].sequence, page_size=2
        )
        third = service.events_after(
            request.taskId, second[-1].sequence, page_size=2
        )
        assert [event.sequence for event in first] == [0, 1]
        assert [event.sequence for event in second] == [2]
        assert third == []
        assert service.create_task(request).state == "SUCCEEDED"
    finally:
        supervisor.close()


def test_real_supervisor_http_create_and_result_are_process_backed(tmp_path):
    service, request, supervisor, launch = _process_service(
        tmp_path, "terminal-event", discrete=True
    )
    headers = {"Authorization": "Bearer process-token"}
    try:
        with TestClient(create_app(service, "process-token")) as client:
            created = client.post(
                "/v1/tasks", headers=headers, json=request.model_dump(mode="json")
            )
            assert created.status_code == 202
            assert created.json()["state"] == "ACCEPTED"
            _emit_terminal(launch)
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                result = client.get(
                    f"/v1/tasks/{request.taskId}", headers=headers
                )
                if result.json()["state"] == "SUCCEEDED":
                    break
                time.sleep(0.005)
            assert result.status_code == 200
            assert result.json()["state"] == "SUCCEEDED"
            assert result.json()["evidence"]["stopReason"] == "TASK_COMPLETE"
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


def test_immediate_post_start_terminal_is_durable_before_callback_retries_expire(
    tmp_path,
):
    """Delaying generation registration must not strand a completed task RUNNING."""
    service, request, supervisor, launch = _process_service(
        tmp_path, "terminal-event", discrete=True
    )
    original_start = supervisor.start

    def start_then_complete_before_return(candidate, register_acknowledgement=None):
        acknowledgement = original_start(candidate, register_acknowledgement)
        _emit_terminal(launch)
        deadline = time.monotonic() + 0.4
        while (
            "TERMINAL_DELIVERY_PERMANENT_FAILURE" not in supervisor.trace
            and time.monotonic() < deadline
        ):
            time.sleep(0.005)
        return acknowledgement

    supervisor.start = start_then_complete_before_return  # type: ignore[method-assign]
    try:
        service.create_task(request)
        terminal = _wait_state(service, request.taskId, "SUCCEEDED")
        assert terminal.stopReason == "TASK_COMPLETE"
        assert "TERMINAL_DELIVERY_PERMANENT_FAILURE" not in supervisor.trace
        deadline = time.monotonic() + 1
        while not supervisor.snapshot().slot_releasable and time.monotonic() < deadline:
            time.sleep(0.005)
        assert supervisor.snapshot().slot_releasable is True
    finally:
        supervisor.close()


def test_reap_between_readiness_and_start_registers_exact_new_generation(tmp_path):
    service, request, supervisor, launch = _process_service(
        tmp_path, "exit-after-ready"
    )
    first_peer = launch.test_peer
    assert first_peer is not None
    first_peer.settimeout(1)
    assert first_peer.recv(64) == b"READY"
    first_pid = supervisor.snapshot().pid
    assert first_pid is not None
    pidfd = os.pidfd_open(first_pid)
    original_start = supervisor.start

    def reap_then_start(candidate, register_acknowledgement=None):
        first_peer.sendall(b"X")
        readable, _, _ = select.select([pidfd], [], [], 1)
        assert readable == [pidfd]
        launch.mode = "normal"
        return original_start(candidate, register_acknowledgement)

    supervisor.start = reap_then_start  # type: ignore[method-assign]

    try:
        running = service.create_task(request)

        assert running.state == "RUNNING"
        assert supervisor.snapshot().generation == 2
        assert service._active is not None
        assert service._active.supervisor_generation == 2
        cancelled = service.cancel_task(request.taskId)
        assert cancelled.state == "CANCELLED"
        deadline = time.monotonic() + 1
        while not supervisor.snapshot().slot_releasable and time.monotonic() < deadline:
            time.sleep(0.005)
        assert supervisor.snapshot().slot_releasable is True
    finally:
        os.close(pidfd)
        supervisor.close()


def test_command_ack_and_terminal_callback_share_one_atomic_durable_result(
    tmp_path, monkeypatch
):
    """A terminal after ACK must give owner and duplicate the same explicit error."""
    service, request, supervisor, launch = _process_service(tmp_path, "block-command")
    service.create_task(request)
    command = TaskCommandRequest(
        commandSequence=9,
        parameters={"vxMps": 0.1, "vyMps": 0.0, "yawRateRadps": 0.0},
        leaseMs=500,
    )
    record_entered, record_release = threading.Event(), threading.Event()
    original_record = service._store.record_command

    def paused_record(*args, **kwargs):
        record_entered.set()
        assert record_release.wait(timeout=1)
        return original_record(*args, **kwargs)

    monkeypatch.setattr(service._store, "record_command", paused_record)
    outcomes = [{}, {}]
    callers = [
        threading.Thread(
            target=lambda outcome=outcome: _capture(
                outcome, lambda: service.command(request.taskId, command)
            ),
            daemon=True,
        )
        for outcome in outcomes
    ]
    callers[0].start()
    peer = launch.test_peer
    assert peer is not None
    peer.settimeout(1)
    assert peer.recv(64) == b"COMMAND"
    callers[1].start()
    peer.sendall(b"R")
    assert record_entered.wait(timeout=1)
    generation = supervisor.snapshot().generation
    delivery = CorrelatedTerminalDelivery(
        generation=generation,
        task_id=request.taskId,
        event_sequence=1,
        terminal=supervisor.snapshot().cached_terminal.terminal
        if supervisor.snapshot().cached_terminal is not None
        else _failed_terminal(request),
    )
    terminal_outcome = {}
    callback = threading.Thread(
        target=lambda: _capture(
            terminal_outcome, lambda: service._terminal_callback(delivery)
        ),
        daemon=True,
    )
    callback.start()
    record_release.set()
    for caller in callers:
        caller.join(timeout=2)
    callback.join(timeout=2)
    assert "error" not in terminal_outcome
    assert service.get_task(request.taskId).state == "FAILED"
    assert all("error" not in outcome for outcome in outcomes)
    results = [outcome.get("result") for outcome in outcomes]
    assert results[0] is not None
    assert results[0] == results[1]
    events = service.events_after(request.taskId, -1)
    command_index = next(
        index for index, event in enumerate(events)
        if event.eventType == "TASK_COMMAND_ACCEPTED"
    )
    terminal_index = next(
        index for index, event in enumerate(events)
        if event.eventType == "TASK_FAILED"
    )
    assert command_index < terminal_index
    supervisor.close()


def test_terminal_winning_after_command_ack_gives_duplicates_same_error(tmp_path):
    service, request, supervisor, _launch = _process_service(tmp_path, "normal")
    service.create_task(request)
    command = TaskCommandRequest(
        commandSequence=10,
        parameters={"vxMps": 0.1, "vyMps": 0.0, "yawRateRadps": 0.0},
        leaseMs=500,
    )
    acknowledged, finalize = threading.Event(), threading.Event()
    original_command = supervisor.command

    def pause_after_ack(task_id, candidate):
        result = original_command(task_id, candidate)
        acknowledged.set()
        assert finalize.wait(timeout=1)
        return result

    supervisor.command = pause_after_ack  # type: ignore[method-assign]
    outcomes = [{}, {}]
    callers = [
        threading.Thread(
            target=lambda outcome=outcome: _capture(
                outcome, lambda: service.command(request.taskId, command)
            ),
            daemon=True,
        )
        for outcome in outcomes
    ]
    callers[0].start()
    callers[1].start()
    assert acknowledged.wait(timeout=1)
    service._terminal_callback(
        CorrelatedTerminalDelivery(
            generation=supervisor.snapshot().generation,
            task_id=request.taskId,
            event_sequence=1,
            terminal=_failed_terminal(request),
        )
    )
    finalize.set()
    for caller in callers:
        caller.join(timeout=2)
    errors = [outcome.get("error") for outcome in outcomes]
    assert all(isinstance(error, RuntimeException) for error in errors)
    assert [str(error) for error in errors] == [
        "task terminalized during command",
        "task terminalized during command",
    ]
    assert all("result" not in outcome for outcome in outcomes)
    assert not any(
        event.eventType == "TASK_COMMAND_ACCEPTED"
        for event in service.events_after(request.taskId, -1)
    )
    supervisor.close()


def test_unexpected_supervisor_command_error_completes_duplicate_reservation(tmp_path):
    service, request, supervisor, _launch = _process_service(tmp_path, "normal")
    service.create_task(request)
    command = TaskCommandRequest(
        commandSequence=11,
        parameters={"vxMps": 0.1, "vyMps": 0.0, "yawRateRadps": 0.0},
        leaseMs=500,
    )
    entered, release = threading.Event(), threading.Event()

    def fail_unexpectedly(_task_id, _command):
        entered.set()
        assert release.wait(timeout=1)
        raise RuntimeError("raw supervisor defect")

    supervisor.command = fail_unexpectedly  # type: ignore[method-assign]
    outcomes = [{}, {}]
    callers = [
        threading.Thread(
            target=lambda outcome=outcome: _capture(
                outcome, lambda: service.command(request.taskId, command)
            ),
            daemon=True,
        )
        for outcome in outcomes
    ]
    callers[0].start()
    assert entered.wait(timeout=1)
    callers[1].start()
    pending = service._active.pending_command
    assert pending is not None
    release.set()
    for caller in callers:
        caller.join(timeout=2)

    errors = [outcome.get("error") for outcome in outcomes]
    assert all(isinstance(error, RuntimeException) for error in errors)
    assert [str(error) for error in errors] == [
        "simulator command was unresponsive",
        "simulator command was unresponsive",
    ]
    assert errors[0] is errors[1] is pending.error
    assert pending.done.is_set()
    assert service._active is None or service._active.pending_command is None
    assert service.get_task(request.taskId).state == "FAILED"
    deadline = time.monotonic() + 1
    while not supervisor.snapshot().slot_releasable and time.monotonic() < deadline:
        time.sleep(0.005)
    assert supervisor.snapshot().slot_releasable is True
    supervisor.close()


def test_store_failure_completes_owner_and_duplicate_with_same_error(
    tmp_path, monkeypatch
):
    service, request, supervisor, launch = _process_service(tmp_path, "block-command")
    service.create_task(request)
    command = TaskCommandRequest(
        commandSequence=12,
        parameters={"vxMps": 0.1, "vyMps": 0.0, "yawRateRadps": 0.0},
        leaseMs=500,
    )
    owner_ident = [None]
    fail_owner_read = threading.Event()
    original_get = service._store.get

    def failing_get(task_id):
        if fail_owner_read.is_set() and threading.get_ident() == owner_ident[0]:
            fail_owner_read.clear()
            raise RuntimeError("store read defect")
        return original_get(task_id)

    monkeypatch.setattr(service._store, "get", failing_get)
    outcomes = [{}, {}]

    def owner_call():
        owner_ident[0] = threading.get_ident()
        _capture(outcomes[0], lambda: service.command(request.taskId, command))

    callers = [
        threading.Thread(target=owner_call, daemon=True),
        threading.Thread(
            target=lambda: _capture(
                outcomes[1], lambda: service.command(request.taskId, command)
            ),
            daemon=True,
        ),
    ]
    callers[0].start()
    peer = launch.test_peer
    assert peer is not None
    peer.settimeout(1)
    assert peer.recv(64) == b"COMMAND"
    callers[1].start()
    pending = service._active.pending_command
    assert pending is not None
    fail_owner_read.set()
    peer.sendall(b"R")
    for caller in callers:
        caller.join(timeout=2)

    errors = [outcome.get("error") for outcome in outcomes]
    assert all(isinstance(error, RuntimeException) for error in errors)
    assert [str(error) for error in errors] == [
        "command durability failed",
        "command durability failed",
    ]
    assert errors[0] is errors[1] is pending.error
    assert pending.done.is_set()
    assert service._active is None or service._active.pending_command is None
    assert service.get_task(request.taskId).state == "FAILED"
    deadline = time.monotonic() + 1
    while not supervisor.snapshot().slot_releasable and time.monotonic() < deadline:
        time.sleep(0.005)
    assert supervisor.snapshot().slot_releasable is True
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

        def delayed_start(candidate, register_acknowledgement=None):
            entered.set()
            assert release.wait(timeout=1)
            return original_start(candidate, register_acknowledgement)

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
        assert service.get_task(request.taskId).state == "RUNNING"
        now[0] += 1.501
        service.tick()
        terminal = service.get_task(request.taskId)
        assert terminal.state == "TIMED_OUT"
        assert terminal.stopReason == "LEASE_EXPIRED"
    finally:
        supervisor.close()


def test_watchdog_stop_defers_to_exact_queued_child_lease_terminal(tmp_path):
    """A queued child deadman result must beat the parent's concurrent stop claim."""
    now = [100.0]
    bundle = walk_bundle_fixture.__wrapped__()
    request = walk_request_fixture.__wrapped__()
    policy = bundle.policies[0]
    delivery = CorrelatedTerminalDelivery(
        generation=1,
        task_id=request.taskId,
        event_sequence=1,
        terminal=TerminalPayload(
            outcome="TIMED_OUT",
            evidence=TaskEvidence(
                bundleDigest=bundle.bundleDigest,
                policyDigest=policy.digest,
                modelDigest=bundle.model.digest,
                metrics={},
                stopReason="LEASE_EXPIRED",
            ),
        ),
    )

    class QueuedTerminalSupervisor:
        def __init__(self, callback) -> None:
            self.callback = callback
            self.running = False
            self.terminal_queued = False

        def snapshot(self):
            if self.terminal_queued:
                return SupervisorSnapshot(
                    SupervisorState.IDLE,
                    1,
                    True,
                    None,
                    None,
                    False,
                    True,
                    delivery,
                )
            return SupervisorSnapshot(
                SupervisorState.RUNNING if self.running else SupervisorState.IDLE,
                1,
                True,
                None,
                None,
                not self.running,
            )

        def ensure_ready(self):
            return self.snapshot()

        def readiness(self):
            return not self.running

        def start(
            self, candidate, register_acknowledgement=None, register_dispatch=None
        ):
            if register_dispatch is not None:
                register_dispatch()
            result = StartAcknowledgement(
                1, candidate.taskId, AckPayload(acknowledgedKind="START")
            )
            if register_acknowledgement is not None:
                register_acknowledgement(result)
            self.running = True
            return result

        def stop(self, _task_id, _reason, register_dispatch=None):
            if register_dispatch is not None:
                register_dispatch()
            self.terminal_queued = True
            raise SupervisorUnavailable("terminal delivery already queued")

        def close(self):
            return None

    holder = {}

    def factory(callback):
        supervisor = QueuedTerminalSupervisor(callback)
        holder["supervisor"] = supervisor
        return supervisor

    service = SimulatorTaskService(
        bundle,
        SqliteTaskStore(tmp_path / "queued-terminal.sqlite3"),
        factory,
        monotonic_clock=lambda: now[0],
    )
    service.create_task(request)
    now[0] += request.leaseMs / 1000 + 0.001
    service.tick()
    assert service.get_task(request.taskId).state == "RUNNING"

    holder["supervisor"].callback(delivery)
    terminal = service.get_task(request.taskId)
    assert terminal.state == "TIMED_OUT"
    assert terminal.stopReason == "LEASE_EXPIRED"


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


def test_watchdog_queued_during_blocked_start_fails_after_ack(tmp_path):
    service, request, supervisor, launch = _process_service(tmp_path, "block-start")
    create_outcome, watchdog_outcome = {}, {}
    creator = threading.Thread(
        target=lambda: _capture(create_outcome, lambda: service.create_task(request)),
        daemon=True,
    )
    creator.start()
    peer = launch.test_peer
    assert peer is not None
    peer.settimeout(1)
    assert peer.recv(64) == b"START"
    watchdog = threading.Thread(
        target=lambda: _capture(watchdog_outcome, service.watchdog_failed),
        daemon=True,
    )
    watchdog.start()
    peer.sendall(b"R")
    creator.join(timeout=2)
    watchdog.join(timeout=2)
    assert "error" not in create_outcome | watchdog_outcome
    terminal = service.get_task(request.taskId)
    assert terminal.state == "FAILED"
    assert terminal.stopReason == "WATCHDOG_FAILURE"
    assert (
        sum(
            event.eventType == "TASK_FAILED"
            for event in service.events_after(request.taskId, -1)
        )
        == 1
    )
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


def test_malformed_unsolicited_packet_is_durable_after_exact_child_reap(tmp_path):
    service, request, supervisor, _launch = _process_service(
        tmp_path, "malformed-event"
    )
    try:
        service.create_task(request)
        deadline = time.monotonic() + 2
        while not supervisor.snapshot().slot_releasable and time.monotonic() < deadline:
            time.sleep(0.005)
        service.tick()
        terminal = service.get_task(request.taskId)
        assert terminal.state == "FAILED"
        assert terminal.stopReason == "RUNTIME_UNRESPONSIVE"
        assert "TASK_STARTED" in {
            event.eventType for event in service.events_after(request.taskId, -1)
        }
        assert "QUARANTINED" in supervisor.trace
        assert "CHILD_REAPED" in supervisor.trace
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

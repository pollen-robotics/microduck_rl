from __future__ import annotations

import time
from dataclasses import replace
from datetime import UTC, datetime
from threading import Event

import pytest

from mjlab_microduck.rom.action_catalog import (
    CODE_OWNED_ACTION_CODES,
    code_owned_action_definition,
)
from mjlab_microduck.rom.contracts import (
    CONTROLLED_SERVO_JOINTS,
    OBSERVATION_FIELDS,
    ActionContract,
    ModelArtifact,
    ObservationContract,
    PolicyArtifact,
    PolicyBundle,
    TaskCreateRequest,
    canonical_json,
    sha256_prefixed,
)
from mjlab_microduck.rom.runtime import RuntimeEvidence, RuntimeSample
from mjlab_microduck.rom.service import (
    ActionUnavailable,
    BundleMismatch,
    InvalidParameters,
    PreconditionFailed,
    RobotBusy,
    RuntimeException,
    SimulatorTaskService,
)
from mjlab_microduck.rom.store import SqliteTaskStore
from tests.fakes.fake_microduck_runtime import FakeMicroduckRuntime, robot_status


def test_discrete_task_records_terminal_evidence(service, stand_request, runtime):
    """Dropping provenance or bounded completion metrics would make a result unreplayable."""
    runtime.complete_next(state="SUCCEEDED", metrics={"upright": True})

    created = service.create_task(stand_request)
    terminal = wait_for_state(
        service, created.taskId, "SUCCEEDED", runtime.safe_stopped
    )

    assert terminal.evidence is not None
    assert terminal.evidence.bundleDigest == stand_request.bundleDigest
    assert terminal.evidence.policyDigest == "sha256:" + "b" * 64
    assert terminal.evidence.metrics == {"safeStop": True, "upright": True}
    assert len(runtime.safe_stop_calls) == 1


def test_only_one_motion_task_runs(service, stand_request, kick_request, runtime):
    """Removing active-slot ownership would let conflicting motions reach one robot."""
    service.create_task(stand_request)
    assert runtime.started.wait(timeout=1.0)

    with pytest.raises(RobotBusy) as error:
        service.create_task(kick_request)

    assert error.value.code == "ROBOT_BUSY"


def test_rejects_unavailable_action_before_creating_task(
    runtime, store, stand_request, bundle
):
    """Treating unavailable artifacts as executable would start a policy with no release evidence."""
    unavailable = bundle.model_copy(
        update={
            "actions": [
                action.model_copy(
                    update={
                        "availability": "UNAVAILABLE",
                        "policyRef": None,
                        "unavailableReason": "POLICY_ARTIFACT_MISSING",
                    }
                )
                if action.actionCode == "STAND"
                else action
                for action in bundle.actions
            ]
        }
    )
    service = SimulatorTaskService(unavailable, store, runtime)

    with pytest.raises(ActionUnavailable) as error:
        service.create_task(stand_request)

    assert error.value.code == "ACTION_UNAVAILABLE"
    assert store.get(stand_request.taskId) is None


def test_rejects_bundle_mismatch_before_creating_task(service, stand_request, store):
    """Ignoring a requested release digest could silently run a different policy than selected."""
    wrong_digest = stand_request.model_copy(
        update={"bundleDigest": "sha256:" + "f" * 64}
    )

    with pytest.raises(BundleMismatch) as error:
        service.create_task(wrong_digest)

    assert error.value.code == "BUNDLE_MISMATCH"
    assert store.get(wrong_digest.taskId) is None


def test_rejects_parameters_outside_action_schema(service, stand_request, store):
    """Relaxing additionalProperties would admit undeclared task controls into execution."""
    invalid = stand_request.model_copy(update={"parameters": {"unexpected": 1}})

    with pytest.raises(InvalidParameters) as error:
        service.create_task(invalid)

    assert error.value.code == "PARAMETER_INVALID"
    assert store.get(invalid.taskId) is None


def test_rejects_a_lease_for_a_discrete_action(service, stand_request, store):
    """Accepting a lease on a discrete action would blur its bounded completion contract."""
    leased = stand_request.model_copy(update={"leaseMs": 100})

    with pytest.raises(InvalidParameters) as error:
        service.create_task(leased)

    assert error.value.code == "PARAMETER_INVALID"
    assert store.get(leased.taskId) is None


def test_rejects_nonflat_or_unhealthy_runtime_before_creating_task(
    service, stand_request, runtime, store
):
    """Skipping terrain and health gates would command an unqualified or unready robot."""
    wrong_terrain = stand_request.model_copy(update={"scenario": {"terrain": "stairs"}})
    with pytest.raises(PreconditionFailed) as terrain_error:
        service.create_task(wrong_terrain)
    assert terrain_error.value.code == "PRECONDITION_FAILED"

    runtime.status_value = robot_status(healthy=False)
    with pytest.raises(PreconditionFailed) as health_error:
        service.create_task(stand_request)
    assert health_error.value.code == "PRECONDITION_FAILED"
    assert store.get(stand_request.taskId) is None


def test_status_runtime_exception_has_a_stable_public_code(
    service, stand_request, runtime
):
    """Leaking a runtime status exception would make callers branch on implementation details."""
    runtime.status_error = RuntimeError("status transport failed")

    with pytest.raises(RuntimeException) as error:
        service.create_task(stand_request)

    assert error.value.code == "RUNTIME_EXCEPTION"


def test_start_exception_safely_stops_before_recording_runtime_failure(
    service, stand_request, runtime
):
    """A start failure after ownership begins must still invoke the runtime's safe stop."""
    runtime.start_error = RuntimeError("start failed")
    created = service.create_task(stand_request)

    terminal = wait_for_state(service, created.taskId, "FAILED", runtime.safe_stopped)

    assert terminal.stopReason == "RUNTIME_EXCEPTION"
    assert len(runtime.safe_stop_calls) == 1


def test_runtime_exception_safely_stops_and_records_failed_evidence(
    service, stand_request, runtime
):
    """Letting runtime failures escape would leave the active robot slot and evidence ambiguous."""
    runtime.fail_next_sample(RuntimeError("simulator fault"))
    created = service.create_task(stand_request)

    terminal = wait_for_state(service, created.taskId, "FAILED", runtime.safe_stopped)

    assert terminal.stopReason == "RUNTIME_EXCEPTION"
    assert terminal.evidence is not None
    assert terminal.evidence.stopReason == "RUNTIME_EXCEPTION"
    assert len(runtime.safe_stop_calls) == 1


def test_discrete_max_duration_times_out_and_safely_stops(
    bundle, store, runtime, stand_request, monkeypatch
):
    """Ignoring a discrete deadline would permit a policy to hold robot ownership indefinitely."""
    from mjlab_microduck.rom import service as service_module

    original_action_template = service_module.action_template
    installed_template = original_action_template("STAND")
    assert installed_template.completion is not None
    monkeypatch.setattr(
        service_module,
        "action_template",
        lambda code: (
            replace(
                installed_template,
                completion=installed_template.completion.model_copy(
                    update={"maxDurationMs": 1}
                ),
            )
            if code == "STAND"
            else original_action_template(code)
        ),
    )
    service = SimulatorTaskService(bundle, store, runtime, pollIntervalS=0.001)
    created = service.create_task(stand_request)

    terminal = wait_for_state(
        service, created.taskId, "TIMED_OUT", runtime.safe_stopped
    )

    assert terminal.stopReason == "MAX_DURATION_EXCEEDED"
    assert len(runtime.safe_stop_calls) == 1


def test_runtime_terminal_reason_is_mapped_to_a_stable_public_result_code(
    service, stand_request, runtime
):
    """Passing through runtime text would make public task results unbounded and unparseable."""
    runtime.complete_next(
        state="SUCCEEDED",
        metrics={"upright": True},
        stop_reason="untrusted-runtime-detail/../../../",
    )
    created = service.create_task(stand_request)

    terminal = wait_for_state(
        service, created.taskId, "SUCCEEDED", runtime.safe_stopped
    )
    events = service.events_after(created.taskId, -1)

    assert terminal.stopReason == "TASK_COMPLETE"
    assert terminal.evidence is not None
    assert terminal.evidence.stopReason == "TASK_COMPLETE"
    assert events[-1].payload == {"code": "TASK_COMPLETE"}


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: RuntimeSample(running=True, metrics={"x" * 65: 1}), "metric names"),
        (lambda: RuntimeEvidence(metrics={"note": "x" * 129}), "string values"),
        (
            lambda: RuntimeEvidence(
                metrics={f"metric-{index}": "x" * 64 for index in range(16)}
            ),
            "encoded size",
        ),
    ],
)
def test_runtime_metrics_reject_oversize_trajectory_shaped_payloads(factory, message):
    """Removing metric byte bounds would let a trajectory hide in one evidence record."""
    with pytest.raises(ValueError, match=message):
        factory()


def test_runtime_metrics_allow_bounded_scalar_summary():
    """Overly broad metric rejection would discard normal task-summary evidence."""
    sample = RuntimeSample(
        running=False,
        terminalState="SUCCEEDED",
        metrics={"upright": True, "score": 0.25},
    )

    assert sample.metrics == {"upright": True, "score": 0.25}


def test_terminal_evidence_drops_lower_priority_metrics_to_stay_aggregate_bounded(
    service, stand_request, runtime
):
    """Concatenating valid sample and stop metrics must not persist an oversized evidence blob."""
    sample_metrics = {f"sample-{index:02d}": "s" * 60 for index in range(12)}
    runtime.safe_stop_metrics = {f"stop-{index:02d}": "t" * 60 for index in range(12)}
    RuntimeSample(running=False, terminalState="SUCCEEDED", metrics=sample_metrics)
    RuntimeEvidence(metrics=runtime.safe_stop_metrics)
    runtime.complete_next(state="SUCCEEDED", metrics=sample_metrics)

    created = service.create_task(stand_request)
    terminal = wait_for_state(
        service, created.taskId, "SUCCEEDED", runtime.safe_stopped
    )

    assert terminal.evidence is not None
    assert terminal.evidence.metrics == sample_metrics | {"stop-00": "t" * 60}
    assert len(canonical_json(terminal.evidence.metrics)) <= 1_024


def test_cancel_during_validation_is_idempotent_and_does_not_skip_store_states(
    service, stand_request, runtime
):
    """Transitioning VALIDATING directly to CANCELLED would violate the durable lifecycle."""
    runtime.validation_release.clear()
    created = service.create_task(stand_request)
    assert runtime.validation_started.wait(timeout=1.0)

    first = service.cancel_task(created.taskId)
    second = service.cancel_task(created.taskId)
    assert first.state == second.state == "VALIDATING"
    runtime.validation_release.set()

    terminal = wait_for_state(service, created.taskId, "CANCELLED")
    assert terminal.stopReason == "CANCELLED"
    assert runtime.safe_stopped.wait(timeout=1.0)
    assert runtime.safe_stop_calls == [(None, "CANCELLED")]
    assert [event.eventType for event in service.events_after(created.taskId, -1)] == [
        "TASK_VALIDATING",
        "TASK_CANCEL_REQUESTED",
        "TASK_STARTED",
        "TASK_CANCELLED",
    ]


def test_service_startup_marks_preexisting_inflight_tasks_unknown_without_redispatch(
    bundle, store, runtime, stand_request
):
    """Re-dispatching persisted in-flight work after restart could issue a duplicate robot motion."""
    store.create(stand_request, sha256_prefixed(stand_request))
    store.transition(stand_request.taskId, "VALIDATING", event_type="TASK_VALIDATING")

    restarted = SimulatorTaskService(bundle, store, runtime, pollIntervalS=0.001)

    assert restarted.get_task(stand_request.taskId).state == "UNKNOWN"
    assert not runtime.validation_started.is_set()


def test_cancelled_running_task_safely_stops_once_and_releases_slot(
    service, stand_request, kick_request, runtime
):
    """A repeated cancel must not double-stop a running runtime or keep the robot busy."""
    created = service.create_task(stand_request)
    assert runtime.started.wait(timeout=1.0)

    service.cancel_task(created.taskId)
    terminal = wait_for_state(
        service, created.taskId, "CANCELLED", runtime.safe_stopped
    )
    service.cancel_task(created.taskId)

    assert terminal.stopReason == "CANCELLED"
    assert len(runtime.safe_stop_calls) == 1
    runtime.complete_next(state="SUCCEEDED", metrics={})
    next_task = service.create_task(kick_request)
    assert next_task.taskId == kick_request.taskId


@pytest.fixture
def bundle() -> PolicyBundle:
    observation = ObservationContract(
        identifier="MICRODUCK_OBS_61_V1",
        dimension=61,
        fields=list(OBSERVATION_FIELDS),
        units={},
        normalization="BAKED_IN_ONNX",
    )
    action_contract = ActionContract(
        identifier="MICRODUCK_ACTION_14_V1",
        dimension=14,
        joints=list(CONTROLLED_SERVO_JOINTS),
        units="rad",
        scaling={},
        clipping={},
    )
    return PolicyBundle(
        schema="MICRODUCK_POLICY_BUNDLE_V1",
        bundleId="microduck-test",
        bundleVersion="1.0.0",
        bundleDigest="sha256:" + "a" * 64,
        createdAt=datetime(2026, 8, 29, tzinfo=UTC),
        sourceRepository="microduck-rl",
        sourceCommit="c" * 40,
        robotModel="MICRODUCK",
        observationContract=observation,
        actionContract=action_contract,
        model=ModelArtifact(path="models/robot.xml", digest="sha256:" + "c" * 64),
        policies=[
            PolicyArtifact(
                policyRef="stand",
                path="policies/stand.onnx",
                digest="sha256:" + "b" * 64,
                taskId="Mjlab-SitStand-Flat-MicroDuck",
            )
        ],
        actions=[
            code_owned_action_definition(
                code,
                availability="AVAILABLE" if code == "STAND" else "UNAVAILABLE",
                policy_ref="stand" if code == "STAND" else None,
                unavailable_reason=(
                    None if code == "STAND" else "POLICY_ARTIFACT_MISSING"
                ),
            )
            for code in CODE_OWNED_ACTION_CODES
        ],
        qualification={},
        license={},
    )


@pytest.fixture
def store(tmp_path) -> SqliteTaskStore:
    return SqliteTaskStore(tmp_path / "simulator.sqlite3")


@pytest.fixture
def runtime() -> FakeMicroduckRuntime:
    return FakeMicroduckRuntime()


@pytest.fixture
def service(bundle, store, runtime) -> SimulatorTaskService:
    return SimulatorTaskService(bundle, store, runtime, pollIntervalS=0.001)


@pytest.fixture
def stand_request() -> TaskCreateRequest:
    return request_for("0" * 32, "STAND")


@pytest.fixture
def kick_request() -> TaskCreateRequest:
    return request_for("1" * 32, "STAND")


def request_for(task_id: str, action_code: str) -> TaskCreateRequest:
    return TaskCreateRequest(
        schema="MICRODUCK_SIM_TASK_V1",
        taskId=task_id,
        actionCode=action_code,
        bundleVersion="1.0.0",
        bundleDigest="sha256:" + "a" * 64,
        parameters={},
        scenario={"terrain": "flat", "seed": 1},
        requestedBy="test-execution",
    )


def wait_for_state(
    service: SimulatorTaskService, task_id: str, state: str, signal: Event | None = None
):
    """Wait on a runtime coordination signal, then observe the durable terminal snapshot."""
    if signal is not None:
        assert signal.wait(timeout=1.0)
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        snapshot = service.get_task(task_id)
        if snapshot.state == state:
            return snapshot
        Event().wait(0.001)
    pytest.fail(f"task {task_id} did not reach {state}")

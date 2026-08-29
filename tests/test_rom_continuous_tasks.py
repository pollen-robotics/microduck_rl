from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from math import nan

import pytest

from mjlab_microduck.rom.contracts import (
    CONTROLLED_SERVO_JOINTS,
    OBSERVATION_FIELDS,
    ActionContract,
    ActionDefinition,
    ModelArtifact,
    ObservationContract,
    PolicyArtifact,
    PolicyBundle,
    TaskCommandRequest,
    TaskCreateRequest,
)
from mjlab_microduck.rom.service import (
    CommandSequenceConflict,
    InvalidParameters,
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


def command(*, sequence: int, vx: float = 0.0, lease_ms: int = 500) -> TaskCommandRequest:
    return TaskCommandRequest(
        commandSequence=sequence,
        parameters={"vxMps": vx, "vyMps": 0.0, "yawRateRadps": 0.0},
        leaseMs=lease_ms,
    )


def test_expired_lease_zeros_velocity_stops_and_times_out(service, walk_request, clock, runtime):
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
        command(sequence=1, vx=nan),
        TaskCommandRequest(commandSequence=1, parameters={"vxMps": 0.0, "vyMps": 0.0}, leaseMs=100),
    ],
)
def test_command_rejects_out_of_manifest_parameter_or_lease_bounds(service, walk_request, invalid):
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
    assert task[1] == '{"commandSequence":3,"leaseMs":500,"parameters":{"vxMps":0.2,"vyMps":0.0,"yawRateRadps":0.0}}'
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
        parameters={},
        scenario={"terrain": "flat", "seed": 1},
        requestedBy="test-continuous",
    )


@pytest.fixture
def bundle() -> PolicyBundle:
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
            )
        ],
        actions=[
            ActionDefinition(
                actionCode="WALK_VELOCITY",
                executionMode="CONTINUOUS_LEASE",
                availability="AVAILABLE",
                policyRef="walk",
                parameterSchema={
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "vxMps": {"type": "number", "minimum": -0.4, "maximum": 0.4},
                        "vyMps": {"type": "number", "minimum": -0.3, "maximum": 0.3},
                        "yawRateRadps": {"type": "number", "minimum": -1.0, "maximum": 1.0},
                    },
                    "required": ["vxMps", "vyMps", "yawRateRadps"],
                },
                lease={
                    "minLeaseMs": 100,
                    "defaultLeaseMs": 500,
                    "maxLeaseMs": 5_000,
                    "commandCadenceMs": 50,
                    "safeStopBehavior": "ZERO_TWIST",
                },
                preconditions={"allowedTerrains": ["flat"]},
            )
        ],
        qualification={},
        license={},
    )

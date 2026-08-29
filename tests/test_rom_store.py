from __future__ import annotations

import sqlite3

import pytest

from mjlab_microduck.rom.contracts import (
    TaskCreateRequest,
    TaskEvidence,
    canonical_json,
)
from mjlab_microduck.rom.store import (
    IllegalTaskTransition,
    SqliteTaskStore,
    TaskIdConflict,
)

REQUEST_HASH = "sha256:" + "1" * 64


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "simulator.sqlite3"


@pytest.fixture
def task_request() -> TaskCreateRequest:
    return TaskCreateRequest.model_validate(
        {
            "schema": "MICRODUCK_SIM_TASK_V1",
            "taskId": "0" * 32,
            "actionCode": "STAND",
            "bundleVersion": "1.0.0",
            "bundleDigest": "sha256:" + "a" * 64,
            "parameters": {"durationS": 3, "label": "caf\u00e9"},
            "scenario": {"terrain": "flat", "seed": 1},
            "requestedBy": "execution-1",
        }
    )


@pytest.fixture
def store(db_path):
    return SqliteTaskStore(db_path)


def test_create_persists_accepted_snapshot_and_task_one_canonical_request(
    store, task_request, db_path
):
    """Replacing Task 1 canonical bytes would make persisted identity differ from the API hash."""
    snapshot, created = store.create(task_request, REQUEST_HASH)

    assert created is True
    assert snapshot.taskId == task_request.taskId
    assert snapshot.state == "ACCEPTED"
    assert snapshot.actionCode == "STAND"
    assert snapshot.requestedAt == snapshot.updatedAt
    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            "SELECT request_canonical_json, request_hash FROM task WHERE task_id = ?",
            (task_request.taskId,),
        ).fetchone()
    assert row == (canonical_json(task_request).decode(), REQUEST_HASH)


def test_same_id_same_hash_returns_existing(store, task_request):
    """Dropping hash comparison would turn retried creates into duplicate simulator tasks."""
    first, created = store.create(task_request, REQUEST_HASH)
    again, created_again = store.create(task_request, REQUEST_HASH)

    assert created is True
    assert created_again is False
    assert again.taskId == first.taskId
    assert again.requestedAt == first.requestedAt


def test_same_id_different_hash_raises_conflict(store, task_request):
    """Accepting a changed request under one task ID would execute ambiguous user intent."""
    store.create(task_request, REQUEST_HASH)

    with pytest.raises(TaskIdConflict):
        store.create(task_request, "sha256:" + "2" * 64)


def test_two_store_instances_share_idempotent_identity(db_path, task_request):
    """Separate connections must not race into duplicate task records for the same ID."""
    first_store = SqliteTaskStore(db_path)
    second_store = SqliteTaskStore(db_path)

    _, first_created = first_store.create(task_request, REQUEST_HASH)
    existing, second_created = second_store.create(task_request, REQUEST_HASH)

    assert first_created is True
    assert second_created is False
    assert existing.taskId == task_request.taskId


def test_transition_updates_snapshot_and_appends_ordered_event_atomically(
    store, task_request
):
    """Splitting state writes from event appends would leave a task history that contradicts its state."""
    store.create(task_request, REQUEST_HASH)
    validating = store.transition(
        task_request.taskId,
        "VALIDATING",
        event_type="TASK_VALIDATING",
        payload={"check": "bundle"},
    )
    running = store.transition(
        task_request.taskId, "RUNNING", event_type="TASK_STARTED"
    )

    assert validating.state == "VALIDATING"
    assert running.state == "RUNNING"
    assert [
        (event.sequence, event.eventType, event.payload)
        for event in store.events_after(task_request.taskId, -1)
    ] == [
        (0, "TASK_VALIDATING", {"check": "bundle"}),
        (1, "TASK_STARTED", {}),
    ]


def test_append_event_orders_events_across_store_instances(db_path, task_request):
    """Per-process event counters would collide when two store instances use one database file."""
    first_store = SqliteTaskStore(db_path)
    second_store = SqliteTaskStore(db_path)
    first_store.create(task_request, REQUEST_HASH)

    first_store.append_event(task_request.taskId, "TASK_QUEUED", {"queue": 1})
    second_store.append_event(
        task_request.taskId, "TASK_OBSERVED", {"observer": "second"}
    )

    assert [
        (event.sequence, event.eventType)
        for event in first_store.events_after(task_request.taskId, -1)
    ] == [
        (0, "TASK_QUEUED"),
        (1, "TASK_OBSERVED"),
    ]


def test_event_pages_are_bounded_without_duplicates_or_sequence_zero_gaps(
    store, task_request
):
    """An unbounded query or zero-only cursor would lose the first event or amplify storage."""
    store.create(task_request, REQUEST_HASH)
    for sequence in range(205):
        store.append_event(task_request.taskId, "TASK_OBSERVED", {"index": sequence})

    first = store.events_after(task_request.taskId, -1, page_size=100)
    second = store.events_after(task_request.taskId, first[-1].sequence, page_size=100)
    third = store.events_after(task_request.taskId, second[-1].sequence, page_size=100)

    assert [event.sequence for event in first] == list(range(100))
    assert [event.sequence for event in second] == list(range(100, 200))
    assert [event.sequence for event in third] == list(range(200, 205))


def test_event_cursor_is_bounded_to_sqlite_signed_integer_range(store, task_request):
    """SQLite cursor binding must never receive a Python integer beyond int64."""
    store.create(task_request, REQUEST_HASH)
    store.append_event(task_request.taskId, "TASK_OBSERVED", {})

    assert store.events_after(task_request.taskId, 2**63 - 1) == []
    with pytest.raises(ValueError, match="signed 64-bit"):
        store.events_after(task_request.taskId, 2**63)


def test_transition_allows_only_linear_lifecycle_and_terminal_is_immutable(
    store, task_request
):
    """Permitting skipped or terminal transitions would let execution bypass validation or rewrite outcomes."""
    store.create(task_request, REQUEST_HASH)

    with pytest.raises(IllegalTaskTransition):
        store.transition(task_request.taskId, "RUNNING", event_type="TASK_STARTED")

    store.transition(task_request.taskId, "VALIDATING", event_type="TASK_VALIDATING")
    store.transition(task_request.taskId, "RUNNING", event_type="TASK_STARTED")
    completed = store.transition(
        task_request.taskId, "SUCCEEDED", event_type="TASK_SUCCEEDED"
    )

    assert completed.state == "SUCCEEDED"
    with pytest.raises(IllegalTaskTransition):
        store.transition(task_request.taskId, "FAILED", event_type="TASK_FAILED")


def test_terminal_transition_persists_evidence_and_stop_reason(store, task_request):
    """Omitting terminal evidence would sever the completed task from its policy provenance."""
    store.create(task_request, REQUEST_HASH)
    store.transition(task_request.taskId, "VALIDATING", event_type="TASK_VALIDATING")
    store.transition(task_request.taskId, "RUNNING", event_type="TASK_STARTED")
    evidence = TaskEvidence(
        bundleDigest="sha256:" + "a" * 64,
        policyDigest="sha256:" + "b" * 64,
        metrics={"score": 0.9},
        stopReason="COMPLETED",
    )

    completed = store.transition(
        task_request.taskId,
        "SUCCEEDED",
        event_type="TASK_SUCCEEDED",
        evidence=evidence,
        stop_reason="COMPLETED",
    )

    assert completed.evidence == evidence
    assert completed.stopReason == "COMPLETED"


def test_restart_marks_running_unknown(db_path, task_request):
    """A restarted simulator must not report a pre-crash running task as still executing."""
    store = SqliteTaskStore(db_path)
    store.create(task_request, REQUEST_HASH)
    store.transition(task_request.taskId, "VALIDATING", event_type="TASK_VALIDATING")
    store.transition(task_request.taskId, "RUNNING", event_type="TASK_STARTED")

    assert SqliteTaskStore(db_path).mark_interrupted_unknown() == 1
    assert SqliteTaskStore(db_path).get(task_request.taskId).state == "UNKNOWN"
    assert [event.eventType for event in store.events_after(task_request.taskId, -1)][
        -1
    ] == "TASK_INTERRUPTED"


def test_initial_schema_reserves_task_five_command_and_deadline_columns(store, db_path):
    """Removing the reserved fields would force an unsafe in-process schema migration for Task 5."""
    del store
    with sqlite3.connect(db_path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(task)")}
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name LIKE 'task%'"
            )
        }

    assert {"task", "task_event", "task_evidence"} <= tables
    assert {
        "command_sequence",
        "command_canonical_json",
        "command_hash",
        "lease_expires_at",
        "deadline_at",
    } <= columns

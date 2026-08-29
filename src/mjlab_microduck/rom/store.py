"""Durable SQLite persistence for MicroDuck simulator tasks."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .contracts import (
    TaskCommandRequest,
    TaskCreateRequest,
    TaskEvent,
    TaskEvidence,
    TaskSnapshot,
    canonical_json,
)

_TERMINAL_STATES = frozenset({"SUCCEEDED", "FAILED", "CANCELLED", "TIMED_OUT"})
_ALLOWED_TRANSITIONS = {
    "ACCEPTED": frozenset({"VALIDATING"}),
    "VALIDATING": frozenset({"RUNNING"}),
    "RUNNING": _TERMINAL_STATES,
}
_INTERRUPTIBLE_STATES = frozenset({"ACCEPTED", "VALIDATING", "RUNNING"})


class TaskIdConflict(ValueError):
    """A task ID is already associated with different canonical request content."""


class IllegalTaskTransition(ValueError):
    """A requested lifecycle change does not follow the simulator task state machine."""


class CommandSequenceConflict(ValueError):
    """One command sequence is associated with different canonical command content."""


class StaleCommand(ValueError):
    """A command sequence is lower than the last durably accepted sequence."""


class SqliteTaskStore:
    """SQLite-backed task store using one connection per operation."""

    def __init__(self, database_path: str | Path) -> None:
        self._database_path = Path(database_path)
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    def create(
        self, request: TaskCreateRequest, request_hash: str
    ) -> tuple[TaskSnapshot, bool]:
        """Persist an accepted task or return the existing task for an idempotent retry."""
        request_content = canonical_json(request).decode("utf-8")
        timestamp = _timestamp()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT request_hash FROM task WHERE task_id = ?", (request.taskId,)
            ).fetchone()
            if existing is not None:
                if existing["request_hash"] != request_hash:
                    connection.rollback()
                    raise TaskIdConflict(f"task ID already exists: {request.taskId}")
                snapshot = self._get_in_connection(connection, request.taskId)
                connection.commit()
                return snapshot, False
            connection.execute(
                """
                INSERT INTO task (
                    task_id, request_canonical_json, request_hash, state, action_code,
                    bundle_version, bundle_digest, requested_at, updated_at
                ) VALUES (?, ?, ?, 'ACCEPTED', ?, ?, ?, ?, ?)
                """,
                (
                    request.taskId,
                    request_content,
                    request_hash,
                    request.actionCode,
                    request.bundleVersion,
                    request.bundleDigest,
                    timestamp,
                    timestamp,
                ),
            )
            snapshot = self._get_in_connection(connection, request.taskId)
            connection.commit()
            return snapshot, True

    def get(self, task_id: str) -> TaskSnapshot | None:
        """Return the durable task snapshot when it exists."""
        with self._connect() as connection:
            return self._get_in_connection(connection, task_id, required=False)

    def transition(
        self,
        task_id: str,
        state: str,
        *,
        event_type: str,
        payload: Mapping[str, Any] | None = None,
        evidence: TaskEvidence | None = None,
        stop_reason: str | None = None,
    ) -> TaskSnapshot:
        """Atomically change lifecycle state and append its ordered audit event."""
        timestamp = _timestamp()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            task = self._get_in_connection(connection, task_id)
            if state not in _ALLOWED_TRANSITIONS.get(task.state, frozenset()):
                connection.rollback()
                raise IllegalTaskTransition(
                    f"cannot transition {task.state} to {state}"
                )
            connection.execute(
                "UPDATE task SET state = ?, updated_at = ?, stop_reason = ? WHERE task_id = ?",
                (state, timestamp, stop_reason, task_id),
            )
            if evidence is not None:
                connection.execute(
                    """
                    INSERT INTO task_evidence (task_id, evidence_json, created_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(task_id) DO UPDATE SET
                        evidence_json = excluded.evidence_json,
                        created_at = excluded.created_at
                    """,
                    (task_id, canonical_json(evidence).decode("utf-8"), timestamp),
                )
            self._append_event_in_connection(
                connection, task_id, event_type, payload or {}, timestamp
            )
            snapshot = self._get_in_connection(connection, task_id)
            connection.commit()
            return snapshot

    def start_continuous(self, task_id: str, deadline_at: float) -> TaskSnapshot:
        """Atomically enter RUNNING with the target-owned initial lease deadline."""
        timestamp = _timestamp()
        deadline = _monotonic_deadline(deadline_at)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            task = self._get_in_connection(connection, task_id)
            if task.state != "VALIDATING":
                connection.rollback()
                raise IllegalTaskTransition(
                    f"cannot transition {task.state} to RUNNING"
                )
            connection.execute(
                """
                UPDATE task
                SET state = 'RUNNING', deadline_at = ?, lease_expires_at = ?, updated_at = ?
                WHERE task_id = ?
                """,
                (deadline, deadline, timestamp, task_id),
            )
            self._append_event_in_connection(
                connection, task_id, "TASK_STARTED", {"initialLease": True}, timestamp
            )
            snapshot = self._get_in_connection(connection, task_id)
            connection.commit()
            return snapshot

    def append_event(
        self, task_id: str, event_type: str, payload: Mapping[str, Any] | None = None
    ) -> TaskEvent:
        """Append an ordered event using a database-derived sequence number."""
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._get_in_connection(connection, task_id)
            event = self._append_event_in_connection(
                connection, task_id, event_type, payload or {}, _timestamp()
            )
            connection.commit()
            return event

    def record_command(
        self,
        task_id: str,
        command: TaskCommandRequest,
        command_hash: str,
        deadline_at: float,
    ) -> tuple[TaskSnapshot, bool]:
        """Atomically record a newer continuous command and its lease deadline."""
        command_content = canonical_json(command).decode("utf-8")
        deadline = _monotonic_deadline(deadline_at)
        timestamp = _timestamp()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            task = self._get_in_connection(connection, task_id)
            row = connection.execute(
                "SELECT command_sequence, command_canonical_json, command_hash FROM task WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            sequence = row["command_sequence"]
            if sequence is not None:
                if command.commandSequence < sequence:
                    connection.rollback()
                    raise StaleCommand(
                        f"command sequence is stale: {command.commandSequence}"
                    )
                if command.commandSequence == sequence:
                    if (
                        row["command_canonical_json"] == command_content
                        and row["command_hash"] == command_hash
                    ):
                        connection.commit()
                        return task, False
                    connection.rollback()
                    raise CommandSequenceConflict(
                        f"command sequence already exists: {command.commandSequence}"
                    )
            if task.state != "RUNNING":
                connection.rollback()
                raise IllegalTaskTransition(f"cannot command task in {task.state}")
            connection.execute(
                """
                UPDATE task
                SET command_sequence = ?, command_canonical_json = ?, command_hash = ?,
                    lease_expires_at = ?, deadline_at = ?, updated_at = ?
                WHERE task_id = ?
                """,
                (
                    command.commandSequence,
                    command_content,
                    command_hash,
                    deadline,
                    deadline,
                    timestamp,
                    task_id,
                ),
            )
            self._append_event_in_connection(
                connection,
                task_id,
                "TASK_COMMAND_ACCEPTED",
                {"commandSequence": command.commandSequence},
                timestamp,
            )
            snapshot = self._get_in_connection(connection, task_id)
            connection.commit()
            return snapshot, True

    def events_after(
        self, task_id: str, sequence: int, *, page_size: int = 100
    ) -> list[TaskEvent]:
        """Return task events strictly after ``sequence`` in durable sequence order."""
        if (
            not isinstance(sequence, int)
            or isinstance(sequence, bool)
            or not -1 <= sequence <= 2**63 - 1
        ):
            raise ValueError("sequence must be a signed 64-bit cursor from -1")
        if (
            not isinstance(page_size, int)
            or isinstance(page_size, bool)
            or not 1 <= page_size <= 100
        ):
            raise ValueError("page_size must be an integer between 1 and 100")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT sequence, event_type, payload_json, created_at
                FROM task_event
                WHERE task_id = ? AND sequence > ?
                ORDER BY sequence
                LIMIT ?
                """,
                (task_id, sequence, page_size),
            ).fetchall()
        return [
            TaskEvent(
                sequence=row["sequence"],
                eventType=row["event_type"],
                payload=json.loads(row["payload_json"]),
                createdAt=row["created_at"],
            )
            for row in rows
        ]

    def mark_interrupted_unknown(self) -> int:
        """Reconcile persisted in-flight tasks after simulator startup."""
        timestamp = _timestamp()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                "SELECT task_id, state FROM task WHERE state IN ('ACCEPTED', 'VALIDATING', 'RUNNING')"
            ).fetchall()
            for row in rows:
                connection.execute(
                    "UPDATE task SET state = 'UNKNOWN', updated_at = ? WHERE task_id = ?",
                    (timestamp, row["task_id"]),
                )
                self._append_event_in_connection(
                    connection,
                    row["task_id"],
                    "TASK_INTERRUPTED",
                    {"previousState": row["state"]},
                    timestamp,
                )
            connection.commit()
            return len(rows)

    def _initialize_schema(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS task (
                    task_id TEXT PRIMARY KEY,
                    request_canonical_json TEXT NOT NULL,
                    request_hash TEXT NOT NULL,
                    state TEXT NOT NULL,
                    action_code TEXT NOT NULL,
                    bundle_version TEXT NOT NULL,
                    bundle_digest TEXT NOT NULL,
                    requested_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    stop_reason TEXT,
                    command_sequence INTEGER,
                    command_canonical_json TEXT,
                    command_hash TEXT,
                    lease_expires_at TEXT,
                    deadline_at TEXT
                );

                CREATE TABLE IF NOT EXISTS task_event (
                    task_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (task_id, sequence),
                    FOREIGN KEY (task_id) REFERENCES task(task_id)
                );

                CREATE TABLE IF NOT EXISTS task_evidence (
                    task_id TEXT PRIMARY KEY,
                    evidence_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (task_id) REFERENCES task(task_id)
                );
                """
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(
            self._database_path, isolation_level=None, timeout=5.0
        )
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA busy_timeout = 5000")
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA journal_mode = WAL")
            yield connection
        finally:
            connection.close()

    def _get_in_connection(
        self, connection: sqlite3.Connection, task_id: str, *, required: bool = True
    ) -> TaskSnapshot | None:
        row = connection.execute(
            """
            SELECT task.*, task_evidence.evidence_json
            FROM task
            LEFT JOIN task_evidence ON task_evidence.task_id = task.task_id
            WHERE task.task_id = ?
            """,
            (task_id,),
        ).fetchone()
        if row is None:
            if required:
                raise KeyError(f"task not found: {task_id}")
            return None
        evidence = (
            TaskEvidence.model_validate_json(row["evidence_json"])
            if row["evidence_json"] is not None
            else None
        )
        return TaskSnapshot(
            taskId=row["task_id"],
            state=row["state"],
            actionCode=row["action_code"],
            bundleVersion=row["bundle_version"],
            bundleDigest=row["bundle_digest"],
            requestedAt=row["requested_at"],
            updatedAt=row["updated_at"],
            evidence=evidence,
            stopReason=row["stop_reason"],
        )

    def _append_event_in_connection(
        self,
        connection: sqlite3.Connection,
        task_id: str,
        event_type: str,
        payload: Mapping[str, Any],
        timestamp: str,
    ) -> TaskEvent:
        sequence = connection.execute(
            "SELECT COALESCE(MAX(sequence), -1) + 1 AS next_sequence FROM task_event WHERE task_id = ?",
            (task_id,),
        ).fetchone()["next_sequence"]
        payload_json = canonical_json(payload).decode("utf-8")
        connection.execute(
            """
            INSERT INTO task_event (task_id, sequence, event_type, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (task_id, sequence, event_type, payload_json, timestamp),
        )
        return TaskEvent(
            sequence=sequence,
            eventType=event_type,
            payload=dict(payload),
            createdAt=timestamp,
        )


def _timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _monotonic_deadline(value: float) -> str:
    """Store an injected monotonic-clock instant without converting it to wall time."""
    return format(value, ".9f")

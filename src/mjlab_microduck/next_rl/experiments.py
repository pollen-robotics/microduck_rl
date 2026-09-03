"""Immutable experiment inputs and durable operational run state."""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import atomic_write_json, canonical_json
from .capabilities import PlanDecision
from .schema import ArtifactRef, ExperimentManifest, SkillSpec


class DuplicateExperimentError(RuntimeError):
    """An equivalent experiment is already reserved or has completed."""


@dataclass(frozen=True)
class ReservationDecision:
    """The explicit action an operator may take for an experiment fingerprint."""

    fingerprint: str
    action: str


_LEARNING_FIELDS = (
    "skill_id",
    "spec_version",
    "task_id",
    "contract",
    "environment_config",
    "agent_config",
    "code_digest",
    "dirty_patch_digest",
    "seed",
    "parent_policy_digest",
    "runner_id",
)
_OPERATING_KEYS = {
    "created_at",
    "hostname",
    "host",
    "pid",
    "output_dir",
    "output_path",
    "log_dir",
    "status",
    "timestamp",
}
_CREDENTIAL_KEYS = frozenset(
    (
        "access_key",
        "access_token",
        "api_key",
        "api_token",
        "auth",
        "authorization",
        "bearer_token",
        "client_secret",
        "credential",
        "credentials",
        "password",
        "private_key",
        "secret",
        "token",
    )
)
_ACTIVE_STATUSES = frozenset(("pending", "running", "succeeded"))
_TRANSITIONS = {
    "planned": frozenset(("pending",)),
    "pending": frozenset(("running",)),
    "running": frozenset(("succeeded", "failed", "interrupted")),
    "failed": frozenset(("pending",)),
    "interrupted": frozenset(("pending",)),
}
_LOCK_TIMEOUT_SECONDS = 10
_STALE_LOCK_SECONDS = 300


def _normalized_key(key: str) -> str:
    return key.strip().lower().replace("-", "_")


def _is_redacted_path(path: tuple[str, ...]) -> bool:
    """Return whether the value at an explicit config path is non-learning data."""
    key = _normalized_key(path[-1])
    return key in _OPERATING_KEYS or key in _CREDENTIAL_KEYS


def _normalize(value: Any, path: tuple[str, ...] = ()) -> Any:
    """Return JSON-compatible learning data without operational or credential paths."""
    if isinstance(value, Mapping):
        return {
            key: _normalize(item, (*path, key))
            for key, item in value.items()
            if isinstance(key, str) and not _is_redacted_path((*path, key))
        }
    if isinstance(value, tuple):
        return [_normalize(item, path) for item in value]
    if isinstance(value, list):
        return [_normalize(item, path) for item in value]
    return value


def _manifest_data(manifest: ExperimentManifest | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(manifest, ExperimentManifest):
        return manifest.as_dict()
    if not isinstance(manifest, Mapping):
        raise TypeError("manifest must be an ExperimentManifest or mapping")
    return manifest


def learning_inputs(manifest: ExperimentManifest | Mapping[str, Any]) -> dict[str, Any]:
    """Project a manifest onto only inputs that can change learned policy bytes."""
    data = _manifest_data(manifest)
    return {field: _normalize(data.get(field)) for field in _LEARNING_FIELDS}


def experiment_fingerprint(manifest: ExperimentManifest | Mapping[str, Any]) -> str:
    """Return the SHA-256 identity of an experiment's immutable learning inputs."""
    encoded = canonical_json(learning_inputs(manifest)).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_experiment_manifest(
    spec: SkillSpec,
    decision: PlanDecision,
    *,
    task_id: str,
    code_digest: str,
    seed: int,
    runner_id: str,
    environment_config: Mapping[str, Any],
    agent_config: Mapping[str, Any],
    parent_policy: ArtifactRef | None = None,
    dirty_patch_digest: str | None = None,
    created_at: str | None = None,
    output_dir: str | None = None,
) -> ExperimentManifest:
    """Build a validated manifest while recording the planner's reproducible choice."""
    plan: dict[str, Any] = {"disposition": decision.disposition.value, "reason": decision.reason}
    if decision.capability is not None:
        plan["capability_id"] = decision.capability.id
    if decision.improve_reason is not None:
        plan["improve_reason"] = decision.improve_reason
    raw: dict[str, Any] = {
        "skill_id": spec.id,
        "spec_version": spec.version,
        "task_id": task_id,
        "contract": spec.contract.as_dict(),
        "code_digest": code_digest,
        "seed": seed,
        "runner_id": runner_id,
        "status": "planned",
        "environment_config": dict(environment_config),
        "agent_config": dict(agent_config),
        "metadata": {"plan": plan},
    }
    if parent_policy is not None:
        raw["parent_policy_digest"] = parent_policy.sha256
    if dirty_patch_digest is not None:
        raw["dirty_patch_digest"] = dirty_patch_digest
    if created_at is not None:
        raw["created_at"] = created_at
    if output_dir is not None:
        raw["output_dir"] = output_dir
    return ExperimentManifest.from_dict(raw)


def _exclusive_write_json(path: Path, value: Any) -> None:
    """Create a JSON record exactly once, including a durable file payload."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(canonical_json(value).encode("utf-8"))
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


class ExperimentStore:
    """Filesystem store that separates immutable learning inputs from run state."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _directory(self, fingerprint: str) -> Path:
        if len(fingerprint) != 64 or any(character not in "0123456789abcdef" for character in fingerprint):
            raise ValueError("fingerprint must be a lowercase SHA-256 digest")
        return self.root / fingerprint

    def _manifest_path(self, fingerprint: str) -> Path:
        return self._directory(fingerprint) / "manifest.json"

    def _status_path(self, fingerprint: str) -> Path:
        return self._directory(fingerprint) / "status.json"

    def _reservation_path(self, fingerprint: str) -> Path:
        return self._directory(fingerprint) / "reservation.json"

    def _lock_path(self, fingerprint: str) -> Path:
        return self._directory(fingerprint) / ".lock"

    @contextmanager
    def _lock(self, fingerprint: str):
        """Acquire an ownership-checked lockfile for one experiment fingerprint."""
        path = self._lock_path(fingerprint)
        owner = uuid.uuid4().hex
        deadline = time.monotonic() + _LOCK_TIMEOUT_SECONDS
        while True:
            try:
                _exclusive_write_json(path, {"owner": owner, "pid": os.getpid()})
                break
            except FileExistsError:
                try:
                    stale = time.time() - path.stat().st_mtime > _STALE_LOCK_SECONDS
                except FileNotFoundError:
                    continue
                if stale:
                    try:
                        path.unlink()
                    except FileNotFoundError:
                        pass
                    continue
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"timed out waiting for experiment lock {fingerprint}")
                time.sleep(0.01)
        try:
            yield
        finally:
            try:
                lock = self._read_json(path)
            except (FileNotFoundError, TypeError, json.JSONDecodeError):
                lock = {}
            if lock.get("owner") == owner:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass

    def _read_json(self, path: Path) -> dict[str, Any]:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise TypeError(f"{path} must contain a JSON object")
        return value

    def create(
        self,
        manifest: ExperimentManifest | Mapping[str, Any],
        *,
        fingerprint: str | None = None,
    ) -> str:
        """Durably record immutable learning inputs and initial operational status."""
        parsed = manifest if isinstance(manifest, ExperimentManifest) else ExperimentManifest.from_dict(manifest)
        calculated = experiment_fingerprint(parsed)
        if fingerprint is not None and fingerprint != calculated:
            raise ValueError("fingerprint does not match immutable learning inputs")
        target = fingerprint or calculated
        immutable = {
            "fingerprint": target,
            "learning_inputs": learning_inputs(parsed),
            "provenance": _normalize(parsed.metadata),
        }
        with self._lock(target):
            manifest_path = self._manifest_path(target)
            try:
                _exclusive_write_json(manifest_path, immutable)
            except FileExistsError:
                if self._read_json(manifest_path) != immutable:
                    raise ValueError("immutable learning inputs cannot be changed") from None

            status_path = self._status_path(target)
            if not status_path.exists():
                atomic_write_json(status_path, {"status": parsed.status, "history": [{"status": parsed.status}]})
        return target

    def reserve(self, fingerprint: str, *, owner: str | None = None) -> ReservationDecision:
        """Claim a planned run exactly once or return the safe next action."""
        reservation_owner = owner or uuid.uuid4().hex
        if not isinstance(reservation_owner, str) or not reservation_owner.strip():
            raise ValueError("reservation owner must be a non-empty string")
        with self._lock(fingerprint):
            manifest_path = self._manifest_path(fingerprint)
            if not manifest_path.exists():
                raise FileNotFoundError(f"no experiment exists for fingerprint {fingerprint}")
            current = self._read_json(self._status_path(fingerprint))["status"]
            if current == "pending":
                try:
                    reservation = self._read_json(self._reservation_path(fingerprint))
                except FileNotFoundError:
                    raise DuplicateExperimentError(
                        f"experiment {fingerprint} is already pending"
                    ) from None
                if reservation.get("owner") == reservation_owner:
                    return ReservationDecision(fingerprint, "confirmed")
                raise DuplicateExperimentError(
                    f"experiment {fingerprint} is already reserved"
                )
            if current in _ACTIVE_STATUSES:
                raise DuplicateExperimentError(f"experiment {fingerprint} is already {current}")
            actions = {"planned": "reserved", "failed": "retry", "interrupted": "resume"}
            action = actions.get(current)
            if action is None:
                raise ValueError(f"unknown experiment status {current!r}")
            try:
                _exclusive_write_json(
                    self._reservation_path(fingerprint),
                    {
                        "fingerprint": fingerprint,
                        "owner": reservation_owner,
                        "previous_status": current,
                    },
                )
            except FileExistsError:
                raise DuplicateExperimentError(f"experiment {fingerprint} is already reserved") from None
            self._update_status_locked(fingerprint, "pending")
            return ReservationDecision(fingerprint, action)

    def confirm(self, fingerprint: str, *, owner: str) -> None:
        """Finish an accepted start claim without changing its lifecycle state."""
        if not isinstance(owner, str) or not owner.strip():
            raise ValueError("reservation owner must be a non-empty string")
        with self._lock(fingerprint):
            reservation_path = self._reservation_path(fingerprint)
            reservation = self._read_json(reservation_path)
            if reservation.get("owner") != owner:
                raise DuplicateExperimentError("experiment reservation belongs to another owner")
            reservation_path.unlink()

    def release(self, fingerprint: str, *, owner: str) -> None:
        """Return an unlaunched reservation to its prior state for its exact owner."""
        if not isinstance(owner, str) or not owner.strip():
            raise ValueError("reservation owner must be a non-empty string")
        with self._lock(fingerprint):
            reservation_path = self._reservation_path(fingerprint)
            reservation = self._read_json(reservation_path)
            if reservation.get("owner") != owner:
                raise DuplicateExperimentError("experiment reservation belongs to another owner")
            status_path = self._status_path(fingerprint)
            current_record = self._read_json(status_path)
            if current_record.get("status") != "pending":
                raise DuplicateExperimentError("experiment reservation is no longer pending")
            previous_status = reservation.get("previous_status", "planned")
            if previous_status not in {"planned", "failed", "interrupted"}:
                raise ValueError("experiment reservation previous status is invalid")
            history = current_record.get("history")
            if not isinstance(history, list):
                raise TypeError("experiment status history must be a list")
            reservation_path.unlink()
            atomic_write_json(
                status_path,
                {
                    "status": previous_status,
                    "history": [
                        *history,
                        {"status": previous_status, "reason": "start_failed"},
                    ],
                },
            )

    def status(self, fingerprint: str) -> dict[str, Any]:
        """Return a copy of the durable local lifecycle record."""
        with self._lock(fingerprint):
            return dict(self._read_json(self._status_path(fingerprint)))

    def update_status(self, fingerprint: str, status: str) -> None:
        """Atomically advance operational state without ever rewriting the manifest."""
        with self._lock(fingerprint):
            self._update_status_locked(fingerprint, status)

    def _update_status_locked(self, fingerprint: str, status: str) -> None:
        """Advance status while the caller holds this fingerprint's lock."""
        status_path = self._status_path(fingerprint)
        current_record = self._read_json(status_path)
        current = current_record.get("status")
        if status not in _TRANSITIONS.get(current, frozenset()):
            raise ValueError(f"cannot transition experiment from {current!r} to {status!r}")
        history = current_record.get("history")
        if not isinstance(history, list):
            raise TypeError("experiment status history must be a list")
        if current == "pending" and status == "running":
            try:
                self._reservation_path(fingerprint).unlink()
            except FileNotFoundError:
                pass
        atomic_write_json(status_path, {"status": status, "history": [*history, {"status": status}]})

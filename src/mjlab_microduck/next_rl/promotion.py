"""Durable, human-gated promotion state for evaluated Next RL capabilities."""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .artifacts import atomic_write_json
from .capabilities import CapabilityInventory
from .review import ReviewBundle, ReviewError
from .schema import ArtifactRef, Capability, EvaluationRef

_LOCK_TIMEOUT_SECONDS = 10.0


class PromotionError(ValueError):
    """A requested capability-promotion transition is invalid."""


def _text(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PromotionError(f"{label} must be non-empty")
    return value.strip()


def _semantic_version(value: str) -> tuple[int, int, int]:
    major, minor, patch = value.split(".")
    return int(major), int(minor), int(patch)


@dataclass(frozen=True)
class ReviewApproval:
    reviewer: str
    review_bundle_digest: str


@dataclass(frozen=True)
class PromotionAudit:
    action: str
    reviewer: str
    reason: str | None
    review_bundle_digest: str


@dataclass(frozen=True)
class PromotionRecord:
    id: str
    skill_id: str
    spec_version: str
    status: str
    policy_digest: str
    review_bundle_digest: str
    approval: ReviewApproval | None = None
    audit: tuple[PromotionAudit, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "skill_id": self.skill_id,
            "spec_version": self.spec_version,
            "status": self.status,
            "policy_digest": self.policy_digest,
            "review_bundle_digest": self.review_bundle_digest,
            "approval": None if self.approval is None else self.approval.__dict__,
            "audit": [item.__dict__ for item in self.audit],
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PromotionRecord:
        approval = raw["approval"]
        return cls(
            raw["id"], raw["skill_id"], raw["spec_version"], raw["status"],
            raw["policy_digest"], raw["review_bundle_digest"],
            None if approval is None else ReviewApproval(**approval),
            tuple(PromotionAudit(**item) for item in raw["audit"]),
        )


class PromotionStore:
    """Filesystem-backed transitions; every mutator holds one ownership lock."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    @property
    def _state_path(self) -> Path:
        return self.root / "state.json"

    @property
    def _lock_path(self) -> Path:
        return self.root / ".lock"

    @contextmanager
    def _lock(self):
        self.root.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + _LOCK_TIMEOUT_SECONDS
        descriptor = os.open(self._lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            while True:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("timed out waiting for promotion lock")
                    time.sleep(0.01)
            yield
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    def _load(self) -> dict[str, Any]:
        if not self._state_path.exists():
            return {"records": {}, "bundles": {}, "capabilities": []}
        raw = json.loads(self._state_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or set(raw) != {"records", "bundles", "capabilities"}:
            raise PromotionError("promotion state is invalid")
        return raw

    def _save(self, state: dict[str, Any]) -> None:
        atomic_write_json(self._state_path, state)

    def _report_path(self, digest: str) -> Path:
        return self.root / "reviews" / digest / "evaluation.json"

    def _write_report_once(self, digest: str, evidence: str) -> Path:
        target = self._report_path(digest)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.read_text(encoding="utf-8") != evidence:
                raise FileExistsError(f"immutable evaluation report differs at {target}")
            return target
        descriptor, temporary_name = tempfile.mkstemp(prefix=".evaluation.", dir=target.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                output.write(evidence)
                output.flush()
                os.fsync(output.fileno())
            try:
                os.link(temporary, target)
            except FileExistsError:
                if target.read_text(encoding="utf-8") != evidence:
                    raise FileExistsError(f"immutable evaluation report differs at {target}") from None
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        return target

    @staticmethod
    def _serialize_capability(value: Capability) -> dict[str, Any]:
        raw = value.as_dict()
        if value.evaluation is not None and value.evaluation.approval_provenance:
            raw["evaluation"]["approval_provenance"] = value.evaluation.approval_provenance
        return raw

    @classmethod
    def _replace_capability(cls, state: dict[str, Any], value: Capability) -> None:
        items = [Capability.from_dict(item) for item in state["capabilities"]]
        items = [item for item in items if (item.id, item.version) != (value.id, value.version)]
        state["capabilities"] = [cls._serialize_capability(item) for item in (*items, value)]

    @staticmethod
    def _active_learned(
        state: dict[str, Any], skill_id: str
    ) -> tuple[PromotionRecord, Capability] | None:
        records = [
            PromotionRecord.from_dict(raw)
            for raw in state["records"].values()
            if raw["skill_id"] == skill_id and raw["status"] == "learned"
        ]
        capabilities = [
            Capability.from_dict(raw)
            for raw in state["capabilities"]
            if raw["id"] == skill_id and raw["status"] == "learned"
        ]
        if len(records) > 1 or len(capabilities) > 1:
            raise PromotionError(
                f"multiple active learned entries exist for capability {skill_id!r}"
            )
        if bool(records) != bool(capabilities):
            raise PromotionError(
                f"active learned promotion state is inconsistent for capability {skill_id!r}"
            )
        if not records:
            return None
        record = records[0]
        capability = capabilities[0]
        if (
            record.spec_version != capability.version
            or capability.policy is None
            or record.policy_digest != capability.policy.sha256
        ):
            raise PromotionError(
                f"active learned promotion state is inconsistent for capability {skill_id!r}"
            )
        return record, capability

    @staticmethod
    def _capability(template: Capability, bundle: ReviewBundle, status: str, report_path: Path, approval: str | None = None) -> Capability:
        evidence = EvaluationRef(
            "evaluation_report", bundle.policy_digest, str(report_path), True,
            bundle.metric_summary, approval_provenance=approval,
        )
        return replace(
            template,
            status=status,
            policy=ArtifactRef(str(bundle.policy_path), "onnx", bundle.policy_digest),
            evaluation=evidence,
        )

    @staticmethod
    def _assert_binding(capability: Capability, bundle: ReviewBundle) -> None:
        try:
            bundle.verify()
            spec = bundle.skill_spec
        except ReviewError as error:
            raise PromotionError(str(error)) from error
        if capability.id != bundle.skill_id:
            raise PromotionError("capability skill_id does not match evaluated skill_id")
        if capability.version != bundle.spec_version:
            raise PromotionError("capability spec_version does not match evaluated spec_version")
        if capability.robot_model != spec.contract.robot_model:
            raise PromotionError("capability robot_model does not match bound skill spec")
        if capability.contract != spec.contract:
            raise PromotionError("capability policy contract does not match bound skill spec")

    def validate(self, capability: Capability, bundle: ReviewBundle) -> PromotionRecord:
        """Persist passing evaluation evidence for an available capability exactly once."""
        self._assert_binding(capability, bundle)
        if capability.status != "available":
            raise PromotionError("validation requires an available capability")
        with self._lock():
            state = self._load()
            active = self._active_learned(state, capability.id)
            if (
                active is not None
                and _semantic_version(capability.version)
                <= _semantic_version(active[0].spec_version)
            ):
                raise PromotionError(
                    "an available improvement requires a higher semantic version "
                    "than the active learned capability"
                )
            existing = next((Capability.from_dict(raw) for raw in state["capabilities"] if raw["id"] == capability.id and raw["version"] == capability.version), None)
            if existing is not None and existing.status != "available":
                raise PromotionError(f"validation cannot start from {existing.status!r}")
            report_path = self._write_report_once(bundle.digest, bundle.evaluation_json)
            record = PromotionRecord(
                uuid.uuid4().hex, capability.id, capability.version, "validated",
                bundle.policy_digest, bundle.digest,
                audit=(PromotionAudit("available", "", None, bundle.digest), PromotionAudit("validated", "", None, bundle.digest)),
            )
            state["records"][record.id] = record.as_dict()
            state["bundles"][record.id] = bundle.as_dict()
            self._replace_capability(state, self._capability(capability, bundle, "validated", report_path))
            self._save(state)
            return record

    def request_review(self, capability: Capability, bundle: ReviewBundle) -> PromotionRecord:
        """Advance only the exact validated candidate into human review."""
        self._assert_binding(capability, bundle)
        with self._lock():
            state = self._load()
            source = next((Capability.from_dict(raw) for raw in state["capabilities"] if raw["id"] == capability.id and raw["version"] == capability.version), None)
            if source is None or source.status != "validated":
                raise PromotionError("review request requires a validated capability")
            if source.policy is None or source.evaluation is None or source.policy.sha256 != bundle.policy_digest:
                raise PromotionError("validated capability evidence does not match review bundle")
            if Path(source.evaluation.report_path).read_text(encoding="utf-8") != bundle.evaluation_json:
                raise PromotionError("validated evaluation report does not match review bundle")
            records = {key: PromotionRecord.from_dict(raw) for key, raw in state["records"].items()}
            record = next((item for item in records.values() if item.skill_id == capability.id and item.spec_version == capability.version and item.policy_digest == bundle.policy_digest and item.status == "validated"), None)
            if record is None:
                raise PromotionError("validated promotion record is missing")
            record = replace(record, status="review_pending", audit=record.audit + (PromotionAudit("requested", "", None, bundle.digest),))
            records[record.id] = record
            state["records"] = {key: item.as_dict() for key, item in records.items()}
            state["bundles"][record.id] = bundle.as_dict()
            self._replace_capability(state, self._capability(source, bundle, "review_pending", Path(source.evaluation.report_path)))
            self._save(state)
            return record

    def _pending(self, state: dict[str, Any], record_id: str) -> tuple[PromotionRecord, ReviewBundle]:
        try:
            record = PromotionRecord.from_dict(state["records"][record_id])
            bundle = ReviewBundle.from_dict(state["bundles"][record_id])
        except KeyError as error:
            raise PromotionError(f"unknown promotion record {record_id!r}") from error
        except ReviewError as error:
            raise PromotionError(str(error)) from error
        if record.status != "review_pending":
            raise PromotionError(f"cannot review a candidate in {record.status!r} state")
        source = Capability.from_dict(next(raw for raw in state["capabilities"] if raw["id"] == record.skill_id and raw["version"] == record.spec_version))
        self._assert_persisted_report(source, bundle)
        self._assert_binding(source, bundle)
        if bundle.digest != record.review_bundle_digest:
            raise PromotionError("approval must use the exact review bundle requested")
        return record, bundle

    @staticmethod
    def _assert_persisted_report(capability: Capability, bundle: ReviewBundle) -> None:
        if capability.evaluation is None or capability.evaluation.report_path is None:
            raise PromotionError("validated evaluation report is missing")
        path = Path(capability.evaluation.report_path)
        if not path.is_file() or path.stat().st_size == 0:
            raise PromotionError("validated evaluation report is missing")
        if path.read_text(encoding="utf-8") != bundle.evaluation_json:
            raise PromotionError("validated evaluation report does not match review bundle")

    def approve(self, record_id: str, *, reviewer: str) -> PromotionRecord:
        reviewer = _text(reviewer, "reviewer")
        with self._lock():
            state = self._load()
            record, bundle = self._pending(state, record_id)
            active = self._active_learned(state, record.skill_id)
            if active is not None:
                prior, prior_capability = active
                if _semantic_version(record.spec_version) <= _semantic_version(
                    prior.spec_version
                ):
                    raise PromotionError(
                        "approval requires a higher semantic version than the current "
                        "learned capability"
                    )
                if bundle.baseline is None:
                    raise PromotionError(
                        "improvement approval requires a baseline bound to the current "
                        "learned policy"
                    )
                if bundle.baseline.policy_digest != prior.policy_digest:
                    raise PromotionError(
                        "improvement baseline policy digest does not match the current "
                        "learned policy"
                    )
            records = {key: PromotionRecord.from_dict(raw) for key, raw in state["records"].items()}
            if active is not None:
                records[prior.id] = replace(prior, status="superseded")
            learned = replace(record, status="learned", approval=ReviewApproval(reviewer, bundle.digest), audit=record.audit + (PromotionAudit("approved", reviewer, None, bundle.digest),))
            records[record.id] = learned
            state["records"] = {key: item.as_dict() for key, item in records.items()}
            if active is not None:
                self._replace_capability(
                    state,
                    replace(prior_capability, status="superseded"),
                )
            source = Capability.from_dict(next(raw for raw in state["capabilities"] if raw["id"] == record.skill_id and raw["version"] == record.spec_version))
            report_path = Path(source.evaluation.report_path)
            self._replace_capability(state, self._capability(source, bundle, "learned", report_path, bundle.digest))
            self._save(state)
            return learned

    def reject(self, record_id: str, *, reviewer: str, reason: str) -> PromotionRecord:
        reviewer = _text(reviewer, "reviewer")
        reason = _text(reason, "reason")
        with self._lock():
            state = self._load()
            record, bundle = self._pending(state, record_id)
            rejected = replace(record, status="validated", audit=record.audit + (PromotionAudit("rejected", reviewer, reason, bundle.digest),))
            state["records"][record_id] = rejected.as_dict()
            source = Capability.from_dict(next(raw for raw in state["capabilities"] if raw["id"] == record.skill_id and raw["version"] == record.spec_version))
            report_path = Path(source.evaluation.report_path)
            self._replace_capability(state, self._capability(source, bundle, "validated", report_path))
            self._save(state)
            return rejected

    def current_learned(self, skill_id: str) -> PromotionRecord | None:
        skill_id = _text(skill_id, "skill_id")
        with self._lock():
            active = self._active_learned(self._load(), skill_id)
        return None if active is None else active[0]

    def inventory(self) -> CapabilityInventory:
        with self._lock():
            capabilities = tuple(Capability.from_dict(raw) for raw in self._load()["capabilities"])
        return CapabilityInventory(capabilities)

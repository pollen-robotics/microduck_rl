"""Durable, locked promotion state with inventory-visible human approval."""

from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .artifacts import atomic_write_json, canonical_json
from .capabilities import CapabilityInventory
from .review import ReviewBundle, ReviewError
from .schema import ArtifactRef, Capability, EvaluationRef


class PromotionError(ValueError): pass


def _text(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip(): raise PromotionError(f"{label} must be non-empty")
    return value.strip()


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
        return {"id": self.id, "skill_id": self.skill_id, "spec_version": self.spec_version, "status": self.status,
                "policy_digest": self.policy_digest, "review_bundle_digest": self.review_bundle_digest,
                "approval": None if self.approval is None else self.approval.__dict__, "audit": [item.__dict__ for item in self.audit]}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PromotionRecord:
        approval = raw["approval"]
        return cls(raw["id"], raw["skill_id"], raw["spec_version"], raw["status"], raw["policy_digest"], raw["review_bundle_digest"],
                   None if approval is None else ReviewApproval(**approval), tuple(PromotionAudit(**item) for item in raw["audit"]))


class PromotionStore:
    def __init__(self, root: str | Path) -> None: self.root = Path(root)

    @property
    def _state_path(self) -> Path: return self.root / "state.json"
    @property
    def _lock_path(self) -> Path: return self.root / ".lock"

    @contextmanager
    def _lock(self):
        self.root.mkdir(parents=True, exist_ok=True); owner = uuid.uuid4().hex; deadline = time.monotonic() + 10
        while True:
            try:
                fd = os.open(self._lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(fd, "w") as output: output.write(owner); output.flush(); os.fsync(output.fileno())
                break
            except FileExistsError:
                if time.monotonic() > deadline: raise TimeoutError("timed out waiting for promotion lock")
                time.sleep(.01)
        try: yield
        finally:
            try:
                if self._lock_path.read_text() == owner: self._lock_path.unlink()
            except FileNotFoundError: pass

    def _load(self) -> dict[str, Any]:
        if not self._state_path.exists(): return {"records": {}, "bundles": {}, "capabilities": []}
        raw = json.loads(self._state_path.read_text())
        if not isinstance(raw, dict) or set(raw) != {"records", "bundles", "capabilities"}: raise PromotionError("promotion state is invalid")
        return raw

    def _save(self, state: dict[str, Any]) -> None: atomic_write_json(self._state_path, state)

    @staticmethod
    def _capability(template: Capability, bundle: ReviewBundle, status: str, approval: str | None = None) -> Capability:
        evaluation = EvaluationRef("evaluation_report", bundle.policy_digest, f"reviews/{bundle.digest}/evaluation.json", True, bundle.metric_summary, approval_provenance=approval)
        return replace(template, status=status, policy=ArtifactRef(str(bundle.policy_path), "onnx", bundle.policy_digest), evaluation=evaluation)

    @staticmethod
    def _replace_capability(state: dict[str, Any], value: Capability) -> None:
        items = [Capability.from_dict(item) for item in state["capabilities"]]
        items = [item for item in items if (item.id, item.version) != (value.id, value.version)] + [value]
        state["capabilities"] = [PromotionStore._serialize_capability(item) for item in items]

    @staticmethod
    def _serialize_capability(value: Capability) -> dict[str, Any]:
        """Schema serialisation omits report approval provenance; retain it durably."""
        raw = value.as_dict()
        if value.evaluation is not None and value.evaluation.approval_provenance:
            raw["evaluation"]["approval_provenance"] = value.evaluation.approval_provenance
        return raw

    def request_review(self, capability: Capability, bundle: ReviewBundle) -> PromotionRecord:
        try: bundle.verify()
        except ReviewError as error: raise PromotionError(str(error)) from error
        if capability.id != bundle.skill_id: raise PromotionError("capability skill_id does not match evaluated skill_id")
        if capability.version != bundle.spec_version: raise PromotionError("capability spec_version does not match evaluated spec_version")
        with self._lock():
            state = self._load(); records = {key: PromotionRecord.from_dict(value) for key, value in state["records"].items()}
            reusable = next((record for record in records.values() if record.skill_id == capability.id and record.spec_version == capability.version and record.policy_digest == bundle.policy_digest and record.status == "validated"), None)
            if reusable is None:
                record = PromotionRecord(
                    uuid.uuid4().hex, capability.id, capability.version, "review_pending", bundle.policy_digest, bundle.digest,
                    audit=(
                        PromotionAudit("available", "", None, bundle.digest),
                        PromotionAudit("validated", "", None, bundle.digest),
                        PromotionAudit("requested", "", None, bundle.digest),
                    ),
                )
            else:
                record = replace(reusable, status="review_pending", review_bundle_digest=bundle.digest, audit=reusable.audit + (PromotionAudit("requested", "", None, bundle.digest),))
            records[record.id] = record; state["records"] = {key: value.as_dict() for key, value in records.items()}; state["bundles"][record.id] = bundle.as_dict()
            current = next((Capability.from_dict(item) for item in state["capabilities"] if item["id"] == capability.id and item["version"] == capability.version), None)
            if current is None or current.status != "learned": self._replace_capability(state, self._capability(capability, bundle, "review_pending"))
            self._save(state); return record

    def _pending(self, state: dict[str, Any], record_id: str) -> tuple[PromotionRecord, ReviewBundle]:
        try: record = PromotionRecord.from_dict(state["records"][record_id]); bundle = ReviewBundle.from_dict(state["bundles"][record_id])
        except KeyError as error: raise PromotionError(f"unknown promotion record {record_id!r}") from error
        if record.status != "review_pending": raise PromotionError(f"cannot review a candidate in {record.status!r} state")
        try: bundle.verify()
        except ReviewError as error: raise PromotionError(f"review bundle is invalid: {error}") from error
        if bundle.digest != record.review_bundle_digest: raise PromotionError("approval must use the exact review bundle requested")
        return record, bundle

    def approve(self, record_id: str, *, reviewer: str) -> PromotionRecord:
        reviewer = _text(reviewer, "reviewer")
        with self._lock():
            state = self._load(); record, bundle = self._pending(state, record_id)
            records = {key: PromotionRecord.from_dict(value) for key, value in state["records"].items()}
            for key, item in records.items():
                if item.skill_id == record.skill_id and item.status == "learned": records[key] = replace(item, status="superseded")
            learned = replace(record, status="learned", approval=ReviewApproval(reviewer, bundle.digest), audit=record.audit + (PromotionAudit("approved", reviewer, None, bundle.digest),))
            records[record_id] = learned; state["records"] = {key: value.as_dict() for key, value in records.items()}
            for index, raw in enumerate(state["capabilities"]):
                item = Capability.from_dict(raw)
                if item.id == record.skill_id and item.status == "learned": state["capabilities"][index] = self._serialize_capability(replace(item, status="superseded"))
            template = next((Capability.from_dict(raw) for raw in state["capabilities"] if raw["id"] == record.skill_id and raw["version"] == record.spec_version), None)
            if template is None: raise PromotionError("reviewed capability is missing")
            self._replace_capability(state, self._capability(template, bundle, "learned", bundle.digest)); self._save(state); return learned

    def reject(self, record_id: str, *, reviewer: str, reason: str) -> PromotionRecord:
        reviewer, reason = _text(reviewer, "reviewer"), _text(reason, "reason")
        with self._lock():
            state = self._load(); record, bundle = self._pending(state, record_id)
            rejected = replace(record, status="validated", audit=record.audit + (PromotionAudit("rejected", reviewer, reason, bundle.digest),))
            state["records"][record_id] = rejected.as_dict()
            current = next((Capability.from_dict(raw) for raw in state["capabilities"] if raw["id"] == record.skill_id and raw["version"] == record.spec_version), None)
            if current is None or current.status != "learned":
                template = current or Capability.from_dict(next(raw for raw in state["capabilities"] if raw["id"] == record.skill_id))
                self._replace_capability(state, self._capability(template, bundle, "validated"))
            self._save(state); return rejected

    def current_learned(self, skill_id: str) -> PromotionRecord | None:
        with self._lock():
            records = [PromotionRecord.from_dict(raw) for raw in self._load()["records"].values()]
        return next((record for record in records if record.skill_id == _text(skill_id, "skill_id") and record.status == "learned"), None)

    def inventory(self) -> CapabilityInventory:
        with self._lock(): return CapabilityInventory(Capability.from_dict(raw) for raw in self._load()["capabilities"])

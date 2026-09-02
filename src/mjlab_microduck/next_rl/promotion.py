"""In-memory state machine for explicit human promotion of learned skills."""

from __future__ import annotations

from dataclasses import dataclass, replace
from uuid import uuid4

from .review import ReviewBundle, ReviewError


class PromotionError(ValueError):
    """Raised when a candidate cannot make the requested promotion transition."""


def _text(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PromotionError(f"{label} must be non-empty")
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
    status: str
    policy_digest: str
    review_bundle_digest: str
    approval: ReviewApproval | None = None
    audit: tuple[PromotionAudit, ...] = ()


class PromotionStore:
    """Keep candidates local until deterministic evidence and human review both agree."""

    def __init__(self) -> None:
        self._records: dict[str, PromotionRecord] = {}
        self._bundles: dict[str, ReviewBundle] = {}

    def request_review(self, skill_id: str, bundle: ReviewBundle) -> PromotionRecord:
        skill_id = _text(skill_id, "skill_id")
        if bundle.passed is not True:
            raise PromotionError("review requires a passing evaluation")
        try:
            bundle.verify()
        except ReviewError as error:
            raise PromotionError(f"review bundle is invalid: {error}") from error
        record = PromotionRecord(
            uuid4().hex,
            skill_id,
            "review_pending",
            bundle.policy_digest,
            bundle.digest,
        )
        self._records[record.id] = record
        self._bundles[record.id] = bundle
        return record

    def approve(self, record_id: str, *, reviewer: str) -> PromotionRecord:
        reviewer = _text(reviewer, "reviewer")
        record, bundle = self._pending(record_id)
        self._verify_exact_bundle(record, bundle)
        for existing_id, existing in tuple(self._records.items()):
            if existing.skill_id == record.skill_id and existing.status == "learned":
                self._records[existing_id] = replace(existing, status="superseded")
        approved = replace(
            record,
            status="learned",
            approval=ReviewApproval(reviewer, bundle.digest),
            audit=record.audit + (PromotionAudit("approved", reviewer, None, bundle.digest),),
        )
        self._records[record_id] = approved
        return approved

    def reject(self, record_id: str, *, reviewer: str, reason: str) -> PromotionRecord:
        reviewer = _text(reviewer, "reviewer")
        reason = _text(reason, "reason")
        record, bundle = self._pending(record_id)
        self._verify_exact_bundle(record, bundle)
        rejected = replace(
            record,
            status="validated",
            audit=record.audit + (PromotionAudit("rejected", reviewer, reason, bundle.digest),),
        )
        self._records[record_id] = rejected
        return rejected

    def current_learned(self, skill_id: str) -> PromotionRecord | None:
        skill_id = _text(skill_id, "skill_id")
        return next(
            (record for record in self._records.values() if record.skill_id == skill_id and record.status == "learned"),
            None,
        )

    def _pending(self, record_id: str) -> tuple[PromotionRecord, ReviewBundle]:
        try:
            record = self._records[record_id]
            bundle = self._bundles[record_id]
        except KeyError as error:
            raise PromotionError(f"unknown promotion record {record_id!r}") from error
        if record.status != "review_pending":
            raise PromotionError(f"cannot review a candidate in {record.status!r} state")
        return record, bundle

    @staticmethod
    def _verify_exact_bundle(record: PromotionRecord, bundle: ReviewBundle) -> None:
        try:
            bundle.verify()
        except ReviewError as error:
            raise PromotionError(f"review bundle is invalid: {error}") from error
        if bundle.digest != record.review_bundle_digest:
            raise PromotionError("approval must use the exact review bundle requested")

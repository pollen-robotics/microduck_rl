"""Behavioural contracts for durable, explicit human promotion of learned skills."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import replace

import pytest

from mjlab_microduck.next_rl.evaluation import EvaluationReport, MetricResult, ScenarioResult, select_worst_case
from mjlab_microduck.next_rl.promotion import PromotionError, PromotionStore
from mjlab_microduck.next_rl.review import ReviewBundle, ReviewClip
from mjlab_microduck.next_rl.artifacts import canonical_json
from mjlab_microduck.next_rl.schema import ArtifactRef, Capability, PolicyContract


@pytest.fixture
def capability() -> Capability:
    return Capability.from_dict({
        "id": "hello", "version": "1.0.0", "aliases": [], "robot_model": "microduck",
        "contract": PolicyContract.microduck().as_dict(), "status": "available",
    })


@pytest.fixture
def bundle(tmp_path) -> ReviewBundle:
    policy = tmp_path / "policy.onnx"
    policy.write_bytes(b"evaluated policy")
    digest = hashlib.sha256(policy.read_bytes()).hexdigest()
    scenarios = tuple(
        ScenarioResult(
            f"{family}-1", family, seed, {"falls": 0.0}, digest,
            (MetricResult(f"{family}-1", "falls", "count", "maximum", 0, 0, True, True, 0),),
        )
        for family, seed in (("nominal", 7), ("entry", 8), ("exit", 9), ("stress", 10))
    )
    report = EvaluationReport("hello", "1.0.0", "eval-a", scenarios, True, ArtifactRef(str(policy), "onnx", digest))
    by_family = {scenario.family: scenario for scenario in scenarios}
    expected = {**by_family, "worst_case": select_worst_case(scenarios)}
    clips = {}
    for role, scenario in expected.items():
        path = tmp_path / f"{role}.mp4"
        path.write_bytes(role.encode())
        clips[role] = ReviewClip(role, scenario.scenario_id, scenario.seed, path, digest)
    return ReviewBundle.build(report, clips)


@pytest.fixture
def failed_bundle(bundle) -> ReviewBundle:
    return replace(bundle, passed=False)


@pytest.fixture
def store(tmp_path) -> PromotionStore:
    return PromotionStore(tmp_path / "promotion-state")


@pytest.fixture
def prior(store, capability, bundle):
    store.validate(capability, bundle)
    pending = store.request_review(capability, bundle)
    return store.approve(pending.id, reviewer="previous-reviewer")


def pending(store, capability, bundle):
    store.validate(capability, bundle)
    return store.request_review(capability, bundle)


def test_passing_evaluation_without_human_approval_is_review_pending(store, capability, bundle):
    with pytest.raises(PromotionError, match="validated"):
        store.request_review(capability, bundle)
    store.validate(capability, bundle)
    record = store.request_review(capability, bundle)
    assert record.status == "review_pending"
    assert [entry.action for entry in record.audit] == ["available", "validated", "requested"]


def test_failed_evaluation_cannot_enter_review(store, capability, failed_bundle):
    with pytest.raises(PromotionError, match="passing evaluation"):
        store.request_review(capability, failed_bundle)


def test_promotion_requires_capability_matching_evaluated_skill_and_version(store, capability, bundle):
    with pytest.raises(PromotionError, match="skill_id"):
        store.request_review(replace(capability, id="other"), bundle)
    with pytest.raises(PromotionError, match="spec_version"):
        store.request_review(replace(capability, version="2.0.0"), bundle)


def test_approval_requires_reviewer_and_exact_bundle(store, capability, bundle):
    review = pending(store, capability, bundle)
    learned = store.approve(review.id, reviewer="rakesh")
    assert learned.status == "learned"
    assert learned.approval.review_bundle_digest == bundle.digest
    with pytest.raises(PromotionError, match="reviewer"):
        store.approve(review.id, reviewer=" ")


def test_approval_updates_durable_inventory_for_planner_observation(store, capability, bundle):
    review = pending(store, capability, bundle)
    store.approve(review.id, reviewer="rakesh")
    learned = PromotionStore(store.root).inventory().resolve("hello").capability
    assert learned is not None
    assert learned.status == "learned"
    assert learned.policy.sha256 == bundle.policy_digest
    assert learned.evaluation.approval_provenance == bundle.digest


def test_rejection_preserves_prior_learned_policy(store, capability, bundle, prior):
    evidence = json.loads(bundle.evaluation_json)
    evidence["spec_version"] = "1.0.1"
    versioned_json = canonical_json(evidence)
    candidate_bundle = replace(bundle, evaluation_json=versioned_json, evaluation_digest=hashlib.sha256(versioned_json.encode()).hexdigest())
    candidate = replace(capability, version="1.0.1")
    rejected = store.reject(pending(store, candidate, candidate_bundle).id, reviewer="rakesh", reason="leans")
    assert rejected.status == "validated"
    assert store.current_learned("hello") == prior


def test_re_request_after_rejection_preserves_audit_and_returns_to_review(store, capability, bundle):
    review = pending(store, capability, bundle)
    rejected = store.reject(review.id, reviewer="rakesh", reason="leans")
    requested = store.request_review(capability, bundle)
    assert requested.id == rejected.id
    assert requested.status == "review_pending"
    assert [entry.action for entry in requested.audit] == ["available", "validated", "requested", "rejected", "requested"]


def test_concurrent_approvals_leave_one_authoritative_learned_policy(store, capability, bundle):
    review = pending(store, capability, bundle)
    outcomes: list[object] = []
    start = threading.Barrier(3)

    def approve() -> None:
        start.wait()
        try:
            outcomes.append(PromotionStore(store.root).approve(review.id, reviewer="rakesh"))
        except PromotionError as error:
            outcomes.append(error)

    workers = [threading.Thread(target=approve) for _ in range(2)]
    for worker in workers:
        worker.start()
    start.wait()
    for worker in workers:
        worker.join(timeout=5)
        assert not worker.is_alive()
    assert len([outcome for outcome in outcomes if getattr(outcome, "status", None) == "learned"]) == 1
    assert store.current_learned("hello").status == "learned"


def test_rejection_requires_an_auditable_reviewer_and_reason(store, capability, bundle):
    review = pending(store, capability, bundle)
    with pytest.raises(PromotionError, match="reason"):
        store.reject(review.id, reviewer="rakesh", reason=" ")


def test_validation_persists_an_immutable_exact_evaluation_report(store, capability, bundle):
    store.validate(capability, bundle)
    validated = store.inventory().resolve("hello").capability
    report_path = validated.evaluation.report_path

    assert open(report_path, encoding="utf-8").read() == bundle.evaluation_json
    with pytest.raises(FileExistsError):
        store._write_report_once(bundle.digest, "{}")


def test_missing_or_mutated_persisted_evaluation_blocks_approval_and_rejection(store, capability, bundle):
    review = pending(store, capability, bundle)
    report_path = store.inventory().resolve("hello").capability.evaluation.report_path
    os.unlink(report_path)
    with pytest.raises(PromotionError, match="evaluation report"):
        store.approve(review.id, reviewer="rakesh")

    store = PromotionStore(store.root.parent / "mutated-state")
    review = pending(store, capability, bundle)
    report_path = store.inventory().resolve("hello").capability.evaluation.report_path
    open(report_path, "w", encoding="utf-8").write("{}")
    with pytest.raises(PromotionError, match="evaluation report"):
        store.reject(review.id, reviewer="rakesh", reason="leans")


def test_advisory_lock_releases_with_descriptor_and_does_not_break_live_owner(store, capability, bundle, monkeypatch):
    import fcntl

    store.root.mkdir(parents=True)
    descriptor = open(store._lock_path, "a+", encoding="utf-8")
    fcntl.flock(descriptor.fileno(), fcntl.LOCK_EX)
    monkeypatch.setattr("mjlab_microduck.next_rl.promotion._LOCK_TIMEOUT_SECONDS", 0)
    with pytest.raises(TimeoutError):
        store.validate(capability, bundle)
    fcntl.flock(descriptor.fileno(), fcntl.LOCK_UN)
    descriptor.close()
    store.validate(capability, bundle)


def test_distinct_pending_versions_serialize_to_one_learned_policy(store, capability, bundle):
    evidence = json.loads(bundle.evaluation_json)
    evidence["spec_version"] = "1.0.1"
    text = canonical_json(evidence)
    candidate = replace(bundle, evaluation_json=text, evaluation_digest=hashlib.sha256(text.encode()).hexdigest())
    pending_a = pending(store, capability, bundle)
    pending_b = pending(store, replace(capability, version="1.0.1"), candidate)

    first = store.approve(pending_a.id, reviewer="a")
    second = store.approve(pending_b.id, reviewer="b")
    records = store._load()["records"].values()

    assert first.status == "learned"
    assert second.status == "learned"
    assert sum(record["status"] == "learned" for record in records) == 1

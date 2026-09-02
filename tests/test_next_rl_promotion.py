"""Behavioural contracts for explicit human promotion of learned skills."""

from __future__ import annotations

import hashlib

import pytest

from mjlab_microduck.next_rl.evaluation import EvaluationReport, MetricResult, ScenarioResult
from mjlab_microduck.next_rl.promotion import PromotionError, PromotionStore
from mjlab_microduck.next_rl.review import ReviewBundle, ReviewClip
from mjlab_microduck.next_rl.schema import ArtifactRef


@pytest.fixture
def bundle(tmp_path) -> ReviewBundle:
    policy = tmp_path / "policy.onnx"
    policy.write_bytes(b"evaluated policy")
    digest = hashlib.sha256(policy.read_bytes()).hexdigest()
    result = MetricResult("nominal-1", "falls", "count", "maximum", 0.0, 0.0, True, True, 0.0)
    report = EvaluationReport(
        "hello", "1.0.0", "eval-a", (ScenarioResult("nominal-1", "nominal", 7, {"falls": 0.0}, digest, (result,)),),
        True, ArtifactRef(str(policy), "onnx", digest),
    )
    clips = {}
    for role in ("nominal", "entry", "exit", "stress", "worst_case"):
        path = tmp_path / f"{role}.mp4"
        path.write_bytes(role.encode())
        clips[role] = ReviewClip(role, f"{role}-scenario", 7, path, digest)
    return ReviewBundle.build(report, clips)


@pytest.fixture
def failed_bundle(bundle) -> ReviewBundle:
    return ReviewBundle(
        bundle.evaluation_json, bundle.evaluation_digest, bundle.policy_digest, bundle.clips,
        bundle.metric_summary, False,
    )


@pytest.fixture
def store() -> PromotionStore:
    return PromotionStore()


@pytest.fixture
def prior(store, bundle):
    pending = store.request_review("hello", bundle)
    return store.approve(pending.id, reviewer="previous-reviewer")


def test_passing_evaluation_without_human_approval_is_review_pending(store, bundle):
    record = store.request_review("hello", bundle)

    assert record.status == "review_pending"


def test_failed_evaluation_cannot_enter_review(store, failed_bundle):
    with pytest.raises(PromotionError, match="passing evaluation"):
        store.request_review("hello", failed_bundle)


def test_forged_passing_flag_cannot_admit_a_failed_evaluation(store, bundle):
    evaluation_json = bundle.evaluation_json.replace('"passed":true', '"passed":false')
    forged = ReviewBundle(
        evaluation_json,
        hashlib.sha256(evaluation_json.encode("utf-8")).hexdigest(),
        bundle.policy_digest,
        bundle.clips,
        bundle.metric_summary,
        True,
        policy_path=bundle.policy_path,
    )

    with pytest.raises(PromotionError, match="passing evaluation"):
        store.request_review("hello", forged)


def test_approval_requires_reviewer_and_exact_bundle(store, bundle):
    pending = store.request_review("hello", bundle)
    learned = store.approve(pending.id, reviewer="rakesh")

    assert learned.status == "learned"
    assert learned.approval.review_bundle_digest == bundle.digest
    with pytest.raises(PromotionError, match="reviewer"):
        store.approve(store.request_review("other", bundle).id, reviewer=" ")


def test_rejection_preserves_prior_learned_policy(store, bundle, prior):
    rejected = store.reject(
        store.request_review("hello-v2", bundle).id, reviewer="rakesh", reason="leans"
    )

    assert rejected.status == "validated"
    assert store.current_learned("hello") == prior


def test_rejection_requires_an_auditable_reviewer_and_reason(store, bundle):
    pending = store.request_review("hello", bundle)

    with pytest.raises(PromotionError, match="reason"):
        store.reject(pending.id, reviewer="rakesh", reason=" ")

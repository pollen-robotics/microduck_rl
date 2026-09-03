"""Behavioural contracts for durable, explicit human promotion of learned skills."""

from __future__ import annotations

import hashlib
import os
import threading
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pytest
from test_next_rl_support import (
    write_renderer_sidecar,
    write_test_video,
    write_tiny_policy,
)

from mjlab_microduck.next_rl.artifacts import canonical_json
from mjlab_microduck.next_rl.capabilities import Disposition, plan_skill
from mjlab_microduck.next_rl.evaluation import (
    EvaluationReport,
    ScenarioResult,
    evaluate_thresholds,
    select_worst_case,
)
from mjlab_microduck.next_rl.promotion import PromotionError, PromotionStore
from mjlab_microduck.next_rl.review import RendererEvidence, ReviewBundle
from mjlab_microduck.next_rl.schema import (
    ArtifactRef,
    Capability,
    MetricThreshold,
    PolicyContract,
    SkillSpec,
)

_SCENARIOS = (("nominal", 7), ("entry", 8), ("exit", 9), ("stress", 10))


def skill_spec(
    version: str,
    *,
    support_limit: float | None = None,
    support_mandatory: bool = True,
) -> SkillSpec:
    metrics = [MetricThreshold("falls", "count", "maximum", 0)]
    if support_limit is not None:
        metrics.append(
            MetricThreshold(
                "support",
                "ratio",
                "minimum",
                support_limit,
                support_mandatory,
            )
        )
    return SkillSpec(
        "hello",
        version,
        "promotion contract",
        PolicyContract.microduck(),
        tuple(metrics),
        (1,),
        tuple(seed for _, seed in _SCENARIOS),
        held_out_scenarios=tuple(family for family, _ in _SCENARIOS),
    )


def evaluation_report(
    spec: SkillSpec,
    policy: Path,
    *,
    support_values: Mapping[str, float] | None = None,
) -> EvaluationReport:
    digest = hashlib.sha256(policy.read_bytes()).hexdigest()
    scenarios = []
    for family, seed in _SCENARIOS:
        metrics = {"falls": 0.0}
        if support_values is not None:
            metrics["support"] = support_values[family]
        scenarios.append(
            ScenarioResult(f"{family}-1", family, seed, metrics, digest)
        )
    return evaluate_thresholds(
        spec,
        scenarios,
        evaluator_revision="eval-a",
        policy=ArtifactRef(str(policy), "onnx", digest),
    )


def build_bundle(
    root: Path,
    *,
    label: str,
    version: str = "1.0.0",
    weight: float = 0.1,
    baseline_policy: Path | None = None,
    support_values: Mapping[str, float] | None = None,
    support_limit: float | None = None,
    support_mandatory: bool = True,
) -> ReviewBundle:
    spec = skill_spec(
        version,
        support_limit=support_limit,
        support_mandatory=support_mandatory,
    )
    policy = write_tiny_policy(root / f"{label}.onnx", weight=weight)
    report = evaluation_report(spec, policy, support_values=support_values)
    evaluation_json = canonical_json(report.as_dict())
    evaluation_digest = hashlib.sha256(evaluation_json.encode()).hexdigest()
    by_family = {scenario.family: scenario for scenario in report.scenarios}
    expected = {**by_family, "worst_case": select_worst_case(report.scenarios)}
    evidence = {}
    for role, scenario in expected.items():
        video = write_test_video(root / f"{label}-{role}.mp4")
        sidecar = write_renderer_sidecar(
            root / f"{label}-{role}.renderer.json",
            role=role,
            scenario_id=scenario.scenario_id,
            seed=scenario.seed,
            policy_sha256=report.policy.sha256,
            evaluation_digest=evaluation_digest,
            video_path=video,
        )
        evidence[role] = RendererEvidence.load(sidecar)
    baseline = None
    if baseline_policy is not None:
        baseline = evaluation_report(
            spec,
            baseline_policy,
            support_values=support_values,
        )
    return ReviewBundle.build(report, evidence, spec=spec, baseline=baseline)


@pytest.fixture
def capability() -> Capability:
    return Capability.from_dict({
        "id": "hello", "version": "1.0.0", "aliases": [], "robot_model": "microduck",
        "contract": PolicyContract.microduck().as_dict(), "status": "available",
    })


@pytest.fixture
def bundle(tmp_path) -> ReviewBundle:
    return build_bundle(tmp_path, label="v1")


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


def test_promotion_requires_capability_matching_bound_robot_and_contract(store, capability, bundle):
    with pytest.raises(PromotionError, match="robot_model"):
        store.validate(replace(capability, robot_model="other"), bundle)
    with pytest.raises(PromotionError, match="policy contract"):
        store.validate(replace(capability, contract=replace(capability.contract, obs_len=62)), bundle)


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


def test_rejection_preserves_prior_learned_policy(
    store, capability, bundle, prior, tmp_path
):
    candidate_bundle = build_bundle(
        tmp_path,
        label="rejected-v101",
        version="1.0.1",
        weight=0.2,
        baseline_policy=bundle.policy_path,
    )
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


@pytest.mark.parametrize("version", ("1.0.0", "0.9.9"))
def test_available_improvement_must_bump_the_active_semantic_version(
    store, capability, bundle, prior, tmp_path, version
):
    candidate_bundle = build_bundle(
        tmp_path,
        label=f"invalid-{version.replace('.', '-')}",
        version=version,
        weight=0.2,
        baseline_policy=bundle.policy_path,
    )

    with pytest.raises(PromotionError, match="higher semantic version"):
        store.validate(replace(capability, version=version), candidate_bundle)

    assert store.current_learned("hello") == prior


def test_approval_rejects_a_lower_candidate_that_was_already_pending(
    store, capability, bundle, tmp_path
):
    lower_bundle = build_bundle(
        tmp_path,
        label="pending-v099",
        version="0.9.9",
        weight=0.2,
    )
    lower = pending(store, replace(capability, version="0.9.9"), lower_bundle)
    current = pending(store, capability, bundle)
    store.approve(current.id, reviewer="current-reviewer")

    with pytest.raises(PromotionError, match="higher semantic version"):
        store.approve(lower.id, reviewer="stale-reviewer")

    assert store.current_learned("hello").id == current.id


def test_approval_rejects_a_distinct_same_version_pending_record(
    store, capability, bundle
):
    review = pending(store, capability, bundle)
    state = store._load()
    duplicate = replace(review, id="duplicate-pending-record")
    state["records"][duplicate.id] = duplicate.as_dict()
    state["bundles"][duplicate.id] = state["bundles"][review.id]
    store._save(state)
    store.approve(review.id, reviewer="first-reviewer")

    with pytest.raises(PromotionError, match="higher semantic version"):
        store.approve(duplicate.id, reviewer="second-reviewer")

    assert store.current_learned("hello").id == review.id


def test_improvement_approval_requires_a_baseline_bound_to_current_policy(
    store, capability, bundle, prior, tmp_path
):
    candidate_bundle = build_bundle(
        tmp_path,
        label="v101-without-baseline",
        version="1.0.1",
        weight=0.2,
    )
    review = pending(store, replace(capability, version="1.0.1"), candidate_bundle)

    with pytest.raises(PromotionError, match="baseline"):
        store.approve(review.id, reviewer="rakesh")

    assert store.current_learned("hello") == prior


def test_approval_rejects_a_stale_improvement_baseline(
    store, capability, bundle, prior, tmp_path
):
    stale_bundle = build_bundle(
        tmp_path,
        label="stale-v200",
        version="2.0.0",
        weight=0.3,
        baseline_policy=bundle.policy_path,
    )
    stale = pending(store, replace(capability, version="2.0.0"), stale_bundle)
    first_bundle = build_bundle(
        tmp_path,
        label="winner-v101",
        version="1.0.1",
        weight=0.2,
        baseline_policy=bundle.policy_path,
    )
    first = pending(store, replace(capability, version="1.0.1"), first_bundle)
    store.approve(first.id, reviewer="first-reviewer")

    with pytest.raises(PromotionError, match="baseline policy digest"):
        store.approve(stale.id, reviewer="stale-reviewer")

    assert store.current_learned("hello").id == first.id
    assert store._load()["records"][stale.id]["status"] == "review_pending"


def test_improvement_lifecycle_requires_bump_then_promotes_and_resolves_new_version(
    store, capability, bundle, prior, tmp_path
):
    same_version = plan_skill(
        skill_spec("1.0.0"),
        store.inventory(),
        improve_reason="Improve the approved policy.",
    )
    assert same_version.disposition == Disposition.BLOCKED
    assert "version bump" in same_version.reason

    improvement = plan_skill(
        skill_spec("1.0.1"),
        store.inventory(),
        improve_reason="Improve the approved policy.",
    )
    assert improvement.disposition == Disposition.TRAIN_NEW
    assert improvement.capability is not None
    assert improvement.capability.version == "1.0.0"

    candidate_bundle = build_bundle(
        tmp_path,
        label="approved-v101",
        version="1.0.1",
        weight=0.2,
        baseline_policy=bundle.policy_path,
    )
    review = pending(store, replace(capability, version="1.0.1"), candidate_bundle)
    learned = store.approve(review.id, reviewer="upgrade-reviewer")

    state = store._load()
    capabilities = {
        item["version"]: item
        for item in state["capabilities"]
        if item["id"] == "hello"
    }
    assert state["records"][prior.id]["status"] == "superseded"
    assert state["records"][learned.id]["status"] == "learned"
    assert capabilities["1.0.0"]["status"] == "superseded"
    assert capabilities["1.0.1"]["status"] == "learned"
    assert store.current_learned("hello") == learned
    assert store.inventory().resolve("hello").capability.version == "1.0.1"
    assert plan_skill(skill_spec("1.0.1"), store.inventory()).disposition == Disposition.REUSE


def test_current_learned_rejects_multiple_active_records(
    store, capability, bundle, prior
):
    state = store._load()
    duplicate = dict(state["records"][prior.id])
    duplicate["id"] = "unexpected-second-active"
    state["records"][duplicate["id"]] = duplicate
    store._save(state)

    with pytest.raises(PromotionError, match="multiple active learned"):
        store.current_learned("hello")


def test_promoted_minimum_metric_retains_the_worst_scenario_for_reuse(
    store, capability, tmp_path
):
    evidence = build_bundle(
        tmp_path,
        label="minimum-summary",
        support_limit=0.9,
        support_mandatory=False,
        support_values={
            "nominal": 1.0,
            "entry": 0.5,
            "exit": 1.0,
            "stress": 1.0,
        },
    )
    review = pending(store, capability, evidence)
    store.approve(review.id, reviewer="metric-reviewer")

    learned = store.inventory().resolve("hello").capability
    assert learned.evaluation.metric_results["support"] == 0.5
    assert plan_skill(evidence.skill_spec, store.inventory()).disposition == Disposition.BLOCKED


def test_concurrent_improvement_approvals_leave_one_authoritative_learned_policy(
    store, capability, bundle, prior, tmp_path
):
    candidate_a = build_bundle(
        tmp_path,
        label="concurrent-v101",
        version="1.0.1",
        weight=0.2,
        baseline_policy=bundle.policy_path,
    )
    candidate_b = build_bundle(
        tmp_path,
        label="concurrent-v102",
        version="1.0.2",
        weight=0.3,
        baseline_policy=bundle.policy_path,
    )
    pending_a = pending(store, replace(capability, version="1.0.1"), candidate_a)
    pending_b = pending(store, replace(capability, version="1.0.2"), candidate_b)
    start = threading.Barrier(3)
    outcomes: list[object] = []

    def approve(record_id: str, reviewer: str) -> None:
        start.wait()
        try:
            outcomes.append(PromotionStore(store.root).approve(record_id, reviewer=reviewer))
        except PromotionError as error:
            outcomes.append(error)

    workers = [
        threading.Thread(target=approve, args=(pending_a.id, "a")),
        threading.Thread(target=approve, args=(pending_b.id, "b")),
    ]
    for worker in workers:
        worker.start()
    start.wait()
    for worker in workers:
        worker.join(timeout=5)
        assert not worker.is_alive()

    state = store._load()
    records = list(state["records"].values())
    capabilities = [item for item in state["capabilities"] if item["id"] == "hello"]

    assert len(outcomes) == 2
    assert sum(isinstance(outcome, PromotionError) for outcome in outcomes) == 1
    assert sum(record["status"] == "learned" for record in records) == 1
    assert sum(record["status"] == "superseded" for record in records) == 1
    assert sum(item["status"] == "learned" for item in capabilities) == 1
    assert sum(item["status"] == "superseded" for item in capabilities) == 1
    assert state["records"][prior.id]["status"] == "superseded"

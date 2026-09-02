"""Behavioural contracts for immutable human-review evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from mjlab_microduck.next_rl.evaluation import EvaluationReport, MetricResult, ScenarioResult, select_worst_case
from mjlab_microduck.next_rl.review import ReviewBundle, ReviewClip, ReviewError
from mjlab_microduck.next_rl.schema import ArtifactRef, MetricThreshold, PolicyContract, SkillSpec


@pytest.fixture
def passing_spec() -> SkillSpec:
    return SkillSpec(
        "hello", "1.0.0", "review contract", PolicyContract.microduck(),
        (MetricThreshold("falls", "count", "maximum", 0),), (2,), (1, 7, 8, 9, 10),
        held_out_scenarios=("nominal", "entry", "exit", "stress"),
    )


@pytest.fixture
def passing_report(tmp_path) -> EvaluationReport:
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
    return EvaluationReport("hello", "1.0.0", "eval-a", scenarios, True, ArtifactRef(str(policy), "onnx", digest))


def expected_scenarios(report: EvaluationReport) -> dict[str, ScenarioResult]:
    by_family = {scenario.family: scenario for scenario in report.scenarios}
    return {
        "nominal": by_family["nominal"], "entry": by_family["entry"],
        "exit": by_family["exit"], "stress": by_family["stress"],
        "worst_case": select_worst_case(report.scenarios),
    }


@pytest.fixture
def clip_files(tmp_path, passing_report) -> dict[str, ReviewClip]:
    clips: dict[str, ReviewClip] = {}
    for role, scenario in expected_scenarios(passing_report).items():
        path = tmp_path / f"{role}.mp4"
        path.write_bytes(f"{role} visual evidence".encode())
        clips[role] = ReviewClip(role, scenario.scenario_id, scenario.seed, path, passing_report.policy.sha256)
    return clips


@pytest.mark.parametrize("missing", ["nominal", "entry", "exit", "stress", "worst_case"])
def test_review_requires_every_mandatory_clip(missing, passing_spec, passing_report, clip_files):
    del clip_files[missing]
    with pytest.raises(ReviewError, match=missing):
        ReviewBundle.build(passing_report, clip_files, spec=passing_spec)


def test_every_clip_is_bound_to_the_evaluated_policy(passing_spec, passing_report, clip_files):
    clip_files["stress"].policy_digest = "b" * 64
    with pytest.raises(ReviewError, match="policy digest"):
        ReviewBundle.build(passing_report, clip_files, spec=passing_spec)


def test_every_clip_role_is_bound_to_its_deterministic_evaluated_scenario(passing_spec, passing_report, clip_files):
    clip_files["entry"].scenario_id = "nominal-1"
    clip_files["entry"].seed = 7
    with pytest.raises(ReviewError, match="entry.*scenario"):
        ReviewBundle.build(passing_report, clip_files, spec=passing_spec)


def test_bundle_verification_is_a_complete_boundary_for_direct_construction(passing_spec, passing_report, clip_files):
    bundle = ReviewBundle.build(passing_report, clip_files, spec=passing_spec)
    with pytest.raises(ReviewError, match="mandatory"):
        replace(bundle, clips=()).verify()
    with pytest.raises(ReviewError, match="exactly once"):
        replace(bundle, clips=(bundle.clips[0],) * 5).verify()
    with pytest.raises(ReviewError, match="policy path"):
        replace(bundle, policy_path=None).verify()


def test_bundle_verification_detects_a_clip_changed_after_review(passing_spec, passing_report, clip_files):
    bundle = ReviewBundle.build(passing_report, clip_files, spec=passing_spec)
    clip_files["entry"].path.write_bytes(b"replaced video")
    with pytest.raises(ReviewError, match="clip digest"):
        bundle.verify()


def test_bundle_recomputes_mandatory_threshold_semantics(passing_spec, passing_report, clip_files):
    bundle = ReviewBundle.build(passing_report, clip_files, spec=passing_spec)
    evidence = json.loads(bundle.evaluation_json)
    result = evidence["scenarios"][0]["threshold_results"][0]
    result.update({"value": 1.0, "passed": True, "normalized_violation": 0.0})
    forged = json.dumps(evidence, sort_keys=True, separators=(",", ":"))

    with pytest.raises(ReviewError, match="threshold"):
        replace(bundle, evaluation_json=forged, evaluation_digest=hashlib.sha256(forged.encode()).hexdigest()).verify()


def test_bundle_requires_complete_nonempty_threshold_contracts(passing_spec, passing_report, clip_files):
    bundle = ReviewBundle.build(passing_report, clip_files, spec=passing_spec)
    evidence = json.loads(bundle.evaluation_json)
    for scenario in evidence["scenarios"]:
        scenario["metrics"]["style"] = 0.0
        scenario["threshold_results"].append({
            "scenario_id": scenario["scenario_id"], "metric_name": "style", "unit": "score",
            "direction": "minimum", "limit": 1.0, "value": 0.0, "mandatory": False,
            "passed": False, "normalized_violation": 1.0,
        })
    complete = json.dumps(evidence, sort_keys=True, separators=(",", ":"))
    valid = replace(bundle, evaluation_json=complete, evaluation_digest=hashlib.sha256(complete.encode()).hexdigest(), metric_summary={"falls": 0.0, "style": 0.0})
    with pytest.raises(ReviewError, match="spec threshold"):
        valid.verify()



def test_bundle_records_canonical_evaluation_and_clip_evidence(passing_spec, passing_report, clip_files):
    bundle = ReviewBundle.build(passing_report, clip_files, spec=passing_spec)
    assert bundle.policy_digest == passing_report.policy.sha256
    assert bundle.evaluation_digest == hashlib.sha256(bundle.evaluation_json.encode("utf-8")).hexdigest()
    assert bundle.metric_summary == {"falls": 0.0}
    assert {clip.role for clip in bundle.clips} == set(clip_files)
    assert all(clip.digest == hashlib.sha256(clip.path.read_bytes()).hexdigest() for clip in bundle.clips)


def test_baseline_requires_identical_evaluation_identity_and_scenario_multiset(passing_spec, passing_report, clip_files):
    baseline = replace(passing_report, evaluator_revision="different-evaluator")
    with pytest.raises(ReviewError, match="identity"):
        ReviewBundle.build(passing_report, clip_files, baseline=baseline, spec=passing_spec)
    incomplete = replace(passing_report, scenarios=passing_report.scenarios[:-1])
    with pytest.raises(ReviewError, match="scenario multiset"):
        ReviewBundle.build(passing_report, clip_files, baseline=incomplete, spec=passing_spec)


def test_bundle_reverifies_retained_baseline_evidence(passing_spec, passing_report, clip_files, tmp_path):
    baseline_policy = tmp_path / "baseline.onnx"
    baseline_policy.write_bytes(b"baseline policy")
    baseline_digest = hashlib.sha256(baseline_policy.read_bytes()).hexdigest()
    baseline = replace(
        passing_report, policy=ArtifactRef(str(baseline_policy), "onnx", baseline_digest),
        scenarios=tuple(replace(scenario, policy_sha256=baseline_digest) for scenario in passing_report.scenarios),
    )
    bundle = ReviewBundle.build(passing_report, clip_files, baseline=baseline, spec=passing_spec)
    baseline_policy.unlink()
    with pytest.raises(ReviewError, match="baseline policy"):
        bundle.verify()


def test_failed_baseline_is_allowed_only_when_its_thresholds_agree(passing_spec, passing_report, clip_files):
    failed_scenarios = tuple(
        replace(
            scenario,
            metrics={"falls": 1.0},
            threshold_results=(replace(scenario.threshold_results[0], value=1.0, passed=False, normalized_violation=1.0),),
        )
        for scenario in passing_report.scenarios
    )
    failed = replace(passing_report, scenarios=failed_scenarios, passed=False)
    assert ReviewBundle.build(passing_report, clip_files, baseline=failed, spec=passing_spec).baseline is not None

    inconsistent = replace(failed, passed=True)
    with pytest.raises(ReviewError, match="threshold"):
        ReviewBundle.build(passing_report, clip_files, baseline=inconsistent, spec=passing_spec)


def test_multi_seed_family_clips_choose_the_stable_lowest_seed(passing_spec, passing_report, clip_files):
    original = passing_report.scenarios[0]
    extra = replace(
        original,
        scenario_id="nominal-0",
        seed=1,
        threshold_results=(replace(original.threshold_results[0], scenario_id="nominal-0"),),
    )
    report = replace(passing_report, scenarios=(*passing_report.scenarios, extra))
    path = clip_files["nominal"].path
    clip_files["nominal"] = ReviewClip("nominal", "nominal-0", 1, path, report.policy.sha256)

    assert ReviewBundle.build(report, clip_files, spec=passing_spec).clips[0].scenario_id == "nominal-0"


def test_spec_binding_rejects_uniformly_deleted_failed_metric_and_duplicate_name(passing_spec, passing_report, clip_files):
    bundle = ReviewBundle.build(passing_report, clip_files, spec=passing_spec)
    evidence = json.loads(bundle.evaluation_json)
    for scenario in evidence["scenarios"]:
        scenario["threshold_results"] = []
        scenario["metrics"] = {"diagnostic": 12.0}
    deleted = json.dumps(evidence, sort_keys=True, separators=(",", ":"))
    with pytest.raises(ReviewError, match="threshold"):
        replace(bundle, evaluation_json=deleted, evaluation_digest=hashlib.sha256(deleted.encode()).hexdigest()).verify()

    duplicate = json.loads(bundle.evaluation_json)
    duplicate["scenarios"][0]["threshold_results"].append({
        **duplicate["scenarios"][0]["threshold_results"][0], "limit": 99.0,
    })
    duplicate_json = json.dumps(duplicate, sort_keys=True, separators=(",", ":"))
    with pytest.raises(ReviewError, match="duplicate"):
        replace(bundle, evaluation_json=duplicate_json, evaluation_digest=hashlib.sha256(duplicate_json.encode()).hexdigest()).verify()

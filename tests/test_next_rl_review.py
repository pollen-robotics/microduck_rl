"""Behavioural contracts for immutable human-review evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest
from test_next_rl_support import (
    write_renderer_sidecar,
    write_test_video,
    write_tiny_policy,
)

import mjlab_microduck.next_rl.review as review_module
from mjlab_microduck.next_rl.artifacts import canonical_json
from mjlab_microduck.next_rl.evaluation import (
    EvaluationReport,
    MetricResult,
    ScenarioResult,
    select_worst_case,
)
from mjlab_microduck.next_rl.review import RendererEvidence, ReviewBundle, ReviewError
from mjlab_microduck.next_rl.schema import (
    ArtifactRef,
    MetricThreshold,
    PolicyContract,
    SkillSpec,
)


@pytest.fixture
def passing_spec() -> SkillSpec:
    return SkillSpec(
        "hello", "1.0.0", "review contract", PolicyContract.microduck(),
        (MetricThreshold("falls", "count", "maximum", 0),), (2,), (1, 7, 8, 9, 10),
        held_out_scenarios=("nominal", "entry", "exit", "stress"),
    )


@pytest.fixture
def passing_report(tmp_path) -> EvaluationReport:
    policy = write_tiny_policy(tmp_path / "policy.onnx")
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


def renderer_evidence(
    tmp_path,
    report: EvaluationReport,
    *,
    prefix: str = "",
) -> dict[str, RendererEvidence]:
    evaluation_digest = hashlib.sha256(canonical_json(report.as_dict()).encode()).hexdigest()
    clips: dict[str, RendererEvidence] = {}
    for role, scenario in expected_scenarios(report).items():
        stem = f"{prefix}-{role}" if prefix else role
        video = write_test_video(tmp_path / f"{stem}.mp4")
        sidecar = write_renderer_sidecar(
            tmp_path / f"{stem}.render.json",
            role=role,
            scenario_id=scenario.scenario_id,
            seed=scenario.seed,
            policy_sha256=report.policy.sha256,
            evaluation_digest=evaluation_digest,
            video_path=video,
        )
        clips[role] = RendererEvidence.load(sidecar)
    return clips


@pytest.fixture
def clip_files(tmp_path, passing_report) -> dict[str, RendererEvidence]:
    return renderer_evidence(tmp_path, passing_report)


@pytest.mark.parametrize("missing", ["nominal", "entry", "exit", "stress", "worst_case"])
def test_review_requires_every_mandatory_clip(missing, passing_spec, passing_report, clip_files):
    del clip_files[missing]
    with pytest.raises(ReviewError, match=missing):
        ReviewBundle.build(passing_report, clip_files, spec=passing_spec)


def test_every_clip_is_bound_to_the_evaluated_policy(
    tmp_path,
    passing_spec,
    passing_report,
    clip_files,
):
    original = clip_files["stress"]
    sidecar = write_renderer_sidecar(
        tmp_path / "wrong-policy.render.json",
        role=original.role,
        scenario_id=original.scenario_id,
        seed=original.seed,
        policy_sha256="b" * 64,
        evaluation_digest=original.evaluation_digest,
        video_path=original.video_path,
    )
    clip_files["stress"] = RendererEvidence.load(sidecar)
    with pytest.raises(ReviewError, match="policy digest"):
        ReviewBundle.build(passing_report, clip_files, spec=passing_spec)


def test_every_clip_role_is_bound_to_its_deterministic_evaluated_scenario(
    tmp_path,
    passing_spec,
    passing_report,
    clip_files,
):
    original = clip_files["entry"]
    sidecar = write_renderer_sidecar(
        tmp_path / "wrong-scenario.render.json",
        role="entry",
        scenario_id="nominal-1",
        seed=7,
        policy_sha256=original.policy_sha256,
        evaluation_digest=original.evaluation_digest,
        video_path=original.video_path,
    )
    clip_files["entry"] = RendererEvidence.load(sidecar)
    with pytest.raises(ReviewError, match="entry.*scenario"):
        ReviewBundle.build(passing_report, clip_files, spec=passing_spec)


def test_every_clip_is_bound_to_the_exact_evaluation(
    tmp_path,
    passing_spec,
    passing_report,
    clip_files,
):
    original = clip_files["exit"]
    sidecar = write_renderer_sidecar(
        tmp_path / "wrong-evaluation.render.json",
        role=original.role,
        scenario_id=original.scenario_id,
        seed=original.seed,
        policy_sha256=original.policy_sha256,
        evaluation_digest="e" * 64,
        video_path=original.video_path,
    )
    clip_files["exit"] = RendererEvidence.load(sidecar)

    with pytest.raises(ReviewError, match="evaluation digest"):
        ReviewBundle.build(passing_report, clip_files, spec=passing_spec)


def test_plain_bytes_named_onnx_fail_review_before_any_bundle_is_built(
    tmp_path,
    passing_spec,
    passing_report,
    clip_files,
):
    policy = tmp_path / "fake.onnx"
    policy.write_bytes(b"not an ONNX graph")
    digest = hashlib.sha256(policy.read_bytes()).hexdigest()
    report = replace(
        passing_report,
        policy=ArtifactRef(str(policy), "onnx", digest),
        scenarios=tuple(
            replace(scenario, policy_sha256=digest)
            for scenario in passing_report.scenarios
        ),
    )
    with pytest.raises(ReviewError, match="ONNX preflight"):
        ReviewBundle.build(report, clip_files, spec=passing_spec)


def test_deserialized_review_bundle_preflights_the_bound_onnx(
    passing_spec,
    passing_report,
    clip_files,
):
    bundle = ReviewBundle.build(passing_report, clip_files, spec=passing_spec)
    bundle.policy_path.write_bytes(b"not an ONNX graph")

    with pytest.raises(ReviewError, match="ONNX preflight"):
        ReviewBundle.from_dict(bundle.as_dict())


def test_plain_text_named_mp4_fails_review_video_decode(
    tmp_path,
    passing_spec,
    passing_report,
    clip_files,
):
    original = clip_files["nominal"]
    video = tmp_path / "text.mp4"
    video.write_text("not a video", encoding="utf-8")
    sidecar = write_renderer_sidecar(
        tmp_path / "text.render.json",
        role=original.role,
        scenario_id=original.scenario_id,
        seed=original.seed,
        policy_sha256=original.policy_sha256,
        evaluation_digest=original.evaluation_digest,
        video_path=video,
    )

    with pytest.raises(ReviewError, match="decode|video"):
        RendererEvidence.load(sidecar)


def test_renderer_evidence_helper_writes_a_create_once_bound_sidecar(tmp_path):
    video = write_test_video(tmp_path / "nominal.mp4")
    sidecar = tmp_path / "nominal.render.json"

    evidence = review_module.write_renderer_evidence(
        sidecar,
        role="nominal",
        scenario_id="nominal-1",
        seed=7,
        policy_sha256="a" * 64,
        evaluation_digest="b" * 64,
        video_path=video,
        renderer_revision="renderer-v1",
    )

    assert evidence.video_sha256 == hashlib.sha256(video.read_bytes()).hexdigest()
    assert json.loads(sidecar.read_text()) == evidence.contract_dict()
    with pytest.raises(FileExistsError, match="immutable"):
        review_module.write_renderer_evidence(
            sidecar,
            role="nominal",
            scenario_id="nominal-1",
            seed=7,
            policy_sha256="a" * 64,
            evaluation_digest="b" * 64,
            video_path=video,
            renderer_revision="renderer-v1",
        )


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
    clip_files["entry"].video_path.write_bytes(b"replaced video")
    with pytest.raises(ReviewError, match="clip digest"):
        bundle.verify()


def test_bundle_verification_detects_renderer_sidecar_tampering(
    passing_spec,
    passing_report,
    clip_files,
):
    bundle = ReviewBundle.build(passing_report, clip_files, spec=passing_spec)
    sidecar = clip_files["stress"].evidence_path
    raw = json.loads(sidecar.read_text(encoding="utf-8"))
    raw["renderer_revision"] = "tampered-renderer"
    sidecar.write_text(canonical_json(raw), encoding="utf-8")

    with pytest.raises(ReviewError, match="sidecar binding|sidecar digest"):
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


def test_metric_summary_keeps_the_direction_aware_worst_scenario(
    tmp_path,
    passing_spec,
    passing_report,
):
    minimum = MetricThreshold("support", "ratio", "minimum", 0.5)
    spec = replace(passing_spec, metrics=(*passing_spec.metrics, minimum))
    scenarios = []
    for index, scenario in enumerate(passing_report.scenarios):
        support = 0.5 if index == 0 else 1.0
        scenarios.append(
            replace(
                scenario,
                metrics={**scenario.metrics, "support": support},
                threshold_results=(
                    *scenario.threshold_results,
                    MetricResult(
                        scenario.scenario_id,
                        "support",
                        "ratio",
                        "minimum",
                        0.5,
                        support,
                        True,
                        True,
                        0.0,
                    ),
                ),
            )
        )
    report = replace(passing_report, scenarios=tuple(scenarios))
    clip_files = renderer_evidence(tmp_path, report, prefix="metric")

    bundle = ReviewBundle.build(report, clip_files, spec=spec)

    assert bundle.metric_summary == {"falls": 0.0, "support": 0.5}


def test_baseline_requires_identical_evaluation_identity_and_scenario_multiset(passing_spec, passing_report, clip_files):
    baseline = replace(passing_report, evaluator_revision="different-evaluator")
    with pytest.raises(ReviewError, match="identity"):
        ReviewBundle.build(passing_report, clip_files, baseline=baseline, spec=passing_spec)
    incomplete = replace(passing_report, scenarios=passing_report.scenarios[:-1])
    with pytest.raises(ReviewError, match="scenario multiset"):
        ReviewBundle.build(passing_report, clip_files, baseline=incomplete, spec=passing_spec)


def test_bundle_reverifies_retained_baseline_evidence(passing_spec, passing_report, clip_files, tmp_path):
    baseline_policy = write_tiny_policy(tmp_path / "baseline.onnx", weight=0.2)
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


def test_multi_seed_family_clips_choose_the_stable_lowest_seed(
    tmp_path,
    passing_spec,
    passing_report,
):
    original = passing_report.scenarios[0]
    extra = replace(
        original,
        scenario_id="nominal-0",
        seed=1,
        threshold_results=(replace(original.threshold_results[0], scenario_id="nominal-0"),),
    )
    report = replace(passing_report, scenarios=(*passing_report.scenarios, extra))
    clip_files = renderer_evidence(tmp_path, report, prefix="multi-seed")

    assert ReviewBundle.build(report, clip_files, spec=passing_spec).clips[0].scenario_id == "nominal-0"


def test_spec_binding_rejects_uniformly_deleted_failed_metric_and_duplicate_name(
    tmp_path,
    passing_spec,
    passing_report,
):
    two_metric_spec = replace(passing_spec, metrics=(
        MetricThreshold("falls", "count", "maximum", 0),
        MetricThreshold("style", "score", "minimum", 1, mandatory=False),
    ))
    report = replace(passing_report, scenarios=tuple(
        replace(
            scenario,
            metrics={"falls": 0.0, "style": 0.0},
            threshold_results=(
                scenario.threshold_results[0],
                MetricResult(scenario.scenario_id, "style", "score", "minimum", 1, 0, False, False, 1),
            ),
        )
        for scenario in passing_report.scenarios
    ))
    clip_files = renderer_evidence(tmp_path, report, prefix="two-metric")
    bundle = ReviewBundle.build(report, clip_files, spec=two_metric_spec)
    evidence = json.loads(bundle.evaluation_json)
    for scenario in evidence["scenarios"]:
        scenario["threshold_results"] = [scenario["threshold_results"][0]]
        scenario["metrics"] = {"falls": 0.0, "diagnostic": 12.0}
    deleted = json.dumps(evidence, sort_keys=True, separators=(",", ":"))
    with pytest.raises(ReviewError, match="spec threshold"):
        replace(bundle, evaluation_json=deleted, evaluation_digest=hashlib.sha256(deleted.encode()).hexdigest(), metric_summary={"falls": 0.0}).verify()

    duplicate = json.loads(bundle.evaluation_json)
    duplicate["scenarios"][0]["threshold_results"].append({
        **duplicate["scenarios"][0]["threshold_results"][0], "limit": 99.0,
    })
    duplicate_json = json.dumps(duplicate, sort_keys=True, separators=(",", ":"))
    with pytest.raises(ReviewError, match="duplicate"):
        replace(bundle, evaluation_json=duplicate_json, evaluation_digest=hashlib.sha256(duplicate_json.encode()).hexdigest()).verify()

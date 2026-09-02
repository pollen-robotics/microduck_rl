"""Behavioural contracts for immutable human-review evidence."""

from __future__ import annotations

import hashlib

import pytest

from mjlab_microduck.next_rl.evaluation import EvaluationReport, MetricResult, ScenarioResult
from mjlab_microduck.next_rl.review import ReviewBundle, ReviewClip, ReviewError
from mjlab_microduck.next_rl.schema import ArtifactRef


POLICY_DIGEST = "a" * 64


@pytest.fixture
def passing_report(tmp_path) -> EvaluationReport:
    policy = tmp_path / "policy.onnx"
    policy.write_bytes(b"evaluated policy")
    digest = hashlib.sha256(policy.read_bytes()).hexdigest()
    result = MetricResult("nominal-1", "falls", "count", "maximum", 0.0, 0.0, True, True, 0.0)
    scenario = ScenarioResult("nominal-1", "nominal", 7, {"falls": 0.0}, digest, (result,))
    return EvaluationReport("hello", "1.0.0", "eval-a", (scenario,), True, ArtifactRef(str(policy), "onnx", digest))


@pytest.fixture
def clip_files(tmp_path, passing_report) -> dict[str, ReviewClip]:
    clips: dict[str, ReviewClip] = {}
    for role in ("nominal", "entry", "exit", "stress", "worst_case"):
        path = tmp_path / f"{role}.mp4"
        path.write_bytes(f"{role} visual evidence".encode())
        clips[role] = ReviewClip(role, f"{role}-scenario", 7, path, passing_report.policy.sha256)
    return clips


@pytest.mark.parametrize("missing", ["nominal", "entry", "exit", "stress", "worst_case"])
def test_review_requires_every_mandatory_clip(missing, passing_report, clip_files):
    del clip_files[missing]

    with pytest.raises(ReviewError, match=missing):
        ReviewBundle.build(passing_report, clip_files)


def test_every_clip_is_bound_to_the_evaluated_policy(passing_report, clip_files):
    clip_files["stress"].policy_digest = "b" * 64

    with pytest.raises(ReviewError, match="policy digest"):
        ReviewBundle.build(passing_report, clip_files)


def test_bundle_verification_detects_a_clip_changed_after_review(passing_report, clip_files):
    bundle = ReviewBundle.build(passing_report, clip_files)
    clip_files["entry"].path.write_bytes(b"replaced video")

    with pytest.raises(ReviewError, match="clip digest"):
        bundle.verify()


def test_bundle_records_canonical_evaluation_and_clip_evidence(passing_report, clip_files):
    bundle = ReviewBundle.build(passing_report, clip_files)

    assert bundle.policy_digest == passing_report.policy.sha256
    assert bundle.evaluation_digest == hashlib.sha256(
        bundle.evaluation_json.encode("utf-8")
    ).hexdigest()
    assert bundle.metric_summary == {"falls": 0.0}
    assert {clip.role for clip in bundle.clips} == set(clip_files)
    assert all(clip.digest == hashlib.sha256(clip.path.read_bytes()).hexdigest() for clip in bundle.clips)


def test_baseline_comparison_requires_seed_matched_scenarios(passing_report, clip_files):
    baseline = EvaluationReport(
        passing_report.skill_id,
        passing_report.spec_version,
        passing_report.evaluator_revision,
        (ScenarioResult("other", "nominal", 99, {"falls": 0.0}, passing_report.policy.sha256),),
        True,
        passing_report.policy,
    )

    with pytest.raises(ReviewError, match="seed-matched"):
        ReviewBundle.build(passing_report, clip_files, baseline=baseline)

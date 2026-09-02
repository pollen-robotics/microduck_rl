"""Contracts for the dependency-light Next RL workspace schemas."""

from __future__ import annotations

import json

import pytest

from mjlab_microduck.next_rl.artifacts import (
    atomic_write_json,
    canonical_json,
    sha256_file,
)
from mjlab_microduck.next_rl.schema import (
    ArtifactRef,
    Capability,
    EvaluationRef,
    ExperimentManifest,
    MetricThreshold,
    PolicyContract,
    SchemaError,
    SkillSpec,
)


def metric(name: str) -> dict[str, object]:
    return {"name": name, "unit": "count", "direction": "maximum", "limit": 0}


def skill_dict(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "id": "standing",
        "version": "1.0.0",
        "description": "Remain upright from ordinary standing.",
        "contract": PolicyContract.microduck().as_dict(),
        "metrics": [metric("falls")],
        "training_seeds": [1, 2],
        "evaluation_seeds": [3, 4],
    }
    value.update(overrides)
    return value


def evaluation_dict(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "kind": "evaluation_report",
        "policy_sha256": "a" * 64,
        "report_path": "evaluation.json",
        "passed": True,
        "metric_results": {"falls": 0},
        "approval_provenance": "review-42.json",
    }
    value.update(overrides)
    return value


def learned_capability_dict(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "id": "standing",
        "version": "1.0.0",
        "aliases": ["stand"],
        "robot_model": "microduck",
        "contract": PolicyContract.microduck().as_dict(),
        "status": "learned",
        "policy": {"path": "policy.onnx", "kind": "onnx", "sha256": "a" * 64},
        "evaluation": evaluation_dict(),
    }
    value.update(overrides)
    return value


def experiment_dict(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "skill_id": "standing",
        "spec_version": "1.0.0",
        "task_id": "Mjlab-Standing-Flat-MicroDuck",
        "contract": PolicyContract.microduck().as_dict(),
        "code_digest": "a" * 64,
        "seed": 1,
        "runner_id": "local",
        "status": "planned",
    }
    value.update(overrides)
    return value


def test_default_contract_is_the_runtime_contract():
    assert PolicyContract.microduck().as_dict() == {
        "model_api": 1,
        "obs_len": 61,
        "action_len": 14,
        "robot_model": "microduck",
        "control_hz": 50,
    }


def test_threshold_requires_units_and_known_direction():
    with pytest.raises(SchemaError, match="direction"):
        MetricThreshold.from_dict(
            {"name": "falls", "unit": "count", "direction": "equal", "limit": 0}
        )


def test_skill_rejects_duplicate_metric_names():
    raw = skill_dict(metrics=[metric("falls"), metric("falls")])
    with pytest.raises(SchemaError, match="duplicate metric"):
        SkillSpec.from_dict(raw)


def test_training_and_eval_seeds_must_be_disjoint():
    raw = skill_dict(training_seeds=[1, 2], evaluation_seeds=[2, 3])
    with pytest.raises(SchemaError, match="overlap"):
        SkillSpec.from_dict(raw)


def test_schema_rejects_unknown_fields_outside_metadata():
    raw = skill_dict(unrecognized=True)
    with pytest.raises(SchemaError, match="unknown field"):
        SkillSpec.from_dict(raw)


def test_schema_allows_forward_compatible_metadata_mapping():
    spec = SkillSpec.from_dict(skill_dict(metadata={"owner": "training"}))
    assert spec.as_dict()["metadata"] == {"owner": "training"}


def test_schema_metadata_is_immutable():
    spec = SkillSpec.from_dict(skill_dict(metadata={"provenance": {"owner": "training"}}))
    with pytest.raises(TypeError):
        spec.metadata["provenance"]["owner"] = "changed"


def test_threshold_rejects_nonfinite_limit():
    with pytest.raises(SchemaError, match="finite"):
        MetricThreshold.from_dict(
            {"name": "falls", "unit": "count", "direction": "maximum", "limit": float("nan")}
        )


def test_legacy_evaluation_has_provenance_but_no_metric_evidence():
    evaluation = EvaluationRef.from_dict(
        {
            "kind": "legacy_runtime_shipped",
            "policy_sha256": "a" * 64,
            "runtime_repository": "pollen-robotics/microduck",
            "runtime_commit": "abc123",
            "approval_provenance": "README.md",
        }
    )
    assert evaluation.has_metric_evidence("falls") is False


def test_strict_evaluation_requires_a_passed_report_and_metric_results():
    with pytest.raises(SchemaError, match="metric_results"):
        EvaluationRef.from_dict(
            {
                "kind": "evaluation_report",
                "policy_sha256": "a" * 64,
                "report_path": "evaluation.json",
                "passed": True,
            }
        )


def test_learned_capability_rejects_a_failed_evaluation():
    raw = learned_capability_dict(evaluation=evaluation_dict(passed=False))
    with pytest.raises(SchemaError, match="passing evaluation"):
        Capability.from_dict(raw)


def test_learned_capability_rejects_legacy_only_evidence():
    raw = learned_capability_dict(
        evaluation={
            "kind": "legacy_runtime_shipped",
            "policy_sha256": "a" * 64,
            "runtime_repository": "pollen-robotics/microduck",
            "runtime_commit": "abc123",
            "approval_provenance": "README.md",
        }
    )
    with pytest.raises(SchemaError, match="evaluation_report"):
        Capability.from_dict(raw)


def test_learned_capability_rejects_an_unapproved_evaluation():
    raw = learned_capability_dict(evaluation=evaluation_dict(approval_provenance=""))
    with pytest.raises(SchemaError, match="approval"):
        Capability.from_dict(raw)


def test_normal_evaluation_rejects_legacy_only_fields():
    raw = evaluation_dict(runtime_repository="pollen-robotics/microduck")
    with pytest.raises(SchemaError, match="unknown field"):
        EvaluationRef.from_dict(raw)


@pytest.mark.parametrize("metadata", [{"tags": {"unsafe"}}, {"limit": float("nan")}, {"value": object()}])
def test_metadata_rejects_non_json_values(metadata):
    with pytest.raises(SchemaError, match="metadata"):
        SkillSpec.from_dict(skill_dict(metadata=metadata))


@pytest.mark.parametrize("field", ["created_at", "output_dir"])
def test_experiment_rejects_empty_optional_text(field):
    with pytest.raises(SchemaError, match=field):
        ExperimentManifest.from_dict(experiment_dict(**{field: ""}))


def test_artifact_requires_a_sha256_digest():
    with pytest.raises(SchemaError, match="sha256"):
        ArtifactRef.from_dict({"path": "policy.onnx", "kind": "onnx", "sha256": "short"})


def test_canonical_json_is_order_independent():
    assert canonical_json({"b": 2, "a": 1}) == canonical_json({"a": 1, "b": 2})


def test_atomic_write_json_round_trips(tmp_path):
    path = tmp_path / "manifest.json"
    atomic_write_json(path, {"b": 2, "a": 1})
    assert json.loads(path.read_text()) == {"a": 1, "b": 2}


def test_sha256_file_detects_changed_content(tmp_path):
    path = tmp_path / "policy.onnx"
    path.write_bytes(b"one")
    first = sha256_file(path)
    path.write_bytes(b"two")
    assert sha256_file(path) != first

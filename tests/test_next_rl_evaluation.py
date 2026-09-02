"""Behavioural contracts for held-out Next RL evaluation gates."""

from __future__ import annotations

import json
import threading

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

import mjlab_microduck.next_rl.evaluation as evaluation_module
from mjlab_microduck.next_rl.artifacts import sha256_file
from mjlab_microduck.next_rl.evaluation import (
    ArtifactIntegrityError,
    EvaluationError,
    ScenarioResult,
    build_report,
    evaluate_thresholds,
    preflight_onnx,
    select_worst_case,
)
from mjlab_microduck.next_rl.schema import MetricThreshold, PolicyContract, SkillSpec


def maximum(name: str, limit: float) -> MetricThreshold:
    return MetricThreshold(name, "count", "maximum", limit)


def minimum(name: str, limit: float) -> MetricThreshold:
    return MetricThreshold(name, "count", "minimum", limit)


def spec_with(*metrics: MetricThreshold, **overrides: object) -> SkillSpec:
    values: dict[str, object] = {
        "id": "standing",
        "version": "1.2.3",
        "description": "Remain upright.",
        "contract": PolicyContract.microduck(),
        "metrics": metrics,
        "training_seeds": (1, 2),
        "evaluation_seeds": (3, 4),
    }
    values.update(overrides)
    return SkillSpec(**values)


def scenario(
    scenario_id: str = "nominal-01",
    *,
    policy_sha256: str = "a" * 64,
    **metrics: float,
) -> ScenarioResult:
    return ScenarioResult(scenario_id, "nominal", 3, metrics, policy_sha256=policy_sha256)


def test_mandatory_safety_failure_fails_the_report():
    report = evaluate_thresholds(spec_with(maximum("falls", 0)), [scenario(falls=1)])

    assert report.passed is False


def test_positive_metrics_cannot_outvote_safety():
    report = evaluate_thresholds(
        spec_with(maximum("falls", 0), minimum("cycles", 3)), [scenario(falls=1, cycles=10)]
    )

    assert report.passed is False


def test_worst_case_selection_is_deterministic():
    spec = spec_with(maximum("falls", 0))
    report = evaluate_thresholds(
        spec,
        [scenario("stress-10", falls=1), scenario("stress-02", falls=1)],
    )

    assert select_worst_case(tuple(reversed(report.scenarios))).scenario_id == "stress-02"


def test_training_and_evaluation_seeds_cannot_overlap():
    overlapping = spec_with(training_seeds=(1, 2), evaluation_seeds=(2, 3))

    with pytest.raises(EvaluationError, match="overlap"):
        evaluate_thresholds(overlapping, [])


def test_every_required_scenario_family_must_be_present():
    spec = spec_with(maximum("falls", 0), held_out_scenarios=("nominal", "stress"))

    with pytest.raises(EvaluationError, match="stress"):
        evaluate_thresholds(spec, [scenario(falls=0)])


def test_each_scenario_records_finite_unit_labelled_threshold_results():
    report = evaluate_thresholds(spec_with(minimum("cycles", 3)), [scenario(cycles=4)])

    result = report.scenarios[0].threshold_results[0]
    assert (result.unit, result.value, result.passed) == ("count", 4.0, True)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_scenario_metrics_are_refused(value: float):
    with pytest.raises(EvaluationError, match="finite"):
        evaluate_thresholds(spec_with(maximum("falls", 0)), [scenario(falls=value)])


def test_nonmandatory_failure_does_not_fail_the_aggregate_report():
    report = evaluate_thresholds(
        spec_with(MetricThreshold("style", "score", "minimum", 3, mandatory=False)),
        [scenario(style=0)],
    )

    assert report.passed is True


def _tiny_policy(path, weight: float = 0.1):
    weights = numpy_helper.from_array(np.full((61, 14), weight, dtype=np.float32), "weights")
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["obs", "weights"], ["actions"])],
        "tiny-policy",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, 61])],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, 14])],
        initializer=[weights],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, path)
    return path


@pytest.fixture
def tiny_onnx(tmp_path):
    return _tiny_policy(tmp_path / "policy.onnx")


@pytest.fixture
def nonfinite_onnx(tmp_path):
    value = numpy_helper.from_array(np.full((1, 14), np.nan, dtype=np.float32), "value")
    graph = helper.make_graph(
        [helper.make_node("Constant", [], ["actions"], value=value)],
        "nonfinite-policy",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, 61])],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, 14])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    path = tmp_path / "nonfinite.onnx"
    onnx.save(model, path)
    return path


def test_report_is_bound_to_exact_policy_bytes(tiny_onnx, tmp_path):
    report = build_report(
        policy=tiny_onnx,
        spec=spec_with(maximum("falls", 0)),
        scenarios=[scenario(policy_sha256=sha256_file(tiny_onnx), falls=0)],
        evaluator_revision="eval-abc123",
        report_path=tmp_path / "evaluation.json",
    )
    tiny_onnx.write_bytes(tiny_onnx.read_bytes() + b"changed")

    with pytest.raises(ArtifactIntegrityError):
        report.verify_policy(tiny_onnx)


def test_nonfinite_onnx_output_fails_preflight(nonfinite_onnx):
    with pytest.raises(EvaluationError, match="non-finite"):
        preflight_onnx(nonfinite_onnx)


def test_report_persistence_is_canonical_and_immutable(tiny_onnx, tmp_path):
    destination = tmp_path / "evaluation.json"
    report = build_report(
        policy=tiny_onnx,
        spec=spec_with(maximum("falls", 0)),
        scenarios=[scenario(policy_sha256=sha256_file(tiny_onnx), falls=0)],
        evaluator_revision="eval-abc123",
        report_path=destination,
    )

    assert json.loads(destination.read_text())["policy"]["sha256"] == report.policy.sha256
    assert destination.read_text().startswith('{"evaluator_revision"')
    with pytest.raises(FileExistsError, match="immutable"):
        report.write(destination)


def test_empty_scenario_collection_is_refused():
    with pytest.raises(EvaluationError, match="at least one scenario"):
        evaluate_thresholds(spec_with(maximum("falls", 0)), [])


def test_policy_a_scenario_evidence_cannot_be_reported_for_policy_b(tmp_path):
    policy_a = _tiny_policy(tmp_path / "policy-a.onnx", weight=0.1)
    policy_b = _tiny_policy(tmp_path / "policy-b.onnx", weight=0.2)

    with pytest.raises(ArtifactIntegrityError, match="scenario.*policy"):
        build_report(
            policy=policy_b,
            spec=spec_with(maximum("falls", 0)),
            scenarios=[scenario(policy_sha256=sha256_file(policy_a), falls=0)],
            evaluator_revision="eval-abc123",
        )


def test_preflight_refuses_a_source_changed_after_its_snapshot_is_checked(tiny_onnx, monkeypatch):
    real_smoke = evaluation_module.smoke_run_onnx

    def mutate_source_after_snapshot(snapshot):
        real_smoke(snapshot)
        tiny_onnx.write_bytes(tiny_onnx.read_bytes() + b"changed")

    monkeypatch.setattr(evaluation_module, "smoke_run_onnx", mutate_source_after_snapshot)

    with pytest.raises(ArtifactIntegrityError, match="changed while preflighting"):
        preflight_onnx(tiny_onnx)


def test_concurrent_report_creation_allows_one_writer_and_preserves_existing_bytes(tmp_path):
    report = evaluate_thresholds(spec_with(maximum("falls", 0)), [scenario(falls=0)])
    destination = tmp_path / "evaluation.json"
    start = threading.Barrier(3)
    outcomes: list[str] = []

    def write_once() -> None:
        start.wait()
        try:
            report.write(destination)
        except FileExistsError:
            outcomes.append("exists")
        else:
            outcomes.append("created")

    writers = [threading.Thread(target=write_once) for _ in range(2)]
    for writer in writers:
        writer.start()
    start.wait()
    for writer in writers:
        writer.join(timeout=5)
        assert not writer.is_alive()

    assert sorted(outcomes) == ["created", "exists"]
    assert destination.read_text() == evaluation_module.canonical_json(report.as_dict())

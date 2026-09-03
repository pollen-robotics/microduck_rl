"""Behavioral contracts for the guarded Next RL operator CLI."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
import tomllib

from mjlab_microduck.next_rl.artifacts import canonical_json
from mjlab_microduck.next_rl.cli import CliDependencies, main
from mjlab_microduck.next_rl.evaluation import EvaluationReport, MetricResult, ScenarioResult, select_worst_case
from mjlab_microduck.next_rl.review import ReviewClip
from mjlab_microduck.next_rl.schema import ArtifactRef, Capability, MetricThreshold, PolicyContract, SkillSpec


def test_next_rl_is_a_declared_script():
    """Catch a distributable CLI that is not reachable from the installed command."""
    scripts = tomllib.loads(Path("pyproject.toml").read_text())["project"]["scripts"]

    assert scripts["next-rl"] == "mjlab_microduck.next_rl.cli:main"


def _write_skill(path: Path, skill_id: str, **overrides: object) -> Path:
    value: dict[str, object] = {
        "id": skill_id,
        "version": "1.0.0",
        "description": "A test skill.",
        "contract": PolicyContract.microduck().as_dict(),
        "metrics": [{"name": "falls", "unit": "count", "direction": "maximum", "limit": 0}],
        "training_seeds": [1],
        "evaluation_seeds": [2],
    }
    value.update(overrides)
    path.write_text(json.dumps(value))
    return path


def test_plan_existing_skill_does_not_construct_training_command(tmp_path: Path, monkeypatch, capsys):
    """Catch planning that builds a train command for an already-known skill."""
    monkeypatch.setenv("NEXT_RL_HOME", str(tmp_path / "state"))
    skill = _write_skill(tmp_path / "walking.json", "walking")

    rc = main(["plan", str(skill)])

    output = json.loads(capsys.readouterr().out)
    assert rc == 2
    assert output["disposition"] in {"reuse", "blocked"}
    assert "train_argv" not in output


@dataclass
class _FakeRunner:
    prepare_calls: list[object] = field(default_factory=list)
    start_calls: list[object] = field(default_factory=list)

    def prepare(self, manifest: object) -> object:
        self.prepare_calls.append(manifest)
        return type("Prepared", (), {"fingerprint": "a" * 64})()

    def start(self, prepared: object) -> None:
        self.start_calls.append(prepared)

    def status(self, fingerprint: str) -> dict[str, object]:
        return {"status": "pending", "fingerprint": fingerprint, "token": "must-not-leak"}


def test_prepare_is_dry_run_by_default(tmp_path: Path, monkeypatch, capsys):
    """Catch a preparation workflow that starts a training job."""
    runner = _FakeRunner()
    monkeypatch.setenv("NEXT_RL_HOME", str(tmp_path / "state"))
    skill = _write_skill(tmp_path / "new.json", "new-skill")

    rc = main(
        ["prepare", str(skill)],
        dependencies=CliDependencies(runner_factory=lambda home: runner),
    )

    assert rc == 0
    assert json.loads(capsys.readouterr().out)["status"] == "planned"
    assert runner.prepare_calls
    assert runner.start_calls == []


def test_prepare_existing_skill_stops_before_runner_preparation(tmp_path: Path, monkeypatch, capsys):
    """Catch a reuse or blocked plan that still prepares a training bundle."""
    runner = _FakeRunner()
    monkeypatch.setenv("NEXT_RL_HOME", str(tmp_path / "state"))
    skill = _write_skill(tmp_path / "walking.json", "walking")

    rc = main(
        ["prepare", str(skill)],
        dependencies=CliDependencies(runner_factory=lambda home: runner),
    )

    assert rc == 2
    assert json.loads(capsys.readouterr().out)["disposition"] == "blocked"
    assert runner.prepare_calls == []
    assert runner.start_calls == []


def test_start_is_not_an_operator_command(capsys):
    """Catch accidental exposure of a training-launch command."""
    assert main(["start"]) == 2
    assert json.loads(capsys.readouterr().out) == {"error": "invalid_request"}


def test_status_uses_injected_runner_and_redacts_transport_data(tmp_path: Path, monkeypatch, capsys):
    """Catch status output that either skips the runner or leaks transport state."""
    runner = _FakeRunner()
    monkeypatch.setenv("NEXT_RL_HOME", str(tmp_path / "state"))
    fingerprint = "b" * 64

    rc = main(
        ["status", fingerprint],
        dependencies=CliDependencies(runner_factory=lambda home: runner),
    )

    output = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert output == {"fingerprint": fingerprint, "status": "pending"}


def test_status_allowlists_only_validated_checkpoint_primitives(tmp_path: Path, monkeypatch, capsys):
    """Catch nested runner data or malformed checkpoint fields reaching JSON output."""
    runner = _FakeRunner()
    runner.status = lambda fingerprint: {
        "status": "running",
        "stderr": "token=leak",
        "last_stable_checkpoint": {
            "name": "model_100.pt",
            "sha256": "c" * 64,
            "size": 12,
            "mtime_ns": 34,
            "relative_path": "source/logs/run/model_100.pt",
            "nested": {"token": "leak"},
        },
        "metadata": {"authorization": "leak"},
    }
    monkeypatch.setenv("NEXT_RL_HOME", str(tmp_path / "state"))

    assert main(["status", "c" * 64], dependencies=CliDependencies(runner_factory=lambda home: runner)) == 0

    assert json.loads(capsys.readouterr().out) == {
        "fingerprint": "c" * 64,
        "last_stable_checkpoint": {
            "mtime_ns": 34,
            "name": "model_100.pt",
            "sha256": "c" * 64,
            "size": 12,
        },
        "status": "running",
    }


def test_status_drops_malformed_checkpoint_values(tmp_path: Path, monkeypatch, capsys):
    """Catch a checkpoint field that bypasses the primitive type allowlist."""
    runner = _FakeRunner()
    runner.status = lambda fingerprint: {
        "status": "running",
        "last_stable_checkpoint": {"name": "model_1.pt", "sha256": "bad", "size": True, "mtime_ns": -1},
    }
    monkeypatch.setenv("NEXT_RL_HOME", str(tmp_path / "state"))

    assert main(["status", "d" * 64], dependencies=CliDependencies(runner_factory=lambda home: runner)) == 0

    assert json.loads(capsys.readouterr().out) == {"fingerprint": "d" * 64, "status": "running"}


def _review_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path, dict[str, Path]]:
    spec = SkillSpec(
        "new-skill", "1.0.0", "A reviewable skill.", PolicyContract.microduck(),
        (MetricThreshold("falls", "count", "maximum", 0),), (1,), (2, 3, 4, 5),
        held_out_scenarios=("nominal", "entry", "exit", "stress"),
    )
    policy = tmp_path / "policy.onnx"
    policy.write_bytes(b"evaluated policy")
    digest = hashlib.sha256(policy.read_bytes()).hexdigest()
    scenarios = tuple(
        ScenarioResult(
            f"{family}-1", family, seed, {"falls": 0.0}, digest,
            (MetricResult(f"{family}-1", "falls", "count", "maximum", 0, 0, True, True, 0),),
        )
        for family, seed in (("nominal", 2), ("entry", 3), ("exit", 4), ("stress", 5))
    )
    report = EvaluationReport("new-skill", "1.0.0", "eval-test", scenarios, True, ArtifactRef(str(policy), "onnx", digest))
    by_family = {scenario.family: scenario for scenario in scenarios}
    clips = {}
    for role, scenario in {**by_family, "worst_case": select_worst_case(scenarios)}.items():
        clip = tmp_path / f"{role}.mp4"
        clip.write_bytes(role.encode())
        clips[role] = ReviewClip(role, scenario.scenario_id, scenario.seed, clip, digest)
    capability = Capability.from_dict({
        "id": "new-skill", "version": "1.0.0", "aliases": [], "robot_model": "microduck",
        "contract": PolicyContract.microduck().as_dict(), "status": "available",
    })
    capability_path = tmp_path / "capability.json"
    spec_path = tmp_path / "skill.json"
    evaluation_path = tmp_path / "evaluation.json"
    capability_path.write_text(json.dumps(capability.as_dict()))
    spec_path.write_text(canonical_json(spec.as_dict()))
    evaluation_path.write_text(canonical_json(report.as_dict()))
    return capability_path, spec_path, evaluation_path, policy, {
        role: clip.path for role, clip in clips.items()
    }


def _review_args(inputs: tuple[Path, Path, Path, Path, dict[str, Path]]) -> list[str]:
    capability, spec, evaluation, _policy, clips = inputs
    return [
        "review",
        "--capability", str(capability),
        "--skill", str(spec),
        "--evaluation", str(evaluation),
        "--nominal-clip", str(clips["nominal"]),
        "--entry-clip", str(clips["entry"]),
        "--exit-clip", str(clips["exit"]),
        "--stress-clip", str(clips["stress"]),
        "--worst-case-clip", str(clips["worst_case"]),
    ]


def test_review_then_approval_requires_a_persisted_record_and_reviewer(tmp_path: Path, monkeypatch, capsys):
    """Catch approval that can bypass the persisted human-review record."""
    monkeypatch.setenv("NEXT_RL_HOME", str(tmp_path / "state"))
    inputs = _review_inputs(tmp_path)

    baseline = tmp_path / "baseline.json"
    baseline.write_text(inputs[2].read_text())
    assert main(_review_args(inputs) + ["--baseline", str(baseline)]) == 0
    review = json.loads(capsys.readouterr().out)
    assert review["status"] == "review_pending"
    assert Path(review["bundle_path"]).is_file()
    assert len(review["bundle_digest"]) == 64
    assert main(["approve", "not-a-persisted-record", "--reviewer", "operator"]) == 2
    assert json.loads(capsys.readouterr().out) == {"error": "invalid_request"}
    assert main(["approve", review["record_id"], "--reviewer", " "]) == 2
    assert json.loads(capsys.readouterr().out) == {"error": "invalid_request"}
    assert main(["approve", review["record_id"], "--reviewer", "operator"]) == 0

    approved = json.loads(capsys.readouterr().out)
    assert approved == {"record_id": review["record_id"], "status": "learned"}


def test_review_after_rejection_reuses_the_exact_validated_candidate(tmp_path: Path, monkeypatch, capsys):
    """Catch re-review that revalidates rather than preserving the original audit."""
    monkeypatch.setenv("NEXT_RL_HOME", str(tmp_path / "state"))
    inputs = _review_inputs(tmp_path)
    assert main(_review_args(inputs)) == 0
    record_id = json.loads(capsys.readouterr().out)["record_id"]

    assert main(["reject", record_id, "--reviewer", "operator", "--reason", "leans"]) == 0
    assert json.loads(capsys.readouterr().out) == {"record_id": record_id, "status": "validated"}
    assert main(_review_args(inputs)) == 0

    reopened = json.loads(capsys.readouterr().out)
    assert reopened["record_id"] == record_id
    state = json.loads((tmp_path / "state" / "promotions" / "state.json").read_text())
    assert [item["action"] for item in state["records"][record_id]["audit"]] == [
        "available", "validated", "requested", "rejected", "requested",
    ]


def test_re_review_rejects_mismatched_capability_spec_or_policy_evidence(tmp_path: Path, monkeypatch, capsys):
    """Catch re-review that overwrites a rejected candidate with different evidence."""
    monkeypatch.setenv("NEXT_RL_HOME", str(tmp_path / "state"))
    inputs = _review_inputs(tmp_path)
    assert main(_review_args(inputs)) == 0
    record_id = json.loads(capsys.readouterr().out)["record_id"]
    assert main(["reject", record_id, "--reviewer", "operator", "--reason", "leans"]) == 0
    capsys.readouterr()

    capability, spec, evaluation, _policy, _clips = inputs
    changed_spec = tmp_path / "changed-spec.json"
    changed_spec_data = json.loads(spec.read_text())
    changed_spec_data["description"] = "Different skill evidence."
    changed_spec.write_text(canonical_json(changed_spec_data))
    assert main(_review_args((capability, changed_spec, evaluation, _policy, _clips))) == 2
    assert json.loads(capsys.readouterr().out) == {"error": "invalid_request"}

    changed_capability = tmp_path / "changed-capability.json"
    changed = json.loads(capability.read_text())
    changed["id"] = "other-skill"
    changed_capability.write_text(canonical_json(changed))
    assert main(_review_args((changed_capability, spec, evaluation, _policy, _clips))) == 2
    assert json.loads(capsys.readouterr().out) == {"error": "invalid_request"}

    changed_policy = tmp_path / "other.onnx"
    changed_policy.write_bytes(b"different evaluated policy")
    changed_evaluation = tmp_path / "changed-evaluation.json"
    changed_evaluation_data = json.loads(evaluation.read_text())
    policy_digest = hashlib.sha256(changed_policy.read_bytes()).hexdigest()
    changed_evaluation_data["policy"]["path"] = str(changed_policy)
    changed_evaluation_data["policy"]["sha256"] = policy_digest
    for scenario in changed_evaluation_data["scenarios"]:
        scenario["policy_sha256"] = policy_digest
    changed_evaluation.write_text(canonical_json(changed_evaluation_data))
    assert main(_review_args((capability, spec, changed_evaluation, changed_policy, _clips))) == 2
    assert json.loads(capsys.readouterr().out) == {"error": "invalid_request"}

    state = json.loads((tmp_path / "state" / "promotions" / "state.json").read_text())
    assert state["records"][record_id]["status"] == "validated"


def test_inventory_combines_shipped_and_promoted_capabilities(tmp_path: Path, monkeypatch, capsys):
    """Catch an inventory that hides durable promoted capabilities or the shipped catalogue."""
    monkeypatch.setenv("NEXT_RL_HOME", str(tmp_path / "state"))
    inputs = _review_inputs(tmp_path)
    assert main(_review_args(inputs)) == 0
    record_id = json.loads(capsys.readouterr().out)["record_id"]
    assert main(["approve", record_id, "--reviewer", "operator"]) == 0
    capsys.readouterr()

    assert main(["inventory"]) == 0

    capabilities = json.loads(capsys.readouterr().out)["capabilities"]
    assert {"id": "new-skill", "status": "learned"}.items() <= next(
        item.items() for item in capabilities if item["id"] == "new-skill"
    )
    assert any(item["id"] == "walking" for item in capabilities)


def test_warm_start_plan_uses_task8_parent_capability_json(tmp_path: Path, monkeypatch, capsys):
    """Catch warm-start output that loses the selected parent capability identity."""
    monkeypatch.setenv("NEXT_RL_HOME", str(tmp_path / "state"))
    skill = _write_skill(tmp_path / "hello.json", "one-leg-hello", allowed_parent_capabilities=["standing"])

    assert main(["plan", str(skill)]) == 0

    assert json.loads(capsys.readouterr().out)["parent_capability_id"] == "standing"

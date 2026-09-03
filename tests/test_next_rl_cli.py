"""Behavioral contracts for the guarded Next RL operator CLI."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
import tomllib

from mjlab_microduck.next_rl.cli import CliDependencies, main
from mjlab_microduck.next_rl.evaluation import EvaluationReport, MetricResult, ScenarioResult, select_worst_case
from mjlab_microduck.next_rl.review import ReviewBundle, ReviewClip
from mjlab_microduck.next_rl.schema import ArtifactRef, Capability, MetricThreshold, PolicyContract, SkillSpec


def test_next_rl_is_a_declared_script():
    """Catch a distributable CLI that is not reachable from the installed command."""
    scripts = tomllib.loads(Path("pyproject.toml").read_text())["project"]["scripts"]

    assert scripts["next-rl"] == "mjlab_microduck.next_rl.cli:main"


def _write_skill(path: Path, skill_id: str) -> Path:
    path.write_text(json.dumps({
        "id": skill_id,
        "version": "1.0.0",
        "description": "A test skill.",
        "contract": PolicyContract.microduck().as_dict(),
        "metrics": [{"name": "falls", "unit": "count", "direction": "maximum", "limit": 0}],
        "training_seeds": [1],
        "evaluation_seeds": [2],
    }))
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
    assert json.loads(capsys.readouterr().out)["status"] == "prepared"
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


def _review_inputs(tmp_path: Path) -> tuple[Path, Path]:
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
    bundle_path = tmp_path / "bundle.json"
    capability_path.write_text(json.dumps(capability.as_dict()))
    bundle_path.write_text(json.dumps(ReviewBundle.build(report, clips, spec=spec).as_dict()))
    return capability_path, bundle_path


def test_review_then_approval_requires_a_persisted_record_and_reviewer(tmp_path: Path, monkeypatch, capsys):
    """Catch approval that can bypass the persisted human-review record."""
    monkeypatch.setenv("NEXT_RL_HOME", str(tmp_path / "state"))
    capability, bundle = _review_inputs(tmp_path)

    assert main(["review", str(capability), str(bundle)]) == 0
    review = json.loads(capsys.readouterr().out)
    assert review["status"] == "review_pending"
    assert main(["approve", "not-a-persisted-record", "--reviewer", "operator"]) == 2
    assert json.loads(capsys.readouterr().out) == {"error": "invalid_request"}
    assert main(["approve", review["record_id"], "--reviewer", " "]) == 2
    assert json.loads(capsys.readouterr().out) == {"error": "invalid_request"}
    assert main(["approve", review["record_id"], "--reviewer", "operator"]) == 0

    approved = json.loads(capsys.readouterr().out)
    assert approved == {"record_id": review["record_id"], "status": "learned"}


def test_reject_returns_the_exact_review_record_to_validated(tmp_path: Path, monkeypatch, capsys):
    """Catch rejection that fails to preserve the durable review lifecycle."""
    monkeypatch.setenv("NEXT_RL_HOME", str(tmp_path / "state"))
    capability, bundle = _review_inputs(tmp_path)
    assert main(["review", str(capability), str(bundle)]) == 0
    record_id = json.loads(capsys.readouterr().out)["record_id"]

    assert main(["reject", record_id, "--reviewer", "operator", "--reason", "leans"]) == 0

    assert json.loads(capsys.readouterr().out) == {"record_id": record_id, "status": "validated"}


def test_inventory_combines_shipped_and_promoted_capabilities(tmp_path: Path, monkeypatch, capsys):
    """Catch an inventory that hides durable promoted capabilities or the shipped catalogue."""
    monkeypatch.setenv("NEXT_RL_HOME", str(tmp_path / "state"))
    capability, bundle = _review_inputs(tmp_path)
    assert main(["review", str(capability), str(bundle)]) == 0
    record_id = json.loads(capsys.readouterr().out)["record_id"]
    assert main(["approve", record_id, "--reviewer", "operator"]) == 0
    capsys.readouterr()

    assert main(["inventory"]) == 0

    capabilities = json.loads(capsys.readouterr().out)["capabilities"]
    assert {"id": "new-skill", "status": "learned"}.items() <= next(
        item.items() for item in capabilities if item["id"] == "new-skill"
    )
    assert any(item["id"] == "walking" for item in capabilities)

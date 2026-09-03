"""Behavioural contracts for the Next RL capability inventory and planner."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from test_next_rl_support import write_tiny_policy

from mjlab_microduck.next_rl.capabilities import (
    CapabilityInventory,
    Disposition,
    plan_skill,
)
from mjlab_microduck.next_rl.schema import (
    ArtifactRef,
    Capability,
    MetricThreshold,
    PolicyContract,
    SkillSpec,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_next_rl_catalog.py"
SHIPPED_FILENAMES = (
    "alpha_stand.onnx",
    "alpha_walking.onnx",
    "alpha_sitstand.onnx",
    "alpha_ground_pick.onnx",
    "ball_kick_left.onnx",
    "ball_kick_right.onnx",
    "roller.onnx",
    "roller_crouch.onnx",
    "roulade.onnx",
)


def metric(name: str, *, direction: str = "maximum", limit: float = 0) -> MetricThreshold:
    return MetricThreshold(name=name, unit="count", direction=direction, limit=limit)


def spec(
    skill_id: str,
    *,
    version: str = "1.0.0",
    metrics: tuple[MetricThreshold, ...] | None = None,
    contract: PolicyContract | None = None,
    aliases: tuple[str, ...] = (),
    allowed_parent_capabilities: tuple[str, ...] = (),
) -> SkillSpec:
    return SkillSpec(
        id=skill_id,
        version=version,
        description=f"Test specification for {skill_id}.",
        contract=contract or PolicyContract.microduck(),
        metrics=metrics or (metric("falls"),),
        training_seeds=(1,),
        evaluation_seeds=(2,),
        aliases=aliases,
        allowed_parent_capabilities=allowed_parent_capabilities,
    )


def capability(
    capability_id: str,
    *,
    aliases: tuple[str, ...] = (),
    status: str = "learned",
    metric_results: dict[str, float] | None = None,
    contract: PolicyContract | None = None,
    version: str = "1.0.0",
) -> Capability:
    raw: dict[str, object] = {
        "id": capability_id,
        "version": version,
        "aliases": list(aliases),
        "robot_model": "microduck",
        "contract": (contract or PolicyContract.microduck()).as_dict(),
        "status": status,
    }
    if status == "learned":
        raw["policy"] = {"path": f"policies/{capability_id}.onnx", "kind": "onnx", "sha256": "a" * 64}
        raw["evaluation"] = {
            "kind": "evaluation_report",
            "policy_sha256": "a" * 64,
            "report_path": "evaluations/approved.json",
            "passed": True,
            "metric_results": metric_results or {"falls": 0},
            "approval_provenance": "reviews/approved.json",
        }
    return Capability.from_dict(raw)


def inventory_with(
    *,
    alias: str | None = None,
    capability_ids: tuple[str, ...] = ("new-skill",),
    status: str = "learned",
) -> CapabilityInventory:
    return CapabilityInventory(
        tuple(capability(capability_id, aliases=(alias,) if alias else (), status=status) for capability_id in capability_ids)
    )


def builtin_inventory() -> CapabilityInventory:
    return CapabilityInventory.load_builtin()


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


@pytest.fixture
def runtime_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "runtime"
    policies = repo / "policies"
    policies.mkdir(parents=True)
    for filename in SHIPPED_FILENAMES:
        (policies / filename).write_bytes(f"HEAD policy: {filename}".encode())
    (policies / "README.md").write_text("Approved policy inventory.\n", encoding="utf-8")
    git(repo, "init", "-q")
    git(repo, "config", "user.email", "tests@example.com")
    git(repo, "config", "user.name", "Next RL tests")
    git(repo, "add", "policies")
    git(repo, "commit", "-qm", "Add tracked runtime policies")
    return repo


def run_catalog(runtime_repo: Path, output: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--runtime-repo", str(runtime_repo), "--output", str(output), *extra],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def load_catalog_module():
    module_spec = importlib.util.spec_from_file_location("next_rl_catalog_test", SCRIPT)
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def test_builtin_catalog_contains_shipped_runtime_skills():
    inventory = CapabilityInventory.load_builtin()
    assert {"walking", "standing", "sitstand", "ground-pick", "kick-left", "kick-right", "roulade"} <= inventory.ids


def test_alias_collision_blocks_instead_of_guessing():
    inventory = inventory_with(alias="kick", capability_ids=("kick-left", "kick-right"))
    decision = inventory.resolve("kick")
    assert decision.disposition == Disposition.BLOCKED


def test_available_task_does_not_block_training():
    decision = plan_skill(spec("new-skill"), inventory_with(status="available"))
    assert decision.disposition == Disposition.TRAIN_NEW


def test_shipped_match_with_missing_new_metrics_requires_evaluation_first():
    decision = plan_skill(spec("walking", metrics=(metric("new_accuracy"),)), builtin_inventory())
    assert decision.disposition == Disposition.BLOCKED
    assert "evaluate existing policy" in decision.reason


def test_shipped_match_requires_evaluation_even_for_legacy_named_metric():
    decision = plan_skill(spec("walking"), builtin_inventory())
    assert decision.disposition == Disposition.BLOCKED
    assert "evaluate existing policy" in decision.reason


def test_requested_alias_matching_existing_exact_id_requires_evaluation_first():
    decision = plan_skill(spec("new-walking", aliases=("walking",)), builtin_inventory())
    assert decision.disposition == Disposition.BLOCKED
    assert "evaluate existing policy" in decision.reason


def test_requested_alias_matching_catalog_alias_requires_evaluation_first():
    decision = plan_skill(spec("new-walking", aliases=("walk",)), builtin_inventory())
    assert decision.disposition == Disposition.BLOCKED
    assert "evaluate existing policy" in decision.reason


def test_requested_id_and_alias_for_same_capability_do_not_create_ambiguity():
    inventory = CapabilityInventory((capability("walking", aliases=("walk",)),))
    decision = plan_skill(spec("walking", aliases=("walk",)), inventory)
    assert decision.disposition == Disposition.REUSE


def test_exact_id_outranks_another_capability_alias():
    inventory = CapabilityInventory(
        (
            capability("walking"),
            capability("forward-gait", aliases=("walking",)),
        )
    )
    decision = inventory.resolve("walking")
    assert decision.disposition == Disposition.REUSE
    assert decision.capability is not None
    assert decision.capability.id == "walking"


def test_contract_mismatch_blocks_reuse():
    incompatible = PolicyContract(model_api=1, obs_len=62, action_len=14, robot_model="microduck", control_hz=50)
    decision = plan_skill(spec("walking", contract=incompatible), inventory_with(capability_ids=("walking",)))
    assert decision.disposition == Disposition.BLOCKED
    assert "contract" in decision.reason


def test_complete_approved_evaluation_reuses_learned_capability():
    decision = plan_skill(spec("walking"), inventory_with(capability_ids=("walking",)))
    assert decision.disposition == Disposition.REUSE


def test_spec_version_mismatch_requires_evaluation_before_reuse():
    decision = plan_skill(spec("walking", version="1.0.1"), inventory_with(capability_ids=("walking",)))
    assert decision.disposition == Disposition.BLOCKED
    assert "evaluate existing policy" in decision.reason


def test_onnx_parent_is_only_a_reference_and_training_starts_new(tmp_path: Path):
    policy = write_tiny_policy(tmp_path / "walking.onnx")
    digest = hashlib.sha256(policy.read_bytes()).hexdigest()
    parent = replace(
        capability("walking"),
        policy=ArtifactRef(str(policy), "onnx", digest),
    )
    decision = plan_skill(
        spec("running", allowed_parent_capabilities=("walking",)),
        CapabilityInventory((parent,)),
    )
    assert decision.disposition == Disposition.TRAIN_NEW
    assert decision.capability is not None
    assert decision.capability.id == "walking"
    assert "training checkpoint" in decision.reason


def test_same_version_improvement_is_blocked_until_the_spec_version_is_bumped():
    decision = plan_skill(
        spec("walking"),
        inventory_with(capability_ids=("walking",)),
        improve_reason="Reduce energy consumption under the approved contract.",
    )
    assert decision.disposition == Disposition.BLOCKED
    assert "version bump" in decision.reason


def test_higher_version_improvement_uses_only_a_loadable_learned_reference(tmp_path: Path):
    policy = write_tiny_policy(tmp_path / "walking.onnx")
    digest = hashlib.sha256(policy.read_bytes()).hexdigest()
    learned = replace(
        capability("walking"),
        policy=ArtifactRef(str(policy), "onnx", digest),
    )

    decision = plan_skill(
        spec("walking", version="1.0.1"),
        CapabilityInventory((learned,)),
        improve_reason="Reduce energy consumption under the approved contract.",
    )

    assert decision.disposition == Disposition.TRAIN_NEW
    assert decision.capability == learned
    assert decision.improve_reason == "Reduce energy consumption under the approved contract."


def test_higher_version_improvement_can_add_a_metric_missing_from_prior_evidence(
    tmp_path: Path,
):
    policy = write_tiny_policy(tmp_path / "walking.onnx")
    digest = hashlib.sha256(policy.read_bytes()).hexdigest()
    learned = replace(
        capability("walking"),
        policy=ArtifactRef(str(policy), "onnx", digest),
    )

    decision = plan_skill(
        spec(
            "walking",
            version="1.0.1",
            metrics=(metric("falls"), metric("new_accuracy", direction="minimum", limit=1)),
        ),
        CapabilityInventory((learned,)),
        improve_reason="Add an explicitly evaluated accuracy target.",
    )

    assert decision.disposition == Disposition.TRAIN_NEW
    assert decision.capability == learned


def test_higher_version_improvement_blocks_an_unloadable_learned_reference():
    decision = plan_skill(
        spec("walking", version="1.0.1"),
        inventory_with(capability_ids=("walking",)),
        improve_reason="Reduce energy consumption under the approved contract.",
    )

    assert decision.disposition == Disposition.BLOCKED
    assert "loadable" in decision.reason


def test_active_learned_version_outranks_its_superseded_history():
    previous = replace(capability("walking"), status="superseded")
    current = capability("walking", version="1.0.1")

    decision = CapabilityInventory((previous, current)).resolve("walking")

    assert decision.disposition == Disposition.REUSE
    assert decision.capability == current


def test_catalog_generator_hashes_tracked_head_policy_bytes(runtime_repo: Path, tmp_path: Path):
    output = tmp_path / "catalog.json"
    result = run_catalog(runtime_repo, output)
    assert result.returncode == 0, result.stderr
    data = json.loads(output.read_text(encoding="utf-8"))
    walking = next(item for item in data["capabilities"] if item["id"] == "walking")
    expected = hashlib.sha256(b"HEAD policy: alpha_walking.onnx").hexdigest()
    assert walking["policy"]["sha256"] == expected
    assert walking["evaluation"]["runtime_commit"] == git(runtime_repo, "rev-parse", "HEAD")
    assert walking["evaluation"]["metadata"]["approval_sha256"] == hashlib.sha256(
        b"Approved policy inventory.\n"
    ).hexdigest()


def test_catalog_check_uses_head_policy_when_worktree_policy_changes(runtime_repo: Path, tmp_path: Path):
    output = tmp_path / "catalog.json"
    assert run_catalog(runtime_repo, output).returncode == 0
    (runtime_repo / "policies/alpha_walking.onnx").write_bytes(b"uncommitted replacement")
    result = run_catalog(runtime_repo, output, "--check")
    assert result.returncode == 0, result.stderr


def test_catalog_generator_reads_every_object_from_the_captured_commit(runtime_repo: Path, monkeypatch):
    catalog = load_catalog_module()
    captured_commit = git(runtime_repo, "rev-parse", "HEAD")
    object_specs: list[tuple[str, ...]] = []
    original_git_bytes = catalog._git_bytes

    def record_object_spec(repo: Path, *args: str) -> bytes:
        object_specs.append(args)
        return original_git_bytes(repo, *args)

    monkeypatch.setattr(catalog, "_git_bytes", record_object_spec)
    catalog.build_catalog(runtime_repo)

    expected_paths = {"policies/README.md", *(f"policies/{filename}" for filename in SHIPPED_FILENAMES)}
    assert set(object_specs) == {("show", f"{captured_commit}:{path}") for path in expected_paths}


def test_catalog_generator_requires_readme_tracked_at_recorded_commit(runtime_repo: Path, tmp_path: Path):
    git(runtime_repo, "rm", "-q", "policies/README.md")
    git(runtime_repo, "commit", "-qm", "Remove approval provenance")
    result = run_catalog(runtime_repo, tmp_path / "catalog.json")
    assert result.returncode != 0
    assert "policies/README.md" in result.stderr

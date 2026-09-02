"""Behavioural contracts for the Next RL capability inventory and planner."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from mjlab_microduck.next_rl.capabilities import (
    CapabilityInventory,
    Disposition,
    plan_skill,
)
from mjlab_microduck.next_rl.schema import Capability, MetricThreshold, PolicyContract, SkillSpec


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_REPO = Path("/Users/rakeshutekar/Documents/microduck")


def metric(name: str, *, direction: str = "maximum", limit: float = 0) -> MetricThreshold:
    return MetricThreshold(name=name, unit="count", direction=direction, limit=limit)


def spec(
    skill_id: str,
    *,
    metrics: tuple[MetricThreshold, ...] | None = None,
    contract: PolicyContract | None = None,
    allowed_parent_capabilities: tuple[str, ...] = (),
) -> SkillSpec:
    return SkillSpec(
        id=skill_id,
        version="1.0.0",
        description=f"Test specification for {skill_id}.",
        contract=contract or PolicyContract.microduck(),
        metrics=metrics or (metric("falls"),),
        training_seeds=(1,),
        evaluation_seeds=(2,),
        allowed_parent_capabilities=allowed_parent_capabilities,
    )


def capability(
    capability_id: str,
    *,
    aliases: tuple[str, ...] = (),
    status: str = "learned",
    metric_results: dict[str, float] | None = None,
    contract: PolicyContract | None = None,
) -> Capability:
    raw: dict[str, object] = {
        "id": capability_id,
        "version": "1.0.0",
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


def test_allowed_compatible_parent_warm_starts_a_new_skill():
    decision = plan_skill(
        spec("running", allowed_parent_capabilities=("walking",)),
        inventory_with(capability_ids=("walking",)),
    )
    assert decision.disposition == Disposition.WARM_START
    assert decision.capability is not None
    assert decision.capability.id == "walking"


def test_improvement_reason_trains_after_approved_reuse():
    decision = plan_skill(
        spec("walking"),
        inventory_with(capability_ids=("walking",)),
        improve_reason="Reduce energy consumption under the approved contract.",
    )
    assert decision.disposition == Disposition.TRAIN_NEW
    assert decision.improve_reason == "Reduce energy consumption under the approved contract."


def test_catalog_check_matches_the_shipped_runtime_policies():
    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_next_rl_catalog.py",
            "--runtime-repo",
            str(RUNTIME_REPO),
            "--check",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

"""Guarded JSON command line interface for the Next RL workspace."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .capabilities import CapabilityInventory, Disposition, plan_skill
from .artifacts import atomic_write_json, canonical_json
from .evaluation import EvaluationReport, MetricResult, ScenarioResult
from .experiments import ExperimentStore, build_experiment_manifest
from .promotion import PromotionError, PromotionStore
from .review import ReviewBundle, ReviewClip
from .runner import NitroConfig, NitroRunner
from .schema import ArtifactRef, Capability, SkillSpec


class _Parser(argparse.ArgumentParser):
    """An argparse parser that lets the public boundary emit JSON errors."""

    def error(self, message: str) -> None:
        raise ValueError(message)


def _default_runner(home: Path) -> NitroRunner:
    """Construct transport lazily; configuration alone does not contact Nitro."""
    return NitroRunner(
        NitroConfig(
            ssh_alias=os.environ["NEXT_RL_NITRO_SSH_ALIAS"],
            ssh_user=os.environ.get("NEXT_RL_NITRO_SSH_USER", "aif_eng"),
            repository=Path.cwd(),
            bundle_root=home / "bundles",
        )
    )


@dataclass(frozen=True)
class CliDependencies:
    """Injectable boundaries keep CLI tests local and credential-free."""

    runner_factory: Callable[[Path], Any] = _default_runner
    experiment_store_factory: Callable[[Path], ExperimentStore] = ExperimentStore
    promotion_store_factory: Callable[[Path], PromotionStore] = PromotionStore


def _emit(value: dict[str, Any]) -> None:
    print(json.dumps(value, sort_keys=True, separators=(",", ":")))


def _skill(path: str) -> SkillSpec:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return SkillSpec.from_dict(raw)


def _decision_json(decision: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "disposition": decision.disposition.value,
        "reason": decision.reason,
    }
    if decision.disposition == Disposition.WARM_START and decision.capability is not None:
        result["parent_capability_id"] = decision.capability.id
    elif decision.capability is not None:
        result["capability_id"] = decision.capability.id
    if decision.improve_reason is not None:
        result["improve_reason"] = decision.improve_reason
    return result


def _home() -> Path:
    return Path(os.environ.get("NEXT_RL_HOME", ".next-rl")).resolve()


def _inventory(home: Path, dependencies: CliDependencies) -> CapabilityInventory:
    """Overlay persisted promotion state over the shipped capability catalogue."""
    builtin = CapabilityInventory.load_builtin()
    persisted = dependencies.promotion_store_factory(home / "promotions").inventory()
    capabilities = {
        (capability.id, capability.version): capability
        for capability in getattr(builtin, "_capabilities")
    }
    for capability in getattr(persisted, "_capabilities"):
        capabilities[(capability.id, capability.version)] = capability
    return CapabilityInventory(capabilities.values())


def _inventory_json(home: Path, dependencies: CliDependencies) -> dict[str, Any]:
    inventory = _inventory(home, dependencies)
    capabilities = sorted(
        (
            {
                "aliases": list(capability.aliases),
                "id": capability.id,
                "robot_model": capability.robot_model,
                "status": capability.status,
                "version": capability.version,
            }
            for capability in getattr(inventory, "_capabilities")
        ),
        key=lambda capability: (capability["id"], capability["version"]),
    )
    return {"capabilities": capabilities}


def _code_digest() -> str:
    """Bind a prepared manifest to the current source revision without shelling out."""
    revision = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=Path.cwd(),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return hashlib.sha256(revision.encode("ascii")).hexdigest()


def _prepare(args: argparse.Namespace, dependencies: CliDependencies) -> tuple[int, dict[str, Any]]:
    home = _home()
    spec = _skill(args.skill)
    decision = plan_skill(spec, _inventory(home, dependencies), args.improve_reason)
    if decision.disposition in {Disposition.REUSE, Disposition.BLOCKED}:
        return 2, _decision_json(decision)

    task_id = args.task_id or f"Mjlab-{spec.id}-MicroDuck"
    seed = spec.training_seeds[0] if spec.training_seeds else 0
    manifest = build_experiment_manifest(
        spec,
        decision,
        task_id=task_id,
        code_digest=_code_digest(),
        seed=seed,
        runner_id="nitro",
        environment_config={"scene": {"num_envs": args.num_envs}},
        agent_config={
            "max_iterations": args.max_iterations,
            "run_name": args.run_name or f"next-rl-{hashlib.sha256(spec.id.encode()).hexdigest()[:12]}",
        },
        parent_policy=(decision.capability.policy if decision.disposition == Disposition.WARM_START and decision.capability else None),
    )
    store = dependencies.experiment_store_factory(home / "experiments")
    fingerprint = store.create(manifest)
    prepared = dependencies.runner_factory(home).prepare(manifest)
    return 0, {"fingerprint": fingerprint, "prepared_fingerprint": prepared.fingerprint, "status": "planned"}


def _status(fingerprint: str, dependencies: CliDependencies) -> dict[str, Any]:
    """Expose only durable, credential-free state returned by the runner."""
    state = dependencies.runner_factory(_home()).status(fingerprint)
    result: dict[str, Any] = {"fingerprint": fingerprint}
    if state.get("status") in {"pending", "running", "succeeded", "failed"}:
        result["status"] = state["status"]
    if state.get("artifact_status") in {"pending", "stable_checkpoint", "missing"}:
        result["artifact_status"] = state["artifact_status"]
    if "exit_code" in state:
        exit_code = state["exit_code"]
        if exit_code is None or (isinstance(exit_code, int) and not isinstance(exit_code, bool)):
            result["exit_code"] = exit_code
    checkpoint = state.get("last_stable_checkpoint")
    if isinstance(checkpoint, Mapping):
        name = checkpoint.get("name")
        digest = checkpoint.get("sha256")
        size = checkpoint.get("size")
        mtime = checkpoint.get("mtime_ns")
        if (
            isinstance(name, str)
            and re.fullmatch(r"model_[0-9]+\.pt", name)
            and isinstance(digest, str)
            and re.fullmatch(r"[0-9a-f]{64}", digest)
            and isinstance(size, int)
            and not isinstance(size, bool)
            and size > 0
            and isinstance(mtime, int)
            and not isinstance(mtime, bool)
            and mtime >= 0
        ):
            result["last_stable_checkpoint"] = {
                "mtime_ns": mtime,
                "name": name,
                "sha256": digest,
                "size": size,
            }
    return result


def _object(path: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON document must be an object")
    return value


def _evaluation(path: str) -> EvaluationReport:
    """Parse only fully typed evaluation evidence before it reaches review state."""
    raw = _object(path)
    if set(raw) != {"skill_id", "spec_version", "evaluator_revision", "scenarios", "passed", "policy"}:
        raise ValueError("evaluation fields are invalid")
    if not all(isinstance(raw[key], str) and raw[key].strip() for key in ("skill_id", "spec_version", "evaluator_revision")):
        raise ValueError("evaluation identity is invalid")
    if not isinstance(raw["passed"], bool) or not isinstance(raw["scenarios"], list) or not raw["scenarios"]:
        raise ValueError("evaluation body is invalid")
    policy = ArtifactRef.from_dict(raw["policy"])
    if policy.kind != "onnx":
        raise ValueError("evaluation policy must be ONNX")
    scenarios: list[ScenarioResult] = []
    for scenario in raw["scenarios"]:
        if not isinstance(scenario, Mapping) or set(scenario) != {
            "scenario_id", "family", "seed", "metrics", "policy_sha256", "threshold_results",
        }:
            raise ValueError("evaluation scenario is invalid")
        threshold_results = scenario["threshold_results"]
        if not isinstance(threshold_results, list) or not threshold_results:
            raise ValueError("evaluation thresholds are invalid")
        parsed_thresholds: list[MetricResult] = []
        for threshold in threshold_results:
            if not isinstance(threshold, Mapping) or set(threshold) != {
                "scenario_id", "metric_name", "unit", "direction", "limit", "value",
                "mandatory", "passed", "normalized_violation",
            }:
                raise ValueError("evaluation threshold is invalid")
            if (
                not all(isinstance(threshold[key], str) and threshold[key] for key in ("scenario_id", "metric_name", "unit"))
                or threshold["direction"] not in {"minimum", "maximum"}
                or not isinstance(threshold["mandatory"], bool)
                or not isinstance(threshold["passed"], bool)
                or any(isinstance(threshold[key], bool) or not isinstance(threshold[key], (int, float)) or not math.isfinite(threshold[key]) for key in ("limit", "value", "normalized_violation"))
            ):
                raise ValueError("evaluation threshold is invalid")
            parsed_thresholds.append(MetricResult(
                threshold["scenario_id"], threshold["metric_name"], threshold["unit"], threshold["direction"],
                float(threshold["limit"]), float(threshold["value"]), threshold["mandatory"], threshold["passed"],
                float(threshold["normalized_violation"]),
            ))
        scenarios.append(ScenarioResult(
            scenario["scenario_id"], scenario["family"], scenario["seed"], scenario["metrics"],
            scenario["policy_sha256"], tuple(parsed_thresholds),
        ))
    report = EvaluationReport(raw["skill_id"], raw["spec_version"], raw["evaluator_revision"], tuple(scenarios), raw["passed"], policy)
    if policy.sha256 != report.scenarios[0].policy_sha256 or any(item.policy_sha256 != policy.sha256 for item in report.scenarios):
        raise ValueError("evaluation policy binding is invalid")
    return report


def _review_clips(report: EvaluationReport, args: argparse.Namespace) -> dict[str, ReviewClip]:
    if report.policy is None:
        raise ValueError("evaluation policy is required")
    paths = {
        "nominal": args.nominal_clip,
        "entry": args.entry_clip,
        "exit": args.exit_clip,
        "stress": args.stress_clip,
        "worst_case": args.worst_case_clip,
    }
    selected: dict[str, ScenarioResult] = {}
    for role in ("nominal", "entry", "exit", "stress"):
        candidates = [scenario for scenario in report.scenarios if scenario.family == role]
        if not candidates:
            raise ValueError(f"evaluation lacks {role} scenario")
        selected[role] = min(candidates, key=lambda scenario: (scenario.seed, scenario.scenario_id))
    selected["worst_case"] = min(
        report.scenarios,
        key=lambda scenario: (-scenario.normalized_violation, scenario.scenario_id),
    )
    return {
        role: ReviewClip(role, scenario.scenario_id, scenario.seed, Path(paths[role]), report.policy.sha256)
        for role, scenario in selected.items()
    }


def _persist_bundle(home: Path, bundle: ReviewBundle) -> Path:
    """Persist the exact canonical bundle once under workspace-owned state."""
    target = home / "review-bundles" / f"{bundle.digest}.json"
    serialized = canonical_json(bundle.as_dict())
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.read_text(encoding="utf-8") != serialized:
            raise ValueError("persisted review bundle differs")
        return target
    atomic_write_json(target, bundle.as_dict())
    return target


def _persisted_capability(store: PromotionStore, capability: Capability) -> Capability | None:
    return next(
        (
            item for item in getattr(store.inventory(), "_capabilities")
            if item.id == capability.id and item.version == capability.version
        ),
        None,
    )


def _require_exact_validated_bundle(home: Path, capability: Capability, bundle: ReviewBundle) -> None:
    """Refuse a rejected candidate whose new review inputs differ by one byte."""
    state = _object(str(home / "promotions" / "state.json"))
    records = state.get("records")
    if not isinstance(records, Mapping):
        raise ValueError("promotion state is invalid")
    matching = [
        record for record in records.values()
        if isinstance(record, Mapping)
        and record.get("skill_id") == capability.id
        and record.get("spec_version") == capability.version
        and record.get("policy_digest") == bundle.policy_digest
        and record.get("status") == "validated"
    ]
    if len(matching) != 1 or matching[0].get("review_bundle_digest") != bundle.digest:
        raise ValueError("validated candidate evidence does not match")


def _review(args: argparse.Namespace, dependencies: CliDependencies) -> dict[str, Any]:
    """Build and durably request review for exact evaluator-produced evidence."""
    capability = Capability.from_dict(_object(args.capability))
    spec = _skill(args.skill)
    report = _evaluation(args.evaluation)
    baseline = _evaluation(args.baseline) if args.baseline is not None else None
    bundle = ReviewBundle.build(report, _review_clips(report, args), spec=spec, baseline=baseline)
    home = _home()
    store = dependencies.promotion_store_factory(home / "promotions")
    persisted = _persisted_capability(store, capability)
    if persisted is not None and persisted.status == "validated":
        _require_exact_validated_bundle(home, capability, bundle)
        bundle_path = _persist_bundle(home, bundle)
        record = store.request_review(capability, bundle)
    else:
        if persisted is not None and persisted.status != "available":
            raise PromotionError("review requires an available or exact validated candidate")
        bundle_path = _persist_bundle(home, bundle)
        store.validate(capability, bundle)
        record = store.request_review(capability, bundle)
    return {
        "bundle_digest": bundle.digest,
        "bundle_path": str(bundle_path),
        "record_id": record.id,
        "status": record.status,
    }


def _approval(record_id: str, reviewer: str, dependencies: CliDependencies) -> dict[str, Any]:
    record = dependencies.promotion_store_factory(_home() / "promotions").approve(record_id, reviewer=reviewer)
    return {"record_id": record.id, "status": record.status}


def _rejection(record_id: str, reviewer: str, reason: str, dependencies: CliDependencies) -> dict[str, Any]:
    record = dependencies.promotion_store_factory(_home() / "promotions").reject(
        record_id, reviewer=reviewer, reason=reason
    )
    return {"record_id": record.id, "status": record.status}


def _parser() -> _Parser:
    parser = _Parser(prog="next-rl")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("inventory")
    plan = commands.add_parser("plan")
    plan.add_argument("skill")
    plan.add_argument("--improve-reason")
    prepare = commands.add_parser("prepare")
    prepare.add_argument("skill")
    prepare.add_argument("--improve-reason")
    prepare.add_argument("--task-id")
    prepare.add_argument("--num-envs", type=int, default=64)
    prepare.add_argument("--max-iterations", type=int, default=5)
    prepare.add_argument("--run-name")
    status = commands.add_parser("status")
    status.add_argument("fingerprint")
    review = commands.add_parser("review")
    review.add_argument("--capability", required=True)
    review.add_argument("--skill", required=True)
    review.add_argument("--evaluation", required=True)
    review.add_argument("--nominal-clip", required=True)
    review.add_argument("--entry-clip", required=True)
    review.add_argument("--exit-clip", required=True)
    review.add_argument("--stress-clip", required=True)
    review.add_argument("--worst-case-clip", required=True)
    review.add_argument("--baseline")
    approve = commands.add_parser("approve")
    approve.add_argument("record_id")
    approve.add_argument("--reviewer", required=True)
    reject = commands.add_parser("reject")
    reject.add_argument("record_id")
    reject.add_argument("--reviewer", required=True)
    reject.add_argument("--reason", required=True)
    return parser


def main(argv: Sequence[str] | None = None, *, dependencies: CliDependencies | None = None) -> int:
    """Run a guarded operator command and emit exactly one JSON result."""
    try:
        args = _parser().parse_args(argv)
        dependencies = dependencies or CliDependencies()
        if args.command == "inventory":
            _emit(_inventory_json(_home(), dependencies))
            return 0
        if args.command == "plan":
            decision = plan_skill(
                _skill(args.skill), _inventory(_home(), dependencies),
                args.improve_reason,
            )
            _emit(_decision_json(decision))
            return 2 if decision.disposition in {Disposition.REUSE, Disposition.BLOCKED} else 0
        if args.command == "prepare":
            status, output = _prepare(args, dependencies)
            _emit(output)
            return status
        if args.command == "status":
            _emit(_status(args.fingerprint, dependencies))
            return 0
        if args.command == "review":
            _emit(_review(args, dependencies))
            return 0
        if args.command == "approve":
            _emit(_approval(args.record_id, args.reviewer, dependencies))
            return 0
        if args.command == "reject":
            _emit(_rejection(args.record_id, args.reviewer, args.reason, dependencies))
            return 0
        raise ValueError("unknown command")
    except Exception:
        _emit({"error": "invalid_request"})
        return 2


if __name__ == "__main__":  # pragma: no cover - console-script entry point
    raise SystemExit(main())

"""Guarded JSON command line interface for the Next RL workspace."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .capabilities import CapabilityInventory, Disposition, plan_skill
from .experiments import ExperimentStore, build_experiment_manifest
from .promotion import PromotionStore
from .review import ReviewBundle
from .runner import NitroConfig, NitroRunner
from .schema import Capability, SkillSpec


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
    if decision.capability is not None:
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
    return 0, {"fingerprint": fingerprint, "prepared_fingerprint": prepared.fingerprint, "status": "prepared"}


def _status(fingerprint: str, dependencies: CliDependencies) -> dict[str, Any]:
    """Expose only durable, credential-free state returned by the runner."""
    state = dependencies.runner_factory(_home()).status(fingerprint)
    result: dict[str, Any] = {"fingerprint": fingerprint}
    for key in ("status", "artifact_status", "exit_code", "last_stable_checkpoint"):
        if key in state:
            result[key] = state[key]
    return result


def _object(path: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON document must be an object")
    return value


def _review(capability_path: str, bundle_path: str, dependencies: CliDependencies) -> dict[str, Any]:
    """Persist verified evidence before opening the exact candidate for review."""
    capability = Capability.from_dict(_object(capability_path))
    bundle = ReviewBundle.from_dict(_object(bundle_path))
    store = dependencies.promotion_store_factory(_home() / "promotions")
    store.validate(capability, bundle)
    record = store.request_review(capability, bundle)
    return {"record_id": record.id, "status": record.status}


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
    review.add_argument("capability")
    review.add_argument("bundle")
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
            _emit(_review(args.capability, args.bundle, dependencies))
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

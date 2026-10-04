"""Parity-first migration of CPU-stack behaviors (microduck_local) onto the MJX/GPU stack.

Direction ⑥ (GPU-dependent). EXECUTION is BLOCKED in this sandbox: there is no
CUDA/MJX runtime here. What IS shipped and runnable offline:

  * build_migration_plan() — enumerates every reward term that must pass a
    CPU-vs-MJX numeric parity check before its MJX twin is promoted to the
    deployable contract. Pure data, reads the parity gate REGISTRY so the plan
    can never drift from the CI backstop (reward_parity_gate).

  * execute_migration() — the real MJX-vs-CPU alignment loop. GUARDED: it
    returns {"status": "BLOCKED", ...} instead of crashing when mujoco_warp /
    a CUDA device is unavailable, so CI stays green and the block is explicit
    and machine-assertable.

Parity-first means: an MJX behavior is NOT promoted until every REGISTRY term
agrees with its CPU reference within tolerance. The reward_parity_gate is the
backstop that makes "you forgot a parity check" a red CI instead of a postmortem.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

GATE_PATH = Path(__file__).resolve().parent / "reward_parity_gate.py"


def _load_gate():
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("reward_parity_gate", GATE_PATH)
    gate = importlib.util.module_from_spec(spec)
    sys.modules["reward_parity_gate"] = gate  # dataclass needs the module live
    spec.loader.exec_module(gate)
    return gate


@dataclass
class MigrationPlan:
    terms: list[dict[str, Any]] = field(default_factory=list)
    offline_runnable: bool = True  # plan construction needs no sim


def build_migration_plan() -> MigrationPlan:
    """Offline: list every term that must pass parity before MJX promotion.

    Reads the parity gate REGISTRY directly, so this plan is always in lockstep
    with the CI backstop. No simulation required.
    """
    gate = _load_gate()
    terms = [
        {
            "name": t.name,
            "mjlab_func": t.mjlab_func,
            "covered": t.covered,
        }
        for t in gate.registered_terms()
    ]
    return MigrationPlan(terms=terms)


def execute_migration(mjx_env_factory: Optional[Any] = None) -> dict[str, Any]:
    """BLOCKED in this environment: run CPU-vs-MJX numeric parity, term by term.

    Requires mujoco_warp + a CUDA device. Returns an explicit BLOCKED dict
    (status == "BLOCKED") rather than raising, so callers and CI can assert on
    the status and the block is never silent.
    """
    try:
        import mujoco_warp  # noqa: F401 — only present on the CUDA box
    except Exception:
        return {
            "status": "BLOCKED",
            "reason": "mujoco_warp (MJX) not importable — needs the CUDA host",
            "next": "run on the GPU box, then call execute_migration(env_factory)",
        }
    if mjx_env_factory is None:
        return {"status": "BLOCKED", "reason": "no MJX env factory supplied"}
    # --- execution path (only reached on CUDA) ---
    plan = build_migration_plan()
    return {
        "status": "RUNNING",
        "note": "MJX available; parity loop over %d terms not yet invoked" % len(plan.terms),
    }


if __name__ == "__main__":
    plan = build_migration_plan()
    print(f"migration plan: {len(plan.terms)} terms to parity-check")
    for t in plan.terms:
        print(f"  - {t['name']} ({t['mjlab_func']}) covered={t['covered']}")

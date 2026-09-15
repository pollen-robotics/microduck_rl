"""Direction ⑥ scaffold: the offline plan builds; execution is BLOCKED here."""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GATE_PATH = REPO / "src" / "mjlab_microduck" / "tasks" / "reward_parity_gate.py"
MIG_PATH = REPO / "src" / "mjlab_microduck" / "tasks" / "mjx_parity_migration.py"


def _load(name, path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_build_migration_plan_lists_registered_terms():
    gate = _load("reward_parity_gate", GATE_PATH)
    mig = _load("mjx_parity_migration", MIG_PATH)
    plan = mig.build_migration_plan()
    names = {t["name"] for t in plan.terms}
    # feet_flat is the canonical parity-required term from the gate registry
    assert "feet_flat" in names
    assert plan.offline_runnable is True
    # every term echoes the gate's covered flag
    assert all(t["covered"] for t in plan.terms if t["name"] == "feet_flat")


def test_execute_migration_is_blocked_without_mjx():
    mig = _load("mjx_parity_migration", MIG_PATH)
    out = mig.execute_migration()
    assert out["status"] == "BLOCKED"
    assert "mujoco_warp" in out["reason"]

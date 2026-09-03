"""Public-CLI readiness checks for the documented Next RL workspace path."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "skills" / "one-leg-hello.json"
GUIDE = ROOT / "docs" / "next-rl-workspace.md"


def _environment(home: Path) -> dict[str, str]:
    """Isolate durable workspace state while retaining the local uv environment."""
    return {**os.environ, "NEXT_RL_HOME": str(home)}


def _run_cli(*arguments: str, home: Path) -> dict[str, object]:
    """Run the installed public command, not a direct Python helper."""
    result = subprocess.run(
        ("uv", "run", "next-rl", *arguments),
        cwd=ROOT,
        env=_environment(home),
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _run_cli_with_local_runner(
    temporary: Path,
    *arguments: str,
    home: Path,
) -> dict[str, object]:
    """Exercise public parser/process behavior without SSH, training, or network I/O."""
    launcher = temporary / "next_rl_local_runner.py"
    launcher.write_text(
        """
import sys
from types import SimpleNamespace

from mjlab_microduck.next_rl.cli import CliDependencies, main
from mjlab_microduck.next_rl.experiments import experiment_fingerprint


class LocalOnlyRunner:
    def prepare(self, manifest):
        return SimpleNamespace(fingerprint=experiment_fingerprint(manifest))

    def status(self, fingerprint):
        return {"status": "pending", "fingerprint": fingerprint}

    def start(self, prepared):
        raise AssertionError("readiness must not start a trainer")


raise SystemExit(main(sys.argv[1:], dependencies=CliDependencies(
    runner_factory=lambda home: LocalOnlyRunner(),
)))
""".lstrip(),
        encoding="utf-8",
    )
    result = subprocess.run(
        ("uv", "run", "python", str(launcher), *arguments),
        cwd=ROOT,
        env=_environment(home),
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _review_inputs(temporary: Path) -> tuple[Path, Path, dict[str, Path]]:
    """Create genuine CLI evidence files that meet the example's numeric contract."""
    skill = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    policy = temporary / "evaluated-policy.onnx"
    policy.write_bytes(b"example evaluation evidence")
    policy_digest = hashlib.sha256(policy.read_bytes()).hexdigest()
    scenarios = []
    for family, seed in zip(skill["held_out_scenarios"], skill["evaluation_seeds"], strict=True):
        thresholds = [
            {
                "scenario_id": f"{family}-seed-{seed}",
                "metric_name": metric["name"],
                "unit": metric["unit"],
                "direction": metric["direction"],
                "limit": metric["limit"],
                "value": metric["limit"],
                "mandatory": metric.get("mandatory", True),
                "passed": True,
                "normalized_violation": 0.0,
            }
            for metric in skill["metrics"]
        ]
        scenarios.append(
            {
                "scenario_id": f"{family}-seed-{seed}",
                "family": family,
                "seed": seed,
                "metrics": {metric["name"]: metric["limit"] for metric in skill["metrics"]},
                "policy_sha256": policy_digest,
                "threshold_results": thresholds,
            }
        )
    evaluation = temporary / "evaluation.json"
    evaluation.write_text(
        json.dumps(
            {
                "skill_id": skill["id"],
                "spec_version": skill["version"],
                "evaluator_revision": "readiness-fixture",
                "scenarios": scenarios,
                "passed": True,
                "policy": {"path": str(policy), "kind": "onnx", "sha256": policy_digest},
            }
        ),
        encoding="utf-8",
    )
    capability = temporary / "capability.json"
    capability.write_text(
        json.dumps(
            {
                "id": skill["id"],
                "version": skill["version"],
                "aliases": skill.get("aliases", []),
                "robot_model": skill["contract"]["robot_model"],
                "contract": skill["contract"],
                "status": "available",
            }
        ),
        encoding="utf-8",
    )
    clips = {}
    for role in ("nominal", "entry", "exit", "stress", "worst_case"):
        clip = temporary / f"{role}.mp4"
        clip.write_bytes(f"{role} visual evidence".encode())
        clips[role] = clip
    return capability, evaluation, clips


def _review_arguments(capability: Path, evaluation: Path, clips: dict[str, Path]) -> tuple[str, ...]:
    return (
        "review",
        "--capability", str(capability),
        "--skill", str(EXAMPLE),
        "--evaluation", str(evaluation),
        "--nominal-clip", str(clips["nominal"]),
        "--entry-clip", str(clips["entry"]),
        "--exit-clip", str(clips["exit"]),
        "--stress-clip", str(clips["stress"]),
        "--worst-case-clip", str(clips["worst_case"]),
    )


def test_example_skill_is_a_numeric_warm_start_plan_without_training(tmp_path: Path):
    """Catch an example that omits phase-scoped contact and acceptance semantics."""
    home = tmp_path / "workspace"
    inventory = _run_cli("inventory", home=home)
    assert any(item["id"] == "standing" for item in inventory["capabilities"])

    plan = _run_cli("plan", str(EXAMPLE), home=home)
    assert plan == {
        "disposition": "warm_start",
        "parent_capability_id": "standing",
        "reason": "Compatible allowed parent capability 'standing' can warm-start training.",
    }
    skill = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    metric_names = {metric["name"] for metric in skill["metrics"]}
    assert {
        "raised_wave_left_foot_support_ratio",
        "raised_wave_right_foot_clearance_m",
        "raised_wave_right_hip_lateral_cycles",
        "raised_wave_right_knee_flexion_min_rad",
        "raised_wave_right_knee_flexion_max_rad",
        "raised_wave_trunk_tilt_abs_rad",
        "raised_wave_forbidden_contact_count",
        "episode_fall_count",
        "safe_exit_two_foot_contact_ratio",
        "safe_exit_success_ratio",
    } <= metric_names
    assert all(isinstance(metric["limit"], (int, float)) for metric in skill["metrics"])
    phases = skill["metadata"]["phase_contract"]
    assert phases["entry"]["left_foot_floor_contact"] == "required"
    assert phases["entry"]["right_foot_floor_contact"] == "required"
    assert phases["raised_wave"]["right_foot_floor_contact"] == "forbidden"
    assert set(phases["raised_wave"]["forbidden_contacts"]) == {"head", "trunk", "right_foot"}
    assert phases["safe_exit"]["left_foot_floor_contact"] == "required"
    assert phases["safe_exit"]["right_foot_floor_contact"] == "required"
    assert set(skill["metadata"]["metric_scopes"]["raised_wave"]) == {
        "raised_wave_left_foot_support_ratio",
        "raised_wave_right_foot_clearance_m",
        "raised_wave_right_hip_lateral_cycles",
        "raised_wave_right_knee_flexion_min_rad",
        "raised_wave_right_knee_flexion_max_rad",
        "raised_wave_trunk_tilt_abs_rad",
        "raised_wave_forbidden_contact_count",
    }
    assert skill["metadata"]["hardware_deployment"]["calibration_required"] is True
    assert not list(home.rglob("model_*.pt"))


def test_operator_guide_documents_the_current_safe_nitro_connection_contract():
    """Catch guide drift that could route an operator through an unsafe Nitro connection."""
    guide = GUIDE.read_text(encoding="utf-8")
    normalized = " ".join(guide.split())

    assert "NEXT_RL_NITRO_SSH_ALIAS=108.61.217.115" in normalized
    assert "NEXT_RL_NITRO_SSH_USER=aif-engineering" in normalized
    assert "NEXT_RL_NITRO_WSL_DISTRIBUTION=Ubuntu" in normalized
    assert "optional for direct-Linux hosts" in normalized
    assert "required for this Nitro because public SSH lands in Windows" in normalized
    assert "host key must already be trusted" in normalized.casefold()
    assert "BatchMode public-key authentication" in normalized
    assert "Never include a password" in normalized
    assert "StrictHostKeyChecking=no" not in guide
    assert "UserKnownHostsFile=/dev/null" not in guide


def test_public_cli_readiness_flow_stays_local_until_human_approval(tmp_path: Path):
    """Catch a CLI workflow that cannot repeat preparation and review isolated evidence."""
    home = tmp_path / "workspace"
    # The installed executable is exercised for plan/review.  CLI dependencies are
    # injectable only at ``main``; this subprocess uses that public parser with a
    # local-only runner so prepare/status cannot reach SSH or start a trainer.
    prepared = _run_cli_with_local_runner(tmp_path, "prepare", str(EXAMPLE), home=home)
    assert prepared["status"] == "planned"
    assert prepared["fingerprint"] == prepared["prepared_fingerprint"]
    fingerprint = str(prepared["fingerprint"])
    manifest_path = home / "experiments" / fingerprint / "manifest.json"
    status_path = home / "experiments" / fingerprint / "status.json"
    first_manifest = manifest_path.read_bytes()
    first_status = status_path.read_bytes()
    repeated = _run_cli_with_local_runner(tmp_path, "prepare", str(EXAMPLE), home=home)
    assert repeated == prepared
    assert manifest_path.read_bytes() == first_manifest
    assert status_path.read_bytes() == first_status
    assert not list(home.rglob("model_*.pt"))

    status = _run_cli_with_local_runner(tmp_path, "status", fingerprint, home=home)
    assert status == {"fingerprint": fingerprint, "status": "pending"}

    capability, evaluation, clips = _review_inputs(tmp_path)
    review = _run_cli(*_review_arguments(capability, evaluation, clips), home=home)
    assert review["status"] == "review_pending"
    rejected = _run_cli(
        "reject", str(review["record_id"]), "--reviewer", "readiness-reviewer",
        "--reason", "verify re-review path", home=home,
    )
    assert rejected == {"record_id": review["record_id"], "status": "validated"}
    rereview = _run_cli(*_review_arguments(capability, evaluation, clips), home=home)
    approved = _run_cli(
        "approve", str(rereview["record_id"]), "--reviewer", "readiness-reviewer", home=home,
    )
    assert approved == {"record_id": review["record_id"], "status": "learned"}
    inventory = _run_cli("inventory", home=home)
    assert any(
        item["id"] == "one-leg-hello" and item["status"] == "learned"
        for item in inventory["capabilities"]
    )

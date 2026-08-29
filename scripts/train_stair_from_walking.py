"""Fine-tune the stair policy from an existing flat walking checkpoint.

The walking policy and the stair policy use the same 61D actor and 14D action
contract.  mjlab's stock loader resolves checkpoints inside the destination
experiment folder, so this helper stages the selected walking checkpoint in a
small reproducible bootstrap directory before launching the stair fine-tune.

Example::

    uv run python scripts/train_stair_from_walking.py \
      --walking-checkpoint logs/rsl_rl/velocity/<run>/model_XXXX.pt
"""

from __future__ import annotations

import argparse
import copy
import os
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP_DIR_NAME = ".bootstrap-walking"
STAIR_TARGETS = {
    "route": {
        "task": "Mjlab-Stairs-Route-MicroDuck",
        "experiment": "microduck_stair_route",
        "iterations": 10_000,
    },
    "specialist": {
        "task": "Mjlab-Stairs-Specialist-MicroDuck",
        "experiment": "microduck_stair_specialist",
        "iterations": 800,
    },
    "specialist-assisted": {
        "task": "Mjlab-Stairs-Assisted-Specialist-MicroDuck",
        "experiment": "microduck_stair_assisted_specialist",
        "iterations": 100,
    },
    "specialist-bridge": {
        "task": "Mjlab-Stairs-Bridge-Specialist-MicroDuck",
        "experiment": "microduck_stair_bridge_specialist",
        "iterations": 200,
    },
    "specialist-walker-bank": {
        "task": "Mjlab-Stairs-Walker-Bank-Specialist-MicroDuck",
        "experiment": "microduck_stair_walker_bank_specialist",
        "iterations": 100,
    },
    "specialist-launch-bank": {
        "task": "Mjlab-Stairs-Launch-Bank-Specialist-MicroDuck",
        "experiment": "microduck_stair_launch_bank_specialist",
        "iterations": 200,
    },
    "specialist-apex-mantle": {
        "task": "Mjlab-Stairs-Apex-Mantle-Specialist-MicroDuck",
        "experiment": "microduck_stair_apex_mantle_specialist",
        "iterations": 100,
    },
    "specialist-roulade-bank": {
        "task": "Mjlab-Stairs-Roulade-Bank-Specialist-MicroDuck",
        "experiment": "microduck_stair_roulade_bank_specialist",
        "iterations": 150,
    },
    "specialist-curriculum-rsi": {
        "task": "Mjlab-Stairs-Curriculum-RSI-Specialist-MicroDuck",
        "experiment": "microduck_stair_curriculum_rsi_specialist",
        "iterations": 600,
    },
    "specialist-tread-contact-bank": {
        "task": "Mjlab-Stairs-Tread-Contact-Bank-Specialist-MicroDuck",
        "experiment": "microduck_stair_tread_contact_bank_specialist",
        "iterations": 150,
    },
    "specialist-foot-anchor-vault": {
        "task": "Mjlab-Stairs-Foot-Anchor-Vault-Specialist-MicroDuck",
        "experiment": "microduck_stair_foot_anchor_vault_specialist",
        "iterations": 100,
    },
    "specialist-ordered-vault": {
        "task": "Mjlab-Stairs-Ordered-Vault-Specialist-MicroDuck",
        "experiment": "microduck_stair_ordered_vault_specialist",
        "iterations": 100,
    },
    "low": {
        "task": "Mjlab-Stairs-MicroDuck",
        "experiment": "microduck_stairs",
        "iterations": 3_000,
    },
    "standard": {
        "task": "Mjlab-Stairs-Standard-MicroDuck",
        "experiment": "microduck_standard_stairs",
        "iterations": 10_000,
    },
}


def _latest_walking_checkpoint() -> Path | None:
    candidates: list[Path] = []
    for experiment in ("velocity", "microduck_velocity"):
        root = REPO_ROOT / "logs" / "rsl_rl" / experiment
        if root.is_dir():
            candidates.extend(root.glob("*/model_*.pt"))
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def _stage_bootstrap_checkpoint(source: Path, destination: Path) -> None:
    """Copy transferable PPO state while restarting the new task curriculum."""

    import torch

    payload = torch.load(source, map_location="cpu", weights_only=False)
    required = {"actor_state_dict", "critic_state_dict", "optimizer_state_dict"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        missing = sorted(required.difference(payload if isinstance(payload, dict) else ()))
        raise SystemExit(f"Checkpoint is not a compatible rsl_rl PPO snapshot; missing: {missing}")
    payload = copy.deepcopy(payload)
    payload["iter"] = 0
    infos = payload.setdefault("infos", {})
    if isinstance(infos, dict):
        env_state = infos.setdefault("env_state", {})
        if isinstance(env_state, dict):
            env_state["common_step_counter"] = 0
    torch.save(payload, destination)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a MicroDuck stair policy from a compatible locomotion checkpoint."
    )
    parser.add_argument(
        "--walking-checkpoint",
        "--source-checkpoint",
        dest="walking_checkpoint",
        type=Path,
        help="Compatible source model_*.pt. If omitted, use the newest local walking checkpoint.",
    )
    parser.add_argument(
        "--stage",
        choices=tuple(STAIR_TARGETS),
        default="route",
        help=(
            "Use specialist for the fixed 170 mm post-handoff policy; "
            "route is the older shared runway policy."
        ),
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=64,
        help="Parallel training environments. Defaults to 64 so the live viewer and other GPU apps retain headroom.",
    )
    parser.add_argument(
        "--stair-iterations",
        type=int,
        help="Override the selected stage default (3000 low, 10000 standard).",
    )
    parser.add_argument("--run-name", help="Run label. Defaults to from_walking_<stage>.")
    parser.add_argument(
        "--video",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Record simulator videos during training; disabled by default for headless mass evaluation.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    target = STAIR_TARGETS[args.stage]
    iterations = args.stair_iterations or target["iterations"]
    run_name = args.run_name or f"from_walking_{args.stage}"
    checkpoint = args.walking_checkpoint or _latest_walking_checkpoint()
    if checkpoint is None:
        raise SystemExit(
            "No flat walking checkpoint found. Train "
            "Mjlab-Velocity-Flat-MicroDuck first, then pass --walking-checkpoint."
        )
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_file() or checkpoint.suffix.lower() != ".pt":
        raise SystemExit(f"Walking checkpoint is not a .pt file: {checkpoint}")

    experiment_root = REPO_ROOT / "logs" / "rsl_rl" / target["experiment"]
    bootstrap_dir = experiment_root / BOOTSTRAP_DIR_NAME
    bootstrap_dir.mkdir(parents=True, exist_ok=True)
    staged_checkpoint = bootstrap_dir / "model_0.pt"
    _stage_bootstrap_checkpoint(checkpoint, staged_checkpoint)

    venv_train = REPO_ROOT / ".venv" / ("Scripts/train.exe" if os.name == "nt" else "bin/train")
    if venv_train.is_file():
        command = [str(venv_train)]
    else:
        uv = shutil.which("uv") or "uv"
        command = [uv, "run", "train"]
    command.extend([
        target["task"],
        "--env.scene.num-envs",
        str(args.num_envs),
        "--agent.max-iterations",
        str(iterations),
        "--agent.run-name",
        run_name,
        "--agent.load-run",
        BOOTSTRAP_DIR_NAME,
        "--agent.load-checkpoint",
        staged_checkpoint.name,
        "--agent.resume",
        "True",
        "--agent.logger",
        "tensorboard",
        "--agent.upload-model",
        "False",
    ])
    if args.video:
        command.extend(
            [
                "--video",
                "True",
                "--video-length",
                "300",
                "--video-interval",
                "100",
            ]
        )

    environment = os.environ.copy()
    environment.setdefault("WANDB_MODE", "offline")
    print(f"[stair-pipeline] source checkpoint: {checkpoint}")
    print(f"[stair-pipeline] staged bootstrap: {staged_checkpoint}")
    print(f"[stair-pipeline] stage: {args.stage} ({target['task']}, {iterations} iterations)")
    print("[stair-pipeline] launching locomotion-to-stairs fine-tune")
    return subprocess.call(command, cwd=REPO_ROOT, env=environment)


if __name__ == "__main__":
    raise SystemExit(main())

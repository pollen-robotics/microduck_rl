"""Launch the MicroDuck repeated-roll sprint PPO run from a full checkpoint.

The source snapshot is staged into the new experiment directory after resetting
only the iteration and environment-step counters. Actor, critic, normalizers,
distribution, optimizer, and PPO state are otherwise preserved.

Example::

    uv run python scripts/train_roll_sprint.py \
      --source-checkpoint logs/rsl_rl/microduck_stair_stratified_shell_reverse_rsi_specialist/2026-08-30_07-54-17_a35_round1_sol_stratified_hard_1024_gate10/model_9.pt \
      --num-envs 1024 --iterations 4000
"""

from __future__ import annotations

import argparse
import copy
import os
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "microduck_roll_sprint"
BOOTSTRAP_DIR_NAME = ".bootstrap-a35-roll-sprint"
DEFAULT_SOURCE = (
    REPO_ROOT
    / "logs"
    / "rsl_rl"
    / "microduck_stair_stratified_shell_reverse_rsi_specialist"
    / "2026-08-30_07-54-17_a35_round1_sol_stratified_hard_1024_gate10"
    / "model_9.pt"
)


def stage_checkpoint(source: Path, destination: Path) -> None:
    """Copy a compatible PPO snapshot while resetting run counters only."""
    import torch

    payload = torch.load(source, map_location="cpu", weights_only=False)
    required = {"actor_state_dict", "critic_state_dict", "optimizer_state_dict"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        missing = sorted(required.difference(payload if isinstance(payload, dict) else ()))
        raise SystemExit(f"Checkpoint is not a complete PPO snapshot; missing: {missing}")
    payload = copy.deepcopy(payload)
    payload["iter"] = 0
    infos = payload.setdefault("infos", {})
    if isinstance(infos, dict):
        env_state = infos.setdefault("env_state", {})
        if isinstance(env_state, dict):
            env_state["common_step_counter"] = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-checkpoint",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Full compatible PPO checkpoint to warm-start from.",
    )
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=4000)
    parser.add_argument(
        "--save-interval",
        type=int,
        default=100,
        help="Save a checkpoint every N PPO iterations.",
    )
    parser.add_argument("--run-name", default="from_a35_roll_sprint")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    source = args.source_checkpoint.expanduser().resolve()
    if not source.is_file() or source.suffix.lower() != ".pt":
        raise SystemExit(f"Source checkpoint is not a .pt file: {source}")
    if args.num_envs < 1 or args.iterations < 1 or args.save_interval < 1:
        raise SystemExit("--num-envs, --iterations, and --save-interval must be positive")

    experiment_root = REPO_ROOT / "logs" / "rsl_rl" / EXPERIMENT
    staged = experiment_root / BOOTSTRAP_DIR_NAME / "model_0.pt"
    stage_checkpoint(source, staged)

    train_exe = REPO_ROOT / ".venv" / (
        "Scripts/train.exe" if os.name == "nt" else "bin/train"
    )
    command = [str(train_exe)] if train_exe.is_file() else [shutil.which("uv") or "uv", "run", "train"]
    command.extend(
        [
            "Mjlab-Roll-Sprint-Flat-MicroDuck",
            "--env.scene.num-envs",
            str(args.num_envs),
            "--agent.max-iterations",
            str(args.iterations),
            "--agent.save-interval",
            str(args.save_interval),
            "--agent.run-name",
            args.run_name,
            "--agent.load-run",
            BOOTSTRAP_DIR_NAME,
            "--agent.load-checkpoint",
            staged.name,
            "--agent.resume",
            "True",
            "--agent.logger",
            "tensorboard",
            "--agent.upload-model",
            "False",
        ]
    )
    environment = os.environ.copy()
    environment.setdefault("WANDB_MODE", "offline")
    print(f"[roll-sprint] source checkpoint: {source}")
    print(f"[roll-sprint] staged checkpoint: {staged}")
    print(f"[roll-sprint] launching {args.num_envs} envs for {args.iterations} iterations")
    return subprocess.call(command, cwd=REPO_ROOT, env=environment)


if __name__ == "__main__":
    raise SystemExit(main())

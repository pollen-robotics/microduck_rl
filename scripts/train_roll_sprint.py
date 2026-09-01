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
TASK_ID = "Mjlab-Roll-Sprint-Flat-MicroDuck"
DEFAULT_SEED = 42
DEFAULT_SOURCE = (
    REPO_ROOT
    / "logs"
    / "rsl_rl"
    / "microduck_stair_stratified_shell_reverse_rsi_specialist"
    / "2026-08-30_07-54-17_a35_round1_sol_stratified_hard_1024_gate10"
    / "model_9.pt"
)
DIRECTION_CUE_OBS_INDEX = 55


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return parsed


def stage_checkpoint(
    source: Path,
    destination: Path,
    *,
    exploration_std: float | None = None,
    reset_optimizer: bool = False,
    optimizer_learning_rate: float | None = None,
    neutralize_direction_cue: bool = False,
) -> None:
    """Copy a PPO snapshot and optionally reopen exploration for a new skill."""
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
    if exploration_std is not None:
        std_keys = [
            key
            for key in payload["actor_state_dict"]
            if key.endswith("distribution.std_param")
        ]
        if not std_keys:
            raise SystemExit("Checkpoint actor has no distribution.std_param")
        for key in std_keys:
            payload["actor_state_dict"][key].fill_(exploration_std)
    if neutralize_direction_cue:
        cue_columns = 0
        for key, value in payload["actor_state_dict"].items():
            if (
                key.endswith("mlp.0.weight")
                and isinstance(value, torch.Tensor)
                and value.ndim == 2
                and value.shape[1] > DIRECTION_CUE_OBS_INDEX
            ):
                value[:, DIRECTION_CUE_OBS_INDEX] = 0.0
                cue_columns += 1
        if cue_columns == 0:
            raise SystemExit(
                "Checkpoint actor has no compatible first-layer direction cue"
            )
    optimizer = payload["optimizer_state_dict"]
    if reset_optimizer:
        optimizer["state"] = {}
    if optimizer_learning_rate is not None:
        for param_group in optimizer["param_groups"]:
            param_group["lr"] = optimizer_learning_rate
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
    parser.add_argument("--seed", type=_nonnegative_int, default=DEFAULT_SEED)
    parser.add_argument(
        "--learning-rate",
        type=_positive_float,
        default=None,
        help="Optional PPO learning-rate override for conservative champion search.",
    )
    parser.add_argument(
        "--exploration-std",
        type=_positive_float,
        default=None,
        help=(
            "Reset the loaded Gaussian action standard deviation while retaining "
            "the champion mean policy. Useful when learning a genuinely new skill."
        ),
    )
    parser.add_argument(
        "--reset-optimizer",
        action="store_true",
        help="Discard inherited Adam moments while retaining policy and critic weights.",
    )
    parser.add_argument(
        "--neutralize-direction-cue",
        action="store_true",
        help=(
            "Zero the new reverse direction-cue input column during warm-start "
            "staging so an old zero-padded policy keeps its launch behavior."
        ),
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=100,
        help="Save a checkpoint every N PPO iterations.",
    )
    parser.add_argument("--run-name", default="from_a35_roll_sprint")
    parser.add_argument("--task-id", default=TASK_ID)
    parser.add_argument("--experiment-name", default=EXPERIMENT)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    source = args.source_checkpoint.expanduser().resolve()
    if not source.is_file() or source.suffix.lower() != ".pt":
        raise SystemExit(f"Source checkpoint is not a .pt file: {source}")
    if args.num_envs < 1 or args.iterations < 1 or args.save_interval < 1:
        raise SystemExit("--num-envs, --iterations, and --save-interval must be positive")

    experiment_root = REPO_ROOT / "logs" / "rsl_rl" / args.experiment_name
    staged = experiment_root / BOOTSTRAP_DIR_NAME / "model_0.pt"
    stage_checkpoint(
        source,
        staged,
        exploration_std=args.exploration_std,
        reset_optimizer=args.reset_optimizer,
        optimizer_learning_rate=args.learning_rate,
        neutralize_direction_cue=args.neutralize_direction_cue,
    )

    train_exe = REPO_ROOT / ".venv" / (
        "Scripts/train.exe" if os.name == "nt" else "bin/train"
    )
    command = [str(train_exe)] if train_exe.is_file() else [shutil.which("uv") or "uv", "run", "train"]
    command.extend(
        [
            args.task_id,
            "--env.scene.num-envs",
            str(args.num_envs),
            "--agent.max-iterations",
            str(args.iterations),
            "--agent.seed",
            str(args.seed),
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
    if args.learning_rate is not None:
        command.extend(
            ["--agent.algorithm.learning-rate", str(args.learning_rate)]
        )
    environment = os.environ.copy()
    environment.setdefault("WANDB_MODE", "offline")
    print(f"[roll-sprint] source checkpoint: {source}")
    print(f"[roll-sprint] staged checkpoint: {staged}")
    print(f"[roll-sprint] task: {args.task_id}")
    print(f"[roll-sprint] experiment: {args.experiment_name}")
    print(f"[roll-sprint] launching {args.num_envs} envs for {args.iterations} iterations")
    return subprocess.call(command, cwd=REPO_ROOT, env=environment)


if __name__ == "__main__":
    raise SystemExit(main())

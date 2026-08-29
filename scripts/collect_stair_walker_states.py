#!/usr/bin/env python3
"""Collect exact manufacturer-walker states before the fixed 170 mm home stair."""

from __future__ import annotations

import argparse
import hashlib
import os
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from tempfile import NamedTemporaryFile

import mjlab.tasks  # noqa: F401  # Populate the task registry.
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

from mjlab_microduck.policies import load_frozen_actor
from mjlab_microduck.tasks.stair_walk_state_bank import (
    BANK_SCHEMA_VERSION,
    STANDARD_NUM_STEPS,
    STANDARD_RISER_HEIGHT_M,
    STANDARD_TREAD_DEPTH_M,
    capture_walk_state_rows,
    concatenate_walk_state_rows,
    walk_state_count,
)

TASK_ID = "Mjlab-Stairs-Route-MicroDuck"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--walker-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".tmp/codex/full170-walker-state-bank.pt"),
    )
    parser.add_argument("--target-states", type=int, default=256)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--min-local-x", type=float, default=0.56)
    parser.add_argument("--max-local-x", type=float, default=0.64)
    parser.add_argument("--max-steps", type=int, default=8_000)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def _write_bank_atomic(path: Path, bank: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(suffix=".tmp", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(bank, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    args = _parse_args()
    checkpoint = args.walker_checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"Walker checkpoint not found: {checkpoint}")
    if (
        args.target_states < 1
        or args.num_envs < 1
        or args.max_steps < 1
        or args.min_local_x >= args.max_local_x
    ):
        raise SystemExit("State count, environment count, steps, and capture band are invalid")

    configure_torch_backends()
    env_cfg = load_env_cfg(TASK_ID, play=True)
    agent_cfg = load_rl_cfg(TASK_ID)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = 0
    base_env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=args.device)
    walker = load_frozen_actor(runner, checkpoint, device=args.device)

    chunks: list[dict[str, object]] = []
    captured_this_episode = torch.zeros(
        args.num_envs, dtype=torch.bool, device=args.device
    )
    observations = env.get_observations()
    steps_run = 0
    best_local_x = float("-inf")
    best_corridor_x = float("-inf")
    try:
        for steps_run in range(1, args.max_steps + 1):
            with torch.inference_mode():
                walker_observations = observations.clone()
                actor_observations = walker_observations["actor"].clone()
                actor_observations[:, 55:61] = 0.0
                walker_observations["actor"] = actor_observations
                actions = walker(walker_observations)
                observations, _, dones, _ = env.step(actions)

            robot = base_env.scene["robot"]
            origins = base_env.scene.terrain.env_origins
            local = robot.data.root_link_pos_w - origins
            best_local_x = max(best_local_x, float(local[:, 0].max().item()))
            in_corridor = torch.abs(local[:, 1]) <= 0.25
            if torch.any(in_corridor):
                best_corridor_x = max(
                    best_corridor_x,
                    float(local[in_corridor, 0].max().item()),
                )
            eligible = (
                (~captured_this_episode)
                & (dones == 0)
                & (base_env.episode_length_buf > 2)
                & (local[:, 0] >= args.min_local_x)
                & (local[:, 0] <= args.max_local_x)
                & (torch.abs(local[:, 1]) <= 0.25)
            )
            ids = eligible.nonzero(as_tuple=False).squeeze(-1)
            remaining = args.target_states - sum(
                int(chunk["root_qpos_local"].shape[0]) for chunk in chunks
            )
            if len(ids) > remaining:
                ids = ids[:remaining]
            if len(ids) > 0:
                chunks.append(capture_walk_state_rows(base_env, ids))
                captured_this_episode[ids] = True
            captured_this_episode[dones.to(torch.bool)] = False

            collected = sum(
                int(chunk["root_qpos_local"].shape[0]) for chunk in chunks
            )
            if steps_run % 500 == 0 or collected >= args.target_states:
                print(
                    f"[walker-bank] step={steps_run} states={collected}/{args.target_states} "
                    f"best_x={best_local_x:.3f} corridor_x={best_corridor_x:.3f}"
                )
            if collected >= args.target_states:
                break
    finally:
        env.close()

    if not chunks:
        raise SystemExit(
            "The immutable walker never entered the requested capture band; "
            f"best_x={best_local_x:.3f}, corridor_x={best_corridor_x:.3f}"
        )
    states = concatenate_walk_state_rows(chunks)
    count = walk_state_count(states)
    if count < args.target_states:
        raise SystemExit(
            f"Collected only {count}/{args.target_states} states in {steps_run} steps"
        )

    bank: dict[str, object] = {
        "schema_version": BANK_SCHEMA_VERSION,
        "metadata": {
            "created_at": datetime.now(UTC).isoformat(),
            "task": TASK_ID,
            "walker_checkpoint": str(checkpoint),
            "walker_checkpoint_sha256": _sha256(checkpoint),
            "capture_local_x_m": [args.min_local_x, args.max_local_x],
            "riser_height_m": STANDARD_RISER_HEIGHT_M,
            "tread_depth_m": STANDARD_TREAD_DEPTH_M,
            "num_steps": STANDARD_NUM_STEPS,
            "num_states": count,
            "num_envs": args.num_envs,
            "steps_run": steps_run,
            "joint_names": list(base_env.scene["robot"].joint_names),
            "physics_dt": base_env.physics_dt,
            "step_dt": base_env.step_dt,
            "decimation": base_env.cfg.decimation,
            "mjlab_version": version("mjlab"),
            "route_cue_slice_zeroed": [55, 61],
        },
        "states": states,
    }
    _write_bank_atomic(output, bank)
    print(f"[walker-bank] wrote {count} states to {output}")
    print(f"[walker-bank] checkpoint_sha256={bank['metadata']['walker_checkpoint_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

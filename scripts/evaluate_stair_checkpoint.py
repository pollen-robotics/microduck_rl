#!/usr/bin/env python3
"""Evaluate a frozen MicroDuck checkpoint on the full 170 mm stair route.

This is deliberately separate from training.  It runs deterministic play
episodes from the flat runway, records physical route/height progress, and
writes an atomic JSON report that can gate the single-robot live preview.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from tempfile import NamedTemporaryFile

import mjlab.tasks  # noqa: F401  # Populate the task registry.
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

from mjlab_microduck.policies import HardStairHandoffPolicy, load_actor_pair

TASK_ID = "Mjlab-Stairs-Route-MicroDuck"
STANDARD_RISER_HEIGHT_M = 0.170
STAIR_CORRIDOR_HALF_WIDTH_M = 0.90 * 0.40


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w", encoding="utf-8", suffix=".tmp", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Stair specialist checkpoint.")
    parser.add_argument(
        "--walker-checkpoint",
        type=Path,
        help=(
            "Immutable manufacturer walking checkpoint. When provided, walk to "
            "the stair and hand off to the specialist."
        ),
    )
    parser.add_argument(
        "--handoff-local-x",
        type=float,
        default=0.56,
        help="Local x coordinate where the walker-to-specialist handoff starts.",
    )
    parser.add_argument(
        "--handoff-blend-steps",
        type=int,
        default=0,
        help="Control frames used to blend actors. Zero preserves the hard switch.",
    )
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument(
        "--steps",
        type=int,
        default=1_500,
        help="Control steps. 1500 is one complete 30 second route episode.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")
    walker_checkpoint = (
        args.walker_checkpoint.expanduser().resolve()
        if args.walker_checkpoint is not None
        else None
    )
    if walker_checkpoint is not None and not walker_checkpoint.is_file():
        raise SystemExit(f"Walker checkpoint not found: {walker_checkpoint}")
    if args.num_envs < 1 or args.steps < 1 or args.handoff_blend_steps < 0:
        raise SystemExit("Environment counts and steps must be positive; blend steps cannot be negative")

    configure_torch_backends()
    env_cfg = load_env_cfg(TASK_ID, play=True)
    agent_cfg = load_rl_cfg(TASK_ID)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = 0

    base_env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=args.device)
    if walker_checkpoint is not None:
        walker, specialist = load_actor_pair(
            runner,
            walker_checkpoint,
            checkpoint,
            device=args.device,
        )
        policy = HardStairHandoffPolicy(
            walker,
            specialist,
            base_env,
            switch_local_x_m=args.handoff_local_x,
            blend_steps=args.handoff_blend_steps,
        )
        policy_mode = (
            "hard_walker_to_specialist_handoff"
            if args.handoff_blend_steps == 0
            else "blended_walker_to_specialist_handoff"
        )
    else:
        runner.load(
            str(checkpoint),
            load_cfg={"actor": True},
            strict=True,
            map_location=args.device,
        )
        policy = runner.get_inference_policy(device=args.device)
        policy_mode = "single_actor"

    device = torch.device(args.device)
    max_x = torch.full((args.num_envs,), -torch.inf, device=device)
    max_z = torch.full((args.num_envs,), -torch.inf, device=device)
    max_any_x = torch.full((args.num_envs,), -torch.inf, device=device)
    max_any_z = torch.full((args.num_envs,), -torch.inf, device=device)
    max_abs_y = torch.zeros(args.num_envs, device=device)
    max_upright = torch.zeros(args.num_envs, device=device)
    success_count = torch.zeros(args.num_envs, dtype=torch.long, device=device)
    previous_latch = torch.zeros(args.num_envs, dtype=torch.bool, device=device)

    try:
        for _ in range(args.steps):
            with torch.inference_mode():
                observations = env.get_observations()
                actions = policy(observations)
                env.step(actions)

            robot = base_env.scene["robot"]
            origins = base_env.scene.terrain.env_origins
            root_pos = robot.data.root_link_pos_w
            local_x = root_pos[:, 0] - origins[:, 0]
            local_y = root_pos[:, 1] - origins[:, 1]
            local_z = root_pos[:, 2] - origins[:, 2]
            quat = robot.data.root_link_quat_w
            upright = torch.clamp(
                1.0 - 2.0 * (quat[:, 1].square() + quat[:, 2].square()),
                min=0.0,
                max=1.0,
            )
            finite_x = torch.nan_to_num(local_x, nan=-torch.inf)
            finite_z = torch.nan_to_num(local_z, nan=-torch.inf)
            abs_y = torch.abs(torch.nan_to_num(local_y, nan=1.0e9))
            in_corridor = abs_y <= STAIR_CORRIDOR_HALF_WIDTH_M
            max_x = torch.maximum(
                max_x, torch.where(in_corridor, finite_x, -torch.inf)
            )
            max_z = torch.maximum(
                max_z, torch.where(in_corridor, finite_z, -torch.inf)
            )
            max_any_x = torch.maximum(max_any_x, finite_x)
            max_any_z = torch.maximum(max_any_z, finite_z)
            max_abs_y = torch.maximum(max_abs_y, abs_y)
            max_upright = torch.maximum(max_upright, torch.nan_to_num(upright, nan=0.0))

            latch = getattr(base_env, "_stair_goal_latched", previous_latch)
            success_count += (latch & ~previous_latch).to(torch.long)
            previous_latch = latch.clone()
            fresh = base_env.episode_length_buf <= 1
            previous_latch[fresh] = False
    finally:
        env.close()

    successes = int(success_count.sum().item())
    report: dict[str, object] = {
        "schema_version": 3,
        "task": TASK_ID,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_iteration": int(checkpoint.stem.rsplit("_", 1)[-1]),
        "walker_checkpoint": str(walker_checkpoint) if walker_checkpoint else None,
        "walker_checkpoint_sha256": (
            _sha256(walker_checkpoint) if walker_checkpoint else None
        ),
        "policy_mode": policy_mode,
        "handoff_local_x_m": (
            policy.switch_local_x_m
            if isinstance(policy, HardStairHandoffPolicy)
            else None
        ),
        "handoff_count": (
            policy.handoff_count
            if isinstance(policy, HardStairHandoffPolicy)
            else 0
        ),
        "handoff_blend_steps": (
            policy.blend_steps if isinstance(policy, HardStairHandoffPolicy) else None
        ),
        "standard_riser_height_m": STANDARD_RISER_HEIGHT_M,
        "stair_corridor_half_width_m": STAIR_CORRIDOR_HALF_WIDTH_M,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "successes": successes,
        "success_rate": successes / args.num_envs,
        "mean_max_route_x_m": float(max_x.mean().item()),
        "best_route_x_m": float(max_x.max().item()),
        "mean_max_corridor_route_x_m": float(max_x.mean().item()),
        "best_corridor_route_x_m": float(max_x.max().item()),
        "mean_max_root_height_m": float(max_z.mean().item()),
        "best_root_height_m": float(max_z.max().item()),
        "best_any_route_x_m": float(max_any_x.max().item()),
        "best_any_root_height_m": float(max_any_z.max().item()),
        "maximum_abs_lateral_offset_m": float(max_abs_y.max().item()),
        "mean_max_upright": float(max_upright.mean().item()),
        "verified_full_route": successes > 0,
    }
    output = args.output or checkpoint.with_suffix(".stair-eval.json")
    _write_json_atomic(output.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"[stair-eval] wrote {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Measure Stage-A clearance on full-height assisted stair reset states.

This report is a curriculum gate only. It must never promote a dashboard video;
strict full-route promotion remains in evaluate_stair_checkpoint.py.
"""

from __future__ import annotations

import argparse
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

TASK_ID = "Mjlab-Stairs-Assisted-Specialist-MicroDuck"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w", encoding="utf-8", suffix=".tmp", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main() -> int:
    args = _parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")
    if args.num_envs < 1 or args.episodes < 1:
        raise SystemExit("--num-envs and --episodes must be positive")

    configure_torch_backends()
    env_cfg = load_env_cfg(TASK_ID, play=True)
    agent_cfg = load_rl_cfg(TASK_ID)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = 7
    base_env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=args.device)
    runner.load(
        str(checkpoint),
        load_cfg={"actor": True},
        strict=True,
        map_location=args.device,
    )
    policy = runner.get_inference_policy(device=args.device)

    clearance_events = 0
    stable_events = 0
    completed_trials = 0
    previous_clearance = torch.zeros(
        args.num_envs, dtype=torch.bool, device=args.device
    )
    previous_stable = torch.zeros_like(previous_clearance)
    max_x = torch.full((args.num_envs,), -torch.inf, device=args.device)
    max_z = torch.full_like(max_x, -torch.inf)
    steps = base_env.max_episode_length * args.episodes
    try:
        for _ in range(steps):
            with torch.inference_mode():
                observations = env.get_observations()
                actions = policy(observations)
                _, _, dones, _ = env.step(actions)

            clearance = getattr(
                base_env, "_stair_first_riser_latched", previous_clearance
            )
            stable = getattr(base_env, "_stair_first_tread_latched", previous_stable)
            clearance_events += int((clearance & ~previous_clearance).sum().item())
            stable_events += int((stable & ~previous_stable).sum().item())
            completed_trials += int(dones.sum().item())
            previous_clearance = clearance.clone()
            previous_stable = stable.clone()
            previous_clearance[dones.bool()] = False
            previous_stable[dones.bool()] = False

            robot = base_env.scene["robot"]
            origins = base_env.scene.terrain.env_origins
            local = robot.data.root_link_pos_w - origins
            max_x = torch.maximum(max_x, torch.nan_to_num(local[:, 0], nan=-torch.inf))
            max_z = torch.maximum(max_z, torch.nan_to_num(local[:, 2], nan=-torch.inf))
    finally:
        env.close()

    denominator = max(completed_trials, 1)
    report: dict[str, object] = {
        "schema_version": 1,
        "task": TASK_ID,
        "checkpoint": str(checkpoint),
        "checkpoint_iteration": int(checkpoint.stem.rsplit("_", 1)[-1]),
        "standard_riser_height_m": 0.17,
        "standard_tread_depth_m": 0.28,
        "num_envs": args.num_envs,
        "requested_episodes_per_env": args.episodes,
        "completed_trials": completed_trials,
        "clearance_events": clearance_events,
        "clearance_rate": clearance_events / denominator,
        "stable_tread_events": stable_events,
        "stable_tread_rate": stable_events / denominator,
        "best_route_x_m": float(max_x.max().item()),
        "best_root_height_m": float(max_z.max().item()),
        "promotion_eligible": False,
    }
    output = args.output or checkpoint.with_suffix(".assisted-eval.json")
    _write_json_atomic(output.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"[assisted-stair-eval] wrote {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

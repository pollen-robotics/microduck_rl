#!/usr/bin/env python3
"""Show one robot using the frozen walker and best stair specialist."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import mjlab.tasks  # noqa: F401  # Populate the task registry.
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer

from mjlab_microduck.policies import HardStairHandoffPolicy, load_actor_pair

TASK_ID = "Mjlab-Stairs-Route-MicroDuck"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("specialist_checkpoint", type=Path)
    parser.add_argument("--walker-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    specialist_checkpoint = args.specialist_checkpoint.expanduser().resolve()
    walker_checkpoint = args.walker_checkpoint.expanduser().resolve()
    for label, checkpoint in (
        ("Specialist", specialist_checkpoint),
        ("Walker", walker_checkpoint),
    ):
        if not checkpoint.is_file():
            raise SystemExit(f"{label} checkpoint not found: {checkpoint}")

    configure_torch_backends()
    env_cfg = load_env_cfg(TASK_ID, play=True)
    agent_cfg = load_rl_cfg(TASK_ID)
    env_cfg.scene.num_envs = 1
    env_cfg.seed = 0

    base_env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=args.device)
    walker, specialist = load_actor_pair(
        runner,
        walker_checkpoint,
        specialist_checkpoint,
        device=args.device,
    )
    policy = HardStairHandoffPolicy(walker, specialist, base_env)

    try:
        with torch.inference_mode():
            NativeMujocoViewer(env, policy).run()
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Record fixed left/right single-leg rollouts from a checkpoint."""

import argparse
from dataclasses import asdict
from pathlib import Path

import torch
from rsl_rl.runners import OnPolicyRunner

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.wrappers import VideoRecorder


TASK = "Mjlab-SingleLegStand-Flat-MicroDuck"


def record(
    checkpoint: Path,
    output_dir: Path,
    side_name: str,
    video_length: int,
    device: str,
    width: int,
    height: int,
) -> Path:
    side = -1 if side_name == "left" else 1
    env_cfg = load_env_cfg(TASK, play=True)
    env_cfg.scene.num_envs = 1
    env_cfg.commands["twist"].fixed_side = side
    env_cfg.viewer.distance = 0.45
    env_cfg.viewer.elevation = -10.0
    env_cfg.viewer.width = width
    env_cfg.viewer.height = height
    agent_cfg = load_rl_cfg(TASK)

    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
    recorded_env = VideoRecorder(
        raw_env,
        video_folder=output_dir,
        step_trigger=lambda step: step == 0,
        video_length=video_length,
        name_prefix=f"single-leg-{side_name}-support",
    )
    env = RslRlVecEnvWrapper(recorded_env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(TASK) or OnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
        str(checkpoint),
        load_cfg={"actor": True},
        strict=True,
        map_location=device,
    )
    policy = runner.get_inference_policy(device=device)
    obs = env.get_observations()

    with torch.inference_mode():
        for _ in range(video_length):
            obs, _, _, _ = env.step(policy(obs))
    env.close()
    return output_dir / f"single-leg-{side_name}-support-step-0.mp4"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--side",
        choices=("left", "right", "both"),
        default="both",
        help="Commanded support side to record.",
    )
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")
    if args.seconds <= 0.0:
        parser.error("--seconds must be positive")

    sides = ("left", "right") if args.side == "both" else (args.side,)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    env_cfg = load_env_cfg(TASK, play=True)
    step_dt = env_cfg.decimation * env_cfg.sim.mujoco.timestep
    video_length = max(1, round(args.seconds / step_dt))

    for side_name in sides:
        path = record(
            args.checkpoint,
            args.output_dir,
            side_name,
            video_length,
            args.device,
            args.width,
            args.height,
        )
        print(path.resolve())


if __name__ == "__main__":
    main()

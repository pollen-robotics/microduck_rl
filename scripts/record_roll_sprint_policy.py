#!/usr/bin/env python3
"""Record four standing-start roll-sprint rollouts in one tiled video."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

import mjlab.tasks  # noqa: F401  # Populate the task registry.
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

from mjlab_microduck.tasks import mdp as microduck_mdp

TASK_ID = "Mjlab-Roll-Sprint-Flat-MicroDuck"
REPO_ROOT = Path(__file__).resolve().parents[1]
RACE_LANE_SPACING = 0.28
RACE_CAMERA_LOOKAT = (0.60, 0.0, 0.08)
RACE_CAMERA_DISTANCE = 2.8
RACE_CAMERA_AZIMUTH = 135.0
RACE_CAMERA_ELEVATION = -55.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--task-id", default=TASK_ID)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    return parser.parse_args()


def _as_rgb8(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 4:
        frame = frame[0]
    if frame.dtype != np.uint8:
        frame = (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)
    return np.ascontiguousarray(frame[:, :, :3])


def _ffmpeg_writer(
    output: Path, *, width: int, height: int, fps: float
) -> subprocess.Popen[bytes]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to record roll-sprint videos")
    output.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s:v",
            f"{width}x{height}",
            "-r",
            f"{fps:g}",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "21",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        stdin=subprocess.PIPE,
    )


def _race_lane_origins(
    num_lanes: int,
    lane_spacing: float,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return equal-width parallel lanes with one shared +x start line."""
    lane_indices = torch.arange(num_lanes, device=device, dtype=dtype)
    lane_indices -= (num_lanes - 1) / 2.0
    origins = torch.zeros((num_lanes, 3), device=device, dtype=dtype)
    origins[:, 1] = lane_indices * lane_spacing
    return origins


def _arrange_race_start(
    base_env: ManagerBasedRlEnv,
    lane_spacing: float = RACE_LANE_SPACING,
) -> None:
    """Place all robots on one deterministic start line facing world +x."""
    robot = base_env.scene["robot"]
    old_origins = base_env.scene.terrain.env_origins.clone()
    origins = _race_lane_origins(
        base_env.num_envs,
        lane_spacing,
        device=base_env.device,
        dtype=old_origins.dtype,
    )
    pose = torch.cat(
        (
            robot.data.root_link_pos_w.clone(),
            robot.data.root_link_quat_w.clone(),
        ),
        dim=-1,
    )
    local_height = pose[:, 2] - old_origins[:, 2]
    pose[:, :3] = origins
    pose[:, 2] += local_height
    pose[:, 3:] = 0.0
    pose[:, 3] = 1.0
    velocity = torch.zeros(
        (base_env.num_envs, 6), device=base_env.device, dtype=pose.dtype
    )

    base_env.scene.terrain.env_origins.copy_(origins)
    robot.write_root_link_pose_to_sim(pose)
    robot.write_root_link_velocity_to_sim(velocity)
    base_env.sim.forward()
    env_ids = torch.arange(base_env.num_envs, device=base_env.device)
    microduck_mdp._reset_roll_sprint_buffers(base_env, env_ids)


def main() -> int:
    args = _parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    temporary_dir = REPO_ROOT / ".tmp" / "codex"
    temporary_dir.mkdir(parents=True, exist_ok=True)
    temporary_output = temporary_dir / f"roll-sprint-recording-{os.getpid()}.mp4"
    if not checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")
    if args.steps < 1 or args.width < 2 or args.height < 2:
        raise SystemExit("--steps, --width, and --height must be positive")

    configure_torch_backends()
    env_cfg = load_env_cfg(args.task_id, play=True)
    agent_cfg = load_rl_cfg(args.task_id)
    env_cfg.scene.num_envs = 4
    env_cfg.scene.env_spacing = RACE_LANE_SPACING
    env_cfg.scene.terrain.env_spacing = RACE_LANE_SPACING
    env_cfg.seed = args.seed
    env_cfg.auto_reset = False
    policy_dt = env_cfg.sim.mujoco.timestep * env_cfg.decimation
    env_cfg.episode_length_s = max(6.0, args.steps * policy_dt)
    env_cfg.viewer.origin_type = type(env_cfg.viewer).OriginType.WORLD
    env_cfg.viewer.lookat = RACE_CAMERA_LOOKAT
    env_cfg.viewer.distance = RACE_CAMERA_DISTANCE
    env_cfg.viewer.azimuth = RACE_CAMERA_AZIMUTH
    env_cfg.viewer.elevation = RACE_CAMERA_ELEVATION
    env_cfg.viewer.max_extra_envs = 3
    env_cfg.viewer.width = args.width
    env_cfg.viewer.height = args.height
    reset_cfg = env_cfg.events["set_roll_sprint_state"]
    reset_cfg.params["standing_prob"] = 1.0
    reset_cfg.params["midroll_prob"] = 0.0
    reset_cfg.params["yaw_range"] = (0.0, 0.0)

    base_env = ManagerBasedRlEnv(
        cfg=env_cfg,
        device=args.device,
        render_mode="rgb_array",
    )
    _arrange_race_start(base_env)
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(args.task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=args.device)
    runner.load(
        str(checkpoint),
        load_cfg={"actor": True},
        strict=True,
        map_location=args.device,
    )
    policy = runner.get_inference_policy(device=args.device)

    writer: subprocess.Popen[bytes] | None = None
    try:
        for _ in range(args.steps):
            with torch.inference_mode():
                observations = env.get_observations()
                actions = policy(observations)
                env.step(actions)
            rendered = base_env.render()
            if rendered is None:
                raise RuntimeError("MuJoCo returned no RGB frame")
            frame = _as_rgb8(rendered)
            if writer is None:
                frame_height, frame_width = frame.shape[:2]
                writer = _ffmpeg_writer(
                    temporary_output,
                    width=frame_width,
                    height=frame_height,
                    fps=float(base_env.metadata.get("render_fps", 50)),
                )
            assert writer.stdin is not None
            writer.stdin.write(frame.tobytes())
    finally:
        env.close()
        if writer is not None and writer.stdin is not None:
            writer.stdin.close()

    if writer is None:
        raise RuntimeError("No frames were recorded")
    return_code = writer.wait()
    if return_code != 0 or not temporary_output.is_file():
        raise RuntimeError(f"ffmpeg failed with exit code {return_code}")
    output.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary_output, output)
    print(f"[roll-sprint-video] wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

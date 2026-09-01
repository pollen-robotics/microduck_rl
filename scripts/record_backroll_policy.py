#!/usr/bin/env python3
"""Record a label-free one-robot proof of a successful grounded backroll."""

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

TASK_ID = "Mjlab-Backroll-Flat-MicroDuck"
REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_WIDTH = 1920
OUTPUT_HEIGHT = 1080
OUTPUT_FPS = 60.0
POST_SUCCESS_HOLD_S = 0.75


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of deterministic variants to simulate in parallel.",
    )
    parser.add_argument(
        "--env-index",
        type=int,
        default=0,
        help="Variant index to follow, render, and validate within the batch.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--width", type=int, default=OUTPUT_WIDTH)
    parser.add_argument("--height", type=int, default=OUTPUT_HEIGHT)
    parser.add_argument(
        "--allow-incomplete-diagnostic",
        action="store_true",
        help=(
            "Publish a clearly named diagnostic even if the physical success "
            "gate is not reached. Proof recordings still require success by default."
        ),
    )
    return parser.parse_args()


def _as_rgb8(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 4:
        frame = frame[0]
    if frame.dtype != np.uint8:
        frame = (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)
    return np.ascontiguousarray(frame[:, :, :3])


def _ffmpeg_writer(
    output: Path,
    *,
    width: int,
    height: int,
    input_fps: float,
) -> subprocess.Popen[bytes]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to record grounded-backroll video")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
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
        f"{input_fps:g}",
        "-i",
        "pipe:0",
        "-an",
        "-r",
        f"{OUTPUT_FPS:g}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "17",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def _load_policy(base_env: ManagerBasedRlEnv, checkpoint: Path, *, device: str):
    agent_cfg = load_rl_cfg(TASK_ID)
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
        str(checkpoint),
        load_cfg={"actor": True},
        strict=True,
        map_location=device,
    )
    return env, runner.get_inference_policy(device=device)


def main() -> int:
    args = _parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")
    if args.duration <= 0.0 or args.width < 2 or args.height < 2:
        raise SystemExit("--duration, --width, and --height must be positive")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")
    if args.env_index < 0 or args.env_index >= args.batch_size:
        raise SystemExit("--env-index must be within [0, --batch-size)")

    temporary_dir = REPO_ROOT / ".tmp" / "codex"
    temporary_dir.mkdir(parents=True, exist_ok=True)
    temporary_output = temporary_dir / f"grounded-backroll-{os.getpid()}.mp4"

    configure_torch_backends()
    env_cfg = load_env_cfg(TASK_ID, play=True)
    env_cfg.scene.num_envs = args.batch_size
    env_cfg.seed = args.seed
    env_cfg.auto_reset = False
    env_cfg.episode_length_s = args.duration
    if args.allow_incomplete_diagnostic:
        env_cfg.terminations.clear()
    # Follow the trunk so a policy that translates during an incomplete attempt
    # cannot walk into the camera or leave the frame.  The camera remains a
    # stable three-quarter view; only its target follows the robot.
    env_cfg.viewer.origin_type = type(env_cfg.viewer).OriginType.ASSET_BODY
    env_cfg.viewer.entity_name = "robot"
    env_cfg.viewer.body_name = "trunk_base"
    env_cfg.viewer.env_idx = args.env_index
    env_cfg.viewer.lookat = (0.0, 0.0, 0.10)
    env_cfg.viewer.distance = 0.80
    env_cfg.viewer.fovy = 35.0
    env_cfg.viewer.azimuth = 135.0
    env_cfg.viewer.elevation = -18.0
    env_cfg.viewer.max_extra_envs = 0
    env_cfg.viewer.width = args.width
    env_cfg.viewer.height = args.height
    reset_cfg = env_cfg.events["set_grounded_backroll_state"]
    reset_cfg.params.update(
        standing_prob=1.0,
        midroll_prob=0.0,
        yaw_range=(0.0, 0.0),
        joint_noise_std=0.02,
    )

    base_env = ManagerBasedRlEnv(
        cfg=env_cfg,
        device=args.device,
        render_mode="rgb_array",
    )
    env, policy = _load_policy(base_env, checkpoint, device=args.device)
    policy_fps = 1.0 / base_env.step_dt
    steps = round(args.duration / base_env.step_dt)
    hold_frames = round(POST_SUCCESS_HOLD_S * policy_fps)
    writer: subprocess.Popen[bytes] | None = None
    last_frame: np.ndarray | None = None
    success = False

    try:
        for _ in range(steps):
            with torch.inference_mode():
                observations = env.get_observations()
                actions = policy(observations)
                env.step(actions)
            rendered = base_env.render()
            if rendered is None:
                raise RuntimeError("MuJoCo returned no RGB frame")
            last_frame = _as_rgb8(rendered)
            if writer is None:
                frame_height, frame_width = last_frame.shape[:2]
                writer = _ffmpeg_writer(
                    temporary_output,
                    width=frame_width,
                    height=frame_height,
                    input_fps=policy_fps,
                )
            assert writer.stdin is not None
            writer.stdin.write(last_frame.tobytes())
            if bool(base_env._backroll_success[args.env_index].item()):
                success = True
                break
            if (
                not args.allow_incomplete_diagnostic
                and bool(base_env._backroll_invalid[args.env_index].item())
            ):
                break

        if success and writer is not None and writer.stdin is not None:
            assert last_frame is not None
            for _ in range(hold_frames):
                writer.stdin.write(last_frame.tobytes())
    finally:
        env.close()
        if writer is not None and writer.stdin is not None:
            writer.stdin.close()

    if writer is None:
        raise RuntimeError("No frames were recorded")
    return_code = writer.wait()
    if not success and not args.allow_incomplete_diagnostic:
        temporary_output.unlink(missing_ok=True)
        raise SystemExit(
            "No proof video was published because this standing trial did not "
            "pass every grounded-backroll gate."
        )
    if return_code != 0 or not temporary_output.is_file():
        raise RuntimeError(f"ffmpeg failed with exit code {return_code}")
    output.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary_output, output)
    kind = "proof" if success else "incomplete diagnostic"
    print(f"[grounded-backroll-video] wrote {kind}: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

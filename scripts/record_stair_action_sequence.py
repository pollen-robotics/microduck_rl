#!/usr/bin/env python3
"""Record a frozen A13 loaded-state stair action sequence.

This is an honest search-artifact replay, not a full-route climb. Each attempt
resets the ordered-vault environment to one exact walker-state-bank row, runs
the frozen baseline actor plus the saved ``best_expanded`` residual sequence,
and then repeats until the requested video duration is filled.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
from search_stair_action_sequence import (
    DEFAULT_ACTION_LIMIT,
    DEFAULT_BASELINE_CHECKPOINT,
    TASK_ID,
    _pin_loaded_foot_bank_row,
    _validate_fixed_stair,
    force_assisted_reset_mode,
    pin_bank_reset_position,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEQUENCE = (
    REPO_ROOT / "artifacts/a13-trajectory-search/a13-cem-row87-refine.npz"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task-id",
        default=TASK_ID,
        help="Registered fixed-stair specialist task used by the source search.",
    )
    parser.add_argument(
        "--forced-reset-mode",
        choices=("task-default", "launch-release", "head-lever", "bank"),
        default="task-default",
    )
    parser.add_argument(
        "--baseline-checkpoint", type=Path, default=DEFAULT_BASELINE_CHECKPOINT
    )
    parser.add_argument("--sequence-npz", type=Path, default=DEFAULT_SEQUENCE)
    parser.add_argument("--bank-row", type=int, default=87)
    parser.add_argument("--bank-local-x", type=float)
    parser.add_argument("--bank-local-y", type=float)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration-seconds", type=float, default=20.0)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument(
        "--action-limit",
        type=float,
        default=DEFAULT_ACTION_LIMIT,
        help="Total actor-plus-residual clamp used by the source A13 search.",
    )
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    return parser.parse_args()


def load_residual_sequence(path: Path, *, action_dim: int = 14) -> np.ndarray:
    """Load and validate the exact expanded residual sequence from A13."""

    source = path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"A13 sequence NPZ not found: {source}")
    with np.load(source, allow_pickle=False) as archive:
        if "best_expanded" not in archive:
            raise ValueError(f"A13 NPZ has no best_expanded array: {source}")
        sequence = np.asarray(archive["best_expanded"], dtype=np.float32)
    if sequence.ndim != 2 or sequence.shape[1] != action_dim:
        raise ValueError(
            f"best_expanded must have shape (steps, {action_dim}), got {sequence.shape}"
        )
    if sequence.shape[0] < 1 or not np.isfinite(sequence).all():
        raise ValueError("best_expanded must be non-empty and finite")
    return sequence


def video_frame_count(duration_seconds: float, fps: float) -> int:
    """Return the exact integer frame count for a fixed-rate video."""

    if not np.isfinite(duration_seconds) or duration_seconds <= 0.0:
        raise ValueError("duration_seconds must be positive and finite")
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError("fps must be positive and finite")
    frames = round(duration_seconds * fps)
    if frames < 1:
        raise ValueError("duration_seconds * fps must produce at least one frame")
    return frames


def attempt_step_indices(frame_count: int, sequence_steps: int) -> np.ndarray:
    """Map output frames to repeated frozen-sequence steps."""

    if frame_count < 1 or sequence_steps < 1:
        raise ValueError("frame_count and sequence_steps must be positive")
    return np.arange(frame_count, dtype=np.int64) % sequence_steps


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
        raise RuntimeError("ffmpeg is required to record A13 videos")
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
            "-metadata",
            "title=MicroDuck A13 loaded-state near-lip search replay",
            "-metadata",
            "comment=Frozen baseline plus CEM residual; no verified stair clearance",
            str(output),
        ],
        stdin=subprocess.PIPE,
    )


def main() -> int:
    args = _parse_args()
    checkpoint = args.baseline_checkpoint.expanduser().resolve()
    sequence_path = args.sequence_npz.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"Baseline checkpoint not found: {checkpoint}")
    if args.bank_row < 0:
        raise SystemExit("--bank-row must be non-negative")
    if args.width < 2 or args.height < 2:
        raise SystemExit("--width and --height must be at least 2")
    if not np.isfinite(args.action_limit) or args.action_limit <= 0.0:
        raise SystemExit("--action-limit must be positive and finite")
    try:
        total_frames = video_frame_count(args.duration_seconds, args.fps)
        residual = load_residual_sequence(sequence_path)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error

    import mjlab.tasks  # noqa: F401  # Populate base registry.
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.utils.torch import configure_torch_backends

    import mjlab_microduck.tasks  # noqa: F401  # Register MicroDuck tasks.
    from mjlab_microduck.policies.stair_handoff import load_frozen_actor

    configure_torch_backends()
    torch.manual_seed(args.seed)
    env_cfg = load_env_cfg(args.task_id, play=True)
    agent_cfg = load_rl_cfg(args.task_id)
    force_assisted_reset_mode(env_cfg, args.forced_reset_mode)
    pin_bank_reset_position(env_cfg, args.bank_local_x, args.bank_local_y)
    _validate_fixed_stair(env_cfg)
    env_cfg.scene.num_envs = 1
    env_cfg.seed = args.seed
    env_cfg.viewer.width = args.width
    env_cfg.viewer.height = args.height
    env_cfg.viewer.distance = 1.25
    env_cfg.viewer.elevation = -8.0

    base_env = ManagerBasedRlEnv(
        cfg=env_cfg,
        device=args.device,
        render_mode="rgb_array",
    )
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
    temporary_dir = REPO_ROOT / ".tmp" / "codex"
    temporary_dir.mkdir(parents=True, exist_ok=True)
    temporary_output = temporary_dir / f"a13-stair-recording-{os.getpid()}.mp4"
    writer: subprocess.Popen[bytes] | None = None
    try:
        selected_row = None
        if args.forced_reset_mode in {"task-default", "bank"}:
            selected_row = _pin_loaded_foot_bank_row(
                base_env, args.bank_row, torch
            )
        runner_cls = load_runner_cls(args.task_id) or MjlabOnPolicyRunner
        runner = runner_cls(env, asdict(agent_cfg), device=args.device)
        baseline_actor = load_frozen_actor(runner, checkpoint, device=args.device)
        if int(env.num_actions) != residual.shape[1]:
            raise RuntimeError(
                f"Environment has {env.num_actions} actions, sequence has {residual.shape[1]}"
            )
        residual_tensor = torch.as_tensor(residual, device=base_env.device)
        indices = attempt_step_indices(total_frames, len(residual))
        observations = None
        attempts = 0
        for frame_index, sequence_step in enumerate(indices):
            if sequence_step == 0:
                with torch.inference_mode():
                    observations, _ = env.reset()
                attempts += 1
            assert observations is not None
            with torch.inference_mode():
                actions = baseline_actor(observations) + residual_tensor[sequence_step]
                actions = torch.clamp(
                    actions, -args.action_limit, args.action_limit
                )
                observations, _, _, _ = env.step(actions)
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
                    fps=args.fps,
                )
            assert writer.stdin is not None
            writer.stdin.write(frame.tobytes())
            if (frame_index + 1) % max(1, int(args.fps * 5)) == 0:
                print(
                    f"[a13-video] {frame_index + 1}/{total_frames} frames",
                    flush=True,
                )
        print(
            f"[a13-video] replayed reset={args.forced_reset_mode} "
            f"bank_row={selected_row} across {attempts} attempts",
            flush=True,
        )
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
    print(f"[a13-video] wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

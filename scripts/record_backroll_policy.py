#!/usr/bin/env python3
"""Record a label-free one-robot proof of a successful grounded backroll."""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import mjlab.tasks  # noqa: F401  # Populate the task registry.
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

TASK_ID = "Mjlab-Backroll-Flat-MicroDuck"
REPEATED_TASK_ID = "Mjlab-Repeated-Backroll-Flat-MicroDuck"
REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_WIDTH = 1920
OUTPUT_HEIGHT = 1080
OUTPUT_FPS = 60.0
POST_SUCCESS_HOLD_S = 0.75
# A stalled diagnostic must be long enough to show the complete failed attempt
# but never waste a 20-second dashboard slot on a motionless robot.  Recovery
# or a new roll frontier resets the separate idle timer below, so active retry
# behavior remains visible for the full configured duration.
DIAGNOSTIC_MIN_SECONDS = 6.0
DIAGNOSTIC_STUCK_SECONDS = 1.0
DIAGNOSTIC_FRONTIER_EPSILON_RAD = math.radians(3.0)
DIAGNOSTIC_ACTIVE_ANGULAR_RATE = 0.75
DIAGNOSTIC_ACTIVE_VERTICAL_RATE = 0.03


@dataclass
class DiagnosticStuckDetector:
    """Cut an incomplete rollout only after meaningful body motion has ended."""

    min_steps: int
    patience_steps: int
    frontier_epsilon_rad: float = DIAGNOSTIC_FRONTIER_EPSILON_RAD
    active_angular_rate: float = DIAGNOSTIC_ACTIVE_ANGULAR_RATE
    active_vertical_rate: float = DIAGNOSTIC_ACTIVE_VERTICAL_RATE
    best_frontier_rad: float = 0.0
    last_cycle_count: int = 0
    last_activity_step: int = 0

    def update(
        self,
        *,
        step: int,
        frontier_rad: float,
        cycle_count: int,
        angular_speed: float,
        vertical_speed: float,
    ) -> bool:
        """Return true once roll, recovery, and retry motion are all dormant."""
        if cycle_count > self.last_cycle_count:
            self.last_cycle_count = cycle_count
            self.best_frontier_rad = frontier_rad
            self.last_activity_step = step
        elif frontier_rad >= self.best_frontier_rad + self.frontier_epsilon_rad:
            self.best_frontier_rad = frontier_rad
            self.last_activity_step = step

        # A recovery can move opposite the rewarded backroll direction, so
        # physical body rotation or upward motion also keeps the diagnostic
        # alive. Joint jitter alone cannot pad a failed video indefinitely.
        if (
            angular_speed >= self.active_angular_rate
            or abs(vertical_speed) >= self.active_vertical_rate
        ):
            self.last_activity_step = step

        return (
            step >= self.min_steps
            and step - self.last_activity_step >= self.patience_steps
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--task-id",
        choices=(TASK_ID, REPEATED_TASK_ID),
        default=TASK_ID,
    )
    parser.add_argument(
        "--required-cycles",
        type=int,
        default=1,
        help="Valid cycle count required before publishing a proof.",
    )
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
    parser.add_argument(
        "--full-duration-diagnostic",
        action="store_true",
        help=(
            "Keep an incomplete diagnostic running for --duration instead of "
            "cutting it once the robot is dormant. Requires "
            "--allow-incomplete-diagnostic."
        ),
    )
    return parser.parse_args()


def _as_rgb8(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 4:
        frame = frame[0]
    if frame.dtype != np.uint8:
        frame = (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)
    return np.ascontiguousarray(frame[:, :, :3])


def _validate_args(args: argparse.Namespace) -> None:
    if args.duration <= 0.0 or args.width < 2 or args.height < 2:
        raise SystemExit("--duration, --width, and --height must be positive")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")
    if args.env_index < 0 or args.env_index >= args.batch_size:
        raise SystemExit("--env-index must be within [0, --batch-size)")
    if args.required_cycles < 1:
        raise SystemExit("--required-cycles must be at least 1")
    if args.full_duration_diagnostic and not args.allow_incomplete_diagnostic:
        raise SystemExit(
            "--full-duration-diagnostic requires --allow-incomplete-diagnostic"
        )


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


def _load_policy(
    base_env: ManagerBasedRlEnv,
    checkpoint: Path,
    *,
    task_id: str,
    device: str,
):
    agent_cfg = load_rl_cfg(task_id)
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
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
    _validate_args(args)

    temporary_dir = REPO_ROOT / ".tmp" / "codex"
    temporary_dir.mkdir(parents=True, exist_ok=True)
    temporary_output = temporary_dir / f"grounded-backroll-{os.getpid()}.mp4"

    configure_torch_backends()
    env_cfg = load_env_cfg(args.task_id, play=True)
    env_cfg.scene.num_envs = args.batch_size
    env_cfg.seed = args.seed
    env_cfg.auto_reset = False
    env_cfg.episode_length_s = args.duration
    # The recorder owns the selected variant's fixed horizon and physical
    # latch checks. Other variants in an exact replay batch must not terminate
    # the shared vector environment before the followed variant finishes.
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
    # Manager construction leaves the compiled default robot state in place;
    # apply the configured standing/repeated reset before the first rollout.
    # Without this, repeated recordings silently run with repeat_mode=False.
    base_env.reset()
    env, policy = _load_policy(
        base_env,
        checkpoint,
        task_id=args.task_id,
        device=args.device,
    )
    policy_fps = 1.0 / base_env.step_dt
    steps = round(args.duration / base_env.step_dt)
    hold_frames = round(POST_SUCCESS_HOLD_S * policy_fps)
    stuck_detector = DiagnosticStuckDetector(
        min_steps=round(DIAGNOSTIC_MIN_SECONDS / base_env.step_dt),
        patience_steps=round(DIAGNOSTIC_STUCK_SECONDS / base_env.step_dt),
    )
    writer: subprocess.Popen[bytes] | None = None
    last_frame: np.ndarray | None = None
    success = False
    stuck_cut = False

    try:
        for step in range(steps):
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
            if args.task_id == REPEATED_TASK_ID:
                success = bool(
                    base_env._backroll_cycle_count[args.env_index].item()
                    >= args.required_cycles
                )
            else:
                success = bool(base_env._backroll_success[args.env_index].item())
            if success:
                success = True
                break
            if args.allow_incomplete_diagnostic and not args.full_duration_diagnostic:
                robot = base_env.scene["robot"]
                env_index = args.env_index
                frontier_rad = float(base_env._roulade_max[env_index].item())
                cycle_count = int(base_env._backroll_cycle_count[env_index].item())
                angular_speed = float(
                    robot.data.root_link_ang_vel_b[env_index].norm().item()
                )
                vertical_speed = float(
                    robot.data.root_link_lin_vel_w[env_index, 2].item()
                )
                stuck_cut = stuck_detector.update(
                    step=step,
                    frontier_rad=frontier_rad,
                    cycle_count=cycle_count,
                    angular_speed=angular_speed,
                    vertical_speed=vertical_speed,
                )
                if stuck_cut:
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
    if success:
        kind = "proof"
    elif stuck_cut:
        kind = "stuck diagnostic"
    else:
        kind = "incomplete diagnostic"
    print(f"[grounded-backroll-video] wrote {kind}: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

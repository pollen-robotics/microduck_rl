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
import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from PIL import Image, ImageDraw, ImageFont

from mjlab_microduck.tasks import mdp as microduck_mdp

TASK_ID = "Mjlab-Roll-Sprint-Flat-MicroDuck"
REPO_ROOT = Path(__file__).resolve().parents[1]
RACE_LANE_SPACING = 0.28
RACE_CAMERA_LOOKAT = (0.60, 0.0, 0.08)
RACE_CAMERA_DISTANCE = 2.8
RACE_CAMERA_AZIMUTH = 135.0
RACE_CAMERA_ELEVATION = -55.0
TARGET_DISTANCE_M = 20.0
LABEL_COLORS = (
    (64, 180, 255),
    (255, 91, 91),
    (119, 221, 119),
    (255, 200, 74),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--task-id", default=TASK_ID)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Render every Nth simulation step while still evaluating every step.",
    )
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
    microduck_mdp.arrange_roll_sprint_race_start(base_env, lane_spacing)


def _race_label_text(
    robot_index: int, max_speed_mps: float, valid_distance_m: float
) -> str:
    return (
        f"R{robot_index + 1}  MAX {max_speed_mps:.2f} m/s"
        f"  |  {valid_distance_m:.1f} m valid"
    )


def _project_world_points(
    points_w: np.ndarray,
    camera: mujoco.MjvGLCamera,
    *,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Project world points through MuJoCo's rendered OpenGL camera."""
    position = np.asarray(camera.pos, dtype=np.float64)
    forward = np.asarray(camera.forward, dtype=np.float64)
    up = np.asarray(camera.up, dtype=np.float64)
    right = np.cross(forward, up)
    right /= max(float(np.linalg.norm(right)), 1.0e-12)
    up /= max(float(np.linalg.norm(up)), 1.0e-12)

    relative = np.asarray(points_w, dtype=np.float64) - position
    depth = relative @ forward
    horizontal = relative @ right
    vertical = relative @ up
    near = max(float(camera.frustum_near), 1.0e-6)
    if not bool(camera.orthographic):
        scale = near / np.maximum(depth, near)
        horizontal *= scale
        vertical *= scale

    vertical_center = 0.5 * (float(camera.frustum_top) + float(camera.frustum_bottom))
    vertical_half = max(
        0.5 * (float(camera.frustum_top) - float(camera.frustum_bottom)),
        1.0e-6,
    )
    horizontal_center = float(camera.frustum_center)
    horizontal_half = 0.5 * float(camera.frustum_width)
    if horizontal_half <= 1.0e-6:
        horizontal_half = vertical_half * width / height

    x_ndc = (horizontal - horizontal_center) / horizontal_half
    y_ndc = (vertical - vertical_center) / vertical_half
    pixels = np.column_stack(
        (
            (x_ndc + 1.0) * 0.5 * width,
            (1.0 - y_ndc) * 0.5 * height,
        )
    )
    visible = (
        (depth > near)
        & (x_ndc >= -1.1)
        & (x_ndc <= 1.1)
        & (y_ndc >= -1.1)
        & (y_ndc <= 1.1)
    )
    return pixels, visible


def _follow_first_robot(base_env: ManagerBasedRlEnv) -> None:
    """Keep robot 1 centered as the straight-lane race diverges."""
    renderer = base_env._offline_renderer
    if renderer is None:
        raise RuntimeError("Offline renderer is not initialized")
    first_position = base_env.scene["robot"].data.root_link_pos_w[0]
    renderer._cam.lookat[:] = (
        float(first_position[0].item()),
        float(first_position[1].item()),
        RACE_CAMERA_LOOKAT[2],
    )


def _overlay_race_labels(
    frame: np.ndarray,
    base_env: ManagerBasedRlEnv,
    *,
    max_speeds_mps: torch.Tensor,
    valid_distances_m: torch.Tensor,
    elapsed_s: float,
) -> np.ndarray:
    """Draw per-robot metrics at the rendered screen position of each robot."""
    renderer = base_env._offline_renderer
    if renderer is None:
        raise RuntimeError("Offline renderer is not initialized")
    scene = renderer.renderer.scene
    camera = mujoco.mjv_averageCamera(scene.camera[0], scene.camera[1])
    robot_positions = (
        base_env.scene["robot"].data.root_link_pos_w.detach().cpu().numpy().copy()
    )
    robot_positions[:, 2] += 0.24
    height, width = frame.shape[:2]
    pixels, visible = _project_world_points(
        robot_positions,
        camera,
        width=width,
        height=height,
    )

    image = Image.fromarray(frame).convert("RGBA")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
        header_font = ImageFont.truetype("arialbd.ttf", 20)
    except OSError:
        font = ImageFont.load_default(size=18)
        header_font = ImageFont.load_default(size=20)

    header = f"20 m ROLL RACE  |  t {elapsed_s:05.1f} s / 40.0 s  |  camera follows R1"
    header_box = draw.textbbox((0, 0), header, font=header_font)
    header_width = header_box[2] - header_box[0]
    draw.rounded_rectangle(
        (12, 10, 32 + header_width, 42),
        radius=7,
        fill=(10, 14, 22, 215),
        outline=(235, 239, 245),
        width=1,
    )
    draw.text((22, 15), header, font=header_font, fill=(255, 255, 255))

    max_speeds = max_speeds_mps.detach().cpu().tolist()
    valid_distances = valid_distances_m.detach().cpu().tolist()
    draw_order = (*range(1, len(pixels)), 0)
    for index in draw_order:
        pixel = pixels[index]
        is_visible = visible[index]
        if not is_visible:
            continue
        label = _race_label_text(index, max_speeds[index], valid_distances[index])
        text_box = draw.textbbox((0, 0), label, font=font)
        label_width = text_box[2] - text_box[0]
        label_height = text_box[3] - text_box[1]
        x = min(
            max(float(pixel[0]) - label_width / 2.0, 8.0), width - label_width - 20.0
        )
        y = min(max(float(pixel[1]) - label_height - 20.0, 50.0), height - 38.0)
        color = LABEL_COLORS[index % len(LABEL_COLORS)]
        draw.rounded_rectangle(
            (x - 6, y - 4, x + label_width + 6, y + label_height + 7),
            radius=6,
            fill=(8, 12, 18, 220),
            outline=color,
            width=2,
        )
        draw.text((x, y), label, font=font, fill=color)
        draw.line(
            (float(pixel[0]), y + label_height + 7, float(pixel[0]), float(pixel[1])),
            fill=color,
            width=2,
        )
    return np.ascontiguousarray(np.asarray(image.convert("RGB")))


def main() -> int:
    args = _parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    temporary_dir = REPO_ROOT / ".tmp" / "codex"
    temporary_dir.mkdir(parents=True, exist_ok=True)
    temporary_output = temporary_dir / f"roll-sprint-recording-{os.getpid()}.mp4"
    if not checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")
    if args.steps < 1 or args.frame_stride < 1 or args.width < 2 or args.height < 2:
        raise SystemExit(
            "--steps, --frame-stride, --width, and --height must be positive"
        )

    configure_torch_backends()
    env_cfg = load_env_cfg(args.task_id, play=True)
    agent_cfg = load_rl_cfg(args.task_id)
    env_cfg.scene.num_envs = 4
    env_cfg.scene.env_spacing = RACE_LANE_SPACING
    env_cfg.scene.terrain.env_spacing = RACE_LANE_SPACING
    env_cfg.seed = args.seed
    env_cfg.auto_reset = False
    policy_dt = env_cfg.sim.mujoco.timestep * env_cfg.decimation
    env_cfg.episode_length_s = max(40.0, args.steps * policy_dt)
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
    reset_cfg.params["postroll_prob"] = 0.0
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
    robot = base_env.scene["robot"]
    race_heading = base_env._roll_sprint_heading_w.clone()
    max_speeds_mps = torch.zeros(4, device=base_env.device)

    writer: subprocess.Popen[bytes] | None = None
    try:
        for step in range(args.steps):
            with torch.inference_mode():
                observations = env.get_observations()
                actions = policy(observations)
                env.step(actions)
                forward_speed = (
                    robot.data.root_link_lin_vel_w[:, :2] * race_heading
                ).sum(dim=-1)
                max_speeds_mps = torch.maximum(
                    max_speeds_mps, forward_speed.clamp_min(0.0)
                )
                valid_distances_m = (
                    base_env._roll_sprint_forward_frontier
                    - base_env._roll_sprint_forward_origin
                ).clamp_min(0.0)
            _follow_first_robot(base_env)
            if step % args.frame_stride != 0 and step != args.steps - 1:
                continue
            rendered = base_env.render()
            if rendered is None:
                raise RuntimeError("MuJoCo returned no RGB frame")
            frame = _as_rgb8(rendered)
            frame = _overlay_race_labels(
                frame,
                base_env,
                max_speeds_mps=max_speeds_mps,
                valid_distances_m=valid_distances_m,
                elapsed_s=(step + 1) * policy_dt,
            )
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

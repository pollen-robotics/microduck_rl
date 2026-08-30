#!/usr/bin/env python3
"""Record four standing-start roll-sprint rollouts in one tiled video."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from collections import deque
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
TARGET_DISTANCE_M = 20.0
RACE_CAMERA_LOOKAT = (0.60, 0.0, 0.08)
RACE_CAMERA_DISTANCE = 3.2
RACE_CAMERA_FOVY = 45.0
RACE_CAMERA_AZIMUTH = 90.0
RACE_CAMERA_ELEVATION = -45.0
RACE_CAMERA_LEAD_M = 0.60
RACE_CAMERA_FOLLOW_ALPHA = 0.25
RACE_CAMERA_MAX_STEP_M = 0.08
RACE_LINE_HEIGHT = 0.008
RACE_LINE_RADIUS = 0.018
SPEED_WINDOW_S = 1.0
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


def _recording_fps(policy_dt: float, frame_stride: int) -> float:
    """Encode sampled frames at their real simulation-time cadence."""
    return 1.0 / (policy_dt * frame_stride)


def _race_header_text(elapsed_s: float, total_s: float, leader_index: int) -> str:
    return (
        f"20 m ROLL RACE  |  t {elapsed_s:05.1f} s / {total_s:.1f} s"
        f"  |  camera follows in-lane leader R{leader_index + 1}"
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
        f"R{robot_index + 1}  MAX 1s {max_speed_mps:.2f} m/s"
        f"  |  {valid_distance_m:.1f} m valid"
    )


def _accumulate_max_forward_speed(
    max_speeds_mps: torch.Tensor,
    previous_forward_position_m: torch.Tensor,
    current_forward_position_m: torch.Tensor,
    linear_velocity_w: torch.Tensor,
    heading_xy: torch.Tensor,
    sample_dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update peak speed from visible travel and the simulator velocity signal.

    The position delta is authoritative for the video because it measures the
    same projected displacement viewers see.  Taking the larger finite signal
    also handles offscreen backends whose cached root velocity can lag a step.
    """
    current_forward_position_m = torch.nan_to_num(
        current_forward_position_m,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    position_speed_mps = (
        current_forward_position_m - previous_forward_position_m
    ) / sample_dt
    velocity_speed_mps = (linear_velocity_w[:, :2] * heading_xy).sum(dim=-1)
    observed_speed_mps = torch.maximum(position_speed_mps, velocity_speed_mps)
    observed_speed_mps = torch.nan_to_num(
        observed_speed_mps,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).clamp_min(0.0)
    return (
        torch.maximum(max_speeds_mps, observed_speed_mps),
        current_forward_position_m,
    )


def _race_corridor_segments(
    *,
    lane_spacing: float = RACE_LANE_SPACING,
    target_distance_m: float = TARGET_DISTANCE_M,
) -> list[tuple[np.ndarray, np.ndarray, tuple[float, float, float, float], float]]:
    """Return visible lane, start, finish, and distance-marker line segments."""
    edge_y = 2.0 * lane_spacing
    lane_boundaries = np.linspace(-edge_y, edge_y, 5)
    segments = [
        (
            np.array([0.0, lane_y, RACE_LINE_HEIGHT]),
            np.array([target_distance_m, lane_y, RACE_LINE_HEIGHT]),
            (0.35, 0.74, 1.0, 0.9),
            RACE_LINE_RADIUS,
        )
        for lane_y in lane_boundaries
    ]
    segments.extend(
        (
            np.array([distance_m, -edge_y, RACE_LINE_HEIGHT]),
            np.array([distance_m, edge_y, RACE_LINE_HEIGHT]),
            (1.0, 1.0, 1.0, 1.0) if distance_m == 0.0 else (1.0, 0.76, 0.15, 1.0),
            RACE_LINE_RADIUS * 1.6,
        )
        for distance_m in (0.0, target_distance_m)
    )
    segments.extend(
        (
            np.array([distance_m, -edge_y, RACE_LINE_HEIGHT]),
            np.array([distance_m, edge_y, RACE_LINE_HEIGHT]),
            (0.72, 0.78, 0.86, 0.5),
            RACE_LINE_RADIUS * 0.55,
        )
        for distance_m in np.arange(1.0, target_distance_m, 1.0)
    )
    return segments


def _draw_race_corridor(visualizer) -> None:
    """Draw a non-colliding four-lane 20 m corridor into the render scene."""
    for start, end, color, radius in _race_corridor_segments():
        visualizer.add_cylinder(start, end, radius=radius, color=color)


def _install_race_corridor_visualizer(base_env: ManagerBasedRlEnv) -> None:
    """Add the corridor after normal debug visuals on every rendered frame."""
    original_update_visualizers = getattr(base_env, "update_visualizers", None)

    def update_visualizers(visualizer) -> None:
        if original_update_visualizers is not None:
            original_update_visualizers(visualizer)
        _draw_race_corridor(visualizer)

    base_env.update_visualizers = update_visualizers


def _label_positions(
    pixels: np.ndarray,
    visible: np.ndarray,
    label_sizes: list[tuple[int, int]],
    *,
    width: int,
    height: int,
) -> dict[int, tuple[float, float]]:
    """Place robot-anchored labels without overlapping one another."""
    margin = 8.0
    gap = 7.0
    top = 54.0
    order = sorted(
        (index for index, is_visible in enumerate(visible) if is_visible),
        key=lambda index: (float(pixels[index, 1]), float(pixels[index, 0])),
    )
    positions: dict[int, tuple[float, float]] = {}
    previous_bottom = top
    for index in order:
        label_width, label_height = label_sizes[index]
        anchor_x = float(pixels[index, 0])
        desired_x = (
            anchor_x + 14.0
            if anchor_x <= width / 2.0
            else anchor_x - label_width - 14.0
        )
        x = min(max(desired_x, margin), width - label_width - margin)
        desired_y = float(pixels[index, 1]) - label_height / 2.0
        y = max(desired_y, previous_bottom + gap)
        positions[index] = (x, y)
        previous_bottom = y + label_height

    if positions and previous_bottom > height - margin:
        shift = previous_bottom - (height - margin)
        for index, (x, y) in positions.items():
            positions[index] = (x, max(top, y - shift))
    return positions


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


def _camera_follow_x(previous_x_m: float, robot_x_m: float) -> float:
    """Smoothly follow the lane leader longitudinally in either direction."""
    if not np.isfinite(robot_x_m):
        return previous_x_m
    target_x_m = float(
        np.clip(
            robot_x_m + RACE_CAMERA_LEAD_M,
            -RACE_CAMERA_LEAD_M,
            TARGET_DISTANCE_M,
        )
    )
    step_m = float(
        np.clip(
            (target_x_m - previous_x_m) * RACE_CAMERA_FOLLOW_ALPHA,
            -RACE_CAMERA_MAX_STEP_M,
            RACE_CAMERA_MAX_STEP_M,
        )
    )
    return previous_x_m + step_m


def _camera_follow_y(previous_y_m: float, robot_y_m: float) -> float:
    """Smoothly follow robot 1 laterally in either direction."""
    if not np.isfinite(robot_y_m):
        return previous_y_m
    step_m = float(
        np.clip(
            (robot_y_m - previous_y_m) * RACE_CAMERA_FOLLOW_ALPHA,
            -RACE_CAMERA_MAX_STEP_M,
            RACE_CAMERA_MAX_STEP_M,
        )
    )
    return previous_y_m + step_m


def _select_in_lane_leader(base_env: ManagerBasedRlEnv) -> int:
    """Return the furthest-forward robot that remains inside its race lane."""
    forward_position = base_env._roll_sprint_forward_position
    lateral_displacement = base_env._roll_sprint_lateral_displacement
    in_lane = (
        torch.isfinite(forward_position)
        & torch.isfinite(lateral_displacement)
        & (lateral_displacement.abs() <= 0.5 * RACE_LANE_SPACING)
    )
    if torch.any(in_lane):
        scores = torch.where(
            in_lane,
            forward_position,
            torch.full_like(forward_position, -torch.inf),
        )
        return int(torch.argmax(scores).item())
    return int(torch.argmin(lateral_displacement.abs()).item())


def _follow_lane_leader(
    base_env: ManagerBasedRlEnv,
    previous_x_m: float,
    previous_y_m: float,
) -> tuple[float, float, int]:
    """Travel with the current furthest-forward in-lane robot."""
    renderer = base_env._offline_renderer
    if renderer is None:
        raise RuntimeError("Offline renderer is not initialized")
    # Use the reward-side position cache. The asset root position exposed by
    # the offscreen backend can lag the physics state and let the leader leave.
    leader_index = _select_in_lane_leader(base_env)
    leader_x_m = float(base_env._roll_sprint_forward_position[leader_index].item())
    leader_y_m = float(
        (
            base_env.scene.terrain.env_origins[leader_index, 1]
            + base_env._roll_sprint_lateral_displacement[leader_index]
        ).item()
    )
    camera_x_m = _camera_follow_x(previous_x_m, leader_x_m)
    camera_y_m = _camera_follow_y(previous_y_m, leader_y_m)
    renderer._cam.lookat[:] = (
        camera_x_m,
        camera_y_m,
        RACE_CAMERA_LOOKAT[2],
    )
    return camera_x_m, camera_y_m, leader_index


def _overlay_race_labels(
    frame: np.ndarray,
    base_env: ManagerBasedRlEnv,
    *,
    max_speeds_mps: torch.Tensor,
    valid_distances_m: torch.Tensor,
    elapsed_s: float,
    total_s: float,
    leader_index: int,
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

    header = _race_header_text(elapsed_s, total_s, leader_index)
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
    labels = [
        _race_label_text(index, max_speeds[index], valid_distances[index])
        for index in range(len(pixels))
    ]
    label_sizes = []
    for label in labels:
        text_box = draw.textbbox((0, 0), label, font=font)
        label_sizes.append((text_box[2] - text_box[0], text_box[3] - text_box[1]))
    positions = _label_positions(
        pixels,
        visible,
        label_sizes,
        width=width,
        height=height,
    )
    for index, (x, y) in positions.items():
        pixel = pixels[index]
        label = labels[index]
        label_width, label_height = label_sizes[index]
        color = LABEL_COLORS[index % len(LABEL_COLORS)]
        draw.rounded_rectangle(
            (x - 6, y - 4, x + label_width + 6, y + label_height + 7),
            radius=6,
            fill=(8, 12, 18, 220),
            outline=color,
            width=2,
        )
        draw.text((x, y), label, font=font, fill=color)
        label_edge_x = x if float(pixel[0]) < x else x + label_width
        draw.line(
            (
                float(pixel[0]),
                float(pixel[1]),
                label_edge_x,
                y + label_height / 2.0,
            ),
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
    env_cfg.viewer.fovy = RACE_CAMERA_FOVY
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
    _install_race_corridor_visualizer(base_env)
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
    previous_forward_position_m = base_env._roll_sprint_forward_position.clone()
    speed_window_steps = max(1, round(SPEED_WINDOW_S / policy_dt))
    forward_position_history: deque[torch.Tensor] = deque(
        [previous_forward_position_m], maxlen=speed_window_steps + 1
    )
    camera_x_m = RACE_CAMERA_LOOKAT[0]
    camera_y_m = RACE_CAMERA_LOOKAT[1]
    leader_index = 0

    writer: subprocess.Popen[bytes] | None = None
    try:
        for step in range(args.steps):
            with torch.inference_mode():
                observations = env.get_observations()
                actions = policy(observations)
                env.step(actions)
                current_forward_position_m = (
                    base_env._roll_sprint_forward_position.clone()
                )
                forward_position_history.append(current_forward_position_m)
                if len(forward_position_history) == speed_window_steps + 1:
                    max_speeds_mps, previous_forward_position_m = (
                        _accumulate_max_forward_speed(
                            max_speeds_mps,
                            forward_position_history[0],
                            current_forward_position_m,
                            robot.data.root_link_lin_vel_w,
                            race_heading,
                            speed_window_steps * policy_dt,
                        )
                    )
                else:
                    previous_forward_position_m = current_forward_position_m
                valid_distances_m = (
                    base_env._roll_sprint_forward_frontier
                    - base_env._roll_sprint_forward_origin
                ).clamp_min(0.0)
            camera_x_m, camera_y_m, leader_index = _follow_lane_leader(
                base_env,
                camera_x_m,
                camera_y_m,
            )
            if (step + 1) % args.frame_stride != 0:
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
                total_s=args.steps * policy_dt,
                leader_index=leader_index,
            )
            if writer is None:
                frame_height, frame_width = frame.shape[:2]
                writer = _ffmpeg_writer(
                    temporary_output,
                    width=frame_width,
                    height=frame_height,
                    fps=_recording_fps(policy_dt, args.frame_stride),
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

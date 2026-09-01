#!/usr/bin/env python3
"""Record deterministic roll-sprint races and recovery rollouts."""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import NamedTuple

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
FIVE_RACER_LANE_SPACING = 0.21
ROAD_HALF_WIDTH_M = microduck_mdp._ROLL_SPRINT_ROAD_HALF_WIDTH
ROAD_SAFE_FULL_REWARD_HALF_WIDTH_M = microduck_mdp._ROLL_SPRINT_ROAD_SAFE_HALF_WIDTH
ROAD_REPOSITION_TRIGGER_M = microduck_mdp._ROLL_SPRINT_REPOSITION_TRIGGER_M
ROAD_REPOSITION_REARM_M = microduck_mdp._ROLL_SPRINT_REPOSITION_REARM_M
TARGET_DISTANCE_M = 10.0
DEFAULT_OUTPUT_WIDTH = 1920
DEFAULT_OUTPUT_HEIGHT = 1080
RACE_CAMERA_LOOKAT = (0.60, 0.0, 0.08)
RACE_CAMERA_DISTANCE = 3.2
RACE_CAMERA_FOVY = 45.0
RACE_CAMERA_AZIMUTH = 90.0
RACE_CAMERA_ELEVATION = -45.0
RACE_CAMERA_LEAD_M = 0.60
RACE_CAMERA_SPRING_GAIN_S2 = 9.0
RACE_CAMERA_DAMPING_S_INV = 6.0
RACE_CAMERA_MAX_SPEED_MPS = 3.0
RACE_CAMERA_MAX_ACCEL_MPS2 = 4.0
BACKROLL_CAMERA_LEAD_M = 0.0
BACKROLL_CAMERA_MAX_SPEED_MPS = 8.0
BACKROLL_CAMERA_MAX_ACCEL_MPS2 = 12.0
RACE_LINE_HEIGHT = 0.008
RACE_LINE_RADIUS = 0.018
FINISH_ARCH_HEIGHT_M = 0.72
FINISH_CELEBRATION_SECONDS = 4.0
SHOWCASE_FINISHER_TARGET = 3
SHOWCASE_POST_FINISH_HOLD_SECONDS = 1.5
FINISH_EFFECT_COLORS = (
    (0.20, 0.72, 1.00, 1.0),
    (1.00, 0.32, 0.42, 1.0),
    (1.00, 0.78, 0.18, 1.0),
    (0.42, 0.95, 0.54, 1.0),
    (0.78, 0.42, 1.00, 1.0),
)
RECOVERY_ORIENTATIONS = ("face_down", "face_up", "left", "right")


class CameraFollowState(NamedTuple):
    """Longitudinal camera state for a damped, frame-rate-independent follow."""

    x_m: float
    velocity_mps: float = 0.0


class FinishCelebrationState:
    """Latch valid finish times and expose deterministic animation time."""

    def __init__(self, num_robots: int) -> None:
        self.current_time_s = 0.0
        self.finish_times_s: list[float | None] = [None] * num_robots
        self.stop_time_s: float | None = None

    def update(self, credited_frontier_m: torch.Tensor, elapsed_s: float) -> None:
        """Latch each robot once, using only valid roll-linked frontier."""
        if credited_frontier_m.numel() != len(self.finish_times_s):
            raise ValueError("one credited frontier is required for every racer")
        self.current_time_s = float(elapsed_s)
        for index, frontier_m in enumerate(credited_frontier_m.detach().cpu().tolist()):
            if self.finish_times_s[index] is None and frontier_m >= TARGET_DISTANCE_M:
                self.finish_times_s[index] = self.current_time_s

    @property
    def finished_count(self) -> int:
        """Return the number of racers with a latched valid finish."""
        return sum(finish_time_s is not None for finish_time_s in self.finish_times_s)

    def arm_stop_after_finishers(
        self,
        required_finishers: int,
        post_finish_hold_s: float,
    ) -> None:
        """Latch one cutoff once enough racers finish, leaving time to celebrate."""
        if required_finishers < 1 or post_finish_hold_s < 0.0:
            raise ValueError("finish target and celebration hold must be non-negative")
        if self.stop_time_s is None and self.finished_count >= required_finishers:
            self.stop_time_s = self.current_time_s + post_finish_hold_s

    @property
    def stop_due(self) -> bool:
        """Return whether the latched celebration cutoff has elapsed."""
        return self.stop_time_s is not None and self.current_time_s >= self.stop_time_s


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
    parser.add_argument(
        "--output-fps",
        type=float,
        help="Write a constant-frame-rate video at this cadence.",
    )
    parser.add_argument(
        "--playback-speed",
        type=float,
        default=1.0,
        help="Play simulation time this many times faster than real time.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--width", type=int, default=DEFAULT_OUTPUT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_OUTPUT_HEIGHT)
    parser.add_argument(
        "--recovery-montage",
        action="store_true",
        help="Record deterministic face-down, face-up, left, and right recovery starts.",
    )
    parser.add_argument(
        "--five-robot-showcase",
        action="store_true",
        help=(
            "Record five aligned racers with a finish arch and valid-frontier "
            "firework celebration."
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
    output_fps: float | None,
) -> subprocess.Popen[bytes]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to record roll-sprint videos")
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
    ]
    if output_fps is not None:
        command.extend(("-r", f"{output_fps:g}"))
    command.extend(
        (
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        )
    )
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def _recording_fps(policy_dt: float, frame_stride: int) -> float:
    """Encode sampled frames at their real simulation-time cadence."""
    return 1.0 / (policy_dt * frame_stride)


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


def _roll_direction_for_task(task_id: str) -> float:
    """Return the signed roll/travel direction encoded by the task id."""
    return -1.0 if "Backroll-Sprint" in task_id else 1.0


def _arrange_race_start(
    base_env: ManagerBasedRlEnv,
    task_id: str = TASK_ID,
    lane_spacing: float = RACE_LANE_SPACING,
) -> None:
    """Place all robots on one deterministic world +x race course."""
    roll_direction = _roll_direction_for_task(task_id)
    if roll_direction > 0.0:
        microduck_mdp.arrange_roll_sprint_race_start(base_env, lane_spacing)
    else:
        microduck_mdp.arrange_roll_sprint_race_start(
            base_env,
            lane_spacing,
            roll_direction=roll_direction,
        )


def _arrange_recovery_montage(
    base_env: ManagerBasedRlEnv,
    *,
    seed: int,
    task_id: str = TASK_ID,
    lane_spacing: float = RACE_LANE_SPACING,
) -> None:
    """Place one deterministic recovery orientation in each visible lane."""
    kwargs = {
        "seed": seed,
        "orientations": RECOVERY_ORIENTATIONS,
    }
    roll_direction = _roll_direction_for_task(task_id)
    if roll_direction < 0.0:
        kwargs["roll_direction"] = roll_direction
    microduck_mdp.arrange_roll_sprint_recovery_start(
        base_env,
        lane_spacing,
        **kwargs,
    )


def _refresh_manual_start_state(base_env: ManagerBasedRlEnv) -> None:
    """Refresh sensors and delayed observations after a manual pose."""
    env_ids = torch.arange(
        base_env.num_envs, device=base_env.device, dtype=torch.long
    )
    base_env.sim.sense()
    base_env.observation_manager.reset(env_ids)
    base_env.obs_buf = base_env.observation_manager.compute(update_history=True)


def _load_policy_then_arrange_start(
    *,
    base_env: ManagerBasedRlEnv,
    agent_cfg,
    task_id: str,
    checkpoint: Path,
    device: str,
    recovery_montage: bool,
    seed: int,
    race_lane_spacing: float = RACE_LANE_SPACING,
    celebration_state: FinishCelebrationState | None = None,
):
    """Load inference first, then apply the final deterministic rollout state."""
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
        str(checkpoint),
        load_cfg={"actor": True},
        strict=True,
        map_location=device,
    )
    policy = runner.get_inference_policy(device=device)
    if recovery_montage:
        if _roll_direction_for_task(task_id) < 0.0:
            _arrange_recovery_montage(base_env, seed=seed, task_id=task_id)
        else:
            _arrange_recovery_montage(base_env, seed=seed)
    else:
        if _roll_direction_for_task(task_id) < 0.0:
            _arrange_race_start(
                base_env,
                task_id=task_id,
                lane_spacing=race_lane_spacing,
            )
        else:
            _arrange_race_start(base_env, lane_spacing=race_lane_spacing)
        _install_race_corridor_visualizer(
            base_env,
            num_lanes=base_env.num_envs,
            lane_spacing=race_lane_spacing,
            celebration_state=celebration_state,
        )
    _refresh_manual_start_state(base_env)
    return env, policy


def _race_corridor_segments(
    *,
    num_lanes: int = 4,
    lane_spacing: float = RACE_LANE_SPACING,
    target_distance_m: float = TARGET_DISTANCE_M,
) -> list[tuple[np.ndarray, np.ndarray, tuple[float, float, float, float], float]]:
    """Return visible lane, start, finish, and distance-marker line segments."""
    if num_lanes < 1 or lane_spacing <= 0.0:
        raise ValueError("the race needs at least one positive-width lane")
    lane_centers = (
        np.arange(num_lanes, dtype=np.float64) - (num_lanes - 1) / 2.0
    ) * lane_spacing
    if np.max(np.abs(lane_centers)) > ROAD_SAFE_FULL_REWARD_HALF_WIDTH_M + 1.0e-9:
        raise ValueError("race starts must remain inside the full-reward road band")
    edge_y = ROAD_HALF_WIDTH_M
    internal_boundaries = 0.5 * (lane_centers[:-1] + lane_centers[1:])
    lane_boundaries = np.concatenate(([-edge_y], internal_boundaries, [edge_y]))
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


def _finish_arch_segments() -> list[
    tuple[np.ndarray, np.ndarray, tuple[float, float, float, float], float]
]:
    """Return a clean gold finish arch and a checkered ground strip."""
    gold = (1.0, 0.72, 0.10, 1.0)
    segments = [
        (
            np.array([TARGET_DISTANCE_M, side * ROAD_HALF_WIDTH_M, 0.02]),
            np.array([TARGET_DISTANCE_M, side * ROAD_HALF_WIDTH_M, FINISH_ARCH_HEIGHT_M]),
            gold,
            0.032,
        )
        for side in (-1.0, 1.0)
    ]
    segments.append(
        (
            np.array([TARGET_DISTANCE_M, -ROAD_HALF_WIDTH_M, FINISH_ARCH_HEIGHT_M]),
            np.array([TARGET_DISTANCE_M, ROAD_HALF_WIDTH_M, FINISH_ARCH_HEIGHT_M]),
            gold,
            0.032,
        )
    )
    checker_width = (2.0 * ROAD_HALF_WIDTH_M) / 10.0
    for index in range(10):
        y0 = -ROAD_HALF_WIDTH_M + index * checker_width
        segments.append(
            (
                np.array([TARGET_DISTANCE_M, y0, RACE_LINE_HEIGHT * 1.4]),
                np.array(
                    [TARGET_DISTANCE_M, y0 + checker_width, RACE_LINE_HEIGHT * 1.4]
                ),
                (1.0, 1.0, 1.0, 1.0) if index % 2 == 0 else (0.08, 0.10, 0.14, 1.0),
                RACE_LINE_RADIUS * 1.45,
            )
        )
    return segments


def _finish_celebration_segments(
    state: FinishCelebrationState,
    lane_centers: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray, tuple[float, float, float, float], float]]:
    """Build deterministic firework rays and falling confetti for valid finishers."""
    segments = []
    for robot_index, finish_time_s in enumerate(state.finish_times_s):
        if finish_time_s is None:
            continue
        age_s = state.current_time_s - finish_time_s
        if age_s < 0.0 or age_s > FINISH_CELEBRATION_SECONDS:
            continue
        color = FINISH_EFFECT_COLORS[robot_index % len(FINISH_EFFECT_COLORS)]
        burst_cycle = int(age_s / 0.85)
        burst_phase = (age_s % 0.85) / 0.85
        burst_radius = 0.08 + 0.42 * math.sin(min(burst_phase, 1.0) * math.pi / 2.0)
        burst_center = np.array(
            [
                TARGET_DISTANCE_M + 0.10 + 0.12 * (burst_cycle % 2),
                lane_centers[robot_index] * 0.72 + 0.06 * ((burst_cycle % 3) - 1),
                0.46 + 0.11 * ((robot_index + burst_cycle) % 3),
            ]
        )
        alpha = max(0.16, 1.0 - burst_phase)
        ray_color = (color[0], color[1], color[2], alpha)
        for ray_index in range(14):
            angle = 2.0 * math.pi * ray_index / 14.0
            depth = 0.10 * math.sin(angle * 3.0 + robot_index)
            endpoint = burst_center + np.array(
                [depth, burst_radius * math.cos(angle), burst_radius * math.sin(angle)]
            )
            segments.append((burst_center, endpoint, ray_color, 0.010))

        for piece_index in range(18):
            fall_phase = (age_s * 0.62 + piece_index * 0.173 + robot_index * 0.071) % 1.0
            piece_color = FINISH_EFFECT_COLORS[
                (robot_index + piece_index) % len(FINISH_EFFECT_COLORS)
            ]
            start = np.array(
                [
                    TARGET_DISTANCE_M - 0.42 + 0.84 * ((piece_index * 0.37) % 1.0),
                    -ROAD_HALF_WIDTH_M + 2.0 * ROAD_HALF_WIDTH_M * ((piece_index * 0.61) % 1.0),
                    0.07 + 0.72 * (1.0 - fall_phase),
                ]
            )
            end = start + np.array([0.025, 0.016 * (-1.0 if piece_index % 2 else 1.0), -0.035])
            segments.append((start, end, piece_color, 0.008))
    return segments


def _draw_race_corridor(
    visualizer,
    *,
    num_lanes: int = 4,
    lane_spacing: float = RACE_LANE_SPACING,
    celebration_state: FinishCelebrationState | None = None,
) -> None:
    """Draw the non-colliding shared-road race corridor into the render scene."""
    segments = _race_corridor_segments(
        num_lanes=num_lanes,
        lane_spacing=lane_spacing,
    )
    if celebration_state is not None:
        lane_centers = (
            np.arange(num_lanes, dtype=np.float64) - (num_lanes - 1) / 2.0
        ) * lane_spacing
        segments.extend(_finish_arch_segments())
        segments.extend(_finish_celebration_segments(celebration_state, lane_centers))
    for start, end, color, radius in segments:
        visualizer.add_cylinder(start, end, radius=radius, color=color)


def _install_race_corridor_visualizer(
    base_env: ManagerBasedRlEnv,
    *,
    num_lanes: int = 4,
    lane_spacing: float = RACE_LANE_SPACING,
    celebration_state: FinishCelebrationState | None = None,
) -> None:
    """Add the corridor after normal debug visuals on every rendered frame."""
    original_update_visualizers = getattr(base_env, "update_visualizers", None)

    def update_visualizers(visualizer) -> None:
        if original_update_visualizers is not None:
            original_update_visualizers(visualizer)
        _draw_race_corridor(
            visualizer,
            num_lanes=num_lanes,
            lane_spacing=lane_spacing,
            celebration_state=celebration_state,
        )

    base_env.update_visualizers = update_visualizers


def _camera_follow_x(
    state: CameraFollowState,
    robot_x_m: float,
    dt_s: float,
    minimum_x_m: float | None = -RACE_CAMERA_LEAD_M,
    lead_m: float = RACE_CAMERA_LEAD_M,
    max_speed_mps: float = RACE_CAMERA_MAX_SPEED_MPS,
    max_accel_mps2: float = RACE_CAMERA_MAX_ACCEL_MPS2,
) -> CameraFollowState:
    """Advance a critically damped camera without position or velocity snaps."""
    if not np.isfinite(robot_x_m) or not np.isfinite(dt_s) or dt_s <= 0.0:
        return state
    # Do not stop the camera at the 10 m line. Valid frontier can lag physical
    # position while a roll is still being verified, so a hard 10 m camera cap
    # leaves every contender off-screen during the decisive final seconds.
    target_x_m = robot_x_m + lead_m
    if minimum_x_m is not None:
        target_x_m = max(target_x_m, minimum_x_m)
    acceleration_mps2 = float(
        np.clip(
            RACE_CAMERA_SPRING_GAIN_S2 * (target_x_m - state.x_m)
            - RACE_CAMERA_DAMPING_S_INV * state.velocity_mps,
            -max_accel_mps2,
            max_accel_mps2,
        )
    )
    velocity_mps = float(
        np.clip(
            state.velocity_mps + acceleration_mps2 * dt_s,
            -max_speed_mps,
            max_speed_mps,
        )
    )
    return CameraFollowState(
        x_m=state.x_m + velocity_mps * dt_s,
        velocity_mps=velocity_mps,
    )


def _sensor_contact(base_env: ManagerBasedRlEnv, name: str) -> torch.Tensor:
    sensor = base_env.scene.sensors.get(name)
    if sensor is None or sensor.data.found is None:
        return torch.zeros(base_env.num_envs, dtype=torch.bool, device=base_env.device)
    found = sensor.data.found
    return (found.view(found.shape[0], -1) > 0).any(dim=-1)


def _course_lateral_positions(base_env: ManagerBasedRlEnv) -> torch.Tensor:
    """Return the reward-side shared-course coordinate used by the policy."""
    return base_env._roll_sprint_course_lateral_position


def _launch_ready_mask(base_env: ManagerBasedRlEnv) -> torch.Tensor:
    """Return the existing feet-supported, upright, course-aligned launch gate."""
    robot = base_env.scene["robot"]
    quat = torch.nan_to_num(robot.data.root_link_quat_w, nan=0.0)
    upright_cos = 1.0 - 2.0 * (quat[:, 1].square() + quat[:, 2].square())
    lateral_axis_z = 2.0 * (quat[:, 2] * quat[:, 3] + quat[:, 0] * quat[:, 1]).abs()
    root_height = (
        robot.data.root_link_pos_w[:, 2] - base_env.scene.terrain.env_origins[:, 2]
    )
    forward_rate = robot.data.root_link_ang_vel_b[:, 1].abs()
    heading_error = base_env._roll_sprint_heading_error_rad.abs()
    return (
        _sensor_contact(base_env, "feet_ground_contact")
        & ~_sensor_contact(base_env, "head_ground_contact")
        & (upright_cos >= microduck_mdp._ROLL_SPRINT_RECOVERY_UPRIGHT_COS)
        & (lateral_axis_z <= microduck_mdp._ROLL_SPRINT_RECOVERY_LATERAL_Z)
        & (root_height >= microduck_mdp._ROLL_SPRINT_RECOVERY_MIN_HEIGHT_M)
        & (forward_rate <= microduck_mdp._ROLL_SPRINT_RECOVERY_MAX_FORWARD_RATE)
        & (heading_error <= math.radians(20.0))
    )


def _furthest_frontier_index(
    frontier: torch.Tensor,
    forward_position: torch.Tensor,
    eligible: torch.Tensor,
) -> int:
    candidates = torch.nonzero(eligible, as_tuple=False).flatten().tolist()
    return max(
        candidates,
        key=lambda index: (
            float(frontier[index].item()),
            float(forward_position[index].item()),
            -index,
        ),
    )


def _select_on_road_leader(
    base_env: ManagerBasedRlEnv,
    previous_leader_index: int | None = None,
) -> int:
    """Choose the furthest credited standing robot anywhere on the shared road."""
    forward_position = base_env._roll_sprint_forward_position
    frontier = (
        base_env._roll_sprint_forward_frontier - base_env._roll_sprint_forward_origin
    ).clamp_min(0.0)
    course_lateral = _course_lateral_positions(base_env)
    on_road = (
        torch.isfinite(forward_position)
        & torch.isfinite(frontier)
        & torch.isfinite(course_lateral)
        & (course_lateral.abs() <= ROAD_HALF_WIDTH_M)
    )
    launch_ready = _launch_ready_mask(base_env)
    eligible = on_road & launch_ready
    if bool(eligible.any()):
        return _furthest_frontier_index(frontier, forward_position, eligible)
    if (
        previous_leader_index is not None
        and 0 <= previous_leader_index < len(on_road)
        and bool(on_road[previous_leader_index].item())
    ):
        return previous_leader_index
    if bool(on_road.any()):
        return _furthest_frontier_index(frontier, forward_position, on_road)
    return int(torch.argmin(course_lateral.abs()).item())


def _follow_on_road_leader(
    base_env: ManagerBasedRlEnv,
    camera_state: CameraFollowState,
    previous_leader_index: int | None,
    dt_s: float,
    minimum_camera_x_m: float | None = -RACE_CAMERA_LEAD_M,
    camera_lead_m: float = RACE_CAMERA_LEAD_M,
    camera_max_speed_mps: float = RACE_CAMERA_MAX_SPEED_MPS,
    camera_max_accel_mps2: float = RACE_CAMERA_MAX_ACCEL_MPS2,
) -> tuple[CameraFollowState, float, int]:
    """Travel longitudinally with the on-road leader while showing the full road."""
    renderer = base_env._offline_renderer
    if renderer is None:
        raise RuntimeError("Offline renderer is not initialized")
    # Use the reward-side position cache. The asset root position exposed by
    # the offscreen backend can lag the physics state and let the leader leave.
    leader_index = _select_on_road_leader(base_env, previous_leader_index)
    leader_x_m = float(base_env._roll_sprint_forward_position[leader_index].item())
    camera_state = _camera_follow_x(
        camera_state,
        leader_x_m,
        dt_s,
        minimum_x_m=minimum_camera_x_m,
        lead_m=camera_lead_m,
        max_speed_mps=camera_max_speed_mps,
        max_accel_mps2=camera_max_accel_mps2,
    )
    camera_y_m = 0.0
    renderer._cam.lookat[:] = (
        camera_state.x_m,
        camera_y_m,
        RACE_CAMERA_LOOKAT[2],
    )
    return camera_state, camera_y_m, leader_index


def main() -> int:
    args = _parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    temporary_dir = REPO_ROOT / ".tmp" / "codex"
    temporary_dir.mkdir(parents=True, exist_ok=True)
    temporary_output = temporary_dir / f"roll-sprint-recording-{os.getpid()}.mp4"
    if not checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")
    if args.five_robot_showcase and args.recovery_montage:
        raise SystemExit("--five-robot-showcase cannot be combined with --recovery-montage")
    if (
        args.steps < 1
        or args.frame_stride < 1
        or args.width < 2
        or args.height < 2
        or args.playback_speed <= 0.0
        or (args.output_fps is not None and args.output_fps <= 0.0)
    ):
        raise SystemExit(
            "--steps, --frame-stride, --width, --height, --playback-speed, "
            "and --output-fps must be positive"
        )

    configure_torch_backends()
    env_cfg = load_env_cfg(args.task_id, play=True)
    agent_cfg = load_rl_cfg(args.task_id)
    race_robots = 5 if args.five_robot_showcase else 4
    race_lane_spacing = (
        FIVE_RACER_LANE_SPACING if args.five_robot_showcase else RACE_LANE_SPACING
    )
    celebration_state = (
        FinishCelebrationState(race_robots) if args.five_robot_showcase else None
    )
    env_cfg.scene.num_envs = race_robots
    env_cfg.scene.env_spacing = race_lane_spacing
    env_cfg.scene.terrain.env_spacing = race_lane_spacing
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
    env_cfg.viewer.max_extra_envs = race_robots - 1
    env_cfg.viewer.width = args.width
    env_cfg.viewer.height = args.height
    reset_cfg = env_cfg.events["set_roll_sprint_state"]
    reset_cfg.params["standing_prob"] = 1.0
    reset_cfg.params["midroll_prob"] = 0.0
    reset_cfg.params["postroll_prob"] = 0.0
    reset_cfg.params["crouch_prob"] = 0.0
    reset_cfg.params["ground_recovery_prob"] = 0.0
    reset_cfg.params["yaw_range"] = (0.0, 0.0)

    base_env = ManagerBasedRlEnv(
        cfg=env_cfg,
        device=args.device,
        render_mode="rgb_array",
    )
    env, policy = _load_policy_then_arrange_start(
        base_env=base_env,
        agent_cfg=agent_cfg,
        task_id=args.task_id,
        checkpoint=checkpoint,
        device=args.device,
        recovery_montage=args.recovery_montage,
        seed=args.seed,
        race_lane_spacing=race_lane_spacing,
        celebration_state=celebration_state,
    )
    camera_state = CameraFollowState(RACE_CAMERA_LOOKAT[0])
    leader_index = 0
    is_backroll = _roll_direction_for_task(args.task_id) < 0.0
    minimum_camera_x_m = None if is_backroll else -RACE_CAMERA_LEAD_M
    camera_lead_m = BACKROLL_CAMERA_LEAD_M if is_backroll else RACE_CAMERA_LEAD_M
    camera_max_speed_mps = (
        BACKROLL_CAMERA_MAX_SPEED_MPS if is_backroll else RACE_CAMERA_MAX_SPEED_MPS
    )
    camera_max_accel_mps2 = (
        BACKROLL_CAMERA_MAX_ACCEL_MPS2
        if is_backroll
        else RACE_CAMERA_MAX_ACCEL_MPS2
    )

    writer: subprocess.Popen[bytes] | None = None
    try:
        for step in range(args.steps):
            with torch.inference_mode():
                observations = env.get_observations()
                actions = policy(observations)
                env.step(actions)
            if celebration_state is not None:
                credited_frontier_m = (
                    base_env._roll_sprint_forward_frontier
                    - base_env._roll_sprint_forward_origin
                ).clamp_min(0.0)
                celebration_state.update(
                    credited_frontier_m,
                    elapsed_s=(step + 1) * policy_dt,
                )
                celebration_state.arm_stop_after_finishers(
                    SHOWCASE_FINISHER_TARGET,
                    SHOWCASE_POST_FINISH_HOLD_SECONDS,
                )
            if not args.recovery_montage:
                camera_state, _camera_y_m, leader_index = _follow_on_road_leader(
                    base_env,
                    camera_state,
                    leader_index,
                    policy_dt,
                    minimum_camera_x_m=minimum_camera_x_m,
                    camera_lead_m=camera_lead_m,
                    camera_max_speed_mps=camera_max_speed_mps,
                    camera_max_accel_mps2=camera_max_accel_mps2,
                )
            if (step + 1) % args.frame_stride != 0:
                continue
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
                    input_fps=(
                        _recording_fps(policy_dt, args.frame_stride)
                        * args.playback_speed
                    ),
                    output_fps=args.output_fps,
                )
            assert writer.stdin is not None
            writer.stdin.write(frame.tobytes())
            if celebration_state is not None and celebration_state.stop_due:
                break
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
    if celebration_state is not None:
        finishers = [
            index + 1
            for index, finish_time_s in enumerate(celebration_state.finish_times_s)
            if finish_time_s is not None
        ]
        print(
            "[roll-sprint-video] valid 10 m finishers: "
            f"{finishers} at {celebration_state.finish_times_s}"
        )
        print(
            "[roll-sprint-video] stopped after "
            f"{SHOWCASE_FINISHER_TARGET} of {len(celebration_state.finish_times_s)} "
            f"finishers at {celebration_state.current_time_s:.2f}s simulation time"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

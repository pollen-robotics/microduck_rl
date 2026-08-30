#!/usr/bin/env python3
"""Evaluate a frozen repeated-roll checkpoint with independent physics gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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

from mjlab_microduck.tasks import mdp as microduck_mdp

TASK_ID = "Mjlab-Roll-Sprint-Flat-MicroDuck"
TARGET_ANGLE = 2.0 * math.pi
HEAD_WINDOW = (math.radians(20.0), math.radians(170.0))
HEAD_TOP_AXIS = (0.882, 0.0, 0.471)
HEAD_TOP_DOWN_MIN = 0.3
FLAT_FULL = 0.5
FLAT_ZERO = math.sin(math.radians(60.0))
MIN_FORWARD_RATE = 0.5
MAX_DISTANCE_PER_RAD = 0.12
RECOVERY_MAX_FORWARD_RATE = 3.0
RECOVERY_UPRIGHT_COS = math.cos(math.radians(50.0))
RECOVERY_LATERAL_Z = math.sin(math.radians(35.0))
RECOVERY_HOLD_STEPS = 3
RACE_LANE_SPACING = 0.28
STRAIGHT_LANE_MAX_DRIFT_M = 0.08
STRAIGHT_LANE_MAX_YAW_DEVIATION_DEG = 20.0
PROMOTION = {
    "repeated_roll_rate": 0.75,
    "mean_valid_roll_count": 2.0,
    "mean_recovered_and_rerolled_count": 1.0,
    "mean_roll_linked_distance_m": 0.35,
    "mean_roll_linked_speed_mps": 0.058,
    "p95_lateral_drift_m": 0.08,
    "mean_uncredited_positive_displacement_m": 0.08,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w", encoding="utf-8", suffix=".tmp", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def heading_from_quat(quat: torch.Tensor) -> torch.Tensor:
    """Planar forward heading from body y, stable at vertical pitch."""
    quat = torch.nan_to_num(quat, nan=0.0)
    w, x, y, z = quat.unbind(dim=-1)
    body_y_x = 2.0 * (x * y - w * z)
    body_y_y = 1.0 - 2.0 * (x.square() + z.square())
    heading = torch.stack((body_y_y, -body_y_x), dim=-1)
    return heading / heading.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)


def lateral_axis_z(quat: torch.Tensor) -> torch.Tensor:
    return 2.0 * (quat[:, 2] * quat[:, 3] + quat[:, 0] * quat[:, 1])


def head_top_down(head_quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = head_quat.unbind(dim=-1)
    a, b, c = HEAD_TOP_AXIS
    axis_world_z = (
        2.0 * (x * z - w * y) * a
        + 2.0 * (y * z + w * x) * b
        + (1.0 - 2.0 * (x.square() + y.square())) * c
    )
    return axis_world_z < -HEAD_TOP_DOWN_MIN


def sensor_contact(env: ManagerBasedRlEnv, name: str) -> torch.Tensor:
    sensor = env.scene.sensors.get(name)
    if sensor is None or sensor.data.found is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    found = sensor.data.found
    return (found.view(found.shape[0], -1) > 0).any(dim=-1)


class RollCycleAuditor:
    """Independent cycle reconstruction that never reads reward state buffers."""

    def __init__(
        self,
        initial_position_xy: torch.Tensor,
        initial_root_quat: torch.Tensor,
        initial_vertical_velocity: torch.Tensor,
        step_dt: float,
    ) -> None:
        self.step_dt = step_dt
        self.heading = heading_from_quat(initial_root_quat)
        self.lateral = torch.stack((-self.heading[:, 1], self.heading[:, 0]), dim=-1)
        self.start_position = initial_position_xy.clone()
        self.last_position = initial_position_xy.clone()
        self.previous_vertical_velocity = initial_vertical_velocity.clone()
        count = initial_position_xy.shape[0]
        device = initial_position_xy.device
        zero = torch.zeros(count, device=device)
        false = torch.zeros(count, dtype=torch.bool, device=device)
        self.accum = zero.clone()
        self.head_latch = false.clone()
        self.lateral_invalid = false.clone()
        self.cycle_start_forward = zero.clone()
        self.forward_frontier = zero.clone()
        self.linked_distance = zero.clone()
        self.valid_count = torch.zeros(count, dtype=torch.long, device=device)
        self.invalid_count = torch.zeros(count, dtype=torch.long, device=device)
        self.awaiting_recovery = false.clone()
        self.recovery_hold_steps = torch.zeros(count, dtype=torch.long, device=device)
        self.recovery_latency_steps = torch.zeros(
            count, dtype=torch.long, device=device
        )
        self.recovery_latency_total_steps = torch.zeros(
            count, dtype=torch.long, device=device
        )
        self.recovery_count = torch.zeros(count, dtype=torch.long, device=device)
        self.recovered_cycle_armed = false.clone()
        self.recovered_and_rerolled_count = torch.zeros(
            count, dtype=torch.long, device=device
        )
        self.head_top_contact_count = torch.zeros(
            count, dtype=torch.long, device=device
        )
        self.max_lateral_drift = zero.clone()
        self.max_heading_deviation = zero.clone()
        self.peak_angular_speed = zero.clone()
        self.peak_impact_acceleration = zero.clone()
        self.nan_seen = false.clone()

    def observe(
        self,
        *,
        position_xy: torch.Tensor,
        root_quat: torch.Tensor,
        head_quat: torch.Tensor,
        linear_velocity_w: torch.Tensor,
        angular_velocity_b: torch.Tensor,
        support: torch.Tensor,
        foot_support: torch.Tensor,
        head_contact: torch.Tensor,
        active: torch.Tensor | None = None,
    ) -> None:
        if active is None:
            active = torch.ones_like(support)
        finite = (
            torch.isfinite(position_xy).all(dim=-1)
            & torch.isfinite(root_quat).all(dim=-1)
            & torch.isfinite(head_quat).all(dim=-1)
            & torch.isfinite(linear_velocity_w).all(dim=-1)
            & torch.isfinite(angular_velocity_b).all(dim=-1)
        )
        self.nan_seen |= active & ~finite
        active = active & finite
        position_xy = torch.nan_to_num(position_xy, nan=0.0)
        root_quat = torch.nan_to_num(root_quat, nan=0.0)
        head_quat = torch.nan_to_num(head_quat, nan=0.0)
        linear_velocity_w = torch.nan_to_num(linear_velocity_w, nan=0.0)
        angular_velocity_b = torch.nan_to_num(angular_velocity_b, nan=0.0)

        lateral_z = lateral_axis_z(root_quat).abs()
        flat_u = torch.clamp(
            (FLAT_ZERO - lateral_z) / (FLAT_ZERO - FLAT_FULL), 0.0, 1.0
        )
        flatness = flat_u.square() * (3.0 - 2.0 * flat_u)
        omega = angular_velocity_b[:, 1]
        upright_cos = 1.0 - 2.0 * (root_quat[:, 1].square() + root_quat[:, 2].square())
        awaiting_before = self.awaiting_recovery
        recovery_candidate = (
            active
            & awaiting_before
            & foot_support
            & ~head_contact
            & (upright_cos >= RECOVERY_UPRIGHT_COS)
            & (lateral_z <= RECOVERY_LATERAL_Z)
            & (omega <= RECOVERY_MAX_FORWARD_RATE)
        )
        old_recovery_hold = self.recovery_hold_steps
        recovery_hold = torch.where(
            recovery_candidate,
            old_recovery_hold + 1,
            torch.zeros_like(old_recovery_hold),
        )
        recovered = (
            recovery_candidate
            & (old_recovery_hold < RECOVERY_HOLD_STEPS)
            & (recovery_hold >= RECOVERY_HOLD_STEPS)
        )
        recovery_latency = torch.where(
            active & awaiting_before,
            self.recovery_latency_steps + 1,
            self.recovery_latency_steps,
        )
        self.recovery_latency_total_steps += torch.where(
            recovered,
            recovery_latency,
            torch.zeros_like(recovery_latency),
        )
        self.recovery_count += recovered.to(torch.long)

        valid_rotation = support.float() * flatness
        rotation_eligible = active & ~awaiting_before & ~recovered
        signed_delta = omega * self.step_dt * valid_rotation * rotation_eligible.float()
        old_accum = torch.where(
            awaiting_before | recovered,
            torch.zeros_like(self.accum),
            self.accum,
        )
        candidate_accum = torch.clamp(old_accum + signed_delta, min=0.0)
        new_accum = torch.where(active, candidate_accum, self.accum)

        in_head_window = (new_accum > HEAD_WINDOW[0]) & (new_accum < HEAD_WINDOW[1])
        top_contact = (
            active & support & head_contact & in_head_window & head_top_down(head_quat)
        )
        self.head_top_contact_count += (top_contact & ~self.head_latch).to(torch.long)
        old_head_latch = torch.where(
            recovered, torch.zeros_like(self.head_latch), self.head_latch
        )
        new_head_latch = old_head_latch | top_contact
        lateral_violation = (
            active
            & ~awaiting_before
            & support
            & (omega >= MIN_FORWARD_RATE)
            & (lateral_z > FLAT_ZERO)
        )
        old_lateral_invalid = torch.where(
            recovered,
            torch.zeros_like(self.lateral_invalid),
            self.lateral_invalid,
        )
        new_lateral_invalid = old_lateral_invalid | lateral_violation

        displacement = position_xy - self.start_position
        forward_position = (displacement * self.heading).sum(dim=-1)

        completed = active & ~awaiting_before & (new_accum >= TARGET_ANGLE)
        valid = completed & new_head_latch & ~new_lateral_invalid
        invalid = completed & ~valid
        rotation_budget = MAX_DISTANCE_PER_RAD * torch.clamp(
            new_accum, min=0.0, max=TARGET_ANGLE
        )
        cycle_net_advance = torch.clamp(
            forward_position - self.cycle_start_forward, min=0.0
        )
        new_frontier_advance = torch.clamp(
            forward_position - self.forward_frontier, min=0.0
        )
        credited_distance = torch.minimum(
            rotation_budget,
            torch.minimum(cycle_net_advance, new_frontier_advance),
        )
        self.valid_count += valid.to(torch.long)
        self.invalid_count += invalid.to(torch.long)
        recovered_rerolled = valid & self.recovered_cycle_armed
        self.recovered_and_rerolled_count += recovered_rerolled.to(torch.long)
        self.linked_distance += torch.where(
            valid, credited_distance, torch.zeros_like(credited_distance)
        )
        self.forward_frontier = torch.where(
            valid,
            torch.maximum(self.forward_frontier, forward_position),
            self.forward_frontier,
        )
        self.cycle_start_forward = torch.where(
            completed | recovered, forward_position, self.cycle_start_forward
        )
        self.accum = torch.where(
            completed | recovered,
            torch.zeros_like(new_accum),
            new_accum,
        )
        self.head_latch = torch.where(
            completed | recovered,
            torch.zeros_like(new_head_latch),
            new_head_latch,
        )
        self.lateral_invalid = torch.where(
            completed | recovered,
            torch.zeros_like(new_lateral_invalid),
            new_lateral_invalid,
        )
        self.awaiting_recovery = torch.where(
            valid,
            torch.ones_like(awaiting_before),
            torch.where(recovered, torch.zeros_like(awaiting_before), awaiting_before),
        )
        self.recovered_cycle_armed = torch.where(
            valid,
            torch.zeros_like(self.recovered_cycle_armed),
            torch.where(
                recovered,
                torch.ones_like(self.recovered_cycle_armed),
                self.recovered_cycle_armed,
            ),
        )
        reset_recovery_clock = recovered | valid
        self.recovery_hold_steps = torch.where(
            reset_recovery_clock,
            torch.zeros_like(recovery_hold),
            recovery_hold,
        )
        self.recovery_latency_steps = torch.where(
            reset_recovery_clock,
            torch.zeros_like(recovery_latency),
            recovery_latency,
        )

        lateral_drift = (displacement * self.lateral).sum(dim=-1).abs()
        self.max_lateral_drift = torch.where(
            active,
            torch.maximum(self.max_lateral_drift, lateral_drift),
            self.max_lateral_drift,
        )
        current_heading = heading_from_quat(root_quat)
        heading_dot = (self.heading * current_heading).sum(dim=-1).clamp(-1.0, 1.0)
        heading_cross = (
            self.heading[:, 0] * current_heading[:, 1]
            - self.heading[:, 1] * current_heading[:, 0]
        )
        heading_deviation = torch.atan2(heading_cross, heading_dot).abs()
        self.max_heading_deviation = torch.where(
            active,
            torch.maximum(self.max_heading_deviation, heading_deviation),
            self.max_heading_deviation,
        )
        self.peak_angular_speed = torch.where(
            active,
            torch.maximum(self.peak_angular_speed, omega.abs()),
            self.peak_angular_speed,
        )
        vertical_acceleration = (
            linear_velocity_w[:, 2] - self.previous_vertical_velocity
        ) / self.step_dt
        self.peak_impact_acceleration = torch.where(
            active,
            torch.maximum(self.peak_impact_acceleration, vertical_acceleration.abs()),
            self.peak_impact_acceleration,
        )
        self.previous_vertical_velocity = torch.where(
            active,
            linear_velocity_w[:, 2],
            self.previous_vertical_velocity,
        )
        self.last_position = torch.where(
            active.unsqueeze(-1), position_xy, self.last_position
        )

    def summary(self, duration_s: float) -> dict[str, float | int]:
        displacement = self.last_position - self.start_position
        raw_forward = (displacement * self.heading).sum(dim=-1)
        uncredited = torch.clamp(
            torch.clamp(raw_forward, min=0.0) - self.linked_distance,
            min=0.0,
        )
        repeated = self.recovered_and_rerolled_count >= 1
        total_recoveries = int(self.recovery_count.sum().item())
        mean_recovery_latency = (
            float(self.recovery_latency_total_steps.sum().item())
            * self.step_dt
            / max(total_recoveries, 1)
        )
        max_heading_deviation_deg = torch.rad2deg(self.max_heading_deviation)
        straight_lane_pass = (
            (self.max_lateral_drift <= STRAIGHT_LANE_MAX_DRIFT_M)
            & (
                max_heading_deviation_deg
                <= STRAIGHT_LANE_MAX_YAW_DEVIATION_DEG
            )
            & (self.recovered_and_rerolled_count >= 1)
            & ~self.nan_seen
        )
        per_robot = [
            {
                "robot_index": index,
                "net_forward_distance_m": float(raw_forward[index].item()),
                "maximum_lateral_drift_m": float(
                    self.max_lateral_drift[index].item()
                ),
                "maximum_heading_yaw_deviation_deg": float(
                    max_heading_deviation_deg[index].item()
                ),
                "valid_roll_recover_reroll_count": int(
                    self.recovered_and_rerolled_count[index].item()
                ),
                "straight_lane_pass": bool(straight_lane_pass[index].item()),
            }
            for index in range(len(raw_forward))
        ]
        return {
            "mean_raw_forward_distance_m": float(raw_forward.mean().item()),
            "best_raw_forward_distance_m": float(raw_forward.max().item()),
            "mean_credited_forward_frontier_m": float(
                self.linked_distance.mean().item()
            ),
            "best_credited_forward_frontier_m": float(
                self.linked_distance.max().item()
            ),
            "mean_roll_linked_distance_m": float(self.linked_distance.mean().item()),
            "best_roll_linked_distance_m": float(self.linked_distance.max().item()),
            "mean_roll_linked_speed_mps": float(
                (self.linked_distance / duration_s).mean().item()
            ),
            "mean_valid_roll_count": float(self.valid_count.float().mean().item()),
            "total_valid_roll_count": int(self.valid_count.sum().item()),
            "mean_invalid_roll_count": float(self.invalid_count.float().mean().item()),
            "total_invalid_roll_count": int(self.invalid_count.sum().item()),
            "mean_recovery_count": float(self.recovery_count.float().mean().item()),
            "total_recovery_count": total_recoveries,
            "mean_recovered_and_rerolled_count": float(
                self.recovered_and_rerolled_count.float().mean().item()
            ),
            "total_recovered_and_rerolled_count": int(
                self.recovered_and_rerolled_count.sum().item()
            ),
            "mean_recovery_latency_s": mean_recovery_latency,
            "repeated_roll_rate": float(repeated.float().mean().item()),
            "head_top_contact_count": int(self.head_top_contact_count.sum().item()),
            "p95_lateral_drift_m": float(
                torch.quantile(self.max_lateral_drift, 0.95).item()
            ),
            "maximum_lateral_drift_m": float(self.max_lateral_drift.max().item()),
            "maximum_heading_yaw_deviation_deg": float(
                max_heading_deviation_deg.max().item()
            ),
            "per_robot": per_robot,
            "four_robot_batch_straight_lane_pass": bool(
                len(per_robot) == 4 and straight_lane_pass.all().item()
            ),
            "mean_uncredited_positive_displacement_m": float(uncredited.mean().item()),
            "p95_peak_angular_speed_rad_s": float(
                torch.quantile(self.peak_angular_speed, 0.95).item()
            ),
            "maximum_angular_speed_rad_s": float(self.peak_angular_speed.max().item()),
            "p95_peak_impact_acceleration_m_s2": float(
                torch.quantile(self.peak_impact_acceleration, 0.95).item()
            ),
            "maximum_impact_acceleration_m_s2": float(
                self.peak_impact_acceleration.max().item()
            ),
            "nan_env_count": int(self.nan_seen.sum().item()),
        }


def promotion_pass(report: dict[str, object]) -> bool:
    return (
        float(report["repeated_roll_rate"]) >= PROMOTION["repeated_roll_rate"]
        and float(report["mean_valid_roll_count"]) >= PROMOTION["mean_valid_roll_count"]
        and float(report["mean_recovered_and_rerolled_count"])
        >= PROMOTION["mean_recovered_and_rerolled_count"]
        and float(report["mean_roll_linked_distance_m"])
        >= PROMOTION["mean_roll_linked_distance_m"]
        and float(report["mean_roll_linked_speed_mps"])
        >= PROMOTION["mean_roll_linked_speed_mps"]
        and float(report["p95_lateral_drift_m"]) <= PROMOTION["p95_lateral_drift_m"]
        and float(report["mean_uncredited_positive_displacement_m"])
        <= PROMOTION["mean_uncredited_positive_displacement_m"]
        and int(report["nan_env_count"]) == 0
        and int(report["out_of_bounds_env_count"]) == 0
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")
    if args.num_envs != 4 or args.duration <= 0.0:
        raise SystemExit("canonical evaluation requires --num-envs 4 and positive duration")

    configure_torch_backends()
    env_cfg = load_env_cfg(TASK_ID, play=True)
    agent_cfg = load_rl_cfg(TASK_ID)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.scene.env_spacing = RACE_LANE_SPACING
    env_cfg.scene.terrain.env_spacing = RACE_LANE_SPACING
    env_cfg.seed = 0
    env_cfg.auto_reset = False
    env_cfg.episode_length_s = args.duration
    reset_cfg = env_cfg.events["set_roll_sprint_state"]
    reset_cfg.params["standing_prob"] = 1.0
    reset_cfg.params["midroll_prob"] = 0.0
    reset_cfg.params["postroll_prob"] = 0.0
    reset_cfg.params["yaw_range"] = (0.0, 0.0)

    base_env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
    race_origins = microduck_mdp.arrange_roll_sprint_race_start(
        base_env, RACE_LANE_SPACING
    )
    robot = base_env.scene["robot"]
    race_headings = base_env._roll_sprint_heading_w.clone()
    race_forward_starts = microduck_mdp._roll_sprint_forward_position(
        base_env, robot, race_headings
    )
    race_alignment_pass = bool(
        torch.allclose(
            race_forward_starts,
            race_forward_starts[:1].expand_as(race_forward_starts),
            atol=1.0e-7,
        )
        and torch.allclose(
            race_headings,
            torch.tensor(
                [[1.0, 0.0]] * 4,
                device=race_headings.device,
                dtype=race_headings.dtype,
            ),
            atol=1.0e-7,
        )
    )
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
    head_ids, _ = robot.find_bodies("jaw_soft")
    head_id = head_ids[0]
    auditor = RollCycleAuditor(
        robot.data.root_link_pos_w[:, :2],
        robot.data.root_link_quat_w,
        robot.data.root_link_lin_vel_w[:, 2],
        base_env.step_dt,
    )
    alive = torch.ones(args.num_envs, dtype=torch.bool, device=args.device)
    termination_seen = {
        name: torch.zeros_like(alive)
        for name in base_env.termination_manager.active_terms
    }
    steps = round(args.duration / base_env.step_dt)

    try:
        for _ in range(steps):
            with torch.inference_mode():
                observations = env.get_observations()
                actions = policy(observations)
                _, _, dones, _ = env.step(actions)
            auditor.observe(
                position_xy=robot.data.root_link_pos_w[:, :2],
                root_quat=robot.data.root_link_quat_w,
                head_quat=robot.data.body_link_quat_w[:, head_id],
                linear_velocity_w=robot.data.root_link_lin_vel_w,
                angular_velocity_b=robot.data.root_link_ang_vel_b,
                support=sensor_contact(base_env, "robot_ground_contact"),
                foot_support=sensor_contact(base_env, "feet_ground_contact"),
                head_contact=sensor_contact(base_env, "head_ground_contact"),
                active=alive,
            )
            for name in termination_seen:
                termination_seen[name] |= alive & base_env.termination_manager.get_term(
                    name
                )
            done = alive & dones
            if bool(done.any()):
                alive[done] = False
            if not bool(alive.any()):
                break
    finally:
        env.close()

    report: dict[str, object] = {
        "schema_version": 2,
        "task": TASK_ID,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_iteration": int(checkpoint.stem.rsplit("_", 1)[-1]),
        "num_envs": args.num_envs,
        "duration_s": args.duration,
        "canonical_race_alignment": {
            "seed": env_cfg.seed,
            "projected_forward_start_m": race_forward_starts.cpu().tolist(),
            "lane_center_y_m": race_origins[:, 1].cpu().tolist(),
            "reward_heading_xy": race_headings.cpu().tolist(),
            "alignment_pass": race_alignment_pass,
        },
        **auditor.summary(args.duration),
        "termination_counts": {
            name: int(values.sum().item()) for name, values in termination_seen.items()
        },
        "out_of_bounds_env_count": int(
            termination_seen.get("out_of_terrain_bounds", torch.zeros_like(alive))
            .sum()
            .item()
        ),
        "promotion_thresholds": PROMOTION,
    }
    report["promotion_pass"] = promotion_pass(report)
    output = args.output or checkpoint.with_suffix(".roll-sprint-eval.json")
    _write_json_atomic(output.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"[roll-sprint-eval] wrote {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

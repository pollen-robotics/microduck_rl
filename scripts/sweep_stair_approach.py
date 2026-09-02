#!/usr/bin/env python3
"""Sweep closed-loop walker commands in one vectorized stair simulation."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from tempfile import NamedTemporaryFile

import mjlab.tasks  # noqa: F401
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs import mdp as base_mdp
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

from mjlab_microduck.policies import (
    HardStairHandoffPolicy,
    StairApproachSupervisor,
    load_frozen_actor,
    resolve_official_walker_checkpoint,
)
from mjlab_microduck.robot.microduck_constants import MICRODUCK_WALK_ROBOT_CFG

TASK_ID = "Mjlab-Stairs-Route-MicroDuck"
REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--walker-checkpoint", type=Path)
    parser.add_argument("--trials-per-candidate", type=int, default=12)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lateral-gains", nargs="+", type=float, default=(-1.0, 0.0, 1.0))
    parser.add_argument("--heading-gains", nargs="+", type=float, default=(-1.5, 0.0, 1.5))
    parser.add_argument("--cross-track-gains", nargs="+", type=float, default=(-1.0, 0.0, 1.0))
    parser.add_argument("--forward-commands", nargs="+", type=float, default=(0.18, 0.22, 0.26))
    parser.add_argument("--lateral-spawn-range", type=float, default=0.03)
    parser.add_argument("--longitudinal-spawn", type=float, default=0.30)
    parser.add_argument("--yaw-spawn-range-deg", type=float, default=6.0)
    parser.add_argument(
        "--walking-collision-model",
        action="store_true",
        help="Use the exact robot model on which the manufacturer walker was trained.",
    )
    parser.add_argument(
        "--nominal-observations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable sensor corruption while selecting the route supervisor.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / ".tmp" / "codex" / "stair-approach-sweep.json",
    )
    return parser.parse_args()


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w", encoding="utf-8", suffix=".tmp", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _p95(values: torch.Tensor) -> float:
    return float(torch.quantile(values.to(torch.float32), 0.95).item())


def main() -> int:
    args = _parse_args()
    if args.trials_per_candidate < 1 or args.steps < 1:
        raise SystemExit("Trial and step counts must be positive")
    try:
        walker_checkpoint = resolve_official_walker_checkpoint(
            REPO_ROOT,
            args.walker_checkpoint,
        )
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error

    candidates = list(
        itertools.product(
            args.lateral_gains,
            args.heading_gains,
            args.cross_track_gains,
            args.forward_commands,
        )
    )
    num_envs = len(candidates) * args.trials_per_candidate
    configure_torch_backends()
    env_cfg = load_env_cfg(TASK_ID, play=True)
    agent_cfg = load_rl_cfg(TASK_ID)
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = 0
    if args.walking_collision_model:
        env_cfg.scene.entities = {"robot": MICRODUCK_WALK_ROBOT_CFG}
    if args.nominal_observations:
        actor_terms = env_cfg.observations["actor"].terms
        env_cfg.observations["actor"].enable_corruption = False
        actor_terms["base_ang_vel"].func = base_mdp.base_ang_vel
        actor_terms["base_ang_vel"].params = {}
        actor_terms["base_ang_vel"].delay_min_lag = 0
        actor_terms["base_ang_vel"].delay_max_lag = 0
        actor_terms["projected_gravity"].func = base_mdp.projected_gravity
        actor_terms["projected_gravity"].params = {}
        actor_terms["projected_gravity"].delay_min_lag = 0
        actor_terms["projected_gravity"].delay_max_lag = 0
        actor_terms["joint_pos"].params["biased"] = False
        for event_name in ("foot_friction", "encoder_bias", "base_com"):
            env_cfg.events.pop(event_name, None)
    pose_range = env_cfg.events["reset_base"].params["pose_range"]
    pose_range["x"] = (args.longitudinal_spawn, args.longitudinal_spawn)
    pose_range["y"] = (-args.lateral_spawn_range, args.lateral_spawn_range)
    yaw_range = math.radians(args.yaw_spawn_range_deg)
    pose_range["yaw"] = (-yaw_range, yaw_range)

    base_env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=args.device)
    walker = load_frozen_actor(runner, walker_checkpoint, device=args.device)

    def expanded(index: int) -> torch.Tensor:
        values = [candidate[index] for candidate in candidates]
        return torch.tensor(values, device=args.device).repeat_interleave(
            args.trials_per_candidate
        )

    policy = HardStairHandoffPolicy(
        walker,
        walker,
        base_env,
        blend_steps=4,
        approach_supervisor=StairApproachSupervisor(
            lateral_gain=expanded(0),
            heading_gain=expanded(1),
            cross_track_heading_gain=expanded(2),
            forward_command_mps=expanded(3),
        ),
    )
    device = torch.device(args.device)
    closest_distance_error = torch.full((num_envs,), torch.inf, device=device)
    closest_abs_lateral = torch.full((num_envs,), torch.inf, device=device)
    closest_abs_heading = torch.full((num_envs,), torch.inf, device=device)
    closest_time_s = torch.full((num_envs,), torch.inf, device=device)
    pre_handoff_fall = torch.zeros(num_envs, dtype=torch.bool, device=device)
    pre_handoff_contact = torch.zeros(num_envs, dtype=torch.bool, device=device)
    pre_handoff_nan = torch.zeros(num_envs, dtype=torch.bool, device=device)
    velocity_low_frames = torch.zeros(num_envs, dtype=torch.long, device=device)
    velocity_high_frames = torch.zeros(num_envs, dtype=torch.long, device=device)

    try:
        for step in range(args.steps):
            with torch.inference_mode():
                observations = env.get_observations()
                actions = policy(observations)
                env.step(actions)
            estimate = policy.route_estimator.estimate()
            still_walking = ~policy.specialist_latched
            in_window = (
                (estimate.distance_to_next_face_m >= 0.080)
                & (estimate.distance_to_next_face_m <= 0.120)
            )
            velocity_low_frames += (
                still_walking & in_window & (estimate.forward_velocity_mps < 0.160)
            ).to(torch.long)
            velocity_high_frames += (
                still_walking & in_window & (estimate.forward_velocity_mps > 0.300)
            ).to(torch.long)
            distance_error = (estimate.distance_to_next_face_m - 0.10).abs()
            improved = still_walking & (distance_error < closest_distance_error)
            closest_distance_error[improved] = distance_error[improved]
            closest_abs_lateral[improved] = estimate.lateral_error_m[improved].abs()
            closest_abs_heading[improved] = estimate.heading_error_rad[improved].abs()
            closest_time_s[improved] = (step + 1) * base_env.step_dt
            robot = base_env.scene["robot"].data
            origins = base_env.scene.terrain.env_origins
            local_z = robot.root_link_pos_w[:, 2] - origins[:, 2]
            pre_handoff_fall |= still_walking & (
                (estimate.upright_score < 0.50) | (local_z < 0.070)
            )
            pre_handoff_contact |= still_walking & estimate.non_foot_contact
            pre_handoff_nan |= still_walking & ~estimate.finite
    finally:
        env.close()

    results: list[dict[str, object]] = []
    for candidate_index, candidate in enumerate(candidates):
        start = candidate_index * args.trials_per_candidate
        stop = start + args.trials_per_candidate
        selection = slice(start, stop)
        handoff_mask = torch.isfinite(policy.handoff_distance_m[selection])
        handoff_count = int(handoff_mask.sum().item())
        if handoff_count:
            lateral_score = _p95(
                policy.handoff_lateral_error_m[selection][handoff_mask].abs()
            )
            heading_score = _p95(
                torch.rad2deg(
                    policy.handoff_heading_error_rad[selection][handoff_mask].abs()
                )
            )
            time_score = _p95(
                policy.handoff_control_step[selection][handoff_mask]
                * base_env.step_dt
            )
        else:
            lateral_score = _p95(closest_abs_lateral[selection])
            heading_score = _p95(
                torch.rad2deg(closest_abs_heading[selection])
            )
            time_score = _p95(closest_time_s[selection])
        lateral_gain, heading_gain, cross_track_gain, forward_command = candidate
        result: dict[str, object] = {
            "candidate_index": candidate_index,
            "lateral_gain": lateral_gain,
            "heading_gain": heading_gain,
            "cross_track_heading_gain": cross_track_gain,
            "forward_command_mps": forward_command,
            "handoff_count": handoff_count,
            "handoff_rate": handoff_count / args.trials_per_candidate,
            "distance_window_count": int(
                policy.distance_window_seen[selection].sum().item()
            ),
            "p95_abs_lateral_m": lateral_score,
            "p95_abs_heading_deg": heading_score,
            "p95_approach_time_s": time_score,
            "fall_count": int(pre_handoff_fall[selection].sum().item()),
            "non_foot_contact_count": int(
                pre_handoff_contact[selection].sum().item()
            ),
            "nan_count": int(pre_handoff_nan[selection].sum().item()),
            "velocity_low_frame_count": int(
                velocity_low_frames[selection].sum().item()
            ),
            "velocity_high_frame_count": int(
                velocity_high_frames[selection].sum().item()
            ),
            "rejection_frame_counts": dict(
                zip(
                    (
                        "finite",
                        "lateral",
                        "heading",
                        "velocity",
                        "upright",
                        "contact",
                    ),
                    policy.handoff_rejection_counts[selection]
                    .sum(dim=0)
                    .tolist(),
                    strict=True,
                )
            ),
        }
        result["rank_key"] = [
            -float(result["handoff_rate"]),
            lateral_score,
            heading_score,
            time_score,
        ]
        results.append(result)

    results.sort(key=lambda result: tuple(result["rank_key"]))
    payload: dict[str, object] = {
        "schema_version": 1,
        "task": TASK_ID,
        "walker_checkpoint": str(walker_checkpoint),
        "walking_collision_model": args.walking_collision_model,
        "nominal_observations": args.nominal_observations,
        "num_candidates": len(candidates),
        "trials_per_candidate": args.trials_per_candidate,
        "num_envs": num_envs,
        "steps": args.steps,
        "selected": results[0],
        "results": results,
    }
    _write_json_atomic(args.output.resolve(), payload)
    print(json.dumps(payload["selected"], indent=2, sort_keys=True))
    print(f"[stair-approach-sweep] wrote {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

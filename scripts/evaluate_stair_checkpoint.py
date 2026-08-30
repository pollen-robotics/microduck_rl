#!/usr/bin/env python3
"""Evaluate a frozen MicroDuck checkpoint on the full 170 mm stair route.

This is deliberately separate from training.  It runs deterministic play
episodes from the flat runway, records physical route/height progress, and
writes an atomic JSON report that can gate the single-robot live preview.
"""

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

from mjlab_microduck.policies import (
    OFFICIAL_WALKER_RELATIVE_PATH,
    HardStairHandoffPolicy,
    StairApproachSupervisor,
    load_actor_pair,
    resolve_official_walker_checkpoint,
)
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.stair_walk_state_bank import (
    BANK_SCHEMA_VERSION,
    STANDARD_NUM_STEPS,
    STANDARD_TREAD_DEPTH_M,
    capture_walk_state_rows,
    concatenate_walk_state_rows,
    walk_state_count,
)

TASK_ID = "Mjlab-Stairs-Route-MicroDuck"
REPO_ROOT = Path(__file__).resolve().parents[1]
STANDARD_RISER_HEIGHT_M = 0.170
STAIR_CORRIDOR_HALF_WIDTH_M = 0.90 * 0.40


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w", encoding="utf-8", suffix=".tmp", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _write_bank_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(suffix=".tmp", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Stair specialist checkpoint.")
    parser.add_argument(
        "--walker-checkpoint",
        type=Path,
        default=REPO_ROOT / OFFICIAL_WALKER_RELATIVE_PATH,
        help=(
            "Pinned immutable manufacturer walking checkpoint. The file hash is "
            "verified before evaluation."
        ),
    )
    parser.add_argument(
        "--single-actor",
        action="store_true",
        help="Disable the walker dispatcher for an explicit specialist-only diagnostic.",
    )
    parser.add_argument(
        "--stair-face-local-x",
        type=float,
        default=0.66,
        help="Local x coordinate of the next riser face.",
    )
    parser.add_argument(
        "--handoff-blend-steps",
        type=int,
        default=4,
        help="Control frames used to blend actors (default: 4).",
    )
    parser.add_argument("--approach-lateral-gain", type=float, default=2.0)
    parser.add_argument("--approach-heading-gain", type=float, default=1.5)
    parser.add_argument(
        "--approach-cross-track-heading-gain", type=float, default=0.0
    )
    parser.add_argument("--approach-forward-command", type=float, default=0.30)
    parser.add_argument(
        "--approach-longitudinal-spawn",
        type=float,
        default=0.30,
        help="Fixed initial local x position in metres.",
    )
    parser.add_argument(
        "--approach-lateral-spawn-range",
        type=float,
        default=0.03,
        help="Uniform randomized initial lateral offset in metres.",
    )
    parser.add_argument(
        "--approach-yaw-spawn-range-deg",
        type=float,
        default=6.0,
        help="Uniform randomized initial heading error in degrees.",
    )
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument(
        "--steps",
        type=int,
        default=1_500,
        help="Control steps. 1500 is one complete 30 second route episode.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--handoff-bank-output",
        type=Path,
        help="Save exact simulator states accepted by the guarded handoff.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")
    try:
        walker_checkpoint = (
            None
            if args.single_actor
            else resolve_official_walker_checkpoint(
                REPO_ROOT,
                args.walker_checkpoint,
            )
        )
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error
    if args.num_envs < 1 or args.steps < 1 or args.handoff_blend_steps < 0:
        raise SystemExit(
            "Environment counts and steps must be positive; "
            "blend steps cannot be negative"
        )

    configure_torch_backends()
    env_cfg = load_env_cfg(TASK_ID, play=True)
    agent_cfg = load_rl_cfg(TASK_ID)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = 0
    pose_range = env_cfg.events["reset_base"].params["pose_range"]
    pose_range["x"] = (
        args.approach_longitudinal_spawn,
        args.approach_longitudinal_spawn,
    )
    pose_range["y"] = (
        -args.approach_lateral_spawn_range,
        args.approach_lateral_spawn_range,
    )
    yaw_spawn_rad = math.radians(args.approach_yaw_spawn_range_deg)
    pose_range["yaw"] = (-yaw_spawn_rad, yaw_spawn_rad)

    base_env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=args.device)
    if walker_checkpoint is not None:
        walker, specialist = load_actor_pair(
            runner,
            walker_checkpoint,
            checkpoint,
            device=args.device,
        )
        policy = HardStairHandoffPolicy(
            walker,
            specialist,
            base_env,
            stair_face_local_x_m=args.stair_face_local_x,
            blend_steps=args.handoff_blend_steps,
            approach_supervisor=StairApproachSupervisor(
                lateral_gain=args.approach_lateral_gain,
                heading_gain=args.approach_heading_gain,
                cross_track_heading_gain=(
                    args.approach_cross_track_heading_gain
                ),
                forward_command_mps=args.approach_forward_command,
            ),
        )
        policy_mode = (
            "hard_walker_to_specialist_handoff"
            if args.handoff_blend_steps == 0
            else "blended_walker_to_specialist_handoff"
        )
    else:
        runner.load(
            str(checkpoint),
            load_cfg={"actor": True},
            strict=True,
            map_location=args.device,
        )
        policy = runner.get_inference_policy(device=args.device)
        policy_mode = "single_actor"

    device = torch.device(args.device)
    max_x = torch.full((args.num_envs,), -torch.inf, device=device)
    max_z = torch.full((args.num_envs,), -torch.inf, device=device)
    max_any_x = torch.full((args.num_envs,), -torch.inf, device=device)
    max_any_z = torch.full((args.num_envs,), -torch.inf, device=device)
    max_abs_y = torch.zeros(args.num_envs, device=device)
    max_upright = torch.zeros(args.num_envs, device=device)
    success_count = torch.zeros(args.num_envs, dtype=torch.long, device=device)
    previous_latch = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
    first_riser_count = torch.zeros(
        args.num_envs, dtype=torch.long, device=device
    )
    previous_first_riser = torch.zeros(
        args.num_envs, dtype=torch.bool, device=device
    )
    secured_tread_count = torch.zeros(
        args.num_envs, dtype=torch.long, device=device
    )
    previous_secured_tread = torch.zeros(
        args.num_envs, dtype=torch.bool, device=device
    )
    pre_handoff_fall = torch.zeros(
        args.num_envs, dtype=torch.bool, device=device
    )
    pre_handoff_non_foot_contact = torch.zeros(
        args.num_envs, dtype=torch.bool, device=device
    )
    pre_handoff_nan = torch.zeros(
        args.num_envs, dtype=torch.bool, device=device
    )
    pre_handoff_max_abs_y = torch.zeros(args.num_envs, device=device)
    handoff_state_chunks: list[dict[str, object]] = []
    previous_policy_latched = torch.zeros(
        args.num_envs, dtype=torch.bool, device=device
    )

    try:
        for _ in range(args.steps):
            with torch.inference_mode():
                observations = env.get_observations()
                actions = policy(observations)
                if (
                    args.handoff_bank_output is not None
                    and isinstance(policy, HardStairHandoffPolicy)
                ):
                    newly_latched = (
                        policy.specialist_latched & ~previous_policy_latched
                    )
                    if bool(newly_latched.any()):
                        handoff_state_chunks.append(
                            capture_walk_state_rows(
                                base_env,
                                torch.nonzero(
                                    newly_latched,
                                    as_tuple=False,
                                ).squeeze(-1),
                            )
                        )
                    previous_policy_latched |= policy.specialist_latched
                env.step(actions)

            robot = base_env.scene["robot"]
            origins = base_env.scene.terrain.env_origins
            root_pos = robot.data.root_link_pos_w
            local_x = root_pos[:, 0] - origins[:, 0]
            local_y = root_pos[:, 1] - origins[:, 1]
            local_z = root_pos[:, 2] - origins[:, 2]
            quat = robot.data.root_link_quat_w
            upright = torch.clamp(
                1.0 - 2.0 * (quat[:, 1].square() + quat[:, 2].square()),
                min=0.0,
                max=1.0,
            )
            finite_x = torch.nan_to_num(local_x, nan=-torch.inf)
            finite_z = torch.nan_to_num(local_z, nan=-torch.inf)
            abs_y = torch.abs(torch.nan_to_num(local_y, nan=1.0e9))
            in_corridor = abs_y <= STAIR_CORRIDOR_HALF_WIDTH_M
            max_x = torch.maximum(
                max_x, torch.where(in_corridor, finite_x, -torch.inf)
            )
            max_z = torch.maximum(
                max_z, torch.where(in_corridor, finite_z, -torch.inf)
            )
            max_any_x = torch.maximum(max_any_x, finite_x)
            max_any_z = torch.maximum(max_any_z, finite_z)
            max_abs_y = torch.maximum(max_abs_y, abs_y)
            max_upright = torch.maximum(max_upright, torch.nan_to_num(upright, nan=0.0))

            if isinstance(policy, HardStairHandoffPolicy):
                estimate = policy.route_estimator.estimate()
                still_walking = ~policy.specialist_latched
                pre_handoff_max_abs_y = torch.maximum(
                    pre_handoff_max_abs_y,
                    torch.where(
                        still_walking,
                        torch.nan_to_num(
                            estimate.lateral_error_m.abs(),
                            nan=1.0e9,
                        ),
                        torch.zeros_like(estimate.lateral_error_m),
                    ),
                )
                pre_handoff_non_foot_contact |= (
                    still_walking & estimate.non_foot_contact
                )
                pre_handoff_nan |= still_walking & ~estimate.finite
                pre_handoff_fall |= still_walking & (
                    (upright < 0.50) | (local_z < 0.070)
                )

            latch = getattr(base_env, "_stair_goal_latched", previous_latch)
            success_count += (latch & ~previous_latch).to(torch.long)
            previous_latch = latch.clone()
            first_riser = getattr(
                base_env,
                "_stair_first_riser_latched",
                previous_first_riser,
            )
            first_riser_count += (
                first_riser & ~previous_first_riser
            ).to(torch.long)
            previous_first_riser = first_riser.clone()

            # The route task does not reward tread-one settlement, so compute
            # the exact physical gate only in this frozen evaluation. This is
            # the first promotion slice before attempting the full staircase.
            microduck_mdp.stair_first_tread_secured(base_env)
            secured_tread = getattr(
                base_env,
                "_stair_first_tread_secured_latched",
                previous_secured_tread,
            )
            secured_tread_count += (
                secured_tread & ~previous_secured_tread
            ).to(torch.long)
            previous_secured_tread = secured_tread.clone()
            fresh = base_env.episode_length_buf <= 1
            previous_latch[fresh] = False
            previous_first_riser[fresh] = False
            previous_secured_tread[fresh] = False
    finally:
        env.close()

    successes = int(success_count.sum().item())
    first_riser_clearances = int(first_riser_count.sum().item())
    secured_first_treads = int(secured_tread_count.sum().item())
    handoff_samples: list[dict[str, float | int]] = []
    handoff_env_count = 0
    handoff_within_four_seconds = 0
    handoff_p95_abs_lateral_m: float | None = None
    handoff_p95_abs_heading_deg: float | None = None
    handoff_p95_time_s: float | None = None
    handoff_velocity_min_mps: float | None = None
    handoff_velocity_max_mps: float | None = None
    handoff_upright_min: float | None = None
    phase_transition_counts = [0, 0]
    handoff_rejection_counts = [0, 0, 0, 0, 0, 0]
    distance_window_env_count = 0
    if isinstance(policy, HardStairHandoffPolicy):
        handoff_mask = torch.isfinite(policy.handoff_distance_m)
        handoff_env_count = int(handoff_mask.sum().item())
        if handoff_env_count:
            lateral = policy.handoff_lateral_error_m[handoff_mask].abs()
            heading_deg = torch.rad2deg(
                policy.handoff_heading_error_rad[handoff_mask].abs()
            )
            velocity = policy.handoff_forward_velocity_mps[handoff_mask]
            upright_at_handoff = policy.handoff_upright_score[handoff_mask]
            handoff_time_s = (
                policy.handoff_control_step[handoff_mask].to(torch.float32)
                * base_env.step_dt
            )
            handoff_p95_abs_lateral_m = float(
                torch.quantile(lateral, 0.95).item()
            )
            handoff_p95_abs_heading_deg = float(
                torch.quantile(heading_deg, 0.95).item()
            )
            handoff_p95_time_s = float(
                torch.quantile(handoff_time_s, 0.95).item()
            )
            handoff_velocity_min_mps = float(velocity.min().item())
            handoff_velocity_max_mps = float(velocity.max().item())
            handoff_upright_min = float(upright_at_handoff.min().item())
            handoff_within_four_seconds = int((handoff_time_s <= 4.0).sum().item())
            indices = torch.nonzero(handoff_mask, as_tuple=False).squeeze(-1)
            for index in indices.tolist():
                handoff_samples.append(
                    {
                        "env_index": index,
                        "control_step": int(policy.handoff_control_step[index].item()),
                        "distance_to_face_m": float(
                            policy.handoff_distance_m[index].item()
                        ),
                        "lateral_error_m": float(
                            policy.handoff_lateral_error_m[index].item()
                        ),
                        "heading_error_deg": float(
                            torch.rad2deg(
                                policy.handoff_heading_error_rad[index]
                            ).item()
                        ),
                        "forward_velocity_mps": float(
                            policy.handoff_forward_velocity_mps[index].item()
                        ),
                        "upright_score": float(
                            policy.handoff_upright_score[index].item()
                        ),
                    }
                )
        phase_transition_counts = (
            policy.phase_transition_counts.sum(dim=0).tolist()
        )
        handoff_rejection_counts = (
            policy.handoff_rejection_counts.sum(dim=0).tolist()
        )
        distance_window_env_count = int(policy.distance_window_seen.sum().item())
    required_handoffs = math.ceil(0.98 * args.num_envs)
    approach_promotion_pass = (
        handoff_within_four_seconds >= required_handoffs
        and handoff_p95_abs_lateral_m is not None
        and handoff_p95_abs_lateral_m <= 0.040
        and handoff_p95_abs_heading_deg is not None
        and handoff_p95_abs_heading_deg <= 8.0
        and not bool(pre_handoff_fall.any())
        and not bool(pre_handoff_nan.any())
    )
    handoff_bank_state_count = 0
    handoff_bank_path: str | None = None
    if args.handoff_bank_output is not None and handoff_state_chunks:
        handoff_states = concatenate_walk_state_rows(handoff_state_chunks)
        handoff_bank_state_count = walk_state_count(handoff_states)
        resolved_bank_output = args.handoff_bank_output.expanduser().resolve()
        _write_bank_atomic(
            resolved_bank_output,
            {
                "schema_version": BANK_SCHEMA_VERSION,
                "metadata": {
                    "task": TASK_ID,
                    "source": "guarded_official_walker_handoff",
                    "walker_checkpoint": str(walker_checkpoint),
                    "walker_checkpoint_sha256": (
                        _sha256(walker_checkpoint) if walker_checkpoint else None
                    ),
                    "riser_height_m": STANDARD_RISER_HEIGHT_M,
                    "tread_depth_m": STANDARD_TREAD_DEPTH_M,
                    "num_steps": STANDARD_NUM_STEPS,
                    "num_states": handoff_bank_state_count,
                    "joint_names": list(base_env.scene["robot"].joint_names),
                    "handoff_criteria": {
                        "distance_to_face_m": [0.08, 0.12],
                        "max_abs_lateral_m": 0.04,
                        "max_abs_heading_deg": 8.0,
                        "forward_velocity_mps": [0.16, 0.30],
                        "min_upright_score": 0.90,
                        "non_foot_contact": False,
                    },
                },
                "states": handoff_states,
            },
        )
        handoff_bank_path = str(resolved_bank_output)
    report: dict[str, object] = {
        "schema_version": 4,
        "task": TASK_ID,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_iteration": int(checkpoint.stem.rsplit("_", 1)[-1]),
        "walker_checkpoint": str(walker_checkpoint) if walker_checkpoint else None,
        "walker_checkpoint_sha256": (
            _sha256(walker_checkpoint) if walker_checkpoint else None
        ),
        "policy_mode": policy_mode,
        "stair_face_local_x_m": args.stair_face_local_x,
        "handoff_count": (
            policy.handoff_count
            if isinstance(policy, HardStairHandoffPolicy)
            else 0
        ),
        "handoff_blend_steps": (
            policy.blend_steps if isinstance(policy, HardStairHandoffPolicy) else None
        ),
        "approach_supervisor": {
            "lateral_gain": args.approach_lateral_gain,
            "heading_gain": args.approach_heading_gain,
            "cross_track_heading_gain": args.approach_cross_track_heading_gain,
            "forward_command_mps": args.approach_forward_command,
        },
        "approach_spawn_randomization": {
            "local_x_m": args.approach_longitudinal_spawn,
            "max_abs_lateral_m": args.approach_lateral_spawn_range,
            "max_abs_yaw_deg": args.approach_yaw_spawn_range_deg,
        },
        "handoff_env_count": handoff_env_count,
        "handoff_rate": handoff_env_count / args.num_envs,
        "handoff_within_four_seconds": handoff_within_four_seconds,
        "handoff_within_four_seconds_rate": (
            handoff_within_four_seconds / args.num_envs
        ),
        "handoff_p95_abs_lateral_m": handoff_p95_abs_lateral_m,
        "handoff_p95_abs_heading_deg": handoff_p95_abs_heading_deg,
        "handoff_p95_time_s": handoff_p95_time_s,
        "handoff_velocity_min_mps": handoff_velocity_min_mps,
        "handoff_velocity_max_mps": handoff_velocity_max_mps,
        "handoff_upright_min": handoff_upright_min,
        "pre_handoff_fall_count": int(pre_handoff_fall.sum().item()),
        "pre_handoff_non_foot_contact_count": int(
            pre_handoff_non_foot_contact.sum().item()
        ),
        "pre_handoff_nan_count": int(pre_handoff_nan.sum().item()),
        "pre_handoff_max_abs_lateral_m": float(
            pre_handoff_max_abs_y.max().item()
        ),
        "phase_transition_counts": {
            "walk_to_blend_or_climb": phase_transition_counts[0],
            "blend_to_climb": phase_transition_counts[1],
        },
        "distance_window_env_count": distance_window_env_count,
        "handoff_rejection_frame_counts": dict(
            zip(
                ("finite", "lateral", "heading", "velocity", "upright", "contact"),
                handoff_rejection_counts,
                strict=True,
            )
        ),
        "approach_promotion_pass": approach_promotion_pass,
        "handoff_samples": handoff_samples,
        "handoff_bank_output": handoff_bank_path,
        "handoff_bank_state_count": handoff_bank_state_count,
        "standard_riser_height_m": STANDARD_RISER_HEIGHT_M,
        "stair_corridor_half_width_m": STAIR_CORRIDOR_HALF_WIDTH_M,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "successes": successes,
        "success_rate": successes / args.num_envs,
        "first_riser_clearances": first_riser_clearances,
        "first_riser_clearance_rate": first_riser_clearances / args.num_envs,
        "secured_first_treads": secured_first_treads,
        "secured_first_tread_rate": secured_first_treads / args.num_envs,
        "mean_max_route_x_m": float(max_x.mean().item()),
        "best_route_x_m": float(max_x.max().item()),
        "mean_max_corridor_route_x_m": float(max_x.mean().item()),
        "best_corridor_route_x_m": float(max_x.max().item()),
        "mean_max_root_height_m": float(max_z.mean().item()),
        "best_root_height_m": float(max_z.max().item()),
        "best_any_route_x_m": float(max_any_x.max().item()),
        "best_any_root_height_m": float(max_any_z.max().item()),
        "maximum_abs_lateral_offset_m": float(max_abs_y.max().item()),
        "mean_max_upright": float(max_upright.mean().item()),
        "verified_full_route": successes > 0,
        "verified_first_riser": secured_first_treads > 0,
    }
    output = args.output or checkpoint.with_suffix(".stair-eval.json")
    _write_json_atomic(output.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"[stair-eval] wrote {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

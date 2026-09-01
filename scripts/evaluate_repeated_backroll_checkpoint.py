#!/usr/bin/env python3
"""Audit consecutive grounded backrolls from deterministic standing starts."""

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

TASK_ID = "Mjlab-Repeated-Backroll-Flat-MicroDuck"
DEFAULT_SEEDS = tuple(range(16))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--required-cycles", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--reset-mode",
        choices=("standing", "precontact"),
        default="standing",
    )
    parser.add_argument("--precontact-z", type=float, default=0.115)
    parser.add_argument("--precontact-pitch-min-deg", type=float, default=160.0)
    parser.add_argument("--precontact-pitch-max-deg", type=float, default=180.0)
    parser.add_argument("--precontact-omega-min", type=float, default=0.5)
    parser.add_argument("--precontact-omega-max", type=float, default=1.0)
    parser.add_argument("--precontact-joint-noise", type=float, default=0.0)
    return parser.parse_args()


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


def _heading_from_quat(quat: torch.Tensor) -> torch.Tensor:
    quat = torch.nan_to_num(quat, nan=0.0)
    w, x, _y, z = quat.unbind(dim=-1)
    body_y_x = 2.0 * (x * _y - w * z)
    body_y_y = 1.0 - 2.0 * (x.square() + z.square())
    heading = torch.stack((body_y_y, -body_y_x), dim=-1)
    return heading / heading.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)


def _checkpoint_iteration(path: Path) -> int | None:
    try:
        return int(path.stem.rsplit("_", 1)[-1])
    except ValueError:
        return None


def evaluate_checkpoint(
    checkpoint: Path,
    *,
    seeds: list[int],
    device: str,
    duration_s: float,
    required_cycles: int,
    reset_mode: str = "standing",
    precontact_z: float = 0.115,
    precontact_pitch_range_deg: tuple[float, float] = (160.0, 180.0),
    precontact_omega_range: tuple[float, float] = (0.5, 1.0),
    precontact_joint_noise: float = 0.0,
) -> dict[str, object]:
    if not seeds or duration_s <= 0.0 or required_cycles < 2:
        raise ValueError("seeds, positive duration, and at least two cycles are required")
    if reset_mode not in {"standing", "precontact"}:
        raise ValueError("reset_mode must be 'standing' or 'precontact'")
    if precontact_z <= 0.0:
        raise ValueError("precontact_z must be positive")
    if precontact_pitch_range_deg[1] < precontact_pitch_range_deg[0]:
        raise ValueError("precontact_pitch_range_deg must be ordered")
    if (
        precontact_omega_range[0] < 0.0
        or precontact_omega_range[1] < precontact_omega_range[0]
    ):
        raise ValueError("precontact_omega_range must be nonnegative and ordered")
    if precontact_joint_noise < 0.0:
        raise ValueError("precontact_joint_noise must be nonnegative")

    configure_torch_backends()
    env_cfg = load_env_cfg(TASK_ID, play=True)
    env_cfg.scene.num_envs = len(seeds)
    env_cfg.seed = min(seeds)
    env_cfg.auto_reset = False
    env_cfg.episode_length_s = duration_s
    env_cfg.terminations.clear()
    reset_params = (
        {"standing_prob": 1.0, "midroll_prob": 0.0}
        if reset_mode == "standing"
        else {
            "standing_prob": 0.0,
            "midroll_prob": 1.0,
            "midroll_pitch_min": math.radians(precontact_pitch_range_deg[0]),
            "midroll_pitch_max": math.radians(precontact_pitch_range_deg[1]),
            "midroll_omega_range": precontact_omega_range,
            "midroll_z_min": precontact_z,
            "midroll_z_max": precontact_z,
            "tuck_factor_range": (1.0, 1.0),
        }
    )
    env_cfg.events["set_grounded_backroll_state"].params.update(
        **reset_params,
        repeat_mode=True,
        yaw_range=(0.0, 0.0),
        joint_noise_std=(
            precontact_joint_noise if reset_mode == "precontact" else 0.02
        ),
    )

    base_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    # Apply the deterministic standing/precontact reset. Environment
    # construction alone does not run the configured reset event, which would
    # leave repeat_mode disabled and make every repeated-cycle audit invalid.
    base_env.reset()
    if reset_mode == "precontact":
        # A phase-start audit must prove physical contact acquisition rather
        # than inheriting the reset helper's synthetic prerequisite latches.
        microduck_mdp._grounded_backroll_state(base_env)
        base_env._backroll_trunk_latch.zero_()
        base_env._backroll_head_latch.zero_()
        base_env._roulade_head_latch.zero_()
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
    policy = runner.get_inference_policy(device=device)
    robot = base_env.scene["robot"]
    start_xy = robot.data.root_link_pos_w[:, :2].clone()
    start_heading = _heading_from_quat(robot.data.root_link_quat_w).clone()
    previous_vz = robot.data.root_link_lin_vel_w[:, 2].clone()
    peak_impact = torch.zeros(len(seeds), device=base_env.device)
    peak_backward_rate = torch.zeros_like(peak_impact)
    nan_seen = torch.zeros(len(seeds), dtype=torch.bool, device=base_env.device)
    out_of_bounds = torch.zeros_like(nan_seen)

    try:
        for _ in range(math.ceil(duration_s / base_env.step_dt)):
            with torch.inference_mode():
                observations = env.get_observations()
                actions = policy(observations)
                env.step(actions)
            current_vz = robot.data.root_link_lin_vel_w[:, 2]
            peak_impact = torch.maximum(
                peak_impact,
                torch.abs(current_vz - previous_vz) / base_env.step_dt,
            )
            previous_vz = current_vz.clone()
            peak_backward_rate = torch.maximum(
                peak_backward_rate,
                torch.clamp(-robot.data.root_link_ang_vel_b[:, 1], min=0.0),
            )
            finite = (
                torch.isfinite(robot.data.root_link_pos_w).all(dim=-1)
                & torch.isfinite(robot.data.root_link_quat_w).all(dim=-1)
                & torch.isfinite(robot.data.root_link_lin_vel_w).all(dim=-1)
                & torch.isfinite(robot.data.root_link_ang_vel_b).all(dim=-1)
                & torch.isfinite(actions).all(dim=-1)
            )
            nan_seen |= ~finite
            local_position = robot.data.root_link_pos_w - base_env.scene.terrain.env_origins
            out_of_bounds |= (
                (local_position[:, :2].abs() > 5.0).any(dim=-1)
                | (local_position[:, 2] < -0.25)
                | (local_position[:, 2] > 1.0)
            )

        heading = _heading_from_quat(robot.data.root_link_quat_w)
        heading_dot = (heading * start_heading).sum(dim=-1).clamp(-1.0, 1.0)
        yaw_deviation = torch.rad2deg(torch.acos(heading_dot))
        displacement = robot.data.root_link_pos_w[:, :2] - start_xy
        lateral_axis = torch.stack((-start_heading[:, 1], start_heading[:, 0]), dim=-1)
        lateral = torch.abs((displacement * lateral_axis).sum(dim=-1))
        rows: list[dict[str, object]] = []
        for index, seed in enumerate(seeds):
            cycles = int(base_env._backroll_cycle_count[index].item())
            invalid = bool(base_env._backroll_invalid[index].item())
            valid = bool(
                cycles >= required_cycles
                and not invalid
                and not nan_seen[index].item()
                and not out_of_bounds[index].item()
            )
            rows.append(
                {
                    "seed": seed,
                    "variant_batch_seed": min(seeds),
                    "variant_batch_size": len(seeds),
                    "variant_batch_index": index,
                    "valid_grounded_backroll_cycles": cycles,
                    "required_cycles": required_cycles,
                    "consecutive_backroll_success": valid,
                    "current_cycle_rotation_deg": float(
                        torch.rad2deg(base_env._roulade_accum[index]).item()
                    ),
                    "current_cycle_frontier_deg": float(
                        torch.rad2deg(base_env._roulade_max[index]).item()
                    ),
                    "trunk_contact_latched": bool(
                        base_env._backroll_trunk_latch[index].item()
                    ),
                    "head_top_contact_latched": bool(
                        base_env._backroll_head_latch[index].item()
                    ),
                    "landing_hold_s": float(
                        base_env._backroll_landing_hold_steps[index].item()
                        * base_env.step_dt
                    ),
                    "max_lateral_axis_z": float(
                        base_env._backroll_episode_max_lateral_axis_z[index].item()
                    ),
                    "max_offaxis_rotation_deg": float(
                        torch.rad2deg(
                            base_env._backroll_episode_max_offaxis_rotation[index]
                        ).item()
                    ),
                    "peak_backward_angular_rate_rad_s": float(
                        peak_backward_rate[index].item()
                    ),
                    "lateral_displacement_m": float(lateral[index].item()),
                    "yaw_deviation_deg": float(yaw_deviation[index].item()),
                    "peak_vertical_impact_mps2": float(peak_impact[index].item()),
                    "invalid_physical_solution": invalid,
                    "nan": bool(nan_seen[index].item()),
                    "out_of_bounds": bool(out_of_bounds[index].item()),
                }
            )
    finally:
        env.close()

    successes = sum(bool(row["consecutive_backroll_success"]) for row in rows)
    report: dict[str, object] = {
        "schema_version": 1,
        "task": TASK_ID,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_iteration": _checkpoint_iteration(checkpoint),
        "duration_s": duration_s,
        "required_cycles": required_cycles,
        "reset_mode": reset_mode,
        "precontact_z": precontact_z if reset_mode == "precontact" else None,
        "precontact_pitch_range_deg": (
            list(precontact_pitch_range_deg) if reset_mode == "precontact" else None
        ),
        "precontact_omega_range": (
            list(precontact_omega_range) if reset_mode == "precontact" else None
        ),
        "precontact_joint_noise": (
            precontact_joint_noise if reset_mode == "precontact" else None
        ),
        "num_standing_trials": len(rows) if reset_mode == "standing" else 0,
        "num_precontact_trials": len(rows) if reset_mode == "precontact" else 0,
        "consecutive_success_count": successes,
        "consecutive_success_rate": successes / len(rows),
        "proof_available": successes > 0,
        "acceptance_12_of_16": len(rows) == 16 and successes >= 12,
        "zero_nan_oob": not any(
            bool(row["nan"] or row["out_of_bounds"]) for row in rows
        ),
        "max_valid_cycle_count": max(
            int(row["valid_grounded_backroll_cycles"]) for row in rows
        ),
        "trials": rows,
    }
    return report


def main() -> int:
    args = _parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")
    report = evaluate_checkpoint(
        checkpoint,
        seeds=list(args.seeds),
        device=args.device,
        duration_s=args.duration,
        required_cycles=args.required_cycles,
        reset_mode=args.reset_mode,
        precontact_z=args.precontact_z,
        precontact_pitch_range_deg=(
            args.precontact_pitch_min_deg,
            args.precontact_pitch_max_deg,
        ),
        precontact_omega_range=(
            args.precontact_omega_min,
            args.precontact_omega_max,
        ),
        precontact_joint_noise=args.precontact_joint_noise,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output is not None:
        _write_json_atomic(args.output.expanduser().resolve(), report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

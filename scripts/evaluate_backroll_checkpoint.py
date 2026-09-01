#!/usr/bin/env python3
"""Audit one grounded-backroll checkpoint on deterministic standing starts."""

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

TASK_ID = "Mjlab-Backroll-Flat-MicroDuck"
DEFAULT_SEEDS = tuple(range(16))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="Deterministic standing-start seed identifiers.",
    )
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
    """Return the planar heading of MicroDuck's body-forward +y axis."""
    quat = torch.nan_to_num(quat, nan=0.0)
    w, x, y, z = quat.unbind(dim=-1)
    body_y_x = 2.0 * (x * y - w * z)
    body_y_y = 1.0 - 2.0 * (x.square() + z.square())
    heading = torch.stack((body_y_y, -body_y_x), dim=-1)
    return heading / heading.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)


def _checkpoint_iteration(path: Path) -> int | None:
    try:
        return int(path.stem.rsplit("_", 1)[-1])
    except ValueError:
        return None


def _load_policy(
    base_env: ManagerBasedRlEnv,
    checkpoint: Path,
    *,
    device: str,
):
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


def _case_rows(
    *,
    seeds: list[int],
    batch_seed: int,
    start_xy: torch.Tensor,
    start_heading: torch.Tensor,
    robot,
    base_env: ManagerBasedRlEnv,
    peak_impact: torch.Tensor,
    nan_seen: torch.Tensor,
    out_of_bounds: torch.Tensor,
) -> list[dict[str, object]]:
    heading = _heading_from_quat(robot.data.root_link_quat_w)
    heading_dot = (heading * start_heading).sum(dim=-1).clamp(-1.0, 1.0)
    yaw_deviation = torch.rad2deg(torch.acos(heading_dot))
    displacement = robot.data.root_link_pos_w[:, :2] - start_xy
    lateral_axis = torch.stack((-start_heading[:, 1], start_heading[:, 0]), dim=-1)
    lateral = torch.abs((displacement * lateral_axis).sum(dim=-1))

    rows: list[dict[str, object]] = []
    for index, seed in enumerate(seeds):
        rows.append(
            {
                "seed": seed,
                "variant_batch_seed": batch_seed,
                "variant_batch_size": len(seeds),
                "variant_batch_index": index,
                "signed_backward_rotation_deg": float(
                    torch.rad2deg(base_env._roulade_max[index]).item()
                ),
                "trunk_contact_latched": bool(
                    base_env._backroll_trunk_latch[index].item()
                ),
                "head_top_contact_latched": bool(
                    base_env._backroll_head_latch[index].item()
                ),
                "max_grounded_support_gap_s": float(
                    base_env._backroll_max_air_steps[index].item() * base_env.step_dt
                ),
                "landing_hold_s": float(
                    base_env._backroll_landing_hold_steps[index].item()
                    * base_env.step_dt
                ),
                "lateral_displacement_m": float(lateral[index].item()),
                "yaw_deviation_deg": float(yaw_deviation[index].item()),
                "peak_vertical_impact_mps2": float(peak_impact[index].item()),
                "nan": bool(nan_seen[index].item()),
                "out_of_bounds": bool(out_of_bounds[index].item()),
                "invalid_physical_solution": bool(
                    base_env._backroll_invalid[index].item()
                ),
                "grounded_backroll_success": bool(
                    base_env._backroll_success[index].item()
                    and not base_env._backroll_invalid[index].item()
                    and not nan_seen[index].item()
                    and not out_of_bounds[index].item()
                ),
            }
        )
    return rows


def evaluate_checkpoint(
    checkpoint: Path,
    *,
    seeds: list[int],
    device: str,
    duration_s: float,
) -> dict[str, object]:
    if not seeds:
        raise ValueError("at least one deterministic standing-start seed is required")
    if duration_s <= 0.0:
        raise ValueError("duration must be positive")

    configure_torch_backends()
    env_cfg = load_env_cfg(TASK_ID, play=True)
    batch_seed = min(seeds)
    env_cfg.scene.num_envs = len(seeds)
    env_cfg.seed = batch_seed
    env_cfg.auto_reset = False
    env_cfg.episode_length_s = duration_s
    # The audit owns the fixed horizon and physical validity checks. Disabling
    # manager terminations lets all 16 trials finish without auto-resetting an
    # early invalid case and losing its one-shot state latches.
    env_cfg.terminations.clear()
    reset_cfg = env_cfg.events["set_grounded_backroll_state"]
    reset_cfg.params.update(
        standing_prob=1.0,
        midroll_prob=0.0,
        yaw_range=(0.0, 0.0),
        joint_noise_std=0.02,
    )

    base_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    # Apply the configured deterministic reset before measuring the rollout;
    # construction alone retains the compiled default state.
    base_env.reset()
    env, policy = _load_policy(base_env, checkpoint, device=device)
    robot = base_env.scene["robot"]
    start_xy = robot.data.root_link_pos_w[:, :2].clone()
    start_heading = _heading_from_quat(robot.data.root_link_quat_w).clone()
    previous_vz = robot.data.root_link_lin_vel_w[:, 2].clone()
    peak_impact = torch.zeros(len(seeds), device=base_env.device)
    nan_seen = torch.zeros(len(seeds), dtype=torch.bool, device=base_env.device)
    out_of_bounds = torch.zeros_like(nan_seen)
    steps = math.ceil(duration_s / base_env.step_dt)

    try:
        for _ in range(steps):
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
            finite = (
                torch.isfinite(robot.data.root_link_pos_w).all(dim=-1)
                & torch.isfinite(robot.data.root_link_quat_w).all(dim=-1)
                & torch.isfinite(robot.data.root_link_lin_vel_w).all(dim=-1)
                & torch.isfinite(robot.data.root_link_ang_vel_b).all(dim=-1)
                & torch.isfinite(actions).all(dim=-1)
            )
            nan_seen |= ~finite
            local_position = (
                robot.data.root_link_pos_w - base_env.scene.terrain.env_origins
            )
            out_of_bounds |= (
                (local_position[:, :2].abs() > 5.0).any(dim=-1)
                | (local_position[:, 2] < -0.25)
                | (local_position[:, 2] > 1.0)
            )
    finally:
        rows = _case_rows(
            seeds=seeds,
            batch_seed=batch_seed,
            start_xy=start_xy,
            start_heading=start_heading,
            robot=robot,
            base_env=base_env,
            peak_impact=peak_impact,
            nan_seen=nan_seen,
            out_of_bounds=out_of_bounds,
        )
        env.close()

    successes = sum(bool(row["grounded_backroll_success"]) for row in rows)
    eligible = [row for row in rows if row["grounded_backroll_success"]]
    report: dict[str, object] = {
        "schema_version": 2,
        "task": TASK_ID,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_iteration": _checkpoint_iteration(checkpoint),
        "duration_s": duration_s,
        "num_standing_trials": len(rows),
        "variant_batch_seed": batch_seed,
        "variant_batch_size": len(seeds),
        "grounded_backroll_success_count": successes,
        "grounded_backroll_success_rate": successes / len(rows),
        "acceptance_12_of_16": len(rows) == 16 and successes >= 12,
        "zero_nan_oob": not any(
            bool(row["nan"] or row["out_of_bounds"]) for row in rows
        ),
        "mean_success_lateral_displacement_m": (
            sum(float(row["lateral_displacement_m"]) for row in eligible)
            / len(eligible)
            if eligible
            else None
        ),
        "mean_success_yaw_deviation_deg": (
            sum(float(row["yaw_deviation_deg"]) for row in eligible) / len(eligible)
            if eligible
            else None
        ),
        "max_success_peak_vertical_impact_mps2": (
            max(float(row["peak_vertical_impact_mps2"]) for row in eligible)
            if eligible
            else None
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
    )
    serialized = json.dumps(report, indent=2, sort_keys=True)
    print(serialized)
    if args.output is not None:
        _write_json_atomic(args.output.expanduser().resolve(), report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

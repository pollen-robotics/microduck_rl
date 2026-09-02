#!/usr/bin/env python3
"""Collect physically realized head-pivot states from a successful backroll policy."""

from __future__ import annotations

import argparse
import hashlib
import math
from dataclasses import asdict
from pathlib import Path

import mjlab.tasks  # noqa: F401  # Populate the task registry.
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

from mjlab_microduck.tasks import mdp as microduck_mdp

TASK_ID = "Mjlab-Backroll-Flat-MicroDuck"
DEFAULT_SEEDS = (0, 3, 5, 9, 10, 15)
PHASE_CENTERS_DEG = (100.0, 140.0, 180.0, 220.0, 260.0, 290.0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--phase-centers-deg",
        type=float,
        nargs="+",
        default=list(PHASE_CENTERS_DEG),
        help="Phase centers to capture; defaults to the full successful-roll bank.",
    )
    parser.add_argument("--max-yaw-deg", type=float, default=20.0)
    parser.add_argument("--phase-tolerance-deg", type=float, default=16.0)
    parser.add_argument(
        "--allow-incomplete-strict",
        action="store_true",
        help=(
            "Keep physically observed partial states that satisfy the live sagittal, "
            "off-axis, and support-history gates without requiring final success."
        ),
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _heading_from_quat(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = torch.nan_to_num(quat, nan=0.0).unbind(dim=-1)
    heading = torch.stack(
        (1.0 - 2.0 * (x.square() + z.square()), -(2.0 * (x * y - w * z))),
        dim=-1,
    )
    return heading / heading.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)


def _load_policy(base_env: ManagerBasedRlEnv, checkpoint: Path, device: str):
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


def collect_reference_states(args: argparse.Namespace) -> dict[str, object]:
    if not args.seeds:
        raise ValueError("at least one seed is required")
    if not args.phase_centers_deg:
        raise ValueError("at least one phase center is required")
    configure_torch_backends()
    env_cfg = load_env_cfg(TASK_ID, play=True)
    env_cfg.scene.num_envs = len(args.seeds)
    env_cfg.seed = min(args.seeds)
    env_cfg.auto_reset = False
    env_cfg.episode_length_s = args.duration
    env_cfg.terminations.clear()
    env_cfg.events["set_grounded_backroll_state"].params.update(
        standing_prob=1.0,
        midroll_prob=0.0,
        yaw_range=(0.0, 0.0),
        joint_noise_std=0.02,
    )

    base_env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
    base_env.reset()
    env, policy = _load_policy(base_env, args.checkpoint, args.device)
    robot = base_env.scene["robot"]
    start_heading = _heading_from_quat(robot.data.root_link_quat_w).clone()
    phase_centers_deg = tuple(float(value) for value in args.phase_centers_deg)
    centers = torch.tensor(phase_centers_deg)
    candidates: dict[tuple[int, int], tuple[float, dict[str, object]]] = {}
    steps = math.ceil(args.duration / base_env.step_dt)

    try:
        for step in range(steps):
            with torch.inference_mode():
                observations = env.get_observations()
                actions = policy(observations)
                env.step(actions)
            phase_deg = torch.rad2deg(base_env._roulade_max).detach().cpu()
            lateral_axis = torch.abs(
                microduck_mdp._lateral_axis_z(robot.data.root_link_quat_w)
            ).detach().cpu()
            finite = (
                torch.isfinite(base_env.sim.data.qpos).all(dim=1)
                & torch.isfinite(base_env.sim.data.qvel).all(dim=1)
            ).detach().cpu()
            for env_index in range(len(args.seeds)):
                if (
                    not finite[env_index]
                    or base_env._backroll_invalid[env_index].item()
                    or lateral_axis[env_index].item() > math.sin(math.radians(35.0))
                ):
                    continue
                phase = float(phase_deg[env_index].item())
                distances = torch.abs(centers - phase)
                center_index = int(torch.argmin(distances).item())
                distance = float(distances[center_index].item())
                if distance > args.phase_tolerance_deg:
                    continue
                key = (env_index, center_index)
                if key in candidates and candidates[key][0] <= distance:
                    continue
                local_qpos = base_env.sim.data.qpos[env_index].detach().cpu().clone()
                local_qpos[:2] -= base_env.scene.terrain.env_origins[
                    env_index, :2
                ].detach().cpu()
                candidates[key] = (
                    distance,
                    {
                        "seed": args.seeds[env_index],
                        "phase_center_deg": phase_centers_deg[center_index],
                        "phase_deg": phase,
                        "step": step,
                        "qpos": local_qpos,
                        "qvel": base_env.sim.data.qvel[env_index]
                        .detach()
                        .cpu()
                        .clone(),
                        "accum": base_env._roulade_accum[env_index]
                        .detach()
                        .cpu()
                        .clone(),
                        "frontier": base_env._roulade_max[env_index]
                        .detach()
                        .cpu()
                        .clone(),
                        "paid": base_env._roulade_paid[env_index]
                        .detach()
                        .cpu()
                        .clone(),
                        "trunk_latch": base_env._backroll_trunk_latch[env_index]
                        .detach()
                        .cpu()
                        .clone(),
                        "head_latch": base_env._backroll_head_latch[env_index]
                        .detach()
                        .cpu()
                        .clone(),
                        "cycle_max_lateral_axis_z": base_env._backroll_cycle_max_lateral_axis_z[
                            env_index
                        ]
                        .detach()
                        .cpu()
                        .clone(),
                        "cycle_offaxis_rotation": base_env._backroll_cycle_offaxis_rotation[
                            env_index
                        ]
                        .detach()
                        .cpu()
                        .clone(),
                        "max_air_steps": base_env._backroll_max_air_steps[env_index]
                        .detach()
                        .cpu()
                        .clone(),
                        "lateral_axis_z": lateral_axis[env_index].clone(),
                    },
                )

        final_heading = _heading_from_quat(robot.data.root_link_quat_w)
        yaw_deg = torch.rad2deg(
            torch.acos((final_heading * start_heading).sum(dim=-1).clamp(-1.0, 1.0))
        ).detach().cpu()
        if args.allow_incomplete_strict:
            max_lateral_axis_z = math.sin(math.radians(20.0))
            max_offaxis_rotation = math.radians(90.0)
            max_air_steps = math.ceil(
                microduck_mdp._BACKROLL_MAX_AIR_SECONDS / base_env.step_dt
            )
            rows = [
                value[1]
                for _, value in sorted(candidates.items())
                if float(value[1]["cycle_max_lateral_axis_z"])
                <= max_lateral_axis_z
                and float(value[1]["cycle_offaxis_rotation"])
                <= max_offaxis_rotation
                and int(value[1]["max_air_steps"]) <= max_air_steps
                and (
                    not bool(value[1]["head_latch"])
                    or bool(value[1]["trunk_latch"])
                )
            ]
        else:
            eligible_envs = {
                index
                for index in range(len(args.seeds))
                if base_env._backroll_success[index].item()
                and not base_env._backroll_invalid[index].item()
                and yaw_deg[index].item() <= args.max_yaw_deg
            }
            rows = [
                value[1]
                for (env_index, _), value in sorted(candidates.items())
                if env_index in eligible_envs
            ]
    finally:
        env.close()

    if not rows:
        raise RuntimeError("no strict successful reference states were collected")
    payload: dict[str, object] = {
        "schema_version": 2,
        "task": TASK_ID,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "seeds": list(args.seeds),
        "max_yaw_deg": args.max_yaw_deg,
        "phase_centers_deg": list(phase_centers_deg),
        "allow_incomplete_strict": bool(args.allow_incomplete_strict),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    return payload


def main() -> None:
    args = _parse_args()
    payload = collect_reference_states(args)
    print(
        f"saved {len(payload['rows'])} strict reference states to {args.output} "
        f"from seeds={payload['seeds']}"
    )


if __name__ == "__main__":
    main()

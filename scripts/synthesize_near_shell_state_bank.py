#!/usr/bin/env python3
"""Synthesize physics-settled, still-failing near-shell reverse states."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import mjlab.tasks  # noqa: F401  # Populate the task registry.
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends

from collect_stair_walker_states import (
    _apply_contact_transfer_overrides,
    _contact_transfer_sensor_state,
    _exact_state_row_digest,
    _finite_state_rows,
    _raw_hard_shell_candidate,
    _raw_shell_frontier_value,
    _select_nested_rows,
    _sha256,
    _write_bank_atomic,
)
from mjlab_microduck.tasks.stair_walk_state_bank import (
    BANK_SCHEMA_VERSION,
    STANDARD_NUM_STEPS,
    STANDARD_RISER_HEIGHT_M,
    STANDARD_TREAD_DEPTH_M,
    capture_walk_state_rows,
    concatenate_walk_state_rows,
    load_walk_state_bank,
    walk_state_count,
)

SOURCE_TASK = "Mjlab-Stairs-Stage2-Reverse-RSI-Specialist-MicroDuck"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".tmp/codex/full170-a34-near-shell-negative-state-bank.pt"),
    )
    parser.add_argument("--target-states", type=int, default=128)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--max-batches", type=int, default=8)
    parser.add_argument("--settle-steps", type=int, default=25)
    parser.add_argument("--validation-steps", type=int, default=25)
    parser.add_argument("--placement-x-range", type=float, nargs=2, default=(0.71, 0.74))
    parser.add_argument("--placement-z-range", type=float, nargs=2, default=(0.15, 0.17))
    parser.add_argument("--frontier-range", type=float, nargs=2, default=(0.50, 0.90))
    parser.add_argument("--max-validation-drift", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if (
        args.target_states < 1
        or args.num_envs < 1
        or args.max_batches < 1
        or args.settle_steps < 1
        or args.validation_steps < 1
        or args.placement_x_range[0] >= args.placement_x_range[1]
        or args.placement_z_range[0] >= args.placement_z_range[1]
        or not 0.0 < args.frontier_range[0] < args.frontier_range[1] < 1.0
        or args.max_validation_drift <= 0.0
    ):
        raise SystemExit("Invalid synthesis count, range, horizon, or drift bound")

    configure_torch_backends()
    torch.manual_seed(args.seed)
    cfg = load_env_cfg(SOURCE_TASK, play=True)
    agent_cfg = load_rl_cfg(SOURCE_TASK)
    cfg.scene.num_envs = args.num_envs
    stage2_event = cfg.events["stage2_reverse_state_bank"]
    source_bank_path = Path(stage2_event.params["bank_path"]).resolve()
    source_bank = load_walk_state_bank(source_bank_path)
    source_force = source_bank["states"].get(
        "captured_strongest_tread_normal_force_n"
    )
    if source_force is None:
        raise SystemExit("Source Stage 2 bank lacks captured tread-force provenance")
    max_source_force = float(source_force.max().item())
    stage2_event.params["local_x_range"] = tuple(args.placement_x_range)
    _apply_contact_transfer_overrides(
        cfg,
        num_envs=args.num_envs,
        terrain_level=2,
        forced_family=2,
    )
    cfg.seed = args.seed

    base = ManagerBasedRlEnv(cfg=cfg, device=args.device, render_mode=None)
    env = RslRlVecEnvWrapper(base, clip_actions=agent_cfg.clip_actions)
    robot = base.scene["robot"]
    root_q = robot.indexing.free_joint_q_adr.to(torch.long)
    root_v = robot.indexing.free_joint_v_adr.to(torch.long)
    joint_v = robot.indexing.joint_v_adr.to(torch.long)
    origins = base.scene.terrain.env_origins
    chunks: list[dict[str, object]] = []
    seen: set[str] = set()
    duplicate_states_rejected = 0
    nonfinite_states_rejected = 0
    total_accepted_before_dedup = 0

    try:
        for batch in range(1, args.max_batches + 1):
            env.reset()
            source_rows = base._stair_walker_bank_row.clone()
            base.sim.data.qpos[:, root_q[2]] = origins[:, 2] + torch.empty(
                args.num_envs, device=base.device
            ).uniform_(*args.placement_z_range)
            base.sim.data.qvel[:, root_v] = 0.0
            base.sim.data.qvel[:, joint_v] = 0.0
            base.sim.forward()
            base.scene.update(dt=0.0)
            placement_root = base.sim.data.qpos[:, root_q[:3]].clone() - origins
            held_actions = (
                base.action_manager.get_term("joint_pos")._raw_actions.clone()
            )

            optimization_done = torch.zeros(
                args.num_envs, dtype=torch.bool, device=base.device
            )
            for _ in range(args.settle_steps):
                _, _, dones, _ = env.step(held_actions)
                optimization_done |= dones.bool()

            settled_root = base.sim.data.qpos[:, root_q[:3]].clone() - origins
            settled_frontier = _raw_shell_frontier_value(
                base, required_latch_name=None
            )
            settled_raw = _raw_hard_shell_candidate(base)
            settled_face, _, settled_force, settled_finite, _ = (
                _contact_transfer_sensor_state(base)
            )
            frontier_low, frontier_high = args.frontier_range
            provisional = (
                (source_rows >= 0)
                & (base.scene.terrain.terrain_levels == 2)
                & (settled_frontier >= frontier_low)
                & (settled_frontier <= frontier_high)
                & ~settled_raw
                & ~settled_face
                & settled_finite
                & ~optimization_done
                & (settled_force <= max_source_force)
            )

            max_drift = torch.zeros(args.num_envs, device=base.device)
            max_force = settled_force.clone()
            ever_raw = settled_raw.clone()
            ever_face = settled_face.clone()
            ever_done = torch.zeros_like(provisional)
            ever_nan = torch.zeros_like(provisional)
            validation_finite = settled_finite.clone()
            for _ in range(args.validation_steps):
                _, _, dones, _ = env.step(held_actions)
                local = base.sim.data.qpos[:, root_q[:3]] - origins
                max_drift = torch.maximum(
                    max_drift,
                    torch.linalg.vector_norm(local - settled_root, dim=-1),
                )
                face, _, force, contact_finite, _ = (
                    _contact_transfer_sensor_state(base)
                )
                ever_face |= face
                max_force = torch.maximum(max_force, force)
                ever_raw |= _raw_hard_shell_candidate(base)
                ever_done |= dones.bool()
                ever_nan |= base.termination_manager.get_term("nan_state").bool()
                validation_finite &= contact_finite

            final_frontier = _raw_shell_frontier_value(
                base, required_latch_name=None
            )
            stage1 = base._stair_contact_transfer_stage1_policy_achieved.bool()
            stage15 = base._stair_contact_transfer_stage15_policy_achieved.bool()
            stage2 = base._stair_contact_transfer_stage2_policy_achieved.bool()
            shell = base._stair_true_shell_clearance_policy_achieved.bool()
            secured = base._stair_first_tread_secured_latched.bool()
            accepted = (
                provisional
                & validation_finite
                & (final_frontier >= frontier_low)
                & (final_frontier <= frontier_high)
                & ~ever_raw
                & ~ever_face
                & ~ever_done
                & ~ever_nan
                & (max_drift <= args.max_validation_drift)
                & (max_force <= max_source_force)
                & stage1
                & stage15
                & stage2
                & ~shell
                & ~secured
            )
            ids = accepted.nonzero(as_tuple=False).squeeze(-1)
            total_accepted_before_dedup += len(ids)
            remaining = args.target_states - sum(
                int(chunk["root_qpos_local"].shape[0]) for chunk in chunks
            )
            if len(ids) > 0 and remaining > 0:
                candidate = capture_walk_state_rows(base, ids)
                finite_rows = _finite_state_rows(candidate)
                nonfinite_states_rejected += int((~finite_rows).sum().item())
                finite_indices = finite_rows.nonzero(as_tuple=False).squeeze(-1)
                candidate = _select_nested_rows(candidate, finite_indices)
                ids = ids[finite_indices.to(base.device)]
                keep_rows: list[int] = []
                for row in range(len(ids)):
                    digest = _exact_state_row_digest(candidate, row)
                    if digest in seen:
                        duplicate_states_rejected += 1
                        continue
                    seen.add(digest)
                    keep_rows.append(row)
                    if len(keep_rows) >= remaining:
                        break
                if keep_rows:
                    keep = torch.tensor(keep_rows, dtype=torch.long)
                    ids = ids[keep.to(base.device)]
                    chunk = _select_nested_rows(candidate, keep)
                    chunk["source_episode_step"] = (
                        base._stair_walker_bank_source_step[ids].cpu().clone()
                    )
                    chunk["source_state_bank_row"] = source_rows[ids].cpu().clone()
                    chunk["source_reset_family"] = torch.full(
                        (len(ids),), 2, dtype=torch.long
                    )
                    chunk["source_terrain_level"] = torch.full(
                        (len(ids),), 2, dtype=torch.long
                    )
                    chunk["synthesis_placement_root_local"] = (
                        placement_root[ids].cpu().clone()
                    )
                    chunk["captured_shell_frontier_value"] = (
                        final_frontier[ids].cpu().clone()
                    )
                    chunk["captured_raw_shell_candidate"] = (
                        ever_raw[ids].cpu().clone()
                    )
                    chunk["captured_union_face_contact"] = (
                        ever_face[ids].cpu().clone()
                    )
                    chunk["captured_max_tread_normal_force_n"] = (
                        max_force[ids].cpu().clone()
                    )
                    chunk["captured_max_validation_root_drift_m"] = (
                        max_drift[ids].cpu().clone()
                    )
                    chunk["captured_stage1_policy_achieved"] = (
                        stage1[ids].cpu().clone()
                    )
                    chunk["captured_stage15_policy_achieved"] = (
                        stage15[ids].cpu().clone()
                    )
                    chunk["captured_stage2_policy_achieved"] = (
                        stage2[ids].cpu().clone()
                    )
                    chunk["captured_true_shell_clearance"] = (
                        shell[ids].cpu().clone()
                    )
                    chunk["captured_secured_tread"] = secured[ids].cpu().clone()
                    chunks.append(chunk)

            collected = sum(
                int(chunk["root_qpos_local"].shape[0]) for chunk in chunks
            )
            print(
                f"[near-shell-synthesis] batch={batch} accepted={len(ids)} "
                f"states={collected}/{args.target_states}"
            )
            if collected >= args.target_states:
                break
    finally:
        env.close()

    if not chunks:
        raise SystemExit("No physics-settled near-shell states passed validation")
    states = concatenate_walk_state_rows(chunks)
    count = walk_state_count(states)
    if count < args.target_states:
        raise SystemExit(
            f"Collected only {count}/{args.target_states} states in "
            f"{args.max_batches} batches"
        )
    frontier = states["captured_shell_frontier_value"]
    bank: dict[str, object] = {
        "schema_version": BANK_SCHEMA_VERSION,
        "metadata": {
            "created_at": datetime.now(UTC).isoformat(),
            "task": SOURCE_TASK,
            "method": "contact_constrained_physics_settled_near_shell_negative",
            "source_bank": str(source_bank_path),
            "source_bank_sha256": _sha256(source_bank_path),
            "source_walker_checkpoint": source_bank["metadata"].get(
                "walker_checkpoint"
            ),
            "source_walker_checkpoint_sha256": source_bank["metadata"].get(
                "walker_checkpoint_sha256"
            ),
            "num_states": count,
            "num_envs": args.num_envs,
            "terrain_level": 2,
            "riser_height_m": STANDARD_RISER_HEIGHT_M,
            "tread_depth_m": STANDARD_TREAD_DEPTH_M,
            "num_steps": STANDARD_NUM_STEPS,
            "placement_x_range_m": list(args.placement_x_range),
            "placement_z_range_m": list(args.placement_z_range),
            "frontier_range": list(args.frontier_range),
            "settle_steps": args.settle_steps,
            "validation_steps": args.validation_steps,
            "max_validation_drift_m": args.max_validation_drift,
            "max_source_tread_normal_force_n": max_source_force,
            "frontier_p50": float(frontier.quantile(0.5).item()),
            "frontier_max": float(frontier.max().item()),
            "total_accepted_before_dedup": total_accepted_before_dedup,
            "duplicate_states_rejected": duplicate_states_rejected,
            "nonfinite_states_rejected": nonfinite_states_rejected,
            "joint_names": list(robot.joint_names),
            "physics_dt": base.physics_dt,
            "step_dt": base.step_dt,
            "decimation": base.cfg.decimation,
            "mjlab_version": version("mjlab"),
        },
        "states": states,
    }
    output = args.output.resolve()
    _write_bank_atomic(output, bank)
    print(f"[near-shell-synthesis] wrote {count} states to {output}")
    print(
        f"[near-shell-synthesis] frontier_p50={bank['metadata']['frontier_p50']:.4f} "
        f"frontier_max={bank['metadata']['frontier_max']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

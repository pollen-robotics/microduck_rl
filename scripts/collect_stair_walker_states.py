#!/usr/bin/env python3
"""Collect exact manufacturer motion states for the fixed 170 mm home stair."""

from __future__ import annotations

import argparse
import hashlib
import os
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from tempfile import NamedTemporaryFile

import mjlab.tasks  # noqa: F401  # Populate the task registry.
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

from mjlab_microduck.policies import load_frozen_actor
from mjlab_microduck.tasks.mdp import classify_standard_stair_contacts
from mjlab_microduck.tasks.stair_walk_state_bank import (
    BANK_SCHEMA_VERSION,
    STANDARD_NUM_STEPS,
    STANDARD_RISER_HEIGHT_M,
    STANDARD_TREAD_DEPTH_M,
    capture_walk_state_rows,
    concatenate_walk_state_rows,
    walk_state_count,
)

DEFAULT_TASK_ID = "Mjlab-Stairs-Route-MicroDuck"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--walker-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--source-task",
        default=DEFAULT_TASK_ID,
        help="Registered task used to generate the source motion.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".tmp/codex/full170-walker-state-bank.pt"),
    )
    parser.add_argument("--target-states", type=int, default=256)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--min-local-x", type=float, default=0.56)
    parser.add_argument("--max-local-x", type=float, default=0.64)
    parser.add_argument(
        "--min-local-z",
        type=float,
        help="Optional minimum root height for frontier-state capture.",
    )
    parser.add_argument(
        "--max-abs-local-y",
        type=float,
        default=0.25,
        help="Maximum absolute lateral root offset for capture.",
    )
    parser.add_argument("--max-steps", type=int, default=8_000)
    parser.add_argument(
        "--capture-every-n-steps",
        type=int,
        default=0,
        help=(
            "Capture every N eligible control steps. Zero preserves the original "
            "one-state-per-episode walking-bank behavior."
        ),
    )
    parser.add_argument(
        "--zero-all-command-observations",
        action="store_true",
        help="Zero actor observation slots 52:61 for manufacturer episodic policies.",
    )
    parser.add_argument(
        "--preserve-command-observations",
        action="store_true",
        help=(
            "Preserve all actor command observations. Required when collecting "
            "from a stair policy that consumes route or time cues."
        ),
    )
    parser.add_argument(
        "--standing-only-reset",
        action="store_true",
        help="Force standing starts when the source task has a set_roulade_state event.",
    )
    parser.add_argument(
        "--capture-first-tread-contact",
        action="store_true",
        help=(
            "Capture only states with a position-and-normal classified contact "
            "on the horizontal first tread."
        ),
    )
    parser.add_argument(
        "--tread-contact-sensor",
        default="robot_ground_contact",
        help=(
            "Position/normal contact sensor used by --capture-first-tread-contact. "
            "Use feet_stair_contact to build a foot-anchored vault bank."
        ),
    )
    parser.add_argument(
        "--min-tread-normal-force",
        type=float,
        default=0.0,
        help="Minimum normal load in newtons for a captured tread-contact slot.",
    )
    parser.add_argument(
        "--capture-riser-face-without-tread",
        action="store_true",
        help=(
            "Capture only states with a classified first-riser face contact and "
            "no first-tread contact in the same frame across dedicated sensors."
        ),
    )
    parser.add_argument(
        "--contact-sensors",
        nargs="+",
        default=(
            "head_ground_contact",
            "trunk_ground_contact",
            "legs_ground_contact",
            "feet_stair_contact",
        ),
        help="Sensors unioned by --capture-riser-face-without-tread.",
    )
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def _write_bank_atomic(path: Path, bank: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(suffix=".tmp", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(bank, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    args = _parse_args()
    checkpoint = args.walker_checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"Walker checkpoint not found: {checkpoint}")
    if (
        args.target_states < 1
        or args.num_envs < 1
        or args.max_steps < 1
        or args.capture_every_n_steps < 0
        or args.min_tread_normal_force < 0.0
        or args.min_local_x >= args.max_local_x
        or args.max_abs_local_y <= 0.0
        or (
            args.zero_all_command_observations
            and args.preserve_command_observations
        )
    ):
        raise SystemExit("State count, environment count, steps, and capture band are invalid")

    configure_torch_backends()
    env_cfg = load_env_cfg(args.source_task, play=True)
    agent_cfg = load_rl_cfg(args.source_task)
    if args.standing_only_reset:
        event = env_cfg.events.get("set_roulade_state")
        if event is None:
            raise SystemExit(
                "--standing-only-reset requires a source task with set_roulade_state"
            )
        event.params["standing_prob"] = 1.0
        event.params["midroll_prob"] = 0.0
        base_pose = env_cfg.events["reset_base"].params["pose_range"]
        base_pose["x"] = (0.0, 0.0)
        base_pose["y"] = (0.0, 0.0)
        base_pose["yaw"] = (0.0, 0.0)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = 0
    base_env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(args.source_task) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=args.device)
    walker = load_frozen_actor(runner, checkpoint, device=args.device)

    chunks: list[dict[str, object]] = []
    captured_this_episode = torch.zeros(
        args.num_envs, dtype=torch.bool, device=args.device
    )
    observations = env.get_observations()
    steps_run = 0
    best_local_x = float("-inf")
    best_corridor_x = float("-inf")
    try:
        for steps_run in range(1, args.max_steps + 1):
            with torch.inference_mode():
                walker_observations = observations.clone()
                actor_observations = walker_observations["actor"].clone()
                command_start = None
                if not args.preserve_command_observations:
                    command_start = (
                        52 if args.zero_all_command_observations else 55
                    )
                    actor_observations[:, command_start:61] = 0.0
                walker_observations["actor"] = actor_observations
                actions = walker(walker_observations)
                observations, _, dones, _ = env.step(actions)

            robot = base_env.scene["robot"]
            origins = base_env.scene.terrain.env_origins
            local = robot.data.root_link_pos_w - origins
            best_local_x = max(best_local_x, float(local[:, 0].max().item()))
            in_corridor = torch.abs(local[:, 1]) <= args.max_abs_local_y
            if torch.any(in_corridor):
                best_corridor_x = max(
                    best_corridor_x,
                    float(local[in_corridor, 0].max().item()),
                )
            unique_episode_gate = (
                ~captured_this_episode
                if args.capture_every_n_steps == 0
                else torch.ones_like(captured_this_episode)
            )
            cadence_gate = (
                torch.ones_like(captured_this_episode)
                if args.capture_every_n_steps == 0
                else (base_env.episode_length_buf % args.capture_every_n_steps == 0)
            )
            contact_gate = torch.ones_like(captured_this_episode)
            contact_sensor_data = None
            tread_contact = None
            if args.capture_first_tread_contact:
                if args.tread_contact_sensor not in base_env.scene.sensors:
                    raise RuntimeError(
                        f"Unknown tread contact sensor: {args.tread_contact_sensor}"
                    )
                sensor = base_env.scene.sensors[args.tread_contact_sensor].data
                if sensor.found is None or sensor.pos is None or sensor.normal is None:
                    raise RuntimeError(
                        f"{args.tread_contact_sensor} must expose found, pos, and normal"
                    )
                _, tread_contact = classify_standard_stair_contacts(
                    sensor.found,
                    sensor.pos,
                    sensor.normal,
                    origins,
                )
                if args.min_tread_normal_force > 0.0:
                    if sensor.force is None:
                        raise RuntimeError(
                            f"{args.tread_contact_sensor} must expose force for "
                            "--min-tread-normal-force"
                        )
                    normal_force = torch.abs(
                        torch.sum(sensor.force * sensor.normal, dim=-1)
                    )
                    tread_contact &= normal_force >= args.min_tread_normal_force
                contact_gate = tread_contact.any(dim=-1)
                contact_sensor_data = sensor
            if args.capture_riser_face_without_tread:
                face_any = torch.zeros_like(captured_this_episode)
                tread_any = torch.zeros_like(captured_this_episode)
                for sensor_name in args.contact_sensors:
                    if sensor_name not in base_env.scene.sensors:
                        raise RuntimeError(f"Unknown contact sensor: {sensor_name}")
                    sensor = base_env.scene.sensors[sensor_name].data
                    if (
                        sensor.found is None
                        or sensor.pos is None
                        or sensor.normal is None
                    ):
                        raise RuntimeError(
                            f"{sensor_name} must expose found, pos, and normal"
                        )
                    face, tread = classify_standard_stair_contacts(
                        sensor.found,
                        sensor.pos,
                        sensor.normal,
                        origins,
                    )
                    face_any |= face.any(dim=-1)
                    tread_any |= tread.any(dim=-1)
                contact_gate &= face_any & ~tread_any
            eligible = (
                unique_episode_gate
                & cadence_gate
                & contact_gate
                & (dones == 0)
                & (base_env.episode_length_buf > 2)
                & (local[:, 0] >= args.min_local_x)
                & (local[:, 0] <= args.max_local_x)
                & (torch.abs(local[:, 1]) <= args.max_abs_local_y)
            )
            if args.min_local_z is not None:
                eligible &= local[:, 2] >= args.min_local_z
            ids = eligible.nonzero(as_tuple=False).squeeze(-1)
            remaining = args.target_states - sum(
                int(chunk["root_qpos_local"].shape[0]) for chunk in chunks
            )
            if len(ids) > remaining:
                ids = ids[:remaining]
            if len(ids) > 0:
                chunk = capture_walk_state_rows(base_env, ids)
                chunk["source_episode_step"] = (
                    base_env.episode_length_buf[ids].detach().cpu().clone()
                )
                if contact_sensor_data is not None and tread_contact is not None:
                    contact_pos_local = contact_sensor_data.pos[ids] - origins[
                        ids, None, :
                    ]
                    chunk["captured_tread_contact"] = {
                        "found": contact_sensor_data.found[ids]
                        .detach()
                        .cpu()
                        .clone(),
                        "mask": tread_contact[ids].detach().cpu().clone(),
                        "pos_local": contact_pos_local.detach().cpu().clone(),
                        "normal": contact_sensor_data.normal[ids]
                        .detach()
                        .cpu()
                        .clone(),
                    }
                    if contact_sensor_data.force is not None:
                        chunk["captured_tread_contact"]["force"] = (
                            contact_sensor_data.force[ids].detach().cpu().clone()
                        )
                chunks.append(chunk)
                captured_this_episode[ids] = True
            captured_this_episode[dones.to(torch.bool)] = False

            collected = sum(
                int(chunk["root_qpos_local"].shape[0]) for chunk in chunks
            )
            if steps_run % 500 == 0 or collected >= args.target_states:
                print(
                    f"[walker-bank] step={steps_run} states={collected}/{args.target_states} "
                    f"best_x={best_local_x:.3f} corridor_x={best_corridor_x:.3f}"
                )
            if collected >= args.target_states:
                break
    finally:
        env.close()

    if not chunks:
        raise SystemExit(
            "The immutable walker never entered the requested capture band; "
            f"best_x={best_local_x:.3f}, corridor_x={best_corridor_x:.3f}"
        )
    states = concatenate_walk_state_rows(chunks)
    count = walk_state_count(states)
    if count < args.target_states:
        raise SystemExit(
            f"Collected only {count}/{args.target_states} states in {steps_run} steps"
        )

    bank: dict[str, object] = {
        "schema_version": BANK_SCHEMA_VERSION,
        "metadata": {
            "created_at": datetime.now(UTC).isoformat(),
            "task": args.source_task,
            "walker_checkpoint": str(checkpoint),
            "walker_checkpoint_sha256": _sha256(checkpoint),
            "capture_local_x_m": [args.min_local_x, args.max_local_x],
            "capture_min_local_z_m": args.min_local_z,
            "capture_max_abs_local_y_m": args.max_abs_local_y,
            "riser_height_m": STANDARD_RISER_HEIGHT_M,
            "tread_depth_m": STANDARD_TREAD_DEPTH_M,
            "num_steps": STANDARD_NUM_STEPS,
            "num_states": count,
            "num_envs": args.num_envs,
            "steps_run": steps_run,
            "joint_names": list(base_env.scene["robot"].joint_names),
            "physics_dt": base_env.physics_dt,
            "step_dt": base_env.step_dt,
            "decimation": base_env.cfg.decimation,
            "mjlab_version": version("mjlab"),
            "actor_command_slice_zeroed": (
                None
                if args.preserve_command_observations
                else [52 if args.zero_all_command_observations else 55, 61]
            ),
            "capture_every_n_steps": args.capture_every_n_steps,
            "standing_only_reset": args.standing_only_reset,
            "capture_first_tread_contact": args.capture_first_tread_contact,
            "capture_riser_face_without_tread": (
                args.capture_riser_face_without_tread
            ),
            "contact_sensors": list(args.contact_sensors),
            "tread_contact_sensor": args.tread_contact_sensor,
            "min_tread_normal_force_n": args.min_tread_normal_force,
            "canonical_source_xy_yaw": args.standing_only_reset,
        },
        "states": states,
    }
    _write_bank_atomic(output, bank)
    print(f"[walker-bank] wrote {count} states to {output}")
    print(f"[walker-bank] checkpoint_sha256={bank['metadata']['walker_checkpoint_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

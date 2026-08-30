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
from typing import Any

import mjlab.tasks  # noqa: F401  # Populate the task registry.
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

from mjlab_microduck.policies import load_frozen_actor
from mjlab_microduck.tasks.mdp import (
    _virtual_lip_stair_contact_masks,
    classify_standard_stair_contacts,
)
from mjlab_microduck.tasks.microduck_standard_stairs_env_cfg import (
    STANDARD_RISER_HEIGHT,
    STANDARD_STAIR_START_DISTANCE,
    STANDARD_STAIR_WIDTH,
    STANDARD_TREAD_DEPTH,
    VIRTUAL_LIP_CURRICULUM_LEVELS,
    VIRTUAL_LIP_MAX_FACE_OFFSET,
)
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
A31_TASK_ID = "Mjlab-Stairs-Contact-Stage-RSI-Specialist-MicroDuck"
A32_TASK_ID = "Mjlab-Stairs-Stage15-Reverse-RSI-Specialist-MicroDuck"
CONTACT_TRANSFER_SENSOR_NAMES = (
    "head_ground_contact",
    "trunk_ground_contact",
    "legs_ground_contact",
    "feet_stair_contact",
)
A12_SIDE_BYPASS_MIN_X_M = 0.660
A12_SIDE_BYPASS_MIN_ABS_Y_M = STANDARD_STAIR_WIDTH * 0.40


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
        "--capture-contact-transfer-stage15",
        action="store_true",
        help=(
            "Capture the first policy-created A31 Stage 1.5 edge in each "
            "episode with strict transfer/contact validation."
        ),
    )
    parser.add_argument(
        "--capture-contact-transfer-stage2",
        action="store_true",
        help=(
            "Capture the first policy-created A32 Stage 2 edge in each "
            "episode with strict transfer/contact validation."
        ),
    )
    parser.add_argument(
        "--terrain-level",
        type=int,
        choices=(0, 1, 2),
        help=(
            "Contact-transfer virtual-lip terrain level. Stage 1.5 and "
            "Stage 2 collection require this to be explicitly set to 0."
        ),
    )
    parser.add_argument(
        "--reset-family",
        choices=("face-no-tread", "stage15-reverse"),
        help=(
            "Replay reset family. Stage 1.5 collection requires face-no-tread; "
            "Stage 2 collection requires stage15-reverse."
        ),
    )
    parser.add_argument(
        "--contact-transfer-min-normal-force",
        type=float,
        default=2.0,
        help="Minimum strongest dedicated-sensor tread normal force in newtons.",
    )
    parser.add_argument(
        "--contact-transfer-max-abs-local-y",
        type=float,
        default=0.20,
        help="Maximum absolute root y for a captured contact-transfer edge.",
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


def _validate_contact_transfer_args(args: argparse.Namespace) -> None:
    stage15_mode = args.capture_contact_transfer_stage15
    stage2_mode = args.capture_contact_transfer_stage2
    if stage15_mode and stage2_mode:
        raise SystemExit("Choose exactly one contact-transfer capture stage")
    if not stage15_mode and not stage2_mode:
        if args.terrain_level is not None or args.reset_family is not None:
            raise SystemExit(
                "--terrain-level and --reset-family require "
                "--capture-contact-transfer-stage15 or "
                "--capture-contact-transfer-stage2"
            )
        return
    expected_task = A31_TASK_ID if stage15_mode else A32_TASK_ID
    expected_family = "face-no-tread" if stage15_mode else "stage15-reverse"
    capture_flag = (
        "--capture-contact-transfer-stage15"
        if stage15_mode
        else "--capture-contact-transfer-stage2"
    )
    stage_label = "Stage 1.5" if stage15_mode else "Stage 2"
    if args.source_task != expected_task:
        raise SystemExit(
            f"{capture_flag} requires --source-task {expected_task}"
        )
    if args.terrain_level != 0:
        raise SystemExit(
            f"{capture_flag} requires explicit --terrain-level 0"
        )
    if args.reset_family != expected_family:
        raise SystemExit(
            f"{capture_flag} requires explicit --reset-family {expected_family}"
        )
    incompatible = (
        args.capture_first_tread_contact
        or args.capture_riser_face_without_tread
        or args.standing_only_reset
        or args.zero_all_command_observations
        or args.capture_every_n_steps != 0
    )
    if incompatible:
        raise SystemExit(
            f"{stage_label} capture cannot be combined with legacy contact/cadence, "
            "standing-reset, or command-zeroing modes"
        )
    if tuple(args.contact_sensors) != CONTACT_TRANSFER_SENSOR_NAMES:
        raise SystemExit(
            f"{stage_label} capture requires the complete dedicated "
            "contact-sensor set"
        )
    if (
        args.contact_transfer_min_normal_force <= 0.0
        or args.contact_transfer_max_abs_local_y <= 0.0
    ):
        raise SystemExit(f"{stage_label} force and lateral bounds must be positive")


def _apply_contact_transfer_overrides(
    env_cfg: Any, *, num_envs: int, terrain_level: int, forced_family: int
) -> None:
    hard_viewer = env_cfg.events.get("a30_hard_viewer")
    if hard_viewer is None:
        raise RuntimeError("A31 play cfg is missing the a30_hard_viewer event")
    hard_viewer.params["terrain_levels"] = (terrain_level,) * num_envs
    hard_viewer.params["terrain_types"] = (0,) * num_envs
    family_event = env_cfg.events.get("state_bank_family")
    if family_event is None:
        raise RuntimeError("A31 play cfg is missing the state_bank_family event")
    family_event.params["forced_family"] = forced_family


def _select_nested_rows(value: Any, rows: torch.Tensor) -> Any:
    if isinstance(value, dict):
        return {key: _select_nested_rows(item, rows) for key, item in value.items()}
    if not isinstance(value, torch.Tensor):
        raise TypeError("Walker-state leaves must be tensors")
    return value[rows].clone()


def _finite_state_rows(states: dict[str, Any]) -> torch.Tensor:
    count = int(states["root_qpos_local"].shape[0])
    finite = torch.ones(count, dtype=torch.bool)

    def visit(value: Any) -> None:
        nonlocal finite
        if isinstance(value, dict):
            for item in value.values():
                visit(item)
            return
        if not isinstance(value, torch.Tensor):
            raise TypeError("Walker-state leaves must be tensors")
        if value.shape[0] != count:
            raise ValueError("Walker-state leaves have inconsistent row counts")
        if value.is_floating_point() or value.is_complex():
            finite &= torch.isfinite(value).reshape(count, -1).all(dim=-1)

    visit(states)
    return finite


def _exact_state_row_digest(states: dict[str, Any], row: int) -> str:
    digest = hashlib.sha256()

    def visit(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key in sorted(value):
                visit(value[key], f"{path}/{key}")
            return
        if not isinstance(value, torch.Tensor):
            raise TypeError("Walker-state leaves must be tensors")
        item = value[row].detach().cpu().contiguous()
        digest.update(path.encode("utf-8"))
        digest.update(str(item.dtype).encode("ascii"))
        digest.update(repr(tuple(item.shape)).encode("ascii"))
        digest.update(item.reshape(-1).view(torch.uint8).numpy().tobytes())

    visit(states, "states")
    return digest.hexdigest()


def _stage15_capture_eligibility(
    *,
    stage15_edge: torch.Tensor,
    stage1: torch.Tensor,
    stage2: torch.Tensor,
    shell_clearance: torch.Tensor,
    face_contact: torch.Tensor,
    tread_contact: torch.Tensor,
    strongest_tread_force: torch.Tensor,
    episode_steps: torch.Tensor,
    root_y: torch.Tensor,
    dones: torch.Tensor,
    nan_termination: torch.Tensor,
    side_bypass_seen: torch.Tensor,
    finite: torch.Tensor,
    min_normal_force: float,
    max_abs_local_y: float,
) -> torch.Tensor:
    return (
        stage15_edge
        & stage1
        & ~stage2
        & ~shell_clearance
        & ~face_contact
        & tread_contact
        & (strongest_tread_force >= min_normal_force)
        & (episode_steps >= 3)
        & (torch.abs(root_y) <= max_abs_local_y)
        & ~dones.bool()
        & ~nan_termination
        & ~side_bypass_seen
        & finite
    )


def _stage2_capture_eligibility(
    *,
    stage2_edge: torch.Tensor,
    stage1: torch.Tensor,
    stage15: torch.Tensor,
    stage2: torch.Tensor,
    shell_clearance: torch.Tensor,
    face_contact: torch.Tensor,
    tread_contact: torch.Tensor,
    strongest_tread_force: torch.Tensor,
    episode_steps: torch.Tensor,
    root_y: torch.Tensor,
    dones: torch.Tensor,
    nan_termination: torch.Tensor,
    side_bypass_seen: torch.Tensor,
    finite: torch.Tensor,
    min_normal_force: float,
    max_abs_local_y: float,
) -> torch.Tensor:
    return (
        stage2_edge
        & stage1
        & stage15
        & stage2
        & ~shell_clearance
        & ~face_contact
        & tread_contact
        & (strongest_tread_force >= min_normal_force)
        & (episode_steps >= 3)
        & (torch.abs(root_y) <= max_abs_local_y)
        & ~dones.bool()
        & ~nan_termination
        & ~side_bypass_seen
        & finite
    )


def _contact_transfer_sensor_state(
    env: ManagerBasedRlEnv,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, dict[str, torch.Tensor]],
]:
    face_any = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    tread_any = torch.zeros_like(face_any)
    strongest_tread_force = torch.zeros(
        env.num_envs, dtype=torch.float32, device=env.device
    )
    finite = torch.ones_like(face_any)
    evidence: dict[str, dict[str, torch.Tensor]] = {}
    for sensor_name in CONTACT_TRANSFER_SENSOR_NAMES:
        if sensor_name not in env.scene.sensors:
            raise RuntimeError(f"Missing required contact sensor: {sensor_name}")
        data = env.scene.sensors[sensor_name].data
        if (
            data.found is None
            or data.pos is None
            or data.normal is None
            or data.force is None
        ):
            raise RuntimeError(
                f"{sensor_name} must expose found, pos, normal, and force"
            )
        face, tread, pos_local = _virtual_lip_stair_contact_masks(
            env,
            sensor_name,
            nominal_stair_face_x=STANDARD_STAIR_START_DISTANCE,
            max_face_offset=VIRTUAL_LIP_MAX_FACE_OFFSET,
            num_terrain_levels=VIRTUAL_LIP_CURRICULUM_LEVELS,
            riser_height=STANDARD_RISER_HEIGHT,
            tread_depth=STANDARD_TREAD_DEPTH,
            corridor_half_width=STANDARD_STAIR_WIDTH * 0.40,
        )
        normal_force = torch.abs(torch.sum(data.force * data.normal, dim=-1))
        tread_normal_force = torch.where(
            tread, normal_force, torch.zeros_like(normal_force)
        )
        face_any |= face.any(dim=-1)
        tread_any |= tread.any(dim=-1)
        strongest_tread_force = torch.maximum(
            strongest_tread_force,
            tread_normal_force.max(dim=-1).values,
        )
        finite &= (
            torch.isfinite(data.pos).reshape(env.num_envs, -1).all(dim=-1)
            & torch.isfinite(data.normal).reshape(env.num_envs, -1).all(dim=-1)
            & torch.isfinite(data.force).reshape(env.num_envs, -1).all(dim=-1)
        )
        evidence[sensor_name] = {
            "found": data.found,
            "face_mask": face,
            "tread_mask": tread,
            "pos_local": pos_local,
            "normal": data.normal,
            "force": data.force,
            "normal_force_n": normal_force,
        }
    return face_any, tread_any, strongest_tread_force, finite, evidence


def _required_env_tensor(env: ManagerBasedRlEnv, name: str) -> torch.Tensor:
    value = getattr(env, name, None)
    if value is None or not isinstance(value, torch.Tensor):
        raise RuntimeError(f"Contact-transfer tracking tensor is unavailable: {name}")
    return value


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
    _validate_contact_transfer_args(args)
    capture_contact_transfer = (
        args.capture_contact_transfer_stage15
        or args.capture_contact_transfer_stage2
    )
    preserve_command_observations = (
        args.preserve_command_observations
        or capture_contact_transfer
    )

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
    if capture_contact_transfer:
        assert args.terrain_level is not None
        _apply_contact_transfer_overrides(
            env_cfg,
            num_envs=args.num_envs,
            terrain_level=args.terrain_level,
            forced_family=(0 if args.capture_contact_transfer_stage15 else 1),
        )
    env_cfg.seed = 0
    base_env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
    if capture_contact_transfer:
        actual_levels = base_env.scene.terrain.terrain_levels
        if not torch.all(actual_levels == args.terrain_level):
            base_env.close()
            raise RuntimeError(
                "Contact-transfer terrain-level override did not reach every "
                "collector environment"
            )
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
    if capture_contact_transfer:
        env.reset()
        actual_families = _required_env_tensor(
            base_env, "_stair_state_bank_family"
        )
        source_rows = _required_env_tensor(base_env, "_stair_walker_bank_row")
        expected_family = 0 if args.capture_contact_transfer_stage15 else 1
        if not torch.all(actual_families == expected_family):
            env.close()
            raise RuntimeError(
                f"Contact-transfer reset family {expected_family} did not reach "
                "every environment"
            )
        if not torch.all(source_rows >= 0):
            env.close()
            raise RuntimeError(
                f"Contact-transfer reset family {expected_family} did not produce "
                "a replay-bank row for "
                "every environment"
            )
    runner_cls = load_runner_cls(args.source_task) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=args.device)
    walker = load_frozen_actor(runner, checkpoint, device=args.device)

    chunks: list[dict[str, object]] = []
    captured_this_episode = torch.zeros(
        args.num_envs, dtype=torch.bool, device=args.device
    )
    previous_stage15_policy = torch.zeros_like(captured_this_episode)
    previous_stage2_policy = torch.zeros_like(captured_this_episode)
    side_bypass_seen = torch.zeros_like(captured_this_episode)
    seen_state_digests: set[str] = set()
    duplicate_states_rejected = 0
    nonfinite_states_rejected = 0
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
                if not preserve_command_observations:
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
            remaining = args.target_states - sum(
                int(chunk["root_qpos_local"].shape[0]) for chunk in chunks
            )
            if capture_contact_transfer:
                (
                    face_any,
                    tread_any,
                    strongest_tread_force,
                    contact_finite,
                    contact_evidence,
                ) = _contact_transfer_sensor_state(base_env)
                stage1 = _required_env_tensor(
                    base_env, "_stair_contact_transfer_stage1_policy_achieved"
                ).bool()
                stage15 = _required_env_tensor(
                    base_env, "_stair_contact_transfer_stage15_policy_achieved"
                ).bool()
                stage2 = _required_env_tensor(
                    base_env, "_stair_contact_transfer_stage2_policy_achieved"
                ).bool()
                shell_clearance = _required_env_tensor(
                    base_env, "_stair_true_shell_clearance_policy_achieved"
                ).bool()
                stage15_edge = stage15 & ~previous_stage15_policy
                stage2_edge = stage2 & ~previous_stage2_policy
                source_reset_families = _required_env_tensor(
                    base_env, "_stair_state_bank_family"
                )
                source_bank_rows = _required_env_tensor(
                    base_env, "_stair_walker_bank_row"
                )
                nan_termination = base_env.termination_manager.get_term(
                    "nan_state"
                ).bool()
                side_bypass_seen |= (
                    (base_env.episode_length_buf >= 3)
                    & (local[:, 0] >= A12_SIDE_BYPASS_MIN_X_M)
                    & (torch.abs(local[:, 1]) > A12_SIDE_BYPASS_MIN_ABS_Y_M)
                    & ~shell_clearance
                )
                finite = (
                    contact_finite
                    & torch.isfinite(base_env.sim.data.qpos)
                    .reshape(args.num_envs, -1)
                    .all(dim=-1)
                    & torch.isfinite(base_env.sim.data.qvel)
                    .reshape(args.num_envs, -1)
                    .all(dim=-1)
                    & torch.isfinite(actions).reshape(args.num_envs, -1).all(dim=-1)
                    & torch.isfinite(local).all(dim=-1)
                    & torch.isfinite(strongest_tread_force)
                )
                if args.capture_contact_transfer_stage15:
                    eligible = _stage15_capture_eligibility(
                        stage15_edge=stage15_edge,
                        stage1=stage1,
                        stage2=stage2,
                        shell_clearance=shell_clearance,
                        face_contact=face_any,
                        tread_contact=tread_any,
                        strongest_tread_force=strongest_tread_force,
                        episode_steps=base_env.episode_length_buf,
                        root_y=local[:, 1],
                        dones=dones,
                        nan_termination=nan_termination,
                        side_bypass_seen=side_bypass_seen,
                        finite=finite,
                        min_normal_force=args.contact_transfer_min_normal_force,
                        max_abs_local_y=args.contact_transfer_max_abs_local_y,
                    )
                else:
                    eligible = _stage2_capture_eligibility(
                        stage2_edge=stage2_edge,
                        stage1=stage1,
                        stage15=stage15,
                        stage2=stage2,
                        shell_clearance=shell_clearance,
                        face_contact=face_any,
                        tread_contact=tread_any,
                        strongest_tread_force=strongest_tread_force,
                        episode_steps=base_env.episode_length_buf,
                        root_y=local[:, 1],
                        dones=dones,
                        nan_termination=nan_termination,
                        side_bypass_seen=side_bypass_seen,
                        finite=finite,
                        min_normal_force=args.contact_transfer_min_normal_force,
                        max_abs_local_y=args.contact_transfer_max_abs_local_y,
                    )
                    eligible &= (
                        (source_reset_families == 1)
                        & (source_bank_rows >= 0)
                    )
                eligible = (~captured_this_episode) & eligible
                candidate_ids = eligible.nonzero(as_tuple=False).squeeze(-1)
                captured_this_episode[candidate_ids] = True
                if len(candidate_ids) > 0 and remaining > 0:
                    candidate = capture_walk_state_rows(base_env, candidate_ids)
                    finite_rows = _finite_state_rows(candidate)
                    nonfinite_states_rejected += int((~finite_rows).sum().item())
                    finite_indices = finite_rows.nonzero(as_tuple=False).squeeze(-1)
                    candidate = _select_nested_rows(candidate, finite_indices)
                    candidate_ids = candidate_ids[finite_indices.to(args.device)]
                    keep_rows: list[int] = []
                    for row in range(len(candidate_ids)):
                        state_digest = _exact_state_row_digest(candidate, row)
                        if state_digest in seen_state_digests:
                            duplicate_states_rejected += 1
                            continue
                        seen_state_digests.add(state_digest)
                        keep_rows.append(row)
                        if len(keep_rows) >= remaining:
                            break
                    if keep_rows:
                        keep = torch.tensor(keep_rows, dtype=torch.long)
                        ids = candidate_ids[keep.to(args.device)]
                        chunk = _select_nested_rows(candidate, keep)
                        chunk["source_episode_step"] = (
                            base_env.episode_length_buf[ids].detach().cpu().clone()
                        )
                        chunk["source_state_bank_row"] = _required_env_tensor(
                            base_env, "_stair_walker_bank_row"
                        )[ids].detach().cpu().clone()
                        chunk["source_state_bank_source_step"] = _required_env_tensor(
                            base_env, "_stair_walker_bank_source_step"
                        )[ids].detach().cpu().clone()
                        chunk["source_reset_family"] = _required_env_tensor(
                            base_env, "_stair_state_bank_family"
                        )[ids].detach().cpu().clone()
                        chunk["source_terrain_level"] = (
                            base_env.scene.terrain.terrain_levels[ids]
                            .detach()
                            .cpu()
                            .clone()
                        )
                        chunk["captured_stage1_policy_achieved"] = (
                            stage1[ids].detach().cpu().clone()
                        )
                        chunk["captured_stage15_policy_achieved"] = (
                            stage15[ids].detach().cpu().clone()
                        )
                        chunk["captured_stage2_policy_achieved"] = (
                            stage2[ids].detach().cpu().clone()
                        )
                        chunk["captured_true_shell_clearance"] = (
                            shell_clearance[ids].detach().cpu().clone()
                        )
                        chunk["captured_union_face_contact"] = (
                            face_any[ids].detach().cpu().clone()
                        )
                        chunk["captured_union_tread_contact"] = (
                            tread_any[ids].detach().cpu().clone()
                        )
                        chunk["captured_strongest_tread_normal_force_n"] = (
                            strongest_tread_force[ids].detach().cpu().clone()
                        )
                        chunk["captured_contact_transfer_sensors"] = {
                            name: {
                                field: value[ids].detach().cpu().clone()
                                for field, value in fields.items()
                            }
                            for name, fields in contact_evidence.items()
                        }
                        chunks.append(chunk)
                previous_stage15_policy.copy_(stage15)
                previous_stage2_policy.copy_(stage2)
            else:
                unique_episode_gate = (
                    ~captured_this_episode
                    if args.capture_every_n_steps == 0
                    else torch.ones_like(captured_this_episode)
                )
                cadence_gate = (
                    torch.ones_like(captured_this_episode)
                    if args.capture_every_n_steps == 0
                    else (
                        base_env.episode_length_buf % args.capture_every_n_steps == 0
                    )
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
                    if (
                        sensor.found is None
                        or sensor.pos is None
                        or sensor.normal is None
                    ):
                        raise RuntimeError(
                            f"{args.tread_contact_sensor} must expose found, pos, "
                            "and normal"
                        )
                    _, tread_contact = classify_standard_stair_contacts(
                        sensor.found, sensor.pos, sensor.normal, origins
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
                            sensor.found, sensor.pos, sensor.normal, origins
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
                                contact_sensor_data.force[ids]
                                .detach()
                                .cpu()
                                .clone()
                            )
                    chunks.append(chunk)
                    captured_this_episode[ids] = True
            done_mask = dones.to(torch.bool)
            captured_this_episode[done_mask] = False
            previous_stage15_policy[done_mask] = False
            previous_stage2_policy[done_mask] = False
            side_bypass_seen[done_mask] = False

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
            "capture_local_x_m": (
                None
                if capture_contact_transfer
                else [args.min_local_x, args.max_local_x]
            ),
            "capture_min_local_z_m": args.min_local_z,
            "capture_max_abs_local_y_m": (
                args.contact_transfer_max_abs_local_y
                if capture_contact_transfer
                else args.max_abs_local_y
            ),
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
                if preserve_command_observations
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
            "capture_contact_transfer_stage15": (
                args.capture_contact_transfer_stage15
            ),
            "capture_contact_transfer_stage2": (
                args.capture_contact_transfer_stage2
            ),
            "contact_transfer_terrain_level": (
                args.terrain_level
                if capture_contact_transfer
                else None
            ),
            "contact_transfer_reset_family": (
                args.reset_family
                if capture_contact_transfer
                else None
            ),
            "contact_transfer_min_normal_force_n": (
                args.contact_transfer_min_normal_force
            ),
            "contact_transfer_max_abs_local_y_m": (
                args.contact_transfer_max_abs_local_y
            ),
            "contact_transfer_first_policy_edge_only": (
                capture_contact_transfer
            ),
            "contact_transfer_exact_state_deduplication": (
                capture_contact_transfer
            ),
            "duplicate_states_rejected": duplicate_states_rejected,
            "nonfinite_states_rejected": nonfinite_states_rejected,
            "command_observations_preserved": preserve_command_observations,
        },
        "states": states,
    }
    _write_bank_atomic(output, bank)
    print(f"[walker-bank] wrote {count} states to {output}")
    print(f"[walker-bank] checkpoint_sha256={bank['metadata']['walker_checkpoint_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

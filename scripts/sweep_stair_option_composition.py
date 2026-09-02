#!/usr/bin/env python3
"""Run a bounded vectorized LAUNCH-to-MANTLE option-composition sweep."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from dataclasses import asdict
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import mjlab.tasks  # noqa: F401  # Populate the task registry.
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

from mjlab_microduck.policies import (
    OFFICIAL_WALKER_SHA256,
    StairOptionPolicy,
    load_frozen_actor,
)
from mjlab_microduck.tasks.mdp import stair_true_shell_clearance_frontier_value
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

A35_TASK_ID = "Mjlab-Stairs-Stratified-Shell-Reverse-RSI-Specialist-MicroDuck"
REPO_ROOT = Path(__file__).resolve().parents[1]
GUARDED_HANDOFF_BANK = REPO_ROOT / ".tmp/codex/official-guarded-handoff-bank.pt"
LAUNCH_CHECKPOINT = REPO_ROOT / ".tmp/codex/official-roulade-stair-bootstrap.pt"
MANTLE_CHECKPOINT = REPO_ROOT / (
    "logs/rsl_rl/microduck_stair_stratified_shell_reverse_rsi_specialist/"
    "2026-08-30_07-54-17_a35_round1_sol_stratified_hard_1024_gate10/"
    "model_9.pt"
)
DEFAULT_OUTPUT = REPO_ROOT / ".tmp/codex/stair-option-composition-sweep.json"
DEFAULT_FRONTIER_BANK = REPO_ROOT / (
    ".tmp/codex/stair-option-composition-frontier-bank.pt"
)

HARD_TERRAIN_LEVEL = 2
SOURCE_RESET_FAMILY = 0
STAIR_FACE_X_M = 0.660
SHELL_CORRIDOR_HALF_WIDTH_M = 0.20
SIDE_BYPASS_HALF_WIDTH_M = 0.36
HEAD_FRONTIER_MIN_ROOT_X_M = 0.52
MIN_VALID_ROOT_Z_M = 0.07
MAX_ENVS = 512
MAX_CONTROL_STEPS = 400
MAX_FRONTIER_STATES = 256
OFFICIAL_ROULADE_BOOTSTRAP_SHA256 = (
    "181d2afe0cd8ede1de320e2b3d95a0bf9e6d142e744e6df1f2be8f8890b1bd92"
)
A35_MODEL_9_SHA256 = (
    "0b39240f2b81781bc55662a3469400386690ea7586ded6100eb3a62309c1be30"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--launch-checkpoint", type=Path, default=LAUNCH_CHECKPOINT
    )
    parser.add_argument(
        "--launch-sha256", default=OFFICIAL_ROULADE_BOOTSTRAP_SHA256
    )
    parser.add_argument(
        "--launch-description", default="official roulade stair bootstrap"
    )
    parser.add_argument("--trials-per-candidate", type=int, default=4)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--blend-steps", type=int, default=4)
    parser.add_argument(
        "--launch-min-steps", nargs="+", type=int, default=(4, 8, 12)
    )
    parser.add_argument(
        "--launch-max-steps", nargs="+", type=int, default=(20, 30, 40)
    )
    parser.add_argument(
        "--mantle-root-x", nargs="+", type=float, default=(0.58, 0.60)
    )
    parser.add_argument(
        "--mantle-root-z", nargs="+", type=float, default=(0.135, 0.145)
    )
    parser.add_argument("--max-frontier-states", type=int, default=128)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--frontier-bank-output", type=Path, default=DEFAULT_FRONTIER_BANK
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
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _write_bank_atomic(path: Path, bank: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(suffix=".tmp", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(bank, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _select_nested_rows(value: Any, rows: torch.Tensor) -> Any:
    if isinstance(value, dict):
        return {key: _select_nested_rows(item, rows) for key, item in value.items()}
    if not isinstance(value, torch.Tensor):
        raise TypeError("Frontier-state leaves must be tensors")
    return value[rows].clone()


def _exact_state_row_digest(states: dict[str, Any], row: int) -> str:
    digest = hashlib.sha256()

    def visit(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key in sorted(value):
                visit(value[key], f"{path}/{key}")
            return
        if not isinstance(value, torch.Tensor):
            raise TypeError("Frontier-state leaves must be tensors")
        item = value[row].detach().cpu().contiguous()
        digest.update(path.encode("utf-8"))
        digest.update(str(item.dtype).encode("ascii"))
        digest.update(repr(tuple(item.shape)).encode("ascii"))
        digest.update(item.reshape(-1).view(torch.uint8).numpy().tobytes())

    visit(states, "states")
    return digest.hexdigest()


def _record_first_step(
    first_steps: torch.Tensor, event: torch.Tensor, step: int
) -> None:
    fresh = event & (first_steps < 0)
    first_steps[fresh] = step


def _step_list(values: torch.Tensor) -> list[int | None]:
    return [None if value < 0 else int(value) for value in values.tolist()]


def _earliest(values: torch.Tensor) -> int | None:
    reached = values[values >= 0]
    return None if len(reached) == 0 else int(reached.min().item())


def _validate_artifacts(
    args: argparse.Namespace,
) -> tuple[Path, Path, Path, dict[str, Any]]:
    bank_path = GUARDED_HANDOFF_BANK.resolve()
    launch_path = args.launch_checkpoint.expanduser().resolve()
    mantle_path = MANTLE_CHECKPOINT.resolve()
    for label, path in (
        ("guarded handoff bank", bank_path),
        (f"{args.launch_description} LAUNCH checkpoint", launch_path),
        ("A35 model_9 MANTLE checkpoint", mantle_path),
    ):
        if not path.is_file():
            raise SystemExit(f"Missing {label}: {path}")
    bank = load_walk_state_bank(bank_path)
    metadata = bank["metadata"]
    if metadata.get("source") != "guarded_official_walker_handoff":
        raise SystemExit("Guarded handoff bank has an unexpected source marker")
    if metadata.get("walker_checkpoint_sha256") != OFFICIAL_WALKER_SHA256:
        raise SystemExit(
            "Guarded handoff bank was not generated by the pinned official walker"
        )
    artifact_hashes = {
        f"{args.launch_description} LAUNCH": (launch_path, args.launch_sha256),
        "A35 model_9 MANTLE": (mantle_path, A35_MODEL_9_SHA256),
    }
    for label, (path, expected) in artifact_hashes.items():
        actual = _sha256(path)
        if actual != expected:
            raise SystemExit(
                f"{label} hash mismatch: expected {expected}, got {actual}"
            )
    return bank_path, launch_path, mantle_path, bank


def _candidate_grid(args: argparse.Namespace) -> list[dict[str, int | float]]:
    if args.trials_per_candidate < 1:
        raise SystemExit("--trials-per-candidate must be positive")
    if not 1 <= args.steps <= MAX_CONTROL_STEPS:
        raise SystemExit(f"--steps must be within [1, {MAX_CONTROL_STEPS}]")
    if args.blend_steps < 0:
        raise SystemExit("--blend-steps must be nonnegative")
    if not 0 <= args.max_frontier_states <= MAX_FRONTIER_STATES:
        raise SystemExit(
            f"--max-frontier-states must be within [0, {MAX_FRONTIER_STATES}]"
        )
    if (
        any(value < 0 for value in args.launch_min_steps)
        or any(value < 1 for value in args.launch_max_steps)
        or any(not 0.45 <= value <= 0.70 for value in args.mantle_root_x)
        or any(not 0.08 <= value <= 0.22 for value in args.mantle_root_z)
    ):
        raise SystemExit("Timing or physics-gate sweep values are outside safe bounds")
    candidates = [
        {
            "launch_min_steps": minimum,
            "launch_max_steps": maximum,
            "mantle_root_x_m": root_x,
            "mantle_root_z_m": root_z,
        }
        for minimum, maximum, root_x, root_z in itertools.product(
            args.launch_min_steps,
            args.launch_max_steps,
            args.mantle_root_x,
            args.mantle_root_z,
        )
        if maximum >= minimum
    ]
    if not candidates:
        raise SystemExit("The timing grid contains no launch_max >= launch_min pair")
    num_envs = len(candidates) * args.trials_per_candidate
    if num_envs > MAX_ENVS:
        raise SystemExit(
            f"Sweep requires {num_envs} environments, bounded maximum is {MAX_ENVS}"
        )
    return candidates


def _expanded_parameter(
    candidates: list[dict[str, int | float]],
    field: str,
    trials_per_candidate: int,
    *,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.tensor(
        [candidate[field] for candidate in candidates],
        device=device,
        dtype=dtype,
    ).repeat_interleave(trials_per_candidate)


def _configure_guarded_resets(
    env_cfg: Any,
    num_envs: int,
    bank_path: Path,
    trials_per_candidate: int,
) -> None:
    family = env_cfg.events.get("state_bank_family")
    guarded_bank = env_cfg.events.get("root_over_lip_state_bank")
    hard_viewer = env_cfg.events.get("a30_hard_viewer")
    if family is None or guarded_bank is None or hard_viewer is None:
        raise RuntimeError("A35 play cfg is missing guarded-reset infrastructure")
    family.params["forced_family"] = SOURCE_RESET_FAMILY
    guarded_bank.params.clear()
    guarded_bank.params.update(
        {
            "bank_path": str(bank_path),
            "replay_fraction": 1.0,
            "reset_family": SOURCE_RESET_FAMILY,
            "paired_replay_group_size": trials_per_candidate,
        }
    )
    hard_viewer.params["terrain_levels"] = (HARD_TERRAIN_LEVEL,) * num_envs
    hard_viewer.params["terrain_types"] = (0,) * num_envs


def _validate_live_resets(base_env: ManagerBasedRlEnv) -> None:
    family = getattr(base_env, "_stair_state_bank_family", None)
    rows = getattr(base_env, "_stair_walker_bank_row", None)
    reset_mode = getattr(base_env, "_stair_assisted_reset_mode", None)
    if family is None or rows is None or reset_mode is None:
        raise RuntimeError("A35 guarded reset attribution is unavailable")
    if not torch.all(family == SOURCE_RESET_FAMILY):
        raise RuntimeError("A35 guarded reset did not remain on family 0")
    if not torch.all(rows >= 0):
        raise RuntimeError("Guarded handoff bank did not write every environment")
    if not torch.all(reset_mode == 3):
        raise RuntimeError("Guarded bank replay requires real-handoff reset mode 3")
    if not torch.all(base_env.scene.terrain.terrain_levels == HARD_TERRAIN_LEVEL):
        raise RuntimeError("A35 sweep did not remain on the hard terrain row")


def _raw_shell_frontier_value(base_env: ManagerBasedRlEnv) -> torch.Tensor:
    shell_cfg = base_env.reward_manager.get_term_cfg(
        "stair_true_shell_clearance"
    )
    params = shell_cfg.params
    return stair_true_shell_clearance_frontier_value(
        env=base_env,
        start_x=0.625,
        target_x=STAIR_FACE_X_M,
        start_height=0.150,
        target_height=STANDARD_RISER_HEIGHT_M,
        corridor_half_width=SHELL_CORRIDOR_HALF_WIDTH_M,
        required_latch_name=None,
        shell_half_extents=tuple(params["shell_half_extents"]),
        asset_cfg=params["asset_cfg"],
    )


def _rank_results(results: list[dict[str, object]]) -> tuple[str, list[dict[str, object]]]:
    total_frontier = sum(
        int(result["trainable_frontier_count"]) for result in results
    )
    total_shell = sum(int(result["shell_clear_count"]) for result in results)
    if total_frontier == 0:
        target = "trainable_frontier"
    elif total_shell == 0:
        target = "shell_clear"
    else:
        target = "secured"

    large_step = MAX_CONTROL_STEPS + 1
    for result in results:
        invalid = (
            int(result["nan_count"])
            + int(result["nonfinite_count"])
            + int(result["side_bypass_count"])
            + int(result["stage_order_violation_count"])
            + int(result["rejected_physics_handoff_count"])
        )
        frontier_count = int(result["trainable_frontier_count"])
        shell_count = int(result["shell_clear_count"])
        secured_count = int(result["secured_count"])
        frontier_step = result["earliest_trainable_frontier_step"]
        shell_step = result["earliest_shell_clear_step"]
        secured_step = result["earliest_secured_step"]
        if target == "trainable_frontier":
            stage_key = (
                -frontier_count,
                frontier_step if frontier_step is not None else large_step,
                -float(result["maximum_shell_frontier_value"]),
            )
        elif target == "shell_clear":
            stage_key = (
                -shell_count,
                -frontier_count,
                shell_step if shell_step is not None else large_step,
                -float(result["maximum_shell_frontier_value"]),
                frontier_step if frontier_step is not None else large_step,
            )
        else:
            stage_key = (
                -secured_count,
                -shell_count,
                -frontier_count,
                secured_step if secured_step is not None else large_step,
                shell_step if shell_step is not None else large_step,
                frontier_step if frontier_step is not None else large_step,
            )
        rank_key = (invalid, *stage_key, int(result["timeout_handoff_count"]))
        result["rank_key"] = list(rank_key)

    ranked = sorted(results, key=lambda result: tuple(result["rank_key"]))
    for rank, result in enumerate(ranked, start=1):
        result["rank"] = rank
    return target, ranked


def main() -> int:
    args = _parse_args()
    candidates = _candidate_grid(args)
    bank_path, launch_path, mantle_path, source_bank = _validate_artifacts(args)
    num_envs = len(candidates) * args.trials_per_candidate

    configure_torch_backends()
    env_cfg = load_env_cfg(A35_TASK_ID, play=True)
    agent_cfg = load_rl_cfg(A35_TASK_ID)
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = args.seed
    _configure_guarded_resets(
        env_cfg,
        num_envs,
        bank_path,
        args.trials_per_candidate,
    )

    base_env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
    env.reset()
    _validate_live_resets(base_env)
    initial_source_rows = base_env._stair_walker_bank_row.detach().cpu().clone()

    runner_cls = load_runner_cls(A35_TASK_ID) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=args.device)
    launch_actor = load_frozen_actor(runner, launch_path, device=args.device)
    mantle_actor = load_frozen_actor(runner, mantle_path, device=args.device)
    launch_min_steps = _expanded_parameter(
        candidates,
        "launch_min_steps",
        args.trials_per_candidate,
        device=args.device,
        dtype=torch.long,
    )
    launch_max_steps = _expanded_parameter(
        candidates,
        "launch_max_steps",
        args.trials_per_candidate,
        device=args.device,
        dtype=torch.long,
    )
    mantle_root_x = _expanded_parameter(
        candidates,
        "mantle_root_x_m",
        args.trials_per_candidate,
        device=args.device,
        dtype=torch.float32,
    )
    mantle_root_z = _expanded_parameter(
        candidates,
        "mantle_root_z_m",
        args.trials_per_candidate,
        device=args.device,
        dtype=torch.float32,
    )
    policy = StairOptionPolicy(
        walker=launch_actor,
        launch=launch_actor,
        mantle=mantle_actor,
        recover=mantle_actor,
        env=base_env,
        start_in_launch=True,
        option_blend_steps=args.blend_steps,
        launch_min_steps=launch_min_steps,
        launch_max_steps=launch_max_steps,
        mantle_root_x_m=mantle_root_x,
        mantle_root_z_m=mantle_root_z,
    )

    device = torch.device(args.device)
    option_handoff_steps = torch.full(
        (num_envs,), -1, dtype=torch.long, device=device
    )
    frontier_steps = torch.full_like(option_handoff_steps, -1)
    rejected_physics_steps = torch.full_like(option_handoff_steps, -1)
    timeout_steps = torch.full_like(option_handoff_steps, -1)
    shell_steps = torch.full_like(option_handoff_steps, -1)
    secured_steps = torch.full_like(option_handoff_steps, -1)
    completed = torch.zeros(num_envs, dtype=torch.bool, device=device)
    nan_seen = torch.zeros_like(completed)
    nonfinite_seen = torch.zeros_like(completed)
    side_bypass_seen = torch.zeros_like(completed)
    order_violation_seen = torch.zeros_like(completed)
    max_shell_frontier = torch.zeros(num_envs, dtype=torch.float32, device=device)
    captured_frontier = torch.zeros_like(completed)
    previous_launch_transition = policy.transition_counts[:, 2].clone()
    previous_timeout_transition = policy.transition_counts[:, 5].clone()
    previous_frontier = getattr(
        base_env,
        "_stair_contact_transfer_stage15_policy_achieved",
        torch.zeros_like(completed),
    ).clone()
    previous_shell = getattr(
        base_env,
        "_stair_true_shell_clearance_policy_achieved",
        torch.zeros_like(completed),
    ).clone()
    previous_secured = getattr(
        base_env,
        "_stair_first_tread_secured_latched",
        torch.zeros_like(completed),
    ).clone()
    frontier_chunks: list[dict[str, Any]] = []
    seen_frontier_digests: set[str] = set()
    duplicate_frontier_states = 0
    steps_run = 0
    step_dt = float(base_env.step_dt)

    try:
        for steps_run in range(1, args.steps + 1):
            active = ~completed
            if not bool(active.any()):
                break
            robot = base_env.scene["robot"].data
            origins = base_env.scene.terrain.env_origins
            local = robot.root_link_pos_w - origins
            head_contact = policy._head_riser_contact()
            root_gate = (local[:, 0] >= mantle_root_x) & (
                local[:, 2] >= mantle_root_z
            )
            exact_policy_physics_gate = head_contact | root_gate
            phase_step_before = policy.phase_step.clone()
            policy_physics_transition = (
                active
                & (policy.phase == policy.LAUNCH)
                & (phase_step_before >= launch_min_steps)
                & exact_policy_physics_gate
            )
            finite_replay_state = (
                torch.isfinite(base_env.sim.data.qpos)
                .reshape(num_envs, -1)
                .all(dim=-1)
                & torch.isfinite(base_env.sim.data.qvel)
                .reshape(num_envs, -1)
                .all(dim=-1)
            )
            valid_frontier_geometry = (
                finite_replay_state
                & torch.isfinite(local).all(dim=-1)
                & (torch.abs(local[:, 1]) <= SHELL_CORRIDOR_HALF_WIDTH_M)
                & (local[:, 0] >= HEAD_FRONTIER_MIN_ROOT_X_M)
                & (local[:, 2] >= MIN_VALID_ROOT_Z_M)
            )
            pre_transition_physics = (
                policy_physics_transition & valid_frontier_geometry
            )

            with torch.inference_mode():
                observations = env.get_observations()
                actions = policy(observations)
            launch_transition = policy.transition_counts[:, 2]
            timeout_transition = policy.transition_counts[:, 5]
            new_launch_transition = active & (
                launch_transition > previous_launch_transition
            )
            new_timeout_transition = active & (
                timeout_transition > previous_timeout_transition
            )
            frontier_event = new_launch_transition & pre_transition_physics
            rejected_physics_event = (
                new_launch_transition
                & policy_physics_transition
                & ~valid_frontier_geometry
            )
            timeout_event = new_timeout_transition
            _record_first_step(
                option_handoff_steps, frontier_event, steps_run - 1
            )
            _record_first_step(
                rejected_physics_steps, rejected_physics_event, steps_run - 1
            )
            _record_first_step(timeout_steps, timeout_event, steps_run - 1)

            previous_launch_transition = launch_transition.clone()
            previous_timeout_transition = timeout_transition.clone()
            with torch.inference_mode():
                _, _, dones, _ = env.step(actions)

            robot = base_env.scene["robot"].data
            origins = base_env.scene.terrain.env_origins
            local = robot.root_link_pos_w - origins
            frontier = getattr(
                base_env,
                "_stair_contact_transfer_stage15_policy_achieved",
                previous_frontier,
            )
            shell = getattr(
                base_env,
                "_stair_true_shell_clearance_policy_achieved",
                previous_shell,
            )
            secured = getattr(
                base_env,
                "_stair_first_tread_secured_latched",
                previous_secured,
            )
            new_shell = active & shell & ~previous_shell
            new_secured = active & secured & ~previous_secured
            new_frontier = active & frontier & ~previous_frontier
            _record_first_step(frontier_steps, new_frontier, steps_run)
            _record_first_step(shell_steps, new_shell, steps_run)
            _record_first_step(secured_steps, new_secured, steps_run)
            order_violation_seen |= new_shell & (frontier_steps < 0)
            order_violation_seen |= new_secured & (shell_steps < 0)

            capture_event = new_frontier & ~captured_frontier
            captured_frontier[capture_event] = True
            remaining_capture = args.max_frontier_states - sum(
                int(chunk["root_qpos_local"].shape[0])
                for chunk in frontier_chunks
            )
            candidate_ids = capture_event.nonzero(as_tuple=False).squeeze(-1)
            if remaining_capture > 0 and len(candidate_ids) > 0:
                candidate_states = capture_walk_state_rows(
                    base_env, candidate_ids
                )
                keep_rows: list[int] = []
                for row in range(len(candidate_ids)):
                    digest = _exact_state_row_digest(candidate_states, row)
                    if digest in seen_frontier_digests:
                        duplicate_frontier_states += 1
                        continue
                    seen_frontier_digests.add(digest)
                    keep_rows.append(row)
                    if len(keep_rows) >= remaining_capture:
                        break
                if keep_rows:
                    keep = torch.tensor(keep_rows, dtype=torch.long)
                    kept_ids = candidate_ids[keep.to(device)]
                    chunk = _select_nested_rows(candidate_states, keep)
                    chunk["sweep_candidate_index"] = torch.div(
                        kept_ids.detach().cpu(),
                        args.trials_per_candidate,
                        rounding_mode="floor",
                    )
                    chunk["sweep_trial_index"] = (
                        kept_ids.detach().cpu() % args.trials_per_candidate
                    )
                    chunk["source_guarded_bank_row"] = (
                        base_env._stair_walker_bank_row[kept_ids]
                        .detach()
                        .cpu()
                        .clone()
                    )
                    chunk["frontier_control_step"] = torch.full(
                        (len(kept_ids),), steps_run, dtype=torch.long
                    )
                    chunk["frontier_launch_phase_step"] = policy.phase_step[
                        kept_ids
                    ].detach().cpu().clone()
                    chunk["frontier_head_contact"] = (
                        policy._head_riser_contact()[kept_ids]
                        .detach()
                        .cpu()
                        .clone()
                    )
                    chunk["frontier_root_threshold"] = (
                        (
                            (local[:, 0] >= mantle_root_x)
                            & (local[:, 2] >= mantle_root_z)
                        )[kept_ids]
                        .detach()
                        .cpu()
                        .clone()
                    )
                    chunk["frontier_launch_min_steps"] = launch_min_steps[
                        kept_ids
                    ].detach().cpu().clone()
                    chunk["frontier_launch_max_steps"] = launch_max_steps[
                        kept_ids
                    ].detach().cpu().clone()
                    chunk["frontier_mantle_root_x_m"] = mantle_root_x[
                        kept_ids
                    ].detach().cpu().clone()
                    chunk["frontier_mantle_root_z_m"] = mantle_root_z[
                        kept_ids
                    ].detach().cpu().clone()
                    frontier_chunks.append(chunk)
            shell_frontier = _raw_shell_frontier_value(base_env)
            max_shell_frontier = torch.maximum(
                max_shell_frontier,
                torch.nan_to_num(shell_frontier, nan=0.0, posinf=0.0, neginf=0.0),
            )
            finite_state = (
                torch.isfinite(base_env.sim.data.qpos)
                .reshape(num_envs, -1)
                .all(dim=-1)
                & torch.isfinite(base_env.sim.data.qvel)
                .reshape(num_envs, -1)
                .all(dim=-1)
                & torch.isfinite(local).all(dim=-1)
            )
            finite_action = torch.isfinite(actions).reshape(num_envs, -1).all(dim=-1)
            nonfinite_seen |= active & ~(finite_state & finite_action)
            nan_termination = base_env.termination_manager.get_term("nan_state").bool()
            nan_seen |= active & nan_termination
            side_bypass_seen |= (
                active
                & (local[:, 0] >= STAIR_FACE_X_M)
                & (torch.abs(local[:, 1]) > SIDE_BYPASS_HALF_WIDTH_M)
                & ~shell
            )
            completed |= active & dones.bool()
            previous_frontier = frontier.clone()
            previous_shell = shell.clone()
            previous_secured = secured.clone()
    finally:
        transition_counts = policy.transition_counts.detach().cpu().clone()
        env.close()

    frontier_bank_path: str | None = None
    frontier_state_count = 0
    if frontier_chunks:
        frontier_states = concatenate_walk_state_rows(frontier_chunks)
        frontier_state_count = walk_state_count(frontier_states)
        frontier_output = args.frontier_bank_output.expanduser().resolve()
        _write_bank_atomic(
            frontier_output,
            {
                "schema_version": BANK_SCHEMA_VERSION,
                "metadata": {
                    "task": A35_TASK_ID,
                    "source": "stair_option_policy_a35_stage15_frontier",
                    "source_guarded_handoff_bank": str(bank_path),
                    "source_guarded_handoff_bank_sha256": _sha256(bank_path),
                    "launch_checkpoint": str(launch_path),
                    "launch_checkpoint_sha256": _sha256(launch_path),
                    "mantle_checkpoint": str(mantle_path),
                    "mantle_checkpoint_sha256": _sha256(mantle_path),
                    "riser_height_m": STANDARD_RISER_HEIGHT_M,
                    "tread_depth_m": STANDARD_TREAD_DEPTH_M,
                    "num_steps": STANDARD_NUM_STEPS,
                    "num_states": frontier_state_count,
                    "joint_names": source_bank["metadata"]["joint_names"],
                    "first_stage15_policy_event_per_environment": True,
                    "timeout_transitions_excluded": True,
                    "exact_full_state_deduplication": True,
                },
                "states": frontier_states,
            },
        )
        frontier_bank_path = str(frontier_output)
    else:
        args.frontier_bank_output.expanduser().resolve().unlink(missing_ok=True)

    results: list[dict[str, object]] = []
    for candidate_index, candidate in enumerate(candidates):
        start = candidate_index * args.trials_per_candidate
        stop = start + args.trials_per_candidate
        selection = slice(start, stop)
        candidate_option_handoff = (
            option_handoff_steps[selection].detach().cpu()
        )
        candidate_frontier = frontier_steps[selection].detach().cpu()
        candidate_rejected_physics = (
            rejected_physics_steps[selection].detach().cpu()
        )
        candidate_timeout = timeout_steps[selection].detach().cpu()
        candidate_shell = shell_steps[selection].detach().cpu()
        candidate_secured = secured_steps[selection].detach().cpu()
        option_handoff_count = int(
            (candidate_option_handoff >= 0).sum().item()
        )
        frontier_count = int((candidate_frontier >= 0).sum().item())
        shell_count = int((candidate_shell >= 0).sum().item())
        secured_count = int((candidate_secured >= 0).sum().item())
        result: dict[str, object] = {
            "candidate_index": candidate_index,
            **candidate,
            "option_handoff_count": option_handoff_count,
            "option_handoff_rate": (
                option_handoff_count / args.trials_per_candidate
            ),
            "trainable_frontier_count": frontier_count,
            "trainable_frontier_rate": (
                frontier_count / args.trials_per_candidate
            ),
            "shell_clear_count": shell_count,
            "shell_clear_rate": shell_count / args.trials_per_candidate,
            "secured_count": secured_count,
            "secured_rate": secured_count / args.trials_per_candidate,
            "timeout_handoff_count": int((candidate_timeout >= 0).sum().item()),
            "rejected_physics_handoff_count": int(
                (candidate_rejected_physics >= 0).sum().item()
            ),
            "earliest_option_handoff_step": _earliest(
                candidate_option_handoff
            ),
            "earliest_trainable_frontier_step": _earliest(
                candidate_frontier
            ),
            "earliest_shell_clear_step": _earliest(candidate_shell),
            "earliest_secured_step": _earliest(candidate_secured),
            "maximum_shell_frontier_value": float(
                max_shell_frontier[selection].max().item()
            ),
            "mean_maximum_shell_frontier_value": float(
                max_shell_frontier[selection].mean().item()
            ),
            "nan_count": int(nan_seen[selection].sum().item()),
            "nonfinite_count": int(nonfinite_seen[selection].sum().item()),
            "side_bypass_count": int(side_bypass_seen[selection].sum().item()),
            "stage_order_violation_count": int(
                order_violation_seen[selection].sum().item()
            ),
            "completed_count": int(completed[selection].sum().item()),
            "option_transition_counts": transition_counts[selection]
            .sum(dim=0)
            .tolist(),
            "raw_trials": {
                "source_guarded_bank_rows": initial_source_rows[selection].tolist(),
                "option_handoff_steps": _step_list(
                    candidate_option_handoff
                ),
                "trainable_frontier_steps": _step_list(
                    candidate_frontier
                ),
                "rejected_physics_handoff_steps": _step_list(
                    candidate_rejected_physics
                ),
                "timeout_handoff_steps": _step_list(candidate_timeout),
                "shell_clear_steps": _step_list(candidate_shell),
                "secured_steps": _step_list(candidate_secured),
                "nan": nan_seen[selection].detach().cpu().tolist(),
                "nonfinite": nonfinite_seen[selection].detach().cpu().tolist(),
                "side_bypass": side_bypass_seen[selection]
                .detach()
                .cpu()
                .tolist(),
                "stage_order_violation": order_violation_seen[selection]
                .detach()
                .cpu()
                .tolist(),
                "completed": completed[selection].detach().cpu().tolist(),
            },
        }
        results.append(result)

    ranking_target, ranked = _rank_results(results)
    output = args.output.expanduser().resolve()
    payload: dict[str, object] = {
        "schema_version": 1,
        "task": A35_TASK_ID,
        "headless": True,
        "seed": args.seed,
        "source_guarded_handoff_bank": str(bank_path),
        "source_guarded_handoff_bank_sha256": _sha256(bank_path),
        "source_guarded_handoff_bank_states": walk_state_count(source_bank["states"]),
        "launch_option": {
            "role": "LAUNCH",
            "checkpoint": str(launch_path),
            "checkpoint_sha256": _sha256(launch_path),
            "description": args.launch_description,
        },
        "mantle_option": {
            "role": "MANTLE",
            "checkpoint": str(mantle_path),
            "checkpoint_sha256": _sha256(mantle_path),
            "description": "A35 model_9",
        },
        "unused_role_aliases": {
            "WALK": "LAUNCH actor, unreachable because start_in_launch=True",
            "RECOVER": "MANTLE actor after secured",
        },
        "terrain_level": HARD_TERRAIN_LEVEL,
        "reset_family": SOURCE_RESET_FAMILY,
        "start_in_launch": True,
        "blend_steps": args.blend_steps,
        "num_candidates": len(candidates),
        "trials_per_candidate": args.trials_per_candidate,
        "num_envs": num_envs,
        "requested_steps": args.steps,
        "steps_run": steps_run,
        "step_dt_s": step_dt,
        "ranking_target": ranking_target,
        "ranking_rule": (
            "first globally unsolved stage in trainable_frontier, shell_clear, "
            "secured; invalid candidates rank after exact valid candidates"
        ),
        "frontier_definition": {
            "event": "A35 Stage 1.5 loaded-tread, face-free policy latch",
            "latch": "_stair_contact_transfer_stage15_policy_achieved",
            "valid_corridor_max_abs_y_m": SHELL_CORRIDOR_HALF_WIDTH_M,
            "captured_after_physics_step": True,
        },
        "option_handoff_definition": {
            "transition": "StairOptionPolicy LAUNCH to LAUNCH_TO_MANTLE/MANTLE",
            "physics": "head-riser contact or candidate root x/z gate after launch_min_steps",
            "minimum_root_x_m": HEAD_FRONTIER_MIN_ROOT_X_M,
            "minimum_root_z_m": MIN_VALID_ROOT_Z_M,
            "timeouts_excluded": True,
        },
        "frontier_bank_output": frontier_bank_path,
        "frontier_state_count": frontier_state_count,
        "duplicate_frontier_states_rejected": duplicate_frontier_states,
        "selected": ranked[0],
        "results": ranked,
    }
    _write_json_atomic(output, payload)
    print(
        json.dumps(
            {
                "ranking_target": ranking_target,
                "selected": ranked[0],
                "frontier_state_count": frontier_state_count,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    print(f"[stair-option-sweep] wrote {output}")
    if frontier_bank_path is not None:
        print(
            f"[stair-option-sweep] captured {frontier_state_count} frontier states "
            f"to {frontier_bank_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

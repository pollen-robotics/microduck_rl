#!/usr/bin/env python3
"""Measure Stage-A clearance on full-height assisted stair reset states.

This report is a curriculum gate only. It must never promote a dashboard video;
strict full-route promotion remains in evaluate_stair_checkpoint.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from tempfile import NamedTemporaryFile

import mjlab.tasks  # noqa: F401  # Populate the task registry.
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

from mjlab_microduck.tasks.mdp import classify_standard_stair_contacts

A30_TASK_ID = "Mjlab-Stairs-Virtual-Lip-Transfer-RSI-Specialist-MicroDuck"
TASK_IDS = (
    "Mjlab-Stairs-Assisted-Specialist-MicroDuck",
    "Mjlab-Stairs-Bridge-Specialist-MicroDuck",
    "Mjlab-Stairs-Walker-Bank-Specialist-MicroDuck",
    "Mjlab-Stairs-Launch-Bank-Specialist-MicroDuck",
    "Mjlab-Stairs-Apex-Mantle-Specialist-MicroDuck",
    "Mjlab-Stairs-Roulade-Bank-Specialist-MicroDuck",
    "Mjlab-Stairs-Curriculum-RSI-Specialist-MicroDuck",
    "Mjlab-Stairs-Contact-Mantle-RSI-Specialist-MicroDuck",
    "Mjlab-Stairs-Soft-Dynamics-RSI-Specialist-MicroDuck",
    "Mjlab-Stairs-Medium-Dynamics-RSI-Specialist-MicroDuck",
    "Mjlab-Stairs-Contact-Release-RSI-Specialist-MicroDuck",
    "Mjlab-Stairs-Lip-Commitment-RSI-Specialist-MicroDuck",
    "Mjlab-Stairs-Lip-Checkpoint-RSI-Specialist-MicroDuck",
    "Mjlab-Stairs-Frontier-Collocation-RSI-Specialist-MicroDuck",
    "Mjlab-Stairs-Terminal-Position-RSI-Specialist-MicroDuck",
    "Mjlab-Stairs-Frontier-Tier-RSI-Specialist-MicroDuck",
    "Mjlab-Stairs-Forward-Propagation-RSI-Specialist-MicroDuck",
    A30_TASK_ID,
    "Mjlab-Stairs-Tread-Contact-Bank-Specialist-MicroDuck",
    "Mjlab-Stairs-Foot-Anchor-Vault-Specialist-MicroDuck",
    "Mjlab-Stairs-Ordered-Vault-Specialist-MicroDuck",
)
RESET_MODES = {
    0: "lip_release",
    1: "shell_brace",
    2: "tread_recovery",
    3: "real_handoff",
}
APEX_MANTLE_RESET_MODES = {
    0: "launch_release",
    1: "head_lever",
    2: "unused_tread",
    3: "real_handoff",
}
ROULADE_BANK_RESET_MODES = {
    0: "launch_release",
    1: "head_lever",
    2: "unused_tread",
    3: "manufacturer_roll_phase",
}
TREAD_CONTACT_BANK_RESET_MODES = {
    0: "unused_launch",
    1: "unused_head_lever",
    2: "unused_tread",
    3: "real_tread_contact",
}
A30_RESET_MODES = {
    0: "face_no_tread_bank",
    1: "tread_contact_bank",
    2: "manufacturer_roulade",
}
A30_RESET_FAMILY_OVERRIDES = {
    "mixed": None,
    "face-no-tread": 0,
    "tread-contact": 1,
    "manufacturer": 2,
}
BODY_PART_CONTACT_SENSORS = {
    "head": "head_ground_contact",
    "trunk": "trunk_ground_contact",
    "legs": "legs_ground_contact",
    "feet": "feet_stair_contact",
}
A12_IGNORE_INITIAL_CONTROL_STEPS = 3
A12_ROOT_CENTER_OVER_LIP_MIN_X_M = 0.665
A12_ROOT_CENTER_OVER_LIP_MIN_Z_M = 0.175
A12_ROOT_CENTER_OVER_LIP_MAX_ABS_Y_M = 0.20
A12_ROOT_CENTER_OVER_LIP_HOLD_STEPS = 2
A12_FULL_SHELL_CLEAR_MIN_X_M = 0.700
A12_FULL_SHELL_CLEAR_MIN_Z_M = 0.198
A12_FULL_SHELL_CLEAR_MAX_ABS_Y_M = 0.20
A12_FULL_SHELL_CLEAR_HOLD_STEPS = 4
A12_SIDE_BYPASS_MIN_X_M = 0.660
A12_SIDE_BYPASS_MIN_ABS_Y_M = 0.36
JOINT_FRONTIER_START_X_M = 0.540
JOINT_FRONTIER_TARGET_X_M = 0.665
JOINT_FRONTIER_START_Z_M = 0.100
JOINT_FRONTIER_TARGET_Z_M = 0.175
JOINT_FRONTIER_MAX_ABS_Y_M = 0.20
JOINT_FRONTIER_MILESTONES = (
    (0.600, 0.150),
    (0.625, 0.160),
    (0.640, 0.170),
    (0.650, 0.175),
    (0.660, 0.175),
    (0.665, 0.175),
    (0.700, 0.198),
)
TERMINAL_TARGET_X_M = 0.720
TERMINAL_TARGET_Y_M = 0.0
TERMINAL_TARGET_Z_M = 0.205
TERMINAL_X_SCALE_M = 0.08
TERMINAL_Y_SCALE_M = 0.12
TERMINAL_Z_SCALE_M = 0.06
TERMINAL_MAX_ABS_Y_M = 0.20
TERMINAL_WINDOW_S = 0.50


class TerminalPositionTrajectoryMetrics:
    """Accumulate the unweighted A27 terminal score per trajectory."""

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        max_episode_length: int,
        step_dt: float,
    ):
        self._score = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self._max_episode_length = max_episode_length
        self._step_dt = step_dt
        self._window_steps = max(1, int(round(TERMINAL_WINDOW_S / step_dt)))

    def observe(
        self,
        local_root_pos: torch.Tensor,
        episode_steps: torch.Tensor,
    ) -> None:
        if local_root_pos.shape != (self._score.numel(), 3):
            raise ValueError("local_root_pos must have shape (num_envs, 3)")
        if episode_steps.shape != self._score.shape:
            raise ValueError("episode_steps must have shape (num_envs,)")

        finite = torch.isfinite(local_root_pos).all(dim=-1)
        dx = (local_root_pos[:, 0] - TERMINAL_TARGET_X_M) / TERMINAL_X_SCALE_M
        dy = (local_root_pos[:, 1] - TERMINAL_TARGET_Y_M) / TERMINAL_Y_SCALE_M
        dz = (local_root_pos[:, 2] - TERMINAL_TARGET_Z_M) / TERMINAL_Z_SCALE_M
        score = 1.0 / (1.0 + dx.square() + dy.square() + dz.square())
        remaining_steps = self._max_episode_length - episode_steps
        eligible = (
            finite
            & (local_root_pos[:, 1].abs() <= TERMINAL_MAX_ABS_Y_M)
            & (remaining_steps >= 1)
            & (remaining_steps <= self._window_steps)
        )
        payout = torch.where(
            eligible,
            score / TERMINAL_WINDOW_S,
            torch.zeros_like(score),
        )
        self._score += payout * self._step_dt

    def complete(self, done_mask: torch.Tensor) -> list[float]:
        completed = self._score[done_mask].detach().cpu().tolist()
        self._score[done_mask] = 0.0
        return completed


class A12TrajectoryMetrics:
    """Latch strict stair evidence without mixing states across trajectories."""

    def __init__(self, num_envs: int, device: torch.device | str):
        self._root_center_hold_steps = torch.zeros(
            num_envs, dtype=torch.long, device=device
        )
        self._full_shell_hold_steps = torch.zeros_like(
            self._root_center_hold_steps
        )
        self.root_center_over_lip_latched = torch.zeros(
            num_envs, dtype=torch.bool, device=device
        )
        self.full_shell_clear_latched = torch.zeros_like(
            self.root_center_over_lip_latched
        )
        self.side_bypass_latched = torch.zeros_like(
            self.root_center_over_lip_latched
        )

    def observe(
        self,
        local_root_pos: torch.Tensor,
        episode_steps: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Observe a pre-step state and return newly latched event masks."""
        if local_root_pos.shape != (self._root_center_hold_steps.numel(), 3):
            raise ValueError("local_root_pos must have shape (num_envs, 3)")
        if episode_steps.shape != self._root_center_hold_steps.shape:
            raise ValueError("episode_steps must have shape (num_envs,)")

        x = local_root_pos[:, 0]
        abs_y = torch.abs(local_root_pos[:, 1])
        z = local_root_pos[:, 2]
        eligible = episode_steps >= A12_IGNORE_INITIAL_CONTROL_STEPS
        finite = torch.isfinite(local_root_pos).all(dim=-1)

        root_center_candidate = (
            eligible
            & finite
            & (x >= A12_ROOT_CENTER_OVER_LIP_MIN_X_M)
            & (z >= A12_ROOT_CENTER_OVER_LIP_MIN_Z_M)
            & (abs_y <= A12_ROOT_CENTER_OVER_LIP_MAX_ABS_Y_M)
        )
        self._root_center_hold_steps = torch.where(
            root_center_candidate,
            self._root_center_hold_steps + 1,
            torch.zeros_like(self._root_center_hold_steps),
        )
        new_root_center_over_lip = (
            self._root_center_hold_steps >= A12_ROOT_CENTER_OVER_LIP_HOLD_STEPS
        ) & ~self.root_center_over_lip_latched
        self.root_center_over_lip_latched |= new_root_center_over_lip

        full_shell_candidate = (
            eligible
            & finite
            & (x >= A12_FULL_SHELL_CLEAR_MIN_X_M)
            & (z >= A12_FULL_SHELL_CLEAR_MIN_Z_M)
            & (abs_y <= A12_FULL_SHELL_CLEAR_MAX_ABS_Y_M)
        )
        self._full_shell_hold_steps = torch.where(
            full_shell_candidate,
            self._full_shell_hold_steps + 1,
            torch.zeros_like(self._full_shell_hold_steps),
        )
        new_full_shell_clear = (
            self._full_shell_hold_steps >= A12_FULL_SHELL_CLEAR_HOLD_STEPS
        ) & ~self.full_shell_clear_latched
        self.full_shell_clear_latched |= new_full_shell_clear

        side_bypass_candidate = (
            eligible
            & finite
            & (x >= A12_SIDE_BYPASS_MIN_X_M)
            & (abs_y > A12_SIDE_BYPASS_MIN_ABS_Y_M)
            & ~self.full_shell_clear_latched
        )
        new_side_bypass = side_bypass_candidate & ~self.side_bypass_latched
        self.side_bypass_latched |= new_side_bypass

        return {
            "root_center_over_lip": new_root_center_over_lip,
            "full_shell_clear": new_full_shell_clear,
            "side_bypass": new_side_bypass,
        }

    def reset(self, dones: torch.Tensor) -> None:
        """Clear all trajectory-local streaks and latches for completed envs."""
        done_mask = dones.bool()
        self._root_center_hold_steps[done_mask] = 0
        self._full_shell_hold_steps[done_mask] = 0
        self.root_center_over_lip_latched[done_mask] = False
        self.full_shell_clear_latched[done_mask] = False
        self.side_bypass_latched[done_mask] = False


class ContactTrajectoryMetrics:
    """Count one policy-created contact transition per assisted trajectory."""

    def __init__(self, num_envs: int, device: torch.device | str):
        self._previous = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._latched = torch.zeros_like(self._previous)

    def observe(
        self,
        current: torch.Tensor,
        episode_steps: torch.Tensor,
    ) -> torch.Tensor:
        if current.shape != self._previous.shape:
            raise ValueError("current must have shape (num_envs,)")
        if episode_steps.shape != self._previous.shape:
            raise ValueError("episode_steps must have shape (num_envs,)")
        eligible = episode_steps >= A12_IGNORE_INITIAL_CONTROL_STEPS
        newly_contacted = current & ~self._previous & eligible & ~self._latched
        self._latched |= newly_contacted
        self._previous = current.clone()
        return newly_contacted

    def reset(self, dones: torch.Tensor) -> None:
        done_mask = dones.bool()
        self._previous[done_mask] = False
        self._latched[done_mask] = False


class JointFrontierTrajectoryMetrics:
    """Measure same-frame x/z progress without mixing trials or timestamps."""

    def __init__(self, num_envs: int, device: torch.device | str):
        self._best_progress = torch.zeros(
            num_envs, dtype=torch.float32, device=device
        )
        self._best_x = torch.full_like(self._best_progress, float("nan"))
        self._best_z = torch.full_like(self._best_progress, float("nan"))
        self._milestone_latched = torch.zeros(
            (num_envs, len(JOINT_FRONTIER_MILESTONES)),
            dtype=torch.bool,
            device=device,
        )

    def observe(
        self,
        local_root_pos: torch.Tensor,
        episode_steps: torch.Tensor,
    ) -> None:
        if local_root_pos.shape != (self._best_progress.numel(), 3):
            raise ValueError("local_root_pos must have shape (num_envs, 3)")
        if episode_steps.shape != self._best_progress.shape:
            raise ValueError("episode_steps must have shape (num_envs,)")

        x = local_root_pos[:, 0]
        abs_y = torch.abs(local_root_pos[:, 1])
        z = local_root_pos[:, 2]
        eligible = (
            (episode_steps >= A12_IGNORE_INITIAL_CONTROL_STEPS)
            & torch.isfinite(local_root_pos).all(dim=-1)
            & (abs_y <= JOINT_FRONTIER_MAX_ABS_Y_M)
        )
        x_progress = torch.clamp(
            (x - JOINT_FRONTIER_START_X_M)
            / (JOINT_FRONTIER_TARGET_X_M - JOINT_FRONTIER_START_X_M),
            0.0,
            1.0,
        )
        z_progress = torch.clamp(
            (z - JOINT_FRONTIER_START_Z_M)
            / (JOINT_FRONTIER_TARGET_Z_M - JOINT_FRONTIER_START_Z_M),
            0.0,
            1.0,
        )
        # The minimum cannot hide a deficient axis the way separate maxima or
        # an additive score can. It reaches one only when both targets do.
        joint_progress = torch.where(
            eligible,
            torch.minimum(x_progress, z_progress),
            torch.zeros_like(x_progress),
        )
        improved = joint_progress > self._best_progress
        self._best_progress = torch.where(
            improved, joint_progress, self._best_progress
        )
        self._best_x = torch.where(improved, x, self._best_x)
        self._best_z = torch.where(improved, z, self._best_z)

        for index, (min_x, min_z) in enumerate(JOINT_FRONTIER_MILESTONES):
            self._milestone_latched[:, index] |= (
                eligible & (x >= min_x) & (z >= min_z)
            )

    def complete(
        self, dones: torch.Tensor
    ) -> tuple[list[float], list[float], list[float], list[int]]:
        done_mask = dones.bool()
        progress = self._best_progress[done_mask].detach().cpu().tolist()
        best_x = self._best_x[done_mask].detach().cpu().tolist()
        best_z = self._best_z[done_mask].detach().cpu().tolist()
        milestone_counts = (
            self._milestone_latched[done_mask].sum(dim=0).detach().cpu().tolist()
        )
        self._best_progress[done_mask] = 0.0
        self._best_x[done_mask] = float("nan")
        self._best_z[done_mask] = float("nan")
        self._milestone_latched[done_mask] = False
        return progress, best_x, best_z, milestone_counts


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--task", choices=TASK_IDS, default=TASK_IDS[0])
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--root-over-lip-replay-fraction",
        type=float,
        help=(
            "Override the forward-propagation task's exact root-over-lip "
            "state-bank fraction for diagnostic evaluation."
        ),
    )
    parser.add_argument("--terrain-level", type=int, choices=(0, 1, 2))
    parser.add_argument(
        "--reset-family",
        choices=tuple(A30_RESET_FAMILY_OVERRIDES),
        default="mixed",
    )
    parser.add_argument("--output", type=Path)
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = _parse_args()
    is_a30 = args.task == A30_TASK_ID
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")
    if args.num_envs < 1 or args.episodes < 1:
        raise SystemExit("--num-envs and --episodes must be positive")
    if args.root_over_lip_replay_fraction is not None and not (
        0.0 <= args.root_over_lip_replay_fraction <= 1.0
    ):
        raise SystemExit("--root-over-lip-replay-fraction must be within [0, 1]")
    if not is_a30 and args.terrain_level is not None:
        raise SystemExit("--terrain-level is supported only for the A30 task")
    if not is_a30 and args.reset_family != "mixed":
        raise SystemExit("--reset-family is supported only for the A30 task")
    if is_a30:
        reset_modes = A30_RESET_MODES
    elif args.task in {
        "Mjlab-Stairs-Tread-Contact-Bank-Specialist-MicroDuck",
        "Mjlab-Stairs-Foot-Anchor-Vault-Specialist-MicroDuck",
        "Mjlab-Stairs-Ordered-Vault-Specialist-MicroDuck",
    }:
        reset_modes = TREAD_CONTACT_BANK_RESET_MODES
    elif args.task in {
        "Mjlab-Stairs-Roulade-Bank-Specialist-MicroDuck",
        "Mjlab-Stairs-Curriculum-RSI-Specialist-MicroDuck",
        "Mjlab-Stairs-Contact-Mantle-RSI-Specialist-MicroDuck",
        "Mjlab-Stairs-Soft-Dynamics-RSI-Specialist-MicroDuck",
        "Mjlab-Stairs-Medium-Dynamics-RSI-Specialist-MicroDuck",
        "Mjlab-Stairs-Contact-Release-RSI-Specialist-MicroDuck",
        "Mjlab-Stairs-Lip-Commitment-RSI-Specialist-MicroDuck",
        "Mjlab-Stairs-Lip-Checkpoint-RSI-Specialist-MicroDuck",
        "Mjlab-Stairs-Frontier-Collocation-RSI-Specialist-MicroDuck",
        "Mjlab-Stairs-Terminal-Position-RSI-Specialist-MicroDuck",
        "Mjlab-Stairs-Frontier-Tier-RSI-Specialist-MicroDuck",
        "Mjlab-Stairs-Forward-Propagation-RSI-Specialist-MicroDuck",
    }:
        reset_modes = ROULADE_BANK_RESET_MODES
    elif args.task == "Mjlab-Stairs-Apex-Mantle-Specialist-MicroDuck":
        reset_modes = APEX_MANTLE_RESET_MODES
    else:
        reset_modes = RESET_MODES

    configure_torch_backends()
    env_cfg = load_env_cfg(args.task, play=True)
    if args.root_over_lip_replay_fraction is not None:
        bank = env_cfg.events.get("root_over_lip_state_bank")
        if bank is None:
            raise SystemExit(
                "--root-over-lip-replay-fraction requires a task with "
                "root_over_lip_state_bank"
            )
        bank.params["replay_fraction"] = args.root_over_lip_replay_fraction
    agent_cfg = load_rl_cfg(args.task)
    env_cfg.scene.num_envs = args.num_envs
    effective_terrain_level = None
    forced_reset_family = None
    if is_a30:
        hard_viewer = env_cfg.events.get("a30_hard_viewer")
        if hard_viewer is None:
            raise RuntimeError("A30 play cfg is missing the a30_hard_viewer event")
        effective_terrain_level = (
            2 if args.terrain_level is None else args.terrain_level
        )
        hard_viewer.params["terrain_levels"] = (
            effective_terrain_level,
        ) * args.num_envs
        hard_viewer.params["terrain_types"] = (0,) * args.num_envs
        forced_reset_family = A30_RESET_FAMILY_OVERRIDES[args.reset_family]
        state_bank_family = env_cfg.events.get("state_bank_family")
        if state_bank_family is None:
            raise RuntimeError("A30 cfg is missing the state_bank_family event")
        if forced_reset_family is None:
            state_bank_family.params.pop("forced_family", None)
        else:
            state_bank_family.params["forced_family"] = forced_reset_family
    env_cfg.seed = 7
    base_env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
    if is_a30:
        actual_levels = base_env.scene.terrain.terrain_levels
        if not torch.all(actual_levels == effective_terrain_level):
            base_env.close()
            raise RuntimeError(
                "A30 terrain-level override did not reach every evaluation environment"
            )
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
    if is_a30:
        env.reset()
        actual_families = getattr(base_env, "_stair_state_bank_family", None)
        if actual_families is None:
            env.close()
            raise RuntimeError("A30 reset-family assignment is unavailable")
        if forced_reset_family is not None and not torch.all(
            actual_families == forced_reset_family
        ):
            env.close()
            raise RuntimeError(
                "A30 reset-family override did not reach every evaluation environment"
            )
    runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=args.device)
    runner.load(
        str(checkpoint),
        load_cfg={"actor": True},
        strict=True,
        map_location=args.device,
    )
    policy = runner.get_inference_policy(device=args.device)

    clearance_events = 0
    stable_events = 0
    completed_trials = 0
    previous_clearance = torch.zeros(
        args.num_envs, dtype=torch.bool, device=args.device
    )
    previous_stable = torch.zeros_like(previous_clearance)
    previous_secured = torch.zeros_like(previous_clearance)
    previous_contact_release = torch.zeros_like(previous_clearance)
    previous_lip_impulse = torch.zeros_like(previous_clearance)
    previous_lip_commitment = torch.zeros_like(previous_clearance)
    previous_lip_checkpoint_progress = torch.zeros_like(previous_clearance)
    previous_lip_checkpoint_target = torch.zeros_like(previous_clearance)
    previous_coupled_raw_gain10 = torch.zeros_like(previous_clearance)
    previous_coupled_raw_gain25 = torch.zeros_like(previous_clearance)
    previous_coupled_gain10 = torch.zeros_like(previous_clearance)
    previous_coupled_gain25 = torch.zeros_like(previous_clearance)
    previous_coupled_target = torch.zeros_like(previous_clearance)
    previous_coupled_bypass = torch.zeros_like(previous_clearance)
    previous_stage1_policy = torch.zeros_like(previous_clearance)
    previous_stage2_policy = torch.zeros_like(previous_clearance)
    mode_trials = {name: 0 for name in reset_modes.values()}
    mode_clearance = {name: 0 for name in reset_modes.values()}
    mode_stable = {name: 0 for name in reset_modes.values()}
    mode_secured = {name: 0 for name in reset_modes.values()}
    mode_face_contact = {name: 0 for name in reset_modes.values()}
    mode_tread_contact = {name: 0 for name in reset_modes.values()}
    mode_contact_release = {name: 0 for name in reset_modes.values()}
    mode_lip_impulse = {name: 0 for name in reset_modes.values()}
    mode_lip_commitment = {name: 0 for name in reset_modes.values()}
    mode_lip_checkpoint_progress = {name: 0 for name in reset_modes.values()}
    mode_lip_checkpoint_target = {name: 0 for name in reset_modes.values()}
    mode_coupled_raw_gain10 = {name: 0 for name in reset_modes.values()}
    mode_coupled_raw_gain25 = {name: 0 for name in reset_modes.values()}
    mode_coupled_gain10 = {name: 0 for name in reset_modes.values()}
    mode_coupled_gain25 = {name: 0 for name in reset_modes.values()}
    mode_coupled_target = {name: 0 for name in reset_modes.values()}
    mode_coupled_bypass = {name: 0 for name in reset_modes.values()}
    mode_stage1_policy = {name: 0 for name in reset_modes.values()}
    mode_stage2_policy = {name: 0 for name in reset_modes.values()}
    secured_events = 0
    face_contact_events = 0
    tread_contact_events = 0
    contact_release_events = 0
    lip_impulse_events = 0
    lip_commitment_events = 0
    lip_checkpoint_progress_events = 0
    lip_checkpoint_target_events = 0
    coupled_raw_gain10_events = 0
    coupled_raw_gain25_events = 0
    coupled_gain10_events = 0
    coupled_gain25_events = 0
    coupled_target_events = 0
    coupled_bypass_events = 0
    stage1_policy_events = 0
    stage2_policy_events = 0
    secured_before_clearance_events = 0
    nan_termination_events = 0
    tread_contact_source_steps: list[int] = []
    face_contact_metrics = ContactTrajectoryMetrics(args.num_envs, args.device)
    tread_contact_metrics = ContactTrajectoryMetrics(args.num_envs, args.device)
    body_part_tread_metrics = {
        name: ContactTrajectoryMetrics(args.num_envs, args.device)
        for name, sensor_name in BODY_PART_CONTACT_SENSORS.items()
        if sensor_name in base_env.scene.sensors
    }
    body_part_tread_events = {name: 0 for name in body_part_tread_metrics}
    body_part_force_max = {
        name: torch.zeros(args.num_envs, dtype=torch.float32, device=args.device)
        for name in body_part_tread_metrics
    }
    body_part_power_max = {
        name: torch.zeros_like(previous_clearance, dtype=torch.float32)
        for name in body_part_tread_metrics
    }
    body_part_trial_forces = {name: [] for name in body_part_tread_metrics}
    body_part_trial_powers = {name: [] for name in body_part_tread_metrics}
    a12_metrics = A12TrajectoryMetrics(args.num_envs, args.device)
    a12_event_counts = {
        "root_center_over_lip": 0,
        "full_shell_clear": 0,
        "side_bypass": 0,
    }
    mode_a12_event_counts = {
        name: {metric: 0 for metric in a12_event_counts}
        for name in reset_modes.values()
    }
    joint_frontier_metrics = JointFrontierTrajectoryMetrics(
        args.num_envs, args.device
    )
    joint_frontier_progress: list[float] = []
    joint_frontier_best_x: list[float] = []
    joint_frontier_best_z: list[float] = []
    joint_frontier_milestone_counts = [0] * len(JOINT_FRONTIER_MILESTONES)
    terminal_position_metrics = TerminalPositionTrajectoryMetrics(
        args.num_envs,
        args.device,
        base_env.max_episode_length,
        base_env.step_dt,
    )
    terminal_position_scores: list[float] = []
    max_x = torch.full((args.num_envs,), -torch.inf, device=args.device)
    max_z = torch.full_like(max_x, -torch.inf)
    steps = base_env.max_episode_length * args.episodes
    try:
        for _ in range(steps):
            reset_mode_attribute = (
                "_stair_state_bank_family"
                if is_a30
                else "_stair_assisted_reset_mode"
            )
            reset_mode = getattr(base_env, reset_mode_attribute, None)
            if reset_mode is None:
                raise RuntimeError(
                    f"Stair reset mode tracking is unavailable: {reset_mode_attribute}"
                )
            episode_mode = reset_mode.clone()
            episode_control_steps = base_env.episode_length_buf.clone()
            robot = base_env.scene["robot"]
            origins = base_env.scene.terrain.env_origins
            local_pre_step = robot.data.root_link_pos_w - origins
            a12_events = a12_metrics.observe(
                local_pre_step,
                base_env.episode_length_buf,
            )
            joint_frontier_metrics.observe(
                local_pre_step,
                base_env.episode_length_buf,
            )
            terminal_position_metrics.observe(
                local_pre_step,
                base_env.episode_length_buf,
            )
            for metric, event_mask in a12_events.items():
                a12_event_counts[metric] += int(event_mask.sum().item())
                for code, name in reset_modes.items():
                    mode_a12_event_counts[name][metric] += int(
                        (event_mask & (episode_mode == code)).sum().item()
                    )
            with torch.inference_mode():
                observations = env.get_observations()
                actions = policy(observations)
                _, _, dones, _ = env.step(actions)
            nan_termination_events += int(
                base_env.termination_manager.get_term("nan_state").sum().item()
            )

            if is_a30:
                clearance = getattr(
                    base_env,
                    "_stair_true_shell_clearance_policy_achieved",
                    None,
                )
                if clearance is None:
                    raise RuntimeError(
                        "A30 policy-created true shell-clearance tracking is unavailable"
                    )
            else:
                clearance = getattr(
                    base_env, "_stair_first_riser_latched", previous_clearance
                )
            stable = getattr(base_env, "_stair_first_tread_latched", previous_stable)
            secured = getattr(
                base_env, "_stair_first_tread_secured_latched", previous_secured
            )
            contact_release = getattr(
                base_env,
                "_stair_contact_loaded_release_latched",
                previous_contact_release,
            )
            lip_impulse = getattr(
                base_env,
                "_stair_lip_commitment_impulse_latched",
                previous_lip_impulse,
            )
            lip_commitment = getattr(
                base_env,
                "_stair_lip_commitment_latched",
                previous_lip_commitment,
            )
            lip_checkpoint_progress = getattr(
                base_env,
                "_stair_lip_checkpoint_progress_latched",
                previous_lip_checkpoint_progress,
            )
            lip_checkpoint_target = getattr(
                base_env,
                "_stair_lip_checkpoint_target_latched",
                previous_lip_checkpoint_target,
            )
            coupled_gain10 = getattr(
                base_env,
                "_stair_coupled_frontier_gain10_latched",
                previous_coupled_gain10,
            )
            coupled_raw_gain10 = getattr(
                base_env,
                "_stair_coupled_frontier_raw_gain10_latched",
                previous_coupled_raw_gain10,
            )
            coupled_raw_gain25 = getattr(
                base_env,
                "_stair_coupled_frontier_raw_gain25_latched",
                previous_coupled_raw_gain25,
            )
            coupled_gain25 = getattr(
                base_env,
                "_stair_coupled_frontier_gain25_latched",
                previous_coupled_gain25,
            )
            coupled_target = getattr(
                base_env,
                "_stair_coupled_frontier_target_latched",
                previous_coupled_target,
            )
            coupled_bypass = getattr(
                base_env,
                "_stair_coupled_frontier_bypass_latched",
                previous_coupled_bypass,
            )
            if is_a30:
                stage1_policy = getattr(
                    base_env,
                    "_stair_contact_transfer_stage1_policy_achieved",
                    None,
                )
                stage2_policy = getattr(
                    base_env,
                    "_stair_contact_transfer_stage2_policy_achieved",
                    None,
                )
                if stage1_policy is None or stage2_policy is None:
                    raise RuntimeError(
                        "A30 contact-transfer policy-achievement tracking is unavailable"
                    )
            else:
                stage1_policy = previous_stage1_policy
                stage2_policy = previous_stage2_policy
            global_contact = base_env.scene.sensors["robot_ground_contact"].data
            if (
                global_contact.found is None
                or global_contact.pos is None
                or global_contact.normal is None
            ):
                raise RuntimeError(
                    "robot_ground_contact must provide found, pos, and normal"
                )
            raw_face, raw_tread = classify_standard_stair_contacts(
                global_contact.found,
                global_contact.pos,
                global_contact.normal,
                base_env.scene.terrain.env_origins,
            )
            new_clearance = clearance & ~previous_clearance
            if is_a30:
                # The environment deliberately pre-latches a reset state that
                # already satisfies the shell gate. Count only a latch edge
                # created after policy control has begun. Use the pre-step
                # counter so a valid event on a terminal step is retained.
                new_clearance &= (
                    episode_control_steps >= A12_IGNORE_INITIAL_CONTROL_STEPS
                )
            new_stable = stable & ~previous_stable
            new_secured = secured & ~previous_secured
            new_contact_release = contact_release & ~previous_contact_release
            new_lip_impulse = lip_impulse & ~previous_lip_impulse
            new_lip_commitment = lip_commitment & ~previous_lip_commitment
            new_lip_checkpoint_progress = (
                lip_checkpoint_progress & ~previous_lip_checkpoint_progress
            )
            new_lip_checkpoint_target = (
                lip_checkpoint_target & ~previous_lip_checkpoint_target
            )
            new_coupled_raw_gain10 = (
                coupled_raw_gain10 & ~previous_coupled_raw_gain10
            )
            new_coupled_raw_gain25 = (
                coupled_raw_gain25 & ~previous_coupled_raw_gain25
            )
            new_coupled_gain10 = coupled_gain10 & ~previous_coupled_gain10
            new_coupled_gain25 = coupled_gain25 & ~previous_coupled_gain25
            new_coupled_target = coupled_target & ~previous_coupled_target
            new_coupled_bypass = coupled_bypass & ~previous_coupled_bypass
            new_stage1_policy = stage1_policy & ~previous_stage1_policy
            new_stage2_policy = stage2_policy & ~previous_stage2_policy
            new_face_contact = face_contact_metrics.observe(
                raw_face.any(dim=-1), base_env.episode_length_buf
            )
            new_tread_contact = tread_contact_metrics.observe(
                raw_tread.any(dim=-1), base_env.episode_length_buf
            )
            clearance_events += int(new_clearance.sum().item())
            stable_events += int(new_stable.sum().item())
            secured_events += int(new_secured.sum().item())
            if is_a30:
                secured_before_clearance_events += int(
                    (new_secured & ~clearance).sum().item()
                )
            contact_release_events += int(new_contact_release.sum().item())
            lip_impulse_events += int(new_lip_impulse.sum().item())
            lip_commitment_events += int(new_lip_commitment.sum().item())
            lip_checkpoint_progress_events += int(
                new_lip_checkpoint_progress.sum().item()
            )
            lip_checkpoint_target_events += int(new_lip_checkpoint_target.sum().item())
            coupled_raw_gain10_events += int(new_coupled_raw_gain10.sum().item())
            coupled_raw_gain25_events += int(new_coupled_raw_gain25.sum().item())
            coupled_gain10_events += int(new_coupled_gain10.sum().item())
            coupled_gain25_events += int(new_coupled_gain25.sum().item())
            coupled_target_events += int(new_coupled_target.sum().item())
            coupled_bypass_events += int(new_coupled_bypass.sum().item())
            stage1_policy_events += int(new_stage1_policy.sum().item())
            stage2_policy_events += int(new_stage2_policy.sum().item())
            face_contact_events += int(new_face_contact.sum().item())
            tread_contact_events += int(new_tread_contact.sum().item())
            source_steps = getattr(base_env, "_stair_walker_bank_source_step", None)
            if source_steps is not None:
                roll_tread = new_tread_contact & (episode_mode == 3)
                tread_contact_source_steps.extend(
                    int(step)
                    for step in source_steps[roll_tread].detach().cpu().tolist()
                    if int(step) >= 0
                )
            completed_trials += int(dones.sum().item())
            done_mask = dones.bool()
            (
                completed_progress,
                completed_best_x,
                completed_best_z,
                completed_milestones,
            ) = joint_frontier_metrics.complete(done_mask)
            joint_frontier_progress.extend(completed_progress)
            joint_frontier_best_x.extend(completed_best_x)
            joint_frontier_best_z.extend(completed_best_z)
            joint_frontier_milestone_counts = [
                total + completed
                for total, completed in zip(
                    joint_frontier_milestone_counts,
                    completed_milestones,
                    strict=True,
                )
            ]
            terminal_position_scores.extend(
                terminal_position_metrics.complete(done_mask)
            )
            origins = base_env.scene.terrain.env_origins
            for body_part, sensor_name in BODY_PART_CONTACT_SENSORS.items():
                if body_part not in body_part_tread_metrics:
                    continue
                sensor_data = base_env.scene.sensors[sensor_name].data
                if (
                    sensor_data.found is None
                    or sensor_data.force is None
                    or sensor_data.pos is None
                    or sensor_data.normal is None
                ):
                    raise RuntimeError(
                        f"{sensor_name} must provide found, force, pos, and normal"
                    )
                _, body_tread = classify_standard_stair_contacts(
                    sensor_data.found,
                    sensor_data.pos,
                    sensor_data.normal,
                    origins,
                )
                current_body_tread = body_tread.any(dim=-1)
                new_body_tread = body_part_tread_metrics[body_part].observe(
                    current_body_tread, base_env.episode_length_buf
                )
                body_part_tread_events[body_part] += int(
                    new_body_tread.sum().item()
                )
                eligible = (
                    base_env.episode_length_buf
                    >= A12_IGNORE_INITIAL_CONTROL_STEPS
                )
                normal_force = torch.abs(
                    torch.sum(sensor_data.force * sensor_data.normal, dim=-1)
                )
                normal_force = torch.where(
                    body_tread, normal_force, torch.zeros_like(normal_force)
                )
                strongest_slot = normal_force.argmax(dim=-1, keepdim=True)
                strongest_force = normal_force.gather(1, strongest_slot).squeeze(1)
                slot_index = strongest_slot.unsqueeze(-1).expand(-1, -1, 3)
                contact_pos = sensor_data.pos.gather(1, slot_index).squeeze(1)
                contact_force = sensor_data.force.gather(1, slot_index).squeeze(1)
                robot = base_env.scene["robot"]
                lever = contact_pos - robot.data.root_link_pos_w
                pitch_torque = torch.cross(lever, contact_force, dim=-1)[:, 1]
                pitch_power = torch.clamp(
                    pitch_torque * robot.data.root_link_ang_vel_w[:, 1], min=0.0
                )
                pitch_power = torch.where(
                    current_body_tread & eligible,
                    pitch_power,
                    torch.zeros_like(pitch_power),
                )
                strongest_force = torch.where(
                    eligible, strongest_force, torch.zeros_like(strongest_force)
                )
                body_part_force_max[body_part] = torch.maximum(
                    body_part_force_max[body_part], strongest_force
                )
                body_part_power_max[body_part] = torch.maximum(
                    body_part_power_max[body_part], pitch_power
                )
                if torch.any(done_mask):
                    body_part_trial_forces[body_part].extend(
                        body_part_force_max[body_part][done_mask]
                        .detach()
                        .cpu()
                        .tolist()
                    )
                    body_part_trial_powers[body_part].extend(
                        body_part_power_max[body_part][done_mask]
                        .detach()
                        .cpu()
                        .tolist()
                    )
                    body_part_force_max[body_part][done_mask] = 0.0
                    body_part_power_max[body_part][done_mask] = 0.0
            for code, name in reset_modes.items():
                mode_mask = episode_mode == code
                mode_trials[name] += int((done_mask & mode_mask).sum().item())
                mode_clearance[name] += int((new_clearance & mode_mask).sum().item())
                mode_stable[name] += int((new_stable & mode_mask).sum().item())
                mode_secured[name] += int((new_secured & mode_mask).sum().item())
                mode_face_contact[name] += int(
                    (new_face_contact & mode_mask).sum().item()
                )
                mode_tread_contact[name] += int(
                    (new_tread_contact & mode_mask).sum().item()
                )
                mode_contact_release[name] += int(
                    (new_contact_release & mode_mask).sum().item()
                )
                mode_lip_impulse[name] += int(
                    (new_lip_impulse & mode_mask).sum().item()
                )
                mode_lip_commitment[name] += int(
                    (new_lip_commitment & mode_mask).sum().item()
                )
                mode_lip_checkpoint_progress[name] += int(
                    (new_lip_checkpoint_progress & mode_mask).sum().item()
                )
                mode_lip_checkpoint_target[name] += int(
                    (new_lip_checkpoint_target & mode_mask).sum().item()
                )
                mode_coupled_raw_gain10[name] += int(
                    (new_coupled_raw_gain10 & mode_mask).sum().item()
                )
                mode_coupled_raw_gain25[name] += int(
                    (new_coupled_raw_gain25 & mode_mask).sum().item()
                )
                mode_coupled_gain10[name] += int(
                    (new_coupled_gain10 & mode_mask).sum().item()
                )
                mode_coupled_gain25[name] += int(
                    (new_coupled_gain25 & mode_mask).sum().item()
                )
                mode_coupled_target[name] += int(
                    (new_coupled_target & mode_mask).sum().item()
                )
                mode_coupled_bypass[name] += int(
                    (new_coupled_bypass & mode_mask).sum().item()
                )
                mode_stage1_policy[name] += int(
                    (new_stage1_policy & mode_mask).sum().item()
                )
                mode_stage2_policy[name] += int(
                    (new_stage2_policy & mode_mask).sum().item()
                )
            previous_clearance = clearance.clone()
            previous_stable = stable.clone()
            previous_secured = secured.clone()
            previous_contact_release = contact_release.clone()
            previous_lip_impulse = lip_impulse.clone()
            previous_lip_commitment = lip_commitment.clone()
            previous_lip_checkpoint_progress = lip_checkpoint_progress.clone()
            previous_lip_checkpoint_target = lip_checkpoint_target.clone()
            previous_coupled_raw_gain10 = coupled_raw_gain10.clone()
            previous_coupled_raw_gain25 = coupled_raw_gain25.clone()
            previous_coupled_gain10 = coupled_gain10.clone()
            previous_coupled_gain25 = coupled_gain25.clone()
            previous_coupled_target = coupled_target.clone()
            previous_coupled_bypass = coupled_bypass.clone()
            previous_stage1_policy = stage1_policy.clone()
            previous_stage2_policy = stage2_policy.clone()
            previous_clearance[done_mask] = False
            previous_stable[done_mask] = False
            previous_secured[done_mask] = False
            previous_contact_release[done_mask] = False
            previous_lip_impulse[done_mask] = False
            previous_lip_commitment[done_mask] = False
            previous_lip_checkpoint_progress[done_mask] = False
            previous_lip_checkpoint_target[done_mask] = False
            previous_coupled_raw_gain10[done_mask] = False
            previous_coupled_raw_gain25[done_mask] = False
            previous_coupled_gain10[done_mask] = False
            previous_coupled_gain25[done_mask] = False
            previous_coupled_target[done_mask] = False
            previous_coupled_bypass[done_mask] = False
            previous_stage1_policy[done_mask] = False
            previous_stage2_policy[done_mask] = False
            face_contact_metrics.reset(done_mask)
            tread_contact_metrics.reset(done_mask)
            for metrics in body_part_tread_metrics.values():
                metrics.reset(done_mask)
            a12_metrics.reset(done_mask)

            local = robot.data.root_link_pos_w - origins
            max_x = torch.maximum(max_x, torch.nan_to_num(local[:, 0], nan=-torch.inf))
            max_z = torch.maximum(max_z, torch.nan_to_num(local[:, 2], nan=-torch.inf))
            if completed_trials >= args.episodes:
                break
    finally:
        env.close()

    denominator = max(completed_trials, 1)

    def quantiles(values: list[float]) -> dict[str, float]:
        if not values:
            return {}
        samples = torch.tensor(values, dtype=torch.float32)
        return {
            name: float(torch.quantile(samples, probability).item())
            for name, probability in (
                ("p50", 0.50),
                ("p75", 0.75),
                ("p90", 0.90),
                ("p95", 0.95),
                ("max", 1.00),
            )
        }
    eligible_names = tuple(
        name
        for name in reset_modes.values()
        if name not in {"tread_recovery", "unused_tread"}
    )
    eligible_trials = sum(mode_trials[name] for name in eligible_names)
    eligible_clearance = sum(mode_clearance[name] for name in eligible_names)
    mode_report = {}
    for name in reset_modes.values():
        trials = mode_trials[name]
        mode_report[name] = {
            "trials": trials,
            "clearance_events": mode_clearance[name],
            "clearance_rate": mode_clearance[name] / max(trials, 1),
            "stable_tread_events": mode_stable[name],
            "stable_tread_rate": mode_stable[name] / max(trials, 1),
            "secured_tread_events": mode_secured[name],
            "secured_tread_rate": mode_secured[name] / max(trials, 1),
            "riser_face_contact_events": mode_face_contact[name],
            "riser_face_contact_rate": mode_face_contact[name] / max(trials, 1),
            "first_tread_contact_events": mode_tread_contact[name],
            "first_tread_contact_rate": mode_tread_contact[name] / max(trials, 1),
            "contact_loaded_release_events": mode_contact_release[name],
            "contact_loaded_release_rate": mode_contact_release[name]
            / max(trials, 1),
            "lip_commitment_impulse_events": mode_lip_impulse[name],
            "lip_commitment_impulse_rate": mode_lip_impulse[name]
            / max(trials, 1),
            "lip_commitment_events": mode_lip_commitment[name],
            "lip_commitment_rate": mode_lip_commitment[name] / max(trials, 1),
            "lip_checkpoint_progress_events": mode_lip_checkpoint_progress[name],
            "lip_checkpoint_progress_rate": mode_lip_checkpoint_progress[name]
            / max(trials, 1),
            "lip_checkpoint_target_events": mode_lip_checkpoint_target[name],
            "lip_checkpoint_target_rate": mode_lip_checkpoint_target[name]
            / max(trials, 1),
            "coupled_frontier_raw_gain10_events": mode_coupled_raw_gain10[name],
            "coupled_frontier_raw_gain10_rate": mode_coupled_raw_gain10[name]
            / max(trials, 1),
            "coupled_frontier_raw_gain25_events": mode_coupled_raw_gain25[name],
            "coupled_frontier_raw_gain25_rate": mode_coupled_raw_gain25[name]
            / max(trials, 1),
            "coupled_frontier_gain10_events": mode_coupled_gain10[name],
            "coupled_frontier_gain10_rate": mode_coupled_gain10[name]
            / max(trials, 1),
            "coupled_frontier_gain25_events": mode_coupled_gain25[name],
            "coupled_frontier_gain25_rate": mode_coupled_gain25[name]
            / max(trials, 1),
            "coupled_frontier_target_events": mode_coupled_target[name],
            "coupled_frontier_target_rate": mode_coupled_target[name]
            / max(trials, 1),
            "coupled_frontier_bypass_events": mode_coupled_bypass[name],
            "coupled_frontier_bypass_rate": mode_coupled_bypass[name]
            / max(trials, 1),
            "root_center_over_lip_events": mode_a12_event_counts[name][
                "root_center_over_lip"
            ],
            "root_center_over_lip_rate": mode_a12_event_counts[name][
                "root_center_over_lip"
            ]
            / max(trials, 1),
            "full_shell_clear_events": mode_a12_event_counts[name][
                "full_shell_clear"
            ],
            "full_shell_clear_rate": mode_a12_event_counts[name][
                "full_shell_clear"
            ]
            / max(trials, 1),
            "side_bypass_events": mode_a12_event_counts[name]["side_bypass"],
            "side_bypass_rate": mode_a12_event_counts[name]["side_bypass"]
            / max(trials, 1),
        }
        if is_a30:
            mode_report[name].update(
                {
                    "contact_transfer_stage1_policy_achieved_events": (
                        mode_stage1_policy[name]
                    ),
                    "contact_transfer_stage1_policy_achieved_rate": (
                        mode_stage1_policy[name] / max(trials, 1)
                    ),
                    "contact_transfer_stage2_policy_achieved_events": (
                        mode_stage2_policy[name]
                    ),
                    "contact_transfer_stage2_policy_achieved_rate": (
                        mode_stage2_policy[name] / max(trials, 1)
                    ),
                }
            )
    iteration_match = re.search(r"model_(\d+)$", checkpoint.stem)
    report: dict[str, object] = {
        "schema_version": 10,
        "task": args.task,
        "checkpoint": str(checkpoint),
        "checkpoint_iteration": (
            int(iteration_match.group(1)) if iteration_match is not None else None
        ),
        "checkpoint_sha256": _sha256(checkpoint),
        "standard_riser_height_m": 0.17,
        "standard_tread_depth_m": 0.28,
        "num_envs": args.num_envs,
        "root_over_lip_replay_fraction_override": (
            args.root_over_lip_replay_fraction
        ),
        "requested_episodes_per_env": args.episodes,
        "completed_trials": completed_trials,
        "clearance_events": clearance_events,
        "clearance_rate": clearance_events / denominator,
        "clearance_eligible_trials": eligible_trials,
        "clearance_eligible_rate": eligible_clearance / max(eligible_trials, 1),
        "stable_tread_events": stable_events,
        "stable_tread_rate": stable_events / denominator,
        "secured_tread_events": secured_events,
        "secured_tread_rate": secured_events / denominator,
        "root_center_over_lip_events": a12_event_counts[
            "root_center_over_lip"
        ],
        "root_center_over_lip_rate": a12_event_counts[
            "root_center_over_lip"
        ]
        / denominator,
        "full_shell_clear_events": a12_event_counts["full_shell_clear"],
        "full_shell_clear_rate": a12_event_counts["full_shell_clear"]
        / denominator,
        "side_bypass_events": a12_event_counts["side_bypass"],
        "side_bypass_rate": a12_event_counts["side_bypass"] / denominator,
        "nan_termination_events": nan_termination_events,
        "nan_termination_rate": nan_termination_events / denominator,
        "terminal_position_objective": {
            "definition": (
                "per-trial integrated unweighted reciprocal position score "
                "during the final 0.5 seconds"
            ),
            "target_m": {
                "x": TERMINAL_TARGET_X_M,
                "y": TERMINAL_TARGET_Y_M,
                "z": TERMINAL_TARGET_Z_M,
            },
            "scales_m": {
                "x": TERMINAL_X_SCALE_M,
                "y": TERMINAL_Y_SCALE_M,
                "z": TERMINAL_Z_SCALE_M,
            },
            "max_abs_y_m": TERMINAL_MAX_ABS_Y_M,
            "window_s": TERMINAL_WINDOW_S,
            "window_control_steps": int(
                round(TERMINAL_WINDOW_S / base_env.step_dt)
            ),
            "integrated_raw_score_mean": (
                sum(terminal_position_scores)
                / max(len(terminal_position_scores), 1)
            ),
            "integrated_raw_score_quantiles": quantiles(
                terminal_position_scores
            ),
            "per_trial_integrated_raw_scores": terminal_position_scores,
        },
        "a12_metric_definitions": {
            "ignored_initial_control_steps": A12_IGNORE_INITIAL_CONTROL_STEPS,
            "sampling": "pre_step_reset_safe",
            "root_center_over_lip": {
                "min_x_m": A12_ROOT_CENTER_OVER_LIP_MIN_X_M,
                "min_z_m": A12_ROOT_CENTER_OVER_LIP_MIN_Z_M,
                "max_abs_y_m": A12_ROOT_CENTER_OVER_LIP_MAX_ABS_Y_M,
                "hold_control_steps": A12_ROOT_CENTER_OVER_LIP_HOLD_STEPS,
            },
            "full_shell_clear": {
                "min_x_m": A12_FULL_SHELL_CLEAR_MIN_X_M,
                "min_z_m": A12_FULL_SHELL_CLEAR_MIN_Z_M,
                "max_abs_y_m": A12_FULL_SHELL_CLEAR_MAX_ABS_Y_M,
                "hold_control_steps": A12_FULL_SHELL_CLEAR_HOLD_STEPS,
            },
            "side_bypass": {
                "min_x_m": A12_SIDE_BYPASS_MIN_X_M,
                "min_abs_y_exclusive_m": A12_SIDE_BYPASS_MIN_ABS_Y_M,
                "must_occur_before": "full_shell_clear",
            },
            "secured_tread": "existing_strict_environment_latch",
        },
        "joint_frontier": {
            "definition": (
                "per-trial maximum of min(normalized_x, normalized_z) in the "
                "strict lateral corridor; x and z are sampled in the same frame"
            ),
            "start_x_m": JOINT_FRONTIER_START_X_M,
            "target_x_m": JOINT_FRONTIER_TARGET_X_M,
            "start_z_m": JOINT_FRONTIER_START_Z_M,
            "target_z_m": JOINT_FRONTIER_TARGET_Z_M,
            "max_abs_y_m": JOINT_FRONTIER_MAX_ABS_Y_M,
            "progress_quantiles": quantiles(joint_frontier_progress),
            "best_same_frame_x_m_quantiles": quantiles(
                [value for value in joint_frontier_best_x if value == value]
            ),
            "best_same_frame_z_m_quantiles": quantiles(
                [value for value in joint_frontier_best_z if value == value]
            ),
            "milestones": {
                f"x>={min_x:.3f},z>={min_z:.3f}": {
                    "events": events,
                    "rate": events / denominator,
                }
                for (min_x, min_z), events in zip(
                    JOINT_FRONTIER_MILESTONES,
                    joint_frontier_milestone_counts,
                    strict=True,
                )
            },
        },
        "riser_face_contact_events": face_contact_events,
        "riser_face_contact_rate": face_contact_events / denominator,
        "first_tread_contact_events": tread_contact_events,
        "first_tread_contact_rate": tread_contact_events / denominator,
        "contact_loaded_release_events": contact_release_events,
        "contact_loaded_release_rate": contact_release_events / denominator,
        "lip_commitment_impulse_events": lip_impulse_events,
        "lip_commitment_impulse_rate": lip_impulse_events / denominator,
        "lip_commitment_events": lip_commitment_events,
        "lip_commitment_rate": lip_commitment_events / denominator,
        "lip_checkpoint_progress_events": lip_checkpoint_progress_events,
        "lip_checkpoint_progress_rate": lip_checkpoint_progress_events / denominator,
        "lip_checkpoint_target_events": lip_checkpoint_target_events,
        "lip_checkpoint_target_rate": lip_checkpoint_target_events / denominator,
        "coupled_frontier_raw_gain10_events": coupled_raw_gain10_events,
        "coupled_frontier_raw_gain10_rate": coupled_raw_gain10_events / denominator,
        "coupled_frontier_raw_gain25_events": coupled_raw_gain25_events,
        "coupled_frontier_raw_gain25_rate": coupled_raw_gain25_events / denominator,
        "coupled_frontier_gain10_events": coupled_gain10_events,
        "coupled_frontier_gain10_rate": coupled_gain10_events / denominator,
        "coupled_frontier_gain25_events": coupled_gain25_events,
        "coupled_frontier_gain25_rate": coupled_gain25_events / denominator,
        "coupled_frontier_target_events": coupled_target_events,
        "coupled_frontier_target_rate": coupled_target_events / denominator,
        "coupled_frontier_bypass_events": coupled_bypass_events,
        "coupled_frontier_bypass_rate": coupled_bypass_events / denominator,
        "body_part_tread_contact_events": body_part_tread_events,
        "body_part_tread_contact_rates": {
            name: events / denominator
            for name, events in body_part_tread_events.items()
        },
        "body_part_tread_normal_force_n": {
            name: quantiles(values)
            for name, values in body_part_trial_forces.items()
        },
        "body_part_positive_pitch_power_w": {
            name: quantiles(values)
            for name, values in body_part_trial_powers.items()
        },
        "manufacturer_source_steps_with_tread_contact": dict(
            sorted(Counter(tread_contact_source_steps).items())
        ),
        "reset_modes": mode_report,
        "best_route_x_m": float(max_x.max().item()),
        "best_root_height_m": float(max_z.max().item()),
        "maxima_include_assisted_reset_state": True,
        "promotion_eligible": False,
    }
    if is_a30:
        report.update(
            {
                "schema_version": 11,
                "clearance_latch_authoritative": (
                    "_stair_true_shell_clearance_policy_achieved"
                ),
                "terrain_level_override": args.terrain_level,
                "effective_terrain_level": effective_terrain_level,
                "reset_family_override": args.reset_family,
                "forced_reset_family": forced_reset_family,
                "contact_transfer_stage1_policy_achieved_events": (
                    stage1_policy_events
                ),
                "contact_transfer_stage1_policy_achieved_rate": (
                    stage1_policy_events / denominator
                ),
                "contact_transfer_stage2_policy_achieved_events": (
                    stage2_policy_events
                ),
                "contact_transfer_stage2_policy_achieved_rate": (
                    stage2_policy_events / denominator
                ),
                "secured_before_true_shell_clearance_events": (
                    secured_before_clearance_events
                ),
            }
        )
    output = args.output or checkpoint.with_suffix(".assisted-eval.json")
    _write_json_atomic(output.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"[assisted-stair-eval] wrote {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

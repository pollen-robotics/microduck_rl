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

TASK_IDS = (
    "Mjlab-Stairs-Assisted-Specialist-MicroDuck",
    "Mjlab-Stairs-Bridge-Specialist-MicroDuck",
    "Mjlab-Stairs-Walker-Bank-Specialist-MicroDuck",
    "Mjlab-Stairs-Launch-Bank-Specialist-MicroDuck",
    "Mjlab-Stairs-Apex-Mantle-Specialist-MicroDuck",
    "Mjlab-Stairs-Roulade-Bank-Specialist-MicroDuck",
    "Mjlab-Stairs-Curriculum-RSI-Specialist-MicroDuck",
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--task", choices=TASK_IDS, default=TASK_IDS[0])
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
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
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")
    if args.num_envs < 1 or args.episodes < 1:
        raise SystemExit("--num-envs and --episodes must be positive")
    if args.task in {
        "Mjlab-Stairs-Tread-Contact-Bank-Specialist-MicroDuck",
        "Mjlab-Stairs-Foot-Anchor-Vault-Specialist-MicroDuck",
        "Mjlab-Stairs-Ordered-Vault-Specialist-MicroDuck",
    }:
        reset_modes = TREAD_CONTACT_BANK_RESET_MODES
    elif args.task in {
        "Mjlab-Stairs-Roulade-Bank-Specialist-MicroDuck",
        "Mjlab-Stairs-Curriculum-RSI-Specialist-MicroDuck",
    }:
        reset_modes = ROULADE_BANK_RESET_MODES
    elif args.task == "Mjlab-Stairs-Apex-Mantle-Specialist-MicroDuck":
        reset_modes = APEX_MANTLE_RESET_MODES
    else:
        reset_modes = RESET_MODES

    configure_torch_backends()
    env_cfg = load_env_cfg(args.task, play=True)
    agent_cfg = load_rl_cfg(args.task)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = 7
    base_env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
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
    mode_trials = {name: 0 for name in reset_modes.values()}
    mode_clearance = {name: 0 for name in reset_modes.values()}
    mode_stable = {name: 0 for name in reset_modes.values()}
    mode_secured = {name: 0 for name in reset_modes.values()}
    mode_face_contact = {name: 0 for name in reset_modes.values()}
    mode_tread_contact = {name: 0 for name in reset_modes.values()}
    secured_events = 0
    face_contact_events = 0
    tread_contact_events = 0
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
    max_x = torch.full((args.num_envs,), -torch.inf, device=args.device)
    max_z = torch.full_like(max_x, -torch.inf)
    steps = base_env.max_episode_length * args.episodes
    try:
        for _ in range(steps):
            reset_mode = getattr(base_env, "_stair_assisted_reset_mode", None)
            if reset_mode is None:
                raise RuntimeError("Assisted stair reset mode tracking is unavailable")
            episode_mode = reset_mode.clone()
            robot = base_env.scene["robot"]
            origins = base_env.scene.terrain.env_origins
            local_pre_step = robot.data.root_link_pos_w - origins
            a12_events = a12_metrics.observe(
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

            clearance = getattr(
                base_env, "_stair_first_riser_latched", previous_clearance
            )
            stable = getattr(base_env, "_stair_first_tread_latched", previous_stable)
            secured = getattr(
                base_env, "_stair_first_tread_secured_latched", previous_secured
            )
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
            new_stable = stable & ~previous_stable
            new_secured = secured & ~previous_secured
            new_face_contact = face_contact_metrics.observe(
                raw_face.any(dim=-1), base_env.episode_length_buf
            )
            new_tread_contact = tread_contact_metrics.observe(
                raw_tread.any(dim=-1), base_env.episode_length_buf
            )
            clearance_events += int(new_clearance.sum().item())
            stable_events += int(new_stable.sum().item())
            secured_events += int(new_secured.sum().item())
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
            previous_clearance = clearance.clone()
            previous_stable = stable.clone()
            previous_secured = secured.clone()
            previous_clearance[done_mask] = False
            previous_stable[done_mask] = False
            previous_secured[done_mask] = False
            face_contact_metrics.reset(done_mask)
            tread_contact_metrics.reset(done_mask)
            for metrics in body_part_tread_metrics.values():
                metrics.reset(done_mask)
            a12_metrics.reset(done_mask)

            local = robot.data.root_link_pos_w - origins
            max_x = torch.maximum(max_x, torch.nan_to_num(local[:, 0], nan=-torch.inf))
            max_z = torch.maximum(max_z, torch.nan_to_num(local[:, 2], nan=-torch.inf))
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
    iteration_match = re.search(r"model_(\d+)$", checkpoint.stem)
    report: dict[str, object] = {
        "schema_version": 4,
        "task": args.task,
        "checkpoint": str(checkpoint),
        "checkpoint_iteration": (
            int(iteration_match.group(1)) if iteration_match is not None else None
        ),
        "checkpoint_sha256": _sha256(checkpoint),
        "standard_riser_height_m": 0.17,
        "standard_tread_depth_m": 0.28,
        "num_envs": args.num_envs,
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
        "riser_face_contact_events": face_contact_events,
        "riser_face_contact_rate": face_contact_events / denominator,
        "first_tread_contact_events": tread_contact_events,
        "first_tread_contact_rate": tread_contact_events / denominator,
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
    output = args.output or checkpoint.with_suffix(".assisted-eval.json")
    _write_json_atomic(output.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"[assisted-stair-eval] wrote {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

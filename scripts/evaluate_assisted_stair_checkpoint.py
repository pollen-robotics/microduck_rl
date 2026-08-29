#!/usr/bin/env python3
"""Measure Stage-A clearance on full-height assisted stair reset states.

This report is a curriculum gate only. It must never promote a dashboard video;
strict full-route promotion remains in evaluate_stair_checkpoint.py.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
import re
from dataclasses import asdict
from pathlib import Path
from tempfile import NamedTemporaryFile

import mjlab.tasks  # noqa: F401  # Populate the task registry.
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

TASK_IDS = (
    "Mjlab-Stairs-Assisted-Specialist-MicroDuck",
    "Mjlab-Stairs-Bridge-Specialist-MicroDuck",
    "Mjlab-Stairs-Walker-Bank-Specialist-MicroDuck",
    "Mjlab-Stairs-Launch-Bank-Specialist-MicroDuck",
    "Mjlab-Stairs-Apex-Mantle-Specialist-MicroDuck",
    "Mjlab-Stairs-Roulade-Bank-Specialist-MicroDuck",
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
    if args.task == "Mjlab-Stairs-Roulade-Bank-Specialist-MicroDuck":
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
    previous_face_contact = torch.zeros_like(previous_clearance)
    previous_tread_contact = torch.zeros_like(previous_clearance)
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
    max_x = torch.full((args.num_envs,), -torch.inf, device=args.device)
    max_z = torch.full_like(max_x, -torch.inf)
    steps = base_env.max_episode_length * args.episodes
    try:
        for _ in range(steps):
            reset_mode = getattr(base_env, "_stair_assisted_reset_mode", None)
            if reset_mode is None:
                raise RuntimeError("Assisted stair reset mode tracking is unavailable")
            episode_mode = reset_mode.clone()
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
            face_contact = getattr(
                base_env,
                "_stair_riser_face_contact_latched",
                previous_face_contact,
            )
            tread_contact = getattr(
                base_env,
                "_stair_first_tread_contact_latched",
                previous_tread_contact,
            )
            new_clearance = clearance & ~previous_clearance
            new_stable = stable & ~previous_stable
            new_secured = secured & ~previous_secured
            new_face_contact = face_contact & ~previous_face_contact
            new_tread_contact = tread_contact & ~previous_tread_contact
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
            previous_face_contact = face_contact.clone()
            previous_tread_contact = tread_contact.clone()
            previous_clearance[done_mask] = False
            previous_stable[done_mask] = False
            previous_secured[done_mask] = False
            previous_face_contact[done_mask] = False
            previous_tread_contact[done_mask] = False

            robot = base_env.scene["robot"]
            origins = base_env.scene.terrain.env_origins
            local = robot.data.root_link_pos_w - origins
            max_x = torch.maximum(max_x, torch.nan_to_num(local[:, 0], nan=-torch.inf))
            max_z = torch.maximum(max_z, torch.nan_to_num(local[:, 2], nan=-torch.inf))
    finally:
        env.close()

    denominator = max(completed_trials, 1)
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
        }
    iteration_match = re.search(r"model_(\d+)$", checkpoint.stem)
    report: dict[str, object] = {
        "schema_version": 3,
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
        "riser_face_contact_events": face_contact_events,
        "riser_face_contact_rate": face_contact_events / denominator,
        "first_tread_contact_events": tread_contact_events,
        "first_tread_contact_rate": tread_contact_events / denominator,
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

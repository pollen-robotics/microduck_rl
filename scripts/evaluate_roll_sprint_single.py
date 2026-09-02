#!/usr/bin/env python3
"""Validate one deterministic roll-race robot and its exported ONNX policy."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import mjlab.tasks  # noqa: F401  # Populate the task registry.
import numpy as np
import onnxruntime as ort
import torch
from evaluate_roll_sprint_checkpoint import (
    CANONICAL_RACE_DURATION_S,
    MIN_RECOVERED_REROLLS_FOR_TARGET,
    MIN_VALID_ROLLS_FOR_TARGET,
    RACE_LANE_SPACING,
    TARGET_DISTANCE_M,
    TASK_ID,
    RollCycleAuditor,
    _refresh_manual_start_state,
    _sha256,
    _write_json_atomic,
    heading_from_quat,
    sensor_contact,
)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

from mjlab_microduck.tasks import mdp as microduck_mdp

ACTOR_OBSERVATION_DIM = 61
ACTION_DIM = 14
ONNX_PARITY_ATOL = 1.0e-4


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=CANONICAL_RACE_DURATION_S)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _actor_tensor(observations) -> torch.Tensor:
    if isinstance(observations, torch.Tensor):
        return observations
    try:
        actor = observations["actor"]
    except (KeyError, TypeError) as error:
        raise TypeError("policy observations do not contain an actor tensor") from error
    if not isinstance(actor, torch.Tensor):
        raise TypeError("actor observations are not a tensor")
    return actor


def deployment_contract_pass(
    *,
    input_dim: int,
    output_dim: int,
    actions_finite: bool,
    max_abs_action_error: float,
) -> bool:
    return bool(
        input_dim == ACTOR_OBSERVATION_DIM
        and output_dim == ACTION_DIM
        and actions_finite
        and math.isfinite(max_abs_action_error)
        and max_abs_action_error <= ONNX_PARITY_ATOL
    )


def single_robot_race_pass(
    robot: dict[str, object],
    *,
    valid_roll_count: int,
    recovered_reroll_count: int,
    nan_count: int,
    out_of_bounds_count: int,
) -> bool:
    return bool(
        robot.get("target_10m_pass") is True
        and robot.get("road_corridor_pass") is True
        and valid_roll_count >= MIN_VALID_ROLLS_FOR_TARGET
        and recovered_reroll_count >= MIN_RECOVERED_REROLLS_FOR_TARGET
        and nan_count == 0
        and out_of_bounds_count == 0
    )


def _single_alignment_pass(
    *,
    forward_start: torch.Tensor,
    lateral_start: torch.Tensor,
    race_origin: torch.Tensor,
    reward_heading: torch.Tensor,
    body_yaw: torch.Tensor,
    reward_forward_origin: torch.Tensor,
    reward_course_lateral: torch.Tensor,
    reward_course_center: torch.Tensor,
) -> bool:
    return bool(
        torch.allclose(forward_start, torch.zeros_like(forward_start), atol=1.0e-7)
        and torch.allclose(lateral_start, torch.zeros_like(lateral_start), atol=1.0e-7)
        and torch.allclose(race_origin, torch.zeros_like(race_origin), atol=1.0e-7)
        and torch.allclose(
            reward_heading,
            torch.tensor(
                [[1.0, 0.0]],
                device=reward_heading.device,
                dtype=reward_heading.dtype,
            ),
            atol=1.0e-7,
        )
        and torch.allclose(body_yaw, torch.zeros_like(body_yaw), atol=1.0e-7)
        and torch.allclose(
            reward_forward_origin, forward_start, atol=1.0e-7
        )
        and torch.allclose(
            reward_course_lateral, torch.zeros_like(reward_course_lateral), atol=1.0e-7
        )
        and torch.allclose(
            reward_course_center, torch.zeros_like(reward_course_center), atol=1.0e-7
        )
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    checkpoint = args.checkpoint.expanduser().resolve()
    onnx_path = args.onnx.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not checkpoint.is_file() or checkpoint.suffix.lower() != ".pt":
        raise SystemExit(f"Checkpoint not found: {checkpoint}")
    if not onnx_path.is_file() or onnx_path.suffix.lower() != ".onnx":
        raise SystemExit(f"ONNX policy not found: {onnx_path}")
    if args.duration <= 0.0:
        raise SystemExit("--duration must be positive")

    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
        raise SystemExit("Expected exactly one ONNX input and one ONNX output")
    onnx_input = session.get_inputs()[0]
    onnx_output = session.get_outputs()[0]
    input_dim = int(onnx_input.shape[-1])
    output_dim = int(onnx_output.shape[-1])

    configure_torch_backends()
    env_cfg = load_env_cfg(TASK_ID, play=True)
    agent_cfg = load_rl_cfg(TASK_ID)
    env_cfg.scene.num_envs = 1
    env_cfg.scene.env_spacing = RACE_LANE_SPACING
    env_cfg.scene.terrain.env_spacing = RACE_LANE_SPACING
    env_cfg.seed = 0
    env_cfg.auto_reset = False
    env_cfg.episode_length_s = args.duration
    reset_cfg = env_cfg.events["set_roll_sprint_state"]
    reset_cfg.params["standing_prob"] = 1.0
    reset_cfg.params["midroll_prob"] = 0.0
    reset_cfg.params["postroll_prob"] = 0.0
    reset_cfg.params["crouch_prob"] = 0.0
    reset_cfg.params["ground_recovery_prob"] = 0.0
    reset_cfg.params["yaw_range"] = (0.0, 0.0)

    base_env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=args.device)
    runner.load(
        str(checkpoint),
        load_cfg={"actor": True},
        strict=True,
        map_location=args.device,
    )
    policy = runner.get_inference_policy(device=args.device)
    race_origins = microduck_mdp.arrange_roll_sprint_race_start(
        base_env, RACE_LANE_SPACING
    )
    _refresh_manual_start_state(base_env)

    robot = base_env.scene["robot"]
    race_headings = base_env._roll_sprint_heading_w.clone()
    race_forward_starts = microduck_mdp._roll_sprint_forward_position(
        base_env, robot, race_headings
    )
    race_lateral_starts = microduck_mdp._roll_sprint_lateral_position(
        base_env, robot, race_headings
    )
    race_forward_origins = base_env._roll_sprint_forward_origin.clone()
    race_course_lateral = base_env._roll_sprint_course_lateral_position.clone()
    race_course_centers = base_env._roll_sprint_course_center_xy_w.clone()
    body_headings = heading_from_quat(robot.data.root_link_quat_w)
    body_yaws = torch.atan2(body_headings[:, 1], body_headings[:, 0])
    alignment_pass = _single_alignment_pass(
        forward_start=race_forward_starts,
        lateral_start=race_lateral_starts,
        race_origin=race_origins,
        reward_heading=race_headings,
        body_yaw=body_yaws,
        reward_forward_origin=race_forward_origins,
        reward_course_lateral=race_course_lateral,
        reward_course_center=race_course_centers,
    )

    head_ids, _ = robot.find_bodies("jaw_soft")
    head_id = head_ids[0]
    auditor = RollCycleAuditor(
        robot.data.root_link_pos_w[:, :2],
        robot.data.root_link_quat_w,
        robot.data.root_link_lin_vel_w[:, 2],
        base_env.step_dt,
        course_center_xy=race_origins[0, :2],
    )
    alive = torch.ones(1, dtype=torch.bool, device=args.device)
    termination_seen = {
        name: torch.zeros_like(alive)
        for name in base_env.termination_manager.active_terms
    }
    parity_error_sum = 0.0
    parity_element_count = 0
    max_abs_action_error = 0.0
    onnx_actions_finite = True
    steps = round(args.duration / base_env.step_dt)

    try:
        for _ in range(steps):
            with torch.inference_mode():
                observations = env.get_observations()
                actor_obs = _actor_tensor(observations)
                actions = policy(observations)
            actor_np = actor_obs.detach().cpu().numpy().astype(np.float32)
            onnx_actions = session.run(
                [onnx_output.name], {onnx_input.name: actor_np}
            )[0]
            finite = bool(np.isfinite(onnx_actions).all())
            onnx_actions_finite &= finite
            if finite:
                error = np.abs(
                    onnx_actions
                    - actions.detach().cpu().numpy().astype(np.float32)
                )
                max_abs_action_error = max(
                    max_abs_action_error, float(error.max(initial=0.0))
                )
                parity_error_sum += float(error.sum())
                parity_element_count += int(error.size)
            else:
                break

            onnx_action_tensor = torch.from_numpy(onnx_actions).to(
                device=args.device,
                dtype=actions.dtype,
            )
            with torch.inference_mode():
                _, _, dones, _ = env.step(onnx_action_tensor)
            auditor.observe(
                position_xy=robot.data.root_link_pos_w[:, :2],
                root_quat=robot.data.root_link_quat_w,
                head_quat=robot.data.body_link_quat_w[:, head_id],
                linear_velocity_w=robot.data.root_link_lin_vel_w,
                angular_velocity_b=robot.data.root_link_ang_vel_b,
                support=sensor_contact(base_env, "robot_ground_contact"),
                foot_support=sensor_contact(base_env, "feet_ground_contact"),
                head_contact=sensor_contact(base_env, "head_ground_contact"),
                root_height=(
                    robot.data.root_link_pos_w[:, 2]
                    - base_env.scene.terrain.env_origins[:, 2]
                ),
                active=alive,
            )
            for name in termination_seen:
                termination_seen[name] |= alive & base_env.termination_manager.get_term(
                    name
                )
            alive &= ~dones.bool()
            if not bool(alive.any()):
                break
    finally:
        env.close()

    race_summary = auditor.summary(args.duration)
    per_robot = race_summary["per_robot"]
    assert isinstance(per_robot, list) and len(per_robot) == 1
    robot_report = per_robot[0]
    assert isinstance(robot_report, dict)
    nan_count = int(race_summary["nan_env_count"])
    out_of_bounds_count = int(
        termination_seen.get("out_of_terrain_bounds", torch.zeros_like(alive))
        .sum()
        .item()
    )
    valid_roll_count = int(race_summary["total_valid_roll_count"])
    recovered_reroll_count = int(
        race_summary["total_recovered_and_rerolled_count"]
    )
    contract_pass = deployment_contract_pass(
        input_dim=input_dim,
        output_dim=output_dim,
        actions_finite=onnx_actions_finite,
        max_abs_action_error=max_abs_action_error,
    )
    race_pass = single_robot_race_pass(
        robot_report,
        valid_roll_count=valid_roll_count,
        recovered_reroll_count=recovered_reroll_count,
        nan_count=nan_count,
        out_of_bounds_count=out_of_bounds_count,
    )
    report: dict[str, object] = {
        "schema_version": 1,
        "evaluation_mode": "single_robot_sim2real_simulation_gate",
        "task": TASK_ID,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "onnx": str(onnx_path),
        "onnx_sha256": _sha256(onnx_path),
        "duration_s": args.duration,
        "rollout_action_source": "cpu_onnx_runtime",
        "target_distance_m": TARGET_DISTANCE_M,
        "seed": env_cfg.seed,
        "alignment_pass": alignment_pass,
        "deployment_contract": {
            "input_name": onnx_input.name,
            "input_dim": input_dim,
            "output_name": onnx_output.name,
            "output_dim": output_dim,
            "cpu_onnx_runtime": True,
            "actions_finite": onnx_actions_finite,
            "max_abs_action_error": max_abs_action_error,
            "mean_abs_action_error": (
                parity_error_sum / parity_element_count
                if parity_element_count
                else None
            ),
            "parity_tolerance": ONNX_PARITY_ATOL,
            "metadata": session.get_modelmeta().custom_metadata_map,
            "pass": contract_pass,
        },
        **race_summary,
        "termination_counts": {
            name: int(values.sum().item())
            for name, values in termination_seen.items()
        },
        "out_of_bounds_env_count": out_of_bounds_count,
        "single_robot_race_pass": race_pass,
        "simulation_side_sim2real_pass": bool(
            alignment_pass and contract_pass and race_pass
        ),
        "hardware_validation_performed": False,
    }
    _write_json_atomic(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"[roll-sprint-single] wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

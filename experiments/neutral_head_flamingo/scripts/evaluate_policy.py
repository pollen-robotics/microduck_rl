"""Independent scripted evaluation of Remi Fabre's official Flamingo ONNX."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np


CONTROL_DT = 0.02
DECIMATION = 4
TIMELINE = (
    (0.0, 0.0, 1.0),   # stand, right side selected
    (2.0, 1.0, 1.0),   # right foot down, left lifted
    (8.0, 0.0, 1.0),   # lower to two feet
    (11.0, 0.0, -1.0), # select opposite side while standing
    (12.0, 1.0, -1.0), # left foot down, right lifted
    (18.0, 0.0, -1.0), # lower to two feet
)
DURATION_S = 22.0


def command_at(time_s: float, timeline=TIMELINE) -> tuple[float, float]:
    flag, side = timeline[0][1:]
    for start, next_flag, next_side in timeline:
        if time_s >= start:
            flag, side = next_flag, next_side
    return flag, side


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worktree", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--freeze-head",
        action="store_true",
        help="Counterfactual: hold the four neck/head actuators at HOME.",
    )
    parser.add_argument(
        "--lock-head-physics",
        action="store_true",
        help="Diagnostic only: constrain the four head joint positions and velocities exactly at HOME.",
    )
    parser.add_argument(
        "--long-hold",
        action="store_true",
        help="Run ten-second holds on each leg with a stand transition.",
    )
    parser.add_argument(
        "--left-first",
        action="store_true",
        help="With --long-hold, hold on the left foot before the right foot.",
    )
    parser.add_argument(
        "--single-leg",
        choices=("right", "left"),
        help="Render only one stationary single-support hold, without the ballet sequence.",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=10.0,
        help="Duration of the --single-leg hold (default: 10 seconds).",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Collect telemetry without creating a renderer or MP4.",
    )
    parser.add_argument(
        "--save-distillation-data",
        action="store_true",
        help="Save deployable observations and teacher actions during --freeze-head.",
    )
    args = parser.parse_args()
    worktree = args.worktree.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.single_leg:
        side = 1.0 if args.single_leg == "right" else -1.0
        timeline = (
            (0.0, 0.0, side),
            (2.0, 1.0, side),
            (2.0 + args.hold_seconds, 0.0, side),
        )
        duration_s = 5.0 + args.hold_seconds
    elif args.long_hold:
        if args.left_first:
            timeline = (
                (0.0, 0.0, -1.0),
                (2.0, 1.0, -1.0),
                (12.0, 0.0, -1.0),
                (15.0, 1.0, 1.0),
                (25.0, 0.0, 1.0),
            )
        else:
            timeline = (
                (0.0, 0.0, 1.0),
                (2.0, 1.0, 1.0),
                (12.0, 0.0, 1.0),
                (15.0, 1.0, -1.0),
                (25.0, 0.0, -1.0),
            )
        duration_s = 30.0
    else:
        timeline = TIMELINE
        duration_s = DURATION_S

    module_path = worktree / "scripts/infer_policy.py"
    spec = importlib.util.spec_from_file_location("official_infer_policy", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    xml_path = worktree / "src/mjlab_microduck/robot/microduck/scene.xml"
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    model.opt.timestep = 0.005
    data = mujoco.MjData(model)

    from bam.model import load_model

    torque_limit = load_model(motor_name="xl330", model="m6").kt.value * 1.75
    model.actuator_forcerange[:, 0] = -torque_limit
    model.actuator_forcerange[:, 1] = torque_limit
    model.actuator_forcelimited[:] = 1

    policy = module.PolicyInference(
        model,
        data,
        action_scale=1.0,
        use_projected_gravity=True,
        new_cmd_obs=True,
        flamingo_onnx_path=str(args.policy.resolve()),
    )

    freejoint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    qpos_adr = model.jnt_qposadr[freejoint_id]
    data.qpos[qpos_adr : qpos_adr + 7] = [0.0, 0.0, 0.125, 1.0, 0.0, 0.0, 0.0]
    for index, joint_qpos in enumerate(policy.joint_qpos_indices):
        data.qpos[joint_qpos] = policy.default_pose[index]
    data.ctrl[:] = policy.default_pose
    mujoco.mj_forward(model, data)

    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    left_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_foot_collision")
    right_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_foot_collision")
    left_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
    right_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")
    trunk_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    foot_geoms = {left_geom, right_geom}

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.0, 0.0, 0.12]
    camera.distance = 0.58
    camera.azimuth = 145
    camera.elevation = -18
    renderer = None if args.no_video else mujoco.Renderer(model, height=480, width=640)
    writer = None if args.no_video else imageio.get_writer(
        output_dir / "official-flamingo-local-evaluation.mp4", fps=25
    )

    rows: list[dict] = []
    distillation_observations: list[np.ndarray] = []
    distillation_actions: list[np.ndarray] = []
    total_steps = int(duration_s / CONTROL_DT)
    for step in range(total_steps):
        time_s = step * CONTROL_DT
        flag, side = command_at(time_s, timeline)
        policy.flamingo_mode = bool(flag)
        policy.flamingo_side = side
        policy.current_policy = "flamingo"
        policy.ort_session = policy.flamingo_session
        policy._update_command()
        deployable_obs = policy.get_observations().copy()
        if args.freeze_head:
            # The deployed controller will observe its physically applied head
            # commands (zero), not the unconstrained teacher's private history.
            deployable_obs[34 + 5 : 34 + 9] = 0.0
        teacher_action = policy.infer()
        applied_action = teacher_action.copy()
        if args.freeze_head:
            applied_action[5:9] = 0.0
        policy.apply_action(teacher_action)
        if args.freeze_head:
            data.ctrl[5:9] = policy.default_pose[5:9]
        if args.save_distillation_data:
            distillation_observations.append(deployable_obs)
            distillation_actions.append(applied_action)
        for _ in range(DECIMATION):
            mujoco.mj_step(model, data)
            if args.lock_head_physics:
                data.qpos[policy.joint_qpos_indices[5:9]] = policy.default_pose[5:9]
                data.qvel[policy.joint_qvel_indices[5:9]] = 0.0
                data.ctrl[5:9] = policy.default_pose[5:9]
                mujoco.mj_forward(model, data)

        left_contact = False
        right_contact = False
        forbidden_contact = False
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            pair = {int(contact.geom1), int(contact.geom2)}
            if floor_id not in pair:
                continue
            other = next(iter(pair - {floor_id}), floor_id)
            left_contact |= other == left_geom
            right_contact |= other == right_geom
            forbidden_contact |= other not in foot_geoms

        rotation = data.xmat[trunk_body].reshape(3, 3)
        tilt_deg = math.degrees(math.acos(float(np.clip(rotation[2, 2], -1.0, 1.0))))
        rows.append(
            {
                "time_s": round((step + 1) * CONTROL_DT, 4),
                "flag": int(flag),
                "side": int(side),
                "left_contact": int(left_contact),
                "right_contact": int(right_contact),
                "left_foot_z_m": float(data.site_xpos[left_site, 2]),
                "right_foot_z_m": float(data.site_xpos[right_site, 2]),
                "trunk_z_m": float(data.xpos[trunk_body, 2]),
                "trunk_tilt_deg": tilt_deg,
                "head_yaw_rad": float(data.qpos[policy.joint_qpos_indices[7]]),
                "head_roll_rad": float(data.qpos[policy.joint_qpos_indices[8]]),
                "neck_pitch_rad": float(data.qpos[policy.joint_qpos_indices[5]]),
                "head_pitch_rad": float(data.qpos[policy.joint_qpos_indices[6]]),
                "action_head_yaw": float(policy.last_action[7]),
                "action_head_roll": float(policy.last_action[8]),
                "forbidden_ground_contact": int(forbidden_contact),
                "root_x_m": float(data.qpos[qpos_adr]),
                "root_y_m": float(data.qpos[qpos_adr + 1]),
            }
        )
        if writer is not None and renderer is not None and step % 2 == 0:
            camera.lookat[0] = data.qpos[qpos_adr]
            camera.lookat[1] = data.qpos[qpos_adr + 1]
            renderer.update_scene(data, camera=camera)
            writer.append_data(renderer.render())

    if writer is not None:
        writer.close()
    if renderer is not None:
        renderer.close()

    csv_path = output_dir / "official-flamingo-local-telemetry.csv"
    with csv_path.open("w", newline="") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        csv_writer.writeheader()
        csv_writer.writerows(rows)

    def window(start: float, end: float) -> list[dict]:
        return [row for row in rows if start <= row["time_s"] < end]

    if args.single_leg:
        hold = window(4.0, 2.0 + args.hold_seconds)
        final_stand = window(3.0 + args.hold_seconds, 5.0 + args.hold_seconds)
    elif args.long_hold:
        if args.left_first:
            left_hold = window(4.0, 12.0)
            right_hold = window(17.0, 25.0)
        else:
            right_hold = window(4.0, 12.0)
            left_hold = window(17.0, 25.0)
        final_stand = window(28.0, 30.0)
    else:
        right_hold = window(4.0, 8.0)
        left_hold = window(14.0, 18.0)
        final_stand = window(20.0, 22.0)

    def fraction(items: list[dict], predicate) -> float:
        return sum(bool(predicate(row)) for row in items) / len(items)

    if args.single_leg:
        support_is_right = args.single_leg == "right"
        single_support = fraction(
            hold,
            lambda row: (
                row["right_contact"] and not row["left_contact"]
                if support_is_right
                else row["left_contact"] and not row["right_contact"]
            ),
        )
        lifted_clearance = float(
            np.median(
                [row["left_foot_z_m"] if support_is_right else row["right_foot_z_m"] for row in hold]
            )
        )
        right_single = single_support if support_is_right else None
        left_single = single_support if not support_is_right else None
        right_clearance = lifted_clearance if support_is_right else None
        left_clearance = lifted_clearance if not support_is_right else None
        hold_rows = hold
    else:
        right_single = fraction(right_hold, lambda row: row["right_contact"] and not row["left_contact"])
        left_single = fraction(left_hold, lambda row: row["left_contact"] and not row["right_contact"])
        right_clearance = float(np.median([row["left_foot_z_m"] for row in right_hold]))
        left_clearance = float(np.median([row["right_foot_z_m"] for row in left_hold]))
        hold_rows = right_hold + left_hold
    final_double = fraction(final_stand, lambda row: row["left_contact"] and row["right_contact"])
    forbidden_samples = sum(row["forbidden_ground_contact"] for row in rows)
    max_tilt = max(row["trunk_tilt_deg"] for row in rows)
    displacement = math.hypot(rows[-1]["root_x_m"] - rows[0]["root_x_m"], rows[-1]["root_y_m"] - rows[0]["root_y_m"])
    median_abs_head_yaw = float(np.median([abs(row["head_yaw_rad"]) for row in hold_rows]))
    max_abs_head_yaw = max(abs(row["head_yaw_rad"]) for row in hold_rows)
    head_joint_ranges = {
        name: max(row[name] for row in hold_rows) - min(row[name] for row in hold_rows)
        for name in ("neck_pitch_rad", "head_pitch_rad", "head_yaw_rad", "head_roll_rad")
    }

    if args.single_leg:
        passed = (
            single_support >= 0.95
            and lifted_clearance >= 0.05
            and final_double >= 0.95
            and forbidden_samples == 0
            and max_tilt < 60.0
        )
    else:
        passed = (
            right_single >= 0.95
            and left_single >= 0.95
            and right_clearance >= 0.05
            and left_clearance >= 0.05
            and final_double >= 0.95
            and forbidden_samples == 0
            and max_tilt < 60.0
        )
    summary = {
        "policy": str(args.policy.resolve()),
        "source_branch": "pollen-robotics/microduck_rl flamingo",
        "head_frozen_at_home": args.freeze_head,
        "head_physics_locked_at_home": args.lock_head_physics,
        "ten_second_holds": args.long_hold,
        "left_first": args.left_first if args.long_hold else None,
        "single_leg": args.single_leg,
        "hold_seconds": args.hold_seconds if args.single_leg else None,
        "local_independent_pass": passed,
        "criteria": {
            "right_support_single_contact_fraction": right_single,
            "left_support_single_contact_fraction": left_single,
            "right_support_lifted_left_foot_median_z_m": right_clearance,
            "left_support_lifted_right_foot_median_z_m": left_clearance,
            "final_two_foot_contact_fraction": final_double,
            "forbidden_ground_contact_samples": forbidden_samples,
            "maximum_trunk_tilt_deg": max_tilt,
            "net_xy_displacement_m": displacement,
            "hold_median_abs_head_yaw_rad": median_abs_head_yaw,
            "hold_max_abs_head_yaw_rad": max_abs_head_yaw,
            "hold_head_joint_ranges_rad": head_joint_ranges,
        },
        "acceptance": {
            "single_support_fraction_min": 0.95,
            "lifted_foot_height_min_m": 0.05,
            "final_two_foot_fraction_min": 0.95,
            "forbidden_contact_samples_max": 0,
            "trunk_tilt_max_deg": 60.0,
        },
    }
    summary_path = output_dir / "official-flamingo-local-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    if args.save_distillation_data:
        if not args.freeze_head:
            raise ValueError("--save-distillation-data requires --freeze-head")
        np.savez_compressed(
            output_dir / "forward-head-distillation.npz",
            observations=np.asarray(distillation_observations, dtype=np.float32),
            actions=np.asarray(distillation_actions, dtype=np.float32),
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

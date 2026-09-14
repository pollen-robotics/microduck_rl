#!/usr/bin/env python3
"""Follow-me behavior demo: queued-footstep following with true opposite turns.

A CPU MuJoCo demo showing that a stock exported walking policy can be steered
to follow a moving target's ACTUAL PATH — turning where the leader turned,
rather than cutting the corner — using only the 3-DoF twist command slots. No
network is trained or modified here.

This is a SIMULATION demo, not robot autonomy: the leader is a scripted
kinematic mocap body and its position is read directly from the simulator.
There is no person detector and this has not been validated on hardware.

Usage (from the repository root):

    uv run python scripts/behavior_demos/follow_me/run_follow_me.py \\
        --policy /path/to/alpha_walking.onnx --no-render

Add ``--out DIR`` to write PNG frames, then encode them with ffmpeg:

    ffmpeg -framerate 50 -i DIR/f%05d.png -c:v libx264 -crf 18 \\
        -pix_fmt yuv420p -movflags +faststart follow-me.mp4
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort
from camera_tracking import HeadCameraTracker
from follow_motion import (
    CTRL_HZ,
    DEMO_SECONDS,
    TRAIL_DISTANCE,
    FollowController,
    FootstepTrail,
    animate_person,
    person_trajectory,
    wrap,
)

# Sibling modules (camera_tracking, follow_motion, video_overlay) are imported
# directly: Python puts this script's own directory on sys.path when it is run
# as a script, and the tests add it explicitly.

DEFAULT_SCENE = Path(__file__).resolve().parent / "scene_follow_me.xml"

# Pose the walking policy's actions are offsets from, and that joint
# observations are measured relative to. Must match DEFAULT_POSE in
# scripts/infer_policy.py and the STAND keyframe in the scene.
DEFAULT_POSE = np.array([
    0.0, -0.0873, -0.4579, -0.0049, 0.4530,
    0.3491, 0.3491, 0.0, 0.0,
    0.0, 0.0873, 0.4579, 0.0049, -0.4530,
], dtype=np.float32)

# Shipped walking action scale. 1.0 crosses the measured stability boundary on
# long forward legs.
ACTION_SCALE = 0.9

# Trunk height below which a step counts as fallen.
FALLEN_Z = 0.09

EXPECTED_ACTUATORS = 14
EXPECTED_COMMAND_DIM = 13

# The angular-velocity sensor is resolved BY NAME. Reading the last sensor by
# index silently picked up a different sensor during development and produced
# unstable, falsely-good results.
GYRO_SENSOR_NAMES = ("imu_gyro", "gyro", "imu_ang_vel", "angular-velocity")


def yaw_of_body(data, body_id):
    """World yaw of a body, taken from its rotation matrix's forward axis."""
    forward = data.xmat[body_id].reshape(3, 3)[:, 0]
    return math.atan2(float(forward[1]), float(forward[0]))


def resolve_gyro_adr(model):
    for sensor_name in GYRO_SENSOR_NAMES:
        sensor_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
        if sensor_id >= 0:
            return int(model.sensor_adr[sensor_id])
    raise RuntimeError(
        "no angular-velocity sensor found; expected one of "
        f"{GYRO_SENSOR_NAMES}")


def make_observation(data, qpos_idx, qvel_idx, trunk_id, gyro_adr,
                     last_action, command, command_dim):
    """Build the 61D actor observation: 48 proprioception + 13 command."""
    quat = data.xquat[trunk_id].astype(np.float32)
    w, xyz = quat[0], quat[1:4]
    gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    cross = np.cross(xyz, gravity_world) * 2.0
    gravity = gravity_world - w * cross + np.cross(xyz, cross)
    gyro = data.sensordata[gyro_adr:gyro_adr + 3].astype(np.float32)
    joint_pos = data.qpos[qpos_idx].astype(np.float32) - DEFAULT_POSE
    joint_vel = data.qvel[qvel_idx].astype(np.float32)
    # Only the twist slots are driven; head_pose and body_pose stay zeroed, as
    # the walking policy family requires the full command block to be present.
    policy_command = np.zeros(command_dim, dtype=np.float32)
    policy_command[:3] = command
    return np.concatenate([
        gyro, gravity, joint_pos, joint_vel, last_action, policy_command
    ]).astype(np.float32)


def phase_summary(records):
    result = {}
    for phase in sorted({r["phase"] for r in records}):
        rows = [r for r in records if r["phase"] == phase]
        result[phase] = {
            "samples": len(rows),
            "follow_rmse_m": math.sqrt(
                sum(r["follow_error_m"] ** 2 for r in rows) / len(rows)),
            "follow_max_m": max(r["follow_error_m"] for r in rows),
            "person_range_mean_m": sum(r["person_range_m"] for r in rows) / len(rows),
            "person_range_min_m": min(r["person_range_m"] for r in rows),
            "person_range_max_m": max(r["person_range_m"] for r in rows),
            "min_trunk_z_m": min(r["trunk_z_m"] for r in rows),
            "camera_visible_pct": 100.0 * sum(r["visible"] for r in rows) / len(rows),
        }
    return result


def turn_yaw_delta(records, phase):
    """Signed yaw change measured on the trunk during a duck command phase."""
    rows = [r for r in records if r["command_phase"] == phase]
    if not rows:
        return float("nan")
    return rows[-1]["duck_yaw_deg"] - rows[0]["duck_yaw_deg"]


def build_summary(records, args, total_steps, frames, tracker,
                  leader_phase_times, command_phase_times):
    errors = [r["follow_error_m"] for r in records]
    return {
        "duration_s": args.seconds,
        "control_steps": total_steps,
        "frames": frames,
        "trail_distance_m": TRAIL_DISTANCE,
        "target_semantics":
            "leader world-space path point TRAIL_DISTANCE metres earlier",
        "follow_rmse_m": math.sqrt(sum(e * e for e in errors) / len(errors)),
        "follow_mean_m": sum(errors) / len(errors),
        "follow_max_m": max(errors),
        "person_range_mean_m": sum(r["person_range_m"] for r in records) / len(records),
        "person_range_min_m": min(r["person_range_m"] for r in records),
        "person_range_max_m": max(r["person_range_m"] for r in records),
        "min_trunk_z_m": min(r["trunk_z_m"] for r in records),
        "final_trunk_z_m": records[-1]["trunk_z_m"],
        "fallen_steps": sum(r["trunk_z_m"] < FALLEN_Z for r in records),
        "camera_visible_steps": total_steps - tracker.lost_steps,
        "camera_lost_steps": tracker.lost_steps,
        "camera_visible_pct": 100.0 * (total_steps - tracker.lost_steps) / total_steps,
        "camera_rms_off_axis_deg": math.degrees(tracker.rms_off_axis),
        "camera_max_off_axis_deg": math.degrees(tracker.max_off_axis),
        "leader_phase_first_seen_s": leader_phase_times,
        "duck_command_phase_first_seen_s": command_phase_times,
        "left_turn_delay_s": (
            command_phase_times.get("LEFT TURN", float("nan"))
            - leader_phase_times.get("LEFT TURN", float("nan"))),
        "right_turn_delay_s": (
            command_phase_times.get("RIGHT TURN", float("nan"))
            - leader_phase_times.get("RIGHT TURN", float("nan"))),
        "backward_delay_s": (
            command_phase_times.get("BACKWARD", float("nan"))
            - leader_phase_times.get("BACKWARD", float("nan"))),
        "leader_left_turn_yaw_delta_deg": 90.0,
        "leader_right_turn_yaw_delta_deg": -90.0,
        "duck_left_turn_yaw_delta_deg": turn_yaw_delta(records, "LEFT TURN"),
        "duck_right_turn_yaw_delta_deg": turn_yaw_delta(records, "RIGHT TURN"),
        "phases": phase_summary(records),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--policy", required=True,
        help="path to an exported walking ONNX policy (e.g. alpha_walking.onnx)")
    parser.add_argument("--xml", default=str(DEFAULT_SCENE),
                        help="scene XML (default: the demo scene next to this script)")
    parser.add_argument("--seconds", type=float, default=DEMO_SECONDS)
    parser.add_argument("--out", default=None,
                        help="directory for PNG frames; omit to skip rendering")
    parser.add_argument("--metrics", default=None,
                        help="path to write the metrics JSON")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--no-render", action="store_true",
                        help="metrics only, no frames (no Pillow/display needed)")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress the per-step progress log")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    render = not args.no_render and args.out is not None
    if render:
        os.makedirs(args.out, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    if model.nu != EXPECTED_ACTUATORS:
        raise RuntimeError(
            f"expected {EXPECTED_ACTUATORS} policy actuators, got {model.nu}")
    qpos_idx = np.array([
        int(model.jnt_qposadr[model.actuator_trnid[i, 0]]) for i in range(model.nu)])
    qvel_idx = np.array([
        int(model.jnt_dofadr[model.actuator_trnid[i, 0]]) for i in range(model.nu)])
    for index, qpos_address in enumerate(qpos_idx):
        data.qpos[qpos_address] = DEFAULT_POSE[index]
    data.ctrl[:] = DEFAULT_POSE

    trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    person_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "person")
    person_mocap = int(model.body_mocapid[person_id])
    trail_target_mocap = int(model.body_mocapid[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trail_target")])
    gyro_adr = resolve_gyro_adr(model)

    initial_person = person_trajectory(0.0)
    animate_person(model, data, initial_person, 0.0)
    mujoco.mj_forward(model, data)

    session = ort.InferenceSession(
        args.policy, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    observation_dim = int(session.get_inputs()[0].shape[1])
    command_dim = observation_dim - (3 + 3 + model.nu * 3)
    if command_dim != EXPECTED_COMMAND_DIM:
        raise RuntimeError(
            f"expected {EXPECTED_COMMAND_DIM} command dimensions, got {command_dim}; "
            f"is {args.policy} a walking policy with the 61D observation layout?")

    sim_dt = model.opt.timestep
    decimation = max(1, round((1.0 / CTRL_HZ) / sim_dt))
    total_steps = int(args.seconds * CTRL_HZ)
    frame_every = max(1, round(CTRL_HZ / args.fps))
    controller = FollowController()
    footsteps = FootstepTrail(initial_person)

    if render:
        from video_overlay import PIP_H, PIP_W, compose
    else:
        PIP_W, PIP_H = 225, 165
    tracker = HeadCameraTracker(model, data, qpos_idx, trunk_id, (PIP_W, PIP_H))

    last_action = np.zeros(model.nu, dtype=np.float32)
    previous_yaw = yaw_of_body(data, trunk_id)
    min_height = float(data.xpos[trunk_id, 2])
    records = []
    frames = 0

    if render:
        renderer = mujoco.Renderer(model, height=args.height, width=args.width)
        pip_renderer = mujoco.Renderer(model, height=PIP_H, width=PIP_W)
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.distance = 1.85
        camera.elevation = -20
        # Rear three-quarter view keeps actor-relative left and right readable.
        camera.azimuth = -45
        camera_lookat = np.array([0.4, 0.0, 0.15], dtype=np.float64)

    if not args.quiet:
        print(f"follow-me: {total_steps} steps, trail={TRAIL_DISTANCE:.2f} m, "
              f"decimation={decimation}, render={render}")

    last_phase = None
    last_command_phase = None
    leader_phase_times = {}
    command_phase_times = {}

    for step in range(total_steps):
        t = step / CTRL_HZ
        person = person_trajectory(t)
        animate_person(model, data, person, t)
        mujoco.mj_forward(model, data)
        duck_pos_before = data.xpos[trunk_id].copy()
        duck_yaw_before = yaw_of_body(data, trunk_id)
        trail = footsteps.update(person)
        data.mocap_pos[trail_target_mocap] = np.array(
            [trail.pos[0], trail.pos[1], 0.012])
        command, follow = controller.update(
            person, trail, duck_pos_before, duck_yaw_before)
        command_phase = follow["replay_phase"] if person.moving else "STOPPED"

        observation = make_observation(
            data, qpos_idx, qvel_idx, trunk_id, gyro_adr,
            last_action, command, command_dim)
        action = session.run(
            [output_name], {input_name: observation.reshape(1, -1)}
        )[0].squeeze(0).astype(np.float32)
        last_action = action.copy()
        data.ctrl[:] = DEFAULT_POSE + ACTION_SCALE * action
        for _ in range(decimation):
            mujoco.mj_step(model, data)

        # Advance the leader to the END of the control interval so the frame
        # and the recorded metrics describe the same instant.
        display_t = min(t + 1.0 / CTRL_HZ, args.seconds)
        display_person = person_trajectory(display_t)
        animate_person(model, data, display_person, display_t)
        mujoco.mj_forward(model, data)
        duck_pos = data.xpos[trunk_id].copy()
        duck_yaw = yaw_of_body(data, trunk_id)
        yaw_rate = math.degrees(wrap(duck_yaw - previous_yaw)) * CTRL_HZ
        previous_yaw = duck_yaw
        min_height = min(min_height, float(duck_pos[2]))
        camera_state = tracker.update(data)

        target_pos = follow["target_pos"]
        follow_error = float(np.linalg.norm(target_pos - duck_pos[:2]))
        person_range = float(np.linalg.norm(
            data.mocap_pos[person_mocap, :2] - duck_pos[:2]))
        follow.update({"error": follow_error, "person_range": person_range})
        records.append({
            "t": display_t,
            "phase": display_person.phase,
            "trail_phase": trail.phase,
            "command_phase": command_phase,
            "trail_target_xy": trail.pos.tolist(),
            "spatial_lag_m": follow["spatial_lag"],
            "follow_error_m": follow_error,
            "person_range_m": person_range,
            "yaw_error_deg": math.degrees(follow["yaw_error"]),
            "leader_yaw_deg": math.degrees(display_person.yaw),
            "duck_yaw_deg": math.degrees(duck_yaw),
            "trunk_z_m": float(duck_pos[2]),
            "visible": bool(camera_state["visible"]),
            "off_axis_deg": math.degrees(camera_state["off_axis"]),
            "command": command.tolist(),
        })

        leader_phase_times.setdefault(display_person.phase, display_t)
        command_phase_times.setdefault(command_phase, display_t)
        if not args.quiet:
            if display_person.phase != last_phase:
                print(f"  t={display_t:5.2f}s LEADER -> {display_person.phase:9s} "
                      f"duck_replays={command_phase:9s}")
                last_phase = display_person.phase
            if command_phase != last_command_phase:
                print(f"  t={display_t:5.2f}s DUCK   -> {command_phase:9s} "
                      f"at trail=({trail.pos[0]:+.3f},{trail.pos[1]:+.3f})")
                last_command_phase = command_phase

        if render and step % frame_every == 0:
            center = 0.5 * (duck_pos + data.mocap_pos[person_mocap])
            center[2] = 0.15
            camera_lookat += 0.08 * (center - camera_lookat)
            camera.lookat[:] = camera_lookat
            renderer.update_scene(tracker.gaze_data, camera=camera)
            pip_renderer.update_scene(tracker.gaze_data, camera=tracker.camera_id)
            image = compose(
                renderer.render(), pip_renderer.render(), t=display_t,
                total_seconds=args.seconds, person=display_person,
                duck_pos=duck_pos, duck_yaw=duck_yaw, follow=follow,
                command=command, camera=camera_state, yaw_rate=yaw_rate,
                min_height=min_height)
            # Imported lazily: the demo runs headless without imageio.
            import imageio.v2 as imageio
            imageio.imwrite(Path(args.out) / f"f{frames:05d}.png", np.asarray(image))
            frames += 1

    summary = build_summary(records, args, total_steps, frames, tracker,
                            leader_phase_times, command_phase_times)
    if args.metrics:
        Path(args.metrics).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()

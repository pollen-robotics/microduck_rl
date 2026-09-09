#!/usr/bin/env python3
"""Follow-me-among-others: color-selective crowd following in CPU MuJoCo.

A behavior demo, not a trained policy and not robot autonomy. It drives the
*stock* exported walking policy with a twist command produced by a geometric
controller, so nothing here needs a GPU, a training run or new weights.

Five pedestrians in blue, green, red, yellow and purple walk continuously on
independent routes. The robot repeatedly searches the crowd through its head
camera, acquires the requested shirt color, follows that person's queued
footsteps 0.55 m back along their own path, and stops:

    BLUE -> GREEN -> RED -> BLUE
    SEARCH -> FOUND -> FOLLOW -> STOP   (once per selection)

Color recognition is an explicit MuJoCo semantic proxy (actor identity plus
camera-frustum and occlusion tests), NOT an RGB classifier -- see camera.py.

Usage (from the repo root, with your own exported walking policy):

    uv run --with imageio --with pillow \
        scripts/behavior_demos/follow_me_among_others/run_demo.py \
        --policy /path/to/walking.onnx --no-render

Drop --no-render (and pass --out) to also write HUD frames as PNGs.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

if __package__ in (None, ""):
    # Allow `python scripts/behavior_demos/follow_me_among_others/run_demo.py`
    # as well as `python -m ...`, without requiring an install step.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from follow_me_among_others import metrics as metrics_module
    from follow_me_among_others.camera import CrowdCameraSearch
    from follow_me_among_others.crowd import (
        COLORS,
        CTRL_HZ,
        TARGET_SEQUENCE,
        TRAIL_DISTANCE,
        CrowdFollowController,
        FootstepTrail,
        SearchFollowStateMachine,
        animate_crowd,
        crowd_trajectory,
    )
else:
    from . import metrics as metrics_module
    from .camera import CrowdCameraSearch
    from .crowd import (
        COLORS,
        CTRL_HZ,
        TARGET_SEQUENCE,
        TRAIL_DISTANCE,
        CrowdFollowController,
        FootstepTrail,
        SearchFollowStateMachine,
        animate_crowd,
        crowd_trajectory,
    )

# Repo-relative default; overridable with --xml. No absolute paths anywhere.
DEFAULT_XML = "src/mjlab_microduck/robot/microduck/scene_follow_me_among_others.xml"

# STAND2 pose, identical to HOME_FRAME in microduck_constants.py and to the
# STAND keyframe in the scene. Actions are offsets from this pose.
DEFAULT_POSE = np.array(
    [
        0.0,
        -0.0873,
        -0.4579,
        -0.0049,
        0.4530,
        0.3491,
        0.3491,
        0.0,
        0.0,
        0.0,
        0.0873,
        0.4579,
        0.0049,
        -0.4530,
    ],
    dtype=np.float32,
)

# Observation contract shared by every policy in this repo (AGENTS.md):
# 48 proprioception + 13 command = 61. Unused command slots are zero-padded,
# never dropped.
EXPECTED_OBS_DIM = 61
EXPECTED_COMMAND_DIM = 13

# Matches the action scale used by infer_policy.py for the walking policy.
ACTION_SCALE = 0.9

# Angular-velocity sensor names accepted, in preference order.
GYRO_SENSOR_NAMES = ("imu_ang_vel", "gyro", "angular-velocity")


def yaw_of_body(data, body_id: int) -> float:
    forward = data.xmat[body_id].reshape(3, 3)[:, 0]
    return math.atan2(float(forward[1]), float(forward[0]))


def make_observation(
    data,
    model,
    qpos_idx,
    qvel_idx,
    trunk_id,
    gyro_adr,
    last_action,
    command,
    command_dim,
):
    """Assemble the 61-D actor observation."""
    quat = data.xquat[trunk_id].astype(np.float32)
    w, xyz = quat[0], quat[1:4]
    gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    cross = np.cross(xyz, gravity_world) * 2.0
    gravity = gravity_world - w * cross + np.cross(xyz, cross)
    gyro = data.sensordata[gyro_adr : gyro_adr + 3].astype(np.float32)
    joint_pos = data.qpos[qpos_idx].astype(np.float32) - DEFAULT_POSE
    joint_vel = data.qvel[qvel_idx].astype(np.float32)
    policy_command = np.zeros(command_dim, dtype=np.float32)
    # Only the twist slots are driven; head_pose and body_pose stay zeroed.
    policy_command[:3] = command
    observation = np.concatenate(
        [gyro, gravity, joint_pos, joint_vel, last_action, policy_command]
    ).astype(np.float32)
    if observation.shape != (EXPECTED_OBS_DIM,):
        raise RuntimeError(
            f"expected {EXPECTED_OBS_DIM}-D observation, got {observation.shape}"
        )
    return observation


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--policy",
        required=True,
        help="Path to an exported walking ONNX policy (scripts/export.py).",
    )
    parser.add_argument("--xml", default=DEFAULT_XML, help="Scene MJCF.")
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument(
        "--targets",
        default=",".join(TARGET_SEQUENCE),
        help="Comma-separated shirt colors to follow in order.",
    )
    parser.add_argument("--metrics", default=None, help="Write the metrics JSON here.")
    parser.add_argument(
        "--out", default=None, help="Directory for rendered PNG frames."
    )
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Skip rendering; metrics and gates are unaffected.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    sequence = tuple(
        part.strip().upper() for part in args.targets.split(",") if part.strip()
    )
    render = not args.no_render
    if render and not args.out:
        raise SystemExit("--out is required unless --no-render is given")
    if render:
        os.makedirs(args.out, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    if model.nu != 14:
        raise RuntimeError(f"expected 14 policy actuators, got {model.nu}")

    qpos_idx = np.array(
        [int(model.jnt_qposadr[model.actuator_trnid[i, 0]]) for i in range(model.nu)]
    )
    qvel_idx = np.array(
        [int(model.jnt_dofadr[model.actuator_trnid[i, 0]]) for i in range(model.nu)]
    )
    for index, address in enumerate(qpos_idx):
        data.qpos[address] = DEFAULT_POSE[index]
    data.ctrl[:] = DEFAULT_POSE

    trunk_id = model.body("trunk_base").id
    people_mocap = {
        color: int(model.body_mocapid[model.body(f"person_{color.lower()}").id])
        for color in COLORS
    }
    target_mocap = int(model.body_mocapid[model.body("trail_target").id])

    gyro_id = -1
    for sensor_name in GYRO_SENSOR_NAMES:
        gyro_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
        if gyro_id >= 0:
            break
    if gyro_id < 0:
        raise RuntimeError("no angular-velocity sensor found in model")
    gyro_adr = int(model.sensor_adr[gyro_id])

    initial_crowd = crowd_trajectory(0.0)
    animate_crowd(model, data, initial_crowd, 0.0)
    mujoco.mj_forward(model, data)
    trails = {color: FootstepTrail(initial_crowd[color]) for color in COLORS}

    session = ort.InferenceSession(args.policy, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    observation_dim = int(session.get_inputs()[0].shape[1])
    command_dim = observation_dim - (3 + 3 + model.nu * 3)
    if observation_dim != EXPECTED_OBS_DIM or command_dim != EXPECTED_COMMAND_DIM:
        raise RuntimeError(
            f"policy shape mismatch: observation={observation_dim}, "
            f"command={command_dim} (expected {EXPECTED_OBS_DIM}/"
            f"{EXPECTED_COMMAND_DIM}); export with scripts/export.py"
        )

    decimation = max(1, round((1.0 / CTRL_HZ) / model.opt.timestep))
    total_steps = int(args.seconds * CTRL_HZ)
    frame_every = max(1, round(CTRL_HZ / args.fps))
    controller = CrowdFollowController()
    machine = SearchFollowStateMachine(sequence)
    camera_search = CrowdCameraSearch(model, data, qpos_idx, trunk_id)
    last_action = np.zeros(model.nu, dtype=np.float32)
    records: list[dict] = []
    transitions: list[dict] = []
    frames = 0
    min_height = float(data.xpos[trunk_id, 2])

    renderer = pip_renderer = compose = None
    if render:
        import imageio.v2 as imageio

        from follow_me_among_others.overlay import PIP_H, PIP_W, compose

        renderer = mujoco.Renderer(model, height=args.height, width=args.width)
        pip_renderer = mujoco.Renderer(model, height=PIP_H, width=PIP_W)
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.distance = 2.85
        camera.elevation = -27
        camera.azimuth = -48
        camera_lookat = np.array([0.75, 0.0, 0.16], dtype=np.float64)

    if not args.quiet:
        print(
            f"follow-me-among-others: {total_steps} steps, "
            f"targets={'->'.join(sequence)}, lag={TRAIL_DISTANCE:.2f} m, "
            f"decimation={decimation}, render={render}"
        )

    for step in range(total_steps):
        t = step / CTRL_HZ
        crowd = crowd_trajectory(t)
        animate_crowd(model, data, crowd, t)
        mujoco.mj_forward(model, data)
        trail_states = {color: trails[color].update(crowd[color]) for color in COLORS}

        state = machine.state if not machine.done else "DONE"
        target = machine.target
        selection = min(machine.index + 1, len(sequence))
        state_elapsed = t - machine.state_since
        duck_pos_before = data.xpos[trunk_id].copy()
        duck_yaw_before = yaw_of_body(data, trunk_id)
        active_trail = trail_states[target]

        command, follow = controller.update(
            state == "FOLLOW", active_trail, duck_pos_before, duck_yaw_before
        )
        if state == "FOLLOW":
            data.mocap_pos[target_mocap] = np.array(
                [active_trail.pos[0], active_trail.pos[1], 0.012]
            )
        else:
            # Park the marker under the floor when it means nothing.
            data.mocap_pos[target_mocap] = np.array([0.0, 0.0, -0.10])

        observation = make_observation(
            data,
            model,
            qpos_idx,
            qvel_idx,
            trunk_id,
            gyro_adr,
            last_action,
            command,
            command_dim,
        )
        action = (
            session.run([output_name], {input_name: observation.reshape(1, -1)})[0]
            .squeeze(0)
            .astype(np.float32)
        )
        last_action = action.copy()
        data.ctrl[:] = DEFAULT_POSE + ACTION_SCALE * action
        for _ in range(decimation):
            mujoco.mj_step(model, data)

        display_t = min(t + 1.0 / CTRL_HZ, args.seconds)
        display_crowd = crowd_trajectory(display_t)
        animate_crowd(model, data, display_crowd, display_t)
        mujoco.mj_forward(model, data)
        duck_pos = data.xpos[trunk_id].copy()
        duck_yaw = yaw_of_body(data, trunk_id)
        min_height = min(min_height, float(duck_pos[2]))

        camera_state = camera_search.update(
            data,
            target_color=target,
            mode=state,
            mode_elapsed=state_elapsed,
            duck_yaw=duck_yaw,
        )
        follow_error = float(np.linalg.norm(active_trail.pos - duck_pos[:2]))
        follow["error"] = follow_error
        person_pos = data.mocap_pos[people_mocap[target], :2]
        records.append(
            {
                "t": display_t,
                "state": state,
                "selection": selection,
                "target": target,
                "state_elapsed_s": state_elapsed,
                "target_visible": bool(camera_state["target_visible"]),
                "target_off_axis_deg": math.degrees(camera_state["target_off_axis"]),
                "visible_colors": camera_state["visible_colors"],
                "trail_target_xy": active_trail.pos.tolist(),
                "follow_error_m": follow_error,
                "person_range_m": float(np.linalg.norm(person_pos - duck_pos[:2])),
                "yaw_error_deg": math.degrees(follow["yaw_error"]),
                "duck_yaw_deg": math.degrees(duck_yaw),
                "duck_xy": duck_pos[:2].tolist(),
                "trunk_z_m": float(duck_pos[2]),
                "command": command.tolist(),
            }
        )

        next_state, next_target, changed = machine.update(display_t, camera_state)
        if changed:
            transitions.append(
                {
                    "t": display_t,
                    "from": state,
                    "to": next_state,
                    "target": next_target,
                    "selection": min(machine.index + 1, len(sequence)),
                }
            )
            if not args.quiet:
                print(
                    f"  t={display_t:5.2f}s {state:10s} -> {next_state:10s} "
                    f"target={next_target}"
                )

        if render and step % frame_every == 0:
            all_positions = np.vstack(
                [duck_pos[:2], *[display_crowd[color].pos for color in COLORS]]
            )
            center = np.array(
                [
                    float(np.mean(all_positions[:, 0])),
                    float(np.mean(all_positions[:, 1])),
                    0.16,
                ]
            )
            camera_lookat += 0.05 * (center - camera_lookat)
            camera.lookat[:] = camera_lookat
            # Render the gaze clone: it carries the same physics plus the head
            # pose and the stabilized rig.
            renderer.update_scene(camera_search.gaze_data, camera=camera)
            pip_renderer.update_scene(
                camera_search.gaze_data, camera=camera_search.camera_id
            )
            image = compose(
                renderer.render(),
                pip_renderer.render(),
                t=display_t,
                total_seconds=args.seconds,
                state=state,
                state_elapsed=state_elapsed,
                selection=selection,
                target=target,
                sequence=sequence,
                duck_pos=duck_pos,
                follow=follow,
                command=command,
                camera=camera_state,
                min_height=min_height,
                completed_cycles=len(machine.cycles),
            )
            imageio.imwrite(Path(args.out) / f"f{frames:05d}.png", np.asarray(image))
            frames += 1

    if not machine.done:
        raise RuntimeError(
            f"sequence incomplete after {args.seconds:.1f}s: "
            f"selection={machine.index + 1}, state={machine.state}"
        )

    summary = metrics_module.summarize(
        records=records,
        transitions=transitions,
        cycles=machine.cycles,
        sequence=sequence,
        duration_s=args.seconds,
        control_steps=total_steps,
        frames=frames,
        trail_distance_m=TRAIL_DISTANCE,
        camera_stats={
            "camera_target_visible_steps": camera_search.target_visible_steps,
            "camera_search_steps": camera_search.search_steps,
            "camera_search_target_visible_steps": (
                camera_search.search_target_visible_steps
            ),
        },
    )
    if args.metrics:
        Path(args.metrics).write_text(json.dumps(summary, indent=2) + "\n")

    failures = metrics_module.check_gates(summary, sequence)
    if not args.quiet:
        print(json.dumps(summary, indent=2))
    if failures:
        for failure in failures:
            print(f"ACCEPTANCE FAILURE: {failure}", file=sys.stderr)
        return 1
    if not args.quiet:
        print(f"all acceptance gates passed ({len(sequence)} selections)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

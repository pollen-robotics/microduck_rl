#!/usr/bin/env python3
"""Move aside from an approaching person without losing sight of them.

A CPU MuJoCo behavior demo. No new policy is trained: a stock exported walking
policy is driven by a velocity command produced by a small state machine, and an
independent kinematic gaze layer keeps the person centred in the head camera.

    IDLE -> RETREAT -> TURN -> CLEAR -> DONE

Run it headless (metrics only) or render frames:

    python scripts/behavior_demos/move_away/run_demo.py --policy path/to/walking.onnx
    python scripts/behavior_demos/move_away/run_demo.py \
        --policy path/to/walking.onnx --render-dir /tmp/move_away_frames

Rendering additionally needs `imageio` and `Pillow`, which the demo imports
lazily so the headless path stays dependency-light.

See README.md in this directory for the measured constants and the limitations
this demo does NOT claim to have solved.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

if __package__ in (None, ""):  # allow `python scripts/behavior_demos/move_away/run_demo.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from move_away.controller import MoveAwayController
    from move_away.gaze import GazeState
    from move_away.perception import PersonTracker, optical_frame_quat
    from move_away.runtime import (
        CTRL_HZ,
        DEFAULT_POSE,
        GYRO_SENSOR,
        HEAD_PITCH_ACT,
        HEAD_YAW_ACT,
        SCENE_XML,
        actuator_indices,
        build_observation,
        quat_rotate_inverse,
        sensor_address,
    )
else:
    from .controller import MoveAwayController
    from .gaze import GazeState
    from .perception import PersonTracker, optical_frame_quat
    from .runtime import (
        CTRL_HZ,
        DEFAULT_POSE,
        GYRO_SENSOR,
        HEAD_PITCH_ACT,
        HEAD_YAW_ACT,
        SCENE_XML,
        actuator_indices,
        build_observation,
        quat_rotate_inverse,
        sensor_address,
    )

PIP_W, PIP_H = 225, 165
NOMINAL_TRUNK_Z = 0.116
FALLEN_TRUNK_Z = 0.09


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--policy",
        required=True,
        type=Path,
        help="Path to an exported walking policy (.onnx). Required: no policy "
        "is bundled with the repository.",
    )
    parser.add_argument("--scene", type=Path, default=SCENE_XML, help="Scene XML to load.")
    parser.add_argument("--seconds", type=float, default=22.0, help="Simulated duration.")
    parser.add_argument("--person-speed", type=float, default=0.12, help="Person speed (m/s).")
    parser.add_argument("--person-x0", type=float, default=1.60, help="Person start x (m).")
    parser.add_argument("--warmup", type=float, default=1.5, help="Seconds before the person moves.")
    parser.add_argument("--turn-sign", type=float, default=1.0, help="+1 turns left, -1 right.")
    parser.add_argument("--render-dir", type=Path, default=None, help="Write PNG frames here.")
    parser.add_argument("--fps", type=int, default=50, help="Frame rate when rendering.")
    parser.add_argument("--width", type=int, default=960, help="Render width.")
    parser.add_argument("--height", type=int, default=640, help="Render height.")
    parser.add_argument("--json", action="store_true", help="Print the summary as JSON.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-second progress.")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict:
    """Run the behavior and return a summary dict of measured outcomes."""
    import onnxruntime as ort

    if not args.policy.is_file():
        raise SystemExit(f"policy not found: {args.policy}")
    if not args.scene.is_file():
        raise SystemExit(f"scene not found: {args.scene}")

    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)
    gaze_data = mujoco.MjData(model)  # isolated kinematic head-tracking state
    mujoco.mj_resetData(model, data)

    nu = model.nu
    qpos_idx, qvel_idx = actuator_indices(model)
    gyro_adr = sensor_address(model, GYRO_SENSOR)
    trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    head_yaw_joint = int(model.actuator_trnid[HEAD_YAW_ACT, 0])
    head_pitch_joint = int(model.actuator_trnid[HEAD_PITCH_ACT, 0])

    # Fix the exported camera's optical frame in memory only (never on disk).
    head_cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
    model.cam_quat[head_cam] = optical_frame_quat()

    tracker = PersonTracker(model, aspect=PIP_W / PIP_H)

    for slot, value in enumerate(qpos_idx):
        data.qpos[value] = DEFAULT_POSE[slot]
    data.ctrl[:] = DEFAULT_POSE[:nu]
    mujoco.mj_forward(model, data)

    session = ort.InferenceSession(str(args.policy), providers=["CPUExecutionProvider"])
    in_name = session.get_inputs()[0].name
    out_name = session.get_outputs()[0].name
    obs_dim = int(session.get_inputs()[0].shape[1])
    command_dim = obs_dim - (3 + 3 + nu * 3)
    if command_dim < 3:
        raise SystemExit(
            f"policy expects a {obs_dim}D observation, which leaves no room for a "
            f"3D twist command with {nu} actuated joints"
        )

    controller = MoveAwayController(ctrl_hz=CTRL_HZ, turn_sign=args.turn_sign)
    gaze = GazeState(
        yaw=float(DEFAULT_POSE[HEAD_YAW_ACT]),
        pitch=float(DEFAULT_POSE[HEAD_PITCH_ACT]),
        yaw_limits=tuple(model.jnt_range[head_yaw_joint]),
        pitch_limits=tuple(model.jnt_range[head_pitch_joint]),
        tau=1.0 / CTRL_HZ,
        dt=1.0 / CTRL_HZ,
    )

    last_action = np.zeros(nu, dtype=np.float32)
    decimation = max(1, round((1.0 / CTRL_HZ) / model.opt.timestep))
    total_steps = int(args.seconds * CTRL_HZ)
    frame_every = max(1, round(CTRL_HZ / args.fps))

    renderer = pip_renderer = camera = None
    if args.render_dir is not None:
        renderer, pip_renderer, camera = _make_renderers(model, args)

    lost_steps = 0
    max_off_axis = 0.0
    min_trunk_z = float("inf")
    transitions: list[dict] = []
    frames = 0

    for step in range(total_steps):
        t = step / CTRL_HZ

        # The person walks straight in from +x and keeps coming. The robot's
        # forward axis (and therefore its head camera) faces +x.
        moving_for = max(0.0, t - args.warmup)
        data.mocap_pos[tracker.person_mocap_id][0] = args.person_x0 - args.person_speed * moving_for

        # --- isolated kinematic gaze layer --------------------------------
        # Copy the physical state, pose the head THERE, and use that copy for
        # perception and rendering only. The walking policy keeps its original
        # head dynamics, which is what keeps it upright.
        mujoco.mj_copyData(gaze_data, model, data)
        gaze_data.qpos[qpos_idx[HEAD_PITCH_ACT]] = gaze.pitch
        gaze_data.qpos[qpos_idx[HEAD_YAW_ACT]] = gaze.yaw
        mujoco.mj_forward(model, gaze_data)

        pre = tracker.look(gaze_data)
        gaze.step(pre["bearing"], pre["elevation"])
        gaze_data.qpos[qpos_idx[HEAD_PITCH_ACT]] = gaze.pitch
        gaze_data.qpos[qpos_idx[HEAD_YAW_ACT]] = gaze.yaw
        mujoco.mj_forward(model, gaze_data)

        view = tracker.look(gaze_data)
        if not view["visible"]:
            lost_steps += 1
        max_off_axis = max(max_off_axis, view["off_axis"])

        # --- heading: ABSOLUTE, never integrated ---------------------------
        forward = data.xmat[trunk].reshape(3, 3)[:, 0]
        yaw = math.atan2(forward[1], forward[0])

        planar = data.mocap_pos[tracker.person_mocap_id][:2] - data.xpos[trunk][:2]
        distance = float(np.linalg.norm(planar))

        previous_state = controller.state
        twist = controller.update(yaw, distance, view["visible"])
        if controller.state != previous_state:
            transitions.append(
                {
                    "t": round(t, 2),
                    "from": previous_state,
                    "to": controller.state,
                    "distance": round(distance, 3),
                    "turned_deg": round(math.degrees(controller.turned), 1),
                }
            )
            if not args.quiet:
                print(
                    f"  t={t:5.2f}s  {previous_state} -> {controller.state}"
                    f"  (dist={distance:.3f} turned={math.degrees(controller.turned):+.1f} deg)"
                )

        gyro = data.sensordata[gyro_adr : gyro_adr + 3].astype(np.float32)
        quat = data.xquat[trunk].astype(np.float32)
        gravity = quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0], dtype=np.float32))
        joint_pos = data.qpos[qpos_idx].astype(np.float32) - DEFAULT_POSE[:nu]
        joint_vel = data.qvel[qvel_idx].astype(np.float32)

        obs = build_observation(
            gyro,
            gravity,
            joint_pos,
            joint_vel,
            last_action,
            np.asarray(twist, dtype=np.float32),
            command_dim,
        )
        action = session.run([out_name], {in_name: obs.reshape(1, -1)})[0]
        action = action.squeeze(0).astype(np.float32)
        last_action = action.copy()
        data.ctrl[:] = DEFAULT_POSE[:nu] + action

        for _ in range(decimation):
            mujoco.mj_step(model, data)
            min_trunk_z = min(min_trunk_z, float(data.xpos[trunk][2]))

        if renderer is not None and step % frame_every == 0:
            _render_frame(
                model,
                data,
                gaze_data,
                gaze,
                qpos_idx,
                renderer,
                pip_renderer,
                camera,
                trunk,
                tracker,
                head_cam,
                controller,
                view,
                twist,
                t,
                distance,
                yaw,
                args,
                frames,
            )
            frames += 1

        if not args.quiet and step % int(CTRL_HZ) == 0:
            pos = data.xpos[trunk]
            print(
                f"  t={t:5.1f}s {controller.state:7s} d={distance:.3f} "
                f"turned={math.degrees(controller.turned):+7.1f} deg "
                f"pos=({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:.3f}) "
                f"cmd=(vx={twist[0]:+.2f}, wz={twist[2]:+.2f}) "
                f"sees={'Y' if view['visible'] else 'N'}"
            )

    position = data.xpos[trunk]
    return {
        "final_state": controller.state,
        "turned_deg": round(math.degrees(controller.turned), 1),
        "final_position": [round(float(v), 3) for v in position],
        "trunk_z": round(float(position[2]), 3),
        "min_trunk_z": round(min_trunk_z, 3),
        "upright": bool(position[2] > FALLEN_TRUNK_Z),
        "control_steps": total_steps,
        "visible_steps": total_steps - lost_steps,
        "lost_steps": lost_steps,
        "max_off_axis_deg": round(math.degrees(max_off_axis), 1),
        "frames": frames,
        "transitions": transitions,
    }


def _make_renderers(model: mujoco.MjModel, args: argparse.Namespace):
    """Build the main renderer, the duck's-eye PiP renderer and the camera.

    A separate ``Renderer`` is needed for the inset because each one caches its
    framebuffer size.
    """
    args.render_dir.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    pip_renderer = mujoco.Renderer(model, height=PIP_H, width=PIP_W)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.distance = 1.9
    camera.elevation = -16
    camera.azimuth = 128
    return renderer, pip_renderer, camera


def _render_frame(
    model,
    data,
    gaze_data,
    gaze,
    qpos_idx,
    renderer,
    pip_renderer,
    camera,
    trunk,
    tracker,
    head_cam,
    controller,
    view,
    twist,
    t,
    distance,
    yaw,
    args,
    frame_index,
) -> None:
    """Write one annotated PNG frame with a duck's-eye picture-in-picture."""
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw

    mujoco.mj_copyData(gaze_data, model, data)
    gaze_data.qpos[qpos_idx[HEAD_PITCH_ACT]] = gaze.pitch
    gaze_data.qpos[qpos_idx[HEAD_YAW_ACT]] = gaze.yaw
    mujoco.mj_forward(model, gaze_data)

    person = data.mocap_pos[tracker.person_mocap_id]
    camera.lookat[:] = 0.5 * (data.xpos[trunk] + person)
    renderer.update_scene(gaze_data, camera=camera)
    image = Image.fromarray(renderer.render())
    draw = ImageDraw.Draw(image)

    visible = view["visible"]
    ok = (120, 255, 140)
    bad = (255, 120, 120)
    trunk_z = float(data.xpos[trunk][2])
    label = {
        "IDLE": "IDLE - standing, watching",
        "RETREAT": "RETREAT - walking BACKWARD, holding heading",
        "TURN": "TURN - walking backward + turning",
        "CLEAR": "CLEAR - backward on the new heading",
        "DONE": "DONE - stopped",
    }[controller.state]

    draw.rectangle([0, 0, args.width, 114], fill=(0, 0, 0))
    draw.text((10, 6), f"t={t:6.2f}s   {label}", fill=(255, 255, 0))
    draw.text(
        (10, 24),
        f"person dist={distance:.3f} m   SEES PERSON: {'YES' if visible else 'NO '}"
        f"  (off-axis {math.degrees(view['off_axis']):.0f} deg)"
        + ("" if visible else "  << OUT OF VIEW / OCCLUDED"),
        fill=ok if visible else bad,
    )
    draw.text(
        (10, 42),
        f"CMD   vx={twist[0]:+.3f} (neg=backward)   wz={twist[2]:+.3f}",
        fill=(255, 255, 255),
    )
    draw.text(
        (10, 60),
        f"heading={math.degrees(yaw):+7.1f} deg   "
        f"turned {math.degrees(controller.turned):+6.1f} / "
        f"{math.degrees(controller.turn_target):.0f} deg",
        fill=(255, 255, 255),
    )
    draw.text(
        (10, 78),
        f"trunk z={trunk_z:.3f} (nominal {NOMINAL_TRUNK_Z:.3f})  "
        f"{'*** FALLEN ***' if trunk_z < FALLEN_TRUNK_Z else 'upright'}",
        fill=bad if trunk_z < FALLEN_TRUNK_Z else ok,
    )
    draw.text(
        (10, 96),
        "gaze is KINEMATIC and isolated - it does not drive the walking policy",
        fill=(170, 170, 170),
    )

    # Duck's-eye PiP: literally what the head camera sees.
    pip_renderer.update_scene(gaze_data, camera=head_cam)
    pip = Image.fromarray(pip_renderer.render())
    x0, y0 = args.width - PIP_W - 12, 124
    draw.rectangle([x0 - 3, y0 - 20, x0 + PIP_W + 3, y0 + PIP_H + 3], fill=(0, 0, 0))
    draw.text((x0, y0 - 17), "DUCK'S-EYE VIEW (head camera)", fill=ok if visible else bad)
    image.paste(pip, (x0, y0))
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        [x0 - 1, y0 - 1, x0 + PIP_W, y0 + PIP_H],
        outline=ok if visible else bad,
        width=2,
    )

    imageio.imwrite(str(args.render_dir / f"f{frame_index:05d}.png"), np.asarray(image))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run(args)
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(
            f"final_state={summary['final_state']} "
            f"turned={summary['turned_deg']:+.1f}deg "
            f"trunk_z={summary['trunk_z']:.3f} "
            f"({'upright' if summary['upright'] else 'FALLEN'}) "
            f"visible={summary['visible_steps']}/{summary['control_steps']} "
            f"max_off_axis={summary['max_off_axis_deg']:.1f}deg "
            f"frames={summary['frames']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

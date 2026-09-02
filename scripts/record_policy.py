#!/usr/bin/env python3
"""Record a short, curated MP4 from a MicroDuck ONNX policy in MuJoCo."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import mujoco

from infer_policy import DEFAULT_POSE, MICRODUCK_XML, PolicyInference


def _reset(model: mujoco.MjModel, data: mujoco.MjData, policy: PolicyInference) -> int:
    freejoint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    qpos_adr = int(model.jnt_qposadr[freejoint_id])
    data.qpos[qpos_adr:qpos_adr + 7] = [0.0, 0.0, 0.125, 1.0, 0.0, 0.0, 0.0]
    for i, qpos_idx in enumerate(policy.joint_qpos_indices):
        data.qpos[qpos_idx] = DEFAULT_POSE[i]
    data.ctrl[:] = DEFAULT_POSE
    mujoco.mj_forward(model, data)
    return qpos_adr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--walking", required=True, help="Path to the 61D/14D walking ONNX policy")
    parser.add_argument("--output", default="artifacts/verified/official-walking-policy.mp4")
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--lin-vel-x", type=float, default=0.25)
    parser.add_argument("--lin-vel-y", type=float, default=0.0)
    parser.add_argument("--ang-vel-z", type=float, default=0.0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(MICRODUCK_XML)
    model.opt.timestep = 0.005
    data = mujoco.MjData(model)
    policy = PolicyInference(
        model,
        data,
        walking_onnx_path=args.walking,
        use_projected_gravity=True,
        new_cmd_obs=True,
    )
    policy.set_vel_cmd(args.lin_vel_x, args.lin_vel_y, args.ang_vel_z)
    qpos_adr = _reset(model, data, policy)

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    camera.lookat[:] = [0.0, 0.0, 0.16]
    camera.distance = 0.85
    camera.azimuth = 140.0
    camera.elevation = -18.0
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    control_hz = 50
    steps = max(1, round(args.duration * control_hz))
    start_x = float(data.qpos[qpos_adr])
    print(f"Recording {steps / control_hz:.1f}s to {output}")
    try:
        with imageio.get_writer(str(output), fps=control_hz, codec="libx264", quality=8, macro_block_size=1) as writer:
            for step in range(steps):
                action = policy.infer()
                policy.apply_action(action)
                for _ in range(4):
                    mujoco.mj_step(model, data)
                renderer.update_scene(data, camera=camera)
                writer.append_data(renderer.render())
                if (step + 1) % control_hz == 0:
                    print(f"  t={(step + 1) / control_hz:4.0f}s  x={float(data.qpos[qpos_adr]) - start_x:+.3f}m  z={float(data.qpos[qpos_adr + 2]) * 1000:.1f}mm")
    finally:
        renderer.close()

    print(f"Saved {output} ({output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()

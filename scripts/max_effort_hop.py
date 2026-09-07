"""What hop is physically FEASIBLE: crouch, then extend flat out. No head.

The CEM probe searched for a strategy. This does not need to -- the strategy is
known (crouch, then extend as fast as the actuators allow) -- so it sweeps the
only two things that matter and reports the envelope:

    crouch depth  x  push duration  ->  takeoff velocity, CoM rise

Both poses come from the model's own kinematics rather than from guesswork: the
extension target is the TALLEST reachable stance, found by gridding hip/knee/
ankle over their true joint ranges, and the crouch is the same search run the
other way with a floor so the robot does not fold on top of itself.

The head is deliberately not used. It is ~31% of body mass and worth something,
but mixing it in makes the answer "what can the legs do?" unanswerable -- and
that is the question, because the legs are what a jump is made of.

    uv run python scripts/max_effort_hop.py
"""

from __future__ import annotations

import argparse
import itertools
import os
import re

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HOP_NO_PUSH", "1")

import mujoco
import numpy as np
import torch

import mjlab_microduck.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab_microduck.robot.microduck_constants import HOME_FRAME

L_HIP, L_KNEE, L_ANKLE = 2, 3, 4
R_HIP, R_KNEE, R_ANKLE = 11, 12, 13
MIRROR = -1.0
TRUNK_FLOOR = 0.075   # m; below this the pose is a robot folded over, not a crouch


def _kinematic_poses(task: str):
    """(extension pose, crouch pose, heights) as hip/knee/ankle offsets from HOME."""
    m = load_env_cfg(task).scene.entities["robot"].spec_fn().compile()
    d = mujoco.MjData(m)
    jid = {n: m.jnt_qposadr[j] for j in range(m.njnt)
           if (n := mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j))}
    tb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    pads = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
            for n in ("left_foot_pad", "right_foot_pad")]
    pads = [p for p in pads if p >= 0]

    def height(dh, dk, da):
        d.qpos[:] = 0.0
        if m.jnt_type[0] == 0:
            d.qpos[3] = 1.0
        for pat, v in HOME_FRAME.joint_pos.items():
            for n, adr in jid.items():
                if re.fullmatch(pat, n):
                    d.qpos[adr] = v
        d.qpos[jid["left_hip_pitch"]] += dh
        d.qpos[jid["right_hip_pitch"]] -= dh
        d.qpos[jid["left_knee"]] += dk
        d.qpos[jid["right_knee"]] -= dk
        d.qpos[jid["left_ankle"]] += da
        d.qpos[jid["right_ankle"]] -= da
        mujoco.mj_forward(m, d)
        return d.xpos[tb][2] - min(d.xpos[p][2] for p in pads)

    def grid(jn, home_v):
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
        lo, hi = m.jnt_range[j]
        return np.linspace(lo - home_v, hi - home_v, 21)

    gh, gk, ga = (grid("left_hip_pitch", -0.4579),
                  grid("left_knee", -0.0049),
                  grid("left_ankle", 0.4530))
    home_h = height(0, 0, 0)
    tall = (home_h, 0.0, 0.0, 0.0)
    low = (home_h, 0.0, 0.0, 0.0)
    for dh, dk, da in itertools.product(gh, gk, ga):
        h = height(dh, dk, da)
        if h > tall[0]:
            tall = (h, dh, dk, da)
        if TRUNK_FLOOR < h < low[0]:
            low = (h, dh, dk, da)
    return tall, low, home_h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="Mjlab-Hop-InPlaceSym-K3344-MicroDuck")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--settle", type=float, default=0.5)
    ap.add_argument("--hold", type=float, default=0.25)
    args = ap.parse_args()

    tall, low, home_h = _kinematic_poses(args.task)
    print(f"[maxeffort] home stance {home_h*1000:.1f} mm")
    print(f"[maxeffort] tallest     {tall[0]*1000:.1f} mm  "
          f"(hip{tall[1]:+.2f} knee{tall[2]:+.2f} ankle{tall[3]:+.2f})")
    print(f"[maxeffort] deepest     {low[0]*1000:.1f} mm  "
          f"(hip{low[1]:+.2f} knee{low[2]:+.2f} ankle{low[3]:+.2f})")
    print(f"[maxeffort] TRAVEL crouch->tall = {(tall[0]-low[0])*1000:.1f} mm")

    # THE SQUAT RUNS ALONG THE JUMP AXIS, as a multiple of the extension offset
    # rather than toward the kinematically deepest pose. The deepest pose puts
    # the ankle on its -90 deg limit in a shape the robot cannot hold, and
    # sweeping 141 deg of ankle in 60 ms is not something an XL330 tracks -- so
    # every such candidate toppled and scored zero. Squatting back down the same
    # line the push travels keeps the feet flat and the pose reachable.
    depths = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])   # x the extension offset
    pushes = np.array([0.05, 0.08, 0.12, 0.18, 0.26])  # seconds
    combos = [(d, p) for d in depths for p in pushes]
    n = len(combos)

    cfg = load_env_cfg(args.task)
    cfg.scene.num_envs = n
    cfg.terminations = type(cfg.terminations)()
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    env.extras.setdefault("log", {})
    dt = cfg.sim.mujoco.timestep * cfg.decimation
    A = env.action_manager.total_action_dim
    dev = args.device
    robot = env.scene["robot"]

    dep = torch.tensor([c[0] for c in combos], device=dev, dtype=torch.float32)
    pdur = torch.tensor([c[1] for c in combos], device=dev, dtype=torch.float32)
    c_off = -torch.tensor(tall[1:], device=dev, dtype=torch.float32)
    e_off = torch.tensor(tall[1:], device=dev, dtype=torch.float32)

    def act(t):
        a = torch.zeros((n, A), device=dev)
        sink = torch.clamp(torch.tensor(t / args.settle, device=dev), 0, 1)
        sink = sink * sink * (3 - 2 * sink)
        pt = (t - args.settle - args.hold) / pdur
        push = torch.clamp(pt, 0, 1)
        push = push * push * (3 - 2 * push)
        crouch = dep[:, None] * c_off[None, :]   # depth x (-extension offset)
        q = crouch * sink + (e_off[None, :] - crouch) * push[:, None]
        a[:, L_HIP], a[:, L_KNEE], a[:, L_ANKLE] = q[:, 0], q[:, 1], q[:, 2]
        a[:, R_HIP] = MIRROR * q[:, 0]
        a[:, R_KNEE] = MIRROR * q[:, 1]
        a[:, R_ANKLE] = MIRROR * q[:, 2]
        return a

    # Body masses from the compiled spec: EntityData does not expose them, and
    # an unweighted mean over bodies is not a centre of mass.
    _m = load_env_cfg(args.task).scene.entities["robot"].spec_fn().compile()
    mass = torch.tensor(_m.body_mass.copy(), device=dev, dtype=torch.float32)

    def com():
        bz = robot.data.body_link_pos_w[:, :, 2].float()
        mm = mass.reshape(1, -1)[:, : bz.shape[1]]
        return (bz * mm).sum(1) / mm.sum()

    from mjlab_microduck.tasks.mdp import _both_feet_airborne
    env.reset()
    z_stance = com().clone()
    prev_z = com().clone()
    best = torch.zeros(n, device=dev)
    v_take = torch.zeros(n, device=dev)
    steps = int((args.settle + args.hold + 1.2) / dt)
    for i in range(steps):
        env.step(act(i * dt))
        air = _both_feet_airborne(env, "feet_ground_contact")
        air = (air > 0.5) if air is not None else torch.zeros(n, dtype=torch.bool, device=dev)
        z = com()
        # CoM vertical velocity by finite difference -- body_link_vel_w is not
        # available on this EntityData either, and a difference of the same
        # quantity being scored is less likely to disagree with it.
        vz = (z - prev_z) / dt
        prev_z = z.clone()
        gz = robot.data.projected_gravity_b[:, 2].float()
        up = torch.arccos(torch.clamp(-gz, -1, 1)) < 0.5236
        z_stance = torch.where(~air, z, z_stance)
        newly = air & up & (vz > 0)
        v_take = torch.maximum(v_take, torch.where(newly, vz, torch.zeros_like(vz)))
        best = torch.maximum(best, torch.where(air & up, z - z_stance, torch.zeros_like(z)))

    b = best.cpu().numpy(); v = v_take.cpu().numpy()
    print(f"\nRESULT {'depth_x':>7s} {'push_s':>7s} {'v_takeoff':>10s} {'rise_mm':>9s} {'ballistic_mm':>13s}")
    for k, (dpt, ps) in enumerate(combos):
        print(f"RESULT {dpt:7.2f} {ps:7.3f} {v[k]:10.3f} {b[k]*1000:9.2f} "
              f"{v[k]**2/(2*9.81)*1000:13.2f}")
    k = int(np.argmax(b))
    print(f"\nRESULT BEST: crouch {combos[k][0]:.2f} push {combos[k][1]:.3f}s -> "
          f"{b[k]*1000:.2f} mm rise, takeoff {v[k]:.3f} m/s "
          f"(= {v[k]**2/(2*9.81)*1000:.2f} mm ballistic)")


if __name__ == "__main__":
    main()

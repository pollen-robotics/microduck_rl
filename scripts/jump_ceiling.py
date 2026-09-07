"""How high can the legs alone jump? A deterministic, verifiable ceiling.

Supersedes the CEM in `openloop_hop.py`, which produced two numbers that were
both artifacts. Everything below exists because one of them was wrong:

  * DOMAIN RANDOMISATION IS OFF. The old probe ran one candidate per env with DR
    active, so every candidate was scored on a DIFFERENT robot and the search
    optimised the draw, not the trajectory. Its winner did not reproduce.
  * THE DATUM IS THE STANDING CoM, fixed before the crouch. The old score used
    the last in-contact height, so a robot that squatted 50 mm and stood back up
    scored 50 mm of "rise" without leaving the ground.
  * A CANDIDATE IS DISQUALIFIED IF ITS BODY TOUCHES THE GROUND, rather than
    relying on terminations -- which cannot be used here, since a mid-episode
    reset hands the next step of one trajectory to a freshly spawned robot. The
    old probe simply disabled them, and scored a robot collapsing onto its
    trunk (root 155 -> 90 mm) as a successful jump.
  * TWO INDEPENDENT MEASURES must agree: peak CoM above standing, and the
    ballistic apex implied by CoM velocity at takeoff. A datum bug moves one and
    not the other.

No head: the legs are what a jump is made of, and mixing in a 31%-of-body-mass
swing makes "what can the legs do?" unanswerable.

    uv run python scripts/jump_ceiling.py --iters 20
    uv run python scripts/jump_ceiling.py --replay      # verify the winner
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HOP_NO_PUSH", "1")

import numpy as np
import torch

import mjlab_microduck.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

L_HIP, L_KNEE, L_ANKLE = 2, 3, 4
R_HIP, R_KNEE, R_ANKLE = 11, 12, 13
MIRROR = -1.0
T_SETTLE = 0.5          # hold HOME_FRAME, then measure the standing datum
UPRIGHT_MAX = 0.5236    # 30 deg

PARAMS = (
    ("crouch_hip",  -1.00, 1.00),
    ("crouch_knee", -1.40, 1.40),
    ("crouch_ankle",-1.00, 1.00),
    ("ext_hip",     -1.20, 1.20),
    ("ext_knee",    -1.60, 1.60),
    ("ext_ankle",   -1.20, 1.20),
    ("t_sink",       0.08, 0.60),
    ("t_hold",       0.00, 0.25),
    ("t_push",       0.05, 0.45),
)
NAMES = [p[0] for p in PARAMS]
LO = np.array([p[1] for p in PARAMS], dtype=np.float32)
HI = np.array([p[2] for p in PARAMS], dtype=np.float32)


def _smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def actions_at(t, p, n_act):
    n = p.shape[0]
    a = np.zeros((n, n_act), dtype=np.float32)
    t = t - T_SETTLE
    if np.all(t < 0.0):
        return a
    (c_h, c_k, c_a, e_h, e_k, e_a, t_s, t_ho, t_p) = [p[:, i] for i in range(len(PARAMS))]
    sink = _smooth(t / np.maximum(t_s, 1e-3))
    push = _smooth((t - (t_s + t_ho)) / np.maximum(t_p, 1e-3))
    hip = c_h * sink + (e_h - c_h) * push
    knee = c_k * sink + (e_k - c_k) * push
    ank = c_a * sink + (e_a - c_a) * push
    a[:, L_HIP], a[:, L_KNEE], a[:, L_ANKLE] = hip, knee, ank
    a[:, R_HIP], a[:, R_KNEE], a[:, R_ANKLE] = MIRROR * hip, MIRROR * knee, MIRROR * ank
    return a


def make_env(task, n, device):
    cfg = load_env_cfg(task)
    cfg.scene.num_envs = n
    # Determinism: no randomisation, no pushes, and no curricula (they drive the
    # randomisation events and raise once those are gone).
    keep = {"reset_base", "reset_robot_joints", "expand_bam_friction_fields",
            "reset_action_history"}
    cfg.events = type(cfg.events)({k: v for k, v in cfg.events.items() if k in keep})
    cfg.curriculum = type(cfg.curriculum)()
    cfg.terminations = type(cfg.terminations)()
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    env.extras.setdefault("log", {})
    dt = cfg.sim.mujoco.timestep * cfg.decimation
    model = load_env_cfg(task).scene.entities["robot"].spec_fn().compile()
    mass = torch.tensor(model.body_mass.copy(), device=device, dtype=torch.float32)
    return env, dt, mass


def evaluate(env, dt, mass, p, duration, trace=False):
    """Returns (jump_mm, ballistic_mm, disqualified, info)."""
    from mjlab_microduck.tasks.mdp import _both_feet_airborne

    n = p.shape[0]
    dev = env.device
    robot = env.scene["robot"]
    n_act = env.action_manager.total_action_dim
    body = env.scene["body_ground_contact"]

    def com():
        bz = robot.data.body_link_pos_w[:, :, 2].float()
        mm = mass.reshape(1, -1)[:, : bz.shape[1]]
        return (bz * mm).sum(1) / mm.sum()

    env.reset()
    steps = int(duration / dt)
    settle_steps = int(T_SETTLE / dt)
    z0 = None
    prev_z = com().clone()
    peak = torch.full((n,), -1e9, device=dev)
    v_take = torch.zeros(n, device=dev)
    was_air = torch.zeros(n, dtype=torch.bool, device=dev)
    dq = torch.zeros(n, dtype=torch.bool, device=dev)
    rows = []

    for i in range(steps):
        env.step(torch.as_tensor(
            actions_at(np.full(n, i * dt, dtype=np.float32), p, n_act), device=dev))
        z = com()
        vz = (z - prev_z) / dt
        prev_z = z.clone()
        if i == settle_steps - 1:
            z0 = z.clone()          # the standing datum, fixed before the crouch

        air = _both_feet_airborne(env, "feet_ground_contact")
        air = (air > 0.5) if air is not None else torch.zeros(n, dtype=torch.bool, device=dev)
        gz = robot.data.projected_gravity_b[:, 2].float()
        tilt = torch.arccos(torch.clamp(-gz, -1, 1))

        # Disqualify: body on the ground, or tipped over. Either makes the CoM
        # trace meaningless as a jump.
        touched = body.data.found
        touched = (touched.sum(dim=-1) > 0) if touched is not None and touched.dim() > 1 \
            else torch.zeros(n, dtype=torch.bool, device=dev)
        dq |= touched | (tilt > UPRIGHT_MAX)

        if z0 is not None:
            flying = air & ~dq
            peak = torch.where(flying, torch.maximum(peak, z - z0), peak)
            takeoff = flying & ~was_air
            v_take = torch.where(takeoff, torch.clamp(vz, min=0.0), v_take)
            was_air = torch.where(~dq, air, was_air)
        if trace and i % 10 == 0:
            rows.append((i * dt, float(z.mean()), float(air.float().mean()),
                         float(touched.float().mean()), float(np.degrees(float(tilt.mean())))))

    jump = torch.where(dq | (peak < -1e8), torch.zeros_like(peak), torch.clamp(peak, min=0.0))
    ballistic = v_take ** 2 / (2 * 9.81)
    return (jump.cpu().numpy(), ballistic.cpu().numpy(), dq.cpu().numpy(), rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="Mjlab-Hop-InPlaceSym-K3344-MicroDuck")
    ap.add_argument("--pop", type=int, default=512)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--elite", type=float, default=0.08)
    ap.add_argument("--duration", type=float, default=1.6)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="jump_ceiling_best.npz")
    ap.add_argument("--replay", action="store_true",
                    help="Re-evaluate the saved winner and print a trace. The "
                         "old probe's numbers did not survive this step.")
    args = ap.parse_args()

    if args.replay:
        d = np.load(args.out, allow_pickle=True)
        p = np.repeat(d["params"].astype(np.float32).reshape(1, -1), 4, axis=0)
        env, dt, mass = make_env(args.task, 4, args.device)
        j, b, dq, rows = evaluate(env, dt, mass, p, args.duration + 0.6, trace=True)
        print(f"RESULT replay jump {j.mean()*1000:.2f} mm   "
              f"ballistic-from-takeoff {b.mean()*1000:.2f} mm   dq={bool(dq.any())}")
        print(f"RESULT saved score was {float(d['score'])*1000:.2f} mm")
        print(f"RESULT {'t':>6s} {'CoM_mm':>8s} {'feet_air':>9s} {'body_gnd':>9s} {'tilt':>6s}")
        for t, z, a, bo, ti in rows:
            print(f"RESULT {t:6.2f} {z*1000:8.1f} {a:9.2f} {bo:9.2f} {ti:6.1f}")
        return

    env, dt, mass = make_env(args.task, args.pop, args.device)
    print(f"[jump] task={args.task} pop={args.pop} dt={dt:.4f} DR=off", flush=True)
    mu = (LO + HI) / 2.0
    sigma = (HI - LO) / 4.0
    n_elite = max(8, int(args.pop * args.elite))
    best, best_p = -1.0, None
    for it in range(args.iters):
        p = np.clip(np.random.normal(mu, sigma, (args.pop, len(PARAMS))), LO, HI).astype(np.float32)
        j, b, dq, _ = evaluate(env, dt, mass, p, args.duration)
        order = np.argsort(-j)
        el = p[order[:n_elite]]
        mu, sigma = el.mean(0), el.std(0) + 1e-3
        if j[order[0]] > best:
            best, best_p = float(j[order[0]]), p[order[0]].copy()
        print(f"[jump] iter {it:2d}  best {j[order[0]]*1000:6.2f} mm  "
              f"elite {j[order[:n_elite]].mean()*1000:6.2f} mm  "
              f"ballistic(best) {b[order[0]]*1000:6.2f} mm  dq {dq.mean()*100:4.1f}%", flush=True)
    np.savez(args.out, params=best_p, names=np.array(NAMES), score=best)
    print(f"\nRESULT jump ceiling (legs only, no head): {best*1000:.2f} mm")
    for nm, v in zip(NAMES, best_p):
        print(f"RESULT   {nm:14s} {v:+.4f}")
    print(f"RESULT wrote {args.out} -- now run with --replay to verify")


if __name__ == "__main__":
    main()

"""What is the best hop this robot can physically do?

Every hop number in the campaign is what RL happened to find. This asks the
other question -- the CEILING -- by optimising an OPEN-LOOP countermovement
jump directly, with no policy and no observations. It answers three things at
once:

  1. How high can this robot + boot + actuator model go, at best?
  2. Is the mechanism viable at all, or is RL already at the ceiling?
  3. A dynamically FEASIBLE reference trajectory, to guide training.

Point 3 is why this runs before any tracking reward. A hand-written reference
can be physically impossible for an XL330 -- the bench found 75% of external
load eaten by gearbox friction and the P-gain lever exhausted by kp 400 -- and
training a policy to track an infeasible trajectory makes things worse, not
better. Whatever comes out of here was executed by the actual actuator model.

METHOD: cross-entropy method. Every env runs a DIFFERENT candidate, so one
rollout of 4096 envs evaluates 4096 candidates; the elite fraction refits the
sampling distribution. No optimiser dependency, and it parallelises for free.

The trajectory is a countermovement jump, as a human does it: sink into a
crouch, then extend everything at once while swinging the head up. Symmetric by
construction -- left and right differ only by the sign the HOME_FRAME mirror
demands -- so this cannot produce the skip that RL keeps converging to.

    uv run python scripts/openloop_hop.py --iters 12 --pop 4096
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HOP_NO_PUSH", "1")  # a ceiling is measured undisturbed

import numpy as np
import torch

import mjlab_microduck.tasks  # noqa: F401  (registers the tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

# Action layout, from the compiled model's actuated-joint order (springs are
# passive and absent). Verified against the exported ONNX's `joint_names`.
L_HIP_PITCH, L_KNEE, L_ANKLE = 2, 3, 4
NECK_PITCH, HEAD_PITCH = 5, 6
R_HIP_PITCH, R_KNEE, R_ANKLE = 11, 12, 13

# HOME_FRAME mirrors the sagittal joints -- left_hip_pitch is -0.4579 against
# right's +0.4579 -- so a SYMMETRIC motion is +d on the left and -d on the
# right. Getting this backwards produces a robot that pikes instead of
# crouching, which is why `--check` measures roll and CoM drop before trusting
# any score.
MIRROR = -1.0

# Beyond this tilt the robot is toppling, not hopping, and its CoM rise is
# measuring a fall. 30 deg is well above the 2.5-5.9 deg an honest hop reaches
# at takeoff and well below the 52.8 deg the head-rest exploit sat at.
UPRIGHT_MAX = 0.5236  # 30 deg

# Parameter vector, in order. Ranges are the search box; CEM samples inside it
# and clips. Amplitudes are radians of joint offset from HOME_FRAME, times in
# seconds.
PARAMS = (
    ("crouch_hip",   0.00, 0.90),   # sink: hip flexion
    ("crouch_knee",  0.00, 1.20),   # sink: knee flexion
    ("crouch_ankle", 0.00, 0.80),   # sink: ankle dorsiflexion
    ("push_hip",     0.00, 1.20),   # drive: hip extension past home
    ("push_knee",    0.00, 1.60),   # drive: knee extension past home
    ("push_ankle",   0.00, 1.20),   # drive: ankle plantarflexion past home
    ("t_sink",       0.06, 0.60),   # seconds to reach full crouch
    ("t_hold",       0.00, 0.20),   # pause at the bottom
    ("t_push",       0.04, 0.40),   # seconds to drive through
    ("head_amp",     0.00, 1.20),   # neck+head pitch swing amplitude
    ("head_lead",   -0.15, 0.15),   # head swing lead (-) or lag (+) vs the push
)
NAMES = [p[0] for p in PARAMS]
LO = np.array([p[1] for p in PARAMS], dtype=np.float32)
HI = np.array([p[2] for p in PARAMS], dtype=np.float32)


def _smooth(x):
    """Smoothstep on [0, 1]; zero velocity at both ends so the servos are not
    asked for a step change they cannot make."""
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


# Held at HOME_FRAME before the trajectory starts, so every candidate begins
# from the SAME settled stance. Without it the score measures how well a
# candidate copes with the spawn transient, not how well it hops.
T_SETTLE = 0.4


def actions_at(t: np.ndarray, p: np.ndarray, n_act: int) -> np.ndarray:
    """The scripted action for every env at time `t`. p is [N, len(PARAMS)]."""
    n = p.shape[0]
    a = np.zeros((n, n_act), dtype=np.float32)
    t = t - T_SETTLE
    if np.all(t < 0.0):
        return a  # still settling: hold HOME_FRAME
    (c_hip, c_knee, c_ank, k_hip, k_knee, k_ank,
     t_sink, t_hold, t_push, h_amp, h_lead) = [p[:, i] for i in range(len(PARAMS))]

    # Phase: sink (0 -> 1), hold, then push (1 -> -push amplitude).
    sink = _smooth(t / np.maximum(t_sink, 1e-3))
    t_push_start = t_sink + t_hold
    push = _smooth((t - t_push_start) / np.maximum(t_push, 1e-3))

    # Leg chain: crouch scaled by `sink`, then driven the other way by `push`.
    hip = c_hip * sink - (c_hip + k_hip) * push
    knee = c_knee * sink - (c_knee + k_knee) * push
    ank = c_ank * sink - (c_ank + k_ank) * push

    a[:, L_HIP_PITCH] = hip
    a[:, L_KNEE] = knee
    a[:, L_ANKLE] = ank
    a[:, R_HIP_PITCH] = MIRROR * hip
    a[:, R_KNEE] = MIRROR * knee
    a[:, R_ANKLE] = MIRROR * ank

    # Head: the countermovement's other half. Pitches DOWN with the crouch and
    # whips UP through the push, offset by `head_lead` so the timing is free.
    hs = _smooth((t - (t_push_start + h_lead)) / np.maximum(t_push, 1e-3))
    head = h_amp * sink - 2.0 * h_amp * hs
    a[:, NECK_PITCH] = head
    a[:, HEAD_PITCH] = head
    return a


def rollout(env, p: np.ndarray, duration: float, dt: float, check: bool = False):
    """Score every candidate by CoM rise during genuine flight."""
    from mjlab_microduck.tasks.mdp import _both_feet_airborne

    n = p.shape[0]
    dev = env.device
    robot = env.scene["robot"]
    n_act = env.action_manager.total_action_dim

    env.reset()
    env.extras.setdefault("log", {})
    mass = robot.data.default_mass.to(dev).float() if hasattr(robot.data, "default_mass") else None

    def com_z():
        bz = robot.data.body_link_pos_w[:, :, 2].float()
        m = mass.reshape(1, -1)[:, : bz.shape[1]] if mass is not None else torch.ones(1, bz.shape[1], device=dev)
        return (bz * m).sum(1) / m.sum()

    steps = int(duration / dt)
    z_stance = com_z().clone()
    best = torch.zeros(n, device=dev)
    ever_air = torch.zeros(n, dtype=torch.bool, device=dev)
    dead = torch.zeros(n, dtype=torch.bool, device=dev)
    roll_max = torch.zeros(n, device=dev)
    sink_drop = torch.zeros(n, device=dev)

    for i in range(steps):
        t = np.full(n, i * dt, dtype=np.float32)
        a = torch.as_tensor(actions_at(t, p, n_act), device=dev)
        out = env.step(a)
        d = out[2] if len(out) > 2 else None
        if torch.is_tensor(d):
            dead |= d.bool()

        air = _both_feet_airborne(env, "feet_ground_contact")
        air = (air > 0.5) if air is not None else torch.zeros(n, dtype=torch.bool, device=dev)
        z = com_z()
        gz = robot.data.projected_gravity_b[:, 2].float()
        tilt = torch.arccos(torch.clamp(-gz, -1, 1))

        # UPRIGHT AT TAKEOFF, or the "rise" is a topple. A falling robot lifts
        # both feet and its CoM keeps moving, which scores beautifully and is
        # not a hop -- the same confusion that produced the head-rest exploit.
        upright = tilt < UPRIGHT_MAX
        z_stance = torch.where(~air, z, z_stance)
        rise = torch.where(air & upright, z - z_stance, torch.zeros_like(z))
        best = torch.maximum(best, torch.nan_to_num(rise))
        ever_air |= air & upright
        roll_max = torch.maximum(roll_max, tilt)
        if check:
            sink_drop = torch.minimum(sink_drop, z - z_stance)

    # A candidate that never left the ground, or that fell, scores nothing.
    score = torch.where(ever_air & ~dead, best, torch.zeros_like(best))
    return (score.cpu().numpy(), ever_air.cpu().numpy(), dead.cpu().numpy(),
            roll_max.cpu().numpy(), sink_drop.cpu().numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="Mjlab-Hop-InPlaceSym-K3344-MicroDuck")
    ap.add_argument("--pop", type=int, default=4096)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--elite", type=float, default=0.05)
    ap.add_argument("--duration", type=float, default=1.6)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="openloop_hop_best.npz")
    ap.add_argument("--check", action="store_true",
                    help="Report crouch depth and roll, to confirm the mirror "
                         "sign produces a symmetric sink rather than a pike.")
    args = ap.parse_args()

    cfg = load_env_cfg(args.task)
    cfg.scene.num_envs = args.pop
    # NO TERMINATIONS. This is an open-loop probe: a reset mid-trajectory would
    # hand the next candidate a robot part-way through someone else's jump,
    # because the script is driven by wall-clock `t` and knows nothing about
    # episode boundaries. Falls are handled by the upright gate in `rollout`
    # instead, which is the honest place for them.
    cfg.terminations = type(cfg.terminations)()
    # Long enough that one full countermovement fits inside an episode.
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    dt = cfg.sim.mujoco.timestep * cfg.decimation
    print(f"[openloop] task={args.task} pop={args.pop} dt={dt:.4f} "
          f"duration={args.duration}s", flush=True)

    mu = (LO + HI) / 2.0
    sigma = (HI - LO) / 4.0
    n_elite = max(8, int(args.pop * args.elite))
    best_ever, best_p = -1.0, None

    for it in range(args.iters):
        p = np.random.normal(mu, sigma, size=(args.pop, len(PARAMS))).astype(np.float32)
        p = np.clip(p, LO, HI)
        score, air, dead, roll, sink = rollout(env, p, args.duration, dt, args.check)
        order = np.argsort(-score)
        elite = p[order[:n_elite]]
        mu = elite.mean(axis=0)
        sigma = elite.std(axis=0) + 1e-3
        top = float(score[order[0]])
        if top > best_ever:
            best_ever, best_p = top, p[order[0]].copy()
        print(f"[openloop] iter {it:2d}  best {top*1000:6.2f} mm  "
              f"elite-mean {score[order[:n_elite]].mean()*1000:6.2f} mm  "
              f"airborne {air.mean()*100:4.1f}%  fell {dead.mean()*100:4.1f}%",
              flush=True)
        if args.check and it == 0:
            print(f"[openloop]   check: max crouch CoM drop "
                  f"{-sink.min()*1000:.1f} mm, max roll {np.degrees(roll).max():.1f} deg")

    print(f"\nRESULT open-loop ceiling: {best_ever*1000:.2f} mm CoM rise")
    print("RESULT best parameters:")
    for name, v in zip(NAMES, best_p):
        print(f"RESULT   {name:14s} {v:+.4f}")
    np.savez(args.out, params=best_p, names=np.array(NAMES), score=best_ever)
    print(f"RESULT written {args.out}")


if __name__ == "__main__":
    main()

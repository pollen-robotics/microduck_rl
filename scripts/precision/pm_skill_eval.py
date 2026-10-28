"""Evaluate a nudge policy the way robotd runs it, in velstand's own training env (all DR, obs noise, backlash).

Each trial: the walking policy (velstand v8) stands at zero command, the nudge skill runs (its constant command for
`--go` s, zeros for `--unwind` s, fed raw as robotd feeds skills), then the walking policy again at zero command for
`--after` s. The stance displacement from the skill start to the end is the nudge. Optionally records the qpos of the
first `--record` envs (npz) for a replay render.

  PYTHONPATH=src .venv-mjlab/bin/python scripts/precision/pm_skill_eval.py NUDGE.onnx --axis x --units 0.3 \
      --envs 16 --reps 3 --out OUT.json [--record 4 --npz OUT.npz] [--scale 1.0]
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pm_env import Policy, Sim, rel  # noqa: E402

AX = {"x": 0, "y": 1, "w": 2}
UNIT = {"x": 50.0, "y": 50.0, "w": 14.3239}  # mm, mm, degrees per command unit (NUDGE_UNIT_M 0.05, NUDGE_UNIT_RAD 0.25)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx")
    ap.add_argument("--axis", choices=list(AX), required=True)
    ap.add_argument("--units", type=float, required=True, help="signed command (x, y: metres / 0.05, so 0.3 = 1.5 cm; w: radians / 0.25, so 0.524 = 7.5°)")
    ap.add_argument("--go", type=float, default=0.5)
    ap.add_argument("--unwind", type=float, default=1.3)
    ap.add_argument("--before", type=float, default=1.0)
    ap.add_argument("--after", type=float, default=1.5)
    ap.add_argument("--scale", type=float, default=None, help="action scale while the skill runs (robot default 0.9)")
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--no-lowpass", action="store_true", help="no robotd leg/head low-pass (training has none)")
    ap.add_argument("--before-nudge", action="store_true", help="the nudge policy (zero command) stands before, not velstand")
    ap.add_argument("--handback-lastact", action="store_true", help="probe: at the handback, velstand sees its own standing action as last action")
    ap.add_argument("--clamp", type=float, default=None, help="clip the nudge output so HOME + action stays within the joint range + this margin (rad; hip roll +0.3)")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--record", type=int, default=0)
    ap.add_argument("--npz", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sim = Sim(n=a.envs, seed=a.seed, lowpass=(1.0, 1.0) if a.no_lowpass else None)
    nudge = Policy(a.onnx)
    if a.clamp is not None:
        t = sim.env.action_manager.get_term("joint_pos")
        lim = sim.robot.data.joint_pos_limits[0, t._target_ids].clone()
        home = sim.robot.data.default_joint_pos[0, t._target_ids].clone()
        m = torch.tensor([a.clamp + (0.3 if "hip_roll" in n else 0.0) for n in t._target_names])
        lo, hi = lim[:, 0] - m - home, lim[:, 1] + m - home
        raw = nudge
        nudge = lambda obs: torch.clamp(raw(obs), lo, hi)
    walk_scale = sim.action_scale
    cmd = [0.0, 0.0, 0.0]
    cmd[AX[a.axis]] = a.units
    acts = []
    for _ in range(100):
        sim.step([0.0, 0.0, 0.0])
        acts.append(sim.obs[:, 34:48].clone())
    stand_act = torch.stack(acts[50:]).mean(dim=0)
    rows, frames, marks = [], [], []
    # commanded-target overrun (robotd clamps only to +-pi): HOME + scale x filtered action vs the joint ranges
    act_term = sim.env.action_manager.get_term("joint_pos")
    lim = sim.robot.data.joint_pos_limits[0, act_term._target_ids].clone()      # (14, 2)
    overrun = torch.zeros(14)
    rec = list(range(min(a.record, a.envs)))
    t0 = time.time()
    s_first = sim.stance()
    for r in range(a.reps):
        for _ in range(int(a.before / sim.dt)):
            sim.step([0.0, 0.0, 0.0], policy=nudge if a.before_nudge else None, smooth=not a.before_nudge)
            if rec: frames.append(sim.qpos(rec))
        s0 = sim.stance()
        marks.append(len(frames))
        gz_max = torch.full((sim.n,), -1.0)          # projected gravity z: -1 upright, -0.7 = 45 deg tilt
        # no resets between reps: a duck already down at the start is not a new fall
        up0 = sim.robot.data.projected_gravity_b[:, 2] < -0.7
        sim.action_scale = a.scale if a.scale is not None else walk_scale
        for k in range(int(round((a.go + a.unwind) / sim.dt))):
            sim.step(cmd if k * sim.dt < a.go - 1e-6 else [0.0, 0.0, 0.0], policy=nudge, smooth=False)
            gz_max = torch.maximum(gz_max, sim.robot.data.projected_gravity_b[:, 2])
            tgt = act_term._processed_actions
            overrun = torch.maximum(overrun, torch.maximum(tgt - lim[:, 1], lim[:, 0] - tgt).max(dim=0).values)
            if rec: frames.append(sim.qpos(rec))
        sim.action_scale = walk_scale
        fell_skill = up0 & (gz_max > -0.7)
        s_skill = sim.stance()
        if a.handback_lastact:
            sim.obs[:, 34:48] = stand_act
        dk = rel(s_skill, s0)
        for _ in range(int(a.after / sim.dt)):
            sim.step([0.0, 0.0, 0.0])
            gz_max = torch.maximum(gz_max, sim.robot.data.projected_gravity_b[:, 2])
            if rec: frames.append(sim.qpos(rec))
        d = rel(sim.stance(), s0)
        dh = rel(sim.stance(), s_skill)
        rows.append({"fwd_mm": (d[:, 0] * 1000).tolist(), "left_mm": (d[:, 1] * 1000).tolist(),
                     "yaw_deg": (d[:, 2] * 180 / math.pi).tolist(),
                     "skill_fwd_mm": (dk[:, 0] * 1000).tolist(), "skill_left_mm": (dk[:, 1] * 1000).tolist(),
                     "skill_yaw_deg": (dk[:, 2] * 180 / math.pi).tolist(),
                     "handback_mm": (torch.norm(dh[:, :2], dim=1) * 1000).tolist(),
                     "handback_deg": (dh[:, 2].abs() * 180 / math.pi).tolist(), "upright": sim.upright().tolist(),
                     "max_tilt_deg": torch.rad2deg(torch.acos(torch.clamp(-gz_max, -1, 1))).tolist(),
                     "up_at_start": up0.tolist(), "new_fall_skill": fell_skill.tolist(),
                     "new_fall_after": (up0 & ~fell_skill & (gz_max > -0.7)).tolist()})
    f = np.array(sum((r["fwd_mm"] for r in rows), []))
    l = np.array(sum((r["left_mm"] for r in rows), []))
    y = np.array(sum((r["yaw_deg"] for r in rows), []))
    hb = np.array(sum((r["handback_mm"] for r in rows), []))
    kf = np.array(sum((r["skill_fwd_mm"] for r in rows), []))
    kl = np.array(sum((r["skill_left_mm"] for r in rows), []))
    ky = np.array(sum((r["skill_yaw_deg"] for r in rows), []))
    up = sum(sum(r["upright"]) for r in rows)
    tilt = np.array(sum((r["max_tilt_deg"] for r in rows), []))
    fell = int((tilt > 45).sum())
    n_up = sum(sum(r["up_at_start"]) for r in rows)
    nf_skill = sum(sum(r["new_fall_skill"]) for r in rows)
    nf_after = sum(sum(r["new_fall_after"]) for r in rows)
    tgt = a.units * UNIT[a.axis]
    print(f"nudge {a.axis} {a.units:+.2f} units (target {tgt:+.1f} {'deg' if a.axis == 'w' else 'mm'}): n={len(f)}  "
          f"fwd {f.mean():+.1f}±{f.std():.1f} mm  left {l.mean():+.1f}±{l.std():.1f} mm  yaw {y.mean():+.1f}±{y.std():.1f}°  "
          f"handback drift {hb.mean():.1f} mm  fell {fell}/{len(f)} (max tilt p95 {np.percentile(tilt, 95):.0f} deg)  upright {up}/{len(f)}  [{time.time() - t0:.0f}s]\n"
          f"   at the skill's end: fwd {kf.mean():+.1f}±{kf.std():.1f} mm  left {kl.mean():+.1f}±{kl.std():.1f} mm  "
          f"yaw {ky.mean():+.1f}±{ky.std():.1f}°\n"
          f"   new falls {nf_skill + nf_after}/{n_up} nudges started upright (during the skill {nf_skill}, after the hand-back {nf_after})",
          flush=True)
    j = int(torch.argmax(overrun))
    print(f"   commanded targets: worst {act_term._target_names[j]} {float(overrun[j]):+.2f} rad past its limit", flush=True)
    out = {"args": vars(a), "rows": rows, "target": tgt, "overrun_rad": overrun.tolist(),
           "overrun_joint": act_term._target_names[j]}
    Path(a.out).write_text(json.dumps(out, default=float))
    if rec and a.npz:
        np.savez_compressed(a.npz, qpos=np.stack(frames), marks=np.array(marks), dt=sim.dt)


if __name__ == "__main__":
    main()

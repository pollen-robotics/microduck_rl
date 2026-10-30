"""Gearbox wear of a policy, per joint and per phase, measured in the training sim.

Runs one or more ONNX policies in the velstand env (CPU is fine: 32 envs × 20 s
takes a few minutes on a laptop) and reports where the XL330 gear trains get
loaded, using the bench's own dose-response (XL330 gear breakage — bench results,
2026-10-30): life ∝ (load − 0.40 N·m)^−2.4, flat below the plateau, and the hold
after an impact matters as much as the impact. See `mdp.gear_wear`.

What it measures, per servo:
  wear    — Σ gear_wear(peak) over loading EVENTS: control steps where the
            physics-rate gear-load peak (FrictionDRBamActuator) is a local maximum
            above the plateau. A 30-step hold counts once here, and in `hold`.
  hold    — seconds pushing > 0.3 N·m (and > 0.4) at < 0.5 rad/s.
  mean|τ| — to compare with the robot's mean-current ranking (2026-10-29 torture
            test: knees 0.10 A, hip pitch 0.09, neck pitch 0.06, the rest ≤ 0.04).
Phases: `upright` (tilt < 25°, z > 0.09), `fall` (the first 0.75 s after leaving
upright), `ground` (everything else: lying, the rise, a failed rise).

Sim torque is not gear load on the robot (no gear-friction wind-up, contacts are
softer or harder than the real ones), so the numbers RANK policies and phases;
they are not a lifetime.

  uv run python scripts/gear_wear_eval.py --scenario rise                 # v7 vs v8 from the Hub
  uv run python scripts/gear_wear_eval.py --scenario push \\
      --policy v8=hub:v8 --policy new=exported/policy.onnx
  uv run python scripts/gear_wear_eval.py --scenario rise --policy v8=hub:v8 --hold-guard 0.35
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

TASK = "Mjlab-VelStand-Flat-Backlash-MicroDuck"
JOINTS = (
    "l_hip_yaw", "l_hip_roll", "l_hip_pitch", "l_knee", "l_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "r_hip_yaw", "r_hip_roll", "r_hip_pitch", "r_knee", "r_ankle",
)
# Flat 61D actor obs (AGENTS.md): joint_pos (rel. HOME) 6:20, joint_vel 20:34.
JOINT_POS = slice(6, 20)
JOINT_VEL = slice(20, 34)
UPRIGHT_TILT_DEG = 25.0
UPRIGHT_Z = 0.09
FALL_WINDOW_S = 0.75
HOLD_VEL = 0.5
HOLD_TORQUES = (0.3, 0.4)
PHASES = ("upright", "fall", "ground")


def resolve_policy(spec: str) -> tuple[str, str]:
    """`name=path`, `name=hub:<tag>[:file]`, or a bare path/`hub:` ref."""
    name, _, ref = spec.partition("=") if "=" in spec else (None, None, spec)
    if ref.startswith("hub:"):
        from huggingface_hub import hf_hub_download

        parts = ref.split(":")
        tag, file = parts[1], (parts[2] if len(parts) > 2 else "velstand.onnx")
        path = hf_hub_download("pollen-robotics/microduck-policies", file, revision=tag)
        return name or f"{tag}/{file}", path
    return name or Path(ref).stem, ref


def make_env(scenario: str, num_envs: int, seed: int):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    import mjlab_microduck.tasks  # noqa: F401  registers the tasks
    from mjlab_microduck.tasks import microduck_velstand_env_cfg as vs

    cfg = load_env_cfg(TASK, play=True)
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    # The play cfg keeps velstand's curricula, and prone_init_prob would rewrite
    # the spawn mix below back to its step-0 stage on every reset.
    cfg.curriculum = {}
    prone = cfg.events["random_prone_init"].params
    topple = cfg.events["topple_push"]
    if scenario == "rise":
        # The final training mix of fallen starts (prone + crouch + sit), no pushes.
        prone.update(vs.PRONE_RAMP_STAGES[-1]["params"])
        prone["prone_prob"] = 0.8
        prone["crouch_prob"] = 0.2
        cfg.events.pop("topple_push")
    elif scenario == "push":
        # Walking under random commands, toppled every few seconds.
        prone.update({"prone_prob": 0.0, "crouch_prob": 0.0})
        prone.pop("sit_prob", None)
        topple.interval_range_s = (3.0, 5.0)
        topple.params["velocity_range"] = {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}
    elif scenario == "walk":
        prone.update({"prone_prob": 0.0, "crouch_prob": 0.0})
        prone.pop("sit_prob", None)
        cfg.events.pop("topple_push")
    else:
        raise ValueError(scenario)
    return ManagerBasedRlEnv(cfg, device="cpu")


def run_policy(env, onnx_path: str, seconds: float, hold_guard: float | None, guard_joints: str) -> dict:
    from mjlab_microduck.tasks import mdp

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    asset = env.scene["robot"]
    trunk = asset.find_bodies("trunk_base")[0][0]
    guard_mask = np.array([bool(re.search(guard_joints, j)) for j in JOINTS])

    obs, _ = env.reset(seed=env.cfg.seed)
    n, steps, dt = env.num_envs, int(seconds / env.step_dt), env.step_dt
    peak = np.zeros((steps, n, 14), np.float32)
    tau = np.zeros((steps, n, 14), np.float32)
    vel = np.zeros((steps, n, 14), np.float32)
    phase = np.zeros((steps, n), np.int8)
    new_ep = np.zeros((steps, n), bool)
    since_up = np.full(n, 1e9)
    was_up = np.zeros(n, bool)
    fresh = np.ones(n, bool)  # first step of an episode: an upright spawn is not a recovery
    falls = recoveries = 0
    rise_times: list[float] = []
    down_since = np.zeros(n)
    guarded = 0
    for t in range(steps):
        o = obs["actor"].numpy().astype(np.float32)
        act = np.concatenate([sess.run(None, {in_name: o[i:i + 1]})[0] for i in range(n)])
        if hold_guard is not None:
            # What robotd could do at 50 Hz: a joint that is not moving may not be
            # commanded more than hold_guard rad away from where it is.
            q, qd = o[:, JOINT_POS], o[:, JOINT_VEL]
            err = act - q
            clamp = guard_mask & (np.abs(qd) < HOLD_VEL) & (np.abs(err) > hold_guard)
            guarded += int(clamp.sum())
            act = np.where(clamp, q + np.sign(err) * hold_guard, act)
        obs, _, term, trunc, _ = env.step(torch.from_numpy(act))
        done = (term | trunc).numpy()
        peak[t] = mdp.gear_load_peak(env).numpy()
        tau[t] = asset.data.actuator_force.numpy()
        vel[t] = mdp._servo_joint_vel(env, asset).numpy()
        g = asset.data.projected_gravity_b.numpy()
        tilt = np.degrees(np.arccos(np.clip(-g[:, 2], -1, 1)))
        z = (asset.data.body_link_pos_w[:, trunk, 2] - env.scene.env_origins[:, 2]).numpy()
        up = (tilt < UPRIGHT_TILT_DEG) & (z > UPRIGHT_Z)
        since_up = np.where(up, 0.0, since_up + dt)
        phase[t] = np.where(up, 0, np.where(since_up <= FALL_WINDOW_S, 1, 2))
        fell = was_up & ~up & ~fresh
        rose = ~was_up & up & ~fresh
        falls += int(fell.sum())
        recoveries += int(rose.sum())
        rise_times += (t * dt - down_since[rose]).tolist()
        down_since = np.where(fell, t * dt, down_since)
        was_up, fresh = up, np.zeros(n, bool)
        # An episode that ended: the next step starts a new one, down from now.
        new_ep[t] = done
        since_up = np.where(done, 1e9, since_up)
        was_up = was_up & ~done
        fresh = done.copy()
        down_since = np.where(done, (t + 1) * dt, down_since)
    return summarize(peak, tau, vel, phase, new_ep, dt, falls, recoveries, rise_times, guarded)


def summarize(peak, tau, vel, phase, new_ep, dt, falls, recoveries, rise_times, guarded) -> dict:
    from mjlab_microduck.tasks import mdp

    steps, n, _ = peak.shape
    minutes = steps * n * dt / 60.0
    # A loading event: a local maximum in time, not across an episode boundary.
    prev = np.concatenate([np.zeros_like(peak[:1]), peak[:-1]])
    nxt = np.concatenate([peak[1:], np.zeros_like(peak[:1])])
    prev[1:][new_ep[:-1]] = 0.0
    nxt[:-1][new_ep[:-1]] = 0.0
    event = (peak >= prev) & (peak > nxt) & (peak > mdp.GEAR_WEAR_PLATEAU_NM)
    wear = mdp.gear_wear(torch.from_numpy(peak)).numpy() * event
    stalled = np.abs(vel) < HOLD_VEL
    out = {
        "env_minutes": minutes, "falls": falls, "recoveries": recoveries,
        "rise_s_p50": float(np.median(rise_times)) if rise_times else None,
        "rise_s_p90": float(np.percentile(rise_times, 90)) if rise_times else None,
        "guarded_joint_steps": guarded,
        "phase_share": {p: float((phase == i).mean()) for i, p in enumerate(PHASES)},
        "joints": {},
    }
    for j, name in enumerate(JOINTS):
        row = {
            "wear_per_min": float(wear[:, :, j].sum() / minutes),
            "events_per_min": float(event[:, :, j].sum() / minutes),
            "mean_abs_tau": float(np.abs(tau[:, :, j]).mean()),
            "p99_peak": float(np.percentile(peak[:, :, j], 99)),
        }
        for thr in HOLD_TORQUES:
            row[f"hold_s_per_min@{thr}"] = float(((np.abs(tau[:, :, j]) > thr) & stalled[:, :, j]).sum() * dt / minutes)
        row["wear_by_phase"] = {p: float(wear[:, :, j][phase == i].sum() / minutes) for i, p in enumerate(PHASES)}
        row["hold_by_phase@0.3"] = {
            p: float(((np.abs(tau[:, :, j]) > 0.3) & stalled[:, :, j])[phase == i].sum() * dt / minutes)
            for i, p in enumerate(PHASES)
        }
        out["joints"][name] = row
    return out


def print_report(results: dict[str, dict]) -> None:
    names = list(results)
    print()
    for name in names:
        r = results[name]
        print(f"## {name}: {r['env_minutes']:.1f} env-min, falls {r['falls']}, recoveries {r['recoveries']}, "
              f"rise p50/p90 {r['rise_s_p50']}/{r['rise_s_p90']} s, phases "
              + ", ".join(f"{p} {v:.0%}" for p, v in r["phase_share"].items())
              + (f", guard clamps {r['guarded_joint_steps']}" if r["guarded_joint_steps"] else ""))
    head = "| joint | " + " | ".join(f"{n} wear/min | {n} hold>0.3 s/min | {n} mean|τ|" for n in names) + " |"
    print("\n" + head)
    print("|" + "---|" * (1 + 3 * len(names)))
    for j in JOINTS:
        cells = []
        for n in names:
            row = results[n]["joints"][j]
            cells += [f"{row['wear_per_min']:.4f}", f"{row['hold_s_per_min@0.3']:.2f}", f"{row['mean_abs_tau']:.3f}"]
        print(f"| {j} | " + " | ".join(cells) + " |")
    for n in names:
        tot = {p: sum(results[n]["joints"][j]["wear_by_phase"][p] for j in JOINTS) for p in PHASES}
        hold = {p: sum(results[n]["joints"][j]["hold_by_phase@0.3"][p] for j in JOINTS) for p in PHASES}
        print(f"\n{n}: wear/min by phase " + ", ".join(f"{p} {v:.4f}" for p, v in tot.items())
              + " · hold s/min by phase " + ", ".join(f"{p} {v:.2f}" for p, v in hold.items())
              + f" · total wear/min {sum(tot.values()):.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", action="append", help="name=path.onnx or name=hub:<tag>[:file] (repeatable)")
    ap.add_argument("--scenario", choices=("rise", "push", "walk"), default="rise")
    ap.add_argument("--num-envs", type=int, default=32)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hold-guard", type=float, default=None, help="clamp |target − q| to this (rad) on stalled guarded joints")
    ap.add_argument("--guard-joints", default="knee|hip_pitch|neck_pitch")
    ap.add_argument("--out", default=None, help="write the results as JSON here")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    specs = args.policy or ["v7=hub:v7", "v8=hub:v8"]
    results = {}
    env = make_env(args.scenario, args.num_envs, args.seed)
    for spec in specs:
        name, path = resolve_policy(spec)
        if args.hold_guard is not None:
            name += f"+guard{args.hold_guard}"
        print(f"[gear_wear] {name}: {path} ({args.scenario}, {args.num_envs} envs × {args.seconds} s)")
        results[name] = run_policy(env, path, args.seconds, args.hold_guard, args.guard_joints)
    print_report(results)
    if args.out:
        Path(args.out).write_text(json.dumps({"scenario": args.scenario, "args": vars(args), "results": results}, indent=1))
        print(f"\n[gear_wear] wrote {args.out}")


if __name__ == "__main__":
    main()

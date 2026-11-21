#!/usr/bin/env python3
"""Headless sim bench for velstand ONNX policies (flat + backlash, training DR at final stage).

Three batteries, each run on one batched mjlab env with every group of envs holding a fixed command:

  probe  yaw tracking (in place / at vx 0.2), straight-walk drift + vx tracking, standing body tilt / z
  rise   get-up from ±90° spawns, HOME joints and random joints (face-down / face-up / side), 12 s
  push   walking at vx 0.15, horizontal push at t = 2 s (0.8 / 1.0 / 1.2 m/s): fell %, recovered %

Usage:
  uv run scripts/velstand_bench.py logs/bench_onnx/a.onnx logs/bench_onnx/b.onnx [--only probe,rise,push]

Policies are run through a batched torch replica of the exported graph (the normalizer is baked in),
so this measures exactly what ships. Numbers are means over the envs that stayed upright unless noted.
"""

import argparse
import math
import os
import time

import onnx
import torch
from onnx import numpy_helper

import mjlab_microduck.tasks  # noqa: F401  (registers tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab_microduck.robot.microduck_constants import MICRODUCK_ALLCOLLISIONS_BACKLASH_ROBOT_CFG
from mjlab_microduck.tasks import mdp as m
from mjlab_microduck.tasks.backlash import make_backlash_variant
from mjlab_microduck.tasks.microduck_velstand_env_cfg import make_microduck_velstand_env_cfg


class OnnxMLP(torch.nn.Module):
    """Batched torch replica of an exported actor (Sub/Div normalizer + Gemm/Elu)."""

    def __init__(self, path, device):
        super().__init__()
        g = onnx.load(path).graph
        init = {i.name: torch.as_tensor(numpy_helper.to_array(i).copy(), device=device) for i in g.initializer}
        self.ops = []
        for n in g.node:
            attrs = {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}
            self.ops.append((n.op_type, [init.get(i) for i in n.input], attrs))

    @torch.no_grad()
    def forward(self, x):
        for op, ins, at in self.ops:
            if op == "Sub":
                x = x - ins[1]
            elif op == "Div":
                x = x / ins[1]
            elif op == "Gemm":
                w = ins[1].T if at.get("transB", 0) else ins[1]
                x = x @ w + (ins[2] if len(ins) > 2 and ins[2] is not None else 0)
            elif op == "Elu":
                x = torch.nn.functional.elu(x)
            else:
                raise ValueError(op)
        return x


def build_env(num_envs, device, seed):
    cfg = make_microduck_velstand_env_cfg(play=False, rough=False)
    cfg = make_backlash_variant(cfg, MICRODUCK_ALLCOLLISIONS_BACKLASH_ROBOT_CFG)
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    cfg.episode_length_s = 1e5
    for k in ("fallen_too_long", "time_out", "fell_over"):
        cfg.terminations.pop(k, None)
    for k in ("push_robot", "topple_push"):
        cfg.events.pop(k, None)
    for k in ("prone_init_prob", "fell_over_disable", "terrain_levels"):
        cfg.curriculum.pop(k, None)
    for k in [k for k, c in cfg.curriculum.items() if (c.params or {}).get("event_name") in ("push_robot", "topple_push")]:
        cfg.curriculum.pop(k)
    cfg.events["random_prone_init"].params.update(prone_prob=0.0, crouch_prob=0.0)
    for c in cfg.commands.values():
        c.resampling_time_range = (1e6, 1e6)
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    env.common_step_counter = 10**9  # every remaining (DR-range) curriculum at its final stage
    return env


class Bench:
    def __init__(self, env):
        self.env = env
        self.robot = env.scene["robot"]
        self.cm = env.command_manager
        self.twist = self.cm.get_term("twist")
        self.head = self.cm.get_term("head_pose")
        self.body = self.cm.get_term("body_pose")
        self.N, self.dt, self.dev = env.num_envs, env.step_dt, env.device

    def set_commands(self, twist, body):
        """Write fixed commands into the term buffers (resampling is disabled)."""
        tw = self.twist
        tw.vel_command_b[:] = twist
        tw.is_standing_env[:] = (twist.abs().sum(-1) == 0)
        tw.is_heading_env[:] = False
        if hasattr(tw, "is_world_env"):
            tw.is_world_env[:] = False
        self.head._command[:] = 0.0
        self.body._raw[:] = body
        self.body._gate()

    def refresh_obs(self):
        e = self.env
        e.scene.write_data_to_sim()
        e.sim.forward()
        e.sim.sense()
        return e.observation_manager.compute(update_history=True)

    def state(self):
        r = self.robot.data
        g = r.projected_gravity_b
        tilt = torch.rad2deg(torch.acos((-g[:, 2]).clamp(-1, 1)))
        w, x, y, z = r.root_link_quat_w.unbind(-1)
        roll = torch.rad2deg(torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
        pitch = torch.rad2deg(torch.asin((2 * (w * y - z * x)).clamp(-1, 1)))
        hz = r.root_link_pos_w[:, 2] - self.env.scene.env_origins[:, 2]
        return dict(tilt=tilt, roll=roll, pitch=pitch, z=hz, wz=r.root_link_ang_vel_b[:, 2], vx=r.root_link_lin_vel_b[:, 0])


def run_probe(bench, policy, k):
    deg = math.radians
    groups = [("stand", (0, 0, 0), (0,) * 6)]
    for wz in (0.3, 0.6, 1.0):
        for s in (1, -1):
            groups.append((f"yaw inplace {s * wz:+.1f}", (0, 0, s * wz), (0,) * 6))
            groups.append((f"yaw vx0.2 {s * wz:+.1f}", (0.2, 0, s * wz), (0,) * 6))
    for vx in (0.2, 0.3, 0.4):
        groups.append((f"straight vx{vx}", (vx, 0, 0), (0,) * 6))
    for name, i, v in (("roll +10", 3, deg(10)), ("roll -10", 3, -deg(10)), ("pitch +10", 4, deg(10)),
                       ("pitch -10", 4, -deg(10)), ("z -2cm", 2, -0.02)):
        b = [0.0] * 6
        b[i] = v
        groups.append((f"body {name}", (0, 0, 0), tuple(b)))
    N = bench.N
    assert len(groups) * k <= N, f"need {len(groups) * k} envs"
    gid = torch.arange(N, device=bench.dev) // k
    twist = torch.zeros(N, 3, device=bench.dev)
    body = torch.zeros(N, 6, device=bench.dev)
    for i, (_, t, b) in enumerate(groups):
        twist[gid == i] = torch.tensor(t, dtype=torch.float, device=bench.dev)
        body[gid == i] = torch.tensor(b, dtype=torch.float, device=bench.dev)

    env = bench.env
    env.reset()
    bench.set_commands(twist, body)
    obs = bench.refresh_obs()
    T, t0 = int(8.0 / bench.dt), int(3.0 / bench.dt)
    acc = {k_: torch.zeros(N, device=bench.dev) for k_ in ("wz", "vx", "roll", "pitch", "z")}
    fell = torch.zeros(N, dtype=torch.bool, device=bench.dev)
    for step in range(T):
        obs, *_ = env.step(policy(obs["actor"]))
        bench.set_commands(twist, body)
        s = bench.state()
        fell |= s["tilt"] > 60
        if step >= t0:
            for k_ in acc:
                acc[k_] += s[k_]
    n = T - t0
    out = {}
    for i, (name, _, _) in enumerate(groups):
        sel = (gid == i) & ~fell
        r = {k_: (acc[k_][sel] / n).mean().item() if sel.any() else float("nan") for k_ in acc}
        r["fell"] = fell[gid == i].float().mean().item() * 100
        out[name] = r
    base = out["stand"]
    for name, r in out.items():
        r["droll"], r["dpitch"], r["dz_mm"] = r["roll"] - base["roll"], r["pitch"] - base["pitch"], (r["z"] - base["z"]) * 1000
    return out


RISE_FAMILIES = [  # (name, face_down_prob, side_prob, joint_random_prob)
    ("home face-down", 1.0, 0.0, 0.0), ("home face-up", 0.0, 0.0, 0.0), ("home side", 0.0, 1.0, 0.0),
    ("rj face-down", 1.0, 0.0, 1.0), ("rj face-up", 0.0, 0.0, 1.0), ("rj side", 0.0, 1.0, 1.0),
]


def _upright(s):
    return (s["tilt"] < 30) & (s["z"] > 0.08)


def run_rise(bench, policy, k, seconds=12.0):
    env, N, dev = bench.env, bench.N, bench.dev
    assert len(RISE_FAMILIES) * k <= N
    gid = torch.arange(N, device=dev) // k
    env.reset()
    for i, (_, fd, side, jr) in enumerate(RISE_FAMILIES):
        ids = (gid == i).nonzero().flatten()
        m.maybe_set_random_prone_orientation(env, ids, prone_prob=1.0, face_down_prob=fd, side_prob=side,
                                             prone_z_min=0.05, prone_z_max=0.09, joint_random_prob=jr, joint_range_frac=0.9)
        env.sim.data.qvel[ids] = 0.0
    zero = torch.zeros(N, 3, device=dev)
    bench.set_commands(zero, torch.zeros(N, 6, device=dev))
    obs = bench.refresh_obs()
    T = int(seconds / bench.dt)
    t_up = torch.full((N,), float("nan"), device=dev)  # first time upright for the rest of the run
    up_since = torch.zeros(N, device=dev)
    peak_w = torch.zeros(N, device=dev)
    ever_rose = torch.zeros(N, dtype=torch.bool, device=dev)
    for step in range(T):
        obs, *_ = env.step(policy(obs["actor"]))
        bench.set_commands(zero, torch.zeros(N, 6, device=dev))
        s = bench.state()
        up = _upright(s)
        ever_rose |= s["tilt"] < 45
        up_since = torch.where(up, up_since, torch.full_like(up_since, (step + 1) * bench.dt))
        peak_w = torch.maximum(peak_w, bench.robot.data.root_link_ang_vel_b.norm(dim=-1))
    end_up = _upright(bench.state())
    out = {}
    for i, (name, *_ ) in enumerate(RISE_FAMILIES):
        g = gid == i
        rec = end_up[g]
        t = up_since[g][rec]
        out[name] = dict(rec=rec.float().mean().item() * 100, never_rose=(~ever_rose[g]).float().mean().item() * 100,
                         t_rec_p50=t.median().item() if rec.any() else float("nan"),
                         t_rec_p90=t.quantile(0.9).item() if rec.any() else float("nan"),
                         peak_w_p50=peak_w[g].median().item())
    return out


def run_push(bench, policy, k, seconds=12.0, speeds=(0.8, 1.0, 1.2), vx=0.15):
    env, N, dev = bench.env, bench.N, bench.dev
    assert len(speeds) * k <= N
    gid = torch.arange(N, device=dev) // k
    env.reset()
    twist = torch.zeros(N, 3, device=dev)
    twist[:, 0] = vx
    body = torch.zeros(N, 6, device=dev)
    bench.set_commands(twist, body)
    obs = bench.refresh_obs()
    push_v = torch.zeros(N, device=dev)
    for i, v in enumerate(speeds):
        push_v[gid == i] = v
    push_v[gid >= len(speeds)] = 0.0
    ang = torch.rand(N, device=dev) * 2 * math.pi
    T, tp = int(seconds / bench.dt), int(2.0 / bench.dt)
    fell = torch.zeros(N, dtype=torch.bool, device=dev)
    pre_fell = torch.zeros(N, dtype=torch.bool, device=dev)
    for step in range(T):
        if step == tp:
            pre_fell = fell.clone()
            env.sim.data.qvel[:, 0] += push_v * torch.cos(ang)
            env.sim.data.qvel[:, 1] += push_v * torch.sin(ang)
        obs, *_ = env.step(policy(obs["actor"]))
        bench.set_commands(twist, body)
        fell |= bench.state()["tilt"] > 60
    end_up = _upright(bench.state())
    out = {}
    for i, v in enumerate(speeds):
        g = (gid == i) & ~pre_fell
        f = fell[g]
        out[f"push {v} m/s"] = dict(fell=f.float().mean().item() * 100,
                                   rec_of_fell=end_up[g][f].float().mean().item() * 100 if f.any() else float("nan"),
                                   up_at_end=end_up[g].float().mean().item() * 100, pre_push_fell=pre_fell[gid == i].float().mean().item() * 100)
    return out


def fmt_table(title, cols, results, names):
    rows = list(next(iter(results.values())).keys())
    w = max(len(r) for r in rows) + 2
    print(f"\n## {title}")
    print("| " + " | ".join(["case".ljust(w)] + [n for n in names]) + " |")
    print("|" + "---|" * (len(names) + 1))
    for r in rows:
        cells = []
        for n in names:
            d = results[n][r]
            cells.append(" ".join(f"{c}={d[c]:.2f}" if abs(d[c]) < 10 else f"{c}={d[c]:.0f}" for c in cols(r)))
        print("| " + " | ".join([r.ljust(w)] + cells) + " |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx", nargs="+")
    ap.add_argument("--only", default="probe,rise,push")
    ap.add_argument("--envs-per-group", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    only = set(args.only.split(","))
    k = args.envs_per_group
    n_env = max(22 * k if "probe" in only else 0, len(RISE_FAMILIES) * k if "rise" in only else 0, 3 * k if "push" in only else 0)
    torch.manual_seed(args.seed)
    env = build_env(n_env, args.device, args.seed)
    bench = Bench(env)
    names = [os.path.splitext(os.path.basename(p))[0] for p in args.onnx]
    res = {"probe": {}, "rise": {}, "push": {}}
    for p, name in zip(args.onnx, names):
        pol = OnnxMLP(p, args.device)
        t0 = time.time()
        for b in ("probe", "rise", "push"):
            if b in only:
                torch.manual_seed(args.seed)  # same spawns / push directions for every policy
                res[b][name] = {"probe": run_probe, "rise": run_rise, "push": run_push}[b](bench, pol, k)
        print(f"[bench] {name} done in {time.time() - t0:.0f}s", flush=True)

    def probe_cols(r):
        if r.startswith("yaw"):
            return ["wz", "fell"]
        if r.startswith("straight"):
            return ["vx", "wz", "fell"]
        if r.startswith("body"):
            return ["droll", "dpitch", "dz_mm", "fell"]
        return ["roll", "pitch", "z", "fell"]

    if "probe" in only:
        fmt_table("probe (wz rad/s, vx m/s, d* = vs zero-command stand in deg / mm, fell %)", probe_cols, res["probe"], names)
    if "rise" in only:
        fmt_table("rise (12 s, rec % upright at end, never_rose %, t_rec s, peak |w| rad/s)",
                  lambda r: ["rec", "never_rose", "t_rec_p50", "t_rec_p90", "peak_w_p50"], res["rise"], names)
    if "push" in only:
        fmt_table("push while walking vx 0.15 (%)", lambda r: ["fell", "rec_of_fell", "up_at_end"], res["push"], names)


if __name__ == "__main__":
    main()

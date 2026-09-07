"""Find a HOME pose the boot-equipped robot holds PASSIVELY.

The robot topples at HOME_FRAME with the boots on: tip margin 4.4 deg, head sag
5-6 deg. A trained policy balances it fine; this is about the unpowered-brain
case -- put the robot down, it stays -- which init needs and which gives the
hop task a better starting stance.

Search over five sagittal offsets from HOME_FRAME (legs mirrored L/R), score
each candidate on the FRACTION of domain-randomised replicas still upright and
body-clear after 6 s, tie-broken by how level and centred it settles. Replicas
are the point: one env per candidate with DR on optimises the draw, not the
pose -- the mistake yesterday's open-loop probe made twice.

    uv run python scripts/find_stable_home.py --iters 8
    uv run python scripts/find_stable_home.py --replay      # verify the winner
"""
from __future__ import annotations
import argparse, os
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("HOP_NO_PUSH", "1")
import numpy as np, torch
import mjlab_microduck.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

L_HIP, L_KNEE, L_ANKLE, NECK, HEAD = 2, 3, 4, 5, 6
R_HIP, R_KNEE, R_ANKLE = 11, 12, 13
MIRROR = -1.0
PARAMS = (("d_hip_pitch", -0.40, 0.40), ("d_knee", -0.50, 0.50), ("d_ankle", -0.40, 0.40),
          ("d_neck_pitch", -0.60, 0.60), ("d_head_pitch", -0.60, 0.60))
NAMES = [p[0] for p in PARAMS]
LO = np.array([p[1] for p in PARAMS], np.float32); HI = np.array([p[2] for p in PARAMS], np.float32)
TILT_OK = np.radians(15.0)


def pose_actions(p, n_act):
    """Constant action = the candidate offsets, mirrored onto both legs."""
    n = p.shape[0]; a = np.zeros((n, n_act), np.float32)
    a[:, L_HIP], a[:, L_KNEE], a[:, L_ANKLE] = p[:, 0], p[:, 1], p[:, 2]
    a[:, R_HIP], a[:, R_KNEE], a[:, R_ANKLE] = MIRROR*p[:, 0], MIRROR*p[:, 1], MIRROR*p[:, 2]
    a[:, NECK], a[:, HEAD] = p[:, 3], p[:, 4]
    return a


def make_env(task, n, device):
    cfg = load_env_cfg(task); cfg.scene.num_envs = n
    cfg.terminations = type(cfg.terminations)()   # measured directly, no resets
    env = ManagerBasedRlEnv(cfg=cfg, device=device); env.extras.setdefault("log", {})
    model = load_env_cfg(task).scene.entities["robot"].spec_fn().compile()
    mass = torch.tensor(model.body_mass.copy(), device=device, dtype=torch.float32)
    return env, cfg.sim.mujoco.timestep * cfg.decimation, mass


def evaluate(env, dt, mass, p_env, duration):
    """p_env: one parameter row per env. Returns per-env (upright, pitch, com_x_off, height)."""
    n = p_env.shape[0]; dev = env.device; robot = env.scene["robot"]; body = env.scene["body_ground_contact"]
    a = torch.as_tensor(pose_actions(p_env, env.action_manager.total_action_dim), device=dev)
    env.reset()
    touched = torch.zeros(n, dtype=torch.bool, device=dev)
    for _ in range(int(duration / dt)):
        env.step(a)
        f = body.data.found
        touched |= (f.sum(-1) > 0) if f is not None and f.dim() > 1 else torch.zeros_like(touched)
    g = robot.data.projected_gravity_b.float()
    tilt = torch.arccos(torch.clamp(-g[:, 2], -1, 1))
    pitch = torch.atan2(g[:, 0], -g[:, 2])
    upright = (tilt < TILT_OK) & ~touched
    bz = robot.data.body_link_pos_w[:, :, 2].float(); mm = mass.reshape(1, -1)[:, :bz.shape[1]]
    com_z = (bz * mm).sum(1) / mm.sum()
    bx = robot.data.body_link_pos_w[:, :, 0].float()
    com_x = (bx * mm).sum(1) / mm.sum()
    # foot centre x: mean of the two pad bodies -- find them by name once
    names = [robot.body_names[i] if hasattr(robot, "body_names") else None for i in range(bz.shape[1])]
    idx = [i for i, nm in enumerate(names) if nm and nm.endswith("_foot_pad")]
    foot_x = bx[:, idx].mean(1) if idx else torch.zeros_like(com_x)
    return (upright.cpu().numpy(), pitch.cpu().numpy(), (com_x - foot_x).cpu().numpy(),
            (com_z - robot.data.root_link_pos_w[:, 2].float() * 0).cpu().numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="Mjlab-Hop-InPlaceSym-K3344-MicroDuck")
    ap.add_argument("--cands", type=int, default=64); ap.add_argument("--reps", type=int, default=16)
    ap.add_argument("--iters", type=int, default=8); ap.add_argument("--duration", type=float, default=6.0)
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", default="stable_home_best.npz")
    ap.add_argument("--replay", action="store_true")
    args = ap.parse_args()

    if args.replay:
        d = np.load(args.out, allow_pickle=True); p = d["params"].astype(np.float32)
        n = 128
        env, dt, mass = make_env(args.task, n, args.device)
        for lab, pp in (("HOME_FRAME (baseline)", np.zeros_like(p)), ("candidate", p)):
            up, pitch, cx, _ = evaluate(env, dt, mass, np.repeat(pp[None], n, 0), args.duration)
            print(f"RESULT {lab:22s} upright {up.mean()*100:5.1f}%  settled pitch "
                  f"{np.degrees(pitch[up]).mean() if up.any() else float('nan'):+5.2f} deg  "
                  f"CoM-x vs feet {cx[up].mean()*1000 if up.any() else float('nan'):+5.1f} mm")
        for nm, v in zip(NAMES, p): print(f"RESULT   {nm:13s} {v:+.4f} rad ({np.degrees(v):+.1f} deg)")
        return

    n = args.cands * args.reps
    env, dt, mass = make_env(args.task, n, args.device)
    mu = np.zeros(len(PARAMS), np.float32); sigma = (HI - LO) / 4
    best, best_p = -1.0, None
    print(f"[home] {args.cands} candidates x {args.reps} DR replicas, {args.duration}s each", flush=True)
    for it in range(args.iters):
        cand = np.clip(np.random.normal(mu, sigma, (args.cands, len(PARAMS))), LO, HI).astype(np.float32)
        if it == 0: cand[0] = 0.0                       # always evaluate HOME itself
        p_env = np.repeat(cand, args.reps, axis=0)
        up, pitch, cx, _ = evaluate(env, dt, mass, p_env, args.duration)
        up = up.reshape(args.cands, args.reps); pitch = pitch.reshape(args.cands, args.reps)
        cx = cx.reshape(args.cands, args.reps)
        frac = up.mean(1)
        lvl = np.array([np.abs(pitch[i][up[i]]).mean() if up[i].any() else np.pi for i in range(args.cands)])
        score = frac - 0.05 * np.degrees(lvl) / 10.0     # upright first; levelness as tie-break
        order = np.argsort(-score); el = cand[order[:max(6, args.cands // 8)]]
        mu, sigma = el.mean(0), el.std(0) + 0.01
        if score[order[0]] > best: best, best_p = float(score[order[0]]), cand[order[0]].copy()
        print(f"[home] iter {it}  best upright {frac[order[0]]*100:5.1f}%  pitch "
              f"{np.degrees(lvl[order[0]]):+5.2f}  HOME upright {frac[0]*100 if it==0 else float('nan'):5.1f}%  "
              f"elite-mean upright {frac[order[:8]].mean()*100:5.1f}%", flush=True)
    np.savez(args.out, params=best_p, names=np.array(NAMES), score=best)
    print("\nRESULT best pose offsets from HOME_FRAME:")
    for nm, v in zip(NAMES, best_p): print(f"RESULT   {nm:13s} {v:+.4f} rad ({np.degrees(v):+.1f} deg)")
    print(f"RESULT wrote {args.out} -- run --replay to verify")


if __name__ == "__main__":
    main()

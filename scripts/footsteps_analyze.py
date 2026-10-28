"""Is the free gait predictable enough to place feet through velocity commands? (footsteps_sample.py output)

  uv run python scripts/footsteps_analyze.py footsteps.npz [--plot out.png]

Footsteps are mirrored to a single "left lands" convention (right landings: dy, dθ, cmd vy, cmd wz negated) so
both sides pool into one model. Reports, in mm / deg:
  1. noise floor   — scatter of footsteps under one constant command (steady envs, per-env std)
  2. map error     — residual of cmd → footstep models (linear like Sigmaban's mapper, quadratic) on steady envs,
                     held-out half; this is how far the foot lands from where the map says it will
  3. response lag  — on switch envs (new cmd at every touchdown): is a footstep explained by the command
                     issued for that swing, or by the one before it?
  4. envelope      — reachable footstep percentiles + feet spacing + cadence (FootstepNet limits)
"""

import argparse

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("npz")
ap.add_argument("--plot", default=None)
A = ap.parse_args()
d = np.load(A.npz)

fs = d["footstep"].astype(np.float64).copy()
cmd = d["cmd"].astype(np.float64).copy()
cmd_prev = d["cmd_prev"].astype(np.float64).copy()
right = d["landing"] == 1
for a in (fs, ):
    a[right, 1:] *= -1  # dy, dθ
for a in (cmd, cmd_prev):
    a[right, 1:] *= -1  # vy, wz
switch = d["switch"].astype(bool)
env = d["env"]
since = d["since_last"]
MM = np.array([1000.0, 1000.0, np.degrees(1.0)])
NAMES = ("dx mm", "dy mm", "dθ deg")


def fmt(v):
    return "  ".join(f"{n} {x:6.1f}" for n, x in zip(NAMES, v * MM))


def feats(c, order):
    vx, vy, wz = c.T
    cols = [np.ones_like(vx), vx, vy, wz]
    if order >= 2:
        cols += [vx * vx, vy * vy, wz * wz, vx * vy, vx * wz, vy * wz]
    if order >= 3:
        cols += [vx ** 3, vy ** 3, wz ** 3, vx * vx * wz, vx * wz * wz, vx * vx * vy, vy * wz * wz]
    return np.stack(cols, -1)


def fit(c, y, order):
    return np.linalg.lstsq(feats(c, order), y, rcond=None)[0]


def rms(e):
    return np.sqrt((e ** 2).mean(0))


st = ~switch
print(f"{len(fs)} footsteps: {st.sum()} steady, {switch.sum()} switch   ({len(np.unique(env[st]))} steady envs)")

# ── 1. noise floor ──
stds = []
for e in np.unique(env[st]):
    for side in (False, True):
        m = st & (env == e) & (right == side)
        if m.sum() >= 4:
            stds.append(fs[m].std(0))
stds = np.array(stds)
print("\n1. NOISE FLOOR — per-env footstep std under a constant command (median / p90 over envs)")
print("   median ", fmt(np.median(stds, 0)))
print("   p90    ", fmt(np.percentile(stds, 90, 0)))

# ── 2. map error (held-out on steady envs, split by env) ──
rng = np.random.default_rng(0)
envs = np.unique(env[st])
test_envs = set(rng.choice(envs, len(envs) // 2, replace=False).tolist())
te = st & np.isin(env, list(test_envs))
tr = st & ~te
print("\n2. MAP ERROR — cmd → footstep, fit on half the steady envs, RMS residual on the other half")
models = {}
for order in (1, 2, 3):
    W = fit(cmd[tr], fs[tr], order)
    models[order] = W
    print(f"   order {order}  ", fmt(rms(feats(cmd[te], order) @ W - fs[te])))
# noise floor counterpart: residual vs per-env mean (best any cmd-only map can do)
groups = [te & (env == e) & (right == side) for e in test_envs for side in (False, True)]
res = np.concatenate([fs[g] - fs[g].mean(0) for g in groups if g.sum() > 1])
print("   floor    ", fmt(rms(res)), " (residual vs per-env, per-side mean: what no cmd → footstep map can beat)")
W1 = models[1]
print("   linear model (rows: bias, vx, vy, wz → cols dx, dy, dθ):")
for name, row in zip(("bias", "vx", "vy", "wz"), W1):
    print(f"     {name:5s} " + "  ".join(f"{x:+.4f}" for x in row))

# ── 3. response lag ──
if switch.any():
    W = models[2]
    sw = switch & np.isfinite(since) & (since < 2.0)
    e_now = rms(feats(cmd[sw], 2) @ W - fs[sw])
    e_prev = rms(feats(cmd_prev[sw], 2) @ W - fs[sw])
    W_mix = np.linalg.lstsq(np.concatenate([feats(cmd[sw], 1), feats(cmd_prev[sw], 1)[:, 1:]], -1), fs[sw], rcond=None)[0]
    e_mix = rms(np.concatenate([feats(cmd[sw], 1), feats(cmd_prev[sw], 1)[:, 1:]], -1) @ W_mix - fs[sw])
    print("\n3. RESPONSE LAG — switch envs (cmd resampled at every touchdown), steady order-2 map")
    print("   predict from cmd of THIS swing   ", fmt(e_now))
    print("   predict from cmd of PREVIOUS one ", fmt(e_prev))
    print("   linear fit on both (this+prev)   ", fmt(e_mix))
    share = np.abs(W_mix[1:4]).sum(0) / (np.abs(W_mix[1:4]).sum(0) + np.abs(W_mix[4:7]).sum(0))
    print("   weight share of THIS swing's cmd ", "  ".join(f"{n.split()[0]} {s:.2f}" for n, s in zip(NAMES, share)))

# ── 4. envelope + cadence ──
s_ok = np.isfinite(since) & (since < 2.0)
print("\n4. ENVELOPE (mirrored to left-lands), steady envs")
for q in (1, 5, 50, 95, 99):
    print(f"   p{q:<3d}", fmt(np.percentile(fs[st], q, 0)))
print(f"   feet spacing (median |dy| at |cmd vy| < 0.05, |wz| < 0.2): "
      f"{1000 * np.median(fs[st & (np.abs(cmd[:, 1]) < 0.05) & (np.abs(cmd[:, 2]) < 0.2), 1]):.1f} mm")
print(f"   neutral dθ (same subset): {np.degrees(np.median(fs[st & (np.abs(cmd[:, 1]) < 0.05) & (np.abs(cmd[:, 2]) < 0.2), 2])):.1f} deg")
p = since[s_ok & st]
print(f"   step period p10/50/90: {np.percentile(p, 10):.3f} / {np.median(p):.3f} / {np.percentile(p, 90):.3f} s")
speed = np.linalg.norm(cmd[:, :2], axis=1)
for lo, hi in ((0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.6)):
    m = s_ok & st & (speed >= lo) & (speed < hi)
    if m.any():
        print(f"     |v| {lo:.1f}-{hi:.1f}: period median {np.median(since[m]):.3f} s  (cv {since[m].std() / since[m].mean():.2f}, n={m.sum()})")

if A.plot:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    for i, (ci, lab) in enumerate(((0, "cmd vx"), (1, "cmd vy"), (2, "cmd wz"))):
        small = np.array([0.08, 0.08, 0.2])  # "others ≈ 0" band per command axis
        others = [j for j in range(3) if j != ci]
        m = st & (np.abs(cmd[:, others]) < small[others]).all(1)
        ax[i].scatter(cmd[m, ci], fs[m, i] * MM[i], s=2, alpha=0.4)
        xs = np.linspace(cmd[:, ci].min(), cmd[:, ci].max(), 50)
        c = np.zeros((50, 3)); c[:, ci] = xs
        for order in (1, 2):
            ax[i].plot(xs, (feats(c, order) @ models[order])[:, i] * MM[i], label=f"order {order}")
        ax[i].set_xlabel(lab + " (others ≈ 0)"); ax[i].set_ylabel(NAMES[i]); ax[i].grid(); ax[i].legend()
    plt.tight_layout(); plt.savefig(A.plot, dpi=110)
    print(f"\nplot → {A.plot}")

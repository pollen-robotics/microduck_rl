"""Watch / evaluate a trained microduck FootstepNet.

  uv run python enjoy.py [MODEL.zip | RUN_DIR] --provisional-limits FWD BWD DY DTHETA_DEG [--episodes 20] [--headless]

MODEL defaults to the newest logs/crossq/<env>_*/ run, taking best_model.zip, else the latest rl_model_*_steps.zip.
The limits MUST be the ones the model was trained with (they shape the action space). The window shows the
footsteps (older = fainter), the goal foot (outline) and the obstacle; --headless prints success rate / step counts
only (no display needed).
"""

import argparse
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("model", nargs="?", default=None)
ap.add_argument("--provisional-limits", type=float, nargs=4, metavar=("FWD", "BWD", "DY", "DTHETA_DEG"))
ap.add_argument("--episodes", type=int, default=20)
ap.add_argument("--headless", action="store_true")
ap.add_argument("--zoom", type=float, default=350.0, help="pixels per metre (FootstepNet default 200)")
ap.add_argument("--pause", type=float, default=0.15, help="seconds between steps in the window")
ap.add_argument("--seed", type=int, default=0)
A = ap.parse_args()
if A.headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

import microduck_env  # noqa: E402
from sbx import CrossQ  # noqa: E402

if A.provisional_limits:
    microduck_env.set_limits(*A.provisional_limits)


def find_model(arg):
    p = Path(arg) if arg else None
    if p is not None and p.is_file():
        return p
    if p is None:
        runs = sorted(Path("logs/crossq").glob(f"{microduck_env.ENV_ID}_*"), key=lambda d: d.stat().st_mtime)
        if not runs:
            sys.exit("no run in logs/crossq/ — pass a model .zip")
        p = runs[-1]
    if (p / "best_model.zip").exists():
        return p / "best_model.zip"
    ckpts = sorted(p.glob("rl_model_*_steps.zip"), key=lambda f: int(re.search(r"_(\d+)_steps", f.name).group(1)))
    if ckpts:
        return ckpts[-1]
    if (p / f"{microduck_env.ENV_ID}.zip").exists():  # final model of a finished run
        return p / f"{microduck_env.ENV_ID}.zip"
    sys.exit(f"no model in {p}")


path = find_model(A.model)
print(f"model: {path}\nlimits: {microduck_env.contract.footstepnet_options()}")
env = microduck_env.MicroduckFootstepsEnv()
env.simulator.pixels_per_meter = A.zoom
model = CrossQ.load(path)

succ, steps = [], []
for ep in range(A.episodes):
    obs, _ = env.reset(seed=A.seed + ep)
    for t in range(microduck_env.MAX_EPISODE_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        if not A.headless:
            env.simulator.render()
            time.sleep(A.pause)
        if terminated:
            break
    succ.append(terminated)
    steps.append(t + 1)
    print(f"episode {ep:3d}: {'reached' if terminated else 'FAILED '} in {t + 1:2d} steps")
    if not A.headless:
        time.sleep(1.0)

s = np.array(succ)
print(f"\nsuccess {100 * s.mean():.0f}% ({s.sum()}/{len(s)}), steps when reached: "
      f"mean {np.mean(np.array(steps)[s]) if s.any() else math.nan:.1f}")

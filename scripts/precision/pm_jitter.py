"""Tick-to-tick target change (mrad) of the walking policy standing / walking and of a nudge policy (whole skill, settle
part): the shakiness Rémi saw on the real duck. PYTHONPATH=src .venv-mjlab/bin/python scripts/precision/pm_jitter.py [NUDGE.onnx]"""
import sys, numpy as np, torch
sys.path.insert(0, "scripts/precision")
from pm_env import Sim, Policy
s = Sim(n=16, seed=6)
t = s.env.action_manager.get_term("joint_pos")
nudge = Policy(sys.argv[1] if len(sys.argv) > 1 else "/Users/remi/microduck/notes/precision-mode/policies/nudge-candidate.onnx")
def run(label, ticks, cmd, pol=None, smooth=True):
    prev, d, raw_prev, draw = None, [], None, []
    for k in range(ticks):
        c = cmd(k)
        s.step(c, policy=pol, smooth=smooth)
        tg = t._processed_actions.clone(); ra = s.obs[:, 34:48].clone()
        if prev is not None:
            d.append((tg - prev).abs().mean(dim=1)); draw.append((ra - raw_prev).abs().mean(dim=1))
        prev, raw_prev = tg, ra
    d = torch.stack(d); draw = torch.stack(draw)
    print(f"{label:34s} mean |target change| per tick {1000 * d.mean():6.1f} mrad   raw action change {1000 * draw.mean():6.1f} mrad")
for _ in range(100): s.step([0, 0, 0])
run("velstand standing", 100, lambda k: [0, 0, 0])
run("velstand walking 0.3 m/s", 100, lambda k: [0.3, 0, 0])
for _ in range(100): s.step([0, 0, 0])
run("nudge fwd (skill, 1.8 s)", 90, lambda k: [0.3, 0, 0] if k < 25 else [0, 0, 0], pol=nudge, smooth=False)
run("nudge settle part only (zeros)", 60, lambda k: [0, 0, 0], pol=nudge, smooth=False)

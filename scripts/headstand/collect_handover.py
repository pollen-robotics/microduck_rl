"""Record the states a policy hands over, to train the next policy from them.

    uv run scripts/headstand/collect_handover.py --task Mjlab-HeadstandFold-Flat-MicroDuck \\
        --checkpoint logs/rsl_rl/microduck_headstand_fold/<run>/model_1000.pt \\
        --condition pike --n 512 --out handover_pike_from_fold.npz

Runs the policy from the task's own spawns, waits until `condition` (pike,
headstand or standing) has held for --hold-s, judged by the same contact gates
the tasks train against, and saves the full qpos (base
pose + 14 joints) and qvel at that moment, with the task, the checkpoint and
a hash of the robot XML. src/mjlab_microduck/tasks/handover_pike_from_fold.npz
is the handover set the kick-up tasks draw from with handover_prob > 0, so they
from the pikes the fold policy produces rather than the one measured resting
pike. CPU is enough: 64 envs, a few minutes for 512 states.
"""
import argparse
import hashlib
import math
from dataclasses import asdict

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from rsl_rl.runners import OnPolicyRunner

from mjlab_microduck.robot.microduck_constants import MICRODUCK_ALLCOLLISIONS_XML
from mjlab_microduck.tasks import mdp as m


def _contacts(env) -> dict:
    """What each env touches the floor with. feet is a COUNT (0, 1, 2), because
    the pike needs both and a one-foot pose passes any check that collapses
    them; the task's own gate counts them too."""
    head = m._sensor_any_contact(env, m._HEADSTAND_HEAD_SENSOR)
    other = m._sensor_any_contact(env, m._HEADSTAND_OTHER_SENSOR)
    found = env.scene.sensors[m._HEADSTAND_FEET_SENSOR].data.found
    per_foot = found.view(found.shape[0], -1) > 0
    if per_foot.shape[-1] != 2:
        raise ValueError(f"the feet sensor must have one slot per foot, got {per_foot.shape[-1]}")
    return {"head": head.cpu().numpy(), "feet": per_foot.sum(dim=-1).cpu().numpy(),
            "other": other.cpu().numpy()}


def condition_met(env, asset, condition: str) -> np.ndarray:
    """The same gates the tasks train against, not the trunk angle alone."""
    c = _contacts(env)
    inv = m._inverted_cos(asset).cpu().numpy()
    if condition == "pike":
        nose = m._nose_up(asset).cpu().numpy()
        pitch = np.degrees(m._trunk_pitch(asset).cpu().numpy())
        return c["head"] & (c["feet"] == 2) & ~c["other"] & (nose < -0.3) & (pitch >= 60) & (pitch <= 95)
    if condition == "headstand":
        return (inv > math.cos(math.radians(35))) & c["head"] & (c["feet"] == 0) & ~c["other"]
    return (c["feet"] == 2) & ~c["head"] & ~c["other"] & (inv < -math.cos(math.radians(30)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True)
    p.add_argument("--checkpoint", required=True, help="path to a model_*.pt of that task")
    p.add_argument("--condition", default="pike", choices=("pike", "headstand", "standing"))
    p.add_argument("--hold-s", type=float, default=0.3)
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--envs", type=int, default=64)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    rl_cfg = load_rl_cfg(args.task)
    cfg = load_env_cfg(args.task, play=True)
    cfg.scene.num_envs = args.envs
    cfg.curriculum.clear()
    for name in list(cfg.terminations):
        if name != "time_out":
            del cfg.terminations[name]
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    wrapped = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)
    runner = (load_runner_cls(args.task) or OnPolicyRunner)(wrapped, asdict(rl_cfg), device="cpu")
    runner.load(args.checkpoint, map_location="cpu")
    policy = runner.get_inference_policy(device="cpu")
    asset = env.scene["robot"]
    n = args.envs
    qpos_bank, qvel_bank = [], []
    while len(qpos_bank) < args.n:
        obs, _ = wrapped.reset()
        held = np.zeros(n)
        taken = np.zeros(n, dtype=bool)
        with torch.no_grad():
            for _ in range(150):
                obs, *_ = wrapped.step(policy(obs))
                ok = condition_met(env, asset, args.condition)
                held = np.where(ok, held + env.step_dt, 0.0)
                ready = (held >= args.hold_s) & ~taken
                if ready.any():
                    q = env.sim.data.qpos.cpu().numpy()
                    v = env.sim.data.qvel.cpu().numpy()
                    for j in np.where(ready)[0]:
                        qpos_bank.append(q[j].copy())
                        qvel_bank.append(v[j].copy())
                        taken[j] = True
        print(f"collected {len(qpos_bank)}", flush=True)
    qpos_bank = np.array(qpos_bank[: args.n])
    qvel_bank = np.array(qvel_bank[: args.n])
    # Base x/y are not part of the state; zero them so a spawn lands at its env's origin.
    qpos_bank[:, 0:2] = 0.0
    # Provenance: the task, the checkpoint and a hash of the robot XML the
    # states were recorded on; the loader refuses a set from another XML.
    xml_sha = hashlib.sha256(open(MICRODUCK_ALLCOLLISIONS_XML, "rb").read()).hexdigest()
    np.savez(args.out, qpos=qpos_bank, qvel=qvel_bank, task=np.array(args.task), checkpoint=np.array(args.checkpoint), xml_sha256=np.array(xml_sha))
    print(f"saved {args.out}: qpos {qpos_bank.shape}, qvel {qvel_bank.shape}, trunk z median {np.median(qpos_bank[:, 2]):.3f}")


if __name__ == "__main__":
    main()

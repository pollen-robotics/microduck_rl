"""Dump per-step hardware signals + policy I/O for the rig visualizer.

Runs one rollout with a checkpoint and records, per policy step:
  - raw joint positions / velocities (the "hardware" signals the XL330s report)
  - the exact 61D actor obs vector (post encoder-bias, lag, noise, normalization
    is applied inside the policy's normalizer — obs here is pre-normalization)
  - the policy's action output (14D joint position targets)

Usage:
  uv run scripts/dump_rig_signals.py --checkpoint logs/.../model_12250.pt \
      --out rig_data.json --seconds 12

`dump_signals()` is importable — make_rig_page.py bakes it straight into the
interactive HTML page without a JSON round-trip.
"""

import argparse
import json
from dataclasses import asdict

import torch

import mjlab.tasks  # noqa: F401
import mjlab_microduck.tasks  # noqa: F401

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

DEFAULT_TASK = "Mjlab-GroundPick-Flat-MicroDuck"


def dump_signals(
  task: str, checkpoint: str, seconds: float, device: str, seed: int = 0
) -> dict:
  """Roll out `checkpoint` for `seconds` and return the rig-visualizer payload."""
  configure_torch_backends()
  torch.manual_seed(seed)

  env_cfg = load_env_cfg(task, play=True)
  agent_cfg = load_rl_cfg(task)
  env_cfg.scene.num_envs = 1
  # Pin the command cycle to phase 0 at every reset so rollouts start aligned
  # (matches runtime behavior; not all tasks have a phase-randomized twist).
  try:
    twist = env_cfg.commands["twist"]
  except (KeyError, TypeError):
    twist = None
  if twist is not None and hasattr(twist, "randomize_phase"):
    twist.randomize_phase = False
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)

  robot = env.scene["robot"]
  joint_names = list(robot.joint_names)
  n_joints = len(joint_names)
  print(f"[INFO] {n_joints} joints: {joint_names}")

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(
    checkpoint, load_cfg={"actor": True}, strict=True, map_location=device
  )
  policy = runner.get_inference_policy(device=device)

  fps = round(1.0 / env.unwrapped.step_dt)
  n_steps = int(seconds * fps)

  rec = {"t": [], "joint_pos": [], "joint_vel": [], "obs": [], "action": []}
  env.reset()
  for i in range(n_steps):
    obs = env.get_observations()
    with torch.no_grad():
      actions = policy(obs)
    env.step(actions)
    rec["t"].append(i / fps)
    rec["joint_pos"].append(robot.data.joint_pos[0].cpu().tolist())
    rec["joint_vel"].append(robot.data.joint_vel[0].cpu().tolist())
    rec["obs"].append(obs["actor"][0].cpu().tolist())
    rec["action"].append(actions[0].cpu().tolist())
    if (i + 1) % fps == 0:
      print(f"  {(i + 1) // fps}/{seconds:.0f}s", flush=True)

  env.close()
  return {
    "fps": fps,
    "joint_names": joint_names,
    "obs_terms": [
      {"name": "base_ang_vel", "dim": 3},
      {"name": "projected_gravity", "dim": 3},
      {"name": "joint_pos", "dim": n_joints},
      {"name": "joint_vel", "dim": n_joints},
      {"name": "actions", "dim": n_joints},
      {"name": "command", "dim": 3},
      {"name": "head_command", "dim": 4},
      {"name": "body_command", "dim": 6},
    ],
    "data": rec,
  }


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--task", default=DEFAULT_TASK)
  p.add_argument("--checkpoint", required=True)
  p.add_argument("--out", required=True)
  p.add_argument("--seconds", type=float, default=12.0)
  p.add_argument("--device", default="cuda:0")
  args = p.parse_args()

  out = dump_signals(args.task, args.checkpoint, args.seconds, args.device)
  with open(args.out, "w") as f:
    json.dump(out, f)
  print(f"Wrote {args.out}")


if __name__ == "__main__":
  main()

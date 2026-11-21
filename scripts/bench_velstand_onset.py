"""Measure cold starts, course drift and get-up in CPU MuJoCo.

    uv run scripts/bench_velstand_onset.py candidate.onnx results.csv

Run each actor with the same fold to compare identical initial placements.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from velstand_onset_rollouts import _Actor, _apply_action, _heading, _make_policy, _reset


_COMMANDS = ((0, 0), (0.15, 0), (0.20, 0), (0, 0.5), (0, -0.5), (0, 1), (0, -1),
             (0.15, 0.5), (0.20, -0.5))
_FOLDS = {
    "primary": ((8920, 8963, 9029, 9067, 9151, 9173, 9257, 9283, 9323, 9397, 9431, 9461), 25100),
    "reserve": ((17573, 17627, 17707, 17789, 17851, 17911, 17977, 18049, 18121, 18199, 18253, 18341), 26201),
}
_SETTLE_TICKS = (90, 101, 113, 122, 97, 109, 119, 94, 106, 117, 98, 111)
_POSES = ("SIT", "FOLD", "back", "face", "left", "right")


def _tick(policy, actor, gains, target):
    policy.vel_cmd += 0.2 * (target - policy.vel_cmd)
    policy._update_command()
    _apply_action(policy, actor(policy.get_observations()), gains)


def _motion(policy, actor, gains, command, seed, index):
    rng = np.random.default_rng(seed)
    _reset(policy, rng)
    if index % 3 == 2:
        policy.head_offset[:] = rng.uniform(-0.1, 0.1, 4)
    rest = np.zeros(3)
    for _ in range(_SETTLE_TICKS[index]):
        _tick(policy, actor, gains, rest)

    target = np.array([command[0], 0, command[1]])
    root = policy.data.joint("trunk_base_freejoint").qpos
    poses = [(*root[:3], _heading(root[3:]))]
    for _ in range(400):
        _tick(policy, actor, gains, target)
        poses.append((*root[:3], _heading(root[3:])))
    x, y, z, yaw = np.asarray(poses).T
    dx, dy = 100 * (x - x[0]), 100 * (y - y[0])
    c, s = np.cos(yaw[0]), np.sin(yaw[0])
    forward, lateral = c * dx + s * dy, -s * dx + c * dy
    heading = np.rad2deg(np.unwrap(yaw)[-1] - yaw[0])
    crossings = np.flatnonzero(forward >= 2)
    return ("motion", seed, *command, int(min(z) > 0.095), forward[-1], lateral[-1], heading,
            crossings[0] / 50 if crossings.size else "")


def _recovery(policy, actor, gains, pose, seed):
    _reset(policy, np.random.default_rng(seed), pose)
    root = policy.data.joint("trunk_base_freejoint").qpos
    trunk = policy.model.body("trunk_base").id
    rest = np.zeros(3)
    recovered = True
    for tick in range(300):
        _tick(policy, actor, gains, rest)
        if tick >= 275:
            recovered &= root[2] > 0.095 and policy.data.xmat[trunk, 8] > 0.9
    return (pose, seed, 0, 0, int(recovered), "", "", "", "")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("actor", type=Path)
    parser.add_argument("output", type=Path, help="new per-trial CSV")
    parser.add_argument("--fold", choices=_FOLDS, default="primary")
    args = parser.parse_args()
    actor = _Actor(args.actor)
    policy, gains = _make_policy(actor)
    motion_seeds, first_recovery_seed = _FOLDS[args.fold]
    with args.output.open("x", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("case", "seed", "vx", "wz", "upright", "forward_cm", "lateral_cm",
                         "degrees", "first_2cm_s"))
        for command in _COMMANDS:
            for index, seed in enumerate(motion_seeds):
                writer.writerow(_motion(policy, actor, gains, command, seed, index))
        for index, pose in enumerate(_POSES):
            for seed in range(first_recovery_seed + 20 * index, first_recovery_seed + 20 * (index + 1)):
                writer.writerow(_recovery(policy, actor, gains, pose, seed))


if __name__ == "__main__":
    main()

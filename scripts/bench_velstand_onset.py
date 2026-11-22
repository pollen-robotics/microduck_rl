"""Measure cold starts, course drift and get-up in CPU MuJoCo.

    uv run scripts/bench_velstand_onset.py candidate.onnx results.csv

Run each actor with the same fold to compare identical initial placements.
Use --head-poses for additional starting postures, or --paired-heads to replay
every placement with and without a four-axis head command. Use --held-head to
apply a specified head command after settling, as in a fresh robotd drive.
"""

import argparse
import csv
import random
from pathlib import Path

import numpy as np

from velstand_onset_rollouts import _Actor, _apply_action, _heading, _make_policy, _reset


_COMMANDS = ((0, 0), (0.15, 0), (0.20, 0), (0, 0.5), (0, -0.5), (0, 0.65), (0, -0.65), (0, 1), (0, -1),
             (0.15, 0.5), (0.20, -0.5))
_FOLDS = {
    "primary": ((8920, 8963, 9029, 9067, 9151, 9173, 9257, 9283, 9323, 9397, 9431, 9461), 25100),
    "reserve": ((17573, 17627, 17707, 17789, 17851, 17911, 17977, 18049, 18121, 18199, 18253, 18341), 26201),
}
_SETTLE_TICKS = (90, 101, 113, 122, 97, 109, 119, 94, 106, 117, 98, 111)
_POSES = ("SIT", "FOLD", "back", "face", "left", "right")


def _tick(policy, actor, gains, target, head_target=None, *, after_physics=False):
    policy.vel_cmd += 0.2 * (target - policy.vel_cmd)
    if head_target is not None:
        policy.head_offset += 0.2 * (head_target - policy.head_offset)
    policy._update_command()
    _apply_action(policy, actor(policy.get_observations()), gains, after_physics=after_physics)


def _motion(policy, actor, gains, command, seed, index, *, pose="STAND", head=None, settle_ticks=None,
            slew_head=False, delay_every=0):
    rng = np.random.default_rng(seed)
    _reset(policy, rng, start_keyframe=pose)
    if head is not None and not slew_head:
        policy.head_offset[:] = head
    elif head is None and index % 3 == 2:
        policy.head_offset[:] = rng.uniform(-0.1, 0.1, 4)
    rest = np.zeros(3)
    for _ in range(_SETTLE_TICKS[index] if settle_ticks is None else settle_ticks):
        _tick(policy, actor, gains, rest)

    head_target = np.asarray(head) if slew_head else None
    if slew_head:
        for _ in range(16):
            _tick(policy, actor, gains, rest, head_target)
    target = np.array([command[0], 0, command[1]])
    root = policy.data.joint("trunk_base_freejoint").qpos
    poses = [(*root[:3], _heading(root[3:]))]
    for tick in range(400):
        delayed = delay_every > 0 and tick % delay_every == 0
        _tick(policy, actor, gains, target, head_target, after_physics=delayed)
        poses.append((*root[:3], _heading(root[3:])))
    x, y, z, yaw = np.asarray(poses).T
    dx, dy = 100 * (x - x[0]), 100 * (y - y[0])
    c, s = np.cos(yaw[0]), np.sin(yaw[0])
    forward, lateral = c * dx + s * dy, -s * dx + c * dy
    heading = np.rad2deg(np.unwrap(yaw)[-1] - yaw[0])
    crossings = np.flatnonzero(forward >= 2)
    radius = np.hypot(dx, dy)
    return ("motion", seed, *command, int(min(z) > 0.095), forward[-1], lateral[-1], heading,
            crossings[0] / 50 if crossings.size else "", radius[-1], max(radius))


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
    return (pose, seed, 0, 0, int(recovered), "", "", "", "", "", "")


def _head_placements(fold):
    """Create independent placements and neutral-head controls with identical joint jitter."""
    sequence, low = (198151, 3100000) if fold == "primary" else (334621, 6400000)
    rng = random.Random(sequence)
    placements = []
    for i, seed in enumerate(rng.sample(range(low, low + 500000), 24)):
        pose = ("STAND", "SIT", "FOLD", "STAND", "FOLD", "SIT")[i % 6]
        head = tuple(round(rng.uniform(-0.12, 0.12), 6) for _ in range(4)) if i % 3 else (0,) * 4
        settle = rng.randrange(90, 125) if pose == "STAND" else rng.randrange(145, 170)
        placements.append((seed, pose, settle, head, False))
        if i in (2, 7, 13, 20):
            placements.append((seed, pose, settle, (0,) * 4, True))
    recovery_seeds = [rng.sample(range(low + 600000 + i * 20000, low + 620000 + i * 20000), 20)
                      for i in range(len(_POSES))]
    return placements, recovery_seeds


def _paired_placements(fold):
    """Replay every placement with a fixed four-axis head offset and a neutral head."""
    sequence, low = (6421971, 101000000) if fold == "primary" else (9865613, 162000000)
    rng = random.Random(sequence)
    placements = []
    for i, seed in enumerate(rng.sample(range(low, low + 400000), 30)):
        pose = ("STAND", "SIT", "FOLD")[i % 3]
        settle = rng.randrange(90, 125) if pose == "STAND" else rng.randrange(145, 170)
        head = tuple(round(rng.uniform(-0.12, 0.12), 6) for _ in range(4))
        placements.append((seed, pose, settle, head, False))
        placements.append((seed, pose, settle, (0,) * 4, True))
    recovery_seeds = [rng.sample(range(low + 500000 + i * 11000, low + 510000 + i * 11000), 20)
                      for i in range(len(_POSES))]
    return placements, recovery_seeds


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("actor", type=Path)
    parser.add_argument("output", type=Path, help="new per-trial CSV")
    parser.add_argument("--fold", choices=_FOLDS, default="primary")
    parser.add_argument("--delay-every", type=int, default=0, metavar="N",
                        help="apply one drive target after physics every N ticks, starting with the first (0 disables)")
    starts = parser.add_mutually_exclusive_group()
    starts.add_argument("--head-poses", action="store_true",
                        help="use settled STAND, SIT and FOLD starts with head offsets and four neutral controls")
    starts.add_argument("--paired-heads", action="store_true",
                        help="run 30 placements twice, once with the offset and once with a neutral head")
    starts.add_argument("--held-head", type=float, nargs=4, metavar=("NECK", "PITCH", "YAW", "ROLL"),
                        help="slew to these offsets (radians) after settling and replay each start with a neutral head")
    args = parser.parse_args()
    if args.delay_every < 0:
        parser.error("--delay-every must be nonnegative")
    if args.held_head is not None and not np.isfinite(args.held_head).all():
        parser.error("Head offsets must be finite radians")
    actor = _Actor(args.actor)
    policy, gains = _make_policy(actor)
    motion_seeds, first_recovery_seed = _FOLDS[args.fold]
    with args.output.open("x", newline="") as output:
        writer = csv.writer(output)
        columns = ("case", "seed", "vx", "wz", "upright", "forward_cm", "lateral_cm", "degrees", "first_2cm_s",
                   "end_radius_cm", "max_radius_cm")
        if args.head_poses or args.paired_heads or args.held_head is not None:
            writer.writerow((*columns, "start_pose", "settle_ticks", "neck_pitch", "head_pitch", "head_yaw",
                             "head_roll", "paired_control"))
            placements, recovery_seeds = (_head_placements if args.head_poses else _paired_placements)(args.fold)
            for command in _COMMANDS:
                for seed, pose, settle, head, control in placements:
                    if args.held_head is not None and not control:
                        head = args.held_head
                    row = _motion(policy, actor, gains, command, seed, 0, pose=pose, head=head, settle_ticks=settle,
                                  slew_head=args.held_head is not None, delay_every=args.delay_every)
                    writer.writerow((*row, pose, settle, *head, int(control)))
            for pose, seeds in zip(_POSES, recovery_seeds):
                for seed in seeds:
                    writer.writerow((*_recovery(policy, actor, gains, pose, seed), pose, "", "", "", "", "", ""))
            return
        writer.writerow(columns)
        for command in _COMMANDS:
            for index, seed in enumerate(motion_seeds):
                writer.writerow(_motion(policy, actor, gains, command, seed, index, delay_every=args.delay_every))
        for index, pose in enumerate(_POSES):
            for seed in range(first_recovery_seed + 20 * index, first_recovery_seed + 20 * (index + 1)):
                writer.writerow(_recovery(policy, actor, gains, pose, seed))


if __name__ == "__main__":
    main()

"""Stand-Sprung: an ACTIVE standing controller for the boot-equipped robot.

WHY THIS EXISTS. The robot cannot hold HOME_FRAME passively with the boots on:
its tip margin is 4.4 deg, the head alone sags 5-6 deg under static load, and on
the bench it topples in ~2 s from any fixed pose -- HOME, a sim-optimised pose,
and a hand-trimmed one all failed the same way. A trained controller has no such
problem (the hop policies hold 800-980 of 1000 steps under pushes), so "stable
home" has to be a policy, not a pose.

The hop policy cannot double as that policy: freezing its phase in the recovery
half gave 4.8 terminations per env in 15 s, 5-7% never falling. It never saw a
stationary phase in training.

WHAT THIS IS. The velocity env's standing recipe on the SAME robot the hop uses
-- sprung foot, all-collisions, 893 g, 25 x 40 mm sole, kp 400, dt 0.002 -- with
the velocity command pinned to zero, the stepping rewards removed, and the hop
tasks' "only the boots may touch the ground" termination. Pushes stay on: a
stand that cannot take a shove is not a stand.

Deployment: the daemon's STAND slot, selected whenever the twist magnitude is
under 0.05 -- which is exactly what this is trained on. Zero command in, hold
still out. `Start` on the pad then means "stand", and the hop policy in the walk
slot takes over the moment the phase driver sends a non-zero twist.

Deliberately NOT symmetry-constrained: this is a stepping stone toward a hop
policy that also stands, not the deliverable, and a stand does not need it.
"""

from __future__ import annotations

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.hop import add_boots_only_ground_contact


def make_stand_variant(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
    """Turn a velocity env cfg into the boot stand task. Run BEFORE make_sprung_variant."""
    # 1. Zero velocity command, always. Every env is a "standing env", and the
    #    ranges collapse so the sampled command is exactly [0, 0, 0] -- which is
    #    what the daemon's stand slot receives.
    command = deepcopy(cfg.commands["twist"])
    cfg.commands["twist"] = command
    for attr in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
        if hasattr(command.ranges, attr):
            setattr(command.ranges, attr, (0.0, 0.0))
    if hasattr(command, "rel_standing_envs"):
        command.rel_standing_envs = 1.0
    if hasattr(command, "rel_heading_envs"):
        command.rel_heading_envs = 0.0

    # 2. The velocity-range curriculum has nothing to ramp; the standing-envs
    #    curriculum would fight the 1.0 above.
    for k in ("velocity_command_ranges", "standing_envs"):
        cfg.curriculum.pop(k, None)

    # 3. Stepping rewards pay for lifting feet. A stand must not.
    for k in ("air_time", "feet_air_time", "alternating_flight", "foot_swing_height",
              "foot_clearance"):
        cfg.rewards.pop(k, None)

    # 4. The hop tasks' fall rule, and the tightened 50 deg fall angle with it.
    add_boots_only_ground_contact(cfg)
    return cfg

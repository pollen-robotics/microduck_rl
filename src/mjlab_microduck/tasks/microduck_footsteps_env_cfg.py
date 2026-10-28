"""Footstep-conditioned walking: place each foot on the target FootstepNet plans (Sigmaban footstep policies, 2024).

Built on the velocity recipe (robot, DR, obs noise/delays, gait regularizers) with the twist command replaced by a
FootstepCommand (mdp.py, contract in mjlab_microduck/footsteps.py): the 3-D twist obs slot carries the swing
foot's target landing pose (dx, dy, dθ) in the current support sole frame; the support foot follows the robot's
odometry rule, so the target switches exactly when the runtime would see the touchdown. Head/body command slots
keep the velocity semantics (head tracked, body zero-weighted).

Why not the velocity policy + a footstep→velocity map (the Sigmaban mapper path)? Measured 2026-10-27 on the
deployed velstand (scripts/footsteps_*): with a new command every step the foot lands 13-25 mm RMS (dx) away from
the mapped target — half of each step is set by the previous command (momentum lag) and the gait has
history-dependent attractors (e.g. the turn split between the two feet). The policy has to see the target itself.

Rewards: velocity tracking → footstep_placement (paid at each switch, exp(-25 pos - 5 yaw)) + potential-based
swing progress. The gait terms (air_time, clearance, swing height, slip, pose) gate on ‖command‖ > 0.01 and stay
always on (|dy| ≈ 95 mm): this policy always steps; standing is the walk policy's job at runtime.

Target range: starts at velstand's comfortable envelope, widened by step curriculum to the planner's step size plus a
margin (user 2026-10-28: 10 cm planner steps looked too long → FootstepNet dx ±5 cm, policy dx ±6 cm). Kinematic
reach is far wider (scripts/footsteps_reach.py: dx ±116 mm, dy -11..+23 mm level-trunk, dθ -52..+44°). The envelope
the trained policy actually tracks becomes FootstepNet's limits (footsteps.py MAX_*).
"""

import copy

from mjlab.managers import RewardTermCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import MicroduckRlCfg, make_microduck_velocity_env_cfg
from mjlab_microduck.tasks.microduck_velstand_env_cfg import _collapse_curricula_to_final

# Warm start from the velocity walker (same recipe, same walk-backlash model, normalizer-compatible twist slot):
#   MICRODUCK_WARM_START=1 uv run train Mjlab-Footsteps-Flat-Backlash-MicroDuck \
#       --init-checkpoint hf://pollen-robotics/microduck-training/bundles/velocity2/441tzs6d_3750 ...
# The source's curricula are pinned at their final stage (it was trained under them); the footstep target ramp
# lives in the command term and starts from scratch (MICRODUCK_WARM_START restarts the step counter).
WARM_START = True

# Reward weights. footstep_placement returns value/dt: its weight is the reward PER STEP TAKEN. At ~4 steps/s,
# 1.0 → ≤ 4/s, the mass of the velocity tracking pair it replaces (2.0 + 2.0 per second).
FOOTSTEP_PLACEMENT_WEIGHT = 1.0
FOOTSTEP_PLACEMENT_POS_SCALE = 25.0  # 1 cm → 0.78, 4 cm → 0.37
FOOTSTEP_PLACEMENT_YAW_SCALE = 5.0  # 6° → 0.59
SWING_PROGRESS_WEIGHT = 2.0  # potential shaping, ≈ 2 x approach speed [m/s]: ~0.6/s while stepping

# Target ranges (FootstepNet neutral convention, right support: + dy outward, + dθ toe-out), rows dx / dy / dθ.
FOOTSTEP_RANGE_START = ((-0.05, 0.035), (-0.012, 0.015), (-0.17, 0.17))  # velstand's own envelope
FOOTSTEP_RANGE_END = ((-0.06, 0.06), (-0.015, 0.030), (-0.52, 0.52))  # planner dx ±5 cm + margin
FOOTSTEP_RAMP_ITERS = (500, 3000)
FOOTSTEP_REPEAT_PROB = 0.4
FOOTSTEP_NEUTRAL_PROB = 0.1

COMMAND_NAME = "footstep"


def make_microduck_footsteps_env_cfg(play: bool = False, rough: bool = False):
    cfg = make_microduck_velocity_env_cfg(play=play, rough=rough)
    if WARM_START and not play:
        _collapse_curricula_to_final(cfg)

    # footstep command replaces twist, first in the manager (obs order: command, head_command, body_command)
    footstep = microduck_mdp.FootstepCommandCfg(
        range_start=FOOTSTEP_RANGE_START,
        range_end=FOOTSTEP_RANGE_END,
        ramp_steps=tuple(i * 24 for i in FOOTSTEP_RAMP_ITERS),
        repeat_prob=FOOTSTEP_REPEAT_PROB,
        neutral_prob=FOOTSTEP_NEUTRAL_PROB,
    )
    if play:
        footstep.ramp_steps = (0, 1)  # full range in play
    commands = {COMMAND_NAME: footstep}
    commands.update({k: v for k, v in cfg.commands.items() if k != "twist"})
    cfg.commands = commands

    for name in ("track_linear_velocity", "track_angular_velocity"):
        cfg.rewards.pop(name)
    cfg.curriculum.pop("standing_envs")  # twist-only (rel_standing_envs)

    # every remaining twist reader → footstep (gait terms only use ‖command‖ > threshold as a "moving" gate)
    for terms in (cfg.rewards, *(g.terms for g in cfg.observations.values())):
        for term in terms.values():
            if term.params.get("command_name") == "twist":
                term.params["command_name"] = COMMAND_NAME

    cfg.rewards["footstep_placement"] = RewardTermCfg(
        func=microduck_mdp.footstep_placement,
        weight=FOOTSTEP_PLACEMENT_WEIGHT,
        params={"command_name": COMMAND_NAME, "pos_scale": FOOTSTEP_PLACEMENT_POS_SCALE,
                "yaw_scale": FOOTSTEP_PLACEMENT_YAW_SCALE},
    )
    cfg.rewards["footstep_swing_progress"] = RewardTermCfg(
        func=microduck_mdp.footstep_swing_progress,
        weight=SWING_PROGRESS_WEIGHT,
        params={"command_name": COMMAND_NAME},
    )
    return cfg


MicroduckFootstepsRlCfg = copy.deepcopy(MicroduckRlCfg)
MicroduckFootstepsRlCfg.experiment_name = "footsteps"
MicroduckFootstepsRlCfg.run_name = "footsteps"
MicroduckFootstepsRlCfg.max_iterations = 6000

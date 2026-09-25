"""Microduck roulade sprint — roll forward endlessly, for the Arena's 2 m sprint.

Built on the roulade env (robot, sensors, DR, obs, mid-roll spawns) with the
task swapped: no landing, no standing up, no speed caps. The only goal is
trunk distance along the spawn heading, which is all the microduck-arena
`sprint-2m` event times.

The Arena's one constraint is its fall rule (arena/runner.py): tipped past 60°
for 0.3 s straight (15 steps at 50 Hz) is a fall, and one step back inside 60°
resets the count. Each inverted phase of a roll (60° → 300°) must therefore
take under 0.3 s — about 14 rad/s while tipped. The roulade env measured its
natural over-the-top transit at 3.5–5.5 rad/s, so this is the thing to watch:
the tipped-run penalty ramps from GRACE_STEPS, and the episode ends at
TIPPED_TERMINATION_STEPS, a margin under the Arena's 15.

The Arena feeds the sweep's forward command (0.20–1.20 m/s) into the twist
slot, so the command is sampled over that range to keep the obs in
distribution. The policy is free to ignore it.

Run 1 (checkpoints 250/500, scored in the Arena) learned a forward-leaning
run, not a roll, and failed two ways: a flip attempt tipped 16 steps (a
fall), and a slow left turn walked it into the lane wall (two timeouts —
training has no walls, and at 35° off course the progress reward still paid
82%). Run 2: heading-error and lateral-offset penalties keep it on the lane's
centre line, and the tipped cutoff tightens 12 → 8 steps.
"""

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.rewards import is_terminated
from mjlab.managers import EventTermCfg, RewardTermCfg, TerminationTermCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_roulade_env_cfg import (
    MicroduckRouladeRlCfg,
    make_microduck_roulade_env_cfg,
)

# Several rolls per episode — the policy must chain them, not do one.
EPISODE_LENGTH_S = 8.0

# Tipped-run shaping against the Arena's 15-step fall.
GRACE_STEPS = 3
TIPPED_TERMINATION_STEPS = 8

# Arena sprint-2m sweep: [sweep] lo / hi.
ARENA_COMMAND_RANGE = (0.20, 1.20)

# Mid-roll spawns carry sprint-grade angular momentum.
MIDROLL_OMEGA_RANGE = (2.0, 12.0)


def make_microduck_roulade_sprint_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create the Microduck roulade-sprint environment configuration."""
    cfg = make_microduck_roulade_env_cfg(play=play)
    cfg.episode_length_s = EPISODE_LENGTH_S

    # ── Rewards: drop the single-roll task (landing, rise, stand, caps) ───────
    for name in [
        "roulade_progress",
        "roulade_overspeed",
        "roulade_head_pivot",
        "roulade_landing_composite",
        "roulade_upright_after_roll",
        "roulade_height_after_roll",
        "roulade_landing_sharp",
        "roulade_stand_tax",
        "roulade_rise_velocity",
        "arrival_damping",
        "joint_torque_rate_l2",
    ]:
        del cfg.rewards[name]

    # ── Rewards: the sprint ───────────────────────────────────────────────────
    cfg.rewards["heading_progress"] = RewardTermCfg(
        func=microduck_mdp.heading_progress_velocity,
        weight=4.0,
        params={"max_vel": 1.5},
    )
    cfg.rewards["tipped_run"] = RewardTermCfg(
        func=microduck_mdp.tipped_run_penalty,
        weight=-2.0,
        params={"grace_steps": GRACE_STEPS, "max_steps": TIPPED_TERMINATION_STEPS},
    )
    # Stay on the lane: the Arena's walls are 0.4 m either side of the start.
    cfg.rewards["heading_error"] = RewardTermCfg(
        func=microduck_mdp.heading_error_penalty,
        weight=-4.0,
    )
    cfg.rewards["lateral_offset"] = RewardTermCfg(
        func=microduck_mdp.lateral_offset_penalty,
        weight=-10.0,
    )
    cfg.rewards["terminated"] = RewardTermCfg(
        func=is_terminated,
        weight=-20.0,
    )

    # ── Terminations: the Arena's fall, with margin ───────────────────────────
    cfg.terminations["tipped_too_long"] = TerminationTermCfg(
        func=microduck_mdp.tipped_run_exceeded,
        time_out=False,
        params={"max_steps": TIPPED_TERMINATION_STEPS},
    )

    # ── Events ────────────────────────────────────────────────────────────────
    cfg.events["set_roulade_state"].params["midroll_omega_range"] = MIDROLL_OMEGA_RANGE
    cfg.events["reset_tipped_run"] = EventTermCfg(
        func=microduck_mdp.reset_tipped_run,
        mode="reset",
    )
    # After reset_base and set_roulade_state, which place the root.
    cfg.events["record_spawn_position"] = EventTermCfg(
        func=microduck_mdp.record_spawn_position,
        mode="reset",
    )

    # ── Command: the Arena's sweep range ──────────────────────────────────────
    cfg.commands["twist"].ranges.lin_vel_x = ARENA_COMMAND_RANGE

    # ── Curriculum: keep DR ramps, drop the single-roll schedule ──────────────
    for name in [
        "roulade_spawn_mix",
        "action_rate_weight",
        "arrival_damping_weight",
        "torque_rate_weight",
        "gentle_landing_weight",
    ]:
        del cfg.curriculum[name]

    return cfg


MicroduckRouladeSprintRlCfg = replace(
    MicroduckRouladeRlCfg,
    experiment_name="microduck_roulade_sprint",
    run_name="microduck_roulade_sprint",
)

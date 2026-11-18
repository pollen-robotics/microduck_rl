"""Microduck headstand exits: from the hold, roll down the back and stand.

The exit from a headstand is the second half of the roulade (forward roll):
the roll passes through "inverted, 180°" on its way from standing to
standing, and the roulade env already trains from mid-roll spawns (its
reverse curriculum). This task IS the roulade env with every episode spawned
at 180° in a hold pose and the rotation accumulator pre-set to 180°, so the
roll's own progress, landing and stand-cost rewards carry it to standing.

Two styles: "legs_together" (from the legs-together hold, a plain back roll) and
"splitover" (from either split hold, the legs kept split and straight while
the trunk goes over, so the split itself is the exit). Without the
split-gated progress and the bent-knee cost, the roll env learns a tuck
roll out of the split hold.

Runtime chaining: the runtime hot-swaps ONNX policies with a shared obs
contract (AGENTS.md), so hold → exit is a policy switch, like the roll
handing to the standing policy after landing.
"""

import math
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import CurriculumTermCfg, RewardTermCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from mjlab_microduck.robot.microduck_constants import MICRODUCK_ALLCOLLISIONS_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_headstand_env_cfg import (
    HEADSTAND_OVERRIDES,
    HEADSTAND_LEGS_TOGETHER_OVERRIDES,
    HEADSTAND_Z,
)
from mjlab_microduck.tasks.microduck_roulade_env_cfg import (
    MicroduckRouladeRlCfg,
    make_microduck_roulade_env_cfg,
)

# The hold hands over at 165-190° of trunk rotation, wobbling up to 1 rad/s,
# with the trunk resting at HEADSTAND_Z; spawns cover that band.
HOLD_PITCH_RANGE_DEG = (165.0, 190.0)
HOLD_Z_RANGE = (HEADSTAND_Z + 0.002, HEADSTAND_Z + 0.006)
HOLD_OMEGA_RANGE = (0.0, 1.0)
HOLD_POSE_LERP_RANGE = (0.9, 1.0)   # the hold pose, nearly, plus joint noise
HOLD_JOINT_NOISE_STD = 0.08


def make_microduck_backroll_env_cfg(
    play: bool = False,
    style: str = "legs_together",
    *,
    progress_w: float = 8.0,
    rate_cap: float = 5.0,
    tuck_w: float = -2.0,
    window_deg: tuple[float, float] = (170.0, 330.0),
    omega_max: float = 7.0,
    overspeed_w: float = -0.1,
) -> ManagerBasedRlEnvCfg:
    """The roulade env spawned in a headstand hold.

    style: "legs_together" (the legs-together hold, a plain back roll) or
    "splitover" (either split hold, the legs kept split and straight while
    the trunk goes over). `play` is accepted for the registry and changes
    nothing. The keyword arguments apply to the split over only (the back roll
    ignores them): the weight and paid rate cap (rad/s) of its progress term,
    the per-step weight of the bent-knee cost, the roll window (degrees) in
    which the legs must stay split, and its overspeed threshold (rad/s) and
    weight.
    """
    if style not in ("legs_together", "splitover"):
        raise ValueError(f"style must be legs_together or splitover, got {style!r}")
    if tuck_w > 0.0 or overspeed_w > 0.0:
        raise ValueError("tuck_w and overspeed_w weight costs (>= 0 terms) and must be <= 0")
    cfg = make_microduck_roulade_env_cfg(play=play)
    overrides = HEADSTAND_LEGS_TOGETHER_OVERRIDES if style == "legs_together" else HEADSTAND_OVERRIDES
    # Same collision model and solver budget as every other headstand policy:
    # a roll trained on the feet-and-shell model lost 6 of 15 chained attempts
    # on the allcollisions model the routine runs on.
    cfg.scene.entities = {"robot": MICRODUCK_ALLCOLLISIONS_ROBOT_CFG}
    # The roulade's sensors plus the headstand's "everything but head and feet"
    # one. No reward here reads it; an evaluation of an exit needs to see a
    # thigh or a shin on the floor to tell standing from a collapse.
    cfg.scene.sensors = tuple(cfg.scene.sensors) + (
        ContactSensorCfg(
            name=microduck_mdp._HEADSTAND_OTHER_SENSOR,
            primary=ContactMatch(
                mode="body",
                pattern=r"^(?!jaw_soft$|yaw_roll_motion$|neck_pitch$|ankle_left$|ankle_right$).*",
                entity="robot",
            ),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found",),
            reduce="none",
            num_slots=1,
        ),
    )
    cfg.sim.nconmax = max(cfg.sim.nconmax or 0, 200)
    cfg.sim.mujoco.iterations = 30
    cfg.sim.mujoco.ls_iterations = 50
    spawn = cfg.events["set_roulade_state"].params
    spawn.update(
        standing_prob=0.0,
        midroll_prob=1.0,
        midroll_pitch_min=math.radians(HOLD_PITCH_RANGE_DEG[0]),
        midroll_pitch_max=math.radians(HOLD_PITCH_RANGE_DEG[1]),
        midroll_z_min=HOLD_Z_RANGE[0],
        midroll_z_max=HOLD_Z_RANGE[1],
        midroll_omega_range=HOLD_OMEGA_RANGE,
        tuck_overrides=overrides,
        tuck_factor_range=HOLD_POSE_LERP_RANGE,
        joint_noise_std=HOLD_JOINT_NOISE_STD,
    )
    # The roulade's spawn-mix curriculum would reintroduce standing spawns; pin it.
    cfg.curriculum["roulade_spawn_mix"] = CurriculumTermCfg(
        func=microduck_mdp.event_param_curriculum,
        params={
            "event_name": "set_roulade_state",
            "param_stages": [{"step": 0, "params": {"standing_prob": 0.0, "midroll_prob": 1.0}}],
        },
    )
    # The exit is a controlled fall onto the back plus the rise: a 6 s episode
    # like the headstand's (the roulade's 5 s starts standing).
    cfg.episode_length_s = 6.0
    if style == "splitover":
        # Half the spawns are the mirrored split, so the exit works after one
        # switch. That also makes the spawn distribution mirror-symmetric,
        # which is what lets this asymmetric-looking task keep the roulade's
        # mirror loss. Progress pays only while the legs are split and
        # straight during the over-the-top window, and a tuck costs per step
        # in the same window; without both the roll env learns a tuck roll.
        spawn["mirror_prob"] = 0.5
        lo, hi = math.radians(window_deg[0]), math.radians(window_deg[1])
        cfg.rewards["roulade_progress"] = RewardTermCfg(
            func=microduck_mdp.roulade_progress_split,
            weight=progress_w,
            params={"target_angle": 2 * math.pi, "max_paid_rate": rate_cap, "angle_lo": lo, "angle_hi": hi},
        )
        cfg.rewards["roulade_tuck"] = RewardTermCfg(
            func=microduck_mdp.roulade_tuck_cost,
            weight=tuck_w,
            params={"angle_lo": lo, "angle_hi": hi},
        )
        cfg.rewards["roulade_overspeed"].params["omega_max"] = omega_max
        cfg.rewards["roulade_overspeed"].weight = overspeed_w
    return cfg


def _runner(name: str) -> RslRlOnPolicyRunnerCfg:
    """The roulade's network and PPO settings under this task's experiment name
    (a shallow copy; the registry deep-copies a runner cfg when it loads one)."""
    return replace(MicroduckRouladeRlCfg, experiment_name=name, run_name=name, save_interval=250, num_steps_per_env=24, max_iterations=6_000)


MicroduckBackrollLegsTogetherRlCfg = _runner("microduck_headstand_backroll_legs_together")
MicroduckSplitOverRlCfg = _runner("microduck_headstand_splitover")

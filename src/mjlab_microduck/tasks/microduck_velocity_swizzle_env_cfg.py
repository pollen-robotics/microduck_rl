"""Microduck roller SWIZZLE environment — clean classic swizzle.

A separate roller task producing a CLASSIC SWIZZLE: both blades stay on the ground,
the legs spread out and pull back in SYMMETRICALLY (hourglass pattern), propelling
the duck forward. Simpler / more stable alternative to the alternating stride
(`Mjlab-Velocity-Flat-MicroDuck-Rollers`), which does not transfer well to the real
robot. The stride env is left untouched.

Approach A (see docs/superpowers/specs/2026-07-23-swizzle-env-design.md): the base
roller recipe NATURALLY converges to a swizzle, so we reuse the stride env wholesale
(robot, 61D obs, command, full DR, curricula, sim2real — deploys identically with
`--roller`) and only swap the reward recipe:
  - REMOVE the anti-swizzle / stride terms.
  - ADD leg_symmetry (legs mirror) + grounded (both blades down).
"""

import dataclasses

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import CurriculumTermCfg, ObservationTermCfg, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_rollers_env_cfg import (
    MicroduckRollersRlCfg,
    _apply_head_override,
    make_microduck_velocity_rollers_env_cfg,
)

# Head-free variant: head command range schedule. Held tiny until iter 1500 —
# the swizzle and the heading curriculum (straight until 1000, follow by 2500)
# come first — then widened in 4 gentle steps. The stride head-free run
# (0otqeivr) lost ~0.07 wheel_speed at every stage boundary of a 750-iter
# spacing, so the steps here are smaller relative to the range.
SWIZZLE_HEAD_RANGE_STAGES = [
    # iter, ((neck_pitch), (head_pitch), (head_yaw), (head_roll))
    (0,    ((-0.05, 0.05), (-0.05, 0.05), (-0.07, 0.07), (-0.015, 0.015))),
    (1500, ((-0.22, 0.22), (-0.22, 0.22), (-0.28, 0.28), (-0.06, 0.06))),
    (2250, ((-0.50, 0.50), (-0.50, 0.50), (-0.63, 0.63), (-0.14, 0.14))),
    (3000, ((-0.83, 0.83), (-0.83, 0.83), (-1.05, 1.05), (-0.23, 0.23))),
    (3750, ((-1.10, 1.10), (-1.10, 1.10), (-1.40, 1.40), (-0.31, 0.31))),
]

# Stride / anti-swizzle rewards to drop for the swizzle task.
_ANTI_SWIZZLE = ("single_support", "glide", "skating_air_time", "gait_symmetry", "hip_roll_neutral")


def make_microduck_velocity_swizzle_env_cfg(
    play: bool = False,
    head_override: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Roller swizzle env: the stride env minus its anti-swizzle terms, plus symmetry
    and grounded rewards. Everything else (robot, obs, command, DR) is identical."""
    cfg = make_microduck_velocity_rollers_env_cfg(play=play)

    for name in _ANTI_SWIZZLE:
        if name in cfg.rewards:
            del cfg.rewards[name]

    # Legs mirror each other (the swizzle's defining symmetry).
    cfg.rewards["leg_symmetry"] = RewardTermCfg(
        func=microduck_mdp.leg_symmetry_reward,
        weight=2.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # Keep both blades on the ground (classic swizzle: no lifting).
    cfg.rewards["grounded"] = RewardTermCfg(
        func=microduck_mdp.grounded_reward,
        weight=1.0,
        params={"sensor_name": "feet_ground_contact", "command_name": "twist"},
    )

    # --- Backward locomotion (option A): cmd_x < 0 means GO BACKWARD (not brake) ---
    # wheel_speed rewards wheel spin in the COMMANDED direction (fwd for +, back for
    # -); the braking reward is dropped (negative no longer means "stop"); command
    # range symmetrised so forward and backward get equal push range. To stop, command
    # cmd_x ~ 0 (coast). grounded uses |cmd_x| so it holds the blades down both ways.
    cfg.rewards["wheel_speed"].params["bidirectional"] = True
    if "braking" in cfg.rewards:
        del cfg.rewards["braking"]
    cfg.commands["twist"].ranges.lin_vel_x = (-0.6, 0.6)

    # --- Heading curriculum: go STRAIGHT first, then FOLLOW a commanded direction ---
    # The stride env disabled heading (ang_vel_z=(0,0), heading_hold, no heading_tracking).
    # Re-enable the heading command so cmd[2] carries the heading error to a sampled
    # target, and add heading_tracking (starts at 0). A curriculum then swaps the two:
    #   phase 1 (straight): heading_hold dominant, heading_tracking off
    #   phase 2 (follow):   heading_hold -> 0, heading_tracking -> up
    # cmd[2] = heading error clip. Reduced ±1.0 -> ±0.5: bounds the OBSERVED heading
    # error, so the turn-correction rate is gentler (a ±1.0-trained policy turned too
    # violently — had to run --max-angular-vel 0.3 to tame it). It can still reach any
    # heading (the error just saturates at 0.5), so it turns fully but smoothly, and
    # the heading_tracking weight stays 3.0 so it still follows direction well.
    cfg.commands["twist"].ranges.ang_vel_z = (-0.5, 0.5)

    cfg.rewards["heading_tracking"] = RewardTermCfg(
        func=microduck_mdp.heading_tracking_reward,
        weight=0.0,  # ramped up by the curriculum below (must match its step-0 value)
        params={"command_name": "twist", "std": 0.5},
    )

    cfg.curriculum["heading_hold_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "heading_hold",
            "weight_stages": [
                {"step": 0,          "weight": 1.0},   # must match heading_hold's initial weight
                {"step": 1000 * 24,  "weight": 1.0},   # hold straight while the swizzle solidifies
                {"step": 1750 * 24,  "weight": 0.5},
                {"step": 2500 * 24,  "weight": 0.0},
            ],
        },
    )
    cfg.curriculum["heading_tracking_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "heading_tracking",
            "weight_stages": [
                {"step": 0,          "weight": 0.0},
                {"step": 1000 * 24,  "weight": 0.0},   # straight-only until here
                {"step": 1750 * 24,  "weight": 1.5},
                {"step": 2500 * 24,  "weight": 3.0},
            ],
        },
    )

    if head_override:
        # Head-free twin: the head is driven externally (neck targets overwritten by
        # a slewed head command), the policy only swizzles. Replaces the
        # policy-managed head control below — see _apply_head_override.
        _apply_head_override(cfg, play, range_stages=SWIZZLE_HEAD_RANGE_STAGES)
    else:
        _add_policy_head_control(cfg)

    return cfg


def _add_policy_head_control(cfg: ManagerBasedRlEnvCfg) -> None:
    """Head-pose control where the POLICY produces the head pose (Y button)."""
    # --- Head-pose control (Y button): the policy produces the head pose ---------
    # Head-pose command (4D deltas from HOME: [neck_pitch, head_pitch, head_yaw,
    # head_roll]). Ported from the velocity env; ranges start small (widened by the
    # curriculum below). Resample every 2-5 s.
    cfg.commands["head_pose"] = microduck_mdp.UniformPoseCommandCfg(
        resampling_time_range=(2.0, 5.0),
        ranges=(
            (-0.05, 0.05),    # neck_pitch
            (-0.05, 0.05),    # head_pitch
            (-0.07, 0.07),    # head_yaw
            (-0.015, 0.015),  # head_roll (tighter — small mechanical range)
        ),
    )

    # Feed the REAL head command into the obs (replaces zero_command_padding) on
    # both groups. body_command stays zero-padded (no body-pose control here).
    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "head_pose"},
        )

    # Reward the head tracking its command. Weight 0 here — ramped in LATE by the
    # curriculum so it doesn't disturb the swizzle before it's solid.
    cfg.rewards["head_pose_tracking"] = RewardTermCfg(
        func=microduck_mdp.head_pose_tracking,
        weight=0.0,
        params={"command_name": "head_pose", "std": 0.5},
    )

    # Reconcile the two HOME-pullers that would fight head_pose_tracking:
    #  1) neck_joint_pos_l2 pulls the neck/head joints to HOME -> remove it.
    if "neck_joint_pos_l2" in cfg.rewards:
        del cfg.rewards["neck_joint_pos_l2"]
    #  2) the pose reward includes neck/head -> scope it to LEG joints only.
    # Remove neck/head/passive patterns from std dicts to match scoped asset_cfg.
    for std_key in ["std_standing", "std_walking", "std_running"]:
        if std_key in cfg.rewards["pose"].params:
            std_dict = cfg.rewards["pose"].params[std_key]
            # Keep only leg joint patterns (filter out neck, head, passive)
            cfg.rewards["pose"].params[std_key] = {
                k: v for k, v in std_dict.items()
                if "neck" not in k and "head" not in k and "passive" not in k
            }
    # Scope asset_cfg to LEG joints only (excludes neck, head, passive wheels)
    cfg.rewards["pose"].params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=(r"^(?!passive_|.*neck.*|.*head.*).*",)
    )

    # head_pose_tracking ramps 0 -> 4.0, staying 0 until ~1500 it. (swizzle solid),
    # so head control is added on top of a stable swizzle.
    cfg.curriculum["head_pose_tracking_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "head_pose_tracking",
            "weight_stages": [
                {"step": 0,          "weight": 0.0},   # must match initial weight
                {"step": 1500 * 24,  "weight": 0.0},   # head off while swizzle solidifies
                {"step": 2250 * 24,  "weight": 2.0},
                {"step": 3000 * 24,  "weight": 4.0},
            ],
        },
    )
    # Head-command range widens over the SAME window (tiny until 1500, full by 3000),
    # so the commanded head barely moves early and reaches full range once the policy
    # can handle it.
    cfg.curriculum["head_pose_range"] = CurriculumTermCfg(
        func=microduck_mdp.pose_command_range_curriculum,
        params={
            "command_name": "head_pose",
            "range_stages": [
                # step,               ((neck_pitch), (head_pitch), (head_yaw),  (head_roll))
                {"step": 0,          "ranges": ((-0.05, 0.05), (-0.05, 0.05), (-0.07, 0.07), (-0.015, 0.015))},
                {"step": 1500 * 24,  "ranges": ((-0.05, 0.05), (-0.05, 0.05), (-0.07, 0.07), (-0.015, 0.015))},
                {"step": 2250 * 24,  "ranges": ((-0.55, 0.55), (-0.55, 0.55), (-0.70, 0.70), (-0.15, 0.15))},
                {"step": 3000 * 24,  "ranges": ((-1.10, 1.10), (-1.10, 1.10), (-1.40, 1.40), (-0.31, 0.31))},
            ],
        },
    )


# Same PPO hyperparameters as the stride roller task, new experiment/run name.
MicroduckSwizzleRlCfg = dataclasses.replace(
    MicroduckRollersRlCfg,
    experiment_name="velocity_swizzle",
    run_name="velocity_swizzle",
)

MicroduckSwizzleHeadFreeRlCfg = dataclasses.replace(
    MicroduckRollersRlCfg,
    experiment_name="velocity_swizzle_headfree",
    run_name="velocity_swizzle_headfree",
)

"""Dedicated Microduck skill for ascending one square-edged step up to 25 mm.

The skill is activated by an external camera/controller roughly 20-40 cm
before the edge. It deliberately keeps the deployable 61D observation and 14D
action contract, so the runtime can hot-swap it with the normal walking policy.
"""

import math
from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import CurriculumTermCfg, RewardTermCfg, TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    MicroduckRlCfg,
    _soften_terrain_contacts,
    make_microduck_velocity_env_cfg,
)
from mjlab_microduck.tasks.step_up_terrain import UpStepTerrainCfg

TILE_SIZE = (3.0, 1.2)
LOWER_LENGTH = 1.0
APPROACH_DISTANCE_RANGE = (0.30, 0.30)
SUCCESS_FORWARD_DISTANCE = 0.50
SUCCESS_MAX_LATERAL_OFFSET = 0.30
BYPASS_MAX_LATERAL_OFFSET = 0.34
# Low enough for the first 5 mm curriculum row; forward distance already
# proves the robot is past the vertical edge, while this gate rejects a body
# lying on the upper floor. A final-height gate here would make easy rows
# impossible to graduate.
SUCCESS_MIN_TRUNK_HEIGHT = 0.105


def make_microduck_step_up_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    # Start from the rough walking recipe to retain the full sim2real DR,
    # delays, observation normalization contract and contact solver settings.
    cfg = make_microduck_velocity_env_cfg(play=play, rough=True)

    generator = TerrainGeneratorCfg(
        size=TILE_SIZE,
        border_width=10.0,
        curriculum=not play,
        num_rows=10 if not play else 1,
        num_cols=1,
        difficulty_range=(0.0, 1.0) if not play else (1.0, 1.0),
        sub_terrains={
            "up_step": UpStepTerrainCfg(
                lower_length=LOWER_LENGTH,
                approach_distance_range=APPROACH_DISTANCE_RANGE,
            )
        },
        add_lights=False,
    )
    cfg.scene.terrain = TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=generator,
        max_init_terrain_level=0 if not play else None,
    )
    cfg.scene.spec_fn = _soften_terrain_contacts
    cfg.sim.nconmax = 200
    cfg.sim.mujoco.iterations = 30
    cfg.sim.mujoco.ls_iterations = 50

    # External vision aligns the robot before switching skills. Training adds
    # modest reset error so deployment need not be pixel-perfect.
    cfg.events["reset_base"].params["pose_range"].update(
        {
            "x": (-0.02, 0.02),
            "y": (-0.04, 0.04),
            "yaw": (-math.radians(12.0), math.radians(12.0)),
        }
    )

    # This is an ascent skill, not a general navigation policy. A small range
    # prevents one brittle, single-speed solution while keeping every rollout
    # aimed at the edge.
    command = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs = 0.0
    command.rel_turn_in_place_envs = 0.0
    # The viewer's slider initializes at zero and requires zero to lie inside
    # its range. Training stays strictly forward-only; play additionally allows
    # zero so a human can pause safely.
    command.ranges.lin_vel_x = (0.0, 0.30) if play else (0.30, 0.30)
    # Viser requires every joystick axis to have a positive display maximum.
    # These ranges exist only in play; training keeps lateral/yaw exactly zero.
    command.ranges.lin_vel_y = (-0.10, 0.10) if play else (0.0, 0.0)
    command.ranges.ang_vel_z = (-0.10, 0.10) if play else (0.0, 0.0)

    # A 25 mm obstacle needs margin beyond the normal 20 mm swing target.
    # The target is relative to the sensed local terrain under each foot.
    cfg.rewards["foot_clearance"].params["target_height"] = 0.040
    cfg.rewards["foot_swing_height"].params["target_height"] = 0.040
    cfg.rewards["foot_swing_height"].weight = -0.5

    # Skill discovery comes before smoothness.  The inherited -0.1 action-rate
    # tax outweighed the extremely sparse endpoint signal in the first run and
    # made a timid gait locally optimal.  Keep a small anti-jitter term, then
    # tune smoothness in a later sim2real fine-tune after ascent is reliable.
    cfg.rewards["action_rate_l2"].weight = -0.01
    cfg.curriculum.pop("action_rate_weight", None)

    # Full-height fine-tuning needs intermediate credit: first keep moving
    # toward the edge, then reward one foot and both feet reaching the top.
    # Milestones are one-shot per episode, so stopping on the edge cannot farm
    # them. The edge is 0.30 m ahead of the spawn origin in this fixed setup.
    feet_cfg = SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))
    cfg.rewards["step_forward_velocity"] = RewardTermCfg(
        func=microduck_mdp.step_up_forward_velocity,
        weight=4.0,
        params={"cap": 0.6, "asset_cfg": SceneEntityCfg("robot")},
    )
    milestone_params = {
        "min_foot_forward": 0.32,
        "min_foot_height": 0.018,
        "max_lateral_offset": SUCCESS_MAX_LATERAL_OFFSET,
        "feet_cfg": feet_cfg,
    }
    cfg.rewards["one_foot_over_step"] = RewardTermCfg(
        func=microduck_mdp.step_up_foot_milestone,
        weight=100.0,
        params={
            **deepcopy(milestone_params),
            "required_feet": 1,
            "state_key": "_step_up_one_foot_seen",
        },
    )
    cfg.rewards["two_feet_over_step"] = RewardTermCfg(
        func=microduck_mdp.step_up_foot_milestone,
        weight=150.0,
        params={
            **deepcopy(milestone_params),
            "required_feet": 2,
            "state_key": "_step_up_two_feet_seen",
        },
    )

    # Strong one-frame endpoint bounty. Success terminates immediately, so it
    # cannot be farmed by standing forever on the upper floor.
    success_params = {
        "min_forward_distance": SUCCESS_FORWARD_DISTANCE,
        "min_trunk_height": SUCCESS_MIN_TRUNK_HEIGHT,
        "min_upright_cos": 0.8,
        "max_lateral_offset": SUCCESS_MAX_LATERAL_OFFSET,
        "asset_cfg": SceneEntityCfg("robot"),
    }
    cfg.rewards["step_up_success"] = RewardTermCfg(
        func=microduck_mdp.step_up_success,
        weight=200.0,
        params=deepcopy(success_params),
    )
    cfg.terminations["step_complete"] = TerminationTermCfg(
        func=microduck_mdp.step_up_complete,
        time_out=False,
        params=deepcopy(success_params),
    )
    cfg.terminations["step_bypass"] = TerminationTermCfg(
        func=microduck_mdp.step_up_out_of_bounds,
        time_out=False,
        params={
            "max_lateral_offset": BYPASS_MAX_LATERAL_OFFSET,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.episode_length_s = 8.0

    # These walking curricula would re-introduce standing rollouts and widen
    # head commands, both distractions for a short dedicated ascent skill.
    for curriculum_name in (
        "standing_envs",
        "head_pose_range",
        "head_pose_bias_weight",
    ):
        cfg.curriculum.pop(curriculum_name, None)
    cfg.events.pop("push_robot", None)

    # Replace the generic terrain curriculum with promotion based on actually
    # reaching the upper floor upright. Ten rows span a flush warm-up -> 25 mm.
    cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
        func=microduck_mdp.terrain_levels_step_up,
        params={
            "min_forward_distance": SUCCESS_FORWARD_DISTANCE,
            "min_trunk_height": SUCCESS_MIN_TRUNK_HEIGHT,
            "min_upright_cos": 0.8,
            "max_lateral_offset": SUCCESS_MAX_LATERAL_OFFSET,
        },
    )

    return cfg


def make_microduck_step_up_full_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Fixed 25 mm task for final fine-tuning after curriculum pretraining."""
    cfg = make_microduck_step_up_env_cfg(play=play)
    generator = cfg.scene.terrain.terrain_generator
    assert generator is not None
    generator.curriculum = False
    generator.num_rows = 1
    generator.difficulty_range = (1.0, 1.0)
    cfg.scene.terrain.max_init_terrain_level = None
    cfg.curriculum.pop("terrain_levels", None)
    return cfg


MicroduckStepUpRlCfg = deepcopy(MicroduckRlCfg)
MicroduckStepUpRlCfg.experiment_name = "step_up"
MicroduckStepUpRlCfg.run_name = "step_up_25mm"
MicroduckStepUpRlCfg.save_interval = 100
MicroduckStepUpRlCfg.max_iterations = 6_000

MicroduckStepUpFullRlCfg = deepcopy(MicroduckStepUpRlCfg)
MicroduckStepUpFullRlCfg.experiment_name = "step_up_full"
MicroduckStepUpFullRlCfg.run_name = "step_up_25mm_full"
MicroduckStepUpFullRlCfg.max_iterations = 1_000

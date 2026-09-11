"""Commanded left/right single-leg standing in one symmetric policy."""

import math
import os

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import (
    CurriculumTermCfg,
    MetricsTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from mjlab_microduck.robot.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    NUM_STEPS_PER_ENV,
    make_microduck_velocity_env_cfg,
)
from mjlab_microduck.tasks.symmetry import SYMMETRY_CFG, PpoWithSymmetryCfg


EPISODE_LENGTH_S = 6.0
STRICT_EPISODE_LENGTH_S = 10.0
RAMP_S = 1.5
TARGET_FOOT_HEIGHT = 0.012
COMMAND_NAME = "twist"
FEET_SENSOR = "feet_ground_contact"
NONFOOT_SENSOR = "nonfoot_ground_contact"
FEET_CFG = SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))


def _fixed_play_side() -> int:
    value = os.environ.get("SINGLE_LEG_PLAY_SIDE", "random").lower()
    if value in ("random", "0"):
        return 0
    if value in ("left", "-1"):
        return -1
    if value in ("right", "1"):
        return 1
    raise ValueError("SINGLE_LEG_PLAY_SIDE must be left, right, or random")


def _play_episode_length_s() -> float:
    value = float(os.environ.get("SINGLE_LEG_PLAY_EPISODE_S", EPISODE_LENGTH_S))
    if value <= 0.0:
        raise ValueError("SINGLE_LEG_PLAY_EPISODE_S must be positive")
    return value


def make_microduck_single_leg_stand_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    episode_length_s = _play_episode_length_s() if play else EPISODE_LENGTH_S
    cfg = make_microduck_velocity_env_cfg(play=play)
    cfg.scene.entities = {"robot": MICRODUCK_STANDUP_ROBOT_CFG}
    cfg.scene.sensors = (
        *cfg.scene.sensors,
        ContactSensorCfg(
            name=NONFOOT_SENSOR,
            primary=ContactMatch(
                mode="body",
                pattern=(
                    "trunk_base",
                    "hip_l",
                    "leg",
                    "jaw_soft",
                    "hip_l_2",
                    "leg_2",
                ),
                entity="robot",
            ),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found",),
            reduce="none",
            num_slots=1,
        ),
    )
    cfg.episode_length_s = episode_length_s

    cfg.commands[COMMAND_NAME] = microduck_mdp.SingleLegStandCommandCfg(
        entity_name="robot",
        resampling_time_range=(episode_length_s, episode_length_s),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        rel_forward_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(0.0, 0.0),
            heading=None,
        ),
        left_prob=0.5,
        ramp_s=RAMP_S,
        fixed_side=_fixed_play_side() if play else 0,
    )
    cfg.commands.pop("head_pose", None)
    cfg.commands.pop("body_pose", None)
    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding,
            params={"dim": 4},
        )
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding,
            params={"dim": 6},
        )

    for name in (
        "track_linear_velocity",
        "track_angular_velocity",
        "upright",
        "pose",
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "head_pose_tracking",
        "body_pose_tracking",
        "head_pose_bias",
    ):
        cfg.rewards.pop(name, None)

    cfg.rewards["single_leg_com"] = RewardTermCfg(
        func=microduck_mdp.single_leg_com_tracking,
        weight=5.0,
        params={"command_name": COMMAND_NAME, "asset_cfg": FEET_CFG},
    )
    cfg.rewards["support_contact"] = RewardTermCfg(
        func=microduck_mdp.single_leg_support_contact,
        weight=1.0,
        params={"command_name": COMMAND_NAME, "sensor_name": FEET_SENSOR},
    )
    cfg.rewards["swing_height"] = RewardTermCfg(
        func=microduck_mdp.single_leg_swing_height_tracking,
        weight=0.0,
        params={
            "command_name": COMMAND_NAME,
            "asset_cfg": FEET_CFG,
            "target_height": TARGET_FOOT_HEIGHT,
            "std": 0.008,
        },
    )
    cfg.rewards["swing_contact"] = RewardTermCfg(
        func=microduck_mdp.single_leg_swing_contact_cost,
        weight=0.0,
        params={"command_name": COMMAND_NAME, "sensor_name": FEET_SENSOR},
    )
    cfg.rewards["support_slip"] = RewardTermCfg(
        func=microduck_mdp.single_leg_support_slip_cost,
        weight=-0.5,
        params={
            "command_name": COMMAND_NAME,
            "sensor_name": FEET_SENSOR,
            "asset_cfg": FEET_CFG,
        },
    )
    cfg.rewards["excess_tilt"] = RewardTermCfg(
        func=microduck_mdp.single_leg_excess_tilt_cost,
        weight=-2.0,
        params={"max_tilt_deg": 32.0},
    )
    cfg.rewards["height_stand"] = RewardTermCfg(
        func=microduck_mdp.height_target_gaussian,
        weight=1.0,
        params={
            "target_height": 0.115,
            "std": 0.03,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )
    cfg.rewards["nonfoot_contact"] = RewardTermCfg(
        func=microduck_mdp.any_contact_cost,
        weight=-5.0,
        params={"sensor_name": NONFOOT_SENSOR},
    )
    cfg.rewards["action_rate_l2"].weight = -0.05
    cfg.rewards["body_ang_vel"].weight = -0.02
    cfg.rewards["angular_momentum"].weight = -0.01
    cfg.rewards["joint_torque_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torque_rate_l2,
        weight=0.0,
    )

    cfg.metrics["single_leg_success"] = MetricsTermCfg(
        func=microduck_mdp.single_leg_hold_success,
        params={
            "command_name": COMMAND_NAME,
            "sensor_name": FEET_SENSOR,
            "asset_cfg": FEET_CFG,
            "nonfoot_sensor_name": NONFOOT_SENSOR,
            "hold_s": 1.0,
        },
    )
    for side_name, support_side in (("left", -1), ("right", 1)):
        cfg.metrics[f"single_leg_success_{side_name}"] = MetricsTermCfg(
            func=microduck_mdp.single_leg_success_rate_for_side,
            params={
                "command_name": COMMAND_NAME,
                "sensor_name": FEET_SENSOR,
                "asset_cfg": FEET_CFG,
                "support_side": support_side,
                "nonfoot_sensor_name": NONFOOT_SENSOR,
                "hold_s": 1.0,
            },
        )

    cfg.terminations["fell_over"].params["limit_angle"] = math.radians(45.0)
    cfg.terminations["root_too_low"] = TerminationTermCfg(
        func=microduck_mdp.root_height_below,
        time_out=False,
        params={"min_height": 0.075},
    )

    cfg.events["push_robot"].params["velocity_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
    }

    for name in (
        "standing_envs",
        "head_pose_range",
        "body_pose_range",
        "head_pose_bias_weight",
    ):
        cfg.curriculum.pop(name, None)

    cfg.curriculum["swing_height_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "swing_height",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": 500 * NUM_STEPS_PER_ENV, "weight": 0.5},
                {"step": 1000 * NUM_STEPS_PER_ENV, "weight": 2.0},
            ],
        },
    )
    cfg.curriculum["swing_contact_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "swing_contact",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": 500 * NUM_STEPS_PER_ENV, "weight": -1.0},
                {"step": 1000 * NUM_STEPS_PER_ENV, "weight": -4.0},
            ],
        },
    )
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": -0.05},
                {"step": 1000 * NUM_STEPS_PER_ENV, "weight": -0.1},
                {"step": 2000 * NUM_STEPS_PER_ENV, "weight": -0.3},
            ],
        },
    )
    cfg.curriculum["torque_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "joint_torque_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": 1500 * NUM_STEPS_PER_ENV, "weight": -1e-3},
            ],
        },
    )
    cfg.curriculum["push_magnitude"] = CurriculumTermCfg(
        func=microduck_mdp.push_curriculum,
        params={
            "event_name": "push_robot",
            "push_stages": [
                {
                    "step": 0,
                    "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)},
                },
                {
                    "step": 2500 * NUM_STEPS_PER_ENV,
                    "velocity_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                },
            ],
        },
    )
    return cfg


def make_microduck_single_leg_stand_strict_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Continuation phase whose only positive task reward is a valid hold."""
    cfg = make_microduck_single_leg_stand_env_cfg(play=play)
    if not play:
        cfg.episode_length_s = STRICT_EPISODE_LENGTH_S
        cfg.commands[COMMAND_NAME].resampling_time_range = (
            STRICT_EPISODE_LENGTH_S,
            STRICT_EPISODE_LENGTH_S,
        )

    for name in (
        "single_leg_com",
        "support_contact",
        "swing_height",
        "height_stand",
        "body_ang_vel",
        "angular_momentum",
        "excess_tilt",
    ):
        cfg.rewards.pop(name)
    cfg.terminations.pop("fell_over", None)
    for name in (
        "swing_height_weight",
        "swing_contact_weight",
        "action_rate_weight",
        "torque_rate_weight",
        "push_magnitude",
    ):
        cfg.curriculum.pop(name)

    cfg.rewards["strict_single_leg_hold"] = RewardTermCfg(
        func=microduck_mdp.single_leg_hold_progress_reward,
        weight=10.0,
        params={
            "command_name": COMMAND_NAME,
            "sensor_name": FEET_SENSOR,
            "asset_cfg": FEET_CFG,
            "nonfoot_sensor_name": NONFOOT_SENSOR,
            "target_s": 1.0,
            "min_clearance": -1.0,
            "max_tilt_deg": 180.0,
            "max_lin_vel": float("inf"),
            "max_ang_vel": float("inf"),
            "require_com_inside": False,
        },
    )
    cfg.rewards["swing_contact"].weight = -10.0
    cfg.rewards["touchdown"] = RewardTermCfg(
        func=microduck_mdp.single_leg_touchdown_cost,
        weight=-5.0,
        params={"command_name": COMMAND_NAME, "sensor_name": FEET_SENSOR},
    )
    # RewardManager scales by dt=0.02, so this is a fixed -10 per failed episode.
    cfg.rewards["failed_episode"] = RewardTermCfg(
        func=mdp.is_terminated,
        weight=-500.0,
    )
    cfg.rewards["action_rate_l2"].weight = -0.3
    cfg.rewards["joint_torque_rate_l2"].weight = -1e-3
    for metric in cfg.metrics.values():
        if "single_leg_success" in metric.func.__name__ or metric.func is microduck_mdp.single_leg_hold_success:
            metric.params["min_clearance"] = -1.0
            metric.params["max_tilt_deg"] = 180.0
            metric.params["max_lin_vel"] = float("inf")
            metric.params["max_ang_vel"] = float("inf")
            metric.params["require_com_inside"] = False
    return cfg


MicroduckSingleLegStandRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=SYMMETRY_CFG,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="single_leg_stand",
    run_name="single_leg_stand",
    save_interval=250,
    num_steps_per_env=NUM_STEPS_PER_ENV,
    max_iterations=6_000,
)

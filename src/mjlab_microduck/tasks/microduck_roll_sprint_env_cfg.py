"""Microduck repeated supported forward-roll sprint task.

This is a separate deployable policy from the one-roll-and-stand roulade.  A
fixed forty-second horizon makes sustained forward distance and speed the
natural objective while leaving enough time to prove a 20 m race. Each cycle
must still be a supported sagittal roll with a flat head-top contact before its
distance is released to PPO.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    MetricsTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_roulade_env_cfg import (
    COM_RANDOMIZATION_RANGE,
    ENABLE_ARMATURE_RANDOMIZATION,
    ENABLE_COM_RANDOMIZATION,
    ENABLE_ENCODER_BIAS,
    ENABLE_HEAD_COM_RANDOMIZATION,
    ENABLE_IMU_ORIENTATION_RANDOMIZATION,
    ENABLE_JOINT_FRICTION_RANDOMIZATION,
    ENABLE_MASS_INERTIA_RANDOMIZATION,
    ENABLE_SYMMETRY,
    HEAD_COM_RANDOMIZATION_RANGE,
    MIDROLL_OMEGA_RANGE,
    MIDROLL_PITCH_MAX,
    MIDROLL_PITCH_MIN,
    ROULADE_FORWARD_VEL_RANGE,
    TUCK_OVERRIDES,
    make_microduck_roulade_env_cfg,
)
from mjlab_microduck.tasks.symmetry import SYMMETRY_CFG, PpoWithSymmetryCfg

EPISODE_LENGTH_S = 40.0
TARGET_DISTANCE_M = 20.0


def make_microduck_roll_sprint_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create the repeated-roll sprint configuration."""
    cfg = make_microduck_roulade_env_cfg(play=play)
    cfg.episode_length_s = EPISODE_LENGTH_S

    # The roulade factory gives us the complete sim2real stack, sensors, 61D
    # observation contract, and flat plane. Keep only sprint-safe regularisers
    # and replace every one-shot landing/standing attractor.
    keep_rewards = {
        "action_rate_l2",
        "joint_torque_rate_l2",
        "body_ang_vel",
        "angular_momentum",
        "self_collisions",
    }
    for name in list(cfg.rewards):
        if name not in keep_rewards:
            del cfg.rewards[name]

    cfg.rewards["roll_sprint_progress"] = RewardTermCfg(
        func=microduck_mdp.roll_sprint_progress,
        weight=1.5,
        params={
            "max_paid_rate": 5.0,
            "lane_half_width": microduck_mdp._ROLL_SPRINT_LANE_HALF_WIDTH,
        },
    )
    # Primary race score: signed net forward frontier released only after a
    # valid full roll. The MDP term rejects revisits, backward travel, and
    # positive-path integration within a cycle.
    cfg.rewards["roll_sprint_distance"] = RewardTermCfg(
        func=microduck_mdp.roll_sprint_distance,
        weight=32.0,
    )
    cfg.rewards["roll_sprint_cycle_rate"] = RewardTermCfg(
        func=microduck_mdp.roll_sprint_cycle_rate,
        weight=1.0,
    )
    cfg.rewards["roll_sprint_recovery"] = RewardTermCfg(
        func=microduck_mdp.roll_sprint_recovery_rate,
        # A53 measured recovery extinction before the first lane stage. This
        # remains a single latched event and is still much smaller than the
        # valid frontier released by even one useful roll.
        weight=1.0,
    )
    cfg.rewards["roll_sprint_recovered_reroll"] = RewardTermCfg(
        func=microduck_mdp.roll_sprint_recovered_reroll_rate,
        weight=0.5,
    )
    cfg.rewards["roll_sprint_head_pivot"] = RewardTermCfg(
        func=microduck_mdp.roll_sprint_head_pivot,
        weight=0.25,
        params={"rate_norm": 2.0},
    )
    cfg.rewards["roll_sprint_invalid_cycle"] = RewardTermCfg(
        func=microduck_mdp.roll_sprint_invalid_cycle_rate,
        weight=0.0,
    )
    cfg.rewards["roll_sprint_overspeed"] = RewardTermCfg(
        func=microduck_mdp.roulade_overspeed_penalty,
        weight=0.0,
        params={"omega_max": 7.0},
    )
    cfg.rewards["roll_sprint_sagittal"] = RewardTermCfg(
        func=microduck_mdp.roulade_sagittal_penalty,
        weight=-0.05,
    )
    cfg.rewards["roll_sprint_lateral_vel"] = RewardTermCfg(
        func=microduck_mdp.roulade_lateral_velocity_penalty,
        weight=-0.35,
    )
    cfg.rewards["roll_sprint_straightness"] = RewardTermCfg(
        func=microduck_mdp.roll_sprint_straightness_penalty,
        weight=-3.0,
        params={"deadband": 0.01},
    )
    cfg.rewards["roll_sprint_lane_centering"] = RewardTermCfg(
        func=microduck_mdp.roll_sprint_lane_centering_progress,
        weight=4.0,
    )
    cfg.rewards["roll_sprint_flatness"] = RewardTermCfg(
        func=microduck_mdp.roulade_flatness_penalty,
        weight=-0.25,
    )
    cfg.rewards["roll_sprint_impact"] = RewardTermCfg(
        func=microduck_mdp.trunk_vertical_accel_penalty,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )

    cfg.rewards["action_rate_l2"].weight = 0.0
    cfg.rewards["joint_torque_rate_l2"].weight = 0.0
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = 0.0
    cfg.rewards["angular_momentum"].weight = 0.0
    cfg.rewards["self_collisions"].weight = 0.0

    # The cyclic reset wraps the existing reverse-roll pose sampler but owns
    # all sprint counters, including the reset heading and completed count.
    old_reset = cfg.events.pop("set_roulade_state")
    cfg.events["set_roll_sprint_state"] = EventTermCfg(
        func=microduck_mdp.reset_roll_sprint_state,
        mode=old_reset.mode,
        params={
            "standing_prob": 1.0 if play else 0.40,
            "midroll_prob": 0.0 if play else 0.20,
            "postroll_prob": 0.0 if play else 0.40,
            "standing_z_min": old_reset.params["standing_z_min"],
            "standing_z_max": old_reset.params["standing_z_max"],
            "standing_tilt_max": old_reset.params["standing_tilt_max"],
            "forward_vel_range": ROULADE_FORWARD_VEL_RANGE,
            "midroll_pitch_min": MIDROLL_PITCH_MIN,
            "midroll_pitch_max": MIDROLL_PITCH_MAX,
            "midroll_z_min": 0.05,
            "midroll_z_max": 0.10,
            "midroll_omega_range": MIDROLL_OMEGA_RANGE,
            "tuck_overrides": TUCK_OVERRIDES,
            "tuck_factor_range": old_reset.params["tuck_factor_range"],
            "joint_noise_std": old_reset.params["joint_noise_std"],
        },
    )

    # Keep the command slots live and update the fixed-horizon command timing,
    # even though the visible command remains zero-padded for deployment parity.
    command = cfg.commands["twist"]
    command.resampling_time_range = (EPISODE_LENGTH_S, EPISODE_LENGTH_S * 2)

    # The A35 bootstrap has a 90D privileged critic: its final 16D block is
    # stair-specific state. Keep the critic contract shape-compatible so its
    # normalizer, MLP, optimizer, and value estimates can load, while leaving
    # the deployable actor at the shared 61D contract. Flat sprint has no
    # equivalent privileged state, so the slot is explicitly zero-padded.
    cfg.observations["critic"].terms["roll_sprint_critic_padding"] = ObservationTermCfg(
        func=microduck_mdp.zero_command_padding,
        params={"dim": 16},
    )

    # These are cumulative latch counters sampled at episode end. They appear
    # in TensorBoard and therefore in the existing dashboard without affecting
    # PPO reward or introducing any per-step upright annuity.
    cfg.metrics["roll_sprint_recovery_count"] = MetricsTermCfg(
        func=microduck_mdp.roll_sprint_recovery_count,
        reduce="last",
    )
    cfg.metrics["roll_sprint_recovered_reroll_count"] = MetricsTermCfg(
        func=microduck_mdp.roll_sprint_recovered_reroll_count,
        reduce="last",
    )
    cfg.metrics["roll_sprint_mean_recovery_latency_s"] = MetricsTermCfg(
        func=microduck_mdp.roll_sprint_mean_recovery_latency,
        reduce="last",
    )

    # Rebuild curricula so no one-roll landing weights or old reward names can
    # silently remain active. DR stages are retained from the roulade recipe.
    for name in list(cfg.curriculum):
        del cfg.curriculum[name]

    cfg.curriculum["roll_sprint_spawn_mix"] = CurriculumTermCfg(
        func=microduck_mdp.event_param_curriculum,
        params={
            "event_name": "set_roll_sprint_state",
            "param_stages": [
                {
                    "step": 0,
                    "params": {
                        "standing_prob": 0.40,
                        "midroll_prob": 0.20,
                        "postroll_prob": 0.40,
                    },
                },
                {
                    "step": 750 * 24,
                    "params": {
                        "standing_prob": 0.50,
                        "midroll_prob": 0.20,
                        "postroll_prob": 0.30,
                    },
                },
                {
                    "step": 1750 * 24,
                    "params": {
                        "standing_prob": 0.70,
                        "midroll_prob": 0.15,
                        "postroll_prob": 0.15,
                    },
                },
                {
                    "step": 2750 * 24,
                    "params": {
                        "standing_prob": 0.90,
                        "midroll_prob": 0.05,
                        "postroll_prob": 0.05,
                    },
                },
                {
                    "step": 3500 * 24,
                    "params": {
                        "standing_prob": 1.0,
                        "midroll_prob": 0.0,
                        "postroll_prob": 0.0,
                    },
                },
            ],
        },
    )

    lane_width_stages = (
        [{"step": 0, "width": microduck_mdp._ROLL_SPRINT_LANE_HALF_WIDTH}]
        if play
        else [
            {
                "step": 0,
                "width": microduck_mdp._ROLL_SPRINT_BOOTSTRAP_LANE_HALF_WIDTH,
            },
            # A53 lost recovered-rerolls even before its 250-iteration lane
            # step. Keep the recovery-rich phase wide, then harden one axis at
            # a time after the cyclic transition has had time to consolidate.
            {"step": 1000 * 24, "width": 0.60},
            {"step": 2000 * 24, "width": 0.40},
            {"step": 2800 * 24, "width": 0.28},
            {"step": 3400 * 24, "width": 0.20},
            {
                "step": 3750 * 24,
                "width": microduck_mdp._ROLL_SPRINT_LANE_HALF_WIDTH,
            },
        ]
    )
    cfg.curriculum["roll_sprint_lane_half_width"] = CurriculumTermCfg(
        func=microduck_mdp.roll_sprint_lane_half_width_curriculum,
        params={"width_stages": lane_width_stages},
    )
    cfg.curriculum["roll_sprint_invalid_cycle_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "roll_sprint_invalid_cycle",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": 2000 * 24, "weight": -0.5},
                {"step": 3000 * 24, "weight": -1.0},
                {"step": 3750 * 24, "weight": -2.0},
            ],
        },
    )

    if ENABLE_COM_RANDOMIZATION:
        cfg.curriculum["com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_com",
                "range_stages": [
                    {"step": 0, "range": COM_RANDOMIZATION_RANGE},
                    {"step": 2000 * 24, "range": 0.005},
                    {"step": 3000 * 24, "range": 0.01},
                    {"step": 3750 * 24, "range": 0.015},
                ],
            },
        )
    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.curriculum["head_com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_head_com",
                "range_stages": [
                    {"step": 0, "range": HEAD_COM_RANDOMIZATION_RANGE},
                    {"step": 2000 * 24, "range": 0.005},
                    {"step": 3000 * 24, "range": 0.01},
                ],
            },
        )

    cfg.curriculum["roll_sprint_distance_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "roll_sprint_distance",
            "weight_stages": [
                {"step": 0, "weight": 12.0},
                {"step": 750 * 24, "weight": 16.0},
                {"step": 1500 * 24, "weight": 24.0},
                {"step": 2500 * 24, "weight": 32.0},
            ],
        },
    )
    cfg.curriculum["roll_sprint_progress_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "roll_sprint_progress",
            "weight_stages": [
                {"step": 0, "weight": 1.5},
                {"step": 1250 * 24, "weight": 1.0},
                {"step": 2500 * 24, "weight": 0.25},
                # Dense rotation is only a bootstrap. The final race objective
                # is exclusively the valid-cycle forward frontier above.
                {"step": 3500 * 24, "weight": 0.0},
            ],
        },
    )
    cfg.curriculum["roll_sprint_head_pivot_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "roll_sprint_head_pivot",
            "weight_stages": [
                {"step": 0, "weight": 0.25},
                {"step": 3000 * 24, "weight": 0.10},
            ],
        },
    )
    if ENABLE_ARMATURE_RANDOMIZATION:
        # The reset event itself is retained from roulade. This assertion is a
        # cheap guard against changing the sim2real recipe accidentally.
        assert "randomize_armature" in cfg.events
    if ENABLE_MASS_INERTIA_RANDOMIZATION:
        assert "randomize_mass_inertia" in cfg.events
    if ENABLE_JOINT_FRICTION_RANDOMIZATION:
        assert "randomize_joint_friction" in cfg.events
    if ENABLE_ENCODER_BIAS:
        assert "encoder_bias" in cfg.events
    if ENABLE_IMU_ORIENTATION_RANDOMIZATION:
        assert cfg.observations["actor"].terms["base_ang_vel"].func.__name__ == (
            "base_ang_vel_imu_misaligned"
        )

    return cfg


MicroduckRollSprintRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 0.35,
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
        # Warm-started roll policies already have broad stochasticity. An
        # entropy bonus drove distribution.std_param from 0.67 to 8.20 and
        # erased the learned roll within 700 iterations, so fine-tuning must
        # optimize the race objective without paying for additional noise.
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=SYMMETRY_CFG if ENABLE_SYMMETRY else None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="microduck_roll_sprint",
    run_name="microduck_roll_sprint",
    # Unique checkpoints are audited every 100 iterations. The dashboard
    # sampler repeats the latest video on its independent 150-second cadence.
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=4_000,
)

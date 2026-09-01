"""Microduck repeated supported forward-roll and backroll sprint tasks.

This is a separate deployable policy from the one-roll-and-stand roulade.  A
fixed forty-second horizon makes sustained forward distance and speed the
natural objective while leaving enough time to prove a 10 m race. Each cycle
must still be a supported sagittal roll with a flat head-top contact before its
distance is released to PPO.
"""

import copy
import math

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
TARGET_DISTANCE_M = 10.0


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
            "road_half_width": microduck_mdp._ROLL_SPRINT_ROAD_HALF_WIDTH,
            "road_safe_half_width": (
                microduck_mdp._ROLL_SPRINT_ROAD_SAFE_HALF_WIDTH
            ),
        },
    )
    cfg.rewards["roll_sprint_directional_bootstrap"] = RewardTermCfg(
        func=microduck_mdp.roll_sprint_directional_bootstrap,
        # Forward continuation needs no partial-cycle translation reward. The
        # separate backroll task enables and later removes this discovery aid.
        weight=0.0,
    )
    # Primary race score: signed net forward frontier released only after a
    # valid full roll. The MDP term rejects revisits, backward travel, and
    # positive-path integration within a cycle. A61 checkpoints 100 and 200
    # recovered reliably but regressed frontier, so distance is dominant from
    # iteration zero instead of waiting until iteration 500.
    cfg.rewards["roll_sprint_distance"] = RewardTermCfg(
        func=microduck_mdp.roll_sprint_distance,
        weight=32.0,
    )
    cfg.rewards["roll_sprint_cycle_rate"] = RewardTermCfg(
        func=microduck_mdp.roll_sprint_cycle_rate,
        # A68 showed that quadrupling this term reduced deterministic valid
        # cycles and frontier. Keep the proven A67 scale and fix the blocked
        # self-right-to-reposition transition directly.
        weight=1.0,
    )
    cfg.rewards["roll_sprint_recovery"] = RewardTermCfg(
        func=microduck_mdp.roll_sprint_recovery_rate,
        # A53 measured recovery extinction before the first course stage. This
        # remains a single latched event and is still much smaller than the
        # valid frontier released by even one useful roll.
        weight=1.0,
    )
    cfg.rewards["roll_sprint_reposition"] = RewardTermCfg(
        func=microduck_mdp.roll_sprint_reposition_rate,
        # A modest one-shot transition reward, worth far less than one valid
        # frontier cycle and impossible to farm by standing inside the road.
        weight=2.0,
    )
    cfg.rewards["roll_sprint_recovered_reroll"] = RewardTermCfg(
        func=microduck_mdp.roll_sprint_recovered_reroll_rate,
        # One-shot only: reward the missing recovery-to-reroll transition,
        # never standing or merely remaining upright after recovery.
        # A68 showed that doubling this term did not increase deterministic
        # rerolls. Preserve the A67 scale while sequencing the command modes.
        weight=4.0,
    )
    cfg.rewards["roll_sprint_self_right_upright"] = RewardTermCfg(
        func=microduck_mdp.roll_sprint_self_right_upright_progress,
        weight=5.0,
    )
    cfg.rewards["roll_sprint_self_right_height"] = RewardTermCfg(
        func=microduck_mdp.roll_sprint_self_right_height_progress,
        weight=30.0,
    )
    cfg.rewards["roll_sprint_self_right_upward"] = RewardTermCfg(
        func=microduck_mdp.roll_sprint_self_right_upward_velocity,
        weight=1.0,
    )
    cfg.rewards["roll_sprint_self_right_fallen_tax"] = RewardTermCfg(
        func=microduck_mdp.roll_sprint_self_right_fallen_tax,
        weight=-0.25,
    )
    cfg.rewards["roll_sprint_self_right_success"] = RewardTermCfg(
        func=microduck_mdp.roll_sprint_self_right_success_rate,
        # A72/A75 continuation audits showed that a weight-5 completion pulse
        # lets PPO replace the race skill with the easier "become upright"
        # edge before repositioning or rerolling.  Keep a modest one-shot
        # completion signal while the distance and recovered-reroll objectives
        # train the full transition.
        weight=1.0,
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
        func=microduck_mdp.roll_sprint_sagittal_penalty,
        weight=-0.05,
    )
    cfg.rewards["roll_sprint_lateral_vel"] = RewardTermCfg(
        func=microduck_mdp.roll_sprint_lateral_velocity_penalty,
        weight=-0.35,
    )
    cfg.rewards["roll_sprint_straightness"] = RewardTermCfg(
        func=microduck_mdp.roll_sprint_straightness_penalty,
        weight=-3.0,
        params={
            "road_safe_half_width": (
                microduck_mdp._ROLL_SPRINT_ROAD_SAFE_HALF_WIDTH
            )
        },
    )
    cfg.rewards["roll_sprint_road_return"] = RewardTermCfg(
        func=microduck_mdp.roll_sprint_road_return_progress,
        # A65 checkpoints 300 and 400 were already physically fast enough but
        # repeatedly left the shared road. This signed potential charges the
        # departure and only repays a real return, with no centering annuity.
        # A67 checkpoints 200 and 300 showed that increasing this to 8 reduced
        # frontier, recovery, heading, and drift together. Keep the proven
        # weight 4 while cadence and rerolling improve.
        weight=4.0,
    )
    cfg.rewards["roll_sprint_heading_alignment"] = RewardTermCfg(
        func=microduck_mdp.roll_sprint_heading_alignment_progress,
        # Signed potential change only: turning away is charged before a
        # correction can repay it, and holding an aligned heading pays zero.
        weight=1.0,
    )
    cfg.rewards["roll_sprint_flatness"] = RewardTermCfg(
        func=microduck_mdp.roll_sprint_flatness_penalty,
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
            "standing_prob": 1.0 if play else 0.45,
            "midroll_prob": 0.0 if play else 0.10,
            "postroll_prob": 0.0 if play else 0.15,
            "crouch_prob": 0.0 if play else 0.10,
            "ground_recovery_prob": 0.0 if play else 0.20,
            "ground_face_down_prob": 0.25,
            "ground_face_up_prob": 0.25,
            "ground_left_prob": 0.25,
            "ground_right_prob": 0.25,
            "road_interior_prob": 1.0 if play else 0.70,
            "road_edge_prob": 0.0 if play else 0.20,
            "road_return_prob": 0.0 if play else 0.10,
            "recovery_road_return_prob": 0.0 if play else 0.35,
            # A71 learned useful course correction by checkpoint 100, then a
            # 30% forced-heading bucket erased the roll skill by checkpoints
            # 200-300. Keep the targeted state present without making it the
            # dominant fine-tuning distribution.
            "heading_return_prob": 0.0 if play else 0.10,
            "heading_return_min_rad": (
                microduck_mdp._ROLL_SPRINT_HEADING_RETURN_RESET_MIN_RAD
            ),
            "heading_return_max_rad": (
                microduck_mdp._ROLL_SPRINT_HEADING_RETURN_RESET_MAX_RAD
            ),
            "standing_z_min": old_reset.params["standing_z_min"],
            "standing_z_max": old_reset.params["standing_z_max"],
            "standing_tilt_max": old_reset.params["standing_tilt_max"],
            "forward_vel_range": ROULADE_FORWARD_VEL_RANGE,
            "midroll_pitch_min": MIDROLL_PITCH_MIN,
            "midroll_pitch_max": MIDROLL_PITCH_MAX,
            "midroll_z_min": 0.05,
            "midroll_z_max": 0.10,
            "midroll_omega_range": MIDROLL_OMEGA_RANGE,
            "midroll_forward_vel_range": (0.0, 0.0),
            "tuck_overrides": TUCK_OVERRIDES,
            "tuck_factor_range": old_reset.params["tuck_factor_range"],
            "joint_noise_std": old_reset.params["joint_noise_std"],
        },
    )

    # Keep the 61D command block shape-compatible. The roll state uses its
    # existing twist slots for self-right mode, road return, and reset-heading
    # correction without adding an actor observation.
    command = cfg.commands["twist"]
    command.resampling_time_range = (EPISODE_LENGTH_S, EPISODE_LENGTH_S * 2)
    for group in ("actor", "critic"):
        command_obs = cfg.observations[group].terms["command"]
        command_obs.func = microduck_mdp.roll_sprint_reposition_command
        command_obs.params = {"command_name": "twist"}

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
    cfg.metrics["roll_sprint_reposition_count"] = MetricsTermCfg(
        func=microduck_mdp.roll_sprint_reposition_count,
        reduce="last",
    )
    cfg.metrics["roll_sprint_mean_reposition_latency_s"] = MetricsTermCfg(
        func=microduck_mdp.roll_sprint_mean_reposition_latency,
        reduce="last",
    )
    cfg.metrics["roll_sprint_self_right_attempt_count"] = MetricsTermCfg(
        func=microduck_mdp.roll_sprint_self_right_attempt_count,
        reduce="last",
    )
    cfg.metrics["roll_sprint_self_right_success_count"] = MetricsTermCfg(
        func=microduck_mdp.roll_sprint_self_right_success_count,
        reduce="last",
    )
    cfg.metrics["roll_sprint_self_right_success_rate"] = MetricsTermCfg(
        func=microduck_mdp.roll_sprint_self_right_success_fraction,
        reduce="last",
    )
    cfg.metrics["roll_sprint_mean_self_right_latency_s"] = MetricsTermCfg(
        func=microduck_mdp.roll_sprint_mean_self_right_latency,
        reduce="last",
    )
    cfg.metrics["roll_sprint_frontier_after_self_right_m"] = MetricsTermCfg(
        func=microduck_mdp.roll_sprint_frontier_after_self_right,
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
                        "standing_prob": 0.45,
                        "midroll_prob": 0.10,
                        "postroll_prob": 0.15,
                        "crouch_prob": 0.10,
                        "ground_recovery_prob": 0.20,
                        "ground_face_down_prob": 0.25,
                        "ground_face_up_prob": 0.25,
                        "ground_left_prob": 0.25,
                        "ground_right_prob": 0.25,
                        "recovery_road_return_prob": 0.35,
                        "heading_return_prob": 0.10,
                    },
                },
                {
                    "step": 400 * 24,
                    "params": {
                        "standing_prob": 0.45,
                        "midroll_prob": 0.05,
                        "postroll_prob": 0.15,
                        "crouch_prob": 0.15,
                        "ground_recovery_prob": 0.20,
                        "ground_face_down_prob": 0.25,
                        "ground_face_up_prob": 0.25,
                        "ground_left_prob": 0.25,
                        "ground_right_prob": 0.25,
                        "recovery_road_return_prob": 0.30,
                        "heading_return_prob": 0.10,
                    },
                },
                {
                    "step": 1000 * 24,
                    "params": {
                        "standing_prob": 0.55,
                        "midroll_prob": 0.05,
                        "postroll_prob": 0.10,
                        "crouch_prob": 0.10,
                        "ground_recovery_prob": 0.20,
                        "ground_face_down_prob": 0.25,
                        "ground_face_up_prob": 0.25,
                        "ground_left_prob": 0.25,
                        "ground_right_prob": 0.25,
                        "recovery_road_return_prob": 0.20,
                        "heading_return_prob": 0.08,
                    },
                },
                {
                    "step": 2000 * 24,
                    "params": {
                        "standing_prob": 0.65,
                        "midroll_prob": 0.0,
                        "postroll_prob": 0.10,
                        "crouch_prob": 0.05,
                        "ground_recovery_prob": 0.20,
                        "ground_face_down_prob": 0.25,
                        "ground_face_up_prob": 0.25,
                        "ground_left_prob": 0.25,
                        "ground_right_prob": 0.25,
                        "recovery_road_return_prob": 0.10,
                        "heading_return_prob": 0.05,
                    },
                },
            ],
        },
    )

    road_width_stages = [
        {"step": 0, "width": microduck_mdp._ROLL_SPRINT_ROAD_HALF_WIDTH}
    ]
    cfg.curriculum["roll_sprint_road_half_width"] = CurriculumTermCfg(
        func=microduck_mdp.roll_sprint_lane_half_width_curriculum,
        params={"width_stages": road_width_stages},
    )
    cfg.curriculum["roll_sprint_road_return_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "roll_sprint_road_return",
            "weight_stages": [
                {"step": 0, "weight": 4.0},
            ],
        },
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
                {"step": 0, "weight": 32.0},
            ],
        },
    )
    for curriculum_name, reward_name, weight_stages in (
        (
            "roll_sprint_self_right_upward_weight",
            "roll_sprint_self_right_upward",
            [
                {"step": 0, "weight": 1.0},
                {"step": 1000 * 24, "weight": 2.0},
            ],
        ),
        (
            "roll_sprint_self_right_fallen_tax_weight",
            "roll_sprint_self_right_fallen_tax",
            [
                {"step": 0, "weight": -0.25},
                {"step": 1000 * 24, "weight": -0.5},
            ],
        ),
        (
            "roll_sprint_self_right_success_weight",
            "roll_sprint_self_right_success",
            [
                {"step": 0, "weight": 1.0},
                {"step": 1000 * 24, "weight": 10.0},
            ],
        ),
    ):
        cfg.curriculum[curriculum_name] = CurriculumTermCfg(
            func=microduck_mdp.reward_weight,
            params={"reward_name": reward_name, "weight_stages": weight_stages},
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


def make_microduck_backroll_sprint_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create the separate repeated supported backroll sprint configuration."""
    cfg = make_microduck_roll_sprint_env_cfg(play=play)
    reset_params = cfg.events["set_roll_sprint_state"].params
    reset_params["roll_direction"] = -1.0
    # Standing forward and standing backroll starts share the same
    # proprioception. Expose the dedicated reverse mode in the existing
    # six-dimensional body-command padding without changing the 61D actor
    # contract or the twist-slot semantics.
    for group in ("actor", "critic"):
        cfg.observations[group].terms["body_command"].func = (
            microduck_mdp.roll_sprint_backroll_direction_flag
        )
        cfg.observations[group].terms["body_command"].params = {"dim": 6}
    # Keep the reverse curriculum balanced between phase-zero launch, targeted
    # completion, and recovery. A94 learned self-righting (about 55% success)
    # but had no valid reverse cycles, so the early mid-roll bucket needs to be
    # large enough to expose the completion latch before it is tapered away.
    # Give both reset buckets a small, direction-matched launch velocity.  A
    # standing robot otherwise has no observable cue that this dedicated
    # policy must choose the reverse roll sign, while mid-roll starts should
    # carry enough course-aligned momentum to keep the late-roll lesson alive.
    reset_params["forward_vel_range"] = (0.08, 0.20)
    reset_params["midroll_forward_vel_range"] = (0.25, 0.65)
    reset_params["recovery_road_return_prob"] = 0.15
    reset_params["heading_return_prob"] = 0.05
    reset_params.update(
        midroll_pitch_min=math.radians(320.0),
        midroll_pitch_max=math.radians(355.0),
        midroll_omega_range=(3.0, 5.5),
    )
    if not play:
        reset_params.update(
            standing_prob=0.35,
            midroll_prob=0.30,
            postroll_prob=0.15,
            crouch_prob=0.10,
            ground_recovery_prob=0.10,
        )
        spawn_stages = cfg.curriculum["roll_sprint_spawn_mix"].params[
            "param_stages"
        ]
        stage_mixes = (
            ((0.35, 0.30, 0.15, 0.10, 0.10), (320.0, 355.0, (3.0, 5.5))),
            ((0.45, 0.25, 0.15, 0.05, 0.10), (260.0, 350.0, (2.0, 5.0))),
            ((0.55, 0.15, 0.10, 0.05, 0.15), (140.0, 340.0, (0.75, 3.5))),
            ((0.65, 0.00, 0.10, 0.05, 0.20), (50.0, 340.0, (0.0, 3.0))),
        )
        for stage, (mix, roll_window) in zip(spawn_stages, stage_mixes, strict=True):
            pitch_min, pitch_max, omega_range = roll_window
            stage["params"].update(
                standing_prob=mix[0],
                midroll_prob=mix[1],
                postroll_prob=mix[2],
                crouch_prob=mix[3],
                ground_recovery_prob=mix[4],
                midroll_pitch_min=math.radians(pitch_min),
                midroll_pitch_max=math.radians(pitch_max),
                midroll_omega_range=omega_range,
            )

    # Backroll discovery needs more dense supported-rotation signal than a
    # continuation of an already learned forward roll. It remains far below
    # the valid frontier objective and decays once complete backrolls emerge.
    cfg.rewards["roll_sprint_progress"].weight = 24.0
    cfg.curriculum["roll_sprint_progress_weight"].params["weight_stages"] = [
        {"step": 0, "weight": 24.0},
        {"step": 300 * 24, "weight": 16.0},
        {"step": 800 * 24, "weight": 8.0},
        {"step": 1500 * 24, "weight": 2.0},
    ]
    cfg.rewards["roll_sprint_head_pivot"].weight = 0.5
    cfg.curriculum["roll_sprint_head_pivot_weight"].params["weight_stages"] = [
        {"step": 0, "weight": 0.5},
        {"step": 1000 * 24, "weight": 0.25},
        {"step": 3000 * 24, "weight": 0.10},
    ]
    cfg.rewards["roll_sprint_directional_bootstrap"].weight = 24.0
    cfg.curriculum["roll_sprint_directional_bootstrap_weight"] = (
        CurriculumTermCfg(
            func=microduck_mdp.reward_weight,
            params={
                "reward_name": "roll_sprint_directional_bootstrap",
                "weight_stages": [
                    {"step": 0, "weight": 24.0},
                    {"step": 300 * 24, "weight": 12.0},
                    {"step": 800 * 24, "weight": 4.0},
                    {"step": 1200 * 24, "weight": 0.0},
                ],
            },
        )
    )
    # A90 found a recovery basin before it found a reverse roll. Keep enough
    # potential shaping to teach the get-up transition, but taper it before
    # the roll-heavy discovery signal can be replaced by repeated recovery.
    for curriculum_name, reward_name, weight_stages in (
        (
            "backroll_self_right_upright_weight",
            "roll_sprint_self_right_upright",
            [
                {"step": 0, "weight": 2.0},
                {"step": 150 * 24, "weight": 0.75},
                {"step": 500 * 24, "weight": 0.5},
            ],
        ),
        (
            "backroll_self_right_height_weight",
            "roll_sprint_self_right_height",
            [
                {"step": 0, "weight": 5.0},
                {"step": 150 * 24, "weight": 2.0},
                {"step": 500 * 24, "weight": 1.0},
            ],
        ),
    ):
        cfg.curriculum[curriculum_name] = CurriculumTermCfg(
            func=microduck_mdp.reward_weight,
            params={"reward_name": reward_name, "weight_stages": weight_stages},
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
        # The compatible race champion is already a narrow local optimum.
        # A71 at 1e-4 improved through checkpoint 100 and catastrophically
        # forgot the roll by 200-300, so use conservative trust-region updates
        # for champion continuation instead of rediscovering the skill.
        clip_param=0.1,
        # Warm-started roll policies already have broad stochasticity. An
        # entropy bonus drove distribution.std_param from 0.67 to 8.20 and
        # erased the learned roll within 700 iterations, so fine-tuning must
        # optimize the race objective without paying for additional noise.
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=2.5e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.005,
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

MicroduckBackrollSprintRlCfg = copy.deepcopy(MicroduckRollSprintRlCfg)
MicroduckBackrollSprintRlCfg.experiment_name = "microduck_backroll_sprint"
MicroduckBackrollSprintRlCfg.run_name = "microduck_backroll_sprint"
MicroduckBackrollSprintRlCfg.save_interval = 50

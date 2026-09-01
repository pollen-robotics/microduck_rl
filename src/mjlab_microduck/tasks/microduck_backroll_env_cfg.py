"""One-shot and repeated grounded backward roulades for MicroDuck.

The policy starts from standing, tucks, rolls backward over the trunk and flat
head top, and lands on both feet. It deliberately has no course, distance,
lane, recovery, or repeated-cycle objective.

The repeated variant preserves the same 61D/14D policy contract, but rearms
after every valid feet landing and rewards only new, fast, sagittal progress
toward the next complete head-over cycle.
"""

import math
from copy import deepcopy

from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    MetricsTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_roulade_env_cfg import (
    TUCK_OVERRIDES,
    MicroduckRouladeRlCfg,
    make_microduck_roulade_env_cfg,
)

EPISODE_LENGTH_S = 5.0
REPEATED_EPISODE_LENGTH_S = 12.0

BACKROLL_CURRICULUM_STAGES = [
    {
        "params": {
            "standing_prob": 0.20,
            "midroll_prob": 0.80,
            "midroll_pitch_min": math.radians(180.0),
            "midroll_pitch_max": math.radians(340.0),
            "midroll_omega_range": (1.0, 3.0),
        }
    },
    {
        "params": {
            "standing_prob": 0.30,
            "midroll_prob": 0.70,
            "midroll_pitch_min": math.radians(180.0),
            "midroll_pitch_max": math.radians(340.0),
            "midroll_omega_range": (1.0, 4.0),
        }
    },
    {
        "params": {
            "standing_prob": 0.40,
            "midroll_prob": 0.60,
            "midroll_pitch_min": math.radians(90.0),
            "midroll_pitch_max": math.radians(340.0),
            "midroll_omega_range": (2.0, 5.0),
        }
    },
    {
        "params": {
            "standing_prob": 0.60,
            "midroll_prob": 0.40,
            "midroll_pitch_min": math.radians(20.0),
            "midroll_pitch_max": math.radians(340.0),
            "midroll_omega_range": (0.0, 4.0),
        }
    },
    {
        "params": {
            "standing_prob": 0.85,
            "midroll_prob": 0.15,
            "midroll_pitch_min": math.radians(20.0),
            "midroll_pitch_max": math.radians(340.0),
            "midroll_omega_range": (0.0, 3.0),
        }
    },
]

REPEATED_BACKROLL_CURRICULUM_STAGES = [
    {
        "params": {
            # Keep enough late-phase starts to teach the missing exit, but make
            # standing the majority.  With a 0.55 advancement threshold the
            # scaffolded 35% cannot advance this stage by itself while pure
            # standing behavior is still failing.
            "standing_prob": 0.65,
            "midroll_prob": 0.35,
            "midroll_pitch_min": math.radians(260.0),
            "midroll_pitch_max": math.radians(340.0),
            "midroll_omega_range": (0.5, 2.5),
            "mastery_cycles": 1,
        }
    },
    {
        "params": {
            "standing_prob": 0.70,
            "midroll_prob": 0.30,
            "midroll_pitch_min": math.radians(180.0),
            "midroll_pitch_max": math.radians(340.0),
            "midroll_omega_range": (1.0, 4.0),
            "mastery_cycles": 1,
        }
    },
    {
        "params": {
            "standing_prob": 0.75,
            "midroll_prob": 0.25,
            "midroll_pitch_min": math.radians(90.0),
            "midroll_pitch_max": math.radians(340.0),
            "midroll_omega_range": (2.0, 5.0),
            "mastery_cycles": 1,
        }
    },
    {
        "params": {
            "standing_prob": 0.85,
            "midroll_prob": 0.15,
            "midroll_pitch_min": math.radians(20.0),
            "midroll_pitch_max": math.radians(340.0),
            "midroll_omega_range": (0.0, 4.0),
            "mastery_cycles": 2,
        }
    },
    {
        "params": {
            "standing_prob": 0.95,
            "midroll_prob": 0.05,
            "midroll_pitch_min": math.radians(20.0),
            "midroll_pitch_max": math.radians(340.0),
            "midroll_omega_range": (0.0, 3.0),
            "mastery_cycles": 2,
        }
    },
    {
        "params": {
            "standing_prob": 1.0,
            "midroll_prob": 0.0,
            "midroll_pitch_min": math.radians(20.0),
            "midroll_pitch_max": math.radians(340.0),
            "midroll_omega_range": (0.0, 3.0),
            "mastery_cycles": 3,
        }
    },
]


def _ground_contact_sensor(name: str, pattern: str) -> ContactSensorCfg:
    return ContactSensorCfg(
        name=name,
        primary=ContactMatch(mode="geom", pattern=pattern, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )


def make_microduck_backroll_env_cfg(play: bool = False):
    """Create the autonomous one-shot grounded-backroll task."""
    cfg = deepcopy(make_microduck_roulade_env_cfg(play=play))
    cfg.episode_length_s = EPISODE_LENGTH_S

    trunk_ground = _ground_contact_sensor(
        "trunk_ground_contact",
        r"^trunk_shell_collision$",
    )
    left_foot_ground = _ground_contact_sensor(
        "left_foot_ground_contact",
        r"^left_foot_collision$",
    )
    right_foot_ground = _ground_contact_sensor(
        "right_foot_ground_contact",
        r"^right_foot_collision$",
    )
    cfg.scene.sensors = (
        *cfg.scene.sensors,
        trunk_ground,
        left_foot_ground,
        right_foot_ground,
    )

    cfg.rewards.clear()
    cfg.rewards["backroll_progress"] = RewardTermCfg(
        func=microduck_mdp.grounded_backroll_progress,
        weight=8.0,
        params={"target_angle": 2.0 * math.pi, "max_paid_rate": 5.0},
    )
    cfg.rewards["backroll_head_pivot"] = RewardTermCfg(
        func=microduck_mdp.grounded_backroll_head_pivot,
        weight=0.5,
        params={"rate_norm": 2.0},
    )
    cfg.rewards["backroll_completion_progress"] = RewardTermCfg(
        func=microduck_mdp.grounded_backroll_completion_progress,
        weight=4.0,
        params={
            "start_angle": math.radians(150.0),
            "target_angle": math.radians(350.0),
            "max_paid_rate": 6.0,
        },
    )
    cfg.rewards["backroll_upright_progress"] = RewardTermCfg(
        func=microduck_mdp.grounded_backroll_upright_progress,
        weight=1.5,
    )
    cfg.rewards["backroll_height_progress"] = RewardTermCfg(
        func=microduck_mdp.grounded_backroll_height_progress,
        weight=1.0,
    )
    cfg.rewards["backroll_success"] = RewardTermCfg(
        func=microduck_mdp.grounded_backroll_success_rate,
        weight=10.0,
    )
    cfg.rewards["backroll_invalid"] = RewardTermCfg(
        func=microduck_mdp.grounded_backroll_invalid_rate,
        weight=-2.0,
    )
    cfg.rewards["backroll_overspeed"] = RewardTermCfg(
        func=microduck_mdp.roulade_overspeed_penalty,
        weight=-0.1,
        params={"omega_max": 7.0},
    )
    cfg.rewards["backroll_sagittal"] = RewardTermCfg(
        func=microduck_mdp.roulade_sagittal_penalty,
        weight=-0.1,
    )
    cfg.rewards["backroll_lateral_velocity"] = RewardTermCfg(
        func=microduck_mdp.roulade_lateral_velocity_penalty,
        weight=-0.5,
    )
    cfg.rewards["backroll_flatness"] = RewardTermCfg(
        func=microduck_mdp.roulade_flatness_penalty,
        weight=-0.5,
    )
    cfg.rewards["action_rate_l2"] = RewardTermCfg(
        func=mdp.action_rate_l2,
        weight=-0.1,
    )
    cfg.rewards["gentle_landing"] = RewardTermCfg(
        func=microduck_mdp.trunk_vertical_accel_penalty,
        weight=0.002,
    )
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-0.05,
        params={"sensor_name": "self_collision"},
    )

    cfg.events.pop("set_roulade_state", None)
    first_stage = BACKROLL_CURRICULUM_STAGES[0]["params"]
    cfg.events["set_grounded_backroll_state"] = EventTermCfg(
        func=microduck_mdp.reset_grounded_backroll_state,
        mode="reset",
        params={
            **first_stage,
            "standing_z_min": 0.11,
            "standing_z_max": 0.12,
            "standing_tilt_max": math.radians(3.0),
            "yaw_range": (0.0, 0.0) if play else (-math.pi, math.pi),
            "midroll_z_min": 0.05,
            "midroll_z_max": 0.11,
            "tuck_overrides": TUCK_OVERRIDES,
            "tuck_factor_range": (0.5, 1.0),
            "joint_noise_std": 0.0 if play else 0.04,
        },
    )
    if play:
        cfg.events["set_grounded_backroll_state"].params.update(
            standing_prob=1.0,
            midroll_prob=0.0,
        )

    cfg.curriculum.clear()
    if not play:
        cfg.curriculum["backroll_phase"] = CurriculumTermCfg(
            func=microduck_mdp.grounded_backroll_curriculum,
            params={
                "event_name": "set_grounded_backroll_state",
                "stages": BACKROLL_CURRICULUM_STAGES,
                "window_episodes": 4096,
                "success_threshold": 0.70,
            },
        )

    cfg.terminations["backroll_success"] = TerminationTermCfg(
        func=microduck_mdp.grounded_backroll_success_termination,
        time_out=False,
    )
    cfg.terminations["backroll_invalid"] = TerminationTermCfg(
        func=microduck_mdp.grounded_backroll_invalid_termination,
        time_out=False,
    )

    cfg.metrics.clear()
    cfg.metrics["backroll_rotation_deg"] = MetricsTermCfg(
        func=microduck_mdp.grounded_backroll_rotation_deg,
    )
    cfg.metrics["backroll_trunk_contact"] = MetricsTermCfg(
        func=microduck_mdp.grounded_backroll_trunk_contact_latched,
    )
    cfg.metrics["backroll_head_top_contact"] = MetricsTermCfg(
        func=microduck_mdp.grounded_backroll_head_contact_latched,
    )
    cfg.metrics["backroll_max_air_gap_s"] = MetricsTermCfg(
        func=microduck_mdp.grounded_backroll_max_air_gap_s,
    )
    cfg.metrics["backroll_landing_hold_s"] = MetricsTermCfg(
        func=microduck_mdp.grounded_backroll_landing_hold_s,
    )
    cfg.metrics["backroll_success"] = MetricsTermCfg(
        func=microduck_mdp.grounded_backroll_success_fraction,
    )
    return cfg


def make_microduck_repeated_backroll_env_cfg(play: bool = False):
    """Create a repeated, fast, strictly sagittal grounded-backroll task."""
    cfg = deepcopy(make_microduck_backroll_env_cfg(play=play))
    cfg.episode_length_s = REPEATED_EPISODE_LENGTH_S

    first_stage = REPEATED_BACKROLL_CURRICULUM_STAGES[0]["params"]
    reset_cfg = cfg.events["set_grounded_backroll_state"]
    reset_cfg.params.update(
        **first_stage,
        repeat_mode=True,
        yaw_range=(0.0, 0.0) if play else (-math.pi, math.pi),
        joint_noise_std=0.0 if play else 0.03,
    )
    if play:
        reset_cfg.params.update(
            standing_prob=1.0,
            midroll_prob=0.0,
        )
    else:
        cfg.curriculum["backroll_phase"].params.update(
            stages=REPEATED_BACKROLL_CURRICULUM_STAGES,
            success_threshold=0.55,
            speed_reward_name="backroll_speed_progress",
            speed_reward_weights=[1.0, 1.0, 1.5, 2.0, 3.0, 3.0],
            invalid_reward_name="backroll_invalid",
            invalid_reward_weights=[-2.0, -3.0, -4.0, -6.0, -8.0, -10.0],
        )

    # A valid landing rearms the next cycle, so it is a reward pulse rather
    # than a termination. Invalid side/airborne/wrong-way solutions still end
    # the episode early to keep the replay buffer physically honest.
    cfg.terminations.pop("backroll_success", None)
    # A120 reached the first inverted support quickly and then parked.  Make
    # extension through the already-latched 180--350 degree arc materially
    # more valuable than repeatedly discovering the known 0--180 degree tuck.
    cfg.rewards["backroll_progress"].weight = 5.0
    cfg.rewards["backroll_contact_sequence"] = RewardTermCfg(
        func=microduck_mdp.grounded_backroll_contact_sequence,
        weight=2.0,
        params={"trunk_value": 1.0, "head_value": 2.0},
    )
    cfg.rewards["backroll_completion_progress"].weight = 18.0
    cfg.rewards["backroll_upright_progress"].weight = 5.0
    cfg.rewards["backroll_height_progress"].weight = 4.0
    cfg.rewards["backroll_rise_velocity"] = RewardTermCfg(
        func=microduck_mdp.grounded_backroll_rise_velocity,
        weight=0.75,
        params={
            "gate_lo": math.radians(180.0),
            "gate_hi": math.radians(260.0),
            "max_height": 0.125,
        },
    )
    cfg.rewards["backroll_success"] = RewardTermCfg(
        func=microduck_mdp.grounded_backroll_repeat_success_rate,
        weight=20.0,
        params={"later_cycle_bonus": 1.0, "max_bonus_cycles": 3},
    )
    cfg.rewards["backroll_speed_progress"] = RewardTermCfg(
        func=microduck_mdp.grounded_backroll_speed_progress,
        weight=1.0,
        params={"minimum_rate": 2.0, "target_rate": 6.0},
    )
    cfg.rewards["backroll_invalid"].weight = -2.0
    cfg.rewards["backroll_overspeed"].weight = -0.02
    cfg.rewards["backroll_overspeed"].params = {"omega_max": 7.5}
    cfg.rewards["backroll_sagittal"].weight = -0.50
    cfg.rewards["backroll_lateral_velocity"].weight = -1.0
    cfg.rewards["backroll_flatness"].weight = -2.0
    cfg.rewards["action_rate_l2"].weight = -0.05

    cfg.metrics["backroll_cycle_count"] = MetricsTermCfg(
        func=microduck_mdp.grounded_backroll_cycle_count,
    )
    cfg.metrics["backroll_max_lateral_axis_z"] = MetricsTermCfg(
        func=microduck_mdp.grounded_backroll_episode_max_lateral_axis_z,
    )
    cfg.metrics["backroll_max_offaxis_deg"] = MetricsTermCfg(
        func=microduck_mdp.grounded_backroll_episode_max_offaxis_deg,
    )
    return cfg


MicroduckBackrollRlCfg = deepcopy(MicroduckRouladeRlCfg)
MicroduckBackrollRlCfg.experiment_name = "microduck_backroll"
MicroduckBackrollRlCfg.run_name = "microduck_backroll"
MicroduckBackrollRlCfg.max_iterations = 4000
MicroduckBackrollRlCfg.save_interval = 50
MicroduckBackrollRlCfg.algorithm.learning_rate = 1.0e-3
MicroduckBackrollRlCfg.actor.distribution_cfg["init_std"] = 1.0

MicroduckRepeatedBackrollRlCfg = deepcopy(MicroduckBackrollRlCfg)
MicroduckRepeatedBackrollRlCfg.experiment_name = "microduck_repeated_backroll"
MicroduckRepeatedBackrollRlCfg.run_name = "microduck_repeated_backroll"
MicroduckRepeatedBackrollRlCfg.max_iterations = 4000
MicroduckRepeatedBackrollRlCfg.save_interval = 50
MicroduckRepeatedBackrollRlCfg.algorithm.learning_rate = 2.5e-5
MicroduckRepeatedBackrollRlCfg.algorithm.entropy_coef = 1.0e-3
MicroduckRepeatedBackrollRlCfg.actor.distribution_cfg["init_std"] = 0.25

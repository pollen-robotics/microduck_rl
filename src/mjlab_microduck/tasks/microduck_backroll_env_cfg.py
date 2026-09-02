"""One-shot and repeated grounded backward roulades for MicroDuck.

The policy starts from standing, tucks, rolls backward over the trunk and flat
head top, and lands on both feet. It deliberately has no course, distance,
lane, recovery, or repeated-cycle objective.

The repeated variant preserves the same 61D/14D policy contract, rearms after
every valid feet landing, and spends one bounded self-right/retry opportunity
instead of ending immediately after the first recoverable failed attempt.
"""

import math
from copy import deepcopy
from pathlib import Path

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
    MicroduckRouladeRlCfg,
    make_microduck_roulade_env_cfg,
)

EPISODE_LENGTH_S = 5.0
REPEATED_EPISODE_LENGTH_S = 12.0
BACKROLL_REFERENCE_STATE_PATH = str(
    Path(__file__).with_name("data") / "backroll_champion_reference_states.pt"
)

# Backward launch posture.  This is deliberately not the forward-roulade tuck:
# the two hip-pitch targets must reverse for the robot to fall onto its back
# before reaching the head.  A221 used the forward target and deterministically
# stopped on a non-top head contact around 180 degrees without ever contacting
# the trunk.  The sign choice is anchored to the measured 100-degree grounded
# backroll bridge in ``backroll_champion_reference_states.pt``; knees, ankles,
# and chin tuck retain their physically measured directions.
BACKROLL_TUCK_OVERRIDES = {
    2: 1.15,  # left hip_pitch: backward entry
    3: 1.25,  # left knee
    4: 1.05,  # left ankle
    5: -1.0,  # neck_pitch (chin tuck)
    6: 1.0,  # head_pitch (chin tuck)
    11: -1.15,  # right hip_pitch: backward entry
    12: -1.25,  # right knee
    13: -1.05,  # right ankle
}

BACKROLL_REFERENCE_PHASE_BUCKETS = (
    (90.0, 110.0),
    (130.0, 150.0),
    (170.0, 190.0),
    (210.0, 230.0),
    (250.0, 270.0),
    (280.0, 300.0),
)

BACKROLL_CURRICULUM_STAGES = [
    {
        "params": {
            # A233 proved that synthetic 150--340-degree resets improve their
            # own suffix metrics but do not transfer to standing. The captured
            # successful physical trajectory instead crosses a brief 28.5
            # degree lateral-axis tilt at 180--220 degrees, just outside the
            # final 20-degree gate. Expose all six real trajectory phases under
            # a 29-degree discovery envelope, retain the 30-degree hard side
            # rejection plus continuous straightness pressure, and tighten the
            # envelope only after standing success. A234 additionally moved
            # the actor normalizer by L2 0.63 in 50 updates and erased every
            # standing contact latch, so half of every discovery batch now
            # rehearses standing while the actor normalizer stays frozen.
            "standing_prob": 0.50,
            "midroll_prob": 0.50,
            "midroll_pitch_min": math.radians(90.0),
            "midroll_pitch_max": math.radians(300.0),
            "midroll_omega_range": (1.5, 4.0),
            "joint_noise_std": 0.0,
            "reference_state_prob": 1.0,
            "reference_state_path": BACKROLL_REFERENCE_STATE_PATH,
            "reference_phase_range_deg": (90.0, 300.0),
            "reference_phase_buckets_deg": BACKROLL_REFERENCE_PHASE_BUCKETS,
            "reference_strict_sagittal": True,
            "synthesize_contact_latches": False,
            "sagittal_tolerance_deg": 29.0,
            "reference_source_seed": None,
            "yaw_range": (0.0, 0.0),
        }
    },
    {
        "params": {
            "standing_prob": 0.60,
            "midroll_prob": 0.40,
            "midroll_pitch_min": math.radians(90.0),
            "midroll_pitch_max": math.radians(300.0),
            "midroll_omega_range": (1.0, 4.0),
            "reference_state_prob": 1.0,
            "reference_phase_range_deg": (90.0, 300.0),
            "reference_phase_buckets_deg": BACKROLL_REFERENCE_PHASE_BUCKETS,
            "reference_strict_sagittal": True,
            "synthesize_contact_latches": False,
            "sagittal_tolerance_deg": 27.0,
        }
    },
    {
        "params": {
            "standing_prob": 0.70,
            "midroll_prob": 0.30,
            "midroll_pitch_min": math.radians(90.0),
            "midroll_pitch_max": math.radians(300.0),
            "midroll_omega_range": (2.0, 5.0),
            "reference_state_prob": 1.0,
            "reference_phase_range_deg": (90.0, 300.0),
            "reference_phase_buckets_deg": BACKROLL_REFERENCE_PHASE_BUCKETS,
            "reference_strict_sagittal": True,
            "synthesize_contact_latches": False,
            "sagittal_tolerance_deg": 25.0,
        }
    },
    {
        "params": {
            "standing_prob": 0.80,
            "midroll_prob": 0.20,
            "midroll_pitch_min": math.radians(90.0),
            "midroll_pitch_max": math.radians(300.0),
            "midroll_omega_range": (0.0, 4.0),
            "reference_state_prob": 1.0,
            "reference_phase_range_deg": (90.0, 300.0),
            "reference_phase_buckets_deg": BACKROLL_REFERENCE_PHASE_BUCKETS,
            "reference_strict_sagittal": True,
            "synthesize_contact_latches": False,
            "sagittal_tolerance_deg": 22.0,
        }
    },
    {
        "params": {
            "standing_prob": 0.90,
            "midroll_prob": 0.10,
            "midroll_pitch_min": math.radians(90.0),
            "midroll_pitch_max": math.radians(300.0),
            "midroll_omega_range": (0.0, 3.0),
            "reference_state_prob": 1.0,
            "reference_phase_range_deg": (90.0, 300.0),
            "reference_phase_buckets_deg": BACKROLL_REFERENCE_PHASE_BUCKETS,
            "reference_strict_sagittal": True,
            "synthesize_contact_latches": False,
            "sagittal_tolerance_deg": 20.0,
        }
    },
]

REPEATED_BACKROLL_CURRICULUM_STAGES = [
    {
        "params": {
            # First preserve the measured late-phase basin while giving the
            # policy an equal-sized standing bucket to learn the launch.
            # A197/A198 standing audits stayed at ~1 degree while the parent
            # only knew a late 180-degree bridge.  Put an earlier, grounded
            # 100--180 degree reference bridge beside a standing bucket so
            # PPO can learn launch -> trunk/head contact before the strict
            # landing gate is introduced.
            "standing_prob": 0.25,
            "midroll_prob": 0.75,
            # Bridge from the audited one-shot champion before enabling the
            # stricter repeated-cycle state machine.  The champion can earn
            # one physically valid roll here; standing mastery then unlocks
            # recovery/rearm and repeated-cycle credit in a later stage.
            "repeat_mode": False,
            "relaxed_first_cycle": False,
            "midroll_pitch_min": math.radians(100.0),
            "midroll_pitch_max": math.radians(180.0),
            "midroll_omega_range": (1.0, 3.0),
            "joint_noise_std": 0.0,
            "synthesize_contact_latches": True,
            # Include the measured pre-contact rows; unlike synthetic pitch
            # starts these carry the correct grounded pose and signed velocity.
            "reference_state_prob": 1.0,
            "reference_phase_range_deg": (100.0, 180.0),
            "reference_source_seed": None,
            # Standing audits drifted far off-axis even from this narrow
            # range; keep the first launch basin aligned while the measured
            # 180-degree reference bridge remains unchanged.
            "yaw_range": (0.0, 0.0),
            "recovery_enabled": False,
            "ground_recovery_prob": 0.0,
            "mastery_cycles": 1,
        }
    },
    {
        "params": {
            "standing_prob": 0.50,
            "midroll_prob": 0.50,
            # Keep one-shot physics through the wider 180--340 degree
            # reference bridge.  Switching the strict repeat gates here
            # destroyed the standing basin immediately (A189, iteration 71).
            # The next stage is the first intentional repeated-cycle handoff.
            "repeat_mode": False,
            "relaxed_first_cycle": False,
            "midroll_pitch_min": math.radians(180.0),
            "midroll_pitch_max": math.radians(340.0),
            "midroll_omega_range": (1.0, 4.0),
            "joint_noise_std": 0.01,
            "synthesize_contact_latches": True,
            "reference_state_prob": 1.0,
            "reference_phase_range_deg": (180.0, 180.0),
            "reference_source_seed": 10,
            # Keep the first repeated handoff in the measured aligned launch
            # basin; broad yaw DR is introduced only after the second stage.
            "yaw_range": (0.0, 0.0),
            "recovery_enabled": False,
            "ground_recovery_prob": 0.0,
            "mastery_cycles": 1,
        }
    },
    {
        "params": {
            # First repeat stage: keep the same aligned, measured
            # 180-degree basin while the policy learns the post-landing
            # rearm.  Broad 90-degree starts and stronger angular momentum
            # wait for the following stage; introducing them here caused the
            # repeated handoff to collapse before a cycle could be chained.
            "standing_prob": 0.50,
            "midroll_prob": 0.50,
            "repeat_mode": True,
            # The reference state can aid discovery, but every cycle must use
            # the same strict sagittal maneuver gate.
            "relaxed_first_cycle": False,
            "midroll_pitch_min": math.radians(180.0),
            "midroll_pitch_max": math.radians(340.0),
            "midroll_omega_range": (1.0, 4.0),
            "joint_noise_std": 0.01,
            "synthesize_contact_latches": True,
            "reference_state_prob": 1.0,
            "reference_phase_range_deg": (180.0, 180.0),
            "reference_source_seed": None,
            # Preserve the aligned sagittal basin through the early repeat
            # bridge; later stages reintroduce yaw robustness once chaining is
            # demonstrated.
            "yaw_range": (0.0, 0.0),
            # Enable the retry path at the first repeated handoff so an
            # off-axis/stalled attempt can learn self-right -> reroll rather
            # than terminating before any recovery transition is observed.
            "recovery_enabled": True,
            "ground_recovery_prob": 0.0,
            "mastery_cycles": 1,
        }
    },
    {
        "params": {
            "standing_prob": 0.80,
            "midroll_prob": 0.20,
            "repeat_mode": True,
            "relaxed_first_cycle": False,
            "midroll_pitch_min": math.radians(20.0),
            "midroll_pitch_max": math.radians(340.0),
            "midroll_omega_range": (0.0, 4.0),
            "joint_noise_std": 0.03,
            "synthesize_contact_latches": True,
            "reference_state_prob": 0.10,
            "reference_phase_range_deg": (0.0, 360.0),
            "reference_source_seed": None,
            "yaw_range": (-math.pi, math.pi),
            "recovery_enabled": True,
            "ground_recovery_prob": 0.05,
            "mastery_cycles": 1,
        }
    },
    {
        "params": {
            "standing_prob": 0.90,
            "midroll_prob": 0.10,
            "repeat_mode": True,
            "relaxed_first_cycle": False,
            "midroll_pitch_min": math.radians(20.0),
            "midroll_pitch_max": math.radians(340.0),
            "midroll_omega_range": (0.0, 3.0),
            "joint_noise_std": 0.03,
            "synthesize_contact_latches": True,
            "reference_state_prob": 0.05,
            "reference_phase_range_deg": (0.0, 360.0),
            "reference_source_seed": None,
            "yaw_range": (-math.pi, math.pi),
            "recovery_enabled": True,
            "ground_recovery_prob": 0.10,
            "mastery_cycles": 2,
        }
    },
    {
        "params": {
            "standing_prob": 1.0,
            "midroll_prob": 0.0,
            "repeat_mode": True,
            "relaxed_first_cycle": False,
            "midroll_pitch_min": math.radians(20.0),
            "midroll_pitch_max": math.radians(340.0),
            "midroll_omega_range": (0.0, 3.0),
            "joint_noise_std": 0.03,
            "synthesize_contact_latches": True,
            "reference_state_prob": 0.0,
            "reference_phase_range_deg": (0.0, 360.0),
            "reference_source_seed": None,
            "yaw_range": (-math.pi, math.pi),
            "recovery_enabled": True,
            "ground_recovery_prob": 0.20,
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
        # Partial rotation is only a bootstrap.  A full ordered grounded
        # backroll and stable feet landing are the task objective.
        weight=4.0,
        params={"target_angle": 2.0 * math.pi, "max_paid_rate": 5.0},
    )
    cfg.rewards["backroll_launch_tuck_progress"] = RewardTermCfg(
        func=microduck_mdp.grounded_backroll_launch_tuck_progress,
        # A218's first three audited checkpoints stayed at HOME from standing
        # while late reference states supplied the visible training reward.
        # This bounded potential pays once on the way to the measured tuck,
        # then gives way to the real signed-rotation objective.
        weight=2.0,
        params={
            "target_overrides": BACKROLL_TUCK_OVERRIDES,
            "entry_angle": math.radians(45.0),
            "max_paid_rate": 2.5,
        },
    )
    cfg.rewards["backroll_head_pivot"] = RewardTermCfg(
        func=microduck_mdp.grounded_backroll_head_pivot,
        weight=0.5,
        params={"rate_norm": 2.0},
    )
    cfg.rewards["backroll_completion_progress"] = RewardTermCfg(
        func=microduck_mdp.grounded_backroll_completion_progress,
        weight=12.0,
        params={
            # A223's independent standing audit reached the ordered trunk and
            # flat-head latches in 14/16 trials, but its 120--155 degree
            # frontier received no completion gradient because this gate
            # opened at 150 degrees.  Release the same max-so-far,
            # contact-gated completion frontier immediately after the
            # physically valid head-top window opens; it still cannot pay for
            # rocking, airborne rotation, or an unlatched/side attempt.
            "start_angle": math.radians(110.0),
            "target_angle": math.radians(350.0),
            "max_paid_rate": 6.0,
        },
    )
    cfg.rewards["backroll_speed_progress"] = RewardTermCfg(
        func=microduck_mdp.grounded_backroll_speed_progress,
        # A229 checkpoint 150 acquired the complete ordered contact sequence
        # but settled on the head at 147 degrees.  Pay only *new* supported
        # frontier while its sagittal backward rate is useful; rocking,
        # revisiting an angle, spinning in the air, or leaving the strict gate
        # remains worth zero.
        weight=3.0,
        params={
            "minimum_rate": 2.0,
            "target_rate": 4.5,
            "target_angle": 2.0 * math.pi,
        },
    )
    cfg.rewards["backroll_feet_recontact"] = RewardTermCfg(
        func=microduck_mdp.grounded_backroll_feet_recontact_rate,
        # One bounded bridge into the true landing gate. It requires 300
        # degrees, ordered contacts, head release, both feet, rising posture,
        # and the same strict sagittal history as final success.
        weight=5.0,
    )
    cfg.rewards["backroll_landing_readiness"] = RewardTermCfg(
        func=microduck_mdp.grounded_backroll_landing_readiness_progress,
        # A231 reached both feet from late starts in 44% of training episodes,
        # yet never converted one into a landing hold. Pay only a new
        # max-so-far composite after that real recontact: more backward phase,
        # upright height, and braking. The inverted approach is never charged.
        weight=8.0,
    )
    cfg.rewards["backroll_success"] = RewardTermCfg(
        func=microduck_mdp.grounded_backroll_success_rate,
        weight=20.0,
    )
    cfg.rewards["backroll_invalid"] = RewardTermCfg(
        func=microduck_mdp.grounded_backroll_invalid_rate,
        # A231's weak -1 event let the policy keep all partial progress and
        # terminate through a side exit in 10/16 deterministic standing trials.
        # A one-shot invalid cost is not a motion blocker: valid sagittal
        # exploration pays nothing here, while the rejected shortcut is now
        # worse than its one-shot feet-recontact bridge.
        weight=-5.0,
    )
    cfg.rewards["backroll_overspeed"] = RewardTermCfg(
        func=microduck_mdp.roulade_overspeed_penalty,
        weight=-0.1,
        params={"omega_max": 7.0},
    )
    cfg.rewards["backroll_sagittal"] = RewardTermCfg(
        func=microduck_mdp.grounded_backroll_sagittal_penalty,
        # Positive skill rewards already hard-gate at 20 degrees and a
        # 30-degree escape terminates the attempt.  A229 showed that abruptly
        # raising this rate tax to -2/-3 before any landing was learned caused
        # shorter episodes and 16/16 side exits.  Keep only a small continuous
        # hint during discovery; tighten after a genuine suffix exists.
        weight=-0.10,
    )
    cfg.rewards["backroll_lateral_velocity"] = RewardTermCfg(
        func=microduck_mdp.roulade_lateral_velocity_penalty,
        weight=-0.5,
    )
    cfg.rewards["backroll_flatness"] = RewardTermCfg(
        func=microduck_mdp.grounded_backroll_flatness_penalty,
        weight=-1.0,
    )
    cfg.rewards["action_rate_l2"] = RewardTermCfg(
        func=mdp.action_rate_l2,
        # Dynamic discovery needs the robot to commit to a tuck.  A117's
        # un-gated action-rate cost was several times larger than all positive
        # task shaping and converged to a static crouch.  Reintroduce only a
        # small smoothness tax after the strict grounded stage has earned its
        # own advancement; see the phase curriculum below.
        weight=0.0,
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
    cfg.rewards["backroll_contact_sequence"] = RewardTermCfg(
        func=microduck_mdp.grounded_backroll_contact_sequence,
        weight=0.5,
        params={"trunk_value": 1.0, "head_value": 2.0},
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
            "tuck_overrides": BACKROLL_TUCK_OVERRIDES,
            "tuck_factor_range": (0.5, 1.0),
            "joint_noise_std": 0.0 if play else 0.04,
        },
    )
    if play:
        cfg.events["set_grounded_backroll_state"].params.update(
            standing_prob=1.0,
            midroll_prob=0.0,
            sagittal_tolerance_deg=20.0,
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
                "standing_only_mastery": True,
                "invalid_reward_name": "backroll_invalid",
                "invalid_reward_weights": [-5.0, -6.0, -8.0, -10.0, -10.0],
                "action_rate_reward_name": "action_rate_l2",
                "action_rate_reward_weights": [
                    0.0,
                    -0.025,
                    -0.05,
                    -0.075,
                    -0.10,
                ],
            },
        )
        cfg.curriculum["backroll_sagittal_weight"] = CurriculumTermCfg(
            func=microduck_mdp.reward_weight,
            params={
                "reward_name": "backroll_sagittal",
                "weight_stages": [
                    {"step": 0, "weight": -0.10},
                    {"step": 600 * 24, "weight": -0.25},
                ],
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
    cfg.metrics["backroll_feet_recontact"] = MetricsTermCfg(
        func=microduck_mdp.grounded_backroll_feet_recontact_fraction,
    )
    cfg.metrics["backroll_landing_readiness"] = MetricsTermCfg(
        func=microduck_mdp.grounded_backroll_landing_readiness,
    )
    return cfg


def make_microduck_repeated_backroll_env_cfg(play: bool = False):
    """Create a repeated, fast, strictly sagittal grounded-backroll task."""
    cfg = deepcopy(make_microduck_backroll_env_cfg(play=play))
    cfg.episode_length_s = REPEATED_EPISODE_LENGTH_S

    first_stage = REPEATED_BACKROLL_CURRICULUM_STAGES[0]["params"]
    reset_cfg = cfg.events["set_grounded_backroll_state"]
    reset_cfg.params.update(
        repeat_mode=True,
        yaw_range=(0.0, 0.0) if play else (-math.pi, math.pi),
        joint_noise_std=0.0 if play else 0.03,
        ground_recovery_prob=0.0,
        ground_z_range=(0.04, 0.05),
        reference_state_path=BACKROLL_REFERENCE_STATE_PATH,
    )
    reset_cfg.params.update(**first_stage)
    if play:
        reset_cfg.params.update(
            standing_prob=1.0,
            midroll_prob=0.0,
            repeat_mode=True,
            relaxed_first_cycle=False,
            reference_state_prob=0.0,
            yaw_range=(0.0, 0.0),
            recovery_enabled=True,
        )
    else:
        cfg.curriculum["backroll_phase"].params.update(
            stages=REPEATED_BACKROLL_CURRICULUM_STAGES,
            success_threshold=0.45,
            required_consecutive_windows=2,
            # Mixed reference starts can look successful while standing
            # starts still fail (A180 model 2600-2750).  Do not advance into
            # the broad phase distribution until the standing launch itself
            # has mastered this bridge.
            standing_only_mastery=True,
            speed_reward_name="backroll_speed_progress",
            speed_reward_weights=[1.0, 1.0, 1.5, 2.0, 3.0, 3.0],
            invalid_reward_name="backroll_invalid",
            invalid_reward_weights=[-2.0, -3.0, -4.0, -6.0, -8.0, -10.0],
        )

    for group in ("actor", "critic"):
        command_obs = cfg.observations[group].terms["command"]
        command_obs.func = microduck_mdp.grounded_backroll_recovery_command
        command_obs.params = {"command_name": "twist"}

    # A valid landing rearms the next cycle, so it is a reward pulse rather
    # than a termination. The first invalid attempt spends one self-right and
    # retry budget; a second invalid attempt or recovery timeout terminates.
    cfg.terminations.pop("backroll_success", None)
    # A120 reached the first inverted support quickly and then parked.  Make
    # extension through the already-latched 180--350 degree arc materially
    # more valuable than repeatedly discovering the known 0--180 degree tuck.
    cfg.rewards["backroll_progress"].weight = 5.0
    cfg.rewards["backroll_contact_sequence"].weight = 2.0
    cfg.rewards["backroll_non_top_head_dwell"] = RewardTermCfg(
        func=microduck_mdp.grounded_backroll_non_top_head_dwell_penalty,
        weight=0.5,
        params={"grace_steps": 9},
    )
    cfg.rewards["backroll_head_alignment_progress"] = RewardTermCfg(
        func=microduck_mdp.grounded_backroll_head_alignment_progress,
        weight=1.5,
        params={"max_paid_rate": 1.5},
    )
    cfg.rewards["backroll_completion_progress"].weight = 18.0
    # The repeated policy retains the broader recovery shaping it already
    # uses. The one-shot discovery task deliberately replaces these signed
    # terms with the post-recontact max-so-far landing bridge above.
    cfg.rewards["backroll_upright_progress"] = RewardTermCfg(
        func=microduck_mdp.grounded_backroll_upright_progress,
        weight=5.0,
    )
    cfg.rewards["backroll_height_progress"] = RewardTermCfg(
        func=microduck_mdp.grounded_backroll_height_progress,
        weight=4.0,
    )
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
    cfg.rewards["backroll_recovery_progress"] = RewardTermCfg(
        func=microduck_mdp.grounded_backroll_recovery_progress,
        weight=0.25,
    )
    cfg.rewards["backroll_recovery_success"] = RewardTermCfg(
        func=microduck_mdp.grounded_backroll_recovery_success,
        weight=5.0,
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
    cfg.metrics["backroll_max_non_top_head_dwell_s"] = MetricsTermCfg(
        func=microduck_mdp.grounded_backroll_max_non_top_head_dwell_s,
    )
    cfg.metrics["backroll_recovery_attempt_count"] = MetricsTermCfg(
        func=microduck_mdp.grounded_backroll_recovery_attempt_count,
    )
    cfg.metrics["backroll_recovery_success_count"] = MetricsTermCfg(
        func=microduck_mdp.grounded_backroll_recovery_success_count,
    )
    cfg.metrics["backroll_recovery_success_rate"] = MetricsTermCfg(
        func=microduck_mdp.grounded_backroll_recovery_success_fraction,
    )
    cfg.metrics["backroll_mean_recovery_latency_s"] = MetricsTermCfg(
        func=microduck_mdp.grounded_backroll_mean_recovery_latency_s,
    )
    cfg.metrics["backroll_recovered_reroll_count"] = MetricsTermCfg(
        func=microduck_mdp.grounded_backroll_recovered_reroll_count,
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

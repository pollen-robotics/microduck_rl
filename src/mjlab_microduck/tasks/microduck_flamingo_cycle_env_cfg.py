"""Microduck *flamingo cycle* task — stand → one foot → stand, both sides, one policy.

Stage 2 of docs/superpowers/specs/2026-08-28-flamingo-design.md, built as a
CONTROLLER-DRIVEN posture command rather than a fixed phase clock (the user
wants to choose when to lift and when to come back, like a button):

    twist command slots = [flamingo_flag, side, 0]
        flag ∈ {0, 1}    0 = stand on two feet (HOME), 1 = stand on one foot
        side ∈ {-1, +1}  +1 = RIGHT foot is the stance foot (left lifted), -1 = LEFT

The command term (``FlamingoCommand``) flips the flag at random dwell times and
keeps an internal slewed blend α (0 = STAND target, 1 = FLAMINGO target,
``RAMP_S`` per full transition) that the shaping rewards track — the SitStand
idiom. The OBS is the raw [flag, side, 0], so a runtime just writes the flag.

Sequencing comes from α-gating, not from a script:
  * CoM target slides from the midpoint of the feet to the stance foot (leader)
  * the swing-foot contact flips from rewarded (α < 0.4) to taxed (α > 0.9)
  * the swing-foot clearance is only asked for once α > 0.6
  * pose / trunk-lean targets are linear blends HOME ↔ FLAMINGO(side)
Coming back down is the same thing in reverse. "Pushed too hard on one foot":
the swing-foot touchdown is a cheap tax and never terminal, staying upright is
worth far more, so stepping down and re-lifting is the trained answer; the
controller can also just drop the flag.

Spawn buckets (``set_flamingo_cycle_state``): in the pose of the commanded side
(``IN_POSE_PROB``, ramped 0.6 → 0.3) or standing at HOME; the flag is drawn
independently → all four spawn × command combinations, incl. "lower from a
static hold" and "hold what you are doing".

Everything else (DR, obs noise/delays, pushes, regularisers, PPO) is the
stage-1 env's, imported from microduck_flamingo_env_cfg.
"""

import math
from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    RewardTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_flamingo_env_cfg import (
    FLAMINGO_BASE_PITCH,
    FLAMINGO_BASE_ROLL,
    FLAMINGO_GRAVITY_B,
    FLAMINGO_POSE,
    FLAMINGO_Z,
    NUM_STEPS_PER_ENV,
    STANCE_SIDE_TILT_THRESHOLD,
    MicroduckFlamingoRlCfg,
    make_microduck_flamingo_env_cfg,
)

EPISODE_LENGTH_S = 10.0
# Dwell time in each posture before the flag may flip (must exceed RAMP_S).
POSTURE_DWELL_S = (2.5, 5.0)
RAMP_S = 1.5             # seconds for α to traverse STAND ↔ FLAMINGO in full
FLAMINGO_PROB = 0.6      # probability a resample commands ONE foot
IN_POSE_PROB = 0.6       # spawn bucket: in the one-foot pose (ramped down by curriculum)
ZERO_SIDE_PROB = 0.5     # P(observed side = 0 | stand): trains the runtime's all-zero idle command
# Swing-foot clearance target. The pose has the foot ~0.17 m up; stage 1 asked
# for 0.05 ± 0.03 which contradicts pose_track (exp(−16) at the pose). 0.10 ± 0.06
# pays 0.5 at 5 cm, 1.0 at 10 cm, 0.26 at 17 cm: clearly lifted, no kicking.
SWING_CLEAR_TARGET_Z = 0.10
SWING_CLEAR_STD = 0.06
MAX_PUSH = 0.15          # m/s, final push stage (0.25 broke stage 1, see curricula)
MAX_ACTION_RATE_W = -0.5
HARD_PUSH = 0.25         # hard variant: extra push stage
HARD_PUSH_IT = 1400


def mirror_pose(pose: list) -> list:
    """Left-stance pose from the right-stance one. The MJCF's right-leg joint
    axes are the negatives of the left-leg ones (HOME: left_hip_pitch −0.458,
    right_hip_pitch +0.458), so mirroring = swap legs and negate every leg
    joint; the head chain negates yaw and roll. Verified 2026-08-29 with
    notes/tools/duck_pose.py: identical 1.20 cm CoM margin on both sides."""
    left_leg = [-v for v in pose[9:14]]
    right_leg = [-v for v in pose[0:5]]
    head = [pose[5], pose[6], -pose[7], -pose[8]]
    return [float(v) + 0.0 for v in left_leg + head + right_leg]


FLAMINGO_POSE_RIGHT = list(FLAMINGO_POSE)
FLAMINGO_POSE_LEFT = mirror_pose(FLAMINGO_POSE)
FLAMINGO_GRAVITY_RIGHT = tuple(FLAMINGO_GRAVITY_B)
FLAMINGO_GRAVITY_LEFT = (FLAMINGO_GRAVITY_B[0], -FLAMINGO_GRAVITY_B[1], FLAMINGO_GRAVITY_B[2])


def make_microduck_flamingo_cycle_env_cfg(play: bool = False, hard: bool = False, gentle: bool = False) -> ManagerBasedRlEnvCfg:
    """``hard``: run-2 variant — pushes ramp on to HARD_PUSH m/s at HARD_PUSH_IT
    (run 1 topped out at 0.15 and fell, rather than stepping down, for pushes
    ≥ 0.18 m/s backward / toward the stance side in the CPU stress test).
    ``gentle``: run-3 pacing for a 3000-it run — both cycle runs stepped DOWN in
    reward / episode length at the in-pose 0.4 → 0.3 spawn step (it 1200) and
    at the 0.25 m/s stage (it 1400); spread the spawn ramp over 2100 it, pushes
    0.08 / 0.15 at it 600 / 1200 and (with ``hard``) 0.25 at it 2400."""
    cfg = make_microduck_flamingo_env_cfg(play=play)
    cfg.episode_length_s = EPISODE_LENGTH_S

    feet_sensor = "feet_ground_contact"
    left_site = SceneEntityCfg("robot", site_names=["left_foot"])
    right_site = SceneEntityCfg("robot", site_names=["right_foot"])

    # ── Rewards: drop the stage-1 (right-only, ungated) task terms ───────────
    for name in (
        "com_over_stance_foot", "stance_foot_grounded", "swing_foot_clear",
        "swing_foot_touch", "pose_flamingo", "gravity_flamingo", "stillness",
        "stance_side_tilt",
    ):
        cfg.rewards.pop(name, None)

    cfg.rewards["com_target"] = RewardTermCfg(
        func=microduck_mdp.fl_com_target, weight=3.0,
        params={"command_name": "twist", "left_cfg": left_site, "right_cfg": right_site, "std": 0.03},
    )
    cfg.rewards["stance_foot_grounded"] = RewardTermCfg(
        func=microduck_mdp.fl_stance_foot_grounded, weight=1.0,
        params={"command_name": "twist", "sensor_name": feet_sensor},
    )
    # +1 in stand, −1 on one foot (mixed sign by design; see mdp docstring).
    cfg.rewards["swing_foot_contact"] = RewardTermCfg(
        func=microduck_mdp.fl_swing_foot_contact, weight=0.5,
        params={"command_name": "twist", "sensor_name": feet_sensor, "lo": 0.4, "hi": 0.9},
    )
    cfg.rewards["swing_foot_clear"] = RewardTermCfg(
        func=microduck_mdp.fl_swing_foot_clear, weight=1.5,
        params={"command_name": "twist", "left_cfg": left_site, "right_cfg": right_site,
                "sensor_name": feet_sensor, "target": SWING_CLEAR_TARGET_Z, "std": SWING_CLEAR_STD,
                "lo": 0.6, "hi": 1.0},
    )
    cfg.rewards["pose_track"] = RewardTermCfg(
        func=microduck_mdp.fl_pose_track, weight=1.5,
        params={"command_name": "twist", "pose_right": FLAMINGO_POSE_RIGHT,
                "pose_left": FLAMINGO_POSE_LEFT, "std": 0.5, "mid_attenuation": 0.75},
    )
    cfg.rewards["gravity_track"] = RewardTermCfg(
        func=microduck_mdp.fl_gravity_track, weight=2.0,
        params={"command_name": "twist", "gravity_right": FLAMINGO_GRAVITY_RIGHT,
                "gravity_left": FLAMINGO_GRAVITY_LEFT, "std": 0.15},
    )
    cfg.rewards["stillness"] = RewardTermCfg(
        func=microduck_mdp.fl_stillness, weight=1.0,
        params={"command_name": "twist", "sensor_name": feet_sensor, "std": 2.0},
    )
    # ≤ 0 → POSITIVE weight.
    cfg.rewards["stance_side_tilt"] = RewardTermCfg(
        func=microduck_mdp.fl_stance_side_tilt, weight=4.0,
        params={"command_name": "twist", "threshold": STANCE_SIDE_TILT_THRESHOLD},
    )
    # Success indicator. mjlab skips weight-0 terms entirely (nothing logged),
    # so it carries a token weight: ≤ 0.3 per episode, readable in wandb.
    cfg.rewards["commanded_support"] = RewardTermCfg(
        func=microduck_mdp.fl_single_support_success, weight=1e-3,
        params={"command_name": "twist", "sensor_name": feet_sensor},
    )

    # ── Command: posture flag + side in the twist slots ───────────────────────
    command = cfg.commands["twist"]
    command.resampling_time_range = POSTURE_DWELL_S
    base_kwargs = {k: v for k, v in vars(command).items() if k != "rel_turn_in_place_envs"}
    cfg.commands["twist"] = microduck_mdp.FlamingoCommandCfg(
        **{**base_kwargs, "flamingo_prob": FLAMINGO_PROB, "ramp_s": RAMP_S,
           "zero_side_prob": ZERO_SIDE_PROB}
    )

    # ── Spawn: in-pose (either side) or standing; flag drawn by the command ───
    cfg.events.pop("set_flamingo_state", None)
    cfg.events["set_flamingo_cycle_state"] = EventTermCfg(
        func=microduck_mdp.set_flamingo_cycle_state,
        mode="reset",
        params={
            "pose_right": FLAMINGO_POSE_RIGHT,
            "pose_left": FLAMINGO_POSE_LEFT,
            "base_roll": FLAMINGO_BASE_ROLL,
            "base_pitch": FLAMINGO_BASE_PITCH,
            "z_min": FLAMINGO_Z,
            "z_max": FLAMINGO_Z + 0.01,
            "command_name": "twist",
            "tilt_noise": math.radians(3.0),
            "joint_noise_std": 0.05,
            "in_pose_prob": IN_POSE_PROB,
            "standing_z_min": 0.11,
            "standing_z_max": 0.12,
        },
    )

    # ── Curricula: transitions are harder than the hold → later ramps ─────────
    # Lesson from stage-1 run flamingo-s1-long (2026-08-29): the 0.25 m/s push
    # stage (it 1000) on top of the −1.0 action-rate tax halved the reward and
    # the policy never recovered (jittery, saturated actions). Pushes stop at
    # MAX_PUSH and the smoothness tax at MAX_ACTION_RATE_W here.
    cfg.curriculum["push_magnitude"].params["push_stages"] = [
        {"step": 0,                         "velocity_range": {"x": (0.0, 0.0),    "y": (0.0, 0.0)}},
        {"step": 500 * NUM_STEPS_PER_ENV,   "velocity_range": {"x": (-0.08, 0.08), "y": (-0.08, 0.08)}},
        {"step": 1000 * NUM_STEPS_PER_ENV,  "velocity_range": {"x": (-MAX_PUSH, MAX_PUSH), "y": (-MAX_PUSH, MAX_PUSH)}},
    ] + ([{"step": HARD_PUSH_IT * NUM_STEPS_PER_ENV, "velocity_range": {"x": (-HARD_PUSH, HARD_PUSH), "y": (-HARD_PUSH, HARD_PUSH)}}] if hard else [])
    cfg.curriculum["action_rate_weight"].params["weight_stages"] = [
        {"step": 0,                         "weight": -0.1},
        {"step": 600 * NUM_STEPS_PER_ENV,   "weight": MAX_ACTION_RATE_W},
    ]
    cfg.curriculum["torque_rate_weight"].params["weight_stages"] = [
        {"step": 0,                         "weight": 0.0},
        {"step": 800 * NUM_STEPS_PER_ENV,   "weight": -1e-3},
    ]
    cfg.curriculum["com_std"].params["reward_name"] = "com_target"
    cfg.curriculum["com_std"].params["param_stages"] = [
        {"step": 0,                         "params": {"std": 0.03}},
        {"step": 700 * NUM_STEPS_PER_ENV,   "params": {"std": 0.02}},
    ]
    cfg.curriculum["in_pose_prob"] = CurriculumTermCfg(
        func=microduck_mdp.event_param_curriculum,
        params={
            "event_name": "set_flamingo_cycle_state",
            "param_stages": [
                {"step": 0,                        "params": {"in_pose_prob": 0.6}},
                {"step": 600 * NUM_STEPS_PER_ENV,  "params": {"in_pose_prob": 0.4}},
                {"step": 1200 * NUM_STEPS_PER_ENV, "params": {"in_pose_prob": 0.3}},
            ],
        },
    )
    if gentle:
        cfg.curriculum["in_pose_prob"].params["param_stages"] = [
            {"step": 0,                        "params": {"in_pose_prob": 0.6}},
            {"step": 700 * NUM_STEPS_PER_ENV,  "params": {"in_pose_prob": 0.5}},
            {"step": 1400 * NUM_STEPS_PER_ENV, "params": {"in_pose_prob": 0.4}},
            {"step": 2100 * NUM_STEPS_PER_ENV, "params": {"in_pose_prob": 0.3}},
        ]
        cfg.curriculum["push_magnitude"].params["push_stages"] = [
            {"step": 0,                        "velocity_range": {"x": (0.0, 0.0),    "y": (0.0, 0.0)}},
            {"step": 600 * NUM_STEPS_PER_ENV,  "velocity_range": {"x": (-0.08, 0.08), "y": (-0.08, 0.08)}},
            {"step": 1200 * NUM_STEPS_PER_ENV, "velocity_range": {"x": (-MAX_PUSH, MAX_PUSH), "y": (-MAX_PUSH, MAX_PUSH)}},
        ] + ([{"step": 2400 * NUM_STEPS_PER_ENV, "velocity_range": {"x": (-HARD_PUSH, HARD_PUSH), "y": (-HARD_PUSH, HARD_PUSH)}}] if hard else [])
    return cfg


MicroduckFlamingoCycleRlCfg: RslRlOnPolicyRunnerCfg = deepcopy(MicroduckFlamingoRlCfg)
MicroduckFlamingoCycleRlCfg.experiment_name = "flamingo_cycle"
MicroduckFlamingoCycleRlCfg.run_name = "flamingo_cycle"

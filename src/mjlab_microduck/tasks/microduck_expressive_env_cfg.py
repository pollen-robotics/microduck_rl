"""Microduck EXPRESSIVE body motions (2026-10-28, notes/expressive/BRIEF.md): the body moves, the feet stay planted.

One env factory per motion name (tasks/expressive_motions.py), registered as Mjlab-Expressive-<Name>-MicroDuck.
Built ON the ground-pick env (same 61-D obs, 14 actions, phase command [cos 2πφ, sin 2πφ, 0] in the twist slots,
head/body command slots zero, DR / obs-noise / delay stack), one motion per episode from φ = 0 (standing):
    φ = t / T, T = the motion's period; episode length = T; robotd drives φ per tick exactly like the ground pick.

Rewards (functions of φ): track the reference (trunk roll / pitch / yaw-from-start / height offset, 4 head joints),
feet planted (horizontal drift from the start position, lift above the start height, slip speed, plus the ground
pick's feet_grounded / feet_flat), back to the standing leg pose at the end, the usual regularisers, targets clamped to
the joint ranges ±0.1 with an overrun tax (grasp lesson), joint overspeed, fall termination.

FRICTION REALISM (Rémi's hypothesis: the real feet are very grippy, the sim let emotions glide): foot friction
U(1.0, 2.0) instead of (0.7, 1.3), and the grasp env's sticky contacts (elliptic cone, impratio 10), so the motion
has to come from the joints, not from sliding the feet.
"""

import math
from copy import deepcopy
from functools import lru_cache

import mujoco
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import CurriculumTermCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg
from mjlab.tasks.velocity import mdp

from mjlab_microduck.robot.microduck_constants import get_ground_pick_spec
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.expressive_motions import get_motion, osc_channels, peak_rate
from mjlab_microduck.tasks.microduck_ground_pick_env_cfg import make_microduck_ground_pick_env_cfg
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg

SERVO_JOINTS = ("left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
                "neck_pitch", "head_pitch", "head_yaw", "head_roll",
                "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle")

FOOT_FRICTION = (1.0, 2.0)
RESET_JOINT_NOISE = 0.08          # robotd starts from the stand net's settled stance (up to ~0.14 rad off HOME)
TARGET_CLIP_MARGIN = 0.10
ACTION_OVERRUN_WEIGHT = -10.0
FALL_ANGLE_DEG = 60.0

# Tracking (positive). std ≈ the error still worth paying for.
W_TRACK_ROLL, STD_ROLL = 2.5, 0.04
W_TRACK_PITCH, STD_PITCH = 2.5, 0.05
W_TRACK_YAW, STD_YAW = 3.0, 0.08
W_TRACK_HEIGHT, STD_HEIGHT = 2.0, 0.004
W_TRACK_HEAD, STD_HEAD = 3.0, 0.12
W_SETTLE_LEGS = 4.0
# Feet planted (costs, negative weights).
W_FEET_DRIFT = -40.0      # per m of summed horizontal drift (1 cm on both feet: -0.8 / step)
W_FEET_LIFT = -100.0      # per m above the start height + 3 mm
W_FOOT_SLIP = -2.0        # per m/s of summed foot speed

# Round 2 (motion key "<Name>-R2", trained from scratch). See LEDGER.md "Round 2".
W_OSC_BOOST = 1.6         # position-tracking weight multiplier on the oscillating trunk channels
W_TRACK_RATE = 3.0        # velocity tracking of each oscillating trunk channel (rate vs d/dt of the reference)
RATE_STD_FRAC = 0.35      # rate kernel std = this x the channel's peak reference rate (floored below)
RATE_STD_MIN = {"roll": 0.15, "pitch": 0.15, "yaw": 0.15, "dz": 0.01}
W_TRACK_HEAD_WIDE, STD_HEAD_WIDE = 1.5, 0.4   # gradient at a 1.5 rad head error (r1's 0.12 kernel had none)
W_NECK_DEV, NECK_DEV_FREE = -5.0, 0.3         # per rad of |neck - reference| beyond 0.3 rad
W_JOINT_LIMIT, JOINT_LIMIT_MARGIN = 2.0, 0.05 # self-negating: -(intrusion into the last 0.05 rad / 0.05)^2 per joint
W_SETTLE_STILL = -0.05    # per (rad/s)^2 of summed servo speed after settle_from (still robot at the hand-back)
RATE_NAMES = {"roll": "track_roll", "pitch": "track_pitch", "yaw": "track_yaw", "dz": "track_height"}


@lru_cache(maxsize=1)
def expressive_joint_limits() -> dict:
    """{servo joint: (lo, hi)} on the ground-pick model, in the 14-servo order."""
    m = get_ground_pick_spec().compile()
    return {jn: tuple(float(v) for v in m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)])
            for jn in SERVO_JOINTS}


def make_microduck_expressive_env_cfg(motion: str, play: bool = False) -> ManagerBasedRlEnvCfg:
    mo = get_motion(motion)
    cfg = make_microduck_ground_pick_env_cfg(play=play, rough=False)

    # ── One motion per episode, from φ = 0 (standing) ──────────────────────────
    cfg.episode_length_s = mo.period
    cfg.commands["twist"].randomize_phase = False
    cfg.commands["twist"].period = mo.period

    # ── Friction realism: grippy feet + sticky contacts ────────────────────────
    cfg.sim.mujoco.cone = "elliptic"
    cfg.sim.mujoco.impratio = 10.0
    cfg.events["foot_friction"].params["ranges"] = FOOT_FRICTION

    # ── Drop the ground-pick task terms ────────────────────────────────────────
    for name in ("mouth_ground_proximity", "mouth_perpendicular_to_ground", "ground_pick_return_pose_legs",
                 "ground_pick_return_pose_neck", "return_upright", "neck_vel_descent", "mouth_payload_force",
                 "upright"):
        cfg.rewards.pop(name, None)
    cfg.events.pop("sample_mouth_payload", None)
    cfg.events["reset_robot_joints"].params["position_range"] = (-RESET_JOINT_NOISE, RESET_JOINT_NOISE)

    # ── Reference tracking ─────────────────────────────────────────────────────
    def track(chs, w, std):
        return RewardTermCfg(func=microduck_mdp.ex_track, weight=w,
                             params={"motion": motion, "channels": chs, "std": std, "command_name": "twist"})

    cfg.rewards["track_roll"] = track(("roll",), W_TRACK_ROLL, STD_ROLL)
    cfg.rewards["track_pitch"] = track(("pitch",), W_TRACK_PITCH, STD_PITCH)
    cfg.rewards["track_yaw"] = track(("yaw",), W_TRACK_YAW, STD_YAW)
    cfg.rewards["track_height"] = track(("dz",), W_TRACK_HEIGHT, STD_HEIGHT)
    cfg.rewards["track_head"] = track(("neck_pitch", "head_pitch", "head_yaw", "head_roll"), W_TRACK_HEAD, STD_HEAD)
    cfg.rewards["settle_legs"] = RewardTermCfg(
        func=microduck_mdp.ex_settle_legs, weight=W_SETTLE_LEGS,
        params={"settle_from": mo.settle_from, "command_name": "twist"},
    )

    # ── Feet planted (feet_grounded / feet_flat come from the ground pick) ─────
    cfg.rewards["feet_drift"] = RewardTermCfg(func=microduck_mdp.ex_feet_drift, weight=W_FEET_DRIFT)
    cfg.rewards["feet_lift"] = RewardTermCfg(func=microduck_mdp.ex_feet_lift, weight=W_FEET_LIFT)
    cfg.rewards["foot_slip"] = RewardTermCfg(func=microduck_mdp.ex_foot_slip, weight=W_FOOT_SLIP)

    # ── Regularisers: motion-blockers low (the trunk is MEANT to rotate) ───────
    cfg.rewards["body_ang_vel"].weight = -0.02
    cfg.rewards["angular_momentum"].weight = -0.01
    cfg.rewards["joint_overspeed"] = RewardTermCfg(
        func=microduck_mdp.joint_overspeed_penalty, weight=0.15, params={"threshold": 5.0},
    )
    jp = cfg.actions["joint_pos"]
    assert isinstance(jp, JointPositionActionCfg)
    limits = expressive_joint_limits()
    jp.clip = {jn: (lo - TARGET_CLIP_MARGIN, hi + TARGET_CLIP_MARGIN) for jn, (lo, hi) in limits.items()}
    cfg.rewards["action_overrun"] = RewardTermCfg(
        func=microduck_mdp.action_target_overrun_l1, weight=ACTION_OVERRUN_WEIGHT,
        params={"limits": [list(v) for v in limits.values()]},
    )
    cfg.curriculum["action_rate_weight"].params["weight_stages"] = [
        {"step": 0, "weight": -0.5},
        {"step": 500 * 24, "weight": -1.0},
        {"step": 1000 * 24, "weight": -1.5},
    ]

    # ── Critic: the reference, the errors, the feet offsets (actor stays 61-D) ─
    critic = cfg.observations["critic"].terms
    critic["ex_reference"] = ObservationTermCfg(func=microduck_mdp.ex_reference_obs, params={"motion": motion})
    critic["ex_error"] = ObservationTermCfg(func=microduck_mdp.ex_error_obs, params={"motion": motion})
    critic["ex_feet_offset"] = ObservationTermCfg(func=microduck_mdp.ex_feet_offset_obs)

    if motion.endswith("-R2"):
        _add_round2_terms(cfg, motion, mo)

    # ── A fall ends the episode ────────────────────────────────────────────────
    cfg.terminations["fell_over"] = TerminationTermCfg(
        func=mdp.bad_orientation, params={"limit_angle": math.radians(FALL_ANGLE_DEG)}, time_out=False,
    )
    return cfg


def _add_round2_terms(cfg: ManagerBasedRlEnvCfg, motion: str, mo) -> None:
    """Round 2: oscillations tracked in velocity too (and weighted up), a wide head kernel + neck-dive cost, joints
    kept off their stops, a still robot at the hand-back."""
    for ch in osc_channels(mo):
        cfg.rewards[RATE_NAMES[ch]].weight *= W_OSC_BOOST
        std = max(RATE_STD_FRAC * peak_rate(mo, ch), RATE_STD_MIN[ch])
        cfg.rewards[f"{RATE_NAMES[ch]}_rate"] = RewardTermCfg(
            func=microduck_mdp.ex_track_rate, weight=W_TRACK_RATE,
            params={"motion": motion, "channel": ch, "std": std, "command_name": "twist"},
        )
    cfg.rewards["track_head_wide"] = RewardTermCfg(
        func=microduck_mdp.ex_track, weight=W_TRACK_HEAD_WIDE,
        params={"motion": motion, "channels": ("neck_pitch", "head_pitch", "head_yaw", "head_roll"),
                "std": STD_HEAD_WIDE, "command_name": "twist"},
    )
    cfg.rewards["neck_dev"] = RewardTermCfg(
        func=microduck_mdp.ex_head_dev_cost, weight=W_NECK_DEV,
        params={"motion": motion, "channels": ("neck_pitch",), "free": NECK_DEV_FREE, "command_name": "twist"},
    )
    cfg.rewards["joint_limit_proximity"] = RewardTermCfg(
        func=microduck_mdp.joint_limit_proximity_penalty, weight=W_JOINT_LIMIT,
        params={"margin": JOINT_LIMIT_MARGIN, "asset_cfg": SceneEntityCfg("robot", joint_names=SERVO_JOINTS)},
    )
    cfg.rewards["settle_still"] = RewardTermCfg(
        func=microduck_mdp.ex_settle_still, weight=W_SETTLE_STILL,
        params={"settle_from": mo.settle_from, "command_name": "twist"},
    )


def make_expressive_rl_cfg(motion: str) -> RslRlOnPolicyRunnerCfg:
    name = f"expressive_{motion.lower().replace('-', '_')}"
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True,
            distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
        ),
        critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True),
        algorithm=PpoWithSymmetryCfg(
            value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.01,
            num_learning_epochs=5, num_mini_batches=4, learning_rate=1.0e-3, schedule="adaptive",
            gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0, symmetry_cfg=None,
        ),
        wandb_project="mjlab_microduck",
        experiment_name=name,
        run_name=name,
        save_interval=250,
        num_steps_per_env=24,
        max_iterations=3000,
    )

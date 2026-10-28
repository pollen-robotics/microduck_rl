"""Precision nudge: one small, exact move along ONE axis, then stand still. Design and lessons:
docs/superpowers/specs/2026-10-27-precision-nudge-design.md.

Why (2026-10-27, Rémi): placing the duck before a grasp is "very random" with the walking policy. Measured on
velstand v8 in its own training env (scripts/precision/pm_probe.py, 32 tries per row, all DR + obs noise): a stick
push under ~0.2 m/s or 0.3 rad/s for half a second does NOTHING (dead zone: uniform command sampling almost never
trains tiny pure commands); a full tap (0.4 m/s, 0.3 s) gives 59 ± 21 mm forward with ±12° of random yaw and ±9 mm
sideways; closed loop on contact odometry asking 20 mm gives 55 ± 23 mm (it always takes >= 2 steps and keeps walking
~0.6 s after the stop).

Contract = robotd's one-shot skill (SkillDef): twist = the requested move / NUDGE_UNIT (x, y: metres / 0.05, so
1.5 cm = 0.3; yaw: radians / 0.25, so 7.5° = 0.52: where the walking policy already steps) for `go_s` (the skill's
`duration`), then zeros (`unwind` = [0, 0, 0] for `unwind_s`), then the walking policy at zero command. Head and body
slots stay zero. Deployment needs no daemon code: six (or twelve) skill entries with constant commands, e.g.
nudge_fwd = [0.3, 0, 0], duration 0.5, unwind_s 1.3.

Run 1 (nudge-r1-1027, 2026-10-27) learned to stand still: the task terms were ~10 % of the reward mass (action rate
alone twice them) and the requests sat in velstand's dead zone. Run 2: task terms x10, an L1 miss term, a stepping
prior during the move, action rate eased in (-0.2 -> -1.0), larger command units. It stepped and landed within a few
mm in its own env @500, but ended 2 cm crouched (neck -1.7 rad), and velstand then wandered 1-3 cm and 10-28 deg after
the hand-back; and robotd's action layer scattered it. Run 3 (Mjlab-NudgeRuntime): an end-at-HOME term and robotd's
action layer eased in from iteration 400.

Episode: PRE_S of zero command (the duck stands as the walking policy left it), the go window, then SETTLE_S of zeros.
The move is measured on the stance frame (feet midpoint, trunk heading) from the go onset.

Built on the velocity recipe (DR, obs noise, delays, backlash variant) with velstand's all-collisions model; the
actor starts from velstand v8 (NudgeFromVelstandRunner), the critic is fresh and sees the remaining error.
"""
import math
from copy import deepcopy
from dataclasses import fields

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers import CurriculumTermCfg, ObservationTermCfg, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp
from mjlab_microduck.robot.microduck_constants import MICRODUCK_ALLCOLLISIONS_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import MicroduckRlCfg, make_microduck_velocity_env_cfg
from mjlab_microduck.tasks.microduck_velstand_env_cfg import NUM_STEPS_PER_ENV, _collapse_curricula_to_final

PRE_S = 0.3          # standing before the go window (the start state the walking policy hands over)
GO_S = 0.5           # robotd skill `duration`: the twist carries the request this long
SETTLE_S = 1.3       # robotd skill `unwind_s`: zeros, finish the step and stand still
EPISODE_S = PRE_S + GO_S + SETTLE_S

AXIS_PROBS = (0.35, 0.30, 0.35)       # x, y, yaw
DIST_RANGE_M = (0.005, 0.03)          # x, y: 0.5-3 cm
YAW_RANGE_RAD = (math.radians(2.5), math.radians(15.0))
ZERO_PROB = 0.1                       # stand still the whole episode

# Rewards (per step, 50 Hz; times are from the go onset).
# Reward mass (per episode, x dt 0.02): a perfect nudge earns ~13.5 from each hold term; standing still on a 1.5 cm
# request earns ~1.4 broad, ~0 fine and pays ~2.7 of miss; upright / head tracking ~4 each either way.
PROGRESS_WEIGHT = 40.0                # potential: sums to 40 x (initial error - final error), error unit = 1 cm / 5°
HOLD_AFTER_S = 0.9                    # the move should be over by then (the go window is 0.5 s)
HOLD_WEIGHT, HOLD_STD = 15.0, 1.0     # broad: gradient from ~2 cm / 10° away
HOLD_FINE_WEIGHT, HOLD_FINE_STD = 15.0, 0.3  # sharp: 3 mm / 1.5°
MISS_WEIGHT = -2.0                    # L1 pull in the hold window, flat where the Gaussians are
LIFT_WEIGHT, LIFT_UNTIL_S = 1.0, 0.9  # stepping prior during the move (single support), move episodes only
POSE_WEIGHT = 0.5                     # velocity recipe 1.0: its tight standing std fights the finishing step
END_POSE_AFTER_S, END_POSE_STD, END_POSE_WEIGHT = 1.1, 0.1, 15.0   # end at HOME: the walking policy takes over there
ACTION_RATE_STAGES = ((0, -0.2), (1500, -0.5), (2500, -1.0))   # (iteration, weight): smoothness after discovery
STILL_AFTER_S = 1.2
STILL_WEIGHT = -0.5
FEET_DOWN_WEIGHT = 0.5
SLIP_WEIGHT = -5.0                    # planted feet must not slide (always on, not only while "walking")
TERMINATION_WEIGHT = -10.0
STAND_Z = (0.116, 0.119)              # spawn just above the measured standing trunk height (0.1158)

# Runtime layer (run 3): robotd's walk-mode action path, eased in after the stepping is discovered (run 2 found it
# within 250 iterations). In sim, run 2's policy @250 landed a 1.5 cm request at 10.4 +- 3.3 mm with the plain action
# and scattered up to +-8 cm sideways through robotd's 0.9 scale + leg/head low-pass (pm_skill_eval.py).
RUNTIME_LAYER_STAGES = (
    (0, 1.0, 1.0, 1.0, 1.0),          # (iteration, leg alpha, head alpha, scale lo, scale hi): the plain action
    (400, 0.85, 0.75, 0.9, 1.0),
    (700, 0.7, 0.5, 0.8, 1.0),        # robotd: legs_lowpass 0.7, head_lowpass 0.5, 0.9 x battery adaptation
)
RUNTIME_ACTION_RATE_STAGES = ((0, -0.2), (1200, -0.5), (2000, -1.0))

# Hand-overs (run 4): robotd runs the walking policy before and after a skill. Run 3 @250 moved in its own env (0.3 s
# after the spawn drop) but not from a settled stand, and run 2's crouched ending made velstand wander after the
# hand-back. So the runtime variant puts velstand v8 itself in the loop (RuntimeLayerJointPositionAction
# handover_policy): it stands for HANDOVER_PRE_S, the trained policy runs the skill, velstand takes over for
# HANDOVER_POST_S, and the hold terms keep scoring through that last part (its drift counts).
HANDOVER_POLICY = ("pollen-robotics/microduck-policies", "velstand.onnx", "v8")
HANDOVER_PRE_S = 0.8
HANDOVER_POST_S = 0.5
# Run 4 (velstand in the loop) lost its end pose and turned jerky from iteration ~150: the action std grew 0.25 ->
# 0.8 rad because the discarded outputs outside the skill cost nothing under the entropy bonus. Price them, and halve
# the bonus.
HANDOVER_IMITATION_WEIGHT = -5.0
# Safe variant (fine-tune of run 5): run 5 @1000 fell in 1-4 of 48 tries per x/y nudge in the robotd-style eval
# (spins in place then tips backward in the settle part), ~4 % of training episodes. Falls cost far more, and an
# episode ends at a smaller tilt.
SAFE_TERMINATION_WEIGHT = -100.0
SAFE_FALL_ANGLE = math.radians(40.0)
SAFE_OVERRUN_WEIGHT = -10.0           # per rad of commanded target past a joint range (beyond the margins)
# Run 6 @1750 still commanded head roll 0.46 and right hip roll 0.55 rad past range (brief peaks the tax hardly
# prices). The safe variant clips the raw action (training = the exported ONNX's clip): range + 0.05 rad, hip roll
# + 0.35 (its model range is narrower than the real joint; shipped velstand goes 0.43 past it).
SAFE_CLAMP_MARGIN = 0.05
# Run 6: the end-pose term fell 2.7 -> 1.7 once the action rate reached -1.0 (iteration 2000); keep -0.5.
# Run 11 (resumed from run 10 @~3350, so the last stage applies at once): the step part jitters 96-104 mrad of target
# change per tick, twice velstand's walking (50); -0.5 -> -2.0 over the whole skill.
SAFE_ACTION_RATE_STAGES = ((0, -0.2), (1200, -0.5), (3300, -2.0))
# Run 12 (fresh from velstand, Mjlab-NudgeSmooth): fine-tunes could not undo the jitter (runs 9-11: step 95-98 mrad of
# target change per tick, settle 19-23, against velstand's 50 walking / 6.6 standing). Smoothness from iteration 0, and a
# moderate pull to velstand's stance in the late settle (run 10's -300 made falls worse).
SMOOTH_ACTION_RATE_STAGES = ((0, -1.0),)
SMOOTH_SETTLE_IMITATION_WEIGHT = -50.0
# Run 8 (fine-tune of run 7 @2000, the file on cream-converse): the real duck fell once in 11 nudges, the tilt growing
# 7 -> 22 deg over 0.6 s of the settle; sim falls ~2 % (6 / 384). Price any lean beyond 8 deg.
SAFE_TILT_WEIGHT = -20.0
SAFE_TILT_FREE_DEG = 8.0
# Run 9: the late settle copies the walking policy's calm stance (run 7: 3-4x velstand's standing jitter there; the
# real duck's legs shook after the step).
# Run 10: run 9's -10 was ~100x too weak (worth ~0.1 per episode against ~10 for the hold terms; settle jitter 21.7 ->
# 20.9 mrad/tick). -300, plus an action-rate term of its own on the settle.
SAFE_SETTLE_IMITATION_WEIGHT = -300.0
SAFE_SETTLE_ACTION_RATE_WEIGHT = -30.0
SAFE_SETTLE_ACTION_RATE_AFTER_S = 0.6
SAFE_SETTLE_IMITATION_AFTER_S = 1.2
RUNTIME_ENTROPY_COEF = 0.005

DROP_REWARDS = ("track_linear_velocity", "track_angular_velocity", "air_time", "foot_clearance",
                "foot_swing_height", "foot_slip", "head_pose_bias", "body_pose_tracking")
DROP_CURRICULA = ("standing_envs", "head_pose_bias_weight", "body_pose_range", "head_pose_range")


def make_microduck_nudge_env_cfg(play: bool = False, rough: bool = False, runtime_layer: bool = False,
                                 safe: bool = False, smooth: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_microduck_velocity_env_cfg(play=play, rough=False)
    pre_s = HANDOVER_PRE_S if runtime_layer else PRE_S
    post_s = HANDOVER_POST_S if runtime_layer else 0.0

    # velstand's model and contact budget (the actor comes from velstand)
    cfg.scene.entities = {"robot": MICRODUCK_ALLCOLLISIONS_ROBOT_CFG}
    cfg.sim.nconmax = max(cfg.sim.nconmax or 0, 200)

    # inherited curricula at their final stage (the actor was trained under them); drop the command ones
    for k in DROP_CURRICULA:
        cfg.curriculum.pop(k, None)
    _collapse_curricula_to_final(cfg)

    # action rate: eased in after the stepping is discovered (the collapse above pinned the walking recipe's final -1.0)
    cfg.rewards["action_rate_l2"].weight = ACTION_RATE_STAGES[0][1]
    if "action_rate_weight" in cfg.curriculum:
        cfg.curriculum["action_rate_weight"].params["weight_stages"] = [
            {"step": it * NUM_STEPS_PER_ENV, "weight": w} for it, w in ACTION_RATE_STAGES]

    if runtime_layer:
        old_act = cfg.actions["joint_pos"]
        keep = {f.name for f in fields(microduck_mdp.RuntimeLayerJointPositionActionCfg)}
        _, leg, head, lo, hi = RUNTIME_LAYER_STAGES[-1] if play else RUNTIME_LAYER_STAGES[0]
        cfg.actions["joint_pos"] = microduck_mdp.RuntimeLayerJointPositionActionCfg(
            **{k: v for k, v in vars(old_act).items() if k in keep}, leg_alpha=leg, head_alpha=head, scale_lo=lo, scale_hi=hi,
            handover_policy=HANDOVER_POLICY, command_name="twist")
        cfg.rewards["handover_imitation"] = RewardTermCfg(
            func=microduck_mdp.handover_imitation, weight=HANDOVER_IMITATION_WEIGHT)
        if not play:
            cfg.curriculum["runtime_layer"] = CurriculumTermCfg(
                func=microduck_mdp.runtime_layer_curriculum,
                params={"stages": [{"step": it * NUM_STEPS_PER_ENV, "leg_alpha": la, "head_alpha": ha,
                                    "scale_lo": sl, "scale_hi": sh} for it, la, ha, sl, sh in RUNTIME_LAYER_STAGES]})
        cfg.rewards["action_rate_l2"].weight = RUNTIME_ACTION_RATE_STAGES[0][1]
        if "action_rate_weight" in cfg.curriculum:
            cfg.curriculum["action_rate_weight"].params["weight_stages"] = [
                {"step": it * NUM_STEPS_PER_ENV, "weight": w} for it, w in RUNTIME_ACTION_RATE_STAGES]

    cfg.episode_length_s = pre_s + GO_S + SETTLE_S + post_s
    cfg.events.pop("push_robot", None)
    rb = cfg.events["reset_base"].params
    rb["pose_range"]["z"] = STAND_Z
    rb["velocity_range"] = {}

    # ── Command: the nudge in the twist slots; head and body zero (robotd feeds skills zeros there) ──
    old = cfg.commands["twist"]
    keep = {f.name for f in fields(microduck_mdp.NudgeCommandCfg)}
    cfg.commands["twist"] = microduck_mdp.NudgeCommandCfg(
        **{**{k: v for k, v in vars(old).items() if k in keep}, "class_type": microduck_mdp.NudgeCommand,
           "rel_standing_envs": 0.0, "rel_heading_envs": 0.0, "resampling_time_range": (1.0e6, 1.0e6)},
        pre_s=pre_s, go_s=GO_S, unwind_s=SETTLE_S, axis_probs=AXIS_PROBS, dist_range_m=DIST_RANGE_M,
        yaw_range_rad=YAW_RANGE_RAD,
        zero_prob=ZERO_PROB,
    )
    head = cfg.commands["head_pose"]
    head.ranges = tuple((-0.005, 0.005) for _ in head.ranges)      # alive, ~zero
    body = cfg.commands["body_pose"]
    body.ranges = tuple((-0.001, 0.001) for _ in body.ranges)

    # ── Critic: the remaining error and the time (privileged) ──
    cfg.observations["critic"].terms["nudge_state"] = ObservationTermCfg(
        func=microduck_mdp.nudge_critic_obs, params={"command_name": "twist"},
    )

    # ── Rewards ──
    for k in DROP_REWARDS:
        cfg.rewards.pop(k, None)
    cfg.rewards["nudge_progress"] = RewardTermCfg(
        func=microduck_mdp.nudge_progress, weight=PROGRESS_WEIGHT, params={"command_name": "twist"})
    cfg.rewards["nudge_hold"] = RewardTermCfg(
        func=microduck_mdp.nudge_hold, weight=HOLD_WEIGHT,
        params={"command_name": "twist", "std": HOLD_STD, "after_s": HOLD_AFTER_S})
    cfg.rewards["nudge_hold_fine"] = RewardTermCfg(
        func=microduck_mdp.nudge_hold, weight=HOLD_FINE_WEIGHT,
        params={"command_name": "twist", "std": HOLD_FINE_STD, "after_s": HOLD_AFTER_S})
    cfg.rewards["nudge_still"] = RewardTermCfg(
        func=microduck_mdp.nudge_still, weight=STILL_WEIGHT,
        params={"command_name": "twist", "after_s": STILL_AFTER_S, "until_s": GO_S + SETTLE_S})
    cfg.rewards["nudge_feet_down"] = RewardTermCfg(
        func=microduck_mdp.nudge_feet_down, weight=FEET_DOWN_WEIGHT,
        params={"sensor_name": "feet_ground_contact", "command_name": "twist", "after_s": STILL_AFTER_S,
                "until_s": GO_S + SETTLE_S})
    cfg.rewards["nudge_slip"] = RewardTermCfg(
        func=mdp.feet_slip, weight=SLIP_WEIGHT,
        params={"sensor_name": "feet_ground_contact", "command_name": "twist", "command_threshold": -1.0,
                "asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))})
    cfg.rewards["nudge_miss"] = RewardTermCfg(
        func=microduck_mdp.nudge_miss, weight=MISS_WEIGHT, params={"command_name": "twist", "after_s": HOLD_AFTER_S})
    cfg.rewards["nudge_lift"] = RewardTermCfg(
        func=microduck_mdp.nudge_lift, weight=LIFT_WEIGHT,
        params={"sensor_name": "feet_ground_contact", "command_name": "twist", "until_s": LIFT_UNTIL_S})
    cfg.rewards["pose"].weight = POSE_WEIGHT
    cfg.rewards["nudge_end_pose"] = RewardTermCfg(
        func=microduck_mdp.nudge_end_pose, weight=END_POSE_WEIGHT,
        params={"command_name": "twist", "after_s": END_POSE_AFTER_S, "std": END_POSE_STD, "until_s": GO_S + SETTLE_S})
    cfg.rewards["termination"] = RewardTermCfg(func=envs_mdp.is_terminated, weight=TERMINATION_WEIGHT)
    cfg.rewards["nudge_metrics"] = RewardTermCfg(
        func=microduck_mdp.nudge_metrics, weight=0.0, params={"command_name": "twist"})
    # weight-0 terms are not logged at all: give the logger a vanishing weight
    cfg.rewards["nudge_metrics"].weight = 1.0e-9
    if safe:
        cfg.rewards["termination"].weight = SAFE_TERMINATION_WEIGHT
        if "fell_over" in cfg.terminations:
            cfg.terminations["fell_over"].params["limit_angle"] = SAFE_FALL_ANGLE
        cfg.rewards["nudge_target_overrun"] = RewardTermCfg(func=microduck_mdp.nudge_target_overrun, weight=SAFE_OVERRUN_WEIGHT)
        cfg.rewards["nudge_tilt"] = RewardTermCfg(
            func=microduck_mdp.nudge_tilt, weight=SAFE_TILT_WEIGHT, params={"free_deg": SAFE_TILT_FREE_DEG})
        cfg.rewards["nudge_settle_imitation"] = RewardTermCfg(
            func=microduck_mdp.nudge_settle_imitation, weight=SAFE_SETTLE_IMITATION_WEIGHT,
            params={"command_name": "twist", "after_s": SAFE_SETTLE_IMITATION_AFTER_S, "until_s": GO_S + SETTLE_S})
        cfg.rewards["nudge_settle_action_rate"] = RewardTermCfg(
            func=microduck_mdp.nudge_settle_action_rate, weight=SAFE_SETTLE_ACTION_RATE_WEIGHT,
            params={"command_name": "twist", "after_s": SAFE_SETTLE_ACTION_RATE_AFTER_S, "until_s": GO_S + SETTLE_S})
        if isinstance(cfg.actions["joint_pos"], microduck_mdp.RuntimeLayerJointPositionActionCfg):
            cfg.actions["joint_pos"].clamp_margin = SAFE_CLAMP_MARGIN
        if "action_rate_weight" in cfg.curriculum:
            cfg.curriculum["action_rate_weight"].params["weight_stages"] = [
                {"step": it * NUM_STEPS_PER_ENV, "weight": w} for it, w in SAFE_ACTION_RATE_STAGES]
    if smooth:
        cfg.rewards["action_rate_l2"].weight = SMOOTH_ACTION_RATE_STAGES[0][1]
        if "action_rate_weight" in cfg.curriculum:
            cfg.curriculum["action_rate_weight"].params["weight_stages"] = [
                {"step": it * NUM_STEPS_PER_ENV, "weight": w} for it, w in SMOOTH_ACTION_RATE_STAGES]
        if "nudge_settle_imitation" in cfg.rewards:
            cfg.rewards["nudge_settle_imitation"].weight = SMOOTH_SETTLE_IMITATION_WEIGHT
    return cfg


MicroduckNudgeRlCfg = deepcopy(MicroduckRlCfg)
MicroduckNudgeRlCfg.experiment_name = "nudge"
MicroduckNudgeRlCfg.run_name = "nudge"
MicroduckNudgeRlCfg.max_iterations = 3000

MicroduckNudgeRuntimeRlCfg = deepcopy(MicroduckNudgeRlCfg)
MicroduckNudgeRuntimeRlCfg.algorithm.entropy_coef = RUNTIME_ENTROPY_COEF

# Safe fine-tune (run 6, from run 5 @1500): run 5 degraded from ~1700 the way run 4 did, only slower (action std
# 0.26 -> 0.34, end pose 2.5 -> 0.09): no entropy bonus at all for the fine-tune.
MicroduckNudgeRuntimeSafeRlCfg = deepcopy(MicroduckNudgeRuntimeRlCfg)
MicroduckNudgeRuntimeSafeRlCfg.algorithm.entropy_coef = 0.0

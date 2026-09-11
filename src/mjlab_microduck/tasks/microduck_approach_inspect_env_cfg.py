"""Microduck approach-and-inspect task.

Walks to a target object, stops at a standoff ring, aims the head at it, holds
the look, then turns away. Obs (61-D) / action (14) spaces are identical to the
walking policy so the two hot-swap at runtime.

Built ON TOP of ``make_microduck_velocity_env_cfg`` (the AGENTS.md workflow):
the returned cfg IS the velocity recipe — every DR / sim2real block (CoM,
head-CoM, mass/inertia, BAM friction, armature, pushes, IMU misalignment,
encoder bias, obs noise/delays, NaN guards, foot friction) is the velocity
env's own object, untouched. This file only swaps the COMMAND, the TASK
rewards, the regulariser weights and the curricula that drove them.
``tests/test_approach_inspect_cfg.py`` locks the DR parity term by term.

Command encoding (3-D twist slot, see approach_inspect_mdp.TargetCommand):
    command = [dx_norm, dy_norm, phase_flag]
dx/dy = target offset in the base yaw frame, ±1 at 1.5 m; phase_flag = 0
during the approach, 1 once the dwell latch has saturated. The head_pose and
body_pose slots keep the velocity env's tiny stage-0 sampling ranges at weight
0 so their input neurons stay alive (never zero-pad a slot).

Gait shaping (air_time, foot_clearance, foot_swing_height, foot_slip, walking
pose std) is gated on the command norm, which here is planar distance / 1.5 m:
GAIT_CMD_THRESHOLD = 0.20 ⇔ d > 0.30 m, i.e. OUTSIDE the standoff ring
(0.22 ± 0.08). Inside the ring the gait terms are silent, the standing pose
std applies, and the dwell/stop terms take over — so stepping in place at the
ring is not paid for by air_time.

Reward design: kit/approach_inspect_reward_design.md. Terms live in
approach_inspect_mdp.py (never in tasks/mdp.py).
"""

import math
from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg

from mjlab_microduck.robot.microduck_constants import (
    MICRODUCK_TARGET_CFG,
    MICRODUCK_WALK_ROBOT_CFG,
)
from mjlab_microduck.tasks import approach_inspect_mdp as ai_mdp
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    NUM_STEPS_PER_ENV,
    make_microduck_velocity_env_cfg,
)
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg

# Symmetry: OFF (the task is asymmetric — the target sits on one side).
ENABLE_SYMMETRY = False

# ── Task constants ────────────────────────────────────────────────────────────
STANDOFF_M      = 0.22            # ring radius the duck stops at
RING_TOL_M      = 0.08            # |d - STANDOFF_M| < tol ⇒ "in the ring"
AIM_SIGMA_RAD   = 0.25            # head-bearing Gaussian std (escapable error only)
DWELL_RATE      = 1.0 / 2.0       # dwell latch saturates after 2 s of held gaze
DWELL_SPEED_TOL = 0.05            # m/s — moving faster than this accrues no dwell
SPAWN_RADIUS_M  = (0.4, 1.5)      # target spawn radius from the robot, any bearing
CMD_SCALE_M     = 1.5             # dx/dy command = offset / CMD_SCALE_M, clipped ±1
EPISODE_LENGTH_S = 12.0           # ~5 s approach + 2 s dwell + turn + settle
# Command-norm gate for the velocity env's gait terms: norm = d / CMD_SCALE_M,
# so 0.20 ⇔ d > 0.30 m ⇔ outside the ring (0.22 + 0.08).
GAIT_CMD_THRESHOLD = (STANDOFF_M + RING_TOL_M) / CMD_SCALE_M

# Regularisers: ~60 % of the velocity env's values (smaller task stack ⇒ the
# same weight lands harder), ramped in from ~0 so no attempt-tax is active
# while the approach skill is being discovered.
REG_SCALE = 0.6


def _head_cfg() -> SceneEntityCfg:
    """Head forward = jaw_soft CoM → mouth_tip site (the pair
    apply_mouth_payload_force uses). A fresh instance per term: the manager
    resolves SceneEntityCfg params in place, and only those passed in params."""
    return SceneEntityCfg("robot", body_names=["jaw_soft"], site_names=["mouth_tip"])


def make_microduck_approach_inspect_env_cfg(
    play: bool = False,
    rough: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create the Microduck approach-and-inspect environment configuration."""

    # ── Walk layer: the velocity recipe, verbatim (DR / sim2real untouched) ──
    cfg = make_microduck_velocity_env_cfg(play=play, rough=rough)

    # ── Scene: add the static target. Robot MUST stay the first entity ───────
    cfg.scene.entities = {
        "robot": MICRODUCK_WALK_ROBOT_CFG,
        "target": MICRODUCK_TARGET_CFG,
    }
    if not rough:
        # Headroom for foot–target contacts on top of the walk model's budget.
        cfg.sim.nconmax = 50
    cfg.episode_length_s = EPISODE_LENGTH_S

    # ── Command: [dx_norm, dy_norm, phase_flag] from state ───────────────────
    twist = deepcopy(cfg.commands["twist"])
    twist.rel_standing_envs = 0.0
    twist.rel_heading_envs = 0.0
    twist.rel_turn_in_place_envs = 0.0
    twist.heading_command = False
    twist.ranges.heading = None
    twist.resampling_time_range = (EPISODE_LENGTH_S * 2, EPISODE_LENGTH_S * 4)
    twist.debug_vis = False
    cfg.commands["twist"] = ai_mdp.TargetCommandCfg(
        **vars(twist), target_name="target", scale_m=CMD_SCALE_M,
    )

    # ── Events: spawn the target AFTER reset_base (dict insertion order) ─────
    cfg.events["reset_target"] = EventTermCfg(
        func=ai_mdp.reset_target_around_robot,
        mode="reset",
        params={"radius_range": SPAWN_RADIUS_M, "asset_name": "target"},
    )

    # ── Observations: actor unchanged (61-D); critic gets target + dwell ────
    cfg.observations["critic"].terms["target_position"] = ObservationTermCfg(
        func=microduck_mdp.ball_pos_in_base, params={"asset_name": "target"},
    )
    cfg.observations["critic"].terms["dwell_progress"] = ObservationTermCfg(
        func=ai_mdp.dwell_progress_obs,
    )

    # ── Rewards: drop velocity tracking, keep gait shaping outside the ring ──
    for name in ("track_linear_velocity", "track_angular_velocity"):
        cfg.rewards.pop(name, None)
    for name in ("air_time", "foot_clearance", "foot_swing_height", "foot_slip"):
        cfg.rewards[name].params["command_threshold"] = GAIT_CMD_THRESHOLD
    cfg.rewards["pose"].params["walking_threshold"] = GAIT_CMD_THRESHOLD

    # Head is the task's tool: keep the head_pose slot alive (tiny ranges,
    # weight 0), drop the droop-bias term and its curriculum.
    cfg.rewards["head_pose_tracking"].weight = 0.0
    cfg.rewards.pop("head_pose_bias", None)
    cfg.curriculum.pop("head_pose_bias_weight", None)
    cfg.curriculum.pop("head_pose_range", None)   # stay at stage-0 tiny ranges
    cfg.curriculum.pop("standing_envs", None)     # velocity-command only

    # ── Rewards: task (approach_inspect_reward_design.md) ────────────────────
    # Potential-based approach: Δ(-|d - standoff|). Closing pays, holding pays
    # zero, overshoot charges. No per-step proximity jackpot.
    cfg.rewards["ring_progress"] = RewardTermCfg(
        func=ai_mdp.delta_ring_distance,
        weight=6.0,
        params={"standoff": STANDOFF_M, "target_name": "target"},
    )
    # Gaussian on head bearing, hard-gated on being in the ring (a GOOD state).
    cfg.rewards["head_aim"] = RewardTermCfg(
        func=ai_mdp.head_bearing_gaussian,
        weight=3.0,
        params={"sigma": AIM_SIGMA_RAD, "gate_tol": RING_TOL_M,
                "standoff": STANDOFF_M, "head_cfg": _head_cfg(),
                "target_name": "target"},
    )
    # L1 on a 1 s EMA of the signed bearing error: charges the DC aim bias,
    # lets the walking head oscillation cancel. Self-negating ⇒ POSITIVE weight.
    cfg.rewards["head_aim_bias"] = RewardTermCfg(
        func=ai_mdp.head_bearing_ema_l1_penalty,
        weight=1.5,
        params={"tau_s": 1.0, "head_cfg": _head_cfg(), "target_name": "target"},
    )
    # Slewed dwell latch (in ring & aimed & stopped): pays Δprogress, no jackpot.
    # MUST precede turn_away and the command's phase flag reads it.
    cfg.rewards["dwell"] = RewardTermCfg(
        func=ai_mdp.delta_dwell_progress,
        weight=4.0,
        params={"rate": DWELL_RATE, "gate_tol": RING_TOL_M, "aim_tol": AIM_SIGMA_RAD,
                "speed_tol": DWELL_SPEED_TOL, "standoff": STANDOFF_M,
                "head_cfg": _head_cfg(), "target_name": "target"},
    )
    # -|v_xy| inside the ring only. Self-negating ⇒ POSITIVE weight.
    cfg.rewards["stop_in_ring"] = RewardTermCfg(
        func=ai_mdp.planar_speed_gated_penalty,
        weight=2.0,
        params={"gate_tol": RING_TOL_M, "standoff": STANDOFF_M, "target_name": "target"},
    )
    # Potential-based turn-away, frozen until the dwell latch saturates.
    cfg.rewards["turn_away"] = RewardTermCfg(
        func=ai_mdp.delta_heading_progress,
        weight=2.0,
        params={"target_offset": math.pi, "require_dwell": 0.999, "target_name": "target"},
    )

    # ── Regularisers: 60 % of velocity, ramped from ~0 ───────────────────────
    cfg.rewards["action_rate_l2"].weight = -0.1 * REG_SCALE
    cfg.rewards["body_ang_vel"].weight = -0.05 * REG_SCALE
    cfg.rewards["angular_momentum"].weight = -0.02 * REG_SCALE
    # Neck-only smoothness: the head is the tool here, so it gets its own
    # damping (the velocity env deliberately has none). Ramped with action_rate.
    cfg.rewards["neck_action_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.neck_action_rate_l2, weight=0.0,
    )
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0,                         "weight": -0.1 * REG_SCALE},
                {"step": 500 * NUM_STEPS_PER_ENV,   "weight": -0.2 * REG_SCALE},
                {"step": 750 * NUM_STEPS_PER_ENV,   "weight": -0.4 * REG_SCALE},
                {"step": 1000 * NUM_STEPS_PER_ENV,  "weight": -0.6 * REG_SCALE},
                {"step": 1250 * NUM_STEPS_PER_ENV,  "weight": -0.8 * REG_SCALE},
                {"step": 1500 * NUM_STEPS_PER_ENV,  "weight": -1.0 * REG_SCALE},
            ],
        },
    )
    cfg.curriculum["neck_action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "neck_action_rate_l2",
            "weight_stages": [
                {"step": 0,                         "weight": 0.0},
                {"step": 500 * NUM_STEPS_PER_ENV,   "weight": -0.15},
                {"step": 1000 * NUM_STEPS_PER_ENV,  "weight": -0.3},
                {"step": 1500 * NUM_STEPS_PER_ENV,  "weight": -0.6},
            ],
        },
    )

    return cfg


MicroduckApproachInspectRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,  # normalizer MUST be baked into ONNX by export.py
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
        symmetry_cfg=None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="approach_inspect",
    run_name="approach_inspect",
    save_interval=250,
    num_steps_per_env=NUM_STEPS_PER_ENV,
    max_iterations=6_000,
)

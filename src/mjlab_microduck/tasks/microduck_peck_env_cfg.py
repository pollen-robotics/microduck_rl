"""Microduck *peck* task — from the sit, extend one leg and peck at that foot.

A cat-grooming trick: seated, the duck extends the PECK_FOOT leg forward,
drops its head onto the foot and pecks it rhythmically. The mouth is NOT part
of the policy (every policy is 14 actions, the mouth joint is skipped) — the
daemon opens/chews the mouth on its own while this policy runs.

Episodic, constant command (publishable as a one-shot skill):
    cmd (twist slot) = [peck_flag, 0, 0]   peck_flag ∈ {0 = SIT, 1 = PECK}
Deployment: [1, 0, 0] for ``duration_s`` (reach + peck), then the manifest's
idle [0, 0, 0] for ``unwind_s`` folds the leg back and sits up straight; then
hand back to the sitstand slot, which resumes seated. entry_pose = "sitting".

Two-sided like the ball kick: PECK_FOOT is a factory argument, and both sides
are registered as separate tasks / runs (Mjlab-Peck-Right / -Left). The left
pose is the exact mirror of the right one through the symmetry tables
(symmetry.py) — the robot was posed once, on its right foot.

Design (sitstand template — commanded two-state from the sit):
  - Built ON make_microduck_sitstand_env_cfg: same full-collision robot,
    seated contact-solver hardening, velocity-parity DR / obs noise / delays /
    regularisers / push ramp. Sitstand's task rewards, head-command curriculum
    and seated hardening events are replaced.
  - Slewed target (PeckCommand.alpha, PECK_RAMP_S): every pose/height/beak
    reward tracks the moving SIT→PECK blend, so being ahead of the ramp pays
    nothing — a gentle reach is the argmax (no-jackpot rule).
  - Single fixed target pose, no keyframes: legs + neck Gaussian + L1, height,
    multiplicative composite, beak-to-foot proximity (scaled by alpha).
  - Feasibility measured in the training env with scripted actions
    (scratchpad peck_scripted_rollout.py): the XL330/BAM is a low-stiffness
    P servo (~0.45 Nm/rad) and the dropped head hangs on the neck's hard stop,
    so small target bobs (±0.4 rad) move the beak only 2-3 mm. With the
    overshoot a policy will learn (head_pitch −1.0 at 1.5 Hz) the beak lifts
    22 mm and 5-6 pecks / 4.7 s pay in every env, tilt ≤ 19°.
  - The peck itself: ``peck_rhythm`` pays per head→foot touchdown after a real
    beak lift (≥ PECK_LIFT_MIN in world z) and ≥ PECK_T_MIN of air; air time
    capped at PECK_T_MAX → best rate is a ~2 Hz peck. Introduced by curriculum
    AFTER the reach exists (skill-discovery ordering). Slamming is priced by
    ``peck_impact_cost`` (head-foot force above the resting head weight).
  - Self-collision cost excludes exactly the allowed head↔target-foot pair
    (the goal IS a self-contact; measured: it is the only self-contact in the
    pose).
  - Reverse-curriculum spawns: a fraction of episodes start partway along
    SIT→PECK, so the last mile and the return-to-sit get data from step 0.

Pose: recorded on the real robot 2026-11-17 (robotctl monitor, daemon
joint 9 = mouth dropped). Several recorded angles are outside what training
allows: hip_roll ±0.51-0.56 and the extended hip_pitch 1.67 exceed the MJCF
ranges (±0.384, ±1.571); the folded knee 1.51 and neck_pitch -1.53 exceed the
0.9 soft limits. Clamped: hip_roll ±0.33, hip_pitch 1.40, knee 1.40 (inside
the soft range — the "parking on hard limits" lesson). neck_pitch is the one
deliberate exception at -1.50 (soft -1.44, hard -1.571): a sweep showed the
head can't reach the foot from -1.43 without the trunk tipping ~12° forward
(transients to 25°), while -1.50 sits at 4°.
Verified in CPU MuJoCo + BAM M6 before training (scratchpad
peck_pose_check.py): from the SIT keyframe a 2 s open-loop slew to this pose
stays seated (end tilt ≈ 4°, ≤ 18° transient; trunk z 0.061 → 0.052) and the
head ends in contact with the target foot (gap ≈ 0 mm), both sides; the
head↔target-ankle contact is the pose's only self-contact.
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
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlModelCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks import microduck_sitstand_env_cfg as sitstand
from mjlab_microduck.tasks.symmetry import (
    PpoWithSymmetryCfg,
    SYMMETRY_CFG,
    _JOINT_PERM,
    _JOINT_SIGN,
)

# Symmetry mirror-loss — must stay OFF: the task is one-footed by design.
ENABLE_SYMMETRY = False

# ── Episode / command ─────────────────────────────────────────────────────────
EPISODE_LENGTH_S = 10.0
# Dwell in each commanded state before a resample may flip it: long enough for
# a full reach (PECK_RAMP_S) plus several pecks, or a full return + rest.
PECK_DWELL_S     = (3.5, 6.0)
# P(resample commands PECK). < 1 so "hold the sit" (all-zero command = the
# deployment idle) and "return to sit" are trained explicitly.
PECK_PROB        = 0.75
# Seconds for the slewed target to traverse SIT↔PECK. The measured pose is
# ~1.9 rad on neck_pitch: 1.5 s keeps the head drop gentle.
PECK_RAMP_S      = 1.5

# ── Peck rhythm ───────────────────────────────────────────────────────────────
PECK_T_MIN    = 0.15   # s of beak air before a touch counts (anti-buzz)
PECK_T_MAX    = 0.40   # s; longer lifts pay no more → ~2 Hz is the best rate
PECK_LIFT_MIN = 0.012  # m the beak tip must rise (world z) between touches
PECK_FORCE_FREE = 3.0  # N; head resting on the foot ≈ 2.4 N is free

# ── Poses (14 servo joints, policy order) ─────────────────────────────────────
_HOME = [0.0, -0.0873, -0.4579, -0.0049, 0.4530,
         0.3491, 0.3491, 0.0, 0.0,
         0.0, 0.0873, 0.4579, 0.0049, -0.4530]

# SIT = HOME + sitstand's stability-verified keyframe (neck/head at HOME).
SIT_POSE = list(_HOME)
for _i, _v in sitstand.SITTING_TARGET_OVERRIDES.items():
    SIT_POSE[_i] = _v

# Recorded on the robot (right foot), hip_roll / right hip_pitch clamped into
# the soft joint range — see the module docstring.
PECK_POSE_RIGHT = [
    -0.0276,  0.33,   -0.3988,  1.40,    0.6903,   # left leg: folded, seated
    -1.50,    0.0476, -0.4709, -0.0951,            # neck_pitch, head_pitch, head_yaw, head_roll
    -0.0169, -0.33,    1.40,   -0.0982, -0.4495,   # right leg: extended forward
]
# Beak tip (mouth_tip site) in the ankle_right body frame at that pose, m
# (measured, CPU MuJoCo + BAM, 4 noisy seeds, std ≤ 2.4 mm). The ankle frames
# are mirror images: the left value is x-negated (measured -18.9, -33.1, -49.0).
BEAK_IN_FOOT_RIGHT = (0.0185, -0.0331, -0.0488)

# Trunk heights (m), measured in the same CPU check: SIT settles at 0.061,
# the peck pose at 0.052.
SIT_Z  = 0.061
PECK_Z = 0.052

_LEG_JOINTS  = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
_NECK_JOINTS = [5, 6, 7, 8]
_ALL_JOINTS  = list(range(14))


def mirror_pose(pose: list) -> list:
    """Left↔right mirror of a 14-joint absolute pose (symmetry.py tables).

    Valid for ABSOLUTE angles because HOME itself is exactly mirror-symmetric
    under the same tables (left/right leg signs opposite, neck pitches even).
    """
    return [pose[_JOINT_PERM[i]] * _JOINT_SIGN[i] for i in range(14)]


def peck_targets(peck_foot: str) -> tuple[list, tuple]:
    """(PECK pose, beak offset in the target ankle frame) for one side."""
    assert peck_foot in ("right", "left")
    if peck_foot == "right":
        return list(PECK_POSE_RIGHT), BEAK_IN_FOOT_RIGHT
    x, y, z = BEAK_IN_FOOT_RIGHT
    return mirror_pose(PECK_POSE_RIGHT), (-x, y, z)


def make_microduck_peck_env_cfg(
    play: bool = False,
    peck_foot: str = "right",
) -> ManagerBasedRlEnvCfg:
    """Create the Microduck peck environment configuration for one side."""
    peck_pose, beak_offset = peck_targets(peck_foot)
    foot_body = f"ankle_{peck_foot}"

    # Sitstand base: seated-capable robot, DR / obs / delays / noise / solver.
    cfg = sitstand.make_microduck_sitstand_env_cfg(play=play, rough=False)
    cfg.episode_length_s = EPISODE_LENGTH_S

    # ── Sensors: + head-on-target-foot ────────────────────────────────────────
    beak_foot_cfg = ContactSensorCfg(
        name="beak_foot_contact",
        primary=ContactMatch(mode="body", pattern="jaw_soft", entity="robot"),
        secondary=ContactMatch(mode="body", pattern=foot_body, entity="robot"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
    )
    cfg.scene.sensors = tuple(cfg.scene.sensors) + (beak_foot_cfg,)

    # ── Rewards: drop sitstand's task stack, keep its regularisers ────────────
    for name in [
        "posture_pose_legs", "head_pose_tracking", "posture_pose_l1",
        "posture_height", "posture_height_sharp", "posture_height_l1",
        "rise_bootstrap", "rise_speed", "upright_while_tall",
        "posture_stillness", "posture_composite", "neck_strain_leaned",
        "self_collisions",
    ]:
        cfg.rewards.pop(name, None)
    # Kept from sitstand (velocity parity): descent_speed (+, self-negating),
    # gentle_motion (+, self-negating), upright_linear 2.5, action_rate_l2,
    # joint_torque_rate_l2, body_ang_vel, angular_momentum.

    pose_params = {"command_name": "twist", "sit_pose": SIT_POSE, "peck_pose": peck_pose}

    # Legs — Gaussian (generous: knee/hip travel up to ~1.4 rad) + L1 pull.
    cfg.rewards["peck_pose_legs"] = RewardTermCfg(
        func=microduck_mdp.peck_pose_match, weight=4.0,
        params={**pose_params, "joint_indices": _LEG_JOINTS, "std": 0.5},
    )
    cfg.rewards["peck_pose_legs_l1"] = RewardTermCfg(
        func=microduck_mdp.peck_pose_l1, weight=1.0,   # self-negating → POSITIVE weight
        params={**pose_params, "joint_indices": _LEG_JOINTS},
    )
    # Neck/head — looser: the peck IS a neck oscillation around this pose, and
    # the oscillation must stay cheap (price only the escapable DC part).
    cfg.rewards["peck_pose_neck"] = RewardTermCfg(
        func=microduck_mdp.peck_pose_match, weight=1.5,
        params={**pose_params, "joint_indices": _NECK_JOINTS, "std": 0.6},
    )
    cfg.rewards["peck_pose_neck_l1"] = RewardTermCfg(
        func=microduck_mdp.peck_pose_l1, weight=0.5,   # self-negating → POSITIVE weight
        params={**pose_params, "joint_indices": _NECK_JOINTS},
    )

    cfg.rewards["peck_height"] = RewardTermCfg(
        func=microduck_mdp.peck_height_gaussian, weight=1.0,
        params={"command_name": "twist", "sit_z": SIT_Z, "peck_z": PECK_Z, "std": 0.015},
    )

    # Beak at the foot — scaled by the slewed blend (0 when seated).
    cfg.rewards["peck_beak_proximity"] = RewardTermCfg(
        func=microduck_mdp.peck_beak_proximity, weight=2.0,
        params={"command_name": "twist", "foot_body": foot_body,
                "beak_offset_in_foot": beak_offset, "std": 0.03},
    )

    # The peck: per good touchdown (≤ 1 each). Weight set by curriculum below
    # (0 until the reach exists); 20/peck at ~2 Hz ≈ +0.8/step ≈ 5% of stack.
    cfg.rewards["peck_rhythm"] = RewardTermCfg(
        func=microduck_mdp.peck_rhythm, weight=0.0,
        params={
            "command_name": "twist", "sensor_name": beak_foot_cfg.name,
            "foot_body": foot_body, "beak_offset_in_foot": beak_offset,
            "t_min": PECK_T_MIN, "t_max": PECK_T_MAX, "lift_min": PECK_LIFT_MIN,
        },
    )
    # Slamming the head into the foot. Cost ≥ 0 → NEGATIVE weight.
    # Scripted-rollout calibration (training env, CPU): a head_pitch nod
    # (−1.0 overshoot, 1.5 Hz) pecks 5-6×/4.7 s at ≤ 6.7 N; a neck slam
    # (+1.5) pecks as often but at 19.5 N and 57° trunk tilt. At −0.5 the
    # slam cost only ≈ −0.03/step vs ≈ +0.5/step peck pay; at −1.5 a 19 N
    # touch costs ≈ 8 of its 20 payout, a gentle 6 N one ≈ 2.
    cfg.rewards["peck_impact"] = RewardTermCfg(
        func=microduck_mdp.peck_impact_cost, weight=-1.5,
        params={"sensor_name": beak_foot_cfg.name, "force_threshold": PECK_FORCE_FREE},
    )

    # Rest stillness (trunk) at either target, upright. posture_stillness is
    # reused with its STAND slot = SIT_Z (alpha 0) and SIT slot = PECK_Z
    # (alpha 1): its target z is stand_z + alpha·(sit_z − stand_z).
    cfg.rewards["rest_stillness"] = RewardTermCfg(
        func=microduck_mdp.posture_stillness, weight=1.5,
        params={
            "command_name": "twist", "stand_z": SIT_Z, "sit_z": PECK_Z,
            "band_full": 0.012, "band_zero": 0.03, "vel_std": 0.05,
            "tilt_full_deg": 25.0, "tilt_zero_deg": 60.0,
        },
    )

    # Multiplicative goal score vs the slewed target: a lean / flop / leg left
    # folded collapses it. Neck factor wide (peck oscillation stays cheap).
    cfg.rewards["peck_composite"] = RewardTermCfg(
        func=_peck_composite, weight=3.0,
        params={**pose_params, "sit_z": SIT_Z, "peck_z": PECK_Z,
                "height_std": 0.03, "upright_std": 0.40, "leg_std": 0.40, "neck_std": 0.60},
    )

    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=microduck_mdp.self_collision_cost_excluding, weight=-1.0,
        params={"sensor_name": "self_collision", "exclude_sensor_name": beak_foot_cfg.name},
    )

    # ── Observations ──────────────────────────────────────────────────────────
    # Actor: unchanged 61D (head_command stays the tiny-range head_pose command
    # → neurons alive, daemon writes zeros). Critic: + peck geometry/progress.
    cfg.observations["critic"].terms["peck_beak_target"] = ObservationTermCfg(
        func=microduck_mdp.peck_beak_target_b,
        params={"foot_body": foot_body, "beak_offset_in_foot": beak_offset},
    )
    cfg.observations["critic"].terms["peck_alpha"] = ObservationTermCfg(
        func=microduck_mdp.peck_alpha_obs, params={"command_name": "twist"},
    )

    # ── Command: peck flag in the twist vx slot, slewed target ───────────────
    old = cfg.commands["twist"]
    fields = {k: v for k, v in vars(old).items()
              if k not in ("class_type", "sit_prob", "ramp_s", "sit_z", "stand_z")}
    fields["resampling_time_range"] = PECK_DWELL_S
    cfg.commands["twist"] = microduck_mdp.PeckCommandCfg(
        **fields, peck_prob=PECK_PROB, ramp_s=PECK_RAMP_S,
    )

    # ── Terminations: sitstand's (nan_state only, no fall termination) ───────
    # Smoke test with fell_over (70°) re-added: the initial random policy
    # (std 1.0) tipped within ~20 steps and every episode was truncated before
    # the reach. Tips play out instead; upright_linear / composite (≈0 on a
    # flop) price them, as in sitstand.
    assert "fell_over" not in cfg.terminations

    # ── Events ────────────────────────────────────────────────────────────────
    for name in ("seated_head_jerk", "seated_tip"):
        cfg.events.pop(name, None)
    cfg.events["set_ground_state"] = EventTermCfg(
        func=microduck_mdp.reset_peck_spawn,
        mode="reset",
        params={
            "sit_pose": SIT_POSE, "peck_pose": peck_pose,
            "mid_prob": 0.3, "mid_alpha_range": (0.3, 1.0),
            "joint_noise_std": 0.05, "tilt_max": math.radians(6),
            "z_range": (0.062, 0.072),
        },
    )

    # ── Curriculum ────────────────────────────────────────────────────────────
    for name in ("head_pose_range", "rise_speed_weight",
                 "seated_head_jerk_prob", "seated_tip_range"):
        cfg.curriculum.pop(name, None)
    # The peck is paid only once the reach exists (sitstand-sized tasks reach
    # their pose by ~300-500 iters). If peck_rhythm stays ~0 after 600, the
    # pose stack is outbidding it — raise the weight, don't move it earlier.
    cfg.curriculum["peck_rhythm_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "peck_rhythm",
            "weight_stages": [
                {"step": 0,         "weight": 0.0},
                {"step": 400 * 24,  "weight": 20.0},
            ],
        },
    )

    return cfg


def _peck_composite(
    env,
    command_name: str,
    sit_pose: list,
    peck_pose: list,
    sit_z: float,
    peck_z: float,
    height_std: float,
    upright_std: float,
    leg_std: float,
    neck_std: float,
) -> "torch.Tensor":
    import torch

    asset = env.scene["robot"]
    alpha, target = microduck_mdp._peck_pose_target(env, command_name, sit_pose, peck_pose)
    q = microduck_mdp._servo_joint_pos(env, asset)
    target_z = sit_z + alpha * (peck_z - sit_z)
    z = microduck_mdp._trunk_z(env)
    h = torch.exp(-((z - target_z) / height_std) ** 2)
    quat = asset.data.root_link_quat_w
    u = torch.exp(-2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2) / upright_std**2)
    leg_err = ((q[:, _LEG_JOINTS] - target[:, _LEG_JOINTS]) ** 2).mean(dim=-1)
    neck_err = ((q[:, _NECK_JOINTS] - target[:, _NECK_JOINTS]) ** 2).mean(dim=-1)
    return h * u * torch.exp(-leg_err / leg_std**2) * torch.exp(-neck_err / neck_std**2)


# ── RL runner configs (one per side, distinct experiment names) ──────────────

def _make_rl_cfg(peck_foot: str) -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
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
            symmetry_cfg=SYMMETRY_CFG if ENABLE_SYMMETRY else None,
        ),
        wandb_project="mjlab_microduck",
        experiment_name=f"microduck_peck_{peck_foot}",
        run_name=f"microduck_peck_{peck_foot}",
        save_interval=250,
        num_steps_per_env=24,
        max_iterations=3_000,
    )


MicroduckPeckRightRlCfg = _make_rl_cfg("right")
MicroduckPeckLeftRlCfg = _make_rl_cfg("left")

"""Microduck JUMP (hop) task — flat ground, walk model, phase-driven.

This is the GPU counterpart of the CPU lab's ``jump`` behaviour: crouch, push
off, get both feet clear of the ground, land softly, return to standing, repeat.

Why this task is worth a GPU at all
-----------------------------------
On the CPU stack (single-threaded MuJoCo + SB3, 2.1k env-steps/s, 8 envs max)
the hop recipe plateaued at v5 — roughly ten low hops — and stopped improving
with more steps. That is a sample-throughput wall, not a reward-design wall:
the push-off/landing cycle is a narrow timing window, and with 8 weakly
de-correlated envs the policy keeps re-learning the same approach. Batch
physics is the fix, so this cfg targets the standard MJLab/Warp path rather
than extending the CPU recipe.

Design note — the flight phase is not tracked
---------------------------------------------
The wind-up and the push-off are actuated and well shaped (``jump_crouch_depth``,
``jump_launch``). Flight is NOT: in free flight the only achievable trunk
height profile is a parabola, so a prescribed curve is unsatisfiable and the
cheapest way for a policy to stop paying that cost is to never leave the
ground. Flight therefore pays results only — ``jump_airborne`` (feet clear) and
``jump_apex`` (how high the body actually got) — and landing charges the impact
speed once, at the touchdown edge. See the header comment in ``tasks/mdp.py``.

Runtime slot
------------
Obs/action layout is the unified 61D one, identical to walking / spin /
ground_pick, so the exported ONNX is interchangeable at runtime with the other
gesture policies.

⚠️ NOT RUNNABLE WITHOUT CUDA. ``mjlab`` / ``mujoco_warp`` / ``jax`` are not part
of the CPU dev environment, so this cfg has never been constructed. It is
written against the same APIs as the neighbouring cfgs and is checked by a
static symbol/registration test; the first real construction must happen on a
CUDA box (see runs/JUMP_TASK.md).
"""

import math
from copy import deepcopy

# Symmetry — same call as every other v1.5 env: SYMMETRY_CFG's _OBS_PERM is
# hardcoded for the old 51D layout and would corrupt the 61D obs. A hop IS
# left/right symmetric, so this is one to revisit once SYMMETRY_CFG is
# rewritten for the new layout — it is free data we are currently leaving out.
ENABLE_SYMMETRY = False

# ── Domain randomisation toggles (matched to the velocity env) ────────────────
ENABLE_COM_RANDOMIZATION             = True
ENABLE_HEAD_COM_RANDOMIZATION        = True
ENABLE_MASS_INERTIA_RANDOMIZATION    = True
ENABLE_JOINT_FRICTION_RANDOMIZATION  = True
ENABLE_ARMATURE_RANDOMIZATION        = True
ENABLE_VELOCITY_PUSHES               = True
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True
ENABLE_ENCODER_BIAS                  = True

# ── Ranges (matched to the velocity env) ──────────────────────────────────────
COM_RANDOMIZATION_RANGE          = 0.003
HEAD_COM_RANDOMIZATION_RANGE     = 0.003
MASS_INERTIA_RANDOMIZATION_RANGE = (0.95, 1.05)
JOINT_FRICTION_RANDOMIZATION_RANGE = (0.9, 1.1)
ARMATURE_RANDOMIZATION_RANGE     = (0.9, 1.1)
VELOCITY_PUSH_INTERVAL_S         = (3.0, 6.0)
# Gentler than velocity's ±0.3: a hop is a narrow balance window, and a hard
# shove mid-flight cannot be answered by anything the policy does — it only
# adds variance to the landing signal it is supposed to learn from.
VELOCITY_PUSH_RANGE              = (-0.15, 0.15)
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0
ENCODER_BIAS_RANGE               = (-0.015, 0.015)

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlModelCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.robot.microduck_constants import MICRODUCK_WALK_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import HEAD_BODY_NAMES
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg, SYMMETRY_CFG

# Cycle length. Shorter than the ground-pick 4 s by design: a hop is a fast
# gesture and a 4 s cycle would spend most of the episode standing, starving
# the launch/landing windows of samples. 1.2 s = ~0.36 s wind-up, 0.18 s push,
# 0.36 s air, 0.30 s recover.
JUMP_PERIOD = 1.2
# Wind-up must END before the trajectory leaves the ground, otherwise the
# crouch-shaping term pays the robot for curling up mid-flight.
assert microduck_mdp.JUMP_CROUCH_END < microduck_mdp.JUMP_LAUNCH_END
assert microduck_mdp.JUMP_LAUNCH_END < microduck_mdp.JUMP_FLIGHT_END

NECK_PATTERN_NO_YAW = r"^(neck_pitch|head_pitch|head_roll)$"


def make_microduck_jump_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Env for the phase-driven hop on flat ground (walk model)."""

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(ankle_l_v1|ankle_r_v1)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    cfg = make_velocity_env_cfg()
    # WALK model, not a jump-specific one: the hop needs no new geometry, and
    # reusing the walking spec is what keeps the exported ONNX loadable in the
    # same runtime slot as the walk/spin/ground-pick policies.
    cfg.scene.entities = {"robot": MICRODUCK_WALK_ROBOT_CFG}
    cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg)
    cfg.viewer.body_name = "trunk_base"

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0

    # === REWARDS ===
    # Drop the locomotion-tracking stack: there is no velocity command here,
    # the command slot carries the phase. Keeping track_linear_velocity with a
    # phase vector as "command" would pay for a nonsense target.
    keep = {"upright", "body_ang_vel", "action_rate_l2", "self_collisions"}
    for name in list(cfg.rewards.keys()):
        if name not in keep:
            del cfg.rewards[name]

    cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["upright"].weight = 2.0
    # Tight vertical std: the hop wants a level body, not a nose-down arc.
    cfg.rewards["upright"].params["std"] = math.sqrt(0.05)
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["action_rate_l2"].weight = -0.1

    # --- the hop itself -----------------------------------------------------
    # Wind-up: settle onto a reachable crouch. Paid only while grounded (via
    # the phase window), so it cannot be farmed from the air.
    cfg.rewards["jump_crouch"] = RewardTermCfg(
        func=microduck_mdp.jump_crouch_depth,
        weight=1.0,
        params={
            "command_name": "twist",
            "stand_z": microduck_mdp.JUMP_STAND_Z,
            "crouch_z": microduck_mdp.JUMP_CROUCH_Z,
            "crouch_end": microduck_mdp.JUMP_CROUCH_END,
        },
    )
    # Push-off: capped upward velocity. Capped so the reward stops paying for
    # escalating violence once the body is already moving up fast enough.
    cfg.rewards["jump_launch"] = RewardTermCfg(
        func=microduck_mdp.jump_launch,
        weight=4.0,
        params={
            "command_name": "twist",
            "max_vz": 1.5,
            "launch_start": microduck_mdp.JUMP_CROUCH_END,
            "launch_end": microduck_mdp.JUMP_LAUNCH_END,
        },
    )
    # Airborne: the term that makes it a hop rather than a squat-and-rise.
    cfg.rewards["jump_airborne"] = RewardTermCfg(
        func=microduck_mdp.jump_airborne,
        weight=3.0,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "flight_start": microduck_mdp.JUMP_LAUNCH_END,
            "flight_end": microduck_mdp.JUMP_FLIGHT_END,
        },
    )
    # Apex: pays the achieved height, never a prescribed curve.
    cfg.rewards["jump_apex"] = RewardTermCfg(
        func=microduck_mdp.jump_apex,
        weight=6.0,
        params={
            "command_name": "twist",
            "stand_z": microduck_mdp.JUMP_STAND_Z,
            "min_rise": 0.0,
            "flight_start": microduck_mdp.JUMP_LAUNCH_END,
            "flight_end": microduck_mdp.JUMP_FLIGHT_END,
        },
    )
    # Soft landing: cost on the touchdown edge only. Charged per-step instead,
    # it would make collapsing on landing cheaper than standing back up.
    cfg.rewards["jump_impact"] = RewardTermCfg(
        func=microduck_mdp.jump_impact_speed,
        weight=-2.0,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "recover_start": microduck_mdp.JUMP_FLIGHT_END,
        },
    )

    # --- shared stability / sim2real ---------------------------------------
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": "self_collision"},
    )
    cfg.rewards["feet_flat"] = RewardTermCfg(
        func=microduck_mdp.feet_flat_penalty,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
            "sensor_name": "feet_ground_contact",
        },
    )
    cfg.rewards["neck_action_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.neck_action_rate_l2, weight=-0.5
    )
    cfg.rewards["neck_joint_pos_l2"] = RewardTermCfg(
        func=microduck_mdp.neck_joint_pos_l2,
        weight=-0.2,
        params={"pattern": NECK_PATTERN_NO_YAW},
    )
    cfg.rewards["joint_torques_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torques_l2, weight=-1e-3
    )

    # === TERMINATIONS ===
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan, time_out=False,
    )

    # === EVENTS ===
    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history, mode="reset",
    )
    del cfg.events["foot_friction"]

    if ENABLE_VELOCITY_PUSHES:
        cfg.events["push_robot"] = EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=VELOCITY_PUSH_INTERVAL_S,
            params={
                "velocity_range": {"x": VELOCITY_PUSH_RANGE, "y": VELOCITY_PUSH_RANGE},
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

    cfg.events["reset_base"].params["pose_range"]["z"] = (0.1335, 0.1435)
    # No entry velocity: the hop starts from a standstill, like the button.
    # Explicit zero ranges (the house convention — see velocity_command_ranges_
    # curriculum / ball_kick stage 0), NOT an empty dict: reset_root_state_uniform
    # indexes this per-axis.
    cfg.events["reset_base"].params["velocity_range"] = {
        "x": (0.0, 0.0), "y": (0.0, 0.0),
    }

    if ENABLE_COM_RANDOMIZATION:
        cfg.events["randomize_com"] = EventTermCfg(
            func=dr.body_ipos, mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "operation": "add",
                "ranges": (-COM_RANDOMIZATION_RANGE, COM_RANDOMIZATION_RANGE),
            },
        )
    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.events["randomize_head_com"] = EventTermCfg(
            func=dr.body_ipos, mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=HEAD_BODY_NAMES),
                "operation": "add",
                "ranges": (-HEAD_COM_RANDOMIZATION_RANGE, HEAD_COM_RANDOMIZATION_RANGE),
            },
        )
    if ENABLE_MASS_INERTIA_RANDOMIZATION:
        _mi_lo, _mi_hi = MASS_INERTIA_RANDOMIZATION_RANGE
        cfg.events["randomize_mass_inertia"] = EventTermCfg(
            func=dr.pseudo_inertia, mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "alpha_range": (math.log(_mi_lo) / 2.0, math.log(_mi_hi) / 2.0),
            },
        )
    if ENABLE_JOINT_FRICTION_RANDOMIZATION:
        cfg.events["randomize_joint_friction"] = EventTermCfg(
            func=microduck_mdp.randomize_bam_friction, mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "scale_range": JOINT_FRICTION_RANDOMIZATION_RANGE,
            },
        )
    if ENABLE_ARMATURE_RANDOMIZATION:
        cfg.events["randomize_armature"] = EventTermCfg(
            func=dr.joint_armature, mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",)),
                "operation": "scale",
                "ranges": ARMATURE_RANDOMIZATION_RANGE,
            },
        )

    # === OBSERVATIONS (unified 61D layout) ===
    del cfg.observations["actor"].terms["base_lin_vel"]
    del cfg.observations["critic"].terms["foot_height"]
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]
    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.base_lin_vel, scale=1.0,
    )

    gravity_term_name = "projected_gravity"
    cfg.observations["actor"].terms[gravity_term_name] = deepcopy(
        cfg.observations["actor"].terms[gravity_term_name]
    )
    cfg.observations["actor"].terms["base_ang_vel"] = deepcopy(
        cfg.observations["actor"].terms["base_ang_vel"]
    )
    cfg.observations["actor"].terms["base_ang_vel"].delay_min_lag = 0
    cfg.observations["actor"].terms["base_ang_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["base_ang_vel"].delay_update_period = 64
    cfg.observations["actor"].terms[gravity_term_name].delay_min_lag = 0
    cfg.observations["actor"].terms[gravity_term_name].delay_max_lag = 1
    cfg.observations["actor"].terms[gravity_term_name].delay_update_period = 64
    cfg.observations["actor"].terms["base_ang_vel"].noise = Unoise(n_min=-0.03, n_max=0.03)
    cfg.observations["actor"].terms[gravity_term_name].noise = Unoise(n_min=-0.01, n_max=0.01)
    cfg.observations["actor"].terms["joint_pos"].noise = Unoise(n_min=-0.001, n_max=0.001)
    cfg.observations["actor"].terms["joint_vel"].noise = Unoise(n_min=-0.25, n_max=0.25)

    if ENABLE_IMU_ORIENTATION_RANDOMIZATION:
        av = cfg.observations["actor"].terms["base_ang_vel"]
        av.func = microduck_mdp.base_ang_vel_imu_misaligned
        av.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}
        g = cfg.observations["actor"].terms[gravity_term_name]
        g.func = microduck_mdp.projected_gravity_imu_misaligned
        g.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}

    cfg.observations["actor"].terms["joint_vel"] = deepcopy(
        cfg.observations["actor"].terms["joint_vel"]
    )
    cfg.observations["actor"].terms["joint_vel"].delay_min_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_update_period = 0

    passive_excluded = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))
    for grp in ("actor", "critic"):
        for term in ("joint_pos", "joint_vel"):
            cfg.observations[grp].terms[term] = deepcopy(cfg.observations[grp].terms[term])
            cfg.observations[grp].terms[term].params["asset_cfg"] = deepcopy(passive_excluded)

    if ENABLE_ENCODER_BIAS:
        cfg.events["encoder_bias"].params["bias_range"] = ENCODER_BIAS_RANGE
        cfg.observations["actor"].terms["joint_pos"].params["biased"] = True
        cfg.observations["critic"].terms["joint_pos"].params["biased"] = False
    else:
        cfg.events.pop("encoder_bias", None)

    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 4},
        )
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 6},
        )

    # === COMMAND: phase (comme spin / ground_pick / roller_crouch) ===
    command: UniformVelocityCommandCfg = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs = 0.0
    # randomize_phase=False: every episode starts standing at phase 0, which is
    # what the runtime button does. Randomising would drop envs straight into
    # the flight window with no wind-up, i.e. train on states the deployment
    # never visits from.
    cfg.commands["twist"] = microduck_mdp.GroundPickPhaseCommandCfg(
        **{
            **vars(command),
            "class_type": microduck_mdp.GroundPickPhaseCommand,
            "period": JUMP_PERIOD,
            "randomize_phase": False,
        }
    )

    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # === CURRICULUM ===
    del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": -0.1},
                {"step": 500 * 24, "weight": -0.3},
                {"step": 1000 * 24, "weight": -0.5},
            ],
        },
    )
    if ENABLE_COM_RANDOMIZATION:
        cfg.curriculum["com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_com",
                "range_stages": [
                    {"step": 0, "range": 0.003},
                    {"step": 500 * 24, "range": 0.005},
                    {"step": 1000 * 24, "range": 0.01},
                ],
            },
        )
    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.curriculum["head_com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_head_com",
                "range_stages": [
                    {"step": 0, "range": 0.003},
                    {"step": 500 * 24, "range": 0.005},
                    {"step": 1000 * 24, "range": 0.01},
                ],
            },
        )

    return cfg


MicroduckJumpRlCfg = RslRlOnPolicyRunnerCfg(
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
        symmetry_cfg=SYMMETRY_CFG if ENABLE_SYMMETRY else None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="jump",
    run_name="jump",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=8_000,
)

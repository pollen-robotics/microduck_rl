"""Microduck RUN environment — fast alternating-leg running with a flight phase.

WHY: the velocity recipe walks conservatively (swing target 2 cm, single-foot air
time ≤ 0.3 s, tight pose stds, forward commands ≤ 0.4 m/s). Going fast on a 25 cm
biped means a different gait, not a quicker walk:

  1. the swing foot lifts and REACHES FORWARD (run_foot_reach_reward, higher
     foot_clearance / swing-height targets);
  2. the other leg is flexed and PUSHES OFF the ground hard enough to leave it
     (run_pushoff_reward: ground-reaction force > body weight);
  3. both feet are briefly airborne (run_flight_reward — capped per flight, so the
     optimum is many short flights, not a leap) and the legs ALTERNATE
     (run_alternating_step_reward — a bunny hop does not pay);
  4. the head — 38% of the robot's mass — thrusts forward in flight and returns in
     support as a counterbalance (run_head_sway_reward, introduced late).

Built on make_microduck_velocity_env_cfg so robot / 61D obs / DR / noise / delays /
NaN guards stay in sync with the deployed family (hot-swappable in the runtime).
Command slots: twist = forward-heavy (up to RUN_SPEED_STAGES' last stage), head_pose
and body_pose keep their small non-zero ranges (live input neurons, per AGENTS.md).

Sim2real: policies are UNFILTERED, like every other task here. The run gait lands
harder than a walk — watch Episode_Reward/* penalties (all ≤ 0) and rehearse in
scripts/infer_policy.py before putting it on the robot.

Every new reward is gated by forward SPEED (not just the command) so hopping or
kicking on the spot earns nothing. Taxes (double-support, head sway) are introduced
by curriculum AFTER the skill exists — see the stages below.
"""

import dataclasses
import math
import os

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import CurriculumTermCfg, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velstand_env_cfg import _collapse_curricula_to_final
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    MicroduckRlCfg,
    NUM_STEPS_PER_ENV,
    make_microduck_velocity_env_cfg,
)

# ── Tuned constants ──────────────────────────────────────────────────────────
ENABLE_HEAD_SWAY = True  # counterbalancing head motion (late curriculum)
ENABLE_DOUBLE_SUPPORT_PENALTY = True  # late tax on planted-both-feet while a run is commanded

RUN_FOOT_TARGET_HEIGHT = 0.035  # m, swing peak (velocity: 0.02)
RUN_AIR_TIME_WINDOW = (0.08, 0.30)  # s per foot (velocity: 0.125–0.30)
# 2026-10 eval of the first run (model_49999): flight 40% and alternation were fine, but
# step frequency sat at ~5 touchdowns/s for every command and step length at 0.12–0.13 m,
# so speed plateaued at ~0.7 (alive-only) / ≤1.1 m/s. Cap was 0.08 → raised so a longer
# stride keeps paying.
RUN_REACH_CAP = 0.14  # m the swing foot may lead the other foot before no extra pay
RUN_MAX_FLIGHT_S = 0.12  # s of each flight that pays (rate limit, no leap jackpot)
RUN_VEL_GATE_REF = 0.4  # m/s forward speed at which the stride-form rewards reach full pay
RUN_TURN_IN_PLACE_FRACTION = 0.05  # velocity: 0.15 — a run task spends its budget running
RUN_FORWARD_ENV_FRACTION = 0.5  # envs with forward-only commands (|vx| ≥ 0.3, vy = wz = 0)

# Forward command range, widened in stages. The last stage is the target top speed
# (~4 body-heights/s for a 25 cm robot). Backward stays small: this is a run task.
RUN_SPEED_STAGES = (
    {"step": 0, "lin_vel_x": (-0.3, 0.6)},
    {"step": 500 * NUM_STEPS_PER_ENV, "lin_vel_x": (-0.3, 0.8)},
    {"step": 1000 * NUM_STEPS_PER_ENV, "lin_vel_x": (-0.3, 1.0)},
    {"step": 1750 * NUM_STEPS_PER_ENV, "lin_vel_x": (-0.3, 1.2)},
    {"step": 2500 * NUM_STEPS_PER_ENV, "lin_vel_x": (-0.3, 1.5)},
)

# MICRODUCK_WARM_START=1: restart from a run checkpoint with the step counter at 0,
# so every inherited curriculum is pinned at its final stage (see AGENTS.md).
WARM_START = os.environ.get("MICRODUCK_WARM_START", "") not in ("", "0")


def make_microduck_run_env_cfg(play: bool = False, rough: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_microduck_velocity_env_cfg(play=play, rough=rough)
    feet = cfg.scene.sensors[0].name  # "feet_ground_contact" (LEFT, RIGHT)
    foot_sites = SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))

    # === Commands: forward-heavy, little lateral/yaw ===
    twist = cfg.commands["twist"]
    twist.ranges.lin_vel_x = RUN_SPEED_STAGES[0]["lin_vel_x"]
    twist.ranges.lin_vel_y = (-0.15, 0.15)
    twist.ranges.ang_vel_z = (-0.8, 0.8)
    twist.rel_forward_envs = RUN_FORWARD_ENV_FRACTION
    twist.rel_turn_in_place_envs = RUN_TURN_IN_PLACE_FRACTION

    cfg.curriculum["run_speed"] = CurriculumTermCfg(
        func=microduck_mdp.run_speed_curriculum,
        params={"command_name": "twist", "stages": list(RUN_SPEED_STAGES)},
    )

    # === Re-tune the walking recipe for a run ===
    # Wider leg pose tolerance once running (speed ≥ 0.5): reaching legs and a deep
    # push-off are NOT deviations to be corrected. Walking/standing stay tight.
    pose = cfg.rewards["pose"].params
    pose["std_walking"] = {
        r".*hip_yaw.*": 0.3, r".*hip_roll.*": 0.08, r".*hip_pitch.*": 0.5,
        r".*knee.*": 0.5, r".*ankle.*": 0.3,
    }
    pose["std_running"] = {
        r".*hip_yaw.*": 0.35, r".*hip_roll.*": 0.12, r".*hip_pitch.*": 0.9,
        r".*knee.*": 0.9, r".*ankle.*": 0.5,
    }
    pose["running_threshold"] = 0.5

    cfg.rewards["air_time"].weight = 2.0
    cfg.rewards["air_time"].params["threshold_min"] = RUN_AIR_TIME_WINDOW[0]
    cfg.rewards["air_time"].params["threshold_max"] = RUN_AIR_TIME_WINDOW[1]
    cfg.rewards["foot_clearance"].params["target_height"] = RUN_FOOT_TARGET_HEIGHT
    cfg.rewards["foot_swing_height"].params["target_height"] = RUN_FOOT_TARGET_HEIGHT

    # Speed matters here: heavier, slightly tighter linear tracking.
    cfg.rewards["track_linear_velocity"].weight = 3.0
    cfg.rewards["track_linear_velocity"].params["std"] = math.sqrt(0.0625)  # 0.25 m/s (was 0.3)

    # Motion-blockers LOW (a run physically needs trunk pitch/roll swing and angular
    # momentum); upright softened from the walk's 2.0 / std²=0.05 (a running trunk
    # leans forward by design).
    cfg.rewards["upright"].weight = 1.0
    cfg.rewards["upright"].params["std"] = math.sqrt(0.12)
    cfg.rewards["body_ang_vel"].weight = -0.02
    cfg.rewards["angular_momentum"].weight = -0.005

    # The head sways with the gait (below) and the commanded head pose is a side
    # input here: keep the tracking term but at a quarter weight, and cap the
    # commanded range at the walk curriculum's 3rd stage so a big commanded head
    # pose cannot dominate the run.
    cfg.rewards["head_pose_tracking"].weight = 0.5
    cfg.curriculum["head_pose_range"].params["range_stages"] = [
        s for s in cfg.curriculum["head_pose_range"].params["range_stages"] if s["step"] <= 1000 * 24
    ]

    # Action smoothness ends at -0.5 (walk: -1.0) — the push-off is a fast action.
    cfg.curriculum["action_rate_weight"].params["weight_stages"] = [
        {"step": 0, "weight": -0.1},
        {"step": 750 * NUM_STEPS_PER_ENV, "weight": -0.2},
        {"step": 1250 * NUM_STEPS_PER_ENV, "weight": -0.3},
        {"step": 1750 * NUM_STEPS_PER_ENV, "weight": -0.5},
    ]

    # === Run rewards (all positive-weight, see mdp.py "Run" section) ===
    gate_kw = {"vel_gate_ref": RUN_VEL_GATE_REF}
    cfg.rewards["run_flight"] = RewardTermCfg(
        func=microduck_mdp.run_flight_reward,
        weight=2.0,
        params={"sensor_name": feet, "max_flight_s": RUN_MAX_FLIGHT_S, **gate_kw},
    )
    cfg.rewards["run_alternating_step"] = RewardTermCfg(
        func=microduck_mdp.run_alternating_step_reward,
        weight=1.5,
        params={"sensor_name": feet, **gate_kw},
    )
    cfg.rewards["run_foot_reach"] = RewardTermCfg(
        func=microduck_mdp.run_foot_reach_reward,
        weight=1.0,
        params={"sensor_name": feet, "asset_cfg": foot_sites, "reach_cap": RUN_REACH_CAP, **gate_kw},
    )
    cfg.rewards["run_pushoff"] = RewardTermCfg(
        func=microduck_mdp.run_pushoff_reward,
        weight=1.0,
        params={"sensor_name": feet, **gate_kw},
    )

    # Late additions: weight 0 now, ramped by the curricula below (weights at step 0
    # MUST match these). A tax active while the gait is being discovered makes
    # "walk slowly" win; a sway reward before a gait exists pays noise.
    if ENABLE_DOUBLE_SUPPORT_PENALTY:
        cfg.rewards["run_double_support"] = RewardTermCfg(
            func=microduck_mdp.run_double_support_penalty,
            weight=0.0,  # self-negating (≤ 0) → POSITIVE weight
            params={"sensor_name": feet},
        )
        cfg.curriculum["run_double_support_weight"] = CurriculumTermCfg(
            func=microduck_mdp.reward_weight,
            params={
                "reward_name": "run_double_support",
                "weight_stages": [
                    {"step": 0, "weight": 0.0},
                    {"step": 800 * NUM_STEPS_PER_ENV, "weight": 0.25},
                    {"step": 1500 * NUM_STEPS_PER_ENV, "weight": 0.5},
                ],
            },
        )
    if ENABLE_HEAD_SWAY:
        cfg.rewards["run_head_sway"] = RewardTermCfg(
            func=microduck_mdp.run_head_sway_reward,
            weight=0.0,
            params={"sensor_name": feet, **gate_kw},
        )
        cfg.curriculum["run_head_sway_weight"] = CurriculumTermCfg(
            func=microduck_mdp.reward_weight,
            params={
                "reward_name": "run_head_sway",
                "weight_stages": [
                    {"step": 0, "weight": 0.0},
                    {"step": 1000 * NUM_STEPS_PER_ENV, "weight": 0.5},
                    {"step": 1750 * NUM_STEPS_PER_ENV, "weight": 1.0},
                ],
            },
        )

    if WARM_START and not play:
        _collapse_curricula_to_final(cfg)

    return cfg


MicroduckRunRlCfg = dataclasses.replace(
    MicroduckRlCfg,
    experiment_name="run",
    run_name="run",
)

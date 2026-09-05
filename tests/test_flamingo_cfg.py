import math

import numpy as np

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks import microduck_flamingo_env_cfg as fl
from mjlab_microduck.tasks.microduck_flamingo_env_cfg import (
    make_microduck_flamingo_env_cfg,
    MicroduckFlamingoRlCfg,
)


def test_pose_has_14_servo_values_and_keeps_a_joint_limit_margin():
    assert len(fl.FLAMINGO_POSE) == 14
    # Ranges from robot_allcollisions.xml (rad). Every joint ≥ 0.05 rad from its limit:
    # parking on a hard limit is a sim2real trap (AGENTS.md).
    ranges = {
        1: (-0.384, 0.384), 10: (-0.384, 0.384),          # hip_roll
        0: (-0.436, 0.524), 9: (-0.436, 0.524),           # hip_yaw
        2: (-1.571, 1.571), 3: (-1.571, 1.571), 4: (-1.571, 1.571),
        11: (-1.571, 1.571), 12: (-1.571, 1.571), 13: (-1.571, 1.571),
        5: (-1.571, 1.047), 6: (-1.571, 1.571), 7: (-2.967, 2.967), 8: (-0.436, 0.436),
    }
    for i, (lo, hi) in ranges.items():
        q = fl.FLAMINGO_POSE[i]
        assert min(q - lo, hi - q) >= 0.05 - 1e-9, f"joint {i} too close to its limit"


def test_stance_is_the_right_foot():
    assert fl.STANCE_SLOT == 1 and fl.SWING_SLOT == 0
    assert fl.STANCE_SITE == "right_foot" and fl.SWING_SITE == "left_foot"
    # the trunk leans toward the stance (−y) side: g_b,y < 0 at the pose
    assert fl.FLAMINGO_GRAVITY_B[1] < -0.3
    assert abs(np.linalg.norm(fl.FLAMINGO_GRAVITY_B) - 1.0) < 0.01


def test_cfg_has_the_balance_rewards_with_the_intended_signs():
    cfg = make_microduck_flamingo_env_cfg()
    r = cfg.rewards
    for name in ("com_over_stance_foot", "stance_foot_grounded", "swing_foot_clear",
                 "pose_flamingo", "gravity_flamingo", "stillness"):
        assert name in r, name
        assert r[name].weight > 0.0, name
    # self-negating penalties (functions return ≤ 0) carry POSITIVE weights
    assert r["swing_foot_touch"].func is microduck_mdp.foot_contact_penalty and r["swing_foot_touch"].weight > 0
    assert r["stance_side_tilt"].func is microduck_mdp.lateral_tilt_penalty and r["stance_side_tilt"].weight > 0
    # positive-cost terms carry NEGATIVE weights
    assert r["joint_limit_proximity"].weight < 0
    assert r["action_rate_l2"].weight < 0
    assert r["body_ang_vel"].weight < 0
    assert r["self_collisions"].weight < 0
    # the walking stack is gone
    for gone in ("track_linear_velocity", "air_time", "foot_slip", "pose", "upright", "head_pose_tracking"):
        assert gone not in r, gone


def test_swing_and_stance_slots_are_wired_consistently():
    cfg = make_microduck_flamingo_env_cfg()
    r = cfg.rewards
    assert r["stance_foot_grounded"].params["slot"] == fl.STANCE_SLOT
    assert r["swing_foot_touch"].params["slot"] == fl.SWING_SLOT
    assert r["swing_foot_clear"].params["gate_slot"] == fl.STANCE_SLOT
    assert r["swing_foot_clear"].params["asset_cfg"].site_names == [fl.SWING_SITE]
    assert r["com_over_stance_foot"].params["asset_cfg"].site_names == [fl.STANCE_SITE]
    # stance-side tilt tax fires for g_b,y more negative than the threshold
    assert r["stance_side_tilt"].params["direction"] == -1.0
    assert r["stance_side_tilt"].params["threshold"] > -fl.FLAMINGO_GRAVITY_B[1]


def test_spawn_is_in_the_pose_and_pushes_start_at_zero():
    cfg = make_microduck_flamingo_env_cfg()
    ev = cfg.events["set_flamingo_state"]
    assert ev.mode == "reset"
    assert ev.params["joint_pose"] == fl.FLAMINGO_POSE
    assert abs(ev.params["base_roll"] - math.radians(22.6)) < 1e-6
    assert 0.11 < ev.params["z_min"] <= ev.params["z_max"] < 0.14
    assert ev.params["standing_prob"] == 0.0
    push = cfg.events["push_robot"]
    assert push.params["velocity_range"] == {"x": (0.0, 0.0), "y": (0.0, 0.0)}
    stages = cfg.curriculum["push_magnitude"].params["push_stages"]
    mags = [abs(s["velocity_range"]["x"][1]) for s in stages]
    assert mags == sorted(mags) and mags[0] == 0.0 and mags[-1] == 0.25


def test_terminations_keep_fell_over_and_add_nan_guard():
    cfg = make_microduck_flamingo_env_cfg()
    assert "fell_over" in cfg.terminations
    assert "nan_state" in cfg.terminations
    assert cfg.episode_length_s == 6.0


def test_actor_observation_keeps_the_61d_slot_layout():
    cfg = make_microduck_flamingo_env_cfg()
    terms = cfg.observations["actor"].terms
    assert "base_lin_vel" not in terms
    assert "height_scan" not in terms
    assert terms["head_command"].params["command_name"] == "head_pose"
    assert terms["body_command"].params["dim"] == 6
    assert "body_pose" not in cfg.commands


def test_obs_parity_with_standup():
    from mjlab_microduck.tasks.microduck_standup_env_cfg import make_microduck_standup_env_cfg
    a = make_microduck_flamingo_env_cfg()
    b = make_microduck_standup_env_cfg()
    for grp in ("actor", "critic"):
        assert list(a.observations[grp].terms.keys()) == list(b.observations[grp].terms.keys()), grp


def test_runner_cfg():
    assert MicroduckFlamingoRlCfg.algorithm.symmetry_cfg is None
    assert MicroduckFlamingoRlCfg.experiment_name == "flamingo"
    assert MicroduckFlamingoRlCfg.actor.obs_normalization is True

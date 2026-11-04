"""Swizzle head-free variant: swizzle recipe + externally-driven head.

The head-free twin must keep every swizzle reward and curriculum, and swap the
policy-managed head control for the head override — never both at once (a
head_pose_tracking reward on a head the policy cannot move is pure noise).
"""

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_swizzle_env_cfg import (
    SWIZZLE_HEAD_RANGE_STAGES,
    make_microduck_velocity_swizzle_env_cfg,
)


def _headfree(play=False):
    return make_microduck_velocity_swizzle_env_cfg(play=play, head_override=True)


def test_swizzle_recipe_kept():
    base = make_microduck_velocity_swizzle_env_cfg()
    cfg = _headfree()
    for name in ("leg_symmetry", "grounded", "heading_tracking", "wheel_speed", "heading_hold"):
        assert name in cfg.rewards
        assert cfg.rewards[name].weight == base.rewards[name].weight
    assert cfg.rewards["wheel_speed"].params["bidirectional"] is True
    assert cfg.commands["twist"].ranges.lin_vel_x == base.commands["twist"].ranges.lin_vel_x
    assert cfg.commands["twist"].ranges.ang_vel_z == base.commands["twist"].ranges.ang_vel_z
    for cur in ("heading_hold_weight", "heading_tracking_weight"):
        assert cur in cfg.curriculum


def test_policy_head_control_replaced_by_override():
    cfg = _headfree()
    assert isinstance(cfg.actions["joint_pos"], microduck_mdp.HeadOverrideJointPositionActionCfg)
    assert "head_pose_tracking" not in cfg.rewards
    assert "head_pose_tracking_weight" not in cfg.curriculum
    assert "neck_joint_pos_l2" not in cfg.rewards
    assert "neck_action_rate_l2" not in cfg.rewards
    for group in ("actor", "critic"):
        assert cfg.observations[group].terms["head_command"].func is microduck_mdp.head_override_target


def test_base_swizzle_keeps_policy_head_control():
    cfg = make_microduck_velocity_swizzle_env_cfg()
    assert type(cfg.actions["joint_pos"]).__name__ == "JointPositionActionCfg"
    assert "head_pose_tracking" in cfg.rewards


def test_head_range_schedule_is_the_swizzle_one():
    cfg = _headfree()
    stages = cfg.curriculum["head_pose_range"].params["range_stages"]
    assert [s["step"] for s in stages] == [it * 24 for it, _ in SWIZZLE_HEAD_RANGE_STAGES]
    # held tiny until the swizzle and heading curricula have had their window
    assert stages[1]["step"] >= 1500 * 24
    assert cfg.commands["head_pose"].ranges == SWIZZLE_HEAD_RANGE_STAGES[0][1]
    assert cfg.commands["head_pose"].zero_command_prob > 0.0


def test_play_full_head_range():
    cfg = _headfree(play=True)
    assert cfg.commands["head_pose"].ranges == SWIZZLE_HEAD_RANGE_STAGES[-1][1]
    assert "head_pose_range" not in cfg.curriculum


def test_obs_order_unchanged():
    base = list(make_microduck_velocity_swizzle_env_cfg().observations["actor"].terms)
    assert list(_headfree().observations["actor"].terms) == base

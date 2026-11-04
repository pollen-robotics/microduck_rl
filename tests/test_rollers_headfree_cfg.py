"""Head-free roller variant: the head is driven externally, the policy only skates.

Locks the contract the runtime must reproduce: neck targets come from a slewed
head command (policy neck outputs ignored), the actor sees that applied target
in the head_pose slot, and no reward prices the head.
"""

from types import SimpleNamespace

import torch

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_rollers_env_cfg import (
    HEAD_RANGE_STAGES,
    HEAD_SLEW_RAD_S,
    make_microduck_velocity_rollers_env_cfg,
)


def _headfree(play=False):
    return make_microduck_velocity_rollers_env_cfg(play=play, head_override=True)


def test_base_task_untouched():
    cfg = make_microduck_velocity_rollers_env_cfg()
    assert type(cfg.actions["joint_pos"]).__name__ == "JointPositionActionCfg"
    assert "head_pose" not in cfg.commands
    assert cfg.observations["actor"].terms["head_command"].func is microduck_mdp.zero_command_padding
    assert "neck_joint_pos_l2" in cfg.rewards


def test_action_is_head_override_with_same_scale():
    base = make_microduck_velocity_rollers_env_cfg().actions["joint_pos"]
    act = _headfree().actions["joint_pos"]
    assert isinstance(act, microduck_mdp.HeadOverrideJointPositionActionCfg)
    assert act.command_name == "head_pose"
    assert act.slew_rate == HEAD_SLEW_RAD_S
    assert act.scale == base.scale
    assert tuple(act.actuator_names) == tuple(base.actuator_names)


def test_head_obs_slot_reads_applied_target():
    cfg = _headfree()
    for group in ("actor", "critic"):
        term = cfg.observations[group].terms["head_command"]
        assert term.func is microduck_mdp.head_override_target
    # body slot stays zero-padded (61D contract)
    assert cfg.observations["actor"].terms["body_command"].func is microduck_mdp.zero_command_padding


def test_obs_order_unchanged():
    base = list(make_microduck_velocity_rollers_env_cfg().observations["actor"].terms)
    assert list(_headfree().observations["actor"].terms) == base


def test_no_reward_prices_the_head():
    cfg = _headfree()
    assert "neck_joint_pos_l2" not in cfg.rewards
    assert "neck_action_rate_l2" not in cfg.rewards
    assert not any("head_pose" in n for n in cfg.rewards)
    for key in ("std_standing", "std_walking", "std_running"):
        std = cfg.rewards["pose"].params[key]
        assert std[r".*neck.*"] >= 100.0 and std[r".*head.*"] >= 100.0
        assert std[r".*knee.*"] < 1.0  # legs still priced


def test_head_command_alive_from_step_zero_and_idle_trained():
    cfg = _headfree()
    cmd = cfg.commands["head_pose"]
    assert all(hi > 0.0 and lo < 0.0 for lo, hi in cmd.ranges)
    assert cmd.zero_command_prob > 0.0
    stages = cfg.curriculum["head_pose_range"].params["range_stages"]
    steps = [s["step"] for s in stages]
    assert steps == sorted(steps) and steps[0] == 0
    # ranges only widen
    for a, b in zip(stages, stages[1:]):
        assert all(bh >= ah for (_, ah), (_, bh) in zip(a["ranges"], b["ranges"]))


def test_play_uses_full_range_without_curriculum():
    cfg = _headfree(play=True)
    assert cfg.commands["head_pose"].ranges == HEAD_RANGE_STAGES[-1][1]
    assert "head_pose_range" not in cfg.curriculum


def _fake_action(cmd, n_envs=2, max_step=0.04):
    """HeadOverrideJointPositionAction with its tensors set by hand (no sim)."""
    a = microduck_mdp.HeadOverrideJointPositionAction.__new__(
        microduck_mdp.HeadOverrideJointPositionAction
    )
    a.cfg = SimpleNamespace(clip=None, command_name="head_pose")
    a._raw_actions = torch.zeros(n_envs, 14)
    a._scale = 1.0
    a._offset = torch.zeros(n_envs, 14)
    a._neck_cols = torch.tensor([5, 6, 7, 8])
    a._neck_home = torch.full((n_envs, 4), 0.3)
    a._head_target = torch.zeros(n_envs, 4)
    a._max_step = max_step
    a._env = SimpleNamespace(
        command_manager=SimpleNamespace(get_command=lambda name: cmd)
    )
    return a


def test_override_ignores_policy_neck_and_slews():
    cmd = torch.tensor([[1.0, -1.0, 0.01, 0.0], [0.0, 0.0, 0.0, 0.0]])
    a = _fake_action(cmd)
    policy = torch.full((2, 14), 7.0)  # absurd neck outputs must not leak through
    a.process_actions(policy)
    neck = a._processed_actions[:, a._neck_cols]
    expected0 = torch.tensor([0.34, 0.26, 0.31, 0.30])  # home 0.3 + slew-clamped delta
    assert torch.allclose(neck[0], expected0, atol=1e-6)
    assert torch.allclose(neck[1], torch.full((4,), 0.3))
    # legs pass through untouched
    legs = [c for c in range(14) if c not in (5, 6, 7, 8)]
    assert torch.all(a._processed_actions[:, legs] == 7.0)
    # raw action (→ last_action obs) is the policy's own output
    assert torch.all(a.raw_action == 7.0)
    # converges to the command, never overshoots
    for _ in range(100):
        a.process_actions(policy)
    assert torch.allclose(a.head_target, cmd)


def test_reset_returns_head_target_home():
    cmd = torch.ones(2, 4)
    a = _fake_action(cmd)
    a.process_actions(torch.zeros(2, 14))
    a.reset(torch.tensor([0]))
    assert torch.all(a.head_target[0] == 0.0)
    assert torch.all(a.head_target[1] > 0.0)

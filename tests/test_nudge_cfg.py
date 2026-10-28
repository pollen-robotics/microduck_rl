"""Precision nudge: the cfg must match robotd's one-shot skill contract and price the right things."""
import math

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks import microduck_nudge_env_cfg as nudge
from mjlab_microduck.tasks.microduck_nudge_env_cfg import make_microduck_nudge_env_cfg


def test_command_is_the_skill_timeline():
    cfg = make_microduck_nudge_env_cfg()
    cmd = cfg.commands["twist"]
    assert isinstance(cmd, microduck_mdp.NudgeCommandCfg)
    # robotd skill: `duration` = go_s with the constant command, then `unwind` zeros for unwind_s
    assert cmd.go_s == nudge.GO_S
    assert math.isclose(cfg.episode_length_s, nudge.PRE_S + nudge.GO_S + nudge.SETTLE_S)
    # never resampled mid-episode, never a standing or heading env
    assert cmd.resampling_time_range[0] > cfg.episode_length_s
    assert cmd.rel_standing_envs == 0.0 and cmd.rel_heading_envs == 0.0
    # one pure axis per episode, small moves, some stand-still episodes
    assert len(cmd.axis_probs) == 3 and math.isclose(sum(cmd.axis_probs), 1.0)
    lo, hi = cmd.dist_range_m
    assert 0.0 < lo < hi <= 0.04                           # centimetres, not a walk
    lo, hi = cmd.yaw_range_rad
    assert 0.0 < lo < hi <= math.radians(20.0)
    assert 0.0 < cmd.zero_prob < 0.5


def test_typical_requests_read_as_speeds_where_the_walking_policy_steps():
    # run 1: 1.5 cm read as 0.15 m/s, inside velstand's dead zone, and the policy never stepped. A typical request
    # must read as ~0.3 m/s / ~0.5 rad/s, and the largest stay near velstand's trained range (0.4 m/s, 1 rad/s).
    cfg = make_microduck_nudge_env_cfg()
    cmd = cfg.commands["twist"]
    assert 0.25 <= 0.015 / microduck_mdp.NUDGE_UNIT_M <= 0.4
    assert cmd.dist_range_m[1] / microduck_mdp.NUDGE_UNIT_M <= 0.8
    assert cmd.yaw_range_rad[1] / microduck_mdp.NUDGE_UNIT_RAD <= 1.1


def test_task_terms_dominate_and_smoothness_comes_later():
    cfg = make_microduck_nudge_env_cfg()
    r = cfg.rewards
    assert r["nudge_hold"].weight >= 10 * r["upright"].weight / 2
    assert r["nudge_miss"].weight < 0.0 and r["nudge_lift"].weight > 0.0
    stages = cfg.curriculum["action_rate_weight"].params["weight_stages"]
    assert stages[0]["step"] == 0 and stages[0]["weight"] > stages[-1]["weight"]
    assert r["action_rate_l2"].weight == stages[0]["weight"]


def test_walking_terms_are_gone_and_slip_is_always_priced():
    cfg = make_microduck_nudge_env_cfg()
    for name in nudge.DROP_REWARDS:
        assert name not in cfg.rewards, name
    slip = cfg.rewards["nudge_slip"]
    assert slip.weight < 0.0
    assert slip.params["command_threshold"] < 0.0          # active at zero command too


def test_reward_signs():
    cfg = make_microduck_nudge_env_cfg()
    for name in ("nudge_progress", "nudge_hold", "nudge_hold_fine", "nudge_feet_down"):
        assert cfg.rewards[name].weight > 0.0, name
    for name in ("nudge_still", "termination"):
        assert cfg.rewards[name].weight < 0.0, name
    # the hold terms only pay once the move should be over (after the go window)
    assert cfg.rewards["nudge_hold"].params["after_s"] > nudge.GO_S
    assert cfg.rewards["nudge_hold_fine"].params["std"] < cfg.rewards["nudge_hold"].params["std"]


def test_no_pushes_and_standing_spawns():
    cfg = make_microduck_nudge_env_cfg()
    assert "push_robot" not in cfg.events
    z = cfg.events["reset_base"].params["pose_range"]["z"]
    assert 0.11 < z[0] < z[1] < 0.125


def test_actor_obs_contract_is_untouched():
    cfg = make_microduck_nudge_env_cfg()
    actor = list(cfg.observations["actor"].terms)
    assert actor[-3:] == ["command", "head_command", "body_command"]
    assert "nudge_state" not in actor and "nudge_state" in cfg.observations["critic"].terms


def test_runtime_layer_variant_ends_on_robotds_defaults():
    cfg = make_microduck_nudge_env_cfg(runtime_layer=True)
    act = cfg.actions["joint_pos"]
    assert isinstance(act, microduck_mdp.RuntimeLayerJointPositionActionCfg)
    stages = cfg.curriculum["runtime_layer"].params["stages"]
    first, last = stages[0], stages[-1]
    assert (first["leg_alpha"], first["scale_lo"]) == (1.0, 1.0)             # starts as the plain action
    assert (last["leg_alpha"], last["head_alpha"]) == (0.7, 0.5)             # robotd legs_lowpass / head_lowpass
    assert last["scale_lo"] <= 0.9 * 7.4 / 8.4 + 0.01 and last["scale_hi"] >= 0.9 * 7.4 / 6.8 - 0.01
    assert cfg.rewards["action_rate_l2"].weight == nudge.RUNTIME_ACTION_RATE_STAGES[0][1]
    assert stages[1]["step"] > 0                                               # plain action first: discovery
    ar = cfg.curriculum["action_rate_weight"].params["weight_stages"]
    assert ar[-1]["weight"] == -1.0 and ar[0]["weight"] > ar[-1]["weight"]
    play = make_microduck_nudge_env_cfg(play=True, runtime_layer=True).actions["joint_pos"]
    assert (play.leg_alpha, play.head_alpha) == (0.7, 0.5)


def test_episode_ends_in_the_standing_pose():
    cfg = make_microduck_nudge_env_cfg()
    end = cfg.rewards["nudge_end_pose"]
    assert end.weight >= cfg.rewards["nudge_hold"].weight          # as valuable as being on target
    assert end.params["after_s"] > nudge.GO_S and end.params["std"] <= 0.15


def test_runtime_variant_puts_the_walking_policy_around_the_skill():
    cfg = make_microduck_nudge_env_cfg(runtime_layer=True)
    act = cfg.actions["joint_pos"]
    assert act.handover_policy == ("pollen-robotics/microduck-policies", "velstand.onnx", "v8")
    cmd = cfg.commands["twist"]
    assert cmd.pre_s == nudge.HANDOVER_PRE_S and cmd.unwind_s == nudge.SETTLE_S
    assert math.isclose(cfg.episode_length_s, nudge.HANDOVER_PRE_S + nudge.GO_S + nudge.SETTLE_S + nudge.HANDOVER_POST_S)
    # the hold terms keep scoring after the hand-back; the end pose and stillness stop at the skill's end
    assert "until_s" not in cfg.rewards["nudge_hold"].params
    assert math.isclose(cfg.rewards["nudge_end_pose"].params["until_s"], nudge.GO_S + nudge.SETTLE_S)


def test_discarded_outputs_are_priced_and_entropy_is_halved():
    cfg = make_microduck_nudge_env_cfg(runtime_layer=True)
    assert cfg.rewards["handover_imitation"].weight < 0.0
    assert nudge.MicroduckNudgeRuntimeRlCfg.algorithm.entropy_coef < nudge.MicroduckNudgeRlCfg.algorithm.entropy_coef


def test_safe_variant_makes_falls_expensive():
    cfg = make_microduck_nudge_env_cfg(runtime_layer=True, safe=True)
    assert cfg.rewards["termination"].weight <= -100.0
    assert cfg.terminations["fell_over"].params["limit_angle"] <= math.radians(45.0)
    assert cfg.rewards["nudge_target_overrun"].weight < 0.0
    assert cfg.rewards["nudge_tilt"].weight < 0.0 and 0.0 < cfg.rewards["nudge_tilt"].params["free_deg"] <= 12.0
    si = cfg.rewards["nudge_settle_imitation"]
    assert si.weight < 0.0 and nudge.GO_S < si.params["after_s"] < si.params["until_s"] <= nudge.GO_S + nudge.SETTLE_S
    assert cfg.actions["joint_pos"].clamp_margin == nudge.SAFE_CLAMP_MARGIN
    assert cfg.curriculum["action_rate_weight"].params["weight_stages"][-1]["weight"] == -0.5


def test_safe_fine_tune_has_no_entropy_bonus():
    assert nudge.MicroduckNudgeRuntimeSafeRlCfg.algorithm.entropy_coef == 0.0

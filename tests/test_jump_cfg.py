"""Cfg invariants for Mjlab-Jump-Flat-MicroDuck.

CPU only — no GPU, no env construction. Locks the 61D contract, hop-vs-sit
gates, reward signs, and spawn mix.
"""

import torch

from mjlab_microduck.robot.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_jump_env_cfg import (
    JUMP_MIN_AIR_S,
    JUMP_MIN_PEAK_ABOVE,
    JUMP_TARGET_ABOVE,
    STAND_Z,
    MicroduckJumpRlCfg,
    make_microduck_jump_env_cfg,
)
from mjlab_microduck.tasks.mdp import (
    hopping_from_values,
    jump_flight,
    jump_head_up,
    jump_landing,
    jump_landing_score_from_values,
    jump_latch_from_values,
    jump_progress,
    jump_progress_from_values,
    jump_takeoff_vz,
    jump_unloading,
    reset_jump_state,
    trunk_vertical_accel_penalty,
)


_SIT_Z = 0.060


def test_jump_task_rewards():
    cfg = make_microduck_jump_env_cfg()
    r = cfg.rewards
    assert r["jump_progress"].func is jump_progress
    assert r["jump_progress"].weight == 8.0
    assert r["jump_progress"].params["stand_z"] == STAND_Z
    assert r["jump_progress"].params["target_above"] == JUMP_TARGET_ABOVE
    assert r["jump_flight"].func is jump_flight
    assert r["jump_takeoff"].func is jump_takeoff_vz
    assert r["jump_unloading"].func is jump_unloading
    assert r["jump_landing"].func is jump_landing
    assert r["jump_head_up"].func is jump_head_up
    assert r["gentle_landing"].func is trunk_vertical_accel_penalty
    assert "jump_success" not in r
    for name in (
        "track_linear_velocity",
        "track_angular_velocity",
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "pose",
        "upright",
    ):
        assert name not in r, name


def test_self_negating_penalties_use_positive_weights():
    cfg = make_microduck_jump_env_cfg()
    r = cfg.rewards
    assert r["gentle_landing"].weight > 0
    for name in ("action_rate_l2", "body_ang_vel", "angular_momentum", "self_collisions"):
        assert r[name].weight < 0, name


def test_jump_obs_slots_padded():
    cfg = make_microduck_jump_env_cfg()
    for grp in ("actor", "critic"):
        terms = cfg.observations[grp].terms
        assert "command" in terms
        assert "head_command" in terms
        assert "body_command" in terms
        assert terms["head_command"].params["dim"] == 4
        assert terms["body_command"].params["dim"] == 6
        assert terms["head_command"].func is microduck_mdp.zero_command_padding
        assert terms["body_command"].func is microduck_mdp.zero_command_padding


def test_twist_command_is_neutralised():
    cfg = make_microduck_jump_env_cfg()
    cmd = cfg.commands["twist"]
    assert isinstance(cmd, microduck_mdp.VelocityCommandCommandOnlyCfg)
    assert cmd.ranges.lin_vel_x == (-0.01, 0.01)
    assert cmd.heading_command is False
    assert cmd.rel_standing_envs == 0.0


def test_jump_reset_mix():
    cfg = make_microduck_jump_env_cfg()
    assert cfg.events["set_jump_state"].func is reset_jump_state
    p = cfg.events["set_jump_state"].params
    assert abs(p["standing_prob"] + p["crouch_prob"] + p["air_prob"] - 1.0) < 1e-6
    assert p["standing_prob"] > p["air_prob"]
    assert p["stand_z"] == STAND_Z
    assert p["min_air_s"] == JUMP_MIN_AIR_S


def test_jump_event_runs_after_base_reset():
    cfg = make_microduck_jump_env_cfg()
    order = list(cfg.events.keys())
    assert order.index("set_jump_state") > order.index("reset_base")
    assert order.index("set_jump_state") > order.index("reset_robot_joints")


def test_no_fall_termination_and_no_push():
    cfg = make_microduck_jump_env_cfg()
    assert "fell_over" not in cfg.terminations
    assert "nan_state" in cfg.terminations
    assert "push_robot" not in cfg.events
    assert "expand_bam_friction_fields" in cfg.events


def test_uses_allcollisions_robot():
    cfg = make_microduck_jump_env_cfg()
    assert cfg.scene.entities["robot"] is MICRODUCK_STANDUP_ROBOT_CFG
    names = [s.name for s in cfg.scene.sensors]
    assert "feet_ground_contact" in names
    assert "self_collision" in names


def test_jump_play_and_curriculum():
    play = make_microduck_jump_env_cfg(play=True)
    assert play.episode_length_s == 3.5
    p = play.events["set_jump_state"].params
    assert abs(p["standing_prob"] + p["crouch_prob"] + p["air_prob"] - 1.0) < 1e-6
    assert "jump_spawn_mix" not in play.curriculum
    train = make_microduck_jump_env_cfg()
    assert "jump_spawn_mix" in train.curriculum
    stages = train.curriculum["jump_spawn_mix"].params["param_stages"]
    stand = [s["params"]["standing_prob"] for s in stages]
    assert stand == sorted(stand)
    for stage in stages:
        p = stage["params"]
        assert abs(p["standing_prob"] + p["crouch_prob"] + p["air_prob"] - 1.0) < 1e-9


def test_task_is_registered_with_backlash_twin():
    from mjlab.tasks.registry import list_tasks

    import mjlab_microduck.tasks  # noqa: F401

    tasks = list_tasks()
    for task_id in (
        "Mjlab-Jump-Flat-MicroDuck",
        "Mjlab-Jump-Flat-Backlash-MicroDuck",
    ):
        assert task_id in tasks, task_id
    assert "Mjlab-Jump-Rough-MicroDuck" not in tasks


def test_runner_cfg():
    assert MicroduckJumpRlCfg.experiment_name == "microduck_jump"
    assert MicroduckJumpRlCfg.algorithm.symmetry_cfg is not None
    assert MicroduckJumpRlCfg.actor.obs_normalization is True


def test_trunk_asset_cfgs_are_distinct_objects():
    cfg = make_microduck_jump_env_cfg()
    names = (
        "jump_progress",
        "jump_flight",
        "jump_takeoff",
        "jump_unloading",
        "jump_landing",
        "upright_linear",
        "jump_head_up",
        "gentle_landing",
    )
    seen = [id(cfg.rewards[n].params["asset_cfg"]) for n in names]
    assert len(set(seen)) == len(seen), "asset_cfg shared across terms"


def test_sit_fall_is_not_a_hop():
    both_air = torch.tensor([True, True, True, False])
    upright = torch.tensor([0.95, 0.95, -0.2, 0.95])
    z = torch.tensor([STAND_Z + 0.01, _SIT_Z, STAND_Z + 0.02, STAND_Z + 0.02])
    hop = hopping_from_values(both_air, upright, z, STAND_Z)
    assert bool(hop[0])  # airborne, upright, above stand
    assert not bool(hop[1])  # sit
    assert not bool(hop[2])  # inverted
    assert not bool(hop[3])  # still in contact


def test_latch_needs_air_or_peak():
    flight = torch.tensor([0.0, JUMP_MIN_AIR_S, 0.0, 0.02])
    peak = torch.tensor([
        STAND_Z,
        STAND_Z,
        STAND_Z + JUMP_MIN_PEAK_ABOVE + 1e-6,
        STAND_Z + 0.001,
    ])
    latch = jump_latch_from_values(flight, peak, STAND_Z, JUMP_MIN_AIR_S, JUMP_MIN_PEAK_ABOVE)
    assert not bool(latch[0])
    assert bool(latch[1])
    assert bool(latch[2])
    assert not bool(latch[3])


def test_progress_is_potential_and_capped():
    stand = torch.tensor([STAND_Z, STAND_Z, STAND_Z, STAND_Z])
    # camping at stand
    camp = jump_progress_from_values(stand, stand, STAND_Z, JUMP_TARGET_ABOVE)
    assert float(camp[0]) == 0.0
    # +3 cm hop from stand, unpaid
    hop = jump_progress_from_values(
        torch.tensor([STAND_Z + JUMP_TARGET_ABOVE]),
        torch.tensor([STAND_Z]),
        STAND_Z,
        JUMP_TARGET_ABOVE,
    )
    assert abs(float(hop) - 1.0) < 1e-5
    # overshoot forfeits the excess
    huge = jump_progress_from_values(
        torch.tensor([STAND_Z + 0.10]),
        torch.tensor([STAND_Z]),
        STAND_Z,
        JUMP_TARGET_ABOVE,
    )
    assert abs(float(huge) - 1.0) < 1e-5
    # already-paid airborne spawn: no jackpot
    air = jump_progress_from_values(
        torch.tensor([STAND_Z + 0.02]),
        torch.tensor([STAND_Z + 0.02]),
        STAND_Z,
        JUMP_TARGET_ABOVE,
    )
    assert float(air) == 0.0


def test_landing_score_sit_and_air_collapse():
    down = torch.tensor([True])
    up = torch.tensor([False])
    stand = jump_landing_score_from_values(
        torch.tensor([STAND_Z]), torch.tensor([1.0]), down, STAND_Z
    )
    sit = jump_landing_score_from_values(
        torch.tensor([_SIT_Z]), torch.tensor([1.0]), down, STAND_Z
    )
    air = jump_landing_score_from_values(
        torch.tensor([STAND_Z + 0.03]), torch.tensor([1.0]), up, STAND_Z
    )
    fallen = jump_landing_score_from_values(
        torch.tensor([0.05]), torch.tensor([0.0]), down, STAND_Z
    )
    assert float(stand) > 0.99
    assert float(sit) < 0.2
    assert float(air) == 0.0
    assert float(fallen) < 0.05
    assert float(stand) > 5.0 * float(sit)

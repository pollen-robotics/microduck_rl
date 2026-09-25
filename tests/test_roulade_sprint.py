import math
from types import SimpleNamespace

import torch

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_roulade_sprint_env_cfg import (
    TIPPED_TERMINATION_STEPS,
    make_microduck_roulade_sprint_env_cfg,
)


def test_cfg_drops_the_single_roll_task():
    cfg = make_microduck_roulade_sprint_env_cfg()
    for name in ("roulade_progress", "roulade_overspeed", "roulade_landing_composite"):
        assert name not in cfg.rewards, name
    assert cfg.rewards["heading_progress"].weight > 0.0
    assert cfg.rewards["tipped_run"].weight < 0.0
    assert cfg.rewards["terminated"].weight < 0.0
    assert cfg.rewards["heading_error"].weight < 0.0
    assert cfg.rewards["lateral_offset"].weight < 0.0


def test_spawn_position_is_recorded_after_the_root_is_placed():
    events = list(make_microduck_roulade_sprint_env_cfg().events)
    assert events.index("record_spawn_position") > events.index("reset_base")
    assert events.index("record_spawn_position") > events.index("set_roulade_state")


def test_termination_keeps_a_margin_under_the_arena_fall():
    cfg = make_microduck_roulade_sprint_env_cfg()
    term = cfg.terminations["tipped_too_long"]
    assert term.time_out is False
    assert term.params["max_steps"] == TIPPED_TERMINATION_STEPS
    assert TIPPED_TERMINATION_STEPS < microduck_mdp.ARENA_FALL_HOLD_STEPS


def test_command_covers_the_arena_sweep():
    cfg = make_microduck_roulade_sprint_env_cfg()
    assert cfg.commands["twist"].ranges.lin_vel_x == (0.20, 1.20)


def _fake_env(num_envs: int):
    asset = SimpleNamespace(data=SimpleNamespace(projected_gravity_b=torch.zeros(num_envs, 3)))
    env = SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        common_step_counter=0,
        scene={"robot": asset},
    )
    return env, asset


def _step(env, asset, g_z):
    env.common_step_counter += 1
    asset.data.projected_gravity_b[:, 2] = torch.tensor(g_z)
    return microduck_mdp.tipped_run_exceeded(env, max_steps=15)


def test_tipped_run_matches_the_arena_rule():
    # env 0 stays inverted; env 1 dips back inside 60° once, which resets it.
    env, asset = _fake_env(2)
    for i in range(14):
        assert not _step(env, asset, [1.0, 1.0 if i != 7 else -0.9]).any()
    done = _step(env, asset, [1.0, 1.0])
    assert done.tolist() == [True, False]
    # Reading the count twice in one step must not advance it.
    assert microduck_mdp.tipped_run_exceeded(env, max_steps=15).tolist() == [True, False]
    assert env._sprint_tipped.tolist() == [15, 7]


def test_reset_clears_the_run():
    env, asset = _fake_env(2)
    for _ in range(10):
        _step(env, asset, [1.0, 1.0])
    microduck_mdp.reset_tipped_run(env, torch.tensor([1]))
    assert env._sprint_tipped.tolist() == [10, 0]


def _pose_env(yaw, spawn_yaw, pos, spawn_xy):
    q = torch.tensor([[math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)] for a in yaw])
    data = SimpleNamespace(root_link_quat_w=q, root_link_pos_w=torch.tensor(pos))
    env = SimpleNamespace(
        num_envs=len(yaw),
        device="cpu",
        scene={"robot": SimpleNamespace(data=data)},
        _roulade_spawn_yaw=torch.tensor(spawn_yaw),
        _roulade_spawn_xy=torch.tensor(spawn_xy),
    )
    return env


def test_heading_error_is_relative_to_the_spawn_heading():
    half = math.pi / 2
    env = _pose_env([half, half + half, -math.pi + 0.01], [half, half, math.pi - 0.01],
                    [[0.0, 0.0, 0.1]] * 3, [[0.0, 0.0]] * 3)
    err = microduck_mdp.heading_error_penalty(env)
    assert torch.allclose(err, torch.tensor([0.0, 1.0, 0.0002]), atol=1e-3)


def test_lateral_offset_ignores_progress_along_the_heading():
    # Spawn facing +y at (1, 1): moving along y is progress, along x is drift.
    half = math.pi / 2
    env = _pose_env([half, half], [half, half],
                    [[1.0, 3.0, 0.1], [1.3, 1.0, 0.1]], [[1.0, 1.0], [1.0, 1.0]])
    off = microduck_mdp.lateral_offset_penalty(env)
    assert torch.allclose(off, torch.tensor([0.0, 0.09]), atol=1e-6)

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

"""Invariant tests for approach_inspect_mdp.

Run inside microduck_rl:  uv run pytest tests/test_approach_inspect_mdp.py
CPU only, no mjlab env — a minimal fake scene is enough to check the reward
properties AGENTS.md cares about:

  * potentials pay zero for holding
  * penalties are <= 0 everywhere
  * the dwell latch cannot be jackpotted by arriving fast
  * orbiting the ring does not accumulate dwell
  * fresh episodes carry no delta from the previous one
"""
import math
import types

import pytest
import torch

import mjlab_microduck.tasks.approach_inspect_mdp as M


# ── fake scene ──────────────────────────────────────────────────────────────

class _Data:
    pass


class _Entity:
    def __init__(self, n):
        self.data = _Data()
        self.data.root_link_pos_w = torch.zeros(n, 3)
        self.data.root_link_quat_w = torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1)
        self.data.root_link_lin_vel_w = torch.zeros(n, 3)
        self.data.site_pos_w = torch.zeros(n, 1, 3)
        self.data.body_com_pos_w = torch.zeros(n, 1, 3)


class _Cfg:
    name = "robot"
    site_ids = [0]
    body_ids = [0]


def make_env(n=4, dt=0.02):
    env = types.SimpleNamespace()
    env.num_envs = n
    env.device = "cpu"
    env.step_dt = dt
    env.episode_length_buf = torch.full((n,), 10)
    robot, target = _Entity(n), _Entity(n)
    env.scene = {"robot": robot, "target": target}
    return env, robot, target


def place(robot, target, dist, yaw=0.0, head_bearing=0.0, speed=0.0):
    """Robot at origin facing +x (yaw), target at dist along +x, head aimed at
    head_bearing relative to +x, moving at speed along +y."""
    n = robot.data.root_link_pos_w.shape[0]
    target.data.root_link_pos_w[:] = torch.tensor([dist, 0.0, 0.0])
    robot.data.root_link_quat_w[:] = torch.tensor(
        [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
    robot.data.body_com_pos_w[:, 0] = torch.tensor([0.0, 0.0, 0.2])
    robot.data.site_pos_w[:, 0] = torch.tensor(
        [0.05 * math.cos(head_bearing), 0.05 * math.sin(head_bearing), 0.2])
    robot.data.root_link_lin_vel_w[:] = torch.tensor([0.0, speed, 0.0])


CFG = _Cfg()


# ── tests ───────────────────────────────────────────────────────────────────

def test_ring_progress_pays_for_closing_and_zero_for_holding():
    env, r, t = make_env()
    place(r, t, 1.0)
    M.delta_ring_distance(env, standoff=0.22, asset_cfg=CFG)   # seed
    place(r, t, 0.8)
    closing = M.delta_ring_distance(env, standoff=0.22, asset_cfg=CFG)
    assert torch.allclose(closing, torch.full((4,), 0.2), atol=1e-6)
    hold = M.delta_ring_distance(env, standoff=0.22, asset_cfg=CFG)
    assert torch.all(hold == 0)


def test_ring_progress_charges_overshoot():
    env, r, t = make_env()
    place(r, t, 0.22)
    M.delta_ring_distance(env, asset_cfg=CFG)
    place(r, t, 0.10)   # walked past the ring into the object
    assert torch.all(M.delta_ring_distance(env, asset_cfg=CFG) < 0)


def test_fresh_episode_has_no_delta():
    env, r, t = make_env()
    place(r, t, 1.0)
    M.delta_ring_distance(env, asset_cfg=CFG)
    place(r, t, 0.22)
    env.episode_length_buf[:] = 0    # reset happened
    assert torch.all(M.delta_ring_distance(env, asset_cfg=CFG) == 0)


def test_aim_is_zero_outside_ring_and_peaks_inside():
    env, r, t = make_env()
    place(r, t, 1.0, head_bearing=0.0)
    assert torch.all(M.head_bearing_gaussian(env, head_cfg=CFG) == 0)
    place(r, t, 0.22, head_bearing=0.0)
    assert torch.allclose(M.head_bearing_gaussian(env, head_cfg=CFG), torch.ones(4))
    place(r, t, 0.22, head_bearing=1.0)
    assert torch.all(M.head_bearing_gaussian(env, head_cfg=CFG) < 0.01)


def test_penalties_are_never_positive():
    env, r, t = make_env()
    for d, hb, sp in [(0.22, 0.0, 0.3), (0.22, 1.2, 0.0), (1.0, 0.0, 1.0), (0.05, -2.0, 0.1)]:
        place(r, t, d, head_bearing=hb, speed=sp)
        assert torch.all(M.planar_speed_gated_penalty(env, asset_cfg=CFG) <= 0)
        assert torch.all(M.head_bearing_ema_l1_penalty(env, head_cfg=CFG) <= 0)


def test_speed_penalty_is_zero_outside_ring():
    env, r, t = make_env()
    place(r, t, 1.0, speed=1.0)
    assert torch.all(M.planar_speed_gated_penalty(env, asset_cfg=CFG) == 0)
    place(r, t, 0.22, speed=1.0)
    assert torch.all(M.planar_speed_gated_penalty(env, asset_cfg=CFG) < 0)


def test_dwell_is_slewed_and_saturates():
    env, r, t = make_env(dt=0.1)
    place(r, t, 0.22, head_bearing=0.0, speed=0.0)
    total = 0.0
    for _ in range(30):
        total += float(M.delta_dwell_progress(env, rate=0.5, head_cfg=CFG)[0])
    assert abs(total - 1.0) < 1e-6           # exactly one unit, no more
    assert float(M.delta_dwell_progress(env, rate=0.5, head_cfg=CFG)[0]) == 0.0


def test_dwell_cannot_be_jackpotted_by_arriving_fast():
    """Arriving instantly vs. arriving slowly must pay the same total."""
    fast_env, r, t = make_env(dt=0.1)
    place(r, t, 0.22, speed=0.0)
    fast = sum(float(M.delta_dwell_progress(fast_env, rate=0.5, head_cfg=CFG)[0]) for _ in range(40))

    slow_env, r2, t2 = make_env(dt=0.1)
    for d in [1.0, 0.8, 0.6, 0.4]:
        place(r2, t2, d, speed=0.0)
        M.delta_dwell_progress(slow_env, rate=0.5, head_cfg=CFG)
    place(r2, t2, 0.22, speed=0.0)
    slow = sum(float(M.delta_dwell_progress(slow_env, rate=0.5, head_cfg=CFG)[0]) for _ in range(40))
    assert abs(fast - slow) < 1e-6


def test_orbiting_does_not_accumulate_dwell():
    env, r, t = make_env(dt=0.1)
    place(r, t, 0.22, head_bearing=0.0, speed=0.4)   # on ring, aimed, moving
    total = sum(float(M.delta_dwell_progress(env, head_cfg=CFG)[0]) for _ in range(20))
    assert total == 0.0


def test_turn_away_frozen_until_dwell_saturates():
    env, r, t = make_env(dt=0.1)
    place(r, t, 0.22, yaw=0.0)
    env._ai_dwell = torch.zeros(4)
    M.delta_heading_progress(env, asset_cfg=CFG)
    place(r, t, 0.22, yaw=1.0)
    assert torch.all(M.delta_heading_progress(env, asset_cfg=CFG) == 0)
    env._ai_dwell = torch.ones(4)
    M.delta_heading_progress(env, asset_cfg=CFG)      # first active step: reseed
    place(r, t, 0.22, yaw=2.0)                        # turning toward pi pays
    assert torch.all(M.delta_heading_progress(env, asset_cfg=CFG) > 0)


def test_phase_flag_follows_dwell():
    env, _, _ = make_env()
    assert torch.all(M.phase_flag(env) == 0)
    env._ai_dwell = torch.ones(4)
    assert torch.all(M.phase_flag(env) == 1)

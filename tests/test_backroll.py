import math
from types import SimpleNamespace

import pytest
import torch

from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_backroll_env_cfg import (
    BACKROLL_CURRICULUM_STAGES,
    EPISODE_LENGTH_S,
    MicroduckBackrollRlCfg,
    make_microduck_backroll_env_cfg,
)
from mjlab_microduck.tasks.microduck_roulade_env_cfg import (
    make_microduck_roulade_env_cfg,
)


class _Scene:
    def __init__(self, asset, sensors):
        self.asset = asset
        self.sensors = sensors
        self.terrain = SimpleNamespace(env_origins=torch.zeros(1, 3))

    def __getitem__(self, name):
        assert name == "robot"
        return self.asset


def _fake_env():
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    asset = SimpleNamespace(
        data=SimpleNamespace(
            root_link_ang_vel_b=torch.zeros(1, 3),
            root_link_lin_vel_b=torch.zeros(1, 3),
            root_link_lin_vel_w=torch.zeros(1, 3),
            root_link_pos_w=torch.tensor([[0.0, 0.0, 0.115]]),
            root_link_quat_w=quat,
        )
    )
    sensors = {
        "robot_ground_contact": SimpleNamespace(
            data=SimpleNamespace(found=torch.ones(1, 1))
        ),
        "trunk_ground_contact": SimpleNamespace(
            data=SimpleNamespace(found=torch.zeros(1, 1))
        ),
        "head_ground_contact": SimpleNamespace(
            data=SimpleNamespace(found=torch.zeros(1, 1))
        ),
        "left_foot_ground_contact": SimpleNamespace(
            data=SimpleNamespace(found=torch.ones(1, 1))
        ),
        "right_foot_ground_contact": SimpleNamespace(
            data=SimpleNamespace(found=torch.ones(1, 1))
        ),
    }
    env = SimpleNamespace(
        num_envs=1,
        device=torch.device("cpu"),
        step_dt=0.02,
        common_step_counter=0,
        scene=_Scene(asset, sensors),
    )
    mdp._grounded_backroll_state(env)
    env._roulade_roll_direction[:] = -1.0
    return env, asset


def _next_step(env, asset, omega_y):
    asset.data.root_link_ang_vel_b[:, 1] = omega_y
    value = mdp.grounded_backroll_progress(env)
    env.common_step_counter += 1
    return value


def test_backroll_is_one_shot_roulade_without_sprint_objectives():
    cfg = make_microduck_backroll_env_cfg()
    roulade = make_microduck_roulade_env_cfg()

    assert cfg.episode_length_s == EPISODE_LENGTH_S == 5.0
    assert list(cfg.observations["actor"].terms) == list(
        roulade.observations["actor"].terms
    )
    assert set(cfg.rewards) == {
        "backroll_progress",
        "backroll_head_pivot",
        "backroll_upright_progress",
        "backroll_height_progress",
        "backroll_success",
        "backroll_overspeed",
        "backroll_sagittal",
        "backroll_lateral_velocity",
        "backroll_flatness",
        "action_rate_l2",
        "gentle_landing",
        "self_collisions",
    }
    forbidden = ("sprint", "distance", "lane", "road", "recovery", "reposition")
    assert not any(token in name for name in cfg.rewards for token in forbidden)
    assert cfg.events["set_grounded_backroll_state"].func is mdp.reset_grounded_backroll_state
    assert cfg.curriculum["backroll_phase"].func is mdp.grounded_backroll_curriculum
    assert MicroduckBackrollRlCfg.experiment_name == "microduck_backroll"
    assert MicroduckBackrollRlCfg.max_iterations == 4000
    assert MicroduckBackrollRlCfg.save_interval == 50
    assert MicroduckBackrollRlCfg.algorithm.learning_rate == pytest.approx(1.0e-3)
    assert MicroduckBackrollRlCfg.actor.distribution_cfg["init_std"] == 1.0


def test_backroll_play_is_deterministic_standing_start():
    cfg = make_microduck_backroll_env_cfg(play=True)
    reset = cfg.events["set_grounded_backroll_state"].params

    assert reset["standing_prob"] == 1.0
    assert reset["midroll_prob"] == 0.0
    assert reset["yaw_range"] == (0.0, 0.0)
    assert reset["joint_noise_std"] == 0.0
    assert "backroll_phase" not in cfg.curriculum


def test_backroll_curriculum_matches_mastery_stages():
    assert [stage["params"]["standing_prob"] for stage in BACKROLL_CURRICULUM_STAGES] == [
        0.20,
        0.30,
        0.40,
        0.60,
        0.85,
    ]
    assert [stage["params"]["midroll_prob"] for stage in BACKROLL_CURRICULUM_STAGES] == [
        0.80,
        0.70,
        0.60,
        0.40,
        0.15,
    ]
    cfg = make_microduck_backroll_env_cfg()
    params = cfg.curriculum["backroll_phase"].params
    assert params["window_episodes"] == 4096
    assert params["success_threshold"] == pytest.approx(0.70)


def test_reverse_phase_reset_uses_negative_pitch_rate_and_contact_prerequisites(
    monkeypatch,
):
    env, _asset = _fake_env()
    env.sim = SimpleNamespace(
        data=SimpleNamespace(
            qpos=torch.zeros(1, 21),
            qvel=torch.zeros(1, 20),
        )
    )
    env.sim.data.qpos[:, 3] = 1.0
    monkeypatch.setattr(mdp, "_servo_joint_ids", lambda _env, _asset: list(range(14)))
    pitch = math.radians(200.0)

    mdp.reset_grounded_backroll_state(
        env,
        torch.tensor([0]),
        standing_prob=0.0,
        midroll_prob=1.0,
        midroll_pitch_min=pitch,
        midroll_pitch_max=pitch,
        midroll_omega_range=(3.0, 3.0),
        standing_tilt_max=0.0,
        yaw_range=(0.0, 0.0),
        joint_noise_std=0.0,
    )

    assert env._roulade_roll_direction.item() == -1.0
    assert env.sim.data.qpos[0, 5].item() < 0.0
    assert env.sim.data.qvel[0, 4].item() == pytest.approx(-3.0)
    assert env._backroll_trunk_latch.item()
    assert env._backroll_head_latch.item()


def test_negative_body_y_advances_but_forward_rocking_cannot_farm(monkeypatch):
    env, asset = _fake_env()
    monkeypatch.setattr(mdp, "_lateral_axis_z", lambda _quat: torch.zeros(1))
    monkeypatch.setattr(
        mdp,
        "_head_top_down",
        lambda _env, _asset: torch.ones(1, dtype=torch.bool),
    )

    first = _next_step(env, asset, -1.0)
    frontier = env._roulade_max.clone()
    backward = _next_step(env, asset, 1.0)
    revisit = _next_step(env, asset, -1.0)
    extension = _next_step(env, asset, -1.0)

    assert first.item() > 0.0
    assert backward.item() == 0.0
    assert revisit.item() == 0.0
    assert torch.equal(env._roulade_max - frontier, torch.tensor([0.02]))
    assert extension.item() > 0.0


def test_airborne_and_sideways_rotation_receive_no_progress(monkeypatch):
    env, asset = _fake_env()
    monkeypatch.setattr(
        mdp,
        "_head_top_down",
        lambda _env, _asset: torch.ones(1, dtype=torch.bool),
    )
    env.scene.sensors["robot_ground_contact"].data.found[:] = 0.0
    assert _next_step(env, asset, -2.0).item() == 0.0

    env.scene.sensors["robot_ground_contact"].data.found[:] = 1.0
    monkeypatch.setattr(mdp, "_lateral_axis_z", lambda _quat: torch.ones(1))
    assert _next_step(env, asset, -2.0).item() == 0.0


def test_head_contact_only_latches_after_trunk_contact(monkeypatch):
    env, _asset = _fake_env()
    monkeypatch.setattr(mdp, "_lateral_axis_z", lambda _quat: torch.zeros(1))
    monkeypatch.setattr(
        mdp,
        "_head_top_down",
        lambda _env, _asset: torch.ones(1, dtype=torch.bool),
    )
    env._roulade_accum[:] = math.radians(150.0)
    env._roulade_max[:] = math.radians(150.0)
    env.scene.sensors["head_ground_contact"].data.found[:] = 1.0
    mdp.grounded_backroll_progress(env)
    assert not env._backroll_head_latch.item()

    env.common_step_counter += 1
    env._roulade_accum[:] = math.radians(50.0)
    env._roulade_max[:] = math.radians(50.0)
    env.scene.sensors["head_ground_contact"].data.found[:] = 0.0
    env.scene.sensors["trunk_ground_contact"].data.found[:] = 1.0
    mdp.grounded_backroll_progress(env)
    assert env._backroll_trunk_latch.item()

    env.common_step_counter += 1
    env._roulade_accum[:] = math.radians(150.0)
    env._roulade_max[:] = math.radians(150.0)
    env.scene.sensors["head_ground_contact"].data.found[:] = 1.0
    mdp.grounded_backroll_progress(env)
    assert env._backroll_head_latch.item()


def test_airborne_gap_and_wrong_way_completion_are_invalid(monkeypatch):
    env, asset = _fake_env()
    monkeypatch.setattr(mdp, "_lateral_axis_z", lambda _quat: torch.zeros(1))
    monkeypatch.setattr(
        mdp,
        "_head_top_down",
        lambda _env, _asset: torch.ones(1, dtype=torch.bool),
    )
    env.scene.sensors["robot_ground_contact"].data.found[:] = 0.0
    for _ in range(math.ceil(mdp._BACKROLL_MAX_AIR_SECONDS / env.step_dt) + 1):
        _next_step(env, asset, -2.0)
    assert mdp.grounded_backroll_invalid_termination(env).item()

    env, asset = _fake_env()
    monkeypatch.setattr(mdp, "_lateral_axis_z", lambda _quat: torch.zeros(1))
    for _ in range(46):
        _next_step(env, asset, 2.0)
    assert mdp.grounded_backroll_invalid_termination(env).item()


def test_success_requires_ordered_contacts_and_is_one_shot(monkeypatch):
    env, _asset = _fake_env()
    monkeypatch.setattr(mdp, "_lateral_axis_z", lambda _quat: torch.zeros(1))
    monkeypatch.setattr(
        mdp,
        "_head_top_down",
        lambda _env, _asset: torch.ones(1, dtype=torch.bool),
    )
    env._roulade_accum[:] = math.radians(355.0)
    env._roulade_max[:] = math.radians(355.0)
    env._roulade_paid[:] = math.radians(355.0)
    env._backroll_previous_frontier[:] = math.radians(355.0)

    hold_steps = math.ceil(mdp._BACKROLL_LANDING_HOLD_SECONDS / env.step_dt)
    for _ in range(hold_steps + 1):
        value = mdp.grounded_backroll_success_rate(env)
        assert value.item() == 0.0
        env.common_step_counter += 1

    env._backroll_trunk_latch[:] = True
    env._backroll_head_latch[:] = True
    pulses = []
    for _ in range(hold_steps + 2):
        pulses.append(mdp.grounded_backroll_success_rate(env).item())
        env.common_step_counter += 1

    assert sum(value > 0.0 for value in pulses) == 1
    assert env._backroll_success.item()


def test_standing_cannot_farm_backroll_success(monkeypatch):
    env, _asset = _fake_env()
    monkeypatch.setattr(mdp, "_lateral_axis_z", lambda _quat: torch.zeros(1))
    monkeypatch.setattr(
        mdp,
        "_head_top_down",
        lambda _env, _asset: torch.ones(1, dtype=torch.bool),
    )
    for _ in range(25):
        assert mdp.grounded_backroll_success_rate(env).item() == 0.0
        env.common_step_counter += 1
    assert not env._backroll_success.item()


def test_forward_roulade_direction_default_is_unchanged(monkeypatch):
    env, asset = _fake_env()
    env._roulade_roll_direction[:] = 1.0
    monkeypatch.setattr(mdp, "_lateral_axis_z", lambda _quat: torch.zeros(1))
    monkeypatch.setattr(
        mdp,
        "_head_top_down",
        lambda _env, _asset: torch.ones(1, dtype=torch.bool),
    )
    asset.data.root_link_ang_vel_b[:, 1] = 1.0

    mdp._update_roulade_accum(env, asset)

    assert env._roulade_accum.item() == pytest.approx(0.02)

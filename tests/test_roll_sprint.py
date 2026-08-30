from types import SimpleNamespace

import torch

from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_roll_sprint_env_cfg import (
    MicroduckRollSprintRlCfg,
    make_microduck_roll_sprint_env_cfg,
)
from mjlab_microduck.tasks.microduck_roulade_env_cfg import (
    make_microduck_roulade_env_cfg,
)


class _FakeScene:
    def __init__(self, asset, sensors, origins):
        self.robot = asset
        self.sensors = sensors
        self.terrain = SimpleNamespace(env_origins=origins)

    def __getitem__(self, name):
        assert name == "robot"
        return self.robot


def _fake_env(num_envs: int = 1):
    identity = torch.zeros(num_envs, 4)
    identity[:, 0] = 1.0
    data = SimpleNamespace(
        root_link_ang_vel_b=torch.zeros(num_envs, 3),
        root_link_lin_vel_w=torch.zeros(num_envs, 3),
        root_link_pos_w=torch.zeros(num_envs, 3),
        root_link_quat_w=identity,
    )
    asset = SimpleNamespace(data=data)
    sensors = {
        "robot_ground_contact": SimpleNamespace(
            data=SimpleNamespace(found=torch.ones(num_envs, 1))
        ),
        "head_ground_contact": SimpleNamespace(
            data=SimpleNamespace(found=torch.zeros(num_envs, 1))
        ),
    }
    env = SimpleNamespace(
        num_envs=num_envs,
        device=torch.device("cpu"),
        step_dt=0.02,
        common_step_counter=0,
        scene=_FakeScene(asset, sensors, torch.zeros(num_envs, 3)),
    )
    return env, asset


def test_roll_sprint_is_separate_six_second_61d_policy():
    cfg = make_microduck_roll_sprint_env_cfg()
    roulade = make_microduck_roulade_env_cfg()

    assert cfg.episode_length_s == 6.0
    assert cfg.scene.terrain.terrain_type == "plane"
    assert list(cfg.observations["actor"].terms) == list(
        roulade.observations["actor"].terms
    )
    assert (
        cfg.observations["critic"].terms["roll_sprint_critic_padding"].params["dim"]
        == 16
    )
    assert cfg.actions["joint_pos"].scale == 1.0
    assert "roll_sprint_distance" in cfg.rewards
    assert (
        cfg.rewards["roll_sprint_distance"].weight
        > cfg.rewards["roll_sprint_progress"].weight
    )
    assert cfg.rewards["roll_sprint_cycle_rate"].weight == 1.0
    assert cfg.rewards["roll_sprint_invalid_cycle"].weight == -2.0
    assert cfg.rewards["roll_sprint_sagittal"].weight == -0.05
    assert cfg.rewards["roll_sprint_flatness"].weight == -0.25
    assert cfg.rewards["roll_sprint_lateral_vel"].weight == -0.20
    for name in (
        "roll_sprint_overspeed",
        "roll_sprint_impact",
        "body_ang_vel",
        "angular_momentum",
        "action_rate_l2",
        "joint_torque_rate_l2",
        "self_collisions",
    ):
        assert cfg.rewards[name].weight == 0.0
    assert not any(
        name.startswith("roulade_") or name in {"upright", "height", "pose"}
        for name in cfg.rewards
    )
    assert "set_roll_sprint_state" in cfg.events
    assert "set_roulade_state" not in cfg.events
    assert cfg.terminations["nan_state"].time_out is False
    assert MicroduckRollSprintRlCfg.experiment_name == "microduck_roll_sprint"
    assert MicroduckRollSprintRlCfg.save_interval == 25


def test_roll_sprint_and_backlash_tasks_are_registered():
    from mjlab.tasks.registry import list_tasks

    import mjlab_microduck.tasks  # noqa: F401

    tasks = list_tasks()
    assert "Mjlab-Roll-Sprint-Flat-MicroDuck" in tasks
    assert "Mjlab-Roll-Sprint-Flat-Backlash-MicroDuck" in tasks


def test_roll_sprint_buffer_reset_is_per_environment():
    env, _ = _fake_env(3)
    mdp._roll_sprint_state(env)
    env._roll_sprint_accum[:] = torch.tensor([0.5, 1.0, 1.5])
    env._roll_sprint_frontier[:] = torch.tensor([0.6, 1.1, 1.6])
    env._roll_sprint_completed[:] = torch.tensor([2.0, 3.0, 4.0])
    env._roll_sprint_head_latch[:] = True
    env._roll_sprint_cycle_eligible[:] = True
    env._roll_sprint_lateral_invalid[:] = True

    mdp._reset_roll_sprint_buffers(
        env,
        torch.tensor([1]),
        spawn_accum=torch.tensor([0.25]),
        spawn_head_latch=torch.tensor([False]),
    )

    assert torch.allclose(env._roll_sprint_accum, torch.tensor([0.5, 0.25, 1.5]))
    assert torch.allclose(env._roll_sprint_frontier, torch.tensor([0.6, 0.25, 1.6]))
    assert torch.equal(env._roll_sprint_completed, torch.tensor([2.0, 0.0, 4.0]))
    assert torch.equal(env._roll_sprint_head_latch, torch.tensor([True, False, True]))
    assert torch.equal(
        env._roll_sprint_lateral_invalid, torch.tensor([True, False, True])
    )


def test_roll_sprint_requires_support_and_sagittal_forward_rotation(monkeypatch):
    env, asset = _fake_env(3)
    asset.data.root_link_ang_vel_b[:, 1] = torch.tensor([5.0, 5.0, -5.0])
    env.scene.sensors["robot_ground_contact"].data.found[:, 0] = torch.tensor(
        [0.0, 1.0, 1.0]
    )
    monkeypatch.setattr(
        mdp,
        "_lateral_axis_z",
        lambda quat: torch.tensor([0.0, 0.99, 0.0]),
    )
    monkeypatch.setattr(
        mdp,
        "_head_top_down",
        lambda env, asset: torch.ones(env.num_envs, dtype=torch.bool),
    )

    mdp._update_roll_sprint_state(env, asset)

    assert env._roll_sprint_accum[0] == 0.0
    assert env._roll_sprint_accum[1] == 0.0
    assert env._roll_sprint_accum[2] == 0.0
    assert not env._roll_sprint_completed_now.any()


def test_roll_sprint_missing_support_sensor_has_zero_credit(monkeypatch):
    env, asset = _fake_env(1)
    del env.scene.sensors["robot_ground_contact"]
    asset.data.root_link_ang_vel_b[:, 1] = 5.0
    monkeypatch.setattr(
        mdp,
        "_head_top_down",
        lambda env, asset: torch.ones(env.num_envs, dtype=torch.bool),
    )

    mdp._update_roll_sprint_state(env, asset)

    assert env._roll_sprint_accum[0] == 0.0
    assert env._roll_sprint_progress_delta[0] == 0.0
    assert env._roll_sprint_cycle_distance[0] == 0.0


def test_roll_sprint_backward_rocking_never_completes_or_repays_progress(monkeypatch):
    env, asset = _fake_env(1)
    monkeypatch.setattr(
        mdp,
        "_head_top_down",
        lambda env, asset: torch.ones(env.num_envs, dtype=torch.bool),
    )
    paid_progress = 0.0

    for step in range(40):
        env.common_step_counter = step
        asset.data.root_link_ang_vel_b[:, 1] = 5.0 if step % 2 == 0 else -5.0
        mdp._update_roll_sprint_state(env, asset)
        paid_progress += float(env._roll_sprint_progress_delta[0])

    assert torch.isclose(env._roll_sprint_accum[0], torch.tensor(0.0))
    assert torch.isclose(torch.tensor(paid_progress), torch.tensor(0.1))
    assert env._roll_sprint_completed[0] == 0.0
    assert not env._roll_sprint_completed_now[0]


def test_roll_sprint_head_top_latch_and_invalid_cycle_zero_credit(monkeypatch):
    env, asset = _fake_env(2)
    asset.data.root_link_ang_vel_b[:, 1] = 1.0
    env.scene.sensors["head_ground_contact"].data.found[:] = 1.0
    monkeypatch.setattr(
        mdp,
        "_head_top_down",
        lambda env, asset: torch.tensor([True, False]),
    )

    mdp._roll_sprint_state(env)
    env._roll_sprint_accum[:] = 0.5
    mdp._update_roll_sprint_state(env, asset)
    assert torch.equal(env._roll_sprint_head_latch, torch.tensor([True, False]))

    env.common_step_counter = 1
    env._roll_sprint_accum[:] = 2.0 * torch.pi - 0.01
    env._roll_sprint_head_latch[:] = False
    mdp._update_roll_sprint_state(env, asset)

    assert not env._roll_sprint_completed_now.any()
    assert env._roll_sprint_invalid_now.all()
    assert env._roll_sprint_completed.sum() == 0.0
    assert env._roll_sprint_completed_distance.sum() == 0.0
    assert env._roll_sprint_accum.sum() == 0.0


def test_midroll_bootstrap_segment_cannot_receive_cycle_credit(monkeypatch):
    env, asset = _fake_env(1)
    asset.data.root_link_ang_vel_b[:, 1] = 1.0
    env.scene.sensors["head_ground_contact"].data.found[:] = 1.0
    monkeypatch.setattr(
        mdp,
        "_head_top_down",
        lambda env, asset: torch.ones(env.num_envs, dtype=torch.bool),
    )
    mdp._reset_roll_sprint_buffers(
        env,
        torch.tensor([0]),
        spawn_accum=torch.tensor([2.0 * torch.pi - 0.01]),
        spawn_head_latch=torch.tensor([True]),
        spawn_cycle_eligible=torch.tensor([False]),
    )

    mdp._update_roll_sprint_state(env, asset)

    assert env._roll_sprint_bootstrap_completed_now[0]
    assert not env._roll_sprint_completed_now[0]
    assert not env._roll_sprint_invalid_now[0]
    assert env._roll_sprint_completed[0] == 0.0
    assert env._roll_sprint_completed_distance[0] == 0.0
    assert env._roll_sprint_cycle_eligible[0]

    env.common_step_counter = 1
    env._roll_sprint_accum[:] = 2.0 * torch.pi - 0.01
    env._roll_sprint_frontier[:] = env._roll_sprint_accum
    env._roll_sprint_head_latch[:] = True
    mdp._update_roll_sprint_state(env, asset)

    assert env._roll_sprint_completed_now[0]
    assert env._roll_sprint_completed[0] == 1.0


def test_roll_sprint_side_violation_invalidates_whole_cycle(monkeypatch):
    env, asset = _fake_env(1)
    asset.data.root_link_ang_vel_b[:, 1] = 1.0
    env.scene.sensors["head_ground_contact"].data.found[:] = 1.0
    lateral = torch.tensor([0.90])
    monkeypatch.setattr(mdp, "_lateral_axis_z", lambda quat: lateral)
    monkeypatch.setattr(
        mdp,
        "_head_top_down",
        lambda env, asset: torch.ones(env.num_envs, dtype=torch.bool),
    )

    mdp._update_roll_sprint_state(env, asset)
    assert env._roll_sprint_lateral_invalid[0]

    env.common_step_counter = 1
    lateral[:] = 0.0
    env._roll_sprint_accum[:] = 2.0 * torch.pi - 0.01
    env._roll_sprint_frontier[:] = env._roll_sprint_accum
    env._roll_sprint_head_latch[:] = True
    mdp._update_roll_sprint_state(env, asset)

    assert env._roll_sprint_invalid_now[0]
    assert not env._roll_sprint_completed_now[0]
    assert env._roll_sprint_completed_distance[0] == 0.0


def test_roll_sprint_heading_survives_vertical_pitch():
    _, asset = _fake_env(1)
    half = torch.tensor(torch.pi / 4.0)
    # ZYX yaw=90 degrees, pitch=90 degrees, roll=0.
    asset.data.root_link_quat_w[:] = torch.tensor(
        [
            [
                half.cos() * half.cos(),
                -half.sin() * half.sin(),
                half.cos() * half.sin(),
                half.sin() * half.cos(),
            ]
        ]
    )

    heading = mdp._roll_sprint_heading(asset)

    assert torch.allclose(heading, torch.tensor([[0.0, 1.0]]), atol=1.0e-5)


def test_roll_sprint_per_env_step_guard_updates_only_reset_environment(monkeypatch):
    env, asset = _fake_env(2)
    asset.data.root_link_ang_vel_b[:, 1] = 1.0
    monkeypatch.setattr(
        mdp,
        "_head_top_down",
        lambda env, asset: torch.ones(env.num_envs, dtype=torch.bool),
    )
    mdp._update_roll_sprint_state(env, asset)
    before = env._roll_sprint_accum.clone()

    mdp._reset_roll_sprint_buffers(env, torch.tensor([0]))
    asset.data.root_link_ang_vel_b[:, 1] = 2.0
    mdp._update_roll_sprint_state(env, asset)

    assert env._roll_sprint_accum[0] > 0.0
    assert env._roll_sprint_accum[1] == before[1]


def test_roll_sprint_releases_heading_aligned_distance_on_completion(monkeypatch):
    env, asset = _fake_env(1)
    yaw_quat = torch.tensor([[2.0**-0.5, 0.0, 0.0, 2.0**-0.5]])
    asset.data.root_link_quat_w[:] = yaw_quat
    asset.data.root_link_lin_vel_w[:, 1] = 1.0
    asset.data.root_link_ang_vel_b[:, 1] = 1.0
    env.scene.sensors["head_ground_contact"].data.found[:] = 1.0
    monkeypatch.setattr(
        mdp,
        "_head_top_down",
        lambda env, asset: torch.ones(env.num_envs, dtype=torch.bool),
    )
    mdp._roll_sprint_state(env)
    env._roll_sprint_accum[:] = 2.0 * torch.pi - 0.01
    env._roll_sprint_head_latch[:] = True

    mdp._update_roll_sprint_state(env, asset)

    assert env._roll_sprint_completed[0] == 1.0
    assert env._roll_sprint_completed_now[0]
    assert env._roll_sprint_completed_distance[0] > 0.0


def test_roll_sprint_distance_is_rotation_linked_and_capped(monkeypatch):
    env, asset = _fake_env(1)
    asset.data.root_link_lin_vel_w[:, 0] = 100.0
    asset.data.root_link_ang_vel_b[:, 1] = 1.0
    monkeypatch.setattr(
        mdp,
        "_head_top_down",
        lambda env, asset: torch.ones(env.num_envs, dtype=torch.bool),
    )
    mdp._roll_sprint_state(env)
    env._roll_sprint_accum[:] = 2.0 * torch.pi - 0.01
    env._roll_sprint_frontier[:] = env._roll_sprint_accum
    env._roll_sprint_head_latch[:] = True

    mdp._update_roll_sprint_state(env, asset)

    expected_cap = 0.12 * 1.0 * env.step_dt
    assert torch.allclose(
        env._roll_sprint_completed_distance,
        torch.tensor([expected_cap]),
        atol=1.0e-6,
    )


def test_roll_sprint_completion_preserves_only_small_overshoot(monkeypatch):
    env, asset = _fake_env(1)
    asset.data.root_link_ang_vel_b[:, 1] = 2.0
    monkeypatch.setattr(
        mdp,
        "_head_top_down",
        lambda env, asset: torch.ones(env.num_envs, dtype=torch.bool),
    )
    mdp._roll_sprint_state(env)
    env._roll_sprint_accum[:] = 2.0 * torch.pi - 0.01
    env._roll_sprint_frontier[:] = env._roll_sprint_accum
    env._roll_sprint_head_latch[:] = True

    mdp._update_roll_sprint_state(env, asset)

    assert env._roll_sprint_completed_now[0]
    assert torch.allclose(env._roll_sprint_accum, torch.tensor([0.03]), atol=1.0e-5)
    assert torch.allclose(env._roll_sprint_frontier, env._roll_sprint_accum)

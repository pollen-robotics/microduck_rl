"""CPU checks for the separate, not-yet-validated side-step training recipe."""

from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

from mjlab_microduck.sim.visualization.lateral.config import (
    MicroduckLateralRlCfg,
    command_families,
    lateral_critic_state,
    lateral_forward_drift_l2,
    lateral_heading_drift_l2,
    lateral_path_drift_l2,
    lateral_progress_values,
    lateral_velocity_progress,
    lateral_yaw_rate_l2,
    make_lateral_cfg,
)
from mjlab_microduck.tasks.microduck_velstand_env_cfg import (
    make_microduck_velstand_env_cfg,
)


def test_lateral_task_discovery_uses_the_isolated_recipe():
    from mjlab.tasks.registry import list_tasks, load_env_cfg, load_runner_cls

    from mjlab_microduck.sim.visualization.lateral.register import TASK_ID
    from mjlab_microduck.tasks import MicroduckOnPolicyRunner

    assert make_lateral_cfg.__module__ == (
        "mjlab_microduck.sim.visualization.lateral.env_cfg"
    )
    assert MicroduckLateralRlCfg.experiment_name == "lateral"
    assert TASK_ID in list_tasks()
    assert load_runner_cls(TASK_ID) is MicroduckOnPolicyRunner
    cfg = load_env_cfg(TASK_ID)
    assert "lateral_progress" in cfg.rewards
    assert "lateral_segment_state" in cfg.observations["critic"].terms


def fake_env(count):
    data = SimpleNamespace(
        root_link_lin_vel_b=torch.zeros(count, 3),
        root_link_ang_vel_b=torch.zeros(count, 3),
        projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]).repeat(count, 1),
        root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(count, 1),
        root_link_pos_w=torch.zeros(count, 3),
    )
    return SimpleNamespace(
        num_envs=count,
        device="cpu",
        step_dt=0.02,
        scene={"robot": SimpleNamespace(data=data)},
    )


def test_lateral_recipe_does_not_mutate_v8_defaults():
    before = make_microduck_velstand_env_cfg()
    train, play = make_lateral_cfg(), make_lateral_cfg(play=True)
    after = make_microduck_velstand_env_cfg()
    assert type(after.commands["twist"]) is type(before.commands["twist"])
    assert {"push_robot", "topple_push", "random_prone_init"} <= set(after.events)
    assert "lateral_progress" not in after.rewards
    assert (
        after.rewards["action_rate_l2"].weight
        == before.rewards["action_rate_l2"].weight
    )
    for cfg in (train, play):
        assert not {"push_robot", "topple_push", "random_prone_init"} & set(cfg.events)
        assert not cfg.curriculum
        assert cfg.sim.mujoco.timestep == 0.005
        assert (cfg.sim.mujoco.iterations, cfg.sim.mujoco.ls_iterations) == (30, 50)
        assert cfg.sim.mujoco.integrator == "implicitfast"
        assert list(cfg.observations["actor"].terms) == list(
            after.observations["actor"].terms
        )
        assert list(cfg.observations["critic"].terms) == [
            *after.observations["critic"].terms,
            "lateral_segment_state",
        ]
        assert cfg.actions["joint_pos"].scale == after.actions["joint_pos"].scale
        assert cfg.observations["actor"].terms["joint_vel"].delay_min_lag == 1
        act = cfg.scene.entities["robot"].articulation.actuators[0]
        assert (act.delay_min_lag, act.delay_max_lag) == (3, 6)
        assert cfg.commands["twist"].heading_command is False
    assert train.observations["actor"].enable_corruption
    assert not play.observations["actor"].enable_corruption


def test_critic_distinguishes_identical_local_states_with_different_segment_costs():
    env = fake_env(2)
    commands = torch.tensor([[0.0, 0.3, 0.0]]).repeat(2, 1)
    term = SimpleNamespace(
        segment_yaw=torch.tensor([0.0, 0.1745329252]),
        segment_position=torch.tensor([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]]),
        mean_yaw_rate=torch.tensor([0.0, 0.1]),
        command_counter=torch.tensor([2, 2]),
        time_left=torch.tensor([8.0, 6.0]),
        family=torch.zeros(2, dtype=torch.long),
        first_sign=torch.ones(2),
        cfg=SimpleNamespace(rest_seconds=2.0, move_seconds=10.0),
    )
    env.command_manager = SimpleNamespace(
        get_command=lambda _: commands, get_term=lambda _: term
    )
    # Robot pose, local velocity, action history and current command are equal;
    # the accumulated reference and reward are different. A local-only value
    # function cannot distinguish these trajectories.
    state = lateral_critic_state(env)
    assert state.shape == (2, 9)
    assert torch.isfinite(state).all()
    assert state[0].tolist() == pytest.approx([0, 1, 0, 0, 0, 1 / 3, 0.8, 0, 1])
    assert state[1, :2].tolist() == pytest.approx([-0.17364818, 0.98480775])
    assert state[1, 4:].tolist() == pytest.approx([0.1, 1 / 3, 0.6, 0, 1])
    assert lateral_heading_drift_l2(env).tolist() == pytest.approx([0, 1])
    assert lateral_path_drift_l2(env)[1] > 0.9


def test_critic_encodes_rest_phase_and_future_command_without_actor_heading_input():
    env = fake_env(100)
    command = make_lateral_cfg(play=True).commands["twist"].build(env)
    env.command_manager = SimpleNamespace(get_term=lambda _: command)
    command.reset(torch.arange(100))
    state = lateral_critic_state(env)
    assert torch.count_nonzero(command.command) == 0
    assert torch.equal(state[:, 5], torch.zeros(100))
    assert torch.equal(state[:, 6], torch.ones(100))
    # At rest the current command cannot reveal the next command family/sign.
    assert state[0, 7:].tolist() == [0.0, 1.0]
    assert state[1, 7:].tolist() == [0.0, -1.0]
    assert state[70, 7:].tolist() == [0.5, 1.0]
    command.compute(1.0)
    assert lateral_critic_state(env)[0, 6] == 0.5
    command.compute(1.0)
    assert lateral_critic_state(env)[0, 5:].tolist() == pytest.approx([1 / 3, 1, 0, 1])
    command.compute(10.0)
    assert torch.count_nonzero(command.command) == 0
    assert lateral_critic_state(env)[0, 5:].tolist() == pytest.approx([2 / 3, 1, 0, 1])


def test_deterministic_cycle_covers_both_sides_rest_and_general_motion():
    env = fake_env(100)
    command = make_lateral_cfg(play=True).commands["twist"].build(env)
    command.reset(torch.arange(100))
    assert torch.count_nonzero(command.command) == 0
    command.compute(command.cfg.rest_seconds)
    moving = command.command.clone()
    assert torch.count_nonzero(moving[:, 1]) == 70
    assert torch.count_nonzero(moving[:, 0]) == 15
    assert torch.count_nonzero(moving[:, 2]) == 15
    assert int((moving[:70, 1] > 0).sum()) == 35
    assert torch.equal(moving[:70, 0], torch.zeros(70))
    assert torch.equal(moving[:70, 2], torch.zeros(70))
    command.compute(command.cfg.move_seconds)
    assert torch.count_nonzero(command.command) == 0
    command.compute(command.cfg.rest_seconds)
    assert torch.equal(command.command, -moving)
    command.compute(command.cfg.move_seconds)
    assert torch.count_nonzero(command.command) == 0
    assert torch.count_nonzero(env.scene["robot"].data.root_link_lin_vel_b) == 0
    assert not command.is_heading_env.any() and not command.is_world_env.any()


def test_training_sampling_keeps_lateral_and_direction_shares_balanced():
    torch.manual_seed(37)
    env = fake_env(10_000)
    command = make_lateral_cfg().commands["twist"].build(env)
    command.reset(torch.arange(env.num_envs))
    command.compute(command.cfg.rest_seconds)
    lateral = command.command[:, 1] != 0
    assert abs(lateral.float().mean().item() - 0.7) < 0.02
    assert abs((command.command[:, 0] != 0).float().mean().item() - 0.15) < 0.02
    assert abs((command.command[:, 2] != 0).float().mean().item() - 0.15) < 0.02
    assert abs((command.command[lateral, 1] > 0).float().mean().item() - 0.5) < 0.03
    assert torch.count_nonzero(command.command[lateral][:, [0, 2]]) == 0


def test_resetting_one_environment_restarts_only_its_rest_phase():
    command = make_lateral_cfg(play=True).commands["twist"].build(fake_env(2))
    command.reset(torch.arange(2))
    command.compute(command.cfg.rest_seconds)
    other = command.command[1].clone()
    command.reset(torch.tensor([0]))
    assert torch.count_nonzero(command.command[0]) == 0
    assert torch.equal(command.command[1], other)


def test_progress_requires_real_signed_lateral_motion_and_upright_state():
    command = torch.tensor([[0.0, 0.3, 0.0]]).repeat(7, 1)
    velocity = torch.zeros(7, 3)
    velocity[:, 1] = torch.tensor([0.0, 0.05, -0.05, 0.3, 0.6, -0.3, 0.3])
    command[5, 1] = -0.3
    gravity = torch.tensor([[0.0, 0.0, -1.0]]).repeat(7, 1)
    gravity[6, 2] = 0
    reward = lateral_progress_values(command, velocity, gravity)
    assert reward.tolist() == pytest.approx([0.0, 1 / 6, -1 / 6, 1.0, 1.0, 1.0, 0.0])
    command[:] = torch.tensor([0.3, 0.0, 0.0])
    assert torch.count_nonzero(lateral_progress_values(command, velocity, gravity)) == 0


def test_drift_costs_only_apply_to_pure_lateral_commands():
    env = fake_env(4)
    command = torch.tensor(
        [[0.0, 0.3, 0.0], [0.0, -0.3, 0.0], [0.3, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )
    env.command_manager = SimpleNamespace(get_command=lambda _: command)
    env.scene["robot"].data.root_link_lin_vel_b[:, 0] = torch.tensor(
        [0.0, 0.1, 0.1, 0.1]
    )
    env.scene["robot"].data.root_link_ang_vel_b[:, 2] = torch.tensor(
        [0.0, 0.2, 0.2, 0.2]
    )
    assert lateral_forward_drift_l2(env).tolist() == pytest.approx([0.0, 1.0, 0.0, 0.0])
    assert lateral_yaw_rate_l2(env).tolist() == pytest.approx([0.0, 4.0, 0.0, 0.0])


def test_segment_rewards_reject_body_frame_progress_that_turns_away_from_requested_side():
    env = fake_env(3)
    term = SimpleNamespace(
        segment_yaw=torch.zeros(3), segment_position=torch.zeros(3, 3)
    )
    commands = torch.tensor([[0.0, 0.3, 0.0]]).repeat(3, 1)
    env.command_manager = SimpleNamespace(
        get_command=lambda _: commands, get_term=lambda _: term
    )
    data = env.scene["robot"].data
    # Same apparent +vy in each rotated trunk: initial-world progress is positive,
    # perpendicular, then backwards. An arc cannot keep earning full progress.
    data.root_link_lin_vel_w = torch.tensor(
        [[0.0, 0.3, 0.0], [-0.3, 0.0, 0.0], [0.0, -0.3, 0.0]]
    )
    data.root_link_lin_vel_b[:] = torch.tensor([0.0, 0.3, 0.0])
    assert lateral_velocity_progress(env).tolist() == pytest.approx([1.0, 0.0, -1.0])
    data.root_link_quat_w = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.9961947, 0, 0, 0.0871557],
            [0.9848078, 0, 0, 0.1736482],
        ]
    )
    assert lateral_heading_drift_l2(env).tolist() == pytest.approx([0, 1, 4], abs=1e-5)
    data.root_link_pos_w = torch.tensor(
        [[0.0, 0.5, 0.0], [0.05, 0.5, 0.0], [0.1, 0.5, 0.0]]
    )
    assert lateral_path_drift_l2(env).tolist() == pytest.approx([0, 1, 4])


def test_plain_ppo_warm_start_recipe_never_downloads_an_expert():
    alg = asdict(MicroduckLateralRlCfg.algorithm)
    assert alg["class_name"] == "PPO" and "bc_cfg" not in alg
    assert alg["symmetry_cfg"] is None
    assert MicroduckLateralRlCfg.actor.hidden_dims == (512, 256, 128)
    assert MicroduckLateralRlCfg.actor.distribution_cfg["init_std"] == 0.05
    assert MicroduckLateralRlCfg.logger == "tensorboard"
    assert MicroduckLateralRlCfg.upload_model is False


@pytest.mark.parametrize("lateral,longitudinal", [(-0.1, 0.1), (0.8, 0.3), (0.1, -1.0)])
def test_invalid_sampling_fractions_fail_early(lateral, longitudinal):
    with pytest.raises(ValueError):
        command_families(torch.empty(0), lateral, longitudinal)

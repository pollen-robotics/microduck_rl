from types import SimpleNamespace

import torch

from mjlab_microduck.tasks import mdp as microduck_mdp


class _Scene:
    def __init__(self, robot):
        self._robot = robot
        self.terrain = SimpleNamespace(env_origins=torch.zeros(1, 3))
        self.sensors = {
            name: object()
            for name in (
                "robot_ground_contact",
                "head_ground_contact",
                "legs_ground_contact",
                "trunk_ground_contact",
            )
        }

    def __getitem__(self, name):
        assert name == "robot"
        return self._robot


def test_lip_commitment_requires_delayed_spatial_hold(monkeypatch):
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_link_pos_w=torch.tensor([[0.620, 0.0, 0.150]]),
            root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            root_link_lin_vel_w=torch.zeros(1, 3),
        )
    )
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        episode_length_buf=torch.tensor([1]),
        scene=_Scene(robot),
        test_contact=torch.tensor([False]),
    )

    def fake_contact_masks(current_env, _sensor_name, **_kwargs):
        contact = current_env.test_contact[:, None]
        return contact, torch.zeros_like(contact), torch.zeros(1, 1, 3)

    monkeypatch.setattr(
        microduck_mdp, "_standard_stair_contact_masks", fake_contact_masks
    )

    assert microduck_mdp.stair_contact_lip_commitment(env).item() == 0.0

    env.episode_length_buf[:] = 3
    env.test_contact[:] = True
    assert microduck_mdp.stair_contact_lip_commitment(env).item() == 0.0
    env.episode_length_buf[:] = 4
    assert microduck_mdp.stair_contact_lip_commitment(env).item() == 0.0
    assert env._stair_lip_commitment_armed.item() is True

    env.episode_length_buf[:] = 5
    env.test_contact[:] = False
    robot.data.root_link_pos_w[:] = torch.tensor([[0.625, 0.0, 0.155]])
    robot.data.root_link_lin_vel_w[:] = torch.tensor([[0.06, 0.0, 0.09]])
    assert microduck_mdp.stair_contact_lip_commitment(env).item() == 0.0
    assert env._stair_lip_commitment_impulse_latched.item() is True

    robot.data.root_link_lin_vel_w[:] = 0.0
    for episode_step in (6, 7, 8):
        env.episode_length_buf[:] = episode_step
        assert microduck_mdp.stair_contact_lip_commitment(env).item() == 0.0

    robot.data.root_link_pos_w[:] = torch.tensor([[0.650, 0.0, 0.180]])
    env.episode_length_buf[:] = 9
    assert microduck_mdp.stair_contact_lip_commitment(env).item() == 0.0
    env.episode_length_buf[:] = 10
    payout = microduck_mdp.stair_contact_lip_commitment(env)

    assert torch.isclose(payout, torch.tensor([0.825])).item()
    assert env._stair_lip_commitment_latched.item() is True
    assert microduck_mdp.stair_contact_lip_commitment(env).item() == 0.0


def test_lip_commitment_does_not_arm_from_an_already_cleared_reset(monkeypatch):
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_link_pos_w=torch.tensor([[0.670, 0.0, 0.180]]),
            root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            root_link_lin_vel_w=torch.tensor([[0.20, 0.0, 0.20]]),
        )
    )
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        episode_length_buf=torch.tensor([3]),
        scene=_Scene(robot),
        test_contact=torch.tensor([True]),
    )

    def fake_contact_masks(current_env, _sensor_name, **_kwargs):
        contact = current_env.test_contact[:, None]
        return contact, torch.zeros_like(contact), torch.zeros(1, 1, 3)

    monkeypatch.setattr(
        microduck_mdp, "_standard_stair_contact_masks", fake_contact_masks
    )

    assert microduck_mdp.stair_contact_lip_commitment(env).item() == 0.0
    env.episode_length_buf[:] = 4
    assert microduck_mdp.stair_contact_lip_commitment(env).item() == 0.0
    assert env._stair_lip_commitment_armed.item() is False


def test_lip_checkpoint_pays_signed_joint_progress_and_latches_target(monkeypatch):
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_link_pos_w=torch.tensor([[0.560, 0.0, 0.110]]),
            root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            root_link_lin_vel_w=torch.zeros(1, 3),
        )
    )
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        step_dt=0.02,
        episode_length_buf=torch.tensor([3]),
        scene=_Scene(robot),
        test_contact=torch.tensor([True]),
    )

    def fake_contact_masks(current_env, _sensor_name, **_kwargs):
        contact = current_env.test_contact[:, None]
        return contact, torch.zeros_like(contact), torch.zeros(1, 1, 3)

    monkeypatch.setattr(
        microduck_mdp, "_standard_stair_contact_masks", fake_contact_masks
    )

    assert microduck_mdp.stair_contact_lip_checkpoint_potential(env).item() == 0.0
    env.episode_length_buf[:] = 4
    assert microduck_mdp.stair_contact_lip_checkpoint_potential(env).item() == 0.0
    assert env._stair_lip_checkpoint_contact_qualified.item() is True
    assert env._stair_lip_checkpoint_armed.item() is False

    # Contact alone only qualifies the trajectory; the A24 physical impulse is
    # the prerequisite that opens A25 shaping.
    env.test_contact[:] = False
    env.episode_length_buf[:] = 5
    assert microduck_mdp.stair_contact_lip_checkpoint_potential(env).item() == 0.0
    assert env._stair_lip_checkpoint_armed.item() is False

    env._stair_lip_commitment_impulse_latched = torch.tensor([True])
    env.episode_length_buf[:] = 6
    assert microduck_mdp.stair_contact_lip_checkpoint_potential(env).item() == 0.0
    assert env._stair_lip_checkpoint_armed.item() is True

    robot.data.root_link_pos_w[:] = torch.tensor([[0.610, 0.0, 0.145]])
    env.episode_length_buf[:] = 7
    positive = microduck_mdp.stair_contact_lip_checkpoint_potential(env)
    assert positive.item() > 0.0
    assert env._stair_lip_checkpoint_progress_latched.item() is True

    robot.data.root_link_pos_w[:] = torch.tensor([[0.590, 0.0, 0.125]])
    env.episode_length_buf[:] = 8
    assert microduck_mdp.stair_contact_lip_checkpoint_potential(env).item() < 0.0

    robot.data.root_link_pos_w[:] = torch.tensor([[0.640, 0.21, 0.160]])
    env.episode_length_buf[:] = 9
    assert microduck_mdp.stair_contact_lip_checkpoint_potential(env).item() < 0.0

    # Re-entering repays only what leaving the corridor removed.
    robot.data.root_link_pos_w[:] = torch.tensor([[0.640, 0.0, 0.160]])
    env.episode_length_buf[:] = 10
    assert microduck_mdp.stair_contact_lip_checkpoint_potential(env).item() > 0.0

    robot.data.root_link_pos_w[:] = torch.tensor([[0.666, 0.0, 0.176]])
    env.episode_length_buf[:] = 11
    microduck_mdp.stair_contact_lip_checkpoint_potential(env)
    env.episode_length_buf[:] = 12
    microduck_mdp.stair_contact_lip_checkpoint_potential(env)
    assert env._stair_lip_checkpoint_target_latched.item() is True
    assert microduck_mdp.stair_contact_lip_checkpoint_potential(env).item() == 0.0


def test_lip_checkpoint_bypass_repays_progress_before_latching(monkeypatch):
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_link_pos_w=torch.tensor([[0.620, 0.0, 0.150]]),
            root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            root_link_lin_vel_w=torch.zeros(1, 3),
        )
    )
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        step_dt=0.02,
        episode_length_buf=torch.tensor([3]),
        scene=_Scene(robot),
        test_contact=torch.tensor([True]),
        _stair_lip_commitment_impulse_latched=torch.tensor([True]),
    )

    def fake_contact_masks(current_env, _sensor_name, **_kwargs):
        contact = current_env.test_contact[:, None]
        return contact, torch.zeros_like(contact), torch.zeros(1, 1, 3)

    monkeypatch.setattr(
        microduck_mdp, "_standard_stair_contact_masks", fake_contact_masks
    )
    microduck_mdp.stair_contact_lip_checkpoint_potential(env)
    env.episode_length_buf[:] = 4
    microduck_mdp.stair_contact_lip_checkpoint_potential(env)
    assert env._stair_lip_checkpoint_armed.item() is True

    robot.data.root_link_pos_w[:] = torch.tensor([[0.650, 0.0, 0.165]])
    env.test_contact[:] = False
    env.episode_length_buf[:] = 5
    assert microduck_mdp.stair_contact_lip_checkpoint_potential(env).item() > 0.0

    robot.data.root_link_pos_w[:] = torch.tensor([[0.670, 0.37, 0.170]])
    env.episode_length_buf[:] = 6
    assert microduck_mdp.stair_contact_lip_checkpoint_potential(env).item() < 0.0
    assert env._stair_lip_checkpoint_bypass_latched.item() is True
    assert microduck_mdp.stair_contact_lip_checkpoint_potential(env).item() == 0.0


def test_lip_checkpoint_does_not_arm_from_target_reset(monkeypatch):
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_link_pos_w=torch.tensor([[0.670, 0.0, 0.180]]),
            root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            root_link_lin_vel_w=torch.zeros(1, 3),
        )
    )
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        step_dt=0.02,
        episode_length_buf=torch.tensor([3]),
        scene=_Scene(robot),
        test_contact=torch.tensor([True]),
    )

    def fake_contact_masks(current_env, _sensor_name, **_kwargs):
        contact = current_env.test_contact[:, None]
        return contact, torch.zeros_like(contact), torch.zeros(1, 1, 3)

    monkeypatch.setattr(
        microduck_mdp, "_standard_stair_contact_masks", fake_contact_masks
    )
    microduck_mdp.stair_contact_lip_checkpoint_potential(env)
    env.episode_length_buf[:] = 4
    microduck_mdp.stair_contact_lip_checkpoint_potential(env)
    assert env._stair_lip_checkpoint_armed.item() is False


def test_coupled_frontier_pays_new_min_axis_progress_and_impulse_gated_target(
    monkeypatch,
):
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_link_pos_w=torch.tensor([[0.620, 0.0, 0.145]]),
            root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            root_link_lin_vel_w=torch.zeros(1, 3),
        )
    )
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        step_dt=0.02,
        episode_length_buf=torch.tensor([3]),
        scene=_Scene(robot),
        test_contact=torch.tensor([True]),
        _stair_lip_commitment_impulse_latched=torch.tensor([False]),
    )

    def fake_contact_masks(current_env, _sensor_name, **_kwargs):
        contact = current_env.test_contact[:, None]
        return contact, torch.zeros_like(contact), torch.zeros(1, 1, 3)

    monkeypatch.setattr(
        microduck_mdp, "_standard_stair_contact_masks", fake_contact_masks
    )
    assert microduck_mdp.stair_coupled_frontier_collocation(env).item() == 0.0
    env.episode_length_buf[:] = 4
    assert microduck_mdp.stair_coupled_frontier_collocation(env).item() == 0.0
    assert env._stair_coupled_frontier_armed.item() is True

    env.test_contact[:] = False
    robot.data.root_link_pos_w[:] = torch.tensor([[0.6425, 0.0, 0.160]])
    env.episode_length_buf[:] = 5
    assert torch.isclose(
        microduck_mdp.stair_coupled_frontier_collocation(env),
        torch.tensor([10.0]),
    ).item()
    env._stair_lip_commitment_impulse_latched[:] = True
    env.episode_length_buf[:] = 6
    assert microduck_mdp.stair_coupled_frontier_collocation(env).item() == 0.0
    assert env._stair_coupled_frontier_gain10_latched.item() is True

    robot.data.root_link_pos_w[:] = torch.tensor([[0.630, 0.0, 0.150]])
    env.episode_length_buf[:] = 7
    assert microduck_mdp.stair_coupled_frontier_collocation(env).item() == 0.0
    robot.data.root_link_pos_w[:] = torch.tensor([[0.6425, 0.0, 0.160]])
    env.episode_length_buf[:] = 8
    assert microduck_mdp.stair_coupled_frontier_collocation(env).item() == 0.0

    robot.data.root_link_pos_w[:] = torch.tensor([[0.665, 0.0, 0.175]])
    env.episode_length_buf[:] = 9
    assert torch.isclose(
        microduck_mdp.stair_coupled_frontier_collocation(env),
        torch.tensor([10.0]),
    ).item()
    env.episode_length_buf[:] = 10
    assert torch.isclose(
        microduck_mdp.stair_coupled_frontier_collocation(env),
        torch.tensor([200.0]),
    ).item()
    assert env._stair_coupled_frontier_target_latched.item() is True
    assert microduck_mdp.stair_coupled_frontier_collocation(env).item() == 0.0


def test_coupled_frontier_bypass_repays_only_policy_created_gain(monkeypatch):
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_link_pos_w=torch.tensor([[0.630, 0.0, 0.150]]),
            root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            root_link_lin_vel_w=torch.zeros(1, 3),
        )
    )
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        step_dt=0.02,
        episode_length_buf=torch.tensor([3]),
        scene=_Scene(robot),
        test_contact=torch.tensor([True]),
        _stair_lip_commitment_impulse_latched=torch.tensor([True]),
    )

    def fake_contact_masks(current_env, _sensor_name, **_kwargs):
        contact = current_env.test_contact[:, None]
        return contact, torch.zeros_like(contact), torch.zeros(1, 1, 3)

    monkeypatch.setattr(
        microduck_mdp, "_standard_stair_contact_masks", fake_contact_masks
    )
    microduck_mdp.stair_coupled_frontier_collocation(env)
    env.episode_length_buf[:] = 4
    microduck_mdp.stair_coupled_frontier_collocation(env)

    env.test_contact[:] = False
    robot.data.root_link_pos_w[:] = torch.tensor([[0.650, 0.0, 0.165]])
    env.episode_length_buf[:] = 5
    gain = microduck_mdp.stair_coupled_frontier_collocation(env)
    assert gain.item() > 0.0

    robot.data.root_link_pos_w[:] = torch.tensor([[0.670, 0.37, 0.180]])
    env.episode_length_buf[:] = 6
    repayment = microduck_mdp.stair_coupled_frontier_collocation(env)
    assert torch.isclose(repayment, -gain).item()
    assert env._stair_coupled_frontier_bypass_latched.item() is True
    assert microduck_mdp.stair_coupled_frontier_collocation(env).item() == 0.0

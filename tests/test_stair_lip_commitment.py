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

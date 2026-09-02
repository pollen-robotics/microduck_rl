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


def test_contact_loaded_release_pays_only_after_armed_motion(monkeypatch):
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
        test_face=torch.tensor([False]),
        test_tread=torch.tensor([False]),
    )

    def fake_contact_masks(current_env, _sensor_name, **_kwargs):
        face = current_env.test_face[:, None]
        tread = current_env.test_tread[:, None]
        return face, tread, torch.zeros(1, 1, 3)

    monkeypatch.setattr(
        microduck_mdp, "_standard_stair_contact_masks", fake_contact_masks
    )

    assert microduck_mdp.stair_contact_loaded_release(env).item() == 0.0

    env.episode_length_buf[:] = 3
    env.test_face[:] = True
    assert microduck_mdp.stair_contact_loaded_release(env).item() == 0.0

    env.episode_length_buf[:] = 4
    assert microduck_mdp.stair_contact_loaded_release(env).item() == 0.0
    assert env._stair_contact_release_armed.item() is True

    env.episode_length_buf[:] = 5
    env.test_face[:] = False
    robot.data.root_link_pos_w[:] = torch.tensor([[0.640, 0.0, 0.165]])
    robot.data.root_link_lin_vel_w[:] = torch.tensor([[0.10, 0.0, 0.20]])
    payout = microduck_mdp.stair_contact_loaded_release(env)

    assert torch.isclose(payout, torch.tensor([0.54])).item()
    assert env._stair_contact_loaded_release_latched.item() is True
    assert microduck_mdp.stair_contact_loaded_release(env).item() == 0.0

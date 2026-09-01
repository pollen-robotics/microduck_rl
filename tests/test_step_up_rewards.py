from types import SimpleNamespace

import torch

from mjlab_microduck.tasks.mdp import (
    _step_up_success_mask,
    step_up_foot_milestone,
    step_up_out_of_bounds,
)


class _Scene(dict):
    def __init__(self, robot, terrain):
        super().__init__(robot=robot)
        self.terrain = terrain


def _fake_env(feet: torch.Tensor, episode_lengths: torch.Tensor):
    robot = SimpleNamespace(data=SimpleNamespace(site_pos_w=feet))
    terrain = SimpleNamespace(env_origins=torch.zeros(feet.shape[0], 3))
    return SimpleNamespace(
        scene=_Scene(robot, terrain),
        episode_length_buf=episode_lengths,
    )


def _feet_cfg():
    return SimpleNamespace(name="robot", site_ids=[0, 1])


def test_foot_milestone_is_paid_only_once_per_episode():
    feet = torch.tensor([[[0.34, 0.0, 0.025], [0.20, 0.0, 0.0]]])
    env = _fake_env(feet, torch.tensor([2]))
    kwargs = {
        "min_foot_forward": 0.32,
        "min_foot_height": 0.018,
        "required_feet": 1,
        "state_key": "_one_foot_seen",
        "feet_cfg": _feet_cfg(),
    }
    assert step_up_foot_milestone(env, **kwargs).tolist() == [1.0]
    assert step_up_foot_milestone(env, **kwargs).tolist() == [0.0]

    env.episode_length_buf[:] = 0
    assert step_up_foot_milestone(env, **kwargs).tolist() == [1.0]


def test_two_foot_milestone_requires_both_feet_past_and_above_edge():
    feet = torch.tensor(
        [
            [[0.34, 0.0, 0.025], [0.20, 0.0, 0.0]],
            [[0.34, 0.0, 0.025], [0.35, 0.0, 0.024]],
        ]
    )
    env = _fake_env(feet, torch.tensor([2, 2]))
    reward = step_up_foot_milestone(
        env,
        min_foot_forward=0.32,
        min_foot_height=0.018,
        required_feet=2,
        state_key="_two_feet_seen",
        feet_cfg=_feet_cfg(),
    )
    assert reward.tolist() == [0.0, 1.0]


def test_strict_success_rejects_side_bypass_and_requires_both_feet_on_top():
    data = SimpleNamespace(
        root_link_pos_w=torch.tensor(
            [[0.55, 0.00, 0.13], [0.55, 0.45, 0.13], [0.55, 0.00, 0.13]]
        ),
        root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 3),
        site_pos_w=torch.tensor(
            [
                [[0.40, 0.05, 0.025], [0.41, -0.05, 0.025]],
                [[0.60, 0.45, 0.000], [0.61, 0.52, 0.000]],
                [[0.40, 0.05, 0.025], [0.30, -0.05, 0.000]],
            ]
        ),
    )
    robot = SimpleNamespace(data=data)
    terrain = SimpleNamespace(env_origins=torch.zeros(3, 3))
    env = SimpleNamespace(scene=_Scene(robot, terrain))
    result = _step_up_success_mask(
        env,
        min_forward_distance=0.36,
        min_trunk_height=0.105,
        min_upright_cos=0.8,
        max_lateral_offset=0.30,
        min_foot_forward=0.34,
        min_foot_height=0.018,
        asset_cfg=SimpleNamespace(name="robot"),
        feet_cfg=SimpleNamespace(name="robot", site_ids=[0, 1]),
    )
    assert result.tolist() == [True, False, False]
    assert step_up_out_of_bounds(
        env,
        max_lateral_offset=0.34,
        asset_cfg=SimpleNamespace(name="robot"),
    ).tolist() == [False, True, False]

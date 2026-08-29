from types import SimpleNamespace

import torch
from tensordict import TensorDict

from mjlab_microduck.policies import HardStairHandoffPolicy


class RecordingActor(torch.nn.Module):
    def __init__(self, action_value: float):
        super().__init__()
        self.action_value = action_value
        self.last_actor_observations: torch.Tensor | None = None

    def forward(self, observations: TensorDict) -> torch.Tensor:
        self.last_actor_observations = observations["actor"].clone()
        return torch.full(
            (observations.batch_size[0], 2),
            self.action_value,
            dtype=observations["actor"].dtype,
        )


class FakeScene:
    def __init__(self, robot: SimpleNamespace, terrain: SimpleNamespace):
        self.robot = robot
        self.terrain = terrain

    def __getitem__(self, key: str) -> SimpleNamespace:
        return getattr(self, key)


def _fake_env(num_envs: int = 2) -> SimpleNamespace:
    robot_data = SimpleNamespace(root_link_pos_w=torch.zeros(num_envs, 3))
    terrain = SimpleNamespace(env_origins=torch.zeros(num_envs, 3))
    scene = FakeScene(SimpleNamespace(data=robot_data), terrain)
    return SimpleNamespace(
        num_envs=num_envs,
        device=torch.device("cpu"),
        episode_length_buf=torch.full((num_envs,), 10, dtype=torch.long),
        scene=scene,
    )


def _observations(num_envs: int = 2) -> TensorDict:
    actor = torch.arange(num_envs * 61, dtype=torch.float32).reshape(num_envs, 61)
    return TensorDict({"actor": actor}, batch_size=[num_envs])


def test_handoff_is_per_environment_and_latches_until_reset() -> None:
    env = _fake_env()
    walker = RecordingActor(1.0)
    specialist = RecordingActor(2.0)
    policy = HardStairHandoffPolicy(walker, specialist, env)
    observations = _observations()

    env.scene["robot"].data.root_link_pos_w[:, 0] = torch.tensor([0.40, 0.60])
    actions = policy(observations)
    assert torch.equal(actions[:, 0], torch.tensor([1.0, 2.0]))
    assert policy.specialist_latched.tolist() == [False, True]

    env.scene["robot"].data.root_link_pos_w[:, 0] = torch.tensor([0.57, 0.20])
    actions = policy(observations)
    assert torch.equal(actions[:, 0], torch.tensor([2.0, 2.0]))
    assert policy.handoff_count == 2

    env.episode_length_buf[1] = 1
    actions = policy(observations)
    assert torch.equal(actions[:, 0], torch.tensor([2.0, 1.0]))
    assert policy.specialist_latched.tolist() == [True, False]


def test_walker_cues_are_zeroed_without_changing_specialist_input() -> None:
    env = _fake_env()
    walker = RecordingActor(1.0)
    specialist = RecordingActor(2.0)
    policy = HardStairHandoffPolicy(walker, specialist, env)
    observations = _observations()
    original = observations["actor"].clone()

    policy(observations)

    assert walker.last_actor_observations is not None
    assert specialist.last_actor_observations is not None
    assert torch.count_nonzero(walker.last_actor_observations[:, 55:61]) == 0
    assert torch.equal(specialist.last_actor_observations, original)
    assert torch.equal(observations["actor"], original)

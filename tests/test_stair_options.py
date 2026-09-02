from types import SimpleNamespace

import torch
from tensordict import TensorDict

from mjlab_microduck.policies import StairOptionPolicy, StairRouteEstimate


class ConstantActor(torch.nn.Module):
    def __init__(self, value: float, action_dim: int = 14):
        super().__init__()
        self.value = value
        self.action_dim = action_dim
        self.last_actor_observations: torch.Tensor | None = None

    def forward(self, observations: TensorDict) -> torch.Tensor:
        self.last_actor_observations = observations["actor"].clone()
        return torch.full(
            (observations.batch_size[0], self.action_dim),
            self.value,
            dtype=observations["actor"].dtype,
        )


class FixedRouteEstimator:
    def __init__(self, num_envs: int):
        self.estimate_value = StairRouteEstimate(
            distance_to_next_face_m=torch.full((num_envs,), 0.10),
            lateral_error_m=torch.zeros(num_envs),
            heading_error_rad=torch.zeros(num_envs),
            forward_velocity_mps=torch.full((num_envs,), 0.22),
            upright_score=torch.ones(num_envs),
            non_foot_contact=torch.zeros(num_envs, dtype=torch.bool),
            finite=torch.ones(num_envs, dtype=torch.bool),
        )

    def estimate(self) -> StairRouteEstimate:
        return self.estimate_value


class FakeScene:
    def __init__(self, num_envs: int):
        robot_data = SimpleNamespace(root_link_pos_w=torch.zeros(num_envs, 3))
        self.robot = SimpleNamespace(data=robot_data)
        self.terrain = SimpleNamespace(env_origins=torch.zeros(num_envs, 3))
        self.sensors = {
            "head_ground_contact": SimpleNamespace(
                data=SimpleNamespace(
                    found=torch.zeros(num_envs, 1, dtype=torch.bool),
                    pos=torch.zeros(num_envs, 1, 3),
                )
            )
        }

    def __getitem__(self, key: str) -> SimpleNamespace:
        return getattr(self, key)


def _fake_env(num_envs: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        num_envs=num_envs,
        device=torch.device("cpu"),
        episode_length_buf=torch.full((num_envs,), 10, dtype=torch.long),
        scene=FakeScene(num_envs),
        _stair_first_tread_secured_latched=torch.zeros(
            num_envs, dtype=torch.bool
        ),
    )


def _observations(num_envs: int = 1) -> TensorDict:
    return TensorDict(
        {"actor": torch.arange(num_envs * 61, dtype=torch.float32).reshape(
            num_envs, 61
        )},
        batch_size=[num_envs],
    )


def _policy(env: SimpleNamespace, **kwargs) -> StairOptionPolicy:
    kwargs.setdefault("handoff_guard_frames", 1)
    return StairOptionPolicy(
        ConstantActor(1.0),
        ConstantActor(2.0),
        ConstantActor(3.0),
        ConstantActor(4.0),
        env,
        route_estimator=FixedRouteEstimator(env.num_envs),
        **kwargs,
    )


def test_walker_requires_four_consecutive_valid_handoff_frames() -> None:
    env = _fake_env()
    policy = _policy(
        env,
        handoff_guard_frames=4,
        option_blend_steps=4,
        launch_min_steps=100,
        launch_max_steps=100,
    )
    observations = _observations()

    values = [float(policy(observations)[0, 0]) for _ in range(4)]

    assert values == [1.0, 1.0, 1.0, 1.25]
    assert policy.transition_counts[0, 0].item() == 1


def test_walker_to_launch_blends_over_exactly_four_frames() -> None:
    env = _fake_env()
    policy = _policy(
        env,
        option_blend_steps=4,
        launch_min_steps=100,
        launch_max_steps=100,
    )
    observations = _observations()

    values = [float(policy(observations)[0, 0]) for _ in range(5)]

    assert values == [1.25, 1.5, 1.75, 2.0, 2.0]
    assert policy.phase.tolist() == [StairOptionPolicy.LAUNCH]
    assert policy.transition_counts[0, :2].tolist() == [1, 1]


def test_launch_zeroes_all_commands_and_preserves_14d_actions() -> None:
    env = _fake_env()
    policy = _policy(
        env,
        start_in_launch=True,
        option_blend_steps=0,
        launch_min_steps=3,
        launch_max_steps=3,
    )
    observations = _observations()

    actions = policy(observations)

    assert actions.shape == (1, 14)
    assert torch.count_nonzero(
        policy.launch.last_actor_observations[:, 48:61]
    ) == 0
    assert torch.equal(observations["actor"], _observations()["actor"])


def test_launch_mantle_recover_transitions_are_ordered_and_reset() -> None:
    env = _fake_env()
    env.scene["robot"].data.root_link_pos_w[0] = torch.tensor(
        [0.61, 0.0, 0.15]
    )
    policy = _policy(
        env,
        start_in_launch=True,
        option_blend_steps=4,
        launch_min_steps=2,
        launch_max_steps=10,
    )
    observations = _observations()

    values = [float(policy(observations)[0, 0]) for _ in range(7)]
    assert values == [2.0, 2.0, 2.25, 2.5, 2.75, 3.0, 3.0]
    assert policy.phase.tolist() == [StairOptionPolicy.MANTLE]
    assert policy.transition_counts[0, 2:4].tolist() == [1, 1]

    env._stair_first_tread_secured_latched[:] = True
    assert float(policy(observations)[0, 0]) == 4.0
    assert policy.phase.tolist() == [StairOptionPolicy.RECOVER]

    env.episode_length_buf[:] = 1
    env._stair_first_tread_secured_latched[:] = False
    assert float(policy(observations)[0, 0]) == 2.0
    assert policy.phase.tolist() == [StairOptionPolicy.LAUNCH]


def test_launch_timeout_fails_instead_of_skipping_to_mantle() -> None:
    env = _fake_env()
    policy = _policy(
        env,
        start_in_launch=True,
        option_blend_steps=0,
        launch_min_steps=2,
        launch_max_steps=3,
        mantle_root_x_m=1.0,
        mantle_root_z_m=1.0,
    )
    observations = _observations()

    values = [float(policy(observations)[0, 0]) for _ in range(4)]

    assert values == [2.0, 2.0, 2.0, 4.0]
    assert policy.phase.tolist() == [StairOptionPolicy.FAILED]
    assert policy.transition_counts[0, 2].item() == 0
    assert policy.transition_counts[0, 5].item() == 1


def test_floor_head_contact_does_not_open_mantle_gate() -> None:
    env = _fake_env()
    sensor = env.scene.sensors["head_ground_contact"].data
    sensor.found[:] = True
    sensor.pos[0, 0] = torch.tensor([0.70, 0.0, 0.0])
    policy = _policy(
        env,
        start_in_launch=True,
        option_blend_steps=0,
        launch_min_steps=0,
        launch_max_steps=10,
        mantle_root_x_m=1.0,
        mantle_root_z_m=1.0,
    )

    assert float(policy(_observations())[0, 0]) == 2.0
    assert policy.phase.tolist() == [StairOptionPolicy.LAUNCH]


def test_option_policy_rejects_non_61d_input_and_non_14d_actions() -> None:
    env = _fake_env()
    policy = _policy(env, start_in_launch=True)
    bad_observations = TensorDict(
        {"actor": torch.zeros(1, 62)}, batch_size=[1]
    )
    with __import__("pytest").raises(ValueError, match="exact 61D"):
        policy(bad_observations)

    policy = StairOptionPolicy(
        ConstantActor(1.0),
        ConstantActor(2.0),
        ConstantActor(3.0),
        ConstantActor(4.0, action_dim=2),
        env,
        route_estimator=FixedRouteEstimator(1),
        start_in_launch=True,
    )
    with __import__("pytest").raises(ValueError, match="recover actor"):
        policy(_observations())

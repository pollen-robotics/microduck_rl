import math
from types import SimpleNamespace

import torch
from tensordict import TensorDict

from mjlab_microduck.policies import (
    OFFICIAL_WALKER_SHA256,
    HardStairHandoffPolicy,
    StairApproachSupervisor,
    resolve_official_walker_checkpoint,
)


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
    def __init__(
        self,
        robot: SimpleNamespace,
        terrain: SimpleNamespace,
        sensors: dict[str, SimpleNamespace],
    ):
        self.robot = robot
        self.terrain = terrain
        self.sensors = sensors

    def __getitem__(self, key: str) -> SimpleNamespace:
        return getattr(self, key)


def _fake_env(num_envs: int = 2) -> SimpleNamespace:
    robot_data = SimpleNamespace(
        root_link_pos_w=torch.zeros(num_envs, 3),
        root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(
            num_envs, 1
        ),
        root_link_lin_vel_b=torch.tensor([[0.22, 0.0, 0.0]]).repeat(
            num_envs, 1
        ),
    )
    terrain = SimpleNamespace(env_origins=torch.zeros(num_envs, 3))
    sensors = {
        name: SimpleNamespace(
            data=SimpleNamespace(found=torch.zeros(num_envs, 1, dtype=torch.bool))
        )
        for name in (
            "head_ground_contact",
            "trunk_ground_contact",
            "legs_ground_contact",
        )
    }
    scene = FakeScene(SimpleNamespace(data=robot_data), terrain, sensors)
    return SimpleNamespace(
        num_envs=num_envs,
        device=torch.device("cpu"),
        common_step_counter=10,
        step_dt=0.02,
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
    policy = HardStairHandoffPolicy(walker, specialist, env, blend_steps=0)
    observations = _observations()

    env.scene["robot"].data.root_link_pos_w[:, 0] = torch.tensor([0.40, 0.56])
    actions = policy(observations)
    assert torch.equal(actions[:, 0], torch.tensor([1.0, 2.0]))
    assert policy.specialist_latched.tolist() == [False, True]

    env.scene["robot"].data.root_link_pos_w[:, 0] = torch.tensor([0.56, 0.20])
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
    assert torch.allclose(
        walker.last_actor_observations[:, 48:51],
        torch.tensor([[0.30, 0.0, 0.0], [0.30, 0.0, 0.0]]),
    )
    assert torch.equal(specialist.last_actor_observations, original)
    assert torch.equal(observations["actor"], original)


def test_handoff_can_blend_over_bounded_control_frames() -> None:
    env = _fake_env(num_envs=1)
    walker = RecordingActor(1.0)
    specialist = RecordingActor(3.0)
    policy = HardStairHandoffPolicy(
        walker,
        specialist,
        env,
        blend_steps=4,
    )
    observations = _observations(num_envs=1)

    env.scene["robot"].data.root_link_pos_w[:, 0] = 0.56
    values = [float(policy(observations)[0, 0]) for _ in range(4)]
    assert values == [1.5, 2.0, 2.5, 3.0]
    assert policy.handoff_count == 1

    env.episode_length_buf[:] = 1
    env.scene["robot"].data.root_link_pos_w[:, 0] = 0.20
    assert float(policy(observations)[0, 0]) == 1.0
    assert int(policy.blend_progress[0]) == 0


def test_guard_rejects_misaligned_or_unsafe_handoffs() -> None:
    env = _fake_env(num_envs=4)
    walker = RecordingActor(1.0)
    specialist = RecordingActor(2.0)
    policy = HardStairHandoffPolicy(walker, specialist, env)
    observations = _observations(num_envs=4)
    robot = env.scene["robot"].data
    robot.root_link_pos_w[:, 0] = 0.56
    robot.root_link_pos_w[0, 1] = 0.041
    yaw = torch.deg2rad(torch.tensor(8.1))
    robot.root_link_quat_w[1] = torch.tensor(
        [torch.cos(yaw / 2), 0.0, 0.0, torch.sin(yaw / 2)]
    )
    robot.root_link_lin_vel_b[2, 0] = 0.159
    env.scene.sensors["head_ground_contact"].data.found[3, 0] = True

    policy(observations)

    assert policy.specialist_latched.tolist() == [False, False, False, False]
    assert policy.handoff_count == 0


def test_supervisor_clamps_lateral_and_heading_commands() -> None:
    env = _fake_env(num_envs=2)
    robot = env.scene["robot"].data
    robot.root_link_pos_w[:, 1] = torch.tensor([0.5, -0.5])
    yaw = torch.deg2rad(torch.tensor([30.0, -30.0]))
    robot.root_link_quat_w[:, 0] = torch.cos(yaw / 2)
    robot.root_link_quat_w[:, 3] = torch.sin(yaw / 2)
    policy = HardStairHandoffPolicy(
        RecordingActor(1.0),
        RecordingActor(2.0),
        env,
        approach_supervisor=StairApproachSupervisor(
            lateral_gain=10.0,
            heading_gain=10.0,
            cross_track_heading_gain=10.0,
        ),
    )

    policy(_observations())

    commands = policy.walker.last_actor_observations[:, 48:51]
    expected_vx = math.cos(math.radians(30.0)) * 0.30 - 0.02
    assert torch.allclose(
        commands[:, 0],
        torch.tensor([expected_vx, expected_vx]),
    )
    assert torch.allclose(commands[:, 1], torch.tensor([-0.04, 0.04]))
    assert torch.allclose(commands[:, 2], torch.tensor([-0.20, 0.20]))


def test_official_walker_checkpoint_is_selected_by_hash() -> None:
    repo_root = __import__("pathlib").Path(__file__).resolve().parents[1]
    checkpoint = resolve_official_walker_checkpoint(repo_root)

    assert checkpoint.name == "model_0.pt"
    assert OFFICIAL_WALKER_SHA256.startswith("c17ce5")

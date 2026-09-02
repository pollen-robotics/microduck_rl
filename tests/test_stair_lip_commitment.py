from types import SimpleNamespace

import torch

from mjlab_microduck.tasks import mdp as microduck_mdp


class _Scene:
    def __init__(self, robot):
        self._robot = robot
        self.terrain = SimpleNamespace(env_origins=torch.zeros(1, 3))
        self.sensors = {
            name: SimpleNamespace(
                data=SimpleNamespace(found=torch.zeros(1, 1))
            )
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


def test_stair_route_cues_expose_raw_time_to_go():
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_link_pos_w=torch.tensor([[0.100, 0.0, 0.100]]),
            root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        )
    )
    scene = _Scene(robot)
    scene.terrain.terrain_levels = torch.zeros(1, dtype=torch.long)
    env = SimpleNamespace(
        max_episode_length=150,
        episode_length_buf=torch.tensor([0]),
        scene=scene,
    )
    params = {
        "stair_start_distance": 0.60,
        "goal_distance": 2.0,
        "min_riser_height": 0.17,
        "max_riser_height": 0.17,
        "num_terrain_levels": 1,
        "tread_depth": 0.28,
        "num_steps": 5,
        "include_time_to_go": True,
    }

    reset_cues = microduck_mdp.stair_route_cues(env, **params)
    assert reset_cues.shape == (1, 6)
    assert torch.isclose(reset_cues[0, 3], torch.tensor(1.0)).item()
    assert reset_cues[0, 0].item() < 1.0e-4

    env.episode_length_buf[:] = 75
    halfway_cues = microduck_mdp.stair_route_cues(env, **params)
    assert torch.isclose(halfway_cues[0, 3], torch.tensor(0.5)).item()

    env.episode_length_buf[:] = 150
    final_cues = microduck_mdp.stair_route_cues(env, **params)
    assert final_cues[0, 3].item() == 0.0


def test_terminal_position_objective_only_pays_in_final_window():
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_link_pos_w=torch.tensor([[0.720, 0.0, 0.205]]),
        )
    )
    env = SimpleNamespace(
        step_dt=0.02,
        max_episode_length=150,
        episode_length_buf=torch.tensor([119]),
        scene=_Scene(robot),
    )

    early = microduck_mdp.stair_terminal_position_objective(env)
    assert early.item() == 0.0

    env.episode_length_buf[:] = 125
    at_target = microduck_mdp.stair_terminal_position_objective(env)
    assert torch.isclose(at_target, torch.tensor([1.0 / 0.50])).item()

    robot.data.root_link_pos_w[:] = torch.tensor([[0.620, 0.0, 0.138]])
    partial = microduck_mdp.stair_terminal_position_objective(env)
    assert 0.0 < partial.item() < at_target.item()

    env.episode_length_buf[:] = 150
    robot.data.root_link_pos_w[:] = torch.tensor([[0.720, 0.0, 0.205]])
    assert microduck_mdp.stair_terminal_position_objective(env).item() == 0.0

    env.episode_length_buf[:] = 125
    robot.data.root_link_pos_w[:] = torch.tensor([[0.720, 0.201, 0.205]])
    assert microduck_mdp.stair_terminal_position_objective(env).item() == 0.0

    robot.data.root_link_pos_w[:] = torch.tensor([[float("nan"), 0.0, 0.205]])
    assert microduck_mdp.stair_terminal_position_objective(env).item() == 0.0


def test_terminal_position_objective_integrates_exactly_one_at_target():
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_link_pos_w=torch.tensor([[0.720, 0.0, 0.205]]),
        )
    )
    env = SimpleNamespace(
        step_dt=0.02,
        max_episode_length=150,
        episode_length_buf=torch.tensor([0]),
        scene=_Scene(robot),
    )

    payments = []
    for episode_step in range(151):
        env.episode_length_buf[:] = episode_step
        payments.append(microduck_mdp.stair_terminal_position_objective(env))
    rewards = torch.cat(payments)

    assert torch.count_nonzero(rewards).item() == 25
    assert torch.isclose(rewards.sum() * env.step_dt, torch.tensor(1.0)).item()


def test_delayed_frontier_tiers_require_new_held_post_control_crossing():
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_link_pos_w=torch.tensor([[0.648, 0.0, 0.190]]),
        )
    )
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        episode_length_buf=torch.tensor([1]),
        scene=_Scene(robot),
    )

    assert microduck_mdp.stair_delayed_frontier_tiers(env).item() == 0.0
    robot.data.root_link_pos_w[:] = torch.tensor([[0.666, 0.0, 0.190]])
    env.episode_length_buf[:] = 2
    assert microduck_mdp.stair_delayed_frontier_tiers(env).item() == 0.0
    env.episode_length_buf[:] = 3
    assert microduck_mdp.stair_delayed_frontier_tiers(env).item() == 0.0
    env.episode_length_buf[:] = 4
    assert microduck_mdp.stair_delayed_frontier_tiers(env).item() == 250.0
    env.episode_length_buf[:] = 5
    assert microduck_mdp.stair_delayed_frontier_tiers(env).item() == 0.0


def test_delayed_frontier_tiers_permanently_reject_lateral_bypass():
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_link_pos_w=torch.tensor([[0.640, 0.0, 0.190]]),
        )
    )
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        episode_length_buf=torch.tensor([1]),
        scene=_Scene(robot),
    )

    microduck_mdp.stair_delayed_frontier_tiers(env)
    env.episode_length_buf[:] = 3
    robot.data.root_link_pos_w[:] = torch.tensor([[0.661, 0.37, 0.190]])
    assert microduck_mdp.stair_delayed_frontier_tiers(env).item() == 0.0
    robot.data.root_link_pos_w[:] = torch.tensor([[0.666, 0.0, 0.190]])
    for step in (4, 5, 6):
        env.episode_length_buf[:] = step
        assert microduck_mdp.stair_delayed_frontier_tiers(env).item() == 0.0


def test_reset_snapshot_prelatches_only_milestones_already_satisfied():
    robot = SimpleNamespace(
        indexing=SimpleNamespace(free_joint_q_adr=torch.arange(7)),
        data=SimpleNamespace(
            # Deliberately stale derived position from the previous episode.
            root_link_pos_w=torch.tensor([[0.100, 0.0, 0.100]]),
        )
    )
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        episode_length_buf=torch.tensor([1]),
        scene=_Scene(robot),
        sim=SimpleNamespace(
            data=SimpleNamespace(
                qpos=torch.tensor([[0.640, 0.0, 0.170, 1.0, 0.0, 0.0, 0.0]])
            )
        ),
    )
    x_thresholds = (0.625, 0.640, 0.650)
    z_thresholds = (0.160, 0.170, 0.175)
    microduck_mdp.snapshot_stair_frontier_tier_baseline(
        env,
        torch.tensor([0]),
        x_thresholds=x_thresholds,
        min_height_thresholds=z_thresholds,
    )
    params = {
        "x_thresholds": x_thresholds,
        "tier_rewards": (10.0, 20.0, 40.0),
        "min_height_thresholds": z_thresholds,
        "prelatch_reset_satisfied": True,
    }

    assert microduck_mdp.stair_delayed_frontier_tiers(env, **params).item() == 0.0
    assert env._stair_delayed_frontier_tier_paid.tolist() == [
        [True, True, False]
    ]
    # Forward kinematics now exposes the same reset pose; holding it cannot
    # repay either pre-latched milestone.
    robot.data.root_link_pos_w[:] = torch.tensor([[0.640, 0.0, 0.170]])
    env.episode_length_buf[:] = 3
    assert microduck_mdp.stair_delayed_frontier_tiers(env, **params).item() == 0.0
    env.episode_length_buf[:] = 4
    assert microduck_mdp.stair_delayed_frontier_tiers(env, **params).item() == 0.0

    robot.data.root_link_pos_w[:] = torch.tensor([[0.650, 0.0, 0.175]])
    env.episode_length_buf[:] = 5
    assert microduck_mdp.stair_delayed_frontier_tiers(env, **params).item() == 0.0
    env.episode_length_buf[:] = 6
    assert microduck_mdp.stair_delayed_frontier_tiers(env, **params).item() == 40.0


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


def test_coupled_frontier_prearm_peak_prevents_baseline_depression():
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_link_pos_w=torch.tensor([[0.6425, 0.0, 0.160]]),
            root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            root_link_lin_vel_w=torch.zeros(1, 3),
        )
    )
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        step_dt=0.02,
        episode_length_buf=torch.tensor([1]),
        scene=_Scene(robot),
        _stair_lip_commitment_impulse_latched=torch.tensor([False]),
    )

    assert microduck_mdp.stair_coupled_frontier_collocation(env).item() == 0.0
    robot.data.root_link_pos_w[:] = torch.tensor([[0.620, 0.0, 0.145]])
    env.episode_length_buf[:] = 2
    assert microduck_mdp.stair_coupled_frontier_collocation(env).item() == 0.0
    assert env._stair_coupled_frontier_armed.item() is True

    robot.data.root_link_pos_w[:] = torch.tensor([[0.6425, 0.0, 0.160]])
    env.episode_length_buf[:] = 3
    assert microduck_mdp.stair_coupled_frontier_collocation(env).item() == 0.0

    robot.data.root_link_pos_w[:] = torch.tensor([[0.665, 0.0, 0.175]])
    env.episode_length_buf[:] = 4
    assert torch.isclose(
        microduck_mdp.stair_coupled_frontier_collocation(env),
        torch.tensor([10.0]),
    ).item()

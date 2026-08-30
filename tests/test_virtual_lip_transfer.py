from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_standard_stairs_env_cfg import (
    STANDARD_RISER_HEIGHT,
    STANDARD_STAIR_START_DISTANCE,
    STANDARD_TREAD_DEPTH,
    VIRTUAL_LIP_CURRICULUM_LEVELS,
    VIRTUAL_LIP_MAX_FACE_OFFSET,
    BoxStandardStaircaseTerrainCfg,
    make_microduck_stair_contact_stage_rsi_env_cfg,
    make_microduck_stair_stage15_reverse_rsi_env_cfg,
    make_microduck_stair_stage2_reverse_rsi_env_cfg,
    make_microduck_stair_forward_propagation_rsi_env_cfg,
    make_microduck_stair_virtual_lip_transfer_rsi_env_cfg,
)


def _contact_env(num_envs: int) -> SimpleNamespace:
    return SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        episode_length_buf=torch.ones(num_envs, dtype=torch.long),
        test_face=torch.zeros(num_envs, dtype=torch.bool),
        test_tread=torch.zeros(num_envs, dtype=torch.bool),
        test_force=torch.ones(num_envs),
    )


def _fake_virtual_contact_masks(env, _sensor_name, **_kwargs):
    return (
        env.test_face[:, None],
        env.test_tread[:, None],
        torch.zeros(env.num_envs, 1, 3),
    )


def _fake_virtual_union(env, _sensor_names, **_kwargs):
    return env.test_face, env.test_tread, env.test_force


class _AssetScene:
    def __init__(self, robot, terrain_levels: torch.Tensor):
        self._robot = robot
        self.terrain = SimpleNamespace(
            env_origins=torch.zeros(len(terrain_levels), 3),
            terrain_levels=terrain_levels,
        )

    def __getitem__(self, name):
        assert name == "robot"
        return self._robot


def test_exact_64_env_family_assignment_is_32_16_16_and_exclusive():
    torch.manual_seed(30)
    env = SimpleNamespace(num_envs=64, device="cpu")
    env_ids = torch.arange(64)

    microduck_mdp.assign_stair_state_bank_family(
        env, env_ids, family_weights=(2, 1, 1)
    )

    labels = env._stair_state_bank_family
    assert torch.bincount(labels, minlength=3).tolist() == [32, 16, 16]
    one_hot = torch.nn.functional.one_hot(labels, num_classes=3)
    assert torch.all(one_hot.sum(dim=1) == 1)
    assert torch.equal(
        torch.sort(env_ids).values,
        torch.sort(torch.where(one_hot)[0]).values,
    )


def test_forced_family_assignment_is_exact_for_diagnostic_evaluation():
    env = SimpleNamespace(num_envs=9, device="cpu")
    env_ids = torch.arange(9)

    microduck_mdp.assign_stair_state_bank_family(
        env,
        env_ids,
        family_weights=(2, 1, 1),
        forced_family=1,
    )

    assert env._stair_state_bank_family.tolist() == [1] * 9


def test_virtual_lip_classifier_separates_recessed_face_from_nominal_tread():
    found = torch.ones(1, 2, dtype=torch.bool)
    positions = torch.tensor([[[0.700, 0.0, 0.080], [0.660, 0.0, 0.170]]])
    normals = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])

    face, tread = microduck_mdp.classify_virtual_lip_stair_contacts(
        found,
        positions,
        normals,
        terrain_origins_w=torch.zeros(1, 3),
        physical_face_x=torch.tensor([0.700]),
        nominal_stair_face_x=0.660,
    )

    assert torch.equal(face, torch.tensor([[True, False]]))
    assert torch.equal(tread, torch.tensor([[False, True]]))


def test_virtual_lip_geometry_hardens_only_first_supporting_face():
    def build(cfg, difficulty):
        cfg.size = (5.0, 3.0)
        spec = mujoco.MjSpec()
        spec.worldbody.add_body(name="terrain")
        return cfg.function(difficulty, spec, np.random.default_rng(0))

    virtual = BoxStandardStaircaseTerrainCfg(
        first_riser_face_offset_range=(0.04, 0.0),
        first_riser_face_offset_levels=3,
    )
    easy, medium, hard = (build(virtual, value) for value in (0.0, 0.5, 1.0))
    standard = build(BoxStandardStaircaseTerrainCfg(), 1.0)

    def left_face(output, index):
        geom = output.geometries[index].geom
        return float(geom.pos[0] - geom.size[0] - output.origin[0])

    assert abs(left_face(easy, 2) - 0.700) < 1.0e-9
    assert abs(left_face(medium, 2) - 0.680) < 1.0e-9
    assert abs(left_face(hard, 1) - 0.660) < 1.0e-9
    assert abs(easy.geometries[1].geom.pos[2] + easy.geometries[1].geom.size[2] - 0.170) < 1.0e-9
    assert tuple(hard.geometries[1].geom.pos) == tuple(standard.geometries[1].geom.pos)
    assert tuple(hard.geometries[1].geom.size) == tuple(standard.geometries[1].geom.size)
    easy_upper = [
        (tuple(item.geom.pos), tuple(item.geom.size))
        for item in easy.geometries[3:7]
    ]
    hard_upper = [
        (tuple(item.geom.pos), tuple(item.geom.size))
        for item in hard.geometries[2:6]
    ]
    assert easy_upper == hard_upper


def test_stage1_prelatches_reset_tread_and_pays_only_one_new_contact(monkeypatch):
    monkeypatch.setattr(
        microduck_mdp,
        "_virtual_lip_union_contact_state",
        _fake_virtual_union,
    )
    env = _contact_env(2)
    env.test_tread[:] = torch.tensor([True, False])

    assert torch.equal(
        microduck_mdp.stair_new_tread_contact_after_reset(env),
        torch.zeros(2),
    )
    assert torch.equal(
        env._stair_contact_transfer_stage1_latched,
        torch.tensor([False, False]),
    )

    env.episode_length_buf[:] = 3
    env.test_tread[:] = True
    assert torch.equal(
        microduck_mdp.stair_new_tread_contact_after_reset(env),
        torch.tensor([0.0, 1.0]),
    )
    assert torch.equal(
        microduck_mdp.stair_new_tread_contact_after_reset(env),
        torch.zeros(2),
    )


def test_stage1_allows_policy_recontact_after_reset_contact_releases(monkeypatch):
    monkeypatch.setattr(
        microduck_mdp,
        "_virtual_lip_union_contact_state",
        _fake_virtual_union,
    )
    env = _contact_env(1)
    env.test_tread[:] = True
    assert microduck_mdp.stair_new_tread_contact_after_reset(env).item() == 0.0

    env.episode_length_buf[:] = 2
    env.test_tread[:] = False
    assert microduck_mdp.stair_new_tread_contact_after_reset(env).item() == 0.0

    env.episode_length_buf[:] = 3
    env.test_tread[:] = True
    assert microduck_mdp.stair_new_tread_contact_after_reset(env).item() == 1.0
    assert microduck_mdp.stair_new_tread_contact_after_reset(env).item() == 0.0


def test_stage2_needs_two_consecutive_tread_no_face_frames_after_stage1(monkeypatch):
    monkeypatch.setattr(
        microduck_mdp,
        "_virtual_lip_union_contact_state",
        _fake_virtual_union,
    )
    env = _contact_env(2)
    env.test_tread[:] = torch.tensor([True, False])
    assert torch.equal(
        microduck_mdp.stair_new_tread_contact_after_reset(env),
        torch.zeros(2),
    )
    assert torch.equal(
        microduck_mdp.stair_loaded_tread_face_release(env),
        torch.zeros(2),
    )
    assert env._stair_contact_transfer_stage2_latched.tolist() == [True, False]

    env.episode_length_buf[:] = 3
    env.test_face[1] = True
    env.test_tread[1] = False
    assert torch.equal(
        microduck_mdp.stair_loaded_tread_face_release(env),
        torch.zeros(2),
    )

    env.test_face[1] = False
    env.test_tread[1] = True
    assert torch.equal(
        microduck_mdp.stair_new_tread_contact_after_reset(env),
        torch.tensor([0.0, 1.0]),
    )
    env.test_force[1] = 0.10
    assert torch.equal(
        microduck_mdp.stair_loaded_tread_face_release(env),
        torch.zeros(2),
    )
    env.episode_length_buf[:] = 4
    env.test_force[1] = 1.0
    assert torch.equal(
        microduck_mdp.stair_loaded_tread_face_release(env),
        torch.zeros(2),
    )
    env.episode_length_buf[:] = 5
    assert torch.equal(
        microduck_mdp.stair_loaded_tread_face_release(env),
        torch.tensor([0.0, 1.0]),
    )
    assert torch.equal(
        microduck_mdp.stair_loaded_tread_face_release(env),
        torch.zeros(2),
    )


def test_stage15_is_ordered_one_shot_and_reset_safe(monkeypatch):
    monkeypatch.setattr(
        microduck_mdp,
        "_virtual_lip_union_contact_state",
        _fake_virtual_union,
    )
    env = _contact_env(3)
    env.episode_length_buf[:] = torch.tensor([3, 3, 1])
    env.test_tread[:] = True
    env.test_face[:] = torch.tensor([False, True, False])
    env._stair_contact_transfer_stage1_policy_achieved = torch.tensor(
        [True, True, True]
    )
    env._stair_contact_transfer_face_seen = torch.ones(3, dtype=torch.bool)

    reward = microduck_mdp.stair_loaded_tread_no_face_first_frame(
        env, min_policy_steps=1
    )

    assert torch.equal(reward, torch.tensor([1.0, 0.0, 0.0]))
    assert torch.equal(
        env._stair_contact_transfer_stage15_reset_baseline,
        torch.tensor([False, False, True]),
    )
    assert torch.equal(
        env._stair_contact_transfer_stage15_policy_achieved,
        torch.tensor([True, False, False]),
    )
    assert torch.equal(
        microduck_mdp.stair_loaded_tread_no_face_first_frame(
            env, min_policy_steps=1
        ),
        torch.zeros(3),
    )


def test_stage2_two_frame_release_follows_stage15_first_frame(monkeypatch):
    monkeypatch.setattr(
        microduck_mdp,
        "_virtual_lip_union_contact_state",
        _fake_virtual_union,
    )
    env = _contact_env(1)
    env.test_face[:] = True
    assert microduck_mdp.stair_new_tread_contact_after_reset(env).item() == 0.0
    assert microduck_mdp.stair_loaded_tread_face_release(env).item() == 0.0

    env.episode_length_buf[:] = 2
    env.test_face[:] = False
    env.test_tread[:] = False
    assert microduck_mdp.stair_new_tread_contact_after_reset(env).item() == 0.0

    env.episode_length_buf[:] = 3
    env.test_tread[:] = True
    assert microduck_mdp.stair_new_tread_contact_after_reset(env).item() == 1.0
    assert microduck_mdp.stair_loaded_tread_no_face_first_frame(env).item() == 1.0
    assert (
        microduck_mdp.stair_loaded_tread_face_release(
            env, require_stage15=True
        ).item()
        == 0.0
    )

    env.episode_length_buf[:] = 4
    assert (
        microduck_mdp.stair_loaded_tread_face_release(
            env, require_stage15=True
        ).item()
        == 1.0
    )
    assert env._stair_contact_transfer_stage15_policy_achieved.item()
    assert env._stair_contact_transfer_stage2_policy_achieved.item()


def test_reverse_stage15_seed_prepays_no_prefix_reward_and_unlocks_hold(
    monkeypatch,
):
    monkeypatch.setattr(
        microduck_mdp,
        "_virtual_lip_union_contact_state",
        _fake_virtual_union,
    )
    env = _contact_env(1)
    env._stair_contact_transfer_reverse_stage15_seed = torch.ones(
        1, dtype=torch.bool
    )
    env._stair_state_bank_family = torch.ones(1, dtype=torch.long)
    env.test_face[:] = False
    env.test_tread[:] = True
    env.test_force[:] = 7.0

    assert microduck_mdp.stair_new_tread_contact_after_reset(env).item() == 0.0
    assert (
        microduck_mdp.stair_loaded_tread_no_face_first_frame(
            env, min_normal_force=2.0
        ).item()
        == 0.0
    )
    assert (
        microduck_mdp.stair_loaded_tread_face_release(
            env, min_normal_force=2.0, require_stage15=True
        ).item()
        == 0.0
    )
    assert env._stair_contact_transfer_stage1_policy_achieved.item()
    assert env._stair_contact_transfer_stage15_policy_achieved.item()
    assert not env._stair_contact_transfer_stage2_latched.item()

    env.episode_length_buf[:] = 3
    assert (
        microduck_mdp.stair_loaded_tread_face_release(
            env, min_normal_force=2.0, require_stage15=True
        ).item()
        == 0.0
    )
    env.episode_length_buf[:] = 4
    assert (
        microduck_mdp.stair_loaded_tread_face_release(
            env, min_normal_force=2.0, require_stage15=True
        ).item()
        == 1.0
    )


def test_reverse_stage2_seed_prepays_no_stage_reward(monkeypatch):
    monkeypatch.setattr(
        microduck_mdp,
        "_virtual_lip_union_contact_state",
        _fake_virtual_union,
    )
    env = _contact_env(1)
    env._stair_contact_transfer_reverse_stage15_seed = torch.ones(
        1, dtype=torch.bool
    )
    env._stair_contact_transfer_reverse_stage2_seed = torch.ones(
        1, dtype=torch.bool
    )
    env._stair_state_bank_family = torch.full((1,), 2, dtype=torch.long)
    env.test_tread[:] = True
    env.test_force[:] = 7.0

    assert microduck_mdp.stair_new_tread_contact_after_reset(env).item() == 0.0
    assert (
        microduck_mdp.stair_loaded_tread_no_face_first_frame(
            env, min_normal_force=2.0
        ).item()
        == 0.0
    )
    assert (
        microduck_mdp.stair_loaded_tread_face_release(
            env, min_normal_force=2.0, require_stage15=True
        ).item()
        == 0.0
    )
    assert env._stair_contact_transfer_stage1_policy_achieved.item()
    assert env._stair_contact_transfer_stage15_policy_achieved.item()
    assert env._stair_contact_transfer_stage2_policy_achieved.item()


def test_penetration_cost_is_gated_before_arming_and_on_hard_level():
    robot = SimpleNamespace(
        indexing=SimpleNamespace(geom_ids=torch.tensor([0])),
        data=SimpleNamespace(
            geom_pos_w=torch.tensor(
                [
                    [[0.680, 0.0, 0.100]],
                    [[0.680, 0.0, 0.100]],
                    [[0.680, 0.0, 0.100]],
                ]
            ),
            geom_lin_vel_w=torch.tensor(
                [
                    [[0.40, 0.0, 0.0]],
                    [[0.40, 0.0, 0.0]],
                    [[0.40, 0.0, 0.0]],
                ]
            ),
            geom_ang_vel_w=torch.zeros(3, 1, 3),
            model=SimpleNamespace(
                geom_rbound=torch.tensor([0.0]),
                geom_contype=torch.tensor([1]),
            ),
        )
    )
    env = SimpleNamespace(
        num_envs=3,
        device="cpu",
        episode_length_buf=torch.tensor([2, 3, 3]),
        scene=_AssetScene(robot, torch.tensor([0, 2, 0])),
    )

    cost = microduck_mdp.stair_virtual_lip_penetration_cost(env)

    assert cost[0].item() == 0.0
    assert cost[1].item() == 0.0
    assert cost[2].item() > 0.0
    faces = microduck_mdp._virtual_lip_physical_face_x(
        env,
        nominal_stair_face_x=0.660,
        max_face_offset=0.040,
        num_terrain_levels=3,
    )
    assert torch.allclose(faces, torch.tensor([0.700, 0.660, 0.700]))


def test_true_shell_clearance_requires_every_corner_and_pays_once():
    geom_pos = torch.tensor([[[0.700, 0.0, 0.190]]])
    robot = SimpleNamespace(
        data=SimpleNamespace(
            geom_pos_w=geom_pos,
            geom_quat_w=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
            root_link_pos_w=torch.tensor([[0.700, 0.0, 0.205]]),
        )
    )
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        episode_length_buf=torch.ones(1, dtype=torch.long),
        scene=_AssetScene(robot, torch.zeros(1, dtype=torch.long)),
    )
    asset_cfg = SimpleNamespace(name="robot", geom_ids=torch.tensor([0]))

    assert microduck_mdp.stair_true_shell_clearance(
        env, hold_steps=2, asset_cfg=asset_cfg
    ).item() == 0.0
    assert not env._stair_true_shell_clearance_policy_achieved.item()
    # Raising the box by 5 mm makes its lowest corner 173 mm. The rear x
    # corner is 666 mm, so every corner now clears the nominal 660/170 lip.
    geom_pos[..., 2] = 0.195
    env.episode_length_buf[:] = 3
    assert microduck_mdp.stair_true_shell_clearance(
        env, hold_steps=2, asset_cfg=asset_cfg
    ).item() == 0.0
    env.episode_length_buf[:] = 4
    assert microduck_mdp.stair_true_shell_clearance(
        env, hold_steps=2, asset_cfg=asset_cfg
    ).item() == 1.0
    assert env._stair_true_shell_clearance_policy_achieved.item()
    assert microduck_mdp.stair_true_shell_clearance(
        env, hold_steps=2, asset_cfg=asset_cfg
    ).item() == 0.0


def test_true_shell_candidate_exposes_single_clear_frame():
    geom_pos = torch.tensor([[[0.700, 0.0, 0.205]]])
    robot = SimpleNamespace(
        data=SimpleNamespace(
            geom_pos_w=geom_pos,
            geom_quat_w=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
            root_link_pos_w=geom_pos[:, 0],
        )
    )
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        scene=_AssetScene(robot, torch.zeros(1, dtype=torch.long)),
    )
    asset_cfg = SimpleNamespace(name="robot", geom_ids=torch.tensor([0]))

    assert microduck_mdp.stair_true_shell_clearance_candidate(
        env, asset_cfg=asset_cfg
    ).item()
    geom_pos[..., 0] = 0.690
    assert not microduck_mdp.stair_true_shell_clearance_candidate(
        env, asset_cfg=asset_cfg
    ).item()


def test_shell_frontier_is_bounded_reset_safe_and_worst_corner_aligned():
    geom_pos = torch.tensor([[[0.665, 0.0, 0.175]]])
    robot = SimpleNamespace(
        data=SimpleNamespace(
            geom_pos_w=geom_pos,
            geom_quat_w=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
            root_link_pos_w=torch.tensor([[0.665, 0.0, 0.175]]),
        )
    )
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        episode_length_buf=torch.ones(1, dtype=torch.long),
        scene=_AssetScene(robot, torch.zeros(1, dtype=torch.long)),
        _stair_contact_transfer_stage15_policy_achieved=torch.ones(
            1, dtype=torch.bool
        ),
    )
    asset_cfg = SimpleNamespace(name="robot", geom_ids=torch.tensor([0]))

    assert microduck_mdp.stair_true_shell_clearance_frontier(
        env, asset_cfg=asset_cfg
    ).item() == 0.0
    env.episode_length_buf[:] = 3
    geom_pos[..., 0] = 0.680
    geom_pos[..., 2] = 0.190
    first_gain = microduck_mdp.stair_true_shell_clearance_frontier(
        env, asset_cfg=asset_cfg
    ).item()
    assert 0.44 < first_gain < 0.46
    assert microduck_mdp.stair_true_shell_clearance_frontier(
        env, asset_cfg=asset_cfg
    ).item() == 0.0
    geom_pos[..., 0] = 0.700
    geom_pos[..., 2] = 0.195
    second_gain = microduck_mdp.stair_true_shell_clearance_frontier(
        env, asset_cfg=asset_cfg
    ).item()
    assert 0.39 < second_gain < 0.41
    assert first_gain + second_gain < 1.0


def test_true_shell_can_be_hard_gated_after_policy_stage2():
    geom_pos = torch.tensor([[[0.700, 0.0, 0.190]]])
    robot = SimpleNamespace(
        data=SimpleNamespace(
            geom_pos_w=geom_pos,
            geom_quat_w=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
            root_link_pos_w=torch.tensor([[0.700, 0.0, 0.205]]),
        )
    )
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        episode_length_buf=torch.ones(1, dtype=torch.long),
        scene=_AssetScene(robot, torch.zeros(1, dtype=torch.long)),
        _stair_contact_transfer_stage2_policy_achieved=torch.zeros(
            1, dtype=torch.bool
        ),
    )
    asset_cfg = SimpleNamespace(name="robot", geom_ids=torch.tensor([0]))
    required = "_stair_contact_transfer_stage2_policy_achieved"

    assert microduck_mdp.stair_true_shell_clearance(
        env, hold_steps=2, required_latch_name=required, asset_cfg=asset_cfg
    ).item() == 0.0
    geom_pos[..., 2] = 0.195
    env.episode_length_buf[:] = 3
    assert microduck_mdp.stair_true_shell_clearance(
        env, hold_steps=2, required_latch_name=required, asset_cfg=asset_cfg
    ).item() == 0.0
    env._stair_contact_transfer_stage2_policy_achieved[:] = True
    env.episode_length_buf[:] = 4
    assert microduck_mdp.stair_true_shell_clearance(
        env, hold_steps=2, required_latch_name=required, asset_cfg=asset_cfg
    ).item() == 0.0
    env.episode_length_buf[:] = 5
    assert microduck_mdp.stair_true_shell_clearance(
        env, hold_steps=2, required_latch_name=required, asset_cfg=asset_cfg
    ).item() == 1.0


def test_secured_tread_requires_ordered_policy_shell_clearance(monkeypatch):
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_link_pos_w=torch.tensor([[0.75, 0.0, 0.205]]),
            root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            root_link_lin_vel_w=torch.zeros(1, 3),
            root_link_ang_vel_w=torch.zeros(1, 3),
        )
    )
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        step_dt=0.02,
        episode_length_buf=torch.full((1,), 3, dtype=torch.long),
        scene=_AssetScene(robot, torch.zeros(1, dtype=torch.long)),
    )

    def fake_standard_masks(*_args, **_kwargs):
        return (
            torch.zeros(1, 1, dtype=torch.bool),
            torch.ones(1, 1, dtype=torch.bool),
            torch.zeros(1, 1, 3),
        )

    monkeypatch.setattr(
        microduck_mdp, "_standard_stair_contact_masks", fake_standard_masks
    )
    required = "_stair_true_shell_clearance_policy_achieved"
    assert microduck_mdp.stair_first_tread_secured(
        env, hold_time_s=0.02, required_latch_name=required
    ).item() == 0.0

    setattr(env, required, torch.ones(1, dtype=torch.bool))
    assert microduck_mdp.stair_first_tread_secured(
        env, hold_time_s=0.02, required_latch_name=required
    ).item() == 1.0


def test_curriculum_ignores_baseline_tread_and_uses_policy_transfer_only():
    class _Terrain:
        def __init__(self):
            self.terrain_origins = torch.zeros(2, 3)
            self.terrain_levels = torch.zeros(2, dtype=torch.long)
            self.last_success = None

        def update_env_origins(self, env_ids, success, move_down):
            self.last_success = success.clone()
            self.terrain_levels[env_ids] += success.to(torch.long)
            assert not torch.any(move_down)

    terrain = _Terrain()
    env = SimpleNamespace(
        device="cpu",
        scene=SimpleNamespace(terrain=terrain),
        _stair_contact_transfer_stage2_latched=torch.tensor([True, True]),
        _stair_contact_transfer_stage2_policy_achieved=torch.tensor([False, True]),
    )

    microduck_mdp.stair_contact_transfer_terrain_levels(env, torch.arange(2))

    assert terrain.last_success.tolist() == [False, True]
    assert terrain.terrain_levels.tolist() == [0, 1]


def test_a30_cfg_wires_reset_families_rewards_contract_and_hard_play():
    inherited = make_microduck_stair_forward_propagation_rsi_env_cfg()
    cfg = make_microduck_stair_virtual_lip_transfer_rsi_env_cfg()
    play_cfg = make_microduck_stair_virtual_lip_transfer_rsi_env_cfg(play=True)

    family_event = cfg.events["state_bank_family"]
    assert family_event.params["family_weights"] == (2, 1, 1)
    assert tuple(cfg.events).index("state_bank_family") < tuple(cfg.events).index(
        "root_over_lip_state_bank"
    )
    families = {
        cfg.events[name].params["reset_family"]
        for name in (
            "root_over_lip_state_bank",
            "tread_contact_state_bank",
            "manufacturer_roulade_state_bank",
        )
    }
    assert families == {0, 1, 2}

    ordered_stages = (
        cfg.rewards["stair_new_tread_contact"],
        cfg.rewards["stair_loaded_tread_face_release"],
        cfg.rewards["stair_true_shell_clearance"],
        cfg.rewards["stair_first_tread_secured"],
    )
    assert [term.weight for term in ordered_stages] == [10.0, 60.0, 500.0, 600.0]
    assert cfg.rewards["stair_loaded_tread_face_release"].params["hold_steps"] == 2
    assert cfg.rewards["stair_riser_face_after_tread"].weight == -1.0
    assert cfg.rewards["stair_virtual_lip_penetration"].weight == -0.5
    assert cfg.rewards["stair_forward_propagation_tiers"].weight == 0.0
    assert cfg.rewards["stair_terminal_position_objective"].weight == 0.0
    assert cfg.rewards["stair_first_riser_clearance"].weight == 0.0
    assert "route_challenge_levels" not in cfg.events
    assert cfg.events["root_over_lip_state_bank"].params["bank_path"].endswith(
        "full170-a28-face-no-tread-state-bank.pt"
    )
    for observation_group in ("actor", "critic"):
        cue = cfg.observations[observation_group].terms["body_command"].params
        assert cue["min_riser_height"] == cue["max_riser_height"] == 0.17

    actor_terms = cfg.observations["actor"].terms
    assert tuple(actor_terms) == tuple(inherited.observations["actor"].terms)
    assert all(
        actor_terms[name].func is inherited.observations["actor"].terms[name].func
        for name in actor_terms
    )
    inherited_contract_dims = {
        "base_ang_vel": 3,
        "projected_gravity": 3,
        "joint_pos": 14,
        "joint_vel": 14,
        "actions": 14,
        "command": 3,
        "head_command": 4,
        "body_command": 6,
    }
    assert tuple(actor_terms) == tuple(inherited_contract_dims)
    assert sum(inherited_contract_dims.values()) == 61

    terrain_generator = cfg.scene.terrain.terrain_generator
    terrain = terrain_generator.sub_terrains["virtual_lip_stairs"]
    assert terrain_generator.num_rows == VIRTUAL_LIP_CURRICULUM_LEVELS == 3
    assert terrain.first_riser_face_offset_range == (
        VIRTUAL_LIP_MAX_FACE_OFFSET,
        0.0,
    )
    assert terrain.first_riser_face_offset_levels == VIRTUAL_LIP_CURRICULUM_LEVELS
    assert terrain.riser_height == STANDARD_RISER_HEIGHT == 0.17
    assert terrain.tread_depth == STANDARD_TREAD_DEPTH == 0.28
    assert abs(STANDARD_STAIR_START_DISTANCE - 0.66) < 1.0e-9
    assert cfg.scene.terrain.max_init_terrain_level == 0
    assert "terrain_levels" not in cfg.curriculum
    assert cfg.rewards["stair_first_tread_secured"].params[
        "required_latch_name"
    ] == "_stair_true_shell_clearance_policy_achieved"

    assert "terrain_levels" not in play_cfg.curriculum
    hard_view = play_cfg.events["a30_hard_viewer"].params
    assert hard_view == {"terrain_levels": (2,), "terrain_types": (0,)}


def test_a31_adds_only_ordered_stage15_reward_and_keeps_fixed_level0():
    a30 = make_microduck_stair_virtual_lip_transfer_rsi_env_cfg()
    cfg = make_microduck_stair_contact_stage_rsi_env_cfg()
    play_cfg = make_microduck_stair_contact_stage_rsi_env_cfg(play=True)

    assert "stair_loaded_tread_no_face_first_frame" not in a30.rewards
    stage15 = cfg.rewards["stair_loaded_tread_no_face_first_frame"]
    stage2 = cfg.rewards["stair_loaded_tread_face_release"]
    assert stage15.func is microduck_mdp.stair_loaded_tread_no_face_first_frame
    assert stage15.weight == 20.0
    assert stage15.params["min_policy_steps"] == 3
    assert stage15.params["min_normal_force"] == 2.0
    assert stage15.params["sensor_names"] == stage2.params["sensor_names"]
    assert "hold_steps" not in stage15.params
    assert stage2.params["hold_steps"] == 2
    assert stage2.params["min_normal_force"] == 2.0
    assert stage2.params["require_stage15"] is True
    reward_names = list(cfg.rewards)
    assert reward_names.index("stair_loaded_tread_no_face_first_frame") < (
        reward_names.index("stair_loaded_tread_face_release")
    )
    assert "terrain_levels" not in cfg.curriculum
    assert cfg.scene.terrain.max_init_terrain_level == 0
    assert play_cfg.events["a30_hard_viewer"].params == {
        "terrain_levels": (2,),
        "terrain_types": (0,),
    }


def test_a32_replaces_generic_banks_with_exact_two_family_reverse_mix():
    cfg = make_microduck_stair_stage15_reverse_rsi_env_cfg()
    play_cfg = make_microduck_stair_stage15_reverse_rsi_env_cfg(play=True)

    assert cfg.events["state_bank_family"].params["family_weights"] == (1, 1)
    assert "tread_contact_state_bank" not in cfg.events
    assert "manufacturer_roulade_state_bank" not in cfg.events
    reverse_bank = cfg.events["stage15_reverse_state_bank"]
    assert reverse_bank.params["reset_family"] == 1
    assert reverse_bank.params["bank_path"].endswith(
        "full170-a32-model14-stage15-state-bank.pt"
    )
    assert cfg.events["root_over_lip_state_bank"].params["reset_family"] == 0
    event_names = tuple(cfg.events)
    assert event_names.index("stage15_reverse_state_bank") < event_names.index(
        "stage15_reverse_context"
    )
    assert play_cfg.events["a30_hard_viewer"].params == {
        "terrain_levels": (2,),
        "terrain_types": (0,),
    }


def test_a33_adds_exact_stage2_family_and_bounded_shell_frontier():
    cfg = make_microduck_stair_stage2_reverse_rsi_env_cfg()
    play_cfg = make_microduck_stair_stage2_reverse_rsi_env_cfg(play=True)

    assert cfg.events["state_bank_family"].params["family_weights"] == (2, 1, 1)
    assert cfg.events["root_over_lip_state_bank"].params["reset_family"] == 0
    assert cfg.events["stage15_reverse_state_bank"].params["reset_family"] == 1
    stage2_bank = cfg.events["stage2_reverse_state_bank"]
    assert stage2_bank.params["reset_family"] == 2
    assert stage2_bank.params["bank_path"].endswith(
        "full170-a33-model10-stage2-state-bank.pt"
    )
    context = cfg.events["stage2_reverse_context"]
    assert context.params == {"stage15_families": (1, 2), "stage2_family": 2}
    event_names = tuple(cfg.events)
    assert event_names.index("stage15_reverse_state_bank") < event_names.index(
        "stage2_reverse_state_bank"
    )
    assert event_names.index("stage2_reverse_state_bank") < event_names.index(
        "stage2_reverse_context"
    )

    rewards = tuple(cfg.rewards)
    assert rewards.index("stair_loaded_tread_no_face_first_frame") < rewards.index(
        "stair_loaded_tread_face_release"
    )
    assert rewards.index("stair_loaded_tread_face_release") < rewards.index(
        "stair_true_shell_clearance_frontier"
    )
    assert rewards.index("stair_true_shell_clearance_frontier") < rewards.index(
        "stair_true_shell_clearance"
    )
    frontier = cfg.rewards["stair_true_shell_clearance_frontier"]
    assert frontier.func is microduck_mdp.stair_true_shell_clearance_frontier
    assert frontier.weight == 40.0
    assert cfg.rewards["stair_true_shell_clearance"].params[
        "required_latch_name"
    ] == "_stair_contact_transfer_stage2_policy_achieved"
    assert play_cfg.events["a30_hard_viewer"].params == {
        "terrain_levels": (2,),
        "terrain_types": (0,),
    }

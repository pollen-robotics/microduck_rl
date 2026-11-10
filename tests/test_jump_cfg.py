"""Configuration and MDP unit tests for Mjlab-Jump-Flat-MicroDuck and its Backlash twin."""

import math
import re
from types import SimpleNamespace

import mujoco
import pytest
import torch

from mjlab_microduck.robot.microduck_constants import (
    BACKLASH_HOME_FRAME,
    MICRODUCK_WALK_BACKLASH_ROBOT_CFG,
    get_walk_backlash_spec,
    get_walk_spec,
)
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.backlash import make_backlash_variant
from mjlab_microduck.tasks.microduck_jump_env_cfg import (
    HEAD_BODY_NAMES,
    STAND_Z,
    MicroduckJumpRlCfg,
    make_microduck_jump_env_cfg,
)
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
)


def test_jump_obs_parity_with_velocity():
    """61D actor and 76D critic obs term order must match velocity env for ONNX hot-swap."""
    jump = make_microduck_jump_env_cfg()
    vel = make_microduck_velocity_env_cfg()
    for grp in ("actor", "critic"):
        assert list(jump.observations[grp].terms.keys()) == list(
            vel.observations[grp].terms.keys()
        ), f"Observation layout mismatch in group {grp}"


def test_jump_upstream_push_robot_is_removed():
    """Regression: ENABLE_VELOCITY_PUSHES=False must pop upstream quadruped push_robot."""
    cfg = make_microduck_jump_env_cfg()
    assert "push_robot" not in cfg.events
    play_cfg = make_microduck_jump_env_cfg(play=True)
    assert "push_robot" not in play_cfg.events


def test_jump_bam_and_nan_guards_are_wired():
    """Standalone env must wire expand_bam_friction_fields, critic _safe terms, and nan_state."""
    cfg = make_microduck_jump_env_cfg()
    assert "expand_bam_friction_fields" in cfg.events
    for name in ("foot_contact_forces", "foot_height", "foot_air_time"):
        assert cfg.observations["critic"].terms[name].func.__name__.endswith("_safe"), (
            f"critic/{name} is missing its NaN guard"
        )
    assert cfg.terminations["nan_state"].params.get("sensor_names") == (
        "feet_ground_contact",
    )


def test_jump_curricula_do_not_crush_takeoff_or_swing_head():
    """Regression: action_rate_weight must match initial action_rate_l2 (-0.01)
    and walking head_pose_range widening must be absent."""
    cfg = make_microduck_jump_env_cfg()
    stages = cfg.curriculum["action_rate_weight"].params["weight_stages"]
    assert stages[0]["weight"] == cfg.rewards["action_rate_l2"].weight == -0.01
    assert stages[-1]["weight"] >= -0.08
    assert "head_pose_range" not in cfg.curriculum
    assert "head_pose_bias_weight" not in cfg.curriculum


def test_jump_reward_signs_and_stand_z():
    """Positive task terms > 0, cost terms (returning >= 0) strictly < 0."""
    cfg = make_microduck_jump_env_cfg()
    r = cfg.rewards
    assert STAND_Z == 0.115
    assert r["jump_height"].params["stand_height"] == STAND_Z
    assert r["air_time"].func is microduck_mdp.both_feet_air_time
    assert r["jump_height"].func is microduck_mdp.both_feet_jump_height
    assert r["base_lin_vel_z"].func is microduck_mdp.jump_takeoff_velocity
    assert r["single_foot_air"].func is microduck_mdp.single_foot_air_cost

    for pos_name in (
        "air_time",
        "jump_height",
        "base_lin_vel_z",
        "upright",
        "pose",
        "head_pose_tracking",
    ):
        assert r[pos_name].weight > 0.0, pos_name
    assert r["head_pose_tracking"].weight <= 0.3

    for cost_name in (
        "single_foot_air",
        "termination_penalty",
        "base_xy_vel_penalty",
        "action_rate_l2",
        "body_ang_vel",
        "angular_momentum",
        "foot_slip",
        "self_collisions",
        "dof_pos_limits",
    ):
        assert r[cost_name].weight < 0.0, cost_name
    assert r["foot_slip"].params["command_threshold"] < 0.0


def test_jump_commands_keep_normalizer_alive_and_hide_arrows():
    """Twist/head/body commands must train exact-zero idle while keeping input std non-zero."""
    cfg = make_microduck_jump_env_cfg()
    twist = cfg.commands["twist"]
    assert twist.debug_vis is False
    assert twist.heading_command is False
    assert twist.ranges.heading is None
    assert twist.rel_forward_envs < 0.0
    assert 0.0 < twist.rel_standing_envs < 1.0
    assert twist.ranges.lin_vel_x[0] < 0.0 < twist.ranges.lin_vel_x[1]
    assert cfg.commands["head_pose"].zero_command_prob == 0.5
    assert cfg.commands["body_pose"].zero_command_prob == 0.5

    play_cfg = make_microduck_jump_env_cfg(play=True)
    assert play_cfg.commands["twist"].debug_vis is False
    assert play_cfg.commands["twist"].rel_standing_envs == 1.0
    assert play_cfg.commands["twist"].ranges.lin_vel_x == (0.0, 0.0)


def test_jump_rl_cfg_has_distinct_experiment_and_symmetry():
    assert MicroduckJumpRlCfg.experiment_name == "jump"
    assert MicroduckJumpRlCfg.run_name == "jump"
    assert MicroduckJumpRlCfg.algorithm.symmetry_cfg is not None


def test_jump_flat_and_backlash_tasks_are_registered():
    """Only the Flat base task and its Flat-Backlash twin are registered."""
    from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg

    from mjlab_microduck.tasks import _BACKLASH_TASKS

    tasks = list_tasks()
    jump_tasks = [t for t in tasks if "Jump" in t]
    assert sorted(jump_tasks) == [
        "Mjlab-Jump-Flat-Backlash-MicroDuck",
        "Mjlab-Jump-Flat-MicroDuck",
    ]

    rows = [r for r in _BACKLASH_TASKS if r[0].startswith("Mjlab-Jump-")]
    assert len(rows) == 1
    task_id, make_fn, kw, rl_cfg, robot_cfg = rows[0]
    assert task_id == "Mjlab-Jump-Flat-Backlash-MicroDuck"
    assert make_fn is make_microduck_jump_env_cfg
    assert kw == {}
    assert rl_cfg is MicroduckJumpRlCfg
    assert robot_cfg is MICRODUCK_WALK_BACKLASH_ROBOT_CFG

    for play in (False, True):
        reg_bl = load_env_cfg("Mjlab-Jump-Flat-Backlash-MicroDuck", play=play)
        assert reg_bl.scene.entities["robot"].spec_fn is get_walk_backlash_spec
    assert load_rl_cfg("Mjlab-Jump-Flat-Backlash-MicroDuck").experiment_name == "jump"


def test_jump_backlash_variant_mirrors_walk_model_and_preserves_invariants():
    """Backlash twin must swap to MICRODUCK_WALK_BACKLASH_ROBOT_CFG, remap both
    actor/critic obs to encoder-through-backlash, preserve encoder-bias split,
    scope dof_pos_limits away from ±1° backlash stops, and leave base cfg unmutated."""
    base_cfg = make_microduck_jump_env_cfg()
    orig_pose_joints = base_cfg.rewards["pose"].params["asset_cfg"].joint_names

    bl_cfg = make_backlash_variant(
        make_microduck_jump_env_cfg(), MICRODUCK_WALK_BACKLASH_ROBOT_CFG
    )
    vel_cfg = make_microduck_velocity_env_cfg()

    assert bl_cfg.scene.entities["robot"] is MICRODUCK_WALK_BACKLASH_ROBOT_CFG
    assert bl_cfg.scene.entities["robot"].init_state is BACKLASH_HOME_FRAME

    for grp in ("actor", "critic"):
        assert list(bl_cfg.observations[grp].terms.keys()) == list(
            vel_cfg.observations[grp].terms.keys()
        ), f"Backlash obs layout mismatch in group {grp}"
        jp = bl_cfg.observations[grp].terms["joint_pos"]
        jv = bl_cfg.observations[grp].terms["joint_vel"]
        assert jp.func is microduck_mdp.joint_pos_rel_backlash
        assert jv.func is microduck_mdp.joint_vel_rel_backlash
        assert jp.params["asset_cfg"].joint_names == (r"^(?!passive_).*",)
        assert jv.params["asset_cfg"].joint_names == (r"^(?!passive_).*",)

    # Actor sees biased encoder view; critic keeps privileged unbiased view.
    assert bl_cfg.observations["actor"].terms["joint_pos"].params["biased"] is True
    assert bl_cfg.observations["critic"].terms["joint_pos"].params["biased"] is False

    # Backlash hinges ride their ±1° hard stops and must be excluded from soft limits.
    assert bl_cfg.rewards["dof_pos_limits"].params["asset_cfg"].joint_names == (
        r"^(?!passive_).*",
    )

    # Pose regex excludes backlash hinges without mutating the base config's SceneEntityCfg.
    bl_pose_joints = bl_cfg.rewards["pose"].params["asset_cfg"].joint_names
    assert all("_backlash" in pat for pat in bl_pose_joints)
    assert base_cfg.rewards["pose"].params["asset_cfg"].joint_names == orig_pose_joints


@pytest.mark.parametrize("spec_fn", [get_walk_spec, get_walk_backlash_spec])
def test_jump_selectors_resolve_on_walk_and_backlash_mjcf(spec_fn):
    """Compile actual walk and walk_backlash MJCFs on CPU to lock joint/geom/site/body selectors."""
    model = spec_fn().compile()
    is_backlash = spec_fn is get_walk_backlash_spec

    articulated = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        for j in range(model.njnt)
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE
    ]
    servo_pat = re.compile(r"^(?!passive_).*")
    servos = [n for n in articulated if servo_pat.match(n)]
    assert len(servos) == 14

    backlash_joints = [
        (j, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j))
        for j in range(model.njnt)
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE
        and mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j).startswith("passive_")
        and mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j).endswith("_backlash")
    ]
    if is_backlash:
        assert len(backlash_joints) == 14
        half_play = math.radians(1.0)
        for j_id, j_name in backlash_joints:
            lo, hi = model.jnt_range[j_id]
            assert math.isclose(lo, -half_play, rel_tol=1e-6), j_name
            assert math.isclose(hi, half_play, rel_tol=1e-6), j_name
            # First-match rule in BACKLASH_HOME_FRAME must pin backlash hinges at 0.0 rad.
            matched_val = next(
                val for pat, val in BACKLASH_HOME_FRAME.joint_pos.items() if re.match(pat, j_name)
            )
            assert matched_val == 0.0, j_name
    else:
        assert len(backlash_joints) == 0

    bl_cfg = make_backlash_variant(
        make_microduck_jump_env_cfg(), MICRODUCK_WALK_BACKLASH_ROBOT_CFG
    )
    pose_pats = [re.compile(p) for p in bl_cfg.rewards["pose"].params["asset_cfg"].joint_names]
    pose_matched = [n for n in articulated if any(p.match(n) for p in pose_pats)]
    assert len(pose_matched) == 10  # 5 left leg + 5 right leg servos only
    assert all(not n.startswith("passive_") for n in pose_matched)

    geom_names = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in range(model.ngeom)
    }
    assert {"left_foot_collision", "right_foot_collision"} <= geom_names

    site_names = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, s) for s in range(model.nsite)
    }
    assert {"left_foot", "right_foot", "imu"} <= site_names

    body_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(model.nbody)
    ]
    assert "trunk_base" in body_names
    for pat in HEAD_BODY_NAMES:
        rx = re.compile(f"^{pat}$")
        assert any(b and rx.match(b) for b in body_names), f"Unmatched head DR body: {pat}"


# ── Minimal CPU mock env for MDP function unit tests ─────────────────────────


def _make_mock_jump_env(
    air_times: list[tuple[float, float]],
    com_z: list[float],
    vz: list[float] | None = None,
    v_xy: list[tuple[float, float]] | None = None,
    ep_step: int = 20,
):
    n = len(air_times)
    if vz is None:
        vz = [0.0] * n
    if v_xy is None:
        v_xy = [(0.0, 0.0)] * n

    pos_w = torch.zeros(n, 3)
    pos_w[:, 2] = torch.tensor(com_z, dtype=torch.float32)

    lin_vel_w = torch.zeros(n, 3)
    lin_vel_w[:, :2] = torch.tensor(v_xy, dtype=torch.float32)
    lin_vel_w[:, 2] = torch.tensor(vz, dtype=torch.float32)

    g_b = torch.zeros(n, 3)
    g_b[:, 2] = -1.0  # Upright in body frame

    asset = SimpleNamespace(
        data=SimpleNamespace(
            root_link_pos_w=pos_w,
            root_link_lin_vel_w=lin_vel_w,
            projected_gravity_b=g_b,
        )
    )
    sensor = SimpleNamespace(
        data=SimpleNamespace(
            current_air_time=torch.tensor(air_times, dtype=torch.float32),
        )
    )
    terrain = SimpleNamespace(env_origins=torch.zeros(n, 3))

    class _Scene:
        def __init__(self):
            self.terrain = terrain
            self._items = {"robot": asset, "feet_ground_contact": sensor}

        def __getitem__(self, key):
            return self._items[key]

    return SimpleNamespace(
        num_envs=n,
        device="cpu",
        scene=_Scene(),
        episode_length_buf=torch.full((n,), ep_step, dtype=torch.long),
        extras={"log": {}},
    )


def test_mdp_both_feet_air_time_and_single_foot_cost():
    """Simultaneous flight rewards only two-foot liftoff; single-foot air triggers cost."""
    # env 0: both grounded; env 1: left only airborne; env 2: both airborne (0.10 s)
    env = _make_mock_jump_env(
        air_times=[(0.0, 0.0), (0.10, 0.0), (0.10, 0.12)],
        com_z=[0.115, 0.125, 0.145],
        ep_step=25,
    )
    air_rew = microduck_mdp.both_feet_air_time(
        env, sensor_name="feet_ground_contact", threshold_min=0.0
    )
    assert air_rew[0].item() == 0.0
    assert air_rew[1].item() == 0.0
    assert torch.isclose(air_rew[2], torch.tensor(1.5), atol=1e-5)

    one_foot = microduck_mdp.single_foot_air_cost(env, sensor_name="feet_ground_contact")
    assert one_foot.tolist() == [0.0, 1.0, 0.0]

    # Settle gate: during first 10 steps after spawn, both terms return 0.0
    early_env = _make_mock_jump_env(
        air_times=[(0.10, 0.10), (0.10, 0.0)],
        com_z=[0.145, 0.125],
        ep_step=5,
    )
    assert torch.all(
        microduck_mdp.both_feet_air_time(early_env, sensor_name="feet_ground_contact")
        == 0.0
    )
    assert torch.all(
        microduck_mdp.single_foot_air_cost(early_env, sensor_name="feet_ground_contact")
        == 0.0
    )


def test_mdp_both_feet_jump_height_and_takeoff_velocity():
    """Jump height pays 1.0 at STAND_Z + 3 cm in flight and 0 on ground; takeoff vz gates at 0.108 m."""
    # env 0: tiptoe on ground at 0.145 m -> 0 jump_height
    # env 1: airborne tuck below STAND_Z (0.105 m) -> 0 jump_height
    # env 2: airborne at STAND_Z + 0.03 m (0.145 m) -> 1.0 jump_height
    env = _make_mock_jump_env(
        air_times=[(0.0, 0.0), (0.08, 0.08), (0.08, 0.08)],
        com_z=[0.145, 0.105, STAND_Z + 0.03],
        vz=[0.5, 0.5, 0.5],
        v_xy=[(0.3, -0.4), (0.0, 0.0), (0.0, 0.0)],
        ep_step=25,
    )
    jh = microduck_mdp.both_feet_jump_height(
        env, sensor_name="feet_ground_contact", stand_height=STAND_Z, ref_lift=0.03
    )
    assert jh[0].item() == 0.0
    assert jh[1].item() == 0.0
    assert torch.isclose(jh[2], torch.tensor(1.0), atol=1e-5)

    # Takeoff velocity: env 1 is at 0.105 m (< 0.108 m floor-bobbing gate) -> 0.0;
    # env 0 and env 2 are at 0.145 m (> 0.108 m) -> 0.5
    tv = microduck_mdp.jump_takeoff_velocity(env, min_height=0.108, max_height=0.175)
    assert tv[1].item() == 0.0
    assert torch.isclose(tv[0], torch.tensor(0.5), atol=1e-5)
    assert torch.isclose(tv[2], torch.tensor(0.5), atol=1e-5)

    # Horizontal XY velocity squared: 0.3^2 + (-0.4)^2 = 0.25
    xy_cost = microduck_mdp.base_xy_velocity_penalty(env)
    assert torch.isclose(xy_cost[0], torch.tensor(0.25), atol=1e-5)
    assert xy_cost[1].item() == 0.0


def test_mdp_reward_param_curriculum_updates_live_manager():
    term_cfg = SimpleNamespace(params={"threshold_min": 0.0})
    reward_manager = SimpleNamespace(get_term_cfg=lambda name: term_cfg)
    env = SimpleNamespace(
        device="cpu",
        common_step_counter=2000 * 24,
        reward_manager=reward_manager,
    )
    stages = [
        {"step": 0, "value": 0.0},
        {"step": 1000 * 24, "value": 0.04},
        {"step": 2000 * 24, "value": 0.08},
        {"step": 3000 * 24, "value": 0.11},
    ]
    out = microduck_mdp.reward_param_curriculum(
        env,
        env_ids=torch.tensor([0]),
        reward_name="air_time",
        param_name="threshold_min",
        stages=stages,
    )
    assert term_cfg.params["threshold_min"] == 0.08
    assert torch.isclose(out, torch.tensor([0.08]))

"""Cfg invariants for Mjlab-PlayDead-*-MicroDuck.

CPU only — no GPU, no env construction. Locks the 61D contract, the supine
(not inverted) orientation signal, reward signs, and the sit-basin audit.
"""

import math

import torch

from mjlab_microduck.robot.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_playdead_env_cfg import (
    DEAD_Z,
    STAND_Z,
    MicroduckPlayDeadRlCfg,
    make_microduck_playdead_env_cfg,
)
from mjlab_microduck.tasks.mdp import (
    body_supine_cos_from_quat,
    body_supine_linear,
    com_downward_velocity,
    playdead_composite,
    playdead_composite_from_values,
    playdead_height_gaussian,
    playdead_height_l1,
    playdead_hold,
    supine_gaussian,
    trunk_grounded_supine,
    trunk_downward_velocity_penalty,
    trunk_vertical_accel_penalty,
)


# Sit env's measured seated rest. Play-dead must not treat this as success.
_SIT_Z = 0.060


def test_playdead_task_rewards():
    cfg = make_microduck_playdead_env_cfg()
    r = cfg.rewards
    assert r["supine_linear"].weight == 1.5
    assert r["supine_linear"].func is body_supine_linear
    assert r["supine_sharp"].func is supine_gaussian
    assert r["height_dead"].func is playdead_height_gaussian
    assert r["height_dead"].params["target_height"] == DEAD_Z
    assert r["height_dead_l1"].func is playdead_height_l1
    assert DEAD_Z < _SIT_Z < STAND_Z
    assert r["com_downward_velocity"].func is com_downward_velocity
    assert r["com_downward_velocity"].params["min_height"] > DEAD_Z
    assert r["com_downward_velocity"].params["min_supine"] == 0.0
    assert r["playdead_hold"].func is playdead_hold
    assert r["trunk_grounded_supine"].func is trunk_grounded_supine
    assert r["playdead_composite"].func is playdead_composite
    assert r["playdead_composite"].params["target_height"] == DEAD_Z
    # No binary per-step jackpot (arriving early then collecting an annuity).
    assert "playdead_success" not in r
    assert "inverted_linear" not in r
    # Walking terms must be gone — upright would fight the flop.
    for name in (
        "track_linear_velocity",
        "track_angular_velocity",
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "pose",
        "upright",
    ):
        assert name not in r, name


def test_self_negating_penalties_use_positive_weights():
    cfg = make_microduck_playdead_env_cfg()
    r = cfg.rewards
    assert r["gentle_impact"].func is trunk_vertical_accel_penalty
    assert r["too_fast_drop"].func is trunk_downward_velocity_penalty
    assert r["height_dead_l1"].func is playdead_height_l1
    for name in ("gentle_impact", "too_fast_drop", "height_dead_l1"):
        assert r[name].weight > 0, name
    for name in ("action_rate_l2", "body_ang_vel", "angular_momentum", "self_collisions"):
        assert r[name].weight < 0, name


def test_playdead_obs_slots_padded():
    cfg = make_microduck_playdead_env_cfg()
    for grp in ("actor", "critic"):
        terms = cfg.observations[grp].terms
        assert "command" in terms  # twist (3)
        assert "head_command" in terms
        assert "body_command" in terms
        assert terms["head_command"].params["dim"] == 4
        assert terms["body_command"].params["dim"] == 6
        assert terms["head_command"].func is microduck_mdp.zero_command_padding
        assert terms["body_command"].func is microduck_mdp.zero_command_padding


def test_twist_command_is_neutralised():
    cfg = make_microduck_playdead_env_cfg()
    cmd = cfg.commands["twist"]
    assert isinstance(cmd, microduck_mdp.VelocityCommandCommandOnlyCfg)
    assert cmd.ranges.lin_vel_x == (-0.01, 0.01)
    assert cmd.ranges.lin_vel_y == (-0.01, 0.01)
    assert cmd.ranges.ang_vel_z == (-0.05, 0.05)
    assert cmd.heading_command is False
    assert cmd.ranges.heading is None
    assert cmd.rel_standing_envs == 0.0


def test_playdead_reset_mix():
    cfg = make_microduck_playdead_env_cfg()
    p = cfg.events["set_ground_state"].params
    assert p["standing_prob"] == 0.75
    assert p["face_up_prob"] == 0.25
    assert p["sitting_prob"] == 0.0
    assert p["face_down_prob"] == 0.0
    assert abs(
        p["standing_prob"] + p["face_up_prob"] + p["sitting_prob"] + p["face_down_prob"] - 1.0
    ) < 1e-6


def test_ground_state_event_runs_after_base_reset():
    cfg = make_microduck_playdead_env_cfg()
    order = list(cfg.events.keys())
    assert order.index("set_ground_state") > order.index("reset_base")
    assert order.index("set_ground_state") > order.index("reset_robot_joints")


def test_no_fall_termination_and_no_push():
    cfg = make_microduck_playdead_env_cfg()
    assert "fell_over" not in cfg.terminations
    assert "nan_state" in cfg.terminations
    assert "push_robot" not in cfg.events
    assert "expand_bam_friction_fields" in cfg.events


def test_uses_allcollisions_robot():
    cfg = make_microduck_playdead_env_cfg()
    assert cfg.scene.entities["robot"] is MICRODUCK_STANDUP_ROBOT_CFG
    names = [s.name for s in cfg.scene.sensors]
    assert "feet_ground_contact" in names
    assert "self_collision" in names
    assert "trunk_ground_contact" in names


def test_playdead_variants_build():
    play = make_microduck_playdead_env_cfg(play=True)
    assert play.episode_length_s == 4.0
    p = play.events["set_ground_state"].params
    assert p["standing_prob"] == 0.5
    assert p["face_up_prob"] == 0.5
    assert "ground_state_mix" not in play.curriculum
    rough = make_microduck_playdead_env_cfg(rough=True)
    assert rough.scene.terrain.terrain_type == "generator"
    train = make_microduck_playdead_env_cfg()
    assert "ground_state_mix" in train.curriculum


def test_ground_state_curriculum_stays_on_back():
    cfg = make_microduck_playdead_env_cfg()
    stages = cfg.curriculum["ground_state_mix"].params["param_stages"]
    steps = [s["step"] for s in stages]
    assert steps[0] == 0 and steps == sorted(steps) and len(set(steps)) == len(steps)
    face_up = [s["params"]["face_up_prob"] for s in stages]
    assert face_up == sorted(face_up)
    for stage in stages:
        p = stage["params"]
        total = p["standing_prob"] + p["sitting_prob"] + p["face_down_prob"] + p["face_up_prob"]
        assert abs(total - 1.0) < 1e-9
        assert p["sitting_prob"] == 0.0
        assert p["face_down_prob"] == 0.0
        assert p["standing_prob"] > 0.0


def test_task_is_registered_with_backlash_twins():
    from mjlab.tasks.registry import list_tasks

    import mjlab_microduck.tasks  # noqa: F401

    tasks = list_tasks()
    for task_id in (
        "Mjlab-PlayDead-Flat-MicroDuck",
        "Mjlab-PlayDead-Rough-MicroDuck",
        "Mjlab-PlayDead-Flat-Backlash-MicroDuck",
        "Mjlab-PlayDead-Rough-Backlash-MicroDuck",
    ):
        assert task_id in tasks, task_id


def test_runner_cfg():
    assert MicroduckPlayDeadRlCfg.experiment_name == "microduck_playdead"
    assert MicroduckPlayDeadRlCfg.algorithm.symmetry_cfg is not None
    assert MicroduckPlayDeadRlCfg.actor.obs_normalization is True


def test_trunk_asset_cfgs_are_distinct_objects():
    cfg = make_microduck_playdead_env_cfg()
    names = (
        "supine_linear",
        "supine_sharp",
        "height_dead",
        "height_dead_l1",
        "com_downward_velocity",
        "gentle_impact",
        "too_fast_drop",
        "playdead_hold",
        "trunk_grounded_supine",
        "playdead_composite",
    )
    seen = [id(cfg.rewards[n].params["asset_cfg"]) for n in names]
    assert len(set(seen)) == len(seen), "asset_cfg shared across terms"


def test_supine_cos_distinguishes_back_from_face_and_headstand():
    s = 0.5**0.5
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    face_up = torch.tensor([[s, 0.0, -s, 0.0]])    # −90° pitch, belly up
    face_down = torch.tensor([[s, 0.0, s, 0.0]])   # +90° pitch, belly down
    headstand = torch.tensor([[0.0, 0.0, 1.0, 0.0]])  # 180° pitch
    assert torch.allclose(body_supine_cos_from_quat(identity), torch.tensor([0.0]), atol=1e-5)
    assert torch.allclose(body_supine_cos_from_quat(face_up), torch.tensor([1.0]), atol=1e-5)
    assert torch.allclose(body_supine_cos_from_quat(face_down), torch.tensor([-1.0]), atol=1e-5)
    # Headstand is inverted but NOT play-dead — this is the bug inverted_linear had.
    assert torch.allclose(body_supine_cos_from_quat(headstand), torch.tensor([0.0]), atol=1e-5)


def test_playdead_composite_sit_basin_is_dead():
    """Sitting / standing / face-down must not score the goal product."""
    dead = playdead_composite_from_values(
        supine_cos=torch.tensor([1.0]),
        z=torch.tensor([DEAD_Z]),
        ang_norm=torch.tensor([0.0]),
        target_height=DEAD_Z,
    )
    sit = playdead_composite_from_values(
        supine_cos=torch.tensor([0.0]),
        z=torch.tensor([_SIT_Z]),
        ang_norm=torch.tensor([0.0]),
        target_height=DEAD_Z,
    )
    stand = playdead_composite_from_values(
        supine_cos=torch.tensor([0.0]),
        z=torch.tensor([STAND_Z]),
        ang_norm=torch.tensor([0.0]),
        target_height=DEAD_Z,
    )
    face = playdead_composite_from_values(
        supine_cos=torch.tensor([-1.0]),
        z=torch.tensor([DEAD_Z]),
        ang_norm=torch.tensor([0.0]),
        target_height=DEAD_Z,
    )
    side = playdead_composite_from_values(
        supine_cos=torch.tensor([0.0]),
        z=torch.tensor([DEAD_Z]),
        ang_norm=torch.tensor([0.0]),
        target_height=DEAD_Z,
    )
    assert float(dead) > 0.99
    assert float(sit) < 0.05
    assert float(stand) < 0.05
    assert float(face) < 0.05
    assert float(side) < 0.05
    assert float(dead) > 10.0 * float(sit)


def test_yawed_face_up_stays_supine():
    # set_random_ground_state face_up at yaw=π/2: [s*cy, s*sy, -s*cy, s*sy]
    s = 0.5**0.5
    yaw = math.pi / 2
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    q = torch.tensor([[s * cy, s * sy, -s * cy, s * sy]])
    assert torch.allclose(body_supine_cos_from_quat(q), torch.tensor([1.0]), atol=1e-4)

"""Headstand cfg invariants (CPU, no GPU). Locks the physics-derived constants,
the reward sign convention, the gate wiring and the 61D obs contract."""

import math
import re

import mujoco
import numpy as np
import torch

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_headstand_env_cfg import (
    HEADSTAND_OVERRIDES,
    HEADSTAND_Z,
    MicroduckHeadstandKickupRlCfg,
    make_microduck_headstand_env_cfg,
)
from mjlab_microduck.tasks.microduck_backroll_env_cfg import make_microduck_backroll_env_cfg
from mjlab_microduck.robot.microduck_constants import HOME_FRAME, MICRODUCK_ALLCOLLISIONS_XML

SCENE_XML = MICRODUCK_ALLCOLLISIONS_XML.parent / "scene_allcollisions.xml"
SERVOS = ("left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
          "neck_pitch", "head_pitch", "head_yaw", "head_roll",
          "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle")


def home_joints() -> np.ndarray:
    """HOME as the robot cfg defines it, resolved per servo name."""
    out = np.zeros(14)
    for i, name in enumerate(SERVOS):
        for pattern, value in HOME_FRAME.joint_pos.items():
            if re.fullmatch(pattern, name):
                out[i] = value
    return out


def floor_bodies_of(m, d) -> set:
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    touching = set()
    for i in range(d.ncon):
        c = d.contact[i]
        if floor in (c.geom1, c.geom2):
            other = c.geom2 if c.geom1 == floor else c.geom1
            touching.add(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[other]))
    return touching


def settle(m, d, qpos, ctrl=None, seconds=3.0):
    """Hold ctrl (the pose's servo targets by default) for `seconds` from qpos."""
    d.qpos[:] = qpos
    d.qvel[:] = 0.0
    d.ctrl[:] = qpos[7:] if ctrl is None else ctrl
    for _ in range(int(seconds / m.opt.timestep)):
        mujoco.mj_step(m, d)

HEADSTAND_TASKS = (
    "Mjlab-HeadstandFold-Flat-MicroDuck",
    "Mjlab-HeadstandKickupLegsTogether-Flat-MicroDuck",
    "Mjlab-HeadstandKickup-Flat-MicroDuck",
    "Mjlab-HeadstandSplitSwitch-Flat-MicroDuck",
    "Mjlab-HeadstandBackrollLegsTogether-Flat-MicroDuck",
    "Mjlab-HeadstandSplitOver-Flat-MicroDuck",
)
# Terms whose function returns ≥ 0 and whose weight must therefore be < 0, and
# the positive task terms, across the six cfgs.
COST_TERMS = {
    "headstand_feet_down", "headstand_other_contact", "headstand_airborne", "headstand_overspeed",
    "headstand_not_inverted", "head_impact", "action_rate_l2",
    "body_ang_vel", "angular_momentum", "self_collisions", "fold_overshoot", "roulade_tuck",
    "roulade_overspeed",
}
PAY_TERMS = {"headstand_progress", "headstand_composite", "headstand_inverted_sharp", "fold_progress",
             "fold_composite", "roulade_progress"}


def test_uses_the_allcollisions_model():
    # The gates need to know WHICH body touched the floor (thigh, shin, trunk),
    # and those bodies only collide on the allcollisions model.
    cfg = make_microduck_headstand_env_cfg()
    from mjlab_microduck.robot.microduck_constants import MICRODUCK_ALLCOLLISIONS_ROBOT_CFG
    assert cfg.scene.entities["robot"] is MICRODUCK_ALLCOLLISIONS_ROBOT_CFG


def test_hold_pose_is_inside_the_hard_joint_limits():
    # The sweep's 1.6 rad left hip sat ON the hard limit and is not used. The
    # neck (1.0 of 1.047) is the one joint past its 0.9 soft limit: that only
    # costs dof_pos_limits 0.06/step against a hold paying ~5/step, and the
    # in-soft-limit alternatives land half as often (the settle sweep).
    m = mujoco.MjModel.from_xml_path(str(MICRODUCK_ALLCOLLISIONS_XML))
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)]
    servo = [n for n in names if n != "trunk_base_freejoint"]
    assert len(servo) == 14 and len(HEADSTAND_OVERRIDES) == 14
    for idx, angle in HEADSTAND_OVERRIDES.items():
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, servo[idx])
        lo, hi = m.jnt_range[jid]
        assert lo + 0.04 <= angle <= hi - 0.04, (servo[idx], angle, lo, hi)


def test_hold_pose_balances_on_the_head_alone():
    # AGENTS.md step 2, locked: drop the hold pose inverted, hold 3 s, and it
    # rests on jaw_soft alone, nose down, near inverted at HEADSTAND_Z. With
    # ±8° pitch and 0.05 rad joint noise (the spawn's), about half the drops
    # still land, which is why the policy balances actively.
    m = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    d = mujoco.MjData(m)
    joints = np.zeros(14)
    for idx, angle in HEADSTAND_OVERRIDES.items():
        joints[idx] = angle
    qpos = np.concatenate([[0.0, 0.0, 0.135], [math.cos(math.pi / 2), 0.0, math.sin(math.pi / 2), 0.0], joints])
    settle(m, d, qpos)
    R = d.xmat[1].reshape(3, 3)
    assert R[2, 2] < -0.85, R[2, 2]                    # inverted
    assert R[2, 0] < -0.3, R[2, 0]                     # nose down: the forward-fold gate is open
    assert floor_bodies_of(m, d) == {"jaw_soft"}
    assert abs(d.xpos[1][2] - HEADSTAND_Z) < 0.01, d.xpos[1][2]
    rng = np.random.default_rng(0)
    landed = 0
    for _ in range(30):
        pitch = math.pi + rng.uniform(-math.radians(8), math.radians(8))
        noisy = np.concatenate([[0.0, 0.0, 0.135], [math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0], joints + rng.normal(0, 0.05, 14)])
        settle(m, d, noisy, ctrl=joints)
        landed += d.xmat[1].reshape(3, 3)[2, 2] < -0.85 and floor_bodies_of(m, d) == {"jaw_soft"}
    assert landed >= 10, f"{landed} of 30 noisy drops landed"


def test_pike_rests_where_it_was_measured():
    # The kick-ups spawn in _HEADSTAND_PIKE_QPOS as a resting state: hold its
    # ctrl for 3 s and it stays a pike, head and both feet down, at its angle.
    m = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    d = mujoco.MjData(m)
    settle(m, d, microduck_mdp._HEADSTAND_PIKE_QPOS.numpy())
    assert floor_bodies_of(m, d) >= {"jaw_soft", "ankle_left", "ankle_right"}, floor_bodies_of(m, d)
    R = d.xmat[1].reshape(3, 3)
    pitch = math.degrees(math.atan2(-R[2, 0], R[2, 2]))
    assert abs(pitch - math.degrees(microduck_mdp._HEADSTAND_PIKE_PITCH)) < 5.0, pitch
    assert abs(d.xpos[1][2] - float(microduck_mdp._HEADSTAND_PIKE_QPOS[2])) < 0.01


def test_reward_signs_follow_the_convention():
    # mjlab-style costs (≥ 0) take a negative weight; the self-negating
    # trunk_vertical_accel_penalty takes a POSITIVE weight. A wrong sign here
    # pays for the violation and the policy farms it. Checked on all six cfgs.
    from mjlab.tasks.registry import load_env_cfg
    seen_costs, seen_pays = set(), set()
    for task in HEADSTAND_TASKS:
        cfg = load_env_cfg(task)
        for name, term in cfg.rewards.items():
            if name in COST_TERMS:
                assert term.weight < 0.0, (task, name); seen_costs.add(name)
            elif name in PAY_TERMS:
                assert term.weight > 0.0, (task, name); seen_pays.add(name)
        if "gentle" in cfg.rewards:
            assert cfg.rewards["gentle"].weight > 0.0
            assert cfg.rewards["gentle"].func is microduck_mdp.trunk_vertical_accel_penalty
        assert "upright" not in cfg.rewards, task   # always-on upright would oppose the trick
    assert seen_costs == COST_TERMS and seen_pays == PAY_TERMS
    cfg = make_microduck_headstand_env_cfg()
    assert cfg.rewards["headstand_arrival_damping"].weight == 0.0  # curriculum introduces it


def test_sensors_are_registered_under_the_names_mdp_reads():
    cfg = make_microduck_headstand_env_cfg()
    names = {s.name for s in cfg.scene.sensors}
    for expected in (microduck_mdp._HEADSTAND_HEAD_SENSOR, microduck_mdp._HEADSTAND_FEET_SENSOR,
                     microduck_mdp._HEADSTAND_OTHER_SENSOR, microduck_mdp._HEADSTAND_SUPPORT_SENSOR):
        assert expected in names, expected


def test_other_contact_pattern_excludes_head_and_feet_only():
    import re
    cfg = make_microduck_headstand_env_cfg()
    other = next(s for s in cfg.scene.sensors if s.name == microduck_mdp._HEADSTAND_OTHER_SENSOR)
    pat = re.compile(other.primary.pattern)
    for allowed in ("jaw_soft", "yaw_roll_motion", "neck_pitch", "ankle_left", "ankle_right"):
        assert not pat.match(allowed), allowed
    for flop in ("trunk_base", "upper_leg_left", "upper_leg_right", "leg", "leg_2", "neck", "hip_l"):
        assert pat.match(flop), flop


def test_spawn_mix_moves_toward_the_pike_and_never_stands():
    # The kick-up's reverse curriculum, as shares of episodes (the weights are
    # normalised by their sum): the hold and the swing shrink, the pikes grow.
    cfg = make_microduck_headstand_env_cfg()
    stages = cfg.curriculum["headstand_spawn_mix"].params["param_stages"]
    def shares(p):
        total = sum(p.values())
        return {k: v / total for k, v in p.items()}
    sh = [shares(s["params"]) for s in stages]
    assert all(x["standing_prob"] == 0.0 for x in sh)
    assert [x["hold_prob"] for x in sh] == sorted([x["hold_prob"] for x in sh], reverse=True)
    pikes = [x["pike_prob"] + x["handover_prob"] for x in sh]
    assert pikes == sorted(pikes)
    assert abs(sh[0]["handover_prob"] - 0.385) < 0.01 and abs(sh[0]["hold_prob"] - 0.23) < 0.01   # the published mix
    assert stages[0]["params"] == {k: cfg.events["set_headstand_spawn"].params[k] for k in stages[0]["params"]}


def test_spawn_height_table_covers_the_partway_range():
    assert len(microduck_mdp._HEADSTAND_PARTWAY_PITCH_DEG) == len(microduck_mdp._HEADSTAND_PARTWAY_Z)
    assert float(microduck_mdp._HEADSTAND_PARTWAY_PITCH_DEG[0]) == 90.0
    assert float(microduck_mdp._HEADSTAND_PARTWAY_PITCH_DEG[-1]) == 180.0


def test_symmetry_is_off_for_the_asymmetric_split():
    from mjlab_microduck.tasks.microduck_headstand_env_cfg import (
        MicroduckHeadstandFoldRlCfg, MicroduckHeadstandSplitSwitchRlCfg,
    )
    assert MicroduckHeadstandKickupRlCfg.algorithm.symmetry_cfg is None
    assert MicroduckHeadstandSplitSwitchRlCfg.algorithm.symmetry_cfg is None
    assert MicroduckHeadstandFoldRlCfg.algorithm.symmetry_cfg is not None
    assert MicroduckHeadstandKickupRlCfg.experiment_name == "microduck_headstand_kickup"


def test_pushes_and_fall_termination_are_off():
    cfg = make_microduck_headstand_env_cfg()
    assert "push_robot" not in cfg.events
    assert "fell_over" not in cfg.terminations
    assert "nan_state" in cfg.terminations


def test_obs_parity_with_standup():
    # The ONNX must load in the runtime's policy slot: same term order as the
    # standup policy, group by group.
    from mjlab_microduck.tasks.microduck_standup_env_cfg import make_microduck_standup_env_cfg
    head = make_microduck_headstand_env_cfg()
    stand = make_microduck_standup_env_cfg()
    for grp in ("actor", "critic"):
        assert list(head.observations[grp].terms.keys()) == list(stand.observations[grp].terms.keys()), grp


# ── Contact model, sensors, spawn table ──────────────────────────────────────

def test_contact_solver_budget_matches_the_full_collision_tasks():
    cfg = make_microduck_headstand_env_cfg()
    assert cfg.sim.nconmax >= 200
    assert cfg.sim.mujoco.iterations >= 30 and cfg.sim.mujoco.ls_iterations >= 50


def test_feet_sensor_covers_the_whole_ankle_bodies():
    # The servo housing on the ankle sits below the sole when pitched; a
    # sole-only sensor let a head + housing rest pass every gate.
    cfg = make_microduck_headstand_env_cfg()
    feet = next(s for s in cfg.scene.sensors if s.name == microduck_mdp._HEADSTAND_FEET_SENSOR)
    assert feet.primary.mode == "body"
    import re
    assert re.fullmatch(feet.primary.pattern, "ankle_left") and re.fullmatch(feet.primary.pattern, "ankle_right")


def test_head_sensor_carries_force_for_the_impact_penalty():
    cfg = make_microduck_headstand_env_cfg()
    head = next(s for s in cfg.scene.sensors if s.name == microduck_mdp._HEADSTAND_HEAD_SENSOR)
    assert "force" in head.fields and "found" in head.fields
    assert cfg.rewards["head_impact"].params["sensor_name"] == head.name


def test_partway_spawns_start_above_the_floor():
    # Every (pitch, lerp, roll, noise) the spawn can draw must leave the lowest
    # collision corner above the floor. The first table put 51% inside it.
    import itertools

    def lowest_point_z(m, d):
        # Lowest corner of any collision geom's bounding box, in world frame.
        lowest = np.inf
        for g in range(1, m.ngeom):
            if not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue
            center, half = m.geom_aabb[g][:3], m.geom_aabb[g][3:]
            rot = d.geom_xmat[g].reshape(3, 3)
            for signs in itertools.product((-1, 1), repeat=3):
                corner = d.geom_xpos[g] + rot @ (center + half * np.array(signs))
                lowest = min(lowest, corner[2])
        return lowest
    m = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    d = mujoco.MjData(m)
    cfg = make_microduck_headstand_env_cfg()
    p = cfg.events["set_headstand_spawn"].params
    target = np.zeros(14)
    for idx, angle in HEADSTAND_OVERRIDES.items():
        target[idx] = angle
    home = home_joints()
    rng = np.random.default_rng(3)
    table_deg = microduck_mdp._HEADSTAND_PARTWAY_PITCH_DEG.numpy()
    table_z = microduck_mdp._HEADSTAND_PARTWAY_Z.numpy()
    below = 0
    for _ in range(300):
        pitch = rng.uniform(p["partway_pitch_min"], p["partway_pitch_max"])
        u = rng.uniform(*p["partway_lerp_range"])
        roll = rng.uniform(-math.radians(5), math.radians(5))
        q = home + u * (target - home) + rng.normal(0, p["joint_noise_std"], 14)
        z = np.interp(math.degrees(pitch), table_deg, table_z)
        cp, sp, cr, sr = math.cos(pitch / 2), math.sin(pitch / 2), math.cos(roll / 2), math.sin(roll / 2)
        d.qpos[7:] = q
        d.qpos[3:7] = [cr * cp, sr * cp, cr * sp, -sr * sp]
        d.qpos[0:3] = [0.0, 0.0, z]
        mujoco.mj_forward(m, d)
        below += lowest_point_z(m, d) < 0.0
    assert below == 0, f"{below} of 300 partway spawns start inside the floor"


def test_partway_spawns_begin_at_the_pike():
    cfg = make_microduck_headstand_env_cfg()
    assert math.radians(90.0) <= cfg.events["set_headstand_spawn"].params["partway_pitch_min"] <= math.radians(100.0)
    assert abs(cfg.events["set_headstand_spawn"].params["hold_z"] - HEADSTAND_Z) < 0.005


def test_curricula_are_paced_like_the_roulade():
    cfg = make_microduck_headstand_env_cfg()
    mix = [s["step"] for s in cfg.curriculum["headstand_spawn_mix"].params["param_stages"]]
    assert mix[1] >= 1000 * 24 and mix[-1] >= 2000 * 24
    for name in ("arrival_damping_weight", "torque_rate_weight"):
        first_nonzero = next(s["step"] for s in cfg.curriculum[name].params["weight_stages"] if s["weight"] != 0.0)
        assert first_nonzero >= 2500 * 24, name
    ladder = [s["weight"] for s in cfg.curriculum["action_rate_weight"].params["weight_stages"]]
    assert ladder[0] == cfg.rewards["action_rate_l2"].weight
    assert abs(ladder[-1]) <= 0.5  # standup's -1.0 scaled by this task's reward mass
    assert MicroduckHeadstandKickupRlCfg.max_iterations >= 6000


def test_sharp_term_is_flat_inside_the_rest_tilt():
    # 1-cos(20°) is the plateau edge; a perfect vertical must not pay more
    # than the measured rest tilt does.
    assert abs(microduck_mdp._HEADSTAND_REST_TILT_COS - math.cos(math.radians(20.0))) < 1e-9
    cfg = make_microduck_headstand_env_cfg()
    params = cfg.rewards["headstand_inverted_sharp"].params
    assert params["target_overrides"] is HEADSTAND_OVERRIDES and params["knee_zero"] > params["knee_full"] > 0.0


def test_not_inverted_cost_always_on_and_slam_gate_set():
    cfg = make_microduck_headstand_env_cfg()
    stages = cfg.curriculum["not_inverted_weight"].params["weight_stages"]
    assert stages[0]["weight"] == cfg.rewards["headstand_not_inverted"].weight < 0.0
    slam_n = cfg.rewards["headstand_composite"].params["slam_n"]
    assert 7.2 < slam_n < 15.0   # above the body weight, below a slam
    assert cfg.rewards["headstand_progress"].params["slam_n"] == slam_n
    # a 21 N landing must cost more than one step of the hold
    assert -cfg.rewards["head_impact"].weight * (21.0 - cfg.rewards["head_impact"].params["threshold"]) > 5.5


def test_forward_fold_gate_is_zero_for_a_backward_drop():
    # A backward drop puts the head top down too; the latch/swing/hold gates
    # must read 0 nose-up and 1 nose-down or vertical.
    from types import SimpleNamespace
    def asset_with_pitch(deg):
        p = math.radians(deg)
        q = torch.tensor([[math.cos(p / 2), 0.0, math.sin(p / 2), 0.0]])
        return SimpleNamespace(data=SimpleNamespace(root_link_quat_w=q))
    g = microduck_mdp._forward_fold_gate
    assert float(g(asset_with_pitch(0))) == 1.0        # standing
    assert float(g(asset_with_pitch(100))) == 1.0      # forward fold, nose down
    assert float(g(asset_with_pitch(180))) == 1.0      # inverted
    assert float(g(asset_with_pitch(-100))) == 0.0     # fallen on the back
    assert float(g(asset_with_pitch(-135))) == 0.0


def test_kickup_never_spawns_standing_and_starts_in_the_pike():
    cfg = make_microduck_headstand_env_cfg(style="split")
    p = cfg.events["set_headstand_spawn"].params
    assert p["standing_prob"] == 0.0
    assert math.radians(95.0) <= p["partway_pitch_min"] <= math.radians(105.0)   # partway starts just past the pike
    assert p["partway_lerp_range"][0] == 0.0
    for stage in cfg.curriculum["headstand_spawn_mix"].params["param_stages"]:
        assert stage["params"]["standing_prob"] == 0.0
    assert MicroduckHeadstandKickupRlCfg.experiment_name == "microduck_headstand_kickup"


def test_flop_terminates_instead_of_costing_per_step():
    cfg = make_microduck_headstand_env_cfg(style="split")
    assert "flopped" in cfg.terminations and cfg.terminations["flopped"].time_out is False
    assert cfg.rewards["headstand_other_contact"].weight > -0.5  # a one-off, not a per-step cost


class _FakeSensor:
    def __init__(self, found, force=None):
        self.data = type("D", (), {})()
        self.data.found = found
        self.data.force = force


def _fake_env(inverted_cos, head=True, feet=False, other=False, support=True, force=0.0, step=100):
    """A one-env stand-in with the sensors the headstand terms read and a
    trunk quaternion at the given inversion (rotation about y)."""
    from types import SimpleNamespace
    pitch = math.acos(-inverted_cos)   # inverted_cos = -cos(pitch)
    q = torch.tensor([[math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0]])
    asset = SimpleNamespace(data=SimpleNamespace(root_link_quat_w=q))
    one = lambda v: torch.tensor([[1.0 if v else 0.0]])
    sensors = {
        microduck_mdp._HEADSTAND_HEAD_SENSOR: _FakeSensor(one(head), torch.tensor([[[0.0, 0.0, force]]])),
        microduck_mdp._HEADSTAND_FEET_SENSOR: _FakeSensor(one(feet)),
        microduck_mdp._HEADSTAND_OTHER_SENSOR: _FakeSensor(one(other)),
        microduck_mdp._HEADSTAND_SUPPORT_SENSOR: _FakeSensor(one(support)),
    }
    scene = type("Scene", (), {"sensors": sensors, "__getitem__": lambda self_, k: asset})()
    env = SimpleNamespace(num_envs=1, device="cpu", step_dt=0.02, common_step_counter=step,
                          episode_length_buf=torch.tensor([step]), scene=scene)
    return env, asset


def test_swing_frontier_pays_once_and_never_charges_falling_back(monkeypatch):
    monkeypatch.setattr(microduck_mdp, "_head_top_down", lambda env, asset: torch.tensor([True]))
    env, asset = _fake_env(inverted_cos=0.5)
    microduck_mdp._headstand_state(env)
    env._headstand_max_inverted[:] = 0.2
    pay = microduck_mdp.headstand_progress(env)
    assert abs(float(pay) - 0.3) < 1e-6          # 0.2 → 0.5: the new frontier pays once
    env.common_step_counter += 1
    assert float(microduck_mdp.headstand_progress(env)) == 0.0   # holding pays nothing
    env2, _ = _fake_env(inverted_cos=0.1, step=102)
    env2._headstand_head_latch = env._headstand_head_latch; env2._headstand_slammed = env._headstand_slammed
    env2._headstand_max_inverted = env._headstand_max_inverted
    env2._headstand_last_update_step = -1
    assert float(microduck_mdp.headstand_progress(env2)) == 0.0   # falling back is never charged


def test_hold_gates_need_the_head_alone_and_a_gentle_arrival(monkeypatch):
    monkeypatch.setattr(microduck_mdp, "_head_top_down", lambda env, asset: torch.tensor([True]))
    env, asset = _fake_env(inverted_cos=1.0, force=5.0)
    microduck_mdp._update_headstand(env, asset, slam_n=12.0)
    assert float(microduck_mdp._headstand_on_head_alone(env)) == 1.0
    assert float(microduck_mdp._headstand_arrived_gently(env)) == 1.0
    env_feet, _ = _fake_env(inverted_cos=1.0, feet=True)
    assert float(microduck_mdp._headstand_on_head_alone(env_feet)) == 0.0     # a foot down is a prop
    env_slam, asset_slam = _fake_env(inverted_cos=1.0, force=30.0)
    microduck_mdp._update_headstand(env_slam, asset_slam, slam_n=12.0)
    assert float(microduck_mdp._headstand_arrived_gently(env_slam)) == 0.0    # a slam forfeits the hold
    env_grace, asset_grace = _fake_env(inverted_cos=1.0, force=30.0, step=5)
    microduck_mdp._update_headstand(env_grace, asset_grace, slam_n=12.0)
    assert float(microduck_mdp._headstand_arrived_gently(env_grace)) == 1.0   # inside the spawn grace


def test_missing_sensor_raises_instead_of_paying_everywhere():
    import pytest
    env, asset = _fake_env(inverted_cos=1.0)
    del env.scene.sensors[microduck_mdp._HEADSTAND_FEET_SENSOR]
    with pytest.raises(KeyError):
        microduck_mdp._headstand_on_head_alone(env)


def test_fold_progress_pays_only_past_the_frontier(monkeypatch):
    # A fold episode spawned in the pike must not be paid the whole fold for
    # doing nothing: the frontier starts where the spawn is.
    monkeypatch.setattr(microduck_mdp, "_head_top_down", lambda env, asset: torch.tensor([True]))
    target = float(microduck_mdp._HEADSTAND_PIKE_PITCH)
    env, asset = _fake_env(inverted_cos=-math.cos(target), feet=True)
    microduck_mdp._headstand_state(env)
    env._fold_max_pitch[:] = target                       # what reset_headstand_spawn sets for a pike spawn
    assert float(microduck_mdp.fold_progress(env, target_pitch=target)) == 0.0
    env2, _ = _fake_env(inverted_cos=-math.cos(0.5), feet=True, step=101)
    microduck_mdp._headstand_state(env2)
    env2._fold_max_pitch[:] = 0.0                         # a standing spawn
    assert abs(float(microduck_mdp.fold_progress(env2, target_pitch=target)) - 0.5) < 1e-5


def test_pike_gate_needs_head_and_both_feet():
    env, _ = _fake_env(inverted_cos=-math.cos(1.3), feet=True)
    two_feet = torch.tensor([[1.0, 1.0]])
    env.scene.sensors[microduck_mdp._HEADSTAND_FEET_SENSOR].data.found = two_feet
    assert float(microduck_mdp._pike_gate(env)) == 1.0
    env.scene.sensors[microduck_mdp._HEADSTAND_FEET_SENSOR].data.found = torch.tensor([[1.0, 0.0]])
    assert float(microduck_mdp._pike_gate(env)) == 0.0    # one foot is not the pike


def test_kickup_spawns_in_the_pike():
    cfg = make_microduck_headstand_env_cfg(style="split")
    p = cfg.events["set_headstand_spawn"].params
    assert p["pike_prob"] > 0.0 and p["standing_prob"] == 0.0
    assert math.radians(70) < microduck_mdp._HEADSTAND_PIKE_PITCH < math.radians(80)
    assert microduck_mdp._HEADSTAND_PIKE_QPOS.shape == (21,)
    assert 0.09 < float(microduck_mdp._HEADSTAND_PIKE_QPOS[2]) < 0.12


def test_legs_together_style_is_symmetric():
    from mjlab_microduck.tasks.microduck_headstand_env_cfg import (
        MicroduckHeadstandKickupLegsTogetherRlCfg, HEADSTAND_LEGS_TOGETHER_OVERRIDES,
    )
    cfg = make_microduck_headstand_env_cfg(style="legs_together")
    assert cfg.rewards["headstand_composite"].params["style"] == "legs_together"
    assert MicroduckHeadstandKickupLegsTogetherRlCfg.algorithm.symmetry_cfg is not None
    ov = cfg.rewards["headstand_composite"].params["target_overrides"]
    assert abs(ov[2] + ov[11]) < 1e-9   # hips symmetric: no split
    assert HEADSTAND_LEGS_TOGETHER_OVERRIDES[3] == 0.0
    assert MicroduckHeadstandKickupRlCfg.algorithm.symmetry_cfg is None   # the split stays asymmetric


def test_fold_has_no_standing_tax_and_no_flop_termination():
    cfg = make_microduck_headstand_env_cfg(style="fold")
    assert "fold_not_folded" not in cfg.rewards and "flopped" not in cfg.terminations
    assert cfg.rewards["fold_progress"].weight >= 5.0 and cfg.rewards["fold_composite"].weight > 0
    assert cfg.rewards["headstand_other_contact"].weight < 0


def test_registered_kickups_carry_the_published_settings():
    # The registered task ids are the envs that produced the published
    # policies: the rotation cap at 2 rad/s, half the pike spawns from the handover set.
    from mjlab.tasks.registry import load_env_cfg
    for task in ("Mjlab-HeadstandKickup-Flat-MicroDuck", "Mjlab-HeadstandKickupLegsTogether-Flat-MicroDuck"):
        cfg = load_env_cfg(task)
        assert cfg.rewards["headstand_overspeed"].params["omega_max"] == 2.0
        assert cfg.rewards["headstand_overspeed"].weight == -0.5
        assert cfg.events["set_headstand_spawn"].params["handover_prob"] == 0.5
    cfg = make_microduck_headstand_env_cfg(omega_max=6.0, handover_prob=0.0)
    assert cfg.rewards["headstand_overspeed"].params["omega_max"] == 6.0
    assert cfg.events["set_headstand_spawn"].params["handover_prob"] == 0.0


def test_backroll_starts_in_the_hold_and_never_standing():
    cfg = make_microduck_backroll_env_cfg(style="legs_together")
    p = cfg.events["set_roulade_state"].params
    assert p["standing_prob"] == 0.0 and p["midroll_prob"] == 1.0
    assert math.radians(160) < p["midroll_pitch_min"] < p["midroll_pitch_max"] < math.radians(195)
    assert p["tuck_factor_range"][1] == 1.0
    from mjlab_microduck.robot.microduck_constants import MICRODUCK_ALLCOLLISIONS_ROBOT_CFG
    assert cfg.scene.entities["robot"] is MICRODUCK_ALLCOLLISIONS_ROBOT_CFG
    for stage in cfg.curriculum["roulade_spawn_mix"].params["param_stages"]:
        assert stage["params"]["standing_prob"] == 0.0


def test_every_task_ends_the_actor_obs_with_the_command_block():
    from mjlab.tasks.registry import load_env_cfg
    for task in HEADSTAND_TASKS:
        cfg = load_env_cfg(task)
        terms = list(cfg.observations["actor"].terms)
        assert terms[-3:] == ["command", "head_command", "body_command"], task


def test_split_switch_flag_is_zero_on_a_fresh_episode_and_holds_the_other_slots():
    # Built for real: the reset resamples the command before episode_length_buf
    # is zeroed, so the guard has to run in compute(), on the first step.
    from mjlab.envs import ManagerBasedRlEnv
    cfg = make_microduck_headstand_env_cfg(play=True, style="split", switch=True)
    cfg.scene.num_envs = 4
    cfg.commands["twist"].flip_prob = 1.0                   # every resample sets the flag
    cfg.commands["twist"].resampling_time_range = (0.3, 0.3)
    cfg.curriculum.clear()
    for n in list(cfg.terminations):
        if n != "time_out":
            del cfg.terminations[n]
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    obs, _ = env.reset()
    assert obs["actor"].shape[-1] == 61                      # the shared obs contract, numerically
    zeros = torch.zeros(4, 14)
    env.step(zeros)
    cmd = env.command_manager.get_command("twist")
    assert torch.all(cmd[:, 0] == 0.0), cmd                 # fresh episodes start at 0
    assert torch.all(cmd[:, 1:] == 0.0)
    for _ in range(40):
        env.step(zeros)                                      # past the 0.3 s resample
    assert torch.all(env.command_manager.get_command("twist")[:, 0] == 1.0)
    alpha = env.command_manager.get_term("twist").alpha
    assert torch.all(alpha > 0.5)                            # slewing toward 1 at 1/ramp_s per second
    env.reset()
    env.step(zeros)
    assert torch.all(env.command_manager.get_command("twist")[:, 0] == 0.0)
    env.close()


def test_mirror_overrides_swaps_legs_and_flips_signs():
    m = microduck_mdp._mirror_overrides(HEADSTAND_OVERRIDES)
    assert m[2] == -HEADSTAND_OVERRIDES[11] and m[11] == -HEADSTAND_OVERRIDES[2]   # hip pitches cross
    assert m[5] == HEADSTAND_OVERRIDES[5] and m[6] == HEADSTAND_OVERRIDES[6]       # neck and head pitch stay
    assert microduck_mdp._mirror_overrides({2: 1.0}) == {11: -1.0}                 # partial dicts work


def test_handover_set_has_the_pike_shape():
    qpos, qvel = microduck_mdp._handover_states("cpu")
    assert qpos.shape == (512, 21) and qvel.shape == (512, 20)
    assert 0.09 < float(qpos[:, 2].mean()) < 0.12
    w, x, y, z = qpos[:, 3], qpos[:, 4], qpos[:, 5], qpos[:, 6]
    pitch = torch.atan2(-2 * (x * z - w * y), 1 - 2 * (x * x + y * y))
    assert 65 < math.degrees(float(pitch.mean())) < 85           # the pike's trunk angle
    cfg = make_microduck_headstand_env_cfg(style="split", handover_prob=0.5)
    p = cfg.events["set_headstand_spawn"].params
    assert p["handover_prob"] == 0.5 and p["pike_prob"] == 0.2


def test_polish_at_zero_applies_final_weights_from_the_start():
    cfg = make_microduck_headstand_env_cfg(style="split", polish_at=0)
    for name in ("action_rate_weight", "arrival_damping_weight", "torque_rate_weight"):
        stages = cfg.curriculum[name].params["weight_stages"]
        assert len(stages) == 1 and stages[0]["step"] == 0, name
    assert cfg.curriculum["action_rate_weight"].params["weight_stages"][0]["weight"] == -0.46


def test_splitover_spawns_both_splits_and_gates_progress():
    cfg = make_microduck_backroll_env_cfg(style="splitover", tuck_w=-6.0)
    assert cfg.events["set_roulade_state"].params["mirror_prob"] == 0.5
    assert cfg.rewards["roulade_progress"].func is microduck_mdp.roulade_progress_split
    assert cfg.rewards["roulade_tuck"].weight == -6.0
    straight = make_microduck_backroll_env_cfg(style="legs_together")
    assert "roulade_tuck" not in straight.rewards and "mirror_prob" not in straight.events["set_roulade_state"].params


def test_split_over_window_and_leg_gate(monkeypatch):
    # The two factors of roulade_tuck_cost, (1 - legs split and straight) and
    # (accumulator inside the window), each checked on its own.
    from types import SimpleNamespace
    def q_of(joints):
        monkeypatch.setattr(microduck_mdp, "_servo_joint_pos", lambda env, asset: torch.tensor([joints], dtype=torch.float32))
    env = SimpleNamespace(num_envs=1, device="cpu")
    microduck_mdp._roulade_state(env)
    lo, hi = math.radians(170.0), math.radians(330.0)
    env._roulade_accum[:] = math.radians(250.0)
    assert float(microduck_mdp._roulade_window(env, lo, hi)) == 1.0
    env._roulade_accum[:] = math.radians(350.0)
    assert float(microduck_mdp._roulade_window(env, lo, hi)) == 0.0
    hold = [0.0] * 14; hold[2], hold[11] = 1.2, 0.8
    q_of(hold)
    assert float(microduck_mdp._legs_split_gate(None, None)) == 1.0
    tuck = [0.0] * 14; tuck[3], tuck[12] = 1.2, 1.2
    q_of(tuck)
    assert float(microduck_mdp._legs_split_gate(None, None)) == 0.0


def _built_env(cfg, num_envs):
    from mjlab.envs import ManagerBasedRlEnv
    cfg.scene.num_envs = num_envs
    cfg.curriculum.clear()
    for n in list(cfg.terminations):
        if n != "time_out":
            del cfg.terminations[n]
    return ManagerBasedRlEnv(cfg=cfg, device="cpu")


def test_reset_headstand_spawn_sets_the_frontiers_and_the_latch_per_bucket():
    # Built for real: each bucket forced to weight 1, reset, and the state the
    # rewards read afterwards checked against what the spawn promises.
    from mjlab_microduck.tasks import mdp as m
    env = _built_env(make_microduck_headstand_env_cfg(play=True, style="split"), 8)
    term = env.event_manager.get_term_cfg("set_headstand_spawn")
    asset = env.scene["robot"]
    for bucket in ("standing", "partway", "hold", "pike", "handover"):
        for k in ("standing_prob", "partway_prob", "hold_prob", "pike_prob", "handover_prob"):
            term.params[k] = 1.0 if k == f"{bucket}_prob" else 0.0
        env.reset()
        inv = m._inverted_cos(asset)
        pitch = m._trunk_pitch(asset)
        if bucket == "hold":
            pitch = torch.where(pitch < 0, pitch + 2 * math.pi, pitch)   # a hold past 180° wraps negative
        assert torch.allclose(env._headstand_max_inverted, inv, atol=0.05), bucket      # the frontier is the spawn
        assert torch.all(env._fold_max_pitch <= float(m._HEADSTAND_PIKE_PITCH) + 1e-4), bucket
        assert torch.allclose(env._fold_max_pitch, torch.clamp(pitch, max=float(m._HEADSTAND_PIKE_PITCH)), atol=0.05), bucket
        assert not torch.any(env._headstand_slammed), bucket
        if bucket == "hold":
            assert torch.all(env._headstand_head_latch) and torch.all(inv > 0.9), bucket
        if bucket in ("pike", "handover"):
            assert not torch.any(env._headstand_head_latch), bucket
            assert torch.all((pitch > math.radians(60)) & (pitch < math.radians(95))), bucket
        if bucket == "standing":
            assert torch.all(inv < -0.95) and not torch.any(env._headstand_head_latch), bucket
        env.step(torch.zeros(8, 14))
        assert not torch.isnan(asset.data.root_link_pos_w).any(), bucket
    env.close()


def test_backroll_spawns_stay_inverted_for_a_step():
    # A hold spawn that started inside the floor would be thrown; after a
    # zero-action step every env is still inverted and finite.
    from mjlab_microduck.tasks import mdp as m
    for style in ("legs_together", "splitover"):
        env = _built_env(make_microduck_backroll_env_cfg(play=True, style=style), 16)
        env.reset()
        env.step(torch.zeros(16, 14))
        asset = env.scene["robot"]
        assert not torch.isnan(asset.data.root_link_pos_w).any()
        assert torch.all(m._inverted_cos(asset) > 0.8), style
        env.close()


def test_costs_are_non_negative_at_representative_states(monkeypatch):
    monkeypatch.setattr(microduck_mdp, "_head_top_down", lambda env, asset: torch.tensor([True]))
    for inv, feet, other, support in ((-1.0, True, False, True), (-0.24, True, False, True), (1.0, False, False, True), (0.3, False, True, True), (0.5, False, False, False)):
        env, asset = _fake_env(inverted_cos=inv, feet=feet, other=other, support=support)
        microduck_mdp._headstand_state(env)
        for fn in (microduck_mdp.headstand_not_inverted_cost, microduck_mdp.headstand_feet_down_cost,
                   microduck_mdp.headstand_other_contact_cost, microduck_mdp.headstand_airborne_cost):
            assert float(fn(env)) >= 0.0, (fn.__name__, inv)
        assert float(microduck_mdp.fold_overshoot_cost(env, target_pitch=1.3)) >= 0.0


def test_curriculum_rungs_keep_cost_weights_non_positive():
    cfg = make_microduck_headstand_env_cfg()
    for name in ("not_inverted_weight", "action_rate_weight", "arrival_damping_weight", "torque_rate_weight"):
        for rung in cfg.curriculum[name].params["weight_stages"]:
            assert rung["weight"] <= 0.0, (name, rung)
    for name in ("gentle_weight",):
        for rung in cfg.curriculum[name].params["weight_stages"]:
            assert rung["weight"] > 0.0, (name, rung)   # self-negating term: positive weight


def test_every_term_reads_the_same_slam_threshold():
    from mjlab.tasks.registry import load_env_cfg
    for task in HEADSTAND_TASKS[:4]:
        cfg = load_env_cfg(task)
        values = {name: term.params["slam_n"] for name, term in cfg.rewards.items() if "slam_n" in (term.params or {})}
        assert len(values) >= 2 and len(set(values.values())) == 1, (task, values)


def test_handover_loader_refuses_another_robot_xml(monkeypatch, tmp_path):
    import numpy as np
    import pytest
    d = dict(np.load(microduck_mdp._HEADSTAND_HANDOVER_PATH))
    d["xml_sha256"] = np.array("0" * 64)
    other = tmp_path / "handovers.npz"
    np.savez(other, **d)
    monkeypatch.setattr(microduck_mdp, "_HEADSTAND_HANDOVER_PATH", str(other))
    monkeypatch.setattr(microduck_mdp, "_headstand_handovers", {})
    with pytest.raises(ValueError):
        microduck_mdp._handover_states("cpu")


def test_mirrored_spawns_swap_the_split(monkeypatch):
    from mjlab_microduck.tasks import mdp as m
    env = _built_env(make_microduck_backroll_env_cfg(play=True, style="splitover"), 8)
    env.event_manager.get_term_cfg("set_roulade_state").params["mirror_prob"] = 1.0
    env.reset()
    q = m._servo_joint_pos(env, env.scene["robot"])
    # Mirrored: both hips behind (negative), the right leg the deeper of the two
    # (targets -0.8 and -1.2). Asserted as signs and an ordering, because the
    # spawn adds 0.08 rad of joint noise on top of a 0.9-1.0 lerp.
    assert torch.all(q[:, 2] < 0.0) and torch.all(q[:, 11] < 0.0)
    assert torch.all(q[:, 11] < q[:, 2])
    env.event_manager.get_term_cfg("set_roulade_state").params["mirror_prob"] = 0.0
    env.reset()
    q = m._servo_joint_pos(env, env.scene["robot"])
    assert torch.all(q[:, 2] > 0.0) and torch.all(q[:, 11] > 0.0)       # the original split, left leg leading
    assert torch.all(q[:, 2] > q[:, 11])
    env.close()

"""Expressive motions: references stay inside joint / trunk bounds, are smooth, start and end at the stand pose;
the env cfgs keep the contract (phase command, episode = T, grippy feet, reward signs)."""

import math

import mujoco
import numpy as np
import pytest
import torch

import mjlab_microduck.tasks  # noqa: F401  (registry first)
from mjlab_microduck.robot.microduck_constants import HOME_FRAME, get_ground_pick_spec
from mjlab_microduck.tasks.expressive_motions import CH, HEAD_JOINTS, MOTIONS, STAND_HEIGHT, bound, reference
from mjlab_microduck.tasks.microduck_expressive_env_cfg import (
    SERVO_JOINTS,
    expressive_joint_limits,
    make_microduck_expressive_env_cfg,
)

TRUNK_MAX = {"roll": 0.12, "pitch": 0.20, "yaw": 0.40}
DZ_RANGE = (-0.015, 0.003)       # HOME is near full leg extension (FK: +3 mm max up), squat ~-15 mm
HEAD_MARGIN = 0.05


def _home(jn):
    import re
    for pat, v in HOME_FRAME.joint_pos.items():
        if re.fullmatch(pat, jn):
            return v
    raise KeyError(jn)


def _ref(motion, n=2001):
    phi = torch.linspace(0.0, 1.0, n, dtype=torch.float64)
    return phi, reference(motion, phi).numpy()


@pytest.mark.parametrize("name", list(MOTIONS))
def test_reference_bounds(name):
    mo = MOTIONS[name]
    _, r = _ref(mo)
    for ch, lim in TRUNK_MAX.items():
        assert bound(mo, ch) <= lim + 1e-9, (ch, bound(mo, ch))
        assert np.abs(r[:, CH[ch]]).max() <= lim + 1e-9
    assert DZ_RANGE[0] - 1e-9 <= r[:, CH["dz"]].min() and r[:, CH["dz"]].max() <= DZ_RANGE[1] + 1e-9
    limits = expressive_joint_limits()
    for jn in HEAD_JOINTS:
        lo, hi = limits[jn]
        q = _home(jn) + r[:, CH[jn]]
        assert q.min() >= lo + HEAD_MARGIN and q.max() <= hi - HEAD_MARGIN, (jn, q.min(), q.max(), lo, hi)
    # trunk yaw on planted feet goes through both hip yaws: stay inside their ranges with margin
    yaw = np.abs(r[:, CH["yaw"]]).max()
    assert yaw <= min(-limits["left_hip_yaw"][0], limits["right_hip_yaw"][1]) - 0.05


@pytest.mark.parametrize("name", list(MOTIONS))
def test_reference_starts_and_ends_at_stand_and_is_smooth(name):
    mo = MOTIONS[name]
    phi, r = _ref(mo)
    assert np.abs(r[0]).max() < 1e-9
    assert np.abs(r[phi.numpy() >= mo.settle_from]).max() < 1e-9, "every envelope must close before the settle"
    for c in mo.comps:
        assert c.env[0] < c.env[1] <= c.env[2] < c.env[3] <= mo.settle_from * mo.period, c
    # no jumps at robotd's 50 Hz: per-tick change bounded by the fastest component's slope
    dt = 0.02
    phi50 = torch.arange(0.0, 1.0, dt / mo.period, dtype=torch.float64)
    r50 = reference(mo, phi50).numpy()
    step = np.abs(np.diff(r50, axis=0)).max(axis=0)
    for ch, i in CH.items():
        b = bound(mo, ch)
        fmax = max([c.freq for c in mo.comps if c.ch == ch] + [0.0])
        assert step[i] <= b * (2 * math.pi * max(fmax, 1.0) + 2 * math.pi) * dt + 1e-9, (ch, step[i])


def test_stand_height_matches_model():
    m = get_ground_pick_spec().compile()
    d = mujoco.MjData(m)
    d.qpos[3] = 1.0
    for jn in SERVO_JOINTS:
        d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)]] = _home(jn)
    mujoco.mj_kinematics(m, d)
    feet = [d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, s)][2] for s in ("left_foot", "right_foot")]
    assert abs((d.qpos[2] - np.mean(feet)) - STAND_HEIGHT - 0.002) < 1e-3   # FK minus the measured 2 mm sag


@pytest.mark.parametrize("name", list(MOTIONS))
def test_env_cfg_contract(name):
    mo = MOTIONS[name]
    cfg = make_microduck_expressive_env_cfg(name)
    assert cfg.episode_length_s == mo.period
    tw = cfg.commands["twist"]
    assert tw.period == mo.period and tw.randomize_phase is False
    assert cfg.events["foot_friction"].params["ranges"] == (1.0, 2.0)
    assert cfg.sim.mujoco.cone == "elliptic" and cfg.sim.mujoco.impratio == 10.0
    # actor obs unchanged: head/body command padding present, no reference leak
    actor = cfg.observations["actor"].terms
    assert "head_command" in actor and "body_command" in actor and "ex_reference" not in actor
    # reward signs: tracking > 0, costs (>= 0 functions) < 0, self-negating overspeed > 0
    for k in ("track_roll", "track_pitch", "track_yaw", "track_height", "track_head", "settle_legs"):
        assert cfg.rewards[k].weight > 0, k
    for k in ("feet_drift", "feet_lift", "foot_slip", "action_overrun"):
        assert cfg.rewards[k].weight < 0, k
    assert cfg.rewards["joint_overspeed"].weight > 0
    for k in ("mouth_ground_proximity", "ground_pick_return_pose_legs", "mouth_payload_force"):
        assert k not in cfg.rewards
    assert "sample_mouth_payload" not in cfg.events
    assert len(cfg.actions["joint_pos"].clip) == 14

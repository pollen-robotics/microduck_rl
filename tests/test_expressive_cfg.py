"""Expressive motions: references stay inside joint / trunk bounds, are smooth, start and end at the stand pose;
the env cfgs keep the contract (phase command, episode = T, grippy feet, reward signs)."""

import math

import mujoco
import numpy as np
import pytest
import torch

import mjlab_microduck.tasks  # noqa: F401  (registry first)
from mjlab_microduck.robot.microduck_constants import HOME_FRAME, get_ground_pick_spec
from mjlab_microduck.tasks.expressive_motions import (
    ALL_MOTIONS, CH, HEAD_JOINTS, MOTIONS, MOTIONS_R2, R2_END_PHI, STAND_HEIGHT, bound, osc_channels, reference,
    reference_rate,
)
from mjlab_microduck.tasks.microduck_expressive_env_cfg import (
    JOINT_LIMIT_MARGIN,
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


@pytest.mark.parametrize("name", list(ALL_MOTIONS))
def test_reference_bounds(name):
    mo = ALL_MOTIONS[name]
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


@pytest.mark.parametrize("name", list(ALL_MOTIONS))
def test_reference_starts_and_ends_at_stand_and_is_smooth(name):
    mo = ALL_MOTIONS[name]
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


@pytest.mark.parametrize("name", list(ALL_MOTIONS))
def test_env_cfg_contract(name):
    mo = ALL_MOTIONS[name]
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


# ── Round 2 ──────────────────────────────────────────────────────────────────
R2 = [f"{k}-R2" for k in MOTIONS_R2]


@pytest.mark.parametrize("name", R2)
def test_r2_ends_still_at_stand_before_handback(name):
    """Round 1: the stand net stepped at the hand-back. Every r2 envelope closes (value and velocity 0) by φ 0.87."""
    mo = ALL_MOTIONS[name]
    assert 0.85 <= mo.settle_from <= 0.9
    for c in mo.comps:
        assert c.env[3] <= R2_END_PHI * mo.period + 1e-9, c
    phi = torch.linspace(R2_END_PHI, 0.999, 200, dtype=torch.float64)
    assert np.abs(reference(mo, phi).numpy()).max() < 1e-9
    assert np.abs(reference_rate(mo, phi).numpy()).max() < 1e-6


def _osc(mo, ch):
    return [c for c in mo.comps if c.ch == ch and c.freq > 0]


def test_r2_feasible_oscillations():
    s, g, e, h = (MOTIONS_R2[k] for k in ("Sassy", "Giggle", "Excited", "Hmph"))
    assert {c.freq for c in _osc(s, "yaw") + _osc(s, "roll")} == {2.0}
    assert osc_channels(s) == ("roll", "yaw")
    assert [(c.amp, c.freq) for c in _osc(g, "roll")] == [(0.08, 4.0)]
    phi = torch.linspace(0.0, 1.0, 4001, dtype=torch.float64)
    dz = reference(e, phi).numpy()[:, CH["dz"]]
    assert abs(dz.min() + 0.014) < 2e-4 and dz.max() < 1e-9 and {c.freq for c in _osc(e, "dz")} == {2.5}
    yaw = reference(h, phi).numpy()[:, CH["yaw"]]
    assert abs(yaw.min() - math.radians(-18.0)) < 1e-6
    # planted-feet trunk yaw goes through the hip yaws: stay clear of the joint-limit proximity band
    lim = expressive_joint_limits()
    assert -yaw.min() < -lim["left_hip_yaw"][0] - 2 * JOINT_LIMIT_MARGIN


def test_home_clear_of_joint_limit_band():
    for jn, (lo, hi) in expressive_joint_limits().items():
        assert min(_home(jn) - lo, hi - _home(jn)) > JOINT_LIMIT_MARGIN, jn


@pytest.mark.parametrize("name", R2)
def test_r2_env_terms(name):
    mo = ALL_MOTIONS[name]
    cfg = make_microduck_expressive_env_cfg(name)
    r = cfg.rewards
    names = {"roll": "track_roll", "pitch": "track_pitch", "yaw": "track_yaw", "dz": "track_height"}
    for ch in osc_channels(mo):
        assert r[f"{names[ch]}_rate"].weight > 0 and r[f"{names[ch]}_rate"].params["std"] > 0
    if name.startswith("Excited"):
        assert "track_height_rate" in r
    assert r["track_head_wide"].weight > 0 and r["track_head_wide"].params["std"] >= 0.3
    assert r["track_head"].params["std"] < 0.2                       # the narrow kernel stays
    assert r["neck_dev"].weight < 0 and r["neck_dev"].params["free"] == 0.3
    assert r["joint_limit_proximity"].weight > 0 and r["joint_limit_proximity"].params["margin"] == JOINT_LIMIT_MARGIN
    assert r["settle_still"].weight < 0
    for k in ("track_head_wide", "neck_dev", "joint_limit_proximity", "settle_still"):
        assert k not in make_microduck_expressive_env_cfg(name[:-3]).rewards     # round 1 untouched


def test_r2_tasks_registered_from_scratch():
    from mjlab.tasks.registry import _REGISTRY
    from mjlab_microduck.tasks import ExpressiveFromPickRunner
    for k in MOTIONS_R2:
        t = _REGISTRY[f"Mjlab-Expressive-{k}-R2-MicroDuck"]
        assert t.runner_cls is not None and not issubclass(t.runner_cls, ExpressiveFromPickRunner), k
        assert t.rl_cfg.experiment_name == f"expressive_{k.lower()}_r2"
        assert t.rl_cfg.max_iterations == 3000

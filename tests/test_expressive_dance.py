"""Dodge dance (2026-10-29): the data-driven planted-feet reference (motion_data/tremendous_planted.csv) is sane,
starts and ends at the stand, stays within plausible speeds, and the Tremendous env cfgs keep the round 2 contract."""

import math

import numpy as np
import pytest
import torch

import mjlab_microduck.tasks  # noqa: F401  (registry first)
from mjlab_microduck.robot.microduck_constants import HOME_FRAME
from mjlab_microduck.tasks.expressive_motions import (
    CH, DATA_MOTIONS, HEAD_JOINTS, TRUNK_CHANNELS, DataMotion, get_motion, osc_channels, reference, reference_rate,
)
from mjlab_microduck.tasks.microduck_expressive_env_cfg import (
    DATA_MAX_ITERATIONS, expressive_joint_limits, make_microduck_expressive_env_cfg,
)

NAMES = list(DATA_MOTIONS)
V3 = [k for k in NAMES if DATA_MOTIONS[k].v3]
# Per-motion reference bounds: |roll|, |pitch|, |yaw| (rad), dz range (m), peak angular-channel rate (rad/s).
# v3 (2026-10-30): the real hip-roll range + trunk twists (LEDGER "Tremendous v3"); reference rates up to the design
# speed x 1.10 (planted_v3.SPEED_TOL), below the task's joint_overspeed threshold.
BOUNDS = {"Tremendous": (0.30, 0.65, 0.0, (-0.03, 0.003), 5.0), "Tremendous-70": (0.30, 0.65, 0.0, (-0.03, 0.003), 5.0),
          "Tremendous-V3": (0.55, 0.70, 0.30, (-0.025, 0.003), 7.7),
          "Tremendous-V3-Bold": (0.55, 0.75, 0.30, (-0.028, 0.003), 8.8),
          "Tremendous-V4": (0.55, 0.70, 0.30, (-0.025, 0.003), 7.7)}


def _home(jn):
    import re
    for pat, v in HOME_FRAME.joint_pos.items():
        if re.fullmatch(pat, jn):
            return v
    raise KeyError(jn)


def _phi(mo, n=None):
    n = n or int(mo.period * 200) + 1
    return torch.linspace(0.0, 1.0, n, dtype=torch.float64)


@pytest.mark.parametrize("name", NAMES)
def test_registered_and_looked_up(name):
    mo = get_motion(name)
    assert isinstance(mo, DataMotion) and mo.period == 18.25          # the music excerpt (phase skill T)
    assert (mo.trunk_scale == 0.7) == name.endswith("-70")
    assert set(BOUNDS) == set(NAMES)


@pytest.mark.parametrize("name", NAMES)
def test_starts_and_ends_still_at_stand(name):
    mo = get_motion(name)
    phi = _phi(mo)
    r, d = reference(mo, phi).numpy(), reference_rate(mo, phi).numpy()
    assert np.abs(r[0]).max() < 1e-6 and np.abs(d[0]).max() < 1e-6
    late = phi.numpy() >= mo.settle_from - 0.005
    assert late.sum() > 10
    assert np.abs(r[late]).max() < 1e-6 and np.abs(d[late]).max() < 1e-6
    assert 0.95 <= mo.settle_from < 0.99


@pytest.mark.parametrize("name", NAMES)
def test_bounds_and_speeds(name):
    mo = get_motion(name)
    roll_max, pitch_max, yaw_max, (dz_lo, dz_hi), vmax = BOUNDS[name]
    r, d = reference(mo, _phi(mo)).numpy(), reference_rate(mo, _phi(mo)).numpy()
    assert np.abs(r[:, CH["yaw"]]).max() <= yaw_max + 1e-9                # 1029 / r2: no trunk yaw in the dance
    assert np.abs(r[:, CH["roll"]]).max() <= roll_max                     # hip roll range on flat planted feet
    assert np.abs(r[:, CH["pitch"]]).max() <= pitch_max
    assert dz_lo <= r[:, CH["dz"]].min() and r[:, CH["dz"]].max() <= dz_hi
    lim = expressive_joint_limits(mo.hip_roll_range)
    for jn in HEAD_JOINTS:
        q = _home(jn) + r[:, CH[jn]]
        assert q.min() >= lim[jn][0] + 0.05 and q.max() <= lim[jn][1] - 0.05, jn
    # trunk yaw on planted feet goes through both hip yaws: inside their ranges with margin
    assert np.abs(r[:, CH["yaw"]]).max() <= min(-lim["left_hip_yaw"][0], lim["right_hip_yaw"][1]) - 0.05
    for ch in ("roll", "pitch", "yaw") + HEAD_JOINTS:
        assert np.abs(d[:, CH[ch]]).max() <= vmax, (ch, np.abs(d[:, CH[ch]]).max())
        assert np.abs(d[:, CH[ch]]).max() <= mo.overspeed * 1.1, ch
    assert np.abs(d[:, CH["dz"]]).max() <= 0.3


@pytest.mark.parametrize("name", ["Tremendous"] + V3)
def test_rates_are_the_derivative_of_the_values(name):
    mo = get_motion(name)
    phi = _phi(mo, 20001)
    r, d = reference(mo, phi).numpy(), reference_rate(mo, phi).numpy()
    num = np.gradient(r, phi.numpy() * mo.period, axis=0)
    for ch in ("roll", "pitch", "yaw", "dz", "head_yaw", "head_pitch", "head_roll"):
        if np.abs(d[:, CH[ch]]).max() < 1e-9:
            continue
        i = CH[ch]
        scale = np.abs(d[:, i]).max()
        # the table is 50 Hz, linearly interpolated: compare smoothed, allow 15 % of the peak rate
        k = np.ones(25) / 25
        assert np.abs(np.convolve(num[:, i] - d[:, i], k, "same")).max() <= 0.15 * scale + 1e-9, ch


def test_soft_variant_scales_trunk_only():
    a, b = get_motion("Tremendous"), get_motion("Tremendous-70")
    phi = _phi(a)
    ra, rb = reference(a, phi).numpy(), reference(b, phi).numpy()
    for ch in TRUNK_CHANNELS:
        assert np.allclose(rb[:, CH[ch]], 0.7 * ra[:, CH[ch]])
    for ch in HEAD_JOINTS:
        assert np.allclose(rb[:, CH[ch]], ra[:, CH[ch]])


@pytest.mark.parametrize("name", NAMES)
def test_env_cfg_dance_terms(name):
    mo = get_motion(name)
    cfg = make_microduck_expressive_env_cfg(name)
    assert cfg.episode_length_s == mo.period
    tw = cfg.commands["twist"]
    assert tw.period == mo.period and tw.randomize_phase is False
    assert cfg.events["foot_friction"].params["ranges"] == (1.0, 2.0)
    r = cfg.rewards
    assert osc_channels(mo) == (("roll", "pitch", "yaw", "dz") if mo.v3 else ("roll", "pitch", "dz"))
    rate_terms = ("track_roll_rate", "track_pitch_rate", "track_height_rate") + (("track_yaw_rate",) if mo.v3 else ())
    for k in rate_terms + ("track_trunk_wide", "track_head_wide"):
        assert r[k].weight > 0 and r[k].params["std"] > 0, k
    for k in ("feet_drift", "feet_lift", "foot_slip", "neck_dev", "settle_still", "action_overrun"):
        assert r[k].weight < 0, k
    assert r["joint_limit_proximity"].weight > 0
    assert r["settle_legs"].params["ramp"] == mo.settle_ramp == r["settle_still"].params["ramp"]
    assert "ex_reference" not in cfg.observations["actor"].terms
    # r2 (2026-10-30): linear deviation costs on the head joints and the trunk roll (the kernels had no gradient)
    dev = {"neck_dev": (("neck_pitch",), 0.3), "head_dev": (("head_pitch", "head_yaw"), 0.25),
           "head_roll_dev": (("head_roll",), 0.15), "trunk_roll_dev": (("roll",), 0.05)}
    if mo.v3:
        dev["trunk_yaw_dev"] = (("yaw",), 0.05)
    else:
        for k in ("trunk_yaw_dev", "track_height_wide", "settle_legs_dev"):
            assert k not in r, k                                         # 1029 / r2 tasks keep their reward set
        assert r["track_trunk_wide"].params["channels"] == ("roll", "pitch")
        assert r["joint_overspeed"].params["threshold"] == 5.0
    for k, (chs, free) in dev.items():
        assert r[k].func is r["neck_dev"].func and r[k].weight < 0, k
        assert tuple(r[k].params["channels"]) == chs and r[k].params["free"] == free, k
        assert r[k].params["motion"] == name, k
    assert r["trunk_roll_dev"].weight <= r["head_dev"].weight          # roll errors are smaller: at least as steep


def test_dev_cost_is_linear_beyond_the_free_band():
    """ex_head_dev_cost reads any REF_CHANNELS channel (trunk roll included): 0 inside the band, then |err| - free."""
    from mjlab_microduck.tasks import mdp as M
    state, ref = torch.zeros(3, len(CH)), torch.zeros(3, len(CH))
    state[:, CH["roll"]] = torch.tensor([0.03, 0.15, -0.25])
    state[:, CH["head_pitch"]] = torch.tensor([0.0, 1.0, -2.0])
    ref[:, CH["head_yaw"]] = torch.tensor([0.1, 0.0, 2.5])          # state 0: yaw errors 0.1 / 0 / 2.5
    orig = M._ex_state, M._ex_ref
    M._ex_state, M._ex_ref = (lambda env: state), (lambda env, motion, command_name: ref)
    try:
        roll = M.ex_head_dev_cost(None, "Tremendous", ("roll",), 0.05)
        head = M.ex_head_dev_cost(None, "Tremendous", ("head_pitch", "head_yaw"), 0.25)
    finally:
        M._ex_state, M._ex_ref = orig
    assert torch.allclose(roll, torch.tensor([0.0, 0.10, 0.20]))
    assert torch.allclose(head, torch.tensor([0.0, 0.75, 1.75 + 2.25]))


def test_tasks_registered_from_scratch():
    from mjlab.tasks.registry import _REGISTRY
    from mjlab_microduck.tasks import ExpressiveFromPickRunner
    for k in NAMES:
        t = _REGISTRY[f"Mjlab-Expressive-{k}-MicroDuck"]
        assert t.runner_cls is not None and not issubclass(t.runner_cls, ExpressiveFromPickRunner), k
        assert t.rl_cfg.max_iterations == DATA_MAX_ITERATIONS
        assert t.rl_cfg.experiment_name == f"expressive_{k.lower().replace('-', '_')}"


# ── v3 (2026-10-30, LEDGER "Tremendous v3") ─────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("name", V3)
def test_v3_robot_has_the_real_hip_roll_range(name):
    """The v3 tasks train on the ground-pick robot with hip roll +-0.52 (ground_grasp.py); the clip / overrun limits
    follow it; the older tasks keep the export's +-0.384."""
    mo = get_motion(name)
    cfg = make_microduck_expressive_env_cfg(name)
    m = cfg.scene.entities["robot"].spec_fn().compile()
    for jn in ("left_hip_roll", "right_hip_roll"):
        assert tuple(np.round(m.jnt_range[m.joint(jn).id], 4)) == (-0.52, 0.52)
    lim = expressive_joint_limits(mo.hip_roll_range)
    assert lim["left_hip_roll"] == (-0.52, 0.52) and expressive_joint_limits()["left_hip_roll"][1] < 0.39
    assert cfg.actions["joint_pos"].clip["left_hip_roll"] == pytest.approx((-0.62, 0.62))
    assert cfg.rewards["action_overrun"].params["limits"][1] == [-0.52, 0.52]
    old = make_microduck_expressive_env_cfg("Tremendous-70")
    m0 = old.scene.entities["robot"].spec_fn().compile()
    assert m0.jnt_range[m0.joint("left_hip_roll").id][1] < 0.39


@pytest.mark.parametrize("name", V3)
def test_v3_terms(name):
    mo = get_motion(name)
    r = make_microduck_expressive_env_cfg(name).rewards
    assert r["track_trunk_wide"].params["channels"] == ("roll", "pitch", "yaw")
    assert r["track_height_wide"].weight > 0 and r["track_height_wide"].params["std"] >= 2.5 * r["track_height"].params["std"]
    sd = r["settle_legs_dev"]
    assert sd.weight < 0 and sd.params["settle_from"] == mo.settle_from == r["settle_legs"].params["settle_from"]
    assert sd.params["ramp"] == mo.settle_ramp and sd.params["free"] > 0
    assert r["joint_overspeed"].params["threshold"] == mo.overspeed >= 7.0


@pytest.mark.parametrize("name", V3)
def test_v3_reference_is_bigger_and_ends_early(name):
    """Bigger than the 1029 / r2 reference on every axis Rémi asked for, back at the stand on beat 32 (17.45 s), and
    the settle window (still stand) is >= 0.6 s before φ = 1."""
    old, mo = get_motion("Tremendous"), get_motion(name)
    phi = _phi(mo)
    r0, r = reference(old, phi).numpy(), reference(mo, phi).numpy()
    amp = lambda a, ch: np.abs(a[:, CH[ch]]).max()
    assert amp(r, "roll") >= 1.5 * amp(r0, "roll")                     # side tilt (r2: 14.3 deg)
    assert amp(r, "yaw") >= math.radians(10.0)                            # trunk twists (r2: none)
    assert r[:, CH["dz"]].min() <= 5 * r0[:, CH["dz"]].min()             # squash (r2: -2.4 mm)
    assert amp(r, "head_yaw") >= 1.25 * amp(r0, "head_yaw")
    t = phi.numpy() * mo.period
    beat = 60.0 / 110.0
    moving = np.abs(reference_rate(mo, phi).numpy()).max(axis=1) > 1e-6
    assert t[moving].max() <= 32 * beat + 1e-3                            # nothing moves after beat 32
    assert (1.0 - mo.settle_from) * mo.period >= 0.6
    # the bounce bottoms ON the beat: the dz minimum in each slip / sway beat window sits within 30 ms of a beat
    dz = r[:, CH["dz"]]
    for b in (2, 3, 6, 7, 17, 25):
        w = (t > (b - 0.5) * beat) & (t < (b + 0.5) * beat)
        assert abs(t[w][np.argmin(dz[w])] - b * beat) < 0.03, b


def test_settle_legs_dev_is_linear_and_gated():
    """ex_settle_legs_dev: 0 before settle_from, then sum(|leg - HOME| - free)+ over the 10 leg servos."""
    from mjlab_microduck.tasks import mdp as M
    q = torch.zeros(2, 14)
    q[:, 2] = torch.tensor([0.23, 0.01])          # left hip pitch 0.23 rad off HOME / inside the band
    q[:, 13] = torch.tensor([-0.13, 0.0])         # right ankle
    q[:, 6] = 1.0                                  # head joints are not legs
    orig = M._gp_phase, M._servo_joint_pos, M._servo_default_joint_pos
    M._servo_joint_pos = lambda env, robot: q
    M._servo_default_joint_pos = lambda env, robot: torch.zeros(2, 14)

    class _E:
        scene = {"robot": None}
    try:
        M._gp_phase = lambda env, cn: torch.tensor([0.99, 0.99])
        on = M.ex_settle_legs_dev(_E(), settle_from=0.96, ramp=0.01, free=0.03)
        M._gp_phase = lambda env, cn: torch.tensor([0.5, 0.5])
        off = M.ex_settle_legs_dev(_E(), settle_from=0.96, ramp=0.01, free=0.03)
    finally:
        M._gp_phase, M._servo_joint_pos, M._servo_default_joint_pos = orig
    assert torch.allclose(on, torch.tensor([0.20 + 0.10, 0.0])) and torch.allclose(off, torch.zeros(2))

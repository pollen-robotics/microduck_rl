"""Dodge dance (2026-10-29): the data-driven planted-feet reference (motion_data/tremendous_planted.csv) is sane,
starts and ends at the stand, stays within plausible speeds, and the Tremendous env cfgs keep the round 2 contract."""

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
VMAX = 5.0                      # rad/s on any angular reference channel (servo-plausible)


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
    r, d = reference(mo, _phi(mo)).numpy(), reference_rate(mo, _phi(mo)).numpy()
    assert np.abs(r[:, CH["yaw"]]).max() < 1e-9                        # no trunk yaw in the dance
    assert np.abs(r[:, CH["roll"]]).max() <= 0.30                      # hip roll range on flat planted feet
    assert np.abs(r[:, CH["pitch"]]).max() <= 0.65
    assert -0.03 <= r[:, CH["dz"]].min() and r[:, CH["dz"]].max() <= 0.003
    lim = expressive_joint_limits()
    for jn in HEAD_JOINTS:
        q = _home(jn) + r[:, CH[jn]]
        assert q.min() >= lim[jn][0] + 0.05 and q.max() <= lim[jn][1] - 0.05, jn
    for ch in ("roll", "pitch") + HEAD_JOINTS:
        assert np.abs(d[:, CH[ch]]).max() <= VMAX, (ch, np.abs(d[:, CH[ch]]).max())
    assert np.abs(d[:, CH["dz"]]).max() <= 0.3


def test_rates_are_the_derivative_of_the_values():
    mo = get_motion("Tremendous")
    phi = _phi(mo, 20001)
    r, d = reference(mo, phi).numpy(), reference_rate(mo, phi).numpy()
    num = np.gradient(r, phi.numpy() * mo.period, axis=0)
    for ch in ("roll", "pitch", "dz", "head_yaw", "head_pitch"):
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
    assert osc_channels(mo) == ("roll", "pitch", "dz")
    for k in ("track_roll_rate", "track_pitch_rate", "track_height_rate", "track_trunk_wide", "track_head_wide"):
        assert r[k].weight > 0 and r[k].params["std"] > 0, k
    for k in ("feet_drift", "feet_lift", "foot_slip", "neck_dev", "settle_still", "action_overrun"):
        assert r[k].weight < 0, k
    assert r["joint_limit_proximity"].weight > 0
    assert r["settle_legs"].params["ramp"] == mo.settle_ramp == r["settle_still"].params["ramp"]
    assert "ex_reference" not in cfg.observations["actor"].terms


def test_tasks_registered_from_scratch():
    from mjlab.tasks.registry import _REGISTRY
    from mjlab_microduck.tasks import ExpressiveFromPickRunner
    for k in NAMES:
        t = _REGISTRY[f"Mjlab-Expressive-{k}-MicroDuck"]
        assert t.runner_cls is not None and not issubclass(t.runner_cls, ExpressiveFromPickRunner), k
        assert t.rl_cfg.max_iterations == DATA_MAX_ITERATIONS
        assert t.rl_cfg.experiment_name == f"expressive_{k.lower().replace('-', '_')}"

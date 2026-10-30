"""STEPPING dance (2026-10-30, branch dance-steps): the stepping Tremendous reference (motion_data/tremendous_steps.csv)
starts and ends still on the stand footprint, its contact schedule is a clean set of single-foot steps (lift, clear the
floor, land, then the foot stays put while loaded), speeds stay servo-plausible, and the TremendousSteps env cfgs swap
the planted-feet costs for the step terms without touching the planted tasks."""

import numpy as np
import pytest
import torch

import mjlab_microduck.tasks  # noqa: F401  (registry first)
from mjlab_microduck.robot.microduck_constants import HOME_FRAME
from mjlab_microduck.tasks.expressive_motions import (
    CH, HEAD_JOINTS, SC, STEP_MOTIONS, StepMotion, get_motion, osc_channels, reference, reference_rate, step_reference,
)
from mjlab_microduck.tasks.microduck_expressive_env_cfg import (
    DATA_MAX_ITERATIONS, STEP_PROFILES, STEP_TERMS, expressive_joint_limits, make_microduck_expressive_env_cfg,
)

NAME = "Tremendous-Steps"
TASKS = {"Mjlab-Expressive-TremendousSteps-MicroDuck": "base", "Mjlab-Expressive-TremendousSteps-Feet-MicroDuck": "feet"}
VMAX = 5.0


def _home(jn):
    import re
    for pat, v in HOME_FRAME.joint_pos.items():
        if re.fullmatch(pat, jn):
            return v
    raise KeyError(jn)


def _phi(mo, n=None):
    n = n or int(mo.period * 200) + 1
    return torch.linspace(0.0, 1.0, n, dtype=torch.float64)


def test_looked_up_and_registered():
    from mjlab.tasks.registry import _REGISTRY
    from mjlab_microduck.tasks import ExpressiveFromPickRunner
    mo = get_motion(NAME)
    assert isinstance(mo, StepMotion) and mo.period == 18.25 and NAME in STEP_MOTIONS
    names = set()
    for tid in TASKS:
        t = _REGISTRY[tid]
        assert t.runner_cls is not None and not issubclass(t.runner_cls, ExpressiveFromPickRunner)   # from scratch
        assert t.rl_cfg.max_iterations == DATA_MAX_ITERATIONS
        names.add(t.rl_cfg.experiment_name)
    assert len(names) == 2


def test_starts_and_ends_still_on_the_stand_footprint():
    mo = get_motion(NAME)
    phi = _phi(mo)
    r, d, s = reference(mo, phi).numpy(), reference_rate(mo, phi).numpy(), step_reference(mo, phi).numpy()
    assert np.abs(r[0]).max() < 1e-3 and np.abs(d[0]).max() < 1e-2
    late = phi.numpy() >= mo.settle_from - 0.005
    assert late.sum() > 10
    assert np.abs(r[late]).max() < 1e-3 and np.abs(d[late]).max() < 1e-2
    for c in ("lf_x", "lf_y", "rf_x", "rf_y", "lf_z", "rf_z"):
        assert np.abs(s[0, SC[c]]) < 1e-6 and np.abs(s[late, SC[c]]).max() < 1e-3, c       # back on the footprint
    assert s[late, SC["l_contact"]].min() == 1.0 and s[late, SC["r_contact"]].min() == 1.0
    assert 0.95 <= mo.settle_from < 0.99


def test_bounds_and_speeds():
    mo = get_motion(NAME)
    r, d = reference(mo, _phi(mo)).numpy(), reference_rate(mo, _phi(mo)).numpy()
    assert np.abs(r[:, CH["roll"]]).max() <= 0.35
    assert np.abs(r[:, CH["pitch"]]).max() <= 0.65
    assert np.abs(r[:, CH["yaw"]]).max() <= 0.35                            # the twist
    assert -0.03 <= r[:, CH["dz"]].min() and r[:, CH["dz"]].max() <= 0.003
    lim = expressive_joint_limits()
    for jn in HEAD_JOINTS:
        q = _home(jn) + r[:, CH[jn]]
        assert q.min() >= lim[jn][0] + 0.05 and q.max() <= lim[jn][1] - 0.05, jn
    for ch in ("roll", "pitch", "yaw") + HEAD_JOINTS:
        assert np.abs(d[:, CH[ch]]).max() <= VMAX, (ch, np.abs(d[:, CH[ch]]).max())


def test_steps_are_clean_single_foot_steps():
    mo = get_motion(NAME)
    phi = _phi(mo, 18251)                                                    # 1 ms
    t = phi.numpy() * mo.period
    s = step_reference(mo, phi).numpy()
    n_steps = 0
    for side in ("l", "r"):
        c = s[:, SC[f"{side}_contact"]]
        xy = s[:, [SC[f"{side}f_x"], SC[f"{side}f_y"]]]
        z = s[:, SC[f"{side}f_z"]]
        down = c >= 0.999
        # a loaded foot stays put and on the floor (between table samples the interpolation blends 20 ms)
        dxy = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        assert dxy[down[1:] & down[:-1]].max() < 2e-5, side
        assert z[down].max() < 5e-4, side
        swing = c < 0.5
        edges = np.flatnonzero(np.diff(swing.astype(int)) == 1)
        ends = np.flatnonzero(np.diff(swing.astype(int)) == -1)
        assert len(edges) == len(ends)
        for a, b in zip(edges, ends):
            n_steps += 1
            assert 0.25 <= t[b] - t[a] <= 0.45, (side, t[a], t[b])            # swing time
            assert 0.015 <= z[a:b + 1].max() <= 0.025, (side, z[a:b + 1].max())   # real clearance
            assert np.linalg.norm(xy[b + 1] - xy[a]) >= 0.03, side            # it goes somewhere
    assert n_steps == 4
    both_up = (s[:, SC["l_contact"]] < 0.5) & (s[:, SC["r_contact"]] < 0.5)
    assert not both_up.any()                                                 # never a jump


@pytest.mark.parametrize("tid", list(TASKS))
def test_env_cfg_step_terms(tid):
    prof = TASKS[tid]
    cfg = make_microduck_expressive_env_cfg(NAME, step_profile=prof)
    mo = get_motion(NAME)
    assert cfg.episode_length_s == mo.period and cfg.commands["twist"].period == mo.period
    assert cfg.events["foot_friction"].params["ranges"] == (1.0, 2.0)
    r = cfg.rewards
    for k in ("feet_drift", "feet_lift", "foot_slip", "feet_grounded"):
        assert k not in r, k                                                  # the planted-feet costs are gone
    for k, (w, params) in STEP_TERMS.items():
        assert r[k].weight == w * STEP_PROFILES[prof] and r[k].params["motion"] == NAME, k
        for p, v in params.items():
            assert r[k].params[p] == v, (k, p)
    assert r["step_slip"].weight < 0 and r["step_unscheduled_lift"].weight < 0 and r["step_foot_dev"].weight < 0
    assert r["feet_flat"].params["sensor_name"] == "feet_ground_contact"     # the swing foot may tilt
    assert osc_channels(mo) == ("roll", "pitch", "yaw", "dz") and r["track_yaw_rate"].weight > 0
    for k in ("track_trunk_wide", "head_dev", "head_roll_dev", "trunk_roll_dev", "neck_dev", "settle_still"):
        assert k in r, k
    crit = cfg.observations["critic"].terms
    assert "ex_step_ref" in crit and "ex_step_err" in crit
    assert "ex_step_ref" not in cfg.observations["actor"].terms


def test_actor_obs_and_planted_tasks_unchanged():
    planted = make_microduck_expressive_env_cfg("Tremendous-70")
    steps = make_microduck_expressive_env_cfg(NAME)
    assert list(planted.observations["actor"].terms) == list(steps.observations["actor"].terms)
    for k in ("feet_drift", "feet_lift", "foot_slip", "feet_grounded"):
        assert k in planted.rewards
    assert not any(k.startswith("step_") for k in planted.rewards)
    assert "sensor_name" not in planted.rewards["feet_flat"].params or planted.rewards["feet_flat"].params["sensor_name"] is None


def test_step_reward_functions_on_a_fake_state():
    """foot tracking / contact / clearance / unscheduled lift against hand-made measurements."""
    from mjlab_microduck.tasks import mdp as M
    mo = get_motion(NAME)
    phi = torch.tensor([0.0, 4.6 / 18.25, 5.0 / 18.25])                     # standing, L mid-swing, L landed back
    ref = step_reference(mo, phi.double()).float()
    disp = torch.stack((ref[:, [SC["lf_x"], SC["lf_y"]]], ref[:, [SC["rf_x"], SC["rf_y"]]]), dim=1).clone()
    clear = ref[:, [SC["lf_z"], SC["rf_z"]]].clone()
    found = ref[:, [SC["l_contact"], SC["r_contact"]]].round()
    orig = (M._ex_feet_step_state, M._ex_step_ref, M._ex_contact_found)
    M._ex_feet_step_state = lambda env: (disp, clear)
    M._ex_step_ref = lambda env, motion, command_name: ref
    M._ex_contact_found = lambda env, sensor_name: found
    try:
        assert torch.allclose(M.ex_step_foot_track(None, NAME), torch.ones(3))
        assert torch.allclose(M.ex_step_foot_dev(None, NAME), torch.zeros(3))
        assert torch.allclose(M.ex_step_contact(None, NAME), torch.ones(3), atol=1e-3)
        assert M.ex_step_clearance(None, NAME)[1] > 0.99 and M.ex_step_clearance(None, NAME)[0] == 0
        assert torch.allclose(M.ex_step_unscheduled_lift(None, NAME), torch.zeros(3))
        disp[1, 0] = 0.0; clear[1, 0] = 0.0                                  # the left foot refuses to step
        assert M.ex_step_foot_track(None, NAME)[1] < 0.75 and M.ex_step_clearance(None, NAME)[1] < 0.01
        clear[0, 1] = 0.02                                                   # a foot lifted while scheduled down
        assert M.ex_step_unscheduled_lift(None, NAME)[0] > 0.016
    finally:
        M._ex_feet_step_state, M._ex_step_ref, M._ex_contact_found = orig

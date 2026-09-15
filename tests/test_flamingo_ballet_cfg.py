"""Flamingo ballet: cfg invariants + pose-blend logic (CPU, fake env / bare command)."""
from types import SimpleNamespace

import numpy as np
import torch

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks import microduck_flamingo_ballet_env_cfg as fb
from mjlab_microduck.tasks import microduck_flamingo_cycle_env_cfg as fc
from mjlab_microduck.tasks.microduck_flamingo_ballet_env_cfg import (
    make_microduck_flamingo_ballet_env_cfg,
    MicroduckFlamingoBalletRlCfg,
)

RANGES = {
    1: (-0.384, 0.384), 10: (-0.384, 0.384),
    0: (-0.436, 0.524), 9: (-0.524, 0.436),
    2: (-1.571, 1.571), 3: (-1.571, 1.571), 4: (-1.571, 1.571),
    11: (-1.571, 1.571), 12: (-1.571, 1.571), 13: (-1.571, 1.571),
    5: (-1.571, 1.047), 6: (-1.571, 1.571), 7: (-2.967, 2.967), 8: (-0.436, 0.436),
}


def test_poses_have_margins_and_share_the_stance_leg():
    for side_poses in (fb.POSES_RIGHT, fb.POSES_LEFT):
        assert len(side_poses) == 3
        for pose in side_poses:
            assert len(pose) == 14
            for i, (lo, hi) in RANGES.items():
                assert min(pose[i] - lo, hi - pose[i]) >= 0.05 - 1e-9, (pose, i)
    # right stance: joints 9-13 (stance leg) identical across the three poses
    for pose in fb.POSES_RIGHT:
        assert pose[9:14] == fb.PASSE_POSE_RIGHT[9:14]
    for pose in fb.POSES_LEFT:
        assert pose[0:5] == fb.POSES_LEFT[1][0:5]
    assert fb.POSES_RIGHT[1] == list(fc.FLAMINGO_POSE_RIGHT)
    assert fb.POSES_LEFT[1] == list(fc.FLAMINGO_POSE_LEFT)
    # arabesque: swing leg back (hip pitch negative on the left leg), head up
    assert fb.ARABESQUE_POSE_RIGHT[2] < 0 and fb.ARABESQUE_POSE_RIGHT[6] < 0
    assert fb.DEVELOPPE_POSE_RIGHT[2] > 0 and fb.DEVELOPPE_POSE_RIGHT[3] == 0.0


def test_cfg_wiring():
    cfg = make_microduck_flamingo_ballet_env_cfg()
    assert isinstance(cfg.commands["twist"], microduck_mdp.FlamingoBalletCommandCfg)
    assert cfg.commands["twist"].pose_probs == fb.POSE_PROBS
    assert cfg.rewards["pose_track"].func is microduck_mdp.flb_pose_track
    assert cfg.rewards["pose_track"].params["poses_left"] == fb.POSES_LEFT
    assert cfg.rewards["swing_foot_clear"].func is microduck_mdp.flb_swing_foot_clear
    assert cfg.rewards["swing_foot_clear"].params["targets"] == fb.CLEAR_TARGETS
    assert "set_flamingo_cycle_state" not in cfg.events
    ev = cfg.events["set_flamingo_ballet_state"]
    assert ev.params["poses_right"] == fb.POSES_RIGHT and ev.params["in_pose_prob"] == fc.IN_POSE_PROB
    assert cfg.curriculum["in_pose_prob"].params["event_name"] == "set_flamingo_ballet_state"
    base = make_microduck_flamingo_cycle_env_cfg_keys()
    assert set(cfg.rewards.keys()) == base
    assert MicroduckFlamingoBalletRlCfg.experiment_name == "flamingo_ballet"


def make_microduck_flamingo_cycle_env_cfg_keys():
    return set(fc.make_microduck_flamingo_cycle_env_cfg().rewards.keys())


def test_obs_layout_unchanged():
    a = make_microduck_flamingo_ballet_env_cfg()
    b = fc.make_microduck_flamingo_cycle_env_cfg()
    for grp in ("actor", "critic"):
        assert list(a.observations[grp].terms.keys()) == list(b.observations[grp].terms.keys())


def test_interp_is_piecewise_linear():
    lo, mid, hi = torch.tensor([[-1.0, 0.0]]), torch.tensor([[0.0, 1.0]]), torch.tensor([[2.0, 0.0]])
    u = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
    out = microduck_mdp._flb_interp(u, lo.expand(5, -1), mid.expand(5, -1), hi.expand(5, -1))
    assert torch.allclose(out, torch.tensor([[-1.0, 0.0], [-0.5, 0.5], [0.0, 1.0], [1.0, 0.5], [2.0, 0.0]]))


def _bare(n):
    term = microduck_mdp.FlamingoBalletCommand.__new__(microduck_mdp.FlamingoBalletCommand)
    term._env = SimpleNamespace(num_envs=n, device="cpu")
    term.vel_command_b = torch.zeros(n, 3)
    term._flamingo_prob = 1.0; term._ramp_s = 1.5; term._zero_side_prob = 0.5
    term._alpha = torch.zeros(n); term._fresh = torch.ones(n, dtype=torch.bool)
    term._pending_side = torch.zeros(n); term._stance_side = torch.ones(n)
    term._pose_ramp_s = 1.5; term._pose_probs = [0.25, 0.5, 0.25]
    term._pose_u = torch.zeros(n); term._pose_id = torch.zeros(n)
    term._pending_pose = torch.full((n,), float("nan"))
    return term


def test_ballet_command_pose_logic():
    torch.manual_seed(0)
    n = 600
    term = _bare(n)
    ids = torch.arange(n)
    pinned = torch.tensor([-1.0, 0.0, 1.0]).repeat(n // 3)
    term.request_pose(ids, pinned)
    assert torch.equal(term.pose_u, pinned) and torch.equal(term.command[:, 2], pinned)
    term._resample_command(ids)                       # flag = 1 for all
    assert torch.equal(term.pose_id, pinned) and torch.equal(term.command[:, 2], pinned)
    assert torch.isnan(term._pending_pose).all()
    # a hold → hold resample redraws the pose (all three ids appear, roughly 1:2:1)
    term._resample_command(ids)
    ids_seen = term.pose_id
    frac0 = (ids_seen == 0).float().mean().item()
    assert set(ids_seen.unique().tolist()) == {-1.0, 0.0, 1.0} and 0.35 < frac0 < 0.65
    # standing: the observed pose slot is 0, the latched pose is kept
    term._flamingo_prob = 0.0
    prev = term.pose_id.clone()
    term._resample_command(ids)
    assert (term.command[:, 2] == 0).all() and torch.equal(term.pose_id, prev)


def test_pose_u_slews():
    term = _bare(2)
    term._pose_id = torch.tensor([1.0, -1.0]); term._pose_u = torch.tensor([0.0, 0.0])
    step = 0.02 / 1.5
    delta = term._pose_id - term._pose_u
    term._pose_u += torch.clamp(delta, -step, step)
    assert abs(term._pose_u[0].item() - step) < 1e-7 and abs(term._pose_u[1].item() + step) < 1e-7


class _Term:
    def __init__(self, cmd, alpha, u):
        self.command = cmd; self.alpha = alpha; self.pose_u = u
        self.stance_side = torch.where(cmd[:, 1] < 0.0, -1.0, 1.0)


def _env(cmd, alpha, u, q, left_z, right_z, found):
    n = cmd.shape[0]
    term = _Term(cmd, alpha, u)
    cm = SimpleNamespace(get_term=lambda name: term, get_command=lambda name: term.command)
    home = torch.tensor([0.0, -0.0873, -0.4579, -0.0049, 0.4530, 0.3491, 0.3491, 0.0, 0.0, 0.0, 0.0873, 0.4579, 0.0049, -0.4530])
    data = SimpleNamespace(
        site_pos_w=torch.stack([torch.tensor([[0.0, 0.04, z] for z in left_z]), torch.tensor([[0.0, -0.04, z] for z in right_z])], dim=1),
        default_joint_pos=home.unsqueeze(0).expand(n, -1).clone(), joint_pos=q, joint_vel=torch.zeros(n, 14),
    )
    robot = SimpleNamespace(data=data, indexing=SimpleNamespace(root_body_id=0))
    sensor = SimpleNamespace(data=SimpleNamespace(found=found))

    class _Scene:
        sensors = {"feet": sensor}
        terrain = SimpleNamespace(env_origins=torch.zeros(n, 3))
        def __getitem__(self, k): return {"robot": robot}[k]
    return SimpleNamespace(scene=_Scene(), command_manager=cm, device="cpu", num_envs=n)


def _site(name):
    return SimpleNamespace(name="robot", site_ids=[0] if name == "left" else [1], site_names=[f"{name}_foot"])


def test_flb_pose_track_hits_each_pose_on_each_side(monkeypatch):
    monkeypatch.setattr(microduck_mdp, "_servo_joint_pos", lambda e, a: a.data.joint_pos)
    monkeypatch.setattr(microduck_mdp, "_servo_default_joint_pos", lambda e, a: a.data.default_joint_pos)
    cmd = torch.tensor([[1, 1, -1], [1, 1, 0], [1, 1, 1], [1, -1, -1], [1, -1, 1], [1, 1, 1]], dtype=torch.float32)
    u = torch.tensor([-1.0, 0.0, 1.0, -1.0, 1.0, 1.0])
    q = torch.tensor([fb.POSES_RIGHT[0], fb.POSES_RIGHT[1], fb.POSES_RIGHT[2], fb.POSES_LEFT[0], fb.POSES_LEFT[2], fb.POSES_RIGHT[0]])
    env = _env(cmd, torch.ones(6), u, q, [0.1] * 6, [0.0] * 6, torch.tensor([[0.0, 1.0]] * 6))
    r = microduck_mdp.flb_pose_track(env, "twist", fb.POSES_RIGHT, fb.POSES_LEFT, std=0.5, mid_attenuation=0.75)
    assert (r[:5] > 0.999).all()
    assert r[5] < 0.9                      # développé commanded, arabesque held → off target


def test_flb_swing_clear_target_follows_the_pose(monkeypatch):
    cmd = torch.tensor([[1, 1, -1], [1, 1, 0], [1, 1, 1]], dtype=torch.float32)
    u = torch.tensor([-1.0, 0.0, 1.0])
    found = torch.tensor([[0.0, 1.0]] * 3)
    env = _env(cmd, torch.ones(3), u, torch.zeros(3, 14), [0.05, 0.10, 0.08], [0.0] * 3, found)
    monkeypatch.setattr(microduck_mdp, "_foot_found", lambda e, s, slot: found[:, slot])
    r = microduck_mdp.flb_swing_foot_clear(env, "twist", _site("left"), _site("right"), "feet", targets=fb.CLEAR_TARGETS, std=0.06)
    assert (r > 0.999).all()
    env2 = _env(cmd, torch.ones(3), u, torch.zeros(3, 14), [0.10, 0.10, 0.10], [0.0] * 3, found)
    r2 = microduck_mdp.flb_swing_foot_clear(env2, "twist", _site("left"), _site("right"), "feet", targets=fb.CLEAR_TARGETS, std=0.06)
    assert r2[1] > 0.999 and r2[0] < 0.6 and r2[2] < 0.95


def test_ballet_curricula_are_gentler():
    cfg = make_microduck_flamingo_ballet_env_cfg()
    ip = cfg.curriculum["in_pose_prob"].params["param_stages"]
    assert [s["params"]["in_pose_prob"] for s in ip] == [0.6, 0.5, 0.4, 0.3]
    ps = cfg.curriculum["push_magnitude"].params["push_stages"]
    assert ps[-1]["velocity_range"]["x"][1] == 0.15 and ps[-1]["step"] == 1200 * 24


def test_v2_weights_and_pacing():
    cfg = make_microduck_flamingo_ballet_env_cfg(v2=True)
    assert cfg.rewards["swing_foot_contact"].weight == 1.5 and cfg.rewards["swing_foot_clear"].weight == 3.0
    assert cfg.rewards["swing_foot_clear"].params["targets"] == fb.CLEAR_TARGETS_V2 and min(fb.CLEAR_TARGETS_V2) >= 0.06
    assert cfg.rewards["pose_track"].params["std"] == 0.4
    assert cfg.curriculum["in_pose_prob"].params["param_stages"][-1]["step"] == 1500 * 24
    assert cfg.curriculum["push_magnitude"].params["push_stages"][-1]["velocity_range"]["x"][1] == 0.15

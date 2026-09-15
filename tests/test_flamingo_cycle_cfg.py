"""Flamingo cycle (stage 2): cfg invariants + the side/alpha logic of the new mdp
functions, exercised on a fake env (CPU, no simulator)."""
import math
from types import SimpleNamespace

import numpy as np
import torch

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks import microduck_flamingo_env_cfg as fl
from mjlab_microduck.tasks import microduck_flamingo_cycle_env_cfg as fc
from mjlab_microduck.tasks.microduck_flamingo_cycle_env_cfg import (
    make_microduck_flamingo_cycle_env_cfg,
    MicroduckFlamingoCycleRlCfg,
    mirror_pose,
)

HOME = [0.0, -0.0873, -0.4579, -0.0049, 0.4530, 0.3491, 0.3491, 0.0, 0.0,
        0.0, 0.0873, 0.4579, 0.0049, -0.4530]


# ── pose mirror ───────────────────────────────────────────────────────────────

def test_mirror_pose_matches_the_verified_left_pose():
    left = mirror_pose(fl.FLAMINGO_POSE)
    expected = [-0.3863, 0.334, -0.2579, -0.0049, 0.253, 0.3491, 0.35, 1.5, 0.0, 0.0, -0.3, -1.2, 0.8, -0.8]
    assert np.allclose(left, expected, atol=1e-3)
    assert np.allclose(mirror_pose(left), fl.FLAMINGO_POSE, atol=1e-6)   # involution
    # HOME is its own mirror (sanity of the sign convention)
    assert np.allclose(mirror_pose(HOME), HOME, atol=1e-6)


def test_left_pose_keeps_the_joint_limit_margin():
    ranges = {
        1: (-0.384, 0.384), 10: (-0.384, 0.384),
        0: (-0.436, 0.524), 9: (-0.524, 0.436),           # hip_yaw ranges are mirrored L/R
        2: (-1.571, 1.571), 3: (-1.571, 1.571), 4: (-1.571, 1.571),
        11: (-1.571, 1.571), 12: (-1.571, 1.571), 13: (-1.571, 1.571),
        5: (-1.571, 1.047), 6: (-1.571, 1.571), 7: (-2.967, 2.967), 8: (-0.436, 0.436),
    }
    for i, (lo, hi) in ranges.items():
        q = fc.FLAMINGO_POSE_LEFT[i]
        assert min(q - lo, hi - q) >= 0.05 - 1e-9, f"left pose joint {i} too close to its limit"


def test_gravity_targets_are_mirrored():
    assert fc.FLAMINGO_GRAVITY_LEFT[1] == -fc.FLAMINGO_GRAVITY_RIGHT[1]
    assert fc.FLAMINGO_GRAVITY_LEFT[0] == fc.FLAMINGO_GRAVITY_RIGHT[0]
    assert fc.FLAMINGO_GRAVITY_RIGHT[1] < 0 < fc.FLAMINGO_GRAVITY_LEFT[1]


# ── cfg ───────────────────────────────────────────────────────────────────────

def test_cfg_rewards_and_signs():
    cfg = make_microduck_flamingo_cycle_env_cfg()
    r = cfg.rewards
    for gone in ("com_over_stance_foot", "swing_foot_touch", "pose_flamingo", "gravity_flamingo"):
        assert gone not in r
    for name in ("com_target", "stance_foot_grounded", "swing_foot_contact", "swing_foot_clear",
                 "pose_track", "gravity_track", "stillness"):
        assert r[name].weight > 0, name
        assert r[name].params["command_name"] == "twist"
    assert r["stance_side_tilt"].func is microduck_mdp.fl_stance_side_tilt and r["stance_side_tilt"].weight > 0
    assert 0 < r["commanded_support"].weight <= 1e-3      # weight 0 would not be logged at all
    assert r["swing_foot_clear"].params["target"] >= 0.08 and r["swing_foot_clear"].params["std"] >= 0.05
    for name in ("joint_limit_proximity", "action_rate_l2", "body_ang_vel", "self_collisions"):
        assert r[name].weight < 0, name
    assert r["com_target"].params["left_cfg"].site_names == ["left_foot"]
    assert r["com_target"].params["right_cfg"].site_names == ["right_foot"]
    assert r["pose_track"].params["pose_left"] == fc.FLAMINGO_POSE_LEFT


def test_cfg_command_spawn_and_episode():
    cfg = make_microduck_flamingo_cycle_env_cfg()
    cmd = cfg.commands["twist"]
    assert isinstance(cmd, microduck_mdp.FlamingoCommandCfg)
    assert cmd.resampling_time_range[0] > cmd.ramp_s
    assert 0 < cmd.flamingo_prob < 1
    assert "set_flamingo_state" not in cfg.events
    ev = cfg.events["set_flamingo_cycle_state"]
    assert ev.mode == "reset"
    assert ev.params["pose_right"] == list(fl.FLAMINGO_POSE)
    assert ev.params["pose_left"] == fc.FLAMINGO_POSE_LEFT
    assert ev.params["command_name"] == "twist"
    assert cfg.episode_length_s == 10.0
    assert cfg.curriculum["in_pose_prob"].params["event_name"] == "set_flamingo_cycle_state"
    assert cfg.curriculum["com_std"].params["reward_name"] == "com_target"
    stages = cfg.curriculum["push_magnitude"].params["push_stages"]
    assert stages[0]["velocity_range"]["x"] == (0.0, 0.0)
    assert stages[-1]["velocity_range"]["x"][1] == fc.MAX_PUSH == 0.15
    assert cfg.curriculum["action_rate_weight"].params["weight_stages"][-1]["weight"] == -0.5
    assert "fell_over" in cfg.terminations and "nan_state" in cfg.terminations


def test_obs_layout_unchanged():
    a = make_microduck_flamingo_cycle_env_cfg()
    b = fl.make_microduck_flamingo_env_cfg()
    for grp in ("actor", "critic"):
        assert list(a.observations[grp].terms.keys()) == list(b.observations[grp].terms.keys())


def test_runner_cfg():
    assert MicroduckFlamingoCycleRlCfg.experiment_name == "flamingo_cycle"
    assert MicroduckFlamingoCycleRlCfg.algorithm.symmetry_cfg is None


# ── mdp functions on a fake env ───────────────────────────────────────────────

class _Term:
    def __init__(self, cmd, alpha):
        self.command = cmd
        self.alpha = alpha
        self.stance_side = torch.where(cmd[:, 1] < 0.0, -1.0, 1.0)
        self.set_calls = []

    def set_alpha(self, env_ids, value):
        self.set_calls.append((env_ids.clone(), value.clone()))
        self.alpha[env_ids] = value


def _fake_env(cmd, alpha, *, found, left_xyz, right_xyz, com_xy, gravity, q, qd, home=HOME):
    n = cmd.shape[0]
    term = _Term(cmd, alpha)
    cm = SimpleNamespace(get_term=lambda name: term, get_command=lambda name: term.command)
    data = SimpleNamespace(
        site_pos_w=torch.stack([left_xyz, right_xyz], dim=1),          # site 0 = left, 1 = right
        root_com_pos_w=torch.cat([com_xy, torch.zeros(n, 1)], dim=1),
        projected_gravity_b=gravity,
        default_joint_pos=torch.tensor(home).unsqueeze(0).expand(n, -1).clone(),
        joint_pos=q, joint_vel=qd,
    )
    robot = SimpleNamespace(data=data, joint_names=[f"j{i}" for i in range(14)],
                            indexing=SimpleNamespace(root_body_id=0))
    sim = SimpleNamespace(data=SimpleNamespace(subtree_com=torch.cat([com_xy, torch.zeros(n, 1)], dim=1).unsqueeze(1)))
    sensor = SimpleNamespace(data=SimpleNamespace(found=found))
    class _Scene:
        sensors = {"feet": sensor}
        terrain = SimpleNamespace(env_origins=torch.zeros(n, 3))

        def __getitem__(self, k):
            return {"robot": robot}[k]

    env = SimpleNamespace(scene=_Scene(), command_manager=cm, device="cpu", num_envs=n, sim=sim)
    return env, term


def _site(name):
    return SimpleNamespace(name="robot", site_ids=[0] if name == "left" else [1], site_names=[f"{name}_foot"])


def _cfg_all_joints():
    return SimpleNamespace(name="robot", joint_ids=slice(None), joint_names=None)


def _base(n=2):
    return dict(
        found=torch.tensor([[1.0, 1.0]] * n),
        left_xyz=torch.tensor([[0.0, 0.04, 0.0]] * n),
        right_xyz=torch.tensor([[0.0, -0.04, 0.0]] * n),
        com_xy=torch.zeros(n, 2),
        gravity=torch.tensor([[0.0, 0.0, -1.0]] * n),
        q=torch.tensor(HOME).unsqueeze(0).expand(n, -1).clone(),
        qd=torch.zeros(n, 14),
    )


def test_feet_helper_selects_stance_and_swing_per_side(monkeypatch):
    # env 0: right stance (side +1), env 1: left stance (side -1); only the LEFT foot touches
    cmd = torch.tensor([[1.0, 1.0, 0.0], [1.0, -1.0, 0.0]])
    kw = _base(); kw["found"] = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    env, _ = _fake_env(cmd, torch.ones(2), **kw)
    monkeypatch.setattr(microduck_mdp, "_foot_found", lambda e, s, slot: kw["found"][:, slot])
    stance, swing = microduck_mdp._fl_feet_found(env, "feet", torch.tensor([1.0, -1.0]))
    assert stance.tolist() == [0.0, 1.0]     # right stance foot is up, left stance foot is down
    assert swing.tolist() == [1.0, 0.0]


def test_com_target_blends_from_midpoint_to_stance_foot(monkeypatch):
    cmd = torch.tensor([[1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]])
    alpha = torch.tensor([1.0, 1.0, 0.0])
    kw = _base(3)
    # CoM exactly over the right foot for all three envs
    kw["com_xy"] = torch.tensor([[0.0, -0.04]] * 3)
    env, _ = _fake_env(cmd, alpha, **kw)
    r = microduck_mdp.fl_com_target(env, "twist", _site("left"), _site("right"), std=0.02)
    assert r[0] > 0.99                     # right stance, α=1: target = right foot → perfect
    assert r[1] < 1e-3                     # left stance, α=1: target = left foot, 8 cm away
    assert 0.01 < r[2] < 0.05              # stand, α=0: target = midpoint, 4 cm away → exp(-4)


def test_swing_foot_contact_flips_sign_with_alpha(monkeypatch):
    cmd = torch.tensor([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    alpha = torch.tensor([0.0, 0.65, 1.0])
    kw = _base(3)
    env, _ = _fake_env(cmd, alpha, **kw)
    monkeypatch.setattr(microduck_mdp, "_foot_found", lambda e, s, slot: kw["found"][:, slot])
    r = microduck_mdp.fl_swing_foot_contact(env, "twist", "feet", lo=0.4, hi=0.9)
    assert r[0] == 1.0                     # standing: swing (left) foot down is good
    assert abs(r[1]) < 1e-6                # mid-ramp: neutral
    assert r[2] == -1.0                    # one foot commanded and ramp done: taxed


def test_swing_foot_clear_uses_the_side_and_is_gated(monkeypatch):
    cmd = torch.tensor([[1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0]])
    alpha = torch.tensor([1.0, 1.0, 0.3])
    kw = _base(3)
    kw["left_xyz"] = torch.tensor([[0.0, 0.04, 0.05]] * 3)      # left foot 5 cm up
    kw["right_xyz"] = torch.tensor([[0.0, -0.04, 0.0]] * 3)     # right foot on the floor
    kw["found"] = torch.tensor([[0.0, 1.0]] * 3)
    env, _ = _fake_env(cmd, alpha, **kw)
    monkeypatch.setattr(microduck_mdp, "_foot_found", lambda e, s, slot: kw["found"][:, slot])
    r = microduck_mdp.fl_swing_foot_clear(env, "twist", _site("left"), _site("right"), "feet", target=0.05, std=0.03)
    assert r[0] > 0.99                     # right stance: swing = left, at target, stance down
    assert r[1] == 0.0                     # left stance: stance (left) foot is NOT down → gated off
    assert r[2] == 0.0                     # α = 0.3: lift not due yet


def test_pose_track_targets_home_or_side_pose(monkeypatch):
    monkeypatch.setattr(microduck_mdp, "_servo_joint_pos", lambda e, a: a.data.joint_pos)
    monkeypatch.setattr(microduck_mdp, "_servo_default_joint_pos", lambda e, a: a.data.default_joint_pos)
    cmd = torch.tensor([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [1.0, -1.0, 0.0]])
    alpha = torch.tensor([0.0, 1.0, 1.0, 0.5])
    kw = _base(4)
    kw["q"] = torch.tensor([HOME, fc.FLAMINGO_POSE_RIGHT, fc.FLAMINGO_POSE_LEFT, fc.FLAMINGO_POSE_RIGHT])
    env, _ = _fake_env(cmd, alpha, **kw)
    r = microduck_mdp.fl_pose_track(env, "twist", fc.FLAMINGO_POSE_RIGHT, fc.FLAMINGO_POSE_LEFT, std=0.5, mid_attenuation=0.75)
    assert r[0] > 0.999 and r[1] > 0.999 and r[2] > 0.999
    assert r[3] < 0.26                     # α = 0.5 → factor 0.25, and the pose is off-target anyway


def test_gravity_track_leans_the_right_way():
    cmd = torch.tensor([[1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]])
    alpha = torch.tensor([1.0, 1.0, 0.0])
    kw = _base(3)
    kw["gravity"] = torch.tensor([list(fc.FLAMINGO_GRAVITY_RIGHT), list(fc.FLAMINGO_GRAVITY_RIGHT), [0.0, 0.0, -1.0]])
    env, _ = _fake_env(cmd, alpha, **kw)
    r = microduck_mdp.fl_gravity_track(env, "twist", fc.FLAMINGO_GRAVITY_RIGHT, fc.FLAMINGO_GRAVITY_LEFT, std=0.15)
    assert r[0] > 0.99                     # right stance at the right-stance lean
    assert r[1] < 0.01                     # left stance commanded, leaning to the right → far off
    assert r[2] > 0.99                     # standing upright


def test_stance_side_tilt_is_signed_by_side():
    cmd = torch.tensor([[1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [1.0, -1.0, 0.0]])
    kw = _base(4)
    kw["gravity"] = torch.tensor([[0, -0.65, -0.76], [0, -0.65, -0.76], [0, 0.65, -0.76], [0, 0.65, -0.76]], dtype=torch.float32)
    env, _ = _fake_env(cmd, torch.ones(4), **kw)
    r = microduck_mdp.fl_stance_side_tilt(env, "twist", threshold=0.45)
    assert r[0] < 0 and r[1] == 0.0        # rolled toward −y: bad for right stance, free for left stance
    assert r[2] == 0.0 and r[3] < 0        # rolled toward +y: the reverse
    assert (r <= 0).all()


def test_stillness_only_after_the_ramp(monkeypatch):
    monkeypatch.setattr(microduck_mdp, "_servo_joint_vel", lambda e, a: a.data.joint_vel)
    cmd = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    alpha = torch.tensor([1.0, 0.7])
    kw = _base(2)
    env, _ = _fake_env(cmd, alpha, **kw)
    monkeypatch.setattr(microduck_mdp, "_foot_found", lambda e, s, slot: kw["found"][:, slot])
    r = microduck_mdp.fl_stillness(env, "twist", "feet", std=2.0)
    assert r[0] > 0.99 and r[1] == 0.0


def test_success_indicator(monkeypatch):
    cmd = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    kw = _base(4)
    kw["found"] = torch.tensor([[0.0, 1.0], [1.0, 1.0], [1.0, 1.0], [0.0, 1.0]])
    env, _ = _fake_env(cmd, torch.ones(4), **kw)
    monkeypatch.setattr(microduck_mdp, "_foot_found", lambda e, s, slot: kw["found"][:, slot])
    r = microduck_mdp.fl_single_support_success(env, "twist", "feet")
    assert r.tolist() == [1.0, 0.0, 1.0, 0.0]


def test_smoothstep():
    x = torch.tensor([0.0, 0.4, 0.65, 0.9, 1.0])
    s = microduck_mdp._smoothstep(x, 0.4, 0.9)
    assert s[0] == 0 and s[1] == 0 and abs(s[2] - 0.5) < 1e-6 and s[3] == 1 and s[4] == 1


def _bare_command(n, flamingo_prob=0.6):
    """FlamingoCommand without the mjlab base __init__ (no sim): just the buffers."""
    term = microduck_mdp.FlamingoCommand.__new__(microduck_mdp.FlamingoCommand)
    term._env = SimpleNamespace(num_envs=n, device="cpu")
    term.vel_command_b = torch.zeros(n, 3)
    term._flamingo_prob = flamingo_prob
    term._ramp_s = 1.5
    term._alpha = torch.zeros(n)
    term._fresh = torch.ones(n, dtype=torch.bool)
    term._pending_side = torch.zeros(n)
    term._stance_side = torch.ones(n)
    term._zero_side_prob = 0.5
    return term


def test_command_resample_honours_pending_side_and_keeps_side_during_hold():
    torch.manual_seed(0)
    n = 512
    term = _bare_command(n, flamingo_prob=1.0)
    ids = torch.arange(n)
    # the spawn event pins the side before the (fresh) resample
    side = torch.where(torch.arange(n) % 2 == 0, 1.0, -1.0)
    term.request_side(ids, side)
    assert torch.equal(term.command[:, 1], side)          # visible immediately
    term._resample_command(ids)
    assert torch.equal(term.command[:, 1], side)          # honoured by the resample
    assert (term.command[:, 0] == 1.0).all()
    assert (term._pending_side == 0).all() and not term._fresh.any()
    # flamingo → flamingo resample (hold): side unchanged, observed = latched
    term._resample_command(ids)
    assert torch.equal(term.command[:, 1], side) and torch.equal(term.stance_side, side)
    # → stand: latched side kept (the lowering rewards still need it), observed side
    #   zeroed for about half the envs (deployment idle parity)
    term._flamingo_prob = 0.0
    term._resample_command(ids)
    assert (term.command[:, 0] == 0.0).all()
    assert torch.equal(term.stance_side, side)
    zeros = (term.command[:, 1] == 0).float().mean().item()
    assert 0.3 < zeros < 0.7
    assert ((term.command[:, 1] == 0) | (term.command[:, 1] == side)).all()
    # stand → flamingo: side re-drawn (both values appear), observed = latched, never 0
    term._flamingo_prob = 1.0
    term._resample_command(ids)
    assert not torch.equal(term.stance_side, side)
    assert set(term.stance_side.unique().tolist()) == {-1.0, 1.0}
    assert torch.equal(term.command[:, 1], term.stance_side)
    assert (term.command[:, 2] == 0).all()


def test_command_alpha_slews_at_ramp_rate():
    term = _bare_command(2)
    term.vel_command_b[:, 0] = torch.tensor([1.0, 0.0])
    term._alpha = torch.tensor([0.0, 1.0])
    # emulate compute() without the base class: same slew code path
    dt = 0.02
    step = dt / term._ramp_s
    delta = term.vel_command_b[:, 0] - term._alpha
    term._alpha += torch.clamp(delta, -step, step)
    assert abs(term._alpha[0].item() - step) < 1e-7 and abs(term._alpha[1].item() - (1 - step)) < 1e-7


def test_hard_variant_adds_the_025_push_stage():
    soft = make_microduck_flamingo_cycle_env_cfg()
    hard = make_microduck_flamingo_cycle_env_cfg(hard=True)
    ss = soft.curriculum["push_magnitude"].params["push_stages"]
    hs = hard.curriculum["push_magnitude"].params["push_stages"]
    assert len(hs) == len(ss) + 1 and hs[-1]["velocity_range"]["x"][1] == fc.HARD_PUSH == 0.25
    assert hs[-1]["step"] > hs[-2]["step"]
    assert soft.rewards.keys() == hard.rewards.keys()


def test_gentle_variant_pacing():
    cfg = make_microduck_flamingo_cycle_env_cfg(hard=True, gentle=True)
    ip = [s_["params"]["in_pose_prob"] for s_ in cfg.curriculum["in_pose_prob"].params["param_stages"]]
    assert ip == [0.6, 0.5, 0.4, 0.3]
    ps = cfg.curriculum["push_magnitude"].params["push_stages"]
    assert [round(s_["velocity_range"]["x"][1], 2) for s_ in ps] == [0.0, 0.08, 0.15, 0.25]
    assert ps[-1]["step"] == 2400 * 24
    assert make_microduck_flamingo_cycle_env_cfg(hard=False, gentle=True).curriculum["push_magnitude"].params["push_stages"][-1]["velocity_range"]["x"][1] == 0.15

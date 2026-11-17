"""Peck task: cfg invariants, pose mirroring, joint limits, peck_rhythm logic (CPU)."""

import math
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks import microduck_peck_env_cfg as peck
from mjlab_microduck.tasks.microduck_peck_env_cfg import (
    MicroduckPeckLeftRlCfg,
    MicroduckPeckRightRlCfg,
    make_microduck_peck_env_cfg,
)

_XML = Path(__file__).parents[1] / "src/mjlab_microduck/robot/microduck/robot_groundcontact.xml"
_JOINT_ORDER = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
]


def _joint_ranges():
    ranges = {}
    for j in ET.parse(_XML).getroot().iter("joint"):
        if j.get("range"):
            lo, hi = map(float, j.get("range").split())
            ranges[j.get("name")] = (lo, hi)
    return [ranges[n] for n in _JOINT_ORDER]


@pytest.fixture(scope="module", params=["right", "left"])
def side_cfg(request):
    return request.param, make_microduck_peck_env_cfg(peck_foot=request.param)


def test_home_is_mirror_symmetric_so_absolute_mirroring_is_valid():
    assert peck.mirror_pose(peck._HOME) == pytest.approx(peck._HOME, abs=1e-9)
    assert peck.mirror_pose(peck.SIT_POSE) == pytest.approx(peck.SIT_POSE, abs=1e-9)


def test_left_pose_is_the_mirror_of_right():
    left, off_l = peck.peck_targets("left")
    right, off_r = peck.peck_targets("right")
    # legs swap with sign flip, sagittal neck joints identical, yaw/roll negated
    assert left[0:5] == pytest.approx([-v for v in right[9:14]])
    assert left[9:14] == pytest.approx([-v for v in right[0:5]])
    assert left[5:7] == pytest.approx(right[5:7])
    assert left[7:9] == pytest.approx([-v for v in right[7:9]])
    assert off_l == pytest.approx((-off_r[0], off_r[1], off_r[2]))
    assert peck.mirror_pose(left) == pytest.approx(right)


@pytest.mark.parametrize("side", ["right", "left"])
def test_targets_inside_soft_joint_limits(side):
    # soft_joint_pos_limit_factor = 0.9 on the robot cfg: targets must not park
    # the joints on (or past) the limits — the real-robot pose exceeds them.
    # One documented exception: neck_pitch -1.50 (soft -1.44) — the head can't
    # reach the foot from inside the soft range without tipping the trunk
    # (see the cfg docstring); it must still keep a margin to the HARD limit.
    pose, _ = peck.peck_targets(side)
    for name, q, (lo, hi) in zip(_JOINT_ORDER, pose, _joint_ranges()):
        mid, half = (lo + hi) / 2, (hi - lo) / 2
        if name == "neck_pitch":
            assert lo + 0.05 <= q <= hi - 0.05, f"{side} neck_pitch={q} too close to hard limit"
            continue
        assert abs(q - mid) <= 0.9 * half, f"{side} {name}={q} outside soft range"


def test_peck_pose_extends_the_named_leg():
    right, _ = peck.peck_targets("right")
    # extended leg: knee near straight; folded leg: knee near the sit fold
    assert abs(right[12]) < 0.2 and abs(right[3]) > 1.3
    assert right[5] < -1.0  # head dropped forward


def test_rewards_and_signs(side_cfg):
    _, cfg = side_cfg
    r = cfg.rewards
    for name in ("peck_pose_legs", "peck_pose_neck", "peck_height", "peck_beak_proximity",
                 "peck_composite", "rest_stillness", "upright_linear"):
        assert r[name].weight > 0, name
    # self-negating (≤ 0) helpers → POSITIVE weights
    for name in ("peck_pose_legs_l1", "peck_pose_neck_l1", "descent_speed", "gentle_motion"):
        assert r[name].weight > 0, name
    # cost functions (≥ 0) → NEGATIVE weights
    for name in ("peck_impact", "self_collisions", "action_rate_l2"):
        assert r[name].weight < 0, name
    # sitstand's task stack is gone
    for name in ("posture_composite", "head_pose_tracking", "rise_bootstrap", "posture_pose_legs"):
        assert name not in r, name


@pytest.mark.parametrize("term,cur", [("peck_rhythm", "peck_rhythm_weight"),
                                      ("peck_beak_bob", "peck_bob_weight")])
def test_peck_terms_start_off_and_are_curriculum_enabled(side_cfg, term, cur):
    _, cfg = side_cfg
    assert cfg.rewards[term].weight == 0.0
    stages = cfg.curriculum[cur].params["weight_stages"]
    assert stages[0]["weight"] == 0.0 and stages[-1]["weight"] > 0.0
    assert stages[-1]["step"] >= 300 * 24  # after the reach exists


def test_stillness_only_pays_under_sit(side_cfg):
    _, cfg = side_cfg
    assert cfg.rewards["rest_stillness"].func is microduck_mdp.peck_sit_stillness


def test_sensors_target_the_right_foot(side_cfg):
    side, cfg = side_cfg
    sensors = {s.name: s for s in cfg.scene.sensors}
    s = sensors["beak_foot_contact"]
    assert s.primary.pattern == "jaw_soft"
    assert s.secondary.pattern == f"ankle_{side}"
    assert cfg.rewards["peck_rhythm"].params["foot_body"] == f"ankle_{side}"
    assert "self_collision" in sensors


def test_command_and_obs_layout(side_cfg):
    _, cfg = side_cfg
    cmd = cfg.commands["twist"]
    assert isinstance(cmd, microduck_mdp.PeckCommandCfg)
    assert 0.0 < cmd.peck_prob < 1.0  # zero command (sit hold / return) trained
    actor = list(cfg.observations["actor"].terms)
    assert actor == ["base_ang_vel", "projected_gravity", "joint_pos", "joint_vel",
                     "actions", "command", "head_command", "body_command"]
    assert "peck_beak_target" not in actor and "peck_alpha" not in actor
    assert "peck_beak_target" in cfg.observations["critic"].terms
    # sitstand's head-range curriculum and seated perturbations removed
    for name in ("head_pose_range", "seated_head_jerk_prob", "seated_tip_range", "rise_speed_weight"):
        assert name not in cfg.curriculum
    for name in ("seated_head_jerk", "seated_tip"):
        assert name not in cfg.events
    assert "expand_bam_friction_fields" in cfg.events
    assert "nan_state" in cfg.terminations


def test_runner_cfgs():
    assert MicroduckPeckRightRlCfg.experiment_name != MicroduckPeckLeftRlCfg.experiment_name
    assert MicroduckPeckRightRlCfg.algorithm.symmetry_cfg is None
    assert MicroduckPeckRightRlCfg.actor.obs_normalization


# ── peck_rhythm logic on a fake env ─────────────────────────────────────────

class _FakeEnv:
    def __init__(self, n=1):
        self.num_envs = n
        self.device = "cpu"
        self.step_dt = 0.02
        self.found = torch.zeros(n, 1)
        self.flag = torch.ones(n)
        self.alpha = torch.ones(n)
        self.scene = {"beak": SimpleNamespace(data=SimpleNamespace(found=self.found))}
        cmd = SimpleNamespace(alpha=self.alpha)
        self.command_manager = SimpleNamespace(
            get_term=lambda name: cmd,
            get_command=lambda name: torch.stack([self.flag, 0 * self.flag, 0 * self.flag], -1),
        )


def _run(seq, monkeypatch, flag=1.0):
    """seq: list of (touch: bool, beak_z: float) per step → per-step payouts."""
    env = _FakeEnv()
    env.flag[:] = flag
    z = torch.zeros(1)
    monkeypatch.setattr(microduck_mdp, "_peck_beak_and_target",
                        lambda *a, **k: (torch.stack([0 * z, 0 * z, z], -1), None))
    term = microduck_mdp.peck_rhythm(SimpleNamespace(), env)
    out = []
    for touch, zz in seq:
        env.found[:] = float(touch)
        z[:] = zz
        out.append(float(term(env, "twist", "beak", "ankle_right", (0, 0, 0))))
    return out


def _peck(air_steps, lift):
    up = [(False, lift)] * air_steps
    return up + [(True, 0.0)] * 3


def test_good_peck_pays(monkeypatch):
    pay = _run([(True, 0.0)] * 3 + _peck(20, 0.02), monkeypatch)  # 0.4 s air, 2 cm lift
    assert max(pay) == pytest.approx(1.0) and sum(p > 0 for p in pay) == 1


def test_peck_pay_is_graded_by_lift(monkeypatch):
    # v2: no hard lift gate — a small bob already pays, proportionally
    small = max(_run([(True, 0.0)] + _peck(20, 0.005), monkeypatch))
    assert small == pytest.approx(0.25)


def test_contact_flicker_pays_nothing(monkeypatch):
    assert max(_run([(True, 0.0)] + _peck(2, 0.02), monkeypatch)) == 0.0   # 0.04 s air
    assert max(_run([(True, 0.0)] + _peck(20, 0.0), monkeypatch)) == 0.0   # no lift


def test_long_lifts_are_capped(monkeypatch):
    long = max(_run([(True, 0.0)] + _peck(100, 0.02), monkeypatch))
    assert long == pytest.approx(1.0)


def test_no_pay_when_sit_commanded(monkeypatch):
    assert max(_run([(True, 0.0)] + _peck(20, 0.02), monkeypatch, flag=0.0)) == 0.0

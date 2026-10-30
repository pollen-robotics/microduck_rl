"""Gear protection (2026-10-30, branch gear_protection): the XL330 bench's wear model in
the reward, the physics-rate gear load, and bounded holds in the stand expert's labels.

CPU only, pure torch on fake envs; never steps a simulation.
"""
import math

import pytest
import torch
from bam.mjlab import BamActuator

import mjlab_microduck.tasks  # noqa: F401  (registers tasks)
from mjlab_microduck.actuator.friction_dr_bam import FrictionDRBamActuator
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks import microduck_velstand_env_cfg as vs
from mjlab_microduck.tasks.distill import clamp_label_holds

SERVOS = ("left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
          "neck_pitch", "head_pitch", "head_yaw", "head_roll",
          "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle")


class _Data:
    pass


class _FakeActuator:
    def __init__(self, peak, names):
        self._peak = peak
        self._target_names = list(names)
        self.taken = 0

    def take_gear_load_peak(self):
        self.taken += 1
        return self._peak.clone()


class _Asset:
    def __init__(self, torque, vel, names=SERVOS, peak=None):
        self.data = _Data()
        self.data.actuator_force = torque
        self.data.joint_vel = vel
        self.joint_names = list(names)
        self.actuators = [] if peak is None else [_FakeActuator(peak, names)]

    def find_joints(self, pattern):
        return list(range(len(self.joint_names))), self.joint_names


class _Env:
    def __init__(self, torque, vel, peak=None):
        self.scene = {"robot": _Asset(torque, vel, peak=peak)}
        self.num_envs = torque.shape[0]
        self.device = "cpu"
        self.step_dt = 0.02
        self.common_step_counter = 0


# ── wear model ───────────────────────────────────────────────────────────────

def test_gear_wear_is_flat_below_the_bench_plateau():
    load = torch.tensor([0.0, 0.39, 0.40, 0.90, 1.40])
    w = microduck_mdp.gear_wear(load)
    assert w[:3].tolist() == [0.0, 0.0, 0.0]
    assert math.isclose(w[3].item(), 0.5 ** 2.4, rel_tol=1e-5)
    assert math.isclose(w[4].item(), 1.0, rel_tol=1e-5)


def test_gear_hold_cost_is_graded_and_only_for_stalled_servos():
    torque = torch.zeros(3, 14)
    vel = torch.zeros(3, 14)
    torque[0, 3] = 0.6           # knee pushing at 0.6, not moving
    torque[1, 3] = 1.0           # same, at 1.0: (0.7/0.3)² ≈ 5.4× the cost
    torque[2, 3] = 1.0
    vel[2, 3] = 3.0              # moving: a stride, not a hold
    out = microduck_mdp.gear_hold_cost(_Env(torque, vel), threshold=0.3, vel_thresh=0.5)
    assert math.isclose(out[0].item(), 0.09, rel_tol=1e-5)
    assert math.isclose(out[1].item(), 0.49, rel_tol=1e-5)
    assert out[2].item() == 0.0


def test_gear_hold_cost_joint_weights():
    torque = torch.zeros(1, 14)
    torque[0, 3] = 0.5           # left knee
    torque[0, 6] = 0.5           # head pitch
    env = _Env(torque, torch.zeros(1, 14))
    out = microduck_mdp.gear_hold_cost(env, threshold=0.3, joint_weights={r".*knee": 2.0, r"head_.*": 0.0})
    assert math.isclose(out.item(), 2.0 * 0.04, rel_tol=1e-5)


def test_gear_load_peak_maps_actuator_targets_to_servo_order_and_caches_per_step():
    names = list(reversed(SERVOS))            # an actuator ordered differently from the joints
    peak = torch.arange(14, dtype=torch.float32).unsqueeze(0) / 10
    env = _Env(torch.zeros(1, 14), torch.zeros(1, 14), peak=None)
    env.scene["robot"].actuators = [_FakeActuator(peak, names)]
    out = microduck_mdp.gear_load_peak(env)
    for k, name in enumerate(names):
        assert math.isclose(out[0, SERVOS.index(name)].item(), k / 10, rel_tol=1e-6)
    microduck_mdp.gear_load_peak(env)           # same step: the cached window
    assert env.scene["robot"].actuators[0].taken == 1
    env.common_step_counter += 1
    microduck_mdp.gear_load_peak(env)           # next step: a new window
    assert env.scene["robot"].actuators[0].taken == 2


def test_gear_impact_cost_is_free_below_threshold():
    peak = torch.zeros(2, 14)
    peak[0, 3] = 0.39
    peak[1, 3] = 0.9
    env = _Env(torch.zeros(2, 14), torch.zeros(2, 14), peak=peak)
    out = microduck_mdp.gear_impact_cost(env, threshold=0.4, power=2.0)
    assert out[0].item() == 0.0
    assert math.isclose(out[1].item(), 0.25, rel_tol=1e-5)


# ── physics-rate gear load in the actuator ───────────────────────────────────

def test_actuator_gear_load_is_motor_torque_minus_rotor_inertia_times_accel(monkeypatch):
    act = FrictionDRBamActuator.__new__(FrictionDRBamActuator)
    act._dt = 0.005
    act._rotor_inertia = 0.002
    n, j = 2, 3
    act.gear_load_peak = torch.zeros(n, j)
    act._gear_prev_vel = torch.zeros(n, j)
    act._gear_prev_torque = torch.zeros(n, j)
    act._gear_valid = torch.zeros(n, 1, dtype=torch.bool)
    torques = iter([torch.full((n, j), 0.3), torch.full((n, j), 0.3), torch.full((n, j), 0.3)])
    monkeypatch.setattr(BamActuator, "compute", lambda self, cmd: next(torques))

    class Cmd:
        def __init__(self, v):
            self.vel = v

    act.compute(Cmd(torch.full((n, j), 5.0)))       # first call after a reset: not billed
    assert act.gear_load_peak.abs().sum() == 0
    act.compute(Cmd(torch.full((n, j), 5.0)))       # steady motion: the load is the torque
    assert torch.allclose(act.gear_load_peak, torch.full((n, j), 0.3))
    act.compute(Cmd(torch.zeros(n, j)))             # stopped dead in one substep: 1000 rad/s²
    # 0.3 − 0.002·(−1000) = 2.3 N·m: the rotor's inertia, stopped by the teeth
    assert torch.allclose(act.take_gear_load_peak(), torch.full((n, j), 2.3))
    assert act.gear_load_peak.abs().sum() == 0      # take() starts a new window


# ── bounded holds in the expert labels ───────────────────────────────────────

def test_clamp_label_holds_keeps_direction_and_bounds_only_stalled_listed_joints():
    obs = torch.zeros(1, 61)
    obs[0, 6 + 3] = 0.2          # left knee at +0.2 from HOME, not moving
    obs[0, 6 + 2] = 0.0          # left hip pitch at HOME, moving fast
    obs[0, 20 + 2] = 2.0
    labels = torch.zeros(1, 14)
    labels[0, 3] = 1.2           # 1.0 rad past the knee: a full-torque hold
    labels[0, 2] = 1.0           # big target on a moving joint: left alone
    labels[0, 4] = 1.0           # ankle: not a listed joint
    out, clamped = clamp_label_holds(obs, labels, {"joints": (2, 3), "max_err": 0.35})
    assert math.isclose(out[0, 3].item(), 0.55, rel_tol=1e-6)
    assert out[0, 2].item() == 1.0 and out[0, 4].item() == 1.0
    assert clamped.tolist() == [[False, True]]
    neg = labels.clone()
    neg[0, 3] = -1.0             # the other way: −1.2 from the knee
    out, _ = clamp_label_holds(obs, neg, {"joints": (3,), "max_err": 0.35})
    assert math.isclose(out[0, 3].item(), -0.15, rel_tol=1e-6)


# ── wiring ───────────────────────────────────────────────────────────────────

def test_gear_protection_wired_with_negative_ramped_weights():
    if not vs.ENABLE_GEAR_PROTECTION:
        pytest.skip("ENABLE_GEAR_PROTECTION off")
    cfg = vs.make_microduck_velstand_env_cfg()
    for name, full in (("gear_hold", vs.GEAR_HOLD_WEIGHT), ("gear_impact", vs.GEAR_IMPACT_WEIGHT)):
        assert full < 0
        assert cfg.rewards[name].weight < 0
        stages = cfg.curriculum[f"{name}_weight"].params["weight_stages"]
        assert stages[0]["step"] == 0 and math.isclose(stages[0]["weight"], cfg.rewards[name].weight)
        assert math.isclose(stages[-1]["weight"], full)
        # ramps from v8's last iteration: a plain resume keeps its step counter
        assert stages[1]["step"] > vs.GEAR_BASE_ITER * vs.NUM_STEPS_PER_ENV
    play = vs.make_microduck_velstand_env_cfg(play=True)
    assert "gear_hold_weight" not in play.curriculum


def test_label_hold_clamp_targets_knees_hip_pitch_and_neck():
    if not vs.ENABLE_LABEL_HOLD_CLAMP:
        pytest.skip("ENABLE_LABEL_HOLD_CLAMP off")
    bc = vs.MicroduckVelStandRlCfg.algorithm.bc_cfg
    hc = bc["label_hold_clamp"]
    assert {SERVOS[i] for i in hc["joints"]} == {
        "left_hip_pitch", "left_knee", "neck_pitch", "right_hip_pitch", "right_knee"}
    assert 0.0 < hc["max_err"] < 1.0

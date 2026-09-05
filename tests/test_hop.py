"""Unit tests for the ported hop reward terms (duck-typed fakes)."""

import math

import pytest
import torch

from mjlab_microduck.tasks.mdp import (
    com_height_target_recovery_only,
    hop_body_height,
    hop_both_feet_airborne,
    hop_energy_monitor,
    hop_load_force,
    hop_symmetric_push,
    hop_upward_velocity,
)

_SENSOR = "feet_ground_contact"
_CMD = "twist"


class _SensorData:
    def __init__(self, found, force=None):
        self.found = torch.tensor(found, dtype=torch.float32)
        # [B, N, 3]. `feet_ground_contact` is reduce="netforce", so this is the
        # summed contact force per foot geom in the GLOBAL frame; MuJoCo reports
        # it pointing DOWN for a loaded foot (probed: -4.905 N under a 4.905 N
        # weight), which is why the fixtures below use negative z.
        self.force = None if force is None else torch.tensor(force, dtype=torch.float32)


class _Sensor:
    def __init__(self, found, force=None):
        self.data = _SensorData(found, force)


class _AssetData:
    def __init__(self, vz, z):
        n = len(vz)
        self.root_link_lin_vel_w = torch.zeros((n, 3), dtype=torch.float32)
        self.root_link_lin_vel_w[:, 2] = torch.tensor(vz, dtype=torch.float32)
        self.root_link_pos_w = torch.zeros((n, 3), dtype=torch.float32)
        self.root_link_pos_w[:, 2] = torch.tensor(z, dtype=torch.float32)


class _Asset:
    def __init__(self, vz, z):
        self.data = _AssetData(vz, z)


class _CommandManager:
    def __init__(self, cmd):
        self._cmd = torch.tensor(cmd, dtype=torch.float32)

    def get_command(self, _name):
        return self._cmd


class _Terrain:
    """Flat ground: `com_height_target` subtracts env_origins[:, 2] from world z."""

    def __init__(self, n):
        self.env_origins = torch.zeros((n, 3), dtype=torch.float32)


class _Scene:
    def __init__(self, sensors, asset, n=1):
        self.sensors = sensors
        self._asset = asset
        self.terrain = _Terrain(n)

    def __getitem__(self, _k):
        return self._asset


class _Env:
    """found: per-foot contact flags; cmd: [cos, sin, 0]; vz/z: base state."""

    def __init__(self, found=((0.0, 0.0),), cmd=((0.0, 1.0, 0.0),),
                 vz=(0.0,), z=(0.15,), force=None):
        self.scene = _Scene({_SENSOR: _Sensor(found, force)}, _Asset(vz, z), len(found))
        self.command_manager = _CommandManager(cmd)
        self.num_envs = len(found)
        self.device = "cpu"
        self.extras = {"log": {}}

    def step(self, found=None, z=None, cmd=None):
        """Advance one step, mutating only the fields the terms read.

        The two airborne-gated hop rewards are STATEFUL -- they latch the base
        height at the instant of takeoff and score rise above it -- so they
        cannot be exercised by a single call the way the stateless terms can.
        """
        if found is not None:
            self.scene.sensors[_SENSOR].data.found = torch.tensor(
                found, dtype=torch.float32
            )
        if z is not None:
            self.scene._asset.data.root_link_pos_w[:, 2] = torch.tensor(
                z, dtype=torch.float32
            )
        if cmd is not None:
            self.command_manager._cmd = torch.tensor(cmd, dtype=torch.float32)
        return self


# Contact and phase fixtures for the stateful terms.
_PLANTED = [[1.0, 1.0]]
_AIRBORNE = [[0.0, 0.0]]
_LAUNCH = [[0.0, 1.0, 0.0]]     # sin = +1, mid-launch
_RECOVERY = [[0.0, -1.0, 0.0]]  # sin = -1, mid-recovery


def _airborne_term(env):
    """Instantiate the class term the way ManagerBase._resolve_common_term_cfg does."""
    return hop_both_feet_airborne(cfg=None, env=env)


def _height_term(env):
    return hop_body_height(cfg=None, env=env)


# --- hop_both_feet_airborne -------------------------------------------------
#
# Rise, not absolute height. The datum is the LAST IN-CONTACT base height --
# not the first airborne sample, which loses up to 14 mm to the 20 ms control
# step and cleared MIN_RISE on only 33% of takeoff phases for a 5 mm hop. It is
# held for the flight, so a term must be driven over several steps:
# planted (stance datum) -> airborne (latch) -> airborne higher (rise).


def test_airborne_rewarded_when_the_body_actually_rises():
    env = _Env(found=_PLANTED, cmd=_LAUNCH, z=[0.147])
    term = _airborne_term(env)
    term(env, sensor_name=_SENSOR, command_name=_CMD)              # planted
    term(env.step(found=_AIRBORNE, z=[0.147]), sensor_name=_SENSOR, command_name=_CMD)
    out = term(env.step(z=[0.180]), sensor_name=_SENSOR, command_name=_CMD)
    assert abs(float(out[0]) - 1.0) < 1e-6


def test_airborne_not_rewarded_when_a_foot_is_down():
    env = _Env(found=[[1.0, 0.0]], cmd=_LAUNCH, z=[0.147])
    term = _airborne_term(env)
    out = term(env.step(z=[0.180]), sensor_name=_SENSOR, command_name=_CMD)
    assert float(out[0]) == 0.0


def test_airborne_not_rewarded_during_the_recovery_half_cycle():
    """sin < 0 is the recovery half — flight there must not be paid for,
    or the policy is rewarded for simply never landing."""
    env = _Env(found=_AIRBORNE, cmd=_RECOVERY, z=[0.147])
    term = _airborne_term(env)
    term(env, sensor_name=_SENSOR, command_name=_CMD)
    out = term(env.step(z=[0.180]), sensor_name=_SENSOR, command_name=_CMD)
    assert float(out[0]) == 0.0


def test_airborne_pays_nothing_for_a_tall_tuck():
    """THE EXPLOIT THIS FRAME CLOSES, and the reason absolute height was wrong.

    Exactly two collision geoms exist on this robot -- the two foot pads; every
    other geom is contype=0 -- so "both feet airborne" is a statement about two
    70 g pads and says NOTHING about the 877 g CoM. Retracting both feet with
    the trunk motionless satisfies the contact predicate outright.

    An absolute-height THRESHOLD does not fix that, which is the part that cost
    us a round: the robot has ~14.2 mm of sag-free posture headroom (max planted
    stance root 0.16133 vs HOME_FRAME's 0.14710), so it can stand tall at
    z ~ 0.155 and tuck from there, sitting above any threshold within 14.2 mm of
    stance for the whole tuck. That bought ~1.8/step of airborne reward for
    ~0.20/step of `pose`, scoring 5.1/step against a genuine 33 mm hop's 5.4.
    And the threshold could not be raised past it without also gating out the
    Locked arm's expected ~5 mm hop, destroying the controlled comparison.

    NOTE THE HEIGHT: 0.155 is ABSOLUTELY HIGH -- higher than the nominal 0.147
    stance, higher than the old 0.1521 gate -- and it still pays zero, because
    the body did not RISE. That is the whole point, so it is asserted explicitly.
    """
    env = _Env(found=_PLANTED, cmd=_LAUNCH, z=[0.155])
    term = _airborne_term(env)
    term(env, sensor_name=_SENSOR, command_name=_CMD)                  # tall stance
    out1 = term(env.step(found=_AIRBORNE), sensor_name=_SENSOR, command_name=_CMD)
    out2 = term(env, sensor_name=_SENSOR, command_name=_CMD)           # still tucked
    out3 = term(env.step(z=[0.1545]), sensor_name=_SENSOR, command_name=_CMD)  # dips
    assert float(out1[0]) == 0.0
    assert float(out2[0]) == 0.0
    assert float(out3[0]) == 0.0
    # ...and it is above the absolute threshold the previous round used.
    assert 0.155 > 0.1521


def test_airborne_rise_is_measured_from_takeoff_not_a_fixed_datum():
    """Identical rise from two very different takeoff heights must score the
    same. Under an absolute target the low hop would score less purely for
    having started lower, which is a posture measurement, not a hop."""
    results = []
    for takeoff in (0.120, 0.160):
        env = _Env(found=_PLANTED, cmd=_LAUNCH, z=[takeoff])
        term = _airborne_term(env)
        term(env, sensor_name=_SENSOR, command_name=_CMD)
        term(env.step(found=_AIRBORNE), sensor_name=_SENSOR, command_name=_CMD)
        results.append(
            float(term(env.step(z=[takeoff + 0.020]), sensor_name=_SENSOR,
                       command_name=_CMD)[0])
        )
    assert results[0] == results[1] == 1.0


def test_airborne_latch_resets_on_regaining_contact():
    """A second hop in the same episode must measure from ITS OWN takeoff.
    Without the reset the first hop's latch would persist and the robot would be
    paid for standing tall after landing."""
    env = _Env(found=_PLANTED, cmd=_LAUNCH, z=[0.147])
    term = _airborne_term(env)
    term(env, sensor_name=_SENSOR, command_name=_CMD)
    term(env.step(found=_AIRBORNE), sensor_name=_SENSOR, command_name=_CMD)
    assert float(term(env.step(z=[0.180]), sensor_name=_SENSOR, command_name=_CMD)[0]) == 1.0

    # Land at the apex height and stay planted there: the stale latch (0.147)
    # would otherwise still read 33 mm of rise.
    term(env.step(found=_PLANTED, z=[0.180]), sensor_name=_SENSOR, command_name=_CMD)
    out = term(env.step(found=_AIRBORNE), sensor_name=_SENSOR, command_name=_CMD)
    assert float(out[0]) == 0.0, "the latch must re-arm at the new takeoff height"


def test_airborne_latch_resets_on_env_reset():
    """RewardManager.reset calls func.reset(env_ids=...) on every class term."""
    env = _Env(found=_PLANTED, cmd=_LAUNCH, z=[0.147])
    term = _airborne_term(env)
    term(env, sensor_name=_SENSOR, command_name=_CMD)
    term(env.step(found=_AIRBORNE), sensor_name=_SENSOR, command_name=_CMD)
    term.reset(env_ids=slice(None))
    assert float(term._z_takeoff[0]) == 0.0
    assert not bool(term._was_airborne[0])
    # A fresh episode: still airborne on the first post-reset step, so the latch
    # re-arms here rather than reporting the pre-reset height as a rise.
    out = term(env.step(z=[0.180]), sensor_name=_SENSOR, command_name=_CMD)
    assert float(out[0]) == 0.0


def test_min_rise_admits_a_locked_arm_hop_and_rejects_a_tuck():
    """The threshold's two-sided constraint. It must reject the ~1 mm a tuck's
    trunk dip produces AND admit the Locked control arm's expected ~5 mm hop --
    if it gated out Locked, the arm-to-arm comparison that IS the experiment
    would be meaningless."""
    def rise_scores(rise_m):
        env = _Env(found=_PLANTED, cmd=_LAUNCH, z=[0.147])
        term = _airborne_term(env)
        term(env, sensor_name=_SENSOR, command_name=_CMD)
        term(env.step(found=_AIRBORNE), sensor_name=_SENSOR, command_name=_CMD)
        return float(
            term(env.step(z=[0.147 + rise_m]), sensor_name=_SENSOR,
                 command_name=_CMD, min_rise=0.003)[0]
        )

    assert rise_scores(0.001) == 0.0, "a 1 mm dip/tuck must not pay"
    assert rise_scores(0.005) == 1.0, "the Locked arm's ~5 mm hop must pay"


def test_rise_is_measured_from_the_last_stance_sample_not_the_first_airborne_one():
    """THE SAMPLING FIX, and the test that distinguishes the two designs.

    Rewards update once per CONTROL step -- sim timestep 0.005 x decimation 4 =
    20 ms -- so takeoff falls uniformly inside a 20 ms window and the first
    airborne SAMPLE has already flown for up to 20 ms. Measuring from that
    sample discards whatever rise happened first.

    Sampled across takeoff phase 0/5/10/15/20 ms, a true 5 mm Locked-arm hop
    read from the first airborne sample measures 4.68 / 3.32 / 2.34 / 1.36 /
    0.38 mm. Against MIN_RISE = 0.003 the mean-phase margin is MINUS 0.66 mm and
    only 33% of takeoff phases clear the gate -- so the Locked CONTROL ARM loses
    two thirds of its airborne reward while genuinely hopping, silently zeroing
    the control and making every sprung-vs-Locked number meaningless.

    The fixture is built so the two designs disagree: stance at 0.147, the first
    airborne sample ALREADY at 0.150 (it flew during part of that step), apex at
    0.152. From the stance datum the rise is 5 mm and clears MIN_RISE; from the
    first airborne sample it is 2 mm and does not.
    """
    env = _Env(found=_PLANTED, cmd=_LAUNCH, z=[0.147])
    term = _airborne_term(env)
    term(env, sensor_name=_SENSOR, command_name=_CMD)               # stance, 0.147
    term(env.step(found=_AIRBORNE, z=[0.150]), sensor_name=_SENSOR, command_name=_CMD)
    out = term(env.step(z=[0.152]), sensor_name=_SENSOR, command_name=_CMD)
    assert float(out[0]) == 1.0, (
        "a 5 mm hop whose takeoff fell mid-control-step must still clear MIN_RISE"
    )


def test_measured_rise_equals_the_stance_referenced_value():
    """The same fixture, asserted on the VALUE rather than on the gate.

    A Gaussian with std 1 mm centred on 5 mm reads ~1.0 for the stance-referenced
    rise (5 mm) and ~1e-4 for the first-airborne-sample rise (2 mm), so this pins
    which datum was used rather than merely that some rise was measured.
    """
    env = _Env(found=_PLANTED, cmd=_LAUNCH, z=[0.147])
    term = _height_term(env)
    p = dict(target_rise=0.005, std=0.001)
    term(env, command_name=_CMD, **p)
    term(env.step(found=_AIRBORNE, z=[0.150]), command_name=_CMD, **p)
    out = float(term(env.step(z=[0.152]), command_name=_CMD, **p)[0])
    assert out > 0.99, f"measured rise is not the stance-referenced 5 mm (scored {out:.5f})"


def test_a_tuck_is_still_rejected_under_the_stance_datum():
    """The stance datum must not re-open the exploit it was not meant to touch.

    A tuck's last in-contact height EQUALS its airborne height -- the trunk does
    not move -- so its rise stays ~0 and MIN_RISE still rejects it. This is why
    MIN_RISE could stay at 0.003 instead of being lowered to 0.0003 to admit the
    worst-case Locked hop, which would have been indistinguishable from no gate.
    """
    env = _Env(found=_PLANTED, cmd=_LAUNCH, z=[0.155])
    term = _airborne_term(env)
    term(env, sensor_name=_SENSOR, command_name=_CMD)
    out1 = term(env.step(found=_AIRBORNE), sensor_name=_SENSOR, command_name=_CMD)
    out2 = term(env.step(z=[0.1555]), sensor_name=_SENSOR, command_name=_CMD)
    assert float(out1[0]) == 0.0
    assert float(out2[0]) == 0.0


def test_a_nan_on_the_takeoff_step_cannot_manufacture_a_rise():
    """NaN must fail SAFE. `nan_to_num` maps a NaN height to 0.0; latched as the
    takeoff datum that reads the NEXT step as ~0.147 m of rise and pays this term
    its full weight of 12.0 during a physics blow-up. The old absolute-height
    gate failed safe here (0 > min_height is False), so the rise frame must not
    regress it. Bounded by the `nan_state` termination, but this reward must not
    be the thing that pays out.

    Step 3 holds the SAME height as stance -- a tuck -- so any non-zero reward is
    manufactured by the NaN rather than earned.
    """
    env = _Env(found=_PLANTED, cmd=_LAUNCH, z=[0.147])
    term = _airborne_term(env)
    term(env, sensor_name=_SENSOR, command_name=_CMD)
    nan_step = term(
        env.step(found=_AIRBORNE, z=[float("nan")]), sensor_name=_SENSOR, command_name=_CMD
    )
    after = term(env.step(z=[0.147]), sensor_name=_SENSOR, command_name=_CMD)
    assert torch.isfinite(nan_step).all() and float(nan_step[0]) == 0.0
    assert float(after[0]) == 0.0, "a NaN step manufactured a rise from a stale datum"


def test_a_nan_step_does_not_consume_the_takeoff_transition():
    """Holding `_was_airborne` across a non-finite step is what makes the guard
    above work: if the NaN step consumed the transition, `took_off` would be
    False forever after and `_z_takeoff` would stay stale. The retry must latch
    the correct datum, so a real hop after a NaN blip still scores."""
    env = _Env(found=_PLANTED, cmd=_LAUNCH, z=[0.147])
    term = _airborne_term(env)
    term(env, sensor_name=_SENSOR, command_name=_CMD)
    term(env.step(found=_AIRBORNE, z=[float("nan")]), sensor_name=_SENSOR, command_name=_CMD)
    term(env.step(z=[0.147]), sensor_name=_SENSOR, command_name=_CMD)
    out = term(env.step(z=[0.180]), sensor_name=_SENSOR, command_name=_CMD)
    assert float(out[0]) == 1.0
    assert float(term._z_takeoff[0]) == pytest.approx(0.147, abs=1e-5)


def test_no_stance_datum_yet_pays_nothing_rather_than_fabricating_a_rise():
    """An env whose first observed step is already airborne has no stance datum.
    It falls back to the current height, giving rise 0 -- silent, not a hop."""
    env = _Env(found=_AIRBORNE, cmd=_LAUNCH, z=[0.147])
    term = _airborne_term(env)
    assert float(term(env, sensor_name=_SENSOR, command_name=_CMD)[0]) == 0.0
    assert float(term(env.step(z=[0.180]), sensor_name=_SENSOR, command_name=_CMD)[0]) == 1.0


def test_a_nan_first_step_with_no_stance_datum_cannot_fabricate_a_rise():
    """The NaN case that actually pins the guard, found by enumerating them.

    The obvious NaN fixture -- stance, then NaN, then a tuck -- does NOT
    distinguish the guard, because the stance datum rescues it: the latch fires
    on the NaN step but uses `z_stance`, which is correct. The case that bites is
    a NaN on a step with NO stance sample yet, where the fallback datum is the
    nan_to_num'd 0.0. If the takeoff transition is consumed there, `_z_takeoff`
    stays 0.0 and the next step reads ~0.147 m of rise -- full weight 12.0 paid
    during a physics blow-up.

    Holding `_was_airborne` across non-finite steps makes the transition retry
    and latch a real height instead. Step 3 holds the same height as step 2, so
    any non-zero reward is manufactured.
    """
    env = _Env(found=_AIRBORNE, cmd=_LAUNCH, z=[float("nan")])
    term = _airborne_term(env)
    first = term(env, sensor_name=_SENSOR, command_name=_CMD)
    second = term(env.step(z=[0.147]), sensor_name=_SENSOR, command_name=_CMD)
    third = term(env.step(z=[0.147]), sensor_name=_SENSOR, command_name=_CMD)
    assert torch.isfinite(first).all() and float(first[0]) == 0.0
    assert float(second[0]) == 0.0, "the NaN step consumed the takeoff transition"
    assert float(third[0]) == 0.0
    assert float(term._z_takeoff[0]) == pytest.approx(0.147, abs=1e-5)


def test_airborne_treats_a_nan_contact_read_as_in_contact():
    """Never pay for flight we cannot actually see."""
    env = _Env(found=_AIRBORNE, cmd=_LAUNCH, z=[0.147])
    term = _airborne_term(env)
    term(env, sensor_name=_SENSOR, command_name=_CMD)
    nan = [[float("nan"), float("nan")]]
    out = term(env.step(found=nan, z=[0.180]), sensor_name=_SENSOR, command_name=_CMD)
    assert float(out[0]) == 0.0


def test_airborne_is_zero_without_the_contact_sensor():
    env = _Env(found=_AIRBORNE, cmd=_LAUNCH, z=[0.147])
    term = _airborne_term(env)
    env.scene.sensors = {}
    out = term(env, sensor_name=_SENSOR, command_name=_CMD)
    assert out.shape == (1,)
    assert float(out[0]) == 0.0


# --- Metrics/hop_rise_mean, Metrics/hop_rise_peak ----------------------------
#
# The headline hop-height number, measured directly off `_HopRiseTracker`'s
# rise instead of backed out of a reward ratio (which previously gave ~27 mm
# as an estimate). Logged from `hop_both_feet_airborne` only -- see the
# docstring there -- and this rides on that term's weight being non-zero
# (registered at 12.0), matching the `spring_compression_loaded_mean` pattern
# of "mean over the subset that currently qualifies", not over all envs.


def test_rise_mean_is_over_airborne_envs_only_not_all_envs():
    """Two envs airborne at different rises, one still planted on the ground.
    The mean must be the mean of the two airborne rises (9 mm), NOT the
    three-env mean (6 mm) that dilutes it with the grounded env's zero."""
    found3 = [[1.0, 1.0]] * 3
    cmd3 = [[0.0, 1.0, 0.0]] * 3
    env = _Env(found=found3, cmd=cmd3, z=[0.147, 0.147, 0.147], vz=[0.0, 0.0, 0.0])
    term = _airborne_term(env)
    term(env, sensor_name=_SENSOR, command_name=_CMD)  # stance datum, all 3

    # Envs 0 and 1 take off; env 2 stays planted throughout.
    term(
        env.step(found=[[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]]),
        sensor_name=_SENSOR, command_name=_CMD,
    )
    term(
        env.step(z=[0.152, 0.160, 0.147]),
        sensor_name=_SENSOR, command_name=_CMD,
    )  # env0 rises 5 mm, env1 rises 13 mm, env2 unchanged

    log = env.extras["log"]
    assert abs(float(log["Metrics/hop_rise_mean"]) - 0.009) < 1e-6
    assert abs(float(log["Metrics/hop_rise_peak"]) - 0.013) < 1e-6


def test_rise_metrics_are_zero_and_present_when_no_env_is_airborne():
    """No env airborne this step: both keys must still exist, reading 0.0
    explicitly rather than being skipped or emitting NaN."""
    env = _Env(found=_PLANTED, cmd=_LAUNCH, z=[0.147])
    term = _airborne_term(env)
    term(env, sensor_name=_SENSOR, command_name=_CMD)

    log = env.extras["log"]
    assert "Metrics/hop_rise_mean" in log
    assert "Metrics/hop_rise_peak" in log
    assert float(log["Metrics/hop_rise_mean"]) == 0.0
    assert float(log["Metrics/hop_rise_peak"]) == 0.0


def test_rise_metrics_are_nan_safe():
    """A NaN base-height read during flight must not leak a NaN into the
    logged metrics."""
    env = _Env(found=_PLANTED, cmd=_LAUNCH, z=[0.147])
    term = _airborne_term(env)
    term(env, sensor_name=_SENSOR, command_name=_CMD)
    term(
        env.step(found=_AIRBORNE, z=[float("nan")]),
        sensor_name=_SENSOR, command_name=_CMD,
    )

    log = env.extras["log"]
    assert torch.isfinite(log["Metrics/hop_rise_mean"])
    assert torch.isfinite(log["Metrics/hop_rise_peak"])


# --- hop_upward_velocity ----------------------------------------------------

def test_upward_velocity_saturates_at_max_vel():
    env = _Env(cmd=[[0.0, 1.0, 0.0]], vz=[10.0])
    out = hop_upward_velocity(env, command_name=_CMD, max_vel=0.5)
    assert abs(float(out[0]) - 1.0) < 1e-6


def test_downward_velocity_is_not_rewarded():
    env = _Env(cmd=[[0.0, 1.0, 0.0]], vz=[-2.0])
    out = hop_upward_velocity(env, command_name=_CMD, max_vel=0.5)
    assert float(out[0]) == 0.0


def test_upward_velocity_scales_below_saturation():
    env = _Env(cmd=[[0.0, 1.0, 0.0]], vz=[0.25])
    out = hop_upward_velocity(env, command_name=_CMD, max_vel=0.5)
    assert abs(float(out[0]) - 0.5) < 1e-6


def test_upward_velocity_is_gated_by_the_launch_phase():
    """sin < 0 is the recovery half — upward velocity there must not be paid
    for, or the policy is rewarded for launching outside the hop cycle."""
    env = _Env(cmd=[[0.0, -1.0, 0.0]], vz=[10.0])
    out = hop_upward_velocity(env, command_name=_CMD, max_vel=0.5)
    assert float(out[0]) == 0.0


# --- hop_body_height --------------------------------------------------------


def _rise_reward(rise_m, takeoff=0.147, cmd=None, found_at_apex=None, **params):
    """Drive the height term through planted -> takeoff -> apex, return the reward."""
    env = _Env(found=_PLANTED, cmd=cmd or _LAUNCH, z=[takeoff])
    term = _height_term(env)
    term(env, command_name=_CMD, **params)
    term(env.step(found=_AIRBORNE), command_name=_CMD, **params)
    return float(
        term(
            env.step(found=found_at_apex, z=[takeoff + rise_m]),
            command_name=_CMD,
            **params,
        )[0]
    )


def test_body_height_peaks_at_the_target_rise():
    assert abs(_rise_reward(0.040, target_rise=0.040, std=0.020) - 1.0) < 1e-5


def test_body_height_falls_off_away_from_the_target_rise():
    assert _rise_reward(0.005, target_rise=0.040, std=0.008) < 0.01


def test_body_height_is_gated_by_the_launch_phase():
    assert _rise_reward(0.040, cmd=_RECOVERY, target_rise=0.040, std=0.020) == 0.0


def test_body_height_pays_nothing_for_a_tall_tuck():
    """The companion to the airborne term's tuck test, and the one that matters
    more: `hop_body_height` is a GAUSSIAN, so it has no threshold to hide behind.
    Under absolute height a tall tuck at z = 0.155 sat only 32 mm below the old
    0.1871 target and collected exp(-(0.032/0.020)^2) = 0.077 -- 0.61/step at
    weight 8.0 for doing nothing. Measured as rise, a tuck is 0 m and scores
    exp(-4) = 0.018, the same as any other non-hop."""
    env = _Env(found=_PLANTED, cmd=_LAUNCH, z=[0.155])
    term = _height_term(env)
    term(env, command_name=_CMD, target_rise=0.040, std=0.020)
    term(env.step(found=_AIRBORNE), command_name=_CMD, target_rise=0.040, std=0.020)
    tuck = float(term(env, command_name=_CMD, target_rise=0.040, std=0.020)[0])
    assert tuck < 0.02, f"a tall tuck scores {tuck:.4f}"
    # Explicitly: the tuck is ABSOLUTELY higher than the nominal stance and
    # still scores the floor, because it did not rise.
    assert 0.155 > 0.147


def test_body_height_rise_is_measured_from_takeoff_not_a_fixed_datum():
    """Identical rise from two different takeoff heights scores identically.
    This is the property that removes the standing-height datum -- and with it
    h_add, actuator sag and posture headroom -- from the reward entirely."""
    low = _rise_reward(0.033, takeoff=0.120, target_rise=0.040, std=0.020)
    high = _rise_reward(0.033, takeoff=0.160, target_rise=0.040, std=0.020)
    assert abs(low - high) < 1e-6
    assert low > 0.8


def test_body_height_is_zero_with_both_feet_in_contact():
    """The airborne gate. HOME_FRAME is a parallelogram crouch, so simply
    STRAIGHTENING THE LEGS raises the trunk ~9 mm with both feet still planted.
    Ungated, that ground-level bob collects a large share of the peak reward and
    is entirely spring-irrelevant -- exactly the confound this experiment exists
    to avoid. Same rise as the passing case; only the contact differs."""
    assert _rise_reward(
        0.040, found_at_apex=_PLANTED, target_rise=0.040, std=0.020
    ) == 0.0


def test_body_height_is_zero_with_only_one_foot_airborne():
    """A single-foot lift is a step, not a hop."""
    assert _rise_reward(
        0.040, found_at_apex=[[0.0, 1.0]], target_rise=0.040, std=0.020
    ) == 0.0


def test_body_height_discriminates_locked_from_sprung_rise():
    """The property the whole campaign rests on: the drop rig rebounded 5 mm on
    the Locked arm and 33 mm at k=3900, and the reward must tell them apart."""
    locked = _rise_reward(0.005, target_rise=0.040, std=0.020)
    sprung = _rise_reward(0.033, target_rise=0.040, std=0.020)
    assert sprung > locked
    assert sprung / locked >= 10.0


def test_body_height_treats_a_nan_contact_read_as_in_contact():
    """Never pay for flight we cannot actually see."""
    nan = [[float("nan"), float("nan")]]
    assert _rise_reward(0.040, found_at_apex=nan, target_rise=0.040, std=0.020) == 0.0


def test_body_height_is_zero_without_the_contact_sensor():
    env = _Env(found=_AIRBORNE, cmd=_LAUNCH, z=[0.147])
    term = _height_term(env)
    env.scene.sensors = {}
    out = term(env, command_name=_CMD, target_rise=0.040)
    assert float(out[0]) == 0.0


def test_body_height_and_airborne_reward_read_the_same_contact_predicate():
    """Both terms must agree about what counts as flight, or the policy can be
    paid an apex the other term does not call a hop. Sweep every contact
    combination at a rise well above MIN_RISE and require agreement."""
    for found in (_AIRBORNE, [[1.0, 0.0]], [[0.0, 1.0]], _PLANTED):
        env = _Env(found=_PLANTED, cmd=_LAUNCH, z=[0.147])
        a_term, h_term = _airborne_term(env), _height_term(env)
        a_term(env, sensor_name=_SENSOR, command_name=_CMD)
        h_term(env, command_name=_CMD, target_rise=0.040, std=0.020)
        env.step(found=_AIRBORNE)
        a_term(env, sensor_name=_SENSOR, command_name=_CMD)
        h_term(env, command_name=_CMD, target_rise=0.040, std=0.020)
        env.step(found=found, z=[0.180])
        a = float(a_term(env, sensor_name=_SENSOR, command_name=_CMD)[0])
        h = float(h_term(env, command_name=_CMD, target_rise=0.040, std=0.020)[0])
        assert (a > 0.0) == (h > 0.0), found


def test_both_stateful_terms_are_nan_safe():
    env = _Env(found=_AIRBORNE, cmd=_LAUNCH, z=[float("nan")])
    a_term, h_term = _airborne_term(env), _height_term(env)
    for out in (
        a_term(env, sensor_name=_SENSOR, command_name=_CMD),
        h_term(env, command_name=_CMD, target_rise=0.040),
        hop_upward_velocity(env, command_name=_CMD),
    ):
        assert torch.isfinite(out).all()


# --- hop_energy_monitor ------------------------------------------------------

_JOINTS = ("passive_left_foot_spring", "passive_right_foot_spring")
_K = 3900.0
_PRELOAD = 0.00074


class _JointData:
    def __init__(self, q):
        self.joint_pos = torch.tensor(q, dtype=torch.float32)


class _JointAsset:
    def __init__(self, q):
        self.data = _JointData(q)

    def find_joints(self, name):
        return [_JOINTS.index(name)], None


class _JointScene:
    def __init__(self, q):
        self._a = _JointAsset(q)

    def __getitem__(self, _k):
        return self._a


class _JointEnv:
    def __init__(self, q):
        self.scene = _JointScene(q)
        self.num_envs = len(q)
        self.device = "cpu"
        self.extras = {"log": {}}


def test_energy_monitor_returns_exactly_zeros():
    env = _JointEnv([[0.005, 0.005]])
    out = hop_energy_monitor(env, joint_names=_JOINTS, stiffness=_K, preload=_PRELOAD)
    assert out.shape == (1,)
    assert float(out[0]) == 0.0


def test_energy_matches_the_closed_form():
    """E = 0.5*k*q^2 + k*preload*q per foot, summed over both."""
    q = 0.006
    env = _JointEnv([[q, q]])
    hop_energy_monitor(env, joint_names=_JOINTS, stiffness=_K, preload=_PRELOAD)
    expected = 2 * (0.5 * _K * q**2 + _K * _PRELOAD * q)
    got = float(env.extras["log"]["Metrics/hop_spring_energy_mean"])
    # 1e-8, not 1e-9: q is stored as float32 in the fixture (matching real
    # joint_pos), so q=0.006 is already quantized to ~6.0000000522e-3 before
    # any arithmetic runs. That alone puts a ~2.7e-9 floor under this
    # comparison against the float64 closed-form -- confirmed by reproducing
    # the closed-form in pure float64 math off the quantized value.
    assert abs(got - expected) < 1e-8


def test_energy_is_zero_at_rest():
    env = _JointEnv([[0.0, 0.0]])
    hop_energy_monitor(env, joint_names=_JOINTS, stiffness=_K, preload=_PRELOAD)
    assert float(env.extras["log"]["Metrics/hop_spring_energy_mean"]) == 0.0


def test_negative_q_contributes_no_energy():
    """Preload holds the pad past its lower limit when unloaded (measured
    -0.59 mm). That is limit penetration, not stored energy."""
    env = _JointEnv([[-0.00059, -0.00059]])
    hop_energy_monitor(env, joint_names=_JOINTS, stiffness=_K, preload=_PRELOAD)
    assert float(env.extras["log"]["Metrics/hop_spring_energy_mean"]) == 0.0


def test_peak_exceeds_mean_when_feet_differ():
    env = _JointEnv([[0.002, 0.002], [0.010, 0.010]])
    hop_energy_monitor(env, joint_names=_JOINTS, stiffness=_K, preload=_PRELOAD)
    log = env.extras["log"]
    assert float(log["Metrics/hop_spring_energy_peak"]) > float(
        log["Metrics/hop_spring_energy_mean"]
    )


def test_energy_monitor_survives_missing_joints():
    """The Locked control arm has NO spring joints; mjlab's find_joints RAISES
    on no match rather than returning empty."""
    class _RaisingAsset(_JointAsset):
        def find_joints(self, name):
            raise ValueError("Not all regular expressions are matched!")

    env = _JointEnv([[0.005, 0.005]])
    env.scene._a = _RaisingAsset([[0.005, 0.005]])
    out = hop_energy_monitor(env, joint_names=_JOINTS, stiffness=_K, preload=_PRELOAD)
    assert float(out[0]) == 0.0
    assert float(env.extras["log"]["Metrics/hop_spring_energy_mean"]) == 0.0


# --- hop_load_force ----------------------------------------------------------
#
# The load half was entirely unrewarded: all three terms above gate on
# sin(2*pi*phi) > 0. Without a load-phase signal there is no actuator
# countermovement, and without a countermovement the spring cannot be charged
# (static sag under body weight alone is 0.48 mm at k=3900, ~0.45 mJ, worth
# 0.1 mm of lift) -- so the spring needs a hop to charge and the hop needs a
# charged spring.

_BW = 8.60  # N -- 0.877 kg * 9.81, matching hop.BODY_WEIGHT_N.


def _force(total_n, feet=2):
    """One env, `total_n` newtons of vertical GRF split evenly over `feet` feet.

    Negative z: MuJoCo's reduce="netforce" contact force points DOWN for a foot
    bearing load (probed at -4.905 N under a 4.905 N weight).
    """
    per = -total_n / feet
    return [[[0.0, 0.0, per] for _ in range(feet)]]


def _phase(phi):
    """The phase command the env actually emits: [cos(2*pi*phi), sin(2*pi*phi), 0]."""
    return [[math.cos(2 * math.pi * phi), math.sin(2 * math.pi * phi), 0.0]]


# `hop_load_force` gates on the COSINE channel, clamp(cos(2*pi*phi), 0), which
# spans phi in [0.75, 1.0) u [0.0, 0.25] -- the countermovement-into-launch
# window bracketing takeoff at phi = 0. NOT the sin < 0 "load half", which peaks
# at phi = 0.75, a quarter cycle before the launch gate peaks at phi = 0.25.

_TAKEOFF = 0.0   # cos = +1, gate wide open
_MID_LAUNCH = 0.25   # cos = 0, gate shut (launch gate's own peak)
_MID_FLIGHT = 0.5    # cos = -1, gate shut
_SINE_LOAD_PEAK = 0.75   # cos = 0, gate shut -- the OLD gate's peak


def test_load_force_pays_at_takeoff():
    """phi = 0 is the countermovement-into-launch instant: press here."""
    env = _Env(cmd=_phase(_TAKEOFF), force=_force(3.5 * _BW))
    out = hop_load_force(
        env, sensor_name=_SENSOR, command_name=_CMD, body_weight_n=_BW, max_ratio=6.0
    )
    # (3.5 - 1) / (6.0 - 1) = 0.5, times a gate of cos(0) = 1.0.
    assert abs(float(out[0]) - 0.5) < 1e-5


def test_load_force_pays_nothing_at_mid_launch():
    """phi = 0.25 is peak flight intent. Pressing into the ground there is the
    opposite of what we want."""
    env = _Env(cmd=_phase(_MID_LAUNCH), force=_force(10 * _BW))
    out = hop_load_force(env, sensor_name=_SENSOR, command_name=_CMD, body_weight_n=_BW)
    assert abs(float(out[0])) < 1e-6


def test_load_force_pays_nothing_at_the_old_sine_gate_peak():
    """THE RETIMING. `clamp(-sin(2*pi*phi), 0)` -- the literal "load half" --
    peaks at phi = 0.75, a QUARTER CYCLE BEFORE the launch gate peaks at 0.25. A
    correctly timed countermovement presses over phi ~ 0.07-0.17, immediately
    before takeoff, which sits in the LAUNCH half where a sine gate reads zero;
    and while the robot is genuinely crouching, GRF drops BELOW body weight, so a
    sine gate reads zero there too. What it actually paid for was a second,
    hop-irrelevant press centred on phi = 0.75: a stamp, or a 2 Hz
    double-bounce. This test is the one that fails if the gate is reverted."""
    env = _Env(cmd=_phase(_SINE_LOAD_PEAK), force=_force(10 * _BW))
    out = hop_load_force(env, sensor_name=_SENSOR, command_name=_CMD, body_weight_n=_BW)
    assert abs(float(out[0])) < 1e-6


def test_load_force_pays_nothing_at_mid_flight():
    env = _Env(cmd=_phase(_MID_FLIGHT), force=_force(10 * _BW))
    out = hop_load_force(env, sensor_name=_SENSOR, command_name=_CMD, body_weight_n=_BW)
    assert float(out[0]) == 0.0


def test_load_force_gate_brackets_takeoff_symmetrically():
    """The window is the quarter cycle either side of phi = 0: the tail of the
    load half AND the head of the launch half. Both must pay, equally."""
    f = dict(sensor_name=_SENSOR, command_name=_CMD, body_weight_n=_BW, max_ratio=6.0)
    before = float(hop_load_force(_Env(cmd=_phase(0.875), force=_force(6 * _BW)), **f)[0])
    after = float(hop_load_force(_Env(cmd=_phase(0.125), force=_force(6 * _BW)), **f)[0])
    assert before > 0.0
    assert abs(before - after) < 1e-5
    # cos(2*pi*0.125) = 0.7071, times a saturated load factor of 1.0.
    assert abs(before - math.cos(2 * math.pi * 0.125)) < 1e-5


def test_load_force_is_zero_at_exactly_body_weight():
    """Merely standing must earn nothing -- standing still is the failure mode
    this whole rebalance exists to defeat."""
    env = _Env(cmd=_phase(_TAKEOFF), force=_force(_BW))
    out = hop_load_force(env, sensor_name=_SENSOR, command_name=_CMD, body_weight_n=_BW)
    assert abs(float(out[0])) < 1e-6


def test_load_force_is_zero_when_unloaded():
    """Airborne, or hanging: below body weight is still nothing, not negative.
    This is also why the launch-half quarter of the gate window costs nothing --
    in flight there is no ground reaction to reward."""
    env = _Env(cmd=_phase(_TAKEOFF), force=_force(0.0))
    out = hop_load_force(env, sensor_name=_SENSOR, command_name=_CMD, body_weight_n=_BW)
    assert float(out[0]) == 0.0


def test_load_force_saturates_at_max_ratio_body_weight():
    env = _Env(cmd=_phase(_TAKEOFF), force=_force(6.0 * _BW))
    out = hop_load_force(
        env, sensor_name=_SENSOR, command_name=_CMD, body_weight_n=_BW, max_ratio=6.0
    )
    assert abs(float(out[0]) - 1.0) < 1e-5


def test_load_force_does_not_exceed_one_above_saturation():
    env = _Env(cmd=_phase(_TAKEOFF), force=_force(20.0 * _BW))
    out = hop_load_force(
        env, sensor_name=_SENSOR, command_name=_CMD, body_weight_n=_BW, max_ratio=6.0
    )
    assert abs(float(out[0]) - 1.0) < 1e-5


def test_load_force_has_live_gradient_across_the_springs_working_range():
    """Why max_ratio is 6.0 and not 2.0. At k = 3900 N/m each mm of pad travel
    costs 3.9 N, so saturating at 2 x body weight (8.6 N/foot) meant saturating
    at 2.2 mm of the pad's 12 mm travel -- zero gradient across ~82% of the
    range, which is precisely the range that stores energy and precisely where
    the three arms differ."""
    f = dict(sensor_name=_SENSOR, command_name=_CMD, body_weight_n=_BW, max_ratio=6.0)

    def at_travel_mm(mm):
        newtons_per_foot = 3900.0 * mm / 1000.0
        env = _Env(cmd=_phase(_TAKEOFF), force=_force(2 * newtons_per_foot))
        return float(hop_load_force(env, **f)[0])

    samples = [at_travel_mm(mm) for mm in (2, 3, 4, 5, 6)]
    assert samples == sorted(samples)
    assert samples[0] < samples[-1], "no gradient across 2-6 mm of travel"
    # Still short of saturation at half travel, so the top of the range is live.
    assert at_travel_mm(6) < 1.0


def test_load_force_sums_over_both_feet():
    """A one-footed press is not a two-footed press; the term reads TOTAL
    vertical GRF, so both feet contribute."""
    f = dict(sensor_name=_SENSOR, command_name=_CMD, body_weight_n=_BW, max_ratio=6.0)
    both = _Env(cmd=_phase(_TAKEOFF), force=[[[0.0, 0.0, -3 * _BW], [0.0, 0.0, -3 * _BW]]])
    one = _Env(cmd=_phase(_TAKEOFF), force=[[[0.0, 0.0, -3 * _BW], [0.0, 0.0, 0.0]]])
    assert abs(float(hop_load_force(both, **f)[0]) - 1.0) < 1e-5
    assert abs(float(hop_load_force(one, **f)[0]) - 0.4) < 1e-5


def test_load_force_ignores_horizontal_shear():
    """Vertical component, not the 3-vector norm. mjlab's `soft_landing` (which
    logs Metrics/landing_force_mean off this same field) takes the norm, which is
    right for an impact magnitude but wrong here: a foot scrubbing sideways would
    otherwise score as loading."""
    env = _Env(
        cmd=_phase(_TAKEOFF), force=[[[50.0, 50.0, -_BW / 2], [0.0, 0.0, -_BW / 2]]]
    )
    out = hop_load_force(env, sensor_name=_SENSOR, command_name=_CMD, body_weight_n=_BW)
    assert abs(float(out[0])) < 1e-6


def test_load_force_is_indifferent_to_the_sign_of_fz():
    """PIN THE SIGN CONVENTION, which |Fz| otherwise hides.

    Probed on this exact sensor shape (0.5 kg box settled on a plane,
    primary=geom / secondary=body, reduce="netforce"): a foot BEARING 4.905 N
    reads z = -4.905 N. MuJoCo reports this net force pointing DOWN for a loaded
    foot. Raw `forces[..., 2]` would therefore be negative through the entire
    load phase and the reward would read identically zero -- a silent null.

    Because the implementation takes |Fz|, no other test can tell the two
    conventions apart, so a future edit that switched to raw `forces[..., 2]`
    AND flipped these fixtures to positive z would pass the whole suite green
    while being that silent null. Assert the symmetry directly instead: the
    reward must be IDENTICAL for +Fz and -Fz, and non-zero for both."""
    f = dict(sensor_name=_SENSOR, command_name=_CMD, body_weight_n=_BW, max_ratio=6.0)
    negative = _Env(cmd=_phase(_TAKEOFF), force=[[[0.0, 0.0, -2 * _BW], [0.0, 0.0, -2 * _BW]]])
    positive = _Env(cmd=_phase(_TAKEOFF), force=[[[0.0, 0.0, 2 * _BW], [0.0, 0.0, 2 * _BW]]])
    neg = float(hop_load_force(negative, **f)[0])
    pos = float(hop_load_force(positive, **f)[0])
    assert neg > 0.0, "the measured convention (negative z under load) must pay"
    assert abs(neg - pos) < 1e-6, "the term must not depend on MuJoCo's sign convention"


def test_load_force_is_nan_safe():
    env = _Env(cmd=_phase(_TAKEOFF), force=_force(float("nan")))
    out = hop_load_force(env, sensor_name=_SENSOR, command_name=_CMD, body_weight_n=_BW)
    assert torch.isfinite(out).all()
    assert float(out[0]) == 0.0

    nan_cmd = _Env(cmd=[[float("nan"), 0.0, 0.0]], force=_force(5 * _BW))
    out = hop_load_force(nan_cmd, sensor_name=_SENSOR, command_name=_CMD, body_weight_n=_BW)
    assert torch.isfinite(out).all()


def test_load_force_is_zero_without_the_contact_sensor():
    env = _Env(cmd=_phase(_TAKEOFF), force=_force(5 * _BW))
    env.scene.sensors = {}
    out = hop_load_force(env, sensor_name=_SENSOR, command_name=_CMD, body_weight_n=_BW)
    assert out.shape == (1,)
    assert float(out[0]) == 0.0


def test_load_force_is_zero_when_the_sensor_carries_no_force_field():
    """A sensor configured without fields=("force",) reports None. Fail silent
    rather than crash a 8000-iteration run, but never invent a load."""
    env = _Env(cmd=_phase(_TAKEOFF), force=None)
    out = hop_load_force(env, sensor_name=_SENSOR, command_name=_CMD, body_weight_n=_BW)
    assert float(out[0]) == 0.0


# --- com_height_target_recovery_only -----------------------------------------


def test_com_height_recovery_only_is_zero_during_the_launch_half():
    """The gate. `com_height_target` pays a flat +1 anywhere in band (x1.2), the
    single largest reward for standing perfectly still. During launch we want the
    robot LEAVING the band, so the term must say nothing there."""
    env = _Env(cmd=[[0.0, 1.0, 0.0]], z=[0.15])
    out = com_height_target_recovery_only(
        env, command_name=_CMD, target_height_min=0.14, target_height_max=0.20
    )
    assert float(out[0]) == 0.0


def test_com_height_recovery_only_pays_in_band_during_recovery():
    env = _Env(cmd=[[0.0, -1.0, 0.0]], z=[0.15])
    out = com_height_target_recovery_only(
        env, command_name=_CMD, target_height_min=0.14, target_height_max=0.20
    )
    assert abs(float(out[0]) - 1.0) < 1e-6


def test_com_height_recovery_only_scales_with_the_gate():
    """Partway through the recovery half the gate is |sin| < 1, not 0 or 1."""
    env = _Env(cmd=[[0.0, -0.5, 0.0]], z=[0.15])
    out = com_height_target_recovery_only(
        env, command_name=_CMD, target_height_min=0.14, target_height_max=0.20
    )
    assert abs(float(out[0]) - 0.5) < 1e-6


def test_com_height_recovery_only_matches_the_wrapped_term_at_full_recovery():
    """It must be the EXISTING reward, gated -- not a reimplementation of it.
    Checked out of band too, where the term is a negative quadratic."""
    from mjlab_microduck.tasks.mdp import com_height_target

    for z in (0.10, 0.15, 0.30):
        env = _Env(cmd=[[0.0, -1.0, 0.0]], z=[z])
        gated = float(
            com_height_target_recovery_only(
                env, command_name=_CMD, target_height_min=0.14, target_height_max=0.20
            )[0]
        )
        plain = float(
            com_height_target(env, target_height_min=0.14, target_height_max=0.20)[0]
        )
        assert abs(gated - plain) < 1e-6, z


def test_com_height_recovery_only_is_nan_safe():
    env = _Env(cmd=[[0.0, float("nan"), 0.0]], z=[float("nan")])
    out = com_height_target_recovery_only(
        env, command_name=_CMD, target_height_min=0.14, target_height_max=0.20
    )
    assert torch.isfinite(out).all()


# ── hop_symmetric_push ───────────────────────────────────────────────────────
#
# The term exists to stop the SKIP: run 59yiy9h6 leaned ~5.9 deg onto one foot
# and launched off it. These tests pin the two properties that make it work --
# it saturates on an even two-footed push, and it cannot be farmed by the three
# cheats a difference penalty would have allowed.


def _split_force(left_n, right_n):
    """One env, per-foot vertical GRF. Negative z, as MuJoCo reports a loaded foot."""
    return [[[0.0, 0.0, -left_n], [0.0, 0.0, -right_n]]]


def test_symmetric_push_saturates_when_both_feet_press_body_weight():
    """An even push scores 1.0. Magnitude is hop_load_force's job."""
    env = _Env(cmd=_phase(_TAKEOFF), force=_split_force(_BW, _BW))
    out = hop_symmetric_push(env, sensor_name=_SENSOR, command_name=_CMD,
                             body_weight_n=_BW)
    assert abs(float(out[0]) - 1.0) < 1e-5


def test_symmetric_push_scores_a_lopsided_push_at_zero():
    """THE WHOLE POINT. A skip loads one foot hard and the other barely; an
    average would score it well.

    Same 2.0 x body weight total as the saturating case, split 9:1. The weaker
    foot carries 0.2 x body weight, under the 0.25 load floor, so it does not
    count as pushing at all.
    """
    env = _Env(cmd=_phase(_TAKEOFF), force=_split_force(1.8 * _BW, 0.2 * _BW))
    out = hop_symmetric_push(env, sensor_name=_SENSOR, command_name=_CMD,
                             body_weight_n=_BW)
    assert abs(float(out[0])) < 1e-6


def test_symmetric_push_scores_the_ratio_not_the_magnitude():
    """It measures EVENNESS; `hop_load_force` measures magnitude.

    A soft even push and a hard even push score the same, because scoring both
    would double-pay for magnitude -- and the first version's conflation of the
    two is what let the term dilute itself to 8.8% of its ceiling on u2ck51xg.
    """
    soft = _Env(cmd=_phase(_TAKEOFF), force=_split_force(0.4 * _BW, 0.4 * _BW))
    hard = _Env(cmd=_phase(_TAKEOFF), force=_split_force(4.0 * _BW, 4.0 * _BW))
    kw = dict(sensor_name=_SENSOR, command_name=_CMD, body_weight_n=_BW)
    assert abs(float(hop_symmetric_push(soft, **kw)[0]) - 1.0) < 1e-5
    assert abs(float(hop_symmetric_push(hard, **kw)[0]) - 1.0) < 1e-5


def test_symmetric_push_grades_partial_asymmetry():
    """Between the extremes it must be graded, not a step: a 2:1 split is worse
    than 1:1 and better than 4:1, so the policy has a gradient to climb."""
    kw = dict(sensor_name=_SENSOR, command_name=_CMD, body_weight_n=_BW)
    even = float(hop_symmetric_push(
        _Env(cmd=_phase(_TAKEOFF), force=_split_force(_BW, _BW)), **kw)[0])
    two_to_one = float(hop_symmetric_push(
        _Env(cmd=_phase(_TAKEOFF), force=_split_force(2 * _BW, _BW)), **kw)[0])
    four_to_one = float(hop_symmetric_push(
        _Env(cmd=_phase(_TAKEOFF), force=_split_force(4 * _BW, _BW)), **kw)[0])
    assert abs(even - 1.0) < 1e-5
    assert abs(two_to_one - 0.5) < 1e-5
    assert abs(four_to_one - 0.25) < 1e-5


def test_symmetric_push_load_floor_rejects_two_feet_merely_resting():
    """Two feet resting together are a PERFECT ratio and no push at all. The
    floor is the only thing standing between the ratio form and that exploit."""
    env = _Env(cmd=_phase(_TAKEOFF), force=_split_force(0.1 * _BW, 0.1 * _BW))
    out = hop_symmetric_push(env, sensor_name=_SENSOR, command_name=_CMD,
                             body_weight_n=_BW)
    assert abs(float(out[0])) < 1e-6


def test_symmetric_push_pays_nothing_for_a_one_legged_launch():
    env = _Env(cmd=_phase(_TAKEOFF), force=_split_force(3.0 * _BW, 0.0))
    out = hop_symmetric_push(env, sensor_name=_SENSOR, command_name=_CMD,
                             body_weight_n=_BW)
    assert abs(float(out[0])) < 1e-6


def test_symmetric_push_cannot_be_farmed_by_unloading_both_feet():
    """The cheat that kills a `-|f_left - f_right|` penalty: two feet carrying
    NOTHING are perfectly symmetric, so a difference penalty scores them
    perfectly. `min` scores them zero, which is why the term is a min."""
    env = _Env(cmd=_phase(_TAKEOFF), force=_split_force(0.0, 0.0))
    out = hop_symmetric_push(env, sensor_name=_SENSOR, command_name=_CMD,
                             body_weight_n=_BW)
    assert abs(float(out[0])) < 1e-6


def test_symmetric_push_cannot_be_farmed_by_tucking_in_the_window():
    """The same cheat with the feet merely light rather than exactly zero: a
    tuck during the gate window must not pay."""
    env = _Env(cmd=_phase(_TAKEOFF), force=_split_force(0.01 * _BW, 0.01 * _BW))
    out = hop_symmetric_push(env, sensor_name=_SENSOR, command_name=_CMD,
                             body_weight_n=_BW)
    assert float(out[0]) < 0.02


def test_symmetric_push_shares_hop_load_forces_gate():
    """Gated on the cosine channel, so it is shut at mid-launch and mid-flight
    and open at takeoff -- the same window hop_load_force uses."""
    for phi in (_MID_LAUNCH, _MID_FLIGHT, _SINE_LOAD_PEAK):
        env = _Env(cmd=_phase(phi), force=_split_force(_BW, _BW))
        out = hop_symmetric_push(env, sensor_name=_SENSOR, command_name=_CMD,
                                 body_weight_n=_BW)
        assert abs(float(out[0])) < 1e-6, f"gate should be shut at phi={phi}"


def test_symmetric_push_does_not_exceed_one_when_both_feet_slam():
    env = _Env(cmd=_phase(_TAKEOFF), force=_split_force(9 * _BW, 9 * _BW))
    out = hop_symmetric_push(env, sensor_name=_SENSOR, command_name=_CMD,
                             body_weight_n=_BW)
    assert abs(float(out[0]) - 1.0) < 1e-6

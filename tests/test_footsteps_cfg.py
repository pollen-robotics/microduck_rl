"""Footstep contract + Mjlab-Footsteps cfg invariants (CPU, no env build)."""

import math

import pytest

from mjlab_microduck import footsteps as F
from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_footsteps_env_cfg import COMMAND_NAME, make_microduck_footsteps_env_cfg


@pytest.mark.parametrize("left", [False, True])
def test_neutral_support_roundtrip(left):
    step = (0.03, -0.01, 0.2)
    back = F.to_neutral_frame(F.to_support_frame(step, left), left)
    assert all(math.isclose(a, b, abs_tol=1e-12) for a, b in zip(step, back))


def test_neutral_step_is_parallel_feet_at_spacing():
    assert F.to_support_frame((0, 0, 0), support_is_left=False) == (0, F.FEET_SPACING, 0)  # left lands at +y
    assert F.to_support_frame((0, 0, 0), support_is_left=True) == (0, -F.FEET_SPACING, 0)


def test_planner_limits_unset_until_measured():
    # FootstepNet limits come from the TRAINED footstep policy's envelope, never from another policy.
    with pytest.raises(RuntimeError):
        F.footstepnet_options()


def test_production_anchor_set():
    left, right = F.odom_anchor_points()
    assert F.ODOM_ANCHOR_SET == "alpha16" and len(left) == len(right) == 16


@pytest.fixture(scope="module")
def cfg():
    return make_microduck_footsteps_env_cfg()


def test_command_replaces_twist_in_obs_order(cfg):
    assert list(cfg.commands)[0] == COMMAND_NAME and "twist" not in cfg.commands
    actor = list(cfg.observations["actor"].terms.items())
    names = [k for k, _ in actor]
    assert names[-3:] == ["command", "head_command", "body_command"]  # 61D layout: twist slot = footstep target
    assert actor[-3][1].params["command_name"] == COMMAND_NAME


def test_no_twist_readers_left(cfg):
    for terms in (cfg.rewards, *(g.terms for g in cfg.observations.values()), cfg.curriculum, cfg.events):
        for name, term in terms.items():
            assert term.params.get("command_name") != "twist", name


def test_velocity_tracking_removed_and_footstep_rewards_positive(cfg):
    assert "track_linear_velocity" not in cfg.rewards and "track_angular_velocity" not in cfg.rewards
    # both return >= 0 (reward / potential shaping) → positive weights
    assert cfg.rewards["footstep_placement"].func is mdp.footstep_placement
    assert cfg.rewards["footstep_placement"].weight > 0
    assert cfg.rewards["footstep_swing_progress"].weight > 0


def test_target_range_widens_and_contains_neutral(cfg):
    c = cfg.commands[COMMAND_NAME]
    for (lo0, hi0), (lo1, hi1) in zip(c.range_start, c.range_end):
        assert lo1 <= lo0 < 0 < hi0 <= hi1
    assert c.ramp_steps[0] < c.ramp_steps[1]


def test_play_uses_full_range():
    c = make_microduck_footsteps_env_cfg(play=True).commands[COMMAND_NAME]
    assert c.ramp_steps == (0, 1)


def test_paid_steps_need_a_real_swing(cfg):
    # run 1 farmed odometry switches from rocking (13/s, feet never left the ground): placement pays only when the
    # landing foot was airborne (contact sensor in the scene) and the step was not faster than the nominal cadence
    c = cfg.commands[COMMAND_NAME]
    assert c.contact_sensor_name in [s.name for s in cfg.scene.sensors]
    assert 0 < c.min_air < c.nominal_step_time and 0 < c.min_step_time < c.nominal_step_time

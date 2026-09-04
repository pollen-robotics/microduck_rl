"""The critic obs must survive a non-finite sensor reading.

Regression for the 2026-08-21 crash: rsl_rl's check_nan killed a
Velocity2-Rough-Backlash run with "observation group 'critic' contains NaN".
`nan_state` (robot_state_is_nan) only covered joint + root state, but the
critic also carries three SENSOR-derived terms (raycast heights, contact
air-time, contact forces). MuJoCo can return a non-finite contact force while
the integrated robot state is still clean, so the env was never reset and the
NaN reached the runner.
"""

import torch

from mjlab_microduck.tasks import mdp as microduck_mdp


class _SensorData:
    def __init__(self, force=None, heights=None):
        self.force = force
        self.heights = heights


class _Sensor:
    def __init__(self, data):
        self.data = data


class _Scene:
    def __init__(self, sensors, asset):
        self.sensors = sensors
        self._asset = asset

    def __getitem__(self, key):
        return self.sensors[key] if key in self.sensors else self._asset


class _AssetData:
    def __init__(self, n):
        self.joint_pos = torch.zeros(n, 4)
        self.joint_vel = torch.zeros(n, 4)
        self.root_link_pos_w = torch.zeros(n, 3)
        self.root_link_quat_w = torch.zeros(n, 4)
        self.root_link_lin_vel_w = torch.zeros(n, 3)
        self.root_link_ang_vel_w = torch.zeros(n, 3)


class _Asset:
    def __init__(self, data):
        self.data = data


class _Env:
    def __init__(self, n, force):
        self.num_envs = n
        self.device = "cpu"
        asset = _Asset(_AssetData(n))
        self.scene = _Scene({"feet": _Sensor(_SensorData(force=force))}, asset)


def _force(n, bad_env=None, value=float("nan")):
    f = torch.ones(n, 2, 3)
    if bad_env is not None:
        f[bad_env, 0, 0] = value
    return f


def test_state_only_check_misses_bad_contact_force():
    # This is the gap that killed the run: robot state is clean, force is not.
    env = _Env(3, _force(3, bad_env=1))
    assert not microduck_mdp.robot_state_is_nan(env).any()


def test_termination_catches_nan_contact_force():
    env = _Env(3, _force(3, bad_env=1))
    out = microduck_mdp.robot_state_is_nan(env, sensor_names=("feet",))
    assert out.tolist() == [False, True, False]


def test_termination_catches_inf_contact_force():
    env = _Env(3, _force(3, bad_env=2, value=float("inf")))
    out = microduck_mdp.robot_state_is_nan(env, sensor_names=("feet",))
    assert out.tolist() == [False, False, True]


def test_termination_ignores_missing_sensor():
    env = _Env(2, _force(2))
    assert not microduck_mdp.robot_state_is_nan(env, sensor_names=("nope",)).any()


def test_finite_helper_sanitizes_nan_and_inf():
    x = torch.tensor([[1.0, float("nan"), float("inf"), float("-inf")]])
    out = microduck_mdp._finite(x)
    assert torch.isfinite(out).all()
    assert out[0, 0] == 1.0


def test_safe_obs_wrappers_are_wired_into_the_critic():
    # Guards must actually be installed on the env cfg, not just exist.
    from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
        make_microduck_velocity_env_cfg,
    )

    cfg = make_microduck_velocity_env_cfg(rough=True)
    terms = cfg.observations["critic"].terms
    for name in ("foot_contact_forces", "foot_height", "foot_air_time"):
        assert terms[name].func.__name__.endswith("_safe"), (
            f"critic/{name} lost its NaN guard"
        )


def test_nan_state_termination_watches_the_contact_sensor():
    from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
        make_microduck_velocity_env_cfg,
    )

    cfg = make_microduck_velocity_env_cfg(rough=True)
    params = cfg.terminations["nan_state"].params
    assert params.get("sensor_names"), "nan_state no longer watches contact forces"


def test_standup_env_is_also_guarded():
    # The deployed standing policy trains on StandUp, which builds on mjlab's
    # base env (NOT the microduck velocity env) and therefore does not inherit
    # the guards wired there.
    from mjlab_microduck.tasks.microduck_standup_env_cfg import (
        make_microduck_standup_env_cfg,
    )

    cfg = make_microduck_standup_env_cfg()
    terms = cfg.observations["critic"].terms
    for name in ("foot_contact_forces", "foot_air_time"):
        assert terms[name].func.__name__.endswith("_safe"), (
            f"standup critic/{name} lost its NaN guard"
        )
    assert cfg.terminations["nan_state"].params.get("sensor_names"), (
        "standup nan_state no longer watches contact forces"
    )

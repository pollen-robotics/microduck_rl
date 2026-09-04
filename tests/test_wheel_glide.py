"""wheel_glide_reward : récompense le ROULEMENT des roues vers l'avant (glisse
par gravité), plafonné à cap_speed, nul si les roues reculent, NaN-safe.
Indépendant de toute commande (la tâche pente a une commande nulle).
"""

import re

import torch

from mjlab_microduck.tasks.mdp import wheel_glide_reward

# Current model joint names (post 2026-07 re-export: underscore spelling).
_WHEELS = {"passive_LF_wheel": 0, "passive_LR_wheel": 1, "passive_RF_wheel": 2, "passive_RR_wheel": 3}


class _Data:
    def __init__(self, omegas):
        # 4 roues, colonnes 0..3 dans l'ordre LF,LR,RF,RR
        self.joint_vel = torch.tensor([omegas], dtype=torch.float32)


class _Asset:
    def __init__(self, data):
        self.data = data

    def find_joints(self, pattern):
        # Regex resolution like the real Entity.find_joints (mdp queries use
        # spelling-tolerant patterns such as "passive_LF_?wheel").
        ids = [i for name, i in _WHEELS.items() if re.fullmatch(pattern, name)]
        assert ids, pattern
        return ids, None


class _Env:
    def __init__(self, omegas):
        self._a = _Asset(_Data(omegas))

    def __getitem__(self, _k):
        return self._a

    @property
    def scene(self):
        return self


def test_rewards_forward_roll_below_cap():
    # omega=10 rad/s sur les 4 -> vitesse = 10*0.0175 = 0.175 m/s (< cap 0.35)
    out = wheel_glide_reward(_Env([10.0, 10.0, 10.0, 10.0]), cap_speed=0.35)
    assert abs(float(out[0]) - 0.175) < 1e-6


def test_caps_fast_roll():
    # omega=40 -> 0.7 m/s -> plafonné à 0.35
    out = wheel_glide_reward(_Env([40.0, 40.0, 40.0, 40.0]), cap_speed=0.35)
    assert abs(float(out[0]) - 0.35) < 1e-6


def test_zero_when_wheels_roll_backward():
    out = wheel_glide_reward(_Env([-10.0, -10.0, -10.0, -10.0]), cap_speed=0.35)
    assert float(out[0]) == 0.0


def test_nan_safe():
    out = wheel_glide_reward(_Env([float("nan"), 10.0, 10.0, 10.0]), cap_speed=0.35)
    assert float(out[0]) == 0.0

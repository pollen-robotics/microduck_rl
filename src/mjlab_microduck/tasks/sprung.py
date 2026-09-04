"""Sprung-foot task variant — an idealised 1-DoF compliant foot.

``make_sprung_variant(cfg, stiffness)`` converts a Run-task env cfg into its
sprung counterpart, in the same shape as ``tasks/backlash.py``. Four changes:

1. Swap the robot for a sprung-foot model at the requested stiffness.
2. Shift the ``com_height_target`` band by ``h_add``. The sprung robot stands
   taller, so without this it is penalised for its geometry before compliance
   is in play — and the whole point of the locked control arm is to isolate
   geometry from compliance.
3. Scope the ``pose`` and ``dof_pos_limits`` rewards off the spring joints. A
   passive spring has no pose target and legitimately rides its limits.
4. Register the compression monitor, whose reading decides whether any speed
   number from this variant means anything.

``travel=0.0`` produces the LOCKED control variant: identical geometry and mass,
no compliance.
"""

from copy import deepcopy
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_microduck.robot.sprung_foot import (
    H_ADD,
    PAD_MASS,
    SPRING_JOINTS,
    TRAVEL,
    make_sprung_foot_robot_cfg,
)
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.run import MicroduckRunRlCfg

SPRING_MONITOR_WEIGHT = 1.0

# Excludes the two foot springs while keeping every other joint, including the
# neck/head exclusions the velocity env already applies.
_NO_SPRING = r"^(?!passive_).*"


def make_sprung_variant(
    cfg: ManagerBasedRlEnvCfg,
    stiffness: float,
    travel: float = TRAVEL,
    h_add: float = H_ADD,
    pad_mass: float = PAD_MASS,
) -> ManagerBasedRlEnvCfg:
    """Convert a Run-task env cfg into its sprung-foot counterpart."""
    # 1. Robot.
    cfg.scene.entities = {
        **cfg.scene.entities,
        "robot": make_sprung_foot_robot_cfg(
            stiffness=stiffness, travel=travel, h_add=h_add, pad_mass=pad_mass
        ),
    }

    # 2. The sprung robot stands h_add taller — translate the CoM band, do not
    #    widen it.
    #
    #    ABSENT IS LEGAL NOW, and it is not the same as broken. develop's
    #    4d34d845 ("merge velocity2 into velocity: one walking recipe") stopped
    #    registering `com_height_target`, so the Run-Sprung arms reach here with
    #    no band at all. There is then nothing to translate, and inventing one
    #    would be worse than skipping: a band this transform fabricated would
    #    not be the band those arms' published numbers were measured against.
    #
    #    The HOP arms are unaffected either way -- `make_hop_variant` runs
    #    BEFORE this transform and registers the term itself, precisely because
    #    the hop task does depend on it.
    com = cfg.rewards.get("com_height_target")
    if com is not None:
        com.params["target_height_min"] = com.params["target_height_min"] + h_add
        com.params["target_height_max"] = com.params["target_height_max"] + h_add

    # 3. A passive spring has no pose target, and rides its own limits by
    #    design. Deepcopy first: base templates share SceneEntityCfg objects
    #    across make() calls, so mutating in place would leak into other tasks.
    pose = cfg.rewards.get("pose")
    if pose is not None and "asset_cfg" in pose.params:
        ac = deepcopy(pose.params["asset_cfg"])
        if not any("passive_" in p for p in ac.joint_names):
            ac.joint_names = tuple(ac.joint_names) + (_NO_SPRING,)
        pose.params["asset_cfg"] = ac

    dof_limits = cfg.rewards.get("dof_pos_limits")
    if dof_limits is not None and "asset_cfg" not in dof_limits.params:
        dof_limits.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=(_NO_SPRING,)
        )

    # 4. Compression monitor. Returns zeros, so the weight only has to be
    #    non-zero for RewardManager.compute to call it at all.
    cfg.rewards["spring_compression_monitor"] = RewardTermCfg(
        func=microduck_mdp.spring_compression_monitor,
        weight=SPRING_MONITOR_WEIGHT,
        params={"joint_names": SPRING_JOINTS, "travel": travel},
    )

    return cfg


# STAGE 1 — mass budget. The design-space question is how heavy a boot can be
# before the swing-inertia penalty eats the compliance benefit. k is held at the
# measured prototype spring (3900 N/m, also near the stance-matched optimum) and
# travel at the measured 12 mm, because both are near-decoupled from mass:
# stance-matched k shifts only ~3150->3560 N/m across this whole mass range, and
# travel_min ~ 2.485*t^2 is mass-independent.
#
# The TWO locked arms are what make this a budget rather than a ranking:
#   sprung vs locked at MATCHED mass -> compliance's benefit
#   locked M30 vs locked M90         -> the pure mass penalty
# Where those two curves cross is the mass constraint.
#
# (label, stiffness N/m, travel m, pad_mass kg per boot).
SWEEP_ARMS = (
    ("m30_locked",  3900.0, 0.0,    0.030),
    ("m90_locked",  3900.0, 0.0,    0.090),
    ("m30_k3900",   3900.0, TRAVEL, 0.030),
    ("m50_k3900",   3900.0, TRAVEL, 0.050),
    ("m70_k3900",   3900.0, TRAVEL, 0.070),   # the built prototype
    ("m90_k3900",   3900.0, TRAVEL, 0.090),
)

ARM_TASK_SUFFIX = {
    "m30_locked": "M30-Locked",
    "m90_locked": "M90-Locked",
    "m30_k3900": "M30-K3900",
    "m50_k3900": "M50-K3900",
    "m70_k3900": "M70-K3900",
    "m90_k3900": "M90-K3900",
}


def sprung_rl_cfg(label: str):
    """Per-arm RL cfg: identical learner, distinct logging identity.

    ``replace`` is shallow, so deepcopy the nested cfgs — otherwise every arm
    would share one actor object and a later change to any of them would alter
    all six plus the Run baseline.
    """
    return replace(
        MicroduckRunRlCfg,
        actor=deepcopy(MicroduckRunRlCfg.actor),
        critic=deepcopy(MicroduckRunRlCfg.critic),
        algorithm=deepcopy(MicroduckRunRlCfg.algorithm),
        experiment_name=f"sprung_{label}",
        run_name=f"sprung_{label}",
    )

"""Backlash task variants — swap in the backlash robot model + encoder obs.

``make_backlash_variant(cfg)`` turns any microduck env cfg into its backlash
counterpart (task ids ``Mjlab-<Task>-<Flat|Rough>-Backlash-MicroDuck``):

1. Robot → the matching backlash robot cfg (an unactuated
   ``passive_<joint>_backlash`` hinge in series with each of the 14 servo
   joints, ±1° play) driven by BacklashEncoderBamActuator, whose firmware PD
   closes on the encoder READING THROUGH the backlash — like the real servo,
   whose encoder sits on the output side of the gear play. Pass the robot cfg
   that mirrors the base task's model: MICRODUCK_WALK_BACKLASH_ROBOT_CFG for
   Velocity (robot_walk_backlash.xml), the default
   MICRODUCK_BACKLASH_ROBOT_CFG for VelStand/StandUp
   (robot_groundcontact_backlash.xml).
2. joint_pos / joint_vel obs → joint_pos_rel_backlash / joint_vel_rel_backlash:
   the policy observes qpos[servo] + qpos[backlash] (encoder view), keeping the
   encoder-bias DR path (``biased`` param) intact. Obs and action dims are
   unchanged (still 14 joints), so runtime/export need no changes.
3. dof_pos_limits reward is scoped to the servo joints: backlash joints spend
   their life pinned against their ±1° limits (that is the point of backlash),
   which would otherwise feed a permanent out-of-soft-limit penalty.

Everything else (rewards, DR events, curricula) carries over untouched — the
``passive_`` prefix on the backlash joints means every existing
``^(?!passive_).*`` regex (actuators, pose reward, joint obs selection)
already excludes them.
"""

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab.entity import EntityCfg

from mjlab_microduck.robot.microduck_constants import MICRODUCK_BACKLASH_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp

_SERVO_JOINTS_ONLY = (r"^(?!passive_).*",)


def make_backlash_variant(
    cfg: ManagerBasedRlEnvCfg,
    robot_cfg: EntityCfg = MICRODUCK_BACKLASH_ROBOT_CFG,
) -> ManagerBasedRlEnvCfg:
    """Convert a microduck env cfg (velocity/velstand/standup/...) to backlash."""
    cfg.scene.entities = {**cfg.scene.entities, "robot": robot_cfg}

    for group in ("actor", "critic"):
        terms = cfg.observations[group].terms
        for term_name, func in (
            ("joint_pos", microduck_mdp.joint_pos_rel_backlash),
            ("joint_vel", microduck_mdp.joint_vel_rel_backlash),
        ):
            term = terms.get(term_name)
            if term is None:
                continue
            term.func = func
            # Envs that never narrowed the selection would otherwise feed the
            # backlash joints themselves into the obs (wrong dim + double count).
            if "asset_cfg" not in term.params:
                term.params["asset_cfg"] = SceneEntityCfg(
                    "robot", joint_names=_SERVO_JOINTS_ONLY
                )

    # Backlash joints legitimately ride their hard limits — exclude them from
    # the soft-limit penalty (its default asset_cfg covers every joint).
    dof_limits = cfg.rewards.get("dof_pos_limits")
    if dof_limits is not None and "asset_cfg" not in dof_limits.params:
        dof_limits.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=_SERVO_JOINTS_ONLY
        )

    # The pose (variable_posture) reward resolves its std dicts against the
    # selected joint names and ERRORS on ambiguous matches — on the backlash
    # model "passive_left_hip_yaw_backlash" matches both ".*hip_yaw.*" and the
    # roller envs' ".*passive_.*" std entry. Prepend a backlash exclusion to
    # the selection; existing lookaheads (velocity's passive/neck/head
    # exclusion) compose fine, and envs that keep wheels selected still get
    # them.
    pose = cfg.rewards.get("pose")
    if pose is not None and "asset_cfg" in pose.params:
        # Deepcopy first — base templates share SceneEntityCfg objects across
        # make() calls; mutating in place would leak into the base tasks.
        ac = deepcopy(pose.params["asset_cfg"])
        ac.joint_names = tuple(
            p if "_backlash" in p else r"^(?!passive_.*_backlash)" + p.lstrip("^")
            for p in ac.joint_names
        )
        pose.params["asset_cfg"] = ac

    return cfg

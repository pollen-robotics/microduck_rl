import os
from pathlib import Path

import mujoco
from mjlab.actuator import XmlActuatorCfg
from mjlab_microduck.actuator import (
    BacklashEncoderBamActuatorCfg,
    FrictionDRBamActuatorCfg,
)
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg


_ROBOT_DIR: Path = Path(os.path.dirname(__file__)) / "microduck"

MICRODUCK_WALK_XML: Path = _ROBOT_DIR / "robot_walk.xml"
# Full-collision model, shared by standup / ground-pick / walk-rollers tasks.
MICRODUCK_ALLCOLLISIONS_XML: Path = _ROBOT_DIR / "robot_allcollisions.xml"
# 70mm / 15g ball prop for the BallKick task.
MICRODUCK_BALL_XML: Path = _ROBOT_DIR / "ball.xml"
# Roller-skate model: 14 actuated joints + passive wheel hinges (passive_*wheel).
MICRODUCK_ALLCOLLISIONS_ROLLERS_XML: Path = _ROBOT_DIR / "robot_allcollisions_rollers.xml"
# Backlash models: every servo joint gets an unactuated passive_<joint>_backlash
# hinge in series (±1° play, 2° total). Exported via
# config_mjcf_{allcollisions,walk}_backlash.json (add_backlash.py post-processor).
MICRODUCK_ALLCOLLISIONS_BACKLASH_XML: Path = _ROBOT_DIR / "robot_allcollisions_backlash.xml"
MICRODUCK_WALK_BACKLASH_XML: Path = _ROBOT_DIR / "robot_walk_backlash.xml"
MICRODUCK_ALLCOLLISIONS_ROLLERS_BACKLASH_XML: Path = _ROBOT_DIR / "robot_allcollisions_rollers_backlash.xml"

assert MICRODUCK_WALK_XML.exists(), f"XML not found: {MICRODUCK_WALK_XML}"
assert MICRODUCK_ALLCOLLISIONS_XML.exists(), f"XML not found: {MICRODUCK_ALLCOLLISIONS_XML}"
assert MICRODUCK_BALL_XML.exists(), f"XML not found: {MICRODUCK_BALL_XML}"
assert MICRODUCK_ALLCOLLISIONS_ROLLERS_XML.exists(), f"XML not found: {MICRODUCK_ALLCOLLISIONS_ROLLERS_XML}"
assert MICRODUCK_ALLCOLLISIONS_BACKLASH_XML.exists(), f"XML not found: {MICRODUCK_ALLCOLLISIONS_BACKLASH_XML}"
assert MICRODUCK_WALK_BACKLASH_XML.exists(), f"XML not found: {MICRODUCK_WALK_BACKLASH_XML}"
assert MICRODUCK_ALLCOLLISIONS_ROLLERS_BACKLASH_XML.exists(), f"XML not found: {MICRODUCK_ALLCOLLISIONS_ROLLERS_BACKLASH_XML}"


# Electronics added inside the head, measured 2026-09-03. `jaw_soft` IS the head
# assembly despite the name -- it sits at the end of the neck chain
# (trunk_base -> neck -> neck_pitch -> yaw_roll_motion -> jaw_soft) and at
# 188.8 g it is the heaviest body after the trunk. 15 g there is not negligible:
# the head is far from the CoM, so it has leverage on pitch inertia.
# ── unmodelled trunk mass ────────────────────────────────────────────────────
#
# The whole robot -- battery and spring boots included -- weighed 908 g on a
# scale (measured 2026-09-04) while a 15 g CPU radiator was fitted in the head.
# The radiator has since been REMOVED: it was not cooling much, and it shifted
# far enough to short the electronics. So the robot the model must match is
# 908 - 15 = 893 g.
#
# The residual is unchanged by that removal, because the 15 g was accounted for
# on both sides: the spec's bare XML is 737.2 g, the boots add 2 x 51 g, and
# 893.0 - 737.2 - 102.0 = 53.8 g of real robot is still not in the model.
#
# CONFIRMED ON THE SCALE: with the radiator out the robot weighs 892 g against
# this model's 893.0 g. The 1 g is at the scale's own resolution and is left
# uncorrected deliberately -- trimming the constant to match it would claim a
# precision nothing else here has, and 1 g is two orders below the open
# questions (where the 53.8 g actually sits, the 51 g boot delta, the actuator
# model). What the agreement does establish is that the total is right, which is
# the part hop height depends on.
#
# THE BATTERY IS THE PRIME SUSPECT, and it is not in the spec: there is no body
# named battery, pack or cell anywhere, and an NP-F550 is 75-80 g. Either the
# pack is partly folded into `trunk_base`'s 199.2 g already, or the residual is
# a mix of pack, fasteners and wiring. It cannot be decomposed further without
# weighing subassemblies, so the whole residual goes into the trunk as one lump.
#
# WHY IT MATTERS HERE RATHER THAN AS A ROUNDING ERROR. For a fixed push impulse
# hop height goes as 1/m^2, so 839 -> 893 g is ~12% of hop height -- the
# difference between a ~21 mm and a ~19 mm ballistic apex. It also detunes
# `hop_load_force` and `hop_symmetric_push`, which both normalise against a
# `body_weight_n` computed from the compiled mass, so they saturated ~6% early.
TRUNK_BODY = "trunk_base"

# 893.0 g real (908 g measured, minus the 15 g radiator now removed)
# - 839.2 g modelled (737.2 g spec + 2 x 51 g boot delta).
#
# PUT ON THE TRUNK, WHICH IS AN ASSUMPTION ABOUT PLACEMENT, NOT MASS. The 53.8 g
# is measured; that it sits at the trunk's existing centre of mass is not. It is
# the least-wrong single location -- the trunk is where the pack mounts and it is
# already the second-heaviest body -- but 54 g displaced from the true position
# tilts the CoM, and CoM offset matters more to a SYMMETRIC hop than mass does.
# `randomize_com` covers +/-15 mm, which brackets any plausible error here.
UNMODELLED_TRUNK_MASS = 0.0538


def _add_trunk_ballast(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """Add the unmodelled 53.8 g to the trunk, scaling inertia with it.

    Scaling `fullinertia` by the mass ratio assumes the added mass is
    distributed like the body already is. That is a 27% change on `trunk_base`,
    so the assumption stretches -- accepted because the alternative, guessing a
    position for the pack, adds an unmeasured offset on top of an unmeasured
    distribution. See UNMODELLED_TRUNK_MASS for how little the placement
    actually costs.
    """
    for body in spec.bodies:
        if body.name == TRUNK_BODY:
            original = body.mass
            if original > 0:
                scale = (original + UNMODELLED_TRUNK_MASS) / original
                body.mass = original + UNMODELLED_TRUNK_MASS
                body.fullinertia = [x * scale for x in body.fullinertia]
            break
    return spec


def get_walk_spec() -> mujoco.MjSpec:
    return _add_trunk_ballast(
        mujoco.MjSpec.from_file(str(MICRODUCK_WALK_XML))
    )


def get_standup_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(MICRODUCK_ALLCOLLISIONS_XML))


def get_ground_pick_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(MICRODUCK_ALLCOLLISIONS_XML))


def get_walk_rollers_spec() -> mujoco.MjSpec:
    # NOTE: was loading robot_allcollisions.xml (no wheels) — the roller env
    # silently ran on the wheel-less standup model.
    return mujoco.MjSpec.from_file(str(MICRODUCK_ALLCOLLISIONS_ROLLERS_XML))


def get_ball_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(MICRODUCK_BALL_XML))


def get_backlash_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(MICRODUCK_ALLCOLLISIONS_BACKLASH_XML))


def get_walk_backlash_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(MICRODUCK_WALK_BACKLASH_XML))


def get_rollers_backlash_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(MICRODUCK_ALLCOLLISIONS_ROLLERS_BACKLASH_XML))


HOME_FRAME = EntityCfg.InitialStateCfg(
    joint_pos={
        # Lower body — STAND2 pose: trunk shifted ~5mm forward over the feet so
        # the CoM sits over the ankle axis (was ~5mm behind it at the old HOME,
        # which biased the robot backward and made the standup policy droop its
        # head forward as a counterweight). Leg pitch chain leaned forward:
        # hip_pitch 30°→26.24°, ankle 30°→25.95°, knee 0°→0.28°. Matches the
        # STAND keyframe in scene.xml / scene_walk.xml.
        r".*hip_yaw.*": 0.0,
        r".*left_hip_roll.*": -0.0873,
        r".*right_hip_roll.*": 0.0873,
        r".*left_hip_pitch.*": -0.4579,
        r".*right_hip_pitch.*": 0.4579,
        r".*left_knee.*": -0.0049,
        r".*right_knee.*": 0.0049,
        r".*left_ankle.*": 0.4530,
        r".*right_ankle.*": -0.4530,
        # Head
        r".*neck_pitch.*": 0.3491,
        r".*head_pitch.*": 0.3491,
        r".*head_yaw.*": 0.0,
        r".*head_roll.*": 0.0,
    },
    joint_vel={".*": 0.0},
)

FULL_COLLISION = CollisionCfg(
    geom_names_expr=[".*_collision"],
    condim={r"^(left|right)_foot_collision$": 3, ".*_collision": 1},
    priority={r"^(left|right)_foot_collision$": 1},
    friction={r"^(left|right)_foot_collision$": (1.0,)},
)

# -- Old actuator (XML position, MuJoCo built-in PD + friction) --
# actuators = DelayedActuatorCfg(
    # delay_min_lag=0,
    # delay_max_lag=3,
    # base_cfg=XmlPositionActuatorCfg(joint_names_expr=(r".*",)),
# )

# -- BAM M6 actuator (full voltage control + load-dependent friction) --
# Exclude passive_* joints (jaw linkage in the new model has no XML actuator).
# Voltage domain randomization (mirrors mjlab_microban):
#   - vin_range: per-env battery voltage sampled at startup (replaces fixed vin)
#   - vin_drop_gain_range: load-dependent voltage sag V_drop = gain * sum(|tau|)
#   - vin_min: hard floor on the effective voltage after sag
# kp_fw kept at 200 (microduck's preserved firmware stiffness; microban uses 125).
_BAM_ACTUATOR_KWARGS = dict(
    motor_name="xl330",
    model="m6",
    target_names_expr=(r"^(?!passive_).*",),
    kp_fw=200.0,  # microduck's preserved firmware stiffness (microban uses 125)
    # vin_range=(6.9, 7.9),
    vin_range=(6.5, 8.2),
    vin_drop_gain_range=(0.0, 0.2),
    vin_min=6.0,
    # max_current=1.75,
    delay_min_lag=3,
    delay_max_lag=6,
)
actuators = FrictionDRBamActuatorCfg(**_BAM_ACTUATOR_KWARGS)

# Same BAM actuator, but the firmware position loop reads the encoder THROUGH
# the passive_<joint>_backlash hinges (the real encoder is on the output side
# of the gear play). Only for the backlash model; the target regex already
# excludes the passive_* backlash joints from actuation.
backlash_actuators = BacklashEncoderBamActuatorCfg(**_BAM_ACTUATOR_KWARGS)

# -- BAM M4 actuator
# actuators = DelayedActuatorCfg(
    # delay_min_lag=0,
    # delay_max_lag=3,
    # base_cfg=make_bam_m4_actuator_cfg(),
# )

# HOME frame for the backlash model. HOME_FRAME's unanchored patterns
# (e.g. r".*left_hip_roll.*") would also match passive_left_hip_roll_backlash
# and try to initialize it at -0.0873 rad — outside its ±1° range. Pattern
# matching is first-match-wins in declaration order, so the anchored backlash
# rule placed FIRST pins every backlash joint at 0 and the servo joints fall
# through to the normal HOME values.
BACKLASH_HOME_FRAME = EntityCfg.InitialStateCfg(
    joint_pos={r".*_backlash$": 0.0, **HOME_FRAME.joint_pos},
    joint_vel={".*": 0.0},
)

MICRODUCK_WALK_ROBOT_CFG = EntityCfg(
    spec_fn=get_walk_spec,
    init_state=HOME_FRAME,
    collisions=(FULL_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)

MICRODUCK_STANDUP_ROBOT_CFG = EntityCfg(
    spec_fn=get_standup_spec,
    init_state=HOME_FRAME,
    collisions=(FULL_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)

MICRODUCK_GROUND_PICK_ROBOT_CFG = EntityCfg(
    spec_fn=get_ground_pick_spec,
    init_state=HOME_FRAME,
    collisions=(FULL_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)

# Backlash robots: base model + ±1° serial backlash hinge per servo.
# Encoder reads through the backlash (BacklashEncoderBamActuator feedback +
# joint_pos/vel_rel_backlash observations — see tasks/backlash.py).
# Allcollisions variant → VelStand/StandUp backlash tasks (mirrors
# MICRODUCK_STANDUP_ROBOT_CFG); walk variant → Velocity/Velocity2 backlash
# tasks (mirrors MICRODUCK_WALK_ROBOT_CFG, keeps backlash-vs-base comparisons
# unconfounded by the collision model).
MICRODUCK_BACKLASH_ROBOT_CFG = EntityCfg(
    spec_fn=get_backlash_spec,
    init_state=BACKLASH_HOME_FRAME,
    collisions=(FULL_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(backlash_actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)

MICRODUCK_WALK_BACKLASH_ROBOT_CFG = EntityCfg(
    spec_fn=get_walk_backlash_spec,
    init_state=BACKLASH_HOME_FRAME,
    collisions=(FULL_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(backlash_actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)

# Roller-skate backlash robot: wheels stay free (passive_*wheel untouched by
# add_backlash.py). collisions=() mirrors MICRODUCK_WALK_ROLLERS_ROBOT_CFG —
# roller wheel collision geoms have no explicit names; XML defaults apply.
MICRODUCK_ROLLERS_BACKLASH_ROBOT_CFG = EntityCfg(
    spec_fn=get_rollers_backlash_spec,
    init_state=BACKLASH_HOME_FRAME,
    collisions=(),
    articulation=EntityArticulationInfoCfg(
        actuators=(backlash_actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)

# Free-floating, non-articulated ball prop for the BallKick task. Position is
# set each episode by the reset_ball_in_front_of_foot event; the init pos here
# only matters for the pristine pre-first-reset state.
MICRODUCK_BALL_CFG = EntityCfg(
    spec_fn=get_ball_spec,
    init_state=EntityCfg.InitialStateCfg(pos=(0.3, 0.0, 0.035)),
)

# Roller skate robot: the 4 passive wheel joints (passive_*wheel) have no XML
# actuators; the BAM cfg's target regex already excludes them, so the action
# space stays 14-dimensional. Uses the SAME canonical BAM actuator as every
# other variant (was a plain XmlActuatorCfg PD — an actuator-physics mismatch
# vs the rest of the family, and joint-friction DR was impossible).
MICRODUCK_WALK_ROLLERS_ROBOT_CFG = EntityCfg(
    spec_fn=get_walk_rollers_spec,
    init_state=HOME_FRAME,
    collisions=(),  # roller wheel collision geoms have no explicit names; XML defaults apply
    articulation=EntityArticulationInfoCfg(
        actuators=(actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)

if __name__ == "__main__":
    import mujoco.viewer as viewer
    from mjlab.scene import Scene, SceneCfg
    from mjlab.terrains import TerrainImporterCfg

    SCENE_CFG = SceneCfg(
        terrain=TerrainImporterCfg(terrain_type="plane"),
        entities={"robot": MICRODUCK_WALK_ROBOT_CFG},
    )

    scene = Scene(SCENE_CFG, device="cuda:0")
    viewer.launch(scene.compile())

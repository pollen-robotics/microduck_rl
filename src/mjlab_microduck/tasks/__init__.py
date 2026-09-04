import os
import sys
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner


class MicroduckOnPolicyRunner(VelocityOnPolicyRunner):
    def __init__(self, env, train_cfg: dict, log_dir=None, device="cpu", **kwargs):
        super().__init__(env, train_cfg, log_dir, device, **kwargs)
        # resolve_symmetry_config injects _env into train_cfg["algorithm"]["symmetry_cfg"]
        # in-place, sharing the same dict object with self.alg.symmetry.  Replace the
        # train_cfg reference with a copy that omits _env so dump_yaml can serialize the
        # config (MjSpec is not picklable), without touching the PPO's internal reference.
        alg = train_cfg.get("algorithm", {})
        sym = alg.get("symmetry_cfg") if isinstance(alg, dict) else None
        if isinstance(sym, dict) and "_env" in sym:
            alg["symmetry_cfg"] = {k: v for k, v in sym.items() if k != "_env"}


# ---------------------------------------------------------------------------
# mjlab 1.3.0 migration — velocity-family-first scope.
# Only velocity / velocity2 / velstand / velstand_tiptoe / standup / sit are
# ported and verified under 1.3.0 + canonical BAM. The remaining env cfgs
# (rollers, pose, sitstand, testbench) are NOT yet migrated — re-enable each
# import + registration once it is ported and verified.
# ---------------------------------------------------------------------------
from .microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
    MicroduckRlCfg,
)
from .microduck_velocity2_env_cfg import (
    make_microduck_velocity2_env_cfg,
    MicroduckVelocity2RlCfg,
)
from .microduck_standup_env_cfg import (
    make_microduck_standup_env_cfg,
    MicroduckStandUpRlCfg,
)
from .microduck_velstand_env_cfg import (
    make_microduck_velstand_env_cfg,
    MicroduckVelStandRlCfg,
)
from .microduck_velstand_tiptoe_env_cfg import (
    make_microduck_velstand_tiptoe_env_cfg,
    MicroduckVelStandTipToeRlCfg,
)
from .microduck_ground_pick_env_cfg import (
    make_microduck_ground_pick_env_cfg,
    MicroduckGroundPickRlCfg,
)
from .microduck_ball_kick_env_cfg import (
    make_microduck_ball_kick_env_cfg,
    MicroduckBallKickRlCfg,
)
from .microduck_sit_env_cfg import (
    make_microduck_sit_env_cfg,
    MicroduckSitRlCfg,
)
from .microduck_velocity_rollers_env_cfg import (
    make_microduck_velocity_rollers_env_cfg,
    MicroduckRollersRlCfg,
)
from .microduck_velocity_swizzle_env_cfg import (
    make_microduck_velocity_swizzle_env_cfg,
    MicroduckSwizzleRlCfg,
)
from .microduck_roller_crouch_env_cfg import (
    make_microduck_roller_crouch_env_cfg,
    MicroduckRollerCrouchRlCfg,
)
from .microduck_roller_slope_env_cfg import (
    make_microduck_roller_slope_env_cfg,
    MicroduckRollerSlopeRlCfg,
)
from .microduck_shoot_env_cfg import (
    make_microduck_shoot_env_cfg,
    MicroduckShootRlCfg,
)
from .microduck_roller_standup_env_cfg import (
    make_microduck_roller_standup_env_cfg,
    MicroduckRollerStandUpRlCfg,
)
from .microduck_spin_env_cfg import (
    make_microduck_spin_env_cfg,
    MicroduckSpinRlCfg,
)
from .backlash import make_backlash_variant
from .run import make_run_variant, MicroduckRunRlCfg
from .sprung import SWEEP_ARMS, make_sprung_variant, sprung_rl_cfg, ARM_TASK_SUFFIX
from .hop import (
    HOP_ARMS,
    HOP_ARM_SUFFIX,
    apply_hop_corrections,
    hop_rl_cfg,
    make_hop_variant,
    make_in_place_variant,
    make_symmetric_variant,
)
from mjlab_microduck.robot.sprung_foot import H_ADD, K_MEASURED, PAD_MASS, TRAVEL

# Standard velocity task
register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-MicroDuck",
    env_cfg=make_microduck_velocity_env_cfg(),
    play_env_cfg=make_microduck_velocity_env_cfg(play=True),
    rl_cfg=MicroduckRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
print("✓ Velocity task registered: Mjlab-Velocity-Flat-MicroDuck")

register_mjlab_task(
    task_id="Mjlab-Velocity-Rough-MicroDuck",
    env_cfg=make_microduck_velocity_env_cfg(rough=True),
    play_env_cfg=make_microduck_velocity_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Run task — rigid running baseline (Phase 1 of the sprung-leg campaign).
# Control for the later sprung comparison; see
# docs/superpowers/specs/2026-08-17-sprung-running-design.md
register_mjlab_task(
    task_id="Mjlab-Run-Flat-MicroDuck",
    env_cfg=make_run_variant(make_microduck_velocity_env_cfg()),
    play_env_cfg=make_run_variant(make_microduck_velocity_env_cfg(play=True)),
    rl_cfg=MicroduckRunRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
print("✓ Run task registered: Mjlab-Run-Flat-MicroDuck")

register_mjlab_task(
    task_id="Mjlab-Run-Rough-MicroDuck",
    env_cfg=make_run_variant(make_microduck_velocity_env_cfg(rough=True)),
    play_env_cfg=make_run_variant(
        make_microduck_velocity_env_cfg(play=True, rough=True)
    ),
    rl_cfg=MicroduckRunRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
print("✓ Run task registered: Mjlab-Run-Rough-MicroDuck")

# Sprung-foot stiffness sweep — Phase 2. See
# docs/superpowers/specs/2026-08-20-sprung-foot-design.md
for _label, _k, _travel, _pad_mass in SWEEP_ARMS:
    _tid = f"Mjlab-Run-Flat-Sprung-{ARM_TASK_SUFFIX[_label]}-MicroDuck"
    register_mjlab_task(
        task_id=_tid,
        env_cfg=make_sprung_variant(
            make_run_variant(make_microduck_velocity_env_cfg()),
            stiffness=_k,
            travel=_travel,
            pad_mass=_pad_mass,
        ),
        play_env_cfg=make_sprung_variant(
            make_run_variant(make_microduck_velocity_env_cfg(play=True)),
            stiffness=_k,
            travel=_travel,
            pad_mass=_pad_mass,
        ),
        rl_cfg=sprung_rl_cfg(_label),
        runner_cls=MicroduckOnPolicyRunner,
    )
    print(f"✓ Sprung task registered: {_tid}")

# Periodic hop on the sprung foot — Phase 4. See
# docs/superpowers/specs/2026-08-24-sprung-hop-design.md
#
# NOTE: stiffness=_k is passed into make_hop_variant (not just
# make_sprung_variant) for every arm below. make_hop_variant registers
# hop_energy_monitor with a stiffness used to compute stored spring energy,
# defaulting to 3900.0 -- left unpassed, the k2500 arm's
# Metrics/hop_spring_energy_* would read 56% high, and the spec requires
# reading the spring instruments before any hop-height number.
#
# NOTE 2: make_hop_variant NO LONGER TAKES h_add, and the two-call sync problem
# this note used to describe is gone with it. It used to need h_add to shift an
# ABSOLUTE hop-height target, which had to stay in step with the h_add
# make_sprung_variant uses to shift the com_height_target band -- overriding one
# and not the other desynchronised them silently, with no error. Both hop height
# rewards now measure RISE ABOVE TAKEOFF HEIGHT, which is invariant to how tall
# the robot stands, so h_add has exactly one consumer again: the CoM band, owned
# by make_sprung_variant. It is still passed explicitly there for visibility.
for _label, _k, _travel, _pad in HOP_ARMS:
    _tid = f"Mjlab-Hop-Flat-Sprung-{HOP_ARM_SUFFIX[_label]}-MicroDuck"
    register_mjlab_task(
        task_id=_tid,
        env_cfg=apply_hop_corrections(make_sprung_variant(
            make_hop_variant(make_microduck_velocity_env_cfg(), stiffness=_k),
            stiffness=_k, travel=_travel, pad_mass=_pad, h_add=H_ADD,
        )),
        play_env_cfg=apply_hop_corrections(make_sprung_variant(
            make_hop_variant(make_microduck_velocity_env_cfg(play=True), stiffness=_k),
            stiffness=_k, travel=_travel, pad_mass=_pad, h_add=H_ADD,
        )),
        rl_cfg=hop_rl_cfg(_label),
        runner_cls=MicroduckOnPolicyRunner,
    )
    print(f"✓ Hop task registered: {_tid}")

# The STANDARD-foot hop arm: the stock robot, no boot at all. This is the
# engineering baseline -- what the robot is today -- and it is what the spring
# boot has to beat. `Locked` is the SCIENTIFIC control (matched 51 g pad and
# matched 30 mm height, zero travel, so only compliance differs); Standard is
# the honest alternative, carrying neither the 51 g of distal mass nor the 30 mm
# of added height that cost -2%/mm of hop.
#
# The stiffness argument only reaches hop_energy_monitor, which reports zeros
# when the spring joints are absent -- so it is inert here.
# It deliberately skips make_sprung_variant: with no spring joints there is no
# CoM band to shift and no pose scoping to do, and hop_energy_monitor already
# reports zeros when the joints are absent.
_std_tid = "Mjlab-Hop-Flat-Standard-MicroDuck"
register_mjlab_task(
    task_id=_std_tid,
    env_cfg=apply_hop_corrections(
        make_hop_variant(make_microduck_velocity_env_cfg(), stiffness=K_MEASURED)
    ),
    play_env_cfg=apply_hop_corrections(
        make_hop_variant(make_microduck_velocity_env_cfg(play=True), stiffness=K_MEASURED)
    ),
    rl_cfg=hop_rl_cfg("standard"),
    runner_cls=MicroduckOnPolicyRunner,
)
print(f"✓ Hop task registered: {_std_tid}")

# ── In-place hop, head freed ─────────────────────────────────────────────────
#
# Two arms on the MEASURED stiffness, answering one question: does paying for a
# TWO-FOOTED launch buy height over the skip the campaign actually produced?
#
# Both arms carry `make_in_place_variant`, which is not part of the question --
# it fixes two things that were wrong for every earlier hop run:
#
#   * `stillness_at_zero_command` (weight 3.0) never fired, because its gate
#     tests the command magnitude and the hop command's magnitude is 1.0 by
#     construction. Run 59yiy9h6 drifted 132 mm in 14 s as a result.
#   * The head -- 39% of body mass, at the top of the body -- was held still by
#     `head_pose_tracking` (+3.0) and `neck_action_rate_l2` (-0.1). Both are
#     dropped so the policy can swing it. This SPENDS runtime head control on
#     this task; see `make_in_place_variant` for the trade.
#
# The baseline for both is the published K3344 run 59yiy9h6, which does not need
# re-running: 27.6 mm reported / ~21 mm ballistic, skipping at ~3.3 Hz.
for _sym, _suffix in ((False, "InPlace"), (True, "InPlaceSym")):
    def _build(play: bool, _sym=_sym):
        cfg = make_in_place_variant(
            make_hop_variant(
                make_microduck_velocity_env_cfg(play=play), stiffness=K_MEASURED
            )
        )
        if _sym:
            cfg = make_symmetric_variant(cfg)
        return apply_hop_corrections(
            make_sprung_variant(
                cfg, stiffness=K_MEASURED, travel=TRAVEL,
                pad_mass=PAD_MASS, h_add=H_ADD,
            )
        )

    _tid = f"Mjlab-Hop-{_suffix}-K3344-MicroDuck"
    register_mjlab_task(
        task_id=_tid,
        env_cfg=_build(play=False),
        play_env_cfg=_build(play=True),
        rl_cfg=hop_rl_cfg("k3344"),
        runner_cls=MicroduckOnPolicyRunner,
    )
    print(f"✓ Hop task registered: {_tid}")

# Velocity2 — microban reward/regularization recipe on the velocity task.
register_mjlab_task(
    task_id="Mjlab-Velocity2-Flat-MicroDuck",
    env_cfg=make_microduck_velocity2_env_cfg(),
    play_env_cfg=make_microduck_velocity2_env_cfg(play=True),
    rl_cfg=MicroduckVelocity2RlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Velocity2-Rough-MicroDuck",
    env_cfg=make_microduck_velocity2_env_cfg(rough=True),
    play_env_cfg=make_microduck_velocity2_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckVelocity2RlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# VelStand — walking + fall recovery + body pose control in one policy.
register_mjlab_task(
    task_id="Mjlab-VelStand-Flat-MicroDuck",
    env_cfg=make_microduck_velstand_env_cfg(),
    play_env_cfg=make_microduck_velstand_env_cfg(play=True),
    rl_cfg=MicroduckVelStandRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
print("✓ VelStand task registered: Mjlab-VelStand-Flat-MicroDuck")

register_mjlab_task(
    task_id="Mjlab-VelStand-Rough-MicroDuck",
    env_cfg=make_microduck_velstand_env_cfg(rough=True),
    play_env_cfg=make_microduck_velstand_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckVelStandRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
print("✓ VelStand task registered: Mjlab-VelStand-Rough-MicroDuck")

# VelStand-TipToe — same as VelStand but with a feet_tiptoe_alignment reward.
register_mjlab_task(
    task_id="Mjlab-VelStandTipToe-Flat-MicroDuck",
    env_cfg=make_microduck_velstand_tiptoe_env_cfg(),
    play_env_cfg=make_microduck_velstand_tiptoe_env_cfg(play=True),
    rl_cfg=MicroduckVelStandTipToeRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
print("✓ VelStand-TipToe task registered: Mjlab-VelStandTipToe-Flat-MicroDuck")

register_mjlab_task(
    task_id="Mjlab-VelStandTipToe-Rough-MicroDuck",
    env_cfg=make_microduck_velstand_tiptoe_env_cfg(rough=True),
    play_env_cfg=make_microduck_velstand_tiptoe_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckVelStandTipToeRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
print("✓ VelStand-TipToe task registered: Mjlab-VelStandTipToe-Rough-MicroDuck")

# Stand-up task — robot starts inverted (lying on back) and must stand up
register_mjlab_task(
    task_id="Mjlab-StandUp-Flat-MicroDuck",
    env_cfg=make_microduck_standup_env_cfg(),
    play_env_cfg=make_microduck_standup_env_cfg(play=True),
    rl_cfg=MicroduckStandUpRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
print("✓ StandUp task registered: Mjlab-StandUp-Flat-MicroDuck")

register_mjlab_task(
    task_id="Mjlab-StandUp-Rough-MicroDuck",
    env_cfg=make_microduck_standup_env_cfg(rough=True),
    play_env_cfg=make_microduck_standup_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckStandUpRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
print("✓ StandUp task registered: Mjlab-StandUp-Rough-MicroDuck")

# Sit task — standing → sitting keyframe, gently (companion to StandUp)
register_mjlab_task(
    task_id="Mjlab-Sit-Flat-MicroDuck",
    env_cfg=make_microduck_sit_env_cfg(),
    play_env_cfg=make_microduck_sit_env_cfg(play=True),
    rl_cfg=MicroduckSitRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
print("✓ Sit task registered: Mjlab-Sit-Flat-MicroDuck")

register_mjlab_task(
    task_id="Mjlab-Sit-Rough-MicroDuck",
    env_cfg=make_microduck_sit_env_cfg(rough=True),
    play_env_cfg=make_microduck_sit_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckSitRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
print("✓ Sit task registered: Mjlab-Sit-Rough-MicroDuck")

# Ground-pick task — crouch, touch the ground with the mouth tip, return to stand
register_mjlab_task(
    task_id="Mjlab-GroundPick-Flat-MicroDuck",
    env_cfg=make_microduck_ground_pick_env_cfg(),
    play_env_cfg=make_microduck_ground_pick_env_cfg(play=True),
    rl_cfg=MicroduckGroundPickRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
print("✓ GroundPick task registered: Mjlab-GroundPick-Flat-MicroDuck")

# BallKick task — kick a 70mm/15g ball forward hard with the right foot from a
# standing start (flat terrain only — a ball on rough terrain is another task).
register_mjlab_task(
    task_id="Mjlab-BallKick-Flat-MicroDuck",
    env_cfg=make_microduck_ball_kick_env_cfg(),
    play_env_cfg=make_microduck_ball_kick_env_cfg(play=True),
    rl_cfg=MicroduckBallKickRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
print("✓ BallKick task registered: Mjlab-BallKick-Flat-MicroDuck")

register_mjlab_task(
    task_id="Mjlab-GroundPick-Rough-MicroDuck",
    env_cfg=make_microduck_ground_pick_env_cfg(rough=True),
    play_env_cfg=make_microduck_ground_pick_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckGroundPickRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
print("✓ GroundPick task registered: Mjlab-GroundPick-Rough-MicroDuck")

# Shoot task — standing kick with the right leg while the left leg stays planted
register_mjlab_task(
    task_id="Mjlab-Shoot-Flat-MicroDuck",
    env_cfg=make_microduck_shoot_env_cfg(),
    play_env_cfg=make_microduck_shoot_env_cfg(play=True),
    rl_cfg=MicroduckShootRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
print("✓ Shoot task registered: Mjlab-Shoot-Flat-MicroDuck")

# Roller skate velocity task (passive-wheel model; historical task id kept)
register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-MicroDuck-Rollers",
    env_cfg=make_microduck_velocity_rollers_env_cfg(),
    play_env_cfg=make_microduck_velocity_rollers_env_cfg(play=True),
    rl_cfg=MicroduckRollersRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
print("✓ Rollers task registered: Mjlab-Velocity-Flat-MicroDuck-Rollers")

# Roller SWIZZLE task — clean classic swizzle (symmetric, feet grounded).
register_mjlab_task(
    task_id="Mjlab-Velocity-Swizzle-MicroDuck",
    env_cfg=make_microduck_velocity_swizzle_env_cfg(),
    play_env_cfg=make_microduck_velocity_swizzle_env_cfg(play=True),
    rl_cfg=MicroduckSwizzleRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
print("✓ Swizzle task registered: Mjlab-Velocity-Swizzle-MicroDuck")

register_mjlab_task(
    task_id="Mjlab-RollerCrouch-Flat-MicroDuck",
    env_cfg=make_microduck_roller_crouch_env_cfg(),
    play_env_cfg=make_microduck_roller_crouch_env_cfg(play=True),
    rl_cfg=MicroduckRollerCrouchRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
print("✓ RollerCrouch task registered: Mjlab-RollerCrouch-Flat-MicroDuck")

register_mjlab_task(
    task_id="Mjlab-RollerSlope-Flat-MicroDuck",
    env_cfg=make_microduck_roller_slope_env_cfg(),
    play_env_cfg=make_microduck_roller_slope_env_cfg(play=True),
    rl_cfg=MicroduckRollerSlopeRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
print("✓ RollerSlope task registered: Mjlab-RollerSlope-Flat-MicroDuck")

# Roller STANDUP — se relever sur rollers (policy dédiée, départ au sol).
register_mjlab_task(
    task_id="Mjlab-RollerStandUp-Flat-MicroDuck",
    env_cfg=make_microduck_roller_standup_env_cfg(),
    play_env_cfg=make_microduck_roller_standup_env_cfg(play=True),
    rl_cfg=MicroduckRollerStandUpRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
print("✓ RollerStandUp task registered: Mjlab-RollerStandUp-Flat-MicroDuck")

# Spin task — rotation rapide sur place, sur rollers (slot ground-pick).
register_mjlab_task(
    task_id="Mjlab-Spin-Flat-MicroDuck",
    env_cfg=make_microduck_spin_env_cfg(),
    play_env_cfg=make_microduck_spin_env_cfg(play=True),
    rl_cfg=MicroduckSpinRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
print("✓ Spin task registered: Mjlab-Spin-Flat-MicroDuck")

# Backlash variants — ±1° serial gear play per servo + encoder-through-backlash
# actuator feedback and joint obs (see tasks/backlash.py). Each family keeps its
# base task's collision model: Velocity/Velocity2 → robot_walk_backlash.xml,
# VelStand/StandUp → robot_allcollisions_backlash.xml. Obs/action dims are
# unchanged vs the base tasks.
from mjlab_microduck.robot.microduck_constants import (
    MICRODUCK_BACKLASH_ROBOT_CFG,
    MICRODUCK_ROLLERS_BACKLASH_ROBOT_CFG,
    MICRODUCK_WALK_BACKLASH_ROBOT_CFG,
)

# (task_id, make_fn, make_kwargs, rl_cfg, backlash robot cfg). Task ids mirror
# the base ids with "-Backlash" inserted. Walk-model tasks get the walk
# backlash robot, roller tasks the wheels+backlash robot, the rest the
# allcollisions backlash robot — same model as their base task in each case.
_BL_ALLCOL = MICRODUCK_BACKLASH_ROBOT_CFG
_BL_WALK = MICRODUCK_WALK_BACKLASH_ROBOT_CFG
_BL_ROLLERS = MICRODUCK_ROLLERS_BACKLASH_ROBOT_CFG
_BACKLASH_TASKS = (
    ("Mjlab-Velocity-Flat-Backlash-MicroDuck", make_microduck_velocity_env_cfg, {}, MicroduckRlCfg, _BL_WALK),
    ("Mjlab-Velocity-Rough-Backlash-MicroDuck", make_microduck_velocity_env_cfg, {"rough": True}, MicroduckRlCfg, _BL_WALK),
    ("Mjlab-Velocity2-Flat-Backlash-MicroDuck", make_microduck_velocity2_env_cfg, {}, MicroduckVelocity2RlCfg, _BL_WALK),
    ("Mjlab-Velocity2-Rough-Backlash-MicroDuck", make_microduck_velocity2_env_cfg, {"rough": True}, MicroduckVelocity2RlCfg, _BL_WALK),
    ("Mjlab-VelStand-Flat-Backlash-MicroDuck", make_microduck_velstand_env_cfg, {}, MicroduckVelStandRlCfg, _BL_ALLCOL),
    ("Mjlab-VelStand-Rough-Backlash-MicroDuck", make_microduck_velstand_env_cfg, {"rough": True}, MicroduckVelStandRlCfg, _BL_ALLCOL),
    ("Mjlab-VelStandTipToe-Flat-Backlash-MicroDuck", make_microduck_velstand_tiptoe_env_cfg, {}, MicroduckVelStandTipToeRlCfg, _BL_ALLCOL),
    ("Mjlab-VelStandTipToe-Rough-Backlash-MicroDuck", make_microduck_velstand_tiptoe_env_cfg, {"rough": True}, MicroduckVelStandTipToeRlCfg, _BL_ALLCOL),
    ("Mjlab-StandUp-Flat-Backlash-MicroDuck", make_microduck_standup_env_cfg, {}, MicroduckStandUpRlCfg, _BL_ALLCOL),
    ("Mjlab-StandUp-Rough-Backlash-MicroDuck", make_microduck_standup_env_cfg, {"rough": True}, MicroduckStandUpRlCfg, _BL_ALLCOL),
    ("Mjlab-Sit-Flat-Backlash-MicroDuck", make_microduck_sit_env_cfg, {}, MicroduckSitRlCfg, _BL_ALLCOL),
    ("Mjlab-Sit-Rough-Backlash-MicroDuck", make_microduck_sit_env_cfg, {"rough": True}, MicroduckSitRlCfg, _BL_ALLCOL),
    ("Mjlab-GroundPick-Flat-Backlash-MicroDuck", make_microduck_ground_pick_env_cfg, {}, MicroduckGroundPickRlCfg, _BL_ALLCOL),
    ("Mjlab-GroundPick-Rough-Backlash-MicroDuck", make_microduck_ground_pick_env_cfg, {"rough": True}, MicroduckGroundPickRlCfg, _BL_ALLCOL),
    ("Mjlab-BallKick-Flat-Backlash-MicroDuck", make_microduck_ball_kick_env_cfg, {}, MicroduckBallKickRlCfg, _BL_ALLCOL),
    ("Mjlab-Shoot-Flat-Backlash-MicroDuck", make_microduck_shoot_env_cfg, {}, MicroduckShootRlCfg, _BL_ALLCOL),
    ("Mjlab-Velocity-Flat-Backlash-MicroDuck-Rollers", make_microduck_velocity_rollers_env_cfg, {}, MicroduckRollersRlCfg, _BL_ROLLERS),
    ("Mjlab-Velocity-Swizzle-Backlash-MicroDuck", make_microduck_velocity_swizzle_env_cfg, {}, MicroduckSwizzleRlCfg, _BL_ROLLERS),
    ("Mjlab-RollerCrouch-Flat-Backlash-MicroDuck", make_microduck_roller_crouch_env_cfg, {}, MicroduckRollerCrouchRlCfg, _BL_ROLLERS),
    ("Mjlab-RollerSlope-Flat-Backlash-MicroDuck", make_microduck_roller_slope_env_cfg, {}, MicroduckRollerSlopeRlCfg, _BL_ROLLERS),
)
for _task_id, _make_cfg, _kw, _rl_cfg, _robot_cfg in _BACKLASH_TASKS:
    register_mjlab_task(
        task_id=_task_id,
        env_cfg=make_backlash_variant(_make_cfg(**_kw), _robot_cfg),
        play_env_cfg=make_backlash_variant(_make_cfg(play=True, **_kw), _robot_cfg),
        rl_cfg=_rl_cfg,
        runner_cls=MicroduckOnPolicyRunner,
    )
    print(f"✓ Backlash task registered: {_task_id}")

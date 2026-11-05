"""Microduck *crouchstand* task — commanded crouch ↔ stand, feet-only, GENTLY.

The sitstand recipe with the seated target swapped for a feet-only SQUAT, in
the spirit of the roller crouch-glide pose but on the plain (no-roller) feet:
    cmd (twist slot) = [crouch_flag, 0, 0]   crouch_flag ∈ {0 = STAND, 1 = CROUCH}
Same runtime contract as sitstand: the runtime writes 0/1 into the vx slot, the
robot glides to the commanded posture (~POSTURE_RAMP_S) and HOLDS it for as
long as the flag stays set — the trick is split in two (go down / come up),
not a fixed-period one-shot cycle like the roller crouch.

Built ON TOP of ``make_microduck_sitstand_env_cfg`` so DR / obs / noise /
delays / regularisers / curricula stay in sync with the transferring recipe.
Only the low-posture target and the contact rules change:
  - CROUCH keyframe + CROUCH_Z replace SIT keyframe + SIT_Z everywhere
    (pose, height, composite, stillness, command alpha init, reset spawn).
  - Feet-only: any non-foot robot body touching the ground costs
    NONFOOT_CONTACT_WEIGHT per step, in BOTH postures and during transitions.
    Without it, "sit on the butt" (sitstand's rest) is the cheap basin.
  - upright_while_tall's gate is moved below the crouch height, so the
    upright booster stays fully on at the squat (no butt-rest regime here).

Why not the roller CROUCH_POSE verbatim (2026-11-05 flat-foot analysis,
allcollisions model, soles level, HOME neck):
  - roller pose (hip ±1.41, knee ±1.52, ankle ≈0) put flat-footed has its CoM
    2 mm behind the TOES (sole is 49 mm long) — it only balances on wheels;
  - knee 1.52 is past the 0.9 soft limit (±1.414) → dof_pos_limits would fire.
  Flat feet force the shank forward (ankle dorsiflexion ≈ +0.9) to bring the
  CoM back over the sole. Chosen pose: knee ±1.35 (same as sit), hip ±0.36,
  ankle ±0.92 → CoM centred (24 mm margin front AND back; HOME is 31/18),
  trunk pitched ~4-7° forward (the roller's "léger penché avant"), only the
  soles touch. Open-loop hold with stiff PD: settles at z 0.0714, pitch 7°,
  feet-only contact. NB: the XML position actuators (kp 0.55) can't even hold
  HOME open-loop, so open-loop holds are NOT the criterion here — balance is
  active, the policy closes the loop. Re-measure CROUCH_Z under the BAM
  actuator (tests / play) if the robot model changes.

Joint layout (14 actuated joints):
    0-4 : left  leg (hip_yaw, hip_roll, hip_pitch, knee, ankle)
    5-8 : neck/head (neck_pitch, head_pitch, head_yaw, head_roll)
    9-13: right leg (hip_yaw, hip_roll, hip_pitch, knee, ankle)
"""

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_sitstand_env_cfg import (
    MicroduckSitStandRlCfg,
    STAND_Z,
    make_microduck_sitstand_env_cfg,
)

# ── CROUCH keyframe (joint_pos index → angle in rad). Single fixed target. ──
# Feet-only squat, CoM centred over the soles (see module docstring). Hip yaw
# and the neck/head are left out: yaw stays HOME, the head is command-steered.
CROUCH_TARGET_OVERRIDES = {
    1:   0.0,    # left  hip_roll   (HOME -0.0873)
    2:   0.36,   # left  hip_pitch  (HOME -0.4579)
    3:   1.35,   # left  knee       (HOME -0.0049) — under the ±1.414 soft limit
    4:   0.92,   # left  ankle      (HOME +0.4530) — shank forward, CoM over sole
    10:  0.0,    # right hip_roll   (HOME +0.0873)
    11: -0.36,   # right hip_pitch  (HOME +0.4579)
    12: -1.35,   # right knee       (HOME +0.0049)
    13: -0.92,   # right ankle      (HOME -0.4530)
}

# Trunk height at the crouch (m) — MEASURED in sim (flat-foot placement 0.076,
# settled under gravity 0.0714). Travel from STAND_Z (0.115) ≈ 43 mm.
CROUCH_Z = 0.072

# Upright booster gate: fully on at and above the crouch (feet-only squat
# must stay vertical everywhere), fading out only well below it.
UPRIGHT_GATE_LOW_Z  = 0.040
UPRIGHT_GATE_HIGH_Z = 0.065

# Non-foot ground contact penalty (self-negating → POSITIVE weight). Sized to
# dominate the height/pose deficit of a butt-rest, so sitting can't pay.
NONFOOT_CONTACT_WEIGHT = 4.0

# Transition duration (s): the command slews its internal target blend over
# this time and the posture rewards track that moving setpoint, so it IS the
# learned crouch / rise duration (baked into the policy — the runtime only
# flips 0/1). Sitstand uses 2.0 s; the 2 s crouch felt too slow on the first
# run → 1.0 s. Travel ≈ 43 mm → ≈ 0.043 m/s, still under sitstand's
# MAX_DESCENT_SPEED 0.05 / MAX_RISE_SPEED 0.08 caps. Going below ~0.85 s
# needs those caps raised too, or the policy is taxed for tracking the ramp.
POSTURE_RAMP_S = 1.0

# Spawn band for the crouched reset bucket: just above the measured rest.
CROUCH_SPAWN_Z = (0.074, 0.084)


def make_microduck_crouchstand_env_cfg(
    play: bool = False,
    rough: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create Microduck crouchstand environment configuration."""
    cfg = make_microduck_sitstand_env_cfg(play=play, rough=rough)

    # ── Sensor: any robot body other than the feet touching the ground ───────
    nonfoot_ground_cfg = ContactSensorCfg(
        name="nonfoot_ground_contact",
        primary=ContactMatch(
            mode="body",
            pattern=r".*",
            entity="robot",
            exclude=("ankle_left", "ankle_right"),
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
    cfg.scene.sensors = tuple(cfg.scene.sensors) + (nonfoot_ground_cfg,)

    # ── Swap the low-posture target: SIT → CROUCH ─────────────────────────────
    for name in ("posture_pose_legs", "posture_pose_l1", "posture_composite"):
        cfg.rewards[name].params["sit_overrides"] = CROUCH_TARGET_OVERRIDES
    for name in (
        "posture_height",
        "posture_height_sharp",
        "posture_height_l1",
        "posture_stillness",
        "posture_composite",
    ):
        cfg.rewards[name].params["sit_z"] = CROUCH_Z
        assert cfg.rewards[name].params["stand_z"] == STAND_Z

    cfg.commands["twist"].sit_z = CROUCH_Z
    cfg.commands["twist"].ramp_s = POSTURE_RAMP_S

    cfg.rewards["upright_while_tall"].params["height_low"] = UPRIGHT_GATE_LOW_Z
    cfg.rewards["upright_while_tall"].params["height_high"] = UPRIGHT_GATE_HIGH_Z

    # ── Feet-only rule ────────────────────────────────────────────────────────
    cfg.rewards["nonfoot_ground_contact"] = RewardTermCfg(
        func=microduck_mdp.nonfoot_ground_contact_penalty,
        weight=NONFOOT_CONTACT_WEIGHT,
        params={"sensor_name": nonfoot_ground_cfg.name},
    )

    # ── Reset mix: 50% standing / 50% already crouched ───────────────────────
    gs = cfg.events["set_ground_state"].params
    gs["sitting_joint_overrides"] = CROUCH_TARGET_OVERRIDES
    gs["sitting_z_min"], gs["sitting_z_max"] = CROUCH_SPAWN_Z

    return cfg


# ── RL runner config ──────────────────────────────────────────────────────────

MicroduckCrouchStandRlCfg = deepcopy(MicroduckSitStandRlCfg)
MicroduckCrouchStandRlCfg.experiment_name = "microduck_crouchstand"
MicroduckCrouchStandRlCfg.run_name = "microduck_crouchstand"

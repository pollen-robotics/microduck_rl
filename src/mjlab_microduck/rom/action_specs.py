"""Code-owned execution semantics for every V1 ROM action.

These records are deliberately not extensible from bundle data.  A policy
artifact only becomes executable after the sidecar has an exact reset,
command, safety, completion, and evidence implementation for its task family.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class RuntimeActionSpec:
    action_code: str
    execution_mode: Literal["DISCRETE", "CONTINUOUS_LEASE"]
    task_ids: tuple[str, ...]
    required_capabilities: tuple[str, ...]
    reset_profile: str
    command_profile: str
    phase_period_s: float | None
    kick_mirror: Literal["NONE", "LEFT_RIGHT_EXACT"]
    fall_policy: str
    completion_profile: str | None
    metric_keys: tuple[str, ...]
    supported: bool
    unavailable_reason: str | None = None


def _unsupported(
    code: str,
    task_id: str,
    *,
    capabilities: tuple[str, ...],
    reset: str,
    command: str,
    period: float | None,
    mirror: Literal["NONE", "LEFT_RIGHT_EXACT"] = "NONE",
    fall: str,
    completion: str,
    metrics: tuple[str, ...],
) -> RuntimeActionSpec:
    return RuntimeActionSpec(
        code,
        "DISCRETE",
        (task_id,),
        capabilities,
        reset,
        command,
        period,
        mirror,
        fall,
        completion,
        metrics,
        False,
        "RUNTIME_SEMANTICS_UNSUPPORTED",
    )


ACTION_RUNTIME_SPECS: dict[str, RuntimeActionSpec] = {
    code: RuntimeActionSpec(
        code,
        "CONTINUOUS_LEASE",
        (task,),
        capabilities,
        "DEFAULT_STANDING",
        command,
        None,
        "NONE",
        "FAIL_ON_FALL",
        None,
        metrics,
        supported,
        None if supported else "RUNTIME_SEMANTICS_UNSUPPORTED",
    )
    for code, task, capabilities, command, metrics, supported in (
        (
            "WALK_VELOCITY",
            "Mjlab-Velocity-Flat-MicroDuck",
            ("FLAT_TERRAIN",),
            "TWIST_VELOCITY",
            ("baseTravelM", "trackingError"),
            True,
        ),
        (
            "VELSTAND_VELOCITY",
            "Mjlab-VelStand-Flat-MicroDuck",
            ("FLAT_TERRAIN",),
            "TWIST_VELOCITY",
            ("baseTravelM", "trackingError", "standFraction"),
            True,
        ),
        (
            "ROLLER_VELOCITY",
            "Mjlab-Velocity-Flat-MicroDuck-Rollers",
            ("FLAT_TERRAIN", "ROLLER_FEET"),
            "TWIST_VELOCITY",
            ("baseTravelM", "trackingError"),
            True,
        ),
        (
            "SWIZZLE",
            "Mjlab-Velocity-Swizzle-MicroDuck",
            ("FLAT_TERRAIN", "ROLLER_FEET"),
            "TWIST_VELOCITY",
            ("baseTravelM", "trackingError", "yawRotationRad"),
            True,
        ),
        (
            "ROLLER_SLOPE",
            "Mjlab-RollerSlope-Flat-MicroDuck",
            ("RAMP_TERRAIN", "ROLLER_FEET"),
            "ZERO_TWIST_LEASE",
            ("slopeProgressM", "terrainExitReached"),
            False,
        ),
    )
}

ACTION_RUNTIME_SPECS.update(
    {
        spec.action_code: spec
        for spec in (
            _unsupported(
                "STAND_UP",
                "Mjlab-StandUp-Flat-MicroDuck",
                capabilities=("PRONE_RESET",),
                reset="PRONE_FACE_MIX",
                command="ZERO_TWIST_WITH_TRAINED_HEAD_BODY_TARGET",
                period=None,
                fall="ALLOW_GROUND_CONTACT_DURING_RECOVERY",
                completion="UPRIGHT_SETTLED",
                metrics=("uprightReached", "settlingError"),
            ),
            _unsupported(
                "SIT",
                "Mjlab-SitStand-Flat-MicroDuck",
                capabilities=("FLAT_TERRAIN",),
                reset="DEFAULT_STANDING",
                command="SIT_FLAG_ONE",
                period=None,
                fall="FAIL_ON_FALL",
                completion="SIT_POSE_SETTLED",
                metrics=("sitPoseError",),
            ),
            _unsupported(
                "STAND",
                "Mjlab-SitStand-Flat-MicroDuck",
                capabilities=("SITTING_RESET",),
                reset="TRAINED_SITTING",
                command="SIT_FLAG_ZERO",
                period=None,
                fall="FAIL_ON_FALL",
                completion="STAND_POSE_SETTLED",
                metrics=("standPoseError",),
            ),
            _unsupported(
                "GROUND_PICK",
                "Mjlab-GroundPick-Flat-MicroDuck",
                capabilities=("MOUTH_TIP", "PAYLOAD_FORCE_SCENARIO"),
                reset="DEFAULT_STANDING",
                command="COS_SIN_PHASE_APPROACH_HOLD_RETURN",
                period=4.0,
                fall="FAIL_ON_FALL",
                completion="RETURN_UPRIGHT_WITH_PAYLOAD",
                metrics=("mouthMinHeightM", "payloadLifted", "returnPoseError"),
            ),
            _unsupported(
                "KICK_LEFT",
                "Mjlab-BallKick-Flat-MicroDuck",
                capabilities=("BALL_FREEJOINT", "LEFT_KICK_SCENARIO"),
                reset="BALL_LEFT_OFFSET",
                command="ZERO_TWIST",
                period=None,
                mirror="LEFT_RIGHT_EXACT",
                fall="FAIL_ON_FALL",
                completion="BALL_TARGET_SPEED_AND_SETTLED",
                metrics=(
                    "ballPeakForwardSpeedMps",
                    "ballTravelM",
                    "supportFootContact",
                ),
            ),
            _unsupported(
                "KICK_RIGHT",
                "Mjlab-BallKick-Flat-MicroDuck",
                capabilities=("BALL_FREEJOINT", "RIGHT_KICK_SCENARIO"),
                reset="BALL_RIGHT_OFFSET",
                command="ZERO_TWIST",
                period=None,
                mirror="LEFT_RIGHT_EXACT",
                fall="FAIL_ON_FALL",
                completion="BALL_TARGET_SPEED_AND_SETTLED",
                metrics=(
                    "ballPeakForwardSpeedMps",
                    "ballTravelM",
                    "supportFootContact",
                ),
            ),
            _unsupported(
                "ROULADE",
                "Mjlab-Roulade-Flat-MicroDuck",
                capabilities=("GROUND_ROLL_CONTACTS",),
                reset="CROUCHED_ROLL_START",
                command="ZERO_TWIST",
                period=None,
                fall="ALLOW_INTENTIONAL_ROLL_CONTACT",
                completion="FULL_ROTATION_AND_UPRIGHT",
                metrics=("rollRotationRad", "uprightReached"),
            ),
            _unsupported(
                "ROLLER_CROUCH",
                "Mjlab-RollerCrouch-Flat-MicroDuck",
                capabilities=("ROLLER_FEET",),
                reset="DEFAULT_STANDING",
                command="COS_SIN_ONE_SHOT_CROUCH_GLIDE_RETURN",
                period=5.0,
                fall="FAIL_ON_FALL",
                completion="RETURN_STAND_AFTER_CROUCH",
                metrics=("minimumCrouchHeightM", "glideDistanceM", "returnPoseError"),
            ),
            _unsupported(
                "ROLLER_STAND_UP",
                "Mjlab-RollerStandUp-Flat-MicroDuck",
                capabilities=("ROLLER_FEET", "PRONE_RESET"),
                reset="ROLLER_PRONE_FACE_MIX",
                command="ZERO_TWIST",
                period=None,
                fall="ALLOW_GROUND_CONTACT_DURING_RECOVERY",
                completion="ROLLER_UPRIGHT_SETTLED",
                metrics=("uprightReached", "settlingError"),
            ),
            _unsupported(
                "SPIN",
                "Mjlab-Spin-Flat-MicroDuck",
                capabilities=("ROLLER_FEET",),
                reset="DEFAULT_STANDING",
                command="COS_SIN_SPIN_PHASE",
                period=4.0,
                fall="FAIL_ON_FALL",
                completion="TARGET_YAW_ROTATION_AND_SETTLED",
                metrics=("yawRotationRad", "yawRateError"),
            ),
        )
    }
)

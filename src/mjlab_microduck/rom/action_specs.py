"""Code-owned execution semantics for every V1 ROM action.

These records are deliberately not extensible from bundle data.  A policy
artifact only becomes executable after the sidecar has an exact reset,
command, safety, completion, and evidence implementation for its task family.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

type QualificationMetricDomain = Literal[
    "NONNEGATIVE", "SIGNED", "UNIT_INTERVAL"
]


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
    scenario_fields: tuple[str, ...] = ("terrain", "seed")
    scenario_profile: str = "SEEDED_SERVO_RESET_V1"
    qualification_terrain: str = "flat"
    qualification_parameters: tuple[tuple[str, float], ...] = ()
    qualification_min_seeds: int = 3
    qualification_max_seeds: int = 16
    qualification_min_steps: int = 100
    qualification_max_steps: int = 2_000
    qualification_metric_operators: tuple[
        tuple[str, Literal["gte", "lte"]], ...
    ] = ()
    qualification_metric_domains: tuple[
        tuple[str, QualificationMetricDomain], ...
    ] = ()
    qualification_success_stop_reason: str | None = None
    qualification_min_settled_steps: int = 0
    qualification_completion_metric_max: float | None = None


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
            RuntimeActionSpec(
                action_code="STAND",
                execution_mode="DISCRETE",
                task_ids=("Mjlab-SitStand-Flat-MicroDuck",),
                required_capabilities=("FLAT_TERRAIN", "SITTING_RESET"),
                reset_profile="TRAINED_SITTING",
                command_profile="SIT_FLAG_ZERO",
                phase_period_s=None,
                kick_mirror="NONE",
                fall_policy="FAIL_ON_FALL",
                completion_profile="STAND_POSE_SETTLED",
                metric_keys=("standPoseError",),
                supported=True,
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

_QUALIFICATION_PARAMETERS = {
    "WALK_VELOCITY": (("vxMps", 0.1), ("vyMps", 0.0), ("yawRateRadps", 0.0)),
    "VELSTAND_VELOCITY": (("vxMps", 0.1), ("vyMps", 0.0), ("yawRateRadps", 0.0)),
    "ROLLER_VELOCITY": (("vxMps", 0.1), ("vyMps", 0.0), ("yawRateRadps", 0.0)),
    "SWIZZLE": (("vxMps", 0.0), ("vyMps", 0.0), ("yawRateRadps", 0.5)),
    "ROLLER_SLOPE": (("vxMps", 0.0), ("vyMps", 0.0), ("yawRateRadps", 0.0)),
}
_LESS_IS_BETTER_METRICS = {
    "trackingError",
    "standPoseError",
    "sitPoseError",
    "settlingError",
    "returnPoseError",
    "yawRateError",
}
_SIGNED_METRICS = {"rollRotationRad", "slopeProgressM", "yawRotationRad"}
_UNIT_INTERVAL_METRICS = {
    "payloadLifted",
    "standFraction",
    "supportFootContact",
    "terrainExitReached",
    "uprightReached",
}
for _code, _spec in tuple(ACTION_RUNTIME_SPECS.items()):
    ACTION_RUNTIME_SPECS[_code] = replace(
        _spec,
        qualification_terrain="ramp" if _code == "ROLLER_SLOPE" else "flat",
        qualification_parameters=_QUALIFICATION_PARAMETERS.get(_code, ()),
        qualification_metric_operators=tuple(
            (
                metric,
                "lte" if metric in _LESS_IS_BETTER_METRICS else "gte",
            )
            for metric in _spec.metric_keys
        ),
        qualification_metric_domains=tuple(
            (
                metric,
                "SIGNED"
                if metric in _SIGNED_METRICS
                else (
                    "UNIT_INTERVAL"
                    if metric in _UNIT_INTERVAL_METRICS
                    else "NONNEGATIVE"
                ),
            )
            for metric in _spec.metric_keys
        ),
        qualification_success_stop_reason=(
            "STAND_POSE_SETTLED" if _code == "STAND" else "MAX_STEPS_REACHED"
        ),
        qualification_min_settled_steps=10 if _code == "STAND" else 0,
        qualification_completion_metric_max=0.08 if _code == "STAND" else None,
    )

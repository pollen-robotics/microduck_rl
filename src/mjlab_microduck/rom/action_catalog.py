"""The complete, user-facing MicroDuck ROM action catalog."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .contracts import CompletionContract, LeaseContract


@dataclass(frozen=True)
class ActionTemplate:
    action_code: str
    execution_mode: Literal["DISCRETE", "CONTINUOUS_LEASE"]
    task_ids: tuple[str, ...]
    parameter_schema: dict[str, Any]
    completion: CompletionContract | None
    lease: LeaseContract | None


def _velocity_schema(
    *, vx: tuple[float, float], vy: tuple[float, float], yaw: tuple[float, float]
) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "vxMps": {"type": "number", "minimum": vx[0], "maximum": vx[1]},
            "vyMps": {"type": "number", "minimum": vy[0], "maximum": vy[1]},
            "yawRateRadps": {"type": "number", "minimum": yaw[0], "maximum": yaw[1]},
        },
        "required": ["vxMps", "vyMps", "yawRateRadps"],
    }


def _discrete_schema(*, fixed_goal: str | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {},
    }
    if fixed_goal is not None:
        schema["x-microduck-fixed-goal"] = fixed_goal
    return schema


_LEASE = LeaseContract(
    minLeaseMs=100,
    defaultLeaseMs=500,
    maxLeaseMs=5_000,
    commandCadenceMs=50,
    safeStopBehavior="ZERO_TWIST",
)
_COMPLETION = CompletionContract(
    terminalConditions=["TASK_COMPLETE", "FALLEN", "TIMEOUT"], maxDurationMs=15_000
)

# Bounds are copied from the actual task command ranges. Terrain and backlash are
# qualification variants of these policies, never additional user action codes.
ACTION_TEMPLATES: tuple[ActionTemplate, ...] = (
    ActionTemplate(
        "WALK_VELOCITY",
        "CONTINUOUS_LEASE",
        ("Mjlab-Velocity-Flat-MicroDuck",),
        _velocity_schema(vx=(-0.4, 0.4), vy=(-0.3, 0.3), yaw=(-1.0, 1.0)),
        None,
        _LEASE,
    ),
    ActionTemplate(
        "VELSTAND_VELOCITY",
        "CONTINUOUS_LEASE",
        ("Mjlab-VelStand-Flat-MicroDuck",),
        _velocity_schema(vx=(-0.4, 0.4), vy=(-0.3, 0.3), yaw=(-1.0, 1.0)),
        None,
        _LEASE,
    ),
    ActionTemplate(
        "ROLLER_VELOCITY",
        "CONTINUOUS_LEASE",
        ("Mjlab-Velocity-Flat-MicroDuck-Rollers",),
        _velocity_schema(vx=(-0.5, 0.6), vy=(0.0, 0.0), yaw=(0.0, 0.0)),
        None,
        _LEASE,
    ),
    ActionTemplate(
        "SWIZZLE",
        "CONTINUOUS_LEASE",
        ("Mjlab-Velocity-Swizzle-MicroDuck",),
        _velocity_schema(vx=(-0.6, 0.6), vy=(0.0, 0.0), yaw=(-0.5, 0.5)),
        None,
        _LEASE,
    ),
    ActionTemplate(
        "ROLLER_SLOPE",
        "DISCRETE",
        ("Mjlab-RollerSlope-Flat-MicroDuck",),
        _discrete_schema(),
        _COMPLETION,
        None,
    ),
    ActionTemplate(
        "STAND_UP",
        "DISCRETE",
        ("Mjlab-StandUp-Flat-MicroDuck",),
        _discrete_schema(),
        _COMPLETION,
        None,
    ),
    ActionTemplate(
        "SIT",
        "DISCRETE",
        ("Mjlab-SitStand-Flat-MicroDuck",),
        _discrete_schema(fixed_goal="SIT"),
        _COMPLETION,
        None,
    ),
    ActionTemplate(
        "STAND",
        "DISCRETE",
        ("Mjlab-SitStand-Flat-MicroDuck",),
        _discrete_schema(fixed_goal="STAND"),
        _COMPLETION,
        None,
    ),
    ActionTemplate(
        "GROUND_PICK",
        "DISCRETE",
        ("Mjlab-GroundPick-Flat-MicroDuck",),
        _discrete_schema(),
        _COMPLETION,
        None,
    ),
    ActionTemplate(
        "KICK_LEFT",
        "DISCRETE",
        ("Mjlab-BallKick-Flat-MicroDuck",),
        _discrete_schema(),
        _COMPLETION,
        None,
    ),
    ActionTemplate(
        "KICK_RIGHT",
        "DISCRETE",
        ("Mjlab-BallKick-Flat-MicroDuck",),
        _discrete_schema(),
        _COMPLETION,
        None,
    ),
    ActionTemplate(
        "ROULADE",
        "DISCRETE",
        ("Mjlab-Roulade-Flat-MicroDuck",),
        _discrete_schema(),
        _COMPLETION,
        None,
    ),
    ActionTemplate(
        "ROLLER_CROUCH",
        "DISCRETE",
        ("Mjlab-RollerCrouch-Flat-MicroDuck",),
        _discrete_schema(),
        _COMPLETION,
        None,
    ),
    ActionTemplate(
        "ROLLER_STAND_UP",
        "DISCRETE",
        ("Mjlab-RollerStandUp-Flat-MicroDuck",),
        _discrete_schema(),
        _COMPLETION,
        None,
    ),
    ActionTemplate(
        "SPIN",
        "DISCRETE",
        ("Mjlab-Spin-Flat-MicroDuck",),
        _discrete_schema(),
        _COMPLETION,
        None,
    ),
)

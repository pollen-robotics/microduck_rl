"""The complete, user-facing MicroDuck ROM action catalog."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from .contracts import (
    ActionDefinition,
    CompletionContract,
    LeaseContract,
    PolicyBundle,
    canonical_json,
)
from .mirroring import (
    MICRODUCK_JOINT_MIRROR_PERMUTATION,
    MICRODUCK_JOINT_MIRROR_SIGNS,
)


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
    zeroCommand={"vxMps": 0.0, "vyMps": 0.0, "yawRateRadps": 0.0},
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
        "CONTINUOUS_LEASE",
        ("Mjlab-RollerSlope-Flat-MicroDuck",),
        _velocity_schema(vx=(0.0, 0.0), vy=(0.0, 0.0), yaw=(0.0, 0.0)),
        None,
        _LEASE,
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

CODE_OWNED_ACTION_CODES = tuple(template.action_code for template in ACTION_TEMPLATES)
_TEMPLATE_BY_CODE = {template.action_code: template for template in ACTION_TEMPLATES}
_CODE_OWNED_UNAVAILABLE_REASONS = frozenset(
    {
        "POLICY_ARTIFACT_MISSING",
        "RUNTIME_SEMANTICS_UNSUPPORTED",
        "MODEL_QUALIFICATION_INCOMPATIBLE",
        "MODEL_CAPABILITY_MISSING",
        "MODEL_RUNTIME_INCOMPATIBLE",
        "POLICY_NORMALIZATION_INVALID",
        "POLICY_TASK_ID_MISMATCH",
        "POLICY_PROVENANCE_MISMATCH",
        "POLICY_INFERENCE_INVALID",
        "QUALIFICATION_FAILED",
    }
)


def action_template(action_code: str) -> ActionTemplate:
    """Return the one code-owned V1 template for an action intent."""
    try:
        return _TEMPLATE_BY_CODE[action_code]
    except KeyError as exc:
        raise ValueError(f"unknown code-owned V1 action: {action_code}") from exc


def code_owned_action_definition(
    action_code: str,
    *,
    availability: Literal["AVAILABLE", "UNAVAILABLE"],
    policy_ref: str | None,
    unavailable_reason: str | None = None,
    qualification_refs: list[str] | None = None,
) -> ActionDefinition:
    """Reconstruct the complete V1 wire action while allowing only release state/binding."""
    from .action_specs import ACTION_RUNTIME_SPECS

    template = action_template(action_code)
    spec = ACTION_RUNTIME_SPECS[action_code]
    if availability == "AVAILABLE":
        unavailable_reason = None
    elif unavailable_reason is None:
        unavailable_reason = (
            spec.unavailable_reason if not spec.supported else "POLICY_ARTIFACT_MISSING"
        )
    mirror_transform = (
        {
            "jointPermutation": list(MICRODUCK_JOINT_MIRROR_PERMUTATION),
            "signFlips": list(MICRODUCK_JOINT_MIRROR_SIGNS),
        }
        if spec.kick_mirror == "LEFT_RIGHT_EXACT"
        else None
    )
    safety: dict[str, Any] = {
        "commandProfile": spec.command_profile,
        "fallPolicy": spec.fall_policy,
        "completionProfile": spec.completion_profile,
        "mirroringRule": spec.kick_mirror,
        "safeStopBehavior": (
            template.lease.safeStopBehavior
            if template.lease is not None
            else "HOLD_CURRENT_POSITION"
        ),
    }
    if mirror_transform is not None:
        safety["mirroringTransform"] = mirror_transform
    return ActionDefinition(
        actionCode=action_code,
        executionMode=template.execution_mode,
        availability=availability,
        policyRef=policy_ref,
        unavailableReason=unavailable_reason,
        parameterSchema=template.parameter_schema,
        completion=template.completion,
        lease=template.lease,
        preconditions={
            "allowedTerrains": [spec.qualification_terrain],
            "requiredCapabilities": list(spec.required_capabilities),
            "scenarioFields": list(spec.scenario_fields),
            "scenarioProfile": spec.scenario_profile,
            "resetProfile": spec.reset_profile,
        },
        safety=safety,
        qualificationRefs=qualification_refs,
    )


def validate_action_definition_envelope(action: ActionDefinition) -> None:
    """Reject any manifest-owned widening or mutation of the V1 action envelope."""
    if (
        action.availability == "UNAVAILABLE"
        and action.unavailableReason not in _CODE_OWNED_UNAVAILABLE_REASONS
    ):
        raise ValueError("action does not match the code-owned V1 action envelope")
    if action.qualificationRefs not in (None, ["qualification/qualification-v1.json"]):
        raise ValueError("action does not match the code-owned V1 action envelope")
    expected = code_owned_action_definition(
        action.actionCode,
        availability=action.availability,
        policy_ref=action.policyRef,
        unavailable_reason=action.unavailableReason,
        qualification_refs=action.qualificationRefs,
    )
    if canonical_json(action) != canonical_json(expected):
        raise ValueError("action does not match the code-owned V1 action envelope")


def validate_bundle_action_envelope(bundle: PolicyBundle) -> None:
    """Require the complete ordered V1 catalog and code-owned policy-family bindings."""
    codes = tuple(action.actionCode for action in bundle.actions)
    if codes != CODE_OWNED_ACTION_CODES:
        raise ValueError(
            "bundle does not contain the complete code-owned V1 action catalog"
        )
    policies = {policy.policyRef: policy for policy in bundle.policies}
    actions_by_ref: dict[str, list[str]] = {}
    for action in bundle.actions:
        validate_action_definition_envelope(action)
        if action.availability == "AVAILABLE":
            from .action_specs import ACTION_RUNTIME_SPECS

            if not ACTION_RUNTIME_SPECS[action.actionCode].supported:
                raise ValueError(
                    "unsupported action cannot widen code-owned availability"
                )
        if action.policyRef is None:
            continue
        policy = policies.get(action.policyRef)
        if policy is None:
            raise ValueError("action policy binding is not declared")
        if policy.taskId not in action_template(action.actionCode).task_ids and not (
            action.availability == "UNAVAILABLE"
            and action.unavailableReason == "POLICY_TASK_ID_MISMATCH"
        ):
            raise ValueError(
                "action policy identity is outside its code-owned task family"
            )
        actions_by_ref.setdefault(action.policyRef, []).append(action.actionCode)
    allowed_shared_sets = {
        frozenset({"SIT", "STAND"}),
        frozenset({"KICK_LEFT", "KICK_RIGHT"}),
    }
    for shared_codes in actions_by_ref.values():
        if len(shared_codes) > 1 and frozenset(shared_codes) not in allowed_shared_sets:
            raise ValueError("policy sharing is outside code-owned identity rules")


def validate_code_owned_parameters(
    action_code: str, parameters: Mapping[str, object]
) -> None:
    """Validate typed intent against code bytes, independently of a manifest copy."""
    schema = action_template(action_code).parameter_schema
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    if set(parameters) != set(properties) or not required.issubset(parameters):
        raise ValueError("parameters violate code-owned action bounds")
    for name, property_schema in properties.items():
        value = parameters[name]
        if (
            not isinstance(value, int | float)
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            raise ValueError("parameters violate code-owned action bounds")
        numeric = float(value)
        if not (
            float(property_schema["minimum"])
            <= numeric
            <= float(property_schema["maximum"])
        ):
            raise ValueError("parameters violate code-owned action bounds")


def validate_code_owned_lease(action_code: str, lease_ms: int | None) -> None:
    """Validate task leases against the installed code envelope, not bundle data."""
    template = action_template(action_code)
    if template.execution_mode == "DISCRETE":
        if lease_ms is not None:
            raise ValueError("lease violates code-owned action bounds")
        return
    lease = template.lease
    if (
        lease is None
        or not isinstance(lease_ms, int)
        or isinstance(lease_ms, bool)
        or not lease.minLeaseMs <= lease_ms <= lease.maxLeaseMs
        or lease_ms < lease.commandCadenceMs
    ):
        raise ValueError("lease violates code-owned action bounds")

from __future__ import annotations

from pathlib import Path

import pytest

import mjlab_microduck.rom.runtime_identity as runtime_identity_module
from mjlab_microduck.rom.action_catalog import (
    CODE_OWNED_ACTION_CODES,
    code_owned_action_definition,
    validate_action_definition_envelope,
)
from mjlab_microduck.rom.runtime_identity import (
    GOVERNED_RUNTIME_MODULES,
    runtime_revision,
)

EXPECTED_ACTION_CODES = (
    "WALK_VELOCITY",
    "VELSTAND_VELOCITY",
    "ROLLER_VELOCITY",
    "SWIZZLE",
    "ROLLER_SLOPE",
    "STAND_UP",
    "SIT",
    "STAND",
    "GROUND_PICK",
    "KICK_LEFT",
    "KICK_RIGHT",
    "ROULADE",
    "ROLLER_CROUCH",
    "ROLLER_STAND_UP",
    "SPIN",
)


def test_v1_envelope_covers_every_action_once_with_literal_walk_safety() -> None:
    """Dropping an intent or weakening WALK safety must change executable behavior."""
    assert CODE_OWNED_ACTION_CODES == EXPECTED_ACTION_CODES

    walk = code_owned_action_definition(
        "WALK_VELOCITY",
        availability="AVAILABLE",
        policy_ref="walk-policy",
    )

    assert walk.model_dump(mode="json", by_alias=True, exclude_none=True) == {
        "actionCode": "WALK_VELOCITY",
        "executionMode": "CONTINUOUS_LEASE",
        "availability": "AVAILABLE",
        "policyRef": "walk-policy",
        "parameterSchema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "vxMps": {"type": "number", "minimum": -0.4, "maximum": 0.4},
                "vyMps": {"type": "number", "minimum": -0.3, "maximum": 0.3},
                "yawRateRadps": {
                    "type": "number",
                    "minimum": -1.0,
                    "maximum": 1.0,
                },
            },
            "required": ["vxMps", "vyMps", "yawRateRadps"],
        },
        "lease": {
            "minLeaseMs": 100,
            "defaultLeaseMs": 500,
            "maxLeaseMs": 5000,
            "commandCadenceMs": 50,
            "zeroCommand": {"vxMps": 0.0, "vyMps": 0.0, "yawRateRadps": 0.0},
            "safeStopBehavior": "ZERO_TWIST",
        },
        "preconditions": {
            "allowedTerrains": ["flat"],
            "requiredCapabilities": ["FLAT_TERRAIN"],
            "scenarioFields": ["terrain", "seed"],
            "scenarioProfile": "SEEDED_SERVO_RESET_V1",
            "resetProfile": "DEFAULT_STANDING",
        },
        "safety": {
            "commandProfile": "TWIST_VELOCITY",
            "fallPolicy": "FAIL_ON_FALL",
            "completionProfile": None,
            "mirroringRule": "NONE",
            "safeStopBehavior": "ZERO_TWIST",
        },
    }


@pytest.mark.parametrize(
    "mutation",
    [
        {"parameterSchema": {"type": "object", "additionalProperties": True}},
        "lease",
        {"preconditions": {"allowedTerrains": ["any"]}},
        {"safety": {"fallPolicy": "IGNORE"}},
    ],
)
def test_v1_envelope_rejects_manifest_owned_safety_mutations(mutation) -> None:
    """A re-signed manifest must not become the authority for action safety fields."""
    walk = code_owned_action_definition(
        "WALK_VELOCITY",
        availability="AVAILABLE",
        policy_ref="walk-policy",
    )
    if mutation == "lease":
        assert walk.lease is not None
        mutation = {
            "lease": walk.lease.model_copy(
                update={
                    "maxLeaseMs": 1_000_000,
                    "zeroCommand": {
                        "vxMps": 1.0,
                        "vyMps": 0.0,
                        "yawRateRadps": 0.0,
                    },
                }
            )
        }
    walk = walk.model_copy(update=mutation)

    with pytest.raises(ValueError, match="code-owned V1 action envelope"):
        validate_action_definition_envelope(walk)


EXPECTED_GOVERNED_RUNTIME_MODULES = (
    "__init__.py",
    "rom/__init__.py",
    "rom/action_catalog.py",
    "rom/action_specs.py",
    "rom/api.py",
    "rom/bundle.py",
    "rom/contracts.py",
    "rom/main.py",
    "rom/mirroring.py",
    "rom/model_semantics.py",
    "rom/mujoco_runtime.py",
    "rom/observation.py",
    "rom/onnx_policy.py",
    "rom/qualification.py",
    "rom/process_protocol.py",
    "rom/process_service.py",
    "rom/process_supervisor.py",
    "rom/runtime_child.py",
    "rom/parent_death.py",
    "rom/runtime.py",
    "rom/runtime_identity.py",
    "rom/service.py",
    "rom/store.py",
    "rom/supervisor_state.py",
)


@pytest.mark.parametrize("module_name", EXPECTED_GOVERNED_RUNTIME_MODULES)
def test_runtime_revision_audits_and_changes_with_every_governed_module(
    monkeypatch: pytest.MonkeyPatch, module_name: str
) -> None:
    """Omitting any qualification/safety module would leave releases bound to stale code."""
    assert GOVERNED_RUNTIME_MODULES == EXPECTED_GOVERNED_RUNTIME_MODULES
    original = runtime_identity_module.Path.read_bytes
    baseline = runtime_revision()

    def changed_bytes(path):
        content = original(path)
        relative_suffix = ("mjlab_microduck", *Path(module_name).parts)
        return (
            content + b"\n# governed mutation\n"
            if path.parts[-len(relative_suffix) :] == relative_suffix
            else content
        )

    monkeypatch.setattr(runtime_identity_module.Path, "read_bytes", changed_bytes)

    assert runtime_revision() != baseline

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mjlab_microduck.rom.contracts import (
    ActionDefinition,
    LeaseContract,
    TaskCreateRequest,
    canonical_json,
    sha256_prefixed,
)


def test_available_action_requires_policy_artifact():
    """Removing the availability/policy binding must reject an unsafe catalog entry."""
    with pytest.raises(ValidationError, match="available action requires policyRef"):
        ActionDefinition(
            actionCode="WALK_VELOCITY",
            executionMode="CONTINUOUS_LEASE",
            availability="AVAILABLE",
            parameterSchema={},
            policyRef=None,
        )


def test_continuous_action_requires_lease_contract():
    """Removing the target-side deadman contract must reject continuous motion."""
    with pytest.raises(ValidationError, match="continuous action requires lease contract"):
        ActionDefinition(
            actionCode="WALK_VELOCITY",
            executionMode="CONTINUOUS_LEASE",
            availability="AVAILABLE",
            parameterSchema={},
            policyRef="walk-policy",
        )


def test_task_rejects_raw_joint_intent():
    """Removing recursive raw-control validation would expose servo targets to ROM."""
    with pytest.raises(ValidationError, match="raw control key is not permitted"):
        TaskCreateRequest.model_validate(
            {
                "schema": "MICRODUCK_SIM_TASK_V1",
                "taskId": "0" * 32,
                "actionCode": "STAND",
                "bundleVersion": "1.0.0",
                "bundleDigest": "sha256:" + "a" * 64,
                "parameters": {"jointTargets": [0] * 14},
                "scenario": {"terrain": "flat", "seed": 1},
                "requestedBy": "execution-1",
            }
        )


@pytest.mark.parametrize("raw_key", ["torque", "PWM", "policyPath", "policyName"])
def test_task_rejects_raw_control_keys_at_every_parameter_depth(raw_key: str):
    """Removing nested inspection would permit a raw control key below a typed envelope."""
    with pytest.raises(ValidationError, match="raw control key is not permitted"):
        TaskCreateRequest.model_validate(
            {
                "schema": "MICRODUCK_SIM_TASK_V1",
                "taskId": "1" * 32,
                "actionCode": "STAND",
                "bundleVersion": "1.0.0",
                "bundleDigest": "sha256:" + "b" * 64,
                "parameters": {"nested": [{raw_key: 0}]},
                "scenario": {"terrain": "flat", "seed": 1},
                "requestedBy": "execution-1",
            }
        )


def test_task_id_and_digest_are_lowercase_fixed_width_values():
    """Weakening identity patterns would make canonical task and bundle binding ambiguous."""
    payload = {
        "schema": "MICRODUCK_SIM_TASK_V1",
        "taskId": "A" * 32,
        "actionCode": "STAND",
        "bundleVersion": "1.0.0",
        "bundleDigest": "sha256:" + "A" * 64,
        "parameters": {},
        "scenario": {"terrain": "flat", "seed": 1},
        "requestedBy": "execution-1",
    }

    with pytest.raises(ValidationError):
        TaskCreateRequest.model_validate(payload)


def test_canonical_digest_ignores_mapping_insertion_order():
    """Dropping canonical key sorting would make a task/bundle hash depend on construction order."""
    assert sha256_prefixed({"b": 2, "a": 1}) == sha256_prefixed({"a": 1, "b": 2})
    assert canonical_json({"b": 2, "a": 1}) == b'{"a":1,"b":2}'


def test_canonical_json_normalizes_nested_models():
    """Failing to normalize embedded models would make manifest digest construction non-serializable."""
    lease = LeaseContract(minLeaseMs=100, defaultLeaseMs=200, maxLeaseMs=500, commandCadenceMs=50)
    assert canonical_json({"lease": lease}) == (
        b'{"lease":{"commandCadenceMs":50,"defaultLeaseMs":200,"maxLeaseMs":500,"minLeaseMs":100}}'
    )

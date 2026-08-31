from __future__ import annotations

import argparse
import hashlib
import json
import os
import select
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from tests.test_rom_mujoco_runtime import (
    _rewrite_as_stand_bundle,
    _write_verified_bundle,
)

from mjlab_microduck.rom.action_catalog import CODE_OWNED_ACTION_CODES
from mjlab_microduck.rom.contracts import (
    CONTROLLED_SERVO_JOINTS,
    OBSERVATION_FIELDS,
    PolicyBundle,
    canonical_json,
    publish_policy_bundle,
    sha256_prefixed,
    unsigned_policy_bundle_manifest,
)
from mjlab_microduck.rom.main import (
    create_configured_app,
    load_qualified_bundle,
    load_verified_bundle,
)
from mjlab_microduck.rom.qualification import (
    ActionQualificationConfig,
    QualificationReport,
    QualificationThresholds,
    ReleaseConfiguration,
    qualify_and_promote,
)
from mjlab_microduck.rom.runtime_identity import runtime_revision

NOW = datetime(2026, 8, 29, 12, 0, tzinfo=UTC)
TOKEN = "handoff-test-token"
AUTHORITATIVE_ASSETS = (
    "microduck-policy-bundle-v1.schema.json",
    "microduck-simulator-api-v1.openapi.yaml",
    "microduck-v1-portability-fixtures.json",
)
PROCESS_FAILURE_TRANSCRIPT = "transcripts/process-child-exit-reap.json"
FORCED_KILL_LEASE_MS = 5_000
SHORT_LEASE_MS = 200


def digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_bytes(canonical_json(value))


def require_distribution_cleared(bundle: PolicyBundle) -> None:
    if bundle.license.modelAssets.distributionStatus != "DISTRIBUTION_CLEARED":
        raise ValueError("model assets are not cleared for distribution handoff")


def _tree_files(root: Path) -> dict[str, Path]:
    return {
        path.relative_to(root).as_posix(): path
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def assert_identical_trees(first: Path, second: Path) -> None:
    """Require two complete handoff trees to have identical paths and bytes."""
    first_files = _tree_files(first)
    second_files = _tree_files(second)
    all_paths = sorted(set(first_files) | set(second_files))
    for relative in all_paths:
        assert relative in first_files, f"only second tree contains {relative}"
        assert relative in second_files, f"only first tree contains {relative}"
        assert (
            first_files[relative].read_bytes() == second_files[relative].read_bytes()
        ), f"handoff bytes differ for {relative}"


def publish_reproducible_tree(first: Path, second: Path, output: Path) -> None:
    """Publish only after both independently generated complete trees match."""
    assert_identical_trees(first, second)
    if output.exists():
        assert output.is_dir(), f"handoff output is not a directory: {output}"
        shutil.rmtree(output)
    shutil.copytree(first, output)


def copy_authoritative_assets(repo: Path, handoff: Path) -> None:
    schemas = handoff / "schemas"
    schemas.mkdir(parents=True, exist_ok=True)
    for name in AUTHORITATIVE_ASSETS:
        shutil.copyfile(repo / "schemas" / name, schemas / name)


def assert_authoritative_assets(repo: Path, handoff: Path) -> None:
    for name in AUTHORITATIVE_ASSETS:
        authoritative = repo / "schemas" / name
        generated = handoff / "schemas" / name
        assert generated.read_bytes() == authoritative.read_bytes(), name
        assert digest(generated) == digest(authoritative), name


def assert_sanitized_transcript(
    transcript: object, *, forbidden_tokens: tuple[str, ...]
) -> None:
    """Reject secrets, filesystem paths, and process identifiers from evidence."""

    def visit(value: object, key: str | None = None) -> None:
        if key is not None:
            normalized = key.lower().replace("_", "").replace("-", "")
            assert not normalized.endswith("pid"), f"process identifier key: {key}"
            assert normalized not in {"processid", "childprocessid"}, key
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                visit(child_value, str(child_key))
        elif isinstance(value, (list, tuple)):
            for child_value in value:
                visit(child_value)
        elif isinstance(value, str):
            assert not value.startswith(("/", "~/")), f"absolute path: {value}"
            assert not (len(value) > 2 and value[1:3] in {":\\", ":/"}), value
            for token in forbidden_tokens:
                assert token not in value, "secret token present"

    visit(transcript)


def build_handoff_manifest(
    handoff: Path,
    *,
    source_commit: str,
    runtime_revision: str,
    bundle_digest: str,
) -> dict[str, object]:
    files: dict[str, dict[str, object]] = {}
    for relative, path in _tree_files(handoff).items():
        if relative == "handoff.json":
            continue
        files[relative] = {
            "bytes": path.stat().st_size,
            "sha256": digest(path),
        }
    assert PROCESS_FAILURE_TRANSCRIPT in files, "process failure transcript is missing"
    for name in AUTHORITATIVE_ASSETS:
        assert f"schemas/{name}" in files, f"authoritative asset is missing: {name}"
    return {
        "schema": "MICRODUCK_ROM_HANDOFF_V1",
        "generatedAt": "2026-08-29T12:00:00Z",
        "sourceRepository": "microduck-rl",
        "sourceCommit": source_commit,
        "runtimeRevision": runtime_revision,
        "qualifiedBundleDigest": bundle_digest,
        "processFailureTranscript": PROCESS_FAILURE_TRANSCRIPT,
        "files": files,
        "limitations": [
            "The deterministic bundle contains non-production test WALK_VELOCITY and STAND policies; it is not a production checkpoint.",
            "STAND is the only governed discrete action demonstrated as AVAILABLE and PASSED in this fixture.",
            "All 13 actions without fixture policies remain explicitly UNAVAILABLE with a code-owned reason; SPIN is demonstrated as rejected by the API.",
            "The bundle uses a minimal test MJCF and contains no production STL, .part, robot-source, training, environment, or checkpoint asset.",
            "A child RUNTIME_UNRESPONSIVE or unexpected exit is quarantined and exactly reaped before a fresh runtime generation may accept motion; a whole process/container restart is not required for child replacement.",
        ],
    }


def exact_head_commit(repo: Path) -> str:
    unstaged = subprocess.run(
        ["git", "diff", "--quiet", "--"], cwd=repo, check=False
    )
    assert unstaged.returncode == 0, "tracked worktree differs from HEAD"
    staged = subprocess.run(
        ["git", "diff", "--cached", "--quiet", "--"], cwd=repo, check=False
    )
    assert staged.returncode == 0, "index differs from HEAD"
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout
    assert not untracked, "untracked non-ignored files are outside exact HEAD"
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def build_process_failure_transcript(
    *,
    source_commit: str,
    runtime_revision: str,
    failed_generation: int,
    failure_trace: tuple[str, ...],
    failure_events: list[dict[str, object]],
    fresh_generation: int,
    fresh_events: list[dict[str, object]],
) -> dict[str, object]:
    required_trace = (
        "OPERATION_FAILED",
        "QUARANTINED",
        "CHILD_REAPED",
        "NO_CHILD",
    )
    positions = [failure_trace.index(item) for item in required_trace]
    assert positions == sorted(positions)
    assert fresh_generation > failed_generation
    return {
        "schema": "MICRODUCK_ROM_SANITIZED_TRANSCRIPT_V1",
        "sourceCommit": source_commit,
        "runtimeRevision": runtime_revision,
        "authorization": "Bearer <redacted>",
        "processFailure": {
            "generation": failed_generation,
            "startAcknowledged": True,
            "exactChildIdentityCaptured": True,
            "signal": "SIGKILL",
            "exitObservedOnOwnedHandle": True,
            "reapConfirmed": True,
            "noUnrelatedRuntimeOperationRequired": True,
            "supervisorTrace": list(failure_trace),
            "events": failure_events,
            "durableState": "FAILED",
            "stopReason": "RUNTIME_UNRESPONSIVE",
        },
        "freshTask": {
            "generation": fresh_generation,
            "generationAdvanced": True,
            "acceptedOnlyAfterReap": True,
            "events": fresh_events,
            "terminalState": "TIMED_OUT",
            "stopReason": "LEASE_EXPIRED",
        },
    }


def wait_for(client: TestClient, task_id: str, state: str) -> dict[str, object]:
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        response = client.get(f"/v1/tasks/{task_id}", headers=client.headers)
        snapshot = response.json()
        if snapshot["state"] == state:
            return snapshot
        time.sleep(0.02)
    raise AssertionError(f"task {task_id} did not reach {state}: {snapshot}")


def wait_ready(client: TestClient) -> dict[str, object]:
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        response = client.get("/v1/ready", headers=client.headers)
        body = response.json()
        if response.status_code == 200 and body["ready"] is True:
            return body
        time.sleep(0.01)
    raise AssertionError(f"simulator did not reopen its motion slot: {body}")


def create_when_motion_slot_reopens(
    client: TestClient, request: dict[str, object]
):
    """Retry only fail-closed readiness while exact reap/delivery completes."""
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        response = client.post("/v1/tasks", json=request)
        if response.status_code == 202:
            return response
        assert response.status_code == 503 and response.json()["code"] == "NOT_READY"
        time.sleep(0.01)
    raise AssertionError("motion slot did not reopen for the next transcript task")


def events(db: Path, task_id: str) -> list[dict[str, object]]:
    connection = sqlite3.connect(db)
    try:
        rows = connection.execute(
            "select sequence,event_type,payload_json from task_event "
            "where task_id=? order by sequence",
            (task_id,),
        )
        return [
            {
                "sequence": sequence,
                "eventType": event_type,
                "payload": json.loads(payload),
            }
            for sequence, event_type, payload in rows
        ]
    finally:
        connection.close()


def durable_state(db: Path, task_id: str) -> tuple[str, str | None] | None:
    connection = sqlite3.connect(db)
    try:
        row = connection.execute(
            "select state,stop_reason from task where task_id=?", (task_id,)
        ).fetchone()
        if row is None:
            return None
        return str(row[0]), None if row[1] is None else str(row[1])
    finally:
        connection.close()


def record_process_failure_transcript(
    *,
    handoff: Path,
    installed: Path,
    bundle: PolicyBundle,
    commit: str,
    revision: str,
    stage: Path,
) -> None:
    """Kill one acknowledged child and record exact-reap/fresh-child evidence."""
    state_db = stage / "process-failure-state/tasks.sqlite"
    state_db.parent.mkdir()
    environment = {
        "MICRODUCK_ROM_BUNDLE_DIR": str(installed),
        "MICRODUCK_ROM_STATE_DB": str(state_db),
        "MICRODUCK_ROM_BEARER_TOKEN": TOKEN,
    }
    headers = {"Authorization": f"Bearer {TOKEN}"}
    failed_task_id = "5" * 32
    fresh_task_id = "6" * 32
    failed_request = {
        "schema": "MICRODUCK_SIM_TASK_V1",
        "taskId": failed_task_id,
        "actionCode": "WALK_VELOCITY",
        "bundleVersion": bundle.bundleVersion,
        "bundleDigest": bundle.bundleDigest,
        "parameters": {"vxMps": 0.0, "vyMps": 0.0, "yawRateRadps": 0.0},
        "scenario": {"terrain": "flat", "seed": 7},
        "leaseMs": FORCED_KILL_LEASE_MS,
        "requestedBy": "handoff-process-failure-smoke",
    }
    app = create_configured_app(environment)
    with TestClient(app, headers=headers) as client:
        service = app.state.task_service
        assert service is not None
        supervisor = service._supervisor
        trace_start = len(supervisor.trace)
        created = client.post("/v1/tasks", json=failed_request)
        assert created.status_code == 202
        wait_for(client, failed_task_id, "RUNNING")
        acknowledged = supervisor.snapshot()
        assert acknowledged.state.value == "RUNNING"
        assert acknowledged.pid is not None
        failed_generation = acknowledged.generation
        child_identity = acknowledged.pid
        owned_handle = os.pidfd_open(child_identity)
        try:
            signal.pidfd_send_signal(owned_handle, signal.SIGKILL)
            readable, _, _ = select.select([owned_handle], [], [], 10.0)
            assert readable, "owned child exit was not observed"
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                snapshot = supervisor.snapshot()
                if snapshot.state.value == "NO_CHILD" and snapshot.slot_releasable:
                    break
                time.sleep(0.01)
            else:
                raise AssertionError("supervisor did not reap the exited child")
            while time.monotonic() < deadline:
                state = durable_state(state_db, failed_task_id)
                if state == ("FAILED", "RUNTIME_UNRESPONSIVE"):
                    break
                time.sleep(0.01)
            else:
                raise AssertionError("child exit did not become a durable failure")
            try:
                os.waitpid(child_identity, os.WNOHANG)
            except ChildProcessError:
                pass
            else:
                raise AssertionError("the exact child was not already reaped")
        finally:
            os.close(owned_handle)

        raw_trace = supervisor.trace[trace_start:]
        failure_start = raw_trace.index("OPERATION_FAILED")
        failure_end = raw_trace.index("NO_CHILD", failure_start) + 1
        failure_trace = tuple(raw_trace[failure_start:failure_end])
        failure_events = events(state_db, failed_task_id)
        assert [item["eventType"] for item in failure_events] == [
            "TASK_VALIDATING",
            "TASK_STARTED",
            "TASK_FAILED",
        ]

        fresh_request = failed_request | {
            "taskId": fresh_task_id,
            "leaseMs": SHORT_LEASE_MS,
            "requestedBy": "handoff-fresh-generation-smoke",
        }
        fresh_created = client.post("/v1/tasks", json=fresh_request)
        assert fresh_created.status_code == 202
        wait_for(client, fresh_task_id, "RUNNING")
        fresh_generation = supervisor.snapshot().generation
        fresh_terminal = wait_for(client, fresh_task_id, "TIMED_OUT")
        assert fresh_terminal["stopReason"] == "LEASE_EXPIRED"
        fresh_events = events(state_db, fresh_task_id)
        assert [item["eventType"] for item in fresh_events] == [
            "TASK_VALIDATING",
            "TASK_STARTED",
            "TASK_TIMED_OUT",
        ]

    transcript = build_process_failure_transcript(
        source_commit=commit,
        runtime_revision=revision,
        failed_generation=failed_generation,
        failure_trace=failure_trace,
        failure_events=failure_events,
        fresh_generation=fresh_generation,
        fresh_events=fresh_events,
    )
    assert_sanitized_transcript(
        transcript,
        forbidden_tokens=(TOKEN, str(stage), str(installed)),
    )
    write_json(handoff / PROCESS_FAILURE_TRANSCRIPT, transcript)


def assert_authoritative_policy_bundle_schema(repo: Path, handoff: Path) -> None:
    """Prove the handoff copied the portable checked-in schema without drift."""
    authoritative = repo / "schemas/microduck-policy-bundle-v1.schema.json"
    generated = handoff / "schemas/microduck-policy-bundle-v1.schema.json"
    assert generated.read_bytes() == authoritative.read_bytes()
    assert digest(generated) == digest(authoritative)
    schema = json.loads(generated.read_text())
    assert "bundleDigest" in schema["required"]
    observation_fields = schema["$defs"]["ObservationContract"]["properties"]["fields"]
    assert observation_fields["minItems"] == observation_fields["maxItems"] == 61
    assert [item["const"] for item in observation_fields["prefixItems"]] == list(
        OBSERVATION_FIELDS
    )
    action_joints = schema["$defs"]["ActionContract"]["properties"]["joints"]
    assert action_joints["minItems"] == action_joints["maxItems"] == 14
    assert [item["const"] for item in action_joints["prefixItems"]] == list(
        CONTROLLED_SERVO_JOINTS
    )
    assert schema["$defs"]["ActionDefinition"]["allOf"] == [
        {
            "if": {
                "properties": {"availability": {"const": "AVAILABLE"}},
                "required": ["availability"],
            },
            "then": {
                "required": ["policyRef"],
                "properties": {"policyRef": {"type": "string", "minLength": 1}},
            },
        },
        {
            "if": {
                "properties": {"executionMode": {"const": "CONTINUOUS_LEASE"}},
                "required": ["executionMode"],
            },
            "then": {
                "required": ["lease"],
                "properties": {"lease": {"not": {"type": "null"}}},
            },
        },
    ]


def generate_handoff(
    *, repo: Path, commit: str, stage: Path, handoff: Path
) -> tuple[str, str, str]:
    stage.mkdir(parents=True)
    assert not handoff.exists(), f"handoff output already exists: {handoff}"
    revision = runtime_revision()
    candidate_root = stage / "candidate"
    walk_root = stage / "walk"
    stand_root = stage / "stand"

    walk = _write_verified_bundle(
        walk_root,
        metadata_overrides={"microduck.source_commit": commit},
        weld_trunk=True,
    )
    stand_source = _write_verified_bundle(
        stand_root,
        policy_output=np.zeros(14, dtype=np.float32),
        action_code="STAND",
        task_id="Mjlab-SitStand-Flat-MicroDuck",
        metadata_overrides={"microduck.source_commit": commit},
    )
    stand = _rewrite_as_stand_bundle(stand_root, stand_source)
    shutil.copytree(walk_root, candidate_root)
    stand_policy_path = candidate_root / "policies/stand.onnx"
    shutil.copy2(stand_root / stand.policies[0].path, stand_policy_path)
    stand_policy = stand.policies[0].model_copy(
        update={
            "policyRef": "stand-policy",
            "path": "policies/stand.onnx",
            "digest": digest(stand_policy_path),
        }
    )
    stand_action = next(
        action for action in stand.actions if action.actionCode == "STAND"
    ).model_copy(update={"policyRef": stand_policy.policyRef})
    combined_actions = [
        (stand_action if action.actionCode == "STAND" else action)
        for action in walk.actions
    ]
    unsigned = unsigned_policy_bundle_manifest(walk).model_copy(
        update={
            "sourceCommit": commit,
            "policies": [walk.policies[0], stand_policy],
            "actions": combined_actions,
        }
    )
    candidate_artifacts = {unsigned.model.path: unsigned.model.digest}
    candidate_artifacts.update(
        {policy.path: policy.digest for policy in unsigned.policies}
    )
    candidate_artifacts.update(
        {
            item["path"]: item["digest"]
            for item in unsigned.qualification.get("modelClosure", [])
        }
    )
    candidate_artifacts.update(
        {item.path: item.digest for item in unsigned.license.artifacts}
    )
    candidate = publish_policy_bundle(unsigned, candidate_artifacts)
    (candidate_root / "microduck-policy-bundle.json").write_bytes(
        canonical_json(candidate)
    )
    candidate = load_verified_bundle(candidate_root)

    common = {
        "minSuccessRate": 1.0,
        "maxFallRate": 0.0,
        "maxMeanTrackingError": 10.0,
        "minMeanDistanceM": 0.0,
        "maxMeanEnergyProxy": 10_000.0,
        "maxActuatorClampSteps": 100,
        "maxPhysicalJointLimitViolations": 0,
    }
    configuration = ReleaseConfiguration(
        release="1.0.1",
        createdAt=NOW,
        actions=(
            ActionQualificationConfig(
                actionCode="WALK_VELOCITY",
                mandatory=True,
                terrain="flat",
                resetProfile="DEFAULT_STANDING",
                seeds=(7, 11, 29),
                maxSteps=100,
                parameters={"vxMps": 0.1, "vyMps": 0.0, "yawRateRadps": 0.0},
                thresholds=QualificationThresholds(
                    **common,
                    actionMetric="trackingError",
                    actionMetricOperator="lte",
                    actionMetricThreshold=10.0,
                ),
            ),
            ActionQualificationConfig(
                actionCode="STAND",
                mandatory=True,
                terrain="flat",
                resetProfile="TRAINED_SITTING",
                seeds=(7, 11, 29),
                maxSteps=100,
                parameters={},
                thresholds=QualificationThresholds(
                    **common,
                    actionMetric="standPoseError",
                    actionMetricOperator="lte",
                    actionMetricThreshold=0.08,
                ),
            ),
        ),
    )
    first_zip = stage / "first.zip"
    second_zip = stage / "second.zip"
    promoted = qualify_and_promote(
        candidate_root, first_zip, configuration, timestamp=lambda: NOW
    )
    repeated = qualify_and_promote(
        candidate_root, second_zip, configuration, timestamp=lambda: NOW
    )
    assert first_zip.read_bytes() == second_zip.read_bytes()
    assert promoted.manifest == repeated.manifest

    installed = stage / "installed"
    with zipfile.ZipFile(first_zip) as archive:
        archive.extractall(installed)
    bundle = load_qualified_bundle(installed)
    require_distribution_cleared(bundle)
    assert (
        tuple(action.actionCode for action in bundle.actions) == CODE_OWNED_ACTION_CODES
    )
    assert len(bundle.actions) == 15
    assert {
        action.actionCode
        for action in bundle.actions
        if action.availability == "AVAILABLE"
    } == {"WALK_VELOCITY", "STAND"}
    assert all(
        action.unavailableReason
        for action in bundle.actions
        if action.availability == "UNAVAILABLE"
    )

    for relative in ("fixture", "schemas", "transcripts"):
        (handoff / relative).mkdir(parents=True)
    fixture_zip = handoff / "fixture/microduck-qualified-minimal-1.0.1.zip"
    shutil.copy2(first_zip, fixture_zip)
    copy_authoritative_assets(repo, handoff)
    write_json(
        handoff / "schemas/microduck-rom-qualification-v1.schema.json",
        QualificationReport.model_json_schema(by_alias=True),
    )
    write_json(
        handoff / "schemas/microduck-rom-release-v1.schema.json",
        ReleaseConfiguration.model_json_schema(by_alias=True),
    )
    assert_authoritative_assets(repo, handoff)
    assert_authoritative_policy_bundle_schema(repo, handoff)

    headers = {"Authorization": f"Bearer {TOKEN}"}
    state_db = stage / "api-state/tasks.sqlite"
    state_db.parent.mkdir()
    environment = {
        "MICRODUCK_ROM_BUNDLE_DIR": str(installed),
        "MICRODUCK_ROM_STATE_DB": str(state_db),
        "MICRODUCK_ROM_BEARER_TOKEN": TOKEN,
    }
    with TestClient(create_configured_app(environment), headers=headers) as client:
        ready_response = client.get("/v1/ready")
        assert ready_response.status_code == 200 and ready_response.json()["ready"]

        walk_request = {
            "schema": "MICRODUCK_SIM_TASK_V1",
            "taskId": "1" * 32,
            "actionCode": "WALK_VELOCITY",
            "bundleVersion": bundle.bundleVersion,
            "bundleDigest": bundle.bundleDigest,
            "parameters": {"vxMps": 0.0, "vyMps": 0.0, "yawRateRadps": 0.0},
            "scenario": {"terrain": "flat", "seed": 7},
            "leaseMs": 200,
            "requestedBy": "handoff-smoke",
        }
        created = client.post("/v1/tasks", json=walk_request)
        assert created.status_code == 202
        walk_terminal = wait_for(client, "1" * 32, "TIMED_OUT")
        walk_events = events(state_db, "1" * 32)
        assert [item["eventType"] for item in walk_events] == [
            "TASK_VALIDATING",
            "TASK_STARTED",
            "TASK_TIMED_OUT",
        ]
        write_json(
            handoff / "transcripts/continuous-lease-timeout.json",
            {
                "schema": "MICRODUCK_ROM_SANITIZED_TRANSCRIPT_V1",
                "sourceCommit": commit,
                "authorization": "Bearer <redacted>",
                "ready": {
                    "httpStatus": ready_response.status_code,
                    "body": ready_response.json(),
                },
                "create": {
                    "httpStatus": created.status_code,
                    "request": walk_request,
                    "responseState": created.json()["state"],
                },
                "events": walk_events,
                "renewal": "INTENTIONALLY_OMITTED",
                "afterLeaseExpiry": {
                    "state": walk_terminal["state"],
                    "stopReason": walk_terminal["stopReason"],
                    "bundleDigest": walk_terminal["bundleDigest"],
                    "modelDigest": walk_terminal["evidence"]["modelDigest"],
                    "policyDigest": walk_terminal["evidence"]["policyDigest"],
                },
            },
        )

        stand_request = {
            "schema": "MICRODUCK_SIM_TASK_V1",
            "taskId": "2" * 32,
            "actionCode": "STAND",
            "bundleVersion": bundle.bundleVersion,
            "bundleDigest": bundle.bundleDigest,
            "parameters": {},
            "scenario": {"terrain": "flat", "seed": 7},
            "requestedBy": "handoff-smoke",
        }
        created = create_when_motion_slot_reopens(client, stand_request)
        stand_terminal = wait_for(client, "2" * 32, "SUCCEEDED")
        wait_ready(client)
        stand_events = events(state_db, "2" * 32)
        assert [item["eventType"] for item in stand_events] == [
            "TASK_VALIDATING",
            "TASK_STARTED",
            "TASK_SUCCEEDED",
        ]
        write_json(
            handoff / "transcripts/discrete-stand-success.json",
            {
                "schema": "MICRODUCK_ROM_SANITIZED_TRANSCRIPT_V1",
                "sourceCommit": commit,
                "authorization": "Bearer <redacted>",
                "create": {
                    "httpStatus": created.status_code,
                    "request": stand_request,
                    "responseState": created.json()["state"],
                },
                "events": stand_events,
                "terminal": {
                    "state": stand_terminal["state"],
                    "stopReason": stand_terminal["stopReason"],
                    "bundleDigest": stand_terminal["bundleDigest"],
                    "modelDigest": stand_terminal["evidence"]["modelDigest"],
                    "policyDigest": stand_terminal["evidence"]["policyDigest"],
                    "standSettledSteps": stand_terminal["evidence"]["metrics"][
                        "standSettledSteps"
                    ],
                },
            },
        )

        spin_request = {
            "schema": "MICRODUCK_SIM_TASK_V1",
            "taskId": "3" * 32,
            "actionCode": "SPIN",
            "bundleVersion": bundle.bundleVersion,
            "bundleDigest": bundle.bundleDigest,
            "parameters": {},
            "scenario": {"terrain": "flat", "seed": 7},
            "requestedBy": "handoff-smoke",
        }
        rejected = client.post("/v1/tasks", json=spin_request)
        assert rejected.status_code == 400
        write_json(
            handoff / "transcripts/discrete-unavailable.json",
            {
                "schema": "MICRODUCK_ROM_SANITIZED_TRANSCRIPT_V1",
                "sourceCommit": commit,
                "authorization": "Bearer <redacted>",
                "request": spin_request,
                "response": {
                    "httpStatus": rejected.status_code,
                    "body": rejected.json(),
                },
                "catalog": next(
                    action.model_dump(mode="json", by_alias=True, exclude_none=True)
                    for action in bundle.actions
                    if action.actionCode == "SPIN"
                ),
            },
        )

    restart_db = stage / "restart-state/tasks.sqlite"
    restart_db.parent.mkdir()
    restart_environment = environment | {"MICRODUCK_ROM_STATE_DB": str(restart_db)}
    restart_request = walk_request | {
        "taskId": "4" * 32,
        "leaseMs": 200,
        "requestedBy": "handoff-restart-smoke",
    }
    with TestClient(
        create_configured_app(restart_environment), headers=headers
    ) as first_client:
        created = first_client.post("/v1/tasks", json=restart_request)
        assert created.status_code == 202
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if any(
                item["eventType"] == "TASK_STARTED"
                for item in events(restart_db, "4" * 32)
            ):
                break
            time.sleep(0.01)
        else:
            raise AssertionError("restart task did not start")
    with TestClient(
        create_configured_app(restart_environment), headers=headers
    ) as restarted_client:
        restarted = restarted_client.get("/v1/tasks/" + "4" * 32)
        assert restarted.status_code == 200
        restarted_body = restarted.json()
        assert restarted_body["state"] == "UNKNOWN"
    time.sleep(0.25)
    restart_events = events(restart_db, "4" * 32)
    assert restart_events[-1] == {
        "sequence": 2,
        "eventType": "TASK_INTERRUPTED",
        "payload": {"previousState": "RUNNING"},
    }
    write_json(
        handoff / "transcripts/restart-unknown.json",
        {
            "schema": "MICRODUCK_ROM_SANITIZED_TRANSCRIPT_V1",
            "sourceCommit": commit,
            "authorization": "Bearer <redacted>",
            "request": restart_request,
            "events": restart_events,
            "afterRestart": {
                "httpStatus": restarted.status_code,
                "state": restarted_body["state"],
                "stopReason": restarted_body["stopReason"],
                "bundleDigest": restarted_body["bundleDigest"],
            },
        },
    )

    record_process_failure_transcript(
        handoff=handoff,
        installed=installed,
        bundle=bundle,
        commit=commit,
        revision=revision,
        stage=stage,
    )

    artifact_digests = {bundle.model.path: bundle.model.digest}
    artifact_digests.update({policy.path: policy.digest for policy in bundle.policies})
    for group in (
        bundle.qualification.get("artifacts", []),
        bundle.qualification.get("modelClosure", []),
    ):
        artifact_digests.update({item["path"]: item["digest"] for item in group})
    artifact_digests.update(
        {item.path: item.digest for item in bundle.license.artifacts}
    )
    write_json(
        handoff / "fixture/canonical-digests.json",
        {
            "schema": "MICRODUCK_ROM_CANONICAL_DIGEST_FIXTURE_V1",
            "sourceCommit": commit,
            "runtimeRevision": revision,
            "candidateBundleId": candidate.bundleId,
            "candidateBundleVersion": candidate.bundleVersion,
            "candidateBundleDigest": candidate.bundleDigest,
            "promotedBundleId": bundle.bundleId,
            "promotedBundleVersion": bundle.bundleVersion,
            "promotedBundleDigest": bundle.bundleDigest,
            "zipSha256": digest(fixture_zip),
            "deterministicZipRegeneration": True,
            "artifactDigests": artifact_digests,
            "releaseConfigurationCanonical": json.loads(canonical_json(configuration)),
            "releaseConfigurationDigest": sha256_prefixed(configuration),
            "qualificationReportCanonical": json.loads(canonical_json(promoted.report)),
            "qualificationReportDigest": sha256_prefixed(promoted.report),
            "unsignedManifestBinding": (
                "Qualification binds the exact verified candidate bundleDigest; "
                "promotion embeds the canonical subject manifest and hashes the "
                "report, configuration, and subject artifacts into the promoted "
                "bundleDigest, avoiding a circular self-reference."
            ),
            "licenseEvidence": bundle.license,
        },
    )

    manifest = build_handoff_manifest(
        handoff,
        source_commit=commit,
        runtime_revision=revision,
        bundle_digest=bundle.bundleDigest,
    )
    write_json(handoff / "handoff.json", manifest)
    file_records = manifest["files"]
    assert isinstance(file_records, dict)
    for relative, record in file_records.items():
        assert isinstance(relative, str) and isinstance(record, dict)
        path = handoff / relative
        assert path.stat().st_size == record["bytes"]
        assert digest(path) == record["sha256"]
    for transcript_path in (handoff / "transcripts").glob("*.json"):
        assert_sanitized_transcript(
            json.loads(transcript_path.read_bytes()),
            forbidden_tokens=(TOKEN, str(stage), str(installed)),
        )
    assert_authoritative_assets(repo, handoff)
    assert TOKEN.encode() not in b"".join(
        path.read_bytes() for path in handoff.rglob("*") if path.is_file()
    )
    return (
        bundle.bundleDigest,
        digest(fixture_zip),
        digest(handoff / "handoff.json"),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate and byte-verify the complete exact-HEAD ROM handoff twice."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("handoff"),
        help="directory to replace only after both complete trees match",
    )
    args = parser.parse_args(argv)
    repo = REPO
    commit = exact_head_commit(repo)
    with tempfile.TemporaryDirectory(prefix="microduck-task8-handoff-") as temporary:
        root = Path(temporary)
        first_stage = root / "first-stage"
        second_stage = root / "second-stage"
        first = root / "first-handoff"
        second = root / "second-handoff"
        first_result = generate_handoff(
            repo=repo,
            commit=commit,
            stage=first_stage,
            handoff=first,
        )
        second_result = generate_handoff(
            repo=repo,
            commit=commit,
            stage=second_stage,
            handoff=second,
        )
        assert exact_head_commit(repo) == commit, "source HEAD changed during generation"
        assert first_result == second_result
        publish_reproducible_tree(first, second, args.output)
    assert_authoritative_assets(repo, args.output)
    print(args.output)
    for value in first_result:
        print(value)


if __name__ == "__main__":
    main()

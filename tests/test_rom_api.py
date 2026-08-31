"""Boundary tests for the V1 authenticated simulator task API.

Each test names the API regression it is intended to catch; requests use the
real task service and its durable SQLite store rather than a mocked endpoint.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient

from mjlab_microduck.rom.action_catalog import (
    CODE_OWNED_ACTION_CODES,
    code_owned_action_definition,
)
from mjlab_microduck.rom.api import create_app
from mjlab_microduck.rom.contracts import (
    ACTION_CONTRACT,
    OBSERVATION_CONTRACT,
    ActionContract,
    ModelArtifact,
    ObservationContract,
    PolicyArtifact,
    PolicyBundle,
    TaskCreateRequest,
    publish_policy_bundle,
    unsigned_policy_bundle_manifest,
)
from mjlab_microduck.rom.main import (
    UnconfiguredRuntime,
    create_configured_app,
    read_configuration,
)
from mjlab_microduck.rom.service import SimulatorTaskService
from mjlab_microduck.rom.store import SqliteTaskStore
from tests.fakes.fake_microduck_runtime import FakeMicroduckRuntime


@pytest.fixture
def bearer_token() -> str:
    return "test-simulator-token"


@pytest.fixture
def auth(bearer_token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {bearer_token}"}


@pytest.fixture
def service(tmp_path: Path) -> SimulatorTaskService:
    actions = [
        code_owned_action_definition(
            code,
            availability="AVAILABLE" if code == "STAND" else "UNAVAILABLE",
            policy_ref="stand-policy" if code == "STAND" else None,
            unavailable_reason=(None if code == "STAND" else "POLICY_ARTIFACT_MISSING"),
        )
        for code in CODE_OWNED_ACTION_CODES
    ]
    bundle = PolicyBundle(
        schema="MICRODUCK_POLICY_BUNDLE_V1",
        bundleId="org.microduck.test",
        bundleVersion="1.0.0",
        bundleDigest="sha256:" + "a" * 64,
        createdAt=datetime(2026, 8, 29, tzinfo=UTC),
        sourceRepository="microduck-rl",
        sourceCommit="a" * 40,
        robotModel="MICRODUCK",
        observationContract=ObservationContract(
            identifier=OBSERVATION_CONTRACT,
            dimension=61,
            fields=[
                "base_ang_vel.roll",
                "base_ang_vel.pitch",
                "base_ang_vel.yaw",
                "projected_gravity.x",
                "projected_gravity.y",
                "projected_gravity.z",
                *[
                    f"{block}.{joint}"
                    for block in ("joint_pos_rel", "joint_vel_rel", "last_action")
                    for joint in (
                        "left_hip_yaw",
                        "left_hip_roll",
                        "left_hip_pitch",
                        "left_knee",
                        "left_ankle",
                        "neck_pitch",
                        "head_pitch",
                        "head_yaw",
                        "head_roll",
                        "right_hip_yaw",
                        "right_hip_roll",
                        "right_hip_pitch",
                        "right_knee",
                        "right_ankle",
                    )
                ],
                "twist.lin_vel_x",
                "twist.lin_vel_y",
                "twist.ang_vel_z",
                "head_pose.neck_pitch",
                "head_pose.head_pitch",
                "head_pose.head_yaw",
                "head_pose.head_roll",
                "body_pose.x",
                "body_pose.y",
                "body_pose.z",
                "body_pose.roll",
                "body_pose.pitch",
                "body_pose.yaw",
            ],
            units={},
            normalization="BAKED_IN_ONNX",
        ),
        actionContract=ActionContract(
            identifier=ACTION_CONTRACT,
            dimension=14,
            joints=[
                "left_hip_yaw",
                "left_hip_roll",
                "left_hip_pitch",
                "left_knee",
                "left_ankle",
                "neck_pitch",
                "head_pitch",
                "head_yaw",
                "head_roll",
                "right_hip_yaw",
                "right_hip_roll",
                "right_hip_pitch",
                "right_knee",
                "right_ankle",
            ],
            units="rad",
            scaling={},
            clipping={},
        ),
        model=ModelArtifact(path="models/robot.xml", digest="sha256:" + "b" * 64),
        policies=[
            PolicyArtifact(
                policyRef="stand-policy",
                path="policies/stand.onnx",
                digest="sha256:" + "c" * 64,
                taskId="Mjlab-SitStand-Flat-MicroDuck",
                runtimeRequirements={},
            )
        ],
        actions=actions,
        qualification={},
        license={
            "software": {
                "identifier": "Apache-2.0",
                "artifactPaths": ["licenses/LICENSE"],
            },
            "modelAssets": {
                "identifier": "LicenseRef-MicroDuck-Model",
                "distributionStatus": "DISTRIBUTION_CLEARED",
                "artifactPaths": ["licenses/MODEL-LICENSE"],
            },
            "artifacts": [
                {
                    "path": "licenses/LICENSE",
                    "digest": "sha256:" + "1" * 64,
                },
                {
                    "path": "licenses/MODEL-LICENSE",
                    "digest": "sha256:" + "2" * 64,
                },
            ],
        },
    )
    return SimulatorTaskService(
        bundle,
        SqliteTaskStore(tmp_path / "simulator.sqlite3"),
        FakeMicroduckRuntime(),
    )


@pytest.fixture
def client(service: SimulatorTaskService, bearer_token: str):
    with TestClient(create_app(service, bearer_token)) as test_client:
        yield test_client


@pytest.fixture
def payload() -> dict[str, Any]:
    return {
        "schema": "MICRODUCK_SIM_TASK_V1",
        "taskId": "0" * 32,
        "actionCode": "STAND",
        "bundleVersion": "1.0.0",
        "bundleDigest": "sha256:" + "a" * 64,
        "parameters": {},
        "scenario": {"terrain": "flat", "seed": 1},
        "requestedBy": "api-test",
    }


def test_health_is_public(client: TestClient):
    """Accidentally protecting liveness would prevent unauthenticated orchestrator probes."""
    response = client.get("/v1/health")

    assert response.status_code == 200
    assert response.json() == {"alive": True}


def test_every_v1_route_except_health_requires_an_exact_bearer_token(
    client: TestClient,
):
    """Weak or partial header matching would expose the simulator task surface."""
    for path in ("/v1/ready", "/v1/catalog", "/v1/robot/status"):
        assert client.get(path).json() == {
            "code": "AUTH_REQUIRED",
            "message": "Authentication is required",
            "details": {},
        }
        assert (
            client.get(path, headers={"Authorization": "Bearer wrong"}).status_code
            == 401
        )
        assert (
            client.get(
                path, headers={"Authorization": "bearer test-simulator-token"}
            ).status_code
            == 401
        )


def test_create_is_idempotent_and_uses_camel_case_task_snapshot(
    client: TestClient, auth: dict[str, str], payload: dict[str, Any]
):
    """Changing a retry into a conflict or reserializing task IDs would break durable callers."""
    first = client.post("/v1/tasks", headers=auth, json=payload)
    second = client.post("/v1/tasks", headers=auth, json=payload)

    assert first.status_code == second.status_code == 202
    assert first.json()["taskId"] == second.json()["taskId"] == "0" * 32


@pytest.mark.parametrize(
    ("operation", "method", "path", "body"),
    [
        ("get", "get", "/v1/tasks/11111111111111111111111111111111", None),
        ("cancel", "post", "/v1/tasks/11111111111111111111111111111111/cancel", None),
        (
            "command",
            "put",
            "/v1/tasks/11111111111111111111111111111111/command",
            {"commandSequence": 0, "parameters": {}, "leaseMs": 100},
        ),
        ("events", "get", "/v1/tasks/11111111111111111111111111111111/events", None),
    ],
)
def test_task_routes_require_auth_before_exposing_task_existence(
    client: TestClient,
    operation: str,
    method: str,
    path: str,
    body: dict[str, Any] | None,
):
    """A missing authorization gate would reveal task state through every task operation."""
    request_kwargs = {"json": body} if body is not None else {}
    response = getattr(client, method)(path, **request_kwargs)

    assert operation
    assert response.status_code == 401
    assert response.json()["code"] == "AUTH_REQUIRED"


def test_service_errors_have_stable_public_envelopes_without_internal_messages(
    client: TestClient, auth: dict[str, str], payload: dict[str, Any]
):
    """Passing service exception text through would disclose internal validation details."""
    response = client.post(
        "/v1/tasks", headers=auth, json=payload | {"bundleVersion": "wrong-version"}
    )

    assert response.status_code == 400
    assert response.json() == {
        "code": "BUNDLE_MISMATCH",
        "message": "Requested bundle does not match the installed bundle",
        "details": {},
    }


def test_bad_wire_payload_uses_the_stable_parameter_error_envelope(
    client: TestClient, auth: dict[str, str], payload: dict[str, Any]
):
    """Leaving FastAPI's default 422 response would make typed intent errors unstable."""
    response = client.post(
        "/v1/tasks", headers=auth, json=payload | {"taskId": "not-an-id"}
    )

    assert response.status_code == 400
    assert response.json() == {
        "code": "PARAMETER_INVALID",
        "message": "Parameters are invalid",
        "details": {},
    }


def test_request_body_limit_rejects_declared_oversize_without_entering_validation(
    client: TestClient, auth: dict[str, str]
):
    """Trusting Content-Length/body parsing would buffer oversized intent before rejection."""
    response = client.post(
        "/v1/tasks",
        headers=auth | {"Content-Type": "application/json"},
        content=b"x" * 65_537,
    )

    assert response.status_code == 413
    assert response.json() == {
        "code": "REQUEST_BODY_TOO_LARGE",
        "message": "Request body exceeds the 65536-byte limit",
        "details": {},
    }


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("POST", "/v1/tasks/NOT-LOWERCASE/cancel"),
        ("PUT", f"/v1/tasks/{'A' * 32}/command"),
        ("PUT", "/v1/tasks/%32%32%32%32/command"),
    ],
)
def test_request_body_limit_precedes_v1_path_parameter_validation(
    client: TestClient,
    auth: dict[str, str],
    method: str,
    path: str,
):
    """Malformed paths must not bypass the global V1 POST/PUT ingress bound."""
    response = client.request(
        method,
        path,
        headers=auth | {"Content-Type": "application/json"},
        content=b"x" * 65_537,
    )

    assert response.status_code == 413
    assert response.json()["code"] == "REQUEST_BODY_TOO_LARGE"


def test_request_body_limit_stops_chunked_input_without_content_length(
    service: SimulatorTaskService, bearer_token: str
):
    """Omitting Content-Length must not permit an unbounded chunked body to be consumed."""
    app = create_app(service, bearer_token)
    chunks = iter(
        [
            {"type": "http.request", "body": b"a" * 40_000, "more_body": True},
            {"type": "http.request", "body": b"b" * 30_000, "more_body": True},
            {"type": "http.request", "body": b"must-not-be-read", "more_body": False},
        ]
    )
    consumed = 0
    sent: list[dict[str, Any]] = []

    async def receive():
        nonlocal consumed
        consumed += 1
        return next(chunks)

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/tasks",
        "raw_path": b"/v1/tasks",
        "query_string": b"",
        "headers": [
            (b"authorization", f"Bearer {bearer_token}".encode()),
            (b"content-type", b"application/json"),
        ],
        "client": ("127.0.0.1", 1),
        "server": ("testserver", 80),
    }

    asyncio.run(app(scope, receive, send))

    assert consumed == 2
    assert sent[0]["status"] == 413
    assert json.loads(sent[1]["body"]) == {
        "code": "REQUEST_BODY_TOO_LARGE",
        "message": "Request body exceeds the 65536-byte limit",
        "details": {},
    }


def test_request_body_limit_replays_a_valid_chunked_v1_request(
    service: SimulatorTaskService,
    bearer_token: str,
    payload: dict[str, Any],
):
    """The bounded ASGI reader must replay valid chunks exactly once to FastAPI."""
    app = create_app(service, bearer_token)
    encoded = json.dumps(payload, separators=(",", ":")).encode()
    split = len(encoded) // 2
    chunks = iter(
        [
            {"type": "http.request", "body": encoded[:split], "more_body": True},
            {"type": "http.request", "body": encoded[split:], "more_body": False},
        ]
    )
    sent: list[dict[str, Any]] = []

    async def receive():
        return next(chunks)

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/tasks",
        "raw_path": b"/v1/tasks",
        "query_string": b"",
        "headers": [
            (b"authorization", f"Bearer {bearer_token}".encode()),
            (b"content-type", b"application/json"),
        ],
        "client": ("127.0.0.1", 1),
        "server": ("testserver", 80),
    }

    asyncio.run(app(scope, receive, send))

    assert sent[0]["status"] == 202
    response_body = b"".join(
        message.get("body", b"")
        for message in sent
        if message["type"] == "http.response.body"
    )
    assert json.loads(response_body)["taskId"] == payload["taskId"]


def test_event_query_defaults_before_sequence_zero_and_enforces_page_size(
    client: TestClient,
    auth: dict[str, str],
    payload: dict[str, Any],
    service: SimulatorTaskService,
):
    """A zero default would permanently skip the first event; unbounded pages amplify history."""
    request = TaskCreateRequest.model_validate(payload | {"taskId": "9" * 32})
    service._store.create(request, "sha256:" + "d" * 64)
    service._store.append_event(request.taskId, "TASK_ACCEPTED", {"code": "ACCEPTED"})

    first = client.get(f"/v1/tasks/{request.taskId}/events?pageSize=1", headers=auth)
    oversized = client.get(
        f"/v1/tasks/{request.taskId}/events?pageSize=101", headers=auth
    )
    signed_max = client.get(
        f"/v1/tasks/{request.taskId}/events?afterSequence={2**63 - 1}",
        headers=auth,
    )
    signed_max_plus_one = client.get(
        f"/v1/tasks/{request.taskId}/events?afterSequence={2**63}", headers=auth
    )

    assert first.status_code == 200
    assert [event["sequence"] for event in first.json()["events"]] == [0]
    assert oversized.status_code == 400
    assert oversized.json()["code"] == "PARAMETER_INVALID"
    assert signed_max.status_code == 200
    assert signed_max.json() == {"events": []}
    assert signed_max_plus_one.status_code == 400
    assert signed_max_plus_one.json()["code"] == "PARAMETER_INVALID"


def test_catalog_is_derived_from_the_installed_service_bundle(
    client: TestClient, auth: dict[str, str], service: SimulatorTaskService
):
    """Returning a hard-coded catalog would let API metadata drift from the verified bundle."""
    response = client.get("/v1/catalog", headers=auth)

    assert response.status_code == 200
    assert response.json()["bundleId"] == "org.microduck.test"
    assert response.json()["actions"] == [
        action.model_dump(mode="json", by_alias=True)
        for action in service._bundle.actions
    ]


def test_generated_openapi_has_exact_v1_operations_security_and_schema_identifiers(
    service: SimulatorTaskService, bearer_token: str
):
    """A route alias or missing security declaration would let the implementation drift from V1."""
    checked_in = yaml.safe_load(
        (
            Path(__file__).parents[1]
            / "schemas/microduck-simulator-api-v1.openapi.yaml"
        ).read_text()
    )
    generated = create_app(service, bearer_token).openapi()

    expected_operations = {
        path: set(path_item) & {"get", "post", "put", "delete", "patch"}
        for path, path_item in checked_in["paths"].items()
    }
    actual_operations = {
        path: set(path_item) & {"get", "post", "put", "delete", "patch"}
        for path, path_item in generated["paths"].items()
        if path.startswith("/v1/")
    }
    assert actual_operations == expected_operations
    for path_item in checked_in["paths"].values():
        for method in {"post", "put"} & set(path_item):
            assert "413" in path_item[method]["responses"]

    for path, methods in expected_operations.items():
        for method in methods:
            expected = checked_in["paths"][path][method]
            actual = generated["paths"][path][method]
            assert actual.get("security") == expected.get("security")
            assert set(actual["responses"]) == set(expected["responses"])
            if "requestBody" in expected:
                assert (
                    actual["requestBody"]["content"]["application/json"]["schema"]
                    == expected["requestBody"]["content"]["application/json"]["schema"]
                )
            for status, response in expected["responses"].items():
                if "$ref" not in response:
                    assert (
                        actual["responses"][status]["content"]["application/json"][
                            "schema"
                        ]
                        == response["content"]["application/json"]["schema"]
                    )
                else:
                    expected_schema = checked_in["components"]["responses"][
                        response["$ref"].removeprefix("#/components/responses/")
                    ]["content"]["application/json"]["schema"]
                    assert (
                        actual["responses"][status]["content"]["application/json"][
                            "schema"
                        ]
                        == expected_schema
                    )

    assert {
        "ActionDefinition",
        "Error",
        "HealthResponse",
        "ReadyResponse",
        "CatalogResponse",
        "TaskCreateRequest",
        "TaskCommandRequest",
        "TaskSnapshot",
        "TaskEvent",
        "TaskEventPage",
        "RobotStatus",
    } <= set(generated["components"]["schemas"])
    assert _normalize_openapi_schema(generated["components"]["schemas"]["Error"]) == (
        _normalize_openapi_schema(checked_in["components"]["schemas"]["Error"])
    )

    for path, path_item in checked_in["paths"].items():
        for method in expected_operations[path]:
            expected_parameters = [
                _resolve_openapi_parameter(parameter, checked_in)
                for parameter in [
                    *path_item.get("parameters", []),
                    *path_item[method].get("parameters", []),
                ]
            ]
            actual_parameters = generated["paths"][path][method].get("parameters", [])
            assert _normalize_openapi_schema(
                actual_parameters
            ) == _normalize_openapi_schema(expected_parameters)

    assert _normalize_openapi_schema(generated) == _normalize_openapi_schema(checked_in)


def _normalize_openapi_schema(value: Any) -> Any:
    """Compare schemas semantically while omitting FastAPI's presentation-only defaults."""
    if isinstance(value, list):
        return [_normalize_openapi_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    normalized = {
        key: _normalize_openapi_schema(item)
        for key, item in value.items()
        if key != "title" and not (key == "additionalProperties" and item is True)
    }
    return normalized


def _resolve_openapi_parameter(
    parameter: dict[str, Any], document: dict[str, Any]
) -> dict[str, Any]:
    """Dereference checked-in reusable parameters before comparing their full wire shape."""
    if "$ref" not in parameter:
        return parameter
    return document["components"]["parameters"][
        parameter["$ref"].removeprefix("#/components/parameters/")
    ]


def _write_verified_bundle(bundle_dir: Path, source: PolicyBundle) -> PolicyBundle:
    """Create a hand-checked installed bundle fixture without using startup verification code."""
    files = {"models/robot.xml": b"<mujoco/>", "policies/stand.onnx": b"policy"}
    for relative_path, content in files.items():
        target = bundle_dir / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    digests = {
        relative_path: f"sha256:{hashlib.sha256(content).hexdigest()}"
        for relative_path, content in files.items()
    }
    unsigned = unsigned_policy_bundle_manifest(source).model_copy(
        update={
            "model": ModelArtifact(
                path="models/robot.xml", digest=digests["models/robot.xml"]
            ),
            "policies": [
                source.policies[0].model_copy(
                    update={
                        "path": "policies/stand.onnx",
                        "digest": digests["policies/stand.onnx"],
                    }
                )
            ],
        }
    )
    bundle = publish_policy_bundle(unsigned, digests)
    (bundle_dir / "microduck-policy-bundle.json").write_text(
        bundle.model_dump_json(by_alias=True, exclude_none=True)
    )
    return bundle


def _running_task(store: SqliteTaskStore, bundle: PolicyBundle) -> TaskCreateRequest:
    request = TaskCreateRequest.model_validate(
        {
            "schema": "MICRODUCK_SIM_TASK_V1",
            "taskId": "f" * 32,
            "actionCode": "STAND",
            "bundleVersion": bundle.bundleVersion,
            "bundleDigest": bundle.bundleDigest,
            "parameters": {},
            "scenario": {"terrain": "flat", "seed": 1},
            "requestedBy": "existing-task",
        }
    )
    store.create(request, "sha256:" + "d" * 64)
    store.transition(request.taskId, "VALIDATING", event_type="TASK_VALIDATING")
    store.transition(request.taskId, "RUNNING", event_type="TASK_STARTED")
    return request


def test_invalid_bundle_keeps_liveness_public_but_fails_readiness_and_task_execution_closed(
    tmp_path: Path,
):
    """Treating an unreadable installed bundle as ready would permit unverified catalog or motion."""
    app = create_configured_app(
        {
            "MICRODUCK_ROM_BUNDLE_DIR": str(tmp_path),
            "MICRODUCK_ROM_STATE_DB": str(tmp_path / "state.sqlite3"),
            "MICRODUCK_ROM_BEARER_TOKEN": "startup-token",
        }
    )
    with TestClient(app) as client:
        assert client.get("/v1/health").json() == {"alive": True}
        ready = client.get(
            "/v1/ready", headers={"Authorization": "Bearer startup-token"}
        )
        task = client.post(
            "/v1/tasks",
            headers={"Authorization": "Bearer startup-token"},
            json={
                "schema": "MICRODUCK_SIM_TASK_V1",
                "taskId": "0" * 32,
                "actionCode": "STAND",
                "bundleVersion": "1.0.0",
                "bundleDigest": "sha256:" + "a" * 64,
                "parameters": {},
                "scenario": {"terrain": "flat", "seed": 1},
                "requestedBy": "api-test",
            },
        )

    assert ready.status_code == 200
    assert ready.json()["ready"] is False
    assert "BUNDLE_UNAVAILABLE" in ready.json()["reasonCodes"]
    assert task.status_code == 503
    assert task.json() == {
        "code": "NOT_READY",
        "message": "Simulator is not ready",
        "details": {},
    }


def test_placeholder_startup_does_not_reconcile_existing_running_task(
    tmp_path: Path, service: SimulatorTaskService
):
    """Constructing an unready service would otherwise rewrite a surviving RUNNING task to UNKNOWN."""
    bundle_dir = tmp_path / "bundle"
    bundle = _write_verified_bundle(bundle_dir, service._bundle)
    state_db = tmp_path / "state.sqlite3"
    store = SqliteTaskStore(state_db)
    task = _running_task(store, bundle)

    create_configured_app(
        {
            "MICRODUCK_ROM_BUNDLE_DIR": str(bundle_dir),
            "MICRODUCK_ROM_STATE_DB": str(state_db),
            "MICRODUCK_ROM_BEARER_TOKEN": "startup-token",
        }
    )

    persisted = SqliteTaskStore(state_db)
    assert persisted.get(task.taskId).state == "RUNNING"
    assert [event.eventType for event in persisted.events_after(task.taskId, -1)] == [
        "TASK_VALIDATING",
        "TASK_STARTED",
    ]


def test_candidate_bundle_cannot_expose_catalog_before_qualification(
    tmp_path: Path, service: SimulatorTaskService
):
    """A hash-valid candidate must not expose a catalog before governed promotion."""
    bundle_dir = tmp_path / "bundle"
    _write_verified_bundle(bundle_dir, service._bundle)
    app = create_configured_app(
        {
            "MICRODUCK_ROM_BUNDLE_DIR": str(bundle_dir),
            "MICRODUCK_ROM_STATE_DB": str(tmp_path / "state.sqlite3"),
            "MICRODUCK_ROM_BEARER_TOKEN": "startup-token",
        }
    )

    with TestClient(app) as client:
        catalog = client.get(
            "/v1/catalog", headers={"Authorization": "Bearer startup-token"}
        )

    assert catalog.status_code == 503
    assert catalog.json()["code"] == "NOT_READY"


def test_database_startup_failure_has_its_own_readiness_reason(
    tmp_path: Path, service: SimulatorTaskService
):
    """Collapsing a state-store open failure into bundle failure would send operators to the wrong repair."""
    bundle_dir = tmp_path / "bundle"
    _write_verified_bundle(bundle_dir, service._bundle)
    blocked_parent = tmp_path / "blocked-parent"
    blocked_parent.write_text("not a directory")

    app = create_configured_app(
        {
            "MICRODUCK_ROM_BUNDLE_DIR": str(bundle_dir),
            "MICRODUCK_ROM_STATE_DB": str(blocked_parent / "state.sqlite3"),
            "MICRODUCK_ROM_BEARER_TOKEN": "startup-token",
        }
    )
    with TestClient(app) as client:
        response = client.get(
            "/v1/ready", headers={"Authorization": "Bearer startup-token"}
        )

    assert response.json()["reasonCodes"] == [
        "BUNDLE_UNAVAILABLE",
        "QUALIFICATION_UNAVAILABLE",
        "RUNTIME_UNAVAILABLE",
        "STATE_DB_UNAVAILABLE",
    ]


def test_configuration_rejects_unsafe_listener_values_and_runtime_placeholder_refuses_motion():
    """Accepting an unsafe bind address or placeholder motion call would make startup unsafe by default."""
    with pytest.raises(ValueError, match="MICRODUCK_ROM_HOST"):
        read_configuration({"MICRODUCK_ROM_HOST": "0.0.0.0"})

    runtime = UnconfiguredRuntime()
    assert runtime.status().health["ready"] is False
    with pytest.raises(RuntimeError, match="not configured"):
        runtime.validate(None, None)


def test_configuration_allows_explicit_container_wildcard_listener_opt_in():
    """Without an explicit opt-in, a published container cannot expose its authenticated API."""
    configuration = read_configuration(
        {
            "MICRODUCK_ROM_HOST": "0.0.0.0",
            "MICRODUCK_ROM_ALLOW_WILDCARD_BIND": "true",
        }
    )

    assert configuration.host == "0.0.0.0"


def test_configuration_rejects_symlinked_bundle_and_state_paths(tmp_path: Path):
    """Resolving configuration symlinks would let the API verify or persist outside its configured roots."""
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    bundle_link = tmp_path / "bundle-link"
    bundle_link.symlink_to(bundle, target_is_directory=True)
    state = tmp_path / "state.sqlite3"
    state.write_text("")
    state_link = tmp_path / "state-link.sqlite3"
    state_link.symlink_to(state)

    with pytest.raises(ValueError, match="BUNDLE_DIR"):
        read_configuration({"MICRODUCK_ROM_BUNDLE_DIR": str(bundle_link)})
    with pytest.raises(ValueError, match="STATE_DB"):
        read_configuration({"MICRODUCK_ROM_STATE_DB": str(state_link)})

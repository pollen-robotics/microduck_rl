"""Authenticated V1 HTTP boundary for durable MicroDuck simulator tasks."""

from __future__ import annotations

import hmac
import json
from contextlib import asynccontextmanager
from threading import Event, Thread
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, Path, Query, Request, Security
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer
from pydantic import Field, StringConstraints
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .contracts import (
    ACTION_CONTRACT,
    OBSERVATION_CONTRACT,
    ActionDefinition,
    BoundedIdentifier,
    ContractModel,
    RobotStatus,
    StatusObject,
    TaskCommandRequest,
    TaskCreateRequest,
    TaskEvent,
    TaskSnapshot,
)
from .service import (
    ActionUnavailable,
    BundleMismatch,
    CommandSequenceConflict,
    InvalidParameters,
    NotReady,
    PreconditionFailed,
    RobotBusy,
    RuntimeException,
    SimulatorServiceError,
    SimulatorTaskService,
    StaleCommand,
    TaskConflict,
    TaskNotFound,
)

_TASK_ID_PATTERN = r"^[0-9a-f]{32}$"
_MAX_REQUEST_BODY_BYTES = 65_536
_SIGNED_INT_MIN = -(2**63)
_SIGNED_INT_MAX = 2**63 - 1
ErrorCode = Literal[
    "AUTH_REQUIRED",
    "NOT_READY",
    "BUNDLE_MISMATCH",
    "ACTION_UNAVAILABLE",
    "PARAMETER_INVALID",
    "PRECONDITION_FAILED",
    "ROBOT_BUSY",
    "TASK_ID_CONFLICT",
    "COMMAND_SEQUENCE_CONFLICT",
    "STALE_COMMAND",
    "TASK_NOT_FOUND",
    "REQUEST_BODY_TOO_LARGE",
    "INTERNAL_ERROR",
]


class Error(ContractModel):
    code: ErrorCode
    message: Annotated[str, StringConstraints(min_length=1, max_length=256)]
    details: StatusObject


class HealthResponse(ContractModel):
    alive: Literal[True] = True


class ReadyResponse(ContractModel):
    ready: bool
    robotModel: BoundedIdentifier | None = None
    bundleId: BoundedIdentifier | None = None
    bundleVersion: BoundedIdentifier | None = None
    bundleDigest: (
        Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")] | None
    ) = None
    reasonCodes: list[BoundedIdentifier] = Field(max_length=32)


class CatalogResponse(ContractModel):
    bundleId: BoundedIdentifier | None = None
    bundleVersion: BoundedIdentifier | None = None
    bundleDigest: (
        Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")] | None
    ) = None
    observationContract: Literal["MICRODUCK_OBS_61_V1"] = OBSERVATION_CONTRACT
    actionContract: Literal["MICRODUCK_ACTION_14_V1"] = ACTION_CONTRACT
    actions: list[ActionDefinition] = Field(max_length=256)


class TaskEventPage(ContractModel):
    events: list[TaskEvent] = Field(max_length=100)


class RequestBodyLimitMiddleware:
    """Reject bounded V1 request bodies without buffering chunked input."""

    def __init__(self, app: ASGIApp, *, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if not _route_has_json_body(scope):
            await self.app(scope, receive, send)
            return
        content_length = _content_length(scope)
        if content_length is not None and content_length > self.max_body_bytes:
            await self._reject(send)
            return

        consumed = 0
        buffered: list[Message] = []
        while True:
            message = await receive()
            buffered.append(message)
            if message["type"] == "http.request":
                consumed += len(message.get("body", b""))
                if consumed > self.max_body_bytes:
                    buffered.clear()
                    await self._reject(send)
                    return
                if not message.get("more_body", False):
                    break
            elif message["type"] == "http.disconnect":
                break

        async def replay_receive() -> Message:
            if buffered:
                return buffered.pop(0)
            return await receive()

        await self.app(scope, replay_receive, send)

    async def _reject(self, send: Send) -> None:
        body = json.dumps(
            {
                "code": "REQUEST_BODY_TOO_LARGE",
                "message": f"Request body exceeds the {self.max_body_bytes}-byte limit",
                "details": {},
            },
            separators=(",", ":"),
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


def _route_has_json_body(scope: Scope) -> bool:
    if scope["type"] != "http":
        return False
    method = scope.get("method")
    path = scope.get("path", "")
    return method in {"POST", "PUT"} and (path == "/v1" or path.startswith("/v1/"))


def _content_length(scope: Scope) -> int | None:
    for name, value in scope.get("headers", []):
        if name.lower() == b"content-length":
            try:
                parsed = int(value)
            except ValueError:
                return _MAX_REQUEST_BODY_BYTES + 1
            return parsed if parsed >= 0 else _MAX_REQUEST_BODY_BYTES + 1
    return None


def _restore_exact_integer_bounds(value: Any) -> None:
    """Emit integer-schema bounds as exact JSON integers, including int64."""
    if isinstance(value, list):
        for item in value:
            _restore_exact_integer_bounds(item)
        return
    if not isinstance(value, dict):
        return
    if value.get("type") == "integer":
        for keyword in (
            "minimum",
            "maximum",
            "exclusiveMinimum",
            "exclusiveMaximum",
        ):
            bound = value.get(keyword)
            if keyword in {"maximum", "exclusiveMaximum"} and (
                isinstance(bound, float) and bound == float(_SIGNED_INT_MAX)
            ):
                value[keyword] = _SIGNED_INT_MAX
            elif keyword in {"minimum", "exclusiveMinimum"} and (
                isinstance(bound, float) and bound == float(_SIGNED_INT_MIN)
            ):
                value[keyword] = _SIGNED_INT_MIN
            elif isinstance(bound, float) and bound.is_integer():
                value[keyword] = int(bound)
    for key, item in value.items():
        if (
            key == "maximum"
            and isinstance(item, float)
            and item == float(_SIGNED_INT_MAX)
        ):
            value[key] = _SIGNED_INT_MAX
        elif (
            key == "minimum"
            and isinstance(item, float)
            and item == float(_SIGNED_INT_MIN)
        ):
            value[key] = _SIGNED_INT_MIN
        else:
            _restore_exact_integer_bounds(item)


_SERVICE_ERRORS: dict[type[SimulatorServiceError], tuple[int, ErrorCode, str]] = {
    BundleMismatch: (
        400,
        "BUNDLE_MISMATCH",
        "Requested bundle does not match the installed bundle",
    ),
    ActionUnavailable: (400, "ACTION_UNAVAILABLE", "Requested action is unavailable"),
    InvalidParameters: (400, "PARAMETER_INVALID", "Parameters are invalid"),
    PreconditionFailed: (
        400,
        "PRECONDITION_FAILED",
        "Task preconditions are not satisfied",
    ),
    RobotBusy: (409, "ROBOT_BUSY", "Robot already has an active task"),
    TaskConflict: (
        409,
        "TASK_ID_CONFLICT",
        "Task ID conflicts with an existing request",
    ),
    CommandSequenceConflict: (
        409,
        "COMMAND_SEQUENCE_CONFLICT",
        "Command sequence conflicts with already accepted content",
    ),
    StaleCommand: (409, "STALE_COMMAND", "Command sequence is stale"),
    TaskNotFound: (404, "TASK_NOT_FOUND", "Task was not found"),
    NotReady: (503, "NOT_READY", "Simulator is not ready"),
    RuntimeException: (500, "INTERNAL_ERROR", "Simulator operation failed"),
}


def _error_response(status_code: int, code: ErrorCode, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=Error(code=code, message=message, details={}).model_dump(),
    )


def _service_error_response(error: SimulatorServiceError) -> JSONResponse:
    status_code, code, message = _SERVICE_ERRORS.get(
        type(error), (500, "INTERNAL_ERROR", "Simulator operation failed")
    )
    return _error_response(status_code, code, message)


def create_app(service: SimulatorTaskService | None, bearer_token: str) -> FastAPI:
    """Create the exact V1 API surface; only liveness remains unauthenticated."""
    configured_token = bearer_token if isinstance(bearer_token, str) else ""
    bearer_scheme = HTTPBearer(
        auto_error=False, scheme_name="bearerAuth", bearerFormat="opaque-token"
    )
    watchdog_stop = Event()

    def watchdog() -> None:
        while not watchdog_stop.is_set():
            if service is not None:
                try:
                    service.tick()
                except Exception:  # noqa: BLE001 - one tick must not kill the deadman.
                    app.state.watchdog_healthy = False
                    try:
                        service.watchdog_failed()
                    except Exception:  # noqa: BLE001 - readiness remains failed closed.
                        app.state.watchdog_terminalization_failed = True
                    return
            watchdog_stop.wait(0.01)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        watchdog_thread = Thread(
            target=watchdog, name="microduck-lease-watchdog", daemon=True
        )
        watchdog_stop.clear()
        app.state.watchdog_thread = watchdog_thread
        watchdog_thread.start()
        try:
            yield
        finally:
            shutdown_failure: BaseException | None = None
            watchdog_stop.set()
            watchdog_thread.join(timeout=2.0)
            if watchdog_thread.is_alive():
                app.state.watchdog_healthy = False
                app.state.watchdog_thread = watchdog_thread
                shutdown_failure = RuntimeError(
                    "watchdog shutdown containment failed"
                )
            else:
                app.state.watchdog_thread = None
            if service is not None:
                try:
                    service.close()
                except Exception as exc:  # noqa: BLE001 - shutdown remains fail closed.
                    app.state.watchdog_terminalization_failed = True
                    shutdown_failure = exc
                finally:
                    app.state.shutdown_reaped_pid = getattr(
                        service, "shutdown_reaped_pid", None
                    )
            app.state.shutdown_failure = shutdown_failure
            if shutdown_failure is not None:
                raise RuntimeError("simulator shutdown containment failed") from (
                    shutdown_failure
                )

    app = FastAPI(
        title="MicroDuck ROM Simulator API",
        version="1.0.0",
        description="Versioned typed-intent API for the MicroDuck MuJoCo simulator.",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )
    app.add_middleware(
        RequestBodyLimitMiddleware, max_body_bytes=_MAX_REQUEST_BODY_BYTES
    )
    app.state.watchdog_healthy = True
    app.state.watchdog_terminalization_failed = False
    app.state.shutdown_failure = None
    app.state.shutdown_reaped_pid = None

    async def require_bearer(
        request: Request,
        _credentials=Security(bearer_scheme),  # noqa: B008 - FastAPI declares OpenAPI security here.
    ) -> None:
        authorization = request.headers.get("authorization")
        expected = f"Bearer {configured_token}"
        if (
            not configured_token
            or authorization is None
            or not hmac.compare_digest(authorization, expected)
        ):
            raise AuthenticationRequired()

    @app.exception_handler(AuthenticationRequired)
    async def authentication_required(
        _: Request, __: AuthenticationRequired
    ) -> JSONResponse:
        return _error_response(401, "AUTH_REQUIRED", "Authentication is required")

    @app.exception_handler(SimulatorServiceError)
    async def simulator_service_error(
        _: Request, error: SimulatorServiceError
    ) -> JSONResponse:
        return _service_error_response(error)

    @app.exception_handler(RequestValidationError)
    async def request_validation_error(
        _: Request, __: RequestValidationError
    ) -> JSONResponse:
        return _error_response(400, "PARAMETER_INVALID", "Parameters are invalid")

    @app.exception_handler(Exception)
    async def unexpected_exception(_: Request, __: Exception) -> JSONResponse:
        return _error_response(500, "INTERNAL_ERROR", "Simulator operation failed")

    def installed_bundle():
        bundle = getattr(
            service, "_bundle", getattr(app.state, "installed_bundle", None)
        )
        if bundle is None:
            raise NotReady("simulator is not ready")
        return bundle

    def motion_readiness() -> tuple[bool, tuple[str, ...]]:
        reason_codes = list(getattr(app.state, "readiness_reason_codes", ()))
        if not app.state.watchdog_healthy:
            reason_codes.append("WATCHDOG_UNHEALTHY")
        bundle = getattr(
            service, "_bundle", getattr(app.state, "installed_bundle", None)
        )
        if not configured_token:
            reason_codes.append("BEARER_TOKEN_MISSING")
        if bundle is None:
            reason_codes.append("BUNDLE_UNAVAILABLE")
        if service is None:
            reason_codes.append("RUNTIME_UNAVAILABLE")
        else:
            _, service_reasons = service.motion_readiness()
            reason_codes.extend(service_reasons)
        unique = tuple(sorted(set(reason_codes)))
        return not unique, unique

    def require_motion_ready() -> None:
        ready, _ = motion_readiness()
        if not ready:
            raise NotReady("simulator is not ready")

    def ready_response() -> ReadyResponse:
        ready, reason_codes = motion_readiness()
        bundle = getattr(
            service, "_bundle", getattr(app.state, "installed_bundle", None)
        )
        return ReadyResponse(
            ready=ready,
            robotModel=bundle.robotModel if bundle is not None else None,
            bundleId=bundle.bundleId if bundle is not None else None,
            bundleVersion=bundle.bundleVersion if bundle is not None else None,
            bundleDigest=bundle.bundleDigest if bundle is not None else None,
            reasonCodes=list(reason_codes),
        )

    @app.get("/v1/health", operation_id="health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse()

    @app.get(
        "/v1/ready",
        operation_id="ready",
        response_model=ReadyResponse,
        dependencies=[Depends(require_bearer)],
        responses={401: {"model": Error}},
    )
    def ready() -> ReadyResponse:
        return ready_response()

    @app.get(
        "/v1/catalog",
        operation_id="catalog",
        response_model=CatalogResponse,
        dependencies=[Depends(require_bearer)],
        responses={401: {"model": Error}, 503: {"model": Error}},
    )
    def catalog() -> CatalogResponse:
        bundle = installed_bundle()
        runtime_ready, reason_codes = motion_readiness()
        actions = bundle.actions
        if not runtime_ready:
            unavailable_reason = (
                "WATCHDOG_UNHEALTHY"
                if "WATCHDOG_UNHEALTHY" in reason_codes
                else "RUNTIME_UNAVAILABLE"
            )
            actions = [
                action.model_copy(
                    update={
                        "availability": "UNAVAILABLE",
                        "unavailableReason": unavailable_reason,
                    }
                )
                if action.availability == "AVAILABLE"
                else action
                for action in bundle.actions
            ]
        return CatalogResponse(
            bundleId=bundle.bundleId,
            bundleVersion=bundle.bundleVersion,
            bundleDigest=bundle.bundleDigest,
            actions=actions,
        )

    @app.get(
        "/v1/robot/status",
        operation_id="robotStatus",
        response_model=RobotStatus,
        dependencies=[Depends(require_bearer)],
        responses={401: {"model": Error}, 503: {"model": Error}},
    )
    def robot_status() -> RobotStatus:
        if service is None:
            raise NotReady("simulator is not ready")
        return service.robot_status()

    @app.post(
        "/v1/tasks",
        operation_id="createTask",
        status_code=202,
        response_model=TaskSnapshot,
        dependencies=[Depends(require_bearer)],
        responses={
            400: {"model": Error},
            401: {"model": Error},
            409: {"model": Error},
            413: {"model": Error},
            503: {"model": Error},
        },
    )
    def create_task(request: TaskCreateRequest) -> TaskSnapshot:
        if service is None:
            raise NotReady("simulator is not ready")
        return service.create_task(request)

    @app.get(
        "/v1/tasks/{taskId}",
        operation_id="getTask",
        response_model=TaskSnapshot,
        dependencies=[Depends(require_bearer)],
        responses={
            400: {"model": Error},
            401: {"model": Error},
            404: {"model": Error},
            503: {"model": Error},
        },
    )
    def get_task(
        task_id: Annotated[str, Path(alias="taskId", pattern=_TASK_ID_PATTERN)],
    ) -> TaskSnapshot:
        if service is None:
            raise NotReady("simulator is not ready")
        return service.get_task(task_id)

    @app.post(
        "/v1/tasks/{taskId}/cancel",
        operation_id="cancelTask",
        response_model=TaskSnapshot,
        dependencies=[Depends(require_bearer)],
        responses={
            400: {"model": Error},
            401: {"model": Error},
            404: {"model": Error},
            413: {"model": Error},
            503: {"model": Error},
        },
    )
    def cancel_task(
        task_id: Annotated[str, Path(alias="taskId", pattern=_TASK_ID_PATTERN)],
    ) -> TaskSnapshot:
        if service is None:
            raise NotReady("simulator is not ready")
        return service.cancel_task(task_id)

    @app.put(
        "/v1/tasks/{taskId}/command",
        operation_id="commandTask",
        response_model=TaskSnapshot,
        dependencies=[Depends(require_bearer)],
        responses={
            400: {"model": Error},
            401: {"model": Error},
            404: {"model": Error},
            409: {"model": Error},
            413: {"model": Error},
            503: {"model": Error},
        },
    )
    def command_task(
        task_id: Annotated[str, Path(alias="taskId", pattern=_TASK_ID_PATTERN)],
        command: TaskCommandRequest,
    ) -> TaskSnapshot:
        if service is None:
            raise NotReady("simulator is not ready")
        require_motion_ready()
        return service.command(task_id, command)

    @app.get(
        "/v1/tasks/{taskId}/events",
        operation_id="taskEvents",
        response_model=TaskEventPage,
        dependencies=[Depends(require_bearer)],
        responses={
            400: {"model": Error},
            401: {"model": Error},
            404: {"model": Error},
            503: {"model": Error},
        },
    )
    def task_events(
        task_id: Annotated[str, Path(alias="taskId", pattern=_TASK_ID_PATTERN)],
        after_sequence: Annotated[
            int, Query(alias="afterSequence", ge=-1, le=2**63 - 1)
        ] = -1,
        page_size: Annotated[int, Query(alias="pageSize", ge=1, le=100)] = 100,
    ) -> TaskEventPage:
        if service is None:
            raise NotReady("simulator is not ready")
        return TaskEventPage(
            events=service.events_after(task_id, after_sequence, page_size=page_size)
        )

    def openapi() -> dict[str, Any]:
        if app.openapi_schema is None:
            app.openapi_schema = get_openapi(
                title=app.title,
                version=app.version,
                openapi_version=app.openapi_version,
                routes=app.routes,
            )
            for path in app.openapi_schema["paths"].values():
                for operation in path.values():
                    if isinstance(operation, dict):
                        operation.get("responses", {}).pop("422", None)
            _restore_exact_integer_bounds(app.openapi_schema)
        return app.openapi_schema

    app.openapi = openapi
    return app


class AuthenticationRequired(Exception):
    """Private control-flow exception for a stable 401 response."""

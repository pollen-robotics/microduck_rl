"""Authenticated V1 HTTP boundary for durable MicroDuck simulator tasks."""

from __future__ import annotations

import hmac
from contextlib import asynccontextmanager
from threading import Event, Thread
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, Path, Query, Request, Security
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer

from .contracts import (
    ACTION_CONTRACT,
    OBSERVATION_CONTRACT,
    ActionDefinition,
    ContractModel,
    RobotStatus,
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
    "INTERNAL_ERROR",
]


class Error(ContractModel):
    code: ErrorCode
    message: str
    details: dict[str, Any]


class HealthResponse(ContractModel):
    alive: Literal[True] = True


class ReadyResponse(ContractModel):
    ready: bool
    robotModel: str | None = None
    bundleId: str | None = None
    bundleVersion: str | None = None
    bundleDigest: str | None = None
    reasonCodes: list[str]


class CatalogResponse(ContractModel):
    bundleId: str | None = None
    bundleVersion: str | None = None
    bundleDigest: str | None = None
    observationContract: Literal["MICRODUCK_OBS_61_V1"] = OBSERVATION_CONTRACT
    actionContract: Literal["MICRODUCK_ACTION_14_V1"] = ACTION_CONTRACT
    actions: list[ActionDefinition]


class TaskEventPage(ContractModel):
    events: list[TaskEvent]


class NotReady(SimulatorServiceError):
    code = "NOT_READY"


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
            watchdog_stop.set()
            watchdog_thread.join(timeout=2.0)
            if watchdog_thread.is_alive():
                app.state.watchdog_healthy = False
                app.state.watchdog_thread = watchdog_thread
            else:
                app.state.watchdog_thread = None

    app = FastAPI(docs_url=None, redoc_url=None, lifespan=lifespan)
    app.state.watchdog_healthy = True

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

    def ready_response() -> ReadyResponse:
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
        runtime_ready = False
        if service is not None:
            try:
                runtime_ready = bool(service.robot_status().health.get("ready"))
            except Exception:  # noqa: BLE001 - readiness must fail closed.
                reason_codes.append("RUNTIME_UNAVAILABLE")
        if not runtime_ready:
            reason_codes.append("RUNTIME_UNAVAILABLE")
        return ReadyResponse(
            ready=not reason_codes,
            robotModel=bundle.robotModel if bundle is not None else None,
            bundleId=bundle.bundleId if bundle is not None else None,
            bundleVersion=bundle.bundleVersion if bundle is not None else None,
            bundleDigest=bundle.bundleDigest if bundle is not None else None,
            reasonCodes=sorted(set(reason_codes)),
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
        responses={401: {"model": Error}},
    )
    def catalog() -> CatalogResponse:
        bundle = installed_bundle()
        runtime_ready = False
        if service is not None:
            try:
                health = service.robot_status().health
                runtime_ready = bool(health.get("ready") and health.get("healthy"))
            except Exception:  # noqa: BLE001 - catalog availability must fail closed.
                runtime_ready = False
        actions = bundle.actions
        if not runtime_ready:
            actions = [
                action.model_copy(
                    update={
                        "availability": "UNAVAILABLE",
                        "unavailableReason": "RUNTIME_UNAVAILABLE",
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
        responses={401: {"model": Error}},
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
            401: {"model": Error},
            404: {"model": Error},
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
            401: {"model": Error},
            404: {"model": Error},
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
        },
    )
    def command_task(
        task_id: Annotated[str, Path(alias="taskId", pattern=_TASK_ID_PATTERN)],
        command: TaskCommandRequest,
    ) -> TaskSnapshot:
        if service is None:
            raise NotReady("simulator is not ready")
        return service.command(task_id, command)

    @app.get(
        "/v1/tasks/{taskId}/events",
        operation_id="taskEvents",
        response_model=TaskEventPage,
        dependencies=[Depends(require_bearer)],
        responses={
            401: {"model": Error},
            404: {"model": Error},
        },
    )
    def task_events(
        task_id: Annotated[str, Path(alias="taskId", pattern=_TASK_ID_PATTERN)],
        after_sequence: Annotated[int, Query(alias="afterSequence", ge=0)] = 0,
    ) -> TaskEventPage:
        if service is None:
            raise NotReady("simulator is not ready")
        return TaskEventPage(events=service.events_after(task_id, after_sequence))

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
        return app.openapi_schema

    app.openapi = openapi
    return app


class AuthenticationRequired(Exception):
    """Private control-flow exception for a stable 401 response."""

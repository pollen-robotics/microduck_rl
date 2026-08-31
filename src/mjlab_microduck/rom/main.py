"""Process composition for the MicroDuck ROM simulator API."""

from __future__ import annotations

import hashlib
import ipaddress
import os
import signal
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import uvicorn

from .action_catalog import validate_bundle_action_envelope
from .api import create_app
from .contracts import (
    ModelArtifact,
    PolicyBundle,
    RobotStatus,
    canonical_json,
    sha256_prefixed,
    unsigned_policy_bundle_manifest,
)
from .process_supervisor import ReapReceipt, RuntimeProcessSupervisor
from .service import SimulatorTaskService
from .store import SqliteTaskStore


@dataclass(frozen=True)
class ServerConfiguration:
    bundle_dir: Path | None
    state_db: Path | None
    bearer_token: str
    host: str
    port: int


def _safe_host(value: str, *, allow_wildcard: bool = False) -> str:
    try:
        address = ipaddress.ip_address(value)
    except ValueError as exc:
        raise ValueError("MICRODUCK_ROM_HOST must be an IP address") from exc
    if (
        (address.is_unspecified and not allow_wildcard)
        or address.is_multicast
        or address.is_reserved
    ):
        raise ValueError("MICRODUCK_ROM_HOST must be a routable local address")
    return str(address)


def read_configuration(environ: Mapping[str, str] = os.environ) -> ServerConfiguration:
    """Read only the documented ROM settings and reject unsafe server inputs."""
    raw_port = environ.get("MICRODUCK_ROM_PORT", "8000")
    try:
        port = int(raw_port)
    except ValueError as exc:
        raise ValueError("MICRODUCK_ROM_PORT must be an integer") from exc
    if not 1 <= port <= 65_535:
        raise ValueError("MICRODUCK_ROM_PORT must be between 1 and 65535")

    bundle_value = environ.get("MICRODUCK_ROM_BUNDLE_DIR", "").strip()
    bundle_path = Path(bundle_value).expanduser() if bundle_value else None
    if bundle_path is not None and (
        not bundle_path.is_dir() or bundle_path.is_symlink()
    ):
        raise ValueError("MICRODUCK_ROM_BUNDLE_DIR must be a real directory")
    bundle_dir = bundle_path.resolve() if bundle_path is not None else None

    state_value = environ.get("MICRODUCK_ROM_STATE_DB", "").strip()
    state_path = Path(state_value).expanduser() if state_value else None
    if state_path is not None and (
        state_path.is_symlink() or (state_path.exists() and not state_path.is_file())
    ):
        raise ValueError("MICRODUCK_ROM_STATE_DB must be a file path")
    state_db = state_path.resolve() if state_path is not None else None

    wildcard_value = environ.get("MICRODUCK_ROM_ALLOW_WILDCARD_BIND", "false")
    if wildcard_value not in {"true", "false"}:
        raise ValueError("MICRODUCK_ROM_ALLOW_WILDCARD_BIND must be true or false")

    return ServerConfiguration(
        bundle_dir=bundle_dir,
        state_db=state_db,
        bearer_token=environ.get("MICRODUCK_ROM_BEARER_TOKEN", ""),
        host=_safe_host(
            environ.get("MICRODUCK_ROM_HOST", "127.0.0.1"),
            allow_wildcard=wildcard_value == "true",
        ),
        port=port,
    )


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1_048_576), b""):
            hasher.update(chunk)
    return f"sha256:{hasher.hexdigest()}"


def _bundle_artifacts(bundle: PolicyBundle) -> list[ModelArtifact]:
    artifacts = [
        bundle.model,
        *(
            ModelArtifact(path=policy.path, digest=policy.digest)
            for policy in bundle.policies
        ),
    ]
    for container, key in (
        (bundle.qualification, "artifacts"),
        (bundle.qualification, "modelClosure"),
    ):
        raw = container.get(key, [])
        if not isinstance(raw, list):
            raise TypeError("bundle artifact declarations must be lists")
        artifacts.extend(ModelArtifact.model_validate(item) for item in raw)
    artifacts.extend(bundle.license.artifacts)
    return artifacts


def _bundle_path(root: Path, declared_path: str) -> Path:
    candidate = (root / declared_path).resolve()
    if (
        not declared_path
        or candidate == root
        or root not in candidate.parents
        or not candidate.is_file()
    ):
        raise ValueError("bundle contains an invalid artifact path")
    return candidate


def load_verified_bundle(bundle_dir: Path) -> PolicyBundle:
    """Load a directory-installed bundle only after manifest, paths, hashes and digest agree."""
    root = bundle_dir.resolve()
    manifest_path = root / "microduck-policy-bundle.json"
    if not root.is_dir() or not manifest_path.is_file():
        raise ValueError("bundle directory does not contain a manifest")
    try:
        bundle = PolicyBundle.model_validate_json(manifest_path.read_bytes())
    except (OSError, ValueError) as exc:
        raise ValueError("bundle manifest is invalid") from exc
    digests: dict[str, str] = {}
    for artifact in _bundle_artifacts(bundle):
        if (
            artifact.path in digests
            or _digest(_bundle_path(root, artifact.path)) != artifact.digest
        ):
            raise ValueError("bundle artifact verification failed")
        digests[artifact.path] = artifact.digest
    unsigned_manifest = unsigned_policy_bundle_manifest(bundle).model_dump(
        mode="json", by_alias=True
    )
    if not hmac_compare(
        bundle.bundleDigest,
        sha256_prefixed({"manifest": unsigned_manifest, "artifacts": digests}),
    ):
        raise ValueError("bundle digest verification failed")
    validate_bundle_action_envelope(bundle)
    return bundle


def load_qualified_bundle(bundle_dir: Path) -> PolicyBundle:
    """Load a promoted bundle only after its qualification chain is exact."""
    root = Path(bundle_dir).resolve()
    bundle = load_verified_bundle(root)
    try:
        from .qualification import (
            QUALIFICATION_REPORT_PATH,
            RELEASE_CONFIGURATION_PATH,
            SUBJECT_MANIFEST_PATH,
            QualificationReport,
            ReleaseConfiguration,
            promoted_action_definition,
            recompute_action_qualification,
            release_action_declarations,
            validate_release_configuration,
        )
        from .runtime_identity import runtime_revision

        qualification = bundle.qualification
        if qualification.get("binding") != "VERIFIED_INPUT_BUNDLE_DIGEST_V1":
            raise ValueError
        declared = [
            ModelArtifact.model_validate(item)
            for item in qualification.get("artifacts", [])
        ]

        def read_bound_artifact(
            path_key: str, digest_key: str, expected_path: str
        ) -> tuple[bytes, str]:
            declared_path = qualification.get(path_key)
            declared_digest = qualification.get(digest_key)
            matching = [item for item in declared if item.path == declared_path]
            if declared_path != expected_path or len(matching) != 1:
                raise ValueError
            if declared_digest != matching[0].digest:
                raise ValueError
            content = _bundle_path(root, declared_path).read_bytes()
            if not hmac_compare(
                _digest(_bundle_path(root, declared_path)), declared_digest
            ):
                raise ValueError
            return content, declared_digest

        report_bytes, report_digest = read_bound_artifact(
            "reportPath", "reportDigest", QUALIFICATION_REPORT_PATH
        )
        configuration_bytes, configuration_digest = read_bound_artifact(
            "releaseConfigurationPath",
            "releaseConfigurationDigest",
            RELEASE_CONFIGURATION_PATH,
        )
        subject_bytes, subject_digest = read_bound_artifact(
            "subjectManifestPath", "subjectManifestDigest", SUBJECT_MANIFEST_PATH
        )
        report = QualificationReport.model_validate_json(report_bytes)
        configuration = ReleaseConfiguration.model_validate_json(configuration_bytes)
        subject = PolicyBundle.model_validate_json(subject_bytes)
        if (
            canonical_json(report) != report_bytes
            or canonical_json(configuration) != configuration_bytes
            or canonical_json(subject) != subject_bytes
        ):
            raise ValueError
        if report_digest != sha256_prefixed(report):
            raise ValueError
        if configuration_digest != sha256_prefixed(configuration):
            raise ValueError
        if subject_digest != sha256_prefixed(subject):
            raise ValueError

        subject_artifacts = {
            artifact.path: artifact.digest for artifact in _bundle_artifacts(subject)
        }
        if len(subject_artifacts) != len(_bundle_artifacts(subject)):
            raise ValueError
        expected_subject_digest = sha256_prefixed(
            {
                "manifest": unsigned_policy_bundle_manifest(subject).model_dump(
                    mode="json", by_alias=True
                ),
                "artifacts": subject_artifacts,
            }
        )
        if subject.bundleDigest != expected_subject_digest:
            raise ValueError
        validate_bundle_action_envelope(subject)
        for path, digest in subject_artifacts.items():
            if _digest(_bundle_path(root, path)) != digest:
                raise ValueError

        if (
            report.binding != "VERIFIED_INPUT_BUNDLE_DIGEST_V1"
            or report.subjectBundleId != subject.bundleId
            or report.subjectBundleVersion != subject.bundleVersion
            or report.subjectBundleDigest != subject.bundleDigest
            or qualification.get("subjectBundleId") != subject.bundleId
            or qualification.get("subjectBundleVersion") != subject.bundleVersion
            or qualification.get("subjectBundleDigest") != subject.bundleDigest
            or report.releaseConfigurationDigest != configuration_digest
            or qualification.get("releaseConfigurationDigest") != configuration_digest
            or report.sourceRepository != subject.sourceRepository
            or report.sourceCommit != subject.sourceCommit
            or report.modelDigest != subject.model.digest
            or report.runtimeRevision != runtime_revision()
            or bundle.bundleId != subject.bundleId
            or bundle.sourceRepository != subject.sourceRepository
            or bundle.sourceCommit != subject.sourceCommit
            or bundle.model != subject.model
            or bundle.policies != subject.policies
            or bundle.license != subject.license
            or configuration.release != bundle.bundleVersion
            or configuration.createdAt != bundle.createdAt
        ):
            raise ValueError
        validate_release_configuration(subject, configuration)
        results = {item.actionCode: item for item in report.actions}
        effective_declarations = release_action_declarations(subject, configuration)
        declarations = {item.actionCode: item for item in effective_declarations}
        installed_actions = {item.actionCode: item for item in bundle.actions}
        subject_actions = {item.actionCode: item for item in subject.actions}
        expected_codes = set(installed_actions)
        if (
            len(results) != len(report.actions)
            or len(declarations) != len(effective_declarations)
            or set(results) != expected_codes
            or set(declarations) != expected_codes
            or set(subject_actions) != expected_codes
        ):
            raise ValueError
        for code in sorted(expected_codes):
            action = installed_actions[code]
            subject_action = subject_actions[code]
            result = results[code]
            declaration = declarations[code]
            expected_result = recompute_action_qualification(
                subject,
                declaration,
                subject_action,
                result.rollouts,
                report.runtimeRevision,
            )
            if canonical_json(result) != canonical_json(expected_result):
                raise ValueError
            expected_action = promoted_action_definition(
                subject_action, expected_result
            )
            if canonical_json(action) != canonical_json(expected_action):
                raise ValueError
            if declaration.mandatory and expected_result.status != "PASSED":
                raise ValueError
        return bundle
    except Exception as exc:
        raise ValueError("bundle qualification verification failed") from exc


def hmac_compare(left: str, right: str) -> bool:
    """Keep digest equality exact without exposing a timing-sensitive comparison at the call site."""
    import hmac

    return hmac.compare_digest(left, right)


def _state_db_path_is_usable(state_db: Path) -> bool:
    """Reject unavailable state locations without opening or changing the database."""
    return state_db.parent.is_dir() and os.access(state_db.parent, os.W_OK | os.X_OK)


def _shutdown_evidence_path(state_db: Path) -> Path:
    return state_db.with_name(f"{state_db.name}.shutdown-v1.json")


def _write_shutdown_evidence(
    state_db: Path, *, reap_receipt: ReapReceipt, exit_code: int
) -> Path:
    """Durably publish exact-reap ordering while this API parent is still PID 1."""
    if (
        reap_receipt.pid <= 1
        or reap_receipt.generation <= 0
        or reap_receipt.ownership_identity <= 0
        or exit_code not in {0, 70}
    ):
        raise ValueError("shutdown evidence values are invalid")
    destination = _shutdown_evidence_path(state_db)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    payload = canonical_json(
        {
            "schema": "MICRODUCK_ROM_PID1_SHUTDOWN_V1",
            "pid1Pid": os.getpid(),
            "reapedChildPid": reap_receipt.pid,
            "childGeneration": reap_receipt.generation,
            "ownershipIdentity": reap_receipt.ownership_identity,
            "exactReapConfirmed": True,
            "exitCode": exit_code,
            "events": [
                {"sequence": 0, "event": "CHILD_REAPED"},
                {"sequence": 1, "event": "PID1_EXITING"},
            ],
        }
    )
    temporary.unlink(missing_ok=True)
    descriptor = os.open(
        temporary,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_CLOEXEC,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as target:
            target.write(payload)
            target.flush()
            os.fsync(target.fileno())
    finally:
        os.close(descriptor)
    os.replace(temporary, destination)
    directory = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return destination


class UnconfiguredRuntime:
    """Legacy diagnostic placeholder; it is never a service execution fallback."""

    def validate(self, *_: Any) -> None:
        raise RuntimeError("simulator runtime is not configured")

    def status(self) -> RobotStatus:
        return RobotStatus(
            schema="BIPED_POSE_V1",
            timestamp=datetime.now(UTC),
            basePositionM=(0.0, 0.0, 0.0),
            baseOrientationXyzw=(0.0, 0.0, 0.0, 1.0),
            baseLinearVelocityMps=(0.0, 0.0, 0.0),
            baseAngularVelocityRadps=(0.0, 0.0, 0.0),
            jointPositionsRad=(0.0,) * 14,
            jointVelocitiesRadps=(0.0,) * 14,
            policyTarget={},
            requestedMotion={},
            appliedMotion={},
            simulationTimeS=0.0,
            loopFrequencyHz=0.0,
            fallen=False,
            limp=True,
            health={
                "ready": False,
                "healthy": False,
                "reasonCodes": ["RUNTIME_UNCONFIGURED"],
            },
        )


def create_configured_app(
    environ: Mapping[str, str] = os.environ, *, runtime: Any | None = None
):
    """Compose the verified concrete runtime, or remain explicitly fail-closed."""
    configuration = read_configuration(environ)
    reasons: list[str] = []
    service: SimulatorTaskService | None = None
    bundle: PolicyBundle | None = None
    if not configuration.bearer_token:
        reasons.append("BEARER_TOKEN_MISSING")
    if configuration.bundle_dir is None:
        reasons.append("BUNDLE_UNAVAILABLE")
    elif configuration.bearer_token:
        try:
            bundle = load_qualified_bundle(configuration.bundle_dir)
        except Exception as exc:  # noqa: BLE001 - stable, non-secret readiness boundary.
            if "qualification" in str(exc):
                reasons.append("QUALIFICATION_UNAVAILABLE")
            else:
                reasons.append("BUNDLE_UNAVAILABLE")
    state_db_usable = configuration.state_db is not None and _state_db_path_is_usable(
        configuration.state_db
    )
    if not state_db_usable:
        reasons.append("STATE_DB_UNAVAILABLE")
    if bundle is not None and state_db_usable:
        assert configuration.state_db is not None
        try:
            assert configuration.bundle_dir is not None
            supervisor_factory = runtime or (
                lambda callback: RuntimeProcessSupervisor(
                    bundle_root=configuration.bundle_dir,
                    bundle_digest=bundle.bundleDigest,
                    terminal_callback=callback,
                    operation_timeout_s=10.0,
                    # Preserve a bounded observable termination window so PID 1
                    # can join an in-flight quarantine before SIGKILL/reap.
                    terminate_timeout_s=2.0,
                    owner_thread_name="microduck-runtime-supervisor-production",
                )
            )
            service = SimulatorTaskService(
                bundle, SqliteTaskStore(configuration.state_db), supervisor_factory
            )
        except Exception:  # noqa: BLE001 - do not leak filesystem or database contents.
            reasons.append("RUNTIME_UNAVAILABLE")
    app = create_app(service, configuration.bearer_token)
    app.state.task_service = service
    app.state.installed_bundle = bundle
    app.state.readiness_reason_codes = reasons
    return app


def main() -> None:
    """Launch the configured HTTP server without logging authorization material."""
    configuration = read_configuration()
    state_db = getattr(configuration, "state_db", None)
    if isinstance(state_db, Path):
        _shutdown_evidence_path(state_db).unlink(missing_ok=True)
    applications: list[Any] = []

    def application_factory():
        app = create_configured_app()
        applications.append(app)
        return app

    server = uvicorn.Server(
        uvicorn.Config(
            application_factory,
            host=configuration.host,
            port=configuration.port,
            # Keep HTTP draining shorter than the durable store's bounded write
            # wait so lifespan containment failures cannot expire invisibly
            # before ``service.close()`` observes the blocked callback worker.
            timeout_graceful_shutdown=1.0,
            factory=True,
        )
    )
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def request_server_shutdown(_signum: int, _frame: object) -> None:
        # Uvicorn restores and replays the prior handler after its signal
        # capture context.  Preserve that replay without letting the bootstrap
        # handler's pre-server SystemExit(0) hide a lifespan close failure.
        server.should_exit = True

    signal.signal(signal.SIGTERM, request_server_shutdown)
    try:
        server.run()
        app = applications[0] if applications else None
        shutdown_failed = (
            app is not None and app.state.shutdown_failure is not None
        ) or getattr(server.lifespan, "shutdown_failed", False)
        reap_receipt = (
            getattr(app.state, "shutdown_reap_receipt", None)
            if app is not None
            else None
        )
        if isinstance(state_db, Path) and isinstance(reap_receipt, ReapReceipt):
            try:
                _write_shutdown_evidence(
                    state_db,
                    reap_receipt=reap_receipt,
                    exit_code=70 if shutdown_failed else 0,
                )
            except Exception:  # noqa: BLE001 - missing reap evidence fails closed.
                shutdown_failed = True
        if shutdown_failed:
            raise SystemExit(70)
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    main()

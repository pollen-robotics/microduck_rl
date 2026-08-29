"""Process composition for the MicroDuck ROM simulator API."""

from __future__ import annotations

import hashlib
import ipaddress
import os
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
from .runtime import RuntimeEvidence, RuntimeHandle, RuntimeSample, SimulationRuntime
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
        (bundle.license, "artifacts"),
    ):
        raw = container.get(key, [])
        if not isinstance(raw, list):
            raise TypeError("bundle artifact declarations must be lists")
        artifacts.extend(ModelArtifact.model_validate(item) for item in raw)
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


def _runtime_is_ready(runtime: SimulationRuntime) -> bool:
    """Only a usable runtime may trigger durable task reconciliation."""
    try:
        return bool(runtime.status().health.get("ready"))
    except Exception:  # noqa: BLE001 - startup must fail closed.
        return False


class UnconfiguredRuntime:
    """Explicit fail-closed placeholder whose motion methods always refuse work."""

    def _unavailable(self) -> None:
        raise RuntimeError("simulator runtime is not configured")

    def validate(self, *_: Any) -> None:
        self._unavailable()

    def start(self, *_: Any) -> RuntimeHandle:
        self._unavailable()

    def command(self, *_: Any) -> None:
        self._unavailable()

    def sample(self, *_: Any) -> RuntimeSample:
        self._unavailable()

    def safe_stop(self, *_: Any) -> RuntimeEvidence:
        self._unavailable()

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
            health={"ready": False, "reason": "RUNTIME_UNCONFIGURED"},
        )


def create_configured_app(
    environ: Mapping[str, str] = os.environ, *, runtime: SimulationRuntime | None = None
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
    if bundle is not None and runtime is None:
        try:
            from .mujoco_runtime import MicroduckMujocoRuntime

            assert configuration.bundle_dir is not None
            runtime = MicroduckMujocoRuntime(configuration.bundle_dir, bundle)
        except Exception:  # noqa: BLE001 - startup must fail closed without artifact details.
            runtime = None
    if bundle is not None:
        if runtime is None or not _runtime_is_ready(runtime):
            reasons.append("RUNTIME_UNAVAILABLE")
        elif state_db_usable:
            assert configuration.state_db is not None
            try:
                service = SimulatorTaskService(
                    bundle, SqliteTaskStore(configuration.state_db), runtime
                )
            except Exception:  # noqa: BLE001 - do not leak filesystem or database contents.
                reasons.append("STATE_DB_UNAVAILABLE")
    app = create_app(service, configuration.bearer_token)
    app.state.installed_bundle = bundle
    app.state.readiness_reason_codes = reasons
    return app


def main() -> None:
    """Launch the configured HTTP server without logging authorization material."""
    configuration = read_configuration()
    uvicorn.run(
        create_configured_app(), host=configuration.host, port=configuration.port
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import pytest
from pydantic import ValidationError

import mjlab_microduck.rom.qualification as qualification_module
from mjlab_microduck.rom.action_catalog import ACTION_TEMPLATES
from mjlab_microduck.rom.contracts import (
    ActionDefinition,
    PolicyBundle,
    canonical_json,
    sha256_prefixed,
)
from mjlab_microduck.rom.main import load_qualified_bundle
from mjlab_microduck.rom.mujoco_runtime import MicroduckMujocoRuntime
from mjlab_microduck.rom.qualification import (
    ActionQualificationConfig,
    QualificationFailed,
    QualificationReport,
    QualificationThresholds,
    ReleaseConfiguration,
    ReleaseConfigurationError,
    qualify_and_promote,
    recompute_action_qualification,
)
from mjlab_microduck.rom.runtime_identity import runtime_revision
from tests.test_rom_mujoco_runtime import (
    _rewrite_as_stand_bundle,
    _write_verified_bundle,
)

NOW = datetime(2026, 8, 29, 12, 0, tzinfo=UTC)


def _config(
    *,
    mandatory: bool,
    min_distance_m: float = 0.0,
    include_spin: bool = False,
) -> ReleaseConfiguration:
    actions = [
        ActionQualificationConfig(
            actionCode="WALK_VELOCITY",
            mandatory=mandatory,
            terrain="flat",
            resetProfile="DEFAULT_STANDING",
            seeds=(7, 11, 29),
            maxSteps=100,
            parameters={"vxMps": 0.1, "vyMps": 0.0, "yawRateRadps": 0.0},
            thresholds=QualificationThresholds(
                minSuccessRate=1.0,
                maxFallRate=0.0,
                maxMeanTrackingError=10.0,
                minMeanDistanceM=min_distance_m,
                maxMeanEnergyProxy=10_000.0,
                maxActuatorClampSteps=100,
                maxPhysicalJointLimitViolations=0,
                actionMetric="trackingError",
                actionMetricOperator="lte",
                actionMetricThreshold=10.0,
            ),
        )
    ]
    if include_spin:
        actions.append(
            ActionQualificationConfig(
                actionCode="SPIN",
                mandatory=False,
                terrain="flat",
                resetProfile="DEFAULT_STANDING",
                seeds=(7, 11, 29),
                maxSteps=100,
                parameters={},
                thresholds=QualificationThresholds(
                    minSuccessRate=1.0,
                    maxFallRate=0.0,
                    maxMeanTrackingError=10.0,
                    minMeanDistanceM=0.0,
                    maxMeanEnergyProxy=10_000.0,
                    maxActuatorClampSteps=100,
                    maxPhysicalJointLimitViolations=0,
                    actionMetric="yawRotationRad",
                    actionMetricOperator="gte",
                    actionMetricThreshold=1.0,
                ),
            )
        )
    return ReleaseConfiguration(
        release="1.0.1",
        createdAt=NOW,
        actions=tuple(actions),
    )


def _stand_config() -> ReleaseConfiguration:
    return ReleaseConfiguration(
        release="1.0.1",
        createdAt=NOW,
        actions=(
            ActionQualificationConfig(
                actionCode="STAND",
                mandatory=True,
                terrain="flat",
                resetProfile="TRAINED_SITTING",
                seeds=(7, 11, 29),
                maxSteps=100,
                parameters={},
                thresholds=QualificationThresholds(
                    minSuccessRate=1.0,
                    maxFallRate=0.0,
                    maxMeanTrackingError=10.0,
                    minMeanDistanceM=0.0,
                    maxMeanEnergyProxy=10_000.0,
                    maxActuatorClampSteps=100,
                    maxPhysicalJointLimitViolations=0,
                    actionMetric="standPoseError",
                    actionMetricOperator="lte",
                    actionMetricThreshold=0.08,
                ),
            ),
        ),
    )


def _manifest(archive: Path) -> dict[str, object]:
    with zipfile.ZipFile(archive) as source:
        return json.loads(source.read("microduck-policy-bundle.json"))


def _extract_promoted_bundle(tmp_path: Path) -> tuple[Path, PolicyBundle]:
    candidate = tmp_path / "candidate"
    _write_verified_bundle(candidate)
    promoted = qualify_and_promote(
        candidate,
        tmp_path / "qualified.zip",
        _config(mandatory=True),
        timestamp=lambda: NOW,
    )
    installed = tmp_path / "installed"
    with zipfile.ZipFile(promoted.output_zip) as archive:
        archive.extractall(installed)
    return installed, promoted.manifest


def _extract_promoted_stand_bundle(tmp_path: Path) -> tuple[Path, PolicyBundle]:
    candidate = tmp_path / "candidate"
    _rewrite_as_stand_bundle(
        candidate,
        _write_verified_bundle(
            candidate,
            policy_output=[0.0] * 14,
            action_code="STAND",
            task_id="Mjlab-SitStand-Flat-MicroDuck",
        ),
    )
    promoted = qualify_and_promote(
        candidate,
        tmp_path / "qualified.zip",
        _stand_config(),
        timestamp=lambda: NOW,
    )
    installed = tmp_path / "installed"
    with zipfile.ZipFile(promoted.output_zip) as archive:
        archive.extractall(installed)
    return installed, promoted.manifest


def _qualified_components(installed: Path):
    subject = PolicyBundle.model_validate_json(
        (installed / "qualification/subject-manifest-v1.json").read_bytes()
    )
    configuration = ReleaseConfiguration.model_validate_json(
        (installed / "qualification/release-v1.json").read_bytes()
    )
    report = QualificationReport.model_validate_json(
        (installed / "qualification/qualification-v1.json").read_bytes()
    )
    return subject, configuration.actions[0], subject.actions[0], report.actions[0]


def _resign_mutated_promoted_bundle(
    root: Path,
    *,
    mutate_report=None,
    mutate_configuration=None,
    mutate_manifest=None,
) -> None:
    manifest_path = root / "microduck-policy-bundle.json"
    manifest = json.loads(manifest_path.read_text())
    report_path = root / "qualification/qualification-v1.json"
    report = json.loads(report_path.read_text())
    configuration_path = root / "qualification/release-v1.json"
    configuration = json.loads(configuration_path.read_text())
    if mutate_configuration is not None:
        mutate_configuration(configuration)
        configuration_path.write_bytes(canonical_json(configuration))
        configuration_digest = (
            "sha256:" + hashlib.sha256(configuration_path.read_bytes()).hexdigest()
        )
        report["releaseConfigurationDigest"] = configuration_digest
        manifest["qualification"]["releaseConfigurationDigest"] = (
            configuration_digest
        )
        for artifact in manifest["qualification"]["artifacts"]:
            if artifact["path"] == "qualification/release-v1.json":
                artifact["digest"] = configuration_digest
    if mutate_report is not None:
        mutate_report(report)
    if mutate_report is not None or mutate_configuration is not None:
        report_path.write_bytes(canonical_json(report))
        report_digest = "sha256:" + hashlib.sha256(report_path.read_bytes()).hexdigest()
        manifest["qualification"]["reportDigest"] = report_digest
        for artifact in manifest["qualification"]["artifacts"]:
            if artifact["path"] == "qualification/qualification-v1.json":
                artifact["digest"] = report_digest
    if mutate_manifest is not None:
        mutate_manifest(manifest)
    manifest["bundleDigest"] = None
    normalized = PolicyBundle.model_validate(manifest)
    artifact_digests = {}
    for artifact in [
        manifest["model"],
        *manifest["policies"],
        *manifest["qualification"].get("artifacts", []),
        *manifest["qualification"].get("modelClosure", []),
        *manifest["license"].get("artifacts", []),
    ]:
        artifact_digests[artifact["path"]] = artifact["digest"]
    manifest["bundleDigest"] = sha256_prefixed(
        {
            "manifest": normalized.model_dump(
                mode="json", by_alias=True, exclude={"bundleDigest"}
            ),
            "artifacts": artifact_digests,
        }
    )
    manifest_path.write_bytes(canonical_json(manifest))


def _candidate_with_unavailable_spin(root: Path):
    bundle = _write_verified_bundle(root)
    template = next(item for item in ACTION_TEMPLATES if item.action_code == "SPIN")
    spin = ActionDefinition(
        actionCode="SPIN",
        executionMode=template.execution_mode,
        availability="UNAVAILABLE",
        unavailableReason="POLICY_ARTIFACT_MISSING",
        parameterSchema=template.parameter_schema,
        completion=template.completion,
        lease=template.lease,
    )
    unsigned = bundle.model_copy(
        update={"bundleDigest": None, "actions": [*bundle.actions, spin]}
    )
    artifact_digests = {
        unsigned.model.path: unsigned.model.digest,
        unsigned.policies[0].path: unsigned.policies[0].digest,
        **{
            item["path"]: item["digest"]
            for item in unsigned.license.get("artifacts", [])
        },
    }
    rewritten = unsigned.model_copy(
        update={
            "bundleDigest": sha256_prefixed(
                {
                    "manifest": unsigned.model_dump(
                        mode="json", by_alias=True, exclude={"bundleDigest"}
                    ),
                    "artifacts": artifact_digests,
                }
            )
        }
    )
    (root / "microduck-policy-bundle.json").write_text(
        rewritten.model_dump_json(by_alias=True, exclude_none=True)
    )
    return rewritten


def test_failed_mandatory_action_blocks_promotion(tmp_path: Path):
    """Removing the mandatory failure gate would publish a release that missed its threshold."""
    source = tmp_path / "candidate"
    _write_verified_bundle(source)
    output = tmp_path / "qualified.zip"

    with pytest.raises(QualificationFailed, match="WALK_VELOCITY"):
        qualify_and_promote(
            source,
            output,
            _config(mandatory=True, min_distance_m=100.0),
            timestamp=lambda: NOW,
        )

    assert not output.exists()


def test_failed_optional_action_is_catalog_visible_as_qualification_failed(
    tmp_path: Path,
):
    """Leaving an optional failure available would overstate the promoted catalog."""
    source = tmp_path / "candidate"
    candidate = _write_verified_bundle(source)
    original_manifest = (source / "microduck-policy-bundle.json").read_bytes()
    output = tmp_path / "qualified.zip"

    promoted = qualify_and_promote(
        source,
        output,
        _config(mandatory=False, min_distance_m=100.0),
        timestamp=lambda: NOW,
    )

    action = next(
        item for item in promoted.manifest.actions if item.actionCode == "WALK_VELOCITY"
    )
    assert action.availability == "UNAVAILABLE"
    assert action.unavailableReason == "QUALIFICATION_FAILED"
    assert promoted.report.subjectBundleDigest == candidate.bundleDigest
    assert (source / "microduck-policy-bundle.json").read_bytes() == original_manifest


def test_optional_action_without_runtime_support_is_not_falsely_qualified(
    tmp_path: Path,
):
    """Treating absent scenario/runtime support as a failed rollout would hide the true limitation."""
    source = tmp_path / "candidate"
    _candidate_with_unavailable_spin(source)
    output = tmp_path / "qualified.zip"

    promoted = qualify_and_promote(
        source,
        output,
        _config(mandatory=False, include_spin=True),
        timestamp=lambda: NOW,
    )

    spin = next(item for item in promoted.manifest.actions if item.actionCode == "SPIN")
    spin_result = next(
        item for item in promoted.report.actions if item.actionCode == "SPIN"
    )
    assert spin.availability == "UNAVAILABLE"
    assert spin.unavailableReason == "POLICY_ARTIFACT_MISSING"
    assert spin_result.status == "UNAVAILABLE"
    assert spin_result.unavailableReason == "POLICY_ARTIFACT_MISSING"
    assert spin_result.rollouts == ()

    installed = tmp_path / "installed"
    with zipfile.ZipFile(promoted.output_zip) as archive:
        archive.extractall(installed)
    loaded = load_qualified_bundle(installed)
    loaded_spin = next(item for item in loaded.actions if item.actionCode == "SPIN")
    assert loaded_spin.availability == "UNAVAILABLE"
    assert loaded_spin.unavailableReason == "POLICY_ARTIFACT_MISSING"


def test_mandatory_action_must_be_supported_by_candidate_capabilities(tmp_path: Path):
    """Allowing mandatory unsupported actions would make the release policy impossible to satisfy."""
    source = tmp_path / "candidate"
    _candidate_with_unavailable_spin(source)
    spin = _config(mandatory=False, include_spin=True).actions[-1].model_copy(
        update={"mandatory": True}
    )
    config = ReleaseConfiguration(
        release="1.0.1",
        createdAt=NOW,
        actions=(_config(mandatory=False).actions[0], spin),
    )

    with pytest.raises(ReleaseConfigurationError, match="SPIN"):
        qualify_and_promote(
            source,
            tmp_path / "qualified.zip",
            config,
            timestamp=lambda: NOW,
        )


def test_battery_uses_governed_runtime_and_records_bounded_exact_identity(
    tmp_path: Path,
):
    """Bypassing runtime evidence would omit the exact model/policy/reset identity being released."""
    source = tmp_path / "candidate"
    candidate = _write_verified_bundle(source)
    output = tmp_path / "qualified.zip"

    promoted = qualify_and_promote(
        source,
        output,
        _config(mandatory=True),
        timestamp=lambda: NOW,
    )

    result = promoted.report.actions[0]
    assert result.status == "PASSED"
    assert result.runtimeClass == "MicroduckMujocoRuntime"
    assert result.runtimeIdentifier == (
        "mjlab_microduck.rom.mujoco_runtime.MicroduckMujocoRuntime"
    )
    assert result.runtimeRevision == runtime_revision()
    assert result.policyDigest == candidate.policies[0].digest
    assert result.modelDigest == candidate.model.digest
    assert result.sourceCommit == candidate.sourceCommit
    assert result.checkpoint == "model_100.pt"
    assert result.runIdentity == "entity/project/run-id"
    assert result.resetProfile == "DEFAULT_STANDING"
    assert result.scenarioProfile == "SEEDED_SERVO_RESET_V1"
    assert [rollout.seed for rollout in result.rollouts] == [7, 11, 29]
    assert all(rollout.steps <= 100 for rollout in result.rollouts)
    assert all(rollout.startedAt == NOW == rollout.finishedAt for rollout in result.rollouts)
    assert all(rollout.energyProxy >= 0.0 for rollout in result.rollouts)
    assert all(rollout.actuatorClampSteps >= 0 for rollout in result.rollouts)
    assert all(
        rollout.physicalJointLimitViolations == 0 for rollout in result.rollouts
    )
    assert all(rollout.maxAbsAction >= 0.0 for rollout in result.rollouts)


def test_stand_qualification_promotes_exact_discrete_runtime_success(
    tmp_path: Path,
) -> None:
    """A qualified STAND must complete the governed sitting-to-standing runtime path."""
    source = tmp_path / "candidate"
    candidate = _rewrite_as_stand_bundle(
        source,
        _write_verified_bundle(
            source,
            policy_output=[0.0] * 14,
            action_code="STAND",
            task_id="Mjlab-SitStand-Flat-MicroDuck",
        ),
    )

    promoted = qualify_and_promote(
        source,
        tmp_path / "stand-qualified.zip",
        _stand_config(),
        timestamp=lambda: NOW,
    )

    result = promoted.report.actions[0]
    assert result.status == "PASSED"
    assert result.actionCode == "STAND"
    assert all(rollout.success for rollout in result.rollouts)
    assert all(
        rollout.stopReason == "STAND_POSE_SETTLED" for rollout in result.rollouts
    )
    assert result.policyDigest == candidate.policies[0].digest
    assert promoted.manifest.actions[0].availability == "AVAILABLE"


def test_qualification_rejects_runtime_evidence_with_wrong_seed_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A runtime result from another seed must never be attributed to this battery."""
    source = tmp_path / "candidate"
    _write_verified_bundle(source)
    original = MicroduckMujocoRuntime._evidence_metrics_locked

    def mismatched_seed(runtime: MicroduckMujocoRuntime):
        metrics = original(runtime)
        return metrics | {"rngSeed": int(metrics["rngSeed"]) + 1}

    monkeypatch.setattr(
        MicroduckMujocoRuntime, "_evidence_metrics_locked", mismatched_seed
    )

    with pytest.raises(QualificationFailed, match="identity"):
        qualify_and_promote(
            source,
            tmp_path / "qualified.zip",
            _config(mandatory=True),
            timestamp=lambda: NOW,
        )


def test_promotion_is_reproducible_refuses_overwrite_and_binds_report_artifact(
    tmp_path: Path,
):
    """Mutable or nondeterministic promotion would break exact release and handoff identity."""
    source = tmp_path / "candidate"
    candidate = _write_verified_bundle(source)
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"

    built_first = qualify_and_promote(
        source, first, _config(mandatory=True), timestamp=lambda: NOW
    )
    built_second = qualify_and_promote(
        source, second, _config(mandatory=True), timestamp=lambda: NOW
    )

    assert first.read_bytes() == second.read_bytes()
    assert built_first.manifest.bundleVersion == "1.0.1"
    assert built_first.manifest.bundleDigest != candidate.bundleDigest
    assert built_first.manifest.bundleDigest == built_second.manifest.bundleDigest
    manifest = _manifest(first)
    report_artifact = next(
        artifact
        for artifact in manifest["qualification"]["artifacts"]
        if artifact["path"] == manifest["qualification"]["reportPath"]
    )
    with zipfile.ZipFile(first) as archive:
        report_bytes = archive.read(report_artifact["path"])
    assert "bundleDigest" not in json.loads(report_bytes)
    assert report_artifact["digest"] == "sha256:" + hashlib.sha256(report_bytes).hexdigest()

    with pytest.raises(FileExistsError, match="already exists"):
        qualify_and_promote(
            source, first, _config(mandatory=True), timestamp=lambda: NOW
        )


def test_promoted_fixture_carries_declared_license_evidence(tmp_path: Path) -> None:
    """A distributable fixture without its declared license bytes loses provenance."""
    source = tmp_path / "candidate"
    _write_verified_bundle(source)
    promoted = qualify_and_promote(
        source,
        tmp_path / "qualified.zip",
        _config(mandatory=True),
        timestamp=lambda: NOW,
    )

    with zipfile.ZipFile(promoted.output_zip) as archive:
        manifest = json.loads(archive.read("microduck-policy-bundle.json"))
        license_artifact = manifest["license"]["artifacts"][0]
        license_bytes = archive.read(license_artifact["path"])

    assert license_artifact["digest"] == (
        "sha256:" + hashlib.sha256(license_bytes).hexdigest()
    )
    assert b"Apache License" in license_bytes


def test_runtime_loader_rejects_candidate_and_accepts_exact_promoted_report(
    tmp_path: Path,
) -> None:
    """A digest-valid candidate must not become executable without a promoted report."""
    candidate = tmp_path / "candidate-only"
    _write_verified_bundle(candidate)

    with pytest.raises(ValueError, match="qualification"):
        load_qualified_bundle(candidate)

    installed, promoted = _extract_promoted_bundle(tmp_path / "promoted-case")
    assert load_qualified_bundle(installed) == promoted


@pytest.mark.parametrize(
    ("mutate_report", "mutate_manifest"),
    [
        (
            lambda report: report["actions"].append(dict(report["actions"][0])),
            None,
        ),
        (
            lambda report: report.update({"subjectBundleId": "forged.bundle"}),
            None,
        ),
        (
            lambda report: report.update({"subjectBundleVersion": "forged"}),
            None,
        ),
        (
            lambda report: report.update(
                {"releaseConfigurationDigest": "sha256:" + "f" * 64}
            ),
            None,
        ),
        (
            lambda report: report.update(
                {"runtimeRevision": "mjlab-microduck@0.1.0+sha256:" + "f" * 64}
            ),
            None,
        ),
        (
            lambda report: report["actions"].clear(),
            None,
        ),
        (
            lambda report: report["actions"][0].update(
                {"status": "FAILED", "unavailableReason": "QUALIFICATION_FAILED"}
            ),
            None,
        ),
        (
            None,
            lambda manifest: manifest["actions"][0].update(
                {"qualificationRefs": []}
            ),
        ),
        (
            None,
            lambda manifest: manifest["actions"][0].update(
                {
                    "qualificationRefs": [
                        "qualification/qualification-v1.json",
                        "qualification/qualification-v1.json",
                    ]
                }
            ),
        ),
    ],
)
def test_runtime_loader_rejects_semantically_forged_or_partial_reports(
    tmp_path: Path, mutate_report, mutate_manifest
) -> None:
    """Re-signing forged report bytes must not bypass qualification semantics."""
    installed, _ = _extract_promoted_bundle(tmp_path)
    _resign_mutated_promoted_bundle(
        installed,
        mutate_report=mutate_report,
        mutate_manifest=mutate_manifest,
    )

    with pytest.raises(ValueError, match="qualification"):
        load_qualified_bundle(installed)


@pytest.mark.parametrize(
    "mutate_action",
    [
        pytest.param(
            lambda action: action["lease"].update({"maxLeaseMs": 4_999}),
            id="lease-max",
        ),
        pytest.param(
            lambda action: action["lease"].update(
                {"safeStopBehavior": "FORGED_STOP"}
            ),
            id="lease-safe-stop",
        ),
        pytest.param(
            lambda action: action["lease"].update({"commandCadenceMs": 25}),
            id="lease-cadence",
        ),
        pytest.param(
            lambda action: action["parameterSchema"]["properties"]["vxMps"].update(
                {"maximum": 0.3}
            ),
            id="parameter-schema",
        ),
        pytest.param(
            lambda action: action["preconditions"].update(
                {"allowedTerrains": ["ramp"]}
            ),
            id="terrain-precondition",
        ),
        pytest.param(
            lambda action: action.update(
                {
                    "completion": {
                        "terminalConditions": ["TASK_COMPLETE"],
                        "maxDurationMs": 1,
                    }
                }
            ),
            id="completion",
        ),
        pytest.param(
            lambda action: action.update(
                {"safety": {"mirroring": "FORGED", "zeroOnStop": False}}
            ),
            id="safety-mirroring",
        ),
    ],
)
def test_runtime_loader_rejects_resigned_promoted_action_contract_mutations(
    tmp_path: Path, mutate_action
) -> None:
    """Re-signing any nested executable action field must not alter qualified behavior."""
    installed, _ = _extract_promoted_bundle(tmp_path)
    _resign_mutated_promoted_bundle(
        installed,
        mutate_manifest=lambda manifest: mutate_action(manifest["actions"][0]),
    )

    with pytest.raises(ValueError, match="qualification"):
        load_qualified_bundle(installed)


@pytest.mark.parametrize(
    "mutate_result",
    [
        pytest.param(
            lambda result: result.update({"successRate": 0.5}),
            id="success-aggregate",
        ),
        pytest.param(
            lambda result: result.update({"actionMetricMean": 9.0}),
            id="action-metric-aggregate",
        ),
        pytest.param(
            lambda result: [
                rollout.update({"success": False, "fallen": True})
                for rollout in result["rollouts"]
            ],
            id="failed-fallen-rollouts-labeled-passed",
        ),
        pytest.param(
            lambda result: result["rollouts"][1].update(
                {"seed": result["rollouts"][0]["seed"]}
            ),
            id="duplicate-seed",
        ),
        pytest.param(
            lambda result: result["rollouts"].pop(),
            id="missing-seed",
        ),
        pytest.param(
            lambda result: result["rollouts"][0].update(
                {"actionMetric": "baseTravelM"}
            ),
            id="wrong-rollout-metric",
        ),
        pytest.param(
            lambda result: result["rollouts"][0].update({"steps": 101}),
            id="steps-over-bound",
        ),
    ],
)
def test_runtime_loader_recomputes_resigned_qualification_results(
    tmp_path: Path, mutate_result
) -> None:
    """Report status and aggregates must be derived from exact governed rollouts."""
    installed, _ = _extract_promoted_bundle(tmp_path)
    _resign_mutated_promoted_bundle(
        installed,
        mutate_report=lambda report: mutate_result(report["actions"][0]),
    )

    with pytest.raises(ValueError, match="qualification"):
        load_qualified_bundle(installed)


def test_runtime_loader_revalidates_resigned_governed_release_configuration(
    tmp_path: Path,
) -> None:
    """A self-consistent report must not make an ungoverned embedded command valid."""
    installed, _ = _extract_promoted_bundle(tmp_path)
    _resign_mutated_promoted_bundle(
        installed,
        mutate_configuration=lambda configuration: configuration["actions"][
            0
        ].update(
            {"parameters": {"vxMps": 0.2, "vyMps": 0.0, "yawRateRadps": 0.0}}
        ),
        mutate_report=lambda report: report["actions"][0].update(
            {"parameters": {"vxMps": 0.2, "vyMps": 0.0, "yawRateRadps": 0.0}}
        ),
    )

    with pytest.raises(ValueError, match="qualification"):
        load_qualified_bundle(installed)


def test_rollout_semantics_reject_invalid_numeric_domains_before_aggregation(
    tmp_path: Path,
) -> None:
    """Negative, non-finite, or impossible raw counters must never become a result."""
    installed, _ = _extract_promoted_bundle(tmp_path)
    subject, declaration, definition, result = _qualified_components(installed)
    rollout = result.rollouts[0]
    mutations = (
        {"trackingError": -0.1},
        {"distanceM": -0.1},
        {"energyProxy": -0.1},
        {"actuatorClampSteps": -1},
        {"physicalJointLimitViolations": -1},
        {"maxAbsAction": -0.1},
        {"actionMetricValue": -0.1},
        {"trackingError": math.nan},
        {"energyProxy": math.inf},
        {"actuatorClampSteps": rollout.steps + 1},
        {"physicalJointLimitViolations": rollout.steps * 14 + 1},
    )

    for mutation in mutations:
        forged_rollouts = (
            rollout.model_copy(update=mutation),
            *result.rollouts[1:],
        )
        with pytest.raises(ValueError, match="rollout"):
            recompute_action_qualification(
                subject,
                declaration,
                definition,
                forged_rollouts,
                result.runtimeRevision,
            )


def test_rollout_semantics_reject_fallen_reason_without_fallen_state(
    tmp_path: Path,
) -> None:
    """A safety stop reason must agree with terminal and fallen state evidence."""
    installed, _ = _extract_promoted_bundle(tmp_path)
    subject, declaration, definition, result = _qualified_components(installed)
    forged = result.rollouts[0].model_copy(
        update={
            "success": False,
            "fallen": False,
            "terminalState": "FAILED",
            "stopReason": "FALLEN",
        }
    )

    with pytest.raises(ValueError, match="qualification"):
        recompute_action_qualification(
            subject,
            declaration,
            definition,
            (forged, *result.rollouts[1:]),
            result.runtimeRevision,
        )


@pytest.mark.parametrize(
    "mutate_rollout",
    [
        pytest.param(
            lambda rollout: rollout.update({"steps": 1}),
            id="one-step-walk-success",
        ),
        pytest.param(
            lambda rollout: rollout.update({"stopReason": "FALLEN"}),
            id="success-with-fallen-stop",
        ),
        pytest.param(
            lambda rollout: rollout.pop("trackingSampleCount", None),
            id="incomplete-tracking-evidence",
        ),
        pytest.param(
            lambda rollout: rollout.pop("requestedMotion", None),
            id="missing-requested-command-identity",
        ),
        pytest.param(
            lambda rollout: rollout.pop("modelDigest", None),
            id="missing-model-identity",
        ),
    ],
)
def test_runtime_loader_rejects_resigned_semantically_invalid_walk_rollouts(
    tmp_path: Path, mutate_rollout
) -> None:
    """A fully re-hashed WALK report still requires complete governed raw evidence."""
    installed, _ = _extract_promoted_bundle(tmp_path)
    _resign_mutated_promoted_bundle(
        installed,
        mutate_report=lambda report: [
            mutate_rollout(rollout) for rollout in report["actions"][0]["rollouts"]
        ],
    )

    with pytest.raises(ValueError, match="qualification"):
        load_qualified_bundle(installed)


def test_runtime_loader_rejects_resigned_forged_stand_completion(
    tmp_path: Path,
) -> None:
    """A STAND success boolean cannot replace sustained settlement evidence."""
    installed, _ = _extract_promoted_stand_bundle(tmp_path)
    _resign_mutated_promoted_bundle(
        installed,
        mutate_report=lambda report: [
            rollout.pop("settledSteps", None)
            for rollout in report["actions"][0]["rollouts"]
        ],
    )

    with pytest.raises(ValueError, match="qualification"):
        load_qualified_bundle(installed)


@pytest.mark.parametrize("mutation", ["duplicate", "mandatory-failed", "parameters"])
def test_private_promotion_revalidates_qualification_correspondence(
    tmp_path: Path, mutation: str
) -> None:
    """Calling the packaging primitive directly must not publish forged qualification."""
    source = tmp_path / "candidate"
    _write_verified_bundle(source)
    configuration = _config(mandatory=True)
    bundle, report = qualification_module.qualify_bundle(
        source, configuration, timestamp=lambda: NOW
    )
    if mutation == "duplicate":
        forged = report.model_copy(update={"actions": (*report.actions, report.actions[0])})
    elif mutation == "mandatory-failed":
        failed = report.actions[0].model_copy(
            update={"status": "FAILED", "unavailableReason": "QUALIFICATION_FAILED"}
        )
        forged = report.model_copy(update={"actions": (failed,)})
    else:
        mismatched = report.actions[0].model_copy(
            update={"parameters": {"vxMps": 9.0, "vyMps": 0.0, "yawRateRadps": 0.0}}
        )
        forged = report.model_copy(update={"actions": (mismatched,)})

    with pytest.raises(ValueError, match="qualification"):
        qualification_module._promote_qualified_bundle(
            source,
            tmp_path / f"{mutation}.zip",
            configuration,
            bundle,
            forged,
        )


def test_release_config_must_match_code_owned_reset_and_cover_available_actions(
    tmp_path: Path,
):
    """A typo or omission in release policy must not silently select another evaluator path."""
    source = tmp_path / "candidate"
    _write_verified_bundle(source)
    walk = _config(mandatory=True).actions[0].model_copy(
        update={"resetProfile": "TRAINING_ONLY_RESET"}
    )

    with pytest.raises(ReleaseConfigurationError, match="reset profile"):
        qualify_and_promote(
            source,
            tmp_path / "wrong-reset.zip",
            ReleaseConfiguration(
                release="1.0.1",
                createdAt=NOW,
                actions=(walk,),
            ),
            timestamp=lambda: NOW,
        )


def test_release_policy_rejects_rubber_stamp_batteries_and_caller_revision() -> None:
    """One-step, one-seed batteries and caller-selected code identities are not governed."""
    base = _config(mandatory=True).actions[0]
    with pytest.raises(ValidationError, match="seeds"):
        ActionQualificationConfig.model_validate(
            base.model_dump() | {"seeds": [7]}
        )
    with pytest.raises(ValidationError, match="maxSteps"):
        ActionQualificationConfig.model_validate(
            base.model_dump() | {"maxSteps": 1}
        )
    with pytest.raises(ValidationError, match="runtimeSourceCommit"):
        ReleaseConfiguration.model_validate(
            _config(mandatory=True).model_dump(by_alias=True)
            | {"runtimeSourceCommit": "b" * 40}
        )


def test_release_policy_rejects_metric_and_command_outside_action_spec(
    tmp_path: Path,
) -> None:
    """A release file must not invent an evaluator metric or qualification command."""
    source = tmp_path / "candidate"
    _write_verified_bundle(source)
    action = _config(mandatory=True).actions[0]
    invalid_metric = action.model_copy(
        update={
            "thresholds": action.thresholds.model_copy(
                update={"actionMetric": "inventedMetric"}
            )
        }
    )
    invalid_command = action.model_copy(
        update={
            "parameters": {"vxMps": 9.0, "vyMps": 0.0, "yawRateRadps": 0.0}
        }
    )

    for declaration in (invalid_metric, invalid_command):
        with pytest.raises(ReleaseConfigurationError, match="code-owned"):
            qualify_and_promote(
                source,
                tmp_path / f"{declaration.thresholds.actionMetric}.zip",
                ReleaseConfiguration(
                    release="1.0.1",
                    createdAt=NOW,
                    actions=(declaration,),
                ),
                timestamp=lambda: NOW,
            )

    with pytest.raises(ReleaseConfigurationError, match="WALK_VELOCITY"):
        qualify_and_promote(
            source,
            tmp_path / "missing.zip",
            ReleaseConfiguration(
                release="1.0.1",
                createdAt=NOW,
                actions=(),
            ),
            timestamp=lambda: NOW,
        )


def test_promotion_never_writes_output_into_source_asset_tree(tmp_path: Path):
    """Writing promotion beneath the mounted source would mutate release inputs."""
    source = tmp_path / "candidate"
    _write_verified_bundle(source)

    with pytest.raises(ValueError, match="outside the source bundle"):
        qualify_and_promote(
            source,
            source / "qualified.zip",
            _config(mandatory=True),
            timestamp=lambda: NOW,
        )

    protected_root = tmp_path / "robot-source-assets"
    protected_root.mkdir()
    with pytest.raises(ValueError, match="protected source root"):
        qualify_and_promote(
            source,
            protected_root / "qualified.zip",
            _config(mandatory=True),
            timestamp=lambda: NOW,
            protected_source_roots=(protected_root,),
        )


def test_qualification_cli_promotes_real_candidate_without_disclosing_paths(
    tmp_path: Path,
):
    """Replacing the CLI with an unchecked wrapper would permit mutable or opaque releases."""
    source = tmp_path / "candidate"
    _write_verified_bundle(source)
    configuration = tmp_path / "release.json"
    configuration.write_bytes(
        json.dumps(
            _config(mandatory=True).model_dump(
                mode="json", by_alias=True, exclude_none=True
            )
        ).encode()
    )
    output = tmp_path / "qualified.zip"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/qualify_rom_bundle.py",
            "--bundle-dir",
            str(source),
            "--release-config",
            str(configuration),
            "--output",
            str(output),
        ],
        cwd=Path(__file__).parents[1],
        env=os.environ | {"MUJOCO_GL": "egl"},
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert output.is_file()
    assert "sha256:" in completed.stdout
    assert completed.stderr == ""


def test_qualification_cli_fails_closed_on_invalid_release_config(tmp_path: Path):
    """A traceback or partial output on invalid config would leak internals and confuse operators."""
    configuration = tmp_path / "invalid.json"
    configuration.write_text('{"secret":"do-not-print"}')
    output = tmp_path / "qualified.zip"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/qualify_rom_bundle.py",
            "--bundle-dir",
            str(tmp_path),
            "--release-config",
            str(configuration),
            "--output",
            str(output),
        ],
        cwd=Path(__file__).parents[1],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert completed.stderr == "qualification failed: release configuration is invalid\n"
    assert "do-not-print" not in completed.stderr
    assert not output.exists()


def test_container_entrypoint_fails_before_server_when_mounts_are_invalid(tmp_path: Path):
    """Starting without verified mount prerequisites would default the container to unready."""
    entrypoint = Path(__file__).parents[1] / "docker/rom-simulator/entrypoint.sh"
    completed = subprocess.run(
        ["bash", str(entrypoint)],
        env=os.environ
        | {
            "MICRODUCK_ROM_BUNDLE_DIR": str(tmp_path / "bundle"),
            "MICRODUCK_ROM_STATE_DB": str(tmp_path / "state/tasks.sqlite3"),
            "MICRODUCK_ROM_BEARER_TOKEN": "test-token",
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert completed.stdout == ""
    assert completed.stderr == "container startup failed: /bundle must contain a readable manifest\n"
    assert "test-token" not in completed.stderr


def _docker_context_includes(policy: str, path: str) -> bool:
    ignored = False
    for raw in policy.splitlines():
        rule = raw.strip()
        if not rule or rule.startswith("#"):
            continue
        include = rule.startswith("!")
        pattern = rule[1:] if include else rule
        normalized = pattern.rstrip("/")
        matches = (
            path == normalized
            if "/" not in normalized
            else PurePosixPath(path).match(normalized)
        )
        if pattern == "**" or matches:
            ignored = not include
    return not ignored


def test_docker_context_policies_allow_only_exact_rom_copy_inputs() -> None:
    """A new training, robot, secret, checkpoint, or output file must stay outside build context."""
    repository = Path(__file__).parents[1]
    tracked = subprocess.run(
        ["git", "ls-files"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    expected = {
        "pyproject.toml",
        "uv.lock",
        "README.md",
        "LICENSE",
        "src/mjlab_microduck/__init__.py",
        *(
            path
            for path in tracked
            if path.startswith("src/mjlab_microduck/rom/")
            and PurePosixPath(path).parent == PurePosixPath("src/mjlab_microduck/rom")
            and path.endswith(".py")
        ),
        "docker/rom-simulator/entrypoint.sh",
    }
    policies = [
        (repository / ".dockerignore").read_text(),
        (repository / "docker/rom-simulator/Dockerfile.dockerignore").read_text(),
    ]
    representatives = [
        *tracked,
        ".env",
        "output/checkpoint.pt",
        "src/mjlab_microduck/robot/microduck/assets/body.stl",
        "src/mjlab_microduck/robot/microduck/assets/source.part",
        "src/mjlab_microduck/tasks/new_training.py",
        "tests/secret_fixture.bin",
    ]
    for policy in policies:
        included = {
            path for path in representatives if _docker_context_includes(policy, path)
        }
        assert included == expected

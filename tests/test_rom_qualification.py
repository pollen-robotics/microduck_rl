from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from mjlab_microduck.rom.action_catalog import ACTION_TEMPLATES
from mjlab_microduck.rom.contracts import ActionDefinition, sha256_prefixed
from mjlab_microduck.rom.qualification import (
    ActionQualificationConfig,
    QualificationFailed,
    QualificationThresholds,
    ReleaseConfiguration,
    ReleaseConfigurationError,
    qualify_and_promote,
)
from tests.test_rom_mujoco_runtime import _write_verified_bundle

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
            seeds=(7, 11),
            maxSteps=3,
            parameters={"vxMps": 0.0, "vyMps": 0.0, "yawRateRadps": 0.0},
            thresholds=QualificationThresholds(
                minSuccessRate=1.0,
                maxFallRate=0.0,
                maxMeanTrackingError=100.0,
                minMeanDistanceM=min_distance_m,
                maxMeanEnergyProxy=1_000_000.0,
                maxLimitViolations=100,
                actionMetric="trackingError",
                actionMetricOperator="lte",
                actionMetricThreshold=100.0,
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
                seeds=(7,),
                maxSteps=3,
                parameters={},
                thresholds=QualificationThresholds(
                    minSuccessRate=1.0,
                    maxFallRate=0.0,
                    maxMeanTrackingError=100.0,
                    minMeanDistanceM=0.0,
                    maxMeanEnergyProxy=1_000_000.0,
                    maxLimitViolations=100,
                    actionMetric="yawRotationRad",
                    actionMetricOperator="gte",
                    actionMetricThreshold=1.0,
                ),
            )
        )
    return ReleaseConfiguration(
        release="1.0.1",
        createdAt=NOW,
        runtimeSourceCommit="b" * 40,
        actions=tuple(actions),
    )


def _manifest(archive: Path) -> dict[str, object]:
    with zipfile.ZipFile(archive) as source:
        return json.loads(source.read("microduck-policy-bundle.json"))


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
        runtimeSourceCommit="b" * 40,
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
    assert result.runtimeSourceCommit == "b" * 40
    assert result.policyDigest == candidate.policies[0].digest
    assert result.modelDigest == candidate.model.digest
    assert result.sourceCommit == candidate.sourceCommit
    assert result.checkpoint == "model_100.pt"
    assert result.runIdentity == "entity/project/run-id"
    assert result.resetProfile == "DEFAULT_STANDING"
    assert result.scenarioProfile == "SEEDED_SERVO_RESET_V1"
    assert [rollout.seed for rollout in result.rollouts] == [7, 11]
    assert all(rollout.steps <= 3 for rollout in result.rollouts)
    assert all(rollout.startedAt == NOW == rollout.finishedAt for rollout in result.rollouts)
    assert all(rollout.energyProxy >= 0.0 for rollout in result.rollouts)
    assert all(rollout.limitViolations >= 0 for rollout in result.rollouts)
    assert all(rollout.maxAbsAction >= 0.0 for rollout in result.rollouts)


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
    report_artifact = manifest["qualification"]["artifacts"][-1]
    with zipfile.ZipFile(first) as archive:
        report_bytes = archive.read(report_artifact["path"])
    assert "bundleDigest" not in json.loads(report_bytes)
    assert report_artifact["digest"] == "sha256:" + hashlib.sha256(report_bytes).hexdigest()

    with pytest.raises(FileExistsError, match="already exists"):
        qualify_and_promote(
            source, first, _config(mandatory=True), timestamp=lambda: NOW
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
                runtimeSourceCommit="b" * 40,
                actions=(walk,),
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
                runtimeSourceCommit="b" * 40,
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

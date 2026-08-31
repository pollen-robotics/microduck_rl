from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

import mjlab_microduck.rom.handoff as handoff_module
from mjlab_microduck.rom.contracts import canonical_json
from mjlab_microduck.rom.handoff import materialize_distribution_bundle
from mjlab_microduck.rom.main import load_qualified_bundle
from mjlab_microduck.rom.qualification import qualify_and_promote
from tests.test_rom_mujoco_runtime import _write_verified_bundle
from tests.test_rom_qualification import NOW, _config


def _installed_promoted_bundle(
    tmp_path: Path, *, status: str, name: str
) -> Path:
    candidate = tmp_path / f"{name}-candidate"
    _write_verified_bundle(candidate, model_license_status=status)
    promoted_zip = tmp_path / f"{name}.zip"
    qualify_and_promote(
        candidate, promoted_zip, _config(mandatory=True), timestamp=lambda: NOW
    )
    installed = tmp_path / f"{name}-installed"
    with zipfile.ZipFile(promoted_zip) as archive:
        archive.extractall(installed)
    return installed


def test_distribution_materialization_gates_development_only_before_destination_creation(
    tmp_path: Path,
) -> None:
    """Removing the materializer's clearance check would publish development-only bytes."""
    installed = _installed_promoted_bundle(
        tmp_path, status="DEVELOPMENT_ONLY", name="development"
    )
    destination = tmp_path / "distribution.zip"

    with pytest.raises(
        ValueError, match="model assets are not cleared for distribution handoff"
    ):
        materialize_distribution_bundle(installed, destination)

    assert not destination.exists()


def test_distribution_materialization_writes_deterministic_cleared_bundle(
    tmp_path: Path,
) -> None:
    """Changing archive ordering or metadata would break a reproducible handoff unit."""
    installed = _installed_promoted_bundle(
        tmp_path, status="DISTRIBUTION_CLEARED", name="cleared"
    )
    manifest_path = installed / "microduck-policy-bundle.json"
    manifest_path.write_text(json.dumps(json.loads(manifest_path.read_text()), indent=2))
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"

    materialize_distribution_bundle(installed, first)
    materialize_distribution_bundle(installed, second)

    assert first.read_bytes() == second.read_bytes()
    bundle = load_qualified_bundle(installed)
    expected = {
        "microduck-policy-bundle.json",
        bundle.model.path,
        *(policy.path for policy in bundle.policies),
        *(item.path for item in bundle.license.artifacts),
    }
    declared_artifacts = [bundle.model, *bundle.license.artifacts]
    for key in ("artifacts", "modelClosure"):
        declared = bundle.qualification.get(key, [])
        assert isinstance(declared, list)
        expected.update(item["path"] for item in declared)
        declared_artifacts.extend(
            type(bundle.model).model_validate(item) for item in declared
        )

    with zipfile.ZipFile(first) as archive:
        assert set(archive.namelist()) == expected
        assert archive.read("microduck-policy-bundle.json") == canonical_json(bundle)
        for artifact in declared_artifacts:
            content = archive.read(artifact.path)
            assert content == (installed / artifact.path).read_bytes()
            assert artifact.digest == "sha256:" + hashlib.sha256(content).hexdigest()


def test_distribution_materialization_rejects_post_verification_artifact_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rereading a mutated source after verification must never publish mismatched bytes."""
    installed = _installed_promoted_bundle(
        tmp_path, status="DISTRIBUTION_CLEARED", name="mutated"
    )
    destination = tmp_path / "distribution.zip"
    real_load = load_qualified_bundle

    def load_then_mutate(root: Path):
        bundle = real_load(root)
        (Path(root) / bundle.model.path).write_bytes(b"post-verification mutation")
        return bundle

    monkeypatch.setattr(handoff_module, "load_qualified_bundle", load_then_mutate)

    with pytest.raises(
        ValueError, match=r"^distribution artifact verification failed$"
    ):
        materialize_distribution_bundle(installed, destination)

    assert not destination.exists()


def test_distribution_materialization_leaves_no_partial_output_on_zip_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed ZIP assembly must leave the publication path entirely absent."""
    installed = _installed_promoted_bundle(
        tmp_path, status="DISTRIBUTION_CLEARED", name="zip-failure"
    )
    destination = tmp_path / "distribution.zip"
    real_writestr = zipfile.ZipFile.writestr
    writes = 0

    def fail_second_write(self, *args, **kwargs):
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError("controlled ZIP assembly failure")
        return real_writestr(self, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "writestr", fail_second_write)

    with pytest.raises(OSError, match="controlled ZIP assembly failure"):
        materialize_distribution_bundle(installed, destination)

    assert not destination.exists()

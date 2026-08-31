from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pytest

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
        assert archive.read("microduck-policy-bundle.json") == (
            installed / "microduck-policy-bundle.json"
        ).read_bytes()
        for artifact in declared_artifacts:
            content = archive.read(artifact.path)
            assert content == (installed / artifact.path).read_bytes()
            assert artifact.digest == "sha256:" + hashlib.sha256(content).hexdigest()

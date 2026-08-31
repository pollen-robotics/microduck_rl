"""Distribution-safe materialization of verified promoted policy bundles."""

from __future__ import annotations

import hashlib
import io
import os
import tempfile
import zipfile
from pathlib import Path

from .contracts import ModelArtifact, PolicyBundle, canonical_json
from .main import load_qualified_bundle


def require_distribution_cleared(bundle: PolicyBundle) -> None:
    if bundle.license.modelAssets.distributionStatus != "DISTRIBUTION_CLEARED":
        raise ValueError("model assets are not cleared for distribution handoff")


def _promoted_artifacts(bundle: PolicyBundle) -> list[ModelArtifact]:
    artifacts = [
        bundle.model,
        *(ModelArtifact(path=item.path, digest=item.digest) for item in bundle.policies),
    ]
    for key in ("artifacts", "modelClosure"):
        declared = bundle.qualification.get(key, [])
        if not isinstance(declared, list):
            raise TypeError("qualified bundle artifact declarations are invalid")
        artifacts.extend(ModelArtifact.model_validate(item) for item in declared)
    artifacts.extend(bundle.license.artifacts)
    return artifacts


def _staged_distribution_contents(
    root: Path, bundle: PolicyBundle
) -> dict[str, bytes]:
    contents = {"microduck-policy-bundle.json": canonical_json(bundle)}
    for artifact in _promoted_artifacts(bundle):
        if artifact.path in contents:
            raise ValueError("distribution artifact declarations are invalid")
        source = (root / artifact.path).resolve()
        if not source.is_file() or not source.is_relative_to(root):
            raise ValueError("distribution artifact verification failed")
        try:
            content = source.read_bytes()
        except OSError:
            raise ValueError("distribution artifact verification failed") from None
        digest = f"sha256:{hashlib.sha256(content).hexdigest()}"
        if digest != artifact.digest:
            raise ValueError("distribution artifact verification failed")
        contents[artifact.path] = content
    return contents


def _deterministic_zip_bytes(contents: dict[str, bytes]) -> bytes:
    target = io.BytesIO()
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_STORED) as archive:
        for relative, content in sorted(contents.items()):
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_STORED
            archive.writestr(info, content)
    return target.getvalue()


def _publish_new_file(destination: Path, content: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as target:
            target.write(content)
            target.flush()
            os.fsync(target.fileno())
            os.fchmod(target.fileno(), 0o644)
        try:
            os.link(temporary, destination)
        except FileExistsError:
            raise FileExistsError(
                f"distribution output already exists: {destination}"
            ) from None
    finally:
        temporary.unlink(missing_ok=True)


def materialize_distribution_bundle(bundle_dir: Path, destination: Path) -> PolicyBundle:
    """Write a deterministic distribution ZIP only from a verified cleared bundle."""
    root = Path(bundle_dir).resolve()
    output = Path(destination).resolve()
    if output.exists():
        raise FileExistsError(f"distribution output already exists: {output}")
    if output.is_relative_to(root):
        raise ValueError("distribution output must remain outside the source bundle")
    bundle = load_qualified_bundle(root)
    require_distribution_cleared(bundle)
    contents = _staged_distribution_contents(root, bundle)
    archive_bytes = _deterministic_zip_bytes(contents)
    _publish_new_file(output, archive_bytes)
    return bundle

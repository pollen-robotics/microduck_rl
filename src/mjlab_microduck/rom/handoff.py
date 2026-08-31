"""Distribution-safe materialization of verified promoted policy bundles."""

from __future__ import annotations

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

    files = {
        "microduck-policy-bundle.json",
        *(artifact.path for artifact in _promoted_artifacts(bundle)),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_STORED) as archive:
        for relative in sorted(files):
            content = (
                canonical_json(bundle)
                if relative == "microduck-policy-bundle.json"
                else (root / relative).read_bytes()
            )
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_STORED
            archive.writestr(info, content)
    return bundle

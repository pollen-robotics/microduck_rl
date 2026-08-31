"""Shared strict license fixtures for ROM boundary tests."""

from __future__ import annotations

from mjlab_microduck.rom.contracts import BundleLicense


def cleared_apache_license(
    *, artifact_digest: str = "sha256:" + "1" * 64
) -> BundleLicense:
    """Return one shared Apache-2.0 evidence declaration cleared for distribution."""
    return BundleLicense.model_validate(
        {
            "software": {
                "identifier": "Apache-2.0",
                "artifactPaths": ["licenses/LICENSE"],
            },
            "modelAssets": {
                "identifier": "Apache-2.0",
                "distributionStatus": "DISTRIBUTION_CLEARED",
                "artifactPaths": ["licenses/LICENSE"],
            },
            "artifacts": [
                {
                    "path": "licenses/LICENSE",
                    "digest": artifact_digest,
                }
            ],
        }
    )

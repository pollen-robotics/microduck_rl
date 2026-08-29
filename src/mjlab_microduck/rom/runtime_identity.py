"""Identity of the installed governed ROM runtime implementation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import tomllib
from pathlib import Path

_DISTRIBUTION = "mjlab-microduck"
_GOVERNED_MODULES = (
    "action_specs.py",
    "main.py",
    "model_semantics.py",
    "mujoco_runtime.py",
    "observation.py",
    "onnx_policy.py",
    "qualification.py",
    "runtime.py",
)


def _package_version() -> str:
    try:
        return importlib.metadata.version(_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        package_dir = Path(__file__).resolve().parent
        candidates = (
            Path("/app/pyproject.toml"),
            *(
                parent / "pyproject.toml"
                for parent in package_dir.parents
            ),
        )
        for candidate in candidates:
            if candidate.is_file():
                project = tomllib.loads(candidate.read_text())["project"]
                if project.get("name") == _DISTRIBUTION:
                    return str(project["version"])
        raise RuntimeError("installed ROM package metadata is unavailable") from None


def runtime_revision() -> str:
    """Return package version plus a digest of the exact governed source bytes."""
    package_dir = Path(__file__).resolve().parent
    hasher = hashlib.sha256()
    for name in _GOVERNED_MODULES:
        content = (package_dir / name).read_bytes()
        hasher.update(name.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(content)
        hasher.update(b"\0")
    return f"{_DISTRIBUTION}@{_package_version()}+sha256:{hasher.hexdigest()}"

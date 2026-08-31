"""Identity of the installed governed ROM runtime implementation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import tomllib
from pathlib import Path

_DISTRIBUTION = "mjlab-microduck"
GOVERNED_RUNTIME_MODULES = (
    "__init__.py",
    "rom/__init__.py",
    "rom/action_catalog.py",
    "rom/action_specs.py",
    "rom/api.py",
    "rom/bundle.py",
    "rom/contracts.py",
    "rom/main.py",
    "rom/mirroring.py",
    "rom/model_semantics.py",
    "rom/mujoco_runtime.py",
    "rom/observation.py",
    "rom/onnx_policy.py",
    "rom/qualification.py",
    "rom/process_protocol.py",
    "rom/process_service.py",
    "rom/process_supervisor.py",
    "rom/runtime_child.py",
    "rom/parent_death.py",
    "rom/runtime.py",
    "rom/runtime_identity.py",
    "rom/secret_file.py",
    "rom/service.py",
    "rom/store.py",
    "rom/supervisor_state.py",
)


def _package_version() -> str:
    try:
        return importlib.metadata.version(_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        package_dir = Path(__file__).resolve().parent
        candidates = (
            Path("/app/pyproject.toml"),
            *(parent / "pyproject.toml" for parent in package_dir.parents),
        )
        for candidate in candidates:
            if candidate.is_file():
                project = tomllib.loads(candidate.read_text())["project"]
                if project.get("name") == _DISTRIBUTION:
                    return str(project["version"])
        raise RuntimeError("installed ROM package metadata is unavailable") from None


def runtime_revision() -> str:
    """Return package version plus a digest of the exact governed source bytes."""
    package_dir = Path(__file__).resolve().parents[1]
    hasher = hashlib.sha256()
    for name in GOVERNED_RUNTIME_MODULES:
        content = (package_dir / name).read_bytes()
        hasher.update(name.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(content)
        hasher.update(b"\0")
    return f"{_DISTRIBUTION}@{_package_version()}+sha256:{hasher.hexdigest()}"

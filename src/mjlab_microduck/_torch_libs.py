"""Pre-load shared libraries a platform's torch wheel links but does not declare.

Some torch builds outside PyPI are linked against libraries their wheel
metadata never mentions, so ``import torch`` dies with ``cannot open shared
object file`` even though ``pyproject.toml`` installs the wheel that carries
it — the missing piece is only that torch's RPATH does not reach the other
wheel's ``lib/``. Loading the library by absolute path with ``RTLD_GLOBAL``
before torch's first import makes the loader find it by SONAME; no
``LD_LIBRARY_PATH`` needed.

Keyed on the wheels being present, not on the host: whatever platform
``[tool.uv.sources]`` did or did not resolve them for, this is a no-op
wherever they are absent (GB10, x86_64, HF Jobs, ...). Today's table is the
NVIDIA SBSA CUDA-13 torch build (what JetPack 7 boards resolve): NVPL
LAPACK (its own RPATH pulls the NVPL BLAS core) and cuDSS.

Called from ``mjlab_microduck/tasks/__init__.py``, next to the ``--hf-jobs``
hook and for the same reason: that module is what mjlab's plugin loader
imports (``import mjlab`` -> ``mjlab_microduck.tasks``) before ``mjlab.envs``
imports torch, on every train/play path. Anything that imports torch before
``mjlab`` does not get it — import ``mjlab`` first.
"""

from __future__ import annotations

import ctypes
import importlib.util
from pathlib import Path

#: (top-level package the wheel installs, path below it). Looked up through
#: the import system rather than sysconfig, so it also works when the venv is
#: layered (``uv run --with``) or the wheels live on another sys.path entry.
UNDECLARED_TORCH_LIBS: tuple[tuple[str, str], ...] = (
    ("nvpl", "lib/libnvpl_lapack_lp64_gomp.so.0"),
    ("nvidia", "cu13/lib/libcudss.so.0"),
)


def _package_dirs(name: str) -> list[Path]:
    """Every directory the (namespace) package ``name`` resolves to."""
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return []
    if spec is None or not spec.submodule_search_locations:
        return []
    return [Path(p) for p in spec.submodule_search_locations]


def preload_undeclared_torch_libs(site_packages: Path | None = None) -> list[str]:
    """Load every library in ``UNDECLARED_TORCH_LIBS`` that is installed.

    ``site_packages`` pins the lookup to one directory (tests); by default
    each wheel is found wherever the import system resolves its package.
    Never raises: a missing wheel just means torch fails on its own with the
    ImportError that names the library.
    """
    loaded: list[str] = []
    for package, rel in UNDECLARED_TORCH_LIBS:
        dirs = [site_packages / package] if site_packages else _package_dirs(package)
        for path in (d / rel for d in dirs):
            if not path.is_file():
                continue
            try:
                ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                continue
            loaded.append(str(path))
            break
    return loaded

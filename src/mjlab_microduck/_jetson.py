"""Jetson (JetPack 7 / SBSA) support: pre-load libraries the torch wheel omits.

On a Jetson AGX Thor, ``[tool.uv.sources]`` routes torch to NVIDIA's Jetson AI
Lab SBSA index (``pypi.jetson-ai-lab.io/sbsa/cu130``). That native
``torch-2.9.1 linux_aarch64`` wheel links NVPL BLAS/LAPACK and cuDSS but
declares neither, and its RPATH does not reach the ``nvpl-blas``,
``nvpl-lapack`` and ``nvidia-cudss-cu13`` wheels ``pyproject.toml`` adds for
Jetson — so a bare ``import torch`` dies with
``libnvpl_lapack_lp64_gomp.so.0: cannot open shared object file`` (then
``libcudss.so.0``). Loading them by absolute path with ``RTLD_GLOBAL`` before
torch's first import makes the loader find them by SONAME; no
``LD_LIBRARY_PATH`` needed.

Runs from ``mjlab_microduck/__init__.py`` so every entry point that reaches
this package — mjlab's plugin loader (``import mjlab`` → ``mjlab_microduck.tasks``,
before ``mjlab.envs`` imports torch), ``duck-body``, ``publish``, the export
and inference scripts — gets it for free. A non-editable install additionally
gets ``mjlab_microduck_jetson.pth`` (``[tool.uv.build-backend] data.purelib``),
whose ``import`` line runs at interpreter start, so there even a bare
``import torch`` works; an *editable* checkout (``uv sync``, ``uv run``) has no
such hook — uv_build's editable wheel ships no data files — so there, import
``mjlab`` or ``mjlab_microduck`` before torch (``tests/conftest.py`` does).
A no-op anywhere that is not a Jetson (the kernel release carries ``-tegra``:
``6.8.12-tegra`` on Thor) or where the wheels are absent, so GB10 / x86_64 /
HF Jobs are untouched.
"""

from __future__ import annotations

import ctypes
import importlib.util
import platform
import sys
from pathlib import Path

#: (top-level package the wheel installs, path below it). Looked up through
#: the import system rather than sysconfig, so it also works when the venv is
#: layered (``uv run --with``) or the wheels live on another sys.path entry.
#: lapack's own RPATH pulls the NVPL BLAS core.
_PRELOAD = (
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


def is_jetson(
    *,
    sys_platform: str = sys.platform,
    machine: str | None = None,
    release: str | None = None,
) -> bool:
    """The same predicate as the ``'tegra' in platform_release`` marker in
    ``pyproject.toml`` — keep the two in step."""
    machine = platform.machine() if machine is None else machine
    release = platform.release() if release is None else release
    return sys_platform == "linux" and machine == "aarch64" and "tegra" in release


def preload_jetson_libs(site_packages: Path | None = None) -> list[str]:
    """Load the undeclared NVPL/cuDSS libraries. Returns what was loaded.

    ``site_packages`` pins the lookup to one directory (tests); by default
    each wheel is found wherever the import system resolves its package.
    Never raises: a missing wheel just means torch will fail on its own with
    the ImportError above, which names the library.
    """
    if not is_jetson():
        return []
    loaded: list[str] = []
    for package, rel in _PRELOAD:
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

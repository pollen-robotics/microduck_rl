"""Per-platform torch sources beyond the GB10 one, and the pre-load of the
libraries a platform's torch wheel links but does not declare.

Today that is JetPack 7 / SBSA (Jetson AGX Thor): torch from NVIDIA's SBSA
index, NVPL + cuDSS pre-loaded. Companion to test_aarch64_cuda_torch.py
(which covers the NON-Jetson aarch64 entry, i.e. DGX Spark / GB10). The
discriminator is the PEP 508 `platform_release` marker: Jetson kernels carry
`-tegra`, GB10's do not. Verified on-box 2026-09-04 (#38).
"""

import platform
import sys
import tomllib
from pathlib import Path

import pytest

from mjlab_microduck._torch_libs import (
    UNDECLARED_TORCH_LIBS,
    preload_undeclared_torch_libs,
)

_ROOT = Path(__file__).resolve().parents[1]
_JETSON_INDEX = "https://pypi.jetson-ai-lab.io/sbsa/cu130"
_JETSON_MARKER = "'tegra' in platform_release"
_JETSON_ONLY_DEPS = ("nvpl-blas", "nvpl-lapack", "nvidia-cudss-cu13")


def _pyproject():
    return tomllib.loads((_ROOT / "pyproject.toml").read_text())


def _packages(name):
    lock = tomllib.loads((_ROOT / "uv.lock").read_text())
    return [p for p in lock["package"] if p["name"] == name]


def _markers(pkg):
    return " ".join(pkg.get("resolution-markers", []))


def test_torch_sources_split_jetson_from_gb10_on_the_kernel_release():
    sources = _pyproject()["tool"]["uv"]["sources"]["torch"]
    jetson = [s for s in sources if _JETSON_MARKER in s["marker"]]
    others = [s for s in sources if "'tegra' not in platform_release" in s["marker"]]
    assert len(jetson) == 1, "exactly one Jetson torch source"
    assert len(others) == 1, "the GB10 source must EXCLUDE Jetson, or both match"
    assert jetson[0]["index"] == "jetson-sbsa-cu130"
    assert others[0]["index"] == "pytorch-cu129"


def test_jetson_index_is_the_live_sbsa_one():
    indexes = {i["name"]: i for i in _pyproject()["tool"]["uv"]["index"]}
    assert indexes["jetson-sbsa-cu130"]["url"] == _JETSON_INDEX
    assert indexes["jetson-sbsa-cu130"].get("explicit") is True, "must stay explicit"


def test_undeclared_torch_deps_are_jetson_scoped_direct_dependencies():
    deps = _pyproject()["project"]["dependencies"]
    for name in _JETSON_ONLY_DEPS:
        hits = [d for d in deps if d.split(";")[0].strip() == name]
        assert len(hits) == 1, f"{name} must be a direct dependency"
        assert _JETSON_MARKER in hits[0], f"{name} must be Jetson-only"


def test_lockfile_routes_jetson_torch_to_the_sbsa_wheel_at_the_pinned_version():
    jetson = [p for p in _packages("torch") if _JETSON_MARKER in _markers(p)]
    assert len(jetson) == 1, "re-run `uv lock`"
    (pkg,) = jetson
    assert pkg["source"]["registry"] == _JETSON_INDEX
    assert pkg["version"] == "2.9.1", (
        "only the SOURCE changes on Jetson, not the version"
    )
    assert all("linux_aarch64" in w["url"] for w in pkg["wheels"])


def test_gb10_entry_still_excludes_jetson():
    gb10 = [
        p
        for p in _packages("torch")
        if "platform_machine == 'aarch64'" in _markers(p)
        and _JETSON_MARKER not in _markers(p)
    ]
    assert len(gb10) == 1
    assert "'tegra' not in platform_release" in _markers(gb10[0])
    assert gb10[0]["source"]["registry"].startswith(
        "https://download.pytorch.org/whl/cu"
    )


def _on_jetson():
    return (
        sys.platform == "linux"
        and platform.machine() == "aarch64"
        and "tegra" in platform.release()
    )


def test_preload_is_a_noop_where_the_wheels_are_absent(tmp_path):
    assert (
        preload_undeclared_torch_libs(tmp_path) == []
    )  # nothing there, nothing raised


@pytest.mark.skipif(
    not _on_jetson(), reason="not a Jetson: the SBSA wheels are not resolved here"
)
def test_on_jetson_the_libraries_load_and_torch_sees_the_gpu():
    loaded = preload_undeclared_torch_libs()
    assert len(loaded) == len(UNDECLARED_TORCH_LIBS), loaded
    import torch

    assert torch.cuda.is_available()
    assert torch.version.cuda is not None

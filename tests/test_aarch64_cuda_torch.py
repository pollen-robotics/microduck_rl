"""On linux-aarch64 (DGX Spark / GB10) PyPI's torch wheel is CPU-ONLY:
torch.version.cuda is None -> torch.cuda.device_count() == 0 -> mjlab's
select_gpus() indexes an empty list and dies with
`IndexError: list index out of range` BEFORE the first training step
(mjlab/utils/gpu.py:70).

The fix (pyproject.toml) routes torch to PyTorch's CUDA index, on aarch64
only. It has two SILENT break points, locked in by these tests — in both
cases `uv sync` succeeds and you only find out when you launch a run:

1. `torch` must stay a DIRECT dependency: uv applies [tool.uv.sources] to
   direct dependencies only, so deleting the `torch==...` line (which looks
   redundant, since torch already comes in via mjlab/rsl_rl) makes the
   source binding a no-op without any warning.
2. The x86_64 resolution must stay on PyPI, otherwise HF Jobs silently
   switch wheels.
"""

import platform
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_CUDA_INDEX = "https://download.pytorch.org/whl/cu"


def _packages(name):
    lock = tomllib.loads((_ROOT / "uv.lock").read_text())
    return [p for p in lock["package"] if p["name"] == name]


def _registry(pkg):
    return pkg.get("source", {}).get("registry", "")


def _markers(pkg):
    return " ".join(pkg.get("resolution-markers", []))


def _aarch64_entry(pkgs):
    """The entry whose resolution-markers SELECT linux-aarch64."""
    hits = [
        p
        for p in pkgs
        if "platform_machine == 'aarch64'" in _markers(p)
        and "sys_platform == 'linux'" in _markers(p)
    ]
    assert len(hits) == 1, f"expected 1 aarch64 entry, found {len(hits)}"
    return hits[0]


def test_torch_is_a_direct_dependency():
    """Without this, [tool.uv.sources] for torch is a silent no-op."""
    pyproject = tomllib.loads((_ROOT / "pyproject.toml").read_text())
    deps = pyproject["project"]["dependencies"]
    assert any(d.split("=")[0].split("[")[0].strip() == "torch" for d in deps), (
        "torch must stay in [project.dependencies]: uv applies "
        "[tool.uv.sources] to DIRECT dependencies only. Removing it silently "
        "drops aarch64 back onto PyPI's CPU-only wheel."
    )


def test_torch_source_is_pinned_to_a_cuda_index_on_aarch64():
    pyproject = tomllib.loads((_ROOT / "pyproject.toml").read_text())
    uv_cfg = pyproject["tool"]["uv"]
    assert "torch" in uv_cfg.get("sources", {}), (
        "[tool.uv.sources] no longer has a torch entry -> aarch64 falls back "
        "to PyPI's CPU wheel and `train` dies with IndexError in select_gpus()."
    )
    sources = uv_cfg["sources"]["torch"]
    indexes = {p["name"]: p["url"] for p in uv_cfg.get("index", [])}
    for src in sources:
        assert "aarch64" in src["marker"], "the torch source must stay aarch64-scoped"
        assert indexes[src["index"]].startswith(_CUDA_INDEX), (
            f"index {src['index']} is not a PyTorch CUDA index"
        )


def test_lockfile_routes_aarch64_torch_to_cuda_wheels():
    torch_pkgs = _packages("torch")
    aarch64 = _aarch64_entry(torch_pkgs)
    assert _registry(aarch64).startswith(_CUDA_INDEX), (
        f"torch on aarch64 comes from {_registry(aarch64)!r} — a CPU wheel. "
        "Re-run `uv lock` after checking [tool.uv.sources]."
    )
    wheels = " ".join(w["url"] for w in aarch64["wheels"])
    assert "aarch64" in wheels, "no aarch64 wheel in the aarch64 torch entry"
    assert "%2Bcu" in wheels or "+cu" in wheels, (
        "the aarch64 wheel has no +cuXXX local version -> CPU build"
    )


def test_x86_64_resolution_stays_on_pypi():
    """HF Jobs run on x86_64: their resolution must not move."""
    others = [
        p
        for p in _packages("torch")
        if "platform_machine == 'aarch64'" not in _markers(p)
    ]
    assert others, "no non-aarch64 torch entry found"
    for pkg in others:
        assert _registry(pkg) == "https://pypi.org/simple", (
            f"x86_64 torch moved to {_registry(pkg)!r} — HF Jobs would switch "
            "wheels."
        )
        assert "+cu" not in pkg["version"], "x86_64 torch must not be CUDA-pinned"


def test_torch_version_identical_across_platforms():
    """The fix changes only the wheel's SOURCE, not its version: the CUDA
    index carries newer builds than the PyPI pin, so a `>=` drags torch
    2.9.1 -> 2.13.0 with nothing having validated that bump."""
    versions = {p["version"].split("+")[0] for p in _packages("torch")}
    assert len(versions) == 1, f"torch versions diverge across platforms: {versions}"


def _on_spark():
    return (
        sys.platform == "linux"
        and platform.machine() == "aarch64"
        and shutil.which("nvidia-smi") is not None
        and subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0
    )


@pytest.mark.skipif(not _on_spark(), reason="not a linux-aarch64 machine with a GPU")
def test_installed_torch_actually_sees_the_gpu():
    """Direct reproduction of the crash: this is exactly what select_gpus() reads."""
    import torch

    assert torch.cuda.device_count() > 0, (
        f"torch {torch.__version__} (cuda={torch.version.cuda}) sees no GPU "
        "although nvidia-smi reports one -> select_gpus() will raise IndexError."
    )

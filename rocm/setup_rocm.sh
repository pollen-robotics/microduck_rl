#!/usr/bin/env bash
# Set up microduck_rl to train on AMD GPUs (ROCm / HIP).
#
# microduck_rl trains through mjlab -> mujoco-warp -> NVIDIA Warp, which is
# CUDA-only upstream. AMD publishes a ROCm/HIP port of Warp
# (https://github.com/AMD-Ecosystem/warp, branch amd-integration). This script:
#
#   1. syncs the repo's own locked deps (uv sync) so the CUDA torch/warp and the
#      right mujoco-warp/mjlab versions are installed,
#   2. builds the AMD ROCm Warp wheel from source for your GPU arch (gfx942 =
#      MI300/MI300X by default) and installs it over the CUDA wheel,
#   3. installs a HIP-major-matched PyTorch (rocm7.0 wheel) so torch and Warp
#      share one HIP runtime in-process,
#   4. applies the in-tree dependency patches (rocm/patch_rocm.py).
#
# Verified on an MI300X (gfx942) with ROCm 10.0. It should also work on a
# ROCm 7.x host; see rocm/README.md for the version matrix and caveats.
#
# Usage:
#   ROCM_PATH=/opt/rocm bash rocm/setup_rocm.sh              # autodetect arch
#   HIP_ARCH=gfx942 ROCM_PATH=/opt/rocm/core-10.0 bash rocm/setup_rocm.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ---- resolve ROCm ---------------------------------------------------------
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
if [ ! -d "$ROCM_PATH" ]; then
  echo "ERROR: ROCM_PATH=$ROCM_PATH does not exist. Set ROCM_PATH to your ROCm install." >&2
  exit 1
fi
export ROCM_PATH
export PATH="$ROCM_PATH/bin:$PATH"
export HIP_PLATFORM=amd

if ! command -v hipcc >/dev/null 2>&1; then
  echo "ERROR: hipcc not found on PATH after adding $ROCM_PATH/bin." >&2
  exit 1
fi
echo "== ROCm: $ROCM_PATH ($(hipcc --version 2>/dev/null | head -1)) =="

# ---- resolve GPU arch -----------------------------------------------------
HIP_ARCH="${HIP_ARCH:-}"
if [ -z "$HIP_ARCH" ]; then
  if command -v rocminfo >/dev/null 2>&1; then
    HIP_ARCH="$(rocminfo 2>/dev/null | grep -oE 'gfx[0-9a-f]+' | sort -u | head -1 || true)"
  fi
fi
HIP_ARCH="${HIP_ARCH:-gfx942}"
echo "== target GPU arch: $HIP_ARCH =="

# ---- tools ----------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv not found. Install from https://docs.astral.sh/uv/ and re-run." >&2
  exit 1
fi

TORCH_ROCM_INDEX="${TORCH_ROCM_INDEX:-https://download.pytorch.org/whl/rocm7.0}"
TORCH_SPEC="${TORCH_SPEC:-torch==2.10.0}"
WARP_REF="${WARP_REF:-amd-integration}"
WARP_REPO="${WARP_REPO:-https://github.com/AMD-Ecosystem/warp.git}"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/.rocm-build}"

# ---- 1. lock-faithful sync ------------------------------------------------
echo "== [1/4] uv sync (locked deps) =="
uv sync

# ---- 2. build ROCm Warp wheel ---------------------------------------------
echo "== [2/4] building AMD ROCm Warp ($WARP_REF) for $HIP_ARCH =="
mkdir -p "$BUILD_DIR"
if [ ! -d "$BUILD_DIR/warp/.git" ]; then
  git clone --depth 1 -b "$WARP_REF" "$WARP_REPO" "$BUILD_DIR/warp"
fi
pushd "$BUILD_DIR/warp" >/dev/null
# build in an isolated venv so build-time deps do not touch the project venv
uv venv --python 3.12 "$BUILD_DIR/venv"
# shellcheck disable=SC1091
source "$BUILD_DIR/venv/bin/activate"
uv pip install numpy setuptools wheel build
python build_lib.py --hip-arch="$HIP_ARCH"
python -m build --wheel --no-isolation
WARP_WHEEL="$(ls -t "$BUILD_DIR"/warp/dist/warp_lang-*rocm*.whl | head -1)"
deactivate
popd >/dev/null
echo "   built: $WARP_WHEEL"

# ---- 3. overlay ROCm torch + ROCm warp over the CUDA wheels ---------------
echo "== [3/4] installing ROCm torch ($TORCH_SPEC) + ROCm warp =="
uv pip install --no-deps --force-reinstall --index-url "$TORCH_ROCM_INDEX" "$TORCH_SPEC"
uv pip install --no-deps --force-reinstall "$WARP_WHEEL"

# ---- 4. apply dependency patches ------------------------------------------
echo "== [4/4] applying ROCm dependency patches =="
uv run --no-sync python rocm/patch_rocm.py

echo
echo "== done. Verify + run through the ROCm launcher: =="
echo "   bash rocm/verify_rocm.sh"
echo "   bash rocm/train_rocm.sh Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096"

#!/usr/bin/env bash
# Launch microduck_rl training/eval on an AMD GPU (ROCm/HIP).
#
# This wrapper sets the LD_PRELOAD needed to make PyTorch and the AMD ROCm Warp
# build share a single HIP runtime in one process, then forwards all arguments
# to `uv run --no-sync train`.
#
#   bash rocm/train_rocm.sh Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096
#   bash rocm/train_rocm.sh Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 64 --agent.max-iterations 5   # smoke test
#
# Why LD_PRELOAD: the ROCm PyTorch wheel bundles its own libamdhip64 /
# libhsa-runtime; the ROCm Warp build links the system ROCm libraries. Warp
# needs versioned HIP symbols (hip_7.1) that torch's bundled copy may lack.
# Preloading the SYSTEM ROCm hip/hsa/hiprtc forces both onto one runtime that
# satisfies both. Override ROCM_PATH if your ROCm lives elsewhere.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
export ROCM_PATH
export PATH="$ROCM_PATH/bin:$PATH"
export HIP_PLATFORM=amd

_lib() {
  # find a versioned lib under the ROCm lib dir, print first match
  local pat="$1"
  ls "$ROCM_PATH"/lib/${pat} 2>/dev/null | head -1
}

HIP_LIB="$(_lib 'libamdhip64.so.7')"
HSA_LIB="$(_lib 'libhsa-runtime64.so.1')"
RTC_LIB="$(_lib 'libhiprtc.so.7')"

PRELOAD=""
for L in "$HIP_LIB" "$HSA_LIB" "$RTC_LIB"; do
  [ -n "$L" ] && PRELOAD="${PRELOAD:+$PRELOAD:}$L"
done
if [ -n "$PRELOAD" ]; then
  export LD_PRELOAD="${PRELOAD}${LD_PRELOAD:+:$LD_PRELOAD}"
  echo "== LD_PRELOAD=$LD_PRELOAD ==" >&2
else
  echo "WARNING: could not find ROCm hip/hsa/hiprtc libs under $ROCM_PATH/lib;" >&2
  echo "         running without LD_PRELOAD (may fail with a HIP symbol error)." >&2
fi

exec uv run --no-sync train "$@"

#!/usr/bin/env bash
# Verify the ROCm/HIP setup: torch sees the AMD GPU, Warp sees it, both work in
# one process, and all dependency patches are applied. Exits non-zero on failure.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
export ROCM_PATH
export PATH="$ROCM_PATH/bin:$PATH"
export HIP_PLATFORM=amd

_lib() { ls "$ROCM_PATH"/lib/$1 2>/dev/null | head -1; }
PRELOAD=""
for L in "$(_lib 'libamdhip64.so.7')" "$(_lib 'libhsa-runtime64.so.1')" "$(_lib 'libhiprtc.so.7')"; do
  [ -n "$L" ] && PRELOAD="${PRELOAD:+$PRELOAD:}$L"
done
[ -n "$PRELOAD" ] && export LD_PRELOAD="${PRELOAD}${LD_PRELOAD:+:$LD_PRELOAD}"

echo "== patch status =="
uv run --no-sync python rocm/patch_rocm.py --check

echo "== torch + warp coexistence =="
uv run --no-sync python - <<'PY'
import torch
assert torch.cuda.is_available(), "torch.cuda.is_available() is False (HIP not visible)"
dev = torch.cuda.get_device_name(0)
x = (torch.ones(1024, device="cuda") * 2).sum().item()
assert x == 2048.0, f"torch matmul wrong: {x}"
print(f"  torch OK on: {dev} (hip {torch.version.hip})")

import warp as wp
d = wp.get_device("cuda:0")
assert d.is_cuda, "warp did not see a CUDA/HIP device"
s = wp.Stream(d)  # this is the call that fails when HIP runtimes clash
print(f"  warp OK on: {d}")
print("  wp.context alias:", hasattr(wp, "context"))
print("BOTH_OK")
PY

echo "== all checks passed =="

# Training microduck_rl on AMD GPUs (ROCm / HIP)

microduck_rl trains through **mjlab → mujoco-warp → NVIDIA Warp**, which targets
CUDA. This directory makes the same training run on **AMD Instinct GPUs** using
AMD's ROCm/HIP port of Warp, with no changes to the physics, task, or training
code (the invariants in `AGENTS.md` are untouched). Everything here is additive
and lives under `rocm/`.

**Verified:** `Mjlab-Velocity-Flat-MicroDuck`, 4096 envs, on an **MI300X
(gfx942), ROCm 10.0**. Training reaches `exit 0`, mean reward climbs across
iterations, and `rocm-smi` shows **100% GPU utilization with ~13 GB VRAM**
sustained through the run (dropping to idle the instant it ends), confirming the
physics and PPO update run on the GPU, not a CPU fallback.

## What enables it

[AMD-Ecosystem/warp](https://github.com/AMD-Ecosystem/warp), branch
`amd-integration`, is a real ROCm/HIP build of Warp (v1.13 base, rocWMMA MFMA
tile ops). Upstream `NVIDIA/warp` currently has only work-in-progress HIP shims,
so this fork is the working path today.

Given that Warp build plus a HIP-major-matched PyTorch, four small
incompatibilities remain in the installed dependencies. `patch_rocm.py` fixes
them idempotently in the active venv's `site-packages`:

| # | Symptom | Fix |
|---|---------|-----|
| 1 | `RuntimeError: Failed to create stream ... out of memory` at `wp_cuda_stream_create`, or `version 'hip_7.1' not found` | Use a HIP-major-7 torch (`torch==2.10.0+rocm7.0`) and `LD_PRELOAD` the **system** ROCm hip/hsa/hiprtc so torch and Warp share one HIP runtime (`train_rocm.sh` / `setup_rocm.sh`) |
| 2 | `AttributeError: module 'warp' has no attribute 'context'` | `.pth` + shim aliasing `warp.context` → `warp._src.context` (the fork relocated it) |
| 3 | `RuntimeError: Conditional graph nodes are not supported on HIP/ROCm` | Set `opt.graph_conditional = False` in mjlab so the mujoco-warp solver takes its native Python-loop branch |
| 4 | `RuntimeError: Failed to create CUDA texture ... error 801: operation not supported` | Guard mujoco-warp's texture creation with its existing `use_textures` flag (raycast-only contexts do not need hipArray textures) |

Fixes 2–4 live in dependencies (`warp`, `mjlab`, `mujoco_warp`) that install into
`site-packages`, so they are applied by a post-install patcher rather than as
edits to this repo's own source. `patch_rocm.py --check` reports status and is
safe to re-run after any `uv sync`.

## Quickstart

Requirements: an AMD GPU (gfx942 verified), a ROCm install (7.x or 10.0), Git,
and [uv](https://docs.astral.sh/uv/).

```bash
# 1. build ROCm Warp, install ROCm torch, apply patches (autodetects arch + ROCm)
ROCM_PATH=/opt/rocm bash rocm/setup_rocm.sh
#    e.g. on a ROCm 10.0 box: ROCM_PATH=/opt/rocm/core-10.0 HIP_ARCH=gfx942 bash rocm/setup_rocm.sh

# 2. verify torch + Warp coexist on the GPU and all patches are present
bash rocm/verify_rocm.sh

# 3. smoke test (always run first, 64 envs, 5 iters)
bash rocm/train_rocm.sh Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 64 --agent.max-iterations 5

# 4. real training
bash rocm/train_rocm.sh Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096
```

`train_rocm.sh` is a thin wrapper: it sets `LD_PRELOAD` to the system ROCm
hip/hsa/hiprtc libraries, then execs `uv run --no-sync train "$@"`. Any argument
you would pass to `train` works unchanged. `export`, `play`, and `infer_policy`
work the same way once the environment from `train_rocm.sh` is applied.

## Environment overrides

| Variable | Default | Meaning |
|----------|---------|---------|
| `ROCM_PATH` | `/opt/rocm` | ROCm install prefix (e.g. `/opt/rocm/core-10.0`) |
| `HIP_ARCH` | autodetected via `rocminfo`, else `gfx942` | GPU arch to compile Warp for |
| `TORCH_ROCM_INDEX` | `https://download.pytorch.org/whl/rocm7.0` | PyTorch ROCm wheel index |
| `TORCH_SPEC` | `torch==2.10.0` | torch version to install |
| `WARP_REF` | `amd-integration` | branch/tag of the AMD Warp fork to build |

## Version matrix that works

| Component | Version | Note |
|-----------|---------|------|
| Warp | `1.13.0+rocm.0` (AMD fork) | built from `amd-integration` for `gfx942` |
| PyTorch | `2.10.0+rocm7.0` | SONAME `libamdhip64.so.7`, HIP major matches Warp |
| mujoco-warp | `3.8.1` | resolved by `uv sync`; uses no `grid_stride` (needs warp ≥ 1.14) |
| mujoco | `3.10.0` | via `uv sync` |
| mjlab | `1.3.0` | repo pin |
| ROCm | `10.0` (also builds on 7.x) | system install providing `libamdhip64.so.7` |

The pin that matters: mujoco-warp ≥ 3.10 calls `wp.kernel(grid_stride=...)`,
which needs Warp ≥ 1.14. The AMD fork is 1.13-based, so let `uv sync` keep
mujoco-warp at the locked 3.8.1 (0 `grid_stride` uses) rather than upgrading it.

## Limitations

- **No camera sensors.** Fix 4 disables hipArray textures, which are used by
  camera rendering. Raycast sensors (height scans used by the velocity task)
  work; camera-based observations do not on ROCm yet.
- **Conditional CUDA graphs are off** (fix 3). The solver uses a Python-loop
  fallback. This is correct and stable; a future ROCm Warp with conditional
  graph support could re-enable it for extra throughput.
- Verified on gfx942 (MI300/MI300X). Other archs should work if the Warp fork
  supports them; set `HIP_ARCH` accordingly.

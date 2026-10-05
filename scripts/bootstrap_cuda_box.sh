#!/usr/bin/env bash
# Bootstrap a CUDA cloud box for microduck_rl (MJLab / MuJoCo Warp).
#
# WHY THIS EXISTS
#   The jump/MJLab configs have never been constructed on real hardware — every
#   check so far was static. This script is the cheapest way to turn that
#   "unvalidated" into "verified": it installs the pinned stack and runs the
#   first-run checklist.
#
# USAGE (on a fresh Linux box that has an NVIDIA GPU + driver)
#   git clone -b feature/cross-stack-parity-audits https://github.com/hzzqq/microduck_rl
#   cd microduck_rl
#   bash scripts/bootstrap_cuda_box.sh
#
# COST NOTE
#   Do the clone + any edits in the platform's "no-GPU" mode (~0.1 CNY/h).
#   Only switch to GPU billing right before running this script, and shut the
#   instance off as soon as it finishes.
set -euo pipefail

say() { printf '\n==> %s\n' "$*"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- 0. location
[ -f pyproject.toml ] || die "run this from the microduck_rl repo root"
grep -q 'name = "mjlab-microduck"' pyproject.toml \
  || die "this does not look like microduck_rl"

# ------------------------------------------------------------- 1. GPU present
say "1/6  Checking that an NVIDIA GPU is actually visible"
nvidia-smi || die "no GPU visible — is this a GPU instance with a driver?"

# ------------------------------------------------------------------ 2. uv
say "2/6  Installing uv (it manages the pinned Python 3.12 for us)"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

# ------------------------------------------------------- 3. Python 3.12 only
# pyproject requires >=3.12,<3.13 (bam pins <3.13). Do NOT let the box's
# system Python (often 3.10) drive the install.
say "3/6  Pinning Python 3.12"
uv python install 3.12
uv venv --python 3.12

# ------------------------------------------------------------ 4. dependencies
# mjlab==1.3.0, warp-lang==1.12.0, torch==2.9.1, better-actuator-models (git).
# On x86_64 the PyPI torch wheel already bundles CUDA via nvidia-*-cu12.
say "4/6  Installing project + pinned deps (this is the slow step)"
uv sync

# --------------------------------------------------- 5. the CUDA first-run gate
say "5/6  CUDA first-run checklist (jump cfg + -Backlash variant)"
set +e
uv run python scripts/jump_first_run.py 2>&1 | tee jump_first_run.log
JUMP_RC=${PIPESTATUS[0]}
set -e

# --------------------------------------------------------- 6. MJX parity plan
say "6/6  MJX parity: offline plan + execution status"
uv run python - <<'PY' 2>&1 | tee mjx_parity.log
from mjlab_microduck.tasks.mjx_parity_migration import (
    build_migration_plan,
    execute_migration,
)

plan = build_migration_plan()
print("terms needing CPU <-> MJX numeric parity:", len(plan.terms))
for t in plan.terms:
    print("  -", t["name"], "->", t["mjlab_func"], "covered=", t["covered"])
print()
print("execute_migration():", execute_migration(None))
PY

# ------------------------------------------------------------------- summary
say "Finished"
if [ "$JUMP_RC" -ne 0 ]; then
  echo "jump_first_run.py exited non-zero ($JUMP_RC) — see jump_first_run.log"
else
  echo "jump_first_run.py passed"
fi
echo
echo "Logs written:  jump_first_run.log   mjx_parity.log"
echo "IMPORTANT: copy these down BEFORE you shut the instance off."
echo "Also confirm the PHASE terms (jump_airborne / jump_apex) show signal —"
echo "total reward alone does NOT prove it jumps rather than squats."

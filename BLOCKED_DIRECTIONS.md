# Blocked directions — GPU / mocap dependent (⑥⑧)

Two of the eight "do-all" directions cannot *execute* in this sandbox. They are
scaffolded with runnable offline logic + explicit `BLOCKED` guards so the block
is machine-asserted (red CI on a wrong call), not a silent no-op. Execution
needs a decision and resources from the boss.

## ⑥ MJX parity-first migration (GPU)

**What's shipped (runnable offline):**
`microduck_rl/src/mjlab_microduck/tasks/mjx_parity_migration.py`
- `build_migration_plan()` enumerates every reward term requiring CPU↔MJX
  numeric parity, read straight from the parity gate `REGISTRY` (so it can
  never drift from the CI backstop).
- `execute_migration()` is the real alignment loop, guarded to return
  `{"status": "BLOCKED"}` when `mujoco_warp` / CUDA is absent.

**Blocked on:** a CUDA host with `mujoco_warp` (MJX) installed.
**Boss decision:** point me at the GPU box (or confirm the existing CUDA host
is where this runs) so `execute_migration(env_factory)` can iterate the parity
loop term-by-term.
**Unblock checklist:**
1. CUDA host reachable; `import mujoco_warp` succeeds.
2. One CPU behavior promoted to MJX at a time; parity within tolerance before
   it enters the deployable contract.
3. `reward_parity_gate` stays green across both stacks.

## ⑧ Ablation self-diagnoser (mocap / robot)

**What's shipped (runnable offline):**
`microduck-lab/microduck_local/src/microduck_local/ablation_diagnoser.py`
- `rank_ablation_degradations()` ranks ablation variants by degradation and
  attributes each to the removed term; flags noise-level degradations. Pure
  Python, unit-tested on synthetic metrics.
- `train_diagnoser()` is the learned mapping (degradation→cause), guarded to
  return `{"status": "BLOCKED"}` without a labeled corpus.

**Blocked on:** a mocap capture / real-robot rollout corpus of ablation runs
(labeled with which term/obs was removed).
**Boss decision:** authorize mocap capture (or sim2real transfer) of ablation
rollouts so `train_diagnoser(corpus)` can learn the mapping.
**Unblock checklist:**
1. Ablation runs collected with consistent baselines.
2. `rank_ablation_degradations` validated against the collected set.
3. `train_diagnoser` trained and its attributions spot-checked.

## Cross-cutting risks (documented, not machine-doable here)

- **6 configs' old perf numbers are suspect** — trained under the wrong
  `feet_flat` sign before the `f0a1b07` fix. Need a CUDA re-run to re-baseline.
- **CPU-stack jump stall at v5** (~10 low hops) — unconfirmed whether related
  to `feet_flat`; re-check after the fix lands on CPU.
- **Boss to run `scripts/jump_first_run.py`** on the CUDA box to validate the
  `-Backlash` variant end-to-end.

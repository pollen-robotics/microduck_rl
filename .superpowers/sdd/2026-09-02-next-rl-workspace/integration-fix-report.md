# Next RL integration-fix report

Date: 2026-09-02

Branch: `feature/next-rl`

Baseline: `e0f8393`

Commit: `fix(next-rl): close integration provenance gaps`

Follow-up: `fix(next-rl): reject static evidence and recover lost starts`

## Outcome

All eight final integration findings are closed without adding dependencies,
starting training, contacting Nitro, exposing a public start command, or
publishing/deploying a policy.

1. Review construction, deserialization, verification, and the CLI boundary now
   run the existing 61-observation/14-action ONNX preflight and bind its exact
   SHA-256. Hash-consistent arbitrary bytes named `.onnx` are rejected.
2. Review accepts five renderer-authored immutable JSON sidecars rather than
   caller-invented clip bindings. Each sidecar binds role, scenario, seed,
   policy, evaluation, renderer revision, video path, and video SHA-256; the
   payload must be an MP4 with at least two decodable temporal frames. Static
   images, one-frame MP4 files, and non-video payloads are rejected. Sidecars
   are written create-once and atomically by `write_renderer_evidence`.
3. `NitroRunner.prepare` compares the manifest code digest with
   `sha256(captured_commit.encode("ascii"))` immediately after source snapshot
   and before adapter/network interaction.
4. A learned-capability improvement requires a higher semantic version and an
   explicit reason. Planning selects only a loadable current learned reference;
   approval rechecks version and exact current-policy baseline under the lock,
   supersedes the exact prior version, and leaves one active learned version.
5. Shipped ONNX policies are described and handled only as evaluation/comparison
   references. They never produce a warm-start disposition or parent training
   digest; training starts from a new policy until real checkpoint staging and
   loading exists.
6. Detached supervisor state persists PID, Linux process start identity, and
   exact supervisor command/job digest. Exact live identities remain
   idempotent; dead, reused, mismatched, and legacy PID-only records recover
   under the existing singleton locks. Status inspection revalidates a
   pending/spawned supervisor even after its start request was acknowledged and
   removed; a lost identity becomes an explicit retryable failure, allowing a
   newly reserved start to recover exactly once.
7. Metric summaries now retain the direction-aware worst scenario: minimum for
   minimum thresholds and maximum for maximum thresholds. Promoted capability
   evidence uses those conservative values.
8. `NitroRunner.start` requires an `ExperimentStore`, reserves before transfer,
   blocks competing owners, supports same-owner retry after an uncertain
   response, and reconciles local lifecycle status from remote state. Failed
   pre-launch transfers release only the owning reservation.

## Changed areas

- Core: `evaluation.py`, `review.py`, `capabilities.py`, `promotion.py`,
  `experiments.py`, `runner.py`, and `cli.py`.
- Remote lifecycle: `scripts/next_rl_remote_job.py`.
- Contracts and fixtures: Next RL tests plus `tests/test_next_rl_support.py`
  with genuine tiny ONNX and decodable MP4 builders.
- Operator surface: `README.md`, `docs/next-rl-workspace.md`, and
  `examples/skills/one-leg-hello.json`.

## Verification

- Baseline before changes:
  `uv run --with pytest pytest tests/test_next_rl_*.py tests/test_next_rl_readiness.py -q`
  — 258 passed.
- Focused RED/GREEN tests covered fake ONNX, text/tampered/unbound video
  evidence, static images and one-frame MP4 files, source-digest mismatch with
  zero transport, semantic-version lifecycle, truthful train-new planning,
  dead/reused/legacy supervisor PID recovery before and after request removal,
  direction-aware aggregation, concurrent experiment reservation,
  response-loss retry, and lifecycle synchronization.
- Final required suite:
  `uv run --with pytest pytest tests/test_next_rl_*.py tests/test_next_rl_readiness.py -q`
  — 305 passed in 76.73 seconds after the scoped follow-up review corrections.
- Compilation:
  `uv run python -m compileall -q src/mjlab_microduck/next_rl scripts/next_rl_remote_job.py tests/test_next_rl_*.py`
  — passed.
- Import/unused checks: Ruff `I,F401` passed on the original integration fix.
  Ruff was not installed for the scoped follow-up, so it was not fetched or
  added; compilation and the full suite exercised the new imports instead.
- `git diff --check` — passed.

## Residual risk and explicit boundaries

- Renderer sidecars make asserted provenance immutable and tamper-evident, but
  cannot independently prove that a renderer showed a truthful simulator
  rollout; human review and trusted renderer execution remain required.
- ONNX preflight proves loadability, dimensions, finite output, and nontrivial
  smoke behavior, not simulator performance or normalization correctness.
- No checkpoint warm-start path is implemented. This is intentional: no shipped
  loadable checkpoint exists and the runner does not stage or load one.
- Source identity intentionally excludes dirty and untracked worktree bytes;
  preparation snapshots the captured committed tree.
- Persisted review bundles created under the older clip schema or older
  minimum-metric aggregation semantics must be regenerated and reviewed.

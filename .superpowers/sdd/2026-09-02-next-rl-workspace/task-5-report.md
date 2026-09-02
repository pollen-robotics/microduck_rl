# Task 5 report: visual review and human learned-state promotion

## Scope

- Added `next_rl.review` with immutable clip manifests, canonical evaluation
  evidence, policy/clip digest checks, and optional seed-matched baseline
  context.
- Added `next_rl.promotion` with review-pending, approval, rejection, audit,
  and supersession transitions. It does not import or call publishing code.
- Added focused behavioral contracts for review integrity and promotion.

## RED evidence

1. Before either module existed:

   ```text
   ModuleNotFoundError: No module named 'mjlab_microduck.next_rl.review'
   ModuleNotFoundError: No module named 'mjlab_microduck.next_rl.promotion'
   ```

2. Before evaluation JSON was cross-checked against a bundle's claimed pass
   status:

   ```text
   FAILED test_forged_passing_flag_cannot_admit_a_failed_evaluation
   Failed: DID NOT RAISE PromotionError
   1 failed, 5 passed
   ```

## GREEN evidence

- Focused review/promotion suite:

  ```text
  uv run --with pytest pytest tests/test_next_rl_review.py tests/test_next_rl_promotion.py -q
  15 passed in 0.04s
  ```

- Relevant combined Next RL suite:

  ```text
  uv run --with pytest pytest tests/test_next_rl_*.py -q
  93 passed in 1.35s
  ```

- Full repository test suite:

  ```text
  uv run --with pytest pytest tests/ -q
  289 passed, 1 skipped in 6.38s
  ```

## Commit

`feat(next-rl): require human review for learned skills`

## Concerns

- `uv run ruff ...` could not run because the Ruff executable is not installed
  in this environment (`Failed to spawn: ruff`).
- Promotion is durable only under its caller-provided local root; publishing
  remains outside this state machine, as required.

## Fix round 1 — durable trust boundary and inventory state

### RED evidence

The added review regressions failed before the hardening work because direct
bundles could omit required roles, clips could name arbitrary scenarios, and
baseline evidence was not retained for later verification. The promotion
regressions also failed before `PromotionStore` accepted a durable temp root:

```text
TypeError: PromotionStore.__init__() takes 1 positional argument but 2 were given
```

The final lifecycle-history regression then failed before explicit transition
auditing:

```text
assert [] == ['available', 'validated', 'requested']
```

### GREEN evidence

```text
uv run --with pytest pytest tests/test_next_rl_review.py tests/test_next_rl_promotion.py -q
21 passed in 0.08s

uv run --with pytest pytest tests/test_next_rl_*.py -q
99 passed in 1.49s

uv run --with pytest pytest tests/ -q
295 passed, 1 skipped in 6.98s
```

### Fixes

- `ReviewBundle.verify()` now re-validates canonical candidate and baseline
  reports, requires exactly one clip per mandatory role, binds every role to
  the deterministic evaluated scenario/seed, and re-hashes mandatory policy,
  baseline-policy, and clip files.
- Promotion now requires a matching `Capability` ID/version, writes canonical
  durable state under a caller-provided root using an exclusive lockfile and
  atomic replacement, and exposes persisted capability records through
  `inventory()` for planner visibility.
- Approval moves the inventory record to learned with review-bundle approval
  provenance; concurrent approvers serialize under the same lock so only one
  succeeds. Rejection restores validated state (or preserves an earlier learned
  policy) and a re-request retains the full audit history.

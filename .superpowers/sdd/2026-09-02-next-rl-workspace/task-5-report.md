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
- Promotion is deliberately local/in-memory; publishing remains outside this
  state machine, as required.

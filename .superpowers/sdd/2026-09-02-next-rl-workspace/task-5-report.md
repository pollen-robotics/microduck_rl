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

## Fix round 2 — semantic reports, validated sources, and lock recovery

### RED evidence

Before this round, a forged mandatory threshold result reached only a derived
metric-summary mismatch rather than an independent threshold-semantics check;
multiple scenarios in a family were rejected outright; and review requests
could start from an unvalidated available capability. The focused RED run
reported these concrete failures:

```text
FAILED test_bundle_recomputes_mandatory_threshold_semantics
FAILED test_multi_seed_family_clips_choose_the_stable_lowest_seed
FAILED test_passing_evaluation_without_human_approval_is_review_pending
AttributeError: 'PromotionStore' object has no attribute 'validate'
```

### GREEN evidence

```text
uv run --with pytest pytest tests/test_next_rl_review.py tests/test_next_rl_promotion.py -q
25 passed in 0.10s

uv run --with pytest pytest tests/test_next_rl_*.py -q
103 passed in 1.70s

uv run --with pytest pytest tests/ -q
299 passed, 1 skipped in 8.51s
```

### Fixes

- Canonical evaluation parsing now recomputes each threshold's pass/violation
  semantics and the aggregate mandatory pass flag. Candidate reports must pass;
  baseline reports may fail, but their declared result must still match the
  threshold conjunction.
- Family-role clips select the lowest `(seed, scenario_id)` evaluated scenario,
  while worst-case selection retains normalized-violation then scenario-ID
  semantics.
- `validate()` is now the only `available -> validated` path. `request_review()`
  accepts only the exact persisted validated evidence, and rejection restores
  that same candidate for audited re-request.
- Validation writes the exact canonical evaluation JSON once to a real report
  path before recording the corresponding `EvaluationRef`; conflicting rewrites
  are refused.
- Promotion locks record owner, PID, and identity, recover only aged locks from
  a dead PID, and leave an aged live-PID lock untouched. The module was also
  reformatted into conventional readable control flow.

## Fix round 3 — complete threshold contracts and advisory locking

### RED evidence

The additional regressions demonstrated that an empty or selectively omitted
threshold result could still be accepted, an approval proceeded after its
persisted evaluation file was deleted, and the previous exclusive-create lock
remained unusable after a descriptor owner released it:

```text
FAILED test_bundle_requires_complete_nonempty_threshold_contracts
FAILED test_missing_or_mutated_persisted_evaluation_blocks_approval_and_rejection
TimeoutError: timed out waiting for promotion lock
```

### GREEN evidence

```text
uv run --with pytest pytest tests/test_next_rl_review.py tests/test_next_rl_promotion.py -q
29 passed in 0.15s

uv run --with pytest pytest tests/test_next_rl_*.py -q
107 passed in 1.58s

uv run --with pytest pytest tests/ -q
303 passed, 1 skipped in 7.03s
```

### Fixes

- Every scenario now requires non-empty threshold results; result keys are
  unique, cover the scenario metrics, bind the containing scenario, and share
  an identical metric/unit/direction/limit/mandatory contract across all
  scenarios. Aggregate pass status remains independently recomputed.
- Failed baselines are valid comparison context only when their threshold
  results and top-level pass value agree; an inconsistent failed baseline is
  rejected.
- Pending approval/rejection re-reads the `EvaluationRef.report_path` and
  requires a non-empty regular file whose canonical bytes exactly match the
  review bundle before changing any capability state.
- The lockfile is now a persistent advisory `fcntl.flock` target with bounded
  nonblocking retry. Closing or crashing its descriptor releases the lock
  automatically, eliminating stale-file unlink races. Distinct pending versions
  are also covered by a serialized-approval test that leaves one learned policy
  per skill.

## Fix round 4 — authoritative SkillSpec threshold binding

### RED evidence

Before this round, report consistency was self-contained: an attacker could
remove the same failed metric from every scenario and leave an internally
consistent, but incomplete, report. The initial TDD run failed at collection
because `ReviewBundle.build(..., spec=...)` did not yet accept authoritative
specification evidence:

```text
TypeError: ReviewBundle.build() got an unexpected keyword argument 'spec'
```

### GREEN evidence

```text
uv run --with pytest pytest tests/test_next_rl_review.py tests/test_next_rl_promotion.py -q
30 passed in 0.15s

uv run --with pytest pytest tests/test_next_rl_*.py -q
108 passed in 1.75s

uv run --with pytest pytest tests/ -q
304 passed, 1 skipped in 7.92s
```

### Fixes

- Bundles now retain canonical `SkillSpec` JSON and its digest. Build and every
  later verification bind skill ID/version, held-out scenario families,
  evaluation seeds, and every threshold result to that exact specification.
- Each scenario must have exactly one result for every spec metric name, with
  matching unit/direction/limit/mandatory fields. Duplicate names are refused
  even when their other fields differ; non-threshold diagnostic metrics remain
  permitted in raw scenario metrics.
- Promotion inherits the check through `ReviewBundle.verify()` at validation,
  request, approval, and rejection.
- Distinct-version approvals now race through a barrier and leave exactly one
  learned record and inventory capability for the skill, with the other marked
  superseded.

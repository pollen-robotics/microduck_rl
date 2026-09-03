# Task 7 report: guarded Next RL operator CLI

## Scope

- Added the JSON-only `next-rl` CLI with `inventory`, `plan`, `prepare`,
  `status`, `review`, `approve`, and `reject` commands.
- Registered the console script and ignored local `/.next-rl/` runtime state.
- Used temporary `NEXT_RL_HOME` directories and injected runner factories in
  every CLI test. No live SSH/Nitro operation, training run, cancellation,
  upload, publish, or network request was performed.

## RED evidence

1. The declared-entry-point contract initially failed:

   ```text
   KeyError: 'next-rl'
   ```

2. Before the CLI module existed, the plan contract failed during collection:

   ```text
   ModuleNotFoundError: No module named 'mjlab_microduck.next_rl.cli'
   ```

3. Before the status command was implemented:

   ```text
   assert 2 == 0
   ```

4. Before rejection was exposed through the durable promotion store:

   ```text
   assert 2 == 0
   ```

5. Before inventory was exposed:

   ```text
   assert 2 == 0
   ```

## GREEN evidence

```text
uv sync
Resolved 154 packages in 2ms
Audited 126 packages in 2ms

uv run --with pytest pytest tests/test_next_rl_cli.py -q
9 passed in 0.07s

uv run --with pytest pytest tests/test_next_rl_*.py -q
222 passed in 7.84s

uv run --with pytest pytest tests/ -q
418 passed, 1 skipped in 14.13s
```

`uv run ruff check ...` could not run because Ruff is not installed in this
environment. Python compilation of the added CLI and test completed, and
`git diff --check` reported no whitespace errors.

## Commit

`feat(next-rl): expose guarded workspace commands`

## Concerns

- `prepare` intentionally only creates the experiment and invokes the runner's
  preparation boundary; it contains no start action and never calls
  `runner.start`. A real preparation requires an explicitly configured
  `NEXT_RL_NITRO_SSH_ALIAS`; tests inject a fake runner, so no transport occurs.
- Review evaluates the already-bound durable `ReviewBundle`; the command does
  not publish a policy or change the learned state without the exact persisted
  review record and a non-empty reviewer.

## Fix round 1/5 — review package and fail-closed status

### RED evidence

The new review regressions failed before the CLI changes:

```text
FAILED test_prepare_is_dry_run_by_default
AssertionError: assert 'prepared' == 'planned'

FAILED test_status_allowlists_only_validated_checkpoint_primitives
AssertionError: nested checkpoint token metadata reached JSON output

FAILED test_review_then_approval_requires_a_persisted_record_and_reviewer
assert 2 == 0

FAILED test_warm_start_plan_uses_task8_parent_capability_json
KeyError: 'parent_capability_id'
6 failed, 5 passed
```

The exact-candidate mutation then exposed that a rejected candidate could be
reopened with a changed, otherwise-valid bound SkillSpec:

```text
assert 0 == 2
```

### Changes

- `review` now requires exact `--capability`, `--skill`, and `--evaluation`
  JSON inputs plus five evaluator-produced video paths; it builds the
  `ReviewBundle` itself, binds clips to deterministic scenario/seed choices,
  optionally binds a baseline evaluation, persists canonical bundle JSON below
  `NEXT_RL_HOME/review-bundles/`, and returns bundle path/digest plus record ID.
- A rejected validated candidate may be reopened only with the exact persisted
  bundle digest and policy binding. Different capability, spec, or policy
  evidence fails before promotion state is modified; the original audit record
  is reused.
- `status` now allowlists validated primitive lifecycle/checkpoint fields and
  drops nested data, arbitrary metadata, transport stderr, and malformed
  checkpoint records.
- Warm-start planning emits `parent_capability_id`; successful preparation is
  reported as non-started `status: planned`.

### GREEN evidence

```text
uv sync
Resolved 154 packages in 2ms
Audited 126 packages in 1ms

uv run --with pytest pytest tests/test_next_rl_cli.py -q
13 passed in 0.08s

uv run --with pytest pytest tests/test_next_rl_*.py -q
226 passed in 6.66s

uv run --with pytest pytest tests/ -q
422 passed, 1 skipped in 13.50s
```

No start, publish, rendering, browser automation, live Nitro, training, or
network action was invoked. Ruff remains unavailable in this environment;
syntax compilation and whitespace checks passed.

### Commit

`fix(next-rl): harden review package workflow`

## Fix round 2/5 — evidence and workspace boundary hardening

### RED evidence

New regressions failed before the fix, covering all three review findings:

```text
9 failed, 12 passed
```

They demonstrated that nested checkpoint data and missing lifecycle state could
still yield success, secret-like capability/SkillSpec/evaluation/baseline
metadata could reach durable review state, and a symlinked `NEXT_RL_HOME` child
could redirect review or preparation state outside the workspace.

### Changes

- Recursively reject secret-like JSON keys at each CLI evidence boundary before
  any persistence. The tokenization matches runner credential safety while
  retaining `token_budget`, `token_count`, and `tokenizer`; rejected documents
  produce only the stable `invalid_request` response and are never redacted or
  persisted.
- Resolve `review-bundles`, `promotions`, `experiments`, and `bundles` through
  one direct-child guard. It rejects workspace/home symlinks and non-directory
  roots, then proves the resolved child remains directly below `NEXT_RL_HOME`
  before state writes or injectable factories run.
- Make `status` fail closed: a lifecycle status is mandatory and all supplied
  artifact, exit-code, and checkpoint data must conform to strict primitive
  schemas. Nested or arbitrary checkpoint fields cannot be emitted.

### GREEN evidence

```text
uv sync
Resolved 154 packages in 2ms
Audited 126 packages in 2ms

uv run --with pytest pytest tests/test_next_rl_cli.py -q
23 passed in 0.14s

uv run --with pytest pytest tests/test_next_rl_*.py -q
236 passed in 6.70s

uv run --with pytest pytest tests/ -q
432 passed, 1 skipped in 13.43s
```

`py_compile` and whitespace checks also passed. No start, publish, rendering,
browser automation, live Nitro, training, or network action was invoked.

### Commit

`fix(next-rl): harden CLI state boundaries`

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

# Task 8 readiness report

## Scope

Added the planning-only `one-leg-hello` skill example, the guarded Next RL
operator guide, a README entry point, and isolated subprocess readiness tests.
No task, reward, runner, model artifact, training, remote sync, publishing, or
hardware deployment was performed.

## RED → GREEN

- **RED:** `uv run --with pytest pytest tests/test_next_rl_readiness.py -q`
  produced two failures before the example existed. The public `plan` command
  and the injected local-only `prepare` process rejected the missing example.
- **GREEN:** after adding the schema-valid example and documentation, the same
  command passed both tests. The tests use a temporary `NEXT_RL_HOME`; the
  `prepare` and `status` subprocess uses a local injected runner that performs
  no SSH, network I/O, or training. The public `next-rl` executable covers
  inventory, plan, review, approve, reject, and the injected public CLI parser
  covers prepare and status.

## Verification

- `uv run --with pytest pytest tests/test_next_rl_readiness.py -q` — 2 passed.
- `uv run --with pytest pytest tests/test_next_rl_*.py -q` — 242 passed.
- `uv run --with pytest pytest tests/ -q` — 438 passed, 1 skipped.
- `uv run next-rl --help` and every subcommand's `--help` were rechecked against
  the guide: inventory, plan, prepare, status, review, approve, and reject.
- `uv run ruff check tests/test_next_rl_readiness.py` could not run because
  `ruff` is not installed in the project environment; it made no changes.
- Final `git diff --check` and `git status --short` verification is performed
  immediately before the commit below.

## Commit and remote sequencing

Commit subject: `docs(next-rl): document guarded training workflow`.

The safe runner archives the committed Git tree. This documentation, example,
and readiness-test commit must therefore exist before the primary agent performs
the separately authorized remote sync or bounded smoke.

## Concerns / handoff

The example's numeric values are simulation acceptance defaults, not proof of a
learned behavior. It requires five review videos, a passing ONNX-bound
evaluation, explicit human approval, and hardware calibration before a separate
deployment decision. The primary agent owns the bounded remote smoke; none was
started here.

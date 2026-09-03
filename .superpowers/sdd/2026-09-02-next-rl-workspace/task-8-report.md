# Task 8 report: guarded Next RL operator workflow

## Scope

Task 8 adds only planning data, operator documentation, a README entry point,
and isolated subprocess readiness tests. It does not implement a task or
reward, create a model artifact, start training, contact Nitro, publish, or
deploy hardware.

## Original RED → GREEN

- **RED:** before the example existed,
  `uv run --with pytest pytest tests/test_next_rl_readiness.py -q` produced two
  failures: the public `plan` command and injected local-only `prepare` process
  rejected the missing specification.
- **GREEN:** the example planned as a `warm_start` from `standing`; the public
  executable covered inventory, plan, review, approve, and reject. The
  dependency-injected `main` subprocess covered the same public parser for
  prepare and status while the fake runner prevented SSH, network I/O, and
  training.

Original commit: `docs(next-rl): document guarded training workflow`.

## Fix round 1/5 — phase semantics and local repeatability

### RED evidence

The phase-contract regression failed before the fix because the initial example
treated the right foot as raised at entry and used unscoped metrics:

```text
FAILED test_example_skill_is_a_numeric_warm_start_plan_without_training
AssertionError: required raised_wave_* and safe_exit_* metric names were absent
```

### Changes

- Made entry and safe exit explicitly require both foot contacts.
- Scoped left support, right-foot clearance/no-contact, hip cycles, knee bounds,
  trunk tilt, and forbidden contacts to `raised_wave`; right-foot contact is
  required at entry/safe exit and forbidden only in that phase.
- Added phase-qualified metric names, metric scopes, and a safe-exit two-foot
  contact ratio to make the evaluator contract unambiguous.
- Repeated the identical injected `prepare` subprocess and checked equal JSON
  output plus byte-identical immutable manifest and planned status state; the
  runner double raises if a start is attempted.
- Documented that `Mjlab-OneLegHello-MicroDuck` is an unregistered,
  non-runnable planning placeholder and that the Velocity 64-environment,
  five-iteration command is a separate workspace/GPU smoke only.
- Documented canonical review-bundle/digest inspection, exact verification, and
  opening all five bound clips before approval or rejection.

### Verification

```text
uv run --with pytest pytest tests/test_next_rl_readiness.py -q
2 passed in 0.86s

uv run --with pytest pytest tests/test_next_rl_*.py -q
242 passed in 7.63s

uv run --with pytest pytest tests/ -q
438 passed, 1 skipped in 15.18s
```

Fresh staged `git diff --check` and `git status --short` verification precede
the separate fix commit below.

### Commit

`fix(next-rl): clarify phase-aware planning workflow`

## Concerns / handoff

The phase-qualified numeric values remain simulation acceptance defaults, not
evidence that the skill exists. A future registered task/reward, passing
ONNX-bound evaluation, review of all five clips, explicit human approval, and
hardware calibration are all required before a separate deployment decision.
The primary agent owns any explicitly authorized bounded remote smoke.

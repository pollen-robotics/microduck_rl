# Task 4 implementation report

## Implemented

- Replaced the parent dispatcher, runtime handles, lifecycle workers, and emergency
  handoff with a `RuntimeProcessSupervisor`-only service.
- Made durable transitions acknowledgement driven: `VALIDATING` precedes START,
  `RUNNING` follows START ACK, and terminal delivery commits exact child evidence
  before its callback returns.
- Retained service ownership through quarantine/containment and reconcile it only
  after durable terminal state plus supervisor slot-release proof.
- Preserved idempotency, commands/renewal, cancel, paging, cached diagnostics,
  readiness/catalog masking, and HTTP behavior.
- Production composition now launches the isolated child; `main.py` does not import
  or construct MuJoCo/ONNX runtimes. API shutdown closes the supervisor.
- Corrected discrete START to carry a null lease and restored public discrete result
  normalization at the child boundary.
- Replaced the in-process service fake with an event-driven supervisor double.
  Tests asserting deleted thread/handle machinery are explicitly skipped; their
  containment cases remain covered by child/process-supervisor suites.

## TDD evidence

Initial RED:

```text
TypeError: 'FakeMicroduckRuntime' object is not callable
```

Real-process integration RED:

```text
qualified STAND create: HTTP 500
trace: OPERATION_FAILED, QUARANTINED, SIGTERM_SENT, CHILD_REAPED, NO_CHILD
```

This exposed the discrete START lease mismatch. Final GREEN:

```text
real STAND + child/protocol: 90 passed in 17.76s
required 20x gate: 1440 passed, 280 skipped in 1266.39s
full ROM sweep: 501 passed, 21 skipped, one thread-name collision
final affected regression after owner-name correction: 3 passed in 11.30s
```

Ruff passed for every changed Python file and `git diff --check` passed.

## Files changed

- `src/mjlab_microduck/rom/{service,process_service,api,main}.py`
- `src/mjlab_microduck/rom/{process_protocol,process_supervisor,runtime_child}.py`
- `tests/fakes/fake_microduck_runtime.py`
- `tests/test_rom_{service_process_integration,continuous_tasks,discrete_tasks,mujoco_runtime}.py`
- this report

## Self-review

No parent runtime fallback remains. Cached terminal persistence is idempotent, and
fresh generations remain blocked until acknowledged cleanup or exact reap.

## Scoped-review fix round 1

- Added immutable `CorrelatedTerminalDelivery` with supervisor-validated generation,
  task ID, event sequence, and terminal payload. Service persistence exact-matches
  active supervisor generation/task. `tick()` no longer replays cached terminals.
- Added direct `VALIDATING -> FAILED` durability for failed START. Timeout, crash,
  and wrong/malformed acknowledgement tests prove no `RUNNING`/`TASK_STARTED` is
  fabricated while process containment still gates the slot.
- Added one bounded in-memory pending command reservation. Durable command sequence,
  event, and deadline are written only after exact COMMAND ACK. Identical concurrent
  duplicates share the ACK/failure; blocked delivery never reports acceptance.
- Expanded the process integration file from structural checks to 21 deterministic
  service-plus-real-supervisor/fake-child tests: STAND completion, WALK renewal and
  lease timeout, cancel during START, blocked START/COMMAND/STOP, crash/protocol
  failure, cached reads during containment, reap gating/fresh generation, paging and
  idempotency, stale callback rejection, and four continuous runtime faults.
- Documented `runtimeCallTimeoutS` as the compatibility bound for duplicate command
  waiters; `pollIntervalS` remains validated solely for constructor compatibility.
- Added an explicit mapping from each removed thread-era behavior assertion to its
  process-backed integration or supervisor replacement. The remaining skips are only
  old implementation-shape tests; behavior is exercised through the process matrix.
- Tightened child terminal publication so receiving a safety terminal implies the
  child-local safety-complete barrier is already set.

Verification after fixes:

```text
process service integration: 21 passed in 15.38s
required 20x gate: 1780 passed, 280 skipped in 1749.10s (0:29:09)
child safety publication repeat: 40 passed in 2.66s
exact-HEAD full ROM suite: 519 passed, 21 skipped in 250.28s (0:04:10)
final focused review regression: 80 passed in 55.75s
Ruff: all checks passed
git diff --check: clean
```

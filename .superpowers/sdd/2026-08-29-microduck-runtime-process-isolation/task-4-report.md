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

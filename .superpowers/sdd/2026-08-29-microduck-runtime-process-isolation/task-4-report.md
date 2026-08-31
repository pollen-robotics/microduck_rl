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

## Scoped-review fix round 2

### Race fixes

- Registered the supervisor generation from the ready snapshot before START is
  submitted. A terminal sent immediately after the exact START ACK is therefore
  eligible for correlated persistence even while the public `start()` call has not
  returned. The service rejects a generation change across the ACK boundary.
- Serialized ACK-to-durable COMMAND finalization and terminal persistence with the
  service lock while keeping all child IPC outside that lock. A pending command now
  has exactly one explicit durable result or service error, and both its owner and
  identical duplicates observe that same outcome. Terminal-first ordering rejects
  the command without an accepted event; command-first ordering durably records the
  command before the terminal event.
- Restored child publication order: a bounded `SOCK_SEQPACKET` send must complete
  before `_safety_complete` is published. Lock acquisition, socket writability, and
  nonblocking send all share the fatal cleanup deadline. Send failure withholds the
  completion claim and falls back to EOF plus exact parent reap.
- Reconciled a malformed unsolicited packet after the supervisor quarantines and
  exactly reaps the child, producing durable `FAILED/RUNTIME_UNRESPONSIVE` instead of
  leaving a stale RUNNING record.

### Exact skipped-test replacement mapping

Every remaining skip is an obsolete parent-thread/direct-runtime service fixture;
the behavioral contract is exercised by these live replacements (the mapping test
imports and resolves every referenced test function):

- `test_blocked_sample_fails_closed_without_holding_service_ownership` -> blocked
  child monitor retirement + malformed process packet durability.
- `test_blocked_command_fails_closed_and_late_return_cannot_renew_ownership` ->
  blocked real-supervisor command, duplicate shared failure, and no renewal.
- `test_blocked_safe_stop_becomes_unresponsive_without_duplicate_stop_attempts` ->
  blocked real-supervisor STOP containment + child refusal of untruthful terminal.
- `test_newer_accepted_command_skips_older_delayed_publication` -> atomic
  COMMAND-ACK/durable-command ordering against a concurrent terminal.
- `test_stop_claim_invalidates_queued_commands_before_zero_publication` ->
  terminal-first COMMAND finalization with identical duplicate errors.
- `test_runtime_supervisor_bounds_twenty_four_stalled_callers` -> real process owner
  24-caller bound.
- `test_constructor_status_stall_is_bounded_and_fail_closed` -> blocked child LOAD
  readiness failure and exact reap.
- `test_cancel_during_start_repeatedly_stops_the_returned_handle` -> cancel during
  real-supervisor START + child-local blocked-START emergency zero.
- `test_watchdog_during_start_publishes_emergency_before_fifo_cleanup` -> watchdog
  during real-supervisor START + child-local blocked-START emergency zero.
- `test_start_timeout_quarantines_slot_until_late_handle_cleanup_finishes` -> blocked
  real START, responsive reads, quarantine, exact reap.
- `test_start_timeout_after_handle_registration_keeps_cleanup_quarantine` -> fresh
  generation rejected until exact reap, then succeeds.
- `test_start_timeout_retains_service_owner_until_emergency_attempt_finishes` ->
  START failure quarantine and exact child reap.
- `test_safety_operation_failure_persists_requested_terminal_and_releases_slot` ->
  blocked real STOP with durable failure only after containment.
- `test_realtime_stop_during_blocked_start_leaves_no_runtime_owner_or_control` ->
  process cancel-during-START + child emergency-zero proof.
- `test_realtime_emergency_after_final_start_check_revokes_publication` -> process
  watchdog-during-START + child emergency-zero proof.
- `test_realtime_stop_after_runtime_start_return_uses_retained_cleanup_handle` ->
  immediate post-ACK terminal correlation + child blocked-START cleanup proof.
- `test_service_tick_observes_concrete_runtime_fault_and_zeros_applied_motion` ->
  process-service fault durability, concrete MuJoCo non-finite fail-safe actuator
  disablement, and the qualified service-to-child-to-MuJoCo HTTP completion path.

The last item is intentionally a three-layer replacement: production does not
expose child MuJoCo memory to the parent merely for a test. The real child/API path
proves composition and durable results, while the concrete runtime test directly
proves zero/disable behavior under an injected non-finite state.

### TDD and verification evidence

The three focused race tests failed before production edits:

```text
immediate terminal: task never reached SUCCEEDED
COMMAND race: owner leaked IllegalTaskTransition while duplicate returned None
paused safety send: _safety_complete was already set before send release
```

After the fixes:

```text
focused process service + runtime child: 82 passed
process/supervisor/child/service/API sweep: 197 passed, 14 mapped skips
required 20x gate: 1900 passed, 280 mapped skips in 1830.41s (0:30:30)
whole ROM suite before commit: 526 passed, 21 mapped/architecture skips in 249.57s
Ruff: all checks passed
git diff --check: clean
```

Commit `db7728e9fed0e08adb92797bdde032e04a294891` was then verified without
working-tree changes: `526 passed, 21 skipped in 266.68s (0:04:26)`.

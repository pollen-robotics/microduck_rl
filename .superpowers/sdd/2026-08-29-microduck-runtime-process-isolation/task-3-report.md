# Task 3 implementation report

## Implemented

- Added `RuntimeProcessSupervisor`, a single daemon-thread owner of the child process, Unix `SOCK_SEQPACKET` socket, generation/sequence counters, and authoritative lifecycle snapshot.
- Added a bounded typed-intent queue. Public start, command, status, stop, readiness, ensure-ready, and close calls use bounded waits; snapshot reads are immutable and perform no child I/O.
- Added exact canonical request/response matching over generation, operation sequence, task identity, and response kind. Malformed, late, blocked, exited, or otherwise ambiguous operations fail closed into quarantine.
- Added exact `Popen`-object containment: SIGTERM, bounded wait, exact-PID SIGKILL escalation, `Popen.wait()`, `poll()` confirmation, exact socket close, and only then `NO_CHILD` plus `slot_releasable=True`.
- Preserved a healthy loaded child across normal start/command/status/stop cycles, including exact same-PID reuse.
- Added immutable cached status, bounded terminal callback, trace evidence, generation replacement, queue admission bounds, and idempotent bounded close.
- Added explicit child-launch ownership for inherited test descriptors so parent copies are closed immediately after spawn.

`runtime.py` did not require modification because Task 1 protocol records and existing V1 contract models provide all supervisor-facing types without importing child-local runtime handles.

## TDD evidence

Observed RED before production code existed:

```text
ModuleNotFoundError: No module named 'mjlab_microduck.rom.process_supervisor'
```

Focused GREEN:

```text
12 passed in 20.50s
```

Coverage includes healthy same-PID reuse, load/start/command/status/stop blocking, malformed/late/exited responses, SIGTERM-ignore SIGKILL escalation, exact reap before slot release, 24 concurrent callers with one owner thread and bounded admission, cached snapshots, and close from idle/running/fault paths.

## Required verification

```text
uv run --with pytest --with pytest-repeat pytest tests/test_rom_process_supervisor.py tests/test_rom_process_protocol.py tests/test_rom_supervisor_state.py -q --count=10
590 passed in 201.94s (0:03:21)

uv run ruff check src/mjlab_microduck/rom/process_supervisor.py tests/test_rom_process_supervisor.py
All checks passed!

git diff --check
(no output)
```

## Scope

Only Task 3 supervisor implementation, tests, and this report were added. Task 4 was not started and public V1 HTTP/task contracts were not changed.

## Scoped-review fix round 1

- Isolated terminal callbacks behind one bounded nonblocking handoff queue and fixed daemon delivery worker; blocking/throwing callbacks cannot block the sole process owner. Slot reuse is published only after successful handoff, and backpressure quarantines.
- `close()` now raises fail-closed unless it proves owner-thread termination and exact child reap.
- Production lifecycle edges invoke Task 1's total transition function; invalid event/state pairs quarantine.
- Timeout quarantine attempts one separately bounded canonical ZERO_AND_STOP only while transport remains trustworthy, then still requires exact reap.
- Spawn construction and inherited descriptor cleanup are exception-safe, and production exec receives only a minimal environment allowlist.
- COMMAND and SHUTDOWN acknowledgements now validate their exact acknowledged operation.
- Trace publication/read is synchronized.
- Added deterministic blocking and throwing callback regression tests.

Focused verification after fixes: `14 passed in 22.32s`; Ruff and `git diff --check` passed.

## Adversarial proof/fix round 2

- Replaced the timer-based late fake with an inherited-socket proof whose response is
  released by the supervisor's post-deadline SIGTERM handler. This proves the packet
  is genuinely late without sleeps or filesystem barriers.
- Added socket-gated exact-PID proofs for LOAD, START, COMMAND, STATUS, and
  ZERO_AND_STOP blocking; malformed response; unexpected exit; SIGTERM-ignore kill
  escalation; and stale generation injection after reap/replacement. Every captured
  PID uses a pidfd opened while that exact child is alive and is readable before the
  slot-release assertion completes.
- Added mismatched acknowledged-operation cases for HELLO, START, COMMAND, and
  SHUTDOWN, plus deterministic bounded close coverage from IDLE, STARTING, RUNNING,
  STOPPING, and a close queued across the QUARANTINED containment path.
- Added a killed-parent subprocess harness. It reports the supervisor-owned child PID
  over an inherited `SOCK_SEQPACKET`; killing only the exact harness PID makes the
  exact child pidfd readable, proving `PR_SET_PDEATHSIG` leaves no orphan.
- Exposed the already-owned child PID in the immutable SPAWNING snapshot before the
  readiness exchange, fixing the defect exposed by malformed/exit containment tests.
- Added explicit readiness-operation trace evidence so deadline and non-timeout spawn
  failures have the same auditable ordering as later guarded exchanges.

TDD evidence for the production fixes:

```text
RED: test_protocol_failure_and_unexpected_exit_reap_captured_exact_pid
     snapshot.pid was None after the gated child had received HELLO.
GREEN: 2 passed in 1.46s

RED: test_late_packet_is_released_only_by_post_deadline_sigterm
     OPERATION_TIMEOUT was absent from the readiness failure trace.
GREEN: 1 passed in 1.34s
```

Required verification:

```text
uv run --with pytest --with pytest-repeat pytest tests/test_rom_process_supervisor.py tests/test_rom_process_protocol.py tests/test_rom_supervisor_state.py -q --count=10
800 passed in 319.06s (0:05:19)

MUJOCO_GL=egl uv run --with pytest pytest tests/test_rom_runtime_child.py -q
21 passed in 9.10s

uv run ruff check src/mjlab_microduck/rom/process_supervisor.py tests/test_rom_process_supervisor.py tests/fakes/fake_runtime_child.py tests/fakes/supervisor_parent_harness.py
All checks passed!

git diff --check
(no output)
```

Files changed in this round:

- `src/mjlab_microduck/rom/process_supervisor.py`
- `tests/test_rom_process_supervisor.py`
- `tests/fakes/fake_runtime_child.py`
- `tests/fakes/supervisor_parent_harness.py`
- this report

Self-review found no remaining Task 3 proof gaps. Task 4 was not started.

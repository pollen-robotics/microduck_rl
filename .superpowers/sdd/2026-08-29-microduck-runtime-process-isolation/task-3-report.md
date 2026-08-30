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

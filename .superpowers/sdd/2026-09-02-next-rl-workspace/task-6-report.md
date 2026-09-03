# Task 6 report: safe Nitro runner and resumable lifecycle

## Scope

- Added credential-free `uv run train` argv construction with narrow task/run/
  checkpoint validation and additional-iteration resume semantics.
- Added a Git-object-backed, deterministic tracked-file archive and fixed
  fingerprint-rooted OpenSSH preparation flow.
- Added an injected OpenSSH command boundary, detached standalone remote
  supervisor, atomic lifecycle records, three-part cancellation identity, and
  stable digest-verified checkpoint synchronization.
- Added temporary-Git and fake-transport tests. No live SSH, Nitro lifecycle,
  training, upload, publishing, or network call was performed.

## RED evidence

1. Before command construction existed:

   ```text
   ModuleNotFoundError: No module named 'mjlab_microduck.next_rl.runner'
   ```

2. Before transport/lifecycle interfaces existed:

   ```text
   ImportError: cannot import name 'CommandResult' from 'mjlab_microduck.next_rl.runner'
   ```

3. Before the remote wrapper and checkpoint/cancellation methods existed:

   ```text
   FileNotFoundError: scripts/next_rl_remote_job.py
   AttributeError: 'NitroRunner' object has no attribute 'cancel'
   AttributeError: 'NitroRunner' object has no attribute 'sync_checkpoint'
   ```

4. The security mutation pass failed before it rejected additional
   secret-like/control-character paths and forged remote checkpoint parents:

   ```text
   FAILED test_prepare_rejects_secret_like_tracked_paths[.netrc]
   FAILED test_prepare_rejects_secret_like_tracked_paths[config/password.txt]
   FAILED test_prepare_rejects_control_characters_in_tracked_paths
   FAILED test_checkpoint_transfer_rejects_remote_shell_syntax_in_parent_path
   4 failed, 31 passed
   ```

5. A pinned-tree regression proved that the initially separate wrapper copy
   still raced the worktree:

   ```text
   FAILED test_prepare_wrapper_comes_from_the_same_pinned_tree
   AssertionError: '# mutable worktree wrapper\n' != '# wrapper\n'
   ```

6. Failure-durability and checkpoint hashing regressions failed before setup
   errors became durable and hashing was followed by a confirming stat:

   ```text
   FAILED test_checkpoint_changed_while_hashing_is_not_declared_stable
   AttributeError: module 'next_rl_remote_job' has no attribute 'run_supervisor'
   TypeError: _start_supervisor() got an unexpected keyword argument 'popen'
   3 failed, 9 passed
   ```

## GREEN evidence

- Focused runner/wrapper suite:

  ```text
  uv run --with pytest pytest tests/test_next_rl_runner.py tests/test_next_rl_remote_job.py -q
  48 passed in 3.97s
  ```

- Complete Next RL suite:

  ```text
  uv run --with pytest pytest tests/test_next_rl_*.py -q
  157 passed in 7.49s
  ```

- Full repository suite:

  ```text
  uv run --with pytest pytest tests/ -q
  353 passed, 1 skipped in 24.66s
  ```

- Syntax compilation:

  ```text
  python3 -m compileall -q src/mjlab_microduck/next_rl/runner.py scripts/next_rl_remote_job.py
  exit 0
  ```

## Commit

`feat(next-rl): add resumable Nitro training runner`

## Concerns

- Ruff is unavailable in the current environment (`Failed to spawn: ruff`),
  so verification used syntax compilation plus focused, Next RL, and full tests.
- Live OpenSSH/Nitro behavior and real training were intentionally not exercised;
  transport and lifecycle behavior were verified with injected adapters and
  temporary local fixtures, as required.

## Fix round 1 — lifecycle ownership and real resume staging

### RED evidence

The first review regression slice failed on every missing lifecycle boundary:

```text
FAILED test_job_directory_rejects_symlink_even_when_target_stays_under_root
AttributeError: module 'next_rl_remote_job' has no attribute 'verified_train_argv'
TypeError: cancel_job() got an unexpected keyword argument 'wait_for_exit'
TypeError: inspect_or_control() got an unexpected keyword argument 'supervisor_popen'
AttributeError: module 'next_rl_remote_job' has no attribute 'finalize_training_state'
9 failed, 11 passed
```

The runner-facing slice then proved command binding, credential filtering,
atomic remote creation, transport retry, and cross-fingerprint resume staging
were absent:

```text
KeyError: 'command'
FAILED test_prepare_rejects_secret_like_tracked_paths[.git-credentials]
FAILED test_prepared_manifest_recursively_removes_credential_like_keys
FAILED test_prepare_atomically_creates_remote_job_before_any_transfer
ConnectionError: scp disconnected
FileNotFoundError: resume-checkpoint.pt
13 failed, 57 passed
```

Finally, focused edge mutations failed before failed spawn claims rolled back,
claimed launches became idempotent, cleanup stopped relying on `poll()`, and
cancellation merged late checkpoint evidence:

```text
FAILED test_prepare_rejects_local_fingerprint_directory_symlink
FAILED test_failed_detached_spawn_keeps_start_request_retryable
FAILED test_claimed_start_is_idempotently_left_for_spawned_supervisor
FAILED test_cleanup_terminates_and_reaps_without_relying_on_poll
FAILED test_prepared_manifest_recursively_removes_credential_like_keys
FAILED test_cancel_merges_checkpoint_evidence_written_while_waiting
```

### GREEN evidence

```text
uv run --with pytest pytest tests/test_next_rl_runner.py tests/test_next_rl_remote_job.py -q
76 passed in 3.11s

uv run --with pytest pytest tests/test_next_rl_*.py -q
185 passed in 5.45s

uv run --with pytest pytest tests/ -q
381 passed, 1 skipped in 13.09s
```

### Fixes

- Start requests now carry a deterministic ID and are claimed under an
  `fcntl.flock` lifecycle lock before spawning. Concurrent and repeated starts
  are idempotent; failed spawns restore pending state, while a recorded spawned
  supervisor survives a wrapper crash without a duplicate launch.
- Every post-spawn supervisor failure invokes exact new-session process-group
  termination and bounded reap before durable failure state. Cancellation
  validates PID/start/command identity, waits, escalates to `SIGKILL` only while
  identity still matches, confirms exit, and merges checkpoint evidence so the
  supervisor cannot erase the cancelled terminal record.
- Resume preparation requires a distinct source fingerprint plus the expected
  checkpoint digest, accepts only the named stable checkpoint, retries and
  verifies SCP, records the complete resume binding, and transfers a staged
  checkpoint. The remote wrapper verifies it again and hard-links it into the
  original `source/logs/<experiment>/<run>/model_N.pt` path before training.
- Remote job creation uses atomic non-following `mkdir` before any SCP, and both
  local and remote fingerprint directories reject symlinks. The wrapper also
  compares `train-argv.json` to its immutable prepared-manifest digest before
  spawning.
- Tracked paths and serialized learning inputs now detect credential words and
  compound `api-key`/`private-key`/token variants recursively, including
  `.git-credentials` and `wandb_api_key`, while retaining `tokenizer`,
  `token_budget`, `passwordless`, and `monkey` names.

### Commit

`fix(next-rl): harden Nitro resume lifecycle`

### Concerns

- Ruff remains unavailable. Syntax compilation and focused, Next RL, and full
  tests are the available verification evidence.
- No live SSH/network operation, Nitro start/cancel, training, upload, or
  publishing was performed.

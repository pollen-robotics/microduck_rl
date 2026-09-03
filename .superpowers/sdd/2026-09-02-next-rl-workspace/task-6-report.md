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

## Fix round 3 — fail-closed preparation recovery

### RED evidence

The benign-token slice failed while `token_budget` and `token_count` were still
treated as credentials:

```text
FAILED test_secret_path_filter_does_not_reject_unrelated_words
FAILED test_prepared_manifest_recursively_removes_credential_like_keys
2 failed, 55 deselected
```

The preparation safety slice then showed the old recovery deleting the entire
fingerprint path and lacking an explicit read-only inspection mode:

```text
FAILED test_prepare_cleans_only_incomplete_fingerprint_and_retries_transfer[before_copy]
FAILED test_prepare_cleans_only_incomplete_fingerprint_and_retries_transfer[partial_copy]
FAILED test_prepare_treats_non_directory_fingerprint_path_as_conflict[file]
FAILED test_prepare_treats_non_directory_fingerprint_path_as_conflict[fifo]
FAILED test_prepare_treats_non_directory_fingerprint_path_as_conflict[socket]
FAILED test_transient_read_only_inspection_disconnect_never_authorizes_cleanup
FAILED test_ambiguous_existing_preparation_fails_closed_without_cleanup[malformed_json]
FAILED test_ambiguous_existing_preparation_fails_closed_without_cleanup[nonzero]
FAILED test_lost_finalize_response_reinspects_complete_digest_without_cleanup
FAILED test_read_only_inspection_ignores_pending_start_and_cancel_requests
10 failed, 3 passed, 84 deselected
```

Affirmative incomplete-state and completed-bundle integrity tests independently
failed before those states were cryptographically distinguished:

```text
FAILED test_prepare_reuses_real_directory_but_resets_only_owned_matching_stage
FAILED test_completed_preparation_rejects_forged_marker_and_bundle_digest
```

### GREEN evidence

```text
uv run --with pytest pytest tests/test_next_rl_runner.py tests/test_next_rl_remote_job.py -q
102 passed in 5.35s

uv run --with pytest pytest tests/test_next_rl_*.py -q
211 passed in 6.65s

uv run --with pytest pytest -q
407 passed, 1 skipped in 13.29s

python3 -m compileall -q src/mjlab_microduck/next_rl/runner.py scripts/next_rl_remote_job.py
exit 0

git diff --check
exit 0
```

### Fixes

- Existing preparation inspection now uses the explicit
  `NEXT_RL_REMOTE_INSPECT=1` wrapper action. It holds the lifecycle lock only to
  read state, ignores pending start/cancel requests, retries transient reads,
  and treats transport, nonzero, or parse ambiguity as a non-mutating error.
- Fingerprint paths must affirmatively be directories and non-symlinks;
  regular files, FIFOs, sockets, and symlinks are conflicts. Recovery never
  deletes a fingerprint directory or its logs, sources, or checkpoints.
- Only the exact digest-named staging child may be reset, after affirmative
  non-symlink, directory, and SSH-user ownership checks. Lost finalize
  responses are read-only inspected for the exact completed digest before any
  stage cleanup. A fully moved but unmarked bundle is considered incomplete
  only after all bundle/source/command digests validate.
- Completed marker and bundle digests are recomputed before a job is accepted.
  Forged matching strings therefore fail closed.
- `token` remains sensitive when standalone or combined with vendor/context or
  credential suffix tokens, while exact benign `token_budget` and
  `token_count` names plus `tokenizer`/`tokenization` remain allowed.

### Commit

`fix(next-rl): make preparation recovery fail closed`

### Concerns

- Ruff remains unavailable; syntax compilation, whitespace validation,
  focused tests, the complete Next RL suite, and the full repository suite
  passed.
- No live SSH/network operation, Nitro start/cancel, training, upload, or
  publishing was performed.

## Fix round 2 — crash convergence and atomic preparation

### RED evidence

The detached-launch regression slice reproduced both crash windows, duplicate
trainer launch, and the command reload race:

```text
FAILED test_training_launch_never_uses_a_shell_and_creates_a_process_group
FAILED test_launch_uses_verified_tuple_without_rereading_argv_file
FAILED test_claimed_start_is_idempotently_left_for_spawned_supervisor
FAILED test_crash_after_claim_before_popen_self_heals_on_retry
FAILED test_crash_after_popen_before_spawn_record_converges_via_duplicate_supervisors
FAILED test_duplicate_supervisors_use_singleton_lock_to_launch_one_trainer
6 failed, 24 passed
```

The credential-token mutation slice then proved that token indicators embedded
inside longer, delimiter-separated names were not rejected or scrubbed:

```text
FAILED test_prepare_rejects_secret_like_tracked_paths[config/wandb_token_value.txt]
FAILED test_prepare_rejects_secret_like_tracked_paths[keys/github_token_backup]
FAILED test_prepare_rejects_secret_like_tracked_paths[auth/oauth2_token_file.json]
FAILED test_prepared_manifest_recursively_removes_credential_like_keys
4 failed, 9 passed, 40 deselected
```

Finally, interrupted preparation and existing-directory tests failed before a
staged, retryable protocol existed:

```text
FAILED test_prepare_cleans_only_incomplete_fingerprint_and_retries_transfer[before_copy]
FAILED test_prepare_cleans_only_incomplete_fingerprint_and_retries_transfer[partial_copy]
FAILED test_prepare_reuses_only_matching_complete_job
FAILED test_prepare_rejects_conflicting_complete_job
FAILED test_prepare_rejects_remote_job_directory_symlink
5 failed, 52 deselected
```

### GREEN evidence

```text
uv run --with pytest pytest tests/test_next_rl_runner.py tests/test_next_rl_remote_job.py -q
89 passed in 4.56s

uv run --with pytest pytest tests/test_next_rl_*.py -q
198 passed in 6.30s

uv run --with pytest pytest -q
394 passed, 1 skipped in 14.68s

python3 -m compileall -q src/mjlab_microduck/next_rl/runner.py scripts/next_rl_remote_job.py
exit 0

git diff --check
exit 0
```

### Fixes

- Detached supervisors now hold a separate per-job `fcntl.flock` singleton for
  launch and monitoring. A stale claimed request spawns a replacement
  supervisor; a crash after `Popen` can spawn a duplicate supervisor, but the
  singleton permits exactly one trainer and retries converge on its terminal
  state.
- The supervisor loads and digest-validates training argv once and passes that
  exact immutable tuple to `Popen`; the launch boundary never rereads
  `train-argv.json`.
- Secret detection tokenizes path components and nested manifest keys on
  non-alphanumerics and now treats `token` anywhere as sensitive. This round
  intentionally supersedes round 1's `token_budget` allowance; unrelated whole
  words such as `tokenizer`, `tokenization`, `passwordless`, and `monkey` remain
  accepted.
- Preparation writes a deterministic bundle manifest, transfers only into a
  fingerprint-local `.incoming-<bundle-digest>` directory, and asks the pinned
  wrapper to validate every size/digest plus the prepared source and command
  manifests before atomically writing `.complete.json`. Interrupted attempts
  inspect for a matching completed bundle, reject conflicting completed jobs
  and symlinks, and clean/retry only the exact incomplete fingerprint path.

### Commit

`fix(next-rl): make Nitro launch and prepare crash-safe`

### Concerns

- Ruff remains unavailable (`Failed to spawn: ruff`); syntax compilation,
  whitespace validation, focused tests, the complete Next RL suite, and the
  full repository suite passed.
- No live SSH/network operation, Nitro start/cancel, training, upload, or
  publishing was performed.

## Fix round 4 — normalized transport exit handling

### RED evidence

Production-shaped tests exposed that `OpenSSHAdapter` raised before expected
nonzero probe results could reach the recovery state machine:

```text
FAILED test_open_ssh_adapter_returns_nonzero_result_for_expected_probe
FAILED test_real_open_ssh_adapter_nonzero_probe_allows_staging_only_retry
2 failed, 67 deselected
```

The integration-shaped failure occurred specifically when the real adapter
converted the expected `test ! -e` exit status 1 into `RunnerError`, while the
same path passed under the permissive fake adapter.

### GREEN evidence

```text
uv run --with pytest pytest tests/test_next_rl_runner.py tests/test_next_rl_remote_job.py -q
104 passed in 6.06s

uv run --with pytest pytest tests/test_next_rl_*.py -q
213 passed in 7.52s

uv run --with pytest pytest -q
409 passed, 1 skipped in 15.05s
```

### Fixes

- `CommandAdapter.run` and `OpenSSHAdapter.run` now consistently return
  `CommandResult` for both zero and nonzero process exits. OS/process-launch
  exceptions still propagate as transport failures.
- `_run` remains the explicit require-success boundary. Every direct adapter
  call was audited: directory, symlink, existence, ownership, and atomic-mkdir
  probes interpret expected nonzero results explicitly and fail closed for
  unexpected or ambiguous states.
- A subprocess-backed preparation test exercises the actual adapter behavior,
  including a nonzero partial SCP, a nonzero `test ! -e` probe, exact staging
  cleanup, and successful retry without any full fingerprint deletion.

### Commit

`fix(next-rl): normalize transport exit handling`

### Concerns

- Ruff remains unavailable; focused, Next RL, and full tests passed.
- No live SSH/network operation, Nitro start/cancel, training, upload, or
  publishing was performed.

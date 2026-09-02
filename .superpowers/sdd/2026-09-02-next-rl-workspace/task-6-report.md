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

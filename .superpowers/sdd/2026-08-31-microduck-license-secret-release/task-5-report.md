# Task 5 Report: Production Container Secret Mount and Operator Workflow

## Status

Complete. The production image accepts the ROM bearer only from an owner-only
read-only bind mount at `/run/secrets/microduck_rom_bearer_token`. Direct
environment input, a missing mount, and a wrong configured path exit 64 in the
entrypoint; Python remains authoritative for content and mode validation.

## Commit

Pending at report creation; committed as
`feat: mount ROM bearer as a container secret` after the final gates.

## Files

- `.dockerignore`
- `README.md`
- `docker/rom-simulator/Dockerfile`
- `docker/rom-simulator/Dockerfile.dockerignore`
- `docker/rom-simulator/entrypoint.sh`
- `docs/rom-simulator.md`
- `tests/test_rom_process_container.py`
- `tests/test_rom_qualification.py`

The two Docker context policies and Dockerfile inventory now include Task 4's
`secret_file.py`; without that literal addition the final image could not import
the secure reader. No other image input was broadened.

## RED evidence

- Direct environment rejection:
  `test_container_entrypoint_rejects_direct_bearer_before_server` failed because
  the old entrypoint reported the missing bundle instead of rejecting direct
  bearer input (`1 failed`).
- Literal image inventory:
  `test_dockerfile_copies_only_literal_host_files` failed with
  `src/mjlab_microduck/rom/secret_file.py` missing from the Docker COPY set
  (`1 failed`).
- Mounted-secret startup against the pre-fix image:
  `test_production_container_mounted_secret_authenticates_without_leaking_metadata`
  failed after 35.27 seconds because the old entrypoint still required
  `MICRODUCK_ROM_BEARER_TOKEN` (`1 failed`).

Each failure was the expected missing production behavior, not a fixture or
syntax error.

## GREEN evidence

- Exact requested suite:
  `uv run pytest tests/test_rom_process_container.py tests/test_rom_qualification.py -q`
  -> `155 passed, 14 skipped in 505.54s`.
- Fresh final-image secret contract (image
  `microduck-rom-sim:task5-green`, freshly qualified bundle): six focused cases
  passed in 38.87 seconds. These cover direct/missing/wrong-path rejection,
  permissive-mode rejection before API startup, owner/group 10001 mode-0400
  authenticated startup, and final-image inventory.
- `uv run ruff check tests/test_rom_process_container.py tests/test_rom_qualification.py`
  -> `All checks passed!`.
- The required leakage expression returned no matches across `README.md`, the
  operator runbook, Docker files, and container tests.
- `git diff --check` returned no output.

## Leak-check evidence

The live-image test inspected image history, image configuration, container
`Config.Env`, `Config.Cmd`, `Path`, `Args`, `docker top` process arguments, and
combined stdout/stderr logs. The known test bearer literal was absent from every
channel. Failure helpers raise only a stable channel label and never echo the
bearer or captured evidence. The accepted container exposed only the fixed
non-secret file-path environment setting and a read-only bind mount.

Secret fixtures are created through the built image as root, with the bearer on
stdin rather than argv/environment metadata, then set to UID/GID 10001 and mode
0400. This does not assume the pytest host user can `chown`.

## Self-review

- Preserved non-root UID/GID 10001, read-only root, dropped capabilities,
  no-new-privileges, noexec/nosuid tmpfs, bundle/state mount modes, SIGTERM,
  PID-1 `exec`, and the 60-second stop timeout.
- Kept the shell entrypoint from reading the secret bytes; it validates only the
  fixed path and basic file properties before Python performs the bounded,
  no-follow, owner-only, UTF-8/content checks.
- Kept Docker context allowlists literal and verified unknown ROM modules remain
  excluded.
- Replaced operator environment-file examples with protected-directory creation,
  `install -m 0400`, a read-only fixed-path mount, container-replacement rotation,
  and explicit credential removal after shutdown.

## Concerns

The exact full-suite command intentionally skips 14 live Docker lifecycle tests
unless its two release-fixture environment variables are set. Task 5's six
image-dependent contract cases were therefore also run separately against the
freshly built image and a freshly qualified current-contract bundle; all passed.

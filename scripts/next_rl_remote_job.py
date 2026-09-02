#!/usr/bin/env python3
"""Standalone, fixed-interface supervisor for one prepared Next RL job."""

from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import BinaryIO, NamedTuple

REMOTE_ROOT = Path("/home/aif_eng/microduck-training/runs")
_FINGERPRINT = re.compile(r"^[0-9a-f]{6,64}$")
_TASK_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]*$")
_CHECKPOINT = re.compile(r"^model_([0-9]+)\.pt$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SUPERVISOR_ENV = "NEXT_RL_REMOTE_SUPERVISOR"
_POLL_SECONDS = 2.0
_CHECKPOINT_PROBE_SECONDS = 1.0


class RemoteJobError(RuntimeError):
    """Prepared remote job input or lifecycle state is unsafe."""


class ProcessIdentity(NamedTuple):
    pid: int
    process_start: str
    command_digest: str

    def as_dict(self) -> dict[str, object]:
        return {
            "pid": self.pid,
            "process_start": self.process_start,
            "command_digest": self.command_digest,
        }


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def atomic_write_json(path: str | Path, value: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json(value).encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _load_object(path: Path, name: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RemoteJobError(f"{name} is missing or invalid JSON") from error
    if not isinstance(value, dict):
        raise RemoteJobError(f"{name} must be a JSON object")
    return value


def validate_job_directory(
    value: str | Path,
    *,
    root: str | Path = REMOTE_ROOT,
) -> Path:
    """Resolve one immediate fingerprint directory beneath the fixed root."""
    root_path = Path(root).resolve()
    candidate = Path(value).resolve()
    try:
        relative = candidate.relative_to(root_path)
    except ValueError as error:
        raise RemoteJobError("job directory must resolve beneath the configured root") from error
    if len(relative.parts) != 1:
        raise RemoteJobError("job directory must be one fingerprint beneath the configured root")
    if not _FINGERPRINT.fullmatch(relative.name):
        raise RemoteJobError("job directory name must be a lowercase hexadecimal fingerprint")
    if not candidate.is_dir():
        raise RemoteJobError("job directory does not exist")
    return candidate


def load_train_argv(job_directory: str | Path) -> tuple[str, ...]:
    path = Path(job_directory) / "train-argv.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RemoteJobError("train argv is missing or invalid JSON") from error
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise RemoteJobError("train argv must be a JSON array")
    if not value or not all(isinstance(part, str) and part for part in value):
        raise RemoteJobError("train argv must be a non-empty JSON array of strings")
    argv = tuple(value)
    if len(argv) < 4 or argv[:3] != ("uv", "run", "train"):
        raise RemoteJobError("train argv must invoke the fixed uv run train interface")
    if ".." in argv[3] or not _TASK_ID.fullmatch(argv[3]):
        raise RemoteJobError("train argv task ID is invalid")
    if any(any(ord(character) < 32 for character in part) for part in argv):
        raise RemoteJobError("train argv contains control characters")
    return argv


def command_digest(argv: Sequence[str]) -> str:
    return hashlib.sha256(canonical_json(list(argv)).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def launch_training(
    job_directory: str | Path,
    *,
    cwd: str | Path,
    stdout: BinaryIO,
    popen: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
) -> subprocess.Popen[bytes]:
    """Launch the validated trainer without a shell in a new process group."""
    return popen(
        load_train_argv(job_directory),
        cwd=Path(cwd),
        stdin=subprocess.DEVNULL,
        stdout=stdout,
        stderr=subprocess.STDOUT,
        shell=False,
        start_new_session=True,
    )


def select_latest_checkpoint(paths: Iterable[Path]) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for path in paths:
        match = _CHECKPOINT.fullmatch(path.name)
        if match and path.is_file() and not path.is_symlink():
            candidates.append((int(match.group(1)), path))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


class StableCheckpointTracker:
    """Recognize a checkpoint only after two identical size/mtime probes."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root).resolve() if root is not None else None
        self._previous: tuple[Path, int, int] | None = None

    def observe(self, paths: Iterable[Path]) -> dict[str, object] | None:
        latest = select_latest_checkpoint(paths)
        if latest is None:
            self._previous = None
            return None
        resolved = latest.resolve()
        stat = resolved.stat()
        current = (resolved, stat.st_size, stat.st_mtime_ns)
        if current != self._previous:
            self._previous = current
            return None
        if self.root is None:
            relative = Path(resolved.name)
        else:
            try:
                relative = resolved.relative_to(self.root)
            except ValueError as error:
                raise RemoteJobError("checkpoint resolved outside the job directory") from error
        digest = sha256_file(resolved)
        verified = resolved.stat()
        if (verified.st_size, verified.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
            self._previous = (resolved, verified.st_size, verified.st_mtime_ns)
            return None
        return {
            "name": resolved.name,
            "relative_path": relative.as_posix(),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": digest,
        }


def _proc_fields(pid: int) -> tuple[str, tuple[str, ...]]:
    try:
        raw_stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        command_bytes = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError as error:
        raise RemoteJobError("process identity is unavailable") from error
    closing = raw_stat.rfind(")")
    fields = raw_stat[closing + 2 :].split()
    if closing < 0 or len(fields) <= 19:
        raise RemoteJobError("process start identity is malformed")
    try:
        argv = tuple(
            part.decode("utf-8") for part in command_bytes.rstrip(b"\0").split(b"\0")
        )
    except UnicodeDecodeError as error:
        raise RemoteJobError("live command identity is not UTF-8") from error
    if not argv:
        raise RemoteJobError("live command identity is empty")
    return fields[19], argv


def live_process_identity(pid: int) -> ProcessIdentity:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise RemoteJobError("recorded PID is invalid")
    start, argv = _proc_fields(pid)
    return ProcessIdentity(pid, start, command_digest(argv))


def _identity_from(value: Mapping[str, object]) -> ProcessIdentity:
    pid = value.get("pid")
    start = value.get("process_start")
    digest = value.get("command_digest")
    if (
        isinstance(pid, bool)
        or not isinstance(pid, int)
        or pid <= 0
        or not isinstance(start, str)
        or not start
        or not isinstance(digest, str)
        or not _SHA256.fullmatch(digest)
    ):
        raise RemoteJobError("cancellation process identity is invalid")
    return ProcessIdentity(pid, start, digest)


def cancel_job(
    job_directory: str | Path,
    *,
    killpg: Callable[[int, int], None] = os.killpg,
) -> dict[str, object]:
    """Signal only the process group whose complete live identity still matches."""
    job = Path(job_directory)
    state = _load_object(job / "status.json", "status")
    request = _load_object(job / "cancel-request.json", "cancel request")
    if request.get("action") != "cancel" or state.get("status") != "running":
        raise RemoteJobError("cancellation requires a running job and cancel request")
    expected = _identity_from(request)
    if _identity_from(state) != expected or live_process_identity(expected.pid) != expected:
        raise RemoteJobError("live process identity does not match the cancellation request")
    if os.getpgid(expected.pid) != expected.pid:
        raise RemoteJobError("trainer is not the leader of its recorded process group")
    killpg(expected.pid, signal.SIGTERM)
    result = {
        **state,
        "status": "failed",
        "cancelled": True,
        "exit_code": None,
    }
    atomic_write_json(job / "status.json", result)
    try:
        (job / "cancel-request.json").unlink()
    except FileNotFoundError:
        pass
    return result


def _verified_source(job: Path) -> Path:
    manifest = _load_object(job / "prepared-manifest.json", "prepared manifest")
    source = manifest.get("source")
    if not isinstance(source, Mapping):
        raise RemoteJobError("prepared manifest has no source record")
    expected = source.get("archive_sha256")
    archive_path = job / "source.tar"
    if not isinstance(expected, str) or not _SHA256.fullmatch(expected):
        raise RemoteJobError("prepared source digest is invalid")
    if sha256_file(archive_path) != expected:
        raise RemoteJobError("prepared source archive digest mismatch")

    destination = job / "source"
    marker = destination / ".next-rl-archive-sha256"
    if destination.exists():
        if destination.is_dir() and not destination.is_symlink() and marker.read_text().strip() == expected:
            return destination
        raise RemoteJobError("existing source directory does not match the prepared archive")
    temporary = Path(tempfile.mkdtemp(prefix=".source.", dir=job))
    try:
        with tarfile.open(archive_path, "r") as archive:
            for member in archive.getmembers():
                member_path = Path(member.name)
                if (
                    not member.isfile()
                    or member_path.is_absolute()
                    or any(part in ("", ".", "..") for part in member_path.parts)
                ):
                    raise RemoteJobError("source archive contains an unsafe entry")
                target = temporary.joinpath(*member_path.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise RemoteJobError("source archive regular file has no payload")
                with target.open("wb") as output:
                    while block := extracted.read(1024 * 1024):
                        output.write(block)
                target.chmod(member.mode & 0o755)
        marker_path = temporary / marker.name
        marker_path.write_text(expected, encoding="ascii")
        os.replace(temporary, destination)
    except BaseException:
        # The temporary directory is fingerprint-local and contains only archive output.
        import shutil

        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def _checkpoint_paths(source: Path) -> Iterable[Path]:
    return source.glob("logs/**/model_*.pt")


def supervise(job_directory: str | Path) -> dict[str, object]:
    """Run and monitor one trainer, atomically recording every lifecycle state."""
    job = Path(job_directory)
    source = _verified_source(job)
    argv = load_train_argv(job)
    digest = command_digest(argv)
    stdout_path = job / "stdout.log"
    tracker = StableCheckpointTracker(job)
    last_checkpoint: dict[str, object] | None = None
    with stdout_path.open("ab", buffering=0) as output:
        process = launch_training(job, cwd=source, stdout=output)
        identity = live_process_identity(process.pid)
        if identity.command_digest != digest:
            process.terminate()
            raise RemoteJobError("launched trainer command identity mismatch")
        state: dict[str, object] = {
            "artifact_status": "pending",
            **identity.as_dict(),
            "exit_code": None,
            "last_stable_checkpoint": None,
            "status": "running",
            "stdout_path": str(stdout_path),
        }
        atomic_write_json(job / "status.json", state)
        while process.poll() is None:
            stable = tracker.observe(_checkpoint_paths(source))
            if stable is not None and stable != last_checkpoint:
                last_checkpoint = stable
                state = {
                    **state,
                    "artifact_status": "stable_checkpoint",
                    "last_stable_checkpoint": stable,
                }
                atomic_write_json(job / "status.json", state)
            time.sleep(_POLL_SECONDS)
        tracker.observe(_checkpoint_paths(source))
        time.sleep(_CHECKPOINT_PROBE_SECONDS)
        stable = tracker.observe(_checkpoint_paths(source))
        if stable is not None:
            last_checkpoint = stable
        exit_code = process.returncode
    final = {
        **state,
        "artifact_status": "stable_checkpoint" if last_checkpoint else "missing",
        "exit_code": exit_code,
        "last_stable_checkpoint": last_checkpoint,
        "status": "succeeded" if exit_code == 0 else "failed",
    }
    atomic_write_json(job / "status.json", final)
    return final


def run_supervisor(job: Path) -> dict[str, object]:
    """Run the supervisor and make pre-launch failures durable."""
    try:
        return supervise(job)
    except Exception as error:
        state = _load_object(job / "status.json", "status")
        failed = {
            **state,
            "error": str(error),
            "exit_code": 1,
            "status": "failed",
        }
        atomic_write_json(job / "status.json", failed)
        raise


def _start_supervisor(
    job: Path,
    *,
    popen: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
) -> dict[str, object]:
    request = _load_object(job / "start-request.json", "start request")
    state = _load_object(job / "status.json", "status")
    if request != {"action": "start"} or state.get("status") != "pending":
        raise RemoteJobError("start requires an exact request and pending state")
    environment = os.environ.copy()
    environment[_SUPERVISOR_ENV] = "1"
    popen(
        (sys.executable, str(Path(__file__).resolve()), str(job)),
        cwd=job,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        shell=False,
        start_new_session=True,
        close_fds=True,
        env=environment,
    )
    try:
        (job / "start-request.json").unlink()
    except FileNotFoundError:
        raise RemoteJobError("start request disappeared") from None
    return state


def inspect_or_control(job: Path) -> dict[str, object]:
    if (job / "cancel-request.json").exists():
        return cancel_job(job)
    if (job / "start-request.json").exists():
        return _start_supervisor(job)
    state = _load_object(job / "status.json", "status")
    if state.get("status") == "running":
        try:
            live = live_process_identity(_identity_from(state).pid)
        except RemoteJobError:
            live = None
        state = {**state, "live_identity": live.as_dict() if live else None}
    return state


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 1:
        raise SystemExit("usage: next_rl_remote_job.py JOB_DIRECTORY")
    job = validate_job_directory(arguments[0])
    if os.environ.get(_SUPERVISOR_ENV) == "1":
        run_supervisor(job)
        return 0
    print(canonical_json(inspect_or_control(job)), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RemoteJobError as error:
        print(f"next_rl_remote_job: {error}", file=sys.stderr)
        raise SystemExit(2) from error

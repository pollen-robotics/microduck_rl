#!/usr/bin/env python3
"""Standalone, fixed-interface supervisor for one prepared Next RL job."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, NamedTuple

REMOTE_ROOT = Path("/home/aif_eng/microduck-training/runs")
_FINGERPRINT = re.compile(r"^[0-9a-f]{6,64}$")
_TASK_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]*$")
_CHECKPOINT = re.compile(r"^model_([0-9]+)\.pt$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SUPERVISOR_ENV = "NEXT_RL_REMOTE_SUPERVISOR"
_INSPECT_ENV = "NEXT_RL_REMOTE_INSPECT"
_POLL_SECONDS = 2.0
_CHECKPOINT_PROBE_SECONDS = 1.0
_TERMINATE_TIMEOUT_SECONDS = 5.0
_BUNDLE_REQUIRED_FILES = frozenset(
    {
        "next_rl_remote_job.py",
        "prepared-manifest.json",
        "source.tar",
        "status.json",
        "train-argv.json",
    }
)
_BUNDLE_OPTIONAL_FILES = frozenset({"resume-checkpoint.pt"})


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
    unresolved = Path(value)
    if unresolved.is_symlink():
        raise RemoteJobError("job directory itself must not be a symlink")
    candidate = unresolved.resolve()
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


@contextmanager
def lifecycle_lock(job_directory: str | Path):
    """Serialize lifecycle decisions using a non-symlink advisory lockfile."""
    path = Path(job_directory) / ".lifecycle.lock"
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as error:
        raise RemoteJobError("cannot open lifecycle lock safely") from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise RemoteJobError("lifecycle lock must be a regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


@contextmanager
def supervisor_lock(job_directory: str | Path):
    """Allow only one detached supervisor to own trainer launch and monitoring."""
    path = Path(job_directory) / ".supervisor.lock"
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as error:
        raise RemoteJobError("cannot open supervisor lock safely") from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise RemoteJobError("supervisor lock must be a regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


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


def verified_train_argv(job_directory: str | Path) -> tuple[str, ...]:
    """Load argv only when it matches the immutable prepared command digest."""
    job = Path(job_directory)
    manifest = _load_object(job / "prepared-manifest.json", "prepared manifest")
    command = manifest.get("command")
    if not isinstance(command, Mapping):
        raise RemoteJobError("prepared manifest has no command digest")
    if command.get("argv_file") != "train-argv.json":
        raise RemoteJobError("prepared command argv file is invalid")
    expected = command.get("sha256")
    if not isinstance(expected, str) or not _SHA256.fullmatch(expected):
        raise RemoteJobError("prepared command digest is invalid")
    argv = load_train_argv(job)
    if command_digest(argv) != expected:
        raise RemoteJobError("train argv digest does not match prepared manifest")
    return argv


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _validated_bundle(stage: Path, job: Path) -> tuple[str, tuple[str, ...]]:
    if stage.is_symlink() or not stage.is_dir() or stage.parent != job:
        raise RemoteJobError("bundle staging directory is unsafe")
    bundle = _load_object(stage / "bundle-manifest.json", "bundle manifest")
    if set(bundle) != {"bundle_sha256", "files", "fingerprint"}:
        raise RemoteJobError("bundle manifest fields are invalid")
    digest = bundle.get("bundle_sha256")
    fingerprint = bundle.get("fingerprint")
    files = bundle.get("files")
    if (
        not isinstance(digest, str)
        or not _SHA256.fullmatch(digest)
        or fingerprint != job.name
        or not isinstance(files, Mapping)
    ):
        raise RemoteJobError("bundle manifest identity is invalid")
    unsigned = {"files": files, "fingerprint": fingerprint}
    if hashlib.sha256(canonical_json(unsigned).encode("utf-8")).hexdigest() != digest:
        raise RemoteJobError("bundle manifest digest mismatch")
    if stage.name != f".incoming-{digest}":
        raise RemoteJobError("bundle staging path does not match its digest")
    names = set(files)
    if not _BUNDLE_REQUIRED_FILES.issubset(names) or not names.issubset(
        _BUNDLE_REQUIRED_FILES | _BUNDLE_OPTIONAL_FILES
    ):
        raise RemoteJobError("bundle file set is invalid")
    actual_names = {path.name for path in stage.iterdir()}
    if actual_names != names | {"bundle-manifest.json"}:
        raise RemoteJobError("bundle staging directory contains unexpected files")
    for name in sorted(names):
        record = files.get(name)
        path = stage / name
        if (
            not isinstance(record, Mapping)
            or set(record) != {"sha256", "size"}
            or not isinstance(record.get("sha256"), str)
            or not _SHA256.fullmatch(str(record.get("sha256")))
            or isinstance(record.get("size"), bool)
            or not isinstance(record.get("size"), int)
            or int(record.get("size")) < 0
            or not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != record.get("size")
            or sha256_file(path) != record.get("sha256")
        ):
            raise RemoteJobError(f"bundle file digest or size mismatch: {name}")

    prepared = _load_object(stage / "prepared-manifest.json", "prepared manifest")
    if prepared.get("fingerprint") != job.name:
        raise RemoteJobError("prepared manifest fingerprint mismatch")
    source = prepared.get("source")
    if not isinstance(source, Mapping) or source.get("archive_sha256") != sha256_file(
        stage / "source.tar"
    ):
        raise RemoteJobError("prepared source archive digest mismatch")
    command = prepared.get("command")
    argv = load_train_argv(stage)
    if (
        not isinstance(command, Mapping)
        or command.get("argv_file") != "train-argv.json"
        or command.get("sha256") != command_digest(argv)
    ):
        raise RemoteJobError("prepared command digest mismatch")
    status = _load_object(stage / "status.json", "status")
    if status.get("status") != "pending":
        raise RemoteJobError("prepared status must be pending")
    return digest, tuple(sorted(names))


def finalize_preparation(
    job_directory: str | Path,
    staging_directory: str | Path,
) -> dict[str, object]:
    """Digest-verify staged content and publish a completion marker last."""
    job = Path(job_directory)
    stage = Path(staging_directory)
    if job.is_symlink() or not job.is_dir():
        raise RemoteJobError("job directory itself must not be a symlink")
    digest, names = _validated_bundle(stage, job)
    for name in (*names, "bundle-manifest.json"):
        source = stage / name
        target = job / name
        if target.exists():
            if target.is_symlink() or not target.is_file() or sha256_file(target) != sha256_file(
                source
            ):
                raise RemoteJobError(f"existing prepared file conflicts: {name}")
            source.unlink()
        else:
            os.replace(source, target)
    stage.rmdir()
    marker = {"bundle_sha256": digest, "fingerprint": job.name}
    atomic_write_json(job / ".complete.json", marker)
    state = _load_object(job / "status.json", "status")
    return {**state, "prepared_bundle_sha256": digest}


def completed_preparation_state(job_directory: str | Path) -> dict[str, object]:
    job = Path(job_directory)
    marker = _load_object(job / ".complete.json", "completion marker")
    bundle = _load_object(job / "bundle-manifest.json", "bundle manifest")
    digest = marker.get("bundle_sha256")
    files = bundle.get("files")
    if (
        set(marker) != {"bundle_sha256", "fingerprint"}
        or set(bundle) != {"bundle_sha256", "files", "fingerprint"}
        or marker.get("fingerprint") != job.name
        or bundle.get("fingerprint") != job.name
        or bundle.get("bundle_sha256") != digest
        or not isinstance(digest, str)
        or not _SHA256.fullmatch(digest)
        or not isinstance(files, Mapping)
    ):
        raise RemoteJobError("completed bundle identity is invalid")
    unsigned = {"files": files, "fingerprint": job.name}
    if hashlib.sha256(canonical_json(unsigned).encode("utf-8")).hexdigest() != digest:
        raise RemoteJobError("completed bundle digest is invalid")
    state = _load_object(job / "status.json", "status")
    return {**state, "prepared_bundle_sha256": digest}


def incomplete_preparation_state(job_directory: str | Path) -> dict[str, object]:
    """Affirm a fully moved bundle whose final completion marker is absent."""
    job = Path(job_directory)
    marker = job / ".complete.json"
    if marker.exists() or marker.is_symlink():
        raise RemoteJobError("incomplete preparation must not have a completion marker")
    bundle = _load_object(job / "bundle-manifest.json", "bundle manifest")
    if set(bundle) != {"bundle_sha256", "files", "fingerprint"}:
        raise RemoteJobError("bundle manifest fields are invalid")
    digest = bundle.get("bundle_sha256")
    files = bundle.get("files")
    if (
        not isinstance(digest, str)
        or not _SHA256.fullmatch(digest)
        or bundle.get("fingerprint") != job.name
        or not isinstance(files, Mapping)
    ):
        raise RemoteJobError("incomplete bundle identity is invalid")
    unsigned = {"files": files, "fingerprint": job.name}
    if hashlib.sha256(canonical_json(unsigned).encode("utf-8")).hexdigest() != digest:
        raise RemoteJobError("incomplete bundle manifest digest mismatch")
    names = set(files)
    if not _BUNDLE_REQUIRED_FILES.issubset(names) or not names.issubset(
        _BUNDLE_REQUIRED_FILES | _BUNDLE_OPTIONAL_FILES
    ):
        raise RemoteJobError("incomplete bundle file set is invalid")
    for name in sorted(names):
        record = files.get(name)
        path = job / name
        if (
            not isinstance(record, Mapping)
            or set(record) != {"sha256", "size"}
            or not isinstance(record.get("sha256"), str)
            or not _SHA256.fullmatch(str(record.get("sha256")))
            or isinstance(record.get("size"), bool)
            or not isinstance(record.get("size"), int)
            or int(record.get("size")) < 0
            or not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != record.get("size")
            or sha256_file(path) != record.get("sha256")
        ):
            raise RemoteJobError(f"incomplete bundle file mismatch: {name}")
    prepared = _load_object(job / "prepared-manifest.json", "prepared manifest")
    source = prepared.get("source")
    command = prepared.get("command")
    argv = load_train_argv(job)
    status = _load_object(job / "status.json", "status")
    if (
        prepared.get("fingerprint") != job.name
        or not isinstance(source, Mapping)
        or source.get("archive_sha256") != sha256_file(job / "source.tar")
        or not isinstance(command, Mapping)
        or command.get("argv_file") != "train-argv.json"
        or command.get("sha256") != command_digest(argv)
        or status.get("status") != "pending"
    ):
        raise RemoteJobError("incomplete prepared manifests are invalid")
    return {
        **status,
        "incomplete_bundle_sha256": digest,
        "preparation_status": "incomplete",
    }


def launch_training(
    argv: tuple[str, ...],
    *,
    cwd: str | Path,
    stdout: BinaryIO,
    popen: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
) -> subprocess.Popen[bytes]:
    """Launch the validated trainer without a shell in a new process group."""
    return popen(
        argv,
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


def wait_for_identity_exit(
    identity: ProcessIdentity,
    timeout: float,
    *,
    identity_fn: Callable[[int], ProcessIdentity] = live_process_identity,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """Wait until the exact recorded process is gone (PID reuse counts as gone)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            current = identity_fn(identity.pid)
        except RemoteJobError:
            return True
        if current != identity:
            return True
        sleep(min(0.05, max(0.0, deadline - time.monotonic())))
    return False


def cancel_job(
    job_directory: str | Path,
    *,
    killpg: Callable[[int, int], None] = os.killpg,
    wait_for_exit: Callable[[ProcessIdentity, float], bool] | None = None,
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
    wait = wait_for_exit or wait_for_identity_exit
    killpg(expected.pid, signal.SIGTERM)
    if not wait(expected, _TERMINATE_TIMEOUT_SECONDS):
        if live_process_identity(expected.pid) != expected:
            terminated = True
        else:
            if os.getpgid(expected.pid) != expected.pid:
                raise RemoteJobError("trainer process group identity changed during cancellation")
            killpg(expected.pid, signal.SIGKILL)
            terminated = wait(expected, _TERMINATE_TIMEOUT_SECONDS)
    else:
        terminated = True
    if not terminated:
        raise RemoteJobError("trainer process group did not terminate within the bounded timeout")
    latest = _load_object(job / "status.json", "status")
    result = {
        **state,
        **latest,
        **expected.as_dict(),
        "status": "failed",
        "cancelled": True,
        "exit_code": None,
        "termination_confirmed": True,
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
        try:
            matches = (
                destination.is_dir()
                and not destination.is_symlink()
                and marker.read_text(encoding="ascii").strip() == expected
            )
        except OSError:
            matches = False
        if not matches:
            raise RemoteJobError("existing source directory does not match the prepared archive")
    else:
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
            shutil.rmtree(temporary, ignore_errors=True)
            raise
    _stage_resume_checkpoint(job, destination, manifest.get("resume"))
    return destination


def _stage_resume_checkpoint(job: Path, source: Path, raw: object) -> None:
    if raw is None:
        return
    if not isinstance(raw, Mapping):
        raise RemoteJobError("resume record must be a JSON object")
    expected_fields = {
        "checkpoint",
        "sha256",
        "size",
        "source_fingerprint",
        "target_relative_path",
    }
    if set(raw) != expected_fields:
        raise RemoteJobError("resume record fields are invalid")
    checkpoint = raw.get("checkpoint")
    digest = raw.get("sha256")
    size = raw.get("size")
    source_fingerprint = raw.get("source_fingerprint")
    relative = raw.get("target_relative_path")
    if (
        not isinstance(checkpoint, str)
        or not _CHECKPOINT.fullmatch(checkpoint)
        or not isinstance(digest, str)
        or not _SHA256.fullmatch(digest)
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size <= 0
        or not isinstance(source_fingerprint, str)
        or not re.fullmatch(r"[0-9a-f]{64}", source_fingerprint)
        or not isinstance(relative, str)
    ):
        raise RemoteJobError("resume record values are invalid")
    relative_path = Path(relative)
    if (
        relative_path.is_absolute()
        or relative_path.parts[:2] != ("source", "logs")
        or any(part in ("", ".", "..") for part in relative_path.parts)
        or relative_path.name != checkpoint
    ):
        raise RemoteJobError("resume target path is invalid")
    staged = job / "resume-checkpoint.pt"
    if (
        not staged.is_file()
        or staged.is_symlink()
        or staged.stat().st_size != size
        or sha256_file(staged) != digest
    ):
        raise RemoteJobError("resume checkpoint digest or size mismatch")
    target = job.joinpath(*relative_path.parts)
    try:
        target.relative_to(source)
    except ValueError as error:
        raise RemoteJobError("resume target must stay beneath extracted source") from error
    target.parent.mkdir(parents=True, exist_ok=True)
    current = target.parent
    while current != source:
        if current.is_symlink():
            raise RemoteJobError("resume target parent must not be a symlink")
        current = current.parent
    if target.exists():
        if target.is_symlink() or target.stat().st_size != size or sha256_file(target) != digest:
            raise RemoteJobError("existing resume target does not match prepared checkpoint")
        return
    temporary = target.with_name(f".{target.name}.staging")
    try:
        temporary.unlink()
    except FileNotFoundError:
        pass
    os.link(staged, temporary, follow_symlinks=False)
    os.replace(temporary, target)


def _checkpoint_paths(source: Path) -> Iterable[Path]:
    return source.glob("logs/**/model_*.pt")


def terminate_and_reap(
    process: subprocess.Popen[bytes],
    identity: ProcessIdentity | None = None,
    *,
    killpg: Callable[[int, int], None] = os.killpg,
) -> None:
    """Terminate the exact new-session group and reap its leader."""
    if process.returncode is not None:
        process.wait()
        return
    if os.getpgid(process.pid) != process.pid:
        raise RemoteJobError("spawned trainer is not its process-group leader")
    if identity is not None and live_process_identity(process.pid) != identity:
        raise RemoteJobError("spawned trainer identity changed before cleanup")
    killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=_TERMINATE_TIMEOUT_SECONDS)
        return
    except subprocess.TimeoutExpired:
        pass
    if identity is not None:
        try:
            if live_process_identity(process.pid) != identity:
                process.wait(timeout=_TERMINATE_TIMEOUT_SECONDS)
                return
        except RemoteJobError:
            process.wait(timeout=_TERMINATE_TIMEOUT_SECONDS)
            return
    killpg(process.pid, signal.SIGKILL)
    try:
        process.wait(timeout=_TERMINATE_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as error:
        raise RemoteJobError("spawned trainer could not be reaped") from error


def finalize_training_state(
    job_directory: str | Path,
    state: Mapping[str, object],
    *,
    exit_code: int,
    last_checkpoint: Mapping[str, object] | None,
) -> dict[str, object]:
    """Persist terminal state without overwriting a confirmed cancellation."""
    job = Path(job_directory)
    try:
        persisted = _load_object(job / "status.json", "status")
    except RemoteJobError:
        persisted = {}
    if persisted.get("cancelled") is True and persisted.get("termination_confirmed") is True:
        final = {**state, **persisted, "exit_code": exit_code, "status": "failed"}
    else:
        final = {
            **state,
            "artifact_status": "stable_checkpoint" if last_checkpoint else "missing",
            "exit_code": exit_code,
            "last_stable_checkpoint": last_checkpoint,
            "status": "succeeded" if exit_code == 0 else "failed",
        }
    atomic_write_json(job / "status.json", final)
    return final


def supervise(
    job_directory: str | Path,
    *,
    terminate_and_reap: Callable[
        [subprocess.Popen[bytes], ProcessIdentity | None], None
    ] = terminate_and_reap,
) -> dict[str, object]:
    """Run and monitor one trainer, atomically recording every lifecycle state."""
    job = Path(job_directory)
    source = _verified_source(job)
    argv = verified_train_argv(job)
    digest = command_digest(argv)
    stdout_path = job / "stdout.log"
    tracker = StableCheckpointTracker(job)
    last_checkpoint: dict[str, object] | None = None
    process: subprocess.Popen[bytes] | None = None
    identity: ProcessIdentity | None = None
    state: dict[str, object] = {
        "artifact_status": "pending",
        "command_digest": digest,
        "exit_code": None,
        "last_stable_checkpoint": None,
        "pid": None,
        "process_start": None,
        "status": "pending",
        "stdout_path": str(stdout_path),
    }
    try:
        with stdout_path.open("ab", buffering=0) as output:
            process = launch_training(argv, cwd=source, stdout=output)
            state = {**state, "pid": process.pid}
            identity = live_process_identity(process.pid)
            if identity.command_digest != digest:
                raise RemoteJobError("launched trainer command identity mismatch")
            state = {**state, **identity.as_dict(), "status": "running"}
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
        if exit_code is None:
            raise RemoteJobError("trainer exited without a return code")
        return finalize_training_state(
            job,
            state,
            exit_code=exit_code,
            last_checkpoint=last_checkpoint,
        )
    except BaseException as error:
        terminated = process is None
        if process is not None:
            terminate_and_reap(process, identity)
            terminated = True
        try:
            persisted = _load_object(job / "status.json", "status")
        except RemoteJobError:
            persisted = {}
        if not (persisted.get("cancelled") is True and persisted.get("termination_confirmed") is True):
            failed = {
                **state,
                **persisted,
                "error": str(error),
                "exit_code": process.returncode if process is not None else 1,
                "status": "failed",
                "termination_confirmed": terminated,
            }
            atomic_write_json(job / "status.json", failed)
        raise


def run_supervisor(job: Path) -> dict[str, object]:
    """Run the supervisor and make pre-launch failures durable."""
    with supervisor_lock(job):
        with lifecycle_lock(job):
            state = _load_object(job / "status.json", "status")
            if state.get("status") in ("running", "succeeded", "failed"):
                return state
            state = {
                **state,
                "launch_state": "supervising",
                "supervisor_pid": os.getpid(),
            }
            atomic_write_json(job / "status.json", state)
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


def _supervisor_argv(job: Path) -> tuple[str, ...]:
    return (sys.executable, str(Path(__file__).resolve()), str(job))


def _spawned_supervisor_is_live(
    state: Mapping[str, object],
    *,
    command_sha256: str,
) -> bool:
    raw = state.get("supervisor_identity")
    if not isinstance(raw, Mapping) or state.get("supervisor_pid") != raw.get("pid"):
        return False
    try:
        recorded = _identity_from(raw)
        current = live_process_identity(recorded.pid)
    except RemoteJobError:
        return False
    return recorded == current and recorded.command_digest == command_sha256


def _start_supervisor(
    job: Path,
    *,
    popen: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
    after_claim: Callable[[Mapping[str, object]], None] | None = None,
    after_popen: Callable[[subprocess.Popen[bytes]], None] | None = None,
    after_spawn: Callable[[Mapping[str, object]], None] | None = None,
) -> dict[str, object]:
    request = _load_object(job / "start-request.json", "start request")
    state = _load_object(job / "status.json", "status")
    request_id = request.get("request_id")
    if (
        request.get("action") != "start"
        or set(request) != {"action", "request_id"}
        or not isinstance(request_id, str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", request_id)
    ):
        raise RemoteJobError("start request is invalid")
    supervisor_argv = _supervisor_argv(job)
    supervisor_command_sha256 = command_digest(supervisor_argv)
    if (
        state.get("status") == "failed"
        and state.get("launch_state") == "supervisor_lost"
        and state.get("retryable_start") is True
    ):
        state = {
            **{
                key: value
                for key, value in state.items()
                if key
                not in {
                    "error",
                    "exit_code",
                    "launch_request_id",
                    "launch_state",
                    "retryable_start",
                    "supervisor_identity",
                    "supervisor_pid",
                }
            },
            "status": "pending",
        }
    if state.get("launch_request_id") == request_id and state.get("launch_state") == "spawned":
        if _spawned_supervisor_is_live(
            state,
            command_sha256=supervisor_command_sha256,
        ):
            try:
                (job / "start-request.json").unlink()
            except FileNotFoundError:
                pass
            return state
    if state.get("status") != "pending":
        if state.get("launch_request_id") == request_id:
            try:
                (job / "start-request.json").unlink()
            except FileNotFoundError:
                pass
            return state
        raise RemoteJobError("start requires an exact request and pending state")
    existing_request = state.get("launch_request_id")
    if existing_request not in (None, request_id):
        raise RemoteJobError("a different start request already owns this job")
    claimed = {
        **{
            key: value
            for key, value in state.items()
            if key not in {"supervisor_identity", "supervisor_pid"}
        },
        "launch_request_id": request_id,
        "launch_state": "claimed",
    }
    atomic_write_json(job / "status.json", claimed)
    if after_claim is not None:
        after_claim(claimed)
    environment = os.environ.copy()
    environment[_SUPERVISOR_ENV] = "1"
    try:
        supervisor = popen(
            supervisor_argv,
            cwd=job,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=False,
            start_new_session=True,
            close_fds=True,
            env=environment,
        )
    except BaseException:
        atomic_write_json(job / "status.json", state)
        raise
    if after_popen is not None:
        after_popen(supervisor)
    supervisor_identity = live_process_identity(supervisor.pid)
    if supervisor_identity.command_digest != supervisor_command_sha256:
        raise RemoteJobError("spawned supervisor command identity mismatch")
    spawned = {
        **claimed,
        "launch_state": "spawned",
        "supervisor_pid": supervisor.pid,
        "supervisor_identity": supervisor_identity.as_dict(),
    }
    atomic_write_json(job / "status.json", spawned)
    if after_spawn is not None:
        after_spawn(spawned)
    try:
        (job / "start-request.json").unlink()
    except FileNotFoundError:
        raise RemoteJobError("start request disappeared") from None
    return spawned


def inspect_or_control(
    job: Path,
    *,
    supervisor_popen: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
    after_claim: Callable[[Mapping[str, object]], None] | None = None,
    after_popen: Callable[[subprocess.Popen[bytes]], None] | None = None,
    after_spawn: Callable[[Mapping[str, object]], None] | None = None,
) -> dict[str, object]:
    with lifecycle_lock(job):
        if (job / "cancel-request.json").exists():
            return cancel_job(job)
        if (job / "start-request.json").exists():
            return _start_supervisor(
                job,
                popen=supervisor_popen,
                after_claim=after_claim,
                after_popen=after_popen,
                after_spawn=after_spawn,
            )
        return _inspection_state(job)


def _inspection_state(job: Path) -> dict[str, object]:
    state = _load_object(job / "status.json", "status")
    if state.get("status") == "pending" and state.get("launch_state") == "spawned":
        supervisor_command_sha256 = command_digest(_supervisor_argv(job))
        if not _spawned_supervisor_is_live(
            state,
            command_sha256=supervisor_command_sha256,
        ):
            state = {
                **{
                    key: value
                    for key, value in state.items()
                    if key != "launch_request_id"
                },
                "error": "detached supervisor identity is no longer live",
                "exit_code": 1,
                "launch_state": "supervisor_lost",
                "retryable_start": True,
                "status": "failed",
            }
            atomic_write_json(job / "status.json", state)
            return state
    if state.get("status") == "running":
        try:
            live = live_process_identity(_identity_from(state).pid)
        except RemoteJobError:
            live = None
        state = {**state, "live_identity": live.as_dict() if live else None}
    return state


def inspect_read_only(job: Path) -> dict[str, object]:
    """Read lifecycle state without consuming start or cancel requests."""
    with lifecycle_lock(job):
        return _inspection_state(job)


def inspect_preparation_read_only(job: Path) -> dict[str, object]:
    """Inspect complete or affirmatively incomplete preparation without mutation."""
    try:
        completed = completed_preparation_state(job)
    except RemoteJobError:
        return incomplete_preparation_state(job)
    state = inspect_read_only(job)
    return {**state, "prepared_bundle_sha256": completed["prepared_bundle_sha256"]}


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 1:
        raise SystemExit("usage: next_rl_remote_job.py JOB_DIRECTORY")
    job = validate_job_directory(arguments[0])
    script_directory = Path(__file__).resolve().parent
    if script_directory.parent == job and script_directory.name.startswith(".incoming-"):
        print(canonical_json(finalize_preparation(job, script_directory)), flush=True)
        return 0
    if os.environ.get(_INSPECT_ENV) == "1":
        print(canonical_json(inspect_preparation_read_only(job)), flush=True)
        return 0
    completed = completed_preparation_state(job)
    if os.environ.get(_SUPERVISOR_ENV) == "1":
        run_supervisor(job)
        return 0
    state = inspect_or_control(job)
    print(
        canonical_json(
            {**state, "prepared_bundle_sha256": completed["prepared_bundle_sha256"]}
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RemoteJobError as error:
        print(f"next_rl_remote_job: {error}", file=sys.stderr)
        raise SystemExit(2) from error

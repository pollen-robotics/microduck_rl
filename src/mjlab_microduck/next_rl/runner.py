"""Safe command construction and Nitro job orchestration."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import subprocess
import tarfile
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol

from .artifacts import atomic_write_json, canonical_json, sha256_file
from .experiments import experiment_fingerprint, learning_inputs
from .schema import ExperimentManifest

_TASK_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]*$")
_RUN_SLUG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_CHECKPOINT = re.compile(r"^model_[0-9]+\.pt$")
_FINGERPRINT = re.compile(r"^[0-9a-f]{6,64}$")
_SSH_ALIAS = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.-]*$")
_SSH_USER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_REMOTE_ROOT = PurePosixPath("/home/aif_eng/microduck-training/runs")
_SECRET_FILENAMES = frozenset(
    {
        ".env",
        ".netrc",
        ".npmrc",
        ".pypirc",
        "authorized_keys",
        "credentials",
        "credentials.json",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
        "id_rsa",
        "private_key",
        "secrets",
        "secrets.json",
    }
)
_SECRET_SUFFIXES = frozenset({".key", ".p12", ".pem", ".pfx"})
_SECRET_STEMS = frozenset(
    {
        "access_token",
        "api_key",
        "credential",
        "credentials",
        "password",
        "passwd",
        "private_key",
        "secret",
        "secrets",
    }
)
_SAFE_PATH_PART = re.compile(r"^[A-Za-z0-9_.-]+$")


class RunnerError(RuntimeError):
    """A requested runner operation is invalid or unsafe."""


@dataclass(frozen=True)
class CommandResult:
    """The captured result of one local transport command."""

    stdout: str = ""
    stderr: str = ""
    returncode: int = 0


class CommandAdapter(Protocol):
    """Injected boundary for local ``ssh`` and ``scp`` processes."""

    def run(self, argv: tuple[str, ...]) -> CommandResult:
        """Run *argv* without a shell and return captured text output."""


class OpenSSHAdapter:
    """Execute OpenSSH client commands as argument arrays without a shell."""

    def run(self, argv: tuple[str, ...]) -> CommandResult:
        completed = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            shell=False,
        )
        result = CommandResult(completed.stdout, completed.stderr, completed.returncode)
        if completed.returncode:
            raise RunnerError(
                f"transport command failed with exit code {completed.returncode}: "
                f"{completed.stderr.strip()}"
            )
        return result


@dataclass(frozen=True)
class NitroConfig:
    """Credential-free Nitro endpoint and local source/bundle locations."""

    ssh_alias: str
    repository: Path
    bundle_root: Path
    ssh_user: str = "aif_eng"

    def __post_init__(self) -> None:
        for value, name, pattern in (
            (self.ssh_alias, "SSH alias", _SSH_ALIAS),
            (self.ssh_user, "SSH user", _SSH_USER),
        ):
            if not isinstance(value, str) or not pattern.fullmatch(value):
                raise RunnerError(f"{name} uses unsafe syntax")
        object.__setattr__(self, "repository", Path(self.repository).resolve())
        object.__setattr__(self, "bundle_root", Path(self.bundle_root).resolve())


@dataclass(frozen=True)
class PreparedJob:
    """A local, credential-free bundle already transferred to its fixed path."""

    fingerprint: str
    manifest: Mapping[str, object]
    local_directory: Path
    remote_directory: PurePosixPath
    archive_path: Path
    archive_sha256: str
    source_commit: str
    source_tree: str
    train_argv: tuple[str, ...]
    dry_run_argv: tuple[tuple[str, ...], ...]


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RunnerError(f"{name} must be a positive integer")
    return value


def _slug(value: object, name: str) -> str:
    if not isinstance(value, str) or ".." in value or not _RUN_SLUG.fullmatch(value):
        raise RunnerError(f"{name} uses unsafe syntax")
    return value


def build_train_argv(manifest: ExperimentManifest) -> tuple[str, ...]:
    """Build the fixed ``uv run train`` argv for one validated experiment."""
    if not isinstance(manifest, ExperimentManifest):
        raise TypeError("manifest must be an ExperimentManifest")
    if ".." in manifest.task_id or not _TASK_ID.fullmatch(manifest.task_id):
        raise RunnerError("task ID uses unsafe syntax")

    scene = manifest.environment_config.get("scene")
    if not isinstance(scene, dict) and not hasattr(scene, "get"):
        raise RunnerError("environment_config.scene must be a mapping")
    num_envs = _positive_integer(scene.get("num_envs"), "scene.num_envs")
    run_name = _slug(manifest.agent_config.get("run_name"), "run name")
    if manifest.seed < 0:
        raise RunnerError("seed must be non-negative")

    argv = [
        "uv",
        "run",
        "train",
        manifest.task_id,
        "--env.scene.num-envs",
        str(num_envs),
    ]
    resume = manifest.agent_config.get("resume", False)
    if not isinstance(resume, bool):
        raise RunnerError("agent resume must be a boolean")
    if resume:
        run = _slug(manifest.agent_config.get("load_run"), "resume run")
        checkpoint = manifest.agent_config.get("load_checkpoint")
        if not isinstance(checkpoint, str) or not _CHECKPOINT.fullmatch(checkpoint):
            raise RunnerError("resume checkpoint must be model_<iteration>.pt")
        iterations = _positive_integer(
            manifest.agent_config.get("additional_iterations"),
            "additional_iterations",
        )
        argv.extend(
            (
                "--agent.seed",
                str(manifest.seed),
                "--agent.run-name",
                run_name,
                "--agent.resume",
                "True",
                "--agent.load-run",
                run,
                "--agent.load-checkpoint",
                checkpoint,
                "--agent.max-iterations",
                str(iterations),
            )
        )
    else:
        iterations = _positive_integer(
            manifest.agent_config.get("max_iterations"),
            "max_iterations",
        )
        argv.extend(
            (
                "--agent.max-iterations",
                str(iterations),
                "--agent.seed",
                str(manifest.seed),
                "--agent.run-name",
                run_name,
            )
        )
    return tuple(argv)


def _git(repository: Path, *args: str) -> bytes:
    completed = subprocess.run(
        ("git", *args),
        cwd=repository,
        check=False,
        capture_output=True,
        shell=False,
    )
    if completed.returncode:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RunnerError(f"git source snapshot failed: {message}")
    return completed.stdout


def _safe_tracked_path(raw: bytes) -> str:
    try:
        value = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RunnerError("tracked paths must be valid UTF-8") from error
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in ("", ".", "..") for part in path.parts):
        raise RunnerError(f"tracked path is unsafe: {value!r}")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise RunnerError(f"tracked path contains control characters: {value!r}")
    for part in path.parts:
        lowered = part.lower()
        stem = PurePosixPath(lowered).stem.replace("-", "_")
        if (
            lowered in _SECRET_FILENAMES
            or lowered.startswith(".env.")
            or PurePosixPath(lowered).suffix in _SECRET_SUFFIXES
            or stem in _SECRET_STEMS
        ):
            raise RunnerError(f"tracked path looks secret-like: {value!r}")
    return value


def _snapshot_source(repository: Path, target: Path) -> tuple[str, str, str]:
    """Archive regular blobs from one captured commit tree with stable metadata."""
    commit = _git(repository, "rev-parse", "--verify", "HEAD").decode("ascii").strip()
    tree = _git(repository, "rev-parse", "--verify", f"{commit}^{{tree}}").decode("ascii").strip()
    listing = _git(repository, "ls-tree", "-rz", "--full-tree", "-r", tree)
    entries: list[tuple[str, str, str]] = []
    for record in listing.rstrip(b"\0").split(b"\0") if listing else ():
        header, separator, raw_path = record.partition(b"\t")
        fields = header.split()
        if not separator or len(fields) != 3:
            raise RunnerError("git returned a malformed tree record")
        mode, object_type, object_id = (field.decode("ascii") for field in fields)
        path = _safe_tracked_path(raw_path)
        if object_type != "blob" or mode not in ("100644", "100755"):
            raise RunnerError(f"tracked entry must be a regular file: {path!r}")
        entries.append((path, mode, object_id))

    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    try:
        with tarfile.open(temporary, "w", format=tarfile.USTAR_FORMAT) as archive:
            for path, mode, object_id in entries:
                content = _git(repository, "cat-file", "blob", object_id)
                info = tarfile.TarInfo(path)
                info.size = len(content)
                info.mode = 0o755 if mode == "100755" else 0o644
                info.mtime = 0
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                archive.addfile(info, io.BytesIO(content))
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return commit, tree, sha256_file(target)


class NitroRunner:
    """Prepare and control fingerprint-isolated Nitro jobs through OpenSSH."""

    def __init__(
        self,
        config: NitroConfig,
        adapter: CommandAdapter | None = None,
    ) -> None:
        self.config = config
        self.adapter = adapter or OpenSSHAdapter()

    @property
    def _target(self) -> str:
        return f"{self.config.ssh_user}@{self.config.ssh_alias}"

    def _remote_directory(self, fingerprint: str) -> PurePosixPath:
        if not isinstance(fingerprint, str) or not _FINGERPRINT.fullmatch(fingerprint):
            raise RunnerError("fingerprint must be lowercase hexadecimal")
        return _REMOTE_ROOT / fingerprint

    def _ssh(self, *remote_argv: str) -> tuple[str, ...]:
        return ("ssh", "-o", "BatchMode=yes", self._target, *remote_argv)

    def _scp(self, sources: Sequence[Path], remote_directory: PurePosixPath) -> tuple[str, ...]:
        return (
            "scp",
            "-o",
            "BatchMode=yes",
            *(str(source) for source in sources),
            f"{self._target}:{remote_directory}/",
        )

    def prepare(self, manifest: ExperimentManifest) -> PreparedJob:
        """Create and transfer a deterministic, credential-free source bundle."""
        train_argv = build_train_argv(manifest)
        fingerprint = experiment_fingerprint(manifest)
        local_directory = self.config.bundle_root / fingerprint
        remote_directory = self._remote_directory(fingerprint)
        local_directory.mkdir(parents=True, exist_ok=True)

        archive_path = local_directory / "source.tar"
        commit, tree, archive_sha256 = _snapshot_source(
            self.config.repository,
            archive_path,
        )
        prepared_manifest: dict[str, object] = {
            "fingerprint": fingerprint,
            "learning_inputs": learning_inputs(manifest),
            "source": {
                "commit": commit,
                "tree": tree,
                "archive": archive_path.name,
                "archive_sha256": archive_sha256,
            },
        }
        manifest_path = local_directory / "prepared-manifest.json"
        argv_path = local_directory / "train-argv.json"
        status_path = local_directory / "status.json"
        atomic_write_json(manifest_path, prepared_manifest)
        atomic_write_json(argv_path, list(train_argv))
        atomic_write_json(
            status_path,
            {
                "artifact_status": "pending",
                "command_digest": hashlib.sha256(
                    canonical_json(list(train_argv)).encode("utf-8")
                ).hexdigest(),
                "exit_code": None,
                "last_stable_checkpoint": None,
                "pid": None,
                "process_start": None,
                "status": "pending",
                "stdout_path": str(remote_directory / "stdout.log"),
            },
        )
        wrapper_path = local_directory / "next_rl_remote_job.py"
        try:
            with tarfile.open(archive_path, "r") as archive:
                member = archive.getmember("scripts/next_rl_remote_job.py")
                source = archive.extractfile(member) if member.isfile() else None
                if source is None:
                    raise RunnerError("remote wrapper must be a regular tracked source file")
                descriptor, temporary = tempfile.mkstemp(
                    prefix=f".{wrapper_path.name}.", suffix=".tmp", dir=local_directory
                )
                try:
                    with os.fdopen(descriptor, "wb") as output:
                        while block := source.read(1024 * 1024):
                            output.write(block)
                        output.flush()
                        os.fsync(output.fileno())
                    os.replace(temporary, wrapper_path)
                except BaseException:
                    try:
                        os.unlink(temporary)
                    except FileNotFoundError:
                        pass
                    raise
        except KeyError as error:
            raise RunnerError("remote wrapper must be a regular tracked source file") from error

        mkdir_argv = self._ssh("mkdir", "-p", "--", str(remote_directory))
        transfer_argv = self._scp(
            (archive_path, manifest_path, argv_path, status_path, wrapper_path),
            remote_directory,
        )
        self.adapter.run(mkdir_argv)
        self.adapter.run(transfer_argv)
        return PreparedJob(
            fingerprint=fingerprint,
            manifest=prepared_manifest,
            local_directory=local_directory,
            remote_directory=remote_directory,
            archive_path=archive_path,
            archive_sha256=archive_sha256,
            source_commit=commit,
            source_tree=tree,
            train_argv=train_argv,
            dry_run_argv=(mkdir_argv, transfer_argv),
        )

    def _invoke_wrapper(self, remote_directory: PurePosixPath) -> CommandResult:
        return self.adapter.run(
            self._ssh(
                "python3",
                str(remote_directory / "next_rl_remote_job.py"),
                str(remote_directory),
            )
        )

    @staticmethod
    def _state(result: CommandResult) -> dict[str, object]:
        try:
            state = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise RunnerError("remote wrapper returned invalid JSON") from error
        if not isinstance(state, dict):
            raise RunnerError("remote wrapper state must be a JSON object")
        return state

    def start(self, prepared: PreparedJob) -> dict[str, object]:
        """Request detached execution; a disconnect never triggers cancellation."""
        expected = self._remote_directory(prepared.fingerprint)
        if prepared.remote_directory != expected:
            raise RunnerError("prepared job remote directory is outside its fingerprint path")
        request_path = prepared.local_directory / "start-request.json"
        atomic_write_json(request_path, {"action": "start"})
        self.adapter.run(self._scp((request_path,), expected))
        return self._state(self._invoke_wrapper(expected))

    def status(self, fingerprint: str) -> dict[str, object]:
        """Return remote state plus a live process identity when applicable."""
        remote_directory = self._remote_directory(fingerprint)
        return self._state(self._invoke_wrapper(remote_directory))

    def cancel(self, fingerprint: str) -> dict[str, object]:
        """Cancel only when live PID, start time, and command digest all match."""
        state = self.status(fingerprint)
        expected = {
            "pid": state.get("pid"),
            "process_start": state.get("process_start"),
            "command_digest": state.get("command_digest"),
        }
        live = state.get("live_identity")
        if state.get("status") != "running" or not isinstance(live, Mapping):
            raise RunnerError("remote job has no running process identity")
        if any(expected[key] != live.get(key) for key in expected):
            raise RunnerError("remote process identity no longer matches recorded job")

        remote_directory = self._remote_directory(fingerprint)
        local_directory = self.config.bundle_root / fingerprint
        local_directory.mkdir(parents=True, exist_ok=True)
        request_path = local_directory / "cancel-request.json"
        atomic_write_json(request_path, {"action": "cancel", **expected})
        self.adapter.run(self._scp((request_path,), remote_directory))
        return self._state(self._invoke_wrapper(remote_directory))

    def sync_checkpoint(self, fingerprint: str, *, attempts: int = 3) -> Path:
        """Download one stable checkpoint and publish it only after digest verification."""
        attempts = _positive_integer(attempts, "checkpoint transfer attempts")
        state = self.status(fingerprint)
        checkpoint = state.get("last_stable_checkpoint")
        if not isinstance(checkpoint, Mapping):
            raise RunnerError("remote job has no stable checkpoint")
        name = checkpoint.get("name")
        relative = checkpoint.get("relative_path")
        digest = checkpoint.get("sha256")
        size = checkpoint.get("size")
        if not isinstance(name, str) or not _CHECKPOINT.fullmatch(name):
            raise RunnerError("stable checkpoint name is invalid")
        if not isinstance(relative, str):
            raise RunnerError("stable checkpoint relative path is invalid")
        relative_path = PurePosixPath(relative)
        if (
            relative_path.is_absolute()
            or not relative_path.parts
            or any(part in ("", ".", "..") for part in relative_path.parts)
            or any(not _SAFE_PATH_PART.fullmatch(part) for part in relative_path.parts)
            or relative_path.parts[:2] != ("source", "logs")
            or relative_path.name != name
        ):
            raise RunnerError("stable checkpoint path is outside the job directory")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise RunnerError("stable checkpoint digest is invalid")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise RunnerError("stable checkpoint size is invalid")

        remote_directory = self._remote_directory(fingerprint)
        destination_directory = self.config.bundle_root / fingerprint / "checkpoints"
        destination_directory.mkdir(parents=True, exist_ok=True)
        destination = destination_directory / name
        temporary = destination.with_name(f".{name}.part")
        source = f"{self._target}:{remote_directory / relative_path}"
        last_digest = "not downloaded"
        for _ in range(attempts):
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            self.adapter.run(
                ("scp", "-o", "BatchMode=yes", source, str(temporary))
            )
            if not temporary.is_file() or temporary.is_symlink():
                last_digest = "missing file"
                continue
            last_digest = sha256_file(temporary)
            if temporary.stat().st_size == size and last_digest == digest:
                os.replace(temporary, destination)
                return destination
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise RunnerError(
            f"checkpoint transfer digest verification failed after {attempts} attempts "
            f"(last result: {last_digest})"
        )

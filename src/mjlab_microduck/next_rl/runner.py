"""Safe command construction and Nitro job orchestration."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol

from .artifacts import atomic_write_json, canonical_json, sha256_file
from .experiments import (
    DuplicateExperimentError,
    ExperimentStore,
    experiment_fingerprint,
    learning_inputs,
)
from .schema import ExperimentManifest

_TASK_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]*$")
_RUN_SLUG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_CHECKPOINT = re.compile(r"^model_[0-9]+\.pt$")
_FINGERPRINT = re.compile(r"^[0-9a-f]{6,64}$")
_SSH_ALIAS = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.-]*$")
_SSH_USER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_WSL_DISTRIBUTION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_REMOTE_ROOT = PurePosixPath("/home/aif_eng/microduck-training/runs")
_STDERR_LIMIT = 64 * 1024
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
_SAFE_PATH_PART = re.compile(r"^[A-Za-z0-9_.-]+$")
_SENSITIVE_WORDS = frozenset(
    {
        "credential",
        "credentials",
        "passwd",
        "password",
        "secret",
        "secrets",
    }
)
_SENSITIVE_PAIRS = frozenset(
    {
        ("access", "key"),
        ("access", "token"),
        ("api", "key"),
        ("api", "token"),
        ("auth", "token"),
        ("client", "secret"),
        ("private", "key"),
        ("refresh", "token"),
        ("session", "token"),
    }
)


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

    def upload(self, source: Path, argv: tuple[str, ...]) -> CommandResult:
        """Stream one local binary file to the stdin of *argv*."""

    def download(self, argv: tuple[str, ...], destination: Path) -> CommandResult:
        """Stream stdout from *argv* into one local binary file."""


class OpenSSHAdapter:
    """Execute OpenSSH argv and return every process exit status as data."""

    def run(self, argv: tuple[str, ...]) -> CommandResult:
        completed = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            shell=False,
        )
        return CommandResult(completed.stdout, completed.stderr, completed.returncode)

    @staticmethod
    def _stderr(error_file) -> str:
        error_file.seek(0)
        return error_file.read(_STDERR_LIMIT).decode("utf-8", errors="replace")

    def upload(self, source: Path, argv: tuple[str, ...]) -> CommandResult:
        with Path(source).open("rb") as input_file, tempfile.TemporaryFile() as error_file:
            completed = subprocess.run(
                argv,
                check=False,
                stdin=input_file,
                stdout=subprocess.DEVNULL,
                stderr=error_file,
                text=False,
                shell=False,
            )
            return CommandResult(
                stderr=self._stderr(error_file),
                returncode=completed.returncode,
            )

    def download(self, argv: tuple[str, ...], destination: Path) -> CommandResult:
        with Path(destination).open("wb") as output_file, tempfile.TemporaryFile() as error_file:
            completed = subprocess.run(
                argv,
                check=False,
                stdin=subprocess.DEVNULL,
                stdout=output_file,
                stderr=error_file,
                text=False,
                shell=False,
            )
            return CommandResult(
                stderr=self._stderr(error_file),
                returncode=completed.returncode,
            )


@dataclass(frozen=True)
class NitroConfig:
    """Credential-free Nitro endpoint and local source/bundle locations."""

    ssh_alias: str
    repository: Path
    bundle_root: Path
    ssh_user: str = "aif_eng"
    wsl_distribution: str | None = None

    def __post_init__(self) -> None:
        for value, name, pattern in (
            (self.ssh_alias, "SSH alias", _SSH_ALIAS),
            (self.ssh_user, "SSH user", _SSH_USER),
        ):
            if not isinstance(value, str) or not pattern.fullmatch(value):
                raise RunnerError(f"{name} uses unsafe syntax")
        if self.wsl_distribution is not None and (
            not isinstance(self.wsl_distribution, str)
            or ".." in self.wsl_distribution
            or not _WSL_DISTRIBUTION.fullmatch(self.wsl_distribution)
        ):
            raise RunnerError("WSL distribution uses unsafe syntax")
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


def _name_tokens(value: str) -> tuple[str, ...]:
    separated = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    stem = PurePosixPath(separated.lower()).stem
    return tuple(part for part in re.split(r"[^a-z0-9]+", stem) if part)


def _is_secret_like_name(value: str) -> bool:
    lowered = value.lower()
    if lowered in _SECRET_FILENAMES or lowered.startswith(".env."):
        return True
    if PurePosixPath(lowered).suffix in _SECRET_SUFFIXES:
        return True
    tokens = _name_tokens(value)
    if any(token in _SENSITIVE_WORDS for token in tokens):
        return True
    if "token" in tokens and tokens not in {("token", "budget"), ("token", "count")}:
        return True
    return any((left, right) in _SENSITIVE_PAIRS for left, right in zip(tokens, tokens[1:]))


def _credential_free(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            key: _credential_free(item)
            for key, item in value.items()
            if isinstance(key, str) and not _is_secret_like_name(key)
        }
    if isinstance(value, (tuple, list)):
        return [_credential_free(item) for item in value]
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
        if _is_secret_like_name(part):
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
        *,
        experiment_store: ExperimentStore | None = None,
    ) -> None:
        self.config = config
        self.adapter = adapter or OpenSSHAdapter()
        self.experiment_store = experiment_store

    @property
    def _target(self) -> str:
        return f"{self.config.ssh_user}@{self.config.ssh_alias}"

    def _remote_directory(self, fingerprint: str) -> PurePosixPath:
        if not isinstance(fingerprint, str) or not _FINGERPRINT.fullmatch(fingerprint):
            raise RunnerError("fingerprint must be lowercase hexadecimal")
        return _REMOTE_ROOT / fingerprint

    def _ssh(self, *remote_argv: str) -> tuple[str, ...]:
        bridge = (
            ("wsl.exe", "-d", self.config.wsl_distribution, "--")
            if self.config.wsl_distribution is not None
            else ()
        )
        return ("ssh", "-o", "BatchMode=yes", self._target, *bridge, *remote_argv)

    def _scp(self, sources: Sequence[Path], remote_directory: PurePosixPath) -> tuple[str, ...]:
        return (
            "scp",
            "-o",
            "BatchMode=yes",
            *(str(source) for source in sources),
            f"{self._target}:{remote_directory}/",
        )

    def _run(self, argv: tuple[str, ...]) -> CommandResult:
        result = self.adapter.run(argv)
        return self._require_success(result)

    @staticmethod
    def _require_success(result: CommandResult) -> CommandResult:
        if result.returncode:
            raise RunnerError(
                f"transport command failed with exit code {result.returncode}: "
                f"{result.stderr.strip()}"
            )
        return result

    def _upload_files(
        self,
        sources: Sequence[Path],
        remote_directory: PurePosixPath,
    ) -> tuple[tuple[str, ...], ...]:
        if self.config.wsl_distribution is None:
            argv = self._scp(sources, remote_directory)
            self._run(argv)
            return (argv,)
        calls: list[tuple[str, ...]] = []
        for source in sources:
            if not source.is_file() or source.is_symlink() or not _SAFE_PATH_PART.fullmatch(
                source.name
            ):
                raise RunnerError("local upload source must be a safe regular file")
            destination = remote_directory / source.name
            if (
                not destination.is_relative_to(_REMOTE_ROOT)
                or len(destination.relative_to(_REMOTE_ROOT).parts) < 2
            ):
                raise RunnerError("remote upload destination escaped the fingerprint root")
            argv = self._ssh("tee", "--", str(destination))
            self._require_success(self.adapter.upload(source, argv))
            calls.append(argv)
        return tuple(calls)

    def _upload_argvs(
        self,
        sources: Sequence[Path],
        remote_directory: PurePosixPath,
    ) -> tuple[tuple[str, ...], ...]:
        if self.config.wsl_distribution is None:
            return (self._scp(sources, remote_directory),)
        return tuple(
            self._ssh("tee", "--", str(remote_directory / source.name))
            for source in sources
        )

    def _existing_preparation_matches(
        self,
        remote_directory: PurePosixPath,
        bundle_sha256: str,
    ) -> bool:
        """Inspect an existing directory without turning ambiguity into mutation."""
        self._require_remote_directory(remote_directory, "fingerprint job")
        last_error: BaseException | None = None
        state: dict[str, object] | None = None
        for _ in range(3):
            try:
                state = self._state(self._invoke_wrapper(remote_directory, read_only=True))
                break
            except Exception as error:
                last_error = error
        if state is None:
            raise RunnerError(
                f"remote preparation inspection remained ambiguous: {last_error}"
            ) from last_error
        existing_digest = state.get("prepared_bundle_sha256")
        if existing_digest is None and (
            state.get("preparation_status") == "incomplete"
            and state.get("incomplete_bundle_sha256") == bundle_sha256
        ):
            return False
        if existing_digest is None:
            raise RunnerError("remote preparation inspection returned ambiguous state")
        if existing_digest != bundle_sha256:
            raise RunnerError("remote fingerprint directory contains a conflicting complete job")
        return True

    def _require_remote_directory(self, path: PurePosixPath, label: str) -> None:
        """Affirm that a remote path is a real directory and not a symlink."""
        try:
            directory = self.adapter.run(self._ssh("test", "-d", str(path)))
            non_symlink = self.adapter.run(self._ssh("test", "!", "-L", str(path)))
        except Exception as error:
            raise RunnerError(f"remote {label} type inspection failed") from error
        if directory.returncode or non_symlink.returncode:
            raise RunnerError(f"remote {label} path is not a non-symlink directory")

    def _reset_staging_directory(
        self,
        remote_directory: PurePosixPath,
        staging_directory: PurePosixPath,
    ) -> None:
        """Clean only an affirmatively owned, digest-named staging directory."""
        if (
            remote_directory.parent != _REMOTE_ROOT
            or not _FINGERPRINT.fullmatch(remote_directory.name)
            or staging_directory.parent != remote_directory
            or not re.fullmatch(r"\.incoming-[0-9a-f]{64}", staging_directory.name)
        ):
            raise RunnerError("refusing to clean an unsafe staging path")
        self._require_remote_directory(remote_directory, "fingerprint job")
        try:
            non_symlink = self.adapter.run(
                self._ssh("test", "!", "-L", str(staging_directory))
            )
            absent = self.adapter.run(self._ssh("test", "!", "-e", str(staging_directory)))
        except Exception as error:
            raise RunnerError("remote staging inspection failed") from error
        if non_symlink.returncode:
            raise RunnerError("remote staging path is a symlink")
        if not absent.returncode:
            return
        self._require_remote_directory(staging_directory, "staging")
        try:
            owned = self.adapter.run(self._ssh("test", "-O", str(staging_directory)))
        except Exception as error:
            raise RunnerError("remote staging ownership inspection failed") from error
        if owned.returncode:
            raise RunnerError("remote staging directory is not owned by the SSH user")
        self._run(self._ssh("rm", "-rf", "--", str(staging_directory)))

    @staticmethod
    def _checkpoint_record(state: Mapping[str, object]) -> dict[str, object]:
        checkpoint = state.get("last_stable_checkpoint")
        if not isinstance(checkpoint, Mapping):
            raise RunnerError("remote job has no stable checkpoint")
        name = checkpoint.get("name")
        relative = checkpoint.get("relative_path")
        digest = checkpoint.get("sha256")
        size = checkpoint.get("size")
        mtime_ns = checkpoint.get("mtime_ns")
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
        if isinstance(mtime_ns, bool) or not isinstance(mtime_ns, int) or mtime_ns < 0:
            raise RunnerError("stable checkpoint mtime is invalid")
        return {
            "name": name,
            "relative_path": relative_path.as_posix(),
            "sha256": digest,
            "size": size,
            "mtime_ns": mtime_ns,
        }

    def _download_checkpoint(
        self,
        fingerprint: str,
        checkpoint: Mapping[str, object],
        *,
        attempts: int,
    ) -> Path:
        name = str(checkpoint["name"])
        relative_path = PurePosixPath(str(checkpoint["relative_path"]))
        digest = str(checkpoint["sha256"])
        size = int(checkpoint["size"])
        remote_directory = self._remote_directory(fingerprint)
        destination_directory = self.config.bundle_root / fingerprint / "checkpoints"
        destination_directory.mkdir(parents=True, exist_ok=True)
        destination = destination_directory / name
        temporary = destination.with_name(f".{name}.part")
        source = f"{self._target}:{remote_directory / relative_path}"
        download_argv = self._ssh("cat", "--", str(remote_directory / relative_path))
        last_result = "not downloaded"
        for _ in range(attempts):
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            try:
                if self.config.wsl_distribution is None:
                    self._run(("scp", "-o", "BatchMode=yes", source, str(temporary)))
                else:
                    self._require_success(self.adapter.download(download_argv, temporary))
            except Exception as error:
                last_result = str(error)
                continue
            if not temporary.is_file() or temporary.is_symlink():
                last_result = "missing file"
                continue
            actual_digest = sha256_file(temporary)
            last_result = actual_digest
            if temporary.stat().st_size == size and actual_digest == digest:
                os.replace(temporary, destination)
                return destination
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise RunnerError(
            f"checkpoint transfer digest verification failed after {attempts} attempts "
            f"(last result: {last_result})"
        )

    def prepare(self, manifest: ExperimentManifest) -> PreparedJob:
        """Create and transfer a deterministic, credential-free source bundle."""
        train_argv = build_train_argv(manifest)
        fingerprint = experiment_fingerprint(manifest)
        local_directory = self.config.bundle_root / fingerprint
        remote_directory = self._remote_directory(fingerprint)
        if local_directory.is_symlink():
            raise RunnerError("local fingerprint directory must not be a symlink")
        local_directory.mkdir(parents=True, exist_ok=True)
        if local_directory.resolve().parent != self.config.bundle_root:
            raise RunnerError("local fingerprint directory escaped the configured bundle root")

        archive_path = local_directory / "source.tar"
        commit, tree, archive_sha256 = _snapshot_source(
            self.config.repository,
            archive_path,
        )
        captured_code_digest = hashlib.sha256(commit.encode("ascii")).hexdigest()
        if manifest.code_digest != captured_code_digest:
            raise RunnerError("manifest code digest does not match the captured source commit")

        resume_record: dict[str, object] | None = None
        staged_checkpoint: Path | None = None
        if manifest.agent_config.get("resume") is True:
            source_fingerprint = manifest.agent_config.get("resume_source_fingerprint")
            expected_digest = manifest.agent_config.get("resume_checkpoint_sha256")
            load_run = manifest.agent_config.get("load_run")
            load_checkpoint = manifest.agent_config.get("load_checkpoint")
            if (
                not isinstance(source_fingerprint, str)
                or not re.fullmatch(r"[0-9a-f]{64}", source_fingerprint)
                or source_fingerprint == fingerprint
            ):
                raise RunnerError("resume source fingerprint must identify a distinct prepared job")
            if not isinstance(expected_digest, str) or not re.fullmatch(
                r"[0-9a-f]{64}", expected_digest
            ):
                raise RunnerError("resume checkpoint digest must be a lowercase SHA-256")
            prior_state = self.status(source_fingerprint)
            checkpoint = self._checkpoint_record(prior_state)
            if checkpoint["name"] != load_checkpoint:
                raise RunnerError("resume checkpoint does not match the prior stable checkpoint")
            if checkpoint["sha256"] != expected_digest:
                raise RunnerError("resume checkpoint digest does not match the prior stable checkpoint")
            relative_path = PurePosixPath(str(checkpoint["relative_path"]))
            if len(relative_path.parts) < 5 or relative_path.parts[-2] != load_run:
                raise RunnerError("resume run does not match the prior checkpoint path")
            downloaded = self._download_checkpoint(
                source_fingerprint,
                checkpoint,
                attempts=3,
            )
            staged_checkpoint = local_directory / "resume-checkpoint.pt"
            temporary_checkpoint = local_directory / ".resume-checkpoint.pt.tmp"
            try:
                temporary_checkpoint.unlink()
            except FileNotFoundError:
                pass
            shutil.copyfile(downloaded, temporary_checkpoint)
            if sha256_file(temporary_checkpoint) != expected_digest:
                temporary_checkpoint.unlink()
                raise RunnerError("resume checkpoint digest changed while staging")
            os.replace(temporary_checkpoint, staged_checkpoint)
            resume_record = {
                "checkpoint": load_checkpoint,
                "sha256": expected_digest,
                "size": checkpoint["size"],
                "source_fingerprint": source_fingerprint,
                "target_relative_path": relative_path.as_posix(),
            }

        digest = hashlib.sha256(canonical_json(list(train_argv)).encode("utf-8")).hexdigest()
        prepared_manifest: dict[str, object] = {
            "command": {"argv_file": "train-argv.json", "sha256": digest},
            "fingerprint": fingerprint,
            "learning_inputs": _credential_free(learning_inputs(manifest)),
            "source": {
                "commit": commit,
                "tree": tree,
                "archive": archive_path.name,
                "archive_sha256": archive_sha256,
            },
        }
        if resume_record is not None:
            prepared_manifest["resume"] = resume_record
        manifest_path = local_directory / "prepared-manifest.json"
        argv_path = local_directory / "train-argv.json"
        status_path = local_directory / "status.json"
        atomic_write_json(manifest_path, prepared_manifest)
        atomic_write_json(argv_path, list(train_argv))
        atomic_write_json(
            status_path,
            {
                "artifact_status": "pending",
                "command_digest": digest,
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

        sources = [archive_path, manifest_path, argv_path, status_path, wrapper_path]
        if staged_checkpoint is not None:
            sources.append(staged_checkpoint)
        file_records = {
            source.name: {
                "sha256": sha256_file(source),
                "size": source.stat().st_size,
            }
            for source in sources
        }
        bundle_record: dict[str, object] = {
            "files": file_records,
            "fingerprint": fingerprint,
        }
        bundle_sha256 = hashlib.sha256(
            canonical_json(bundle_record).encode("utf-8")
        ).hexdigest()
        bundle_manifest_path = local_directory / "bundle-manifest.json"
        atomic_write_json(
            bundle_manifest_path,
            {**bundle_record, "bundle_sha256": bundle_sha256},
        )
        sources.append(bundle_manifest_path)
        staging_directory = remote_directory / f".incoming-{bundle_sha256}"
        mkdir_argv = self._ssh("mkdir", "--", str(remote_directory))
        staging_mkdir_argv = self._ssh("mkdir", "--", str(staging_directory))
        transfer_argvs = self._upload_argvs(sources, staging_directory)
        finalize_argv = self._ssh(
            "python3",
            str(staging_directory / "next_rl_remote_job.py"),
            str(remote_directory),
        )
        last_error: BaseException | None = None
        complete = False
        try:
            claim = self.adapter.run(mkdir_argv)
        except Exception as error:
            if self._existing_preparation_matches(remote_directory, bundle_sha256):
                complete = True
            else:
                raise RunnerError("remote fingerprint claim was ambiguous") from error
        else:
            if claim.returncode:
                complete = self._existing_preparation_matches(
                    remote_directory, bundle_sha256
                )
            else:
                self._require_remote_directory(remote_directory, "fingerprint job")
        for _ in range(3) if not complete else ():
            try:
                stage_claim = self.adapter.run(staging_mkdir_argv)
                if stage_claim.returncode:
                    self._reset_staging_directory(remote_directory, staging_directory)
                    self._run(staging_mkdir_argv)
                self._upload_files(sources, staging_directory)
            except Exception as error:
                last_error = error
                self._reset_staging_directory(remote_directory, staging_directory)
                continue
            try:
                self._run(finalize_argv)
            except Exception as error:
                last_error = error
                if self._existing_preparation_matches(remote_directory, bundle_sha256):
                    complete = True
                    break
                self._reset_staging_directory(remote_directory, staging_directory)
                continue
            complete = True
            break
        if not complete:
            raise RunnerError(
                f"remote bundle preparation failed after 3 attempts: {last_error}"
            ) from last_error
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
            dry_run_argv=(
                mkdir_argv,
                staging_mkdir_argv,
                *transfer_argvs,
                finalize_argv,
            ),
        )

    def _invoke_wrapper(
        self,
        remote_directory: PurePosixPath,
        *,
        read_only: bool = False,
    ) -> CommandResult:
        prefix = ("env", "NEXT_RL_REMOTE_INSPECT=1") if read_only else ()
        return self._run(
            self._ssh(
                *prefix,
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

    def _sync_experiment_status(self, fingerprint: str, state: Mapping[str, object]) -> None:
        store = self.experiment_store
        if store is None:
            return
        remote_status = state.get("status")
        if remote_status not in {"pending", "running", "succeeded", "failed"}:
            raise RunnerError("remote lifecycle state is invalid")
        local_status = store.status(fingerprint).get("status")
        if remote_status == "pending" or local_status == remote_status:
            return
        if local_status == "pending" and remote_status in {"succeeded", "failed"}:
            store.update_status(fingerprint, "running")
            local_status = "running"
        if (local_status, remote_status) in {
            ("pending", "running"),
            ("running", "succeeded"),
            ("running", "failed"),
        }:
            store.update_status(fingerprint, str(remote_status))
            return
        raise RunnerError(
            f"remote lifecycle {remote_status!r} conflicts with local experiment {local_status!r}"
        )

    def start(self, prepared: PreparedJob, *, owner: str | None = None) -> dict[str, object]:
        """Request detached execution; a disconnect never triggers cancellation."""
        store = self.experiment_store
        if store is None:
            raise RunnerError("an experiment store is required to reserve a start")
        expected = self._remote_directory(prepared.fingerprint)
        if prepared.remote_directory != expected:
            raise RunnerError("prepared job remote directory is outside its fingerprint path")
        reservation_owner = owner or uuid.uuid4().hex
        store.reserve(prepared.fingerprint, owner=reservation_owner)
        request_path = prepared.local_directory / "start-request.json"
        atomic_write_json(
            request_path,
            {"action": "start", "request_id": f"start-{prepared.fingerprint}"},
        )
        try:
            self._upload_files((request_path,), expected)
        except BaseException:
            try:
                store.release(prepared.fingerprint, owner=reservation_owner)
            except (DuplicateExperimentError, FileNotFoundError):
                pass
            raise
        state = self._state(self._invoke_wrapper(expected))
        self._sync_experiment_status(prepared.fingerprint, state)
        if state.get("status") == "pending":
            store.confirm(prepared.fingerprint, owner=reservation_owner)
        return state

    def status(self, fingerprint: str) -> dict[str, object]:
        """Return remote state plus a live process identity when applicable."""
        remote_directory = self._remote_directory(fingerprint)
        state = self._state(self._invoke_wrapper(remote_directory, read_only=True))
        self._sync_experiment_status(fingerprint, state)
        return state

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
        self._upload_files((request_path,), remote_directory)
        return self._state(self._invoke_wrapper(remote_directory))

    def sync_checkpoint(self, fingerprint: str, *, attempts: int = 3) -> Path:
        """Download one stable checkpoint and publish it only after digest verification."""
        attempts = _positive_integer(attempts, "checkpoint transfer attempts")
        state = self.status(fingerprint)
        checkpoint = self._checkpoint_record(state)
        return self._download_checkpoint(fingerprint, checkpoint, attempts=attempts)

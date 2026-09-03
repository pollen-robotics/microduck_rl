"""Behavioral contracts for safe Nitro training orchestration."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tarfile
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from mjlab_microduck.next_rl.artifacts import canonical_json, sha256_file
from mjlab_microduck.next_rl.experiments import experiment_fingerprint
from mjlab_microduck.next_rl.runner import (
    CommandResult,
    NitroConfig,
    NitroRunner,
    RunnerError,
    build_train_argv,
)
from mjlab_microduck.next_rl.schema import ExperimentManifest, PolicyContract


def job(
    *,
    task_id: str = "Mjlab-Velocity-Flat-MicroDuck",
    run_name: str = "hello-a1b2c3",
) -> ExperimentManifest:
    return ExperimentManifest.from_dict(
        {
            "skill_id": "velocity",
            "spec_version": "1.0.0",
            "task_id": task_id,
            "contract": PolicyContract.microduck().as_dict(),
            "code_digest": "a" * 64,
            "seed": 42,
            "runner_id": "nitro",
            "status": "planned",
            "environment_config": {"scene": {"num_envs": 1024}},
            "agent_config": {"max_iterations": 1000, "run_name": run_name},
        }
    )


def resume_job(
    *,
    run: str,
    checkpoint: str,
    additional_iterations: int,
    source_fingerprint: str | None = None,
    checkpoint_sha256: str | None = None,
) -> ExperimentManifest:
    raw = job().as_dict()
    raw["agent_config"] = {
        "run_name": "hello-a1b2c3",
        "resume": True,
        "load_run": run,
        "load_checkpoint": checkpoint,
        "additional_iterations": additional_iterations,
    }
    if source_fingerprint is not None:
        raw["agent_config"]["resume_source_fingerprint"] = source_fingerprint
    if checkpoint_sha256 is not None:
        raw["agent_config"]["resume_checkpoint_sha256"] = checkpoint_sha256
    return ExperimentManifest.from_dict(raw)


def test_train_command_uses_real_tyro_flags():
    assert build_train_argv(job()) == (
        "uv",
        "run",
        "train",
        "Mjlab-Velocity-Flat-MicroDuck",
        "--env.scene.num-envs",
        "1024",
        "--agent.max-iterations",
        "1000",
        "--agent.seed",
        "42",
        "--agent.run-name",
        "hello-a1b2c3",
    )


def test_resume_uses_exact_run_and_checkpoint_and_additional_iterations():
    argv = build_train_argv(
        resume_job(
            run="2026-09-02_hello",
            checkpoint="model_250.pt",
            additional_iterations=500,
        )
    )

    assert argv[-8:] == (
        "--agent.resume",
        "True",
        "--agent.load-run",
        "2026-09-02_hello",
        "--agent.load-checkpoint",
        "model_250.pt",
        "--agent.max-iterations",
        "500",
    )
    assert argv.count("--agent.max-iterations") == 1


def test_command_is_argv_not_shell_text():
    argv = build_train_argv(job())

    assert isinstance(argv, tuple)
    assert all(isinstance(part, str) for part in argv)


@pytest.mark.parametrize(
    "task_id",
    (
        "--help",
        "../Mjlab-Velocity",
        "Mjlab/Velocity",
        "Mjlab Velocity",
        "Mjlab;reboot",
        "Mjlab\nVelocity",
    ),
)
def test_task_id_rejects_shell_and_path_syntax(task_id: str):
    with pytest.raises(RunnerError, match="task ID"):
        build_train_argv(job(task_id=task_id))


@pytest.mark.parametrize(
    "run_name",
    ("--help", "../run", "run/name", "run name", "run;reboot", "run\tname"),
)
def test_run_name_rejects_shell_and_path_syntax(run_name: str):
    with pytest.raises(RunnerError, match="run name"):
        build_train_argv(job(run_name=run_name))


@pytest.mark.parametrize(
    ("run", "checkpoint"),
    (
        ("../run", "model_250.pt"),
        ("2026-09-02_hello", "../model_250.pt"),
        ("2026-09-02_hello", "model_latest.pt"),
        ("run;reboot", "model_250.pt"),
    ),
)
def test_resume_identifiers_use_narrow_allowlists(run: str, checkpoint: str):
    with pytest.raises(RunnerError, match="resume"):
        build_train_argv(
            resume_job(run=run, checkpoint=checkpoint, additional_iterations=500)
        )


@dataclass
class FakeAdapter:
    outputs: list[CommandResult | BaseException] = field(default_factory=list)
    calls: list[tuple[str, ...]] = field(default_factory=list)
    cancel_calls: list[tuple[str, ...]] = field(default_factory=list)

    def run(self, argv: tuple[str, ...]) -> CommandResult:
        self.calls.append(tuple(argv))
        if self.outputs:
            result = self.outputs.pop(0)
            if isinstance(result, BaseException):
                raise result
            return result
        return CommandResult()


def _git(repository: Path, *args: str) -> str:
    return subprocess.run(
        ("git", *args),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "tests@example.invalid")
    _git(root, "config", "user.name", "Test Runner")
    (root / "src").mkdir()
    (root / "src" / "module.py").write_text("VERSION = 'committed'\n")
    (root / "scripts").mkdir()
    (root / "scripts" / "next_rl_remote_job.py").write_text("# wrapper\n")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "fixture")
    return root


def config(repository: Path, tmp_path: Path) -> NitroConfig:
    return NitroConfig(
        ssh_alias="nitro",
        ssh_user="aif_eng",
        repository=repository,
        bundle_root=tmp_path / "bundles",
    )


def test_ssh_enforces_noninteractive_host_checked_access(repository: Path, tmp_path: Path):
    adapter = FakeAdapter(outputs=[CommandResult(stdout='{"status":"running"}')])

    NitroRunner(config(repository, tmp_path), adapter).status("abc123")

    argv = adapter.calls[0]
    assert argv[:3] == ("ssh", "-o", "BatchMode=yes")
    assert "StrictHostKeyChecking=no" not in argv
    assert "aif_eng@nitro" in argv


def test_prepare_archives_only_the_pinned_tracked_tree(repository: Path, tmp_path: Path):
    adapter = FakeAdapter()
    runner = NitroRunner(config(repository, tmp_path), adapter)
    (repository / "src" / "module.py").write_text("VERSION = 'worktree'\n")
    (repository / "untracked.txt").write_text("not in archive\n")

    prepared = runner.prepare(job())

    with tarfile.open(prepared.archive_path, "r") as archive:
        assert archive.getnames() == ["scripts/next_rl_remote_job.py", "src/module.py"]
        source = archive.extractfile("src/module.py")
        assert source is not None
        assert source.read() == b"VERSION = 'committed'\n"
    assert prepared.source_commit == _git(repository, "rev-parse", "HEAD")
    assert prepared.source_tree == _git(repository, "rev-parse", "HEAD^{tree}")
    assert prepared.archive_sha256 == sha256_file(prepared.archive_path)
    assert prepared.manifest["source"]["archive_sha256"] == prepared.archive_sha256
    assert prepared.manifest["command"] == {
        "argv_file": "train-argv.json",
        "sha256": hashlib.sha256(
            canonical_json(list(prepared.train_argv)).encode("utf-8")
        ).hexdigest(),
    }
    assert "untracked.txt" not in prepared.archive_path.read_bytes().decode(
        "utf-8", errors="ignore"
    )


def test_prepare_wrapper_comes_from_the_same_pinned_tree(repository: Path, tmp_path: Path):
    wrapper = repository / "scripts" / "next_rl_remote_job.py"
    wrapper.write_text("# mutable worktree wrapper\n")

    prepared = NitroRunner(config(repository, tmp_path), FakeAdapter()).prepare(job())

    assert (prepared.local_directory / "next_rl_remote_job.py").read_text() == "# wrapper\n"


def test_prepare_archive_is_deterministic(repository: Path, tmp_path: Path):
    runner = NitroRunner(config(repository, tmp_path), FakeAdapter())

    first = runner.prepare(job()).archive_path.read_bytes()
    second = runner.prepare(job()).archive_path.read_bytes()

    assert second == first


def test_prepare_rejects_local_fingerprint_directory_symlink(
    repository: Path,
    tmp_path: Path,
):
    runner_config = config(repository, tmp_path)
    fingerprint = experiment_fingerprint(job())
    outside = tmp_path / "outside"
    outside.mkdir()
    runner_config.bundle_root.mkdir()
    (runner_config.bundle_root / fingerprint).symlink_to(outside, target_is_directory=True)

    with pytest.raises(RunnerError, match="symlink"):
        NitroRunner(runner_config, FakeAdapter()).prepare(job())

    assert list(outside.iterdir()) == []


def test_prepare_transfers_only_into_fingerprint_directory(repository: Path, tmp_path: Path):
    adapter = FakeAdapter()

    prepared = NitroRunner(config(repository, tmp_path), adapter).prepare(job())

    expected_prefix = f"aif_eng@nitro:{prepared.remote_directory}/"
    scp_calls = [call for call in adapter.calls if call[0] == "scp"]
    assert scp_calls
    assert all(call[-1].startswith(expected_prefix) for call in scp_calls)
    assert str(prepared.remote_directory).startswith(
        "/home/aif_eng/microduck-training/runs/"
    )


def test_manifest_and_dry_run_contain_no_credentials(repository: Path, tmp_path: Path):
    raw = job().as_dict()
    raw["environment_config"]["password"] = "environment-secret"
    raw["agent_config"]["private_key"] = "private-key-secret"
    prepared = NitroRunner(config(repository, tmp_path), FakeAdapter()).prepare(
        ExperimentManifest.from_dict(raw)
    )

    serialized = canonical_json(prepared.manifest).lower()
    dry_run = " ".join(part for call in prepared.dry_run_argv for part in call).lower()

    assert "password" not in serialized
    assert "private_key" not in serialized
    assert "environment-secret" not in serialized
    assert "private-key-secret" not in serialized
    assert "environment-secret" not in dry_run
    assert "private-key-secret" not in dry_run


def test_prepare_rejects_tracked_symlink(repository: Path, tmp_path: Path):
    (repository / "linked.py").symlink_to("src/module.py")
    _git(repository, "add", "linked.py")
    _git(repository, "commit", "-qm", "track symlink")

    with pytest.raises(RunnerError, match="regular file"):
        NitroRunner(config(repository, tmp_path), FakeAdapter()).prepare(job())


@pytest.mark.parametrize(
    "secret_path",
    (
        ".env",
        ".git-credentials",
        ".netrc",
        "keys/deploy.pem",
        "config/api-key.yaml",
        "config/password.txt",
        "config/private-key-backup.json",
        "private_key",
        "wandb_api_key.txt",
        "config/wandb_token_value.txt",
        "keys/github_token_backup",
        "auth/oauth2_token_file.json",
    ),
)
def test_prepare_rejects_secret_like_tracked_paths(
    repository: Path,
    tmp_path: Path,
    secret_path: str,
):
    path = repository / secret_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("secret\n")
    _git(repository, "add", secret_path)
    _git(repository, "commit", "-qm", "track unsafe secret")

    with pytest.raises(RunnerError, match="secret-like"):
        NitroRunner(config(repository, tmp_path), FakeAdapter()).prepare(job())


def test_secret_path_filter_does_not_reject_unrelated_words(repository: Path, tmp_path: Path):
    for name in ("tokenizer.py", "passwordless.py", "monkey.py"):
        (repository / "src" / name).write_text("SAFE = True\n")
    _git(repository, "add", "src")
    _git(repository, "commit", "-qm", "track safe names")

    prepared = NitroRunner(config(repository, tmp_path), FakeAdapter()).prepare(job())

    with tarfile.open(prepared.archive_path, "r") as archive:
        assert all(f"src/{name}" in archive.getnames() for name in ("tokenizer.py", "passwordless.py", "monkey.py"))


def test_prepared_manifest_recursively_removes_credential_like_keys(
    repository: Path,
    tmp_path: Path,
):
    raw = job().as_dict()
    raw["environment_config"]["telemetry"] = {
        "wandb_api_key": "wandb-secret",
        "wandb_token_value": "wandb-token-secret",
        "github_token_backup": "github-token-secret",
        "tokenizer": "kept-learning-setting",
        "tokenization_budget": 128,
    }
    raw["agent_config"]["nested"] = {
        "private-key-backup": "private-secret",
        "oauth2_token_file": "oauth-token-secret",
        "monkey": "kept-agent-setting",
    }

    prepared = NitroRunner(config(repository, tmp_path), FakeAdapter()).prepare(
        ExperimentManifest.from_dict(raw)
    )
    serialized = canonical_json(prepared.manifest)

    assert "wandb_api_key" not in serialized
    assert "wandb-secret" not in serialized
    assert "private-key-backup" not in serialized
    assert "private-secret" not in serialized
    assert "wandb-token-secret" not in serialized
    assert "github-token-secret" not in serialized
    assert "oauth-token-secret" not in serialized
    assert "tokenizer" in serialized
    assert '"tokenization_budget":128' in serialized
    assert "monkey" in serialized


def test_prepare_rejects_control_characters_in_tracked_paths(repository: Path, tmp_path: Path):
    unsafe = repository / "line\nbreak.py"
    unsafe.write_text("unsafe name\n")
    _git(repository, "add", "line\nbreak.py")
    _git(repository, "commit", "-qm", "track unsafe path")

    with pytest.raises(RunnerError, match="control"):
        NitroRunner(config(repository, tmp_path), FakeAdapter()).prepare(job())


def test_disconnect_does_not_cancel_remote_process(repository: Path, tmp_path: Path):
    adapter = FakeAdapter()
    runner = NitroRunner(config(repository, tmp_path), adapter)
    prepared = runner.prepare(job())
    adapter.outputs.append(ConnectionError("ssh disconnected"))

    with pytest.raises(ConnectionError, match="disconnected"):
        runner.start(prepared)

    assert adapter.cancel_calls == []


class PrepareAdapter(FakeAdapter):
    def __init__(
        self,
        bundle_root: Path,
        *,
        fail_stage_mkdir: bool = False,
        fail_upload: bool = False,
        existing: str | None = None,
    ):
        super().__init__()
        self.bundle_root = bundle_root
        self.fail_stage_mkdir = fail_stage_mkdir
        self.fail_upload = fail_upload
        self.existing = existing

    def _bundle_digest(self) -> str:
        manifest = next(self.bundle_root.glob("*/bundle-manifest.json"))
        return json.loads(manifest.read_text())["bundle_sha256"]

    def run(self, argv: tuple[str, ...]) -> CommandResult:
        self.calls.append(tuple(argv))
        joined = " ".join(argv)
        if argv[0] == "scp" and self.fail_upload:
            self.fail_upload = False
            raise ConnectionError("partial scp disconnect")
        if argv[0] == "ssh" and "mkdir --" in joined and "/.incoming-" in joined:
            if self.fail_stage_mkdir:
                self.fail_stage_mkdir = False
                raise ConnectionError("disconnect before scp")
            return CommandResult()
        if argv[0] == "ssh" and "mkdir --" in joined and "/.incoming-" not in joined:
            if self.existing is not None:
                return CommandResult(returncode=1, stderr="already exists")
            return CommandResult()
        if argv[0] == "ssh" and "test ! -L" in joined and self.existing == "symlink":
            return CommandResult(returncode=1, stderr="symlink")
        if argv[0] == "ssh" and "python3" in argv and "/.incoming-" not in joined:
            if self.existing == "matching":
                return CommandResult(
                    stdout=json.dumps({"prepared_bundle_sha256": self._bundle_digest()})
                )
            if self.existing == "conflicting":
                return CommandResult(stdout=json.dumps({"prepared_bundle_sha256": "f" * 64}))
            return CommandResult(returncode=2, stderr="incomplete")
        if argv[0] == "ssh" and "python3" in argv and "/.incoming-" in joined:
            return CommandResult(
                stdout=json.dumps({"prepared_bundle_sha256": self._bundle_digest()})
            )
        return CommandResult()


@pytest.mark.parametrize("failure", ("before_copy", "partial_copy"))
def test_prepare_cleans_only_incomplete_fingerprint_and_retries_transfer(
    repository: Path,
    tmp_path: Path,
    failure: str,
):
    runner_config = config(repository, tmp_path)
    adapter = PrepareAdapter(
        runner_config.bundle_root,
        fail_stage_mkdir=failure == "before_copy",
        fail_upload=failure == "partial_copy",
    )

    prepared = NitroRunner(runner_config, adapter).prepare(job())

    uploads = [call for call in adapter.calls if call[0] == "scp"]
    assert len(uploads) == (1 if failure == "before_copy" else 2)
    assert all(
        call[-1].startswith(f"aif_eng@nitro:{prepared.remote_directory}/.incoming-")
        for call in uploads
    )
    removals = [call for call in adapter.calls if call[0] == "ssh" and "rm" in call]
    assert removals
    assert all(call[-1] == str(prepared.remote_directory) for call in removals)
    assert not any(str(prepared.remote_directory.parent) == part for call in removals for part in call)


def test_prepare_reuses_only_matching_complete_job(repository: Path, tmp_path: Path):
    runner_config = config(repository, tmp_path)
    adapter = PrepareAdapter(runner_config.bundle_root, existing="matching")

    prepared = NitroRunner(runner_config, adapter).prepare(job())

    assert prepared.fingerprint
    assert not any(call[0] == "scp" for call in adapter.calls)
    assert not any("rm" in call for call in adapter.calls)


def test_prepare_rejects_conflicting_complete_job(repository: Path, tmp_path: Path):
    runner_config = config(repository, tmp_path)
    adapter = PrepareAdapter(runner_config.bundle_root, existing="conflicting")

    with pytest.raises(RunnerError, match="conflicting"):
        NitroRunner(runner_config, adapter).prepare(job())

    assert not any(call[0] == "scp" for call in adapter.calls)
    assert not any("rm" in call for call in adapter.calls)


def test_prepare_rejects_remote_job_directory_symlink(repository: Path, tmp_path: Path):
    runner_config = config(repository, tmp_path)
    adapter = PrepareAdapter(runner_config.bundle_root, existing="symlink")

    with pytest.raises(RunnerError, match="symlink"):
        NitroRunner(runner_config, adapter).prepare(job())

    assert not any(call[0] == "scp" for call in adapter.calls)
    assert not any("rm" in call for call in adapter.calls)


def test_cancel_refuses_reused_pid(repository: Path, tmp_path: Path):
    state = {
        "status": "running",
        "pid": 123,
        "process_start": "100",
        "command_digest": "a" * 64,
        "live_identity": {
            "pid": 123,
            "process_start": "100",
            "command_digest": "b" * 64,
        },
    }
    adapter = FakeAdapter(outputs=[CommandResult(stdout=json.dumps(state))])

    with pytest.raises(RunnerError, match="identity"):
        NitroRunner(config(repository, tmp_path), adapter).cancel("abc123")

    assert not any(call[0] == "scp" for call in adapter.calls)


class CheckpointAdapter(FakeAdapter):
    def __init__(
        self,
        state: dict[str, object],
        payloads: list[bytes | BaseException | CommandResult],
    ):
        super().__init__()
        self.state = state
        self.payloads = payloads

    def run(self, argv: tuple[str, ...]) -> CommandResult:
        self.calls.append(tuple(argv))
        if argv[0] == "ssh":
            return CommandResult(stdout=json.dumps(self.state))
        destination = Path(argv[-1])
        payload = self.payloads.pop(0)
        if isinstance(payload, BaseException):
            destination.write_bytes(b"partial")
            raise payload
        if isinstance(payload, CommandResult):
            destination.write_bytes(b"partial")
            return payload
        destination.write_bytes(payload)
        return CommandResult()


def test_checkpoint_transfer_retries_until_download_digest_matches(
    repository: Path,
    tmp_path: Path,
):
    payload = b"verified checkpoint"
    checkpoint = {
        "name": "model_1000.pt",
        "relative_path": "source/logs/velocity/model_1000.pt",
        "size": len(payload),
        "mtime_ns": 100,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    state = {"status": "running", "last_stable_checkpoint": checkpoint}
    adapter = CheckpointAdapter(state, [b"truncated", payload])
    runner = NitroRunner(config(repository, tmp_path), adapter)

    local = runner.sync_checkpoint("abc123", attempts=2)

    assert local.read_bytes() == payload
    downloads = [call for call in adapter.calls if call[0] == "scp"]
    assert len(downloads) == 2
    assert not local.with_name(f".{local.name}.part").exists()


def test_checkpoint_transfer_retries_transport_exception_and_cleans_partial(
    repository: Path,
    tmp_path: Path,
):
    payload = b"verified checkpoint"
    checkpoint = {
        "name": "model_1000.pt",
        "relative_path": "source/logs/velocity/run/model_1000.pt",
        "size": len(payload),
        "mtime_ns": 100,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    adapter = CheckpointAdapter(
        {"status": "running", "last_stable_checkpoint": checkpoint},
        [ConnectionError("scp disconnected"), payload],
    )

    local = NitroRunner(config(repository, tmp_path), adapter).sync_checkpoint(
        "abc123", attempts=2
    )

    assert local.read_bytes() == payload
    assert not local.with_name(f".{local.name}.part").exists()


def test_checkpoint_transfer_retries_nonzero_scp_result(
    repository: Path,
    tmp_path: Path,
):
    payload = b"verified checkpoint"
    checkpoint = {
        "name": "model_1000.pt",
        "relative_path": "source/logs/velocity/run/model_1000.pt",
        "size": len(payload),
        "mtime_ns": 100,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    adapter = CheckpointAdapter(
        {"status": "running", "last_stable_checkpoint": checkpoint},
        [CommandResult(returncode=1, stderr="scp failed"), payload],
    )

    local = NitroRunner(config(repository, tmp_path), adapter).sync_checkpoint(
        "abc123", attempts=2
    )

    assert local.read_bytes() == payload
    assert not local.with_name(f".{local.name}.part").exists()


def test_checkpoint_transfer_rejects_remote_shell_syntax_in_parent_path(
    repository: Path,
    tmp_path: Path,
):
    payload = b"checkpoint"
    checkpoint = {
        "name": "model_1000.pt",
        "relative_path": "source/logs/run;touch-pwned/model_1000.pt",
        "size": len(payload),
        "mtime_ns": 100,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    adapter = CheckpointAdapter(
        {"status": "running", "last_stable_checkpoint": checkpoint},
        [payload],
    )

    with pytest.raises(RunnerError, match="path"):
        NitroRunner(config(repository, tmp_path), adapter).sync_checkpoint("abc123")

    assert not any(call[0] == "scp" for call in adapter.calls)


class ResumeAdapter(FakeAdapter):
    def __init__(self, state: dict[str, object], payload: bytes):
        super().__init__()
        self.state = state
        self.payload = payload

    def run(self, argv: tuple[str, ...]) -> CommandResult:
        self.calls.append(tuple(argv))
        if argv[0] == "ssh" and "next_rl_remote_job.py" in " ".join(argv):
            return CommandResult(stdout=json.dumps(self.state))
        if argv[0] == "scp" and not argv[-1].endswith("/"):
            Path(argv[-1]).write_bytes(self.payload)
        return CommandResult()


def test_resume_stages_digest_verified_checkpoint_from_explicit_prior_job(
    repository: Path,
    tmp_path: Path,
):
    payload = b"stable prior checkpoint"
    digest = hashlib.sha256(payload).hexdigest()
    source_fingerprint = "b" * 64
    checkpoint = {
        "name": "model_250.pt",
        "relative_path": "source/logs/velocity/2026-09-02_hello/model_250.pt",
        "size": len(payload),
        "mtime_ns": 100,
        "sha256": digest,
    }
    adapter = ResumeAdapter(
        {"status": "failed", "last_stable_checkpoint": checkpoint},
        payload,
    )
    manifest = resume_job(
        run="2026-09-02_hello",
        checkpoint="model_250.pt",
        additional_iterations=500,
        source_fingerprint=source_fingerprint,
        checkpoint_sha256=digest,
    )

    prepared = NitroRunner(config(repository, tmp_path), adapter).prepare(manifest)

    assert (prepared.local_directory / "resume-checkpoint.pt").read_bytes() == payload
    assert prepared.manifest["resume"] == {
        "checkpoint": "model_250.pt",
        "sha256": digest,
        "size": len(payload),
        "source_fingerprint": source_fingerprint,
        "target_relative_path": "source/logs/velocity/2026-09-02_hello/model_250.pt",
    }
    final_upload = [call for call in adapter.calls if call[0] == "scp"][-1]
    assert str(prepared.local_directory / "resume-checkpoint.pt") in final_upload


@pytest.mark.parametrize("problem", ("missing", "digest", "checkpoint"))
def test_resume_rejects_missing_unstable_or_mismatched_prior_checkpoint(
    repository: Path,
    tmp_path: Path,
    problem: str,
):
    payload = b"stable prior checkpoint"
    digest = hashlib.sha256(payload).hexdigest()
    checkpoint = {
        "name": "model_250.pt" if problem != "checkpoint" else "model_251.pt",
        "relative_path": "source/logs/velocity/2026-09-02_hello/"
        + ("model_250.pt" if problem != "checkpoint" else "model_251.pt"),
        "size": len(payload),
        "mtime_ns": 100,
        "sha256": digest,
    }
    state = {
        "status": "failed",
        "last_stable_checkpoint": None if problem == "missing" else checkpoint,
    }
    expected_digest = "c" * 64 if problem == "digest" else digest
    adapter = ResumeAdapter(state, payload)
    manifest = resume_job(
        run="2026-09-02_hello",
        checkpoint="model_250.pt",
        additional_iterations=500,
        source_fingerprint="b" * 64,
        checkpoint_sha256=expected_digest,
    )

    with pytest.raises(RunnerError, match="resume|stable|digest|checkpoint"):
        NitroRunner(config(repository, tmp_path), adapter).prepare(manifest)

    assert not any(call[0] == "scp" and call[-1].endswith("/") for call in adapter.calls)

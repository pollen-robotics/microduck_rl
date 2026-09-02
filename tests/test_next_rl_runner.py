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
) -> ExperimentManifest:
    raw = job().as_dict()
    raw["agent_config"] = {
        "run_name": "hello-a1b2c3",
        "resume": True,
        "load_run": run,
        "load_checkpoint": checkpoint,
        "additional_iterations": additional_iterations,
    }
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


def test_prepare_transfers_only_into_fingerprint_directory(repository: Path, tmp_path: Path):
    adapter = FakeAdapter()

    prepared = NitroRunner(config(repository, tmp_path), adapter).prepare(job())

    expected = f"aif_eng@nitro:{prepared.remote_directory}/"
    scp_calls = [call for call in adapter.calls if call[0] == "scp"]
    assert scp_calls
    assert all(call[-1] == expected for call in scp_calls)
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
        ".netrc",
        "keys/deploy.pem",
        "config/password.txt",
        "private_key",
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
    def __init__(self, state: dict[str, object], payloads: list[bytes]):
        super().__init__()
        self.state = state
        self.payloads = payloads

    def run(self, argv: tuple[str, ...]) -> CommandResult:
        self.calls.append(tuple(argv))
        if argv[0] == "ssh":
            return CommandResult(stdout=json.dumps(self.state))
        destination = Path(argv[-1])
        destination.write_bytes(self.payloads.pop(0))
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

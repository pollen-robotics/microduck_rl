"""Behavioral contracts for the standalone Nitro remote job wrapper."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest


def load_remote_job_module():
    path = Path(__file__).parents[1] / "scripts" / "next_rl_remote_job.py"
    spec = importlib.util.spec_from_file_location("next_rl_remote_job", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def remote_job():
    return load_remote_job_module()


@pytest.fixture
def job_directory(tmp_path: Path) -> Path:
    root = tmp_path / "runs"
    job = root / "abc123"
    job.mkdir(parents=True)
    (job / "train-argv.json").write_text(
        json.dumps(
            [
                "uv",
                "run",
                "train",
                "Mjlab-Velocity-Flat-MicroDuck",
                "--env.scene.num-envs",
                "64",
                "--agent.max-iterations",
                "5",
            ]
        )
    )
    return job


def test_job_directory_must_be_a_fingerprint_beneath_root(remote_job, job_directory: Path):
    root = job_directory.parent

    assert remote_job.validate_job_directory(job_directory, root=root) == job_directory
    with pytest.raises(remote_job.RemoteJobError, match="beneath"):
        remote_job.validate_job_directory(root.parent / "elsewhere", root=root)
    with pytest.raises(remote_job.RemoteJobError, match="fingerprint"):
        remote_job.validate_job_directory(root / "not-a-fingerprint!", root=root)


def test_job_directory_rejects_symlink_escape(remote_job, job_directory: Path, tmp_path: Path):
    outside = tmp_path / "outside" / "abc123"
    outside.mkdir(parents=True)
    link = job_directory.parent / "def456"
    link.symlink_to(outside, target_is_directory=True)

    with pytest.raises(remote_job.RemoteJobError, match="beneath"):
        remote_job.validate_job_directory(link, root=job_directory.parent)


def test_train_argv_is_a_validated_json_array(remote_job, job_directory: Path):
    assert remote_job.load_train_argv(job_directory)[:4] == (
        "uv",
        "run",
        "train",
        "Mjlab-Velocity-Flat-MicroDuck",
    )

    (job_directory / "train-argv.json").write_text(json.dumps("uv run train"))
    with pytest.raises(remote_job.RemoteJobError, match="JSON array"):
        remote_job.load_train_argv(job_directory)


class FakeProcess:
    pid = 123

    def poll(self):
        return None


def test_training_launch_never_uses_a_shell_and_creates_a_process_group(
    remote_job,
    job_directory: Path,
):
    calls = []

    def popen(argv, **kwargs):
        calls.append((argv, kwargs))
        return FakeProcess()

    with open(os.devnull, "wb") as output:
        process = remote_job.launch_training(
            job_directory,
            cwd=job_directory,
            stdout=output,
            popen=popen,
        )

    assert process.pid == 123
    assert calls[0][0] == remote_job.load_train_argv(job_directory)
    assert calls[0][1]["shell"] is False
    assert calls[0][1]["start_new_session"] is True


def test_latest_checkpoint_is_selected_numerically(remote_job, job_directory: Path):
    older = job_directory / "model_999.pt"
    newer = job_directory / "model_1000.pt"
    older.write_bytes(b"older")
    newer.write_bytes(b"newer")

    assert remote_job.select_latest_checkpoint([newer, older]) == newer


def test_checkpoint_requires_two_unchanged_size_and_mtime_probes(
    remote_job,
    job_directory: Path,
):
    checkpoint = job_directory / "model_1000.pt"
    checkpoint.write_bytes(b"first")
    tracker = remote_job.StableCheckpointTracker()

    assert tracker.observe([checkpoint]) is None
    checkpoint.write_bytes(b"changed-length")
    assert tracker.observe([checkpoint]) is None
    stable = tracker.observe([checkpoint])

    assert stable["name"] == "model_1000.pt"
    assert stable["size"] == len(b"changed-length")
    assert len(stable["sha256"]) == 64


def test_checkpoint_changed_while_hashing_is_not_declared_stable(
    remote_job,
    job_directory: Path,
    monkeypatch,
):
    checkpoint = job_directory / "model_1000.pt"
    checkpoint.write_bytes(b"first")
    tracker = remote_job.StableCheckpointTracker()
    assert tracker.observe([checkpoint]) is None
    real_hash = remote_job.sha256_file

    def mutate_while_hashing(path):
        digest = real_hash(path)
        checkpoint.write_bytes(b"other")
        return digest

    monkeypatch.setattr(remote_job, "sha256_file", mutate_while_hashing)

    assert tracker.observe([checkpoint]) is None


def test_cancel_refuses_reused_pid_identity(remote_job, job_directory: Path, monkeypatch):
    expected = {
        "pid": 123,
        "process_start": "100",
        "command_digest": "a" * 64,
    }
    remote_job.atomic_write_json(
        job_directory / "status.json",
        {"status": "running", **expected},
    )
    remote_job.atomic_write_json(
        job_directory / "cancel-request.json",
        {"action": "cancel", **expected},
    )
    monkeypatch.setattr(
        remote_job,
        "live_process_identity",
        lambda pid: remote_job.ProcessIdentity(pid, "100", "b" * 64),
    )
    killed = []

    with pytest.raises(remote_job.RemoteJobError, match="identity"):
        remote_job.cancel_job(job_directory, killpg=lambda *args: killed.append(args))

    assert killed == []


def test_cancel_targets_only_the_verified_new_process_group(
    remote_job,
    job_directory: Path,
    monkeypatch,
):
    expected = {
        "pid": 123,
        "process_start": "100",
        "command_digest": "a" * 64,
    }
    remote_job.atomic_write_json(
        job_directory / "status.json",
        {"status": "running", **expected, "artifact_status": "pending"},
    )
    remote_job.atomic_write_json(
        job_directory / "cancel-request.json",
        {"action": "cancel", **expected},
    )
    monkeypatch.setattr(
        remote_job,
        "live_process_identity",
        lambda pid: remote_job.ProcessIdentity(pid, "100", "a" * 64),
    )
    monkeypatch.setattr(remote_job.os, "getpgid", lambda pid: pid)
    killed = []

    result = remote_job.cancel_job(job_directory, killpg=lambda *args: killed.append(args))

    assert killed == [(123, remote_job.signal.SIGTERM)]
    assert result["status"] == "failed"
    assert result["cancelled"] is True


def test_cli_accepts_exactly_one_job_directory_argument(remote_job):
    with pytest.raises(SystemExit, match="usage"):
        remote_job.main([])
    with pytest.raises(SystemExit, match="usage"):
        remote_job.main(["abc123", "extra"])


def test_failed_supervisor_setup_records_a_complete_failed_state(
    remote_job,
    job_directory: Path,
    monkeypatch,
):
    initial = {
        "artifact_status": "pending",
        "command_digest": "a" * 64,
        "exit_code": None,
        "last_stable_checkpoint": None,
        "pid": None,
        "process_start": None,
        "status": "pending",
        "stdout_path": str(job_directory / "stdout.log"),
    }
    remote_job.atomic_write_json(job_directory / "status.json", initial)
    monkeypatch.setattr(
        remote_job,
        "supervise",
        lambda job: (_ for _ in ()).throw(remote_job.RemoteJobError("bad archive")),
    )

    with pytest.raises(remote_job.RemoteJobError, match="bad archive"):
        remote_job.run_supervisor(job_directory)

    state = json.loads((job_directory / "status.json").read_text())
    assert state == {**initial, "error": "bad archive", "exit_code": 1, "status": "failed"}


def test_failed_detached_spawn_keeps_start_request_retryable(
    remote_job,
    job_directory: Path,
):
    remote_job.atomic_write_json(
        job_directory / "status.json",
        {"status": "pending"},
    )
    remote_job.atomic_write_json(
        job_directory / "start-request.json",
        {"action": "start"},
    )

    def fail_spawn(*args, **kwargs):
        raise OSError("cannot spawn")

    with pytest.raises(OSError, match="cannot spawn"):
        remote_job._start_supervisor(job_directory, popen=fail_spawn)

    assert (job_directory / "start-request.json").exists()

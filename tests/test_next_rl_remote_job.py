"""Behavioral contracts for the standalone Nitro remote job wrapper."""

from __future__ import annotations

import importlib.util
import hashlib
import io
import json
import os
import threading
import tarfile
from concurrent.futures import ThreadPoolExecutor
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
    with pytest.raises(remote_job.RemoteJobError, match="symlink|beneath"):
        remote_job.validate_job_directory(root.parent / "elsewhere", root=root)
    with pytest.raises(remote_job.RemoteJobError, match="fingerprint"):
        remote_job.validate_job_directory(root / "not-a-fingerprint!", root=root)


def test_job_directory_rejects_symlink_escape(remote_job, job_directory: Path, tmp_path: Path):
    outside = tmp_path / "outside" / "abc123"
    outside.mkdir(parents=True)
    link = job_directory.parent / "def456"
    link.symlink_to(outside, target_is_directory=True)

    with pytest.raises(remote_job.RemoteJobError, match="symlink|beneath"):
        remote_job.validate_job_directory(link, root=job_directory.parent)


def test_job_directory_rejects_symlink_even_when_target_stays_under_root(
    remote_job,
    job_directory: Path,
):
    target = job_directory.parent / "def456"
    target.mkdir()
    link = job_directory.parent / "abc789"
    link.symlink_to(target, target_is_directory=True)

    with pytest.raises(remote_job.RemoteJobError, match="symlink"):
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


def test_train_argv_digest_must_match_prepared_manifest_before_spawn(
    remote_job,
    job_directory: Path,
):
    argv = remote_job.load_train_argv(job_directory)
    remote_job.atomic_write_json(
        job_directory / "prepared-manifest.json",
        {
            "command": {
                "argv_file": "train-argv.json",
                "sha256": hashlib.sha256(
                    remote_job.canonical_json(list(argv)).encode("utf-8")
                ).hexdigest(),
            }
        },
    )
    assert remote_job.verified_train_argv(job_directory) == argv

    raw = list(argv)
    raw[-1] = "5000"
    remote_job.atomic_write_json(job_directory / "train-argv.json", raw)
    with pytest.raises(remote_job.RemoteJobError, match="digest"):
        remote_job.verified_train_argv(job_directory)


def _prepared_source_archive(remote_job, job_directory: Path) -> str:
    archive_path = job_directory / "source.tar"
    content = b"source\n"
    with tarfile.open(archive_path, "w") as archive:
        member = tarfile.TarInfo("README.md")
        member.size = len(content)
        archive.addfile(member, io.BytesIO(content))
    return remote_job.sha256_file(archive_path)


def test_verified_source_links_resume_checkpoint_at_mjlab_run_path(
    remote_job,
    job_directory: Path,
):
    archive_digest = _prepared_source_archive(remote_job, job_directory)
    payload = b"prior checkpoint"
    checkpoint_digest = hashlib.sha256(payload).hexdigest()
    (job_directory / "resume-checkpoint.pt").write_bytes(payload)
    target = "source/logs/velocity/2026-09-02_hello/model_250.pt"
    remote_job.atomic_write_json(
        job_directory / "prepared-manifest.json",
        {
            "source": {"archive_sha256": archive_digest},
            "resume": {
                "checkpoint": "model_250.pt",
                "sha256": checkpoint_digest,
                "size": len(payload),
                "source_fingerprint": "b" * 64,
                "target_relative_path": target,
            },
        },
    )

    source = remote_job._verified_source(job_directory)

    staged = job_directory / target
    assert source == job_directory / "source"
    assert staged.read_bytes() == payload
    assert staged.stat().st_ino == (job_directory / "resume-checkpoint.pt").stat().st_ino


def test_verified_source_rejects_mismatched_resume_checkpoint_digest(
    remote_job,
    job_directory: Path,
):
    archive_digest = _prepared_source_archive(remote_job, job_directory)
    (job_directory / "resume-checkpoint.pt").write_bytes(b"wrong")
    remote_job.atomic_write_json(
        job_directory / "prepared-manifest.json",
        {
            "source": {"archive_sha256": archive_digest},
            "resume": {
                "checkpoint": "model_250.pt",
                "sha256": "a" * 64,
                "size": 5,
                "source_fingerprint": "b" * 64,
                "target_relative_path": "source/logs/velocity/run/model_250.pt",
            },
        },
    )

    with pytest.raises(remote_job.RemoteJobError, match="resume.*digest"):
        remote_job._verified_source(job_directory)


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

    waits = []
    result = remote_job.cancel_job(
        job_directory,
        killpg=lambda *args: killed.append(args),
        wait_for_exit=lambda identity, timeout: waits.append((identity, timeout)) or True,
    )

    assert killed == [(123, remote_job.signal.SIGTERM)]
    assert waits and waits[0][0] == remote_job.ProcessIdentity(123, "100", "a" * 64)
    assert result["status"] == "failed"
    assert result["cancelled"] is True


def test_cancel_escalates_and_refuses_terminal_state_until_exit_confirmed(
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
    confirmations = iter((False, True))

    result = remote_job.cancel_job(
        job_directory,
        killpg=lambda *args: killed.append(args),
        wait_for_exit=lambda identity, timeout: next(confirmations),
    )

    assert killed == [
        (123, remote_job.signal.SIGTERM),
        (123, remote_job.signal.SIGKILL),
    ]
    assert result["termination_confirmed"] is True


def test_cancel_merges_checkpoint_evidence_written_while_waiting(
    remote_job,
    job_directory: Path,
    monkeypatch,
):
    expected = {
        "pid": 123,
        "process_start": "100",
        "command_digest": "a" * 64,
    }
    initial = {"status": "running", **expected, "artifact_status": "pending"}
    remote_job.atomic_write_json(job_directory / "status.json", initial)
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
    checkpoint = {"name": "model_250.pt", "sha256": "b" * 64}

    def finish(identity, timeout):
        remote_job.atomic_write_json(
            job_directory / "status.json",
            {
                **initial,
                "artifact_status": "stable_checkpoint",
                "last_stable_checkpoint": checkpoint,
            },
        )
        return True

    result = remote_job.cancel_job(
        job_directory,
        killpg=lambda *args: None,
        wait_for_exit=finish,
    )

    assert result["last_stable_checkpoint"] == checkpoint
    assert result["artifact_status"] == "stable_checkpoint"


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
    assert state == {
        **initial,
        "error": "bad archive",
        "exit_code": 1,
        "launch_state": "supervising",
        "status": "failed",
        "supervisor_pid": os.getpid(),
    }


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
        {"action": "start", "request_id": "start-abc123"},
    )

    def fail_spawn(*args, **kwargs):
        raise OSError("cannot spawn")

    with pytest.raises(OSError, match="cannot spawn"):
        remote_job._start_supervisor(job_directory, popen=fail_spawn)

    assert (job_directory / "start-request.json").exists()
    assert json.loads((job_directory / "status.json").read_text()) == {"status": "pending"}


def test_claimed_start_is_idempotently_left_for_spawned_supervisor(
    remote_job,
    job_directory: Path,
):
    _pending_start(remote_job, job_directory)
    remote_job.atomic_write_json(
        job_directory / "status.json",
        {
            "status": "pending",
            "launch_request_id": "start-abc123",
            "launch_state": "claimed",
        },
    )

    result = remote_job.inspect_or_control(
        job_directory,
        supervisor_popen=lambda *args, **kwargs: pytest.fail("duplicate spawn"),
    )

    assert result["launch_state"] == "claimed"
    assert (job_directory / "start-request.json").exists()


class SpawnedSupervisor:
    pid = 456


def _pending_start(remote_job, job_directory: Path) -> None:
    remote_job.atomic_write_json(
        job_directory / "status.json",
        {"status": "pending", "pid": None, "process_start": None},
    )
    remote_job.atomic_write_json(
        job_directory / "start-request.json",
        {"action": "start", "request_id": "start-abc123"},
    )


def test_concurrent_starts_atomically_claim_and_spawn_once(remote_job, job_directory: Path):
    _pending_start(remote_job, job_directory)
    calls = []
    barrier = threading.Barrier(2)

    def popen(*args, **kwargs):
        state = json.loads((job_directory / "status.json").read_text())
        calls.append(state["launch_state"])
        return SpawnedSupervisor()

    def start():
        barrier.wait()
        return remote_job.inspect_or_control(job_directory, supervisor_popen=popen)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: start(), range(2)))

    assert calls == ["claimed"]
    assert all(result["launch_state"] == "spawned" for result in results)
    assert not (job_directory / "start-request.json").exists()


def test_post_spawn_crash_retry_is_idempotent(remote_job, job_directory: Path):
    _pending_start(remote_job, job_directory)
    spawned = []

    def popen(*args, **kwargs):
        spawned.append(SpawnedSupervisor())
        return spawned[-1]

    def crash_after_spawn(state):
        raise RuntimeError("wrapper crashed after spawn")

    with pytest.raises(RuntimeError, match="after spawn"):
        remote_job.inspect_or_control(
            job_directory,
            supervisor_popen=popen,
            after_spawn=crash_after_spawn,
        )

    recorded = json.loads((job_directory / "status.json").read_text())
    assert recorded["launch_state"] == "spawned"
    assert recorded["supervisor_pid"] == 456
    assert (job_directory / "start-request.json").exists()

    result = remote_job.inspect_or_control(
        job_directory,
        supervisor_popen=lambda *args, **kwargs: pytest.fail("spawned twice"),
    )
    assert result["launch_state"] == "spawned"
    assert len(spawned) == 1
    assert not (job_directory / "start-request.json").exists()


class LaunchedTrainer:
    pid = 789
    returncode = None

    def poll(self):
        return None


@pytest.mark.parametrize("failure_stage", ("identity", "checkpoint", "poll"))
def test_post_spawn_failure_terminates_and_reaps_process_group(
    remote_job,
    job_directory: Path,
    monkeypatch,
    failure_stage: str,
):
    process = LaunchedTrainer()
    cleaned = []
    monkeypatch.setattr(remote_job, "_verified_source", lambda job: job_directory)
    monkeypatch.setattr(remote_job, "verified_train_argv", remote_job.load_train_argv)
    monkeypatch.setattr(remote_job, "launch_training", lambda *args, **kwargs: process)
    if failure_stage == "identity":
        monkeypatch.setattr(
            remote_job,
            "live_process_identity",
            lambda pid: (_ for _ in ()).throw(remote_job.RemoteJobError("identity failed")),
        )
    else:
        monkeypatch.setattr(
            remote_job,
            "live_process_identity",
            lambda pid: remote_job.ProcessIdentity(
                pid,
                "100",
                remote_job.command_digest(remote_job.load_train_argv(job_directory)),
            ),
        )
        if failure_stage == "checkpoint":
            monkeypatch.setattr(
                remote_job.StableCheckpointTracker,
                "observe",
                lambda self, paths: (_ for _ in ()).throw(
                    remote_job.RemoteJobError("checkpoint failed")
                ),
            )
        else:
            monkeypatch.setattr(
                process,
                "poll",
                lambda: (_ for _ in ()).throw(remote_job.RemoteJobError("poll failed")),
            )
    monkeypatch.setattr(remote_job.time, "sleep", lambda _: None)

    with pytest.raises(remote_job.RemoteJobError, match="failed"):
        remote_job.supervise(
            job_directory,
            terminate_and_reap=lambda child, identity=None: cleaned.append((child, identity)),
        )

    assert cleaned and cleaned[0][0] is process
    state = json.loads((job_directory / "status.json").read_text())
    assert state["status"] == "failed"
    assert state["pid"] == 789
    assert state["termination_confirmed"] is True


class ReapableTrainer:
    pid = 901
    returncode = None

    def __init__(self):
        self.waits = []

    def poll(self):
        raise RuntimeError("poll is broken")

    def wait(self, timeout=None):
        self.waits.append(timeout)
        self.returncode = -15
        return self.returncode


def test_cleanup_terminates_and_reaps_without_relying_on_poll(
    remote_job,
    monkeypatch,
):
    process = ReapableTrainer()
    identity = remote_job.ProcessIdentity(901, "100", "a" * 64)
    monkeypatch.setattr(remote_job.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(remote_job, "live_process_identity", lambda pid: identity)
    killed = []

    remote_job.terminate_and_reap(
        process,
        identity,
        killpg=lambda *args: killed.append(args),
    )

    assert killed == [(901, remote_job.signal.SIGTERM)]
    assert process.waits == [remote_job._TERMINATE_TIMEOUT_SECONDS]


def test_supervisor_finalization_preserves_confirmed_cancellation(
    remote_job,
    job_directory: Path,
):
    cancelled = {
        "status": "failed",
        "cancelled": True,
        "termination_confirmed": True,
        "pid": 123,
        "process_start": "100",
        "command_digest": "a" * 64,
    }
    remote_job.atomic_write_json(job_directory / "status.json", cancelled)

    result = remote_job.finalize_training_state(
        job_directory,
        {**cancelled, "status": "running", "cancelled": False},
        exit_code=-15,
        last_checkpoint=None,
    )

    assert result["cancelled"] is True
    assert result["termination_confirmed"] is True
    assert result["status"] == "failed"

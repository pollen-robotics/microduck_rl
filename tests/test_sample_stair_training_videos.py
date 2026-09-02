import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "sample_stair_training_videos.py"
)
SPEC = importlib.util.spec_from_file_location("sample_stair_training_videos", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
sampler = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sampler
SPEC.loader.exec_module(sampler)


def _args(tmp_path: Path, checkpoint_root: Path) -> argparse.Namespace:
    output_dir = tmp_path / "samples"
    return argparse.Namespace(
        checkpoint_root=checkpoint_root,
        output_dir=output_dir,
        state_file=output_dir / "state.json",
        interval_seconds=150.0,
        task_id="Mjlab-Stairs-Route-MicroDuck",
        walker_checkpoint=tmp_path / "walker.pt",
        device="cpu",
        allow_repeats=False,
        once=True,
    )


def test_default_interval_is_150_seconds() -> None:
    args = sampler._parse_args(["--once"])

    assert args.interval_seconds == 150.0
    assert args.task_id == "Mjlab-Stairs-Route-MicroDuck"


def test_find_newest_checkpoint_uses_modification_time(tmp_path: Path) -> None:
    older = tmp_path / "run-newer-name" / "model_99.pt"
    newer = tmp_path / "run" / "model_25.pt"
    older.parent.mkdir()
    newer.parent.mkdir()
    older.write_bytes(b"older")
    newer.write_bytes(b"newer")
    os.utime(older, ns=(100, 100))
    os.utime(newer, ns=(200, 200))

    assert sampler.find_newest_checkpoint(tmp_path) == newer


def test_sample_once_records_1000_steps_and_persists_state(
    tmp_path: Path, monkeypatch
) -> None:
    checkpoint_root = tmp_path / "run"
    checkpoint_root.mkdir()
    checkpoint = checkpoint_root / "model_25.pt"
    checkpoint.write_bytes(b"checkpoint-25")
    args = _args(tmp_path, checkpoint_root)
    args.walker_checkpoint.write_bytes(b"walker")
    observed: dict[str, object] = {}

    def fake_run(command, *, cwd, check):
        observed["command"] = command
        observed["cwd"] = cwd
        observed["check"] = check
        Path(command[3]).parent.mkdir(parents=True, exist_ok=True)
        Path(command[3]).write_bytes(b"video")
        return argparse.Namespace(returncode=0)

    monkeypatch.setattr(sampler.subprocess, "run", fake_run)

    assert sampler.sample_once(args) is True

    command = observed["command"]
    assert command[2] == str(checkpoint)
    assert command[command.index("--steps") + 1] == "1000"
    assert command[command.index("--task-id") + 1] == args.task_id
    assert command[command.index("--walker-checkpoint") + 1] == str(
        args.walker_checkpoint
    )
    output = Path(command[3])
    assert "checkpoint-000025" in output.name
    assert output.suffix == ".mp4"
    assert observed["cwd"] == sampler.REPO_ROOT
    assert observed["check"] is False

    state = json.loads(args.state_file.read_text(encoding="utf-8"))
    assert state["last_checkpoint"]["iteration"] == 25
    assert state["last_checkpoint"]["path"] == str(checkpoint.resolve())
    assert state["last_video"] == str(output.resolve())


def test_restart_skips_same_path_or_content_unless_repeats_allowed(
    tmp_path: Path, monkeypatch
) -> None:
    checkpoint_root = tmp_path / "run"
    checkpoint_root.mkdir()
    first = checkpoint_root / "model_25.pt"
    first.write_bytes(b"identical-checkpoint")
    args = _args(tmp_path, checkpoint_root)
    args.walker_checkpoint.write_bytes(b"walker")
    calls: list[list[str]] = []

    def fake_run(command, *, cwd, check):
        del cwd, check
        calls.append(command)
        Path(command[3]).parent.mkdir(parents=True, exist_ok=True)
        Path(command[3]).write_bytes(b"video")
        return argparse.Namespace(returncode=0)

    monkeypatch.setattr(sampler.subprocess, "run", fake_run)

    assert sampler.sample_once(args) is True
    assert sampler.sample_once(args) is False

    copied = checkpoint_root / "model_50.pt"
    copied.write_bytes(first.read_bytes())
    newer_time = first.stat().st_mtime_ns + 10_000
    os.utime(copied, ns=(newer_time, newer_time))
    assert sampler.sample_once(args) is False

    args.allow_repeats = True
    assert sampler.sample_once(args) is True
    assert len(calls) == 2


def test_failed_recording_does_not_advance_state(tmp_path: Path, monkeypatch) -> None:
    checkpoint_root = tmp_path / "run"
    checkpoint_root.mkdir()
    (checkpoint_root / "model_74.pt").write_bytes(b"checkpoint")
    args = _args(tmp_path, checkpoint_root)
    args.walker_checkpoint.write_bytes(b"walker")

    monkeypatch.setattr(
        sampler.subprocess,
        "run",
        lambda *unused_args, **unused_kwargs: argparse.Namespace(returncode=7),
    )

    assert sampler.sample_once(args) is False
    assert not args.state_file.exists()

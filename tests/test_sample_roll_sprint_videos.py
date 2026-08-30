import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "sample_roll_sprint_videos.py"
)
SPEC = importlib.util.spec_from_file_location("sample_roll_sprint_videos", SCRIPT_PATH)
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
        task_id="Mjlab-Roll-Sprint-Flat-MicroDuck",
        device="cpu",
        allow_repeats=False,
        once=True,
    )


def test_defaults_record_long_race_every_150_seconds() -> None:
    args = sampler._parse_args(["--once"])

    assert args.interval_seconds == 150.0
    assert args.task_id == "Mjlab-Roll-Sprint-Flat-MicroDuck"
    assert sampler.RECORDING_STEPS == 2000
    assert sampler.RECORDING_FRAME_STRIDE == 7
    assert sampler.EVALUATION_DURATION == 40.0


def test_find_newest_checkpoint_uses_modification_time(tmp_path: Path) -> None:
    older = tmp_path / "new-name" / "model_99.pt"
    newer = tmp_path / "run" / "model_25.pt"
    older.parent.mkdir()
    newer.parent.mkdir()
    older.write_bytes(b"older")
    newer.write_bytes(b"newer")
    os.utime(older, ns=(100, 100))
    os.utime(newer, ns=(200, 200))

    assert sampler.find_newest_checkpoint(tmp_path) == newer


def test_sample_once_records_four_robot_video_and_persists_state(
    tmp_path: Path, monkeypatch
) -> None:
    checkpoint_root = tmp_path / "run"
    checkpoint_root.mkdir()
    checkpoint = checkpoint_root / "model_25.pt"
    checkpoint.write_bytes(b"checkpoint-25")
    args = _args(tmp_path, checkpoint_root)
    observed: dict[str, object] = {"commands": []}

    def fake_run(command, *, cwd, check):
        observed["commands"].append(command)
        observed["cwd"] = cwd
        observed["check"] = check
        if command[1] == str(sampler.EVALUATOR):
            output = Path(command[command.index("--output") + 1])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("{}", encoding="utf-8")
        else:
            Path(command[3]).parent.mkdir(parents=True, exist_ok=True)
            Path(command[3]).write_bytes(b"video")
        return argparse.Namespace(returncode=0)

    monkeypatch.setattr(sampler.subprocess, "run", fake_run)

    assert sampler.sample_once(args) is True
    evaluation_command, command = observed["commands"]
    assert evaluation_command[2] == str(checkpoint)
    assert evaluation_command[evaluation_command.index("--num-envs") + 1] == "4"
    assert evaluation_command[evaluation_command.index("--duration") + 1] == "40"
    assert command[2] == str(checkpoint)
    assert command[command.index("--steps") + 1] == "2000"
    assert command[command.index("--frame-stride") + 1] == "7"
    assert command[command.index("--task-id") + 1] == args.task_id
    output = Path(command[3])
    assert "checkpoint-000025" in output.name
    assert output.suffix == ".mp4"
    assert observed["cwd"] == sampler.REPO_ROOT
    assert observed["check"] is False

    state = json.loads(args.state_file.read_text(encoding="utf-8"))
    assert state["last_checkpoint"]["iteration"] == 25
    assert Path(state["last_evaluation"]).is_file()
    assert state["last_video"] == str(output.resolve())


def test_duplicate_checkpoint_is_skipped(tmp_path: Path, monkeypatch) -> None:
    checkpoint_root = tmp_path / "run"
    checkpoint_root.mkdir()
    checkpoint = checkpoint_root / "model_25.pt"
    checkpoint.write_bytes(b"checkpoint")
    args = _args(tmp_path, checkpoint_root)
    calls: list[list[str]] = []

    def fake_run(command, *, cwd, check):
        del cwd, check
        calls.append(command)
        if command[1] == str(sampler.EVALUATOR):
            output = Path(command[command.index("--output") + 1])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("{}", encoding="utf-8")
        else:
            Path(command[3]).parent.mkdir(parents=True, exist_ok=True)
            Path(command[3]).write_bytes(b"video")
        return argparse.Namespace(returncode=0)

    monkeypatch.setattr(sampler.subprocess, "run", fake_run)

    assert sampler.sample_once(args) is True
    assert sampler.sample_once(args) is False
    assert len(calls) == 2


def test_allow_repeats_records_every_interval_but_audits_checkpoint_once(
    tmp_path: Path, monkeypatch
) -> None:
    checkpoint_root = tmp_path / "run"
    checkpoint_root.mkdir()
    (checkpoint_root / "model_100.pt").write_bytes(b"checkpoint")
    args = _args(tmp_path, checkpoint_root)
    args.allow_repeats = True
    calls: list[list[str]] = []

    def fake_run(command, *, cwd, check):
        del cwd, check
        calls.append(command)
        if command[1] == str(sampler.EVALUATOR):
            output = Path(command[command.index("--output") + 1])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("{}", encoding="utf-8")
        else:
            Path(command[3]).parent.mkdir(parents=True, exist_ok=True)
            Path(command[3]).write_bytes(b"video")
        return argparse.Namespace(returncode=0)

    monkeypatch.setattr(sampler.subprocess, "run", fake_run)

    assert sampler.sample_once(args) is True
    assert sampler.sample_once(args) is True
    assert sum(command[1] == str(sampler.EVALUATOR) for command in calls) == 1
    assert sum(command[1] == str(sampler.RECORDER) for command in calls) == 2

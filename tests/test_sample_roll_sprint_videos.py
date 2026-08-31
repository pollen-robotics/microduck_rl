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
        champion_dir=None,
        interval_seconds=150.0,
        task_id="Mjlab-Roll-Sprint-Flat-MicroDuck",
        device="cpu",
        parent_frontier_m=None,
        allow_repeats=False,
        video_only=False,
        audit_only=False,
        montage_only=False,
        once=True,
    )


def test_defaults_record_long_race_every_150_seconds() -> None:
    args = sampler._parse_args(["--once"])

    assert args.interval_seconds == 150.0
    assert args.task_id == "Mjlab-Roll-Sprint-Flat-MicroDuck"
    assert sampler.RECORDING_STEPS == 2000
    assert sampler.RECOVERY_RECORDING_STEPS == 600
    assert sampler.RECORDING_FRAME_STRIDE == 2
    assert sampler.PLAYBACK_SPEED == 2.0
    assert sampler.OUTPUT_FPS == 60.0
    assert (sampler.OUTPUT_WIDTH, sampler.OUTPUT_HEIGHT) == (1920, 1080)
    assert sampler.RECORDING_STEPS / sampler.SIMULATION_HZ == 40.0
    distinct_motion_fps = (
        sampler.SIMULATION_HZ
        / sampler.RECORDING_FRAME_STRIDE
        * sampler.PLAYBACK_SPEED
    )
    assert distinct_motion_fps == 50.0
    assert (
        sampler.RECORDING_STEPS
        / sampler.RECORDING_FRAME_STRIDE
        / distinct_motion_fps
        == sampler.OUTPUT_VIDEO_SECONDS
        == 20.0
    )
    assert sampler.EVALUATION_DURATION == 40.0
    assert sampler.EVALUATION_SCHEMA_VERSION == 8
    assert args.parent_frontier_m is None


def test_evaluator_command_carries_selected_parent_frontier() -> None:
    command = sampler._evaluator_command(
        checkpoint=Path("model_100.pt"),
        output=Path("evaluation.json"),
        device="cpu",
        parent_frontier_m=0.5306979418,
    )

    assert command[command.index("--parent-frontier-m") + 1] == "0.530698"


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
    evaluation_command, command, recovery_command = observed["commands"]
    assert evaluation_command[2] == str(checkpoint)
    assert evaluation_command[evaluation_command.index("--num-envs") + 1] == "4"
    assert evaluation_command[evaluation_command.index("--duration") + 1] == "40"
    assert command[2] == str(checkpoint)
    assert command[command.index("--steps") + 1] == "2000"
    assert command[command.index("--frame-stride") + 1] == "2"
    assert command[command.index("--output-fps") + 1] == "60.0"
    assert command[command.index("--playback-speed") + 1] == "2.0"
    assert command[command.index("--width") + 1] == "1920"
    assert command[command.index("--height") + 1] == "1080"
    assert command[command.index("--task-id") + 1] == args.task_id
    assert "--recovery-montage" not in command
    assert "--recovery-montage" in recovery_command
    assert recovery_command[recovery_command.index("--steps") + 1] == "600"
    output = Path(command[3])
    assert "checkpoint-000025" in output.name
    assert output.suffix == ".mp4"
    assert observed["cwd"] == sampler.REPO_ROOT
    assert observed["check"] is False

    state = json.loads(args.state_file.read_text(encoding="utf-8"))
    assert state["last_checkpoint"]["iteration"] == 25
    assert Path(state["last_evaluation"]).is_file()
    assert "race-40s-v8" in Path(state["last_evaluation"]).name
    assert state["last_video"] == str(output.resolve())
    assert Path(state["last_recovery_montage"]).is_file()


def test_champion_video_is_retained_by_exact_checkpoint_hash(tmp_path: Path) -> None:
    champion_dir = tmp_path / "champion"
    champion_dir.mkdir()
    checkpoint = champion_dir / "model_25.pt"
    checkpoint.write_bytes(b"champion-checkpoint")
    identity = sampler.checkpoint_identity(checkpoint)
    (champion_dir / "champion.json").write_text(
        json.dumps(
            {
                "retained_checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": identity.sha256,
            }
        ),
        encoding="utf-8",
    )
    recording = tmp_path / "race.mp4"
    recording.write_bytes(b"clean-hd-video")

    retained = sampler._retain_champion_video(checkpoint, identity, recording)

    assert retained == (
        champion_dir / f"champion-000025-{identity.sha256[:12]}.mp4"
    )
    assert retained.read_bytes() == b"clean-hd-video"

    other = champion_dir / "model_26.pt"
    other.write_bytes(b"other-checkpoint")
    other_identity = sampler.checkpoint_identity(other)
    assert sampler._retain_champion_video(other, other_identity, recording) is None
    assert list(champion_dir.glob("champion-*.mp4")) == [retained]


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
    assert len(calls) == 3


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
    assert sum(command[1] == str(sampler.RECORDER) for command in calls) == 3
    assert sum("--recovery-montage" in command for command in calls) == 1


def test_video_only_keeps_recording_independent_from_audit(
    tmp_path: Path, monkeypatch
) -> None:
    checkpoint_root = tmp_path / "run"
    checkpoint_root.mkdir()
    (checkpoint_root / "model_100.pt").write_bytes(b"checkpoint")
    args = _args(tmp_path, checkpoint_root)
    args.video_only = True
    calls: list[list[str]] = []

    def fake_run(command, *, cwd, check):
        del cwd, check
        calls.append(command)
        Path(command[3]).parent.mkdir(parents=True, exist_ok=True)
        Path(command[3]).write_bytes(b"video")
        return argparse.Namespace(returncode=0)

    monkeypatch.setattr(sampler.subprocess, "run", fake_run)

    assert sampler.sample_once(args) is True
    assert len(calls) == 1
    assert all(command[1] == str(sampler.RECORDER) for command in calls)
    assert "--recovery-montage" not in calls[0]
    state = json.loads(args.state_file.read_text(encoding="utf-8"))
    assert state["last_evaluation"] is None
    assert state["last_recovery_montage"] is None


def test_montage_only_records_one_unique_checkpoint_without_race_or_audit(
    tmp_path: Path, monkeypatch
) -> None:
    checkpoint_root = tmp_path / "run"
    checkpoint_root.mkdir()
    (checkpoint_root / "model_100.pt").write_bytes(b"checkpoint")
    args = _args(tmp_path, checkpoint_root)
    args.montage_only = True
    calls: list[list[str]] = []

    def fake_run(command, *, cwd, check):
        del cwd, check
        calls.append(command)
        Path(command[3]).parent.mkdir(parents=True, exist_ok=True)
        Path(command[3]).write_bytes(b"video")
        return argparse.Namespace(returncode=0)

    monkeypatch.setattr(sampler.subprocess, "run", fake_run)

    assert sampler.sample_once(args) is True
    assert len(calls) == 1
    assert calls[0][1] == str(sampler.RECORDER)
    assert "--recovery-montage" in calls[0]
    assert sampler.sample_once(args) is False
    state = json.loads(args.state_file.read_text(encoding="utf-8"))
    assert state["last_video"] is None
    assert Path(state["last_recovery_montage"]).is_file()


def test_audit_only_skips_video_recording(tmp_path: Path, monkeypatch) -> None:
    checkpoint_root = tmp_path / "run"
    checkpoint_root.mkdir()
    (checkpoint_root / "model_100.pt").write_bytes(b"checkpoint")
    args = _args(tmp_path, checkpoint_root)
    args.audit_only = True
    calls: list[list[str]] = []

    def fake_run(command, *, cwd, check):
        del cwd, check
        calls.append(command)
        output = Path(command[command.index("--output") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("{}", encoding="utf-8")
        return argparse.Namespace(returncode=0)

    monkeypatch.setattr(sampler.subprocess, "run", fake_run)

    assert sampler.sample_once(args) is True
    assert len(calls) == 1
    assert calls[0][1] == str(sampler.EVALUATOR)
    state = json.loads(args.state_file.read_text(encoding="utf-8"))
    assert state["version"] == 4
    assert state["last_checkpoint"]["iteration"] == 100
    assert Path(state["last_evaluation"]).is_file()

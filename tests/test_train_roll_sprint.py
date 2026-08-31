from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "train_roll_sprint.py"
SPEC = importlib.util.spec_from_file_location("train_roll_sprint", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_short_burst_save_interval_is_configurable(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT_PATH),
            "--num-envs",
            "4096",
            "--iterations",
            "10",
            "--save-interval",
            "1",
            "--seed",
            "7",
            "--run-name",
            "short_burst",
        ],
    )

    args = MODULE._parse_args()

    assert args.num_envs == 4096
    assert args.iterations == 10
    assert args.save_interval == 1
    assert args.seed == 7
    assert args.run_name == "short_burst"


def test_seed_defaults_to_mjlab_default_and_accepts_zero(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", [str(SCRIPT_PATH)])
    defaults = MODULE._parse_args()

    monkeypatch.setattr(sys, "argv", [str(SCRIPT_PATH), "--seed", "0"])
    zero_seed = MODULE._parse_args()

    assert defaults.seed == 42
    assert zero_seed.seed == 0


def test_negative_seed_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", [str(SCRIPT_PATH), "--seed", "-1"])

    with pytest.raises(SystemExit):
        MODULE._parse_args()


def test_training_command_forwards_seed_exactly_once(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    source.write_bytes(b"checkpoint")
    train_exe = tmp_path / ".venv" / (
        "Scripts/train.exe" if MODULE.os.name == "nt" else "bin/train"
    )
    train_exe.parent.mkdir(parents=True)
    train_exe.write_bytes(b"executable")
    captured: dict[str, object] = {}
    monkeypatch.setattr(MODULE, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(MODULE, "stage_checkpoint", lambda *_args: None)

    def capture_call(command, **kwargs) -> int:
        captured["command"] = command
        captured["kwargs"] = kwargs
        return 0

    monkeypatch.setattr(MODULE.subprocess, "call", capture_call)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT_PATH),
            "--source-checkpoint",
            str(source),
            "--seed",
            "31415",
        ],
    )

    assert MODULE.main() == 0
    command = captured["command"]
    assert isinstance(command, list)
    assert command[0] == str(train_exe)
    assert command.count("--agent.seed") == 1
    seed_index = command.index("--agent.seed")
    assert command[seed_index : seed_index + 2] == ["--agent.seed", "31415"]

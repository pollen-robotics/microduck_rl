from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

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
            "--run-name",
            "short_burst",
        ],
    )

    args = MODULE._parse_args()

    assert args.num_envs == 4096
    assert args.iterations == 10
    assert args.save_interval == 1
    assert args.run_name == "short_burst"

"""`microduck <command>`: one name for the training side, mirroring `robotctl` on the robot.

Every command delegates to the entry point that already exists — `train` to mjlab's trainer,
`publish` to ours — with argv[0] rewritten to the command's name BEFORE mjlab is imported,
because the --hf-jobs interception and the provenance hook both key on argv[0] == "train".
`check` is what publish verifies, without the Hub."""

from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

import pytest

from mjlab_microduck import cli

_ROOT = Path(__file__).resolve().parents[1]


def test_microduck_is_a_declared_script():
    scripts = tomllib.loads((_ROOT / "pyproject.toml").read_text())["project"]["scripts"]
    assert scripts.get("microduck") == "mjlab_microduck.cli:main"
    assert scripts.get("train") == "mjlab_microduck.train_cli:main", "the old names stay"


def test_every_command_names_a_real_entry_point():
    import importlib

    for name, (module, attr) in cli.COMMANDS.items():
        assert callable(getattr(importlib.import_module(module), attr)), name
    assert set(cli.COMMANDS) == {"train", "play", "list-envs", "export", "publish", "check", "infer"}


def test_train_is_delegated_with_argv0_rewritten(monkeypatch):
    import mjlab.scripts.train

    seen = []
    monkeypatch.setattr(mjlab.scripts.train, "main", lambda: seen.append(list(sys.argv)) or 7)
    monkeypatch.setattr(sys, "argv", ["/v/bin/microduck", "train", "Mjlab-Velocity-Flat-MicroDuck", "--agent.seed", "3"])
    assert cli.main(["train", "Mjlab-Velocity-Flat-MicroDuck", "--agent.seed", "3"]) == 7
    assert seen == [["train", "Mjlab-Velocity-Flat-MicroDuck", "--agent.seed", "3"]]


def test_the_hooks_see_train(monkeypatch):
    """What the rewrite is for: both import-time hooks decide on argv[0]'s name."""
    import mjlab.scripts.train

    from mjlab_microduck.train_hook import _invoked_as_train

    verdict = []
    monkeypatch.setattr(mjlab.scripts.train, "main", lambda: verdict.append(_invoked_as_train()) or 0)
    cli.main(["train", "T"])
    assert verdict == [True]


def test_no_command_lists_them(capsys):
    assert cli.main([]) == 0
    out = capsys.readouterr().out
    for name in cli.COMMANDS:
        assert name in out


def test_an_unknown_command_is_named(capsys):
    assert cli.main(["dance"]) == 2
    assert "dance" in capsys.readouterr().err


def test_check_passes_a_good_policy(tmp_path, capsys):
    from mjlab_microduck.publish.check import CheckConfig, run
    from tests.test_publish_manifest import _tiny_policy

    policy = _tiny_policy(tmp_path / "out.onnx")
    assert run(CheckConfig(onnx=str(policy))) == 0
    assert "61 -> 14" in capsys.readouterr().out


def test_check_refuses_a_legacy_policy(tmp_path, capsys):
    from mjlab_microduck.publish.check import CheckConfig, run
    from tests.test_publish_manifest import _tiny_policy

    policy = _tiny_policy(tmp_path / "old.onnx", obs_len=51)
    assert run(CheckConfig(onnx=str(policy))) == 2
    assert "51" in capsys.readouterr().err


def test_check_reads_a_run(tmp_path, capsys):
    from mjlab_microduck.publish.check import CheckConfig, run

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    assert run(CheckConfig(run=str(run_dir))) == 2
    assert "provenance.json" in capsys.readouterr().err
    (run_dir / "provenance.json").write_text(json.dumps({"task": "T", "commit": "abc", "dirty": True, "command": ["train", "T"]}))
    (run_dir / "model_10.pt").write_bytes(b"x")
    assert run(CheckConfig(run=str(run_dir))) == 0
    out = capsys.readouterr().out
    assert "model_10.pt" in out and "uncommitted" in out


def test_check_validates_a_manifest(tmp_path, capsys):
    from mjlab_microduck.publish.check import CheckConfig, run

    good = tmp_path / "manifest.json"
    good.write_text(json.dumps({"schema_version": 2, "obs_len": 61, "action_len": 14, "kind": "perpetual"}))
    assert run(CheckConfig(manifest=str(good))) == 0
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"obs_len": 51}))
    assert run(CheckConfig(manifest=str(bad))) == 2
    assert "obs_len" in capsys.readouterr().err


def test_check_with_nothing_to_check_says_so(capsys):
    from mjlab_microduck.publish.check import CheckConfig, run

    assert run(CheckConfig()) == 2
    assert "--onnx" in capsys.readouterr().err

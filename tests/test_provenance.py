"""`provenance.json` is what turns "here is my model" into "here is how I got it".

Written once, at the start of a training run, from the checkout `train` runs in (the
working directory's repo — a challenges checkout, not this library), so the commit it
names is the code that trained, whatever is published later.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from mjlab_microduck import provenance


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A one-commit checkout with an `origin` pointing at a fork."""
    root = tmp_path / "challenges"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "t")
    (root / "challenge.toml").write_text("event = 'sprint-2m'\n")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "first")
    _git(root, "remote", "add", "origin", "https://github.com/alice/microduck-challenges")
    return root


def test_checkout_names_the_fork_commit_and_cleanliness(repo):
    got = provenance.checkout(repo)
    assert got["repo"] == "https://github.com/alice/microduck-challenges"
    assert got["commit"] == _git(repo, "rev-parse", "--short=9", "HEAD")
    assert got["branch"] == "main"
    assert got["dirty"] is False


def test_a_modified_tracked_file_is_dirty(repo):
    (repo / "challenge.toml").write_text("event = 'hill-climb-1m5'\n")
    assert provenance.checkout(repo)["dirty"] is True


def test_credentials_in_the_origin_are_not_recorded(repo):
    _git(repo, "remote", "set-url", "origin", "https://alice:ghp_x@github.com/alice/microduck-challenges")
    assert provenance.checkout(repo)["repo"] == "https://github.com/alice/microduck-challenges"


def test_an_ssh_origin_is_recorded_as_is(repo):
    _git(repo, "remote", "set-url", "origin", "git@example.com:alice/microduck-challenges.git")
    assert provenance.checkout(repo)["repo"] == "git@example.com:alice/microduck-challenges.git"


def test_no_origin_records_no_repo(repo):
    _git(repo, "remote", "remove", "origin")
    got = provenance.checkout(repo)
    assert "repo" not in got and got["commit"]


def test_contains_knows_the_commits_in_the_history(repo, tmp_path):
    first = _git(repo, "rev-parse", "HEAD")
    (repo / "env.py").write_text("")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "second")
    assert provenance.contains(repo, first)
    assert provenance.contains(repo, first[:9])
    assert not provenance.contains(repo, "0123456789abcdef0123456789abcdef01234567")
    outside = tmp_path / "plain"
    outside.mkdir()
    assert not provenance.contains(outside, first)


def test_outside_git_records_no_checkout(tmp_path, monkeypatch):
    monkeypatch.delenv(provenance.JOB_ENV, raising=False)
    assert provenance.checkout(tmp_path) == {}
    path = provenance.record_run_start(tmp_path / "run", argv=["train", "T"], seed=1, cwd=tmp_path)
    got = json.loads(path.read_text())
    assert "commit" not in got and got["command"] == ["train", "T"]


def test_a_job_falls_back_to_the_submitters_checkout(tmp_path, monkeypatch):
    """Inside an HF Job the tarball has no .git; submit() hands over the checkout it came from."""
    monkeypatch.setenv(provenance.JOB_ENV, json.dumps(
        {"repo": "https://github.com/alice/microduck-challenges", "commit": "abc123def", "branch": "main", "dirty": False}
    ))
    path = provenance.record_run_start(tmp_path / "run", argv=["train", "T"], seed=1, cwd=tmp_path)
    got = json.loads(path.read_text())
    assert got["commit"] == "abc123def" and got["repo"].endswith("microduck-challenges")


def test_base_names_this_package():
    assert provenance.base().startswith("mjlab-microduck ")


def test_record_run_start_writes_the_recipe(repo, tmp_path):
    log_dir = tmp_path / "logs" / "rsl_rl" / "velocity" / "2026-09-25_10-00-00_first"
    argv = ["/x/.venv/bin/train", "Mjlab-Velocity-Flat-MicroDuck", "--env.scene.num-envs", "4096"]
    path = provenance.record_run_start(log_dir, argv=argv, seed=7, cwd=repo)
    assert path == log_dir / provenance.FILE
    got = json.loads(path.read_text())
    assert got["command"] == ["train", "Mjlab-Velocity-Flat-MicroDuck", "--env.scene.num-envs", "4096"]
    assert got["task"] == "Mjlab-Velocity-Flat-MicroDuck"
    assert got["seed"] == 7
    assert got["repo"] == "https://github.com/alice/microduck-challenges"
    assert got["dirty"] is False
    assert got["base"].startswith("mjlab-microduck ")
    assert got["started"].endswith("Z")


def test_command_starts_with_bare_train(tmp_path):
    path = provenance.record_run_start(tmp_path / "run", argv=["/a/b/train-script.py", "T"], seed=1, cwd=tmp_path)
    assert json.loads(path.read_text())["command"][0] == "train"


def test_written_once(tmp_path):
    log_dir = tmp_path / "run"
    first = provenance.record_run_start(log_dir, argv=["train", "T"], seed=1, cwd=tmp_path)
    before = first.read_text()
    assert provenance.record_run_start(log_dir, argv=["train", "T", "--other"], seed=2, cwd=tmp_path) is None
    assert first.read_text() == before, "a second construction of the runner (play, export) must not rewrite it"


def test_read_names_the_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="provenance.json"):
        provenance.read(tmp_path / "run")
    provenance.record_run_start(tmp_path / "run", argv=["train", "T"], seed=3, cwd=tmp_path)
    assert provenance.read(tmp_path / "run")["seed"] == 3


def test_runner_records_only_when_invoked_as_train(tmp_path, monkeypatch):
    """The hook is a one-liner in the runner; check it by calling the same helper the runner does."""
    import sys

    from mjlab_microduck.tasks import _record_if_training

    log_dir = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["/v/bin/play", "Mjlab-Velocity-Flat-MicroDuck"])
    _record_if_training(str(log_dir), {"seed": 4})
    assert not (log_dir / provenance.FILE).exists(), "play must not write a run's provenance"

    monkeypatch.setattr(sys, "argv", ["/v/bin/train", "Mjlab-Velocity-Flat-MicroDuck"])
    _record_if_training(None, {"seed": 4})
    assert not (log_dir / provenance.FILE).exists(), "export builds the runner with no log_dir"

    _record_if_training(str(log_dir), {"seed": 4})
    assert provenance.read(log_dir)["seed"] == 4


def test_the_runner_calls_the_hook():
    """`MicroduckOnPolicyRunner.__init__` is what every task registers; the hook must sit there."""
    import inspect

    from mjlab_microduck.tasks import MicroduckOnPolicyRunner

    assert "_record_if_training(log_dir, train_cfg)" in inspect.getsource(MicroduckOnPolicyRunner.__init__)

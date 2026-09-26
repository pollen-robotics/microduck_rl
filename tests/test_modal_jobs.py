"""`train <task> ... --modal` must submit to Modal, and the container must be
able to import the submitter module with the bare Modal runtime python.

The interception is the same mechanism as --hf-jobs (train_hook.py, tested in
test_hf_jobs_flag.py); this file locks what is specific to the Modal backend.
"""

import ast
import sys
import time
from pathlib import Path

import pytest

from mjlab_microduck import modal_jobs, train_hook

_ROOT = Path(__file__).resolve().parents[1]
_TASK = "Mjlab-Velocity-Flat-MicroDuck"


@pytest.fixture
def fake_submit(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "mjlab_microduck.modal_jobs.submit", lambda argv: calls.append(argv) or 0
    )
    return calls


def test_hook_submits_to_modal_and_strips_the_flag(monkeypatch, fake_submit):
    monkeypatch.setattr(
        sys,
        "argv",
        ["/x/.venv/bin/train", _TASK, "--env.scene.num-envs", "4096", "--modal"],
    )
    with pytest.raises(SystemExit) as exc:
        train_hook.maybe_submit_to_hf_jobs()
    assert exc.value.code == 0
    # --modal must not reach the container's own `uv run train`, which is mjlab's.
    assert fake_submit == [[_TASK, "--env.scene.num-envs", "4096"]]


def test_two_backends_is_an_error(monkeypatch, fake_submit):
    monkeypatch.setattr(sys, "argv", ["train", _TASK, "--modal", "--hf-jobs"])
    with pytest.raises(SystemExit) as exc:
        train_hook.maybe_submit_to_hf_jobs()
    assert exc.value.code == 2
    assert fake_submit == []


def test_interception_is_disarmed_inside_the_container(monkeypatch, fake_submit):
    monkeypatch.setenv(train_hook._IN_JOB_ENV, "1")
    monkeypatch.setattr(sys, "argv", ["train", _TASK, "--modal"])
    assert train_hook.maybe_submit_to_hf_jobs() is None
    assert fake_submit == []


def test_container_env_disarms_the_interception():
    """The hook checks one variable; the Modal submitter must set that one."""
    assert modal_jobs.IN_JOB_ENV == train_hook._IN_JOB_ENV


_STDLIB_OK = {
    "__future__", "argparse", "datetime", "os", "re", "shlex", "subprocess",
    "sys", "tempfile", "threading", "time", "pathlib",
}


def test_module_level_imports_are_stdlib_only():
    """Modal's runtime python imports modal_jobs to find _train_in_container.

    That interpreter has the modal client and the stdlib — no mjlab, torch,
    huggingface_hub, wandb. A top-level import of any of those makes every
    container die at import with ModuleNotFoundError before training starts.
    """
    tree = ast.parse((_ROOT / "src/mjlab_microduck/modal_jobs.py").read_text())
    top_level = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            top_level.add((node.module or "").split(".")[0])
    assert top_level <= _STDLIB_OK, (
        f"non-stdlib module-level import(s) in modal_jobs.py: {top_level - _STDLIB_OK}"
    )


def test_no_wandb_falls_back_to_tensorboard_only_when_unset():
    assert not modal_jobs._has_logger_flag(["--env.scene.num-envs", "64"])
    assert modal_jobs._has_logger_flag(["--agent.logger", "tensorboard"])
    assert modal_jobs._has_logger_flag(["--agent.logger=wandb"])


def test_app_name_is_modal_legal():
    name = modal_jobs._app_name("Mjlab-Velocity-Flat-MicroDuck-20260904-120000")
    assert name == "microduck-rl-mjlab-velocity-flat-microduck-20260904-120000"
    assert len(modal_jobs._app_name("x" * 200)) <= 64
    assert modal_jobs._app_name("a_b/c d") == "microduck-rl-a-b-c-d"


def test_pick_checkpoint_prefers_this_run_and_highest_iteration(tmp_path):
    old = tmp_path / "rsl_rl" / "velocity" / "old_run"
    new = tmp_path / "rsl_rl" / "velocity" / "new_run"
    for d in (old, new):
        d.mkdir(parents=True)
    (old / "model_9999.pt").write_bytes(b"x")
    since = time.time() + 1  # everything so far is "before this run"
    assert modal_jobs._pick_checkpoint(tmp_path, since) is None

    for it in (250, 1000, 500):
        (new / f"model_{it}.pt").write_bytes(b"x")
    now = time.time()
    for p in list(new.glob("*.pt")) + [old / "model_9999.pt"]:
        # old run stays stale; the new run's files are fresh
        stamp = now - 3600 if p.parent == old else now + 5
        import os
        os.utime(p, (stamp, stamp))
    picked = modal_jobs._pick_checkpoint(tmp_path, since=now)
    assert picked == new / "model_1000.pt"

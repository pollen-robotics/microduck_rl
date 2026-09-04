"""`train <task> ... --hf-jobs` must keep submitting to HF Jobs.

The flag was owned by a `train` console script of ours that was supposed to
shadow mjlab's. Same-name console scripts are last-writer-wins, mjlab's shim
won, and the flag silently vanished: `uv run train ... --hf-jobs` hit mjlab's
tyro parser and died with `Unrecognized options: --hf-jobs` (2026-08-31).

It is now intercepted from the `mjlab.tasks` plugin entry point instead
(train_hook.py), which mjlab imports itself, so either shim handles the flag.

The `train` declaration stays, though — dropping it made things worse, not
better: both RECORDs claim bin/train, so `uv sync` uninstalled the file and
nothing recreated mjlab's, leaving `uv run train` to find liblinear's `train`
on PATH ("can't open input file Mjlab-Velocity-Flat-MicroDuck").

Three things must hold, none of which fails loudly on its own:

1. `train` stays declared, and stays behaviorally identical to mjlab's.
2. mjlab must still reach `mjlab_microduck.tasks` before it parses argv — a
   refactor of mjlab's plugin loading would silently drop the flag.
3. Both shims must intercept, since which one lands in bin/ is not ours to
   decide.
"""

import shutil
import subprocess
import sys
import tomllib
from importlib.metadata import distribution
from pathlib import Path

import pytest

from mjlab_microduck import train_hook

_ROOT = Path(__file__).resolve().parents[1]
_TASK = "Mjlab-Velocity-Flat-MicroDuck"


def _our_scripts():
    pyproject = tomllib.loads((_ROOT / "pyproject.toml").read_text())
    return pyproject["project"].get("scripts", {})


def test_train_script_stays_declared():
    """Removing it uninstalls bin/train and recreates nothing (see docstring)."""
    assert _our_scripts().get("train") == "mjlab_microduck.train_cli:main", (
        "[project.scripts] must keep declaring `train`. Both this package and "
        "mjlab claim bin/train in their RECORD, so dropping the declaration "
        "makes `uv sync` DELETE the script instead of reverting it to mjlab's, "
        "and `uv run train` then runs an unrelated `train` from PATH."
    )


def test_our_train_script_only_delegates_to_mjlab():
    """The collision is only safe while the two shims are interchangeable."""
    import mjlab.scripts.train

    from mjlab_microduck import train_cli

    called = []
    original = mjlab.scripts.train.main
    mjlab.scripts.train.main = lambda: called.append(True) or 5
    try:
        assert train_cli.main() == 5, "our `train` must return mjlab's exit code"
    finally:
        mjlab.scripts.train.main = original
    assert called == [True], "our `train` must call mjlab's trainer, unmodified"


def test_train_on_path_is_a_mjlab_trainer():
    """Catches the vanished-script failure directly: `train` must be ours or
    mjlab's, never whatever else is installed on the machine."""
    exe = shutil.which("train")
    assert exe is not None, (
        "no `train` on PATH — the venv script was uninstalled and not recreated; "
        "run `uv sync --reinstall-package mjlab-microduck`."
    )
    head = Path(exe).read_bytes()[:8192]
    assert b"mjlab" in head, (
        f"`train` resolves to {exe}, which is not a mjlab trainer. bin/train was "
        "uninstalled and this is an unrelated binary from PATH."
    )


def test_we_register_the_task_plugin_entry_point():
    """The interception rides on this entry point; without it, no flag."""
    groups = {ep.group: ep.value for ep in distribution("mjlab-microduck").entry_points}
    assert groups.get("mjlab.tasks") == "mjlab_microduck.tasks", (
        "the `mjlab.tasks` entry point must stay pointed at mjlab_microduck.tasks: "
        "it is both how tasks register AND where --hf-jobs is intercepted."
    )


@pytest.fixture
def fake_submit(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "mjlab_microduck.hf_jobs.submit", lambda argv: calls.append(argv) or 0
    )
    return calls


def test_hook_submits_and_strips_the_flag(monkeypatch, fake_submit):
    monkeypatch.setattr(
        sys,
        "argv",
        ["/x/.venv/bin/train", _TASK, "--env.scene.num-envs", "4096", "--hf-jobs"],
    )
    with pytest.raises(SystemExit) as exc:
        train_hook.maybe_submit_to_hf_jobs()
    assert exc.value.code == 0
    # --hf-jobs must not reach the job's own `uv run train`, which is mjlab's.
    assert fake_submit == [[_TASK, "--env.scene.num-envs", "4096"]]


def test_hook_propagates_the_submission_exit_code(monkeypatch):
    monkeypatch.setattr("mjlab_microduck.hf_jobs.submit", lambda argv: 3)
    monkeypatch.setattr(sys, "argv", ["train", _TASK, "--hf-jobs"])
    with pytest.raises(SystemExit) as exc:
        train_hook.maybe_submit_to_hf_jobs()
    assert exc.value.code == 3


def test_no_flag_trains_locally(monkeypatch, fake_submit):
    monkeypatch.setattr(sys, "argv", ["train", _TASK, "--env.scene.num-envs", "4096"])
    assert train_hook.maybe_submit_to_hf_jobs() is None
    assert fake_submit == []


def test_other_commands_do_not_submit(monkeypatch, fake_submit):
    """`play --hf-jobs` must reach play's parser, not submit a training job."""
    monkeypatch.setattr(sys, "argv", ["/x/.venv/bin/play", _TASK, "--hf-jobs"])
    assert train_hook.maybe_submit_to_hf_jobs() is None
    assert fake_submit == []


def test_interception_is_disarmed_inside_the_job(monkeypatch, fake_submit):
    """Otherwise a leaked flag would make the job submit another job."""
    monkeypatch.setenv("MICRODUCK_IN_HF_JOB", "1")
    monkeypatch.setattr(sys, "argv", ["train", _TASK, "--hf-jobs"])
    assert train_hook.maybe_submit_to_hf_jobs() is None
    assert fake_submit == []


def test_submitted_job_env_disarms_the_interception():
    """The env var above is only useful if submit() actually sets it."""
    src = (_ROOT / "src/mjlab_microduck/hf_jobs.py").read_text()
    assert f'"{train_hook._IN_JOB_ENV}": "1"' in src, (
        f"submit() must put {train_hook._IN_JOB_ENV} on the job's env"
    )


# The load-bearing assumption, exercised through the real import paths: both
# `from mjlab.scripts.train import main` (mjlab's shim) and our own shim must
# reach the hook before mjlab parses argv. In a subprocess, because it ends in
# SystemExit inside an import.
_PROBE = """
import sys
sys.argv = ["train", "{task}", "--env.scene.num-envs", "4096", "--hf-jobs"]

import mjlab_microduck.hf_jobs as hf_jobs

def fake_submit(argv):
    print("SUBMIT", argv, flush=True)
    return 7

hf_jobs.submit = fake_submit

try:
    {trigger}
except SystemExit as e:
    print("EXIT", e.code, flush=True)
    raise SystemExit(0)

print("NOT-INTERCEPTED", flush=True)
raise SystemExit(1)
"""

_SHIMS = {
    # exactly what a bin/train written by mjlab does
    "mjlab": "from mjlab.scripts.train import main",
    # ... and what one written by this package does
    "ours": "from mjlab_microduck.train_cli import main; main()",
}


@pytest.mark.parametrize("shim", sorted(_SHIMS))
def test_both_train_shims_reach_the_hook_before_parsing_argv(shim):
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE.format(task=_TASK, trigger=_SHIMS[shim])],
        capture_output=True,
        text=True,
        check=False,
        timeout=900,
        cwd=_ROOT,
    )
    assert "NOT-INTERCEPTED" not in proc.stdout, (
        f"the {shim} `train` shim no longer reaches the --hf-jobs interception "
        "(mjlab's plugin loading changed, or it now runs after argv is parsed).\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr[-2000:]}"
    )
    assert proc.returncode == 0, f"probe failed:\n{proc.stderr[-2000:]}"
    assert f"SUBMIT ['{_TASK}', '--env.scene.num-envs', '4096']" in proc.stdout
    assert "EXIT 7" in proc.stdout

"""What trained a run, written when the run starts.

`train` writes `provenance.json` into the run's log directory the moment the runner is built
(tasks/__init__.py), from the git checkout it runs IN — the working directory's repo, which
is a challenges checkout when this package is a dependency — so the commit it names is the
code that trained, whatever the tree looks like when someone publishes weeks later.
`publish --run` reads it back and puts it in the manifest's `training` block.

Inside an HF Job there is no checkout: the tarball has no `.git`. `hf_jobs.submit` hands the
job the checkout it was built from, as JSON in `MICRODUCK_PROVENANCE`, and that is used then.

Standard library only: this runs inside `train`, and must cost nothing there.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import Any

FILE = "provenance.json"
JOB_ENV = "MICRODUCK_PROVENANCE"
_PACKAGE = "mjlab-microduck"


def _git(root: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, check=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def checkout(cwd: Path | None = None) -> dict[str, Any]:
    """`repo`, `commit`, `branch`, `dirty` of the checkout at `cwd`; `{}` outside git."""
    root = Path(cwd or Path.cwd())
    commit = _git(root, "rev-parse", "--short=9", "HEAD")
    if commit is None:
        return {}
    record: dict[str, Any] = {
        "commit": commit,
        "branch": _git(root, "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(_git(root, "status", "--porcelain", "--untracked-files=no")),
    }
    remote = _git(root, "remote", "get-url", "origin")
    if remote:
        record["repo"] = remote
    return record


def contains(cwd: Path | None, commit: str) -> bool:
    """Whether the checkout at `cwd` has `commit` in its history (its HEAD descends from it)."""
    return _git(Path(cwd or Path.cwd()), "merge-base", "--is-ancestor", commit, "HEAD") is not None


def _from_job() -> dict[str, Any]:
    raw = os.environ.get(JOB_ENV)
    return json.loads(raw) if raw else {}


def base() -> str:
    """This package's version, with its git commit when it was installed from git."""
    try:
        dist = distribution(_PACKAGE)
    except PackageNotFoundError:
        return f"{_PACKAGE} (not installed)"
    text = f"{_PACKAGE} {dist.version}"
    raw = dist.read_text("direct_url.json")
    commit = (json.loads(raw).get("vcs_info") or {}).get("commit_id") if raw else None
    return f"{text} @ {commit[:9]}" if commit else text


def record_run_start(
    log_dir: Path, *, argv: list[str], seed: int | None, cwd: Path | None = None
) -> Path | None:
    """Write the run's `provenance.json` once. Returns None when it already exists."""
    path = Path(log_dir) / FILE
    if path.exists():
        return None
    record: dict[str, Any] = {
        # argv[0] is the console script's absolute path; the recipe says `uv run train …`.
        "command": ["train", *argv[1:]],
        "task": argv[1] if len(argv) > 1 else None,
        "seed": seed,
        **(checkout(cwd) or _from_job()),
        "base": base(),
        "started": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n")
    return path


def read(log_dir: Path) -> dict[str, Any]:
    path = Path(log_dir) / FILE
    if not path.exists():
        raise FileNotFoundError(f"{path}: no provenance; this run was not started by `train`")
    return json.loads(path.read_text())

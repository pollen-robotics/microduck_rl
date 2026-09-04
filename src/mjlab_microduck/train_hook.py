"""Keep `train <task> ... --hf-jobs` working, whoever owns the `train` script.

The flag used to live in a `train` console script of our own, declared in
`[project.scripts]` and documented as "shadowing" mjlab's. It shadows nothing:
mjlab 1.3.0 declares `train` too, two distributions declaring the SAME script
name is last-writer-wins at install time, and mjlab won — `uv sync` left
`mjlab.scripts.train:main` in `.venv/bin/train`, so our wrapper was never
invoked and `uv run train ... --hf-jobs` died on tyro's
`Unrecognized options: --hf-jobs` (2026-08-31). Nothing warns about it: the
install succeeds and the flag silently disappears.

So the flag is not implemented in a console script at all any more. It is
intercepted here, from the `mjlab.tasks` plugin entry point: mjlab's own
`mjlab/__init__.py` calls `_import_registered_packages()` at module scope,
which imports `mjlab_microduck.tasks` — and mjlab's `train` reaches that while
executing `from mjlab.scripts.train import main`, i.e. before its two-stage
tyro parse ever sees argv. That path is mjlab's own, so no install order can
take it away from us.

`uv run scripts/hf/train_hf.py <task> ...` calls `submit()` directly and stays
the escape hatch if this interception ever stops firing.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_FLAG = "--hf-jobs"

#: Set on the job's environment by ``hf_jobs.submit`` — inside the job,
#: ``uv run train`` must always mean "train locally".
_IN_JOB_ENV = "MICRODUCK_IN_HF_JOB"


def _invoked_as_train() -> bool:
    """True when argv[0] is mjlab's trainer (console script or `-m`).

    `play --hf-jobs` must NOT submit a training job; let that command's own
    parser reject the flag instead.
    """
    prog = Path(sys.argv[0]).name
    return prog.removesuffix(".py").removesuffix("-script") == "train"


def maybe_submit_to_hf_jobs() -> None:
    """Consume `--hf-jobs` and exit the process; a no-op without the flag.

    Called at import time of `mjlab_microduck.tasks`, so it runs inside mjlab's
    plugin loader. `SystemExit` is a `BaseException`, so it propagates through
    the loader's `except Exception` and out of `import mjlab` — the local
    trainer never starts.
    """
    if _FLAG not in sys.argv[1:]:
        return
    if os.environ.get(_IN_JOB_ENV):
        return
    if not _invoked_as_train():
        return

    from mjlab_microduck.hf_jobs import submit

    sys.exit(submit([a for a in sys.argv[1:] if a != _FLAG]))

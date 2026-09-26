"""Keep `train <task> ... --hf-jobs` / `--modal` working, whoever owns `train`.

The flag used to live in a `train` console script of our own, declared in
`[project.scripts]` and documented as "shadowing" mjlab's. It shadows nothing:
mjlab 1.3.0 declares `train` too, two distributions declaring the SAME script
name is last-writer-wins at install time, and mjlab won — `uv sync` left
`mjlab.scripts.train:main` in `.venv/bin/train`, so our wrapper was never
invoked and `uv run train ... --hf-jobs` died on tyro's
`Unrecognized options: --hf-jobs` (2026-08-31). Nothing warns about it: the
install succeeds and the flag silently disappears.

So the flags are not implemented in a console script at all any more. They
are intercepted here, from the `mjlab.tasks` plugin entry point: mjlab's own
`mjlab/__init__.py` calls `_import_registered_packages()` at module scope,
which imports `mjlab_microduck.tasks` — and mjlab's `train` reaches that while
executing `from mjlab.scripts.train import main`, i.e. before its two-stage
tyro parse ever sees argv. That path is mjlab's own, so no install order can
take it away from us.

One flag per remote backend; each maps to a module exposing
`submit(argv) -> int` that receives argv minus the flag:

    --hf-jobs  -> mjlab_microduck.hf_jobs    (Hugging Face Jobs)
    --modal    -> mjlab_microduck.modal_jobs (Modal)

`uv run scripts/hf/train_hf.py <task> ...` and
`uv run scripts/modal/train_modal.py <task> ...` call `submit()` directly and
stay the escape hatch if this interception ever stops firing.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

_FLAG = "--hf-jobs"  # kept for callers/tests that reference the original flag

#: flag -> module with `submit(argv) -> int`, imported lazily on use.
_SUBMITTERS: dict[str, str] = {
    "--hf-jobs": "mjlab_microduck.hf_jobs",
    "--modal": "mjlab_microduck.modal_jobs",
}

#: Set on the job's environment by every submitter — inside the job,
#: `uv run train` must always mean "train locally".
_IN_JOB_ENV = "MICRODUCK_IN_HF_JOB"


def _invoked_as_train() -> bool:
    """True when argv[0] is mjlab's trainer (console script or `-m`).

    `play --hf-jobs` must NOT submit a training job; let that command's own
    parser reject the flag instead.
    """
    prog = Path(sys.argv[0]).name
    return prog.removesuffix(".py").removesuffix("-script") == "train"


def maybe_submit_to_hf_jobs() -> None:
    """Consume a remote-backend flag and exit the process; a no-op without one.

    Called at import time of `mjlab_microduck.tasks`, so it runs inside mjlab's
    plugin loader. `SystemExit` is a `BaseException`, so it propagates through
    the loader's `except Exception` and out of `import mjlab` — the local
    trainer never starts. (Name kept from when --hf-jobs was the only flag.)
    """
    flags = [f for f in _SUBMITTERS if f in sys.argv[1:]]
    if not flags:
        return
    if os.environ.get(_IN_JOB_ENV):
        return
    if not _invoked_as_train():
        return
    if len(flags) > 1:
        print(f"error: pick one remote backend, not {' and '.join(flags)}", file=sys.stderr)
        sys.exit(2)

    (flag,) = flags
    module = importlib.import_module(_SUBMITTERS[flag])
    sys.exit(module.submit([a for a in sys.argv[1:] if a != flag]))

"""`train` console script — deliberately identical to mjlab's own.

This project must keep DECLARING a `train` script even though it no longer
needs one: `--hf-jobs` is intercepted from the `mjlab.tasks` plugin entry point
(train_hook.py), on mjlab's own import path, so the wrapper has nothing left to
do.

Why keep it anyway: same-name console scripts are last-writer-wins, and after
an install BOTH dist-info RECORDs claim `.venv/bin/train`. Removing this
declaration therefore does not hand the name back to mjlab — it makes the next
`uv sync` UNINSTALL the file (ours, per our RECORD) while nothing reinstalls
mjlab's. `train` then vanishes from the venv entirely and `uv run train` falls
through to whatever `train` sits on PATH: on one machine liblinear's, which
answered `can't open input file Mjlab-Velocity-Flat-MicroDuck` (2026-08-31).

So the collision stays, and is made harmless instead: whichever shim wins,
`train` behaves the same, and the flag is handled in exactly one place.
"""

from __future__ import annotations

import sys


def main() -> int | None:
    # This import runs mjlab's plugin loader, which imports
    # mjlab_microduck.tasks -> train_hook.maybe_submit_to_hf_jobs(). A
    # `--hf-jobs` invocation submits and exits inside the import below; it
    # never comes back here.
    from mjlab.scripts.train import main as mjlab_train_main

    return mjlab_train_main()


if __name__ == "__main__":
    sys.exit(main())

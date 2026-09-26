"""Escape hatch: the submission logic lives in mjlab_microduck.modal_jobs.

Prefer the integrated flag:
    uv run train <task> <train args...> --modal [--gpu L4] [--detach] [...]

This script keeps working if the --modal interception (train_hook.py) ever
stops firing:
    uv run scripts/modal/train_modal.py <task> [submission flags] <train args...>
"""

import sys

from mjlab_microduck.modal_jobs import submit

if __name__ == "__main__":
    sys.exit(submit(sys.argv[1:]))

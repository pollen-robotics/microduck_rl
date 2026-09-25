"""Kept so `uv run python scripts/hf/uploader.py` keeps working; the watcher lives in
mjlab_microduck.hf_uploader, which the HF Jobs bootstrap runs as a module."""

import sys

from mjlab_microduck.hf_uploader import main

if __name__ == "__main__":
    sys.exit(main())

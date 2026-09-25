#!/usr/bin/env python3
"""Kept so every `uv run scripts/infer_policy.py …` in the docs keeps working. The rehearsal
lives in mjlab_microduck.infer and is also `uv run infer`."""

import sys

from mjlab_microduck.infer import main

if __name__ == "__main__":
    sys.exit(main())

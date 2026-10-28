"""Locate a trained FootstepNet model (shared by enjoy.py and export_onnx.py)."""

import re
import sys
from pathlib import Path

import microduck_env


def find_model(arg):
    """A .zip, a run dir, or None (newest logs/crossq/<env>_*/ run): best_model.zip, else the latest
    rl_model_*_steps.zip, else the final model of a finished run."""
    p = Path(arg) if arg else None
    if p is not None and p.is_file():
        return p
    if p is None:
        runs = sorted(Path("logs/crossq").glob(f"{microduck_env.ENV_ID}_*"), key=lambda d: d.stat().st_mtime)
        if not runs:
            sys.exit("no run in logs/crossq/ — pass a model .zip")
        p = runs[-1]
    if (p / "best_model.zip").exists():
        return p / "best_model.zip"
    ckpts = sorted(p.glob("rl_model_*_steps.zip"), key=lambda f: int(re.search(r"_(\d+)_steps", f.name).group(1)))
    if ckpts:
        return ckpts[-1]
    if (p / f"{microduck_env.ENV_ID}.zip").exists():  # final model of a finished run
        return p / f"{microduck_env.ENV_ID}.zip"
    sys.exit(f"no model in {p}")

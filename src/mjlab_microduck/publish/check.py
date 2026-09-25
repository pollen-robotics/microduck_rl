"""`microduck check` — what `publish` verifies, without the Hub.

    microduck check --onnx output.onnx          # 61 -> 14, runs, no NaN, output moves
    microduck check --run logs/rsl_rl/<run>     # provenance.json, latest checkpoint, dirty?
    microduck check --manifest manifest.json    # what the daemon would refuse

Any combination; the first failure ends it with exit 2 and the reason on stderr.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import tyro

from mjlab_microduck import provenance
from mjlab_microduck.publish import manifest as m


@dataclass(frozen=True)
class CheckConfig:
    onnx: str | None = None
    """An exported policy: the shape gate and the smoke run publish applies."""
    run: str | None = None
    """A local run's log directory: its provenance and the checkpoint publish would pick."""
    manifest: str | None = None
    """A manifest.json: validated as the daemon would."""
    smoke: bool = True
    """With --onnx, run the network on plausible inputs (off: shape only)."""


def _fail(msg: str) -> int:
    print(f"[check] error: {msg}", file=sys.stderr)
    return 2


def run(cfg: CheckConfig) -> int:
    if not (cfg.onnx or cfg.run or cfg.manifest):
        return _fail("nothing to check; give --onnx <file>, --run <dir> and/or --manifest <file>")
    try:
        if cfg.onnx:
            shape = m.check_onnx(Path(cfg.onnx))
            print(f"[check] {Path(cfg.onnx).name}: {shape.obs_len} -> {shape.action_len}, ok")
            if cfg.smoke:
                m.smoke_run_onnx(Path(cfg.onnx))
                print("[check] smoke run: finite, non-constant output")
        if cfg.run:
            # Imported here: cli.py pulls tyro's PublishConfig, which check need not pay for.
            from mjlab_microduck.publish.cli import _pick_checkpoint

            run_dir = Path(cfg.run)
            record = provenance.read(run_dir)
            checkpoint = _pick_checkpoint(run_dir, None)
            print(f"[check] {run_dir.name}: task {record.get('task')}, commit {record.get('commit', 'unknown')}, "
                  f"would publish {checkpoint.name}")
            if record.get("dirty"):
                print("[check] started from a checkout with uncommitted changes: publish will refuse it without --allow-dirty")
        if cfg.manifest:
            m.validate_manifest(json.loads(Path(cfg.manifest).read_text()))
            print(f"[check] {Path(cfg.manifest).name}: the daemon would load it")
    except (m.ManifestError, FileNotFoundError) as e:
        return _fail(str(e))
    except SystemExit as e:  # _pick_checkpoint fails through publish's _fail
        return int(e.code or 2)
    return 0


def main() -> int:
    return run(tyro.cli(CheckConfig))


if __name__ == "__main__":
    sys.exit(main())

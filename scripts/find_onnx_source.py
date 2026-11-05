"""Find the wandb checkpoint a deployed ONNX was exported from (bit-exact actor weights).

    uv run scripts/find_onnx_source.py hf://pollen-robotics/microduck-policies/roulade.onnx \\
        --experiment microduck_roulade [--project pollen-robotics/mjlab_microduck]

Scans every run of the experiment (config train_cfg.experiment_name), newest run and newest
checkpoint first, and stops at the first match (each candidate costs one ~5 MB download).
Prints `MATCH <run_path> <iteration>`.
Feed the result to `uv run share-training --wandb-run-path ... --checkpoint ... --onnx <same spec>`.
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
from pathlib import Path

import wandb

from mjlab_microduck.publish.training import _fetch_onnx, _retry, onnx_matches_checkpoint


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx", help="local path or hf://owner/repo/file.onnx")
    ap.add_argument("--experiment", action="append", required=True, help="experiment_name(s) to scan")
    ap.add_argument("--project", default="pollen-robotics/mjlab_microduck")
    ap.add_argument("--max-per-run", type=int, default=60)
    a = ap.parse_args()

    work = Path(tempfile.mkdtemp(prefix="find-onnx-"))
    onnx_path, _ = _fetch_onnx(a.onnx, work)
    api = wandb.Api()
    runs = [r for r in api.runs(a.project, per_page=100)
            if (r.config.get("train_cfg") or {}).get("experiment_name") in a.experiment]
    runs.sort(key=lambda r: r.created_at, reverse=True)
    print(f"[find] {len(runs)} runs of {a.experiment}", flush=True)
    for r in runs:
        path = "/".join(r.path)
        try:
            names = _retry(lambda: [f.name for f in r.files(per_page=50) if re.fullmatch(r"model_\d+\.pt", f.name)], path)
        except RuntimeError as e:
            print(f"[find] {path}: {e}", flush=True)
            continue
        its = sorted((int(n[6:-3]) for n in names), reverse=True)[: a.max_per_run]
        for it in its:
            f = r.file(f"model_{it}.pt")
            d = work / r.id
            try:
                _retry(lambda: f.download(root=str(d), replace=True), f"download {path} {it}")
            except RuntimeError as e:  # a flaky/missing file must not end the whole scan
                print(f"[find] {path} {it}: {e}", flush=True)
                continue
            ck = d / f"model_{it}.pt"
            ok = onnx_matches_checkpoint(onnx_path, ck)
            ck.unlink()
            if ok:
                print(f"MATCH {path} {it}", flush=True)
                return 0
        print(f"[find] {path} ({r.name}): no match in {len(its)} checkpoints", flush=True)
    print("NO MATCH", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())

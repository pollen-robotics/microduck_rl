"""Download a wandb run's full metric history to a local CSV.

We pull run logs frequently when diagnosing training (noise_std drift,
reward collapse, NaN onset, curriculum transitions, ...).  This script
is the canonical way to do it instead of re-writing the wandb.Api
boilerplate each time.

Requires the real wandb client + a valid API key.  In this repo the
Pollen key is loaded via direnv from .envrc, so just run with
`direnv exec . uv run ...` (or from a shell where direnv has loaded).

Examples:
    # Full history → /tmp/wandb_logs/<run_id>.csv
    direnv exec . uv run scripts/fetch_wandb_logs.py \\
        pollen-robotics/mjlab_microduck/ajzu256z

    # Custom output path
    direnv exec . uv run scripts/fetch_wandb_logs.py \\
        pollen-robotics/mjlab_microduck/ajzu256z --out /tmp/myrun.csv

    # Only specific metric keys (faster for big runs)
    direnv exec . uv run scripts/fetch_wandb_logs.py \\
        pollen-robotics/mjlab_microduck/ajzu256z \\
        --keys _step Policy/mean_noise_std Train/mean_reward

    # Print a quick summary of a few key metrics at sampled iters,
    # without writing a CSV (handy for a fast look)
    direnv exec . uv run scripts/fetch_wandb_logs.py \\
        pollen-robotics/mjlab_microduck/ajzu256z --summary
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path


def _parse(v):
    """CSV/wandb value → float | None.  'NaN' string → math.nan."""
    if v is None or v == "":
        return None
    if v == "NaN":
        return math.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


# Metrics shown by --summary.  These are the ones we look at most when
# diagnosing the velocity-sprung task; harmless if a run lacks some
# (they just print as '-').
_SUMMARY_KEYS = [
    "Train/mean_reward",
    "Train/mean_episode_length",
    "Policy/mean_noise_std",
    "Loss/value",
    "Loss/surrogate",
    "Episode_Reward/track_linear_velocity",
    "Episode_Reward/air_time",
    "Episode_Reward/flight_phase",
    "Metrics/flight_phase_frac",
    "Metrics/twist/error_vel_xy",
    "Curriculum/lin_vel_range",
    "Episode_Termination/nan_state",
]


def fetch(
    run_path: str,
    out: Path,
    keys: list[str] | None,
    samples: int | None = None,
) -> list[dict]:
    """Download history, write CSV, return the rows as parsed dicts.

    Two backends:
    - ``samples`` set → ``run.history(samples=N)``: server-side
      downsampled to ~N points.  Fast (seconds) even for runs with
      tens of thousands of steps.  Fills missing keys with NaN, so
      keyed pulls don't return empty (unlike scan_history).  Use this
      for diagnosis — a few thousand points show every trajectory.
    - ``samples`` None → ``run.scan_history()``: every logged point,
      exact.  Can be very slow / effectively hang for long runs
      (60k+ steps × many metrics).  Use only when you truly need
      every step.
    """
    import wandb

    api = wandb.Api()
    run = api.run(run_path)
    last_step = run.summary.get("_step", "?")
    print(f"Run: {run.name}  ({run.state})  last step: {last_step}", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    if samples is not None:
        # Fast downsampled path.  run.history returns a pandas DataFrame;
        # we avoid a hard pandas dep by using pandas=False (list of dicts).
        print(f"Fetching ~{samples} downsampled points "
              f"(keys={'all' if keys is None else len(keys)}) ...", flush=True)
        hist = run.history(
            keys=keys, samples=samples, pandas=False
        )
        print(f"  received {len(hist)} points from wandb, writing CSV ...", flush=True)
        # Determine column order
        if keys is not None:
            cols = ["_step", *[k for k in keys if k != "_step"]]
        else:
            cols = sorted({k for row in hist for k in row.keys()})
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            writer.writeheader()
            for row in hist:
                writer.writerow(row)
                rows.append(row)
        print(f"Wrote {len(rows)} sampled rows ({len(cols)} cols) to {out}", flush=True)
        return rows

    # Exact path (scan_history) — slow for long runs.  WARNING: for runs
    # with 50k+ steps × many metrics this can take many minutes or
    # effectively hang.  Prefer the downsampled path unless you need
    # every step.
    print("Exact scan_history path (every logged step — may be SLOW for "
          "long runs; Ctrl-C and use --samples N if it hangs) ...", flush=True)
    if keys is None:
        sample = next(run.scan_history(page_size=1))
        keys = sorted(sample.keys())
    elif "_step" not in keys:
        keys = ["_step", *keys]
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in run.scan_history(keys=keys, page_size=2000):
            writer.writerow(row)
            rows.append(row)
            if len(rows) % 5000 == 0:
                print(f"  ... {len(rows)} rows", flush=True)
    print(f"Wrote {len(rows)} rows ({len(keys)} metrics) to {out}", flush=True)
    return rows


def print_summary(rows: list[dict], n_samples: int = 12) -> None:
    """Print a few key metrics at evenly-sampled iterations."""
    if not rows:
        print("(no rows)")
        return
    parsed = [{k: _parse(v) for k, v in r.items()} for r in rows]
    parsed.sort(key=lambda r: r.get("_step", 0) or 0)
    n = len(parsed)
    idxs = sorted({0, n - 1, *[round(i * (n - 1) / (n_samples - 1)) for i in range(n_samples)]})

    present = [k for k in _SUMMARY_KEYS if any(k in r for r in parsed)]
    header = "step".rjust(7) + "".join(
        f"  {k.split('/')[-1][:13]:>13}" for k in present
    )
    print(header)
    print("-" * len(header))
    for i in idxs:
        r = parsed[i]
        line = f"{int(r.get('_step', 0) or 0):>7}"
        for k in present:
            v = r.get(k)
            if v is None:
                line += f"  {'-':>13}"
            elif isinstance(v, float) and math.isnan(v):
                line += f"  {'NaN':>13}"
            else:
                try:
                    line += f"  {v:>13.4f}"
                except (TypeError, ValueError):
                    line += f"  {str(v)[:13]:>13}"
        print(line)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_path", help="<entity>/<project>/<run_id>")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output CSV path. Default: /tmp/wandb_logs/<run_id>.csv")
    ap.add_argument("--keys", nargs="+", default=None,
                    help="Only download these metric keys (faster). _step is added automatically.")
    ap.add_argument("--summary", action="store_true",
                    help="Print a sampled summary table after downloading.")
    ap.add_argument("--samples", type=int, default=2000,
                    help="Use run.history server-side downsampling to ~N points "
                         "(fast). Set --samples 0 to force exact scan_history "
                         "(slow, every step). Default 2000.")
    args = ap.parse_args()
    samples = None if args.samples == 0 else args.samples

    run_id = args.run_path.rstrip("/").split("/")[-1]
    out = args.out or Path(f"/tmp/wandb_logs/{run_id}.csv")

    try:
        rows = fetch(args.run_path, out, args.keys, samples=samples)
    except Exception as e:  # noqa: BLE001 — surface any wandb/auth error plainly
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        print("Hint: needs real wandb + API key. Run via `direnv exec . uv run ...` "
              "so .envrc loads WANDB_API_KEY.", file=sys.stderr)
        return 1

    if args.summary:
        print()
        print_summary(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())

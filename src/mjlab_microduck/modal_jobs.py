"""Submit a mjlab-microduck training run to Modal (https://modal.com).

Invoked by the --modal interception (see train_hook.py):

    uv run train Mjlab-Velocity-Flat-MicroDuck \
        --env.scene.num-envs 4096 --agent.max_iterations 5000 --modal

Anything that isn't a submission flag is forwarded verbatim to `uv run train`
inside the container. This is the Modal sibling of hf_jobs.py and keeps its
contract: same source snapshot (tracked files, committed or not), same wandb
key forwarding, same in-job disarm of the interception, same best-effort
auto-export of the final checkpoint to ONNX.

Auth: `uv run modal setup` once (writes ~/.modal.toml). Nothing to configure
in the dashboard: the wandb key travels as a per-run Secret built here.

Image: CUDA runtime base + uv. The dependency set is installed at IMAGE BUILD
time from pyproject.toml + uv.lock, so Modal caches that layer and only
rebuilds it when those two files change. The source tarball is a later,
cheap layer (tar + editable project install). A warm start is seconds; HF
Jobs pays apt + full `uv sync` on every run.

Outputs:
- wandb: metrics + checkpoints every save_interval, as for any run — the
  normal `uv run play/export ... --wandb-run-path <entity/project/run_id>`.
- Modal Volume `microduck-rl-logs`, mounted at /work/logs: the whole
  `logs/rsl_rl/<experiment>/<run>/` tree (checkpoints, params, and
  `exported/policy.onnx` from the auto-export). Browse / fetch with
  `uv run modal volume ls microduck-rl-logs rsl_rl/<experiment>` and
  `uv run modal volume get microduck-rl-logs <path> .`

Container side: Modal's runtime python (the image python, NOT /work/.venv)
imports THIS module to find `_train_in_container`, so the module-level
imports must stay stdlib-only (tests/test_modal_jobs.py locks this). The
training itself is a subprocess, `uv run train ...`, inside /work/.venv.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import shlex
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

DEFAULT_GPU = "L4"
DEFAULT_TIMEOUT_H = 12.0
MAX_TIMEOUT_H = 24.0  # Modal's hard cap on a single function call.
# Same CUDA major as the HF Jobs image (pytorch 2.5.1-cuda12.4); torch's PyPI
# wheel brings its own CUDA libs via nvidia-*-cu12 and warp bundles NVRTC,
# so a runtime (not devel) image is enough — the driver comes from the host.
DEFAULT_BASE_IMAGE = "nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04"
PYTHON = "3.12"  # requires-python is >=3.12,<3.13 (bam pin) — see pyproject.
# Pinned like the HF bootstrap: a floating uv reading a cache written by an
# older one has corrupted installs before (2026-07-21).
UV_VERSION = "0.9.21"
VOLUME_NAME = "microduck-rl-logs"
WORK = "/work"
LOGS = f"{WORK}/logs"
COMMIT_INTERVAL_S = 60.0

#: Set on the container env so the job's own `uv run train` can only ever
#: train locally (train_hook.py). Shared with hf_jobs on purpose: the hook
#: checks one variable, whichever backend set it.
IN_JOB_ENV = "MICRODUCK_IN_HF_JOB"

_CKPT_RE = re.compile(r"model_(\d+)\.pt$")


# --------------------------------------------------------------------------
# Container side (stdlib only at module level — see docstring)
# --------------------------------------------------------------------------


def _pick_checkpoint(logs_root: Path, since: float) -> Path | None:
    """Highest-iteration checkpoint of the run that wrote most recently.

    The volume is shared across runs, so "newest file" alone could pick up a
    previous run if this one saved nothing — hence the `since` cutoff.
    """
    fresh = [
        p
        for p in Path(logs_root, "rsl_rl").glob("*/*/model_*.pt")
        if p.stat().st_mtime >= since
    ]
    if not fresh:
        return None
    run_dir = max(fresh, key=lambda p: p.stat().st_mtime).parent
    return max(
        (p for p in run_dir.glob("model_*.pt") if _CKPT_RE.search(p.name)),
        key=lambda p: int(_CKPT_RE.search(p.name).group(1)),  # type: ignore[union-attr]
    )


def _auto_export(task_id: str, since: float) -> None:
    """Best-effort ONNX export of the final checkpoint while the env is warm.

    Mirrors the HF bootstrap: a separate export job would pay the whole
    container start again for one command. Failure must not fail the run.
    """
    ckpt = _pick_checkpoint(Path(LOGS), since)
    if ckpt is None:
        print("[modal] no checkpoint found, skipping auto-export", flush=True)
        return
    onnx = ckpt.parent / "exported" / "policy.onnx"
    onnx.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "uv", "run", "--no-sync", "python", "scripts/export.py", task_id,
        "--checkpoint-file", str(ckpt), "--num-envs", "1", "--onnx-file", str(onnx),
    ]
    print(f"[modal] auto-exporting {ckpt.name}: {shlex.join(cmd)}", flush=True)
    rc = subprocess.call(cmd, cwd=WORK)
    if rc == 0:
        print(f"[modal] exported {onnx.relative_to(LOGS)} (in volume {VOLUME_NAME})", flush=True)
    else:
        print(f"[modal] auto-export failed with code {rc} (training still OK)", flush=True)


def _train_in_container(task_id: str, train_args: list[str], auto_export: bool) -> int:
    """Runs on the GPU container: train, keep the volume committed, export."""
    import modal  # runtime python has the client; nothing else is importable here

    os.chdir(WORK)
    vol = modal.Volume.from_name(VOLUME_NAME)
    stop = threading.Event()

    def _committer() -> None:
        # Volume writes are only visible outside the container (and survive
        # a killed container) once committed; do it periodically so a
        # timeout or OOM keeps every checkpoint saved so far.
        while not stop.wait(COMMIT_INTERVAL_S):
            try:
                vol.commit()
            except Exception as e:  # noqa: BLE001 — best effort, keep training
                print(f"[modal] volume commit failed: {e}", flush=True)

    threading.Thread(target=_committer, daemon=True).start()

    started = time.time()
    # --no-sync: /work/.venv was built at image-build time with --no-dev; a
    # bare `uv run` would re-sync the dev group into it on every start.
    cmd = ["uv", "run", "--no-sync", "train", task_id, *train_args]
    print(f"[modal] $ {shlex.join(cmd)}", flush=True)
    rc = subprocess.call(cmd, env={**os.environ, IN_JOB_ENV: "1"})
    print(f"[modal] training exited with code {rc}", flush=True)

    if rc == 0 and auto_export:
        _auto_export(task_id, since=started)

    stop.set()
    vol.commit()
    return rc


# --------------------------------------------------------------------------
# Submission side
# --------------------------------------------------------------------------


def _has_logger_flag(train_args: list[str]) -> bool:
    return any(a == "--agent.logger" or a.startswith("--agent.logger=") for a in train_args)


def _app_name(run_name: str) -> str:
    """Modal App names: <=64 chars of [a-z0-9-._]; we use lowercase + dashes."""
    return re.sub(r"[^a-z0-9-]+", "-", f"microduck-rl-{run_name}".lower()).strip("-")[:64]


def _build_image(modal, repo_root: Path, tar_path: Path, base_image: str):
    """Deps layer keyed on pyproject+lock (cached), then the source snapshot."""
    return (
        modal.Image.from_registry(base_image, add_python=PYTHON)
        .apt_install("git", "curl", "ca-certificates")  # git: bam is a git dependency
        .pip_install(f"uv=={UV_VERSION}")
        .env({"UV_PYTHON": PYTHON, "UV_LINK_MODE": "copy"})
        .workdir(WORK)
        .add_local_file(repo_root / "pyproject.toml", f"{WORK}/pyproject.toml", copy=True)
        .add_local_file(repo_root / "uv.lock", f"{WORK}/uv.lock", copy=True)
        # Everything but our own package: this is the expensive, cacheable layer.
        .run_commands("uv sync --frozen --no-dev --no-install-project --no-progress")
        .add_local_file(tar_path, f"{WORK}/src.tar.gz", copy=True)
        .run_commands(
            "tar -xzf src.tar.gz && rm src.tar.gz",
            "uv sync --frozen --no-dev --no-progress",
        )
        # Lets the runtime python import THIS module to find _train_in_container.
        .add_local_python_source("mjlab_microduck")
    )


def submit(argv: list[str]) -> int:
    """Parse submission args from ``argv`` and launch the Modal run."""
    ap = argparse.ArgumentParser(
        prog="train --modal",
        description="Submit a microduck training run to Modal.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,  # never let `--time...` swallow a tyro flag
    )
    ap.add_argument("task", help="mjlab task id, e.g. Mjlab-Velocity-Flat-MicroDuck")
    ap.add_argument(
        "--gpu", default=DEFAULT_GPU,
        help="Modal GPU type: L4 (default), A10G, L40S, A100-40GB, A100-80GB, H100",
    )
    ap.add_argument(
        "--timeout", default=DEFAULT_TIMEOUT_H, type=float,
        help=f"Max run duration in hours (default {DEFAULT_TIMEOUT_H:g}, Modal caps at {MAX_TIMEOUT_H:g})",
    )
    ap.add_argument("--cpu", default=4.0, type=float, help="CPU cores reserved (default 4)")
    ap.add_argument("--memory", default=16384, type=int, help="RAM reserved in MiB (default 16384)")
    ap.add_argument("--base-image", default=DEFAULT_BASE_IMAGE, help="Docker base image")
    ap.add_argument("--volume", default=VOLUME_NAME, help="Modal Volume for logs/ (created if missing)")
    ap.add_argument("--run-name", default=None, help="Short tag for this run; defaults to task+timestamp")
    ap.add_argument("--detach", action="store_true", help="Submit and return immediately (do not stream logs).")
    ap.add_argument("--dry-run", action="store_true", help="Build the tarball and print the spec without submitting.")
    ap.add_argument("--no-export", action="store_true", help="Skip the ONNX auto-export after training.")
    ap.add_argument(
        "--no-wandb", action="store_true",
        help="Do not forward a wandb API key; trains with --agent.logger tensorboard "
             "instead (checkpoints then live ONLY in the Modal volume).",
    )
    args, train_args = ap.parse_known_args(argv)

    if not 0 < args.timeout <= MAX_TIMEOUT_H:
        print(f"error: --timeout must be in (0, {MAX_TIMEOUT_H:g}] hours", file=sys.stderr)
        return 1

    try:
        import modal
    except ImportError:
        print(
            "error: the `modal` client is not installed. It is in the dev dependency "
            "group: run `uv sync` (or `uv sync --group dev`).",
            file=sys.stderr,
        )
        return 1

    # Lazy: hf_jobs imports huggingface_hub, which the container python lacks.
    from mjlab_microduck.hf_jobs import _build_tarball, _repo_root, _wandb_api_key

    env: dict[str, str] = {IN_JOB_ENV: "1"}
    if args.no_wandb:
        if not _has_logger_flag(train_args):
            train_args = [*train_args, "--agent.logger", "tensorboard"]
        print("[wandb] --no-wandb: logging to tensorboard inside the volume")
    else:
        key = _wandb_api_key()
        if not key:
            print(
                "[wandb] ✗ no API key found (checked $WANDB_API_KEY and ~/.netrc).\n"
                "        Run `uv run wandb login`, or pass --no-wandb to log to tensorboard only.",
                file=sys.stderr,
            )
            return 1
        env["WANDB_API_KEY"] = key
        src = "env" if os.environ.get("WANDB_API_KEY") else "~/.netrc"
        print(f"[wandb] forwarding API key from {src}")
        for k in ("WANDB_PROJECT", "WANDB_ENTITY"):
            if os.environ.get(k):
                env[k] = os.environ[k]

    repo_root = _repo_root()
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = args.run_name or f"{args.task}-{stamp}".lower()
    app_name = _app_name(run_name)
    timeout_s = int(args.timeout * 3600)

    with tempfile.TemporaryDirectory() as td:
        # Constant file name: Modal keys the layer on content, so an unchanged
        # tree is a no-op rebuild. The tempdir must outlive the image build,
        # which happens lazily inside app.run().
        tar_path = Path(td) / "src.tar.gz"
        print(f"[src] building tarball (from {repo_root})")
        sha = _build_tarball(repo_root, tar_path)
        print(f"[src] HEAD={sha}, {tar_path.stat().st_size / 1e6:.1f} MB")

        print(f"[modal] app: {app_name}")
        print(f"[modal] gpu: {args.gpu}, cpu: {args.cpu:g}, memory: {args.memory} MiB, timeout: {args.timeout:g}h")
        print(f"[modal] image: {args.base_image} + python {PYTHON} + uv {UV_VERSION}")
        print(f"[modal] volume: {args.volume} -> {LOGS}")
        print(f"[modal] train: uv run train {shlex.join([args.task, *train_args])}")
        print(f"[modal] env: {sorted(k for k in env if k != 'WANDB_API_KEY')}"
              + (" + WANDB_API_KEY (secret)" if "WANDB_API_KEY" in env else ""))
        if args.dry_run:
            print("[dry-run] not submitting")
            return 0

        image = _build_image(modal, repo_root, tar_path, args.base_image)
        vol = modal.Volume.from_name(args.volume, create_if_missing=True)
        app = modal.App(app_name)
        train_fn = app.function(
            image=image,
            gpu=args.gpu,
            cpu=args.cpu,
            memory=args.memory,
            timeout=timeout_s,
            volumes={LOGS: vol},
            secrets=[modal.Secret.from_dict(env)],
        )(_train_in_container)

        # detach=True always: Ctrl-C must detach, never kill a multi-hour run.
        with modal.enable_output(), app.run(detach=True):
            call = train_fn.spawn(args.task, train_args, not args.no_export)
            # An ephemeral app has no resolvable name: `modal app logs <name>`
            # only finds DEPLOYED apps and errors with "No App with name ...".
            # The app id works for both, so hand that out instead.
            app_id = app.app_id or app_name
            print(f"[modal] call id: {call.object_id}")
            print(f"[modal] logs:    uv run modal app logs {app_id}")
            print(f"[modal] stop:    uv run modal app stop {app_id}")
            if args.detach:
                print("[modal] --detach: not streaming logs")
                return 0
            print("[modal] streaming logs (Ctrl-C detaches; the run keeps going)")
            try:
                rc = call.get()
            except KeyboardInterrupt:
                print(f"\n[modal] detached. Run continues: uv run modal app logs {app_id}")
                return 0
    print(f"[modal] {'✓ completed' if rc == 0 else f'✗ training exited with code {rc}'}")
    return int(rc)

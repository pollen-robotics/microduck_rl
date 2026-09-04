"""Submit a mjlab-microduck training run as a Hugging Face Job.

Invoked by the --hf-jobs interception (see train_hook.py):

    uv run train Mjlab-Kick-Flat-MicroDuck \
        --env.scene.num-envs 4096 --agent.max_iterations 4000 --hf-jobs

Anything that isn't an --hf-* / submission flag is forwarded verbatim to
`uv run train` inside the job.

Auth: the cached HF token from `hf auth login` / HF_TOKEN env. The account's
orgs are listed at submission and you pick the namespace to run under
(personal or org) — repos, uv-cache bucket and the job itself all live in
that namespace. Pass --namespace to skip the prompt (automation).

Everything goes through the huggingface_hub Python API (Jobs API, hub >= 1.x)
— the standalone `hf` CLI is NOT required.

Source: a snapshot of tracked files (committed or not) is uploaded to a
private HF dataset repo and mounted read-only inside the job. Checkpoints are
pushed by a watcher running alongside training (scripts/hf/uploader.py) to a
private HF model repo.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import shlex
import subprocess
import sys
import tarfile
import tempfile
import time
from netrc import netrc
from pathlib import Path

from huggingface_hub import HfApi, Volume, get_token

DEFAULT_IMAGE = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"
DEFAULT_FLAVOR = "l4x1"
DEFAULT_TIMEOUT = "12h"

# Bootstrap script run inside the container. `$VAR` is expanded by the
# container shell from the job's env vars.
BOOTSTRAP = r"""
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -qq -y --no-install-recommends git curl ca-certificates xz-utils >/dev/null
# Pinned uv: the cache bucket persists across jobs, and a floating "latest" uv
# reading entries written by an older uv corrupts installs (seen 2026-07-21:
# bam's built-wheel cache entry from a 0.9.x-era job made 0.11.30 fail with
# "The wheel is invalid: Missing .dist-info directory").
curl -LsSf https://astral.sh/uv/0.11.30/install.sh | sh >/dev/null
export PATH="/root/.local/bin:$PATH"
# HF Jobs met le cache uv et /work sur des FS différents -> uv ne peut pas
# hardlink et son fallback corrompt les wheels construits (bam -> "Missing
# .dist-info directory"). copy = install fiable (le remède que uv suggère).
export UV_LINK_MODE=copy

mkdir -p /work && cd /work
echo "[bootstrap] extracting source $SRC_TARBALL"
tar -xzf "/src/$SRC_TARBALL"

echo "[bootstrap] uv sync"
# Self-heal a poisoned persistent cache: a bad entry fails sync
# deterministically, so nuke the cache and rebuild it once before giving up.
uv sync --no-progress || {
    echo "[bootstrap] uv sync failed — cleaning uv cache and retrying"
    uv cache clean || true
    uv sync --no-progress
}

echo "[bootstrap] launching checkpoint uploader"
mkdir -p logs/rsl_rl
nohup uv run python scripts/hf/uploader.py > /tmp/uploader.log 2>&1 &
UPLOADER_PID=$!

echo "[bootstrap] starting training: uv run train $TRAIN_ARGS"
set +e
uv run train $TRAIN_ARGS
TRAIN_RC=$?
set -e

echo "[bootstrap] training exited with code $TRAIN_RC, final upload pass"
# kill watcher loop, then run one final upload pass synchronously
kill $UPLOADER_PID 2>/dev/null || true
CKPT_ONE_SHOT=1 uv run python scripts/hf/uploader.py || true

# Auto-export the final checkpoint to daemon-ready ONNX while the env is
# still warm — a separate export job would pay the full bootstrap (image
# pull + apt + uv sync) again just to run this one command. Best-effort:
# an export failure must not mark a successful training as failed.
if [ "$TRAIN_RC" -eq 0 ] && [ "${AUTO_EXPORT:-1}" = "1" ]; then
    set +e
    TASK_ID=${TRAIN_ARGS%% *}
    CKPT=$(ls -t logs/rsl_rl/*/model_*.pt 2>/dev/null | head -1)
    if [ -n "$CKPT" ]; then
        echo "[bootstrap] auto-exporting ONNX from $(basename "$CKPT")"
        uv run python scripts/export.py "$TASK_ID" \
            --checkpoint-file "$(basename "$CKPT")" \
            --num-envs 1 --onnx-file /work/policy.onnx \
        && uv run python - <<'PY'
import os
from huggingface_hub import HfApi
HfApi().upload_file(path_or_fileobj="/work/policy.onnx",
                    path_in_repo="exported/policy.onnx",
                    repo_id=os.environ["CKPT_REPO"], repo_type="model")
print("[bootstrap] uploaded exported/policy.onnx")
PY
        [ $? -ne 0 ] && echo "[bootstrap] auto-export failed (training still OK)"
    else
        echo "[bootstrap] no checkpoint found, skipping auto-export"
    fi
    set -e
fi

exit $TRAIN_RC
"""


def _wandb_api_key() -> str | None:
    """Best-effort lookup of the user's wandb API key.

    Order: WANDB_API_KEY env -> ~/.netrc (machine api.wandb.ai).
    """
    if k := os.environ.get("WANDB_API_KEY"):
        return k
    try:
        n = netrc(str(Path.home() / ".netrc"))
        auth = n.authenticators("api.wandb.ai")
        if auth and auth[2]:
            return auth[2]
    except (FileNotFoundError, OSError):
        pass
    return None


def _repo_root() -> Path:
    """Repo root of the CURRENT directory — worktree-aware.

    (The old scripts/hf/train_hf.py used the script file's location; resolving
    from cwd instead means running from a worktree snapshots that worktree.)
    """
    out = subprocess.check_output(["git", "rev-parse", "--show-toplevel"])
    return Path(out.decode().strip())


def _build_tarball(repo_root: Path, out_path: Path) -> str:
    """Create a tarball of HEAD + uncommitted tracked changes. Returns short SHA."""
    sha = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=repo_root
    ).decode().strip()

    # Use `git ls-files` so we include tracked-but-modified files (working tree
    # state) but skip ignored junk (.venv, logs, *.onnx, wandb/, etc.).
    files = subprocess.check_output(
        ["git", "ls-files", "-co", "--exclude-standard"], cwd=repo_root
    ).decode().splitlines()

    with tarfile.open(out_path, "w:gz") as tar:
        for rel in files:
            p = repo_root / rel
            if p.exists() and p.is_file():
                tar.add(p, arcname=rel)
    return sha


def _pick_namespace(api: HfApi, preset: str | None) -> str:
    """Choose the namespace (personal account or org) the job runs under.

    Interactive prompt unless --namespace was given or there is nothing to
    choose. Non-tty (scripts, CI) falls back to the personal account.
    """
    info = api.whoami()
    user = info.get("name") or info.get("email")
    if not user:
        raise RuntimeError("Could not determine HF username. Run `hf auth login` first.")
    orgs = [o["name"] for o in info.get("orgs", []) if o.get("name")]

    if preset is not None:
        if preset not in (user, *orgs):
            raise RuntimeError(
                f"--namespace {preset!r} is neither your account ({user}) "
                f"nor one of your orgs ({', '.join(orgs) or 'none'})."
            )
        return preset

    if not orgs:
        return user

    if not sys.stdin.isatty():
        print(f"[hf] non-interactive, defaulting to personal namespace: {user}")
        return user

    choices = [user, *orgs]
    print("[hf] run under which namespace?")
    print(f"  1) {user} (personal)")
    for i, org in enumerate(orgs, start=2):
        print(f"  {i}) {org}")
    while True:
        raw = input(f"Choice [1-{len(choices)}, default 1]: ").strip()
        if raw == "":
            return user
        if raw.isdigit() and 1 <= int(raw) <= len(choices):
            return choices[int(raw) - 1]
        if raw in choices:
            return raw
        print(f"  invalid choice: {raw!r}")


def _await_scheduling(
    api: HfApi, job_id: str, namespace: str, budget_s: float = 1200.0
) -> tuple[str, str | None]:
    """Poll until the job leaves SCHEDULING (image pull / queue / mounts).

    This phase legitimately takes minutes (the pytorch image pull alone is
    ~5 min on a cold GPU node) and is also where volume-mount failures
    surface, ~7 min in ("init container exhausted retries"). Returns
    (stage, message) at the first non-SCHEDULING stage, or the last observed
    one when the budget runs out (a long queue is not an error — the caller's
    streaming loop keeps supervising).
    """
    deadline = time.monotonic() + budget_s
    last_note = time.monotonic()
    stage, message = "", None
    while time.monotonic() < deadline:
        status = api.inspect_job(job_id=job_id, namespace=namespace).status
        stage, message = status.stage, status.message
        if stage and stage != "SCHEDULING":
            return stage, message
        if time.monotonic() - last_note > 60:
            print("[job] still scheduling (queue / image pull / volume mounts)...")
            last_note = time.monotonic()
        time.sleep(10)
    return stage, message


def submit(argv: list[str]) -> int:
    """Parse submission args from ``argv`` and launch the HF job."""
    ap = argparse.ArgumentParser(
        prog="train --hf-jobs",
        description="Submit a microduck training run to HF Jobs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("task", help="mjlab task id, e.g. Mjlab-Kick-Flat-MicroDuck")
    ap.add_argument("--flavor", default=DEFAULT_FLAVOR, help="HF Jobs hardware flavor")
    ap.add_argument("--image", default=DEFAULT_IMAGE, help="Docker image to run in")
    ap.add_argument("--timeout", default=DEFAULT_TIMEOUT, help="Job max duration")
    ap.add_argument(
        "--namespace",
        default=None,
        help="HF namespace (your username or an org) to run under; skips the prompt.",
    )
    ap.add_argument(
        "--run-name",
        default=None,
        help="Short tag for this run; defaults to task+timestamp",
    )
    ap.add_argument(
        "--detach", action="store_true",
        help="Submit and return immediately (do not stream logs).",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Build tarball and print the job spec without submitting.",
    )
    ap.add_argument(
        "--src-repo",
        default=None,
        help="HF dataset repo for source tarballs. Defaults to <namespace>/mjlab-microduck-src",
    )
    ap.add_argument(
        "--ckpt-repo",
        default=None,
        help="HF model repo for checkpoints. Defaults to <namespace>/<run-name>",
    )
    ap.add_argument(
        "--uv-cache-bucket",
        default=None,
        help="HF bucket used as UV_CACHE_DIR to persist wheels across runs. "
             "Defaults to <namespace>/mjlab-uv-cache. Requires --uv-cache.",
    )
    ap.add_argument(
        "--uv-cache", action="store_true",
        help="Mount a persistent uv cache bucket (OFF by default: the FUSE "
             "bucket mount does not support hardlinks, so `uv sync` falls back "
             "to full-copying ~6 GB of unpacked packages through the network "
             "mount — far slower than just re-downloading wheels from PyPI, "
             "which HF's datacenter bandwidth handles in ~1 min; it also "
             "poisons across uv versions: a 0.9.x-era entry made 0.11.30 die "
             "with 'wheel is invalid: Missing .dist-info', 2026-07-21).",
    )
    ap.add_argument(
        "--no-wandb", action="store_true",
        help="Do not forward a wandb API key (training will fail if wandb is enabled).",
    )
    args, train_args = ap.parse_known_args(argv)

    api = HfApi()
    try:
        namespace = _pick_namespace(api, args.namespace)
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"[hf] namespace: {namespace}")

    token = get_token()
    if not token:
        print("error: no cached HF token. Run `hf auth login`.", file=sys.stderr)
        return 1

    repo_root = _repo_root()
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = args.run_name or f"{args.task}-{stamp}".lower()
    src_repo = args.src_repo or f"{namespace}/mjlab-microduck-src"
    ckpt_repo = args.ckpt_repo or f"{namespace}/{run_name}"

    env: dict[str, str] = {
        # Disarms the --hf-jobs interception inside the job, so the job's own
        # `uv run train` can only ever train locally (see train_hook.py).
        "MICRODUCK_IN_HF_JOB": "1",
        "CKPT_REPO": ckpt_repo,
        "TRAIN_ARGS": " ".join(shlex.quote(a) for a in [args.task, *train_args]),
    }
    secrets: dict[str, str] = {"HF_TOKEN": token}

    # Forward wandb credentials (env var, then ~/.netrc)
    if not args.no_wandb:
        wb_key = _wandb_api_key()
        if not wb_key:
            print(
                "[wandb] ✗ no API key found (checked $WANDB_API_KEY and ~/.netrc).\n"
                "        Run `wandb login` locally, or pass --no-wandb to skip.",
                file=sys.stderr,
            )
            return 1
        secrets["WANDB_API_KEY"] = wb_key
        src = "env" if os.environ.get("WANDB_API_KEY") else "~/.netrc"
        print(f"[wandb] forwarding API key from {src}")
        for k in ("WANDB_PROJECT", "WANDB_ENTITY"):
            if os.environ.get(k):
                env[k] = os.environ[k]

    volumes = [Volume(type="dataset", source=src_repo, mount_path="/src", read_only=True)]

    # Persistent uv cache — opt-in via --uv-cache (see the flag's help text:
    # cross-filesystem installs from the FUSE bucket are slower than fresh
    # PyPI downloads, and a stale entry deterministically killed uv sync on
    # 2026-07-21).
    cache_bucket: str | None = None
    if args.uv_cache:
        cache_bucket = args.uv_cache_bucket or f"{namespace}/mjlab-uv-cache"
        volumes.append(Volume(type="bucket", source=cache_bucket, mount_path="/uv-cache"))
        env["UV_CACHE_DIR"] = "/uv-cache"
        print(f"[uv-cache] using bucket {cache_bucket}")

    # 1. Build tarball
    with tempfile.TemporaryDirectory() as td:
        tar_path = Path(td) / f"src-{stamp}.tar.gz"
        print(f"[src] building tarball -> {tar_path.name} (from {repo_root})")
        sha = _build_tarball(repo_root, tar_path)
        size_mb = tar_path.stat().st_size / 1e6
        print(f"[src] HEAD={sha}, {size_mb:.1f} MB")
        env["SRC_TARBALL"] = tar_path.name
        env["GIT_SHA"] = sha

        if args.dry_run:
            print("[dry-run] would submit job:")
            print(f"  namespace: {namespace}")
            print(f"  image:     {args.image}")
            print(f"  flavor:    {args.flavor}, timeout: {args.timeout}")
            print(f"  volumes:   {[f'{v.type}:{v.source} -> {v.mount_path}' for v in volumes]}")
            print(f"  env:       { {k: v for k, v in env.items()} }")
            print(f"  secrets:   { {k: '***' for k in secrets} }")
            print(f"  ckpt repo: https://huggingface.co/{ckpt_repo}")
            return 0

        # 2. Upload tarball + pre-create repos/bucket
        api.create_repo(src_repo, repo_type="dataset", private=True, exist_ok=True)
        print(f"[src] uploading to dataset {src_repo}")
        api.upload_file(
            path_or_fileobj=str(tar_path),
            path_in_repo=tar_path.name,
            repo_id=src_repo,
            repo_type="dataset",
        )
        api.create_repo(ckpt_repo, repo_type="model", private=True, exist_ok=True)
        if cache_bucket is not None:
            api.create_bucket(cache_bucket, private=True, exist_ok=True)

        # 3. Submit. GPU-node volume mounts fail transiently ("init container
        # exhausted retries", surfacing ~7 min into SCHEDULING — observed
        # 2026-07 while an identical probe job mounted fine at the same time).
        # Supervise the scheduling phase and resubmit on a mount failure.
        print(f"[ckpt] checkpoints -> https://huggingface.co/{ckpt_repo}")
        print(f"[job] submitting (namespace={namespace}, flavor={args.flavor}, timeout={args.timeout})")
        job = None
        stage, message = "", None
        for attempt in range(3):
            if attempt:
                print(f"[job] ✗ volume mount failed (flaky node) — resubmitting ({attempt + 1}/3)")
                time.sleep(10)
            try:
                job = api.run_job(
                    image=args.image,
                    command=["bash", "-c", BOOTSTRAP],
                    env=env,
                    secrets=secrets,
                    flavor=args.flavor,
                    timeout=args.timeout,
                    volumes=volumes,
                    namespace=namespace,
                )
            except Exception as e:
                msg = str(e)
                if "402" in msg or "Payment Required" in msg or "credit" in msg.lower():
                    print(
                        "\n[job] ✗ Hugging Face Jobs billing error.\n"
                        f"    The namespace {namespace!r} has insufficient Jobs credits.\n"
                        "    → Add credits:   https://huggingface.co/settings/billing\n"
                        "    → Or get HF Pro: https://huggingface.co/settings/billing/subscription",
                        file=sys.stderr,
                    )
                    return 2
                if "403" in msg or "Forbidden" in msg or "required permissions" in msg.lower():
                    print(
                        "\n[job] ✗ Hugging Face Jobs permission error (403).\n"
                        "    Your HF token authenticates fine but is NOT allowed to use the Jobs API\n"
                        f"    for namespace {namespace!r}. This is a token-scope problem, not billing.\n"
                        "    → Create/edit a fine-grained token WITH the Jobs permission enabled:\n"
                        "        https://huggingface.co/settings/tokens\n"
                        "      (fine-grained → under your user AND/OR the org, tick the 'Jobs' permissions),\n"
                        "      then re-login locally:  hf auth login\n"
                        "    → Verify:  python -c \"from huggingface_hub import HfApi; \"\n"
                        "               \"print(list(HfApi().list_jobs(namespace='<ns>')))\"  (must not 403)\n"
                        "    (If Jobs are enabled but still blocked, the namespace may also need an HF\n"
                        "     plan/credits that include Jobs — see the billing link above.)",
                        file=sys.stderr,
                    )
                    return 3
                raise
            print(f"[job] id:  {job.id}")
            if getattr(job, "url", None):
                print(f"[job] url: {job.url}")
            if args.detach:
                print("[job] --detach: not supervising startup — check the URL above; "
                      "transient 'Volume mount failed' errors need a manual resubmit.")
                return 0
            stage, message = _await_scheduling(api, job.id, namespace)
            if stage == "ERROR" and "mount" in (message or "").lower():
                continue  # flaky node — resubmit
            break

    assert job is not None
    if stage == "ERROR":
        print(f"[job] ✗ failed to start: {message}", file=sys.stderr)
        return 1

    # Supervise to completion: stream logs, re-attach if the stream drops
    # (it returns empty while the container is still starting), and report
    # the terminal status.
    print("[job] streaming logs (Ctrl-C detaches; the job keeps running)")
    try:
        while True:
            try:
                for line in api.fetch_job_logs(job_id=job.id, namespace=namespace, follow=True):
                    print(line)
            except Exception as e:
                print(f"[job] log stream dropped ({e}); re-attaching")
            status = api.inspect_job(job_id=job.id, namespace=namespace).status
            if status.stage == "COMPLETED":
                print("[job] ✓ completed")
                return 0
            if status.stage in ("ERROR", "DELETED", "CANCELED"):
                print(f"[job] ✗ {status.stage}: {status.message}", file=sys.stderr)
                return 1
            time.sleep(10)
    except KeyboardInterrupt:
        print(f"\n[job] detached. Job {job.id} is still running: {getattr(job, 'url', job.id)}")
        return 0

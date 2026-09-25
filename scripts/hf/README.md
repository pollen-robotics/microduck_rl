# HF Jobs training

Train mjlab-microduck on Hugging Face's managed GPUs. Auth is the cached HF
token (`hf auth login` or `HF_TOKEN`); everything goes through the
`huggingface_hub` Python API — the standalone `hf` CLI is not required.

## One-time setup

```fish
hf auth login    # or export HF_TOKEN (any tool that caches the token works)
wandb login      # auto-detected from ~/.netrc and forwarded
```

## Submit a run

Your normal train command, plus `--hf-jobs`:

```fish
uv run train Mjlab-Kick-Flat-MicroDuck \
    --env.scene.num-envs 4096 --agent.max_iterations 4000 --hf-jobs
```

You'll be asked which namespace to run under — your personal account or one
of your orgs. Repos, uv-cache bucket, billing and the job itself all live in
the chosen namespace. Pass `--namespace <name>` to skip the prompt
(non-interactive runs default to personal).

Without `--hf-jobs` the command behaves exactly as before (local training).
Submission flags are consumed locally; everything else is forwarded to
`uv run train` inside the job.

Useful flags:
- `--namespace <name>` — account/org to run under; skips the prompt
- `--flavor l4x1` (default) / `a10g-large` / `a100-large`
- `--timeout 12h` (default) — job is killed past this
- `--detach` — submit and return immediately (default streams logs; Ctrl-C detaches without killing the job)
- `--dry-run` — build tarball, print the job spec, do not submit
- `--run-name <tag>` — overrides the auto-generated `<task>-<timestamp>` name
- `--no-uv-cache` — disable the persistent `uv` cache bucket (first-run cost on every run)
- `--no-wandb` — don't forward a wandb key

(`uv run scripts/hf/train_hf.py <task> ...` still works — it's a shim to the
same code, which lives in `src/mjlab_microduck/hf_jobs.py`.)

## What happens under the hood

1. `git ls-files` snapshots tracked + uncommitted files of the repo you run
   from (worktree-aware) → `src-<stamp>.tar.gz`.
2. Tarball is uploaded to private dataset `<namespace>/mjlab-microduck-src`.
3. Private model repo `<namespace>/<run-name>` is created for checkpoints.
4. A private HF bucket `<namespace>/mjlab-uv-cache` is mounted at `/uv-cache`
   and used as `UV_CACHE_DIR` so wheel downloads persist across runs (first
   run cold, subsequent runs fast).
5. `HfApi.run_job` launches a container that:
   - installs `uv`, extracts the tarball, runs `uv sync` (warm-cached),
   - starts the checkpoint uploader (`python -m mjlab_microduck.hf_uploader`) in background (watches `logs/rsl_rl/**/model_*.pt`, pushes every 60s),
   - runs `uv run train <task> <args>`,
   - does a final one-shot upload on exit.
6. wandb credentials are forwarded as a secret — runs show up live in your
   wandb project.

## Browsing checkpoints

The submitter prints `https://huggingface.co/<namespace>/<run-name>` at
start; new `.pt` files appear there during training.

## Managing jobs

The job id and URL are printed at submission. From Python:

```python
from huggingface_hub import HfApi
api = HfApi()
api.list_jobs()                                  # or namespace="pollen-robotics"
for l in api.fetch_job_logs(job_id="...", follow=True): print(l)
api.cancel_job(job_id="...")
```

(or the `hf jobs ps/logs/cancel` CLI if you have it installed.)

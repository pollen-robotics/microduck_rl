# Modal training

Train mjlab-microduck on [Modal](https://modal.com) GPUs. Sibling of the
[HF Jobs path](../hf/README.md): same source snapshot, same wandb forwarding,
same auto-export — but the dependency install is baked into a cached image,
so a warm start is seconds instead of minutes.

## One-time setup

```bash
uv sync                  # the `modal` client is in the dev dependency group
uv run modal setup       # browser login, writes ~/.modal.toml
uv run wandb login       # auto-detected from ~/.netrc and forwarded
```

## Submit a run

Your normal train command, plus `--modal`. Smoke test first (cents, ~5 min
cold for the first image build, then seconds):

```bash
uv run train Mjlab-Velocity-Flat-MicroDuck \
    --env.scene.num-envs 64 --agent.max_iterations 5 --modal

uv run train Mjlab-Velocity-Flat-MicroDuck \
    --env.scene.num-envs 4096 --agent.max_iterations 5000 --modal
```

Without `--modal` the command behaves exactly as before (local training).
Submission flags are consumed locally; everything else is forwarded to
`uv run train` inside the container.

Useful flags:
- `--gpu L4` (default) / `A10G` / `L40S` / `A100-40GB` / `A100-80GB` / `H100`
- `--timeout 12` — hours; the run is killed past this (Modal caps at 24)
- `--detach` — submit and return (default streams logs; Ctrl-C detaches without killing the run)
- `--dry-run` — build the tarball, print the spec, do not submit
- `--run-name <tag>` — overrides the auto-generated `<task>-<timestamp>` name (it becomes the Modal app name)
- `--no-wandb` — don't forward a wandb key; logs to tensorboard in the volume instead
- `--no-export` — skip the ONNX auto-export after training
- `--cpu 4`, `--memory 16384` — reserved cores / MiB

(`uv run scripts/modal/train_modal.py <task> ...` still works — it's a shim to
the same code, which lives in `src/mjlab_microduck/modal_jobs.py`.)

## What happens under the hood

1. `git ls-files` snapshots tracked + uncommitted files of the repo you run
   from → `src.tar.gz`.
2. Image = `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` + python 3.12 + uv,
   then `uv sync --frozen --no-dev --no-install-project` from
   `pyproject.toml` + `uv.lock` (cached until those change), then the
   tarball and a final `uv sync` for the editable project.
3. A Modal Volume `microduck-rl-logs` is mounted at `/work/logs` and
   committed every 60 s, so checkpoints survive a timeout or a crash.
4. The container runs `uv run train <task> <args>`; wandb credentials ride
   along as a per-run Secret, so runs show up live in your wandb project.
5. On success the newest checkpoint is exported to
   `logs/rsl_rl/<experiment>/<run>/exported/policy.onnx` in the volume.

## Checkpoints and logs

Checkpoints go to wandb as usual (`play` / `export.py` / `publish` with
`--wandb-run-path`). The volume holds the same files plus the exported ONNX:

```bash
uv run modal volume ls microduck-rl-logs rsl_rl/velocity
uv run modal volume get microduck-rl-logs rsl_rl/velocity/<run>/exported/policy.onnx .
```

## Managing runs

Training runs are *ephemeral* apps, and `modal app logs <name>` resolves names
only for DEPLOYED apps — passing `microduck-rl-<run-name>` errors with "No App
with name ... found". Use the app id (`ap-...`), printed at submission next to
the call id and listed under App ID by `modal app list`:

```bash
uv run modal app list                             # App ID column; ours is "ephemeral"
uv run modal app logs ap-xxxxxxxxxxxx             # last 100 entries, then exits
uv run modal app logs ap-xxxxxxxxxxxx -f          # live stream instead
uv run modal app stop ap-xxxxxxxxxxxx             # kills the run
```

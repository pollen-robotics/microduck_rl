# Training checkpoints on the Hub (training bundles)

Deployable policies are ONNX files in their own Hub repos (`uv run publish`, or the hand-curated
set `pollen-robotics/microduck-policies`). Those can be RUN, not trained further: the ONNX has no
critic, optimizer or curriculum state. A **training bundle** is the full rsl_rl checkpoint plus
everything needed to reproduce or continue it, in a separate repo so the ONNX repos stay small.

Sharing is **opt-in**: nothing is uploaded unless you run `share-training` or pass
`--share-training` to `publish`. Bundle repos are created **private** unless you say `--public`.

## Layout

One repo per owner, default `<owner>/microduck-training` (Pollen: `pollen-robotics/microduck-training`):

```
README.md                                   index of every bundle (regenerated on each share)
bundles/<experiment>/<run_id>_<iteration>/
  model_<iteration>.pt    full checkpoint: actor + critic + obs normalizers + optimizer + iteration
                          + common_step_counter (so a resume keeps every curriculum where it was)
  bundle.json             machine-readable provenance (below)
  wandb_config.yaml       the run's resolved env + agent cfg
  uncommitted.diff        only when the run had uncommitted changes (apply on top of the commit)
  README.md               resume / warm-start / export commands for this bundle
```

`bundle.json` (schema 1): `id`, `task_id`, `experiment_name`, `iteration`, `checkpoint_file`,
`checkpoint_sha256`, `source.wandb_run`, `code.{repo, commit, on_remote, uncommitted_diff}`,
`command` (the exact `uv run train ...` argv), `loaded_from` (the checkpoint the run resumed from,
with its `bundle` when that one is shared too), `teachers` (distillation experts from the run's own
`bc_cfg`, each with its `bundle`), `onnx[]` (`repo`, `file`, `revision`, `sha256`, verified bit-exact
against the actor weights).

Bundles are immutable: same id + different checkpoint bytes is refused. Re-sharing an existing
bundle only merges new `onnx` records. Teachers shared by many runs are uploaded once.

## Share

```bash
# while publishing a policy (exports, uploads the ONNX repo, AND bundles the checkpoint + teachers;
# the policy's manifest gets training.bundle and its README the resume command)
uv run publish --task <TASK> --wandb-run-path <entity/project/run> --checkpoint N \
    --repo <owner>/microduck-<name> --kind episodic --duration-s 4.0 --share-training

# any checkpoint, after the fact; --onnx records (after verifying) what it was deployed as
uv run share-training --wandb-run-path <entity/project/run> --checkpoint N \
    [--onnx hf://<owner>/<policy-repo>/<file>.onnx] [--tag <name>-vX] [--public] [--dry-run]

# a local checkpoint (no wandb): provenance = this checkout's HEAD; refuses a dirty tree
uv run share-training --checkpoint-file logs/rsl_rl/<exp>/<run>/model_N.pt --task <TASK>
```

Everything else is discovered from the wandb run: task id and parent checkpoint from its command
line, commit from its git metadata, uncommitted diff from `microduck_rl.diff`, teachers from
`train_cfg.algorithm.bc_cfg` (`*wandb_run_path` + `*checkpoint_name` pairs).

When you update a policy in `pollen-robotics/microduck-policies` (hand-curated, no `publish`),
share the checkpoint with `--onnx hf://pollen-robotics/microduck-policies/<file>.onnx` and tag the
training repo like the policy set (`--tag velstand-v9`).

Every ONNX exported by `scripts/export.py` / `publish` (from 2026-10-27) carries its provenance in
the ONNX metadata: `run_path` (wandb run or checkpoint spec), `task_id`, `checkpoint` (iteration),
`checkpoint_sha256` (equal to the bundle's `checkpoint_sha256`), `export_commit`/`export_dirty`
(the exporting checkout; the training commit is in the bundle). Read it with
`onnx.load(f).metadata_props`. Earlier exports have only `run_path = "None"`.

Lost provenance of an older ONNX? `uv run scripts/find_onnx_source.py <onnx> --experiment <exp>`
scans that experiment's wandb runs for the bit-exact checkpoint.

## Use

A spec is a local path or `hf://<owner>/<repo>[@<revision>]/<path>`, where `<path>` is a `.pt` or a
bundle directory. It works everywhere a checkpoint is read:

```bash
# resume (same task: iteration and curricula continue)
uv run train <TASK> --env.scene.num-envs 4096 \
    --init-checkpoint hf://pollen-robotics/microduck-training/bundles/velstand/0045i2si_2749

# warm start into another task (weights/normalizer/optimizer kept, curricula restart at 0)
MICRODUCK_WARM_START=1 uv run train <OTHER_TASK> --init-checkpoint hf://...

# export to ONNX (normalizer baked in), or publish from it
uv run scripts/export.py <TASK> --checkpoint-file hf://...
```

`--init-checkpoint` is consumed by the train hook (`train_hook.maybe_init_checkpoint`, after the
`--hf-jobs` check, so HF Jobs submissions forward it and the job resolves it with its own token).
It replaces `--agent.resume` + `--wandb-run-path`; giving both is an error. Check out the bundle's
`code.commit` first: a checkpoint only resumes cleanly under the obs/action layout and curricula it
was trained with.

Distillation teachers (velstand's `bc_cfg`) resolve in order `checkpoint_path` → `hf_checkpoint`
(bundle spec) → `wandb_run_path`, falling back to wandb when the Hub fetch fails. Setting
`hf_checkpoint` / `anchor_hf_checkpoint` to the teacher bundles is what lets someone without access
to the Pollen wandb project train velstand.

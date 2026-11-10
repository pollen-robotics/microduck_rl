# Training and simulation troubleshooting

Use the checks below to locate a failure in setup, training, checkpoint loading,
or policy replay. Run commands from the repository root. Start with a small
training run before committing to a longer experiment.

| Symptom | Start here |
|---|---|
| CUDA is unavailable, or training fails before iteration 0 | [Check the environment](#check-the-environment) |
| `train` is missing or reports `can't open input file` | [Check the command](#check-the-command) |
| Training runs but falls, stalls, or produces non-finite values | [Inspect training behaviour](#inspect-training-behaviour) |
| Simulation is slow or runs out of GPU memory | [Check simulation performance](#check-simulation-performance) |
| A checkpoint cannot be found or has incompatible shapes | [Load the matching checkpoint](#load-the-matching-checkpoint) |
| ONNX replay expects 61 inputs but receives 51 | [Replay the exported policy](#replay-the-exported-policy) |

## Check the environment

Install the locked dependencies with `uv sync --frozen`, then inspect the
environment used by the project:

```bash
uv run python - <<'PY'
import platform
import sys
import torch

print("Python:", sys.version.split()[0])
print("Platform:", platform.system(), platform.machine())
print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("CUDA devices:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

The training path in this repository uses MuJoCo Warp and a CUDA GPU. CPU ONNX
replay and CPU configuration tests are separate ways to use a machine without
CUDA. Apple MPS availability does not provide a CUDA device.

On a machine expected to have an NVIDIA GPU, also run `nvidia-smi`. If that fails,
check the host driver and whether the GPU is exposed to the container or notebook
runtime. If it works but PyTorch reports no CUDA devices, inspect the CUDA build
and visible devices in the same environment as `uv run train`. A `None` CUDA
build means that PyTorch was built without CUDA support.

For ARM CUDA-wheel selection, see the
[architecture notes](../AGENTS.md#sim2real-footguns-cost-real-debugging-weeks).
For timeouts while downloading large CUDA wheels, see the
[Quickstart download note](../README.md#quickstart). Use the repository's package
pins and source configuration when repairing an installation.

Without a local CUDA machine, a hosted GPU notebook can run the smoke test below;
select a GPU runtime and perform the environment check there. Colab offers a
free tier with [variable GPU availability and runtime limits](https://research.google.com/colaboratory/faq.html).
Install dependencies in that runtime and download results before releasing it.
[HF Jobs](../scripts/hf/README.md) is another remote route with its own account,
permissions and billing requirements.

## Check the command

Confirm that the task is registered and inspect its accepted options:

```bash
uv run list-envs
uv run train Mjlab-Velocity-Flat-MicroDuck --help
```

If `train` is missing or reports `can't open input file <task>`, check its location:

```bash
uv run python -c "import shutil; print(shutil.which('train'))"
```

It should resolve inside the project environment, not to an unrelated program
on the system path. Reinstall both packages that declare the training entry point:

```bash
uv sync --frozen --reinstall-package mjlab --reinstall-package mjlab-microduck
```

For a single GPU, the smoke recipe uses the default GPU selection. mjlab parses
collection arguments with Python syntax: an explicit device selection is
`--gpu-ids '[0]'`. An unbracketed `--gpu-ids 0` can produce a parser error.
Boolean options in these commands take explicit values such as `--video False`.

The smoke test uses TensorBoard, so it does not require a W&B login. For a run
that uses W&B or HF Jobs, follow the corresponding account setup in the
[HF Jobs guide](../scripts/hf/README.md#one-time-setup).

## Run a small CUDA smoke test

On a CUDA machine, use a fresh five-iteration run with 64 environments:

```bash
uv run train Mjlab-Velocity-Flat-MicroDuck \
    --env.scene.num-envs 64 \
    --agent.max-iterations 5 \
    --agent.seed 65 \
    --agent.logger tensorboard \
    --agent.experiment-name troubleshooting_smoke \
    --agent.run-name smoke \
    --agent.save-interval 1 \
    --agent.upload-model False \
    --enable-nan-guard True \
    --video False
```

The trainer prints its device and log directory. Check that the device is CUDA,
that iterations 0 through 4 complete, and that losses are finite. With 24 rollout
steps per environment, this is 7,680 environment steps. Inspect `nan_state`
termination values and any NaN-guard output as well as the process exit status.

The run directory under `logs/rsl_rl/troubleshooting_smoke/` contains
`params/env.yaml`, `params/agent.yaml`, TensorBoard events and checkpoints.
Check the saved environment count, task settings and agent options against the
command. With this fresh run and save interval, `model_4.pt` is the final
checkpoint: iteration numbers start at zero.

A fresh policy can fall during this run. Completion checks the training path;
five iterations do not establish a usable gait. Repeat the small run for the
task you intend to train before increasing the training budget.

### Export and check the result

Set `SMOKE_CHECKPOINT` to `model_4.pt` in the exact run directory printed above.
Replace `<run-directory>` in this example:

```bash
SMOKE_CHECKPOINT="logs/rsl_rl/troubleshooting_smoke/<run-directory>/model_4.pt"
uv run python scripts/export.py Mjlab-Velocity-Flat-MicroDuck \
    --checkpoint-file "$SMOKE_CHECKPOINT" \
    --num-envs 1 --device cuda:0 --video False --onnx-file smoke.onnx
```

The official exporter includes the observation normalizer in the ONNX graph.
Check the current policy family's input/output contract and a finite inference:

```bash
uv run python - <<'PY'
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession("smoke.onnx", providers=["CPUExecutionProvider"])
inputs = session.get_inputs()
assert len(inputs) == 1 and inputs[0].shape == [1, 61]
outputs = session.run(None, {inputs[0].name: np.zeros((1, 61), dtype=np.float32)})
assert len(outputs) == 1 and outputs[0].shape == (1, 14)
assert np.isfinite(outputs[0]).all()
print("ONNX shape and zero-input inference checks passed")
PY
```

This single input checks basic export compatibility. Evaluate the policy's
behaviour in simulation before using it for a longer evaluation or deployment.

## Load the matching checkpoint

A `.pt` training checkpoint and a deployable `.onnx` serve different purposes.
For missing files, unavailable W&B runs, Hub bundles, resume versus warm start,
and teacher checkpoints, follow the
[training checkpoint reference](training_checkpoints.md#use).

If loading reports tensor-size or state-dictionary mismatches, compare the task,
observation/action layout and source revision with the checkpoint's provenance.
Check out the bundle's recorded commit and apply its recorded diff when present.
Matching file extensions or actor input dimensions alone does not establish
checkpoint compatibility.

## Replay the exported policy

These CPU replay examples cover Linux and macOS. Copy `smoke.onnx` to the
machine running the viewer. Current 61-input policies use the unified command
layout; enable it in CPU replay:

```bash
uv run python scripts/infer_policy.py --walking smoke.onnx --new-cmd-obs
```

On macOS, launch the native viewer with MuJoCo's `mjpython`:

```bash
uv run mjpython scripts/infer_policy.py --walking smoke.onnx --new-cmd-obs
```

`--new-cmd-obs` selects a 13-value command block instead of the legacy 3-value
block. This explains a 51-versus-61 input-size error. Match the flag to the ONNX
input layout; legacy 51-input policies use the legacy layout.

Training without video and interactive viewing have different display needs.
On a headless GPU machine, run the smoke test with `--video False`; inspect or
replay the exported ONNX on a machine with a working display. For a viewer-only
exception, include the task, viewer backend and traceback in the report.

## Inspect training behaviour

Before changing rewards, inspect the saved configuration and a rollout from the
exact checkpoint. Compare task rewards, weighted penalty terms, episode lengths,
falls and resets. A reward increase alone does not show that the intended skill
has been learned, and an environment reset can look like recovery in a video.
The [training and reward-design notes](../AGENTS.md#reward-design--rules-that-were-each-learned-the-hard-way)
explain the project's reward signs and curriculum conventions.

For non-finite losses or `nan_state` terminations, keep the first failure's log,
NaN-guard output and resolved configuration. Use the small run to narrow down
the affected task or setting before continuing a long run.

## Check simulation performance

Initial Warp kernel compilation can dominate startup. Compare later iteration
timings, collection time and steps per second for the same task, environment
count and device. The 64-environment smoke test is a correctness check, not a
throughput benchmark.

Use `nvidia-smi` on the training host to check the selected GPU and memory use.
For GPU out-of-memory failures, reduce `--env.scene.num-envs` and repeat the small
run. Keep video disabled while diagnosing training throughput, then evaluate
recording separately. Changing terrain or collision models also changes the
simulation workload; compare equivalent configurations.

## Include enough information to reproduce a problem

Record the source revision (`git rev-parse HEAD`), working-tree changes
(`git status --short`), exact command, task, platform and the environment-check
output. Include the first traceback, relevant training metrics and
`params/env.yaml` / `params/agent.yaml`. For checkpoint or export failures,
include the checkpoint's task and provenance. These connect a reported symptom
to the configuration that actually ran.

# Reproduce independent mouth policy replay

Adding an independently actuated `mouth` joint to the stock scene used to make
`PolicyInference.get_observations()` fail: it subtracted the 14-joint default
pose from 15 joint positions. The fix keeps the walking policy's observation and
action layout at 61 and 14 values and controls the mouth separately.

The headless reproducer uses Pollen's released walking ONNX policy and the
checkout's stock scene, then adds a simple world-mounted hinge to represent the
extra actuator. It calls the existing observation, inference, and action methods
and takes ten MuJoCo steps using XML position control. This exercises actuator
routing; the added hinge is not a model of the stock beak and does not demonstrate
grasping.

## Run before and after

Run these commands from this PR's checkout root, with the repository environment
already installed. `--no-sync` uses that environment without changing its
dependencies. The reproducer needs Python 3.12, MuJoCo, NumPy, and ONNX Runtime;
it does not import Torch or mjlab and needs neither a display nor an NVIDIA GPU.
It makes no network requests itself. The recorded runs used MuJoCo 3.10.0 and
ONNX Runtime 1.24.4; exact versions and outcomes are in
[the before/after result record](mouth-replay-results.json).

Download the pinned official policy and extract the original inference script:

```sh
replay_dir=$(mktemp -d)
curl --disable --fail --location --output "$replay_dir/alpha_walking.onnx" \
  https://huggingface.co/pollen-robotics/microduck-policies/resolve/1b56c396825c052a4e26e95cf2b8d8298af9e9b4/alpha_walking.onnx
git show cb70b792312d559a4da09064d92009079671815f:scripts/infer_policy.py \
  > "$replay_dir/infer_policy_before.py"
```

The reproducer verifies policy SHA-256
`e36332d383997d51401897734cd3e79cf5038406feddb18b4d57ecfb141daa6c`
before loading it. The baseline script's SHA-256 is
`03bfd0642a880cd379e057536ee819bd1377d4af7b5206a57907d10afd1d4270`.

Run the baseline separately; **exit status 1 is the expected failure**, not a
successful check:

```sh
uv run --no-sync python scripts/reproduce_mouth_replay.py \
  --policy "$replay_dir/alpha_walking.onnx" \
  --inference-script "$replay_dir/infer_policy_before.py" \
  > "$replay_dir/before.json"
```

Then run the patched script. This command should exit **0**:

```sh
uv run --no-sync python scripts/reproduce_mouth_replay.py \
  --policy "$replay_dir/alpha_walking.onnx" \
  > "$replay_dir/after.json"
```

The JSON reports the source and policy hashes, dependency versions, and each
case's outcome. Compare the stock scene with the extra-actuator scene: the stock
case should work in both versions; the extra actuator should reproduce the
observation-shape failure before the fix and run after it. A zero exit status
does not establish walking stability, physical beak accuracy, or sim-to-real
transfer.

## Regression tests

With pytest installed in the selected environment (the review used
`pytest==9.0.3`), run:

```sh
uv run --no-sync python -m pytest -q \
  tests/test_infer_policy_mouth.py tests/test_infer_policy_bam.py
```

The mouth tests check policy/mouth target isolation, actuator reordering,
interleaved passive joints, and the native MuJoCo position actuator's `timeconst`
option. The latter protects valid filtered position actuators from rejection by
the new validator. The existing BAM tests compare CPU actuator setup with the
training configuration; unlike the standalone reproducer, those tests import
the training dependencies.

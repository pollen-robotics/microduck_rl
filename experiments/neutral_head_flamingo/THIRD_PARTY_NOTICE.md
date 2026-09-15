# Third-party notice

The committed ONNX model is derived from the Pollen Robotics / Remi Fabre
Microduck Flamingo cycle policy:

- Source repository: https://github.com/pollen-robotics/microduck_rl
- Published policy: https://huggingface.co/RemiFabre/microduck-flamingo-cycle
- Base revision used by this experiment:
  `4757d6aed51ff059232a66cdb8a4ee3a77fa2ce8`
- License: Apache License 2.0 (see the repository root `LICENSE`)

The original actor's observation normalizer and hidden layers are retained.
Its final layer was distilled against fixed-head teacher rollouts, and all four
head action rows were constrained to the neutral `HOME` output.


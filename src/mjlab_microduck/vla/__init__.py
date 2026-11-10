"""Language-conditioned (free-sentence) policy support.

This package is the *text side* of the VLA work:

- :mod:`mjlab_microduck.vla.text_encoder` — frozen text encoders.
- :mod:`mjlab_microduck.vla.instructions` — intent table + balanced sampler.

Design contract (see ``docs/vla-plan.md``):

- The actor sees the text embedding, never the numeric command vector.
- The numeric command continues to drive rewards and is available to the
  privileged critic.
- ``encoder.onnx`` and ``policy.onnx`` are separate artifacts; the policy
  ONNX takes ``obs`` (61D) plus ``text_emb`` (D) as two inputs.
- The text embedding is NOT covered by the observation normalizer.
"""

from .instructions import (
    COMMAND_LAYOUT,
    DEFAULT_INSTRUCTIONS_PATH,
    NUM_COMMANDS,
    InstructionIntent,
    InstructionPool,
    PoolSample,
)
from .model import DEFAULT_TEXT_GROUP, MicroduckVLAModel
from .text_encoder import (
    DEFAULT_DIM,
    DEFAULT_MODEL,
    HFTextEncoder,
    HashTextEncoder,
    TextEncoder,
)

__all__ = [
    "COMMAND_LAYOUT",
    "DEFAULT_DIM",
    "DEFAULT_INSTRUCTIONS_PATH",
    "DEFAULT_MODEL",
    "DEFAULT_TEXT_GROUP",
    "HFTextEncoder",
    "HashTextEncoder",
    "InstructionIntent",
    "InstructionPool",
    "MicroduckVLAModel",
    "NUM_COMMANDS",
    "PoolSample",
    "TextEncoder",
]

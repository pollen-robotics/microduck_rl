"""Instruction table + sampling pool for the free-sentence policy.

Why this exists
---------------
The actor only sees the **text embedding**. The numeric 13D command vector
(``[twist(3), head_pose(4), body_pose(6)]``) is kept for the reward stack and
for the privileged critic, but it is never concatenated into the actor
observation. Every paraphrase in the table therefore has to be unambiguous
enough that the policy can recover its intent from the sentence alone.

Each intent carries **several paraphrases** (Chinese + English here). Training
on paraphrases is what makes a free-sentence encoder worth its weight: the
policy learns the mapping *embedding -> behavior*, so it generalizes to unseen
phrasings. One sentence per intent would just memorize one embedding.

Sampling is intentionally **balanced over intents** (intent uniformly, then a
paraphrase uniformly inside it). Uniform sampling over all sentences would
over-weight intents that happen to have more paraphrases, reproducing the
"rare command regions never train" failure the repo already learned the hard
way.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .text_encoder import TextEncoder

# Canonical command block layout, shared with the whole policy family:
# [twist(3), head_pose(4), body_pose(6)] — see AGENTS.md.
COMMAND_LAYOUT: tuple[str, ...] = (
    "twist_x",
    "twist_y",
    "twist_z",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "body_x",
    "body_y",
    "body_z",
    "body_roll",
    "body_pitch",
    "body_yaw",
)
NUM_COMMANDS = len(COMMAND_LAYOUT)  # 13

DEFAULT_INSTRUCTIONS_PATH = Path(__file__).with_name("instructions.json")


@dataclass(frozen=True)
class InstructionIntent:
    """One intent: a numeric command plus the paraphrases that express it."""

    name: str
    command: np.ndarray  # (13,) float32
    paraphrases: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.command.shape != (NUM_COMMANDS,):
            raise ValueError(
                f"intent {self.name!r}: command must have shape "
                f"({NUM_COMMANDS},), got {self.command.shape}"
            )
        if not np.isfinite(self.command).all():
            raise ValueError(f"intent {self.name!r}: command contains non-finite values")
        if not self.paraphrases:
            raise ValueError(f"intent {self.name!r}: needs at least one paraphrase")


@dataclass(frozen=True)
class PoolSample:
    """A batch sampled from the pool.

    ``intent_index`` and ``sentence_index`` are both length-``n``; the latter
    indexes the flattened sentence/embedding tables.
    """

    intent_index: np.ndarray  # (n,) int64
    sentence_index: np.ndarray  # (n,) int64
    sentences: tuple[str, ...]
    commands: np.ndarray  # (n, 13) float32
    embeddings: np.ndarray  # (n, dim) float32


class InstructionPool:
    """Loads the intent table, precomputes embeddings, samples instructions."""

    def __init__(
        self,
        intents: Sequence[InstructionIntent],
        *,
        encoder_model: str | None = None,
        encoder_dim: int | None = None,
        version: int = 1,
    ) -> None:
        if not intents:
            raise ValueError("InstructionPool needs at least one intent")

        self.version = int(version)
        self.encoder_model = encoder_model
        self.encoder_dim = encoder_dim
        self.intents: tuple[InstructionIntent, ...] = tuple(intents)

        names = [intent.name for intent in self.intents]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate intent names: {names}")

        # Flatten sentences and remember which intent each one belongs to.
        sentences: list[str] = []
        sentence_intent: list[int] = []
        per_intent_sentence_ids: list[np.ndarray] = []
        for intent_id, intent in enumerate(self.intents):
            ids = []
            for sentence in intent.paraphrases:
                ids.append(len(sentences))
                sentences.append(sentence)
                sentence_intent.append(intent_id)
            per_intent_sentence_ids.append(np.asarray(ids, dtype=np.int64))

        # The whole point is unambiguous grounding: the same sentence must not
        # appear under two intents, and paraphrases within an intent are
        # redundant (they would silently skew that intent's paraphrase mix).
        if len(set(sentences)) != len(sentences):
            seen: dict[str, int] = {}
            duplicates: list[str] = []
            for sentence in sentences:
                if sentence in seen:
                    duplicates.append(sentence)
                seen[sentence] = 1
            raise ValueError(f"duplicate paraphrase(s) across the table: {duplicates}")

        self.sentences: tuple[str, ...] = tuple(sentences)
        self.sentence_intent = np.asarray(sentence_intent, dtype=np.int64)
        self._per_intent_sentence_ids = tuple(per_intent_sentence_ids)
        self._commands = np.stack([i.command for i in self.intents]).astype(np.float32)
        self._embeddings: np.ndarray | None = None

        # A balanced sampler only needs the intent -> sentence-id ranges.
        self._intent_ids = np.arange(len(self.intents), dtype=np.int64)

    # -- construction -----------------------------------------------------

    @classmethod
    def from_json(
        cls,
        path: str | Path = DEFAULT_INSTRUCTIONS_PATH,
        *,
        encoder_dim: int | None = None,
    ) -> InstructionPool:
        """Load the intent table.

        ``encoder_dim`` overrides the dimension declared in the JSON. The
        training/deployment encoder must match it; tests use it to run a
        small dependency-free encoder against the same table.
        """
        path = Path(path)
        with path.open(encoding="utf-8") as handle:
            raw = json.load(handle)

        layout = tuple(raw.get("command_layout", COMMAND_LAYOUT))
        if layout != COMMAND_LAYOUT:
            raise ValueError(
                f"{path}: command_layout does not match the 13D policy contract.\n"
                f"  expected: {COMMAND_LAYOUT}\n"
                f"  got:      {layout}"
            )

        intents: list[InstructionIntent] = []
        for entry in raw.get("intents", []):
            cmd = entry["command"]
            twist = np.asarray(cmd.get("twist", [0.0, 0.0, 0.0]), dtype=np.float32)
            head = np.asarray(
                cmd.get("head_pose", [0.0, 0.0, 0.0, 0.0]), dtype=np.float32
            )
            body = np.asarray(
                cmd.get("body_pose", [0.0] * 6), dtype=np.float32
            )
            if twist.shape != (3,) or head.shape != (4,) or body.shape != (6,):
                raise ValueError(
                    f"{path}: intent {entry.get('name')!r} has bad command "
                    f"shapes: twist={twist.shape}, head_pose={head.shape}, "
                    f"body_pose={body.shape}"
                )
            intents.append(
                InstructionIntent(
                    name=str(entry["name"]),
                    command=np.concatenate([twist, head, body]).astype(np.float32),
                    paraphrases=tuple(str(p) for p in entry.get("paraphrases", ())),
                )
            )

        encoder = raw.get("encoder", {})
        declared_dim = int(encoder["dim"]) if "dim" in encoder else None
        return cls(
            intents,
            encoder_model=encoder.get("model"),
            encoder_dim=declared_dim if encoder_dim is None else int(encoder_dim),
            version=int(raw.get("version", 1)),
        )

    # -- properties -------------------------------------------------------

    @property
    def num_intents(self) -> int:
        return len(self.intents)

    @property
    def num_sentences(self) -> int:
        return len(self.sentences)

    @property
    def intent_names(self) -> tuple[str, ...]:
        return tuple(intent.name for intent in self.intents)

    @property
    def command_vectors(self) -> np.ndarray:
        """``(num_intents, 13)`` numeric command per intent."""
        return self._commands

    @property
    def embeddings(self) -> np.ndarray:
        """``(num_sentences, dim)`` precomputed embeddings.

        Raises until :meth:`embed` has been called.
        """
        if self._embeddings is None:
            raise RuntimeError(
                "InstructionPool has no embeddings yet — call pool.embed(encoder) "
                "once at environment/runner setup."
            )
        return self._embeddings

    @property
    def is_embedded(self) -> bool:
        return self._embeddings is not None

    # -- encoding ---------------------------------------------------------

    def embed(self, encoder: TextEncoder) -> np.ndarray:
        """Precompute one embedding per sentence. Call exactly once.

        This is the only expensive step; the training/inference loops then do
        a cheap table lookup. Returns the ``(num_sentences, dim)`` table.
        """
        embeddings = np.asarray(encoder.encode(self.sentences), dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] != self.num_sentences:
            raise ValueError(
                f"encoder returned shape {embeddings.shape}; expected "
                f"({self.num_sentences}, dim)"
            )
        if not np.isfinite(embeddings).all():
            raise ValueError("encoder returned non-finite embeddings")
        if self.encoder_dim is not None and embeddings.shape[1] != self.encoder_dim:
            raise ValueError(
                f"encoder output dim {embeddings.shape[1]} does not match the "
                f"table's encoder.dim {self.encoder_dim}"
            )
        self._embeddings = embeddings
        return embeddings

    # -- sampling ---------------------------------------------------------

    def sample(
        self, n: int, rng: np.random.Generator | None = None
    ) -> PoolSample:
        """Draw ``n`` instructions, balanced over intents.

        Intent is sampled uniformly, then a paraphrase uniformly within it.
        Requires :meth:`embed` first (the caller needs the embedding anyway).
        """
        if n < 0:
            raise ValueError(f"n must be >= 0, got {n}")
        if self._embeddings is None:
            raise RuntimeError(
                "InstructionPool.sample() needs embeddings — call "
                "pool.embed(encoder) first."
            )
        rng = np.random.default_rng() if rng is None else rng

        intent_index = rng.integers(0, self.num_intents, size=n, dtype=np.int64)
        sentence_index = np.empty(n, dtype=np.int64)
        for row, intent_id in enumerate(intent_index):
            choices = self._per_intent_sentence_ids[int(intent_id)]
            sentence_index[row] = choices[rng.integers(0, len(choices))]

        return PoolSample(
            intent_index=intent_index,
            sentence_index=sentence_index,
            sentences=tuple(self.sentences[i] for i in sentence_index),
            commands=self._commands[intent_index],
            embeddings=self._embeddings[sentence_index],
        )

    def __len__(self) -> int:
        return self.num_intents

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"InstructionPool(intents={self.num_intents}, "
            f"sentences={self.num_sentences}, "
            f"embedded={self.is_embedded})"
        )

"""Frozen text encoders for the language-conditioned (VLA) policy.

The text encoder is a *fixed* module: PPO never updates it. It runs only when
the instruction changes (not at 50 Hz), and its output is cached as a
``(D,)`` float32 vector. That vector is fed to ``policy.onnx`` as a separate
input — it is deliberately NOT part of the 61D proprio observation and NOT
covered by the observation normalizer.

Two backends are provided:

``HFTextEncoder``
    The real deployment encoder (a frozen ``SentenceTransformer``). Requires
    the optional ``vla`` extra::

        uv sync --extra vla        # or: uv add sentence-transformers

``HashTextEncoder``
    Dependency-free deterministic character-n-gram hashing. It carries no
    semantics, but it is stable across processes and machines, so it lets the
    pool/model/export plumbing be tested on CPU/CI without downloading a
    transformer. Never ship it to the robot.

Both backends return ``float32`` arrays of shape ``(N, dim)`` and are pure
functions of the input text (same input -> same output).
"""

from __future__ import annotations

import hashlib
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_DIM = 384


@runtime_checkable
class TextEncoder(Protocol):
    """Minimal interface consumed by :class:`InstructionPool`."""

    dim: int

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Encode texts into an ``(N, dim)`` float32 array."""
        ...


class HashTextEncoder:
    """Deterministic character-n-gram hashing encoder (tests/CI only).

    Each text is lowercased, whitespace-collapsed, split into character
    n-grams, and each n-gram is hashed into one of ``dim`` buckets with
    ``blake2b`` (not Python's salted ``hash()``, so the result is identical
    across processes). The count vector is L2-normalized. Texts that share
    n-grams therefore get similar vectors, which is enough to exercise the
    lookup/sampling/export paths without a transformer.

    It is NOT semantically meaningful: do not use it for a trained policy.
    """

    def __init__(self, dim: int = 64, ngram_sizes: Sequence[int] = (2, 3, 4, 5)) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if not ngram_sizes:
            raise ValueError("ngram_sizes must not be empty")
        self.dim = int(dim)
        self.ngram_sizes = tuple(int(n) for n in ngram_sizes)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            normalized = " ".join(str(text).lower().split())
            for n in self.ngram_sizes:
                # max(1, ...) so a string shorter than n still contributes one gram.
                for start in range(max(1, len(normalized) - n + 1)):
                    gram = normalized[start : start + n]
                    digest = hashlib.blake2b(
                        gram.encode("utf-8"), digest_size=8
                    ).digest()
                    out[row, int.from_bytes(digest, "little") % self.dim] += 1.0
            norm = float(np.linalg.norm(out[row]))
            if norm > 0.0:
                out[row] /= norm
        return out

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"HashTextEncoder(dim={self.dim}, ngram_sizes={self.ngram_sizes})"


class HFTextEncoder:
    """Frozen ``SentenceTransformer`` encoder (the real deployment backend).

    The model is loaded in eval mode and is never trained. ``encode`` runs
    under ``torch.inference_mode()``.

    Args:
        model_name: Hugging Face sentence-transformers id.
        dim: Expected output dimension. Checked on every encode; must match
            the ``encoder.dim`` recorded in ``instructions.json`` and the
            ``text_emb`` input width of ``policy.onnx``.
        device: ``"cpu"``, ``"cuda"``, ...
        normalize_embeddings: L2-normalize outputs (MiniLM models are trained
            with normalized embeddings; keep this on).
        max_length: Optional truncation length for the tokenizer.
        batch_size: Encode batch size.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        dim: int = DEFAULT_DIM,
        device: str = "cpu",
        normalize_embeddings: bool = True,
        max_length: int | None = 64,
        batch_size: int = 64,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "HFTextEncoder requires the optional VLA dependencies. "
                "Install them with `uv sync --extra vla` (or "
                "`uv add sentence-transformers`). "
                "The dependency-free HashTextEncoder works without them."
            ) from exc

        self.model_name = model_name
        self.dim = int(dim)
        self._batch_size = int(batch_size)
        self._normalize = bool(normalize_embeddings)
        self._model = SentenceTransformer(model_name, device=device)
        self._model.eval()
        if max_length is not None:
            self._model.max_seq_length = int(max_length)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        import torch

        with torch.inference_mode():
            embeddings = self._model.encode(
                list(texts),
                batch_size=self._batch_size,
                convert_to_numpy=True,
                normalize_embeddings=self._normalize,
                show_progress_bar=False,
            )
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dim:
            raise ValueError(
                f"{self.model_name} returned shape {embeddings.shape}; "
                f"expected (N, {self.dim}). Update encoder.dim in "
                f"instructions.json or the model name."
            )
        return embeddings

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"HFTextEncoder(model={self.model_name!r}, dim={self.dim})"

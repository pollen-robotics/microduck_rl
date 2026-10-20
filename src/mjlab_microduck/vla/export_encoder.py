"""Export the frozen sentence encoder to ``encoder.onnx``.

Contract (see ``docs/vla-plan.md``)::

    input_ids      [1, L] int64
    attention_mask [1, L] int64
        -> text_emb [1, D] float32

The graph is mean-pooled and L2-normalized, matching
``SentenceTransformer.encode(normalize_embeddings=True)`` for the
mean-pooling MiniLM family used by the training table. When
``sentence-transformers`` is installed, the export validates itself against
that reference before returning.

Requires the optional VLA extra::

    uv sync --extra vla
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

DEFAULT_MAX_LENGTH = 64
DEFAULT_OPSET = 18


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_encoder_onnx(
    model_name: str,
    output_path: str | Path,
    *,
    expected_dim: int | None = None,
    max_length: int = DEFAULT_MAX_LENGTH,
    opset: int = DEFAULT_OPSET,
    device: str = "cpu",
) -> dict[str, Any]:
    """Export a mean-pooled, L2-normalized sentence encoder.

    Args:
        model_name: Hugging Face id, e.g.
            ``sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2``.
        output_path: Destination ``.onnx``.
        expected_dim: Reject a model whose embedding width differs (must match
            the policy's ``text_emb`` input).
        max_length: Tokenizer truncation length recorded in the metadata.
        opset: ONNX opset (the policy export uses 18).
        device: Export device.

    Returns:
        Metadata for the manifest / ONNX metadata (model, dim, sha256,
        inputs, pooling, normalize).
    """
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "export_encoder_onnx requires the optional VLA dependencies: "
            "`uv sync --extra vla` (or `uv add sentence-transformers`)."
        ) from exc

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name).to(device).eval()

    class MeanPoolEncoder(nn.Module):
        """Mean-pool ``last_hidden_state`` over the mask, then L2-normalize."""

        def __init__(self, backbone: nn.Module) -> None:
            super().__init__()
            self.backbone = backbone

        def forward(
            self, input_ids: torch.Tensor, attention_mask: torch.Tensor
        ) -> torch.Tensor:
            hidden = self.backbone(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
            return F.normalize(pooled, p=2, dim=1)

    wrapper = MeanPoolEncoder(encoder).to(device).eval()

    sample = tokenizer(
        ["往前走"],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    input_ids = sample["input_ids"].to(device)
    attention_mask = sample["attention_mask"].to(device)

    with torch.inference_mode():
        embedding = wrapper(input_ids, attention_mask)
    dim = int(embedding.shape[-1])
    if expected_dim is not None and dim != expected_dim:
        raise ValueError(
            f"{model_name} produces {dim}-D embeddings, the policy expects "
            f"{expected_dim}. Pick a matching encoder."
        )

    # Self-check against the training-time encoder when it is available.
    try:
        from sentence_transformers import SentenceTransformer

        reference_model = SentenceTransformer(model_name, device=device)
        with torch.inference_mode():
            reference = reference_model.encode(
                ["往前走"],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        cosine = (reference * embedding.cpu().numpy()).sum(axis=-1)
        if float(cosine.min()) < 0.999:
            raise ValueError(
                f"encoder export disagrees with SentenceTransformer.encode "
                f"(min cosine {cosine.min():.4f}); the model may not use mean "
                "pooling. Add its pooling mode to export_encoder_onnx()."
            )
    except ImportError:  # pragma: no cover - without the ST backend
        pass

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (input_ids, attention_mask),
        str(output_path),
        export_params=True,
        opset_version=opset,
        input_names=["input_ids", "attention_mask"],
        output_names=["text_emb"],
        dynamic_axes={
            "input_ids": {1: "seq"},
            "attention_mask": {1: "seq"},
        },
        dynamo=False,
    )

    # The traced graph tends to leave the output shape symbolic even though the
    # embedding width is fixed. Pin it to [1, D]: the runtime encodes one
    # instruction at a time, and the manifest gate requires a concrete width.
    import onnx

    onnx_model = onnx.load(str(output_path))
    output = onnx_model.graph.output[0]
    dims = output.type.tensor_type.shape.dim
    del dims[:]
    dims.add().dim_value = 1
    dims.add().dim_value = dim
    onnx.save(onnx_model, str(output_path))

    return {
        "model": model_name,
        "file": output_path.name,
        "embedding_dim": dim,
        "max_length": int(max_length),
        "pooling": "mean",
        "normalize": True,
        "inputs": ["input_ids", "attention_mask"],
        "sha256": _sha256(output_path),
    }


def main() -> int:
    import sys

    import tyro

    def cli(
        model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        out: str = "encoder.onnx",
        expected_dim: int | None = None,
        max_length: int = DEFAULT_MAX_LENGTH,
        device: str = "cpu",
    ) -> int:
        """Export a frozen sentence encoder to ONNX.

        Args:
            model: Hugging Face sentence-transformers id.
            out: Destination .onnx path.
            expected_dim: Require this embedding width (match the policy).
            max_length: Tokenizer truncation length.
            device: cpu or cuda.
        """
        info = export_encoder_onnx(
            model,
            out,
            expected_dim=expected_dim,
            max_length=max_length,
            device=device,
        )
        print(f"[encoder] wrote {out}: {info['embedding_dim']}D, sha256={info['sha256']}")
        return 0

    return tyro.cli(cli, args=sys.argv[1:], prog="export_encoder")


if __name__ == "__main__":
    raise SystemExit(main())

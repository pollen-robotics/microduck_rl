"""Small real artifact fixtures shared by the Next RL contract tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def write_tiny_policy(path: Path, *, weight: float = 0.1) -> Path:
    """Write a nonconstant 61-to-14 ONNX policy without importing torch."""
    weights = numpy_helper.from_array(
        np.full((61, 14), weight, dtype=np.float32),
        "weights",
    )
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["obs", "weights"], ["actions"])],
        "tiny-policy",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, 61])],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, 14])],
        initializer=[weights],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, path)
    return path


def write_test_video(path: Path) -> Path:
    """Write a tiny two-frame MP4 that exercises the real ffmpeg decoder."""
    frames = np.zeros((2, 16, 16, 3), dtype=np.uint8)
    frames[1, :, :, 1] = 255
    iio.imwrite(path, frames, fps=10)
    return path


def write_renderer_sidecar(
    path: Path,
    *,
    role: str,
    scenario_id: str,
    seed: int,
    policy_sha256: str,
    evaluation_digest: str,
    video_path: Path,
    renderer_revision: str = "test-renderer-v1",
) -> Path:
    """Write independently-derived canonical renderer evidence for boundary tests."""
    evidence = {
        "evaluation_digest": evaluation_digest,
        "policy_sha256": policy_sha256,
        "renderer_revision": renderer_revision,
        "role": role,
        "scenario_id": scenario_id,
        "seed": seed,
        "video_path": str(video_path),
        "video_sha256": hashlib.sha256(video_path.read_bytes()).hexdigest(),
    }
    path.write_text(
        json.dumps(evidence, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    return path

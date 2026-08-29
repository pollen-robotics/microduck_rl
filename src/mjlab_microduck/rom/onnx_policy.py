"""Independent structural proof that an ONNX actor consumes one baked normalizer."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import onnx
from onnx import numpy_helper


@dataclass(frozen=True)
class NormalizedActorGraph:
    fingerprint: str
    graph_sha256: str


@dataclass(frozen=True)
class _Trace:
    stage: int
    mean_name: str | None = None
    std_name: str | None = None
    extra_stats_transform: bool = False


def _is_empirical_stats_initializer(
    initializers: dict[str, onnx.TensorProto], name: str
) -> bool:
    tensor = initializers.get(name)
    return bool(
        tensor is not None
        and tensor.data_type == onnx.TensorProto.FLOAT
        and tuple(tensor.dims) == (61,)
    )


def _float32_vector(
    initializers: dict[str, onnx.TensorProto], name: str, *, positive: bool
) -> np.ndarray:
    tensor = initializers.get(name)
    if tensor is None or tensor.data_type != onnx.TensorProto.FLOAT:
        raise ValueError("ONNX normalization statistics must be tensor(float)")
    values = numpy_helper.to_array(tensor)
    if values.shape != (61,) or values.dtype != np.float32:
        raise ValueError("ONNX normalization statistics must have float32 shape [61]")
    if not np.isfinite(values).all() or (positive and not np.all(values > 0.0)):
        raise ValueError("ONNX normalization statistics are not finite and safe")
    return values


def inspect_normalized_actor(model: onnx.ModelProto) -> NormalizedActorGraph:
    """Trace the sole actor output and prove all input dependence passed Sub/Div."""
    graph = model.graph
    if len(graph.input) != 1 or len(graph.output) != 1:
        raise ValueError("ONNX normalization requires one actor input and output")
    initializers = {item.name: item for item in graph.initializer}
    traces: dict[str, _Trace] = {graph.input[0].name: _Trace(0)}
    statistics: dict[str, np.ndarray] = {}
    for node in graph.node:
        dynamic = [traces[name] for name in node.input if name in traces]
        trace: _Trace | None = None
        if (
            node.op_type == "Sub"
            and len(node.input) == 2
            and node.input[0] in traces
            and traces[node.input[0]].stage == 0
            and node.input[1] in initializers
        ):
            statistics[node.input[1]] = _float32_vector(
                initializers, node.input[1], positive=False
            )
            source = traces[node.input[0]]
            trace = _Trace(
                1,
                mean_name=node.input[1],
                extra_stats_transform=source.extra_stats_transform,
            )
        elif (
            node.op_type == "Div"
            and len(node.input) == 2
            and node.input[0] in traces
            and traces[node.input[0]].stage == 1
            and node.input[1] in initializers
        ):
            source = traces[node.input[0]]
            statistics[node.input[1]] = _float32_vector(
                initializers, node.input[1], positive=True
            )
            trace = _Trace(
                2,
                mean_name=source.mean_name,
                std_name=node.input[1],
                extra_stats_transform=source.extra_stats_transform,
            )
        elif dynamic:
            first = dynamic[0]
            same_normalizer = all(
                (item.stage, item.mean_name, item.std_name)
                == (first.stage, first.mean_name, first.std_name)
                for item in dynamic
            )
            has_stats_initializer = any(
                _is_empirical_stats_initializer(initializers, name)
                for name in node.input
                if name not in traces
            )
            trace = _Trace(
                min(item.stage for item in dynamic),
                mean_name=first.mean_name if same_normalizer else None,
                std_name=first.std_name if same_normalizer else None,
                extra_stats_transform=(
                    any(item.extra_stats_transform for item in dynamic)
                    or (
                        node.op_type in {"Add", "Sub", "Mul", "Div"}
                        and has_stats_initializer
                    )
                ),
            )
        if trace is not None:
            for output in node.output:
                traces[output] = trace
    output_trace = traces.get(graph.output[0].name)
    if (
        output_trace is None
        or output_trace.stage != 2
        or output_trace.mean_name is None
        or output_trace.std_name is None
        or output_trace.extra_stats_transform
    ):
        raise ValueError(
            "ONNX actor must apply exactly one empirical normalization stage "
            "on every input-output path"
        )
    mean = statistics[output_trace.mean_name]
    std = statistics[output_trace.std_name]
    graph_sha256 = hashlib.sha256(graph.SerializeToString()).hexdigest()
    fingerprint_payload = json.dumps(
        {
            "graphSha256": graph_sha256,
            "meanSha256": hashlib.sha256(mean.tobytes()).hexdigest(),
            "stdSha256": hashlib.sha256(std.tobytes()).hexdigest(),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return NormalizedActorGraph(
        fingerprint=f"sha256:{hashlib.sha256(fingerprint_payload).hexdigest()}",
        graph_sha256=graph_sha256,
    )

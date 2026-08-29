"""Independent structural proof that an ONNX actor consumes one baked normalizer."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import onnx
from onnx import helper, numpy_helper


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


@dataclass(frozen=True)
class _StaticTensor:
    value: np.ndarray | None
    ambiguous: bool = False


def _attribute(node: onnx.NodeProto, name: str, default: object = None) -> object:
    item = next((item for item in node.attribute if item.name == name), None)
    return default if item is None else helper.get_attribute_value(item)


def _constant_node_value(node: onnx.NodeProto) -> np.ndarray | None:
    value = _attribute(node, "value")
    if isinstance(value, onnx.TensorProto):
        return np.asarray(numpy_helper.to_array(value))
    for name in ("value_float", "value_int", "value_floats", "value_ints"):
        value = _attribute(node, name)
        if value is not None:
            return np.asarray(value)
    return None


def _reshape(value: np.ndarray, shape: np.ndarray, *, allowzero: bool) -> np.ndarray:
    dimensions = [int(item) for item in np.asarray(shape).reshape(-1)]
    if not allowzero:
        dimensions = [
            value.shape[index] if dimension == 0 else dimension
            for index, dimension in enumerate(dimensions)
        ]
    if any(item < -1 for item in dimensions) or dimensions.count(-1) > 1:
        raise ValueError("invalid static ONNX Reshape dimensions")
    return np.reshape(value, dimensions)


def _static_node_outputs(
    node: onnx.NodeProto, constants: dict[str, _StaticTensor]
) -> tuple[_StaticTensor, ...] | None:
    """Evaluate a bounded exporter-style constant node or retain ambiguity."""
    if node.op_type == "Constant":
        value = _constant_node_value(node)
        result = _StaticTensor(value, ambiguous=value is None)
        return tuple(result for _ in node.output)
    if any(name and name not in constants for name in node.input):
        return None
    inputs = [constants[name] for name in node.input if name]
    if not inputs:
        return None
    if any(item.value is None or item.ambiguous for item in inputs):
        return tuple(_StaticTensor(None, ambiguous=True) for _ in node.output)
    values = [item.value for item in inputs]
    assert all(value is not None for value in values)
    try:
        if node.op_type == "Identity":
            output = values[0]
        elif node.op_type == "Cast":
            target = int(_attribute(node, "to"))
            output = values[0].astype(helper.tensor_dtype_to_np_dtype(target))
        elif node.op_type == "Reshape" and len(values) == 2:
            output = _reshape(
                values[0],
                values[1],
                allowzero=bool(int(_attribute(node, "allowzero", 0))),
            )
        else:
            return tuple(_StaticTensor(None, ambiguous=True) for _ in node.output)
    except (TypeError, ValueError, IndexError, OverflowError) as exc:
        raise ValueError("invalid static ONNX constant transform") from exc
    return (_StaticTensor(np.asarray(output)),)


def _is_empirical_stats_constant(
    constants: dict[str, _StaticTensor], name: str
) -> bool:
    tensor = constants.get(name)
    if tensor is None:
        return False
    if tensor.value is None:
        return tensor.ambiguous
    return bool(tensor.value.size == 61)


def _float32_vector(
    constants: dict[str, _StaticTensor], name: str, *, positive: bool
) -> np.ndarray:
    tensor = constants.get(name)
    if tensor is None or tensor.value is None or tensor.ambiguous:
        raise ValueError("ONNX normalization statistics have ambiguous static lineage")
    values = tensor.value
    if values.dtype != np.float32:
        raise ValueError("ONNX normalization statistics must be tensor(float)")
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
    constants: dict[str, _StaticTensor] = {}
    for item in graph.initializer:
        try:
            constants[item.name] = _StaticTensor(
                np.asarray(numpy_helper.to_array(item))
            )
        except (TypeError, ValueError):
            constants[item.name] = _StaticTensor(None, ambiguous=True)
    traces: dict[str, _Trace] = {graph.input[0].name: _Trace(0)}
    statistics: dict[str, np.ndarray] = {}
    for node in graph.node:
        dynamic = [traces[name] for name in node.input if name in traces]
        if not dynamic:
            static_outputs = _static_node_outputs(node, constants)
            if static_outputs is not None:
                if len(static_outputs) != len(node.output):
                    static_outputs = tuple(
                        _StaticTensor(None, ambiguous=True) for _ in node.output
                    )
                constants.update(zip(node.output, static_outputs, strict=True))
            continue
        trace: _Trace | None = None
        if (
            node.op_type == "Sub"
            and len(node.input) == 2
            and node.input[0] in traces
            and traces[node.input[0]].stage == 0
            and node.input[1] in constants
        ):
            statistics[node.input[1]] = _float32_vector(
                constants, node.input[1], positive=False
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
            and node.input[1] in constants
        ):
            source = traces[node.input[0]]
            statistics[node.input[1]] = _float32_vector(
                constants, node.input[1], positive=True
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
            has_stats_constant = any(
                _is_empirical_stats_constant(constants, name)
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
                        and has_stats_constant
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

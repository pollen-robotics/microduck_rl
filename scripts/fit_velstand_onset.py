"""Fit a simulation candidate from the exported v10 and v8 VelStand actors.

    uv run scripts/fit_velstand_onset.py v10/velstand.onnx v8/velstand.onnx candidate.onnx

The student keeps its observation normalizer and hidden layers. The fit uses
v8's upright locomotion examples and v10's examples of the other commands and
recovery poses. See docs/velstand_policy.md for the rehearsal and its limits.
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from velstand_onset_rollouts import collect_examples


def fit_head(model, observations, actions, ridge=0.001):
    """Fit the change from the original output weights with ridge regression."""
    graph = model.graph
    head = graph.node[-1]
    attrs = {a.name: helper.get_attribute_value(a) for a in head.attribute}
    if (head.op_type != "Gemm" or list(head.output) != [graph.output[0].name]
            or attrs.get("transB", 0) != 1 or attrs.get("transA", 0) != 0
            or attrs.get("alpha", 1) != 1 or attrs.get("beta", 1) != 1):
        raise ValueError("Expected an exported VelStand actor with a final linear action layer")
    inputs = {v.name: v for v in graph.initializer}
    weight, bias = (inputs[name] for name in head.input[1:])
    old = np.vstack((numpy_helper.to_array(weight).T, numpy_helper.to_array(bias)))

    # Read the hidden features from the original graph, including its baked-in
    # normalizer. Release exports take one observation per inference.
    features = helper.make_tensor_value_info(head.input[0], TensorProto.FLOAT, [1, old.shape[0] - 1])
    graph.output.append(features)
    try:
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        encoder = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
    finally:
        del graph.output[-1]
    x = np.ones((len(observations), old.shape[0]), dtype=np.float64)
    for i, obs in enumerate(observations):
        x[i, :-1] = encoder.run([features.name], {graph.input[0].name: obs[None]})[0][0]

    gram = x.T @ x + ridge * len(x) * np.eye(x.shape[1])
    delta = np.linalg.solve(gram, x.T @ (actions - x @ old))
    fitted = (old + delta).astype(np.float32)
    if not np.isfinite(fitted).all():
        raise ValueError("Non-finite output weights in the fitted actor")
    weight.CopyFrom(numpy_helper.from_array(fitted[:-1].T.copy(), weight.name))
    bias.CopyFrom(numpy_helper.from_array(fitted[-1], bias.name))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("base", type=Path, help="v10 velstand.onnx")
    parser.add_argument("teacher", type=Path, help="v8 velstand.onnx")
    parser.add_argument("output", type=Path, help="new ONNX file for rehearsal in duck-sim")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Choose a new output path. {args.output} already exists.")

    model = onnx.load(args.base)
    observations, actions = collect_examples(args.base, args.teacher)
    fit_head(model, observations, actions)
    # Retain the deployment contract. Source run/checkpoint metadata belongs
    # to the input graph, while this file comes from an offline fit.
    contract = {"joint_names", "joint_stiffness", "joint_damping", "default_joint_pos",
                "command_names", "observation_names", "action_scale"}
    metadata = {p.key: p.value for p in model.metadata_props if p.key in contract}
    metadata.update(
        fit="fit_velstand_onset (simulation experiment)",
        base_sha256=hashlib.sha256(args.base.read_bytes()).hexdigest(),
        teacher_sha256=hashlib.sha256(args.teacher.read_bytes()).hexdigest(),
    )
    helper.set_model_props(model, metadata)
    onnx.checker.check_model(model)
    payload = model.SerializeToString()
    with args.output.open("xb") as output:
        output.write(payload)
    print(f"Fit on {len(actions)} examples. Wrote {args.output} (sha256 {hashlib.sha256(payload).hexdigest()})")


if __name__ == "__main__":
    main()

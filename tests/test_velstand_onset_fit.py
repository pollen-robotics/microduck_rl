"""The offline gait fit only changes the actor's final weight and bias."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

def test_fit_retains_normalizer_and_learns_from_raw_observations(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from fit_velstand_onset import fit_head

    rng = np.random.default_rng(17)
    observations = rng.normal(size=(64, 3)).astype(np.float32)
    mean = np.array([2, -4, 1], dtype=np.float32)
    scale = np.array([0.5, 2, 4], dtype=np.float32)
    initial = np.full((2, 3), 0.25, dtype=np.float32)
    bias = np.zeros(2, dtype=np.float32)
    graph = helper.make_graph(
        [helper.make_node("Sub", ["obs", "mean"], ["centered"]),
         helper.make_node("Div", ["centered", "scale"], ["hidden"]),
         helper.make_node("Gemm", ["hidden", "weight", "bias"], ["action"], transB=1)],
        "test_actor",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, 3])],
        [helper.make_tensor_value_info("action", TensorProto.FLOAT, [1, 2])],
        initializer=[numpy_helper.from_array(a, name) for a, name in
                     [(mean, "mean"), (scale, "scale"), (initial, "weight"), (bias, "bias")]],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    labels = (observations - mean) / scale @ np.array([[1, 2], [-1, 0], [0, 3]]) + [0.2, -0.4]
    fit_head(model, observations, labels, ridge=1e-8)

    assert list(model.graph.output) == list(graph.output)
    for before, after in zip(graph.initializer[:2], model.graph.initializer[:2]):
        assert before.SerializeToString() == after.SerializeToString()
    engine = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    predictions = [engine.run(None, {"obs": row[None]})[0][0] for row in observations]
    np.testing.assert_allclose(predictions, labels, atol=1e-4)

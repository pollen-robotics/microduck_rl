from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
from onnx import TensorProto, helper, numpy_helper

from scripts.onnx_bootstrap_checkpoint import (
    ACTION_DIM,
    CONTRACT_KEYS,
    EXPECTED_COMMANDS,
    EXPECTED_JOINTS,
    EXPECTED_OBSERVATIONS,
    LINEAR_KEYS,
    NORMALIZER_EPS,
    OBS_DIM,
    BootstrapCompatibilityError,
    bootstrap_checkpoint,
    inspect_onnx_policy,
    validate_checkpoint_state,
    validate_output_parity,
    validate_target_run,
)


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_ONNX = ROOT / ".tmp" / "codex" / "BEST_alpha_walking.onnx"
MODEL_ZERO = ROOT / "logs" / "rsl_rl" / "velocity" / "2026-08-29_08-50-53_base_walk" / "model_0.pt"


def _metadata() -> dict[str, str]:
    return {
        "run_path": "test",
        "joint_names": ",".join(EXPECTED_JOINTS),
        "joint_stiffness": ",".join(["1.000"] * ACTION_DIM),
        "joint_damping": ",".join(["0.000"] * ACTION_DIM),
        "default_joint_pos": ",".join(["0.000"] * ACTION_DIM),
        "command_names": ",".join(EXPECTED_COMMANDS),
        "observation_names": ",".join(EXPECTED_OBSERVATIONS),
        "action_scale": "1.0",
    }


def _write_policy(path: Path, *, activation: str = "Elu") -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    mean = rng.normal(0.0, 0.1, (1, OBS_DIM)).astype(np.float32)
    raw_std = rng.uniform(0.05, 2.0, (1, OBS_DIM)).astype(np.float32)
    divisor = raw_std + NORMALIZER_EPS
    tensors: dict[str, np.ndarray] = {
        "obs_normalizer._mean": mean,
        "normalizer_divisor": divisor,
    }
    for weight_key, bias_key, weight_shape, bias_shape in LINEAR_KEYS:
        tensors[weight_key] = rng.normal(0.0, 0.03, weight_shape).astype(np.float32)
        tensors[bias_key] = rng.normal(0.0, 0.01, bias_shape).astype(np.float32)

    initializers = [numpy_helper.from_array(value, name) for name, value in tensors.items()]
    nodes = [
        helper.make_node("Sub", ["obs", "obs_normalizer._mean"], ["sub"], name="normalizer_sub"),
        helper.make_node("Div", ["sub", "normalizer_divisor"], ["normalized"], name="normalizer_div"),
    ]
    previous = "normalized"
    for layer_index, (weight_key, bias_key, _, _) in enumerate(LINEAR_KEYS):
        gemm_output = "actions" if layer_index == len(LINEAR_KEYS) - 1 else f"linear_{layer_index}"
        nodes.append(
            helper.make_node(
                "Gemm",
                [previous, weight_key, bias_key],
                [gemm_output],
                name=f"gemm_{layer_index}",
                transB=1,
                alpha=1.0,
                beta=1.0,
            )
        )
        previous = gemm_output
        if layer_index < len(LINEAR_KEYS) - 1:
            activation_output = f"activation_{layer_index}"
            activation_kwargs = {"alpha": 1.0} if activation == "Elu" else {}
            nodes.append(
                helper.make_node(
                    activation,
                    [previous],
                    [activation_output],
                    name=f"activation_{layer_index}",
                    **activation_kwargs,
                )
            )
            previous = activation_output

    graph = helper.make_graph(
        nodes,
        "microduck_bootstrap_test",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, OBS_DIM])],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, ACTION_DIM])],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    helper.set_model_props(model, _metadata())
    onnx.save(model, path)
    return tensors


def _checkpoint() -> dict:
    actor: dict[str, torch.Tensor] = {
        "obs_normalizer._mean": torch.zeros(1, OBS_DIM),
        "obs_normalizer._var": torch.ones(1, OBS_DIM),
        "obs_normalizer._std": torch.ones(1, OBS_DIM),
        "obs_normalizer.count": torch.tensor(98304, dtype=torch.long),
        "distribution.std_param": torch.full((ACTION_DIM,), 0.75),
    }
    for weight_key, bias_key, weight_shape, bias_shape in LINEAR_KEYS:
        actor[weight_key] = torch.zeros(weight_shape)
        actor[bias_key] = torch.zeros(bias_shape)
    return {
        "actor_state_dict": actor,
        "critic_state_dict": {"kept": torch.tensor(3.0)},
        "optimizer_state_dict": {"state": {0: {"step": torch.tensor(1.0)}}, "param_groups": [{"params": [0]}]},
        "iter": 0,
        "infos": {},
    }


def test_direct_bootstrap_recovers_normalizer_and_matches_onnx(tmp_path: Path):
    onnx_path = tmp_path / "walking.onnx"
    tensors = _write_policy(onnx_path)
    source = inspect_onnx_policy(onnx_path)
    target = _checkpoint()
    exploration_std = target["actor_state_dict"]["distribution.std_param"].clone()

    output = bootstrap_checkpoint(target, source)
    actor = output["actor_state_dict"]

    assert torch.allclose(actor["obs_normalizer._mean"], torch.from_numpy(tensors["obs_normalizer._mean"]))
    expected_std = torch.from_numpy(tensors["normalizer_divisor"] - NORMALIZER_EPS)
    assert torch.allclose(actor["obs_normalizer._std"], expected_std)
    assert torch.allclose(actor["obs_normalizer._var"], expected_std.square())
    assert actor["obs_normalizer.count"].item() == 98304
    assert torch.equal(actor["distribution.std_param"], exploration_std)
    assert output["optimizer_state_dict"]["state"] == {}
    assert torch.equal(output["critic_state_dict"]["kept"], torch.tensor(3.0))

    report = validate_output_parity(source, output, samples=64, seed=11)
    assert report.max_abs_error < 2e-5


def test_bootstrap_does_not_mutate_input_and_can_override_count(tmp_path: Path):
    onnx_path = tmp_path / "walking.onnx"
    _write_policy(onnx_path)
    source = inspect_onnx_policy(onnx_path)
    target = _checkpoint()
    original = copy.deepcopy(target)

    output = bootstrap_checkpoint(target, source, normalizer_count=50_000_000)

    assert output["actor_state_dict"]["obs_normalizer.count"].item() == 50_000_000
    assert torch.equal(
        target["actor_state_dict"]["mlp.0.weight"], original["actor_state_dict"]["mlp.0.weight"]
    )
    assert target["optimizer_state_dict"]["state"].keys() == original["optimizer_state_dict"]["state"].keys()


def test_refuses_non_elu_graph_instead_of_guessing(tmp_path: Path):
    onnx_path = tmp_path / "relu.onnx"
    _write_policy(onnx_path, activation="Relu")

    with pytest.raises(BootstrapCompatibilityError, match="operation sequence"):
        inspect_onnx_policy(onnx_path)


def test_refuses_wrong_checkpoint_shape():
    checkpoint = _checkpoint()
    checkpoint["actor_state_dict"]["mlp.0.weight"] = torch.zeros(511, OBS_DIM)

    with pytest.raises(BootstrapCompatibilityError, match="mlp.0.weight"):
        validate_checkpoint_state(checkpoint)


@pytest.mark.skipif(not (OFFICIAL_ONNX.is_file() and MODEL_ZERO.is_file()), reason="local bootstrap artifacts unavailable")
def test_official_onnx_maps_to_requested_model_zero_with_tight_parity():
    source = inspect_onnx_policy(OFFICIAL_ONNX)
    checkpoint = torch.load(MODEL_ZERO, map_location="cpu", weights_only=False)
    validate_checkpoint_state(checkpoint)
    contract_path = validate_target_run(MODEL_ZERO, source)
    assert contract_path.is_file()
    assert all(key in source.metadata for key in CONTRACT_KEYS)

    output = bootstrap_checkpoint(checkpoint, source)
    report = validate_output_parity(source, output, samples=128, seed=20260829)

    assert report.max_abs_error < 2e-5
    assert torch.equal(
        output["actor_state_dict"]["distribution.std_param"],
        checkpoint["actor_state_dict"]["distribution.std_param"],
    )

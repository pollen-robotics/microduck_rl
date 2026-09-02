#!/usr/bin/env python3
"""Bootstrap an rsl_rl actor from the official MicroDuck walking ONNX.

This utility intentionally supports one architecture only:

    61 -> 512 -> 256 -> 128 -> 14, with ELU hidden activations

It validates the ONNX graph, deployment metadata, target run metadata, and
checkpoint tensor layout before copying anything. It then compares the
bootstrapped actor against ONNX Runtime on bounded, deterministic samples.

The exported ONNX contains the deterministic policy mean, but not PPO's
exploration standard deviation or the observation normalizer sample count.
Those values are therefore preserved from the target checkpoint unless an
explicit normalizer count is supplied. Optimizer moments are cleared by
default because they belong to the replaced random actor parameters.

Only load checkpoints from trusted sources: torch checkpoints use pickle.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
from onnx import numpy_helper


OBS_DIM = 61
ACTION_DIM = 14
HIDDEN_DIMS = (512, 256, 128)
NORMALIZER_EPS = 1e-2
MAX_PARITY_SAMPLES = 4096

EXPECTED_OPS = ("Sub", "Div", "Gemm", "Elu", "Gemm", "Elu", "Gemm", "Elu", "Gemm")
EXPECTED_OBSERVATIONS = (
    "base_ang_vel",
    "projected_gravity",
    "joint_pos",
    "joint_vel",
    "actions",
    "command",
    "head_command",
    "body_command",
)
EXPECTED_COMMANDS = ("twist", "head_pose", "body_pose")
EXPECTED_JOINTS = (
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
)
CONTRACT_KEYS = (
    "joint_names",
    "joint_stiffness",
    "joint_damping",
    "default_joint_pos",
    "command_names",
    "observation_names",
    "action_scale",
)
LINEAR_KEYS = (
    ("mlp.0.weight", "mlp.0.bias", (512, 61), (512,)),
    ("mlp.2.weight", "mlp.2.bias", (256, 512), (256,)),
    ("mlp.4.weight", "mlp.4.bias", (128, 256), (128,)),
    ("mlp.6.weight", "mlp.6.bias", (14, 128), (14,)),
)


class BootstrapCompatibilityError(RuntimeError):
    """Raised when direct parameter mapping cannot be proven safe."""


@dataclass(frozen=True)
class OnnxPolicy:
    path: Path
    mean: np.ndarray
    effective_divisor: np.ndarray
    actor_tensors: dict[str, np.ndarray]
    metadata: dict[str, str]


@dataclass(frozen=True)
class ParityReport:
    samples: int
    max_abs_error: float
    mean_abs_error: float
    rmse: float
    max_tolerance_ratio: float


def _fail(message: str) -> None:
    raise BootstrapCompatibilityError(message)


def _value_info_shape(value_info: onnx.ValueInfoProto) -> tuple[int, ...]:
    dims: list[int] = []
    for dim in value_info.type.tensor_type.shape.dim:
        if not dim.HasField("dim_value"):
            _fail(f"dynamic dimension is not supported for {value_info.name!r}")
        dims.append(dim.dim_value)
    return tuple(dims)


def _node_attributes(node: onnx.NodeProto) -> dict[str, Any]:
    return {attribute.name: onnx.helper.get_attribute_value(attribute) for attribute in node.attribute}


def _require_serial_edge(node: onnx.NodeProto, previous_output: str) -> None:
    if not node.input or node.input[0] != previous_output:
        _fail(f"node {node.name or node.op_type!r} is not serially connected to {previous_output!r}")


def _metadata(model: onnx.ModelProto) -> dict[str, str]:
    return {item.key: item.value for item in model.metadata_props}


def _validate_official_metadata(
    metadata: dict[str, str], *, allow_episodic_zero_command: bool = False
) -> None:
    missing = [key for key in CONTRACT_KEYS if key not in metadata]
    if missing:
        _fail(f"ONNX metadata is missing deployment contract keys: {missing}")
    if tuple(metadata["observation_names"].split(",")) != EXPECTED_OBSERVATIONS:
        _fail("ONNX observation order does not match the MicroDuck 61D actor contract")
    command_names = tuple(metadata["command_names"].split(","))
    episodic_command_contract = allow_episodic_zero_command and command_names == (
        "twist",
    )
    if command_names != EXPECTED_COMMANDS and not episodic_command_contract:
        _fail("ONNX command order does not match twist, head_pose, body_pose")
    if tuple(metadata["joint_names"].split(",")) != EXPECTED_JOINTS:
        _fail("ONNX joint order does not match the 14-servo MicroDuck contract")
    try:
        action_scale = float(metadata["action_scale"])
    except ValueError as exc:
        raise BootstrapCompatibilityError("ONNX action_scale is not numeric") from exc
    if action_scale != 1.0:
        _fail(f"ONNX action_scale must be 1.0, got {action_scale}")


def inspect_onnx_policy(
    path: Path, *, allow_episodic_zero_command: bool = False
) -> OnnxPolicy:
    """Load and strictly validate the direct-map ONNX architecture."""
    path = path.resolve()
    if not path.is_file():
        _fail(f"ONNX file does not exist: {path}")

    model = onnx.load(path)
    try:
        onnx.checker.check_model(model)
    except onnx.checker.ValidationError as exc:
        raise BootstrapCompatibilityError(f"invalid ONNX model: {exc}") from exc

    if len(model.graph.input) != 1 or model.graph.input[0].name != "obs":
        _fail("ONNX must expose exactly one input named 'obs'")
    if len(model.graph.output) != 1 or model.graph.output[0].name != "actions":
        _fail("ONNX must expose exactly one output named 'actions'")
    if _value_info_shape(model.graph.input[0]) != (1, OBS_DIM):
        _fail(f"ONNX input must be [1, {OBS_DIM}]")
    if _value_info_shape(model.graph.output[0]) != (1, ACTION_DIM):
        _fail(f"ONNX output must be [1, {ACTION_DIM}]")

    nodes = list(model.graph.node)
    ops = tuple(node.op_type for node in nodes)
    if ops != EXPECTED_OPS:
        _fail(f"unsupported ONNX operation sequence: {ops}; expected {EXPECTED_OPS}")

    initializers = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    sub, div = nodes[:2]
    if list(sub.input) != ["obs", "obs_normalizer._mean"]:
        _fail("normalizer Sub must compute obs - obs_normalizer._mean")
    _require_serial_edge(div, sub.output[0])
    if len(div.input) != 2 or div.input[1] not in initializers:
        _fail("normalizer Div must use a constant effective divisor")
    if sub.input[1] not in initializers:
        _fail("normalizer mean initializer is missing")

    mean = np.asarray(initializers[sub.input[1]], dtype=np.float32)
    divisor = np.asarray(initializers[div.input[1]], dtype=np.float32)
    if mean.shape != (1, OBS_DIM) or divisor.shape != (1, OBS_DIM):
        _fail("normalizer mean and divisor must both have shape [1, 61]")
    if not np.isfinite(mean).all() or not np.isfinite(divisor).all():
        _fail("normalizer contains non-finite values")
    divisor_tolerance = np.finfo(np.float32).eps
    if not np.all(divisor >= NORMALIZER_EPS - divisor_tolerance):
        _fail("effective divisor must be at least rsl_rl normalizer epsilon 0.01")

    actor_tensors: dict[str, np.ndarray] = {}
    previous_output = div.output[0]
    for layer_index, (weight_key, bias_key, weight_shape, bias_shape) in enumerate(LINEAR_KEYS):
        gemm = nodes[2 + layer_index * 2]
        _require_serial_edge(gemm, previous_output)
        if len(gemm.input) != 3 or gemm.input[1:] != [weight_key, bias_key]:
            _fail(f"Gemm layer {layer_index} does not map to {weight_key!r} and {bias_key!r}")
        attrs = _node_attributes(gemm)
        if attrs.get("transB", 0) != 1 or float(attrs.get("alpha", 1.0)) != 1.0 or float(
            attrs.get("beta", 1.0)
        ) != 1.0:
            _fail(f"Gemm layer {layer_index} uses unsupported attributes: {attrs}")
        for key, expected_shape in ((weight_key, weight_shape), (bias_key, bias_shape)):
            if key not in initializers:
                _fail(f"missing ONNX initializer {key!r}")
            value = np.asarray(initializers[key], dtype=np.float32)
            if value.shape != expected_shape:
                _fail(f"initializer {key!r} has shape {value.shape}, expected {expected_shape}")
            if not np.isfinite(value).all():
                _fail(f"initializer {key!r} contains non-finite values")
            actor_tensors[key] = value
        previous_output = gemm.output[0]

        if layer_index < len(LINEAR_KEYS) - 1:
            activation = nodes[3 + layer_index * 2]
            _require_serial_edge(activation, previous_output)
            alpha = float(_node_attributes(activation).get("alpha", 1.0))
            if alpha != 1.0:
                _fail(f"ELU layer {layer_index} has alpha={alpha}, expected 1.0")
            previous_output = activation.output[0]

    if nodes[-1].output != ["actions"]:
        _fail("final Gemm must directly produce the 'actions' output")

    expected_initializers = {"obs_normalizer._mean", div.input[1]}
    for weight_key, bias_key, _, _ in LINEAR_KEYS:
        expected_initializers.update((weight_key, bias_key))
    extras = set(initializers) - expected_initializers
    missing = expected_initializers - set(initializers)
    if extras or missing:
        _fail(f"unexpected ONNX initializer set; extras={sorted(extras)}, missing={sorted(missing)}")

    metadata = _metadata(model)
    _validate_official_metadata(
        metadata, allow_episodic_zero_command=allow_episodic_zero_command
    )
    return OnnxPolicy(path, mean.copy(), divisor.copy(), actor_tensors, metadata)


def validate_checkpoint_state(checkpoint: dict[str, Any], *, require_iteration_zero: bool = True) -> None:
    """Validate the rsl_rl checkpoint fields needed for direct mapping."""
    if "actor_state_dict" not in checkpoint or not isinstance(checkpoint["actor_state_dict"], dict):
        _fail("checkpoint has no actor_state_dict")
    if require_iteration_zero and checkpoint.get("iter") != 0:
        _fail(f"bounded bootstrap accepts only an iteration-0 checkpoint, got iter={checkpoint.get('iter')!r}")

    actor = checkpoint["actor_state_dict"]
    expected_shapes = {
        "obs_normalizer._mean": (1, OBS_DIM),
        "obs_normalizer._var": (1, OBS_DIM),
        "obs_normalizer._std": (1, OBS_DIM),
        "obs_normalizer.count": (),
        "distribution.std_param": (ACTION_DIM,),
    }
    for weight_key, bias_key, weight_shape, bias_shape in LINEAR_KEYS:
        expected_shapes[weight_key] = weight_shape
        expected_shapes[bias_key] = bias_shape

    for key, expected_shape in expected_shapes.items():
        if key not in actor or not isinstance(actor[key], torch.Tensor):
            _fail(f"checkpoint actor is missing tensor {key!r}")
        if tuple(actor[key].shape) != expected_shape:
            _fail(f"checkpoint tensor {key!r} has shape {tuple(actor[key].shape)}, expected {expected_shape}")
        if key != "obs_normalizer.count" and not torch.isfinite(actor[key]).all():
            _fail(f"checkpoint tensor {key!r} contains non-finite values")


def _extract_actor_terms(env_yaml: str) -> tuple[str, ...]:
    normalized = env_yaml.replace("\r\n", "\n")
    marker = "observations:\n  actor:\n    terms:\n"
    start = normalized.find(marker)
    if start < 0:
        _fail("target env.yaml has no observations.actor.terms block")
    start += len(marker)
    end = normalized.find("\n  critic:\n", start)
    if end < 0:
        _fail("target env.yaml has no critic block after actor observations")
    block = normalized[start:end]
    return tuple(re.findall(r"^      ([A-Za-z_][A-Za-z0-9_]*):\s*$", block, flags=re.MULTILINE))


def _validate_agent_yaml(agent_yaml: str) -> None:
    normalized = agent_yaml.replace("\r\n", "\n")
    match = re.search(r"^actor:\s*\n(?P<body>.*?)(?=^critic:\s*$)", normalized, flags=re.MULTILINE | re.DOTALL)
    if match is None:
        _fail("target agent.yaml has no actor configuration block")
    body = match.group("body")
    hidden_match = re.search(
        r"^  hidden_dims:.*?\n(?P<dims>(?:^  - \d+\s*\n)+)", body, flags=re.MULTILINE | re.DOTALL
    )
    if hidden_match is None:
        _fail("target agent.yaml has no actor hidden_dims")
    hidden_dims = tuple(int(value) for value in re.findall(r"^  - (\d+)\s*$", hidden_match.group("dims"), re.MULTILINE))
    if hidden_dims != HIDDEN_DIMS:
        _fail(f"target actor hidden_dims are {hidden_dims}, expected {HIDDEN_DIMS}")
    required_patterns = {
        "ELU activation": r"^  activation:\s*elu\s*$",
        "observation normalization": r"^  obs_normalization:\s*true\s*$",
        "MLPModel actor": r"^  class_name:\s*MLPModel\s*$",
    }
    for description, pattern in required_patterns.items():
        if re.search(pattern, body, flags=re.MULTILINE) is None:
            _fail(f"target agent.yaml does not prove {description}")


def _load_contract(path: Path) -> tuple[dict[str, str], tuple[int, ...], tuple[int, ...]]:
    model = onnx.load(path)
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        _fail(f"target contract ONNX must have one input and one output: {path}")
    return _metadata(model), _value_info_shape(model.graph.input[0]), _value_info_shape(model.graph.output[0])


def validate_target_run(
    checkpoint_path: Path,
    source: OnnxPolicy,
    *,
    target_contract_onnx: Path | None = None,
    allow_episodic_zero_command: bool = False,
) -> Path:
    """Prove architecture and semantic observation compatibility from run sidecars."""
    run_dir = checkpoint_path.resolve().parent
    agent_yaml_path = run_dir / "params" / "agent.yaml"
    env_yaml_path = run_dir / "params" / "env.yaml"
    if not agent_yaml_path.is_file() or not env_yaml_path.is_file():
        _fail(f"checkpoint run is missing params/agent.yaml or params/env.yaml under {run_dir}")

    _validate_agent_yaml(agent_yaml_path.read_text(encoding="utf-8"))
    actor_terms = _extract_actor_terms(env_yaml_path.read_text(encoding="utf-8"))
    if actor_terms != EXPECTED_OBSERVATIONS:
        _fail(f"target actor observation order is {actor_terms}, expected {EXPECTED_OBSERVATIONS}")

    if target_contract_onnx is None:
        candidates = sorted(run_dir.glob("*.onnx"))
        if len(candidates) != 1:
            _fail(
                f"expected exactly one target contract ONNX beside the checkpoint, found {len(candidates)}; "
                "pass --target-contract-onnx explicitly"
            )
        target_contract_onnx = candidates[0]
    target_contract_onnx = target_contract_onnx.resolve()
    metadata, input_shape, output_shape = _load_contract(target_contract_onnx)
    if input_shape != (1, OBS_DIM) or output_shape != (1, ACTION_DIM):
        _fail(f"target contract ONNX has incompatible shapes {input_shape} -> {output_shape}")
    for key in CONTRACT_KEYS:
        if (
            key == "command_names"
            and allow_episodic_zero_command
            and source.metadata.get(key) == "twist"
            and tuple(metadata.get(key, "").split(",")) == EXPECTED_COMMANDS
        ):
            continue
        if metadata.get(key) != source.metadata.get(key):
            _fail(f"target deployment metadata differs from source for {key!r}")
    return target_contract_onnx


def bootstrap_checkpoint(
    checkpoint: dict[str, Any],
    source: OnnxPolicy,
    *,
    normalizer_count: int | None = None,
    reset_optimizer_state: bool = True,
) -> dict[str, Any]:
    """Return a deep-copied checkpoint with the ONNX deterministic actor mapped in."""
    validate_checkpoint_state(checkpoint)
    if normalizer_count is not None and normalizer_count <= 0:
        _fail("normalizer_count must be positive when supplied")

    output = copy.deepcopy(checkpoint)
    actor = output["actor_state_dict"]

    def assign(key: str, value: np.ndarray) -> None:
        target = actor[key]
        actor[key] = torch.as_tensor(value.copy(), dtype=target.dtype, device=target.device)

    assign("obs_normalizer._mean", source.mean)
    raw_std = np.maximum(source.effective_divisor - NORMALIZER_EPS, 0.0)
    assign("obs_normalizer._std", raw_std)
    assign("obs_normalizer._var", np.square(raw_std))
    if normalizer_count is not None:
        count = actor["obs_normalizer.count"]
        actor["obs_normalizer.count"] = torch.tensor(normalizer_count, dtype=count.dtype, device=count.device)
    for key, value in source.actor_tensors.items():
        assign(key, value)

    if reset_optimizer_state and isinstance(output.get("optimizer_state_dict"), dict):
        output["optimizer_state_dict"]["state"] = {}
    return output


def actor_actions(checkpoint: dict[str, Any], observations: np.ndarray) -> np.ndarray:
    """Evaluate the deterministic checkpoint actor directly from its state dict."""
    actor = checkpoint["actor_state_dict"]
    x = torch.from_numpy(np.asarray(observations, dtype=np.float32))
    mean = actor["obs_normalizer._mean"].detach().cpu()
    std = actor["obs_normalizer._std"].detach().cpu()
    x = (x - mean) / (std + NORMALIZER_EPS)
    for layer_index, (weight_key, bias_key, _, _) in enumerate(LINEAR_KEYS):
        weight = actor[weight_key].detach().cpu()
        bias = actor[bias_key].detach().cpu()
        x = F.linear(x, weight, bias)
        if layer_index < len(LINEAR_KEYS) - 1:
            x = F.elu(x, alpha=1.0)
    return x.numpy()


def parity_observations(source: OnnxPolicy, samples: int, seed: int) -> np.ndarray:
    if samples < 4 or samples > MAX_PARITY_SAMPLES:
        _fail(f"parity samples must be between 4 and {MAX_PARITY_SAMPLES}")
    rng = np.random.default_rng(seed)
    normalized = rng.standard_normal((samples, OBS_DIM), dtype=np.float32)
    observations = source.mean + normalized * source.effective_divisor
    observations[0] = 0.0
    observations[1] = source.mean
    observations[2] = source.mean + source.effective_divisor
    observations[3] = source.mean - source.effective_divisor
    return np.asarray(observations, dtype=np.float32)


def validate_output_parity(
    source: OnnxPolicy,
    checkpoint: dict[str, Any],
    *,
    samples: int = 512,
    seed: int = 20260829,
    atol: float = 2e-5,
    rtol: float = 1e-5,
) -> ParityReport:
    """Require checkpoint and ONNX outputs to agree on bounded samples."""
    observations = parity_observations(source, samples, seed)
    session = ort.InferenceSession(str(source.path), providers=["CPUExecutionProvider"])
    expected = np.concatenate(
        [session.run(["actions"], {"obs": row[None, :]})[0] for row in observations], axis=0
    )
    actual = actor_actions(checkpoint, observations)
    difference = np.abs(actual - expected)
    tolerance = atol + rtol * np.abs(expected)
    ratio = difference / np.maximum(tolerance, np.finfo(np.float32).tiny)
    report = ParityReport(
        samples=samples,
        max_abs_error=float(difference.max()),
        mean_abs_error=float(difference.mean()),
        rmse=float(np.sqrt(np.mean(np.square(actual - expected)))),
        max_tolerance_ratio=float(ratio.max()),
    )
    if not np.all(difference <= tolerance):
        _fail(
            "bootstrapped actor failed ONNX output parity: "
            f"max_abs={report.max_abs_error:.3e}, max_tolerance_ratio={report.max_tolerance_ratio:.3f}"
        )
    return report


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_checkpoint(checkpoint: dict[str, Any], output_path: Path) -> None:
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f"{output_path.name}.", suffix=".tmp", dir=output_path.parent, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
        torch.save(checkpoint, temporary_path)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, default=Path(".tmp/codex/BEST_alpha_walking.onnx"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("logs/rsl_rl/velocity/2026-08-29_08-50-53_base_walk/model_0.pt"),
    )
    parser.add_argument("--output", type=Path, help="new checkpoint path; required unless --validate-only")
    parser.add_argument("--target-contract-onnx", type=Path)
    parser.add_argument(
        "--allow-episodic-zero-command",
        action="store_true",
        help=(
            "accept a manufacturer episodic policy whose metadata lists only "
            "twist while its 61D observation keeps zero-padded head/body slots"
        ),
    )
    parser.add_argument("--normalizer-count", type=int, help="override count; default preserves target checkpoint count")
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--atol", type=float, default=2e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--validate-only", action="store_true", help="prove mapping and parity without writing output")
    parser.add_argument(
        "--keep-optimizer-state",
        action="store_true",
        help="retain stale optimizer moments; unsafe for normal bootstrap use",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.is_file():
        _fail(f"checkpoint does not exist: {checkpoint_path}")
    if not args.validate_only and args.output is None:
        _fail("--output is required unless --validate-only is used")
    if args.output is not None:
        output_path = args.output.resolve()
        if output_path == checkpoint_path:
            _fail("refusing to overwrite the input checkpoint")
        if output_path.exists():
            _fail(f"refusing to overwrite existing output: {output_path}")

    source = inspect_onnx_policy(
        args.onnx,
        allow_episodic_zero_command=args.allow_episodic_zero_command,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    validate_checkpoint_state(checkpoint)
    contract_path = validate_target_run(
        checkpoint_path,
        source,
        target_contract_onnx=args.target_contract_onnx,
        allow_episodic_zero_command=args.allow_episodic_zero_command,
    )
    bootstrapped = bootstrap_checkpoint(
        checkpoint,
        source,
        normalizer_count=args.normalizer_count,
        reset_optimizer_state=not args.keep_optimizer_state,
    )
    parity = validate_output_parity(
        source,
        bootstrapped,
        samples=args.samples,
        seed=args.seed,
        atol=args.atol,
        rtol=args.rtol,
    )

    actor = bootstrapped["actor_state_dict"]
    infos = bootstrapped.setdefault("infos", {})
    infos["onnx_bootstrap"] = {
        "mapping": "direct",
        "source_onnx": str(source.path),
        "source_onnx_sha256": _sha256(source.path),
        "target_checkpoint": str(checkpoint_path),
        "target_checkpoint_sha256": _sha256(checkpoint_path),
        "target_contract_onnx": str(contract_path),
        "normalizer_epsilon": NORMALIZER_EPS,
        "normalizer_count": int(actor["obs_normalizer.count"].item()),
        "exploration_std": "preserved_from_target_checkpoint",
        "optimizer_state_reset": not args.keep_optimizer_state,
        "episodic_zero_command_contract": args.allow_episodic_zero_command,
        "parity_samples": parity.samples,
        "parity_max_abs_error": parity.max_abs_error,
        "parity_mean_abs_error": parity.mean_abs_error,
        "parity_rmse": parity.rmse,
        "created_utc": datetime.now(UTC).isoformat(),
    }

    print("Direct mapping proved compatible")
    print(f"  ONNX graph: {OBS_DIM} -> {HIDDEN_DIMS[0]} -> {HIDDEN_DIMS[1]} -> {HIDDEN_DIMS[2]} -> {ACTION_DIM}")
    print(f"  target contract: {contract_path}")
    print(f"  normalizer count: {int(actor['obs_normalizer.count'].item())} (ONNX does not encode count)")
    print(f"  exploration std: preserved from target checkpoint (ONNX is deterministic)")
    print(f"  optimizer state reset: {not args.keep_optimizer_state}")
    print(
        f"Parity passed on {parity.samples} samples: max_abs={parity.max_abs_error:.3e}, "
        f"mean_abs={parity.mean_abs_error:.3e}, rmse={parity.rmse:.3e}"
    )

    if args.validate_only:
        print("Validation-only mode: no checkpoint written")
        return 0

    _save_checkpoint(bootstrapped, args.output)
    saved = torch.load(args.output, map_location="cpu", weights_only=False)
    validate_checkpoint_state(saved)
    saved_parity = validate_output_parity(
        source,
        saved,
        samples=args.samples,
        seed=args.seed,
        atol=args.atol,
        rtol=args.rtol,
    )
    print(f"Wrote {args.output.resolve()}")
    print(f"Reload parity max_abs={saved_parity.max_abs_error:.3e}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BootstrapCompatibilityError as exc:
        raise SystemExit(f"ONNX bootstrap refused: {exc}") from exc

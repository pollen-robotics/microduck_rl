"""Distill fixed-head Flamingo behavior into a single deployable ONNX actor."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def elu(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0.0, x, np.expm1(x))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-onnx", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ridge", type=float, default=25.0)
    args = parser.parse_args()

    model = onnx.load(args.teacher_onnx)
    init = {item.name: item for item in model.graph.initializer}
    arrays = {name: numpy_helper.to_array(item).copy() for name, item in init.items()}
    dataset = np.load(args.data)
    obs = dataset["observations"].astype(np.float64)
    target = dataset["actions"].astype(np.float64)

    mean = arrays["obs_normalizer._mean"].astype(np.float64)
    div = arrays["onnx::Div_24"].astype(np.float64)
    hidden = (obs - mean) / div
    for index in (0, 2, 4):
        hidden = elu(
            hidden @ arrays[f"mlp.{index}.weight"].astype(np.float64).T
            + arrays[f"mlp.{index}.bias"].astype(np.float64)
        )

    design = np.concatenate([hidden, np.ones((hidden.shape[0], 1))], axis=1)
    old = np.concatenate(
        [arrays["mlp.6.weight"].astype(np.float64), arrays["mlp.6.bias"].astype(np.float64)[:, None]],
        axis=1,
    ).T
    ridge_eye = np.eye(design.shape[1], dtype=np.float64) * args.ridge
    fitted = np.linalg.solve(
        design.T @ design + ridge_eye,
        design.T @ target + args.ridge * old,
    )
    prediction = design @ fitted
    baseline = design @ old

    # The requested pose uses legs only. All four head-chain targets remain at
    # HOME, preventing pitch nodding as well as yaw/roll turning.
    fitted[:, 5:9] = 0.0
    prediction[:, 5:9] = 0.0

    leg_indices = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
    print(f"samples={len(obs)} ridge={args.ridge:g}")
    print(f"baseline_leg_rmse={np.sqrt(np.mean((baseline[:, leg_indices] - target[:, leg_indices]) ** 2)):.6f}")
    print(f"student_leg_rmse={np.sqrt(np.mean((prediction[:, leg_indices] - target[:, leg_indices]) ** 2)):.6f}")

    weight = fitted[:-1, :].T.astype(np.float32)
    bias = fitted[-1, :].astype(np.float32)
    init["mlp.6.weight"].CopyFrom(numpy_helper.from_array(weight, "mlp.6.weight"))
    init["mlp.6.bias"].CopyFrom(numpy_helper.from_array(bias, "mlp.6.bias"))
    model.doc_string = (
        "Forward-head Flamingo student distilled from the official balance actor. "
        "Leg controls imitate the stable fixed-head teacher rollout; all four head joints remain at HOME."
    )
    onnx.checker.check_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)


if __name__ == "__main__":
    main()

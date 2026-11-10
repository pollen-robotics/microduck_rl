"""Per-joint BAM servo models, for robots mixing servo variants.

microduck runs metal-gear XL330-M288-T (``xl330m288t-MG``, ROBOTIS V4 firmware
enforcing a 1 A supply current limit in position control) on some joints and
plastic-gear XL330 (``xl330``) on the others. Both are BAM m6 models of the
same voltage-controlled servo, so they can share ONE actuator: every BAM
parameter becomes a per-joint array, which the control law, torque equation and
friction budget broadcast over the joints (numpy on CPU, torch ``(N, J)`` in
warp).

Keeping one actuator matters: a BAM actuator owns the battery (per-env voltage,
load-dependent sag from the sum of ITS joints' torques), so splitting the joints
into two actuators would give them two uncorrelated batteries, each seeing only
part of the load.

Kept free of torch/mjlab imports so the CPU rehearsal (scripts/infer_policy.py)
can use it without the warp import cost.
"""

from __future__ import annotations

import re
from typing import Callable, Sequence

import numpy as np
from bam.model import Model, load_model

# Current limit [A] for joints whose servo model has none: large enough that the
# supply-current limiter never binds, where it reduces exactly to the plain XL330
# voltage law.
NO_CURRENT_LIMIT = 1.0e6


def joint_motor_names(
    joint_names: Sequence[str],
    default_motor: str,
    per_joint_motors: Sequence[tuple[str, str]],
) -> list[str]:
    """Bundled BAM motor of each joint: ``default_motor`` unless a
    ``(joint_name_regex, motor)`` entry fully matches its name (last one wins)."""
    names = []
    for joint in joint_names:
        motor = default_motor
        for pattern, candidate in per_joint_motors:
            if re.fullmatch(pattern, joint):
                motor = candidate
        names.append(motor)
    return names


def resolve_joint_models(
    joint_names: Sequence[str],
    default_motor: str,
    variant: str,
    per_joint_motors: Sequence[tuple[str, str]],
    default_model: Model | None = None,
) -> tuple[Model, list[Model], list[str]]:
    """Load the BAM model of every joint.

    Returns ``(carrier, joint_models, motor_names)``. ``carrier`` is the model
    whose actuator (control law) is used for all joints: the most derived
    actuator class among the variants, the others being its special cases (the
    plain XL330 is the supply-current limited XL330 without a limit). Its
    parameters are still scalars; see :func:`vectorize_parameters`.
    """
    motor_names = joint_motor_names(joint_names, default_motor, per_joint_motors)
    models: dict[str, Model] = {}
    if default_model is not None:
        models[default_motor] = default_model
    for name in motor_names:
        if name not in models:
            models[name] = load_model(motor_name=name, model=variant)

    classes = {type(m.actuator) for m in models.values()}
    carriers = [
        m for m in models.values()
        if all(issubclass(type(m.actuator), c) for c in classes)
    ]
    if not carriers:
        raise ValueError(
            "per-joint BAM motors need actuator classes in a single inheritance "
            f"chain, got {sorted(c.__name__ for c in classes)}"
        )
    return carriers[0], [models[n] for n in motor_names], motor_names


def vectorize_parameters(
    carrier: Model,
    joint_models: Sequence[Model],
    convert: Callable[[list[float]], object] = lambda v: np.asarray(v, dtype=float),
) -> None:
    """Replace every parameter of ``carrier`` by a per-joint array, in
    ``joint_models`` order (``convert`` builds it: numpy by default, a torch
    tensor on the simulation device in warp)."""
    for key, parameter in carrier.get_parameters().items():
        values = []
        for m in joint_models:
            p = m.get_parameters().get(key)
            if p is not None:
                values.append(float(p.value))
            elif key == "current_limit":
                values.append(NO_CURRENT_LIMIT)
            else:
                raise ValueError(f"BAM parameter {key!r} missing from a per-joint model")
        parameter.value = convert(values)


def describe(joint_names: Sequence[str], motor_names: Sequence[str]) -> str:
    groups: dict[str, list[str]] = {}
    for joint, motor in zip(joint_names, motor_names):
        groups.setdefault(motor, []).append(joint)
    return ", ".join(f"{m} x{len(j)} ({', '.join(j)})" for m, j in groups.items())

"""Calibrate the fitted actor's low-rate in-place yaw response in duck-sim."""

import numpy as np
from onnx import helper, numpy_helper


def calibrate_slow_turns(model):
    """Prepend a stateless correction to the yaw feature before the baked normalizer.

    With almost stationary legs, a stronger cue gets the first step underway.
    Once walking, the fitted actor needs a slightly larger positive cue and a
    smaller negative cue. The correction fades with distance from pure ±0.5
    rad/s and with translational speed. All units refer to raw observations.
    """
    graph = model.graph
    observation = graph.input[0]
    shape = [d.dim_value for d in observation.type.tensor_type.shape.dim]
    metadata = {p.key: p.value for p in model.metadata_props}
    joints = metadata.get("joint_names", "").split(",")
    terms = metadata.get("observation_names", "").split(",")
    if (shape != [1, 61] or len(joints) != 14
            or terms[:6] != ["base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions", "command"]):
        raise ValueError("Expected the exported 61-D VelStand observation layout")
    # Joint velocity starts after gyro, projected gravity and the 14 positions.
    leg_columns = [20 + i for i, name in enumerate(joints) if name.startswith(("left_", "right_"))]
    if len(leg_columns) != 10:
        raise ValueError("Expected five servo joint velocities per leg")

    nodes = []

    def constant(name, value, dtype=np.float32):
        graph.initializer.append(numpy_helper.from_array(np.asarray(value, dtype=dtype), name))
        return name

    def op(kind, output, *inputs, **attrs):
        nodes.append(helper.make_node(kind, inputs, [output], name=output, **attrs))
        return output

    def columns(name, indices):
        return op("Gather", name, observation.name, constant(name + "_idx", indices, np.int64), axis=1)

    yaw = columns("yaw", [50])
    legs = columns("leg_velocity", leg_columns)
    axis = constant("axis", [1], np.int64)
    half = constant("half", 0.5)
    one = constant("one", 1)
    speed = op("ReduceL2", "leg_speed", legs, axis, keepdims=1)
    relative_speed = op("Div", "relative_speed", speed, constant("velocity_reference", 3.0))
    stillness = op("Relu", "stillness", op("Sub", "below_reference", one, relative_speed))

    # For a pure ±0.5 command the window below has weight 0.5. The internal
    # cue starts at +1.0 / −0.70 rad/s with still legs, and reaches +0.575 /
    # −0.275 rad/s when the leg joint velocity norm reaches 3 rad/s.
    left = op("Greater", "left", yaw, constant("zero", 0))
    steady = op("Where", "steady_shift", left, constant("left_shift", 0.15), constant("right_shift", 0.45))
    signed_stillness = op("Mul", "signed_stillness", stillness, op("Sign", "direction", yaw))
    onset = op("Mul", "onset_shift", constant("startup_gain", 0.85), signed_stillness)

    # max(0, 0.5 − (0.5/0.15) * abs(abs(yaw) − 0.5) − (0.5/0.10) * speed_xy)
    offset = op("Sub", "yaw_from_target", op("Abs", "abs_yaw", yaw), half)
    distance = op("Abs", "distance_from_target", offset)
    linear_speed = op("ReduceL2", "linear_speed", columns("linear_command", [48, 49]), axis, keepdims=1)
    angular = op("Mul", "frequency_distance", distance, constant("frequency_slope", 0.5 / 0.15))
    linear = op("Mul", "linear_distance", linear_speed, constant("linear_slope", 0.5 / 0.10))
    near_turn = op("Sub", "near_low_turn", half, op("Add", "command_distance", angular, linear))
    weight = op("Relu", "turn_weight", near_turn)
    shift = op("Mul", "yaw_shift", weight, op("Add", "calibration", steady, onset))
    selector = np.zeros(61, dtype=np.float32)
    selector[50] = 1
    correction = op("Mul", "turn_correction", shift, constant("yaw_selection", selector))
    adjusted = op("Add", "tracking_observation", observation.name, correction)

    for node in graph.node:
        for i, name in enumerate(node.input):
            if name == observation.name:
                node.input[i] = adjusted
    original = list(graph.node)
    del graph.node[:]
    graph.node.extend(nodes)
    graph.node.extend(original)

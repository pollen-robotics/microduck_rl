"""Regression tests for the VelStand demonstration labels and output-layer fit."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


def test_head_controls_replay_the_same_start(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from bench_velstand_onset import _head_placements

    used_seeds = set()
    for fold in ("primary", "reserve"):
        rows, recovery = _head_placements(fold)
        starts = {seed: (pose, ticks, head) for seed, pose, ticks, head, control in rows if not control}
        assert len(starts) == 24 and used_seeds.isdisjoint(starts)
        used_seeds.update(starts)
        controls = [row for row in rows if row[-1]]
        assert len(controls) == 4
        for seed, pose, ticks, head, _ in controls:
            assert starts[seed][:2] == (pose, ticks)
            assert any(starts[seed][2]) and not any(head)
        assert len({s for group in recovery for s in group}) == 120


def test_turn_calibration_preserves_other_observations_and_commands(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from velstand_turn_calibration import calibrate_slow_turns

    # Expose the observation that would enter the fitted actor's normalizer.
    graph = helper.make_graph(
        [helper.make_node("Identity", ["obs"], ["adjusted"])], "observation_probe",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, 61])],
        [helper.make_tensor_value_info("adjusted", TensorProto.FLOAT, [1, 61])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=8)
    leg = ["hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle"]
    neck = ["neck_pitch", "head_pitch", "head_yaw", "head_roll"]
    joints = [f"left_{name}" for name in leg] + neck + [f"right_{name}" for name in leg]
    helper.set_model_props(model, {
        "joint_names": ",".join(joints),
        "observation_names": "base_ang_vel,projected_gravity,joint_pos,joint_vel,actions,command,head_command,body_command",
    })
    calibrate_slow_turns(model)
    engine = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    for vx, vy, yaw, leg_speed, expected in (
        (0, 0, 0.5, 0, 1.0), (0, 0, -0.5, 0, -0.70),
        (0, 0, 0.5, 3, 0.575), (0, 0, -0.5, 4, -0.275),
        (0, 0, 0.5, 1.5, 0.7875), (0, 0, 0, 2, 0), (0.15, 0, 0, 0, 0),
        (0.15, 0, 0.5, 0, 0.5), (0, 0.12, -0.5, 2, -0.5),
        (0, 0, -0.30, 0, -0.30), (0, 0, 1.0, 0, 1.0),
    ):
        obs = np.linspace(-0.12, 0.12, 61, dtype=np.float32)
        obs[20:34] = 0
        obs[33] = leg_speed
        obs[25:29] = 5  # Head motion alone must not attenuate the first step.
        obs[48:51] = [vx, vy, yaw]
        adjusted = engine.run(None, {"obs": obs[None]})[0][0]
        np.testing.assert_array_equal(adjusted[:50], obs[:50])
        np.testing.assert_array_equal(adjusted[51:], obs[51:])
        np.testing.assert_allclose(adjusted[50], expected, atol=1e-7)


def test_turn_labels_preserve_the_requested_twist(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from velstand_onset_rollouts import _tracking_label_obs

    for requested in (0.5, -0.5):
        obs = np.linspace(-1, 1, 61, dtype=np.float32)
        obs[48:51] = [0, 0, requested]
        original = obs.copy()
        teacher = _tracking_label_obs(obs, True, np.zeros(7), np.zeros(2), 0)
        np.testing.assert_array_equal(obs, original)
        expected = original.copy()
        expected[50] = requested * 1.4
        np.testing.assert_array_equal(teacher, expected)


def test_straight_label_uses_the_starting_heading_and_line(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from velstand_onset_rollouts import _tracking_label_obs

    # Advance 0.25 m along a rotated line, 0.05 m to its left, and cross +pi.
    heading = np.pi - 0.02
    yaw = -np.pi + 0.02
    forward = np.array([np.cos(heading), np.sin(heading)])
    left = np.array([-np.sin(heading), np.cos(heading)])
    origin = np.array([2.0, -3.0])
    root = np.r_[origin + 0.25 * forward + 0.05 * left, 0.12,
                 np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    obs = np.zeros(61, dtype=np.float32)
    obs[48] = 0.15
    teacher = _tracking_label_obs(obs, False, root, origin, heading)
    np.testing.assert_allclose(teacher[48:51], [0.15, 0, -2 * 0.04 - 0.05])
    np.testing.assert_allclose(obs[48:51], [0.15, 0, 0])


def test_start_cue_labels_only_the_first_second(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import velstand_onset_rollouts as rollouts

    # A standing observation with the daemon's smoothed forward command. Have
    # the teacher return what it sees so we can inspect labels after the ramp.
    root = np.array([0, 0, 0.12, 1, 0, 0, 0])
    policy = SimpleNamespace(vel_cmd=np.zeros(3), _update_command=lambda: None,
                             data=SimpleNamespace(joint=lambda name: SimpleNamespace(qpos=root)))

    def observe():
        obs = np.zeros(61, dtype=np.float32)
        obs[5] = -1
        obs[48:51] = policy.vel_cmd
        return obs

    policy.get_observations = observe
    monkeypatch.setattr(rollouts, "_apply_action", lambda *args: None)
    for follow_teacher in (False, True):
        policy.vel_cmd.fill(0)
        rows = list(rollouts._rollout(policy, lambda obs: -1, lambda obs: obs[48], None,
                                     (0.15, 0), 2, 54, follow_teacher, lesson="start"))
        assert len(rows) == 27
        raw = np.array([obs[48] for obs, _ in rows])
        labels = np.array([label for _, label in rows])
        np.testing.assert_allclose(labels[:25], raw[:25] * (0.20 / 0.15))
        np.testing.assert_array_equal(labels[25:], raw[25:])
        np.testing.assert_allclose(raw[-1], 0.15, atol=2e-6)


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

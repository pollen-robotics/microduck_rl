"""Regression tests for the VelStand demonstration labels and output-layer fit."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper


@pytest.mark.parametrize("after_physics", [False, True])
def test_motor_write_can_cross_a_physics_batch(monkeypatch, after_physics):
    import mujoco

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from velstand_onset_rollouts import _apply_action

    model = mujoco.MjModel.from_xml_string("""
        <mujoco><option timestep=".005"/><worldbody><body>
          <joint name="hinge"/><geom type="sphere" size=".1"/>
        </body></worldbody><actuator><position joint="hinge" kp="2"/></actuator></mujoco>
    """)
    data, reference = mujoco.MjData(model), mujoco.MjData(model)
    policy = SimpleNamespace(model=model, data=data, default_pose=np.zeros(1), action_scale=1.0)
    for action in (0.4, -0.2, 0.1):
        target = 0.3 * reference.ctrl.copy() + 0.7 * action
        if not after_physics:
            reference.ctrl[:] = target
        mujoco.mj_step(model, reference, nstep=4)
        reference.ctrl[:] = target
        _apply_action(policy, np.array([action]), np.array([0.7]), after_physics=after_physics)
        np.testing.assert_allclose(data.qpos, reference.qpos, atol=1e-14)
        np.testing.assert_allclose(data.qvel, reference.qvel, atol=1e-14)
        np.testing.assert_allclose(data.ctrl, target)
        assert data.time == reference.time


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


def test_paired_head_benchmark_matches_every_shifted_start(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from bench_velstand_onset import _paired_placements

    used_seeds = set()
    for fold in ("primary", "reserve"):
        rows, _ = _paired_placements(fold)
        shifted = {seed: (pose, ticks, head) for seed, pose, ticks, head, control in rows if not control}
        neutral = {seed: (pose, ticks, head) for seed, pose, ticks, head, control in rows if control}
        assert len(rows) == 60 and len(shifted) == len(neutral) == 30
        assert shifted.keys() == neutral.keys() and used_seeds.isdisjoint(shifted)
        used_seeds.update(shifted)
        for seed, (pose, ticks, head) in shifted.items():
            assert neutral[seed] == (pose, ticks, (0,) * 4)
            assert len(head) == 4 and any(head)


@pytest.fixture(params=[False, True], ids=["default", "damped"])
def damp_left_turn(request):
    return request.param


@pytest.fixture
def calibration_engine(monkeypatch, damp_left_turn):
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
    calibrate_slow_turns(model, damp_left_turn=damp_left_turn)
    return ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])


def test_turn_calibration_preserves_other_observations_and_commands(calibration_engine):
    right_pivot_vx = -0.085
    right_steady_shift = 0.39
    # At -0.5 rad/s the window weight is 0.5. Still legs receive an additional onset cue.
    right_steady_yaw = -0.5 + 0.5 * right_steady_shift
    right_start_yaw = right_steady_yaw - 0.5 * 0.85
    for vx, vy, yaw, leg_speed, expected_vx, expected_yaw in (
        (0, 0, 0.5, 0, 0, 1.0), (0, 0, -0.5, 0, right_pivot_vx, right_start_yaw),
        (0, 0, 0.5, 3, 0, 0.575), (0, 0, -0.5, 4, right_pivot_vx, right_steady_yaw),
        (0, 0, 0.5, 1.5, 0, 0.7875), (0, 0, 0, 2, 0, 0), (0.15, 0, 0, 0, 0.15, 0),
        (0.15, 0, 0.5, 0, 0.15, 0.5), (0, 0.12, -0.5, 2, 0, -0.5),
        (0.05, 0, -0.5, 3, 0.05 + 0.5 * right_pivot_vx, -0.5 + 0.25 * right_steady_shift),
        (0, 0, -0.30, 0, 0, -0.30), (0, 0, 1.0, 0, 0, 1.0),
    ):
        obs = np.linspace(-0.12, 0.12, 61, dtype=np.float32)
        obs[20:34] = 0
        obs[33] = leg_speed
        obs[25:29] = 5  # Head motion alone must not attenuate the first step.
        obs[48:51] = [vx, vy, yaw]
        obs[52] = obs[51]  # Equal neck and head pitch give zero differential head cue.
        adjusted = calibration_engine.run(None, {"obs": obs[None]})[0][0]
        np.testing.assert_array_equal(adjusted[:48], obs[:48])
        assert adjusted[49] == obs[49]
        np.testing.assert_array_equal(adjusted[51:], obs[51:])
        np.testing.assert_allclose(adjusted[[48, 50]], [expected_vx, expected_yaw], atol=1e-7)


@pytest.mark.parametrize("world_yaw_rate", [-0.57, 0.0, 1.5])
def test_positive_yaw_cue_is_monotone_at_fixed_neck_pose_and_leg_speed(calibration_engine, world_yaw_rate):
    requests = np.linspace(0.30, 1.05, 76, dtype=np.float32)
    for neck, measured in ((0, 0), (-0.1, -0.1), (-0.1, -0.12), (-0.1, -0.2), (-0.5, -1.0)):
        for leg_speed in (0.0, 0.75, 1.5, 3.0, 4.0):
            obs = np.zeros((1, 61), dtype=np.float32)
            obs[0, 2] = world_yaw_rate
            obs[0, 5] = -1
            obs[0, 51] = neck
            obs[0, 11] = measured  # sixth joint position, relative to the default pose
            obs[0, 20] = leg_speed
            cues = []
            for requested in requests:
                obs[0, 50] = requested
                cues.append(calibration_engine.run(None, {"obs": obs})[0][0, 50])
            assert np.all(np.diff(cues) >= -np.finfo(np.float32).eps), (neck, measured, leg_speed)
            assert cues[-1] == requests[-1]


def test_left_turn_cue_depends_on_measured_neck_pitch_during_motion(calibration_engine, damp_left_turn):
    for vx, yaw, leg_speed, neck, measured, default_yaw, damped_yaw in (
        (0, 0.5, 3, -0.1, -0.1, 0.575, 0.425),   # at the requested position
        (0, 0.5, 3, -0.1, -0.08, 0.575, 0.425),  # above the request
        (0, 0.5, 3, -0.1, -0.12, 0.630, 0.635),  # limited by the measured difference
        (0, 0.5, 3, -0.1, -0.2, 0.685, 0.845),   # limited by the requested neck angle
        (0, 0.65, 3, -0.1, -0.1, 0.7275, 0.7275),
        (0, 0.65, 3, -0.1, -0.12, 0.822, 0.822),
        (0, 0.65, 3, -0.5, -1.0, 0.9825, 0.9825),  # combined upper limit
        (0, 0.65, 3, 0.1, -0.2, 0.7275, 0.7275),   # positive neck request
        (0, 0.65, 0, -0.1, -0.2, 1.0, 1.0),        # still legs retain the onset cue
        (0, 0, 3, -0.1, -0.2, 0, 0),
        (0.15, 0, 3, -0.1, -0.2, 0, 0),
        (0.15, 0.65, 3, -0.1, -0.2, 0.65, 0.65),
    ):
        expected_yaw = damped_yaw if damp_left_turn else default_yaw
        obs = np.zeros((1, 61), dtype=np.float32)
        obs[0, 11] = measured
        obs[0, 20] = leg_speed
        obs[0, 48:51] = [vx, 0, yaw]
        obs[0, 51:55] = [neck, 0.07, -0.09, 0.03]
        adjusted = calibration_engine.run(None, {"obs": obs})[0]
        np.testing.assert_allclose(adjusted[0, 50], expected_yaw, atol=1e-7)
        np.testing.assert_array_equal(adjusted[:, :50], obs[:, :50])
        np.testing.assert_array_equal(adjusted[:, 51:], obs[:, 51:])


def test_optional_left_turn_damper_uses_world_vertical(calibration_engine, damp_left_turn):
    obs = np.zeros((1, 61), dtype=np.float32)
    obs[0, 4] = -1  # Tilting the trunk puts world yaw on the second gyro axis.
    obs[0, 11] = -0.1
    obs[0, 20] = 3
    obs[0, 50:52] = (0.5, -0.1)
    for gyro, damped_yaw in (
        ((0, 0, 1.5), 0.425),   # spin about a horizontal axis
        ((0, 0.62, 0), 0.395),  # 0.03 rad/s above the stride allowance
        ((0, 1.5, 0), 0.355),   # lower limit on the moving cue
    ):
        obs[0, :3] = gyro
        result = calibration_engine.run(None, {"obs": obs})[0][0, 50]
        np.testing.assert_allclose(result, damped_yaw if damp_left_turn else 0.575, atol=1e-7)


def test_right_turn_retains_the_bounded_commanded_pitch_cue(calibration_engine):
    for leg_speed in (0.0, 3.0):
        obs = np.zeros((1, 61), dtype=np.float32)
        obs[0, 50] = -0.5
        obs[0, 20] = leg_speed
        neutral = calibration_engine.run(None, {"obs": obs})[0][0, [48, 50]]
        # At the center of the window the yaw adjustment is half the clipped
        # head cue. Vary each pitch independently and then saturate both signs.
        for neck, head, right_yaw_delta in ((0.04, 0, 0.0128), (0, 0.04, -0.0128),
                                            (0.12, -0.12, 0.06), (-0.12, 0.12, -0.06)):
            obs[0, 51:53] = [neck, head]
            obs[0, 11] = neck - 0.2
            adjusted = calibration_engine.run(None, {"obs": obs})[0][0, [48, 50]]
            np.testing.assert_allclose(adjusted - neutral, [0, right_yaw_delta], atol=1e-7)


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

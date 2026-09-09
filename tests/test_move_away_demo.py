"""CPU tests for the move-away behavior demo.

These cover the pure decision/control layer and the model invariants the demo
relies on. None of them needs a policy file, a GPU, or a rendering context: the
state machine and gaze maths are plain Python/numpy, and the model checks only
compile the MJCF.
"""

import math
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

# The demo lives under scripts/, which is not an installed package.
_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "behavior_demos"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from move_away.controller import (
    RETREAT_D,
    VX_RETREAT,
    WZ_MAX,
    MoveAwayController,
    wrap_angle,
)
from move_away.gaze import (
    HEAD_PITCH_MARGIN,
    HEAD_YAW_MARGIN,
    GazeState,
    bearing_elevation,
    camera_axes,
    in_frustum,
)
from move_away.perception import body_subtree, optical_frame_quat
from move_away.runtime import (
    DEFAULT_POSE,
    GYRO_SENSOR,
    HEAD_PITCH_ACT,
    HEAD_YAW_ACT,
    SCENE_XML,
    build_observation,
    sensor_address,
)

# --- state machine -----------------------------------------------------------


def _drive(controller, steps, yaw=0.0, distance=5.0, visible=True):
    """Run the controller for `steps` ticks at a fixed observation."""
    out = None
    for _ in range(steps):
        out = controller.update(yaw, distance, visible)
    return out


def test_idle_until_person_is_close_and_visible():
    c = MoveAwayController()
    _drive(c, 10, distance=RETREAT_D + 0.2)
    assert c.state == "IDLE"
    # Close enough but NOT visible must not trigger: detection is
    # line-of-sight AND range, never range alone.
    _drive(c, 10, distance=RETREAT_D - 0.2, visible=False)
    assert c.state == "IDLE"
    _drive(c, 1, distance=RETREAT_D - 0.2, visible=True)
    assert c.state == "RETREAT"


def test_full_state_sequence_is_reached_in_order():
    c = MoveAwayController()
    seen = [c.state]
    yaw = 0.0
    for _ in range(int(30.0 * c.ctrl_hz)):
        # Follow the heading setpoint closely so TURN can complete.
        yaw += 0.25 * wrap_angle(c.heading_setpoint - yaw)
        c.update(yaw, 0.5, True)
        if c.state != seen[-1]:
            seen.append(c.state)
    assert seen == ["IDLE", "RETREAT", "TURN", "CLEAR", "DONE"]


def test_idle_and_done_command_zero_velocity():
    c = MoveAwayController()
    assert _drive(c, 5, distance=9.0) == (0.0, 0.0, 0.0)
    c.state = "DONE"
    out = _drive(c, 200, distance=0.5)
    assert all(abs(v) < 1e-3 for v in out)


def test_retreat_commands_backward_velocity_inside_the_measured_band():
    c = MoveAwayController()
    _drive(c, 1, distance=0.5)
    assert c.state == "RETREAT"
    vx, vy, _ = _drive(c, 100, distance=0.5)
    # MEASURED: the backward gait does not engage above about -0.30.
    assert vx < -0.30
    assert vx == pytest.approx(VX_RETREAT, abs=0.02)
    assert vy == 0.0


def test_heading_hold_opposes_drift_with_the_correct_sign():
    # A positive wz command produces a positive (left) yaw rate when the policy
    # is fed the real gyro, so drifting right (yaw < 0) must command wz > 0.
    c = MoveAwayController()
    _drive(c, 1, distance=0.5)
    assert c.state == "RETREAT"
    c.yaw_ref = 0.0
    _, _, wz_right = _drive(c, 20, yaw=math.radians(-20.0), distance=0.5)
    assert wz_right > 0.0
    c2 = MoveAwayController()
    _drive(c2, 1, distance=0.5)
    c2.yaw_ref = 0.0
    _, _, wz_left = _drive(c2, 20, yaw=math.radians(20.0), distance=0.5)
    assert wz_left < 0.0


def test_yaw_command_is_clamped():
    c = MoveAwayController()
    _drive(c, 1, distance=0.5)
    c.yaw_ref = 0.0
    _, _, wz = _drive(c, 400, yaw=math.radians(-170.0), distance=0.5)
    assert abs(wz) <= WZ_MAX + 1e-6


def test_turn_sign_selects_the_heading_setpoint_direction():
    left = MoveAwayController(turn_sign=1.0)
    right = MoveAwayController(turn_sign=-1.0)
    for c in (left, right):
        c.yaw_ref = 0.0
        c.state = "TURN"
    assert left.heading_setpoint > 0.0
    assert right.heading_setpoint < 0.0
    assert left.heading_setpoint == pytest.approx(-right.heading_setpoint)


def test_turn_falls_through_on_the_safety_timeout():
    # If the heading never converges the demo must not hang in TURN forever.
    c = MoveAwayController()
    _drive(c, 1, distance=0.5)
    _drive(c, int(c.retreat_hold * c.ctrl_hz) + 2, distance=0.5)
    assert c.state == "TURN"
    _drive(c, int(c.turn_max * c.ctrl_hz) + 2, yaw=0.0, distance=0.5)
    assert c.state == "CLEAR"


def test_command_is_low_pass_filtered_not_stepped():
    c = MoveAwayController()
    first = c.update(0.0, 0.5, True)
    assert c.state == "RETREAT"
    # One tick must not jump to the full commanded speed.
    assert abs(first[0]) < abs(VX_RETREAT)


def test_wrap_angle_folds_into_pi_range():
    assert wrap_angle(math.radians(370.0)) == pytest.approx(math.radians(10.0))
    assert wrap_angle(math.radians(-190.0)) == pytest.approx(math.radians(170.0))
    for a in (0.0, 1.0, -1.0, 3.0, -3.0, 7.0):
        assert -math.pi <= wrap_angle(a) < math.pi


# --- gaze maths --------------------------------------------------------------


def _camera_matrix(forward, up):
    """Build a MuJoCo camera xmat whose optical -Z is `forward`, +Y is `up`."""
    forward = np.asarray(forward, dtype=np.float64)
    forward = forward / np.linalg.norm(forward)
    up = np.asarray(up, dtype=np.float64)
    up = up - np.dot(up, forward) * forward
    up = up / np.linalg.norm(up)
    right = np.cross(forward, up)
    return np.column_stack([right, up, -forward]).reshape(9)


def test_camera_axes_follow_mujoco_optical_convention():
    fwd, right, up = camera_axes(_camera_matrix([1, 0, 0], [0, 0, 1]))
    assert np.allclose(fwd, [1, 0, 0], atol=1e-9)
    assert np.allclose(up, [0, 0, 1], atol=1e-9)
    # Looking along +x with +z up, image-right is -y (checked against the real
    # model in test_head_camera_optical_correction_... below).
    assert np.allclose(right, [0, -1, 0], atol=1e-9)
    assert np.allclose(np.cross(fwd, up), right, atol=1e-9)


def test_bearing_is_positive_to_the_cameras_left():
    fwd, right, up = camera_axes(_camera_matrix([1, 0, 0], [0, 0, 1]))
    # camera looks along +x with +z up, so its left is +y
    bearing, elevation = bearing_elevation(np.array([1.0, 1.0, 0.0]), fwd, right, up)
    assert bearing == pytest.approx(math.radians(45.0))
    assert elevation == pytest.approx(0.0, abs=1e-9)


def test_elevation_is_positive_above_the_optical_axis():
    fwd, right, up = camera_axes(_camera_matrix([1, 0, 0], [0, 0, 1]))
    _, elevation = bearing_elevation(np.array([1.0, 0.0, 1.0]), fwd, right, up)
    assert elevation == pytest.approx(math.radians(45.0))


def test_frustum_rejects_targets_behind_and_outside():
    fwd, right, up = camera_axes(_camera_matrix([1, 0, 0], [0, 0, 1]))
    tan_h = tan_v = math.tan(math.radians(22.5))
    assert in_frustum(np.array([2.0, 0.0, 0.0]), fwd, right, up, tan_h, tan_v)
    # directly behind the camera
    assert not in_frustum(np.array([-2.0, 0.0, 0.0]), fwd, right, up, tan_h, tan_v)
    # 60 deg off axis horizontally
    far_side = np.array([1.0, math.tan(math.radians(60.0)), 0.0])
    assert not in_frustum(far_side, fwd, right, up, tan_h, tan_v)


def test_gaze_servo_reduces_the_pointing_error():
    gaze = GazeState(
        yaw=0.0, pitch=0.0, yaw_limits=(-2.9, 2.9), pitch_limits=(-1.5, 1.5), tau=0.02, dt=0.02
    )
    yaw, _ = gaze.step(math.radians(20.0), 0.0)
    assert yaw == pytest.approx(math.radians(20.0), abs=1e-6)


def test_gaze_pitch_sign_matches_the_actuator_convention():
    # A positive head_pitch command points the camera DOWN, so a target ABOVE
    # the axis (positive elevation) must DECREASE the pitch setpoint.
    gaze = GazeState(
        yaw=0.0, pitch=0.0, yaw_limits=(-2.9, 2.9), pitch_limits=(-1.5, 1.5), tau=0.02, dt=0.02
    )
    _, pitch = gaze.step(0.0, math.radians(15.0))
    assert pitch < 0.0


def test_gaze_setpoints_stay_inside_the_joint_limits_with_margin():
    gaze = GazeState(
        yaw=0.0, pitch=0.0, yaw_limits=(-0.5, 0.5), pitch_limits=(-0.4, 0.4), tau=0.02, dt=0.02
    )
    for _ in range(200):
        gaze.step(math.radians(90.0), math.radians(-90.0))
    assert gaze.yaw <= 0.5 - HEAD_YAW_MARGIN + 1e-9
    assert gaze.pitch <= 0.4 - HEAD_PITCH_MARGIN + 1e-9
    for _ in range(400):
        gaze.step(math.radians(-90.0), math.radians(90.0))
    assert gaze.yaw >= -0.5 + HEAD_YAW_MARGIN - 1e-9
    assert gaze.pitch >= -0.4 + HEAD_PITCH_MARGIN - 1e-9


# --- observation contract ----------------------------------------------------


def test_observation_layout_and_zero_padded_command_block():
    nu = 14
    gyro = np.arange(3, dtype=np.float32)
    gravity = np.arange(3, dtype=np.float32) + 10.0
    joint_pos = np.arange(nu, dtype=np.float32) + 100.0
    joint_vel = np.arange(nu, dtype=np.float32) + 200.0
    last_action = np.arange(nu, dtype=np.float32) + 300.0
    twist = np.array([-0.36, 0.0, 0.2], dtype=np.float32)

    obs = build_observation(gyro, gravity, joint_pos, joint_vel, last_action, twist, 13)

    # 3 + 3 + 14 + 14 + 14 + 13 = 61, the shared policy-family layout.
    assert obs.shape == (61,)
    assert obs.dtype == np.float32
    assert np.allclose(obs[0:3], gyro)
    assert np.allclose(obs[3:6], gravity)
    assert np.allclose(obs[6:20], joint_pos)
    assert np.allclose(obs[20:34], joint_vel)
    assert np.allclose(obs[34:48], last_action)
    assert np.allclose(obs[48:51], twist)
    # head_pose(4) + body_pose(6) are ZERO-PADDED, never dropped.
    assert np.allclose(obs[51:61], 0.0)


# --- model invariants (compile the MJCF, no policy needed) -------------------


@pytest.fixture(scope="module")
def scene_model():
    return mujoco.MjModel.from_xml_path(str(SCENE_XML))


def test_demo_scene_path_resolves_inside_the_repo():
    assert SCENE_XML.is_file(), SCENE_XML
    assert SCENE_XML.name == "scene_move_away.xml"


def test_scene_keeps_the_14_actuator_layout_and_adds_no_dof(scene_model):
    # The person is a MOCAP body: it must not add degrees of freedom, otherwise
    # the keyframes' qpos sizes break and the obs layout shifts.
    assert scene_model.nu == 14
    assert scene_model.nq == 21  # freejoint(7) + 14 servos
    names = [
        mujoco.mj_id2name(scene_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        for i in range(scene_model.nu)
    ]
    assert names[HEAD_PITCH_ACT] == "head_pitch"
    assert names[HEAD_YAW_ACT] == "head_yaw"
    assert len(DEFAULT_POSE) == scene_model.nu


def test_person_is_a_non_colliding_mocap_prop(scene_model):
    person = mujoco.mj_name2id(scene_model, mujoco.mjtObj.mjOBJ_BODY, "person")
    assert person >= 0
    assert int(scene_model.body_mocapid[person]) >= 0
    geoms = [g for g in range(scene_model.ngeom) if scene_model.geom_bodyid[g] == person]
    assert geoms, "the person prop has no geoms"
    for g in geoms:
        # contype/conaffinity 0: the prop can never push the robot over.
        assert scene_model.geom_contype[g] == 0
        assert scene_model.geom_conaffinity[g] == 0


def test_gyro_sensor_name_exists_and_is_not_the_fallback(scene_model):
    # REGRESSION: mj_name2id returns -1 for an unknown sensor, and
    # model.sensor_adr[-1] is a VALID index (the last sensor). A typo
    # therefore does NOT raise, it silently feeds a different quantity into
    # the policy's angular-velocity slot. Assert the name resolves, and that
    # it is not accidentally the last sensor in the model.
    assert mujoco.mj_name2id(scene_model, mujoco.mjtObj.mjOBJ_SENSOR, GYRO_SENSOR) >= 0
    adr = sensor_address(scene_model, GYRO_SENSOR)
    assert adr != scene_model.sensor_adr[-1]
    with pytest.raises(ValueError):
        sensor_address(scene_model, "definitely_not_a_sensor")


def test_head_camera_optical_correction_points_along_the_robot_forward_axis(scene_model):
    # The MJCF's exported head_camera quaternion points optical -Z BACKWARDS,
    # into the robot's own geometry. The demo corrects it in memory; verify the
    # correction actually aligns the optical axis with the trunk's +X forward.
    data = mujoco.MjData(scene_model)
    qpos_idx = [
        int(scene_model.jnt_qposadr[scene_model.actuator_trnid[i, 0]])
        for i in range(scene_model.nu)
    ]
    for slot, address in enumerate(qpos_idx):
        data.qpos[address] = DEFAULT_POSE[slot]
    cam = mujoco.mj_name2id(scene_model, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
    trunk = mujoco.mj_name2id(scene_model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")

    mujoco.mj_forward(scene_model, data)
    trunk_forward = data.xmat[trunk].reshape(3, 3)[:, 0]
    before, _, _ = camera_axes(data.cam_xmat[cam])
    assert float(np.dot(before, trunk_forward)) < -0.9  # points backwards

    original = scene_model.cam_quat[cam].copy()
    try:
        scene_model.cam_quat[cam] = optical_frame_quat()
        mujoco.mj_forward(scene_model, data)
        after, _, up = camera_axes(data.cam_xmat[cam])
        assert float(np.dot(after, trunk_forward)) > 0.99
        assert float(up[2]) > 0.99  # image-up follows world up
        # Same handedness the pure-maths test above asserts.
        _, real_right, _ = camera_axes(data.cam_xmat[cam])
        assert np.allclose(np.cross(after, up), real_right, atol=1e-6)
    finally:
        scene_model.cam_quat[cam] = original


def test_body_subtree_collects_the_whole_robot_and_the_person(scene_model):
    trunk = mujoco.mj_name2id(scene_model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    person = mujoco.mj_name2id(scene_model, mujoco.mjtObj.mjOBJ_BODY, "person")
    robot = body_subtree(scene_model, trunk)
    people = body_subtree(scene_model, person)
    assert trunk in robot and person in people
    # The two sets must be disjoint, otherwise the ray cast would skip the
    # person as "our own body" and never see them.
    assert not (robot & people)
    assert 0 not in robot  # never treat the world body as part of the robot


def test_default_pose_matches_the_scene_stand_keyframe(scene_model):
    key = mujoco.mj_name2id(scene_model, mujoco.mjtObj.mjOBJ_KEY, "STAND")
    assert key >= 0, "scene_move_away.xml must keep the STAND keyframe"
    stand_ctrl = scene_model.key_ctrl[key]
    assert np.allclose(stand_ctrl, DEFAULT_POSE, atol=1e-4)

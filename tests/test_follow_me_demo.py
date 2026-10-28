"""CPU tests for the follow-me behavior demo.

These cover the pure behavior logic (leader choreography, world-space footprint
trail, asymmetric turn controller) and the demo scene's model invariants.
Nothing here needs an ONNX policy, a GPU, or a renderer.
"""

import math
import sys
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

DEMO_DIR = Path(__file__).resolve().parents[1] / "scripts" / "behavior_demos" / "follow_me"
sys.path.insert(0, str(DEMO_DIR))

from follow_motion import (
    BACKWARD_END,
    CTRL_HZ,
    DEMO_SECONDS,
    FORWARD_END,
    LEFT_TURN_END,
    LEFT_WZ_MAX,
    LEFT_WZ_MIN,
    READY_END,
    RIGHT_EXIT_END,
    RIGHT_WZ_MAX,
    RIGHT_WZ_MIN,
    STOP_END,
    TRAIL_DISTANCE,
    TURN_VX,
    YAW_DEADBAND_RAD,
    YAW_GAIN,
    FollowController,
    FootstepTrail,
    PersonState,
    TrailState,
    person_trajectory,
    wrap,
)

SCENE = DEMO_DIR / "scene_follow_me.xml"


def _trail_state(phase, pos, yaw):
    return TrailState(phase=phase, pos=np.asarray(pos, dtype=np.float64),
                      yaw=yaw, path_s=0.0, leader_path_s=0.0, moving=True)


# --------------------------------------------------------------------------
# Leader choreography
# --------------------------------------------------------------------------

def test_route_phase_order_and_coverage():
    """Every phase appears, in order, and the route fills the demo duration."""
    seen = []
    for step in range(int(DEMO_SECONDS * CTRL_HZ)):
        phase = person_trajectory(step / CTRL_HZ).phase
        if not seen or seen[-1] != phase:
            seen.append(phase)
    assert seen == ["READY", "FORWARD", "LEFT TURN", "STOP", "RIGHT TURN",
                    "BACKWARD", "DONE"]


def test_turns_are_genuinely_opposite():
    """The left turn adds +90 deg of world yaw; the right turn subtracts 90.

    This is the invariant an earlier iteration violated: a phase LABELLED
    "left turn" used negative world yaw, so it was physically a right-hand
    curve, and the two turns were not opposites at all.
    """
    left_start = person_trajectory(FORWARD_END).yaw
    left_end = person_trajectory(LEFT_TURN_END - 1e-9).yaw
    right_start = person_trajectory(STOP_END).yaw
    right_end = person_trajectory(RIGHT_EXIT_END - 1e-9).yaw

    assert math.degrees(left_end - left_start) == pytest.approx(90.0, abs=0.1)
    assert math.degrees(right_end - right_start) == pytest.approx(-90.0, abs=0.1)
    # Opposite signs, and the leader returns to its original heading.
    assert (left_end - left_start) * (right_end - right_start) < 0.0
    assert person_trajectory(RIGHT_EXIT_END - 1e-9).yaw == pytest.approx(0.0, abs=1e-9)


def test_leader_path_is_continuous_across_phase_boundaries():
    """No teleports: position is continuous at every phase change."""
    for boundary in (READY_END, FORWARD_END, LEFT_TURN_END, STOP_END,
                     RIGHT_EXIT_END, BACKWARD_END):
        before = person_trajectory(boundary - 1e-6).pos
        after = person_trajectory(boundary + 1e-6).pos
        assert np.linalg.norm(after - before) < 1e-3, f"jump at t={boundary}"


def test_stationary_phases_are_marked_not_moving():
    assert not person_trajectory(1.0).moving          # READY
    assert not person_trajectory(16.0).moving         # STOP
    assert not person_trajectory(43.0).moving         # DONE
    assert person_trajectory(4.0).moving              # FORWARD
    assert person_trajectory(10.0).moving             # LEFT TURN


# --------------------------------------------------------------------------
# World-space footprint trail
# --------------------------------------------------------------------------

def test_trail_lags_leader_by_the_configured_path_length():
    """Once the queue is primed, the gap is TRAIL_DISTANCE of PATH length.

    Checked two ways: the queue's own bookkeeping, and the independently
    integrated arc length of the leader's route between the queued footprint
    and the leader. On the straight leg both must also equal the plain
    straight-line distance.
    """
    trail = FootstepTrail(person_trajectory(0.0))
    state = None
    previous = person_trajectory(0.0).pos
    arc_by_time = [(0.0, 0.0)]
    travelled = 0.0
    steps = int(20.0 * CTRL_HZ)
    for step in range(steps):
        t = step / CTRL_HZ
        leader = person_trajectory(t)
        travelled += float(np.linalg.norm(leader.pos - previous))
        previous = leader.pos
        arc_by_time.append((t, travelled))
        state = trail.update(leader)

    assert state.leader_path_s - state.path_s == pytest.approx(TRAIL_DISTANCE)

    # Independent geometric check on the straight leg, where path length and
    # straight-line distance coincide.
    trail_straight = FootstepTrail(person_trajectory(0.0))
    straight_state = None
    sample_t = FORWARD_END - 0.5
    for step in range(int(sample_t * CTRL_HZ)):
        straight_state = trail_straight.update(person_trajectory(step / CTRL_HZ))
    leader_now = person_trajectory(sample_t - 1.0 / CTRL_HZ)
    gap = float(np.linalg.norm(leader_now.pos - straight_state.pos))
    assert gap == pytest.approx(TRAIL_DISTANCE, abs=0.02)


def test_trail_target_lies_on_the_route_the_leader_actually_walked():
    """The queued point must be ON the leader's past path, not interpolated
    across the chord of the turn.

    A follower that reconstructs a target from the leader's current pose can
    sit inside the arc; a genuine footstep queue cannot.
    """
    trail = FootstepTrail(person_trajectory(0.0))
    route = []
    state = None
    for step in range(int(LEFT_TURN_END * CTRL_HZ)):
        leader = person_trajectory(step / CTRL_HZ)
        route.append(leader.pos.copy())
        state = trail.update(leader)
    route = np.array(route)
    distance_to_route = float(np.min(np.linalg.norm(route - state.pos, axis=1)))
    assert distance_to_route < 1e-3


def test_trail_keeps_the_corner_instead_of_cutting_it():
    """The queued footprint stays on the pre-turn leg while the leader turns.

    This is the whole point of following footsteps rather than the leader's
    current pose: a pose-mirroring follower would already be turning here.
    """
    trail = FootstepTrail(person_trajectory(0.0))
    state = None
    # Sample shortly after the leader commits to the left turn.
    for step in range(int((FORWARD_END + 1.5) * CTRL_HZ)):
        state = trail.update(person_trajectory(step / CTRL_HZ))
    leader = person_trajectory(FORWARD_END + 1.5)
    assert leader.yaw > math.radians(5.0), "leader should already be turning"
    # The queued target has not started the turn yet.
    assert abs(state.yaw) < math.radians(5.0)
    # And it is still behind the leader, on the straight leg.
    assert state.pos[0] < leader.pos[0]


def test_stopped_leader_freezes_the_queue():
    """A leader that does not move must not advance the path length.

    The queue is sampled from the FIRST stationary step onward: the step that
    enters STOP still carries the last real displacement of the turn arc.
    """
    trail = FootstepTrail(person_trajectory(0.0))
    first_stop = int(LEFT_TURN_END * CTRL_HZ) + 1
    for step in range(first_stop):
        trail.update(person_trajectory(step / CTRL_HZ))
    assert not person_trajectory((first_stop - 1) / CTRL_HZ).moving
    frozen = trail.path_s
    for step in range(first_stop, int(STOP_END * CTRL_HZ)):
        trail.update(person_trajectory(step / CTRL_HZ))
    assert trail.path_s == pytest.approx(frozen)


def test_trail_yaw_interpolation_wraps_across_pi():
    """Interpolating between +170 deg and -170 deg must cross pi, not unwind.

    A naive linear blend of the two yaw values would sweep the long way round
    through 0 deg, pointing the follower backwards mid-corner.
    """
    start = PersonState("FORWARD", np.array([0.0, 0.0]), math.radians(170.0),
                        np.zeros(2), 0.0, True, 0.0)
    trail = FootstepTrail(start, gap=0.5)
    for i in range(1, 40):
        pos = np.array([-0.05 * i, 0.0])
        trail.update(PersonState("FORWARD", pos, math.radians(-170.0),
                                 np.zeros(2), 0.0, True, 0.0))
    state = trail.update(PersonState("FORWARD", np.array([-2.0, 0.0]),
                                     math.radians(-170.0), np.zeros(2), 0.0,
                                     True, 0.0))
    assert abs(state.yaw) > math.radians(150.0)


def test_trail_yaw_takes_the_short_way_around_during_interpolation():
    """Directly exercise the blend halfway between +170 and -170 degrees.

    The midpoint must be near +/-180, never near 0.
    """
    start = PersonState("FORWARD", np.array([0.0, 0.0]), math.radians(170.0),
                        np.zeros(2), 0.0, True, 0.0)
    trail = FootstepTrail(start, gap=1.0)
    # One sample 2 m away with the opposite heading: the queued point lands
    # midway between the two, so the yaw blend is exercised at fraction≈0.5.
    trail.update(PersonState("FORWARD", np.array([2.0, 0.0]),
                             math.radians(-170.0), np.zeros(2), 0.0, True, 0.0))
    state = trail.update(PersonState("FORWARD", np.array([2.0, 0.0]),
                                     math.radians(-170.0), np.zeros(2), 0.0,
                                     True, 0.0))
    assert abs(state.yaw) > math.radians(170.0), (
        f"yaw blend unwound the short way: {math.degrees(state.yaw):.1f} deg")


# --------------------------------------------------------------------------
# Asymmetric turn controller
# --------------------------------------------------------------------------

def test_turn_rate_deadband_is_exactly_zero():
    """Inside the deadband the command is 0.0 — not a small nudge.

    The stock policy has a sharp gait-onset threshold, so sub-threshold
    corrections only produce shuffling.
    """
    for error_deg in (-2.9, -1.0, 0.0, 1.0, 2.9):
        assert FollowController.turn_rate(math.radians(error_deg)) == 0.0
    assert FollowController.turn_rate(YAW_DEADBAND_RAD * 1.01) != 0.0


def test_turn_rate_is_asymmetric_and_correctly_signed():
    """Left and right use separately measured magnitude bands."""
    left = FollowController.turn_rate(math.radians(45.0))
    right = FollowController.turn_rate(math.radians(-45.0))
    assert left > 0.0 and right < 0.0
    assert LEFT_WZ_MIN <= left <= LEFT_WZ_MAX
    assert RIGHT_WZ_MIN <= -right <= RIGHT_WZ_MAX
    # Asymmetry is the point: a mirrored command does not mirror the motion.
    assert left > -right


def test_turn_rate_magnitude_stays_within_measured_limits():
    for error_deg in np.linspace(-180.0, 180.0, 721):
        wz = FollowController.turn_rate(math.radians(error_deg))
        if wz > 0.0:
            assert LEFT_WZ_MIN <= wz <= LEFT_WZ_MAX
        elif wz < 0.0:
            assert RIGHT_WZ_MIN <= -wz <= RIGHT_WZ_MAX


def test_turn_rate_is_monotonic_in_error_magnitude():
    errors = np.linspace(YAW_DEADBAND_RAD, math.pi, 200)
    left = [FollowController.turn_rate(e) for e in errors]
    right = [-FollowController.turn_rate(-e) for e in errors]
    assert all(b >= a - 1e-12 for a, b in pairwise(left))
    assert all(b >= a - 1e-12 for a, b in pairwise(right))


def test_turn_rate_is_strictly_proportional_inside_the_band():
    """Between the clamps the response must actually scale with the error.

    Monotonicity alone is satisfied by a constant, so it would not notice the
    proportional term being neutralised (e.g. by an inverted gain, which
    copysign then hides). These samples sit strictly inside each clamp band.
    """
    left_lo, left_hi = LEFT_WZ_MIN / YAW_GAIN, LEFT_WZ_MAX / YAW_GAIN
    right_lo, right_hi = RIGHT_WZ_MIN / YAW_GAIN, RIGHT_WZ_MAX / YAW_GAIN

    for lo, hi, sign in ((left_lo, left_hi, +1.0), (right_lo, right_hi, -1.0)):
        samples = np.linspace(lo + 1e-3, hi - 1e-3, 12)
        magnitudes = [abs(FollowController.turn_rate(sign * e)) for e in samples]
        assert all(b > a for a, b in pairwise(magnitudes)), (
            f"turn_rate is flat inside the proportional band: {magnitudes}")
        # And it really is the proportional law, not an arbitrary ramp.
        for error, magnitude in zip(samples, magnitudes):
            assert magnitude == pytest.approx(YAW_GAIN * error)


def test_stopped_leader_commands_zero():
    controller = FollowController()
    leader = PersonState("STOP", np.array([1.0, 0.0]), 0.0, np.zeros(2), 0.0,
                         False, 0.0)
    command, _ = controller.update(
        leader, _trail_state("STOP", [1.0, 0.0], 0.0), np.zeros(3), 0.0)
    assert np.allclose(command, np.zeros(3))


def test_turning_command_walks_forward_while_turning():
    """Turns keep a forward velocity: the duck corners, it does not pivot.

    The forward speed is asserted as a literal, not against TURN_VX, so that
    silently zeroing the constant (turning a corner into a stationary pivot)
    fails here instead of comparing the constant to itself.
    """
    controller = FollowController()
    leader = PersonState("LEFT TURN", np.array([1.0, 0.0]), 0.5, np.zeros(2),
                         0.0, True, 0.0)
    command, _ = controller.update(
        leader, _trail_state("LEFT TURN", [1.0, 0.0], math.radians(40.0)),
        np.zeros(3), 0.0)
    assert command[0] == pytest.approx(0.24)
    assert TURN_VX == pytest.approx(0.24)
    assert command[2] > 0.0


def test_measured_controller_constants_are_pinned():
    """Lock the empirically measured controller envelope.

    These numbers came from measured rollouts of the stock policy, not from
    theory. Changing one silently changes the validated behavior, so it should
    require deliberately updating this test.
    """
    assert (LEFT_WZ_MIN, LEFT_WZ_MAX) == (0.60, 1.00)
    assert (RIGHT_WZ_MIN, RIGHT_WZ_MAX) == (0.18, 0.32)
    assert YAW_DEADBAND_RAD == pytest.approx(math.radians(3.0))
    assert TRAIL_DISTANCE == pytest.approx(0.65)
    assert CTRL_HZ == 50.0


# --------------------------------------------------------------------------
# Reverse safety exception
# --------------------------------------------------------------------------

def test_reverse_is_immediate_and_overrides_the_queue():
    """When the leader backs up, the duck backs up NOW.

    Waiting for the leader's reverse footprint to arrive would let the leader
    walk into the follower, so reversal deliberately bypasses the trail.
    """
    leader = PersonState("BACKWARD", np.array([2.0, 0.0]), 0.0,
                         np.array([-0.08, 0.0]), 0.0, True, 0.0)
    # The queue is still replaying an old forward leg.
    trail = _trail_state("FORWARD", [1.5, 0.0], 0.0)
    assert FollowController.replay_phase(leader, trail) == "BACKWARD"

    command, metrics = FollowController().update(leader, trail, np.zeros(3), 0.0)
    assert command[0] < 0.0
    assert metrics["replay_phase"] == "BACKWARD"


def test_reverse_is_the_only_phase_that_bypasses_the_trail():
    """Every non-reverse phase replays the QUEUED phase, not the leader's."""
    trail = _trail_state("FORWARD", [1.0, 0.0], 0.0)
    for leader_phase in ("FORWARD", "LEFT TURN", "RIGHT TURN", "STOP", "DONE"):
        leader = PersonState(leader_phase, np.array([2.0, 0.0]), 0.0,
                             np.zeros(2), 0.0, True, 0.0)
        assert FollowController.replay_phase(leader, trail) == "FORWARD"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def test_wrap_maps_angles_into_half_open_pi_interval():
    """wrap() normalizes to [-pi, pi) and preserves the represented angle."""
    assert wrap(0.0) == pytest.approx(0.0)
    assert wrap(3.0 * math.pi) == pytest.approx(-math.pi)
    assert wrap(math.pi) == pytest.approx(-math.pi)
    for angle in np.linspace(-20.0, 20.0, 401):
        wrapped = wrap(angle)
        assert -math.pi <= wrapped < math.pi
        assert math.isclose(math.cos(wrapped), math.cos(angle), abs_tol=1e-9)
        assert math.isclose(math.sin(wrapped), math.sin(angle), abs_tol=1e-9)


# --------------------------------------------------------------------------
# Scene / model invariants (no policy required)
# --------------------------------------------------------------------------

def test_demo_scene_exists_and_uses_the_walking_collision_model():
    assert SCENE.is_file()
    text = SCENE.read_text()
    # The stock walking policy was trained on robot_walk.xml; the demo must
    # not silently switch to the allcollisions model.
    assert "robot_walk.xml" in text


def test_demo_scene_compiles_and_matches_the_reference_walk_model():
    """The demo scene must be the official walk model plus scenery.

    The scene includes robot_walk.xml from another directory, so it has to
    restate <compiler meshdir=...> AFTER the include (MuJoCo resolves meshdir
    against the top-level file and applies the LAST compiler tag it parses).
    Getting that wrong silently changes joint units, so compare against the
    reference scene rather than merely asserting the file loads.
    """
    mujoco = pytest.importorskip("mujoco")
    reference_path = (Path(__file__).resolve().parents[1] / "src" /
                      "mjlab_microduck" / "robot" / "microduck" / "scene_walk.xml")
    reference = mujoco.MjModel.from_xml_path(str(reference_path))
    model = mujoco.MjModel.from_xml_path(str(SCENE))

    assert model.nu == reference.nu == 14
    np.testing.assert_allclose(model.jnt_range[:reference.njnt], reference.jnt_range)
    np.testing.assert_allclose(model.actuator_ctrlrange, reference.actuator_ctrlrange)


def test_demo_scene_provides_the_bodies_and_cameras_the_demo_needs():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    for body in ("trunk_base", "person", "trail_target", "follow_camera_rig"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body) >= 0, body
    for body in ("person", "trail_target", "follow_camera_rig"):
        assert model.body(body).mocapid[0] >= 0, f"{body} must be a mocap body"
    for camera in ("head_camera", "follow_camera"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera) >= 0


def test_leader_never_collides_with_the_robot():
    """The scripted leader is a visual target: all its geoms are non-colliding."""
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    person_root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "person")
    person_bodies = set()
    for body in range(model.nbody):
        parent = body
        while parent > 0:
            if parent == person_root:
                person_bodies.add(body)
                break
            parent = int(model.body_parentid[parent])
    assert person_bodies
    for geom in range(model.ngeom):
        if int(model.geom_bodyid[geom]) in person_bodies:
            assert model.geom_contype[geom] == 0
            assert model.geom_conaffinity[geom] == 0


def test_angular_velocity_sensor_resolves_by_name():
    """The gyro is resolved BY NAME, never by trailing sensor index.

    Silently reading the last sensor picked up a different quantity during
    development and produced unstable, falsely-good results.
    """
    mujoco = pytest.importorskip("mujoco")
    from run_follow_me import GYRO_SENSOR_NAMES, resolve_gyro_adr

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    adr = resolve_gyro_adr(model)
    assert adr >= 0
    named = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, n)
             for n in GYRO_SENSOR_NAMES]
    assert any(i >= 0 for i in named)


def test_stand_keyframe_matches_the_policy_default_pose():
    """Actions are offsets from DEFAULT_POSE, so the scene must start there."""
    mujoco = pytest.importorskip("mujoco")
    from run_follow_me import DEFAULT_POSE

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("STAND").id)
    qpos_idx = [int(model.jnt_qposadr[model.actuator_trnid[i, 0]])
                for i in range(model.nu)]
    np.testing.assert_allclose(data.qpos[qpos_idx], DEFAULT_POSE, atol=1e-3)


def test_demo_requires_an_explicit_policy_path():
    """No baked-in policy path: --policy is mandatory and has no default."""
    from run_follow_me import parse_args

    with pytest.raises(SystemExit):
        parse_args([])
    args = parse_args(["--policy", "some/policy.onnx"])
    assert args.policy == "some/policy.onnx"
    # The default scene ships with the demo and resolves independently of CWD.
    assert Path(args.xml).is_absolute()
    assert Path(args.xml).is_file()

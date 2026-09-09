"""Acceptance-gate and repo-invariant tests for follow-me-among-others.

Runs without a policy, a renderer or a GPU. Two things are locked here:

1. The acceptance gates reject the failure modes they were written for. A demo
   whose gates cannot fail is not validating anything, so each gate is shown
   rejecting a synthetic rollout that violates exactly it.
2. The demo respects the repo-wide contracts from AGENTS.md: the 61-D
   observation layout with a 13-D command block, and the documented joint
   layout for the head chain.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = REPO_ROOT / "scripts" / "behavior_demos"
sys.path.insert(0, str(DEMO_DIR))

from follow_me_among_others import metrics as metrics_module
from follow_me_among_others.crowd import CTRL_HZ, TARGET_SEQUENCE

SEQUENCE = TARGET_SEQUENCE
NOMINAL_TRUNK_Z = 0.116


def rollout(
    *,
    sequence=SEQUENCE,
    follow_steps=450,
    step_m=0.004,
    trunk_z=NOMINAL_TRUNK_Z,
    stationary_command=(0.0, 0.0, 0.0),
    target_visible_in_follow=True,
    approach=True,
):
    """Synthesize a records/cycles pair that passes every gate by default."""
    records, cycles, transitions = [], [], []
    t = 0.0
    dt = 1.0 / CTRL_HZ
    x = 0.0
    for selection, target in enumerate(sequence, start=1):
        cycles.append(
            {
                "selection": selection,
                "target": target,
                "search_start_s": t,
                "found_s": t + 0.5,
                "search_duration_s": 0.5,
                "follow_start_s": t + 1.5,
                "stop_s": t + 1.5 + follow_steps * dt,
                "cycle_end_s": t + 3.0 + follow_steps * dt,
            }
        )
        # One stationary step at the FOUND instant, so the "visible at FOUND"
        # gate has a record to look at.
        records.append(
            {
                "t": t + 0.5,
                "state": "FOUND",
                "selection": selection,
                "target": target,
                "target_visible": True,
                "follow_error_m": 1.0,
                "person_range_m": 1.0,
                "yaw_error_deg": 0.0,
                "duck_xy": [x, 0.0],
                "trunk_z_m": trunk_z,
                "command": list(stationary_command),
            }
        )
        t += 1.5
        start_error = 1.0
        for step in range(follow_steps):
            x += step_m
            error = (
                start_error - step * step_m if approach else start_error + step * step_m
            )
            records.append(
                {
                    "t": t,
                    "state": "FOLLOW",
                    "selection": selection,
                    "target": target,
                    "target_visible": target_visible_in_follow,
                    "follow_error_m": max(error, 0.05),
                    "person_range_m": 0.9,
                    "yaw_error_deg": 0.0,
                    "duck_xy": [x, 0.0],
                    "trunk_z_m": trunk_z,
                    "command": [0.24, 0.0, 0.0],
                }
            )
            t += dt
        records.append(
            {
                "t": t,
                "state": "STOP",
                "selection": selection,
                "target": target,
                "target_visible": True,
                "follow_error_m": 0.2,
                "person_range_m": 0.9,
                "yaw_error_deg": 0.0,
                "duck_xy": [x, 0.0],
                "trunk_z_m": trunk_z,
                "command": list(stationary_command),
            }
        )
        t += 1.5
    return records, cycles, transitions


def summarize(records, cycles, transitions, sequence=SEQUENCE):
    return metrics_module.summarize(
        records=records,
        transitions=transitions,
        cycles=cycles,
        sequence=sequence,
        duration_s=60.0,
        control_steps=len(records),
        frames=0,
        trail_distance_m=0.55,
        camera_stats={
            "camera_target_visible_steps": 0,
            "camera_search_steps": 0,
            "camera_search_target_visible_steps": 0,
        },
    )


# --- the happy path --------------------------------------------------------


def test_nominal_rollout_passes_every_gate():
    summary = summarize(*rollout())
    assert metrics_module.check_gates(summary, SEQUENCE) == []
    assert summary["cycles_completed"] == 4
    assert summary["target_sequence_completed"] == list(SEQUENCE)
    assert summary["wrong_color_locks"] == 0
    assert summary["camera_target_visible_follow_pct"] == 100.0
    assert summary["stationary_state_command_max"] == 0.0
    assert summary["fallen_steps"] == 0


# --- each gate must be able to fail ----------------------------------------


def test_gate_rejects_locomotion_in_a_stationary_state():
    # The "zero locomotion when not following" claim.
    summary = summarize(*rollout(stationary_command=(0.01, 0.0, 0.0)))
    failures = metrics_module.check_gates(summary, SEQUENCE)
    assert any("stationary" in f for f in failures)
    assert summary["stationary_state_command_max"] > 0.0


def test_gate_rejects_a_follow_segment_that_never_moved():
    # The RED regression: a command is emitted but the robot stands still.
    summary = summarize(*rollout(step_m=0.0001))
    assert not summary["all_follow_segments_moved"]
    assert any("locomotion" in f for f in metrics_module.check_gates(summary, SEQUENCE))


def test_gate_rejects_a_segment_that_moves_without_approaching():
    summary = summarize(*rollout(approach=False))
    assert not summary["all_follow_segments_approached"]
    assert any("approach" in f for f in metrics_module.check_gates(summary, SEQUENCE))


def test_gate_rejects_losing_the_target_during_follow():
    summary = summarize(*rollout(target_visible_in_follow=False))
    assert summary["camera_target_visible_follow_pct"] == 0.0
    assert any("100%" in f for f in metrics_module.check_gates(summary, SEQUENCE))


def test_gate_rejects_a_fall():
    summary = summarize(*rollout(trunk_z=0.05))
    failures = metrics_module.check_gates(summary, SEQUENCE)
    assert summary["fallen_steps"] > 0
    assert any("fallen_steps" in f for f in failures)
    assert any("trunk z" in f for f in failures)


def test_gate_rejects_a_wrong_color_lock():
    records, cycles, transitions = rollout()
    cycles[2]["target"] = "YELLOW"  # a distractor was followed instead of RED
    summary = summarize(records, cycles, transitions)
    failures = metrics_module.check_gates(summary, SEQUENCE)
    assert summary["wrong_color_locks"] == 1
    assert any("wrong_color_locks" in f for f in failures)


def test_gate_rejects_an_incomplete_sequence():
    records, cycles, transitions = rollout(sequence=SEQUENCE[:3])
    summary = summarize(records, cycles, transitions, sequence=SEQUENCE[:3])
    # Measured against the full four-selection request, three cycles must fail.
    failures = metrics_module.check_gates(summary, SEQUENCE)
    assert any("cycles completed" in f for f in failures)


def test_gate_rejects_found_while_target_not_visible():
    records, cycles, transitions = rollout()
    for record in records:
        if record["state"] == "FOUND":
            record["target_visible"] = False
    summary = summarize(records, cycles, transitions)
    assert not summary["found_while_target_visible"]
    assert any("FOUND" in f for f in metrics_module.check_gates(summary, SEQUENCE))


def test_missing_follow_data_is_an_error_not_a_pass():
    records, cycles, transitions = rollout()
    records = [
        r for r in records if not (r["state"] == "FOLLOW" and r["selection"] == 2)
    ]
    with pytest.raises(RuntimeError, match="never followed"):
        summarize(records, cycles, transitions)


# --- documented thresholds -------------------------------------------------


def test_acceptance_thresholds_are_the_published_ones():
    assert metrics_module.MIN_FOLLOW_PATH_M == pytest.approx(0.40)
    assert metrics_module.MIN_FOLLOW_DISPLACEMENT_M == pytest.approx(0.30)
    assert metrics_module.MIN_FOLLOW_APPROACH_M == pytest.approx(0.05)
    assert metrics_module.FALLEN_TRUNK_Z_M == pytest.approx(0.09)
    assert set(metrics_module.STATIONARY_STATES) == {
        "SEARCH",
        "FOUND",
        "STOP",
        "DONE",
    }


def test_segment_helpers_agree_with_the_thresholds():
    at_limit = {
        "path_distance_m": metrics_module.MIN_FOLLOW_PATH_M,
        "net_displacement_m": metrics_module.MIN_FOLLOW_DISPLACEMENT_M,
        "start_error_m": 1.0,
        "min_error_m": 1.0 - metrics_module.MIN_FOLLOW_APPROACH_M,
    }
    assert metrics_module.segment_moved(at_limit)
    assert metrics_module.segment_approached(at_limit)
    just_under = dict(at_limit, path_distance_m=metrics_module.MIN_FOLLOW_PATH_M - 1e-6)
    assert not metrics_module.segment_moved(just_under)
    no_approach = dict(at_limit, min_error_m=1.0)
    assert not metrics_module.segment_approached(no_approach)


# --- repo invariants -------------------------------------------------------


def test_observation_contract_is_the_shared_61d_layout():
    from follow_me_among_others import run_demo

    # 48 proprioception + 13 command; the command block is never dropped.
    assert run_demo.EXPECTED_OBS_DIM == 61
    assert run_demo.EXPECTED_COMMAND_DIM == 13
    assert 3 + 3 + 14 * 3 + run_demo.EXPECTED_COMMAND_DIM == run_demo.EXPECTED_OBS_DIM


def test_default_pose_matches_the_home_frame():
    from follow_me_among_others import run_demo

    order = [
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
    ]
    assert len(run_demo.DEFAULT_POSE) == len(order) == 14

    # The documented STAND2 pose (HOME_FRAME in microduck_constants.py, and the
    # STAND keyframe in the scene). Actions are offsets from exactly this, so a
    # drift here silently biases every observation the policy sees.
    expected = {
        "left_hip_yaw": 0.0,
        "left_hip_roll": -0.0873,
        "left_hip_pitch": -0.4579,
        "left_knee": -0.0049,
        "left_ankle": 0.4530,
        "neck_pitch": 0.3491,
        "head_pitch": 0.3491,
        "head_yaw": 0.0,
        "head_roll": 0.0,
        "right_hip_yaw": 0.0,
        "right_hip_roll": 0.0873,
        "right_hip_pitch": 0.4579,
        "right_knee": 0.0049,
        "right_ankle": -0.4530,
    }
    for index, joint in enumerate(order):
        assert run_demo.DEFAULT_POSE[index] == pytest.approx(
            expected[joint], abs=1e-4
        ), joint
    # Left and right legs mirror each other.
    assert run_demo.DEFAULT_POSE[order.index("left_hip_pitch")] == pytest.approx(
        -run_demo.DEFAULT_POSE[order.index("right_hip_pitch")], abs=1e-6
    )


def test_head_actuator_indices_follow_the_documented_joint_layout():
    from follow_me_among_others.camera import CrowdCameraSearch

    # AGENTS.md: 0-4 left leg, 5-8 neck/head, 9-13 right leg.
    assert CrowdCameraSearch.HEAD_PITCH_ACT == 6
    assert CrowdCameraSearch.HEAD_YAW_ACT == 7
    assert CrowdCameraSearch.HEAD_ROLL_ACT == 8


def test_demo_runs_at_the_policy_control_rate():
    assert CTRL_HZ == 50.0


def test_scene_declares_the_five_people_and_the_camera_rig():
    scene = (
        REPO_ROOT
        / "src/mjlab_microduck/robot/microduck/scene_follow_me_among_others.xml"
    ).read_text()
    for color in ("blue", "green", "red", "yellow", "purple"):
        assert f'name="person_{color}"' in scene
        assert f"{color}_shirt" in scene
    assert 'name="follow_camera_rig"' in scene
    assert 'name="follow_camera"' in scene
    assert 'name="trail_target"' in scene
    # The demo builds on the official walk model rather than a private copy.
    assert 'include file="robot_walk.xml"' in scene


def test_scene_people_are_non_colliding_scenery():
    scene = (
        REPO_ROOT
        / "src/mjlab_microduck/robot/microduck/scene_follow_me_among_others.xml"
    ).read_text()
    # Every pedestrian geom is contype/conaffinity 0: the crowd can never push
    # the robot, so following is not an artifact of being bumped.
    person_lines = [
        line
        for line in scene.splitlines()
        if "<geom" in line
        and any(c in line for c in ("blue_", "green_", "red_", "yellow_", "purple_"))
    ]
    assert person_lines
    for line in person_lines:
        assert 'contype="0"' in line and 'conaffinity="0"' in line


def test_demo_declares_no_hardcoded_local_paths():
    for path in sorted((DEMO_DIR / "follow_me_among_others").glob("*.py")):
        source = path.read_text()
        assert "/Users/" not in source, path
        assert "/home/" not in source, path
        assert "/tmp/" not in source, path


def test_policy_argument_is_required():
    from follow_me_among_others import run_demo

    # No default weights path: the user always says which policy to run.
    with pytest.raises(SystemExit):
        run_demo.parse_args([])
    args = run_demo.parse_args(["--policy", "walk.onnx", "--no-render"])
    assert args.policy == "walk.onnx"
    assert args.xml.startswith("src/mjlab_microduck/robot/microduck/")


def test_no_binary_artifacts_are_committed_with_the_demo():
    for pattern in ("*.onnx", "*.mp4", "*.png", "*.pt"):
        assert list(DEMO_DIR.rglob(pattern)) == []


def test_make_observation_builds_a_61d_vector_without_a_policy():
    from follow_me_among_others import run_demo

    class _Data:
        xquat = np.array([[1.0, 0.0, 0.0, 0.0]])
        sensordata = np.zeros(16, dtype=np.float64)
        qpos = np.zeros(32, dtype=np.float64)
        qvel = np.zeros(32, dtype=np.float64)

    data = _Data()
    qpos_idx = np.arange(14)
    qvel_idx = np.arange(14)
    data.qpos[qpos_idx] = run_demo.DEFAULT_POSE
    observation = run_demo.make_observation(
        data,
        model=None,
        qpos_idx=qpos_idx,
        qvel_idx=qvel_idx,
        trunk_id=0,
        gyro_adr=0,
        last_action=np.zeros(14, dtype=np.float32),
        command=np.array([0.24, 0.0, 0.3], dtype=np.float32),
        command_dim=13,
    )
    assert observation.shape == (61,)
    assert observation.dtype == np.float32
    # Joint positions are relative to the default pose, so they vanish here.
    assert np.allclose(observation[6:20], 0.0)
    # Twist occupies the first three command slots; the rest is zero-padded.
    assert observation[48:51] == pytest.approx([0.24, 0.0, 0.3])
    assert np.allclose(observation[51:], 0.0)

"""Collect CPU MuJoCo demonstrations for the VelStand gait-onset fit."""

import math
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

from infer_policy import PolicyInference


_SCENE = Path(__file__).resolve().parents[1] / "src/mjlab_microduck/robot/microduck/scene_backlash.xml"
_CONTROL_HZ = 50
_SUBSTEPS = 4

# Entries are (forward m/s, yaw rad/s).
_BASE_COMMANDS = (
    (0, 0), (0.20, 0), (0.25, 0), (-0.12, 0), (0.10, 0), (0.30, 0),
    (0, 2), (0, -2), (0.20, 0.5), (0.20, -0.5),
)
_TEACHER_COMMANDS = (
    (0.14, 0), (0.15, 0), (0.16, 0),
    (0, 1), (0, -1), (0, 0.55), (0, -0.55), (0, 0.8), (0, -0.8),
)
_SLOW_TURNS = tuple((0, sign * rate) for rate in (0.35, 0.50, 0.65, 0.80) for sign in (1, -1))
_STRAIGHT_WALKS = tuple((vx, 0) for vx in (0.14, 0.15, 0.16, 0.18, 0.20, 0.22, 0.25))
_START_SPEEDS = (0.13, 0.14, 0.15, 0.16, 0.17)
_FORWARD_COMMAND = 48
_YAW_COMMAND = 50  # third component of the 61-D actor's twist
# (quaternion vector component, sign) for a quarter turn from STAND.
_FALLS = {"back": (2, -1), "face": (2, 1), "left": (1, 1), "right": (1, -1)}


class _Actor:
    def __init__(self, path):
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def __call__(self, observation):
        return self.session.run([self.output_name], {self.input_name: observation[None]})[0][0]


def _make_policy(actor):
    model = mujoco.MjModel.from_xml_path(str(_SCENE))
    model.opt.timestep = 1 / (_CONTROL_HZ * _SUBSTEPS)
    policy = PolicyInference(
        model, mujoco.MjData(model), walking_session=actor.session, action_scale=0.9,
        use_projected_gravity=True, new_cmd_obs=True,
    )
    gains = np.full(policy.n_joints, 0.7)
    for name in ("neck_pitch", "head_pitch", "head_yaw", "head_roll"):
        gains[model.actuator(name).id] = 0.5
    return policy, gains


def _apply_action(policy, action, gains, *, after_physics=False):
    # robotd filters each motor target against the previous target at 50 Hz.
    policy.last_action = action.copy()
    target_positions = policy.default_pose + policy.action_scale * action
    # A daemon write can arrive just after the simulator advances its 20 ms batch.
    if after_physics:
        mujoco.mj_step(policy.model, policy.data, nstep=_SUBSTEPS)
    policy.data.ctrl[:] = (1 - gains) * policy.data.ctrl + gains * target_positions
    if not after_physics:
        mujoco.mj_step(policy.model, policy.data, nstep=_SUBSTEPS)


def _reset(policy, rng, posture=None, *, start_keyframe="STAND"):
    model, data = policy.model, policy.data
    key = posture if posture in ("SIT", "FOLD") else start_keyframe
    mujoco.mj_resetDataKeyframe(model, data, model.key(key).id)
    root = data.joint("trunk_base_freejoint").qpos
    joint_noise = 0.13 if posture else 0.012
    if posture is None:
        root[2] += rng.uniform(0, 0.004)
    data.qpos[policy.joint_qpos_indices] += rng.uniform(-joint_noise, joint_noise, policy.n_joints)
    if posture in _FALLS:
        component, sign = _FALLS[posture]
        angle = sign * np.pi / 2 + rng.uniform(-0.14, 0.14)
        root[2] = 0.075
        root[3:] = 0
        root[3] = np.cos(angle / 2)
        root[3 + component] = np.sin(angle / 2)
    policy.vel_cmd.fill(0)
    policy.last_action.fill(0)
    policy.head_offset.fill(0)
    mujoco.mj_forward(model, data)


def _heading(quat):
    w, x, y, z = quat
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _tracking_label_obs(obs, turning, root, origin, heading):
    """Give the demonstrator a yaw cue while keeping the student's observation intact."""
    label_obs = obs.copy()
    if turning:
        # v8 often stays still at +0.5 rad/s. A 1.4x teacher cue initiates
        # a turn during the low-rate demonstrations in either direction.
        label_obs[_YAW_COMMAND] *= 1.4
    else:
        dx, dy = root[:2] - origin
        cross_track = -math.sin(heading) * dx + math.cos(heading) * dy
        delta = _heading(root[3:]) - heading
        heading_error = math.atan2(math.sin(delta), math.cos(delta))
        # Positive error means left of the starting heading or line. Return
        # towards it at 2 rad/s per radian and 1 rad/s per metre of error.
        label_obs[_YAW_COMMAND] += np.clip(-2 * heading_error - cross_track, -0.7, 0.7)
    return label_obs


def _rollout(policy, base, teacher, gains, command, settle_steps, active_steps, follow_teacher,
             *, lesson=None):
    data = policy.data
    rest = np.zeros(3)
    target = np.array([command[0], 0.0, command[1]])
    root = data.joint("trunk_base_freejoint").qpos
    for step in range(settle_steps + active_steps):
        moving = step >= settle_steps
        if lesson == "tracking" and step == settle_steps:
            origin = root[:2].copy()
            heading = _heading(root[3:])
        policy.vel_cmd += 0.2 * ((target if moving else rest) - policy.vel_cmd)
        policy._update_command()
        obs = policy.get_observations()
        # Base labels protect standing, get-up and off-target commands. Teach
        # the gait only while upright (projected gravity z and trunk height).
        label_actor = teacher if moving and teacher is not None and obs[5] < -0.8 and root[2] > 0.102 else base
        label_obs = obs
        if lesson == "tracking" and label_actor is teacher:
            label_obs = _tracking_label_obs(obs, command[1] != 0, root, origin, heading)
        elif lesson == "start" and label_actor is teacher and step - settle_steps < _CONTROL_HZ:
            # For the first second of a slow start, let v8 see a command that
            # ramps up to 0.20 m/s. Train v10 on the original, slower command.
            label_obs = obs.copy()
            label_obs[_FORWARD_COMMAND] *= 0.20 / command[0]
        sample = step % 2 == 0 and (moving or lesson is None)
        if follow_teacher:
            action = label_actor(label_obs)
            if sample:
                yield obs, action
        else:
            action = base(obs)
            if sample:
                yield obs, action if label_actor is base else label_actor(label_obs)
        _apply_action(policy, action, gains)


def collect_examples(
    base_path: Path, teacher_path: Path, *, seed: int = 0, recovery_seeds: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Return float32 observations and action labels for a final-layer fit.

    Collect each command twice, first following the label and then the base
    actor. This covers states visited by either policy near the start of a
    gait. Both passes label standing and every unselected command with base.
    Each recovery seed adds six base-only, zero-command episodes of six seconds.
    """
    if seed < 0 or recovery_seeds < 0:
        raise ValueError("seed and recovery_seeds must be nonnegative")
    base, teacher = _Actor(base_path), _Actor(teacher_path)
    policy, gains = _make_policy(base)

    samples = []
    for run, follow_teacher in enumerate((True, False)):
        for index, command in enumerate(_BASE_COMMANDS + _TEACHER_COMMANDS):
            rng = np.random.default_rng(seed + run * 43 + index + 1)
            _reset(policy, rng)
            settle_steps = round((2.3 + (index % 4) * 0.15) * _CONTROL_HZ)
            samples.extend(_rollout(
                policy, base, teacher if command in _TEACHER_COMMANDS else None,
                gains, command, settle_steps, 8 * _CONTROL_HZ, follow_teacher,
            ))
    for posture in ("SIT", "FOLD", *_FALLS):
        for offset in range(recovery_seeds):
            _reset(policy, np.random.default_rng(seed + offset), posture)
            samples.extend(_rollout(policy, base, None, gains, (0, 0), 0, 6 * _CONTROL_HZ, False))

    # Both tracking passes label actions with v8 while upright. Include states
    # reached following the teacher as well as those reached driving the base.
    for commands, first_seed, run_stride, case_stride in (
        (_SLOW_TURNS, 200, 50, 1), (_STRAIGHT_WALKS, 4000, 100, 4),
    ):
        for run, follow_teacher in enumerate((True, False)):
            for index, command in enumerate(commands):
                episode_seed = seed + first_seed + run * run_stride + index * case_stride
                _reset(policy, np.random.default_rng(episode_seed))
                settle_s = 2.0 + 0.1 * (episode_seed % 7) if command[1] else 2.3
                samples.extend(_rollout(
                    policy, base, teacher, gains, command, round(settle_s * _CONTROL_HZ),
                    8 * _CONTROL_HZ, follow_teacher, lesson="tracking",
                ))

    # Course and turn labels share the same action layer as the first step.
    # Weight the first 1.28 s of slow walks explicitly so that tracking can
    # improve without the fitted actor learning to remain still at 0.15 m/s.
    starts = []
    for repeat in range(4):
        keyframe = "SIT" if repeat % 2 else "STAND"
        settle_s = (4.0 if keyframe == "SIT" else 2.0) + 0.18 * repeat
        for index, vx in enumerate(_START_SPEEDS):
            for follow_teacher in (True, False):
                rng = np.random.default_rng(seed + 5200 + 61 * repeat + 3 * index + int(follow_teacher))
                _reset(policy, rng, start_keyframe=keyframe)
                if repeat == 2:
                    policy.head_offset[:] = rng.uniform(-0.1, 0.1, 4)
                starts.extend(_rollout(
                    policy, base, teacher, gains, (vx, 0), round(settle_s * _CONTROL_HZ),
                    64, follow_teacher, lesson="start",
                ))
    samples.extend(starts * 3)
    observations, labels = zip(*samples)
    return np.stack(observations), np.stack(labels)

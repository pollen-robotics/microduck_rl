"""Collect CPU MuJoCo demonstrations for the VelStand gait-onset fit."""

from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

from infer_policy import PolicyInference


_SCENE = Path(__file__).resolve().parents[1] / "src/mjlab_microduck/robot/microduck/scene_backlash.xml"
_CONTROL_HZ = 50
_SUBSTEPS = 4

# Entries are (forward m/s, yaw rad/s). Every actor receives the same command.
_BASE_COMMANDS = (
    (0, 0), (0.20, 0), (0.25, 0), (-0.12, 0), (0.10, 0), (0.30, 0),
    (0, 2), (0, -2), (0.20, 0.5), (0.20, -0.5),
)
_TEACHER_COMMANDS = (
    (0.14, 0), (0.15, 0), (0.16, 0),
    (0, 1), (0, -1), (0, 0.55), (0, -0.55), (0, 0.8), (0, -0.8),
)
# (quaternion vector component, sign) for a quarter turn from STAND.
_FALLS = {"back": (2, -1), "face": (2, 1), "left": (1, 1), "right": (1, -1)}


class _Actor:
    def __init__(self, path, options):
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def __call__(self, observation):
        return self.session.run([self.output_name], {self.input_name: observation[None]})[0][0]


def _reset(policy, rng, posture=None):
    model, data = policy.model, policy.data
    key = posture if posture in ("SIT", "FOLD") else "STAND"
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
    mujoco.mj_forward(model, data)


def _rollout(policy, base, teacher, gains, command, settle_steps, active_steps, follow_teacher):
    data = policy.data
    rest = np.zeros(3)
    target = np.array([command[0], 0.0, command[1]])
    root = data.joint("trunk_base_freejoint").qpos
    for step in range(settle_steps + active_steps):
        moving = step >= settle_steps
        policy.vel_cmd += 0.2 * ((target if moving else rest) - policy.vel_cmd)
        policy._update_command()
        obs = policy.get_observations()
        # Base labels protect standing, get-up and off-target commands. Teach
        # the gait only while upright (projected gravity z and trunk height).
        label_actor = teacher if moving and teacher is not None and obs[5] < -0.8 and root[2] > 0.102 else base
        action_actor = label_actor if follow_teacher else base
        action = action_actor(obs)
        if step % 2 == 0:
            yield obs, action if action_actor is label_actor else label_actor(obs)
        policy.last_action = action.copy()
        # robotd walking defaults: relative to HOME, scale 0.9 and filter the
        # target against the preceding motor target at each 50 Hz tick.
        target_positions = policy.default_pose + policy.action_scale * action
        data.ctrl[:] = (1 - gains) * data.ctrl + gains * target_positions
        mujoco.mj_step(policy.model, data, nstep=_SUBSTEPS)


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
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    base, teacher = _Actor(base_path, options), _Actor(teacher_path, options)
    model = mujoco.MjModel.from_xml_path(str(_SCENE))
    model.opt.timestep = 1 / (_CONTROL_HZ * _SUBSTEPS)
    policy = PolicyInference(
        model, mujoco.MjData(model), walking_session=base.session, action_scale=0.9,
        use_projected_gravity=True, new_cmd_obs=True,
    )
    gains = np.full(policy.n_joints, 0.7)
    for name in ("neck_pitch", "head_pitch", "head_yaw", "head_roll"):
        gains[model.actuator(name).id] = 0.5

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
    observations, labels = zip(*samples)
    return np.stack(observations), np.stack(labels)

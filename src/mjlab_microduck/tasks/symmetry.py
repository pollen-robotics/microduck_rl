"""Bilateral symmetry augmentation for 14-DOF Microduck and 16-DOF Growbot.

Migrated 2026-08-13 from the old 51-D layout to the current 61-D family
(velocity/velstand/standup/roulade — twist + head_command + body_command obs
slots), and the augmented-obs output key fixed "policy" → "actor" (mjlab
1.3.0 group naming; the old key would KeyError in rsl_rl 5.0.1's mirror-loss
path — dead code until now since no env had symmetry enabled).

Actor observation layout (61-dim flat tensor, concatenated in term insertion order):
    [0:3]   base_ang_vel      (roll, pitch, yaw  — body-frame IMU)
    [3:6]   projected_gravity (gx, gy, gz         — body-frame)
    [6:20]  joint_pos_rel     (14 joints, relative to default pose)
    [20:34] joint_vel_rel     (14 joints)
    [34:48] last_action       (14 joints)
    [48:51] twist command     (lin_vel_x, lin_vel_y, ang_vel_z)
    [51:55] head command      (neck_pitch, head_pitch, head_yaw, head_roll deltas)
    [55:61] body command      (x, y, z, roll, pitch, yaw deltas)

Joint ordering within each 14-dim block (from robot_walk.xml body tree):
    0: left_hip_yaw    5: neck_pitch    9:  right_hip_yaw
    1: left_hip_roll   6: head_pitch    10: right_hip_roll
    2: left_hip_pitch  7: head_yaw      11: right_hip_pitch
    3: left_knee       8: head_roll     12: right_knee
    4: left_ankle                       13: right_ankle

Mirroring rules (left-right reflection about the sagittal plane):
- Swap left legs (0-4) with right legs (9-13); midline joints (5-8) stay.
- Negate after swap:
    - hip_yaw, hip_roll: yaw/roll axes reverse under L-R reflection
    - hip_pitch, knee, ankle: home frame uses opposite-sign conventions for
      left vs right (e.g., left_hip_pitch = +0.6, right_hip_pitch = -0.6),
      so relative deviations also negate
    - head_yaw, head_roll: same yaw/roll reasoning
    - neck_pitch, head_pitch: sagittal-plane joints, no sign change
- base_ang_vel: negate roll ([0]) and yaw ([2]); pitch stays
- projected_gravity: negate gy ([4]); gx and gz stay
- twist command: negate lin_vel_y ([49]) and ang_vel_z ([50]); lin_vel_x stays
- head command: negate head_yaw ([53]) and head_roll ([54]); pitches stay
- body command: negate y ([56]), roll ([58]), yaw ([60]); x, z, pitch stay
"""

from dataclasses import dataclass

import torch
from tensordict import TensorDict
from mjlab.rl import RslRlPpoAlgorithmCfg


@dataclass
class PpoWithSymmetryCfg(RslRlPpoAlgorithmCfg):
    """PPO algorithm config extended with an optional symmetry_cfg field."""

    symmetry_cfg: dict | None = None


SYMMETRY_CFG = {
    "use_data_augmentation": False,
    "use_mirror_loss": True,
    "mirror_loss_coeff": 0.5,
    "data_augmentation_func": "mjlab_microduck.tasks.symmetry.microduck_vel_symmetry",
}

# ---------------------------------------------------------------------------
# Permutation and sign tables
# ---------------------------------------------------------------------------

# Within a 14-joint block: left (0-4) <-> right (9-13), midline (5-8) fixed.
# Growbot appends left/right elbow pitch at indices 14/15; those swap without a
# sign change because both elbow hinge axes use the same lateral convention.
_MICRODUCK_JOINT_PERM: list[int] = [9, 10, 11, 12, 13, 5, 6, 7, 8, 0, 1, 2, 3, 4]

# Signs applied AFTER permutation for each joint position
_MICRODUCK_JOINT_SIGN: list[float] = [
    -1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, -1
]

# Cache tensors per device to avoid reallocating on every call
_cache: dict[
    tuple[torch.device, int],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
] = {}


def _symmetry_tables(num_joints: int) -> tuple[list[int], list[float], list[int], list[float]]:
    if num_joints == 14:
        joint_perm = _MICRODUCK_JOINT_PERM
        joint_sign = _MICRODUCK_JOINT_SIGN
    elif num_joints == 16:
        joint_perm = _MICRODUCK_JOINT_PERM + [15, 14]
        joint_sign = _MICRODUCK_JOINT_SIGN + [1.0, 1.0]
    else:
        raise ValueError(f"Symmetry supports 14 or 16 joints, got {num_joints}")

    pos_start = 6
    vel_start = pos_start + num_joints
    action_start = vel_start + num_joints
    command_start = action_start + num_joints
    obs_perm = (
        [0, 1, 2]
        + [3, 4, 5]
        + [pos_start + j for j in joint_perm]
        + [vel_start + j for j in joint_perm]
        + [action_start + j for j in joint_perm]
        + list(range(command_start, command_start + 13))
    )
    obs_sign = (
        [-1.0, 1.0, -1.0]
        + [1.0, -1.0, 1.0]
        + joint_sign * 3
        + [1.0, -1.0, -1.0]
        + [1.0, 1.0, -1.0, -1.0]
        + [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
    )
    return obs_perm, obs_sign, joint_perm, joint_sign


def _get_tensors(
    device: torch.device,
    num_joints: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    key = (device, num_joints)
    if key not in _cache:
        obs_perm, obs_sign, joint_perm, joint_sign = _symmetry_tables(num_joints)
        _cache[key] = (
            torch.tensor(obs_perm, dtype=torch.long, device=device),
            torch.tensor(obs_sign, dtype=torch.float32, device=device),
            torch.tensor(joint_perm, dtype=torch.long, device=device),
            torch.tensor(joint_sign, dtype=torch.float32, device=device),
        )
    return _cache[key]


# ---------------------------------------------------------------------------
# Public augmentation function
# ---------------------------------------------------------------------------


def microduck_vel_symmetry(
    env,
    obs: TensorDict | None,
    actions: torch.Tensor | None,
) -> tuple[TensorDict | None, torch.Tensor | None]:
    """Bilateral symmetry augmentation / mirror function for the microduck vel env.

    Returns [original, mirrored] concatenated along the batch dimension.
    Compatible with the rsl_rl PPO ``symmetry_cfg`` interface (use_data_augmentation
    and/or use_mirror_loss).

    Args:
        env: The vectorised environment (unused, present for interface compatibility).
        obs: TensorDict with keys ``"policy"`` and ``"critic"``, shape ``[B, obs_dim]``.
             Pass ``None`` when only actions need to be mirrored.
        actions: Float tensor of shape ``[B, 14]``.
                 Pass ``None`` when only obs need to be mirrored.

    Returns:
        Tuple ``(aug_obs, aug_actions)`` where each non-None input is doubled
        along the batch axis as ``[original; mirrored]``.
    """
    aug_obs: TensorDict | None = None
    aug_actions: torch.Tensor | None = None

    if obs is not None:
        actor_orig: torch.Tensor = obs["actor"]
        num_joints = (actor_orig.shape[-1] - 19) // 3
        obs_perm, obs_sign, _, _ = _get_tensors(actor_orig.device, num_joints)
        actor_sym = actor_orig[:, obs_perm] * obs_sign

        critic_orig: torch.Tensor = obs["critic"]
        # Critic obs mirroring is not implemented (not needed for use_mirror_loss).
        # For use_data_augmentation the critic sees a repeated unmirrored obs,
        # which is a harmless approximation since the critic uses privileged info
        # not present in the actor obs.
        critic_repeated = torch.cat([critic_orig, critic_orig], dim=0)

        aug_obs = TensorDict(
            {
                "actor": torch.cat([actor_orig, actor_sym], dim=0),
                "critic": critic_repeated,
            },
            batch_size=[actor_orig.shape[0] * 2],
            device=actor_orig.device,
        )

    if actions is not None:
        _, _, act_perm, act_sign = _get_tensors(actions.device, actions.shape[-1])
        actions_sym = actions[:, act_perm] * act_sign
        aug_actions = torch.cat([actions, actions_sym], dim=0)

    return aug_obs, aug_actions

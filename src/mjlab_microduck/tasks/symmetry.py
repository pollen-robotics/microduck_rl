"""Bilateral (left-right) symmetry augmentation for the microduck 61-D envs.

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

# Within a 14-joint block: left (0-4) <-> right (9-13), midline (5-8) fixed
_JOINT_PERM: list[int] = [9, 10, 11, 12, 13, 5, 6, 7, 8, 0, 1, 2, 3, 4]

# Signs applied AFTER permutation for each joint position
_JOINT_SIGN: list[float] = [-1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, -1]

# Full 61-dim actor obs permutation (all command slots mirror in place)
_OBS_PERM: list[int] = (
    [0, 1, 2]                           # base_ang_vel (indices unchanged)
    + [3, 4, 5]                         # projected_gravity
    + [6 + j for j in _JOINT_PERM]     # joint_pos
    + [20 + j for j in _JOINT_PERM]    # joint_vel
    + [34 + j for j in _JOINT_PERM]    # last_action
    + [48, 49, 50]                      # twist command
    + [51, 52, 53, 54]                  # head command
    + [55, 56, 57, 58, 59, 60]          # body command
)

# Full 61-dim sign vector
_OBS_SIGN: list[float] = (
    [-1.0, 1.0, -1.0]   # base_ang_vel: negate roll, yaw
    + [1.0, -1.0, 1.0]  # projected_gravity: negate gy
    + _JOINT_SIGN       # joint_pos
    + _JOINT_SIGN       # joint_vel
    + _JOINT_SIGN       # last_action
    + [1.0, -1.0, -1.0] # twist: negate lin_vel_y, ang_vel_z
    + [1.0, 1.0, -1.0, -1.0]  # head: negate head_yaw, head_roll
    + [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]  # body: negate y, roll, yaw
)

# Cache tensors per device to avoid reallocating on every call
_cache: dict[torch.device, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _get_tensors(
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if device not in _cache:
        obs_perm = torch.tensor(_OBS_PERM, dtype=torch.long, device=device)
        obs_sign = torch.tensor(_OBS_SIGN, dtype=torch.float32, device=device)
        act_perm = torch.tensor(_JOINT_PERM, dtype=torch.long, device=device)
        act_sign = torch.tensor(_JOINT_SIGN, dtype=torch.float32, device=device)
        _cache[device] = (obs_perm, obs_sign, act_perm, act_sign)
    return _cache[device]


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
        actor_orig: torch.Tensor = obs["actor"]  # [B, 51]
        obs_perm, obs_sign, _, _ = _get_tensors(actor_orig.device)
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
        _, _, act_perm, act_sign = _get_tensors(actions.device)
        actions_sym = actions[:, act_perm] * act_sign
        aug_actions = torch.cat([actions, actions_sym], dim=0)

    return aug_obs, aug_actions

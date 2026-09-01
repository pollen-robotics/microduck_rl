"""MDP functions for microduck tasks"""

import math
from dataclasses import dataclass as _dataclass

import numpy as np
import torch
from typing import TYPE_CHECKING, Optional
import mujoco

from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.reward_manager import RewardManager as _RewardManager
from mjlab.entity import Entity
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand, UniformVelocityCommandCfg
from mjlab.tasks.velocity.mdp import rewards as _velocity_rewards
from mjlab.tasks.velocity.mdp import observations as _velocity_obs
from mjlab.managers.command_manager import CommandTerm
from mjlab.managers import CommandTermCfg
from mjlab.managers.event_manager import requires_model_fields
from mjlab.utils.lab_api.math import matrix_from_quat, wrap_to_pi, quat_apply, quat_from_angle_axis
from rsl_rl.algorithms.ppo import PPO as _PPO

# ---------------------------------------------------------------------------
# Patch 1: RewardManager.compute — sanitize NaN rewards before they enter the
# PPO buffer.  mjlab computes rewards BEFORE resetting environments, so any
# reward term operating on a NaN physics state returns NaN.  That NaN
# propagates: NaN reward → NaN advantage → NaN loss → NaN gradient →
# NaN/negative std → crash in torch.normal on the next mini-batch.
# ---------------------------------------------------------------------------
_orig_reward_compute = _RewardManager.compute

def _nan_safe_reward_compute(self, dt: float) -> torch.Tensor:
    result = _orig_reward_compute(self, dt)
    # _episode_sums is updated inside compute() before nan_to_num can act.
    # Sanitize in-place so per-term metrics don't show NaN.
    for key in self._episode_sums:
        torch.nan_to_num_(self._episode_sums[key], nan=0.0)
    return torch.nan_to_num(result, nan=0.0)

_RewardManager.compute = _nan_safe_reward_compute

# ---------------------------------------------------------------------------
# Patch 2: PPO.compute_returns — sanitize advantages before normalization.
# At a sudden curriculum step (e.g. reward weight ×2.5) the value function is
# badly wrong: all TD errors shift by the same amount, std(advantages) → tiny,
# and (A − mean) / (std + 1e-8) → huge.  That blows up the gradient for std,
# which the optimizer then pushes below zero.  Zeroing NaN/Inf advantages
# before normalization keeps them in a safe range.
# ---------------------------------------------------------------------------
_orig_compute_returns = _PPO.compute_returns

def _safe_compute_returns(self, obs) -> None:
    _orig_compute_returns(self, obs)
    st = self.storage
    torch.nan_to_num_(st.advantages, nan=0.0, posinf=0.0, neginf=0.0)
    torch.nan_to_num_(st.returns,    nan=0.0, posinf=0.0, neginf=0.0)

_PPO.compute_returns = _safe_compute_returns

# Patch 3 (ActorCritic._update_distribution std-clamp) was REMOVED in the mjlab
# 1.3.0 migration: rsl_rl 5.0.1 refactored the policy (no ActorCritic class; the
# distribution now lives in rsl_rl.modules.distribution). It was a defensive
# band-aid against std going negative/NaN (microban runs fine without it). If
# std-blowup recurs under 1.3.0, reinstate it against the new GaussianDistribution.

print("[mdp] Patches 1-2 active: NaN-safe reward/advantage")

# ---------------------------------------------------------------------------
# Patch 4: exporter_utils.get_base_metadata — the new microduck model has
# passive joints (jaw linkage closed via equality constraints) that are part
# of the articulation but have no XML actuator.  The upstream exporter
# iterates robot.joint_names (16) and indexes joint_name_to_ctrl_id (14),
# crashing with KeyError on passive_*.  Filter passive joints out of the
# exported metadata so policies stay consistent with the 14-dim action space.
# ---------------------------------------------------------------------------
from mjlab.rl import exporter_utils as _exporter_utils  # noqa: E402
from mjlab.envs.mdp.actions import JointPositionAction as _JointAction  # noqa: E402

def _get_base_metadata_no_passive(env, run_path):
    robot = env.scene["robot"]
    joint_action = env.action_manager.get_term("joint_pos")
    assert isinstance(joint_action, _JointAction)
    full_names = list(robot.joint_names)
    keep_idx = [i for i, n in enumerate(full_names) if not n.startswith("passive_")]
    joint_names = [full_names[i] for i in keep_idx]
    joint_name_to_ctrl_id = {a.target.split("/")[-1]: a.id for a in robot.spec.actuators}
    ctrl_ids = [joint_name_to_ctrl_id[n] for n in joint_names]
    stiffness = env.sim.mj_model.actuator_gainprm[ctrl_ids, 0]
    damping = -env.sim.mj_model.actuator_biasprm[ctrl_ids, 2]
    default_jp = robot.data.default_joint_pos[0].cpu().tolist()
    return {
        "run_path": run_path,
        "joint_names": joint_names,
        "joint_stiffness": stiffness.tolist(),
        "joint_damping": damping.tolist(),
        "default_joint_pos": [default_jp[i] for i in keep_idx],
        "command_names": list(env.command_manager.active_terms),
        "observation_names": env.observation_manager.active_terms["actor"],
        "action_scale": joint_action._scale[0].cpu().tolist()
        if isinstance(joint_action._scale, torch.Tensor)
        else joint_action._scale,
    }

_exporter_utils.get_base_metadata = _get_base_metadata_no_passive
# Also patch the already-imported reference in the velocity task exporter.
try:
    from mjlab.tasks.velocity.rl import exporter as _vel_exporter  # noqa: E402
    if hasattr(_vel_exporter, "get_base_metadata"):
        _vel_exporter.get_base_metadata = _get_base_metadata_no_passive
except Exception:
    pass

print("[mdp] Patch 4 active: ONNX export filters passive_* joints")

if TYPE_CHECKING:
    from mjlab.viewer.debug_visualizer import DebugVisualizer


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

# Name patterns matching the 4 neck/head actuated joints. Used by head_pose
# tracking reward and by UniformPoseCommand asset hookups.
_NECK_JOINT_PATTERNS = [r".*neck_pitch.*", r".*head_pitch.*", r".*head_yaw.*", r".*head_roll.*"]


def _servo_joint_ids(env: "ManagerBasedRlEnv", asset: Entity) -> list:
    """Entity-local indices of the servo (non-``passive_``) joints, cached.

    All joint-index-based reward/event params in this module (``joint_indices``,
    ``target_overrides``, qpos-column math) are written against the canonical
    14-servo layout. On models with extra unactuated joints — backlash hinges,
    roller wheels, the jaw linkage, all named ``passive_*`` — the entity joint
    array is wider and interleaved, so raw indices would select the wrong
    joints. Index through this list to recover the servo-only view; on plain
    models it is the identity.
    """
    cache = env.__dict__.setdefault("_servo_joint_ids_cache", {})
    key = id(asset)
    ids = cache.get(key)
    if ids is None:
        ids, _ = asset.find_joints(r"^(?!passive_).*")
        cache[key] = ids
    return ids


def _servo_joint_pos(env: "ManagerBasedRlEnv", asset: Entity) -> torch.Tensor:
    return asset.data.joint_pos[:, _servo_joint_ids(env, asset)]


def _servo_joint_vel(env: "ManagerBasedRlEnv", asset: Entity) -> torch.Tensor:
    return asset.data.joint_vel[:, _servo_joint_ids(env, asset)]


def _servo_default_joint_pos(env: "ManagerBasedRlEnv", asset: Entity) -> torch.Tensor:
    return asset.data.default_joint_pos[:, _servo_joint_ids(env, asset)]


def reset_with_forward_velocity(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    velocity_range: tuple[float, float] = (0.3, 0.8),
    fraction_stages: list[dict] | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """Warm-start a fraction of reset environments with a random forward velocity.

    The robot spawns already moving in its body-forward direction, so it first
    discovers what coasting at speed feels like. The fraction decreases over
    training, forcing it to progressively earn that speed from rest.

    Args:
        velocity_range: (min, max) forward speed in m/s.
        fraction_stages: list of {"step": int, "fraction": float} dicts, sorted by step.
            The fraction active at the current training step is used.
            Example: [{"step":0,"fraction":0.8}, {"step":2000*24,"fraction":0.0}]
        asset_cfg: robot entity config.
    """
    if fraction_stages is None:
        fraction_stages = [{"step": 0, "fraction": 0.8}]

    # Determine current fraction from training step
    step = env.common_step_counter
    fraction = fraction_stages[0]["fraction"]
    for stage in fraction_stages:
        if step >= stage["step"]:
            fraction = stage["fraction"]

    if len(env_ids) == 0 or fraction <= 0.0:
        return

    n_warmstart = max(1, int(len(env_ids) * fraction))
    perm = torch.randperm(len(env_ids), device=env.device)[:n_warmstart]
    warmstart_ids = env_ids[perm]

    lo, hi = velocity_range
    vx = lo + torch.rand(n_warmstart, device=env.device) * (hi - lo)

    # Build horizontal forward direction from yaw only — ignoring pitch/roll.
    # IMPORTANT: read quaternion from qpos, NOT from root_link_quat_w.
    # root_link_quat_w reads xquat which requires sim.forward() to be current.
    # After reset_base writes a new yaw to qpos, xquat is still stale (old episode).
    # qpos is updated immediately by write_root_pose, so it's always fresh.
    asset: Entity = env.scene[asset_cfg.name]
    qpos_q_adr = asset.data.indexing.free_joint_q_adr[3:7]  # quat indices in qpos
    q = asset.data.data.qpos[warmstart_ids][:, qpos_q_adr]  # (n, 4) [w, x, y, z]
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    forward_world = torch.stack([torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)], dim=-1)

    velocities = torch.zeros(n_warmstart, 6, device=env.device)
    velocities[:, :3] = vx.unsqueeze(-1) * forward_world

    asset.write_root_link_velocity_to_sim(velocities, env_ids=warmstart_ids)

    # Spin wheels to match forward velocity — prevents instantaneous no-slip braking.
    # Wheel radius = 0.0175 m (measured).
    # All 4 wheels spin at +ω for forward motion (verified by test_wheel_direction.py).
    _WHEEL_RADIUS = 0.0175
    all_wheel_ids, _ = asset.find_joints(r"^passive_.*")

    if all_wheel_ids:
        joint_pos = asset.data.joint_pos[warmstart_ids].clone()
        joint_vel = asset.data.joint_vel[warmstart_ids].clone()
        omega = vx / _WHEEL_RADIUS  # (n,) rad/s, positive = forward
        joint_vel[:, all_wheel_ids] = omega.unsqueeze(-1).expand(-1, len(all_wheel_ids))
        asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=warmstart_ids)


def reset_action_history(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
):
    """
    Reset cached action history for environments that are being reset.
    This is critical for action rate and acceleration penalty terms.

    This function should be called in the post_reset callback or at episode termination.

    Args:
        env: The environment
        env_ids: Indices of environments being reset
        asset_cfg: Asset configuration
    """
    if len(env_ids) == 0:
        return

    asset: Entity = env.scene[asset_cfg.name]

    # Reset leg action rate cache
    if hasattr(env, '_prev_leg_actions'):
        # Set to current action (or zero if no action yet)
        if hasattr(env, 'action_manager') and env.action_manager.action is not None:
            leg_joint_indices = list(range(0, 5)) + list(range(9, 14))
            env._prev_leg_actions[env_ids] = env.action_manager.action[env_ids][:, leg_joint_indices]
        else:
            env._prev_leg_actions[env_ids] = 0.0

    # Reset neck action rate cache
    if hasattr(env, '_prev_neck_actions'):
        if hasattr(env, 'action_manager') and env.action_manager.action is not None:
            neck_joint_indices = list(range(5, 9))
            env._prev_neck_actions[env_ids] = env.action_manager.action[env_ids][:, neck_joint_indices]
        else:
            env._prev_neck_actions[env_ids] = 0.0

    # Reset leg action acceleration cache
    if hasattr(env, '_prev_leg_actions_for_acc'):
        if hasattr(env, 'action_manager') and env.action_manager.action is not None:
            leg_joint_indices = list(range(0, 5)) + list(range(9, 14))
            current_action = env.action_manager.action[env_ids][:, leg_joint_indices]
            env._prev_leg_actions_for_acc[env_ids] = current_action
            env._prev_prev_leg_actions_for_acc[env_ids] = current_action
        else:
            env._prev_leg_actions_for_acc[env_ids] = 0.0
            env._prev_prev_leg_actions_for_acc[env_ids] = 0.0

    # Reset neck action acceleration cache
    if hasattr(env, '_prev_neck_actions_for_acc'):
        if hasattr(env, 'action_manager') and env.action_manager.action is not None:
            neck_joint_indices = list(range(5, 9))
            current_action = env.action_manager.action[env_ids][:, neck_joint_indices]
            env._prev_neck_actions_for_acc[env_ids] = current_action
            env._prev_prev_neck_actions_for_acc[env_ids] = current_action
        else:
            env._prev_neck_actions_for_acc[env_ids] = 0.0
            env._prev_prev_neck_actions_for_acc[env_ids] = 0.0

    # Reset joint velocity cache for joint accelerations
    if hasattr(asset.data, '_prev_joint_vel'):
        # Get current joint velocities for reset environments
        joint_vel = asset.data.joint_vel[env_ids, :][:, asset_cfg.joint_ids]
        asset.data._prev_joint_vel[env_ids] = joint_vel

    # Reset contact frequency tracking
    if hasattr(env, '_contact_change_count'):
        env._contact_change_count[env_ids] = 0.0
    if hasattr(env, '_contact_change_timer'):
        env._contact_change_timer[env_ids] = 0.0
    if hasattr(env, '_prev_contacts_for_freq'):
        if "feet_ground_contact" in env.scene.sensors:
            contacts = env.scene.sensors["feet_ground_contact"].data.found[env_ids, :2]
            env._prev_contacts_for_freq[env_ids] = contacts

    # Reset foot force smoothness tracking
    if hasattr(env, '_prev_foot_forces'):
        if "feet_ground_contact" in env.scene.sensors:
            forces = env.scene.sensors["feet_ground_contact"].data.found[env_ids, :2].squeeze(-1)
            env._prev_foot_forces[env_ids] = forces

    # Reset actuator torque rate tracking
    if hasattr(env, '_prev_actuator_forces'):
        env._prev_actuator_forces[env_ids] = asset.data.actuator_force[env_ids].clone()


def joint_accelerations_l2(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """
    Penalize joint accelerations using L2 squared norm.
    Joint accelerations are computed using finite differences of joint velocities.

    Args:
        env: The environment
        asset_cfg: Asset configuration

    Returns:
        Penalty tensor of shape (num_envs,) - sum of squared joint accelerations
    """
    asset: Entity = env.scene[asset_cfg.name]

    # Get current joint velocities
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]

    # Get previous joint velocities (stored in asset data)
    # Note: This assumes the environment stores previous joint velocities
    if not hasattr(asset.data, '_prev_joint_vel'):
        # Initialize on first call
        asset.data._prev_joint_vel = joint_vel.clone()
        return torch.zeros(env.num_envs, device=env.device)

    # Compute joint accelerations using finite differences
    dt = env.step_dt
    joint_acc = (joint_vel - asset.data._prev_joint_vel) / dt

    # Store current velocities for next step
    asset.data._prev_joint_vel = joint_vel.clone()

    # Return L2 squared norm
    return torch.sum(torch.square(joint_acc), dim=1)


def leg_action_rate_l2(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """
    Penalize the rate of change of leg actions (action_t - action_{t-1}).
    Leg joints are indices 0-4 and 9-13 (10 joints total).

    Args:
        env: The environment
        asset_cfg: Asset configuration

    Returns:
        Penalty tensor of shape (num_envs,)
    """
    # Get leg joint indices
    leg_joint_indices = list(range(0, 5)) + list(range(9, 14))

    # Get current and previous actions for leg joints only
    # Actions are stored in env (assuming the action is available)
    if not hasattr(env, 'action_manager'):
        return torch.zeros(env.num_envs, device=env.device)

    # Get the joint position action
    actions = env.action_manager.action
    if actions.shape[1] < 14:
        return torch.zeros(env.num_envs, device=env.device)

    leg_actions = actions[:, leg_joint_indices]

    if not hasattr(env, '_prev_leg_actions'):
        env._prev_leg_actions = leg_actions.clone()
        return torch.zeros(env.num_envs, device=env.device)

    action_rate = leg_actions - env._prev_leg_actions
    env._prev_leg_actions = leg_actions.clone()

    return torch.sum(torch.square(action_rate), dim=1)


def neck_action_rate_l2(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """
    Penalize the rate of change of neck actions (action_t - action_{t-1}).
    Neck joints are indices 5-8 (4 joints total).

    Args:
        env: The environment
        asset_cfg: Asset configuration

    Returns:
        Penalty tensor of shape (num_envs,)
    """
    # Get neck joint indices
    neck_joint_indices = list(range(5, 9))

    # Get current and previous actions for neck joints only
    if not hasattr(env, 'action_manager'):
        return torch.zeros(env.num_envs, device=env.device)

    actions = env.action_manager.action
    if actions.shape[1] < 14:
        return torch.zeros(env.num_envs, device=env.device)

    neck_actions = actions[:, neck_joint_indices]

    if not hasattr(env, '_prev_neck_actions'):
        env._prev_neck_actions = neck_actions.clone()
        return torch.zeros(env.num_envs, device=env.device)

    action_rate = neck_actions - env._prev_neck_actions
    env._prev_neck_actions = neck_actions.clone()

    return torch.sum(torch.square(action_rate), dim=1)


def leg_action_acceleration_l2(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """
    Penalize leg action accelerations (action_t - 2*action_{t-1} + action_{t-2}).
    Leg joints are indices 0-4 and 9-13 (10 joints total).

    Args:
        env: The environment
        asset_cfg: Asset configuration

    Returns:
        Penalty tensor of shape (num_envs,)
    """
    # Get leg joint indices
    leg_joint_indices = list(range(0, 5)) + list(range(9, 14))

    if not hasattr(env, 'action_manager'):
        return torch.zeros(env.num_envs, device=env.device)

    actions = env.action_manager.action
    if actions.shape[1] < 14:
        return torch.zeros(env.num_envs, device=env.device)

    leg_actions = actions[:, leg_joint_indices]

    if not hasattr(env, '_prev_leg_actions_for_acc'):
        env._prev_leg_actions_for_acc = leg_actions.clone()
        env._prev_prev_leg_actions_for_acc = leg_actions.clone()
        return torch.zeros(env.num_envs, device=env.device)

    action_acc = leg_actions - 2 * env._prev_leg_actions_for_acc + env._prev_prev_leg_actions_for_acc

    env._prev_prev_leg_actions_for_acc = env._prev_leg_actions_for_acc.clone()
    env._prev_leg_actions_for_acc = leg_actions.clone()

    return torch.sum(torch.square(action_acc), dim=1)


def neck_action_acceleration_l2(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """
    Penalize neck action accelerations (action_t - 2*action_{t-1} + action_{t-2}).
    Neck joints are indices 5-8 (4 joints total).

    Args:
        env: The environment
        asset_cfg: Asset configuration

    Returns:
        Penalty tensor of shape (num_envs,)
    """
    # Get neck joint indices
    neck_joint_indices = list(range(5, 9))

    if not hasattr(env, 'action_manager'):
        return torch.zeros(env.num_envs, device=env.device)

    actions = env.action_manager.action
    if actions.shape[1] < 14:
        return torch.zeros(env.num_envs, device=env.device)

    neck_actions = actions[:, neck_joint_indices]

    if not hasattr(env, '_prev_neck_actions_for_acc'):
        env._prev_neck_actions_for_acc = neck_actions.clone()
        env._prev_prev_neck_actions_for_acc = neck_actions.clone()
        return torch.zeros(env.num_envs, device=env.device)

    action_acc = neck_actions - 2 * env._prev_neck_actions_for_acc + env._prev_prev_neck_actions_for_acc

    env._prev_prev_neck_actions_for_acc = env._prev_neck_actions_for_acc.clone()
    env._prev_neck_actions_for_acc = neck_actions.clone()

    return torch.sum(torch.square(action_acc), dim=1)


def _fallen_mask(
    env: ManagerBasedRlEnv,
    asset,
    gate_z_below: float,
    gate_tilt_above_deg: float,
) -> torch.Tensor:
    """Per-env float mask: 1.0 where the robot counts as FALLEN — trunk height
    below `gate_z_below` OR tilt beyond `gate_tilt_above_deg`. Used to gate the
    recovery rewards so they only steer while actually fallen and contribute
    exactly zero during clean walking (no walk tax / bounce farming)."""
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    quat = asset.data.root_link_quat_w
    # cos(tilt) = R22 = 1 - 2(qx² + qy²)
    cos_tilt = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    fallen = (z < gate_z_below) | (cos_tilt < math.cos(math.radians(gate_tilt_above_deg)))
    return fallen.float()


def feet_air_time_upright(
    env: ManagerBasedRlEnv,
    gate_tilt_above_deg: float = 40.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    **air_time_kwargs,
) -> torch.Tensor:
    """velocity template feet_air_time, zeroed while FALLEN (tilt > gate).

    velstand: a robot lying on its trunk can still tap its feet rhythmically
    through the air-time window — the observed "lies there shaking a leg"
    exploit. Air time is only meaningful upright.
    """
    from mjlab.tasks.velocity.mdp import feet_air_time as _template_air_time
    reward = _template_air_time(env, **air_time_kwargs)
    asset: Entity = env.scene[asset_cfg.name]
    upright = 1.0 - _fallen_mask(env, asset, 0.0, gate_tilt_above_deg)
    return reward * upright


def upright_progress(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Potential-based upright shaping: Δcos(tilt) per step.

    Pays for PROGRESS toward upright, charges for progress toward fallen, and
    pays exactly ZERO for holding any pose — so no state can farm it (the gated
    state-reward it replaces was farmed from sitting, lying flat, and a
    head-tripod lean across three velstand runs). Potential-based shaping is
    policy-invariant (Ng et al.): it accelerates learning of recovery without
    creating new optima. A full prone→stand recovery collects Δ≈+1 total
    (× weight); a fall costs the same on the way down.
    """
    asset: Entity = env.scene[asset_cfg.name]
    quat = asset.data.root_link_quat_w
    cos_tilt = torch.nan_to_num(
        1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2), nan=1.0
    )
    if not hasattr(env, "_upright_potential_prev"):
        env._upright_potential_prev = cos_tilt.clone()
    # Freshly reset envs: no spurious delta from the previous episode's pose.
    fresh = env.episode_length_buf <= 1
    env._upright_potential_prev[fresh] = cos_tilt[fresh]
    delta = cos_tilt - env._upright_potential_prev
    env._upright_potential_prev = cos_tilt.clone()
    return delta


def height_progress(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ceiling: float = 0.115,
) -> torch.Tensor:
    """Potential-based height shaping: Δ min(trunk z, ceiling) per step.

    The z-axis companion to ``upright_progress`` (velstand crouch-endpoint
    lesson): the last mile of a recovery — extending the knees out of a deep
    crouch — is mostly a HEIGHT change at modest tilt, exactly where the
    Gaussian upright/pose rewards are flat and Δcos(tilt) is tiny. Rising pays,
    falling charges, holding pays zero, so gait bobbing nets zero and nothing
    can farm it. Capped at ``ceiling`` (just below full-stand trunk z ≈ 0.117)
    so hopping above stance height pays nothing extra.
    """
    asset: Entity = env.scene[asset_cfg.name]
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    pot = torch.clamp(z, max=ceiling)
    if not hasattr(env, "_height_potential_prev"):
        env._height_potential_prev = pot.clone()
    fresh = env.episode_length_buf <= 1
    env._height_potential_prev[fresh] = pot[fresh]
    delta = pot - env._height_potential_prev
    env._height_potential_prev = pot.clone()
    return delta


def _stair_local_state(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return root x/z and upright score in the local terrain frame."""
    asset: Entity = env.scene[asset_cfg.name]
    terrain = env.scene.terrain
    x = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 0] - terrain.env_origins[:, 0], nan=0.0
    )
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - terrain.env_origins[:, 2], nan=0.0
    )
    quat = asset.data.root_link_quat_w
    upright = torch.clamp(
        torch.nan_to_num(
            1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2), nan=0.0
        ),
        min=0.0,
        max=1.0,
    )
    return x, z, upright


def _stair_approach_gate(
    env: ManagerBasedRlEnv,
    stair_start_distance: float,
    fade_distance: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    x, _, _ = _stair_local_state(env, asset_cfg)
    return torch.sigmoid(
        (stair_start_distance - x) / max(fade_distance, 1e-6)
    )


def _stair_corridor_gate(
    env: ManagerBasedRlEnv,
    corridor_half_width: float | None,
    corridor_fade: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    if corridor_half_width is None:
        return torch.ones(env.num_envs, device=env.device)
    asset: Entity = env.scene[asset_cfg.name]
    y = torch.abs(
        torch.nan_to_num(
            asset.data.root_link_pos_w[:, 1] - env.scene.terrain.env_origins[:, 1],
            nan=0.0,
        )
    )
    return torch.sigmoid(
        (corridor_half_width - y) / max(corridor_fade, 1e-6)
    )


def stair_approach_upright(
    env: ManagerBasedRlEnv,
    std: float,
    stair_start_distance: float,
    fade_distance: float = 0.20,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Preserve the transferred walk posture only on the flat runway."""
    _, _, upright = _stair_local_state(env, asset_cfg)
    sin_tilt_squared = torch.clamp(1.0 - upright.square(), min=0.0, max=1.0)
    score = torch.exp(-sin_tilt_squared / max(std**2, 1e-6))
    return score * _stair_approach_gate(
        env, stair_start_distance, fade_distance, asset_cfg
    )


def stair_approach_linear_tracking(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    stair_start_distance: float,
    fade_distance: float = 0.20,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Track the normal walking command until the first stair, then fade it."""
    tracking = _velocity_rewards.track_linear_velocity(
        env, std=std, command_name=command_name, asset_cfg=asset_cfg
    )
    return tracking * _stair_approach_gate(
        env, stair_start_distance, fade_distance, asset_cfg
    )


def stair_approach_angular_tracking(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    stair_start_distance: float,
    fade_distance: float = 0.20,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Fade turn tracking at the first riser so the climb can self-organize."""
    tracking = _velocity_rewards.track_angular_velocity(
        env, std=std, command_name=command_name, asset_cfg=asset_cfg
    )
    return tracking * _stair_approach_gate(
        env, stair_start_distance, fade_distance, asset_cfg
    )


def configure_stair_viewer_grid(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None = None,
    terrain_levels: tuple[int, ...] = (0, 1, 0, 1),
    terrain_types: tuple[int, ...] = (0, 0, 1, 1),
) -> None:
    """Place four play environments on four distinct 2x2 stair patches."""
    del env_ids
    terrain = env.scene.terrain
    if terrain.terrain_origins is None:
        return
    if env.num_envs < len(terrain_levels) or len(terrain_levels) != len(terrain_types):
        raise ValueError("The stair viewer grid requires four matching environment origins.")
    levels = torch.tensor(terrain_levels, device=env.device, dtype=torch.long)
    types = torch.tensor(terrain_types, device=env.device, dtype=torch.long)
    rows, cols = terrain.terrain_origins.shape[:2]
    if int(levels.max()) >= rows or int(types.max()) >= cols:
        raise ValueError("Stair viewer grid indices exceed the generated terrain grid.")
    terrain.terrain_levels[: len(levels)] = levels
    terrain.terrain_types[: len(types)] = types
    terrain.env_origins[: len(levels)] = terrain.terrain_origins[levels, types]


def force_stair_terrain_level(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None = None,
    terrain_level: int = 2,
    terrain_type: int = 0,
) -> None:
    """Place every vectorized environment on one exact stair geometry."""

    del env_ids
    terrain = env.scene.terrain
    if terrain.terrain_origins is None:
        return
    rows, cols = terrain.terrain_origins.shape[:2]
    if not 0 <= terrain_level < rows or not 0 <= terrain_type < cols:
        raise ValueError("Forced stair terrain index exceeds the generated grid")
    terrain.terrain_levels[:] = terrain_level
    terrain.terrain_types[:] = terrain_type
    terrain.env_origins[:] = terrain.terrain_origins[terrain_level, terrain_type]


def seed_route_challenge_levels(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    standard_fraction: float = 0.125,
) -> None:
    """Keep part of each batch on the full-height stair from the beginning."""
    terrain = env.scene.terrain
    if terrain.terrain_origins is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device, dtype=torch.long)
    if not hasattr(env, "_route_challenge_mask"):
        env._route_challenge_mask = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
    env._route_challenge_mask[env_ids] = False
    if len(env_ids) == env.num_envs:
        challenge_count = min(
            len(env_ids), max(1, int(round(len(env_ids) * standard_fraction)))
        )
        selected_ids = env_ids[
            torch.randperm(len(env_ids), device=env.device)[:challenge_count]
        ]
    else:
        selected_ids = env_ids[
            torch.rand(len(env_ids), device=env.device) < standard_fraction
        ]
    if len(selected_ids) == 0:
        return
    env._route_challenge_mask[selected_ids] = True
    max_level = terrain.terrain_origins.shape[0] - 1
    terrain.terrain_levels[selected_ids] = max_level
    terrain.env_origins[selected_ids] = terrain.terrain_origins[
        terrain.terrain_levels[selected_ids], terrain.terrain_types[selected_ids]
    ]


def stair_route_cues(
    env: ManagerBasedRlEnv,
    stair_start_distance: float,
    goal_distance: float,
    min_riser_height: float,
    max_riser_height: float,
    num_terrain_levels: int,
    tread_depth: float,
    num_steps: int,
    head_sensor_name: str = "head_ground_contact",
    trunk_sensor_name: str = "trunk_ground_contact",
    cue_distance: float = 0.18,
    cue_width: float = 0.025,
    include_time_to_go: bool = False,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Six real-sensor-compatible route cues in the dormant body-command slot.

    The physical robot can derive the first-riser distance and obstacle height
    from its head ToF/camera. Spatial cues stay near zero on the runway to
    preserve the manufacturer gait. Finite-horizon specialists may replace the
    redundant route-progress slot with a raw time-to-go countdown.
    """
    x, _, _ = _stair_local_state(env, asset_cfg)
    passed_faces = torch.where(
        x >= stair_start_distance,
        torch.floor((x - stair_start_distance) / max(tread_depth, 1e-6)).long()
        + 1,
        torch.zeros_like(x, dtype=torch.long),
    )
    next_face_index = torch.clamp(passed_faces, min=0, max=max(num_steps - 1, 0))
    next_face = stair_start_distance + next_face_index.to(x.dtype) * tread_depth
    distance = next_face - x
    mode = torch.sigmoid((cue_distance - distance) / max(cue_width, 1e-6))
    signed_distance = torch.clamp(
        distance / max(cue_distance, 1e-6), min=-1.0, max=1.0
    )
    route_progress = torch.clamp(x / max(goal_distance, 1e-6), 0.0, 1.0)
    if include_time_to_go:
        route_or_time = torch.clamp(
            (env.max_episode_length - env.episode_length_buf.to(x.dtype))
            / max(env.max_episode_length, 1),
            min=0.0,
            max=1.0,
        )
    else:
        route_or_time = route_progress
    levels = env.scene.terrain.terrain_levels.to(dtype=x.dtype)
    level_fraction = torch.clamp(
        levels / max(num_terrain_levels - 1, 1), 0.0, 1.0
    )
    riser_height = min_riser_height + level_fraction * (
        max_riser_height - min_riser_height
    )
    height_fraction = riser_height / max(max_riser_height, 1e-6)
    def contact_bit(sensor_name: str) -> torch.Tensor:
        if sensor_name not in env.scene.sensors:
            return torch.zeros_like(x)
        found = env.scene.sensors[sensor_name].data.found
        return (found.view(found.shape[0], -1) > 0).any(dim=-1).to(x.dtype)

    return torch.stack(
        (
            mode,
            mode * signed_distance,
            mode * height_fraction,
            route_or_time if include_time_to_go else mode * route_or_time,
            contact_bit(head_sensor_name),
            contact_bit(trunk_sensor_name),
        ),
        dim=-1,
    )


def stair_first_riser_clearance(
    env: ManagerBasedRlEnv,
    stair_start_distance: float,
    min_riser_height: float,
    max_riser_height: float,
    num_terrain_levels: int,
    corridor_half_width: float,
    riser_height: float | None = None,
    x_margin: float = 0.04,
    z_margin: float = 0.025,
    max_vertical_speed: float = 0.35,
    hold_time_s: float = 0.15,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Pay once for durable physical clearance of the first riser.

    No posture or contact type is prescribed. A jump, roll, shell mantle, or
    head-supported lever all qualify if the root crosses the lip, stays above
    it with bounded vertical speed, and remains inside the stair corridor.
    """
    asset: Entity = env.scene[asset_cfg.name]
    x, z, _ = _stair_local_state(env, asset_cfg)
    if riser_height is None:
        levels = env.scene.terrain.terrain_levels.to(dtype=x.dtype)
        level_fraction = torch.clamp(
            levels / max(num_terrain_levels - 1, 1), min=0.0, max=1.0
        )
        riser = min_riser_height + level_fraction * (
            max_riser_height - min_riser_height
        )
    else:
        riser = torch.full_like(x, riser_height)
    y = torch.abs(
        asset.data.root_link_pos_w[:, 1] - env.scene.terrain.env_origins[:, 1]
    )
    vertical_speed = torch.abs(
        torch.nan_to_num(asset.data.root_link_lin_vel_w[:, 2], nan=torch.inf)
    )
    candidate = (
        (x >= stair_start_distance + x_margin)
        & (z >= riser + z_margin)
        & (y <= corridor_half_width)
        & (vertical_speed <= max_vertical_speed)
    )
    skip_clearance = getattr(env, "_stair_skip_first_clearance", None)
    if skip_clearance is not None:
        candidate &= ~skip_clearance
    if not hasattr(env, "_stair_first_riser_hold_steps"):
        env._stair_first_riser_hold_steps = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
    if not hasattr(env, "_stair_first_riser_latched"):
        env._stair_first_riser_latched = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
    fresh = env.episode_length_buf <= 1
    env._stair_first_riser_hold_steps[fresh] = 0
    env._stair_first_riser_latched[fresh] = False
    env._stair_first_riser_hold_steps = torch.where(
        candidate,
        env._stair_first_riser_hold_steps + 1,
        torch.zeros_like(env._stair_first_riser_hold_steps),
    )
    required_steps = max(1, int(round(hold_time_s / env.step_dt)))
    newly_cleared = (
        env._stair_first_riser_hold_steps >= required_steps
    ) & ~env._stair_first_riser_latched
    env._stair_first_riser_latched |= newly_cleared
    return newly_cleared.float()


def stair_first_riser_frontier(
    env: ManagerBasedRlEnv,
    stair_start_distance: float,
    standing_root_height: float,
    riser_height: float,
    approach_distance: float = 0.18,
    preload_distance: float = 0.08,
    x_margin: float = 0.04,
    z_margin: float = 0.025,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward only new episode-best progress through the first 170 mm riser.

    The three equal phases are reaching the face, lifting above the lip, and
    crossing the face while elevated. Holding or repeating an already reached
    state pays zero, so the dense discovery signal cannot be farmed.
    """
    x, z, _ = _stair_local_state(env, asset_cfg)
    approach = torch.clamp(
        (x - (stair_start_distance - approach_distance))
        / max(approach_distance - 0.02, 1e-6),
        min=0.0,
        max=1.0,
    )
    preload_gate = torch.sigmoid(
        (x - (stair_start_distance - preload_distance)) / 0.02
    )
    lip_height = riser_height + z_margin
    lift = torch.clamp(
        (z - standing_root_height)
        / max(lip_height - standing_root_height, 1e-6),
        min=0.0,
        max=1.0,
    ) * preload_gate
    crossing = torch.clamp(
        (x - (stair_start_distance - 0.02)) / max(x_margin + 0.02, 1e-6),
        min=0.0,
        max=1.0,
    ) * lift
    frontier = (approach + lift + crossing) / 3.0
    if not hasattr(env, "_stair_first_riser_frontier"):
        env._stair_first_riser_frontier = frontier.clone()
    fresh = env.episode_length_buf <= 1
    env._stair_first_riser_frontier[fresh] = frontier[fresh]
    new_best = torch.maximum(env._stair_first_riser_frontier, frontier)
    delta = new_best - env._stair_first_riser_frontier
    env._stair_first_riser_frontier = new_best
    return delta / env.step_dt


def stair_curriculum_mantle_frontier(
    env: ManagerBasedRlEnv,
    stair_start_distance: float,
    min_riser_height: float,
    max_riser_height: float,
    num_terrain_levels: int,
    standing_root_height: float,
    approach_distance: float = 0.18,
    preload_distance: float = 0.08,
    x_margin: float = 0.04,
    z_margin: float = 0.025,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Mechanism-agnostic first-riser frontier for a height curriculum."""

    x, z, _ = _stair_local_state(env, asset_cfg)
    levels = env.scene.terrain.terrain_levels.to(dtype=x.dtype)
    level_fraction = torch.clamp(
        levels / max(num_terrain_levels - 1, 1), min=0.0, max=1.0
    )
    riser_height = min_riser_height + level_fraction * (
        max_riser_height - min_riser_height
    )
    approach = torch.clamp(
        (x - (stair_start_distance - approach_distance))
        / max(approach_distance - 0.02, 1e-6),
        min=0.0,
        max=1.0,
    )
    preload_gate = torch.sigmoid(
        (x - (stair_start_distance - preload_distance)) / 0.02
    )
    lip_height = riser_height + z_margin
    lift = torch.clamp(
        (z - standing_root_height)
        / torch.clamp(lip_height - standing_root_height, min=1e-6),
        min=0.0,
        max=1.0,
    ) * preload_gate
    crossing = torch.clamp(
        (x - (stair_start_distance - 0.02)) / max(x_margin + 0.02, 1e-6),
        min=0.0,
        max=1.0,
    ) * lift
    frontier = (approach + lift + crossing) / 3.0
    return _new_frontier_delta(env, frontier, "_stair_curriculum_mantle_frontier")


def stair_curriculum_contact_mantle_frontier(
    env: ManagerBasedRlEnv,
    stair_start_distance: float,
    min_riser_height: float,
    max_riser_height: float,
    num_terrain_levels: int,
    tread_depth: float,
    corridor_half_width: float,
    x_start_margin: float = 0.005,
    x_target_margin: float = 0.040,
    z_start_margin: float = 0.005,
    z_target_margin: float = 0.025,
    support_sensor_name: str = "robot_ground_contact",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Pay only new last-mile center progress after real stair contact.

    A first-riser face or tread contact arms the potential for the remainder of
    the episode. The potential then concentrates all of its gradient in the
    final centimeters needed to put the root over the lip. Repeated bumps and
    holding an already reached pose pay zero because only a new episode-best
    potential is rewarded.
    """

    asset: Entity = env.scene[asset_cfg.name]
    x, z, _ = _stair_local_state(env, asset_cfg)
    y = torch.abs(
        asset.data.root_link_pos_w[:, 1] - env.scene.terrain.env_origins[:, 1]
    )
    levels = env.scene.terrain.terrain_levels.to(dtype=x.dtype)
    level_fraction = torch.clamp(
        levels / max(num_terrain_levels - 1, 1), min=0.0, max=1.0
    )
    riser_height = min_riser_height + level_fraction * (
        max_riser_height - min_riser_height
    )

    if support_sensor_name not in env.scene.sensors:
        raise RuntimeError(f"Missing required contact sensor: {support_sensor_name}")
    contact_data = env.scene.sensors[support_sensor_name].data
    if (
        contact_data.found is None
        or contact_data.pos is None
        or contact_data.normal is None
    ):
        raise RuntimeError(
            f"{support_sensor_name} must provide found, pos, and normal"
        )
    face, tread = classify_curriculum_stair_contacts(
        contact_data.found,
        contact_data.pos,
        contact_data.normal,
        env.scene.terrain.env_origins,
        stair_face_x=stair_start_distance,
        riser_height=riser_height,
        tread_depth=tread_depth,
        corridor_half_width=corridor_half_width,
    )
    contact = (face | tread).any(dim=-1)

    armed = getattr(env, "_stair_curriculum_contact_mantle_armed", None)
    if armed is None:
        armed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._stair_curriculum_contact_mantle_armed = armed
    fresh = env.episode_length_buf <= 1
    armed[fresh] = False
    cleared = getattr(env, "_stair_first_riser_latched", None)
    if cleared is None:
        cleared = torch.zeros_like(armed)
    newly_eligible = (
        (env.episode_length_buf >= 3)
        & (y <= corridor_half_width)
        & contact
        & ~cleared
    )
    armed |= newly_eligible

    x_progress = torch.clamp(
        (x - (stair_start_distance + x_start_margin))
        / max(x_target_margin - x_start_margin, 1e-6),
        min=0.0,
        max=1.0,
    )
    z_progress = torch.clamp(
        (z - (riser_height + z_start_margin))
        / max(z_target_margin - z_start_margin, 1e-6),
        min=0.0,
        max=1.0,
    )
    potential = 0.55 * x_progress + 0.45 * z_progress
    potential = torch.where(
        armed & (y <= corridor_half_width), potential, torch.zeros_like(potential)
    )
    return _new_frontier_delta(
        env, potential, "_stair_curriculum_contact_mantle_frontier"
    )


def stair_critic_privileged_state(
    env: ManagerBasedRlEnv,
    stair_start_distance: float,
    min_riser_height: float,
    max_riser_height: float,
    num_terrain_levels: int,
    source_episode_step_range: tuple[int, int] = (15, 60),
    corridor_half_width: float = 0.36,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Critic-only obstacle, contact, and reference-phase information."""

    asset: Entity = env.scene[asset_cfg.name]
    x, z, upright = _stair_local_state(env, asset_cfg)
    y = asset.data.root_link_pos_w[:, 1] - env.scene.terrain.env_origins[:, 1]
    levels = env.scene.terrain.terrain_levels.to(dtype=x.dtype)
    level_fraction = torch.clamp(
        levels / max(num_terrain_levels - 1, 1), min=0.0, max=1.0
    )
    riser_height = min_riser_height + level_fraction * (
        max_riser_height - min_riser_height
    )
    linear_velocity = torch.nan_to_num(asset.data.root_link_lin_vel_w, nan=0.0)
    angular_velocity = torch.nan_to_num(asset.data.root_link_ang_vel_w, nan=0.0)
    source_step = getattr(env, "_stair_walker_bank_source_step", None)
    if source_step is None:
        phase = torch.zeros_like(x)
    else:
        step_low, step_high = source_episode_step_range
        phase = torch.clamp(
            (source_step.to(x.dtype) - step_low) / max(step_high - step_low, 1),
            min=0.0,
            max=1.0,
        )

    def contact_bit(sensor_name: str) -> torch.Tensor:
        if sensor_name not in env.scene.sensors:
            return torch.zeros_like(x)
        found = env.scene.sensors[sensor_name].data.found
        return (found.view(found.shape[0], -1) > 0).any(dim=-1).to(x.dtype)

    clearance = getattr(env, "_stair_first_riser_latched", None)
    if clearance is None:
        clearance_bit = torch.zeros_like(x)
    else:
        clearance_bit = clearance.to(x.dtype)
    return torch.stack(
        (
            torch.clamp((stair_start_distance - x) / 0.25, -1.0, 1.0),
            torch.clamp(y / max(corridor_half_width, 1e-6), -1.0, 1.0),
            torch.clamp((z - riser_height) / max_riser_height, -1.0, 1.0),
            upright,
            torch.clamp(linear_velocity[:, 0], -2.0, 2.0),
            torch.clamp(linear_velocity[:, 2], -2.0, 2.0),
            torch.clamp(angular_velocity[:, 1] / 8.0, -2.0, 2.0),
            level_fraction,
            phase,
            contact_bit("head_ground_contact"),
            contact_bit("trunk_ground_contact"),
            contact_bit("legs_ground_contact"),
            contact_bit("feet_stair_contact"),
            clearance_bit,
        ),
        dim=-1,
    )


def reset_route_learning_states(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    stair_start_distance: float,
    min_riser_height: float,
    max_riser_height: float,
    num_terrain_levels: int,
    tread_depth: float,
    num_steps: int,
    standing_root_height: float,
    near_face_fraction: float = 0.25,
    partial_mantle_fraction: float = 0.25,
    on_tread_fraction: float = 0.15,
    min_tread_step: int = 1,
    max_tread_step: int | None = None,
) -> None:
    """Reverse curriculum for discovering contact and recovery subskills.

    Normal runway starts remain the majority. Other episodes begin immediately
    before the face or upright on a random tread, allowing PPO to learn the
    obstacle reaction and later landing/recovery phases before it can execute
    the entire sequence from the approach.
    """
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device, dtype=torch.long)
    if not hasattr(env, "_stair_skip_first_clearance"):
        env._stair_skip_first_clearance = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
    env._stair_skip_first_clearance[env_ids] = False
    terrain = env.scene.terrain
    levels = terrain.terrain_levels[env_ids].to(dtype=torch.float32)
    level_fraction = torch.clamp(
        levels / max(num_terrain_levels - 1, 1), 0.0, 1.0
    )
    riser = min_riser_height + level_fraction * (
        max_riser_height - min_riser_height
    )
    draw = torch.rand(len(env_ids), device=env.device)
    near_mask = draw < near_face_fraction
    mantle_mask = (draw >= near_face_fraction) & (
        draw < near_face_fraction + partial_mantle_fraction
    )
    tread_mask = (draw >= near_face_fraction + partial_mantle_fraction) & (
        draw
        < near_face_fraction + partial_mantle_fraction + on_tread_fraction
    )

    near_ids = env_ids[near_mask]
    if len(near_ids) > 0:
        jitter = (torch.rand(len(near_ids), device=env.device) - 0.5) * 0.04
        env.sim.data.qpos[near_ids, 0] = (
            terrain.env_origins[near_ids, 0]
            + stair_start_distance
            - 0.07
            + jitter
        )
        env.sim.data.qpos[near_ids, 1] = terrain.env_origins[near_ids, 1]
        env.sim.data.qvel[near_ids, :6] = 0.0
        env.sim.data.qvel[near_ids, 0] = 0.10 + 0.20 * torch.rand(
            len(near_ids), device=env.device
        )
        pitch = torch.deg2rad(
            10.0 + 15.0 * torch.rand(len(near_ids), device=env.device)
        )
        env.sim.data.qpos[near_ids, 3] = torch.cos(0.5 * pitch)
        env.sim.data.qpos[near_ids, 4] = 0.0
        env.sim.data.qpos[near_ids, 5] = torch.sin(0.5 * pitch)
        env.sim.data.qpos[near_ids, 6] = 0.0

    mantle_ids = env_ids[mantle_mask]
    if len(mantle_ids) > 0:
        # Begin just before the clearance boundary, pitched so head or shell
        # contact can finish the transition. The policy must still move the
        # root forward and hold above the lip, so reset itself earns nothing.
        env.sim.data.qpos[mantle_ids, 0] = (
            terrain.env_origins[mantle_ids, 0]
            + stair_start_distance
            - 0.015
            + (torch.rand(len(mantle_ids), device=env.device) - 0.5) * 0.02
        )
        env.sim.data.qpos[mantle_ids, 1] = terrain.env_origins[mantle_ids, 1]
        env.sim.data.qpos[mantle_ids, 2] = (
            terrain.env_origins[mantle_ids, 2]
            + riser[mantle_mask]
            + 0.060
        )
        pitch = torch.deg2rad(
            55.0 + 25.0 * torch.rand(len(mantle_ids), device=env.device)
        )
        env.sim.data.qpos[mantle_ids, 3] = torch.cos(0.5 * pitch)
        env.sim.data.qpos[mantle_ids, 4] = 0.0
        env.sim.data.qpos[mantle_ids, 5] = torch.sin(0.5 * pitch)
        env.sim.data.qpos[mantle_ids, 6] = 0.0
        env.sim.data.qvel[mantle_ids, :6] = 0.0
        env.sim.data.qvel[mantle_ids, 0] = 0.08 * torch.rand(
            len(mantle_ids), device=env.device
        )

    tread_ids = env_ids[tread_mask]
    if len(tread_ids) > 0:
        # These episodes teach tread recovery, not first-riser traversal. They
        # must never claim the clearance latch merely because reset placed the
        # robot beyond the first face.
        env._stair_skip_first_clearance[tread_ids] = True
        upper_tread_step = num_steps if max_tread_step is None else max_tread_step
        if not 1 <= min_tread_step <= upper_tread_step <= num_steps:
            raise ValueError("Tread reset range must stay within the staircase.")
        step_index = torch.randint(
            min_tread_step,
            upper_tread_step + 1,
            (len(tread_ids),),
            device=env.device,
        )
        tread_center = stair_start_distance + (step_index.float() - 0.5) * tread_depth
        env.sim.data.qpos[tread_ids, 0] = (
            terrain.env_origins[tread_ids, 0] + tread_center
        )
        env.sim.data.qpos[tread_ids, 1] = terrain.env_origins[tread_ids, 1]
        env.sim.data.qpos[tread_ids, 2] = (
            terrain.env_origins[tread_ids, 2]
            + standing_root_height
            + step_index.float() * riser[tread_mask]
        )
        env.sim.data.qvel[tread_ids, :6] = 0.0


def _seed_action_history_from_joint_pose(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """Keep the 61D previous-action observation consistent with a reset pose."""
    if len(env_ids) == 0:
        return
    asset: Entity = env.scene[asset_cfg.name]
    action_term = env.action_manager.get_term("joint_pos")
    joint_pos = asset.data.joint_pos[env_ids][:, action_term.target_ids]
    offset = action_term.offset
    scale = action_term.scale
    offset_value = offset[env_ids] if isinstance(offset, torch.Tensor) else offset
    scale_value = scale[env_ids] if isinstance(scale, torch.Tensor) else scale
    raw_action = (joint_pos - offset_value) / scale_value
    manager = env.action_manager
    if not hasattr(env, "_stair_reset_action_history"):
        env._stair_reset_action_history = {
            "current": torch.zeros_like(manager._action),
            "previous": torch.zeros_like(manager._prev_action),
            "previous_previous": torch.zeros_like(manager._prev_prev_action),
        }
    history = env._stair_reset_action_history
    history["current"][env_ids] = raw_action
    history["previous"][env_ids] = raw_action
    history["previous_previous"][env_ids] = raw_action

    # These writes make the state immediately coherent inside the reset event.
    # StairHistoryJointPositionAction repeats them after ActionManager.reset,
    # which otherwise zeros all three buffers later in mjlab's reset sequence.
    manager._action[env_ids] = raw_action
    manager._prev_action[env_ids] = raw_action
    manager._prev_prev_action[env_ids] = raw_action
    action_term._raw_actions[env_ids] = raw_action
    action_term._processed_actions[env_ids] = joint_pos


def reset_assisted_stair_states(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    stair_start_distance: float,
    standing_root_height: float,
    lip_release_fraction: float = 0.50,
    shell_brace_fraction: float = 0.25,
    tread_recovery_fraction: float = 0.15,
    real_handoff_fraction: float = 0.10,
    lip_local_x_range: tuple[float, float] | None = None,
    lip_root_height_range: tuple[float, float] | None = None,
    lip_pitch_deg_range: tuple[float, float] | None = None,
    lip_forward_speed_range: tuple[float, float] | None = None,
    lip_vertical_speed_range: tuple[float, float] | None = None,
    lip_pitch_rate_range: tuple[float, float] | None = None,
    shell_local_x_range: tuple[float, float] | None = None,
    shell_root_height_range: tuple[float, float] | None = None,
    shell_pitch_deg_range: tuple[float, float] | None = None,
    shell_forward_speed_range: tuple[float, float] | None = None,
    shell_vertical_speed_range: tuple[float, float] | None = None,
    shell_pitch_rate_range: tuple[float, float] | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """Stage-A reverse curriculum on one fixed 170 mm home stair.

    These are physically demanding full-height contact states, not shorter
    geometry. Action history is reconstructed from the reset joint pose so the
    actor does not receive the all-zero history mismatch from the stalled run.
    """
    if env_ids is None or len(env_ids) == 0:
        return
    fractions = (
        lip_release_fraction,
        shell_brace_fraction,
        tread_recovery_fraction,
        real_handoff_fraction,
    )
    if any(fraction < 0.0 for fraction in fractions) or not math.isclose(
        sum(fractions), 1.0, abs_tol=1e-6
    ):
        raise ValueError(
            "Assisted stair reset fractions must be nonnegative and sum to 1."
        )

    env_ids = env_ids.to(env.device, dtype=torch.long)
    terrain = env.scene.terrain
    origins = terrain.env_origins
    draw = torch.rand(len(env_ids), device=env.device)
    lip_end = lip_release_fraction
    shell_end = lip_end + shell_brace_fraction
    tread_end = shell_end + tread_recovery_fraction
    lip_ids = env_ids[draw < lip_end]
    shell_ids = env_ids[(draw >= lip_end) & (draw < shell_end)]
    tread_ids = env_ids[(draw >= shell_end) & (draw < tread_end)]
    handoff_ids = env_ids[draw >= tread_end]

    # Keep the sampled reset family available to evaluators.  Aggregate
    # clearance rates are otherwise misleading because tread-recovery starts
    # are intentionally ineligible for the first-riser clearance latch.
    if not hasattr(env, "_stair_assisted_reset_mode"):
        env._stair_assisted_reset_mode = torch.full(
            (env.num_envs,), -1, dtype=torch.long, device=env.device
        )
    env._stair_assisted_reset_mode[lip_ids] = 0
    env._stair_assisted_reset_mode[shell_ids] = 1
    env._stair_assisted_reset_mode[tread_ids] = 2
    env._stair_assisted_reset_mode[handoff_ids] = 3

    if not hasattr(env, "_stair_skip_first_clearance"):
        env._stair_skip_first_clearance = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
    env._stair_skip_first_clearance[env_ids] = False
    env._stair_skip_first_clearance[tread_ids] = True

    def sample(count: int, low: float, high: float) -> torch.Tensor:
        return low + (high - low) * torch.rand(count, device=env.device)

    def resolved_range(
        value: tuple[float, float] | None, default: tuple[float, float]
    ) -> tuple[float, float]:
        result = default if value is None else value
        if result[0] > result[1]:
            raise ValueError(f"Invalid assisted stair reset range: {result}")
        return result

    def set_pitch(ids: torch.Tensor, low_deg: float, high_deg: float) -> None:
        pitch = torch.deg2rad(sample(len(ids), low_deg, high_deg))
        env.sim.data.qpos[ids, 3] = torch.cos(0.5 * pitch)
        env.sim.data.qpos[ids, 4] = 0.0
        env.sim.data.qpos[ids, 5] = torch.sin(0.5 * pitch)
        env.sim.data.qpos[ids, 6] = 0.0

    if len(lip_ids) > 0:
        local_x = resolved_range(
            lip_local_x_range,
            (stair_start_distance + 0.020, stair_start_distance + 0.037),
        )
        root_height = resolved_range(lip_root_height_range, (0.198, 0.225))
        pitch = resolved_range(lip_pitch_deg_range, (30.0, 55.0))
        forward_speed = resolved_range(lip_forward_speed_range, (0.18, 0.30))
        vertical_speed = resolved_range(lip_vertical_speed_range, (-0.03, 0.08))
        pitch_rate = resolved_range(lip_pitch_rate_range, (-1.5, -0.3))
        env.sim.data.qpos[lip_ids, 0] = origins[lip_ids, 0] + sample(
            len(lip_ids), *local_x
        )
        env.sim.data.qpos[lip_ids, 1] = origins[lip_ids, 1]
        env.sim.data.qpos[lip_ids, 2] = origins[lip_ids, 2] + sample(
            len(lip_ids), *root_height
        )
        set_pitch(lip_ids, *pitch)
        env.sim.data.qvel[lip_ids, :6] = 0.0
        env.sim.data.qvel[lip_ids, 0] = sample(len(lip_ids), *forward_speed)
        env.sim.data.qvel[lip_ids, 2] = sample(len(lip_ids), *vertical_speed)
        env.sim.data.qvel[lip_ids, 4] = sample(len(lip_ids), *pitch_rate)

    if len(shell_ids) > 0:
        local_x = resolved_range(
            shell_local_x_range,
            (stair_start_distance - 0.020, stair_start_distance + 0.015),
        )
        root_height = resolved_range(shell_root_height_range, (0.165, 0.205))
        pitch = resolved_range(shell_pitch_deg_range, (45.0, 70.0))
        forward_speed = resolved_range(shell_forward_speed_range, (0.10, 0.22))
        vertical_speed = resolved_range(shell_vertical_speed_range, (0.08, 0.20))
        pitch_rate = resolved_range(shell_pitch_rate_range, (0.5, 2.0))
        env.sim.data.qpos[shell_ids, 0] = origins[shell_ids, 0] + sample(
            len(shell_ids), *local_x
        )
        env.sim.data.qpos[shell_ids, 1] = origins[shell_ids, 1]
        env.sim.data.qpos[shell_ids, 2] = origins[shell_ids, 2] + sample(
            len(shell_ids), *root_height
        )
        set_pitch(shell_ids, *pitch)
        env.sim.data.qvel[shell_ids, :6] = 0.0
        env.sim.data.qvel[shell_ids, 0] = sample(len(shell_ids), *forward_speed)
        env.sim.data.qvel[shell_ids, 2] = sample(len(shell_ids), *vertical_speed)
        env.sim.data.qvel[shell_ids, 4] = sample(len(shell_ids), *pitch_rate)

    if len(tread_ids) > 0:
        env.sim.data.qpos[tread_ids, 0] = origins[tread_ids, 0] + sample(
            len(tread_ids), stair_start_distance + 0.080, stair_start_distance + 0.200
        )
        env.sim.data.qpos[tread_ids, 1] = origins[tread_ids, 1]
        env.sim.data.qpos[tread_ids, 2] = origins[tread_ids, 2] + sample(
            len(tread_ids), 0.260, 0.310
        )
        set_pitch(tread_ids, 35.0, 55.0)
        env.sim.data.qvel[tread_ids, :6] = 0.0
        env.sim.data.qvel[tread_ids, 0] = sample(len(tread_ids), 0.03, 0.12)
        env.sim.data.qvel[tread_ids, 2] = sample(len(tread_ids), -0.08, 0.02)
        env.sim.data.qvel[tread_ids, 4] = sample(len(tread_ids), -1.2, -0.2)

    if len(handoff_ids) > 0:
        env.sim.data.qpos[handoff_ids, 0] = origins[handoff_ids, 0] + sample(
            len(handoff_ids), stair_start_distance - 0.120, stair_start_distance - 0.080
        )
        env.sim.data.qpos[handoff_ids, 1] = origins[handoff_ids, 1]
        env.sim.data.qpos[handoff_ids, 2] = (
            origins[handoff_ids, 2] + standing_root_height
        )
        set_pitch(handoff_ids, -12.0, 12.0)
        env.sim.data.qvel[handoff_ids, :6] = 0.0
        env.sim.data.qvel[handoff_ids, 0] = sample(len(handoff_ids), 0.14, 0.30)
        env.sim.data.qvel[handoff_ids, 2] = sample(len(handoff_ids), -0.10, 0.10)
        env.sim.data.qvel[handoff_ids, 4] = sample(len(handoff_ids), -2.0, 2.0)

    _seed_action_history_from_joint_pose(env, env_ids, asset_cfg)


def _new_frontier_delta(
    env: ManagerBasedRlEnv, value: torch.Tensor, state_name: str
) -> torch.Tensor:
    state = getattr(env, state_name, None)
    if state is None:
        state = value.clone()
        setattr(env, state_name, state)
    fresh = env.episode_length_buf <= 1
    state[fresh] = value[fresh]
    new_best = torch.maximum(state, value)
    delta = new_best - state
    setattr(env, state_name, new_best)
    return delta / env.step_dt


def assign_stair_state_bank_family(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    family_weights: tuple[int, ...] = (2, 1, 1),
    forced_family: int | None = None,
) -> None:
    """Assign one mutually exclusive replay family to every resetting env.

    Full reset batches receive exact integer quotas. Smaller asynchronous
    batches use the same largest-remainder allocation with randomized ties, so
    no state-bank event can overwrite a state written by another family.
    """

    if env_ids is None or len(env_ids) == 0:
        return
    if len(family_weights) == 0 or any(weight <= 0 for weight in family_weights):
        raise ValueError("State-bank family weights must be positive")
    if forced_family is not None and not 0 <= forced_family < len(family_weights):
        raise ValueError("Forced state-bank family is outside the configured range")
    env_ids = env_ids.to(device=env.device, dtype=torch.long)
    if not hasattr(env, "_stair_state_bank_family"):
        env._stair_state_bank_family = torch.full(
            (env.num_envs,), -1, dtype=torch.long, device=env.device
        )
    if forced_family is not None:
        env._stair_state_bank_family[env_ids] = forced_family
        return
    weights = torch.tensor(family_weights, device=env.device, dtype=torch.float32)
    exact = weights * (len(env_ids) / float(sum(family_weights)))
    counts = torch.floor(exact).to(torch.long)
    remainder = len(env_ids) - int(counts.sum().item())
    if remainder > 0:
        fractions = exact - counts.to(exact.dtype)
        positive = fractions > 0
        if not torch.any(positive):
            fractions = weights / weights.sum()
        else:
            fractions = torch.where(positive, fractions, torch.zeros_like(fractions))
        extra = torch.multinomial(fractions, remainder, replacement=False)
        counts[extra] += 1

    permutation = env_ids[torch.randperm(len(env_ids), device=env.device)]
    start = 0
    for family, count in enumerate(counts.tolist()):
        stop = start + count
        env._stair_state_bank_family[permutation[start:stop]] = family
        start = stop


def seed_stair_contact_transfer_reverse_context(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    reverse_family: int = 1,
    stage15_families: tuple[int, ...] | None = None,
    stage2_family: int | None = None,
    stage2_families: tuple[int, ...] | None = None,
) -> None:
    """Mark policy-generated replay rows as proven contact prefixes.

    The reset itself never pays a stage reward. Reward terms use this marker
    only to prequalify already-observed history so PPO can learn the next
    missing transition from that exact state.
    """

    if env_ids is None or len(env_ids) == 0:
        return
    family = getattr(env, "_stair_state_bank_family", None)
    if family is None:
        raise RuntimeError(
            "Reverse contact context requires preceding state-bank assignment"
        )
    env_ids = env_ids.to(device=env.device, dtype=torch.long)
    if not hasattr(env, "_stair_contact_transfer_reverse_stage15_seed"):
        env._stair_contact_transfer_reverse_stage15_seed = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
    if not hasattr(env, "_stair_contact_transfer_reverse_stage2_seed"):
        env._stair_contact_transfer_reverse_stage2_seed = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
    if stage15_families is None:
        stage15_families = (reverse_family,)
    if len(stage15_families) == 0:
        raise ValueError("At least one Stage 1.5 reverse family is required")
    if stage2_family is not None and stage2_families is not None:
        raise ValueError("Use either stage2_family or stage2_families, not both")
    if stage2_families is None:
        stage2_families = () if stage2_family is None else (stage2_family,)
    stage15_seed = torch.zeros(len(env_ids), dtype=torch.bool, device=env.device)
    for family_id in stage15_families:
        stage15_seed |= family[env_ids] == family_id
    stage2_seed = torch.zeros(len(env_ids), dtype=torch.bool, device=env.device)
    for family_id in stage2_families:
        stage2_seed |= family[env_ids] == family_id
    env._stair_contact_transfer_reverse_stage15_seed[env_ids] = stage15_seed
    env._stair_contact_transfer_reverse_stage2_seed[env_ids] = stage2_seed


def stair_preload_frontier(
    env: ManagerBasedRlEnv,
    start_x: float = 0.42,
    end_x: float = 0.62,
    standing_root_height: float = 0.115,
    target_root_height: float = 0.095,
    upright_floor: float = 0.55,
    corridor_half_width: float = 0.36,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward one episode-best crouch while approaching the first riser.

    The preload is deliberately a frontier instead of a per-step pose reward:
    reaching a deeper crouch can pay once, but camping there cannot. The route,
    corridor, and upright gates prevent lying down away from the stair from
    becoming a substitute for loading the legs.
    """
    asset: Entity = env.scene[asset_cfg.name]
    x, z, upright = _stair_local_state(env, asset_cfg)
    y = torch.abs(
        asset.data.root_link_pos_w[:, 1] - env.scene.terrain.env_origins[:, 1]
    )
    approach_gate = torch.clamp(
        (x - start_x) / max(end_x - start_x, 1e-6), min=0.0, max=1.0
    )
    crouch = torch.clamp(
        (standing_root_height - z)
        / max(standing_root_height - target_root_height, 1e-6),
        min=0.0,
        max=1.0,
    )
    upright_gate = torch.clamp(
        (upright - upright_floor) / max(1.0 - upright_floor, 1e-6),
        min=0.0,
        max=1.0,
    )
    value = approach_gate * crouch * upright_gate
    value = torch.where(y <= corridor_half_width, value, torch.zeros_like(value))
    return _new_frontier_delta(env, value, "_stair_preload_frontier")


def stair_launch_sequence(
    env: ManagerBasedRlEnv,
    min_x: float = 0.46,
    max_x: float = 0.68,
    preload_root_height: float = 0.098,
    min_upward_speed: float = 0.30,
    min_forward_speed: float = 0.05,
    preload_max_vertical_speed: float = 0.30,
    upright_threshold: float = 0.55,
    corridor_half_width: float = 0.36,
    preload_hold_time_s: float = 0.04,
    launch_bonus: float = 4.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Pay once for a grounded preload followed by a forward/upward launch.

    This is a minimal phase variable, not a prescribed jump trajectory. After
    the compression gate is reached, the actor remains free to jump, roll,
    shell-mantle, or plant its head. The later physical clearance and support
    latches still decide whether the attempt actually climbed the stair.
    """
    asset: Entity = env.scene[asset_cfg.name]
    x, z, upright = _stair_local_state(env, asset_cfg)
    y = torch.abs(
        asset.data.root_link_pos_w[:, 1] - env.scene.terrain.env_origins[:, 1]
    )
    velocity = torch.nan_to_num(asset.data.root_link_lin_vel_w, nan=0.0)
    if not hasattr(env, "_stair_launch_phase"):
        env._stair_launch_phase = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
        env._stair_preload_hold_steps = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
    fresh = env.episode_length_buf <= 1
    env._stair_launch_phase[fresh] = 0
    env._stair_preload_hold_steps[fresh] = 0

    in_launch_zone = (
        (x >= min_x) & (x <= max_x) & (y <= corridor_half_width)
    )
    preload_candidate = (
        (env._stair_launch_phase == 0)
        & in_launch_zone
        & (z <= preload_root_height)
        & (upright >= upright_threshold)
        & (torch.abs(velocity[:, 2]) <= preload_max_vertical_speed)
    )
    env._stair_preload_hold_steps = torch.where(
        preload_candidate,
        env._stair_preload_hold_steps + 1,
        torch.zeros_like(env._stair_preload_hold_steps),
    )
    required_steps = max(1, int(round(preload_hold_time_s / env.step_dt)))
    newly_preloaded = (
        (env._stair_launch_phase == 0)
        & (env._stair_preload_hold_steps >= required_steps)
    )
    env._stair_launch_phase[newly_preloaded] = 1

    newly_launched = (
        (env._stair_launch_phase == 1)
        & in_launch_zone
        & (velocity[:, 0] >= min_forward_speed)
        & (velocity[:, 2] >= min_upward_speed)
    )
    env._stair_launch_phase[newly_launched] = 2
    return newly_preloaded.float() + launch_bonus * newly_launched.float()


def stair_takeoff_frontier(
    env: ManagerBasedRlEnv,
    min_x: float = 0.46,
    max_x: float = 0.68,
    target_vertical_speed: float = 1.20,
    corridor_half_width: float = 0.36,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward new upward-speed records only after the preload gate."""
    asset: Entity = env.scene[asset_cfg.name]
    x, _, _ = _stair_local_state(env, asset_cfg)
    y = torch.abs(
        asset.data.root_link_pos_w[:, 1] - env.scene.terrain.env_origins[:, 1]
    )
    phase = getattr(env, "_stair_launch_phase", None)
    if phase is None:
        return torch.zeros(env.num_envs, device=env.device)
    upward_speed = torch.clamp(
        torch.nan_to_num(asset.data.root_link_lin_vel_w[:, 2], nan=0.0)
        / max(target_vertical_speed, 1e-6),
        min=0.0,
        max=1.0,
    )
    eligible = (
        (phase >= 1)
        & (x >= min_x)
        & (x <= max_x)
        & (y <= corridor_half_width)
    )
    value = torch.where(eligible, upward_speed, torch.zeros_like(upward_speed))
    return _new_frontier_delta(env, value, "_stair_takeoff_frontier")


def classify_standard_stair_contacts(
    found: torch.Tensor,
    positions_w: torch.Tensor,
    normals_w: torch.Tensor,
    terrain_origins_w: torch.Tensor,
    *,
    stair_face_x: float = 0.66,
    riser_height: float = 0.17,
    tread_depth: float = 0.28,
    corridor_half_width: float = 0.36,
    position_tolerance: float = 0.018,
    normal_alignment: float = 0.70,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Classify contact slots on the first riser face and first tread top."""

    if found.ndim != 2 or positions_w.shape != normals_w.shape:
        raise ValueError("Stair contact tensors have incompatible shapes")
    if positions_w.shape[:2] != found.shape or positions_w.shape[-1] != 3:
        raise ValueError("Stair contact positions must have shape [B, N, 3]")
    local_pos = positions_w - terrain_origins_w[:, None, :]
    active = found > 0
    in_corridor = torch.abs(local_pos[..., 1]) <= corridor_half_width
    normal = torch.abs(torch.nan_to_num(normals_w, nan=0.0))
    face = (
        active
        & in_corridor
        & (torch.abs(local_pos[..., 0] - stair_face_x) <= position_tolerance)
        & (local_pos[..., 2] >= -position_tolerance)
        & (local_pos[..., 2] <= riser_height + position_tolerance)
        & (normal[..., 0] >= normal_alignment)
    )
    tread = (
        active
        & in_corridor
        & (local_pos[..., 0] >= stair_face_x - position_tolerance)
        & (
            local_pos[..., 0]
            <= stair_face_x + tread_depth + position_tolerance
        )
        & (torch.abs(local_pos[..., 2] - riser_height) <= position_tolerance)
        & (normal[..., 2] >= normal_alignment)
    )
    return face, tread


def classify_virtual_lip_stair_contacts(
    found: torch.Tensor,
    positions_w: torch.Tensor,
    normals_w: torch.Tensor,
    terrain_origins_w: torch.Tensor,
    physical_face_x: torch.Tensor,
    *,
    nominal_stair_face_x: float = 0.66,
    riser_height: float = 0.17,
    tread_depth: float = 0.28,
    corridor_half_width: float = 0.36,
    position_tolerance: float = 0.018,
    normal_alignment: float = 0.70,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Classify a recessed physical face and the nominal horizontal tread."""

    if found.ndim != 2 or positions_w.shape != normals_w.shape:
        raise ValueError("Stair contact tensors have incompatible shapes")
    if positions_w.shape[:2] != found.shape or positions_w.shape[-1] != 3:
        raise ValueError("Stair contact positions must have shape [B, N, 3]")
    if physical_face_x.shape != (found.shape[0],):
        raise ValueError("physical_face_x must have shape [B]")
    local_pos = positions_w - terrain_origins_w[:, None, :]
    active = found > 0
    in_corridor = torch.abs(local_pos[..., 1]) <= corridor_half_width
    normal = torch.abs(torch.nan_to_num(normals_w, nan=0.0))
    face = (
        active
        & in_corridor
        & (
            torch.abs(local_pos[..., 0] - physical_face_x[:, None])
            <= position_tolerance
        )
        & (local_pos[..., 2] >= -position_tolerance)
        & (local_pos[..., 2] <= riser_height + position_tolerance)
        & (normal[..., 0] >= normal_alignment)
    )
    tread = (
        active
        & in_corridor
        & (local_pos[..., 0] >= nominal_stair_face_x - position_tolerance)
        & (
            local_pos[..., 0]
            <= nominal_stair_face_x + tread_depth + position_tolerance
        )
        & (torch.abs(local_pos[..., 2] - riser_height) <= position_tolerance)
        & (normal[..., 2] >= normal_alignment)
    )
    return face, tread


def classify_curriculum_stair_contacts(
    found: torch.Tensor,
    positions_w: torch.Tensor,
    normals_w: torch.Tensor,
    terrain_origins_w: torch.Tensor,
    *,
    stair_face_x: float,
    riser_height: torch.Tensor,
    tread_depth: float,
    corridor_half_width: float,
    position_tolerance: float = 0.018,
    normal_alignment: float = 0.70,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Classify first-riser contacts with a per-environment riser height."""

    if found.ndim != 2 or positions_w.shape != normals_w.shape:
        raise ValueError("Stair contact tensors have incompatible shapes")
    if positions_w.shape[:2] != found.shape or positions_w.shape[-1] != 3:
        raise ValueError("Stair contact positions must have shape [B, N, 3]")
    if riser_height.shape != (found.shape[0],):
        raise ValueError("riser_height must have shape [B]")
    local_pos = positions_w - terrain_origins_w[:, None, :]
    active = found > 0
    in_corridor = torch.abs(local_pos[..., 1]) <= corridor_half_width
    normal = torch.abs(torch.nan_to_num(normals_w, nan=0.0))
    riser = riser_height[:, None]
    face = (
        active
        & in_corridor
        & (torch.abs(local_pos[..., 0] - stair_face_x) <= position_tolerance)
        & (local_pos[..., 2] >= -position_tolerance)
        & (local_pos[..., 2] <= riser + position_tolerance)
        & (normal[..., 0] >= normal_alignment)
    )
    tread = (
        active
        & in_corridor
        & (local_pos[..., 0] >= stair_face_x - position_tolerance)
        & (local_pos[..., 0] <= stair_face_x + tread_depth + position_tolerance)
        & (torch.abs(local_pos[..., 2] - riser) <= position_tolerance)
        & (normal[..., 2] >= normal_alignment)
    )
    return face, tread


def _standard_stair_contact_masks(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    *,
    stair_face_x: float,
    riser_height: float,
    tread_depth: float,
    corridor_half_width: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if sensor_name not in env.scene.sensors:
        empty = torch.zeros((env.num_envs, 1), dtype=torch.bool, device=env.device)
        positions = torch.zeros((env.num_envs, 1, 3), device=env.device)
        return empty, empty, positions
    data = env.scene.sensors[sensor_name].data
    if data.found is None or data.pos is None or data.normal is None:
        raise RuntimeError(
            f"{sensor_name} must provide found, pos, and normal contact fields"
        )
    face, tread = classify_standard_stair_contacts(
        data.found,
        data.pos,
        data.normal,
        env.scene.terrain.env_origins,
        stair_face_x=stair_face_x,
        riser_height=riser_height,
        tread_depth=tread_depth,
        corridor_half_width=corridor_half_width,
    )
    local_positions = data.pos - env.scene.terrain.env_origins[:, None, :]
    return face, tread, local_positions


def _virtual_lip_physical_face_x(
    env: ManagerBasedRlEnv,
    *,
    nominal_stair_face_x: float,
    max_face_offset: float,
    num_terrain_levels: int,
) -> torch.Tensor:
    levels = env.scene.terrain.terrain_levels.to(dtype=torch.float32)
    fraction = torch.clamp(
        levels / max(num_terrain_levels - 1, 1), min=0.0, max=1.0
    )
    return nominal_stair_face_x + max_face_offset * (1.0 - fraction)


def _virtual_lip_stair_contact_masks(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    *,
    nominal_stair_face_x: float,
    max_face_offset: float,
    num_terrain_levels: int,
    riser_height: float,
    tread_depth: float,
    corridor_half_width: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if sensor_name not in env.scene.sensors:
        empty = torch.zeros((env.num_envs, 1), dtype=torch.bool, device=env.device)
        positions = torch.zeros((env.num_envs, 1, 3), device=env.device)
        return empty, empty, positions
    data = env.scene.sensors[sensor_name].data
    if data.found is None or data.pos is None or data.normal is None:
        raise RuntimeError(
            f"{sensor_name} must provide found, pos, and normal contact fields"
        )
    physical_face_x = _virtual_lip_physical_face_x(
        env,
        nominal_stair_face_x=nominal_stair_face_x,
        max_face_offset=max_face_offset,
        num_terrain_levels=num_terrain_levels,
    )
    face, tread = classify_virtual_lip_stair_contacts(
        data.found,
        data.pos,
        data.normal,
        env.scene.terrain.env_origins,
        physical_face_x,
        nominal_stair_face_x=nominal_stair_face_x,
        riser_height=riser_height,
        tread_depth=tread_depth,
        corridor_half_width=corridor_half_width,
    )
    local_positions = data.pos - env.scene.terrain.env_origins[:, None, :]
    return face, tread, local_positions


def _virtual_lip_union_contact_state(
    env: ManagerBasedRlEnv,
    sensor_names: tuple[str, ...],
    *,
    nominal_stair_face_x: float,
    max_face_offset: float,
    num_terrain_levels: int,
    riser_height: float,
    tread_depth: float,
    corridor_half_width: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Union dedicated body sensors without losing weaker tread contacts."""

    face_contact = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    tread_contact = torch.zeros_like(face_contact)
    strongest_tread_force = torch.zeros(
        env.num_envs, dtype=torch.float32, device=env.device
    )
    for sensor_name in sensor_names:
        if sensor_name not in env.scene.sensors:
            raise RuntimeError(f"Missing required contact sensor: {sensor_name}")
        face, tread, _ = _virtual_lip_stair_contact_masks(
            env,
            sensor_name,
            nominal_stair_face_x=nominal_stair_face_x,
            max_face_offset=max_face_offset,
            num_terrain_levels=num_terrain_levels,
            riser_height=riser_height,
            tread_depth=tread_depth,
            corridor_half_width=corridor_half_width,
        )
        face_contact |= face.any(dim=-1)
        tread_contact |= tread.any(dim=-1)
        data = env.scene.sensors[sensor_name].data
        if data.force is not None:
            normal_force = torch.abs(
                torch.sum(data.force * data.normal, dim=-1)
            )
            normal_force = torch.where(
                tread, normal_force, torch.zeros_like(normal_force)
            )
            strongest_tread_force = torch.maximum(
                strongest_tread_force,
                normal_force.max(dim=-1).values,
            )
    return face_contact, tread_contact, strongest_tread_force


def _stair_contact_event(
    env: ManagerBasedRlEnv, contact: torch.Tensor, latch_name: str
) -> torch.Tensor:
    current = contact.any(dim=-1)
    latch = getattr(env, latch_name, None)
    if latch is None:
        latch = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        setattr(env, latch_name, latch)
    fresh = env.episode_length_buf <= 1
    latch[fresh] = False
    newly_contacted = current & ~latch
    latch |= current
    return newly_contacted.to(torch.float32)


def stair_riser_face_contact(
    env: ManagerBasedRlEnv,
    stair_face_x: float = 0.66,
    riser_height: float = 0.17,
    tread_depth: float = 0.28,
    corridor_half_width: float = 0.36,
    sensor_name: str = "robot_ground_contact",
) -> torch.Tensor:
    """Reward progressively higher physical contact on the first riser face."""

    face, _, positions = _standard_stair_contact_masks(
        env,
        sensor_name,
        stair_face_x=stair_face_x,
        riser_height=riser_height,
        tread_depth=tread_depth,
        corridor_half_width=corridor_half_width,
    )
    _stair_contact_event(env, face, "_stair_riser_face_contact_latched")
    face_height = torch.where(
        face,
        positions[..., 2],
        torch.zeros_like(positions[..., 2]),
    ).max(dim=-1).values
    value = torch.clamp(face_height / max(riser_height, 1e-6), 0.0, 1.0)
    return _new_frontier_delta(env, value, "_stair_riser_face_contact_frontier")


def stair_first_tread_contact(
    env: ManagerBasedRlEnv,
    stair_face_x: float = 0.66,
    riser_height: float = 0.17,
    tread_depth: float = 0.28,
    corridor_half_width: float = 0.36,
    sensor_name: str = "robot_ground_contact",
) -> torch.Tensor:
    """Pay once for actual robot contact with the horizontal first tread."""

    _, tread, _ = _standard_stair_contact_masks(
        env,
        sensor_name,
        stair_face_x=stair_face_x,
        riser_height=riser_height,
        tread_depth=tread_depth,
        corridor_half_width=corridor_half_width,
    )
    return _stair_contact_event(env, tread, "_stair_first_tread_contact_latched")


def stair_new_tread_contact_after_reset(
    env: ManagerBasedRlEnv,
    nominal_stair_face_x: float = 0.66,
    max_face_offset: float = 0.04,
    num_terrain_levels: int = 3,
    riser_height: float = 0.17,
    tread_depth: float = 0.28,
    corridor_half_width: float = 0.36,
    min_policy_steps: int = 3,
    sensor_names: tuple[str, ...] = (
        "head_ground_contact",
        "trunk_ground_contact",
        "legs_ground_contact",
        "feet_stair_contact",
    ),
) -> torch.Tensor:
    """Pay once for tread contact created after the replayed reset state."""

    if min_policy_steps < 1:
        raise ValueError("min_policy_steps must be positive")
    _, current, _ = _virtual_lip_union_contact_state(
        env,
        sensor_names,
        nominal_stair_face_x=nominal_stair_face_x,
        max_face_offset=max_face_offset,
        num_terrain_levels=num_terrain_levels,
        riser_height=riser_height,
        tread_depth=tread_depth,
        corridor_half_width=corridor_half_width,
    )
    if not hasattr(env, "_stair_contact_transfer_stage1_latched"):
        env._stair_contact_transfer_stage1_latched = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        env._stair_contact_transfer_stage1_previous = torch.zeros_like(
            env._stair_contact_transfer_stage1_latched
        )
        env._stair_contact_transfer_stage1_policy_achieved = torch.zeros_like(
            env._stair_contact_transfer_stage1_latched
        )
    fresh = env.episode_length_buf <= 1
    reverse_seed = getattr(
        env,
        "_stair_contact_transfer_reverse_stage15_seed",
        torch.zeros_like(current),
    )
    # Existing replay-bank contact is a baseline, never an achievement. Track
    # the live contact edge separately so a baseline contact that releases and
    # is later re-created by the policy can still count exactly once.
    env._stair_contact_transfer_stage1_latched[fresh] = reverse_seed[fresh]
    env._stair_contact_transfer_stage1_previous[fresh] = current[fresh]
    env._stair_contact_transfer_stage1_policy_achieved[fresh] = reverse_seed[
        fresh
    ]
    armed = env.episode_length_buf >= min_policy_steps
    newly_contacted = (
        armed
        & current
        & ~env._stair_contact_transfer_stage1_previous
        & ~env._stair_contact_transfer_stage1_latched
    )
    env._stair_contact_transfer_stage1_latched |= newly_contacted
    env._stair_contact_transfer_stage1_policy_achieved |= newly_contacted
    env._stair_contact_transfer_stage1_previous.copy_(current)
    return newly_contacted.to(torch.float32)


def stair_loaded_tread_no_face_first_frame(
    env: ManagerBasedRlEnv,
    nominal_stair_face_x: float = 0.66,
    max_face_offset: float = 0.04,
    num_terrain_levels: int = 3,
    riser_height: float = 0.17,
    tread_depth: float = 0.28,
    corridor_half_width: float = 0.36,
    min_policy_steps: int = 3,
    min_normal_force: float = 0.40,
    sensor_names: tuple[str, ...] = (
        "head_ground_contact",
        "trunk_ground_contact",
        "legs_ground_contact",
        "feet_stair_contact",
    ),
) -> torch.Tensor:
    """Pay once for the first loaded tread frame after releasing the face.

    This is the dense contact-stage bridge between the one-frame Stage 1 tread
    edge and the two-frame Stage 2 hold. Reset-born states are recorded as a
    baseline and never pay, and the separate latch prevents one-frame contact
    oscillations from farming the reward.
    """

    if min_policy_steps < 1 or min_normal_force <= 0.0:
        raise ValueError("Policy steps and tread normal force must be positive")
    face_contact, tread_contact, tread_force = _virtual_lip_union_contact_state(
        env,
        sensor_names,
        nominal_stair_face_x=nominal_stair_face_x,
        max_face_offset=max_face_offset,
        num_terrain_levels=num_terrain_levels,
        riser_height=riser_height,
        tread_depth=tread_depth,
        corridor_half_width=corridor_half_width,
    )
    stage1 = getattr(env, "_stair_contact_transfer_stage1_policy_achieved", None)
    prior_face = getattr(env, "_stair_contact_transfer_face_seen", None)
    if stage1 is None:
        stage1 = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if prior_face is None:
        prior_face = torch.zeros_like(stage1)
    candidate = (
        (env.episode_length_buf >= min_policy_steps)
        & stage1
        & prior_face
        & tread_contact
        & (tread_force >= min_normal_force)
        & ~face_contact
    )
    if not hasattr(env, "_stair_contact_transfer_stage15_latched"):
        env._stair_contact_transfer_stage15_reset_baseline = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        env._stair_contact_transfer_stage15_latched = torch.zeros_like(
            env._stair_contact_transfer_stage15_reset_baseline
        )
        env._stair_contact_transfer_stage15_policy_achieved = torch.zeros_like(
            env._stair_contact_transfer_stage15_reset_baseline
        )
    fresh = env.episode_length_buf <= 1
    reverse_seed = getattr(
        env,
        "_stair_contact_transfer_reverse_stage15_seed",
        torch.zeros_like(candidate),
    )
    env._stair_contact_transfer_stage15_reset_baseline[fresh] = candidate[fresh]
    env._stair_contact_transfer_stage15_latched[fresh] = (
        candidate[fresh] | reverse_seed[fresh]
    )
    env._stair_contact_transfer_stage15_policy_achieved[fresh] = reverse_seed[
        fresh
    ]
    newly_reached = candidate & ~env._stair_contact_transfer_stage15_latched
    env._stair_contact_transfer_stage15_latched |= candidate
    env._stair_contact_transfer_stage15_policy_achieved |= newly_reached
    return newly_reached.to(torch.float32)


def stair_loaded_tread_face_release(
    env: ManagerBasedRlEnv,
    nominal_stair_face_x: float = 0.66,
    max_face_offset: float = 0.04,
    num_terrain_levels: int = 3,
    riser_height: float = 0.17,
    tread_depth: float = 0.28,
    corridor_half_width: float = 0.36,
    hold_steps: int = 2,
    min_policy_steps: int = 3,
    min_normal_force: float = 0.40,
    require_stage15: bool = False,
    sensor_names: tuple[str, ...] = (
        "head_ground_contact",
        "trunk_ground_contact",
        "legs_ground_contact",
        "feet_stair_contact",
    ),
) -> torch.Tensor:
    """Pay once when newly created tread support persists without face contact."""

    if hold_steps < 1 or min_policy_steps < 1 or min_normal_force <= 0.0:
        raise ValueError(
            "hold_steps, min_policy_steps, and min_normal_force must be positive"
        )
    face_contact, tread_contact, tread_force = _virtual_lip_union_contact_state(
        env,
        sensor_names,
        nominal_stair_face_x=nominal_stair_face_x,
        max_face_offset=max_face_offset,
        num_terrain_levels=num_terrain_levels,
        riser_height=riser_height,
        tread_depth=tread_depth,
        corridor_half_width=corridor_half_width,
    )
    if not hasattr(env, "_stair_contact_transfer_hold"):
        env._stair_contact_transfer_hold = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
        env._stair_contact_transfer_stage2_latched = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        env._stair_contact_transfer_stage2_policy_achieved = torch.zeros_like(
            env._stair_contact_transfer_stage2_latched
        )
        env._stair_contact_transfer_face_seen = torch.zeros_like(
            env._stair_contact_transfer_stage2_latched
        )
    fresh = env.episode_length_buf <= 1
    reverse_seed = getattr(
        env,
        "_stair_contact_transfer_reverse_stage15_seed",
        torch.zeros_like(tread_contact),
    )
    stage2_seed = getattr(
        env,
        "_stair_contact_transfer_reverse_stage2_seed",
        torch.zeros_like(tread_contact),
    )
    env._stair_contact_transfer_hold[fresh] = 0
    # A tread-contact bank state is useful for downstream clearance but cannot
    # claim discovery of the transfer that already existed at reset.
    env._stair_contact_transfer_stage2_latched[fresh] = (
        (tread_contact[fresh] & ~reverse_seed[fresh]) | stage2_seed[fresh]
    )
    env._stair_contact_transfer_stage2_policy_achieved[fresh] = stage2_seed[fresh]
    reset_family = getattr(env, "_stair_state_bank_family", None)
    validated_face_source = (
        torch.zeros_like(face_contact)
        if reset_family is None
        else reset_family == 0
    )
    # Family 0 is collected only from same-frame hard-stair face-contact,
    # no-tread states. The recessed level deliberately removes that contact,
    # so retain the validated source fact as the prior-face condition.
    env._stair_contact_transfer_face_seen[fresh] = (
        face_contact[fresh]
        | validated_face_source[fresh]
        | reverse_seed[fresh]
        | stage2_seed[fresh]
    )
    env._stair_contact_transfer_face_seen |= face_contact
    stage1 = getattr(env, "_stair_contact_transfer_stage1_policy_achieved", None)
    if stage1 is None:
        stage1 = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if require_stage15:
        stage15 = getattr(
            env,
            "_stair_contact_transfer_stage15_policy_achieved",
            None,
        )
        if stage15 is None:
            stage15 = torch.zeros_like(stage1)
    else:
        stage15 = torch.ones_like(stage1)
    candidate = (
        (env.episode_length_buf >= min_policy_steps)
        & stage1
        & stage15
        & env._stair_contact_transfer_face_seen
        & tread_contact
        & (tread_force >= min_normal_force)
        & ~face_contact
        & ~env._stair_contact_transfer_stage2_latched
    )
    env._stair_contact_transfer_hold = torch.where(
        candidate,
        env._stair_contact_transfer_hold + 1,
        torch.zeros_like(env._stair_contact_transfer_hold),
    )
    released = (
        env._stair_contact_transfer_hold >= hold_steps
    ) & ~env._stair_contact_transfer_stage2_latched
    env._stair_contact_transfer_stage2_latched |= released
    env._stair_contact_transfer_stage2_policy_achieved |= released
    return released.to(torch.float32)


def stair_riser_face_contact_after_tread(
    env: ManagerBasedRlEnv,
    nominal_stair_face_x: float = 0.66,
    max_face_offset: float = 0.04,
    num_terrain_levels: int = 3,
    riser_height: float = 0.17,
    tread_depth: float = 0.28,
    corridor_half_width: float = 0.36,
    sensor_names: tuple[str, ...] = (
        "head_ground_contact",
        "trunk_ground_contact",
        "legs_ground_contact",
        "feet_stair_contact",
    ),
) -> torch.Tensor:
    """Return a face-contact cost only after the policy found the tread."""

    face, _, _ = _virtual_lip_union_contact_state(
        env,
        sensor_names,
        nominal_stair_face_x=nominal_stair_face_x,
        max_face_offset=max_face_offset,
        num_terrain_levels=num_terrain_levels,
        riser_height=riser_height,
        tread_depth=tread_depth,
        corridor_half_width=corridor_half_width,
    )
    stage1 = getattr(env, "_stair_contact_transfer_stage1_policy_achieved", None)
    if stage1 is None:
        return torch.zeros(env.num_envs, device=env.device)
    return (stage1 & face).to(torch.float32)


def stair_virtual_lip_penetration_cost(
    env: ManagerBasedRlEnv,
    nominal_stair_face_x: float = 0.66,
    max_face_offset: float = 0.04,
    num_terrain_levels: int = 3,
    riser_height: float = 0.17,
    corridor_half_width: float = 0.36,
    speed_scale: float = 0.40,
    min_policy_steps: int = 3,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize fast collision-support penetration into the virtual riser."""

    if speed_scale <= 0.0 or min_policy_steps < 1:
        raise ValueError("speed_scale and min_policy_steps must be positive")
    asset: Entity = env.scene[asset_cfg.name]
    geom_ids = asset.indexing.geom_ids
    centers = asset.data.geom_pos_w
    linear_velocity = torch.nan_to_num(asset.data.geom_lin_vel_w, nan=0.0)
    angular_velocity = torch.nan_to_num(asset.data.geom_ang_vel_w, nan=0.0)
    rbound = asset.data.model.geom_rbound
    if rbound.ndim == 1:
        radii = rbound[geom_ids][None, :].expand(env.num_envs, -1)
    else:
        radii = rbound[:, geom_ids]
    directions = torch.tensor(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
        ),
        dtype=centers.dtype,
        device=centers.device,
    )
    point_offsets = radii[..., None, None] * directions[None, None, :, :]
    positions = (
        centers[..., None, :]
        + point_offsets
        - env.scene.terrain.env_origins[:, None, None, :]
    )
    velocities = linear_velocity[..., None, :] + torch.cross(
        angular_velocity[..., None, :].expand_as(point_offsets),
        point_offsets,
        dim=-1,
    )
    contype = asset.data.model.geom_contype[geom_ids]
    collision_geom = contype > 0
    physical_face_x = _virtual_lip_physical_face_x(
        env,
        nominal_stair_face_x=nominal_stair_face_x,
        max_face_offset=max_face_offset,
        num_terrain_levels=num_terrain_levels,
    )
    offset = physical_face_x - nominal_stair_face_x
    inside = (
        (offset[:, None, None] > 1.0e-6)
        & collision_geom[None, :, None]
        & (positions[..., 0] >= nominal_stair_face_x)
        & (positions[..., 0] <= physical_face_x[:, None, None])
        & (positions[..., 2] >= 0.0)
        & (positions[..., 2] <= riser_height)
        & (positions[..., 1].abs() <= corridor_half_width)
    )
    depth = torch.clamp(
        (positions[..., 0] - nominal_stair_face_x)
        / torch.clamp(offset[:, None, None], min=1.0e-6),
        min=0.0,
        max=1.0,
    )
    speed = torch.clamp(
        torch.linalg.vector_norm(velocities, dim=-1) / speed_scale,
        min=0.0,
        max=2.0,
    )
    cost = torch.where(inside, depth * speed, torch.zeros_like(depth))
    cost = cost.flatten(start_dim=1).max(dim=-1).values
    return torch.where(
        env.episode_length_buf >= min_policy_steps,
        cost,
        torch.zeros_like(cost),
    )


def _stair_shell_corner_state(
    env: ManagerBasedRlEnv,
    shell_half_extents: tuple[float, float, float],
    asset_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return trunk-shell box corners in terrain-local coordinates and root y."""

    if min(shell_half_extents) <= 0.0:
        raise ValueError("Shell half extents must be positive")
    asset: Entity = env.scene[asset_cfg.name]
    centers = asset.data.geom_pos_w[:, asset_cfg.geom_ids]
    quaternions = asset.data.geom_quat_w[:, asset_cfg.geom_ids]
    if centers.shape[1] != 1:
        raise ValueError("True shell clearance requires exactly one shell geom")
    signs = torch.tensor(
        (
            (-1.0, -1.0, -1.0),
            (-1.0, -1.0, 1.0),
            (-1.0, 1.0, -1.0),
            (-1.0, 1.0, 1.0),
            (1.0, -1.0, -1.0),
            (1.0, -1.0, 1.0),
            (1.0, 1.0, -1.0),
            (1.0, 1.0, 1.0),
        ),
        dtype=centers.dtype,
        device=centers.device,
    )
    half_extents = torch.tensor(
        shell_half_extents, dtype=centers.dtype, device=centers.device
    )
    local_corners = signs * half_extents
    corner_count = local_corners.shape[0]
    rotated = quat_apply(
        quaternions[:, :, None, :].expand(-1, -1, corner_count, -1),
        local_corners[None, None, :, :].expand(env.num_envs, 1, -1, -1),
    )
    corners = (
        centers[:, :, None, :]
        + rotated
        - env.scene.terrain.env_origins[:, None, None, :]
    )
    root_y = torch.abs(
        asset.data.root_link_pos_w[:, 1] - env.scene.terrain.env_origins[:, 1]
    )
    return corners, root_y


def stair_true_shell_clearance_candidate(
    env: ManagerBasedRlEnv,
    nominal_stair_face_x: float = 0.66,
    riser_height: float = 0.17,
    corridor_half_width: float = 0.36,
    shell_half_extents: tuple[float, float, float] = (0.034, 0.031, 0.022),
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Return the instantaneous eight-corner hard-shell geometry gate."""

    corners, root_y = _stair_shell_corner_state(
        env, shell_half_extents, asset_cfg
    )
    return (
        torch.isfinite(corners).all(dim=(-1, -2, -3))
        & (corners[..., 0] >= nominal_stair_face_x).all(dim=(-1, -2))
        & (corners[..., 2] >= riser_height).all(dim=(-1, -2))
        & (root_y <= corridor_half_width)
    )


def stair_true_shell_clearance_frontier_value(
    env: ManagerBasedRlEnv,
    start_x: float = 0.625,
    target_x: float = 0.660,
    start_height: float = 0.150,
    target_height: float = 0.170,
    corridor_half_width: float = 0.20,
    required_latch_name: str | None = (
        "_stair_contact_transfer_stage15_policy_achieved"
    ),
    shell_half_extents: tuple[float, float, float] = (0.034, 0.031, 0.022),
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Return bounded worst-corner progress toward true shell clearance."""

    if target_x <= start_x or target_height <= start_height:
        raise ValueError("Shell frontier targets must exceed their starts")
    corners, root_y = _stair_shell_corner_state(
        env, shell_half_extents, asset_cfg
    )
    min_x = corners[..., 0].amin(dim=(-1, -2))
    min_z = corners[..., 2].amin(dim=(-1, -2))
    x_progress = torch.clamp(
        (min_x - start_x) / (target_x - start_x), min=0.0, max=1.0
    )
    z_progress = torch.clamp(
        (min_z - start_height) / (target_height - start_height),
        min=0.0,
        max=1.0,
    )
    value = torch.minimum(x_progress, z_progress)
    required = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    if required_latch_name is not None:
        required = getattr(env, required_latch_name, None)
        if required is None:
            required = torch.zeros(
                env.num_envs, dtype=torch.bool, device=env.device
            )
    finite = torch.isfinite(corners).all(dim=(-1, -2, -3))
    eligible = required & finite & (root_y <= corridor_half_width)
    return torch.where(eligible, value, torch.zeros_like(value))


def stair_true_shell_clearance_frontier(
    env: ManagerBasedRlEnv,
    start_x: float = 0.625,
    target_x: float = 0.660,
    start_height: float = 0.150,
    target_height: float = 0.170,
    corridor_half_width: float = 0.20,
    required_latch_name: str = "_stair_contact_transfer_stage15_policy_achieved",
    shell_half_extents: tuple[float, float, float] = (0.034, 0.031, 0.022),
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Pay bounded episode-best progress of the worst trunk-shell corner.

    The minimum of normalized x and z margins makes both dimensions necessary.
    Reset states establish the baseline, so replayed progress pays nothing.
    Holding or oscillating around a previously reached margin cannot be farmed.
    """

    value = stair_true_shell_clearance_frontier_value(
        env=env,
        start_x=start_x,
        target_x=target_x,
        start_height=start_height,
        target_height=target_height,
        corridor_half_width=corridor_half_width,
        required_latch_name=required_latch_name,
        shell_half_extents=shell_half_extents,
        asset_cfg=asset_cfg,
    )
    state_name = "_stair_true_shell_clearance_frontier_best"
    previous_best = getattr(env, state_name, None)
    if previous_best is None:
        previous_best = value.clone()
        setattr(env, state_name, previous_best)
    fresh = env.episode_length_buf <= 1
    previous_best[fresh] = value[fresh]
    new_best = torch.maximum(previous_best, value)
    delta = new_best - previous_best
    setattr(env, state_name, new_best)
    return delta


def stair_true_shell_clearance(
    env: ManagerBasedRlEnv,
    nominal_stair_face_x: float = 0.66,
    riser_height: float = 0.17,
    corridor_half_width: float = 0.36,
    hold_steps: int = 4,
    required_latch_name: str | None = None,
    shell_half_extents: tuple[float, float, float] = (0.034, 0.031, 0.022),
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Pay once when every trunk-shell box corner clears the nominal lip."""

    if hold_steps < 1:
        raise ValueError("Shell hold must be positive")
    required = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    if required_latch_name is not None:
        required = getattr(env, required_latch_name, None)
        if required is None:
            required = torch.zeros(
                env.num_envs, dtype=torch.bool, device=env.device
            )
    candidate = stair_true_shell_clearance_candidate(
        env=env,
        nominal_stair_face_x=nominal_stair_face_x,
        riser_height=riser_height,
        corridor_half_width=corridor_half_width,
        shell_half_extents=shell_half_extents,
        asset_cfg=asset_cfg,
    )
    candidate &= required
    if not hasattr(env, "_stair_true_shell_clearance_hold"):
        env._stair_true_shell_clearance_hold = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
        env._stair_true_shell_clearance_latched = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        env._stair_true_shell_clearance_policy_achieved = torch.zeros_like(
            env._stair_true_shell_clearance_latched
        )
    fresh = env.episode_length_buf <= 1
    env._stair_true_shell_clearance_hold[fresh] = 0
    env._stair_true_shell_clearance_latched[fresh] = candidate[fresh]
    env._stair_true_shell_clearance_policy_achieved[fresh] = False
    eligible = candidate & ~env._stair_true_shell_clearance_latched
    env._stair_true_shell_clearance_hold = torch.where(
        eligible,
        env._stair_true_shell_clearance_hold + 1,
        torch.zeros_like(env._stair_true_shell_clearance_hold),
    )
    newly_cleared = (
        env._stair_true_shell_clearance_hold >= hold_steps
    ) & ~env._stair_true_shell_clearance_latched
    env._stair_true_shell_clearance_latched |= newly_cleared
    env._stair_true_shell_clearance_policy_achieved |= newly_cleared
    return newly_cleared.to(torch.float32)


def stair_contact_loaded_release(
    env: ManagerBasedRlEnv,
    stair_face_x: float = 0.66,
    riser_height: float = 0.17,
    tread_depth: float = 0.28,
    corridor_half_width: float = 0.20,
    arm_hold_steps: int = 2,
    release_window_steps: int = 8,
    min_forward_speed: float = 0.05,
    min_vertical_speed: float = 0.08,
    target_forward_delta: float = 0.040,
    target_vertical_delta: float = 0.025,
    sensor_names: tuple[str, ...] = (
        "robot_ground_contact",
        "head_ground_contact",
        "legs_ground_contact",
        "trunk_ground_contact",
    ),
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Pay once for an upward-forward release after stable stair contact.

    Contact itself pays nothing. Two consecutive classified face or tread
    contact frames arm a short release window and snapshot the root pose. A
    payout is possible only after the contact releases, or a face contact
    shifts onto the tread, while the root has positive forward and vertical
    velocity. Progress is measured from the arm snapshot, so reset placement,
    persistent bracing, and repeated bumps cannot create a jackpot.
    """

    if arm_hold_steps < 1 or release_window_steps < 1:
        raise ValueError("arm_hold_steps and release_window_steps must be positive")

    asset: Entity = env.scene[asset_cfg.name]
    x, z, _ = _stair_local_state(env, asset_cfg)
    y = torch.abs(
        asset.data.root_link_pos_w[:, 1] - env.scene.terrain.env_origins[:, 1]
    )
    face_contact = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    tread_contact = torch.zeros_like(face_contact)
    for sensor_name in sensor_names:
        if sensor_name not in env.scene.sensors:
            raise RuntimeError(f"Missing required contact sensor: {sensor_name}")
        face, tread, _ = _standard_stair_contact_masks(
            env,
            sensor_name,
            stair_face_x=stair_face_x,
            riser_height=riser_height,
            tread_depth=tread_depth,
            corridor_half_width=corridor_half_width,
        )
        face_contact |= face.any(dim=-1)
        tread_contact |= tread.any(dim=-1)
    contact = face_contact | tread_contact

    state_names = (
        "_stair_contact_release_streak",
        "_stair_contact_release_armed",
        "_stair_contact_release_paid",
        "_stair_contact_release_window",
        "_stair_contact_release_arm_x",
        "_stair_contact_release_arm_z",
        "_stair_contact_release_arm_face",
        "_stair_contact_loaded_release_latched",
    )
    if not hasattr(env, state_names[0]):
        env._stair_contact_release_streak = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
        env._stair_contact_release_armed = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        env._stair_contact_release_paid = torch.zeros_like(
            env._stair_contact_release_armed
        )
        env._stair_contact_release_window = torch.zeros_like(
            env._stair_contact_release_streak
        )
        env._stair_contact_release_arm_x = torch.zeros_like(x)
        env._stair_contact_release_arm_z = torch.zeros_like(z)
        env._stair_contact_release_arm_face = torch.zeros_like(
            env._stair_contact_release_armed
        )
        env._stair_contact_loaded_release_latched = torch.zeros_like(
            env._stair_contact_release_armed
        )

    fresh = env.episode_length_buf <= 1
    env._stair_contact_release_streak[fresh] = 0
    env._stair_contact_release_armed[fresh] = False
    env._stair_contact_release_paid[fresh] = False
    env._stair_contact_release_window[fresh] = 0
    env._stair_contact_release_arm_x[fresh] = 0.0
    env._stair_contact_release_arm_z[fresh] = 0.0
    env._stair_contact_release_arm_face[fresh] = False
    env._stair_contact_loaded_release_latched[fresh] = False

    cleared = getattr(env, "_stair_first_riser_latched", None)
    if cleared is None:
        cleared = torch.zeros_like(env._stair_contact_release_armed)
    arm_candidate = (
        (env.episode_length_buf >= 3)
        & (y <= corridor_half_width)
        & contact
        & ~cleared
        & ((x < 0.665) | (z < 0.175))
        & ~env._stair_contact_release_armed
    )
    env._stair_contact_release_streak = torch.where(
        arm_candidate,
        env._stair_contact_release_streak + 1,
        torch.zeros_like(env._stair_contact_release_streak),
    )
    newly_armed = (
        env._stair_contact_release_streak >= arm_hold_steps
    ) & ~env._stair_contact_release_armed
    env._stair_contact_release_armed |= newly_armed
    env._stair_contact_release_arm_x[newly_armed] = x[newly_armed]
    env._stair_contact_release_arm_z[newly_armed] = z[newly_armed]
    env._stair_contact_release_arm_face[newly_armed] = face_contact[newly_armed]
    env._stair_contact_release_window = torch.where(
        newly_armed,
        torch.full_like(
            env._stair_contact_release_window, release_window_steps
        ),
        torch.clamp(env._stair_contact_release_window - 1, min=0),
    )

    linear_velocity = torch.nan_to_num(asset.data.root_link_lin_vel_w, nan=0.0)
    contact_transition = (~contact) | (
        env._stair_contact_release_arm_face & tread_contact
    )
    release = (
        env._stair_contact_release_armed
        & ~newly_armed
        & ~env._stair_contact_release_paid
        & (env._stair_contact_release_window > 0)
        & contact_transition
        & (linear_velocity[:, 0] > min_forward_speed)
        & (linear_velocity[:, 2] > min_vertical_speed)
        & (y <= corridor_half_width)
    )
    forward_progress = torch.clamp(
        (x - env._stair_contact_release_arm_x)
        / max(target_forward_delta, 1e-6),
        0.0,
        1.0,
    )
    vertical_progress = torch.clamp(
        (z - env._stair_contact_release_arm_z)
        / max(target_vertical_delta, 1e-6),
        0.0,
        1.0,
    )
    progress = 0.60 * forward_progress + 0.40 * vertical_progress
    payout = torch.where(release, progress, torch.zeros_like(progress))
    env._stair_contact_release_paid |= release
    env._stair_contact_loaded_release_latched |= release
    return payout


def stair_contact_lip_commitment(
    env: ManagerBasedRlEnv,
    stair_face_x: float = 0.66,
    riser_height: float = 0.17,
    tread_depth: float = 0.28,
    corridor_half_width: float = 0.20,
    arm_hold_steps: int = 2,
    impulse_window_steps: int = 12,
    min_forward_velocity_gain: float = 0.05,
    min_vertical_velocity_gain: float = 0.08,
    commitment_delay_steps: int = 4,
    commitment_hold_steps: int = 2,
    commitment_x: float = 0.645,
    commitment_z: float = 0.175,
    root_over_lip_x: float = 0.665,
    target_forward_delta: float = 0.040,
    target_vertical_delta: float = 0.025,
    sensor_names: tuple[str, ...] = (
        "robot_ground_contact",
        "head_ground_contact",
        "legs_ground_contact",
        "trunk_ground_contact",
    ),
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Pay once when a contact-created impulse commits the root to the lip.

    Stable face or tread contact snapshots root pose and velocity. A later
    impulse must increase both forward and vertical velocity relative to that
    snapshot. Four control steps later, the root must occupy the lip-height
    corridor for two consecutive frames. Instantaneous collision bounces and
    contact duration therefore pay nothing.
    """

    if min(
        arm_hold_steps,
        impulse_window_steps,
        commitment_delay_steps,
        commitment_hold_steps,
    ) < 1:
        raise ValueError("Lip-commitment step counts must be positive")

    asset: Entity = env.scene[asset_cfg.name]
    x, z, _ = _stair_local_state(env, asset_cfg)
    local_y = (
        asset.data.root_link_pos_w[:, 1] - env.scene.terrain.env_origins[:, 1]
    )
    abs_y = torch.abs(local_y)
    linear_velocity = torch.nan_to_num(asset.data.root_link_lin_vel_w, nan=0.0)
    vx = linear_velocity[:, 0]
    vz = linear_velocity[:, 2]
    finite = torch.isfinite(torch.stack((x, z, abs_y, vx, vz), dim=-1)).all(
        dim=-1
    )

    face_contact = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    tread_contact = torch.zeros_like(face_contact)
    for sensor_name in sensor_names:
        if sensor_name not in env.scene.sensors:
            raise RuntimeError(f"Missing required contact sensor: {sensor_name}")
        face, tread, _ = _standard_stair_contact_masks(
            env,
            sensor_name,
            stair_face_x=stair_face_x,
            riser_height=riser_height,
            tread_depth=tread_depth,
            corridor_half_width=corridor_half_width,
        )
        face_contact |= face.any(dim=-1)
        tread_contact |= tread.any(dim=-1)
    contact = face_contact | tread_contact

    if not hasattr(env, "_stair_lip_commitment_load_streak"):
        env._stair_lip_commitment_load_streak = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
        env._stair_lip_commitment_armed = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        env._stair_lip_commitment_paid = torch.zeros_like(
            env._stair_lip_commitment_armed
        )
        env._stair_lip_commitment_impulse_seen = torch.zeros_like(
            env._stair_lip_commitment_armed
        )
        env._stair_lip_commitment_impulse_window = torch.zeros_like(
            env._stair_lip_commitment_load_streak
        )
        env._stair_lip_commitment_delay = torch.zeros_like(
            env._stair_lip_commitment_load_streak
        )
        env._stair_lip_commitment_check_remaining = torch.zeros_like(
            env._stair_lip_commitment_load_streak
        )
        env._stair_lip_commitment_hold_streak = torch.zeros_like(
            env._stair_lip_commitment_load_streak
        )
        env._stair_lip_commitment_arm_x = torch.zeros_like(x)
        env._stair_lip_commitment_arm_z = torch.zeros_like(z)
        env._stair_lip_commitment_arm_vx = torch.zeros_like(vx)
        env._stair_lip_commitment_arm_vz = torch.zeros_like(vz)
        env._stair_lip_commitment_impulse_latched = torch.zeros_like(
            env._stair_lip_commitment_armed
        )
        env._stair_lip_commitment_latched = torch.zeros_like(
            env._stair_lip_commitment_armed
        )

    fresh = env.episode_length_buf <= 1
    env._stair_lip_commitment_load_streak[fresh] = 0
    env._stair_lip_commitment_armed[fresh] = False
    env._stair_lip_commitment_paid[fresh] = False
    env._stair_lip_commitment_impulse_seen[fresh] = False
    env._stair_lip_commitment_impulse_window[fresh] = 0
    env._stair_lip_commitment_delay[fresh] = 0
    env._stair_lip_commitment_check_remaining[fresh] = 0
    env._stair_lip_commitment_hold_streak[fresh] = 0
    env._stair_lip_commitment_arm_x[fresh] = 0.0
    env._stair_lip_commitment_arm_z[fresh] = 0.0
    env._stair_lip_commitment_arm_vx[fresh] = 0.0
    env._stair_lip_commitment_arm_vz[fresh] = 0.0
    env._stair_lip_commitment_impulse_latched[fresh] = False
    env._stair_lip_commitment_latched[fresh] = False

    cleared = getattr(env, "_stair_first_riser_latched", None)
    if cleared is None:
        cleared = torch.zeros_like(env._stair_lip_commitment_armed)
    arm_candidate = (
        (env.episode_length_buf >= 3)
        & finite
        & (abs_y <= corridor_half_width)
        & contact
        & ~cleared
        & ((x < root_over_lip_x) | (z < commitment_z))
        & ~env._stair_lip_commitment_armed
    )
    env._stair_lip_commitment_load_streak = torch.where(
        arm_candidate,
        env._stair_lip_commitment_load_streak + 1,
        torch.zeros_like(env._stair_lip_commitment_load_streak),
    )
    newly_armed = (
        env._stair_lip_commitment_load_streak >= arm_hold_steps
    ) & ~env._stair_lip_commitment_armed
    env._stair_lip_commitment_armed |= newly_armed
    env._stair_lip_commitment_arm_x[newly_armed] = x[newly_armed]
    env._stair_lip_commitment_arm_z[newly_armed] = z[newly_armed]
    env._stair_lip_commitment_arm_vx[newly_armed] = vx[newly_armed]
    env._stair_lip_commitment_arm_vz[newly_armed] = vz[newly_armed]
    env._stair_lip_commitment_impulse_window = torch.where(
        newly_armed,
        torch.full_like(
            env._stair_lip_commitment_impulse_window, impulse_window_steps
        ),
        env._stair_lip_commitment_impulse_window,
    )

    impulse = (
        env._stair_lip_commitment_armed
        & ~newly_armed
        & ~env._stair_lip_commitment_impulse_seen
        & (env._stair_lip_commitment_impulse_window > 0)
        & finite
        & ((vx - env._stair_lip_commitment_arm_vx) >= min_forward_velocity_gain)
        & ((vz - env._stair_lip_commitment_arm_vz) >= min_vertical_velocity_gain)
        & (vx > 0.0)
        & (vz > 0.0)
        & (abs_y <= corridor_half_width)
    )
    env._stair_lip_commitment_impulse_seen |= impulse
    env._stair_lip_commitment_impulse_latched |= impulse
    decay_window = (
        env._stair_lip_commitment_armed
        & ~newly_armed
        & ~env._stair_lip_commitment_impulse_seen
    )
    env._stair_lip_commitment_impulse_window = torch.where(
        decay_window,
        torch.clamp(env._stair_lip_commitment_impulse_window - 1, min=0),
        env._stair_lip_commitment_impulse_window,
    )

    previous_delay = env._stair_lip_commitment_delay.clone()
    env._stair_lip_commitment_delay = torch.where(
        impulse,
        torch.full_like(
            env._stair_lip_commitment_delay, commitment_delay_steps
        ),
        env._stair_lip_commitment_delay,
    )
    delay_active = (
        env._stair_lip_commitment_impulse_seen
        & ~impulse
        & (env._stair_lip_commitment_delay > 0)
    )
    start_commitment_check = delay_active & (previous_delay == 1)
    env._stair_lip_commitment_delay = torch.where(
        delay_active,
        torch.clamp(env._stair_lip_commitment_delay - 1, min=0),
        env._stair_lip_commitment_delay,
    )
    env._stair_lip_commitment_check_remaining = torch.where(
        start_commitment_check,
        torch.full_like(
            env._stair_lip_commitment_check_remaining, commitment_hold_steps
        ),
        env._stair_lip_commitment_check_remaining,
    )

    check_active = (
        env._stair_lip_commitment_impulse_seen
        & ~env._stair_lip_commitment_paid
        & (env._stair_lip_commitment_check_remaining > 0)
    )
    commitment_candidate = (
        check_active
        & finite
        & (x >= commitment_x)
        & (z >= commitment_z)
        & (abs_y <= corridor_half_width)
    )
    env._stair_lip_commitment_hold_streak = torch.where(
        commitment_candidate,
        env._stair_lip_commitment_hold_streak + 1,
        torch.where(
            check_active,
            torch.zeros_like(env._stair_lip_commitment_hold_streak),
            env._stair_lip_commitment_hold_streak,
        ),
    )
    commitment = (
        env._stair_lip_commitment_hold_streak >= commitment_hold_steps
    ) & ~env._stair_lip_commitment_paid
    env._stair_lip_commitment_check_remaining = torch.where(
        check_active,
        torch.clamp(env._stair_lip_commitment_check_remaining - 1, min=0),
        env._stair_lip_commitment_check_remaining,
    )

    forward_progress = torch.clamp(
        (x - env._stair_lip_commitment_arm_x) / max(target_forward_delta, 1e-6),
        0.0,
        1.0,
    )
    vertical_progress = torch.clamp(
        (z - env._stair_lip_commitment_arm_z) / max(target_vertical_delta, 1e-6),
        0.0,
        1.0,
    )
    progress = 0.70 * forward_progress + 0.30 * vertical_progress
    payout = torch.where(commitment, progress, torch.zeros_like(progress))
    env._stair_lip_commitment_paid |= commitment
    env._stair_lip_commitment_latched |= commitment
    return payout


def stair_contact_lip_checkpoint_potential(
    env: ManagerBasedRlEnv,
    stair_face_x: float = 0.66,
    riser_height: float = 0.17,
    tread_depth: float = 0.28,
    corridor_half_width: float = 0.20,
    bypass_half_width: float = 0.36,
    arm_hold_steps: int = 2,
    target_hold_steps: int = 2,
    x_start: float = 0.540,
    x_target: float = 0.665,
    x_softness: float = 0.010,
    z_start: float = 0.100,
    z_target: float = 0.175,
    z_softness: float = 0.005,
    sensor_names: tuple[str, ...] = (
        "robot_ground_contact",
        "head_ground_contact",
        "legs_ground_contact",
        "trunk_ground_contact",
    ),
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward signed simultaneous x/z progress toward the first stair lip.

    Two classified stair-contact frames arm a mentor-style center-of-mass
    checkpoint.  Its geometric-mean potential stays informative before the
    exact goal but collapses when either forward or vertical progress is poor.
    Only the signed change is paid, so holding and oscillating cannot create a
    jackpot.  The shaping stops at a held root-over-lip state or side bypass.
    """

    if min(arm_hold_steps, target_hold_steps) < 1:
        raise ValueError("Lip-checkpoint hold steps must be positive")
    if min(x_softness, z_softness) <= 0.0:
        raise ValueError("Lip-checkpoint softness must be positive")
    if x_target <= x_start or z_target <= z_start:
        raise ValueError("Lip-checkpoint targets must exceed their starts")

    asset: Entity = env.scene[asset_cfg.name]
    x, z, _ = _stair_local_state(env, asset_cfg)
    abs_y = torch.abs(
        asset.data.root_link_pos_w[:, 1] - env.scene.terrain.env_origins[:, 1]
    )
    finite = torch.isfinite(torch.stack((x, z, abs_y), dim=-1)).all(dim=-1)

    face_contact = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    tread_contact = torch.zeros_like(face_contact)
    for sensor_name in sensor_names:
        if sensor_name not in env.scene.sensors:
            raise RuntimeError(f"Missing required contact sensor: {sensor_name}")
        face, tread, _ = _standard_stair_contact_masks(
            env,
            sensor_name,
            stair_face_x=stair_face_x,
            riser_height=riser_height,
            tread_depth=tread_depth,
            corridor_half_width=corridor_half_width,
        )
        face_contact |= face.any(dim=-1)
        tread_contact |= tread.any(dim=-1)
    contact = face_contact | tread_contact

    def soft_progress(
        value: torch.Tensor, start: float, target: float, softness: float
    ) -> torch.Tensor:
        numerator = torch.nn.functional.softplus((value - start) / softness)
        denominator = torch.nn.functional.softplus(
            torch.as_tensor(
                (target - start) / softness,
                dtype=value.dtype,
                device=value.device,
            )
        )
        return torch.clamp(numerator / denominator, min=0.0, max=1.0)

    px = soft_progress(x, x_start, x_target, x_softness)
    pz = soft_progress(z, z_start, z_target, z_softness)
    potential = torch.sqrt(torch.clamp(px * pz, min=0.0))

    if not hasattr(env, "_stair_lip_checkpoint_load_streak"):
        env._stair_lip_checkpoint_load_streak = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
        env._stair_lip_checkpoint_armed = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        env._stair_lip_checkpoint_contact_qualified = torch.zeros_like(
            env._stair_lip_checkpoint_armed
        )
        env._stair_lip_checkpoint_previous = torch.zeros_like(x)
        env._stair_lip_checkpoint_target_hold = torch.zeros_like(
            env._stair_lip_checkpoint_load_streak
        )
        env._stair_lip_checkpoint_progress_latched = torch.zeros_like(
            env._stair_lip_checkpoint_armed
        )
        env._stair_lip_checkpoint_target_latched = torch.zeros_like(
            env._stair_lip_checkpoint_armed
        )
        env._stair_lip_checkpoint_bypass_latched = torch.zeros_like(
            env._stair_lip_checkpoint_armed
        )

    fresh = env.episode_length_buf <= 1
    env._stair_lip_checkpoint_load_streak[fresh] = 0
    env._stair_lip_checkpoint_armed[fresh] = False
    env._stair_lip_checkpoint_contact_qualified[fresh] = False
    env._stair_lip_checkpoint_previous[fresh] = 0.0
    env._stair_lip_checkpoint_target_hold[fresh] = 0
    env._stair_lip_checkpoint_progress_latched[fresh] = False
    env._stair_lip_checkpoint_target_latched[fresh] = False
    env._stair_lip_checkpoint_bypass_latched[fresh] = False

    already_target = (x >= x_target) & (z >= z_target)
    contact_candidate = (
        (env.episode_length_buf >= 3)
        & finite
        & (abs_y <= corridor_half_width)
        & contact
        & ~already_target
        & ~env._stair_lip_checkpoint_contact_qualified
    )
    env._stair_lip_checkpoint_load_streak = torch.where(
        contact_candidate,
        env._stair_lip_checkpoint_load_streak + 1,
        torch.zeros_like(env._stair_lip_checkpoint_load_streak),
    )
    newly_contact_qualified = (
        env._stair_lip_checkpoint_load_streak >= arm_hold_steps
    ) & ~env._stair_lip_checkpoint_contact_qualified
    env._stair_lip_checkpoint_contact_qualified |= newly_contact_qualified
    impulse_latched = getattr(env, "_stair_lip_commitment_impulse_latched", None)
    if impulse_latched is None:
        impulse_latched = torch.zeros_like(env._stair_lip_checkpoint_armed)
    newly_armed = (
        env._stair_lip_checkpoint_contact_qualified
        & impulse_latched
        & finite
        & (abs_y <= corridor_half_width)
        & ~already_target
        & ~env._stair_lip_checkpoint_armed
    )
    env._stair_lip_checkpoint_armed |= newly_armed
    env._stair_lip_checkpoint_previous[newly_armed] = potential[newly_armed]

    bypass_candidate = (
        env._stair_lip_checkpoint_armed
        & (x >= stair_face_x)
        & (abs_y > bypass_half_width)
    )
    active = (
        env._stair_lip_checkpoint_armed
        & ~env._stair_lip_checkpoint_target_latched
        & ~env._stair_lip_checkpoint_bypass_latched
        & finite
    )
    effective_potential = torch.where(
        abs_y <= corridor_half_width, potential, torch.zeros_like(potential)
    )
    signed_delta = torch.where(
        active & ~newly_armed,
        effective_potential - env._stair_lip_checkpoint_previous,
        torch.zeros_like(potential),
    )
    positive_progress = signed_delta > 1.0e-6
    env._stair_lip_checkpoint_progress_latched |= positive_progress
    env._stair_lip_checkpoint_previous = torch.where(
        active, effective_potential, env._stair_lip_checkpoint_previous
    )
    # Calculate the corridor-collapse repayment before making a bypass terminal.
    env._stair_lip_checkpoint_bypass_latched |= bypass_candidate

    target_candidate = (
        active
        & ~bypass_candidate
        & (x >= x_target)
        & (z >= z_target)
        & (abs_y <= corridor_half_width)
    )
    env._stair_lip_checkpoint_target_hold = torch.where(
        target_candidate,
        env._stair_lip_checkpoint_target_hold + 1,
        torch.zeros_like(env._stair_lip_checkpoint_target_hold),
    )
    reached_target = (
        env._stair_lip_checkpoint_target_hold >= target_hold_steps
    ) & ~env._stair_lip_checkpoint_target_latched
    env._stair_lip_checkpoint_target_latched |= reached_target
    return signed_delta / env.step_dt


def stair_coupled_frontier_collocation(
    env: ManagerBasedRlEnv,
    stair_face_x: float = 0.66,
    corridor_half_width: float = 0.20,
    bypass_half_width: float = 0.36,
    arm_after_control_steps: int = 2,
    target_hold_steps: int = 2,
    x_start: float = 0.540,
    x_target: float = 0.665,
    z_start: float = 0.100,
    z_target: float = 0.175,
    target_bonus: float = 4.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Pay finite new coupled x/z frontier; gate only completion on impulse.

    This continuation is paired with elevated manufacturer roll phases placed
    immediately before the first lip.  The minimum of normalized forward and
    vertical progress prevents either axis from compensating for the other.
    A running maximum makes dense progress finite and non-farmable. Progress
    starts from the maximum seen before arming, so lowering progress during the
    first policy actions cannot create a profitable rebound. The held target
    bonus still requires the physical A24 two-contact impulse. This keeps an
    attainable mentor signal without paying reset state or accepting a target
    reached by unrelated motion.
    """

    if min(arm_after_control_steps, target_hold_steps) < 1:
        raise ValueError("Coupled-frontier hold steps must be positive")
    if x_target <= x_start or z_target <= z_start:
        raise ValueError("Coupled-frontier targets must exceed their starts")
    if target_bonus < 0.0:
        raise ValueError("Coupled-frontier target bonus must be nonnegative")

    asset: Entity = env.scene[asset_cfg.name]
    x, z, _ = _stair_local_state(env, asset_cfg)
    abs_y = torch.abs(
        asset.data.root_link_pos_w[:, 1] - env.scene.terrain.env_origins[:, 1]
    )
    finite = torch.isfinite(torch.stack((x, z, abs_y), dim=-1)).all(dim=-1)

    x_progress = torch.clamp(
        (x - x_start) / (x_target - x_start), min=0.0, max=1.0
    )
    z_progress = torch.clamp(
        (z - z_start) / (z_target - z_start), min=0.0, max=1.0
    )
    coupled = torch.minimum(x_progress, z_progress)

    if not hasattr(env, "_stair_coupled_frontier_armed"):
        env._stair_coupled_frontier_armed = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        env._stair_coupled_frontier_prearm_peak = torch.zeros_like(x)
        env._stair_coupled_frontier_arm_value = torch.zeros_like(x)
        env._stair_coupled_frontier_max_value = torch.zeros_like(x)
        env._stair_coupled_frontier_paid_gain = torch.zeros_like(x)
        env._stair_coupled_frontier_target_hold = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
        env._stair_coupled_frontier_gain10_latched = torch.zeros_like(
            env._stair_coupled_frontier_armed
        )
        env._stair_coupled_frontier_gain25_latched = torch.zeros_like(
            env._stair_coupled_frontier_armed
        )
        env._stair_coupled_frontier_raw_gain10_latched = torch.zeros_like(
            env._stair_coupled_frontier_armed
        )
        env._stair_coupled_frontier_raw_gain25_latched = torch.zeros_like(
            env._stair_coupled_frontier_armed
        )
        env._stair_coupled_frontier_target_latched = torch.zeros_like(
            env._stair_coupled_frontier_armed
        )
        env._stair_coupled_frontier_bypass_latched = torch.zeros_like(
            env._stair_coupled_frontier_armed
        )

    fresh = env.episode_length_buf <= 1
    env._stair_coupled_frontier_armed[fresh] = False
    env._stair_coupled_frontier_prearm_peak[fresh] = 0.0
    env._stair_coupled_frontier_arm_value[fresh] = 0.0
    env._stair_coupled_frontier_max_value[fresh] = 0.0
    env._stair_coupled_frontier_paid_gain[fresh] = 0.0
    env._stair_coupled_frontier_target_hold[fresh] = 0
    env._stair_coupled_frontier_gain10_latched[fresh] = False
    env._stair_coupled_frontier_gain25_latched[fresh] = False
    env._stair_coupled_frontier_raw_gain10_latched[fresh] = False
    env._stair_coupled_frontier_raw_gain25_latched[fresh] = False
    env._stair_coupled_frontier_target_latched[fresh] = False
    env._stair_coupled_frontier_bypass_latched[fresh] = False

    already_target = (x >= x_target) & (z >= z_target)
    prearm_eligible = (
        ~env._stair_coupled_frontier_armed
        & finite
        & (abs_y <= corridor_half_width)
        & ~already_target
    )
    env._stair_coupled_frontier_prearm_peak = torch.where(
        prearm_eligible,
        torch.maximum(env._stair_coupled_frontier_prearm_peak, coupled),
        env._stair_coupled_frontier_prearm_peak,
    )
    newly_armed = (
        (env.episode_length_buf >= arm_after_control_steps)
        & finite
        & (abs_y <= corridor_half_width)
        & ~already_target
        & ~env._stair_coupled_frontier_armed
    )
    env._stair_coupled_frontier_armed |= newly_armed
    env._stair_coupled_frontier_arm_value[newly_armed] = (
        env._stair_coupled_frontier_prearm_peak[newly_armed]
    )
    env._stair_coupled_frontier_max_value[newly_armed] = (
        env._stair_coupled_frontier_prearm_peak[newly_armed]
    )
    env._stair_coupled_frontier_paid_gain[newly_armed] = 0.0

    active = (
        env._stair_coupled_frontier_armed
        & ~env._stair_coupled_frontier_target_latched
        & ~env._stair_coupled_frontier_bypass_latched
        & finite
    )
    new_maximum = torch.where(
        active & (abs_y <= corridor_half_width),
        torch.maximum(env._stair_coupled_frontier_max_value, coupled),
        env._stair_coupled_frontier_max_value,
    )
    env._stair_coupled_frontier_max_value = new_maximum
    total_gain = torch.clamp(
        env._stair_coupled_frontier_max_value
        - env._stair_coupled_frontier_arm_value,
        min=0.0,
    )
    impulse_latched = getattr(env, "_stair_lip_commitment_impulse_latched", None)
    if impulse_latched is None:
        impulse_latched = torch.zeros_like(env._stair_coupled_frontier_armed)
    positive_gain = torch.clamp(
        total_gain - env._stair_coupled_frontier_paid_gain, min=0.0
    )
    paid_after_step = env._stair_coupled_frontier_paid_gain + positive_gain
    env._stair_coupled_frontier_paid_gain = torch.where(
        active, paid_after_step, env._stair_coupled_frontier_paid_gain
    )
    env._stair_coupled_frontier_raw_gain10_latched |= active & (
        total_gain >= 0.10
    )
    env._stair_coupled_frontier_raw_gain25_latched |= active & (
        total_gain >= 0.25
    )
    env._stair_coupled_frontier_gain10_latched |= (
        active & impulse_latched & (total_gain >= 0.10)
    )
    env._stair_coupled_frontier_gain25_latched |= (
        active & impulse_latched & (total_gain >= 0.25)
    )

    bypass_candidate = (
        active & (x >= stair_face_x) & (abs_y > bypass_half_width)
    )
    newly_bypassed = bypass_candidate & ~env._stair_coupled_frontier_bypass_latched

    target_candidate = (
        active
        & impulse_latched
        & ~bypass_candidate
        & (x >= x_target)
        & (z >= z_target)
        & (abs_y <= corridor_half_width)
    )
    env._stair_coupled_frontier_target_hold = torch.where(
        target_candidate,
        env._stair_coupled_frontier_target_hold + 1,
        torch.zeros_like(env._stair_coupled_frontier_target_hold),
    )
    reached_target = (
        env._stair_coupled_frontier_target_hold >= target_hold_steps
    ) & ~env._stair_coupled_frontier_target_latched

    payout = positive_gain + target_bonus * reached_target.to(positive_gain.dtype)
    # A lateral escape cannot retain progress collected in the strict corridor.
    payout -= paid_after_step * newly_bypassed.to(paid_after_step.dtype)
    env._stair_coupled_frontier_target_latched |= reached_target
    env._stair_coupled_frontier_bypass_latched |= newly_bypassed
    return payout / env.step_dt


def stair_terminal_position_objective(
    env: ManagerBasedRlEnv,
    target_x: float = 0.720,
    target_y: float = 0.0,
    target_z: float = 0.205,
    reward_window_s: float = 0.50,
    x_scale: float = 0.08,
    y_scale: float = 0.12,
    z_scale: float = 0.06,
    max_abs_y: float = 0.20,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward absolute goal arrival only during the terminal time window.

    This is a robot-scale version of the final-position objective from Rudin
    et al. (IROS 2022). The policy is free to jump, roll, or use its shell
    before the final window. A reciprocal distance score remains visible from
    the measured near-lip frontier, while the fixed target prevents partial
    progress followed by a crash from becoming the objective.
    """

    if reward_window_s <= 0.0:
        raise ValueError("Terminal-position reward window must be positive")
    if min(x_scale, y_scale, z_scale) <= 0.0:
        raise ValueError("Terminal-position scales must be positive")
    if max_abs_y <= 0.0:
        raise ValueError("Terminal-position lateral corridor must be positive")

    asset: Entity = env.scene[asset_cfg.name]
    local = asset.data.root_link_pos_w - env.scene.terrain.env_origins
    finite = torch.isfinite(local).all(dim=-1)
    dx = (local[:, 0] - target_x) / x_scale
    dy = (local[:, 1] - target_y) / y_scale
    dz = (local[:, 2] - target_z) / z_scale
    score = 1.0 / (1.0 + dx.square() + dy.square() + dz.square())

    reward_window_steps = max(1, int(round(reward_window_s / env.step_dt)))
    remaining_steps = env.max_episode_length - env.episode_length_buf
    terminal_window = (remaining_steps >= 1) & (
        remaining_steps <= reward_window_steps
    )
    lateral_corridor = local[:, 1].abs() <= max_abs_y
    return torch.where(
        terminal_window & lateral_corridor & finite,
        score / reward_window_s,
        torch.zeros_like(score),
    )


def stair_delayed_frontier_tiers(
    env: ManagerBasedRlEnv,
    x_thresholds: tuple[float, ...] = (0.650, 0.660, 0.665),
    tier_rewards: tuple[float, ...] = (25.0, 75.0, 150.0),
    min_height: float = 0.175,
    min_height_thresholds: tuple[float, ...] | None = None,
    corridor_half_width: float = 0.20,
    hold_steps: int = 2,
    min_policy_steps: int = 3,
    bypass_x: float = 0.660,
    bypass_half_width: float = 0.36,
    prelatch_reset_satisfied: bool = False,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Pay once for held, post-control progress beyond a replay frontier.

    Frontier replay rows are deliberately below the first threshold. The
    control-step delay and per-tier latches prevent PPO from collecting a
    reset jackpot, while cumulative tiers preserve credit for a motion that
    crosses more than one threshold in a single control interval. A lateral
    bypass permanently disables every unpaid tier for that episode.
    """

    if len(x_thresholds) == 0 or len(x_thresholds) != len(tier_rewards):
        raise ValueError(
            "Frontier thresholds and rewards must be non-empty and aligned"
        )
    if tuple(sorted(x_thresholds)) != x_thresholds:
        raise ValueError("Frontier x thresholds must be ascending")
    if any(reward <= 0.0 for reward in tier_rewards):
        raise ValueError("Frontier tier rewards must be positive")
    if min_height_thresholds is not None and len(min_height_thresholds) != len(
        x_thresholds
    ):
        raise ValueError("Frontier height thresholds must align with x thresholds")
    if hold_steps < 1 or min_policy_steps < 1:
        raise ValueError("Frontier hold and policy-step delays must be positive")
    if min(corridor_half_width, bypass_half_width) <= 0.0:
        raise ValueError("Frontier lateral widths must be positive")

    asset: Entity = env.scene[asset_cfg.name]
    local = asset.data.root_link_pos_w - env.scene.terrain.env_origins
    x = local[:, 0]
    z = local[:, 2]
    abs_y = local[:, 1].abs()
    finite = torch.isfinite(local).all(dim=-1)
    tier_count = len(x_thresholds)

    if not hasattr(env, "_stair_delayed_frontier_tier_hold"):
        env._stair_delayed_frontier_tier_hold = torch.zeros(
            (env.num_envs, tier_count), dtype=torch.long, device=env.device
        )
        env._stair_delayed_frontier_tier_paid = torch.zeros(
            (env.num_envs, tier_count), dtype=torch.bool, device=env.device
        )
        env._stair_delayed_frontier_bypassed = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )

    fresh = env.episode_length_buf <= 1
    env._stair_delayed_frontier_tier_hold[fresh] = 0
    env._stair_delayed_frontier_tier_paid[fresh] = False
    if prelatch_reset_satisfied:
        reset_satisfied = getattr(
            env, "_stair_reset_frontier_tiers_satisfied", None
        )
        expected_shape = (env.num_envs, tier_count)
        if reset_satisfied is None or reset_satisfied.shape != expected_shape:
            raise RuntimeError(
                "Reset-safe frontier tiers require a matching reset snapshot event"
            )
        env._stair_delayed_frontier_tier_paid[fresh] = reset_satisfied[fresh]
    env._stair_delayed_frontier_bypassed[fresh] = False

    bypass = finite & (x >= bypass_x) & (abs_y > bypass_half_width)
    env._stair_delayed_frontier_bypassed |= bypass
    armed = (
        finite
        & (env.episode_length_buf >= min_policy_steps)
        & ~env._stair_delayed_frontier_bypassed
    )
    thresholds = torch.tensor(x_thresholds, dtype=x.dtype, device=x.device)
    height_thresholds = torch.tensor(
        (
            (min_height,) * tier_count
            if min_height_thresholds is None
            else min_height_thresholds
        ),
        dtype=z.dtype,
        device=z.device,
    )
    candidates = (
        armed[:, None]
        & (x[:, None] >= thresholds[None, :])
        & (z[:, None] >= height_thresholds[None, :])
        & (abs_y[:, None] <= corridor_half_width)
    )
    env._stair_delayed_frontier_tier_hold = torch.where(
        candidates,
        env._stair_delayed_frontier_tier_hold + 1,
        torch.zeros_like(env._stair_delayed_frontier_tier_hold),
    )
    newly_reached = (
        env._stair_delayed_frontier_tier_hold >= hold_steps
    ) & ~env._stair_delayed_frontier_tier_paid
    env._stair_delayed_frontier_tier_paid |= newly_reached
    rewards = torch.tensor(tier_rewards, dtype=x.dtype, device=x.device)
    return torch.sum(newly_reached.to(x.dtype) * rewards[None, :], dim=-1)


def snapshot_stair_frontier_tier_baseline(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    x_thresholds: tuple[float, ...] = (0.625, 0.640, 0.650, 0.660, 0.665),
    min_height_thresholds: tuple[float, ...] = (
        0.160,
        0.170,
        0.175,
        0.175,
        0.175,
    ),
    corridor_half_width: float = 0.20,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """Snapshot milestone tiers already satisfied by an exact reset state."""

    if len(x_thresholds) == 0 or len(x_thresholds) != len(
        min_height_thresholds
    ):
        raise ValueError("Reset milestone thresholds must be non-empty and aligned")
    if corridor_half_width <= 0.0:
        raise ValueError("Reset milestone corridor must be positive")
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env_ids = env_ids.to(device=env.device, dtype=torch.long)
    asset: Entity = env.scene[asset_cfg.name]
    # Reset events run before MuJoCo forward kinematics refresh derived xpos.
    # State-bank events immediately preceding this one write the free-joint
    # qpos directly, so qpos is the only authoritative reset pose here.
    root_qpos = env.sim.data.qpos[env_ids][
        :, asset.indexing.free_joint_q_adr
    ]
    local = root_qpos[:, :3] - env.scene.terrain.env_origins[env_ids]
    x_threshold = torch.tensor(
        x_thresholds, dtype=local.dtype, device=local.device
    )
    z_threshold = torch.tensor(
        min_height_thresholds, dtype=local.dtype, device=local.device
    )
    satisfied = (
        torch.isfinite(local).all(dim=-1)[:, None]
        & (local[:, 0, None] >= x_threshold[None, :])
        & (local[:, 2, None] >= z_threshold[None, :])
        & (local[:, 1].abs()[:, None] <= corridor_half_width)
    )
    expected_shape = (env.num_envs, len(x_thresholds))
    current = getattr(env, "_stair_reset_frontier_tiers_satisfied", None)
    if current is None or current.shape != expected_shape:
        env._stair_reset_frontier_tiers_satisfied = torch.zeros(
            expected_shape, dtype=torch.bool, device=env.device
        )
    env._stair_reset_frontier_tiers_satisfied[env_ids] = satisfied


def stair_apex_or_mantle_frontier(
    env: ManagerBasedRlEnv,
    approach_start_x: float = 0.40,
    stair_face_x: float = 0.66,
    crossing_end_x: float = 0.70,
    standing_root_height: float = 0.115,
    clearance_root_height: float = 0.195,
    max_vertical_speed: float = 1.50,
    corridor_half_width: float = 0.36,
    riser_height: float = 0.17,
    tread_depth: float = 0.28,
    support_sensor_name: str = "robot_ground_contact",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward new ballistic-clearance or contact-mantle frontiers.

    The actor is free to jump directly or exploit the head and shell as a
    temporary support. Ballistic height is predicted at the riser face, rather
    than at an unconstrained apex that could occur before or after the stair.
    The mantle branch pays only for a classified face or tread contact.
    """
    asset: Entity = env.scene[asset_cfg.name]
    x, z, _ = _stair_local_state(env, asset_cfg)
    y = torch.abs(
        asset.data.root_link_pos_w[:, 1] - env.scene.terrain.env_origins[:, 1]
    )
    vertical_speed = torch.clamp(
        torch.nan_to_num(asset.data.root_link_lin_vel_w[:, 2], nan=0.0),
        min=0.0,
        max=max_vertical_speed,
    )
    forward_speed = torch.nan_to_num(
        asset.data.root_link_lin_vel_w[:, 0], nan=0.0
    )
    time_to_face = torch.clamp(
        (stair_face_x - x) / torch.clamp(forward_speed, min=0.05),
        min=0.0,
        max=0.60,
    )
    predicted_face_height = (
        z + vertical_speed * time_to_face - 0.5 * 9.81 * time_to_face.square()
    )
    predicted_face_height = torch.where(
        (x < stair_face_x) & (forward_speed > 0.05),
        predicted_face_height,
        z,
    )
    approach_gate = torch.clamp(
        (x - approach_start_x) / max(stair_face_x - approach_start_x, 1e-6),
        min=0.0,
        max=1.0,
    )
    height_progress = torch.clamp(
        (predicted_face_height - standing_root_height)
        / max(clearance_root_height - standing_root_height, 1e-6),
        min=0.0,
        max=1.0,
    )
    ballistic = approach_gate * height_progress

    face_contact, tread_contact, _ = _standard_stair_contact_masks(
        env,
        support_sensor_name,
        stair_face_x=stair_face_x,
        riser_height=riser_height,
        tread_depth=tread_depth,
        corridor_half_width=corridor_half_width,
    )
    contact = face_contact.any(dim=-1) | tread_contact.any(dim=-1)
    crossing_progress = torch.clamp(
        (x - stair_face_x) / max(crossing_end_x - stair_face_x, 1e-6),
        min=0.0,
        max=1.0,
    )
    current_height_progress = torch.clamp(
        (z - standing_root_height)
        / max(clearance_root_height - standing_root_height, 1e-6),
        min=0.0,
        max=1.0,
    )
    mantle = contact.to(x.dtype) * 0.5 * (
        current_height_progress + crossing_progress
    )
    value = torch.maximum(ballistic, mantle)
    value = torch.where(y <= corridor_half_width, value, torch.zeros_like(value))
    return _new_frontier_delta(env, value, "_stair_apex_or_mantle_frontier")


def stair_assisted_approach_frontier(
    env: ManagerBasedRlEnv,
    start_x: float = 0.54,
    end_x: float = 0.64,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    x, _, _ = _stair_local_state(env, asset_cfg)
    value = torch.clamp((x - start_x) / max(end_x - start_x, 1e-6), 0.0, 1.0)
    return _new_frontier_delta(env, value, "_stair_assisted_approach")


def stair_assisted_lift_frontier(
    env: ManagerBasedRlEnv,
    start_height: float = 0.115,
    clearance_height: float = 0.195,
    x_gate: float = 0.60,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    x, z, _ = _stair_local_state(env, asset_cfg)
    value = torch.clamp(
        (z - start_height) / max(clearance_height - start_height, 1e-6), 0.0, 1.0
    ) * torch.sigmoid((x - x_gate) / 0.015)
    return _new_frontier_delta(env, value, "_stair_assisted_lift")


def stair_assisted_crossing_frontier(
    env: ManagerBasedRlEnv,
    start_x: float = 0.64,
    end_x: float = 0.72,
    clearance_height: float = 0.190,
    corridor_half_width: float | None = None,
    hard_height_gate: bool = False,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    x, z, _ = _stair_local_state(env, asset_cfg)
    height_gate = (
        (z >= clearance_height).to(z.dtype)
        if hard_height_gate
        else torch.sigmoid((z - clearance_height) / 0.008)
    )
    value = torch.clamp((x - start_x) / max(end_x - start_x, 1e-6), 0.0, 1.0)
    value *= height_gate
    if corridor_half_width is not None:
        asset: Entity = env.scene[asset_cfg.name]
        y = torch.abs(
            asset.data.root_link_pos_w[:, 1] - env.scene.terrain.env_origins[:, 1]
        )
        value = torch.where(
            y <= corridor_half_width, value, torch.zeros_like(value)
        )
    return _new_frontier_delta(env, value, "_stair_assisted_crossing")


def stair_tread_support_frontier(
    env: ManagerBasedRlEnv,
    stair_face_x: float = 0.66,
    target_x: float = 0.74,
    riser_height: float = 0.17,
    tread_depth: float = 0.28,
    target_root_height: float = 0.205,
    corridor_half_width: float = 0.36,
    support_sensor_name: str = "robot_ground_contact",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward new contact-supported progress onto the first real tread.

    Forward translation alone is insufficient. The robot must remain inside
    the stair corridor, place its root beyond the vertical face and above the
    170 mm lip, and have actual robot-terrain contact. This makes a foot plant,
    shell mantle, or head-assisted lever valid while rejecting a sideways fall
    that merely produces a large global x coordinate.
    """
    asset: Entity = env.scene[asset_cfg.name]
    x, z, _ = _stair_local_state(env, asset_cfg)
    y = torch.abs(
        asset.data.root_link_pos_w[:, 1] - env.scene.terrain.env_origins[:, 1]
    )
    _, tread_contact, contact_positions = _standard_stair_contact_masks(
        env,
        support_sensor_name,
        stair_face_x=stair_face_x,
        riser_height=riser_height,
        tread_depth=tread_depth,
        corridor_half_width=corridor_half_width,
    )
    support = tread_contact.any(dim=-1)
    contact_x = torch.where(
        tread_contact,
        contact_positions[..., 0],
        torch.full_like(contact_positions[..., 0], stair_face_x),
    ).max(dim=-1).values

    x_progress = torch.clamp(
        (contact_x - stair_face_x) / max(target_x - stair_face_x, 1e-6),
        0.0,
        1.0,
    )
    z_progress = torch.clamp(
        (z - riser_height)
        / max(target_root_height - riser_height, 1e-6),
        0.0,
        1.0,
    )
    value = 0.40 + 0.30 * x_progress + 0.30 * z_progress
    eligible = (
        support
        & (y <= corridor_half_width)
    )
    value = torch.where(eligible, value, torch.zeros_like(value))
    return _new_frontier_delta(env, value, "_stair_tread_support_frontier")


def stair_tread_pullup_frontier(
    env: ManagerBasedRlEnv,
    start_x: float = 0.56,
    target_x: float = 0.72,
    start_height: float = 0.10,
    target_height: float = 0.205,
    corridor_half_width: float = 0.36,
    stair_face_x: float = 0.66,
    riser_height: float = 0.17,
    tread_depth: float = 0.28,
    support_sensor_name: str = "robot_ground_contact",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward new root lift and advance after a real first-tread contact.

    This is the continuation objective for contact-state reverse curriculum.
    A limb, shell, or head may release the tread after creating leverage, but
    progress is eligible only after the classified tread-contact latch fired.
    """

    asset: Entity = env.scene[asset_cfg.name]
    x, z, _ = _stair_local_state(env, asset_cfg)
    y = torch.abs(
        asset.data.root_link_pos_w[:, 1] - env.scene.terrain.env_origins[:, 1]
    )
    _, tread_contact, _ = _standard_stair_contact_masks(
        env,
        support_sensor_name,
        stair_face_x=stair_face_x,
        riser_height=riser_height,
        tread_depth=tread_depth,
        corridor_half_width=corridor_half_width,
    )
    current_contact = tread_contact.any(dim=-1)
    contacted = getattr(env, "_stair_tread_pullup_contact_latched", None)
    if contacted is None:
        contacted = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._stair_tread_pullup_contact_latched = contacted
    fresh = env.episode_length_buf <= 1
    contacted[fresh] = False
    contacted |= current_contact
    x_progress = torch.clamp(
        (x - start_x) / max(target_x - start_x, 1e-6), 0.0, 1.0
    )
    height_progress = torch.clamp(
        (z - start_height) / max(target_height - start_height, 1e-6),
        0.0,
        1.0,
    )
    value = 0.65 * x_progress + 0.35 * height_progress
    eligible = contacted & (y <= corridor_half_width)
    value = torch.where(eligible, value, torch.zeros_like(value))
    return _new_frontier_delta(env, value, "_stair_tread_pullup_frontier")


def stair_foot_anchor_vault_frontier(
    env: ManagerBasedRlEnv,
    start_x: float = 0.60,
    target_x: float = 0.72,
    start_height: float = 0.12,
    target_height: float = 0.205,
    min_normal_force: float = 0.40,
    min_positive_power: float = 0.01,
    target_positive_power: float = 0.25,
    release_window_s: float = 0.25,
    corridor_half_width: float = 0.36,
    stair_face_x: float = 0.66,
    riser_height: float = 0.17,
    tread_depth: float = 0.28,
    support_sensor_name: str = "feet_stair_contact",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward a loaded foot plant that performs positive vault work.

    The A9 geometry audit showed that almost every real first-tread contact is
    a foot, not a head or shell. This continuation therefore treats the foot
    as a temporary anchor. Root progress is eligible only after a loaded tread
    contact produces positive sagittal mechanical power, with a short release
    window so the foot can leave the tread during the vault.
    """

    asset: Entity = env.scene[asset_cfg.name]
    x, z, _ = _stair_local_state(env, asset_cfg)
    y = torch.abs(
        asset.data.root_link_pos_w[:, 1] - env.scene.terrain.env_origins[:, 1]
    )
    if support_sensor_name not in env.scene.sensors:
        return torch.zeros(env.num_envs, device=env.device)
    sensor_data = env.scene.sensors[support_sensor_name].data
    if (
        sensor_data.found is None
        or sensor_data.force is None
        or sensor_data.pos is None
        or sensor_data.normal is None
    ):
        raise RuntimeError(
            f"{support_sensor_name} must provide found, force, pos, and normal"
        )
    _, tread_contact, _ = _standard_stair_contact_masks(
        env,
        support_sensor_name,
        stair_face_x=stair_face_x,
        riser_height=riser_height,
        tread_depth=tread_depth,
        corridor_half_width=corridor_half_width,
    )
    normal_force = torch.abs(
        torch.sum(sensor_data.force * sensor_data.normal, dim=-1)
    )
    normal_force = torch.where(
        tread_contact, normal_force, torch.zeros_like(normal_force)
    )
    strongest_slot = normal_force.argmax(dim=-1, keepdim=True)
    strongest_force = normal_force.gather(1, strongest_slot).squeeze(1)
    slot_index = strongest_slot.unsqueeze(-1).expand(-1, -1, 3)
    contact_pos = sensor_data.pos.gather(1, slot_index).squeeze(1)
    contact_force = sensor_data.force.gather(1, slot_index).squeeze(1)
    lever = contact_pos - asset.data.root_link_pos_w
    pitch_torque = torch.cross(lever, contact_force, dim=-1)[:, 1]
    positive_power = torch.clamp(
        pitch_torque * asset.data.root_link_ang_vel_w[:, 1], min=0.0
    )
    loaded = tread_contact.any(dim=-1) & (strongest_force >= min_normal_force)

    if not hasattr(env, "_stair_foot_anchor_release_steps"):
        env._stair_foot_anchor_release_steps = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
    fresh = env.episode_length_buf <= 1
    env._stair_foot_anchor_release_steps[fresh] = 0
    work = loaded & (positive_power >= min_positive_power)
    release_steps = max(1, int(round(release_window_s / env.step_dt)))
    env._stair_foot_anchor_release_steps = torch.where(
        work,
        torch.full_like(env._stair_foot_anchor_release_steps, release_steps),
        torch.clamp(env._stair_foot_anchor_release_steps - 1, min=0),
    )
    active_vault = env._stair_foot_anchor_release_steps > 0

    power_progress = torch.clamp(
        positive_power / max(target_positive_power, 1e-6), 0.0, 1.0
    )
    x_progress = torch.clamp(
        (x - start_x) / max(target_x - start_x, 1e-6), 0.0, 1.0
    )
    height_progress = torch.clamp(
        (z - start_height) / max(target_height - start_height, 1e-6),
        0.0,
        1.0,
    )
    root_progress = 0.60 * x_progress + 0.40 * height_progress
    value = 0.25 * power_progress + 0.75 * root_progress
    eligible = active_vault & (y <= corridor_half_width)
    value = torch.where(eligible, value, torch.zeros_like(value))
    return _new_frontier_delta(env, value, "_stair_foot_anchor_vault_frontier")


def stair_ordered_foot_vault_frontier(
    env: ManagerBasedRlEnv,
    target_x: float = 0.72,
    target_height: float = 0.205,
    required_lift: float = 0.04,
    lip_gate_height: float = 0.165,
    min_normal_force: float = 0.40,
    min_positive_power: float = 0.02,
    min_positive_work_j: float = 0.004,
    min_control_steps: int = 3,
    corridor_half_width: float = 0.20,
    stair_face_x: float = 0.66,
    riser_height: float = 0.17,
    tread_depth: float = 0.28,
    support_sensor_name: str = "feet_stair_contact",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward an ordered loaded-work, lift, then lip-crossing sequence.

    A11 could earn its frontier while oscillating around a static foot anchor.
    This stage uses contact power only as an eligibility event, never as direct
    reward. The root baseline is captured after the policy has controlled the
    robot for several steps and accumulated real positive pitch work. Only
    subsequent lift earns the first half of the frontier; forward crossing is
    multiplied by an absolute lip-height gate.
    """

    asset: Entity = env.scene[asset_cfg.name]
    x, z, _ = _stair_local_state(env, asset_cfg)
    y = torch.abs(
        asset.data.root_link_pos_w[:, 1] - env.scene.terrain.env_origins[:, 1]
    )
    if support_sensor_name not in env.scene.sensors:
        return torch.zeros(env.num_envs, device=env.device)
    sensor_data = env.scene.sensors[support_sensor_name].data
    if (
        sensor_data.found is None
        or sensor_data.force is None
        or sensor_data.pos is None
        or sensor_data.normal is None
    ):
        raise RuntimeError(
            f"{support_sensor_name} must provide found, force, pos, and normal"
        )
    _, tread_contact, _ = _standard_stair_contact_masks(
        env,
        support_sensor_name,
        stair_face_x=stair_face_x,
        riser_height=riser_height,
        tread_depth=tread_depth,
        corridor_half_width=corridor_half_width,
    )
    normal_force = torch.abs(
        torch.sum(sensor_data.force * sensor_data.normal, dim=-1)
    )
    normal_force = torch.where(
        tread_contact, normal_force, torch.zeros_like(normal_force)
    )
    strongest_slot = normal_force.argmax(dim=-1, keepdim=True)
    strongest_force = normal_force.gather(1, strongest_slot).squeeze(1)
    slot_index = strongest_slot.unsqueeze(-1).expand(-1, -1, 3)
    contact_pos = sensor_data.pos.gather(1, slot_index).squeeze(1)
    contact_force = sensor_data.force.gather(1, slot_index).squeeze(1)
    lever = contact_pos - asset.data.root_link_pos_w
    pitch_torque = torch.cross(lever, contact_force, dim=-1)[:, 1]
    positive_power = torch.clamp(
        pitch_torque * asset.data.root_link_ang_vel_w[:, 1], min=0.0
    )
    loaded = tread_contact.any(dim=-1) & (strongest_force >= min_normal_force)

    if not hasattr(env, "_stair_ordered_vault_armed"):
        env._stair_ordered_vault_armed = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        env._stair_ordered_vault_work_j = torch.zeros(
            env.num_envs, device=env.device
        )
        env._stair_ordered_vault_start_x = x.clone()
        env._stair_ordered_vault_start_z = z.clone()
    fresh = env.episode_length_buf <= 1
    env._stair_ordered_vault_armed[fresh] = False
    env._stair_ordered_vault_work_j[fresh] = 0.0
    env._stair_ordered_vault_start_x[fresh] = x[fresh]
    env._stair_ordered_vault_start_z[fresh] = z[fresh]

    controlled = env.episode_length_buf >= max(1, int(min_control_steps))
    positive_work_step = torch.where(
        controlled & loaded & (positive_power >= min_positive_power),
        positive_power * env.step_dt,
        torch.zeros_like(positive_power),
    )
    not_armed = ~env._stair_ordered_vault_armed
    env._stair_ordered_vault_work_j += positive_work_step * not_armed
    newly_armed = not_armed & (
        env._stair_ordered_vault_work_j >= min_positive_work_j
    )
    env._stair_ordered_vault_start_x[newly_armed] = x[newly_armed]
    env._stair_ordered_vault_start_z[newly_armed] = z[newly_armed]
    env._stair_ordered_vault_armed |= newly_armed

    lift_progress = torch.clamp(
        (z - env._stair_ordered_vault_start_z) / max(required_lift, 1e-6),
        0.0,
        1.0,
    )
    cross_span = torch.clamp(
        torch.full_like(x, target_x) - env._stair_ordered_vault_start_x,
        min=0.03,
    )
    cross_progress = torch.clamp(
        (x - env._stair_ordered_vault_start_x) / cross_span, 0.0, 1.0
    )
    lip_gate = torch.clamp(
        (z - lip_gate_height) / max(target_height - lip_gate_height, 1e-6),
        0.0,
        1.0,
    )
    value = 0.55 * lift_progress + 0.45 * cross_progress * lip_gate
    eligible = env._stair_ordered_vault_armed & (y <= corridor_half_width)
    value = torch.where(eligible, value, torch.zeros_like(value))
    return _new_frontier_delta(env, value, "_stair_ordered_foot_vault_frontier")


def stair_first_tread_stable(
    env: ManagerBasedRlEnv,
    min_x: float = 0.72,
    max_x: float = 0.94,
    min_height: float = 0.235,
    upright_threshold: float = 0.82,
    max_vertical_speed: float = 0.15,
    max_angular_speed: float = 1.0,
    hold_time_s: float = 0.20,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    x, z, upright = _stair_local_state(env, asset_cfg)
    vertical_speed = torch.abs(asset.data.root_link_lin_vel_w[:, 2])
    angular_speed = torch.linalg.vector_norm(asset.data.root_link_ang_vel_w, dim=-1)
    candidate = (
        (x >= min_x)
        & (x <= max_x)
        & (z >= min_height)
        & (upright >= upright_threshold)
        & (vertical_speed <= max_vertical_speed)
        & (angular_speed <= max_angular_speed)
    )
    if not hasattr(env, "_stair_first_tread_hold_steps"):
        env._stair_first_tread_hold_steps = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
        env._stair_first_tread_latched = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
    fresh = env.episode_length_buf <= 1
    env._stair_first_tread_hold_steps[fresh] = 0
    env._stair_first_tread_latched[fresh] = False
    env._stair_first_tread_hold_steps = torch.where(
        candidate,
        env._stair_first_tread_hold_steps + 1,
        torch.zeros_like(env._stair_first_tread_hold_steps),
    )
    required_steps = max(1, int(round(hold_time_s / env.step_dt)))
    newly_stable = (
        env._stair_first_tread_hold_steps >= required_steps
    ) & ~env._stair_first_tread_latched
    env._stair_first_tread_latched |= newly_stable
    return newly_stable.float()


def stair_first_tread_secured(
    env: ManagerBasedRlEnv,
    min_x: float = 0.70,
    max_x: float = 0.94,
    min_height: float = 0.195,
    max_linear_speed: float = 0.40,
    max_angular_speed: float = 2.5,
    hold_time_s: float = 0.12,
    support_sensor_name: str = "robot_ground_contact",
    stair_face_x: float = 0.66,
    riser_height: float = 0.17,
    tread_depth: float = 0.28,
    corridor_half_width: float = 0.36,
    required_latch_name: str | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Pay once when any mechanically settled pose is secured on tread one.

    Unlike ``stair_first_tread_stable``, this bridge milestone deliberately has
    no upright requirement.  A shell landing or head-supported brace is valid
    if the robot is fully beyond the 170 mm lip and has stopped tumbling.
    """
    asset: Entity = env.scene[asset_cfg.name]
    x, z, _ = _stair_local_state(env, asset_cfg)
    linear_speed = torch.linalg.vector_norm(asset.data.root_link_lin_vel_w, dim=-1)
    angular_speed = torch.linalg.vector_norm(asset.data.root_link_ang_vel_w, dim=-1)
    _, tread_contact, _ = _standard_stair_contact_masks(
        env,
        support_sensor_name,
        stair_face_x=stair_face_x,
        riser_height=riser_height,
        tread_depth=tread_depth,
        corridor_half_width=corridor_half_width,
    )
    support = tread_contact.any(dim=-1)
    ordered_gate = torch.ones_like(support)
    if required_latch_name is not None:
        required_latch = getattr(env, required_latch_name, None)
        if required_latch is None:
            ordered_gate = torch.zeros_like(support)
        else:
            ordered_gate = required_latch
    candidate = (
        (x >= min_x)
        & (x <= max_x)
        & (z >= min_height)
        & (linear_speed <= max_linear_speed)
        & (angular_speed <= max_angular_speed)
        & support
        & ordered_gate
    )
    if not hasattr(env, "_stair_first_tread_secured_steps"):
        env._stair_first_tread_secured_steps = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
        env._stair_first_tread_secured_latched = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
    fresh = env.episode_length_buf <= 1
    env._stair_first_tread_secured_steps[fresh] = 0
    env._stair_first_tread_secured_latched[fresh] = False
    env._stair_first_tread_secured_steps = torch.where(
        candidate,
        env._stair_first_tread_secured_steps + 1,
        torch.zeros_like(env._stair_first_tread_secured_steps),
    )
    required_steps = max(1, int(round(hold_time_s / env.step_dt)))
    newly_secured = (
        env._stair_first_tread_secured_steps >= required_steps
    ) & ~env._stair_first_tread_secured_latched
    env._stair_first_tread_secured_latched |= newly_secured
    return newly_secured.float()


def stair_secured_tread_success(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Terminate only after the strict secured-tread hold has completed."""

    secured = getattr(env, "_stair_first_tread_secured_latched", None)
    if secured is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return secured


def stair_progress_stalled(
    env: ManagerBasedRlEnv,
    start_x: float = 0.50,
    target_x: float = 0.70,
    start_height: float = 0.09,
    target_height: float = 0.198,
    warmup_s: float = 0.50,
    max_stall_s: float = 1.10,
    min_progress_delta: float = 0.01,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Recycle attempts that stop improving before they secure the tread.

    The timer observes a monotonic physical potential plus ordered contact
    milestones. It does not require upright posture, so head levers and rolls
    remain valid. Reaching a new milestone resets the full settling window.
    """

    if target_x <= start_x or target_height <= start_height:
        raise ValueError("Stair stall targets must exceed their start values")
    if warmup_s < 0.0 or max_stall_s <= 0.0 or min_progress_delta <= 0.0:
        raise ValueError("Stair stall timing and progress delta must be positive")

    x, z, _ = _stair_local_state(env, asset_cfg)
    x_progress = torch.clamp((x - start_x) / (target_x - start_x), 0.0, 1.0)
    z_progress = torch.clamp(
        (z - start_height) / (target_height - start_height), 0.0, 1.0
    )

    def latch(name: str) -> torch.Tensor:
        value = getattr(env, name, None)
        if value is None:
            return torch.zeros(env.num_envs, device=env.device)
        return value.float()

    progress = (
        0.35 * x_progress
        + 0.25 * z_progress
        + 0.10 * latch("_stair_contact_transfer_stage15_policy_achieved")
        + 0.15 * latch("_stair_contact_transfer_stage2_policy_achieved")
        + 0.15 * latch("_stair_true_shell_clearance_policy_achieved")
    )
    if not hasattr(env, "_stair_stall_best_progress"):
        env._stair_stall_best_progress = torch.zeros(
            env.num_envs, device=env.device
        )
        env._stair_stall_timer_s = torch.zeros(env.num_envs, device=env.device)

    fresh = env.episode_length_buf <= 1
    env._stair_stall_best_progress[fresh] = progress[fresh]
    env._stair_stall_timer_s[fresh] = 0.0
    improved = progress >= env._stair_stall_best_progress + min_progress_delta
    improved &= ~fresh
    env._stair_stall_best_progress = torch.maximum(
        env._stair_stall_best_progress, progress
    )
    env._stair_stall_timer_s = torch.where(
        improved,
        torch.zeros_like(env._stair_stall_timer_s),
        env._stair_stall_timer_s + env.step_dt,
    )
    env._stair_stall_timer_s[fresh] = 0.0

    secured = getattr(env, "_stair_first_tread_secured_latched", None)
    if secured is None:
        secured = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    age_s = env.episode_length_buf.float() * env.step_dt
    return (
        (age_s >= warmup_s)
        & (env._stair_stall_timer_s >= max_stall_s)
        & ~secured
    )


def stair_first_tread_settle_quality(
    env: ManagerBasedRlEnv,
    min_x: float = 0.70,
    max_x: float = 0.94,
    min_height: float = 0.195,
    linear_speed_scale: float = 0.35,
    angular_speed_scale: float = 2.5,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Dense bridge shaping for slowing down after a full-height mantle."""
    asset: Entity = env.scene[asset_cfg.name]
    x, z, _ = _stair_local_state(env, asset_cfg)
    x_gate = torch.sigmoid((x - min_x) / 0.015) * torch.sigmoid(
        (max_x - x) / 0.020
    )
    z_gate = torch.sigmoid((z - min_height) / 0.010)
    linear_speed = torch.linalg.vector_norm(asset.data.root_link_lin_vel_w, dim=-1)
    angular_speed = torch.linalg.vector_norm(asset.data.root_link_ang_vel_w, dim=-1)
    settle = 0.6 * torch.exp(-((linear_speed / linear_speed_scale) ** 2))
    settle += 0.4 * torch.exp(-((angular_speed / angular_speed_scale) ** 2))
    return x_gate * z_gate * settle


def stair_goal_progress(
    env: ManagerBasedRlEnv,
    goal_distance: float,
    corridor_half_width: float | None = None,
    corridor_fade: float = 0.08,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Potential-based forward progress toward the stair top.

    The robot receives the change in clamped route progress, not a per-step
    reward for merely remaining near the destination.  Backward motion is
    charged, holding position is zero, and episode resets clear the potential.
    This gives the policy a useful walking signal while avoiding a goal-state
    jackpot that could reward thrashing in place.
    """
    x, _, _ = _stair_local_state(env, asset_cfg)
    progress = torch.clamp(x / max(goal_distance, 1e-6), min=0.0, max=1.0)
    progress = progress * _stair_corridor_gate(
        env, corridor_half_width, corridor_fade, asset_cfg
    )
    if not hasattr(env, "_stair_progress_prev"):
        env._stair_progress_prev = progress.clone()
    fresh = env.episode_length_buf <= 1
    env._stair_progress_prev[fresh] = progress[fresh]
    delta = progress - env._stair_progress_prev
    env._stair_progress_prev = progress.clone()
    return torch.clamp(delta / env.step_dt, min=-5.0, max=5.0)


def stair_top_approach(
    env: ManagerBasedRlEnv,
    goal_distance: float,
    goal_height: float | None = None,
    goal_height_range: tuple[float, float] | None = None,
    num_terrain_levels: int | None = None,
    start_height: float = 0.115,
    route_gate_start: float = 0.20,
    route_gate_width: float = 0.08,
    upright_power: float = 1.0,
    corridor_half_width: float | None = None,
    corridor_fade: float = 0.08,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Potential-based shaping for climbing after entering the staircase.

    Progress along the route gates a normalized height potential.  This gives
    the robot a learnable signal for getting onto higher treads, while the
    separate one-shot goal term below remains the definition of success.
    """
    x, z, upright = _stair_local_state(env, asset_cfg)
    progress = torch.clamp(x / max(goal_distance, 1e-6), min=0.0, max=1.0)
    route_gate = torch.sigmoid((progress - route_gate_start) / route_gate_width)
    target_height = _stair_target_height(
        env, goal_height, goal_height_range, num_terrain_levels, z
    )
    height = torch.clamp(
        (z - start_height) / torch.clamp(target_height - start_height, min=1e-6),
        min=0.0,
        max=1.0,
    )
    posture_gate = (
        torch.ones_like(upright)
        if upright_power == 0.0
        else torch.pow(upright, upright_power)
    )
    corridor_gate = _stair_corridor_gate(
        env, corridor_half_width, corridor_fade, asset_cfg
    )
    potential = route_gate * height * posture_gate * corridor_gate
    if not hasattr(env, "_stair_top_potential_prev"):
        env._stair_top_potential_prev = potential.clone()
    fresh = env.episode_length_buf <= 1
    env._stair_top_potential_prev[fresh] = potential[fresh]
    delta = potential - env._stair_top_potential_prev
    env._stair_top_potential_prev = potential.clone()
    return torch.clamp(delta / env.step_dt, min=-5.0, max=5.0)


def stair_top_goal(
    env: ManagerBasedRlEnv,
    goal_distance: float,
    goal_height: float | None = None,
    goal_height_range: tuple[float, float] | None = None,
    num_terrain_levels: int | None = None,
    x_tolerance: float = 0.20,
    z_tolerance: float = 0.13,
    upright_threshold: float = 0.78,
    corridor_half_width: float | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """One-shot success signal for arriving upright on the top landing.

    The success state requires the route endpoint, the expected top-level
    height, and a non-inverted trunk.  A per-episode latch makes the reward
    one-shot, so standing on the landing cannot be farmed indefinitely.
    """
    x, z, upright = _stair_local_state(env, asset_cfg)
    # The goal is the top landing, not one exact pixel on its far edge.  Once
    # the robot has crossed the near goal boundary, any further position on
    # the landing remains valid.
    x_ok = x >= goal_distance - x_tolerance
    target_height = _stair_target_height(
        env, goal_height, goal_height_range, num_terrain_levels, z
    )
    z_ok = torch.abs(z - target_height) <= z_tolerance
    asset: Entity = env.scene[asset_cfg.name]
    y = torch.abs(
        asset.data.root_link_pos_w[:, 1] - env.scene.terrain.env_origins[:, 1]
    )
    y_ok = (
        torch.ones_like(x_ok)
        if corridor_half_width is None
        else y <= corridor_half_width
    )
    success = (x >= goal_distance - x_tolerance) & x_ok & y_ok & z_ok & (
        upright >= upright_threshold
    )
    if not hasattr(env, "_stair_goal_latched"):
        env._stair_goal_latched = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
    fresh = env.episode_length_buf <= 1
    env._stair_goal_latched[fresh] = False
    newly_reached = success & ~env._stair_goal_latched
    env._stair_goal_latched |= success
    return newly_reached.float()


def _stair_target_height(
    env: ManagerBasedRlEnv,
    goal_height: float | None,
    goal_height_range: tuple[float, float] | None,
    num_terrain_levels: int | None,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Resolve fixed or terrain-level-dependent stair landing height."""
    if goal_height_range is None:
        if goal_height is None:
            raise ValueError("A fixed goal_height or goal_height_range is required.")
        return torch.full_like(reference, goal_height)
    low, high = goal_height_range
    terrain = env.scene.terrain
    levels = terrain.terrain_levels.to(dtype=reference.dtype)
    level_count = num_terrain_levels
    if level_count is None and terrain.terrain_origins is not None:
        level_count = terrain.terrain_origins.shape[0]
    denominator = max((level_count or 1) - 1, 1)
    fraction = torch.clamp(levels / denominator, min=0.0, max=1.0)
    return low + fraction * (high - low)


def route_terrain_levels(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Promote only after latched top-landing success, never mere distance."""
    terrain = env.scene.terrain
    if terrain.terrain_origins is None:
        zero = torch.tensor(0.0, device=env.device)
        return {"mean": zero, "max": zero}
    goal_latched = getattr(env, "_stair_goal_latched", None)
    first_riser_latched = getattr(env, "_stair_first_riser_latched", None)
    if goal_latched is None and first_riser_latched is None:
        success = torch.zeros(len(env_ids), dtype=torch.bool, device=env.device)
    else:
        success = torch.zeros(len(env_ids), dtype=torch.bool, device=env.device)
        if goal_latched is not None:
            success |= goal_latched[env_ids]
        if first_riser_latched is not None:
            success |= first_riser_latched[env_ids]
    challenge_mask = getattr(env, "_route_challenge_mask", None)
    protected = (
        torch.zeros(len(env_ids), dtype=torch.bool, device=env.device)
        if challenge_mask is None
        else challenge_mask[env_ids]
    )
    move_down = ~success & ~protected & (terrain.terrain_levels[env_ids] > 0)
    terrain.update_env_origins(env_ids, success, move_down)
    levels = terrain.terrain_levels.float()
    return {
        "mean": torch.mean(levels),
        "max": torch.max(levels),
        "route_stairs": torch.mean(levels),
    }


def stair_contact_transfer_terrain_levels(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Advance virtual-lip hardness only after a new support transfer."""

    terrain = env.scene.terrain
    if terrain.terrain_origins is None:
        zero = torch.tensor(0.0, device=env.device)
        return {"mean": zero, "max": zero, "contact_transfer": zero}
    transferred = getattr(
        env, "_stair_contact_transfer_stage2_policy_achieved", None
    )
    success = (
        torch.zeros(len(env_ids), dtype=torch.bool, device=env.device)
        if transferred is None
        else transferred[env_ids]
    )
    # Failure holds the current level. A discovered transfer must consolidate
    # before the physical face moves toward the exact hard stair.
    move_down = torch.zeros_like(success)
    terrain.update_env_origins(env_ids, success, move_down)
    levels = terrain.terrain_levels.float()
    return {
        "mean": torch.mean(levels),
        "max": torch.max(levels),
        "contact_transfer": torch.mean(levels),
    }


def fallen_state_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    gate_tilt_above_deg: float = 40.0,
    release_tilt_below_deg: float | None = None,
    release_z_above: float | None = None,
) -> torch.Tensor:
    """1.0 while FALLEN (weight it negative): a flat per-step tax on staying
    down. Without it, lying still is ~0/step while attempting recovery costs
    action-rate/torque penalties — waiting for the fallen_too_long recycle was
    the rational policy. (Penalties on bad states are safe; it's POSITIVE
    rewards gated on bad states that get farmed.)

    With ``release_*`` set, the tax has HYSTERESIS (velstand crouch-endpoint
    lesson): a fall arms it and it keeps paying until the robot is genuinely
    up (tilt < release_tilt AND z > release_z), not merely under the arming
    gate. Without it, a crouch just below the 40° gate is a zero-cost rest
    state — recoveries learned to park there instead of finishing the stand.
    Arms only on a genuine fall, so gait-cycle tilt wobble is never taxed."""
    asset: Entity = env.scene[asset_cfg.name]
    fallen = _fallen_mask(env, asset, 0.0, gate_tilt_above_deg).bool()
    if release_tilt_below_deg is None:
        return fallen.float()
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    quat = asset.data.root_link_quat_w
    cos_tilt = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    up = cos_tilt > math.cos(math.radians(release_tilt_below_deg))
    if release_z_above is not None:
        up &= z > release_z_above
    if not hasattr(env, "_fallen_tax_armed"):
        env._fallen_tax_armed = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
    fresh = env.episode_length_buf <= 1
    env._fallen_tax_armed[fresh] = False
    env._fallen_tax_armed |= fallen
    env._fallen_tax_armed &= ~up
    return env._fallen_tax_armed.float()


def recovery_success(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    fallen_tilt_deg: float = 40.0,
    min_fallen_s: float = 0.5,
    up_tilt_deg: float = 25.0,
    up_z: float = 0.105,
) -> torch.Tensor:
    """One-shot bounty on a COMPLETED recovery: fires on the frame where an env
    that has been fallen (tilt > fallen_tilt for ≥ min_fallen_s) becomes
    genuinely upright (tilt < up_tilt AND trunk z > up_z). Hysteresis: re-arms
    only by being fallen again, so oscillating around the gate pays nothing.
    Gives the sparse-but-strong endpoint gradient the dense gated terms lack.
    """
    asset: Entity = env.scene[asset_cfg.name]
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    quat = asset.data.root_link_quat_w
    cos_tilt = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    fallen = cos_tilt < math.cos(math.radians(fallen_tilt_deg))
    up = (cos_tilt > math.cos(math.radians(up_tilt_deg))) & (z > up_z)
    if not hasattr(env, "_recovery_fallen_s"):
        env._recovery_fallen_s = torch.zeros(env.num_envs, device=env.device)
        env._recovery_armed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    fresh = env.episode_length_buf <= 1
    env._recovery_fallen_s[fresh] = 0.0
    env._recovery_armed[fresh] = False
    env._recovery_fallen_s = torch.where(
        fallen, env._recovery_fallen_s + env.step_dt, torch.zeros_like(env._recovery_fallen_s)
    )
    env._recovery_armed |= env._recovery_fallen_s >= min_fallen_s
    fired = env._recovery_armed & up
    env._recovery_armed &= ~fired
    return fired.float()


def body_upright_linear(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    gate_z_below: float | None = None,
    gate_tilt_above_deg: float = 40.0,
) -> torch.Tensor:
    """Linear reward for body uprightness — provides gradient at every tilt angle.

    Returns +1 when fully upright, 0 when horizontal (prone/supine), -1 when inverted.
    Unlike flat_orientation (Gaussian), this has non-zero gradient everywhere, so the
    robot always has a signal to rotate toward upright even when starting from prone.

    Computed as the z-component of the body's local Z-axis expressed in world frame,
    which equals R[2,2] = 1 - 2*(qx² + qy²) for quaternion [w, x, y, z].
    """
    asset: Entity = env.scene[asset_cfg.name]
    quat = asset.data.root_link_quat_w  # (N, 4): [w, x, y, z]
    qx = quat[:, 1]
    qy = quat[:, 2]
    reward = 1.0 - 2.0 * (qx * qx + qy * qy)
    if gate_z_below is not None:
        # Recovery-gated variant (velstand): active only while fallen, exactly
        # zero during clean walking so it can't dilute the tracking rewards.
        reward = reward * _fallen_mask(env, asset, gate_z_below, gate_tilt_above_deg)
    return reward


def body_upright_gaussian(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    std: float = 0.1,
) -> torch.Tensor:
    """Gaussian reward on tilt magnitude — sharp pull toward fully vertical.

    Complements ``body_upright_linear`` (which is ``cos(tilt)`` and whose
    gradient ``sin(tilt)`` *vanishes* at the target). This Gaussian's
    gradient is non-zero near vertical and tapers as you move away, so it
    creates a strong differential pull in the regime where the linear
    version is weakest.

    Uses ``2*(qx² + qy²) = 1 - cos(tilt) ≈ tilt²/2`` as a tilt-squared
    proxy and applies ``exp(-tilt²/std²)``. Default std=0.1 rad ≈ 5.7°.
    """
    asset: Entity = env.scene[asset_cfg.name]
    quat = asset.data.root_link_quat_w
    qx = quat[:, 1]
    qy = quat[:, 2]
    tilt_sq = 2.0 * (qx * qx + qy * qy)  # ≈ 1 − cos(tilt); small-angle: tilt²/2
    return torch.exp(-tilt_sq / (std * std))


def upright_gaussian_at_height(
    env: ManagerBasedRlEnv,
    std: float,
    height_low: float,
    height_high: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """``body_upright_gaussian`` weighted by smoothstep on trunk z.

    Full Gaussian-upright reward when ``z >= height_high``, zero when
    ``z <= height_low``, smoothstep in between. Use this when the upright
    incentive should only apply at the target standing height — otherwise
    the policy can find a "crouch low and vertical" local optimum that
    collects upright reward without ever rising.
    """
    asset = env.scene[asset_cfg.name]
    quat = asset.data.root_link_quat_w
    qx = quat[:, 1]
    qy = quat[:, 2]
    tilt_sq = 2.0 * (qx * qx + qy * qy)
    upright_g = torch.exp(-tilt_sq / (std * std))
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    t = torch.clamp((z - height_low) / max(height_high - height_low, 1e-6), 0.0, 1.0)
    smooth = t * t * (3.0 - 2.0 * t)
    return upright_g * smooth


def body_ang_vel_at_height(
    env: ManagerBasedRlEnv,
    height_low: float,
    height_high: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    tilt_full_deg: float | None = None,
    tilt_zero_deg: float = 45.0,
) -> torch.Tensor:
    """Trunk ``sum(ω_xy²)`` penalty gated by trunk z (and optionally tilt).

    Height-gated arrival damper: zero below ``height_low`` (ground recovery —
    flips/rolls need large trunk rotation and must stay free), full above
    ``height_high``. Same formula as mjlab's body_angular_velocity_penalty
    (world-frame ω_xy, z-rotation free) but returns the gated POSITIVE cost;
    use a negative weight.

    ``tilt_full_deg`` (optional but STRONGLY recommended): additionally gate
    by tilt — full cost only when tilt ≤ tilt_full_deg, zero when
    ≥ tilt_zero_deg, smoothstep between. LESSON (2026-07 run that broke
    front-recovery): with a height gate alone, the final straighten of a
    bent-over rise (tilt 60°→0 happening INSIDE the z gate) is itself a
    large trunk rotation — taxing it builds a reward wall right before the
    finish, and the policy parks bent-over below the gate instead. With the
    tilt gate, the approach TO vertical is free; only residual wobble
    AROUND vertical (the overshoot→tip→retry oscillation) is damped.
    """
    asset = env.scene[asset_cfg.name]
    ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :].squeeze(1)
    cost = torch.sum(torch.square(ang_vel[:, :2]), dim=1)
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    t = torch.clamp((z - height_low) / max(height_high - height_low, 1e-6), 0.0, 1.0)
    gate = t * t * (3.0 - 2.0 * t)
    if tilt_full_deg is not None:
        quat = asset.data.root_link_quat_w
        cos_tilt = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
        tilt_deg = torch.rad2deg(torch.acos(cos_tilt.clamp(-1.0, 1.0)))
        s = torch.clamp(
            (tilt_zero_deg - tilt_deg) / max(tilt_zero_deg - tilt_full_deg, 1e-6),
            0.0,
            1.0,
        )
        gate = gate * (s * s * (3.0 - 2.0 * s))
    return cost * gate


def standing_composite_score(
    env: ManagerBasedRlEnv,
    target_height: float,
    height_std: float,
    upright_std: float,
    pose_std: float,
    joint_indices: list,
    target_overrides: Optional[dict] = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Smooth multiplicative goal-state score (product of three Gaussians).

    Returns ``height_score * upright_score * pose_score``, each ∈ [0, 1].
    Because the factors *multiply*, a deficiency in any one term collapses
    the whole reward — the policy can't claim 80% of this by being perfect
    on 2-of-3. Gradient is non-zero everywhere, so the score works during
    the rise (not just at the goal like a binary bonus would).

    Use to break Nash-equilibrium compromises (e.g., a "lean trunk at the
    right height" basin that satisfies the additive rewards' partial sums).
    """
    asset = env.scene[asset_cfg.name]

    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    height_score = torch.exp(-((z - target_height) / height_std) ** 2)

    quat = asset.data.root_link_quat_w
    qx = quat[:, 1]
    qy = quat[:, 2]
    tilt_sq = 2.0 * (qx * qx + qy * qy)
    upright_score = torch.exp(-tilt_sq / (upright_std * upright_std))

    target = _servo_default_joint_pos(env, asset).clone()
    if target_overrides:
        for idx, val in target_overrides.items():
            target[:, idx] = val
    joint_pos = _servo_joint_pos(env, asset)[:, joint_indices]
    target = target[:, joint_indices]
    pose_err_sq = ((joint_pos - target) ** 2).mean(dim=-1)
    pose_score = torch.exp(-pose_err_sq / (pose_std * pose_std))

    return height_score * upright_score * pose_score


def standing_success_bonus(
    env: ManagerBasedRlEnv,
    target_height: float,
    height_tol: float,
    upright_threshold: float,
    pose_tol: float,
    joint_indices: list,
    target_overrides: Optional[dict] = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Binary bonus: 1.0 iff height, uprightness AND pose are all within tol.

    Creates a discrete goal-state attractor that gradient-based pose/upright/
    height rewards can't fully match by themselves. Surrounding compromises
    (lean trunk to balance head-forward CoM, park 1cm short of target z,
    etc.) collect partial gradient credit but ZERO bonus — the bonus is
    available only at the true goal state, so it changes the policy's
    relative preference once the rest of the rewards have brought it close.
    """
    asset = env.scene[asset_cfg.name]

    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    height_ok = (z - target_height).abs() <= height_tol

    quat = asset.data.root_link_quat_w
    qx = quat[:, 1]
    qy = quat[:, 2]
    upright = 1.0 - 2.0 * (qx * qx + qy * qy)
    upright_ok = upright >= upright_threshold

    target = _servo_default_joint_pos(env, asset).clone()
    if target_overrides:
        for idx, val in target_overrides.items():
            target[:, idx] = val
    joint_pos = _servo_joint_pos(env, asset)[:, joint_indices]
    target = target[:, joint_indices]
    pose_err = (joint_pos - target).abs().max(dim=-1).values  # tightest joint
    pose_ok = pose_err <= pose_tol

    return (height_ok & upright_ok & pose_ok).float()


def com_upward_velocity(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    max_height: float = 0.08,
    gate_z_below: float | None = None,
    gate_tilt_above_deg: float = 40.0,
    max_vz: float | None = None,
) -> torch.Tensor:
    """Reward upward CoM velocity to incentivize dynamic standup motion.

    Gated by height: only active while the CoM is below `max_height` (the
    standing target). Once standing, the reward is zero so the robot has no
    incentive to keep squatting to farm upward-velocity reward.

    ``max_vz`` (optional): cap the rewarded velocity. Uncapped, the reward is
    proportional to vz, which pays MORE per step for an explosive launch —
    a violent-rise incentive. With a cap, any rise ≥ max_vz earns the same,
    so the gentlest rise that reaches the cap is optimal (the |a_z| penalty
    then picks the smooth one). The bootstrap property is preserved: any
    upward motion still pays immediately.
    """
    asset: Entity = env.scene[asset_cfg.name]
    # nan_to_num: MuJoCo can produce NaN on contact instability; treat as z=0
    com_z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    vz = torch.nan_to_num(asset.data.root_link_lin_vel_w[:, 2], nan=0.0)
    below_target = (com_z < max_height).float()
    reward = torch.clamp(vz, min=0.0, max=max_vz) * below_target
    if gate_z_below is not None:
        # Recovery-gated (velstand): without the gate this pays for dip-and-rise
        # during gait whenever the trunk crosses max_height → bounce incentive.
        reward = reward * _fallen_mask(env, asset, gate_z_below, gate_tilt_above_deg)
    return reward


def fallen_too_long(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    gate_z_below: float = 0.10,
    gate_tilt_above_deg: float = 40.0,
    max_duration_s: float = 5.0,
) -> torch.Tensor:
    """Terminate envs that have been continuously FALLEN for `max_duration_s`.

    For envs that mix walking with fall recovery (velstand): the fell_over
    termination gets disabled by curriculum so the policy can attempt recovery,
    but without a backstop a failed recovery farms recovery-reward for the whole
    20 s episode, starving the walk of data (audit: ~25% walking share). This
    gives every fall a fair recovery window, then recycles the env.
    """
    asset: Entity = env.scene[asset_cfg.name]
    fallen = _fallen_mask(env, asset, gate_z_below, gate_tilt_above_deg).bool()
    if not hasattr(env, "_fallen_timer_s"):
        env._fallen_timer_s = torch.zeros(env.num_envs, device=env.device)
    # Freshly reset envs start with a clean timer.
    env._fallen_timer_s[env.episode_length_buf <= 1] = 0.0
    env._fallen_timer_s = torch.where(
        fallen, env._fallen_timer_s + env.step_dt, torch.zeros_like(env._fallen_timer_s)
    )
    return env._fallen_timer_s >= max_duration_s


def robot_state_is_nan(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    sensor_names: tuple[str, ...] = (),
) -> torch.Tensor:
    """Terminate environments where MuJoCo produced NaN joint positions.

    MuJoCo's contact solver can overflow to NaN under extreme penetration or
    impulse (e.g. robot landing at high velocity). A NaN simulation state
    propagates into observations, corrupting the policy network weights.

    Terminating immediately resets the environment before the cascade spreads:
    - The observation returned to the runner is from the valid reset state.
    - NaN rewards are avoided on subsequent steps.

    Note: the reward at THIS terminal step may still be NaN from the simulation;
    mjlab computes rewards before resetting (see manager_based_rl_env.py step()).
    Our custom reward functions guard against NaN internally with nan_to_num,
    but standard mjlab rewards can still be NaN here. One NaN reward is
    tolerable because done=True prevents it propagating backward through GAE.

    Couvre TOUT l'état physique, pas seulement joint_pos : la divergence du
    contact fait souvent exploser le FREE-JOINT de base (position/orientation/
    vitesse) ou les ROUES passives, pas les joints actionnés. Ces quantités
    alimentent des termes d'obs critic (base_lin_vel, base_ang_vel,
    projected_gravity, wheel_vel) ; si on ne les surveille pas, l'env ne se
    reset pas et le NaN atteint l'obs → le check_nan de rsl_rl tue tout
    l'entraînement. On teste la non-finitude (NaN ET inf, l'inf devenant NaN en
    aval lors de la normalisation de projected_gravity).
    """
    asset: Entity = env.scene[asset_cfg.name]
    d = asset.data
    bad = ~torch.isfinite(d.joint_pos).all(dim=1)
    bad |= ~torch.isfinite(d.joint_vel).all(dim=1)
    bad |= ~torch.isfinite(d.root_link_pos_w).all(dim=1)
    bad |= ~torch.isfinite(d.root_link_quat_w).all(dim=1)
    bad |= ~torch.isfinite(d.root_link_lin_vel_w).all(dim=1)
    bad |= ~torch.isfinite(d.root_link_ang_vel_w).all(dim=1)

    # Contact FORCES can blow up a step before qpos/qvel do: MuJoCo resolves a
    # degenerate contact into an inf/NaN impulse while the integrated state is
    # still finite. That force feeds the critic-only `foot_contact_forces` obs
    # (sign(F)*log1p(|F|)), which the state checks above do NOT cover — so the
    # env was not reset and the NaN reached the runner's check_nan, killing the
    # whole run (crash 2026-08-21, Velocity2-Rough-Backlash with hfield slopes).
    for name in sensor_names:
        if name not in env.scene.sensors:
            continue
        force = getattr(env.scene.sensors[name].data, "force", None)
        if force is not None:
            bad |= ~torch.isfinite(force).flatten(start_dim=1).all(dim=1)
    return bad


def root_height_below(
    env: ManagerBasedRlEnv,
    min_height: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Terminate when the trunk drops below ``min_height`` in world z.

    Utilisé par roller_slope comme « tombé dans le vide » : le terrain a un
    plat de sortie au bas de la rampe, donc une descente normale ne passe
    jamais sous le niveau du plat de sortie le plus bas. Choisir min_height
    en dessous de ce niveau => la terminaison ne se déclenche que si le robot
    quitte le solide et chute dans le vide. Indépendant de la géométrie exacte
    de la rampe (longueur/pente).
    """
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_pos_w[:, 2] < min_height


def descent_speed_reward(
    env: ManagerBasedRlEnv,
    cap: float = 0.8,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Récompense la vitesse d'avance vers le BAS de la pente (monde +x).

    La rampe descend en +x, donc la vitesse linéaire monde en x mesure la
    progression de descente. Plafonnée à ``cap`` m/s : encourage à se laisser
    glisser sans pousser à dévaler de plus en plus vite. Nulle si le robot
    recule/remonte (vx < 0). Sans cette récompense, l'optimum est de rester
    immobile et droit (le robot « freine » au lieu de glisser). NaN-safe.
    """
    asset: Entity = env.scene[asset_cfg.name]
    vx = torch.nan_to_num(
        asset.data.root_link_lin_vel_w[:, 0], nan=0.0, posinf=0.0, neginf=0.0
    )
    return torch.clamp(vx, min=0.0, max=cap)


def reset_rolling_entry(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    speed_range: tuple = (0.25, 0.45),
    wheel_radius: float = 0.0175,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """Départ en ROULEMENT sans glissement (élan aux roues).

    Tire une vitesse d'avance v par env ; met la vitesse LINÉAIRE de base (x
    monde) = v ET la vitesse de ROTATION des 4 roues passives = v / r, donc
    ω·r = v => zéro glissement au contact. Évite l'à-coup de l'ancienne poussée
    base-seule (base qui bouge, roues immobiles = patinage brutal au 1er pas).
    À exécuter APRÈS reset_base (qui pose la base ; ne plus lui donner de
    velocity_range).
    """
    asset: Entity = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    n = int(env_ids.shape[0])
    lo, hi = speed_range
    v = torch.rand(n, device=env.device) * (hi - lo) + lo  # (n,) vitesse avant

    # Vitesse de base (monde) : uniquement +x.
    root_vel = torch.zeros(n, 6, device=env.device)
    root_vel[:, 0] = v
    asset.write_root_link_velocity_to_sim(root_vel, env_ids=env_ids)

    # Rotation des 4 roues passives = v / r (positif = avant, cf. wheel_speed).
    wheel_ids = []
    for name in ("passive_LF_?wheel", "passive_LR_?wheel", "passive_RF_?wheel", "passive_RR_?wheel"):
        ids, _ = asset.find_joints(name)
        wheel_ids.append(ids[0])
    wheel_ids_t = torch.tensor(wheel_ids, device=env.device)
    omega = (v / wheel_radius).unsqueeze(1).repeat(1, len(wheel_ids))  # (n, 4)
    asset.write_joint_velocity_to_sim(omega, joint_ids=wheel_ids_t, env_ids=env_ids)


def wheel_glide_reward(
    env: ManagerBasedRlEnv,
    cap_speed: float = 0.35,
    wheel_radius: float = 0.0175,
) -> torch.Tensor:
    """Récompense le ROULEMENT des roues vers l'avant (glisse), plafonné.

    Contrairement à descent_speed (vitesse de la BASE, qu'on peut atteindre en
    "courant"/poussant), on récompense la rotation des ROUES passives = vraie
    glisse par roulement. Indépendant de toute commande (la tâche pente a une
    commande nulle : la glisse vient de la gravité). Plafonné à ``cap_speed``
    (m/s de vitesse de roulement) -> AUCUNE incitation à accélérer au-delà ; nul
    si les roues reculent (remontée). NaN-safe.
    """
    asset: Entity = env.scene["robot"]
    lf, _ = asset.find_joints("passive_LF_?wheel")
    lr, _ = asset.find_joints("passive_LR_?wheel")
    rf, _ = asset.find_joints("passive_RF_?wheel")
    rr, _ = asset.find_joints("passive_RR_?wheel")
    vel = asset.data.joint_vel
    # Les 4 roues tournent en positif pour l'avant (cf. wheel_speed_reward).
    omega = (vel[:, lf[0]] + vel[:, lr[0]] + vel[:, rf[0]] + vel[:, rr[0]]) / 4.0
    speed = torch.nan_to_num(omega * wheel_radius, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.clamp(speed, min=0.0, max=cap_speed)


def is_alive(env: ManagerBasedRlEnv) -> torch.Tensor:
    """
    Reward for staying alive (not terminated)

    Args:
        env: The environment

    Returns:
        Reward tensor of shape (num_envs,) - ones for all envs
    """
    return torch.ones(env.num_envs, device=env.device)


def com_height_target(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    target_height_min: float = 0.1,
    target_height_max: float = 0.15,
) -> torch.Tensor:
    """
    Reward for keeping the center of mass within a target height range.
    Returns positive reward when in range, negative penalty when outside.

    Args:
        env: The environment
        asset_cfg: Asset configuration
        target_height_min: Minimum target height for CoM (meters)
        target_height_max: Maximum target height for CoM (meters)

    Returns:
        Reward tensor of shape (num_envs,)
    """
    asset: Entity = env.scene[asset_cfg.name]

    # Height above terrain spawn origin (world z minus terrain z).
    # env_origins[:, 2] is 0 for flat ground, so this is safe unconditionally.
    # nan_to_num: MuJoCo can produce NaN on contact instability; treat as z=0
    # so the penalty is finite (small, since 0 is near the target range).
    com_height = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )

    # Reward when in range, penalty when outside
    # Use smooth penalty that increases quadratically with distance from range
    below_min = com_height < target_height_min
    above_max = com_height > target_height_max
    in_range = ~(below_min | above_max)

    # Compute penalties for being outside range
    penalty_below = torch.square(com_height - target_height_min) * below_min.float()
    penalty_above = torch.square(com_height - target_height_max) * above_max.float()

    # Reward: +1 when in range, -squared_distance when outside
    reward = in_range.float() - (penalty_below + penalty_above)

    return reward


def crouch_height_target(
    phase: torch.Tensor,
    height_low: float,
    height_high: float,
    hold_lo: float = 0.375,
    hold_hi: float = 0.625,
) -> torch.Tensor:
    """Cible de hauteur du tronc « en trapèze » le long de la phase [0,1).

    phase ∈ [0, hold_lo)      : descente   height_high -> height_low
    phase ∈ [hold_lo, hold_hi): palier      height_low   (la glisse accroupie)
    phase ∈ [hold_hi, 1.0)    : remontée    height_low  -> height_high

    Args:
        phase: (B,) phase par env, dans [0, 1).
        height_low: hauteur du tronc accroupi (m).
        height_high: hauteur du tronc debout (m).
        hold_lo, hold_hi: bornes du palier bas en fraction de phase.
    Returns:
        (B,) hauteur-cible en mètres.
    """
    descend = phase < hold_lo
    hold = (phase >= hold_lo) & (phase < hold_hi)

    frac_d = phase / hold_lo
    t_descend = height_high + (height_low - height_high) * frac_d

    t_hold = torch.full_like(phase, height_low)

    frac_r = (phase - hold_hi) / (1.0 - hold_hi)
    t_rise = height_low + (height_high - height_low) * frac_r

    return torch.where(descend, t_descend, torch.where(hold, t_hold, t_rise))


def crouch_glide_reward_from_values(
    com_height: torch.Tensor,
    cmd_cos: torch.Tensor,
    cmd_sin: torch.Tensor,
    height_low: float,
    height_high: float,
    hold_lo: float = 0.375,
    hold_hi: float = 0.625,
    std: float = 0.02,
) -> torch.Tensor:
    """Récompense gaussienne du suivi de la cible de hauteur (fonction pure).

    Décode la phase depuis [cos, sin] puis compare la hauteur mesurée à la
    cible-trapèze. Retourne exp(-((h - cible)/std)^2) ∈ (0, 1].
    """
    phase = (torch.atan2(cmd_sin, cmd_cos) / (2 * torch.pi)) % 1.0
    target = crouch_height_target(phase, height_low, height_high, hold_lo, hold_hi)
    return torch.exp(-((com_height - target) / std) ** 2)


def crouch_glide_height_by_phase(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    height_low: float = 0.075,
    height_high: float = 0.11,
    hold_lo: float = 0.375,
    hold_hi: float = 0.625,
    std: float = 0.02,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward principale : suit la cible de hauteur du tronc le long de la phase.

    La hauteur du CoM est calculée comme dans `com_height_target` (world z moins
    l'origine du terrain, nan->0). La phase provient de la commande GroundPick.
    """
    asset: Entity = env.scene[asset_cfg.name]
    com_height = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    cmd = env.command_manager.get_command(command_name)
    return crouch_glide_reward_from_values(
        com_height, cmd[:, 0], cmd[:, 1],
        height_low, height_high, hold_lo, hold_hi, std,
    )


def forward_speed_reward(
    env: ManagerBasedRlEnv,
    vel_ref: float = 0.2,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Récompense la vitesse avant du tronc (conserver l'élan / ne pas freiner).

    Indépendante de la commande (la commande porte la phase, pas la vitesse).
    tanh(clamp(vx, 0)/vel_ref) → sature à ~1, ne récompense jamais reculer.
    """
    asset: Entity = env.scene[asset_cfg.name]
    vx = asset.data.root_link_lin_vel_b[:, 0]
    return torch.tanh(torch.clamp(vx, min=0.0) / vel_ref)


def crouch_pose_blend(
    phase: torch.Tensor,
    descent_end: float,
    hold_end: float,
    rise_end: float,
) -> torch.Tensor:
    """Blend 0..1 le long de la phase [0,1) — 0 = pose debout, 1 = pose accroupie.

    [0, descent_end)      : 0 -> 1  (se baisser)
    [descent_end, hold_end): 1      (bas / accroupi)
    [hold_end, rise_end)  : 1 -> 0  (se lever)
    [rise_end, 1.0)       : 0       (haut / debout, repos)
    """
    b = torch.zeros_like(phase)
    descend = phase < descent_end
    b = torch.where(descend, phase / descent_end, b)
    low = (phase >= descent_end) & (phase < hold_end)
    b = torch.where(low, torch.ones_like(phase), b)
    rise = (phase >= hold_end) & (phase < rise_end)
    b = torch.where(rise, 1.0 - (phase - hold_end) / (rise_end - hold_end), b)
    return b


def _crouch_pose_error(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    crouch_pose: dict,
    descent_end: float,
    hold_end: float,
    rise_end: float,
    stand_pose: Optional[dict] = None,
):
    """(cur, target) joint tensors for the phase-interpolated crouch pose.

    Target interpolates per joint STAND <-> crouch_pose by the 4-segment blend
    b(phase) in [0,1] (0 = standing, 1 = crouch). STAND is `stand_pose` where
    given, else the model DEFAULT (HOME). Joints are resolved BY NAME so the
    passive-wheel interspersing on the roller robot never shifts an index.
    """
    asset: Entity = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    phase = (torch.atan2(cmd[:, 1], cmd[:, 0]) / (2 * torch.pi)) % 1.0  # (B,)
    blend = crouch_pose_blend(phase, descent_end, hold_end, rise_end)   # (B,) 0..1

    names = list(crouch_pose.keys())
    ids = [int(asset.find_joints([n])[0][0]) for n in names]
    default = asset.data.default_joint_pos[:, ids]                     # (B,k)

    stand = default.clone()                                            # source pose
    if stand_pose:
        for j, n in enumerate(names):
            if n in stand_pose:
                stand[:, j] = stand_pose[n]
    crouch = torch.tensor(
        [crouch_pose[n] for n in names], device=env.device, dtype=default.dtype
    ).unsqueeze(0)                                                     # (1,k)

    target = stand + blend.unsqueeze(-1) * (crouch - stand)           # (B,k)
    cur = asset.data.joint_pos[:, ids]                                # (B,k)
    return cur, target


def crouch_glide_pose_by_phase(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    crouch_pose: Optional[dict] = None,
    stand_pose: Optional[dict] = None,
    std: float = 0.4,
    descent_end: float = 0.10,
    hold_end: float = 0.50,
    rise_end: float = 0.60,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Gaussian match to a phase-interpolated joint pose (stand <-> crouch).

    Directive reward: tells the robot the exact joint configuration to be in at
    each phase. Standing back up (target = stand_pose) is rewarded exactly like
    crouching (target = crouch_pose) — symmetric by construction.
    """
    cur, target = _crouch_pose_error(
        env, asset_cfg, command_name, crouch_pose or {},
        descent_end, hold_end, rise_end, stand_pose,
    )
    return torch.exp(-((cur - target) / std) ** 2).mean(dim=-1)


def crouch_glide_pose_l1(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    crouch_pose: Optional[dict] = None,
    stand_pose: Optional[dict] = None,
    descent_end: float = 0.10,
    hold_end: float = 0.50,
    rise_end: float = 0.60,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """L1 bootstrap toward the phase-interpolated crouch pose (negative penalty).

    Constant gradient everywhere — gives the policy a direction to the target
    pose even when the Gaussian above has saturated to ~0 far from it.
    """
    cur, target = _crouch_pose_error(
        env, asset_cfg, command_name, crouch_pose or {},
        descent_end, hold_end, rise_end, stand_pose,
    )
    return -(cur - target).abs().mean(dim=-1)


def crouch_forward_lean(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    target_pitch: float = 0.08,
    std: float = 0.1,
    descent_end: float = 0.10,
    hold_end: float = 0.50,
    rise_end: float = 0.60,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=("trunk_base",)),
) -> torch.Tensor:
    """Léger penché AVANT du tronc pendant l'accroupi (gaté par le blend crouch).

    Contre la bascule arrière induite par la flexion rapide des hanches. Proxy de
    pitch = projected_gravity_b[:,0] (positif = vers l'avant, vérifié). La porte
    (blend) vaut 1 pendant descente+bas, 0 debout → ne biaise QUE l'accroupi.
    target_pitch petit = "de très peu".
    """
    asset: Entity = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    phase = (torch.atan2(cmd[:, 1], cmd[:, 0]) / (2 * torch.pi)) % 1.0
    gate = crouch_pose_blend(phase, descent_end, hold_end, rise_end)
    lean = asset.data.projected_gravity_b[:, 0]
    return gate * torch.exp(-((lean - target_pitch) ** 2) / std ** 2)


def neck_joint_vel_l2(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """
    Penalize neck joint velocities to keep head stable.
    Neck joints are indices 5-8 (4 joints total).

    Args:
        env: The environment
        asset_cfg: Asset configuration

    Returns:
        Penalty tensor of shape (num_envs,)
    """
    asset: Entity = env.scene[asset_cfg.name]

    # Get neck joint indices (neck_pitch, head_pitch, head_yaw, head_roll).
    # Servo view: passive_* joints (backlash, wheels) don't shift the indices.
    neck_joint_indices = list(range(5, 9))
    joint_vel = _servo_joint_vel(env, asset)
    neck_joint_vel = joint_vel[:, neck_joint_indices]

    # Return L2 squared norm of neck joint velocities
    return torch.sum(torch.square(neck_joint_vel), dim=1)


def leg_joint_vel_l2(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """
    Penalize leg joint velocities to encourage smoother, less dynamic motion.
    Leg joints are indices 0-4 and 9-13 (10 joints total).

    Args:
        env: The environment
        asset_cfg: Asset configuration

    Returns:
        Penalty tensor of shape (num_envs,)
    """
    asset: Entity = env.scene[asset_cfg.name]

    # Get leg joint indices (left hip-ankle: 0-4, right hip-ankle: 9-13).
    # Servo view: passive_* joints (backlash, wheels) don't shift the indices.
    leg_joint_indices = list(range(0, 5)) + list(range(9, 14))
    joint_vel = _servo_joint_vel(env, asset)
    leg_joint_vel = joint_vel[:, leg_joint_indices]

    # Return L2 squared norm of leg joint velocities
    return torch.sum(torch.square(leg_joint_vel), dim=1)

_NECK_JOINT_CFG = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*(neck|head).*",))
_HIP_PITCH_KNEE_CFG = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*(hip_pitch|knee).*",))
_ROLLER_FEET_SITE_CFG = SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))


def feet_flat_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _ROLLER_FEET_SITE_CFG,
    sensor_name: str | None = None,
) -> torch.Tensor:
    """Penalize foot sites not being parallel to the ground.

    The foot site frame has Z+ pointing up when flat. We project a unit gravity
    vector (pointing down) into each foot site's local frame. When flat, gravity
    maps to [0,0,-1] in site frame (xy=0, penalty=0). Any tilt rotates Z away
    from world-up, giving nonzero xy components.

    Max value ≈ 2.0 per foot (foot fully sideways), total ≈ 4.0.

    When ``sensor_name`` is given, each foot's penalty is GATED by that foot's own
    ground contact: the airborne (swing) foot is free to tilt, only the stance
    blade is asked to stay flat (so its wheels keep gripping). Without this gate
    the penalty punishes the recovery-foot lift a stride needs — it is minimised
    by keeping BOTH blades flat on the ground, i.e. the swizzle. Assumes the site
    order (left, right) matches the sensor slot order (ankle_l_v1,
    ankle_r_v1) — both left-first in this model.

    Bug note: must normalize gravity PER ENV with dim=-1. Using torch.norm()
    without dim computes a scalar over all envs × 3 dims, making the vector
    ~1/sqrt(num_envs) in magnitude → penalty ~num_envs times too small.
    """
    from mjlab.utils.lab_api.math import quat_apply_inverse
    import torch.nn.functional as F

    asset: Entity = env.scene[asset_cfg.name]
    gravity_w_n = F.normalize(asset.data.gravity_vec_w, dim=-1)  # (B, 3), unit vector per env

    foot_quats = asset.data.site_quat_w[:, asset_cfg.site_ids, :]  # (B, N_feet, 4)
    per_foot = torch.zeros(env.num_envs, foot_quats.shape[1], device=env.device)
    for i in range(foot_quats.shape[1]):
        proj = quat_apply_inverse(foot_quats[:, i, :], gravity_w_n)  # (B, 3)
        per_foot[:, i] = torch.sum(torch.square(proj[:, :2]), dim=1)  # xy² only

    if sensor_name is not None:
        from mjlab.sensor import ContactSensor
        sensor: ContactSensor = env.scene[sensor_name]
        contact_time = sensor.data.current_contact_time  # (B, N_feet)
        assert contact_time is not None
        per_foot = per_foot * (contact_time > 0.0).float()

    return per_foot.sum(dim=1)


def feet_tiptoe_alignment(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _ROLLER_FEET_SITE_CFG,
    command_name: str = "twist",
    command_threshold: float = 0.01,
) -> torch.Tensor:
    """Reward each foot site's local x-axis pointing downward — tiptoe stance.

    When flat, foot site x points roughly forward (horizontal). Pitching the
    foot forward (heel up, toe down) rotates x toward world -Z. We reward the
    z-component of the foot x-axis being -1 (perfectly downward).

    Per foot: alignment ∈ [-1, 1], summed over both feet ∈ [-2, 2].

    Gated on |vel_cmd_xy| > command_threshold so the policy isn't required to
    stand on tiptoes at rest — only while walking. The companion
    feet_flat_penalty is NOT used in this task; the two would fight.
    """
    asset: Entity = env.scene[asset_cfg.name]
    quats = asset.data.site_quat_w[:, asset_cfg.site_ids, :]  # (B, N, 4) [w, x, y, z]
    w, qx, qy, qz = quats[:, :, 0], quats[:, :, 1], quats[:, :, 2], quats[:, :, 3]
    x_axis_z = 2.0 * (qx * qz - w * qy)  # (B, N) — z-component of local x-axis in world
    alignment = (-x_axis_z).sum(dim=-1)  # +1 per foot when pointing straight down

    cmd = env.command_manager.get_command(command_name)
    cmd_mag = torch.linalg.norm(cmd[:, :2], dim=1)
    active = (cmd_mag > command_threshold).float()
    return alignment * active


def hip_pitch_knee_vel_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _HIP_PITCH_KNEE_CFG,
) -> torch.Tensor:
    """Penalize hip_pitch and knee joint velocities (L2 squared).

    Walking requires rapid oscillation of these sagittal-plane joints.
    Skating uses hip_roll laterally and glides with minimal sagittal movement.
    This penalizes the oscillation without preventing static balance adjustments.
    """
    asset: Entity = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def neck_joint_pos_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _NECK_JOINT_CFG,
    pattern: str = r".*(neck|head).*",
) -> torch.Tensor:
    """Penalize neck/head joint position deviation from default (L2 squared).

    Uses find_joints() every call to avoid stale cached indices when the same
    SceneEntityCfg singleton is reused across robots with different joint layouts
    (e.g. walk robot vs rollers robot where passive wheels shift neck indices).

    ``pattern`` sélectionne les joints comptés (défaut : toute la nuque + la tête).
    La tâche spin passe un motif qui EXCLUT `head_yaw`, pour laisser la tête servir
    de volant d'inertie au lancement de la rotation.
    """
    asset: Entity = env.scene[asset_cfg.name]
    # Exclude passive_* joints (backlash hinges also contain "neck"/"head").
    if not pattern.startswith(r"^(?!passive_)"):
        pattern = r"^(?!passive_)" + pattern.lstrip("^")
    joint_ids, _ = asset.find_joints(pattern)
    error = asset.data.joint_pos[:, joint_ids] - asset.data.default_joint_pos[:, joint_ids]
    return torch.sum(torch.square(error), dim=1)


def joint_torques_l2(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """
    Penalize actuator forces (torques) to encourage energy-efficient motion.

    Args:
        env: The environment
        asset_cfg: Asset configuration

    Returns:
        Penalty tensor of shape (num_envs,) - sum of squared actuator forces
    """
    asset: Entity = env.scene[asset_cfg.name]

    # Get actuator forces (scalar actuation in actuation space)
    actuator_forces = asset.data.actuator_force

    # Return L2 squared norm
    return torch.sum(torch.square(actuator_forces), dim=1)


def joint_torque_rate_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize rate of change in actuator torques (proxy for gearbox shock).

    Sudden torque spikes occur when the robot impacts the ground and actuators
    resist the impulse. Penalising this rate encourages soft landings and smooth
    force transitions that protect gearboxes.

    Returns the sum of squared torque differences from the previous step.
    """
    asset: Entity = env.scene[asset_cfg.name]
    current = asset.data.actuator_force  # (num_envs, num_actuators)

    if not hasattr(env, '_prev_actuator_forces'):
        env._prev_actuator_forces = current.clone()
        return torch.zeros(env.num_envs, device=env.device)

    rate = current - env._prev_actuator_forces
    env._prev_actuator_forces = current.clone()
    return torch.sum(torch.square(rate), dim=1)


def feet_grounded_reward(
    env: ManagerBasedRlEnv,
    sensor_name: str,
) -> torch.Tensor:
    """Positive reward for feet contacting the ground (0, +0.5, or +1.0).

    Uses the contact sensor's `found` field. For the feet_ground_contact sensor
    which has 2 primary foot geoms, `found` has shape (num_envs, 2) with per-foot
    binary contact. We sum and normalize to [0, 1].
    """
    if sensor_name not in env.scene.sensors:
        return torch.zeros(env.num_envs, device=env.device)
    sensor = env.scene.sensors[sensor_name]
    found = sensor.data.found  # (num_envs, num_feet) or (num_envs, 1)
    if found.dim() > 1:
        found = found.sum(dim=-1)  # collapse foot dimension
    return torch.clamp(found, 0.0, 2.0) / 2.0


def body_impact_cost(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize terrain contact forces above a threshold on protected body parts.

    Used to discourage slamming the trunk shell or head into the ground during
    falls. The sensor should cover the relevant body or subtree with
    reduce='netforce'. Forces below threshold are free; above that the penalty
    grows linearly.

    Args:
        sensor_name: Name of a ContactSensorCfg with fields=("force",),
            reduce="netforce".
        threshold: Contact force (N) below which no penalty is applied.

    Returns:
        Penalty tensor (num_envs,) — N above threshold per step.
    """
    if sensor_name not in env.scene.sensors:
        return torch.zeros(env.num_envs, device=env.device)

    sensor = env.scene.sensors[sensor_name]
    forces = sensor.data.force  # (num_envs, N_bodies, 3)
    total_force = forces.sum(dim=1)  # sum over bodies in the subtree
    force_mag = torch.norm(total_force, dim=1)
    return torch.clamp(force_mag - threshold, min=0.0)


def wheel_speed_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    wheel_radius: float = 0.0175,
    vel_scale: float = 0.5,
    bidirectional: bool = False,
) -> torch.Tensor:
    """Reward wheel spin proportional to commanded push.

    All 4 wheels spin positive for forward motion (verified visually).
    tanh saturation at vel_scale m/s equivalent prevents runaway.

    - ``bidirectional=False`` (default): forward only — reward forward spin for
      cmd_x > 0, silent otherwise (cmd_x < 0 handled by the braking reward).
    - ``bidirectional=True``: reward wheel spin in the COMMANDED direction —
      forward for cmd_x > 0, backward for cmd_x < 0 — with magnitude |cmd_x|.
      Lets cmd_x < 0 mean "go backward" instead of "brake".
    """
    cmd_x = env.command_manager.get_command(command_name)[:, 0]  # (B,)

    asset: Entity = env.scene["robot"]
    lf_ids, _ = asset.find_joints("passive_LF_?wheel")
    lr_ids, _ = asset.find_joints("passive_LR_?wheel")
    rf_ids, _ = asset.find_joints("passive_RF_?wheel")
    rr_ids, _ = asset.find_joints("passive_RR_?wheel")

    vel = asset.data.joint_vel
    # All 4 wheels spin positive for forward motion (verified by test_wheel_direction.py)
    forward_omega = (vel[:, lf_ids[0]] + vel[:, lr_ids[0]] + vel[:, rf_ids[0]] + vel[:, rr_ids[0]]) / 4.0

    omega_scale = vel_scale / wheel_radius
    if bidirectional:
        # spin aligned with the command sign (fwd for +, back for -)
        aligned = torch.sign(cmd_x) * forward_omega
        return torch.abs(cmd_x) * torch.tanh(torch.clamp(aligned, min=0.0) / omega_scale)
    return torch.clamp(cmd_x, min=0.0) * torch.tanh(torch.clamp(forward_omega, min=0.0) / omega_scale)


def coasting_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    vel_std: float = 0.3,
    stillness_std: float = 5.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=(r".*(hip|knee|ankle).*",)),
) -> torch.Tensor:
    """Reward coasting: low leg-joint velocity while at target speed.

    Returns exp(-vel_error / vel_std²) × exp(-sum(joint_vel²) / stillness_std²).
    Both factors must be high simultaneously — robot is rewarded for being at
    target speed AND keeping its legs still (gliding), not for either alone.

    Typical values when coasting well: ~0.7–1.0.  When actively stomping at
    speed the joint_vel term suppresses the reward toward 0.
    """
    cmd = env.command_manager.get_command(command_name)
    vel_b = env.scene["robot"].data.root_link_lin_vel_b[:, :2]
    vel_error = torch.sum(torch.square(cmd[:, :2] - vel_b), dim=1)
    at_speed = torch.exp(-vel_error / vel_std ** 2)

    asset: Entity = env.scene[asset_cfg.name]
    joint_vel_sq = torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)
    stillness = torch.exp(-joint_vel_sq / stillness_std ** 2)

    return at_speed * stillness


def braking_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    vel_std: float = 0.3,
) -> torch.Tensor:
    """Reward coming to a stop when cmd_x < 0 (brake commanded).

    Returns clamp(-cmd_x, 0) * exp(-fwd_vel² / vel_std²).
    - Silent when cmd_x ≥ 0 (coast or push).
    - At cmd_x = -1 and vel = 0: reward = 1.0 (full stop achieved).
    - At cmd_x = -1 and vel = vel_std: reward ≈ 0.37 (strong gradient).
    vel_std=0.3 m/s gives meaningful gradient down to walking-pace speeds.
    """
    cmd = env.command_manager.get_command(command_name)
    cmd_x = cmd[:, 0]
    braking_strength = torch.clamp(-cmd_x, min=0.0)
    fwd_vel = env.scene["robot"].data.root_link_lin_vel_b[:, 0]
    stopped = torch.exp(-(fwd_vel.clamp(min=0.0) ** 2) / (vel_std ** 2))
    return braking_strength * stopped


def contact_frequency_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str = "feet_ground_contact",
    max_contact_changes_per_sec: float = 4.0,
    command_threshold: float = 0.01,
) -> torch.Tensor:
    """
    Penalize high frequency of contact changes to encourage slower stepping.
    Tracks the number of contact state changes per second and penalizes when above threshold.

    Args:
        env: The environment
        sensor_name: Name of the contact sensor
        max_contact_changes_per_sec: Maximum allowed contact changes per second
        command_threshold: Minimum command magnitude to apply penalty

    Returns:
        Penalty tensor of shape (num_envs,) - negative when exceeding threshold
    """
    if sensor_name not in env.scene.sensors:
        return torch.zeros(env.num_envs, device=env.device)

    # Check if command is above threshold
    if "twist" in env.command_manager._terms:
        cmd = env.command_manager.get_command("twist")
        cmd_vel = cmd[:, :3]
        cmd_norm = torch.linalg.norm(cmd_vel, dim=1)
        active_mask = cmd_norm > command_threshold
    else:
        active_mask = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)

    sensor = env.scene.sensors[sensor_name]
    contacts = sensor.data.found[:, :2]  # (num_envs, 2)

    # Initialize tracking if needed
    if not hasattr(env, '_contact_change_count'):
        env._contact_change_count = torch.zeros(env.num_envs, device=env.device)
        env._contact_change_timer = torch.zeros(env.num_envs, device=env.device)
        env._prev_contacts_for_freq = contacts.clone()
        return torch.zeros(env.num_envs, device=env.device)

    # Detect any contact changes (either foot)
    contact_changed = torch.any(contacts != env._prev_contacts_for_freq, dim=1)

    # Increment change counter
    env._contact_change_count += contact_changed.float()

    # Update timer
    env._contact_change_timer += env.step_dt

    # Calculate current frequency (changes per second)
    # Avoid division by zero
    freq = env._contact_change_count / torch.clamp(env._contact_change_timer, min=0.01)

    # Reset counter and timer every 1 second
    reset_mask = env._contact_change_timer >= 1.0
    env._contact_change_count[reset_mask] = 0.0
    env._contact_change_timer[reset_mask] = 0.0

    # Penalize when frequency exceeds maximum
    # Use quadratic penalty for frequencies above threshold
    excess_freq = torch.clamp(freq - max_contact_changes_per_sec, min=0.0)
    penalty = -torch.square(excess_freq)

    # Update previous contacts
    env._prev_contacts_for_freq = contacts.clone()

    # Apply command threshold mask
    penalty = penalty * active_mask.float()

    return penalty


# ==============================================================================
# Ground Pick Rewards
# ==============================================================================

def mouth_ground_proximity(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=["mouth_tip"]),
    std: float = 0.03,
    target_height: float = 0.0,
    command_name: str = "twist",
) -> torch.Tensor:
    """Reward for mouth tip approaching the ground, weighted by the approach phase.

    The command for the ground pick task is [cos(2π*phase), sin(2π*phase), 0].
    The approach phase is the first half-cycle (sin > 0, phase ∈ [0, 0.5]),
    smoothly weighted by max(0, sin(2π*phase)).

    Args:
        std: Gaussian std on mouth_tip height (m). 0.03 m gives strong gradient.
        target_height: Target z-height for the mouth tip (m). 0 = ground level.
    """
    asset = env.scene[asset_cfg.name]
    mouth_z = asset.data.site_pos_w[:, asset_cfg.site_ids[0], 2]  # (num_envs,)
    proximity = torch.exp(-((mouth_z - target_height) / std) ** 2)

    # Approach weight: max(0, sin(2π*phase)) — peaks at 1 at phase=0.25, zero at 0 and 0.5
    cmd = env.command_manager.get_command(command_name)
    approach_weight = torch.clamp(cmd[:, 1], min=0.0)

    return approach_weight * proximity


def mouth_perpendicular_to_ground(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=["mouth_tip"]),
    command_name: str = "twist",
) -> torch.Tensor:
    """Reward the mouth tip x-axis being vertical (pointing down) during the approach phase.

    A perfectly perpendicular contact gives alignment=1; horizontal gives 0; pointing up gives -1.
    Weighted by max(0, sin(2π*phase)) so it only applies during the descent.
    """
    asset = env.scene[asset_cfg.name]
    # site_quat_w: (num_envs, num_sites, 4) as [w, x, y, z]
    q = asset.data.site_quat_w[:, asset_cfg.site_ids[0], :]  # (num_envs, 4)
    w, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    # z-component of the site x-axis in world frame (first column of rotation matrix)
    x_axis_z = 2.0 * (qx * qz - w * qy)
    # dot with [0, 0, -1]: 1 = perfectly downward, -1 = upward
    alignment = -x_axis_z

    cmd = env.command_manager.get_command(command_name)
    approach_weight = torch.clamp(cmd[:, 1], min=0.0)

    return approach_weight * alignment


def sit_grounded(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: Optional[str] = None,
    sin_threshold: float = 0.7,
    min_progress_frac: float = 0.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    upright_cos_threshold: float = 0.5,
) -> torch.Tensor:
    """Positive reward for trunk-ground contact WHILE upright.

    Gated additionally on the trunk's body-frame +Z axis pointing in roughly the
    world-up direction (cosine >= ``upright_cos_threshold``, default 0.5 → up to
    60° tilt accepted). Without this gate, the policy can earn the contact
    bonus by tipping sideways or face-forward — the trunk hits the ground in
    those weird poses, sit_grounded fires, and the policy converges to a
    "fallen" mode that competes with the actual sit pose.

    When ``command_name`` is provided, the reward is gated to the sit window of
    a phase command. Otherwise it's always-on, optionally gated to the late
    part of the episode via ``min_progress_frac``.
    """
    if sensor_name not in env.scene.sensors:
        return torch.zeros(env.num_envs, device=env.device)
    sensor = env.scene.sensors[sensor_name]
    found = sensor.data.found
    if found.dim() > 1:
        found = found.sum(dim=-1)
    has_contact = (found > 0).float()

    # Upright check: trunk body's +Z (world frame, third column of rotation matrix
    # derived from the trunk quaternion) dot world-up = trunk's body-up · world-up.
    # Equivalently: 1 - 2*(qx² + qy²) for a unit quaternion (w, x, y, z).
    asset: Entity = env.scene[asset_cfg.name]
    quat = asset.data.root_link_quat_w  # (N, 4) = (w, x, y, z)
    qx, qy = quat[:, 1], quat[:, 2]
    upright_cos = 1.0 - 2.0 * (qx * qx + qy * qy)
    is_upright = (upright_cos >= upright_cos_threshold).float()

    contact_upright = has_contact * is_upright

    if command_name is None:
        if min_progress_frac > 0.0:
            progress = env.episode_length_buf.float() / float(env.max_episode_length)
            late_enough = (progress >= min_progress_frac).float()
            return late_enough * contact_upright
        return contact_upright
    cmd = env.command_manager.get_command(command_name)
    in_sit_window = (cmd[:, 1] > sin_threshold).float()
    return in_sit_window * contact_upright


def sit_stability(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    command_name: Optional[str] = None,
    ang_vel_std: float = 0.5,
    sin_threshold: float = 0.7,
    min_progress_frac: float = 0.0,
) -> torch.Tensor:
    """Bonus for low body angular velocity.

    Phase-gated when ``command_name`` is set (sit window of a phase command).
    Always-on otherwise, optionally restricted to the late part of the episode
    via ``min_progress_frac``. Encourages a stable rest pose.
    """
    asset = env.scene[asset_cfg.name]
    ang_vel_norm = asset.data.root_link_ang_vel_w.norm(dim=-1)
    stillness = torch.exp(-((ang_vel_norm / ang_vel_std) ** 2))
    if command_name is None:
        if min_progress_frac > 0.0:
            progress = env.episode_length_buf.float() / float(env.max_episode_length)
            late_enough = (progress >= min_progress_frac).float()
            return late_enough * stillness
        return stillness
    cmd = env.command_manager.get_command(command_name)
    in_sit_window = (cmd[:, 1] > sin_threshold).float()
    return in_sit_window * stillness


def joint_deviation_l1(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """L1 penalty for joint positions deviating from their default (HOME).

    Returns sum of |joint_pos - default| over the selected joints. Unlike the
    Gaussian `pose` reward (which saturates near 1.0 for any small deviation),
    this gives a *linear* gradient at all deviation magnitudes — useful as a
    focused penalty on a subset of joints (e.g. hip_yaw / hip_roll) to prevent
    them drifting to wide-base stances even when other joints are near HOME.
    """
    asset = env.scene[asset_cfg.name]
    jnt_ids = asset_cfg.joint_ids
    err = asset.data.joint_pos[:, jnt_ids] - asset.data.default_joint_pos[:, jnt_ids]
    return torch.sum(torch.abs(err), dim=-1)


def joint_pos_limit_proximity(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    margin: float = 0.15,
) -> torch.Tensor:
    """L1 penalty for joint positions entering a ``margin`` (rad) band next to
    their *hard* range limits.

    The base ``joint_pos_limits`` reward only fires past the *soft* limit
    (global ``soft_joint_pos_limit_factor`` = 0.9 → roughly the last 7.5% of
    range) and only by the radians-overshoot magnitude, so it's near-useless
    against a joint parked on its stop. This term instead reads the *hard*
    limits directly and lets each reward set its own wide margin, scoped to
    specific joints.

    Motivating case: with a low-kp position servo and wide ctrlrange the policy
    can command far past a joint's limit "for free" (no command-side cost) and
    park the joint on its hard stop — e.g. hip_yaw slammed to ±limit so the foot
    slides/pivots. The overshoot is *intended* (it's how a low-kp servo reaches
    its target), so the deterrent must live on the qpos side and bite well
    before the stop.

    For each selected joint with hard limits ``[lo, hi]``::

        soft_lo = lo + margin,  soft_hi = hi - margin
        penalty = relu(soft_lo - q) + relu(q - soft_hi)

    summed over joints: zero in the interior, ramping linearly toward each stop.
    """
    asset = env.scene[asset_cfg.name]
    jnt_ids = asset_cfg.joint_ids
    q = asset.data.joint_pos[:, jnt_ids]
    hard = asset.data.joint_pos_limits[:, jnt_ids]  # (num_envs, num_sel_joints, 2)
    soft_lo = hard[..., 0] + margin
    soft_hi = hard[..., 1] - margin
    below = (soft_lo - q).clip(min=0.0)
    above = (q - soft_hi).clip(min=0.0)
    return torch.sum(below + above, dim=-1)


def phase_height_track(
    env: ManagerBasedRlEnv,
    command_name: str,
    stand_z: float,
    sit_z: float,
    std: float = 0.02,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward trunk_z tracking a sin-interpolated target between stand and sit heights.

    Used for the sitstand task instead of joint-angle matching for the sit pose —
    rewards the END STATE (low trunk) without prescribing HOW the robot gets there.
    The policy is free to find any motion strategy (deep squat, head-supported
    descent, etc.).

    Command (from GroundPickPhaseCommand): cmd[:, 1] = sin(2π·phase).
    sin = +1 at phase 0.25 (sit peak) → target = sit_z.
    sin = -1 at phase 0.75 (stand peak) → target = stand_z.
    sin = 0 at transitions → target = midpoint.
    """
    cmd = env.command_manager.get_command(command_name)
    sin_phase = cmd[:, 1]
    target_z = (stand_z + sit_z) * 0.5 - (stand_z - sit_z) * 0.5 * sin_phase
    asset = env.scene[asset_cfg.name]
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    return torch.exp(-((z - target_z) / std) ** 2)


def pose_target_match(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    std: float = 0.3,
    joint_indices: Optional[list] = None,
    target_overrides: Optional[dict] = None,
) -> torch.Tensor:
    """Always-on Gaussian on joint positions vs a target pose.

    Non-phase analog of ``phase_pose_match``: useful for episodic tasks (e.g.
    the sit env) where there's no cyclic command to weight the reward by, and
    the target pose is constant for the whole episode.

    Args:
        std: Gaussian std per joint (rad).
        joint_indices: Optional subset of joints to evaluate.
        target_overrides: ``{joint_index: angle_rad}``. Joints not listed default
            to ``asset.data.default_joint_pos`` (the home/standing pose).
    """
    asset = env.scene[asset_cfg.name]
    joint_pos = _servo_joint_pos(env, asset)
    target = _servo_default_joint_pos(env, asset).clone()
    if target_overrides:
        for idx, val in target_overrides.items():
            target[:, idx] = val
    if joint_indices is not None:
        joint_pos = joint_pos[:, joint_indices]
        target = target[:, joint_indices]
    return torch.exp(-((joint_pos - target) / std) ** 2).mean(dim=-1)


def interpolated_pose_target_match(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    std: float = 0.3,
    joint_indices: Optional[list] = None,
    source_overrides: Optional[dict] = None,
    target_overrides: Optional[dict] = None,
    ramp_start_frac: float = 0.0,
    ramp_end_frac: float = 1.0,
) -> torch.Tensor:
    """Gaussian on joint positions vs a time-interpolated target pose.

    Tracks a target that linearly interpolates from a source pose to a target
    pose over the episode, between progress fractions ``ramp_start_frac`` and
    ``ramp_end_frac``. Before/after the ramp the target is clamped to source /
    final target respectively.

    The point is to enforce smooth descent: snapping to the final target early
    leaves the robot *off-target* relative to where the interpolated target
    currently is, costing pose reward for the duration of the mismatch.

    Args:
        std: Gaussian std per joint (rad).
        joint_indices: Optional subset of joints to evaluate.
        source_overrides: ``{joint_index: angle_rad}`` defining the source pose
            (start of the ramp). ``None`` = default/HOME pose.
        target_overrides: same, for the target pose (end of the ramp).
        ramp_start_frac, ramp_end_frac: episode-progress window in [0, 1] over
            which the target moves from source to target.
    """
    asset = env.scene[asset_cfg.name]
    joint_pos = _servo_joint_pos(env, asset)
    source = _servo_default_joint_pos(env, asset).clone()
    target = _servo_default_joint_pos(env, asset).clone()
    if source_overrides:
        for idx, val in source_overrides.items():
            source[:, idx] = val
    if target_overrides:
        for idx, val in target_overrides.items():
            target[:, idx] = val

    progress = env.episode_length_buf.float() / float(env.max_episode_length)
    span = max(ramp_end_frac - ramp_start_frac, 1e-6)
    tau = ((progress - ramp_start_frac) / span).clamp(0.0, 1.0).unsqueeze(-1)
    interp = source * (1.0 - tau) + target * tau

    if joint_indices is not None:
        joint_pos = joint_pos[:, joint_indices]
        interp = interp[:, joint_indices]
    return torch.exp(-((joint_pos - interp) / std) ** 2).mean(dim=-1)


def interpolated_pose_l1_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    joint_indices: Optional[list] = None,
    source_overrides: Optional[dict] = None,
    target_overrides: Optional[dict] = None,
    ramp_start_frac: float = 0.0,
    ramp_end_frac: float = 1.0,
) -> torch.Tensor:
    """L1 distance from a time-interpolated target pose (negative — used as penalty).

    Same interpolation schedule as ``interpolated_pose_target_match`` but
    returns ``-mean(|joint_pos - interp|)`` instead of a Gaussian. The L1
    gradient is constant everywhere — useful as a bootstrap signal when the
    Gaussian variant saturates to zero far from target and leaves the policy
    no gradient to discover the target direction.
    """
    asset = env.scene[asset_cfg.name]
    joint_pos = _servo_joint_pos(env, asset)
    source = _servo_default_joint_pos(env, asset).clone()
    target = _servo_default_joint_pos(env, asset).clone()
    if source_overrides:
        for idx, val in source_overrides.items():
            source[:, idx] = val
    if target_overrides:
        for idx, val in target_overrides.items():
            target[:, idx] = val

    progress = env.episode_length_buf.float() / float(env.max_episode_length)
    span = max(ramp_end_frac - ramp_start_frac, 1e-6)
    tau = ((progress - ramp_start_frac) / span).clamp(0.0, 1.0).unsqueeze(-1)
    interp = source * (1.0 - tau) + target * tau

    if joint_indices is not None:
        joint_pos = joint_pos[:, joint_indices]
        interp = interp[:, joint_indices]
    return -torch.abs(joint_pos - interp).mean(dim=-1)


def interpolated_height_l1_penalty(
    env: ManagerBasedRlEnv,
    start_height: float,
    end_height: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ramp_start_frac: float = 0.0,
    ramp_end_frac: float = 1.0,
) -> torch.Tensor:
    """L1 distance from a time-interpolated target height (negative — penalty).

    Same role as ``interpolated_pose_l1_penalty`` but on trunk z. Provides a
    constant gradient toward the target height regardless of how far off the
    current z is, complementing the Gaussian ``interpolated_height_target``.
    """
    progress = env.episode_length_buf.float() / float(env.max_episode_length)
    span = max(ramp_end_frac - ramp_start_frac, 1e-6)
    tau = ((progress - ramp_start_frac) / span).clamp(0.0, 1.0)
    target_z = start_height * (1.0 - tau) + end_height * tau

    asset = env.scene[asset_cfg.name]
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    return -torch.abs(z - target_z)


def interpolated_height_target(
    env: ManagerBasedRlEnv,
    start_height: float,
    end_height: float,
    std: float = 0.02,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ramp_start_frac: float = 0.0,
    ramp_end_frac: float = 1.0,
) -> torch.Tensor:
    """Gaussian on trunk z vs a time-interpolated target height.

    Companion to ``interpolated_pose_target_match`` — same time-interpolation
    logic applied to the trunk height.
    """
    progress = env.episode_length_buf.float() / float(env.max_episode_length)
    span = max(ramp_end_frac - ramp_start_frac, 1e-6)
    tau = ((progress - ramp_start_frac) / span).clamp(0.0, 1.0)
    target_z = start_height * (1.0 - tau) + end_height * tau

    asset = env.scene[asset_cfg.name]
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    return torch.exp(-((z - target_z) / std) ** 2)


def bilateral_symmetry_penalty(
    env: ManagerBasedRlEnv,
    left_indices: list,
    right_indices: list,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """L1 penalty on left/right leg asymmetry.

    For a bilaterally-symmetric robot the leg HOME and any symmetric target
    (FOLD, SIT) satisfy ``q_left + q_right == 0`` on each matched joint pair
    (because the left/right joints use mirrored sign conventions). This term
    penalises departures from that constraint.

    Useful when ``mean()`` of pose-target rewards lets the policy get away
    with one-leg-correct solutions (you collect ~half the reward for free
    and the gradient toward fixing the second leg is too weak to escape that
    local minimum). The penalty here has constant L1 gradient regardless of
    magnitude, so any asymmetry pays a cost and the unique zero is the
    fully-symmetric configuration.

    Returns ``-sum_i |q[left_i] + q[right_i]|`` averaged over the N pairs.
    """
    asset: Entity = env.scene[asset_cfg.name]
    pos = asset.data.joint_pos
    left = pos[:, left_indices]
    right = pos[:, right_indices]
    return -torch.abs(left + right).mean(dim=-1)


def _multistage_target_pose(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    waypoints,
) -> torch.Tensor:
    """Compute the time-interpolated joint target across N waypoints.

    waypoints: ordered list of dicts {"frac": float in [0,1],
                                       "overrides": dict[int,float] | None}.
    First waypoint should have frac=0.0 (typically HOME, overrides=None).
    Subsequent waypoints define milestones. Between two waypoints the target
    linearly interpolates. Before the first / after the last it clamps.

    Returns a (num_envs, num_joints) tensor of target joint angles.
    """
    asset = env.scene[asset_cfg.name]
    default = _servo_default_joint_pos(env, asset)

    def build_pose(overrides):
        pose = default.clone()
        if overrides:
            for idx, val in overrides.items():
                pose[:, idx] = val
        return pose

    progress = env.episode_length_buf.float() / float(env.max_episode_length)
    # Find which segment we're in (broadcast over envs).
    out = build_pose(waypoints[0]["overrides"])
    for i in range(1, len(waypoints)):
        f0 = waypoints[i - 1]["frac"]
        f1 = waypoints[i]["frac"]
        span = max(f1 - f0, 1e-6)
        tau = ((progress - f0) / span).clamp(0.0, 1.0).unsqueeze(-1)
        prev_pose = build_pose(waypoints[i - 1]["overrides"])
        next_pose = build_pose(waypoints[i]["overrides"])
        seg = prev_pose * (1.0 - tau) + next_pose * tau
        # Take this segment's value when progress is in [f0, f1] or past it.
        mask = (progress >= f0).float().unsqueeze(-1)
        out = torch.where(mask > 0, seg, out)
    return out


def _multistage_target_height(
    env: ManagerBasedRlEnv,
    waypoints,
) -> torch.Tensor:
    """Same logic as _multistage_target_pose but for trunk z height.

    waypoints: [{"frac": float, "height": float}, ...].
    """
    progress = env.episode_length_buf.float() / float(env.max_episode_length)
    out = torch.full_like(progress, waypoints[0]["height"])
    for i in range(1, len(waypoints)):
        f0 = waypoints[i - 1]["frac"]
        f1 = waypoints[i]["frac"]
        span = max(f1 - f0, 1e-6)
        tau = ((progress - f0) / span).clamp(0.0, 1.0)
        seg = waypoints[i - 1]["height"] * (1.0 - tau) + waypoints[i]["height"] * tau
        mask = (progress >= f0).float()
        out = torch.where(mask > 0, seg, out)
    return out


def multistage_pose_target_match(
    env: ManagerBasedRlEnv,
    waypoints: list,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    std: float = 0.3,
    joint_indices: Optional[list] = None,
) -> torch.Tensor:
    """Multi-waypoint variant of interpolated_pose_target_match.

    waypoints: [{"frac": 0.0, "overrides": None},
                {"frac": 0.4, "overrides": FOLD_OVERRIDES},
                {"frac": 0.7, "overrides": SIT_OVERRIDES}]

    Use this to enforce a curriculum-style trajectory through one or more
    intermediate poses (e.g. stand → fold → sit). Same per-joint Gaussian
    semantics as the single-stage version.
    """
    asset = env.scene[asset_cfg.name]
    target = _multistage_target_pose(env, asset_cfg, waypoints)
    joint_pos = _servo_joint_pos(env, asset)
    if joint_indices is not None:
        joint_pos = joint_pos[:, joint_indices]
        target = target[:, joint_indices]
    return torch.exp(-((joint_pos - target) / std) ** 2).mean(dim=-1)


def multistage_pose_l1_penalty(
    env: ManagerBasedRlEnv,
    waypoints: list,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    joint_indices: Optional[list] = None,
) -> torch.Tensor:
    """L1 companion to multistage_pose_target_match."""
    asset = env.scene[asset_cfg.name]
    target = _multistage_target_pose(env, asset_cfg, waypoints)
    joint_pos = _servo_joint_pos(env, asset)
    if joint_indices is not None:
        joint_pos = joint_pos[:, joint_indices]
        target = target[:, joint_indices]
    return -torch.abs(joint_pos - target).mean(dim=-1)


def multistage_height_target(
    env: ManagerBasedRlEnv,
    waypoints: list,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    std: float = 0.03,
) -> torch.Tensor:
    """Multi-waypoint Gaussian on trunk z."""
    target_z = _multistage_target_height(env, waypoints)
    asset = env.scene[asset_cfg.name]
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    return torch.exp(-((z - target_z) / std) ** 2)


def multistage_height_l1_penalty(
    env: ManagerBasedRlEnv,
    waypoints: list,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """L1 companion to multistage_height_target."""
    target_z = _multistage_target_height(env, waypoints)
    asset = env.scene[asset_cfg.name]
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    return -torch.abs(z - target_z)


def pose_target_match(
    env: ManagerBasedRlEnv,
    target_overrides: Optional[dict] = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    std: float = 0.3,
    joint_indices: Optional[list] = None,
) -> torch.Tensor:
    """Gaussian pose-match against a single fixed target.

    target = ``default_joint_pos`` with the per-index overrides applied. No
    waypoints, no episode-progress interpolation — the same target is rewarded
    from t=0 to the end of the episode.
    """
    asset = env.scene[asset_cfg.name]
    target = _servo_default_joint_pos(env, asset).clone()
    if target_overrides:
        for idx, val in target_overrides.items():
            target[:, idx] = val
    joint_pos = _servo_joint_pos(env, asset)
    if joint_indices is not None:
        joint_pos = joint_pos[:, joint_indices]
        target = target[:, joint_indices]
    return torch.exp(-((joint_pos - target) / std) ** 2).mean(dim=-1)


def pose_l1_penalty(
    env: ManagerBasedRlEnv,
    target_overrides: Optional[dict] = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    joint_indices: Optional[list] = None,
) -> torch.Tensor:
    """L1 companion to ``pose_target_match`` (constant gradient toward target)."""
    asset = env.scene[asset_cfg.name]
    target = _servo_default_joint_pos(env, asset).clone()
    if target_overrides:
        for idx, val in target_overrides.items():
            target[:, idx] = val
    joint_pos = _servo_joint_pos(env, asset)
    if joint_indices is not None:
        joint_pos = joint_pos[:, joint_indices]
        target = target[:, joint_indices]
    return -torch.abs(joint_pos - target).mean(dim=-1)


def height_target_gaussian(
    env: ManagerBasedRlEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    std: float = 0.02,
) -> torch.Tensor:
    """Gaussian on trunk z against a single fixed target."""
    asset = env.scene[asset_cfg.name]
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    return torch.exp(-((z - target_height) / std) ** 2)


def height_l1_penalty(
    env: ManagerBasedRlEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """L1 companion to ``height_target_gaussian``."""
    asset = env.scene[asset_cfg.name]
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    return -torch.abs(z - target_height)


def trunk_vertical_accel_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalty proportional to ``|a_z|`` of the trunk (finite-diff of v_z).

    Captures hard impacts (large deceleration spike on landing) AND incentivises
    a smooth quasi-static descent (constant velocity → a_z ≈ 0). At rest a_z is
    zero so the seated robot pays no cost.

    State is kept on the env in ``_prev_trunk_vz``; at episode reset the
    accel is zeroed to avoid a transient from the previous episode's final
    state leaking into the new one.
    """
    asset = env.scene[asset_cfg.name]
    vz = torch.nan_to_num(asset.data.root_link_lin_vel_w[:, 2], nan=0.0)
    prev = getattr(env, "_prev_trunk_vz", None)
    if prev is None or prev.shape[0] != vz.shape[0]:
        prev = vz.detach().clone()
    a_z = (vz - prev) / env.step_dt
    # Zero out a_z at reset steps to suppress the cross-episode transient.
    if hasattr(env, "episode_length_buf"):
        reset_mask = env.episode_length_buf <= 1
        a_z = torch.where(reset_mask, torch.zeros_like(a_z), a_z)
    env._prev_trunk_vz = vz.detach().clone()
    return -torch.abs(a_z)


def trunk_downward_velocity_penalty(
    env: ManagerBasedRlEnv,
    max_down_vel: float = 0.05,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalty on downward trunk velocity beyond ``max_down_vel``.

    Caps descent SPEED, which ``trunk_vertical_accel_penalty`` alone cannot:
    a fast constant-velocity drop has a_z ≈ 0 the whole way down and pays only
    one impact spike at the bottom — cheap relative to arriving at the target
    pose sooner. This term makes every step of a too-fast descent cost reward,
    so the gentlest descent that stays under the cap is optimal. Zero at rest
    and for any motion slower than the cap (including all upward motion).
    """
    asset = env.scene[asset_cfg.name]
    vz = torch.nan_to_num(asset.data.root_link_lin_vel_w[:, 2], nan=0.0)
    return -torch.clamp(-vz - max_down_vel, min=0.0)


def seated_stillness(
    env: ManagerBasedRlEnv,
    height_full: float = 0.06,
    height_zero: float = 0.08,
    vel_std: float = 0.05,
    tilt_full_deg: float = 25.0,
    tilt_zero_deg: float = 60.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward trunk stillness while seated UPRIGHT: |v| Gaussian, z- and tilt-gated.

    exp(-(|v|/vel_std)²) · smoothstep(z) · smoothstep(tilt). The z gate is full
    below ``height_full`` and zero above ``height_zero`` (inactive during the
    descent). The tilt gate is full below ``tilt_full_deg`` and zero above
    ``tilt_zero_deg`` — WITHOUT it, "lie still on your back" scores as well as
    "sit still upright" (the trunk on its back is inside the seated z band and
    perfectly motionless), which is exactly the exploit run 2 converged to.
    Makes "rest quietly, upright, at the seated height" the only rewarded rest.
    """
    asset = env.scene[asset_cfg.name]
    v = torch.nan_to_num(asset.data.root_link_lin_vel_w, nan=0.0).norm(dim=-1)
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    t = torch.clamp((height_zero - z) / max(height_zero - height_full, 1e-6), 0.0, 1.0)
    z_gate = t * t * (3.0 - 2.0 * t)
    quat = asset.data.root_link_quat_w
    cos_tilt = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    cos_full = math.cos(math.radians(tilt_full_deg))
    cos_zero = math.cos(math.radians(tilt_zero_deg))
    u = torch.clamp((cos_tilt - cos_zero) / max(cos_full - cos_zero, 1e-6), 0.0, 1.0)
    tilt_gate = u * u * (3.0 - 2.0 * u)
    return torch.exp(-((v / vel_std) ** 2)) * z_gate * tilt_gate


def upright_while_tall(
    env: ManagerBasedRlEnv,
    height_low: float,
    height_high: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Linear upright reward weighted by a smoothstep on trunk z.

    Returns ``body_upright_linear * smoothstep((z - low)/(high - low))`` so the
    upright incentive is full while the robot is still standing tall, and
    fades to zero once it has committed to the lower sit configuration (where
    butt-on-ground orientation is fine). Prevents the policy from learning to
    tip backward while still high (which would otherwise farm the descent
    reward via a controlled fall).
    """
    asset = env.scene[asset_cfg.name]
    quat = asset.data.root_link_quat_w
    qx = quat[:, 1]
    qy = quat[:, 2]
    upright = 1.0 - 2.0 * (qx * qx + qy * qy)
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    t = torch.clamp((z - height_low) / max(height_high - height_low, 1e-6), 0.0, 1.0)
    smooth = t * t * (3.0 - 2.0 * t)
    return upright * smooth


def phase_pose_blend(
    phase: torch.Tensor,
    descent_end: float,
    hold_end: float,
    rise_end: float,
) -> torch.Tensor:
    """Blend 0..1 le long de la phase [0,1) — 0 = pose STAND, 1 = pose DOWN.

    [0, descent_end)       : 0 -> 1  (se baisser)
    [descent_end, hold_end): 1       (bas)
    [hold_end, rise_end)   : 1 -> 0  (se lever)
    [rise_end, 1.0)        : 0       (haut / repos)
    """
    b = torch.zeros_like(phase)
    descend = phase < descent_end
    b = torch.where(descend, phase / descent_end, b)
    low = (phase >= descent_end) & (phase < hold_end)
    b = torch.where(low, torch.ones_like(phase), b)
    rise = (phase >= hold_end) & (phase < rise_end)
    b = torch.where(rise, 1.0 - (phase - hold_end) / (rise_end - hold_end), b)
    return b


def kick_pose_target(
    phase: torch.Tensor,
    stand: torch.Tensor,
    back: torch.Tensor,
    forward: torch.Tensor,
    windup_end: float,
    kick_end: float,
    return_end: float,
) -> torch.Tensor:
    """Cible articulaire interpolée d'un geste de shoot à 4 keyframes.

    phase (B,) ∈ [0,1). stand/back/forward (k,) ou (1,k). Retour (B,k).

    [0, windup_end)        STAND   -> BACK     (armement)
    [windup_end, kick_end) BACK    -> FORWARD  (frappe sèche)
    [kick_end, return_end) FORWARD -> STAND    (retour)
    [return_end, 1.0)      STAND             (repos)
    """
    p = phase.unsqueeze(-1)  # (B,1)

    def interp(a, b, s):
        return a + s * (b - a)

    s1 = (p / windup_end).clamp(0.0, 1.0)
    s2 = ((p - windup_end) / (kick_end - windup_end)).clamp(0.0, 1.0)
    s3 = ((p - kick_end) / (return_end - kick_end)).clamp(0.0, 1.0)

    seg1 = interp(stand, back, s1)
    seg2 = interp(back, forward, s2)
    seg3 = interp(forward, stand, s3)  # à s3=1 (phase>=return_end) => STAND

    out = seg1
    out = torch.where(p >= windup_end, seg2, out)
    out = torch.where(p >= kick_end, seg3, out)
    return out


def _kick_pose_error(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    stand_pose: dict,
    back_pose: dict,
    forward_pose: dict,
    windup_end: float,
    kick_end: float,
    return_end: float,
    joint_names: Optional[list] = None,
):
    """(cur, target) pour le geste de shoot, joints résolus PAR NOM.

    Les 3 poses partagent les mêmes clés (14 joints). L'ordre des noms est
    donné par `stand_pose` (ou par `joint_names` si fourni — un sous-ensemble
    des clés, ex. jambe droite + cou d'un côté, jambe gauche de l'autre, pour
    appliquer des std différents au geste vs à la jambe d'appui).
    """
    if not stand_pose:
        raise ValueError("_kick_pose_error requires a non-empty stand_pose dict")
    asset: Entity = env.scene[asset_cfg.name]
    names = list(joint_names) if joint_names is not None else list(stand_pose.keys())
    ids = [int(asset.find_joints([n])[0][0]) for n in names]

    def vec(d):
        return torch.tensor([d[n] for n in names], device=env.device,
                            dtype=asset.data.joint_pos.dtype)

    stand_v, back_v, fwd_v = vec(stand_pose), vec(back_pose), vec(forward_pose)

    cmd = env.command_manager.get_command(command_name)
    phase = (torch.atan2(cmd[:, 1], cmd[:, 0]) / (2 * torch.pi)) % 1.0  # (B,)
    target = kick_pose_target(phase, stand_v, back_v, fwd_v,
                              windup_end, kick_end, return_end)          # (B,k)
    cur = asset.data.joint_pos[:, ids]                                   # (B,k)
    return cur, target


def kick_pose_track(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    stand_pose: Optional[dict] = None,
    back_pose: Optional[dict] = None,
    forward_pose: Optional[dict] = None,
    std: float = 0.4,
    windup_end: float = 0.35,
    kick_end: float = 0.45,
    return_end: float = 0.75,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    joint_names: Optional[list] = None,
) -> torch.Tensor:
    """Gaussienne sur la pose articulaire vs cible interpolée du shoot.

    Reward directif et symétrique : chaque phase impose la config articulaire
    exacte. Résolution PAR NOM. `joint_names` restreint l'évaluation à un
    sous-ensemble (ex. jambe droite + cou tracés serré, jambe gauche d'appui
    tracée lâche pour la laisser équilibrer).
    """
    cur, target = _kick_pose_error(
        env, asset_cfg, command_name, stand_pose or {}, back_pose or {},
        forward_pose or {}, windup_end, kick_end, return_end, joint_names,
    )
    return torch.exp(-((cur - target) / std) ** 2).mean(dim=-1)


def kick_pose_track_l1(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    stand_pose: Optional[dict] = None,
    back_pose: Optional[dict] = None,
    forward_pose: Optional[dict] = None,
    windup_end: float = 0.35,
    kick_end: float = 0.45,
    return_end: float = 0.75,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    joint_names: Optional[list] = None,
) -> torch.Tensor:
    """Bootstrap L1 vers la cible interpolée (gradient constant, pénalité<=0)."""
    cur, target = _kick_pose_error(
        env, asset_cfg, command_name, stand_pose or {}, back_pose or {},
        forward_pose or {}, windup_end, kick_end, return_end, joint_names,
    )
    return -(cur - target).abs().mean(dim=-1)


def kick_engagement(
    phase: torch.Tensor,
    windup_end: float,
    return_end: float,
) -> torch.Tensor:
    """Gate d'engagement du geste ∈ [0,1] (pur) — pour pondérer les rewards
    d'équilibre unipède qui ne doivent s'appliquer que hors du repos STAND.

    [0, windup_end)        : 0 -> 1  (montée pendant l'armement)
    [windup_end, return_end): 1       (phase de frappe = appui unipède attendu)
    [return_end, 1.0)      : 0        (repos STAND, appui bipède, CoM centré OK)
    """
    g = torch.zeros_like(phase)
    ramp = phase < windup_end
    g = torch.where(ramp, phase / windup_end, g)
    hold = (phase >= windup_end) & (phase < return_end)
    g = torch.where(hold, torch.ones_like(phase), g)
    return g


def com_over_support_foot(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str = "twist",
    std: float = 0.04,
    windup_end: float = 0.35,
    return_end: float = 0.75,
) -> torch.Tensor:
    """Reward gaussien : projection horizontale du CoM proche du pied d'appui,
    gaté sur la phase de frappe (kick_engagement).

    Apprend le transfert latéral du poids sur le pied d'appui (support). Sans
    ça, un geste à un pied issu de poses relevées en appui bipède garde le CoM
    centré entre les deux pieds → bascule et chute dès que l'autre pied se lève.
    Au repos STAND le gate est 0 (appui bipède, CoM centré autorisé).

    `asset_cfg` doit cibler le site du pied d'appui (ex. site_names=["left_foot"]).
    `std` en mètres (rayon de tolérance CoM↔pied, ~taille du pied).
    """
    asset: Entity = env.scene[asset_cfg.name]
    com_xy = asset.data.root_com_pos_w[:, :2]
    foot_id = asset_cfg.site_ids[0]
    foot_xy = asset.data.site_pos_w[:, foot_id, :2]
    dist2 = ((com_xy - foot_xy) ** 2).sum(dim=-1)
    reward = torch.exp(-dist2 / (std ** 2))

    cmd = env.command_manager.get_command(command_name)
    phase = (torch.atan2(cmd[:, 1], cmd[:, 0]) / (2 * torch.pi)) % 1.0
    gate = kick_engagement(phase, windup_end, return_end)
    return gate * reward


def _phase_pose_error(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    target_pose: dict,
    descent_end: float,
    hold_end: float,
    rise_end: float,
    source_pose: Optional[dict] = None,
):
    """(cur, target) pour la pose interpolée par la phase, résolue PAR NOM.

    Cible = source + blend(phase)·(target_pose - source), source = STAND
    (`source_pose` si fourni, sinon le DEFAULT/HOME du modèle). blend ∈ [0,1]
    (0 = STAND, 1 = target_pose) via `phase_pose_blend`.
    """
    if not target_pose:
        raise ValueError("_phase_pose_error requires a non-empty target_pose dict")

    asset: Entity = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    phase = (torch.atan2(cmd[:, 1], cmd[:, 0]) / (2 * torch.pi)) % 1.0  # (B,)
    blend = phase_pose_blend(phase, descent_end, hold_end, rise_end)     # (B,)

    names = list(target_pose.keys())
    ids = [int(asset.find_joints([n])[0][0]) for n in names]
    default = asset.data.default_joint_pos[:, ids]                       # (B,k)

    source = default.clone()
    if source_pose:
        for j, n in enumerate(names):
            if n in source_pose:
                source[:, j] = source_pose[n]
    target_vec = torch.tensor(
        [target_pose[n] for n in names], device=env.device, dtype=default.dtype
    ).unsqueeze(0)                                                       # (1,k)

    target = source + blend.unsqueeze(-1) * (target_vec - source)        # (B,k)
    cur = asset.data.joint_pos[:, ids]                                   # (B,k)
    return cur, target


def phase_pose_track(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    target_pose: Optional[dict] = None,
    source_pose: Optional[dict] = None,
    std: float = 0.3,
    descent_end: float = 0.15,
    hold_end: float = 0.50,
    rise_end: float = 0.65,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Gaussienne sur la pose articulaire vs cible interpolée STAND<->DOWN.

    Reward directif : indique la config articulaire exacte à chaque phase. Se
    relever (cible → STAND) est récompensé exactement comme se baisser (cible →
    DOWN) — symétrique par construction. Résolution PAR NOM.
    """
    cur, target = _phase_pose_error(
        env, asset_cfg, command_name, target_pose or {},
        descent_end, hold_end, rise_end, source_pose,
    )
    return torch.exp(-((cur - target) / std) ** 2).mean(dim=-1)


def phase_pose_track_l1(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    target_pose: Optional[dict] = None,
    source_pose: Optional[dict] = None,
    descent_end: float = 0.15,
    hold_end: float = 0.50,
    rise_end: float = 0.65,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Bootstrap L1 vers la cible interpolée (pénalité négative).

    Gradient constant partout — donne une direction vers la cible même quand la
    gaussienne ci-dessus a saturé à ~0 loin de la cible.
    """
    cur, target = _phase_pose_error(
        env, asset_cfg, command_name, target_pose or {},
        descent_end, hold_end, rise_end, source_pose,
    )
    return -(cur - target).abs().mean(dim=-1)


def phase_pose_match(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    std: float = 0.3,
    command_name: str = "twist",
    joint_indices: Optional[list] = None,
    target_overrides: Optional[dict] = None,
    phase: str = "approach",
) -> torch.Tensor:
    """Reward matching a target pose, weighted by phase-cycle command.

    Generic helper for phase-conditioned tasks (e.g. sit/stand). The command
    encodes phase as [cos(2π·phase), sin(2π·phase), 0]:
      - "approach" weight = max(0, sin(2π·phase)) — peaks at phase 0.25.
      - "return"   weight = max(0,-sin(2π·phase)) — peaks at phase 0.75.

    Args:
        std: Gaussian std per joint (rad).
        joint_indices: Optional subset of joints to evaluate (rest ignored).
        target_overrides: {joint_index: angle_rad}. Joints not listed default
            to asset.data.default_joint_pos (the home/standing pose).
        phase: "approach" or "return".
    """
    asset = env.scene[asset_cfg.name]
    joint_pos = _servo_joint_pos(env, asset)
    target = _servo_default_joint_pos(env, asset).clone()
    if target_overrides:
        for idx, val in target_overrides.items():
            target[:, idx] = val
    if joint_indices is not None:
        joint_pos = joint_pos[:, joint_indices]
        target = target[:, joint_indices]
    pose_reward = torch.exp(-((joint_pos - target) / std) ** 2).mean(dim=-1)

    cmd = env.command_manager.get_command(command_name)
    if phase == "approach":
        weight = torch.clamp(cmd[:, 1], min=0.0)
    else:
        weight = torch.clamp(-cmd[:, 1], min=0.0)
    return weight * pose_reward


def ground_pick_return_pose(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    std: float = 0.3,
    command_name: str = "twist",
    joint_indices: Optional[list] = None,
) -> torch.Tensor:
    """Reward for returning to the standing pose after ground pick, weighted by the return phase.

    The return phase is the second half-cycle (sin < 0, phase ∈ [0.5, 1.0]),
    smoothly weighted by max(0, -sin(2π*phase)).

    Args:
        std: Gaussian std per joint (rad).
        joint_indices: Subset of joints to evaluate. Use to apply different stds
            to leg joints vs neck/head joints (call this reward twice).
    """
    asset = env.scene[asset_cfg.name]
    joint_pos  = _servo_joint_pos(env, asset)        # (num_envs, n_servo_joints)
    default_pos = _servo_default_joint_pos(env, asset)

    if joint_indices is not None:
        joint_pos   = joint_pos[:, joint_indices]
        default_pos = default_pos[:, joint_indices]

    pose_reward = torch.exp(-((joint_pos - default_pos) / std) ** 2).mean(dim=-1)

    # Return weight: max(0, -sin(2π*phase)) — peaks at 1 at phase=0.75, zero at 0.5 and 1
    cmd = env.command_manager.get_command(command_name)
    return_weight = torch.clamp(-cmd[:, 1], min=0.0)

    return return_weight * pose_reward


def ground_pick_return_upright(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    std: float = 0.4,
    command_name: str = "twist",
) -> torch.Tensor:
    """Reward trunk verticality, weighted by the RETURN phase (stand-up aid).

    Same return weighting as ``ground_pick_return_pose`` (``max(0, -sin(2π·phase))``)
    so it only rewards being upright during the stand-up, never fighting the
    forward lean of the approach. Verticality = ``exp(-tilt²/std²)`` with the same
    tilt proxy as ``body_upright_gaussian`` (``2*(qx²+qy²) ≈ 1-cos(tilt)``). A broad
    std (0.4 rad ≈ 23°) gives gradient even from a fairly tilted crouch.
    """
    asset: Entity = env.scene[asset_cfg.name]
    quat = asset.data.root_link_quat_w
    tilt_sq = 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)  # qx² + qy²
    upright = torch.exp(-tilt_sq / (std * std))
    cmd = env.command_manager.get_command(command_name)
    return_weight = torch.clamp(-cmd[:, 1], min=0.0)
    return return_weight * upright


# --------------------------------------------------------------------------- #
# Ground-pick : gating de phase SEGMENTÉ (durées descente/palier/remontée/repos #
# indépendantes, au lieu de la pondération sinusoïdale max(0,±sin)).            #
#   down-gate  = phase_pose_blend(phase, descent_end, hold_end, rise_end)       #
#               0 (haut) -> 1 (descente) -> 1 (palier bas) -> 0 (remontée/repos) #
#   up-gate    = phase_rise_gate(phase, hold_end, rise_end)                      #
#               0 avant la remontée -> 0..1 (remontée) -> 1 (repos debout)       #
# --------------------------------------------------------------------------- #
def phase_rise_gate(
    phase: torch.Tensor, hold_end: float, rise_end: float
) -> torch.Tensor:
    """Gate montante pour le RETOUR : 0 avant hold_end, 0->1 sur [hold_end,
    rise_end), 1 après (repos debout)."""
    g = torch.zeros_like(phase)
    rising = (phase >= hold_end) & (phase < rise_end)
    g = torch.where(rising, (phase - hold_end) / (rise_end - hold_end), g)
    g = torch.where(phase >= rise_end, torch.ones_like(phase), g)
    return g


def _gp_phase(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    cmd = env.command_manager.get_command(command_name)
    return (torch.atan2(cmd[:, 1], cmd[:, 0]) / (2 * torch.pi)) % 1.0


def mouth_ground_proximity_phased(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=["mouth_tip"]),
    std: float = 0.10,
    target_height: float = 0.0,
    command_name: str = "twist",
    descent_end: float = 0.25,
    hold_end: float = 0.35,
    rise_end: float = 0.60,
) -> torch.Tensor:
    """mouth_ground_proximity gaté par la down-gate segmentée (descente+palier)."""
    asset = env.scene[asset_cfg.name]
    mouth_z = asset.data.site_pos_w[:, asset_cfg.site_ids[0], 2]
    proximity = torch.exp(-((mouth_z - target_height) / std) ** 2)
    gate = phase_pose_blend(_gp_phase(env, command_name), descent_end, hold_end, rise_end)
    return gate * proximity


def mouth_perpendicular_phased(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=["mouth_tip"]),
    command_name: str = "twist",
    descent_end: float = 0.25,
    hold_end: float = 0.35,
    rise_end: float = 0.60,
) -> torch.Tensor:
    """mouth_perpendicular_to_ground gaté par la down-gate segmentée."""
    asset = env.scene[asset_cfg.name]
    q = asset.data.site_quat_w[:, asset_cfg.site_ids[0], :]
    w, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    x_axis_z = 2.0 * (qx * qz - w * qy)
    alignment = -x_axis_z  # 1 = bouche pointe droit vers le bas
    gate = phase_pose_blend(_gp_phase(env, command_name), descent_end, hold_end, rise_end)
    return gate * alignment


def ground_pick_return_pose_phased(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    std: float = 0.3,
    command_name: str = "twist",
    joint_indices: Optional[list] = None,
    hold_end: float = 0.35,
    rise_end: float = 0.60,
) -> torch.Tensor:
    """ground_pick_return_pose gaté par la up-gate segmentée (remontée+repos)."""
    asset = env.scene[asset_cfg.name]
    joint_pos = _servo_joint_pos(env, asset)
    default_pos = _servo_default_joint_pos(env, asset)
    if joint_indices is not None:
        joint_pos = joint_pos[:, joint_indices]
        default_pos = default_pos[:, joint_indices]
    pose_reward = torch.exp(-((joint_pos - default_pos) / std) ** 2).mean(dim=-1)
    gate = phase_rise_gate(_gp_phase(env, command_name), hold_end, rise_end)
    return gate * pose_reward


def ground_pick_return_upright_phased(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    std: float = 0.4,
    command_name: str = "twist",
    hold_end: float = 0.35,
    rise_end: float = 0.60,
) -> torch.Tensor:
    """ground_pick_return_upright gaté par la up-gate segmentée."""
    asset: Entity = env.scene[asset_cfg.name]
    quat = asset.data.root_link_quat_w
    tilt_sq = 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    upright = torch.exp(-tilt_sq / (std * std))
    gate = phase_rise_gate(_gp_phase(env, command_name), hold_end, rise_end)
    return gate * upright


def neck_vel_descent_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    command_name: str = "twist",
    joint_indices: Optional[list] = None,
    hold_end: float = 0.35,
) -> torch.Tensor:
    """Pénalise la vitesse des joints du cou pendant la DESCENTE+palier (freine le
    piqué de la tête).

    Coût = mean(joint_vel²) sur les joints donnés, gaté à 1 pour phase < hold_end
    (descente + palier bas) et 0 ensuite (remontée + repos) -> ne gêne PAS le
    relever du cou. Retourne un coût positif ; à utiliser avec un poids négatif.
    """
    asset = env.scene[asset_cfg.name]
    vel = _servo_joint_vel(env, asset)
    if joint_indices is not None:
        vel = vel[:, joint_indices]
    cost = (vel ** 2).mean(dim=-1)
    phase = _gp_phase(env, command_name)
    gate = (phase < hold_end).to(vel.dtype)  # descente + palier bas uniquement
    return gate * cost


def sample_mouth_payload(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    min_kg: float = 0.01,
    max_kg: float = 0.04,
) -> None:
    """Event de reset : tire une masse d'objet 'tenu dans la bouche' par env (kg),
    stockée sur env._mouth_payload_kg. Utilisée par apply_mouth_payload_force."""
    buf = getattr(env, "_mouth_payload_kg", None)
    if buf is None:
        buf = torch.zeros(env.num_envs, device=env.device)
        env._mouth_payload_kg = buf
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    buf[env_ids] = torch.rand(len(env_ids), device=env.device) * (max_kg - min_kg) + min_kg


def apply_mouth_payload_force(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["jaw_soft"], site_names=["mouth_tip"]
    ),
    command_name: str = "twist",
    hold_end: float = 0.35,
    ramp: float = 0.05,
    gravity: float = 9.81,
) -> torch.Tensor:
    """Hook par-step (utilisé comme reward de poids 0) : applique le POIDS de
    l'objet tenu dans la bouche comme force externe verticale au mouth_tip, gaté
    sur la remontée (phase >= hold_end, rampe rapide au moment du 'grab').

    Émule une masse ponctuelle au bout de la bouche pendant le relever : la force
    m·g est appliquée au CoM du corps + le couple (p_mouth - p_com) × F, ce qui
    équivaut à l'appliquer au mouth_tip (bon bras de levier pour le cou). Retourne
    0 (ce n'est pas une vraie récompense — juste le hook d'application)."""
    asset: Entity = env.scene[asset_cfg.name]
    payload = getattr(env, "_mouth_payload_kg", None)
    if payload is None:
        return torch.zeros(env.num_envs, device=env.device)
    phase = _gp_phase(env, command_name)
    gate = ((phase - hold_end) / ramp).clamp(0.0, 1.0)  # 0 avant grab -> 1 après
    fz = -(gate * payload) * gravity                     # (N,) force verticale (bas)

    bid = int(asset_cfg.body_ids[0])
    sid = int(asset_cfg.site_ids[0])
    p_mouth = asset.data.site_pos_w[:, sid, :]           # (N,3)
    p_com = asset.data.body_com_pos_w[:, bid, :]         # (N,3)
    F = torch.zeros((env.num_envs, 3), device=env.device, dtype=p_mouth.dtype)
    F[:, 2] = fz
    tau = torch.cross(p_mouth - p_com, F, dim=-1)        # applique F au mouth_tip
    asset.write_external_wrench_to_sim(
        forces=F.unsqueeze(1), torques=tau.unsqueeze(1), body_ids=[bid],
    )
    return torch.zeros(env.num_envs, device=env.device)


# ==============================================================================
# Domain Randomization Events
# ==============================================================================


def randomize_delayed_actuator_gains(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    kp_range: tuple[float, float],
    kd_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    operation: str = "scale",
):
    """Randomize firmware PD gains per episode (NON-accumulating).

    Under the canonical BAM actuator (``bam.mjlab.BamActuator``) gains are scaled
    per-env via ``set_gains``/``reset_gains`` (the actuator owns ``kp_scale``/
    ``kd_scale``), so we never touch the MuJoCo model — no accumulation risk. The
    sampled per-joint factors are averaged into a single scalar per env (the
    actuator applies one scale across its joints), matching the previous behavior.
    Non-BAM actuators are skipped (e.g. the roller XmlActuator, which doesn't
    expose set_gains).

    Args:
        env: The environment
        env_ids: Environment IDs to randomize (None = all envs)
        kp_range: (min, max) for kp randomization
        kd_range: (min, max) for kd randomization
        asset_cfg: Asset configuration
        operation: unused (kept for cfg compatibility; scaling is always applied)
    """
    del operation
    from bam.mjlab import BamActuator

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.int)

    asset: Entity = env.scene[asset_cfg.name]

    for actuator in asset.actuators:
        if not isinstance(actuator, BamActuator):
            continue
        n_joints = len(actuator.ctrl_ids)
        kp_samples = torch.rand(len(env_ids), n_joints, device=env.device) * (kp_range[1] - kp_range[0]) + kp_range[0]
        kd_samples = torch.rand(len(env_ids), n_joints, device=env.device) * (kd_range[1] - kd_range[0]) + kd_range[0]
        # Restore nominal first (prevents accumulation), then apply fresh scale.
        actuator.reset_gains(env_ids)
        actuator.set_gains(
            env_ids,
            kp_scale=kp_samples.mean(dim=1, keepdim=True),
            kd_scale=kd_samples.mean(dim=1, keepdim=True),
        )


@requires_model_fields("dof_frictionloss", "dof_damping")
def expand_bam_friction_fields(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
):
    """No-op startup event whose only purpose is the decorator above.

    bam's BamActuator (mjlab_frictionloss branch) writes a per-env friction
    budget into MuJoCo's dof_frictionloss/dof_damping every step, which
    requires those model fields to be expanded per world. mjlab expands
    exactly the fields declared by event functions via requires_model_fields,
    so every env using the BAM actuator must register this event.
    """


def randomize_bam_friction(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    scale_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
):
    """Per-episode joint-friction randomization for the BAM actuator (NON-accumulating).

    Under BAM, MuJoCo's dof_frictionloss is zeroed (BAM computes friction in
    compute()), so stock dr.dof_frictionloss is a no-op. Instead this samples a
    per-env scalar in ``scale_range`` and applies it to the FrictionDRBamActuator's
    ``friction_scale``, which multiplies BAM's velocity-independent friction budget
    (Coulomb + Stribeck + load). Restores nominal (1.0) first to avoid accumulation.
    No-op on actuators without a friction_scale hook.
    """
    from mjlab_microduck.actuator.friction_dr_bam import FrictionDRBamActuator

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.int)

    asset: Entity = env.scene[asset_cfg.name]
    lo, hi = scale_range
    for actuator in asset.actuators:
        if isinstance(actuator, FrictionDRBamActuator):
            actuator.reset_friction_scale(env_ids)
            samples = torch.rand(len(env_ids), 1, device=env.device) * (hi - lo) + lo
            actuator.set_friction_scale(env_ids, samples)


def randomize_mass_and_inertia(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    scale_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
):
    """Randomize body mass and inertia together with the same scaling factor.

    This maintains physical consistency - mass and inertia must scale together
    to avoid creating invalid inertia tensors that cause simulation instability.

    Args:
        env: The environment
        env_ids: Environment IDs to randomize
        scale_range: (min, max) scaling factor applied to both mass and inertia
        asset_cfg: Asset configuration specifying which bodies to randomize
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.int)

    asset: Entity = env.scene[asset_cfg.name]

    # Get body indices
    body_ids = asset_cfg.body_ids
    if isinstance(body_ids, slice):
        body_ids = list(range(asset.num_bodies))[body_ids]
    body_indices = asset.indexing.body_ids[body_ids]

    # Sample ONE random scale per environment (applied to both mass and inertia)
    num_envs = len(env_ids)
    num_bodies = len(body_indices)
    scales = torch.rand(num_envs, num_bodies, device=env.device) * (scale_range[1] - scale_range[0]) + scale_range[0]

    # Store original values on first call
    if not hasattr(env, '_original_mass_inertia'):
        env._original_mass_inertia = {
            'mass': env.sim.model.body_mass[0, body_indices].clone(),
            'inertia': env.sim.model.body_inertia[0, body_indices].clone(),
        }

    # Reset to original first (to prevent accumulation)
    original = env._original_mass_inertia
    env.sim.model.body_mass[env_ids[:, None], body_indices] = original['mass'].unsqueeze(0).expand(num_envs, -1)
    env.sim.model.body_inertia[env_ids[:, None], body_indices] = original['inertia'].unsqueeze(0).expand(num_envs, -1, -1)

    # Apply same scale to both mass and inertia
    env.sim.model.body_mass[env_ids[:, None], body_indices] *= scales
    env.sim.model.body_inertia[env_ids[:, None], body_indices] *= scales.unsqueeze(-1)  # Scale all 3 inertia components


def standing_envs_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    standing_stages: list[dict],
) -> torch.Tensor:
    """Update the relative number of standing environments based on training progress.

    Args:
        env: The RL environment
        env_ids: Environment IDs (unused, but required by curriculum interface)
        command_name: Name of the velocity command term
        standing_stages: List of dicts with 'step' and 'rel_standing_envs' keys
            Example: [
                {"step": 0, "rel_standing_envs": 0.02},
                {"step": 1000, "rel_standing_envs": 0.1},
                {"step": 2000, "rel_standing_envs": 0.2},
            ]

    Returns:
        Current rel_standing_envs value as a tensor
    """
    del env_ids  # Unused

    from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
    from typing import cast

    command_term = env.command_manager.get_term(command_name)
    assert command_term is not None, f"Command term '{command_name}' not found"

    cfg = cast(UniformVelocityCommandCfg, command_term.cfg)

    # Update rel_standing_envs based on current step
    for stage in standing_stages:
        if env.common_step_counter > stage["step"]:
            cfg.rel_standing_envs = stage["rel_standing_envs"]

    return torch.tensor([cfg.rel_standing_envs])


def velocity_tracking_std_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    reward_name: str,
    std_stages: list[dict],
) -> torch.Tensor:
    """Update velocity tracking std parameter based on training progress.

    Starts with loose std (easy rewards) to learn basic walking, then gradually
    tightens to improve velocity tracking accuracy.

    Args:
        env: The RL environment
        env_ids: Environment IDs (unused, but required by curriculum interface)
        reward_name: Name of the reward term (e.g., "track_linear_velocity")
        std_stages: List of dicts with 'step' and 'std' keys
            Example: [
                {"step": 0, "std": 0.5},      # Start loose - learn to walk
                {"step": 250, "std": 0.3},     # Moderate - refine gait
                {"step": 500, "std": 0.2},     # Strict - accurate tracking
            ]

    Returns:
        Current std value as a tensor
    """
    del env_ids  # Unused

    # Get reward term configuration
    reward_term_cfg = env.reward_manager.get_term_cfg(reward_name)

    # Update std based on current step
    current_std = std_stages[0]["std"]  # Default to first stage

    for stage in std_stages:
        if env.common_step_counter > stage["step"]:
            current_std = stage["std"]

    # Update the reward term's std parameter
    reward_term_cfg.params["std"] = current_std

    return torch.tensor([current_std])


def push_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    event_name: str,
    push_stages: list[dict],
) -> torch.Tensor:
    """Update push velocity range based on training progress.

    Starts with no/small pushes to learn clean walking, then gradually increases
    to build robustness without disrupting early learning.

    Args:
        env: The RL environment
        env_ids: Environment IDs (unused, but required by curriculum interface)
        event_name: Name of the push event term (e.g., "push_robot")
        push_stages: List of dicts with 'step' and 'velocity_range' keys
            Example: [
                {"step": 0, "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)}},
                {"step": 250, "velocity_range": {"x": (-0.15, 0.15), "y": (-0.15, 0.15)}},
                {"step": 500, "velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}},
            ]

    Returns:
        Current max push magnitude as a tensor
    """
    del env_ids  # Unused

    # NOTE: must update the live EventManager term_cfg, not env.cfg.events —
    # EventManager.__init__ does deepcopy(cfg), so mutating env.cfg.events is a no-op.
    event_cfg = env.event_manager.get_term_cfg(event_name)

    # Update velocity_range based on current step
    current_range = push_stages[0]["velocity_range"]  # Default to first stage

    for stage in push_stages:
        if env.common_step_counter > stage["step"]:
            current_range = stage["velocity_range"]

    # Update the event configuration's velocity_range parameter
    event_cfg.params["velocity_range"] = current_range

    # Return max magnitude for logging
    max_push = max(abs(current_range["x"][0]), abs(current_range["x"][1]))
    return torch.tensor([max_push])


def wheel_friction_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    event_name: str,
    ranges_stages: list[dict],
) -> torch.Tensor:
    """Update wheel friction based on training step stages."""
    del env_ids  # Unused

    current_ranges = ranges_stages[0]["ranges"]
    for stage in ranges_stages:
        if env.common_step_counter > stage["step"]:
            current_ranges = stage["ranges"]

    env.event_manager.get_term_cfg(event_name).params["ranges"] = current_ranges
    return torch.tensor([current_ranges[0]])


def reward_weight(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    reward_name: str,
    weight_stages: list[dict],
) -> torch.Tensor:
    """Step-staged reward weight curriculum.

    mjlab 1.3.0 dropped the built-in ``mdp.reward_weight`` helper, so microduck
    provides its own. ``weight_stages`` is a list of ``{"step": int, "weight":
    float}`` dicts; the weight of the latest stage whose step has elapsed is
    applied. Mutates the live RewardManager term cfg (not env.cfg, which is a
    deepcopy at manager init).
    """
    del env_ids
    term_cfg = env.reward_manager.get_term_cfg(reward_name)
    for stage in weight_stages:
        if env.common_step_counter > stage["step"]:
            term_cfg.weight = stage["weight"]
    return torch.tensor([term_cfg.weight])


def com_range_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    event_name: str,
    range_stages: list[dict],
) -> torch.Tensor:
    """Update CoM randomization range based on training progress.

    Gradually increases the CoM offset range so the robot first learns to walk
    with a small CoM uncertainty, then progressively larger.

    Args:
        env: The RL environment
        env_ids: Environment IDs (unused)
        event_name: Name of the CoM randomization event (e.g., "randomize_com")
        range_stages: List of dicts with 'step' and 'range' keys (range in meters)
            Example: [
                {"step": 0,          "range": 0.003},
                {"step": 1000 * 24,  "range": 0.005},
                {"step": 2000 * 24,  "range": 0.008},
            ]

    Returns:
        Current range value as a tensor (for logging)
    """
    del env_ids

    # NOTE: must update the live EventManager term_cfg, not env.cfg.events —
    # EventManager.__init__ does deepcopy(cfg), so mutating env.cfg.events is a no-op.
    event_cfg = env.event_manager.get_term_cfg(event_name)

    current_range = range_stages[0]["range"]
    for stage in range_stages:
        if env.common_step_counter > stage["step"]:
            current_range = stage["range"]

    event_cfg.params["ranges"] = (-current_range, current_range)
    return torch.tensor([current_range])


def slope_move_masks(distance: "torch.Tensor", size_x: float):
    """Masques de promotion/rétrogradation du curriculum de pente.

    move_up   : a parcouru plus de 40% de la tuile → il a dévalé la rampe,
                on la rend plus raide. Aligné sur la termination
                terrain_edge_reached (~3.8 m, threshold_fraction=0.95 par
                défaut sur size_x=8.0), qui termine l'épisode avant le seuil
                de moitié (4.0 m) — sans cet alignement un traverseur réussi
                n'est jamais promu.
    move_down : a à peine avancé (< 20% de la tuile) → chute/blocage précoce,
                on adoucit la rampe.
    """
    move_up = distance > size_x * 0.4
    move_down = (distance < size_x * 0.2) & (~move_up)
    return move_up, move_down


def terrain_levels_slope(env: ManagerBasedRlEnv, env_ids: torch.Tensor) -> torch.Tensor:
    """Curriculum de raideur pour roller_slope (pas de vitesse commandée).

    Progression basée sur la distance en x parcourue depuis l'origine de spawn.
    """
    asset = env.scene["robot"]
    terrain = env.scene.terrain
    assert terrain is not None
    terrain_generator = terrain.cfg.terrain_generator
    assert terrain_generator is not None

    distance = (
        asset.data.root_link_pos_w[env_ids, 0] - env.scene.env_origins[env_ids, 0]
    )
    move_up, move_down = slope_move_masks(distance, terrain_generator.size[0])
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


def velocity_command_ranges_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    velocity_stages: list[dict],
    update_lin_vel_y: bool = True,
    update_ang_vel_z: bool = True,
    forward_only: bool = False,
) -> torch.Tensor:
    """Update velocity command ranges based on training progress.

    Gradually increases the commanded velocity ranges to allow the robot to learn
    higher speeds progressively. Starts with smaller ranges for stable learning,
    then expands to more challenging velocities.

    Args:
        env: The RL environment
        env_ids: Environment IDs (unused, but required by curriculum interface)
        command_name: Name of the velocity command term (e.g., "twist")
        velocity_stages: List of dicts with 'step', 'lin_vel_range', and 'ang_vel_range' keys
            Example: [
                {"step": 0, "lin_vel_range": 0.3, "ang_vel_range": 1.5},
                {"step": 500 * 24, "lin_vel_range": 0.4, "ang_vel_range": 1.75},
                {"step": 1000 * 24, "lin_vel_range": 0.5, "ang_vel_range": 2.0},
            ]

    Returns:
        Current max linear velocity as a tensor
    """
    del env_ids  # Unused

    from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
    from typing import cast

    command_term = env.command_manager.get_term(command_name)
    assert command_term is not None, f"Command term '{command_name}' not found"

    cfg = cast(UniformVelocityCommandCfg, command_term.cfg)

    # Update velocity ranges based on current step
    current_lin_vel = velocity_stages[0]["lin_vel_range"]
    current_ang_vel = velocity_stages[0]["ang_vel_range"]

    for stage in velocity_stages:
        if env.common_step_counter > stage["step"]:
            current_lin_vel = stage["lin_vel_range"]
            current_ang_vel = stage["ang_vel_range"]

    # Update command ranges
    if forward_only:
        cfg.ranges.lin_vel_x = (0.0, current_lin_vel)
    else:
        cfg.ranges.lin_vel_x = (-current_lin_vel, current_lin_vel)
    if update_lin_vel_y:
        cfg.ranges.lin_vel_y = (-current_lin_vel, current_lin_vel)
    if update_ang_vel_z:
        cfg.ranges.ang_vel_z = (-current_ang_vel, current_ang_vel)

    return torch.tensor([current_lin_vel])


def projected_gravity(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Projected gravity vector in body frame.

    Returns the gravity vector projected into the robot's body frame,
    representing pure orientation without linear acceleration.
    This is simpler than raw accelerometer and only depends on orientation.

    Returns:
        torch.Tensor: Projected gravity in body frame (num_envs, 3)
    """
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.projected_gravity_b


def _imu_misalignment_quat(env: ManagerBasedRlEnv, max_angle_rad: float) -> torch.Tensor:
    """Per-env constant IMU mounting-misalignment rotation (sampled once).

    Models a fixed small mounting/calibration error of the IMU on each robot.
    Sampled lazily on first use and cached — constant per env for the whole run
    (like a startup randomization), so it's a *systematic per-robot bias*, not
    per-step noise. Replaces the old randomize_imu_orientation event, which wrote
    site_quat (not per-env expanded under mjlab 1.3.0, and not read by the
    projected_gravity / base_ang_vel observations anyway).

    Returns a (num_envs, 4) unit quaternion (w, x, y, z).
    """
    q = getattr(env, "_imu_misalign_quat", None)
    if q is None:
        n = env.num_envs
        axis = torch.randn(n, 3, device=env.device)
        axis = axis / (torch.norm(axis, dim=-1, keepdim=True) + 1e-8)
        angle = torch.rand(n, device=env.device) * max_angle_rad  # [0, max]
        q = quat_from_angle_axis(angle, axis)
        env._imu_misalign_quat = q
    return q


def projected_gravity_imu_misaligned(
    env: ManagerBasedRlEnv,
    max_angle_deg: float = 1.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """projected_gravity with a per-env constant IMU mounting misalignment."""
    asset: Entity = env.scene[asset_cfg.name]
    q = _imu_misalignment_quat(env, math.radians(max_angle_deg))
    return quat_apply(q, asset.data.projected_gravity_b)


def base_ang_vel_imu_misaligned(
    env: ManagerBasedRlEnv,
    max_angle_deg: float = 1.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """base angular velocity with the SAME per-env IMU misalignment as gravity."""
    asset: Entity = env.scene[asset_cfg.name]
    q = _imu_misalignment_quat(env, math.radians(max_angle_deg))
    return quat_apply(q, asset.data.root_link_ang_vel_b)


def raw_accelerometer(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Raw accelerometer reading (includes gravity + linear acceleration).

    Returns normalized raw accelerometer which mimics what a real IMU measures.
    This is different from pure projected_gravity which only reflects orientation.
    Reads from the MuJoCo accelerometer sensor "imu_accel".

    Returns:
        torch.Tensor: Normalized raw accelerometer reading (num_envs, 3)
    """
    asset: Entity = env.scene[asset_cfg.name]

    # Access the model to find the sensor address
    # The accelerometer sensor is the 5th sensor (index 4) in robot.xml
    # Sensors: framequat, gyro, gyro, velocimeter, accelerometer, subtreeangmom
    mj_model = asset.data.model

    # Get sensor address from model arrays (sensor_adr is torch tensor)
    sensor_adr_array = mj_model.sensor_adr  # This is a TorchArray/tensor
    sensor_id = 4  # imu_accel is the 5th sensor (0-indexed)
    sensor_adr = int(sensor_adr_array[sensor_id].item())  # Convert to Python int

    # Read accelerometer data (specific force measured by sensor)
    # Shape: (num_envs, 3)
    accel_raw = asset.data.data.sensordata[:, sensor_adr:sensor_adr+3]

    # MuJoCo accelerometer measures specific force (like real sensor)
    # Negate to match convention: when at rest upright, should point down
    accel_negated = -accel_raw

    # Normalize to unit vector
    accel_norm = torch.norm(accel_negated, dim=-1, keepdim=True)
    accel_normalized = torch.where(
        accel_norm > 0.1,
        accel_negated / accel_norm,
        asset.data.projected_gravity_b  # Fallback to projected gravity
    )

    return accel_normalized

def randomize_imu_orientation(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    max_angle_deg: float = 2.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
):
    """Randomize IMU sensor mounting orientation by small angles.
    
    Simulates slight mounting errors or calibration offsets in the real robot.
    The IMU orientation is randomized by rotating around random axes by up to max_angle_deg.
    
    Args:
        env: The environment
        env_ids: Environment IDs to randomize
        max_angle_deg: Maximum rotation angle in degrees (default 2.0°)
        asset_cfg: Asset configuration
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.int)
    
    asset: Entity = env.scene[asset_cfg.name]

    # IMU site is the first site (index 0) in robot.xml
    # Sites: imu (0), left_foot (1), right_foot (2)
    site_id = 0
    
    # Store original orientation on first call
    if not hasattr(env, '_original_imu_quat'):
        env._original_imu_quat = env.sim.model.site_quat[0, site_id].clone()
    
    # Generate random rotations for each environment
    num_envs = len(env_ids)
    max_angle_rad = max_angle_deg * torch.pi / 180.0
    
    # Random rotation angles [-max_angle, +max_angle] for each axis
    angles = (torch.rand(num_envs, 3, device=env.device) * 2 - 1) * max_angle_rad
    
    # Convert Euler angles to quaternions (small angle approximation for efficiency)
    # For small angles: quat ≈ [1, θx/2, θy/2, θz/2]
    half_angles = angles / 2.0
    quats_delta = torch.zeros(num_envs, 4, device=env.device)
    quats_delta[:, 0] = 1.0  # w component
    quats_delta[:, 1:] = half_angles  # x, y, z components
    
    # Normalize the quaternion
    quats_delta = quats_delta / torch.norm(quats_delta, dim=1, keepdim=True)
    
    # Get original quaternion and apply delta rotation
    original_quat = env._original_imu_quat.unsqueeze(0).expand(num_envs, -1)
    
    # Quaternion multiplication: q_new = q_delta * q_original
    # q1 * q2 = [w1*w2 - dot(v1,v2), w1*v2 + w2*v1 + cross(v1,v2)]
    w1, x1, y1, z1 = quats_delta[:, 0], quats_delta[:, 1], quats_delta[:, 2], quats_delta[:, 3]
    w2, x2, y2, z2 = original_quat[:, 0], original_quat[:, 1], original_quat[:, 2], original_quat[:, 3]
    
    new_quat = torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,  # w
        w1*x2 + x1*w2 + y1*z2 - z1*y2,  # x
        w1*y2 - x1*z2 + y1*w2 + z1*x2,  # y
        w1*z2 + x1*y2 - y1*x2 + z1*w2,  # z
    ], dim=1)
    
    # Apply to the selected environments
    env.sim.model.site_quat[env_ids, site_id] = new_quat


def standing_phase(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Simple time-based phase for standing task.

    Returns a scalar phase value that cycles from 0 to 1 based on time.
    This allows the policy to have a sense of time progression even when standing.

    Args:
        env: The RL environment
        asset_cfg: Not used, but kept for API consistency

    Returns:
        Phase value [0, 1] as tensor of shape (num_envs, 1)
    """
    # Simple time-based phase that cycles every 2 seconds
    # This gives the policy a time-varying signal
    phase_period = 2.0  # seconds
    time = env.episode_length_buf * env.step_dt
    phase = (time % phase_period) / phase_period

    return phase.unsqueeze(-1)  # Shape: (num_envs, 1)


def air_time_adaptive(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str = "twist",
    command_threshold: float = 0.01,    # below this: no reward (standing)
    running_threshold: float = 0.5,     # above this: use running air-time window
    walk_threshold_min: float = 0.10,
    walk_threshold_max: float = 0.25,
    run_threshold_min: float = 0.05,
    run_threshold_max: float = 0.25,
) -> torch.Tensor:
    """Air-time reward with separate swing-time windows for walking vs running.

    - command < command_threshold  → 0 (standing, no reward)
    - command_threshold–running_threshold → walk window [walk_min, walk_max]
    - command > running_threshold  → run  window [run_min,  run_max]

    This lets the walking gait keep its deliberate 100–250 ms swing while
    running can use a faster 50–250 ms cadence.
    """
    sensor = env.scene.sensors[sensor_name]
    current_air_time = sensor.data.current_air_time  # (num_envs, num_feet)
    assert current_air_time is not None

    command = env.command_manager.get_command(command_name)
    total_speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])

    is_walking = ((total_speed >= command_threshold) & (total_speed < running_threshold)).float()  # (num_envs,)
    is_running = (total_speed >= running_threshold).float()

    # Per-env thresholds broadcast over feet
    tmin = (is_walking * walk_threshold_min + is_running * run_threshold_min).unsqueeze(1)
    tmax = (is_walking * walk_threshold_max + is_running * run_threshold_max).unsqueeze(1)

    in_range = (current_air_time > tmin) & (current_air_time < tmax)
    reward = torch.sum(in_range.float(), dim=1)  # sum over feet

    # Zero reward when standing
    active = (total_speed >= command_threshold).float()
    return reward * active


def stillness_at_zero_command(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    command_name: str = "twist",
    command_threshold: float = 0.01,
    vel_std: float = 0.1,
) -> torch.Tensor:
    """Reward staying still when command is near zero.

    Returns exp(-body_vel² / vel_std²) when command < threshold, else 0.
    This is monotonically decreasing with body speed — moving faster is always
    less rewarding. There is no threshold the robot can cross to 'escape' it,
    unlike gate-based stepping penalties.
    """
    asset: Entity = env.scene[asset_cfg.name]

    command = env.command_manager.get_command(command_name)
    total_speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    is_standing_cmd = (total_speed < command_threshold).float()

    body_vel = torch.norm(asset.data.root_link_vel_w[:, :2], dim=1)
    stillness = torch.exp(-body_vel ** 2 / vel_std ** 2)

    return is_standing_cmd * stillness


def joint_vel_l2_when_standing(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    command_name: str = "twist",
    command_threshold: float = 0.01,
) -> torch.Tensor:
    """Penalise leg joint velocities only when command is near zero.

    Targets the standing-shake problem: the policy makes rapid oscillating
    corrections around the home pose when standing. Gated on command so it
    does not affect the walking gait at all.
    """
    asset: Entity = env.scene[asset_cfg.name]

    command = env.command_manager.get_command(command_name)
    total_speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    is_standing_cmd = (total_speed < command_threshold).float()

    leg_indices = list(range(0, 5)) + list(range(9, 14))
    joint_vel = asset.data.joint_vel[:, leg_indices]
    vel_sq = torch.sum(joint_vel ** 2, dim=-1)

    return is_standing_cmd * vel_sq


def foot_step_penalty_when_standing(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    command_name: str = "twist",
    command_threshold: float = 0.01,
    body_vel_threshold: float = 0.2,
    air_time_threshold: float = 0.05,
) -> torch.Tensor:
    """Penalise stepping when at zero command and the body is not being pushed.

    Symmetric counterpart to the air_time reward:
    - air_time gives  +reward for stepping when command > threshold  (walk)
    - this gives      -reward for stepping when command < threshold  (stand)

    The body-velocity gate prevents penalising recovery steps after a push:
    if the robot is already moving fast (pushed), no penalty is applied so it
    can still take steps to catch itself.

    Returns a value in [0, 1] (use a negative weight in the config).
    """
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor = env.scene.sensors["feet_ground_contact"]

    # Was either foot recently lifted? (last completed air phase > threshold)
    air_time = contact_sensor.data.last_air_time[:, :2]  # (num_envs, 2)
    any_foot_stepped = (air_time > air_time_threshold).any(dim=1).float()

    # Are we in standing mode? (command near zero)
    command = env.command_manager.get_command(command_name)
    total_speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    is_standing = (total_speed < command_threshold).float()

    # Is the body still? (not being pushed)
    body_vel = torch.norm(asset.data.root_link_vel_w[:, :2], dim=1)
    is_still = (body_vel < body_vel_threshold).float()

    return any_foot_stepped * is_standing * is_still


def recovery_stepping_reward(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    command_name: str = "twist",
    command_threshold: float = 0.01,
    velocity_threshold: float = 0.3,
    air_time_threshold: float = 0.05,
) -> torch.Tensor:
    """Reward foot air time only when at zero command AND robot has high velocity (recovering from push).

    This encourages the robot to take steps to recover balance when pushed,
    but does NOT fire during normal walking (command > threshold).

    Args:
        env: The RL environment
        asset_cfg: Asset configuration (unused but kept for API consistency)
        command_name: Name of the velocity command in the command manager
        command_threshold: Speed below which the robot is considered to be in standing mode
        velocity_threshold: Linear velocity threshold to activate stepping reward (m/s)
        air_time_threshold: Minimum air time to count as a step (seconds)

    Returns:
        Reward tensor of shape (num_envs,)
    """
    asset: Entity = env.scene[asset_cfg.name]

    # Only fire for standing envs (command near zero)
    command = env.command_manager.get_command(command_name)
    total_speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    is_standing_cmd = (total_speed < command_threshold).float()

    # Get base linear velocity magnitude
    base_lin_vel = asset.data.root_link_vel_w[:, :3]  # (num_envs, 3)
    vel_magnitude = torch.norm(base_lin_vel[:, :2], dim=1)  # Only XY plane

    # Only reward stepping when velocity is high (being pushed)
    should_step = vel_magnitude > velocity_threshold

    # Get foot air time from contact sensor
    contact_sensor = env.scene.sensors["feet_ground_contact"]
    air_time = contact_sensor.data.last_air_time[:, :2]  # (num_envs, 2) - left and right foot

    # Reward if either foot has been in air recently
    foot_in_air = (air_time > air_time_threshold).any(dim=1)  # (num_envs,)

    # Only give reward when: standing command AND high body velocity AND foot stepped
    reward = is_standing_cmd * should_step.float() * foot_in_air.float()

    return reward


def adaptive_pose_weight(
    env: ManagerBasedRlEnv,
    base_pose_reward: torch.Tensor,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    velocity_threshold: float = 0.3,
    min_weight: float = 0.3,
) -> torch.Tensor:
    """Reduce pose tracking weight when robot has high velocity (recovering from push).

    This gives the robot freedom to deviate from the standing pose when taking
    recovery steps, while maintaining strict pose tracking when standing still.

    Args:
        env: The RL environment
        base_pose_reward: The original pose reward (before weighting)
        asset_cfg: Asset configuration (unused but kept for API consistency)
        velocity_threshold: Linear velocity threshold to start reducing weight (m/s)
        min_weight: Minimum weight multiplier (0-1) at high velocities

    Returns:
        Weighted reward tensor of shape (num_envs,)
    """
    asset: Entity = env.scene[asset_cfg.name]

    # Get base linear velocity magnitude
    base_lin_vel = asset.data.root_link_vel_w[:, :3]  # (num_envs, 3)
    vel_magnitude = torch.norm(base_lin_vel[:, :2], dim=1)  # Only XY plane

    # Compute weight: 1.0 when stationary, min_weight at high velocity
    # Use smooth transition via sigmoid-like function
    weight = min_weight + (1.0 - min_weight) * torch.exp(
        -((vel_magnitude - velocity_threshold) / velocity_threshold).clamp(min=0.0) ** 2
    )

    return base_pose_reward * weight


def randomize_base_orientation(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    max_pitch_deg: float = 10.0,
    max_roll_deg: float = 5.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
):
    """Randomize base orientation at episode start to force reactive behavior.

    Adds random pitch and roll to the robot's base orientation at the start of
    each episode. This prevents the policy from memorizing a single initial state
    and forces it to use feedback to adapt to different orientations.

    Args:
        env: The environment
        env_ids: Environment IDs to randomize
        max_pitch_deg: Maximum pitch angle in degrees (forward/backward tilt)
        max_roll_deg: Maximum roll angle in degrees (side-to-side tilt)
        asset_cfg: Asset configuration
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.int)

    asset: Entity = env.scene[asset_cfg.name]
    num_envs = len(env_ids)

    # Generate random pitch and roll angles
    max_pitch_rad = max_pitch_deg * torch.pi / 180.0
    max_roll_rad = max_roll_deg * torch.pi / 180.0

    pitch = (torch.rand(num_envs, device=env.device) * 2 - 1) * max_pitch_rad
    roll = (torch.rand(num_envs, device=env.device) * 2 - 1) * max_roll_rad
    yaw = torch.zeros(num_envs, device=env.device)  # Keep yaw at 0

    # Convert Euler angles (roll, pitch, yaw) to quaternion
    # Using the standard aerospace sequence (ZYX)
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)

    quat_w = cr * cp * cy + sr * sp * sy
    quat_x = sr * cp * cy - cr * sp * sy
    quat_y = cr * sp * cy + sr * cp * sy
    quat_z = cr * cp * sy - sr * sp * cy

    new_quat = torch.stack([quat_w, quat_x, quat_y, quat_z], dim=1)

    # Normalize quaternion
    new_quat = new_quat / torch.norm(new_quat, dim=1, keepdim=True)

    # Get root position index (freejoint starts at qpos index 0)
    # Freejoint: [x, y, z, qw, qx, qy, qz]
    root_quat_idx = 3  # Quaternion starts at index 3

    # Apply the randomized orientation to selected environments
    env.sim.data.qpos[env_ids, root_quat_idx:root_quat_idx+4] = new_quat


def set_face_down_orientation(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
):
    """Set the robot to a prone (belly-down) orientation for stand-up training.

    Rotates the robot 90° forward around the pitch axis (Y) so the front/belly
    faces the ground and legs point upward. Combined with a random yaw.

    Quaternion derivation:
        quat_pitch90 = [s, 0, s, 0]   where s = sqrt(2)/2  (90° around Y)
        quat_yaw     = [cy, 0, 0, sy]
        combined     = quat_yaw * quat_pitch90 = [s*cy, -s*sy, s*cy, s*sy]
    """
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device, dtype=torch.int)
    num = len(env_ids)

    yaw = torch.rand(num, device=env.device) * 2 * np.pi - np.pi
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    s = 2.0 ** -0.5  # sqrt(2)/2

    new_quat = torch.stack(
        [
            s * cy,   # w
            -s * sy,  # x
            s * cy,   # y
            s * sy,   # z
        ],
        dim=1,
    )

    # Freejoint qpos: [x, y, z, qw, qx, qy, qz, ...]
    env.sim.data.qpos[env_ids, 3:7] = new_quat
    env.sim.data.qvel[env_ids, :6] = 0.0


def set_random_prone_orientation(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    face_down_prob: float = 0.5,
):
    """Randomly initialize each env as face-down (belly) or face-up (back), with random yaw.

    Face-down:  +90° pitch → quat = [s*cy, -s*sy,  s*cy,  s*sy]
    Face-up:    -90° pitch → quat = [s*cy,  s*sy, -s*cy,  s*sy]

    Args:
        face_down_prob: probability of sampling face-down (vs face-up). A curriculum
            can ramp this from a high initial value (easier task) toward 0.5.
    """
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device, dtype=torch.int)
    num = len(env_ids)

    yaw = torch.rand(num, device=env.device) * 2 * np.pi - np.pi
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    s = 2.0 ** -0.5  # sqrt(2)/2

    face_down = torch.stack([ s * cy, -s * sy,  s * cy,  s * sy], dim=1)
    face_up   = torch.stack([ s * cy,  s * sy, -s * cy,  s * sy], dim=1)

    mask = torch.rand(num, device=env.device) < face_down_prob  # True → face-down
    new_quat = torch.where(mask.unsqueeze(1), face_down, face_up)

    env.sim.data.qpos[env_ids, 3:7] = new_quat
    env.sim.data.qvel[env_ids, :6] = 0.0


def set_random_ground_state(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    face_down_prob: float = 0.4,
    face_up_prob: float = 0.4,
    sitting_prob: float = 0.2,
    standing_prob: float = 0.0,
    prone_z_min: float = 0.20,
    prone_z_max: float = 0.25,
    sitting_z_min: float = 0.07,
    sitting_z_max: float = 0.09,
    standing_z_min: float = 0.11,
    standing_z_max: float = 0.12,
    sitting_joint_overrides: Optional[dict] = None,
    sitting_joint_noise_std: float = 0.0,
    sitting_tilt_max: float = 0.0,
    face_up_roll_max: float = 0.0,
):
    """Reset to a random ground state: face-down, face-up, sitting, or standing.

    Broader than ``set_random_prone_orientation`` — used by the stand-up env so
    the policy learns to recover from any plausible pose, including the sitting
    keyframe (rest state of the sit policy) and an already-standing pose (so it
    also learns to *hold* a stand, not only to rise).

    Modes (probabilities are normalized; they need not sum to 1.0):
      - face-down (belly to floor): +90° pitch, random yaw, z in [prone_z_min, prone_z_max].
      - face-up   (back to floor):  -90° pitch, random yaw, z in [prone_z_min, prone_z_max].
      - sitting:                    upright (±sitting_tilt_max), random yaw, z low,
                                    joints set to ``sitting_joint_overrides``.
      - standing:                   upright (±sitting_tilt_max), random yaw, z in
                                    [standing_z_min, standing_z_max], joints left at
                                    HOME (whatever ``reset_robot_joints`` set).

    Args:
        sitting_joint_overrides: ``{qpos_joint_index: angle_rad}`` to write into
            ``qpos[7+idx]`` for envs sampled into the sitting bucket. ``None``
            keeps joints at whatever ``reset_robot_joints`` already set.
    """
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device, dtype=torch.int)
    num = len(env_ids)

    total = face_down_prob + face_up_prob + sitting_prob + standing_prob
    p_fd  = face_down_prob / total
    p_fu  = (face_down_prob + face_up_prob) / total
    p_sit = (face_down_prob + face_up_prob + sitting_prob) / total

    yaw = torch.rand(num, device=env.device) * 2 * np.pi - np.pi
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    s = 2.0 ** -0.5  # sqrt(2)/2

    face_down = torch.stack([ s * cy, -s * sy,  s * cy,  s * sy], dim=1)
    face_up   = torch.stack([ s * cy,  s * sy, -s * cy,  s * sy], dim=1)
    # Upright sitting: yaw-only by default, with optional ±sitting_tilt_max
    # pitch/roll noise so the policy doesn't overfit to perfectly-upright starts.
    if sitting_tilt_max > 0.0:
        pitch = (torch.rand(num, device=env.device) * 2 - 1) * sitting_tilt_max
        roll  = (torch.rand(num, device=env.device) * 2 - 1) * sitting_tilt_max
        cp = torch.cos(pitch * 0.5); sp = torch.sin(pitch * 0.5)
        cr = torch.cos(roll  * 0.5); sr = torch.sin(roll  * 0.5)
        # ZYX intrinsic Euler → quaternion (yaw * pitch * roll).
        sit_w = cr * cp * cy + sr * sp * sy
        sit_x = sr * cp * cy - cr * sp * sy
        sit_y = cr * sp * cy + sr * cp * sy
        sit_z = cr * cp * sy - sr * sp * cy
        sitting = torch.stack([sit_w, sit_x, sit_y, sit_z], dim=1)
    else:
        sitting = torch.stack([cy, torch.zeros_like(cy), torch.zeros_like(cy), sy], dim=1)

    u = torch.rand(num, device=env.device)
    is_fd    = u < p_fd
    is_fu    = (u >= p_fd) & (u < p_fu)
    is_sit   = (u >= p_fu) & (u < p_sit)
    is_stand = u >= p_sit

    # Face-up partial-roll noise: rotate the supine pose about the body's long
    # axis by uniform ±face_up_roll_max. WHY (2026-07, back-recovery was
    # seed-lucky): the reward landscape between supine and prone is FLAT —
    # upright_linear (cos tilt) is ≈0 through the whole roll, height doesn't
    # change — so rolling off the back only pays via the front-rise path that
    # follows, a long-horizon dependency that noisy exploration rarely finds
    # from a perfectly flat supine start. With roll noise, a fraction of
    # face-up spawns start near-on-side (partway along the roll): the policy
    # learns roll-completion from easy starts and generalizes back to flat
    # supine — a built-in reverse curriculum. Uniform sampling keeps every
    # difficulty represented (flat back |roll|<15° ≈ 17% at ±90°), so no
    # annealing schedule is needed, and varied post-fall poses are realistic
    # DR for deployment anyway.
    if face_up_roll_max > 0.0:
        theta = (torch.rand(num, device=env.device) * 2 - 1) * face_up_roll_max
        ct = torch.cos(theta * 0.5)
        st = torch.sin(theta * 0.5)
        # Log-roll = rotation about the body's LONG axis, which is body z (the
        # spine: trunk z is up when standing → horizontal when lying). NOT body
        # x — supine leaves body x pointing skyward, so an x-roll would only
        # spin the robot in place like the yaw noise already does.
        # Body-frame rotation → right-multiply: q_fu ⊗ [ct, 0, 0, st].
        w, x, y, z = face_up[:, 0], face_up[:, 1], face_up[:, 2], face_up[:, 3]
        face_up = torch.stack(
            [
                w * ct - z * st,
                x * ct + y * st,
                y * ct - x * st,
                w * st + z * ct,
            ],
            dim=1,
        )

    # Sitting and standing share the same upright orientation (identity + optional
    # ±sitting_tilt_max); they differ only in trunk height and joint pose.
    new_quat = face_down.clone()
    new_quat[is_fu]    = face_up[is_fu]
    new_quat[is_sit]   = sitting[is_sit]
    new_quat[is_stand] = sitting[is_stand]

    # Random z per env: prone heights for face-down/up, low for sit, ~standing for stand.
    z_prone = torch.rand(num, device=env.device) * (prone_z_max - prone_z_min) + prone_z_min
    z_sit   = torch.rand(num, device=env.device) * (sitting_z_max - sitting_z_min) + sitting_z_min
    z_stand = torch.rand(num, device=env.device) * (standing_z_max - standing_z_min) + standing_z_min
    new_z = z_prone.clone()
    new_z = torch.where(is_sit, z_sit, new_z)
    new_z = torch.where(is_stand, z_stand, new_z)

    env.sim.data.qpos[env_ids, 2]   = new_z
    env.sim.data.qpos[env_ids, 3:7] = new_quat
    env.sim.data.qvel[env_ids, :6]  = 0.0

    # Sitting-bucket joint overrides (e.g. knee/ankle bent to keyframe).
    # Override keys are SERVO indices (14-joint layout); translate to entity
    # joint indices so models with interleaved passive_* joints (backlash)
    # write the intended joints. qpos column = 7 + entity joint index
    # (robot free joint first, all hinges 1-dof).
    asset: Entity = env.scene[asset_cfg.name]
    servo_ids = _servo_joint_ids(env, asset)
    if sitting_joint_overrides:
        sit_env_ids = env_ids[is_sit]
        if len(sit_env_ids) > 0:
            for jnt_idx, angle in sitting_joint_overrides.items():
                env.sim.data.qpos[sit_env_ids, 7 + servo_ids[jnt_idx]] = angle

    # Joint noise for sitting envs: Gaussian noise on every actuated joint
    # so the policy sees a distribution of plausible "sit" starts rather than
    # a single canonical pose. Captures real-world transfer where the robot's
    # joint angles won't match the SIT keyframe exactly when the standup
    # policy takes over from the sit policy.
    if sitting_joint_noise_std > 0.0:
        sit_env_ids = env_ids[is_sit]
        if len(sit_env_ids) > 0:
            # Servo joints only: passive_* joints (backlash hinges) have tiny
            # ranges and must stay at 0 on reset.
            n_sit = len(sit_env_ids)
            cols = torch.tensor([7 + j for j in servo_ids], device=env.device, dtype=torch.long)
            noise = torch.randn(n_sit, len(cols), device=env.device) * sitting_joint_noise_std
            env.sim.data.qpos[sit_env_ids.unsqueeze(1).long(), cols.unsqueeze(0)] += noise


# Deep-crouch anchor pose (velstand run-5): the "stuck" mid-recovery basin —
# knees folded under the body, trunk pitched forward, feet flat. Values chosen
# by extending the HOME zig-zag (hip fwd / knee back / ankle fwd, sign
# conventions per the SIT keyframe fold directions) to deep flexion, inside
# the ±1.57 joint limits. hip_yaw/hip_roll/neck stay at HOME.
_CROUCH_ANCHOR_BY_NAME = {
    "left_hip_pitch": -1.15,
    "left_knee": 1.25,
    "left_ankle": 1.05,
    "right_hip_pitch": 1.15,
    "right_knee": -1.25,
    "right_ankle": -1.05,
}


def set_random_crouch_state(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    depth_min: float = 0.35,
    depth_max: float = 1.0,
    pitch_max_deg: float = 55.0,
    joint_noise: float = 0.12,
    z_stand: float = 0.115,
    z_deep: float = 0.06,
):
    """Reset selected envs into a random mid-recovery crouch.

    Reverse curriculum for the recovery last mile (velstand run-5 lesson):
    prone-init episodes spend most of their fallen budget getting TO the deep
    crouch and are recycled shortly after reaching it, so the crouch→stand
    mile gets almost no on-policy data — the policy converged to parking
    there. Seeding resets ACROSS that mile (depth λ ∈ [depth_min, depth_max]
    between standing and the deep-crouch anchor, trunk pitch and z scaled
    with λ) makes the frontier dense from step 0 of the episode.
    """
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device, dtype=torch.long)
    num = len(env_ids)
    asset: Entity = env.scene[asset_cfg.name]

    lam = torch.rand(num, device=env.device) * (depth_max - depth_min) + depth_min

    # Joints: lerp HOME → anchor on the leg pitch chain, uniform noise on the
    # servo joints only (passive_* backlash hinges have ±1° ranges — noise
    # there would spawn them pinned outside their limits).
    joints = asset.data.default_joint_pos[env_ids].clone()
    for name, anchor in _CROUCH_ANCHOR_BY_NAME.items():
        ids, _ = asset.find_joints(f"^{name}$")
        j = ids[0]
        joints[:, j] = joints[:, j] + lam * (anchor - joints[:, j])
    noise_mask = torch.zeros(joints.shape[1], device=joints.device)
    noise_mask[_servo_joint_ids(env, asset)] = 1.0
    joints += (torch.rand_like(joints) * 2 - 1) * joint_noise * noise_mask

    # Base orientation: forward pitch scaled with depth (the stuck basin is a
    # forward crouch from both fall directions), random yaw, small roll noise.
    pitch = lam * math.radians(pitch_max_deg) \
        + (torch.rand(num, device=env.device) * 2 - 1) * math.radians(10.0)
    pitch = torch.clamp(pitch, min=math.radians(5.0))
    roll = (torch.rand(num, device=env.device) * 2 - 1) * math.radians(8.0)
    yaw = torch.rand(num, device=env.device) * 2 * np.pi - np.pi
    cy = torch.cos(yaw * 0.5); sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5); sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5); sr = torch.sin(roll * 0.5)
    # ZYX intrinsic Euler → quaternion (yaw * pitch * roll), as in
    # set_random_ground_state's sitting branch.
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    quat = torch.stack([qw, qx, qy, qz], dim=1)

    # Trunk height scaled with depth, small upward margin to settle cleanly.
    z = z_stand + lam * (z_deep - z_stand) \
        + torch.rand(num, device=env.device) * 0.01

    env.sim.data.qpos[env_ids, 2] = z
    env.sim.data.qpos[env_ids, 3:7] = quat
    env.sim.data.qpos[env_ids, 7:] = joints
    env.sim.data.qvel[env_ids, :] = 0.0


def maybe_set_random_prone_orientation(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    prone_prob: float = 0.0,
    face_down_prob: float = 0.5,
    prone_z_min: float = 0.20,
    prone_z_max: float = 0.25,
    crouch_prob: float = 0.0,
):
    """Reset event that overrides orientation to prone with probability `prone_prob`.

    With prob `prone_prob`, replaces the upright orientation (already set by
    reset_base) with a prone orientation; otherwise leaves it upright. Among the
    overridden envs, `face_down_prob` picks face-down (belly) vs face-up (back).

    Also lifts z to [prone_z_min, prone_z_max] for the overridden envs so the
    head/neck clearance is sufficient — the vel-env reset z (~0.125) would
    clip the head through the ground at 90° pitch.

    At prone_prob=2/3 and face_down_prob=0.5 you get a balanced 33/33/33 split
    of upright/face-down/face-up resets, which is the standard mixture for
    learning fall recovery alongside normal upright start.

    With ``crouch_prob`` > 0, an additional exclusive slice of envs is reset
    into a random mid-recovery crouch via ``set_random_crouch_state`` (reverse
    curriculum for the recovery last mile — see its docstring).
    """
    if prone_prob <= 0.0 and crouch_prob <= 0.0:
        return
    # env_ids=None means "all envs" (the initial global reset passes None —
    # the old early-return silently skipped prone init there).
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    if len(env_ids) == 0:
        return
    env_ids_t = env_ids.to(env.device, dtype=torch.long) if isinstance(env_ids, torch.Tensor) else torch.tensor(env_ids, device=env.device, dtype=torch.long)
    # One draw partitions envs into exclusive prone / crouch / untouched slices.
    u = torch.rand(len(env_ids_t), device=env.device)
    selected = env_ids_t[u < prone_prob]
    crouch_selected = env_ids_t[(u >= prone_prob) & (u < prone_prob + crouch_prob)]
    if len(selected) > 0:
        set_random_prone_orientation(
            env, selected, asset_cfg=asset_cfg, face_down_prob=face_down_prob
        )
        # Override z so the prone body has head/neck clearance when settling.
        z = torch.rand(len(selected), device=env.device) * (prone_z_max - prone_z_min) + prone_z_min
        env.sim.data.qpos[selected, 2] = z
    if len(crouch_selected) > 0:
        set_random_crouch_state(env, crouch_selected, asset_cfg=asset_cfg)


def event_param_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    event_name: str,
    param_stages: list[dict],
) -> torch.Tensor:
    """Mutate an event term's params at scheduled steps.

    Mirror of termination_param_curriculum but for events. Uses the live
    EventManager term cfg via get_term_cfg, since env.cfg.events is a deepcopy.
    param_stages: list of {step: int, params: dict}. Shallow-merged into the
    live event term's params at the latest matching stage.
    """
    del env_ids
    event_cfg = env.event_manager.get_term_cfg(event_name)
    current = param_stages[0]["params"]
    for stage in param_stages:
        if env.common_step_counter >= stage["step"]:
            current = stage["params"]
    event_cfg.params.update(current)
    first_val = next(iter(current.values()))
    return torch.tensor(float(first_val) if isinstance(first_val, (int, float)) else 0.0)


def face_down_prob_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    event_name: str,
    prob_stages: list[dict],
) -> torch.Tensor:
    """Ramp face_down_prob on a reset event over training.

    Args:
        event_name: name of the event term using set_random_prone_orientation
        prob_stages: list of {step: int, prob: float}. Higher prob = more
            face-down resets (easier task); ramp toward 0.5 as training proceeds.
    """
    del env_ids

    # NOTE: must update the live EventManager term_cfg, not env.cfg.events —
    # EventManager.__init__ does deepcopy(cfg), so mutating env.cfg.events is a no-op.
    event_cfg = env.event_manager.get_term_cfg(event_name)

    current_prob = prob_stages[0]["prob"]
    for stage in prob_stages:
        if env.common_step_counter > stage["step"]:
            current_prob = stage["prob"]

    event_cfg.params["face_down_prob"] = current_prob
    return torch.tensor([current_prob])


class VelocityCommandCommandOnly(UniformVelocityCommand):
    """Like UniformVelocityCommand but only draws the command arrows (no actual velocity arrows)."""

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        # Turn-in-place practice: for a fraction of envs, zero the linear velocity
        # and force a meaningful (away-from-zero) yaw command. Independent uniform
        # sampling almost never produces "lin≈0, |ang| large" (~2% of samples), so
        # spinning on the spot was effectively untrained → slow/unstable real-robot
        # turning. Mirrors the base rel_forward_envs mechanism.
        p = getattr(self.cfg, "rel_turn_in_place_envs", 0.0)
        if p <= 0.0:
            return
        r = torch.empty(len(env_ids), device=self.device)
        turn_ids = env_ids[r.uniform_(0.0, 1.0) < p]
        if len(turn_ids) == 0:
            return
        self.vel_command_b[turn_ids, 0] = 0.0
        self.vel_command_b[turn_ids, 1] = 0.0
        lo, hi = self.cfg.ranges.ang_vel_z
        maxr = max(abs(lo), abs(hi))
        rr = torch.empty(len(turn_ids), device=self.device)
        sign = torch.where(rr.uniform_(0.0, 1.0) < 0.5, -1.0, 1.0)
        mag = torch.empty(len(turn_ids), device=self.device).uniform_(0.4 * maxr, maxr)
        self.vel_command_b[turn_ids, 2] = sign * mag
        # These envs must actually turn — un-mark them as standing (which would
        # zero the command) and refresh the world-frame reference copy.
        self.is_standing_env[turn_ids] = False
        self.vel_command_w[turn_ids] = self.vel_command_b[turn_ids]

    def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
        batch = visualizer.env_idx
        if batch >= self.num_envs:
            return

        cmds = self.command.cpu().numpy()
        base_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()
        base_quat_w = self.robot.data.root_link_quat_w
        base_mat_ws = matrix_from_quat(base_quat_w).cpu().numpy()

        base_pos_w = base_pos_ws[batch]
        base_mat_w = base_mat_ws[batch]
        cmd = cmds[batch]

        if np.linalg.norm(base_pos_w) < 1e-6:
            return

        def local_to_world(vec: np.ndarray) -> np.ndarray:
            return base_pos_w + base_mat_w @ vec

        scale = self.cfg.viz.scale * 2.0
        z_offset = self.cfg.viz.z_offset

        # Command linear velocity arrow (blue).
        cmd_lin_from = local_to_world(np.array([0, 0, z_offset]) * scale)
        cmd_lin_to = local_to_world(
            (np.array([0, 0, z_offset]) + np.array([cmd[0], cmd[1], 0])) * scale
        )
        visualizer.add_arrow(cmd_lin_from, cmd_lin_to, color=(0.2, 0.2, 0.6, 0.6), width=0.015)


@_dataclass(kw_only=True)
class VelocityCommandCommandOnlyCfg(UniformVelocityCommandCfg):
    # Fraction of envs commanded to turn in place (lin=0, |ang| forced to
    # [0.4·max, max]) each resample. 0 = disabled (base uniform sampling only).
    rel_turn_in_place_envs: float = 0.0

    def build(self, env: ManagerBasedRlEnv) -> "VelocityCommandCommandOnly":
        return VelocityCommandCommandOnly(self, env)


class RelativeHeadingVelocityCommand(VelocityCommandCommandOnly):
    """Velocity command where cmd[2] is the heading error in the robot's body frame.

    cmd[0] = lin_vel_x  (throttle: 0=coast, +push, -brake)
    cmd[1] = lin_vel_y  (unused, 0)
    cmd[2] = heading_error  (+ = target is to the right/CW, - = to the left/CCW)
             0 → go straight, ±max = target is max_angle rad to the right/left

    During training: a random world-frame heading is sampled at each episode reset.
    At every step, cmd[2] = clamp(wrap(current_yaw - target_yaw), ±max_angle).
    Positive when the robot is pointing CCW (left) of the target → needs to turn right.

    At inference: the user feeds cmd[2] directly.  Holding cmd[2] = constant gives
    a proportional heading correction = approximately constant turn rate.

    Set heading_command=False and rel_heading_envs=0.0 in the cfg (we handle
    heading internally).  ang_vel_z range in cfg is used as the clip limit for cmd[2].
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        # Sampled target heading per env, world frame (rad)
        self._target_heading_w = torch.zeros(self.num_envs, device=self.device)
        # Clip limit for cmd[2]: use ang_vel_z[1] from cfg (the positive bound)
        ang_rng = cfg.ranges.ang_vel_z
        self._heading_max = float(ang_rng[1]) if ang_rng else 1.0

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        n = len(env_ids)
        # Sample random world-frame target heading uniformly in [-π, π]
        self._target_heading_w[env_ids] = (
            torch.rand(n, device=self.device) * 2.0 * math.pi - math.pi
        )
        # Zero ang_vel slot; _update_command will fill it each step
        self.vel_command_b[env_ids, 2] = 0.0

    def _update_command(self) -> None:
        # Do NOT call super()._update_command() — it would run the heading
        # proportional controller and overwrite cmd[2] with a yaw rate.
        # Instead recompute heading error from scratch each step.
        quat = self.robot.data.root_link_quat_w  # (N, 4) [w, x, y, z]
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        current_yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        # Positive = target is CCW (left) of robot → turn left. Standard convention.
        delta = self._target_heading_w - current_yaw
        heading_error = torch.atan2(torch.sin(delta), torch.cos(delta))
        self.vel_command_b[:, 2] = heading_error.clamp(-self._heading_max, self._heading_max)

    def _update_metrics(self) -> None:
        pass  # No velocity tracking metrics for heading command


class RelativeHeadingVelocityCommandCfg(UniformVelocityCommandCfg):
    def build(self, env: ManagerBasedRlEnv) -> "RelativeHeadingVelocityCommand":
        return RelativeHeadingVelocityCommand(self, env)


def heading_tracking_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float = 0.5,
) -> torch.Tensor:
    """Reward for reducing heading error when cmd[2] encodes heading error.

    Returns exp(-cmd[2]² / std²).
    - At error = 0 (on heading): reward = 1.0.
    - At error = std: reward ≈ 0.37 (strong gradient).
    - At error = 1.0 rad with std=0.5: reward ≈ 0.018 (nearly zero).

    std=0.5 rad (≈28°) gives a meaningful gradient across the expected range.
    """
    cmd = env.command_manager.get_command(command_name)
    heading_error = cmd[:, 2]
    return torch.exp(-(heading_error ** 2) / (std ** 2))


def skating_air_time_reward(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    threshold_min: float = 0.05,
    threshold_max: float = 0.4,
    vel_gate_ref: float = 0.0,
) -> torch.Tensor:
    """Reward feet air time only when pushing (cmd_x > 0).

    Encourages the robot to lift each foot during the recovery phase of the
    skating stroke rather than dragging it on the ground.
    Scaled by cmd_x so the incentive grows with push intensity.

    When ``vel_gate_ref`` > 0 the reward is also multiplied by a forward-speed
    gate so lifting feet without propelling the body (tap-dancing on the spot)
    earns nothing. ``threshold_min`` sets the shortest swing that counts — raise
    it to forbid a frantic high-cadence flutter.
    """
    from mjlab.sensor import ContactSensor
    sensor: ContactSensor = env.scene[sensor_name]
    current_air_time = sensor.data.current_air_time
    assert current_air_time is not None

    in_range = (current_air_time > threshold_min) & (current_air_time < threshold_max)
    reward = torch.sum(in_range.float(), dim=1)

    cmd_x = env.command_manager.get_command(command_name)[:, 0]
    reward = reward * torch.clamp(cmd_x, min=0.0)
    gate = _forward_progress_gate(env, vel_gate_ref)
    if gate is not None:
        reward = reward * gate
    return reward


def _forward_progress_gate(env: ManagerBasedRlEnv, v_ref: float) -> torch.Tensor | None:
    """0→1 ramp in body forward speed: 0 when standing still, 1 at/above v_ref.

    Used to gate stride-shaping rewards so that stepping which does NOT propel
    the body (e.g. tap-dancing on the spot) earns nothing — the reward for the
    FORM of a stride is only paid when the stride actually does its JOB (moving
    forward). Returns None when disabled (v_ref <= 0)."""
    if v_ref <= 0.0:
        return None
    v_fwd = env.scene["robot"].data.root_link_lin_vel_b[:, 0]
    return (v_fwd.clamp(min=0.0) / v_ref).clamp(max=1.0)


def single_support_reward(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    vel_gate_ref: float = 0.0,
    double_penalty: float = 0.25,
) -> torch.Tensor:
    """Reward single-support (a skating stride), mildly discourage the swizzle.

    Real skating is a STRIDE: push off one blade while the other swings, i.e.
    single support that alternates left/right. A symmetric swizzle keeps BOTH
    blades grounded the whole time and still spins the wheels, so wheel_speed
    alone converges to it.

    Per step, counting blades in contact:
      - exactly 1 blade down (stride)  → + clamp(cmd_x,0) · gate
      - 2 blades down    (double supp) → − double_penalty · clamp(cmd_x,0)
      - 0 blades down    (flight/hop)  →  0

    The POSITIVE single-support reward is gated by forward speed (``vel_gate_ref``)
    so stepping in place (no propulsion) earns nothing — kills the tap-dance hack.
    The double-support penalty is small and UNGATED: brief double support during
    weight transfer / push-off is NORMAL skating, so we only lightly discourage
    PERMANENT double support (the swizzle) rather than forbid it. The real
    anti-swizzle signal is skating_air_time — the swizzle never lifts a foot.
    """
    from mjlab.sensor import ContactSensor
    sensor: ContactSensor = env.scene[sensor_name]
    contact_time = sensor.data.current_contact_time  # (num_envs, num_feet)
    assert contact_time is not None

    n_contact = torch.sum((contact_time > 0.0).float(), dim=1)  # (num_envs,)
    single = (n_contact == 1).float()
    double = (n_contact >= 2).float()

    cmd_x = torch.clamp(env.command_manager.get_command(command_name)[:, 0], min=0.0)
    single_r = single * cmd_x
    gate = _forward_progress_gate(env, vel_gate_ref)
    if gate is not None:
        single_r = single_r * gate
    return single_r - double_penalty * double * cmd_x


def glide_reward(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    vel_ref: float = 0.2,
    stillness_std: float = 5.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", joint_names=(r".*(hip|knee|ankle).*",)
    ),
) -> torch.Tensor:
    """Reward the GLIDE phase of a stride: coast on ONE blade with quiet legs.

    Nothing else rewards gliding — skating_air_time pays each swing, so the policy
    maximises swing FREQUENCY (frantic kicking). This term pays staying on one
    foot and coasting, giving the policy a reason to slow down and commit to each
    stroke:

        reward = single_support · forward_gate · stillness · (cmd_x >= 0)

    - single_support: exactly ONE blade in contact. REQUIRED — this is the fix vs
      the earlier broken glide, which omitted it and let a two-blade swizzle-coast
      farm the reward and regress the gait.
    - forward_gate = clamp(v_fwd,0,vel_ref)/vel_ref → 0 when not moving forward.
    - stillness = exp(-Σ leg_joint_vel² / stillness_std²) → high only when legs
      are quiet; a kick (fast joint motion) gets ~0, so only a real glide pays.
    - active on push/coast only (cmd_x >= 0); silent on brake.
    """
    from mjlab.sensor import ContactSensor
    sensor: ContactSensor = env.scene[sensor_name]
    contact_time = sensor.data.current_contact_time  # (num_envs, num_feet)
    assert contact_time is not None
    single = (torch.sum((contact_time > 0.0).float(), dim=1) == 1).float()

    forward_gate = _forward_progress_gate(env, vel_ref)
    if forward_gate is None:
        forward_gate = torch.ones(env.num_envs, device=env.device)

    asset: Entity = env.scene[asset_cfg.name]
    joint_vel_sq = torch.sum(
        torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1
    )
    stillness = torch.exp(-joint_vel_sq / stillness_std ** 2)

    cmd_x = env.command_manager.get_command(command_name)[:, 0]
    active = (cmd_x >= 0.0).float()
    return single * forward_gate * stillness * active


def leg_symmetry_reward(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    joint_bases: tuple = ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle"),
) -> torch.Tensor:
    """Reward left/right legs mirroring — the swizzle's defining symmetry.

    The robot uses mirrored L/R sign conventions, so a bilaterally-symmetric config
    satisfies q_left + q_right ≈ 0 per matched joint pair. Returns
    ``-mean_pairs |q_left + q_right|`` (L1, constant gradient); use with a POSITIVE
    weight so asymmetry is penalised and the symmetric swizzle is favoured. L/R index
    pairs are resolved once by name and cached on env.
    """
    asset: Entity = env.scene[asset_cfg.name]
    if not hasattr(env, "_leg_sym_ids"):
        left, right = [], []
        for base in joint_bases:
            li, _ = asset.find_joints([f"left_{base}"])
            ri, _ = asset.find_joints([f"right_{base}"])
            left.append(li[0])
            right.append(ri[0])
        env._leg_sym_ids = (
            torch.tensor(left, device=env.device),
            torch.tensor(right, device=env.device),
        )
    lids, rids = env._leg_sym_ids
    q = asset.data.joint_pos
    return -torch.abs(q[:, lids] + q[:, rids]).mean(dim=-1)


def grounded_reward(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
) -> torch.Tensor:
    """Reward BOTH blades in contact — a classic swizzle stays grounded (no lifting).

    Mirror of single_support_reward but rewarding double support (n_contact >= 2),
    scaled by |cmd_x| so it shapes the push phase in EITHER direction (forward or
    backward — the swizzle env drives cmd_x < 0 as "go backward").
    """
    from mjlab.sensor import ContactSensor
    sensor: ContactSensor = env.scene[sensor_name]
    contact_time = sensor.data.current_contact_time  # (num_envs, num_feet)
    assert contact_time is not None
    n_contact = torch.sum((contact_time > 0.0).float(), dim=1)
    grounded = (n_contact >= 2).float()
    cmd_x = torch.abs(env.command_manager.get_command(command_name)[:, 0])
    return grounded * cmd_x


def gait_symmetry_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str,
) -> torch.Tensor:
    """Penalize lopsided left/right foot usage (one blade doing most of the work).

    With symmetry augmentation OFF, nothing stops the policy learning an asymmetric
    stride that pushes mostly with one leg — which veers and destabilises (esp. at
    launch). Accumulates per-foot swing time over the episode and penalises the
    normalised imbalance |L - R| / (L + R):
      - balanced alternating stride  -> ~0 (no penalty)
      - one foot swinging much more   -> ~1 (max penalty)
    Only the CUMULATIVE imbalance is penalised — the instantaneous single-support
    asymmetry of a real stride (one foot swinging now) is fine.
    """
    from mjlab.sensor import ContactSensor
    sensor: ContactSensor = env.scene[sensor_name]
    air = sensor.data.current_air_time  # (N, num_feet)
    assert air is not None

    if not hasattr(env, "_swing_accum") or env._swing_accum.shape[0] != env.num_envs:
        env._swing_accum = torch.zeros(env.num_envs, air.shape[1], device=env.device)
    reset = env.episode_length_buf <= 1
    env._swing_accum[reset] = 0.0
    env._swing_accum += (air > 0.0).float() * env.step_dt

    L = env._swing_accum[:, 0]
    R = env._swing_accum[:, 1]
    return torch.abs(L - R) / (L + R + 1e-3)


def heading_hold_reward(
    env: ManagerBasedRlEnv,
    std: float = 0.4,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward holding the SPAWN heading (go straight) — corrective, angle-based.

    Rewards the yaw ANGLE staying near the heading captured at reset:
        reward = exp(-wrap(yaw - yaw_spawn)² / std²)

    This is the RIGHT way to go straight (vs penalising yaw-RATE, which just tells
    the policy 'never turn' → it can't steer back and drifts open-loop). Here a
    drift lowers the reward, and the policy is free to yaw back to recover it.

    The spawn heading is captured per-env on the first step(s) after reset
    (episode_length_buf <= 1), when the robot is still ~at its spawn pose. Reads
    root_link_quat_w, which is fresh at reward time (post physics step). Heading-
    invariant: the reference is each env's own random spawn yaw, so it works with
    the full-circle yaw randomisation at reset.
    """
    asset: Entity = env.scene[asset_cfg.name]
    quat = asset.data.root_link_quat_w  # (N, 4) [w, x, y, z]
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    if not hasattr(env, "_heading_ref") or env._heading_ref.shape[0] != env.num_envs:
        env._heading_ref = yaw.clone()
    just_reset = env.episode_length_buf <= 1
    env._heading_ref = torch.where(just_reset, yaw, env._heading_ref)

    err = yaw - env._heading_ref
    err = torch.atan2(torch.sin(err), torch.cos(err))  # wrap to [-π, π]
    return torch.exp(-(err ** 2) / std ** 2)


def action_over_limit_penalty(
    env: ManagerBasedRlEnv,
    action_name: str = "joint_pos",
    overshoot: float = 0.3,
) -> torch.Tensor:
    """Penalise commanding a joint target beyond its hard limit (+ overshoot).

    Policy-side deterrent against over-driving a joint onto its mechanical stop:
    e.g. hip_roll has a ±0.38 rad limit but a ±10 rad ctrlrange, so the low-kp
    servo can be commanded far past the stop to slam it with max torque — a
    fragile sim-only trick that will not transfer.

    Reads the commanded target (raw_action · scale + offset) and penalises only
    the part BEYOND (hard_limit + overshoot):

        penalty = Σ relu(target - (hi + overshoot)) + relu((lo - overshoot) - target)

    Unlike a qpos-limit penalty, this fires on the COMMAND, not the joint
    position — so the joint may still reach its full range (command ≈ limit) and
    no usable amplitude is stolen. Because it constrains the policy's OUTPUT, the
    learned behaviour is baked into the network and transfers to deployment
    WITHOUT any env-side action clip (which would only exist in sim → mismatch).
    ``overshoot`` gives the low-kp servo the headroom to reach near-limit targets
    under load; only the wild over-drive past that is penalised.
    """
    term = env.action_manager.get_term(action_name)
    target = term.raw_action * term.scale + term.offset  # (B, action_dim) abs targets
    jnt_ids = term.target_ids
    hard = env.scene["robot"].data.joint_pos_limits[:, jnt_ids]  # (B, action_dim, 2)
    lo = hard[..., 0] - overshoot
    hi = hard[..., 1] + overshoot
    over = (target - hi).clip(min=0.0) + (lo - target).clip(min=0.0)
    return torch.sum(over, dim=-1)


def forward_lean_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    target_pitch: float = 0.08,
    std: float = 0.08,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=("trunk_base",)),
) -> torch.Tensor:
    """Reward leaning slightly forward when pushing, to counteract the backward
    torque from skating strokes.

    Uses projected_gravity_b x-component as a pitch proxy:
      forward_lean = -gravity_b[:, 0]  (positive when leaning forward)

    Only fires when cmd_x > 0. Peaks at target_pitch radians of forward lean.
    """
    asset: Entity = env.scene[asset_cfg.name]
    cmd_x = env.command_manager.get_command(command_name)[:, 0]
    forward_lean = asset.data.projected_gravity_b[:, 0]
    push = torch.clamp(cmd_x, min=0.0)
    return push * torch.exp(-((forward_lean - target_pitch) ** 2) / (std ** 2))


class GroundPickPhaseCommand(UniformVelocityCommand):
    """Phase-encoding command for the ground pick / sit-stand tasks.

    Replaces the velocity command with a cyclic phase signal:
        command = [cos(2π*phase), sin(2π*phase), 0]

    Phase ∈ [0, 0.5]: approach (go down).
    Phase ∈ [0.5, 1.0]: return (come back up).

    Phase is randomized per environment on episode reset to decorrelate envs.
    Period defaults to 4s; override via the cfg.period field (sitstand uses 8s
    for a slower, gentler sit-down).
    """

    PERIOD: float = 4.0  # default; cfg.period overrides

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._gp_phase = torch.zeros(self.num_envs, device=self.device)
        self._period = float(getattr(cfg, "period", self.PERIOD))
        # When False, each episode starts at phase 0 (standing) instead of a
        # random phase. Matches the runtime, where the button starts the cycle
        # at phase 0 from standing. Default True keeps the historical ground_pick
        # behavior (random phase to decorrelate envs).
        self._randomize_phase = bool(getattr(cfg, "randomize_phase", True))

    @property
    def command(self) -> torch.Tensor:
        return self.vel_command_b

    def compute(self, dt: float) -> None:
        self._gp_phase = (self._gp_phase + dt / self._period) % 1.0
        self.vel_command_b[:, 0] = torch.cos(2 * torch.pi * self._gp_phase)
        self.vel_command_b[:, 1] = torch.sin(2 * torch.pi * self._gp_phase)
        self.vel_command_b[:, 2] = 0.0

    def reset(self, env_ids: torch.Tensor | None) -> dict:
        if env_ids is not None and len(env_ids) > 0:
            if self._randomize_phase:
                self._gp_phase[env_ids] = torch.rand(len(env_ids), device=self.device)
            else:
                self._gp_phase[env_ids] = 0.0
        return {}

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        pass  # Phase is continuous; no resampling needed

    def _update_command(self) -> None:
        pass  # Updated in compute()

    def _update_metrics(self) -> None:
        pass  # No velocity tracking metrics for ground pick


from dataclasses import dataclass as _dataclass

@_dataclass(kw_only=True)
class GroundPickPhaseCommandCfg(UniformVelocityCommandCfg):
    class_type: type = GroundPickPhaseCommand
    period: float = 4.0  # cycle length in seconds; sitstand uses 8.0
    randomize_phase: bool = True  # False -> each episode starts at phase 0 (standing)

    def build(self, env: ManagerBasedRlEnv) -> "GroundPickPhaseCommand":
        return GroundPickPhaseCommand(self, env)


# --------------------------------------------------------------------------- #
# Unified pose command machinery                                               #
# --------------------------------------------------------------------------- #
#
# Background: we deprecated the old NeckOffsetJointPositionAction +
# disturbance-randomization approach (where head/body movement was an external
# perturbation the policy was supposed to be robust to). That trained a weak,
# indirect signal — see `project_neck_offset_decoupling.md` for the
# post-mortem.
#
# Replacement: head and body pose are now *commands* — direct, dense policy
# inputs with tracking rewards. At deployment, the runtime feeds those slots
# with whatever pose the user requests; at training, they're sampled uniformly
# from per-dim ranges (kept non-zero from step 0 so input neurons stay alive)
# and ramped via curriculum.
#
# Layout, unified across all microduck policies for runtime obs compatibility:
#   command vector (13D) = [vx, vy, vtheta,           ← "twist" (velocity)
#                           neck_pitch, head_pitch,   ← "head_pose" (deltas)
#                           head_yaw, head_roll,
#                           body_x, body_y, body_z,   ← "body_pose" (deltas)
#                           body_roll, body_pitch, body_yaw]
# Total policy obs becomes 61D (51 - 3 + 13).
# --------------------------------------------------------------------------- #


from dataclasses import dataclass, field


class UniformPoseCommand(CommandTerm):
    """Generic N-dim uniform pose command.

    Samples each dim independently uniform in cfg.ranges[i] = (lo, hi) and holds
    the value between resamples. No metrics, no debug viz — keep it lightweight
    since we have many of these.
    """

    cfg: "UniformPoseCommandCfg"

    def __init__(self, cfg: "UniformPoseCommandCfg", env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.dim = len(cfg.ranges)
        self._command = torch.zeros(self.num_envs, self.dim, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _update_metrics(self) -> None:
        pass

    def _update_command(self) -> None:
        pass

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        if n == 0:
            return
        r = torch.empty(n, device=self.device)
        for i, (lo, hi) in enumerate(self.cfg.ranges):
            self._command[env_ids, i] = r.uniform_(lo, hi)
        # Explicit zero-command bucket. Uniform sampling essentially never
        # produces the all-zero command, so the deployment idle case ("hold the
        # nominal pose") would otherwise be absent from training (velocity
        # body-control run-1 lesson: the policy only stood still when a command
        # was present).
        if self.cfg.zero_command_prob > 0.0:
            zero_mask = torch.rand(n, device=self.device) < self.cfg.zero_command_prob
            self._command[env_ids[zero_mask]] = 0.0


@dataclass(kw_only=True)
class UniformPoseCommandCfg(CommandTermCfg):
    """Per-dim uniform ranges; builds a UniformPoseCommand."""
    # Tuple of (lo, hi) per dim. Length defines the command dim.
    ranges: tuple[tuple[float, float], ...] = ()
    # Probability that a resample yields the exact all-zero command.
    zero_command_prob: float = 0.0

    def build(self, env: ManagerBasedRlEnv) -> "UniformPoseCommand":
        return UniformPoseCommand(self, env)


def zero_command_padding(
    env: ManagerBasedRlEnv,
    dim: int,
) -> torch.Tensor:
    """Constant-zero obs term of width `dim`.

    Used by envs that don't actively track head/body commands (e.g. sitstand,
    ground_pick) but still need the unified 61D obs shape so the runtime can
    feed all policies with the same buffer layout.
    """
    return torch.zeros(env.num_envs, dim, device=env.device)


def head_pose_tracking(
    env: ManagerBasedRlEnv,
    command_name: str = "head_pose",
    std: float = 0.5,
    fine_std: float | None = None,
    fine_weight: float = 0.5,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Per-joint Gaussian reward for matching commanded neck/head deltas.

    Mean over the 4 neck/head joints of exp(-(err/std)^2). Result is (N,) in
    [0, 1]. Mean form (vs sum-of-squares) keeps gradient alive when only one
    joint is off — vs SOS where a single big error kills the whole reward.

    `std` is the per-joint tolerance: at err=std the per-joint reward is 1/e
    (~0.37). Pick std on the order of the command range so the gradient
    doesn't die as the curriculum widens.

    `fine_std` (optional) blends in a second, narrow Gaussian:
    (1-fine_weight)·exp(-(err/std)²) + fine_weight·exp(-(err/fine_std)²).
    Rationale: a single wide std (0.5 rad ≈ 29°) makes small errors nearly
    free — a 10° gravity sag on the heavy head costs ~0.03 reward, so the
    policy lets it droop. The narrow component (~0.1 rad) prices those small
    errors while the wide one keeps gradient alive at far commands during
    curriculum widening.

    cmd has shape (N, 4) = deltas from default joint positions in the order
    [neck_pitch, head_pitch, head_yaw, head_roll].

    On backlash models the measured angle is qpos[servo] + qpos[backlash] —
    the OUTPUT link, which is also what the encoder obs
    (joint_pos_rel_backlash) reports. Measuring the servo alone would let the
    head droop the backlash play reward-free AND penalize the policy for
    compensating it (servo biased up = servo-side "error"). On models without
    passive_*_backlash joints the mask is 0 and this reduces to the servo.
    """
    asset: Entity = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)  # (N, 4)

    if not hasattr(env, "_head_pose_neck_ids"):
        ids, names = asset.find_joints_by_actuator_names(_NECK_JOINT_PATTERNS)
        env._head_pose_neck_ids = torch.tensor(ids, device=env.device, dtype=torch.long)
        name_to_id = {n: i for i, n in enumerate(asset.joint_names)}
        bl = [name_to_id.get(f"passive_{n}_backlash") for n in names]
        env._head_pose_bl_ids = torch.tensor(
            [0 if b is None else b for b in bl], device=env.device, dtype=torch.long
        )
        env._head_pose_bl_mask = torch.tensor(
            [0.0 if b is None else 1.0 for b in bl], device=env.device
        )

    neck_ids = env._head_pose_neck_ids
    joint_pos = asset.data.joint_pos
    measured = (
        joint_pos[:, neck_ids]
        + joint_pos[:, env._head_pose_bl_ids] * env._head_pose_bl_mask
    )
    actual = measured - asset.data.default_joint_pos[:, neck_ids]
    err = actual - cmd
    per_joint = torch.exp(-(err / std) ** 2)
    if fine_std is not None:
        per_joint = (1.0 - fine_weight) * per_joint + fine_weight * torch.exp(
            -(err / fine_std) ** 2
        )
    return per_joint.mean(dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# NaN-safe wrappers for the sensor-derived critic observations.
#
# `robot_state_is_nan` covers joint + root state, so every obs derived from
# those is protected by the reset it triggers. The three terms below are NOT:
# they read sensor data (raycast heights, contact air-time, contact forces),
# which MuJoCo can return non-finite for while the integrated robot state is
# still clean. They are critic-only, so a single sanitized step costs the
# policy nothing, whereas letting the value through kills the entire run via
# rsl_rl's check_nan. Sanitizing here does not hide real physics blowups —
# those still terminate through nan_state and show up as
# Episode_Termination/nan_state in wandb.
# ─────────────────────────────────────────────────────────────────────────────


def _finite(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def foot_contact_forces_safe(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """NaN-safe `foot_contact_forces` (see note above)."""
    return _finite(_velocity_obs.foot_contact_forces(env, sensor_name))


def foot_height_safe(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """NaN-safe `foot_height` (see note above)."""
    return _finite(_velocity_obs.foot_height(env, sensor_name))


def foot_air_time_safe(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """NaN-safe `foot_air_time` (see note above)."""
    return _finite(_velocity_obs.foot_air_time(env, sensor_name))


def head_pose_bias_penalty(
    env: ManagerBasedRlEnv,
    command_name: str = "head_pose",
    tau_s: float = 1.0,
    gate_height_low: float | None = None,
    gate_height_high: float = 0.11,
    gate_tilt_full_deg: float = 20.0,
    gate_tilt_zero_deg: float = 45.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize the time-averaged (DC) neck/head tracking error: -mean(|EMA(err)|).

    Companion to ``head_pose_tracking``, which scores the INSTANTANEOUS error.
    Why a separate DC term instead of just tightening that Gaussian's std:
    walking unavoidably shakes a head that is 38% of the robot's mass, so an
    instantaneous tight-tolerance term is a permanent tax on walking that no
    policy can escape — measured at ~0.77/step against an air_time reward of
    ~1.01/step, which is exactly what made velocity run 2026-08-20 abandon
    stepping altogether (wandb 5yay13u4). The steady-state droop IS escapable:
    the policy can bias its neck command up to cancel gravity sag. Averaging
    over ``tau_s`` lets the oscillation cancel and prices only the bias.

    L1 (not Gaussian) on purpose: the gradient stays constant at large bias,
    where a tight Gaussian would be flat and dead.

    On backlash models the measured angle reads through the play, matching
    head_pose_tracking and the encoder obs.

    ``gate_height_low`` (optional): upright gate for recovery envs (standup /
    velstand), same smoothstep shape and semantics as body_ang_vel_at_height —
    zero below gate_height_low or above gate_tilt_zero_deg tilt, full above
    gate_height_high and below gate_tilt_full_deg. The gate multiplies the
    ERROR feeding the EMA (not just the output): while fallen/rising the EMA
    sees zero and decays, so arriving upright starts the bias clock from ~0
    instead of charging the whole ground phase's accumulated error at the
    finish line — that would be a reward wall right before recovery completes,
    the exact failure mode of the retired head_impact_penalty. The output is
    gated too, so a fresh fall stops the charge immediately.
    """
    asset: Entity = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)  # (N, 4)

    if not hasattr(env, "_head_pose_neck_ids"):
        # Share the id cache with head_pose_tracking (either may run first).
        head_pose_tracking(env, command_name=command_name, asset_cfg=asset_cfg)

    neck_ids = env._head_pose_neck_ids
    joint_pos = asset.data.joint_pos
    measured = (
        joint_pos[:, neck_ids]
        + joint_pos[:, env._head_pose_bl_ids] * env._head_pose_bl_mask
    )
    err = (measured - asset.data.default_joint_pos[:, neck_ids]) - cmd

    if gate_height_low is not None:
        z = torch.nan_to_num(
            asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2],
            nan=0.0,
        )
        t = torch.clamp(
            (z - gate_height_low) / max(gate_height_high - gate_height_low, 1e-6),
            0.0, 1.0,
        )
        gate = t * t * (3.0 - 2.0 * t)
        quat = asset.data.root_link_quat_w
        cos_tilt = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
        tilt_deg = torch.rad2deg(torch.acos(cos_tilt.clamp(-1.0, 1.0)))
        st = torch.clamp(
            (gate_tilt_zero_deg - tilt_deg)
            / max(gate_tilt_zero_deg - gate_tilt_full_deg, 1e-6),
            0.0, 1.0,
        )
        gate = gate * (st * st * (3.0 - 2.0 * st))
        err = err * gate.unsqueeze(-1)
    else:
        gate = None

    if not hasattr(env, "_head_bias_ema"):
        env._head_bias_ema = torch.zeros_like(err)
    # Freshly reset envs: drop the previous episode's accumulated bias.
    fresh = env.episode_length_buf <= 1
    env._head_bias_ema[fresh] = 0.0

    alpha = min(1.0, float(env.step_dt) / max(tau_s, 1e-6))
    env._head_bias_ema = (1.0 - alpha) * env._head_bias_ema + alpha * err
    out = -env._head_bias_ema.abs().mean(dim=-1)
    if gate is not None:
        out = out * gate
    return out


def body_pose_tracking_6d(
    env: ManagerBasedRlEnv,
    command_name: str = "body_pose",
    nominal_height: float = 0.095,
    xy_std: float = 0.02,
    z_std: float = 0.01,
    angle_std: float = math.radians(8),
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Mean of 6 per-axis Gaussian rewards for tracking commanded body pose.

    cmd has shape (N, 6) = [x, y, z, roll, pitch, yaw] all as deltas from the
    nominal standing pose (xy delta from spawn origin, z delta from
    nominal_height, angles delta from upright = 0).
    """
    asset: Entity = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)  # (N, 6)
    dx, dy, dz = cmd[:, 0], cmd[:, 1], cmd[:, 2]
    droll, dpitch, dyaw = cmd[:, 3], cmd[:, 4], cmd[:, 5]

    # Position relative to env spawn origin. nan_to_num because MuJoCo can
    # produce NaN on contact instability and we don't want to taint the reward.
    pos_w = asset.data.root_link_pos_w
    origin = env.scene.terrain.env_origins
    rel = torch.nan_to_num(pos_w - origin, nan=0.0)
    x_err = rel[:, 0] - dx
    y_err = rel[:, 1] - dy
    z_err = rel[:, 2] - (nominal_height + dz)

    # ZYX Euler from quat.
    quat = asset.data.root_link_quat_w
    qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    roll  = torch.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
    pitch = torch.asin(torch.clamp(2.0 * (qw * qy - qz * qx), -1.0, 1.0))
    yaw   = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

    roll_err  = roll  - droll
    pitch_err = pitch - dpitch
    yaw_err   = wrap_to_pi(yaw - dyaw)

    r_x = torch.exp(-(x_err / xy_std) ** 2)
    r_y = torch.exp(-(y_err / xy_std) ** 2)
    r_z = torch.exp(-(z_err / z_std) ** 2)
    r_r = torch.exp(-(roll_err  / angle_std) ** 2)
    r_p = torch.exp(-(pitch_err / angle_std) ** 2)
    r_w = torch.exp(-(yaw_err   / angle_std) ** 2)

    return (r_x + r_y + r_z + r_r + r_p + r_w) / 6.0


def termination_param_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    term_name: str,
    param_stages: list[dict],
) -> torch.Tensor:
    """Mutate a termination term's params at scheduled steps.

    TerminationManager keeps its own deepcopy of the cfg dict, so the live
    term_cfgs list must be edited directly — env.cfg.terminations is a no-op.
    Useful for disabling a termination later in training (e.g. set
    bad_orientation's limit_angle to pi at iter N so the robot can fall over
    without ending the episode and learn to recover).

    param_stages: list of {step: int, params: dict}. The dict is shallow-merged
    into the live term_cfg.params at the latest matching stage.
    """
    del env_ids
    tm = env.termination_manager
    if term_name not in tm._term_names:
        # Term was removed (e.g. play mode disables fell_over entirely).
        return torch.tensor(0.0)
    idx = tm._term_names.index(term_name)
    term_cfg = tm._term_cfgs[idx]

    current = param_stages[0]["params"]
    for stage in param_stages:
        if env.common_step_counter >= stage["step"]:
            current = stage["params"]
    term_cfg.params.update(current)

    first_val = next(iter(current.values()))
    return torch.tensor(float(first_val) if isinstance(first_val, (int, float)) else 0.0)


def body_pose_tracking_locomotion(
    env: ManagerBasedRlEnv,
    command_name: str = "body_pose",
    nominal_height: float = 0.105,
    xy_std: float = 0.02,
    z_std: float = 0.03,
    angle_std: float = math.radians(30),
    axis_weights: tuple[float, float, float, float, float, float] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    vel_gate_command_name: str | None = None,
    vel_gate_std: float = 0.1,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    feet_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
) -> torch.Tensor:
    """Locomotion-aware 6D body pose tracking.

    Same shape as body_pose_tracking_6d (6D cmd, mean of 6 Gaussians), but
    x/y/yaw are measured *relative to the feet support polygon*, not the spawn
    origin. This makes the reward meaningful while the robot walks (or stands):

      x, y  : trunk position − feet-centroid, rotated into trunk body frame.
              dx = +0.02 means "lean trunk 2 cm forward of foot centroid."
      z     : trunk world height (− nominal_height) — locomotion-neutral.
      roll  : trunk world roll                     — locomotion-neutral.
      pitch : trunk world pitch                    — locomotion-neutral.
      yaw   : trunk world yaw − circular-mean(feet site yaws). dyaw = +0.3 rad
              means "twist the trunk 17° relative to where the feet point."

    The body_pose_tracking_6d reward measures x/y/yaw vs spawn origin / world
    yaw, which kills the gradient as soon as the robot translates or turns. This
    version stays meaningful regardless of where in the world the robot is.
    """
    asset: Entity = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)  # (N, 6)
    dx, dy, dz = cmd[:, 0], cmd[:, 1], cmd[:, 2]
    droll, dpitch, dyaw = cmd[:, 3], cmd[:, 4], cmd[:, 5]

    pos_w = asset.data.root_link_pos_w
    quat = asset.data.root_link_quat_w
    qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    trunk_yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    roll  = torch.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
    pitch = torch.asin(torch.clamp(2.0 * (qw * qy - qz * qx), -1.0, 1.0))

    # Feet centroid in world frame.
    foot_pos = asset.data.site_pos_w[:, feet_cfg.site_ids]   # (N, 2, 3)
    foot_quat = asset.data.site_quat_w[:, feet_cfg.site_ids] # (N, 2, 4)
    feet_centroid = foot_pos.mean(dim=1)                     # (N, 3)

    # Trunk xy in body frame relative to feet centroid (rotate world Δxy by −yaw).
    dx_w = pos_w[:, 0] - feet_centroid[:, 0]
    dy_w = pos_w[:, 1] - feet_centroid[:, 1]
    cos_y = torch.cos(trunk_yaw)
    sin_y = torch.sin(trunk_yaw)
    x_body =  cos_y * dx_w + sin_y * dy_w
    y_body = -sin_y * dx_w + cos_y * dy_w

    # Z relative to spawn-origin terrain height (still in world).
    origin = env.scene.terrain.env_origins
    z_world = torch.nan_to_num(pos_w[:, 2] - origin[:, 2], nan=0.0)

    # Feet yaws → circular mean. NOTE: this depends on the site orientation
    # matching the foot pointing direction; if the site frame is rotated, this
    # yaw reference may have an offset (constant per-env, so dyaw=0 still maps
    # to "feet-aligned").
    fqw, fqx, fqy, fqz = foot_quat[..., 0], foot_quat[..., 1], foot_quat[..., 2], foot_quat[..., 3]
    foot_yaws = torch.atan2(2.0 * (fqw * fqz + fqx * fqy), 1.0 - 2.0 * (fqy * fqy + fqz * fqz))  # (N, 2)
    mean_foot_yaw = torch.atan2(torch.sin(foot_yaws).mean(dim=1), torch.cos(foot_yaws).mean(dim=1))

    x_err     = x_body - dx
    y_err     = y_body - dy
    z_err     = z_world - (nominal_height + dz)
    roll_err  = roll  - droll
    pitch_err = pitch - dpitch
    yaw_err   = wrap_to_pi(trunk_yaw - mean_foot_yaw - dyaw)

    r_x = torch.exp(-(x_err / xy_std) ** 2)
    r_y = torch.exp(-(y_err / xy_std) ** 2)
    r_z = torch.exp(-(z_err / z_std) ** 2)
    r_r = torch.exp(-(roll_err  / angle_std) ** 2)
    r_p = torch.exp(-(pitch_err / angle_std) ** 2)
    r_w = torch.exp(-(yaw_err   / angle_std) ** 2)

    # Per-axis weighted mean. Pass axis_weights=(0,0,1,1,1,1) to disable xy
    # tracking — useful when xy lean is mechanically coupled to pitch/roll on
    # the robot, making independent xy commands a noise source rather than a
    # learnable objective.
    wx, wy, wz, wr, wp, wyaw = axis_weights
    total_w = wx + wy + wz + wr + wp + wyaw
    reward = (wx*r_x + wy*r_y + wz*r_z + wr*r_r + wp*r_p + wyaw*r_w) / max(total_w, 1e-6)

    # Optional gate: when vel_gate_command_name is set, scale the reward by a
    # Gaussian on the velocity command's magnitude. With vel_gate_std ≈ 0.1,
    # the gate is ~1 when commanded velocity is 0 and decays to ~exp(-9)≈0
    # by |vel_cmd|≥0.3 — body tracking only meaningfully contributes when the
    # robot is supposed to be standing still. Avoids the tracking vs walking
    # conflict that prevented the previous run from learning either well.
    if vel_gate_command_name is not None:
        # Gate on commanded LINEAR velocity only (xy) — turning in place still
        # leaves body pose meaningful, but walking forward/sideways doesn't.
        vel_cmd = env.command_manager.get_command(vel_gate_command_name)  # (N, 3)
        vel_mag = torch.linalg.vector_norm(vel_cmd[:, :2], dim=-1)
        gate = torch.exp(-(vel_mag / vel_gate_std) ** 2)
        reward = reward * gate

    return reward


def pose_command_range_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    range_stages: list[dict],
) -> torch.Tensor:
    """Ramp a UniformPoseCommand's per-dim ranges over training.

    range_stages: list of {step: int, ranges: tuple[(lo, hi), ...]}.
    The first stage applies before its step; latest passed stage wins.
    Always uses the live CommandManager term cfg (NOT env.cfg.commands) so
    updates take effect — CommandManager keeps its own term refs and reads
    `term.cfg.ranges` each resample.
    """
    del env_ids

    term = env.command_manager.get_term(command_name)
    assert term is not None, f"Command term '{command_name}' not found"
    cfg = term.cfg  # type: ignore[assignment]

    current = range_stages[0]["ranges"]
    for stage in range_stages:
        if env.common_step_counter >= stage["step"]:
            current = stage["ranges"]

    cfg.ranges = tuple(current)
    # Return the max abs range as a scalar for wandb visibility.
    max_abs = max((max(abs(lo), abs(hi)) for lo, hi in current), default=0.0)
    return torch.tensor(max_abs)


# ─────────────────────────────────────────────────────────────────────────────
# Gait-shaping penalties ported from mjlab_microban (microban velocity recipe).
# ─────────────────────────────────────────────────────────────────────────────
def no_stepping_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str = "twist",
    command_threshold: float = 0.01,
) -> torch.Tensor:
    """Penalize feet in the air when the commanded speed is below threshold.

    Discourages marching in place when the robot should stand still. Returns the
    count of airborne feet per environment (use with a negative weight).
    Ported from mjlab_microban.
    """
    command = env.command_manager.get_command(command_name)  # (N, 3)
    cmd_speed = torch.norm(command[:, :2], dim=-1) + torch.abs(command[:, 2])
    below_threshold = cmd_speed < command_threshold

    sensor = env.scene.sensors[sensor_name]
    found = sensor.data.found  # (N, num_feet) or (N, num_feet, num_slots)
    if found.dim() == 3:
        found = found.any(dim=-1)  # (N, num_feet)
    in_air = ~found.bool()

    return in_air.float().sum(dim=-1) * below_threshold.float()


def feet_distance_penalty(
    env: ManagerBasedRlEnv,
    min_dist: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize the feet getting too close to each other in the horizontal plane.

    Returns ``clamp(min_dist - d, min=0)`` per env (use with a negative weight),
    where ``d`` is the horizontal (xy) distance between the two foot sites.
    Ported from mjlab_microban. Not wired into velocity yet — pinned for later.
    """
    asset: Entity = env.scene[asset_cfg.name]
    foot_pos_xy = asset.data.site_pos_w[:, asset_cfg.site_ids, :2]  # (N, 2, 2)
    dist = torch.norm(foot_pos_xy[:, 0] - foot_pos_xy[:, 1], dim=-1)  # (N,)
    return torch.clamp(min_dist - dist, min=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Non-accumulating domain randomization (restore-nominal-then-apply).
#
# The stock mdp.randomize_field with operation="add"/"scale" + mode="reset"
# reads the CURRENT model value and applies the op to it, with no restore to
# nominal — so on every episode reset the perturbation STACKS on the previous
# one and the parameter random-walks away from nominal over training. For
# body_ipos (CoM) this was the long-standing microduck instability: the CoM
# drifted centimeters off-center over hundreds of resets → progressively
# unbalanced robot → falls more → reward/episode-length collapse after the early
# peak. These functions mirror randomize_mass_and_inertia: cache the nominal
# once, restore it before each draw, then apply a freshly-sampled perturbation —
# so it is re-sampled per episode but never accumulates.
# ─────────────────────────────────────────────────────────────────────────────
def randomize_com(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    ranges: tuple[float, float],
    field: str = "body_ipos",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Randomize body CoM (body_ipos) per episode WITHOUT accumulating.

    Drop-in replacement for the buggy mdp.randomize_field(add, body_ipos, reset).
    ``ranges`` is (lo, hi) applied to all 3 CoM axes; the com_range curriculum
    updates this same ``ranges`` param. ``field`` is declared so the event can run
    with ``domain_randomization=True`` (mjlab reads params["field"] to expand that
    model field per-env).
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.int)

    asset: Entity = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    if isinstance(body_ids, slice):
        body_ids = list(range(asset.num_bodies))[body_ids]
    body_indices = asset.indexing.body_ids[body_ids]

    mf = getattr(env.sim.model, field)
    # Key the cache by (field, body set): multiple randomize_com events can share
    # the same field (e.g. trunk + head both randomize body_ipos) and must NOT
    # collide on a single _original_body_ipos attr — their body counts differ.
    _bidx = body_indices.tolist() if hasattr(body_indices, "tolist") else list(body_indices)
    cache_attr = f"_original_{field}_" + "_".join(str(int(i)) for i in _bidx)
    # Cache nominal on first call (model[0] is still nominal at that point).
    if not hasattr(env, cache_attr):
        setattr(env, cache_attr, mf[0, body_indices].clone())
    nominal = getattr(env, cache_attr)

    num_envs = len(env_ids)
    num_bodies = len(body_indices)

    # Restore nominal first (prevents accumulation), then add a fresh offset.
    mf[env_ids[:, None], body_indices] = nominal.unsqueeze(0).expand(num_envs, -1, -1)
    lo, hi = ranges
    offsets = torch.rand(num_envs, num_bodies, 3, device=env.device) * (hi - lo) + lo
    mf[env_ids[:, None], body_indices] += offsets
    return torch.tensor(float(hi))


def randomize_dof_field_scaled(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    field: str,
    scale_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Scale a per-dof model field (e.g. dof_frictionloss/dof_damping) per episode
    WITHOUT accumulating: restore nominal, then apply a fresh scale.

    ``field`` doubles as the domain_randomization field name. NOTE: under the BAM
    actuator, dof_frictionloss and dof_damping are zeroed in edit_spec (BAM models
    friction itself), so scaling them is a no-op — these only matter with the XML
    position actuator. Kept correct to avoid the accumulation footgun if re-enabled.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.int)

    asset: Entity = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, slice):
        joint_ids = list(range(len(asset.indexing.joint_ids)))[joint_ids]
    dof_indices = asset.indexing.joint_v_adr[joint_ids]

    mf = getattr(env.sim.model, field)
    cache_attr = f"_original_{field}"
    if not hasattr(env, cache_attr):
        setattr(env, cache_attr, mf[0, dof_indices].clone())
    nominal = getattr(env, cache_attr)

    num_envs = len(env_ids)
    num_dofs = len(dof_indices)

    mf[env_ids[:, None], dof_indices] = nominal.unsqueeze(0).expand(num_envs, -1)
    lo, hi = scale_range
    scales = torch.rand(num_envs, num_dofs, device=env.device) * (hi - lo) + lo
    mf[env_ids[:, None], dof_indices] *= scales
    return torch.tensor(float(hi))


# =============================================================================
# BallKick task — ball reset event, kick rewards, critic-only ball observations
# =============================================================================


def _ball_kick_dir(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Per-env world-frame kick direction (XY unit vector), lazily allocated.

    Set by ``reset_ball_in_front_of_foot`` to the robot's forward direction at
    episode reset. Frozen for the episode so the policy can't redefine "forward"
    by turning after the kick.
    """
    if not hasattr(env, "_ball_kick_dir_w"):
        env._ball_kick_dir_w = torch.zeros(env.num_envs, 2, device=env.device)
        env._ball_kick_dir_w[:, 0] = 1.0
    return env._ball_kick_dir_w


def reset_ball_in_front_of_foot(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    offset: tuple = (0.09, -0.042),
    noise_xy: float = 0.015,
    ball_radius: float = 0.035,
    asset_name: str = "ball",
):
    """Place the ball in front of the (right) foot; store the kick direction.

    ``offset`` is the nominal ball-center position in the robot's yaw frame:
    at HOME the right foot is centered at (0, -0.042) with the toe tip at
    x≈0.034, so (0.08, -0.042) puts a 35mm-radius ball ~1cm in front of the
    toe. ``noise_xy`` (uniform ± per axis) is the placement DR: the policy is
    BLIND to the ball, so this is what forces a swing that works across the
    real-world placement error.

    Reads the robot root from qpos directly (root_link_pos_w lags until the
    next forward()); must be registered AFTER reset_base / set_ground_state
    (events run in dict insertion order) so the robot pose is final.
    """
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device)
    robot: Entity = env.scene["robot"]
    ball: Entity = env.scene[asset_name]

    root = env.sim.data.qpos[env_ids][:, robot.indexing.free_joint_q_adr]
    qw, qx, qy, qz = root[:, 3], root[:, 4], root[:, 5], root[:, 6]
    yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)

    n = len(env_ids)
    off = torch.tensor(offset, device=env.device, dtype=torch.float).repeat(n, 1)
    off += (torch.rand(n, 2, device=env.device) * 2.0 - 1.0) * noise_xy

    pose = torch.zeros(n, 7, device=env.device)
    pose[:, 0] = root[:, 0] + cos_y * off[:, 0] - sin_y * off[:, 1]
    pose[:, 1] = root[:, 1] + sin_y * off[:, 0] + cos_y * off[:, 1]
    pose[:, 2] = env.scene.terrain.env_origins[env_ids, 2] + ball_radius
    pose[:, 3] = 1.0  # identity quat
    ball.write_root_link_pose_to_sim(pose, env_ids)
    ball.write_root_link_velocity_to_sim(
        torch.zeros(n, 6, device=env.device), env_ids
    )

    kick_dir = _ball_kick_dir(env)
    kick_dir[env_ids, 0] = cos_y
    kick_dir[env_ids, 1] = sin_y


def ball_forward_velocity(
    env: ManagerBasedRlEnv,
    asset_name: str = "ball",
    max_speed: float = 5.0,
) -> torch.Tensor:
    """Ball XY velocity along the per-env kick direction, clamped to [0, max].

    Dense and linear-in-speed up to ``max_speed``: every extra bit of forward
    ball speed pays more every step the ball keeps rolling, so exploration
    nudges bootstrap the kick with no peak-detection machinery. Backward /
    lateral ball motion earns 0 rather than a penalty — a mis-hit shouldn't
    scare the policy away from contacting the ball at all.

    With ``max_speed`` set to a TARGET speed (rather than a large cap), pair
    with ``ball_speed_overshoot_penalty``: the reward saturating at the target
    alone does NOT remove "harder is better" — a harder kick keeps the ball
    at/above the cap for more steps, so the rolling-time integral still grows
    with strike speed. The overshoot penalty is what makes the target the
    actual optimum.
    """
    ball: Entity = env.scene[asset_name]
    vel_xy = ball.data.root_link_lin_vel_w[:, :2]
    fwd = (vel_xy * _ball_kick_dir(env)).sum(dim=1)
    return torch.nan_to_num(fwd, nan=0.0).clamp(0.0, max_speed)


def ball_speed_overshoot_penalty(
    env: ManagerBasedRlEnv,
    asset_name: str = "ball",
    target_speed: float = 1.0,
    max_penalty: float = 5.0,
) -> torch.Tensor:
    """Ball forward speed in excess of ``target_speed`` (linear, ≥ 0).

    Companion to ``ball_forward_velocity`` for a target-speed kick: below the
    target this is 0 (the capped linear reward provides the upward gradient);
    above it, each m/s of overshoot costs linearly every step it persists.
    Keep this term's |weight| BELOW the capped reward's weight so the combined
    landscape peaks at the target with a gentler slope on the overshoot side —
    erring slightly hard must stay cheaper than not kicking at all.
    """
    ball: Entity = env.scene[asset_name]
    vel_xy = ball.data.root_link_lin_vel_w[:, :2]
    fwd = (vel_xy * _ball_kick_dir(env)).sum(dim=1)
    over = torch.nan_to_num(fwd, nan=0.0) - target_speed
    return over.clamp(0.0, max_penalty)


def single_foot_grounded_reward(
    env: ManagerBasedRlEnv,
    sensor_name: str,
) -> torch.Tensor:
    """Binary reward: 1 while the sensed foot touches the terrain.

    Single-foot variant of ``feet_grounded_reward`` — used to pin the SUPPORT
    foot during the kick (anti-hop): swinging the right leg is free, lifting
    the left foot costs this reward every step.
    """
    if sensor_name not in env.scene.sensors:
        return torch.zeros(env.num_envs, device=env.device)
    found = env.scene.sensors[sensor_name].data.found
    if found.dim() > 1:
        found = found.sum(dim=-1)
    return torch.clamp(found, 0.0, 1.0)


def ball_pos_in_base(
    env: ManagerBasedRlEnv,
    asset_name: str = "ball",
) -> torch.Tensor:
    """Ball position relative to the robot root, in the robot's base frame.

    CRITIC-ONLY observation (asymmetric actor-critic): the deployed policy has
    no ball sensing, so the actor must stay blind to the ball — the critic can
    still use it to predict the kick payoff.
    """
    robot: Entity = env.scene["robot"]
    ball: Entity = env.scene[asset_name]
    rel = ball.data.root_link_pos_w - robot.data.root_link_pos_w
    rot = matrix_from_quat(robot.data.root_link_quat_w)
    return torch.bmm(rot.transpose(1, 2), rel.unsqueeze(-1)).squeeze(-1)


def ball_vel_in_base(
    env: ManagerBasedRlEnv,
    asset_name: str = "ball",
) -> torch.Tensor:
    """Ball linear velocity in the robot's base frame. CRITIC-ONLY (see above)."""
    robot: Entity = env.scene["robot"]
    ball: Entity = env.scene[asset_name]
    rot = matrix_from_quat(robot.data.root_link_quat_w)
    vel = ball.data.root_link_lin_vel_w
    return torch.bmm(rot.transpose(1, 2), vel.unsqueeze(-1)).squeeze(-1)


# --------------------------------------------------------------------------- #
# Tâche SPIN — rotation rapide sur place sur rollers                            #
# --------------------------------------------------------------------------- #
# Enveloppe de phase : la commande du slot bouton porte une phase, qui pilote
# une VITESSE DE LACET cible en trapèze (et non une pose comme le crouch).
#   [0, accel_end)        0.5 s   0 -> rate_max    (lancement)
#   [accel_end, hold_end) 1.6 s   rate_max         (régime)
#   [hold_end, brake_end) 0.5 s   rate_max -> 0    (freinage)
#   [brake_end, 1.0)      1.4 s   0                (repos debout)
# Aire sous l'enveloppe sur un cycle = 2.1 * SPIN_RATE_MAX rad. À 3.0 rad/s :
# 2.1 * 3.0 = 6.3 rad ~ 1 tour (et non ~2, comme avec l'ancienne cible 6.0).
SPIN_PERIOD = 4.0
SPIN_RATE_MAX = 3.0
SPIN_ACCEL_END = 0.125
SPIN_HOLD_END = 0.525
SPIN_BRAKE_END = 0.650


def spin_rate_by_phase(
    phase: torch.Tensor,
    rate_max: float = SPIN_RATE_MAX,
    accel_end: float = SPIN_ACCEL_END,
    hold_end: float = SPIN_HOLD_END,
    brake_end: float = SPIN_BRAKE_END,
) -> torch.Tensor:
    """Vitesse de lacet cible (rad/s, positive = anti-horaire) le long de la phase."""
    w = torch.zeros_like(phase)
    accel = phase < accel_end
    w = torch.where(accel, rate_max * phase / accel_end, w)
    hold = (phase >= accel_end) & (phase < hold_end)
    w = torch.where(hold, torch.full_like(phase, rate_max), w)
    brake = (phase >= hold_end) & (phase < brake_end)
    w = torch.where(
        brake, rate_max * (1.0 - (phase - hold_end) / (brake_end - hold_end)), w
    )
    return w


def spin_gate_by_phase(
    phase: torch.Tensor,
    rate_max: float = SPIN_RATE_MAX,
    accel_end: float = SPIN_ACCEL_END,
    hold_end: float = SPIN_HOLD_END,
    brake_end: float = SPIN_BRAKE_END,
) -> torch.Tensor:
    """Porte de shaping dans [0,1] = enveloppe normalisée.

    Vaut 0 sur tout le segment de repos : les amorces (ciseau des jambes,
    différentiel des roues) ne s'appliquent que pendant lancement + régime, donc
    le robot revient en station neutre avant de rendre la main à la policy roller.
    """
    return spin_rate_by_phase(phase, rate_max, accel_end, hold_end, brake_end) / rate_max


def spin_phase_from_command(cmd: torch.Tensor) -> torch.Tensor:
    """Récupère la phase [0,1) depuis la commande [cos(2πφ), sin(2πφ), 0] du slot."""
    return (torch.atan2(cmd[:, 1], cmd[:, 0]) / (2 * torch.pi)) % 1.0


def _spin_target_rate(
    env: ManagerBasedRlEnv,
    command_name: str,
    rate_max: float,
    accel_end: float,
    hold_end: float,
    brake_end: float,
) -> torch.Tensor:
    phase = spin_phase_from_command(env.command_manager.get_command(command_name))
    return spin_rate_by_phase(phase, rate_max, accel_end, hold_end, brake_end)


def _spin_gate(
    env: ManagerBasedRlEnv,
    command_name: str,
    rate_max: float,
    accel_end: float,
    hold_end: float,
    brake_end: float,
) -> torch.Tensor:
    phase = spin_phase_from_command(env.command_manager.get_command(command_name))
    return spin_gate_by_phase(phase, rate_max, accel_end, hold_end, brake_end)


def spin_rate_reward_from_values(
    omega_z: torch.Tensor, omega_target: torch.Tensor, std: float
) -> torch.Tensor:
    """Gaussienne sur l'erreur de vitesse de lacet (fonction pure, testable)."""
    return torch.exp(-(((omega_z - omega_target) / std) ** 2))


def spin_rate_track(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    std: float = 1.5,
    rate_max: float = SPIN_RATE_MAX,
    accel_end: float = SPIN_ACCEL_END,
    hold_end: float = SPIN_HOLD_END,
    brake_end: float = SPIN_BRAKE_END,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Objectif principal du spin : suivre la vitesse de lacet cible ω*(φ).

    ω_z est pris en repère corps (c'est ce que voit le gyro de l'IMU, donc ce que
    la policy observe). Une rotation dans le mauvais sens est plus punie que
    l'immobilité, la gaussienne étant centrée sur une cible positive.
    """
    asset: Entity = env.scene[asset_cfg.name]
    omega_z = asset.data.root_link_ang_vel_b[:, 2]
    target = _spin_target_rate(env, command_name, rate_max, accel_end, hold_end, brake_end)
    return spin_rate_reward_from_values(omega_z, target, std)


def spin_rate_l1(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    rate_max: float = SPIN_RATE_MAX,
    accel_end: float = SPIN_ACCEL_END,
    hold_end: float = SPIN_HOLD_END,
    brake_end: float = SPIN_BRAKE_END,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Bootstrap L1 : gradient constant vers la cible même quand la gaussienne
    de `spin_rate_track` sature loin de la cible. À utiliser avec un poids
    POSITIF (la valeur retournée est déjà négative)."""
    asset: Entity = env.scene[asset_cfg.name]
    omega_z = asset.data.root_link_ang_vel_b[:, 2]
    target = _spin_target_rate(env, command_name, rate_max, accel_end, hold_end, brake_end)
    return -torch.abs(omega_z - target)


SPIN_LAUNCH_DRIFT_SCALE = 0.2  # atténuation du coût de dérive pendant le lancement


def spin_stay_in_place(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    launch_scale: float = SPIN_LAUNCH_DRIFT_SCALE,
    accel_end: float = SPIN_ACCEL_END,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Coût ‖v_xy‖² du tronc : tourner SUR PLACE, et tuer l'élan d'entrée.

    Pas d'état de référence (contrairement à une dérive mesurée depuis le reset),
    donc reste valide sur les 5 cycles d'un épisode. À utiliser avec un poids
    NÉGATIF.

    ATTÉNUÉ PENDANT LE LANCEMENT : sur `[0, accel_end)` le robot doit pousser au
    sol pour s'injecter du moment angulaire, et l'état d'entrée lui donne jusqu'à
    0.3 m/s qu'il est censé CONVERTIR en rotation. Facturer la translation à plein
    tarif à cet instant s'oppose donc directement à l'objectif. Le coût est
    multiplié par `launch_scale` sur ce seul segment, et vaut plein tarif ensuite
    (régime, freinage, repos) où « sur place » est le vrai critère.

    Contrairement aux autres amorces du spin, ce terme n'est PAS éteint par
    `spin_gate_by_phase` : pendant le repos on veut justement qu'il reste plein,
    puisque c'est là que le robot doit être immobile.
    """
    asset: Entity = env.scene[asset_cfg.name]
    v_xy = asset.data.root_link_lin_vel_b[:, :2]
    cost = torch.sum(torch.square(v_xy), dim=1)

    phase = spin_phase_from_command(env.command_manager.get_command(command_name))
    scale = torch.where(
        phase < accel_end,
        torch.full_like(cost, launch_scale),
        torch.ones_like(cost),
    )
    return cost * scale


# Demi-voie mesurée sur le modèle rollers (pose HOME, sites left_foot/right_foot) :
# 0.0499 m, contre 0.03 m estimé au spec. Conséquence mécanique de SPIN_RATE_MAX
# (A1) : différentiel attendu = 2*SPIN_RATE_MAX*demi_voie/r, r = 0.0175 m.
# À l'ancienne cible 6.0 rad/s : 2*6.0*0.0499/0.0175 = 34.2 rad/s (retenu comme
# 34.0, soit +71% par rapport aux 20.0 estimés au spec -> seuil de 30% dépassé).
# À la nouvelle cible 3.0 rad/s : 2*3.0*0.0499/0.0175 = 17.1 rad/s. Laisser 34.0
# ici plafonnerait le terme à tanh(17.1/34) = 0.47 de son propre maximum, ce qui
# affaiblirait exactement le shaping qu'on veut renforcer (cf. spin_stay_in_place).
SPIN_WHEEL_OMEGA_SCALE = 17.0  # rad/s ; recalibré sur la demi-voie mesurée et SPIN_RATE_MAX = 3.0


def spin_wheel_differential_from_values(
    diff: torch.Tensor, gate: torch.Tensor, omega_scale: float
) -> torch.Tensor:
    """Fonction pure : tanh du différentiel de roues, portée par gate, clampée ≥ 0."""
    return gate * torch.tanh(torch.clamp(diff, min=0.0) / omega_scale)


def spin_wheel_differential(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    omega_scale: float = SPIN_WHEEL_OMEGA_SCALE,
    rate_max: float = SPIN_RATE_MAX,
    accel_end: float = SPIN_ACCEL_END,
    hold_end: float = SPIN_HOLD_END,
    brake_end: float = SPIN_BRAKE_END,
) -> torch.Tensor:
    """Récompense la rotation EN ROULEMENT (et non en patinage).

    Pour un spin anti-horaire, le patin gauche recule et le droit avance ; les 4
    roues tournant positif en marche avant, cela donne ω_D − ω_G > 0. Le tanh
    sature à `omega_scale` pour éviter la course à la vitesse de roue.
    """
    asset: Entity = env.scene["robot"]
    lf_ids, _ = asset.find_joints("passive_LF_?wheel")
    lr_ids, _ = asset.find_joints("passive_LR_?wheel")
    rf_ids, _ = asset.find_joints("passive_RF_?wheel")
    rr_ids, _ = asset.find_joints("passive_RR_?wheel")

    vel = asset.data.joint_vel
    omega_left = (vel[:, lf_ids[0]] + vel[:, lr_ids[0]]) / 2.0
    omega_right = (vel[:, rf_ids[0]] + vel[:, rr_ids[0]]) / 2.0
    gate = _spin_gate(env, command_name, rate_max, accel_end, hold_end, brake_end)
    return spin_wheel_differential_from_values(
        omega_right - omega_left, gate, omega_scale
    )


def spin_grounded(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str = "twist",
    rate_max: float = SPIN_RATE_MAX,
    accel_end: float = SPIN_ACCEL_END,
    hold_end: float = SPIN_HOLD_END,
    brake_end: float = SPIN_BRAKE_END,
) -> torch.Tensor:
    """Les deux lames au sol pendant le spin — empêche « je saute et je vrille ».

    Variante de `grounded_reward` du swizzle, qui n'est pas réutilisable ici :
    elle se pondère par cmd_x, qui vaut cos(2πφ) sur la commande de phase.
    """
    from mjlab.sensor import ContactSensor

    sensor: ContactSensor = env.scene[sensor_name]
    contact_time = sensor.data.current_contact_time  # (num_envs, num_feet)
    assert contact_time is not None
    n_contact = torch.sum((contact_time > 0.0).float(), dim=1)
    grounded = (n_contact >= 2).float()
    gate = _spin_gate(env, command_name, rate_max, accel_end, hold_end, brake_end)
    return grounded * gate


def leg_antisymmetry(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    joint_bases: tuple = ("hip_pitch", "knee"),
    rate_max: float = SPIN_RATE_MAX,
    accel_end: float = SPIN_ACCEL_END,
    hold_end: float = SPIN_HOLD_END,
    brake_end: float = SPIN_BRAKE_END,
) -> torch.Tensor:
    """Amorce le CISEAU des jambes (une avant / une arrière) pendant le spin.

    Le robot a des conventions de signe MIROIR gauche/droite : une pose
    symétrique satisfait q_G + q_D ≈ 0 (cf. `leg_symmetry_reward`), donc le
    ciseau satisfait q_G ≈ q_D. On retourne `gate(φ) · (−mean|q_G − q_D|)` — à
    utiliser avec un poids POSITIF, décroissant par curriculum : l'amorce
    s'efface pour laisser la policy affiner son propre geste.
    """
    asset: Entity = env.scene[asset_cfg.name]
    left, right = [], []
    for base in joint_bases:
        li, _ = asset.find_joints([f"left_{base}"])
        ri, _ = asset.find_joints([f"right_{base}"])
        left.append(li[0])
        right.append(ri[0])
    lids = torch.tensor(left, device=env.device)
    rids = torch.tensor(right, device=env.device)

    q = asset.data.joint_pos
    scissor = -torch.abs(q[:, lids] - q[:, rids]).mean(dim=-1)
    gate = _spin_gate(env, command_name, rate_max, accel_end, hold_end, brake_end)
    return gate * scissor


# =============================================================================
# Backlash model — encoder-through-backlash joint observations
# =============================================================================
# The backlash model (robot_allcollisions_backlash.xml) puts an unactuated
# ``passive_<joint>_backlash`` hinge in series with each servo joint. The link
# angle is qpos[servo] + qpos[backlash], and the real encoder sits on the
# OUTPUT side of the play — it reads the sum. These obs replace joint_pos_rel /
# joint_vel_rel in backlash tasks (see tasks/backlash.py) so the policy sees
# exactly what the runtime will feed it. The asset_cfg regex is expected to
# select only the servo joints (the usual ``^(?!passive_).*``).


def _backlash_encoder_ids(
    env: "ManagerBasedRlEnv",
    asset: Entity,
    asset_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(main_ids, backlash_ids, mask) — cached per (entity, joint selection).

    mask is 1.0 where a matching passive_<name>_backlash joint exists, so the
    same obs functions run unchanged on models without backlash joints.
    """
    key = (asset_cfg.name, str(asset_cfg.joint_ids))
    cache = env.__dict__.setdefault("_backlash_encoder_cache", {})
    hit = cache.get(key)
    if hit is not None:
        return hit

    names = asset.joint_names
    jnt_ids = asset_cfg.joint_ids
    if isinstance(jnt_ids, slice):
        main_ids = list(range(len(names)))[jnt_ids]
    else:
        main_ids = [int(i) for i in jnt_ids]
    name_to_id = {n: i for i, n in enumerate(names)}
    bl_ids, mask = [], []
    for i in main_ids:
        bl = name_to_id.get(f"passive_{names[i]}_backlash")
        bl_ids.append(0 if bl is None else bl)
        mask.append(0.0 if bl is None else 1.0)

    device = asset.data.joint_pos.device
    out = (
        torch.tensor(main_ids, dtype=torch.long, device=device),
        torch.tensor(bl_ids, dtype=torch.long, device=device),
        torch.tensor(mask, dtype=torch.float32, device=device),
    )
    cache[key] = out
    return out


def joint_pos_rel_backlash(
    env: "ManagerBasedRlEnv",
    biased: bool = False,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """joint_pos_rel where the encoder reads through the backlash hinge.

    Returns (qpos[servo] + qpos[backlash]) - default[servo]. With biased=True
    the per-env encoder-calibration bias is applied to the servo reading (one
    encoder per servo → one bias per joint; the backlash summand stays raw).
    """
    asset: Entity = env.scene[asset_cfg.name]
    main_ids, bl_ids, mask = _backlash_encoder_ids(env, asset, asset_cfg)
    joint_pos = asset.data.joint_pos_biased if biased else asset.data.joint_pos
    pos = joint_pos[:, main_ids] + asset.data.joint_pos[:, bl_ids] * mask
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    return pos - default_joint_pos[:, main_ids]


def joint_vel_rel_backlash(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """joint_vel_rel where the encoder reads through the backlash hinge.

    The firmware derives present_velocity from encoder positions, so it also
    sees the backlash motion: qvel[servo] + qvel[backlash].
    """
    asset: Entity = env.scene[asset_cfg.name]
    main_ids, bl_ids, mask = _backlash_encoder_ids(env, asset, asset_cfg)
    vel = asset.data.joint_vel[:, main_ids] + asset.data.joint_vel[:, bl_ids] * mask
    default_joint_vel = asset.data.default_joint_vel
    assert default_joint_vel is not None
    return vel - default_joint_vel[:, main_ids]


# ─────────────────────────────────────────────────────────────────────────────
# Sit↔Stand posture command + posture-conditioned rewards (sitstand env).
#
# One policy, both directions: the command is a single sit/stand flag carried
# in the twist slot (cmd = [sit_flag, 0, 0], so "stand" is the all-zero
# command — same deployment idle as every other policy). All task rewards
# below select their target (SIT keyframe + SIT_Z vs HOME + STAND_Z) from the
# live command, per env, so the same reward stack drives the descent, the
# seated rest, the rise and the standing rest. Uses the _servo_* helpers →
# backlash-model compatible.
# ─────────────────────────────────────────────────────────────────────────────


class SitStandCommand(UniformVelocityCommand):
    """Posture command: cmd = [sit_flag, 0, 0] with dwell-time resampling and a
    SLEWED internal target blend.

    sit_flag ∈ {0.0, 1.0}. Resampled by the command manager on the cfg's
    resampling_time_range (the dwell time in each posture) and on episode
    reset. cfg.sit_prob is the probability a resample commands SIT; with the
    reset-state mix this trains all four (start-state × command) combinations,
    including "hold what you're already doing".

    ``alpha`` (0 = STAND target, 1 = SIT target) slews toward the flag at a
    constant rate (full transition in cfg.ramp_s seconds) and is what the
    posture_* rewards track. THE anti-crash mechanism: with a binary target,
    arriving early pays the full goal-state jackpot for every step saved,
    while the linear speed-cap penalties integrate to a bounded excess-
    distance cost — an instant drop beat a 1 s descent by ~7×. With the
    slewed target, being AHEAD of the ramp scores ~0 on the height/composite
    stack (z far from the commanded height), so tracking the slow setpoint IS
    the argmax; the caps remain as backstops for overshoot/bounce. The OBS
    stays the raw binary flag (deployment: runtime writes 0/1; the trained
    response to a flip is the ~ramp_s glide).

    On episode reset, alpha is initialised from the robot's ACTUAL trunk
    height, not the flag — a seated spawn must not be dragged upward by a
    stand-initialised ramp (and vice versa).
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._sit_prob = float(getattr(cfg, "sit_prob", 0.5))
        self._ramp_s = float(getattr(cfg, "ramp_s", 2.0))
        self._sit_z = float(getattr(cfg, "sit_z", 0.060))
        self._stand_z = float(getattr(cfg, "stand_z", 0.115))
        self._env_ref = env
        self._alpha = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.vel_command_b

    @property
    def alpha(self) -> torch.Tensor:
        """Slewed target blend: 0 = STAND target, 1 = SIT target."""
        return self._alpha

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        if n == 0:
            return
        sit = (torch.rand(n, device=self.device) < self._sit_prob).float()
        self.vel_command_b[env_ids] = 0.0
        self.vel_command_b[env_ids, 0] = sit

    def _alpha_from_height(self) -> torch.Tensor:
        z = torch.nan_to_num(
            self.robot.data.root_link_pos_w[:, 2]
            - self._env_ref.scene.terrain.env_origins[:, 2],
            nan=self._stand_z,
        )
        return torch.clamp(
            (self._stand_z - z) / max(self._stand_z - self._sit_z, 1e-6), 0.0, 1.0
        )

    def compute(self, dt: float) -> None:
        super().compute(dt)
        # Episode-start re-init of the blend from the ACTUAL trunk height.
        # Done here (not in reset()) because the command manager resets BEFORE
        # the set_ground_state event teleports the robot, so reset() would read
        # the pre-teleport height. On the first compute of an episode the spawn
        # state is in place.
        fresh = self._env_ref.episode_length_buf <= 1
        if fresh.any():
            self._alpha = torch.where(fresh, self._alpha_from_height(), self._alpha)
        # Constant-rate slew of the target blend toward the commanded flag.
        step = dt / max(self._ramp_s, 1e-6)
        delta = self.vel_command_b[:, 0] - self._alpha
        self._alpha += torch.clamp(delta, -step, step)

    def _update_command(self) -> None:
        pass  # No heading controller / standing-env machinery.

    def _update_metrics(self) -> None:
        pass  # No velocity-tracking metrics for a posture flag.


@_dataclass(kw_only=True)
class SitStandCommandCfg(UniformVelocityCommandCfg):
    class_type: type = SitStandCommand
    # Probability that a resample commands SIT (vs STAND).
    sit_prob: float = 0.5
    # Seconds for the internal target blend to traverse STAND↔SIT in full.
    ramp_s: float = 2.0
    # Rest heights, used to initialise the blend from the spawn state.
    sit_z: float = 0.060
    stand_z: float = 0.115

    def build(self, env: ManagerBasedRlEnv) -> "SitStandCommand":
        return SitStandCommand(self, env)


def _posture_blend(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Target blend ∈ [0, 1] (0 = STAND, 1 = SIT) for the posture rewards.

    Uses the SitStandCommand's slewed ``alpha`` (the moving setpoint) when the
    term exposes it; falls back to the raw binary flag otherwise.
    """
    term = env.command_manager.get_term(command_name)
    alpha = getattr(term, "alpha", None)
    if alpha is not None:
        return alpha
    return env.command_manager.get_command(command_name)[:, 0]


def _posture_targets(
    env: ManagerBasedRlEnv,
    asset: Entity,
    command_name: str,
    sit_overrides: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(target blend, per-env joint target) for the commanded posture.

    STAND target = default_joint_pos (HOME); SIT target = HOME with the
    keyframe overrides applied; the SLEWED blend interpolates between them,
    so mid-ramp the rewarded pose folds in sync with the descending height.
    """
    blend = _posture_blend(env, command_name)
    stand_target = _servo_default_joint_pos(env, asset)
    sit_target = stand_target.clone()
    for idx, val in sit_overrides.items():
        sit_target[:, idx] = val
    target = stand_target + blend.unsqueeze(-1) * (sit_target - stand_target)
    return blend, target


def _posture_height(
    env: ManagerBasedRlEnv,
    command_name: str,
    sit_z: float,
    stand_z: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(slewed target trunk z, actual trunk z) per env."""
    blend = _posture_blend(env, command_name)
    target_z = stand_z + blend * (sit_z - stand_z)
    asset = env.scene["robot"]
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    return target_z, z


def posture_pose_match(
    env: ManagerBasedRlEnv,
    command_name: str,
    sit_overrides: dict,
    joint_indices: list,
    std: float = 0.5,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Gaussian pose-match against the commanded posture's target pose."""
    asset = env.scene[asset_cfg.name]
    _, target = _posture_targets(env, asset, command_name, sit_overrides)
    joint_pos = _servo_joint_pos(env, asset)[:, joint_indices]
    target = target[:, joint_indices]
    return torch.exp(-((joint_pos - target) / std) ** 2).mean(dim=-1)


def posture_pose_l1(
    env: ManagerBasedRlEnv,
    command_name: str,
    sit_overrides: dict,
    joint_indices: list,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """L1 companion to ``posture_pose_match`` (constant gradient to target)."""
    asset = env.scene[asset_cfg.name]
    _, target = _posture_targets(env, asset, command_name, sit_overrides)
    joint_pos = _servo_joint_pos(env, asset)[:, joint_indices]
    target = target[:, joint_indices]
    return -torch.abs(joint_pos - target).mean(dim=-1)


def posture_height_gaussian(
    env: ManagerBasedRlEnv,
    command_name: str,
    sit_z: float,
    stand_z: float,
    std: float = 0.02,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Gaussian on trunk z against the commanded posture's target height."""
    del asset_cfg  # trunk z read via _posture_height
    target_z, z = _posture_height(env, command_name, sit_z, stand_z)
    return torch.exp(-((z - target_z) / std) ** 2)


def posture_height_l1(
    env: ManagerBasedRlEnv,
    command_name: str,
    sit_z: float,
    stand_z: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """L1 companion to ``posture_height_gaussian`` — the transition driver.

    While the robot rests in the *wrong* posture this charges a constant
    per-step cost (~|Δz| = 55 mm), which is what makes "ignore the command"
    a net-negative strategy in both directions.
    """
    del asset_cfg
    target_z, z = _posture_height(env, command_name, sit_z, stand_z)
    return -torch.abs(z - target_z)


def posture_composite(
    env: ManagerBasedRlEnv,
    command_name: str,
    sit_overrides: dict,
    joint_indices: list,
    sit_z: float,
    stand_z: float,
    height_std: float = 0.03,
    upright_std: float = 0.40,
    pose_std: float = 0.40,
    head_std: float | None = None,
    head_command_name: str = "head_pose",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Multiplicative goal score vs the commanded posture (height·upright·pose
    [·head]).

    The posture-conditioned version of ``standing_composite_score``: a
    deficiency in any factor collapses the whole term, so partial-sum
    compromises (plank, flop, lean) never pay. Both rest states demand an
    upright trunk, so the upright factor is posture-independent.

    ``head_std`` (optional): adds a fourth factor on the neck/head joints vs
    the ``head_pose`` command (same error convention as head_pose_tracking).
    Without it the goal state is head-blind: the trained policy rested with
    the head dangling to the floor — trunk upright, legs in pose, z on target
    all held while the head hung, costing only the light tracking term. With
    the factor, "arrived" REQUIRES the head at its commanded pose, so head
    assist stays free mid-transition (composite is ≈0 there anyway) but must
    be retracted to collect the goal reward.
    """
    asset = env.scene[asset_cfg.name]
    _, target = _posture_targets(env, asset, command_name, sit_overrides)
    target_z, z = _posture_height(env, command_name, sit_z, stand_z)

    height_score = torch.exp(-((z - target_z) / height_std) ** 2)

    quat = asset.data.root_link_quat_w
    tilt_sq = 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    upright_score = torch.exp(-tilt_sq / (upright_std * upright_std))

    joint_pos = _servo_joint_pos(env, asset)[:, joint_indices]
    pose_err_sq = ((joint_pos - target[:, joint_indices]) ** 2).mean(dim=-1)
    pose_score = torch.exp(-pose_err_sq / (pose_std * pose_std))

    score = height_score * upright_score * pose_score

    if head_std is not None:
        if not hasattr(env, "_head_pose_neck_ids"):
            ids, _ = asset.find_joints_by_actuator_names(_NECK_JOINT_PATTERNS)
            env._head_pose_neck_ids = torch.tensor(ids, device=env.device, dtype=torch.long)
        neck_ids = env._head_pose_neck_ids
        head_cmd = env.command_manager.get_command(head_command_name)
        actual = asset.data.joint_pos[:, neck_ids] - asset.data.default_joint_pos[:, neck_ids]
        head_err_sq = ((actual - head_cmd) ** 2).mean(dim=-1)
        score = score * torch.exp(-head_err_sq / (head_std * head_std))

    return score


def posture_stillness(
    env: ManagerBasedRlEnv,
    command_name: str,
    sit_z: float,
    stand_z: float,
    band_full: float = 0.012,
    band_zero: float = 0.03,
    vel_std: float = 0.05,
    tilt_full_deg: float = 25.0,
    tilt_zero_deg: float = 60.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward trunk stillness while AT the commanded posture, upright.

    Generalizes ``seated_stillness`` to both rest states: exp(-(|v|/std)²)
    gated by a smoothstep on |z − commanded z| (full inside ``band_full``,
    zero beyond ``band_zero`` → inactive during transitions) and by trunk
    tilt (a tilted rest — back/face/side — earns nothing). Additionally gated
    on the target ramp being COMPLETE (|flag − alpha| small), so stillness
    never pays mid-transition. Makes "rest quietly, upright, at the commanded
    height" the peak of the stack.
    """
    asset = env.scene[asset_cfg.name]
    target_z, z = _posture_height(env, command_name, sit_z, stand_z)
    v = torch.nan_to_num(asset.data.root_link_lin_vel_w, nan=0.0).norm(dim=-1)

    flag = env.command_manager.get_command(command_name)[:, 0]
    blend = _posture_blend(env, command_name)
    ramp_done = ((flag - blend).abs() < 0.02).float()

    err = torch.abs(z - target_z)
    t = torch.clamp((band_zero - err) / max(band_zero - band_full, 1e-6), 0.0, 1.0)
    z_gate = t * t * (3.0 - 2.0 * t)

    quat = asset.data.root_link_quat_w
    cos_tilt = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    cos_full = math.cos(math.radians(tilt_full_deg))
    cos_zero = math.cos(math.radians(tilt_zero_deg))
    u = torch.clamp((cos_tilt - cos_zero) / max(cos_full - cos_zero, 1e-6), 0.0, 1.0)
    tilt_gate = u * u * (3.0 - 2.0 * u)

    return torch.exp(-((v / vel_std) ** 2)) * z_gate * tilt_gate * ramp_done


def posture_rise_bootstrap(
    env: ManagerBasedRlEnv,
    command_name: str,
    max_height: float,
    max_vz: float | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Upward-vz reward, active only when STAND is commanded and z < max_height.

    The standup-env lesson: destination-only rewards have zero gradient at
    zero motion, so "stay seated and eat the L1" is a local optimum — paying
    for the rise *motion* itself makes any attempt immediately positive.
    Gated off above ``max_height`` (set just ABOVE the stand target so the
    final cm still pays; gating at exactly STAND_Z parks the policy short).
    Zero whenever SIT is commanded, so it can never fight the descent.
    ``max_vz`` caps the rewarded speed (any rise ≥ the cap earns the same, so
    an explosive launch can't out-earn a gentle one).
    """
    asset = env.scene[asset_cfg.name]
    sit = env.command_manager.get_command(command_name)[:, 0]
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    vz = torch.nan_to_num(asset.data.root_link_lin_vel_w[:, 2], nan=0.0)
    return torch.clamp(vz, min=0.0, max=max_vz) * (z < max_height).float() * (1.0 - sit)


def trunk_upward_velocity_penalty(
    env: ManagerBasedRlEnv,
    max_up_vel: float = 0.08,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalty on upward trunk velocity beyond ``max_up_vel``.

    Mirror of ``trunk_downward_velocity_penalty`` for the rise: charges every
    step of a too-fast (violent) stand-up, so the explosive rise can't be
    amortised against arriving-standing reward. Zero at rest, for any rise
    slower than the cap, and for all downward motion. Introduce via
    curriculum AFTER the rise is discovered (attempt-tax lesson).
    """
    asset = env.scene[asset_cfg.name]
    vz = torch.nan_to_num(asset.data.root_link_lin_vel_w[:, 2], nan=0.0)
    return -torch.clamp(vz - max_up_vel, min=0.0)
# ==============================================================================
# Roulade (forward roll) task — episodic dynamic maneuver
# ==============================================================================
#
# Third attempt at the roulade. What the first two taught us:
#   • origin/roulade (phase-clock + time-windowed reward stages): plateaued
#     face-down at ~90° — time windows are keyframes-in-time, campable local
#     optima (the sit/standup lesson exactly). Also integrated -ω_y as forward
#     progress, which by this codebase's own convention (face-down = +90° pitch
#     = rotation about +y, see set_random_ground_state) is the WRONG SIGN — the
#     progress reward paid for backward rotation.
#   • origin/roulade later commits (keyframe imitation): same waypoint-camping
#     family, dropped per feedback-episodic-pose-landing.
#
# This design uses the proven episodic recipe instead:
#   • ONE dense progress signal: paid INCREMENTS of the max-so-far cumulative
#     forward rotation (potential-based — a camping policy earns zero/step, a
#     full roll earns exactly 2π worth no matter the path or speed).
#   • Landing rewards (composite product, upright, height, rise velocity) are
#     gated on ROLL COMPLETION (max rotation ≥ threshold) — state-based gates,
#     not clock-based. "Do nothing" earns nothing; standing at spawn earns
#     nothing; only rolling opens the standing-attractor annuity.
#   • Reverse curriculum via mid-roll spawns (the face-up partial-roll trick
#     that fixed back-recovery): a slice of episodes starts pitched 50°–185°
#     into the roll, tucked, optionally with forward angular momentum, and the
#     rotation accumulator is initialized to the spawn angle so the progress
#     accounting stays consistent.
#
# RUN-1 LESSON (2026-08): with unsupported rotation counting and uncapped
# paid rate, the optimal policy is a violent ballistic whip ("breakdance") —
# same 2π, finishes sooner, more discounted annuity. Doesn't transfer. Fixes:
#   • SUPPORT GATE: the accumulator only integrates while some robot geom
#     touches the terrain (robot_ground_contact sensor) — a real roulade never
#     leaves the ground; airborne rotation now earns nothing and cannot open
#     the completion gate.
#   • HEAD LATCH: the landing annuity additionally requires head-ground
#     contact to have occurred while accum was in the first-quadrant window —
#     "went over the head" is a requirement, not a 0.5-weight suggestion.
#   • PAID-RATE CAP: progress increments are capped at max_paid_rate; rotation
#     faster than the cap FORFEITS the excess (not deferred), so speed no
#     longer pays. An explicit overspeed penalty backs this up.
#
# Per-env state on the env object (created lazily, reset by
# reset_roulade_state):
#   env._roulade_accum      — supported-only integral of forward pitch rate (rad)
#   env._roulade_max        — max(accum) so far this episode (progress frontier)
#   env._roulade_paid       — frontier already paid out by roulade_progress
#   env._roulade_head_latch — True once the head touched ground mid-first-quadrant

# Forward-roll sign: face-down is +90° pitch = rotation about body +y
# (set_random_ground_state convention), so forward roll = POSITIVE body-frame
# ω_y. Verified empirically (see claude_experiments smoke test): a positive
# qvel about +y pitches the robot nose-down/forward and drives accum upward.
_ROULADE_FWD_SIGN = 1.0

# Sensor names read by the accumulator update (must match the env cfg).
_ROULADE_SUPPORT_SENSOR = "robot_ground_contact"
_ROULADE_HEAD_SENSOR = "head_ground_contact"

# Head-latch window: head-ground contact while accum is inside this window
# marks the episode as a genuine over-the-head roll. In a real roulade the
# head plants at ~60–120° of body rotation; the window is generous around it.
_HEAD_LATCH_LO = math.radians(20.0)
_HEAD_LATCH_HI = math.radians(170.0)

# Head-top axis in jaw_soft's LOCAL frame (measured empirically 2026-08-13:
# world-up expressed in jaw_soft's frame with the robot settled at HOME).
# The latch requires this axis to point DOWN at contact — "the flat top of
# the head on the floor", not the face or the side of the shell (run-5 fix:
# the run-4 policy rolled over the shoulder, which still touched jaw_soft).
_HEAD_TOP_AXIS = (0.882, 0.0, 0.471)
# dot(top_axis_world, -z) threshold. Measured landmarks (trunk pitched 110°):
# passive face-plant (neck at HOME) reads +0.6, full chin-tuck (neck_pitch −1,
# head_pitch +1) reads −0.99 — 0.3 accepts partial tucks while staying far
# from any face/side contact.
_HEAD_TOP_DOWN_MIN = 0.3

# Sagittal flatness gate on the accumulator (run-5): in a clean forward roll
# the body's LATERAL axis stays horizontal the whole way — its world-z
# component is 2(q_y·q_z + q_w·q_x) ≈ 0 for ANY amount of pure pitch, and
# grows toward ±1 as the roll goes over the shoulder instead. Full rotation
# credit while the lateral axis is within ~30° of horizontal, zero beyond
# ~60°: a side roll does not count as rotation, earns no progress, and never
# opens the landing gate.
_FLAT_FULL = 0.5    # |lateral_axis_z| = sin(30°): full credit below
_FLAT_ZERO = 0.866  # sin(60°): zero credit above


def _lateral_axis_z(quat: torch.Tensor) -> torch.Tensor:
    """World-z component of the body's lateral (y) axis. 0 = flat/sagittal."""
    return 2.0 * (quat[:, 2] * quat[:, 3] + quat[:, 0] * quat[:, 1])


def _head_top_down(env: ManagerBasedRlEnv, asset: Entity) -> torch.Tensor:
    """True where the head-top axis points at the floor (dot with -z > min)."""
    if not hasattr(env, "_roulade_head_body_id"):
        ids, _ = asset.find_bodies("jaw_soft")
        env._roulade_head_body_id = ids[0]
    q = asset.data.body_link_quat_w[:, env._roulade_head_body_id]
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    a, b, c = _HEAD_TOP_AXIS
    # z-component of R(q) @ axis_local
    axis_world_z = (
        2.0 * (x * z - w * y) * a + 2.0 * (y * z + w * x) * b + (1.0 - 2.0 * (x * x + y * y)) * c
    )
    return axis_world_z < -_HEAD_TOP_DOWN_MIN


def headstand_contact(
    env: ManagerBasedRlEnv,
    head_sensor_name: str = "head_ground_contact",
    feet_sensor_name: str = "feet_ground_contact",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Return a binary head-supported pose gate.

    A valid MicroDuck headstand is intentionally defined as a controlled
    head-supported freeze: the flat top of ``jaw_soft`` points down, the head
    touches the terrain, and neither foot is touching.  Keeping this contact
    definition explicit prevents the policy from collecting credit for a
    face-plant or a three-point crouch.
    """
    asset = env.scene[asset_cfg.name]
    head_contact = _sensor_any_contact(env, head_sensor_name)
    feet_contact = _sensor_any_contact(env, feet_sensor_name)
    if head_contact is None:
        return torch.zeros(env.num_envs, device=env.device)
    if feet_contact is None:
        feet_contact = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return (
        head_contact
        & ~feet_contact
        & _head_top_down(env, asset)
    ).float()


def headstand_hold(
    env: ManagerBasedRlEnv,
    head_sensor_name: str = "head_ground_contact",
    feet_sensor_name: str = "feet_ground_contact",
    lin_vel_std: float = 0.04,
    ang_vel_std: float = 0.45,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward quiet head-supported balance, not merely making contact."""
    asset = env.scene[asset_cfg.name]
    gate = headstand_contact(
        env,
        head_sensor_name=head_sensor_name,
        feet_sensor_name=feet_sensor_name,
        asset_cfg=asset_cfg,
    )
    lin_vel = torch.nan_to_num(asset.data.root_link_lin_vel_w, nan=0.0).norm(dim=-1)
    ang_vel = torch.nan_to_num(asset.data.root_link_ang_vel_b, nan=0.0).norm(dim=-1)
    still = torch.exp(-((lin_vel / lin_vel_std) ** 2) - ((ang_vel / ang_vel_std) ** 2))
    return gate * still


def headstand_height(
    env: ManagerBasedRlEnv,
    target_height: float,
    std: float = 0.02,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Shape the head-supported pose toward its measured root height."""
    asset = env.scene[asset_cfg.name]
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    return headstand_contact(env, asset_cfg=asset_cfg) * torch.exp(-((z - target_height) / std) ** 2)


def _backflip_state(env: ManagerBasedRlEnv) -> tuple:
    if not hasattr(env, "_backflip_accum"):
        z = torch.zeros(env.num_envs, device=env.device)
        env._backflip_accum = z.clone()
        env._backflip_max = z.clone()
        env._backflip_paid = z.clone()
        env._backflip_foot_latch = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        env._backflip_last_update_step = -1
    return env._backflip_accum, env._backflip_max, env._backflip_paid


def _update_backflip_accum(env: ManagerBasedRlEnv, asset: Entity) -> None:
    """Integrate backward body rotation during the aerial part of a flip."""
    accum, max_accum, _ = _backflip_state(env)
    step = int(env.common_step_counter)
    if step == env._backflip_last_update_step:
        return

    # Backward rotation is -body-y for the MicroDuck convention. Allow feet to
    # remain in contact during takeoff/landing, but never count a head/trunk
    # supported roll as aerial rotation.
    omega_back = torch.nan_to_num(-asset.data.root_link_ang_vel_b[:, 1], nan=0.0)
    support = _sensor_any_contact(env, "robot_ground_contact")
    feet = _sensor_any_contact(env, "feet_ground_contact")
    if support is None or feet is None:
        aerial_or_feet = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    else:
        aerial_or_feet = ~support | feet
    delta = torch.clamp(omega_back, min=0.0) * env.step_dt * aerial_or_feet.float()
    env._backflip_accum = accum + delta
    env._backflip_max = torch.maximum(max_accum, env._backflip_accum)

    if feet is not None:
        completion = env._backflip_max > math.radians(300.0)
        env._backflip_foot_latch = env._backflip_foot_latch | (feet & completion)
    env._backflip_last_update_step = step


def reset_backflip_state(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    late_spawn_prob: float = 0.25,
    late_angle_range: tuple = (math.radians(25.0), math.radians(320.0)),
    late_height_range: tuple = (0.10, 0.18),
    late_omega_range: tuple = (0.0, 4.0),
    standing_height_range: tuple = (0.11, 0.12),
    tuck_overrides: Optional[dict] = None,
    joint_noise_std: float = 0.02,
) -> None:
    """Reset standing or late-phase backward-flip states.

    The late-phase bucket is a reverse curriculum, not a claim that a pure
    from-stand PPO run will discover an aerial 360-degree flip immediately.
    It gives the landing controller on-policy data while the takeoff policy is
    still being learned.
    """
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device, dtype=torch.long)
    asset = env.scene[asset_cfg.name]
    accum, max_accum, paid = _backflip_state(env)
    num = len(env_ids)
    late = torch.rand(num, device=env.device) < late_spawn_prob

    yaw = torch.rand(num, device=env.device) * 2 * np.pi - np.pi
    angle = torch.rand(num, device=env.device) * (late_angle_range[1] - late_angle_range[0]) + late_angle_range[0]
    angle = torch.where(late, angle, torch.zeros_like(angle))
    pitch = -angle
    roll = (torch.rand(num, device=env.device) * 2 - 1) * math.radians(3.0)
    cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    quat = torch.stack(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dim=1,
    )

    standing_z = torch.rand(num, device=env.device) * (standing_height_range[1] - standing_height_range[0]) + standing_height_range[0]
    late_z = torch.rand(num, device=env.device) * (late_height_range[1] - late_height_range[0]) + late_height_range[0]
    env.sim.data.qpos[env_ids, 2] = torch.where(late, late_z, standing_z)
    env.sim.data.qpos[env_ids, 3:7] = quat
    env.sim.data.qvel[env_ids, :6] = 0.0

    servo_ids = _servo_joint_ids(env, asset)
    late_ids = env_ids[late]
    if len(late_ids) > 0 and tuck_overrides:
        for joint_index, target in tuck_overrides.items():
            col = 7 + servo_ids[joint_index]
            env.sim.data.qpos[late_ids, col] = target
    if len(late_ids) > 0 and joint_noise_std > 0.0:
        cols = torch.tensor([7 + j for j in servo_ids], device=env.device, dtype=torch.long)
        noise = torch.randn(len(late_ids), len(cols), device=env.device) * joint_noise_std
        env.sim.data.qpos[late_ids.unsqueeze(1), cols.unsqueeze(0)] += noise
    if len(late_ids) > 0:
        omega = torch.rand(len(late_ids), device=env.device) * (late_omega_range[1] - late_omega_range[0]) + late_omega_range[0]
        env.sim.data.qvel[late_ids, 4] = -omega

    accum[env_ids] = angle
    max_accum[env_ids] = angle
    paid[env_ids] = angle
    env._backflip_foot_latch[env_ids] = False


def backflip_progress(
    env: ManagerBasedRlEnv,
    target_angle: float = 2 * math.pi,
    max_paid_rate: float = 10.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Pay only new backward-rotation frontier, with an aerial support gate."""
    asset = env.scene[asset_cfg.name]
    _update_backflip_accum(env, asset)
    _, max_accum, paid = _backflip_state(env)
    new_paid = torch.clamp(max_accum, max=target_angle)
    delta = torch.clamp(new_paid - torch.clamp(paid, max=target_angle), min=0.0)
    delta = torch.clamp(delta, max=max_paid_rate * env.step_dt)
    env._backflip_paid = torch.maximum(paid, new_paid)
    return delta / (env.step_dt * target_angle)


def backflip_landing(
    env: ManagerBasedRlEnv,
    target_height: float,
    height_std: float = 0.04,
    upright_std: float = 0.4,
    pose_std: float = 0.4,
    joint_indices: list = None,
    gate_lo: float = math.radians(300.0),
    gate_hi: float = math.radians(355.0),
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward a real feet-first landing after a near-complete backflip."""
    asset = env.scene[asset_cfg.name]
    _update_backflip_accum(env, asset)
    _, max_accum, _ = _backflip_state(env)
    t = torch.clamp((max_accum - gate_lo) / max(gate_hi - gate_lo, 1e-6), 0.0, 1.0)
    gate = t * t * (3.0 - 2.0 * t) * env._backflip_foot_latch.float()
    score = standing_composite_score(
        env,
        target_height=target_height,
        height_std=height_std,
        upright_std=upright_std,
        pose_std=pose_std,
        joint_indices=joint_indices or list(range(14)),
        asset_cfg=asset_cfg,
    )
    return score * gate


def _sensor_any_contact(env: ManagerBasedRlEnv, name: str) -> torch.Tensor | None:
    if name not in env.scene.sensors:
        return None
    found = env.scene.sensors[name].data.found
    return (found.view(found.shape[0], -1) > 0).any(dim=-1)


def _roulade_state(env: ManagerBasedRlEnv) -> tuple:
    if not hasattr(env, "_roulade_accum"):
        z = torch.zeros(env.num_envs, device=env.device)
        env._roulade_accum = z.clone()
        env._roulade_max = z.clone()
        env._roulade_paid = z.clone()
        env._roulade_roll_direction = torch.ones(env.num_envs, device=env.device)
        env._roulade_head_latch = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._roulade_last_update_step = -1
    elif not hasattr(env, "_roulade_roll_direction"):
        env._roulade_roll_direction = torch.ones(env.num_envs, device=env.device)
    return env._roulade_accum, env._roulade_max, env._roulade_paid


def _update_roulade_accum(env: ManagerBasedRlEnv, asset: Entity) -> None:
    """Integrate forward pitch rate into the per-env rotation accumulator.

    Step-guarded so that multiple reward terms reading the accumulator in the
    same control step don't double-integrate. The frontier (max) only moves
    forward; backward rocking (wind-up) neither pays nor un-pays.

    SUPPORT GATE (run-1 fix): rotation is integrated only while the robot
    touches the terrain — a roulade is a supported motion; ballistic flips
    accumulate nothing, so they neither get paid nor open the completion gate.

    Also latches env._roulade_head_latch when the head touches the ground
    while accum is inside the first-quadrant window — the landing annuity
    requires this, making "over the head" a hard requirement of the task.
    """
    _roulade_state(env)
    step = int(env.common_step_counter)
    if step != env._roulade_last_update_step:
        omega_fwd = (
            _ROULADE_FWD_SIGN
            * env._roulade_roll_direction
            * asset.data.root_link_ang_vel_b[:, 1]
        )
        delta = torch.nan_to_num(omega_fwd, nan=0.0) * env.step_dt
        supported = _sensor_any_contact(env, _ROULADE_SUPPORT_SENSOR)
        if supported is not None:
            delta = delta * supported.float()
        # Sagittal flatness gate (run-5): side/shoulder rolls don't count.
        y_z = torch.nan_to_num(_lateral_axis_z(asset.data.root_link_quat_w), nan=1.0).abs()
        t = torch.clamp((_FLAT_ZERO - y_z) / (_FLAT_ZERO - _FLAT_FULL), 0.0, 1.0)
        delta = delta * (t * t * (3.0 - 2.0 * t))
        env._roulade_accum = env._roulade_accum + delta
        env._roulade_max = torch.maximum(env._roulade_max, env._roulade_accum)

        head_contact = _sensor_any_contact(env, _ROULADE_HEAD_SENSOR)
        if head_contact is not None:
            in_window = (env._roulade_accum > _HEAD_LATCH_LO) & (
                env._roulade_accum < _HEAD_LATCH_HI
            )
            # Run-5: contact must be with the FLAT TOP of the head (top axis
            # pointing at the floor) — face/side shell contacts don't latch.
            env._roulade_head_latch = env._roulade_head_latch | (
                head_contact & in_window & _head_top_down(env, asset)
            )
        env._roulade_last_update_step = step


def _roulade_completion_gate(
    env: ManagerBasedRlEnv,
    gate_lo: float,
    gate_hi: float,
    require_head: bool = False,
) -> torch.Tensor:
    """Smoothstep on the progress frontier: 0 below gate_lo rad, 1 above gate_hi.

    State-based replacement for the old phase-clock landing window — it can
    only be opened by actually rotating (while SUPPORTED — the accumulator is
    contact-gated), so neither pre-roll standing nor a ballistic flip collects.
    With require_head=True the gate additionally requires the head latch —
    the episode must have rolled over the head to unlock the landing annuity.
    """
    _, max_accum, _ = _roulade_state(env)
    t = torch.clamp((max_accum - gate_lo) / max(gate_hi - gate_lo, 1e-6), 0.0, 1.0)
    gate = t * t * (3.0 - 2.0 * t)
    if require_head:
        gate = gate * env._roulade_head_latch.float()
    return gate


def reset_roulade_state(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    standing_prob: float = 0.5,
    midroll_prob: float = 0.5,
    standing_z_min: float = 0.11,
    standing_z_max: float = 0.12,
    standing_tilt_max: float = 0.0,
    yaw_range: tuple = (-math.pi, math.pi),
    forward_vel_range: tuple = (0.0, 0.0),
    midroll_pitch_min: float = math.radians(50.0),
    midroll_pitch_max: float = math.radians(185.0),
    midroll_z_min: float = 0.05,
    midroll_z_max: float = 0.10,
    midroll_omega_range: tuple = (0.0, 0.0),
    midroll_forward_vel_range: tuple = (0.0, 0.0),
    tuck_overrides: Optional[dict] = None,
    tuck_factor_range: tuple = (0.3, 1.0),
    joint_noise_std: float = 0.0,
    roll_direction: float = 1.0,
):
    """Reset to a standing start or a mid-roll state (reverse curriculum).

    Standing bucket: upright (±standing_tilt_max pitch/roll noise), sampled yaw,
    HOME joints (left from reset_robot_joints), z in [standing_z_min, _max].
    ``forward_vel_range`` is the élan hook: a per-env forward base velocity
    (body x, mapped to world through the spawn yaw) sampled uniformly — 0 for
    a standstill roll, widen it later to train rolls out of a walk.

    Mid-roll bucket: pitched ``midroll_pitch_min..max`` into the roll (90° =
    on the head, 180° = on the back), random yaw, legs lerped HOME→tuck by a
    per-env factor in ``tuck_factor_range``, z in [midroll_z_min, _max],
    optional forward angular momentum from ``midroll_omega_range`` and
    direction-matched translation from ``midroll_forward_vel_range``. The
    rotation accumulator is initialized to the spawn pitch so progress
    accounting (and the completion gates) stay consistent: a 170° spawn only
    gets paid for the remaining ~190°.
    """
    if env_ids is None or len(env_ids) == 0:
        return
    if roll_direction not in (-1.0, 1.0):
        raise ValueError("roll_direction must be -1.0 or 1.0")
    env_ids = env_ids.to(env.device, dtype=torch.long)
    num = len(env_ids)
    asset: Entity = env.scene[asset_cfg.name]
    accum, max_accum, paid = _roulade_state(env)

    total = standing_prob + midroll_prob
    is_mid = torch.rand(num, device=env.device) < (midroll_prob / max(total, 1e-6))

    yaw = (
        torch.rand(num, device=env.device) * (yaw_range[1] - yaw_range[0])
        + yaw_range[0]
    )
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)

    # Pitch per bucket: small noise for standing, mid-roll angle otherwise.
    pitch = (torch.rand(num, device=env.device) * 2 - 1) * standing_tilt_max
    mid_pitch = (
        torch.rand(num, device=env.device) * (midroll_pitch_max - midroll_pitch_min)
        + midroll_pitch_min
    )
    pitch = torch.where(is_mid, roll_direction * mid_pitch, pitch)
    roll = (torch.rand(num, device=env.device) * 2 - 1) * max(standing_tilt_max, math.radians(5.0))

    cp = torch.cos(pitch * 0.5); sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5); sr = torch.sin(roll * 0.5)
    # ZYX intrinsic Euler → quaternion (yaw * pitch * roll), as in
    # set_random_ground_state.
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    quat = torch.stack([qw, qx, qy, qz], dim=1)

    z_stand = torch.rand(num, device=env.device) * (standing_z_max - standing_z_min) + standing_z_min
    z_mid = torch.rand(num, device=env.device) * (midroll_z_max - midroll_z_min) + midroll_z_min
    new_z = torch.where(is_mid, z_mid, z_stand)

    env.sim.data.qpos[env_ids, 2] = new_z
    env.sim.data.qpos[env_ids, 3:7] = quat
    env.sim.data.qvel[env_ids, :6] = 0.0

    servo_ids = _servo_joint_ids(env, asset)

    # Mid-roll joints: lerp HOME → tuck on the overridden joints, noise on all
    # servo joints (passive_* backlash hinges must stay at 0).
    mid_env_ids = env_ids[is_mid]
    if len(mid_env_ids) > 0 and tuck_overrides:
        u = (
            torch.rand(len(mid_env_ids), device=env.device)
            * (tuck_factor_range[1] - tuck_factor_range[0])
            + tuck_factor_range[0]
        )
        for jnt_idx, angle in tuck_overrides.items():
            col = 7 + servo_ids[jnt_idx]
            home = env.sim.data.qpos[mid_env_ids, col]
            env.sim.data.qpos[mid_env_ids, col] = home + u * (angle - home)
    if len(mid_env_ids) > 0 and joint_noise_std > 0.0:
        cols = torch.tensor([7 + j for j in servo_ids], device=env.device, dtype=torch.long)
        noise = torch.randn(len(mid_env_ids), len(cols), device=env.device) * joint_noise_std
        env.sim.data.qpos[mid_env_ids.unsqueeze(1), cols.unsqueeze(0)] += noise

    # Mid-roll forward angular momentum: rotation about body +y. MuJoCo free
    # joint qvel[3:6] is the angular velocity in the BODY frame, so [0, ω, 0]
    # is the forward-roll axis regardless of spawn yaw (verified in the smoke
    # test — a yawed spawn still rolls straight ahead in its own frame).
    if len(mid_env_ids) > 0 and midroll_omega_range[1] > 0.0:
        omega = (
            torch.rand(len(mid_env_ids), device=env.device)
            * (midroll_omega_range[1] - midroll_omega_range[0])
            + midroll_omega_range[0]
        )
        env.sim.data.qvel[mid_env_ids, 4] = (
            roll_direction * _ROULADE_FWD_SIGN * omega
        )

    # Reverse-skill discovery needs examples where opposite signed rotation
    # and opposite-body translation occur together. This is world velocity
    # opposite body +x for a backroll, which moves canonical racers toward
    # their fixed +x finish while they face -x.
    if len(mid_env_ids) > 0 and midroll_forward_vel_range[1] > 0.0:
        mid_vx = (
            torch.rand(len(mid_env_ids), device=env.device)
            * (midroll_forward_vel_range[1] - midroll_forward_vel_range[0])
            + midroll_forward_vel_range[0]
        )
        yaw_m = yaw[is_mid]
        env.sim.data.qvel[mid_env_ids, 0] = (
            roll_direction * mid_vx * torch.cos(yaw_m)
        )
        env.sim.data.qvel[mid_env_ids, 1] = (
            roll_direction * mid_vx * torch.sin(yaw_m)
        )

    # Élan hook: forward base velocity for STANDING spawns, body x → world xy
    # through the spawn yaw. (0, 0) = standstill start, disabled.
    stand_env_ids = env_ids[~is_mid]
    if len(stand_env_ids) > 0 and forward_vel_range[1] > 0.0:
        vx = (
            torch.rand(len(stand_env_ids), device=env.device)
            * (forward_vel_range[1] - forward_vel_range[0])
            + forward_vel_range[0]
        )
        yaw_s = yaw[~is_mid]
        env.sim.data.qvel[stand_env_ids, 0] = (
            roll_direction * vx * torch.cos(yaw_s)
        )
        env.sim.data.qvel[stand_env_ids, 1] = (
            roll_direction * vx * torch.sin(yaw_s)
        )

    # Progress accounting: standing starts at 0, mid-roll at the spawn pitch.
    spawn_angle = torch.where(is_mid, mid_pitch, torch.zeros_like(mid_pitch))
    accum[env_ids] = spawn_angle
    max_accum[env_ids] = spawn_angle
    paid[env_ids] = spawn_angle
    env._roulade_roll_direction[env_ids] = roll_direction
    # Head latch: mid-roll spawns are considered already past the head phase
    # (the reverse curriculum teaches roll COMPLETION; requiring a latch they
    # never had the chance to earn would keep their landing gate shut forever).
    # Standing spawns must earn it by actually rolling over the head.
    env._roulade_head_latch[env_ids] = is_mid


def roulade_progress(
    env: ManagerBasedRlEnv,
    target_angle: float = 2 * math.pi,
    max_paid_rate: float = 3.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Pay increments of the progress frontier, up to one full roll.

    reward = Δ(min(max_accum, target)) / (step_dt · target), CAPPED at
    max_paid_rate rad/s of paid rotation. Nothing to farm by camping
    face-down (0/step), rocking below the frontier (0/step), or spinning past
    2π (clamped). The accumulator is support-gated, so airborne rotation pays
    nothing either.

    max_paid_rate (run-1 fix): rotation faster than the cap FORFEITS the
    excess — the paid pointer still jumps to the frontier, it just pays the
    capped amount. A violent whip therefore collects LESS total progress
    reward than a controlled ≤cap roll, instead of the same total sooner.
    """
    asset: Entity = env.scene[asset_cfg.name]
    _update_roulade_accum(env, asset)
    _, max_accum, paid = _roulade_state(env)
    new_paid = torch.clamp(max_accum, max=target_angle)
    delta = torch.clamp(new_paid - torch.clamp(paid, max=target_angle), min=0.0)
    delta = torch.clamp(delta, max=max_paid_rate * env.step_dt)
    env._roulade_paid = torch.maximum(paid, new_paid)
    return delta / (env.step_dt * target_angle)


def roulade_head_pivot(
    env: ManagerBasedRlEnv,
    sensor_name: str = "head_ground_contact",
    angle_lo: float = math.radians(30.0),
    angle_hi: float = math.radians(240.0),
    rate_norm: float = 2.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward head-ground contact while rotating forward mid-roll.

    contact × window(accum ∈ [angle_lo, angle_hi]) × clamp(ω_fwd/rate_norm, 0, 1)
    × (0.3 + 0.7·top_down).
    The rate factor is the anti-camping guard: a face-planted robot resting its
    head on the floor has ω_fwd ≈ 0 and earns nothing — the term only pays for
    pivoting OVER the head. The top_down factor (run-5) aligns this dense
    shaping with the latch: any head contact mid-roll pays 30%, contact on the
    FLAT TOP (chin tucked) pays full — the gradient that teaches the tuck.
    """
    asset: Entity = env.scene[asset_cfg.name]
    _update_roulade_accum(env, asset)
    accum, _, _ = _roulade_state(env)

    if sensor_name not in env.scene.sensors:
        return torch.zeros(env.num_envs, device=env.device)
    found = env.scene.sensors[sensor_name].data.found
    contact = (found.view(found.shape[0], -1) > 0).any(dim=-1).float()

    in_window = ((accum > angle_lo) & (accum < angle_hi)).float()
    omega_fwd = (
        _ROULADE_FWD_SIGN
        * env._roulade_roll_direction
        * asset.data.root_link_ang_vel_b[:, 1]
    )
    rate = torch.clamp(torch.nan_to_num(omega_fwd, nan=0.0) / rate_norm, 0.0, 1.0)
    top = 0.3 + 0.7 * _head_top_down(env, asset).float()
    return contact * in_window * rate * top


def roulade_landing_composite(
    env: ManagerBasedRlEnv,
    target_height: float,
    height_std: float,
    upright_std: float,
    pose_std: float,
    joint_indices: list,
    gate_lo: float = math.radians(260.0),
    gate_hi: float = math.radians(330.0),
    target_overrides: Optional[dict] = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """standing_composite_score × completion gate.

    The big annuity: once the roll is (nearly) complete, every step spent
    standing at HOME pose pays — finishing on the feet and staying there
    dominates every partial outcome. Zero before gate_lo of rotation, so the
    standing spawn cannot farm it by doing nothing.
    """
    asset: Entity = env.scene[asset_cfg.name]
    _update_roulade_accum(env, asset)
    score = standing_composite_score(
        env,
        target_height=target_height,
        height_std=height_std,
        upright_std=upright_std,
        pose_std=pose_std,
        joint_indices=joint_indices,
        target_overrides=target_overrides,
        asset_cfg=asset_cfg,
    )
    return score * _roulade_completion_gate(env, gate_lo, gate_hi, require_head=True)


def roulade_upright_after_roll(
    env: ManagerBasedRlEnv,
    gate_lo: float = math.radians(260.0),
    gate_hi: float = math.radians(330.0),
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Linear cos(tilt) × completion gate — bootstrap pull toward vertical.

    Gradient from ANY orientation (the composite is near-zero far from the
    goal), but only after the roll: before gate_lo it is exactly zero, so it
    cannot oppose the flip the way the old always-on upright term did.
    """
    asset: Entity = env.scene[asset_cfg.name]
    _update_roulade_accum(env, asset)
    quat = asset.data.root_link_quat_w
    upright = 1.0 - 2.0 * (quat[:, 1].pow(2) + quat[:, 2].pow(2))
    return torch.clamp(upright, min=0.0) * _roulade_completion_gate(
        env, gate_lo, gate_hi, require_head=True
    )


def roulade_height_after_roll(
    env: ManagerBasedRlEnv,
    target_height: float,
    std: float = 0.04,
    gate_lo: float = math.radians(260.0),
    gate_hi: float = math.radians(330.0),
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Broad height Gaussian × completion gate — pull up to standing height."""
    asset: Entity = env.scene[asset_cfg.name]
    _update_roulade_accum(env, asset)
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    g = torch.exp(-((z - target_height) / std) ** 2)
    return g * _roulade_completion_gate(env, gate_lo, gate_hi, require_head=True)


def roulade_landing_sharp(
    env: ManagerBasedRlEnv,
    target_height: float,
    height_std: float = 0.015,
    upright_std: float = 0.3,
    gate_lo: float = math.radians(260.0),
    gate_hi: float = math.radians(330.0),
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Tight-std upright × height Gaussians × completion gate — the last mile.

    Run-4 fix for the 27°-lean / 1-cm-crouch end basin: the broad landing
    composite (upright_std 0.40) scores ~0.5 at that pose, so the policy
    parks there. This is standup's two-layer lesson — the broad layers reach,
    the sharp layers finish. At 27° tilt this term scores ~0.1 (real
    gradient); at vertical it pays ~1.
    """
    asset: Entity = env.scene[asset_cfg.name]
    _update_roulade_accum(env, asset)
    quat = asset.data.root_link_quat_w
    tilt_sq = 2.0 * (quat[:, 1].pow(2) + quat[:, 2].pow(2))
    upright_g = torch.exp(-tilt_sq / (upright_std * upright_std))
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    height_g = torch.exp(-((z - target_height) / height_std) ** 2)
    gate = _roulade_completion_gate(env, gate_lo, gate_hi, require_head=True)
    return upright_g * height_g * gate


def roulade_stand_tax(
    env: ManagerBasedRlEnv,
    target_height: float,
    gate_lo: float = math.radians(260.0),
    gate_hi: float = math.radians(330.0),
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """SELF-NEGATING height L1 below target, active only after roll completion.

    Returns −max(0, target − z) × completion_gate — use a POSITIVE weight
    (penalty sign convention). The run-3 fix for post-roll crumple-camping:
    the gated landing rewards made standing better than lying in a heap, but
    the heap itself was FREE — with only positive gated terms, "stay crumpled"
    collects ≈0/step, a comfortable basin (the standup static-sit lesson:
    the basin must be net NEGATIVE to force the rise). The gate keeps the
    roll itself untaxed, and requires the head latch so a no-roll episode
    can't be punished into weird avoidance behaviors.
    """
    asset: Entity = env.scene[asset_cfg.name]
    _update_roulade_accum(env, asset)
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    shortfall = torch.clamp(target_height - z, min=0.0)
    return -shortfall * _roulade_completion_gate(env, gate_lo, gate_hi, require_head=True)


def roulade_rise_velocity(
    env: ManagerBasedRlEnv,
    max_height: float = 0.125,
    gate_lo: float = math.radians(180.0),
    gate_hi: float = math.radians(260.0),
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """com_upward_velocity × late-roll gate — bootstrap the exit rise.

    The second half of a roulade (supine → sitting-up → standing) is the
    face-up recovery problem, and the standup env proved end-state rewards
    alone have zero gradient at zero motion there: pay for rising vz directly.
    Gated to open from ~180° (on the back) so pre-roll bobbing earns nothing,
    and gated off above max_height so it can't be farmed by hopping.
    """
    asset: Entity = env.scene[asset_cfg.name]
    _update_roulade_accum(env, asset)
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    vz = torch.nan_to_num(asset.data.root_link_lin_vel_w[:, 2], nan=0.0)
    reward = torch.clamp(vz, min=0.0) * (z < max_height).float()
    return reward * _roulade_completion_gate(env, gate_lo, gate_hi, require_head=True)


def roulade_overspeed_penalty(
    env: ManagerBasedRlEnv,
    omega_max: float = 4.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """max(0, |ω_y| − omega_max)² — quadratic tax on whip-speed rotation.

    Positive quantity; use a negative weight. Complements the paid-rate cap
    in roulade_progress: the cap removes the INCENTIVE to rotate faster than
    ~3 rad/s, this adds an explicit COST above omega_max, so "violent" is
    strictly worse than "controlled" rather than merely not-better. A
    controlled full roll (~2–3 rad/s average) never touches it.
    """
    asset: Entity = env.scene[asset_cfg.name]
    omega_y = torch.nan_to_num(asset.data.root_link_ang_vel_b[:, 1], nan=0.0)
    excess = torch.clamp(omega_y.abs() - omega_max, min=0.0)
    return excess.pow(2)


def roulade_flatness_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """(lateral-axis world-z)² — dense gradient toward a sagittal roll.

    Positive quantity; use a negative weight. Zero when standing, zero
    through an arbitrarily deep CLEAN forward roll (pure pitch keeps the
    lateral axis horizontal), up to 1 when tipped fully onto a shoulder.
    The accumulator's flatness gate makes side rolls unprofitable; this term
    adds the per-step gradient that steers back toward the plane.
    """
    asset: Entity = env.scene[asset_cfg.name]
    return torch.nan_to_num(_lateral_axis_z(asset.data.root_link_quat_w), nan=0.0).pow(2)


def roulade_sagittal_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Rotation out of the sagittal plane: body-frame ω_x² + ω_z² (positive;
    use a negative weight). ω_y is the roll axis and stays free."""
    asset: Entity = env.scene[asset_cfg.name]
    omega_b = asset.data.root_link_ang_vel_b
    return torch.nan_to_num(omega_b[:, 0].pow(2) + omega_b[:, 2].pow(2), nan=0.0)


def roulade_lateral_velocity_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Body-frame lateral (y) linear velocity² — keeps the roll straight."""
    asset: Entity = env.scene[asset_cfg.name]
    return torch.nan_to_num(asset.data.root_link_lin_vel_b[:, 1].pow(2), nan=0.0)


# ============================================================================
# Grounded backroll task: one supported reverse roulade and feet landing
# ============================================================================
_BACKROLL_TRUNK_SENSOR = "trunk_ground_contact"
_BACKROLL_LEFT_FOOT_SENSOR = "left_foot_ground_contact"
_BACKROLL_RIGHT_FOOT_SENSOR = "right_foot_ground_contact"
_BACKROLL_TRUNK_LATCH_LO = math.radians(30.0)
_BACKROLL_TRUNK_LATCH_HI = math.radians(260.0)
_BACKROLL_HEAD_LATCH_LO = math.radians(100.0)
_BACKROLL_HEAD_LATCH_HI = math.radians(300.0)
_BACKROLL_LANDING_ANGLE = math.radians(350.0)
_BACKROLL_POTENTIAL_ANGLE = math.radians(300.0)
_BACKROLL_UPRIGHT_COS = math.cos(math.radians(20.0))
_BACKROLL_STALL_TILT_COS = math.cos(math.radians(60.0))
_BACKROLL_MIN_LANDING_HEIGHT = 0.10
_BACKROLL_MAX_LANDING_ANG_VEL = 2.0
_BACKROLL_MAX_AIR_SECONDS = 0.08
_BACKROLL_LANDING_HOLD_SECONDS = 0.25
_BACKROLL_INVALID_STALL_SECONDS = 0.50
_BACKROLL_REPEAT_LANDING_HOLD_SECONDS = 0.10
_BACKROLL_REPEAT_PRE_EXIT_STALL_SECONDS = 1.00
_BACKROLL_REPEAT_LANDING_TIMEOUT_SECONDS = 2.00
_BACKROLL_REPEAT_MAX_LANDING_ANG_VEL = 4.5
_BACKROLL_REPEAT_SAGITTAL_LATERAL_AXIS_MAX = math.sin(math.radians(35.0))
_BACKROLL_REPEAT_SIDE_INVALID_Z = math.sin(math.radians(55.0))
_BACKROLL_REPEAT_MAX_OFFAXIS_ROTATION = math.radians(45.0)


def _grounded_backroll_state(env: ManagerBasedRlEnv) -> None:
    """Create grounded-backroll buffers lazily."""
    _roulade_state(env)
    if hasattr(env, "_backroll_trunk_latch"):
        return
    z = torch.zeros(env.num_envs, device=env.device)
    b = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    i = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env._backroll_trunk_latch = b.clone()
    env._backroll_head_latch = b.clone()
    env._backroll_trunk_latch_now = b.clone()
    env._backroll_head_latch_now = b.clone()
    env._backroll_success = b.clone()
    env._backroll_success_now = b.clone()
    env._backroll_invalid = b.clone()
    env._backroll_invalid_now = b.clone()
    env._backroll_started = b.clone()
    env._backroll_repeat_mode = b.clone()
    env._backroll_air_steps = i.clone()
    env._backroll_max_air_steps = i.clone()
    env._backroll_landing_hold_steps = i.clone()
    env._backroll_invalid_stall_steps = i.clone()
    env._backroll_landing_timeout_steps = i.clone()
    env._backroll_previous_frontier = env._roulade_max.clone()
    env._backroll_completion_paid = env._roulade_max.clone()
    env._backroll_upright_previous = z.clone()
    env._backroll_height_previous = z.clone()
    env._backroll_upright_delta = z.clone()
    env._backroll_height_delta = z.clone()
    env._backroll_potential_ready = b.clone()
    env._backroll_frontier_delta = z.clone()
    env._backroll_cycle_count = i.clone()
    env._backroll_mastery_cycles = torch.ones_like(i)
    env._backroll_cycle_max_lateral_axis_z = z.clone()
    env._backroll_cycle_offaxis_rotation = z.clone()
    env._backroll_episode_max_lateral_axis_z = z.clone()
    env._backroll_episode_max_offaxis_rotation = z.clone()
    env._backroll_last_update_step = -1
    env._backroll_curriculum_stage = 0
    env._backroll_window_episodes = torch.zeros((), dtype=torch.long, device=env.device)
    env._backroll_window_successes = torch.zeros((), dtype=torch.long, device=env.device)
    env._backroll_window_bad_states = torch.zeros((), dtype=torch.long, device=env.device)


def _update_grounded_backroll_state(
    env: ManagerBasedRlEnv,
    asset: Entity,
) -> None:
    """Update ordered contacts, grounding, landing hold, and invalid state."""
    _grounded_backroll_state(env)
    _update_roulade_accum(env, asset)
    step = int(env.common_step_counter)
    if step == env._backroll_last_update_step:
        return

    accum, frontier, _ = _roulade_state(env)
    support = _sensor_any_contact(env, _ROULADE_SUPPORT_SENSOR)
    trunk = _sensor_any_contact(env, _BACKROLL_TRUNK_SENSOR)
    head = _sensor_any_contact(env, _ROULADE_HEAD_SENSOR)
    left_foot = _sensor_any_contact(env, _BACKROLL_LEFT_FOOT_SENSOR)
    right_foot = _sensor_any_contact(env, _BACKROLL_RIGHT_FOOT_SENSOR)
    false = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    support = false if support is None else support
    trunk = false if trunk is None else trunk
    head = false if head is None else head
    left_foot = false if left_foot is None else left_foot
    right_foot = false if right_foot is None else right_foot

    env._backroll_air_steps = torch.where(
        support,
        torch.zeros_like(env._backroll_air_steps),
        env._backroll_air_steps + 1,
    )
    env._backroll_max_air_steps = torch.maximum(
        env._backroll_max_air_steps,
        env._backroll_air_steps,
    )

    trunk_phase = (frontier >= _BACKROLL_TRUNK_LATCH_LO) & (
        frontier <= _BACKROLL_TRUNK_LATCH_HI
    )
    previous_trunk_latch = env._backroll_trunk_latch.clone()
    env._backroll_trunk_latch |= trunk & trunk_phase
    env._backroll_trunk_latch_now = (
        env._backroll_trunk_latch & ~previous_trunk_latch
    )
    head_phase = (frontier >= _BACKROLL_HEAD_LATCH_LO) & (
        frontier <= _BACKROLL_HEAD_LATCH_HI
    )
    previous_head_latch = env._backroll_head_latch.clone()
    env._backroll_head_latch |= (
        env._backroll_trunk_latch
        & head
        & head_phase
        & _head_top_down(env, asset)
    )
    env._backroll_head_latch_now = env._backroll_head_latch & ~previous_head_latch

    quat = asset.data.root_link_quat_w
    upright = torch.nan_to_num(
        1.0 - 2.0 * (quat[:, 1].pow(2) + quat[:, 2].pow(2)),
        nan=-1.0,
    )
    height = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2],
        nan=0.0,
    )
    ang_vel_b = torch.nan_to_num(asset.data.root_link_ang_vel_b, nan=0.0)
    ang_speed = ang_vel_b.norm(dim=-1)
    lateral_axis_z = torch.nan_to_num(_lateral_axis_z(quat), nan=1.0).abs()
    cycle_active = (
        (frontier >= math.radians(10.0))
        | env._backroll_trunk_latch
        | env._backroll_head_latch
    )
    env._backroll_cycle_max_lateral_axis_z = torch.where(
        cycle_active,
        torch.maximum(env._backroll_cycle_max_lateral_axis_z, lateral_axis_z),
        env._backroll_cycle_max_lateral_axis_z,
    )
    offaxis_rate = torch.sqrt(ang_vel_b[:, 0].pow(2) + ang_vel_b[:, 2].pow(2))
    env._backroll_cycle_offaxis_rotation += (
        offaxis_rate * env.step_dt * cycle_active.float()
    )
    env._backroll_episode_max_lateral_axis_z = torch.maximum(
        env._backroll_episode_max_lateral_axis_z,
        env._backroll_cycle_max_lateral_axis_z,
    )
    env._backroll_episode_max_offaxis_rotation = torch.maximum(
        env._backroll_episode_max_offaxis_rotation,
        env._backroll_cycle_offaxis_rotation,
    )
    repeated_sagittal = (
        (env._backroll_cycle_max_lateral_axis_z
         <= _BACKROLL_REPEAT_SAGITTAL_LATERAL_AXIS_MAX)
        & (env._backroll_cycle_offaxis_rotation
           <= _BACKROLL_REPEAT_MAX_OFFAXIS_ROTATION)
    )
    sagittal_landing = ~env._backroll_repeat_mode | repeated_sagittal
    landing_ang_vel_ok = torch.where(
        env._backroll_repeat_mode,
        ang_speed <= _BACKROLL_REPEAT_MAX_LANDING_ANG_VEL,
        ang_speed <= _BACKROLL_MAX_LANDING_ANG_VEL,
    )
    landing_candidate = (
        (frontier >= _BACKROLL_LANDING_ANGLE)
        & ~env._backroll_invalid
        & env._backroll_trunk_latch
        & env._backroll_head_latch
        & left_foot
        & right_foot
        & (height >= _BACKROLL_MIN_LANDING_HEIGHT)
        & (upright >= _BACKROLL_UPRIGHT_COS)
        & landing_ang_vel_ok
        & sagittal_landing
    )
    env._backroll_landing_hold_steps = torch.where(
        landing_candidate,
        env._backroll_landing_hold_steps + 1,
        torch.zeros_like(env._backroll_landing_hold_steps),
    )
    one_shot_hold_steps = max(
        1,
        math.ceil(_BACKROLL_LANDING_HOLD_SECONDS / env.step_dt),
    )
    repeat_hold_steps = max(
        1,
        math.ceil(_BACKROLL_REPEAT_LANDING_HOLD_SECONDS / env.step_dt),
    )
    hold_complete = torch.where(
        env._backroll_repeat_mode,
        env._backroll_landing_hold_steps >= repeat_hold_steps,
        env._backroll_landing_hold_steps >= one_shot_hold_steps,
    )
    env._backroll_success_now = (
        hold_complete
        & ~env._backroll_success
    )
    env._backroll_success |= env._backroll_success_now
    env._backroll_cycle_count += (
        env._backroll_success_now & env._backroll_repeat_mode
    ).long()

    progress = frontier - env._backroll_previous_frontier
    env._backroll_frontier_delta = torch.clamp(progress, min=0.0)
    progressed = progress > 1.0e-4
    invalid_pose = (lateral_axis_z > _FLAT_ZERO) | (
        (upright < _BACKROLL_STALL_TILT_COS) & ~trunk & ~head & ~left_foot & ~right_foot
    )
    repeat_cycle_stalled = (
        env._backroll_repeat_mode
        & cycle_active
        & (frontier < _BACKROLL_LANDING_ANGLE)
    )
    one_shot_invalid_pose = (
        ~env._backroll_repeat_mode
        & (frontier < _BACKROLL_LANDING_ANGLE)
        & invalid_pose
    )
    stalled_invalid = (
        ~env._backroll_success
        & ~progressed
        & (one_shot_invalid_pose | repeat_cycle_stalled)
    )
    env._backroll_invalid_stall_steps = torch.where(
        stalled_invalid,
        env._backroll_invalid_stall_steps + 1,
        torch.zeros_like(env._backroll_invalid_stall_steps),
    )
    max_air_steps = math.ceil(_BACKROLL_MAX_AIR_SECONDS / env.step_dt)
    one_shot_stall_steps = math.ceil(_BACKROLL_INVALID_STALL_SECONDS / env.step_dt)
    repeat_stall_steps = math.ceil(
        _BACKROLL_REPEAT_PRE_EXIT_STALL_SECONDS / env.step_dt
    )
    stalled_too_long = torch.where(
        env._backroll_repeat_mode,
        env._backroll_invalid_stall_steps >= repeat_stall_steps,
        env._backroll_invalid_stall_steps >= one_shot_stall_steps,
    )
    repeat_landing_wait = (
        env._backroll_repeat_mode
        & (frontier >= _BACKROLL_LANDING_ANGLE)
        & ~env._backroll_success
    )
    env._backroll_landing_timeout_steps = torch.where(
        repeat_landing_wait,
        env._backroll_landing_timeout_steps + 1,
        torch.zeros_like(env._backroll_landing_timeout_steps),
    )
    landing_timeout_steps = math.ceil(
        _BACKROLL_REPEAT_LANDING_TIMEOUT_SECONDS / env.step_dt
    )
    landing_timed_out = (
        env._backroll_repeat_mode
        & (env._backroll_landing_timeout_steps >= landing_timeout_steps)
    )
    wrong_way = accum < -math.radians(90.0)
    repeated_side_flop = (
        env._backroll_repeat_mode
        & (frontier >= math.radians(30.0))
        & (lateral_axis_z > _BACKROLL_REPEAT_SIDE_INVALID_Z)
    )
    invalid_now = (
        (env._backroll_max_air_steps > max_air_steps)
        | stalled_too_long
        | landing_timed_out
        | wrong_way
        | repeated_side_flop
    )
    env._backroll_invalid_now = invalid_now & ~env._backroll_invalid
    env._backroll_invalid |= invalid_now

    potential_gate = frontier >= _BACKROLL_POTENTIAL_ANGLE
    upright_potential = torch.clamp(upright, min=0.0, max=1.0)
    height_potential = torch.clamp(height / 0.115, min=0.0, max=1.0)
    env._backroll_upright_delta = torch.where(
        potential_gate & env._backroll_potential_ready,
        upright_potential - env._backroll_upright_previous,
        torch.zeros_like(upright_potential),
    )
    env._backroll_height_delta = torch.where(
        potential_gate & env._backroll_potential_ready,
        height_potential - env._backroll_height_previous,
        torch.zeros_like(height_potential),
    )
    env._backroll_upright_previous = upright_potential
    env._backroll_height_previous = height_potential
    env._backroll_potential_ready = potential_gate
    env._backroll_previous_frontier = frontier.clone()

    # A repeated skill earns one pulse for the completed cycle, then clears all
    # cycle-local latches/frontiers immediately.  The next attempt must again
    # earn trunk contact, flat head-top contact, supported rotation, and a
    # feet-supported sagittal landing; merely remaining upright cannot pay.
    rearm = env._backroll_success_now & env._backroll_repeat_mode
    if rearm.any():
        env._roulade_accum[rearm] = 0.0
        env._roulade_max[rearm] = 0.0
        env._roulade_paid[rearm] = 0.0
        env._roulade_head_latch[rearm] = False
        env._backroll_trunk_latch[rearm] = False
        env._backroll_head_latch[rearm] = False
        env._backroll_trunk_latch_now[rearm] = False
        env._backroll_head_latch_now[rearm] = False
        env._backroll_success[rearm] = False
        env._backroll_air_steps[rearm] = 0
        env._backroll_max_air_steps[rearm] = 0
        env._backroll_landing_hold_steps[rearm] = 0
        env._backroll_invalid_stall_steps[rearm] = 0
        env._backroll_landing_timeout_steps[rearm] = 0
        env._backroll_previous_frontier[rearm] = 0.0
        env._backroll_completion_paid[rearm] = 0.0
        env._backroll_upright_previous[rearm] = 0.0
        env._backroll_height_previous[rearm] = 0.0
        env._backroll_upright_delta[rearm] = 0.0
        env._backroll_height_delta[rearm] = 0.0
        env._backroll_potential_ready[rearm] = False
        env._backroll_cycle_max_lateral_axis_z[rearm] = 0.0
        env._backroll_cycle_offaxis_rotation[rearm] = 0.0
    env._backroll_last_update_step = step


def reset_grounded_backroll_state(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    standing_prob: float = 0.20,
    midroll_prob: float = 0.80,
    midroll_pitch_min: float = math.radians(260.0),
    midroll_pitch_max: float = math.radians(340.0),
    midroll_omega_range: tuple = (0.0, 2.0),
    standing_z_min: float = 0.11,
    standing_z_max: float = 0.12,
    standing_tilt_max: float = math.radians(3.0),
    yaw_range: tuple = (-math.pi, math.pi),
    midroll_z_min: float = 0.05,
    midroll_z_max: float = 0.11,
    tuck_overrides: Optional[dict] = None,
    tuck_factor_range: tuple = (0.5, 1.0),
    joint_noise_std: float = 0.04,
    repeat_mode: bool = False,
    mastery_cycles: int = 1,
) -> None:
    """Reset a grounded backroll and account for the previous episode."""
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device, dtype=torch.long)
    asset: Entity = env.scene[asset_cfg.name]
    _grounded_backroll_state(env)

    previous_started = env._backroll_started[env_ids]
    if previous_started.any():
        previous_ids = env_ids[previous_started]
        env._backroll_window_episodes += previous_started.sum()
        previous_success = torch.where(
            env._backroll_repeat_mode[previous_ids],
            env._backroll_cycle_count[previous_ids]
            >= env._backroll_mastery_cycles[previous_ids],
            env._backroll_success[previous_ids],
        )
        env._backroll_window_successes += previous_success.sum()
        finite = torch.isfinite(env.sim.data.qpos[previous_ids]).all(dim=1) & torch.isfinite(
            env.sim.data.qvel[previous_ids]
        ).all(dim=1)
        env._backroll_window_bad_states += (~finite).sum()

    reset_roulade_state(
        env,
        env_ids,
        asset_cfg=asset_cfg,
        standing_prob=standing_prob,
        midroll_prob=midroll_prob,
        standing_z_min=standing_z_min,
        standing_z_max=standing_z_max,
        standing_tilt_max=standing_tilt_max,
        yaw_range=yaw_range,
        forward_vel_range=(0.0, 0.0),
        midroll_pitch_min=midroll_pitch_min,
        midroll_pitch_max=midroll_pitch_max,
        midroll_z_min=midroll_z_min,
        midroll_z_max=midroll_z_max,
        midroll_omega_range=midroll_omega_range,
        midroll_forward_vel_range=(0.0, 0.0),
        tuck_overrides=tuck_overrides,
        tuck_factor_range=tuck_factor_range,
        joint_noise_std=joint_noise_std,
        roll_direction=-1.0,
    )
    spawn_angle = env._roulade_accum[env_ids]
    is_midroll = spawn_angle > 0.0
    env._backroll_trunk_latch[env_ids] = is_midroll & (
        spawn_angle >= _BACKROLL_TRUNK_LATCH_LO
    )
    env._backroll_head_latch[env_ids] = is_midroll & (
        spawn_angle >= _BACKROLL_HEAD_LATCH_LO
    )
    env._roulade_head_latch[env_ids] = env._backroll_head_latch[env_ids]
    env._backroll_trunk_latch_now[env_ids] = False
    env._backroll_head_latch_now[env_ids] = False
    env._backroll_success[env_ids] = False
    env._backroll_success_now[env_ids] = False
    env._backroll_invalid[env_ids] = False
    env._backroll_invalid_now[env_ids] = False
    env._backroll_started[env_ids] = True
    env._backroll_repeat_mode[env_ids] = repeat_mode
    env._backroll_air_steps[env_ids] = 0
    env._backroll_max_air_steps[env_ids] = 0
    env._backroll_landing_hold_steps[env_ids] = 0
    env._backroll_invalid_stall_steps[env_ids] = 0
    env._backroll_landing_timeout_steps[env_ids] = 0
    env._backroll_previous_frontier[env_ids] = spawn_angle
    env._backroll_completion_paid[env_ids] = spawn_angle
    env._backroll_upright_previous[env_ids] = 0.0
    env._backroll_height_previous[env_ids] = 0.0
    env._backroll_upright_delta[env_ids] = 0.0
    env._backroll_height_delta[env_ids] = 0.0
    env._backroll_potential_ready[env_ids] = False
    env._backroll_frontier_delta[env_ids] = 0.0
    env._backroll_cycle_count[env_ids] = 0
    env._backroll_mastery_cycles[env_ids] = max(1, int(mastery_cycles))
    env._backroll_cycle_max_lateral_axis_z[env_ids] = 0.0
    env._backroll_cycle_offaxis_rotation[env_ids] = 0.0
    env._backroll_episode_max_lateral_axis_z[env_ids] = 0.0
    env._backroll_episode_max_offaxis_rotation[env_ids] = 0.0
    env._backroll_last_update_step = -1


def grounded_backroll_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    event_name: str,
    stages: list[dict],
    window_episodes: int = 4096,
    success_threshold: float = 0.70,
    speed_reward_name: Optional[str] = None,
    speed_reward_weights: Optional[list[float]] = None,
    invalid_reward_name: Optional[str] = None,
    invalid_reward_weights: Optional[list[float]] = None,
) -> torch.Tensor:
    """Advance phase initialization only after a clean mastery window."""
    del env_ids
    _grounded_backroll_state(env)
    stage = int(env._backroll_curriculum_stage)
    if env.common_step_counter % 50 == 0:
        episodes = int(env._backroll_window_episodes.item())
        if episodes >= window_episodes:
            successes = int(env._backroll_window_successes.item())
            bad_states = int(env._backroll_window_bad_states.item())
            success_rate = successes / max(episodes, 1)
            if (
                success_rate >= success_threshold
                and bad_states == 0
                and stage + 1 < len(stages)
            ):
                stage += 1
                env._backroll_curriculum_stage = stage
            env._backroll_window_episodes.zero_()
            env._backroll_window_successes.zero_()
            env._backroll_window_bad_states.zero_()
    event_cfg = env.event_manager.get_term_cfg(event_name)
    event_cfg.params.update(stages[stage]["params"])
    if speed_reward_name is not None:
        if speed_reward_weights is None or len(speed_reward_weights) != len(stages):
            raise ValueError("speed_reward_weights must match the curriculum stages")
        env.reward_manager.get_term_cfg(speed_reward_name).weight = (
            speed_reward_weights[stage]
        )
    if invalid_reward_name is not None:
        if invalid_reward_weights is None or len(invalid_reward_weights) != len(stages):
            raise ValueError("invalid_reward_weights must match the curriculum stages")
        env.reward_manager.get_term_cfg(invalid_reward_name).weight = (
            invalid_reward_weights[stage]
        )
    return torch.tensor(float(stage), device=env.device)


def grounded_backroll_progress(
    env: ManagerBasedRlEnv,
    target_angle: float = 2.0 * math.pi,
    max_paid_rate: float = 5.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Pay only new signed, supported, sagittal backward rotation."""
    asset: Entity = env.scene[asset_cfg.name]
    _update_grounded_backroll_state(env, asset)
    _, frontier, paid = _roulade_state(env)
    new_paid = torch.clamp(frontier, max=target_angle)
    delta = torch.clamp(new_paid - torch.clamp(paid, max=target_angle), min=0.0)
    delta = torch.clamp(delta, max=max_paid_rate * env.step_dt)
    env._roulade_paid = torch.maximum(paid, new_paid)
    # In repeated mode, generic ground support is not enough to earn the
    # maneuver.  A120 rotated to ~170 degrees while missing both the trunk and
    # flat head-top contacts, then parked inverted.  Stop paying that shortcut
    # as soon as each ordered contact becomes physically due.  Phase resets
    # synthesize the already-passed latches, so the reverse curriculum retains
    # its dense late-roll gradient.
    ordered_contacts = (
        ((frontier < _BACKROLL_TRUNK_LATCH_LO) | env._backroll_trunk_latch)
        & ((frontier < _BACKROLL_HEAD_LATCH_LO) | env._backroll_head_latch)
    )
    valid = ~env._backroll_repeat_mode | ordered_contacts
    return torch.where(valid, delta / (env.step_dt * target_angle), 0.0)


def grounded_backroll_head_pivot(
    env: ManagerBasedRlEnv,
    rate_norm: float = 2.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward active flat-head pivoting after ordered trunk contact."""
    asset: Entity = env.scene[asset_cfg.name]
    _update_grounded_backroll_state(env, asset)
    head = _sensor_any_contact(env, _ROULADE_HEAD_SENSOR)
    if head is None:
        return torch.zeros(env.num_envs, device=env.device)
    phase = (env._roulade_max >= _BACKROLL_HEAD_LATCH_LO) & (
        env._roulade_max <= _BACKROLL_HEAD_LATCH_HI
    )
    omega_back = torch.nan_to_num(
        -asset.data.root_link_ang_vel_b[:, 1],
        nan=0.0,
    )
    rate = torch.clamp(omega_back / rate_norm, 0.0, 1.0)
    return (
        env._backroll_trunk_latch.float()
        * head.float()
        * phase.float()
        * _head_top_down(env, asset).float()
        * rate
    )


def grounded_backroll_contact_sequence(
    env: ManagerBasedRlEnv,
    trunk_value: float = 1.0,
    head_value: float = 2.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Bounded one-shot credit for the ordered trunk then flat-head contacts."""
    asset: Entity = env.scene[asset_cfg.name]
    _update_grounded_backroll_state(env, asset)
    return (
        trunk_value * env._backroll_trunk_latch_now.float()
        + head_value * env._backroll_head_latch_now.float()
    ) / env.step_dt


def grounded_backroll_completion_progress(
    env: ManagerBasedRlEnv,
    start_angle: float = math.radians(150.0),
    target_angle: float = math.radians(350.0),
    max_paid_rate: float = 6.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Pay only new supported backward rotation after flat head contact.

    This is the small second-half push missing from the original objective.
    Its own max-so-far payment frontier prevents backward/forward rocking from
    re-earning reward, while the shared roulade frontier already rejects
    airborne, unsupported, forward, and non-sagittal rotation.
    """
    if target_angle <= start_angle:
        raise ValueError("target_angle must be greater than start_angle")
    asset: Entity = env.scene[asset_cfg.name]
    _update_grounded_backroll_state(env, asset)
    _, frontier, _ = _roulade_state(env)
    paid = env._backroll_completion_paid
    clipped_frontier = torch.clamp(frontier, min=start_angle, max=target_angle)
    clipped_paid = torch.clamp(paid, min=start_angle, max=target_angle)
    delta = torch.clamp(clipped_frontier - clipped_paid, min=0.0)
    delta = torch.clamp(delta, max=max_paid_rate * env.step_dt)
    valid = (
        env._backroll_head_latch
        & ~env._backroll_invalid
        & (env._backroll_max_air_steps * env.step_dt <= _BACKROLL_MAX_AIR_SECONDS)
    )
    env._backroll_completion_paid = torch.maximum(paid, frontier)
    span = target_angle - start_angle
    return torch.where(valid, delta / (env.step_dt * span), 0.0)


def grounded_backroll_upright_progress(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    _update_grounded_backroll_state(env, asset)
    return env._backroll_upright_delta / env.step_dt


def grounded_backroll_height_progress(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    _update_grounded_backroll_state(env, asset)
    return env._backroll_height_delta / env.step_dt


def grounded_backroll_rise_velocity(
    env: ManagerBasedRlEnv,
    gate_lo: float = math.radians(180.0),
    gate_hi: float = math.radians(260.0),
    max_height: float = 0.125,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Modest late-roll upward-velocity bootstrap after ordered head contact."""
    asset: Entity = env.scene[asset_cfg.name]
    _update_grounded_backroll_state(env, asset)
    if gate_hi <= gate_lo:
        raise ValueError("gate_hi must be greater than gate_lo")
    frontier = env._roulade_max
    phase = torch.clamp((frontier - gate_lo) / (gate_hi - gate_lo), 0.0, 1.0)
    phase = phase * phase * (3.0 - 2.0 * phase)
    height = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2],
        nan=0.0,
    )
    upward = torch.clamp(
        torch.nan_to_num(asset.data.root_link_lin_vel_w[:, 2], nan=0.0),
        min=0.0,
    )
    valid = env._backroll_head_latch & ~env._backroll_invalid
    return upward * phase * (height < max_height).float() * valid.float()


def grounded_backroll_speed_progress(
    env: ManagerBasedRlEnv,
    minimum_rate: float = 2.0,
    target_rate: float = 4.5,
    target_angle: float = 2.0 * math.pi,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward fast *new* supported backward progress without rocking annuities.

    The payment is proportional to the current cycle's newly extended angular
    frontier.  Reversing, revisiting an old angle, spinning in the air, or
    holding angular velocity without progress therefore earns zero.
    """
    if target_rate <= minimum_rate:
        raise ValueError("target_rate must be greater than minimum_rate")
    asset: Entity = env.scene[asset_cfg.name]
    _update_grounded_backroll_state(env, asset)
    omega_back = torch.nan_to_num(-asset.data.root_link_ang_vel_b[:, 1], nan=0.0)
    speed = torch.clamp(
        (omega_back - minimum_rate) / (target_rate - minimum_rate),
        min=0.0,
        max=1.0,
    )
    strict_sagittal = (
        env._backroll_cycle_max_lateral_axis_z
        <= _BACKROLL_REPEAT_SAGITTAL_LATERAL_AXIS_MAX
    )
    ordered_contacts = (
        ((env._roulade_max < _BACKROLL_TRUNK_LATCH_LO) | env._backroll_trunk_latch)
        & ((env._roulade_max < _BACKROLL_HEAD_LATCH_LO) | env._backroll_head_latch)
    )
    return (
        env._backroll_frontier_delta
        * speed
        * strict_sagittal.float()
        * ordered_contacts.float()
        / (env.step_dt * target_angle)
    )


def grounded_backroll_success_rate(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    _update_grounded_backroll_state(env, asset)
    return env._backroll_success_now.float() / env.step_dt


def grounded_backroll_repeat_success_rate(
    env: ManagerBasedRlEnv,
    later_cycle_bonus: float = 0.5,
    max_bonus_cycles: int = 3,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """One pulse per valid cycle, with a bounded preference for chaining."""
    asset: Entity = env.scene[asset_cfg.name]
    _update_grounded_backroll_state(env, asset)
    completed_before = torch.clamp(
        env._backroll_cycle_count - 1,
        min=0,
        max=max_bonus_cycles,
    ).float()
    multiplier = 1.0 + later_cycle_bonus * completed_before
    return env._backroll_success_now.float() * multiplier / env.step_dt


def grounded_backroll_success_termination(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    _update_grounded_backroll_state(env, asset)
    return env._backroll_success


def grounded_backroll_invalid_termination(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    _update_grounded_backroll_state(env, asset)
    return env._backroll_invalid


def grounded_backroll_invalid_rate(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """One-shot terminal cost so deliberately flopping cannot shorten taxes."""
    asset: Entity = env.scene[asset_cfg.name]
    _update_grounded_backroll_state(env, asset)
    return env._backroll_invalid_now.float() / env.step_dt


def grounded_backroll_rotation_deg(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    _update_grounded_backroll_state(env, asset)
    return torch.rad2deg(env._roulade_max)


def grounded_backroll_trunk_contact_latched(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    _update_grounded_backroll_state(env, asset)
    return env._backroll_trunk_latch.float()


def grounded_backroll_head_contact_latched(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    _update_grounded_backroll_state(env, asset)
    return env._backroll_head_latch.float()


def grounded_backroll_max_air_gap_s(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    _update_grounded_backroll_state(env, asset)
    return env._backroll_max_air_steps.float() * env.step_dt


def grounded_backroll_landing_hold_s(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    _update_grounded_backroll_state(env, asset)
    return env._backroll_landing_hold_steps.float() * env.step_dt


def grounded_backroll_success_fraction(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    _update_grounded_backroll_state(env, asset)
    return env._backroll_success.float()


def grounded_backroll_cycle_count(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    _update_grounded_backroll_state(env, asset)
    return env._backroll_cycle_count.float()


def grounded_backroll_episode_max_lateral_axis_z(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    _update_grounded_backroll_state(env, asset)
    return env._backroll_episode_max_lateral_axis_z


def grounded_backroll_episode_max_offaxis_deg(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    _update_grounded_backroll_state(env, asset)
    return torch.rad2deg(env._backroll_episode_max_offaxis_rotation)


# ============================================================================
# Roll sprint task: repeated supported forward rolls
# ============================================================================
#
# This state is deliberately separate from the one-roll roulade state.  The
# existing task uses a monotonic angular frontier and landing rewards; a sprint
# needs a cyclic accumulator plus a separate global forward frontier. Distance
# is released only at a valid cycle boundary and only for newly covered ground.
_ROLL_SPRINT_TARGET_ANGLE = 2.0 * math.pi
_ROLL_SPRINT_SUPPORT_SENSOR = _ROULADE_SUPPORT_SENSOR
_ROLL_SPRINT_HEAD_SENSOR = _ROULADE_HEAD_SENSOR
_ROLL_SPRINT_FEET_SENSOR = "feet_ground_contact"
_ROLL_SPRINT_MIN_FORWARD_RATE = 0.5
_ROLL_SPRINT_MAX_DISTANCE_PER_RAD = 0.12
_ROLL_SPRINT_ROAD_HALF_WIDTH = 0.56
_ROLL_SPRINT_ROAD_SAFE_HALF_WIDTH = 0.42
# Compatibility aliases for older checkpoint tooling. These now describe the
# fixed shared road, not an individual start lane.
_ROLL_SPRINT_LANE_HALF_WIDTH = _ROLL_SPRINT_ROAD_HALF_WIDTH
_ROLL_SPRINT_BOOTSTRAP_LANE_HALF_WIDTH = _ROLL_SPRINT_ROAD_HALF_WIDTH
_ROLL_SPRINT_LATERAL_INVALID_Z = math.sin(math.radians(60.0))
# A launch-ready recovery is deliberately dynamic. Normal MicroDuck roll
# transit is 3.5--5.5 rad/s. The frozen parent policy produced 94 feet-down,
# head-released, upright, sagittal recovery steps in 10 s, but the old 3 rad/s
# cutoff rejected 92 of them. Six rad/s preserves that dynamic transition
# without accepting the inverted phase, which is excluded independently by
# feet support, head release, and the upright cone. Five 50 Hz steps enforce
# the required 100 ms continuous launch-ready latch.
_ROLL_SPRINT_RECOVERY_MAX_FORWARD_RATE = 6.0
_ROLL_SPRINT_RECOVERY_UPRIGHT_COS = math.cos(math.radians(25.0))
_ROLL_SPRINT_RECOVERY_LATERAL_Z = math.sin(math.radians(35.0))
_ROLL_SPRINT_RECOVERY_MIN_HEIGHT_M = 0.09
_ROLL_SPRINT_RECOVERY_HOLD_STEPS = 5
_ROLL_SPRINT_REARM_HOLD_STEPS = 2
_ROLL_SPRINT_SELF_RIGHT_TILT_COS = math.cos(math.radians(60.0))
_ROLL_SPRINT_SELF_RIGHT_STALL_RATE = 1.0
_ROLL_SPRINT_SELF_RIGHT_STALL_SECONDS = 0.30
_ROLL_SPRINT_SELF_RIGHT_HEIGHT_CEILING_M = 0.115
# The four visible lane guides share one 1.12 m-wide course road. Internal lane
# crossings are legal. Reposition begins before the hard road edge, rearms with
# hysteresis, and aims only for the nearest safe edge instead of the center.
_ROLL_SPRINT_REPOSITION_TRIGGER_M = 0.50
_ROLL_SPRINT_REPOSITION_REARM_M = 0.46
_ROLL_SPRINT_REPOSITION_LATERAL_COMMAND_MPS = 0.20
_ROLL_SPRINT_REPOSITION_LATERAL_KP = 2.0
_ROLL_SPRINT_REPOSITION_HEADING_TRIGGER_RAD = math.radians(20.0)
_ROLL_SPRINT_REPOSITION_HEADING_REARM_RAD = math.radians(10.0)
_ROLL_SPRINT_REPOSITION_HEADING_COMMAND_KP = 1.0
_ROLL_SPRINT_HEADING_RETURN_RESET_MIN_RAD = math.radians(20.0)
_ROLL_SPRINT_HEADING_RETURN_RESET_MAX_RAD = math.radians(40.0)
# The inherited actor's yaw-command normalizer has std about 0.024. Keeping
# this at 0.05 exposes direction and error without a 15-sigma distribution
# shift that would destabilize the compatible A60 warm start.
_ROLL_SPRINT_REPOSITION_YAW_COMMAND_RPS = 0.05


def _roll_sprint_state(env: ManagerBasedRlEnv) -> None:
    """Create the per-environment cyclic roll-sprint buffers lazily."""
    if hasattr(env, "_roll_sprint_accum"):
        if not hasattr(env, "_roll_sprint_roll_direction"):
            env._roll_sprint_roll_direction = torch.ones(
                env.num_envs, device=env.device
            )
        if not hasattr(env, "_roll_sprint_body_heading_w"):
            env._roll_sprint_body_heading_w = env._roll_sprint_heading_w.clone()
        if not hasattr(env, "_roll_sprint_directional_bootstrap_frontier"):
            env._roll_sprint_directional_bootstrap_frontier = torch.zeros(
                env.num_envs, device=env.device
            )
            env._roll_sprint_directional_position_frontier = torch.zeros(
                env.num_envs, device=env.device
            )
            env._roll_sprint_directional_bootstrap_delta = torch.zeros(
                env.num_envs, device=env.device
            )
        return
    z = torch.zeros(env.num_envs, device=env.device)
    env._roll_sprint_accum = z.clone()
    env._roll_sprint_phase_frontier = z.clone()
    env._roll_sprint_completed = z.clone()
    env._roll_sprint_head_latch = torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device
    )
    env._roll_sprint_cycle_eligible = torch.ones(
        env.num_envs, dtype=torch.bool, device=env.device
    )
    env._roll_sprint_awaiting_recovery = torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device
    )
    env._roll_sprint_awaiting_reposition = torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device
    )
    env._roll_sprint_self_righting = torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device
    )
    env._roll_sprint_self_right_started_now = torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device
    )
    env._roll_sprint_self_righted_now = torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device
    )
    env._roll_sprint_self_right_stall_steps = torch.zeros(
        env.num_envs, dtype=torch.long, device=env.device
    )
    env._roll_sprint_self_right_hold_steps = torch.zeros(
        env.num_envs, dtype=torch.long, device=env.device
    )
    env._roll_sprint_self_right_latency_steps = torch.zeros(
        env.num_envs, dtype=torch.long, device=env.device
    )
    env._roll_sprint_self_right_latency_total_steps = z.clone()
    env._roll_sprint_self_right_attempt_count = z.clone()
    env._roll_sprint_self_right_success_count = z.clone()
    env._roll_sprint_self_right_upright_delta = z.clone()
    env._roll_sprint_self_right_height_delta = z.clone()
    env._roll_sprint_self_right_upright_previous = z.clone()
    env._roll_sprint_self_right_height_previous = z.clone()
    env._roll_sprint_self_right_potential_ready = torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device
    )
    env._roll_sprint_frontier_after_self_right = z.clone()
    env._roll_sprint_recovery_spawn_kind = torch.zeros(
        env.num_envs, dtype=torch.long, device=env.device
    )
    env._roll_sprint_reposition_latency_steps = torch.zeros(
        env.num_envs, dtype=torch.long, device=env.device
    )
    env._roll_sprint_reposition_latency_total_steps = z.clone()
    env._roll_sprint_reposition_count = z.clone()
    env._roll_sprint_recovery_hold_steps = torch.zeros(
        env.num_envs, dtype=torch.long, device=env.device
    )
    env._roll_sprint_recovery_latency_steps = torch.zeros(
        env.num_envs, dtype=torch.long, device=env.device
    )
    env._roll_sprint_recovery_latency_total_steps = z.clone()
    env._roll_sprint_recovery_count = z.clone()
    env._roll_sprint_recovered_cycle_armed = torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device
    )
    env._roll_sprint_recovered_and_rerolled = z.clone()
    env._roll_sprint_recovered_now = torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device
    )
    env._roll_sprint_repositioned_now = torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device
    )
    env._roll_sprint_recovered_rerolled_now = torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device
    )
    env._roll_sprint_lateral_invalid = torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device
    )
    env._roll_sprint_forward_position = z.clone()
    env._roll_sprint_forward_origin = z.clone()
    env._roll_sprint_cycle_start_position = z.clone()
    env._roll_sprint_forward_frontier = z.clone()
    env._roll_sprint_lateral_origin = z.clone()
    env._roll_sprint_lateral_displacement = z.clone()
    env._roll_sprint_lane_centering_delta = z.clone()
    env._roll_sprint_lane_half_width_m = _ROLL_SPRINT_ROAD_HALF_WIDTH
    env._roll_sprint_course_center_xy_w = torch.zeros(
        (env.num_envs, 2), device=env.device
    )
    env._roll_sprint_course_lateral_position = z.clone()
    env._roll_sprint_roll_direction = torch.ones(env.num_envs, device=env.device)
    env._roll_sprint_body_heading_w = torch.zeros(
        (env.num_envs, 2), device=env.device
    )
    env._roll_sprint_heading_w = torch.zeros((env.num_envs, 2), device=env.device)
    env._roll_sprint_heading_ready = torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device
    )
    env._roll_sprint_reset_heading_override_w = torch.zeros(
        (env.num_envs, 2), device=env.device
    )
    env._roll_sprint_reset_heading_override_valid = torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device
    )
    env._roll_sprint_spawn_heading_error_rad = z.clone()
    env._roll_sprint_heading_error_rad = z.clone()
    env._roll_sprint_heading_alignment_delta = z.clone()
    env._roll_sprint_progress_delta = z.clone()
    env._roll_sprint_directional_bootstrap_frontier = z.clone()
    env._roll_sprint_directional_position_frontier = z.clone()
    env._roll_sprint_directional_bootstrap_delta = z.clone()
    env._roll_sprint_completed_distance = z.clone()
    env._roll_sprint_credited_distance = z.clone()
    env._roll_sprint_completed_now = torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device
    )
    env._roll_sprint_invalid_now = torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device
    )
    env._roll_sprint_bootstrap_completed_now = torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device
    )
    env._roll_sprint_last_update_step = torch.full(
        (env.num_envs,), -1, dtype=torch.long, device=env.device
    )


def _roll_sprint_heading(asset: Entity) -> torch.Tensor:
    """Return planar forward heading, including at a 90-degree roll pose.

    Projecting body x fails when the robot is pitched vertically because the
    planar vector collapses to zero.  Pure sagittal pitch leaves body y in the
    horizontal plane, so rotating its projected left vector by -90 degrees
    recovers the reset heading without a gimbal singularity.
    """
    quat = torch.nan_to_num(asset.data.root_link_quat_w, nan=0.0)
    w, x, y, z = quat.unbind(dim=-1)
    body_y_x = 2.0 * (x * y - w * z)
    body_y_y = 1.0 - 2.0 * (x.square() + z.square())
    heading = torch.stack(
        [body_y_y, -body_y_x],
        dim=-1,
    )
    return heading / heading.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)


def _roll_sprint_signed_heading_error(
    current_heading_w: torch.Tensor,
    target_heading_w: torch.Tensor,
) -> torch.Tensor:
    """Return the wrap-safe signed yaw correction from current to target."""
    dot = (current_heading_w * target_heading_w).sum(dim=-1).clamp(-1.0, 1.0)
    cross = (
        current_heading_w[:, 0] * target_heading_w[:, 1]
        - current_heading_w[:, 1] * target_heading_w[:, 0]
    )
    return torch.atan2(cross, dot)


def _roll_sprint_rotate_heading(
    heading_w: torch.Tensor,
    angle_rad: torch.Tensor,
) -> torch.Tensor:
    """Rotate planar headings by signed angles without changing their norm."""
    cosine = torch.cos(angle_rad)
    sine = torch.sin(angle_rad)
    return torch.stack(
        (
            heading_w[:, 0] * cosine - heading_w[:, 1] * sine,
            heading_w[:, 0] * sine + heading_w[:, 1] * cosine,
        ),
        dim=-1,
    )


def _roll_sprint_forward_position(
    env: ManagerBasedRlEnv,
    asset: Entity,
    heading_w: torch.Tensor,
) -> torch.Tensor:
    """Project root position onto the frozen global course heading."""
    position_xy = torch.nan_to_num(asset.data.root_link_pos_w[:, :2], nan=0.0)
    course_center_xy = torch.nan_to_num(
        env._roll_sprint_course_center_xy_w,
        nan=0.0,
    )
    return ((position_xy - course_center_xy) * heading_w).sum(dim=-1)


def _roll_sprint_lateral_position(
    env: ManagerBasedRlEnv,
    asset: Entity,
    heading_w: torch.Tensor,
) -> torch.Tensor:
    """Project root position onto the frozen shared course's lateral axis."""
    position_xy = torch.nan_to_num(asset.data.root_link_pos_w[:, :2], nan=0.0)
    course_center_xy = torch.nan_to_num(
        env._roll_sprint_course_center_xy_w,
        nan=0.0,
    )
    lateral_w = torch.stack((-heading_w[:, 1], heading_w[:, 0]), dim=-1)
    return ((position_xy - course_center_xy) * lateral_w).sum(dim=-1)


def _roll_sprint_bounded_cycle_credit(
    rotation_accum: torch.Tensor,
    forward_position: torch.Tensor,
    cycle_start_position: torch.Tensor,
    forward_frontier: torch.Tensor,
) -> torch.Tensor:
    """Return the valid-cycle distance credit without advancing raw frontier state."""
    rotation_budget = _ROLL_SPRINT_MAX_DISTANCE_PER_RAD * torch.clamp(
        rotation_accum, min=0.0, max=_ROLL_SPRINT_TARGET_ANGLE
    )
    cycle_net_advance = torch.clamp(
        forward_position - cycle_start_position, min=0.0
    )
    new_frontier_advance = torch.clamp(
        forward_position - forward_frontier, min=0.0
    )
    return torch.minimum(
        rotation_budget,
        torch.minimum(cycle_net_advance, new_frontier_advance),
    )


def _roll_sprint_directional_bootstrap_update(
    previous_frontier: torch.Tensor,
    previous_position_frontier: torch.Tensor,
    rotation_accum: torch.Tensor,
    progress_delta: torch.Tensor,
    forward_position: torch.Tensor,
    cycle_start_position: torch.Tensor,
    eligible: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance a non-farmable partial-cycle toward-goal potential.

    This is not positive-velocity integration. Credit is bounded by both the
    current signed rotation budget and net course advance from cycle start,
    advances only on a step with new signed phase progress, and retains a
    per-cycle frontier so backtracking and revisiting cannot pay twice.
    """
    rotation_budget = _ROLL_SPRINT_MAX_DISTANCE_PER_RAD * torch.clamp(
        rotation_accum,
        min=0.0,
        max=_ROLL_SPRINT_TARGET_ANGLE,
    )
    cycle_net_advance = torch.clamp(
        forward_position - cycle_start_position,
        min=0.0,
    )
    potential = torch.minimum(rotation_budget, cycle_net_advance)
    can_advance = (
        eligible
        & (progress_delta > 1.0e-8)
        & (cycle_net_advance > previous_position_frontier + 1.0e-8)
    )
    next_position_frontier = torch.where(
        can_advance,
        torch.maximum(previous_position_frontier, cycle_net_advance),
        previous_position_frontier,
    )
    next_frontier = torch.where(
        can_advance,
        torch.maximum(previous_frontier, potential),
        previous_frontier,
    )
    return (
        next_frontier,
        next_position_frontier,
        torch.clamp(next_frontier - previous_frontier, min=0.0),
    )


def _update_roll_sprint_state(env: ManagerBasedRlEnv, asset: Entity) -> None:
    """Advance one cyclic roll state update, guarded against double updates.

    Signed angular progress requires ground support and sagittal orientation,
    so backward rocking removes phase instead of accumulating a fake turn.
    Dense shaping advances only at a new per-cycle angular frontier. Forward
    distance is the signed net cycle advance beyond the globally credited
    frontier, bounded by the cycle's supported rotation budget, and released
    only after a witnessed flat head-top contact on an eligible full cycle.
    """
    _roll_sprint_state(env)
    step = int(env.common_step_counter)
    active = env._roll_sprint_last_update_step != step
    if not bool(active.any()):
        return

    body_heading = _roll_sprint_heading(asset)
    not_ready = active & ~env._roll_sprint_heading_ready
    use_reset_heading = not_ready & env._roll_sprint_reset_heading_override_valid
    reset_body_heading = torch.where(
        use_reset_heading.unsqueeze(-1),
        env._roll_sprint_reset_heading_override_w,
        body_heading,
    )
    initial_body_heading = _roll_sprint_rotate_heading(
        reset_body_heading,
        torch.where(
            not_ready,
            env._roll_sprint_spawn_heading_error_rad,
            torch.zeros_like(env._roll_sprint_spawn_heading_error_rad),
        ),
    )
    reference_body_heading = torch.where(
        not_ready.unsqueeze(-1),
        initial_body_heading,
        env._roll_sprint_body_heading_w,
    )
    reference_heading = (
        reference_body_heading * env._roll_sprint_roll_direction.unsqueeze(-1)
    )
    env._roll_sprint_body_heading_w = reference_body_heading
    env._roll_sprint_heading_w = reference_heading
    position_xy = torch.nan_to_num(asset.data.root_link_pos_w[:, :2], nan=0.0)
    reference_lateral_w = torch.stack(
        (-reference_heading[:, 1], reference_heading[:, 0]), dim=-1
    )
    initialized_course_center_xy = position_xy - (
        env._roll_sprint_course_lateral_position.unsqueeze(-1) * reference_lateral_w
    )
    env._roll_sprint_course_center_xy_w = torch.where(
        not_ready.unsqueeze(-1),
        initialized_course_center_xy,
        env._roll_sprint_course_center_xy_w,
    )
    heading_error = _roll_sprint_signed_heading_error(
        body_heading, reference_body_heading
    )
    heading_error_abs = heading_error.abs()
    previous_heading_error_abs = torch.where(
        not_ready,
        heading_error_abs,
        env._roll_sprint_heading_error_rad.abs(),
    )
    env._roll_sprint_heading_alignment_delta = torch.where(
        active,
        previous_heading_error_abs - heading_error_abs,
        env._roll_sprint_heading_alignment_delta,
    )
    env._roll_sprint_heading_error_rad = torch.where(
        active,
        heading_error,
        env._roll_sprint_heading_error_rad,
    )
    position = _roll_sprint_forward_position(env, asset, reference_heading)
    env._roll_sprint_forward_origin = torch.where(
        not_ready, position, env._roll_sprint_forward_origin
    )
    cycle_start_position = torch.where(
        not_ready, position, env._roll_sprint_cycle_start_position
    )
    forward_frontier = torch.where(
        not_ready, position, env._roll_sprint_forward_frontier
    )
    lateral_position = _roll_sprint_lateral_position(
        env, asset, reference_heading
    )
    lateral_origin = torch.where(
        not_ready, lateral_position, env._roll_sprint_lateral_origin
    )
    env._roll_sprint_heading_ready |= not_ready
    env._roll_sprint_reset_heading_override_valid &= ~not_ready
    env._roll_sprint_spawn_heading_error_rad = torch.where(
        active & not_ready,
        torch.zeros_like(env._roll_sprint_spawn_heading_error_rad),
        env._roll_sprint_spawn_heading_error_rad,
    )
    env._roll_sprint_forward_position = torch.where(
        active, position, env._roll_sprint_forward_position
    )
    lateral_displacement = lateral_position - lateral_origin
    previous_course_lateral = torch.where(
        not_ready,
        lateral_position,
        env._roll_sprint_course_lateral_position,
    )
    previous_road_edge_excess = torch.clamp(
        previous_course_lateral.abs() - _ROLL_SPRINT_ROAD_SAFE_HALF_WIDTH,
        min=0.0,
    )
    road_edge_excess = torch.clamp(
        lateral_position.abs() - _ROLL_SPRINT_ROAD_SAFE_HALF_WIDTH,
        min=0.0,
    )
    env._roll_sprint_lane_centering_delta = torch.where(
        active,
        previous_road_edge_excess - road_edge_excess,
        env._roll_sprint_lane_centering_delta,
    )
    env._roll_sprint_lateral_displacement = torch.where(
        active,
        lateral_displacement,
        env._roll_sprint_lateral_displacement,
    )
    env._roll_sprint_course_lateral_position = torch.where(
        active,
        lateral_position,
        env._roll_sprint_course_lateral_position,
    )

    support = _sensor_any_contact(env, _ROLL_SPRINT_SUPPORT_SENSOR)
    if support is None:
        support = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    foot_support = _sensor_any_contact(env, _ROLL_SPRINT_FEET_SENSOR)
    if foot_support is None:
        foot_support = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    head_contact = _sensor_any_contact(env, _ROLL_SPRINT_HEAD_SENSOR)
    if head_contact is None:
        head_contact = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    omega_fwd = torch.nan_to_num(
        _ROULADE_FWD_SIGN
        * env._roll_sprint_roll_direction
        * asset.data.root_link_ang_vel_b[:, 1],
        nan=0.0,
    )
    positive_rate = torch.clamp(omega_fwd, min=0.0)

    lateral_z = torch.nan_to_num(
        _lateral_axis_z(asset.data.root_link_quat_w), nan=1.0
    ).abs()
    flat_u = torch.clamp((_FLAT_ZERO - lateral_z) / (_FLAT_ZERO - _FLAT_FULL), 0.0, 1.0)
    flatness = flat_u.square() * (3.0 - 2.0 * flat_u)

    quat = torch.nan_to_num(asset.data.root_link_quat_w, nan=0.0)
    upright_cos = 1.0 - 2.0 * (quat[:, 1].square() + quat[:, 2].square())
    relative_height = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2],
        nan=0.0,
    )
    awaiting_before = env._roll_sprint_awaiting_recovery
    reposition_before = env._roll_sprint_awaiting_reposition
    self_right_before = env._roll_sprint_self_righting
    reposition_triggered = (
        active
        & ~self_right_before
        & ~reposition_before
        & (upright_cos > _ROLL_SPRINT_SELF_RIGHT_TILT_COS)
        & (
            (lateral_position.abs() > _ROLL_SPRINT_REPOSITION_TRIGGER_M)
            | (
                heading_error_abs
                > _ROLL_SPRINT_REPOSITION_HEADING_TRIGGER_RAD
            )
        )
    )
    waiting_before = awaiting_before | reposition_before | self_right_before

    launch_ready = (
        foot_support
        & ~head_contact
        & (upright_cos >= _ROLL_SPRINT_RECOVERY_UPRIGHT_COS)
        & (relative_height >= _ROLL_SPRINT_RECOVERY_MIN_HEIGHT_M)
        & (omega_fwd.abs() <= _ROLL_SPRINT_RECOVERY_MAX_FORWARD_RATE)
    )

    # Recovery shaping is potential-based and exists only while the explicit
    # self-righting mode was active before this update. A newly entered mode
    # snapshots the current potential so neither reset pose nor mode entry can
    # create a jackpot.
    upright_potential = upright_cos
    height_potential = torch.clamp(
        relative_height,
        max=_ROLL_SPRINT_SELF_RIGHT_HEIGHT_CEILING_M,
    )
    potential_ready = env._roll_sprint_self_right_potential_ready
    self_right_potential_active = active & self_right_before & potential_ready
    env._roll_sprint_self_right_upright_delta = torch.where(
        active,
        torch.where(
            self_right_potential_active,
            upright_potential - env._roll_sprint_self_right_upright_previous,
            torch.zeros_like(upright_potential),
        ),
        env._roll_sprint_self_right_upright_delta,
    )
    env._roll_sprint_self_right_height_delta = torch.where(
        active,
        torch.where(
            self_right_potential_active,
            height_potential - env._roll_sprint_self_right_height_previous,
            torch.zeros_like(height_potential),
        ),
        env._roll_sprint_self_right_height_delta,
    )

    self_right_candidate = active & self_right_before & launch_ready
    old_self_right_hold = env._roll_sprint_self_right_hold_steps
    self_right_hold = torch.where(
        self_right_candidate,
        old_self_right_hold + 1,
        torch.zeros_like(old_self_right_hold),
    )
    self_righted_now = (
        self_right_candidate
        & (old_self_right_hold < _ROLL_SPRINT_RECOVERY_HOLD_STEPS)
        & (self_right_hold >= _ROLL_SPRINT_RECOVERY_HOLD_STEPS)
    )

    regular_transition_candidate = (
        active
        & ~self_right_before
        & (awaiting_before | reposition_before)
        & (lateral_position.abs() <= _ROLL_SPRINT_REPOSITION_REARM_M)
        & (heading_error_abs <= _ROLL_SPRINT_REPOSITION_HEADING_REARM_RAD)
        & launch_ready
    )
    old_recovery_hold = env._roll_sprint_recovery_hold_steps
    recovery_hold = torch.where(
        regular_transition_candidate,
        old_recovery_hold + 1,
        torch.zeros_like(old_recovery_hold),
    )
    regular_transitioned_now = (
        regular_transition_candidate
        & (old_recovery_hold < _ROLL_SPRINT_REARM_HOLD_STEPS)
        & (recovery_hold >= _ROLL_SPRINT_REARM_HOLD_STEPS)
    )
    self_right_rearmed_now = self_righted_now & (
        lateral_position.abs() <= _ROLL_SPRINT_REPOSITION_REARM_M
    ) & (heading_error_abs <= _ROLL_SPRINT_REPOSITION_HEADING_REARM_RAD)
    transitioned_now = regular_transitioned_now | self_right_rearmed_now
    recovered_now = transitioned_now & awaiting_before
    repositioned_now = transitioned_now & reposition_before

    self_right_latency_steps = torch.where(
        active & self_right_before,
        env._roll_sprint_self_right_latency_steps + 1,
        env._roll_sprint_self_right_latency_steps,
    )
    env._roll_sprint_self_right_latency_total_steps += torch.where(
        self_righted_now,
        self_right_latency_steps.float(),
        torch.zeros_like(env._roll_sprint_self_right_latency_total_steps),
    )
    env._roll_sprint_self_right_success_count += self_righted_now.float()
    recovery_latency_steps = torch.where(
        active & awaiting_before,
        env._roll_sprint_recovery_latency_steps + 1,
        env._roll_sprint_recovery_latency_steps,
    )
    env._roll_sprint_recovery_latency_total_steps += torch.where(
        recovered_now,
        recovery_latency_steps.float(),
        torch.zeros_like(env._roll_sprint_recovery_latency_total_steps),
    )
    env._roll_sprint_recovery_count += recovered_now.float()
    reposition_latency_steps = torch.where(
        active & reposition_before,
        env._roll_sprint_reposition_latency_steps + 1,
        env._roll_sprint_reposition_latency_steps,
    )
    env._roll_sprint_reposition_latency_total_steps += torch.where(
        repositioned_now,
        reposition_latency_steps.float(),
        torch.zeros_like(env._roll_sprint_reposition_latency_total_steps),
    )
    env._roll_sprint_reposition_count += repositioned_now.float()
    env._roll_sprint_recovered_now = torch.where(
        active, recovered_now, env._roll_sprint_recovered_now
    )
    env._roll_sprint_repositioned_now = torch.where(
        active, repositioned_now, env._roll_sprint_repositioned_now
    )
    env._roll_sprint_self_righted_now = torch.where(
        active, self_righted_now, env._roll_sprint_self_righted_now
    )
    env._roll_sprint_self_right_hold_steps = torch.where(
        active, self_right_hold, env._roll_sprint_self_right_hold_steps
    )
    env._roll_sprint_recovery_hold_steps = torch.where(
        active, recovery_hold, env._roll_sprint_recovery_hold_steps
    )
    env._roll_sprint_self_right_latency_steps = torch.where(
        self_righted_now,
        torch.zeros_like(self_right_latency_steps),
        self_right_latency_steps,
    )
    env._roll_sprint_recovery_latency_steps = torch.where(
        transitioned_now,
        torch.zeros_like(recovery_latency_steps),
        recovery_latency_steps,
    )
    env._roll_sprint_reposition_latency_steps = torch.where(
        transitioned_now,
        torch.zeros_like(reposition_latency_steps),
        reposition_latency_steps,
    )

    # Recovery and lane reposition both block angular progress. Returning to a
    # launch-ready pose inside the lane rearms phase zero, but that transition
    # cannot also rotate the new cycle on the same simulation step.
    locked_before = waiting_before | reposition_triggered
    rotation_eligible = active & ~locked_before & ~transitioned_now
    supported = support.float()
    valid_rotation = supported * flatness
    signed_rotation_delta = (
        omega_fwd * env.step_dt * valid_rotation * rotation_eligible.float()
    )

    old_accum = torch.where(
        locked_before | transitioned_now,
        torch.zeros_like(env._roll_sprint_accum),
        env._roll_sprint_accum,
    )
    candidate_accum = torch.clamp(old_accum + signed_rotation_delta, min=0.0)
    new_accum = torch.where(active, candidate_accum, old_accum)
    old_phase_frontier = env._roll_sprint_phase_frontier
    new_phase_frontier = torch.where(
        active,
        torch.maximum(old_phase_frontier, new_accum),
        old_phase_frontier,
    )
    progress_delta = torch.clamp(new_phase_frontier - old_phase_frontier, min=0.0)
    env._roll_sprint_progress_delta = torch.where(
        active, progress_delta, env._roll_sprint_progress_delta
    )

    # A low, supported robot that has made no new phase progress for 0.30 s is
    # an accidental stalled fall. Active forward phase progress resets this
    # timer every step, so the mode cannot interrupt a moving roll.
    stalled_fall_candidate = (
        active
        & ~waiting_before
        & ~reposition_triggered
        & support
        & (upright_cos <= _ROLL_SPRINT_SELF_RIGHT_TILT_COS)
        & (omega_fwd.abs() < _ROLL_SPRINT_SELF_RIGHT_STALL_RATE)
        & (progress_delta <= 1.0e-8)
    )
    old_stall_steps = env._roll_sprint_self_right_stall_steps
    stall_steps = torch.where(
        stalled_fall_candidate,
        old_stall_steps + 1,
        torch.zeros_like(old_stall_steps),
    )
    stall_threshold_steps = max(
        1,
        round(_ROLL_SPRINT_SELF_RIGHT_STALL_SECONDS / env.step_dt),
    )
    stalled_fall_now = (
        stalled_fall_candidate
        & (old_stall_steps < stall_threshold_steps)
        & (stall_steps >= stall_threshold_steps)
    )

    in_head_window = (new_accum > _HEAD_LATCH_LO) & (new_accum < _HEAD_LATCH_HI)
    old_head_latch = torch.where(
        transitioned_now | reposition_triggered,
        torch.zeros_like(env._roll_sprint_head_latch),
        env._roll_sprint_head_latch,
    )
    new_head_latch = old_head_latch | (
        active
        & ~locked_before
        & head_contact
        & support
        & in_head_window
        & _head_top_down(env, asset)
    )

    orientation_violation = (
        active
        & ~locked_before
        & support
        & (positive_rate >= _ROLL_SPRINT_MIN_FORWARD_RATE)
        & (lateral_z > _ROLL_SPRINT_LATERAL_INVALID_Z)
    )
    corridor_violation = (
        active
        & ~locked_before
        & (lateral_position.abs() > env._roll_sprint_lane_half_width_m)
    )
    heading_violation = (
        active
        & ~locked_before
        & (
            heading_error_abs
            > _ROLL_SPRINT_REPOSITION_HEADING_TRIGGER_RAD
        )
    )
    lateral_violation = (
        orientation_violation | corridor_violation | heading_violation
    )
    old_lateral_invalid = torch.where(
        transitioned_now | reposition_triggered,
        torch.zeros_like(env._roll_sprint_lateral_invalid),
        env._roll_sprint_lateral_invalid,
    )
    new_lateral_invalid = old_lateral_invalid | lateral_violation

    cycle_eligible = env._roll_sprint_cycle_eligible | transitioned_now
    completed = active & ~locked_before & (new_accum >= _ROLL_SPRINT_TARGET_ANGLE)
    bootstrap_completion = completed & ~cycle_eligible
    valid_completion = (
        completed & cycle_eligible & new_head_latch & ~new_lateral_invalid
    )
    invalid_completion = completed & cycle_eligible & ~valid_completion
    any_completion = valid_completion | invalid_completion | bootstrap_completion

    (
        directional_bootstrap_frontier,
        directional_position_frontier,
        directional_bootstrap_delta,
    ) = (
        _roll_sprint_directional_bootstrap_update(
            env._roll_sprint_directional_bootstrap_frontier,
            env._roll_sprint_directional_position_frontier,
            new_accum,
            progress_delta,
            position,
            cycle_start_position,
            active
            & (env._roll_sprint_roll_direction < 0.0)
            & rotation_eligible
            & ~new_lateral_invalid
            & ~any_completion,
        )
    )
    env._roll_sprint_directional_bootstrap_delta = torch.where(
        active,
        directional_bootstrap_delta,
        env._roll_sprint_directional_bootstrap_delta,
    )

    completed_distance = _roll_sprint_bounded_cycle_credit(
        new_accum,
        position,
        cycle_start_position,
        forward_frontier,
    )

    env._roll_sprint_completed_now = torch.where(
        active, valid_completion, env._roll_sprint_completed_now
    )
    env._roll_sprint_invalid_now = torch.where(
        active, invalid_completion, env._roll_sprint_invalid_now
    )
    env._roll_sprint_bootstrap_completed_now = torch.where(
        active, bootstrap_completion, env._roll_sprint_bootstrap_completed_now
    )
    env._roll_sprint_completed_distance = torch.where(
        active & valid_completion,
        completed_distance,
        torch.where(
            active,
            torch.zeros_like(completed_distance),
            env._roll_sprint_completed_distance,
        ),
    )
    env._roll_sprint_credited_distance += torch.where(
        active & valid_completion,
        completed_distance,
        torch.zeros_like(completed_distance),
    )
    env._roll_sprint_completed += valid_completion.float()
    recovered_rerolled_now = valid_completion & env._roll_sprint_recovered_cycle_armed
    env._roll_sprint_recovered_and_rerolled += recovered_rerolled_now.float()
    env._roll_sprint_recovered_rerolled_now = torch.where(
        active,
        recovered_rerolled_now,
        env._roll_sprint_recovered_rerolled_now,
    )

    env._roll_sprint_frontier_after_self_right += torch.where(
        valid_completion & env._roll_sprint_recovered_cycle_armed,
        completed_distance,
        torch.zeros_like(completed_distance),
    )

    completion_needs_self_right = (
        valid_completion | bootstrap_completion
    ) & ~launch_ready
    begin_self_right = (
        (completion_needs_self_right | stalled_fall_now) & ~self_right_before
    )
    post_self_right_reposition = self_righted_now & (
        (lateral_position.abs() > _ROLL_SPRINT_REPOSITION_REARM_M)
        | (heading_error_abs > _ROLL_SPRINT_REPOSITION_HEADING_REARM_RAD)
    )
    completion_reposition = (
        (valid_completion | bootstrap_completion)
        & launch_ready
        & (
            (lateral_position.abs() > _ROLL_SPRINT_REPOSITION_REARM_M)
            | (heading_error_abs > _ROLL_SPRINT_REPOSITION_HEADING_REARM_RAD)
        )
    )
    reposition_started = (
        reposition_triggered | post_self_right_reposition | completion_reposition
    )
    env._roll_sprint_self_right_started_now = torch.where(
        active,
        begin_self_right,
        env._roll_sprint_self_right_started_now,
    )
    env._roll_sprint_self_right_attempt_count += begin_self_right.float()

    # Valid and bootstrap completions require a launch-ready recovery. A
    # non-upright completion enters self-righting immediately. A successful
    # self-right either rearms in-lane or hands off to lane repositioning.
    state_reset = (
        any_completion
        | transitioned_now
        | reposition_started
        | begin_self_right
    )
    next_accum = torch.where(
        state_reset,
        torch.zeros_like(new_accum),
        new_accum,
    )
    env._roll_sprint_accum = next_accum
    env._roll_sprint_phase_frontier = torch.where(
        state_reset,
        torch.zeros_like(new_phase_frontier),
        new_phase_frontier,
    )
    env._roll_sprint_directional_bootstrap_frontier = torch.where(
        state_reset,
        torch.zeros_like(directional_bootstrap_frontier),
        directional_bootstrap_frontier,
    )
    env._roll_sprint_directional_position_frontier = torch.where(
        state_reset,
        torch.zeros_like(directional_position_frontier),
        directional_position_frontier,
    )
    env._roll_sprint_forward_frontier = torch.where(
        active & valid_completion,
        torch.maximum(forward_frontier, position),
        torch.where(active, forward_frontier, env._roll_sprint_forward_frontier),
    )
    env._roll_sprint_cycle_start_position = torch.where(
        active & (invalid_completion | transitioned_now),
        position,
        torch.where(
            active,
            cycle_start_position,
            env._roll_sprint_cycle_start_position,
        ),
    )
    env._roll_sprint_lateral_origin = torch.where(
        active, lateral_origin, env._roll_sprint_lateral_origin
    )
    env._roll_sprint_head_latch = torch.where(
        state_reset,
        torch.zeros_like(new_head_latch),
        new_head_latch,
    )
    env._roll_sprint_lateral_invalid = torch.where(
        state_reset,
        torch.zeros_like(new_lateral_invalid),
        new_lateral_invalid,
    )
    env._roll_sprint_cycle_eligible = torch.where(
        valid_completion | bootstrap_completion | stalled_fall_now | reposition_started,
        torch.zeros_like(cycle_eligible),
        torch.where(transitioned_now, torch.ones_like(cycle_eligible), cycle_eligible),
    )
    env._roll_sprint_awaiting_recovery = torch.where(
        valid_completion | bootstrap_completion | stalled_fall_now,
        torch.ones_like(awaiting_before),
        torch.where(
            transitioned_now,
            torch.zeros_like(awaiting_before),
            awaiting_before,
        ),
    )
    env._roll_sprint_awaiting_reposition = torch.where(
        reposition_started,
        torch.ones_like(reposition_before),
        torch.where(
            transitioned_now,
            torch.zeros_like(reposition_before),
            reposition_before,
        ),
    )
    env._roll_sprint_recovered_cycle_armed = torch.where(
        valid_completion | bootstrap_completion | stalled_fall_now,
        torch.zeros_like(env._roll_sprint_recovered_cycle_armed),
        torch.where(
            recovered_now,
            torch.ones_like(env._roll_sprint_recovered_cycle_armed),
            env._roll_sprint_recovered_cycle_armed,
        ),
    )
    next_self_righting = (self_right_before | begin_self_right) & ~self_righted_now
    env._roll_sprint_self_righting = torch.where(
        active,
        next_self_righting,
        env._roll_sprint_self_righting,
    )
    initialize_potential = active & next_self_righting & (
        begin_self_right | ~potential_ready
    )
    update_potential = active & next_self_righting
    env._roll_sprint_self_right_upright_previous = torch.where(
        update_potential,
        upright_potential,
        env._roll_sprint_self_right_upright_previous,
    )
    env._roll_sprint_self_right_height_previous = torch.where(
        update_potential,
        height_potential,
        env._roll_sprint_self_right_height_previous,
    )
    env._roll_sprint_self_right_potential_ready = torch.where(
        active,
        next_self_righting & (potential_ready | initialize_potential),
        env._roll_sprint_self_right_potential_ready,
    )
    env._roll_sprint_self_right_stall_steps = torch.where(
        active,
        torch.where(begin_self_right, torch.zeros_like(stall_steps), stall_steps),
        env._roll_sprint_self_right_stall_steps,
    )
    env._roll_sprint_self_right_hold_steps = torch.where(
        begin_self_right | self_righted_now,
        torch.zeros_like(env._roll_sprint_self_right_hold_steps),
        env._roll_sprint_self_right_hold_steps,
    )
    env._roll_sprint_self_right_latency_steps = torch.where(
        begin_self_right | self_righted_now,
        torch.zeros_like(env._roll_sprint_self_right_latency_steps),
        env._roll_sprint_self_right_latency_steps,
    )
    reset_recovery_clock = (
        transitioned_now
        | valid_completion
        | bootstrap_completion
        | stalled_fall_now
    )
    env._roll_sprint_recovery_hold_steps = torch.where(
        reset_recovery_clock,
        torch.zeros_like(env._roll_sprint_recovery_hold_steps),
        env._roll_sprint_recovery_hold_steps,
    )
    env._roll_sprint_recovery_latency_steps = torch.where(
        reset_recovery_clock,
        torch.zeros_like(env._roll_sprint_recovery_latency_steps),
        env._roll_sprint_recovery_latency_steps,
    )
    env._roll_sprint_reposition_latency_steps = torch.where(
        transitioned_now | reposition_started,
        torch.zeros_like(env._roll_sprint_reposition_latency_steps),
        env._roll_sprint_reposition_latency_steps,
    )
    env._roll_sprint_last_update_step = torch.where(
        active,
        torch.full_like(env._roll_sprint_last_update_step, step),
        env._roll_sprint_last_update_step,
    )


def _set_roll_sprint_ground_state(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    *,
    yaw_range: tuple[float, float],
    face_down_prob: float,
    face_up_prob: float,
    left_prob: float,
    right_prob: float,
    z_range: tuple[float, float] = (0.20, 0.25),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Place selected envs in four lying orientations and return codes/headings."""
    env_ids = env_ids.to(env.device, dtype=torch.long)
    count = len(env_ids)
    total = face_down_prob + face_up_prob + left_prob + right_prob
    if count == 0:
        return (
            torch.zeros(0, dtype=torch.long, device=env.device),
            torch.zeros((0, 2), device=env.device),
        )
    if total <= 0.0:
        raise ValueError("ground recovery orientation probabilities need positive mass")

    yaw = (
        torch.rand(count, device=env.device) * (yaw_range[1] - yaw_range[0])
        + yaw_range[0]
    )
    cy = torch.cos(0.5 * yaw)
    sy = torch.sin(0.5 * yaw)
    s = 2.0**-0.5
    face_down = torch.stack((s * cy, -s * sy, s * cy, s * sy), dim=1)
    face_up = torch.stack((s * cy, s * sy, -s * cy, s * sy), dim=1)
    left = torch.stack((s * cy, s * cy, s * sy, s * sy), dim=1)
    right = torch.stack((s * cy, -s * cy, -s * sy, s * sy), dim=1)

    sample = torch.rand(count, device=env.device) * total
    fd_limit = face_down_prob
    fu_limit = fd_limit + face_up_prob
    left_limit = fu_limit + left_prob
    is_fd = sample < fd_limit
    is_fu = (sample >= fd_limit) & (sample < fu_limit)
    is_left = (sample >= fu_limit) & (sample < left_limit)
    orientation = torch.full((count,), 4, dtype=torch.long, device=env.device)
    orientation[is_fd] = 1
    orientation[is_fu] = 2
    orientation[is_left] = 3

    quat = right
    quat[is_fd] = face_down[is_fd]
    quat[is_fu] = face_up[is_fu]
    quat[is_left] = left[is_left]
    height = (
        torch.rand(count, device=env.device) * (z_range[1] - z_range[0])
        + z_range[0]
    )
    env.sim.data.qpos[env_ids, 2] = height
    env.sim.data.qpos[env_ids, 3:7] = quat
    env.sim.data.qvel[env_ids, :6] = 0.0
    heading = torch.stack((torch.cos(yaw), torch.sin(yaw)), dim=1)
    return orientation, heading


def reset_roll_sprint_state(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    standing_prob: float = 0.50,
    midroll_prob: float = 0.25,
    postroll_prob: float = 0.25,
    crouch_prob: float = 0.0,
    ground_recovery_prob: float = 0.0,
    ground_face_down_prob: float = 0.25,
    ground_face_up_prob: float = 0.25,
    ground_left_prob: float = 0.25,
    ground_right_prob: float = 0.25,
    standing_z_min: float = 0.11,
    standing_z_max: float = 0.12,
    standing_tilt_max: float = 0.0,
    yaw_range: tuple = (-math.pi, math.pi),
    forward_vel_range: tuple = (0.0, 0.0),
    midroll_pitch_min: float = math.radians(50.0),
    midroll_pitch_max: float = math.radians(340.0),
    midroll_z_min: float = 0.05,
    midroll_z_max: float = 0.10,
    midroll_omega_range: tuple = (0.0, 3.0),
    midroll_forward_vel_range: tuple = (0.0, 0.0),
    postroll_pitch_min: float = math.radians(300.0),
    postroll_pitch_max: float = math.radians(350.0),
    postroll_omega_range: tuple = (0.0, 2.5),
    tuck_overrides: Optional[dict] = None,
    tuck_factor_range: tuple = (0.3, 1.0),
    joint_noise_std: float = 0.08,
    road_interior_prob: float = 0.70,
    road_edge_prob: float = 0.20,
    road_return_prob: float = 0.10,
    recovery_road_return_prob: float = 0.0,
    heading_return_prob: float = 0.0,
    heading_return_min_rad: float = _ROLL_SPRINT_HEADING_RETURN_RESET_MIN_RAD,
    heading_return_max_rad: float = _ROLL_SPRINT_HEADING_RETURN_RESET_MAX_RAD,
    roll_direction: float = 1.0,
) -> None:
    """Reset pose plus every cyclic sprint buffer for the selected envs."""
    if env_ids is None or len(env_ids) == 0:
        return
    if roll_direction not in (-1.0, 1.0):
        raise ValueError("roll_direction must be -1.0 or 1.0")
    env_ids = env_ids.to(env.device, dtype=torch.long)
    total = (
        standing_prob
        + midroll_prob
        + postroll_prob
        + crouch_prob
        + ground_recovery_prob
    )
    if total <= 0.0:
        raise ValueError("roll-sprint spawn probabilities must have positive mass")
    sample = torch.rand(len(env_ids), device=env.device) * max(total, 1.0e-6)
    is_standing = sample < standing_prob
    is_midroll = (sample >= standing_prob) & (sample < standing_prob + midroll_prob)
    postroll_limit = standing_prob + midroll_prob + postroll_prob
    crouch_limit = postroll_limit + crouch_prob
    is_postroll = (sample >= standing_prob + midroll_prob) & (sample < postroll_limit)
    is_crouch = (sample >= postroll_limit) & (sample < crouch_limit)
    is_ground = ~(is_standing | is_midroll | is_postroll | is_crouch)

    def _reset_bucket(
        selected: torch.Tensor,
        *,
        standing: bool,
        pitch_range: tuple[float, float] | None = None,
        omega_range: tuple[float, float] | None = None,
    ) -> None:
        bucket_ids = env_ids[selected]
        if len(bucket_ids) == 0:
            return
        reset_roulade_state(
            env,
            bucket_ids,
            asset_cfg=asset_cfg,
            standing_prob=1.0 if standing else 0.0,
            midroll_prob=0.0 if standing else 1.0,
            standing_z_min=standing_z_min,
            standing_z_max=standing_z_max,
            standing_tilt_max=standing_tilt_max,
            yaw_range=yaw_range,
            forward_vel_range=forward_vel_range,
            midroll_pitch_min=(pitch_range or (midroll_pitch_min, midroll_pitch_max))[
                0
            ],
            midroll_pitch_max=(pitch_range or (midroll_pitch_min, midroll_pitch_max))[
                1
            ],
            midroll_z_min=midroll_z_min,
            midroll_z_max=midroll_z_max,
            midroll_omega_range=omega_range or midroll_omega_range,
            midroll_forward_vel_range=midroll_forward_vel_range,
            tuck_overrides=tuck_overrides,
            tuck_factor_range=tuck_factor_range,
            joint_noise_std=joint_noise_std,
            roll_direction=roll_direction,
        )

    _reset_bucket(is_standing | is_crouch | is_ground, standing=True)
    _reset_bucket(is_midroll, standing=False)
    _reset_bucket(
        is_postroll,
        standing=False,
        pitch_range=(postroll_pitch_min, postroll_pitch_max),
        omega_range=postroll_omega_range,
    )
    crouch_ids = env_ids[is_crouch]
    if len(crouch_ids) > 0:
        set_random_crouch_state(env, crouch_ids, asset_cfg=asset_cfg)
    ground_ids = env_ids[is_ground]
    ground_orientation, ground_heading = _set_roll_sprint_ground_state(
        env,
        ground_ids,
        yaw_range=yaw_range,
        face_down_prob=ground_face_down_prob,
        face_up_prob=ground_face_up_prob,
        left_prob=ground_left_prob,
        right_prob=ground_right_prob,
    )

    spawn_accum = torch.zeros(len(env_ids), device=env.device)
    # The physical reverse pose and angular velocity remain signed through
    # ``roll_direction``. Phase progress is a magnitude, however: a reverse
    # mid-roll must start at its already-covered positive angle rather than
    # spending the first half-cycle climbing from a negative accumulator.
    # Keeping this buffer non-negative lets the shared completion gate require
    # the same full 2*pi turn for forward and reverse policies.
    spawn_accum[is_midroll] = env._roulade_accum[env_ids[is_midroll]].abs()
    spawn_cycle_eligible = is_standing.clone()
    spawn_self_righting = is_postroll | is_crouch | is_ground
    spawn_kind = torch.zeros(len(env_ids), dtype=torch.long, device=env.device)
    spawn_kind[is_midroll] = 6
    spawn_kind[is_postroll] = 7
    spawn_kind[is_crouch] = 5
    spawn_kind[is_ground] = ground_orientation
    road_total = road_interior_prob + road_edge_prob + road_return_prob
    if road_total <= 0.0:
        raise ValueError("roll-sprint road start probabilities need positive mass")
    if not 0.0 <= recovery_road_return_prob <= 1.0:
        raise ValueError("recovery road-return probability must be in [0, 1]")
    if not 0.0 <= heading_return_prob <= 1.0:
        raise ValueError("heading-return probability must be in [0, 1]")
    if not 0.0 <= heading_return_min_rad <= heading_return_max_rad:
        raise ValueError("heading-return range must be non-negative and ordered")
    road_sample = torch.rand(len(env_ids), device=env.device) * road_total
    is_road_interior = road_sample < road_interior_prob
    is_road_edge = (road_sample >= road_interior_prob) & (
        road_sample < road_interior_prob + road_edge_prob
    )
    # A69 never completed a road reposition and side recoveries never rerolled.
    # Train the combined self-right -> road-return handoff directly, rather
    # than sampling those states only in isolation. Forced cases still use the
    # normal return geometry below.
    force_recovery_road_return = spawn_self_righting & (
        torch.rand(len(env_ids), device=env.device) < recovery_road_return_prob
    )
    force_heading_return = torch.zeros(
        len(env_ids), dtype=torch.bool, device=env.device
    )
    if heading_return_prob > 0.0:
        force_heading_return = (
            torch.rand(len(env_ids), device=env.device) < heading_return_prob
        )
    is_road_interior &= ~force_recovery_road_return
    is_road_edge &= ~force_recovery_road_return
    road_sign = torch.where(
        torch.rand(len(env_ids), device=env.device) < 0.5,
        -torch.ones(len(env_ids), device=env.device),
        torch.ones(len(env_ids), device=env.device),
    )
    road_lateral_position = (
        torch.rand(len(env_ids), device=env.device) * 0.70 - 0.35
    )
    road_lateral_position = torch.where(
        is_road_edge,
        road_sign
        * (
            _ROLL_SPRINT_ROAD_SAFE_HALF_WIDTH
            + torch.rand(len(env_ids), device=env.device) * 0.07
        ),
        road_lateral_position,
    )
    road_lateral_position = torch.where(
        ~(is_road_interior | is_road_edge),
        road_sign
        * (
            0.51
            + torch.rand(len(env_ids), device=env.device) * 0.13
        ),
        road_lateral_position,
    )
    spawn_awaiting_reposition = (
        road_lateral_position.abs() > _ROLL_SPRINT_REPOSITION_TRIGGER_M
    ) | force_heading_return
    heading_return_sign = torch.where(
        torch.rand(len(env_ids), device=env.device) < 0.5,
        -torch.ones(len(env_ids), device=env.device),
        torch.ones(len(env_ids), device=env.device),
    )
    spawn_heading_error = torch.where(
        force_heading_return,
        heading_return_sign
        * (
            heading_return_min_rad
            + torch.rand(len(env_ids), device=env.device)
            * (heading_return_max_rad - heading_return_min_rad)
        ),
        torch.zeros(len(env_ids), device=env.device),
    )
    spawn_cycle_eligible &= ~spawn_awaiting_reposition
    _reset_roll_sprint_buffers(
        env,
        env_ids,
        spawn_accum=spawn_accum,
        spawn_head_latch=torch.zeros(len(env_ids), dtype=torch.bool, device=env.device),
        spawn_cycle_eligible=spawn_cycle_eligible,
        spawn_awaiting_recovery=spawn_self_righting,
        spawn_awaiting_reposition=spawn_awaiting_reposition,
        spawn_self_righting=spawn_self_righting,
        spawn_recovery_kind=spawn_kind,
        spawn_course_lateral_position=road_lateral_position,
        spawn_heading_error_rad=spawn_heading_error,
        roll_direction=roll_direction,
    )
    if len(ground_ids) > 0:
        env._roll_sprint_reset_heading_override_w[ground_ids] = ground_heading
        env._roll_sprint_reset_heading_override_valid[ground_ids] = True
    # Raw qpos writes above do not necessarily refresh ``asset.data`` before
    # this reset event returns. Keep the intended road coordinate in the
    # buffer, then freeze heading and course center from the first post-reset
    # physics state in ``_update_roll_sprint_state``. Canonical evaluation
    # helpers call ``sim.forward()`` and initialize their fixed +x frame
    # explicitly, so they remain exactly deterministic.


def _reset_roll_sprint_buffers(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    spawn_accum: torch.Tensor | None = None,
    spawn_head_latch: torch.Tensor | None = None,
    spawn_cycle_eligible: torch.Tensor | None = None,
    spawn_awaiting_recovery: torch.Tensor | None = None,
    spawn_awaiting_reposition: torch.Tensor | None = None,
    spawn_self_righting: torch.Tensor | None = None,
    spawn_recovery_kind: torch.Tensor | None = None,
    spawn_course_lateral_position: torch.Tensor | None = None,
    spawn_heading_error_rad: torch.Tensor | None = None,
    roll_direction: float = 1.0,
) -> None:
    """Reset only cyclic sprint bookkeeping, preserving other env state."""
    _roll_sprint_state(env)
    env_ids = env_ids.to(env.device, dtype=torch.long)
    if spawn_accum is None:
        spawn_accum = torch.zeros(len(env_ids), device=env.device)
    if spawn_head_latch is None:
        spawn_head_latch = torch.zeros(
            len(env_ids), dtype=torch.bool, device=env.device
        )
    if spawn_cycle_eligible is None:
        spawn_cycle_eligible = spawn_accum <= 1.0e-6
    if spawn_awaiting_recovery is None:
        spawn_awaiting_recovery = torch.zeros(
            len(env_ids), dtype=torch.bool, device=env.device
        )
    if spawn_self_righting is None:
        spawn_self_righting = torch.zeros(
            len(env_ids), dtype=torch.bool, device=env.device
        )
    if spawn_awaiting_reposition is None:
        spawn_awaiting_reposition = torch.zeros(
            len(env_ids), dtype=torch.bool, device=env.device
        )
    if spawn_recovery_kind is None:
        spawn_recovery_kind = torch.zeros(
            len(env_ids), dtype=torch.long, device=env.device
        )
    if spawn_course_lateral_position is None:
        spawn_course_lateral_position = torch.zeros(len(env_ids), device=env.device)
    if spawn_heading_error_rad is None:
        spawn_heading_error_rad = torch.zeros(len(env_ids), device=env.device)
    env._roll_sprint_accum[env_ids] = spawn_accum
    env._roll_sprint_phase_frontier[env_ids] = spawn_accum
    env._roll_sprint_head_latch[env_ids] = spawn_head_latch
    env._roll_sprint_cycle_eligible[env_ids] = spawn_cycle_eligible
    env._roll_sprint_awaiting_recovery[env_ids] = spawn_awaiting_recovery
    env._roll_sprint_awaiting_reposition[env_ids] = spawn_awaiting_reposition
    env._roll_sprint_self_righting[env_ids] = spawn_self_righting
    env._roll_sprint_self_right_started_now[env_ids] = False
    env._roll_sprint_self_righted_now[env_ids] = False
    env._roll_sprint_self_right_stall_steps[env_ids] = 0
    env._roll_sprint_self_right_hold_steps[env_ids] = 0
    env._roll_sprint_self_right_latency_steps[env_ids] = 0
    env._roll_sprint_self_right_latency_total_steps[env_ids] = 0.0
    env._roll_sprint_self_right_attempt_count[env_ids] = spawn_self_righting.float()
    env._roll_sprint_self_right_success_count[env_ids] = 0.0
    env._roll_sprint_self_right_upright_delta[env_ids] = 0.0
    env._roll_sprint_self_right_height_delta[env_ids] = 0.0
    env._roll_sprint_self_right_upright_previous[env_ids] = 0.0
    env._roll_sprint_self_right_height_previous[env_ids] = 0.0
    env._roll_sprint_self_right_potential_ready[env_ids] = False
    env._roll_sprint_frontier_after_self_right[env_ids] = 0.0
    env._roll_sprint_recovery_spawn_kind[env_ids] = spawn_recovery_kind
    env._roll_sprint_reposition_latency_steps[env_ids] = 0
    env._roll_sprint_reposition_latency_total_steps[env_ids] = 0.0
    env._roll_sprint_reposition_count[env_ids] = 0.0
    env._roll_sprint_recovery_hold_steps[env_ids] = 0
    env._roll_sprint_recovery_latency_steps[env_ids] = 0
    env._roll_sprint_recovery_latency_total_steps[env_ids] = 0.0
    env._roll_sprint_recovery_count[env_ids] = 0.0
    env._roll_sprint_recovered_cycle_armed[env_ids] = False
    env._roll_sprint_recovered_and_rerolled[env_ids] = 0.0
    env._roll_sprint_recovered_now[env_ids] = False
    env._roll_sprint_repositioned_now[env_ids] = False
    env._roll_sprint_recovered_rerolled_now[env_ids] = False
    env._roll_sprint_lateral_invalid[env_ids] = False
    env._roll_sprint_completed[env_ids] = 0.0
    env._roll_sprint_forward_position[env_ids] = 0.0
    env._roll_sprint_forward_origin[env_ids] = 0.0
    env._roll_sprint_cycle_start_position[env_ids] = 0.0
    env._roll_sprint_forward_frontier[env_ids] = 0.0
    env._roll_sprint_lateral_origin[env_ids] = 0.0
    env._roll_sprint_lateral_displacement[env_ids] = 0.0
    env._roll_sprint_lane_centering_delta[env_ids] = 0.0
    env._roll_sprint_course_center_xy_w[env_ids] = 0.0
    env._roll_sprint_course_lateral_position[env_ids] = (
        spawn_course_lateral_position
    )
    env._roll_sprint_roll_direction[env_ids] = roll_direction
    env._roll_sprint_body_heading_w[env_ids] = 0.0
    env._roll_sprint_heading_w[env_ids] = 0.0
    env._roll_sprint_heading_ready[env_ids] = False
    env._roll_sprint_reset_heading_override_w[env_ids] = 0.0
    env._roll_sprint_reset_heading_override_valid[env_ids] = False
    env._roll_sprint_spawn_heading_error_rad[env_ids] = spawn_heading_error_rad
    env._roll_sprint_heading_error_rad[env_ids] = 0.0
    env._roll_sprint_heading_alignment_delta[env_ids] = 0.0
    env._roll_sprint_progress_delta[env_ids] = 0.0
    env._roll_sprint_directional_bootstrap_frontier[env_ids] = 0.0
    env._roll_sprint_directional_position_frontier[env_ids] = 0.0
    env._roll_sprint_directional_bootstrap_delta[env_ids] = 0.0
    env._roll_sprint_completed_distance[env_ids] = 0.0
    env._roll_sprint_credited_distance[env_ids] = 0.0
    env._roll_sprint_completed_now[env_ids] = False
    env._roll_sprint_invalid_now[env_ids] = False
    env._roll_sprint_bootstrap_completed_now[env_ids] = False
    env._roll_sprint_last_update_step[env_ids] = -1


def arrange_roll_sprint_race_start(
    env: ManagerBasedRlEnv,
    lane_spacing: float,
    roll_direction: float = 1.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Align every canonical race robot on one deterministic world +x line.

    Backroll robots face world -x while their credited course remains world
    +x, making the reverse direction visible without changing the race track.
    """
    if roll_direction not in (-1.0, 1.0):
        raise ValueError("roll_direction must be -1.0 or 1.0")
    asset: Entity = env.scene[asset_cfg.name]
    old_origins = env.scene.terrain.env_origins.clone()
    lane_indices = torch.arange(env.num_envs, device=env.device, dtype=old_origins.dtype)
    lane_indices -= (env.num_envs - 1) / 2.0
    origins = torch.zeros_like(old_origins)
    origins[:, 1] = lane_indices * lane_spacing

    pose = torch.cat(
        (asset.data.root_link_pos_w.clone(), asset.data.root_link_quat_w.clone()),
        dim=-1,
    )
    local_height = pose[:, 2] - old_origins[:, 2]
    pose[:, :3] = origins
    pose[:, 2] += local_height
    pose[:, 3:] = 0.0
    if roll_direction > 0.0:
        pose[:, 3] = 1.0
    else:
        pose[:, 6] = 1.0
    velocity = torch.zeros((env.num_envs, 6), device=env.device, dtype=pose.dtype)

    env.scene.terrain.env_origins.copy_(origins)
    asset.write_root_link_pose_to_sim(pose)
    asset.write_root_link_velocity_to_sim(velocity)
    env.sim.forward()

    env_ids = torch.arange(env.num_envs, device=env.device)
    course_lateral_position = lane_indices * lane_spacing
    _reset_roll_sprint_buffers(
        env,
        env_ids,
        spawn_course_lateral_position=course_lateral_position,
        roll_direction=roll_direction,
    )
    body_heading = _roll_sprint_heading(asset)
    heading = body_heading * roll_direction
    env._roll_sprint_course_center_xy_w.zero_()
    forward_position = _roll_sprint_forward_position(env, asset, heading)
    lateral_position = _roll_sprint_lateral_position(env, asset, heading)
    env._roll_sprint_body_heading_w.copy_(body_heading)
    env._roll_sprint_heading_w.copy_(heading)
    env._roll_sprint_heading_ready[:] = True
    env._roll_sprint_heading_error_rad.zero_()
    env._roll_sprint_heading_alignment_delta.zero_()
    env._roll_sprint_forward_position.copy_(forward_position)
    env._roll_sprint_forward_origin.copy_(forward_position)
    env._roll_sprint_cycle_start_position.copy_(forward_position)
    env._roll_sprint_forward_frontier.copy_(forward_position)
    env._roll_sprint_lateral_origin.copy_(lateral_position)
    env._roll_sprint_lateral_displacement.zero_()
    env._roll_sprint_course_lateral_position.copy_(lateral_position)
    env._roll_sprint_lane_centering_delta.zero_()
    return origins


def arrange_roll_sprint_recovery_start(
    env: ManagerBasedRlEnv,
    lane_spacing: float,
    seed: int = 0,
    orientations: tuple[str, ...] = ("face_down", "face_up", "left", "right"),
    roll_direction: float = 1.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Arrange deterministic four-orientation self-right and reroll starts."""
    asset: Entity = env.scene[asset_cfg.name]
    if roll_direction not in (-1.0, 1.0):
        raise ValueError("roll_direction must be -1.0 or 1.0")
    if len(orientations) != env.num_envs:
        raise ValueError("one recovery orientation is required for every environment")
    orientation_to_code = {"face_down": 1, "face_up": 2, "left": 3, "right": 4}
    try:
        codes = torch.tensor(
            [orientation_to_code[name] for name in orientations],
            dtype=torch.long,
            device=env.device,
        )
    except KeyError as error:
        raise ValueError(f"unsupported recovery orientation: {error.args[0]}") from error

    old_origins = env.scene.terrain.env_origins.clone()
    lane_indices = torch.arange(env.num_envs, device=env.device, dtype=old_origins.dtype)
    lane_indices -= (env.num_envs - 1) / 2.0
    origins = torch.zeros_like(old_origins)
    origins[:, 1] = lane_indices * lane_spacing
    pose = torch.zeros((env.num_envs, 7), device=env.device, dtype=old_origins.dtype)
    pose[:, :3] = origins
    pose[:, 2] += 0.20 + 0.002 * float(seed % 4)
    s = 2.0**-0.5
    quaternions = {
        1: (s, 0.0, s, 0.0),
        2: (s, 0.0, -s, 0.0),
        3: (s, s, 0.0, 0.0),
        4: (s, -s, 0.0, 0.0),
    }
    for code, quat in quaternions.items():
        pose[codes == code, 3:] = torch.tensor(
            quat,
            device=env.device,
            dtype=pose.dtype,
        )
    if roll_direction < 0.0:
        # Pre-multiply every lying pose by a deterministic world-yaw of pi.
        w, x, y, z = pose[:, 3:].unbind(dim=-1)
        pose[:, 3:] = torch.stack((-z, -y, x, w), dim=-1)
    velocity = torch.zeros((env.num_envs, 6), device=env.device, dtype=pose.dtype)

    joint_pos = asset.data.default_joint_pos.clone()
    joint_vel = torch.zeros_like(joint_pos)
    servo_ids = _servo_joint_ids(env, asset)
    phase = torch.arange(
        env.num_envs * len(servo_ids),
        device=env.device,
        dtype=joint_pos.dtype,
    ).reshape(env.num_envs, len(servo_ids))
    deterministic_offset = 0.025 * torch.sin(phase + float(seed) * 1.618)
    joint_pos[:, servo_ids] += deterministic_offset

    env.scene.terrain.env_origins.copy_(origins)
    asset.write_root_link_pose_to_sim(pose)
    asset.write_root_link_velocity_to_sim(velocity)
    asset.write_joint_state_to_sim(joint_pos, joint_vel)
    env.sim.forward()

    env_ids = torch.arange(env.num_envs, device=env.device)
    true = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    _reset_roll_sprint_buffers(
        env,
        env_ids,
        spawn_cycle_eligible=~true,
        spawn_awaiting_recovery=true,
        spawn_self_righting=true,
        spawn_recovery_kind=codes,
        spawn_course_lateral_position=lane_indices * lane_spacing,
        roll_direction=roll_direction,
    )
    body_heading = torch.zeros(
        (env.num_envs, 2), device=env.device, dtype=pose.dtype
    )
    body_heading[:, 0] = roll_direction
    heading = body_heading * roll_direction
    env._roll_sprint_course_center_xy_w.zero_()
    forward_position = _roll_sprint_forward_position(env, asset, heading)
    lateral_position = _roll_sprint_lateral_position(env, asset, heading)
    env._roll_sprint_body_heading_w.copy_(body_heading)
    env._roll_sprint_heading_w.copy_(heading)
    env._roll_sprint_heading_ready[:] = True
    env._roll_sprint_heading_error_rad.zero_()
    env._roll_sprint_heading_alignment_delta.zero_()
    env._roll_sprint_forward_position.copy_(forward_position)
    env._roll_sprint_forward_origin.copy_(forward_position)
    env._roll_sprint_cycle_start_position.copy_(forward_position)
    env._roll_sprint_forward_frontier.copy_(forward_position)
    env._roll_sprint_lateral_origin.copy_(lateral_position)
    env._roll_sprint_lateral_displacement.zero_()
    env._roll_sprint_course_lateral_position.copy_(lateral_position)
    env._roll_sprint_lane_centering_delta.zero_()
    return origins


def roll_sprint_progress(
    env: ManagerBasedRlEnv,
    max_paid_rate: float = 5.0,
    road_half_width: float = _ROLL_SPRINT_ROAD_HALF_WIDTH,
    road_safe_half_width: float = _ROLL_SPRINT_ROAD_SAFE_HALF_WIDTH,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Pay valid forward-roll progress, but never a known-invalid cycle."""
    asset = env.scene[asset_cfg.name]
    _update_roll_sprint_state(env, asset)
    paid_rate = torch.clamp(
        env._roll_sprint_progress_delta,
        max=max_paid_rate * env.step_dt,
    )
    active_road_half_width = getattr(
        env,
        "_roll_sprint_lane_half_width_m",
        road_half_width,
    )
    safe_half_width = min(road_safe_half_width, active_road_half_width)
    road_edge_fraction = torch.clamp(
        (
            env._roll_sprint_course_lateral_position.abs()
            - safe_half_width
        )
        / max(active_road_half_width - safe_half_width, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    road_quality = torch.clamp(
        1.0
        - road_edge_fraction.square(),
        min=0.0,
        max=1.0,
    )
    heading_quality = torch.clamp(
        1.0
        - (
            env._roll_sprint_heading_error_rad.abs()
            / _ROLL_SPRINT_REPOSITION_HEADING_TRIGGER_RAD
        ).square(),
        min=0.0,
        max=1.0,
    )
    # Bootstrap through the complete head-contact opportunity. Once that phase
    # has passed, the rest of the dense cycle is eligible only if the required
    # flat head-top contact was actually witnessed. Orientation/road-invalid
    # cycles stop paying immediately, and their completion step cannot leak a
    # final progress payment after the state machine resets its phase frontier.
    bootstrap_segment = (
        ~env._roll_sprint_cycle_eligible
        & ~env._roll_sprint_awaiting_recovery
        & ~env._roll_sprint_awaiting_reposition
        & ~env._roll_sprint_self_righting
    )
    head_phase_valid = (
        (env._roll_sprint_phase_frontier < _HEAD_LATCH_HI)
        | env._roll_sprint_head_latch
        | bootstrap_segment
    )
    cycle_valid = (
        head_phase_valid
        & ~env._roll_sprint_lateral_invalid
        & ~env._roll_sprint_invalid_now
    )
    return (
        road_quality
        * heading_quality
        * cycle_valid.float()
        * paid_rate
        / (env.step_dt * _ROLL_SPRINT_TARGET_ANGLE)
    )


def roll_sprint_distance(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Release only newly extended signed forward frontier on valid cycles."""
    asset = env.scene[asset_cfg.name]
    _update_roll_sprint_state(env, asset)
    return env._roll_sprint_completed_distance / env.step_dt


def roll_sprint_directional_bootstrap(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Pay bounded reverse-roll translation discovery, never the final score."""
    asset = env.scene[asset_cfg.name]
    _update_roll_sprint_state(env, asset)
    return env._roll_sprint_directional_bootstrap_delta / env.step_dt


def roll_sprint_straightness_penalty(
    env: ManagerBasedRlEnv,
    road_safe_half_width: float = _ROLL_SPRINT_ROAD_SAFE_HALF_WIDTH,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Charge only motion in the road-edge band during active roll progress."""
    asset = env.scene[asset_cfg.name]
    _update_roll_sprint_state(env, asset)
    road_edge_excess = torch.clamp(
        env._roll_sprint_course_lateral_position.abs() - road_safe_half_width,
        min=0.0,
    )
    roll_activity = torch.clamp(
        env._roll_sprint_progress_delta
        / (env.step_dt * _ROLL_SPRINT_TARGET_ANGLE),
        min=0.0,
        max=1.0,
    )
    return road_edge_excess * roll_activity


def _roll_sprint_roll_penalty_mask(env: ManagerBasedRlEnv) -> torch.Tensor:
    """One only while rolling, not during recovery or lane repositioning."""
    transition_mode = (
        env._roll_sprint_awaiting_recovery
        | env._roll_sprint_self_righting
        | env._roll_sprint_self_righted_now
        | env._roll_sprint_awaiting_reposition
    )
    return (~transition_mode).float()


def _roll_sprint_lane_reward_mask(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Pause road shaping only while getting up, not while repositioning."""
    getting_up = (
        env._roll_sprint_self_righting
        | env._roll_sprint_self_righted_now
        | (
            env._roll_sprint_awaiting_recovery
            & ~env._roll_sprint_awaiting_reposition
        )
    )
    return (~getting_up).float()


def roll_sprint_sagittal_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Apply roll-plane angular pressure only during roll attempts."""
    asset = env.scene[asset_cfg.name]
    _update_roll_sprint_state(env, asset)
    return roulade_sagittal_penalty(
        env, asset_cfg
    ) * _roll_sprint_roll_penalty_mask(env)


def roll_sprint_lateral_velocity_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Apply lateral-speed pressure only during roll attempts."""
    asset = env.scene[asset_cfg.name]
    _update_roll_sprint_state(env, asset)
    return roulade_lateral_velocity_penalty(
        env, asset_cfg
    ) * _roll_sprint_roll_penalty_mask(env)


def roll_sprint_flatness_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Apply shoulder/side pressure only during roll attempts."""
    asset = env.scene[asset_cfg.name]
    _update_roll_sprint_state(env, asset)
    return roulade_flatness_penalty(
        env, asset_cfg
    ) * _roll_sprint_roll_penalty_mask(env)


def roll_sprint_road_return_progress(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Potential reward for reducing road-edge excess, with no center pull."""
    asset = env.scene[asset_cfg.name]
    _update_roll_sprint_state(env, asset)
    return (
        env._roll_sprint_lane_centering_delta
        / env.step_dt
        * _roll_sprint_lane_reward_mask(env)
    )


def roll_sprint_lane_centering_progress(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Compatibility wrapper for the shared-road return potential."""
    return roll_sprint_road_return_progress(env, asset_cfg)


def roll_sprint_heading_alignment_progress(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Potential rate for reset-heading alignment, with no aligned annuity."""
    asset = env.scene[asset_cfg.name]
    _update_roll_sprint_state(env, asset)
    active = (
        ~env._roll_sprint_self_righting
        & ~env._roll_sprint_self_righted_now
    )
    return env._roll_sprint_heading_alignment_delta / env.step_dt * active.float()


def roll_sprint_reposition_command(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    lateral_speed: float = _ROLL_SPRINT_REPOSITION_LATERAL_COMMAND_MPS,
    lateral_gain: float = _ROLL_SPRINT_REPOSITION_LATERAL_KP,
    yaw_speed: float = _ROLL_SPRINT_REPOSITION_YAW_COMMAND_RPS,
    heading_gain: float = _ROLL_SPRINT_REPOSITION_HEADING_COMMAND_KP,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Expose self-right and shared-road return modes in existing twist slots.

    ``twist[0]`` is one only during self-righting. During explicit road
    repositioning, ``twist[1]`` gives the nearest-safe-edge return while
    ``twist[2]`` corrects yaw concurrently. It stays zero while self-righting
    so the actor is never asked to get upright and translate laterally at the
    same time. Concurrent correction avoids deadlocking side recoveries far
    from the road behind a heading-only gate.
    ``twist[2]`` continuously exposes the bounded reset-heading correction,
    including while self-righting, so yaw-invariant proprioception can hold a
    straight line. This preserves the shared 61D contract.
    """
    asset = env.scene[asset_cfg.name]
    _update_roll_sprint_state(env, asset)
    command = torch.zeros_like(env.command_manager.get_command(command_name))
    road_position = env._roll_sprint_course_lateral_position
    road_return_error = road_position - torch.clamp(
        road_position,
        min=-_ROLL_SPRINT_ROAD_SAFE_HALF_WIDTH,
        max=_ROLL_SPRINT_ROAD_SAFE_HALF_WIDTH,
    )
    needs_return = road_return_error.abs() > 0.0
    needs_heading_return = env._roll_sprint_heading_ready
    lateral_return = torch.clamp(
        -lateral_gain * road_return_error,
        min=-lateral_speed,
        max=lateral_speed,
    )
    heading_return = torch.clamp(
        heading_gain * env._roll_sprint_heading_error_rad,
        min=-yaw_speed,
        max=yaw_speed,
    )
    command[:, 0] = env._roll_sprint_self_righting.float()
    command[:, 1] = torch.where(
        needs_return
        & env._roll_sprint_awaiting_reposition
        & ~env._roll_sprint_self_righting,
        lateral_return,
        command[:, 1],
    )
    command[:, 2] = torch.where(
        needs_heading_return,
        heading_return,
        command[:, 2],
    )
    return command


def roll_sprint_backroll_direction_flag(
    env: ManagerBasedRlEnv,
    dim: int = 6,
) -> torch.Tensor:
    """Expose the dedicated reverse-roll mode in existing body padding.

    A standing forward start and a standing backroll start have identical
    proprioception.  One signed flag in the pre-existing six-dimensional
    body-command padding gives the separately trained backroll policy an
    unambiguous task cue without changing the 61D actor contract or any of
    the three twist-slot semantics.  The reverse value is -1 rather than +1
    so a warm-started forward champion receives the cue polarity that already
    produces toward-goal reverse motion; zero remains the forward/default
    padding value.
    """
    if dim < 1:
        raise ValueError("backroll direction flag needs at least one slot")
    _roll_sprint_state(env)
    command = torch.zeros(
        (env.num_envs, dim),
        dtype=torch.float32,
        device=env.device,
    )
    command[:, 0] = torch.where(
        env._roll_sprint_roll_direction < 0.0,
        -torch.ones(env.num_envs, device=env.device, dtype=command.dtype),
        torch.zeros(env.num_envs, device=env.device, dtype=command.dtype),
    )
    return command


def roll_sprint_lane_half_width_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    width_stages: list[dict],
) -> torch.Tensor:
    """Tighten the training cycle gate while canonical play stays exact."""
    del env_ids
    _roll_sprint_state(env)
    current_width = width_stages[0]["width"]
    for stage in width_stages:
        if env.common_step_counter >= stage["step"]:
            current_width = stage["width"]
    env._roll_sprint_lane_half_width_m = float(current_width)
    return torch.tensor([current_width], device=env.device)


def roll_sprint_cycle_rate(
    env: ManagerBasedRlEnv,
    include_bootstrap: bool = False,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Low-weight completion-rate reward for valid or skill-bootstrap cycles."""
    asset = env.scene[asset_cfg.name]
    _update_roll_sprint_state(env, asset)
    completed = env._roll_sprint_completed_now
    if include_bootstrap:
        completed = completed | env._roll_sprint_bootstrap_completed_now
    return completed.float() / env.step_dt


def roll_sprint_recovery_rate(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """One-shot recovery edge rate; never an upright per-step annuity."""
    asset = env.scene[asset_cfg.name]
    _update_roll_sprint_state(env, asset)
    return env._roll_sprint_recovered_now.float() / env.step_dt


def roll_sprint_reposition_rate(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """One-shot road-reposition completion rate, never a centering annuity."""
    asset = env.scene[asset_cfg.name]
    _update_roll_sprint_state(env, asset)
    return env._roll_sprint_repositioned_now.float() / env.step_dt


def roll_sprint_self_right_upright_progress(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Self-right-only cosine-tilt potential rate, with no upright annuity."""
    asset = env.scene[asset_cfg.name]
    _update_roll_sprint_state(env, asset)
    return env._roll_sprint_self_right_upright_delta / env.step_dt


def roll_sprint_self_right_height_progress(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Self-right-only capped trunk-height potential rate."""
    asset = env.scene[asset_cfg.name]
    _update_roll_sprint_state(env, asset)
    return env._roll_sprint_self_right_height_delta / env.step_dt


def roll_sprint_self_right_upward_velocity(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Bootstrap upward motion only while the self-right mode is active."""
    asset = env.scene[asset_cfg.name]
    _update_roll_sprint_state(env, asset)
    upward = torch.clamp(
        torch.nan_to_num(asset.data.root_link_lin_vel_w[:, 2], nan=0.0),
        min=0.0,
    )
    below_ceiling = (
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2]
        < _ROLL_SPRINT_SELF_RIGHT_HEIGHT_CEILING_M
    )
    return upward * env._roll_sprint_self_righting.float() * below_ceiling.float()


def roll_sprint_self_right_fallen_tax(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Positive self-right occupancy cost; configure it with negative weight."""
    asset = env.scene[asset_cfg.name]
    _update_roll_sprint_state(env, asset)
    return env._roll_sprint_self_righting.float()


def roll_sprint_self_right_success_rate(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """One-shot completed self-right edge rate."""
    asset = env.scene[asset_cfg.name]
    _update_roll_sprint_state(env, asset)
    return env._roll_sprint_self_righted_now.float() / env.step_dt


def roll_sprint_recovered_reroll_rate(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """One-shot rate for a valid roll completed after a latched recovery."""
    asset = env.scene[asset_cfg.name]
    _update_roll_sprint_state(env, asset)
    return env._roll_sprint_recovered_rerolled_now.float() / env.step_dt


def roll_sprint_recovery_count(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Cumulative feet-supported recoveries for episode-end metrics."""
    asset = env.scene[asset_cfg.name]
    _update_roll_sprint_state(env, asset)
    return env._roll_sprint_recovery_count


def roll_sprint_self_right_attempt_count(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Cumulative entries into explicit self-righting mode."""
    asset = env.scene[asset_cfg.name]
    _update_roll_sprint_state(env, asset)
    return env._roll_sprint_self_right_attempt_count


def roll_sprint_self_right_success_count(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Cumulative launch-ready self-right completions."""
    asset = env.scene[asset_cfg.name]
    _update_roll_sprint_state(env, asset)
    return env._roll_sprint_self_right_success_count


def roll_sprint_self_right_success_fraction(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Per-episode self-right success fraction."""
    asset = env.scene[asset_cfg.name]
    _update_roll_sprint_state(env, asset)
    attempts = env._roll_sprint_self_right_attempt_count.clamp_min(1.0)
    return env._roll_sprint_self_right_success_count / attempts


def roll_sprint_mean_self_right_latency(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Mean seconds from self-right entry to launch-ready completion."""
    asset = env.scene[asset_cfg.name]
    _update_roll_sprint_state(env, asset)
    successes = env._roll_sprint_self_right_success_count.clamp_min(1.0)
    return env._roll_sprint_self_right_latency_total_steps * env.step_dt / successes


def roll_sprint_frontier_after_self_right(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Cumulative valid frontier distance earned after a self-right/recovery."""
    asset = env.scene[asset_cfg.name]
    _update_roll_sprint_state(env, asset)
    return env._roll_sprint_frontier_after_self_right


def roll_sprint_recovered_reroll_count(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Cumulative valid rolls completed from a recovery-armed cycle."""
    asset = env.scene[asset_cfg.name]
    _update_roll_sprint_state(env, asset)
    return env._roll_sprint_recovered_and_rerolled


def roll_sprint_mean_recovery_latency(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Mean seconds from roll completion to the latched recovery edge."""
    asset = env.scene[asset_cfg.name]
    _update_roll_sprint_state(env, asset)
    count = env._roll_sprint_recovery_count.clamp_min(1.0)
    return env._roll_sprint_recovery_latency_total_steps * env.step_dt / count


def roll_sprint_reposition_count(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Cumulative completed lane-reposition transitions."""
    asset = env.scene[asset_cfg.name]
    _update_roll_sprint_state(env, asset)
    return env._roll_sprint_reposition_count


def roll_sprint_mean_reposition_latency(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Mean seconds from excessive drift to launch-ready lane return."""
    asset = env.scene[asset_cfg.name]
    _update_roll_sprint_state(env, asset)
    count = env._roll_sprint_reposition_count.clamp_min(1.0)
    return env._roll_sprint_reposition_latency_total_steps * env.step_dt / count


def roll_sprint_invalid_cycle_rate(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Event rate for eligible full turns that fail a structural roll gate."""
    asset = env.scene[asset_cfg.name]
    _update_roll_sprint_state(env, asset)
    return env._roll_sprint_invalid_now.float() / env.step_dt


def roll_sprint_head_pivot(
    env: ManagerBasedRlEnv,
    rate_norm: float = 2.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward actively rotating on the flat head top inside the cycle window."""
    asset = env.scene[asset_cfg.name]
    _update_roll_sprint_state(env, asset)
    contact = _sensor_any_contact(env, _ROLL_SPRINT_HEAD_SENSOR)
    support = _sensor_any_contact(env, _ROLL_SPRINT_SUPPORT_SENSOR)
    if contact is None or support is None:
        return torch.zeros(env.num_envs, device=env.device)
    in_window = (env._roll_sprint_accum > _HEAD_LATCH_LO) & (
        env._roll_sprint_accum < _HEAD_LATCH_HI
    )
    omega_fwd = torch.nan_to_num(
        _ROULADE_FWD_SIGN
        * env._roll_sprint_roll_direction
        * asset.data.root_link_ang_vel_b[:, 1],
        nan=0.0,
    )
    rate = torch.clamp(omega_fwd / rate_norm, 0.0, 1.0)
    return (
        contact.float()
        * support.float()
        * in_window.float()
        * rate
        * _head_top_down(env, asset).float()
    )

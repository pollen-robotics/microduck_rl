"""Reward terms for the approach-and-inspect task.

Drop this file into ``src/mjlab_microduck/tasks/`` and import it from the env
cfg as ``microduck_ai_mdp``. Nothing here touches the existing ``mdp.py``.

Conventions (from AGENTS.md — do not break these):

* Potential-based shaping pays ``Δ(potential)`` per step. Holding pays zero, so
  no state can farm it. Freshly reset envs (``episode_length_buf <= 1``) reseed
  their previous potential so there is no spurious delta across episodes.
* Anything named ``*_penalty`` returns ``<= 0`` and takes a POSITIVE weight.
* Every ``*_progress`` latch is slewed at a constant rate: being ahead of the
  ramp pays zero, so arriving fast is never a jackpot.
* Gates are hard state checks, never soft nudges. Positive rewards are never
  gated on a bad state (fallen, low).

The target object is a scene entity (default name ``"target"``) with
``data.root_link_pos_w``. The head forward direction is the vector from the
``jaw_soft`` body CoM to the ``mouth_tip`` site — the same pair
``apply_mouth_payload_force`` uses — so this needs no new sites.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:  # pragma: no cover
    from mjlab.entity import Entity
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_HEAD_CFG = SceneEntityCfg("robot", body_names=["jaw_soft"], site_names=["mouth_tip"])
_ROBOT_CFG = SceneEntityCfg("robot")


# ── geometry helpers ────────────────────────────────────────────────────────

def _planar_offset_to_target(env, asset_cfg, target_name):
    """(N,2) vector from robot base to target, world XY."""
    robot: Entity = env.scene[asset_cfg.name]
    target: Entity = env.scene[target_name]
    return target.data.root_link_pos_w[:, :2] - robot.data.root_link_pos_w[:, :2]


def _ring_error(env, asset_cfg, target_name, standoff):
    """|d - standoff|, (N,). Zero on the ring, grows either side."""
    d = torch.linalg.norm(_planar_offset_to_target(env, asset_cfg, target_name), dim=-1)
    return torch.abs(d - standoff)


def _resolved(env, cfg):
    """mjlab managers resolve a SceneEntityCfg only when it is passed in a term's
    ``params``; a module-level default keeps ``site_ids``/``body_ids`` as
    ``slice(None)``. Resolve lazily so the defaults work too."""
    if isinstance(cfg.site_ids, slice) or isinstance(cfg.body_ids, slice):
        cfg.resolve(env.scene)
    return cfg


def _head_bearing_error(env, head_cfg, target_name):
    """Planar angle (rad, >= 0) between where the head points and the target."""
    head_cfg = _resolved(env, head_cfg)
    robot: Entity = env.scene[head_cfg.name]
    target: Entity = env.scene[target_name]
    sid = int(head_cfg.site_ids[0])
    bid = int(head_cfg.body_ids[0])
    p_tip = robot.data.site_pos_w[:, sid, :2]
    p_head = robot.data.body_com_pos_w[:, bid, :2]
    fwd = p_tip - p_head
    to_t = target.data.root_link_pos_w[:, :2] - p_head
    fwd_ang = torch.atan2(fwd[:, 1], fwd[:, 0])
    tgt_ang = torch.atan2(to_t[:, 1], to_t[:, 0])
    err = tgt_ang - fwd_ang
    err = torch.atan2(torch.sin(err), torch.cos(err))  # wrap to [-pi, pi]
    return torch.abs(err)


def _planar_speed(env, asset_cfg):
    robot: Entity = env.scene[asset_cfg.name]
    return torch.linalg.norm(robot.data.root_link_lin_vel_w[:, :2], dim=-1)


def _fresh(env):
    return env.episode_length_buf <= 1


def _delta_potential(env, attr: str, potential: torch.Tensor) -> torch.Tensor:
    """Generic Δpotential with fresh-episode reseed. Stores prev on env.<attr>."""
    if not hasattr(env, attr):
        setattr(env, attr, potential.clone())
    prev = getattr(env, attr)
    fresh = _fresh(env)
    prev[fresh] = potential[fresh]
    delta = potential - prev
    setattr(env, attr, potential.clone())
    return delta


# ── task rewards ────────────────────────────────────────────────────────────

def delta_ring_distance(
    env: ManagerBasedRlEnv,
    standoff: float = 0.22,
    asset_cfg: SceneEntityCfg = _ROBOT_CFG,
    target_name: str = "target",
) -> torch.Tensor:
    """Potential-based approach: Δ(-|d - standoff|) per step.

    Closing on the ring pays, holding pays zero, overshooting past it charges.
    A raw exp(-d) proximity term would pay per step for parking on the ring —
    that is the jackpot AGENTS.md warns about. This one cannot be farmed.
    """
    potential = -_ring_error(env, asset_cfg, target_name, standoff)
    return _delta_potential(env, "_ai_ring_potential_prev", potential)


def head_bearing_gaussian(
    env: ManagerBasedRlEnv,
    sigma: float = 0.25,
    gate_tol: float = 0.08,
    standoff: float = 0.22,
    head_cfg: SceneEntityCfg = _HEAD_CFG,
    target_name: str = "target",
) -> torch.Tensor:
    """exp(-θ² / 2σ²) on head bearing error, hard-gated on being in the ring.

    The gate is a state check (|d - standoff| < gate_tol), not a shaped weight,
    so the policy cannot collect aim reward from across the room. It is a
    positive reward gated on a GOOD state (in the ring), which is allowed; the
    forbidden pattern is gating on a bad one.
    """
    theta = _head_bearing_error(env, head_cfg, target_name)
    in_ring = _ring_error(env, head_cfg, target_name, standoff) < gate_tol
    g = torch.exp(-(theta ** 2) / (2.0 * sigma ** 2))
    return torch.where(in_ring, g, torch.zeros_like(g))


def head_bearing_ema_l1_penalty(
    env: ManagerBasedRlEnv,
    tau_s: float = 1.0,
    head_cfg: SceneEntityCfg = _HEAD_CFG,
    target_name: str = "target",
) -> torch.Tensor:
    """-L1 of a τ-second EMA of the signed bearing error. Self-negating: POSITIVE weight.

    Charges the DC bias of the aim and lets the walking oscillation cancel.
    The head is a large fraction of body mass and MUST swing while walking;
    an instantaneous tight-std aim term taxed walking so hard a previous
    policy stood still. Price only the escapable part.
    """
    head_cfg = _resolved(env, head_cfg)
    robot: Entity = env.scene[head_cfg.name]
    target: Entity = env.scene[target_name]
    sid = int(head_cfg.site_ids[0])
    bid = int(head_cfg.body_ids[0])
    p_tip = robot.data.site_pos_w[:, sid, :2]
    p_head = robot.data.body_com_pos_w[:, bid, :2]
    fwd = p_tip - p_head
    to_t = target.data.root_link_pos_w[:, :2] - p_head
    err = torch.atan2(to_t[:, 1], to_t[:, 0]) - torch.atan2(fwd[:, 1], fwd[:, 0])
    err = torch.atan2(torch.sin(err), torch.cos(err))  # signed, wrapped

    dt = float(env.step_dt)
    alpha = dt / max(tau_s, dt)
    if not hasattr(env, "_ai_bearing_ema"):
        env._ai_bearing_ema = err.clone()
    fresh = _fresh(env)
    env._ai_bearing_ema[fresh] = err[fresh]
    env._ai_bearing_ema = (1.0 - alpha) * env._ai_bearing_ema + alpha * err
    return -torch.abs(env._ai_bearing_ema)


def delta_dwell_progress(
    env: ManagerBasedRlEnv,
    rate: float = 0.5,
    gate_tol: float = 0.08,
    aim_tol: float = 0.25,
    speed_tol: float = 0.05,
    standoff: float = 0.22,
    head_cfg: SceneEntityCfg = _HEAD_CFG,
    target_name: str = "target",
) -> torch.Tensor:
    """Slewed dwell latch. Pays Δprogress; progress ramps at ``rate`` per second
    while ALL of (in ring, aimed, stopped) hold, and decays at the same rate
    otherwise. Saturates at 1 and stays there — the phase flag reads it.

    ``speed_tol`` is inside the gate on purpose: without it the policy can orbit
    the target at the standoff radius, farming aim while never stopping. That
    is the failure I expect on the first run, and the fix is the gate, not a
    bigger speed penalty.
    """
    in_ring = _ring_error(env, head_cfg, target_name, standoff) < gate_tol
    aimed = _head_bearing_error(env, head_cfg, target_name) < aim_tol
    stopped = _planar_speed(env, head_cfg) < speed_tol
    ok = in_ring & aimed & stopped

    dt = float(env.step_dt)
    if not hasattr(env, "_ai_dwell"):
        env._ai_dwell = torch.zeros(env.num_envs, device=env.device)
    fresh = _fresh(env)
    env._ai_dwell[fresh] = 0.0
    prev = env._ai_dwell.clone()
    step = rate * dt
    env._ai_dwell = torch.where(ok, prev + step, prev - step).clamp(0.0, 1.0)
    return env._ai_dwell - prev


def planar_speed_gated_penalty(
    env: ManagerBasedRlEnv,
    gate_tol: float = 0.08,
    standoff: float = 0.22,
    asset_cfg: SceneEntityCfg = _ROBOT_CFG,
    target_name: str = "target",
) -> torch.Tensor:
    """-|v_xy| while inside the ring, zero outside. Self-negating: POSITIVE weight.

    Zero outside the ring so it never taxes the approach itself.
    """
    in_ring = _ring_error(env, asset_cfg, target_name, standoff) < gate_tol
    v = _planar_speed(env, asset_cfg)
    return torch.where(in_ring, -v, torch.zeros_like(v))


def delta_heading_progress(
    env: ManagerBasedRlEnv,
    target_offset: float = math.pi,
    require_dwell: float = 0.999,
    asset_cfg: SceneEntityCfg = _ROBOT_CFG,
    target_name: str = "target",
) -> torch.Tensor:
    """Potential-based turn-away: Δ(-|heading error to (bearing + offset)|).

    Only active once the dwell latch has saturated (``require_dwell``). Before
    that the potential is frozen, so there is no reward for turning away early
    and no waypoint to camp at.
    """
    robot: Entity = env.scene[asset_cfg.name]
    q = robot.data.root_link_quat_w
    yaw = torch.atan2(2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                      1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2))
    off = _planar_offset_to_target(env, asset_cfg, target_name)
    bearing = torch.atan2(off[:, 1], off[:, 0])
    want = bearing + target_offset
    err = want - yaw
    err = torch.atan2(torch.sin(err), torch.cos(err))
    potential = -torch.abs(err)

    dwell = getattr(env, "_ai_dwell", None)
    if dwell is None:
        return torch.zeros(env.num_envs, device=env.device)
    active = dwell >= require_dwell

    if not hasattr(env, "_ai_heading_potential_prev"):
        env._ai_heading_potential_prev = potential.clone()
    prev = env._ai_heading_potential_prev
    fresh = _fresh(env)
    prev[fresh] = potential[fresh]
    # Frozen while inactive: reseed prev every step so the first active step
    # starts from the current pose rather than paying a backlog.
    prev[~active] = potential[~active]
    delta = potential - prev
    env._ai_heading_potential_prev = potential.clone()
    return torch.where(active, delta, torch.zeros_like(delta))


def phase_flag(env: ManagerBasedRlEnv) -> torch.Tensor:
    """0 during approach, 1 once the dwell latch has saturated. For the command slot."""
    dwell = getattr(env, "_ai_dwell", None)
    if dwell is None:
        return torch.zeros(env.num_envs, device=env.device)
    return (dwell >= 0.999).float()


# ── critic-only observations ────────────────────────────────────────────────

def dwell_progress_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """(N,1) dwell latch value in [0, 1]. CRITIC ONLY — the actor sees the
    binary phase flag through the command slot, which is what the runtime can
    reproduce."""
    dwell = getattr(env, "_ai_dwell", None)
    if dwell is None:
        return torch.zeros(env.num_envs, 1, device=env.device)
    return dwell.unsqueeze(-1)


# ── target spawn event ──────────────────────────────────────────────────────

def reset_target_around_robot(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    radius_range: tuple[float, float] = (0.4, 1.5),
    asset_name: str = "target",
    z_offset: float = 0.0,
) -> None:
    """Place the (mocap) target at a uniform bearing and radius from the robot.

    Reads the robot root from qpos directly (root_link_pos_w lags until the next
    forward()) — same trick as ``reset_ball_in_front_of_foot``. Must be
    registered AFTER ``reset_base`` (events run in dict insertion order) so the
    robot pose is final.
    """
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device)
    robot: Entity = env.scene["robot"]
    target: Entity = env.scene[asset_name]

    root = env.sim.data.qpos[env_ids][:, robot.indexing.free_joint_q_adr]
    n = len(env_ids)
    radius = torch.empty(n, device=env.device).uniform_(*radius_range)
    bearing = torch.empty(n, device=env.device).uniform_(-math.pi, math.pi)

    pose = torch.zeros(n, 7, device=env.device)
    pose[:, 0] = root[:, 0] + radius * torch.cos(bearing)
    pose[:, 1] = root[:, 1] + radius * torch.sin(bearing)
    pose[:, 2] = env.scene.terrain.env_origins[env_ids, 2] + z_offset
    pose[:, 3] = 1.0  # identity quat
    target.write_mocap_pose_to_sim(pose, env_ids)


# ── command term: target offset in the base frame + phase flag ──────────────

from mjlab.tasks.velocity.mdp import UniformVelocityCommand  # noqa: E402
from mjlab_microduck.tasks.mdp import VelocityCommandCommandOnlyCfg  # noqa: E402


class TargetCommand(UniformVelocityCommand):
    """Fills the 3-D twist slot with ``[dx_norm, dy_norm, phase_flag]``.

    ``dx_norm, dy_norm`` = target offset in the robot's yaw frame divided by
    ``scale_m`` (1.5 m → ±1) and clipped to ±1. ``phase_flag`` = 0 during the
    approach, 1 once the dwell latch has saturated (see ``phase_flag``).

    Subclasses the velocity command so the ``twist`` obs term, the viz hooks and
    the runtime's 13-D command layout are untouched: at deployment the runtime
    writes the same three numbers from its own target estimate.

    Nothing is sampled here — the command is a pure function of state, so
    ``_resample_command`` is a no-op and ``reset`` just refreshes the values.
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._ai_env = env
        self._target_name = str(getattr(cfg, "target_name", "target"))
        self._scale_m = float(getattr(cfg, "scale_m", 1.5))

    @property
    def command(self) -> torch.Tensor:
        return self.vel_command_b

    def _refresh(self) -> None:
        robot: Entity = self.robot
        target: Entity = self._ai_env.scene[self._target_name]
        rel = target.data.root_link_pos_w[:, :2] - robot.data.root_link_pos_w[:, :2]
        q = robot.data.root_link_quat_w
        yaw = torch.atan2(2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                          1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2))
        c, s = torch.cos(yaw), torch.sin(yaw)
        dx = c * rel[:, 0] + s * rel[:, 1]
        dy = -s * rel[:, 0] + c * rel[:, 1]
        self.vel_command_b[:, 0] = (dx / self._scale_m).clamp(-1.0, 1.0)
        self.vel_command_b[:, 1] = (dy / self._scale_m).clamp(-1.0, 1.0)
        self.vel_command_b[:, 2] = phase_flag(self._ai_env)

    def compute(self, dt: float) -> None:
        self._refresh()

    def reset(self, env_ids: torch.Tensor | None) -> dict:
        self._refresh()
        return {}

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        pass  # pure function of state

    def _update_command(self) -> None:
        pass  # done in compute()

    def _update_metrics(self) -> None:
        pass  # no velocity-tracking metrics


from dataclasses import dataclass as _dataclass  # noqa: E402


@_dataclass(kw_only=True)
class TargetCommandCfg(VelocityCommandCommandOnlyCfg):
    """Cfg for :class:`TargetCommand`. Inherits the velocity cfg so it can be
    built from ``vars()`` of the velocity env's twist command (ranges etc. are
    carried but unused)."""
    class_type: type = TargetCommand
    target_name: str = "target"
    scale_m: float = 1.5

    def build(self, env: ManagerBasedRlEnv) -> "TargetCommand":
        return TargetCommand(self, env)

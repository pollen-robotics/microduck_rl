"""Supervised handoff from the immutable walker to a stair specialist."""

from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
from tensordict import TensorDict

OFFICIAL_WALKER_RELATIVE_PATH = Path(
    "logs/rsl_rl/velocity/official_onnx_bootstrap/model_0.pt"
)
OFFICIAL_WALKER_SHA256 = (
    "c17ce5ffd8270eb26cf45ebfcb002a80146b45fa89222b6c025af3c6caf24d43"
)
TWIST_COMMAND_SLICE = slice(48, 51)
ROUTE_CUE_SLICE = slice(55, 61)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_official_walker_checkpoint(
    repo_root: str | Path,
    checkpoint: str | Path | None = None,
) -> Path:
    """Resolve and hash-check the one approved manufacturer walker."""

    path = (
        Path(checkpoint).expanduser()
        if checkpoint is not None
        else Path(repo_root).expanduser() / OFFICIAL_WALKER_RELATIVE_PATH
    ).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Official walker checkpoint not found: {path}")
    actual_sha256 = _sha256(path)
    if actual_sha256 != OFFICIAL_WALKER_SHA256:
        raise ValueError(
            "Walker checkpoint is not the pinned manufacturer policy: "
            f"expected {OFFICIAL_WALKER_SHA256}, got {actual_sha256} for {path}"
        )
    return path


@dataclass(frozen=True)
class StairRouteEstimate:
    """Robot state relative to the next stair face.

    The interface contains only quantities that can later be supplied by a
    calibrated ToF/camera route estimator plus the robot state estimator.
    """

    distance_to_next_face_m: torch.Tensor
    lateral_error_m: torch.Tensor
    heading_error_rad: torch.Tensor
    forward_velocity_mps: torch.Tensor
    upright_score: torch.Tensor
    non_foot_contact: torch.Tensor
    finite: torch.Tensor


class StairRouteEstimator(Protocol):
    """Source of stair-frame route estimates for a policy dispatcher."""

    def estimate(self) -> StairRouteEstimate: ...


class SimulationStairRouteEstimator:
    """Read a :class:`StairRouteEstimate` from the vectorized MuJoCo scene."""

    def __init__(
        self,
        env: Any,
        *,
        stair_face_local_x_m: float = 0.66,
        velocity_ema_tau_s: float = 0.001,
        non_foot_sensor_names: tuple[str, ...] = (
            "head_ground_contact",
            "trunk_ground_contact",
            "legs_ground_contact",
        ),
    ) -> None:
        self._env = env
        self.stair_face_local_x_m = float(stair_face_local_x_m)
        if velocity_ema_tau_s <= 0.0:
            raise ValueError("velocity_ema_tau_s must be positive")
        self.velocity_ema_tau_s = float(velocity_ema_tau_s)
        self.non_foot_sensor_names = non_foot_sensor_names
        self._forward_velocity_ema: torch.Tensor | None = None
        self._last_velocity_step = -1

    def _non_foot_contact(self) -> torch.Tensor:
        contact = torch.zeros(
            self._env.num_envs,
            dtype=torch.bool,
            device=self._env.device,
        )
        for name in self.non_foot_sensor_names:
            if name not in self._env.scene.sensors:
                continue
            found = self._env.scene.sensors[name].data.found
            contact |= (found.reshape(found.shape[0], -1) > 0).any(dim=-1)
        return contact

    def estimate(self) -> StairRouteEstimate:
        robot = self._env.scene["robot"].data
        origins = self._env.scene.terrain.env_origins
        local_position = robot.root_link_pos_w - origins
        quaternion = robot.root_link_quat_w
        w, x, y, z = quaternion.unbind(dim=-1)
        yaw = torch.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y.square() + z.square()),
        )
        heading_error = torch.atan2(torch.sin(yaw), torch.cos(yaw))
        raw_forward_velocity = robot.root_link_lin_vel_b[:, 0]
        current_step = int(self._env.common_step_counter)
        if self._forward_velocity_ema is None:
            self._forward_velocity_ema = raw_forward_velocity.clone()
        elif current_step != self._last_velocity_step:
            alpha = 1.0 - math.exp(
                -float(self._env.step_dt) / self.velocity_ema_tau_s
            )
            updated = self._forward_velocity_ema + alpha * (
                raw_forward_velocity - self._forward_velocity_ema
            )
            fresh = self._env.episode_length_buf <= 1
            self._forward_velocity_ema = torch.where(
                fresh,
                raw_forward_velocity,
                updated,
            )
        self._last_velocity_step = current_step
        forward_velocity = self._forward_velocity_ema
        upright = 1.0 - 2.0 * (x.square() + y.square())
        distance = self.stair_face_local_x_m - local_position[:, 0]
        lateral = local_position[:, 1]
        finite = (
            torch.isfinite(distance)
            & torch.isfinite(lateral)
            & torch.isfinite(heading_error)
            & torch.isfinite(forward_velocity)
            & torch.isfinite(upright)
        )
        return StairRouteEstimate(
            distance_to_next_face_m=distance,
            lateral_error_m=lateral,
            heading_error_rad=heading_error,
            forward_velocity_mps=forward_velocity,
            upright_score=upright,
            non_foot_contact=self._non_foot_contact(),
            finite=finite,
        )


@dataclass(frozen=True)
class StairHandoffCriteria:
    """Physical acceptance window for walker-to-climber transitions."""

    min_distance_m: float = 0.080
    max_distance_m: float = 0.120
    max_abs_lateral_m: float = 0.040
    max_abs_heading_rad: float = math.radians(8.0)
    min_forward_velocity_mps: float = 0.160
    max_forward_velocity_mps: float = 0.300
    min_upright_score: float = 0.900

    def components(
        self,
        estimate: StairRouteEstimate,
    ) -> dict[str, torch.Tensor]:
        return {
            "finite": estimate.finite,
            "distance": (
                (estimate.distance_to_next_face_m >= self.min_distance_m)
                & (estimate.distance_to_next_face_m <= self.max_distance_m)
            ),
            "lateral": (
                estimate.lateral_error_m.abs() <= self.max_abs_lateral_m
            ),
            "heading": (
                estimate.heading_error_rad.abs() <= self.max_abs_heading_rad
            ),
            "velocity": (
                (estimate.forward_velocity_mps >= self.min_forward_velocity_mps)
                & (estimate.forward_velocity_mps <= self.max_forward_velocity_mps)
            ),
            "upright": estimate.upright_score >= self.min_upright_score,
            "contact": ~estimate.non_foot_contact,
        }

    def evaluate(self, estimate: StairRouteEstimate) -> torch.Tensor:
        components = self.components(estimate)
        return torch.stack(tuple(components.values()), dim=-1).all(dim=-1)


@dataclass(frozen=True)
class StairApproachSupervisor:
    """Convert stair-frame errors into commands understood by the walker."""

    lateral_gain: float | torch.Tensor = 2.0
    heading_gain: float | torch.Tensor = 1.5
    cross_track_heading_gain: float | torch.Tensor = 0.0
    max_abs_vy_mps: float = 0.040
    max_abs_wz_radps: float = 0.200
    min_vx_mps: float = 0.100
    max_vx_mps: float = 0.300
    forward_command_mps: float | torch.Tensor = 0.300
    twist_command_slice: slice = TWIST_COMMAND_SLICE
    route_cue_slice: slice = ROUTE_CUE_SLICE

    @staticmethod
    def _parameter_like(
        value: float | torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        parameter = torch.as_tensor(
            value,
            dtype=reference.dtype,
            device=reference.device,
        )
        if parameter.ndim == 0:
            return parameter.expand_as(reference)
        if parameter.shape != reference.shape:
            raise ValueError(
                "Vectorized approach parameter shape does not match environments: "
                f"{tuple(parameter.shape)} != {tuple(reference.shape)}"
            )
        return parameter

    def apply(
        self,
        observations: TensorDict,
        estimate: StairRouteEstimate,
    ) -> TensorDict:
        supervised = observations.clone()
        actor = supervised["actor"].clone()
        if actor.shape[-1] < self.route_cue_slice.stop:
            raise ValueError(
                "Walker observation is shorter than the required 61D contract: "
                f"{actor.shape[-1]}"
            )
        world_vy = torch.clamp(
            -self._parameter_like(
                self.lateral_gain,
                estimate.lateral_error_m,
            )
            * estimate.lateral_error_m,
            min=-self.max_abs_vy_mps,
            max=self.max_abs_vy_mps,
        )
        target_vx = self._parameter_like(
            self.forward_command_mps,
            estimate.lateral_error_m,
        )
        cosine = torch.cos(estimate.heading_error_rad)
        sine = torch.sin(estimate.heading_error_rad)
        body_vx = torch.clamp(
            cosine * target_vx + sine * world_vy,
            min=self.min_vx_mps,
            max=self.max_vx_mps,
        )
        body_vy = torch.clamp(
            -sine * target_vx + cosine * world_vy,
            min=-self.max_abs_vy_mps,
            max=self.max_abs_vy_mps,
        )
        wz = torch.clamp(
            -self._parameter_like(
                self.heading_gain,
                estimate.heading_error_rad,
            )
            * estimate.heading_error_rad
            - self._parameter_like(
                self.cross_track_heading_gain,
                estimate.lateral_error_m,
            )
            * estimate.lateral_error_m,
            min=-self.max_abs_wz_radps,
            max=self.max_abs_wz_radps,
        )
        actor[:, self.twist_command_slice] = torch.stack(
            (
                body_vx,
                body_vy,
                wz,
            ),
            dim=-1,
        )
        actor[:, self.route_cue_slice] = 0.0
        supervised["actor"] = actor
        return supervised


def load_frozen_actor(
    runner: Any, checkpoint: str | Path, *, device: str
) -> torch.nn.Module:
    """Load one deterministic actor without retaining optimizer or critic state."""

    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Actor checkpoint not found: {checkpoint}")
    runner.load(
        str(checkpoint),
        load_cfg={"actor": True},
        strict=True,
        map_location=device,
    )
    actor = copy.deepcopy(runner.alg.actor).to(device)
    actor.eval()
    actor.requires_grad_(False)
    return actor


def load_actor_pair(
    runner: Any,
    walker_checkpoint: str | Path,
    specialist_checkpoint: str | Path,
    *,
    device: str,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Load independent frozen actors through mjlab's checkpoint migration path."""

    walker_path = Path(walker_checkpoint).expanduser().resolve()
    specialist_path = Path(specialist_checkpoint).expanduser().resolve()
    for label, checkpoint in (
        ("walker", walker_path),
        ("specialist", specialist_path),
    ):
        if not checkpoint.is_file():
            raise FileNotFoundError(f"{label.title()} checkpoint not found: {checkpoint}")

    walker = load_frozen_actor(runner, walker_path, device=device)
    specialist = load_frozen_actor(runner, specialist_path, device=device)
    return walker, specialist


class HardStairHandoffPolicy(torch.nn.Module):
    """Supervise WALK, blend for four frames, then latch onto CLIMB."""

    WALK = 0
    BLEND = 1
    CLIMB = 2

    def __init__(
        self,
        walker: torch.nn.Module,
        specialist: torch.nn.Module,
        env: Any,
        *,
        route_estimator: StairRouteEstimator | None = None,
        approach_supervisor: StairApproachSupervisor | None = None,
        handoff_criteria: StairHandoffCriteria | None = None,
        stair_face_local_x_m: float = 0.66,
        blend_steps: int = 4,
    ) -> None:
        super().__init__()
        self.walker = walker
        self.specialist = specialist
        self._env = env
        if blend_steps < 0:
            raise ValueError("blend_steps must be nonnegative")
        self.blend_steps = int(blend_steps)
        self.route_estimator = route_estimator or SimulationStairRouteEstimator(
            env,
            stair_face_local_x_m=stair_face_local_x_m,
        )
        self.approach_supervisor = approach_supervisor or StairApproachSupervisor()
        self.handoff_criteria = handoff_criteria or StairHandoffCriteria()
        self.register_buffer(
            "phase",
            torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
            persistent=False,
        )
        self.register_buffer(
            "specialist_latched",
            torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
            persistent=False,
        )
        self.register_buffer(
            "handoff_events",
            torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
            persistent=False,
        )
        self.register_buffer(
            "blend_progress",
            torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
            persistent=False,
        )
        self.register_buffer(
            "phase_transition_counts",
            torch.zeros(env.num_envs, 2, dtype=torch.long, device=env.device),
            persistent=False,
        )
        self.register_buffer(
            "handoff_rejection_counts",
            torch.zeros(env.num_envs, 6, dtype=torch.long, device=env.device),
            persistent=False,
        )
        self.register_buffer(
            "distance_window_seen",
            torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
            persistent=False,
        )
        for name in (
            "handoff_distance_m",
            "handoff_lateral_error_m",
            "handoff_heading_error_rad",
            "handoff_forward_velocity_mps",
            "handoff_upright_score",
        ):
            self.register_buffer(
                name,
                torch.full(
                    (env.num_envs,),
                    torch.nan,
                    dtype=torch.float32,
                    device=env.device,
                ),
                persistent=False,
            )
        self.register_buffer(
            "handoff_control_step",
            torch.full(
                (env.num_envs,),
                -1,
                dtype=torch.long,
                device=env.device,
            ),
            persistent=False,
        )

    @property
    def handoff_count(self) -> int:
        return int(self.handoff_events.sum().item())

    def _reset_fresh(self, fresh: torch.Tensor) -> None:
        self.phase[fresh] = self.WALK
        self.specialist_latched[fresh] = False
        self.blend_progress[fresh] = 0
        self.handoff_distance_m[fresh] = torch.nan
        self.handoff_lateral_error_m[fresh] = torch.nan
        self.handoff_heading_error_rad[fresh] = torch.nan
        self.handoff_forward_velocity_mps[fresh] = torch.nan
        self.handoff_upright_score[fresh] = torch.nan
        self.handoff_control_step[fresh] = -1
        self.distance_window_seen[fresh] = False

    def _capture_handoff(
        self,
        newly_latched: torch.Tensor,
        estimate: StairRouteEstimate,
    ) -> None:
        if not bool(newly_latched.any()):
            return
        self.handoff_distance_m[newly_latched] = (
            estimate.distance_to_next_face_m[newly_latched]
        )
        self.handoff_lateral_error_m[newly_latched] = (
            estimate.lateral_error_m[newly_latched]
        )
        self.handoff_heading_error_rad[newly_latched] = (
            estimate.heading_error_rad[newly_latched]
        )
        self.handoff_forward_velocity_mps[newly_latched] = (
            estimate.forward_velocity_mps[newly_latched]
        )
        self.handoff_upright_score[newly_latched] = (
            estimate.upright_score[newly_latched]
        )
        self.handoff_control_step[newly_latched] = self._env.episode_length_buf[
            newly_latched
        ]

    def forward(self, observations: TensorDict) -> torch.Tensor:
        fresh = self._env.episode_length_buf <= 1
        self._reset_fresh(fresh)
        estimate = self.route_estimator.estimate()
        components = self.handoff_criteria.components(estimate)
        in_distance_window = components["distance"]
        self.distance_window_seen |= in_distance_window
        rejection_components = torch.stack(
            tuple(
                components[name]
                for name in (
                    "finite",
                    "lateral",
                    "heading",
                    "velocity",
                    "upright",
                    "contact",
                )
            ),
            dim=-1,
        )
        self.handoff_rejection_counts += (
            in_distance_window.unsqueeze(-1) & ~rejection_components
        ).to(torch.long)
        gate = torch.stack(tuple(components.values()), dim=-1).all(dim=-1)
        newly_latched = (self.phase == self.WALK) & gate
        self.specialist_latched |= newly_latched
        self.handoff_events += newly_latched.to(torch.long)
        self.phase_transition_counts[:, 0] += newly_latched.to(torch.long)
        self._capture_handoff(newly_latched, estimate)
        self.phase[newly_latched] = (
            self.CLIMB if self.blend_steps == 0 else self.BLEND
        )

        walker_observations = self.approach_supervisor.apply(observations, estimate)
        walker_actions = self.walker(walker_observations)
        specialist_actions = self.specialist(observations)
        if self.blend_steps == 0:
            alpha = self.specialist_latched.to(walker_actions.dtype)
        else:
            blending = self.phase == self.BLEND
            self.blend_progress[blending] += 1
            completed = blending & (self.blend_progress >= self.blend_steps)
            self.phase[completed] = self.CLIMB
            self.phase_transition_counts[:, 1] += completed.to(torch.long)
            alpha = torch.where(
                self.phase == self.CLIMB,
                torch.full_like(self.blend_progress, self.blend_steps),
                self.blend_progress,
            ).to(walker_actions.dtype) / self.blend_steps
            alpha = torch.clamp(alpha, 0.0, 1.0)
        return torch.lerp(walker_actions, specialist_actions, alpha.unsqueeze(-1))

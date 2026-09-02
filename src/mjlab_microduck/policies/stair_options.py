"""Composed walker, launch, mantle, and recovery option policy."""

from __future__ import annotations

from typing import Any

import torch
from tensordict import TensorDict

from .stair_handoff import (
    ROUTE_CUE_SLICE,
    TWIST_COMMAND_SLICE,
    SimulationStairRouteEstimator,
    StairApproachSupervisor,
    StairHandoffCriteria,
    StairRouteEstimator,
)


class StairOptionPolicy(torch.nn.Module):
    """Run frozen option actors with latched, physics-gated transitions."""

    WALK = 0
    WALK_TO_LAUNCH = 1
    LAUNCH = 2
    LAUNCH_TO_MANTLE = 3
    MANTLE = 4
    RECOVER = 5
    FAILED = 6

    def __init__(
        self,
        walker: torch.nn.Module,
        launch: torch.nn.Module,
        mantle: torch.nn.Module,
        recover: torch.nn.Module,
        env: Any,
        *,
        route_estimator: StairRouteEstimator | None = None,
        approach_supervisor: StairApproachSupervisor | None = None,
        handoff_criteria: StairHandoffCriteria | None = None,
        start_in_launch: bool = False,
        handoff_guard_frames: int = 4,
        option_blend_steps: int = 4,
        launch_min_steps: int | torch.Tensor = 8,
        launch_max_steps: int | torch.Tensor = 40,
        mantle_root_x_m: float | torch.Tensor = 0.60,
        mantle_root_z_m: float | torch.Tensor = 0.145,
        stair_face_local_x_m: float = 0.66,
        min_head_stair_contact_z_m: float = 0.03,
    ) -> None:
        super().__init__()
        if handoff_guard_frames < 1:
            raise ValueError("handoff_guard_frames must be positive")
        if option_blend_steps < 0:
            raise ValueError("option_blend_steps must be nonnegative")
        self.walker = walker
        self.launch = launch
        self.mantle = mantle
        self.recover = recover
        self._env = env
        self.route_estimator = route_estimator or SimulationStairRouteEstimator(env)
        self.approach_supervisor = approach_supervisor or StairApproachSupervisor()
        self.handoff_criteria = handoff_criteria or StairHandoffCriteria()
        self.start_in_launch = bool(start_in_launch)
        self.handoff_guard_frames = int(handoff_guard_frames)
        self.option_blend_steps = int(option_blend_steps)
        self.launch_min_steps = self._long_parameter(launch_min_steps)
        self.launch_max_steps = self._long_parameter(launch_max_steps)
        self.mantle_root_x_m = self._float_parameter(mantle_root_x_m)
        self.mantle_root_z_m = self._float_parameter(mantle_root_z_m)
        self.stair_face_local_x_m = float(stair_face_local_x_m)
        self.min_head_stair_contact_z_m = float(min_head_stair_contact_z_m)
        if bool((self.launch_max_steps < self.launch_min_steps).any()):
            raise ValueError("launch_max_steps must be at least launch_min_steps")
        initial_phase = self.LAUNCH if self.start_in_launch else self.WALK
        self.register_buffer(
            "phase",
            torch.full(
                (env.num_envs,), initial_phase, dtype=torch.long, device=env.device
            ),
            persistent=False,
        )
        self.register_buffer(
            "phase_step",
            torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
            persistent=False,
        )
        self.register_buffer(
            "transition_counts",
            torch.zeros(env.num_envs, 6, dtype=torch.long, device=env.device),
            persistent=False,
        )
        self.register_buffer(
            "_handoff_hold",
            torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
            persistent=False,
        )
        self.register_buffer(
            "_previous_episode_length",
            torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device),
            persistent=False,
        )

    def _float_parameter(self, value: float | torch.Tensor) -> torch.Tensor:
        result = torch.as_tensor(value, dtype=torch.float32, device=self._env.device)
        if result.ndim == 0:
            return result.expand(self._env.num_envs).clone()
        if result.shape != (self._env.num_envs,):
            raise ValueError("Option parameter must have one value per environment")
        return result

    def _long_parameter(self, value: int | torch.Tensor) -> torch.Tensor:
        result = torch.as_tensor(value, dtype=torch.long, device=self._env.device)
        if result.ndim == 0:
            return result.expand(self._env.num_envs).clone()
        if result.shape != (self._env.num_envs,):
            raise ValueError("Option parameter must have one value per environment")
        return result

    def _reset_new_episodes(self) -> None:
        episode_length = self._env.episode_length_buf
        reset = (self._previous_episode_length < 0) | (
            episode_length < self._previous_episode_length
        )
        self.phase[reset] = self.LAUNCH if self.start_in_launch else self.WALK
        self.phase_step[reset] = 0
        self._handoff_hold[reset] = 0
        self._previous_episode_length.copy_(episode_length)

    def _transition(self, mask: torch.Tensor, target: int, counter: int) -> None:
        if not bool(mask.any()):
            return
        self.phase[mask] = target
        self.phase_step[mask] = 0
        self.transition_counts[:, counter] += mask.to(torch.long)

    def _launch_observations(self, observations: TensorDict) -> TensorDict:
        launch_observations = observations.clone()
        actor = launch_observations["actor"].clone()
        if actor.shape[-1] != ROUTE_CUE_SLICE.stop:
            raise ValueError("Option policy requires the 61D actor observation")
        actor[:, TWIST_COMMAND_SLICE] = 0.0
        actor[:, slice(TWIST_COMMAND_SLICE.stop, ROUTE_CUE_SLICE.stop)] = 0.0
        launch_observations["actor"] = actor
        return launch_observations

    def _head_riser_contact(self) -> torch.Tensor:
        head_contact = torch.zeros(
            self._env.num_envs, dtype=torch.bool, device=self._env.device
        )
        if "head_ground_contact" not in self._env.scene.sensors:
            return head_contact
        sensor = self._env.scene.sensors["head_ground_contact"].data
        found = sensor.found.reshape(sensor.found.shape[0], -1) > 0
        if not hasattr(sensor, "pos"):
            return head_contact
        positions = sensor.pos.reshape(sensor.pos.shape[0], -1, 3)
        origins = self._env.scene.terrain.env_origins.unsqueeze(1)
        local_positions = positions - origins
        on_riser_or_tread = (
            (local_positions[..., 0] >= self.stair_face_local_x_m - 0.02)
            & (local_positions[..., 2] >= self.min_head_stair_contact_z_m)
        )
        return (found & on_riser_or_tread).any(dim=-1)

    def _launch_to_mantle_gate(self) -> tuple[torch.Tensor, torch.Tensor]:
        robot = self._env.scene["robot"].data
        origins = self._env.scene.terrain.env_origins
        local_x = robot.root_link_pos_w[:, 0] - origins[:, 0]
        local_z = robot.root_link_pos_w[:, 2] - origins[:, 2]
        minimum_reached = self.phase_step >= self.launch_min_steps
        physical_frontier = self._head_riser_contact() | (
            (local_x >= self.mantle_root_x_m) & (local_z >= self.mantle_root_z_m)
        )
        timed_out = self.phase_step >= self.launch_max_steps
        return minimum_reached & physical_frontier, timed_out & ~physical_frontier

    def _actions(
        self,
        label: str,
        actor: torch.nn.Module,
        observations: TensorDict,
    ) -> torch.Tensor:
        actions = actor(observations)
        expected_shape = (self._env.num_envs, 14)
        if tuple(actions.shape) != expected_shape:
            raise ValueError(
                f"{label} actor must return {expected_shape}, got {tuple(actions.shape)}"
            )
        if not bool(torch.isfinite(actions).all()):
            raise ValueError(f"{label} actor returned non-finite actions")
        return actions

    def forward(self, observations: TensorDict) -> torch.Tensor:
        self._reset_new_episodes()
        estimate = self.route_estimator.estimate()
        handoff_candidate = (
            (self.phase == self.WALK)
            & self.handoff_criteria.evaluate(estimate)
        )
        self._handoff_hold = torch.where(
            handoff_candidate,
            self._handoff_hold + 1,
            torch.zeros_like(self._handoff_hold),
        )
        handoff = self._handoff_hold >= self.handoff_guard_frames
        self._transition(
            handoff,
            self.WALK_TO_LAUNCH if self.option_blend_steps else self.LAUNCH,
            0,
        )
        self._handoff_hold[handoff] = 0

        walk_blend_done = (self.phase == self.WALK_TO_LAUNCH) & (
            self.phase_step >= self.option_blend_steps
        )
        self._transition(walk_blend_done, self.LAUNCH, 1)
        physical_frontier, launch_timed_out = self._launch_to_mantle_gate()
        launch_to_mantle = (self.phase == self.LAUNCH) & physical_frontier
        failed_launch = (self.phase == self.LAUNCH) & launch_timed_out
        self._transition(
            launch_to_mantle,
            self.LAUNCH_TO_MANTLE if self.option_blend_steps else self.MANTLE,
            2,
        )
        self._transition(failed_launch, self.FAILED, 5)
        mantle_blend_done = (self.phase == self.LAUNCH_TO_MANTLE) & (
            self.phase_step >= self.option_blend_steps
        )
        self._transition(mantle_blend_done, self.MANTLE, 3)
        secured = getattr(self._env, "_stair_first_tread_secured_latched", None)
        if secured is not None:
            self._transition((self.phase == self.MANTLE) & secured, self.RECOVER, 4)

        if observations["actor"].shape[-1] != ROUTE_CUE_SLICE.stop:
            raise ValueError("Option policy requires the exact 61D actor contract")
        walker_actions = self._actions(
            "walker",
            self.walker,
            self.approach_supervisor.apply(observations, estimate),
        )
        launch_actions = self._actions(
            "launch", self.launch, self._launch_observations(observations)
        )
        mantle_actions = self._actions("mantle", self.mantle, observations)
        recover_actions = self._actions("recover", self.recover, observations)
        actions = walker_actions
        actions = torch.where(
            (self.phase == self.LAUNCH).unsqueeze(-1), launch_actions, actions
        )
        actions = torch.where(
            (self.phase == self.MANTLE).unsqueeze(-1), mantle_actions, actions
        )
        actions = torch.where(
            (self.phase == self.RECOVER).unsqueeze(-1), recover_actions, actions
        )
        actions = torch.where(
            (self.phase == self.FAILED).unsqueeze(-1), recover_actions, actions
        )
        if self.option_blend_steps:
            walk_alpha = torch.clamp(
                (self.phase_step.to(actions.dtype) + 1.0)
                / self.option_blend_steps,
                0.0,
                1.0,
            )
            walk_blend = torch.lerp(
                walker_actions, launch_actions, walk_alpha.unsqueeze(-1)
            )
            actions = torch.where(
                (self.phase == self.WALK_TO_LAUNCH).unsqueeze(-1),
                walk_blend,
                actions,
            )
            mantle_alpha = torch.clamp(
                (self.phase_step.to(actions.dtype) + 1.0)
                / self.option_blend_steps,
                0.0,
                1.0,
            )
            mantle_blend = torch.lerp(
                launch_actions, mantle_actions, mantle_alpha.unsqueeze(-1)
            )
            actions = torch.where(
                (self.phase == self.LAUNCH_TO_MANTLE).unsqueeze(-1),
                mantle_blend,
                actions,
            )
        self.phase_step += 1
        self._previous_episode_length.copy_(self._env.episode_length_buf)
        return actions

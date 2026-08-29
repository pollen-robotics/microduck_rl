"""Hard handoff from the manufacturer walker to a stair specialist."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch
from tensordict import TensorDict


def _load_frozen_actor(runner: Any, checkpoint: Path, device: str) -> torch.nn.Module:
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

    walker = _load_frozen_actor(runner, walker_path, device)
    specialist = _load_frozen_actor(runner, specialist_path, device)
    return walker, specialist


class HardStairHandoffPolicy(torch.nn.Module):
    """Use walking before the stair, then latch onto the specialist until reset.

    The walker receives the exact zero-padded 61D observation contract used by
    the manufacturer policy. The specialist receives the full route cues in
    slots 55:61. Once an environment crosses the handoff threshold it cannot
    switch back during that episode, even if the robot rebounds from the stair.
    """

    def __init__(
        self,
        walker: torch.nn.Module,
        specialist: torch.nn.Module,
        env: Any,
        *,
        switch_local_x_m: float = 0.56,
        route_cue_slice: slice = slice(55, 61),
    ) -> None:
        super().__init__()
        self.walker = walker
        self.specialist = specialist
        self._env = env
        self.switch_local_x_m = float(switch_local_x_m)
        self.route_cue_slice = route_cue_slice
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

    @property
    def handoff_count(self) -> int:
        return int(self.handoff_events.sum().item())

    def _local_x(self) -> torch.Tensor:
        robot_x = self._env.scene["robot"].data.root_link_pos_w[:, 0]
        origin_x = self._env.scene.terrain.env_origins[:, 0]
        return robot_x - origin_x

    def forward(self, observations: TensorDict) -> torch.Tensor:
        fresh = self._env.episode_length_buf <= 1
        self.specialist_latched[fresh] = False
        newly_latched = (~self.specialist_latched) & (
            self._local_x() >= self.switch_local_x_m
        )
        self.specialist_latched |= newly_latched
        self.handoff_events += newly_latched.to(torch.long)

        walker_observations = observations.clone()
        walker_actor_observations = walker_observations["actor"].clone()
        if walker_actor_observations.shape[-1] < self.route_cue_slice.stop:
            raise ValueError(
                "Walker observation is shorter than the required 61D contract: "
                f"{walker_actor_observations.shape[-1]}"
            )
        walker_actor_observations[:, self.route_cue_slice] = 0.0
        walker_observations["actor"] = walker_actor_observations

        walker_actions = self.walker(walker_observations)
        specialist_actions = self.specialist(observations)
        return torch.where(
            self.specialist_latched.unsqueeze(-1),
            specialist_actions,
            walker_actions,
        )

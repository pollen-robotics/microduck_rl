"""Composable inference policies for MicroDuck."""

from .stair_handoff import HardStairHandoffPolicy, load_actor_pair

__all__ = ["HardStairHandoffPolicy", "load_actor_pair"]

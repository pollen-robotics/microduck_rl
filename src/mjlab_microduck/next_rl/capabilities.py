"""Inventory and conservative planning for existing Microduck RL capabilities."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .schema import Capability, SkillSpec


_IDENTIFIER_SEPARATORS = re.compile(r"[\s_]+")


class Disposition(str, Enum):
    """The only safe next actions the planner can recommend."""

    REUSE = "reuse"
    TRAIN_NEW = "train_new"
    WARM_START = "warm_start"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class PlanDecision:
    """A reproducible disposition, its evidence, and an optional selected policy."""

    disposition: Disposition
    reason: str
    capability: Capability | None = None
    improve_reason: str | None = None


def normalize_identifier(value: str) -> str:
    """Return the case-insensitive, hyphenated form used for capability matching."""
    if not isinstance(value, str):
        raise TypeError("capability identifier must be a string")
    normalized = _IDENTIFIER_SEPARATORS.sub("-", value.strip().lower())
    if not normalized:
        raise ValueError("capability identifier must not be empty")
    return normalized


class CapabilityInventory:
    """Immutable lookup index over capabilities recorded by this workspace."""

    def __init__(self, capabilities: Iterable[Capability]) -> None:
        self._capabilities = tuple(capabilities)
        self._ids: dict[str, tuple[Capability, ...]] = self._index(
            (normalize_identifier(capability.id), capability) for capability in self._capabilities
        )
        self._aliases: dict[str, tuple[Capability, ...]] = self._index(
            (normalize_identifier(alias), capability)
            for capability in self._capabilities
            for alias in capability.aliases
        )

    @staticmethod
    def _index(items: Iterable[tuple[str, Capability]]) -> dict[str, tuple[Capability, ...]]:
        indexed: dict[str, list[Capability]] = {}
        for key, capability in items:
            indexed.setdefault(key, []).append(capability)
        return {key: tuple(value) for key, value in indexed.items()}

    @property
    def ids(self) -> frozenset[str]:
        """Normalized identifiers for all known capabilities."""
        return frozenset(self._ids)

    @classmethod
    def load_builtin(cls) -> CapabilityInventory:
        """Load the checked-in inventory generated from runtime-shipped policies."""
        catalog_path = Path(__file__).with_name("catalog.json")
        raw = json.loads(catalog_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or set(raw) != {"capabilities"} or not isinstance(raw["capabilities"], list):
            raise ValueError("catalog.json must contain only a capabilities list")
        return cls(Capability.from_dict(item) for item in raw["capabilities"])

    def resolve(self, query: str) -> PlanDecision:
        """Resolve an exact ID before considering aliases; never guess an ambiguity."""
        normalized = normalize_identifier(query)
        exact = self._ids.get(normalized, ())
        if exact:
            return self._decide_match(exact, "exact ID")
        return self._decide_match(self._aliases.get(normalized, ()), "alias")

    @staticmethod
    def _decide_match(matches: tuple[Capability, ...], source: str) -> PlanDecision:
        if not matches:
            return PlanDecision(Disposition.TRAIN_NEW, "No matching existing capability was found.")
        if len(matches) > 1:
            return PlanDecision(
                Disposition.BLOCKED,
                f"Ambiguous {source}: multiple capabilities match; choose an exact ID.",
            )
        capability = matches[0]
        if capability.status == "learned":
            return PlanDecision(Disposition.REUSE, f"Matched approved learned capability by {source}.", capability)
        if capability.status == "available":
            return PlanDecision(Disposition.TRAIN_NEW, f"Matched available task by {source}; training is permitted.", capability)
        return PlanDecision(
            Disposition.BLOCKED,
            f"Matched {capability.status} capability by {source}; evaluate existing policy before reuse or retraining.",
            capability,
        )

    def _exact_matches(self, identifier: str) -> tuple[Capability, ...]:
        return self._ids.get(normalize_identifier(identifier), ())


def _contracts_match(spec: SkillSpec, capability: Capability) -> bool:
    return capability.robot_model == spec.contract.robot_model and capability.contract == spec.contract


def _has_requested_metric_evidence(spec: SkillSpec, capability: Capability) -> bool:
    evaluation = capability.evaluation
    if evaluation is None or evaluation.kind != "evaluation_report" or evaluation.passed is not True:
        return False
    for threshold in spec.metrics:
        result = evaluation.metric_results.get(threshold.name)
        if result is None:
            return False
        if threshold.direction == "minimum" and result < threshold.limit:
            return False
        if threshold.direction == "maximum" and result > threshold.limit:
            return False
    return True


def _select_parent(spec: SkillSpec, inventory: CapabilityInventory) -> PlanDecision | None:
    for parent_id in spec.allowed_parent_capabilities:
        matches = inventory._exact_matches(parent_id)
        if len(matches) > 1:
            return PlanDecision(
                Disposition.BLOCKED,
                f"Allowed parent capability {parent_id!r} is ambiguous; choose an exact ID.",
            )
        if len(matches) != 1:
            continue
        parent = matches[0]
        if parent.policy is not None and _contracts_match(spec, parent) and parent.status in {"validated", "learned"}:
            return PlanDecision(
                Disposition.WARM_START,
                f"Compatible allowed parent capability {parent.id!r} can warm-start training.",
                parent,
            )
    return None


def _resolve_requested_skill(spec: SkillSpec, inventory: CapabilityInventory) -> PlanDecision:
    direct = inventory.resolve(spec.id)
    if direct.disposition == Disposition.BLOCKED:
        return direct

    matches: dict[str, Capability] = {}
    if direct.capability is not None:
        matches[normalize_identifier(direct.capability.id)] = direct.capability
    alias_match = False
    for alias in spec.aliases:
        resolution = inventory.resolve(alias)
        if resolution.disposition == Disposition.BLOCKED:
            return resolution
        if resolution.capability is not None:
            alias_match = True
            matches[normalize_identifier(resolution.capability.id)] = resolution.capability

    if len(matches) > 1:
        return PlanDecision(
            Disposition.BLOCKED,
            "Requested ID and aliases match multiple existing capabilities; evaluate existing policy before reuse or retraining.",
        )
    if alias_match and direct.capability is None:
        capability = next(iter(matches.values()))
        return PlanDecision(
            Disposition.BLOCKED,
            "Requested alias matches an existing capability; evaluate existing policy before reuse or retraining.",
            capability,
        )
    if direct.capability is not None:
        return direct
    return PlanDecision(Disposition.TRAIN_NEW, "No matching existing capability was found.")


def plan_skill(
    spec: SkillSpec,
    inventory: CapabilityInventory,
    improve_reason: str | None = None,
) -> PlanDecision:
    """Return the safest action for a requested skill without inventing evidence."""
    if improve_reason is not None and (not isinstance(improve_reason, str) or not improve_reason.strip()):
        raise ValueError("improve_reason must be a non-empty string when provided")

    direct = _resolve_requested_skill(spec, inventory)
    if direct.disposition == Disposition.BLOCKED:
        return direct
    if direct.capability is not None:
        capability = direct.capability
        if not _contracts_match(spec, capability):
            return PlanDecision(
                Disposition.BLOCKED,
                "Matching capability has an incompatible policy contract; evaluate or train under the requested contract.",
                capability,
            )
        if capability.status == "available":
            return direct
        if capability.version != spec.version:
            return PlanDecision(
                Disposition.BLOCKED,
                "Matching learned capability version differs from the requested spec; evaluate existing policy first.",
                capability,
            )
        if not _has_requested_metric_evidence(spec, capability):
            return PlanDecision(
                Disposition.BLOCKED,
                "Matching learned capability lacks requested metric evidence; evaluate existing policy first.",
                capability,
            )
        if improve_reason is not None:
            return PlanDecision(
                Disposition.TRAIN_NEW,
                "Approved learned capability satisfies the request; improvement training was explicitly requested.",
                capability,
                improve_reason,
            )
        return direct

    parent = _select_parent(spec, inventory)
    if parent is not None:
        return parent
    return direct

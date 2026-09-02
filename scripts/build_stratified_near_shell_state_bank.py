#!/usr/bin/env python3
"""Build a balanced upper/lower reverse-curriculum state bank."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import torch

from collect_stair_walker_states import (
    _exact_state_row_digest,
    _finite_state_rows,
    _select_nested_rows,
    _sha256,
    _write_bank_atomic,
)
from mjlab_microduck.tasks.stair_walk_state_bank import (
    BANK_SCHEMA_VERSION,
    STANDARD_NUM_STEPS,
    STANDARD_RISER_HEIGHT_M,
    STANDARD_TREAD_DEPTH_M,
    concatenate_walk_state_rows,
    load_walk_state_bank,
    walk_state_count,
)

STRATUM_FIELD = "reverse_curriculum_stratum"
LOWER_STRATUM = 0
UPPER_STRATUM = 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upper-bank",
        type=Path,
        default=Path(".tmp/codex/full170-a34-near-shell-negative-state-bank.pt"),
    )
    parser.add_argument(
        "--lower-bank",
        type=Path,
        default=Path(
            ".tmp/codex/full170-a35-lower-near-shell-negative-state-bank.pt"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".tmp/codex/full170-a35-stratified-shell-state-bank.pt"),
    )
    parser.add_argument("--states-per-stratum", type=int, default=64)
    return parser.parse_args()


def _frontier(states: dict[str, object]) -> torch.Tensor:
    value = states.get("captured_shell_frontier_value")
    if not isinstance(value, torch.Tensor) or value.ndim != 1:
        raise ValueError(
            "Source bank must contain one captured_shell_frontier_value per row"
        )
    return value


def _stratified_indices(frontier: torch.Tensor, count: int) -> torch.Tensor:
    if len(frontier) < count:
        raise ValueError(f"Source bank has only {len(frontier)}/{count} required rows")
    order = torch.argsort(frontier)
    if len(frontier) == count:
        return order
    positions = torch.linspace(0, len(frontier) - 1, count).round().to(torch.long)
    return order[positions]


def _validated_stratum(
    bank: dict[str, object],
    *,
    count: int,
    label: int,
    frontier_range: tuple[float, float],
) -> tuple[dict[str, object], torch.Tensor]:
    states = bank["states"]
    if not isinstance(states, dict):
        raise ValueError("Source bank states must be a mapping")
    frontier = _frontier(states)
    low, high = frontier_range
    if not torch.all((frontier >= low) & (frontier <= high)):
        raise ValueError(
            f"Source frontier values must remain within [{low}, {high}]"
        )
    finite = _finite_state_rows(states)
    if not torch.all(finite):
        raise ValueError("Source bank contains nonfinite state rows")
    indices = _stratified_indices(frontier, count)
    selected = _select_nested_rows(states, indices)
    selected[STRATUM_FIELD] = torch.full((count,), label, dtype=torch.long)
    selected["stratified_source_row"] = indices.clone()
    return selected, frontier[indices]


def main() -> int:
    args = _parse_args()
    if args.states_per_stratum < 1:
        raise SystemExit("--states-per-stratum must be positive")

    upper_path = args.upper_bank.expanduser().resolve()
    lower_path = args.lower_bank.expanduser().resolve()
    upper_bank = load_walk_state_bank(upper_path)
    lower_bank = load_walk_state_bank(lower_path)
    for source_path, bank in ((upper_path, upper_bank), (lower_path, lower_bank)):
        metadata = bank["metadata"]
        if metadata.get("terrain_level") != 2:
            raise SystemExit(f"Source bank is not hard terrain level 2: {source_path}")
        if metadata.get("joint_names") != upper_bank["metadata"].get("joint_names"):
            raise SystemExit("Upper and lower source banks use different joint order")

    upper, upper_frontier = _validated_stratum(
        upper_bank,
        count=args.states_per_stratum,
        label=UPPER_STRATUM,
        frontier_range=(0.50, 0.90),
    )
    lower, lower_frontier = _validated_stratum(
        lower_bank,
        count=args.states_per_stratum,
        label=LOWER_STRATUM,
        frontier_range=(0.40, 0.60),
    )
    states = concatenate_walk_state_rows((lower, upper))
    count = walk_state_count(states)
    expected = 2 * args.states_per_stratum
    if count != expected:
        raise RuntimeError(f"Stratified bank has {count}/{expected} rows")
    labels = states[STRATUM_FIELD]
    if int((labels == LOWER_STRATUM).sum().item()) != args.states_per_stratum:
        raise RuntimeError("Lower reverse-curriculum stratum is not balanced")
    if int((labels == UPPER_STRATUM).sum().item()) != args.states_per_stratum:
        raise RuntimeError("Upper reverse-curriculum stratum is not balanced")

    digests = {
        _exact_state_row_digest(states, row)
        for row in range(walk_state_count(states))
    }
    if len(digests) != expected:
        raise RuntimeError("Stratified bank contains duplicate exact states")

    bank: dict[str, object] = {
        "schema_version": BANK_SCHEMA_VERSION,
        "metadata": {
            "created_at": datetime.now(UTC).isoformat(),
            "method": "balanced_stratified_reverse_curriculum",
            "num_states": count,
            "states_per_stratum": args.states_per_stratum,
            "stratum_field": STRATUM_FIELD,
            "strata": {
                "lower": {
                    "value": LOWER_STRATUM,
                    "frontier_range": [0.40, 0.60],
                    "frontier_p50": float(lower_frontier.quantile(0.5).item()),
                },
                "upper": {
                    "value": UPPER_STRATUM,
                    "frontier_range": [0.50, 0.90],
                    "frontier_p50": float(upper_frontier.quantile(0.5).item()),
                },
            },
            "upper_bank": str(upper_path),
            "upper_bank_sha256": _sha256(upper_path),
            "lower_bank": str(lower_path),
            "lower_bank_sha256": _sha256(lower_path),
            "terrain_level": 2,
            "riser_height_m": STANDARD_RISER_HEIGHT_M,
            "tread_depth_m": STANDARD_TREAD_DEPTH_M,
            "num_steps": STANDARD_NUM_STEPS,
            "joint_names": upper_bank["metadata"]["joint_names"],
            "physics_dt": upper_bank["metadata"]["physics_dt"],
            "step_dt": upper_bank["metadata"]["step_dt"],
            "decimation": upper_bank["metadata"]["decimation"],
        },
        "states": states,
    }
    output = args.output.expanduser().resolve()
    _write_bank_atomic(output, bank)
    print(f"[stratified-shell-bank] wrote {count} states to {output}")
    print(
        "[stratified-shell-bank] "
        f"lower_p50={bank['metadata']['strata']['lower']['frontier_p50']:.4f} "
        f"upper_p50={bank['metadata']['strata']['upper']['frontier_p50']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


SCRIPT = Path(__file__).parents[1] / "scripts" / "capture_stair_stage2_bank.py"
SPEC = importlib.util.spec_from_file_location("capture_stair_stage2_bank", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
collector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collector)


def test_genuine_stage2_edges_enforce_reset_order_and_first_edge() -> None:
    selected = collector._genuine_stage2_edges(
        stage1=torch.tensor([True, True, False, True, True, True]),
        stage15=torch.tensor([True, True, True, False, True, True]),
        stage2=torch.ones(6, dtype=torch.bool),
        previous_stage2=torch.tensor([False, False, False, False, True, False]),
        episode_steps=torch.tensor([2, 3, 3, 3, 3, 3]),
        dones=torch.tensor([False, False, False, False, False, False]),
        nan_termination=torch.tensor([False, False, False, False, False, True]),
        already_captured=torch.tensor([False, False, False, False, False, False]),
        reset_family=torch.tensor([3, 3, 3, 3, 3, 3]),
        reset_mode=torch.tensor([3, 3, 3, 3, 3, 3]),
        source_rows=torch.arange(6),
    )

    assert selected.tolist() == [False, True, False, False, False, False]


def test_exact_state_digest_covers_nested_full_state() -> None:
    states = {
        "root_qpos_local": torch.tensor([[1.0], [1.0]]),
        "commands": {"twist": torch.tensor([[0.1], [0.2]])},
    }

    first = collector._exact_state_row_digest(states, 0)
    second = collector._exact_state_row_digest(states, 1)

    assert first != second
    states["commands"]["twist"][1] = 0.1
    assert first == collector._exact_state_row_digest(states, 1)


def test_atomic_bank_write_replaces_destination(tmp_path: Path) -> None:
    output = tmp_path / "bank.pt"
    output.write_bytes(b"old")
    bank = {
        "schema_version": collector.BANK_SCHEMA_VERSION,
        "metadata": {"num_states": 1},
        "states": {"root_qpos_local": torch.zeros(1, 7)},
    }

    collector._write_bank_atomic(output, bank)

    loaded = torch.load(output, map_location="cpu", weights_only=False)
    assert loaded["metadata"]["num_states"] == 1
    assert list(tmp_path.glob("*.tmp")) == []


def test_requested_defaults_are_bounded() -> None:
    args = collector._build_arg_parser().parse_args([])

    assert args.checkpoint == collector.DEFAULT_CHECKPOINT
    assert args.input_bank == collector.DEFAULT_INPUT_BANK
    assert args.output == collector.DEFAULT_OUTPUT
    assert args.target_states == 128
    assert args.num_envs == 256
    assert args.max_steps == 4_000

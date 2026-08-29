from types import SimpleNamespace

import torch

from mjlab_microduck.tasks.stair_walk_state_bank import (
    _canonicalize_root_heading,
    _preappend_history,
    _restore_circular_rows,
    concatenate_walk_state_rows,
    eligible_walk_state_rows,
)


def test_preappend_history_reconstructs_state_before_newest_frame():
    history = torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]])
    restored = _preappend_history(history)
    assert torch.equal(restored, torch.tensor([[[1.0], [1.0]], [[3.0], [3.0]]]))


def test_circular_restore_preserves_shared_pointer_and_row_order():
    circular = SimpleNamespace(
        _device="cpu",
        _max_len=3,
        _batch_size=2,
        _pointer=1,
        _buffer=torch.zeros(3, 2, 1),
        _num_pushes=torch.zeros(2, dtype=torch.long),
    )
    ids = torch.tensor([1])
    history = torch.tensor([[[10.0], [20.0], [30.0]]])
    _restore_circular_rows(circular, ids, history, torch.tensor([7]))

    start = (circular._pointer + 1) % circular._max_len
    order = [(start + index) % circular._max_len for index in range(3)]
    assert torch.equal(circular._buffer[order, 1], history[0])
    assert int(circular._num_pushes[1]) == 7
    assert circular._pointer == 1


def test_nested_walker_state_chunks_concatenate_along_state_axis():
    chunks = [
        {
            "root_qpos_local": torch.zeros(1, 7),
            "nested": {"x": torch.ones(1, 2)},
        },
        {
            "root_qpos_local": torch.ones(2, 7),
            "nested": {"x": torch.zeros(2, 2)},
        },
    ]
    merged = concatenate_walk_state_rows(chunks)
    assert merged["root_qpos_local"].shape == (3, 7)
    assert merged["nested"]["x"].shape == (3, 2)


def test_heading_canonicalization_preserves_motion_in_forward_frame():
    half_sqrt = 2.0**-0.5
    qpos = torch.tensor([[0.0, 0.2, 0.1, half_sqrt, 0.0, 0.0, half_sqrt]])
    qvel = torch.tensor([[0.0, 0.4, 0.1, 0.0, 2.0, 0.0]])

    aligned_qpos, aligned_qvel = _canonicalize_root_heading(qpos, qvel)

    assert torch.allclose(
        aligned_qpos[:, 3:7], torch.tensor([[1.0, 0.0, 0.0, 0.0]]), atol=1e-6
    )
    assert torch.allclose(
        aligned_qvel[:, :3], torch.tensor([[0.4, 0.0, 0.1]]), atol=1e-6
    )
    assert torch.equal(aligned_qvel[:, 3:], qvel[:, 3:])


def test_dynamic_phase_filter_rejects_backward_low_and_crash_rows():
    states = {
        "root_qpos_local": torch.tensor(
            [
                [0.0, 0.0, 0.12, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.05, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.12, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.12, 1.0, 0.0, 0.0, 0.0],
            ]
        ),
        "root_qvel": torch.tensor(
            [
                [0.50, 0.0, 0.20, 0.0, 0.0, 0.0],
                [0.50, 0.0, 0.20, 0.0, 0.0, 0.0],
                [-0.10, 0.0, 0.20, 0.0, 0.0, 0.0],
                [0.50, 0.0, -0.80, 0.0, 0.0, 0.0],
            ]
        ),
        "source_episode_step": torch.tensor([30, 30, 30, 30]),
    }

    rows = eligible_walk_state_rows(
        states,
        source_episode_step_range=(15, 60),
        min_forward_speed=0.20,
        min_vertical_speed=-0.25,
        min_root_height=0.08,
    )

    assert torch.equal(rows, torch.tensor([0]))


def test_vault_momentum_accepts_forward_upward_or_pivot_motion():
    states = {
        "root_qpos_local": torch.zeros(4, 7),
        "root_qvel": torch.tensor(
            [
                [0.13, 0.0, 0.00, 0.0, 0.0, 0.0],
                [0.00, 0.0, 0.00, 0.0, 2.0, 0.0],
                [0.00, 0.0, 0.13, 0.0, 0.0, 0.0],
                [-0.20, 0.0, -0.20, 0.0, -2.0, 0.0],
            ]
        ),
    }

    rows = eligible_walk_state_rows(
        states,
        min_vault_momentum=0.12,
        vault_lever_arm=0.06,
    )

    assert torch.equal(rows, torch.tensor([0, 1, 2]))


def test_vault_filter_rejects_sideways_offset_and_yawing_rows():
    states = {
        "root_qpos_local": torch.tensor(
            [
                [0.60, 0.02, 0.12, 1.0, 0.0, 0.0, 0.0],
                [0.60, 0.12, 0.12, 1.0, 0.0, 0.0, 0.0],
                [0.60, 0.02, 0.12, 1.0, 0.0, 0.0, 0.0],
                [0.60, 0.02, 0.12, 1.0, 0.0, 0.0, 0.0],
            ]
        ),
        "root_qvel": torch.tensor(
            [
                [0.20, 0.05, 0.10, 0.0, 0.0, 1.0],
                [0.20, 0.05, 0.10, 0.0, 0.0, 1.0],
                [0.20, 0.30, 0.10, 0.0, 0.0, 1.0],
                [0.20, 0.05, 0.10, 0.0, 0.0, 5.0],
            ]
        ),
    }

    rows = eligible_walk_state_rows(
        states,
        min_vault_momentum=0.12,
        max_abs_local_y=0.08,
        max_abs_lateral_speed=0.20,
        max_abs_yaw_rate=4.0,
    )

    assert torch.equal(rows, torch.tensor([0]))

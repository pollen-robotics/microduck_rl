"""Tests for the language-conditioned velocity env (Phase 2).

No simulator/GPU needed: the cfg is inspected statically, and the two command
terms are exercised against a tiny fake env (they only use ``num_envs`` /
``device`` / ``command_manager``).
"""

from __future__ import annotations

import pytest
import torch

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_vla_env_cfg import (
    MicroduckVlaRlCfg,
    make_microduck_vla_env_cfg,
)

NUM_COMMANDS = 13
PROPRIO_TERMS = ("base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions")
NUMERIC_COMMAND_TERMS = ("command", "head_command", "body_command")


# --------------------------------------------------------------------------
# Config structure
# --------------------------------------------------------------------------


def test_vla_task_is_registered():
    import mjlab_microduck.tasks  # noqa: F401  (populates the registry)
    from mjlab.tasks.registry import list_tasks

    assert "Mjlab-VLA-Flat-MicroDuck" in list_tasks()


def test_commands_are_one_instruction_term_plus_slices_in_order():
    cfg = make_microduck_vla_env_cfg(encoder="hash")
    # Insertion order matters: the parent must precede the slice views.
    assert list(cfg.commands) == ["instruction", "twist", "head_pose", "body_pose"]

    instruction = cfg.commands["instruction"]
    assert isinstance(instruction, microduck_mdp.InstructionCommandCfg)
    assert instruction.encoder == "hash"

    slices = {
        name: cfg.commands[name]
        for name in ("twist", "head_pose", "body_pose")
    }
    assert all(isinstance(s, microduck_mdp.CommandSliceCfg) for s in slices.values())
    assert all(s.parent_name == "instruction" for s in slices.values())


def test_slice_ranges_partition_the_13d_command_block():
    cfg = make_microduck_vla_env_cfg(encoder="hash")
    intervals = [
        (cfg.commands[name].start, cfg.commands[name].stop)
        for name in ("twist", "head_pose", "body_pose")
    ]
    assert intervals == [(0, 3), (3, 7), (7, 13)]
    # contiguous, no gaps/overlaps, covering the whole command vector
    assert intervals[0][0] == 0
    assert intervals[-1][1] == NUM_COMMANDS
    for (_, stop), (start, _) in zip(intervals, intervals[1:]):
        assert stop == start


def test_actor_loses_the_numeric_command_and_gains_text():
    cfg = make_microduck_vla_env_cfg(encoder="hash")

    actor_terms = cfg.observations["actor"].terms
    for term in NUMERIC_COMMAND_TERMS:
        assert term not in actor_terms, f"actor must not see numeric {term!r}"
    assert set(actor_terms) == set(PROPRIO_TERMS)

    assert "text" in cfg.observations
    text_group = cfg.observations["text"]
    assert text_group.enable_corruption is False  # embeddings are never noised
    assert list(text_group.terms) == ["instruction"]
    term_cfg = text_group.terms["instruction"]
    assert term_cfg.func is microduck_mdp.text_instruction_embedding
    assert term_cfg.params["command_name"] == "instruction"


def test_critic_keeps_the_privileged_numeric_command():
    cfg = make_microduck_vla_env_cfg(encoder="hash")
    for term in NUMERIC_COMMAND_TERMS:
        assert term in cfg.observations["critic"].terms


def test_irrelevant_command_curricula_are_dropped():
    cfg = make_microduck_vla_env_cfg(encoder="hash")
    for name in ("standing_envs", "head_pose_range", "body_pose_range"):
        assert name not in cfg.curriculum
    # curriculum-freezing terms that do not touch commands stay
    assert "com_range" in cfg.curriculum
    assert "head_pose_bias_weight" in cfg.curriculum


def test_rl_cfg_uses_its_own_experiment_and_reads_the_text_group():
    assert MicroduckVlaRlCfg.experiment_name == "vla"
    assert MicroduckVlaRlCfg.run_name == "vla"
    assert MicroduckVlaRlCfg.obs_groups == {
        "actor": ("actor", "text"),
        "critic": ("critic",),
    }
    # A text command is not left/right symmetric.
    assert MicroduckVlaRlCfg.algorithm.symmetry_cfg is None


# --------------------------------------------------------------------------
# Command terms against a fake env
# --------------------------------------------------------------------------


class _FakeCommandManager:
    def __init__(self, terms: dict | None = None) -> None:
        self._terms = terms or {}

    def get_term(self, name: str):
        return self._terms[name]


class _FakeEnv:
    def __init__(self, num_envs: int = 8, terms: dict | None = None) -> None:
        self.num_envs = num_envs
        self.device = "cpu"
        self.command_manager = _FakeCommandManager(terms)


def _build_instruction(num_envs: int = 8, dim: int = 64):
    cfg = microduck_mdp.InstructionCommandCfg(
        resampling_time_range=(4.0, 8.0),
        encoder="hash",
        encoder_dim=dim,
    )
    env = _FakeEnv(num_envs=num_envs)
    term = cfg.build(env)
    return term, env


def test_instruction_command_grounds_text_and_numeric_command():
    term, env = _build_instruction()
    assert term.command.shape == (env.num_envs, NUM_COMMANDS)
    assert term.text_embedding.shape == (env.num_envs, 64)

    for row in range(env.num_envs):
        intent = int(term.intent_index[row])
        sentence = int(term.sentence_index[row])
        # text embedding is exactly the table row for this env's sentence
        torch.testing.assert_close(term.text_embedding[row], term.embeddings[sentence])
        # numeric command is exactly the intent's command vector
        torch.testing.assert_close(
            term.command[row], torch.as_tensor(term.pool.command_vectors[intent])
        )
        # sentence -> intent table agrees
        assert int(term.pool.sentence_intent[sentence]) == intent


def test_instruction_command_resamples_only_the_requested_envs():
    term, env = _build_instruction()
    before_command = term.command.clone()
    before_sentence = term.sentence_index.clone()

    term._sample(torch.tensor([1, 3]))

    for row in range(env.num_envs):
        if row in (1, 3):
            continue  # resampled; the new value may coincidentally match
        assert term.sentence_index[row] == before_sentence[row]
        torch.testing.assert_close(term.command[row], before_command[row])


@pytest.mark.parametrize(
    ("name", "start", "stop"),
    [("twist", 0, 3), ("head_pose", 3, 7), ("body_pose", 7, 13)],
)
def test_command_slice_is_a_read_only_view_of_the_parent(name, start, stop):
    term, env = _build_instruction()
    env.command_manager._terms["instruction"] = term

    cfg = microduck_mdp.CommandSliceCfg(
        resampling_time_range=(1.0e9, 1.0e9),
        parent_name="instruction",
        start=start,
        stop=stop,
    )
    slice_term = cfg.build(env)

    assert slice_term.command.shape == (env.num_envs, stop - start), name
    torch.testing.assert_close(slice_term.command, term.command[:, start:stop])

    # Read-only: the slice has no buffer of its own. A write through the parent
    # must be visible through the slice (view semantics, not a copy).
    with torch.no_grad():
        term.command[0, start] = 123.0
    assert slice_term.command[0, 0] == 123.0


def test_command_slice_rejects_empty_ranges():
    cfg = microduck_mdp.CommandSliceCfg(
        resampling_time_range=(1.0e9, 1.0e9), start=3, stop=3
    )
    with pytest.raises(ValueError, match="stop > start"):
        cfg.build(_FakeEnv())

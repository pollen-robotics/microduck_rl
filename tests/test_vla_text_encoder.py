"""Tests for the language-conditioned input plumbing (Phase 1).

These run on CPU with no network and no transformer: ``HashTextEncoder`` is
the dependency-free backend. The real encoder is only exercised on a machine
with the optional ``vla`` extra.
"""

from __future__ import annotations

import numpy as np
import pytest

from mjlab_microduck.vla import (
    COMMAND_LAYOUT,
    HFTextEncoder,
    HashTextEncoder,
    InstructionIntent,
    InstructionPool,
    NUM_COMMANDS,
)


# --------------------------------------------------------------------------
# Instruction table
# --------------------------------------------------------------------------


def test_command_layout_matches_the_policy_contract():
    """The 13D layout is an AGENTS.md invariant: twist, head_pose, body_pose."""
    assert COMMAND_LAYOUT == (
        "twist_x",
        "twist_y",
        "twist_z",
        "neck_pitch",
        "head_pitch",
        "head_yaw",
        "head_roll",
        "body_x",
        "body_y",
        "body_z",
        "body_roll",
        "body_pitch",
        "body_yaw",
    )
    assert NUM_COMMANDS == 13


def test_instruction_table_loads_and_is_well_formed():
    pool = InstructionPool.from_json()
    assert pool.num_intents >= 8
    assert pool.num_sentences == sum(
        len(intent.paraphrases) for intent in pool.intents
    )
    assert len(set(pool.intent_names)) == pool.num_intents

    for intent in pool.intents:
        assert intent.command.shape == (NUM_COMMANDS,)
        # Free-sentence generalization depends on paraphrase coverage: one
        # sentence per intent would just memorize one embedding.
        assert len(intent.paraphrases) >= 5, intent.name
        assert len(set(intent.paraphrases)) == len(intent.paraphrases)


def test_intents_ground_to_expected_command_slots():
    """Twist-only intents must zero the head/body slots and vice versa."""
    pool = InstructionPool.from_json()
    by_name = {intent.name: intent for intent in pool.intents}

    forward = by_name["walk_forward"].command
    assert forward[0] > 0.0  # twist_x
    assert np.all(forward[1:3] == 0.0)
    assert np.all(forward[3:] == 0.0)  # head_pose + body_pose zero-padded

    look_up = by_name["look_up"].command
    assert np.all(look_up[0:3] == 0.0)
    assert look_up[4] > 0.0  # head_pitch slot
    assert np.all(look_up[7:] == 0.0)

    stand = by_name["stand"].command
    assert np.all(stand == 0.0)


def test_duplicate_paraphrases_are_rejected():
    intents = [
        InstructionIntent("a", np.zeros(NUM_COMMANDS, np.float32), ("同一个句子",)),
        InstructionIntent("b", np.zeros(NUM_COMMANDS, np.float32), ("同一个句子",)),
    ]
    with pytest.raises(ValueError, match="duplicate paraphrase"):
        InstructionPool(intents)


def test_bad_command_shape_is_rejected():
    with pytest.raises(ValueError, match="shape"):
        InstructionIntent("bad", np.zeros(3, np.float32), ("x",))


# --------------------------------------------------------------------------
# Encoders
# --------------------------------------------------------------------------


def test_hash_encoder_is_deterministic_normalized_float32():
    encoder = HashTextEncoder(dim=32)
    texts = ["往前走", "walk forward", "往前走"]

    first = encoder.encode(texts)
    second = encoder.encode(texts)

    assert first.shape == (3, 32)
    assert first.dtype == np.float32
    np.testing.assert_array_equal(first, second)
    # Identical texts -> identical rows.
    np.testing.assert_array_equal(first[0], first[2])
    norms = np.linalg.norm(first, axis=1)
    assert np.allclose(norms, 1.0)


def test_hash_encoder_gives_similar_texts_similar_vectors():
    encoder = HashTextEncoder(dim=128)
    vector = encoder.encode(["walk forward", "walk", "完全无关"])
    cosine_similar_walk = float(np.dot(vector[0], vector[1]))
    cosine_unrelated = float(np.dot(vector[0], vector[2]))
    assert cosine_similar_walk > cosine_unrelated


def test_hf_encoder_without_optional_dependency_has_actionable_error():
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="vla"):
            HFTextEncoder()
    else:  # pragma: no cover - depends on environment
        pytest.skip("sentence-transformers is installed")


# --------------------------------------------------------------------------
# Embedding + sampling
# --------------------------------------------------------------------------


def _embedded_pool(dim: int = 64, seed: int = 0) -> tuple[InstructionPool, HashTextEncoder]:
    table = InstructionPool.from_json()
    # The table declares the deployment encoder's dim (384); rebuild with the
    # test encoder's dim so embed() performs its consistency check against 64.
    pool = InstructionPool(
        table.intents,
        encoder_model="hash-test",
        encoder_dim=dim,
        version=table.version,
    )
    encoder = HashTextEncoder(dim=dim)
    pool.embed(encoder)
    return pool, encoder


def test_embed_builds_the_lookup_table():
    pool, _ = _embedded_pool(dim=64)
    assert pool.is_embedded
    assert pool.embeddings.shape == (pool.num_sentences, 64)
    assert pool.embeddings.dtype == np.float32


def test_embed_checks_the_declared_dimension():
    table = InstructionPool.from_json()  # declares encoder_dim=384
    with pytest.raises(ValueError, match="does not match"):
        table.embed(HashTextEncoder(dim=64))


def test_sample_requires_embeddings():
    pool = InstructionPool.from_json()
    with pytest.raises(RuntimeError, match="embed"):
        pool.sample(4)


def test_sample_shapes_and_grounding_are_consistent():
    pool, _ = _embedded_pool(dim=64)
    cmd13 = pool.command_vectors

    rng = np.random.default_rng(123)
    sample = pool.sample(256, rng)

    assert sample.intent_index.shape == (256,)
    assert sample.sentence_index.shape == (256,)
    assert len(sample.sentences) == 256
    assert sample.commands.shape == (256, NUM_COMMANDS)
    assert sample.embeddings.shape == (256, 64)

    for row in range(256):
        intent_id = int(sample.intent_index[row])
        sentence_id = int(sample.sentence_index[row])
        # sentence -> intent table agrees with the sampled intent
        assert int(pool.sentence_intent[sentence_id]) == intent_id
        assert sample.sentences[row] == pool.sentences[sentence_id]
        # numeric command and embedding rows agree with the tables
        np.testing.assert_array_equal(sample.commands[row], cmd13[intent_id])
        np.testing.assert_array_equal(
            sample.embeddings[row], pool.embeddings[sentence_id]
        )


def test_sampling_is_balanced_over_intents_and_covers_all_intents():
    pool, _ = _embedded_pool(dim=64)
    rng = np.random.default_rng(7)
    n = 12_000
    sample = pool.sample(n, rng)

    counts = np.bincount(sample.intent_index, minlength=pool.num_intents)
    assert np.all(counts > 0), "every intent must be reachable"

    # Intent is sampled uniformly, then a paraphrase within it: the counts
    # should be close to n / num_intents regardless of paraphrase counts.
    expected = n / pool.num_intents
    assert counts.max() / expected < 1.25
    assert counts.min() / expected > 0.75

    # Every paraphrase of every intent must be reachable too.
    for intent_id, intent in enumerate(pool.intents):
        seen = set(sample.sentences[i] for i in np.flatnonzero(sample.intent_index == intent_id))
        assert seen == set(intent.paraphrases), intent.name


def test_sample_zero_is_allowed():
    pool, _ = _embedded_pool(dim=64)
    sample = pool.sample(0)
    assert sample.commands.shape == (0, NUM_COMMANDS)
    assert sample.embeddings.shape == (0, 64)

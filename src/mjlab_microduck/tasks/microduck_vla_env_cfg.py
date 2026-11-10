"""Language-conditioned (free-sentence) velocity environment.

This task is the velocity recipe with ONE structural change: the actor no
longer sees the numeric 13D command block. It sees a text embedding instead.
The numeric command still exists — it drives every reward and is privileged
input for the critic — but both views are produced by the *same* sampled
intent, so text and reward can never disagree.

Layout:

    InstructionCommand ("instruction")  samples one intent -> 13D numeric cmd
        ├── CommandSlice "twist"       = cmd[:, 0:3]    (rewards unchanged)
        ├── CommandSlice "head_pose"   = cmd[:, 3:7]
        ├── CommandSlice "body_pose"   = cmd[:, 7:13]
        └── text_instruction_embedding -> actor obs group "text" (B, D)

    actor  : "actor" (48D proprio) + "text" (D)     <- no numeric command
    critic : "critic" (proprio + 13D numeric command) <- privileged

The instruction set lives in ``mjlab_microduck/vla/instructions.json``. The
embedding table is built once per env process from either a frozen
SentenceTransformer (``encoder="hf"``), a precomputed ``.npz``
(``embedding_cache=...``), or the dependency-free hash backend used by tests
(``encoder="hash"``).

PHASE 3 DONE: ``MicroduckVlaRlCfg.actor.class_name`` points at
``MicroduckVLAModel``, which normalizes only the proprio groups and appends the
raw embedding. The critic stays a stock ``MLPModel`` (it sees the numeric
command, not text).
"""

from __future__ import annotations

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import ObservationGroupCfg, ObservationTermCfg

from . import mdp as microduck_mdp
from .microduck_velocity_env_cfg import (
    MicroduckRlCfg,
    make_microduck_velocity_env_cfg,
)

# Intent resampling period. Instructions are coarser than the old 3-6 s twist
# commands: the policy needs enough steps per instruction to both decode the
# embedding and execute the behavior.
INSTRUCTION_RESAMPLE_S = (4.0, 8.0)

# The slice views never sample; a huge interval keeps their no-op timers inert.
_SLICE_RESAMPLE_S = (1.0e9, 1.0e9)

# Command block layout, shared with the whole policy family.
_TWIST_SLICE = (0, 3)
_HEAD_POSE_SLICE = (3, 7)
_BODY_POSE_SLICE = (7, 13)


def make_microduck_vla_env_cfg(
    play: bool = False,
    rough: bool = False,
    *,
    encoder: str = "hf",
    embedding_cache: str | None = None,
    encoder_dim: int | None = None,
) -> ManagerBasedRlEnvCfg:
    """Create the language-conditioned velocity env.

    Args:
        play: Build the play variant (matches ``make_microduck_velocity_env_cfg``).
        rough: Rough terrain variant.
        encoder: ``"hf"`` (frozen SentenceTransformer, needs the ``vla``
            extra), ``"hash"`` (dependency-free fallback for tests/CI), or
            ``"cache"`` when ``embedding_cache`` is supplied.
        embedding_cache: Optional ``.npz`` with an ``"embeddings"`` array. When
            present, the env needs neither ``transformers`` nor a download.
        encoder_dim: Override the table's declared embedding dim (tests).
    """
    cfg = make_microduck_velocity_env_cfg(play=play, rough=rough)

    # ------------------------------------------------------------------
    # Commands: one intent-driven 13D term + read-only slices.
    # Insertion order matters: "instruction" must precede its slices.
    # ------------------------------------------------------------------
    cfg.commands = {
        "instruction": microduck_mdp.InstructionCommandCfg(
            resampling_time_range=INSTRUCTION_RESAMPLE_S,
            encoder=encoder,
            embedding_cache=embedding_cache,
            encoder_dim=encoder_dim,
        ),
        "twist": microduck_mdp.CommandSliceCfg(
            resampling_time_range=_SLICE_RESAMPLE_S,
            parent_name="instruction",
            start=_TWIST_SLICE[0],
            stop=_TWIST_SLICE[1],
        ),
        "head_pose": microduck_mdp.CommandSliceCfg(
            resampling_time_range=_SLICE_RESAMPLE_S,
            parent_name="instruction",
            start=_HEAD_POSE_SLICE[0],
            stop=_HEAD_POSE_SLICE[1],
        ),
        "body_pose": microduck_mdp.CommandSliceCfg(
            resampling_time_range=_SLICE_RESAMPLE_S,
            parent_name="instruction",
            start=_BODY_POSE_SLICE[0],
            stop=_BODY_POSE_SLICE[1],
        ),
    }

    # ------------------------------------------------------------------
    # Observations: actor loses the numeric command, gains the text group.
    # The critic keeps the numeric command (privileged) and does not need text.
    # ------------------------------------------------------------------
    for term in ("command", "head_command", "body_command"):
        cfg.observations["actor"].terms.pop(term, None)

    cfg.observations["text"] = ObservationGroupCfg(
        terms={
            "instruction": ObservationTermCfg(
                func=microduck_mdp.text_instruction_embedding,
                params={"command_name": "instruction"},
            ),
        },
        concatenate_terms=True,
        enable_corruption=False,  # never noise a frozen embedding
    )

    # ------------------------------------------------------------------
    # Curricula: the intent table is fixed, so the old command-range and
    # standing-fraction ramps no longer apply (and their targets are now
    # read-only slices).
    # ------------------------------------------------------------------
    for name in ("standing_envs", "head_pose_range", "body_pose_range"):
        cfg.curriculum.pop(name, None)

    return cfg


# --------------------------------------------------------------------------
# RL config
# --------------------------------------------------------------------------
# The actor normalizes proprio only and appends the raw embedding. The critic
# keeps the stock MLPModel (numeric command, no text).
MicroduckVlaRlCfg = deepcopy(MicroduckRlCfg)
MicroduckVlaRlCfg.experiment_name = "vla"
MicroduckVlaRlCfg.run_name = "vla"
MicroduckVlaRlCfg.obs_groups = {
    "actor": ("actor", "text"),
    "critic": ("critic",),
}
MicroduckVlaRlCfg.actor.class_name = (
    "mjlab_microduck.vla.model:MicroduckVLAModel"
)
# The 61D mirror-loss table has no meaning for a text embedding, and a text
# command is not left/right symmetric. Force it off even if the source recipe
# later enables it.
MicroduckVlaRlCfg.algorithm.symmetry_cfg = None

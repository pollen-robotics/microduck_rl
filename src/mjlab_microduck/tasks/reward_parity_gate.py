"""Reusable cross-stack reward parity gate.

The feet_flat_penalty bug (see REWARD_PARITY.md) was a *sign inversion*: the
MJLab side assumed a foot site's local Z points at world-up, the asset rolls
it 90 degrees, so a perfectly flat foot scored the penalty's maximum. Both
stacks trained to completion and produced green numbers — nothing crashed, so
nothing was investigated. The fix is a test that compares the two
implementations at a real, model-derived pose; this module makes that test a
*registry* so the gate is extensible and self-policing.

Two allowlists, not one
----------------------
The naive version of this gate flagged every function that projects world
gravity into a site/geom frame. That is wrong: the *fix itself*
(``_foot_rest_gravity_ref``) does exactly that — it reads the rest orientation
off the model instead of assuming an axis. So a single allowlist would make the
gate red on correct code. We split it:

1. ``REWARD_TERMS`` — reward/penalty terms implemented on BOTH stacks. Each must
   carry a parity check (``covered=True``). Adding a GPU term without one fails
   the census below.
2. ``FRAME_HELPERS`` — internal helpers that legitimately rotate world gravity
   into a site/geom frame. These are the self-calibrating fix pattern, so they
   are audited-but-allowed rather than parity-required.

``scan_uncovered_frame_terms`` fails only on frame-assuming functions in NEITHER
list. That turns "you forgot a parity test" into a red CI, not a postmortem,
without penalizing the fix.

The docstring trap: history notes in these functions quote the OLD formula
(``sum(proj[:, :2] ** 2)``), so a text search finds the explanation of the bug
before the fix. Both scanners parse the AST and strip the docstring first.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import ast

MDP_PATH = Path(__file__).resolve().parent / "mdp.py"

# Marker pair that identifies "gravity is rotated into a site/geom frame":
# the term reads the per-env gravity vector AND applies a site/geom quaternion
# to it. That is precisely the operation that silently inverts when the asset's
# rest pose does not match the textbook one.
_FRAME_ASSUMING_MARKERS = ("gravity_vec_w", "quat_apply_inverse")


@dataclass
class TermPair:
    """One reward term implemented on both the CPU and GPU stacks."""

    name: str               # human-readable name
    mjlab_func: str         # function name in mdp.py
    covered: bool           # has a parity check in tests/


# Reward/penalty terms implemented on BOTH stacks. Each MUST carry a parity check
# (covered=True) once registered. Adding a GPU term without a parity test fails
# the census below.
REWARD_TERMS: list[TermPair] = [
    TermPair("feet_flat", "feet_flat_penalty", covered=True),
]

# Internal helpers that legitimately rotate world gravity into a site/geom frame.
# These are the SELF-CALIBRATING fix pattern (read the orientation off the model),
# not new reward terms, so they are audited-but-allowed rather than
# parity-required. The census fails only on frame-assuming functions that are in
# NEITHER list.
FRAME_HELPERS: set[str] = {
    "_foot_rest_gravity_ref",
}


def _body_without_docstring(func: ast.FunctionDef, src: str) -> str:
    """Return the function source with any leading docstring constant removed.

    The docstring is a ``Constant`` expression statement and is exactly where
    old formulas get quoted — stripping it stops a text/syntax scan from
    matching the *explanation* of a bug instead of the code.
    """
    body = func.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    return "\n".join(ast.unparse(stmt) for stmt in body)


def _is_frame_assuming(func: ast.FunctionDef, src: str) -> bool:
    body = _body_without_docstring(func, src)
    return all(m in body for m in _FRAME_ASSUMING_MARKERS)


def scan_uncovered_frame_terms(mdp_src: str | None = None) -> list[str]:
    """Frame-assuming functions in mdp.py that are accounted for by NEITHER list.

    Returns the names of frame-assuming functions with no parity registration
    and no helper whitelist entry. An empty list means the GPU stack is
    self-policed; non-empty means a new term landed without a cross-stack check
    and CI should fail.
    """
    src = mdp_src if mdp_src is not None else MDP_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)
    allowed = {t.mjlab_func for t in REWARD_TERMS} | set(FRAME_HELPERS)
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and _is_frame_assuming(node, src):
            if node.name not in allowed:
                offenders.append(node.name)
    return offenders


def registered_terms() -> list[TermPair]:
    """Exposed for the test that asserts every registered term is covered."""
    return list(REWARD_TERMS)


def frame_helpers() -> set[str]:
    """Exposed for the test that asserts the self-calibrating helper is known."""
    return set(FRAME_HELPERS)

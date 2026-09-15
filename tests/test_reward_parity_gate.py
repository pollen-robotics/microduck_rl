"""Unit tests for the reusable reward parity gate (tasks/reward_parity_gate.py).

The gate is the single source of truth for "which frame-assuming functions are
accounted for". This file checks two things:

1. The gate PASSES on the real mdp.py — i.e. every function that projects world
   gravity into a site/geom frame is either a registered reward term
   (REWARD_TERMS, covered=True) or the sanctioned self-calibrating helper
   (FRAME_HELPERS). In particular the fix itself, ``_foot_rest_gravity_ref``,
   must NOT trip the gate.
2. The gate actually CATCHES a new, unregistered frame-assuming term — so it is
   a real backstop rather than a vacuous pass. A gate that cannot fail is not a
   gate.

The gate is imported by path (importlib) so the test runs even where the mjlab
package is not pip-installed; the module only depends on the stdlib.
"""

from pathlib import Path

import importlib.util
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
MDP_PATH = REPO / "src" / "mjlab_microduck" / "tasks" / "mdp.py"
GATE_PATH = MDP_PATH.parent / "reward_parity_gate.py"


def _load_gate():
    # Register in sys.modules before exec: the module uses a dataclass, and
    # dataclasses inspects cls.__module__ via sys.modules — a module loaded only
    # by path with no sys.modules entry raises AttributeError on the dataclass.
    spec = importlib.util.spec_from_file_location("reward_parity_gate", GATE_PATH)
    gate = importlib.util.module_from_spec(spec)
    sys.modules["reward_parity_gate"] = gate
    spec.loader.exec_module(gate)
    return gate


def test_real_mdp_has_no_uncovered_frame_terms():
    """The current mdp.py must be self-policed: no off-list frame terms."""
    gate = _load_gate()
    offenders = gate.scan_uncovered_frame_terms()
    assert offenders == [], (
        f"frame-assuming reward terms with no parity check: {offenders}. "
        "Register each in REWARD_TERMS (covered=True) or, if it is the "
        "self-calibrating helper, in FRAME_HELPERS."
    )


def test_gate_catches_a_new_unregistered_term():
    """A brand-new frame-assuming reward term with no registration must fail."""
    gate = _load_gate()
    src = '''
import torch.nn.functional as F
from mjlab.utils.lab_api.math import quat_apply_inverse

def fresh_foot_pitch_penalty(env):
    """A new term that projects world gravity into a site frame."""
    gravity_w_n = F.normalize(env.asset.data.gravity_vec_w, dim=-1)
    proj = quat_apply_inverse(env.asset.data.site_quat_w, gravity_w_n)
    return torch.sum(torch.square(proj), dim=-1)
'''
    offenders = gate.scan_uncovered_frame_terms(src)
    assert "fresh_foot_pitch_penalty" in offenders


def test_gate_does_not_flag_the_self_calibrating_helper():
    """The fix helper _foot_rest_gravity_ref must NOT be an offender."""
    gate = _load_gate()
    offenders = gate.scan_uncovered_frame_terms()
    assert "_foot_rest_gravity_ref" not in offenders


def test_registered_feet_flat_is_covered():
    """The one term we found and fixed is registered and marked covered."""
    gate = _load_gate()
    terms = {t.mjlab_func: t for t in gate.registered_terms()}
    assert "feet_flat_penalty" in terms
    assert terms["feet_flat_penalty"].covered is True


def test_frame_helpers_contains_self_calibrating_ref():
    """The whitelist knows the sanctioned helper, so the census can skip it."""
    gate = _load_gate()
    assert "_foot_rest_gravity_ref" in gate.frame_helpers()

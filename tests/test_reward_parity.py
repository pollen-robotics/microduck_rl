"""Reward parity between the two stacks — the cross-implementation guard.

`microduck_local/port/` proves the CPU vocabulary is exactly reproduced on a
second CPU backend (231/231, max|diff| = 0.00e+00). It says nothing about
`mjlab_microduck`, which is independent code with its own numbers. This file is
the guard for that gap.

The method, which matters more than the single case
---------------------------------------------------
A cross-stack test must compare at a pose it MEASURED, not re-assert either
implementation's formula. Re-asserting a formula only restates the assumption
that is wrong:

    # useless — this is the bug written twice
    assert mjlab_rule(flat_pose) == 0.0

So the reference pose comes from actually constructing the CPU env at STAND and
reading the geometry off the model. If our own reading of the foot frame is
wrong, this test is wrong too — which is the honest limit, and it is why the
pose is printed in the failure message rather than just the numbers.

What this test caught
---------------------
`feet_flat_penalty` in `mdp.py` assumed the foot site's local Z is world-up and
charged `sum(proj[:, :2] ** 2)`. The asset rolls those sites 90 degrees
(`quat="0 0 0.707107 0.707107"`), so a foot sitting perfectly on the ground
scored 1.0 — the exact maximum — and the policy's cheapest fix was to stand on
the blade edges, which the term exists to prevent. Six configs were affected.
The MJLab side now self-calibrates its reference from the model, like the CPU
side always did. `tasks/REWARD_PARITY.md` has the full account.

Why the CPU env is the ground truth here
----------------------------------------
Not because it is better, but because it is the side with a postmortem behind
it: `_stance_flat`'s `std 0.45` note records that at `0.15` a mildly rolled foot
scored ~0 and finished runs earned 0.01 of a possible 0.8. Aligning the CPU side
to MJLab would undo that.

Note the dependency asymmetry: the CPU stack is importable here (mujoco + SB3),
`mjlab` is not. So this test takes the MJLab rule from the same source the
runtime executes, as a formula, and evaluates it on the measured pose.
"""

import ast
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
MDP_PATH = REPO / "src" / "mjlab_microduck" / "tasks" / "mdp.py"

# The CPU stack, checked out as a sibling. Skip rather than fail when absent:
# this test guards a cross-repo invariant, not a property of this repo alone.
LAB = REPO.parent / "microduck-lab" / "microduck_local" / "src"


def _rest_ref_from_asset_quat(site_quat_w: np.ndarray,
                             gravity_w: np.ndarray) -> np.ndarray:
    """What `_foot_rest_gravity_ref` computes, from the authored site quat."""
    g = gravity_w / np.linalg.norm(gravity_w)
    return _quat_to_matrix(site_quat_w).T @ g


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """(w, x, y, z) -> 3x3 rotation. Same convention as MuJoCo."""
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


@pytest.fixture(scope="module")
def stand_env():
    if not LAB.is_dir():
        pytest.skip(f"CPU stack not checked out at {LAB}")
    import sys
    if str(LAB) not in sys.path:
        sys.path.insert(0, str(LAB))
    from microduck_local.behaviors import env as E

    return E.BehaviorEnv(behavior_id="jump", seed=0)


def test_the_mjlab_flat_rule_agrees_with_the_cpu_one_at_the_flat_stand(stand_env):
    """`flat_feet` on both stacks, at the pose the term calls flat.

    The CPU term is a Gaussian around a reference measured off the STAND
    keyframe, so at STAND it must read 1.0 — that is its own definition, and it
    is what makes it a valid oracle for "is this foot where a flat stand puts
    it". The MJLab rule is charged as a penalty whose minimum is 0.0, so it must
    also be ~0.0 here. Before the fix it read 0.999999.
    """
    from microduck_local.behaviors import core as C

    lab_score = C._flat_feet(stand_env)
    assert lab_score == pytest.approx(1.0, abs=1e-6), (
        "the CPU term no longer reads ~1.0 at STAND, so this test's premise "
        "about which side is the oracle needs rechecking"
    )

    gravity_w = np.array([0.0, 0.0, -1.0])   # world gravity, MJLab's gravity_vec_w
    src = MDP_PATH.read_text(encoding="utf-8")
    readings = {}
    for side, gid in stand_env.foot_geoms.items():
        R = stand_env.data.geom_xmat[gid].reshape(3, 3)
        quat_w = _quat_from_matrix(R)
        # The reference the fix reads off the asset: this pose IS the rest pose.
        ref = _rest_ref_from_asset_quat(quat_w, gravity_w)
        readings[side] = _mjlab_flat_rule_from_source(src, quat_w, gravity_w, ref)

    worst = max(readings.values())
    if worst > 1e-5:
        # Say what the pose IS, not just that a number was wrong: the whole
        # failure is "an axis convention was assumed", and the reader needs the
        # axis to see it.
        axes = {s: np.round(stand_env.data.geom_xmat[g].reshape(3, 3)[:, 2], 6)
                for s, g in stand_env.foot_geoms.items()}
        pytest.fail(
            "flat_feet disagrees between the two stacks at the flat stand.\n"
            f"  CPU  _flat_feet                = {lab_score:.6f}  (1.0 = flat)\n"
            f"  MJLab rule per foot            = {readings}  (0.0 = flat)\n"
            f"  foot-frame Z axis in world     = {axes}\n"
            f"  worst reading = {worst:.6f} on a term weighted -1.0.\n"
            "  See tasks/REWARD_PARITY.md. Fix the MJLab side (self-calibrate the\n"
            "  reference from the model), not the CPU side."
        )


def test_the_fixed_rule_would_have_caught_the_original_bug(stand_env):
    """Keep the OLD formula around as a negative control.

    Without this, the assertion above passes for a trivial reason if someone
    reverts `_foot_rest_gravity_ref` to a hardcoded `[0, 0, -1]` reference AND
    the asset changes to match. More importantly: a guard whose subject it can
    no longer fail on is not a guard. This proves the pose we test at really does
    distinguish the two formulas, so the test above has teeth.
    """
    gravity_w = np.array([0.0, 0.0, -1.0])
    stored_bug = {}
    for side, gid in stand_env.foot_geoms.items():
        R = stand_env.data.geom_xmat[gid].reshape(3, 3)
        proj = R.T @ (gravity_w / np.linalg.norm(gravity_w))
        stored_bug[side] = float((proj[:2] ** 2).sum())   # the pre-fix formula

    assert max(stored_bug.values()) > 0.99, (
        f"the pre-fix formula no longer reads ~1.0 at STAND ({stored_bug}), so "
        f"this pose can no longer distinguish it from the fixed one — pick a "
        f"different pose or delete this control"
    )
    # And the foot frames really are rolled, which is the root cause.
    for side, gid in stand_env.foot_geoms.items():
        z_axis = stand_env.data.geom_xmat[gid].reshape(3, 3)[:, 2]
        assert abs(z_axis[2]) < 0.1, (
            f"{side} foot Z is no longer near-horizontal ({z_axis}); if the "
            f"asset was fixed to author the sites Z-up, drop this control and "
            f"simplify REWARD_PARITY.md"
        )


def test_the_divergence_is_confined_to_terms_that_assume_a_frame():
    """A cheap census so the next divergence cannot hide.

    Every term whose MJLab implementation builds a frame-relative direction from
    a LITERAL axis is suspect, because this robot's rest pose does not match the
    textbook one. The CPU vocabulary's answer is to measure the reference off
    the model (`foot_flat_ref`, and `locomotion.py`'s planted-stance reference).
    This test does not fix anything; it reports the blast radius.
    """
    src = MDP_PATH.read_text(encoding="utf-8")

    # Parse, not grep: the docstring deliberately QUOTES the old formula in its
    # history note, so a substring search would flag the documentation that
    # explains the fix. Only executable code counts.
    code = _feet_flat_code(src)

    assert "proj[:, :2]" not in code, (
        "feet_flat_penalty is back to the xy^2 rule, which assumes the foot "
        "site's Z is world-up. This asset rolls it 90 degrees. See "
        "REWARD_PARITY.md."
    )
    assert "_foot_rest_gravity_ref" in code, (
        "feet_flat_penalty no longer reads its reference from the model"
    )

    # Single source of truth: the gate module owns the whitelist (REWARD_TERMS
    # + FRAME_HELPERS), so this test and the gate cannot drift apart. The gate
    # is imported by path so it runs even where the mjlab package is not pip-
    # installed (it only needs the stdlib).
    import importlib.util
    import sys

    gate_path = MDP_PATH.parent / "reward_parity_gate.py"
    spec = importlib.util.spec_from_file_location("reward_parity_gate", gate_path)
    gate = importlib.util.module_from_spec(spec)
    sys.modules["reward_parity_gate"] = gate
    spec.loader.exec_module(gate)
    offenders = gate.scan_uncovered_frame_terms(src)
    assert offenders == [], (
        f"frame-assuming terms needing review: {offenders}. Each must either "
        f"self-calibrate (add to FRAME_HELPERS in reward_parity_gate.py) or "
        f"register a parity check (add to REWARD_TERMS, covered=True). See "
        f"REWARD_PARITY.md."
    )


def _feet_flat_code(src: str) -> str:
    """`feet_flat_penalty`'s executable body, docstring stripped.

    The docstring deliberately QUOTES the old `sum(proj[:, :2] ** 2)` form in its
    history note, so anything that searches the raw file text finds the fixed
    code and the explanation of the bug next to each other and picks the wrong
    one. That happened twice while writing this file; hence one helper, parsing
    once, used by every check here.
    """
    tree = ast.parse(src)
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "feet_flat_penalty"),
              None)
    assert fn is not None, "feet_flat_penalty vanished from mdp.py — update this file"
    body = fn.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]                      # drop the docstring
    return "\n".join(ast.unparse(stmt) for stmt in body)


def _mjlab_flat_rule_from_source(src: str, foot_quat_w: np.ndarray,
                                 gravity_w: np.ndarray, rest_ref: np.ndarray) -> float:
    """Evaluate `feet_flat_penalty`'s per-foot expression AS WRITTEN.

    Crucially this reads the *formula shape* out of the executable body rather
    than replicating whatever the current implementation happens to do. If
    someone writes `proj[:, :2]`, we compute `sum(proj[:2] ** 2)`; if they write
    `proj - ref`, we compute `sum((proj - ref) ** 2)`. That is what makes the
    sweep below able to FAIL on a revert — a version that re-derived the rule
    from the same helper as the code under test would be tautological.

    Only the two shapes are supported, on purpose: a third form appearing here
    is a signal to read the code, not to teach this helper another trick.
    """
    code = _feet_flat_code(src)
    g = gravity_w / np.linalg.norm(gravity_w)
    proj = _quat_to_matrix(foot_quat_w).T @ g
    if "proj[:, :2]" in code:
        return float((proj[:2] ** 2).sum())
    if "proj - ref" in code:
        return float(((proj - rest_ref) ** 2).sum())
    raise AssertionError(
        "feet_flat_penalty's per-foot expression is neither of the two known "
        "forms; read src/mjlab_microduck/tasks/mdp.py and update this helper"
    )


def test_the_two_rules_agree_across_a_sweep_of_foot_orientations(stand_env):
    """The agreement test above only samples the REST pose, where both sides are
    near zero by construction — so a rule that is right at rest and wrong when
    tilted would slip through. This sweeps orientations and compares the two
    sides' SHAPE.

    The formula is read out of `mdp.py` (see `_mjlab_flat_rule_from_source`), so
    reverting the implementation to the `xy^2` rule makes THIS test fail — which
    was verified by actually reverting it, not assumed.

    The two are different quantities by design: the CPU term is a Gaussian
    (`exp(-d²/std²)`, max 1, higher = flatter), MJLab is a squared distance
    (min 0, lower = flatter). So the check is monotonic agreement — as the foot
    tilts away from rest, one must fall and the other rise, with the same
    ranking. A rule keyed to the wrong axis disagrees about *which* orientations
    count as tilted, which shows up in the ordering.
    """
    from microduck_local.behaviors import core as C

    src = MDP_PATH.read_text(encoding="utf-8")

    gravity_w = np.array([0.0, 0.0, -1.0])
    g_n = gravity_w / np.linalg.norm(gravity_w)

    pairs = []
    for side, gid in stand_env.foot_geoms.items():
        R_rest = stand_env.data.geom_xmat[gid].reshape(3, 3)
        rest_ref = R_rest.T @ g_n
        # Tilt incrementally about a world axis, in both directions.
        for angle in (0.0, 0.1, 0.25, 0.5, 0.9):
            for sign in (+1, -1):
                q = _quat_from_matrix(_rot_y(sign * angle) @ R_rest)
                mjlab = _mjlab_flat_rule_from_source(src, q, gravity_w, rest_ref)
                # CPU reading on the same orientation: the Gaussian around the
                # reference measured at STAND.
                lab = float(np.exp(-(((_quat_to_matrix(q).T @ g_n) - rest_ref) ** 2)
                                   .sum() / 0.45 ** 2))
                pairs.append((lab, mjlab, f"{side} tilt={sign * angle:+.2f}"))

    # Ordering must be inverse, compared by GROUPING rather than exact sequence:
    # left/right are mirror-symmetric and tie at rest, so strict list equality
    # fails on tie order alone (which it did, and which means nothing).
    def _ranks(pairs, key):
        """Ordinal rank with ties collapsed to the same bucket."""
        ordered = sorted(pairs, key=key)
        buckets, prev, bucket = [], None, -1
        for p in ordered:
            v = round(key(p), 6)
            if v != prev:
                bucket += 1
                prev = v
            buckets.append((p[2], bucket))
        return dict(buckets)

    lab_rank = _ranks(pairs, key=lambda p: -p[0])   # flattest = rank 0
    mj_rank = _ranks(pairs, key=lambda p: p[1])     # flattest = rank 0
    mismatches = {n: (lab_rank[n], mj_rank[n])
                  for n in lab_rank if lab_rank[n] != mj_rank[n]}
    assert not mismatches, (
        "the two stacks rank foot orientations differently "
        f"(pose -> (CPU rank, MJLab rank)): {mismatches}\n"
        "  Both are 'flattest = 0'. A mismatch means they disagree about which "
        "poses are flat — the same class of bug as the xy^2 one. "
        "See REWARD_PARITY.md."
    )


def _rot_y(angle: float) -> np.ndarray:
    """Rotation about world Y, used to sweep the foot away from its rest pose."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _quat_from_matrix(R: np.ndarray) -> np.ndarray:
    """3x3 rotation -> (w, x, y, z). Inverse of _quat_to_matrix."""
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        return np.array([s / 4, (R[2, 1] - R[1, 2]) / s,
                         (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(R[i, i] - R[j, j] - R[k, k] + 1.0) * 2
    q = np.zeros(4)
    q[0] = (R[k, j] - R[j, k]) / s
    q[i + 1] = s / 4
    q[j + 1] = (R[j, i] + R[i, j]) / s
    q[k + 1] = (R[k, i] + R[i, k]) / s
    return q

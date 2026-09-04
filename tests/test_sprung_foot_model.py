"""Sprung-foot robot model: geometry, spring parameters, and compression sign.

These tests legitimately compile a real MjSpec — the thing under test IS the
model, so a duck-typed fake would test nothing.
"""

import mujoco
import numpy as np
import pytest

from mjlab_microduck.robot.sprung_foot import (
    _PAD_HALF_EXTENTS,
    DAMPING_RATIO,
    damping_for,
    H_ADD,
    PAD_MASS,
    SPRING_ARMATURE,
    SPRING_FRICTIONLOSS,
    SPRING_JOINTS,
    SPRING_PRELOAD,
    TRAVEL,
    make_sprung_foot_spec_fn,
)

# `<default class="collision"><geom group="3"/>` in robot_walk.xml. Every
# collidable geom on the rigid robot sits in this group.
COLLISION_GROUP = 3


@pytest.fixture(scope="module")
def model():
    return make_sprung_foot_spec_fn(stiffness=1500.0)().compile()


def test_both_spring_joints_exist_and_are_passive(model):
    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(model.njnt)
    ]
    for j in SPRING_JOINTS:
        assert j in names
        # The passive_ prefix is load-bearing: every actuator/reward/obs regex
        # of the form ^(?!passive_).* relies on it to exclude these joints.
        assert j.startswith("passive_")


def test_spring_joints_are_slide_with_the_requested_range(model):
    for j in SPRING_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        assert model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_SLIDE
        assert model.jnt_range[jid][0] == pytest.approx(0.0)
        assert model.jnt_range[jid][1] == pytest.approx(TRAVEL)


def test_stiffness_and_damping_reach_the_compiled_model(model):
    # MjsJoint.stiffness/.damping need a 3-array; a scalar raises. This test
    # is what catches a regression to scalar assignment.
    for j in SPRING_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        assert model.jnt_stiffness[jid] == pytest.approx(1500.0)
        # Damping is DERIVED per arm as c = 2*zeta*sqrt(k*pad_mass), so the
        # pad's damping ratio stays constant as pad mass varies. An absolute
        # 0.5 N.s/m left zeta at 0.013-0.023 — the pad rang at 33-57 Hz against
        # a 50 Hz controller and never settled between steps.
        expected_c = damping_for(1500.0, PAD_MASS)
        assert model.dof_damping[model.jnt_dofadr[jid]] == pytest.approx(expected_c)


def test_stiffness_is_actually_parameterised():
    m800 = make_sprung_foot_spec_fn(stiffness=800.0)().compile()
    m3000 = make_sprung_foot_spec_fn(stiffness=3000.0)().compile()
    jid800 = mujoco.mj_name2id(m800, mujoco.mjtObj.mjOBJ_JOINT, SPRING_JOINTS[0])
    jid3000 = mujoco.mj_name2id(m3000, mujoco.mjtObj.mjOBJ_JOINT, SPRING_JOINTS[0])
    assert m800.jnt_stiffness[jid800] == pytest.approx(800.0)
    assert m3000.jnt_stiffness[jid3000] == pytest.approx(3000.0)


def test_positive_q_is_compression(model):
    """The sign convention the whole study rests on.

    q=0 must be the extended (unloaded) pose and q>0 must move the pad UP,
    toward the body. If this inverts, the spring pushes the robot into the
    ground and every sweep result is meaningless.
    """
    data = mujoco.MjData(model)
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, SPRING_JOINTS[0])
    pad = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_foot_pad")
    adr = model.jnt_qposadr[jid]

    data.qpos[adr] = 0.0
    mujoco.mj_forward(model, data)
    z_extended = data.xpos[pad][2]

    data.qpos[adr] = TRAVEL
    mujoco.mj_forward(model, data)
    z_compressed = data.xpos[pad][2]

    assert z_compressed > z_extended
    # Nearly all of the travel should show up as vertical motion; the small
    # shortfall is the ~5 deg ankle tilt at the home pose.
    assert (z_compressed - z_extended) == pytest.approx(TRAVEL, rel=0.05)


def test_pad_mass_is_added_not_idealised_away(model):
    from mjlab_microduck.robot.microduck_constants import get_walk_spec
    rigid_mass = get_walk_spec().compile().body_mass.sum()
    assert model.body_mass.sum() == pytest.approx(rigid_mass + 2 * PAD_MASS, abs=1e-6)


def test_locked_variant_has_zero_travel():
    """The locked (travel=0) variant must be a true rigid control, not a
    spring with an unenforced [0, 0] range (that would leave it unconstrained
    -- see fix-round-1 notes in the report).
    """
    m = make_sprung_foot_spec_fn(stiffness=1500.0, travel=0.0)().compile()
    for j in SPRING_JOINTS:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        assert jid == -1, f"{j} should not exist in the locked (travel=0) variant"


def test_locked_variant_has_two_fewer_dofs(model):
    locked = make_sprung_foot_spec_fn(stiffness=1500.0, travel=0.0)().compile()
    assert locked.nv == model.nv - 2


def test_contact_geom_and_site_live_on_the_pad(model):
    """The most load-bearing assertion in this file.

    feet_ground_contact matches ^(left_foot_collision|right_foot_collision)$ and
    foot_height_scan frames off the left_foot/right_foot sites. If either still
    resolves to the ankle body, contact is read from a geom floating above the
    ground and every gait metric silently reads garbage.
    """
    for side in ("left", "right"):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_foot_collision")
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_foot")
        assert gid >= 0 and sid >= 0
        pad = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_foot_pad")
        assert model.geom_bodyid[gid] == pad
        assert model.site_bodyid[sid] == pad


def test_old_sole_no_longer_collides(model):
    for side in ("left", "right"):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_sole_disabled")
        assert gid >= 0, f"the rigid {side} sole should be renamed, not deleted"
        assert model.geom_contype[gid] == 0
        assert model.geom_conaffinity[gid] == 0


def test_pad_geom_is_in_the_collision_group(model):
    """The pad geom must inherit `<default class="collision">` (group 3).

    `foot_height_scan` is configured with `include_geom_groups=(0,)` — terrain
    ONLY. A group-0 pad geom is therefore a valid ray target, so one foot's
    height ray terminates on the OPPOSITE foot's pad and reports it as ground,
    feeding a wrong height to `foot_clearance` (weight -2.0) and
    `foot_swing_height`. `exclude_parent_body=True` removes a foot's own pad but
    not the other one. This silently held while every other test passed.
    """
    for side in ("left", "right"):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_foot_collision")
        assert model.geom_group[gid] == COLLISION_GROUP
        # The class change must not cost the pad its contact with the floor.
        assert model.geom_contype[gid] != 0
        assert model.geom_conaffinity[gid] != 0


def test_no_group_zero_collidable_geoms_on_the_robot(model):
    """The stronger form of the test above: the rigid model has NO group-0
    collidable geom (every collision geom comes from the `collision` class), so
    the sprung model must not introduce one either.
    """
    offenders = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        for i in range(model.ngeom)
        if model.geom_group[i] == 0
        and (model.geom_contype[i] or model.geom_conaffinity[i])
    ]
    assert offenders == []


def test_spring_joint_does_not_inherit_xml_joint_defaults(model):
    """`<default class="microduck"><joint frictionloss="0.1" armature="0.005"/>`.

    The spring joint is added inside that childclass scope, so it silently
    inherited both: a dry-friction term worth roughly a third of total
    dissipation, and 5 g of effective inertia on a 20 g pad (+25%), invisible in
    the mass check. The spec's spring is IDEALISED — its only dissipation is
    `damping`.
    """
    for j in SPRING_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        dof = model.jnt_dofadr[jid]
        assert model.dof_frictionloss[dof] == pytest.approx(SPRING_FRICTIONLOSS)
        assert model.dof_armature[dof] == pytest.approx(SPRING_ARMATURE)
        assert model.dof_frictionloss[dof] == 0.0
        assert model.dof_armature[dof] == 0.0


def test_springref_encodes_the_preload(model):
    """The Sarrus mechanism's intentional 0.74 mm precompression.

    MuJoCo has no per-joint `jnt_springref` array on the compiled model (only
    `qpos_spring`, indexed by qpos address); for a 1-dof slide joint that is
    the same one number. Our sign convention is q=0 extended, q>0 compressed,
    so the precompression is encoded as a NEGATIVE springref.
    """
    for j in SPRING_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        adr = model.jnt_qposadr[jid]
        assert model.qpos_spring[adr] == pytest.approx(-SPRING_PRELOAD)


def test_preload_spring_force_opposes_compression_at_zero(model):
    """Sign-convention check: at q=0 the spring must push toward EXTENSION
    (negative q), i.e. it presses the pad against its own lower (extension)
    stop rather than sagging into the travel. That is what "preload" means
    physically: a spring that is already loaded at the unloaded (q=0) pose.
    """
    data = mujoco.MjData(model)
    for j in SPRING_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        dof = model.jnt_dofadr[jid]
        data.qpos[model.jnt_qposadr[jid]] = 0.0
    mujoco.mj_forward(model, data)
    for j in SPRING_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        dof = model.jnt_dofadr[jid]
        assert data.qfrc_spring[dof] < 0.0


def test_h_add_lowers_the_pad(model):
    """Larger h_add must put the contact pad further below the ankle."""
    shallow = make_sprung_foot_spec_fn(stiffness=1500.0, h_add=0.010)().compile()
    d_deep, d_shallow = mujoco.MjData(model), mujoco.MjData(shallow)
    mujoco.mj_forward(model, d_deep)
    mujoco.mj_forward(shallow, d_shallow)
    pad_deep = d_deep.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_foot_pad")]
    pad_shallow = d_shallow.xpos[mujoco.mj_name2id(shallow, mujoco.mjtObj.mjOBJ_BODY, "left_foot_pad")]
    assert pad_deep[2] < pad_shallow[2]


def test_damping_scales_with_pad_mass_to_hold_zeta_constant():
    """The pad's damping RATIO must not vary across the mass sweep.

    A fixed absolute damping made zeta vary 0.023 -> 0.013 across 30-90 g pads,
    so resonance changed with mass and confounded the very axis the sweep
    measures: sprung speed improved monotonically with pad mass (lighter pad =
    faster ringing = worse), the opposite of the locked arms' trend.
    """
    import math

    k = 3900.0
    for pad in (0.030, 0.090):
        m = make_sprung_foot_spec_fn(stiffness=k, pad_mass=pad)().compile()
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, SPRING_JOINTS[0])
        c = m.dof_damping[m.jnt_dofadr[jid]]
        zeta = c / (2.0 * math.sqrt(k * pad))
        assert zeta == pytest.approx(DAMPING_RATIO, rel=1e-6)


def test_explicit_damping_overrides_the_ratio():
    """An absolute value must win, so a measured hysteresis figure can replace
    the estimate without touching the ratio machinery."""
    m = make_sprung_foot_spec_fn(stiffness=3900.0, damping=7.5)().compile()
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, SPRING_JOINTS[0])
    assert m.dof_damping[m.jnt_dofadr[jid]] == pytest.approx(7.5)


def test_spring_joint_limit_is_stiffened(model):
    """The mechanical end-stop must be stiffer than MuJoCo's default.

    The default ([0.02, 1] at dt=0.002) let a 100 mm drop drive a 1500 N/m
    spring to 149% of its 12 mm travel — a stop that stores and returns energy
    instead of dissipating it overstates rebound, and bottoming is common in
    hop/jump regimes. 0.004 is the solver's stability floor (2*timestep).
    """
    for j in SPRING_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        assert model.jnt_solref[jid][0] == pytest.approx(0.004)
        assert model.jnt_solimp[jid][1] == pytest.approx(0.99)


# ── the contact footprint ────────────────────────────────────────────────────


def test_pad_footprint_matches_the_measured_prototype():
    """25 mm fore-aft x 40 mm lateral, measured on the Sarrus boot 2026-09-04.

    These were TRANSPOSED until 2026-09-04 -- 40 fore-aft x 28 lateral -- which
    gave the simulated robot 60% more pitch base of support than it owns and
    flattered every landing in the campaign. The transposition is invisible by
    inspection because both spellings are "a 40-and-a-small-number box", so it
    is pinned here per axis rather than as a set.
    """
    fore_aft, thickness, lateral = _PAD_HALF_EXTENTS
    assert fore_aft == pytest.approx(0.0125), "fore-aft half-extent: 25 mm sole"
    assert lateral == pytest.approx(0.0200), "lateral half-extent: 40 mm sole"
    assert thickness == pytest.approx(0.004)
    assert lateral > fore_aft, (
        "the boot is WIDER THAN LONG -- the linkage eats the fore-aft footprint. "
        "If this ever flips, check it is a real design change and not a "
        "transposition, and re-tune ANKLE_TO_SOLE against the settled delta."
    )


def test_pad_local_axes_map_to_fore_aft_and_lateral(model):
    """The half-extents are only meaningful given this mapping, so assert it.

    Local y is world-up at the home pose (that is why the spring slides along
    it), which leaves local x as fore-aft and local z as lateral. A change to
    the ankle frame would silently redefine which number is the pitch base.
    """
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_foot_pad")
    assert bid >= 0
    rot = data.xmat[bid].reshape(3, 3)
    # Columns are the pad's local axes expressed in world.
    assert abs(rot[0, 0]) == pytest.approx(1.0, abs=0.02), "local x is fore-aft"
    assert abs(rot[2, 1]) == pytest.approx(1.0, abs=0.02), "local y is vertical"
    assert abs(rot[1, 2]) == pytest.approx(1.0, abs=0.02), "local z is lateral"

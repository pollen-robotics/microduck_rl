import mujoco

from mjlab_microduck.robot.microduck_constants import (
    MICRODUCK_LAUNCHER_CFG,
    get_launcher_spec,
)


def test_launcher_spec_compiles_with_one_free_body():
    model = get_launcher_spec().compile()
    assert model.nbody == 2  # world + plate
    assert model.njnt == 1
    assert model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, 1)
    assert name == "plate"


def test_plate_is_heavy_enough_to_ignore_the_duck():
    # The plate is re-prescribed every control step; mass only has to keep the
    # intra-step deviation negligible against an ~800 g duck pushing off.
    model = get_launcher_spec().compile()
    assert model.body_mass[1] >= 20.0


def test_plate_geom_is_a_box_wide_enough_to_stand_on():
    model = get_launcher_spec().compile()
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "plate_geom")
    assert gid >= 0
    assert model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX
    half_x, half_y, half_z = model.geom_size[gid]
    assert half_x >= 0.08 and half_y >= 0.08   # >= 16 cm wide: both feet fit
    assert half_z <= 0.02


def test_launcher_entity_cfg_has_no_articulation():
    # A prop, not a robot: no actuators, no articulation info (cf. the ball).
    assert MICRODUCK_LAUNCHER_CFG.articulation is None


import torch

from mjlab_microduck.tasks import mdp as microduck_mdp


def _params(n=1, t_hold=0.5, t_launch=0.1, z0=0.15, vz=3.0, w0=10.0):
    f = lambda v: torch.full((n,), float(v))
    return dict(t_hold=f(t_hold), t_launch=f(t_launch), z0=f(z0), vz=f(vz), w0=f(w0))


def test_plate_is_parked_and_still_during_hold():
    z, pitch, vz_t, w_t, phase = microduck_mdp.backflip_plate_kinematics(
        torch.tensor([0.0, 0.25, 0.499]), **_params(3)
    )
    assert torch.allclose(z, torch.full((3,), 0.15))
    assert torch.allclose(pitch, torch.zeros(3))
    assert torch.allclose(vz_t, torch.zeros(3))
    assert torch.allclose(w_t, torch.zeros(3))
    assert torch.all(phase == microduck_mdp.BACKFLIP_PHASE_HOLD)


def test_launch_is_a_ramp_not_a_step():
    # THE point of the ramp: velocity rises linearly through the window, so the
    # contact solver never has to absorb a step change in plate velocity.
    t = torch.tensor([0.5, 0.55, 0.6])
    _, _, vz_t, w_t, phase = microduck_mdp.backflip_plate_kinematics(t, **_params(3))
    assert torch.allclose(vz_t, torch.tensor([0.0, 1.5, 3.0]), atol=1e-5)
    # w_t is NEGATIVE for a positive w0: w0 > 0 means "backward flick" (the
    # public contract), and a backward roll is a negative rotation about +y
    # in this codebase's convention (face-down = +90deg about +y is FORWARD;
    # see set_random_ground_state). This test previously pinned +5.0/+10.0,
    # which was the bug the Task 3 flip-envelope probe caught (an orientation
    # trace showed w0 > 0 was driving the robot face-down/forward, not
    # backward) — corrected here alongside the backflip_plate_kinematics fix.
    assert torch.allclose(w_t, torch.tensor([0.0, -5.0, -10.0]), atol=1e-5)
    assert torch.all(phase[:2] == microduck_mdp.BACKFLIP_PHASE_LAUNCH)


def test_launch_position_is_the_integral_of_the_ramp():
    # z(t_hold + t_launch) = z0 + 0.5 * vz * t_launch
    z, pitch, _, _, _ = microduck_mdp.backflip_plate_kinematics(
        torch.tensor([0.6]), **_params()
    )
    assert torch.allclose(z, torch.tensor([0.15 + 0.5 * 3.0 * 0.1]), atol=1e-5)
    # pitch is NEGATIVE for a positive w0 — see the sign-convention comment
    # in test_launch_is_a_ramp_not_a_step above. Previously pinned +0.5,
    # which was the same mislabeled-direction bug.
    assert torch.allclose(pitch, torch.tensor([-0.5 * 10.0 * 0.1]), atol=1e-5)


def test_plate_is_gone_after_the_ramp():
    z, _, vz_t, w_t, phase = microduck_mdp.backflip_plate_kinematics(
        torch.tensor([0.6001, 1.0]), **_params(2)
    )
    assert torch.all(phase == microduck_mdp.BACKFLIP_PHASE_GONE)
    # The parked height is ABOVE the floor, not below it: terrain_type="plane"
    # is an INFINITE half-space, so a negative parking z is 3 m of penetration
    # (4 spurious contacts per env per step, measured), not absence. This
    # assertion used to pin BACKFLIP_GONE_Z = -3.0.
    assert torch.allclose(z, torch.full((2,), microduck_mdp.BACKFLIP_GONE_POS[2]))
    assert microduck_mdp.BACKFLIP_GONE_POS[2] > 0.0
    assert torch.allclose(vz_t, torch.zeros(2))
    assert torch.allclose(w_t, torch.zeros(2))


def test_kinematics_are_batched_per_env():
    n = 4
    t = torch.tensor([0.0, 0.55, 0.7, 0.05])
    p = _params(n)
    p["t_hold"] = torch.tensor([0.5, 0.5, 0.5, 0.5])
    z, _, _, _, phase = microduck_mdp.backflip_plate_kinematics(t, **p)
    assert z.shape == (n,)
    assert phase.tolist() == [
        microduck_mdp.BACKFLIP_PHASE_HOLD,
        microduck_mdp.BACKFLIP_PHASE_LAUNCH,
        microduck_mdp.BACKFLIP_PHASE_GONE,
        microduck_mdp.BACKFLIP_PHASE_HOLD,
    ]

"""Crouchstand cfg invariants + the flat-foot balance check of the CROUCH keyframe.

CPU-only: builds the cfg (no env) and runs plain MuJoCo kinematics on the
allcollisions model the task uses.
"""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from mjlab_microduck.robot.microduck_constants import (
    MICRODUCK_ALLCOLLISIONS_XML,
    MICRODUCK_STANDUP_ROBOT_CFG,
)
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_crouchstand_env_cfg import (
    CROUCH_TARGET_OVERRIDES,
    CROUCH_Z,
    MicroduckCrouchStandRlCfg,
    make_microduck_crouchstand_env_cfg,
)
from mjlab_microduck.tasks.microduck_sitstand_env_cfg import (
    SITTING_TARGET_OVERRIDES,
    STAND_Z,
)

# qpos order of the 14 servos on the allcollisions model (ctrl idx = joint idx).
JOINTS = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
]
HOME = [0, -0.0873, -0.4579, -0.0049, 0.4530, 0.349, 0.349, 0, 0,
        0, 0.0873, 0.4579, 0.0049, -0.4530]


@pytest.fixture(scope="module")
def cfg():
    return make_microduck_crouchstand_env_cfg()


def test_uses_allcollisions_model_like_sitstand(cfg):
    assert cfg.scene.entities["robot"] is MICRODUCK_STANDUP_ROBOT_CFG


def test_low_posture_is_crouch_everywhere(cfg):
    for name in ("posture_pose_legs", "posture_pose_l1", "posture_composite"):
        assert cfg.rewards[name].params["sit_overrides"] is CROUCH_TARGET_OVERRIDES
    for name in ("posture_height", "posture_height_sharp", "posture_height_l1",
                 "posture_stillness", "posture_composite"):
        assert cfg.rewards[name].params["sit_z"] == CROUCH_Z
        assert cfg.rewards[name].params["stand_z"] == STAND_Z
    assert cfg.commands["twist"].sit_z == CROUCH_Z
    gs = cfg.events["set_ground_state"].params
    assert gs["sitting_joint_overrides"] is CROUCH_TARGET_OVERRIDES
    assert gs["sitting_z_min"] >= CROUCH_Z - 0.005


def test_sitstand_untouched():
    # building crouchstand must not mutate sitstand's shared module-level cfg
    from mjlab_microduck.tasks.microduck_sitstand_env_cfg import make_microduck_sitstand_env_cfg
    make_microduck_crouchstand_env_cfg()
    s = make_microduck_sitstand_env_cfg()
    assert s.rewards["posture_pose_legs"].params["sit_overrides"] is SITTING_TARGET_OVERRIDES
    assert "nonfoot_ground_contact" not in s.rewards


def test_upright_booster_fully_on_at_crouch(cfg):
    p = cfg.rewards["upright_while_tall"].params
    assert p["height_high"] < CROUCH_Z


def test_crouch_keyframe_mirrored_and_within_soft_limits():
    m = mujoco.MjModel.from_xml_path(str(MICRODUCK_ALLCOLLISIONS_XML))
    for idx, val in CROUCH_TARGET_OVERRIDES.items():
        lo, hi = m.joint(JOINTS[idx]).range
        mid, half = (lo + hi) / 2, (hi - lo) / 2
        assert mid - 0.9 * half < val < mid + 0.9 * half, JOINTS[idx]
        mirror = idx + 9 if idx < 5 else idx - 9
        assert CROUCH_TARGET_OVERRIDES[mirror] == pytest.approx(-val)
    # neck/head left to the head_pose command
    assert not set(CROUCH_TARGET_OVERRIDES) & {5, 6, 7, 8}


def test_nonfoot_penalty_sign_and_sensor(cfg):
    term = cfg.rewards["nonfoot_ground_contact"]
    assert term.weight > 0  # self-negating function → positive weight
    sensor = next(s for s in cfg.scene.sensors if s.name == term.params["sensor_name"])
    assert set(sensor.primary.exclude) == {"ankle_left", "ankle_right"}
    m = mujoco.MjModel.from_xml_path(str(MICRODUCK_ALLCOLLISIONS_XML))
    for b in sensor.primary.exclude:
        foot_geom = "left_foot_collision" if "left" in b else "right_foot_collision"
        assert m.body(m.geom_bodyid[m.geom(foot_geom).id]).name == b

    found = torch.tensor([[0, 0, 0], [0, 2, 0]])
    env = SimpleNamespace(scene=SimpleNamespace(sensors={
        "s": SimpleNamespace(data=SimpleNamespace(found=found))}))
    out = microduck_mdp.nonfoot_ground_contact_penalty(env, "s")
    assert out.tolist() == [0.0, -1.0]


def test_self_negating_penalties_have_positive_weights(cfg):
    for name in ("descent_speed", "gentle_motion", "posture_pose_l1",
                 "posture_height_l1", "nonfoot_ground_contact"):
        assert cfg.rewards[name].weight > 0, name


def test_runner_has_own_experiment():
    assert MicroduckCrouchStandRlCfg.experiment_name == "microduck_crouchstand"


def _flat_foot_balance(vals):
    """Level the soles as at HOME, put them on z=0; return (CoM margins mm, trunk z, pitch deg)."""
    from scipy.spatial.transform import Rotation as Rot
    m = mujoco.MjModel.from_xml_path(str(MICRODUCK_ALLCOLLISIONS_XML))
    d = mujoco.MjData(m)
    jq = [m.joint(n).qposadr[0] for n in JOINTS]
    feet = [m.geom("left_foot_collision").id, m.geom("right_foot_collision").id]
    trunk = m.body("trunk_base").id

    def setq(v, quat=(1, 0, 0, 0), z=0.3):
        d.qpos[:] = 0
        d.qpos[2], d.qpos[3:7] = z, quat
        d.qpos[jq] = v
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)

    def soles():
        pts = []
        for g in feet:
            a, n = m.mesh_vertadr[m.geom_dataid[g]], m.mesh_vertnum[m.geom_dataid[g]]
            pts.append(m.mesh_vert[a:a + n] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])
        return np.vstack(pts)

    setq(HOME)
    r_home = d.geom_xmat[feet[0]].reshape(3, 3).copy()
    setq(vals)
    q = Rot.from_matrix(r_home @ d.geom_xmat[feet[0]].reshape(3, 3).T).as_quat()
    quat = (q[3], q[0], q[1], q[2])
    setq(vals, quat)
    setq(vals, quat, z=0.3 - soles()[:, 2].min())
    pts = soles()
    patch = pts[pts[:, 2] < pts[:, 2].min() + 0.002][:, 0]
    com_x = d.subtree_com[trunk][0]
    rmat = d.xmat[trunk].reshape(3, 3)
    return ((patch.max() - com_x) * 1e3, (com_x - patch.min()) * 1e3,
            d.xpos[trunk][2], np.degrees(np.arctan2(rmat[0, 2], rmat[2, 2])))


def test_crouch_keyframe_balances_on_flat_feet():
    vals = list(HOME)
    for idx, v in CROUCH_TARGET_OVERRIDES.items():
        vals[idx] = v
    front, back, z, pitch = _flat_foot_balance(vals)
    assert min(front, back) > 15.0, (front, back)   # CoM well inside the 49 mm sole
    assert abs(z - CROUCH_Z) < 0.008, z               # target height matches geometry
    assert 0.0 <= pitch <= 12.0, pitch                # slight forward lean, roller-like

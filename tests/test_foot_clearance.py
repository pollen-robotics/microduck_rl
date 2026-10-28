"""Foot-clearance FK, label projection and the velstand wiring (branch standup_foot_catch).

CPU only.
"""
import mujoco
import numpy as np
import torch

import mjlab_microduck.tasks  # noqa: F401
from mjlab_microduck.robot.microduck_constants import MICRODUCK_ALLCOLLISIONS_XML
from mjlab_microduck.tasks import distill
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks import microduck_velstand_env_cfg as vs
from mjlab_microduck.tasks.foot_clearance import (
    SERVO_NAMES,
    _PARAMS,
    foot_clearance,
    home_servo_pos,
    lateral_gap,
    path_clearance,
    project_min_clearance,
)

HOME = torch.tensor(home_servo_pos())
L_ROLL, R_ROLL = 1, 10


def _cpu_box_clearance(q: np.ndarray) -> float:
    """Reference: same box proxy, placed by MuJoCo's own kinematics."""
    m = mujoco.MjModel.from_xml_path(str(MICRODUCK_ALLCOLLISIONS_XML))
    d = mujoco.MjData(m)
    for n, v in zip(SERVO_NAMES, q):
        d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]] = v
    mujoco.mj_kinematics(m, d)
    pose = {}
    for side, body in (("left", "ankle_left"), ("right", "ankle_right")):
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
        pose[side] = (d.xmat[b].reshape(3, 3), d.xpos[b])
    out = []
    for a, b in (("left", "right"), ("right", "left")):
        (Ra, pa), (Rb, pb) = pose[a], pose[b]
        lb = (_PARAMS[a]["points"] @ Ra.T + pa - pb) @ Rb
        qd = np.abs(lb - _PARAMS[b]["center"]) - _PARAMS[b]["half"]
        out.append((np.linalg.norm(np.maximum(qd, 0), axis=1) + np.minimum(qd.max(1), 0)).min())
    return float(min(out))


def test_home_feet_are_clear_and_crossed_feet_overlap():
    q = HOME.repeat(2, 1)
    q[1, L_ROLL], q[1, R_ROLL] = -0.6, 0.6  # legs swung across each other
    c = foot_clearance(q)
    assert 0.03 < c[0] < 0.06  # HOME: ~42 mm mesh, box proxy a little less
    assert c[1] < 0.0


def test_fk_matches_mujoco_kinematics():
    g = torch.Generator().manual_seed(0)
    q = HOME + (torch.rand(6, 14, generator=g) * 2 - 1) * 0.5
    fk = foot_clearance(q).numpy()
    ref = np.array([_cpu_box_clearance(r) for r in q.numpy()])
    np.testing.assert_allclose(fk, ref, atol=1e-4)


def test_projection_restores_clearance_and_leaves_clear_rows_alone():
    q = HOME.repeat(3, 1)
    q[1, L_ROLL], q[1, R_ROLL] = -0.3, 0.3
    q[2, L_ROLL], q[2, R_ROLL] = -0.45, 0.45
    out = project_min_clearance(q, 0.015, (0, 1, 9, 10))
    assert torch.equal(out[0], q[0])
    assert (foot_clearance(out[1:]) > 0.013).all()
    non_proj = [i for i in range(14) if i not in (0, 1, 9, 10)]
    assert torch.equal(out[:, non_proj], q[:, non_proj])


def test_projection_never_pushes_a_leg_through_the_other():
    q = HOME.repeat(4, 1)
    for i, r in enumerate((0.2, 0.3, 0.45, 0.6)):
        q[i, L_ROLL], q[i, R_ROLL] = -r, r
    out = project_min_clearance(q, 0.015, (0, 1, 9, 10))
    assert (lateral_gap(out) >= lateral_gap(q) - 1e-6).all()


def test_side_mask_from_obs():
    obs = torch.zeros(3, 61)
    obs[0, 3:6] = torch.tensor([0.0, 0.0, -1.0])  # upright
    obs[1, 3:6] = torch.tensor([0.0, 1.0, 0.0])   # on a side
    obs[2, 3:6] = torch.tensor([1.0, 0.0, 0.0])   # face down / up
    assert distill.side_mask_from_obs(obs, (3, 6), 30.0).tolist() == [False, True, False]


def test_label_projection_clears_the_swing_path():
    cfg = {"label_min_foot_clearance": 0.015, "label_clearance_joints": (0, 1, 2, 3, 4, 9, 10, 11, 12, 13),
           "home_joint_pos": home_servo_pos()}
    a = torch.zeros(2, 14)
    a[1, L_ROLL] = -1.0 - HOME[L_ROLL]  # left hip_roll → −1.0: the leg thrown across the body, as alpha_stand's roll does
    cur = torch.zeros(2, 14)  # at HOME
    with torch.inference_mode():  # rollout storage tensors + no_grad label pass, as in training
        a, cur = a.clone(), cur.clone()
    with torch.no_grad():
        out = distill.project_label_clearance(a, cur, cfg)
    assert torch.equal(out[0], a[0])  # a clear path is untouched
    assert path_clearance(HOME + cur[1:], HOME + a[1:]) < 0.0
    assert path_clearance(HOME + cur[1:], HOME + out[1:]) > 0.012


def test_still_side_mask():
    obs = torch.zeros(3, 61)
    obs[:, 3:6] = torch.tensor([0.0, 1.0, 0.0])
    obs[1, 0:3] = torch.tensor([1.0, 0.0, 0.0])   # rolling
    obs[2, 20:34] = 1.0                           # legs swinging
    assert distill.still_mask_from_obs(obs, 0.3, 0.4).tolist() == [True, False, False]


def test_velstand_wiring():
    assert vs.ENABLE_FOOT_CLEARANCE and vs.ENABLE_SIDE_UNMASKED_TEACHER
    cfg = vs.make_microduck_velstand_env_cfg()
    term = cfg.rewards["foot_clearance"]
    assert term.func is microduck_mdp.foot_clearance_cost and term.weight < 0  # cost >= 0 → negative weight
    bc = vs.MicroduckVelStandRlCfg.algorithm.bc_cfg
    assert bc["label_min_foot_clearance"] == vs.FOOT_CLEARANCE_LABEL_MIN
    assert bc["expert_mask_keep_side_cone_deg"] == vs.TEACHER_MASK_KEEP_SIDE_CONE_DEG
    assert tuple(bc["home_joint_pos"]) == home_servo_pos()


def test_label_projection_leaves_clear_rows_bit_exact():
    cfg = {"label_min_foot_clearance": 0.015, "label_clearance_joints": (0, 1, 2, 3, 4, 9, 10, 11, 12, 13),
           "home_joint_pos": home_servo_pos()}
    a = (torch.rand(64, 14, generator=torch.Generator().manual_seed(1)) * 2 - 1) * 0.05  # near HOME: feet clear
    with torch.no_grad():
        out = distill.project_label_clearance(a, torch.zeros_like(a), cfg)
    assert torch.equal(out, a)  # the expert_bc_label_projected_frac stat counts exact changes


def test_velstand_stuck_spawns_wired():
    import os

    import numpy as np

    assert vs.ENABLE_STUCK_SPAWNS and 0.0 < vs.STUCK_SPAWN_PROB <= 0.5
    assert os.path.isfile(vs.STUCK_LIBRARY)  # shipped in the package: HF Jobs read it from the source tree
    d = np.load(vs.STUCK_LIBRARY)
    n = len(d["quat"])
    assert n >= 100 and d["quat"].shape == (n, 4) and d["z"].shape == (n,) and d["servo"].shape == (n, 14)
    np.testing.assert_allclose(np.linalg.norm(d["quat"], axis=1), 1.0, atol=1e-4)
    p = vs.make_microduck_velstand_env_cfg().events["random_prone_init"].params
    assert p["stuck_prob"] == vs.STUCK_SPAWN_PROB and p["stuck_params"]["library_path"] == vs.STUCK_LIBRARY


def test_still_gate_ignores_joint_jitter():
    """PPO action noise shakes the joints of a robot lying on its side; the still gate must keep it."""
    bc = vs.MicroduckVelStandRlCfg.algorithm.bc_cfg
    obs = torch.zeros(2, 61)
    obs[:, 3:6] = torch.tensor([0.0, 1.0, 0.0])
    obs[0, 20:34] = 2.0                           # joints jittering under exploration noise
    obs[1, 0:3] = torch.tensor([3.0, 0.0, 0.0])   # rolling through the side
    assert distill.still_mask_from_obs(obs, bc["expert_mask_still_gyro"], bc["expert_mask_still_qvel"]).tolist() == [True, False]


def test_velstand_caps_exploration_std():
    bc = vs.MicroduckVelStandRlCfg.algorithm.bc_cfg
    assert bc["max_action_std"] == vs.MAX_ACTION_STD and 0.0 < vs.MAX_ACTION_STD < 0.35

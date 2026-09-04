"""infer_policy.py drives the CPU MuJoCo rehearsal with the SAME BAM M6 actuator
the policies are trained against in warp (bam.mujoco.MujocoController on a
motor-converted model). These tests lock the two halves together:

* the script's hardcoded BAM constants mirror ``_BAM_ACTUATOR_KWARGS`` in
  microduck_constants (not imported there to keep the script torch/warp-free);
* the motor conversion matches what ``bam.mjlab.BamActuator.edit_spec`` does
  (torque motors, voltage-bounded forcerange, armature, zeroed XML friction,
  stiff friction constraint) and a step loop runs with a live friction budget.
"""

import importlib.util
from pathlib import Path

import mujoco
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def ip():
    spec = importlib.util.spec_from_file_location(
        "infer_policy", REPO / "scripts" / "infer_policy.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cpu_bam_constants_mirror_training_cfg(ip):
    from bam.mjlab import BamActuator
    from mjlab_microduck.robot.microduck_constants import _BAM_ACTUATOR_KWARGS as k

    assert ip.BAM_MOTOR_NAME == k["motor_name"]
    assert ip.BAM_MODEL == k["model"]
    assert ip.BAM_KP_FW == k["kp_fw"]
    assert ip.BAM_VIN_RANGE == k["vin_range"]
    assert ip.BAM_VIN_DROP_GAIN_RANGE == k["vin_drop_gain_range"]
    assert ip.BAM_VIN_MIN == k["vin_min"]
    assert ip.BAM_MAX_CURRENT == k.get("max_current")
    assert ip.BAM_STIFF_SOLREF_FRICTION == BamActuator._STIFF_SOLREF_FRICTION
    assert ip.BAM_STIFF_SOLIMP_FRICTION == BamActuator._STIFF_SOLIMP_FRICTION


@pytest.fixture(scope="module")
def bam_sim(ip):
    bam_model = ip.load_bam_model(ip.BAM_KP_FW, 7.4, ip.BAM_MAX_CURRENT)
    model, data, ctrl, names = ip.load_mujoco_with_bam(
        str(REPO / ip.MICRODUCK_XML), bam_model, 0.005, 0.1, ip.BAM_VIN_MIN
    )
    return ip, bam_model, model, data, ctrl, names


def test_actuators_converted_like_warp(bam_sim):
    ip, bam_model, model, data, ctrl, names = bam_sim
    kt, R = bam_model.kt.value, bam_model.R.value
    assert len(names) == 14 and model.nu == 14
    assert not any(n.startswith("passive_") for n in names)
    # Torque motors: ctrl is the BAM torque, no MuJoCo PD left over.
    assert (model.actuator_gaintype == mujoco.mjtGain.mjGAIN_FIXED).all()
    assert (model.actuator_biastype == mujoco.mjtBias.mjBIAS_NONE).all()
    assert np.allclose(model.actuator_gainprm[:, 0], 1.0)
    # (set_to_motor leaves the old PD biasprm bytes behind; inert under BIAS_NONE,
    # exactly as in warp's edit_spec.)
    assert (model.actuator_forcelimited == 1).all()
    assert np.allclose(model.actuator_forcerange[:, 1], 7.4 * kt / R)
    dofs = model.jnt_dofadr[model.actuator_trnid[:, 0]]
    assert np.allclose(model.dof_armature[dofs], bam_model.actuator.get_extra_inertia())
    assert np.allclose(model.dof_solref[dofs], ip.BAM_STIFF_SOLREF_FRICTION)
    assert np.allclose(model.dof_solimp[dofs], ip.BAM_STIFF_SOLIMP_FRICTION)
    assert bam_model.actuator.kp == ip.BAM_KP_FW
    assert bam_model.actuator.max_current is None  # training has no current limiter


def test_bam_step_loop_runs_with_live_friction(bam_sim):
    ip, bam_model, model, data, ctrl, names = bam_sim
    fj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    qa = model.jnt_qposadr[fj]
    mujoco.mj_resetData(model, data)
    data.qpos[qa + 2] = 0.125
    data.qpos[qa + 3 : qa + 7] = [1, 0, 0, 0]
    jq = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    data.qpos[jq] = ip.DEFAULT_POSE
    ctrl.reset(data.qpos)
    ctrl.q_target[:] = ip.DEFAULT_POSE
    mujoco.mj_forward(model, data)
    dofs = model.jnt_dofadr[model.actuator_trnid[:, 0]]
    for _ in range(100):
        ctrl.update()
        mujoco.mj_step(model, data)
    assert not np.isnan(data.qpos).any()
    limit = model.actuator_forcerange[0, 1]
    assert (np.abs(data.ctrl) <= limit + 1e-9).all()  # ctrl IS the motor torque
    assert (model.dof_frictionloss[dofs] > 0).all()  # BAM budget written every step
    assert np.allclose(model.dof_damping[dofs], bam_model.friction_viscous.value)

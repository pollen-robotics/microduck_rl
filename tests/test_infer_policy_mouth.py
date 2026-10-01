"""CPU replay keeps a 14-joint policy separate from an optional mouth target.

The extra hinge below is a routing fixture, not a model of Microduck's beak.
It deliberately has a different actuator name, interleaved passive coordinates,
and shuffled actuator order. The ONNX graph is a deterministic test policy,
not a trained behavior or evidence that the robot can grasp anything.
"""

import copy
import importlib.util
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import pytest


REPO = Path(__file__).resolve().parents[1]
JOINTS = (
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
)
MOUTH_ACTUATOR = "synthetic_mouth_motor"
MOUTH_RANGE = (-0.1, 0.6)


@pytest.fixture(scope="module")
def ip():
    spec = importlib.util.spec_from_file_location(
        "infer_policy_mouth_test", REPO / "scripts" / "infer_policy.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mouth_xml(ip, tmp_path_factory):
    spec = mujoco.MjSpec.from_file(str(REPO / ip.MICRODUCK_XML))
    # Existing keyframes have 14 controls and cannot describe this test model.
    for key in list(spec.keys):
        spec.delete(key)
    spec.meshdir = str((REPO / ip.MICRODUCK_XML).parent / "assets")
    spec.body("yaw2roll").add_joint(
        name="passive_test_hinge",
        type=mujoco.mjtJoint.mjJNT_HINGE,
        axis=[1, 0, 0],
        damping=0.1,
        armature=0.001,
    )
    # An isolated toy hinge avoids inventing a jaw linkage or altering head mass.
    body = spec.worldbody.add_body(name="synthetic_mouth_fixture", pos=[0, 0, 2])
    body.add_joint(
        name="mouth",
        type=mujoco.mjtJoint.mjJNT_HINGE,
        axis=[0, 1, 0],
        limited=True,
        range=MOUTH_RANGE,
        damping=0.01,
        armature=0.001,
    )
    body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.01, 0, 0],
        mass=0.01,
        contype=0,
        conaffinity=0,
    )
    actuator = spec.add_actuator(
        name=MOUTH_ACTUATOR,
        target="mouth",
        trntype=mujoco.mjtTrn.mjTRN_JOINT,
        ctrllimited=True,
        ctrlrange=MOUTH_RANGE,
    )
    actuator.set_to_position(kp=0.55, kv=0.0)
    spec.compile()
    root = ET.fromstring(spec.to_xml())
    actuators = root.find("actuator")
    by_name = {act.get("name"): act for act in actuators}
    # Put mouth in the middle, shift the right leg, and break policy/XML order.
    order = list(reversed(JOINTS[:7])) + [MOUTH_ACTUATOR] + list(reversed(JOINTS[7:]))
    actuators[:] = [by_name[name] for name in order]
    path = tmp_path_factory.mktemp("mouth-model") / "synthetic.xml"
    ET.ElementTree(root).write(path, encoding="unicode")
    return path


@pytest.fixture(scope="module")
def policy_files(tmp_path_factory):
    directory = tmp_path_factory.mktemp("mouth-policy")
    policies = {}
    for width in (51, 61):
        weights = np.zeros((width, 14), dtype=np.float32)
        # Each action reads its own joint observation plus its own sentinel bias.
        weights[6 + np.arange(14), np.arange(14)] = 0.25
        bias = np.linspace(-0.065, 0.065, 14, dtype=np.float32)
        graph = helper.make_graph(
            [
                helper.make_node("MatMul", ["obs", "W"], ["weighted"]),
                helper.make_node("Add", ["weighted", "B"], ["actions"]),
            ],
            "routing_fixture",
            [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, width])],
            [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, 14])],
            initializer=[
                numpy_helper.from_array(weights, "W"),
                numpy_helper.from_array(bias, "B"),
            ],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 8
        onnx.checker.check_model(model)
        path = directory / f"policy-{width}.onnx"
        onnx.save(model, path)
        policies[width] = (path, weights, bias)
    return policies


def _load(ip, xml, bam):
    if bam:
        bam_model = ip.load_bam_model(ip.BAM_KP_FW, 7.4, ip.BAM_MAX_CURRENT)
        model, data, ctrl, _ = ip.load_mujoco_with_bam(
            str(xml), bam_model, 0.005, 0.1, ip.BAM_VIN_MIN
        )
        return model, data, ctrl
    model = mujoco.MjModel.from_xml_path(str(xml))
    model.opt.timestep = 0.005
    return model, mujoco.MjData(model), None


def _make_policy(ip, xml, policy_files, bam, width=61, delay=0):
    model, data, ctrl = _load(ip, xml, bam)
    policy = ip.PolicyInference(
        model,
        data,
        walking_onnx_path=str(policy_files[width][0]),
        bam_ctrl=ctrl,
        action_scale=0.4,
        new_cmd_obs=width == 61,
        delay_min_lag=delay,
        delay_max_lag=delay,
    )
    # Resolve expected coordinates independently of the policy's mapping.
    joint_ids = [model.joint(name).id for name in JOINTS]
    data.qpos[model.jnt_qposadr[joint_ids]] = ip.DEFAULT_POSE
    if ctrl is not None:
        ctrl.reset(data.qpos)
    mujoco.mj_forward(model, data)
    return policy, model, data, ctrl


def _targets(model, data, ctrl, names):
    if ctrl is not None:
        return np.array([ctrl.get_q_target(name) for name in names])
    return np.array([data.ctrl[model.actuator(name).id] for name in names])


@pytest.mark.parametrize("bam", [False, True], ids=["xml-pd", "bam"])
@pytest.mark.parametrize("width", [51, 61])
def test_released_model_preserves_policy_observations_and_targets(
    ip, policy_files, bam, width
):
    policy, model, data, ctrl = _make_policy(
        ip, REPO / ip.MICRODUCK_XML, policy_files, bam, width
    )
    assert policy.n_joints == 14
    assert model.nu == 14
    policy.set_vel_cmd(0.1, -0.02, 0.3)
    policy.head_offset[:] = [0.01, -0.02, 0.03, -0.04]
    policy._update_command()
    obs = policy.get_observations()
    assert obs.shape == (width,)
    np.testing.assert_array_equal(obs[6:20], np.zeros(14, np.float32))
    np.testing.assert_array_equal(obs[20:48], np.zeros(28, np.float32))
    np.testing.assert_array_equal(obs[48:51], policy.vel_cmd)
    if width == 61:
        np.testing.assert_array_equal(obs[51:55], policy.head_offset)
        np.testing.assert_array_equal(obs[55:61], np.zeros(6, np.float32))
    _, weights, bias = policy_files[width]
    action = policy.infer()
    np.testing.assert_allclose(action, obs @ weights + bias, rtol=0, atol=1e-7)
    policy.apply_action(action)
    expected = ip.DEFAULT_POSE + action * 0.4
    if width == 51:
        expected[5:9] += policy.head_offset
    np.testing.assert_array_equal(_targets(model, data, ctrl, JOINTS), expected)
    before = _targets(model, data, ctrl, JOINTS)
    with pytest.raises(ValueError, match="mouth"):
        policy.set_mouth_target(0.2)
    np.testing.assert_array_equal(_targets(model, data, ctrl, JOINTS), before)


@pytest.mark.parametrize("bam", [False, True], ids=["xml-pd", "bam"])
@pytest.mark.parametrize("width", [51, 61])
def test_optional_mouth_does_not_change_policy_layout_or_targets(
    ip, mouth_xml, policy_files, bam, width
):
    policy, model, data, ctrl = _make_policy(ip, mouth_xml, policy_files, bam, width)
    assert model.nu == 15
    assert policy.n_joints == 14
    expected_ids = [model.actuator(name).id for name in JOINTS]
    np.testing.assert_array_equal(policy.policy_actuator_ids, expected_ids)
    joint_ids = [model.joint(name).id for name in JOINTS]
    np.testing.assert_array_equal(
        policy.joint_qpos_indices, model.jnt_qposadr[joint_ids]
    )
    np.testing.assert_array_equal(
        policy.joint_qvel_indices, model.jnt_dofadr[joint_ids]
    )
    assert np.diff(model.jnt_qposadr[joint_ids]).max() > 1  # passive joint interleaves
    delta = np.linspace(-0.014, 0.014, 14, dtype=np.float32)
    velocity = np.linspace(-0.14, 0.14, 14, dtype=np.float32)
    data.qpos[model.jnt_qposadr[joint_ids]] = ip.DEFAULT_POSE + delta
    data.qvel[model.jnt_dofadr[joint_ids]] = velocity
    data.qpos[model.joint("passive_test_hinge").qposadr] = 0.37
    data.qvel[model.joint("passive_test_hinge").dofadr] = 0.81
    data.qpos[model.joint("mouth").qposadr] = 0.11
    data.qvel[model.joint("mouth").dofadr] = 0.63
    mujoco.mj_forward(model, data)
    obs = policy.get_observations()
    assert obs.shape == (width,)
    np.testing.assert_allclose(obs[6:20], delta, rtol=0, atol=3e-8)
    np.testing.assert_array_equal(obs[20:34], velocity)
    _, weights, bias = policy_files[width]
    policy.set_mouth_target(0.2)
    for _ in range(3):
        obs = policy.get_observations()
        action = policy.infer()
        np.testing.assert_allclose(action, obs @ weights + bias, rtol=0, atol=1e-7)
        np.testing.assert_array_equal(policy.last_action, action)
        policy.apply_action(action)
        np.testing.assert_array_equal(
            _targets(model, data, ctrl, JOINTS), ip.DEFAULT_POSE + action * 0.4
        )
        assert _targets(model, data, ctrl, [MOUTH_ACTUATOR])[0] == 0.2
        for _ in range(4):
            if ctrl is not None:
                ctrl.update()
            mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
    before = _targets(model, data, ctrl, JOINTS)
    policy.set_mouth_target(0.4)
    np.testing.assert_array_equal(_targets(model, data, ctrl, JOINTS), before)
    assert _targets(model, data, ctrl, [MOUTH_ACTUATOR])[0] == 0.4
    # Direct posture/initialization writes must also leave the mouth alone.
    policy.set_position_targets(ip.DEFAULT_POSE)
    assert _targets(model, data, ctrl, [MOUTH_ACTUATOR])[0] == 0.4
    np.testing.assert_array_equal(_targets(model, data, ctrl, JOINTS), ip.DEFAULT_POSE)


@pytest.mark.parametrize("bam", [False, True], ids=["xml-pd", "bam"])
def test_invalid_commands_leave_targets_and_delay_unchanged(
    ip, mouth_xml, policy_files, bam
):
    policy, model, data, ctrl = _make_policy(ip, mouth_xml, policy_files, bam, delay=1)
    policy.set_mouth_target(0.2)
    policy.set_position_targets(ip.DEFAULT_POSE)
    names = JOINTS + (MOUTH_ACTUATOR,)
    targets = _targets(model, data, ctrl, names)
    history = policy.last_action.copy()
    buffer = np.array(policy.action_buffer)
    index = policy.buffer_index
    for target in (np.nan, np.inf, -np.inf, -0.1001, 0.6001):
        with pytest.raises(ValueError):
            policy.set_mouth_target(target)
        np.testing.assert_array_equal(_targets(model, data, ctrl, names), targets)
    bad_vectors = [np.zeros(13), np.zeros(15), np.zeros((1, 14)), np.array(0.1)]
    for value in (np.nan, np.inf, -np.inf):
        vector = np.zeros(14)
        vector[9] = value
        bad_vectors.append(vector)
    for method in (policy.apply_action, policy.set_position_targets):
        for vector in bad_vectors:
            with pytest.raises(ValueError):
                method(vector)
            np.testing.assert_array_equal(_targets(model, data, ctrl, names), targets)
            np.testing.assert_array_equal(policy.action_buffer, buffer)
            np.testing.assert_array_equal(policy.last_action, history)
            assert policy.buffer_index == index


@pytest.mark.parametrize("bam", [False, True], ids=["xml-pd", "bam"])
def test_policy_delay_does_not_delay_or_overwrite_mouth(
    ip, mouth_xml, policy_files, bam
):
    policy, model, data, ctrl = _make_policy(ip, mouth_xml, policy_files, bam, delay=1)
    first = np.linspace(-0.1, 0.1, 14, dtype=np.float32)
    second = -first
    policy.set_mouth_target(0.2)
    policy.apply_action(first)
    np.testing.assert_array_equal(_targets(model, data, ctrl, JOINTS), ip.DEFAULT_POSE)
    policy.set_mouth_target(0.4)
    policy.apply_action(second)
    np.testing.assert_array_equal(
        _targets(model, data, ctrl, JOINTS), ip.DEFAULT_POSE + first * 0.4
    )
    assert _targets(model, data, ctrl, [MOUTH_ACTUATOR])[0] == 0.4
    assert all(entry.shape == (14,) for entry in policy.action_buffer)


@pytest.mark.parametrize("bam", [False, True], ids=["xml-pd", "bam"])
def test_mouth_target_reaches_physics(ip, mouth_xml, policy_files, bam):
    positions = []
    for mouth_target in (0.0, 0.2):
        policy, model, data, ctrl = _make_policy(ip, mouth_xml, policy_files, bam)
        mouth_qpos = model.joint("mouth").qposadr[0]
        assert data.qpos[mouth_qpos] == 0.0
        assert data.qvel[model.joint("mouth").dofadr[0]] == 0.0
        policy.set_position_targets(ip.DEFAULT_POSE)
        policy.set_mouth_target(mouth_target)
        # One simulated second per command lets this isolated damped hinge
        # respond without requiring precise tracking from either actuator model.
        for _ in range(200):
            if ctrl is not None:
                ctrl.update()
            mujoco.mj_step(model, data)
        positions.append(data.qpos[mouth_qpos])
        np.testing.assert_array_equal(
            _targets(model, data, ctrl, JOINTS), ip.DEFAULT_POSE
        )
        if mouth_target > 0:
            policy.set_mouth_target(0.0)
            for _ in range(200):
                if ctrl is not None:
                    ctrl.update()
                mujoco.mj_step(model, data)
            assert abs(data.qpos[mouth_qpos]) < abs(positions[-1])
            np.testing.assert_array_equal(
                _targets(model, data, ctrl, JOINTS), ip.DEFAULT_POSE
            )
    # The world-mounted sphere has no gravity moment about its centered hinge.
    # Its motion must come from the commanded actuator, not robot motion/inertia.
    assert positions[0] == 0.0
    assert positions[1] > positions[0]


def _filtered_position_xml(ip, source, actuator_name, tmp_path):
    spec = mujoco.MjSpec.from_file(str(source))
    spec.meshdir = str((REPO / ip.MICRODUCK_XML).parent / "assets")
    actuator = spec.actuator(actuator_name)
    actuator.set_to_position(
        kp=actuator.gainprm[0], kv=-actuator.biasprm[2], timeconst=0.02
    )
    xml = tmp_path / "filtered-position.xml"
    xml.write_text(spec.to_xml())
    return xml


def test_filtered_policy_position_actuator_preserves_legacy_replay(
    ip, policy_files, tmp_path
):
    xml = _filtered_position_xml(ip, REPO / ip.MICRODUCK_XML, "left_hip_yaw", tmp_path)
    policy, model, data, _ = _make_policy(ip, xml, policy_files, bam=False)
    assert model.nu == 14
    actuator_id = model.actuator("left_hip_yaw").id
    assert model.actuator_dyntype[actuator_id] == mujoco.mjtDyn.mjDYN_FILTEREXACT
    activation_id = model.actuator_actadr[actuator_id]
    assert data.act[activation_id] == 0.0
    obs = policy.get_observations()
    _, weights, bias = policy_files[61]
    action = policy.infer()
    np.testing.assert_allclose(action, obs @ weights + bias, rtol=0, atol=1e-7)
    # Exercise only the pre-existing 14-joint API: no optional mouth methods.
    policy.apply_action(action)
    targets = ip.DEFAULT_POSE + action * 0.4
    np.testing.assert_array_equal(_targets(model, data, None, JOINTS), targets)
    mujoco.mj_step(model, data)
    np.testing.assert_allclose(
        data.act[activation_id],
        targets[0] * (1 - np.exp(-model.opt.timestep / 0.02)),
        rtol=0,
        atol=1e-12,
    )
    for _ in range(11):
        mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
    assert data.actuator_force[actuator_id] != 0.0
    np.testing.assert_array_equal(_targets(model, data, None, JOINTS), targets)


def test_filtered_mouth_position_actuator_reaches_physics(
    ip, mouth_xml, policy_files, tmp_path
):
    xml = _filtered_position_xml(ip, mouth_xml, MOUTH_ACTUATOR, tmp_path)
    policy, model, data, _ = _make_policy(ip, xml, policy_files, bam=False)
    actuator_id = model.actuator(MOUTH_ACTUATOR).id
    assert model.actuator_dyntype[actuator_id] == mujoco.mjtDyn.mjDYN_FILTEREXACT
    activation_id = model.actuator_actadr[actuator_id]
    mouth_qpos = model.joint("mouth").qposadr[0]
    assert data.act[activation_id] == data.qpos[mouth_qpos] == 0.0
    policy.set_mouth_target(0.2)
    obs = policy.get_observations()
    _, weights, bias = policy_files[61]
    action = policy.infer()
    np.testing.assert_allclose(action, obs @ weights + bias, rtol=0, atol=1e-7)
    policy.apply_action(action)
    targets = ip.DEFAULT_POSE + action * 0.4
    np.testing.assert_array_equal(_targets(model, data, None, JOINTS), targets)
    assert data.ctrl[actuator_id] == 0.2
    mujoco.mj_step(model, data)
    np.testing.assert_allclose(
        data.act[activation_id],
        0.2 * (1 - np.exp(-model.opt.timestep / 0.02)),
        rtol=0,
        atol=1e-12,
    )
    for _ in range(199):
        mujoco.mj_step(model, data)
    opened = data.qpos[mouth_qpos]
    assert opened > 0.0
    policy.set_mouth_target(0.0)
    for _ in range(200):
        mujoco.mj_step(model, data)
    assert abs(data.qpos[mouth_qpos]) < abs(opened)
    assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
    np.testing.assert_array_equal(_targets(model, data, None, JOINTS), targets)


@pytest.mark.parametrize("defect", ["gain", "bias", "integrator"])
def test_xml_rejects_controls_that_are_not_position_targets(ip, policy_files, defect):
    spec = mujoco.MjSpec.from_file(str(REPO / ip.MICRODUCK_XML))
    actuator = spec.actuator("left_hip_yaw")
    if defect == "gain":
        actuator.gaintype = mujoco.mjtGain.mjGAIN_AFFINE
    elif defect == "bias":
        actuator.biasprm[1] = 0.0
    else:
        actuator.dyntype = mujoco.mjtDyn.mjDYN_INTEGRATOR
    model = spec.compile()
    data = mujoco.MjData(model)
    with pytest.raises(ValueError, match="position actuator"):
        ip.PolicyInference(
            model, data, walking_onnx_path=str(policy_files[61][0]), new_cmd_obs=True
        )


def test_xml_mouth_respects_actuator_range_inside_joint_range(
    ip, mouth_xml, policy_files
):
    model, data, _ = _load(ip, mouth_xml, bam=False)
    mouth_id = model.actuator(MOUTH_ACTUATOR).id
    model.actuator_ctrlrange[mouth_id] = [-0.05, 0.25]
    policy = ip.PolicyInference(
        model, data, walking_onnx_path=str(policy_files[61][0]), new_cmd_obs=True
    )
    policy.set_position_targets(ip.DEFAULT_POSE)
    policy.set_mouth_target(0.2)
    before = data.ctrl.copy()
    # These values satisfy the joint range but exceed the XML PD control range.
    for target in (-0.075, 0.3):
        with pytest.raises(ValueError):
            policy.set_mouth_target(target)
        np.testing.assert_array_equal(data.ctrl, before)
    for boundary in (-0.05, 0.25):
        policy.set_mouth_target(boundary)
        assert data.ctrl[mouth_id] == boundary
        np.testing.assert_array_equal(
            _targets(model, data, None, JOINTS), ip.DEFAULT_POSE
        )


def test_bam_target_order_can_differ_from_xml_actuator_order(
    ip, mouth_xml, policy_files
):
    from bam.mujoco import MujocoController

    model, data, original = _load(ip, mouth_xml, bam=True)
    ctrl = MujocoController(
        original.model,
        list(reversed(original.actuator)),
        model,
        data,
        vin_drop_gain=0.1,
        vin_min=ip.BAM_VIN_MIN,
    )
    policy = ip.PolicyInference(
        model,
        data,
        walking_onnx_path=str(policy_files[61][0]),
        bam_ctrl=ctrl,
        new_cmd_obs=True,
    )
    sentinels = np.linspace(-0.13, 0.13, 14, dtype=np.float32)
    policy.set_mouth_target(0.2)
    policy.set_position_targets(sentinels)
    np.testing.assert_array_equal(_targets(model, data, ctrl, JOINTS), sentinels)
    assert ctrl.get_q_target(MOUTH_ACTUATOR) == 0.2
    policy.set_mouth_target(0.4)
    np.testing.assert_array_equal(_targets(model, data, ctrl, JOINTS), sentinels)


@pytest.mark.parametrize("defect", ["missing", "duplicate", "tendon", "gear", "slide"])
def test_bad_actuator_layout_is_rejected(ip, mouth_xml, policy_files, tmp_path, defect):
    tree = ET.parse(mouth_xml)
    root = tree.getroot()
    actuators = root.find("actuator")
    actuator = next(act for act in actuators if act.get("name") == "right_ankle")
    if defect == "missing":
        actuators.remove(actuator)
    elif defect == "duplicate":
        duplicate = copy.deepcopy(actuator)
        duplicate.set("name", "duplicate_right_ankle")
        actuators.append(duplicate)
    elif defect == "tendon":
        tendon = ET.SubElement(root, "tendon")
        fixed = ET.SubElement(tendon, "fixed", name="test_transmission")
        ET.SubElement(fixed, "joint", joint="right_ankle", coef="1")
        del actuator.attrib["joint"]
        actuator.set("tendon", "test_transmission")
    elif defect == "gear":
        actuator.set("gear", "2 0 0 0 0 0")
    else:
        root.find(".//joint[@name='mouth']").set("type", "slide")
    xml = tmp_path / f"{defect}.xml"
    tree.write(xml, encoding="unicode")
    # Ensure rejection is from replay validation, not invalid fixture XML.
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    with pytest.raises(ValueError):
        ip.PolicyInference(
            model, data, walking_onnx_path=str(policy_files[61][0]), new_cmd_obs=True
        )
    if defect in ("tendon", "gear", "slide"):
        bam_model = ip.load_bam_model(ip.BAM_KP_FW, 7.4, ip.BAM_MAX_CURRENT)
        with pytest.raises(ValueError):
            ip.load_mujoco_with_bam(str(xml), bam_model, 0.005, 0.1, ip.BAM_VIN_MIN)

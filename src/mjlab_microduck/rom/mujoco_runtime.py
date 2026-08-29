"""Governed MuJoCo/ONNX implementation of the ROM simulation runtime."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import mujoco
import numpy as np
import onnx
import onnxruntime as ort
from numpy.typing import NDArray

from .action_specs import ACTION_RUNTIME_SPECS, STAND_SETTLEMENT_LIMITS
from .contracts import (
    ACTION_CONTRACT,
    CONTROLLED_SERVO_JOINTS,
    OBSERVATION_CONTRACT,
    ActionDefinition,
    ModelArtifact,
    PolicyArtifact,
    PolicyBundle,
    RobotStatus,
    TaskCreateRequest,
    sha256_prefixed,
)
from .model_semantics import has_exact_passive_roller_topology, has_flat_world_floor
from .observation import (
    DEFAULT_JOINT_POSE,
    OBSERVATION_NORMALIZATION,
    DeploymentCommand,
    DeploymentState,
    build_actor_observation,
    project_gravity_wxyz,
)
from .onnx_policy import inspect_normalized_actor
from .runtime import RuntimeEvidence, RuntimeHandle, RuntimeSample

_CONTROL_PERIOD_S = 0.02
_CONTINUOUS_ACTIONS = {
    code
    for code, spec in ACTION_RUNTIME_SPECS.items()
    if spec.execution_mode == "CONTINUOUS_LEASE" and spec.supported
}
_SITTING_JOINT_POSE = DEFAULT_JOINT_POSE.astype(np.float64).copy()
for _index, _value in {
    1: 0.0,
    2: -0.4079,
    3: 1.35,
    4: 0.0,
    10: 0.0,
    11: 0.4079,
    12: -1.35,
    13: 0.0,
}.items():
    _SITTING_JOINT_POSE[_index] = _value


def _digest_bytes(content: bytes) -> str:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def _declared_artifacts(bundle: PolicyBundle) -> list[ModelArtifact]:
    artifacts = [
        bundle.model,
        *(
            ModelArtifact(path=item.path, digest=item.digest)
            for item in bundle.policies
        ),
    ]
    for container, key in (
        (bundle.qualification, "artifacts"),
        (bundle.qualification, "modelClosure"),
        (bundle.license, "artifacts"),
    ):
        raw = container.get(key, [])
        if not isinstance(raw, list):
            raise TypeError("bundle artifact declarations must be lists")
        artifacts.extend(ModelArtifact.model_validate(item) for item in raw)
    return artifacts


def _motion(command: DeploymentCommand) -> dict[str, list[float]]:
    return {
        "twist": np.asarray(command.twist, dtype=np.float64).tolist(),
        "headPose": np.asarray(command.head_pose, dtype=np.float64).tolist(),
        "bodyPose": np.asarray(command.body_pose, dtype=np.float64).tolist(),
    }


class MicroduckMujocoRuntime:
    """Execute verified MicroDuck policy artifacts at a governed 50 Hz rate."""

    controlled_joint_names = CONTROLLED_SERVO_JOINTS

    def __init__(
        self,
        bundle_root: Path,
        bundle: PolicyBundle,
        *,
        realtime: bool = True,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._root = Path(bundle_root).resolve()
        self._bundle = bundle
        self._realtime = realtime
        self._clock = monotonic_clock
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._wait: Callable[[float], bool] = self._stop_event.wait
        self._thread: threading.Thread | None = None
        self._active_handle: RuntimeHandle | None = None
        self._active_action: ActionDefinition | None = None
        self._active_request: TaskCreateRequest | None = None
        self._active_policy: PolicyArtifact | None = None
        self._active_session: ort.InferenceSession | None = None
        self._stopped_evidence: dict[str, RuntimeEvidence] = {}
        self._command = DeploymentCommand.zero()
        self._requested_command = DeploymentCommand.zero()
        self._previous_action = np.zeros(14, dtype=np.float32)
        self._policy_target = DEFAULT_JOINT_POSE.copy()
        self._limiting_reason: str | None = None
        self._terminal_state: str | None = None
        self._terminal_reason: str | None = None
        self._fatal_reason: str | None = None
        self._fallen = False
        self._limp = False
        self._step_count = 0
        self._start_sim_time = 0.0
        self._min_base_height_m = math.inf
        self._max_tilt_rad = 0.0
        self._max_abs_action = 0.0
        self._energy_proxy = 0.0
        self._actuator_clamp_steps = 0
        self._physical_joint_limit_violations = 0
        self._tracking_error_sum = 0.0
        self._tracking_error_max = 0.0
        self._tracking_error_samples = 0
        self._settled_steps = 0
        self._settled_pose_error_max = 0.0
        self._settled_trunk_height_min_m = math.inf
        self._settled_trunk_height_max_m = -math.inf
        self._settled_trunk_tilt_max_rad = 0.0
        self._settled_joint_speed_max_radps = 0.0
        self._upright_steps = 0
        self._last_yaw_rad = 0.0
        self._yaw_rotation_rad = 0.0
        self._start_base_position = np.zeros(3, dtype=np.float64)
        self._last_loop_start: float | None = None
        self._loop_frequency_hz = 50.0 if not realtime else 0.0
        self._loop_overruns = 0
        self._consecutive_overruns = 0
        self._applied_seed = 0
        self._rng = np.random.default_rng(0)
        self._reset_perturbation_l2_rad = 0.0

        artifact_bytes = self._verify_bundle_identity_and_artifacts()
        model_closure = self._derive_model_closure(artifact_bytes)
        self._snapshot = tempfile.TemporaryDirectory(prefix="microduck-mjcf-")
        snapshot_root = Path(self._snapshot.name)
        for declared_path, content in model_closure.items():
            target = snapshot_root / declared_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        model_path = snapshot_root / bundle.model.path
        self._model = mujoco.MjModel.from_xml_path(str(model_path))
        self._data = mujoco.MjData(self._model)
        self._configure_model_addresses()
        self._validate_action_contract_semantics()
        self._model_capabilities = self._detect_model_capabilities()
        self._steps_per_control = round(_CONTROL_PERIOD_S / self._model.opt.timestep)
        if self._steps_per_control < 1 or not math.isclose(
            self._steps_per_control * self._model.opt.timestep,
            _CONTROL_PERIOD_S,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "MuJoCo timestep must divide the 20 ms policy period exactly"
            )
        required_policy_refs = {
            action.policyRef
            for action in bundle.actions
            if action.availability == "AVAILABLE" and action.policyRef is not None
        }
        self._sessions = {
            policy.policyRef: self._load_policy(policy, artifact_bytes[policy.path])
            for policy in bundle.policies
            if policy.policyRef in required_policy_refs
        }
        self._validate_installed_actions()
        self._reset_model_locked()

    def _safe_path(self, declared_path: str) -> Path:
        pure = PurePosixPath(declared_path)
        if (
            not declared_path
            or pure.is_absolute()
            or ".." in pure.parts
            or str(pure) != declared_path
        ):
            raise ValueError("bundle path must not be empty")
        lexical = self._root.joinpath(*pure.parts)
        cursor = self._root
        for part in pure.parts:
            cursor /= part
            if cursor.is_symlink():
                raise ValueError(
                    "bundle artifact symlink is not allowed beneath bundle root"
                )
        candidate = lexical.resolve()
        if candidate == self._root or not candidate.is_relative_to(self._root):
            raise ValueError("bundle artifact must remain beneath the bundle root")
        if not candidate.is_file():
            raise ValueError("bundle artifact is not a file")
        return candidate

    def _verify_bundle_identity_and_artifacts(self) -> dict[str, bytes]:
        manifest_path = self._safe_path("microduck-policy-bundle.json")
        try:
            installed = PolicyBundle.model_validate(
                json.loads(manifest_path.read_text())
            )
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError("bundle manifest verification failed") from exc
        if installed != self._bundle or installed.bundleDigest is None:
            raise ValueError(
                "runtime requires the exact previously verified bundle manifest"
            )

        contents: dict[str, bytes] = {}
        digests: dict[str, str] = {}
        for artifact in _declared_artifacts(installed):
            if artifact.path in contents:
                raise ValueError("bundle artifact verification failed")
            path = self._safe_path(artifact.path)
            content = path.read_bytes()
            if not hmac.compare_digest(_digest_bytes(content), artifact.digest):
                raise ValueError("bundle artifact verification failed")
            contents[artifact.path] = content
            digests[artifact.path] = artifact.digest
        expected_bundle_digest = sha256_prefixed(
            {
                "manifest": installed.model_dump(
                    mode="json", by_alias=True, exclude={"bundleDigest"}
                ),
                "artifacts": digests,
            }
        )
        if not hmac.compare_digest(installed.bundleDigest, expected_bundle_digest):
            raise ValueError("bundle digest verification failed")
        return contents

    def _derive_model_closure(self, declared: Mapping[str, bytes]) -> dict[str, bytes]:
        """Derive MJCF include/mesh/texture closure independently from manifest claims."""
        model_path = PurePosixPath(self._bundle.model.path)
        model_root = model_path.parent
        pending: list[tuple[PurePosixPath, PurePosixPath, PurePosixPath]] = [
            (model_path, model_root, model_root)
        ]
        derived: dict[str, bytes] = {}
        while pending:
            source, inherited_mesh_dir, inherited_texture_dir = pending.pop()
            source_key = str(source)
            if source_key in derived:
                continue
            content = self._safe_path(source_key).read_bytes()
            derived[source_key] = content
            if source.suffix.lower() != ".xml":
                continue
            try:
                root = ET.fromstring(content)
            except ET.ParseError as exc:
                raise ValueError("MJCF dependency XML is invalid") from exc
            compiler = root.find("compiler")
            mesh_dir = inherited_mesh_dir
            texture_dir = inherited_texture_dir
            if compiler is not None and compiler.get("meshdir"):
                mesh_dir = model_root / compiler.get("meshdir", "")
            if compiler is not None and compiler.get("texturedir"):
                texture_dir = model_root / compiler.get("texturedir", "")
            for element in root.iter():
                file_name = element.get("file")
                if not file_name:
                    continue
                tag = element.tag.rsplit("}", 1)[-1]
                if tag == "include":
                    target = source.parent / file_name
                    pending.append((target, mesh_dir, texture_dir))
                elif tag == "mesh":
                    pending.append((mesh_dir / file_name, mesh_dir, texture_dir))
                elif tag == "texture":
                    pending.append((texture_dir / file_name, mesh_dir, texture_dir))
        declared_closure = {
            item.path
            for item in (
                ModelArtifact.model_validate(raw)
                for raw in self._bundle.qualification.get("modelClosure", [])
            )
        }
        if set(derived) - {self._bundle.model.path} != declared_closure:
            raise ValueError("declared model dependency closure is not exact")
        for path, content in derived.items():
            expected = (
                self._bundle.model.digest
                if path == self._bundle.model.path
                else next(
                    ModelArtifact.model_validate(raw).digest
                    for raw in self._bundle.qualification.get("modelClosure", [])
                    if ModelArtifact.model_validate(raw).path == path
                )
            )
            if not hmac.compare_digest(_digest_bytes(content), expected):
                raise ValueError("model dependency closure hash is invalid")
            if path not in declared:
                raise ValueError("model dependency closure contains undeclared bytes")
        return derived

    def _load_policy(
        self, policy: PolicyArtifact, content: bytes
    ) -> ort.InferenceSession:
        requirements = policy.runtimeRequirements
        if requirements.get("observationContract") != OBSERVATION_CONTRACT:
            raise ValueError("policy runtime observation contract is incompatible")
        if requirements.get("actionContract") != ACTION_CONTRACT:
            raise ValueError("policy runtime action contract is incompatible")
        if requirements.get("normalization") != OBSERVATION_NORMALIZATION:
            raise ValueError("policy normalization ownership is incompatible")

        if not hmac.compare_digest(_digest_bytes(content), policy.digest):
            raise ValueError("bundle artifact verification failed before policy load")
        model = onnx.load_from_string(content)
        if (
            len(model.graph.input) != 1
            or len(model.graph.output) != 1
            or model.graph.input[0].type.tensor_type.elem_type != onnx.TensorProto.FLOAT
            or model.graph.output[0].type.tensor_type.elem_type
            != onnx.TensorProto.FLOAT
        ):
            raise ValueError("ONNX policy input and output must be tensor(float)")
        input_dims = [
            dimension.dim_value
            for dimension in model.graph.input[0].type.tensor_type.shape.dim
        ]
        output_dims = [
            dimension.dim_value
            for dimension in model.graph.output[0].type.tensor_type.shape.dim
        ]
        if input_dims != [1, 61]:
            raise ValueError("ONNX policy must have one input of shape [1, 61]")
        if output_dims != [1, 14]:
            raise ValueError("ONNX policy must have one output of shape [1, 14]")
        metadata_proto = {item.key: item.value for item in model.metadata_props}
        graph_digest = hashlib.sha256(model.graph.SerializeToString()).hexdigest()
        normalized_graph = inspect_normalized_actor(model)
        if (
            metadata_proto.get("microduck.normalization")
            != "EMPIRICAL_NORMALIZATION_V1"
            or not hmac.compare_digest(
                metadata_proto.get("microduck.normalization_graph_sha256", ""),
                graph_digest,
            )
            or not hmac.compare_digest(
                str(requirements.get("normalizedGraphFingerprint", "")),
                normalized_graph.fingerprint,
            )
        ):
            raise ValueError("ONNX policy normalization provenance is invalid")
        session = ort.InferenceSession(content, providers=["CPUExecutionProvider"])
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        if len(inputs) != 1 or inputs[0].shape != [1, 61]:
            raise ValueError("ONNX policy must have one input of shape [1, 61]")
        if len(outputs) != 1 or outputs[0].shape != [1, 14]:
            raise ValueError("ONNX policy must have one output of shape [1, 14]")
        metadata = session.get_modelmeta().custom_metadata_map
        expected_metadata = {
            "microduck.task_id": policy.taskId or "",
            "microduck.source_commit": self._bundle.sourceCommit,
            "microduck.observation_contract": OBSERVATION_CONTRACT,
            "microduck.action_contract": ACTION_CONTRACT,
            "microduck.checkpoint": policy.checkpoint or "",
            "microduck.run_identity": policy.experimentRef or "",
        }
        if any(metadata.get(key) != value for key, value in expected_metadata.items()):
            raise ValueError("ONNX metadata does not match bundle provenance")
        output = session.run(
            [outputs[0].name],
            {inputs[0].name: np.zeros((1, 61), dtype=np.float32)},
        )[0]
        if output.shape != (1, 14) or not np.isfinite(output).all():
            raise ValueError("ONNX policy must produce a finite [1, 14] output")
        return session

    def _configure_model_addresses(self) -> None:
        if self._model.nu != len(self.controlled_joint_names):
            raise ValueError("deployment model must have exactly 14 total actuators")
        joint_ids: list[int] = []
        qpos_indices: list[int] = []
        qvel_indices: list[int] = []
        backlash_joint_ids: list[int] = []
        backlash_qpos_indices: list[int] = []
        backlash_qvel_indices: list[int] = []
        backlash_mask: list[float] = []
        actuator_indices: list[int] = []
        for name in self.controlled_joint_names:
            joint_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"controlled joint is missing from model: {name}")
            matching_actuators = np.flatnonzero(
                self._model.actuator_trnid[:, 0] == joint_id
            )
            if matching_actuators.size != 1:
                raise ValueError(
                    f"controlled joint must have exactly one actuator: {name}"
                )
            if self._model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
                raise ValueError("radian controlled joint must be a scalar hinge")
            actuator_id = int(matching_actuators[0])
            gear = self._model.actuator_gear[actuator_id]
            if not np.isfinite(gear).all() or not np.allclose(
                gear, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), atol=1e-12
            ):
                raise ValueError(
                    "controlled joint actuator requires unit positive joint gear"
                )
            if (
                self._model.actuator_gaintype[actuator_id]
                != mujoco.mjtGain.mjGAIN_FIXED
            ):
                raise ValueError("position actuator requires fixed gain semantics")
            if self._model.actuator_dyntype[actuator_id] != mujoco.mjtDyn.mjDYN_NONE:
                raise ValueError("position actuator must not declare actuator dynamics")
            gain_parameters = self._model.actuator_gainprm[actuator_id]
            bias_parameters = self._model.actuator_biasprm[actuator_id]
            if (
                self._model.actuator_trntype[actuator_id] != mujoco.mjtTrn.mjTRN_JOINT
                or self._model.actuator_biastype[actuator_id]
                != mujoco.mjtBias.mjBIAS_AFFINE
                or not np.isfinite(gain_parameters).all()
                or not np.isfinite(bias_parameters).all()
                or gain_parameters[0] <= 0.0
                or not math.isclose(
                    bias_parameters[0],
                    0.0,
                    abs_tol=1e-12,
                    rel_tol=0.0,
                )
                or not math.isclose(
                    bias_parameters[1],
                    -gain_parameters[0],
                    abs_tol=1e-12,
                    rel_tol=0.0,
                )
                or not math.isclose(
                    bias_parameters[2],
                    0.0,
                    abs_tol=1e-12,
                    rel_tol=0.0,
                )
            ):
                raise ValueError(
                    f"controlled joint requires joint-transmission position actuator semantics: {name}"
                )
            if not self._model.actuator_ctrllimited[actuator_id]:
                raise ValueError("position actuator must declare finite control limits")
            low, high = self._model.actuator_ctrlrange[actuator_id]
            if not math.isfinite(low) or not math.isfinite(high) or low >= high:
                raise ValueError("position actuator control limits are invalid")
            backlash_id = mujoco.mj_name2id(
                self._model,
                mujoco.mjtObj.mjOBJ_JOINT,
                f"passive_{name}_backlash",
            )
            has_backlash = backlash_id >= 0
            effective_backlash_id = backlash_id if has_backlash else joint_id
            if (
                has_backlash
                and self._model.jnt_type[backlash_id] != mujoco.mjtJoint.mjJNT_HINGE
            ):
                raise ValueError("backlash companion must be a scalar hinge")
            joint_ids.append(joint_id)
            qpos_indices.append(int(self._model.jnt_qposadr[joint_id]))
            qvel_indices.append(int(self._model.jnt_dofadr[joint_id]))
            backlash_joint_ids.append(backlash_id)
            backlash_qpos_indices.append(
                int(self._model.jnt_qposadr[effective_backlash_id])
            )
            backlash_qvel_indices.append(
                int(self._model.jnt_dofadr[effective_backlash_id])
            )
            backlash_mask.append(1.0 if has_backlash else 0.0)
            actuator_indices.append(actuator_id)
        self._joint_ids = np.asarray(joint_ids, dtype=np.int32)
        self._joint_qpos_indices = np.asarray(qpos_indices, dtype=np.int32)
        self._joint_qvel_indices = np.asarray(qvel_indices, dtype=np.int32)
        self._backlash_joint_ids = np.asarray(backlash_joint_ids, dtype=np.int32)
        self._backlash_qpos_indices = np.asarray(backlash_qpos_indices, dtype=np.int32)
        self._backlash_qvel_indices = np.asarray(backlash_qvel_indices, dtype=np.int32)
        self._backlash_mask = np.asarray(backlash_mask, dtype=np.float64)
        self._actuator_indices = np.asarray(actuator_indices, dtype=np.int32)

        self._trunk_body_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base"
        )
        freejoint_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint"
        )
        if self._trunk_body_id < 0 or freejoint_id < 0:
            raise ValueError("model must declare trunk_base and trunk_base_freejoint")
        if (
            self._model.jnt_type[freejoint_id] != mujoco.mjtJoint.mjJNT_FREE
            or self._model.jnt_bodyid[freejoint_id] != self._trunk_body_id
        ):
            raise ValueError(
                "trunk_base_freejoint must be a FREE joint owned by trunk_base"
            )
        self._free_qpos_address = int(self._model.jnt_qposadr[freejoint_id])
        self._free_qvel_address = int(self._model.jnt_dofadr[freejoint_id])
        self._gyro_sensor_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel"
        )
        if self._gyro_sensor_id < 0:
            raise ValueError("model must provide trunk-frame imu_ang_vel gyro")
        sensor_id = self._gyro_sensor_id
        if (
            self._model.sensor_type[sensor_id] != mujoco.mjtSensor.mjSENS_GYRO
            or self._model.sensor_dim[sensor_id] != 3
            or self._model.sensor_objtype[sensor_id] != mujoco.mjtObj.mjOBJ_SITE
        ):
            raise ValueError("imu_ang_vel must be a three-axis gyro sensor")
        site_id = int(self._model.sensor_objid[sensor_id])
        if self._model.site_bodyid[site_id] != self._trunk_body_id or not np.allclose(
            self._model.site_quat[site_id],
            np.array([1.0, 0.0, 0.0, 0.0]),
            atol=1e-7,
        ):
            raise ValueError("imu_ang_vel site must be identity-aligned on trunk_base")

    def _reset_model_locked(
        self,
        rng: np.random.Generator | None = None,
        reset_profile: str = "DEFAULT_STANDING",
    ) -> None:
        mujoco.mj_resetData(self._model, self._data)
        perturbation = (
            np.zeros(14, dtype=np.float64)
            if rng is None
            else rng.uniform(-0.005, 0.005, size=14)
        )
        if reset_profile == "TRAINED_SITTING":
            reset_pose = _SITTING_JOINT_POSE + perturbation
            self._data.qpos[self._free_qpos_address + 2] = 0.07
        elif reset_profile == "DEFAULT_STANDING":
            reset_pose = DEFAULT_JOINT_POSE.astype(np.float64) + perturbation
        else:
            raise ValueError("runtime reset profile is not implemented")
        self._data.qpos[self._joint_qpos_indices] = reset_pose
        self._data.ctrl[self._actuator_indices] = reset_pose
        self._reset_perturbation_l2_rad = float(np.linalg.norm(perturbation))
        mujoco.mj_forward(self._model, self._data)

    def _detect_model_capabilities(self) -> frozenset[str]:
        capabilities: set[str] = set()
        terrain = self._bundle.qualification.get("modelTerrain")
        if terrain == "flat" and has_flat_world_floor(self._model):
            capabilities.add("FLAT_TERRAIN")
            capabilities.add("SITTING_RESET")
        if terrain in {"ramp", "slope"}:
            capabilities.add("RAMP_TERRAIN")
        if has_exact_passive_roller_topology(self._model):
            capabilities.add("ROLLER_FEET")
        return frozenset(capabilities)

    def _validate_action_contract_semantics(self) -> None:
        if set(self._bundle.actionContract.scaling) - {"actionScale"}:
            raise ValueError("action contract declares unsupported scaling semantics")
        self._action_scale()
        clipping = self._bundle.actionContract.clipping
        if clipping not in ({}, {"strategy": "ACTUATOR_CTRL_RANGE"}):
            raise ValueError(
                "action contract clipping is incompatible with position targets"
            )

    def _validate_installed_actions(self) -> None:
        """Make readiness prove every manifest-available action can reach start()."""
        policies = {item.policyRef: item for item in self._bundle.policies}
        for action in self._bundle.actions:
            if action.availability != "AVAILABLE":
                continue
            spec = ACTION_RUNTIME_SPECS.get(action.actionCode)
            if spec is None or not spec.supported:
                raise ValueError("available action has no runtime semantics")
            if action.executionMode != spec.execution_mode:
                raise ValueError("available action execution mode is incompatible")
            if (
                action.actionCode == "STAND"
                and action.parameterSchema.get("x-microduck-fixed-goal") != "STAND"
            ):
                raise ValueError("STAND requires the fixed SitStand posture goal")
            policy = policies.get(action.policyRef or "")
            if policy is None or policy.policyRef not in self._sessions:
                raise ValueError("available action has no verified policy")
            if policy.taskId not in spec.task_ids:
                raise ValueError("policy task identity does not match action semantics")
            if set(spec.required_capabilities) - self._model_capabilities:
                raise ValueError("available action lacks qualified model capabilities")
            if (
                self._bundle.qualification.get("scenarioProfile")
                != spec.scenario_profile
            ):
                raise ValueError(
                    "available action lacks its code-owned scenario profile"
                )
            expected_preconditions = {
                "allowedTerrains": [self._bundle.qualification.get("modelTerrain")],
                "scenarioProfile": spec.scenario_profile,
            }
            if action.preconditions != expected_preconditions:
                raise ValueError("action preconditions do not match qualification")

    def validate(self, action: ActionDefinition, request: TaskCreateRequest) -> None:
        if request.bundleDigest != self._bundle.bundleDigest:
            raise ValueError("request bundle digest does not match runtime bundle")
        if request.bundleVersion != self._bundle.bundleVersion:
            raise ValueError("request bundle version does not match runtime bundle")
        if (
            request.actionCode != action.actionCode
            or action not in self._bundle.actions
        ):
            raise ValueError("runtime action does not match the installed bundle")
        if action.availability != "AVAILABLE" or action.policyRef not in self._sessions:
            raise ValueError("runtime action has no verified policy")
        spec = ACTION_RUNTIME_SPECS.get(action.actionCode)
        if spec is None or not spec.supported:
            raise ValueError("runtime action semantics are unavailable")
        if action.executionMode != spec.execution_mode:
            raise ValueError("runtime action execution mode is incompatible")
        missing_capabilities = (
            set(spec.required_capabilities) - self._model_capabilities
        )
        if missing_capabilities:
            raise ValueError("loaded model lacks required action capabilities")
        policy = next(
            item for item in self._bundle.policies if item.policyRef == action.policyRef
        )
        if policy.taskId not in spec.task_ids:
            raise ValueError(
                "policy task identity does not match code-owned action semantics"
            )
        self._command_for(action.actionCode, request.parameters, action)
        if set(request.scenario) != set(spec.scenario_fields):
            raise ValueError("scenario must contain exact terrain and seed fields")
        if self._bundle.qualification.get("scenarioProfile") != spec.scenario_profile:
            raise ValueError("bundle scenario profile is incompatible")
        seed = request.scenario.get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32:
            raise ValueError("scenario seed must be an unsigned 32-bit integer")
        terrain = request.scenario.get("terrain")
        if terrain != self._bundle.qualification.get("modelTerrain"):
            raise ValueError(
                "requested terrain is not bound to the qualified loaded model"
            )

    def start(
        self, action: ActionDefinition, request: TaskCreateRequest
    ) -> RuntimeHandle:
        self.validate(action, request)
        with self._lock:
            if self._active_handle is not None:
                raise RuntimeError("runtime already has an active task")
            if self._fatal_reason is not None:
                raise RuntimeError("runtime requires restart after a safety fault")
            self._applied_seed = int(request.scenario.get("seed", 0))
            self._rng = np.random.default_rng(self._applied_seed)
            spec = ACTION_RUNTIME_SPECS[action.actionCode]
            self._reset_model_locked(self._rng, spec.reset_profile)
            self._active_handle = RuntimeHandle(taskId=request.taskId)
            self._active_action = action
            self._active_request = request
            self._active_policy = next(
                item
                for item in self._bundle.policies
                if item.policyRef == action.policyRef
            )
            self._active_session = self._sessions[self._active_policy.policyRef]
            requested_command, command, limiting_reason = self._command_for(
                action.actionCode, request.parameters, action
            )
            self._requested_command = requested_command
            self._command = command
            self._limiting_reason = limiting_reason
            self._previous_action = np.zeros(14, dtype=np.float32)
            self._policy_target = DEFAULT_JOINT_POSE.copy()
            self._terminal_state = None
            self._terminal_reason = None
            self._fallen = False
            self._limp = False
            self._step_count = 0
            self._start_sim_time = float(self._data.time)
            self._min_base_height_m = float(self._base_position()[2])
            self._max_tilt_rad = 0.0
            self._max_abs_action = 0.0
            self._energy_proxy = 0.0
            self._actuator_clamp_steps = 0
            self._physical_joint_limit_violations = 0
            self._tracking_error_sum = 0.0
            self._tracking_error_max = 0.0
            self._tracking_error_samples = 0
            self._settled_steps = 0
            self._reset_stand_settlement_window_locked()
            self._upright_steps = 0
            self._last_yaw_rad = self._yaw_rad()
            self._yaw_rotation_rad = 0.0
            self._start_base_position = self._base_position()
            self._last_loop_start = None
            self._loop_frequency_hz = 50.0 if not self._realtime else 0.0
            self._loop_overruns = 0
            self._consecutive_overruns = 0
            self._stop_event.clear()
            handle = self._active_handle
            if self._realtime:
                self._thread = threading.Thread(
                    target=self._governed_loop,
                    name=f"microduck-policy-{request.taskId}",
                    daemon=True,
                )
                self._thread.start()
            return handle

    def command(self, handle: RuntimeHandle, parameters: Mapping[str, object]) -> None:
        with self._lock:
            self._require_handle(handle)
            assert self._active_action is not None
            requested_command, command, limiting_reason = self._command_for(
                self._active_action.actionCode, parameters, self._active_action
            )
            self._requested_command = requested_command
            self._command = command
            self._limiting_reason = limiting_reason

    def sample(self, handle: RuntimeHandle) -> RuntimeSample:
        with self._lock:
            self._require_handle(handle)
        if not self._realtime:
            self._control_step()
        with self._lock:
            self._require_handle(handle)
            if self._terminal_state is not None:
                return RuntimeSample(
                    running=False,
                    terminalState=self._terminal_state,  # type: ignore[arg-type]
                    metrics=self._action_metrics_locked(),
                    stopReason=self._terminal_reason,
                )
            assert self._active_action is not None
            return RuntimeSample(running=True, metrics=self._action_metrics_locked())

    def safe_stop(self, handle: RuntimeHandle | None, reason: str) -> RuntimeEvidence:
        with self._lock:
            if handle is None and self._active_handle is not None:
                raise RuntimeError("an active task requires its owned runtime handle")
            if handle is not None:
                if (
                    self._active_handle is None
                    and handle.taskId in self._stopped_evidence
                ):
                    return self._stopped_evidence[handle.taskId]
                self._require_handle(handle)
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        with self._lock:
            if self._active_handle is None:
                return RuntimeEvidence(stopReason=reason)
            metrics = self._evidence_metrics_locked()
            if self._fatal_reason is not None:
                self._disable_actuators_locked()
            else:
                self._hold_current_position_locked()
            evidence = RuntimeEvidence(metrics=metrics, stopReason=reason)
            stopped_task_id = self._active_handle.taskId
            self._active_handle = None
            self._active_action = None
            self._active_request = None
            self._active_policy = None
            self._active_session = None
            self._thread = None
            self._limp = self._fatal_reason is not None
            self._stopped_evidence[stopped_task_id] = evidence
            return evidence

    def status(self) -> RobotStatus:
        with self._lock:
            position = self._finite_tuple(self._base_position(), 3)
            quaternion_wxyz = self._finite_array(self._base_quaternion_wxyz(), 4)
            orientation_xyzw = (
                float(quaternion_wxyz[1]),
                float(quaternion_wxyz[2]),
                float(quaternion_wxyz[3]),
                float(quaternion_wxyz[0]),
            )
            joints = self._finite_tuple(self._encoder_positions(), 14)
            joint_velocities = self._finite_tuple(self._encoder_velocities(), 14)
            ready = self._fatal_reason is None and not self._fallen
            reason_codes = (
                [self._fatal_reason]
                if self._fatal_reason is not None
                else (["FALLEN"] if self._fallen else [])
            )
            return RobotStatus(
                schema="BIPED_POSE_V1",
                timestamp=datetime.now(UTC),
                basePositionM=position,
                baseOrientationXyzw=orientation_xyzw,
                baseLinearVelocityMps=self._finite_tuple(
                    self._data.qvel[
                        self._free_qvel_address : self._free_qvel_address + 3
                    ],
                    3,
                ),
                baseAngularVelocityRadps=self._finite_tuple(
                    self._base_angular_velocity(), 3
                ),
                jointPositionsRad=joints,
                jointVelocitiesRadps=joint_velocities,
                policyTarget={
                    "jointPositionsRad": self._finite_array(self._policy_target, 14)
                    .astype(float)
                    .tolist()
                },
                requestedMotion=_motion(self._requested_command),
                appliedMotion=_motion(self._command),
                limitingReason=self._limiting_reason,
                activePolicyRef=(
                    self._active_policy.policyRef if self._active_policy else None
                ),
                activeActionCode=(
                    self._active_action.actionCode if self._active_action else None
                ),
                activeTaskId=(
                    self._active_request.taskId if self._active_request else None
                ),
                simulationTimeS=max(0.0, float(self._data.time)),
                loopFrequencyHz=max(0.0, self._loop_frequency_hz),
                fallen=self._fallen,
                limp=self._limp,
                health={
                    "ready": ready,
                    "healthy": ready,
                    "reasonCodes": reason_codes,
                    "baseLinearVelocityFrame": "WORLD",
                    "baseAngularVelocityFrame": "TRUNK_BODY",
                },
            )

    def _governed_loop(self) -> None:
        while not self._stop_event.is_set():
            started_at = self._clock()
            with self._lock:
                if self._last_loop_start is not None:
                    interval = started_at - self._last_loop_start
                    if interval > 0.0:
                        self._loop_frequency_hz = 1.0 / interval
                self._last_loop_start = started_at
            self._control_step()
            elapsed = self._clock() - started_at
            with self._lock:
                if elapsed > _CONTROL_PERIOD_S + 1e-9:
                    self._loop_overruns += 1
                    self._consecutive_overruns += 1
                    if self._consecutive_overruns >= 3:
                        self._fail_locked("CONTROL_LOOP_OVERRUN")
                else:
                    self._consecutive_overruns = 0
            # Every deadline is based on this iteration's measured start.  A
            # late iteration therefore never creates catch-up bursts.
            remaining = started_at + _CONTROL_PERIOD_S - self._clock()
            if remaining > 0.0:
                self._wait(remaining)

    def _control_step(self) -> None:
        with self._lock:
            if self._active_handle is None or self._terminal_state is not None:
                return
            try:
                if self._limiting_reason == "ACTUATOR_LIMIT":
                    self._limiting_reason = (
                        "COMMAND_LIMIT"
                        if _motion(self._requested_command) != _motion(self._command)
                        else None
                    )
                self._require_finite_simulation_state()
                state = DeploymentState(
                    base_angular_velocity_radps=self._base_angular_velocity(),
                    base_orientation_wxyz=self._base_quaternion_wxyz(),
                    joint_positions_rad=self._encoder_positions(),
                    joint_velocities_radps=self._encoder_velocities(),
                    previous_action=self._previous_action,
                )
                observation = build_actor_observation(state, self._command)
                assert self._active_session is not None
                actor_input = self._active_session.get_inputs()[0]
                actor_output = self._active_session.get_outputs()[0]
                action = self._active_session.run(
                    [actor_output.name],
                    {actor_input.name: observation.reshape(1, 61)},
                )[0]
                if action.shape != (1, 14) or not np.isfinite(action).all():
                    raise FloatingPointError("NON_FINITE_POLICY_OUTPUT")
                policy_action = action[0].astype(np.float32, copy=False)
                target = DEFAULT_JOINT_POSE + policy_action * self._action_scale()
                target, actuator_limited = self._limit_targets(target)
                if actuator_limited:
                    self._limiting_reason = "ACTUATOR_LIMIT"
                    self._actuator_clamp_steps += 1
                self._data.ctrl[self._actuator_indices] = target
                self._policy_target = target.copy()
                self._previous_action = policy_action.copy()
                for _ in range(self._steps_per_control):
                    mujoco.mj_step(self._model, self._data)
                self._step_count += 1
                self._update_safety_metrics_locked(policy_action)
                if self._active_action.actionCode in _CONTINUOUS_ACTIONS:
                    tracking_error = self._velocity_tracking_error_locked()
                    self._tracking_error_sum += tracking_error
                    self._tracking_error_max = max(
                        self._tracking_error_max, tracking_error
                    )
                    self._tracking_error_samples += 1
                elif self._active_action.actionCode == "STAND":
                    tracking_error = self._stand_pose_error_locked()
                    self._tracking_error_sum += tracking_error
                    self._tracking_error_max = max(
                        self._tracking_error_max, tracking_error
                    )
                    self._tracking_error_samples += 1
                    self._update_stand_settlement_locked(tracking_error)
                self._require_finite_simulation_state()
                self._check_joint_limits()
                self._check_fall_locked()
                if (
                    self._active_action.actionCode == "STAND"
                    and self._settled_steps
                    >= STAND_SETTLEMENT_LIMITS.required_consecutive_steps
                ):
                    self._terminal_state = "SUCCEEDED"
                    self._terminal_reason = "STAND_POSE_SETTLED"
                    self._stop_event.set()
            except FloatingPointError as exc:
                self._fail_locked(str(exc))
            except Exception:  # noqa: BLE001 - any runtime failure must safe-stop.
                self._fail_locked("RUNTIME_EXCEPTION")

    def _action_scale(self) -> float:
        raw = self._bundle.actionContract.scaling.get("actionScale", 1.0)
        if (
            not isinstance(raw, int | float)
            or isinstance(raw, bool)
            or not math.isfinite(raw)
        ):
            raise ValueError("action scale must be a finite number")
        return float(raw)

    def _limit_targets(
        self, target: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], bool]:
        limited = target.copy()
        changed = False
        for index, actuator_id in enumerate(self._actuator_indices):
            if self._model.actuator_ctrllimited[actuator_id]:
                low, high = self._model.actuator_ctrlrange[actuator_id]
                clipped = float(np.clip(limited[index], low, high))
                changed |= clipped != float(limited[index])
                limited[index] = clipped
        return limited, changed

    def _command_for(
        self,
        action_code: str,
        parameters: Mapping[str, object],
        action: ActionDefinition,
    ) -> tuple[DeploymentCommand, DeploymentCommand, str | None]:
        if action_code in _CONTINUOUS_ACTIONS:
            expected = ("vxMps", "vyMps", "yawRateRadps")
            if set(parameters) != set(expected):
                raise ValueError(
                    "continuous command must contain the exact velocity fields"
                )
            requested: list[float] = []
            applied: list[float] = []
            properties = action.parameterSchema.get("properties", {})
            for name in expected:
                value = parameters[name]
                if (
                    not isinstance(value, int | float)
                    or isinstance(value, bool)
                    or not math.isfinite(float(value))
                ):
                    raise ValueError("continuous command values must be finite numbers")
                numeric = float(value)
                requested.append(numeric)
                schema = properties.get(name, {})
                low = float(schema.get("minimum", numeric))
                high = float(schema.get("maximum", numeric))
                applied.append(float(np.clip(numeric, low, high)))
            zero_head = np.zeros(4, dtype=np.float32)
            zero_body = np.zeros(6, dtype=np.float32)
            return (
                DeploymentCommand(
                    twist=np.asarray(requested, dtype=np.float64),
                    head_pose=zero_head,
                    body_pose=zero_body,
                ),
                DeploymentCommand(
                    twist=np.asarray(applied, dtype=np.float64),
                    head_pose=zero_head,
                    body_pose=zero_body,
                ),
                "COMMAND_LIMIT" if applied != requested else None,
            )
        if action_code == "STAND":
            if parameters:
                raise ValueError("STAND command does not accept parameters")
            if action.parameterSchema.get("x-microduck-fixed-goal") != "STAND":
                raise ValueError("STAND fixed posture goal is invalid")
            command = DeploymentCommand.zero()
            return command, command, None
        raise ValueError("runtime action has no implemented typed command profile")

    def _require_handle(self, handle: RuntimeHandle) -> None:
        if self._active_handle != handle:
            raise RuntimeError("runtime handle does not own the active task")

    def _require_finite_simulation_state(self) -> None:
        for values in (self._data.qpos, self._data.qvel, self._data.ctrl):
            if not np.isfinite(values).all():
                raise FloatingPointError("NON_FINITE_STATE")

    def _check_joint_limits(self) -> None:
        for joint_id, qpos_index in zip(
            self._joint_ids, self._joint_qpos_indices, strict=True
        ):
            if self._model.jnt_limited[joint_id]:
                low, high = self._model.jnt_range[joint_id]
                value = self._data.qpos[qpos_index]
                if value < low - 1e-4 or value > high + 1e-4:
                    self._physical_joint_limit_violations += 1
                    raise FloatingPointError("JOINT_LIMIT")

    def _check_fall_locked(self) -> None:
        gravity = project_gravity_wxyz(self._base_quaternion_wxyz())
        tilt = math.acos(float(np.clip(-gravity[2], -1.0, 1.0)))
        height = float(self._base_position()[2])
        if height < 0.025 or tilt > math.radians(75.0):
            self._fallen = True
            self._fail_locked("FALLEN", fallen=True)

    def _update_safety_metrics_locked(self, action: NDArray[np.float32]) -> None:
        gravity = project_gravity_wxyz(self._base_quaternion_wxyz())
        tilt = math.acos(float(np.clip(-gravity[2], -1.0, 1.0)))
        height = float(self._base_position()[2])
        self._min_base_height_m = min(self._min_base_height_m, height)
        self._max_tilt_rad = max(self._max_tilt_rad, tilt)
        self._max_abs_action = max(
            self._max_abs_action, float(np.max(np.abs(action), initial=0.0))
        )
        # Integral of squared normalized policy action. This is deliberately an
        # energy proxy, not a claim about electrical joules or actuator torque.
        self._energy_proxy += float(np.dot(action, action)) * _CONTROL_PERIOD_S
        if height >= 0.025 and tilt <= math.radians(75.0):
            self._upright_steps += 1
        yaw = self._yaw_rad()
        yaw_delta = (yaw - self._last_yaw_rad + math.pi) % (2.0 * math.pi) - math.pi
        self._yaw_rotation_rad += yaw_delta
        self._last_yaw_rad = yaw

    def _fail_locked(self, reason: str, *, fallen: bool = False) -> None:
        self._fatal_reason = reason
        self._terminal_state = "FAILED"
        self._terminal_reason = "FALLEN" if fallen else reason
        self._fallen |= fallen
        self._limp = True
        self._disable_actuators_locked()
        self._stop_event.set()

    def _hold_current_position_locked(self) -> None:
        current = self._finite_array(self._data.qpos[self._joint_qpos_indices], 14)
        self._data.ctrl[self._actuator_indices] = current

    def _disable_actuators_locked(self) -> None:
        """Make fatal limp truthful by removing position-servo gain and bias."""
        self._data.ctrl[self._actuator_indices] = 0.0
        self._model.actuator_gainprm[self._actuator_indices] = 0.0
        self._model.actuator_biasprm[self._actuator_indices] = 0.0

    def _encoder_positions(self) -> NDArray[np.float64]:
        return (
            self._data.qpos[self._joint_qpos_indices]
            + self._data.qpos[self._backlash_qpos_indices] * self._backlash_mask
        )

    def _encoder_velocities(self) -> NDArray[np.float64]:
        return (
            self._data.qvel[self._joint_qvel_indices]
            + self._data.qvel[self._backlash_qvel_indices] * self._backlash_mask
        )

    def _base_position(self) -> NDArray[np.float64]:
        return self._data.xpos[self._trunk_body_id].copy()

    def _base_quaternion_wxyz(self) -> NDArray[np.float64]:
        return self._data.xquat[self._trunk_body_id].copy()

    def _base_angular_velocity(self) -> NDArray[np.float64]:
        address = int(self._model.sensor_adr[self._gyro_sensor_id])
        return self._data.sensordata[address : address + 3].copy()

    def _yaw_rad(self) -> float:
        w, x, y, z = self._base_quaternion_wxyz()
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def _duration_locked(self) -> float:
        return max(0.0, float(self._data.time) - self._start_sim_time)

    def _velocity_tracking_error_locked(self) -> float:
        world_linear = self._data.qvel[
            self._free_qvel_address : self._free_qvel_address + 3
        ]
        world_from_body = self._data.xmat[self._trunk_body_id].reshape(3, 3)
        body_linear = world_from_body.T @ world_linear
        measured = np.array(
            [body_linear[0], body_linear[1], self._base_angular_velocity()[2]]
        )
        return float(np.linalg.norm(measured - np.asarray(self._command.twist)))

    def _stand_pose_error_locked(self) -> float:
        delta = self._encoder_positions() - DEFAULT_JOINT_POSE
        return float(np.sqrt(np.mean(np.square(delta))))

    def _reset_stand_settlement_window_locked(self) -> None:
        self._settled_steps = 0
        self._settled_pose_error_max = 0.0
        self._settled_trunk_height_min_m = math.inf
        self._settled_trunk_height_max_m = -math.inf
        self._settled_trunk_tilt_max_rad = 0.0
        self._settled_joint_speed_max_radps = 0.0

    def _update_stand_settlement_locked(self, pose_error: float) -> None:
        gravity = project_gravity_wxyz(self._base_quaternion_wxyz())
        tilt = math.acos(float(np.clip(-gravity[2], -1.0, 1.0)))
        trunk_height = float(self._base_position()[2])
        joint_speed = float(np.max(np.abs(self._encoder_velocities()), initial=0.0))
        limits = STAND_SETTLEMENT_LIMITS
        if not (
            pose_error <= limits.pose_error_max_rad
            and limits.trunk_height_min_m <= trunk_height <= limits.trunk_height_max_m
            and tilt <= limits.trunk_tilt_max_rad
            and joint_speed <= limits.joint_speed_max_radps
        ):
            self._reset_stand_settlement_window_locked()
            return
        self._settled_steps += 1
        self._settled_pose_error_max = max(self._settled_pose_error_max, pose_error)
        self._settled_trunk_height_min_m = min(
            self._settled_trunk_height_min_m, trunk_height
        )
        self._settled_trunk_height_max_m = max(
            self._settled_trunk_height_max_m, trunk_height
        )
        self._settled_trunk_tilt_max_rad = max(self._settled_trunk_tilt_max_rad, tilt)
        self._settled_joint_speed_max_radps = max(
            self._settled_joint_speed_max_radps, joint_speed
        )

    def _action_metrics_locked(self) -> dict[str, int | float | bool | str]:
        assert self._active_action is not None
        base_position = self._base_position()
        gravity = project_gravity_wxyz(self._base_quaternion_wxyz())
        final_tilt = math.acos(float(np.clip(-gravity[2], -1.0, 1.0)))
        metrics: dict[str, int | float | bool | str] = {
            "actionCode": self._active_action.actionCode,
            "baseTravelM": round(
                float(np.linalg.norm(base_position - self._start_base_position)), 6
            ),
            "durationS": round(self._duration_locked(), 6),
            "fallen": self._fallen,
            "finalBaseHeightM": round(float(base_position[2]), 6),
            "finalTiltRad": round(final_tilt, 6),
            "maxAbsAction": round(self._max_abs_action, 6),
            "energyProxy": round(self._energy_proxy, 6),
            "actuatorClampSteps": self._actuator_clamp_steps,
            "physicalJointLimitViolations": self._physical_joint_limit_violations,
            "maxTiltRad": round(self._max_tilt_rad, 6),
            "minBaseHeightM": round(self._min_base_height_m, 6),
            "steps": self._step_count,
            "loopOverruns": self._loop_overruns,
        }
        if self._active_action.actionCode in {
            "WALK_VELOCITY",
            "VELSTAND_VELOCITY",
            "ROLLER_VELOCITY",
            "SWIZZLE",
        }:
            metrics["trackingError"] = round(
                self._tracking_error_sum / max(self._tracking_error_samples, 1),
                6,
            )
            metrics["trackingErrorSum"] = round(self._tracking_error_sum, 6)
            metrics["trackingErrorMax"] = round(self._tracking_error_max, 6)
            metrics["trackingErrorSamples"] = self._tracking_error_samples
        if self._active_action.actionCode == "VELSTAND_VELOCITY":
            metrics["uprightSteps"] = self._upright_steps
            metrics["standFraction"] = round(
                self._upright_steps / max(self._step_count, 1), 6
            )
        if self._active_action.actionCode == "SWIZZLE":
            metrics["yawRotationRad"] = round(self._yaw_rotation_rad, 6)
        if self._active_action.actionCode == "STAND":
            for redundant_key in (
                "durationS",
                "finalBaseHeightM",
                "finalTiltRad",
                "loopOverruns",
                "maxTiltRad",
                "minBaseHeightM",
            ):
                metrics.pop(redundant_key)
            metrics["standPoseError"] = round(self._stand_pose_error_locked(), 6)
            metrics["trackingError"] = round(
                self._tracking_error_sum / max(self._tracking_error_samples, 1), 6
            )
            metrics["trackingErrorSum"] = round(self._tracking_error_sum, 6)
            metrics["trackingErrorMax"] = round(self._tracking_error_max, 6)
            metrics["trackingErrorSamples"] = self._tracking_error_samples
            metrics["standSettledSteps"] = self._settled_steps
            if self._settled_steps:
                metrics["settledPoseErrorMax"] = round(self._settled_pose_error_max, 6)
                metrics["settledHeightMinM"] = round(
                    self._settled_trunk_height_min_m, 6
                )
                metrics["settledHeightMaxM"] = round(
                    self._settled_trunk_height_max_m, 6
                )
                metrics["settledTiltMaxRad"] = round(
                    self._settled_trunk_tilt_max_rad, 6
                )
                metrics["settledJointSpeedMaxRadps"] = round(
                    self._settled_joint_speed_max_radps, 6
                )
        return metrics

    def _evidence_metrics_locked(self) -> dict[str, int | float | bool | str | None]:
        assert self._active_policy is not None
        assert self._active_request is not None
        evidence: dict[str, int | float | bool | str | None] = {
            **self._action_metrics_locked(),
            "bundleDigest": self._bundle.bundleDigest,
            "onnxDigest": self._active_policy.digest,
            "mjcfDigest": self._bundle.model.digest,
            "sourceCommit": self._bundle.sourceCommit,
            "checkpoint": self._active_policy.checkpoint,
            "runIdentity": self._active_policy.experimentRef,
            "terrainIdentity": str(self._bundle.qualification.get("modelTerrain", "")),
            "rngSeed": self._applied_seed,
            "scenarioProfile": str(
                self._bundle.qualification.get("scenarioProfile", "")
            ),
            "resetProfile": ACTION_RUNTIME_SPECS[
                self._active_action.actionCode
            ].reset_profile,
        }
        if self._active_action.actionCode != "STAND":
            evidence["resetPerturbationL2Rad"] = round(
                self._reset_perturbation_l2_rad, 8
            )
        return evidence

    @staticmethod
    def _finite_array(values: Any, length: int) -> NDArray[np.float64]:
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (length,):
            return np.zeros(length, dtype=np.float64)
        return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)

    @classmethod
    def _finite_tuple(cls, values: Any, length: int) -> tuple[float, ...]:
        return tuple(float(value) for value in cls._finite_array(values, length))

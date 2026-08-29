"""Governed MuJoCo/ONNX implementation of the ROM simulation runtime."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import threading
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray

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
from .observation import (
    DEFAULT_JOINT_POSE,
    OBSERVATION_NORMALIZATION,
    DeploymentCommand,
    DeploymentState,
    build_actor_observation,
    project_gravity_wxyz,
)
from .runtime import RuntimeEvidence, RuntimeHandle, RuntimeSample

_CONTROL_PERIOD_S = 0.02
_CONTINUOUS_ACTIONS = {
    "WALK_VELOCITY",
    "VELSTAND_VELOCITY",
    "ROLLER_VELOCITY",
    "SWIZZLE",
}


def _digest_bytes(content: bytes) -> str:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def _digest_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1_048_576), b""):
            hasher.update(chunk)
    return f"sha256:{hasher.hexdigest()}"


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
        self._thread: threading.Thread | None = None
        self._active_handle: RuntimeHandle | None = None
        self._active_action: ActionDefinition | None = None
        self._active_request: TaskCreateRequest | None = None
        self._active_policy: PolicyArtifact | None = None
        self._active_session: ort.InferenceSession | None = None
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
        self._start_base_position = np.zeros(3, dtype=np.float64)

        paths = self._verify_bundle_identity_and_artifacts()
        model_path = paths[bundle.model.path]
        if not hmac.compare_digest(_digest_file(model_path), bundle.model.digest):
            raise ValueError("bundle artifact verification failed before model load")
        self._model = mujoco.MjModel.from_xml_path(str(model_path))
        self._data = mujoco.MjData(self._model)
        self._configure_model_addresses()
        self._steps_per_control = round(_CONTROL_PERIOD_S / self._model.opt.timestep)
        if self._steps_per_control < 1 or not math.isclose(
            self._steps_per_control * self._model.opt.timestep,
            _CONTROL_PERIOD_S,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "MuJoCo timestep must divide the 20 ms policy period exactly"
            )
        self._sessions = {
            policy.policyRef: self._load_policy(policy, paths[policy.path])
            for policy in bundle.policies
        }
        self._reset_model_locked()

    def _safe_path(self, declared_path: str) -> Path:
        if not declared_path:
            raise ValueError("bundle path must not be empty")
        candidate = (self._root / declared_path).resolve()
        if candidate == self._root or not candidate.is_relative_to(self._root):
            raise ValueError("bundle artifact must remain beneath the bundle root")
        if not candidate.is_file():
            raise ValueError("bundle artifact is not a file")
        return candidate

    def _verify_bundle_identity_and_artifacts(self) -> dict[str, Path]:
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

        paths: dict[str, Path] = {}
        digests: dict[str, str] = {}
        for artifact in _declared_artifacts(installed):
            if artifact.path in paths:
                raise ValueError("bundle artifact verification failed")
            path = self._safe_path(artifact.path)
            if not hmac.compare_digest(_digest_file(path), artifact.digest):
                raise ValueError("bundle artifact verification failed")
            paths[artifact.path] = path
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
        return paths

    def _load_policy(
        self, policy: PolicyArtifact, policy_path: Path
    ) -> ort.InferenceSession:
        requirements = policy.runtimeRequirements
        if requirements.get("observationContract") != OBSERVATION_CONTRACT:
            raise ValueError("policy runtime observation contract is incompatible")
        if requirements.get("actionContract") != ACTION_CONTRACT:
            raise ValueError("policy runtime action contract is incompatible")
        if requirements.get("normalization") != OBSERVATION_NORMALIZATION:
            raise ValueError("policy normalization ownership is incompatible")

        content = policy_path.read_bytes()
        if not hmac.compare_digest(_digest_bytes(content), policy.digest):
            raise ValueError("bundle artifact verification failed before policy load")
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
        joint_ids: list[int] = []
        qpos_indices: list[int] = []
        qvel_indices: list[int] = []
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
            joint_ids.append(joint_id)
            qpos_indices.append(int(self._model.jnt_qposadr[joint_id]))
            qvel_indices.append(int(self._model.jnt_dofadr[joint_id]))
            actuator_indices.append(int(matching_actuators[0]))
        self._joint_ids = np.asarray(joint_ids, dtype=np.int32)
        self._joint_qpos_indices = np.asarray(qpos_indices, dtype=np.int32)
        self._joint_qvel_indices = np.asarray(qvel_indices, dtype=np.int32)
        self._actuator_indices = np.asarray(actuator_indices, dtype=np.int32)

        self._trunk_body_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base"
        )
        freejoint_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint"
        )
        if self._trunk_body_id < 0 or freejoint_id < 0:
            raise ValueError("model must declare trunk_base and trunk_base_freejoint")
        self._free_qpos_address = int(self._model.jnt_qposadr[freejoint_id])
        self._free_qvel_address = int(self._model.jnt_dofadr[freejoint_id])
        self._gyro_sensor_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel"
        )

    def _reset_model_locked(self) -> None:
        mujoco.mj_resetData(self._model, self._data)
        self._data.qpos[self._joint_qpos_indices] = DEFAULT_JOINT_POSE
        self._data.ctrl[self._actuator_indices] = DEFAULT_JOINT_POSE
        mujoco.mj_forward(self._model, self._data)

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
        if action.executionMode == "DISCRETE":
            if action.actionCode not in _COMPLETION_EVALUATORS:
                raise ValueError("runtime action has no internal completion evaluator")
        elif action.actionCode not in _CONTINUOUS_ACTIONS:
            raise ValueError("runtime action has no typed continuous command mapping")
        self._command_for(action.actionCode, request.parameters, action)
        seed = request.scenario.get("seed", 0)
        if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32:
            raise ValueError("scenario seed must be an unsigned 32-bit integer")

    def start(
        self, action: ActionDefinition, request: TaskCreateRequest
    ) -> RuntimeHandle:
        self.validate(action, request)
        with self._lock:
            if self._active_handle is not None:
                raise RuntimeError("runtime already has an active task")
            if self._fatal_reason is not None:
                raise RuntimeError("runtime requires restart after a safety fault")
            self._reset_model_locked()
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
            self._start_base_position = self._base_position()
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
            evaluator = _COMPLETION_EVALUATORS.get(self._active_action.actionCode)
            if evaluator is not None and evaluator(self):
                self._terminal_state = "SUCCEEDED"
                self._terminal_reason = "TASK_COMPLETE"
                self._stop_event.set()
                return RuntimeSample(
                    running=False,
                    terminalState="SUCCEEDED",
                    metrics=self._action_metrics_locked(),
                    stopReason="TASK_COMPLETE",
                )
            return RuntimeSample(running=True, metrics=self._action_metrics_locked())

    def safe_stop(self, handle: RuntimeHandle | None, reason: str) -> RuntimeEvidence:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        with self._lock:
            if handle is not None:
                self._require_handle(handle)
            if self._active_handle is None:
                return RuntimeEvidence(stopReason=reason)
            metrics = self._evidence_metrics_locked()
            self._hold_current_position_locked()
            self._active_handle = None
            self._active_action = None
            self._active_request = None
            self._active_policy = None
            self._active_session = None
            self._thread = None
            self._limp = self._fatal_reason is not None
            return RuntimeEvidence(metrics=metrics, stopReason=reason)

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
            joints = self._finite_tuple(self._data.qpos[self._joint_qpos_indices], 14)
            joint_velocities = self._finite_tuple(
                self._data.qvel[self._joint_qvel_indices], 14
            )
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
                loopFrequencyHz=50.0,
                fallen=self._fallen,
                limp=self._limp,
                health={
                    "ready": ready,
                    "healthy": ready,
                    "reasonCodes": reason_codes,
                },
            )

    def _governed_loop(self) -> None:
        deadline = self._clock()
        while not self._stop_event.is_set():
            self._control_step()
            deadline += _CONTROL_PERIOD_S
            remaining = deadline - self._clock()
            if remaining > 0.0:
                self._stop_event.wait(remaining)

    def _control_step(self) -> None:
        with self._lock:
            if self._active_handle is None or self._terminal_state is not None:
                return
            try:
                self._require_finite_simulation_state()
                state = DeploymentState(
                    base_angular_velocity_radps=self._base_angular_velocity(),
                    base_orientation_wxyz=self._base_quaternion_wxyz(),
                    joint_positions_rad=self._data.qpos[self._joint_qpos_indices],
                    joint_velocities_radps=self._data.qvel[self._joint_qvel_indices],
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
                self._data.ctrl[self._actuator_indices] = target
                self._policy_target = target.copy()
                self._previous_action = policy_action.copy()
                for _ in range(self._steps_per_control):
                    mujoco.mj_step(self._model, self._data)
                self._step_count += 1
                self._update_action_command_locked()
                self._update_safety_metrics_locked(policy_action)
                self._require_finite_simulation_state()
                self._check_joint_limits()
                self._check_fall_locked()
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
        command = DeploymentCommand.zero()
        if action_code == "SIT":
            command = DeploymentCommand(
                twist=np.array([1.0, 0.0, 0.0], dtype=np.float32),
                head_pose=np.zeros(4, dtype=np.float32),
                body_pose=np.zeros(6, dtype=np.float32),
            )
        return command, command, None

    def _update_action_command_locked(self) -> None:
        if self._active_action is None:
            return
        if self._active_action.actionCode == "GROUND_PICK":
            phase = min(0.7, self._duration_locked() / 4.0)
            self._command = DeploymentCommand(
                twist=np.array(
                    [
                        math.cos(2.0 * math.pi * phase),
                        math.sin(2.0 * math.pi * phase),
                        0.0,
                    ],
                    dtype=np.float32,
                ),
                head_pose=np.zeros(4, dtype=np.float32),
                body_pose=np.zeros(6, dtype=np.float32),
            )

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

    def _fail_locked(self, reason: str, *, fallen: bool = False) -> None:
        self._fatal_reason = reason
        self._terminal_state = "FAILED"
        self._terminal_reason = "FALLEN" if fallen else reason
        self._fallen |= fallen
        self._limp = True
        self._hold_current_position_locked()
        self._stop_event.set()

    def _hold_current_position_locked(self) -> None:
        current = self._finite_array(self._data.qpos[self._joint_qpos_indices], 14)
        self._data.ctrl[self._actuator_indices] = current

    def _base_position(self) -> NDArray[np.float64]:
        return self._data.xpos[self._trunk_body_id].copy()

    def _base_quaternion_wxyz(self) -> NDArray[np.float64]:
        return self._data.xquat[self._trunk_body_id].copy()

    def _base_angular_velocity(self) -> NDArray[np.float64]:
        if self._gyro_sensor_id >= 0:
            address = int(self._model.sensor_adr[self._gyro_sensor_id])
            return self._data.sensordata[address : address + 3].copy()
        return self._data.qvel[
            self._free_qvel_address + 3 : self._free_qvel_address + 6
        ].copy()

    def _duration_locked(self) -> float:
        return max(0.0, float(self._data.time) - self._start_sim_time)

    def _action_metrics_locked(self) -> dict[str, int | float | bool | str]:
        assert self._active_action is not None
        base_position = self._base_position()
        gravity = project_gravity_wxyz(self._base_quaternion_wxyz())
        final_tilt = math.acos(float(np.clip(-gravity[2], -1.0, 1.0)))
        return {
            "actionCode": self._active_action.actionCode,
            "baseTravelM": round(
                float(np.linalg.norm(base_position - self._start_base_position)), 6
            ),
            "durationS": round(self._duration_locked(), 6),
            "fallen": self._fallen,
            "finalBaseHeightM": round(float(base_position[2]), 6),
            "finalTiltRad": round(final_tilt, 6),
            "maxAbsAction": round(self._max_abs_action, 6),
            "maxTiltRad": round(self._max_tilt_rad, 6),
            "minBaseHeightM": round(self._min_base_height_m, 6),
            "steps": self._step_count,
        }

    def _evidence_metrics_locked(self) -> dict[str, int | float | bool | str | None]:
        assert self._active_policy is not None
        assert self._active_request is not None
        return {
            **self._action_metrics_locked(),
            "bundleDigest": self._bundle.bundleDigest,
            "onnxDigest": self._active_policy.digest,
            "mjcfDigest": self._bundle.model.digest,
            "sourceCommit": self._bundle.sourceCommit,
            "checkpoint": self._active_policy.checkpoint,
            "runIdentity": self._active_policy.experimentRef,
            "terrain": str(self._active_request.scenario.get("terrain", "")),
            "seed": int(self._active_request.scenario.get("seed", 0)),
        }

    @staticmethod
    def _finite_array(values: Any, length: int) -> NDArray[np.float64]:
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (length,):
            return np.zeros(length, dtype=np.float64)
        return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)

    @classmethod
    def _finite_tuple(cls, values: Any, length: int) -> tuple[float, ...]:
        return tuple(float(value) for value in cls._finite_array(values, length))


def _elapsed(runtime: MicroduckMujocoRuntime, minimum_s: float) -> bool:
    return runtime._duration_locked() + 1e-9 >= minimum_s


def _upright(runtime: MicroduckMujocoRuntime, *, minimum_height_m: float) -> bool:
    gravity = project_gravity_wxyz(runtime._base_quaternion_wxyz())
    tilt = math.acos(float(np.clip(-gravity[2], -1.0, 1.0)))
    return (
        tilt <= math.radians(35.0) and runtime._base_position()[2] >= minimum_height_m
    )


def _stand_complete(runtime: MicroduckMujocoRuntime) -> bool:
    return _elapsed(runtime, 2.0) and _upright(runtime, minimum_height_m=0.08)


def _roller_stand_complete(runtime: MicroduckMujocoRuntime) -> bool:
    return _elapsed(runtime, 2.0) and _upright(runtime, minimum_height_m=0.10)


def _sit_complete(runtime: MicroduckMujocoRuntime) -> bool:
    gravity = project_gravity_wxyz(runtime._base_quaternion_wxyz())
    tilt = math.acos(float(np.clip(-gravity[2], -1.0, 1.0)))
    return (
        _elapsed(runtime, 2.0)
        and tilt <= math.radians(45.0)
        and runtime._base_position()[2] <= 0.10
    )


def _roller_crouch_complete(runtime: MicroduckMujocoRuntime) -> bool:
    return _elapsed(runtime, 2.0) and runtime._base_position()[2] <= 0.13


def _ground_pick_complete(runtime: MicroduckMujocoRuntime) -> bool:
    return _elapsed(runtime, 2.8) and _upright(runtime, minimum_height_m=0.08)


def _kick_complete(runtime: MicroduckMujocoRuntime) -> bool:
    return _elapsed(runtime, 3.0) and _upright(runtime, minimum_height_m=0.08)


def _dynamic_complete(runtime: MicroduckMujocoRuntime) -> bool:
    return _elapsed(runtime, 2.0) and _upright(runtime, minimum_height_m=0.08)


def _roller_slope_complete(runtime: MicroduckMujocoRuntime) -> bool:
    distance = float(
        np.linalg.norm(runtime._base_position() - runtime._start_base_position)
    )
    return _elapsed(runtime, 3.0) and distance >= 0.10


# Only this code-owned table can select executable completion logic. Manifest fields
# remain data and are never resolved as Python names or expressions.
_COMPLETION_EVALUATORS: dict[str, Callable[[MicroduckMujocoRuntime], bool]] = {
    "ROLLER_SLOPE": _roller_slope_complete,
    "STAND_UP": _stand_complete,
    "SIT": _sit_complete,
    "STAND": _stand_complete,
    "GROUND_PICK": _ground_pick_complete,
    "KICK_LEFT": _kick_complete,
    "KICK_RIGHT": _kick_complete,
    "ROULADE": _dynamic_complete,
    "ROLLER_CROUCH": _roller_crouch_complete,
    "ROLLER_STAND_UP": _roller_stand_complete,
    "SPIN": _dynamic_complete,
}

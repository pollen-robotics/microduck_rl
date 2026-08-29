"""Capture and replay exact walker states for the full-height stair specialist."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

BANK_SCHEMA_VERSION = 1
STANDARD_RISER_HEIGHT_M = 0.170
STANDARD_TREAD_DEPTH_M = 0.280
STANDARD_NUM_STEPS = 5

_TWIST_FIELDS = (
    "time_left",
    "command_counter",
    "vel_command_b",
    "vel_command_w",
    "heading_target",
    "heading_error",
    "is_heading_env",
    "is_standing_env",
    "is_world_env",
    "is_forward_env",
)
_POSE_FIELDS = ("_command", "time_left", "command_counter")
_CONTACT_FIELDS = (
    "current_air_time",
    "last_air_time",
    "current_contact_time",
    "last_contact_time",
)
_ACTUATOR_FIELDS = (
    "_prev_motor_torque",
    "kp_scale",
    "kd_scale",
    "friction_scale",
    "vin_tensor",
    "vin_drop_gain",
)


def _cpu_rows(value: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
    return value[env_ids].detach().cpu().clone()


def _capture_delay_buffer(delay: Any, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    circular = delay._buffer
    if not circular.is_initialized:
        raise RuntimeError("Cannot capture an uninitialized temporal buffer")
    return {
        "history": _cpu_rows(circular.buffer, env_ids),
        "num_pushes": _cpu_rows(circular._num_pushes, env_ids),
        "current_lags": _cpu_rows(delay._current_lags, env_ids),
        "step_count": _cpu_rows(delay._step_count, env_ids),
        "phase_offsets": _cpu_rows(delay._phase_offsets, env_ids),
    }


def _capture_commands(env: Any, env_ids: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
    result: dict[str, dict[str, torch.Tensor]] = {}
    for name, fields in (
        ("twist", _TWIST_FIELDS),
        ("head_pose", _POSE_FIELDS),
        ("body_pose", _POSE_FIELDS),
    ):
        try:
            term = env.command_manager.get_term(name)
        except KeyError:
            # Episodic manufacturer tasks zero-pad the shared observation tail
            # instead of registering head/body commands. Keep those target
            # stair cues untouched when this state is later transplanted.
            continue
        result[name] = {
            field: _cpu_rows(getattr(term, field), env_ids) for field in fields
        }
    return result


def capture_walk_state_rows(env: Any, env_ids: torch.Tensor) -> dict[str, Any]:
    """Capture safely restorable manufacturer-walker state for selected worlds."""

    env_ids = env_ids.to(device=env.device, dtype=torch.long)
    robot = env.scene["robot"]
    origins = env.scene.terrain.env_origins[env_ids]
    root_qpos = env.sim.data.qpos[env_ids][
        :, robot.indexing.free_joint_q_adr.to(torch.long)
    ].detach().clone()
    root_qpos[:, :3] -= origins
    root_qvel = env.sim.data.qvel[env_ids][
        :, robot.indexing.free_joint_v_adr.to(torch.long)
    ]
    joint_qpos = env.sim.data.qpos[env_ids][
        :, robot.indexing.joint_q_adr.to(torch.long)
    ]
    joint_qvel = env.sim.data.qvel[env_ids][
        :, robot.indexing.joint_v_adr.to(torch.long)
    ]

    action_term = env.action_manager.get_term("joint_pos")
    if len(robot.actuators) != 1:
        raise RuntimeError("Walker state replay requires exactly one robot actuator")
    actuator = robot.actuators[0]
    actuator_state = {
        field: _cpu_rows(getattr(actuator, field), env_ids)
        for field in _ACTUATOR_FIELDS
    }
    actuator_state.update(
        {
            "dof_frictionloss": _cpu_rows(
                env.sim.model.dof_frictionloss[:, actuator._dof_ids], env_ids
            ),
            "dof_damping": _cpu_rows(
                env.sim.model.dof_damping[:, actuator._dof_ids], env_ids
            ),
            "delay": _capture_delay_buffer(actuator._delay_buffer, env_ids),
        }
    )

    observation_delays: dict[str, dict[str, torch.Tensor]] = {}
    for group, terms in env.observation_manager._group_obs_term_delay_buffer.items():
        for term_name, delay in terms.items():
            observation_delays[f"{group}/{term_name}"] = _capture_delay_buffer(
                delay, env_ids
            )

    contact = env.scene.sensors["feet_ground_contact"]._air_time_state
    if contact is None:
        raise RuntimeError("feet_ground_contact does not track contact timing")

    return {
        "root_qpos_local": root_qpos.cpu(),
        "root_qvel": root_qvel.detach().cpu().clone(),
        "joint_qpos": joint_qpos.detach().cpu().clone(),
        "joint_qvel": joint_qvel.detach().cpu().clone(),
        "action": _cpu_rows(env.action_manager._action, env_ids),
        "previous_action": _cpu_rows(env.action_manager._prev_action, env_ids),
        "previous_previous_action": _cpu_rows(
            env.action_manager._prev_prev_action, env_ids
        ),
        "raw_action": _cpu_rows(action_term._raw_actions, env_ids),
        "processed_action": _cpu_rows(action_term._processed_actions, env_ids),
        "joint_pos_target": _cpu_rows(robot.data.joint_pos_target, env_ids),
        "joint_vel_target": _cpu_rows(robot.data.joint_vel_target, env_ids),
        "joint_effort_target": _cpu_rows(robot.data.joint_effort_target, env_ids),
        "encoder_bias": _cpu_rows(robot.data.encoder_bias, env_ids),
        "imu_misalign_quat": _cpu_rows(env._imu_misalign_quat, env_ids),
        "actuator": actuator_state,
        "commands": _capture_commands(env, env_ids),
        "observation_delays": observation_delays,
        "feet_contact": {
            field: _cpu_rows(getattr(contact, field), env_ids)
            for field in _CONTACT_FIELDS
        },
    }


def concatenate_walk_state_rows(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Concatenate a list of nested state chunks along their environment axis."""

    if not chunks:
        raise ValueError("At least one walker-state chunk is required")

    def merge(values: list[Any]) -> Any:
        first = values[0]
        if isinstance(first, dict):
            keys = set(first)
            if any(set(value) != keys for value in values[1:]):
                raise ValueError("Walker-state chunks have inconsistent fields")
            return {key: merge([value[key] for value in values]) for key in first}
        if not all(isinstance(value, torch.Tensor) for value in values):
            raise TypeError("Walker-state leaves must be tensors")
        return torch.cat(values, dim=0)

    return merge(chunks)


def walk_state_count(states: dict[str, Any]) -> int:
    return int(states["root_qpos_local"].shape[0])


def _restore_circular_rows(
    circular: Any,
    env_ids: torch.Tensor,
    history: torch.Tensor,
    num_pushes: torch.Tensor,
) -> None:
    """Restore chronological rows without changing the shared circular pointer."""

    history = history.to(circular._device)
    num_pushes = num_pushes.to(circular._device)
    max_len = circular._max_len
    if history.shape[1] != max_len:
        raise ValueError(
            f"Temporal history length {history.shape[1]} does not match {max_len}"
        )
    if circular._buffer is None:
        circular._buffer = torch.zeros(
            (max_len, circular._batch_size, *history.shape[2:]),
            dtype=history.dtype,
            device=circular._device,
        )
    start = (circular._pointer + 1) % max_len
    for chronological_index in range(max_len):
        raw_index = (start + chronological_index) % max_len
        circular._buffer[raw_index, env_ids] = history[:, chronological_index]
    circular._num_pushes[env_ids] = num_pushes


def _preappend_history(history: torch.Tensor) -> torch.Tensor:
    """Reconstruct the buffer immediately before its captured newest append."""

    if history.shape[1] == 1:
        return history.clone()
    return torch.cat((history[:, :1], history[:, :-1]), dim=1)


def _restore_delay_buffer(
    delay: Any,
    env_ids: torch.Tensor,
    saved: dict[str, torch.Tensor],
    *,
    before_next_append: bool = False,
) -> None:
    history = saved["history"]
    pushes = saved["num_pushes"]
    step_count = saved["step_count"]
    if before_next_append:
        history = _preappend_history(history)
        pushes = torch.clamp(pushes - 1, min=1)
        step_count = torch.clamp(step_count - 1, min=0)
    _restore_circular_rows(delay._buffer, env_ids, history, pushes)
    delay._current_lags[env_ids] = saved["current_lags"].to(delay.device)
    delay._step_count[env_ids] = step_count.to(delay.device)
    delay._phase_offsets[env_ids] = saved["phase_offsets"].to(delay.device)


def _select_rows(value: Any, rows: torch.Tensor, device: str) -> Any:
    if isinstance(value, dict):
        return {key: _select_rows(child, rows, device) for key, child in value.items()}
    return value[rows.cpu()].to(device)


def _canonicalize_root_heading(
    root_qpos: torch.Tensor, root_qvel: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate a saved free-joint state so its travelled heading is +x."""
    root_qpos = root_qpos.clone()
    root_qvel = root_qvel.clone()
    quat = root_qpos[:, 3:7]
    w, x, y, z = (component.clone() for component in quat.unbind(dim=-1))
    # Euler yaw is ambiguous during a forward roll near 90 degrees of pitch.
    # The source starts at the terrain origin, so its horizontal displacement
    # is the stable path heading across every captured roll phase.
    heading = torch.atan2(root_qpos[:, 1], root_qpos[:, 0])
    half = -0.5 * heading
    yaw_w = torch.cos(half)
    yaw_z = torch.sin(half)

    # Left-multiply q by the inverse world-yaw quaternion [cy, 0, 0, sy].
    root_qpos[:, 3] = yaw_w * w - yaw_z * z
    root_qpos[:, 4] = yaw_w * x - yaw_z * y
    root_qpos[:, 5] = yaw_w * y + yaw_z * x
    root_qpos[:, 6] = yaw_w * z + yaw_z * w

    cos_yaw = torch.cos(heading)
    sin_yaw = torch.sin(heading)
    vx = root_qvel[:, 0].clone()
    vy = root_qvel[:, 1].clone()
    root_qvel[:, 0] = cos_yaw * vx + sin_yaw * vy
    root_qvel[:, 1] = -sin_yaw * vx + cos_yaw * vy
    return root_qpos, root_qvel


def load_walk_state_bank(path: str | Path) -> dict[str, Any]:
    bank_path = Path(path).expanduser().resolve()
    if not bank_path.is_file():
        raise FileNotFoundError(f"Walker state bank not found: {bank_path}")
    bank = torch.load(bank_path, map_location="cpu", weights_only=False)
    if not isinstance(bank, dict) or bank.get("schema_version") != BANK_SCHEMA_VERSION:
        raise ValueError(f"Unsupported walker state bank schema in {bank_path}")
    metadata = bank.get("metadata", {})
    expected = {
        "riser_height_m": STANDARD_RISER_HEIGHT_M,
        "tread_depth_m": STANDARD_TREAD_DEPTH_M,
        "num_steps": STANDARD_NUM_STEPS,
    }
    for key, expected_value in expected.items():
        if metadata.get(key) != expected_value:
            raise ValueError(
                f"Walker state bank {key}={metadata.get(key)!r}, expected {expected_value!r}"
            )
    states = bank.get("states")
    if not isinstance(states, dict) or walk_state_count(states) < 1:
        raise ValueError("Walker state bank contains no states")
    bank["path"] = str(bank_path)
    return bank


def eligible_walk_state_rows(
    states: dict[str, Any],
    *,
    source_episode_step_range: tuple[int, int] | None = None,
    min_forward_speed: float | None = None,
    min_vertical_speed: float | None = None,
    min_root_height: float | None = None,
    min_vault_momentum: float | None = None,
    vault_lever_arm: float = 0.06,
    max_abs_local_y: float | None = None,
    max_abs_lateral_speed: float | None = None,
    max_abs_yaw_rate: float | None = None,
) -> torch.Tensor:
    """Return source rows that contain useful dynamic handoff phases."""

    count = walk_state_count(states)
    eligible = torch.ones(count, dtype=torch.bool)
    if source_episode_step_range is not None:
        if "source_episode_step" not in states:
            raise ValueError(
                "source_episode_step_range requires source_episode_step in the bank"
            )
        low, high = source_episode_step_range
        step = states["source_episode_step"]
        eligible &= (step >= low) & (step <= high)
    if min_forward_speed is not None:
        eligible &= states["root_qvel"][:, 0] >= min_forward_speed
    if min_vertical_speed is not None:
        eligible &= states["root_qvel"][:, 2] >= min_vertical_speed
    if min_root_height is not None:
        eligible &= states["root_qpos_local"][:, 2] >= min_root_height
    if min_vault_momentum is not None:
        velocity = states["root_qvel"]
        # A useful tread-contact handoff can carry translational momentum or
        # convert forward somersault rate into linear speed about a head/shell
        # pivot. Negative components do not help the robot vault the lip.
        vault_momentum = (
            torch.clamp(velocity[:, 0], min=0.0)
            + torch.clamp(velocity[:, 2], min=0.0)
            + vault_lever_arm * torch.clamp(velocity[:, 4], min=0.0)
        )
        eligible &= vault_momentum >= min_vault_momentum
    if max_abs_local_y is not None:
        eligible &= torch.abs(states["root_qpos_local"][:, 1]) <= max_abs_local_y
    if max_abs_lateral_speed is not None:
        eligible &= torch.abs(states["root_qvel"][:, 1]) <= max_abs_lateral_speed
    if max_abs_yaw_rate is not None:
        eligible &= torch.abs(states["root_qvel"][:, 5]) <= max_abs_yaw_rate
    rows = torch.nonzero(eligible, as_tuple=False).squeeze(-1)
    if len(rows) == 0:
        raise ValueError("Walker-state phase filters rejected every bank row")
    return rows


def phase_balanced_row_buckets(
    states: dict[str, Any],
    eligible_rows: torch.Tensor,
    *,
    source_episode_step_range: tuple[int, int],
    bucket_count: int = 4,
) -> tuple[torch.Tensor, ...]:
    """Split eligible reference states into equal temporal phase buckets.

    Reference-state initialization is most useful when every rollout sees a
    comparable amount of preload, contact, apex, and release motion. The raw
    bank is not uniformly distributed in time, so sampling eligible rows
    uniformly can almost eliminate the late-release states. Buckets are based
    on the source episode step, while the rows inside each bucket remain
    stochastic.
    """
    if bucket_count < 2:
        raise ValueError("Phase-balanced sampling requires at least two buckets")
    if len(eligible_rows) == 0:
        raise ValueError("Phase-balanced sampling requires eligible rows")
    if "source_episode_step" not in states:
        raise ValueError(
            "Phase-balanced sampling requires source_episode_step in the bank"
        )
    low, high = source_episode_step_range
    if high < low:
        raise ValueError("Source episode step range must be ascending")
    source_steps = states["source_episode_step"][eligible_rows]
    span = max(high - low + 1, 1)
    bucket_ids = torch.div(
        (source_steps - low).clamp(min=0, max=span - 1) * bucket_count,
        span,
        rounding_mode="floor",
    ).clamp(max=bucket_count - 1)
    buckets = tuple(
        eligible_rows[bucket_ids == bucket_index]
        for bucket_index in range(bucket_count)
    )
    if any(len(bucket) == 0 for bucket in buckets):
        counts = [len(bucket) for bucket in buckets]
        raise ValueError(
            "Phase-balanced sampling produced an empty bucket: "
            f"counts={counts}, range={source_episode_step_range}"
        )
    return buckets


def phase_aligned_local_x(
    source_steps: torch.Tensor,
    *,
    source_episode_step_range: tuple[int, int],
    local_x_range: tuple[float, float],
) -> torch.Tensor:
    """Map reference time monotonically onto the stair approach corridor."""

    step_low, step_high = source_episode_step_range
    x_low, x_high = local_x_range
    if step_high <= step_low:
        raise ValueError("Phase-aligned replay requires an ascending step range")
    if x_high <= x_low:
        raise ValueError("Phase-aligned replay requires an ascending x range")
    phase = torch.clamp(
        (source_steps.to(torch.float32) - step_low) / (step_high - step_low),
        min=0.0,
        max=1.0,
    )
    return x_low + phase * (x_high - x_low)


class WalkerStateBankReset:
    """Replace assisted mode 3 with a real frozen-walker handoff state."""

    def __init__(self, cfg: Any, env: Any):
        self._env = env
        self._bank = load_walk_state_bank(cfg.params["bank_path"])
        self._states = self._bank["states"]
        self._pending_env_ids: torch.Tensor | None = None
        self._pending_rows: torch.Tensor | None = None
        self._canonicalize_heading = bool(
            cfg.params.get("canonicalize_heading", False)
        )
        self._local_x_range = cfg.params.get("local_x_range")
        self._local_y_range = cfg.params.get("local_y_range")
        self._phase_aligned_local_x_range = cfg.params.get(
            "phase_aligned_local_x_range"
        )
        self._phase_aligned_x_jitter = float(
            cfg.params.get("phase_aligned_x_jitter", 0.0)
        )
        self._zero_missing_pose_commands = bool(
            cfg.params.get("zero_missing_pose_commands", False)
        )
        self._phase_balanced = bool(cfg.params.get("phase_balanced", False))
        self._phase_bucket_count = int(cfg.params.get("phase_bucket_count", 4))
        self._source_episode_step_range = cfg.params.get(
            "source_episode_step_range"
        )
        self._eligible_rows = eligible_walk_state_rows(
            self._states,
            source_episode_step_range=self._source_episode_step_range,
            min_forward_speed=cfg.params.get("min_forward_speed"),
            min_vertical_speed=cfg.params.get("min_vertical_speed"),
            min_root_height=cfg.params.get("min_root_height"),
            min_vault_momentum=cfg.params.get("min_vault_momentum"),
            vault_lever_arm=cfg.params.get("vault_lever_arm", 0.06),
            max_abs_local_y=cfg.params.get("max_abs_local_y"),
            max_abs_lateral_speed=cfg.params.get("max_abs_lateral_speed"),
            max_abs_yaw_rate=cfg.params.get("max_abs_yaw_rate"),
        )
        self._phase_rows: tuple[torch.Tensor, ...] | None = None
        if self._phase_balanced:
            source_range = self._source_episode_step_range
            if source_range is None:
                raise ValueError(
                    "Phase-balanced walker replay requires source_episode_step_range"
                )
            self._phase_rows = phase_balanced_row_buckets(
                self._states,
                self._eligible_rows,
                source_episode_step_range=source_range,
                bucket_count=self._phase_bucket_count,
            )
        if self._phase_aligned_local_x_range is not None:
            if self._local_x_range is not None:
                raise ValueError(
                    "Use local_x_range or phase_aligned_local_x_range, not both"
                )
            if self._source_episode_step_range is None:
                raise ValueError(
                    "Phase-aligned replay requires source_episode_step_range"
                )
            if self._phase_aligned_x_jitter < 0.0:
                raise ValueError("Phase-aligned x jitter must be nonnegative")

        robot = env.scene["robot"]
        saved_joint_names = self._bank["metadata"].get("joint_names")
        if saved_joint_names != list(robot.joint_names):
            raise ValueError("Walker state bank joint ordering does not match the robot")

    def __call__(
        self,
        env: Any,
        env_ids: torch.Tensor,
        bank_path: str,
        canonicalize_heading: bool = False,
        local_x_range: tuple[float, float] | None = None,
        local_y_range: tuple[float, float] | None = None,
        zero_missing_pose_commands: bool = False,
        source_episode_step_range: tuple[int, int] | None = None,
        min_forward_speed: float | None = None,
        min_vertical_speed: float | None = None,
        min_root_height: float | None = None,
        min_vault_momentum: float | None = None,
        vault_lever_arm: float = 0.06,
        max_abs_local_y: float | None = None,
        max_abs_lateral_speed: float | None = None,
        max_abs_yaw_rate: float | None = None,
        phase_balanced: bool = False,
        phase_bucket_count: int = 4,
        phase_aligned_local_x_range: tuple[float, float] | None = None,
        phase_aligned_x_jitter: float = 0.0,
    ) -> None:
        del (
            bank_path,
            canonicalize_heading,
            local_x_range,
            local_y_range,
            zero_missing_pose_commands,
            source_episode_step_range,
            min_forward_speed,
            min_vertical_speed,
            min_root_height,
            min_vault_momentum,
            vault_lever_arm,
            max_abs_local_y,
            max_abs_lateral_speed,
            max_abs_yaw_rate,
            phase_balanced,
            phase_bucket_count,
            phase_aligned_local_x_range,
            phase_aligned_x_jitter,
        )
        mode = getattr(env, "_stair_assisted_reset_mode", None)
        if mode is None:
            raise RuntimeError("Walker-state replay requires assisted reset mode tracking")
        env_ids = env_ids.to(env.device, dtype=torch.long)
        selected_ids = env_ids[mode[env_ids] == 3]
        if not hasattr(env, "_stair_walker_bank_row"):
            env._stair_walker_bank_row = torch.full(
                (env.num_envs,), -1, dtype=torch.long, device=env.device
            )
            env._stair_walker_bank_source_step = torch.full_like(
                env._stair_walker_bank_row, -1
            )
        env._stair_walker_bank_row[env_ids] = -1
        env._stair_walker_bank_source_step[env_ids] = -1
        if len(selected_ids) == 0:
            self._pending_env_ids = None
            self._pending_rows = None
            return

        if self._phase_rows is None:
            row_indices = torch.randint(
                0,
                len(self._eligible_rows),
                (len(selected_ids),),
                device=env.device,
            )
            rows = self._eligible_rows.to(env.device)[row_indices]
        else:
            bucket_indices = torch.randint(
                0,
                len(self._phase_rows),
                (len(selected_ids),),
                device=env.device,
            )
            rows = torch.empty(len(selected_ids), dtype=torch.long, device=env.device)
            for bucket_index, bucket_rows in enumerate(self._phase_rows):
                bucket_mask = bucket_indices == bucket_index
                if torch.any(bucket_mask):
                    choices = torch.randint(
                        0,
                        len(bucket_rows),
                        (int(bucket_mask.sum().item()),),
                        device=env.device,
                    )
                    rows[bucket_mask] = bucket_rows.to(env.device)[choices]
        env._stair_walker_bank_row[selected_ids] = rows
        if "source_episode_step" in self._states:
            env._stair_walker_bank_source_step[selected_ids] = self._states[
                "source_episode_step"
            ][rows.cpu()].to(env.device)
        saved = _select_rows(self._states, rows, env.device)
        robot = env.scene["robot"]
        root_qpos = saved["root_qpos_local"].clone()
        root_qvel = saved["root_qvel"].clone()
        if self._canonicalize_heading:
            root_qpos, root_qvel = _canonicalize_root_heading(root_qpos, root_qvel)
        if self._phase_aligned_local_x_range is not None:
            source_range = self._source_episode_step_range
            assert source_range is not None
            source_steps = self._states["source_episode_step"][rows.cpu()].to(
                env.device
            )
            root_qpos[:, 0] = phase_aligned_local_x(
                source_steps,
                source_episode_step_range=tuple(source_range),
                local_x_range=tuple(self._phase_aligned_local_x_range),
            ).to(root_qpos.dtype)
            if self._phase_aligned_x_jitter > 0.0:
                root_qpos[:, 0] += torch.empty_like(root_qpos[:, 0]).uniform_(
                    -self._phase_aligned_x_jitter,
                    self._phase_aligned_x_jitter,
                )
        elif self._local_x_range is not None:
            low, high = self._local_x_range
            root_qpos[:, 0].uniform_(low, high)
        if self._local_y_range is not None:
            low, high = self._local_y_range
            root_qpos[:, 1].uniform_(low, high)
        root_qpos[:, :3] += env.scene.terrain.env_origins[selected_ids]
        env.sim.data.qpos[selected_ids[:, None], robot.indexing.free_joint_q_adr] = root_qpos
        env.sim.data.qvel[selected_ids[:, None], robot.indexing.free_joint_v_adr] = root_qvel
        env.sim.data.qpos[selected_ids[:, None], robot.indexing.joint_q_adr] = saved[
            "joint_qpos"
        ]
        env.sim.data.qvel[selected_ids[:, None], robot.indexing.joint_v_adr] = saved[
            "joint_qvel"
        ]

        robot.data.joint_pos_target[selected_ids] = saved["joint_pos_target"]
        robot.data.joint_vel_target[selected_ids] = saved["joint_vel_target"]
        robot.data.joint_effort_target[selected_ids] = saved["joint_effort_target"]
        robot.data.encoder_bias[selected_ids] = saved["encoder_bias"]
        env._imu_misalign_quat[selected_ids] = saved["imu_misalign_quat"]

        actuator = robot.actuators[0]
        for field in _ACTUATOR_FIELDS:
            getattr(actuator, field)[selected_ids] = saved["actuator"][field]
        env.sim.model.dof_frictionloss[
            selected_ids[:, None], actuator._dof_ids
        ] = saved["actuator"]["dof_frictionloss"]
        env.sim.model.dof_damping[
            selected_ids[:, None], actuator._dof_ids
        ] = saved["actuator"]["dof_damping"]
        _restore_delay_buffer(
            actuator._delay_buffer, selected_ids, saved["actuator"]["delay"]
        )

        contact = env.scene.sensors["feet_ground_contact"]._air_time_state
        if contact is None:
            raise RuntimeError("feet_ground_contact does not track contact timing")
        for field in _CONTACT_FIELDS:
            getattr(contact, field)[selected_ids] = saved["feet_contact"][field]
        contact.last_time[selected_ids] = env.sim.data.time[selected_ids]

        self._pending_env_ids = selected_ids
        self._pending_rows = rows

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        del env_ids
        if self._pending_env_ids is None or self._pending_rows is None:
            return
        ids = self._pending_env_ids
        saved = _select_rows(self._states, self._pending_rows, self._env.device)
        manager = self._env.action_manager
        manager._action[ids] = saved["action"]
        manager._prev_action[ids] = saved["previous_action"]
        manager._prev_prev_action[ids] = saved["previous_previous_action"]
        action_term = manager.get_term("joint_pos")
        action_term._raw_actions[ids] = saved["raw_action"]
        action_term._processed_actions[ids] = saved["processed_action"]

        command_fields = {
            "twist": _TWIST_FIELDS,
            "head_pose": _POSE_FIELDS,
            "body_pose": _POSE_FIELDS,
        }
        for name, saved_fields in saved["commands"].items():
            term = self._env.command_manager.get_term(name)
            for field in command_fields[name]:
                value = saved_fields[field]
                if field == "time_left":
                    value = value + self._env.step_dt
                getattr(term, field)[ids] = value
        if self._zero_missing_pose_commands:
            for name in ("head_pose", "body_pose"):
                if name not in saved["commands"]:
                    term = self._env.command_manager.get_term(name)
                    term._command[ids] = 0.0

        delays = self._env.observation_manager._group_obs_term_delay_buffer
        for key, delay_state in saved["observation_delays"].items():
            group, term_name = key.split("/", 1)
            _restore_delay_buffer(
                delays[group][term_name],
                ids,
                delay_state,
                before_next_append=True,
            )
        self._env.observation_manager._obs_buffer = None
        self._pending_env_ids = None
        self._pending_rows = None

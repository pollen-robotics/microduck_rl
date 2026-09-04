#!/usr/bin/env python3
"""Simple script to run ONNX policy inference in MuJoCo with rendering."""

import argparse
import csv
import math
import os
import pickle
import queue
import select
import sys
import termios
import threading
import time
import tty
import numpy as np
import mujoco
import mujoco.viewer
import onnxruntime as ort

MICRODUCK_XML = "src/mjlab_microduck/robot/microduck/scene.xml"
# MICRODUCK_XML = "src/mjlab_microduck/robot/microduck/scene_ramps.xml"
# MICRODUCK_XML = "src/mjlab_microduck/robot/microduck/scene_floor_objects.xml"
# MICRODUCK_XML = "src/mjlab_microduck/robot/microduck/scene_robot_walk.xml"
MICRODUCK_ROLLERS_XML = "src/mjlab_microduck/robot/microduck/scene_rollers.xml"
MICRODUCK_BALL_XML = "src/mjlab_microduck/robot/microduck/scene_ball.xml"

# BAM M6 defaults — MUST mirror `_BAM_ACTUATOR_KWARGS` in
# src/mjlab_microduck/robot/microduck_constants.py (the actuator every policy is
# trained against in warp). Not imported from there: that module drags in
# mjlab/torch/warp (~16 s import) for a CPU rehearsal script. Locked by
# tests/test_infer_policy_bam.py.
BAM_MOTOR_NAME = "xl330"
BAM_MODEL = "m6"
BAM_KP_FW = 200.0                 # microduck's preserved firmware stiffness
BAM_VIN_RANGE = (6.5, 8.2)        # per-env battery voltage DR in training
BAM_VIN_DROP_GAIN_RANGE = (0.0, 0.2)  # load-dependent sag V_drop = gain * sum|tau|
BAM_VIN_MIN = 6.0                 # floor on effective voltage after sag
BAM_MAX_CURRENT = None            # training runs WITHOUT the firmware current limiter
# Stiff joint-friction constraint, copied from bam.mjlab.BamActuator
# (stiff_frictionloss=True in training): warp has no noslip solver, so BAM
# stiffens frictionloss so a statically-held joint does not creep. Mirrored
# here so CPU and warp apply the same friction budget the same way.
BAM_STIFF_SOLREF_FRICTION = (-5.0e4, -2.0e2)
BAM_STIFF_SOLIMP_FRICTION = (0.99, 0.9999, 0.001, 0.5, 2.0)


def load_bam_model(kp_fw: float, vin: float, max_current):
    """Build the BAM M6 model + XL330 voltage-controlled actuator."""
    from bam.model import load_model
    bam_model = load_model(motor_name=BAM_MOTOR_NAME, model=BAM_MODEL)
    bam_model.actuator.kp = kp_fw
    bam_model.actuator.vin = vin
    bam_model.actuator.max_current = max_current if (max_current and max_current > 0) else None
    return bam_model


def load_mujoco_with_bam(xml_path: str, bam_model, timestep: float, vin_drop_gain, vin_min):
    """Load the scene and hand every non-passive actuator to bam.mujoco.MujocoController.

    Mirrors bam.mjlab.BamActuator.edit_spec (what warp does at training time):
    position actuators -> torque motors with the voltage-bounded forcerange,
    joint damping/frictionloss zeroed (BAM rewrites them every step), stiff
    friction constraint. Armature is set on the dofs by MujocoController.
    Returns (model, data, bam_ctrl, actuator_names).
    """
    from bam.mujoco import MujocoController

    kt = bam_model.kt.value
    R = bam_model.R.value
    force_limit = bam_model.actuator.vin * kt / R

    spec = mujoco.MjSpec.from_file(xml_path)
    names = []
    for act in spec.actuators:
        tgt = act.target
        tgt_name = tgt.name if hasattr(tgt, "name") else str(tgt)
        if tgt_name.startswith("passive_"):
            continue
        act.set_to_motor()
        act.forcelimited = True
        act.forcerange = (-force_limit, force_limit)
        act.ctrllimited = False
        act.gear = [1.0, 0, 0, 0, 0, 0]
        names.append(act.name)
        for joint in spec.joints:
            if joint.name == tgt_name:
                joint.damping = np.zeros((3, 1))  # MjsJoint expects a (3,1) array
                joint.frictionloss = 0.0
                joint.solref_friction = BAM_STIFF_SOLREF_FRICTION
                joint.solimp_friction = BAM_STIFF_SOLIMP_FRICTION
                break

    model = spec.compile()
    model.opt.timestep = timestep
    data = mujoco.MjData(model)
    bam_ctrl = MujocoController(bam_model, names, model, data,
                                vin_drop_gain=vin_drop_gain, vin_min=vin_min)
    print(f"BAM {BAM_MODEL} actuators on {len(names)} joints: kt={kt:.4f} R={R:.4f} "
          f"vin={bam_model.actuator.vin:.2f}V kp_fw={bam_model.actuator.kp:.0f} "
          f"vin_drop_gain={vin_drop_gain} vin_min={vin_min} "
          f"max_current={bam_model.actuator.max_current} forcerange=+/-{force_limit:.3f}Nm "
          f"armature={bam_model.actuator.get_extra_inertia():.2e}")
    return model, data, bam_ctrl, names


# Body pose command constants (must match training constants)
BODY_CMD_MAX_Z = 0.03              # ±30 mm
BODY_CMD_MAX_XY = 0.02             # ±20 mm
BODY_CMD_MAX_ANGLE = math.radians(30)  # ±30°

# Ball placement for kick behaviors (must match microduck_ball_kick_env_cfg's
# reset_ball_in_front_of_foot params: ball center in the robot's yaw frame).
BALL_OFFSET_X = 0.09
BALL_OFFSET_ABS_Y = 0.042
BALL_RADIUS = 0.035

# Default pose used by the policy (legs flexed, standing position)
# This is the reference pose that:
# - Actions are offsets from (motor_target = DEFAULT_POSE + action * scale)
# - Joint observations are relative to (obs_joint_pos = current_pos - DEFAULT_POSE)
# STAND2 pose (matches HOME_FRAME in microduck_constants.py): trunk shifted
# ~5mm forward so the CoM sits over the ankle axis. Leg pitch chain leaned
# forward vs the old pose: hip_pitch 30°→26.24°, ankle 30°→25.95°, knee 0°→0.28°.
DEFAULT_POSE = np.array([
    0.0,      # left_hip_yaw
    -0.0873,  # left_hip_roll
    -0.4579,  # left_hip_pitch
    -0.0049,  # left_knee
    0.4530,   # left_ankle
    0.3491,   # neck_pitch
    0.3491,   # head_pitch
    0.0,      # head_yaw
    0.0,      # head_roll
    0.0,      # right_hip_yaw
    0.0873,   # right_hip_roll
    0.4579,   # right_hip_pitch
    0.0049,   # right_knee
    -0.4530,  # right_ankle
], dtype=np.float32)


class TerminalInput:
    """Single-keypress reader on stdin (cbreak mode, background thread).

    Replaces the MuJoCo viewer key_callback: keypresses in the viewer window
    also fire the viewer's built-in visualization shortcuts (frames, labels,
    rendering toggles…), so commands are read from the terminal instead.
    Arrow keys arrive as ESC [ A/B/C/D escape sequences and are translated to
    symbolic names ("up"/"down"/"left"/"right"); letters are lowercased.
    cbreak (not raw) mode keeps ISIG enabled, so Ctrl+C still works.
    """

    _ARROWS = {"A": "up", "B": "down", "C": "right", "D": "left"}

    def __init__(self):
        self._queue = queue.Queue()
        self.enabled = sys.stdin.isatty()
        self._fd = sys.stdin.fileno() if self.enabled else -1
        self._old_attrs = None
        self._stop = threading.Event()

    def __enter__(self):
        if not self.enabled:
            print("WARNING: stdin is not a TTY — keyboard control disabled")
            return self
        self._old_attrs = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        threading.Thread(target=self._reader, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._old_attrs is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_attrs)

    def _read1(self, timeout):
        """Read one byte from stdin, or None on timeout. os.read (unbuffered):
        buffered sys.stdin.read would swallow escape-sequence bytes past what
        select reported ready."""
        r, _, _ = select.select([self._fd], [], [], timeout)
        if not r:
            return None
        data = os.read(self._fd, 1)
        return data.decode(errors="ignore") if data else None

    def _reader(self):
        while not self._stop.is_set():
            ch = self._read1(0.1)
            if not ch:
                continue
            if ch == "\x1b":  # possible arrow-key escape sequence
                if self._read1(0.05) == "[":
                    final = self._read1(0.05)
                    name = self._ARROWS.get(final) if final else None
                    if name:
                        self._queue.put(name)
                continue  # bare ESC / unknown sequence: ignore
            self._queue.put(ch.lower() if ch.isalpha() else ch)

    def get_keys(self):
        """Drain and return all pending keys (symbolic names / characters)."""
        keys = []
        while True:
            try:
                keys.append(self._queue.get_nowait())
            except queue.Empty:
                return keys


class PolicyInference:
    def __init__(self, model, data, walking_onnx_path=None, action_scale=1.0, bam_ctrl=None,
                 delay_min_lag=0, delay_max_lag=0,
                 standing_onnx_path=None, switch_threshold=0.05,
                 use_projected_gravity=False, ground_pick_onnx_path=None, ground_pick_period=4.0,
                 sit_onnx_path=None, new_cmd_obs=False, slope_onnx_path=None,
                 sitstand_onnx_path=None,
                 kick_left_onnx_path=None, kick_right_onnx_path=None,
                 roulade_onnx_path=None,
                 kick_duration=3.0, roulade_duration=2.0):
        self.bam_ctrl = bam_ctrl  # bam.mujoco.MujocoController (None = legacy position actuators)
        self.model = model
        self.data = data
        self.action_scale = action_scale
        self.use_projected_gravity = use_projected_gravity
        self.delay_min_lag = delay_min_lag
        self.delay_max_lag = delay_max_lag
        self.switch_threshold = switch_threshold
        # When True: emit the unified 13D command vector and treat head_offset /
        # body_cmd as policy COMMANDS (no add to ctrl, no joint_pos correction).
        # When False: legacy behaviour (3D command, head_offset added to ctrl[5:9]).
        self.new_cmd_obs = new_cmd_obs

        # Load walking policy
        self.walking_session = None
        self.default_gait_period_from_onnx = None
        if walking_onnx_path:
            print(f"Loading walking policy from: {walking_onnx_path}")
            self.walking_session = ort.InferenceSession(walking_onnx_path)
            w_input_shape = self.walking_session.get_inputs()[0].shape
            w_output_shape = self.walking_session.get_outputs()[0].shape
            print(f"Walking policy input: {self.walking_session.get_inputs()[0].name}, shape: {w_input_shape}")
            print(f"Walking policy output: {self.walking_session.get_outputs()[0].name}, shape: {w_output_shape}")

            # Try to read gait period from ONNX metadata
            try:
                model_metadata = self.walking_session.get_modelmeta()
                if hasattr(model_metadata, 'custom_metadata_map') and 'gait_period' in model_metadata.custom_metadata_map:
                    self.default_gait_period_from_onnx = float(model_metadata.custom_metadata_map['gait_period'])
                    print(f"Found gait period in ONNX metadata: {self.default_gait_period_from_onnx:.4f}s")
            except Exception as e:
                print(f"Could not read gait period from ONNX metadata: {e}")

        # Load standing policy
        self.standing_session = None
        if standing_onnx_path:
            print(f"\nLoading standing policy from: {standing_onnx_path}")
            self.standing_session = ort.InferenceSession(standing_onnx_path)
            s_input_shape = self.standing_session.get_inputs()[0].shape
            s_output_shape = self.standing_session.get_outputs()[0].shape
            print(f"Standing policy input: {self.standing_session.get_inputs()[0].name}, shape: {s_input_shape}")
            print(f"Standing policy output: {self.standing_session.get_outputs()[0].name}, shape: {s_output_shape}")
            if self.walking_session:
                print(f"Policy switching threshold: {switch_threshold} (vel command magnitude)")

        # Load ground pick policy
        self.ground_pick_session = None
        self.ground_pick_mode = False
        self.ground_pick_phase = 0.0
        self.ground_pick_period = ground_pick_period
        if ground_pick_onnx_path:
            print(f"\nLoading ground pick policy from: {ground_pick_onnx_path}")
            self.ground_pick_session = ort.InferenceSession(ground_pick_onnx_path)
            gp_input_shape = self.ground_pick_session.get_inputs()[0].shape
            print(f"Ground pick policy input shape: {gp_input_shape}")

        # Load sit policy. Two flavours share the Y key and self.sit_session:
        #  - --sit (is_sitstand=False): the OLD one-way sit policy. Sits
        #    unconditionally on a zero twist command; standing back up is done
        #    by switching back to the standing/walking session.
        #  - --sitstand (is_sitstand=True): the commanded sit↔stand policy.
        #    twist[0] is a posture flag (0=stand, 1=sit); the SAME policy sits,
        #    holds, and stands back up — Y just flips the flag.
        self.sit_session = None
        self.sit_mode = False
        self.is_sitstand = False
        if sit_onnx_path and sitstand_onnx_path:
            raise ValueError("Provide only one of --sit / --sitstand")
        if sit_onnx_path:
            print(f"\nLoading sit policy from: {sit_onnx_path}")
            self.sit_session = ort.InferenceSession(sit_onnx_path)
            sit_input_shape = self.sit_session.get_inputs()[0].shape
            print(f"Sit policy input shape: {sit_input_shape}")
        elif sitstand_onnx_path:
            if not self.new_cmd_obs:
                raise ValueError(
                    "--sitstand policies use the unified 13D command obs (61D); run with --new-cmd-obs"
                )
            print(f"\nLoading sitstand policy from: {sitstand_onnx_path}")
            self.sit_session = ort.InferenceSession(sitstand_onnx_path)
            self.is_sitstand = True
            ss_input_shape = self.sit_session.get_inputs()[0].shape
            print(f"Sitstand policy input shape: {ss_input_shape}")

        # Load slope policy (passive descent, runs with zero twist command)
        self.slope_session = None
        self.slope_mode = False
        if slope_onnx_path:
            print(f"\nLoading slope policy from: {slope_onnx_path}")
            self.slope_session = ort.InferenceSession(slope_onnx_path)
            sl_input_shape = self.slope_session.get_inputs()[0].shape
            print(f"Slope policy input shape: {sl_input_shape}")

        # Episodic behavior policies (kick left/right, roulade). All three use
        # the unified 61D obs layout with an ALL-ZERO 13D command (twist forced
        # ~0 in training, head/body slots zero-padded), so triggering one is a
        # plain session swap; after `duration` seconds control hands back to
        # walking/standing (the behavior policies end standing on their own).
        self.behavior_sessions = {}
        self.behavior_durations = {}
        self.behavior_mode = None       # name of the running behavior, or None
        self.behavior_time_left = 0.0
        for name, path, duration in (
            ("kick_left", kick_left_onnx_path, kick_duration),
            ("kick_right", kick_right_onnx_path, kick_duration),
            ("roulade", roulade_onnx_path, roulade_duration),
        ):
            if not path:
                continue
            if not self.new_cmd_obs:
                raise ValueError(
                    f"--{name.replace('_', '-')} policies use the unified 13D "
                    "command obs (61D); run with --new-cmd-obs"
                )
            print(f"\nLoading {name} policy from: {path}")
            self.behavior_sessions[name] = ort.InferenceSession(path)
            self.behavior_durations[name] = duration
            print(f"{name} policy input shape: {self.behavior_sessions[name].get_inputs()[0].shape}"
                  f"  (auto-return after {duration:.1f}s)")

        # Validate at least one policy loaded. A sitstand policy can run alone
        # (it holds the stand at flag=0), unlike the old one-way sit policy.
        if not self.walking_session and not self.standing_session and not self.is_sitstand:
            raise ValueError("At least one of --walking, --standing or --sitstand must be provided")

        # Determine initial active session and policy
        if self.walking_session:
            self.current_policy = "walking"
            self.ort_session = self.walking_session
        elif self.standing_session:
            self.current_policy = "standing"
            self.ort_session = self.standing_session
        else:
            # sitstand-only: start standing (posture flag 0).
            self.current_policy = "sit"
            self.ort_session = self.sit_session

        # Get input/output names from active session
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_name = self.ort_session.get_outputs()[0].name

        # Get sensor IDs and body IDs
        self.imu_ang_vel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")
        self.trunk_base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")

        # Trunk freejoint qpos address (needed to place the ball in the robot's
        # yaw frame) and optional ball freejoint (present in scene_ball.xml).
        _trunk_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
        self._trunk_qpos_adr = int(model.jnt_qposadr[_trunk_jid])
        _ball_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
        if _ball_jid >= 0:
            self.ball_qpos_adr = int(model.jnt_qposadr[_ball_jid])
            self.ball_qvel_adr = int(model.jnt_dofadr[_ball_jid])
        else:
            self.ball_qpos_adr = None
            self.ball_qvel_adr = None

        print(f"Sensors found:")
        print(f"  imu_ang_vel: id={self.imu_ang_vel_id}")
        print(f"Body IDs:")
        print(f"  trunk_base: id={self.trunk_base_id}")

        # Joint information
        self.n_joints = model.nu

        # For robots with passive/interspersed joints (e.g. roller skates), the actuated
        # joints are not contiguous in qpos/qvel. Compute the correct indices from the
        # actuator transmission joint IDs so extraction works for any joint ordering.
        self.joint_qpos_indices = [
            int(model.jnt_qposadr[model.actuator_trnid[i, 0]]) for i in range(model.nu)
        ]
        self.joint_qvel_indices = [
            int(model.jnt_dofadr[model.actuator_trnid[i, 0]]) for i in range(model.nu)
        ]

        # Default pose for the policy (flexed legs)
        self.default_pose = DEFAULT_POSE[:self.n_joints]
        print(f"Number of actuators: {self.n_joints}")
        print(f"Default pose: {self.default_pose}")
        print(f"Action scale: {self.action_scale}")

        # Last action (for observation history)
        self.last_action = np.zeros(self.n_joints, dtype=np.float32)

        # Velocity command [lin_vel_x, lin_vel_y, ang_vel_z] — controls walking / policy switching
        self.vel_cmd = np.zeros(3, dtype=np.float32)
        # Key-press step sizes and limits (overridden per mode in main())
        self.vel_step_x = 0.05
        self.vel_step_y = 0.05
        self.vel_step_ang = 0.3
        self.vel_max_x = 0.3
        self.vel_min_x = -0.3
        self.vel_max_y = 0.3
        self.vel_min_y = -0.3
        self.vel_max_ang = 1.5
        # Body pose command. In new_cmd_obs mode this is 6D
        #   [x, y, z, roll, pitch, yaw] (m, m, m, rad, rad, rad)
        # In legacy mode only [z, pitch, roll] (first 3 indices reused as
        # [z, pitch, roll] to keep the legacy normalization path working).
        self.body_cmd = np.zeros(6 if self.new_cmd_obs else 3, dtype=np.float32)
        # Obs command vector (3D in legacy mode, 13D when new_cmd_obs=True).
        self.command = np.zeros(13 if self.new_cmd_obs else 3, dtype=np.float32)

        # Body pose mode (like head mode but for standing body pose control)
        self.body_pose_mode = False
        self.body_cmd_step_xy = 0.005             # 5 mm per keypress (4 to max)
        self.body_cmd_step_z = 0.01               # 10 mm per keypress (3 to max)
        self.body_cmd_step_angle = math.radians(10) # 10° per keypress (3 to max)

        # Head control mode. In legacy mode head_offset is added on top of
        # ctrl[5:9]; in new_cmd_obs mode it's a *command* fed to the policy.
        # Final per-joint training caps: neck/head_pitch ±1.1, head_yaw ±1.4,
        # head_roll ±0.31. Slider max = widest joint cap; head_roll naturally
        # gets clipped by the policy since it was never trained beyond 0.31.
        self.head_mode = False
        self.head_offset = np.zeros(4, dtype=np.float32)
        if self.new_cmd_obs:
            self.head_max = 1.4
            self.head_step = 0.1
        else:
            self.head_max = 2.5
            self.head_step = 0.83

        # Action delay buffer
        self.use_delay = self.delay_max_lag > 0
        if self.use_delay:
            buffer_size = self.delay_max_lag + 1
            self.action_buffer = [np.zeros(self.n_joints, dtype=np.float32) for _ in range(buffer_size)]
            self.buffer_index = 0
            self.current_lag = np.random.randint(self.delay_min_lag, self.delay_max_lag + 1)
            print(f"\nActuator delay enabled:")
            print(f"  Min lag: {self.delay_min_lag} timesteps")
            print(f"  Max lag: {self.delay_max_lag} timesteps")
            print(f"  Sampled lag: {self.current_lag} timesteps")
            print(f"  Buffer size: {buffer_size}")
        else:
            self.action_buffer = None
            self.current_lag = 0

    def _update_command(self):
        """Update self.command (fed into obs) based on current policy and commands.

        Legacy mode (new_cmd_obs=False): self.command is 3D.
        New mode (new_cmd_obs=True): self.command is 13D:
            [vx, vy, vtheta,                                  ← twist
             neck_pitch, head_pitch, head_yaw, head_roll,     ← head_pose deltas
             body_x, body_y, body_z, body_roll, body_pitch, body_yaw]  ← body_pose
        We keep the existing keyboard mappings: head_offset (4D) drives the head
        slots; body_cmd[0..2] currently mean (Δz, Δpitch, Δroll) and are routed
        into body_pose slots [z, pitch, roll]; x/y/yaw stay 0 (not exposed on
        keyboard yet). ground_pick still owns slots [0..2] for phase encoding.
        """
        if self.new_cmd_obs:
            if self.behavior_mode is not None:
                # Kick/roulade were trained with an all-zero 13D command
                # (twist ~0, head/body slots zero-padded) — feeding stale
                # head/body commands would be out-of-distribution.
                self.command = np.zeros(13, dtype=np.float32)
                return
            cmd = np.zeros(13, dtype=np.float32)
            # twist slot (or phase encoding for ground_pick — overwritten there)
            if self.current_policy == "walking":
                cmd[0:3] = self.vel_cmd
            elif self.current_policy == "sit" and self.is_sitstand:
                # Sitstand posture flag: 1 = sit, 0 = stand. NOT zeros — the
                # all-zero twist is the STAND command for this policy, which is
                # why feeding it the old sit-policy zero command did nothing.
                cmd[0] = 1.0 if self.sit_mode else 0.0
            # else standing/old-sit/ground_pick: leave twist 0 (ground_pick
            # writes its phase encoding later)
            cmd[3:7]  = self.head_offset
            cmd[7:13] = self.body_cmd  # [x, y, z, roll, pitch, yaw]
            self.command = cmd
            return

        # Legacy 3D command
        if self.current_policy == "walking":
            self.command = self.vel_cmd.copy()
        elif self.current_policy == "sit":
            # Sit was trained with a near-zero twist command.
            self.command = np.zeros(3, dtype=np.float32)
        elif self.current_policy == "standing":
            # Normalize body pose cmd to match training's body_pose_cmd_obs
            self.command = np.array([
                self.body_cmd[0] / BODY_CMD_MAX_Z,
                self.body_cmd[1] / BODY_CMD_MAX_ANGLE,
                self.body_cmd[2] / BODY_CMD_MAX_ANGLE,
            ], dtype=np.float32)
        elif self.current_policy == "slope":
            # Passive descent: zero command (like standing coast)
            self.command = np.zeros(3, dtype=np.float32)
        # ground_pick: command is set directly by update_ground_pick_phase

    def _update_policy_session(self):
        """Switch between walking and standing sessions based on vel_cmd magnitude."""
        if not (self.walking_session and self.standing_session):
            return  # Only one policy loaded, no switching
        if self.ground_pick_mode:
            return  # Don't switch during ground pick
        if self.sit_mode:
            return  # Don't switch while sitting
        if self.slope_mode:
            return  # Don't switch during slope mode
        if self.behavior_mode is not None:
            return  # Don't switch during a kick/roulade

        magnitude = float(np.linalg.norm(self.vel_cmd))
        new_policy = "standing" if magnitude <= self.switch_threshold else "walking"
        if new_policy != self.current_policy:
            self.current_policy = new_policy
            self.ort_session = self.standing_session if new_policy == "standing" else self.walking_session
            print(f"Switched to {self.current_policy} policy (vel magnitude: {magnitude:.3f})")
            self._update_command()

    def set_vel_cmd(self, lin_vel_x=0.0, lin_vel_y=0.0, ang_vel_z=0.0):
        """Set velocity command (used for walking / policy switching)."""
        self.vel_cmd = np.array([lin_vel_x, lin_vel_y, ang_vel_z], dtype=np.float32)
        self._update_policy_session()
        self._update_command()
        print(f"Vel cmd: [{lin_vel_x:.2f}, {lin_vel_y:.2f}, {ang_vel_z:.2f}] [{self.current_policy}]")

    def toggle_body_pose_mode(self):
        """Toggle body pose control mode on/off."""
        self.body_pose_mode = not self.body_pose_mode
        if self.body_pose_mode:
            print("Body pose mode: ON")
            print(f"  UP/DOWN: Δz ±{self.body_cmd_step_z*1000:.0f}mm  (max ±{BODY_CMD_MAX_Z*1000:.0f}mm)")
            print(f"  LEFT/RIGHT: Δpitch ±{math.degrees(self.body_cmd_step_angle):.0f}°  (max ±{math.degrees(BODY_CMD_MAX_ANGLE):.0f}°)")
            print(f"  A/E: Δroll ±{math.degrees(self.body_cmd_step_angle):.0f}°  (max ±{math.degrees(BODY_CMD_MAX_ANGLE):.0f}°)")
            if self.new_cmd_obs:
                print(f"  Z/S: Δyaw ±{math.degrees(self.body_cmd_step_angle):.0f}°  (max ±{math.degrees(BODY_CMD_MAX_ANGLE):.0f}°)")
            print(f"  SPACE: reset body pose to zero")
            self._print_body_cmd()
        else:
            print("Body pose mode: OFF")

    def toggle_slope_mode(self):
        """Toggle slope policy mode on/off (passive descent, zero twist command)."""
        if self.slope_session is None:
            print("Slope unavailable: no --slope policy loaded")
            return
        if self.behavior_mode is not None:
            print(f"Cannot toggle slope mode during {self.behavior_mode}")
            return
        self.slope_mode = not self.slope_mode
        if self.slope_mode:
            self.ort_session = self.slope_session
            self.current_policy = "slope"
            self.set_vel_cmd(0.0, 0.0, 0.0)  # passive descent: zero command
            print("Slope mode: ON (passive descent)")
        else:
            self.vel_cmd = np.zeros(3, dtype=np.float32)
            if self.walking_session:
                self.current_policy = "walking"
                self.ort_session = self.walking_session
            else:
                self.current_policy = "standing"
                self.ort_session = self.standing_session
            self._update_command()
            print("Slope mode: OFF")

    def _print_body_cmd(self):
        if self.new_cmd_obs:
            x, y, z, roll, pitch, yaw = self.body_cmd
            print(
                f"Body cmd: x={x*1000:5.1f}mm  y={y*1000:5.1f}mm  z={z*1000:5.1f}mm  "
                f"roll={math.degrees(roll):5.1f}°  pitch={math.degrees(pitch):5.1f}°  "
                f"yaw={math.degrees(yaw):5.1f}°"
            )
        else:
            print(
                f"Body cmd: z={self.body_cmd[0]*1000:.1f}mm  "
                f"pitch={math.degrees(self.body_cmd[1]):.1f}°  "
                f"roll={math.degrees(self.body_cmd[2]):.1f}°"
            )

    # --- body command bumpers (index differs between legacy 3D and new 6D) ---
    def _body_idx(self, axis: str) -> int:
        """Map an axis name to the body_cmd index, depending on the active mode."""
        if self.new_cmd_obs:
            return {"x": 0, "y": 1, "z": 2, "roll": 3, "pitch": 4, "yaw": 5}[axis]
        return {"z": 0, "pitch": 1, "roll": 2}[axis]

    def bump_body(self, axis: str, delta: float):
        idx = self._body_idx(axis)
        cap = BODY_CMD_MAX_Z if axis == "z" else BODY_CMD_MAX_XY if axis in ("x", "y") else BODY_CMD_MAX_ANGLE
        self.body_cmd[idx] = float(np.clip(self.body_cmd[idx] + delta, -cap, cap))
        self._update_command()
        self._print_body_cmd()

    def quat_rotate_inverse(self, quat, vec):
        """Rotate a vector by the inverse of a quaternion [w, x, y, z]."""
        w = quat[0]
        xyz = quat[1:4]
        t = np.cross(xyz, vec) * 2
        return vec - w * t + np.cross(xyz, t)

    def get_raw_accelerometer(self):
        """Get raw accelerometer reading from MuJoCo sensor."""
        sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_accel")
        if sensor_id < 0:
            raise ValueError("Sensor 'imu_accel' not found in model")

        sensor_adr = self.model.sensor_adr[sensor_id]
        accel_raw = self.data.sensordata[sensor_adr:sensor_adr+3].copy().astype(np.float32)
        accel_negated = -accel_raw
        mag = np.linalg.norm(accel_negated)
        if mag > 0.1:
            return accel_negated / mag
        else:
            quat = self.data.xquat[self.trunk_base_id].copy().astype(np.float32)
            world_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
            return self.quat_rotate_inverse(quat, world_gravity)

    def get_projected_gravity(self):
        """Get projected gravity in body frame."""
        quat = self.data.xquat[self.trunk_base_id].copy().astype(np.float32)
        world_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        return self.quat_rotate_inverse(quat, world_gravity)

    def get_base_ang_vel(self):
        """Get base angular velocity from IMU gyro sensor."""
        sensor_adr = self.model.sensor_adr[self.imu_ang_vel_id]
        return self.data.sensordata[sensor_adr:sensor_adr + 3].copy().astype(np.float32)

    def get_joint_pos_relative(self):
        """Get joint positions relative to default pose."""
        current_pos = self.data.qpos[self.joint_qpos_indices].copy().astype(np.float32)
        return current_pos - self.default_pose

    def get_joint_vel(self):
        """Get joint velocities."""
        return self.data.qvel[self.joint_qvel_indices].copy().astype(np.float32)

    def get_observations(self):
        """Collect observations matching policy input.

        Order for velocity/standing task:
        1. base_ang_vel (3D)
        2. raw_accelerometer OR projected_gravity (3D)
        3. joint_pos (14D) - relative to default
        4. joint_vel (14D)
        5. actions (14D) - last action
        6. command (3D) - vel cmd (walking) or normalized body pose cmd (standing)
        Total: 51D
        """
        obs = []

        obs.append(self.get_base_ang_vel())

        if self.use_projected_gravity:
            obs.append(self.get_projected_gravity())
        else:
            obs.append(self.get_raw_accelerometer())

        obs.append(self.get_joint_pos_relative())
        obs.append(self.get_joint_vel())
        obs.append(self.last_action)
        obs.append(self.command)

        return np.concatenate(obs).astype(np.float32)

    def trigger_ground_pick(self):
        """Start one ground pick cycle. Automatically returns to walking when done."""
        if self.ground_pick_session is None:
            print("Ground pick unavailable: no --ground-pick policy loaded")
            return
        if self.ground_pick_mode:
            print("Ground pick already in progress")
            return
        if self.sit_mode:
            print("Cannot ground pick while sitting (press Y to stand up first)")
            return
        if self.behavior_mode is not None:
            print(f"Cannot ground pick during {self.behavior_mode}")
            return
        self.ground_pick_mode = True
        self.ground_pick_phase = 0.0
        self.ort_session = self.ground_pick_session
        self.current_policy = "ground_pick"
        print(f"Ground pick: started (period={self.ground_pick_period:.1f}s)")

    def _end_ground_pick(self):
        """Switch back after a ground pick cycle completes."""
        self.ground_pick_mode = False
        self.vel_cmd = np.zeros(3, dtype=np.float32)
        if self.walking_session:
            self.current_policy = "walking"
            self.ort_session = self.walking_session
        else:
            self.current_policy = "standing"
            self.ort_session = self.standing_session
        self._update_command()
        print(f"Ground pick: done → back to {self.current_policy}")

    def update_ground_pick_phase(self, dt: float):
        """Advance the ground pick phase; auto-exit when one full cycle completes."""
        if not self.ground_pick_mode:
            return
        new_phase = self.ground_pick_phase + dt / self.ground_pick_period
        if new_phase >= 0.7:
            self._end_ground_pick()
            return
        self.ground_pick_phase = new_phase
        # ground_pick policies use the first 3 slots (twist) as phase encoding.
        # Higher slots (head/body) stay at whatever _update_command set them to.
        self.command[0] = np.cos(2 * np.pi * self.ground_pick_phase)
        self.command[1] = np.sin(2 * np.pi * self.ground_pick_phase)
        self.command[2] = 0.0

    def trigger_behavior(self, name):
        """Start an episodic behavior (kick_left / kick_right / roulade).

        The behavior policies were trained to run from a standing start with an
        all-zero command and end standing, so triggering is a session swap; a
        timer hands control back to walking/standing afterwards.
        """
        session = self.behavior_sessions.get(name)
        if session is None:
            print(f"{name} unavailable: no --{name.replace('_', '-')} policy loaded")
            return
        if self.behavior_mode is not None:
            print(f"Cannot start {name}: {self.behavior_mode} already in progress")
            return
        if self.ground_pick_mode:
            print(f"Cannot start {name} during ground pick")
            return
        if self.sit_mode:
            print(f"Cannot start {name} while sitting (press Y to stand up first)")
            return
        if self.slope_mode:
            print(f"Cannot start {name} during slope mode")
            return
        if name in ("kick_left", "kick_right"):
            self._place_ball(name)
        self.behavior_mode = name
        self.behavior_time_left = self.behavior_durations[name]
        self.vel_cmd = np.zeros(3, dtype=np.float32)
        self.current_policy = name
        self.ort_session = session
        self._update_command()
        print(f"{name}: started (auto-return in {self.behavior_time_left:.1f}s)")

    def _place_ball(self, behavior):
        """Teleport the ball in front of the kicking foot, matching training's
        reset_ball_in_front_of_foot (offset in the robot's yaw frame)."""
        if self.ball_qpos_adr is None or self.ball_qvel_adr is None:
            print("No ball in scene (kick will swing at air)")
            return
        adr = self._trunk_qpos_adr
        x, y = float(self.data.qpos[adr]), float(self.data.qpos[adr + 1])
        qw, qx, qy, qz = self.data.qpos[adr + 3:adr + 7]
        yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        off_y = -BALL_OFFSET_ABS_Y if behavior == "kick_right" else BALL_OFFSET_ABS_Y
        bx = x + math.cos(yaw) * BALL_OFFSET_X - math.sin(yaw) * off_y
        by = y + math.sin(yaw) * BALL_OFFSET_X + math.cos(yaw) * off_y
        self.data.qpos[self.ball_qpos_adr:self.ball_qpos_adr + 7] = [bx, by, BALL_RADIUS, 1, 0, 0, 0]
        self.data.qvel[self.ball_qvel_adr:self.ball_qvel_adr + 6] = 0.0
        foot = behavior.split("_")[1]
        print(f"Ball placed at ({bx:.3f}, {by:.3f}) in front of the {foot} foot")

    def update_behavior(self, dt: float):
        """Advance the behavior timer; hand back to walking/standing when done."""
        if self.behavior_mode is None:
            return
        self.behavior_time_left -= dt
        if self.behavior_time_left <= 0.0:
            self._end_behavior()

    def _end_behavior(self):
        name = self.behavior_mode
        self.behavior_mode = None
        self.vel_cmd = np.zeros(3, dtype=np.float32)
        if self.walking_session:
            self.current_policy = "walking"
            self.ort_session = self.walking_session
        elif self.standing_session:
            self.current_policy = "standing"
            self.ort_session = self.standing_session
        else:
            # sitstand-only setup: the sitstand policy holds the stand (flag 0).
            self.current_policy = "sit"
            self.ort_session = self.sit_session
        self._update_command()
        print(f"{name}: done → back to {self.current_policy}")

    def toggle_sit(self):
        """Toggle sitting on/off (Y key).

        Old one-way sit policy (--sit): Y off switches back to the standing/
        walking session, which does the standing back up.
        Sitstand policy (--sitstand): Y just flips the posture flag — the SAME
        policy sits, holds the sit, and stands back up gently (trained response
        to a flag flip is a ~2 s glide). The session stays active after
        standing (it holds the stand); a velocity command switches back to
        walking/standing as usual.
        """
        if self.sit_session is None:
            print("Sit unavailable: no --sit/--sitstand policy loaded")
            return
        if self.ground_pick_mode:
            print("Cannot sit during ground pick")
            return
        if self.behavior_mode is not None:
            print(f"Cannot sit during {self.behavior_mode}")
            return
        self.sit_mode = not self.sit_mode
        if self.sit_mode:
            self.vel_cmd = np.zeros(3, dtype=np.float32)
            self.current_policy = "sit"
            self.ort_session = self.sit_session
            print("Sit: ON" + (" (sitstand flag=1; Y again to stand up)" if self.is_sitstand else ""))
        elif self.is_sitstand:
            # Stay on the sitstand session — it stands up itself (flag → 0).
            # Do NOT swap to the standing policy here: it would take over
            # mid-rise from a seated state it wasn't trained on.
            print("Sit: OFF → sitstand policy standing up (flag=0)")
        else:
            if self.standing_session:
                self.current_policy = "standing"
            else:
                self.current_policy = "walking"
            self.ort_session = self.standing_session if self.current_policy == "standing" else self.walking_session
            print(f"Sit: OFF → back to {self.current_policy}")
        self._update_command()

    def toggle_head_mode(self):
        """Toggle head control mode on/off."""
        self.head_mode = not self.head_mode
        if self.head_mode:
            print("Head mode: ON")
            print(f"  Z/S: neck_pitch  |  UP/DOWN: head_pitch  |  LEFT/RIGHT: head_yaw  |  A/E: head_roll  |  SPACE: reset  (max ±{self.head_max:.2f} rad)")
        else:
            print("Head mode: OFF")

    def infer(self):
        """Run policy inference and return action."""
        obs = self.get_observations()
        obs_batch = obs.reshape(1, -1)
        action = self.ort_session.run([self.output_name], {self.input_name: obs_batch})[0]
        action = action.squeeze(0).astype(np.float32)
        self.last_action = action.copy()
        return action

    def apply_action(self, action):
        """Apply action to MuJoCo controls with optional delay."""
        if self.use_delay:
            self.action_buffer[self.buffer_index] = action.copy()
            delayed_index = (self.buffer_index - self.current_lag) % len(self.action_buffer)
            delayed_action = self.action_buffer[delayed_index]
            self.buffer_index = (self.buffer_index + 1) % len(self.action_buffer)
            target_positions = self.default_pose + delayed_action * self.action_scale
        else:
            target_positions = self.default_pose + action * self.action_scale

        # Legacy mode: head_offset is an external perturbation added on top of
        # the policy output. New mode: head_offset is a COMMAND fed into the
        # policy's obs, so the policy itself produces the offset head pose.
        if not self.new_cmd_obs:
            target_positions = target_positions.copy()
            target_positions[5:9] += self.head_offset
        self.set_position_targets(target_positions)

    def set_position_targets(self, target_positions):
        """Send joint position targets to the actuators.

        BAM: the firmware position loop lives in the controller (ctrl is the
        motor TORQUE it writes on update()). Legacy: MuJoCo position actuators.
        """
        if self.bam_ctrl is not None:
            self.bam_ctrl.q_target[:] = target_positions
        else:
            self.data.ctrl[:] = target_positions


def main():
    parser = argparse.ArgumentParser(description="Run ONNX policy in MuJoCo")
    parser.add_argument("--roller", action="store_true", help="Use roller skate robot XML (robot_walk_rollers.xml)")
    parser.add_argument("--scene", type=str, default=None, help="Path to a scene XML, overriding the default pick (e.g. src/mjlab_microduck/robot/microduck/scene_allcollisions.xml)")
    parser.add_argument("--walking", type=str, default=None, help="Path to walking policy ONNX file")
    parser.add_argument("--standing", "-s", type=str, default=None, help="Path to standing policy ONNX file")
    parser.add_argument("--ground-pick", type=str, default=None, help="Path to ground pick policy ONNX file (press G to activate)")
    parser.add_argument("--sit", type=str, default=None, help="Path to OLD one-way sitting policy ONNX file (press Y to sit, Y again switches back to standing/walking policy)")
    parser.add_argument("--sitstand", type=str, default=None, help="Path to sitstand policy ONNX (commanded sit<->stand; press Y to sit, Y again the SAME policy stands back up). Requires --new-cmd-obs. Can run standalone.")
    parser.add_argument("--slope", type=str, default=None, help="Path to slope policy ONNX file (press Y to toggle)")
    parser.add_argument("--kick-left", type=str, default=None, help="Path to LEFT-foot ball kick policy ONNX (press K to trigger). Requires --new-cmd-obs. Loads a scene with a ball.")
    parser.add_argument("--kick-right", type=str, default=None, help="Path to RIGHT-foot ball kick policy ONNX (press L to trigger). Requires --new-cmd-obs. Loads a scene with a ball.")
    parser.add_argument("--roulade", type=str, default=None, help="Path to roulade (forward roll) policy ONNX (press R to trigger). Requires --new-cmd-obs.")
    parser.add_argument("--kick-duration", type=float, default=3.0, help="Seconds a kick policy stays active before handing back to standing/walking (default: 3.0)")
    parser.add_argument("--roulade-duration", type=float, default=2.0, help="Seconds the roulade policy stays active before handing back to standing/walking (default: 2.0, ~the roll itself; the standing/walking policy takes over for the settle)")
    parser.add_argument("--lin-vel-x", type=float, default=0.0, help="Initial linear velocity X command (m/s)")
    parser.add_argument("--lin-vel-y", type=float, default=0.0, help="Initial linear velocity Y command (m/s)")
    parser.add_argument("--ang-vel-z", type=float, default=0.0, help="Initial angular velocity Z command (rad/s)")
    parser.add_argument("--action-scale", type=float, default=1.0, help="Action scale (default: 1.0)")
    parser.add_argument("--raw-accelerometer", action="store_true", help="Use raw accelerometer instead of projected gravity")
    parser.add_argument("--delay", type=int, nargs='*', default=None, help="Enable actuator delay: --delay MIN MAX or --delay LAG")
    parser.add_argument("--debug", action="store_true", help="Print observations and actions")
    parser.add_argument("--save-csv", type=str, default=None, help="Save observations and actions to CSV file")
    parser.add_argument("--record", type=str, default=None, help="Enable recording mode: save observations to pickle file on Ctrl+C")
    parser.add_argument("--switch-threshold", type=float, default=0.05, help="Vel command magnitude threshold for walking/standing switch (default: 0.05)")
    parser.add_argument("--ground-pick-period", type=float, default=4.0, help="Ground pick phase period in seconds (default: 4.0)")
    parser.add_argument("--new-cmd-obs", action="store_true",
                        help="Use the unified 13D command obs layout (twist+head_pose+body_pose). "
                             "Required for policies trained with the new pose-command-tracking setup. "
                             "Old policies (51D obs, head_offset added to ctrl) need this flag OFF.")
    parser.add_argument("--no-bam", action="store_true",
                        help="Use the XML MuJoCo position actuators instead of the BAM M6 "
                             "voltage/friction model the policies are trained against.")
    parser.add_argument("--vin", type=float, default=7.4,
                        help="BAM battery voltage [V]. Training samples per-env in "
                             f"{BAM_VIN_RANGE}; 7.4 = nominal 2S LiPo.")
    parser.add_argument("--vin-drop-gain", type=float, default=0.1,
                        help="BAM load-dependent voltage sag gain [V/Nm], V = vin - gain*sum|tau|. "
                             f"Training samples per-env in {BAM_VIN_DROP_GAIN_RANGE}. 0 disables.")
    parser.add_argument("--kp-fw", type=float, default=BAM_KP_FW,
                        help="BAM firmware P-gain (training uses %(default)s).")
    parser.add_argument("--current-limit", type=float, default=0.0,
                        help="XL330 firmware current limit [A]. With BAM this is the duty-cycle "
                             "limiter of the voltage model (as bam models it); with --no-bam the "
                             "actuator force is clipped to +/- current_limit * kt. Training runs "
                             "WITHOUT a current limit, so the default is off (<=0).")
    parser.add_argument("--foot-friction", type=float, default=None,
                        help="Override the foot sliding friction (mu) to emulate the real grippy "
                             "PU sole. Training used mu~1.0 (range 0.7-1.3); real PU is likely "
                             "~1.5-2.5. e.g. --foot-friction 2.0")
    parser.add_argument("--foot-solref", type=float, default=None,
                        help="Soften foot contact: solref time constant (s) for the foot geoms "
                             "(default sim ~0.02 = stiff/rigid). Larger = softer, to emulate the "
                             "compliant PU sole. e.g. --foot-solref 0.04")
    args = parser.parse_args()

    if not args.walking and not args.standing and not args.sitstand:
        parser.error("At least one of --walking, --standing or --sitstand must be provided")
    if args.sitstand and not args.new_cmd_obs:
        parser.error("--sitstand policies use the unified 13D command obs (61D); add --new-cmd-obs")
    if (args.kick_left or args.kick_right or args.roulade) and not args.new_cmd_obs:
        parser.error("--kick-left/--kick-right/--roulade policies use the unified 13D command obs (61D); add --new-cmd-obs")
    if (args.kick_left or args.kick_right or args.roulade) and args.roller:
        parser.error("kick/roulade policies are trained on the walking robot, not the roller model")

    # Parse delay arguments
    delay_min_lag = 0
    delay_max_lag = 0
    if args.delay is not None:
        if len(args.delay) == 0:
            delay_min_lag = 1
            delay_max_lag = 2
        elif len(args.delay) == 1:
            delay_min_lag = args.delay[0]
            delay_max_lag = args.delay[0]
        elif len(args.delay) == 2:
            delay_min_lag = args.delay[0]
            delay_max_lag = args.delay[1]
        else:
            print("Error: --delay accepts 0, 1, or 2 arguments")
            return

    # Load MuJoCo model. Kick policies get a scene with a ball to kick.
    # --scene overrides everything (any scene whose robot has the standard
    # 14-servo layout works, e.g. scene_allcollisions.xml).
    if args.scene:
        xml_path = args.scene
    elif args.roller:
        xml_path = MICRODUCK_ROLLERS_XML
    elif args.kick_left or args.kick_right:
        xml_path = MICRODUCK_BALL_XML
    else:
        xml_path = MICRODUCK_XML
    print(f"Loading MuJoCo model from: {xml_path}")
    bam_ctrl = None
    if not args.no_bam:
        # Same actuator the policies are trained against in warp (BAM M6 XL330,
        # voltage control + load-dependent friction budget), driven on CPU by
        # bam.mujoco.MujocoController. Voltage DR collapses to fixed --vin /
        # --vin-drop-gain (training samples them per env).
        bam_model = load_bam_model(args.kp_fw, args.vin, args.current_limit)
        vin_drop_gain = args.vin_drop_gain if args.vin_drop_gain > 0 else None
        model, data, bam_ctrl, _bam_names = load_mujoco_with_bam(
            xml_path, bam_model, 0.005, vin_drop_gain, BAM_VIN_MIN)
    else:
        model = mujoco.MjModel.from_xml_path(xml_path)
        model.opt.timestep = 0.005
        data = mujoco.MjData(model)
        print("Legacy MuJoCo position actuators (--no-bam): NOT the actuator the policy was trained with")

    # (--no-bam only) XL330 firmware current limit. The motors saturate current
    # at ~1.75 A; since torque = kt * current, this caps the actuator force at
    # +/- kt * I_max. With BAM the limiter is modelled inside the voltage
    # controller instead (see load_bam_model). kt comes from the bam package.
    if args.no_bam and args.current_limit and args.current_limit > 0:
        from bam.model import load_model
        kt = load_model(motor_name="xl330", model="m6").kt.value
        torque_limit = kt * args.current_limit
        model.actuator_forcerange[:, 0] = -torque_limit
        model.actuator_forcerange[:, 1] = torque_limit
        model.actuator_forcelimited[:] = 1
        print(f"Current limit: {args.current_limit:.2f} A -> torque limit "
              f"+/-{torque_limit:.4f} Nm (kt={kt:.4f})")

    # Foot contact override — emulate the real grippy + soft PU sole to check
    # whether it reproduces the on-robot forward-fall-at-speed. Training used
    # rigid feet at mu~1.0; the real sole is grippier (higher mu) and compliant
    # (softer solref). Applied to the foot collision geoms only.
    if args.foot_friction is not None or args.foot_solref is not None:
        import re as _re
        n_feet = 0
        for g in range(model.ngeom):
            gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
            if gname and _re.match(r"^(left|right)_foot_collision$", gname):
                if args.foot_friction is not None:
                    model.geom_friction[g, 0] = args.foot_friction  # tangential mu
                if args.foot_solref is not None:
                    model.geom_solref[g, 0] = args.foot_solref       # softer contact
                    model.geom_solref[g, 1] = 1.0
                n_feet += 1
        print(f"Foot override on {n_feet} geoms: "
              f"mu={args.foot_friction if args.foot_friction is not None else 'default'}, "
              f"solref={args.foot_solref if args.foot_solref is not None else 'default'}")

    # Initialize policy
    policy = PolicyInference(
        model, data,
        bam_ctrl=bam_ctrl,
        walking_onnx_path=args.walking,
        action_scale=args.action_scale,
        delay_min_lag=delay_min_lag,
        delay_max_lag=delay_max_lag,
        standing_onnx_path=args.standing,
        switch_threshold=args.switch_threshold,
        use_projected_gravity=not args.raw_accelerometer,
        ground_pick_onnx_path=args.ground_pick,
        ground_pick_period=args.ground_pick_period,
        sit_onnx_path=args.sit,
        new_cmd_obs=args.new_cmd_obs,
        slope_onnx_path=args.slope,
        sitstand_onnx_path=args.sitstand,
        kick_left_onnx_path=args.kick_left,
        kick_right_onnx_path=args.kick_right,
        roulade_onnx_path=args.roulade,
        kick_duration=args.kick_duration,
        roulade_duration=args.roulade_duration,
    )
    policy.set_vel_cmd(args.lin_vel_x, args.lin_vel_y, args.ang_vel_z)

    # Set realistic wheel bearing friction for roller inference (must be done
    # programmatically — non-zero frictionloss in the XML breaks training)
    if args.roller:
        import re
        for j in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if name and re.match(r"^passive_.*", name):
                dof_adr = model.jnt_dofadr[j]
                model.dof_frictionloss[dof_adr] = 0.003

    # Per-mode velocity command limits matching training ranges
    if args.roller:
        policy.vel_step_x = 0.05      # lin_vel_x step (range -0.5..0.6)
        policy.vel_step_y = 0.0       # no lateral command for rollers
        policy.vel_step_ang = 0.1     # heading error step (range ±1.0 rad)
        policy.vel_max_x = 0.6
        policy.vel_min_x = -0.5       # negative = brake
        policy.vel_max_y = 0.0
        policy.vel_min_y = 0.0
        policy.vel_max_ang = 1.0      # ±1.0 rad heading error
    else:
        policy.vel_max_x = 0.3
        policy.vel_min_x = -0.3
        policy.vel_max_y = 0.2
        policy.vel_min_y = -0.2
        policy.vel_max_ang = 1.5

    # Set initial position to default pose
    freejoint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    qpos_adr = model.jnt_qposadr[freejoint_id]
    data.qpos[qpos_adr + 0] = 0.0
    data.qpos[qpos_adr + 1] = 0.0
    data.qpos[qpos_adr + 2] = 0.1385 if args.roller else 0.125  # rollers add 13.5mm height
    data.qpos[qpos_adr + 3:qpos_adr + 7] = [1, 0, 0, 0]
    for i, qpos_idx in enumerate(policy.joint_qpos_indices):
        data.qpos[qpos_idx] = policy.default_pose[i]
    if bam_ctrl is not None:
        bam_ctrl.reset(data.qpos)   # clears voltage-drop state, q_target = current qpos
    policy.set_position_targets(policy.default_pose)
    mujoco.mj_forward(model, data)

    # Verify observation size
    test_obs = policy.get_observations()
    cmd_dim = 13 if policy.new_cmd_obs else 3
    expected_obs_size = 3 + 3 + policy.n_joints + policy.n_joints + policy.n_joints + cmd_dim
    breakdown = (
        f"3(ang_vel) + 3(proj_grav) + {policy.n_joints}(joint_pos) + "
        f"{policy.n_joints}(joint_vel) + {policy.n_joints}(last_action) + {cmd_dim}(command)"
    )

    if test_obs.size != expected_obs_size:
        print(f"\nWARNING: Observation size mismatch!")
        print(f"  Expected: {expected_obs_size}")
        print(f"  Got: {test_obs.size}")
        print(f"  Breakdown: {breakdown}")
        print()

    print("\n" + "="*80)
    print("MicroDuck Policy Inference")
    print("="*80)
    print(f"Control frequency: 50 Hz (decimation: 4)")
    print(f"Simulation timestep: {model.opt.timestep}s")
    print(f"Observation size: {test_obs.size} (expected: {expected_obs_size})")
    if policy.walking_session:
        print(f"Walking policy: loaded")
    if policy.standing_session:
        print(f"Standing policy: loaded  (body pose: z=±{BODY_CMD_MAX_Z*1000:.0f}mm, pitch/roll=±{math.degrees(BODY_CMD_MAX_ANGLE):.0f}°)")
    if policy.walking_session and policy.standing_session:
        print(f"  Switch threshold: {policy.switch_threshold} (vel cmd magnitude)")
    if policy.ground_pick_session:
        print(f"Ground pick policy: loaded  (press G)")
    if policy.sit_session:
        kind = "Sitstand" if policy.is_sitstand else "Sit"
        print(f"{kind} policy: loaded  (press Y to toggle)")
    if policy.slope_session:
        print(f"Slope policy: loaded  (press Y to toggle, passive descent)")
    _behavior_keys = {"kick_left": "K", "kick_right": "L", "roulade": "R"}
    for _name in policy.behavior_sessions:
        print(f"{_name} policy: loaded  (press {_behavior_keys[_name]}, "
              f"auto-return after {policy.behavior_durations[_name]:.1f}s)")
    print(f"Active policy: {policy.current_policy}")
    print("Close viewer window to exit")
    print()

    decimation = 4
    control_step_count = 0
    control_dt = decimation * model.opt.timestep

    # Rolling buffer of trunk world-frame xy velocity over the last 1 s, used
    # to print a running average so we can compare commanded vs achieved speed.
    from collections import deque
    _vel_window_steps = max(1, int(round(1.0 / control_dt)))   # ≈ 50 @ 50 Hz
    vel_history = deque(maxlen=_vel_window_steps)

    csv_data = [] if args.save_csv else None
    recorded_observations = [] if args.record else None
    policy_enabled = not args.record
    policy_enable_time = None
    original_kp = None
    if args.record:
        original_kp = model.actuator_gainprm[:, 0].copy()

    # Standby (--record) gains: the legacy path sets the position-actuator kp
    # to 2.0 (XML kp 0.55 ~ kp_fw 200). Under BAM apply the same ratio to the
    # firmware gain so both paths hold with the same relative stiffness.
    _XML_KP_NOMINAL = 0.55
    _STANDBY_KP = 2.0

    def set_standby_gains(on: bool):
        if bam_ctrl is not None:
            bam_ctrl.model.actuator.kp = args.kp_fw * (_STANDBY_KP / _XML_KP_NOMINAL if on else 1.0)
            print(f"  BAM kp_fw set to {bam_ctrl.model.actuator.kp:.0f}")
            return
        for i in range(model.nu):
            kp = _STANDBY_KP if on else original_kp[i]
            model.actuator_gainprm[i, 0] = kp
            model.actuator_biasprm[i, 1] = -kp

    # Cache the trunk freejoint qvel address so the push handler can write to
    # the trunk's world-frame linear velocity directly (qvel[0..3]).
    _freejoint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    _trunk_qvel_adr = int(model.jnt_dofadr[_freejoint_id])
    PUSH_MAX = 1.0   # matches the final velstand push_magnitude curriculum cap

    def random_push():
        """Set the trunk's world-frame xy velocity to a random vector of
        magnitude PUSH_MAX, simulating the push_by_setting_velocity training
        event. Doesn't accumulate — overwrites current linear velocity."""
        import random
        angle = random.uniform(0, 2 * np.pi)
        vx = PUSH_MAX * np.cos(angle)
        vy = PUSH_MAX * np.sin(angle)
        data.qvel[_trunk_qvel_adr + 0] = vx
        data.qvel[_trunk_qvel_adr + 1] = vy
        print(f"PUSH applied: v=[{vx:.2f}, {vy:.2f}, 0] m/s (angle={np.degrees(angle):.0f}°)")

    # Keys come from the TERMINAL (raw stdin, see TerminalInput) — not from the
    # MuJoCo viewer window, whose keypresses also fire built-in visualization
    # shortcuts. `key` is a symbolic name: "up"/"down"/"left"/"right", " ", or
    # a lowercase letter.
    quit_requested = False

    def handle_key(key):
        nonlocal policy_enabled, quit_requested
        try:
            if key == "up":
                if policy.head_mode:
                    policy.head_offset[1] = np.clip(policy.head_offset[1] + policy.head_step, -policy.head_max, policy.head_max)
                    policy._update_command()
                    print(f"Head offset: neck={policy.head_offset[0]:.2f} pitch={policy.head_offset[1]:.2f} yaw={policy.head_offset[2]:.2f} roll={policy.head_offset[3]:.2f}")
                elif policy.body_pose_mode:
                    policy.bump_body("z", policy.body_cmd_step_z)
                else:
                    policy.set_vel_cmd(policy.vel_max_x, policy.vel_cmd[1], policy.vel_cmd[2])
            elif key == "down":
                if policy.head_mode:
                    policy.head_offset[1] = np.clip(policy.head_offset[1] - policy.head_step, -policy.head_max, policy.head_max)
                    policy._update_command()
                    print(f"Head offset: neck={policy.head_offset[0]:.2f} pitch={policy.head_offset[1]:.2f} yaw={policy.head_offset[2]:.2f} roll={policy.head_offset[3]:.2f}")
                elif policy.body_pose_mode:
                    policy.bump_body("z", -policy.body_cmd_step_z)
                else:
                    policy.set_vel_cmd(policy.vel_min_x, policy.vel_cmd[1], policy.vel_cmd[2])
            elif key == "right":
                if policy.head_mode:
                    policy.head_offset[2] = np.clip(policy.head_offset[2] - policy.head_step, -policy.head_max, policy.head_max)
                    policy._update_command()
                    print(f"Head offset: neck={policy.head_offset[0]:.2f} pitch={policy.head_offset[1]:.2f} yaw={policy.head_offset[2]:.2f} roll={policy.head_offset[3]:.2f}")
                elif policy.body_pose_mode:
                    policy.bump_body("pitch", -policy.body_cmd_step_angle)
                elif args.roller:
                    new_ang = np.clip(policy.vel_cmd[2] - policy.vel_step_ang, -policy.vel_max_ang, policy.vel_max_ang)
                    policy.set_vel_cmd(policy.vel_cmd[0], policy.vel_cmd[1], new_ang)
                else:
                    policy.set_vel_cmd(policy.vel_cmd[0], policy.vel_min_y, policy.vel_cmd[2])
            elif key == "left":
                if policy.head_mode:
                    policy.head_offset[2] = np.clip(policy.head_offset[2] + policy.head_step, -policy.head_max, policy.head_max)
                    policy._update_command()
                    print(f"Head offset: neck={policy.head_offset[0]:.2f} pitch={policy.head_offset[1]:.2f} yaw={policy.head_offset[2]:.2f} roll={policy.head_offset[3]:.2f}")
                elif policy.body_pose_mode:
                    policy.bump_body("pitch", policy.body_cmd_step_angle)
                elif args.roller:
                    new_ang = np.clip(policy.vel_cmd[2] + policy.vel_step_ang, -policy.vel_max_ang, policy.vel_max_ang)
                    policy.set_vel_cmd(policy.vel_cmd[0], policy.vel_cmd[1], new_ang)
                else:
                    policy.set_vel_cmd(policy.vel_cmd[0], policy.vel_max_y, policy.vel_cmd[2])
            elif key == " ":
                if policy.head_mode:
                    policy.head_offset[:] = 0.0
                    policy._update_command()
                    print("Head offset reset to zero")
                elif policy.body_pose_mode:
                    policy.body_cmd[:] = 0.0
                    policy._update_command()
                    print("Body pose cmd reset to zero")
                else:
                    policy.set_vel_cmd(0.0, 0.0, 0.0)
            elif key == "t":
                # Toggle policy inference on/off. When OFF the controller stops
                # querying the ONNX policy and the motors hold the last applied
                # target (no fresh ctrl writes).
                policy_enabled = not policy_enabled
                print(f"Policy inference: {'ON' if policy_enabled else 'OFF (paused)'}")
            elif key == "g":
                policy.trigger_ground_pick()
            elif key == "k":
                policy.trigger_behavior("kick_left")
            elif key == "l":
                policy.trigger_behavior("kick_right")
            elif key == "r":
                policy.trigger_behavior("roulade")
            elif key == "q":
                quit_requested = True
                print("Quit requested")
            elif key == "y":
                # Y toggles whichever aux policy is loaded (--sit or --slope).
                if policy.sit_session is not None:
                    policy.toggle_sit()
                else:
                    policy.toggle_slope_mode()
            elif key == "h":
                policy.toggle_head_mode()
            elif key == "b":
                policy.toggle_body_pose_mode()
            elif key == "p":
                random_push()
            elif key == "a":
                if policy.head_mode:
                    policy.head_offset[3] = np.clip(policy.head_offset[3] + policy.head_step, -policy.head_max, policy.head_max)
                    policy._update_command()
                    print(f"Head offset: neck={policy.head_offset[0]:.2f} pitch={policy.head_offset[1]:.2f} yaw={policy.head_offset[2]:.2f} roll={policy.head_offset[3]:.2f}")
                elif policy.body_pose_mode:
                    policy.bump_body("roll", policy.body_cmd_step_angle)
                else:
                    policy.set_vel_cmd(policy.vel_cmd[0], policy.vel_cmd[1], policy.vel_max_ang)
            elif key == "e":
                if policy.head_mode:
                    policy.head_offset[3] = np.clip(policy.head_offset[3] - policy.head_step, -policy.head_max, policy.head_max)
                    policy._update_command()
                    print(f"Head offset: neck={policy.head_offset[0]:.2f} pitch={policy.head_offset[1]:.2f} yaw={policy.head_offset[2]:.2f} roll={policy.head_offset[3]:.2f}")
                elif policy.body_pose_mode:
                    policy.bump_body("roll", -policy.body_cmd_step_angle)
                else:
                    policy.set_vel_cmd(policy.vel_cmd[0], policy.vel_cmd[1], -policy.vel_max_ang)
            elif key == "z":
                if policy.head_mode:
                    policy.head_offset[0] = np.clip(policy.head_offset[0] + policy.head_step, -policy.head_max, policy.head_max)
                    policy._update_command()
                    print(f"Head offset: neck={policy.head_offset[0]:.2f} pitch={policy.head_offset[1]:.2f} yaw={policy.head_offset[2]:.2f} roll={policy.head_offset[3]:.2f}")
                elif policy.body_pose_mode and policy.new_cmd_obs:
                    policy.bump_body("yaw", policy.body_cmd_step_angle)
            elif key == "s":
                if policy.head_mode:
                    policy.head_offset[0] = np.clip(policy.head_offset[0] - policy.head_step, -policy.head_max, policy.head_max)
                    policy._update_command()
                    print(f"Head offset: neck={policy.head_offset[0]:.2f} pitch={policy.head_offset[1]:.2f} yaw={policy.head_offset[2]:.2f} roll={policy.head_offset[3]:.2f}")
                elif policy.body_pose_mode and policy.new_cmd_obs:
                    policy.bump_body("yaw", -policy.body_cmd_step_angle)
        except Exception as e:
            print(f"Key press error: {e}")

    print("\nKeyboard controls (type in THIS terminal — the viewer window no longer captures keys):")
    print("  [ Velocity mode (default) ]")
    print("  UP arrow:         increase lin_vel_x (push/accelerate)")
    print("  DOWN arrow:       decrease lin_vel_x (0=coast, negative=brake)")
    if args.roller:
        print("  LEFT/RIGHT arrow: turn left/right (ang_vel_z heading error)")
        print("  A / E:            turn left/right (ang_vel_z, incremental)")
    else:
        print("  LEFT/RIGHT arrow: strafe left/right (lin_vel_y)")
        print("  A / E:            turn left/right (ang_vel_z)")
    print("  SPACE:            coast (zero all commands)")
    print("  T:                toggle policy inference on/off (paused = motors hold last target)")
    print("  G:                trigger ground pick (requires --ground-pick)")
    print("  Y:                toggle sit (with --sit/--sitstand) or slope mode (with --slope)")
    print("  K:                kick with LEFT foot (requires --kick-left)")
    print("  L:                kick with RIGHT foot (requires --kick-right)")
    print("  R:                roulade / forward roll (requires --roulade)")
    print(f"  P:                random push (trunk vel = {PUSH_MAX:.1f} m/s in random direction)")
    print("  Q:                quit")
    print("  [ Body pose mode — press B to toggle ]")
    print(f"  UP/DOWN arrow:    Δz ±10mm  (max ±{BODY_CMD_MAX_Z*1000:.0f}mm)")
    print(f"  LEFT/RIGHT arrow: Δpitch ±10°  (max ±{math.degrees(BODY_CMD_MAX_ANGLE):.0f}°)")
    print(f"  A / E:            Δroll ±10°  (max ±{math.degrees(BODY_CMD_MAX_ANGLE):.0f}°)")
    if args.new_cmd_obs:
        print(f"  Z / S:            Δyaw ±10°  (new_cmd_obs only, max ±{math.degrees(BODY_CMD_MAX_ANGLE):.0f}°)")
    print("  SPACE:            reset body pose to zero")
    print("  [ Head mode — press H to toggle ]")
    print("  Z / S:            neck_pitch ±step")
    print("  UP/DOWN arrow:    head_pitch ±step")
    print("  LEFT/RIGHT arrow: head_yaw ±step")
    print("  A / E:            head_roll ±step")
    print("  SPACE:            reset head offset to zero")

    with TerminalInput() as term, \
         mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        viewer.sync()
        start_time = time.time()

        if args.record:
            policy_enable_time = start_time + 1.0
            print("Recording mode: policy will be enabled after 1 second standby")
            set_standby_gains(True)
            print("  Standby mode: kp set to 2.0 (XML units)")

        try:
            prev_step_time = time.time()

            while viewer.is_running() and not quit_requested:
                step_start = time.time()

                for key in term.get_keys():
                    handle_key(key)

                if not policy_enabled and policy_enable_time is not None:
                    if step_start >= policy_enable_time:
                        policy_enabled = True
                        if original_kp is not None:
                            set_standby_gains(False)
                            print("Policy inference enabled (after 1s standby)")
                            print(f"  Restored original kp gains (range: [{original_kp.min():.2f}, {original_kp.max():.2f}])")

                actual_dt = step_start - prev_step_time
                prev_step_time = step_start

                policy.update_ground_pick_phase(actual_dt)
                policy.update_behavior(actual_dt)

                if policy_enabled:
                    action = policy.infer()
                    policy.apply_action(action)
                else:
                    # Paused: keep last ctrl, don't query the policy. Motors
                    # hold position. Use a zero action just so downstream
                    # logging (csv/debug) sees something consistent.
                    action = np.zeros(policy.n_joints, dtype=np.float32)

                control_step_count += 1

                # Track BODY-frame forward/lateral velocity + yaw rate, print the
                # 1-second moving average once per second vs the commanded values.
                # Body frame so "forward" / "turn" are directly comparable to the
                # command (which is in the robot frame): lets us see if the policy
                # actually achieves commanded forward speed and turn rate.
                quat = data.qpos[qpos_adr + 3:qpos_adr + 7].astype(np.float32)
                v_world = np.array([
                    data.qvel[_trunk_qvel_adr + 0],
                    data.qvel[_trunk_qvel_adr + 1],
                    data.qvel[_trunk_qvel_adr + 2],
                ], dtype=np.float32)
                v_body = policy.quat_rotate_inverse(quat, v_world)
                yaw_rate = float(data.qvel[_trunk_qvel_adr + 5])  # body-frame wz
                vel_history.append((float(v_body[0]), float(v_body[1]), yaw_rate))
                if control_step_count % _vel_window_steps == 0 and len(vel_history) > 0:
                    n = len(vel_history)
                    avg_fwd = sum(v[0] for v in vel_history) / n
                    avg_lat = sum(v[1] for v in vel_history) / n
                    avg_yaw = sum(v[2] for v in vel_history) / n
                    cmd_x, cmd_y, cmd_yaw = policy.vel_cmd[0], policy.vel_cmd[1], policy.vel_cmd[2]
                    trunk_z = float(data.qpos[qpos_adr + 2])
                    print(
                        f"[vel 1s avg] achieved/cmd  fwd={avg_fwd:+.2f}/{cmd_x:+.2f}  "
                        f"lat={avg_lat:+.2f}/{cmd_y:+.2f} m/s  "
                        f"yaw={avg_yaw:+.2f}/{cmd_yaw:+.2f} rad/s   "
                        f"trunk_z={trunk_z*1000:.1f} mm"
                    )

                if csv_data is not None:
                    obs = policy.get_observations()
                    row = {'step': control_step_count, 'time': control_step_count * control_dt}
                    for i in range(obs.size):
                        row[f'obs_{i}'] = obs[i]
                    for i in range(action.size):
                        row[f'action_{i}'] = action[i]
                    csv_data.append(row)

                if recorded_observations is not None:
                    obs = policy.get_observations()
                    timestamp = time.time() - start_time
                    recorded_observations.append({'timestamp': timestamp, 'observation': obs.tolist()})

                if args.debug:
                    should_print = control_step_count <= 10 or control_step_count % 50 == 0
                    if should_print:
                        obs = policy.get_observations()
                        pos = data.qpos[qpos_adr:qpos_adr + 3]
                        quat = data.qpos[qpos_adr + 3:qpos_adr + 7]
                        com_height = pos[2]

                        print(f"\n{'='*70}")
                        print(f"Step {control_step_count} DEBUG:")
                        print(f"{'='*70}")
                        print(f"Active policy: {policy.current_policy}")
                        print(f"Base state:")
                        print(f"  Position: [{pos[0]:7.4f}, {pos[1]:7.4f}, {pos[2]:7.4f}]")
                        print(f"  CoM height: {com_height:7.4f}")
                        print(f"  Quaternion: [{quat[0]:7.4f}, {quat[1]:7.4f}, {quat[2]:7.4f}, {quat[3]:7.4f}]")
                        print(f"\nObservation (shape {obs.shape}, total {obs.size}):")
                        print(f"  Ang vel [0:3]:        {obs[0:3]}")
                        print(f"  Proj grav [3:6]:      {obs[3:6]}")
                        print(f"  Joint pos [6:{6+policy.n_joints}]:     {obs[6:6+policy.n_joints]}")
                        print(f"  Joint vel [{6+policy.n_joints}:{6+2*policy.n_joints}]:    {obs[6+policy.n_joints:6+2*policy.n_joints]}")
                        print(f"  Last action [{6+2*policy.n_joints}:{6+3*policy.n_joints}]:  {obs[6+2*policy.n_joints:6+3*policy.n_joints]}")
                        cmd_end = 6+3*policy.n_joints+3
                        print(f"  Command [{6+3*policy.n_joints}:{cmd_end}]:      {obs[6+3*policy.n_joints:cmd_end]}")
                        if policy.current_policy == "standing":
                            print(f"  Body cmd (raw): z={policy.body_cmd[0]*1000:.1f}mm  pitch={math.degrees(policy.body_cmd[1]):.1f}°  roll={math.degrees(policy.body_cmd[2]):.1f}°")
                        print(f"\nAction output:")
                        print(f"  Raw action: {action}")
                        print(f"  Action min/max: [{action.min():.4f}, {action.max():.4f}]")
                        if policy.use_delay:
                            print(f"  Delay: {policy.current_lag} timesteps (buffered)")
                        ctrl_kind = "torque [Nm]" if bam_ctrl is not None else "position target"
                        print(f"  Applied ctrl ({ctrl_kind}, first 5): {data.ctrl[:5]}")
                        print(f"  Applied ctrl ({ctrl_kind}, last 5):  {data.ctrl[-5:]}")

                for _ in range(decimation):
                    if bam_ctrl is not None:
                        # BAM owns control/torque/friction: update() runs the
                        # firmware P-loop + DC-motor equation, writes the torque
                        # to data.ctrl and pushes the friction budget onto the
                        # dofs so MuJoCo's solver applies it on this step.
                        bam_ctrl.update()
                    mujoco.mj_step(model, data)

                viewer.sync()

                elapsed = time.time() - step_start
                sleep_time = control_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n\nKeyboardInterrupt received (Ctrl+C). Saving data...")

    print("\nInference stopped.")

    if csv_data is not None and len(csv_data) > 0:
        print(f"\nSaving {len(csv_data)} steps to: {args.save_csv}")
        with open(args.save_csv, 'w', newline='') as csvfile:
            fieldnames = csv_data[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_data)
        print(f"CSV file saved successfully!")
        print(f"  Columns: {len(fieldnames)}")
        print(f"  Rows: {len(csv_data)}")

    if recorded_observations is not None and len(recorded_observations) > 0:
        print(f"\nSaving {len(recorded_observations)} recorded observations to: {args.record}")
        with open(args.record, 'wb') as f:
            pickle.dump(recorded_observations, f)
        print(f"Recorded observations saved to {args.record}")
        print(f"  Observations: {len(recorded_observations)}")
        print(f"  Duration: {recorded_observations[-1]['timestamp']:.2f}s")


if __name__ == "__main__":
    main()

import math

import pytest
import torch

from mjlab_microduck.tasks import mdp as microduck_mdp


class _FakeEnv:
    """Minimal stand-in: the backflip buffers only need num_envs/device/step_dt."""

    def __init__(self, num_envs=4):
        self.num_envs = num_envs
        self.device = "cpu"
        self.step_dt = 0.02
        self.common_step_counter = 0
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)


class _FakeEntity:
    """Records every write_root_link_*_to_sim call so tests can assert on it."""

    def __init__(self):
        self.pose_calls = []
        self.vel_calls = []

    def write_root_link_pose_to_sim(self, root_pose, env_ids=None):
        self.pose_calls.append((root_pose.clone(), env_ids))

    def write_root_link_velocity_to_sim(self, root_velocity, env_ids=None):
        self.vel_calls.append((root_velocity.clone(), env_ids))


class _FakeTerrain:
    def __init__(self, env_origins):
        self.env_origins = env_origins


class _FakeAssetData:
    """Minimal root-state view: only the fields the backflip mdp functions read.

    Identity quaternion (upright, sagittally flat) and zero velocity by
    default, so a test only has to set the one or two fields it cares about.
    """

    def __init__(self, num_envs):
        self.root_link_ang_vel_b = torch.zeros(num_envs, 3)
        self.root_link_lin_vel_w = torch.zeros(num_envs, 3)
        self.root_link_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * num_envs)
        self.root_link_pos_w = torch.zeros(num_envs, 3)


class _FakeAsset:
    """Stand-in for `env.scene["robot"]`: just carries `.data`."""

    def __init__(self, num_envs):
        self.data = _FakeAssetData(num_envs)


class _FakeSensorData:
    def __init__(self, found):
        self.found = found


class _FakeSensor:
    """Stand-in for a contact sensor: `found` is (num_envs, K), >0 = contact."""

    def __init__(self, found):
        self.data = _FakeSensorData(found)


class _FakeScene:
    def __init__(self, entities, env_origins, sensors=None):
        self._entities = entities
        self.terrain = _FakeTerrain(env_origins)
        self.sensors = sensors if sensors is not None else {}

    def __getitem__(self, name):
        return self._entities[name]


class _FakeEnvWithScene(_FakeEnv):
    """_FakeEnv plus a scene with plate/robot entities and (optional) sensors,
    for tests that exercise backflip_plate_step, the rotation accumulator,
    and the reward functions that read root state / contact sensors."""

    def __init__(self, num_envs=1, env_origins=None, sensors=None):
        super().__init__(num_envs=num_envs)
        if env_origins is None:
            env_origins = torch.zeros(num_envs, 3)
        self.plate = _FakeEntity()
        self.robot = _FakeAsset(num_envs)
        self.scene = _FakeScene(
            {"plate": self.plate, "robot": self.robot}, env_origins, sensors=sensors
        )


def test_state_buffers_are_created_lazily_and_sized_per_env():
    env = _FakeEnv(num_envs=7)
    bufs = microduck_mdp._backflip_state(env)
    assert len(bufs) == 8
    for b in bufs:
        assert b.shape == (7,)


def test_reset_only_touches_the_given_envs():
    env = _FakeEnv(num_envs=4)
    microduck_mdp._backflip_state(env)
    env._backflip_t_hold[:] = 99.0
    microduck_mdp.reset_backflip_launch_params(
        env, torch.tensor([1, 3]), hold_range=(0.2, 0.2)
    )
    assert env._backflip_t_hold[0] == 99.0
    assert env._backflip_t_hold[2] == 99.0
    assert env._backflip_t_hold[1] == 0.2
    assert env._backflip_t_hold[3] == 0.2


def test_reset_clears_the_rotation_accumulator_for_those_envs():
    # No state may survive a reset — an accumulator that carries over would pay
    # the next episode for last episode's rotation.
    env = _FakeEnv(num_envs=3)
    microduck_mdp._backflip_state(env)
    env._backflip_accum[:] = 5.0
    env._backflip_max[:] = 5.0
    env._backflip_paid[:] = 5.0
    microduck_mdp.reset_backflip_launch_params(env, torch.tensor([0, 2]))
    assert env._backflip_accum.tolist() == [0.0, 5.0, 0.0]
    assert env._backflip_max.tolist() == [0.0, 5.0, 0.0]
    assert env._backflip_paid.tolist() == [0.0, 5.0, 0.0]


def test_sampled_params_stay_inside_the_requested_ranges():
    env = _FakeEnv(num_envs=256)
    microduck_mdp.reset_backflip_launch_params(
        env,
        torch.arange(256),
        hold_range=(0.1, 0.4),
        launch_range=(0.08, 0.15),
        z0_range=(0.10, 0.20),
        vz_range=(2.00, 2.25),
        w0_range=(24.0, 30.0),
    )
    assert torch.all((env._backflip_t_hold >= 0.1) & (env._backflip_t_hold <= 0.4))
    assert torch.all((env._backflip_t_launch >= 0.08) & (env._backflip_t_launch <= 0.15))
    assert torch.all((env._backflip_z0 >= 0.10) & (env._backflip_z0 <= 0.20))
    assert torch.all((env._backflip_vz >= 2.00) & (env._backflip_vz <= 2.25))
    assert torch.all((env._backflip_w0 >= 24.0) & (env._backflip_w0 <= 30.0))


def test_sampled_params_use_measured_envelope_defaults():
    # Defaults must be the measured launch box, not the brief's stale placeholders.
    env = _FakeEnv(num_envs=256)
    microduck_mdp.reset_backflip_launch_params(env, torch.arange(256))
    assert torch.all((env._backflip_vz >= 2.00) & (env._backflip_vz <= 2.25))
    assert torch.all((env._backflip_w0 >= 24.0) & (env._backflip_w0 <= 30.0))


def test_phase_of_env_follows_episode_time():
    env = _FakeEnv(num_envs=3)
    microduck_mdp.reset_backflip_launch_params(
        env, torch.arange(3), hold_range=(0.5, 0.5), launch_range=(0.1, 0.1)
    )
    # episode_length_buf counts control steps; step_dt = 0.02 s.
    env.episode_length_buf = torch.tensor([5, 27, 40])  # 0.10 s, 0.54 s, 0.80 s
    phase = microduck_mdp.backflip_phase(env)
    assert phase.tolist() == [
        microduck_mdp.BACKFLIP_PHASE_HOLD,
        microduck_mdp.BACKFLIP_PHASE_LAUNCH,
        microduck_mdp.BACKFLIP_PHASE_GONE,
    ]


def test_plate_step_during_hold_parks_at_z0_with_identity_orientation_and_zero_velocity():
    # Nonzero terrain origin so a missing/mishandled offset is visible.
    origin = torch.tensor([[2.0, 3.0, 1.0]])
    env = _FakeEnvWithScene(num_envs=1, env_origins=origin)
    microduck_mdp.reset_backflip_launch_params(
        env,
        torch.tensor([0]),
        hold_range=(0.5, 0.5),
        launch_range=(0.2, 0.2),
        z0_range=(0.15, 0.15),
        vz_range=(2.0, 2.0),
        w0_range=(24.0, 24.0),
    )
    env.episode_length_buf = torch.tensor([5])  # t = 0.10 s < t_hold = 0.5 s
    microduck_mdp.backflip_plate_step(env)

    pose, pose_ids = env.plate.pose_calls[-1]
    vel, vel_ids = env.plate.vel_calls[-1]

    assert torch.equal(pose_ids, torch.tensor([0]))
    assert torch.equal(vel_ids, torch.tensor([0]))
    assert torch.equal(pose[:, 0:2], origin[:, 0:2])
    assert torch.allclose(pose[:, 2], origin[:, 2] + 0.15)
    assert torch.allclose(pose[:, 3], torch.tensor([1.0]))  # qw = cos(0) = 1
    assert torch.allclose(pose[:, 4], torch.tensor([0.0]))  # qx
    assert torch.allclose(pose[:, 5], torch.tensor([0.0]))  # qy = sin(0) = 0
    assert torch.allclose(pose[:, 6], torch.tensor([0.0]))  # qz
    assert torch.equal(vel, torch.zeros(1, 6))


def test_plate_step_mid_launch_writes_signed_pitch_quat_and_matching_rates():
    origin = torch.tensor([[0.0, 0.0, 1.0]])
    env = _FakeEnvWithScene(num_envs=1, env_origins=origin)
    t_hold, t_launch, z0, vz, w0 = 0.5, 0.2, 0.15, 2.0, 24.0
    microduck_mdp.reset_backflip_launch_params(
        env,
        torch.tensor([0]),
        hold_range=(t_hold, t_hold),
        launch_range=(t_launch, t_launch),
        z0_range=(z0, z0),
        vz_range=(vz, vz),
        w0_range=(w0, w0),
    )
    tau = 0.1  # 0.1 s into the 0.2 s ramp
    t = t_hold + tau  # 0.6 s
    env.episode_length_buf = torch.tensor([round(t / env.step_dt)])  # 30 steps
    microduck_mdp.backflip_plate_step(env)

    # Independently derived (constant-acceleration ramp), NOT obtained by
    # calling backflip_plate_kinematics -- this is what pins the WIRING in
    # backflip_plate_step (axis indices, quaternion construction, offset),
    # not the kinematics function itself (already pinned elsewhere).
    a_lin = vz / t_launch  # 10.0
    # w0 > 0 is the caller-facing "backward" knob; a backward roll is a
    # NEGATIVE rotation about +y in this codebase's convention, so the
    # angular accel is negative for positive w0.
    a_ang = -w0 / t_launch  # -120.0
    expected_z = z0 + 0.5 * a_lin * tau * tau  # 0.20
    expected_pitch = 0.5 * a_ang * tau * tau  # -0.6
    expected_vz_t = a_lin * tau  # 1.0
    expected_w_t = a_ang * tau  # -12.0

    pose, _ = env.plate.pose_calls[-1]
    vel, _ = env.plate.vel_calls[-1]

    assert torch.allclose(pose[:, 0:2], origin[:, 0:2])
    assert torch.allclose(pose[:, 2], origin[:, 2] + expected_z, atol=1e-4)
    assert torch.allclose(
        pose[:, 3], torch.tensor([math.cos(expected_pitch / 2)]), atol=1e-5
    )
    assert torch.allclose(pose[:, 4], torch.tensor([0.0]), atol=1e-8)  # qx stays 0
    assert torch.allclose(
        pose[:, 5], torch.tensor([math.sin(expected_pitch / 2)]), atol=1e-5
    )
    assert torch.allclose(pose[:, 6], torch.tensor([0.0]), atol=1e-8)  # qz stays 0
    # Sign pin: expected_pitch is negative, so qy = sin(pitch/2) must be
    # negative too. A build that dropped the kinematics minus sign, or
    # negated pitch while building the quaternion, would write +sin(0.3)
    # here instead of -sin(0.3) -- this assertion fails under that bug.
    assert pose[0, 5].item() < 0.0

    assert torch.allclose(vel[:, 2], torch.tensor([expected_vz_t]), atol=1e-4)
    assert torch.allclose(vel[:, 4], torch.tensor([expected_w_t]), atol=1e-3)
    # Sign pin on the angular-rate channel too: w0 > 0 => backward => w_t < 0.
    assert vel[0, 4].item() < 0.0
    assert torch.allclose(vel[:, [0, 1, 3, 5]], torch.zeros(1, 4))


def test_plate_step_once_gone_parks_above_and_beside_the_floor_with_zero_velocity():
    origin = torch.tensor([[5.0, -2.0, 0.5]])
    env = _FakeEnvWithScene(num_envs=1, env_origins=origin)
    microduck_mdp.reset_backflip_launch_params(
        env,
        torch.tensor([0]),
        hold_range=(0.1, 0.1),
        launch_range=(0.05, 0.05),
        z0_range=(0.15, 0.15),
        vz_range=(2.0, 2.0),
        w0_range=(24.0, 24.0),
    )
    env.episode_length_buf = torch.tensor([100])  # 2.0 s, well past hold + launch
    microduck_mdp.backflip_plate_step(env)

    pose, _ = env.plate.pose_calls[-1]
    vel, _ = env.plate.vel_calls[-1]

    # WAS: parked at origin x/y and BACKFLIP_GONE_Z = -3.0, i.e. 3 m INSIDE an
    # infinite ground plane — maximal penetration, not absence (4 spurious
    # contacts per env per step, and the solver ejecting the 50 kg plate at
    # 59 m/s between step-event writes). Now parked above and beside the floor.
    gx, gy, gz = microduck_mdp.BACKFLIP_GONE_POS
    assert torch.allclose(pose[:, 0], origin[:, 0] + gx)
    assert torch.allclose(pose[:, 1], origin[:, 1] + gy)
    assert torch.allclose(pose[:, 2], origin[:, 2] + gz)
    assert torch.allclose(pose[:, 3], torch.tensor([1.0]))  # identity: no rotation once gone
    assert torch.allclose(pose[:, 5], torch.tensor([0.0]))
    assert torch.equal(vel, torch.zeros(1, 6))


def test_the_parked_plate_is_clear_of_the_floor_and_far_from_the_env_origin():
    """The property that matters, stated directly on the constant.

    ``terrain_type="plane"`` is an INFINITE half-space at the env origin's z, so
    "out of the way" can only mean ABOVE it: any negative parking z is
    penetration depth. The plate is an 18x18x2 cm box, so its underside sits
    one half-thickness below the parked z. The measured flight envelope apexes
    at 0.90 m, which is the highest anything in the scene ever gets.
    """
    gx, gy, gz = microduck_mdp.BACKFLIP_GONE_POS
    plate_half_thickness = 0.01
    assert gz - plate_half_thickness > 0.90, "parked plate must clear the flight apex"
    assert math.hypot(gx, gy) > 1.0, "parked plate must be well off the env origin"


def test_the_plate_only_moves_laterally_once_it_is_gone():
    """HOLD and LAUNCH must stay over the env origin — the robot stands on it."""
    origin = torch.tensor([[5.0, -2.0, 0.5], [5.0, -2.0, 0.5]])
    env = _FakeEnvWithScene(num_envs=2, env_origins=origin)
    microduck_mdp.reset_backflip_launch_params(
        env,
        torch.tensor([0, 1]),
        hold_range=(0.5, 0.5),
        launch_range=(0.1, 0.1),
        z0_range=(0.15, 0.15),
        vz_range=(2.0, 2.0),
        w0_range=(24.0, 24.0),
    )
    for buf in (torch.tensor([0, 0]), torch.tensor([27, 27])):  # HOLD, LAUNCH
        env.episode_length_buf = buf
        microduck_mdp.backflip_plate_step(env)
        pose, _ = env.plate.pose_calls[-1]
        assert torch.allclose(pose[:, 0:2], origin[:, 0:2]), buf


def test_completion_gate_is_closed_below_and_open_above():
    env = _FakeEnv(num_envs=3)
    microduck_mdp._backflip_state(env)
    env._backflip_max = torch.tensor(
        [math.radians(90.0), math.radians(322.5), math.radians(359.0)]
    )
    gate = microduck_mdp._backflip_completion_gate(
        env,
        microduck_mdp.BACKFLIP_LANDING_GATE_LO,
        microduck_mdp.BACKFLIP_LANDING_GATE_HI,
    )
    assert gate[0] == 0.0                      # a quarter turn earns no landing
    assert 0.4 < float(gate[1]) < 0.6          # mid-gate smoothstep
    assert gate[2] == 1.0                      # a full flip opens it


def test_progress_pays_only_new_frontier():
    # Potential-based: re-reaching an angle already paid for earns nothing, so
    # rocking back and forth cannot farm it.
    env = _FakeEnv(num_envs=1)
    microduck_mdp._backflip_state(env)
    env._backflip_max = torch.tensor([1.0])
    first = microduck_mdp._backflip_pay(env, target_angle=2 * math.pi, max_paid_rate=14.0)
    second = microduck_mdp._backflip_pay(env, target_angle=2 * math.pi, max_paid_rate=14.0)
    assert float(first) > 0.0
    assert float(second) == 0.0


def test_progress_is_rate_capped():
    # Spinning faster than the cap FORFEITS the excess: a more violent flip
    # collects less, not the same amount sooner.
    env = _FakeEnv(num_envs=1)
    microduck_mdp._backflip_state(env)
    env._backflip_max = torch.tensor([6.0])   # huge jump in one step
    paid = microduck_mdp._backflip_pay(env, target_angle=2 * math.pi, max_paid_rate=14.0)
    # capped at max_paid_rate * step_dt = 14 * 0.02 = 0.28 rad of paid rotation
    assert float(paid) <= 0.28 / (env.step_dt * 2 * math.pi) + 1e-6


def test_progress_never_exceeds_one_full_turn_in_total():
    env = _FakeEnv(num_envs=1)
    microduck_mdp._backflip_state(env)
    total = 0.0
    for _ in range(500):
        env._backflip_max += 0.05
        total += float(
            microduck_mdp._backflip_pay(env, target_angle=2 * math.pi, max_paid_rate=1e9)
        ) * env.step_dt
    assert total <= 1.0 + 1e-6   # normalized: a full flip pays 1.0 in total


def test_landing_pays_nothing_before_the_flip_is_complete():
    # The anti-jackpot rule: an upright robot that never flipped must earn 0
    # from the landing term, or "stand still on the plate" becomes the argmax.
    env = _FakeEnv(num_envs=1)
    microduck_mdp._backflip_state(env)
    env._backflip_max = torch.tensor([math.radians(45.0)])
    gate = microduck_mdp._backflip_completion_gate(
        env,
        microduck_mdp.BACKFLIP_LANDING_GATE_LO,
        microduck_mdp.BACKFLIP_LANDING_GATE_HI,
    )
    assert float(gate) == 0.0


def test_ready_stance_window_is_hold_only():
    env = _FakeEnv(num_envs=3)
    microduck_mdp.reset_backflip_launch_params(
        env, torch.arange(3), hold_range=(0.5, 0.5), launch_range=(0.1, 0.1)
    )
    env.episode_length_buf = torch.tensor([5, 27, 40])
    window = microduck_mdp.backflip_hold_window(env)
    assert window.tolist() == [1.0, 0.0, 0.0]


# --- Anti-farming mechanisms, pinned end-to-end through the public/private
# accumulator and reward entry points (not just via hand-set _backflip_max). ---


def test_accum_backward_rotation_airborne_accumulates_but_grounded_does_not():
    # The floor-roll farm: without the airborne gate, flopping onto the back
    # and rolling along the floor would pay exactly like an honest airborne
    # flip. omega_y = -10 body-frame -> _BACKFLIP_BWD_SIGN(-1) * (-10) = +10,
    # i.e. a genuine backward rotation.
    airborne = _FakeEnvWithScene(
        num_envs=1,
        sensors={"robot_ground_contact": _FakeSensor(torch.zeros(1, 1))},
    )
    microduck_mdp._backflip_state(airborne)
    airborne.robot.data.root_link_ang_vel_b[:, 1] = -10.0
    microduck_mdp._update_backflip_accum(airborne, airborne.robot)
    assert float(airborne._backflip_accum[0]) > 0.0

    grounded = _FakeEnvWithScene(
        num_envs=1,
        sensors={"robot_ground_contact": _FakeSensor(torch.ones(1, 1))},
    )
    microduck_mdp._backflip_state(grounded)
    grounded.robot.data.root_link_ang_vel_b[:, 1] = -10.0
    microduck_mdp._update_backflip_accum(grounded, grounded.robot)
    assert float(grounded._backflip_accum[0]) == 0.0


def test_accum_forward_rotation_does_not_move_the_frontier():
    # omega_y = +10 body-frame -> _BACKFLIP_BWD_SIGN(-1) * 10 = -10: a
    # forward roll must not advance the paid-rotation high-water mark.
    env = _FakeEnvWithScene(
        num_envs=1,
        sensors={"robot_ground_contact": _FakeSensor(torch.zeros(1, 1))},
    )
    microduck_mdp._backflip_state(env)
    env.robot.data.root_link_ang_vel_b[:, 1] = 10.0
    microduck_mdp._update_backflip_accum(env, env.robot)
    assert float(env._backflip_max[0]) == 0.0


def test_accum_sagittal_flatness_gates_a_sideways_tumble_to_zero():
    # Same backward rate, two orientations: identity (sagittally flat) must
    # accumulate fully; rotated 90 deg about the body's own +x (a sideways
    # tumble -- the lateral/y body axis now points along world z) must be
    # gated to nothing, or a shoulder-roll counts as a backflip.
    flat = _FakeEnvWithScene(
        num_envs=1, sensors={"robot_ground_contact": _FakeSensor(torch.zeros(1, 1))}
    )
    microduck_mdp._backflip_state(flat)
    flat.robot.data.root_link_ang_vel_b[:, 1] = -10.0
    microduck_mdp._update_backflip_accum(flat, flat.robot)
    flat_delta = float(flat._backflip_accum[0])
    assert flat_delta > 0.0

    tumble = _FakeEnvWithScene(
        num_envs=1, sensors={"robot_ground_contact": _FakeSensor(torch.zeros(1, 1))}
    )
    microduck_mdp._backflip_state(tumble)
    tumble.robot.data.root_link_ang_vel_b[:, 1] = -10.0
    tumble.robot.data.root_link_quat_w[:] = torch.tensor(
        [[math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0]]
    )
    microduck_mdp._update_backflip_accum(tumble, tumble.robot)
    assert float(tumble._backflip_accum[0]) == 0.0


def test_progress_pays_zero_camping_and_positive_once_the_frontier_advances():
    env = _FakeEnvWithScene(
        num_envs=1, sensors={"robot_ground_contact": _FakeSensor(torch.zeros(1, 1))}
    )
    microduck_mdp._backflip_state(env)
    # Camping: zero angular velocity, any pose -- no rotation, so no pay.
    camped = microduck_mdp.backflip_progress(env)
    assert float(camped[0]) == 0.0

    env.common_step_counter += 1  # new control step: lift the step-guard
    env.robot.data.root_link_ang_vel_b[:, 1] = -10.0
    advancing = microduck_mdp.backflip_progress(env)
    assert float(advancing[0]) > 0.0


def test_landing_pays_zero_when_gate_closed_however_upright_and_well_placed():
    # However good the pose, the completion gate at max_accum=0 must still
    # zero the whole multiplicative composite.
    env = _FakeEnvWithScene(
        num_envs=1,
        sensors={"feet_ground_contact": _FakeSensor(torch.ones(1, 1))},
    )
    microduck_mdp._backflip_state(env)
    env._backflip_max[:] = 0.0
    env.robot.data.root_link_pos_w[:, 2] = 0.115  # exactly at stand_z
    reward = microduck_mdp.backflip_landing(env)
    assert float(reward[0]) == 0.0


def test_landing_does_not_pay_on_a_bounce_with_high_linear_velocity():
    # Anti-bounce-farming: gate x feet x upright x height can all be
    # satisfied momentarily during a hard-landing rebound while the trunk is
    # still carrying several m/s of translational velocity (the "omega" calm
    # factor alone is blind to this -- angular rate can be near zero while
    # the robot is bouncing). Without a linear-velocity factor, bouncing
    # collects close to full reward on every bounce, which is cheaper for a
    # policy to discover than actually settling into a stand -- and it's
    # also exactly the kind of landing that damages the real hardware.
    env = _FakeEnvWithScene(
        num_envs=1,
        sensors={"feet_ground_contact": _FakeSensor(torch.ones(1, 1))},
    )
    microduck_mdp._backflip_state(env)
    env._backflip_max[:] = math.radians(360.0)  # gate fully open
    env.robot.data.root_link_pos_w[:, 2] = 0.115  # exactly at stand_z
    env.robot.data.root_link_lin_vel_w[:, 2] = 3.0  # m/s: mid-bounce rebound
    reward = microduck_mdp.backflip_landing(env)
    assert float(reward[0]) < 0.05


# --- The two accumulator clamps (fix wave: CRITICAL 2 + IMPORTANT 3). ---------


def _accum_env(num_envs=1):
    """Env whose ground-contact sensor tensor can be flipped between steps."""
    found = torch.zeros(num_envs, 1)
    env = _FakeEnvWithScene(
        num_envs=num_envs,
        sensors={"robot_ground_contact": _FakeSensor(found)},
    )
    microduck_mdp._backflip_state(env)
    return env, found


def _accum_steps(env, found, n, omega_y, contact):
    """Advance the accumulator n control steps at a fixed rate and contact state.

    omega_y is the BODY-frame lateral rate: negative = backward (a real flip),
    positive = forward. common_step_counter must move or the step guard makes
    every extra call a no-op.
    """
    env.robot.data.root_link_ang_vel_b[:, 1] = omega_y
    found[:] = 1.0 if contact else 0.0
    for _ in range(n):
        env.common_step_counter += 1
        microduck_mdp._update_backflip_accum(env, env.robot)


def test_accum_is_floored_at_zero_so_a_forward_arc_digs_no_hole():
    # A launch that comes out FORWARD is not hypothetical: the measured
    # standing-spawn envelope reaches -275 deg. Unfloored, the signed
    # accumulator would sit at about -4.8 rad while the frontier stayed at 0,
    # and the policy would have to buy that back before the only dense term in
    # the task produced any signal at all.
    env, found = _accum_env()
    _accum_steps(env, found, 30, omega_y=+10.0, contact=False)  # forward, airborne
    assert float(env._backflip_accum[0]) == 0.0
    assert float(env._backflip_max[0]) == 0.0


def test_a_forward_launch_then_a_real_backward_flip_still_earns_progress():
    # The regression CRITICAL 2 asks for: the forward half must not mortgage
    # the backward half. 30 steps at 10 rad/s backward = 6.0 rad of frontier,
    # which is what a fresh accumulator would have earned.
    env, found = _accum_env()
    _accum_steps(env, found, 30, omega_y=+10.0, contact=False)
    _accum_steps(env, found, 30, omega_y=-10.0, contact=False)
    earned = float(env._backflip_max[0])
    assert abs(earned - 30 * 10.0 * env.step_dt) < 1e-4

    fresh, fresh_found = _accum_env()
    _accum_steps(fresh, fresh_found, 30, omega_y=-10.0, contact=False)
    assert abs(earned - float(fresh._backflip_max[0])) < 1e-6

    # And it actually pays: progress is potential-based off the frontier.
    paid = microduck_mdp._backflip_pay(env, 2 * math.pi, 1e9)
    assert float(paid[0]) > 0.0


def test_terrain_contact_resets_the_arc_but_keeps_the_frontier():
    env, found = _accum_env()
    _accum_steps(env, found, 10, omega_y=-10.0, contact=False)
    banked = float(env._backflip_accum[0])
    assert banked > 0.0
    assert abs(float(env._backflip_max[0]) - banked) < 1e-9

    _accum_steps(env, found, 1, omega_y=-10.0, contact=True)
    assert float(env._backflip_accum[0]) == 0.0            # the arc ended
    assert abs(float(env._backflip_max[0]) - banked) < 1e-9  # the peak survives


def test_airborne_bank_grounded_unwind_cycling_does_not_advance_the_frontier():
    # THE RATCHET, which zeroing delta on contact (rather than resetting the
    # accumulator) left open: hop, nod ~23 deg backward in the air, land,
    # unwind it on the ground for free, repeat. ~11 cycles used to reach the
    # landing gate's 300 deg with no flip anywhere, and then collect the
    # landing annuity. One continuous arc is the fix.
    env, found = _accum_env()
    bank_steps, bank_rate = 2, -10.0            # 0.4 rad = ~23 deg per hop
    for _ in range(11):
        _accum_steps(env, found, bank_steps, omega_y=bank_rate, contact=False)
        _accum_steps(env, found, 4, omega_y=+10.0, contact=True)   # grounded unwind

    one_hop = bank_steps * abs(bank_rate) * env.step_dt
    frontier = float(env._backflip_max[0])
    assert abs(frontier - one_hop) < 1e-6
    assert frontier < math.radians(30.0)
    # Nowhere near the landing gate, so the annuity stays shut.
    gate = microduck_mdp._backflip_completion_gate(
        env,
        microduck_mdp.BACKFLIP_LANDING_GATE_LO,
        microduck_mdp.BACKFLIP_LANDING_GATE_HI,
    )
    assert float(gate[0]) == 0.0


def test_one_continuous_arc_of_the_same_total_rotation_does_advance_it():
    # The control for the ratchet test: identical per-step rotation, flown in
    # one arc instead of 11 hops, must clear the landing gate. The fix has to
    # refuse the exploit without refusing the maneuver.
    env, found = _accum_env()
    _accum_steps(env, found, 32, omega_y=-10.0, contact=False)   # 6.4 rad = 367 deg
    assert float(env._backflip_max[0]) > microduck_mdp.BACKFLIP_LANDING_GATE_HI
    gate = microduck_mdp._backflip_completion_gate(
        env,
        microduck_mdp.BACKFLIP_LANDING_GATE_LO,
        microduck_mdp.BACKFLIP_LANDING_GATE_HI,
    )
    assert float(gate[0]) == 1.0


# --- Fix-wave additions: the arrival-damper window, the reset-time plate
# placement, and the GONE-masked critic plate observations. -------------------


class _StubAssetCfg:
    """SceneEntityCfg stand-in: name/body_ids are all the function reads."""

    name = "robot"
    body_ids = [0]


class _FakePlateData:
    def __init__(self, pos, vel):
        self.root_link_pos_w = torch.tensor([pos])
        self.root_link_lin_vel_w = torch.tensor([vel])


def _arrival_gate_at(z):
    """body_ang_vel_at_height at the backflip cfg's own numbers, unit omega_x."""
    env = _FakeEnvWithScene(num_envs=1)
    env.robot.data.root_link_pos_w[:, 2] = z
    env.robot.data.body_link_ang_vel_w = torch.zeros(1, 1, 3)
    env.robot.data.body_link_ang_vel_w[:, 0, 0] = 1.0
    return float(
        microduck_mdp.body_ang_vel_at_height(
            env,
            height_low=0.09,
            height_high=0.11,
            asset_cfg=_StubAssetCfg(),
            tilt_full_deg=20.0,
            tilt_zero_deg=45.0,
            height_full_max=0.16,
            height_zero_max=0.22,
        )[0]
    )


def test_arrival_damper_is_free_in_flight_and_active_at_standing_height():
    # The bug this pins: height_low/height_high alone is a FLOOR, so the whole
    # flight of a backflip (apex 0.4-1.2 m measured) paid FULL cost while the
    # comment claimed "the flip itself is never taxed". Only the tilt gate was
    # protecting it, and a rotating robot passes tilt < 20 deg twice per turn.
    assert _arrival_gate_at(0.115) > 0.9   # settled at standing height: damped
    assert _arrival_gate_at(0.50) == 0.0   # mid-flight: free
    assert _arrival_gate_at(1.00) == 0.0   # high apex: free
    assert _arrival_gate_at(0.275) == 0.0  # standing on the plate in HOLD: free
    assert _arrival_gate_at(0.05) == 0.0   # down on the ground: free


def test_arrival_damper_without_a_ceiling_is_still_a_floor():
    # Guard against someone "simplifying" the ceiling away: with no upper edge
    # the gate is wide open at flight altitude.
    env = _FakeEnvWithScene(num_envs=1)
    env.robot.data.root_link_pos_w[:, 2] = 0.50
    env.robot.data.body_link_ang_vel_w = torch.zeros(1, 1, 3)
    env.robot.data.body_link_ang_vel_w[:, 0, 0] = 1.0
    no_ceiling = float(
        microduck_mdp.body_ang_vel_at_height(
            env, height_low=0.09, height_high=0.11, asset_cfg=_StubAssetCfg()
        )[0]
    )
    assert no_ceiling == 1.0


def test_arrival_damper_ceiling_needs_both_edges():
    env = _FakeEnvWithScene(num_envs=1)
    env.robot.data.body_link_ang_vel_w = torch.zeros(1, 1, 3)
    with pytest.raises(ValueError):
        microduck_mdp.body_ang_vel_at_height(
            env, height_low=0.09, height_high=0.11,
            asset_cfg=_StubAssetCfg(), height_full_max=0.16,
        )


def test_plate_reset_places_the_plate_at_z0_despite_a_stale_episode_buffer():
    # episode_length_buf[env_ids] = 0 runs AFTER reset events, so at
    # reset-event time _backflip_time returns the TERMINAL episode's time and
    # the phase reads GONE. Benign under auto_reset (the step event re-places
    # the plate in the same step()), wrong for one control step after a manual
    # reset(env_ids=...) — which play/eval use.
    env = _FakeEnvWithScene(num_envs=2)
    microduck_mdp.reset_backflip_launch_params(
        env, torch.arange(2), hold_range=(0.3, 0.3), launch_range=(0.1, 0.1),
        z0_range=(0.17, 0.17), vz_range=(2.0, 2.0), w0_range=(24.0, 24.0),
    )
    env.episode_length_buf[:] = 199          # a whole terminal episode of steps

    microduck_mdp.backflip_plate_step(env, asset_name="plate", t_override=0.0)
    pose, _ = env.plate.pose_calls[-1]
    assert abs(float(pose[0, 2]) - 0.17) < 1e-6
    assert float(pose[0, 0]) == 0.0          # not the +5 m parking spot
    assert float(pose[0, 1]) == 0.0
    vel, _ = env.plate.vel_calls[-1]
    assert float(vel.abs().sum()) == 0.0     # parked, still

    # Without the override the stale buffer really does park it away: that is
    # the bug, kept here so the fix cannot be quietly reverted.
    microduck_mdp.backflip_plate_step(env, asset_name="plate")
    stale, _ = env.plate.pose_calls[-1]
    assert float(stale[0, 0]) == microduck_mdp.BACKFLIP_GONE_POS[0]


def _plate_obs_env():
    env = _FakeEnvWithScene(num_envs=1)
    env.plate.data = _FakePlateData((5.0, 5.0, 5.0), (1.0, 2.0, 3.0))
    microduck_mdp.reset_backflip_launch_params(
        env, torch.arange(1), hold_range=(0.3, 0.3), launch_range=(0.1, 0.1),
        z0_range=(0.15, 0.15), vz_range=(2.0, 2.0), w0_range=(24.0, 24.0),
    )
    return env


def test_critic_plate_obs_are_zeroed_once_the_plate_is_gone():
    # Not about hiding information — about the obs normalizer. The plate is
    # parked ~8.7 m from the robot for ~85% of every episode's steps, so an
    # unmasked plate_position normalizer converges to std ~3 m and squashes the
    # informative HOLD/LAUNCH range (0-0.3 m) below 0.1 normalized units,
    # destroying the z0 signal the value function needs.
    env = _plate_obs_env()
    env.episode_length_buf[:] = 0                       # HOLD: plate present
    assert float(microduck_mdp.backflip_plate_pos_obs(env).abs().sum()) > 0.0
    assert float(microduck_mdp.backflip_plate_vel_obs(env).abs().sum()) > 0.0

    env.episode_length_buf[:] = 100                     # GONE: masked
    assert float(microduck_mdp.backflip_plate_pos_obs(env).abs().sum()) == 0.0
    assert float(microduck_mdp.backflip_plate_vel_obs(env).abs().sum()) == 0.0


def test_masked_plate_obs_keep_the_unmasked_shape_and_values_while_present():
    env = _plate_obs_env()
    env.episode_length_buf[:] = 0
    masked = microduck_mdp.backflip_plate_pos_obs(env)
    raw = microduck_mdp.ball_pos_in_base(env, asset_name="plate")
    assert masked.shape == raw.shape == (1, 3)
    assert torch.allclose(masked, raw)

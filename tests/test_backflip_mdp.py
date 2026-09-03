import math

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

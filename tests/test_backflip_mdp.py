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

from mjlab_microduck.tasks.microduck_standard_stairs_env_cfg import (
    MicroduckStairPhaseBalancedRsiRlCfg,
    make_microduck_stair_phase_balanced_rsi_env_cfg,
)


def test_phase_balanced_rsi_replays_four_roll_phases():
    cfg = make_microduck_stair_phase_balanced_rsi_env_cfg()
    bank = cfg.events["walker_state_bank"].params

    assert bank["bank_path"].endswith("full170-roulade-state-bank.pt")
    assert bank["phase_balanced"] is True
    assert bank["phase_bucket_count"] == 4
    assert bank["source_episode_step_range"] == (15, 60)
    assert cfg.episode_length_s == 8.0
    assert cfg.rewards["stair_first_tread_secured"].weight == 400.0
    assert MicroduckStairPhaseBalancedRsiRlCfg.max_iterations == 400
    assert MicroduckStairPhaseBalancedRsiRlCfg.save_interval == 25

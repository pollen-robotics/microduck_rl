"""Config invariants for the PoliteBow episodic task."""

from mjlab_microduck.tasks.microduck_polite_bow_env_cfg import (
    BOW_TARGET_LEAN,
    EPISODE_LENGTH_S,
    make_microduck_polite_bow_env_cfg,
)


def test_polite_bow_cfg_registers_task_rewards():
    cfg = make_microduck_polite_bow_env_cfg()
    r = cfg.rewards
    assert "bow_progress" in r and r["bow_progress"].weight == 8.0
    assert r["bow_progress"].params["target_lean"] == BOW_TARGET_LEAN
    assert "bow_hold" in r
    assert "bow_stand_after" in r
    assert "feet_grounded" in r and r["feet_grounded"].weight == 3.0
    assert "head_tripod" in r and r["head_tripod"].weight == -5.0
    assert cfg.episode_length_s == EPISODE_LENGTH_S


def test_polite_bow_hard_gates_present():
    cfg = make_microduck_polite_bow_env_cfg()
    sensor_names = {s.name for s in cfg.scene.sensors}
    assert "feet_ground_contact" in sensor_names
    assert "head_ground_contact" in sensor_names
    assert "nan_state" in cfg.terminations
    assert "set_polite_bow_state" in cfg.events


def test_polite_bow_play_variant_builds():
    cfg = make_microduck_polite_bow_env_cfg(play=True)
    assert "bow_progress" in cfg.rewards


def test_polite_bow_zero_pads_command_slots():
    cfg = make_microduck_polite_bow_env_cfg()
    for group in ("actor", "critic"):
        assert "head_command" in cfg.observations[group].terms
        assert "body_command" in cfg.observations[group].terms
        assert cfg.observations[group].terms["head_command"].params["dim"] == 4
        assert cfg.observations[group].terms["body_command"].params["dim"] == 6

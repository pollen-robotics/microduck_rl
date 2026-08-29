from mjlab.tasks.registry import list_tasks

from mjlab_microduck.tasks.growbot_roulade_env_cfg import (
    GrowbotRouladeRlCfg,
    make_growbot_roulade_env_cfg,
)


def test_growbot_env_builds_train_and_play_configs():
    assert make_growbot_roulade_env_cfg() is not None
    assert make_growbot_roulade_env_cfg(play=True) is not None


def test_growbot_task_uses_16_actuator_footed_asset():
    cfg = make_growbot_roulade_env_cfg()
    model = cfg.scene.entities["robot"].spec_fn().compile()

    assert model.nu == 16
    assert cfg.actions["joint_pos"].scale == 1.0
    assert cfg.scene.terrain.terrain_type == "plane"
    assert all("wheel" not in model.joint(index).name for index in range(model.njnt))


def test_growbot_has_separate_experiment_name():
    assert GrowbotRouladeRlCfg.experiment_name == "growbot_roulade"
    assert GrowbotRouladeRlCfg.run_name == "growbot_footed_16dof"


def test_growbot_task_is_registered():
    import mjlab_microduck.tasks  # noqa: F401

    assert "Mjlab-Growbot-Roulade-Flat" in list_tasks()

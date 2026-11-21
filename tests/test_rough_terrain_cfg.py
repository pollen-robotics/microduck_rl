"""Rough task registration must not let play configs change training terrain.

Regression coverage for pollen-robotics/microduck_rl#65, reported by
notwitcheer. These tests build configs on CPU; they do not train a policy.
"""

import pytest

from mjlab.tasks.registry import load_env_cfg
from mjlab_microduck.tasks.microduck_ground_pick_env_cfg import (
    make_microduck_ground_pick_env_cfg,
)
from mjlab_microduck.tasks.microduck_sitstand_env_cfg import (
    make_microduck_sitstand_env_cfg,
)
from mjlab_microduck.tasks.microduck_standup_env_cfg import (
    make_microduck_standup_env_cfg,
)
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    MICRODUCK_ROUGH_TERRAINS_CFG,
    make_microduck_velocity_env_cfg,
)
from mjlab_microduck.tasks.microduck_velstand_env_cfg import (
    make_microduck_velstand_env_cfg,
)


FAMILIES = [
    pytest.param("Velocity", make_microduck_velocity_env_cfg, True, id="velocity"),
    pytest.param("VelStand", make_microduck_velstand_env_cfg, True, id="velstand"),
    pytest.param("SitStand", make_microduck_sitstand_env_cfg, False, id="sitstand"),
    pytest.param("StandUp", make_microduck_standup_env_cfg, False, id="standup"),
    pytest.param(
        "GroundPick", make_microduck_ground_pick_env_cfg, False, id="ground-pick"
    ),
]


def _terrain(cfg):
    return cfg.scene.terrain.terrain_generator


def test_task_registration_preserves_the_shared_template():
    # Importing any task module above also registers both train and play configs.
    assert MICRODUCK_ROUGH_TERRAINS_CFG.num_rows == 10
    assert MICRODUCK_ROUGH_TERRAINS_CFG.num_cols == 20
    assert MICRODUCK_ROUGH_TERRAINS_CFG.curriculum is False


@pytest.mark.parametrize("family,factory,training_curriculum", FAMILIES)
@pytest.mark.parametrize("play_first", [False, True], ids=["train-first", "play-first"])
def test_factory_terrain_is_independent_of_creation_order(
    family, factory, training_curriculum, play_first
):
    configs = {
        play: _terrain(factory(play=play, rough=True))
        for play in (play_first, not play_first)
    }
    train, play = configs[False], configs[True]
    assert train is not play, family
    assert train is not MICRODUCK_ROUGH_TERRAINS_CFG, family
    assert play is not MICRODUCK_ROUGH_TERRAINS_CFG, family
    assert (train.num_rows, train.num_cols) == (10, 20), family
    assert train.curriculum is training_curriculum, family
    assert (play.num_rows, play.num_cols, play.curriculum) == (5, 5, False), family


@pytest.mark.parametrize("family,factory,training_curriculum", FAMILIES)
def test_nested_terrain_edits_do_not_leak_to_other_configs(
    family, factory, training_curriculum, monkeypatch
):
    first = _terrain(factory(rough=True))
    second = _terrain(factory(rough=True))
    original = MICRODUCK_ROUGH_TERRAINS_CFG.sub_terrains[
        "pyramid_stairs"
    ].step_height_range
    changed = (0.0, original[1] + 0.01)

    # A shallow copy would still share the individual sub-terrain configs.
    # monkeypatch restores the field even when run against the broken baseline.
    monkeypatch.setattr(
        first.sub_terrains["pyramid_stairs"], "step_height_range", changed
    )
    assert second.sub_terrains["pyramid_stairs"].step_height_range == original, family
    assert (
        MICRODUCK_ROUGH_TERRAINS_CFG.sub_terrains["pyramid_stairs"].step_height_range
        == original
    )


@pytest.mark.parametrize("family,factory,training_curriculum", FAMILIES)
@pytest.mark.parametrize("backlash", [False, True], ids=["base", "backlash"])
def test_registered_tasks_preserve_training_and_play_terrain(
    family, factory, training_curriculum, backlash
):
    suffix = "-Backlash" if backlash else ""
    task_id = f"Mjlab-{family}-Rough{suffix}-MicroDuck"
    train = _terrain(load_env_cfg(task_id))
    play = _terrain(load_env_cfg(task_id, play=True))

    assert (train.num_rows, train.num_cols) == (10, 20), task_id
    assert train.curriculum is training_curriculum, task_id
    assert (play.num_rows, play.num_cols, play.curriculum) == (5, 5, False), task_id

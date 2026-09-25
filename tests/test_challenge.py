"""A challenge folder (in microduck-challenges) carries its contract in challenge.toml, beside
its tasks.py: the Arena event, the mjlab task, the kind publish writes, the first recipe, and the
knobs a novice may tune. tasks.py registers it; env.py reads the knobs. Neither the workshop nor
a shell user passes any of it on a command line."""

from pathlib import Path

import pytest

from mjlab_microduck import challenge as ch

TOML = """
event = "sprint-2m"
task  = "Mjlab-Sprint2m-MicroDuck"
kind  = "perpetual"

[recipe]
num_envs       = 4096
max_iterations = 3000
seed           = 1

[params.forward_reward]
label = "Speed reward"
value = 2.0
min   = 0.5
max   = 5.0

[params.upright_reward]
label = "Stay upright"
value = 1
"""


@pytest.fixture(autouse=True)
def _clean_registry(monkeypatch):
    monkeypatch.setattr(ch, "_REGISTRY", {})


def _folder(tmp_path: Path, toml: str = TOML) -> Path:
    folder = tmp_path / "sprint_2m"
    folder.mkdir()
    (folder / "challenge.toml").write_text(toml)
    (folder / "tasks.py").write_text("")
    (folder / "env.py").write_text("")
    return folder


def test_load_reads_the_contract(tmp_path):
    folder = _folder(tmp_path)
    got = ch.load(folder / "tasks.py")
    assert got == ch.Challenge(
        folder=folder, event="sprint-2m", task="Mjlab-Sprint2m-MicroDuck", kind="perpetual",
        recipe={"num_envs": 4096, "max_iterations": 3000, "seed": 1},
    )


def test_register_makes_the_task_findable(tmp_path):
    folder = _folder(tmp_path)
    assert ch.for_task("Mjlab-Sprint2m-MicroDuck") is None
    ch.register(folder / "tasks.py")
    assert ch.for_task("Mjlab-Sprint2m-MicroDuck").event == "sprint-2m"
    assert ch.for_task("Mjlab-Velocity-Flat-MicroDuck") is None, "a library task belongs to no challenge"


def test_params_reads_each_knobs_value_as_a_float(tmp_path):
    assert ch.params(_folder(tmp_path) / "env.py") == {"forward_reward": 2.0, "upright_reward": 1.0}


def test_a_challenge_without_knobs_or_recipe_has_none(tmp_path):
    folder = _folder(tmp_path, 'event = "sprint-2m"\ntask = "T"\nkind = "perpetual"\n')
    assert ch.params(folder / "env.py") == {}
    assert ch.load(folder / "tasks.py").recipe == {}


def test_a_missing_file_is_named(tmp_path):
    with pytest.raises(FileNotFoundError, match="challenge.toml"):
        ch.load(tmp_path / "elsewhere" / "tasks.py")


@pytest.mark.parametrize("toml, why", [
    ('task = "T"\nkind = "perpetual"\n', "event"),
    ('event = "e"\nkind = "perpetual"\n', "task"),
    ('event = "e"\ntask = "T"\n', "kind"),
    ('event = "e"\ntask = "T"\nkind = "forever"\n', "forever"),
])
def test_a_wrong_contract_is_named(tmp_path, toml, why):
    with pytest.raises(ValueError, match=why):
        ch.load(_folder(tmp_path, toml) / "tasks.py")


def test_a_knob_without_a_value_is_named(tmp_path):
    with pytest.raises(ValueError, match="params.forward_reward"):
        ch.params(_folder(tmp_path, TOML.replace("value = 2.0\n", "")) / "env.py")

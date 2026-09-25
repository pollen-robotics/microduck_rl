"""A challenge's contract, read from the `challenge.toml` beside its code.

A challenge (a folder in `microduck-challenges`) keeps its training env in `env.py`, its
registration in `tasks.py`, and its contract in `challenge.toml` next to them:

    event = "sprint-2m"                  # the Arena event it trains for
    task  = "Mjlab-Sprint2m-MicroDuck"   # the mjlab task tasks.py registers
    kind  = "perpetual"                  # what publish writes in the manifest
    [recipe]                             # train's arguments for the first run
    num_envs = 4096
    [params.forward_reward]              # a knob a novice tunes from the workshop
    label = "Speed reward"
    value = 2.0
    min = 0.5
    max = 5.0

`tasks.py` calls `register(__file__)` next to mjlab's `register_mjlab_task`, so `publish`
finds the event and the kind from the task id and needs only `--repo`. `env.py` calls
`params(__file__)` and builds its config from the values; the workshop and a shell user edit
the same file, so both train the same thing.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mjlab_microduck.publish.manifest import KINDS

FILE = "challenge.toml"


@dataclass(frozen=True)
class Challenge:
    folder: Path
    event: str
    task: str
    kind: str
    recipe: dict[str, Any] = field(default_factory=dict)


_REGISTRY: dict[str, Challenge] = {}


def _table(tasks_file: str | Path) -> tuple[Path, dict[str, Any]]:
    path = Path(tasks_file).resolve().parent / FILE
    if not path.exists():
        raise FileNotFoundError(f"{path}: a challenge's code sits next to its {FILE}")
    return path, tomllib.loads(path.read_text())


def load(tasks_file: str | Path) -> Challenge:
    path, table = _table(tasks_file)
    for key in ("event", "task", "kind"):
        if key not in table:
            raise ValueError(f"{path}: no `{key}`")
    if table["kind"] not in KINDS:
        raise ValueError(f"{path}: kind must be one of {KINDS}, not {table['kind']!r}")
    return Challenge(
        folder=path.parent,
        event=str(table["event"]),
        task=str(table["task"]),
        kind=str(table["kind"]),
        recipe=dict(table.get("recipe", {})),
    )


def register(tasks_file: str | Path) -> Challenge:
    """Called by a challenge's tasks.py at import; `publish` then knows the task's contract."""
    found = load(tasks_file)
    _REGISTRY[found.task] = found
    return found


def for_task(task_id: str) -> Challenge | None:
    return _REGISTRY.get(task_id)


def params(tasks_file: str | Path) -> dict[str, float]:
    """`{name: value}` for every `[params.<name>]` in the challenge.toml beside `tasks_file`."""
    path, table = _table(tasks_file)
    values: dict[str, float] = {}
    for name, knob in table.get("params", {}).items():
        if "value" not in knob:
            raise ValueError(f"{path}: params.{name} has no value")
        values[name] = float(knob["value"])
    return values

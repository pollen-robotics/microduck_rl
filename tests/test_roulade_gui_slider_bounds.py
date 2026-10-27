"""Regression test for issue #79: Viser viewer crash on Roulade.

mjlab's ``UniformVelocityCommand.create_gui`` adds a "Max <axis>" slider with
``min=0.1`` and ``initial_value=ranges.<axis>[1]``. For Roulade the ranges are
``(-0.01, 0.01)`` / ``(-0.05, 0.05)``, so viser's
``assert max >= value >= min`` fails and the viewer dies at startup.

``VelocityCommandCommandOnly`` must override ``create_gui`` to skip the
joystick sliders when any command upper bound is below 0.1.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field

from mjlab_microduck.tasks.mdp import VelocityCommandCommandOnly


@dataclass
class _Ranges:
    lin_vel_x: tuple = (-10.0, 10.0)
    lin_vel_y: tuple = (-10.0, 10.0)
    ang_vel_z: tuple = (-10.0, 10.0)
    heading: tuple | None = None


@dataclass
class _Cfg:
    ranges: _Ranges = field(default_factory=_Ranges)


class _GuiTracker:
    """Mock viser GUI that enforces viser's own invariant on add_slider."""

    def __init__(self) -> None:
        self.sliders: list[dict] = []

    def add_slider(self, label, initial_value=0.0, step=0.1, min=0.0, max=1.0):
        # viser/_gui_api.py: assert max >= value >= min
        assert max >= initial_value >= min, (
            f"viser assertion failed for slider {label!r}: "
            f"max={max} value={initial_value} min={min}"
        )
        self.sliders.append({"label": label, "value": initial_value, "min": min, "max": max})

        class _Handle:
            def on_update(self, fn):
                return fn

        return _Handle()

    def add_checkbox(self, label, initial_value=False):
        class _Handle:
            value = initial_value

            def on_update(self, fn):
                return fn

        return _Handle()

    def add_button(self, label, icon=None):
        class _Handle:
            def on_click(self, fn):
                return fn

        return _Handle()

    @contextmanager
    def add_folder(self, label):
        yield


class _Server:
    def __init__(self) -> None:
        self.gui = _GuiTracker()


def _make_command(roulade: bool) -> VelocityCommandCommandOnly:
    cmd = object.__new__(VelocityCommandCommandOnly)
    cmd.cfg = _Cfg()
    if roulade:
        cmd.cfg.ranges = _Ranges(
            lin_vel_x=(-0.01, 0.01),
            lin_vel_y=(-0.01, 0.01),
            ang_vel_z=(-0.05, 0.05),
        )
    return cmd


def test_create_gui_skips_sliders_for_roulade_ranges():
    """Roulade-scale ranges must not raise viser's slider assertion (issue #79)."""
    cmd = _make_command(roulade=True)
    gui = _GuiTracker()
    server = _Server()
    server.gui = gui

    cmd.create_gui("twist", server, lambda: 0)  # must not raise
    assert gui.sliders == [], "joystick sliders should be skipped for near-zero ranges"


def test_create_gui_normal_ranges_still_create_sliders():
    """Ordinary velocity tasks keep their joystick sliders."""
    cmd = _make_command(roulade=False)
    gui = _GuiTracker()
    server = _Server()
    server.gui = gui

    cmd.create_gui("twist", server, lambda: 0)
    labels = [s["label"] for s in gui.sliders]
    assert any("lin_vel_x" in label for label in labels)


def test_roulade_cfg_ranges_are_the_trigger():
    """Guard: the shipped Roulade cfg really uses the tiny ranges from the issue."""
    with open(
        "src/mjlab_microduck/tasks/microduck_roulade_env_cfg.py", encoding="utf-8"
    ) as f:
        cfg_src = f.read()
    assert "command.ranges.lin_vel_x = (-0.01, 0.01)" in cfg_src
    assert "command.ranges.ang_vel_z = (-0.05, 0.05)" in cfg_src

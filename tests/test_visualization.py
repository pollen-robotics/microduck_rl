import threading
from argparse import Namespace

import numpy as np
import pytest

from mjlab_microduck.sim.visualization import stream as visualization_stream
from mjlab_microduck.sim.visualization.service import VisualizationService
from mjlab_microduck.sim.visualization.stream import RenderStream


class FakeData:
    def __init__(self, _model):
        self.time = 2.5
        self.qpos = np.zeros(8)


class FakeWorld:
    def __init__(self):
        self.model = object()
        self.data = FakeData(self.model)
        self.lock = threading.Lock()
        self.bodies = []


class FakeCamera:
    def __init__(self):
        self.type = None
        self.azimuth = 0
        self.elevation = 0
        self.distance = 0
        self.lookat = np.zeros(3)


def test_render_stream_releases_world_lock_before_slow_render(monkeypatch):
    entered_render = threading.Event()
    release_render = threading.Event()

    class SlowRenderer:
        def __init__(self, _model, *, height, width):
            self.height = height
            self.width = width

        def update_scene(self, _data, *, camera):
            del camera

        def render(self):
            entered_render.set()
            assert release_render.wait(2)
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

        def close(self):
            pass

    def copy_data(destination, _model, source):
        destination.time = source.time
        destination.qpos[:] = source.qpos

    monkeypatch.setattr(visualization_stream.mujoco, "MjData", FakeData)
    monkeypatch.setattr(visualization_stream.mujoco, "Renderer", SlowRenderer)
    monkeypatch.setattr(visualization_stream.mujoco, "MjvCamera", FakeCamera)
    monkeypatch.setattr(visualization_stream.mujoco, "mj_copyData", copy_data)

    world = FakeWorld()
    stream = RenderStream(world, width=16, height=9, fps=1)
    stream.start()
    try:
        assert entered_render.wait(1)
        assert world.lock.acquire(timeout=0.1), "rendering held the physics lock"
        world.lock.release()
        release_render.set()
        frame = stream.get(timeout_ms=1000)
        assert frame["seq"] == 1
        assert frame["sim_time"] == 2.5
        assert frame["width"] == 16
        assert frame["height"] == 9
        assert frame["mime"] == "image/jpeg"
        assert frame["image_b64"]
        assert stream.get(after=frame["seq"], timeout_ms=1) == {
            "seq": frame["seq"],
            "timeout": True,
        }
    finally:
        release_render.set()
        stream.close()


def test_render_stream_surfaces_initialization_failure(monkeypatch):
    def fail_renderer(*_args, **_kwargs):
        raise RuntimeError("no OpenGL device")

    monkeypatch.setattr(visualization_stream.mujoco, "MjData", FakeData)
    monkeypatch.setattr(visualization_stream.mujoco, "Renderer", fail_renderer)
    stream = RenderStream(FakeWorld())
    stream.start()
    try:
        with pytest.raises(RuntimeError, match="no OpenGL device"):
            stream.get(timeout_ms=1000)
    finally:
        stream.close()


def test_render_stream_exits_only_after_last_client_stays_disconnected():
    stream = RenderStream(FakeWorld(), exit_idle_seconds=0.03)
    stream.client_connected()
    stream.client_disconnected()
    stream.client_connected()
    assert not stream.exit_requested.wait(0.05)
    stream.client_disconnected()
    assert stream.exit_requested.wait(0.2)
    stream.close()


def test_render_stream_camera_orbits_zooms_clamps_and_resets():
    stream = RenderStream(FakeWorld())
    moved = stream.camera("orbit", dx=20, dy=-1000)
    assert moved["azimuth"] == 129
    assert moved["elevation"] == 85
    assert stream.camera("zoom", dy=10_000)["distance"] == 5
    assert stream.camera("zoom", dy=-10_000)["distance"] == 0.25
    assert stream.camera("reset") == {
        "azimuth": 135,
        "elevation": -18,
        "distance": 0.85,
    }
    with pytest.raises(ValueError, match="unknown camera action"):
        stream.camera("pan")


def test_render_stream_switches_quality_profile():
    stream = RenderStream(FakeWorld())
    assert stream.configure(
        width=1920,
        height=1080,
        fps=24,
        quality=100,
        image_format="png",
    ) == {
        "width": 1920,
        "height": 1080,
        "fps": 24,
        "quality": 100,
        "format": "png",
    }
    assert stream.image_format == "png"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"width": 0}, "dimensions"),
        ({"fps": 0}, "FPS"),
        ({"quality": 101}, "quality"),
        ({"image_format": "webp"}, "format"),
    ],
)
def test_render_stream_validates_settings(kwargs, message):
    with pytest.raises(ValueError, match=message):
        RenderStream(FakeWorld(), **kwargs)


def test_disabled_visualization_preserves_protocol_but_rejects_requests():
    service = VisualizationService()
    assert service.handles("render")
    assert service.handles("camera")
    assert service.handles("render_config")
    assert not service.handles("read")
    assert service.exit_requested is None
    with pytest.raises(RuntimeError, match="disabled"):
        service.dispatch({"op": "render"})


def test_visualization_requires_render_for_idle_exit():
    args = Namespace(render=False, exit_on_render_idle=1.0)
    with pytest.raises(ValueError, match="requires --render"):
        VisualizationService.from_arguments(args, FakeWorld())


def test_visualization_service_delegates_wire_operations():
    class FakeStream:
        exit_requested = threading.Event()

        def get(self, *, after, timeout_ms):
            return {"after": after, "timeout_ms": timeout_ms}

        def camera(self, *, action, dx, dy):
            return {"action": action, "dx": dx, "dy": dy}

        def configure(self, **settings):
            return settings

    service = VisualizationService(FakeStream())
    assert service.dispatch({"op": "render", "after": 4, "timeout_ms": 20}) == {
        "after": 4,
        "timeout_ms": 20,
    }
    assert service.dispatch({"op": "camera", "action": "orbit", "dx": 2}) == {
        "action": "orbit",
        "dx": 2.0,
        "dy": 0.0,
    }
    assert service.dispatch({"op": "render_config", "width": 8, "height": 6}) == {
        "width": 8,
        "height": 6,
        "fps": 24,
        "quality": 95,
        "image_format": "jpeg",
    }

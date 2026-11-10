"""Cached image rendering from snapshots of a live MuJoCo world."""

from __future__ import annotations

import base64
import io
import math
import os
import threading
import time
from typing import Any, Protocol

import mujoco
from PIL import Image

DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS = 24
DEFAULT_QUALITY = 95
DEFAULT_FORMAT = "jpeg"


class RenderWorld(Protocol):
    """The small portion of the body server world required for rendering."""

    model: Any
    data: Any
    lock: threading.Lock
    bodies: list[Any]


class RenderStream:
    """Render cached images from snapshots of the authoritative MuJoCo world.

    Snapshotting is the only work performed under ``world.lock``. OpenGL rendering and image
    encoding happen against a private ``MjData`` after the lock is released, so a slow software
    renderer cannot stall the 50 Hz physics and robot control path.
    """

    def __init__(
        self,
        world: RenderWorld,
        *,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        fps: int = DEFAULT_FPS,
        quality: int = DEFAULT_QUALITY,
        image_format: str = DEFAULT_FORMAT,
        exit_idle_seconds: float | None = None,
    ):
        self._validate_settings(width, height, fps, quality, image_format)
        if exit_idle_seconds is not None and exit_idle_seconds <= 0:
            raise ValueError("render idle exit must be positive")
        self.world = world
        self.width = width
        self.height = height
        self.fps = fps
        self.quality = quality
        self.image_format = image_format
        self.backend = os.getenv("MUJOCO_GL", "default")
        self.exit_idle_seconds = exit_idle_seconds
        self.exit_requested = threading.Event()
        self._condition = threading.Condition()
        self._frame: dict | None = None
        self._error: str | None = None
        self._stopping = threading.Event()
        self._clients = 0
        self._had_client = False
        self._idle_timer: threading.Timer | None = None
        self._camera_azimuth = 135.0
        self._camera_elevation = -18.0
        self._camera_distance = max(0.85, 0.65 + len(self.world.bodies) * 0.2)
        self._thread = threading.Thread(
            target=self._run, name="mujoco-render", daemon=True
        )

    @staticmethod
    def _validate_settings(
        width: int, height: int, fps: int, quality: int, image_format: str
    ) -> None:
        if width < 1 or height < 1 or width * height > 3840 * 2160:
            raise ValueError(
                "render dimensions must be positive and at most 3840x2160 pixels"
            )
        if not 1 <= fps <= 60:
            raise ValueError("render FPS must be between 1 and 60")
        if not 1 <= quality <= 100:
            raise ValueError("JPEG quality must be between 1 and 100")
        if image_format not in {"jpeg", "png"}:
            raise ValueError("render format must be jpeg or png")

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._stopping.set()
        with self._condition:
            if self._idle_timer is not None:
                self._idle_timer.cancel()
                self._idle_timer = None
            self._condition.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=3.0)

    def client_connected(self) -> None:
        with self._condition:
            self._clients += 1
            self._had_client = True
            if self._idle_timer is not None:
                self._idle_timer.cancel()
                self._idle_timer = None

    def client_disconnected(self) -> None:
        with self._condition:
            self._clients = max(0, self._clients - 1)
            if (
                self._clients == 0
                and self._had_client
                and self.exit_idle_seconds is not None
                and not self._stopping.is_set()
            ):
                if self._idle_timer is not None:
                    self._idle_timer.cancel()
                self._idle_timer = threading.Timer(
                    self.exit_idle_seconds, self._exit_if_still_idle
                )
                self._idle_timer.daemon = True
                self._idle_timer.start()

    def _exit_if_still_idle(self) -> None:
        with self._condition:
            self._idle_timer = None
            if self._clients == 0 and not self._stopping.is_set():
                self.exit_requested.set()

    def camera(self, action: str, dx: float = 0.0, dy: float = 0.0) -> dict:
        if not math.isfinite(dx) or not math.isfinite(dy):
            raise ValueError("camera movement must be finite")
        with self._condition:
            if action == "orbit":
                self._camera_azimuth = (self._camera_azimuth - dx * 0.3) % 360
                self._camera_elevation = min(
                    85.0, max(-85.0, self._camera_elevation - dy * 0.25)
                )
            elif action == "zoom":
                self._camera_distance = min(
                    5.0, max(0.25, self._camera_distance * math.exp(dy * 0.0015))
                )
            elif action == "reset":
                self._camera_azimuth = 135.0
                self._camera_elevation = -18.0
                self._camera_distance = max(0.85, 0.65 + len(self.world.bodies) * 0.2)
            else:
                raise ValueError(f"unknown camera action {action!r}")
            return {
                "azimuth": self._camera_azimuth,
                "elevation": self._camera_elevation,
                "distance": self._camera_distance,
            }

    def configure(
        self,
        *,
        width: int,
        height: int,
        fps: int,
        quality: int,
        image_format: str,
    ) -> dict:
        self._validate_settings(width, height, fps, quality, image_format)
        with self._condition:
            self.width = width
            self.height = height
            self.fps = fps
            self.quality = quality
            self.image_format = image_format
            self._condition.notify_all()
            return {
                "width": width,
                "height": height,
                "fps": fps,
                "quality": quality,
                "format": image_format,
            }

    def get(self, after: int = 0, timeout_ms: int = 1000) -> dict:
        if after < 0:
            raise ValueError("render sequence must not be negative")
        if not 0 <= timeout_ms <= 30_000:
            raise ValueError("render timeout_ms must be between 0 and 30000")
        deadline = time.monotonic() + timeout_ms / 1000
        with self._condition:
            while self._frame is None or self._frame["seq"] <= after:
                if self._error is not None:
                    raise RuntimeError(f"MuJoCo renderer failed: {self._error}")
                remaining = deadline - time.monotonic()
                if remaining <= 0 or self._stopping.is_set():
                    return {"seq": after, "timeout": True}
                self._condition.wait(remaining)
            return dict(self._frame)

    def _run(self) -> None:
        renderer = None
        try:
            render_data = mujoco.MjData(self.world.model)
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            next_frame = time.monotonic()
            seq = 0
            active_size = None
            while not self._stopping.is_set():
                with self._condition:
                    width = self.width
                    height = self.height
                    fps = self.fps
                    quality = self.quality
                    image_format = self.image_format
                    camera.azimuth = self._camera_azimuth
                    camera.elevation = self._camera_elevation
                    camera.distance = self._camera_distance
                if active_size != (width, height):
                    if renderer is not None:
                        renderer.close()
                    visual_global = getattr(
                        getattr(self.world.model, "vis", None), "global_", None
                    )
                    if visual_global is not None:
                        visual_global.offwidth = max(visual_global.offwidth, width)
                        visual_global.offheight = max(visual_global.offheight, height)
                    renderer = mujoco.Renderer(
                        self.world.model, height=height, width=width
                    )
                    active_size = (width, height)
                period = 1.0 / fps
                delay = next_frame - time.monotonic()
                if delay > 0 and self._stopping.wait(delay):
                    break

                with self.world.lock:
                    mujoco.mj_copyData(render_data, self.world.model, self.world.data)
                sim_time = float(render_data.time)
                if self.world.bodies:
                    trunk = self.world.bodies[0].trunk
                    camera.lookat[:] = render_data.qpos[trunk : trunk + 3]
                    camera.lookat[2] = max(
                        0.08, float(render_data.qpos[trunk + 2]) * 0.6
                    )
                renderer.update_scene(render_data, camera=camera)
                rgb = renderer.render()
                output = io.BytesIO()
                if image_format == "png":
                    Image.fromarray(rgb).save(output, format="PNG", compress_level=1)
                    mime = "image/png"
                else:
                    Image.fromarray(rgb).save(output, format="JPEG", quality=quality)
                    mime = "image/jpeg"
                seq += 1
                frame = {
                    "seq": seq,
                    "sim_time": sim_time,
                    "width": width,
                    "height": height,
                    "mime": mime,
                    "backend": self.backend,
                    "image_b64": base64.b64encode(output.getvalue()).decode("ascii"),
                }
                with self._condition:
                    self._frame = frame
                    self._condition.notify_all()
                next_frame = max(next_frame + period, time.monotonic())
        except Exception as error:  # noqa: BLE001 - renderer failures are surfaced to clients
            with self._condition:
                self._error = str(error)
                self._condition.notify_all()
        finally:
            if renderer is not None:
                renderer.close()

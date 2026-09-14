"""CLI, lifecycle, and wire adapter for optional MuJoCo visualization."""

from __future__ import annotations

import argparse
import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mjlab_microduck.sim.visualization.stream import RenderStream, RenderWorld

DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS = 24
DEFAULT_QUALITY = 95
DEFAULT_FORMAT = "jpeg"

VISUALIZATION_OPERATIONS = frozenset({"render", "camera", "render_config"})


def add_visualization_arguments(parser: argparse.ArgumentParser) -> None:
    """Add optional visualization controls to a body-server argument parser."""

    parser.add_argument(
        "--render", action="store_true", help="serve cached image frames"
    )
    parser.add_argument("--render-width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--render-height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--render-fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--render-quality", type=int, default=DEFAULT_QUALITY)
    parser.add_argument(
        "--render-format", choices=("jpeg", "png"), default=DEFAULT_FORMAT
    )
    parser.add_argument(
        "--exit-on-render-idle",
        type=float,
        metavar="SECONDS",
        help="exit after the last render client stays disconnected for this long",
    )


class VisualizationService:
    """Optional image stream attached to one or more simulated-body servers."""

    def __init__(self, stream: RenderStream | None = None):
        self.stream = stream

    @classmethod
    def from_arguments(
        cls, args: argparse.Namespace, world: RenderWorld
    ) -> VisualizationService:
        if args.exit_on_render_idle is not None and not args.render:
            raise ValueError("--exit-on-render-idle requires --render")
        if not args.render:
            return cls()
        try:
            from mjlab_microduck.sim.visualization.stream import RenderStream
        except ImportError as error:
            if error.name == "PIL":
                raise RuntimeError(
                    "rendering requires the visualization extra; "
                    "run `uv sync --extra visualization`"
                ) from error
            raise
        return cls(
            RenderStream(
                world,
                width=args.render_width,
                height=args.render_height,
                fps=args.render_fps,
                quality=args.render_quality,
                image_format=args.render_format,
                exit_idle_seconds=args.exit_on_render_idle,
            )
        )

    @property
    def exit_requested(self) -> threading.Event | None:
        return self.stream.exit_requested if self.stream is not None else None

    def start(self) -> None:
        if self.stream is not None:
            self.stream.start()

    def close(self) -> None:
        if self.stream is not None:
            self.stream.close()

    def handles(self, operation: Any) -> bool:
        return operation in VISUALIZATION_OPERATIONS

    def client_connected(self) -> None:
        if self.stream is not None:
            self.stream.client_connected()

    def client_disconnected(self) -> None:
        if self.stream is not None:
            self.stream.client_disconnected()

    def dispatch(self, request: dict) -> dict:
        operation = request.get("op")
        if not self.handles(operation):
            raise ValueError(f"unknown visualization operation {operation!r}")
        if self.stream is None:
            raise RuntimeError(
                "MuJoCo rendering is disabled; start duck-body with --render"
            )
        if operation == "render":
            return self.stream.get(
                after=int(request.get("after", 0)),
                timeout_ms=int(request.get("timeout_ms", 1000)),
            )
        if operation == "camera":
            return self.stream.camera(
                action=str(request.get("action", "")),
                dx=float(request.get("dx", 0.0)),
                dy=float(request.get("dy", 0.0)),
            )
        return self.stream.configure(
            width=int(request["width"]),
            height=int(request["height"]),
            fps=int(request.get("fps", DEFAULT_FPS)),
            quality=int(request.get("quality", DEFAULT_QUALITY)),
            image_format=str(request.get("format", DEFAULT_FORMAT)),
        )

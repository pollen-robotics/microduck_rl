#!/usr/bin/env python3
"""Serve the MicroDuck training dashboard on the local network.

Usage::

    python scripts/serve_dashboard.py
    python scripts/serve_dashboard.py --port 9999

The server is intentionally standard-library-only. It binds to ``0.0.0.0``
so a Tailscale device can reach it at ``http://<tailscale-ip>:9999``. The
dashboard reads, but never modifies, ``logs/rsl_rl``, ``output/playwright``,
``artifacts``, and ``dashboard/media``. Media visibility is curated by
``dashboard/featured_media.json`` when that file is present.
"""

from __future__ import annotations

import argparse
import json
import math
import mimetypes
import os
import re
import struct
from threading import Lock, Thread
import time
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import unquote, urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_ROOT = REPO_ROOT / "dashboard"
FEATURED_MEDIA_FILE = DASHBOARD_ROOT / "featured_media.json"
LOG_ROOT = REPO_ROOT / "logs" / "rsl_rl"
MEDIA_ROOTS = {
    # Training and play recorders write MP4s next to their checkpoints. Keep
    # this root read-only and expose only known media extensions below.
    "runs": LOG_ROOT,
    "playwright": REPO_ROOT / "output" / "playwright",
    "artifacts": REPO_ROOT / "artifacts",
    "dashboard-media": REPO_ROOT / "dashboard" / "media",
}
EVENT_NAME = re.compile(r"(?:events?\.out\.)?tfevents\.")
MEDIA_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg", ".mp4", ".webm",
    ".mov", ".m4v", ".avi", ".mkv",
}
CHECKPOINT_EXTENSIONS = {".pt", ".pth", ".onnx", ".ckpt"}
MAX_EVENT_BYTES = 64 * 1024 * 1024
MAX_SERIES_POINTS = 180
_STATE_LOCK = Lock()
_STATE_CACHE: dict[str, Any] | None = None
_STATE_REFRESHING = False
_STATE_LAST_REFRESH = 0.0
_STATE_REFRESH_INTERVAL = 15.0


def _read_varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while offset < len(data) and shift < 70:
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, offset
        shift += 7
    raise ValueError("invalid protobuf varint")


def _protobuf_fields(data: bytes) -> Iterable[tuple[int, int, bytes | int]]:
    """Yield (field number, wire type, value) without requiring protobuf."""

    offset = 0
    while offset < len(data):
        key, offset = _read_varint(data, offset)
        field_number, wire_type = key >> 3, key & 7
        if field_number == 0:
            raise ValueError("invalid protobuf field")
        if wire_type == 0:
            value, offset = _read_varint(data, offset)
        elif wire_type == 1:
            value = data[offset:offset + 8]
            offset += 8
        elif wire_type == 2:
            length, offset = _read_varint(data, offset)
            value = data[offset:offset + length]
            offset += length
        elif wire_type == 5:
            value = data[offset:offset + 4]
            offset += 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wire_type}")
        yield field_number, wire_type, value


def _float_from_tensor(tensor: bytes) -> float | None:
    """Extract the scalar float used by modern TensorBoard Summary.Value."""

    floats: list[float] = []
    doubles: list[float] = []
    tensor_content: bytes | None = None
    try:
        for field, wire, value in _protobuf_fields(tensor):
            if field == 4 and wire == 2:
                tensor_content = value  # TensorProto.tensor_content
            elif field == 5 and wire == 5:
                floats.append(struct.unpack("<f", value)[0])
            elif field == 5 and wire == 2:
                floats.extend(struct.unpack(f"<{len(value) // 4}f", value))
            elif field == 6 and wire == 1:
                doubles.append(struct.unpack("<d", value)[0])
            elif field == 6 and wire == 2:
                doubles.extend(struct.unpack(f"<{len(value) // 8}d", value))
    except (ValueError, struct.error):
        return None
    if floats:
        return floats[0]
    if doubles:
        return doubles[0]
    if tensor_content and len(tensor_content) >= 4:
        return struct.unpack("<f", tensor_content[:4])[0]
    return None


def _event_scalars(path: Path) -> dict[str, list[list[float | int]]]:
    """Read scalar Summary values from a TensorFlow event file.

    CRCs are deliberately ignored. The record framing remains useful for
    diagnostics even when a run is still writing its final record.
    """

    series: dict[str, list[list[float | int]]] = {}
    try:
        with path.open("rb") as stream:
            consumed = 0
            while consumed < MAX_EVENT_BYTES:
                header = stream.read(12)
                if len(header) != 12:
                    break
                length = struct.unpack("<Q", header[:8])[0]
                if length > MAX_EVENT_BYTES or length < 1:
                    break
                payload = stream.read(length)
                trailer = stream.read(4)
                if len(payload) != length or len(trailer) != 4:
                    break
                consumed += 16 + length
                try:
                    step = 0
                    summary = None
                    for field, wire, value in _protobuf_fields(payload):
                        if field == 2 and wire == 0:
                            step = int(value)
                        elif field == 5 and wire == 2:
                            summary = value
                    if summary is None:
                        continue
                    for field, wire, value in _protobuf_fields(summary):
                        if field != 1 or wire != 2:
                            continue
                        tag = None
                        scalar = None
                        tensor = None
                        for value_field, value_wire, value_value in _protobuf_fields(value):
                            if value_field == 1 and value_wire == 2:
                                tag = value_value.decode("utf-8", "replace")
                            elif value_field == 2 and value_wire == 5:
                                scalar = struct.unpack("<f", value_value)[0]
                            elif value_field == 8 and value_wire == 2:
                                tensor = value_value
                        if tag:
                            if scalar is None and tensor is not None:
                                scalar = _float_from_tensor(tensor)
                            if scalar is not None and math.isfinite(scalar):
                                points = series.setdefault(tag, [])
                                if not points or points[-1] != [step, scalar]:
                                    points.append([step, scalar])
                                    del points[:-MAX_SERIES_POINTS]
                except (ValueError, UnicodeError, struct.error):
                    continue
    except (OSError, PermissionError):
        return {}
    return series


def _read_text_value(path: Path, names: tuple[str, ...]) -> str | None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for name in names:
        match = re.search(rf"^\s*{re.escape(name)}\s*:\s*['\"]?([^'\"\r\n#]+)", text, re.MULTILINE)
        if match:
            return match.group(1).strip()
    return None


def _human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"


def _iso_timestamp(timestamp: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp))


def _run_status(latest_timestamp: float) -> str:
    age = max(0.0, time.time() - latest_timestamp)
    return "active" if age < 90 else "idle"


def _run_id(path: Path) -> str:
    try:
        return path.relative_to(LOG_ROOT).as_posix()
    except ValueError:
        return path.name


def _discover_runs(*, include_metrics: bool = True) -> list[dict[str, Any]]:
    if not LOG_ROOT.is_dir():
        return []
    candidates: set[Path] = set()
    try:
        for event_path in LOG_ROOT.rglob("*"):
            if event_path.is_file() and EVENT_NAME.search(event_path.name):
                candidates.add(event_path.parent)
    except OSError:
        return []

    runs: list[dict[str, Any]] = []
    for path in candidates:
        try:
            files = [item for item in path.rglob("*") if item.is_file()]
            latest = max((item.stat().st_mtime for item in files), default=path.stat().st_mtime)
        except OSError:
            continue
        event_files = [item for item in files if EVENT_NAME.search(item.name)]
        checkpoints = [
            {
                "name": item.name,
                "path": item.relative_to(path).as_posix(),
                "size": _human_size(item.stat().st_size),
                "modified": _iso_timestamp(item.stat().st_mtime),
            }
            for item in files
            if item.suffix.lower() in CHECKPOINT_EXTENSIONS
        ]
        checkpoints.sort(key=lambda item: item["modified"], reverse=True)
        metrics: dict[str, list[list[float | int]]] = {}
        if include_metrics:
            for event_file in sorted(event_files, key=lambda item: item.stat().st_mtime):
                for tag, points in _event_scalars(event_file).items():
                    metrics.setdefault(tag, []).extend(points)
            for points in metrics.values():
                points.sort(key=lambda point: point[0])
                del points[:-MAX_SERIES_POINTS]
        env_params = path / "params" / "env.yaml"
        agent_params = path / "params" / "agent.yaml"
        task = _read_text_value(env_params, ("task", "task_id")) or path.name
        experiment = _read_text_value(agent_params, ("experiment_name", "experiment"))
        runs.append({
            "id": _run_id(path),
            "name": path.name,
            "task": task,
            "experiment": experiment or path.parent.name,
            "status": _run_status(latest),
            "lastActivity": _iso_timestamp(latest),
            "ageSeconds": round(max(0.0, time.time() - latest)),
            "checkpoints": checkpoints,
            "metrics": _metric_payload(metrics),
        })
    runs.sort(key=lambda item: item["lastActivity"], reverse=True)
    return runs


def _metric_payload(series: dict[str, list[list[float | int]]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for tag, points in series.items():
        if not points:
            continue
        values = [float(point[1]) for point in points]
        payload[tag] = {
            "points": points,
            "latest": values[-1],
            "minimum": min(values),
            "maximum": max(values),
        }
    return payload


def _featured_media_config() -> tuple[set[str] | None, tuple[str, ...], bool]:
    """Return exact videos, video prefixes, and whether images are visible."""

    if not FEATURED_MEDIA_FILE.is_file():
        return None, (), True
    try:
        payload = json.loads(FEATURED_MEDIA_FILE.read_text(encoding="utf-8"))
        values = payload.get("featuredVideos", [])
        if not isinstance(values, list):
            values = []
        prefixes = payload.get("featuredVideoPrefixes", [])
        if not isinstance(prefixes, list):
            prefixes = []
        exact = {
            str(value).strip().replace("\\", "/")
            for value in values
            if str(value).strip()
        }
        normalized_prefixes = tuple(
            str(value).strip().replace("\\", "/")
            for value in prefixes
            if str(value).strip()
        )
        return exact, normalized_prefixes, bool(payload.get("showImages", True))
    except (OSError, ValueError, AttributeError):
        return set(), (), False


def _discover_media() -> list[dict[str, Any]]:
    media: list[dict[str, Any]] = []
    featured_videos, featured_prefixes, show_images = _featured_media_config()
    for source, root in MEDIA_ROOTS.items():
        if not root.is_dir():
            continue
        try:
            files = root.rglob("*")
            for path in files:
                if not path.is_file() or path.suffix.lower() not in MEDIA_EXTENSIONS:
                    continue
                stat = path.stat()
                relative_path = path.relative_to(root).as_posix()
                kind = "video" if path.suffix.lower() in {".mp4", ".webm", ".mov", ".m4v", ".avi", ".mkv"} else "image"
                media_key = f"{source}/{relative_path}"
                if (
                    kind == "video"
                    and featured_videos is not None
                    and media_key not in featured_videos
                    and not any(
                        media_key.startswith(prefix) for prefix in featured_prefixes
                    )
                ):
                    continue
                if kind == "image" and not show_images:
                    continue
                media.append({
                    "name": path.name,
                    "source": source,
                    "path": relative_path,
                    "kind": kind,
                    "url": "/media/" + source + "/" + relative_path,
                    "size": _human_size(stat.st_size),
                    "modified": _iso_timestamp(stat.st_mtime),
                    "timestamp": stat.st_mtime,
                })
        except OSError:
            continue
    media.sort(key=lambda item: item["timestamp"], reverse=True)
    for item in media:
        item.pop("timestamp", None)
    return media[:160]


def dashboard_state(*, include_metrics: bool = True) -> dict[str, Any]:
    runs = _discover_runs(include_metrics=include_metrics)
    media = _discover_media()
    checkpoints = sum(len(run["checkpoints"]) for run in runs)
    return {
        "generatedAt": _iso_timestamp(time.time()),
        "repo": REPO_ROOT.name,
        "runs": runs,
        "media": media,
        "summary": {
            "runs": len(runs),
            "activeRuns": sum(run["status"] == "active" for run in runs),
            "checkpoints": checkpoints,
            "media": len(media),
        },
    }


def _refresh_state_in_background() -> None:
    global _STATE_CACHE, _STATE_REFRESHING, _STATE_LAST_REFRESH
    try:
        snapshot = dashboard_state()
        with _STATE_LOCK:
            _STATE_CACHE = snapshot
    finally:
        with _STATE_LOCK:
            _STATE_REFRESHING = False
            _STATE_LAST_REFRESH = time.monotonic()


def cached_dashboard_state() -> dict[str, Any]:
    """Return quickly, then refresh expensive TensorBoard metrics off-thread."""

    global _STATE_CACHE, _STATE_REFRESHING
    with _STATE_LOCK:
        if _STATE_CACHE is None:
            _STATE_CACHE = dashboard_state(include_metrics=False)
        refresh_due = time.monotonic() - _STATE_LAST_REFRESH >= _STATE_REFRESH_INTERVAL
        if not _STATE_REFRESHING and refresh_due:
            _STATE_REFRESHING = True
            Thread(
                target=_refresh_state_in_background,
                name="dashboard-state-refresh",
                daemon=True,
            ).start()
        return _STATE_CACHE


class DashboardHandler(SimpleHTTPRequestHandler):
    """Static files plus a read-only dashboard API."""

    server_version = "MicroDuckDashboard/1.0"

    def do_GET(self) -> None:  # noqa: N802, inherited HTTP API
        route = urlparse(self.path).path
        if route == "/api/state":
            self._send_json(cached_dashboard_state())
            return
        if route.startswith("/media/"):
            self._serve_media(route)
            return
        if route == "/" or route == "":
            self.path = "/index.html"
        super().do_GET()

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _send_json(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            return

    def _serve_media(self, route: str) -> None:
        parts = unquote(route).split("/")
        if len(parts) < 3:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        source = parts[2]
        root = MEDIA_ROOTS.get(source)
        if root is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        candidate = (root / Path(*parts[3:])).resolve()
        try:
            inside_root = candidate.is_relative_to(root.resolve())
        except AttributeError:
            inside_root = str(candidate).startswith(str(root.resolve()))
        if not inside_root or not candidate.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            body = candidate.read_bytes()
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mimetypes.guess_type(candidate.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "public, max-age=60")
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0", help="listen address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=int(os.environ.get("MICRODUCK_DASHBOARD_PORT", "9999")))
    args = parser.parse_args()
    if not DASHBOARD_ROOT.is_dir():
        raise SystemExit(f"Dashboard assets are missing: {DASHBOARD_ROOT}")
    server = ThreadingHTTPServer((args.host, args.port), lambda *handler_args: DashboardHandler(*handler_args, directory=str(DASHBOARD_ROOT)))
    print(f"MicroDuck dashboard: http://{args.host}:{args.port}/")
    print(f"Tailscale URL: http://<tailscale-ip>:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

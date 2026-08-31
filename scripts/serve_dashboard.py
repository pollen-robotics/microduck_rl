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
import time
from collections.abc import Iterable
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import Any
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
ROLL_SPRINT_SAMPLE_ROOT = (
    REPO_ROOT / "artifacts" / "training" / "roll-sprint-samples"
)
ROLL_SPRINT_CHAMPION_ROOT = (
    REPO_ROOT / "artifacts" / "training" / "roll-sprint-champion"
)
ROLL_SPRINT_CHAMPION_MANIFEST = ROLL_SPRINT_CHAMPION_ROOT / "champion.json"
ROLL_SPRINT_EVALUATION_ROOT = (
    ROLL_SPRINT_SAMPLE_ROOT / "evaluations"
)
EVENT_NAME = re.compile(r"(?:events?\.out\.)?tfevents\.")
MEDIA_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg", ".mp4", ".webm",
    ".mov", ".m4v", ".avi", ".mkv",
}
VIDEO_COLLECTIONS = {
    "roll-sprint": {
        "label": "Roll sprint",
        "description": "Repeated flat-ground rolls from the current sprint policy.",
        "default": True,
    },
    "stairs": {
        "label": "Stair training",
        "description": "The existing staircase-policy rollout samples.",
        "default": False,
    },
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


def _nested_value(payload: dict[str, Any], *paths: str) -> Any:
    """Return the first non-null value from a list of dotted JSON paths."""

    for path in paths:
        value: Any = payload
        for key in path.split("."):
            if not isinstance(value, dict) or key not in value:
                value = None
                break
            value = value[key]
        if value is not None:
            return value
    return None


def _dashboard_number(payload: dict[str, Any], *paths: str) -> float | int | None:
    value = _nested_value(payload, *paths)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(float(value)):
        return None
    return value


def _dashboard_bool(payload: dict[str, Any], *paths: str) -> bool | None:
    value = _nested_value(payload, *paths)
    return value if isinstance(value, bool) else None


def _orientation_dashboard_metrics(payload: dict[str, Any]) -> list[dict[str, Any]]:
    source = _nested_value(
        payload,
        "self_right_by_orientation",
        "self_righting_by_orientation",
        "recovery_by_orientation",
        "recovery_battery.by_orientation",
        "recovery.orientation_results",
    )
    if isinstance(source, list):
        entries = {
            str(item.get("orientation", item.get("name", index))): item
            for index, item in enumerate(source)
            if isinstance(item, dict)
        }
    elif isinstance(source, dict):
        entries = {str(name): item for name, item in source.items() if isinstance(item, dict)}
    else:
        return []

    preferred_order = ("face_down", "face_up", "left_side", "right_side")
    aliases = {
        "facedown": "face_down",
        "face-down": "face_down",
        "faceup": "face_up",
        "face-up": "face_up",
        "left": "left_side",
        "left-side": "left_side",
        "right": "right_side",
        "right-side": "right_side",
    }
    normalized = {
        aliases.get(name.lower(), name.lower().replace(" ", "_")): item
        for name, item in entries.items()
    }
    ordered_names = [name for name in preferred_order if name in normalized]
    ordered_names.extend(sorted(set(normalized) - set(ordered_names)))
    results: list[dict[str, Any]] = []
    for name in ordered_names:
        item = normalized[name]
        results.append({
            "id": name,
            "label": name.replace("_", " ").title(),
            "attempts": _dashboard_number(
                item, "attempts", "self_right_attempts", "self_righting_attempts"
            ),
            "successes": _dashboard_number(
                item, "successes", "self_right_successes", "self_righting_successes"
            ),
            "successRate": _dashboard_number(
                item, "success_rate", "self_right_success_rate", "recovery_rate"
            ),
            "latencyMeanS": _dashboard_number(
                item, "recovery_latency_mean_s", "mean_recovery_latency_s", "latency_mean_s"
            ),
            "latencyP95S": _dashboard_number(
                item, "recovery_latency_p95_s", "p95_recovery_latency_s", "latency_p95_s"
            ),
            "rerollCount": _dashboard_number(
                item,
                "self_right_then_reroll_count",
                "recovered_then_rerolled_count",
                "reroll_count",
            ),
            "frontierAfterRecoveryM": _dashboard_number(
                item,
                "frontier_after_recovery_m",
                "frontier_distance_after_recovery_m",
            ),
            "pass": _dashboard_bool(item, "pass", "recovery_pass", "acceptance_pass"),
        })
    return results


def _roll_sprint_evaluation_payload(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    passes = {
        "overall": _dashboard_bool(
            payload, "promotion_pass", "acceptance_pass", "evaluation_pass"
        ),
        "recovery": _dashboard_bool(
            payload,
            "recovery_battery.overall_pass",
            "self_right_recovery_pass",
            "recovery_pass",
            "recovery.overall_pass",
        ),
        "reroll": _dashboard_bool(
            payload, "self_right_then_reroll_pass", "recovery.reroll_pass"
        ),
        "raceFrontier": _dashboard_bool(
            payload,
            "race_frontier_improvement_pass",
            "recovery_battery.race_frontier_improved_over_parent",
            "race_frontier_retention_pass",
            "race.frontier_retention_pass",
        ),
        "sharedRoad": _dashboard_bool(
            payload,
            "four_robot_batch_road_corridor_pass",
            "straight_lane_batch_pass",
            "four_robot_batch_straight_lane_pass",
        ),
        "target10m": _dashboard_bool(
            payload,
            "four_robot_batch_target_10m_pass",
            "four_robot_batch_target_20m_pass",
        ),
    }
    nan_count = _dashboard_number(payload, "nan_env_count")
    out_of_bounds_count = _dashboard_number(payload, "out_of_bounds_env_count")
    reroll_rate = _dashboard_number(
        payload,
        "self_right_then_reroll_rate",
        "recovery_battery.self_right_then_reroll_rate",
    )
    if passes["reroll"] is None and reroll_rate is not None:
        passes["reroll"] = reroll_rate >= 0.5
    standing_target_rate = _dashboard_number(
        payload, "standing_on_road_target_reach_rate"
    )
    if standing_target_rate is not None:
        passes["target10m"] = standing_target_rate >= 0.75
    passes["finite"] = (
        nan_count == 0 and out_of_bounds_count == 0
        if nan_count is not None and out_of_bounds_count is not None
        else None
    )
    return {
        "available": True,
        "file": path.name,
        "modified": _iso_timestamp(path.stat().st_mtime),
        "checkpoint": Path(str(payload.get("checkpoint", ""))).name or None,
        "checkpointIteration": _dashboard_number(payload, "checkpoint_iteration"),
        "checkpointSha256": payload.get("checkpoint_sha256"),
        "meanFrontierM": _dashboard_number(
            payload, "mean_credited_forward_frontier_m", "mean_roll_linked_distance_m"
        ),
        "parentFrontierM": _dashboard_number(
            payload, "recovery_battery.parent_frontier_m"
        ),
        "frontierDeltaM": _dashboard_number(
            payload, "recovery_battery.race_frontier_delta_to_parent_m"
        ),
        "selfRightAttempts": _dashboard_number(
            payload,
            "total_self_right_attempt_count",
            "self_right_attempt_count",
            "self_right_attempts",
            "recovery_battery.total_attempts",
            "self_righting.attempts",
            "recovery.attempts",
        ),
        "selfRightSuccesses": _dashboard_number(
            payload,
            "total_self_right_success_count",
            "self_right_success_count",
            "self_right_successes",
            "recovery_battery.total_successes",
            "self_righting.successes",
            "recovery.successes",
        ),
        "selfRightSuccessRate": _dashboard_number(
            payload,
            "self_right_success_rate",
            "recovery_battery.success_rate",
            "self_righting.success_rate",
            "recovery.success_rate",
        ),
        "recoveryLatencyMeanS": _dashboard_number(
            payload,
            "recovery_latency_mean_s",
            "recovery_battery.recovery_latency_mean_s",
            "mean_self_right_latency_s",
            "mean_recovery_latency_s",
            "self_righting.latency_mean_s",
        ),
        "recoveryLatencyP95S": _dashboard_number(
            payload,
            "recovery_latency_p95_s",
            "recovery_battery.recovery_latency_p95_s",
            "p95_self_right_latency_s",
            "p95_recovery_latency_s",
            "self_righting.latency_p95_s",
        ),
        "selfRightThenRerollCount": _dashboard_number(
            payload,
            "total_self_right_then_reroll_count",
            "self_right_then_reroll_count",
            "recovery_battery.self_right_then_reroll_count",
            "total_recovered_and_rerolled_count",
        ),
        "selfRightThenRerollRate": reroll_rate,
        "frontierAfterRecoveryM": _dashboard_number(
            payload,
            "frontier_distance_after_recovery_m",
            "credited_frontier_after_recovery_m",
            "recovery_battery.frontier_after_recovery_m",
            "self_righting.frontier_after_recovery_m",
        ),
        "roadReturnCount": _dashboard_number(
            payload,
            "recovery_battery.lane_reposition_count",
            "total_lane_reposition_count",
            "lane_reposition_count",
        ),
        "roadReturnLatencyMeanS": _dashboard_number(
            payload,
            "recovery_battery.lane_reposition_latency_mean_s",
            "mean_lane_reposition_latency_s",
            "lane_reposition_latency_mean_s",
        ),
        "roadExitEnvCount": _dashboard_number(payload, "road_exit_env_count"),
        "maximumRoadOvershootM": _dashboard_number(
            payload, "maximum_road_boundary_overshoot_m"
        ),
        "standingOnRoadTargetRate": standing_target_rate,
        "standingOnRoadWinnerRobotIndex": _dashboard_number(
            payload, "standing_on_road_winner_robot_index"
        ),
        "orientations": _orientation_dashboard_metrics(payload),
        "passes": passes,
    }


def _latest_roll_sprint_evaluation() -> dict[str, Any]:
    """Load the newest valid roll-sprint evaluator JSON for the live summary."""

    if not ROLL_SPRINT_EVALUATION_ROOT.is_dir():
        return {"available": False}
    try:
        candidates = sorted(
            ROLL_SPRINT_EVALUATION_ROOT.glob("*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return {"available": False}
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return _roll_sprint_evaluation_payload(payload, path)
        except (OSError, ValueError):
            continue
    return {"available": False}


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


def _curated_media_candidates(
    featured_videos: set[str] | None,
    featured_prefixes: tuple[str, ...],
) -> set[tuple[str, Path]]:
    """Resolve curated paths without traversing the full training log tree."""

    candidates: set[tuple[str, Path]] = set()
    if featured_videos is None:
        return candidates
    for media_key in featured_videos:
        source, separator, relative = media_key.partition("/")
        root = MEDIA_ROOTS.get(source)
        if separator and root is not None:
            candidates.add((source, root / relative))
    for prefix in featured_prefixes:
        source, separator, relative = prefix.partition("/")
        root = MEDIA_ROOTS.get(source)
        if not separator or root is None:
            continue
        curated_root = root / relative
        if curated_root.is_dir():
            candidates.update(
                (source, path) for path in curated_root.rglob("*") if path.is_file()
            )
    return candidates


def _discover_media() -> list[dict[str, Any]]:
    media: list[dict[str, Any]] = []
    featured_videos, featured_prefixes, show_images = _featured_media_config()
    curated = _curated_media_candidates(featured_videos, featured_prefixes)
    sources = (
        curated
        if featured_videos is not None
        else {
            (source, path)
            for source, root in MEDIA_ROOTS.items()
            if root.is_dir()
            for path in root.rglob("*")
            if path.is_file()
        }
    )
    for source, path in sources:
        root = MEDIA_ROOTS[source]
        try:
            if path.suffix.lower() not in MEDIA_EXTENSIONS:
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
                "collection": _media_collection(source, relative_path),
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
    return media


def _media_collection(source: str, relative_path: str) -> str:
    """Map a video path to the dashboard's selectable training run."""

    normalized = f"{source}/{relative_path}".lower().replace("_", "-")
    if "roll-sprint" in normalized or "rollsprint" in normalized:
        return "roll-sprint"
    if "stair" in normalized:
        return "stairs"
    # Curated legacy media is staircase media by convention. Keep unknown
    # images/videos available without making them look like the new run.
    return "stairs"


def _media_absolute_path(item: dict[str, Any]) -> Path | None:
    root = MEDIA_ROOTS.get(str(item.get("source", "")))
    relative = item.get("path")
    if root is None or not isinstance(relative, str):
        return None
    return (root / relative).resolve()


def _champion_video(
    checkpoint_sha256: str,
    media: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Return the newest visible MP4 recorded from this exact champion hash."""

    companion_suffix = f"-{checkpoint_sha256[:12]}.mp4"
    for item in media:
        name = str(item.get("name", ""))
        if (
            item.get("kind") == "video"
            and name.startswith("champion-")
            and name.endswith(companion_suffix)
        ):
            return dict(item)
    if not ROLL_SPRINT_SAMPLE_ROOT.is_dir():
        return None
    recorded_paths: set[Path] = set()
    try:
        state_files = list(ROLL_SPRINT_SAMPLE_ROOT.glob("*.json"))
    except OSError:
        return None
    for state_file in state_files:
        try:
            payload = json.loads(state_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        checkpoint = payload.get("last_checkpoint") if isinstance(payload, dict) else None
        video = payload.get("last_video") if isinstance(payload, dict) else None
        if (
            isinstance(checkpoint, dict)
            and checkpoint.get("sha256") == checkpoint_sha256
            and isinstance(video, str)
        ):
            candidate = Path(video).expanduser().resolve()
            if candidate.is_file() and candidate.suffix.lower() == ".mp4":
                recorded_paths.add(candidate)
    for item in media:
        if item.get("kind") != "video":
            continue
        absolute = _media_absolute_path(item)
        if absolute in recorded_paths:
            return dict(item)
    return None


def _roll_sprint_champion(media: list[dict[str, Any]]) -> dict[str, Any]:
    """Normalize the retained champion and its exact-hash video for the UI."""

    try:
        payload = json.loads(
            ROLL_SPRINT_CHAMPION_MANIFEST.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return {"available": False}
    if not isinstance(payload, dict):
        return {"available": False}
    checkpoint_sha256 = payload.get("checkpoint_sha256")
    retained_value = payload.get("retained_checkpoint")
    source_value = payload.get("source_checkpoint")
    if not all(
        isinstance(value, str) and value
        for value in (checkpoint_sha256, retained_value, source_value)
    ):
        return {"available": False}
    retained = Path(retained_value)
    source = Path(source_value)
    match = re.fullmatch(r"model_(\d+)\.pt", retained.name)
    featured_video = None
    featured_value = payload.get("featured_video")
    if isinstance(featured_value, str) and featured_value:
        featured_path = Path(featured_value).expanduser().resolve()
        featured_video = next(
            (
                dict(item)
                for item in media
                if item.get("kind") == "video"
                and _media_absolute_path(item) == featured_path
            ),
            None,
        )
    featured_sha256 = payload.get("featured_video_checkpoint_sha256")
    if not isinstance(featured_sha256, str) or not featured_sha256:
        featured_sha256 = None
    return {
        "available": True,
        "manifest": ROLL_SPRINT_CHAMPION_MANIFEST.name,
        "manifestSchemaVersion": _dashboard_number(payload, "schema_version"),
        "evaluationSchemaVersion": _dashboard_number(
            payload, "evaluation_schema_version"
        ),
        "version": source.parent.name,
        "sourceCheckpoint": source.name,
        "retainedCheckpoint": retained.name,
        "checkpointIteration": int(match.group(1)) if match else None,
        "checkpointSha256": checkpoint_sha256,
        "checkpointHash": checkpoint_sha256[:12],
        "targetDistanceReachCount": _dashboard_number(
            payload, "target_distance_reach_count"
        ),
        "meanFrontierM": _dashboard_number(
            payload, "mean_credited_forward_frontier_m"
        ),
        "meanTimeTo10mS": _dashboard_number(payload, "mean_time_to_valid_10m_s"),
        "slowestTimeTo10mS": _dashboard_number(
            payload, "slowest_time_to_valid_10m_s"
        ),
        "featuredVideoCheckpointIteration": _dashboard_number(
            payload, "featured_video_checkpoint_iteration"
        ),
        "featuredVideoCheckpointSha256": featured_sha256,
        "featuredVideoCheckpointHash": (
            featured_sha256[:12] if featured_sha256 else None
        ),
        "videoIsFeatured": featured_video is not None,
        "video": featured_video or _champion_video(checkpoint_sha256, media),
    }


def dashboard_state(*, include_metrics: bool = True) -> dict[str, Any]:
    media = _discover_media()
    video_counts = {
        collection_id: sum(
            item["kind"] == "video" and item["collection"] == collection_id
            for item in media
        )
        for collection_id in VIDEO_COLLECTIONS
    }
    video_collections = [
        {
            "id": collection_id,
            **definition,
            "videoCount": video_counts[collection_id],
        }
        for collection_id, definition in VIDEO_COLLECTIONS.items()
    ]
    return {
        "generatedAt": _iso_timestamp(time.time()),
        "repo": REPO_ROOT.name,
        "media": media,
        "videoCollections": video_collections,
        "defaultVideoCollection": "roll-sprint",
        "rollSprintChampion": _roll_sprint_champion(media),
        "rollSprintEvaluation": (
            _latest_roll_sprint_evaluation()
            if include_metrics
            else {"available": False}
        ),
        "summary": {
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
    """Build the retained video snapshot once per browser request."""

    return dashboard_state(include_metrics=True)


class DashboardHandler(SimpleHTTPRequestHandler):
    """Static files plus a read-only dashboard API."""

    server_version = "MicroDuckDashboard/1.0"

    def do_GET(self) -> None:
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

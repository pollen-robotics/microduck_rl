"""Acceptance gates and the metrics summary for the crowd-following demo.

Kept apart from the runner so the gates can be unit-tested against synthetic
rollouts on CPU, with no model, policy or renderer involved. A demo whose
success criteria can only be checked by running the demo is not validated.
"""

import math
from itertools import pairwise

import numpy as np

# --- Acceptance thresholds -------------------------------------------------
# A FOLLOW segment must produce real locomotion, not a policy that emits a
# command and stands there. Both a path-length and a net-displacement floor are
# required: walking in a circle passes the first and fails the second.
MIN_FOLLOW_PATH_M = 0.40
MIN_FOLLOW_DISPLACEMENT_M = 0.30

# ...and it must actually close distance on its target at some point, so that
# "moved" cannot be satisfied by wandering away.
MIN_FOLLOW_APPROACH_M = 0.05

# Trunk height below which the robot counts as fallen (nominal is ~0.116 m).
FALLEN_TRUNK_Z_M = 0.09

# States in which the locomotion command must be exactly zero.
STATIONARY_STATES = ("SEARCH", "FOUND", "STOP", "DONE")


def follow_segment_metrics(records, sequence):
    """Per-selection FOLLOW statistics, in requested order."""
    follow_rows = [r for r in records if r["state"] == "FOLLOW"]
    segments = []
    for selection_index, target_color in enumerate(sequence, start=1):
        rows = [r for r in follow_rows if r["selection"] == selection_index]
        if not rows:
            raise RuntimeError(
                f"selection {selection_index} ({target_color}) never followed"
            )
        positions = [np.asarray(r["duck_xy"], dtype=np.float64) for r in rows]
        path_distance = sum(
            float(np.linalg.norm(b - a)) for a, b in pairwise(positions)
        )
        segments.append(
            {
                "selection": selection_index,
                "target": target_color,
                "path_distance_m": path_distance,
                "net_displacement_m": float(
                    np.linalg.norm(positions[-1] - positions[0])
                ),
                "start_error_m": rows[0]["follow_error_m"],
                "end_error_m": rows[-1]["follow_error_m"],
                "min_error_m": min(r["follow_error_m"] for r in rows),
                "start_yaw_error_deg": rows[0]["yaw_error_deg"],
                "end_yaw_error_deg": rows[-1]["yaw_error_deg"],
            }
        )
    return segments


def segment_moved(segment) -> bool:
    return (
        segment["path_distance_m"] >= MIN_FOLLOW_PATH_M
        and segment["net_displacement_m"] >= MIN_FOLLOW_DISPLACEMENT_M
    )


def segment_approached(segment) -> bool:
    return segment["min_error_m"] <= segment["start_error_m"] - MIN_FOLLOW_APPROACH_M


def summarize(
    *,
    records,
    transitions,
    cycles,
    sequence,
    duration_s,
    control_steps,
    frames,
    trail_distance_m,
    camera_stats,
):
    """Build the metrics dictionary written next to a rollout."""
    follow_rows = [r for r in records if r["state"] == "FOLLOW"]
    if not follow_rows:
        raise RuntimeError("rollout contains no FOLLOW steps")
    segments = follow_segment_metrics(records, sequence)

    return {
        "duration_s": duration_s,
        "control_steps": control_steps,
        "frames": frames,
        "target_sequence_requested": list(sequence),
        "target_sequence_completed": [c["target"] for c in cycles],
        "pattern": ["SEARCH", "FOUND", "FOLLOW", "STOP"],
        "cycles_completed": len(cycles),
        "cycles": cycles,
        "follow_by_selection": segments,
        "all_follow_segments_moved": all(segment_moved(s) for s in segments),
        "all_follow_segments_approached": all(segment_approached(s) for s in segments),
        "transitions": transitions,
        "trail_distance_m": trail_distance_m,
        "follow_rmse_m": math.sqrt(
            sum(r["follow_error_m"] ** 2 for r in follow_rows) / len(follow_rows)
        ),
        "follow_max_m": max(r["follow_error_m"] for r in follow_rows),
        "person_range_follow_mean_m": sum(r["person_range_m"] for r in follow_rows)
        / len(follow_rows),
        "person_range_follow_min_m": min(r["person_range_m"] for r in follow_rows),
        "person_range_follow_max_m": max(r["person_range_m"] for r in follow_rows),
        "search_duration_mean_s": sum(c["search_duration_s"] for c in cycles)
        / len(cycles),
        "search_duration_max_s": max(c["search_duration_s"] for c in cycles),
        # Every FOUND transition must coincide with a step in which the camera
        # actually reported the requested color visible.
        "found_while_target_visible": all(
            any(
                r["target_visible"]
                for r in records
                if r["target"] == cycle["target"]
                and abs(r["t"] - cycle["found_s"]) < 0.04
            )
            for cycle in cycles
        ),
        # Any completed cycle whose target differs from the requested color at
        # that position is a wrong-color lock.
        "wrong_color_locks": sum(
            cycle["target"] != sequence[index] for index, cycle in enumerate(cycles)
        ),
        "camera_target_visible_follow_pct": 100.0
        * sum(r["target_visible"] for r in follow_rows)
        / len(follow_rows),
        "stationary_state_command_max": max(
            float(np.linalg.norm(r["command"]))
            for r in records
            if r["state"] in STATIONARY_STATES
        ),
        "min_trunk_z_m": min(r["trunk_z_m"] for r in records),
        "final_trunk_z_m": records[-1]["trunk_z_m"],
        "fallen_steps": sum(r["trunk_z_m"] < FALLEN_TRUNK_Z_M for r in records),
        **camera_stats,
    }


def check_gates(summary, sequence) -> list[str]:
    """Return the list of violated acceptance gates (empty means pass)."""
    failures = []
    if summary["target_sequence_completed"] != list(sequence):
        failures.append(
            f"completed sequence {summary['target_sequence_completed']} != "
            f"requested {list(sequence)}"
        )
    if summary["cycles_completed"] != len(sequence):
        failures.append(
            f"{summary['cycles_completed']} cycles completed, expected {len(sequence)}"
        )
    if not summary["found_while_target_visible"]:
        failures.append("a FOUND transition happened while the target was not visible")
    if summary["wrong_color_locks"] != 0:
        failures.append(f"wrong_color_locks={summary['wrong_color_locks']}")
    if summary["camera_target_visible_follow_pct"] < 100.0:
        failures.append(
            "target not visible for 100% of FOLLOW steps "
            f"({summary['camera_target_visible_follow_pct']:.1f}%)"
        )
    if not summary["all_follow_segments_moved"]:
        failures.append("a FOLLOW segment did not produce locomotion")
    if not summary["all_follow_segments_approached"]:
        failures.append("a FOLLOW segment never approached its target")
    if summary["stationary_state_command_max"] != 0.0:
        failures.append(
            "non-zero locomotion command in a stationary state "
            f"({summary['stationary_state_command_max']})"
        )
    if summary["fallen_steps"] != 0:
        failures.append(f"fallen_steps={summary['fallen_steps']}")
    if summary["min_trunk_z_m"] <= FALLEN_TRUNK_Z_M:
        failures.append(f"min trunk z={summary['min_trunk_z_m']:.4f} m")
    return failures

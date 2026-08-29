#!/usr/bin/env python3
"""Keep one native viewer on the best physics-gated stair checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    from scripts.select_best_stair_preview import (
        build_manifest,
        select_best_checkpoint,
        write_json_atomic,
    )
except ModuleNotFoundError:  # Direct execution puts scripts/ on sys.path.
    from select_best_stair_preview import (
        build_manifest,
        select_best_checkpoint,
        write_json_atomic,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "Mjlab-Stairs-Route-MicroDuck"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def physics_improves(candidate: dict[str, object], current: dict[str, object]) -> bool:
    """Require a real success or a material route/height advance."""
    candidate_success = int(candidate["successes"])
    current_success = int(current["successes"])
    if candidate_success != current_success:
        return candidate_success > current_success

    candidate_x = float(
        candidate.get("best_corridor_route_x_m", candidate["best_route_x_m"])
    )
    current_x = float(
        current.get("best_corridor_route_x_m", current["best_route_x_m"])
    )
    if candidate_x >= current_x + 0.02:
        return True

    candidate_z = float(candidate["best_root_height_m"])
    current_z = float(current["best_root_height_m"])
    return candidate_x >= current_x - 0.03 and candidate_z >= current_z + 0.02


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def _report_matches(
    report: dict[str, object], checkpoint: Path, walker_checkpoint: Path
) -> bool:
    return (
        report.get("checkpoint_sha256") == _sha256(checkpoint)
        and report.get("walker_checkpoint_sha256") == _sha256(walker_checkpoint)
    )


class NativeViewer:
    def __init__(self, log_path: Path, walker_checkpoint: Path):
        self.log_path = log_path
        self.walker_checkpoint = walker_checkpoint
        self.process: subprocess.Popen[bytes] | None = None
        self._log_handle = None

    def start(self, checkpoint: Path) -> None:
        self.stop()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.log_path.open("ab", buffering=0)
        environment = os.environ.copy()
        environment.setdefault("PYTHONUTF8", "1")
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "view_stair_handoff.py"),
            str(checkpoint),
            "--walker-checkpoint",
            str(self.walker_checkpoint),
            "--device",
            "cpu",
        ]
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        self.process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
        )

    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(self.process.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            else:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
        self.process = None
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None


def _evaluate(
    checkpoint: Path,
    walker_checkpoint: Path,
    output: Path,
    *,
    num_envs: int,
    steps: int,
    device: str,
) -> dict[str, object]:
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "evaluate_stair_checkpoint.py"),
        str(checkpoint),
        "--walker-checkpoint",
        str(walker_checkpoint),
        "--num-envs",
        str(num_envs),
        "--steps",
        str(steps),
        "--device",
        device,
        "--output",
        str(output),
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if completed.returncode != 0 or not output.is_file():
        raise RuntimeError(
            f"Physics evaluation failed for {checkpoint.name} (exit {completed.returncode})"
        )
    return _load_json(output)


def _record_video(
    checkpoint: Path,
    walker_checkpoint: Path,
    video_dir: Path,
    *,
    steps: int,
    device: str,
) -> Path:
    output = _video_path(checkpoint, video_dir)
    if output.is_file() and output.stat().st_size > 0:
        return output.resolve()

    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "record_stair_policy.py"),
        str(checkpoint),
        str(output),
        "--walker-checkpoint",
        str(walker_checkpoint),
        "--steps",
        str(steps),
        "--device",
        device,
    ]
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if completed.returncode != 0 or not output.is_file():
        raise RuntimeError(
            f"Video recording failed for {checkpoint.name} (exit {completed.returncode})"
        )
    return output.resolve()


def _video_path(checkpoint: Path, video_dir: Path) -> Path:
    iteration = int(checkpoint.stem.rsplit("_", 1)[-1])
    checkpoint_id = _sha256(checkpoint)[:10]
    return (
        video_dir
        / f"standard-home-stairs-{checkpoint_id}-iter-{iteration:05d}.mp4"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--walker-checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--eval-num-envs", type=int, default=8)
    parser.add_argument("--eval-steps", type=int, default=1_500)
    parser.add_argument("--eval-device", default="cuda:0")
    parser.add_argument("--video-steps", type=int, default=700)
    parser.add_argument("--video-device", default="cpu")
    parser.add_argument(
        "--record-initial-video",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Record the starting checkpoint too; disabled so mass evaluation stays headless.",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "artifacts"
            / "verified"
            / "stair-policy-promotions"
        ),
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=REPO_ROOT / "artifacts" / "verified" / "live-preview-state.json",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_dir = args.run_dir.resolve()
    walker_checkpoint = args.walker_checkpoint.expanduser().resolve()
    if not walker_checkpoint.is_file():
        raise SystemExit(f"Missing walker checkpoint: {walker_checkpoint}")
    baseline_report = _load_json(args.baseline_report.resolve())
    current_report = baseline_report
    current_checkpoint = run_dir / "model_0.pt"
    if not current_checkpoint.is_file():
        raise SystemExit(f"Missing baseline checkpoint: {current_checkpoint}")
    expected_specialist_hash = _sha256(current_checkpoint)
    expected_walker_hash = _sha256(walker_checkpoint)
    if baseline_report.get("checkpoint_sha256") != expected_specialist_hash:
        raise SystemExit("Baseline report does not match model_0.pt")
    if baseline_report.get("walker_checkpoint_sha256") != expected_walker_hash:
        raise SystemExit("Baseline report does not match the immutable walker checkpoint")

    manifest_path = args.state.resolve().with_name("best-stair-preview.json")
    evaluation_dir = run_dir / "stair-evaluations"
    current_video: Path | None = None
    viewer = NativeViewer(
        REPO_ROOT / ".tmp" / "codex" / "best-preview-viewer.log",
        walker_checkpoint,
    )
    evaluated: dict[str, dict[str, object]] = {str(current_checkpoint.resolve()): baseline_report}
    rejections: dict[str, str] = {}
    if evaluation_dir.is_dir():
        existing_reports = sorted(
            evaluation_dir.glob("model_*.json"),
            key=lambda path: int(path.stem.rsplit("_", 1)[-1]),
        )
        for report_path in existing_reports:
            checkpoint = run_dir / f"{report_path.stem}.pt"
            if not checkpoint.is_file():
                continue
            if checkpoint.resolve() == current_checkpoint.resolve():
                continue
            report = _load_json(report_path)
            if not _report_matches(report, checkpoint, walker_checkpoint):
                continue
            key = str(checkpoint.resolve())
            evaluated[key] = report
            if physics_improves(report, current_report):
                current_checkpoint = checkpoint
                current_report = report
            else:
                rejections[key] = (
                    "Rejected because full-height physics did not materially improve "
                    "success, route distance, or root height."
                )

    def publish(reason: str) -> None:
        state = {
            "schema_version": 1,
            "run_directory": str(run_dir),
            "viewer_checkpoint": str(current_checkpoint.resolve()),
            "viewer_evaluation": current_report,
            "viewer_running": viewer.running(),
            "viewer_video": str(current_video) if current_video else None,
            "reason": reason,
            "rejections": rejections,
        }
        write_json_atomic(args.state.resolve(), state)

    viewer.start(current_checkpoint)
    video_dir = args.video_dir.resolve()
    existing_video = _video_path(current_checkpoint, video_dir)
    current_video = existing_video.resolve() if existing_video.is_file() else None
    if args.record_initial_video and current_video is None:
        current_video = _record_video(
            current_checkpoint,
            walker_checkpoint,
            video_dir,
            steps=args.video_steps,
            device=args.video_device,
        )
    publish(
        "Showing the best existing physics-gated checkpoint; new candidates are checked automatically."
    )
    try:
        while True:
            if not viewer.running():
                viewer.start(current_checkpoint)
            try:
                ranking = select_best_checkpoint(run_dir)
                write_json_atomic(manifest_path, build_manifest(ranking, run_dir))
                periodic_candidates = [
                    item
                    for item in ranking.candidates
                    if item.iteration > 0
                    and item.iteration % 200 == 0
                    and str(item.path.resolve()) not in evaluated
                    and str(item.path.resolve()) not in rejections
                ]
                immediate_candidates = [
                    item
                    for item in ranking.candidates
                    if item.iteration > 0
                    and item.metrics["full_stair_success"].value > 0.0
                    and str(item.path.resolve()) not in evaluated
                    and str(item.path.resolve()) not in rejections
                ]
                if immediate_candidates:
                    candidate = max(
                        immediate_candidates, key=lambda item: item.iteration
                    ).path.resolve()
                elif periodic_candidates:
                    candidate = max(
                        periodic_candidates, key=lambda item: item.iteration
                    ).path.resolve()
                else:
                    candidate = ranking.selected.path.resolve()
                if candidate != current_checkpoint.resolve():
                    key = str(candidate)
                    if key not in evaluated and key not in rejections:
                        evaluation_path = evaluation_dir / f"{candidate.stem}.json"
                        report = _evaluate(
                            candidate,
                            walker_checkpoint,
                            evaluation_path,
                            num_envs=args.eval_num_envs,
                            steps=args.eval_steps,
                            device=args.eval_device,
                        )
                        evaluated[key] = report
                        if physics_improves(report, current_report):
                            promoted_video = _record_video(
                                candidate,
                                walker_checkpoint,
                                video_dir,
                                steps=args.video_steps,
                                device=args.video_device,
                            )
                            current_checkpoint = candidate
                            current_report = report
                            current_video = promoted_video
                            viewer.start(current_checkpoint)
                            publish(
                                f"Promoted {candidate.name} after full-height physics evaluation."
                            )
                        else:
                            rejections[key] = (
                                "Rejected because full-height physics did not materially improve "
                                "success, route distance, or root height."
                            )
                            publish(rejections[key])
                else:
                    publish(ranking.reason)
            except Exception as exc:  # noqa: BLE001
                # A transient event-file parse or evaluator error must never
                # kill the last physics-proven viewer.
                publish(
                    "Preview monitor kept the current checkpoint: "
                    f"{type(exc).__name__}: {exc}"
                )
            time.sleep(max(args.poll_seconds, 5.0))
    except KeyboardInterrupt:
        return 0
    finally:
        viewer.stop()


if __name__ == "__main__":
    raise SystemExit(main())

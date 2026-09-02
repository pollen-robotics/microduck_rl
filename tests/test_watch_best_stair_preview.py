import sys

from scripts import train_stair_from_walking, watch_best_stair_preview
from scripts.watch_best_stair_preview import physics_improves


def _report(successes=0, x=0.60, z=0.13):
    return {
        "successes": successes,
        "best_route_x_m": x,
        "best_root_height_m": z,
    }


def test_physics_gate_prefers_success_then_material_progress_or_height():
    baseline = _report()
    assert physics_improves(_report(successes=1), baseline)
    assert physics_improves(_report(x=0.63), baseline)
    assert physics_improves(_report(x=0.58, z=0.16), baseline)
    assert not physics_improves(_report(x=0.61, z=0.14), baseline)
    assert not physics_improves(_report(x=0.55, z=0.20), baseline)


def test_physics_gate_rejects_lateral_progress_outside_stair_corridor():
    baseline = _report()
    candidate = _report(x=0.90)
    candidate["best_corridor_route_x_m"] = 0.59
    candidate["best_any_route_x_m"] = 0.90
    assert not physics_improves(candidate, baseline)


def test_video_recording_is_opt_in_for_headless_mass_evaluation(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "watch_best_stair_preview.py",
            str(tmp_path / "run"),
            "--walker-checkpoint",
            str(tmp_path / "walker.pt"),
            "--baseline-report",
            str(tmp_path / "baseline.json"),
        ],
    )

    args = watch_best_stair_preview._parse_args()

    assert args.record_initial_video is False
    assert args.video_dir.name == "stair-policy-promotions"


def test_stair_finetune_video_recording_is_disabled_by_default(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_stair_from_walking.py"])

    args = train_stair_from_walking._parse_args()

    assert args.video is False

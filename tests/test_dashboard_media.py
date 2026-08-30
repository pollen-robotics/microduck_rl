import json
import os

from scripts import serve_dashboard


def test_featured_media_can_be_video_only_by_prefix(tmp_path, monkeypatch) -> None:
    config = tmp_path / "featured_media.json"
    config.write_text(
        json.dumps(
            {
                "showImages": False,
                "featuredVideos": [],
                "featuredVideoPrefixes": [
                    "artifacts/verified/stair-policy-evolution/"
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(serve_dashboard, "FEATURED_MEDIA_FILE", config)

    exact, prefixes, show_images = serve_dashboard._featured_media_config()

    assert exact == set()
    assert prefixes == ("artifacts/verified/stair-policy-evolution/",)
    assert show_images is False


def test_dashboard_lists_all_curated_videos_newest_first(tmp_path, monkeypatch) -> None:
    artifacts = tmp_path / "artifacts"
    promotions = artifacts / "verified" / "stair-policy-promotions"
    promotions.mkdir(parents=True)
    for index in range(12):
        video = promotions / f"promotion-{index:02d}.mp4"
        video.write_bytes(b"video")
        os.utime(video, (index + 1, index + 1))
    (promotions / "ignored.png").write_bytes(b"image")
    unrelated = artifacts / "old-failures"
    unrelated.mkdir()
    (unrelated / "failure.mp4").write_bytes(b"failure")

    config = tmp_path / "featured_media.json"
    config.write_text(
        json.dumps(
            {
                "showImages": False,
                "featuredVideos": [],
                "featuredVideoPrefixes": [
                    "artifacts/verified/stair-policy-promotions/"
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(serve_dashboard, "FEATURED_MEDIA_FILE", config)
    monkeypatch.setattr(serve_dashboard, "MEDIA_ROOTS", {"artifacts": artifacts})

    media = serve_dashboard._discover_media()

    assert len(media) == 12
    assert all(item["kind"] == "video" for item in media)
    assert media[0]["name"] == "promotion-11.mp4"
    assert media[-1]["name"] == "promotion-00.mp4"


def test_dashboard_separates_roll_sprint_and_stair_video_collections(tmp_path, monkeypatch) -> None:
    media_root = tmp_path / "artifacts"
    roll_dir = media_root / "training" / "roll-sprint-samples"
    stair_dir = media_root / "training" / "stair-policy-samples"
    roll_dir.mkdir(parents=True)
    stair_dir.mkdir(parents=True)
    (roll_dir / "roll-0001.mp4").write_bytes(b"roll")
    (stair_dir / "stair-0001.mp4").write_bytes(b"stair")
    monkeypatch.setattr(serve_dashboard, "FEATURED_MEDIA_FILE", tmp_path / "missing.json")
    monkeypatch.setattr(serve_dashboard, "MEDIA_ROOTS", {"artifacts": media_root})

    state = serve_dashboard.dashboard_state(include_metrics=False)
    by_name = {item["name"]: item["collection"] for item in state["media"]}

    assert by_name == {"roll-0001.mp4": "roll-sprint", "stair-0001.mp4": "stairs"}
    assert state["defaultVideoCollection"] == "roll-sprint"
    counts = {item["id"]: item["videoCount"] for item in state["videoCollections"]}
    assert counts == {"roll-sprint": 1, "stairs": 1}


def test_dashboard_normalizes_latest_self_righting_evaluation(tmp_path, monkeypatch) -> None:
    evaluation_root = tmp_path / "evaluations"
    evaluation_root.mkdir()
    report = evaluation_root / "checkpoint-000300-deadbeef.json"
    report.write_text(
        json.dumps(
            {
                "checkpoint": str(tmp_path / "model_300.pt"),
                "checkpoint_iteration": 300,
                "checkpoint_sha256": "deadbeef",
                "mean_credited_forward_frontier_m": 1.25,
                "recovery_battery": {
                    "total_attempts": 16,
                    "total_successes": 14,
                    "success_rate": 0.875,
                    "recovery_latency_mean_s": 2.4,
                    "recovery_latency_p95_s": 5.7,
                    "self_right_then_reroll_count": 11,
                    "self_right_then_reroll_rate": 0.6875,
                    "frontier_after_recovery_m": 8.25,
                    "lane_reposition_count": 3,
                    "lane_reposition_latency_mean_s": 0.8,
                    "parent_frontier_m": 1.0,
                    "race_frontier_delta_to_parent_m": 0.25,
                    "race_frontier_improved_over_parent": True,
                    "by_orientation": {
                        "face_down": {
                            "attempts": 4,
                            "successes": 4,
                            "success_rate": 1.0,
                            "recovery_latency_mean_s": 1.9,
                            "recovery_latency_p95_s": 2.7,
                            "self_right_then_reroll_count": 3,
                            "frontier_after_recovery_m": 2.5,
                            "pass": True,
                        },
                        "right": {
                            "attempts": 4,
                            "successes": 3,
                            "success_rate": 0.75,
                            "pass": True,
                        },
                    },
                    "overall_pass": True,
                },
                "acceptance_pass": False,
                "promotion_pass": True,
                "race_frontier_improvement_pass": True,
                "race_frontier_retention_pass": True,
                "straight_lane_batch_pass": True,
                "four_robot_batch_target_20m_pass": False,
                "p95_lateral_drift_pass": True,
                "nan_env_count": 0,
                "out_of_bounds_env_count": 0,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(serve_dashboard, "ROLL_SPRINT_EVALUATION_ROOT", evaluation_root)

    evaluation = serve_dashboard._latest_roll_sprint_evaluation()

    assert evaluation["available"] is True
    assert evaluation["checkpoint"] == "model_300.pt"
    assert evaluation["meanFrontierM"] == 1.25
    assert evaluation["parentFrontierM"] == 1.0
    assert evaluation["frontierDeltaM"] == 0.25
    assert evaluation["selfRightAttempts"] == 16
    assert evaluation["selfRightSuccesses"] == 14
    assert evaluation["selfRightSuccessRate"] == 0.875
    assert evaluation["recoveryLatencyMeanS"] == 2.4
    assert evaluation["recoveryLatencyP95S"] == 5.7
    assert evaluation["selfRightThenRerollCount"] == 11
    assert evaluation["selfRightThenRerollRate"] == 0.6875
    assert evaluation["frontierAfterRecoveryM"] == 8.25
    assert evaluation["laneRepositionCount"] == 3
    assert evaluation["laneRepositionLatencyMeanS"] == 0.8
    assert [item["id"] for item in evaluation["orientations"]] == [
        "face_down",
        "right_side",
    ]
    assert evaluation["orientations"][0]["frontierAfterRecoveryM"] == 2.5
    assert evaluation["passes"] == {
        "overall": True,
        "recovery": True,
        "reroll": True,
        "raceFrontier": True,
        "straightLane": True,
        "target20m": False,
        "lateralDrift": True,
        "finite": True,
    }


def test_dashboard_skips_malformed_newest_evaluation(tmp_path, monkeypatch) -> None:
    evaluation_root = tmp_path / "evaluations"
    evaluation_root.mkdir()
    valid = evaluation_root / "checkpoint-000100-valid.json"
    invalid = evaluation_root / "checkpoint-000200-invalid.json"
    valid.write_text(json.dumps({"checkpoint_iteration": 100}), encoding="utf-8")
    invalid.write_text("{", encoding="utf-8")
    os.utime(valid, (1, 1))
    os.utime(invalid, (2, 2))
    monkeypatch.setattr(serve_dashboard, "ROLL_SPRINT_EVALUATION_ROOT", evaluation_root)

    evaluation = serve_dashboard._latest_roll_sprint_evaluation()

    assert evaluation["available"] is True
    assert evaluation["file"] == valid.name
    assert evaluation["checkpointIteration"] == 100

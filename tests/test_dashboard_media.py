import json
import os
from pathlib import Path

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
    backroll_dir = media_root / "training" / "backroll-sprint-samples"
    grounded_backroll_dir = media_root / "training" / "backroll-skill"
    stair_dir = media_root / "training" / "stair-policy-samples"
    roll_dir.mkdir(parents=True)
    backroll_dir.mkdir(parents=True)
    grounded_backroll_dir.mkdir(parents=True)
    stair_dir.mkdir(parents=True)
    (roll_dir / "roll-0001.mp4").write_bytes(b"roll")
    (backroll_dir / "backroll-0001.mp4").write_bytes(b"backroll")
    (grounded_backroll_dir / "proof-0001.mp4").write_bytes(b"grounded")
    (stair_dir / "stair-0001.mp4").write_bytes(b"stair")
    monkeypatch.setattr(serve_dashboard, "FEATURED_MEDIA_FILE", tmp_path / "missing.json")
    monkeypatch.setattr(serve_dashboard, "MEDIA_ROOTS", {"artifacts": media_root})

    state = serve_dashboard.dashboard_state(include_metrics=False)
    by_name = {item["name"]: item["collection"] for item in state["media"]}

    assert by_name == {
        "roll-0001.mp4": "roll-sprint",
        "backroll-0001.mp4": "backroll-sprint",
        "proof-0001.mp4": "backroll",
        "stair-0001.mp4": "stairs",
    }
    assert state["defaultVideoCollection"] == "backroll"
    counts = {item["id"]: item["videoCount"] for item in state["videoCollections"]}
    assert counts == {
        "backroll": 1,
        "roll-sprint": 1,
        "backroll-sprint": 1,
        "stairs": 1,
    }


def test_dashboard_pins_exact_retained_champion_above_latest_videos(
    tmp_path, monkeypatch
) -> None:
    artifacts = tmp_path / "artifacts"
    champion_root = artifacts / "training" / "roll-sprint-champion"
    sample_root = artifacts / "training" / "roll-sprint-samples"
    champion_root.mkdir(parents=True)
    sample_root.mkdir(parents=True)
    champion_hash = "a" * 64
    newer_hash = "b" * 64
    retained = champion_root / "model_25.pt"
    retained.write_bytes(b"policy")
    champion_video = champion_root / f"champion-000025-{champion_hash[:12]}.mp4"
    champion_video.write_bytes(b"champion-video")
    newer_video = sample_root / "newer-checkpoint-000026.mp4"
    newer_video.write_bytes(b"newer-video")
    os.utime(champion_video, (1, 1))
    os.utime(newer_video, (2, 2))
    manifest = champion_root / "champion.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "evaluation_schema_version": 8,
                "source_checkpoint": str(
                    tmp_path
                    / "logs"
                    / "2026-08-31_13-48-51_a72_conservative_champion_4096x4000"
                    / "model_25.pt"
                ),
                "retained_checkpoint": str(retained),
                "checkpoint_sha256": champion_hash,
                "target_distance_reach_count": 3,
                "mean_credited_forward_frontier_m": 8.95,
                "mean_time_to_valid_10m_s": 36.59,
            }
        ),
        encoding="utf-8",
    )
    (sample_root / "newer-video-state.json").write_text(
        json.dumps(
            {
                "last_checkpoint": {"sha256": newer_hash},
                "last_video": str(newer_video),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(serve_dashboard, "MEDIA_ROOTS", {"artifacts": artifacts})
    monkeypatch.setattr(
        serve_dashboard, "FEATURED_MEDIA_FILE", tmp_path / "missing.json"
    )
    monkeypatch.setattr(serve_dashboard, "ROLL_SPRINT_SAMPLE_ROOT", sample_root)
    monkeypatch.setattr(serve_dashboard, "ROLL_SPRINT_CHAMPION_ROOT", champion_root)
    monkeypatch.setattr(serve_dashboard, "ROLL_SPRINT_CHAMPION_MANIFEST", manifest)

    state = serve_dashboard.dashboard_state(include_metrics=False)
    champion = state["rollSprintChampion"]

    assert champion["available"] is True
    assert champion["evaluationSchemaVersion"] == 8
    assert champion["version"] == (
        "2026-08-31_13-48-51_a72_conservative_champion_4096x4000"
    )
    assert champion["retainedCheckpoint"] == "model_25.pt"
    assert champion["checkpointIteration"] == 25
    assert champion["checkpointSha256"] == champion_hash
    assert champion["checkpointHash"] == champion_hash[:12]
    assert champion["video"]["name"] == champion_video.name
    assert champion["video"]["url"] == (
        f"/media/artifacts/training/roll-sprint-champion/{champion_video.name}"
    )

    featured_hash = "c" * 64
    featured_video = champion_root / "champion-featured-000300-cccccccccccc.mp4"
    featured_video.write_bytes(b"featured-video")
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_payload.update(
        {
            "featured_video": str(featured_video),
            "featured_video_checkpoint_iteration": 300,
            "featured_video_checkpoint_sha256": featured_hash,
        }
    )
    manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")

    featured_state = serve_dashboard.dashboard_state(include_metrics=False)
    featured_champion = featured_state["rollSprintChampion"]
    assert featured_champion["retainedCheckpoint"] == "model_25.pt"
    assert featured_champion["checkpointSha256"] == champion_hash
    assert featured_champion["videoIsFeatured"] is True
    assert featured_champion["featuredVideoCheckpointIteration"] == 300
    assert featured_champion["featuredVideoCheckpointSha256"] == featured_hash
    assert featured_champion["featuredVideoCheckpointHash"] == featured_hash[:12]
    assert featured_champion["video"]["name"] == featured_video.name
    assert featured_champion["video"]["url"] == (
        f"/media/artifacts/training/roll-sprint-champion/{featured_video.name}"
    )

    dashboard_root = Path(__file__).resolve().parents[1] / "dashboard"
    html = (dashboard_root / "index.html").read_text(encoding="utf-8")
    assert html.index('id="roll-champion"') < html.index('id="media-grid"')
    app = (dashboard_root / "app.js").read_text(encoding="utf-8")
    assert "champion.retainedCheckpoint" in app
    assert "champion.checkpointHash" in app
    assert "champion.videoIsFeatured" in app


def test_dashboard_shows_backroll_sampler_media_in_fifteen_video_pages() -> None:
    dashboard_root = Path(__file__).resolve().parents[1] / "dashboard"
    config = json.loads(
        (dashboard_root / "featured_media.json").read_text(encoding="utf-8")
    )
    assert (
        "artifacts/training/backroll-skill-samples/"
        in config["featuredVideoPrefixes"]
    )

    html = (dashboard_root / "index.html").read_text(encoding="utf-8")
    app = (dashboard_root / "app.js").read_text(encoding="utf-8")
    styles = (dashboard_root / "styles.css").read_text(encoding="utf-8")
    assert 'id="media-pagination"' in html
    assert 'id="media-page-previous"' in html
    assert 'id="media-page-next"' in html
    assert "const MEDIA_PAGE_SIZE = 15;" in app
    assert "media.slice(pageStart, pageStart + MEDIA_PAGE_SIZE)" in app
    assert ".media-pagination[hidden]" in styles


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
                "four_robot_batch_road_corridor_pass": True,
                "four_robot_batch_target_10m_pass": False,
                "standing_on_road_target_reach_rate": 0.75,
                "road_exit_env_count": 0,
                "maximum_road_boundary_overshoot_m": 0.0,
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
    assert evaluation["roadReturnCount"] == 3
    assert evaluation["roadReturnLatencyMeanS"] == 0.8
    assert evaluation["roadExitEnvCount"] == 0
    assert evaluation["maximumRoadOvershootM"] == 0.0
    assert evaluation["standingOnRoadTargetRate"] == 0.75
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
        "sharedRoad": True,
        "target10m": True,
        "finite": True,
    }


def test_dashboard_maps_historical_v4_20m_pass_to_current_10m_gate(tmp_path) -> None:
    report_path = tmp_path / "historical-v4.json"
    report_path.write_text("{}", encoding="utf-8")

    evaluation = serve_dashboard._roll_sprint_evaluation_payload(
        {
            "schema_version": 4,
            "four_robot_batch_target_20m_pass": True,
        },
        report_path,
    )

    assert evaluation["passes"]["target10m"] is True


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

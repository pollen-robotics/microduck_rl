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

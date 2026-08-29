import json

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

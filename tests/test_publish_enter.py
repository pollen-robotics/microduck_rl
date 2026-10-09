"""`uv run publish --enter`: after the upload, the Arena races the new revision and says where to see it.

The Hub is a fake HfApi; the Arena is a local HTTP server that records what it was sent and
answers what the Space answers.
"""

from __future__ import annotations

import json
import socket
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs

import pytest

from tests.test_publish_manifest import _run_dir, fake_mjlab, sprint_challenge  # noqa: F401  (fixtures)

REV = "c0ffee" + "0" * 34
PAGE = "https://huggingface.co/spaces/pollen-robotics/microduck-arena#/r/run42"
ENTERED = {"run_id": "run42", "score": 4.892, "seeds_finished": 5, "seeds_total": 5, "command_vx": 0.55,
           "page_url": PAGE}


class FakeHub:
    """huggingface_hub.HfApi, as publish uses it."""

    private = False
    exists = False
    calls: list = []

    def __init__(self, *args, **kwargs):
        pass

    def repo_exists(self, repo):
        return FakeHub.exists

    def create_repo(self, repo, **kwargs):
        FakeHub.calls.append(("create_repo", repo))
        if not FakeHub.exists:  # as the Hub does: exist_ok keeps an existing repo's visibility
            FakeHub.private, FakeHub.exists = kwargs["private"], True

    def repo_info(self, repo):
        return SimpleNamespace(private=FakeHub.private)

    def list_repo_files(self, repo):
        return []

    def upload_folder(self, repo_id, folder_path, **kwargs):
        FakeHub.calls.append(("upload_folder", repo_id))
        FakeHub.uploaded = {p.name: p.read_bytes() for p in Path(folder_path).iterdir()}
        return SimpleNamespace(oid=REV, commit_url=f"https://huggingface.co/{repo_id}/commit/{REV}")

    def create_tag(self, *args, **kwargs):
        pass


@pytest.fixture
def hub(monkeypatch):
    import huggingface_hub

    FakeHub.private, FakeHub.exists, FakeHub.calls = False, False, []
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeHub)
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    return FakeHub


@contextmanager
def _server():
    """A local HTTP server: records each request, answers `reply` (`hangup`: reads it and closes)."""
    seen = []
    reply = {"status": 200, "type": "application/json", "body": json.dumps(ENTERED), "headers": {},
             "hangup": False}

    class Recorder(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            fields = {k: v[0] for k, v in parse_qs(self.rfile.read(length).decode()).items()}
            seen.append({"path": self.path, "authorization": self.headers.get("Authorization"), "fields": fields})
            if reply["hangup"]:
                return
            self.send_response(reply["status"])
            self.send_header("Content-Type", reply["type"])
            for header, value in reply["headers"].items():
                self.send_header(header, value)
            self.end_headers()
            self.wfile.write(reply["body"].encode())

        do_GET = do_POST

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Recorder)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield SimpleNamespace(url=f"http://127.0.0.1:{server.server_port}", seen=seen, reply=reply)
    server.shutdown()


@pytest.fixture
def arena():
    """The Arena's submit route."""
    with _server() as server:
        yield server


@pytest.fixture
def elsewhere():
    """Another host, where a redirect from the Arena would point."""
    with _server() as server:
        yield server


def _publish(tmp_path, monkeypatch, **fields):
    from mjlab_microduck.publish.cli import PublishConfig, run

    monkeypatch.chdir(tmp_path)
    return run(PublishConfig(repo="alice/microduck-sprint", run=str(_run_dir(tmp_path)), enter=True,
                             **{"private": False, **fields}))


def test_enter_races_the_revision_just_uploaded_with_the_token(tmp_path, monkeypatch, fake_mjlab,
                                                               sprint_challenge, hub, arena, capsys):
    code_url = "https://github.com/alice/microduck-challenges/tree/3f9c2d1ab"
    assert _publish(tmp_path, monkeypatch, arena=arena.url, livery="yellow", speed=0.55, code_url=code_url) == 0
    (sent,) = arena.seen
    assert sent["path"] == "/api/events/sprint-2m/submit"
    assert sent["authorization"] == "Bearer hf_test"
    assert sent["fields"] == {"repo_id": "alice/microduck-sprint", "revision": REV, "name": "sprint",
                              "livery": "yellow", "command_vx": "0.55", "env_url": code_url}
    out = capsys.readouterr().out
    assert f"[publish] entered on sprint-2m: 4.892 s (5/5 seeds) at 0.55 m/s → {PAGE}" in out
    assert "hf_test" not in out


def test_a_sweep_sends_no_speed_and_no_code_link(tmp_path, monkeypatch, fake_mjlab, sprint_challenge, hub, arena):
    assert _publish(tmp_path, monkeypatch, arena=arena.url) == 0
    fields = arena.seen[0]["fields"]
    assert "command_vx" not in fields and "env_url" not in fields
    assert fields["livery"] == "classic"


def test_a_timeline_is_in_the_revision_entered(tmp_path, monkeypatch, fake_mjlab, sprint_challenge, hub, arena):
    """--timeline with --enter: the Arena is sent the revision whose upload carries timeline.json."""
    timeline = tmp_path / "moves.json"
    timeline.write_text('{"timeline_version": 1, "duration_s": 20.0, "keyframes": [{"t": 0.0}]}\n')
    assert _publish(tmp_path, monkeypatch, arena=arena.url, timeline=str(timeline)) == 0
    assert hub.uploaded["timeline.json"] == timeline.read_bytes()
    assert arena.seen[0]["fields"]["revision"] == REV


def test_an_entry_already_on_the_board_is_said(tmp_path, monkeypatch, fake_mjlab, sprint_challenge, hub, arena,
                                               capsys):
    arena.reply["body"] = json.dumps(ENTERED | {"already_entered": True})
    assert _publish(tmp_path, monkeypatch, arena=arena.url) == 0
    assert "[publish] already entered on sprint-2m: 4.892 s" in capsys.readouterr().out


def test_a_refusal_is_said_after_the_upload_and_exits_3(tmp_path, monkeypatch, fake_mjlab, sprint_challenge, hub,
                                                         arena, capsys):
    arena.reply.update(status=403, body=json.dumps({"detail": "bob/x is not yours to enter"}))
    assert _publish(tmp_path, monkeypatch, arena=arena.url) == 3
    err = capsys.readouterr().err
    assert (f"[publish] uploaded https://huggingface.co/alice/microduck-sprint/commit/{REV}, "
            "but the Arena did not enter it: 403: bob/x is not yours to enter") in err
    assert ("upload_folder", "alice/microduck-sprint") in hub.calls
    assert "hf_test" not in err


def test_a_gateway_error_page_is_said_not_crashed_on(tmp_path, monkeypatch, fake_mjlab, sprint_challenge, hub,
                                                     arena, capsys):
    arena.reply.update(status=502, type="text/html", body="<html><h1>Bad Gateway</h1></html>")
    assert _publish(tmp_path, monkeypatch, arena=arena.url) == 3
    assert "did not enter it: 502: Bad Gateway" in capsys.readouterr().err


def _not_entered(capsys) -> str:
    """What publish said, once it has checked the upload is named and the token is not."""
    out, err = capsys.readouterr()
    assert (f"[publish] uploaded https://huggingface.co/alice/microduck-sprint/commit/{REV}, "
            "but the Arena did not enter it: ") in err
    assert "hf_test" not in out + err
    return err


def test_an_arena_that_hangs_up_without_answering_is_said(tmp_path, monkeypatch, fake_mjlab, sprint_challenge,
                                                          hub, arena, capsys):
    arena.reply["hangup"] = True
    assert _publish(tmp_path, monkeypatch, arena=arena.url) == 3
    assert f"did not enter it: {arena.url} did not answer: " in _not_entered(capsys)


def test_an_answer_cut_short_is_said(tmp_path, monkeypatch, fake_mjlab, sprint_challenge, hub, arena, capsys):
    arena.reply["headers"] = {"Content-Length": "1000"}
    assert _publish(tmp_path, monkeypatch, arena=arena.url) == 3
    assert f"did not enter it: {arena.url} did not answer: " in _not_entered(capsys)


def test_an_html_answer_is_not_an_entry(tmp_path, monkeypatch, fake_mjlab, sprint_challenge, hub, arena, capsys):
    arena.reply.update(type="text/html", body="<html><h1>Welcome</h1></html>")
    assert _publish(tmp_path, monkeypatch, arena=arena.url) == 3
    assert "did not enter it: the Arena's answer was not an entry: " in _not_entered(capsys)


def test_an_answer_without_a_run_id_is_not_an_entry(tmp_path, monkeypatch, fake_mjlab, sprint_challenge, hub,
                                                    arena, capsys):
    arena.reply["body"] = json.dumps({k: v for k, v in ENTERED.items() if k != "run_id"})
    assert _publish(tmp_path, monkeypatch, arena=arena.url) == 3
    assert "did not enter it: the Arena's answer was not an entry: " in _not_entered(capsys)


def test_a_redirect_is_not_followed_with_the_token(tmp_path, monkeypatch, fake_mjlab, sprint_challenge, hub, arena,
                                                   elsewhere, capsys):
    arena.reply.update(status=302, body="", headers={"Location": f"{elsewhere.url}/api/events/sprint-2m/submit"})
    assert _publish(tmp_path, monkeypatch, arena=arena.url) == 3
    assert "did not enter it: 302" in _not_entered(capsys)
    assert elsewhere.seen == []


def test_an_arena_that_does_not_answer_is_said(tmp_path, monkeypatch, fake_mjlab, sprint_challenge, hub, capsys):
    with socket.socket() as nobody:  # bound, then closed: nothing listens there
        nobody.bind(("127.0.0.1", 0))
        port = nobody.getsockname()[1]
    assert _publish(tmp_path, monkeypatch, arena=f"http://127.0.0.1:{port}") == 3
    assert "did not answer" in capsys.readouterr().err


@pytest.mark.parametrize("url", ["http://arena.example", "ftp://127.0.0.1"])
def test_the_token_travels_only_over_https_or_to_this_machine(url, tmp_path, monkeypatch, fake_mjlab,
                                                             sprint_challenge, hub, capsys):
    with pytest.raises(SystemExit):
        _publish(tmp_path, monkeypatch, arena=url)
    assert "--arena" in capsys.readouterr().err
    assert fake_mjlab == [] and hub.calls == [], "refused before the export and the upload"


@pytest.mark.parametrize("private", [True, False])
def test_an_existing_private_repo_is_refused_before_the_upload(private, tmp_path, monkeypatch, fake_mjlab,
                                                               sprint_challenge, hub, arena, capsys):
    hub.exists, hub.private = True, True  # --no-private or not: an existing repo keeps its visibility
    with pytest.raises(SystemExit):
        _publish(tmp_path, monkeypatch, arena=arena.url, private=private)
    assert "public" in capsys.readouterr().err
    assert hub.calls == [("create_repo", "alice/microduck-sprint")] and arena.seen == []


def test_a_new_repo_made_private_is_refused_before_it_is_created(tmp_path, monkeypatch, fake_mjlab,
                                                                 sprint_challenge, hub, arena, capsys):
    """Created private, it would be refused, and kept private on every retry: a refusal loop."""
    with pytest.raises(SystemExit):
        _publish(tmp_path, monkeypatch, arena=arena.url, private=True)
    assert ("--enter needs a public repo: pass --no-private (a new repo is created private by default)"
            in capsys.readouterr().err)
    assert hub.calls == [] and arena.seen == []


def test_an_existing_public_repo_enters_without_no_private(tmp_path, monkeypatch, fake_mjlab, sprint_challenge,
                                                           hub, arena):
    hub.exists = True
    assert _publish(tmp_path, monkeypatch, arena=arena.url, private=True) == 0
    assert len(arena.seen) == 1


def test_enter_needs_a_token(tmp_path, monkeypatch, fake_mjlab, sprint_challenge, hub, arena, capsys):
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "get_token", lambda: None)
    with pytest.raises(SystemExit):
        _publish(tmp_path, monkeypatch, arena=arena.url)
    assert "token" in capsys.readouterr().err
    assert hub.calls == [] and arena.seen == []


def test_a_policy_not_a_challenges_needs_its_event_and_records_it(tmp_path, monkeypatch, fake_mjlab, hub, arena,
                                                                  capsys):
    from mjlab_microduck.publish.cli import PublishConfig, run

    monkeypatch.chdir(tmp_path)
    run_dir = _run_dir(tmp_path, task="Mjlab-Velocity-Flat-MicroDuck")
    common = dict(repo="alice/microduck-glide", run=str(run_dir), kind="perpetual", slot="walk", private=False,
                  enter=True, arena=arena.url)
    with pytest.raises(SystemExit):
        run(PublishConfig(**common))
    assert "--event" in capsys.readouterr().err
    assert run(PublishConfig(**common, event="sprint-2m")) == 0
    assert arena.seen[0]["path"] == "/api/events/sprint-2m/submit"
    assert run(PublishConfig(**common, event="sprint-2m", dry_run=True)) == 0
    manifest = json.loads((tmp_path / "publish-glide" / "manifest.json").read_text())
    assert manifest["arena"] == {"event": "sprint-2m"}


def test_event_is_for_a_policy_that_is_not_a_challenges(tmp_path, monkeypatch, fake_mjlab, sprint_challenge, hub,
                                                        arena, capsys):
    with pytest.raises(SystemExit):
        _publish(tmp_path, monkeypatch, arena=arena.url, event="hill-climb-1m5")
    assert "--event" in capsys.readouterr().err
    assert hub.calls == []


def test_without_enter_nothing_goes_to_the_arena(tmp_path, monkeypatch, fake_mjlab, sprint_challenge, hub, arena):
    from mjlab_microduck.publish.cli import PublishConfig, run

    monkeypatch.chdir(tmp_path)
    assert run(PublishConfig(repo="alice/microduck-sprint", run=str(_run_dir(tmp_path)), private=False,
                             arena=arena.url)) == 0
    assert arena.seen == []


def test_a_stage_entry_is_said_as_a_performance(tmp_path, monkeypatch, fake_mjlab, sprint_challenge, hub, arena, capsys):
    arena.reply["body"] = json.dumps(ENTERED | {"score": 20.0, "seeds_finished": 1, "seeds_total": 1,
                                                "command_vx": 0.0, "course": "stage"})
    assert _publish(tmp_path, monkeypatch, arena=arena.url) == 0
    assert f"[publish] entered on sprint-2m: a 20.00 s performance → {PAGE}" in capsys.readouterr().out

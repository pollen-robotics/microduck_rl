"""`uv run publish` writes what the microduck daemon loads — schema 2, checked before upload.

The daemon (`pollen-robotics/microduck`) refuses a policy whose manifest disagrees with its
`duck_ipc_proto` constants, refuses at load a graph that is not 61 -> 14, and turns only a
constant-command `episodic` entry into a skill. These tests pin that this side writes exactly
that, on CPU, without mjlab.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from mjlab_microduck.publish import manifest as m

_ROOT = Path(__file__).resolve().parents[1]

# `RemiFabre/microduck-flamingo-cycle`'s manifest as published — the community convention this
# schema had to stay compatible with, verbatim except for trimmed prose.
FLAMINGO = {
    "schema_version": 2,
    "model_api": 1,
    "name": "flamingo-cycle",
    "kind": "perpetual",
    "obs_len": 61,
    "action_len": 14,
    "action_scale": 1.0,
    "entry_pose": "standing",
    "duration_s": None,
    "description": "Stand on one foot, either side, on command: twist = [flag, side, 0].",
    "command": {
        "twist": ["flag: 0 = two feet, 1 = one foot", "side: +1 right down, -1 left down", "unused"],
        "head": "unused (zeros)",
        "body": "unused (zeros)",
        "idle": [0, 0, 0],
    },
    "robot": {"model": "microduck", "hw_rev": 1, "servos": "xl330", "control_hz": 50},
    "training": {"task_id": "Mjlab-FlamingoCycleHard-Flat-MicroDuck"},
}

# The official set, as uploaded 2026-09-02 (schema 2).
OFFICIAL_SET = {
    "schema_version": 2,
    "model_api": 1,
    "obs_len": 61,
    "action_len": 14,
    "robot": {"model": "microduck", "hw_rev": 1, "servos": "xl330", "control_hz": 50},
    "policies": [
        {"file": "alpha_walking.onnx", "kind": "perpetual"},
        {"file": "alpha_sitstand.onnx", "name": "sitstand", "kind": "scripted",
         "command": {"encoding": "posture_flag", "sit": 1.0, "stand": 0.0, "idle": [0, 0, 0]},
         "ramp_s": 2.0, "unwind_s": 1.0},
        {"file": "alpha_ground_pick.onnx", "name": "ground_pick", "kind": "episodic",
         "duration_s": 2.8, "command": {"encoding": "phase", "period_s": 4.0, "end_phase": 0.7}},
        {"file": "roulade.onnx", "kind": "episodic", "duration_s": 1.0, "chain": True},
    ],
}


def _tiny_policy(path: Path, obs_len: int = m.OBS_LEN, action_len: int = m.ACTION_LEN) -> Path:
    """A one-layer 'policy' with the daemon's shape, so the ONNX checks run without torch."""
    rng = np.random.default_rng(0)
    w = numpy_helper.from_array(rng.normal(0, 0.1, (obs_len, action_len)).astype(np.float32), "W")
    node = helper.make_node("MatMul", ["obs", "W"], ["actions"])
    graph = helper.make_graph(
        [node], "policy",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, obs_len])],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, action_len])],
        initializer=[w],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, str(path))
    return path


# -- the numbers the daemon refuses on -------------------------------------------------------


def test_constants_are_the_daemons():
    """`duck_ipc_proto`: POLICY_OBS_LEN 61, POLICY_ACTION_LEN 14, ROBOT_MODEL microduck. A drift
    here is a refusal on every robot, before the download."""
    assert (m.OBS_LEN, m.ACTION_LEN) == (61, 14)
    assert m.ROBOT["model"] == "microduck"
    assert m.MODEL_API == 1
    assert m.SCHEMA_VERSION == 2


def test_publish_is_a_declared_script():
    scripts = tomllib.loads((_ROOT / "pyproject.toml").read_text())["project"]["scripts"]
    assert scripts["publish"] == "mjlab_microduck.publish.cli:main"


def test_infer_is_a_declared_script():
    """A challenges checkout has no scripts/ of ours; the rehearsal must be reachable as `uv run infer`."""
    pyproject = tomllib.loads((_ROOT / "pyproject.toml").read_text())
    assert pyproject["project"]["scripts"].get("infer") == "mjlab_microduck.infer:main"
    shim = (_ROOT / "scripts" / "infer_policy.py").read_text()
    assert "from mjlab_microduck.infer import main" in shim


# -- both shapes validate ----------------------------------------------------------------------


def test_the_flamingo_manifest_is_schema_2_and_valid():
    m.validate_manifest(FLAMINGO)


def test_the_official_set_validates_per_entry():
    m.validate_manifest(OFFICIAL_SET)
    broken = json.loads(json.dumps(OFFICIAL_SET))
    broken["policies"][3]["duration_s"] = None  # roulade: episodic constant with no length
    with pytest.raises(m.ManifestError, match="duration_s"):
        m.validate_manifest(broken)


@pytest.mark.parametrize(
    "bad, why",
    [
        ({"obs_len": 51}, "obs_len"),
        ({"action_len": 12}, "action_len"),
        ({"model_api": 2}, "model_api"),
        ({"robot": {"model": "reachy"}}, "robot.model"),
        ({"kind": "oneshot"}, "kind"),
        ({"command": {"encoding": "telepathy"}}, "encoding"),
    ],
)
def test_a_present_and_wrong_claim_is_refused(bad, why):
    with pytest.raises(m.ManifestError, match=why):
        m.validate_manifest(bad)


def test_absence_is_not_evidence():
    m.validate_manifest({})
    m.validate_manifest({"name": "something", "unknown_field": 3})


# -- what the builder writes ---------------------------------------------------------------------


def test_an_episodic_manifest_is_a_loadable_skill():
    built = m.build_manifest(
        name="polite-bow", kind="episodic", description="Bows.", duration_s=4.0,
        training={"task_id": "Mjlab-PoliteBow-Flat-MicroDuck", "commit": "abc"},
    )
    m.validate_manifest(built)
    assert built["schema_version"] == 2
    assert (built["obs_len"], built["action_len"], built["model_api"]) == (61, 14, 1)
    assert built["robot"] == {**m.ROBOT, "accessories": []}
    assert built["command"]["encoding"] == "constant"
    assert built["command"]["idle"] == [0.0, 0.0, 0.0]
    assert built["duration_s"] == 4.0 and built["chain"] is False
    assert "unwind_s" not in built
    assert built["training"]["task_id"].startswith("Mjlab-")


def test_a_perpetual_manifest_says_how_to_come_back():
    built = m.build_manifest(
        name="flamingo", kind="perpetual", description="One foot.", unwind_s=1.5,
        idle=(0.0, 1.0, 0.0), command_help={"twist": "[flag, side, 0]"},
    )
    m.validate_manifest(built)
    assert built["duration_s"] is None
    assert built["unwind_s"] == 1.5
    assert built["command"]["idle"] == [0.0, 1.0, 0.0]
    assert built["command"]["twist"] == "[flag, side, 0]"


@pytest.mark.parametrize(
    "kwargs, why",
    [
        (dict(kind="episodic"), "duration_s"),
        (dict(kind="episodic", duration_s=0.0), "duration_s"),
        (dict(kind="episodic", duration_s=1.0, unwind_s=2.0), "unwind_s"),
        (dict(kind="perpetual", unwind_s=0.0), "unwind_s"),
        (dict(kind="perpetual", slot="jetpack"), "slot"),
        (dict(kind="perpetual", unwind_s=1.0, duration_s=3.0), "duration_s"),
        (dict(kind="perpetual", unwind_s=1.0, chain=True), "chain"),
        (dict(kind="scripted", duration_s=1.0), "kind"),
        (dict(kind="episodic", duration_s=1.0, action_scale=5.0), "action_scale"),
    ],
)
def test_the_builder_refuses_what_the_kind_cannot_mean(kwargs, why):
    with pytest.raises(m.ManifestError, match=why):
        m.build_manifest(name="x", description="d", **kwargs)


def test_a_name_is_a_bare_word():
    with pytest.raises(m.ManifestError, match="name"):
        m.build_manifest(name="user/thing", kind="episodic", description="d", duration_s=1.0)


def test_a_gait_is_perpetual_with_nothing_to_unwind():
    """A walking policy is perpetual too, and goes in a slot — no hold, no unwind, no skill."""
    gait = m.build_manifest(name="my-walk", kind="perpetual", description="Walks.", slot="walk")
    m.validate_manifest(gait)
    assert gait["duration_s"] is None and "unwind_s" not in gait and gait["slot"] == "walk"
    assert m.install_commands(gait, "u/microduck-my-walk") == "sudo robotctl policy load walk u/microduck-my-walk"
    no_slot = m.build_manifest(name="g", kind="perpetual", description="d")
    assert "policy load <slot>" in m.install_commands(no_slot, "u/g")


def test_the_readme_tells_the_owner_how_to_run_it():
    ep = m.build_manifest(name="bow", kind="episodic", description="Bows.", duration_s=4.0, chain=True)
    text = m.render_readme(ep, "someone/microduck-bow")
    assert "robotctl policy add bow someone/microduck-bow" in text
    assert "robot do bow" in text and "chains" in text
    pp = m.build_manifest(name="flamingo", kind="perpetual", description="d", unwind_s=1.5)
    assert "--hold <seconds>" in m.render_readme(pp, "someone/microduck-flamingo")


# -- the ONNX gate ------------------------------------------------------------------------------


def test_a_61_to_14_graph_passes_and_smoke_runs(tmp_path):
    path = _tiny_policy(tmp_path / "policy.onnx")
    shape = m.check_onnx(path)
    assert (shape.obs_len, shape.action_len) == (61, 14)
    m.smoke_run_onnx(path)


def test_a_legacy_51d_graph_is_refused_before_upload(tmp_path):
    path = _tiny_policy(tmp_path / "old.onnx", obs_len=51)
    with pytest.raises(m.ManifestError, match="51"):
        m.check_onnx(path)


def test_a_wrong_action_width_is_refused(tmp_path):
    path = _tiny_policy(tmp_path / "wide.onnx", action_len=16)
    with pytest.raises(m.ManifestError, match="16 actions"):
        m.check_onnx(path)


def test_a_constant_network_fails_the_smoke_run(tmp_path):
    """A graph that ignores its input is not a policy — the shape gate alone would pass it."""
    zero = numpy_helper.from_array(np.zeros((m.OBS_LEN, m.ACTION_LEN), np.float32), "W")
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["obs", "W"], ["actions"])], "dead",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, m.OBS_LEN])],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, m.ACTION_LEN])],
        initializer=[zero],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    path = tmp_path / "dead.onnx"
    onnx.save(model, str(path))
    with pytest.raises(m.ManifestError, match="never changes"):
        m.smoke_run_onnx(path)


def test_the_cli_dry_run_writes_a_repo(tmp_path, monkeypatch, capsys):
    """End to end without the Hub or a GPU: an ONNX in, the three repo files out."""
    from mjlab_microduck.publish.cli import PublishConfig, run

    policy = _tiny_policy(tmp_path / "out.onnx")
    monkeypatch.chdir(tmp_path)
    code = run(PublishConfig(
        repo="someone/microduck-bow", kind="episodic", onnx=str(policy),
        duration_s=4.0, description="Bows.", dry_run=True,
    ))
    assert code == 0
    out = tmp_path / "publish-bow"
    assert (out / "policy.onnx").exists()
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["name"] == "bow" and manifest["kind"] == "episodic"
    assert manifest["training"]["source_file"] == "out.onnx"
    assert "commit" in manifest["training"], "git provenance is filled from the checkout"
    assert "robotctl policy add bow someone/microduck-bow" in (out / "README.md").read_text()
    assert f"dry run: wrote {out}/ (policy.onnx, manifest.json, README.md)\n" in capsys.readouterr().out


def _manifest_with_training(training: dict) -> dict:
    return m.build_manifest(
        name="sprint", kind="perpetual", description="Walks fast.", slot="walk", training=training
    )


def test_the_readme_reproduces_a_recorded_run():
    readme = m.render_readme(_manifest_with_training({
        "task_id": "Mjlab-Sprint2m-MicroDuck",
        "repo": "https://github.com/alice/microduck-challenges",
        "commit": "3f9c2d1ab",
        "branch": "main",
        "dirty": False,
        "command": ["train", "Mjlab-Sprint2m-MicroDuck", "--env.scene.num-envs", "4096", "--agent.seed", "7"],
        "seed": 7,
        "base": "mjlab-microduck 0.1.0 @ 8a1b2c3d4",
        "started": "2026-09-25T10:00:00Z",
    }), "alice/microduck-sprint")
    assert "## Reproduce" in readme
    assert "git clone https://github.com/alice/microduck-challenges" in readme
    assert "cd microduck-challenges" in readme
    assert "git checkout 3f9c2d1ab" in readme
    assert "uv run train Mjlab-Sprint2m-MicroDuck --env.scene.num-envs 4096 --agent.seed 7" in readme
    assert "comparable policy, not the same weights" in readme
    assert "- **seed**: `7`" in readme
    assert "- **base**: `mjlab-microduck 0.1.0 @ 8a1b2c3d4`" in readme


def test_no_reproduce_block_without_a_commit():
    """A run trained outside git (an HF Jobs tarball) has a command but nothing to check out."""
    readme = m.render_readme(_manifest_with_training({
        "task_id": "T", "command": ["train", "T"], "seed": 1,
    }), "alice/microduck-sprint")
    assert "## Reproduce" not in readme
    assert "- **seed**: `1`" in readme


def test_a_readme_without_a_command_is_unchanged():
    """The existing publish paths never set `command`; their README must not grow a block."""
    readme = m.render_readme(_manifest_with_training({
        "task_id": "T", "repo": "pollen-robotics/microduck_rl", "commit": "abc", "branch": "develop", "dirty": False,
    }), "alice/microduck-sprint")
    assert "## Reproduce" not in readme


def test_accessories_are_read_from_the_robot_model():
    import mujoco

    bare = mujoco.MjSpec()
    bare.add_mesh().name = "left_shell"
    assert m.accessories_of(bare) == ()
    wheeled = mujoco.MjSpec()
    wheeled.add_mesh().name = "left_shell"
    wheeled.add_mesh().name = "roller_blade"
    assert m.accessories_of(wheeled) == ("rollers",)


def test_accessories_are_read_from_the_real_robot_specs():
    from mjlab_microduck.robot.microduck_constants import get_walk_rollers_spec, get_walk_spec

    assert m.accessories_of(get_walk_spec()) == ()
    assert m.accessories_of(get_walk_rollers_spec()) == ("rollers",)


def test_the_manifest_says_what_the_robot_wears():
    manifest = m.build_manifest(name="glide", kind="perpetual", description="Glides.", slot="walk",
                                accessories=("rollers",), arena={"event": "roller-sprint-2m"})
    assert manifest["robot"]["accessories"] == ["rollers"]
    assert manifest["robot"]["model"] == "microduck", "the daemon refuses any other model name"
    assert manifest["arena"] == {"event": "roller-sprint-2m"}
    m.validate_manifest(manifest)
    plain = m.build_manifest(name="walk", kind="perpetual", description="Walks.", slot="walk")
    assert plain["robot"]["accessories"] == [] and "arena" not in plain
    assert "on rollers" in m.render_readme(manifest, "alice/microduck-glide")
    assert "on rollers" not in m.render_readme(plain, "alice/microduck-walk")


def test_an_unknown_accessory_is_refused():
    manifest = m.build_manifest(name="ski", kind="perpetual", description="Skis.", slot="walk")
    manifest["robot"]["accessories"] = ["skis"]
    with pytest.raises(m.ManifestError, match="skis"):
        m.validate_manifest(manifest)


def test_absence_of_accessories_is_not_evidence():
    m.validate_manifest(FLAMINGO)  # the community manifest predates the field


# -- publish --run ------------------------------------------------------------------------------


def _run_dir(tmp_path: Path, *, dirty: bool = False, iterations=(100, 250), with_checkout: bool = True,
             task: str = "Mjlab-Sprint2m-MicroDuck") -> Path:
    """A finished local run: provenance from `train`, two checkpoints."""
    run = tmp_path / "logs" / "rsl_rl" / "sprint" / "2026-09-25_10-00-00_first"
    run.mkdir(parents=True)
    record = {
        "command": ["train", task, "--agent.seed", "7"],
        "task": task,
        "seed": 7,
        "base": "mjlab-microduck 0.1.0",
        "started": "2026-09-25T10:00:00Z",
    }
    if with_checkout:
        record |= {
            "repo": "https://github.com/alice/microduck-challenges",
            "commit": "3f9c2d1ab", "branch": "main", "dirty": dirty,
        }
    (run / "provenance.json").write_text(json.dumps(record))
    for n in iterations:
        (run / f"model_{n}.pt").write_bytes(b"checkpoint %d" % n)
    return run


@pytest.fixture
def fake_mjlab(monkeypatch, tmp_path):
    """Stand in for the GPU export and the registry: records what was asked, writes a tiny policy,
    says the sprint task's robot wears nothing and the roller task's wears rollers."""
    from mjlab_microduck.publish import cli

    calls = []

    def _export(task, checkpoint, out, device):
        calls.append((task, Path(checkpoint).name))
        return _tiny_policy(out)

    monkeypatch.setattr(cli, "_load_registry", lambda: None)
    monkeypatch.setattr(cli, "_export_checkpoint", _export)
    monkeypatch.setattr(cli, "_accessories_of_task", lambda task: ("rollers",) if "Roller" in task else ())
    return calls


@pytest.fixture
def sprint_challenge(tmp_path, monkeypatch):
    from mjlab_microduck import challenge as ch

    monkeypatch.setattr(ch, "_REGISTRY", {})
    folder = tmp_path / "sprint_2m"
    folder.mkdir()
    (folder / "challenge.toml").write_text(
        'event = "sprint-2m"\ntask = "Mjlab-Sprint2m-MicroDuck"\nkind = "perpetual"\n'
    )
    (folder / "tasks.py").write_text("")
    return ch.register(folder / "tasks.py")


def test_publish_run_needs_only_the_repo_for_a_challenge(tmp_path, monkeypatch, fake_mjlab, sprint_challenge,
                                                        capsys):
    from mjlab_microduck.publish.cli import PublishConfig, run

    run_dir = _run_dir(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert run(PublishConfig(repo="alice/microduck-sprint", run=str(run_dir), dry_run=True)) == 0
    assert fake_mjlab == [("Mjlab-Sprint2m-MicroDuck", "model_250.pt")], "latest checkpoint by default"
    out = tmp_path / "publish-sprint"
    assert (out / "policy.onnx").exists()
    assert (out / "checkpoint.pt").read_bytes() == b"checkpoint 250"
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["kind"] == "perpetual", "from the challenge, not a flag"
    assert manifest["arena"] == {"event": "sprint-2m"}
    assert manifest["robot"]["accessories"] == []
    training = manifest["training"]
    assert training["repo"] == "https://github.com/alice/microduck-challenges"
    assert training["commit"] == "3f9c2d1ab"
    assert training["command"] == ["train", "Mjlab-Sprint2m-MicroDuck", "--agent.seed", "7"]
    assert training["seed"] == 7 and training["base"] == "mjlab-microduck 0.1.0"
    assert training["task_id"] == "Mjlab-Sprint2m-MicroDuck"
    assert training["checkpoint"] == 250 and training["source_file"] == "model_250.pt"
    readme = (out / "README.md").read_text()
    assert "## Reproduce" in readme and "git checkout 3f9c2d1ab" in readme
    assert "(policy.onnx, manifest.json, README.md, checkpoint.pt)" in capsys.readouterr().out


def test_publish_run_of_a_library_task_needs_kind_and_reads_accessories(tmp_path, monkeypatch, fake_mjlab, capsys):
    from mjlab_microduck.publish.cli import PublishConfig, run

    run_dir = _run_dir(tmp_path, task="Mjlab-Velocity-Flat-MicroDuck-Rollers")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        run(PublishConfig(repo="alice/microduck-glide", run=str(run_dir), dry_run=True))
    assert "--kind" in capsys.readouterr().err
    assert fake_mjlab == [], "refused before any export"
    assert run(PublishConfig(repo="alice/microduck-glide", run=str(run_dir), kind="perpetual", slot="walk",
                             dry_run=True)) == 0
    manifest = json.loads((tmp_path / "publish-glide" / "manifest.json").read_text())
    assert manifest["robot"]["accessories"] == ["rollers"] and "arena" not in manifest


def _fork(root: Path) -> str:
    """A git checkout whose origin is alice's fork; returns its HEAD commit."""
    import subprocess

    def git(*args):
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True,
                              check=True).stdout.strip()

    root.mkdir()
    git("init", "-q", "-b", "main")
    git("-c", "user.email=t@example.com", "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", "first")
    git("remote", "add", "origin", "https://github.com/alice/microduck-challenges")
    return git("rev-parse", "--short=9", "HEAD")


@pytest.mark.parametrize("contained", [True, False])
def test_publish_run_names_the_fork_that_holds_the_commit(tmp_path, monkeypatch, fake_mjlab, sprint_challenge,
                                                          contained):
    """Trained on a clone of Pollen's repo, forked afterwards: the recipe points at the fork."""
    from mjlab_microduck.publish.cli import PublishConfig, run

    head = _fork(tmp_path / "fork")
    run_dir = _run_dir(tmp_path)
    record = json.loads((run_dir / "provenance.json").read_text())
    record |= {"repo": "https://github.com/pollen-robotics/microduck-challenges",
               "commit": head if contained else "3f9c2d1ab"}
    (run_dir / "provenance.json").write_text(json.dumps(record))
    monkeypatch.chdir(tmp_path / "fork")
    assert run(PublishConfig(repo="alice/microduck-sprint", run=str(run_dir), dry_run=True)) == 0
    training = json.loads((tmp_path / "fork" / "publish-sprint" / "manifest.json").read_text())["training"]
    owner = "alice" if contained else "pollen-robotics"
    assert training["repo"] == f"https://github.com/{owner}/microduck-challenges"


def test_publish_task_of_a_library_task_needs_kind_before_the_export(tmp_path, monkeypatch, fake_mjlab, capsys):
    from mjlab_microduck.publish import cli

    exports = []
    monkeypatch.setattr(cli, "_resolve_weights", lambda cfg, workdir: exports.append(cfg.task))
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        cli.run(cli.PublishConfig(repo="alice/microduck-glide", task="Mjlab-Velocity-Flat-MicroDuck",
                                  checkpoint_file=str(tmp_path / "model_1.pt"), dry_run=True))
    assert "--kind is required" in capsys.readouterr().err
    assert exports == [], "refused before any export"


def test_publish_run_picks_the_asked_checkpoint(tmp_path, monkeypatch, fake_mjlab, sprint_challenge):
    from mjlab_microduck.publish.cli import PublishConfig, run

    run_dir = _run_dir(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert run(PublishConfig(repo="alice/microduck-sprint", run=str(run_dir), checkpoint=100, dry_run=True)) == 0
    assert fake_mjlab == [("Mjlab-Sprint2m-MicroDuck", "model_100.pt")]
    assert (tmp_path / "publish-sprint" / "checkpoint.pt").read_bytes() == b"checkpoint 100"


def test_a_dirty_run_is_refused_unless_allowed(tmp_path, monkeypatch, fake_mjlab, sprint_challenge, capsys):
    from mjlab_microduck.publish.cli import PublishConfig, run

    run_dir = _run_dir(tmp_path, dirty=True)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        run(PublishConfig(repo="alice/microduck-sprint", run=str(run_dir), dry_run=True))
    assert "uncommitted changes" in capsys.readouterr().err
    assert fake_mjlab == [], "refused before any export"
    assert run(PublishConfig(repo="alice/microduck-sprint", run=str(run_dir), dry_run=True, allow_dirty=True)) == 0
    assert json.loads((tmp_path / "publish-sprint" / "manifest.json").read_text())["training"]["dirty"] is True


def test_missing_checkpoint_is_named(tmp_path, monkeypatch, fake_mjlab, sprint_challenge, capsys):
    from mjlab_microduck.publish.cli import PublishConfig, run

    run_dir = _run_dir(tmp_path)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        run(PublishConfig(repo="alice/microduck-sprint", run=str(run_dir), checkpoint=999, dry_run=True))
    assert "model_999.pt" in capsys.readouterr().err
    assert fake_mjlab == []


def test_a_run_without_checkpoints_is_refused(tmp_path, monkeypatch, fake_mjlab, sprint_challenge, capsys):
    from mjlab_microduck.publish.cli import PublishConfig, run

    run_dir = _run_dir(tmp_path, iterations=())
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        run(PublishConfig(repo="alice/microduck-sprint", run=str(run_dir), dry_run=True))
    assert "no model_<N>.pt" in capsys.readouterr().err


def test_a_run_trained_outside_git_publishes_without_a_recipe(tmp_path, monkeypatch, fake_mjlab, sprint_challenge):
    from mjlab_microduck.publish.cli import PublishConfig, run

    run_dir = _run_dir(tmp_path, with_checkout=False)
    monkeypatch.chdir(tmp_path)
    assert run(PublishConfig(repo="alice/microduck-sprint", run=str(run_dir), dry_run=True)) == 0
    training = json.loads((tmp_path / "publish-sprint" / "manifest.json").read_text())["training"]
    assert "commit" not in training and training["command"][0] == "train"
    assert "## Reproduce" not in (tmp_path / "publish-sprint" / "README.md").read_text()


def test_run_is_a_source_of_its_own(tmp_path, monkeypatch, fake_mjlab, sprint_challenge, capsys):
    from mjlab_microduck.publish.cli import PublishConfig, run

    run_dir = _run_dir(tmp_path)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        run(PublishConfig(repo="alice/microduck-sprint", run=str(run_dir), onnx="x.onnx", dry_run=True))
    assert "--run" in capsys.readouterr().err


def test_an_onnx_publish_takes_accessories_from_the_flag(tmp_path, monkeypatch):
    from mjlab_microduck.publish.cli import PublishConfig, run

    policy = _tiny_policy(tmp_path / "out.onnx")
    monkeypatch.chdir(tmp_path)
    assert run(PublishConfig(repo="someone/microduck-glide", kind="perpetual", slot="walk", onnx=str(policy),
                             accessories=("rollers",), dry_run=True)) == 0
    manifest = json.loads((tmp_path / "publish-glide" / "manifest.json").read_text())
    assert manifest["robot"]["accessories"] == ["rollers"]

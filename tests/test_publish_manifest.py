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
    scripts = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["scripts"]
    assert scripts["publish"] == "mjlab_microduck.publish.cli:main"


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
    assert built["robot"] == m.ROBOT
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


def test_the_readme_puts_the_repo_under_the_robotics_pipeline():
    ep = m.build_manifest(name="bow", kind="episodic", description="Bows.", duration_s=4.0)
    front = m.render_readme(ep, "someone/microduck-bow").split("---")[1]
    assert "pipeline_tag: robotics" in front.splitlines()
    assert "base_model" not in front, "an original policy declares no parent"


def test_a_gait_tags_its_slot_for_the_hub_snippet():
    gait = m.build_manifest(name="my-walk", kind="perpetual", description="d", slot="walk")
    front = m.render_readme(gait, "u/microduck-my-walk").split("---")[1].splitlines()
    assert "- microduck-slot:walk" in front
    no_slot = m.build_manifest(name="bow", kind="episodic", description="d", duration_s=4.0)
    assert "microduck-slot" not in m.render_readme(no_slot, "u/microduck-bow")


def test_a_remix_declares_its_base_model():
    ep = m.build_manifest(name="bow", kind="episodic", description="Bows.", duration_s=4.0)
    text = m.render_readme(ep, "someone/microduck-deep-bow", base_model="pollen/microduck-bow")
    front = text.split("---")[1].splitlines()
    assert "base_model: pollen/microduck-bow" in front
    assert "base_model_relation: finetune" in front
    assert "https://huggingface.co/pollen/microduck-bow" in text


@pytest.mark.parametrize("base", ["microduck-bow", "a/b/c", "", "someone/microduck-bow"])
def test_a_bad_base_model_is_refused(base):
    ep = m.build_manifest(name="bow", kind="episodic", description="Bows.", duration_s=4.0)
    with pytest.raises(m.ManifestError, match="base_model"):
        m.render_readme(ep, "someone/microduck-bow", base_model=base)


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


def test_the_cli_dry_run_writes_a_repo(tmp_path, monkeypatch):
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
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["name"] == "bow" and manifest["kind"] == "episodic"
    assert manifest["training"]["source_file"] == "out.onnx"
    assert "commit" in manifest["training"], "git provenance is filled from the checkout"
    assert "robotctl policy add bow someone/microduck-bow" in (out / "README.md").read_text(encoding="utf-8")


def test_the_cli_ships_a_video_as_replay_mp4_and_nowhere_else(tmp_path, monkeypatch):
    """The Hub's replay widget finds `replay.mp4` by name; the manifest and README stay silent."""
    from mjlab_microduck.publish.cli import PublishConfig, run

    policy = _tiny_policy(tmp_path / "out.onnx")
    clip = tmp_path / "bow_take3.mp4"
    clip.write_bytes(b"not really an mp4")
    monkeypatch.chdir(tmp_path)
    code = run(PublishConfig(
        repo="someone/microduck-bow", kind="episodic", onnx=str(policy),
        duration_s=4.0, description="Bows.", video=str(clip), dry_run=True,
    ))
    assert code == 0
    out = tmp_path / "publish-bow"
    assert (out / m.REPLAY_FILE).read_bytes() == clip.read_bytes()
    for text in ((out / "manifest.json").read_text(encoding="utf-8"), (out / "README.md").read_text(encoding="utf-8")):
        assert "mp4" not in text and "bow_take3" not in text


def test_the_cli_refuses_a_missing_video(tmp_path):
    from mjlab_microduck.publish.cli import PublishConfig, run

    policy = _tiny_policy(tmp_path / "out.onnx")
    with pytest.raises(SystemExit):
        run(PublishConfig(
            repo="someone/microduck-bow", kind="episodic", onnx=str(policy),
            duration_s=4.0, video=str(tmp_path / "nope.mp4"), dry_run=True,
        ))


# ---------------------------------------------------------------------------------------------
# Schema 3: free-sentence (VLA) policies — proprio `obs` + `text_emb`, plus encoder.onnx.


def _tiny_vla_policy(
    path: Path,
    proprio_len: int = m.VLA_OBS_LEN,
    text_dim: int = 8,
    action_len: int = m.ACTION_LEN,
    text_name: str = "text_emb",
) -> Path:
    """A two-input policy with the VLA shape, hand-built (no torch)."""
    rng = np.random.default_rng(0)
    w_obs = numpy_helper.from_array(
        rng.normal(0, 0.1, (proprio_len, action_len)).astype(np.float32), "W_obs"
    )
    w_text = numpy_helper.from_array(
        rng.normal(0, 0.1, (text_dim, action_len)).astype(np.float32), "W_text"
    )
    nodes = [
        helper.make_node("MatMul", ["obs", "W_obs"], ["a"]),
        helper.make_node("MatMul", [text_name, "W_text"], ["b"]),
        helper.make_node("Add", ["a", "b"], ["actions"]),
    ]
    graph = helper.make_graph(
        nodes,
        "vla",
        [
            helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, proprio_len]),
            helper.make_tensor_value_info(text_name, TensorProto.FLOAT, [1, text_dim]),
        ],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, action_len])],
        initializer=[w_obs, w_text],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, str(path))
    return path


def _tiny_encoder(path: Path, dim: int = 8, input_names: tuple[str, str] = ("input_ids", "attention_mask")) -> Path:
    """A two-input encoder with a dynamic sequence axis, hand-built."""
    w = numpy_helper.from_array(np.ones((4, dim), np.float32), "W")
    nodes = [
        helper.make_node("Cast", [input_names[0]], ["ids_f"], to=TensorProto.FLOAT),
        helper.make_node("MatMul", ["ids_f", "W"], ["text_emb"]),
    ]
    graph = helper.make_graph(
        nodes,
        "encoder",
        [
            helper.make_tensor_value_info(input_names[0], TensorProto.INT64, [1, 4]),
            helper.make_tensor_value_info(input_names[1], TensorProto.INT64, [1, 4]),
        ],
        [helper.make_tensor_value_info("text_emb", TensorProto.FLOAT, [1, dim])],
        initializer=[w],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, str(path))
    return path


def test_a_two_input_vla_graph_is_recognized_and_smoke_runs(tmp_path):
    path = _tiny_vla_policy(tmp_path / "policy.onnx")
    shape = m.check_onnx(path)
    assert shape.is_vla
    assert (shape.obs_len, shape.text_dim, shape.action_len) == (48, 8, 14)
    assert shape.text_input_name == "text_emb"
    m.smoke_run_onnx(path)


def test_a_vla_graph_with_the_61d_obs_len_is_refused(tmp_path):
    path = _tiny_vla_policy(tmp_path / "bad.onnx", proprio_len=m.OBS_LEN)
    with pytest.raises(m.ManifestError, match="48"):
        m.check_onnx(path)


def test_a_vla_graph_with_a_misnamed_text_input_is_refused(tmp_path):
    path = _tiny_vla_policy(tmp_path / "bad.onnx", text_name="text")
    with pytest.raises(m.ManifestError, match="text_emb"):
        m.check_onnx(path)


def test_an_encoder_is_checked_against_the_policy_dim(tmp_path):
    path = _tiny_encoder(tmp_path / "encoder.onnx", dim=8)
    shape = m.check_encoder_onnx(path, expected_dim=8)
    assert shape.embedding_dim == 8
    assert shape.input_names == ("input_ids", "attention_mask")
    with pytest.raises(m.ManifestError, match="expects 16"):
        m.check_encoder_onnx(path, expected_dim=16)


def test_an_encoder_with_wrong_input_names_is_refused(tmp_path):
    path = _tiny_encoder(tmp_path / "encoder.onnx", input_names=("tokens", "mask"))
    with pytest.raises(m.ManifestError, match="input_ids"):
        m.check_encoder_onnx(path)


VLA = {
    "embedding_dim": 8,
    "encoder": {
        "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "sha256": "deadbeef",
        "max_length": 64,
    },
}


def test_build_manifest_vla_is_schema_3_with_the_encoder_pair():
    built = m.build_manifest(
        name="vla-walk", kind="perpetual", description="Walks on free text.",
        slot="walk", vla=VLA,
    )
    m.validate_manifest(built)
    assert built["schema_version"] == 3
    assert built["obs_len"] == 48
    assert built["command"]["encoding"] == "text"
    assert built["text"]["input"] == "text_emb"
    assert built["text"]["embedding_dim"] == 8
    assert built["text"]["encoder"]["file"] == "encoder.onnx"
    assert built["artifacts"] == {"policy": "policy.onnx", "encoder": "encoder.onnx"}


def test_schema_3_refuses_the_standard_61d_obs_len():
    built = m.build_manifest(
        name="vla-walk", kind="perpetual", description="d", slot="walk", vla=VLA
    )
    built["obs_len"] = m.OBS_LEN
    with pytest.raises(m.ManifestError, match="VLA"):
        m.validate_manifest(built)


def test_schema_3_requires_a_resolvable_encoder():
    built = m.build_manifest(
        name="vla-walk", kind="perpetual", description="d", slot="walk", vla=VLA
    )
    built["text"]["encoder"] = {"file": "encoder.onnx"}  # no model
    with pytest.raises(m.ManifestError, match="encoder.model"):
        m.validate_manifest(built)


def test_schema_3_requires_a_positive_embedding_dim():
    built = m.build_manifest(
        name="vla-walk", kind="perpetual", description="d", slot="walk", vla=VLA
    )
    built["text"]["embedding_dim"] = 0
    with pytest.raises(m.ManifestError, match="embedding_dim"):
        m.validate_manifest(built)


def test_the_vla_readme_names_the_encoder_contract():
    built = m.build_manifest(
        name="vla-walk", kind="perpetual", description="Walks on free text.",
        slot="walk", vla=VLA,
    )
    readme = m.render_readme(built, "someone/microduck-vla-walk")
    assert "schema-3 VLA" in readme
    assert "text_emb" in readme
    assert "encoder.onnx" in readme


def test_the_cli_dry_run_ships_policy_and_encoder_for_a_vla_policy(tmp_path, monkeypatch):
    from mjlab_microduck.publish.cli import PublishConfig, run

    policy = _tiny_vla_policy(tmp_path / "out.onnx")
    encoder = _tiny_encoder(tmp_path / "encoder.onnx", dim=8)
    monkeypatch.chdir(tmp_path)
    code = run(PublishConfig(
        repo="someone/microduck-vla-walk",
        kind="perpetual",
        onnx=str(policy),
        encoder_onnx=str(encoder),
        encoder_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        slot="walk",
        description="Walks on free text.",
        dry_run=True,
    ))
    assert code == 0
    out = tmp_path / "publish-vla-walk"
    assert (out / "policy.onnx").exists()
    assert (out / "encoder.onnx").exists()
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 3
    assert manifest["text"]["embedding_dim"] == 8
    assert manifest["text"]["encoder"]["sha256"] == m.sha256_file(encoder)


def test_the_cli_refuses_a_vla_policy_without_its_encoder(tmp_path, monkeypatch):
    from mjlab_microduck.publish.cli import PublishConfig, run

    policy = _tiny_vla_policy(tmp_path / "out.onnx")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as exc:
        run(PublishConfig(
            repo="someone/microduck-vla-walk",
            kind="perpetual",
            onnx=str(policy),
            slot="walk",
            description="d",
            dry_run=True,
        ))
    assert exc.value.code == 2

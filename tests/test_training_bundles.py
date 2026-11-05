"""Training bundles: hf:// checkpoint specs, the train hook flag, bundle provenance helpers. Offline."""
import os
import sys

import onnx
import pytest
import torch
from onnx import helper, numpy_helper

from mjlab_microduck import checkpoints, train_hook
from mjlab_microduck.publish import manifest as m
from mjlab_microduck.publish import training as tr
from mjlab_microduck.tasks import distill


def test_parse_hf_specs():
    r = checkpoints.parse_hf("hf://pollen-robotics/microduck-training@velstand-v8/bundles/velstand/0045i2si_2749")
    assert (r.repo, r.revision, r.path) == ("pollen-robotics/microduck-training", "velstand-v8", "bundles/velstand/0045i2si_2749")
    assert str(r) == "hf://pollen-robotics/microduck-training@velstand-v8/bundles/velstand/0045i2si_2749"
    assert checkpoints.parse_hf("hf://a/b").path == ""
    assert checkpoints.parse_hf("logs/model_1.pt") is None
    with pytest.raises(ValueError):
        checkpoints.parse_hf("hf://only-owner")


def test_resolve_local_and_missing(tmp_path):
    p = tmp_path / "model_3.pt"
    p.write_bytes(b"x")
    assert checkpoints.resolve(str(p)) == p
    with pytest.raises(FileNotFoundError):
        checkpoints.resolve(str(tmp_path / "nope.pt"))


def test_init_flag_is_consumed_into_the_env(monkeypatch):
    monkeypatch.delenv(checkpoints.INIT_CHECKPOINT_ENV, raising=False)
    out = checkpoints.consume_init_flag(["Task", "--init-checkpoint", "hf://a/b/bundles/x/y_1", "--agent.max-iterations", "5"])
    assert out == ["Task", "--agent.max-iterations", "5"]
    assert os.environ[checkpoints.INIT_CHECKPOINT_ENV] == "hf://a/b/bundles/x/y_1"
    assert checkpoints.consume_init_flag(["Task", "--init-checkpoint=m.pt"]) == ["Task"]
    assert os.environ[checkpoints.INIT_CHECKPOINT_ENV] == "m.pt"
    with pytest.raises(SystemExit):  # two sources of truth for the starting weights
        checkpoints.consume_init_flag(["Task", "--init-checkpoint", "m.pt", "--agent.resume", "True"])


def test_hook_leaves_hf_jobs_submissions_alone(monkeypatch):
    """--hf-jobs exits before maybe_init_checkpoint; and a non-train program never consumes the flag."""
    monkeypatch.delenv(checkpoints.INIT_CHECKPOINT_ENV, raising=False)
    monkeypatch.setattr(sys, "argv", ["play", "Task", "--init-checkpoint", "x.pt"])
    train_hook.maybe_init_checkpoint()
    assert sys.argv[-1] == "x.pt" and checkpoints.INIT_CHECKPOINT_ENV not in os.environ


def test_parse_train_args_finds_the_parent():
    a = tr.parse_train_args(["Mjlab-VelStand-Rough-Backlash-MicroDuck", "--agent.resume", "True",
                             "--wandb-run-path", "pollen-robotics/mjlab_microduck/qz5g3bwi",
                             "--wandb-checkpoint-name", "model_1750.pt"])
    assert a["task_id"] == "Mjlab-VelStand-Rough-Backlash-MicroDuck"
    assert a["loaded_from"] == {"wandb_run": "pollen-robotics/mjlab_microduck/qz5g3bwi", "checkpoint": 1750}
    assert tr.parse_train_args(["T", "--agent.max-iterations=10"])["loaded_from"] is None
    assert tr.parse_train_args(["T", "--init-checkpoint", "hf://a/b/c"])["loaded_from"] == {"spec": "hf://a/b/c"}


def test_find_teacher_refs_in_the_velstand_bc_cfg():
    refs = tr.find_teacher_refs({"algorithm": {"bc_cfg": distill.default_bc_cfg()}})
    assert refs["algorithm.bc_cfg.expert"] == {"wandb_run": "pollen-robotics/mjlab_microduck/69u48n8l", "checkpoint": 9750}
    assert refs["algorithm.bc_cfg.anchor"] == {"wandb_run": "pollen-robotics/mjlab_microduck/441tzs6d", "checkpoint": 3750}


def test_distill_prefers_a_local_path_then_hf(tmp_path, monkeypatch):
    p = tmp_path / "model_1.pt"
    p.write_bytes(b"x")
    assert distill._resolve_checkpoint({"checkpoint_path": str(p), "hf_checkpoint": "hf://a/b/c"}) == p
    assert distill._resolve_checkpoint({"hf_checkpoint": str(p), "wandb_run_path": "e/p/r"}) == p
    with pytest.raises(FileNotFoundError):  # no wandb fallback configured: the hf failure surfaces
        distill._resolve_checkpoint({"hf_checkpoint": str(tmp_path / "nope.pt")})


def _ckpt_and_onnx(tmp_path, tamper=False):
    g = torch.Generator().manual_seed(0)
    sd = {"actor_state_dict": {"obs_normalizer._mean": torch.zeros(1, 4), "mlp.0.weight": torch.randn(8, 4, generator=g),
                               "mlp.0.bias": torch.randn(8, generator=g), "mlp.2.weight": torch.randn(3, 8, generator=g)}}
    ck = tmp_path / "model_5.pt"
    torch.save(sd, ck)
    inits = [numpy_helper.from_array(v.numpy().copy(), name=f"t{i}") for i, v in enumerate(sd["actor_state_dict"].values())]
    if tamper:
        inits[-1] = numpy_helper.from_array(sd["actor_state_dict"]["mlp.2.weight"].numpy() + 1e-6, name="t3")
    graph = helper.make_graph([], "g", [], [], initializer=inits)
    ox = tmp_path / "p.onnx"
    onnx.save(helper.make_model(graph), str(ox))
    return ck, ox


def test_onnx_weight_verification(tmp_path):
    ck, ox = _ckpt_and_onnx(tmp_path)
    assert tr.onnx_matches_checkpoint(ox, ck)
    ck, ox = _ckpt_and_onnx(tmp_path, tamper=True)
    assert not tr.onnx_matches_checkpoint(ox, ck)


def test_cards_carry_the_commands():
    meta = {"id": "velstand/abc_10", "task_id": "Mjlab-VelStand-Rough-Backlash-MicroDuck", "experiment_name": "velstand",
            "checkpoint_file": "model_10.pt", "code": {"commit": "deadbeef", "on_remote": True, "uncommitted_diff": None},
            "source": {}, "loaded_from": None, "teachers": {}, "onnx": [], "command": ["uv", "run", "train", "X"]}
    readme = tr.render_bundle_readme(meta, "o/microduck-training")
    assert "--init-checkpoint hf://o/microduck-training/bundles/velstand/abc_10" in readme
    assert "MICRODUCK_WARM_START=1" in readme and "git checkout deadbeef" in readme
    assert "bundles/velstand/abc_10" in tr.render_index([meta], "o/microduck-training")


def test_policy_readme_points_to_the_bundle():
    man = m.build_manifest(name="x", kind="episodic", description="d", duration_s=1.0,
                           training={"task_id": "T", "repo": "pollen-robotics/microduck_rl", "bundle": "hf://o/microduck-training/bundles/e/r_1"})
    m.validate_manifest(man)
    assert "--init-checkpoint hf://o/microduck-training/bundles/e/r_1" in m.render_readme(man, "o/microduck-x", None)


def test_share_training_is_opt_in():
    from mjlab_microduck.publish.cli import PublishConfig

    assert PublishConfig(repo="o/r", kind="episodic").share_training is False
    assert tr.ShareConfig().public is False  # bundle repos are created private

"""Tests for ``MicroduckVLAModel`` (Phase 3).

The critical invariants:
- only proprio is normalized; the text embedding is appended raw;
- the ONNX wrapper exposes two inputs (``obs`` + ``text_emb``) and exports;
- a model without a text group degrades to a plain MLP (critic case).

Runs on CPU, no simulator/GPU.
"""

from __future__ import annotations

import numpy as np
import onnxruntime as ort
import pytest
import torch
from tensordict import TensorDict

from mjlab_microduck.vla import MicroduckVLAModel

OBS_DIM = 48
TEXT_DIM = 8
BATCH = 16
N_ACTIONS = 14

DIST_CFG = {
    "class_name": "GaussianDistribution",
    "init_std": 1.0,
    "std_type": "scalar",
}


def _actor_obs(batch: int = BATCH) -> TensorDict:
    return TensorDict(
        {
            "actor": torch.randn(batch, OBS_DIM),
            "text": torch.randn(batch, TEXT_DIM),
            "critic": torch.randn(batch, OBS_DIM + 13),
        },
        batch_size=[batch],
    )


def _actor_groups() -> dict[str, tuple[str, ...]]:
    return {"actor": ("actor", "text"), "critic": ("critic",)}


def _make_actor(obs: TensorDict, output_dim: int = N_ACTIONS) -> MicroduckVLAModel:
    return MicroduckVLAModel(
        obs,
        _actor_groups(),
        "actor",
        output_dim,
        hidden_dims=(32, 16),
        activation="elu",
        obs_normalization=True,
        distribution_cfg=dict(DIST_CFG),
    )


def test_model_splits_proprio_and_text():
    obs = _actor_obs()
    model = _make_actor(obs)

    assert model.text_group == "text"
    assert model._text_dim == TEXT_DIM
    # MLPModel's normalizer/obs_dim see only the proprio group.
    assert model.obs_dim == OBS_DIM
    assert model.obs_groups == ["actor"]
    assert model.mlp[0].in_features == OBS_DIM + TEXT_DIM


def test_latent_normalizes_proprio_and_appends_raw_text():
    obs = _actor_obs()
    model = _make_actor(obs)

    latent = model.get_latent(obs)
    assert latent.shape == (BATCH, OBS_DIM + TEXT_DIM)
    # The text half is byte-for-byte the input embedding (not normalized).
    torch.testing.assert_close(latent[:, OBS_DIM:], obs["text"])
    # And the proprio half is the normalizer's output.
    torch.testing.assert_close(
        latent[:, :OBS_DIM], model.obs_normalizer(obs["actor"])
    )


def test_normalizer_never_sees_text():
    obs = _actor_obs()
    model = _make_actor(obs)
    assert tuple(model.obs_normalizer.mean.shape) == (OBS_DIM,)

    model.update_normalization(obs)
    assert int(model.obs_normalizer.count) == BATCH

    # Text must not move the running statistics: two batches that differ ONLY
    # in text produce identical normalizer state.
    model_a = _make_actor(obs)
    model_b = _make_actor(obs)
    obs_b = obs.clone()
    obs_b["text"] = obs["text"] + 100.0
    model_a.update_normalization(obs)
    model_b.update_normalization(obs_b)
    torch.testing.assert_close(model_a.obs_normalizer.mean, model_b.obs_normalizer.mean)
    torch.testing.assert_close(model_a.obs_normalizer.std, model_b.obs_normalizer.std)


def test_text_changes_the_action():
    obs = _actor_obs()
    model = _make_actor(obs)
    model.eval()

    with torch.no_grad():
        action_a = model(obs)
        obs_b = obs.clone()
        obs_b["text"] = obs["text"] + 1.0
        action_b = model(obs_b)

    assert action_a.shape == (BATCH, N_ACTIONS)
    assert not torch.allclose(action_a, action_b)


def test_onnx_wrapper_has_two_inputs():
    obs = _actor_obs()
    model = _make_actor(obs)
    onnx_model = model.as_onnx(verbose=False)

    dummies = onnx_model.get_dummy_inputs()
    assert len(dummies) == 2
    assert dummies[0].shape == (1, OBS_DIM)
    assert dummies[1].shape == (1, TEXT_DIM)
    assert onnx_model.input_names == ["obs", "text_emb"]
    assert onnx_model.output_names == ["actions"]


def test_onnx_roundtrip_matches_pytorch(tmp_path):
    """Actually export the two-input graph and run it through onnxruntime."""
    obs = _actor_obs()
    model = _make_actor(obs)
    onnx_model = model.as_onnx(verbose=False)
    onnx_model.eval()
    path = tmp_path / "vla_policy.onnx"

    torch.onnx.export(
        onnx_model,
        onnx_model.get_dummy_inputs(),
        str(path),
        export_params=True,
        opset_version=18,
        input_names=onnx_model.input_names,
        output_names=onnx_model.output_names,
        dynamic_axes={},
        dynamo=False,
    )

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    # Production export uses fixed batch 1 (dynamic_axes={} in the runner),
    # exactly like the existing 61D policy.onnx.
    obs_in = torch.randn(1, OBS_DIM)
    text_in = torch.randn(1, TEXT_DIM)
    got = session.run(
        None,
        {"obs": obs_in.numpy(), "text_emb": text_in.numpy()},
    )[0]
    with torch.no_grad():
        expected = onnx_model(obs_in, text_in).numpy()
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)

    assert [i.name for i in session.get_inputs()] == ["obs", "text_emb"]
    assert [i.shape for i in session.get_inputs()] == [[1, OBS_DIM], [1, TEXT_DIM]]


def test_jit_script_handles_two_inputs():
    obs = _actor_obs()
    model = _make_actor(obs)
    scripted = torch.jit.script(model.as_jit())
    out = scripted(torch.randn(2, OBS_DIM), torch.randn(2, TEXT_DIM))
    assert out.shape == (2, N_ACTIONS)


def test_no_text_group_degrades_to_plain_mlp():
    """The same class is safe for the critic, which has no text group."""
    obs = _actor_obs()
    groups = {"critic": ("critic",)}
    model = MicroduckVLAModel(
        obs,
        groups,
        "critic",
        1,
        hidden_dims=(32,),
        activation="elu",
        obs_normalization=True,
    )

    assert model.text_group is None
    assert model._text_dim == 0
    latent = model.get_latent(obs)
    assert latent.shape == (BATCH, OBS_DIM + 13)

    onnx_model = model.as_onnx(verbose=False)
    assert len(onnx_model.get_dummy_inputs()) == 1
    assert onnx_model.input_names == ["obs"]


def test_class_name_resolves_through_rsl_rl():
    from rsl_rl.utils import resolve_callable

    resolved = resolve_callable("mjlab_microduck.vla.model:MicroduckVLAModel")
    assert resolved is MicroduckVLAModel


def test_rl_cfg_points_at_the_vla_model():
    from mjlab_microduck.tasks.microduck_vla_env_cfg import MicroduckVlaRlCfg

    assert (
        MicroduckVlaRlCfg.actor.class_name
        == "mjlab_microduck.vla.model:MicroduckVLAModel"
    )
    # The critic must NOT use the VLA model: it has no text group and should
    # stay on the stock MLP path.
    assert MicroduckVlaRlCfg.critic.class_name == "MLPModel"

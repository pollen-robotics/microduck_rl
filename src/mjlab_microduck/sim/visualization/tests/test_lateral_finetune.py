"""Small CPU checks for restoring and attributing a lateral-policy candidate."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
from onnx import TensorProto, helper, numpy_helper
from rsl_rl.models import MLPModel
from tensordict import TensorDict
from torch import nn

from mjlab_microduck.sim.visualization.lateral import finetune


def test_recipe_snapshots_point_to_real_isolated_implementations():
    assert finetune.REPO == Path(__file__).resolve().parents[5]
    assert all((finetune.REPO / name).is_file() for name in finetune.SNAPSHOT_FILES)
    required = {
        "config.py",
        "env_cfg.py",
        "finetune.py",
        "validate.py",
        "register.py",
        "__init__.py",
    }
    snapshots = {
        Path(name).name
        for name in finetune.SNAPSHOT_FILES
        if "/visualization/lateral/" in name
    }
    assert required <= snapshots
    assert {
        "src/mjlab_microduck/sim/visualization/bam.py",
        "src/mjlab_microduck/sim/visualization/physics.py",
    } <= set(finetune.SNAPSHOT_FILES)


@pytest.mark.parametrize("name", ["finetune", "validate"])
def test_module_cli_help_is_available_without_starting_an_environment(name):
    module = f"mjlab_microduck.sim.visualization.lateral.{name}"
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert "--policy" in result.stdout and "--output" in result.stdout


@pytest.mark.parametrize(
    "statement",
    [
        (
            "from mjlab_microduck.sim.visualization.lateral.config import make_lateral_cfg; "
            "assert make_lateral_cfg().commands['twist'].lateral_speed == 0.3"
        ),
        (
            "import mjlab_microduck.tasks; "
            "from mjlab.tasks.registry import list_tasks; "
            "assert 'Mjlab-Lateral-Flat-Backlash-MicroDuck' in list_tasks()"
        ),
        (
            "import sys; import mjlab_microduck.sim.visualization.lateral.finetune; "
            "assert 'torch' not in sys.modules"
        ),
    ],
)
def test_isolated_modules_and_task_plugin_load_in_a_fresh_process(statement):
    subprocess.run(
        [sys.executable, "-c", statement],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.fixture
def policy(tmp_path):
    rng = np.random.default_rng(7)
    tensors = [
        numpy_helper.from_array(rng.normal(0, 0.01, shape).astype(np.float32), name)
        for name, shape in finetune.SHAPES.items()
    ]
    tensors += [
        numpy_helper.from_array(rng.normal(0, 0.1, (1, 61)).astype(np.float32), "mean"),
        numpy_helper.from_array(
            rng.uniform(0.1, 0.8, (1, 61)).astype(np.float32), "denominator"
        ),
    ]
    nodes = [
        helper.make_node("Sub", ["obs", "mean"], ["centered"]),
        helper.make_node("Div", ["centered", "denominator"], ["normalized"]),
    ]
    current = "normalized"
    for layer in (0, 2, 4, 6):
        output = "actions" if layer == 6 else f"linear_{layer}"
        nodes.append(
            helper.make_node(
                "Gemm",
                [current, f"mlp.{layer}.weight", f"mlp.{layer}.bias"],
                [output],
                transB=1,
            )
        )
        current = output
        if layer != 6:
            nodes.append(helper.make_node("Elu", [current], [f"elu_{layer}"]))
            current = f"elu_{layer}"
    graph = helper.make_graph(
        nodes,
        "fixture",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, 61])],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, 14])],
        tensors,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=9
    )
    for key, value in {
        "joint_names": "contract-joints",
        "action_scale": "1.0",
        "run_path": "upstream:original-run",
        "training_commit": "old-commit",
        "training": '{"run":"old-run"}',
        "seed": "999",
    }.items():
        item = model.metadata_props.add()
        item.key, item.value = key, value
    path = tmp_path / "base.onnx"
    onnx.save(model, path)
    return path


def actor():
    observations = TensorDict(
        {"actor": torch.zeros(1, 61), "critic": torch.zeros(1, 76)}, batch_size=[1]
    )
    return MLPModel(
        observations,
        {"actor": ["actor"], "critic": ["critic"]},
        "actor",
        14,
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 0.1,
            "std_type": "scalar",
        },
    )


def test_restored_actor_matches_onnx_and_keeps_normalizer_frozen(policy):
    original = policy.read_bytes()
    model = actor()
    source = finetune.read_actor(policy)
    assert finetune.restore_actor(model, source, 0.1) < 1e-5
    assert torch.allclose(model.distribution.std_param, torch.full((14,), 0.1))
    buffers = {
        name: value.clone() for name, value in model.obs_normalizer.state_dict().items()
    }
    model.train()
    model.update_normalization({"actor": torch.full((128, 61), 100.0)})
    for name, value in model.obs_normalizer.state_dict().items():
        assert torch.equal(value, buffers[name])
    assert policy.read_bytes() == original


def test_checkpoint_load_requires_reapplying_nonserialized_freeze(policy):
    first = actor()
    finetune.restore_actor(first, finetune.read_actor(policy), 0.1)
    loaded = actor()
    loaded.load_state_dict(first.state_dict())
    assert loaded.obs_normalizer.until is None
    finetune.freeze_actor_normalizer(loaded)
    mean = loaded.obs_normalizer._mean.clone()
    loaded.train()
    loaded.update_normalization({"actor": torch.full((128, 61), 100.0)})
    assert torch.equal(mean, loaded.obs_normalizer._mean)


@pytest.mark.parametrize("change", ["relu", "zero_denominator", "bad_shape"])
def test_incompatible_actor_graphs_are_rejected(policy, change):
    model = onnx.load(policy)
    if change == "relu":
        model.graph.node[3].op_type = "Relu"
    else:
        index = next(
            i
            for i, value in enumerate(model.graph.initializer)
            if value.name
            == ("denominator" if change == "zero_denominator" else "mlp.0.weight")
        )
        name = model.graph.initializer[index].name
        array = np.zeros(
            (1, 61) if change == "zero_denominator" else (511, 61), dtype=np.float32
        )
        model.graph.initializer[index].CopyFrom(numpy_helper.from_array(array, name))
    onnx.save(model, policy)
    with pytest.raises(ValueError):
        finetune.read_actor(policy)


def test_candidate_preserves_contract_but_does_not_claim_upstream_training(
    policy, tmp_path
):
    original = policy.read_bytes()
    source = finetune.read_actor(policy)
    candidate = tmp_path / "candidate.onnx"
    candidate.write_bytes(original)
    provenance = {
        "run_id": "local-test",
        "task": finetune.TASK,
        "git_revision": "new-commit",
        "git_dirty": True,
        "seed": 3,
        "updates_completed": 25,
        "base_policy_sha256": source.sha256,
        "resume_semantics": "actor_from_onnx_fresh_critic_optimizer",
    }
    finetune.annotate_candidate(candidate, source, provenance)
    metadata = {item.key: item.value for item in onnx.load(candidate).metadata_props}
    assert metadata["joint_names"] == "contract-joints"
    assert metadata["action_scale"] == "1.0"
    assert metadata["run_path"] == "local:local-test"
    assert metadata["training_git_revision"] == "new-commit"
    assert metadata["training_iterations_completed"] == "25"
    assert metadata["validation_status"] == "unvalidated_candidate"
    assert "training_commit" not in metadata and "seed" not in metadata
    assert (
        json.loads(metadata["base_policy_metadata"])["run_path"]
        == "upstream:original-run"
    )
    assert (
        json.loads(metadata["training"])["resume_semantics"]
        == provenance["resume_semantics"]
    )
    assert policy.read_bytes() == original
    with pytest.raises(ValueError, match="overwrite"):
        finetune.annotate_candidate(policy, source, provenance)


def test_source_snapshots_preserve_dirty_files_and_keep_resume_runs_separate(
    tmp_path, monkeypatch
):
    repo = tmp_path / "repo"
    for name in finetune.SNAPSHOT_FILES:
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"original {name}\n")
    monkeypatch.setattr(finetune, "REPO", repo)
    output = tmp_path / "run"

    # Dataclass configs record function identities; no Python object-address reprs.
    cfg = {"reward": finetune.read_actor, "solver": {"iterations": 30}}
    first = finetune.save_source_snapshot(output, "first", cfg, {"epochs": 4})
    changed = repo / finetune.SNAPSHOT_FILES[0]
    changed.write_text("modified before resume\n")
    second = finetune.save_source_snapshot(output, "resume", cfg, {"epochs": 1})
    assert first["directory"] != second["directory"]
    assert (
        (Path(first["directory"]) / finetune.SNAPSHOT_FILES[0])
        .read_text()
        .startswith("original")
    )
    assert (
        Path(second["directory"]) / finetune.SNAPSHOT_FILES[0]
    ).read_text() == "modified before resume\n"
    recorded = json.loads((Path(first["directory"]) / "configuration.json").read_text())
    assert recorded["environment"]["reward"].endswith(".read_actor")
    assert recorded["runner"] == {"epochs": 4}
    with pytest.raises(FileExistsError):
        finetune.save_source_snapshot(output, "first", cfg, {})


class TinyActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(61, 14)
        self.obs_normalizer = nn.Identity()

    def forward(self, values):
        return self.linear(values["actor"])


def test_anchor_scope_excludes_only_raw_pure_sideways_commands():
    observations = torch.zeros(8, 61)
    observations[:, 48:51] = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.3, 0.0],
            [0.0, -0.3, 0.0],
            [0.3, 0.0, 0.0],
            [-0.3, 0.0, 0.0],
            [0.0, 0.0, 0.6],
            [0.0, 0.3, 0.0],
            [0.0, 0.3, 0.0],
        ]
    )
    observations[6, 53] = 0.1  # preserve a requested head pose, even while strafing
    observations[7, 58] = 0.1  # likewise body pose commands
    assert finetune.anchor_frame_mask(observations).tolist() == [
        True,
        False,
        False,
        True,
        True,
        True,
        True,
        True,
    ]


def test_anchor_backpropagates_only_on_nonlateral_samples_and_never_into_teacher():
    actor_model, teacher = TinyActor(), TinyActor().eval()
    teacher.requires_grad_(False)
    observations = torch.randn(4, 61, requires_grad=True)
    with torch.no_grad():
        observations[:, 48:61] = 0
        observations[1, 49] = 0.3
        observations[2, 49] = -0.3
        observations[3, 48] = -0.3
    anchor, mirror, fraction = finetune.actor_regularization_losses(
        actor_model, teacher, observations, mirror=False
    )
    assert fraction.item() == 0.5 and mirror.item() == 0
    anchor.backward()
    assert torch.count_nonzero(observations.grad[1:3]) == 0
    assert observations.grad[[0, 3]].abs().sum() > 0
    assert all(parameter.grad is None for parameter in teacher.parameters())


def test_mirror_is_a_training_loss_not_an_action_average():
    actor_model = TinyActor()
    observations = torch.randn(5, 61)
    original = actor_model({"actor": observations}).detach().clone()
    anchor, mirror, _fraction = finetune.actor_regularization_losses(
        actor_model, None, observations, mirror=True
    )
    assert anchor.item() == 0 and mirror.item() > 0
    mirror.backward()
    assert actor_model.linear.weight.grad.abs().sum() > 0
    assert torch.equal(actor_model({"actor": observations}).detach(), original)


def test_zero_regularizer_weights_keep_the_unmodified_ppo_path(policy):
    from rsl_rl.algorithms import PPO

    assert finetune.regularized_ppo_class(finetune.read_actor(policy), 0.0, 0.0) is PPO
    args = finetune.parser().parse_args(["--policy", str(policy), "--output", "unused"])
    assert args.anchor_weight == 0 and args.mirror_loss_coef == 0


def test_regularized_update_joins_existing_steps_and_excludes_stale_rollout(
    policy, monkeypatch
):
    from copy import deepcopy

    from rsl_rl.algorithms import PPO

    cls = finetune.regularized_ppo_class(finetune.read_actor(policy), 2.0, 0.1)
    algorithm = cls.__new__(cls)
    algorithm.actor, algorithm.critic = TinyActor(), nn.Linear(76, 1)
    algorithm.local_reference = deepcopy(algorithm.actor).eval().requires_grad_(False)
    with torch.no_grad():
        algorithm.actor.linear.bias.add_(0.2)
    # The third storage row is stale and must not participate in the extra loss.
    observations = torch.randn(3, 2, 61)
    observations[:, :, 48:61] = 0
    observations[0, 0, 49] = 0.3
    observations[2] = float("nan")
    algorithm.storage = type(
        "Storage",
        (),
        {
            "step": 2,
            "observations": {"actor": observations},
        },
    )()
    algorithm.optimizer = torch.optim.Adam(
        [*algorithm.actor.parameters(), *algorithm.critic.parameters()], lr=0.001
    )
    algorithm.max_grad_norm = 1.0
    algorithm.is_multi_gpu = False
    expected_critic = deepcopy(algorithm.critic)
    critic_optimizer = torch.optim.Adam(expected_critic.parameters(), lr=0.001)
    calls = 0

    def ordinary_update(self):
        nonlocal calls
        # Exercise the real optimizer hook twice, as upstream's minibatch loop
        # does. A preservation pass must not add any more Adam steps.
        for _ in range(2):
            self.optimizer.zero_grad()
            value = self.critic(torch.ones(2, 76)).square().mean()
            value.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            calls += 1
            critic_optimizer.zero_grad()
            expected_critic(torch.ones(2, 76)).square().mean().backward()
            torch.nn.utils.clip_grad_norm_(expected_critic.parameters(), 1.0)
            critic_optimizer.step()
        self.storage.step = 0
        return {"surrogate": 1.0}

    monkeypatch.setattr(PPO, "update", ordinary_update)
    actor_before = algorithm.actor.linear.bias.detach().clone()
    result = algorithm.update()
    assert result["surrogate"] == 1.0
    assert result["local_anchor_fraction"] == 0.75
    assert result["local_anchor"] > 0 and result["local_mirror"] > 0
    assert not torch.equal(actor_before, algorithm.actor.linear.bias)
    assert all(
        torch.equal(expected_critic.state_dict()[key], value)
        for key, value in algorithm.critic.state_dict().items()
    )
    assert calls == 2
    for parameter in algorithm.actor.parameters():
        assert algorithm.optimizer.state[parameter]["step"].item() == 2
    assert algorithm.actor.obs_normalizer.until == 0
    assert algorithm.max_grad_norm == 1.0
    assert not algorithm.optimizer._optimizer_step_pre_hooks


def test_preservation_gradient_is_summed_before_clipping_and_single_sgd_step(
    policy, monkeypatch
):
    from copy import deepcopy

    from rsl_rl.algorithms import PPO

    cls = finetune.regularized_ppo_class(finetune.read_actor(policy), 2.0, 0.0)
    algorithm = cls.__new__(cls)
    algorithm.actor, algorithm.critic = TinyActor(), nn.Linear(76, 1)
    algorithm.local_reference = deepcopy(algorithm.actor).eval().requires_grad_(False)
    with torch.no_grad():
        algorithm.actor.linear.bias.add_(0.2)
    observations = torch.ones(2, 61)
    observations[:, 48:61] = 0
    algorithm.storage = type(
        "Storage", (), {"step": 1, "observations": {"actor": observations[None]}}
    )()
    algorithm.optimizer = torch.optim.SGD(algorithm.actor.parameters(), lr=0.1)
    algorithm.max_grad_norm, algorithm.is_multi_gpu = 0.25, False
    expected = deepcopy(algorithm.actor)
    expected_optimizer = torch.optim.SGD(expected.parameters(), lr=0.1)
    anchor, _, _ = finetune.actor_regularization_losses(
        expected, algorithm.local_reference, observations, mirror=False
    )
    # The PPO stand-in deliberately pulls against the preservation target.
    combined = -expected({"actor": observations}).mean() + 2.0 * anchor
    combined.backward()
    torch.nn.utils.clip_grad_norm_(expected.parameters(), 0.25)
    expected_optimizer.step()

    def ordinary_update(self):
        self.optimizer.zero_grad()
        (-self.actor({"actor": observations}).mean()).backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.storage.step = 0
        return {}

    monkeypatch.setattr(PPO, "update", ordinary_update)
    algorithm.update()
    for actual, target in zip(algorithm.actor.parameters(), expected.parameters()):
        torch.testing.assert_close(actual, target, rtol=1e-6, atol=1e-7)

    def failed_update(self):
        raise RuntimeError("test update failure")

    algorithm.storage.step = 1
    monkeypatch.setattr(PPO, "update", failed_update)
    with pytest.raises(RuntimeError, match="test update failure"):
        algorithm.update()
    assert algorithm.max_grad_norm == 0.25
    assert not algorithm.optimizer._optimizer_step_pre_hooks

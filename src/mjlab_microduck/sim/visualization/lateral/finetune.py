"""Fine-tune a new lateral policy locally, without modifying the base ONNX.

The exported candidate is unvalidated. This entry point performs actual PPO on
the lateral task; it never adds a heading controller to deployment commands.
ONNX supplies actor weights and normalization only. A fresh run starts a new
critic, action distribution and optimizer. --resume restores this script's
runner checkpoint, but resets environment trajectories and random-number state;
it is not a bit-for-bit continuation of a previous simulation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import subprocess
from copy import deepcopy
from dataclasses import asdict, dataclass, fields, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

TASK = "Mjlab-Lateral-Flat-Backlash-MicroDuck"
REPO = Path(__file__).resolve().parents[5]
SHAPES = {
    "mlp.0.weight": (512, 61),
    "mlp.0.bias": (512,),
    "mlp.2.weight": (256, 512),
    "mlp.2.bias": (256,),
    "mlp.4.weight": (128, 256),
    "mlp.4.bias": (128,),
    "mlp.6.weight": (14, 128),
    "mlp.6.bias": (14,),
}

SNAPSHOT_FILES = (
    "src/mjlab_microduck/sim/visualization/lateral/__init__.py",
    "src/mjlab_microduck/sim/visualization/lateral/config.py",
    "src/mjlab_microduck/sim/visualization/lateral/env_cfg.py",
    "src/mjlab_microduck/sim/visualization/lateral/finetune.py",
    "src/mjlab_microduck/sim/visualization/lateral/register.py",
    "src/mjlab_microduck/sim/visualization/lateral/validate.py",
    "src/mjlab_microduck/tasks/__init__.py",
    "src/mjlab_microduck/sim/visualization/bam.py",
    "src/mjlab_microduck/sim/visualization/physics.py",
    "src/mjlab_microduck/sim/body_server.py",
    "src/mjlab_microduck/sim/visualization/scenes/__init__.py",
    "src/mjlab_microduck/sim/visualization/scenes/scene_velstand.xml",
    "uv.lock",
)


def config_value(value):
    """Stable JSON view of config values, including reward and command callables."""
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: config_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, dict):
        return {str(key): config_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [config_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if callable(value):
        return f"{value.__module__}.{value.__qualname__}"
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def save_source_snapshot(output: Path, run_id: str, cfg, train: dict) -> dict:
    """Keep dirty/untracked recipe content; a Git SHA plus dirty=True cannot reproduce it."""
    snapshot = output / "source_snapshots" / run_id
    snapshot.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name in SNAPSHOT_FILES:
        source = REPO / name
        destination = snapshot / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    settings = snapshot / "configuration.json"
    settings.write_text(
        json.dumps(
            {"environment": config_value(cfg), "runner": config_value(train)}, indent=2
        )
        + "\n"
    )
    hashes[settings.name] = hashlib.sha256(settings.read_bytes()).hexdigest()
    manifest = snapshot / "manifest.json"
    manifest.write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n")
    return {
        "directory": str(snapshot),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }


@dataclass(frozen=True)
class ActorSource:
    path: Path
    sha256: str
    weights: dict[str, Any]
    mean: Any
    denominator: Any
    metadata: dict[str, str]


def read_actor(path: Path) -> ActorSource:
    """Accept only the known, feed-forward v8 actor graph and tensor contract."""
    import numpy as np
    import onnx
    from onnx import numpy_helper

    path = path.resolve(strict=True)
    model = onnx.load(str(path), load_external_data=False)
    onnx.checker.check_model(model)
    nodes = list(model.graph.node)
    if [node.op_type for node in nodes] != [
        "Sub",
        "Div",
        "Gemm",
        "Elu",
        "Gemm",
        "Elu",
        "Gemm",
        "Elu",
        "Gemm",
    ]:
        raise ValueError("base policy must be the v8 normalized ELU MLP graph")
    for tensors, shape in ((model.graph.input, (1, 61)), (model.graph.output, (1, 14))):
        if len(tensors) != 1:
            raise ValueError("base policy must have one input and one output")
        tensor = tensors[0].type.tensor_type
        if (
            tensor.elem_type != onnx.TensorProto.FLOAT
            or tuple(dim.dim_value for dim in tensor.shape.dim) != shape
        ):
            raise ValueError("base policy must use float32 [1,61] -> [1,14]")
    values = {}
    for value in model.graph.initializer:
        if value.data_location == onnx.TensorProto.EXTERNAL:
            raise ValueError("external ONNX tensors are not supported")
        array = numpy_helper.to_array(value).copy()
        if array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError(f"invalid ONNX tensor: {value.name}")
        values[value.name] = array
    for name, shape in SHAPES.items():
        if name not in values or values[name].shape != shape:
            raise ValueError(f"base policy has an incompatible tensor: {name}")
    current = model.graph.input[0].name
    for index, node in enumerate(nodes):
        if node.input[0] != current or len(node.output) != 1:
            raise ValueError("base policy graph is not a single feed-forward chain")
        attributes = {
            attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute
        }
        if node.op_type == "Gemm":
            layer = {2: 0, 4: 2, 6: 4, 8: 6}[index]
            if list(node.input[1:]) != [f"mlp.{layer}.weight", f"mlp.{layer}.bias"] or (
                attributes.get("transB", 0) != 1
                or attributes.get("transA", 0) != 0
                or attributes.get("alpha", 1.0) != 1.0
                or attributes.get("beta", 1.0) != 1.0
            ):
                raise ValueError("base policy has an incompatible Gemm operation")
        elif node.op_type == "Elu" and attributes.get("alpha", 1.0) != 1.0:
            raise ValueError("base policy must use ELU alpha=1")
        current = node.output[0]
    if current != model.graph.output[0].name:
        raise ValueError("base policy output does not match the MLP chain")
    try:
        mean = values[nodes[0].input[1]]
        denominator = values[nodes[1].input[1]]
    except (IndexError, KeyError) as error:
        raise ValueError(
            "base policy normalization must use constant tensors"
        ) from error
    if (
        mean.shape != (1, 61)
        or denominator.shape != (1, 61)
        or (denominator <= 0).any()
    ):
        raise ValueError("invalid base policy normalization")
    return ActorSource(
        path,
        hashlib.sha256(path.read_bytes()).hexdigest(),
        values,
        mean,
        denominator,
        {item.key: item.value for item in model.metadata_props},
    )


def freeze_actor_normalizer(actor) -> None:
    # until is a Python attribute, not part of state_dict: reapply after resume.
    actor.obs_normalizer.until = 0


def restore_actor(actor, source: ActorSource, init_std: float) -> float:
    """Restore the deterministic actor and verify its outputs against ONNX Runtime."""
    import numpy as np
    import onnxruntime as ort
    import torch

    state = actor.state_dict()
    for name in SHAPES:
        state[name] = torch.from_numpy(source.weights[name])
    raw_std = source.denominator - actor.obs_normalizer.eps
    if (raw_std < -1e-7).any():
        raise ValueError("ONNX denominator is incompatible with normalizer epsilon")
    state["obs_normalizer._mean"] = torch.from_numpy(source.mean)
    state["obs_normalizer._std"] = torch.from_numpy(np.maximum(raw_std, 0))
    state["obs_normalizer._var"] = state["obs_normalizer._std"].square()
    state["obs_normalizer.count"] = torch.tensor(1, dtype=torch.long)
    state["distribution.std_param"] = torch.full_like(
        state["distribution.std_param"], init_std
    )
    actor.load_state_dict(state, strict=True)
    freeze_actor_normalizer(actor)
    rng = np.random.default_rng(0)
    samples = (source.mean + rng.normal(0, 0.5, (64, 61)) * source.denominator).astype(
        np.float32
    )
    session = ort.InferenceSession(str(source.path), providers=["CPUExecutionProvider"])
    expected = np.concatenate(
        [
            session.run(None, {session.get_inputs()[0].name: row[None]})[0]
            for row in samples
        ]
    )
    device = next(actor.parameters()).device
    with torch.no_grad():
        actual = actor({"actor": torch.from_numpy(samples).to(device)}).cpu().numpy()
    difference = float(np.max(np.abs(actual - expected)))
    if not np.isfinite(difference) or difference > 1e-5:
        raise ValueError(f"restored actor differs from ONNX by {difference}")
    return difference


def anchor_frame_mask(observation):
    """Protect the base gait everywhere except an exact, nonzero pure strafe.

    This consumes raw rollout observations, before the frozen actor normalizer.
    Standing, longitudinal, turning, and mixed commands keep a local v8 target;
    no command or inference action is remapped.
    """
    twist = observation[:, 48:51]
    pure_lateral = (
        (twist[:, 0].abs() < 1e-6)
        & (twist[:, 1].abs() >= 0.05)
        & (twist[:, 2].abs() < 1e-6)
        & (observation[:, 51:61].abs().amax(dim=1) < 1e-6)
    )
    return ~pure_lateral


def actor_regularization_losses(actor, reference, observation, *, mirror: bool):
    """Differentiable training losses; never average mirrored inference outputs."""
    import torch

    prediction = actor({"actor": observation})
    selected = anchor_frame_mask(observation)
    anchor = prediction.sum() * 0.0
    if reference is not None and selected.any():
        with torch.no_grad():
            target = reference({"actor": observation[selected]})
        anchor = (prediction[selected] - target).square().mean()
    mirror_loss = prediction.sum() * 0.0
    if mirror:
        from tensordict import TensorDict

        from mjlab_microduck.tasks.symmetry import microduck_vel_symmetry

        count = observation.shape[0]
        values = TensorDict(
            {"actor": observation, "critic": observation.new_zeros(count, 76)},
            batch_size=[count],
        )
        mirrored, _ = microduck_vel_symmetry(None, values, None)
        _, target = microduck_vel_symmetry(None, None, prediction.detach())
        mirrored_prediction = actor({"actor": mirrored["actor"][count:]})
        mirror_loss = (mirrored_prediction - target[count:]).square().mean()
    return anchor, mirror_loss, selected.float().mean()


def regularized_ppo_class(
    source: ActorSource, anchor_weight: float, mirror_loss_coef: float
):
    """Combine preservation and PPO gradients in each existing optimizer step.

    The local v8 teacher is rebuilt from the supplied ONNX on each start/resume;
    no expert checkpoint or cloud download is involved. The PPO implementation,
    critic updates, optimizer state, and gradient norm bound remain upstream's.
    A second Adam step would give the preservation loss its own momentum update,
    so its weight would not have the meaning of a joint loss coefficient.
    Zero weights use the original PPO class and consume no extra RNG draws.
    """
    import torch
    from rsl_rl.algorithms import PPO

    if anchor_weight == 0 and mirror_loss_coef == 0:
        return PPO

    class LocalRegularizedPPO(PPO):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.local_reference = None
            if anchor_weight:
                self.local_reference = deepcopy(self.actor).eval()
                restore_actor(self.local_reference, source, 0.1)
                self.local_reference.requires_grad_(False)

        def update(self):
            # storage.clear() in PPO.update retires its valid step counter.
            # Snapshot only the completed rollout; do not reuse prior trajectories.
            steps = self.storage.step
            observation = self.storage.observations["actor"][:steps].reshape(-1, 61)
            observation = observation.detach().clone()
            if not observation.numel():
                return super().update()
            if self.is_multi_gpu:
                raise ValueError(
                    "local actor preservation supports one training device"
                )
            freeze_actor_normalizer(self.actor)
            totals = observation.new_zeros(3)
            updates = 0
            bound = self.max_grad_norm

            def add_preservation_gradient(optimizer, args, kwargs):
                nonlocal updates
                sample = observation
                if sample.shape[0] > 512:
                    order = torch.randperm(sample.shape[0], device=sample.device)[:512]
                    sample = sample[order]
                # Upstream has already backpropagated PPO, but has not stepped.
                # Add independent raw-rollout preservation gradients before the
                # SAME clipping/Adam step. There is no second optimizer update.
                with torch.enable_grad():
                    anchor, mirror, fraction = actor_regularization_losses(
                        self.actor,
                        self.local_reference,
                        sample,
                        mirror=bool(mirror_loss_coef),
                    )
                    loss = anchor_weight * anchor + mirror_loss_coef * mirror
                    if not torch.isfinite(loss):
                        raise RuntimeError("non-finite local actor preservation loss")
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), bound)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), bound)
                totals.add_(torch.stack((anchor, fraction, mirror)).detach())
                updates += 1

            handle = self.optimizer.register_step_pre_hook(add_preservation_gradient)
            # PPO normally clips immediately before optimizer.step(). Defer that
            # clipping to the hook above so it sees the sum of both gradients.
            self.max_grad_norm = float("inf")
            try:
                losses = super().update()
            finally:
                handle.remove()
                self.max_grad_norm = bound
            if not updates:
                raise RuntimeError("PPO did not perform an optimizer update")
            losses.update(
                local_anchor=float(totals[0] / updates),
                local_anchor_fraction=float(totals[1] / updates),
                local_mirror=float(totals[2] / updates),
            )
            return losses

    return LocalRegularizedPPO


def annotate_candidate(path: Path, source: ActorSource, provenance: dict) -> None:
    """Preserve the robot contract while replacing inherited training attribution."""
    import onnx

    if path.resolve() == source.path:
        raise ValueError("candidate must not overwrite the base policy")
    if provenance.get("base_policy_sha256") != source.sha256:
        raise ValueError("candidate provenance does not match the base policy")
    model = onnx.load(str(path))
    inherited_provenance = {
        "run_path",
        "checkpoint",
        "seed",
        "iteration",
        "exported",
        "exported_at",
        "export_commit",
        "git_revision",
        "git_commit",
        "revision",
        "commit",
    }
    metadata = {
        key: value
        for key, value in source.metadata.items()
        if key not in inherited_provenance and not key.startswith(("training", "wandb"))
    }
    metadata.update(
        base_policy_metadata=json.dumps(source.metadata, sort_keys=True),
        base_policy_sha256=source.sha256,
        run_path=f"local:{provenance['run_id']}",
        training=json.dumps(provenance, sort_keys=True),
        training_task=provenance["task"],
        training_git_revision=provenance["git_revision"],
        training_git_dirty=str(provenance["git_dirty"]).lower(),
        training_seed=str(provenance["seed"]),
        training_iterations_completed=str(provenance["updates_completed"]),
        validation_status="unvalidated_candidate",
    )
    del model.metadata_props[:]
    for key, value in metadata.items():
        item = model.metadata_props.add()
        item.key, item.value = key, value
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--policy", type=Path, required=True, help="base v8 ONNX; never overwritten"
    )
    result.add_argument(
        "--output", type=Path, required=True, help="candidate/checkpoint directory"
    )
    result.add_argument(
        "--device", default="cpu", help="cpu or a CUDA device such as cuda:0"
    )
    result.add_argument("--num-envs", type=int, default=16)
    result.add_argument(
        "--iterations", type=int, default=100, help="additional PPO updates this run"
    )
    result.add_argument("--seed", type=int, default=0)
    result.add_argument(
        "--resume",
        type=Path,
        help="this script's local checkpoint; environment/RNG reset",
    )
    result.add_argument("--task", default=TASK)
    result.add_argument(
        "--init-std",
        type=float,
        default=0.1,
        help="fresh-run exploration std, absent from ONNX",
    )
    result.add_argument("--learning-rate", type=float, default=3e-5)
    result.add_argument(
        "--anchor-weight",
        type=float,
        default=0.0,
        help="training-only local v8 action MSE on non-pure-lateral frames; 0 disables",
    )
    result.add_argument(
        "--mirror-loss-coef",
        type=float,
        default=0.0,
        help="training-only bilateral action mirror loss; 0 disables",
    )
    result.add_argument("--steps-per-env", type=int, default=24)
    result.add_argument("--epochs", type=int, default=4)
    result.add_argument("--mini-batches", type=int, default=4)
    result.add_argument("--checkpoint-interval", type=int, default=25)
    result.add_argument(
        "--prepare-only", action="store_true", help="construct/verify, without learning"
    )
    return result


def run(args: argparse.Namespace) -> dict:
    import numpy as np
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    import mjlab_microduck.tasks  # noqa: F401 - registers project tasks

    for name in (
        "num_envs",
        "iterations",
        "steps_per_env",
        "epochs",
        "mini_batches",
        "checkpoint_interval",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if (
        args.seed < 0
        or not np.isfinite(args.init_std)
        or args.init_std <= 0
        or not np.isfinite(args.learning_rate)
        or args.learning_rate <= 0
        or not np.isfinite(args.anchor_weight)
        or args.anchor_weight < 0
        or not np.isfinite(args.mirror_loss_coef)
        or args.mirror_loss_coef < 0
    ):
        raise ValueError(
            "seed and loss weights must be nonnegative; std and learning rate must be positive"
        )
    if args.num_envs * args.steps_per_env < args.mini_batches:
        raise ValueError("mini-batches exceed the rollout size")
    device = torch.device(args.device)
    if device.type not in {"cpu", "cuda"} or (
        device.type == "cuda" and not torch.cuda.is_available()
    ):
        raise ValueError("requested CPU/CUDA device is unavailable")
    source = read_actor(args.policy)
    output = args.output.resolve()
    if output == source.path.parent or source.path.parent in output.parents:
        raise ValueError("output must be outside the base policy directory")
    if (output / "candidate.onnx").exists() and args.resume is None:
        raise ValueError(
            "output already contains a candidate; choose a new output or use --resume"
        )
    output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    revision = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    dirty = subprocess.run(
        ["git", "-C", str(REPO), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=False,
    )
    provenance = {
        "run_id": datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ"),
        "task": args.task,
        "git_revision": revision.stdout.strip()
        if revision.returncode == 0
        else "unknown",
        "git_dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else True,
        "seed": args.seed,
        "base_policy_sha256": source.sha256,
        "updates_completed": 0,
        "initial_critic": "new",
        "initial_optimizer": "new",
        "initial_exploration_std": args.init_std,
        "resume_semantics": "actor_from_onnx_fresh_critic_optimizer",
        "device": str(device),
        "validation_status": "unvalidated_candidate",
        "normalizer_frozen": True,
        "hyperparameters": {
            "num_envs": args.num_envs,
            "learning_rate": args.learning_rate,
            "steps_per_env": args.steps_per_env,
            "epochs": args.epochs,
            "mini_batches": args.mini_batches,
            "checkpoint_interval": args.checkpoint_interval,
            "anchor_weight": args.anchor_weight,
            "mirror_loss_coef": args.mirror_loss_coef,
        },
    }
    provenance["local_actor_regularization"] = {
        "reference_policy_sha256": source.sha256,
        "anchor_scope": "non_pure_lateral_raw_observations",
        "anchor_weight": args.anchor_weight,
        "mirror_loss_coef": args.mirror_loss_coef,
        "gradient_integration": "same_ppo_minibatch_optimizer_step",
        "extra_optimizer_steps": 0,
        "max_rollout_samples_per_step": 512,
        "runtime_action_transform": False,
    }
    cfg = load_env_cfg(args.task, play=False)
    cfg.scene.num_envs, cfg.seed = args.num_envs, args.seed
    train = asdict(load_rl_cfg(args.task))
    train.update(
        logger="tensorboard",
        upload_model=False,
        num_steps_per_env=args.steps_per_env,
        save_interval=1,
    )  # runner callback below saves only complete intervals/final update
    train["algorithm"].update(
        class_name=regularized_ppo_class(
            source, args.anchor_weight, args.mirror_loss_coef
        )
        if args.anchor_weight or args.mirror_loss_coef
        else "PPO",
        num_learning_epochs=args.epochs,
        num_mini_batches=args.mini_batches,
        learning_rate=args.learning_rate,
        schedule="fixed",
        symmetry_cfg=None,
    )
    train["algorithm"].pop("bc_cfg", None)
    train["actor"]["distribution_cfg"].update(init_std=args.init_std, std_type="scalar")
    provenance["source_snapshot"] = save_source_snapshot(
        output, provenance["run_id"], cfg, train
    )
    env = ManagerBasedRlEnv(cfg=cfg, device=str(device), render_mode=None)
    target_update = args.iterations
    saved_updates: set[int] = set()

    class LateralRunner(MjlabOnPolicyRunner):
        def save(self, path: str, infos=None) -> None:
            completed = self.current_learning_iteration + 1
            if completed in saved_updates or (
                completed % args.checkpoint_interval and completed != target_update
            ):
                return
            freeze_actor_normalizer(self.alg.actor)
            provenance["updates_completed"] = completed
            super().save(
                str(output / f"model_{completed}.pt"),
                {**(infos or {}), "lateral_finetune": dict(provenance)},
            )
            saved_updates.add(completed)

    try:
        runner = LateralRunner(
            RslRlVecEnvWrapper(env), train, log_dir=str(output), device=str(device)
        )
        if args.resume:
            infos = runner.load(
                str(args.resume.resolve(strict=True)), map_location=str(device)
            )
            prior = (infos or {}).get("lateral_finetune", {})
            if (
                prior.get("base_policy_sha256") != source.sha256
                or prior.get("task") != args.task
            ):
                raise ValueError(
                    "resume checkpoint does not match this script's base policy/task"
                )
            completed = prior["updates_completed"]
            runner.current_learning_iteration = completed
            provenance.update(
                updates_completed=completed,
                initial_critic="checkpoint",
                initial_optimizer="checkpoint",
                resume_checkpoint=str(args.resume.resolve()),
                initial_exploration_std="checkpoint",
                resume_semantics="runner_weights_optimizer_counters_only_environment_rng_reset",
            )
            target_update = completed + args.iterations
            runner.alg.learning_rate = args.learning_rate
            for group in runner.alg.optimizer.param_groups:
                group["lr"] = args.learning_rate
        else:
            provenance["restored_actor_max_abs_error"] = restore_actor(
                runner.alg.actor, source, args.init_std
            )
        freeze_actor_normalizer(runner.alg.actor)
        provenance["requested_additional_updates"] = args.iterations
        provenance["status"] = "prepared" if args.prepare_only else "running"
        (output / "training.json").write_text(json.dumps(provenance, indent=2) + "\n")
        if args.prepare_only:
            return provenance
        frozen_mean = runner.alg.actor.obs_normalizer._mean.detach().clone()
        frozen_std = runner.alg.actor.obs_normalizer._std.detach().clone()
        runner.learn(
            num_learning_iterations=args.iterations, init_at_random_ep_len=False
        )
        if not torch.equal(
            frozen_mean, runner.alg.actor.obs_normalizer._mean
        ) or not torch.equal(frozen_std, runner.alg.actor.obs_normalizer._std):
            raise RuntimeError("actor normalization changed during fine-tuning")
        provenance.update(
            updates_completed=runner.current_learning_iteration + 1, status="candidate"
        )
        runner.export_policy_to_onnx(str(output), filename="candidate.onnx")
        annotate_candidate(output / "candidate.onnx", source, provenance)
        (output / "training.json").write_text(json.dumps(provenance, indent=2) + "\n")
        return provenance
    finally:
        env.close()


def main() -> None:
    args = parser().parse_args()
    try:
        result = run(args)
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

"""`uv run publish` — put a policy on the Hub in the shape the microduck daemon loads.

    # From a wandb run (exports with the normalizer baked in — the only safe path from a checkpoint)
    uv run publish --task Mjlab-PoliteBow-Flat-MicroDuck --wandb-run-path ent/proj/run --checkpoint 3000 \\
        --repo <user>/microduck-polite-bow --kind episodic --duration-s 4.0

    # From an ONNX file you already exported
    uv run publish --onnx out.onnx --repo <user>/microduck-flamingo --kind perpetual --unwind-s 1.5

    # A gait for a slot (no hold, no unwind: it runs until told otherwise)
    uv run publish --onnx walk.onnx --repo <user>/microduck-my-walk --kind perpetual --slot walk

Either way the repo gets `policy.onnx`, a schema-2 `manifest.json` and a README (plus `replay.mp4`
when given `--video`, which the Hub's replay widget shows), the file is
checked for the 61 -> 14 shape and smoke-run before anything is uploaded, and an existing
`policy.onnx` is not overwritten without `--force`. `--dry-run` writes the repo contents to a
local directory and stops.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NoReturn

import tyro

from mjlab_microduck.publish import manifest as m


@dataclass(frozen=True)
class PublishConfig:
    # -- where it goes
    repo: str
    """Hub repo id, `<user-or-org>/microduck-<name>`. Created (private) if it does not exist."""
    kind: Literal["episodic", "perpetual"]
    """episodic: runs `duration_s` and comes back on its own. perpetual: holds until told."""

    # -- where the weights come from: exactly one of (--task + checkpoint) or --onnx
    task: str | None = None
    """Task id to export from, e.g. Mjlab-PoliteBow-Flat-MicroDuck. Needs a checkpoint."""
    wandb_run_path: str | None = None
    """`entity/project/run_id`. With --task, the checkpoint source."""
    checkpoint: int | None = None
    """Checkpoint iteration (model_<N>.pt). Default: the run's latest."""
    checkpoint_file: str | None = None
    """A local model_<N>.pt instead of wandb."""
    onnx: str | None = None
    """An already-exported ONNX. Validated, not re-exported."""
    video: str | None = None
    """An MP4 of the policy running (e.g. from scripts/render_policy.py). Uploaded as replay.mp4."""

    base_model: str | None = None
    """Remix: the Hub repo of the policy this one was trained from (warm start / fine-tune).
    Declared as `base_model` in the model card so the Hub links the two."""

    # -- what the manifest says
    name: str | None = None
    """What a client asks for (`robotctl robot do <name>`). Default: the repo's stem minus `microduck-`."""
    description: str | None = None
    """One line. Default: the task id."""
    duration_s: float | None = None
    """episodic only: seconds it runs."""
    chain: bool = False
    """episodic only: a held button chains another run (roulade does, a kick does not)."""
    unwind_s: float | None = None
    """perpetual held pose (flamingo): seconds the daemon drives `idle` before handing back. Leave unset for a gait."""
    slot: Literal["walk", "stand", "sitstand", "ground_pick", "kick_left", "kick_right", "roulade"] | None = None
    """perpetual gait: which slot it is for (walk, stand, ...). Display-only; drives the install hint."""
    idle: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """The twist that means 'stop doing the thing'. Zeros for every one-shot published so far."""
    action_scale: float | None = None
    """The policy's own output scale, if it wants one. Default: the gait's."""
    entry_pose: str = "standing"
    """The pose the policy expects to start from."""
    twist_help: str | None = None
    """Prose for `command.twist` when the slots mean something (flamingo: '[flag, side, 0]')."""

    # -- training checkpoint (opt-in)
    share_training: bool = False
    """Also share the training checkpoint (+ its distillation teachers) as a bundle in a SEPARATE repo
    (`share-training`, docs/training_checkpoints.md); the manifest's `training.bundle` points to it.
    Needs a checkpoint source (--wandb-run-path or --checkpoint-file), not --onnx."""
    training_repo: str | None = None
    """Bundle repo. Default: `<owner of --repo>/microduck-training`."""
    training_public: bool = False
    """Create the bundle repo public (existing repos keep their visibility)."""

    # -- how
    private: bool = True
    """Create the repo private (--no-private for public). Existing repos keep their visibility."""
    force: bool = False
    """Overwrite an existing policy.onnx in the repo."""
    tag: str | None = None
    """Tag the resulting revision, e.g. v1."""
    smoke: bool = True
    """Run the network on plausible inputs and refuse NaNs before uploading."""
    dry_run: bool = False
    """Write policy.onnx, manifest.json and README.md to ./publish-<name>/ and stop."""
    device: str | None = None
    """Export device. Default: cuda:0 if available, else cpu."""


def _fail(msg: str) -> NoReturn:
    print(f"[publish] error: {msg}", file=sys.stderr)
    sys.exit(2)


def _resolve_weights(cfg: PublishConfig, workdir: Path) -> tuple[Path, dict]:
    """The ONNX to publish and the provenance it carries. Exports when given a checkpoint."""
    from_checkpoint = cfg.task is not None or cfg.checkpoint_file is not None
    if (cfg.onnx is None) == (not from_checkpoint):
        _fail("give exactly one source: --onnx <file>, or --task <id> with --wandb-run-path/--checkpoint-file")

    training: dict = {"repo": "pollen-robotics/microduck_rl", **m.git_provenance()}
    if cfg.onnx is not None:
        src = Path(cfg.onnx)
        training["source_file"] = src.name
        if cfg.task:
            training["task_id"] = cfg.task
        return src, training

    if cfg.task is None:
        _fail("--checkpoint-file needs --task <id> to build the env it was trained in")
    if cfg.wandb_run_path is None and cfg.checkpoint_file is None:
        _fail("--task needs --wandb-run-path (and optionally --checkpoint) or --checkpoint-file")

    # Heavy imports only on this path: the ONNX path must work without a GPU or mjlab's registry.
    import mjlab.tasks  # noqa: F401  (populates the registry)
    from mjlab_microduck.export import ExportConfig, run_export

    out = workdir / m.POLICY_FILE
    result = run_export(
        cfg.task,
        ExportConfig(
            onnx_file=str(out),
            wandb_run_path=cfg.wandb_run_path,
            checkpoint=cfg.checkpoint,
            checkpoint_file=cfg.checkpoint_file,
            num_envs=1,
            device=cfg.device,
        ),
    )
    training["task_id"] = cfg.task
    if result.wandb_run_path:
        training["run"] = result.wandb_run_path
    if result.checkpoint_iteration is not None:
        training["checkpoint"] = result.checkpoint_iteration
    return result.onnx_path, training


def _share_training(cfg: PublishConfig, onnx_path: Path, training: dict) -> str:
    """Bundle the checkpoint this ONNX came from; returns its hf:// spec for the manifest."""
    from mjlab_microduck.publish.training import DEFAULT_REPO_NAME, ShareConfig, share

    if cfg.onnx is not None:
        _fail("--share-training needs the checkpoint: publish with --task + --wandb-run-path/--checkpoint-file, not --onnx")
    owner = cfg.repo.split("/", 1)[0]
    return share(
        ShareConfig(
            wandb_run_path=cfg.wandb_run_path if cfg.checkpoint_file is None else None,
            checkpoint=training.get("checkpoint"),
            checkpoint_file=cfg.checkpoint_file,
            task=cfg.task,
            repo=cfg.training_repo or f"{owner}/{DEFAULT_REPO_NAME}",
            public=cfg.training_public,
            dry_run=cfg.dry_run,
        ),
        onnx_files=[(onnx_path, {"repo": cfg.repo, "file": m.POLICY_FILE})],
    )


def _default_name(repo: str) -> str:
    stem = repo.rsplit("/", 1)[-1]
    return stem.removeprefix("microduck-").removeprefix("microduck_") or stem


def run(cfg: PublishConfig) -> int:
    if "/" not in cfg.repo:
        _fail("--repo must be `<user-or-org>/<name>`")
    name = cfg.name or _default_name(cfg.repo)
    video = Path(cfg.video) if cfg.video is not None else None
    if video is not None and (not video.is_file() or video.suffix.lower() != ".mp4"):
        _fail(f"--video {cfg.video}: expected an existing .mp4 file")

    workdir = Path(tempfile.mkdtemp(prefix="microduck-publish-"))
    try:
        onnx_path, training = _resolve_weights(cfg, workdir)
        shape = m.check_onnx(onnx_path)
        print(f"[publish] {onnx_path.name}: {shape.obs_len} -> {shape.action_len}, ok")
        if cfg.smoke:
            m.smoke_run_onnx(onnx_path)
            print("[publish] smoke run: finite, non-constant output")

        if cfg.share_training:
            training["bundle"] = _share_training(cfg, onnx_path, training)

        command_help = {"twist": cfg.twist_help} if cfg.twist_help else None
        manifest = m.build_manifest(
            name=name,
            kind=cfg.kind,
            description=cfg.description or training.get("task_id") or name,
            duration_s=cfg.duration_s,
            chain=cfg.chain,
            unwind_s=cfg.unwind_s,
            idle=cfg.idle,
            action_scale=cfg.action_scale,
            entry_pose=cfg.entry_pose,
            slot=cfg.slot,
            command_help=command_help,
            training=training,
        )
        m.validate_manifest(manifest)

        staged = workdir / "repo"
        staged.mkdir()
        shutil.copyfile(onnx_path, staged / m.POLICY_FILE)
        (staged / "manifest.json").write_text(m.dump_manifest(manifest))
        (staged / "README.md").write_text(m.render_readme(manifest, cfg.repo, cfg.base_model))
        if video is not None:
            shutil.copyfile(video, staged / m.REPLAY_FILE)

        if cfg.dry_run:
            dest = Path.cwd() / f"publish-{name}"
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(staged, dest)
            print(f"[publish] dry run: wrote {dest}/ ({', '.join(sorted(p.name for p in dest.iterdir()))})")
            return 0

        from huggingface_hub import HfApi

        api = HfApi()
        if cfg.base_model is not None and not api.repo_exists(cfg.base_model, repo_type="model"):
            _fail(f"--base-model {cfg.base_model}: no such model repo on the Hub (or no access)")
        api.create_repo(cfg.repo, repo_type="model", private=cfg.private, exist_ok=True)
        existing = set(api.list_repo_files(cfg.repo))
        onnx_files = {f for f in existing if f.endswith(".onnx")}
        if onnx_files and not cfg.force:
            _fail(
                f"{cfg.repo} already carries {sorted(onnx_files)}; --force overwrites. "
                "A repo carries exactly one .onnx, so a second name is a new repo."
            )
        stale = onnx_files - {m.POLICY_FILE}
        commit = api.upload_folder(
            repo_id=cfg.repo,
            folder_path=str(staged),
            commit_message=f"publish {name}: {cfg.kind}, {training.get('task_id', onnx_path.name)}",
            delete_patterns=sorted(stale) or None,
        )
        url = getattr(commit, "commit_url", None) or f"https://huggingface.co/{cfg.repo}"
        print(f"[publish] uploaded: {url}")
        if cfg.tag:
            api.create_tag(cfg.repo, tag=cfg.tag, tag_message=f"{name} {cfg.tag}")
            print(f"[publish] tagged {cfg.tag}")
        first = m.install_commands(manifest, cfg.repo).splitlines()[0]
        print(f"[publish] on a robot: {first}")
        return 0
    except m.ManifestError as e:
        _fail(str(e))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return 0


def main() -> int:
    cfg = tyro.cli(PublishConfig)
    return run(cfg)


if __name__ == "__main__":
    sys.exit(main())

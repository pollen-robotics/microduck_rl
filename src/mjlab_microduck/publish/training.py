"""`uv run share-training` — put a training checkpoint on the Hub so anyone can resume or warm-start from it.

    # A wandb checkpoint (+ the distillation teachers its run used, found in the run's own config)
    uv run share-training --wandb-run-path pollen-robotics/mjlab_microduck/0045i2si --checkpoint 2749

    # ... and record (after verifying the weights) the deployed ONNX it became
    uv run share-training --wandb-run-path ent/proj/run --checkpoint 2749 \\
        --onnx hf://pollen-robotics/microduck-policies/velstand.onnx

    # Or opt in while publishing a policy: `uv run publish ... --share-training`

A bundle is one folder of a SEPARATE model repo (default `<owner>/microduck-training`, private until
you say `--public`), never the ONNX repo:

    bundles/<experiment>/<run>_<iter>/
      model_<iter>.pt     full rsl_rl checkpoint: weights, obs normalizers, optimizer, iteration,
                          common_step_counter — resuming keeps the curricula where they were
      bundle.json         task, code commit (+ uncommitted.diff when the run had one), the exact train
                          command, the checkpoint it was resumed from, teacher bundles, verified ONNX
      wandb_config.yaml   the run's resolved env/agent cfg
      README.md           resume / warm-start / export commands

Bundles are immutable and keyed by run + iteration: re-sharing one only merges new ONNX records, and a
teacher shared by many runs is uploaded once. The repo root README indexes every bundle. Load side:
`hf://<repo>/bundles/<experiment>/<run>_<iter>` works wherever a checkpoint is read (checkpoints.py).
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, NoReturn

import tyro

from mjlab_microduck.checkpoints import BUNDLE_FILE, HF_SCHEME, parse_hf

BUNDLE_SCHEMA = 1
DEFAULT_REPO_NAME = "microduck-training"
CODE_REPO = "pollen-robotics/microduck_rl"
_REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class ShareConfig:
    wandb_run_path: str | None = None
    """`entity/project/run_id` of the training run."""
    checkpoint: int | None = None
    """Iteration (model_<N>.pt). Default: the run's latest."""
    checkpoint_file: str | None = None
    """A local model_<N>.pt instead of wandb (needs --task; provenance = this checkout's HEAD)."""
    task: str | None = None
    """Task id. Default: read from the wandb run's command line."""
    onnx: tuple[str, ...] = ()
    """ONNX files this checkpoint was exported to (local path or hf://owner/repo/file.onnx). Each is
    verified against the checkpoint's actor weights before being recorded."""
    repo: str | None = None
    """Training repo. Default: `<owner>/microduck-training`."""
    owner: str | None = None
    """Owner for the default repo. Default: your HF user name."""
    public: bool = False
    """Create the training repo public. Existing repos keep their visibility."""
    teachers: bool = True
    """Also share the distillation teachers referenced in the run's config (skipped if already shared)."""
    allow_dirty: bool = False
    """--checkpoint-file only: allow a dirty checkout (the diff is NOT recorded: commit first)."""
    tag: str | None = None
    """Tag the training repo after the upload, e.g. velstand-v8."""
    dry_run: bool = False
    """Stage the bundle(s) under ./share-training/ and stop."""


def _fail(msg: str) -> NoReturn:
    print(f"[share-training] error: {msg}", file=sys.stderr)
    sys.exit(2)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _retry(fn: Callable[[], Any], what: str, tries: int = 4) -> Any:
    """The wandb file listing intermittently returns an empty GraphQL page (TypeError inside wandb)."""
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if i == tries - 1:
                raise RuntimeError(f"{what}: {e!r}") from e
            time.sleep(2 * (i + 1))


# --------------------------------------------------------------------------- provenance helpers


def parse_train_args(args: list[str]) -> dict[str, Any]:
    """Task id and the checkpoint a `train` command line loaded (`--wandb-run-path` + resume, or
    `--init-checkpoint`). Accepts `--flag value` and `--flag=value`, dashes or underscores."""
    flat: list[str] = []
    for a in args:
        flat.extend(a.split("=", 1) if a.startswith("--") and "=" in a else [a])
    opts: dict[str, str] = {}
    for i, a in enumerate(flat):
        if a.startswith("--") and i + 1 < len(flat) and not flat[i + 1].startswith("--"):
            opts[a[2:].replace("-", "_")] = flat[i + 1]
    out: dict[str, Any] = {"task_id": flat[0] if flat and not flat[0].startswith("-") else None, "loaded_from": None}
    if opts.get("init_checkpoint"):
        out["loaded_from"] = {"spec": opts["init_checkpoint"]}
    elif opts.get("wandb_run_path") and opts.get("agent.resume", "").lower() == "true":
        m = re.search(r"model_(\d+)\.pt", opts.get("wandb_checkpoint_name", ""))
        out["loaded_from"] = {"wandb_run": opts["wandb_run_path"], "checkpoint": int(m.group(1)) if m else None}
    return out


def find_teacher_refs(cfg: Any, path: str = "") -> dict[str, dict[str, Any]]:
    """Every `<prefix>wandb_run_path` (+ `<prefix>checkpoint_name`) in a run's config, keyed by
    `<dotted.path>.<prefix>` — today the distillation expert and walk anchor in `bc_cfg`."""
    found: dict[str, dict[str, Any]] = {}
    if isinstance(cfg, dict):
        for k, v in cfg.items():
            if isinstance(k, str) and k.endswith("wandb_run_path") and isinstance(v, str) and v:
                prefix = k[: -len("wandb_run_path")]
                name = cfg.get(f"{prefix}checkpoint_name") or ""
                m = re.search(r"model_(\d+)\.pt", name)
                found[f"{path}{prefix or 'expert_'}".rstrip("_")] = {"wandb_run": v, "checkpoint": int(m.group(1)) if m else None}
            elif isinstance(v, (dict, list)):
                found.update(find_teacher_refs(v, f"{path}{k}."))
    elif isinstance(cfg, list):
        for i, v in enumerate(cfg):
            found.update(find_teacher_refs(v, f"{path}{i}."))
    return found


def commit_on_remote(commit: str | None) -> bool | None:
    """True if `commit` is reachable from a remote-tracking branch of this checkout (None: unknown)."""
    if not commit:
        return None
    try:
        out = subprocess.run(["git", "-C", str(_REPO_ROOT), "branch", "-r", "--contains", commit],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(out.stdout.strip()) if out.returncode == 0 else False


def onnx_matches_checkpoint(onnx_path: Path, ckpt_path: Path) -> bool:
    """Every actor MLP tensor of the checkpoint appears bit-exact among the ONNX initializers.

    The test used to recover deployed-policy provenance by hand (AGENTS.md); names are ignored so
    older exports with renamed initializers still match.
    """
    import numpy as np
    import onnx
    import torch
    from onnx import numpy_helper

    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    actor = sd.get("actor_state_dict") or sd.get("model_state_dict") or {}
    want = [v.numpy() for k, v in actor.items() if ("mlp." in k or k.startswith("actor.")) and v.ndim >= 1]
    if not want:
        return False
    have: dict[tuple, list] = {}
    for init in onnx.load(str(onnx_path)).graph.initializer:
        a = numpy_helper.to_array(init)
        have.setdefault(a.shape, []).append(a)
        if a.ndim == 2:
            have.setdefault(a.T.shape, []).append(a.T)
    return all(any(np.array_equal(w, h) for h in have.get(w.shape, [])) for w in want)


# --------------------------------------------------------------------------- bundle staging


@dataclass
class Bundle:
    id: str                       # "<experiment>/<run>_<iter>"
    files: dict[str, Path]        # name in the bundle dir -> local file
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def dir(self) -> str:
        return f"bundles/{self.id}"


def _wandb_checkpoint(run, iteration: int | None) -> tuple[str, int]:
    if iteration is not None:  # direct lookup: listing a long run's files can fail at wandb's default page size
        name = f"model_{iteration}.pt"
        if not _retry(lambda: run.file(name).size, f"stat {name}"):
            _fail(f"{name} not in run {'/'.join(run.path)}")
        return name, iteration
    names = _retry(lambda: [f.name for f in run.files(per_page=50) if re.fullmatch(r"model_\d+\.pt", f.name)],
                   f"list files of {run.path}")
    if not names:
        _fail(f"run {'/'.join(run.path)} has no model_<N>.pt files")
    its = sorted(int(n[6:-3]) for n in names)
    it = its[-1] if iteration is None else iteration
    if it not in its:
        _fail(f"model_{it}.pt not in run {'/'.join(run.path)} (has {its[-5:]})")
    return f"model_{it}.pt", it


def stage_from_wandb(run_path: str, iteration: int | None, workdir: Path, task: str | None = None) -> Bundle:
    import wandb

    run = wandb.Api().run(run_path)
    ckpt_name, it = _wandb_checkpoint(run, iteration)
    cfg = dict(run.config)
    exp = (cfg.get("train_cfg") or {}).get("experiment_name") or "unknown"
    meta = run.metadata or {}
    args = parse_train_args(list(meta.get("args") or []))
    run_id = run_path.rstrip("/").split("/")[-1]
    bid = f"{exp}/{run_id}_{it}"
    d = workdir / bid.replace("/", "__")
    d.mkdir(parents=True, exist_ok=True)

    files: dict[str, Path] = {}
    wanted = {ckpt_name: ckpt_name, "config.yaml": "wandb_config.yaml", "microduck_rl.diff": "uncommitted.diff"}
    for src, dst in wanted.items():
        f = _retry(lambda src=src: run.file(src), f"stat {src}")
        if not f.size:  # absent files come back with size 0; so does a clean run's empty diff
            continue
        _retry(lambda f=f: f.download(root=str(d), replace=True), f"download {src}")
        (d / src).rename(d / dst)
        files[dst] = d / dst

    commit = run.commit or (meta.get("git") or {}).get("commit")
    meta_out = {
        "schema": BUNDLE_SCHEMA,
        "id": bid,
        "task_id": task or args["task_id"],
        "experiment_name": exp,
        "iteration": it,
        "checkpoint_file": ckpt_name,
        "checkpoint_sha256": _sha256(files[ckpt_name]),
        "source": {"wandb_run": run_path, "run_name": run.name, "created": str(run.created_at)},
        "code": {"repo": CODE_REPO, "commit": commit, "on_remote": commit_on_remote(commit),
                 "uncommitted_diff": "uncommitted.diff" if "uncommitted.diff" in files else None},
        "command": ["uv", "run", "train", *(meta.get("args") or [])],
        "loaded_from": args["loaded_from"],
        "teachers": {k: {"wandb_run": v["wandb_run"], "checkpoint": v["checkpoint"]}
                     for k, v in find_teacher_refs(cfg.get("train_cfg") or {}).items()},
        "onnx": [],
    }
    return Bundle(bid, files, meta_out)


def stage_from_file(ckpt: Path, task: str, workdir: Path, allow_dirty: bool) -> Bundle:
    import mjlab.tasks  # noqa: F401  (registry)
    from mjlab.tasks.registry import load_rl_cfg

    from mjlab_microduck.publish.manifest import git_provenance

    git = git_provenance(_REPO_ROOT)
    if git.get("dirty") and not allow_dirty:
        _fail("this checkout is dirty: commit (and push) first so the bundle's commit reproduces it, or --allow-dirty")
    m = re.search(r"model_(\d+)\.pt$", ckpt.name)
    it = int(m.group(1)) if m else 0
    exp = load_rl_cfg(task).experiment_name
    full = subprocess.run(["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or None
    bid = f"{exp}/local-{_sha256(ckpt)[:10]}_{it}"
    d = workdir / bid.replace("/", "__")
    d.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ckpt, d / ckpt.name)
    meta = {
        "schema": BUNDLE_SCHEMA, "id": bid, "task_id": task, "experiment_name": exp, "iteration": it,
        "checkpoint_file": ckpt.name, "checkpoint_sha256": _sha256(ckpt), "source": {"file": ckpt.name},
        "code": {"repo": CODE_REPO, "commit": full, "on_remote": commit_on_remote(full),
                 "dirty": bool(git.get("dirty")), "uncommitted_diff": None},
        "command": None, "loaded_from": None, "teachers": {}, "onnx": [],
    }
    return Bundle(bid, {ckpt.name: d / ckpt.name}, meta)


def _fetch_onnx(spec: str, workdir: Path) -> tuple[Path, dict[str, Any]]:
    ref = parse_hf(spec)
    if ref is None:
        p = Path(spec)
        if not p.is_file():
            _fail(f"--onnx {spec}: no such file")
        return p, {"file": p.name}
    from huggingface_hub import HfApi, hf_hub_download

    local = Path(hf_hub_download(ref.repo, ref.path, revision=ref.revision, local_dir=str(workdir / "onnx" / ref.repo)))
    sha = HfApi().list_repo_commits(ref.repo, revision=ref.revision)[0].commit_id if ref.revision is None else ref.revision
    return local, {"repo": ref.repo, "file": ref.path, "revision": sha}


def attach_onnx(bundle: Bundle, specs: tuple[str, ...], workdir: Path,
                local: list[tuple[Path, dict[str, Any]]] = ()) -> None:
    """Verify and record ONNX exports: `specs` (paths / hf:// refs) plus pre-fetched (file, where) pairs."""
    ckpt = bundle.files[bundle.meta["checkpoint_file"]]
    for spec, (local, where) in [(s, _fetch_onnx(s, workdir)) for s in specs] + [(str(p), (p, w)) for p, w in local]:
        if not onnx_matches_checkpoint(local, ckpt):
            _fail(f"{spec} does not carry the actor weights of {bundle.id}: refusing to record it")
        bundle.meta["onnx"].append({**where, "sha256": _sha256(local), "verified": "actor weights bit-exact"})
        print(f"[share-training] {spec}: weights match {bundle.id}")


# --------------------------------------------------------------------------- cards


def bundle_spec(repo: str, bundle_id: str) -> str:
    return f"{HF_SCHEME}{repo}/bundles/{bundle_id}"


def render_bundle_readme(meta: dict[str, Any], repo: str) -> str:
    spec = bundle_spec(repo, meta["id"])
    task = meta.get("task_id") or "<task-id>"
    code = meta["code"]
    lines = [
        f"# {meta['id']}",
        "",
        f"Training checkpoint `{meta['checkpoint_file']}` of `{task}` (experiment `{meta['experiment_name']}`), "
        f"from [{CODE_REPO}](https://github.com/{CODE_REPO}).",
        "",
        "| | |",
        "|---|---|",
        f"| code | `{code.get('commit') or '?'}`"
        + ("" if code.get("on_remote") else " (not on a remote branch of the code repo)")
        + (" + `uncommitted.diff`" if code.get("uncommitted_diff") else "") + " |",
    ]
    src = meta.get("source") or {}
    if src.get("wandb_run"):
        lines.append(f"| wandb run | `{src['wandb_run']}` ({src.get('run_name', '')}) |")
    lf = meta.get("loaded_from")
    if lf:
        lines.append(f"| resumed from | `{lf.get('bundle') or lf.get('spec') or (str(lf.get('wandb_run')) + ' @ ' + str(lf.get('checkpoint')))}` |")
    for k, t in (meta.get("teachers") or {}).items():
        lines.append(f"| teacher `{k}` | `{t.get('bundle') or (t['wandb_run'] + ' @ ' + str(t['checkpoint']))}` |")
    for o in meta.get("onnx") or []:
        lines.append(f"| exported to | `{o.get('repo', '')}/{o['file']}` (sha256 `{o['sha256'][:12]}…`, {o['verified']}) |")
    lines += [
        "",
        "## Use it",
        "",
        "```bash",
        f"git clone https://github.com/{CODE_REPO} && cd microduck_rl && git checkout {code.get('commit') or '<commit>'}",
    ]
    if code.get("uncommitted_diff"):
        lines.append(f"# the run had uncommitted changes: hf download {repo} bundles/{meta['id']}/uncommitted.diff, then git apply it")
    lines += [
        "",
        "# resume (same task: iteration and curricula continue where this checkpoint left them)",
        f"uv run train {task} --env.scene.num-envs 4096 --init-checkpoint {spec} --agent.max-iterations 1000",
        "",
        "# warm start (another task: weights/normalizer/optimizer kept, curricula restart at 0)",
        f"MICRODUCK_WARM_START=1 uv run train <other-task> --init-checkpoint {spec}",
        "",
        "# ONNX for the robot (normalizer baked in)",
        f"uv run scripts/export.py {task} --checkpoint-file {spec}",
        "```",
        "",
    ]
    if meta.get("command"):
        lines += ["The command that trained it:", "", "```bash", " ".join(meta["command"]), "```", ""]
    lines += ["`bundle.json` holds the same provenance, machine-readable.", ""]
    return "\n".join(lines)


def render_index(metas: list[dict[str, Any]], repo: str) -> str:
    rows = sorted(metas, key=lambda m: m["id"])
    out = [
        "---",
        "tags: [microduck, reinforcement-learning, mjlab, rsl-rl]",
        "---",
        "",
        f"# {repo}",
        "",
        f"Training checkpoints of [{CODE_REPO}](https://github.com/{CODE_REPO}) policies (full rsl_rl `.pt`: "
        "weights, normalizers, optimizer, curriculum step counter), to resume or warm-start training from. "
        "The deployable ONNX policies live in their own repos; each bundle below records which ONNX it "
        "was exported to.",
        "",
        "```bash",
        f"uv run train <task> --init-checkpoint hf://{repo}/bundles/<experiment>/<run>_<iter>",
        "```",
        "",
        "| bundle | task | exported to | code |",
        "|---|---|---|---|",
    ]
    for m in rows:
        onnx = ", ".join(f"`{o.get('repo', '')}/{o['file']}`" for o in m.get("onnx") or []) or "—"
        commit = (m.get("code") or {}).get("commit") or "?"
        out.append(f"| [`{m['id']}`](bundles/{m['id']}) | `{m.get('task_id') or '?'}` | {onnx} | `{commit[:9]}` |")
    out += ["", "Written by `uv run share-training` (see `docs/training_checkpoints.md` in the code repo).", ""]
    return "\n".join(out)


# --------------------------------------------------------------------------- upload


def _existing_meta(api, repo: str, bundle_id: str) -> dict[str, Any] | None:
    from huggingface_hub import hf_hub_download

    if not api.file_exists(repo, f"bundles/{bundle_id}/{BUNDLE_FILE}"):
        return None
    return json.loads(Path(hf_hub_download(repo, f"bundles/{bundle_id}/{BUNDLE_FILE}", force_download=True)).read_text())


def _merge_onnx(old: list[dict], new: list[dict]) -> list[dict]:
    seen = {o["sha256"] for o in old}
    return old + [o for o in new if o["sha256"] not in seen]


def _write(bundle: Bundle, repo: str, dest: Path) -> None:
    d = dest / bundle.dir
    d.mkdir(parents=True, exist_ok=True)
    for name, src in bundle.files.items():
        shutil.copyfile(src, d / name)
    (d / BUNDLE_FILE).write_text(json.dumps(bundle.meta, indent=2) + "\n")
    (d / "README.md").write_text(render_bundle_readme(bundle.meta, repo))


def share(cfg: ShareConfig, onnx_files: list[tuple[Path, dict[str, Any]]] = ()) -> str:
    """Stage (and unless dry-run, upload) the bundle and its teachers. Returns the bundle's hf:// spec.

    `onnx_files`: (local file, {"repo", "file"}) exports to verify and record — `publish` passes the
    policy it is about to upload."""
    from huggingface_hub import HfApi

    api = HfApi()
    if cfg.repo:
        repo = cfg.repo
    else:
        owner = cfg.owner or (api.whoami()["name"] if not cfg.dry_run else "<owner>")
        repo = f"{owner}/{DEFAULT_REPO_NAME}"
    if (cfg.wandb_run_path is None) == (cfg.checkpoint_file is None):
        _fail("give exactly one source: --wandb-run-path [--checkpoint N], or --checkpoint-file with --task")

    workdir = Path(tempfile.mkdtemp(prefix="microduck-share-"))
    staged = workdir / "repo"
    try:
        if cfg.wandb_run_path:
            main = stage_from_wandb(cfg.wandb_run_path, cfg.checkpoint, workdir, cfg.task)
        else:
            if cfg.task is None or cfg.checkpoint_file is None:
                _fail("--checkpoint-file needs --task")
            main = stage_from_file(Path(cfg.checkpoint_file), cfg.task, workdir, cfg.allow_dirty)
        attach_onnx(main, cfg.onnx, workdir, list(onnx_files))

        if not cfg.dry_run:
            api.create_repo(repo, repo_type="model", private=not cfg.public, exist_ok=True)

        # Teachers first, so the main bundle can point at them.
        todo: list[Bundle] = []
        if cfg.teachers:
            for key, t in main.meta["teachers"].items():
                tid_probe = t["wandb_run"].rstrip("/").split("/")[-1]
                tb = stage_from_wandb(t["wandb_run"], t["checkpoint"], workdir)
                t["bundle"] = bundle_spec(repo, tb.id)
                if not cfg.dry_run and _existing_meta(api, repo, tb.id) is not None:
                    print(f"[share-training] teacher {key} ({tid_probe}) already shared: {tb.id}")
                    continue
                todo.append(tb)
        lf = main.meta.get("loaded_from") or {}
        if lf.get("wandb_run") and lf.get("checkpoint") is not None and not cfg.dry_run:
            pid = lf["wandb_run"].rstrip("/").split("/")[-1]
            parents = [p for p in api.list_repo_files(repo) if re.fullmatch(rf"bundles/[^/]+/{pid}_{lf['checkpoint']}/{BUNDLE_FILE}", p)]
            if parents:
                lf["bundle"] = f"{HF_SCHEME}{repo}/{parents[0].rsplit('/', 1)[0]}"

        if not cfg.dry_run:
            old = _existing_meta(api, repo, main.id)
            if old is not None:
                if old.get("checkpoint_sha256") != main.meta["checkpoint_sha256"]:
                    _fail(f"{main.id} already exists with a different checkpoint: bundles are immutable")
                old["onnx"] = _merge_onnx(old.get("onnx") or [], main.meta["onnx"])
                main.meta = old
                main.files = {}  # the checkpoint is already there: only bundle.json/README change
        todo.append(main)

        for b in todo:
            _write(b, repo, staged)

        if cfg.dry_run:
            dest = Path.cwd() / "share-training"
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(staged, dest)
            (dest / "README.md").write_text(render_index([b.meta for b in todo], repo))
            print(f"[share-training] dry run: wrote {dest}/ ({', '.join(b.id for b in todo)})")
            return bundle_spec(repo, main.id)

        from huggingface_hub import hf_hub_download

        metas = {b.id: b.meta for b in todo}
        for p in api.list_repo_files(repo):
            m = re.fullmatch(rf"bundles/(.+)/{BUNDLE_FILE}", p)
            if m and m.group(1) not in metas:
                metas[m.group(1)] = json.loads(Path(hf_hub_download(repo, p)).read_text())
        (staged / "README.md").write_text(render_index(list(metas.values()), repo))
        commit = api.upload_folder(repo_id=repo, folder_path=str(staged),
                                   commit_message=f"share {', '.join(b.id for b in todo)}")
        print(f"[share-training] uploaded: {getattr(commit, 'commit_url', repo)}")
        if cfg.tag:
            api.create_tag(repo, tag=cfg.tag, tag_message=main.id, exist_ok=True)
            print(f"[share-training] tagged {cfg.tag}")
        spec = bundle_spec(repo, main.id)
        print(f"[share-training] resume with: uv run train {main.meta.get('task_id') or '<task>'} --init-checkpoint {spec}")
        return spec
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def main() -> int:
    share(tyro.cli(ShareConfig))
    return 0


if __name__ == "__main__":
    sys.exit(main())

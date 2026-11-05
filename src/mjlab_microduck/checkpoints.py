"""Checkpoint specs: a local file, or a training bundle on the Hugging Face Hub.

    hf://<owner>/<repo>[@<revision>]/<path>

`<path>` is either a `.pt` file or a bundle directory (`bundles/<experiment>/<run>_<iter>`), which
resolves to the checkpoint its `bundle.json` names. Bundles are written by `uv run share-training`
(or `uv run publish --share-training`); see `docs/training_checkpoints.md`.

One resolver serves every place a checkpoint is read:
  - `uv run train <task> --init-checkpoint <spec>`: resume from a bundle (the train hook consumes the
    flag; the load happens just before `learn`, exactly where mjlab's own `--agent.resume` loads).
    Prefix `MICRODUCK_WARM_START=1` to restart the curricula when loading into another task.
  - distillation teachers: `hf_checkpoint` / `anchor_hf_checkpoint` in `bc_cfg` (distill.py).
  - `scripts/export.py --checkpoint-file <spec>` and `uv run publish --checkpoint-file <spec>`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

HF_SCHEME = "hf://"
INIT_FLAG = "--init-checkpoint"
#: Set by the train hook from `--init-checkpoint`; read once, just before the runner learns.
INIT_CHECKPOINT_ENV = "MICRODUCK_INIT_CHECKPOINT"
BUNDLE_FILE = "bundle.json"
_PATCHED = False


@dataclass(frozen=True)
class HfRef:
    repo: str
    revision: str | None
    path: str

    def __str__(self) -> str:
        rev = f"@{self.revision}" if self.revision else ""
        return f"{HF_SCHEME}{self.repo}{rev}/{self.path}".rstrip("/")


def parse_hf(spec: str) -> HfRef | None:
    """`hf://owner/repo[@rev][/path]` → HfRef; anything else → None."""
    if not spec.startswith(HF_SCHEME):
        return None
    parts = spec[len(HF_SCHEME):].split("/", 2)
    if len(parts) < 2 or not parts[0] or not parts[1]:
        raise ValueError(f"{spec!r}: expected hf://<owner>/<repo>[@<revision>]/<path>")
    repo, _, rev = parts[1].partition("@")
    return HfRef(f"{parts[0]}/{repo}", rev or None, (parts[2] if len(parts) == 3 else "").strip("/"))


def resolve(spec: str | os.PathLike) -> Path:
    """A local checkpoint path for `spec`, downloading (into the HF cache) when it is an hf:// ref."""
    ref = parse_hf(str(spec))
    if ref is None:
        path = Path(spec)
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint {path} does not exist")
        return path
    from huggingface_hub import hf_hub_download

    path = ref.path
    if not path.endswith(".pt"):
        meta = hf_hub_download(ref.repo, f"{path}/{BUNDLE_FILE}".lstrip("/"), revision=ref.revision)
        path = f"{path}/{json.loads(Path(meta).read_text())['checkpoint_file']}".lstrip("/")
    local = Path(hf_hub_download(ref.repo, path, revision=ref.revision))
    print(f"[checkpoints] {spec} -> {local}")
    return local


def consume_init_flag(argv: list[str]) -> list[str]:
    """Strip `--init-checkpoint <spec>` (or `=<spec>`) from argv into INIT_CHECKPOINT_ENV.

    Called by the train hook at import time, before mjlab's tyro parse would reject the flag.
    """
    out, spec, i = [], None, 0
    while i < len(argv):
        a = argv[i]
        if a == INIT_FLAG:
            if i + 1 >= len(argv):
                raise SystemExit(f"{INIT_FLAG} needs a value: a .pt path or hf://<owner>/<repo>/<bundle>")
            spec, i = argv[i + 1], i + 2
            continue
        if a.startswith(INIT_FLAG + "="):
            spec = a.split("=", 1)[1]
        else:
            out.append(a)
        i += 1
    if spec is not None:
        if any(a in ("--agent.resume", "--agent.resume=True") for a in out):
            raise SystemExit(f"{INIT_FLAG} replaces --agent.resume/--wandb-run-path: give one or the other")
        os.environ[INIT_CHECKPOINT_ENV] = spec
    return out


def install_init_checkpoint_patch() -> None:
    """Load INIT_CHECKPOINT_ENV into the runner right before its first `learn` (a no-op without it).

    mjlab's train script resumes with `runner.load(path)` then `runner.learn(...)`; this does the same
    for a spec mjlab cannot resolve itself. `load` restores the iteration and `common_step_counter`
    (a resume: curricula stay where the checkpoint left them) — with MICRODUCK_WARM_START=1, mdp.py's
    Patch 5 resets both (a warm start).
    """
    from mjlab.rl.runner import MjlabOnPolicyRunner

    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True
    orig_learn = MjlabOnPolicyRunner.learn

    def learn(self, *args, **kwargs):
        spec = os.environ.pop(INIT_CHECKPOINT_ENV, None)
        if spec:
            path = resolve(spec)
            print(f"[checkpoints] loading {INIT_FLAG} {spec}")
            self.load(str(path))
        return orig_learn(self, *args, **kwargs)

    MjlabOnPolicyRunner.learn = learn

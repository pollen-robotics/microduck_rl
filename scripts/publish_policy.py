#!/usr/bin/env python
"""Share a Microduck policy on the Hugging Face Hub — see docs/sharing-policies.md.

    uv run scripts/publish_policy.py validate policy.onnx
    uv run scripts/publish_policy.py publish policy.onnx --name flamingo-cycle --namespace RemiFabre \
        --manifest manifest.json --card-extra card_body.md [--media preview.mp4] [--public] [--dry-run]
    uv run scripts/publish_policy.py fetch RemiFabre/microduck-flamingo-cycle --to /tmp/flamingo

Needs only numpy, onnxruntime and huggingface_hub (no mjlab / GPU).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

OBS_LEN = 61
ACTION_LEN = 14
OBS_INPUT_NAME = "obs"
SCHEMA_VERSION = 2
MODEL_API = 1
REQUIRED_MANIFEST = ("name", "kind", "action_scale", "entry_pose", "command", "robot", "training", "description")
TAGS = ["microduck", "microduck-policy", "mjlab", "robotics"]
TEMPLATE = Path(__file__).with_name("policy_card_template.md")


# ── contract ──────────────────────────────────────────────────────────────────

@dataclass
class Report:
    ok: bool
    problems: list
    warnings: list
    input_name: str
    input_shape: list
    output_shape: list
    metadata: dict

    def text(self) -> str:
        lines = ["daemon contract OK" if self.ok else "CONTRACT VIOLATIONS:"]
        lines += [f"  - {p}" for p in self.problems]
        lines += [f"  ! {w}" for w in self.warnings]
        lines.append(f"  input {self.input_name}{self.input_shape} -> output {self.output_shape}")
        if self.metadata:
            lines.append(f"  metadata keys: {sorted(self.metadata)}")
        return "\n".join(lines)


def validate_onnx(path: Path) -> Report:
    """Same checks as robotd at load (duck-control/src/policy.rs) and the studio's contract.py."""
    problems, warnings = [], []
    sess = ort.InferenceSession(str(path))
    inp, out = sess.get_inputs()[0], sess.get_outputs()[0]
    if inp.name != OBS_INPUT_NAME:
        problems.append(f'input tensor is named "{inp.name}" - the daemon requires exactly "{OBS_INPUT_NAME}"')
    if inp.type != "tensor(float)":
        problems.append(f"input dtype is {inp.type}, must be f32")
    if not inp.shape or inp.shape[-1] != OBS_LEN:
        problems.append(f"input shape {inp.shape}: trailing dim must be {OBS_LEN}")
    if not out.shape or out.shape[-1] != ACTION_LEN:
        problems.append(f"output shape {out.shape}: trailing dim must be {ACTION_LEN}")
    meta = dict(sess.get_modelmeta().custom_metadata_map or {})
    if not meta:
        warnings.append("no ONNX metadata - expected export via scripts/export.py (which bakes the obs normalizer)")
    if not problems:
        act = sess.run(None, {inp.name: np.zeros((1, OBS_LEN), np.float32)})[0]
        if not np.all(np.isfinite(act)):
            problems.append("zero-obs inference produced non-finite actions")
        elif np.abs(act).max() > 10.0:
            warnings.append(f"zero-obs inference gives |action| up to {np.abs(act).max():.1f} - suspicious at rest")
    return Report(not problems, problems, warnings, inp.name, list(inp.shape), list(out.shape), meta)


# ── manifest ──────────────────────────────────────────────────────────────────

def build_manifest(user: dict, name: str) -> dict:
    missing = [k for k in REQUIRED_MANIFEST if k not in user]
    if missing:
        raise SystemExit(f"manifest is missing {missing} (see docs/sharing-policies.md)")
    if user["kind"] not in ("perpetual", "episodic"):
        raise SystemExit("manifest.kind must be 'perpetual' or 'episodic'")
    if user["kind"] == "episodic" and not user.get("duration_s"):
        raise SystemExit("episodic policies need duration_s")
    cmd = user["command"]
    if not (isinstance(cmd.get("twist"), list) and len(cmd["twist"]) == 3 and isinstance(cmd.get("idle"), list) and len(cmd["idle"]) == 3):
        raise SystemExit("manifest.command needs 'twist' (3 strings) and 'idle' (3 numbers)")
    for k in ("task_id", "repo", "commit", "run"):
        if k not in user["training"]:
            raise SystemExit(f"manifest.training.{k} is required")
    if user.get("name", name) != name:
        raise SystemExit(f"manifest.name {user['name']!r} != --name {name!r}")
    m = {"schema_version": SCHEMA_VERSION, "model_api": MODEL_API, "name": name, "kind": user["kind"],
         "obs_len": OBS_LEN, "action_len": ACTION_LEN, "action_scale": float(user["action_scale"]),
         "entry_pose": user["entry_pose"], "duration_s": user.get("duration_s"),
         "description": user["description"], "command": cmd, "robot": user["robot"],
         "training": user["training"], "eval": user.get("eval", {})}
    return m


# ── card ──────────────────────────────────────────────────────────────────────

def build_card(manifest: dict, repo_id: str, extra_md: str, has_media: bool, media_name: str = "preview.mp4", extra_files: list | None = None) -> str:
    tpl = TEMPLATE.read_text()
    t = manifest["training"]
    cmd = manifest["command"]
    twist_rows = "\n".join(f"| twist[{i}] | {s} |" for i, s in enumerate(cmd["twist"]))
    video = (f'<video controls muted loop src="https://huggingface.co/{repo_id}/resolve/main/media/{media_name}" style="max-width:640px;width:100%"></video>'
             if has_media else "")
    fields = {
        "VIDEO": video,
        "EXTRA_FILES": "".join(f" · `{n}`" for n in (extra_files or [])),
        "REPO_ID": repo_id, "NAME": manifest["name"], "DESCRIPTION": manifest["description"],
        "KIND": manifest["kind"], "ENTRY_POSE": manifest["entry_pose"],
        "TAGS": ", ".join(TAGS), "TWIST_ROWS": twist_rows,
        "HEAD": cmd.get("head", "unused (zeros)"), "BODY": cmd.get("body", "unused (zeros)"),
        "IDLE": json.dumps(cmd["idle"]), "EXTRA": extra_md.strip(),
        "TASK_ID": t["task_id"], "TRAIN_REPO": t["repo"], "COMMIT": t["commit"], "RUN": t["run"],
        "ACTION_SCALE": manifest["action_scale"],
        "MEDIA_LINE": f" · `media/{media_name}` (sim rollout)" if has_media else "",
    }
    for k, v in fields.items():
        tpl = tpl.replace("{{" + k + "}}", str(v))
    if "{{" in tpl:
        raise SystemExit("card template has unfilled fields: " + tpl[tpl.index("{{"):tpl.index("{{") + 40])
    return tpl


# ── commands ──────────────────────────────────────────────────────────────────

def cmd_validate(a):
    r = validate_onnx(Path(a.onnx))
    print(r.text())
    sys.exit(0 if r.ok else 1)


def cmd_publish(a):
    onnx = Path(a.onnx).expanduser().resolve()
    r = validate_onnx(onnx)
    print(r.text())
    if not r.ok:
        raise SystemExit("refusing to publish: the ONNX fails the daemon contract")
    manifest = build_manifest(json.loads(Path(a.manifest).read_text()), a.name)
    repo_id = f"{a.namespace}/microduck-{a.name}"
    extra = Path(a.card_extra).read_text() if a.card_extra else ""
    media = Path(a.media).expanduser().resolve() if a.media else None
    extra_files = [Path(e).name for e in (a.extra or [])]
    card = build_card(manifest, repo_id, extra, media is not None, "preview" + (media.suffix if media else ".mp4"), extra_files)
    out = Path(a.build_dir).expanduser().resolve() if a.build_dir else onnx.parent / f"hub-{a.name}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "README.md").write_text(card)
    files = [(onnx, "policy.onnx"), (out / "manifest.json", "manifest.json"), (out / "README.md", "README.md")]
    if media:
        files.append((media, "media/preview" + media.suffix))
    for extra in a.extra or []:
        ep = Path(extra).expanduser().resolve()
        if ep.suffix not in (".py", ".md", ".json", ".toml", ".txt"):
            raise SystemExit(f"--extra {ep.name}: only small text files (.py/.md/.json/.toml/.txt) belong in a policy repo")
        files.append((ep, ep.name))
    print(f"\nrepo: https://huggingface.co/{repo_id} ({'public' if a.public else 'private'})")
    for local, remote in files:
        print(f"  {remote:22s} <- {local}")
    if a.dry_run:
        print("dry run: nothing uploaded; card + manifest written to", out)
        return
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=not a.public, exist_ok=True)
    for local, remote in files:
        api.upload_file(path_or_fileobj=str(local), path_in_repo=remote, repo_id=repo_id, repo_type="model",
                        commit_message=f"publish {a.name}: {remote}")
    print("published https://huggingface.co/" + repo_id)


def cmd_fetch(a):
    from huggingface_hub import hf_hub_download
    to = Path(a.to).expanduser().resolve()
    to.mkdir(parents=True, exist_ok=True)
    for f in ("policy.onnx", "manifest.json"):
        p = hf_hub_download(a.repo_id, f, local_dir=str(to))
        print("fetched", p)
    r = validate_onnx(to / "policy.onnx")
    print(r.text())
    m = json.loads((to / "manifest.json").read_text())
    print(f"{m['name']}: {m['description']}\n  twist = {m['command']['twist']}\n  idle  = {m['command']['idle']}\n  entry_pose = {m['entry_pose']}, kind = {m['kind']}")
    sys.exit(0 if r.ok else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate", help="check an ONNX against the daemon contract"); v.add_argument("onnx"); v.set_defaults(fn=cmd_validate)
    p = sub.add_parser("publish", help="validate + build manifest/card + upload")
    p.add_argument("onnx"); p.add_argument("--name", required=True, help="slug; repo = <namespace>/microduck-<name>")
    p.add_argument("--namespace", required=True); p.add_argument("--manifest", required=True, help="json with the user-provided manifest fields")
    p.add_argument("--card-extra", default=None, help="markdown: What it does / Command notes / Known limits / Try it sections")
    p.add_argument("--media", default=None, help="mp4/gif uploaded as media/preview.<ext>")
    p.add_argument("--extra", action="append", default=None, help="optional small text file uploaded at its basename, e.g. control.py (repeatable)")
    p.add_argument("--public", action="store_true"); p.add_argument("--dry-run", action="store_true")
    p.add_argument("--build-dir", default=None, help="where manifest.json/README.md are written (default next to the onnx)")
    p.set_defaults(fn=cmd_publish)
    f = sub.add_parser("fetch", help="download policy.onnx + manifest.json and re-validate"); f.add_argument("repo_id"); f.add_argument("--to", required=True); f.set_defaults(fn=cmd_fetch)
    a = ap.parse_args(); a.fn(a)


if __name__ == "__main__":
    main()

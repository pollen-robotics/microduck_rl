"""Checkpoint uploader run inside an HF Job.

Watches `logs/rsl_rl/**/model_*.pt` (with the run's params and `provenance.json`) and uploads new/updated files to the
target HF Model repo. Run as `python -m mjlab_microduck.hf_uploader` from the job bootstrap
(a module, because the job's tarball may be a dependent repo with no scripts/ of ours), with
auth coming from the HF_TOKEN secret injected by `hf jobs run`. `scripts/hf/uploader.py` still
runs it by path.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, CommitOperationAdd


def _watched(root: Path) -> list[Path]:
    """The checkpoints, the dumped configs and the provenance `publish --run` needs."""
    files = list(root.glob("**/model_*.pt"))
    files += root.glob("**/params/*.yaml")
    files += root.glob("**/params/*.json")
    files += root.glob("**/provenance.json")
    return files


def main() -> int:
    repo_id = os.environ.get("CKPT_REPO")
    if not repo_id:
        print("[uploader] CKPT_REPO not set, exiting", flush=True)
        return 1

    poll_interval = float(os.environ.get("CKPT_POLL_INTERVAL", "60"))
    root = Path(os.environ.get("CKPT_ROOT", "logs/rsl_rl"))

    one_shot = os.environ.get("CKPT_ONE_SHOT") == "1"

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
    mode = "one-shot" if one_shot else f"every {poll_interval}s"
    print(f"[uploader] watching {root} -> {repo_id} ({mode})", flush=True)

    sent: dict[Path, float] = {}
    while True:
        try:
            files = _watched(root)

            to_upload: list[CommitOperationAdd] = []
            for f in files:
                try:
                    mtime = f.stat().st_mtime
                except FileNotFoundError:
                    continue
                if sent.get(f) == mtime:
                    continue
                # use path-in-repo relative to logs/rsl_rl so the repo mirrors run dirs
                rel = f.relative_to(root)
                to_upload.append(
                    CommitOperationAdd(path_in_repo=str(rel), path_or_fileobj=str(f))
                )
                sent[f] = mtime

            if to_upload:
                msg = f"upload {len(to_upload)} file(s)"
                api.create_commit(
                    repo_id=repo_id,
                    repo_type="model",
                    operations=to_upload,
                    commit_message=msg,
                )
                print(f"[uploader] pushed {len(to_upload)} file(s)", flush=True)
        except Exception as e:
            print(f"[uploader] error: {e}", flush=True)

        if one_shot:
            return 0
        time.sleep(poll_interval)


if __name__ == "__main__":
    sys.exit(main())

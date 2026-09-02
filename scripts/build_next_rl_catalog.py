"""Generate the immutable Next RL inventory from a runtime policy checkout."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

from mjlab_microduck.next_rl.artifacts import atomic_write_json, canonical_json
from mjlab_microduck.next_rl.schema import PolicyContract


SHIPPED = {
    "standing": "alpha_stand.onnx",
    "walking": "alpha_walking.onnx",
    "sitstand": "alpha_sitstand.onnx",
    "ground-pick": "alpha_ground_pick.onnx",
    "kick-left": "ball_kick_left.onnx",
    "kick-right": "ball_kick_right.onnx",
    "roller": "roller.onnx",
    "roller-crouch": "roller_crouch.onnx",
    "roulade": "roulade.onnx",
}

ALIASES = {
    "standing": ("stand",),
    "walking": ("walk",),
    "sitstand": ("sit-stand",),
    "ground-pick": ("ground-pickup",),
    "kick-left": ("left-kick",),
    "kick-right": ("right-kick",),
    "roulade": ("forward-roll",),
}


def _git(runtime_repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(runtime_repo), *args],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise ValueError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout.strip()


def _git_bytes(runtime_repo: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(runtime_repo), *args],
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise ValueError(result.stderr.decode(errors="replace").strip() or f"git {' '.join(args)} failed")
    return result.stdout


def _runtime_repository(runtime_repo: Path) -> str:
    for remote in ("upstream", "origin"):
        result = subprocess.run(
            ["git", "-C", str(runtime_repo), "remote", "get-url", remote],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    return str(runtime_repo.resolve())


def build_catalog(runtime_repo: Path) -> dict[str, object]:
    """Read only tracked runtime ONNX files and return their evidence records."""
    runtime_repo = runtime_repo.resolve()
    runtime_commit = _git(runtime_repo, "rev-parse", "HEAD")
    repository = _runtime_repository(runtime_repo)
    readme = _git_bytes(runtime_repo, "show", f"{runtime_commit}:policies/README.md")
    approval_sha256 = hashlib.sha256(readme).hexdigest()
    records: list[dict[str, object]] = []
    for capability_id, filename in SHIPPED.items():
        relative_path = f"policies/{filename}"
        digest = hashlib.sha256(_git_bytes(runtime_repo, "show", f"{runtime_commit}:{relative_path}")).hexdigest()
        records.append(
            {
                "id": capability_id,
                "version": "1.0.0",
                "aliases": list(ALIASES.get(capability_id, ())),
                "robot_model": "microduck",
                "contract": PolicyContract.microduck().as_dict(),
                "status": "validated",
                "policy": {"path": relative_path, "kind": "onnx", "sha256": digest},
                "evaluation": {
                    "kind": "legacy_runtime_shipped",
                    "policy_sha256": digest,
                    "runtime_repository": repository,
                    "runtime_commit": runtime_commit,
                    "approval_provenance": "policies/README.md",
                    "metadata": {
                        "approval_sha256": approval_sha256,
                        "evidence_scope": "named_capability_only",
                    },
                },
            }
        )
    return {"capabilities": records}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-repo", type=Path, required=True, help="Sibling runtime checkout to inventory")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "src/mjlab_microduck/next_rl/catalog.json",
        help="Catalog path to write or check",
    )
    parser.add_argument("--check", action="store_true", help="Fail when the committed catalog differs")
    args = parser.parse_args()

    try:
        generated = build_catalog(args.runtime_repo)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    if args.check:
        try:
            committed = args.output.read_text(encoding="utf-8")
        except FileNotFoundError:
            print(f"catalog is missing: {args.output}")
            return 1
        if committed != canonical_json(generated):
            print(f"catalog differs from generated runtime evidence: {args.output}")
            return 1
        print(f"catalog matches runtime evidence: {args.output}")
        return 0
    atomic_write_json(args.output, generated)
    print(f"wrote catalog: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

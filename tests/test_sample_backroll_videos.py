from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "sample_backroll_videos.py"
SPEC = importlib.util.spec_from_file_location("sample_backroll_videos", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
SAMPLER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SAMPLER
SPEC.loader.exec_module(SAMPLER)


def test_newest_checkpoint_ignores_non_models_and_uses_iteration(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    root.mkdir()
    (root / "model_50.pt").write_bytes(b"a")
    (root / "model_100.pt").write_bytes(b"b")
    (root / "checkpoint.pt").write_bytes(b"ignored")

    assert SAMPLER._newest_checkpoint(root) == root / "model_100.pt"


def test_output_paths_include_iteration_and_checkpoint_hash(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model_50.pt"
    checkpoint.write_bytes(b"checkpoint")

    video, audit = SAMPLER._output_paths(tmp_path, checkpoint, "a" * 64)

    assert "checkpoint-000050-aaaaaaaaaaaa" in video.name
    assert video.suffix == ".mp4"
    assert audit.parent.name == "evaluations"
    assert audit.suffix == ".json"

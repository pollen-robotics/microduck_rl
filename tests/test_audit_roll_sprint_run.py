from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "audit_roll_sprint_run.py"
)
SPEC = importlib.util.spec_from_file_location("audit_roll_sprint_run", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _checkpoint(root: Path, iteration: int) -> Path:
    path = root / f"model_{iteration}.pt"
    path.write_bytes(f"checkpoint-{iteration}".encode())
    return path


def _report(
    checkpoint: Path,
    *,
    finish_count: int = 0,
    frontier_m: float = 1.25,
    promotion_pass: bool = False,
) -> dict[str, object]:
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    iteration = int(checkpoint.stem.rsplit("_", 1)[-1])
    finish_times = [20.0 + index for index in range(finish_count)]
    return {
        "schema_version": 8,
        "task": "Mjlab-Roll-Sprint-Flat-MicroDuck",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": digest,
        "checkpoint_iteration": iteration,
        "num_envs": 4,
        "duration_s": 40.0,
        "target_distance_m": 10.0,
        "target_distance_reach_count": finish_count,
        "mean_credited_forward_frontier_m": frontier_m,
        "slowest_time_to_valid_10m_s": max(finish_times, default=None),
        "per_robot": [
            {
                "target_10m_pass": index < finish_count,
                "time_to_valid_10m_s": (
                    finish_times[index] if index < finish_count else None
                ),
            }
            for index in range(4)
        ],
        "canonical_race_alignment": {"alignment_pass": True},
        "recovery_battery": {"overall_pass": promotion_pass},
        "promotion_pass": promotion_pass,
    }


def _completed(returncode: int = 0, stdout: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], returncode, stdout=stdout)


def test_find_checkpoints_uses_numeric_order_and_one_run_only(tmp_path: Path) -> None:
    for iteration in (100, 2, 10):
        _checkpoint(tmp_path, iteration)
    (tmp_path / "model_latest.pt").write_bytes(b"ignored")
    nested = tmp_path / "nested"
    nested.mkdir()
    _checkpoint(nested, 1)

    checkpoints = MODULE.find_checkpoints(tmp_path)

    assert [path.name for path in checkpoints] == [
        "model_2.pt",
        "model_10.pt",
        "model_100.pt",
    ]


def test_cli_defaults_to_cpu_and_accepts_explicit_output_directories(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    defaults = MODULE._parse_args([str(run_dir)])
    explicit = MODULE._parse_args(
        [
            str(run_dir),
            "--evaluation-dir",
            str(tmp_path / "shared-evaluations"),
            "--champion-dir",
            str(tmp_path / "shared-champion"),
            "--device",
            "cuda:1",
        ]
    )

    assert defaults.device == "cpu"
    assert defaults.evaluation_dir == run_dir / "checkpoint-audits"
    assert defaults.champion_dir == run_dir / "checkpoint-champion"
    assert explicit.device == "cuda:1"
    assert explicit.evaluation_dir == (tmp_path / "shared-evaluations").resolve()
    assert explicit.champion_dir == (tmp_path / "shared-champion").resolve()


def test_matching_audit_is_identity_based_and_reuses_failed_result(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    evaluation_dir = tmp_path / "evaluations"
    run_dir.mkdir()
    evaluation_dir.mkdir()
    checkpoint = _checkpoint(run_dir, 7)
    identity = MODULE.checkpoint_identity(checkpoint)
    report = _report(checkpoint, finish_count=0, promotion_pass=False)
    custom_name = evaluation_dir / "manually-front-loaded.json"
    custom_name.write_text(json.dumps(report), encoding="utf-8")

    match = MODULE.load_matching_audit(evaluation_dir, identity)

    assert match is not None
    assert match[0] == custom_name.resolve()
    assert match[1]["promotion_pass"] is False

    report["checkpoint_sha256"] = "0" * 64
    custom_name.write_text(json.dumps(report), encoding="utf-8")
    assert MODULE.load_matching_audit(evaluation_dir, identity) is None


def test_series_reuses_existing_audit_and_invokes_selector_after_each(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    evaluation_dir = tmp_path / "evaluations"
    champion_dir = tmp_path / "champion"
    run_dir.mkdir()
    evaluation_dir.mkdir()
    checkpoints = {iteration: _checkpoint(run_dir, iteration) for iteration in (10, 2, 1)}
    reused = evaluation_dir / "preloaded-model-1.json"
    reused.write_text(json.dumps(_report(checkpoints[1])), encoding="utf-8")
    evaluator_iterations: list[int] = []
    selector_calls = 0

    def runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        nonlocal selector_calls
        if Path(command[1]) == MODULE.EVALUATOR:
            checkpoint = Path(command[2])
            evaluator_iterations.append(int(checkpoint.stem.rsplit("_", 1)[-1]))
            output = Path(command[command.index("--output") + 1])
            output.write_text(json.dumps(_report(checkpoint)), encoding="utf-8")
            return _completed()
        assert Path(command[1]) == MODULE.SELECTOR
        selector_calls += 1
        champion_dir.mkdir(exist_ok=True)
        (champion_dir / "champion.json").write_text(
            json.dumps(
                {
                    "evaluation_schema_version": 8,
                    "target_distance_reach_count": 0,
                }
            ),
            encoding="utf-8",
        )
        return _completed()

    results = MODULE.audit_run(
        run_dir=run_dir,
        evaluation_dir=evaluation_dir,
        champion_dir=champion_dir,
        device="cpu",
        parent_frontier_m=None,
        stop_on_promoted_four_of_four=False,
        runner=runner,
    )

    assert [result.checkpoint.iteration for result in results] == [1, 2, 10]
    assert [result.reused for result in results] == [True, False, False]
    assert evaluator_iterations == [2, 10]
    assert selector_calls == 3


def test_stop_gate_reads_successful_selector_manifest_not_current_audit(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    evaluation_dir = tmp_path / "evaluations"
    champion_dir = tmp_path / "champion"
    run_dir.mkdir()
    checkpoints = {iteration: _checkpoint(run_dir, iteration) for iteration in (1, 2, 3)}
    evaluator_iterations: list[int] = []
    selector_calls = 0

    def runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        nonlocal selector_calls
        if Path(command[1]) == MODULE.EVALUATOR:
            checkpoint = Path(command[2])
            iteration = int(checkpoint.stem.rsplit("_", 1)[-1])
            evaluator_iterations.append(iteration)
            output = Path(command[command.index("--output") + 1])
            output.write_text(
                json.dumps(
                    _report(
                        checkpoint,
                        finish_count=4 if iteration == 1 else 0,
                        promotion_pass=iteration == 1,
                    )
                ),
                encoding="utf-8",
            )
            return _completed()
        selector_calls += 1
        champion_dir.mkdir(exist_ok=True)
        (champion_dir / "champion.json").write_text(
            json.dumps(
                {
                    "evaluation_schema_version": 8,
                    "target_distance_reach_count": 4 if selector_calls == 2 else 3,
                }
            ),
            encoding="utf-8",
        )
        return _completed()

    results = MODULE.audit_run(
        run_dir=run_dir,
        evaluation_dir=evaluation_dir,
        champion_dir=champion_dir,
        device="cpu",
        parent_frontier_m=None,
        stop_on_promoted_four_of_four=True,
        runner=runner,
    )

    assert [result.checkpoint.iteration for result in results] == [1, 2]
    assert evaluator_iterations == [1, 2]
    assert selector_calls == 2
    assert checkpoints[3].is_file()

"""Behavioural contracts for reproducible Next RL experiment records."""

from __future__ import annotations

import json
import multiprocessing
import threading
from queue import Empty
from typing import Any

import pytest

from mjlab_microduck.next_rl.capabilities import Disposition, PlanDecision
from mjlab_microduck.next_rl.experiments import (
    DuplicateExperimentError,
    ExperimentStore,
    build_experiment_manifest,
    experiment_fingerprint,
)
from mjlab_microduck.next_rl.schema import (
    ArtifactRef,
    ExperimentManifest,
    MetricThreshold,
    PolicyContract,
    SkillSpec,
)


def manifest(**overrides: object) -> ExperimentManifest:
    value: dict[str, object] = {
        "skill_id": "standing",
        "spec_version": "1.0.0",
        "task_id": "Mjlab-Standing-Flat-MicroDuck",
        "contract": PolicyContract.microduck().as_dict(),
        "code_digest": "a" * 64,
        "dirty_patch_digest": "b" * 64,
        "seed": 7,
        "parent_policy_digest": "c" * 64,
        "runner_id": "nitro-a100",
        "environment_config": {"scene": {"num_envs": 1024}},
        "agent_config": {"max_iterations": 1000},
        "status": "planned",
    }
    value.update(overrides)
    return ExperimentManifest.from_dict(value)


def _compete_terminal_transition(
    root: str,
    fingerprint: str,
    status: str,
    start: Any,
    results: Any,
) -> None:
    """Attempt one terminal transition simultaneously from a separate process."""
    start.wait()
    try:
        ExperimentStore(root).update_status(fingerprint, status)
    except ValueError:
        results.put(("rejected", status))
    else:
        results.put(("updated", status))


def test_manifest_builder_records_planning_provenance_and_parent_artifact_digest():
    spec = SkillSpec(
        id="standing",
        version="1.0.0",
        description="Remain upright.",
        contract=PolicyContract.microduck(),
        metrics=(MetricThreshold("falls", "count", "maximum", 0),),
        training_seeds=(7,),
        evaluation_seeds=(8,),
    )
    parent = ArtifactRef("policies/standing.onnx", "onnx", "c" * 64)

    result = build_experiment_manifest(
        spec,
        PlanDecision(Disposition.WARM_START, "Compatible parent selected."),
        task_id="Mjlab-Standing-Flat-MicroDuck",
        code_digest="a" * 64,
        seed=7,
        runner_id="nitro-a100",
        environment_config={"scene": {"num_envs": 1024}},
        agent_config={"max_iterations": 1000},
        parent_policy=parent,
    )

    assert result.parent_policy_digest == parent.sha256
    assert result.metadata["plan"] == {
        "disposition": "warm_start",
        "reason": "Compatible parent selected.",
    }


def test_fingerprint_ignores_timestamp_and_output_path():
    left = manifest(created_at="one", output_dir="a")
    right = manifest(created_at="two", output_dir="b")

    assert experiment_fingerprint(left) == experiment_fingerprint(right)


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("seed", 8),
        ("spec_version", "1.0.1"),
        ("code_digest", "d" * 64),
        ("parent_policy_digest", "e" * 64),
        ("runner_id", "local"),
    ],
)
def test_fingerprint_changes_for_learning_inputs(field: str, changed: object):
    assert experiment_fingerprint(manifest()) != experiment_fingerprint(manifest(**{field: changed}))


def test_fingerprint_ignores_operational_values_and_credentials_in_configs():
    left = manifest(
        environment_config={"scene": {"num_envs": 1024}, "output_dir": "left", "hostname": "one"},
        agent_config={"max_iterations": 1000, "token": "one", "pid": 1},
    )
    right = manifest(
        environment_config={"scene": {"num_envs": 1024}, "output_dir": "right", "hostname": "two"},
        agent_config={"max_iterations": 1000, "token": "two", "pid": 2},
    )

    assert experiment_fingerprint(left) == experiment_fingerprint(right)


def test_credential_structures_are_redacted_without_collapsing_learning_settings(tmp_path):
    first = manifest(
        environment_config={"auth": {"token": "environment-secret"}, "tokenizer": "v1"},
        agent_config={
            "transport": {"authorization": "Bearer agent-secret"},
            "token_budget": 128,
        },
        metadata={"remote": {"credentials": {"password": "provenance-secret"}}},
    )
    second = manifest(
        environment_config={"auth": {"token": "changed-secret"}, "tokenizer": "v2"},
        agent_config={
            "transport": {"authorization": "Bearer changed-secret"},
            "token_budget": 256,
        },
        metadata={"remote": {"credentials": {"password": "changed-provenance-secret"}}},
    )

    store = ExperimentStore(tmp_path)
    fingerprint = store.create(first)
    saved = (tmp_path / fingerprint / "manifest.json").read_text()

    assert experiment_fingerprint(first) != experiment_fingerprint(second)
    assert all(secret not in saved for secret in ("environment-secret", "agent-secret", "provenance-secret"))
    assert "tokenizer" in saved
    assert "token_budget" in saved


def test_duplicate_active_experiment_is_rejected(tmp_path):
    store = ExperimentStore(tmp_path)
    record = manifest(status="running")
    store.create(record)

    with pytest.raises(DuplicateExperimentError):
        store.reserve(experiment_fingerprint(record))


def test_reservation_claims_a_planned_experiment_once(tmp_path):
    store = ExperimentStore(tmp_path)
    fingerprint = store.create(manifest())

    reservation = store.reserve(fingerprint)

    assert reservation.action == "reserved"
    assert json.loads((tmp_path / fingerprint / "status.json").read_text()) == {
        "history": [{"status": "planned"}, {"status": "pending"}],
        "status": "pending",
    }
    with pytest.raises(DuplicateExperimentError):
        store.reserve(fingerprint)


def test_concurrent_reservations_allow_exactly_one_start_owner(tmp_path):
    store = ExperimentStore(tmp_path)
    fingerprint = store.create(manifest())
    start = threading.Barrier(3)
    outcomes: list[str] = []

    def reserve(owner: str) -> None:
        start.wait()
        try:
            store.reserve(fingerprint, owner=owner)
        except DuplicateExperimentError:
            outcomes.append("blocked")
        else:
            outcomes.append("reserved")

    workers = [
        threading.Thread(target=reserve, args=("starter-a",)),
        threading.Thread(target=reserve, args=("starter-b",)),
    ]
    for worker in workers:
        worker.start()
    start.wait()
    for worker in workers:
        worker.join(timeout=5)
        assert not worker.is_alive()

    assert sorted(outcomes) == ["blocked", "reserved"]
    assert store.status(fingerprint)["status"] == "pending"


def test_only_the_reservation_owner_can_release_a_failed_start(tmp_path):
    store = ExperimentStore(tmp_path)
    fingerprint = store.create(manifest())
    store.reserve(fingerprint, owner="starter-a")

    with pytest.raises(DuplicateExperimentError, match="owner"):
        store.release(fingerprint, owner="starter-b")

    assert store.status(fingerprint)["status"] == "pending"
    store.release(fingerprint, owner="starter-a")
    assert store.status(fingerprint)["status"] == "planned"
    assert store.reserve(fingerprint, owner="starter-b").action == "reserved"


def test_same_owner_can_retry_an_uncertain_start_but_another_owner_is_blocked(tmp_path):
    store = ExperimentStore(tmp_path)
    fingerprint = store.create(manifest())
    store.reserve(fingerprint, owner="starter-a")

    assert store.reserve(fingerprint, owner="starter-a").action == "confirmed"
    with pytest.raises(DuplicateExperimentError, match="reserved"):
        store.reserve(fingerprint, owner="starter-b")

    store.confirm(fingerprint, owner="starter-a")
    with pytest.raises(DuplicateExperimentError, match="pending"):
        store.reserve(fingerprint, owner="starter-a")


@pytest.mark.parametrize(("status", "action"), [("failed", "retry"), ("interrupted", "resume")])
def test_terminal_unsuccessful_experiment_returns_an_explicit_next_action(tmp_path, status: str, action: str):
    store = ExperimentStore(tmp_path)
    fingerprint = store.create(manifest(status=status))

    assert store.reserve(fingerprint, owner="retry-owner").action == action
    assert store.status(fingerprint)["status"] == "pending"
    store.release(fingerprint, owner="retry-owner")
    assert store.status(fingerprint)["status"] == status


def test_lifecycle_is_linear_and_status_history_is_atomic(tmp_path):
    store = ExperimentStore(tmp_path)
    fingerprint = store.create(manifest())
    store.reserve(fingerprint)

    store.update_status(fingerprint, "running")
    store.update_status(fingerprint, "succeeded")

    assert json.loads((tmp_path / fingerprint / "status.json").read_text()) == {
        "history": [
            {"status": "planned"},
            {"status": "pending"},
            {"status": "running"},
            {"status": "succeeded"},
        ],
        "status": "succeeded",
    }
    with pytest.raises(ValueError, match="cannot transition"):
        store.update_status(fingerprint, "running")


def test_confirmed_remote_start_clears_claim_so_a_failed_run_can_be_retried(tmp_path):
    store = ExperimentStore(tmp_path)
    fingerprint = store.create(manifest())
    store.reserve(fingerprint, owner="first-owner")
    store.update_status(fingerprint, "running")
    store.update_status(fingerprint, "failed")

    assert store.reserve(fingerprint, owner="retry-owner").action == "retry"


def test_simultaneous_terminal_transitions_allow_exactly_one_and_preserve_history(tmp_path):
    store = ExperimentStore(tmp_path)
    fingerprint = store.create(manifest())
    store.reserve(fingerprint)
    store.update_status(fingerprint, "running")

    context = multiprocessing.get_context()
    start = context.Barrier(9)
    results = context.Queue()
    workers = [
        context.Process(
            target=_compete_terminal_transition,
            args=(str(tmp_path), fingerprint, "succeeded" if index % 2 else "failed", start, results),
        )
        for index in range(8)
    ]
    for worker in workers:
        worker.start()
    start.wait(timeout=10)
    for worker in workers:
        worker.join(timeout=10)
        assert worker.exitcode == 0
    try:
        outcomes = [results.get(timeout=1) for _ in workers]
    except Empty as error:
        raise AssertionError("every competing process must report an outcome") from error

    assert sum(outcome == "updated" for outcome, _ in outcomes) == 1
    status = json.loads((tmp_path / fingerprint / "status.json").read_text())
    assert len(status["history"]) == 4
    assert status["history"][-1] == {"status": status["status"]}


def test_create_rejects_mutation_of_immutable_learning_inputs(tmp_path):
    store = ExperimentStore(tmp_path)
    fingerprint = store.create(manifest())

    changed = manifest(seed=8)
    with pytest.raises(ValueError, match="fingerprint"):
        store.create(changed, fingerprint=fingerprint)


def test_create_keeps_plan_provenance_in_the_immutable_record(tmp_path):
    store = ExperimentStore(tmp_path)
    record = manifest(metadata={"plan": {"disposition": "train_new", "reason": "No match."}})

    fingerprint = store.create(record)

    saved = json.loads((tmp_path / fingerprint / "manifest.json").read_text())
    assert saved["provenance"] == {"plan": {"disposition": "train_new", "reason": "No match."}}

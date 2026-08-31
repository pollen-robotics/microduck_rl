from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

GENERATOR = Path(__file__).with_name("generate-task8-handoff.py")
SPEC = importlib.util.spec_from_file_location("generate_task8_handoff", GENERATOR)
assert SPEC is not None and SPEC.loader is not None
handoff = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(handoff)


def test_complete_tree_comparison_rejects_any_path_or_byte_difference(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    for root in (first, second):
        (root / "schemas").mkdir(parents=True)
        (root / "handoff.json").write_bytes(b'{"schema":"HANDOFF"}')
        (root / "schemas/api.yaml").write_bytes(b"openapi: 3.1.0\n")

    handoff.assert_identical_trees(first, second)

    (second / "schemas/api.yaml").write_bytes(b"openapi: 3.0.3\n")
    with pytest.raises(AssertionError, match="schemas/api.yaml"):
        handoff.assert_identical_trees(first, second)

    (second / "schemas/api.yaml").write_bytes(b"openapi: 3.1.0\n")
    (second / "unexpected.txt").write_bytes(b"unexpected")
    with pytest.raises(AssertionError, match="unexpected.txt"):
        handoff.assert_identical_trees(first, second)


def test_publish_reproducible_tree_materializes_only_a_complete_matching_tree(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    output = tmp_path / "published"
    for root in (first, second):
        (root / "nested").mkdir(parents=True)
        (root / "handoff.json").write_bytes(b'{"schema":"HANDOFF"}')
        (root / "nested/evidence.json").write_bytes(b'{"reaped":true}')

    handoff.publish_reproducible_tree(first, second, output)
    handoff.assert_identical_trees(first, output)

    (second / "nested/evidence.json").write_bytes(b'{"reaped":false}')
    with pytest.raises(AssertionError, match="nested/evidence.json"):
        handoff.publish_reproducible_tree(first, second, output)
    handoff.assert_identical_trees(first, output)


def test_all_authoritative_assets_are_copied_and_verified_byte_exactly(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    output = tmp_path / "handoff"
    source_bytes = {
        "microduck-policy-bundle-v1.schema.json": b'{"policy":1}\n',
        "microduck-simulator-api-v1.openapi.yaml": b"openapi: 3.1.0\n",
        "microduck-v1-portability-fixtures.json": b'{"fixtures":1}\n',
    }
    (repository / "schemas").mkdir(parents=True)
    for name, content in source_bytes.items():
        (repository / "schemas" / name).write_bytes(content)

    handoff.copy_authoritative_assets(repository, output)
    handoff.assert_authoritative_assets(repository, output)
    assert {
        path.name: path.read_bytes() for path in (output / "schemas").iterdir()
    } == source_bytes

    (output / "schemas/microduck-simulator-api-v1.openapi.yaml").write_bytes(
        b"openapi: 3.0.3\n"
    )
    with pytest.raises(AssertionError, match="microduck-simulator-api-v1.openapi.yaml"):
        handoff.assert_authoritative_assets(repository, output)


def test_distribution_handoff_gate_accepts_cleared_and_rejects_development_only_before_output(
    tmp_path: Path,
) -> None:
    """Removing the gate would allow a development-only bundle to create handoff bytes."""
    cleared = handoff._write_verified_bundle(tmp_path / "cleared")
    development_only = handoff._write_verified_bundle(
        tmp_path / "development-only", model_license_status="DEVELOPMENT_ONLY"
    )
    destination = tmp_path / "handoff"

    handoff.require_distribution_cleared(cleared)
    with pytest.raises(
        ValueError, match="model assets are not cleared for distribution handoff"
    ):
        handoff.require_distribution_cleared(development_only)

    assert not destination.exists()


def test_manifest_records_runtime_revision_process_transcript_and_every_file(
    tmp_path: Path,
) -> None:
    output = tmp_path / "handoff"
    (output / "transcripts").mkdir(parents=True)
    (output / "schemas").mkdir()
    for name in handoff.AUTHORITATIVE_ASSETS:
        (output / "schemas" / name).write_bytes(name.encode())
    transcript = output / "transcripts/process-child-exit-reap.json"
    transcript.write_bytes(b'{"schema":"PROCESS"}')
    (output / "fixture.bin").write_bytes(b"fixture")

    manifest = handoff.build_handoff_manifest(
        output,
        source_commit="a" * 40,
        runtime_revision="mjlab-microduck@0.1.0+sha256:" + "b" * 64,
        bundle_digest="sha256:" + "c" * 64,
    )

    assert manifest["runtimeRevision"] == (
        "mjlab-microduck@0.1.0+sha256:" + "b" * 64
    )
    assert manifest["processFailureTranscript"] == (
        "transcripts/process-child-exit-reap.json"
    )
    assert set(manifest["files"]) == {
        "fixture.bin",
        "schemas/microduck-policy-bundle-v1.schema.json",
        "schemas/microduck-simulator-api-v1.openapi.yaml",
        "schemas/microduck-v1-portability-fixtures.json",
        "transcripts/process-child-exit-reap.json",
    }
    assert manifest["files"]["fixture.bin"] == {
        "bytes": 7,
        "sha256": "sha256:" + handoff.hashlib.sha256(b"fixture").hexdigest(),
    }


def test_manifest_refuses_to_reference_missing_required_release_evidence(
    tmp_path: Path,
) -> None:
    output = tmp_path / "handoff"
    (output / "schemas").mkdir(parents=True)
    for name in handoff.AUTHORITATIVE_ASSETS:
        (output / "schemas" / name).write_bytes(name.encode())

    with pytest.raises(AssertionError, match="process failure transcript"):
        handoff.build_handoff_manifest(
            output,
            source_commit="a" * 40,
            runtime_revision="mjlab-microduck@0.1.0+sha256:" + "b" * 64,
            bundle_digest="sha256:" + "c" * 64,
        )

    (output / "transcripts").mkdir()
    (output / handoff.PROCESS_FAILURE_TRANSCRIPT).write_bytes(b"{}")
    (output / "schemas/microduck-simulator-api-v1.openapi.yaml").unlink()
    with pytest.raises(AssertionError, match="authoritative asset"):
        handoff.build_handoff_manifest(
            output,
            source_commit="a" * 40,
            runtime_revision="mjlab-microduck@0.1.0+sha256:" + "b" * 64,
            bundle_digest="sha256:" + "c" * 64,
        )


@pytest.mark.parametrize(
    "leaked",
    (
        {"authorization": "Bearer handoff-test-token"},
        {"bundlePath": "/tmp/private/bundle"},
        {"childPid": 12345},
    ),
)
def test_sanitized_transcript_validator_rejects_token_path_and_pid_leaks(
    leaked: dict[str, object],
) -> None:
    transcript = {
        "schema": "MICRODUCK_ROM_SANITIZED_TRANSCRIPT_V1",
        "authorization": "Bearer <redacted>",
        "processFailure": leaked,
    }
    with pytest.raises(AssertionError):
        handoff.assert_sanitized_transcript(
            transcript,
            forbidden_tokens=("handoff-test-token",),
        )


def test_sanitized_process_transcript_accepts_only_stable_containment_evidence() -> None:
    transcript = {
        "schema": "MICRODUCK_ROM_SANITIZED_TRANSCRIPT_V1",
        "sourceCommit": "a" * 40,
        "authorization": "Bearer <redacted>",
        "processFailure": {
            "startAcknowledged": True,
            "exactChildIdentityCaptured": True,
            "signal": "SIGKILL",
            "exitObservedOnOwnedHandle": True,
            "reapConfirmed": True,
            "durableState": "FAILED",
            "stopReason": "RUNTIME_UNRESPONSIVE",
        },
        "freshTask": {
            "generationAdvanced": True,
            "terminalState": "TIMED_OUT",
        },
    }
    handoff.assert_sanitized_transcript(
        json.loads(json.dumps(transcript)),
        forbidden_tokens=("handoff-test-token",),
    )


def test_process_failure_transcript_reports_reap_and_fresh_generation_without_pid() -> None:
    transcript = handoff.build_process_failure_transcript(
        source_commit="a" * 40,
        runtime_revision="mjlab-microduck@0.1.0+sha256:" + "b" * 64,
        failed_generation=4,
        failure_trace=(
            "OPERATION_FAILED",
            "QUARANTINED",
            "CHILD_REAPED",
            "NO_CHILD",
        ),
        failure_events=[
            {"sequence": 0, "eventType": "TASK_VALIDATING", "payload": {}},
            {"sequence": 1, "eventType": "TASK_STARTED", "payload": {}},
            {
                "sequence": 2,
                "eventType": "TASK_FAILED",
                "payload": {"code": "RUNTIME_UNRESPONSIVE"},
            },
        ],
        fresh_generation=5,
        fresh_events=[
            {"sequence": 0, "eventType": "TASK_VALIDATING", "payload": {}},
            {"sequence": 1, "eventType": "TASK_STARTED", "payload": {}},
            {"sequence": 2, "eventType": "TASK_TIMED_OUT", "payload": {}},
        ],
    )

    assert transcript["processFailure"]["startAcknowledged"] is True
    assert transcript["processFailure"]["reapConfirmed"] is True
    assert transcript["processFailure"]["durableState"] == "FAILED"
    assert transcript["freshTask"] == {
        "generation": 5,
        "generationAdvanced": True,
        "acceptedOnlyAfterReap": True,
        "events": [
            {"sequence": 0, "eventType": "TASK_VALIDATING", "payload": {}},
            {"sequence": 1, "eventType": "TASK_STARTED", "payload": {}},
            {"sequence": 2, "eventType": "TASK_TIMED_OUT", "payload": {}},
        ],
        "terminalState": "TIMED_OUT",
        "stopReason": "LEASE_EXPIRED",
    }
    serialized = json.dumps(transcript, sort_keys=True)
    assert "pid" not in serialized.lower()
    assert "/tmp/" not in serialized


def test_process_failure_generation_uses_max_lease_before_forced_kill() -> None:
    """The forced-kill proof must not race the ordinary child lease deadman."""
    assert handoff.FORCED_KILL_LEASE_MS == 5_000
    assert handoff.SHORT_LEASE_MS == 200


def test_exact_head_guard_rejects_tracked_worktree_and_index_drift(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(
        ["git", "config", "user.email", "handoff@example.invalid"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Handoff Test"],
        cwd=repository,
        check=True,
    )
    tracked = repository / "runtime.py"
    tracked.write_bytes(b"governed = True\n")
    subprocess.run(["git", "add", "runtime.py"], cwd=repository, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "fixture"], cwd=repository, check=True
    )

    commit = handoff.exact_head_commit(repository)
    assert len(commit) == 40

    tracked.write_bytes(b"governed = False\n")
    with pytest.raises(AssertionError, match="tracked worktree"):
        handoff.exact_head_commit(repository)

    subprocess.run(["git", "add", "runtime.py"], cwd=repository, check=True)
    with pytest.raises(AssertionError, match="index"):
        handoff.exact_head_commit(repository)

    tracked.write_bytes(b"governed = True\n")
    subprocess.run(["git", "add", "runtime.py"], cwd=repository, check=True)
    (repository / "unknown.py").write_bytes(b"not_committed = True\n")
    with pytest.raises(AssertionError, match="untracked"):
        handoff.exact_head_commit(repository)

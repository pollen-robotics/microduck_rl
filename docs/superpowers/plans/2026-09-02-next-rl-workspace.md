# Next RL Workspace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tested Microduck RL orchestration workspace that avoids duplicate training, creates reproducible Nitro experiments, evaluates exact ONNX artifacts, prepares human video review, and gates learned status on passing evidence plus approval.

**Architecture:** Add a dependency-light `mjlab_microduck.next_rl` package beside the existing training stack. Pure dataclasses and canonical JSON define capabilities, skill specifications, experiments, evaluation, review, and promotion; adapters reuse the current exporter and construct safe Nitro commands without changing the existing `train`/HF hooks.

**Tech Stack:** Python 3.12 standard library, pytest, existing MuJoCo/mjlab/RSL-RL/ONNX Runtime stack, system OpenSSH, JSON artifacts.

**Spec:** `docs/plans/2026-09-02-next-rl-design.md`

## Global Constraints

- Preserve the existing Microduck MJCF, BAM XL330 actuator model, backlash variants, 50 Hz control loop, 61-observation actor contract, 14-action output, normalized ONNX export, and RSL-RL PPO baseline.
- Native MuJoCo is the reference evaluation backend; MuJoCo Warp is the Nitro training backend.
- Do not change `train_cli.py`, `train_hook.py`, or the `train` console-script behavior.
- Do not start long training during workspace implementation; only the documented 64-environment, five-iteration smoke test is permitted.
- Only passing evaluation bound to an exact policy digest plus explicit human approval may produce `learned`.
- Existing shipped policies must be checked before new training; insufficient evidence means evaluate-first, not retrain-first.
- Publishing and robot deployment remain separate explicit actions.
- No password, token, private-key content, host-key bypass, or shell-composed training command may enter source, manifests, logs, or tests.

---

### Task 1: Canonical schemas and artifact I/O

**Files:**
- Create: `src/mjlab_microduck/next_rl/__init__.py`
- Create: `src/mjlab_microduck/next_rl/schema.py`
- Create: `src/mjlab_microduck/next_rl/artifacts.py`
- Test: `tests/test_next_rl_schema.py`

**Interfaces:**
- Produces: `PolicyContract`, `MetricThreshold`, `SkillSpec`, `ArtifactRef`, `EvaluationRef`, `Capability`, `ExperimentManifest`, `canonical_json`, `sha256_file`, `atomic_write_json`, and schema-specific `from_dict`/`as_dict` methods.
- Consumes: constants `MODEL_API`, `OBS_LEN`, `ACTION_LEN`, and `ROBOT` from `mjlab_microduck.publish.manifest`.

- [ ] **Step 1: Write failing schema tests**

```python
def test_default_contract_is_the_runtime_contract():
    assert PolicyContract.microduck().as_dict() == {
        "model_api": 1, "obs_len": 61, "action_len": 14,
        "robot_model": "microduck", "control_hz": 50,
    }

def test_threshold_requires_units_and_known_direction():
    with pytest.raises(SchemaError, match="direction"):
        MetricThreshold.from_dict({"name": "falls", "unit": "count", "direction": "equal", "limit": 0})

def test_skill_rejects_duplicate_metric_names():
    raw = skill_dict(metrics=[metric("falls"), metric("falls")])
    with pytest.raises(SchemaError, match="duplicate metric"):
        SkillSpec.from_dict(raw)

def test_training_and_eval_seeds_must_be_disjoint():
    raw = skill_dict(training_seeds=[1, 2], evaluation_seeds=[2, 3])
    with pytest.raises(SchemaError, match="overlap"):
        SkillSpec.from_dict(raw)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `uv run --with pytest pytest tests/test_next_rl_schema.py -q`

Expected: collection fails because `mjlab_microduck.next_rl` does not exist.

- [ ] **Step 3: Implement immutable schemas**

Use frozen dataclasses, explicit `SchemaError(ValueError)`, strict required-field validation, semantic versions matching `^[0-9]+\.[0-9]+\.[0-9]+$`, unique aliases/metrics, finite numeric thresholds, non-empty units, and disjoint training/evaluation seeds. Reject unknown enum values; retain forward-compatible optional metadata only under a `metadata` mapping.

- [ ] **Step 4: Add failing canonical-I/O tests**

```python
def test_canonical_json_is_order_independent():
    assert canonical_json({"b": 2, "a": 1}) == canonical_json({"a": 1, "b": 2})

def test_atomic_write_json_round_trips(tmp_path):
    path = tmp_path / "manifest.json"
    atomic_write_json(path, {"b": 2, "a": 1})
    assert json.loads(path.read_text()) == {"a": 1, "b": 2}

def test_sha256_file_detects_changed_content(tmp_path):
    path = tmp_path / "policy.onnx"
    path.write_bytes(b"one")
    first = sha256_file(path)
    path.write_bytes(b"two")
    assert sha256_file(path) != first
```

- [ ] **Step 5: Run tests and verify RED, then implement canonical I/O**

`canonical_json` uses UTF-8, sorted keys, compact separators, and rejects NaN. `atomic_write_json` writes a sibling temporary file, flushes/fsyncs it, and replaces the target. `sha256_file` streams fixed-size blocks.

- [ ] **Step 6: Run focused tests and commit**

Run: `uv run --with pytest pytest tests/test_next_rl_schema.py -q`

Commit: `feat(next-rl): add canonical workspace schemas`

---

### Task 2: Existing capability inventory and conservative planner

**Files:**
- Create: `src/mjlab_microduck/next_rl/capabilities.py`
- Create: `src/mjlab_microduck/next_rl/catalog.json`
- Create: `scripts/build_next_rl_catalog.py`
- Test: `tests/test_next_rl_capabilities.py`

**Interfaces:**
- Consumes: `Capability`, `SkillSpec`, `ArtifactRef`, and `EvaluationRef` from Task 1.
- Produces: `CapabilityInventory.load_builtin()`, `CapabilityInventory.resolve(query)`, `Disposition`, `PlanDecision`, and `plan_skill(spec, inventory, improve_reason=None)`.

- [ ] **Step 1: Write failing inventory tests**

```python
def test_builtin_catalog_contains_shipped_runtime_skills():
    inventory = CapabilityInventory.load_builtin()
    assert {"walking", "standing", "sitstand", "ground-pick", "kick-left", "kick-right", "roulade"} <= inventory.ids

def test_alias_collision_blocks_instead_of_guessing():
    inventory = inventory_with(alias="kick", capability_ids=("kick-left", "kick-right"))
    decision = inventory.resolve("kick")
    assert decision.disposition == Disposition.BLOCKED

def test_available_task_does_not_block_training():
    decision = plan_skill(spec("new-skill"), inventory_with(status="available"))
    assert decision.disposition == Disposition.TRAIN_NEW

def test_shipped_match_with_missing_new_metrics_requires_evaluation_first():
    decision = plan_skill(spec("walking", metrics=[metric("new_accuracy")]), builtin_inventory())
    assert decision.disposition == Disposition.BLOCKED
    assert "evaluate existing policy" in decision.reason
```

- [ ] **Step 2: Run tests and verify RED**

Run: `uv run --with pytest pytest tests/test_next_rl_capabilities.py -q`

- [ ] **Step 3: Generate the built-in catalog from shipped policies**

The script accepts an explicit sibling runtime checkout, reads only tracked `policies/*.onnx`, calculates SHA-256, and maps these known artifacts:

```python
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
```

Each record is `learned` with evidence kind `legacy_runtime_shipped`, immutable policy digest, runtime repository/commit, and the runtime README as approval provenance. Legacy evidence covers the named capability only; it does not claim arbitrary new numeric thresholds.

- [ ] **Step 4: Implement conservative matching**

Normalize case, spaces, and underscores to hyphenated identifiers. Exact IDs outrank aliases. Multiple alias matches block. Contract mismatch blocks reuse. A complete learned match returns `reuse`; an existing learned match lacking requested metric evidence returns `blocked/evaluate-first`; a compatible related capability listed in `allowed_parent_capabilities` returns `warm_start`; otherwise return `train_new`. A non-empty `improve_reason` permits training after reuse and is copied into the decision.

- [ ] **Step 5: Verify catalog hashes and focused tests**

Run:

```bash
uv run python scripts/build_next_rl_catalog.py \
  --runtime-repo /Users/rakeshutekar/Documents/microduck --check
uv run --with pytest pytest tests/test_next_rl_capabilities.py -q
```

Expected: generated data matches committed `catalog.json`; all focused tests pass.

- [ ] **Step 6: Commit**

Commit: `feat(next-rl): guard against duplicate skill training`

---

### Task 3: Reproducible experiment manifests and duplicate-run reservations

**Files:**
- Create: `src/mjlab_microduck/next_rl/experiments.py`
- Test: `tests/test_next_rl_experiments.py`

**Interfaces:**
- Consumes: `SkillSpec`, `PlanDecision`, `ArtifactRef`, `canonical_json`, and `atomic_write_json`.
- Produces: `experiment_fingerprint(manifest) -> str`, `ExperimentStore.create(manifest)`, `ExperimentStore.reserve(fingerprint)`, and `ExperimentStore.update_status(...)`.

- [ ] **Step 1: Write failing fingerprint tests**

```python
def test_fingerprint_ignores_timestamp_and_output_path():
    left = manifest(created_at="one", output_dir="a")
    right = manifest(created_at="two", output_dir="b")
    assert experiment_fingerprint(left) == experiment_fingerprint(right)

@pytest.mark.parametrize("field", ["seed", "spec_version", "code_digest", "parent_policy_digest", "runner_id"])
def test_fingerprint_changes_for_learning_inputs(field):
    assert experiment_fingerprint(manifest()) != experiment_fingerprint(manifest(**{field: "changed"}))

def test_duplicate_active_experiment_is_rejected(tmp_path):
    store = ExperimentStore(tmp_path)
    store.create(manifest(status="running"))
    with pytest.raises(DuplicateExperimentError):
        store.reserve(experiment_fingerprint(manifest()))
```

- [ ] **Step 2: Run tests and verify RED**

Run: `uv run --with pytest pytest tests/test_next_rl_experiments.py -q`

- [ ] **Step 3: Implement manifests, fingerprints, and atomic reservations**

Fingerprint only learning inputs: skill ID/version, task ID, normalized environment/agent config, code and dirty-patch digest, seed, parent-policy digest, simulator/actuator contract, and runner. Exclude timestamps, hostnames, PID, output paths, and credentials. Reserve with exclusive file creation. Existing `pending/running/succeeded` blocks; `failed/interrupted` returns an explicit resume/retry decision.

- [ ] **Step 4: Add lifecycle tests and implement transitions**

Allow `planned -> pending -> running -> succeeded|failed|interrupted`. Reject skipping states and mutation of immutable learning inputs. Write status updates atomically and retain status history.

- [ ] **Step 5: Run focused tests and commit**

Run: `uv run --with pytest pytest tests/test_next_rl_experiments.py -q`

Commit: `feat(next-rl): record reproducible experiments`

---

### Task 4: Evaluation aggregation and exact-artifact binding

**Files:**
- Create: `src/mjlab_microduck/next_rl/evaluation.py`
- Test: `tests/test_next_rl_evaluation.py`

**Interfaces:**
- Consumes: `SkillSpec`, `MetricThreshold`, `ArtifactRef`, `sha256_file`, `check_onnx`, and `smoke_run_onnx`.
- Produces: `ScenarioResult`, `MetricResult`, `EvaluationReport`, `evaluate_thresholds(...)`, `select_worst_case(...)`, and `preflight_onnx(...)`.

- [ ] **Step 1: Write failing threshold tests**

```python
def test_mandatory_safety_failure_fails_the_report():
    report = evaluate_thresholds(spec_with(maximum("falls", 0)), [scenario(falls=1)])
    assert report.passed is False

def test_positive_metrics_cannot_outvote_safety():
    report = evaluate_thresholds(spec_with(maximum("falls", 0), minimum("cycles", 3)), [scenario(falls=1, cycles=10)])
    assert report.passed is False

def test_worst_case_selection_is_deterministic():
    assert select_worst_case(results_in_different_order()).scenario_id == "stress-02"

def test_training_and_evaluation_seeds_cannot_overlap():
    with pytest.raises(EvaluationError, match="overlap"):
        evaluate_thresholds(overlapping_seed_spec(), [])
```

- [ ] **Step 2: Run tests and verify RED**

Run: `uv run --with pytest pytest tests/test_next_rl_evaluation.py -q`

- [ ] **Step 3: Implement aggregation**

Support `minimum` and `maximum`, explicit units, required scenario families, finite values, per-scenario threshold results, aggregate pass only when every mandatory threshold passes, and deterministic worst-case ordering by normalized violation then scenario ID.

- [ ] **Step 4: Add failing artifact-binding tests**

```python
def test_report_is_bound_to_exact_policy_bytes(tiny_onnx, tmp_path):
    report = build_report(policy=tiny_onnx, ...)
    tiny_onnx.write_bytes(tiny_onnx.read_bytes() + b"changed")
    with pytest.raises(ArtifactIntegrityError):
        report.verify_policy(tiny_onnx)

def test_nonfinite_onnx_output_fails_preflight(nonfinite_onnx):
    with pytest.raises(EvaluationError):
        preflight_onnx(nonfinite_onnx)
```

- [ ] **Step 5: Implement ONNX preflight and report persistence**

Reuse existing shape/smoke functions, calculate policy SHA-256, bind evaluator revision and spec version, and write an immutable canonical `evaluation.json`. Define a `ScenarioEvaluator` protocol; concrete skill physics metrics are supplied with each future skill rather than guessed in the generic workspace.

- [ ] **Step 6: Run focused tests and commit**

Run: `uv run --with pytest pytest tests/test_next_rl_evaluation.py -q`

Commit: `feat(next-rl): gate policies on held-out evaluation`

---

### Task 5: Visual review bundle and learned-state promotion

**Files:**
- Create: `src/mjlab_microduck/next_rl/review.py`
- Create: `src/mjlab_microduck/next_rl/promotion.py`
- Test: `tests/test_next_rl_review.py`
- Test: `tests/test_next_rl_promotion.py`

**Interfaces:**
- Consumes: passing `EvaluationReport`, policy `ArtifactRef`, capability inventory records, and canonical artifact helpers.
- Produces: `ReviewBundle.build(...)`, `ReviewBundle.verify()`, `PromotionStore.request_review(...)`, `PromotionStore.approve(...)`, and `PromotionStore.reject(...)`.

- [ ] **Step 1: Write failing review tests**

```python
@pytest.mark.parametrize("missing", ["nominal", "entry", "exit", "stress", "worst_case"])
def test_review_requires_every_mandatory_clip(missing, passing_report, clip_files):
    del clip_files[missing]
    with pytest.raises(ReviewError, match=missing):
        ReviewBundle.build(passing_report, clip_files)

def test_every_clip_is_bound_to_the_evaluated_policy(passing_report, clip_files):
    clip_files["stress"].policy_digest = "other"
    with pytest.raises(ReviewError, match="policy digest"):
        ReviewBundle.build(passing_report, clip_files)
```

- [ ] **Step 2: Run review tests and verify RED, then implement review manifests**

The bundle records evaluation digest, exact ONNX digest, clip role/scenario/seed/path/digest, metric summary, and optional seed-matched baseline comparison. Verify non-empty regular files and hashes. Do not trigger publishing or browser automation.

- [ ] **Step 3: Write failing promotion tests**

```python
def test_passing_evaluation_without_human_approval_is_review_pending(store, bundle):
    record = store.request_review("hello", bundle)
    assert record.status == "review_pending"

def test_failed_evaluation_cannot_enter_review(store, failed_bundle):
    with pytest.raises(PromotionError, match="passing evaluation"):
        store.request_review("hello", failed_bundle)

def test_approval_requires_reviewer_and_exact_bundle(store, bundle):
    pending = store.request_review("hello", bundle)
    learned = store.approve(pending.id, reviewer="rakesh")
    assert learned.status == "learned"
    assert learned.approval.review_bundle_digest == bundle.digest

def test_rejection_preserves_prior_learned_policy(store, bundle, prior):
    store.reject(store.request_review("hello-v2", bundle).id, reviewer="rakesh", reason="leans")
    assert store.current_learned("hello") == prior
```

- [ ] **Step 4: Implement the promotion state machine**

Permit `available -> validated -> review_pending -> learned -> superseded`; rejection returns the candidate to `validated` with an audit record. Require non-empty reviewer identity and reason for rejection. Never call the publisher from promotion code.

- [ ] **Step 5: Run focused tests and commit**

Run: `uv run --with pytest pytest tests/test_next_rl_review.py tests/test_next_rl_promotion.py -q`

Commit: `feat(next-rl): require human review for learned skills`

---

### Task 6: Safe Nitro runner and resumable lifecycle

**Files:**
- Create: `src/mjlab_microduck/next_rl/runner.py`
- Create: `scripts/next_rl_remote_job.py`
- Test: `tests/test_next_rl_runner.py`
- Test: `tests/test_next_rl_remote_job.py`

**Interfaces:**
- Consumes: `ExperimentManifest`, experiment fingerprint/store, canonical JSON, and the existing `uv run train` interface.
- Produces: `build_train_argv(...) -> tuple[str, ...]`, `NitroConfig`, `NitroRunner.prepare/start/status/cancel`, `CommandAdapter` protocol, and a fixed remote wrapper accepting one validated job-directory argument.

- [ ] **Step 1: Write failing argv tests**

```python
def test_train_command_uses_real_tyro_flags():
    assert build_train_argv(job()) == (
        "uv", "run", "train", "Mjlab-Velocity-Flat-MicroDuck",
        "--env.scene.num-envs", "1024", "--agent.max-iterations", "1000",
        "--agent.seed", "42", "--agent.run-name", "hello-a1b2c3",
    )

def test_resume_uses_exact_run_and_checkpoint_and_additional_iterations():
    argv = build_train_argv(resume_job(run="2026-09-02_hello", checkpoint="model_250.pt", additional_iterations=500))
    assert argv[-8:] == (
        "--agent.resume", "True", "--agent.load-run", "2026-09-02_hello",
        "--agent.load-checkpoint", "model_250.pt", "--agent.max-iterations", "500",
    )

def test_command_is_argv_not_shell_text():
    assert isinstance(build_train_argv(job()), tuple)
    assert all(isinstance(part, str) for part in build_train_argv(job()))
```

- [ ] **Step 2: Run tests and verify RED, then implement command construction**

Validate task ID and run slug against narrow allowlists. Reject separators, `..`, control characters, leading dashes, and shell metacharacters. Represent resume count as `additional_iterations` in the schema and CLI.

- [ ] **Step 3: Write failing transport/lifecycle tests**

```python
def test_ssh_enforces_noninteractive_host_checked_access(fake_adapter):
    NitroRunner(config(), fake_adapter).status("abc123")
    argv = fake_adapter.calls[0]
    assert "BatchMode=yes" in argv
    assert "StrictHostKeyChecking=no" not in argv

def test_disconnect_does_not_cancel_remote_process(fake_adapter):
    fake_adapter.fail_tail_with_disconnect()
    runner.start(prepared_job())
    assert fake_adapter.cancel_calls == []

def test_cancel_refuses_reused_pid(fake_adapter):
    fake_adapter.remote_identity = "different-command-digest"
    with pytest.raises(RunnerError, match="identity"):
        runner.cancel("abc123")

def test_manifest_and_dry_run_contain_no_credentials(prepared_job):
    serialized = canonical_json(prepared_job.manifest)
    assert "password" not in serialized.lower()
    assert "private_key" not in serialized.lower()
```

- [ ] **Step 4: Implement injected OpenSSH adapter and fixed remote layout**

Use `/home/aif_eng/microduck-training/runs/<fingerprint>/`. The transport invokes `ssh` as an argv sequence with `BatchMode=yes` and the configured SSH alias/user; variable training argv lives in `train-argv.json`. The runner never imports transitive Paramiko/Fabric or disables host verification.

- [ ] **Step 5: Implement and test detached remote wrapper**

The wrapper validates the resolved job directory is beneath the configured root, reads a JSON array, launches with `shell=False`, creates a new process group, and atomically records `pending/running/succeeded/failed`, PID, process start identity, command digest, stdout path, exit code, artifact status, and last stable checkpoint. Cancellation validates PID plus start identity and targets only that process group.

- [ ] **Step 6: Add checkpoint-stability tests**

Verify numeric selection (`model_1000.pt` after `model_999.pt`), require unchanged size/mtime across two probes before sync, mark transfer complete only after digest verification, and retry a failed transfer.

- [ ] **Step 7: Run focused tests and commit**

Run: `uv run --with pytest pytest tests/test_next_rl_runner.py tests/test_next_rl_remote_job.py -q`

Commit: `feat(next-rl): add resumable Nitro training runner`

---

### Task 7: Operator CLI and dry-run workflow

**Files:**
- Create: `src/mjlab_microduck/next_rl/cli.py`
- Modify: `pyproject.toml`
- Modify: `.gitignore`
- Test: `tests/test_next_rl_cli.py`

**Interfaces:**
- Consumes: capability planner, experiment store, evaluator, review/promotion stores, and Nitro runner.
- Produces: `next-rl inventory`, `next-rl plan`, `next-rl prepare`, `next-rl status`, `next-rl review`, `next-rl approve`, and `next-rl reject`.

- [ ] **Step 1: Write failing entry-point and plan-output tests**

```python
def test_next_rl_is_a_declared_script():
    scripts = tomllib.loads(Path("pyproject.toml").read_text())["project"]["scripts"]
    assert scripts["next-rl"] == "mjlab_microduck.next_rl.cli:main"

def test_plan_existing_skill_does_not_construct_training_command(cli, capsys):
    rc = cli(["plan", "examples/skills/walking.json"])
    assert rc == 2
    output = json.loads(capsys.readouterr().out)
    assert output["disposition"] in {"reuse", "blocked"}
    assert "train_argv" not in output

def test_prepare_is_dry_run_by_default(cli, fake_runner):
    assert cli(["prepare", "examples/skills/new.json"]) == 0
    assert fake_runner.start_calls == []
```

- [ ] **Step 2: Run tests and verify RED**

Run: `uv run --with pytest pytest tests/test_next_rl_cli.py -q`

- [ ] **Step 3: Implement argparse CLI with JSON output**

Keep orchestration separate from `train`. `prepare` writes a local experiment and credential-free Nitro bundle but does not start it. A future `start` action is intentionally absent from setup; adding it occurs only when the user requests an actual skill run. Approval requires `--reviewer` and exact review ID. Return nonzero for blocked/reuse when a caller attempted to prepare training.

- [ ] **Step 4: Register entry point and ignore runtime state**

Add `next-rl = "mjlab_microduck.next_rl.cli:main"`. Add `/.next-rl/` to `.gitignore`; tests use temporary `NEXT_RL_HOME`, never the real user directory.

- [ ] **Step 5: Run focused tests and commit**

Run: `uv sync && uv run --with pytest pytest tests/test_next_rl_cli.py -q`

Commit: `feat(next-rl): expose guarded workspace commands`

---

### Task 8: Example specification, operator guide, and readiness verification

**Files:**
- Create: `examples/skills/one-leg-hello.json`
- Create: `docs/next-rl-workspace.md`
- Modify: `README.md`
- Test: `tests/test_next_rl_readiness.py`

**Interfaces:**
- Consumes: public CLI and all persisted schemas.
- Produces: an end-to-end dry-run fixture and documented Mac/Nitro operator workflow.

- [ ] **Step 1: Write failing readiness test**

```python
def test_example_skill_plans_without_training(tmp_path):
    result = run_cli_json(
        "plan", "examples/skills/one-leg-hello.json",
        env={"NEXT_RL_HOME": str(tmp_path)},
    )
    assert result["disposition"] == "warm_start"
    assert result["parent_capability_id"] == "standing"
    assert not list(tmp_path.rglob("model_*.pt"))

def test_prepare_writes_reproducible_manifest_without_starting(tmp_path):
    first = run_cli_json("prepare", EXAMPLE, env=home(tmp_path))
    second = run_cli_json("prepare", EXAMPLE, env=home(tmp_path))
    assert first["fingerprint"] == second["fingerprint"]
    assert first["status"] == "planned"
```

- [ ] **Step 2: Run tests and verify RED**

Run: `uv run --with pytest pytest tests/test_next_rl_readiness.py -q`

- [ ] **Step 3: Add the non-training example specification**

Define left-foot support, right-foot clearance, lateral right-hip cycles, trunk tilt, forbidden contacts, falls, entry from ordinary standing, and safe return. Use explicit placeholder-free numeric thresholds labelled as simulation acceptance defaults; mark hardware calibration as required before deployment. List `standing` as an allowed parent capability. The example is planning data only and launches nothing.

- [ ] **Step 4: Document the operator flow**

Document inventory, plan, prepare, Nitro status, evaluation ingestion, review bundle inspection, approve/reject, resume semantics, output locations, and the separation between learned/publish/deploy. Include the exact rule: an overlapping shipped capability is evaluated before any retraining decision.

- [ ] **Step 5: Run all local verification**

Run:

```bash
uv run --with pytest pytest tests/test_next_rl_schema.py tests/test_next_rl_capabilities.py tests/test_next_rl_experiments.py tests/test_next_rl_evaluation.py tests/test_next_rl_review.py tests/test_next_rl_promotion.py tests/test_next_rl_runner.py tests/test_next_rl_remote_job.py tests/test_next_rl_cli.py tests/test_next_rl_readiness.py -q
uv run --with pytest pytest tests/ -q
git diff --check
git status --short
```

Expected: all tests pass; only intended documentation/code/test changes remain.

- [ ] **Step 6: Sync the branch to the Nitro workspace and run bounded verification**

Push only after explicit repository-upload authorization. Without pushing, use the runner's safe tracked-file archive to prepare a Nitro dry run. Then run the repository-mandated smoke command only:

```bash
WANDB_MODE=disabled uv run train Mjlab-Velocity-Flat-MicroDuck \
  --env.scene.num-envs 64 --agent.max-iterations 5
```

Verify CUDA/MuJoCo Warp initialization, 61D actor, privileged critic, finite rewards, checkpoint creation, clean exit, and no remaining trainer process. Do not promote or publish the smoke artifact.

- [ ] **Step 7: Commit documentation and example**

Commit: `docs(next-rl): document guarded training workflow`

---

## Plan self-review

- Spec coverage: capability inventory, duplicate guard, versioned specs, immutable experiments, Nitro lifecycle, evaluation, video review, human approval, and separate publishing are each assigned to a task.
- Scope control: PPO, task rewards, runtime contract, existing train/HF hooks, automatic upload, and hardware deployment remain unchanged.
- Type consistency: every downstream interface is produced by an earlier task; `SkillSpec`, `PlanDecision`, `EvaluationReport`, `ReviewBundle`, and `ExperimentManifest` names remain consistent.
- Security: runner variables remain JSON argv, host verification stays enabled, credentials are excluded, and remote paths are fixed beneath the Nitro training root.
- Verification: every production behavior starts with a failing focused test, then focused and full-suite checks; final GPU work is limited to the required smoke run.

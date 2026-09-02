# Next RL Workspace Design

**Date:** 2026-09-02
**Status:** Approved

## Goal

Build a Microduck-specific orchestration layer on the validated `microduck_rl`
sim-to-real stack. The workspace must prevent unnecessary retraining, make every
new skill measurable before training starts, run resumable jobs on the Nitro GPU
runner, and require both automatic evaluation and explicit human video approval
before a policy can be called learned.

Faster learning means less wall-clock time to a fixed, held-out success
threshold. Good accuracy means the deterministic exported policy satisfies the
skill's physical metrics across nominal and held-out conditions. Simulator
throughput and training return alone are not success measures.

## Boundaries

The workspace preserves the existing Microduck MJCF, BAM XL330 actuator model,
backlash variants, domain-randomization conventions, 50 Hz control loop,
61-observation actor contract, 14-action output, normalized ONNX export, and
RSL-RL PPO baseline. Native MuJoCo is the reference evaluation backend and
MuJoCo Warp is the Nitro training backend.

The first version does not rewrite PPO, change the runtime policy contract,
train a universal all-skills policy, automatically deploy to hardware, or
automatically publish to a remote repository. It also does not start a long
training job as part of workspace setup.

## Architecture

The new `mjlab_microduck.next_rl` package contains six focused areas:

- `capabilities`: inventory of requested capabilities, existing task families,
  policy artifacts, validation evidence, and compatibility constraints.
- `specifications`: versioned, machine-readable skill definitions and metric
  thresholds that are validated before a run can be planned.
- `experiments`: immutable experiment manifests with code revision, seed,
  runner, environment count, parent checkpoint, configuration, and output paths.
- `runner`: command construction and lifecycle metadata for local and Nitro
  execution. SSH commands are argument arrays; credentials never enter manifests
  or logs.
- `evaluation`: deterministic ONNX checks, held-out scenario aggregation,
  transition/safety metrics, and a pass/fail evaluation report.
- `promotion`: a state machine that permits review only after evaluation passes
  and permits `learned` only after explicit human approval.

`review` builds a review manifest containing the nominal, entry, exit, stress,
and worst-case rollout videos plus their metrics. Rendering can use the existing
MuJoCo export/inference infrastructure; browser presentation remains a consumer,
not a dependency of the core state machine.

## Capability and duplicate-training rules

A task name is not evidence that a capability is learned. Each capability record
has a stable identifier, aliases, semantic version, robot model, policy contract,
status, policy artifact, and optional evaluation report. The initial statuses are
`available`, `validated`, `review_pending`, `learned`, and `superseded`.

Planning a request produces exactly one disposition:

- `reuse`: an existing `learned` policy satisfies the requested capability,
  contract, and minimum evaluation thresholds.
- `warm_start`: a compatible validated or learned policy is related but does not
  satisfy the complete request.
- `train_new`: no compatible evidence-backed policy exists.
- `blocked`: the request or its acceptance criteria are incomplete or invalid.

Only `learned` records block duplicate training by default. `available` tasks and
unreviewed checkpoints remain candidates, never proof. A deliberate improvement
experiment may override `reuse`, but the override and reason must be recorded in
the experiment manifest. Matching is deterministic and conservative; ambiguous
aliases produce `blocked`, not a guessed selection.

## Skill specifications

A skill specification includes identity, description, compatible robot and
policy contract, entry and exit states, commands, required metrics, held-out
scenarios, curriculum stages, allowed parent capabilities, rendering views, and
minimum review clips. Thresholds are numeric and unit-labelled. Every positive
metric declares whether larger or smaller is better, and every safety metric is
mandatory.

Curriculum stages may alter spawn distributions, physical assistance, physics
difficulty, or tolerances. They may not silently redefine the final success
criteria. The actor always receives deployable observations; the critic may add
simulator-only velocity, contact, randomized dynamics, latency, and disturbance
state.

## Run flow

1. Validate the skill specification.
2. Resolve the capability request against the inventory.
3. Stop on `reuse` unless an improvement override was explicitly requested.
4. Select a compatible parent policy for `warm_start`, otherwise the approved
   standing baseline.
5. Write an immutable experiment manifest.
6. Run the repository-required 64-environment, five-iteration smoke test.
7. Benchmark supported environment counts on the target runner and select the
   fastest count that fits its memory guardrails.
8. Launch a resumable full run only after an explicit training request.
9. Export the deterministic mean policy using the existing normalized exporter.
10. Evaluate nominal, held-out, stress, long-horizon, entry, and exit scenarios.
11. Build the visual review package, including the worst detected rollout.
12. Present videos and metrics to the human reviewer.
13. Record explicit approval or rejection. Only approval after a passing report
    transitions the capability to `learned`.
14. Publishing remains a separate, explicit action.

## Evaluation and visual review

An evaluation suite is tied to a skill-specification version. A report records
the policy digest, evaluator revision, scenario seeds, raw metric summaries,
threshold results, aggregate result, and artifact paths. Training seeds and
held-out evaluation seeds cannot overlap.

The review bundle contains at least one nominal rollout, ordinary-standing entry,
safe exit, held-out stress rollout, and worst-case rollout. When an existing
policy is being improved, it also contains a seed-matched side-by-side comparison.
All review videos are produced from the exported deterministic ONNX artifact, not
the stochastic policy used to collect PPO rollouts.

## Failure handling

- Contract or specification mismatch blocks planning before simulation.
- NaN physics or non-finite policy output fails the run or evaluation.
- CUDA out-of-memory preserves existing checkpoints and records a smaller
  environment-count recommendation.
- SSH disconnect does not own the trainer process; the remote job records a PID,
  status file, log path, and last checkpoint for later inspection or resume.
- An interrupted or failed run never changes capability status.
- Failed automatic evaluation remains an experiment and cannot enter review.
- Rejected human review records the rejection and leaves the prior learned policy
  unchanged.
- ONNX parity, shape, or normalization failure blocks review and promotion.

## Security and operations

The Nitro runner address and username may be configured outside experiment
artifacts. Passwords and private keys are never stored in the repository or
command strings. Host-key verification remains enabled. Remote paths are rooted
under `/home/aif_eng/microduck-training`, and runner operations may not delete or
modify unrelated laptop projects.

The setup can create plans and perform bounded smoke tests, but a full training
run requires an explicit skill request and validated specification. Upload,
publication, and hardware deployment each require separate explicit actions.

## Verification

Unit tests cover schema validation, conservative capability matching, duplicate
guards, manifest immutability, state transitions, threshold aggregation, command
construction, and secret exclusion. Integration tests use temporary local
directories and fake process adapters rather than a live SSH host. Existing
repository tests remain green.

Readiness requires: a clean isolated worktree; passing baseline and new tests; a
valid example specification; deterministic planning for reuse, warm-start, and
new-skill cases; a dry-run Nitro command; evaluation and review manifests; and a
five-iteration Nitro GPU smoke test. No long skill training is part of readiness.

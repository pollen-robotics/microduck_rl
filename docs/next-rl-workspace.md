# Next RL workspace operator guide

`next-rl` is the guarded workspace for planning, recording, evaluating, and
human-reviewing a Microduck skill. It is not a new task implementation, reward
definition, trainer, policy artifact, publisher, or deployment tool.

The workspace keeps its state below `NEXT_RL_HOME`, which defaults to
`.next-rl` under the current directory. Use a named, writable location when an
operator needs to retain a job's state:

```bash
export NEXT_RL_HOME="$PWD/.next-rl"
```

The workspace stores immutable experiment records in
`$NEXT_RL_HOME/experiments/<fingerprint>/`, staged runner bundles in
`$NEXT_RL_HOME/bundles/<fingerprint>/`, review manifests in
`$NEXT_RL_HOME/review-bundles/`, and durable promotion state in
`$NEXT_RL_HOME/promotions/`. Treat a fingerprint as the identity of one set of
learning inputs, rather than as a model artifact or deployment identifier.

## Plan before any training

First inspect the shipped catalogue and any capability already promoted in this
workspace:

```bash
uv run next-rl inventory
uv run next-rl plan examples/skills/one-leg-hello.json
```

`plan` returns a JSON disposition. `reuse` means an approved learned policy
has the requested, evidenced contract and metrics. `warm_start` names a
compatible parent policy; the `one-leg-hello` example can use `standing` as
that parent. `train_new` means the catalogue has no matching evidence.
`blocked` means the request is ambiguous, incompatible, or has insufficient
evidence.

An overlapping shipped capability is evaluated before any retraining decision.
Shipped or otherwise validated capability records are not automatically proof
that they satisfy a new skill's numeric acceptance thresholds. Do not turn a
blocked/evaluate-first result into a train command; evaluate the exact existing
policy first. A deliberate improvement of a learned matching capability needs
an auditable reason:

```bash
uv run next-rl plan examples/skills/one-leg-hello.json \
  --improve-reason "lower held-out trunk tilt while preserving safe return"
```

## Prepare and inspect a bounded job

The current public syntax is:

```bash
uv run next-rl prepare examples/skills/one-leg-hello.json \
  --task-id Mjlab-OneLegHello-MicroDuck \
  --num-envs 64 --max-iterations 5 --run-name one-leg-hello-smoke
uv run next-rl status <fingerprint>
```

`prepare` records an immutable manifest and stages a safe, tracked-source
bundle for the configured Nitro runner. It does **not** expose a `start`
subcommand and does not start a trainer. A real runner preparation requires
`NEXT_RL_NITRO_SSH_ALIAS` (and optionally `NEXT_RL_NITRO_SSH_USER`); it uses
host-key-checked transport. The displayed fingerprint is the value to pass to
`status`, which reports lifecycle state and any stable checkpoint metadata
without transport credentials.

The safe runner archives the committed Git tree, not local uncommitted edits.
Therefore commit the example, guide, README, and readiness tests before a
remote sync or bounded Nitro smoke. The required documentation commit is:

```text
docs(next-rl): document guarded training workflow
```

After a prepared job is deliberately started by the authorized runner workflow,
run only the repository's bounded smoke before considering a longer run:

```bash
WANDB_MODE=disabled uv run train Mjlab-Velocity-Flat-MicroDuck \
  --env.scene.num-envs 64 --agent.max-iterations 5
```

This smoke is a configuration check, not evidence that `one-leg-hello` is
learned. Do not publish or deploy its checkpoint.

### Resume semantics

There is currently no `next-rl resume` public subcommand. A guarded runner
resume is a new manifest that identifies a distinct prior fingerprint, exact
stable checkpoint name and SHA-256, and the original run name. Its
`additional_iterations` value is an increment: the runner maps it to the one
`--agent.max-iterations` value for the resumed invocation; it is not the
original run's total iteration count. A failed or interrupted run must be
inspected with `status` and its checkpoint verified before such a resume is
prepared.

## Evaluate and request human review

Evaluation is supplied by the simulation/evaluator workflow as a canonical
JSON report bound to the exported ONNX digest. The workspace does not invent
physics metrics or evaluate a checkpoint solely because training completed.
Once the report passes, provide the capability document, skill specification,
evaluation report, and all five required video inputs:

```bash
uv run next-rl review \
  --capability capability.json \
  --skill examples/skills/one-leg-hello.json \
  --evaluation evaluation.json \
  --nominal-clip nominal.mp4 \
  --entry-clip entry.mp4 \
  --exit-clip exit.mp4 \
  --stress-clip stress.mp4 \
  --worst-case-clip worst-case.mp4
```

Those clips are required evidence for nominal behavior, ordinary-standing
entry, safe exit, held-out stress, and the worst observed rollout. The review
bundle binds each clip and the report to the exact ONNX digest. It is not a
replacement for a human reviewer.

Use the returned record ID to make the human decision explicit:

```bash
uv run next-rl approve <record-id> --reviewer "operator-name"
uv run next-rl reject <record-id> --reviewer "operator-name" \
  --reason "specific observable safety or behavior concern"
```

Only a passing evaluation followed by `approve` changes a candidate to
`learned`. `reject` returns the candidate to `validated`, records its reason,
and preserves any prior learned policy. Review, approval, and rejection never
start training, publish a repository, or command hardware.

## The example specification

[`examples/skills/one-leg-hello.json`](../examples/skills/one-leg-hello.json)
is planning data for a small, non-training acceptance contract. It specifies:

- left-foot support while the right foot is raised at least 0.03 m;
- three lateral right-hip side-to-side cycles, with right-knee flexion held
  between 0.15 and 0.45 rad;
- trunk tilt at most 0.25 rad, no forbidden contacts or falls, and a two-foot
  ordinary-standing return held for 2.0 s;
- explicit, unit-labelled simulation acceptance defaults and disjoint training
  and evaluation seeds.

Its allowed parent is `standing`. Its `hardware_deployment.calibration_required`
flag is deliberately true: approval never waives hardware calibration.

## Learned is not published or deployed

`learned` is an evidence and human-approval status inside this workspace. It
does not upload an ONNX, alter a remote policy repository, install a policy,
calibrate the robot, or execute the motion on hardware. Exporting through the
existing normalized ONNX path, publishing with the separate `publish` command,
and robot deployment are separate, explicitly authorized operations. Complete
hardware calibration and the runtime's own safety procedure before any
deployment.

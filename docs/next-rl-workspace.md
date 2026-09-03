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
has the requested, evidenced contract and metrics. `train_new` means training
must start from a new policy; when a compatible, loadable ONNX exists, the
result can name it as `reference_capability_id` for evaluation and comparison.
An ONNX policy is not a resumable training checkpoint and therefore is never
presented as a warm start. `blocked` means the request is ambiguous,
incompatible, or has insufficient evidence.

An overlapping shipped capability is evaluated before any retraining decision.
Shipped or otherwise validated capability records are not automatically proof
that they satisfy a new skill's numeric acceptance thresholds. Do not turn a
blocked/evaluate-first result into a train command; evaluate the exact existing
policy first. A deliberate improvement of a learned matching capability needs
a higher semantic spec version and an auditable reason:

```bash
uv run next-rl plan examples/skills/one-leg-hello.json \
  --improve-reason "lower held-out trunk tilt while preserving safe return"
```

## Prepare and inspect a bounded job

The current public `prepare` syntax is:

```bash
uv run next-rl prepare examples/skills/one-leg-hello.json \
  --task-id Mjlab-OneLegHello-MicroDuck \
  --num-envs 64 --max-iterations 5 --run-name one-leg-hello-smoke
uv run next-rl status <fingerprint>
```

`Mjlab-OneLegHello-MicroDuck` is a non-runnable planning placeholder: this
repository does not yet have its task or reward registration. The
`one-leg-hello` specification is valid planning data, but it **MUST NOT** be
started. `prepare` can record and stage the planning boundary, but it cannot
turn the placeholder into a runnable hello task and must never be followed by a
hello start on Nitro.

`prepare` records an immutable manifest and stages a safe, tracked-source
bundle for the configured Nitro runner. It does not expose a `start`
subcommand. A real runner preparation requires
`NEXT_RL_NITRO_SSH_ALIAS` (and optionally `NEXT_RL_NITRO_SSH_USER`); it uses
host-key-checked transport. The displayed fingerprint is the value to pass to
`status`, which reports lifecycle state and any stable checkpoint metadata
without transport credentials.

`NitroRunner.start` remains an internal API for a separately authorized
launcher. That API requires the same durable experiment store: it reserves the
experiment fingerprint before transport, rejects another launch of the same
inputs, releases only its own reservation if transfer fails before launch,
retains the same-owner claim when the launch response is uncertain, and
synchronizes pending, running, succeeded, or failed status as the remote
lifecycle advances. While a launch is pending, status inspection revalidates
the recorded detached supervisor PID, Linux start time, and exact command/job
digest in both spawned and supervising states. If an acknowledged supervisor
disappears before launching the trainer, inspection marks that launch failed
and retryable so a newly reserved start can recover it.
None of this creates a public `start` command, and the runner does not publish
or deploy a policy.

### Current Nitro connection configuration

For the current Nitro endpoint, configure the authorized runner environment
exactly as follows:

```bash
export NEXT_RL_NITRO_SSH_ALIAS=108.61.217.115
export NEXT_RL_NITRO_SSH_USER=aif-engineering
export NEXT_RL_NITRO_WSL_DISTRIBUTION=Ubuntu
```

`NEXT_RL_NITRO_WSL_DISTRIBUTION` is optional for direct-Linux hosts. It is
required for this Nitro because public SSH lands in Windows; the runner bridges
each remote Linux command through `wsl.exe -d Ubuntu --`. The host key must
already be trusted. Transport uses BatchMode public-key authentication. Never
include a password in an environment variable, command, manifest, or
documentation. Do not disable host checking or replace the known-hosts file.

### One-time remote root bootstrap

The fixed runner root is `/home/aif_eng/microduck-training/runs`. If that exact
directory is absent, first prove its parent exists, then create only the fixed
child with the same host-checked, public-key transport boundary:

```bash
ssh -o BatchMode=yes aif-engineering@108.61.217.115 \
  wsl.exe -d Ubuntu -- test -d /home/aif_eng/microduck-training
ssh -o BatchMode=yes aif-engineering@108.61.217.115 \
  wsl.exe -d Ubuntu -- mkdir -- /home/aif_eng/microduck-training/runs
```

The second command is permitted only after the first succeeds and creates only
the fixed runner root. The runner intentionally will not create the broader
parent, so do not substitute a broader path, use a recursive create,
host-check bypass, or put a password in the command. Once the root exists,
normal guarded `prepare` operations own only their fingerprint directory.

The safe runner archives the committed Git tree, not local uncommitted edits.
Therefore commit the example, guide, README, and readiness tests before a
remote sync or bounded Nitro smoke. The required documentation commit is:

```text
docs(next-rl): document guarded training workflow
```

The separately registered `Mjlab-Velocity-Flat-MicroDuck` task is the only
bounded workspace/GPU smoke shown here:

```bash
WANDB_MODE=disabled uv run train Mjlab-Velocity-Flat-MicroDuck \
  --env.scene.num-envs 64 --agent.max-iterations 5
```

This is not a prepared hello job and does not validate the hello task, reward,
or phase contract. It is only a workspace/GPU configuration check; do not
publish or deploy its checkpoint.

### Verified live readiness — 2026-09-02

The following evidence was collected for source commit
`b60c85c6569bfe9767ef565333bdd9aed0052c1a` only. Later documentation-only
commits were not GPU tested.

- The fixed runner root was absent once. Its parent
  `/home/aif_eng/microduck-training` already existed before root creation, so
  historical creation made only the missing fixed root; the checked-parent
  procedure above is the safe documented procedure. Afterwards, `next-rl
  prepare` returned `planned` for fingerprint
  `bf220169cd67449688a97cdf9d5c3dcc3e20aa983f9af9e91ca193707435fd29`.
  Its source tree was `110b84b16b24d45cb6bdbb4ca81f29ede6a5a5ca`, and archive
  SHA-256 `55e80206a875e7acc2c06593dfe7f375b74f3bf2a1400b894adaedf527afb3a0`
  matched the local and remote copies.
- A separate registered `Mjlab-Velocity-Flat-MicroDuck` smoke used
  `WANDB_MODE=disabled`, 64 environments, and 5 iterations on RTX 5050
  `cuda:0` with 8 GiB. It reported actor 61→14 and critic 76→1, displayed
  finite losses and rewards, exited 0, and produced `model_0.pt`, `model_4.pt`,
  and an ONNX export. Afterwards, `pgrep -af train` exited 1 with no output.
- No hello start, promotion, or publish occurred. The staged hello preparation
  is not a task registration, learned-policy claim, or deployment authorization.

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
The CLI preflights the actual 61-input/14-output ONNX and verifies the recorded
digest before creating workspace state. Once the report passes, provide the
capability document, skill specification, evaluation report, and five immutable
renderer evidence sidecars:

```bash
uv run next-rl review \
  --capability capability.json \
  --skill examples/skills/one-leg-hello.json \
  --evaluation evaluation.json \
  --nominal-evidence nominal.render.json \
  --entry-evidence entry.render.json \
  --exit-evidence exit.render.json \
  --stress-evidence stress.render.json \
  --worst-case-evidence worst-case.render.json
```

Each renderer writes its sidecar only after producing a decodable video: an MP4
with at least two temporal frames. Static PNG/JPEG/GIF images, one-frame MP4
files, and non-video payloads are rejected. The canonical sidecar records role,
scenario ID, evaluation seed, video path and SHA-256, renderer revision, and
the exact policy and evaluation digests. The five roles cover nominal behavior,
ordinary-standing entry, safe exit, held-out stress, and the worst observed
rollout. `next-rl review` loads these renderer-authored bindings; it does not
infer provenance from filenames or invent scenario, seed, policy, or report
bindings. The resulting review bundle is not a replacement for a human
reviewer.

Before approving or rejecting, retain the returned `bundle_path` and
`bundle_digest`, then inspect the canonical manifest and metrics and verify its
exact file binding:

```bash
uv run python - "$bundle_path" "$bundle_digest" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

from mjlab_microduck.next_rl.artifacts import canonical_json
from mjlab_microduck.next_rl.review import ReviewBundle

path = Path(sys.argv[1])
expected_digest = sys.argv[2]
text = path.read_text(encoding="utf-8")
raw = json.loads(text)
assert canonical_json(raw) == text
assert hashlib.sha256(text.encode("utf-8")).hexdigest() == expected_digest
bundle = ReviewBundle.from_dict(raw)
assert bundle.digest == expected_digest
bundle.verify()
print(json.dumps(json.loads(bundle.evaluation_json)["scenarios"], indent=2))
for clip in bundle.clips:
    print(
        clip.role, clip.path, clip.digest, clip.policy_digest,
        clip.evaluation_digest, clip.renderer_revision,
        clip.evidence_path, clip.evidence_digest,
    )
PY
```

`bundle.verify()` repeats ONNX preflight and checks the exact evaluation,
policy, sidecar, and video digests, the renderer revision, each role's evaluated
scenario and seed, and that every file remains an MP4 with at least two
decodable temporal frames. Open all five printed videos—nominal, entry, exit,
stress, and worst case—and inspect the printed metrics before making the human
decision. Use the returned record ID only after that inspection:

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

- entry from ordinary two-foot standing, with both left and right foot contacts
  required;
- only during the raised-wave phase, left-foot support, right-foot clearance of
  at least 0.03 m, three lateral right-hip side-to-side cycles, right-knee
  flexion between 0.15 and 0.45 rad, and trunk tilt at most 0.25 rad;
- only during the raised-wave phase, no head, trunk, or right-foot contact;
  right-foot contact is required, not forbidden, at entry and safe exit;
- no episode falls and a safe two-foot ordinary-standing return held for 2.0 s;
- explicit, unit-labelled simulation acceptance defaults and disjoint training
  and evaluation seeds.

Its allowed reference is `standing`, but the shipped ONNX is evaluation and
comparison evidence only, not a checkpoint warm start; training initialization
remains a new policy. Its `hardware_deployment.calibration_required` flag is
deliberately true: approval never waives hardware calibration.

## Learned is not published or deployed

`learned` is an evidence and human-approval status inside this workspace. It
does not upload an ONNX, alter a remote policy repository, install a policy,
calibrate the robot, or execute the motion on hardware. Exporting through the
existing normalized ONNX path, publishing with the separate `publish` command,
and robot deployment are separate, explicitly authorized operations. Complete
hardware calibration and the runtime's own safety procedure before any
deployment.

# MicroDuck ROM simulator release and operations

The ROM simulator runs only verified, qualified policy bundles. Qualification and
serving use the same code-owned action specifications and
`MicroduckMujocoRuntime`; there is no manifest-selected Python evaluator.

## 1. Export and build a candidate

Export with the repository exporter so the observation normalizer and exact
source, task, checkpoint, and run identities are embedded in ONNX metadata:

```bash
COMMIT=$(git rev-parse HEAD)
uv run scripts/export.py Mjlab-Velocity-Flat-MicroDuck \
  --wandb-run-path entity/project/run-id \
  --checkpoint 3000

uv run scripts/build_rom_bundle.py \
  --release 1.0.0-candidate.1 \
  --artifact WALK_VELOCITY=output.onnx \
  --output ../release/microduck-candidate.zip \
  --model src/mjlab_microduck/robot/microduck/scene_walk.xml \
  --terrain flat \
  --scenario-profile SEEDED_SERVO_RESET_V1 \
  --source-commit "$COMMIT" \
  --checkpoint model_3000.pt \
  --experiment-ref entity/project/run-id \
  --created-at 2026-08-29T12:00:00+00:00 \
  --license-file LICENSE
```

Never hand-convert a checkpoint. Candidate and promoted outputs must live
outside `src/mjlab_microduck/robot/microduck`; the qualification CLI resolves
and protects that repository robot-source root in addition to the candidate
bundle root. Additional protected roots can be repeated with
`--protected-source-root`.

Extract the candidate into a new staging directory:

```bash
mkdir -p ../release/candidate
python -m zipfile -e ../release/microduck-candidate.zip ../release/candidate
```

## 2. Declare and run qualification

Mandatory versus optional is release policy, not an implication of the action
catalog. Every candidate action currently marked `AVAILABLE` must appear in the
release configuration. A mandatory action must already have verified policy,
model, scenario, and runtime support. Optional actions without that support stay
`UNAVAILABLE` with their original reason; they are not falsely qualified.
The builder and promoted bundle always retain the complete ordered 15-action V1
catalog. Actions already marked `UNAVAILABLE` may be omitted from `release.json`;
qualification carries each one deterministically as `UNAVAILABLE` using its
code-owned reason. Any action a release intends to qualify, and every action the
candidate marks `AVAILABLE`, still requires an explicit governed declaration.

Example `release.json` for the currently supported flat walking runtime:

```json
{
  "schema": "MICRODUCK_ROM_RELEASE_V1",
  "release": "1.0.0",
  "createdAt": "2026-08-29T12:00:00Z",
  "actions": [
    {
      "actionCode": "WALK_VELOCITY",
      "mandatory": true,
      "terrain": "flat",
      "resetProfile": "DEFAULT_STANDING",
      "seeds": [7, 11, 29],
      "maxSteps": 500,
      "parameters": {"vxMps": 0.1, "vyMps": 0.0, "yawRateRadps": 0.0},
      "thresholds": {
        "minSuccessRate": 1.0,
        "maxFallRate": 0.0,
        "maxMeanTrackingError": 0.25,
        "minMeanDistanceM": 0.5,
        "maxMeanEnergyProxy": 200.0,
        "maxActuatorClampSteps": 0,
        "maxPhysicalJointLimitViolations": 0,
        "actionMetric": "trackingError",
        "actionMetricOperator": "lte",
        "actionMetricThreshold": 0.25
      }
    }
  ]
}
```

Run the bounded batteries and promote to a new immutable version/output:

```bash
MUJOCO_GL=egl uv run scripts/qualify_rom_bundle.py \
  --bundle-dir ../release/candidate \
  --release-config ../release/release.json \
  --output ../release/microduck-qualified-1.0.0.zip
```

The output path must not exist. Promotion never changes the candidate directory
or ZIP. An optional threshold failure remains in the catalog as
`UNAVAILABLE / QUALIFICATION_FAILED`; a mandatory failure produces no promoted
ZIP.

Release files cannot select a runtime revision. Qualification derives it from
the installed `mjlab-microduck` package version and a digest of the exact
governed runtime modules. Batteries require 3–16 unique seeds and 100–2,000
steps, and action commands, terrain, reset, action metric, and metric direction
must match the code-owned action specification.

The auditable runtime-revision source set is `action_catalog.py`,
`action_specs.py`, `api.py`, `bundle.py`, `contracts.py`, `main.py`,
`mirroring.py`, `model_semantics.py`, `mujoco_runtime.py`, `observation.py`,
`onnx_policy.py`, `qualification.py`, `runtime.py`, `runtime_identity.py`,
`service.py`, and `store.py`. Tests require that changing any one of these files
changes the revision.

The qualification report contains bounded per-seed success, fall, stepwise
tracking statistics, distance, normalized-action energy proxy, separately
counted actuator-clamp steps and physical joint-limit violations, max-action, and
action-specific metrics, with timestamps and exact simulator, runtime, model,
policy, source, checkpoint, and run identities. Its
`subjectBundleDigest` is the digest of the verified candidate bytes actually
executed. The promoted manifest and promoted digest then cover that report as an
artifact. This `VERIFIED_INPUT_BUNDLE_DIGEST_V1` rule avoids an impossible
self-referential digest: the report never claims to contain the final ZIP digest.

Raw rollout domains are code-owned. `rollRotationRad`, `slopeProgressM`, and
`yawRotationRad` are signed. `payloadLifted`, `standFraction`,
`supportFootContact`, `terrainExitReached`, and `uprightReached` are fractions
in `[0, 1]`. Every other declared action metric is nonnegative. Tracking error,
its sum and maximum, distance, energy, and maximum absolute action are also
nonnegative and finite. Each rollout carries one tracking sample per completed
control step plus the accumulated sum. The runtime serializes the sum to six
decimal places and defines the canonical tracking mean as that serialized sum
divided by a positive sample count, rounded to six decimal places. Startup
recomputes this value rather than trusting a terminal snapshot; zero or invalid
sample counts fail qualification.

Action-metric values are never an independent authority. Startup derives
`trackingError` from the verified tracking sum/sample mean, `baseTravelM` from
`distanceM`, `standFraction` from `uprightSteps / steps`, `yawRotationRad` from
its explicit yaw accumulator, and `standPoseError` from the final STAND pose
field. The carried `actionMetricValue` must equal that code-owned derivation
exactly, and aggregates, thresholds, and promotion status use the derived
value. The carried `trackingError` duplicate must likewise equal the canonical
sum/sample mean exactly; no trust-boundary tolerance may cross a threshold. A
metric key without an installed derivation fails closed.

Governed STAND completion requires ten consecutive samples that each satisfy
the same shared runtime limits: pose error at most 0.08 rad, trunk height in
`[0.09, 0.14]` m, trunk tilt at most 15 degrees, and maximum absolute joint
speed at most 0.5 rad/s. Raw evidence records the final consecutive count plus
the maximum pose error, minimum/maximum trunk height, maximum tilt, and maximum
joint speed over exactly that claimed window. Qualification checks those
extrema against the shared limits and against the rollout tracking maximum;
ten high-error samples cannot be relabeled as settled. Runtime task evidence
remains bounded to 32 scalar fields and 1 KiB.

## 3. Build and run the container

```bash
docker build -f docker/rom-simulator/Dockerfile \
  -t microduck-rom-sim:1.0.0 .

mkdir -p ../release/qualified-bundle ../release/state
python -m zipfile -e ../release/microduck-qualified-1.0.0.zip \
  ../release/qualified-bundle

umask 077
printf 'MICRODUCK_ROM_BEARER_TOKEN=%s\n' 'replace-with-a-random-token' \
  > ../release/rom-token.env

docker run --name microduck-rom-sim --rm \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=64m \
  --mount type=bind,src="$(realpath ../release/qualified-bundle)",dst=/bundle,readonly \
  --mount type=bind,src="$(realpath ../release/state)",dst=/state \
  --env-file ../release/rom-token.env \
  --publish 127.0.0.1:8000:8000 \
  microduck-rom-sim:1.0.0
```

The image runs as UID/GID 10001, listens only on its configured internal port
8000, and uses an exec entrypoint. `/bundle` is read-only; `/state` is the only
persistent writable mount; `/tmp` is ephemeral workspace for verified MJCF
snapshots. A valid qualified mount activates the Task 7 runtime during startup.
Invalid bundle, token, state, model, or policy inputs fail closed.

The build syncs only the exact-pinned `rom` dependency group from the frozen
lockfile. The ROM runtime does not import BAM, Torch, CUDA, or `mjlab`; it
executes the verified deployment bundle with Task 7's position-actuator
semantics. This keeps Git and training-only integration code out of the runtime
image without floating any installed dependency version.

The Dockerfile-specific ignore file and root-context `.dockerignore` are
deny-by-default allowlists. They admit only lock/package/readme/license metadata,
the direct ROM Python package, and the entrypoint; robot, training, tests,
checkpoints, outputs, env files, STL, and `.part` paths remain excluded. The
final image carries `/app/LICENSE` and the package metadata used for runtime
revision binding. It contains no production or distribution-restricted model
bytes; runtime model resolution is exclusively from the verified `/bundle`
mount.

## 4. Authenticated API checks

Keep tokens out of source control, command arguments, screenshots, transcripts,
and shell tracing. Load the token without printing it:

```bash
set -a
. ../release/rom-token.env
set +a
AUTH="Authorization: Bearer ${MICRODUCK_ROM_BEARER_TOKEN}"
```

Liveness is public; all other endpoints require the exact bearer token:

```bash
curl --fail --silent http://127.0.0.1:8000/v1/health
curl --fail --silent -H "$AUTH" http://127.0.0.1:8000/v1/ready
curl --fail --silent -H "$AUTH" http://127.0.0.1:8000/v1/catalog
```

Read the installed identities from the catalog before creating a task:

```bash
CATALOG=$(curl --fail --silent -H "$AUTH" http://127.0.0.1:8000/v1/catalog)
BUNDLE_VERSION=$(printf '%s' "$CATALOG" | jq -r .bundleVersion)
BUNDLE_DIGEST=$(printf '%s' "$CATALOG" | jq -r .bundleDigest)
```

Every body sent with `POST` or `PUT` below `/v1` is limited to 65,536 bytes
before route or path-parameter validation. The limit is enforced while ASGI
chunks arrive even when `Content-Length` is absent; an oversized body receives
HTTP 413 with `REQUEST_BODY_TOO_LARGE`. Contract strings,
collections, parameter maps, scenarios, event payloads, and persisted canonical
requests have smaller schema bounds. V1 accepts only the exact seeded scenario
object `{"terrain":"flat"|"ramp","seed":<uint32>}` and rejects extra fields
or JSON type coercion.

Standard JSON Schema keywords encode types, ranges, collection counts, string
bounds, and raw-control property-name rejection where portable keywords exist.
Canonical UTF-8 byte limits, maximum nesting depth, finite/strict scalar rules,
and case-insensitive semantic checks are declared as `x-unergy-invariants` and
must also be enforced by a semantic validator; standard JSON Schema alone does
not implement those checks. Shared Python/ROM accept-and-reject examples live in
`schemas/microduck-v1-portability-fixtures.json`.

Create a short continuous task, renew it once, then cancel it:

```bash
TASK=11111111111111111111111111111111
curl --fail --silent -H "$AUTH" -H 'Content-Type: application/json' \
  -X POST http://127.0.0.1:8000/v1/tasks \
  -d "{\"schema\":\"MICRODUCK_SIM_TASK_V1\",\"taskId\":\"$TASK\",\"actionCode\":\"WALK_VELOCITY\",\"bundleVersion\":\"$BUNDLE_VERSION\",\"bundleDigest\":\"$BUNDLE_DIGEST\",\"parameters\":{\"vxMps\":0.0,\"vyMps\":0.0,\"yawRateRadps\":0.0},\"scenario\":{\"terrain\":\"flat\",\"seed\":7},\"leaseMs\":500,\"requestedBy\":\"operator-smoke\"}"

curl --fail --silent -H "$AUTH" -H 'Content-Type: application/json' \
  -X PUT "http://127.0.0.1:8000/v1/tasks/$TASK/command" \
  -d '{"commandSequence":1,"parameters":{"vxMps":0.0,"vyMps":0.0,"yawRateRadps":0.0},"leaseMs":500}'

curl --fail --silent -H "$AUTH" -X POST \
  "http://127.0.0.1:8000/v1/tasks/$TASK/cancel"
```

Task events are returned in bounded pages. `afterSequence` defaults to `-1` so
the first request includes lifecycle sequence `0`; `pageSize` defaults to 100
and must be from 1 through 100. The cursor range is `-1` through the signed
64-bit maximum (`9223372036854775807`). Continue with the last returned sequence
until the `events` array is empty. The exclusive cursor prevents duplicates:

```bash
curl --fail --silent -H "$AUTH" \
  "http://127.0.0.1:8000/v1/tasks/$TASK/events?afterSequence=-1&pageSize=100"
```

One fail-closed readiness predicate governs executable catalog availability,
new task creation, and continuous command renewal. If the watchdog fails,
`/v1/ready` reports `ready:false` with `WATCHDOG_UNHEALTHY`, installed catalog
entries are masked unavailable, and create/command return HTTP 503 `NOT_READY`.
An active continuous owner is immediately zeroed/stopped and durably becomes
`FAILED / WATCHDOG_FAILURE`. Authenticated cancellation and diagnostic task,
status, and event reads remain available. Catalog and robot status also
document HTTP 503 `NOT_READY` for startup states in which their required
installed bundle or runtime is absent.

Every runtime call is supervised outside the service ownership mutex by one
long-lived daemon worker, a bounded admission queue, and a monotonic deadline.
If sampling, validation, start, command, status, or safe stop stalls, the
dispatcher rejects all further work, readiness fails closed as
`RUNTIME_UNRESPONSIVE`, active ownership is durably terminalized as `FAILED`,
and one lock-independent emergency zero/disable attempt is made. Emergency stop
publishes fatal/zero intent before attempting the native-data guard and never
waits for that guard. Late runtime returns have no task authority and cannot
republish command or ownership. Diagnostic reads and cancellation remain
available from durable/cached state. At most one native runtime call can remain
hung; if it cannot return/join or the direct zero/disable guard is unavailable,
do not reuse that process: restart the process/container before accepting any
further motion.

To smoke the deadman, create a distinct continuous task with `"leaseMs":200`,
send no renewal, wait more than 200 ms, and query it:

```bash
DEADMAN_TASK=22222222222222222222222222222222
curl --fail --silent -H "$AUTH" -H 'Content-Type: application/json' \
  -X POST http://127.0.0.1:8000/v1/tasks \
  -d "{\"schema\":\"MICRODUCK_SIM_TASK_V1\",\"taskId\":\"$DEADMAN_TASK\",\"actionCode\":\"WALK_VELOCITY\",\"bundleVersion\":\"$BUNDLE_VERSION\",\"bundleDigest\":\"$BUNDLE_DIGEST\",\"parameters\":{\"vxMps\":0.0,\"vyMps\":0.0,\"yawRateRadps\":0.0},\"scenario\":{\"terrain\":\"flat\",\"seed\":11},\"leaseMs\":200,\"requestedBy\":\"operator-deadman-smoke\"}"

sleep 0.4
curl --fail --silent -H "$AUTH" \
  "http://127.0.0.1:8000/v1/tasks/$DEADMAN_TASK"
```

The durable terminal state must be `TIMED_OUT`; continuous renewal cessation
never means success.

Discrete smoke is catalog-driven. `STAND` has exact SitStand-family runtime
semantics: `TRAINED_SITTING` reset, fixed `SIT_FLAG_ZERO` posture goal, fall
handling, and sustained `STAND_POSE_SETTLED` completion. It is `AVAILABLE` only
when the mounted bundle includes the matching SitStand ONNX/model and a passing
governed STAND result. The deterministic nonrestricted handoff fixture
demonstrates an authenticated `STAND` request progressing
`ACCEPTED` → `RUNNING` → `SUCCEEDED`; it does not claim a production robot
checkpoint. Other discrete actions such as `SPIN` remain unavailable until
their own exact artifacts and semantics pass qualification, and return the
stable `ACTION_UNAVAILABLE` error.

## 5. Persistence, restart, and backup

The SQLite database and related state live under `/state`. For the simplest
consistent backup, stop the container, archive the whole host state directory,
then restart with the same mounts and token:

```bash
docker stop microduck-rom-sim
tar -C ../release -czf "state-backup-$(date -u +%Y%m%dT%H%M%SZ).tgz" state
```

On process restart, tasks that were in a nonterminal state are reconciled to
`UNKNOWN`; the simulator never assumes physical/simulation motion continued
across process death. Clients must inspect the task and create a new intent.

## 6. License boundary and troubleshooting

The Python code and container packaging are Apache-2.0; the image includes the
repository `LICENSE`. Repository 3D model
files are CC BY-SA-NC and are deliberately excluded from the image. A bundle
mount is a separately reviewed distribution unit: include only assets and
attributions whose redistribution is authorized for the target use. The
deterministic handoff fixture uses a minimal test MJCF, embeds a declared
Apache-2.0 license artifact, and contains no production STL or `.part` file.

Stable operator signals:

| Signal | Meaning / action |
|---|---|
| `BEARER_TOKEN_MISSING` | Configure a nonempty secret through an env file or secret manager. |
| `BUNDLE_UNAVAILABLE` | Mount an extracted promoted bundle at `/bundle`; verify manifest and artifact bytes. |
| `QUALIFICATION_UNAVAILABLE` | Candidate, missing, forged, duplicate, mismatched, or partial qualification data was rejected; mount the exact promoted output. |
| `STATE_DB_UNAVAILABLE` | Make `/state` writable by UID/GID 10001 and verify free space. |
| `RUNTIME_UNAVAILABLE` | Bundle verification passed but model/policy/runtime preflight failed; rebuild from exact governed artifacts. |
| `RUNTIME_UNRESPONSIVE` | A runtime call exceeded its monotonic deadline; inspect resource/runtime failure and restart the process/container before motion. |
| `WATCHDOG_UNHEALTHY` | Restart and investigate host scheduling/resource pressure before accepting motion. |
| `ACTION_UNAVAILABLE` | Inspect the catalog reason; do not bypass qualification or runtime support. |
| `BUNDLE_MISMATCH` | Refresh catalog identities and recreate the request against the installed release. |
| `TIMED_OUT` | Lease renewal ceased; this is the expected safe stop, not success. |
| restart `UNKNOWN` | Reconcile client intent; never resume an old task implicitly. |

CLI errors are intentionally concise and omit paths, model contents, and
tracebacks. Preserve the candidate, release configuration, promoted digest, and
sanitized API transcript for audit; never preserve the bearer value.

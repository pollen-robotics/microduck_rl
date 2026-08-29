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
outside `src/mjlab_microduck/robot/microduck`; the qualification CLI refuses to
write beneath its source bundle directory.

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

Example `release.json` for the currently supported flat walking runtime:

```json
{
  "schema": "MICRODUCK_ROM_RELEASE_V1",
  "release": "1.0.0",
  "createdAt": "2026-08-29T12:00:00Z",
  "runtimeSourceCommit": "0123456789abcdef0123456789abcdef01234567",
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
        "maxLimitViolations": 0,
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

The qualification report contains bounded per-seed success, fall, tracking,
distance, normalized-action energy proxy, limit-violation, max-action, and
action-specific metrics, with timestamps and exact simulator, runtime, model,
policy, source, checkpoint, and run identities. Its
`subjectBundleDigest` is the digest of the verified candidate bytes actually
executed. The promoted manifest and promoted digest then cover that report as an
artifact. This `VERIFIED_INPUT_BUNDLE_DIGEST_V1` rule avoids an impossible
self-referential digest: the report never claims to contain the final ZIP digest.

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

The Dockerfile-specific ignore file and root-context `.dockerignore` exclude
all `*.part` files and MicroDuck STL assets. The image contains no production
or distribution-restricted model bytes; runtime model resolution is exclusively
from the verified `/bundle` mount.

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

To smoke the deadman, create another continuous task with `"leaseMs":200`, do
not send a renewal, wait more than 200 ms, and query it:

```bash
sleep 0.4
curl --fail --silent -H "$AUTH" \
  "http://127.0.0.1:8000/v1/tasks/22222222222222222222222222222222"
```

The durable terminal state must be `TIMED_OUT`; continuous renewal cessation
never means success.

Discrete smoke is catalog-driven. Submit a discrete action only when its catalog
entry says `AVAILABLE`. The current conservative runtime intentionally exposes
all discrete actions as unavailable because their reset/completion scenario
semantics are not implemented. A request such as `SPIN` therefore returns the
stable `ACTION_UNAVAILABLE` error; do not reinterpret that as a successful
discrete smoke or claim discrete production support.

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

The Python code and container packaging are Apache-2.0. Repository 3D model
files are CC BY-SA-NC and are deliberately excluded from the image. A bundle
mount is a separately reviewed distribution unit: include only assets and
attributions whose redistribution is authorized for the target use. The
deterministic handoff fixture uses a minimal test MJCF and contains no production
STL or `.part` file.

Stable operator signals:

| Signal | Meaning / action |
|---|---|
| `BEARER_TOKEN_MISSING` | Configure a nonempty secret through an env file or secret manager. |
| `BUNDLE_UNAVAILABLE` | Mount an extracted promoted bundle at `/bundle`; verify manifest and artifact bytes. |
| `STATE_DB_UNAVAILABLE` | Make `/state` writable by UID/GID 10001 and verify free space. |
| `RUNTIME_UNAVAILABLE` | Bundle verification passed but model/policy/runtime preflight failed; rebuild from exact governed artifacts. |
| `WATCHDOG_UNHEALTHY` | Restart and investigate host scheduling/resource pressure before accepting motion. |
| `ACTION_UNAVAILABLE` | Inspect the catalog reason; do not bypass qualification or runtime support. |
| `BUNDLE_MISMATCH` | Refresh catalog identities and recreate the request against the installed release. |
| `TIMED_OUT` | Lease renewal ceased; this is the expected safe stop, not success. |
| restart `UNKNOWN` | Reconcile client intent; never resume an old task implicitly. |

CLI errors are intentionally concise and omit paths, model contents, and
tracebacks. Preserve the candidate, release configuration, promoted digest, and
sanitized API transcript for audit; never preserve the bearer value.

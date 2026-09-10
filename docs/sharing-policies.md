# Sharing Microduck policies on the Hugging Face Hub

A policy is shared as **one public model repo per policy** holding the exported ONNX, a
machine-readable manifest and a model card. Anyone — the Pollen team or the community — can
publish under their own namespace; the Hub tag `microduck-policy` is the registry.

```
<namespace>/microduck-<name>          # model repo, public by default
├── policy.onnx                       # obs[1,61] f32 → actions[1,14] f32, obs normalizer baked in
├── manifest.json                     # schema below — what the policy is, what it reads, where it came from
├── README.md                         # model card (front-matter tags + the sections below)
├── media/preview.mp4                 # optional, one short sim clip
└── control.py                        # optional: one-file driver for a physical duck over robotd's IPC (see below)
```

Publish with `scripts/publish_policy.py` — it validates the ONNX against the daemon contract,
writes the manifest and the card from `scripts/policy_card_template.md`, and uploads. Never
publish checkpoints (`model_*.pt`), hand-converted ONNX, or source tarballs into a policy repo;
those stay in the private training-run repos.

## Find and fetch

```bash
# every shared policy, any namespace
hf models ls --filter microduck-policy          # or: HfApi().list_models(filter="microduck-policy")
# get one
uv run scripts/publish_policy.py fetch RemiFabre/microduck-flamingo-cycle --to /tmp/flamingo
# → /tmp/flamingo/policy.onnx (re-validated) + manifest.json; then:
uv run scripts/infer_policy.py --walking /tmp/flamingo/policy.onnx
```

## The contract (what the robot daemon accepts)

Enforced by `robotd` at load (`duck-control/src/policy.rs`) and by `publish_policy.py validate`:

- input tensor named `obs`, f32, trailing dim **61**; first output f32, trailing dim **14**
- observation normalizer baked in (`scripts/export.py` does this; in-sim play hides its absence)
- obs layout: 48 proprioception + 13-D command `[twist(3), head_pose(4), body_pose(6)]`
- the daemon only feeds command slots twist 48–50, head 51–54, body z/roll/pitch 57–59;
  body x/y (55, 56) and body yaw (60) are always 0 on the robot
- actions are joint-position targets around HOME, unfiltered, 50 Hz

## `manifest.json` (schema_version 2)

Superset of the studio's schema v1 (`microduck_rl_studio/rl_space/contract.py`), adding the
command contract and provenance. Extend it **with the daemon team**, not unilaterally.

| field | meaning |
|---|---|
| `schema_version` | 2 |
| `model_api` | 1 (the 61/14 contract) |
| `name` | short slug, same as the repo suffix |
| `kind` | `perpetual` (runs until told otherwise: walking, standing, flamingo) or `episodic` (runs `duration_s` then returns: kick, roulade) |
| `obs_len`, `action_len` | 61, 14 |
| `action_scale` | multiplier the daemon applies to the network output (1.0 for the current envs) |
| `entry_pose` | state the robot must be in when the policy takes over (`standing`, `sitting`, `lying_face_down`, …) |
| `duration_s` | episodic only |
| `command.twist` | 3 strings: meaning of each twist slot (`"unused"` if ignored) |
| `command.head`, `command.body` | `"unused (zeros)"` or the meaning |
| `command.idle` | the 3 twist values that mean "do nothing" — the daemon's rest state |
| `robot` | `{model, hw_rev, servos, control_hz}` |
| `training` | `{task_id, repo, commit, run}` — enough to retrain or continue |
| `eval` | free-form: what was measured in sim and the known limits |
| `description` | one sentence |

## The card (`README.md`)

Front-matter (this is what makes the policy findable):

```yaml
---
library_name: onnx
pipeline_tag: reinforcement-learning
tags: [microduck, microduck-policy, mjlab, robotics]
license: apache-2.0
---
```

Sections, in this order: **What it does** · **Command** (a table of the twist/head/body slots) ·
**Known limits** (honest, measured: "falls on backward pushes ≥ 0.18 m/s") · **Try it in sim** ·
**On the robot** (`deploy/robotd.toml` `[policy]` path, `entry_pose`) · **Provenance** (task id,
training repo + commit, run) · **Files**.

## Publishing

```bash
cd microduck_rl
uv run scripts/publish_policy.py validate /path/to/policy.onnx
uv run scripts/publish_policy.py publish /path/to/policy.onnx \
    --name flamingo-cycle --namespace RemiFabre \
    --manifest /path/to/manifest.json \            # the fields above; the script fills obs/action lens, checks the rest
    --card-extra /path/to/card_body.md \           # your "What it does / Command / Limits / …" sections
    --media /path/to/preview.mp4 \
    --public                                       # default is private; --public is the intended setting for sharing
```

The script refuses to publish an ONNX that fails the contract, and it never uploads anything
but `policy.onnx`, `manifest.json`, `README.md` and `media/*`.

## In sim

`scripts/infer_policy.py` is the keyboard driver (`--walking`, `--standing`, `--sitstand`, `--flamingo`, …);
a policy with a new command scheme gets a mode there (a few lines: session, command-block encoding, a key).
On macOS the MuJoCo viewer must be launched with `uv run mjpython scripts/infer_policy.py …`; if that fails with
`Library not loaded: @rpath/libpython3.12.dylib`, symlink the interpreter's `libpython3.12.dylib` into the venv root
(`ln -s "$(.venv/bin/python -c 'import sys; print(sys.base_prefix)')/lib/libpython3.12.dylib" .venv/libpython3.12.dylib`).
A card's "Try it in simulation" recipe must be complete from a fresh laptop: clone + branch, `uv sync`, fetch the
policy, the exact command, the keys. Keep the robot recipe in a separate block.
Headless timelines / pushes / videos: the training workspace's `notes/tools/duck_rollout.py`.

## On the robot

The daemon does not read the manifest yet: point a `[policy]` role in `deploy/robotd.toml` at the
downloaded `policy.onnx` (absolute path) and restart `robotd`. The `[policy]` roles are what the
daemon knows how to drive (walk, stand, sitstand, ground_pick, kicks, roulade …); a policy with a
new command scheme eventually needs a matching role in the daemon. The updater's planned `model`
component (`microduck/policies/README.md`) is meant to install exactly these repos.

**Testing a twist-driven policy without touching the daemon.** `robot.move {vx, vy, vyaw}` is written
verbatim into the network's twist slots (only an EMA and a 500 ms deadman in between, no magnitude
clamp), and `stand = "none"` removes the "twist ≤ 0.05 → standing network" switch. So a policy whose
command lives in the twist slots can run under the `walk` role:

```toml
[control]
cmd_alpha = 1.0            # pass the flag through unsmoothed (default 0.2 = a 0.4 s ramp through fractional values)

[policy]
walk = "/home/radxa/policies/flamingo/policy.onnx"
stand = "none"             # no "twist ≤ 0.05 → standing network" diversion
sitstand = "none"          # no skill may take the command block over
ground_pick = "none"
kick_left = "none"
kick_right = "none"
roulade = "none"

[safety]
# limp_fall stays on: its sequence holds the twist at zero, which for a posture policy is "stand".
# But its "already tilted" gate is gravity-z −0.90 (≈ 26°) and a one-foot hold leans ≈ 24°: widen it
# so a wobble in the hold is not mistaken for a fall.
limp_fall_tilt_z = -0.80   # ≈ 37°
```

then drive it with the repo's optional `control.py` (JSON-RPC over `/run/robotd.sock`, forwarded to a
laptop with `ssh -L /tmp/robotd.sock:/run/robotd.sock radxa@<robot>`). A `control.py` must: take
`--socket`, send `robot.enable {on: true}`, resend its command faster than the deadman (10 Hz), and put
the robot back into the policy's idle command for a couple of seconds before exiting. It is a test aid,
not a product: no pad, no supervision beyond what `robot.state` reports.

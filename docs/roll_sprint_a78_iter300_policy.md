# A78 iteration 300 roll-sprint policy

## Saved result

A78 iteration 300 is the preserved policy behind the deterministic race preview recorded on 2026-08-31 at 21:37:07 Europe/Berlin. The user selected this preview as the preferred visible behavior.

This snapshot is intentionally classified as a **saved visual candidate**, not the formal 10 m champion. It shows stronger roll, self-right, and reroll behavior, but its strict evaluator still reports insufficient distance and excessive road drift.

Checkpoint 1000 later became the A78 race leader with 3 of 4 valid 10 m finishes and zero shared-road exits. It is documented separately in `docs/roll_sprint_a78_iter1000_race_leader.md`; this checkpoint-300 archive remains the user-selected visual baseline.

| Identity | Value |
|---|---|
| Policy ID | `a78-iter300-2794ff28ed96` |
| Task | `Mjlab-Roll-Sprint-Flat-MicroDuck` |
| Checkpoint | `artifacts/training/roll-sprint-saved-policies/a78-iter300-2794ff28ed96/model_300.pt` |
| Checkpoint SHA-256 | `2794ff28ed960dae27615c61c2a526202b8cf68ea2991d7b5c6fa9da60c98654` |
| Original run | `2026-08-31_21-00-47_a78_rearm2_champion_8192x4000_seed57` |
| Policy code commit | `4599dfd153a1636ed497e7c66cf84d0b870c7c58` |
| Warm-start checkpoint | `artifacts/training/roll-sprint-champion/model_0.pt` |
| Warm-start SHA-256 | `9983a45438be8b32da89a83892ff3eadee1ea41f8dec324aa10b4c3f7e64398b` |
| Selected preview | `artifacts/training/roll-sprint-saved-policies/a78-iter300-2794ff28ed96/race-preview.mp4` |
| Manifest | `artifacts/training/roll-sprint-saved-policies/a78-iter300-2794ff28ed96/manifest.json` |

The archived checkpoint is a complete PPO snapshot. It contains the actor, critic, observation normalizer, action distribution, optimizer, and PPO state. It is suitable for another controlled warm start through `scripts/train_roll_sprint.py`. It is not an ONNX deployment artifact.

## Policy contract

- Actor observation: 61 dimensions.
- Critic observation: 90 dimensions.
- Action output: 14 joint-position actions.
- Actor network: MLP `(512, 256, 128)` with ELU activations.
- Observation normalization: enabled and stored in the checkpoint.
- Shared command block: `twist(3), head_pose(4), body_pose(6)` after the 48D proprioceptive block.
- `twist[0]`: self-righting mode flag.
- `twist[1]`: signed shared-road return command while repositioning.
- `twist[2]`: bounded correction toward the frozen reset heading.

The 61D actor and 14D action layouts must remain unchanged for warm starts and future sim2real export.

## Behavior state machine

The intended autonomous loop is:

```text
ROLL -> SELF-RIGHT OR RECOVER -> REPOSITION IF NEEDED -> REARM AT PHASE ZERO -> REROLL
```

A valid full forward roll requires supported signed sagittal rotation, full-cycle eligibility from phase zero, a valid flat head-top contact, and no side or shoulder invalidation. Completing a valid roll enters recovery-required state. No second cycle or roll-linked distance is eligible until recovery or self-righting rearms the policy.

Launch-ready recovery requires foot support, released head-top contact, trunk tilt within 25 degrees of upright, trunk height at least 0.09 m, and absolute forward angular rate no greater than 6 rad/s. Self-right completion holds those conditions for five 50 Hz steps, or 0.10 seconds. The regular post-recovery rearm uses two consecutive steps so the policy can dynamically reroll instead of waiting through a redundant five-frame dwell.

An accidental stalled fall enters self-righting only after ground support, at least 60 degrees trunk tilt, less than 1 rad/s forward rotation, and 0.30 seconds without new roll-phase progress. An actively progressing roll does not enter self-righting.

The global forward frontier survives recovery and repositioning. Partial rotation and translation from an interrupted cycle are discarded.

## Distance and road objective

Distance is released only when a valid roll cycle completes. The credited distance is bounded by all three quantities:

1. The cycle's supported rotation-linked translation budget.
2. The positive net projected forward displacement from cycle start to cycle end.
3. The positive extension beyond the globally credited forward frontier.

This makes backward translation reduce the cycle result, gives zero reward for revisiting covered ground, and prevents positive-velocity integration, rocking, sliding, or back-and-forth motion from farming distance.

The course is a fixed 1.12 m-wide shared road with half-width 0.56 m and a safe full-reward half-width of 0.42 m. Internal visual lane crossings are allowed. Repositioning starts near the road edge and targets the nearest safe edge, not the center. Heading correction is always measured relative to the frozen reset heading.

At iteration 300, the main resolved reward weights included roll-linked frontier distance `32`, dense rotation progress `1.5`, valid cycle rate `1`, recovery rate `1`, reposition completion `2`, recovered reroll `4`, self-right upright potential `5`, self-right height potential `30`, upward bootstrap `1`, fallen tax `-0.25`, and one-shot self-right success `1`. Recovery shaping is active only during self-righting and does not pay an upright or standing annuity.

## Training lineage

The A78 run started from the retained A72 checkpoint, not an interrupted A77 policy.

| Parameter | Value |
|---|---:|
| Parallel environments | 8192 |
| Requested PPO iterations | 4000 |
| Saved snapshot iteration | 300 |
| Seed | 57 |
| Learning rate | `3e-6` |
| Save cadence | 100 iterations |
| Steps per environment per iteration | 24 |
| Episode duration | 6 seconds |

The policy-generating environment and state-machine code is commit `4599dfd`. Later commits through `a5686b9` changed only dashboard, video sampling, and their tests, not the trained MDP or policy configuration.

The exact resolved run configurations are archived as `agent.yaml` and `env.yaml`. Their hashes are recorded in `manifest.json`.

## Deterministic evaluation

The stored schema-v8 evaluation used four perfectly aligned robots for a 40-second race plus sixteen recovery starts, four orientations by four seeds.

| Metric | Result |
|---|---:|
| Canonical four-robot alignment | Pass |
| Formal 10 m finishers | 0 / 4 |
| Mean credited forward frontier | 3.666 m |
| Best credited forward frontier | 8.774 m |
| Valid roll cycles | 24 |
| Recovery transitions during race | 23 |
| Recovered-and-rerolled cycles during race | 21 |
| Recovery battery successes | 16 / 16 |
| Recovery battery subsequent rerolls | 8 / 16 |
| Recovery latency mean / p95 | 2.484 s / 5.465 s |
| NaN environments | 0 |
| Out-of-bounds environments | 0 |
| Road-exit environments | 3 / 4 |
| Race p95 lateral drift | 1.445 m |
| Maximum measured forward speed | 1.087 m/s |

The checkpoint did not pass promotion because no robot formally finished 10 m and road drift exceeded the acceptance limit. It remains valuable as a fixed behavior baseline and future warm-start candidate because its visible repeated rolling and self-righting are materially different from earlier failed recovery runs.

## Verify the archive

From the dedicated worktree:

```powershell
Get-FileHash -Algorithm SHA256 `
  artifacts\training\roll-sprint-saved-policies\a78-iter300-2794ff28ed96\model_300.pt
```

The expected lowercase hash is:

```text
2794ff28ed960dae27615c61c2a526202b8cf68ea2991d7b5c6fa9da60c98654
```

Re-run the deterministic audit without modifying training:

```powershell
uv run python scripts\evaluate_roll_sprint_checkpoint.py `
  artifacts\training\roll-sprint-saved-policies\a78-iter300-2794ff28ed96\model_300.pt `
  --num-envs 4 `
  --duration 40 `
  --device cuda:0 `
  --parent-frontier-m 8.949602127075195 `
  --output artifacts\training\roll-sprint-saved-policies\a78-iter300-2794ff28ed96\evaluation-rerun.json
```

Record another deterministic preview on CPU:

```powershell
uv run python scripts\record_roll_sprint_policy.py `
  artifacts\training\roll-sprint-saved-policies\a78-iter300-2794ff28ed96\model_300.pt `
  artifacts\training\roll-sprint-saved-policies\a78-iter300-2794ff28ed96\race-preview-rerun.mp4 `
  --steps 1000 `
  --frame-stride 3 `
  --output-fps 60 `
  --seed 0 `
  --device cpu `
  --width 1280 `
  --height 720
```

Start a new controlled warm-start run from the saved checkpoint only after the normal focused tests and mandatory 64-environment, 5-iteration smoke test:

```powershell
uv run python scripts\train_roll_sprint.py `
  --source-checkpoint artifacts\training\roll-sprint-saved-policies\a78-iter300-2794ff28ed96\model_300.pt `
  --num-envs 8192 `
  --iterations 4000 `
  --seed 58 `
  --learning-rate 3e-6 `
  --save-interval 100 `
  --run-name a79_from_saved_a78_iter300
```

`train_roll_sprint.py` preserves the complete PPO state but deliberately resets the iteration and common environment-step counters for the new run. Do not call this exact warm-start operation an uninterrupted resume.

## Deployment status

This snapshot is simulation-first and has not passed the 10 m race, shared-road, or sim2real gates. No exact checkpoint-300 ONNX export is archived. Before hardware use, select a formally promoted checkpoint, export it through `scripts/export.py` so the observation normalizer is baked into ONNX, and run the CPU deployment rehearsal.

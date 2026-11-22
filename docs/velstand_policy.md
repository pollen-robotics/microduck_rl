# VelStand policy — how the current one was trained

One policy that walks, stands still at zero command, falls protectively and gets back up, with
the head steerable, (from v7) the body steerable while standing, and (from v8) a smooth, reliable
get-up and yaw commands that are actually tracked. Task
`Mjlab-VelStand-Rough-Backlash-MicroDuck`, cfg
`src/mjlab_microduck/tasks/microduck_velstand_env_cfg.py` (its module docstring and the comments
on each `ENABLE_*` toggle hold the design and lesson arc), BC machinery in
`src/mjlab_microduck/tasks/distill.py`.

**Current policy: wandb `pollen-robotics/mjlab_microduck/j4i6yoq2` @ `model_1250.pt`** —
`velstand.onnx` in Hub `pollen-robotics/microduck-policies`, tag **v9** (2026-11), which is
**the v7 file re-published unchanged**. v8 (`0045i2si` @ 2749, 2026-10-12) turned out worse than
v7 on the robot and was rolled back; v9 exists only so the robot-side policy updater sees a newer
version instead of a downgrade.

**Newest policy (2026-11-10): v7 + yaw remap = wandb `pollen-robotics/mjlab_microduck/n3fg9pgq` @
`model_2000.pt`** (Hub `apirrone/velstand_v7_yaw_remap`, not yet in the official set). On the robot
it behaves like v7 and turns better. See "Stage v7 + yaw remap" below.

**This checkout trains the v7 + yaw remap recipe**: `ENABLE_ANCHOR_YAW_REMAP` ON, the two v8 get-up
pieces (`ENABLE_JOINT_MASKED_TEACHER`, `ENABLE_WIDE_SPAWNS`) OFF. With all three off, every env cfg
(flat / rough, train / play) is identical to commit `454ddb91` (itself checked identical to
`53fb7d1`, the commit that trained v7) by full cfg dump; the runner cfg differs only by no-op
bc_cfg defaults and the teachers being loaded from their Hub training bundles (same checkpoints).
Turn all three on to reproduce v8.

## Lineage

Up to v7 every stage is a **warm start** (`MICRODUCK_WARM_START=1` + `--agent.resume True` on
another run's checkpoint): weights, obs normalizer and optimizer are kept, `common_step_counter`
and the iteration restart at 0 so velstand's own curricula (fall-over disable, prone/crouch
spawns, recovery economics, protection costs, topple pushes) run again from stage 0. Inherited
velocity-recipe curricula are collapsed to their final stage (`WARM_START = True`).
The v8 stages are **plain resumes** (no `MICRODUCK_WARM_START`): the step counter continues, so
every curriculum stays at its final stage — the policy is fine-tuned, not re-taught.

| stage | run @ ckpt | code | what it added |
|---|---|---|---|
| walk teacher | `441tzs6d` @ 3750 | velocity recipe | `alpha_walking.onnx` — the walk the student starts from and the walk BC anchor |
| stand teacher | `69u48n8l` @ 9750 | standup env | `alpha_stand.onnx` — the get-up BC expert (and, from v7, the body-tilt teacher) |
| velstand v1 | `6op8a8u8` @ 5999 | `d3edc1f` (tag `velstand-best-6op8a8u8`) | protective fall + get-up via expert BC + walk anchor, flat |
| v5/v6 prod | `fhathosb` @ 3750 | `e40fbda`, branch `protective_fall` | same recipe on Rough + Backlash, rough-terrain spawn fix |
| **v7 prod = v9** | **`j4i6yoq2` @ 1250** | `53fb7d1`, branch `improve_velstand2` | standing body control (below) |
| v8 get-up | `qz5g3bwi` @ 1750 | `333ef7b`, branch `improve_standup_velstand` | joint-masked stand teacher + wide get-up spawns (below); robot-tested, not published |
| v8 (rolled back) | `0045i2si` @ 2749 | `3750776`, branch `improve_standup_velstand` | walk anchor fed a calibrated yaw remap + fine yaw reward (below); Hub v8, replaced by v9 = v7 |
| **v7 + yaw remap** | **`n3fg9pgq` @ 2000** | branch `velstand_v7_yaw_remap` | v8b's anchor yaw remap only, plain resume from v7 (below) |

## Stage v7 + yaw remap — `n3fg9pgq` (2026-11-10)

| | |
|---|---|
| host | HF Jobs, `rtx-pro-6000` |
| start | `j4i6yoq2` @ `model_1250.pt`, **plain resume** (curricula stay final) |
| checkpoint used | **`model_2000.pt`** (run ended at 2249) — robot-tested 2026-11-10: like v7, turns better |

```bash
uv run train Mjlab-VelStand-Rough-Backlash-MicroDuck --env.scene.num-envs 4096 --agent.resume True \
  --wandb-run-path pollen-robotics/mjlab_microduck/j4i6yoq2 --wandb-checkpoint-name model_1250.pt \
  --agent.max_iterations 1000
```

### Why

v8 turned much better than v7 on the robot but was rolled back for another reason, so the
v8 get-up changes (v8a) are the suspects and the yaw remap (v8b) is not. This stage applies only
`ENABLE_ANCHOR_YAW_REMAP` (anchor yaw remap + `track_yaw_fine` + turn-in-place bucket ≥ 0.2 rad/s)
on top of v7.

A re-run of the v7 stage (warm start from `fhathosb@3750`, run `cg9ssbo9`, 4000 iterations)
showed that **training longer does not fix yaw**: in-place 1.0 rad/s stays ≤ v7's ≈ 0.45 at
every checkpoint up to 3000, 0.6 stays ≈ 0, and the left/right bias and straight-walk drift
wander from checkpoint to checkpoint. It is capped by the walk anchor, not by training time.

### Sim eval (`scripts/velstand_bench.py`, flat + backlash, 64 envs per case, 128 for push)

| | v7 `j4i6@1250` | Hub v8 `0045i2si@2749` | **`n3fg9pgq@2000`** | `n3fg9pgq@2249` |
|---|---|---|---|---|
| in place ±1.0 / ±0.6 rad/s | +0.48 / −0.41 · ≈ ±0.08 | +0.96 / −0.89 · ±0.56 | **+0.88 / −0.94 · +0.35 / −0.54** | +0.87 / −1.01 · +0.42 / −0.58 |
| vx 0.2, ±0.6 / ±1.0 | +0.33 / −0.20 · +0.53 / −0.41 | ≈ ±0.56 · ≈ ±0.9 | **±0.55 · +0.91 / −0.95** | ±0.55 · +0.91 / −1.01 |
| straight-walk drift | ≤ 0.05 | ≤ 0.06 | ≤ 0.01 | ≤ 0.02 |
| rise HOME fd / fu / side | 100 / 98 / 100 | 100 / 86 / 92 | 100 / 95 / 100 | 100 / 95 / 100 |
| rise rj fd / fu / side | 100 / 88 / 91 | 98 / 95 / 94 | 98 / 88 / 95 | 97 / 78 / 83 |
| push 0.8 / 1.0 / 1.2 m/s, up at end | 98 / 95 / 98 | 100 / 97 / 97 (64 envs) | 96 / 94 / 91 | — |

Checkpoint choice: get-up reliability swings between checkpoints of this run (1750: HOME face-up
80 %, 2249: rj face-up 78 %); 2000 is the one at v7's level on every family. Slow in-place
turning (0.3 rad/s) is still ≈ 0, and +0.6 in place is weaker to the left.

## Stage v7 — `j4i6yoq2` (standing body control)

| | |
|---|---|
| host | COACH2, RTX 5090 |
| start | `fhathosb` @ `model_3750.pt`, warm start |
| checkpoint used | **`model_1250.pt`** (run ended at 1499; see checkpoint choice) |

```bash
MICRODUCK_WARM_START=1 uv run train Mjlab-VelStand-Rough-Backlash-MicroDuck --env.scene.num-envs 4096 \
  --agent.resume True --wandb-run-path pollen-robotics/mjlab_microduck/fhathosb \
  --wandb-checkpoint-name model_3750.pt --agent.max_iterations 1500
```

### Why

The v5/v6 velstand ignored body commands completely (0 mm / 0° response, measured with
`claude_experiments/velstand_body_probe.py`): the velocity recipe keeps `body_pose` at reward
weight 0 with ±5 mm / ±0.05 rad "alive" ranges, and the walk anchor pins standing to
`alpha_walking`, which never learned it. The old pipeline's `alpha_stand` tracks roll/pitch
(9.2–9.5° for 10°) but not z.

### What changed (all behind `ENABLE_BODY_CONTROL`)

| piece | what |
|---|---|
| `StandingGatedPoseCommand` (mdp.py) | body command = sampled value × `twist.is_standing_env` → exact zero while walking / turning, as the runtime sends; 30 % all-zero bucket (plain standing stays trained); z zeroed on half the resamples so tilt-only commands exist |
| ranges | roll/pitch ±6° → **±12°** at iter 300 (≈ `alpha_stand`'s trained range), z −2.5 / +1 cm, x/y/yaw tiny (inside `alpha_stand`'s normalizer) |
| distill body routing (`body_slice`) | standing frames with an active body command leave the walk anchor; **tilt-only → `alpha_stand` BC**, z frames → PPO only (`alpha_stand` cannot crouch). The command is in the obs, so these frames are separable (no teacher leak) |
| `body_pose_tracking` | standing-gated z/roll/pitch Gaussians (z std 1 cm, angle 5°, weight 2, nominal z 0.116 = measured standing trunk height) |
| `upright_body_cmd_relative` | stock `upright` measured against the commanded tilt (identical at zero command) |
| `variable_posture_body_relaxed` | looser leg-pose stds while a body command is active (hip_roll 0.05 → 0.25 etc.; the tight stand std priced a 10° roll at −0.85/step) |

Reward check before training (per step, standing, same env): tracking a 10° tilt beats ignoring it
by +1.35 (pitch), +1.57 (roll), +2.1 (both); zero command unchanged.

### Sim eval at the chosen checkpoint (flat + backlash, vs v6 `fhathosb@3750`)

| | v7 `j4i6yoq2@1250` | v6 `fhathosb@3750` |
|---|---|---|
| body roll ±10° / pitch ±10° | +8.4 / −8.2° · +9.8 / −7.0° | 0 (ignored) |
| body z −2 cm | ≈ 0 (not learned) | 0 |
| turn in place +1.0 / −1.0 rad/s | 0.45 / −0.37 | 0.12 / −0.19 |
| straight-walk yaw drift (0.3 / 0.4 m/s) | 0.05 / 0.03 rad/s | −0.03 / −0.07 |
| push falls, 0.3 m/s (fwd 0.35 / backward / turning) | 0 / 9 / 4 % | 1 / 12 / 9 % |
| get-up after push fall (1.0 / 1.2 m/s) | 93 / 95 % | 91 / 85 % |

Tools: `velstand_body_probe.py`, `velstand_walk_stability.py`, `velstand_improve_bench.py`
(`--only walk|rise`), `velstand_ckpt_eval.sh <run> <ckpt> <ref.onnx>` (all in `claude_experiments/`,
local, gitignored).

### Checkpoint choice

Checkpoints differ a lot in yaw behaviour although the training metrics do not: `500` turns
right when asked to walk straight (−0.20 rad/s at 0.4 m/s), `1499` turns left (+0.27). `1250`
has neither drift. Always run the walk sweep before picking a checkpoint.

## Stage v8a — `qz5g3bwi` (smooth, reliable get-up)

| | |
|---|---|
| host | COACH2 |
| start | `j4i6yoq2` @ `model_1250.pt`, **plain resume** (curricula stay final) |
| checkpoint used | **`model_1750.pt`** (run ended at 2749) — robot-tested 2026-10-08, base of v8b |

```bash
uv run train Mjlab-VelStand-Rough-Backlash-MicroDuck --env.scene.num-envs 4096 --agent.resume True \
  --wandb-run-path pollen-robotics/mjlab_microduck/j4i6yoq2 --wandb-checkpoint-name model_1250.pt \
  --agent.max_iterations 1500
```

### Why

v7 froze on its back on ~26 % of face-up starts with random joints (`rj_fu`, battery below) and the
rise was violent (|ω| p50 ≈ 11 rad/s face-up). Root cause: an attractor in the **stand teacher**
`alpha_stand`, copied by BC — in some leg configurations (hip_roll ≈ −0.8 / −0.7, head_yaw ≈ −1.6) the
teacher's label is "stay there", and the student agrees. Masking only the head obs for the teacher
(run `s4jdgkux`, commit `2ee6dc8`) made it **worse** (`rj_fu` 73 → 55 %): the attractor lives in the
teacher's response to the *leg* configuration. Feeding `alpha_stand` closed loop with **all joint
obs zeroed** above 75° tilt rescues 93–100 % of every family and rises more gently.

### What changed

| piece | what |
|---|---|
| `ENABLE_JOINT_MASKED_TEACHER` | distill `bc_cfg` `expert_mask_cols = range(6, 48)` (joint pos / vel / last action), `expert_mask_tilt_deg = 75`: on frames tilted more than 75° the stand teacher sees HOME joints, so it labels a get-up from the orientation alone; below 75° (mid-rise) it sees everything |
| `ENABLE_WIDE_SPAWNS` | prone slice: random joints (incl. head) on 50 %, uniform SO(3) orientation + random joints 25 %, sit spawns 5 % — the starts the attractor lived in |

Gate for this kind of change = the rise battery plus `--handoff-s` (masked teacher takes over the
student's stuck envs) on the student, not a short local run (`s4jdgkux`'s 400-iteration local check
looked good and the full run failed).

## Stage v8b — `0045i2si` (yaw tracking)

| | |
|---|---|
| host | COACH2 |
| start | `qz5g3bwi` @ `model_1750.pt`, **plain resume** |
| checkpoint used | **`model_2749.pt`** (the last one) — robot-tested 2026-10-12 |

```bash
uv run train Mjlab-VelStand-Rough-Backlash-MicroDuck --env.scene.num-envs 4096 --agent.resume True \
  --wandb-run-path pollen-robotics/mjlab_microduck/qz5g3bwi --wandb-checkpoint-name model_1750.pt \
  --agent.max_iterations 1000
```

### Why

Turning was poor on the robot: in place 0.3 / 0.6 rad/s → ≈ 0, 1.0 → ~0.4; walking ≈ half the
command. The **walk anchor** `alpha_walking` (`441tzs6d` @ 3750) has exactly this curve (fed 0.6 → 0,
1.0 → 0.35 in place; ≈ 0.6× while walking) and BC at coefficient 1.0 copies it. Taking turn frames off
the anchor (`ENABLE_YAW_FIX`, runs `sape62zb`, `brkpcnn8`) lost turning altogether — kept OFF.

### What changed (`ENABLE_ANCHOR_YAW_REMAP`)

| piece | what |
|---|---|
| `bc_cfg["anchor_yaw_remap"]` (distill `remap_anchor_yaw`) | the anchor is fed the yaw that makes **it** achieve the commanded rate: per-sign inverse table `ANCHOR_YAW_REMAP`, bilinear in \|v_xy\| (0–0.4) × target \|wz\| (0.1–1.0), 0 → 0. Only the anchor's input changes; the student's obs keep the true command and the stand teacher's frames are untouched |
| table calibration | `claude_experiments/calibrate_walk_yaw.py`: the anchor .pt run batched on GPU over \|vx\| 0–0.4 × fed ±0.2–2.4 rad/s, monotone inverse of the measured response. Closed loop through the remap (`validate_anchor_yaw_remap.py`): in place 0.5 / 0.7 / 1.0 → 0.56 / 0.68 / 1.07, 0 falls |
| `track_yaw_fine` | yaw-only Gaussian (`track_yaw_rate_fine`, std 0.35, weight 2) — the stock term's std 0.71 has little gradient at small errors |
| turn-in-place bucket | 20 % of twist resamples are turn-in-place with \|wz\| ≥ 0.2 |

## v8 sim eval (flat + backlash, 0.2 = vx 0.2 m/s)

Yaw (`velstand_yaw_probe.py --backlash`, achieved rad/s, mean over upright envs after 2 s):

| command | v7 `j4i6@1250` | v8a `qz5g@1750` | **v8 `0045i2si@2749`** |
|---|---|---|---|
| in place ±0.3 | 0.00 / 0.00 | 0.00 / 0.00 | **+0.08 / −0.22** |
| in place ±0.6 | +0.08 / −0.03 | +0.05 / −0.04 | **+0.52 / −0.57** |
| in place ±1.0 | +0.47 / −0.35 | +0.34 / −0.42 | **+0.97 / −0.87** |
| 0.2, ±0.3 | +0.18 / −0.07 | +0.22 / −0.04 | **+0.25 / −0.32** |
| 0.2, ±0.6 | +0.32 / −0.20 | +0.35 / −0.19 | **+0.54 / −0.56** |
| 0.2, ±1.0 | +0.55 / −0.37 | +0.50 / −0.40 | **+0.92 / −0.88** |

Stand-up (`velstand_rise_battery.py --backlash`, 2 × 128 envs per family, 12 s; `rj` = random
joints, `so3` = uniform orientation, `limp` = limp-fall landings):

| family | v7 rec % / stuck % | **v8 rec % / stuck %** | v7 → v8 face-up \|ω\| p50 |
|---|---|---|---|
| home face-down / up / side | 100 / 100 / 100 | **100 / 100 / 99** | 11.1 → **6.8** rad/s (face-up) |
| rj face-down | 96 / 3.5 | **100 / 0.4** | |
| rj face-up | 73 / **26.6** | **94 / 5.9** | 10.1 → **7.8** |
| rj side | 91 / 9.4 | **96 / 3.5** | |
| so3 home / rj / limp | 100 / 86 / 91 | **100 / 96 / 98** | |
| sit | 89 / 11.3 | **99 / 0.8** | |

Overshoot past vertical p50 4–5° in both; the rise is slower (face-up t_rec p50 1.1 → 1.5 s) and
gentler — what the user saw on the robot.

Walking / push (`velstand_bench.py --backlash --only walking,push`): vx tracking ≈ v7 and
`alpha_walking` (0.4 m/s → 0.17 achieved, same as the teacher); straight-walk yaw drift
−0.02 / +0.03 / +0.08 rad/s at 0.2 / 0.3 / 0.4 m/s (earlier checkpoints of the run drifted left
+0.10–0.13, it settled by 2749); push while walking 0.15 m/s: 0.8 / 1.0 / 1.2 m/s → fell 53 / 90 / 98 %,
**recovered 100 / 100 / 99 %** (v7 93–95 %), t_rec p90 3.2 s.

### Checkpoint choice

`0045i2si` checkpoints differ mainly in yaw symmetry and straight-walk drift: @2000 turned right
~80 % as well as left and drifted left +0.10–0.13 rad/s walking straight; @2749 is near-symmetric
with drift ≤ 0.08. Stand-up was equally good at every checkpoint. As before: always run the yaw
probe and the walk bench before picking.

## Known limits

- **Body z (crouch)** is not learned (≤ 1 mm for −2 cm): the axis has no teacher and 1500
  iterations of PPO did not find it.
- **Slow in-place yaw** (fixed in v8 from 0.5 rad/s up). Turning in place at 0.3 rad/s is still
  weak and bursty (+0.08 / −0.22): the remapped anchor itself has a bimodal onset there. The pad's
  default range is ±1.0 rad/s (`[pad_drive]` in microduck), inside what v8 tracks.
- **Yaw history (v7).** v7's dead zone came from the walk anchor (see v8b). A runtime remap
  `wz_obs = sign(c)·(0.33 + |c|/0.6)` was a sim-only workaround for v7 and is **not** for v8, which
  takes the true command.

## Reproduce / deploy

```bash
uv run scripts/export.py Mjlab-VelStand-Rough-Backlash-MicroDuck \
  --wandb-run-path pollen-robotics/mjlab_microduck/j4i6yoq2 --checkpoint 1250
uv run python claude_experiments/velstand_yaw_probe.py output.onnx --backlash
uv run python claude_experiments/velstand_rise_battery.py output.onnx --backlash
```

Always export through `scripts/export.py` (obs normalizer is baked into the ONNX). The official
set on the Hub is updated by hand (replace `velstand.onnx`, edit only its manifest entry, commit,
`create_tag vN`); `uv run publish` is for single-policy repos.

## Slow gait starts in duck-sim

With set v10 on the backlash scene, the duck can keep walking at 0.15 m/s after
walking at 0.20 m/s, yet barely move when the same 0.15 m/s command starts from
a stand. The same effect occurs in the first gait-onset fit at ±0.5 rad/s.
From a stand, the trunk turned about +8° or −10° in 8 s. One second at ±1 rad/s
before the same slow command was enough to sustain a turn in either direction.
The v8 teacher also has a direction-dependent threshold. In four fresh starts
it turned about +15° at +0.5 rad/s and +242° to +256° at +0.6 rad/s.
[microduck#597](https://github.com/pollen-robotics/microduck/issues/597)

`scripts/fit_velstand_onset.py` builds an experimental actor for this simulator.
During training, v8 gets a stronger yaw command to demonstrate starting a turn.
For straight walks, it steers back toward the initial heading and line using
the simulated trunk pose. Each demonstration pairs the original observation
and requested twist with the resulting action. The fitted actor uses the
usual 61 observations and 14 actions in `robotd`. The training set also includes
v10 examples of holding still, getting up and following other commands.
To preserve the first step at 0.15 m/s, short demonstrations begin from both
SIT and STAND. For their first second the teacher gets a forward cue that ramps
to 0.20 m/s. The fit gives these short examples triple weight and updates only
v10's final linear layer. The ONNX records the hashes of both source policies.

From this checkout, set `duck` to the companion microduck checkout. These commands
build the candidate and start a new simulator state with the other v10 policies.

```bash
export DUCK_SIM_RL="$PWD" DUCK_SIM_STATE="$(mktemp -d)" DUCK_SIM_VIEWER=0
POLICY_VERSION=v10 sh "$duck/scripts/seed-policies.sh" "$DUCK_SIM_STATE/base"
uv run hf download pollen-robotics/microduck-policies velstand.onnx --revision v8 --local-dir "$DUCK_SIM_STATE/v8"
uv run scripts/fit_velstand_onset.py "$DUCK_SIM_STATE/base/current/velstand.onnx" \
  "$DUCK_SIM_STATE/v8/velstand.onnx" "$DUCK_SIM_STATE/candidate.onnx"
mkdir -p "$DUCK_SIM_STATE/policies/current"
cp "$DUCK_SIM_STATE/base/current/"*.onnx "$DUCK_SIM_STATE/policies/current/"
cp "$DUCK_SIM_STATE/candidate.onnx" "$DUCK_SIM_STATE/policies/current/velstand.onnx"
"$duck/scripts/duck-sim" up
"$duck/scripts/duck-sim" drive
"$duck/scripts/duck-sim" down
```

Use a microduck checkout without a local `policies/velstand.onnx`, which takes
precedence over the simulator state. For a turn, start a fresh simulation,
let the duck stand up and run `drive 0 0.5` or `drive 0 -0.5`. Start fresh
for each direction. An exact 0.5 rad/s turn would cover 229° in 8 s. Measure
the accumulated signed heading change to include turns that cross ±180°.

The CPU benchmark runs eleven commands from 12 perturbed starting placements
and tests get-up from 20 placements for each of six postures. It records every
trial, including stalls. Walking distance and lateral drift are measured along
and across the starting heading. A blank time to the first 2 cm means the duck
never reached that distance. For pivots, `end_radius_cm` is the horizontal distance
from the trunk's starting point after 8 s. `max_radius_cm` is the greatest such
distance at any 50 Hz sample, including turns that come back toward the origin.
Run both actors on the same fold for a paired comparison. Use `--fold reserve`
for a second set of placements.

```bash
uv run scripts/bench_velstand_onset.py "$DUCK_SIM_STATE/base/current/velstand.onnx" v10.csv
uv run scripts/bench_velstand_onset.py "$DUCK_SIM_STATE/candidate.onnx" candidate.csv
```

### Slow turns with a head-pose command

Before yaw calibration, the fitted actor turned a median +184° / −275° at
±0.5 rad/s on the original 24 placements. Two positive turns stopped near
+15° and +12°. Setting the head command at the same time as yaw reproduced
both stalls even from a neutral head position. Clearing the head command at
turn onset gave +181° and +174°. In the first placement head roll alone
reproduced the stall. In the second, none of the four axes alone did.

The experimental ONNX uses the norm of the ten leg joint velocities to
strengthen the internal yaw cue while the legs are nearly still. At a requested
+0.5 rad/s, the cue falls from +1.0 to +0.575 rad/s as that norm rises from
0 to 3 rad/s. With a neutral head at −0.5 it changes from −0.73 to −0.305 rad/s.
These values were calibrated in CPU MuJoCo. The correction tapers to zero as
the requested yaw rate moves away from ±0.5 rad/s or the requested linear
speed increases. The simulator's default forward speed is 0.15 m/s.

Use `--head-poses` to exercise this case. Each fold includes 24 independent
starts from STAND, SIT and FOLD, 16 with head offsets. Four additional rows per
command replay an offset placement with a neutral head, marked `paired_control=1`.
Exclude those repeats when computing the 24-placement median. Each fold also
includes 120 get-up trials.

```bash
uv run scripts/bench_velstand_onset.py "$DUCK_SIM_STATE/candidate.onnx" head-primary.csv --head-poses
uv run scripts/bench_velstand_onset.py "$DUCK_SIM_STATE/candidate.onnx" head-reserve.csv --head-poses --fold reserve
```

### Starting at +0.65 rad/s and turning about the starting point

The first yaw calibration (`42c0d3f4`) restored the two +0.5 starts to +232°
and +198°. At +0.65 rad/s, one shifted-head SIT start then turned only +17°
in 8 s (target +298°). To reproduce it, use `--head-poses --fold primary` and
inspect seed `3519081`. As the filtered request rose from +0.514 to +0.580,
the internal yaw cue *fell* from +0.858 to +0.742 rad/s. Both feet stayed
loaded for all 400 ticks. Bypassing the old correction on ticks 4–10 alone
gave +251°. Bypassing it after tick 20 still gave +17°.

Above +0.5 the new correction fades at unit slope, reaching zero at +1.0.
At a given leg speed the internal yaw cue can now stay level or rise as the
request increases. The same SIT start transfers its weight to one foot and
turns +266°.

At −0.5 rad/s the previous actor accumulated about 25 cm of forward motion
in its rotating trunk frame. The new calibration feeds the actor a backward
cue of 0.085 m/s at the center of the right-turn window. Neck pitch and head
pitch affect the turn angle in opposite directions. The difference between
their requested offsets adjusts the internal yaw cue by up to ±0.06 rad/s.
The head command itself still goes to the actor.

Use `--paired-heads` to compare two revisions. Each fold has 30 independent
placements (ten each for STAND, SIT and FOLD). Each placement is run twice,
with a four-axis head offset within ±0.12 rad and with a neutral head. Each
fold also runs 120 get-ups from six postures.

```bash
uv run scripts/bench_velstand_onset.py "$DUCK_SIM_STATE/candidate.onnx" paired-primary.csv --paired-heads
uv run scripts/bench_velstand_onset.py "$DUCK_SIM_STATE/candidate.onnx" paired-reserve.csv --paired-heads --fold reserve
```

The table compares the two calibration revisions `42c0d3f4` and `4136174e`.
Each entry is the median of 60 runs on that fold, including the neutral replays.
Angle error is the absolute difference from the requested angle in each run.

| Eight-second measurement | Primary | Reserve |
| --- | ---: | ---: |
| Angle at +0.65 rad/s (target +298°) | +256° → +285° | +254° → +283° |
| Angle at +0.5 rad/s (target +229°) | +232° → +232° | +232° → +232° |
| Angle at −0.5 rad/s (target −229°) | −230° → −227° | −231° → −227° |
| Absolute angle error at −0.5 rad/s | 5.5° → 3.3° | 7.5° → 3.8° |
| Endpoint distance at −0.5 rad/s | 11.1 → 1.5 cm | 10.6 → 1.7 cm |
| Maximum distance at −0.5 rad/s | 13.3 → 2.3 cm | 13.4 → 2.2 cm |

The reserve fold contained another shifted-head SIT stall at +0.65 rad/s
(seed `162282558`), which improved from +16° to +250°. All 120 final runs
per direction at ±0.5 rad/s reached at least a quarter of the requested turn.
At 0.15 m/s both actors reached 30 cm in 119/120 runs. In the remaining
placement both moved only 0.77 cm. Both actors recovered on all 240 get-ups
with both feet supporting the robot for the final 0.5 s. Hardware transfer
has yet to be measured.

### Holding the head during a positive turn

On revision `4136174e`, holding neck pitch, head pitch, head yaw and head roll
at (−0.068, −0.072, −0.012, +0.016) rad shortened an eight-second turn at
both +0.5 and +0.65 rad/s. The daemon reported the requested head pose and
the simulated joints followed it. The duck kept stepping. In matched SIT
rollouts at +0.5 rad/s, a full gait cycle took 0.38 s with either head pose,
but the trunk turned 9.6° per cycle with the offset versus 11.4° with a
neutral head. At +0.65 the corresponding angles were 11.2° and 13.4°.
Applying the offset only after the first 1.5 s also shortened the turn.

Negative neck pitch alone accounted for most of this loss. Boosting yaw
from the head command alone sometimes produced a faster gait in `robotd`,
with an eight-second turn near +295° at +0.5 rad/s. On those boots the
measured neck pitch stayed close to its target. On the slower gait the neck
settled farther below it. The actor now uses both the requested offset and
the measured joint position relative to home to limit the boost. It blends
the correction in as the legs start moving. The right pivot retains its
previous calibration.

The table compares eight-second turns with the held head on 48 new matched
placements, 16 each from SIT, STAND and FOLD.

| Command | Median angle, previous → revised | Median absolute error, previous → revised |
| --- | ---: | ---: |
| +0.5 rad/s | +202.1° → +225.5° | 27.1° → 4.3° |
| +0.65 rad/s | +248.8° → +297.1° | 49.1° → 4.0° |

To replay that head command with matched neutral controls, run the benchmark
on each actor. On each of 30 STAND, SIT and FOLD placements it settles the
duck with a neutral head, slews the head command for 16 ticks at zero twist,
then holds the target throughout the eight-second drive. The control replays
the same placement and wait. Add `--fold reserve` for another 30 placements.

```bash
uv run scripts/bench_velstand_onset.py "$DUCK_SIM_STATE/candidate.onnx" held-head.csv \
  --held-head -0.067575 -0.071532 -0.011921 0.016375
```

For a daemon run, start a fresh SIT simulation and send the following line to
the simulator's JSON-RPC socket at `$DUCK_SIM_STATE/duck.sock`. Wait for the
head to settle, then run `"$duck/scripts/duck-sim" drive 0 0.5`. Repeat from
a fresh boot with `drive 0 0.65`. The respective targets are +229° and +298°.

```json
{"jsonrpc":"2.0","id":1,"method":"robot.head","params":{"neck_pitch":-0.067575,"head_pitch":-0.071532,"head_yaw":-0.011921,"head_roll":0.016375}}
```

### Repeating straight walks and right pivots in robotd

The earlier 0.20 m/s comparison used one boot per actor. To measure the spread,
we compared revisions `4136174e` and `b4a9069d` on 52 pairs of fresh
`robotd` boots in the backlash scene. The order of the two actors was
balanced on each of two ports. Each boot started from SIT. After the daemon
reported standing, we sent a neutral head command or the four offsets above
and waited 0.5 s. The straight and right-turn commands were
`duck-sim drive 0.20 0` and `duck-sim drive 0 -0.5`. Each ran for eight
seconds. Use a fresh boot for every drive, including repeats of the same actor.

| Command and measurement over 8 s | Pairs | Previous | Current | Paired change [95% interval] |
| --- | ---: | ---: | ---: | ---: |
| 0.20 m/s, absolute heading change (°) | 16 | 10.5 | 9.0 | -1.5 [-4.0, +2.5] |
| 0.20 m/s, absolute lateral travel (cm) | 16 | 5.6 | 5.4 | +0.3 [-3.9, +2.3] |
| 0.20 m/s, forward travel (cm) | 16 | 73.4 | 73.7 | -0.2 [-1.2, +0.7] |
| −0.5 rad/s, neutral head, endpoint distance (cm) | 12 | 1.48 | 1.62 | +0.24 [-0.87, +0.45] |
| −0.5 rad/s, neutral head, maximum distance (cm) | 12 | 2.01 | 1.88 | -0.18 [-0.53, +0.10] |
| −0.5 rad/s, held head, endpoint distance (cm) | 12 | 1.29 | 1.80 | +0.40 [+0.16, +0.79] |
| −0.5 rad/s, held head, maximum distance (cm) | 12 | 2.10 | 2.25 | +0.21 [-0.09, +1.75] |

Previous and current are medians of the individual boots. The last column is
the median change within pairs and its 95% bootstrap interval, resampling
pairs within each port and order. For lateral travel, project the change in
`trunk[:2]` from the simulator body socket (`read`) onto the sideways axis at the start of the drive.
For the maximum distance, take the largest horizontal distance from that
starting point across the 20 Hz trace and both endpoints. The angle is the
accumulated heading reported by the stock drive.

At 0.20 m/s both actors drifted. The current actor finished between −1°
and +19° from its initial heading. The earlier increase from +6° to +15°
did not repeat across pairs. At −0.5 rad/s with a neutral head, the median
angle was −224.5° before and −223° after. With the held head it was −218.5°
and −216°. The held-head endpoint moved farther from the start in 9 of 12
pairs. Its paired increase was 0.40 cm. A second, distribution-free 95%
interval for that increase spans −0.03 to +1.21 cm. The peak excursion and
angle show no clear increase across the same boots.

On identical physics trajectories the actors gave identical actions at both
commands. In a separate simulation with the same actor and eight starting
states, waiting three extra control ticks (60 ms) before sending 0.20 m/s
changed the final angle by a median of 9.6° in absolute value. The timing of
the first step can therefore produce differences of the size seen in a single
pair. Individual cold boots also differ in their control and contact state.

The current actor with the held head turned +220°, +223°, +224° and +265°
at +0.5 rad/s (target +229°). At +0.65 rad/s it turned +284° to +294°
(target +298°). The default 0.15 m/s drive started on all four boots with
each actor, with median forward travel of 60.7 and 60.6 cm. Time to the first
2 cm increased in all four pairs, by a median of 0.09 s.

A separate get-up check placed the already-running simulated duck on its back,
front or side using a test fixture in the MuJoCo body. The stock daemon and
its fall handling stayed active. We verified floor contact outside the soles
before each attempt. Both actors recovered on six of six placements, with
both feet bearing weight throughout the last second. Median time to the first
0.5 s of continuous support was 1.45 s before and 1.73 s after. The daemon
engaged its fall reflex on three and four of the six boots, respectively.

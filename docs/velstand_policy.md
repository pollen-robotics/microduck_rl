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

**This checkout reproduces the v7 / v9 recipe**: the v8 pieces (`ENABLE_ANCHOR_YAW_REMAP`,
`ENABLE_JOINT_MASKED_TEACHER`, `ENABLE_WIDE_SPAWNS`) are kept in the cfg but OFF. With them off,
every env cfg (flat / rough, train / play) is identical to commit `454ddb91` (itself checked
identical to `53fb7d1`, the commit that trained v7) by full cfg dump; the runner cfg differs only
by no-op bc_cfg defaults and the teachers being loaded from their Hub training bundles (same
checkpoints). Turn the three toggles back on to reproduce v8.

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

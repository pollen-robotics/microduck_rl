# Backflip flip-envelope probe — measured results

Measured on CPU MuJoCo with `scripts/backflip_envelope.py`, before any RL
training, per AGENTS.md "verify physics assumptions in sim BEFORE training".
The robot is held at a **fixed tuck pose** via `data.ctrl` (no policy); the
launcher plate's motion is prescribed by `backflip_plate_kinematics` (Task 2).
The question this answers: is there a launch setting where the robot
completes a full 360° backward rotation at a landing speed the hardware can
survive?

## Commands run

```
uv run python scripts/backflip_envelope.py --z0 0.15
uv run python scripts/backflip_envelope.py --z0 0.10
uv run python scripts/backflip_envelope.py --z0 0.20
```

`t_hold=0.3s`, `t_launch=0.12s` (`run_cell` defaults, per the brief). Sweep
grid: `vz ∈ {1.5, 2.0, 2.5, 3.0}` m/s, `w0 ∈ {6, 9, 12, 15}` rad/s,
`tuck_factor ∈ {0.5, 1.0}` — 32 cells per `z0`.

## Two probe bugs found and fixed before trusting any number

The first run of the probe (exactly as given in the task brief, `z0=0.15`)
produced `rot_deg == 0.0` in **every one of the 32 cells**, while `apex_m`
increased sensibly with `vz`. Per the brief's own step-3 criterion ("if
rotation is ~0 everywhere ... the probe is broken"), this was investigated
before recording any table. Both bugs turned out to be in the probe script
itself (`run_cell`), not in `backflip_plate_kinematics` or the scene/launcher
from Tasks 1–2:

1. **`touching` false positive from the parked plate.** `BACKFLIP_GONE_Z =
   -3.0` teleports the plate deep into MuJoCo's floor `type="plane"` geom,
   which is an infinite half-space with no lower bound — so the parked plate
   is *itself* permanently in contact with `floor` from the instant `phase`
   becomes GONE. The original `touching = any(floor_gid in contact geoms)`
   check therefore read `True` continuously during flight, `not touching`
   never held, and `accum_pitch` never accumulated. Confirmed by dumping
   `mj_id2name` on `data.contact` mid-flight: contacts were
   `('floor', 'plate_geom')` at trunk height 0.5 m, well off the ground.
   Fix: exclude `plate_geom` from the floor-touching test (require the
   *other* geom in the contact not be the plate).
2. **Inverted sign on the rotation integral.** The brief's code integrated
   `accum_pitch += -qvel[4] * dt` with the comment "backward pitch =
   -omega_y". Tracing `data.qvel[4]` during LAUNCH showed it ramping strongly
   **positive** (~15 rad/s at `w0=15`) in lockstep with the plate's own
   prescribed `w_t`, which `backflip_plate_kinematics` documents as
   "positive = backward". Both freejoints (plate and robot) index angular
   velocity the same way, and contact transfer preserves sign — so backward
   pitch is `+omega_y`, not `-omega_y`. With the bug, every cell reported a
   large *negative* `rot_deg`; fixed to `+qvel[4] * dt`.

Both fixes are inline in `scripts/backflip_envelope.py` with comments
explaining the reasoning; neither `mdp.py` nor the launcher/scene XML from
Tasks 1–2 was touched.

## A third issue found and fixed: the brief's spawn offset floats/drops the robot

The brief's `data.qpos[0:3] = [0, 0, z0 + 0.01 + 0.10]` ("feet on the plate
top") was checked directly (`mj_forward` + longer holds) before trusting the
probe, per the task instructions to verify this estimate rather than accept
it blindly:

- At `z0=0.10` the robot **interpenetrates** the plate at spawn (~2.5 cm
  overlap).
- At `z0=0.15`/`0.20` the robot instead spawns **floating** ~5–6 cm above the
  true contact height; it is still falling at **-0.18 to -0.22 m/s** at the
  moment `t_hold` ends (0.3 s), i.e. it has not settled before launch.
- Held longer (1.5 s, well past the nominal hold), the deep-tuck
  (`tuck_factor=1.0`) pose is **unstable** on the 18×18 cm plate at this
  offset: depending on tiny numeric differences it either resettles near the
  plate or topples clean off it onto the floor beneath — chaotic edge-of-
  stability behavior from the impact of a 10 cm drop onto a footprint barely
  bigger than the tucked robot's own base.

A settle sweep at offsets `0.0–0.10` (both tuck factors, `z0=0.15`, holding
0.3 s and 1.5 s) found a **stable, reproducible** value at `+0.02` instead of
`+0.10`: at `t_hold` end, `|vz| < 0.002 m/s`, horizontal drift `< 1 cm`,
upright cosine `≈ 0.97`, consistent across `z0 ∈ {0.10, 0.15, 0.20}` and both
tuck factors, with the robot resting ~2.7 cm above the plate top. The probe
was changed to use this measured `SPAWN_OFFSET = 0.02` in place of the
brief's `0.10` guess. (Diagnostic scripts used for this measurement were
scratch, not committed; the reasoning and the final constant are documented
inline in `backflip_envelope.py`.)

## Step 3 sanity check (post-fix)

Ran `--z0 0.10` and `--z0 0.20` alongside the default `--z0 0.15` and
compared matched cells. Both `apex_m` and `rot_deg` increase monotonically
with `z0` for every `(vz, w0, tuck)` combination, and with `vz` within every
fixed `z0`. Examples:

- `vz=1.5, w0=6.0, tuck=0.5`: apex 0.302 → 0.352 → 0.402 m as `z0` goes
  0.10 → 0.15 → 0.20; rotation 106.8° → 116.7° → 124.5°.
- `vz=3.0, w0=15.0, tuck=1.0`: apex 0.461 → 0.511 → 0.561 m; rotation
  375.9° → 393.2° → 406.0°.
- Within `z0=0.15`, `w0=6.0, tuck=0.5`: apex 0.352, 0.447, 0.564, 0.701 m and
  rotation 116.7°, 157.0°, 182.4°, 208.7° for `vz = 1.5, 2.0, 2.5, 3.0`.

Both monotonic as required — the probe is transferring real momentum and
angular momentum from the plate to the robot, and the tables below are
trusted.

## Table — `z0 = 0.15` (the brief's default sweep)

```
   vz     w0  tuck  rot_deg  land_m/s  apex_m
  1.5    6.0  0.50    116.7      1.61   0.352
  1.5    6.0  1.00    134.9      1.79   0.355
  1.5    9.0  0.50    137.9      1.70   0.329
  1.5    9.0  1.00    168.8      1.91   0.326
  1.5   12.0  0.50    165.1      2.03   0.307
  1.5   12.0  1.00    184.7      1.73   0.303
  1.5   15.0  0.50    180.6      1.75   0.286
  1.5   15.0  1.00    190.8      1.61   0.282
  2.0    6.0  0.50    157.0      2.30   0.447
  2.0    6.0  1.00    171.8      2.29   0.450
  2.0    9.0  0.50    198.5      2.12   0.414
  2.0    9.0  1.00    205.2      2.05   0.416
  2.0   12.0  0.50    226.7      1.93   0.382
  2.0   12.0  1.00    248.0      1.64   0.374
  2.0   15.0  0.50    236.6      1.70   0.352
  2.0   15.0  1.00    267.5      1.42   0.343
  2.5    6.0  0.50    182.4      2.92   0.564
  2.5    6.0  1.00    214.6      2.47   0.568
  2.5    9.0  0.50    242.0      2.27   0.515
  2.5    9.0  1.00    276.5      2.07   0.514
  2.5   12.0  0.50    285.7      2.04   0.472
  2.5   12.0  1.00    307.9      1.94   0.463
  2.5   15.0  0.50    301.0      1.75   0.432
  2.5   15.0  1.00    342.8      1.92   0.417
  3.0    6.0  0.50    208.7      2.84   0.701
  3.0    6.0  1.00    272.6      2.57   0.700
  3.0    9.0  0.50    275.0      2.79   0.644
  3.0    9.0  1.00    342.1      2.58   0.628
  3.0   12.0  0.50    360.8      2.64   0.580
  3.0   12.0  1.00    376.5      2.60   0.569
  3.0   15.0  0.50    379.1      2.29   0.529
  3.0   15.0  1.00    393.2      2.38   0.511
```

Cells with `rot_deg ≥ 360`, sorted by landing speed (lowest first):

| vz  | w0   | tuck | rot_deg | land_m/s | apex_m |
|-----|------|------|---------|----------|--------|
| 3.0 | 15.0 | 0.50 | 379.1   | **2.29** | 0.529  |
| 3.0 | 15.0 | 1.00 | 393.2   | 2.38     | 0.511  |
| 3.0 | 12.0 | 1.00 | 376.5   | 2.60     | 0.569  |
| 3.0 | 12.0 | 0.50 | 360.8   | 2.64     | 0.580  |

Only 4 of the 32 swept cells close 360°, and **every one of them sits at the
top edge of the swept `vz` range** (`vz = 3.0`, the fastest launch speed
tested) with `w0 ∈ {12, 15}` (the two fastest flick rates tested). No cell at
`vz ≤ 2.5` closes 360° at `z0=0.15` (best is 342.8° at `vz=2.5, w0=15,
tuck=1.0`).

## Sanity tables — `z0 = 0.10` and `z0 = 0.20`

```
=== z0=0.10 ===
   vz     w0  tuck  rot_deg  land_m/s  apex_m
  1.5    6.0  0.50    106.8      1.38   0.302
  1.5    6.0  1.00    118.2      1.50   0.305
  1.5    9.0  0.50    126.5      1.35   0.279
  1.5    9.0  1.00    150.4      1.97   0.276
  1.5   12.0  0.50    145.7      1.48   0.257
  1.5   12.0  1.00    170.2      1.59   0.253
  1.5   15.0  0.50    149.1      1.63   0.236
  1.5   15.0  1.00    167.6      1.52   0.232
  2.0    6.0  0.50    151.6      2.00   0.397
  2.0    6.0  1.00    160.6      2.50   0.400
  2.0    9.0  0.50    189.8      1.97   0.364
  2.0    9.0  1.00    194.9      1.90   0.366
  2.0   12.0  0.50    216.7      1.84   0.332
  2.0   12.0  1.00    228.4      1.57   0.324
  2.0   15.0  0.50    222.3      1.61   0.302
  2.0   15.0  1.00    247.5      1.32   0.293
  2.5    6.0  0.50    176.1      2.69   0.514
  2.5    6.0  1.00    209.2      2.33   0.518
  2.5    9.0  0.50    234.8      2.17   0.465
  2.5    9.0  1.00    269.1      1.94   0.464
  2.5   12.0  0.50    276.3      1.98   0.422
  2.5   12.0  1.00    295.0      1.74   0.413
  2.5   15.0  0.50    286.0      1.64   0.382
  2.5   15.0  1.00    331.8      1.78   0.367
  3.0    6.0  0.50    204.4      2.72   0.651
  3.0    6.0  1.00    268.3      2.46   0.650
  3.0    9.0  0.50    268.8      2.44   0.594
  3.0    9.0  1.00    332.8      2.38   0.578
  3.0   12.0  0.50    350.4      2.39   0.528
  3.0   12.0  1.00    365.6      2.39   0.519
  3.0   15.0  0.50    369.3      2.18   0.479
  3.0   15.0  1.00    375.9      2.22   0.461

=== z0=0.20 ===
   vz     w0  tuck  rot_deg  land_m/s  apex_m
  1.5    6.0  0.50    124.5      1.83   0.402
  1.5    6.0  1.00    134.0      2.09   0.405
  1.5    9.0  0.50    149.6      2.04   0.379
  1.5    9.0  1.00    179.6      2.01   0.376
  1.5   12.0  0.50    183.0      1.99   0.357
  1.5   12.0  1.00    200.5      1.86   0.353
  1.5   15.0  0.50    195.8      1.93   0.336
  1.5   15.0  1.00    210.5      1.69   0.332
  2.0    6.0  0.50    164.8      2.50   0.497
  2.0    6.0  1.00    178.2      2.41   0.500
  2.0    9.0  0.50    206.3      2.27   0.464
  2.0    9.0  1.00    215.5      2.19   0.466
  2.0   12.0  0.50    236.7      2.02   0.432
  2.0   12.0  1.00    265.8      1.75   0.424
  2.0   15.0  0.50    251.0      1.80   0.402
  2.0   15.0  1.00    281.0      1.53   0.393
  2.5    6.0  0.50    188.6      3.12   0.614
  2.5    6.0  1.00    223.4      2.59   0.618
  2.5    9.0  0.50    250.1      2.34   0.565
  2.5    9.0  1.00    285.4      2.17   0.564
  2.5   12.0  0.50    296.3      2.11   0.522
  2.5   12.0  1.00    322.5      2.09   0.513
  2.5   15.0  0.50    315.9      1.92   0.482
  2.5   15.0  1.00    355.4      2.11   0.467
  3.0    6.0  0.50    213.0      2.97   0.751
  3.0    6.0  1.00    280.6      2.68   0.750
  3.0    9.0  0.50    281.1      2.85   0.694
  3.0    9.0  1.00    349.0      2.85   0.678
  3.0   12.0  0.50    369.9      2.70   0.628
  3.0   12.0  1.00    347.3      2.73   0.619
  3.0   15.0  0.50    389.0      2.43   0.579
  3.0   15.0  1.00    406.0      2.54   0.561
```

`z0=0.10`: 3 cells close 360° (best: `vz=3.0, w0=15.0, tuck=0.5`, rot=369.3°,
land=**2.18 m/s**). `z0=0.20`: 3 cells close 360° (best: `vz=3.0, w0=12.0,
tuck=0.5`, rot=369.9°, land=2.70 m/s, though `vz=3.0, w0=15.0, tuck=0.5` at
389.0°/2.43 m/s is a better rotation/speed trade). Higher `z0` gives more
airborne time and generally more margin above 360°, but does not lower the
landing speed at closure — the extra apex height converts back into extra
fall speed by the time the rotation completes.

## Conclusion

**360° does close, but only at the extreme corner of the swept envelope**
(`vz = 3.0` m/s — the fastest tested — with `w0 ∈ {12, 15}` rad/s), and the
landing speed there is **2.3–2.6 m/s** (best cell: `vz=3.0, w0=15.0,
tuck=0.5`, 379.1°, 2.29 m/s at `z0=0.15`). Every cell at `vz ≤ 2.5` m/s falls
short of 360° (best at that speed: 342.8° at `vz=2.5, w0=15, tuck=1.0`). No
cell reaches 360° with a landing speed below ~2.2 m/s anywhere in the sweep
(including the `z0=0.10`/`0.20` sanity sweeps).

For an 800 g, 25 cm robot this is a physically real "how hard does the sole
hit" number — a free vertical drop reaching 2.3 m/s corresponds to roughly a
27 cm fall. Whether that is acceptable for the XL330 servos and the
airframe is a hardware call, not a simulation call; this probe only answers
"what does closing 360° cost," and the answer is: nothing below ~2.2 m/s
closes.

**Recommended DR ranges for Task 6, pending the user's landing-speed
decision:**

- `vz ∈ [2.5, 3.0]` m/s and `w0 ∈ [12, 15]` rad/s bracket the region where
  360° is achievable at all; narrower ranges here have poor coverage of
  successful trials.
- `tuck_factor`: both 0.5 and 1.0 close 360° in this region — tuck depth is
  not the limiting factor at these speeds, so DR can keep both in range
  rather than forcing full tuck.
- `t_launch=0.12 s` was held fixed for this sweep (not part of the grid); it
  was not varied, so no recommendation is made on it here — a follow-up
  sweep over `t_launch` may still be worth running before Task 6 if launch
  duration turns out to matter for reward shaping.
- If a landing speed below ~2.2 m/s is a hard hardware requirement, **no
  setting in this sweep satisfies it while also closing 360°** — the project
  should either accept a higher landing speed, extend the plate's flick
  authority beyond `w0=15` / `vz=3.0` (a hardware-side decision about how
  hard the human operator can safely flick), or reconsider whether a true
  360° close is required versus training for the largest safe rotation.

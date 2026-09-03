# Backflip flip-envelope probe — measured results

Measured on CPU MuJoCo with `scripts/backflip_envelope.py`, before any RL
training, per AGENTS.md "verify physics assumptions in sim BEFORE training".
The robot is held at a **fixed tuck pose** via `data.ctrl` (no policy); the
launcher plate's motion is prescribed by `backflip_plate_kinematics` (Task 2).
The question this answers: is there a launch setting where the robot
completes a full 360° backward rotation at a landing speed the hardware can
survive?

> **Revision notice:** an earlier version of this document (and the earlier
> `backflip_plate_kinematics`) reported numbers for what turned out to be a
> **forward roll mislabeled as a backward flip** — a real bug in Task 2's
> `backflip_plate_kinematics`, caught by code review, not by this probe's own
> sanity check (the earlier check verified momentum transfer and monotonicity,
> not rotation direction). That bug is now fixed (see "Direction bug" below),
> and every number in this document is from the corrected code. The old
> numbers are superseded and should not be used for anything.

## Commands run

```
uv run python scripts/backflip_envelope.py --check-direction --z0 0.15
uv run python scripts/backflip_envelope.py --z0 0.15
uv run python scripts/backflip_envelope.py --z0 0.10
uv run python scripts/backflip_envelope.py --z0 0.20
```

`t_hold=0.3s`, `t_launch=0.12s` (`run_cell` defaults, per the brief). Sweep
grid: `vz ∈ {1.5, 2.0, 2.5, 3.0}` m/s, `w0 ∈ {6, 9, 12, 15}` rad/s,
`tuck_factor ∈ {0.5, 1.0}` — 32 cells per `z0`.

## Direction bug (Task 2): forward roll mislabeled as backward

`backflip_plate_kinematics`'s pitch/`w_t` were built from `a_ang = w0 /
t_launch` (positive for positive `w0`), producing a quaternion
`[cos(pitch/2), 0, sin(pitch/2), 0]` — a rotation about **+y** by a
**positive** angle for `w0 > 0`. This codebase already has an established
convention for what that means: `set_random_ground_state`'s `face_down`
quaternion (`mdp.py`) is exactly a +90° rotation about +y, and the codebase's
own roulade design notes (`mdp.py:6601-6604`) record this explicitly: "+90°
pitch about +y = face-down" is the **forward**-roll direction. So `w0 > 0`,
as originally implemented, was driving the robot into a forward roll
(nose-down/face-down), not a backward flip — the opposite of the function's
own docstring claim ("positive = the flick that drives the robot BACKWARD").

This was not caught by this probe's Task 3 sanity check, because that check
only verified that momentum was transferring at all and that apex/rotation
scaled with `z0`/`vz` — it never checked rotation *direction*, and a forward
roll passes that check exactly as well as a backward one. It was caught by
code review, which traced the quaternion convention through
`set_random_ground_state` and an independent orientation trace.

**Fix:** negated `a_ang` inside `backflip_plate_kinematics`
(`a_ang = -w0 / t_launch`) so the function's return values (`pitch`, `w_t`)
are now negative for a positive `w0`, while `w0 > 0` keeps its public meaning
("the flick that drives the robot BACKWARD") — the contract callers rely on
is unchanged, only the internal sign of the geometric quantities. Updated the
function's docstring to state the sign convention explicitly, and updated
`tests/test_backflip_launcher.py`'s two sign-asserting tests
(`test_launch_is_a_ramp_not_a_step`, `test_launch_position_is_the_integral_of_the_ramp`),
which had pinned the old (wrong) positive sign — they now assert `w_t` and
`pitch` are negative for `w0=10.0`. All 9 tests in that file pass after the
fix (`uv run --with pytest pytest tests/test_backflip_launcher.py -q`).

The probe script's own rotation accumulator was also reverted from `+qvel[4]`
back to the brief's original `-qvel[4]` ("backward pitch = -omega_y") — that
sign was correct all along, by this same convention; Task 3's earlier "fix"
that flipped it to `+qvel[4]` was itself the wrong move (it made the sign
error self-consistent with the *original*, wrong `backflip_plate_kinematics`,
instead of catching that the kinematics function was the actual bug).

### Direction verified by measurement, not by convention alone

```
uv run python scripts/backflip_envelope.py --check-direction --z0 0.15
```

```
Direction check (z0=0.15):
  direction check @ t=0.566s accum=91.2deg: local +z in world = (-0.888, 0.010, -0.460)
  -> -x component: robot is leaning BACKWARD-and-up. Correct direction.
  cell result: vz=3.0 w0=15.0 tuck=1.0 z0=0.15 -> rot_deg=476.1 land_m/s=3.58 apex_m=0.901
```

At ~90° of accumulated rotation, the robot's own local +z axis (its "up")
points to world `(-0.888, 0.010, -0.460)` — a strong **negative** x
component. Before the fix, the same trace (independently reproduced by the
reviewer on the same cell, `vz=3.0, w0=15.0, tuck=1.0, z0=0.15`) showed local
+z pointing to world `(0.99, 0, 0.13)` — a strong **positive** x component,
i.e. face-down (belly to floor), the forward-roll signature. The sign flip
from +0.99 to -0.888 on the x-component is the direction test: the robot is
now leaning backward-and-up over its heels (arcing toward its own back)
rather than pitching nose-down. This is the qualitative test the reviewer
asked for; the z-component of -0.460 (rather than ~0, which a pure single-axis
rotation about y would give at exactly -90°) reflects some non-planar
richness in the real contact-driven launch (slight roll/yaw coupling from the
tucked pose, not a pure y-axis rotation) and does not affect the direction
conclusion, which rests on the x-component's sign.

## Two other probe bugs found and fixed (unrelated to direction)

Independently of the direction bug, three issues were fixed in the probe
script itself in the original Task 3 pass and are unchanged by this
revision:

1. **`touching` false positive from the parked plate**, because
   `BACKFLIP_GONE_Z = -3.0` teleports the plate into MuJoCo's infinite
   `type="plane"` floor half-space, so the parked plate is itself
   permanently "touching" `floor`. Fixed by excluding `plate_geom` from the
   floor-contact test.
2. **Spawn offset floats/interpenetrates.** The brief's
   `z0 + 0.01 + 0.10` either interpenetrates (`z0=0.10`) or leaves the robot
   still falling at `t_hold` end (`z0=0.15`/`0.20`), and at full tuck is
   unstable on the plate over longer holds. Replaced with a measured
   `SPAWN_OFFSET = 0.02`, verified stable (`|vz| < 0.002 m/s`, drift `< 1cm`,
   upright cosine `≈ 0.97`) across all three `z0` and both tuck factors.

This revision adds two more fixes, both requested by code review:

3. **Landing speed read after the contact response, not before.**
   `landing_speed = abs(qvel[2])` was read AFTER `mujoco.mj_step`, so the
   contact constraint solved inside that step had already partially
   decelerated the robot — understating "peak downward speed at first
   contact," the number a human uses to judge hardware risk. Fixed by
   snapshotting `qvel[2]` at the TOP of the loop, before `mj_step`, and using
   that pre-step value when first contact is detected after the step.
4. **Post-landing bounce could keep accumulating rotation.** After the first
   ground touch, a bounce back into "airborne, not touching" resumed adding
   to `accum_pitch`. Fixed with a `landed` latch: once the first true contact
   is detected, both the rotation accumulator and the landing-speed
   measurement are frozen for the rest of the episode. Verified directly: for
   the cell `vz=2.0, w0=15.0, tuck=1.0, z0=0.15` there are 16 touch/release
   transitions after first contact (real bouncing), and `accum_pitch` at
   first touch (377.15°) exactly equals `accum_pitch` at the end of the
   2-second window (377.15°) — the latch holds. Two other spot-checked cells
   have zero post-landing bounce transitions and are unaffected either way.

## Step 3 sanity check (post-fix)

Ran `--z0 0.10` and `--z0 0.20` alongside the default `--z0 0.15` and
compared matched cells.

**Apex is monotonic in both `z0` and `vz` everywhere, with no exception** —
checked across all 32 `(w0, tuck)` × `vz` combinations at every `z0`.

**Rotation is monotonic in `z0` everywhere.** Examples:
- `vz=1.5, w0=6.0, tuck=0.5`: rotation 82.3° → 87.4° → 92.5° as `z0` goes
  0.10 → 0.15 → 0.20.
- `vz=3.0, w0=15.0, tuck=1.0`: rotation 465.4° → 476.1° → 484.1°.

**Rotation is monotonic in `vz` for most `(w0, tuck)` combinations, with a
small, physically-explained exception at the weakest flick setting
(`w0=6.0`)**, checked programmatically across all 32 `(z0, w0, tuck)` groups.
At `z0=0.15, w0=6.0, tuck=0.5`: rotation 87.4°, 107.6°, 109.7°, 109.4° for
`vz = 1.5, 2.0, 2.5, 3.0` — the last step (`vz=2.5→3.0`) is a 0.3° *decrease*,
not an increase. The same pattern (a small dip at the top of the `vz` range)
appears at `w0=6.0, tuck=1.0` and `w0=9.0, tuck=1.0` for all three `z0`
values, with the largest dip 272.9°→263.1° (`z0=0.15, w0=9.0, tuck=1.0`,
`vz=2.5→3.0`). Apex is unaffected (still strictly monotonic) in every one of
these rows, which points to the mechanism: at a weak angular flick, a
strong-enough vertical kick can separate the robot's feet from the plate
(losing normal force / grip) before the full `t_launch` window has imparted
its angular impulse, so pushing `vz` higher still while `w0` stays weak can
buy less rotation, not more, even though it still buys more apex height. This
dip is confined to `w0 ∈ {6, 9}` and does not appear at `w0 ∈ {12, 15}` —
which is also where every 360°-closing cell lives — so it does not affect
this document's conclusion.

Given apex's clean monotonicity everywhere, and rotation's clean
monotonicity in the region that actually closes 360°, the probe is
transferring real, well-behaved momentum and is trusted for the tables below.
The `vz`-monotonicity dip at weak `w0` is reported here for completeness
rather than silently dropped, per the project's "measure and report, don't
paper over" convention.

## Table — `z0 = 0.15` (the brief's default sweep)

```
   vz     w0  tuck  rot_deg  land_m/s  apex_m
  1.5    6.0  0.50     87.4      2.51   0.389
  1.5    6.0  1.00    134.2      2.45   0.391
  1.5    9.0  0.50    151.9      2.26   0.393
  1.5    9.0  1.00    198.1      2.21   0.396
  1.5   12.0  0.50    213.1      2.40   0.408
  1.5   12.0  1.00    265.7      2.77   0.412
  1.5   15.0  0.50    276.3      2.91   0.422
  1.5   15.0  1.00    318.7      3.09   0.421
  2.0    6.0  0.50    107.6      2.91   0.508
  2.0    6.0  1.00    153.6      2.91   0.513
  2.0    9.0  0.50    178.8      2.70   0.524
  2.0    9.0  1.00    235.7      2.85   0.529
  2.0   12.0  0.50    265.0      3.15   0.549
  2.0   12.0  1.00    319.9      3.49   0.555
  2.0   15.0  0.50    322.3      3.23   0.560
  2.0   15.0  1.00    377.1      3.21   0.564
  2.5    6.0  0.50    109.7      3.39   0.658
  2.5    6.0  1.00    164.8      3.30   0.662
  2.5    9.0  0.50    206.1      3.23   0.675
  2.5    9.0  1.00    272.9      3.59   0.686
  2.5   12.0  0.50    303.1      3.73   0.705
  2.5   12.0  1.00    356.3      3.85   0.718
  2.5   15.0  0.50    386.6      3.48   0.719
  2.5   15.0  1.00    424.3      3.35   0.722
  3.0    6.0  0.50    109.4      3.86   0.829
  3.0    6.0  1.00    156.6      3.83   0.833
  3.0    9.0  0.50    232.3      3.80   0.853
  3.0    9.0  1.00    263.1      3.95   0.862
  3.0   12.0  0.50    350.2      4.09   0.890
  3.0   12.0  1.00    374.3      4.21   0.899
  3.0   15.0  0.50    450.8      3.67   0.904
  3.0   15.0  1.00    476.1      3.58   0.901
```

Cells with `rot_deg ≥ 360`, sorted by landing speed (lowest first):

| vz  | w0   | tuck | rot_deg | land_m/s | apex_m |
|-----|------|------|---------|----------|--------|
| 2.0 | 15.0 | 1.00 | 377.1   | **3.21** | 0.564  |
| 2.5 | 15.0 | 1.00 | 424.3   | 3.35     | 0.722  |
| 2.5 | 15.0 | 0.50 | 386.6   | 3.48     | 0.719  |
| 3.0 | 15.0 | 1.00 | 476.1   | 3.58     | 0.901  |
| 3.0 | 15.0 | 0.50 | 450.8   | 3.67     | 0.904  |
| 3.0 | 12.0 | 1.00 | 374.3   | 4.21     | 0.899  |

6 of 32 cells close 360° at `z0=0.15`. All require `w0 ≥ 12`, and all but one
require `w0 = 15` (the fastest flick tested) — `w0=12` closes 360° only at
`vz=3.0, tuck=1.0`. Every closing cell's landing speed is **3.2–4.2 m/s**,
notably higher than this document's superseded first draft reported
(2.3–2.6 m/s) — that earlier number was both direction-mislabeled AND
measured after the contact response had already slowed the robot down (bug
#3 above), so it understated the true risk in two independent ways.

## Sanity tables — `z0 = 0.10` and `z0 = 0.20`

```
=== z0=0.10 ===
   vz     w0  tuck  rot_deg  land_m/s  apex_m
  1.5    6.0  0.50     82.3      2.30   0.339
  1.5    6.0  1.00    127.3      2.25   0.341
  1.5    9.0  0.50    145.3      2.09   0.343
  1.5    9.0  1.00    188.5      1.99   0.346
  1.5   12.0  0.50    200.3      2.10   0.358
  1.5   12.0  1.00    253.1      2.49   0.362
  1.5   15.0  0.50    257.8      2.57   0.372
  1.5   15.0  1.00    306.5      2.93   0.371
  2.0    6.0  0.50    104.0      2.76   0.458
  2.0    6.0  1.00    148.3      2.74   0.463
  2.0    9.0  0.50    172.9      2.53   0.474
  2.0    9.0  1.00    226.4      2.61   0.479
  2.0   12.0  0.50    253.6      2.89   0.499
  2.0   12.0  1.00    310.3      3.32   0.505
  2.0   15.0  0.50    313.3      3.10   0.510
  2.0   15.0  1.00    366.8      3.14   0.514
  2.5    6.0  0.50    107.0      3.25   0.608
  2.5    6.0  1.00    161.5      3.18   0.612
  2.5    9.0  0.50    200.5      3.06   0.625
  2.5    9.0  1.00    265.0      3.38   0.636
  2.5   12.0  0.50    298.1      3.59   0.655
  2.5   12.0  1.00    348.2      3.73   0.668
  2.5   15.0  0.50    375.3      3.37   0.669
  2.5   15.0  1.00    413.1      3.28   0.672
  3.0    6.0  0.50    107.0      3.72   0.779
  3.0    6.0  1.00    153.9      3.72   0.783
  3.0    9.0  0.50    228.3      3.67   0.803
  3.0    9.0  1.00    257.9      3.79   0.812
  3.0   12.0  0.50    343.2      3.97   0.840
  3.0   12.0  1.00    368.0      4.12   0.849
  3.0   15.0  0.50    443.3      3.58   0.854
  3.0   15.0  1.00    465.4      3.46   0.851

=== z0=0.20 ===
   vz     w0  tuck  rot_deg  land_m/s  apex_m
  1.5    6.0  0.50     92.5      2.71   0.439
  1.5    6.0  1.00    140.5      2.64   0.441
  1.5    9.0  0.50    158.7      2.45   0.443
  1.5    9.0  1.00    206.8      2.42   0.446
  1.5   12.0  0.50    222.6      2.62   0.458
  1.5   12.0  1.00    278.2      3.04   0.462
  1.5   15.0  0.50    286.9      3.08   0.472
  1.5   15.0  1.00    330.8      3.23   0.471
  2.0    6.0  0.50    111.2      3.07   0.558
  2.0    6.0  1.00    157.1      3.02   0.563
  2.0    9.0  0.50    184.6      2.87   0.574
  2.0    9.0  1.00    244.0      3.07   0.579
  2.0   12.0  0.50    276.5      3.41   0.599
  2.0   12.0  1.00    328.3      3.63   0.605
  2.0   15.0  0.50    332.6      3.36   0.610
  2.0   15.0  1.00    389.0      3.28   0.614
  2.5    6.0  0.50    112.5      3.52   0.708
  2.5    6.0  1.00    168.5      3.44   0.712
  2.5    9.0  0.50    211.7      3.41   0.725
  2.5    9.0  1.00    279.1      3.76   0.736
  2.5   12.0  0.50    311.1      3.86   0.755
  2.5   12.0  1.00    364.3      3.96   0.768
  2.5   15.0  0.50    395.4      3.57   0.769
  2.5   15.0  1.00    434.0      3.42   0.772
  3.0    6.0  0.50    111.4      3.97   0.879
  3.0    6.0  1.00    159.2      3.95   0.883
  3.0    9.0  0.50    237.1      3.95   0.903
  3.0    9.0  1.00    268.3      4.11   0.912
  3.0   12.0  0.50    356.2      4.19   0.940
  3.0   12.0  1.00    380.5      4.31   0.949
  3.0   15.0  0.50    460.5      3.78   0.953
  3.0   15.0  1.00    484.1      3.68   0.951
```

`z0=0.10`: 6 cells close 360° (best: `vz=2.0, w0=15.0, tuck=1.0`, rot=366.8°,
land=**3.14 m/s**). `z0=0.20`: 7 cells close 360° (best: `vz=2.0, w0=15.0,
tuck=1.0`, rot=389.0°, land=**3.28 m/s**). The best (lowest-landing-speed)
closing cell is consistently `vz=2.0, w0=15.0, tuck=1.0` across all three
`z0` — a moderate vertical launch with the fastest tested flick rate and full
tuck.

## Conclusion

**360° does close, but requires the fastest flick rate tested (`w0=15`
rad/s) in nearly every closing cell**, and the landing speed there is
**3.1–4.3 m/s** — a meaningfully harder number than this document's earlier,
superseded draft (2.3–2.6 m/s), because that earlier draft both mislabeled a
forward roll as backward AND measured landing speed after the contact
response had already slowed the robot. The best cell across all three `z0`
sanity sweeps is consistently `vz=2.0, w0=15.0, tuck=1.0`: 366.8–389.0°
rotation at 3.14–3.28 m/s landing speed depending on `z0`.

For an 800 g, 25 cm robot, 3.2 m/s corresponds to roughly a 52 cm
free-fall-equivalent impact — a materially harder hit than the previous
draft implied, and a hardware-acceptability call for the user, not a
simulation call.

**Recommended DR ranges for Task 6, pending the user's landing-speed
decision:**

- `w0 = 15` rad/s (the fastest flick tested) is required for nearly every
  360°-closing cell; only one cell closes at `w0=12` (`vz=3.0, tuck=1.0`,
  z0=0.15). DR ranges narrower than `w0 ∈ [12, 15]` would have poor coverage
  of successful trials, and a range that excludes `w0=15` would have almost
  none.
- `vz`: the best (lowest-landing-speed) closing cells cluster around
  `vz=2.0–2.5` m/s, not the top of the swept range — pushing `vz` to 3.0
  increases rotation margin but also increases landing speed and apex
  without buying a lower landing speed. `vz ∈ [2.0, 3.0]` covers the closing
  region.
- `tuck_factor = 1.0` (full tuck) gives the lowest landing speed at matched
  `vz, w0` in every closing case found; `tuck=0.5` also closes 360° at
  `vz ≥ 2.5, w0=15` but at a higher landing speed. DR can keep both in range,
  but full tuck is the better default if a single value is needed.
- `t_launch=0.12s` was held fixed for this sweep, as in the first draft; not
  varied here either.
- If a landing speed materially below ~3.1 m/s is a hard hardware
  requirement, **no setting in this corrected sweep satisfies it while also
  closing 360°** — this is a stronger form of the same caveat as the earlier
  draft, now backed by a corrected measurement. The project should accept
  the higher landing speed, extend the plate's flick authority beyond
  `w0=15` (a hardware/operator decision), or reconsider whether a true 360°
  close is required versus training for the largest safe rotation.

---

# Extended sweep (Task 3 addendum)
This section extends the sweep above, per the same measurement discipline
(fixed tuck pose, no policy, prescribed plate kinematics). It answers two
concerns with the original 32-cell grid: the best cell sat at `w0=15`, the
edge of the swept range, and the best landing speed found (3.21 m/s) is a
real hardware risk. The question: does pushing the flick rate `w0` higher
while pulling the launch speed `vz` lower trade rotation margin for a softer
landing, and if so, by how much?

**Short answer: yes, but the naive best-by-landing-speed cells across the
full extended grid are a false lead — a direction-check catches them as
FORWARD rolls, not backward flips (see "The reversal" below). Restricted to
the region where the flip direction is verified backward, the genuine
improvement is real but smaller: landing speed drops from 3.1-4.3 m/s to
roughly 1.5-2.6 m/s, not below 1 m/s.**

## Commands run
Script changes: `scripts/backflip_envelope.py` gained `--vz`, `--w0`, `--tuck`
(comma-separated grids, defaults below) and `--check-vz`/`--check-w0`/`--check-tuck`
for `--check-direction` on an arbitrary cell. `--z0` is unchanged.
```
# Default extended grid: vz in {1.00..3.00 step 0.25} (9 values),
# w0 in {6..54 step 3} (17 values), tuck in {0.5, 0.75, 1.0} (3 values)
# = 459 cells per z0.
uv run python scripts/backflip_envelope.py --z0 0.15
uv run python scripts/backflip_envelope.py --z0 0.10
uv run python scripts/backflip_envelope.py --z0 0.20

# Direction check on the naive (uncorrected) best cell -- FAILS:
uv run python scripts/backflip_envelope.py --check-direction --z0 0.15 \
  --check-vz 3.0 --check-w0 45.0 --check-tuck 1.0

# Direction check on the genuine (direction-verified) best cell -- PASSES:
uv run python scripts/backflip_envelope.py --check-direction --z0 0.10 \
  --check-vz 2.25 --check-w0 30.0 --check-tuck 1.0
```
`t_hold=0.3s`, `t_launch=0.12s` unchanged. The default `w0` ceiling (54) was
not chosen up front: an initial pass to `w0<=30` (the brief's suggested
ceiling) still had every new-best cell sitting at the edge of that range, so
it was pushed further until the trend visibly turned over (see the full
tables below -- landing speed among closing cells bottoms out around
`w0=39-48` and rises sharply beyond `w0~51-54`, and 360 deg stops closing at
all past that). That the true numeric optimum is interior, not at the grid
edge, is confirmed. What is NOT confirmed -- and turns out to be false -- is
that this numeric optimum is a real backward flip.

## The reversal: the deepest "valley" is a forward roll, not a backward flip
Sorting all closing cells (`rot_deg >= 360`) by landing speed, the top of
the list at every `z0` is dominated by `w0` in the high-30s to low-50s:

| z0   | best-by-landing-speed cell (vz, w0, tuck) | rot_deg | land_m/s |
|------|--------------------------------------------|---------|----------|
| 0.15 | 3.00, 45.0, 1.00                            | 424.9   | **1.11** |
| 0.10 | 3.00, 45.0, 1.00                            | 404.8   | **0.92** |
| 0.20 | 2.75, 45.0, 1.00                             | 412.7   | **1.26** |

These look spectacular -- under half the original 3.21 m/s. Per the brief's
instruction to look directly at a cell that reports a closed flip at an
implausibly low apex before believing it (apex here is 0.34-0.39 m, well
below the original best cell's 0.56 m), `--check-direction` was run on the
`z0=0.15` cell:

```
uv run python scripts/backflip_envelope.py --check-direction --z0 0.15 \
  --check-vz 3.0 --check-w0 45.0 --check-tuck 1.0
```

```
Direction check (z0=0.15):
  direction check @ t=0.472s accum=91.7deg: local +z in world = (0.250, 0.041, -0.967)
  -> +x component: robot is FACE-DOWN (forward roll). WRONG direction.
  cell result: vz=3.0 w0=45.0 tuck=1.0 z0=0.15 -> rot_deg=424.9 land_m/s=1.11 apex_m=0.375
```

**This fails the check.** At `w0=45`, the "424.9 deg of backward rotation"
the probe reports is not a backward flip -- by the same local-+z-axis
convention the original document used to verify the `w0=15` baseline
(positive world-x component = face-down = forward roll), this cell is a
forward roll that happens to still integrate a large `-omega_y` early in
flight.

A boundary scan (same `check-direction`, sweeping `w0` at fixed `vz=3.0,
tuck=1.0, z0=0.15`) shows this is not a fluke at one cell -- it is a
continuous drift that flips sign:

| w0   | local +z world x-component | verdict |
|------|------------------------------|---------|
| 15.0 | -0.888                        | backward (this is the original reviewed baseline) |
| 21.0 | -0.713                        | backward |
| 27.0 | -0.497                        | backward |
| 30.0 | -0.373                        | backward |
| 33.0 | -0.220                        | backward, weakening |
| 36.0 | -0.115                        | backward, marginal |
| 39.0 | -0.007                         | ambiguous (essentially zero) |
| 42.0 | +0.131                        | **forward -- WRONG** |
| 45.0 | +0.250                        | **forward -- WRONG** |

The same pattern repeats at `vz=2.0, tuck=1.0`: x-component is -0.533 at
`w0=24`, shrinks monotonically to -0.043 at `w0=36` (by which point rotation
also stops closing 360, 346.0 deg). Every other "top of the naive list" cell
checked directly (`vz=2.5, w0=39/42, tuck=0.75`; `vz=2.75, w0=42/45,
tuck=1.0`) also comes back FACE-DOWN / WRONG. **None of the sub-`2.0 m/s`
landing speeds from `w0` above roughly 36-39 are trustworthy backward
flips** -- they are excluded from the recommendation below.

The likely mechanism (not independently re-derived here, flagged for
whoever tunes reward shaping next): `backflip_plate_kinematics` prescribes
a rotation ramping to `w0` in just `t_launch=0.12s`. At `w0=45 rad/s` that
peak rate is roughly 2 rotations of the PLATE itself within the launch
window -- an extremely violent flick. The robot is not kinematically
driven; only the plate is. At these rates the contact interaction between
the robot's feet and the fast-moving plate surface is plausibly leaving the
"clean single-axis backward pitch" regime entirely (off-axis roll/yaw
coupling grows steadily through the boundary scan above, well before the
sign flips), so the scalar `-omega_y` integral the probe reports stops
correlating with the actual body orientation change. Whether that is a
genuine physical effect of an overdriven contact or a numerical artifact of
the timestep being too coarse to resolve such a fast plate motion was not
determined here -- either way, `w0` above ~33-36 needs a direction check
before its numbers are used for anything.

## The genuine improvement: best cells restricted to the direction-verified region (`w0 <= 30`)
Restricting to `w0 <= 30` (comfortably clear of the `w0~33-36` boundary
where the direction signal goes ambiguous), the top closing cells by landing
speed at each `z0`:

**z0 = 0.15**
```
   vz     w0  tuck  rot_deg  land_m/s  apex_m
 2.00   30.0  1.00    430.0      1.56   0.392
 2.00   30.0  0.75    440.2      1.59   0.406
 2.25   30.0  1.00    460.2      1.61   0.442
 1.75   30.0  0.75    401.3      1.67   0.361
 2.00   27.0  1.00    447.4      1.71   0.434
 2.00   30.0  0.50    437.6      1.71   0.408
 1.75   30.0  0.50    400.7      1.73   0.361
 1.75   27.0  0.75    427.1      1.77   0.397
 2.00   27.0  0.75    455.5      1.79   0.450
 2.25   30.0  0.75    482.9      1.80   0.452
```

**z0 = 0.10**
```
   vz     w0  tuck  rot_deg  land_m/s  apex_m
 2.25   30.0  1.00    445.4      1.46   0.392
 2.25   30.0  0.75    459.9      1.48   0.402
 2.00   30.0  0.75    425.0      1.50   0.356
 2.00   30.0  1.00    399.5      1.54   0.342
 2.00   30.0  0.50    420.4      1.58   0.358
 2.00   27.0  1.00    433.3      1.64   0.384
 2.25   30.0  0.50    454.7      1.64   0.404
 2.25   27.0  1.00    464.5      1.67   0.440
 2.00   27.0  0.75    441.7      1.69   0.400
 2.50   30.0  1.00    484.7      1.69   0.443
```

**z0 = 0.20**
```
   vz     w0  tuck  rot_deg  land_m/s  apex_m
 2.00   30.0  1.00    445.2      1.65   0.442
 1.75   30.0  0.75    427.0      1.72   0.411
 2.00   30.0  0.75    455.5      1.72   0.456
 1.75   30.0  1.00    408.3      1.75   0.398
 1.75   30.0  0.50    420.7      1.81   0.411
 2.00   27.0  1.00    461.5      1.83   0.484
 1.75   27.0  0.75    441.3      1.84   0.447
 1.75   27.0  1.00    432.6      1.84   0.433
 2.00   30.0  0.50    452.3      1.87   0.458
 2.25   30.0  1.00    479.9      1.88   0.492
```

The best genuine (direction-verified-region) cell is consistently
`vz~2.0-2.25, w0=30.0, tuck=0.75-1.0` across all three `z0` -- and unlike the
false `w0=45` lead, this one is NOT sitting at the swept grid's edge: `w0=30`
is an interior point of the direction-trustworthy range (`w0<=~33-36`), and
pushing `w0` from 27 to 30 continues to improve landing speed at fixed `vz`
in every `z0` (e.g. `vz=2.0, tuck=1.0, z0=0.15`: 447.4/1.71 at `w0=27` ->
430.0/1.56 at `w0=30`), so this is a real local optimum, not an artifact of
where the safe region happens to end.

**Direct confirmation on the actual best cell** (per the brief's requirement
to check at least one new best cell):

```
uv run python scripts/backflip_envelope.py --check-direction --z0 0.10 \
  --check-vz 2.25 --check-w0 30.0 --check-tuck 1.0
```
```
Direction check (z0=0.1):
  direction check @ t=0.492s accum=90.3deg: local +z in world = (-0.332, 0.019, -0.943)
  -> -x component: robot is leaning BACKWARD-and-up. Correct direction.
  cell result: vz=2.25 w0=30.0 tuck=1.0 z0=0.1 -> rot_deg=445.4 land_m/s=1.46 apex_m=0.392
```

Confirmed backward. 1.46 m/s at `z0=0.10` is the single best genuine number
found in this sweep -- down from the original document's 3.14 m/s at the
same `z0` (a ~53% reduction), not the ~70% reduction the (wrong) `w0=45`
cell would have implied.

## DR margin: is there a robust box, or only a knife-edge cell?
A single best cell is not a DR range. Checking the box `vz in [2.00, 2.25]
m/s x w0 in [24.0, 30.0] rad/s x tuck in [0.5, 1.0]` -- all direction-trustworthy
(`w0<=30`) -- for closure and landing speed across all three `z0`:

**z0 = 0.15**
```
   vz     w0  tuck  rot_deg  land_m/s  apex_m
 2.00   24.0  0.50    443.6      2.18   0.489
 2.00   24.0  0.75    461.9      2.03   0.495
 2.00   24.0  1.00    457.3      1.90   0.477
 2.00   27.0  0.50    445.6      1.93   0.449
 2.00   27.0  0.75    455.5      1.79   0.450
 2.00   27.0  1.00    447.4      1.71   0.434
 2.00   30.0  0.50    437.6      1.71   0.408
 2.00   30.0  0.75    440.2      1.59   0.406
 2.00   30.0  1.00    430.0      1.56   0.392
 2.25   24.0  0.50    477.9      2.33   0.552
 2.25   24.0  0.75    499.4      2.26   0.557
 2.25   24.0  1.00    488.1      2.10   0.540
 2.25   27.0  0.50    482.9      2.17   0.503
 2.25   27.0  0.75    501.3      2.11   0.505
 2.25   27.0  1.00    480.5      1.87   0.490
 2.25   30.0  0.50    476.8      1.95   0.454
 2.25   30.0  0.75    482.9      1.80   0.452
 2.25   30.0  1.00    460.2      1.61   0.442
```

**z0 = 0.10** (all close except the one flagged cell)
```
   vz     w0  tuck  rot_deg  land_m/s  apex_m
 2.00   24.0  0.50    431.6      2.09   0.439
 2.00   24.0  0.75    449.2      1.93   0.445
 2.00   24.0  1.00    444.5      1.81   0.427
 2.00   27.0  0.50    432.3      1.81   0.399
 2.00   27.0  0.75    441.7      1.69   0.400
 2.00   27.0  1.00    433.3      1.64   0.384
 2.00   30.0  0.50    420.4      1.58   0.358
 2.00   30.0  0.75    425.0      1.50   0.356
 2.00   30.0  1.00    399.5      1.54   0.342
 2.25   24.0  0.50    464.0      2.15   0.502
 2.25   24.0  0.75    482.6      2.02   0.507
 2.25   24.0  1.00    473.3      1.92   0.490
 2.25   27.0  0.50    465.6      1.90   0.455
 2.25   27.0  0.75    480.4      1.78   0.455
 2.25   27.0  1.00    464.5      1.67   0.440
 2.25   30.0  0.50    454.7      1.64   0.404
 2.25   30.0  0.75    459.9      1.48   0.402
 2.25   30.0  1.00    445.4      1.46   0.392
```

**z0 = 0.20**
```
   vz     w0  tuck  rot_deg  land_m/s  apex_m
 2.00   24.0  0.50    455.6      2.30   0.539
 2.00   24.0  0.75    474.5      2.16   0.545
 2.00   24.0  1.00    472.3      2.05   0.527
 2.00   27.0  0.50    459.0      2.07   0.499
 2.00   27.0  0.75    469.4      1.94   0.500
 2.00   27.0  1.00    461.5      1.83   0.484
 2.00   30.0  0.50    452.3      1.87   0.458
 2.00   30.0  0.75    455.5      1.72   0.456
 2.00   30.0  1.00    445.2      1.65   0.442
 2.25   24.0  0.50    491.8      2.53   0.602
 2.25   24.0  0.75    516.1      2.55   0.607
 2.25   24.0  1.00    502.8      2.33   0.590
 2.25   27.0  0.50    505.3      2.53   0.555
 2.25   27.0  0.75    526.9      2.59   0.555
 2.25   27.0  1.00    498.9      2.16   0.540
 2.25   30.0  0.50    501.4      2.37   0.504
 2.25   30.0  0.75    510.9      2.31   0.502
 2.25   30.0  1.00    479.9      1.88   0.492
```

Every cell in this box closes 360 deg at every `z0`, with landing speed
`1.46-2.59 m/s` -- meaningfully below the original document's closing-cell
range (`3.1-4.3 m/s`) with margin on both sides of every dimension, not a
knife edge. **One caveat found while building this box**: widening it to
include `vz=1.75` breaks at `z0=0.10` -- `vz=1.75, w0=30.0, tuck=1.00,
z0=0.10` gives only `336.3 deg` (does not close). That corner (low vz, high
w0, high tuck, shallow z0) is excluded from the recommended range below for
exactly this reason.

`vz=2.5` was also checked at `w0 in [24,30]` and rejected: it still closes
360 everywhere, but landing speed at `z0=0.20` reaches `3.10-3.72 m/s` --
worse than the original best cell, not better -- so `vz` should not be
pushed past `2.25` in the recommended range.

## Full raw sweep output (verbatim, all 459 cells per z0)
Collapsed for length -- reproduced with the commands in "Commands run"
above. `rot_deg` is backward rotation for `w0<=~33`; treat `rot_deg` for
`w0>36` as an ungrounded number until independently direction-checked (see
"The reversal" above).

<details>
<summary>z0 = 0.15 (179/459 close)</summary>

```
   vz     w0  tuck  rot_deg  land_m/s  apex_m
 1.00    6.0  0.50     72.4      2.09   0.295
 1.00    6.0  0.75     84.5      2.13   0.300
 1.00    6.0  1.00    116.9      2.04   0.295
 1.00    9.0  0.50    125.4      1.98   0.292
 1.00    9.0  0.75    143.2      1.93   0.297
 1.00    9.0  1.00    166.6      1.79   0.295
 1.00   12.0  0.50    160.7      1.71   0.295
 1.00   12.0  0.75    183.5      1.69   0.298
 1.00   12.0  1.00    200.3      1.73   0.296
 1.00   15.0  0.50    197.1      1.87   0.302
 1.00   15.0  0.75    227.2      2.06   0.307
 1.00   15.0  1.00    243.8      2.26   0.302
 1.00   18.0  0.50    230.5      2.17   0.304
 1.00   18.0  0.75    271.5      2.63   0.309
 1.00   18.0  1.00    275.2      2.64   0.301
 1.00   21.0  0.50    261.2      2.46   0.297
 1.00   21.0  0.75    277.8      2.59   0.301
 1.00   21.0  1.00    282.6      2.69   0.289
 1.00   24.0  0.50    259.4      2.36   0.282
 1.00   24.0  0.75    276.8      2.52   0.283
 1.00   24.0  1.00    277.8      2.68   0.271
 1.00   27.0  0.50    249.1      2.32   0.262
 1.00   27.0  0.75    260.2      2.47   0.263
 1.00   27.0  1.00    261.4      2.65   0.252
 1.00   30.0  0.50    233.3      2.36   0.241
 1.00   30.0  0.75    240.7      2.51   0.241
 1.00   30.0  1.00    239.9      2.65   0.232
 1.00   33.0  0.50    213.6      2.45   0.222
 1.00   33.0  0.75    167.9      1.82   0.221
 1.00   33.0  1.00    168.8      1.77   0.215
 1.00   36.0  0.50    128.0      1.47   0.211
 1.00   36.0  0.75    136.1      1.54   0.213
 1.00   36.0  1.00    142.0      1.55   0.211
 1.00   39.0  0.50    108.8      1.37   0.209
 1.00   39.0  0.75    117.4      1.52   0.211
 1.00   39.0  1.00    112.3      1.21   0.209
 1.00   42.0  0.50     96.2      1.45   0.207
 1.00   42.0  0.75     93.6      1.30   0.209
 1.00   42.0  1.00     94.0      1.19   0.207
 1.00   45.0  0.50     77.4      1.32   0.206
 1.00   45.0  0.75     79.4      1.31   0.208
 1.00   45.0  1.00     79.6      1.28   0.206
 1.00   48.0  0.50     63.9      1.26   0.205
 1.00   48.0  0.75     64.5      1.34   0.206
 1.00   48.0  1.00     66.0      1.42   0.205
 1.00   51.0  0.50     55.5      1.39   0.203
 1.00   51.0  0.75     54.9      1.44   0.206
 1.00   51.0  1.00     57.4      1.65   0.204
 1.00   54.0  0.50     45.6      1.49   0.203
 1.00   54.0  0.75     45.4      1.59   0.204
 1.00   54.0  1.00     47.7      1.85   0.204
 1.25    6.0  0.50     79.8      2.30   0.337
 1.25    6.0  0.75     94.9      2.33   0.344
 1.25    6.0  1.00    123.5      2.24   0.340
 1.25    9.0  0.50    140.1      2.08   0.338
 1.25    9.0  0.75    160.3      2.05   0.343
 1.25    9.0  1.00    182.2      1.98   0.341
 1.25   12.0  0.50    187.1      2.01   0.348
 1.25   12.0  0.75    213.0      2.09   0.352
 1.25   12.0  1.00    233.9      2.26   0.350
 1.25   15.0  0.50    233.1      2.33   0.358
 1.25   15.0  0.75    270.6      2.68   0.365
 1.25   15.0  1.00    285.4      2.83   0.359
 1.25   18.0  0.50    276.8      2.67   0.359
 1.25   18.0  0.75    300.6      2.78   0.367
 1.25   18.0  1.00    312.5      2.88   0.356
 1.25   21.0  0.50    292.3      2.54   0.348
 1.25   21.0  0.75    315.0      2.66   0.353
 1.25   21.0  1.00    319.4      2.75   0.340
 1.25   24.0  0.50    294.9      2.38   0.328
 1.25   24.0  0.75    317.1      2.51   0.331
 1.25   24.0  1.00    314.4      2.64   0.317
 1.25   27.0  0.50    285.0      2.29   0.303
 1.25   27.0  0.75    299.4      2.45   0.305
 1.25   27.0  1.00    296.2      2.62   0.292
 1.25   30.0  0.50    264.3      2.28   0.278
 1.25   30.0  0.75    274.1      2.45   0.279
 1.25   30.0  1.00    274.4      2.67   0.268
 1.25   33.0  0.50    239.8      2.36   0.253
 1.25   33.0  0.75    246.6      2.54   0.254
 1.25   33.0  1.00    246.9      2.74   0.246
 1.25   36.0  0.50    214.4      2.52   0.233
 1.25   36.0  0.75    219.1      2.70   0.231
 1.25   36.0  1.00    180.9      2.16   0.227
 1.25   39.0  0.50    126.7      1.71   0.218
 1.25   39.0  0.75    133.8      1.76   0.220
 1.25   39.0  1.00    146.3      1.77   0.218
 1.25   42.0  0.50    107.6      1.65   0.215
 1.25   42.0  0.75    113.0      1.65   0.217
 1.25   42.0  1.00    123.7      1.70   0.215
 1.25   45.0  0.50     91.3      1.59   0.213
 1.25   45.0  0.75    100.1      1.74   0.215
 1.25   45.0  1.00     97.0      1.48   0.214
 1.25   48.0  0.50     80.3      1.66   0.211
 1.25   48.0  0.75     82.1      1.65   0.213
 1.25   48.0  1.00     80.4      1.46   0.212
 1.25   51.0  0.50     69.7      1.67   0.210
 1.25   51.0  0.75     66.2      1.57   0.211
 1.25   51.0  1.00     68.9      1.57   0.211
 1.25   54.0  0.50     57.4      1.64   0.208
 1.25   54.0  0.75     53.2      1.55   0.210
 1.25   54.0  1.00     58.8      1.76   0.209
 1.50    6.0  0.50     87.4      2.51   0.389
 1.50    6.0  0.75    105.3      2.51   0.395
 1.50    6.0  1.00    134.2      2.45   0.391
 1.50    9.0  0.50    151.9      2.26   0.393
 1.50    9.0  0.75    173.3      2.24   0.399
 1.50    9.0  1.00    198.1      2.21   0.396
 1.50   12.0  0.50    213.1      2.40   0.408
 1.50   12.0  0.75    242.9      2.55   0.414
 1.50   12.0  1.00    265.7      2.77   0.412
 1.50   15.0  0.50    276.3      2.91   0.422
 1.50   15.0  0.75    306.5      3.02   0.429
 1.50   15.0  1.00    318.7      3.09   0.421
 1.50   18.0  0.50    308.3      2.78   0.420
 1.50   18.0  0.75    346.3      2.89   0.429
 1.50   18.0  1.00    347.6      2.90   0.418
 1.50   21.0  0.50    334.6      2.54   0.404
 1.50   21.0  0.75    364.3      2.57   0.412
 1.50   21.0  1.00    361.0      2.58   0.395
 1.50   24.0  0.50    361.8      2.26   0.380
 1.50   24.0  0.75    370.4      2.26   0.383
 1.50   24.0  1.00    360.2      2.37   0.367
 1.50   27.0  0.50    358.7      2.05   0.347
 1.50   27.0  0.75    362.6      2.10   0.350
 1.50   27.0  1.00    342.3      2.33   0.335
 1.50   30.0  0.50    331.6      2.01   0.316
 1.50   30.0  0.75    336.9      2.14   0.318
 1.50   30.0  1.00    316.6      2.44   0.306
 1.50   33.0  0.50    275.1      2.17   0.288
 1.50   33.0  0.75    287.6      2.37   0.289
 1.50   33.0  1.00    287.1      2.60   0.281
 1.50   36.0  0.50    243.4      2.35   0.263
 1.50   36.0  0.75    251.4      2.55   0.264
 1.50   36.0  1.00    253.3      2.78   0.258
 1.50   39.0  0.50    215.0      2.57   0.243
 1.50   39.0  0.75    217.9      2.77   0.242
 1.50   39.0  1.00    225.3      2.89   0.238
 1.50   42.0  0.50    193.7      2.86   0.226
 1.50   42.0  0.75    194.4      2.94   0.226
 1.50   42.0  1.00    147.1      2.01   0.225
 1.50   45.0  0.50    105.1      1.83   0.221
 1.50   45.0  0.75    110.7      1.85   0.223
 1.50   45.0  1.00    124.5      1.94   0.222
 1.50   48.0  0.50     92.7      1.84   0.219
 1.50   48.0  0.75     96.4      1.89   0.220
 1.50   48.0  1.00    105.6      1.84   0.220
 1.50   51.0  0.50     81.2      1.92   0.216
 1.50   51.0  0.75     84.9      1.98   0.218
 1.50   51.0  1.00     84.4      1.71   0.218
 1.50   54.0  0.50     69.6      1.91   0.215
 1.50   54.0  0.75     72.7      1.96   0.216
 1.50   54.0  1.00     71.6      1.77   0.216
 1.75    6.0  0.50    100.4      2.72   0.444
 1.75    6.0  0.75    114.3      2.72   0.453
 1.75    6.0  1.00    144.0      2.68   0.450
 1.75    9.0  0.50    166.4      2.48   0.456
 1.75    9.0  0.75    188.5      2.47   0.461
 1.75    9.0  1.00    217.0      2.50   0.459
 1.75   12.0  0.50    237.5      2.72   0.476
 1.75   12.0  0.75    271.8      2.97   0.483
 1.75   12.0  1.00    294.5      3.18   0.480
 1.75   15.0  0.50    304.0      3.09   0.491
 1.75   15.0  0.75    339.1      3.18   0.500
 1.75   15.0  1.00    347.3      3.19   0.490
 1.75   18.0  0.50    345.8      2.85   0.486
 1.75   18.0  0.75    385.7      2.78   0.495
 1.75   18.0  1.00    383.6      2.77   0.480
 1.75   21.0  0.50    384.5      2.51   0.464
 1.75   21.0  0.75    416.5      2.33   0.472
 1.75   21.0  1.00    408.1      2.30   0.454
 1.75   24.0  0.50    403.8      2.18   0.432
 1.75   24.0  0.75    433.6      1.99   0.436
 1.75   24.0  1.00    414.0      1.99   0.420
 1.75   27.0  0.50    411.9      1.93   0.396
 1.75   27.0  0.75    427.1      1.77   0.397
 1.75   27.0  1.00    401.7      1.83   0.383
 1.75   30.0  0.50    400.7      1.73   0.361
 1.75   30.0  0.75    401.3      1.67   0.361
 1.75   30.0  1.00    377.5      1.85   0.348
 1.75   33.0  0.50    379.1      1.67   0.327
 1.75   33.0  0.75    368.1      1.73   0.324
 1.75   33.0  1.00    335.0      2.13   0.317
 1.75   36.0  0.50    289.7      2.03   0.296
 1.75   36.0  0.75    315.7      2.12   0.294
 1.75   36.0  1.00    295.8      2.50   0.292
 1.75   39.0  0.50    243.8      2.34   0.270
 1.75   39.0  0.75    251.2      2.55   0.271
 1.75   39.0  1.00    257.0      2.77   0.269
 1.75   42.0  0.50    212.9      2.67   0.250
 1.75   42.0  0.75    220.7      2.84   0.251
 1.75   42.0  1.00    229.2      3.02   0.247
 1.75   45.0  0.50    189.5      2.92   0.236
 1.75   45.0  0.75    191.5      3.08   0.235
 1.75   45.0  1.00    151.1      2.35   0.233
 1.75   48.0  0.50    109.8      2.26   0.227
 1.75   48.0  0.75    114.9      2.25   0.228
 1.75   48.0  1.00    124.5      2.17   0.229
 1.75   51.0  0.50     92.0      2.11   0.224
 1.75   51.0  0.75     96.0      2.16   0.225
 1.75   51.0  1.00    106.8      2.14   0.227
 1.75   54.0  0.50     80.8      2.16   0.222
 1.75   54.0  0.75     81.5      2.08   0.223
 1.75   54.0  1.00     85.4      1.97   0.224
 2.00    6.0  0.50    107.6      2.91   0.508
 2.00    6.0  0.75    123.8      2.91   0.516
 2.00    6.0  1.00    153.6      2.91   0.513
 2.00    9.0  0.50    178.8      2.70   0.524
 2.00    9.0  0.75    204.8      2.71   0.530
 2.00    9.0  1.00    235.7      2.85   0.529
 2.00   12.0  0.50    265.0      3.15   0.549
 2.00   12.0  0.75    303.5      3.42   0.561
 2.00   12.0  1.00    319.9      3.49   0.555
 2.00   15.0  0.50    322.3      3.23   0.560
 2.00   15.0  0.75    364.0      3.30   0.573
 2.00   15.0  1.00    377.1      3.21   0.564
 2.00   18.0  0.50    380.5      2.93   0.552
 2.00   18.0  0.75    417.6      2.80   0.563
 2.00   18.0  1.00    428.4      2.63   0.550
 2.00   21.0  0.50    424.5      2.51   0.527
 2.00   21.0  0.75    453.4      2.35   0.534
 2.00   21.0  1.00    459.0      2.22   0.518
 2.00   24.0  0.50    443.6      2.18   0.489
 2.00   24.0  0.75    461.9      2.03   0.495
 2.00   24.0  1.00    457.3      1.90   0.477
 2.00   27.0  0.50    445.6      1.93   0.449
 2.00   27.0  0.75    455.5      1.79   0.450
 2.00   27.0  1.00    447.4      1.71   0.434
 2.00   30.0  0.50    437.6      1.71   0.408
 2.00   30.0  0.75    440.2      1.59   0.406
 2.00   30.0  1.00    430.0      1.56   0.392
 2.00   33.0  0.50    418.7      1.52   0.367
 2.00   33.0  0.75    420.0      1.44   0.364
 2.00   33.0  1.00    395.3      1.59   0.358
 2.00   36.0  0.50    394.4      1.45   0.331
 2.00   36.0  0.75    387.5      1.48   0.331
 2.00   36.0  1.00    346.0      1.93   0.327
 2.00   39.0  0.50    354.6      1.59   0.302
 2.00   39.0  0.75    329.5      1.87   0.302
 2.00   39.0  1.00    308.2      2.32   0.302
 2.00   42.0  0.50    246.8      2.28   0.279
 2.00   42.0  0.75    252.2      2.51   0.279
 2.00   42.0  1.00    267.7      2.73   0.278
 2.00   45.0  0.50    214.2      2.63   0.261
 2.00   45.0  0.75    217.4      2.86   0.261
 2.00   45.0  1.00    232.5      3.13   0.255
 2.00   48.0  0.50    190.5      2.99   0.247
 2.00   48.0  0.75    193.9      3.18   0.244
 2.00   48.0  1.00    205.5      3.43   0.240
 2.00   51.0  0.50    169.4      3.33   0.234
 2.00   51.0  0.75    175.7      3.48   0.234
 2.00   51.0  1.00    116.4      2.23   0.236
 2.00   54.0  0.50     93.6      2.46   0.230
 2.00   54.0  0.75     97.1      2.37   0.231
 2.00   54.0  1.00     93.6      2.17   0.233
 2.25    6.0  0.50    111.3      3.15   0.580
 2.25    6.0  0.75    133.7      3.14   0.587
 2.25    6.0  1.00    158.3      3.10   0.584
 2.25    9.0  0.50    193.7      2.97   0.597
 2.25    9.0  0.75    223.4      3.01   0.602
 2.25    9.0  1.00    252.2      3.21   0.606
 2.25   12.0  0.50    287.3      3.53   0.623
 2.25   12.0  0.75    315.0      3.63   0.633
 2.25   12.0  1.00    344.0      3.65   0.634
 2.25   15.0  0.50    354.5      3.35   0.636
 2.25   15.0  0.75    390.6      3.37   0.650
 2.25   15.0  1.00    411.7      3.19   0.643
 2.25   18.0  0.50    421.3      2.95   0.626
 2.25   18.0  0.75    453.7      2.80   0.637
 2.25   18.0  1.00    461.1      2.71   0.622
 2.25   21.0  0.50    461.7      2.56   0.593
 2.25   21.0  0.75    488.2      2.40   0.600
 2.25   21.0  1.00    483.2      2.35   0.585
 2.25   24.0  0.50    477.9      2.33   0.552
 2.25   24.0  0.75    499.4      2.26   0.557
 2.25   24.0  1.00    488.1      2.10   0.540
 2.25   27.0  0.50    482.9      2.17   0.503
 2.25   27.0  0.75    501.3      2.11   0.505
 2.25   27.0  1.00    480.5      1.87   0.490
 2.25   30.0  0.50    476.8      1.95   0.454
 2.25   30.0  0.75    482.9      1.80   0.452
 2.25   30.0  1.00    460.2      1.61   0.442
 2.25   33.0  0.50    458.3      1.66   0.410
 2.25   33.0  0.75    453.2      1.46   0.405
 2.25   33.0  1.00    438.1      1.45   0.402
 2.25   36.0  0.50    428.2      1.40   0.369
 2.25   36.0  0.75    427.3      1.30   0.367
 2.25   36.0  1.00    413.8      1.43   0.367
 2.25   39.0  0.50    400.7      1.33   0.336
 2.25   39.0  0.75    400.0      1.31   0.337
 2.25   39.0  1.00    377.1      1.53   0.338
 2.25   42.0  0.50    369.5      1.45   0.308
 2.25   42.0  0.75    352.3      1.52   0.308
 2.25   42.0  1.00    319.6      2.07   0.309
 2.25   45.0  0.50    254.4      2.19   0.288
 2.25   45.0  0.75    256.8      2.45   0.285
 2.25   45.0  1.00    268.9      2.72   0.283
 2.25   48.0  0.50    214.5      2.67   0.268
 2.25   48.0  0.75    224.7      2.83   0.271
 2.25   48.0  1.00    223.9      3.26   0.259
 2.25   51.0  0.50    192.3      3.03   0.256
 2.25   51.0  0.75    196.2      3.24   0.255
 2.25   51.0  1.00    196.7      3.55   0.247
 2.25   54.0  0.50    171.4      3.44   0.244
 2.25   54.0  0.75    172.3      3.61   0.242
 2.25   54.0  1.00    104.4      2.30   0.242
 2.50    6.0  0.50    109.7      3.39   0.658
 2.50    6.0  0.75    133.6      3.37   0.661
 2.50    6.0  1.00    164.8      3.30   0.662
 2.50    9.0  0.50    206.1      3.23   0.675
 2.50    9.0  0.75    225.0      3.26   0.682
 2.50    9.0  1.00    272.9      3.59   0.686
 2.50   12.0  0.50    303.1      3.73   0.705
 2.50   12.0  0.75    331.1      3.83   0.718
 2.50   12.0  1.00    356.3      3.85   0.718
 2.50   15.0  0.50    386.6      3.48   0.719
 2.50   15.0  0.75    421.1      3.40   0.732
 2.50   15.0  1.00    424.3      3.35   0.722
 2.50   18.0  0.50    457.2      3.05   0.705
 2.50   18.0  0.75    486.1      2.90   0.716
 2.50   18.0  1.00    483.3      2.84   0.700
 2.50   21.0  0.50    493.1      2.76   0.667
 2.50   21.0  0.75    524.0      2.78   0.676
 2.50   21.0  1.00    510.5      2.61   0.658
 2.50   24.0  0.50    528.8      2.87   0.618
 2.50   24.0  0.75    559.0      3.06   0.618
 2.50   24.0  1.00    531.7      2.66   0.606
 2.50   27.0  0.50    537.5      2.84   0.560
 2.50   27.0  0.75    564.7      3.07   0.561
 2.50   27.0  1.00    535.0      2.61   0.548
 2.50   30.0  0.50    531.9      2.68   0.504
 2.50   30.0  0.75    543.7      2.73   0.503
 2.50   30.0  1.00    511.7      2.20   0.493
 2.50   33.0  0.50    509.2      2.31   0.451
 2.50   33.0  0.75    516.3      2.27   0.454
 2.50   33.0  1.00    477.2      1.67   0.447
 2.50   36.0  0.50    479.7      1.89   0.406
 2.50   36.0  0.75    471.7      1.59   0.405
 2.50   36.0  1.00    450.4      1.42   0.411
 2.50   39.0  0.50    435.7      1.35   0.368
 2.50   39.0  0.75    429.5      1.22   0.367
 2.50   39.0  1.00    421.8      1.27   0.375
 2.50   42.0  0.50    404.2      1.24   0.340
 2.50   42.0  0.75    400.3      1.20   0.339
 2.50   42.0  1.00    395.1      1.30   0.340
 2.50   45.0  0.50    375.9      1.33   0.314
 2.50   45.0  0.75    366.1      1.36   0.313
 2.50   45.0  1.00    323.8      1.89   0.311
 2.50   48.0  0.50    339.3      1.57   0.295
 2.50   48.0  0.75    304.8      1.93   0.296
 2.50   48.0  1.00    258.4      2.85   0.285
 2.50   51.0  0.50    218.0      2.60   0.278
 2.50   51.0  0.75    227.4      2.82   0.279
 2.50   51.0  1.00    215.9      3.37   0.264
 2.50   54.0  0.50    191.5      3.09   0.264
 2.50   54.0  0.75    195.3      3.32   0.264
 2.50   54.0  1.00    133.4      2.82   0.253
 2.75    6.0  0.50    113.6      3.62   0.739
 2.75    6.0  0.75    128.7      3.60   0.744
 2.75    6.0  1.00    164.4      3.56   0.746
 2.75    9.0  0.50    226.6      3.56   0.760
 2.75    9.0  0.75    238.0      3.56   0.766
 2.75    9.0  1.00    261.1      3.72   0.772
 2.75   12.0  0.50    328.0      3.90   0.794
 2.75   12.0  0.75    351.1      4.04   0.807
 2.75   12.0  1.00    361.5      4.04   0.803
 2.75   15.0  0.50    418.3      3.56   0.806
 2.75   15.0  0.75    452.8      3.47   0.822
 2.75   15.0  1.00    448.6      3.44   0.809
 2.75   18.0  0.50    487.1      3.16   0.786
 2.75   18.0  0.75    515.5      3.09   0.799
 2.75   18.0  1.00    507.6      3.01   0.780
 2.75   21.0  0.50    539.0      3.30   0.743
 2.75   21.0  0.75    571.9      3.48   0.753
 2.75   21.0  1.00    555.4      3.22   0.734
 2.75   24.0  0.50    582.1      3.61   0.685
 2.75   24.0  0.75    619.3      3.96   0.693
 2.75   24.0  1.00    578.4      3.59   0.673
 2.75   27.0  0.50    616.1      3.72   0.620
 2.75   27.0  0.75    631.4      3.82   0.623
 2.75   27.0  1.00    607.9      3.77   0.609
 2.75   30.0  0.50    610.4      3.53   0.557
 2.75   30.0  0.75    622.1      3.65   0.559
 2.75   30.0  1.00    591.0      3.54   0.549
 2.75   33.0  0.50    596.1      3.36   0.498
 2.75   33.0  0.75    602.1      3.48   0.501
 2.75   33.0  1.00    545.0      2.83   0.494
 2.75   36.0  0.50    577.6      3.24   0.448
 2.75   36.0  0.75    575.7      3.30   0.450
 2.75   36.0  1.00    502.1      2.12   0.452
 2.75   39.0  0.50    487.9      2.04   0.404
 2.75   39.0  0.75    490.2      1.95   0.406
 2.75   39.0  1.00    464.0      1.51   0.413
 2.75   42.0  0.50    443.3      1.43   0.371
 2.75   42.0  0.75    431.6      1.18   0.367
 2.75   42.0  1.00    425.9      1.16   0.372
 2.75   45.0  0.50    402.9      1.17   0.343
 2.75   45.0  0.75    395.1      1.12   0.341
 2.75   45.0  1.00    395.4      1.18   0.340
 2.75   48.0  0.50    376.5      1.23   0.320
 2.75   48.0  0.75    374.2      1.24   0.324
 2.75   48.0  1.00    315.0      1.96   0.313
 2.75   51.0  0.50    354.3      1.42   0.303
 2.75   51.0  0.75    329.7      1.62   0.303
 2.75   51.0  1.00    243.5      3.04   0.287
 2.75   54.0  0.50    225.7      2.51   0.287
 2.75   54.0  0.75    228.1      2.82   0.287
 2.75   54.0  1.00    200.1      3.52   0.265
 3.00    6.0  0.50    109.4      3.86   0.829
 3.00    6.0  0.75    140.9      3.82   0.827
 3.00    6.0  1.00    156.6      3.83   0.833
 3.00    9.0  0.50    232.3      3.80   0.853
 3.00    9.0  0.75    255.7      3.89   0.857
 3.00    9.0  1.00    263.1      3.95   0.862
 3.00   12.0  0.50    350.2      4.09   0.890
 3.00   12.0  0.75    370.9      4.23   0.904
 3.00   12.0  1.00    374.3      4.21   0.899
 3.00   15.0  0.50    450.8      3.67   0.904
 3.00   15.0  0.75    481.5      3.60   0.916
 3.00   15.0  1.00    476.1      3.58   0.901
 3.00   18.0  0.50    519.1      3.45   0.873
 3.00   18.0  0.75    554.0      3.56   0.887
 3.00   18.0  1.00    535.3      3.38   0.871
 3.00   21.0  0.50    586.5      3.93   0.820
 3.00   21.0  0.75    628.6      4.28   0.834
 3.00   21.0  1.00    602.2      3.98   0.814
 3.00   24.0  0.50    634.3      4.04   0.754
 3.00   24.0  0.75    663.4      4.14   0.765
 3.00   24.0  1.00    646.3      4.22   0.745
 3.00   27.0  0.50    661.3      3.75   0.684
 3.00   27.0  0.75    685.7      3.80   0.692
 3.00   27.0  1.00    652.1      4.01   0.673
 3.00   30.0  0.50    668.1      3.42   0.610
 3.00   30.0  0.75    686.7      3.50   0.616
 3.00   30.0  1.00    643.9      3.83   0.602
 3.00   33.0  0.50    648.7      3.29   0.551
 3.00   33.0  0.75    660.7      3.41   0.548
 3.00   33.0  1.00    616.6      3.70   0.543
 3.00   36.0  0.50    627.0      3.20   0.495
 3.00   36.0  0.75    626.1      3.40   0.493
 3.00   36.0  1.00    594.4      3.56   0.493
 3.00   39.0  0.50    583.3      3.20   0.440
 3.00   39.0  0.75    582.2      3.32   0.442
 3.00   39.0  1.00    533.6      2.67   0.450
 3.00   42.0  0.50    556.2      3.14   0.399
 3.00   42.0  0.75    495.6      2.10   0.406
 3.00   42.0  1.00    479.2      1.71   0.409
 3.00   45.0  0.50    458.4      1.75   0.374
 3.00   45.0  0.75    443.1      1.34   0.376
 3.00   45.0  1.00    424.9      1.11   0.375
 3.00   48.0  0.50    415.6      1.19   0.348
 3.00   48.0  0.75    403.8      1.11   0.350
 3.00   48.0  1.00    378.3      1.23   0.342
 3.00   51.0  0.50    382.3      1.17   0.328
 3.00   51.0  0.75    373.9      1.19   0.328
 3.00   51.0  1.00    297.7      2.18   0.314
 3.00   54.0  0.50    356.9      1.30   0.312
 3.00   54.0  0.75    337.9      1.46   0.309
 3.00   54.0  1.00    233.5      3.20   0.288
```

</details>

<details>
<summary>z0 = 0.10 (156/459 close)</summary>

```
   vz     w0  tuck  rot_deg  land_m/s  apex_m
 1.00    6.0  0.50     65.9      1.85   0.245
 1.00    6.0  0.75     77.0      1.87   0.250
 1.00    6.0  1.00    107.8      1.82   0.245
 1.00    9.0  0.50    117.5      1.79   0.242
 1.00    9.0  0.75    135.5      1.76   0.247
 1.00    9.0  1.00    156.4      1.59   0.245
 1.00   12.0  0.50    148.9      1.46   0.245
 1.00   12.0  0.75    170.7      1.43   0.248
 1.00   12.0  1.00    184.8      1.41   0.246
 1.00   15.0  0.50    178.5      1.49   0.252
 1.00   15.0  0.75    205.6      1.61   0.257
 1.00   15.0  1.00    216.1      1.67   0.252
 1.00   18.0  0.50    198.5      1.58   0.254
 1.00   18.0  0.75    229.2      1.87   0.259
 1.00   18.0  1.00    239.7      2.01   0.251
 1.00   21.0  0.50    214.9      1.77   0.247
 1.00   21.0  0.75    260.6      2.36   0.251
 1.00   21.0  1.00    242.7      2.10   0.239
 1.00   24.0  0.50    242.5      2.18   0.232
 1.00   24.0  0.75    254.7      2.28   0.233
 1.00   24.0  1.00    222.6      1.88   0.221
 1.00   27.0  0.50    171.4      1.28   0.212
 1.00   27.0  0.75    200.4      1.65   0.213
 1.00   27.0  1.00    189.9      1.50   0.202
 1.00   30.0  0.50    141.4      0.97   0.191
 1.00   30.0  0.75    154.0      1.06   0.191
 1.00   30.0  1.00    155.7      1.09   0.182
 1.00   33.0  0.50    122.8      0.88   0.172
 1.00   33.0  0.75    126.9      0.81   0.171
 1.00   33.0  1.00    117.6      0.58   0.165
 1.00   36.0  0.50     95.2      0.49   0.177
 1.00   36.0  0.75     96.0      0.39   0.173
 1.00   36.0  1.00     96.7      0.51   0.162
 1.00   39.0  0.50     77.2      0.36   0.189
 1.00   39.0  0.75     78.0      0.35   0.177
 1.00   39.0  1.00     80.4      0.54   0.165
 1.00   42.0  0.50     62.4      0.37   0.193
 1.00   42.0  0.75     64.6      0.47   0.181
 1.00   42.0  1.00     69.2      0.69   0.166
 1.00   45.0  0.50     51.3      0.53   0.195
 1.00   45.0  0.75     52.3      0.60   0.179
 1.00   45.0  1.00     59.5      0.90   0.165
 1.00   48.0  0.50     42.8      0.70   0.190
 1.00   48.0  0.75     42.7      0.80   0.172
 1.00   48.0  1.00     48.2      1.13   0.164
 1.00   51.0  0.50     33.0      0.87   0.173
 1.00   51.0  0.75     36.1      1.04   0.162
 1.00   51.0  1.00     38.8      1.37   0.154
 1.00   54.0  0.50     25.7      1.10   0.161
 1.00   54.0  0.75     25.4      1.26   0.162
 1.00   54.0  1.00     32.2      1.64   0.154
 1.25    6.0  0.50     74.1      2.08   0.287
 1.25    6.0  0.75     88.7      2.11   0.294
 1.25    6.0  1.00    115.7      2.03   0.290
 1.25    9.0  0.50    132.9      1.90   0.288
 1.25    9.0  0.75    152.5      1.87   0.293
 1.25    9.0  1.00    172.2      1.76   0.291
 1.25   12.0  0.50    174.2      1.73   0.298
 1.25   12.0  0.75    196.7      1.73   0.302
 1.25   12.0  1.00    214.9      1.84   0.300
 1.25   15.0  0.50    212.9      1.92   0.308
 1.25   15.0  0.75    247.6      2.23   0.315
 1.25   15.0  1.00    261.0      2.38   0.309
 1.25   18.0  0.50    264.1      2.48   0.309
 1.25   18.0  0.75    285.5      2.59   0.317
 1.25   18.0  1.00    294.9      2.68   0.306
 1.25   21.0  0.50    273.9      2.35   0.298
 1.25   21.0  0.75    293.8      2.49   0.303
 1.25   21.0  1.00    299.5      2.61   0.290
 1.25   24.0  0.50    273.8      2.24   0.278
 1.25   24.0  0.75    288.9      2.38   0.281
 1.25   24.0  1.00    290.2      2.53   0.267
 1.25   27.0  0.50    258.8      2.14   0.253
 1.25   27.0  0.75    270.7      2.31   0.255
 1.25   27.0  1.00    269.9      2.48   0.242
 1.25   30.0  0.50    240.4      2.15   0.228
 1.25   30.0  0.75    247.8      2.30   0.229
 1.25   30.0  1.00    248.6      2.49   0.218
 1.25   33.0  0.50    145.4      1.22   0.203
 1.25   33.0  0.75    161.9      1.39   0.204
 1.25   33.0  1.00    164.2      1.41   0.196
 1.25   36.0  0.50    118.1      0.96   0.183
 1.25   36.0  0.75    128.7      1.06   0.181
 1.25   36.0  1.00    128.7      0.98   0.177
 1.25   39.0  0.50     99.7      0.85   0.168
 1.25   39.0  0.75    100.0      0.72   0.170
 1.25   39.0  1.00     98.9      0.64   0.168
 1.25   42.0  0.50     76.2      0.53   0.183
 1.25   42.0  0.75     77.6      0.50   0.177
 1.25   42.0  1.00     82.5      0.67   0.166
 1.25   45.0  0.50     61.8      0.52   0.193
 1.25   45.0  0.75     63.0      0.53   0.182
 1.25   45.0  1.00     66.8      0.75   0.164
 1.25   48.0  0.50     49.3      0.57   0.194
 1.25   48.0  0.75     50.1      0.66   0.183
 1.25   48.0  1.00     55.7      0.92   0.164
 1.25   51.0  0.50     40.4      0.70   0.189
 1.25   51.0  0.75     40.1      0.83   0.174
 1.25   51.0  1.00     46.7      1.13   0.165
 1.25   54.0  0.50     33.9      0.90   0.185
 1.25   54.0  0.75     33.5      1.04   0.165
 1.25   54.0  1.00     39.7      1.39   0.159
 1.50    6.0  0.50     82.3      2.30   0.339
 1.50    6.0  0.75    100.2      2.33   0.345
 1.50    6.0  1.00    127.3      2.25   0.341
 1.50    9.0  0.50    145.3      2.09   0.343
 1.50    9.0  0.75    165.8      2.06   0.349
 1.50    9.0  1.00    188.5      1.99   0.346
 1.50   12.0  0.50    200.3      2.10   0.358
 1.50   12.0  0.75    230.2      2.26   0.364
 1.50   12.0  1.00    253.1      2.49   0.362
 1.50   15.0  0.50    257.8      2.57   0.372
 1.50   15.0  0.75    295.0      2.84   0.379
 1.50   15.0  1.00    306.5      2.93   0.371
 1.50   18.0  0.50    294.1      2.63   0.370
 1.50   18.0  0.75    325.9      2.75   0.379
 1.50   18.0  1.00    330.2      2.80   0.368
 1.50   21.0  0.50    310.8      2.42   0.354
 1.50   21.0  0.75    343.3      2.53   0.362
 1.50   21.0  1.00    341.2      2.57   0.345
 1.50   24.0  0.50    318.1      2.21   0.330
 1.50   24.0  0.75    346.8      2.29   0.333
 1.50   24.0  1.00    336.0      2.41   0.317
 1.50   27.0  0.50    302.3      2.08   0.297
 1.50   27.0  0.75    331.5      2.19   0.300
 1.50   27.0  1.00    313.5      2.40   0.285
 1.50   30.0  0.50    275.7      2.04   0.266
 1.50   30.0  0.75    287.0      2.24   0.268
 1.50   30.0  1.00    285.7      2.47   0.256
 1.50   33.0  0.50    245.7      2.12   0.238
 1.50   33.0  0.75    253.0      2.32   0.239
 1.50   33.0  1.00    256.5      2.54   0.231
 1.50   36.0  0.50    218.0      2.27   0.213
 1.50   36.0  0.75    223.4      2.43   0.214
 1.50   36.0  1.00    170.7      1.71   0.208
 1.50   39.0  0.50    117.3      1.13   0.193
 1.50   39.0  0.75    123.9      1.18   0.192
 1.50   39.0  1.00    137.0      1.30   0.188
 1.50   42.0  0.50     98.2      1.03   0.176
 1.50   42.0  0.75    107.6      1.14   0.176
 1.50   42.0  1.00    105.7      0.92   0.175
 1.50   45.0  0.50     81.8      0.95   0.171
 1.50   45.0  0.75     80.2      0.78   0.173
 1.50   45.0  1.00     84.0      0.79   0.172
 1.50   48.0  0.50     64.8      0.72   0.177
 1.50   48.0  0.75     64.7      0.73   0.179
 1.50   48.0  1.00     70.5      0.85   0.170
 1.50   51.0  0.50     51.9      0.73   0.187
 1.50   51.0  0.75     51.7      0.77   0.183
 1.50   51.0  1.00     58.6      1.01   0.168
 1.50   54.0  0.50     42.4      0.83   0.191
 1.50   54.0  0.75     41.7      0.87   0.178
 1.50   54.0  1.00     48.7      1.18   0.166
 1.75    6.0  0.50     96.2      2.55   0.394
 1.75    6.0  0.75    109.3      2.53   0.403
 1.75    6.0  1.00    138.0      2.49   0.400
 1.75    9.0  0.50    159.6      2.29   0.406
 1.75    9.0  0.75    181.2      2.28   0.411
 1.75    9.0  1.00    207.5      2.27   0.409
 1.75   12.0  0.50    226.9      2.47   0.426
 1.75   12.0  0.75    260.4      2.71   0.433
 1.75   12.0  1.00    281.0      2.91   0.430
 1.75   15.0  0.50    293.4      2.93   0.441
 1.75   15.0  0.75    326.1      3.03   0.450
 1.75   15.0  1.00    333.8      3.06   0.440
 1.75   18.0  0.50    328.6      2.73   0.436
 1.75   18.0  0.75    370.4      2.74   0.445
 1.75   18.0  1.00    367.9      2.74   0.430
 1.75   21.0  0.50    364.8      2.45   0.414
 1.75   21.0  0.75    395.4      2.31   0.422
 1.75   21.0  1.00    384.7      2.32   0.404
 1.75   24.0  0.50    377.7      2.13   0.382
 1.75   24.0  0.75    407.8      1.94   0.386
 1.75   24.0  1.00    389.9      2.03   0.370
 1.75   27.0  0.50    384.9      1.87   0.346
 1.75   27.0  0.75    398.6      1.75   0.347
 1.75   27.0  1.00    368.2      1.96   0.333
 1.75   30.0  0.50    365.6      1.73   0.311
 1.75   30.0  0.75    362.6      1.79   0.311
 1.75   30.0  1.00    336.3      2.09   0.298
 1.75   33.0  0.50    291.8      1.91   0.277
 1.75   33.0  0.75    319.5      2.01   0.274
 1.75   33.0  1.00    296.3      2.34   0.267
 1.75   36.0  0.50    248.3      2.08   0.246
 1.75   36.0  0.75    253.2      2.32   0.244
 1.75   36.0  1.00    264.0      2.55   0.242
 1.75   39.0  0.50    212.3      2.34   0.220
 1.75   39.0  0.75    221.3      2.51   0.221
 1.75   39.0  1.00    227.0      2.65   0.219
 1.75   42.0  0.50    113.5      1.34   0.200
 1.75   42.0  0.75    124.0      1.41   0.201
 1.75   42.0  1.00    138.5      1.53   0.197
 1.75   45.0  0.50     95.3      1.18   0.186
 1.75   45.0  0.75    102.6      1.29   0.185
 1.75   45.0  1.00    114.3      1.31   0.183
 1.75   48.0  0.50     81.4      1.18   0.177
 1.75   48.0  0.75     86.2      1.22   0.178
 1.75   48.0  1.00     85.9      0.95   0.179
 1.75   51.0  0.50     66.3      1.01   0.174
 1.75   51.0  0.75     66.4      0.99   0.175
 1.75   51.0  1.00     70.2      0.95   0.177
 1.75   54.0  0.50     54.6      0.96   0.172
 1.75   54.0  0.75     54.2      0.97   0.173
 1.75   54.0  1.00     55.5      1.06   0.174
 2.00    6.0  0.50    104.0      2.76   0.458
 2.00    6.0  0.75    119.8      2.76   0.466
 2.00    6.0  1.00    148.3      2.74   0.463
 2.00    9.0  0.50    172.9      2.53   0.474
 2.00    9.0  0.75    198.3      2.54   0.480
 2.00    9.0  1.00    226.4      2.61   0.479
 2.00   12.0  0.50    253.6      2.89   0.499
 2.00   12.0  0.75    293.3      3.21   0.511
 2.00   12.0  1.00    310.3      3.32   0.505
 2.00   15.0  0.50    313.3      3.10   0.510
 2.00   15.0  0.75    352.9      3.20   0.523
 2.00   15.0  1.00    366.8      3.14   0.514
 2.00   18.0  0.50    365.2      2.84   0.502
 2.00   18.0  0.75    399.5      2.73   0.513
 2.00   18.0  1.00    412.8      2.60   0.500
 2.00   21.0  0.50    410.4      2.43   0.477
 2.00   21.0  0.75    436.6      2.24   0.484
 2.00   21.0  1.00    441.2      2.12   0.468
 2.00   24.0  0.50    431.6      2.09   0.439
 2.00   24.0  0.75    449.2      1.93   0.445
 2.00   24.0  1.00    444.5      1.81   0.427
 2.00   27.0  0.50    432.3      1.81   0.399
 2.00   27.0  0.75    441.7      1.69   0.400
 2.00   27.0  1.00    433.3      1.64   0.384
 2.00   30.0  0.50    420.4      1.58   0.358
 2.00   30.0  0.75    425.0      1.50   0.356
 2.00   30.0  1.00    399.5      1.54   0.342
 2.00   33.0  0.50    404.6      1.45   0.317
 2.00   33.0  0.75    394.8      1.41   0.314
 2.00   33.0  1.00    355.2      1.80   0.308
 2.00   36.0  0.50    361.5      1.47   0.281
 2.00   36.0  0.75    335.8      1.73   0.281
 2.00   36.0  1.00    312.1      2.18   0.277
 2.00   39.0  0.50    246.7      2.05   0.252
 2.00   39.0  0.75    256.1      2.29   0.252
 2.00   39.0  1.00    269.6      2.50   0.252
 2.00   42.0  0.50    213.2      2.35   0.229
 2.00   42.0  0.75    216.5      2.55   0.229
 2.00   42.0  1.00    235.6      2.74   0.228
 2.00   45.0  0.50    186.8      2.65   0.211
 2.00   45.0  0.75    191.1      2.82   0.211
 2.00   45.0  1.00    136.1      1.69   0.205
 2.00   48.0  0.50     97.2      1.48   0.197
 2.00   48.0  0.75     99.7      1.43   0.194
 2.00   48.0  1.00    114.9      1.58   0.190
 2.00   51.0  0.50     81.4      1.41   0.184
 2.00   51.0  0.75     85.9      1.36   0.184
 2.00   51.0  1.00     84.6      1.20   0.186
 2.00   54.0  0.50     70.1      1.40   0.180
 2.00   54.0  0.75     73.6      1.39   0.181
 2.00   54.0  1.00     61.9      1.18   0.183
 2.25    6.0  0.50    107.9      3.00   0.530
 2.25    6.0  0.75    129.8      2.99   0.537
 2.25    6.0  1.00    155.0      2.98   0.534
 2.25    9.0  0.50    188.0      2.79   0.547
 2.25    9.0  0.75    217.0      2.83   0.552
 2.25    9.0  1.00    244.2      2.99   0.556
 2.25   12.0  0.50    277.1      3.30   0.573
 2.25   12.0  0.75    307.3      3.48   0.583
 2.25   12.0  1.00    335.6      3.53   0.584
 2.25   15.0  0.50    342.9      3.22   0.586
 2.25   15.0  0.75    381.0      3.30   0.600
 2.25   15.0  1.00    398.4      3.12   0.593
 2.25   18.0  0.50    407.6      2.85   0.576
 2.25   18.0  0.75    439.2      2.70   0.587
 2.25   18.0  1.00    444.2      2.60   0.572
 2.25   21.0  0.50    451.1      2.45   0.543
 2.25   21.0  0.75    476.9      2.27   0.550
 2.25   21.0  1.00    471.7      2.23   0.535
 2.25   24.0  0.50    464.0      2.15   0.502
 2.25   24.0  0.75    482.6      2.02   0.507
 2.25   24.0  1.00    473.3      1.92   0.490
 2.25   27.0  0.50    465.6      1.90   0.455
 2.25   27.0  0.75    480.4      1.78   0.455
 2.25   27.0  1.00    464.5      1.67   0.440
 2.25   30.0  0.50    454.7      1.64   0.404
 2.25   30.0  0.75    459.9      1.48   0.402
 2.25   30.0  1.00    445.4      1.46   0.392
 2.25   33.0  0.50    436.6      1.40   0.360
 2.25   33.0  0.75    433.7      1.27   0.355
 2.25   33.0  1.00    422.2      1.37   0.352
 2.25   36.0  0.50    410.3      1.26   0.319
 2.25   36.0  0.75    409.2      1.20   0.317
 2.25   36.0  1.00    377.7      1.48   0.317
 2.25   39.0  0.50    381.3      1.28   0.286
 2.25   39.0  0.75    364.6      1.37   0.287
 2.25   39.0  1.00    326.6      1.90   0.288
 2.25   42.0  0.50    249.3      2.01   0.258
 2.25   42.0  0.75    256.7      2.22   0.258
 2.25   42.0  1.00    278.1      2.41   0.259
 2.25   45.0  0.50    210.7      2.36   0.238
 2.25   45.0  0.75    214.7      2.61   0.235
 2.25   45.0  1.00    233.9      2.86   0.233
 2.25   48.0  0.50    184.7      2.76   0.218
 2.25   48.0  0.75    193.2      2.86   0.221
 2.25   48.0  1.00    131.4      1.90   0.209
 2.25   51.0  0.50     97.0      1.77   0.206
 2.25   51.0  0.75    104.7      1.80   0.205
 2.25   51.0  1.00    109.0      1.74   0.197
 2.25   54.0  0.50     82.0      1.67   0.194
 2.25   54.0  0.75     85.4      1.66   0.192
 2.25   54.0  1.00     73.2      1.31   0.192
 2.50    6.0  0.50    107.0      3.25   0.608
 2.50    6.0  0.75    130.1      3.21   0.611
 2.50    6.0  1.00    161.5      3.18   0.612
 2.50    9.0  0.50    200.5      3.06   0.625
 2.50    9.0  0.75    219.0      3.08   0.632
 2.50    9.0  1.00    265.0      3.38   0.636
 2.50   12.0  0.50    298.1      3.59   0.655
 2.50   12.0  0.75    323.7      3.69   0.668
 2.50   12.0  1.00    348.2      3.73   0.668
 2.50   15.0  0.50    375.3      3.37   0.669
 2.50   15.0  0.75    410.3      3.33   0.682
 2.50   15.0  1.00    413.1      3.28   0.672
 2.50   18.0  0.50    446.6      2.94   0.655
 2.50   18.0  0.75    476.4      2.79   0.666
 2.50   18.0  1.00    473.5      2.74   0.650
 2.50   21.0  0.50    482.6      2.61   0.617
 2.50   21.0  0.75    509.2      2.53   0.626
 2.50   21.0  1.00    497.6      2.41   0.608
 2.50   24.0  0.50    510.8      2.54   0.568
 2.50   24.0  0.75    540.0      2.68   0.568
 2.50   24.0  1.00    513.0      2.30   0.556
 2.50   27.0  0.50    519.7      2.50   0.510
 2.50   27.0  0.75    541.3      2.60   0.511
 2.50   27.0  1.00    507.6      2.06   0.498
 2.50   30.0  0.50    509.9      2.26   0.454
 2.50   30.0  0.75    520.9      2.26   0.453
 2.50   30.0  1.00    484.7      1.69   0.443
 2.50   33.0  0.50    484.9      1.85   0.401
 2.50   33.0  0.75    483.3      1.63   0.404
 2.50   33.0  1.00    456.3      1.37   0.397
 2.50   36.0  0.50    447.1      1.35   0.356
 2.50   36.0  0.75    444.6      1.18   0.355
 2.50   36.0  1.00    431.1      1.24   0.361
 2.50   39.0  0.50    412.9      1.11   0.318
 2.50   39.0  0.75    406.7      1.06   0.317
 2.50   39.0  1.00    404.3      1.20   0.325
 2.50   42.0  0.50    386.8      1.16   0.290
 2.50   42.0  0.75    376.1      1.18   0.289
 2.50   42.0  1.00    340.3      1.59   0.290
 2.50   45.0  0.50    255.8      1.92   0.264
 2.50   45.0  0.75    298.6      1.84   0.263
 2.50   45.0  1.00    274.6      2.41   0.261
 2.50   48.0  0.50    213.1      2.37   0.245
 2.50   48.0  0.75    221.8      2.57   0.246
 2.50   48.0  1.00    225.6      2.99   0.235
 2.50   51.0  0.50    182.0      2.80   0.228
 2.50   51.0  0.75    193.9      2.95   0.229
 2.50   51.0  1.00    124.6      2.02   0.214
 2.50   54.0  0.50    163.7      3.19   0.214
 2.50   54.0  0.75    169.1      3.33   0.214
 2.50   54.0  1.00     97.7      1.78   0.203
 2.75    6.0  0.50    111.0      3.48   0.689
 2.75    6.0  0.75    125.9      3.46   0.694
 2.75    6.0  1.00    161.4      3.44   0.696
 2.75    9.0  0.50    220.9      3.39   0.710
 2.75    9.0  0.75    232.2      3.38   0.716
 2.75    9.0  1.00    255.6      3.56   0.722
 2.75   12.0  0.50    321.9      3.79   0.744
 2.75   12.0  0.75    343.8      3.91   0.757
 2.75   12.0  1.00    353.9      3.92   0.753
 2.75   15.0  0.50    411.7      3.46   0.756
 2.75   15.0  0.75    443.4      3.38   0.772
 2.75   15.0  1.00    439.1      3.36   0.759
 2.75   18.0  0.50    477.9      3.03   0.736
 2.75   18.0  0.75    505.8      2.95   0.749
 2.75   18.0  1.00    498.0      2.86   0.730
 2.75   21.0  0.50    525.0      3.03   0.693
 2.75   21.0  0.75    558.9      3.21   0.703
 2.75   21.0  1.00    538.8      2.88   0.684
 2.75   24.0  0.50    560.5      3.22   0.635
 2.75   24.0  0.75    594.4      3.56   0.643
 2.75   24.0  1.00    562.9      3.25   0.623
 2.75   27.0  0.50    602.6      3.57   0.570
 2.75   27.0  0.75    617.5      3.67   0.573
 2.75   27.0  1.00    576.0      3.20   0.559
 2.75   30.0  0.50    595.6      3.39   0.507
 2.75   30.0  0.75    606.9      3.50   0.509
 2.75   30.0  1.00    556.8      2.89   0.499
 2.75   33.0  0.50    579.8      3.20   0.448
 2.75   33.0  0.75    585.6      3.28   0.451
 2.75   33.0  1.00    516.4      2.19   0.444
 2.75   36.0  0.50    512.5      2.23   0.398
 2.75   36.0  0.75    507.3      2.02   0.400
 2.75   36.0  1.00    469.6      1.47   0.402
 2.75   39.0  0.50    456.2      1.44   0.354
 2.75   39.0  0.75    448.8      1.18   0.356
 2.75   39.0  1.00    437.9      1.12   0.363
 2.75   42.0  0.50    415.7      1.07   0.321
 2.75   42.0  0.75    407.0      0.95   0.317
 2.75   42.0  1.00    406.6      1.04   0.322
 2.75   45.0  0.50    384.4      1.07   0.293
 2.75   45.0  0.75    376.6      1.08   0.291
 2.75   45.0  1.00    357.1      1.29   0.290
 2.75   48.0  0.50    348.0      1.25   0.270
 2.75   48.0  0.75    335.6      1.39   0.274
 2.75   48.0  1.00    264.0      2.52   0.263
 2.75   51.0  0.50    216.8      2.34   0.253
 2.75   51.0  0.75    220.3      2.61   0.253
 2.75   51.0  1.00    210.5      3.11   0.237
 2.75   54.0  0.50    187.5      2.80   0.237
 2.75   54.0  0.75    192.4      3.06   0.237
 2.75   54.0  1.00    109.3      1.95   0.215
 3.00    6.0  0.50    107.0      3.72   0.779
 3.00    6.0  0.75    138.1      3.68   0.777
 3.00    6.0  1.00    153.9      3.72   0.783
 3.00    9.0  0.50    228.3      3.67   0.803
 3.00    9.0  0.75    250.6      3.74   0.807
 3.00    9.0  1.00    257.9      3.79   0.812
 3.00   12.0  0.50    343.2      3.97   0.840
 3.00   12.0  0.75    363.7      4.12   0.854
 3.00   12.0  1.00    368.0      4.12   0.849
 3.00   15.0  0.50    443.3      3.58   0.854
 3.00   15.0  0.75    473.5      3.51   0.866
 3.00   15.0  1.00    465.4      3.46   0.851
 3.00   18.0  0.50    508.6      3.27   0.823
 3.00   18.0  0.75    541.2      3.31   0.837
 3.00   18.0  1.00    524.5      3.17   0.821
 3.00   21.0  0.50    570.8      3.64   0.770
 3.00   21.0  0.75    608.3      3.95   0.784
 3.00   21.0  1.00    589.3      3.72   0.764
 3.00   24.0  0.50    622.3      3.92   0.704
 3.00   24.0  0.75    650.9      4.05   0.715
 3.00   24.0  1.00    631.9      4.06   0.695
 3.00   27.0  0.50    645.5      3.69   0.634
 3.00   27.0  0.75    664.9      3.78   0.642
 3.00   27.0  1.00    634.5      3.90   0.623
 3.00   30.0  0.50    648.1      3.42   0.560
 3.00   30.0  0.75    658.8      3.55   0.566
 3.00   30.0  1.00    624.6      3.72   0.552
 3.00   33.0  0.50    627.3      3.29   0.501
 3.00   33.0  0.75    633.1      3.42   0.498
 3.00   33.0  1.00    598.7      3.54   0.493
 3.00   36.0  0.50    606.4      3.17   0.445
 3.00   36.0  0.75    602.3      3.30   0.443
 3.00   36.0  1.00    544.3      2.70   0.443
 3.00   39.0  0.50    564.2      3.05   0.390
 3.00   39.0  0.75    524.7      2.42   0.392
 3.00   39.0  1.00    501.0      1.93   0.400
 3.00   42.0  0.50    468.7      1.59   0.349
 3.00   42.0  0.75    458.4      1.32   0.356
 3.00   42.0  1.00    444.1      1.08   0.359
 3.00   45.0  0.50    422.2      1.09   0.324
 3.00   45.0  0.75    414.5      0.95   0.326
 3.00   45.0  1.00    404.8      0.92   0.325
 3.00   48.0  0.50    387.5      0.94   0.298
 3.00   48.0  0.75    380.9      0.99   0.300
 3.00   48.0  1.00    330.3      1.50   0.292
 3.00   51.0  0.50    361.0      1.11   0.278
 3.00   51.0  0.75    341.1      1.27   0.278
 3.00   51.0  1.00    249.9      2.69   0.264
 3.00   54.0  0.50    220.4      2.24   0.262
 3.00   54.0  0.75    220.9      2.62   0.259
 3.00   54.0  1.00    200.2      3.22   0.238
```

</details>

<details>
<summary>z0 = 0.20 (207/459 close)</summary>

```
   vz     w0  tuck  rot_deg  land_m/s  apex_m
 1.00    6.0  0.50     78.4      2.31   0.345
 1.00    6.0  0.75     90.9      2.34   0.350
 1.00    6.0  1.00    124.6      2.24   0.345
 1.00    9.0  0.50    132.5      2.15   0.342
 1.00    9.0  0.75    150.9      2.10   0.347
 1.00    9.0  1.00    175.7      1.98   0.345
 1.00   12.0  0.50    173.5      1.98   0.345
 1.00   12.0  0.75    197.6      1.99   0.348
 1.00   12.0  1.00    218.4      2.12   0.346
 1.00   15.0  0.50    213.2      2.20   0.352
 1.00   15.0  0.75    245.9      2.45   0.357
 1.00   15.0  1.00    263.8      2.66   0.352
 1.00   18.0  0.50    264.1      2.73   0.354
 1.00   18.0  0.75    285.0      2.83   0.359
 1.00   18.0  1.00    296.5      2.94   0.351
 1.00   21.0  0.50    276.0      2.64   0.347
 1.00   21.0  0.75    295.0      2.78   0.351
 1.00   21.0  1.00    302.5      2.88   0.339
 1.00   24.0  0.50    280.5      2.55   0.332
 1.00   24.0  0.75    299.0      2.67   0.333
 1.00   24.0  1.00    299.9      2.83   0.321
 1.00   27.0  0.50    270.4      2.48   0.312
 1.00   27.0  0.75    284.2      2.64   0.313
 1.00   27.0  1.00    285.4      2.83   0.302
 1.00   30.0  0.50    254.3      2.49   0.291
 1.00   30.0  0.75    264.9      2.67   0.291
 1.00   30.0  1.00    265.7      2.88   0.282
 1.00   33.0  0.50    233.9      2.58   0.272
 1.00   33.0  0.75    242.6      2.79   0.271
 1.00   33.0  1.00    243.0      2.96   0.265
 1.00   36.0  0.50    215.0      2.76   0.261
 1.00   36.0  0.75    218.3      2.96   0.263
 1.00   36.0  1.00    185.3      2.50   0.261
 1.00   39.0  0.50    196.9      2.98   0.259
 1.00   39.0  0.75    154.9      2.51   0.261
 1.00   39.0  1.00    150.7      2.12   0.259
 1.00   42.0  0.50    118.2      2.18   0.257
 1.00   42.0  0.75    122.7      2.25   0.259
 1.00   42.0  1.00    128.6      2.08   0.257
 1.00   45.0  0.50    100.4      2.18   0.256
 1.00   45.0  0.75    106.4      2.24   0.258
 1.00   45.0  1.00    113.4      2.18   0.256
 1.00   48.0  0.50     88.1      2.20   0.255
 1.00   48.0  0.75     92.9      2.36   0.256
 1.00   48.0  1.00     94.3      2.16   0.255
 1.00   51.0  0.50     81.1      2.44   0.253
 1.00   51.0  0.75     84.2      2.49   0.256
 1.00   51.0  1.00     79.5      2.22   0.254
 1.00   54.0  0.50     72.8      2.59   0.253
 1.00   54.0  0.75     72.5      2.56   0.254
 1.00   54.0  1.00     66.9      2.31   0.254
 1.25    6.0  0.50     85.1      2.50   0.387
 1.25    6.0  0.75    100.1      2.51   0.394
 1.25    6.0  1.00    130.7      2.44   0.390
 1.25    9.0  0.50    147.2      2.26   0.388
 1.25    9.0  0.75    168.0      2.23   0.393
 1.25    9.0  1.00    191.1      2.18   0.391
 1.25   12.0  0.50    200.0      2.31   0.398
 1.25   12.0  0.75    228.1      2.44   0.402
 1.25   12.0  1.00    251.7      2.66   0.400
 1.25   15.0  0.50    254.5      2.75   0.408
 1.25   15.0  0.75    289.3      3.01   0.415
 1.25   15.0  1.00    303.7      3.11   0.409
 1.25   18.0  0.50    291.0      2.85   0.409
 1.25   18.0  0.75    317.4      2.96   0.417
 1.25   18.0  1.00    328.4      3.01   0.406
 1.25   21.0  0.50    308.8      2.66   0.398
 1.25   21.0  0.75    340.0      2.76   0.403
 1.25   21.0  1.00    339.2      2.81   0.390
 1.25   24.0  0.50    320.0      2.48   0.378
 1.25   24.0  0.75    349.6      2.52   0.381
 1.25   24.0  1.00    338.6      2.65   0.367
 1.25   27.0  0.50    313.3      2.35   0.353
 1.25   27.0  0.75    339.9      2.42   0.355
 1.25   27.0  1.00    325.0      2.63   0.342
 1.25   30.0  0.50    293.3      2.34   0.328
 1.25   30.0  0.75    318.8      2.46   0.329
 1.25   30.0  1.00    302.7      2.72   0.318
 1.25   33.0  0.50    266.2      2.43   0.303
 1.25   33.0  0.75    272.5      2.62   0.304
 1.25   33.0  1.00    277.3      2.86   0.296
 1.25   36.0  0.50    236.6      2.60   0.283
 1.25   36.0  0.75    244.1      2.82   0.281
 1.25   36.0  1.00    251.3      3.02   0.277
 1.25   39.0  0.50    212.2      2.85   0.268
 1.25   39.0  0.75    218.8      3.06   0.270
 1.25   39.0  1.00    225.0      3.12   0.268
 1.25   42.0  0.50    193.7      3.13   0.265
 1.25   42.0  0.75    194.6      3.26   0.267
 1.25   42.0  1.00    156.1      2.50   0.265
 1.25   45.0  0.50    125.3      2.71   0.263
 1.25   45.0  0.75    127.7      2.62   0.265
 1.25   45.0  1.00    127.4      2.36   0.264
 1.25   48.0  0.50    101.3      2.47   0.261
 1.25   48.0  0.75    103.4      2.44   0.263
 1.25   48.0  1.00    112.2      2.41   0.262
 1.25   51.0  0.50     87.9      2.45   0.260
 1.25   51.0  0.75     92.1      2.57   0.261
 1.25   51.0  1.00     98.3      2.46   0.261
 1.25   54.0  0.50     80.4      2.67   0.258
 1.25   54.0  0.75     80.3      2.63   0.260
 1.25   54.0  1.00     81.6      2.44   0.259
 1.50    6.0  0.50     92.5      2.71   0.439
 1.50    6.0  0.75    109.9      2.68   0.445
 1.50    6.0  1.00    140.5      2.64   0.441
 1.50    9.0  0.50    158.7      2.45   0.443
 1.50    9.0  0.75    180.8      2.43   0.449
 1.50    9.0  1.00    206.8      2.42   0.446
 1.50   12.0  0.50    222.6      2.62   0.458
 1.50   12.0  0.75    254.3      2.82   0.464
 1.50   12.0  1.00    278.2      3.04   0.462
 1.50   15.0  0.50    286.9      3.08   0.472
 1.50   15.0  0.75    319.5      3.19   0.479
 1.50   15.0  1.00    330.8      3.23   0.471
 1.50   18.0  0.50    322.4      2.91   0.470
 1.50   18.0  0.75    361.5      2.96   0.479
 1.50   18.0  1.00    363.3      2.96   0.468
 1.50   21.0  0.50    363.8      2.63   0.454
 1.50   21.0  0.75    385.1      2.58   0.462
 1.50   21.0  1.00    380.8      2.56   0.445
 1.50   24.0  0.50    388.7      2.28   0.430
 1.50   24.0  0.75    400.3      2.22   0.433
 1.50   24.0  1.00    386.6      2.28   0.417
 1.50   27.0  0.50    393.6      2.04   0.397
 1.50   27.0  0.75    395.9      2.00   0.400
 1.50   27.0  1.00    373.5      2.19   0.385
 1.50   30.0  0.50    384.0      1.89   0.366
 1.50   30.0  0.75    378.6      1.95   0.368
 1.50   30.0  1.00    344.9      2.30   0.356
 1.50   33.0  0.50    350.5      1.97   0.338
 1.50   33.0  0.75    342.1      2.14   0.339
 1.50   33.0  1.00    320.3      2.48   0.331
 1.50   36.0  0.50    275.2      2.34   0.313
 1.50   36.0  0.75    288.8      2.51   0.314
 1.50   36.0  1.00    285.5      2.79   0.308
 1.50   39.0  0.50    242.6      2.59   0.293
 1.50   39.0  0.75    244.8      2.82   0.292
 1.50   39.0  1.00    255.8      3.04   0.288
 1.50   42.0  0.50    215.8      2.89   0.276
 1.50   42.0  0.75    222.4      3.07   0.276
 1.50   42.0  1.00    228.6      3.27   0.275
 1.50   45.0  0.50    193.0      3.20   0.271
 1.50   45.0  0.75    197.4      3.39   0.273
 1.50   45.0  1.00    207.5      3.57   0.272
 1.50   48.0  0.50    175.8      3.53   0.269
 1.50   48.0  0.75    178.6      3.68   0.270
 1.50   48.0  1.00    133.8      2.70   0.270
 1.50   51.0  0.50    111.2      3.06   0.266
 1.50   51.0  0.75    107.4      2.82   0.268
 1.50   51.0  1.00    113.8      2.68   0.268
 1.50   54.0  0.50     88.9      2.79   0.265
 1.50   54.0  0.75     92.1      2.76   0.266
 1.50   54.0  1.00     98.3      2.68   0.266
 1.75    6.0  0.50    104.1      2.87   0.494
 1.75    6.0  0.75    118.3      2.87   0.503
 1.75    6.0  1.00    150.1      2.87   0.500
 1.75    9.0  0.50    172.4      2.64   0.506
 1.75    9.0  0.75    195.0      2.64   0.511
 1.75    9.0  1.00    225.5      2.72   0.509
 1.75   12.0  0.50    249.0      2.99   0.526
 1.75   12.0  0.75    284.4      3.25   0.533
 1.75   12.0  1.00    306.8      3.42   0.530
 1.75   15.0  0.50    313.2      3.22   0.541
 1.75   15.0  0.75    353.5      3.32   0.550
 1.75   15.0  1.00    360.7      3.30   0.540
 1.75   18.0  0.50    364.5      2.97   0.536
 1.75   18.0  0.75    402.6      2.83   0.545
 1.75   18.0  1.00    401.1      2.80   0.530
 1.75   21.0  0.50    406.0      2.58   0.514
 1.75   21.0  0.75    435.6      2.39   0.522
 1.75   21.0  1.00    427.6      2.32   0.504
 1.75   24.0  0.50    423.9      2.27   0.482
 1.75   24.0  0.75    448.7      2.08   0.486
 1.75   24.0  1.00    442.6      2.05   0.470
 1.75   27.0  0.50    430.0      2.02   0.446
 1.75   27.0  0.75    441.3      1.84   0.447
 1.75   27.0  1.00    432.6      1.84   0.433
 1.75   30.0  0.50    420.7      1.81   0.411
 1.75   30.0  0.75    427.0      1.72   0.411
 1.75   30.0  1.00    408.3      1.75   0.398
 1.75   33.0  0.50    406.8      1.71   0.377
 1.75   33.0  0.75    404.9      1.65   0.374
 1.75   33.0  1.00    376.1      1.84   0.367
 1.75   36.0  0.50    377.1      1.70   0.346
 1.75   36.0  0.75    365.1      1.80   0.344
 1.75   36.0  1.00    330.2      2.27   0.342
 1.75   39.0  0.50    289.2      2.19   0.320
 1.75   39.0  0.75    314.3      2.25   0.321
 1.75   39.0  1.00    292.6      2.66   0.319
 1.75   42.0  0.50    243.2      2.60   0.300
 1.75   42.0  0.75    249.4      2.81   0.301
 1.75   42.0  1.00    258.6      3.05   0.297
 1.75   45.0  0.50    216.8      2.90   0.286
 1.75   45.0  0.75    217.7      3.14   0.285
 1.75   45.0  1.00    230.0      3.39   0.283
 1.75   48.0  0.50    192.1      3.26   0.277
 1.75   48.0  0.75    199.3      3.46   0.278
 1.75   48.0  1.00    207.8      3.72   0.279
 1.75   51.0  0.50    176.4      3.61   0.274
 1.75   51.0  0.75    179.2      3.78   0.275
 1.75   51.0  1.00    136.5      3.07   0.277
 1.75   54.0  0.50    160.7      4.04   0.272
 1.75   54.0  0.75    158.9      4.14   0.273
 1.75   54.0  1.00    111.8      2.89   0.274
 2.00    6.0  0.50    111.2      3.07   0.558
 2.00    6.0  0.75    128.2      3.08   0.566
 2.00    6.0  1.00    157.1      3.02   0.563
 2.00    9.0  0.50    184.6      2.87   0.574
 2.00    9.0  0.75    212.0      2.91   0.580
 2.00    9.0  1.00    244.0      3.07   0.579
 2.00   12.0  0.50    276.5      3.41   0.599
 2.00   12.0  0.75    311.4      3.58   0.611
 2.00   12.0  1.00    328.3      3.63   0.605
 2.00   15.0  0.50    332.6      3.36   0.610
 2.00   15.0  0.75    376.6      3.40   0.623
 2.00   15.0  1.00    389.0      3.28   0.614
 2.00   18.0  0.50    394.2      3.01   0.602
 2.00   18.0  0.75    430.7      2.85   0.613
 2.00   18.0  1.00    444.0      2.70   0.600
 2.00   21.0  0.50    440.3      2.63   0.577
 2.00   21.0  0.75    464.5      2.44   0.584
 2.00   21.0  1.00    470.8      2.31   0.568
 2.00   24.0  0.50    455.6      2.30   0.539
 2.00   24.0  0.75    474.5      2.16   0.545
 2.00   24.0  1.00    472.3      2.05   0.527
 2.00   27.0  0.50    459.0      2.07   0.499
 2.00   27.0  0.75    469.4      1.94   0.500
 2.00   27.0  1.00    461.5      1.83   0.484
 2.00   30.0  0.50    452.3      1.87   0.458
 2.00   30.0  0.75    455.5      1.72   0.456
 2.00   30.0  1.00    445.2      1.65   0.442
 2.00   33.0  0.50    437.8      1.68   0.417
 2.00   33.0  0.75    436.8      1.55   0.414
 2.00   33.0  1.00    421.8      1.61   0.408
 2.00   36.0  0.50    412.2      1.54   0.381
 2.00   36.0  0.75    411.6      1.50   0.381
 2.00   36.0  1.00    388.0      1.69   0.377
 2.00   39.0  0.50    387.1      1.56   0.352
 2.00   39.0  0.75    382.1      1.59   0.352
 2.00   39.0  1.00    346.5      2.01   0.352
 2.00   42.0  0.50    353.1      1.74   0.329
 2.00   42.0  0.75    326.6      1.97   0.329
 2.00   42.0  1.00    306.0      2.47   0.328
 2.00   45.0  0.50    249.8      2.49   0.311
 2.00   45.0  0.75    255.2      2.71   0.311
 2.00   45.0  1.00    263.9      3.03   0.305
 2.00   48.0  0.50    219.7      2.89   0.297
 2.00   48.0  0.75    221.6      3.15   0.294
 2.00   48.0  1.00    235.3      3.45   0.290
 2.00   51.0  0.50    195.5      3.28   0.284
 2.00   51.0  0.75    200.4      3.52   0.284
 2.00   51.0  1.00    201.0      3.83   0.286
 2.00   54.0  0.50    175.2      3.71   0.280
 2.00   54.0  0.75    179.9      3.90   0.281
 2.00   54.0  1.00    119.0      2.97   0.283
 2.25    6.0  0.50    114.3      3.29   0.630
 2.25    6.0  0.75    137.5      3.29   0.637
 2.25    6.0  1.00    162.2      3.23   0.634
 2.25    9.0  0.50    199.4      3.14   0.647
 2.25    9.0  0.75    230.7      3.21   0.652
 2.25    9.0  1.00    260.2      3.42   0.656
 2.25   12.0  0.50    294.4      3.68   0.673
 2.25   12.0  0.75    321.5      3.75   0.683
 2.25   12.0  1.00    352.4      3.77   0.684
 2.25   15.0  0.50    366.0      3.46   0.686
 2.25   15.0  0.75    400.2      3.44   0.700
 2.25   15.0  1.00    425.0      3.26   0.693
 2.25   18.0  0.50    433.5      3.05   0.676
 2.25   18.0  0.75    464.9      2.90   0.687
 2.25   18.0  1.00    472.9      2.81   0.672
 2.25   21.0  0.50    472.3      2.69   0.643
 2.25   21.0  0.75    501.4      2.59   0.650
 2.25   21.0  1.00    494.6      2.49   0.635
 2.25   24.0  0.50    491.8      2.53   0.602
 2.25   24.0  0.75    516.1      2.55   0.607
 2.25   24.0  1.00    502.8      2.33   0.590
 2.25   27.0  0.50    505.3      2.53   0.555
 2.25   27.0  0.75    526.9      2.59   0.555
 2.25   27.0  1.00    498.9      2.16   0.540
 2.25   30.0  0.50    501.4      2.37   0.504
 2.25   30.0  0.75    510.9      2.31   0.502
 2.25   30.0  1.00    479.9      1.88   0.492
 2.25   33.0  0.50    485.4      2.10   0.460
 2.25   33.0  0.75    478.3      1.83   0.455
 2.25   33.0  1.00    456.6      1.63   0.452
 2.25   36.0  0.50    448.9      1.65   0.419
 2.25   36.0  0.75    445.4      1.48   0.417
 2.25   36.0  1.00    430.4      1.51   0.417
 2.25   39.0  0.50    419.9      1.47   0.386
 2.25   39.0  0.75    416.0      1.38   0.387
 2.25   39.0  1.00    409.6      1.49   0.388
 2.25   42.0  0.50    390.6      1.47   0.358
 2.25   42.0  0.75    386.9      1.44   0.358
 2.25   42.0  1.00    370.0      1.65   0.359
 2.25   45.0  0.50    369.9      1.60   0.338
 2.25   45.0  0.75    347.4      1.72   0.335
 2.25   45.0  1.00    310.7      2.31   0.333
 2.25   48.0  0.50    257.4      2.41   0.318
 2.25   48.0  0.75    295.5      2.30   0.321
 2.25   48.0  1.00    256.4      3.14   0.309
 2.25   51.0  0.50    223.3      2.86   0.306
 2.25   51.0  0.75    225.3      3.13   0.305
 2.25   51.0  1.00    226.8      3.59   0.297
 2.25   54.0  0.50    199.1      3.31   0.294
 2.25   54.0  0.75    198.5      3.58   0.292
 2.25   54.0  1.00    188.5      3.98   0.292
 2.50    6.0  0.50    112.5      3.52   0.708
 2.50    6.0  0.75    136.8      3.50   0.711
 2.50    6.0  1.00    168.5      3.44   0.712
 2.50    9.0  0.50    211.7      3.41   0.725
 2.50    9.0  0.75    230.9      3.44   0.732
 2.50    9.0  1.00    279.1      3.76   0.736
 2.50   12.0  0.50    311.1      3.86   0.755
 2.50   12.0  0.75    338.5      3.97   0.768
 2.50   12.0  1.00    364.3      3.96   0.768
 2.50   15.0  0.50    395.4      3.57   0.769
 2.50   15.0  0.75    432.0      3.48   0.782
 2.50   15.0  1.00    434.0      3.42   0.772
 2.50   18.0  0.50    464.8      3.14   0.755
 2.50   18.0  0.75    494.1      3.00   0.766
 2.50   18.0  1.00    491.5      2.94   0.750
 2.50   21.0  0.50    505.4      2.96   0.717
 2.50   21.0  0.75    538.8      3.06   0.726
 2.50   21.0  1.00    523.3      2.84   0.708
 2.50   24.0  0.50    542.8      3.13   0.668
 2.50   24.0  0.75    576.0      3.40   0.668
 2.50   24.0  1.00    554.4      3.13   0.656
 2.50   27.0  0.50    559.7      3.24   0.610
 2.50   27.0  0.75    602.1      3.72   0.611
 2.50   27.0  1.00    557.8      3.10   0.598
 2.50   30.0  0.50    581.0      3.49   0.554
 2.50   30.0  0.75    591.9      3.59   0.553
 2.50   30.0  1.00    541.1      2.84   0.543
 2.50   33.0  0.50    549.7      3.05   0.501
 2.50   33.0  0.75    546.5      2.90   0.504
 2.50   33.0  1.00    506.0      2.22   0.497
 2.50   36.0  0.50    503.3      2.34   0.456
 2.50   36.0  0.75    504.8      2.24   0.455
 2.50   36.0  1.00    469.7      1.69   0.461
 2.50   39.0  0.50    471.3      1.94   0.418
 2.50   39.0  0.75    455.5      1.56   0.417
 2.50   39.0  1.00    442.1      1.45   0.425
 2.50   42.0  0.50    428.4      1.47   0.390
 2.50   42.0  0.75    417.5      1.31   0.389
 2.50   42.0  1.00    414.2      1.36   0.390
 2.50   45.0  0.50    394.5      1.40   0.364
 2.50   45.0  0.75    388.2      1.36   0.363
 2.50   45.0  1.00    379.1      1.47   0.361
 2.50   48.0  0.50    371.6      1.51   0.345
 2.50   48.0  0.75    366.5      1.54   0.346
 2.50   48.0  1.00    301.8      2.41   0.335
 2.50   51.0  0.50    338.7      1.73   0.328
 2.50   51.0  0.75    315.3      2.00   0.329
 2.50   51.0  1.00    245.5      3.27   0.314
 2.50   54.0  0.50    229.4      2.79   0.314
 2.50   54.0  0.75    235.2      3.04   0.314
 2.50   54.0  1.00    211.5      3.77   0.303
 2.75    6.0  0.50    115.9      3.73   0.789
 2.75    6.0  0.75    131.5      3.73   0.794
 2.75    6.0  1.00    167.3      3.67   0.796
 2.75    9.0  0.50    230.8      3.70   0.810
 2.75    9.0  0.75    243.0      3.71   0.816
 2.75    9.0  1.00    267.4      3.90   0.822
 2.75   12.0  0.50    334.0      4.01   0.844
 2.75   12.0  0.75    358.5      4.16   0.857
 2.75   12.0  1.00    367.9      4.14   0.853
 2.75   15.0  0.50    430.8      3.65   0.856
 2.75   15.0  0.75    462.2      3.56   0.872
 2.75   15.0  1.00    458.1      3.52   0.859
 2.75   18.0  0.50    496.2      3.29   0.836
 2.75   18.0  0.75    525.1      3.26   0.849
 2.75   18.0  1.00    517.2      3.16   0.830
 2.75   21.0  0.50    549.5      3.50   0.793
 2.75   21.0  0.75    586.7      3.77   0.803
 2.75   21.0  1.00    573.9      3.61   0.784
 2.75   24.0  0.50    608.0      4.01   0.735
 2.75   24.0  0.75    631.8      4.12   0.743
 2.75   24.0  1.00    597.6      3.96   0.723
 2.75   27.0  0.50    629.6      3.84   0.670
 2.75   27.0  0.75    647.7      3.94   0.673
 2.75   27.0  1.00    626.1      4.01   0.659
 2.75   30.0  0.50    627.6      3.65   0.607
 2.75   30.0  0.75    639.9      3.76   0.609
 2.75   30.0  1.00    612.9      3.84   0.599
 2.75   33.0  0.50    615.0      3.49   0.548
 2.75   33.0  0.75    621.4      3.62   0.551
 2.75   33.0  1.00    589.0      3.65   0.544
 2.75   36.0  0.50    595.4      3.38   0.498
 2.75   36.0  0.75    593.5      3.50   0.500
 2.75   36.0  1.00    531.8      2.80   0.502
 2.75   39.0  0.50    560.6      3.28   0.454
 2.75   39.0  0.75    525.2      2.70   0.456
 2.75   39.0  1.00    501.6      2.31   0.463
 2.75   42.0  0.50    477.4      2.07   0.421
 2.75   42.0  0.75    469.9      1.83   0.417
 2.75   42.0  1.00    448.3      1.43   0.422
 2.75   45.0  0.50    432.1      1.53   0.393
 2.75   45.0  0.75    420.7      1.33   0.391
 2.75   45.0  1.00    412.7      1.26   0.390
 2.75   48.0  0.50    396.6      1.33   0.370
 2.75   48.0  0.75    393.2      1.29   0.374
 2.75   48.0  1.00    372.3      1.50   0.363
 2.75   51.0  0.50    375.5      1.44   0.353
 2.75   51.0  0.75    366.5      1.47   0.353
 2.75   51.0  1.00    287.2      2.62   0.337
 2.75   54.0  0.50    353.0      1.61   0.337
 2.75   54.0  0.75    330.6      1.78   0.337
 2.75   54.0  1.00    229.7      3.52   0.315
 3.00    6.0  0.50    111.4      3.97   0.879
 3.00    6.0  0.75    143.3      3.93   0.877
 3.00    6.0  1.00    159.2      3.95   0.883
 3.00    9.0  0.50    237.1      3.95   0.903
 3.00    9.0  0.75    260.7      4.05   0.907
 3.00    9.0  1.00    268.3      4.11   0.912
 3.00   12.0  0.50    356.2      4.19   0.940
 3.00   12.0  0.75    377.1      4.33   0.954
 3.00   12.0  1.00    380.5      4.31   0.949
 3.00   15.0  0.50    460.5      3.78   0.953
 3.00   15.0  0.75    488.2      3.69   0.966
 3.00   15.0  1.00    484.1      3.68   0.951
 3.00   18.0  0.50    528.2      3.62   0.923
 3.00   18.0  0.75    565.2      3.78   0.937
 3.00   18.0  1.00    544.5      3.56   0.921
 3.00   21.0  0.50    604.0      4.23   0.870
 3.00   21.0  0.75    637.8      4.42   0.884
 3.00   21.0  1.00    616.8      4.25   0.864
 3.00   24.0  0.50    646.3      4.14   0.804
 3.00   24.0  0.75    678.0      4.21   0.815
 3.00   24.0  1.00    658.7      4.33   0.795
 3.00   27.0  0.50    679.3      3.78   0.734
 3.00   27.0  0.75    704.1      3.77   0.742
 3.00   27.0  1.00    667.4      4.05   0.723
 3.00   30.0  0.50    698.7      3.35   0.660
 3.00   30.0  0.75    707.0      3.42   0.666
 3.00   30.0  1.00    660.8      3.85   0.652
 3.00   33.0  0.50    678.3      3.21   0.601
 3.00   33.0  0.75    691.0      3.26   0.598
 3.00   33.0  1.00    639.6      3.78   0.593
 3.00   36.0  0.50    656.4      3.13   0.545
 3.00   36.0  0.75    655.9      3.33   0.543
 3.00   36.0  1.00    613.9      3.74   0.543
 3.00   39.0  0.50    605.7      3.25   0.490
 3.00   39.0  0.75    604.6      3.46   0.492
 3.00   39.0  1.00    586.8      3.61   0.500
 3.00   42.0  0.50    573.7      3.28   0.449
 3.00   42.0  0.75    566.5      3.39   0.456
 3.00   42.0  1.00    514.1      2.50   0.459
 3.00   45.0  0.50    541.3      3.26   0.424
 3.00   45.0  0.75    482.2      2.13   0.426
 3.00   45.0  1.00    458.0      1.69   0.425
 3.00   48.0  0.50    451.1      1.80   0.398
 3.00   48.0  0.75    430.3      1.43   0.400
 3.00   48.0  1.00    398.5      1.30   0.392
 3.00   51.0  0.50    403.2      1.34   0.378
 3.00   51.0  0.75    394.2      1.28   0.378
 3.00   51.0  1.00    351.6      1.64   0.364
 3.00   54.0  0.50    379.2      1.38   0.362
 3.00   54.0  0.75    368.3      1.40   0.359
 3.00   54.0  1.00    270.2      2.91   0.338
```

</details>

## Conclusion
**Yes, there is a launch setting that completes 360 deg backward at a
materially lower landing speed than the original 3.21 m/s -- but the honest
answer required catching and discarding a much larger, false improvement
first.**

The naive sweep, taken at face value, suggested landing speeds as low as
0.92-1.26 m/s (roughly a 70% reduction) at `w0` in the high 30s to mid 40s.
**That result is wrong.** A direction check on the new best cell -- required
by this task's own instructions, and by the same convention the original
document used to catch an earlier direction bug in `backflip_plate_kinematics`
-- shows those cells are FORWARD rolls (face-down), not backward flips. A
boundary scan traces this to a continuous loss of single-axis "clean
backward pitch" character as `w0` rises past roughly 33-36 rad/s, culminating
in an outright sign flip by `w0~40`. Every top-of-list cell independently
checked in that region (multiple `vz`, multiple `tuck`) failed the same way.
This is exactly the "implausibly low apex" trap the brief warned about: apex
at those cells (0.25-0.39 m) is well below the original best cell's 0.56 m,
and the number is untrustworthy precisely because it looked too good.

Restricted to the region where direction is verified (`w0<=~30`, with margin
before the ~33-36 boundary where the signal degrades), the genuine picture
is:

- **The best single cell**: `vz=2.25, w0=30.0, tuck=1.0` -> lands at
  **1.46-1.88 m/s** depending on `z0` (was 3.14-3.28 m/s in the original
  document's best cell at the matching `z0` range) -- a ~45-55% reduction,
  confirmed backward by direct measurement.
- **A robust DR box, not a knife edge**: `vz in [2.00, 2.25] m/s`, `w0 in
  [24.0, 30.0] rad/s`, `tuck in [0.5, 1.0]` closes 360 deg at every combination
  tested across all three `z0 in {0.10, 0.15, 0.20}`, with landing speed
  `1.46-2.59 m/s` throughout -- every cell in the box beats the original
  document's worst closing-cell landing speed (4.3 m/s) and the large
  majority beat its best (3.14 m/s).
- **Do not widen the box on either axis without re-checking**: `vz=1.75`
  fails to close at `z0=0.10, w0=30, tuck=1.0` (336.3 deg); `vz=2.5` closes
  everywhere but its landing speed at `z0=0.20` (up to 3.72 m/s) is worse
  than the original best cell, undoing the whole point of the extension.
  `w0` above ~33-36 is excluded on direction grounds, independent of its
  landing-speed numbers.

**Recommended DR ranges (supersedes the previous document's `w0 in [12,15],
vz in [2.0,3.0]` recommendation)**:

- `vz ~ [2.0, 2.25]` m/s (narrower than the original `[2.0, 3.0]` -- `2.5`+
  no longer buys a softer landing once `w0` is also raised, and only adds
  risk).
- `w0 ~ [24, 30]` rad/s (roughly double the original `[12, 15]` ceiling --
  this is the real, direction-verified finding of this extension: a faster
  flick genuinely trades into a softer landing, but only up to the point
  where the flip stops being single-axis-backward, well before the
  mathematical landing-speed minimum).
- `tuck = [0.5, 1.0]` (unchanged; `1.0` still gives the best single number
  but the whole range closes with margin).
- `t_launch=0.12s` still held fixed, not varied here either.

**On the physically-optimal-looking `w0` 39-48 region**: it produces a
different maneuver (something closer to a forward roll, possibly a
tumble with substantial off-axis coupling, or a numerical artifact of the
plate outrunning what the timestep can resolve at ~2 plate-rotations within
a 0.12 s launch window) -- not a validated cheaper backward flip. It should
not be used for training or hardware without first understanding, by
someone reviewing `backflip_plate_kinematics` and the contact dynamics
directly (out of scope here, and explicitly off-limits per this task's
constraints), whether that region is real robot behavior worth its own
investigation, or purely a probe/kinematics artifact. Flagging it here as a
finding, not fixing it.

This is still a real hardware-acceptability call for the user, not a
simulation call, and it is NOT uniformly resolved by this extension: the
recommended DR box's free-fall-equivalent impact height
(`h = v^2 / 2g`) ranges from **~11 cm at its best corner** (`1.46 m/s`) to
**~34 cm at its worst corner** (`2.59 m/s`, `z0=0.20`) -- the best cell is
comfortably under the reported ~30 cm damage threshold, but the box's own
worst corner sits slightly above it. The single best cell (1.46-1.88 m/s
depending on `z0`, i.e. ~11-18 cm-equivalent) is clearly under the
threshold; a DR range that also covers the box's `w0=24, tuck=0.5` and
`z0=0.20` corner is not. The original best cell (3.21 m/s, ~52 cm-equivalent)
was unambiguously over threshold at every corner -- that part of the
question is answered cleanly: this extension does buy real margin, just not
a margin-free landing.

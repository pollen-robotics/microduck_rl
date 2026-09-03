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

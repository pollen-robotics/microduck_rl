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

> **SUPERSEDED FOR ENV PURPOSES (posture notice, added by the final-review fix
> wave).** Every table in this document above the section
> "**Standing-spawn re-measurement (final-review fix wave)**" was measured
> from a **TUCKED** spawn: the robot pre-squatted in the tuck pose with its
> trunk ~2 cm above the plate top (CoM ~3 cm up), holding that ctrl from t=0.
> **The env never produces that posture.**
> `reset_backflip_robot_on_plate` spawns the robot **STANDING** at
> `z0 + PLATE_HALF_THICKNESS + STAND_Z` in the HOME pose (CoM ~11 cm up), and
> the `ready_stance` reward pays it to still be standing when the flick
> arrives. Re-measured from the standing spawn, **no launch setting closes
> 360° below the ~2.6 m/s hardware landing limit**, and the DR box this
> document recommended (`vz ∈ [2.00, 2.25]`, `w0 ∈ [24, 30]`) rotates the
> standing robot **FORWARD** (face-down, orientation-verified). The recommended
> box below is therefore valid *only as a statement about a tucked launch
> posture*; do not read it as the env's envelope. See the new section for the
> full re-measurement.
>
> **RESOLVED 2026-09-07:** the user's decision was to switch the env's HOLD
> posture to TUCKED, so the spawn, the hold reward and the envelope now all
> describe one posture. The env's CURRENT, whole-box-verified box is in the
> last section, "**Tucked hold**": `z0 in [0.10, 0.20]`, `vz in [2.00, 2.10]`,
> `w0 in [21, 23]`, `t_launch in [0.12, 0.14]`. Neither the tucked-probe box
> at the top of this document nor the standing re-measurement is the env's
> box; both are history. The tucked box was corrected once more after a
> re-review: the `z0` curriculum tail is **0.21**, not 0.225 (whole-box margin
> 0.07 m/s vs 0.01), and `backflip_ready_stance` regained an upright factor
> after the flop audit found the side-lying tuck outscoring the upright one.
> See "**Flop audit, and two corrections**".

> **CURRENT BOX — the final section, "Gentler ejection":**
> STANDING hold on a plate RESTING ON THE GROUND, `z0` in [0.01, 0.03],
> `vz` in [2.20, 2.80], `w0` in [18, 24], `t_launch` in [0.14, 0.16], hold
> 1-5 s uniform (no curriculum), episode 7.5 s, landing annuity over a fixed
> 1.4 s window. Direction-verified backward in every cell; rotation
> 98-442 deg; landing 1.78-3.65 m/s; apex 0.28-0.64 m; 27% close 360 deg
> open-loop.

> **Superseded box — "After the first training run":**
> STANDING hold on a plate RESTING ON THE GROUND, `z0` in [0.01, 0.03],
> `vz` in [3.00, 4.00], `w0` in [15, 24], `t_launch` in [0.12, 0.16], hold
> 1-5 s (curriculum-ramped), episode 7.5 s. Direction-verified backward in
> every cell; 62% close 360 deg open-loop; landing 2.4-4.9 m/s.

> **Superseded box — "v0 — a plausible human throw":**
> STANDING hold, `z0` in [0.07, 0.09], `vz` in [2.50, 3.50], `w0` in
> [15, 24], `t_launch` in [0.12, 0.16]. Direction-verified backward in every
> cell; 37% close 360 deg open-loop, which is information about starting
> difficulty rather than a gate; landing 2.4-4.4 m/s. Whole-box open-loop
> closure is NO LONGER an acceptance criterion — see that section for why.

> **Superseded box — "Back to standing" (2026-09-07):**
> STANDING hold, `z0` in [0.07, 0.09], `vz` in [2.80, 2.90], `w0` = 18.5,
> `t_launch` in [0.155, 0.16]. Rotation 363-458 deg, landing 3.1-3.9 m/s,
> whole-box closure, 33/33 direction checks backward, spawn geometrically
> valid. Landing is ABOVE the operator's ~2.6 m/s comfort threshold and that is
> knowingly accepted; the probe measures the launch, not the skill.

> **!! EVERY OTHER BOX IN THIS DOCUMENT IS INVALID (2026-09-07).** Every launch
> envelope here — including the one below and the original Task 3 tables — was
> measured with the robot's FEET TUNNELLED UNDER the launcher plate slab. The
> geometrically valid rest does not close a backflip under the hardware landing
> limit in any posture measured. Read the FINAL section, "**The feet are under
> the plate**", before using any number in this file.

> **SUPERSEDED BOX (2026-09-07, after the user watched the env): `z0` in
> [0.07, 0.09], `vz` in [1.90, 2.00], `w0` in [23, 24], `t_launch` in
> [0.12, 0.13]** — retuned for ONE clean turn from the lowest plate the hold
> pose allows: rotation 372-457 deg, apex 0.29-0.40 m, worst landing 2.15 m/s.
> Every box above it is history. The last section, "**Lower and gentler**",
> also records why the plate CANNOT lie on the ground with this hold pose, and
> the real bug behind "the plate is not under the feet".

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
   > **Superseded (Task 6 review):** the *root cause* was fixed later — the
   > plate no longer parks below the floor at all. `BACKFLIP_GONE_Z = -3.0`
   > was not "out of the way" but 3 m of PENETRATION into the infinite plane:
   > 4 spurious contacts per env every step, and the solver ejecting the 50 kg
   > plate at 59 m/s between step-event writes. It is replaced by
   > `BACKFLIP_GONE_POS = (5.0, 5.0, 5.0)` — above and beside the floor, where
   > it touches nothing. The probe's `plate_geom` filter is kept as
   > belt-and-braces; none of the measurements in this document are affected
   > (the filter made the probe correct under the old parking).
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
(comma-separated GRIDS, defaults below, drive the main sweep) and
`--check-vz`/`--check-w0`/`--check-tuck` (single numbers, only apply under
`--check-direction`, trace one cell). `--z0` is unchanged. A later revision
(review round) also added: a `*` marker on any printed row with
`w0 > SAFE_W0_CEILING` (30.0 rad/s) plus a warning banner above the table
when the swept `w0` grid includes any such rows, so sorting printed output
by `land_m/s` cannot silently surface an unverified (or known-wrong) cell;
and a docstring note distinguishing the grid flags from the single-cell
check flags, since typing `--w0` when `--check-w0` was meant silently
sweeps the whole default grid instead of checking one cell.
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

# Review-round additions: direction checks at the recommended box's corners
# (see "Direction verification at the box corners") and a finer vz scan at
# its worst-margin corner (see "The vz floor: cliff or margin?") -- both
# reproduced verbatim in their own sections below.
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

The same pattern repeats at `vz=2.0, tuck=1.0`, with intermediate points
sampled (not just the two endpoints), so this is a second full curve, not an
interpolation between two checks:

| w0   | local +z world x-component | verdict |
|------|------------------------------|---------|
| 24.0 | -0.533                        | backward |
| 27.0 | -0.422                        | backward |
| 30.0 | -0.280                        | backward |
| 33.0 | -0.182                        | backward, marginal |
| 36.0 | -0.043                         | ambiguous (rotation also stops closing 360 here, 346.0 deg) |

Every other "top of the naive list" cell checked directly (`vz=2.5,
w0=39/42, tuck=0.75`; `vz=2.75, w0=42/45, tuck=1.0`) also comes back
FACE-DOWN / WRONG. **None of the sub-`2.0 m/s` landing speeds from `w0`
above roughly 36-39 are trustworthy backward flips** -- they are excluded
from the recommendation below.

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
knife edge for CLOSURE and LANDING SPEED. (Direction is a separate claim,
checked separately below -- see "Direction verification at the box
corners".) **One caveat found while building this box**: widening it to
include `vz=1.75` breaks at `z0=0.10` -- `vz=1.75, w0=30.0, tuck=1.00,
z0=0.10` gives only `336.3 deg` (does not close). That corner (low vz, high
w0, high tuck, shallow z0) is excluded from the recommended range below for
exactly this reason -- see "The vz floor" below for how close `vz=2.00`
itself sits to that same cliff.

`vz=2.5` was also checked at `w0 in [24,30]` and rejected: it still closes
360 everywhere, but landing speed at `z0=0.20` reaches `3.10-3.72 m/s` --
worse than the original best cell, not better -- so `vz` should not be
pushed past `2.25` in the recommended range.

### Direction verification at the box corners

**Gap this closes**: the sections above establish CLOSURE and LANDING SPEED
robustness across the whole box (all 54 rows, all three `z0`) by direct
measurement. That is a different claim from DIRECTION robustness. Before
this check, DIRECTION had been confirmed at exactly one point inside the
box (`vz=2.25, w0=30.0, tuck=1.0, z0=0.10`, in "The genuine improvement"
above) plus the general `w0<=30` boundary-scan trend at `tuck=1.0` only
(two `vz` cuts, "The reversal" above) -- `tuck=0.5` had zero direct
direction checks anywhere, and `z0=0.20` had zero direct direction checks
anywhere, despite both being inside the recommended range. That gap is
closed here with checks at both `tuck` extremes crossed with both `z0`
extremes at `w0=30` (the box's `w0` ceiling, where direction margin is
thinnest), plus the box's single worst-landing-speed point
(`vz=2.25, w0=27.0, tuck=0.75, z0=0.20`, `2.59 m/s`):

```
uv run python scripts/backflip_envelope.py --check-direction --z0 0.10 \
  --check-vz 2.0 --check-w0 30.0 --check-tuck 0.5
uv run python scripts/backflip_envelope.py --check-direction --z0 0.20 \
  --check-vz 2.0 --check-w0 30.0 --check-tuck 0.5
uv run python scripts/backflip_envelope.py --check-direction --z0 0.10 \
  --check-vz 2.0 --check-w0 30.0 --check-tuck 1.0
uv run python scripts/backflip_envelope.py --check-direction --z0 0.20 \
  --check-vz 2.0 --check-w0 30.0 --check-tuck 1.0
uv run python scripts/backflip_envelope.py --check-direction --z0 0.20 \
  --check-vz 2.25 --check-w0 27.0 --check-tuck 0.75
```

| vz   | w0   | tuck | z0   | local +z world x-component | verdict | rot_deg | land_m/s |
|------|------|------|------|-------------------------------|-----------|---------|----------|
| 2.00 | 30.0 | 0.5  | 0.10 | -0.292                         | backward  | 420.4   | 1.58     |
| 2.00 | 30.0 | 0.5  | 0.20 | -0.292                         | backward  | 452.3   | 1.87     |
| 2.00 | 30.0 | 1.0  | 0.10 | -0.280                         | backward  | 399.5   | 1.54     |
| 2.00 | 30.0 | 1.0  | 0.20 | -0.280                         | backward  | 445.2   | 1.65     |
| 2.25 | 27.0 | 0.75 | 0.20 | -0.510                         | backward  | 526.9   | 2.59     |

All five come back backward (negative world-x component, same convention
as every other check in this document). Note the `tuck=0.5` and `tuck=1.0`
x-components at `w0=30` are identical to three decimal places across both
`z0` values -- the direction check samples the flight trajectory at a fixed
*accumulated-rotation* instant (~90 deg), not a fixed time, so at matched
`vz/w0/tuck` the early-flight orientation is close to `z0`-independent (the
plate's prescribed kinematics before liftoff are the same shape regardless
of `z0`; `z0` mainly changes what happens after liftoff, i.e. apex and
landing). This is expected, not a bug in the check.

With this, every corner of the recommended box that could plausibly hide a
reversal (`w0` at its ceiling, both `tuck` extremes, both `z0` extremes, and
the single worst-landing point) has been directly checked and confirmed
backward. The interior of the box was not exhaustively checked cell-by-cell
-- that inference rests on the boundary-scan trend (monotonic movement away
from zero as `w0` decreases from the ~33-39 reversal zone, confirmed on two
independent `vz` cuts in "The reversal") plus these five corner points, not
on a claim that all 54 rows were individually traced.

### The vz floor: cliff or margin?

`vz=1.75` fails closure (336.3 deg) while `vz=2.00` passes (399.5 deg) at
the box's worst-margin corner (`w0=30, tuck=1.0, z0=0.10`) -- a single
`0.25 m/s` step spanning "clearly fails" to "passes" with nothing sampled
between them, and the eventual DR sampler will draw `vz` continuously from
the recommended range, so it will land on every value in between. Sampled
finer at that same corner:

```
uv run python scripts/backflip_envelope.py --z0 0.10 \
  --vz 1.75,1.80,1.85,1.90,1.95,2.00 --w0 30 --tuck 1.0
```

```
   vz     w0  tuck  rot_deg  land_m/s  apex_m
 1.75   30.0  1.00    336.3      2.09   0.298
 1.80   30.0  1.00    346.8      1.98   0.307
 1.85   30.0  1.00    360.8      1.84   0.314
 1.90   30.0  1.00    378.8      1.69   0.323
 1.95   30.0  1.00    389.7      1.59   0.331
 2.00   30.0  1.00    399.5      1.54   0.342
```

Closure crosses 360 deg between `vz=1.80` (fails, 346.8 deg) and `vz=1.85`
(passes, but only barely -- 360.8 deg, 0.8 deg of margin). It does not
recover gradually across a wide band; it is a real, fairly sharp threshold
around `vz~1.82-1.84`, and `vz=1.85` itself is too close to that threshold
to trust. **The box's stated floor of `vz=2.00` sits about 0.15-0.20 m/s
above the measured cliff, with 399.5 deg of margin (39.5 deg clear of the
360 deg line) rather than the ~1 deg margin at `vz=1.85`** -- so the floor
does NOT need to move up. It is, however, closer to the cliff than a first
glance at the box (`vz in [2.00, 2.25]`) suggests, and this is why: at
`tuck=0.5/0.75` the same `vz=1.75` point already passes comfortably (365.6
and 362.6 deg respectively, both `z0=0.10, w0=30`), so `tuck=1.0` is
specifically the worst-case tuck for this cliff, consistent with full tuck
being the most demanding pose for maintaining plate contact through the
whole `t_launch` window. Do not treat `vz=2.00` as having a large buffer at
other `(w0, tuck)` combinations without checking; this measurement is
specific to the box's `w0=30, tuck=1.0` corner, its worst case.

## Full raw sweep output (verbatim, all 459 cells per z0)
Collapsed for length -- reproduced with the commands in "Commands run"
above. `rot_deg` is backward rotation for `w0<=~33`; treat `rot_deg` for
`w0>36` as an ungrounded number until independently direction-checked (see
"The reversal" above). These tables predate the script's `*` marker column
(added after this sweep, per "Minor items" below); a fresh run of the same
commands now marks every `w0 > 30.0` row with a trailing `*` and prints a
warning banner above the table for exactly this reason -- don't sort these
rows by `land_m/s` and trust the top one without checking it.

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

Restricted to `w0<=~30` (with margin before the ~33-36 boundary where the
general direction signal degrades, per the two boundary-scan curves in "The
reversal"), the genuine picture is:

- **The best single cell**: `vz=2.25, w0=30.0, tuck=1.0` -> lands at
  **1.46-1.88 m/s** depending on `z0` (was 3.14-3.28 m/s in the original
  document's best cell at the matching `z0` range) -- a ~45-55% reduction,
  confirmed backward by direct measurement.
- **A robust DR box, not a knife edge, for closure AND direction**: `vz in
  [2.00, 2.25] m/s`, `w0 in [24.0, 30.0] rad/s`, `tuck in [0.5, 1.0]` closes
  360 deg at every combination tested across all three `z0 in {0.10, 0.15,
  0.20}` (54 rows, directly measured), with landing speed `1.46-2.59 m/s`
  throughout -- every cell in the box beats the original document's worst
  closing-cell landing speed (4.3 m/s) and the large majority beat its best
  (3.14 m/s). Direction was directly checked at every combination of both
  `tuck` extremes x both `z0` extremes at the box's `w0` ceiling (30, where
  direction margin is thinnest), plus the box's single worst-landing point
  -- five points total, all confirmed backward (see "Direction verification
  at the box corners"). The box's interior was not checked cell-by-cell;
  that relies on the boundary-scan trend, not an exhaustive trace.
- **The `vz=2.00` floor is close to a real cliff, but clears it with
  measured margin**: at the box's worst-margin corner (`w0=30, tuck=1.0,
  z0=0.10`), closure fails at `vz=1.80` (346.8 deg) and barely passes at
  `vz=1.85` (360.8 deg, ~1 deg of margin) -- `vz=2.00` sits roughly
  0.15-0.20 m/s above that cliff, with a much healthier 39.5 deg of margin
  (399.5 deg). The floor does not need to move up, but it is closer to a
  real threshold than the box's headline numbers suggest (see "The vz
  floor: cliff or margin?").
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


# Standing-spawn re-measurement (final-review fix wave)

**Why this section exists.** A final whole-branch review found that the whole
document above measures a posture the env never produces. The probe spawned
the robot pre-**TUCKED**, squatting with its trunk ~2 cm above the plate top
and holding the tuck ctrl from t=0. The env spawns it **STANDING**:
`reset_backflip_robot_on_plate` puts the trunk at
`z0 + PLATE_HALF_THICKNESS + STAND_Z` (plate top + 0.115 m) in the HOME pose,
and the `ready_stance` reward pays the policy to still be there when the flick
arrives. CoM height is ~3 cm tucked vs ~11 cm standing, and CoM height is
exactly what decides whether the flick tips the robot backward over its heels
or overdrives the sole contact.

**What changed in the probe** (`scripts/backflip_envelope.py`, same script — new
flags, no second script):

- `--posture standing|tucked`. `standing` is now the DEFAULT and reproduces the
  env's spawn arithmetic exactly (no settle offset: the env spawns at rest at
  that height and the flick arrives `t_hold` later, whatever the pose has
  drifted to). `tucked` reproduces the original mode, so every table above is
  still reproducible verbatim and the two are directly diffable.
- `--tuck-at-flick`: command the tuck pose (HOME with the TUCK overrides, times
  the tuck factor) from `t >= t_hold`. The policy can act during HOLD and
  LAUNCH, so this is the fair "what the policy could do" variant; without it a
  standing cell's `tuck` column is inert.
- `--bam`: run the 14 servos through the **BAM M6 XL330** model — the actuator
  training actually uses (AGENTS.md: "Actuators are BAM") — at the training sim
  timestep (0.005). Reuses `scripts/infer_policy.py`'s loader rather than
  re-deriving it. **All tables in this section are BAM unless stated
  otherwise**; the tables above are XML-PD at dt 0.002. BAM turned out to
  matter a lot for the HOLD drift (see the settle test), which is why it was
  added rather than assumed away.
- `--hold` / `--launch` / `--timestep`, and a `tilt0` column: trunk tilt (deg,
  angle between the robot's own +z and world +z) at the instant the flick
  starts. For the standing spawn that column is the diagnostic — it says how
  much of the commanded stance actually survived the hold.
- `--settle` / `--settle-on-floor` / `--settle-trials` / `--settle-noiseless`:
  the AGENTS.md-mandated HOLD-equilibrium settle test (below).

Every correctness mechanism the probe already had is untouched: the pre-step
landing-speed snapshot, the first-contact accumulator latch, the
plate-excluded floor-contact filter, and `--check-direction`. **Sign
convention unchanged**: `rot_deg > 0` = backward (a real backflip),
`rot_deg < 0` = FORWARD (face-down). Every cell quoted as a recommendation or
as a headline failure below was orientation-verified directly, not inferred
from the accumulator's sign.

## Head-to-head: only the posture changed

BAM, `t_hold=0.3`, `t_launch=0.12`, `tuck=1.0`, at the DR box the cfg
currently samples (`vz` in {2.00, 2.25}, `w0` in {24, 30}), all three `z0`:

```
# posture=standing tuck_at_flick=False z0=0.1 hold=0.3 launch=0.12 bam=True dt=0.005
   vz     w0  tuck  rot_deg  land_m/s  apex_m  tilt0
 2.00   24.0  1.00    -83.5      2.72   0.488    4.9  
 2.00   30.0  1.00   -250.8      3.03   0.477    4.9  
 2.25   24.0  1.00    -85.9      2.88   0.533    4.9  
 2.25   30.0  1.00   -260.9      3.11   0.505    4.9  
# posture=tucked tuck_at_flick=False z0=0.1 hold=0.3 launch=0.12 bam=True dt=0.005
   vz     w0  tuck  rot_deg  land_m/s  apex_m  tilt0
 2.00   24.0  1.00    447.5      1.73   0.421   15.1  
 2.00   30.0  1.00    409.9      1.37   0.332   15.1  
 2.25   24.0  1.00    478.9      1.84   0.482   15.1  
 2.25   30.0  1.00    445.2      1.27   0.380   15.1  
# posture=standing tuck_at_flick=False z0=0.15 hold=0.3 launch=0.12 bam=True dt=0.005
   vz     w0  tuck  rot_deg  land_m/s  apex_m  tilt0
 2.00   24.0  1.00    -86.2      2.86   0.538    4.9  
 2.00   30.0  1.00   -259.4      3.16   0.527    4.9  
 2.25   24.0  1.00    -88.6      3.02   0.583    4.9  
 2.25   30.0  1.00   -269.4      3.24   0.555    4.9  
# posture=tucked tuck_at_flick=False z0=0.15 hold=0.3 launch=0.12 bam=True dt=0.005
   vz     w0  tuck  rot_deg  land_m/s  apex_m  tilt0
 2.00   24.0  1.00    464.1      1.86   0.471   15.1  
 2.00   30.0  1.00    429.9      1.41   0.382   15.1  
 2.25   24.0  1.00    494.6      2.11   0.532   15.1  
 2.25   30.0  1.00    464.3      1.52   0.430   15.1  
# posture=standing tuck_at_flick=False z0=0.2 hold=0.3 launch=0.12 bam=True dt=0.005
   vz     w0  tuck  rot_deg  land_m/s  apex_m  tilt0
 2.00   24.0  1.00    -89.0      3.01   0.588    4.9  
 2.00   30.0  1.00   -268.0      3.29   0.577    4.9  
 2.25   24.0  1.00    -91.2      3.17   0.633    4.9  
 2.25   30.0  1.00   -275.1      3.33   0.605    4.9  
# posture=tucked tuck_at_flick=False z0=0.2 hold=0.3 launch=0.12 bam=True dt=0.005
   vz     w0  tuck  rot_deg  land_m/s  apex_m  tilt0
 2.00   24.0  1.00    475.1      1.98   0.521   15.1  
 2.00   30.0  1.00    449.8      1.56   0.432   15.1  
 2.25   24.0  1.00    515.3      2.52   0.582   15.1  
 2.25   30.0  1.00    489.5      1.99   0.480   15.1  
```

Read the `rot_deg` column: **standing is NEGATIVE in every one of those 12
cells** — the currently-configured DR box rotates the env's actual spawn
FORWARD, up to -275 deg, while the same box from a tucked spawn closes
410-515 deg backward at 1.27-2.52 m/s. `tilt0` also differs (4.9 deg standing
vs 15.1 deg tucked), but see "the drift is not the cause" below: it isn't the
mechanism.

**Direction verified, not inferred.** The forward result is what a mislabeled
sign would look like, so the four worst standing cells were traced directly
(local +z axis in world coords at 90 deg of accumulated rotation):

```
standing vz=2.00 w0=30 tuck=1.0 z0=0.15 -> rot=-259.4 land=3.16 apex=0.527
  t=0.580 accum=-90.7 local+z=(0.917,0.004,-0.398) tilt=113.5  -> +x = FACE-DOWN = FORWARD
standing vz=2.25 w0=30 tuck=1.0 z0=0.10 -> rot=-260.9 land=3.11 apex=0.505
  t=0.580 accum=-90.2 local+z=(0.924,0.001,-0.383) tilt=112.5  -> +x = FACE-DOWN = FORWARD
standing vz=2.25 w0=24 tuck=1.0 z0=0.20 -> rot=-91.2  land=3.17 apex=0.633
  t=0.940 accum=-90.3 local+z=(0.964,0.019,-0.267) tilt=105.5  -> +x = FACE-DOWN = FORWARD
standing vz=2.00 w0=24 tuck=1.0 z0=0.15 -> rot=-86.2  land=2.86 apex=0.538
  (never reaches 90 deg in EITHER direction: it barely rotates at all)
```

## The drift is not the cause

The standing robot drifts forward during HOLD (4.9 deg of tilt by t=0.3 s under
BAM), and an upward-accelerating plate amplifies whatever lean is already
there. That is real but it is NOT the mechanism, because shortening the hold to
near zero does not rescue the flip. BAM, `z0=0.15`, `vz=2.00`, `tuck=1.0`:

```
        hold=0.02          hold=0.10          hold=0.30
        (tilt0=0.1 deg)    (tilt0=1.2 deg)    (tilt0=4.9 deg)
  w0     rot_deg            rot_deg            rot_deg
   6.0     +9.9               +0.3              -19.9
  12.0    +27.6              +13.8              -12.3
  18.0     +7.4              -10.6              -45.2
  24.0    -33.3              -52.7              -86.2
  30.0   -145.5             -172.7             -259.4
```

Even with essentially zero drift the best standing cell banks **+27.6 deg** —
7.7% of a flip — and the rotation reverses to forward by `w0` ~ 24. The
reversal boundary is at `w0` ~ 18-24 standing versus `w0` ~ 36-39 tucked, i.e.
**inside the box the env samples**. The posture, not the drift, moves it.

## Standing sweep, no policy action (BAM)

`vz` in {2.0 ... 4.0} step 0.5, `w0` in {3 ... 36} step 3, `tuck=1.0`
(inert here), `t_hold=0.3`, `t_launch=0.12`. 180 cells across three `z0`.
**Not one cell in any of them is positive.** Best (least-forward) cells per
`z0`:

```
z0=0.10:  4.00  18.0   -7.9   4.11  0.981     z0=0.15:  4.00  18.0   -8.0   4.21  1.031
          4.00  21.0  -10.5   4.01  0.948               4.00  21.0  -10.7   4.15  0.998
          2.00   9.0  -12.3   2.76  0.506               2.00   9.0  -12.7   2.91  0.556
z0=0.20:  4.00  18.0   -8.1   4.36  1.081
          4.00  21.0  -10.8   4.25  1.048
          2.00  12.0  -12.7   3.04  0.598
```

Full raw output, all three `z0`:

```
NOTE: rows with w0 > 30.0 rad/s are marked '*' below. Those cells are NOT direction-verified -- multiple cells in this exact range have been directly checked (--check-direction) and came back FORWARD rolls despite reporting a large 'backward' rot_deg. See docs/backflip_envelope_results.md "The reversal" before trusting any of them, especially the best-looking (lowest land_m/s) ones.
# posture=standing tuck_at_flick=False z0=0.15 hold=0.3 launch=0.12 bam=True dt=0.005
   vz     w0  tuck  rot_deg  land_m/s  apex_m  tilt0
 2.00    3.0  1.00    -39.7      3.06   0.586    4.9  
 2.00    6.0  1.00    -19.9      2.94   0.562    4.9  
 2.00    9.0  1.00    -12.7      2.91   0.556    4.9  
 2.00   12.0  1.00    -12.3      2.89   0.548    4.9  
 2.00   15.0  1.00    -26.2      2.94   0.549    4.9  
 2.00   18.0  1.00    -45.2      2.98   0.550    4.9  
 2.00   21.0  1.00    -66.6      3.10   0.547    4.9  
 2.00   24.0  1.00    -86.2      2.86   0.538    4.9  
 2.00   27.0  1.00   -176.3      2.90   0.536    4.9  
 2.00   30.0  1.00   -259.4      3.16   0.527    4.9  
 2.00   33.0  1.00   -288.4      2.97   0.538    4.9 *
 2.00   36.0  1.00   -295.7      2.83   0.523    4.9 *
 2.50    3.0  1.00    -66.4      3.63   0.729    4.9  
 2.50    6.0  1.00    -39.1      3.43   0.694    4.9  
 2.50    9.0  1.00    -23.8      3.31   0.673    4.9  
 2.50   12.0  1.00    -16.9      3.29   0.666    4.9  
 2.50   15.0  1.00    -15.1      3.21   0.648    4.9  
 2.50   18.0  1.00    -34.7      3.27   0.646    4.9  
 2.50   21.0  1.00    -60.3      3.33   0.641    4.9  
 2.50   24.0  1.00    -88.8      3.19   0.630    4.9  
 2.50   27.0  1.00   -107.3      3.01   0.611    4.9  
 2.50   30.0  1.00   -262.2      3.35   0.597    4.9  
 2.50   33.0  1.00   -289.5      2.98   0.546    4.9 *
 2.50   36.0  1.00   -304.3      2.86   0.554    4.9 *
 3.00    3.0  1.00    -94.4      3.88   0.894    4.9  
 3.00    6.0  1.00    -69.0      3.93   0.862    4.9  
 3.00    9.0  1.00    -41.9      3.76   0.825    4.9  
 3.00   12.0  1.00    -22.2      3.63   0.790    4.9  
 3.00   15.0  1.00    -16.6      3.58   0.774    4.9  
 3.00   18.0  1.00    -16.6      3.51   0.753    4.9  
 3.00   21.0  1.00    -44.2      3.54   0.742    4.9  
 3.00   24.0  1.00    -81.7      3.54   0.731    4.9  
 3.00   27.0  1.00   -111.5      3.31   0.707    4.9  
 3.00   30.0  1.00   -212.5      3.52   0.694    4.9  
 3.00   33.0  1.00   -279.2      3.37   0.627    4.9 *
 3.00   36.0  1.00   -290.0      2.96   0.546    4.9 *
 3.50    3.0  1.00   -133.3      4.25   1.088    4.9  
 3.50    6.0  1.00   -103.4      4.21   1.054    4.9  
 3.50    9.0  1.00    -68.1      4.31   0.999    4.9  
 3.50   12.0  1.00    -33.6      4.07   0.947    4.9  
 3.50   15.0  1.00    -14.4      3.93   0.909    4.9  
 3.50   18.0  1.00    -17.0      3.89   0.890    4.9  
 3.50   21.0  1.00    -18.4      3.80   0.856    4.9  
 3.50   24.0  1.00    -64.3      3.86   0.836    4.9  
 3.50   27.0  1.00   -106.1      3.61   0.814    4.9  
 3.50   30.0  1.00   -121.6      3.49   0.772    4.9  
 3.50   33.0  1.00   -263.3      3.78   0.744    4.9 *
 3.50   36.0  1.00   -288.0      3.31   0.652    4.9 *
 4.00    3.0  1.00   -182.7      4.82   1.304    4.9  
 4.00    6.0  1.00   -145.7      4.69   1.270    4.9  
 4.00    9.0  1.00    -99.2      4.57   1.196    4.9  
 4.00   12.0  1.00    -61.5      4.59   1.144    4.9  
 4.00   15.0  1.00    -26.3      4.37   1.083    4.9  
 4.00   18.0  1.00     -8.0      4.21   1.031    4.9  
 4.00   21.0  1.00    -10.7      4.15   0.998    4.9  
 4.00   24.0  1.00    -33.8      4.09   0.951    4.9  
 4.00   27.0  1.00    -89.3      3.98   0.922    4.9  
 4.00   30.0  1.00   -116.4      3.76   0.876    4.9  
 4.00   33.0  1.00   -128.0      3.64   0.830    4.9 *
 4.00   36.0  1.00   -258.7      3.85   0.781    4.9 *
```

```
NOTE: rows with w0 > 30.0 rad/s are marked '*' below. Those cells are NOT direction-verified -- multiple cells in this exact range have been directly checked (--check-direction) and came back FORWARD rolls despite reporting a large 'backward' rot_deg. See docs/backflip_envelope_results.md "The reversal" before trusting any of them, especially the best-looking (lowest land_m/s) ones.
# posture=standing tuck_at_flick=False z0=0.1 hold=0.3 launch=0.12 bam=True dt=0.005
   vz     w0  tuck  rot_deg  land_m/s  apex_m  tilt0
 2.00    3.0  1.00    -38.5      2.92   0.536    4.9  
 2.00    6.0  1.00    -19.2      2.79   0.512    4.9  
 2.00    9.0  1.00    -12.3      2.76   0.506    4.9  
 2.00   12.0  1.00    -12.0      2.75   0.498    4.9  
 2.00   15.0  1.00    -25.2      2.74   0.499    4.9  
 2.00   18.0  1.00    -43.3      2.79   0.500    4.9  
 2.00   21.0  1.00    -64.0      2.90   0.497    4.9  
 2.00   24.0  1.00    -83.5      2.72   0.488    4.9  
 2.00   27.0  1.00   -168.4      2.68   0.486    4.9  
 2.00   30.0  1.00   -250.8      3.03   0.477    4.9  
 2.00   33.0  1.00   -278.6      2.85   0.488    4.9 *
 2.00   36.0  1.00   -285.2      2.73   0.473    4.9 *
 2.50    3.0  1.00    -64.8      3.49   0.679    4.9  
 2.50    6.0  1.00    -38.1      3.28   0.644    4.9  
 2.50    9.0  1.00    -23.2      3.16   0.623    4.9  
 2.50   12.0  1.00    -16.4      3.14   0.616    4.9  
 2.50   15.0  1.00    -14.7      3.07   0.598    4.9  
 2.50   18.0  1.00    -33.5      3.07   0.596    4.9  
 2.50   21.0  1.00    -58.7      3.18   0.591    4.9  
 2.50   24.0  1.00    -86.3      3.05   0.580    4.9  
 2.50   27.0  1.00   -104.2      2.86   0.561    4.9  
 2.50   30.0  1.00   -254.6      3.21   0.547    4.9  
 2.50   33.0  1.00   -279.6      2.87   0.496    4.9 *
 2.50   36.0  1.00   -293.9      2.76   0.504    4.9 *
 3.00    3.0  1.00    -93.0      3.79   0.844    4.9  
 3.00    6.0  1.00    -68.0      3.83   0.812    4.9  
 3.00    9.0  1.00    -41.2      3.66   0.775    4.9  
 3.00   12.0  1.00    -21.7      3.48   0.740    4.9  
 3.00   15.0  1.00    -16.2      3.44   0.724    4.9  
 3.00   18.0  1.00    -16.2      3.36   0.703    4.9  
 3.00   21.0  1.00    -43.2      3.39   0.692    4.9  
 3.00   24.0  1.00    -79.7      3.39   0.681    4.9  
 3.00   27.0  1.00   -108.6      3.16   0.657    4.9  
 3.00   30.0  1.00   -205.1      3.31   0.644    4.9  
 3.00   33.0  1.00   -271.1      3.24   0.577    4.9 *
 3.00   36.0  1.00   -279.8      2.85   0.496    4.9 *
 3.50    3.0  1.00   -131.6      4.15   1.038    4.9  
 3.50    6.0  1.00   -102.1      4.11   1.004    4.9  
 3.50    9.0  1.00    -66.8      4.16   0.949    4.9  
 3.50   12.0  1.00    -32.9      3.92   0.897    4.9  
 3.50   15.0  1.00    -14.3      3.83   0.859    4.9  
 3.50   18.0  1.00    -16.7      3.75   0.840    4.9  
 3.50   21.0  1.00    -18.0      3.66   0.806    4.9  
 3.50   24.0  1.00    -63.4      3.76   0.786    4.9  
 3.50   27.0  1.00   -104.5      3.51   0.764    4.9  
 3.50   30.0  1.00   -118.7      3.34   0.722    4.9  
 3.50   33.0  1.00   -256.7      3.64   0.694    4.9 *
 3.50   36.0  1.00   -282.4      3.23   0.602    4.9 *
 4.00    3.0  1.00   -180.7      4.71   1.254    4.9  
 4.00    6.0  1.00   -144.0      4.59   1.220    4.9  
 4.00    9.0  1.00    -98.0      4.47   1.146    4.9  
 4.00   12.0  1.00    -60.7      4.50   1.094    4.9  
 4.00   15.0  1.00    -25.9      4.22   1.033    4.9  
 4.00   18.0  1.00     -7.9      4.11   0.981    4.9  
 4.00   21.0  1.00    -10.5      4.01   0.948    4.9  
 4.00   24.0  1.00    -33.2      3.94   0.901    4.9  
 4.00   27.0  1.00    -88.1      3.88   0.872    4.9  
 4.00   30.0  1.00   -114.7      3.66   0.826    4.9  
 4.00   33.0  1.00   -125.0      3.49   0.780    4.9 *
 4.00   36.0  1.00   -254.6      3.75   0.731    4.9 *
```

```
NOTE: rows with w0 > 30.0 rad/s are marked '*' below. Those cells are NOT direction-verified -- multiple cells in this exact range have been directly checked (--check-direction) and came back FORWARD rolls despite reporting a large 'backward' rot_deg. See docs/backflip_envelope_results.md "The reversal" before trusting any of them, especially the best-looking (lowest land_m/s) ones.
# posture=standing tuck_at_flick=False z0=0.2 hold=0.3 launch=0.12 bam=True dt=0.005
   vz     w0  tuck  rot_deg  land_m/s  apex_m  tilt0
 2.00    3.0  1.00    -41.3      3.26   0.636    4.9  
 2.00    6.0  1.00    -20.7      3.13   0.612    4.9  
 2.00    9.0  1.00    -13.3      3.10   0.606    4.9  
 2.00   12.0  1.00    -12.7      3.04   0.598    4.9  
 2.00   15.0  1.00    -27.0      3.09   0.599    4.9  
 2.00   18.0  1.00    -46.5      3.13   0.600    4.9  
 2.00   21.0  1.00    -68.6      3.25   0.597    4.9  
 2.00   24.0  1.00    -89.0      3.01   0.588    4.9  
 2.00   27.0  1.00   -182.3      3.06   0.586    4.9  
 2.00   30.0  1.00   -268.0      3.29   0.577    4.9  
 2.00   33.0  1.00   -298.1      3.08   0.588    4.9 *
 2.00   36.0  1.00   -306.2      2.93   0.573    4.9 *
 2.50    3.0  1.00    -68.0      3.78   0.779    4.9  
 2.50    6.0  1.00    -40.1      3.57   0.744    4.9  
 2.50    9.0  1.00    -24.4      3.46   0.723    4.9  
 2.50   12.0  1.00    -17.3      3.43   0.716    4.9  
 2.50   15.0  1.00    -15.5      3.36   0.698    4.9  
 2.50   18.0  1.00    -35.7      3.41   0.696    4.9  
 2.50   21.0  1.00    -61.9      3.48   0.691    4.9  
 2.50   24.0  1.00    -90.4      3.29   0.680    4.9  
 2.50   27.0  1.00   -110.5      3.16   0.661    4.9  
 2.50   30.0  1.00   -267.3      3.44   0.647    4.9  
 2.50   33.0  1.00   -299.3      3.09   0.596    4.9 *
 2.50   36.0  1.00   -314.7      2.96   0.604    4.9 *
 3.00    3.0  1.00    -95.7      3.98   0.944    4.9  
 3.00    6.0  1.00    -70.5      4.08   0.912    4.9  
 3.00    9.0  1.00    -42.8      3.90   0.875    4.9  
 3.00   12.0  1.00    -22.7      3.77   0.840    4.9  
 3.00   15.0  1.00    -16.9      3.73   0.824    4.9  
 3.00   18.0  1.00    -17.0      3.65   0.803    4.9  
 3.00   21.0  1.00    -45.3      3.69   0.792    4.9  
 3.00   24.0  1.00    -83.0      3.63   0.781    4.9  
 3.00   27.0  1.00   -114.3      3.46   0.757    4.9  
 3.00   30.0  1.00   -217.9      3.67   0.744    4.9  
 3.00   33.0  1.00   -284.6      3.45   0.677    4.9 *
 3.00   36.0  1.00   -300.3      3.06   0.596    4.9 *
 3.50    3.0  1.00   -135.9      4.40   1.138    4.9  
 3.50    6.0  1.00   -105.4      4.36   1.104    4.9  
 3.50    9.0  1.00    -69.0      4.40   1.049    4.9  
 3.50   12.0  1.00    -34.0      4.17   0.997    4.9  
 3.50   15.0  1.00    -14.7      4.08   0.959    4.9  
 3.50   18.0  1.00    -17.3      3.99   0.940    4.9  
 3.50   21.0  1.00    -18.7      3.95   0.906    4.9  
 3.50   24.0  1.00    -65.7      4.00   0.886    4.9  
 3.50   27.0  1.00   -108.6      3.75   0.864    4.9  
 3.50   30.0  1.00   -123.6      3.59   0.822    4.9  
 3.50   33.0  1.00   -267.6      3.88   0.794    4.9 *
 3.50   36.0  1.00   -296.4      3.43   0.702    4.9 *
 4.00    3.0  1.00   -184.8      4.92   1.354    4.9  
 4.00    6.0  1.00   -147.4      4.79   1.320    4.9  
 4.00    9.0  1.00   -100.3      4.67   1.246    4.9  
 4.00   12.0  1.00    -62.2      4.69   1.194    4.9  
 4.00   15.0  1.00    -26.6      4.46   1.133    4.9  
 4.00   18.0  1.00     -8.1      4.36   1.081    4.9  
 4.00   21.0  1.00    -10.8      4.25   1.048    4.9  
 4.00   24.0  1.00    -34.2      4.19   1.001    4.9  
 4.00   27.0  1.00    -91.1      4.13   0.972    4.9  
 4.00   30.0  1.00   -118.9      3.91   0.926    4.9  
 4.00   33.0  1.00   -130.0      3.74   0.880    4.9 *
 4.00   36.0  1.00   -265.0      3.99   0.831    4.9 *
```

## Standing sweep WITH the policy tucking at the flick (BAM)

This is the fair version — the policy feels the flick and can tuck.
`--tuck-at-flick`, `vz` in {2.0 ... 4.0} step 0.5, `w0` in {3 ... 36} step 3,
`tuck` in {0.5, 1.0}, `t_launch=0.12`. **360 deg now closes — but only at
`vz >= 3.0`, i.e. only at landing speeds far over the hardware limit.**

Lowest-landing cells that close 360 deg (all `z0`):

```
 vz     w0  tuck  rot_deg  land_m/s  apex_m
 3.50  21.0  1.00    524.6      3.69   0.986   (z0=0.10)  <- lowest landing of any 360-closing cell at t_launch=0.12
 3.50  24.0  1.00    495.6      3.65   0.912   (z0=0.10)
 3.50  21.0  1.00    531.4      3.78   1.036   (z0=0.15)
 4.00  18.0  1.00    563.3      4.24   1.249   (z0=0.15)
 3.00  18.0  1.00    362.6      4.18   0.812   (z0=0.10)
```

Cells matching `rot_deg >= 360` **and** `land_m/s <= 2.6`: **none.**
Cells matching `rot_deg >= 300` **and** `land_m/s <= 2.9`: **none.**

Direction traces on the three best 360-closing cells (all PASS, backward):

```
vz=3.5 w0=21 tuck=1.0 z0=0.15: accum=90.7 @0.555s local+z=(-0.995,0.020,0.101)  -> BACKWARD. rot=531.4 land=3.78
vz=3.5 w0=24 tuck=1.0 z0=0.15: accum=90.1 @0.555s local+z=(-0.999,-0.020,0.031) -> BACKWARD. rot=502.4 land=3.72
vz=4.0 w0=18 tuck=1.0 z0=0.15: accum=91.3 @0.560s local+z=(-0.988,-0.044,0.150) -> BACKWARD. rot=563.3 land=4.24
```

Full raw output (`z0=0.15`; `z0=0.10` / `0.20` follow the same surface,
shifted by the extra/reduced drop):

```
NOTE: rows with w0 > 30.0 rad/s are marked '*' below. Those cells are NOT direction-verified -- multiple cells in this exact range have been directly checked (--check-direction) and came back FORWARD rolls despite reporting a large 'backward' rot_deg. See docs/backflip_envelope_results.md "The reversal" before trusting any of them, especially the best-looking (lowest land_m/s) ones.
# posture=standing tuck_at_flick=True z0=0.15 hold=0.3 launch=0.12 bam=True dt=0.005
   vz     w0  tuck  rot_deg  land_m/s  apex_m  tilt0
 2.00    3.0  0.50   -172.1      3.03   0.559    4.9  
 2.00    3.0  1.00   -271.1      2.49   0.477    4.9  
 2.00    6.0  0.50   -144.9      2.83   0.534    4.9  
 2.00    6.0  1.00   -261.0      2.56   0.463    4.9  
 2.00    9.0  0.50   -120.6      2.75   0.510    4.9  
 2.00    9.0  1.00   -224.7      2.74   0.445    4.9  
 2.00   12.0  0.50    -99.1      2.62   0.488    4.9  
 2.00   12.0  1.00   -190.3      2.88   0.423    4.9  
 2.00   15.0  0.50    -86.7      2.56   0.473    4.9  
 2.00   15.0  1.00     15.3      2.78   0.462    4.9  
 2.00   18.0  0.50    -85.1      2.51   0.463    4.9  
 2.00   18.0  1.00    127.5      2.80   0.477    4.9  
 2.00   21.0  0.50    -89.7      2.48   0.454    4.9  
 2.00   21.0  1.00    147.5      2.61   0.464    4.9  
 2.00   24.0  0.50   -102.7      2.47   0.449    4.9  
 2.00   24.0  1.00    112.1      2.63   0.441    4.9  
 2.00   27.0  0.50   -117.0      2.49   0.445    4.9  
 2.00   27.0  1.00   -105.1      2.31   0.396    4.9  
 2.00   30.0  0.50   -122.3      2.46   0.435    4.9  
 2.00   30.0  1.00   -150.1      2.51   0.365    4.9  
 2.00   33.0  0.50   -154.1      2.47   0.413    4.9 *
 2.00   33.0  1.00   -157.0      2.57   0.362    4.9 *
 2.00   36.0  0.50   -152.5      2.44   0.378    4.9 *
 2.00   36.0  1.00   -167.1      2.65   0.360    4.9 *
 2.50    3.0  0.50   -236.7      3.54   0.679    4.9  
 2.50    3.0  1.00   -371.0      2.58   0.564    4.9  
 2.50    6.0  0.50   -209.3      3.51   0.658    4.9  
 2.50    6.0  1.00   -365.7      2.53   0.550    4.9  
 2.50    9.0  0.50   -165.3      3.20   0.626    4.9  
 2.50    9.0  1.00   -309.7      2.48   0.530    4.9  
 2.50   12.0  0.50   -135.4      3.02   0.598    4.9  
 2.50   12.0  1.00     -8.9      3.05   0.559    4.9  
 2.50   15.0  0.50   -109.9      2.90   0.570    4.9  
 2.50   15.0  1.00    187.6      3.09   0.607    4.9  
 2.50   18.0  0.50   -101.8      2.85   0.551    4.9  
 2.50   18.0  1.00    162.2      2.98   0.576    4.9  
 2.50   21.0  0.50   -100.7      2.78   0.534    4.9  
 2.50   21.0  1.00    331.6      3.69   0.639    4.9  
 2.50   24.0  0.50   -105.8      2.74   0.517    4.9  
 2.50   24.0  1.00      2.7      2.93   0.519    4.9  
 2.50   27.0  0.50   -123.0      2.73   0.508    4.9  
 2.50   27.0  1.00    123.5      2.84   0.499    4.9  
 2.50   30.0  0.50   -131.2      2.66   0.495    4.9  
 2.50   30.0  1.00   -162.8      2.68   0.401    4.9  
 2.50   33.0  0.50   -136.6      2.65   0.482    4.9 *
 2.50   33.0  1.00   -172.6      2.73   0.396    4.9 *
 2.50   36.0  0.50   -178.5      2.73   0.438    4.9 *
 2.50   36.0  1.00   -188.5      2.83   0.393    4.9 *
 3.00    3.0  0.50   -299.6      3.57   0.812    4.9  
 3.00    3.0  1.00   -456.4      3.39   0.651    4.9  
 3.00    6.0  0.50   -269.6      3.68   0.792    4.9  
 3.00    6.0  1.00   -452.1      3.31   0.639    4.9  
 3.00    9.0  0.50   -228.7      3.77   0.761    4.9  
 3.00    9.0  1.00   -398.3      2.84   0.621    4.9  
 3.00   12.0  0.50   -183.8      3.54   0.727    4.9  
 3.00   12.0  1.00    -82.2      3.47   0.747    4.9  
 3.00   15.0  0.50   -140.9      3.29   0.685    4.9  
 3.00   15.0  1.00    226.1      3.41   0.754    4.9  
 3.00   18.0  0.50   -118.4      3.20   0.652    4.9  
 3.00   18.0  1.00    362.6      4.18   0.812    4.9  
 3.00   21.0  0.50   -116.3      3.14   0.629    4.9  
 3.00   21.0  1.00    247.6      3.88   0.783    4.9  
 3.00   24.0  0.50   -116.0      3.05   0.604    4.9  
 3.00   24.0  1.00    265.1      3.49   0.686    4.9  
 3.00   27.0  0.50   -123.1      2.95   0.579    4.9  
 3.00   27.0  1.00     34.3      3.20   0.583    4.9  
 3.00   30.0  0.50   -135.6      2.91   0.559    4.9  
 3.00   30.0  1.00    131.0      2.88   0.535    4.9  
 3.00   33.0  0.50   -144.0      2.86   0.543    4.9 *
 3.00   33.0  1.00   -187.8      2.93   0.435    4.9 *
 3.00   36.0  0.50   -150.9      2.84   0.530    4.9 *
 3.00   36.0  1.00   -192.8      2.87   0.428    4.9 *
 3.50    3.0  0.50   -381.5      3.82   0.956    4.9  
 3.50    3.0  1.00   -584.6      3.95   0.747    4.9  
 3.50    6.0  0.50   -343.3      3.71   0.939    4.9  
 3.50    6.0  1.00   -581.8      3.95   0.734    4.9  
 3.50    9.0  0.50   -286.4      3.88   0.901    4.9  
 3.50    9.0  1.00    -70.5      3.47   0.832    4.9  
 3.50   12.0  0.50   -239.6      4.00   0.867    4.9  
 3.50   12.0  1.00    108.6      4.19   0.962    4.9  
 3.50   15.0  0.50   -179.3      3.77   0.821    4.9  
 3.50   15.0  1.00    384.4      4.58   1.009    4.9  
 3.50   18.0  0.50   -145.7      3.54   0.777    4.9  
 3.50   18.0  1.00    403.9      4.48   1.002    4.9  
 3.50   21.0  0.50   -126.5      3.42   0.738    4.9  
 3.50   21.0  1.00    531.4      3.78   1.036    4.9  
 3.50   24.0  0.50   -131.8      3.36   0.707    4.9  
 3.50   24.0  1.00    502.4      3.72   0.962    4.9  
 3.50   27.0  0.50   -127.8      3.21   0.666    4.9  
 3.50   27.0  1.00    382.2      4.10   0.798    4.9  
 3.50   30.0  0.50   -130.6      3.10   0.629    4.9  
 3.50   30.0  1.00    -21.6      3.34   0.661    4.9  
 3.50   33.0  0.50   -145.2      3.07   0.608    4.9 *
 3.50   33.0  1.00    174.4      3.02   0.601    4.9 *
 3.50   36.0  0.50   -156.4      3.06   0.592    4.9 *
 3.50   36.0  1.00     42.2      3.06   0.558    4.9 *
 4.00    3.0  0.50   -461.0      4.44   1.105    4.9  
 4.00    3.0  1.00   -713.8      3.48   0.926    4.9  
 4.00    6.0  0.50   -421.1      4.20   1.089    4.9  
 4.00    6.0  1.00   -721.6      3.46   0.912    4.9  
 4.00    9.0  0.50   -364.4      4.02   1.059    4.9  
 4.00    9.0  1.00   -315.8      4.26   1.120    4.9  
 4.00   12.0  0.50   -300.4      4.14   1.020    4.9  
 4.00   12.0  1.00    112.5      4.66   1.186    4.9  
 4.00   15.0  0.50   -237.6      4.28   0.977    4.9  
 4.00   15.0  1.00    289.8      4.95   1.211    4.9  
 4.00   18.0  0.50   -183.5      4.03   0.916    4.9  
 4.00   18.0  1.00    563.3      4.24   1.249    4.9  
 4.00   21.0  0.50   -158.8      3.82   0.872    4.9  
 4.00   21.0  1.00    236.7      3.99   1.100    4.9  
 4.00   24.0  0.50   -141.6      3.66   0.816    4.9  
 4.00   24.0  1.00    315.3      3.98   1.062    4.9  
 4.00   27.0  0.50   -148.1      3.55   0.773    4.9  
 4.00   27.0  1.00    388.0      4.40   0.953    4.9  
 4.00   30.0  0.50   -140.4      3.39   0.719    4.9  
 4.00   30.0  1.00    279.4      4.23   0.843    4.9  
 4.00   33.0  0.50   -138.1      3.27   0.677    4.9 *
 4.00   33.0  1.00     20.2      3.55   0.711    4.9 *
 4.00   36.0  0.50   -153.9      3.23   0.656    4.9 *
 4.00   36.0  1.00    -77.7      3.09   0.616    4.9 *
```

## Does the longest launch ramp rescue it? (BAM, `t_launch=0.15`)

`t_launch` is sampled in `[0.08, 0.15]`, and a longer ramp transfers more
angular impulse. It helps materially — and still does not reach the hardware
limit. `--tuck-at-flick`, `vz` in {2.0 ... 3.0} step 0.25, `w0` in
{9 ... 24} step 3, `tuck` in {0.75, 1.0}, all three `z0` (180 cells):

```
Cells with rot_deg >= 360 AND land_m/s <= 2.6:  NONE
Best rot_deg among cells landing <= 2.6 m/s:
 2.00  21.0  1.00    208.8      2.45   0.464   (z0=0.15)
 2.00  21.0  1.00    201.8      2.28   0.414   (z0=0.10)   <- direction-verified BACKWARD
 2.25  21.0  1.00    160.5      2.53   0.445
Lowest landing among cells that close 360:
 2.50  18.0  1.00    370.8      3.61   0.600   (z0=0.10)
 2.50  21.0  1.00    382.9      3.43   0.574   (z0=0.10)   <- direction-verified BACKWARD; BEST STANDING CELL FOUND
 2.50  21.0  1.00    393.7      3.51   0.624   (z0=0.15)
```

Direction traces:

```
vz=2.5 w0=21 tuck=1.0 z0=0.10 launch=0.15: accum=92.1 @0.575s local+z=(-0.986,-0.089,-0.144) -> BACKWARD. rot=382.9 land=3.43
vz=2.0 w0=21 tuck=1.0 z0=0.10 launch=0.15: accum=90.9 @0.640s local+z=(-0.990, 0.137, 0.041) -> BACKWARD. rot=201.8 land=2.28
```

The `t_launch=0.08` end of the range is strictly worse: every cell in
`vz` in {2.0, 2.25, 2.5} x `w0` in {15 ... 27} comes out forward
(-141 to -226 deg).

## The tucked box still holds under BAM (so the gap really is the posture)

To rule out "BAM breaks the flip" as the explanation, the published box was
re-run from the TUCKED spawn under BAM, `z0=0.15`:

```
NOTE: rows with w0 > 30.0 rad/s are marked '*' below. Those cells are NOT direction-verified -- multiple cells in this exact range have been directly checked (--check-direction) and came back FORWARD rolls despite reporting a large 'backward' rot_deg. See docs/backflip_envelope_results.md "The reversal" before trusting any of them, especially the best-looking (lowest land_m/s) ones.
# posture=tucked tuck_at_flick=False z0=0.15 hold=0.3 launch=0.12 bam=True dt=0.005
   vz     w0  tuck  rot_deg  land_m/s  apex_m  tilt0
 2.00   12.0  0.50    285.2      3.36   0.559   13.2  
 2.00   12.0  1.00    331.3      3.50   0.570   15.1  
 2.00   18.0  0.50    406.7      2.73   0.553   13.2  
 2.00   18.0  1.00    427.4      2.52   0.555   15.1  
 2.00   24.0  0.50    454.5      2.00   0.476   13.2  
 2.00   24.0  1.00    464.1      1.86   0.471   15.1  
 2.00   30.0  0.50    430.5      1.51   0.384   13.2  
 2.00   30.0  1.00    429.9      1.41   0.382   15.1  
 2.00   36.0  0.50    381.6      1.43   0.315   13.2 *
 2.00   36.0  1.00    352.6      1.71   0.318   15.1 *
 2.25   12.0  0.50    304.7      3.53   0.640   13.2  
 2.25   12.0  1.00    359.6      3.67   0.652   15.1  
 2.25   18.0  0.50    447.5      2.78   0.626   13.2  
 2.25   18.0  1.00    474.9      2.59   0.632   15.1  
 2.25   24.0  0.50    503.8      2.44   0.537   13.2  
 2.25   24.0  1.00    494.6      2.11   0.532   15.1  
 2.25   30.0  0.50    479.1      1.90   0.428   13.2  
 2.25   30.0  1.00    464.3      1.52   0.430   15.1  
 2.25   36.0  0.50    411.3      1.28   0.352   13.2 *
 2.25   36.0  1.00    411.5      1.26   0.350   15.1 *
 2.50   12.0  0.50    331.7      3.74   0.730   13.2  
 2.50   12.0  1.00    382.1      3.78   0.749   15.1  
 2.50   18.0  0.50    467.6      2.95   0.703   13.2  
 2.50   18.0  1.00    504.1      2.77   0.711   15.1  
 2.50   24.0  0.50    531.7      2.93   0.603   13.2  
 2.50   24.0  1.00    556.3      3.04   0.594   15.1  
 2.50   30.0  0.50    523.6      2.59   0.487   13.2  
 2.50   30.0  1.00    532.1      2.50   0.473   15.1  
 2.50   36.0  0.50    459.4      1.69   0.390   13.2 *
 2.50   36.0  1.00    440.2      1.23   0.382   15.1 *
 3.00   12.0  0.50    358.2      4.11   0.915   13.2  
 3.00   12.0  1.00    410.1      4.05   0.934   15.1  
 3.00   18.0  0.50    528.9      3.52   0.878   13.2  
 3.00   18.0  1.00    499.7      3.73   0.877   15.1  
 3.00   24.0  0.50    644.1      4.00   0.747   13.2  
 3.00   24.0  1.00    654.3      4.24   0.744   15.1  
 3.00   30.0  0.50    658.2      3.38   0.597   13.2  
 3.00   30.0  1.00    626.6      3.68   0.587   15.1  
 3.00   36.0  0.50    610.9      3.19   0.475   13.2 *
 3.00   36.0  1.00    597.7      3.60   0.472   15.1 *
```

`vz` in [2.00, 2.25] x `w0` in [24, 30] x `tuck` in [0.5, 1.0] closes
410-504 deg backward at **1.41-2.44 m/s** — the published box reproduces under
the training actuator. The tucked/standing gap is the posture, full stop.

## Verdict on the DR box: LEFT UNTOUCHED

No launch setting closes 360 deg from the env's standing spawn below the
~2.6 m/s hardware limit. The best direction-verified 360-closing standing cell
is `vz=2.5, w0=21, tuck-at-flick 1.0, t_launch=0.15, z0=0.10` -> **382.9 deg at
3.43 m/s** (~60 cm free-fall equivalent, ~32% over the limit). The best cell
that lands under 2.6 m/s reaches **201.8 deg** — a bit over half a flip.

Per the fix-wave brief, `Z0_RANGE` / `VZ_RANGE` / `W0_RANGE` / `MAX_PAID_RATE`
are therefore **left exactly as they were**, and no redesign was invented. The
cfg and `reset_backflip_launch_params` docstrings now carry the posture warning
instead. `MAX_PAID_RATE = 25.0` rad/s remains adequate on the measurement it
was set from (13-16 rad/s average over a closing flip) and is not the binding
constraint here.

**The option, for the user to decide.** The one launch posture that IS measured
to close 360 deg at a survivable landing speed is the tucked one, and the env
could produce it — the policy has `t_hold` (0.1-1.0 s) to crouch before the
flick. That means rethinking `ready_stance`, which currently pays the policy
`up * exp(-(z - (z0 + STAND_Z + PLATE_HALF_THICKNESS))^2 / 0.03^2)` for the
whole HOLD window, i.e. pays it specifically NOT to crouch. That is a design
change to the reward stack, so it is reported as an option with the evidence
above, not made here.

## HOLD-equilibrium settle test (AGENTS.md requirement)

`--settle`: hold HOME on the parked plate from noisy inits matching the env's
resets exactly (HOME joints + U(-0.05, 0.05) rad per `reset_robot_joints`,
x/y +-0.01 m, roll/pitch +-0.02 rad, yaw +-0.05 rad per the cfg's narrowed
`reset_base`, zero root velocity), plate rewritten at `z0` every step as
`backflip_plate_step` does during HOLD. **TILT, not just height** — a settle
test that only records z reports a toppled robot as resting fine.
`n_fallen` counts trials past 45 deg of tilt.

```
# settle: on plate z0=0.1 trials=32 noisy=True duration=3.0s dt=0.005 bam=True
  t[s]  tilt_mean  tilt_max  xy_mean  xy_max  z_mean  n_fallen
  0.10        0.9       2.0    0.003   0.007   0.222         0
  0.30        3.5       6.5    0.005   0.010   0.221         0
  0.50        7.3      11.7    0.008   0.016   0.220         0
  0.70       11.6      17.7    0.012   0.022   0.219         0
  1.00       23.2      42.4    0.028   0.064   0.217         0
  3.00      113.3     113.4    0.128   0.133   0.078        32
# settle: on plate z0=0.15 trials=32 noisy=True duration=3.0s dt=0.005 bam=True
  t[s]  tilt_mean  tilt_max  xy_mean  xy_max  z_mean  n_fallen
  0.10        0.9       2.0    0.003   0.007   0.272         0
  0.30        3.5       6.5    0.005   0.010   0.271         0
  0.50        7.3      11.7    0.008   0.016   0.270         0
  0.70       11.6      17.7    0.012   0.022   0.269         0
  1.00       23.2      42.4    0.028   0.064   0.267         0
  3.00      123.4     124.4    0.140   0.146   0.116        32
# settle: on plate z0=0.2 trials=32 noisy=True duration=3.0s dt=0.005 bam=True
  t[s]  tilt_mean  tilt_max  xy_mean  xy_max  z_mean  n_fallen
  0.10        0.9       2.0    0.003   0.007   0.322         0
  0.30        3.5       6.5    0.005   0.010   0.321         0
  0.50        7.3      11.7    0.008   0.016   0.320         0
  0.70       11.6      17.7    0.012   0.022   0.319         0
  1.00       23.2      42.4    0.028   0.064   0.317         0
  3.00      118.4     133.7    0.196   0.421   0.117        32
# settle: ON FLOOR (control) trials=32 noisy=True duration=3.0s dt=0.005 bam=True
  t[s]  tilt_mean  tilt_max  xy_mean  xy_max  z_mean  n_fallen
  0.10        0.9       2.0    0.003   0.007   0.116         0
  0.30        3.3       6.7    0.004   0.010   0.115         0
  0.50        6.9      12.7    0.007   0.016   0.115         0
  0.70       11.2      19.7    0.012   0.022   0.114         0
  1.00       23.6      54.1    0.030   0.082   0.111         2
  3.00       80.6      80.8    0.130   0.134   0.044        32
# settle: on plate z0=0.15 trials=32 noisy=True duration=3.0s dt=0.002 bam=False
  t[s]  tilt_mean  tilt_max  xy_mean  xy_max  z_mean  n_fallen
  0.10        0.9       2.0    0.003   0.007   0.272         0
  0.30        4.9       8.5    0.005   0.011   0.271         0
  0.50       12.7      19.0    0.013   0.022   0.269         0
  0.70       27.2      40.5    0.029   0.049   0.264         0
  1.00       98.5     124.5    0.139   0.174   0.180        31
  3.00      129.3     137.6    0.143   0.152   0.080        32
# settle: on plate z0=0.15 trials=1 noisy=False duration=3.0s dt=0.005 bam=True
  t[s]  tilt_mean  tilt_max  xy_mean  xy_max  z_mean  n_fallen
  0.10        1.2       1.2    0.001   0.001   0.272         0
  0.30        4.9       4.9    0.005   0.005   0.271         0
  0.50        8.9       8.9    0.009   0.009   0.270         0
  0.70       13.8      13.8    0.014   0.014   0.269         0
  1.00       26.8      26.8    0.033   0.033   0.266         0
  3.00      123.5     123.5    0.140   0.140   0.116         1
```

**Reading it:**

- **Under BAM the robot does NOT hold the pose for the hold curriculum's 1.0 s
  ceiling**: tilt is 0.9 deg at 0.1 s, 3.5 deg at 0.3 s, 7.3 deg at 0.5 s,
  11.6 deg at 0.7 s, and **23.2 deg mean / 42.4 deg worst at 1.0 s**, with x/y
  drift growing to 2.8 cm mean / 6.4 cm worst on an 18 cm plate. By AGENTS.md's
  3 s criterion it fails outright: 32/32 trials toppled (113-123 deg of tilt).
  The drift is monotonic, not oscillatory — this is a slow topple, not a wobble.
  Identical at all three `z0` (the plate height does not enter the pose's
  stability).
- **BAM matters, exactly as the fix-wave brief guessed**: the XML-PD row
  (dt 0.002, the mode every earlier table used) is ~4x worse at 1.0 s — 98.5
  deg mean, 31/32 already fallen — against BAM's 23.2 deg. The reviewer's
  open-loop XML-PD numbers were pessimistic, but not pessimistic enough to
  change the conclusion.
- **It is NOT the plate.** The `--settle-on-floor` control (robot on the
  terrain, plate parked at `BACKFLIP_GONE_POS`) drifts identically: 0.9 / 3.3 /
  6.9 / 11.2 / 23.6 deg. Open-loop HOME is simply not a passive equilibrium for
  this robot, on the plate or off it. Every other microduck env shares that
  property; their policies actively balance, which is what `ready_stance` is
  paying for here, and a closed-loop policy is not measured by this test.
- **The honest conclusion about `HOLD_RANGE`**: nothing here says the plate
  perch is unusually bad, but nothing here supports the premise that "standing
  still on the hands" is a free, passive state the policy merely has to avoid
  disturbing. At the curriculum's 1.0 s ceiling the pose has decayed by ~23 deg
  open-loop, so the policy must be actively balancing through most of every
  long hold, and the launch condition at `t_hold = 1.0 s` is whatever that
  active balancing leaves — not the spawn pose. If the flip turns out to be
  sensitive to `tilt0` (and the drift sweep above says it is: 4.9 deg of tilt
  costs 30-115 deg of rotation), the cheap mitigation is to keep `HOLD_RANGE`
  short rather than to widen it to 1.0 s.

## Reproducing everything in this section

```bash
# head-to-head at the current DR box (per z0 in 0.10 0.15 0.20)
uv run python scripts/backflip_envelope.py --bam --posture standing --z0 0.15 --vz 2.00,2.25 --w0 24,30 --tuck 1.0
uv run python scripts/backflip_envelope.py --bam --posture tucked   --z0 0.15 --vz 2.00,2.25 --w0 24,30 --tuck 1.0

# standing sweep, no policy action
uv run python scripts/backflip_envelope.py --bam --posture standing --z0 0.15 --hold 0.3 \
    --vz 2.0,2.5,3.0,3.5,4.0 --w0 3,6,9,12,15,18,21,24,27,30,33,36 --tuck 1.0

# standing sweep with the policy tucking at the flick
uv run python scripts/backflip_envelope.py --bam --posture standing --tuck-at-flick --z0 0.15 --hold 0.3 \
    --vz 2.0,2.5,3.0,3.5,4.0 --w0 3,6,9,12,15,18,21,24,27,30,33,36 --tuck 0.5,1.0

# longest launch ramp
uv run python scripts/backflip_envelope.py --bam --posture standing --tuck-at-flick --z0 0.15 --hold 0.3 --launch 0.15 \
    --vz 2.0,2.25,2.5,2.75,3.0 --w0 9,12,15,18,21,24 --tuck 0.75,1.0

# drift-is-not-the-cause scan (repeat with --hold 0.02 / 0.10 / 0.30)
uv run python scripts/backflip_envelope.py --bam --posture standing --z0 0.15 --hold 0.02 --vz 2.0 --w0 6,12,18,24,30 --tuck 1.0

# direction checks (one cell each)
uv run python scripts/backflip_envelope.py --bam --posture standing --tuck-at-flick --z0 0.10 --hold 0.3 --launch 0.15 \
    --check-direction --check-vz 2.5 --check-w0 21.0 --check-tuck 1.0

# settle test
uv run python scripts/backflip_envelope.py --settle --bam --z0 0.15 --settle-trials 32
uv run python scripts/backflip_envelope.py --settle --bam --settle-on-floor --settle-trials 32
uv run python scripts/backflip_envelope.py --settle --z0 0.15 --settle-trials 32          # XML-PD comparison
uv run python scripts/backflip_envelope.py --settle --bam --z0 0.15 --settle-noiseless
```


# Tucked hold — the posture switch, and the box measured from it

**Decision, and who made it.** The standing-spawn re-measurement above put the
question to the user: the env spawned the robot STANDING, the envelope had been
measured TUCKED, and from the standing spawn nothing closed a safe flip. The
user chose to switch the HOLD posture to TUCKED. This section is the
measurement that follows from that decision — the tucked resting height, the
whole-box re-verification from the ACTUAL new spawn, and the tucked settle
test. Everything here is CPU MuJoCo with **BAM actuators** at the training sim
timestep (0.005).

The probe now IMPORTS `TUCK_OVERRIDES`, `TUCK_FACTOR`, `TUCK_Z`, `STAND_Z`,
`PLATE_HALF_THICKNESS` and the four DR ranges from
`microduck_backflip_env_cfg.py` instead of copying them, so
`--posture tucked_env` cannot silently drift away from what the env does —
which is exactly how the previous envelope came to describe a posture the env
never produced.

## 1. TUCK_Z: the measured tucked resting height

`uv run python scripts/backflip_envelope.py --measure-tuck-z --bam --z0 0.15
--tuck 0.5,0.75,1.0`

Drop the tucked robot from four different offsets above the plate top, hold the
tuck ctrl, settle 3 s, read `trunk_z - plate_top`:

```
# measure-tuck-z: z0=0.15 duration=3.0s dt=0.005 bam=True   (TUCK_Z in the script = 0.029)
 tuck  offset   h@0.1   h@1.0   h@end  tilt@0.1  tilt@1.0  tilt@end  xy@end
 0.50   0.020  0.0280  0.0287  0.0285      11.1      11.8      12.5  0.0060
 0.50   0.026  0.0283  0.0286  0.0286      11.1      12.0      12.4  0.0064
 0.50   0.029  0.0283  0.0285  0.0285      11.4      12.5      12.5  0.0064
 0.50   0.032  0.0283  0.0286  0.0286      11.8      12.3      12.4  0.0066
 0.75   0.020  0.0282  0.0287  0.0287      15.7      13.8      14.0  0.0073
 0.75   0.026  0.0283  0.0286  0.0286      14.1      13.8      14.1  0.0081
 0.75   0.029  0.0284  0.0286  0.0286      13.7      13.8      13.9  0.0085
 0.75   0.032  0.0286  0.0287  0.0286      13.8      13.9      14.0  0.0091
 1.00   0.020  0.0259  0.0258  0.0259      13.4      15.0      14.9  0.0113
 1.00   0.026  0.0265  0.0262  0.0256      14.0      14.7      15.1  0.0115
 1.00   0.029  0.0264  0.0256  0.0259      13.9      15.9      15.4  0.0118
 1.00   0.032  0.0261  0.0256  0.0256      14.2      15.0      15.3  0.0118
```

**TUCK_Z = 0.029 m.** The pose rests at **0.0286 m** above the plate top for
tuck factors 0.5-0.75 and **0.0257 m** at full tuck, converging from every
spawn offset tried, within 0.1-0.3 s. Two things this measurement is for:

- It is **not** `STAND_Z`. The standing trunk sits at 0.115 m; the tucked one
  at 0.029 m. Carrying a height across poses is precisely the failure
  AGENTS.md records as having cost days, so the constant is measured and
  carries the date.
- It is a **floor** as much as a target: spawning much below ~0.010 m jams the
  folded legs into the plate and the contact solver ejects the robot clean off
  it (measured: `h` goes negative and tilt jumps to 80-155 deg).

## 2. Settle test on the TUCKED pose (AGENTS.md requirement, re-run)

Same protocol as the standing settle test above — noisy inits matching the
env's resets exactly (target pose + U(-0.05, 0.05) rad per joint, x/y ±0.01 m,
roll/pitch ±0.02 rad, yaw ±0.05 rad, zero root velocity), plate rewritten at
`z0` every step, TILT and x/y drift reported, `n_fallen` counting trials past
45 deg.

**Read the tilt against the pose's own equilibrium.** A folded robot rests with
its trunk pitched 12-15 deg *by geometry*. "Tilt 14.0 deg, unchanged from 0.1 s
to 3 s" is a settled tuck, not a falling one; what matters is whether it MOVES.

```
# settle: posture=tucked_env tuck=0.75 on plate z0=0.1 trials=32 noisy=True duration=3.0s dt=0.005 bam=True
  t[s]  tilt_mean  tilt_max  xy_mean  xy_max  z_mean  n_fallen
  0.10       13.7      14.4    0.008   0.009   0.138         0
  0.30       14.3      14.6    0.008   0.009   0.139         0
  0.50       14.1      14.5    0.008   0.009   0.139         0
  0.70       14.0      14.5    0.008   0.009   0.139         0
  1.00       13.9      14.4    0.008   0.010   0.139         0
  3.00       14.0      14.5    0.008   0.011   0.139         0
# settle: posture=tucked_env tuck=0.75 on plate z0=0.15 trials=32 noisy=True duration=3.0s dt=0.005 bam=True
  t[s]  tilt_mean  tilt_max  xy_mean  xy_max  z_mean  n_fallen
  0.10       13.7      14.4    0.008   0.009   0.188         0
  0.30       14.3      14.6    0.008   0.009   0.189         0
  0.50       14.1      14.5    0.008   0.009   0.189         0
  0.70       14.0      14.5    0.008   0.009   0.189         0
  1.00       13.9      14.4    0.008   0.010   0.189         0
  3.00       14.0      14.5    0.008   0.011   0.189         0
# settle: posture=tucked_env tuck=0.75 on plate z0=0.2 trials=32 noisy=True duration=3.0s dt=0.005 bam=True
  t[s]  tilt_mean  tilt_max  xy_mean  xy_max  z_mean  n_fallen
  0.10       13.7      14.4    0.008   0.009   0.238         0
  0.30       14.3      14.6    0.008   0.009   0.239         0
  0.50       14.1      14.5    0.008   0.009   0.239         0
  0.70       14.0      14.5    0.008   0.009   0.239         0
  1.00       13.9      14.4    0.008   0.010   0.239         0
  3.00       14.0      14.5    0.008   0.011   0.239         0
# settle: posture=tucked_env tuck=0.5 on plate z0=0.15 trials=32 noisy=True duration=3.0s dt=0.005 bam=True
  t[s]  tilt_mean  tilt_max  xy_mean  xy_max  z_mean  n_fallen
  0.10       11.0      11.5    0.006   0.007   0.188         0
  0.30       12.4      12.6    0.006   0.007   0.188         0
  0.50       12.2      12.5    0.006   0.007   0.189         0
  0.70       12.2      12.5    0.006   0.007   0.189         0
  1.00       12.3      12.5    0.006   0.007   0.189         0
  3.00       12.3      12.5    0.006   0.007   0.189         0
# settle: posture=tucked_env tuck=1.0 on plate z0=0.15 trials=32 noisy=True duration=3.0s dt=0.005 bam=True
  t[s]  tilt_mean  tilt_max  xy_mean  xy_max  z_mean  n_fallen
  0.10       13.2      14.2    0.007   0.008   0.186         0
  0.30       15.0      15.9    0.008   0.009   0.186         0
  0.50       15.3      16.2    0.008   0.009   0.186         0
  0.70       15.2      16.1    0.008   0.010   0.186         0
  1.00       15.1      17.4    0.009   0.011   0.186         0
  3.00       15.2      16.4    0.011   0.015   0.186         0
# settle: posture=tucked_env tuck=0.75 on plate z0=0.15 trials=32 noisy=True duration=3.0s dt=0.002 bam=False
  t[s]  tilt_mean  tilt_max  xy_mean  xy_max  z_mean  n_fallen
  0.10       13.6      14.3    0.008   0.009   0.189         0
  0.30       14.6      16.2    0.009   0.010   0.188         0
  0.50       14.1      14.5    0.008   0.009   0.188         0
  0.70       14.0      14.5    0.008   0.009   0.188         0
  1.00       14.0      14.5    0.008   0.009   0.188         0
  3.00       14.0      14.4    0.009   0.010   0.188         0
```

**Verdict: the tucked hold is a genuine stable equilibrium.**

| | standing (superseded) | tucked (`TUCK_FACTOR` 0.75) |
|---|---|---|
| tilt @ 0.1 s | 0.9 deg | 13.7 deg |
| tilt @ 0.3 s | 3.5 deg | 14.3 deg |
| **tilt @ 0.5 s** | **7.3 deg** | **14.1 deg** |
| tilt @ 0.7 s | 11.6 deg | 14.0 deg |
| **tilt @ 1.0 s** | **23.2 deg (max 42.4)** | **13.9 deg (max 14.4)** |
| tilt @ 3.0 s | 123.4 deg | 14.0 deg (max 14.5) |
| fallen by 3 s | **32/32** | **0/32** |
| x/y drift @ 3 s | 0.140 m | 0.008 m |

The tucked number is flat, not merely smaller: it reaches its equilibrium by
0.1 s and stays there for 3 s. Identical at `z0` = 0.10 / 0.15 / 0.20 (plate
height does not enter the pose's stability), and the XML-PD run agrees within
0.5 deg — unlike the standing pose, this one does not depend on the actuator
model at all, which is what a passively stable configuration looks like.

Tuck depth barely matters for stability: 0.5 / 0.75 / 1.0 rest at 12.3 / 14.0 /
15.3 deg with 3 s drifts of 7 / 11 / 15 mm and 0/32 falls each. 0.75 was chosen
as the spawn depth for being interior on both the stability and the closure
axes.

**Consequence for `HOLD_RANGE`: the curriculum's 1.0 s ceiling stands**, with
3x margin — the pose holds 3 s, and the box closes at `t_hold` = 1.0 s as well
as 0.1 s (below). No cap was needed.

## 3. Whole-box verification from the ACTUAL new spawn

`uv run python scripts/backflip_envelope.py --box-check --bam`

The acceptance rule is the one the previous box failed. Not "the best corner
closes" — **every** sampled combination of the CORNERS AND MIDPOINTS of `z0`,
`vz`, `w0` and `t_launch`, crossed with the hold extremes (0.1 s and the
curriculum's 1.0 s) and tuck depths 0.5 / 0.75 / 1.0, must close >= 360 deg
BACKWARD at <= 2.6 m/s. Tuck depth is swept even though the spawn is fixed at
0.75, because the policy can deepen or open the tuck during HOLD and LAUNCH.

```
# box-check posture=tucked_env bam=True dt=0.005
#   z0 (0.1, 0.2) vz (2.0, 2.1) w0 (21.0, 23.0) launch (0.12, 0.14)
#   hold (0.1, 1.0) tuck (0.5, 0.75, 1.0)
  486 cells | min rot = 393.6 deg | max landing = 2.51 m/s | short of 360: 0 | over 2.6 m/s: 0
  worst by rotation:
    rot=  393.6 land= 1.79 apex=0.381 tilt0= 13.9  z0=0.100 vz=2.000 w0=23.00 launch=0.140 hold=0.10 tuck=1.00
    rot=  398.8 land= 2.07 apex=0.424 tilt0= 15.9  z0=0.100 vz=2.000 w0=23.00 launch=0.140 hold=1.00 tuck=1.00
    rot=  400.9 land= 2.32 apex=0.458 tilt0= 15.9  z0=0.100 vz=2.000 w0=21.00 launch=0.140 hold=1.00 tuck=1.00
    rot=  403.6 land= 1.87 apex=0.395 tilt0= 13.9  z0=0.100 vz=2.000 w0=22.00 launch=0.140 hold=0.10 tuck=1.00
    rot=  404.6 land= 1.94 apex=0.404 tilt0= 13.9  z0=0.100 vz=2.000 w0=21.00 launch=0.140 hold=0.10 tuck=1.00
  worst by landing speed:
    rot=  475.7 land= 2.51 apex=0.621 tilt0= 15.9  z0=0.200 vz=2.100 w0=21.00 launch=0.120 hold=1.00 tuck=1.00
    rot=  473.9 land= 2.50 apex=0.598 tilt0= 12.5  z0=0.200 vz=2.100 w0=21.00 launch=0.120 hold=1.00 tuck=0.50
    rot=  465.0 land= 2.49 apex=0.585 tilt0= 12.5  z0=0.200 vz=2.050 w0=21.00 launch=0.120 hold=1.00 tuck=0.50
    rot=  464.2 land= 2.49 apex=0.585 tilt0= 11.4  z0=0.200 vz=2.050 w0=21.00 launch=0.120 hold=0.10 tuck=0.50
    rot=  466.9 land= 2.48 apex=0.594 tilt0= 15.9  z0=0.200 vz=2.000 w0=21.00 launch=0.120 hold=1.00 tuck=1.00
  RESULT: PASS — whole box closes 360 deg under 2.6 m/s
```

**486 cells, 0 short of 360 deg, 0 over 2.6 m/s. Worst cell 393.6 deg
(33.6 deg of margin) at 1.79 m/s; worst landing 2.51 m/s (0.09 m/s of margin)
at 475.7 deg.**

### The old box does NOT pass this rule from the real spawn

For comparison, the previously configured box
(`vz` [2.00, 2.25], `w0` [24, 30], `t_launch` [0.08, 0.15]) run from the same
tucked_env spawn over its corners x hold x launch, 324 cells:

```
min rot = 278.8 deg   max land = 3.97 m/s
closed >= 360: 308/324    land <= 2.6: 234/324    BOTH: 218/324
worst by rotation:
  rot=278.8 land=1.93 z0=0.10 vz=2.00 w0=30 tuck=0.5 hold=0.4 launch=0.15
  rot=278.9 land=1.94 z0=0.10 vz=2.00 w0=30 tuck=0.5 hold=0.1 launch=0.15
  rot=283.1 land=2.19 z0=0.10 vz=2.00 w0=30 tuck=1.0 hold=0.1 launch=0.15
worst by landing speed:
  rot=627.9 land=3.97 z0=0.20 vz=2.25 w0=30 tuck=1.0 hold=0.4 launch=0.08
  rot=655.1 land=3.92 z0=0.20 vz=2.25 w0=30 tuck=1.0 hold=0.1 launch=0.08
  rot=627.8 land=3.84 z0=0.20 vz=2.25 w0=24 tuck=0.75 hold=0.4 launch=0.08
```

Two independent failures, and both are instructive:

- **A SHORT flick is the violent one.** `t_launch` = 0.08 lands at up to
  3.97 m/s. The operator's flick duration was being treated as free DR; it is
  not. The low end moved to 0.12.
- **`w0` = 30 runs out of AIRTIME, not spin.** A harder flick trades apex for
  rotation rate: at `z0` = 0.10, `vz` = 2.00, `t_launch` = 0.15 it reaches only
  278.8 deg with an apex of 0.285 m. That is why the ceiling came DOWN to 23,
  not up.

### The axis scans behind the chosen ranges

Per `(t_launch, vz, w0)`, worst rotation and worst landing over
`z0` ∈ {0.10, 0.15, 0.20} × tuck ∈ {0.5, 1.0} × hold ∈ {0.1, 1.0}:

```
  lau    vz    w0   minrot  maxland  OK  worstrot  worstland
0.120  2.00  21.0    431.8     2.48  YES  (0.1, 0.5, 1.0)  (0.2, 1.0, 1.0)
0.120  2.00  22.5    435.6     2.31  YES  (0.1, 1.0, 1.0)  (0.2, 0.5, 1.0)
0.120  2.00  24.0    437.3     2.21  YES  (0.1, 0.5, 0.1)  (0.2, 0.5, 0.1)
0.120  2.00  25.5    431.4     2.10  YES  (0.1, 1.0, 0.1)  (0.2, 0.5, 1.0)
0.120  2.10  21.0    443.0     2.51  YES  (0.1, 0.5, 0.1)  (0.2, 1.0, 1.0)
0.120  2.10  22.5    447.7     2.39  YES  (0.1, 1.0, 1.0)  (0.2, 0.5, 1.0)
0.120  2.10  24.0    445.1     2.34  YES  (0.1, 1.0, 1.0)  (0.2, 0.5, 1.0)
0.120  2.10  25.5    447.3     2.34  YES  (0.1, 1.0, 0.1)  (0.2, 0.5, 1.0)
0.120  2.20  21.0    437.8     2.59  YES  (0.1, 1.0, 1.0)  (0.2, 0.5, 0.1)
0.120  2.20  22.5    459.8     2.58  YES  (0.1, 1.0, 1.0)  (0.2, 0.5, 0.1)
0.120  2.20  24.0    453.9     2.65  no   (0.1, 1.0, 1.0)  (0.2, 0.5, 0.1)
0.120  2.20  25.5    460.4     2.61  no   (0.1, 1.0, 0.1)  (0.2, 0.5, 1.0)
0.135  2.00  21.0    412.1     2.43  YES  (0.1, 1.0, 1.0)  (0.2, 1.0, 1.0)
0.135  2.00  22.5    401.5     2.25  YES  (0.1, 1.0, 1.0)  (0.2, 1.0, 1.0)
0.135  2.00  24.0    398.8     2.12  YES  (0.1, 1.0, 0.1)  (0.2, 1.0, 1.0)
0.135  2.00  25.5    386.2     1.96  YES  (0.1, 1.0, 0.1)  (0.2, 1.0, 1.0)
0.135  2.10  21.0    429.7     2.43  YES  (0.1, 1.0, 0.1)  (0.2, 1.0, 1.0)
0.135  2.10  22.5    432.7     2.27  YES  (0.1, 1.0, 0.1)  (0.2, 0.5, 0.1)
0.135  2.10  24.0    426.9     2.13  YES  (0.1, 1.0, 0.1)  (0.2, 0.5, 0.1)
0.135  2.10  25.5    415.9     2.07  YES  (0.1, 1.0, 0.1)  (0.2, 0.5, 1.0)
0.135  2.20  21.0    441.8     2.47  YES  (0.1, 0.5, 0.1)  (0.2, 1.0, 1.0)
0.135  2.20  22.5    444.1     2.43  YES  (0.1, 1.0, 0.1)  (0.2, 0.5, 0.1)
0.135  2.20  24.0    435.1     2.34  YES  (0.1, 1.0, 0.1)  (0.2, 0.5, 0.1)
0.135  2.20  25.5    426.4     2.25  YES  (0.1, 1.0, 0.1)  (0.2, 0.5, 1.0)
0.150  2.00  21.0    380.8     2.35  YES  (0.1, 1.0, 1.0)  (0.2, 1.0, 1.0)
0.150  2.00  22.5    372.1     2.22  YES  (0.1, 1.0, 0.1)  (0.2, 1.0, 1.0)
0.150  2.00  24.0    362.2     2.08  YES  (0.1, 1.0, 0.1)  (0.2, 1.0, 1.0)
0.150  2.00  25.5    345.6     2.04  no   (0.1, 1.0, 0.1)  (0.1, 1.0, 1.0)
0.150  2.10  21.0    407.9     2.35  YES  (0.1, 1.0, 1.0)  (0.2, 1.0, 1.0)
0.150  2.10  22.5    394.7     2.21  YES  (0.1, 1.0, 0.1)  (0.2, 1.0, 1.0)
0.150  2.10  24.0    392.0     2.06  YES  (0.1, 1.0, 0.1)  (0.2, 1.0, 1.0)
0.150  2.10  25.5    367.6     1.96  YES  (0.1, 1.0, 0.1)  (0.2, 1.0, 1.0)
0.150  2.20  21.0    426.7     2.36  YES  (0.1, 1.0, 1.0)  (0.2, 1.0, 1.0)
0.150  2.20  22.5    425.3     2.27  YES  (0.1, 1.0, 0.1)  (0.2, 0.5, 1.0)
0.150  2.20  24.0    413.9     2.15  YES  (0.1, 1.0, 1.0)  (0.2, 0.5, 1.0)
0.150  2.20  25.5    396.4     1.96  YES  (0.1, 1.0, 0.1)  (0.2, 0.5, 1.0)
```

The chosen rectangle is the largest one in that surface with margin on both
constraints. The failures just outside it, each measured:

| widening | what breaks |
|---|---|
| `vz` > 2.10 | `z0`=0.20 corner lands at 2.65-2.70 m/s (over the limit) |
| `w0` > 23 | `t_launch`=0.14 drops to 372 deg; `w0`=25 at `t_launch`=0.15 to 345.6 deg |
| `t_launch` < 0.12 | landings to 3.97 m/s |
| `t_launch` > 0.14 | rotation margin thins: 362 deg at `t_launch`=0.15, `vz`=2.00, `w0`=24 |
| `vz` < 2.00 | the flip stops closing (the vz cliff measured in the earlier sections) |

## 4. Direction checks — 20/20 backward

Every corner of the recommended box plus its two worst cells and both tuck
extremes, traced by the robot's own local +z axis in world coordinates at
90 deg of accumulated rotation (`-x` = leaning backward-and-up = a real
backflip; `+x` = face-down = a forward roll):

```
   z0    vz    w0   lau  hold  tuck     rot  land  verdict
 0.10  2.00  21.0 0.120  0.40  0.75   451.0  2.12  BACKWARD ok  (+z=(-0.675,0.004,-0.738))
 0.10  2.00  21.0 0.140  0.40  0.75   428.6  2.01  BACKWARD ok  (+z=(-0.490,-0.023,-0.871))
 0.10  2.00  23.0 0.120  0.40  0.75   453.8  1.88  BACKWARD ok  (+z=(-0.530,-0.015,-0.848))
 0.10  2.00  23.0 0.140  0.40  0.75   425.4  1.81  BACKWARD ok  (+z=(-0.334,0.001,-0.943))
 0.10  2.10  21.0 0.120  0.40  0.75   466.0  2.14  BACKWARD ok  (+z=(-0.678,0.004,-0.735))
 0.10  2.10  21.0 0.140  0.40  0.75   444.2  2.00  BACKWARD ok  (+z=(-0.413,-0.018,-0.910))
 0.10  2.10  23.0 0.120  0.40  0.75   468.7  1.91  BACKWARD ok  (+z=(-0.548,0.007,-0.837))
 0.10  2.10  23.0 0.140  0.40  0.75   438.8  1.77  BACKWARD ok  (+z=(-0.332,-0.013,-0.943))
 0.20  2.00  21.0 0.120  0.40  0.75   475.4  2.34  BACKWARD ok  (+z=(-0.675,0.004,-0.738))
 0.20  2.00  21.0 0.140  0.40  0.75   454.4  2.19  BACKWARD ok  (+z=(-0.490,-0.023,-0.871))
 0.20  2.00  23.0 0.120  0.40  0.75   480.4  2.16  BACKWARD ok  (+z=(-0.530,-0.015,-0.848))
 0.20  2.00  23.0 0.140  0.40  0.75   453.3  2.02  BACKWARD ok  (+z=(-0.334,0.001,-0.943))
 0.20  2.10  21.0 0.120  0.40  0.75   490.5  2.42  BACKWARD ok  (+z=(-0.678,0.004,-0.735))
 0.20  2.10  21.0 0.140  0.40  0.75   470.0  2.26  BACKWARD ok  (+z=(-0.413,-0.018,-0.910))
 0.20  2.10  23.0 0.120  0.40  0.75   500.5  2.34  BACKWARD ok  (+z=(-0.548,0.007,-0.837))
 0.20  2.10  23.0 0.140  0.40  0.75   466.9  2.06  BACKWARD ok  (+z=(-0.332,-0.013,-0.943))
 0.10  2.00  23.0 0.140  0.10  1.00   393.6  1.79  BACKWARD ok  (+z=(-0.141,0.076,-0.987))
 0.20  2.10  21.0 0.120  1.00  1.00   475.7  2.51  BACKWARD ok  (+z=(-0.684,0.043,-0.728))
 0.15  2.05  22.0 0.130  0.55  0.50   449.6  2.20  BACKWARD ok  (+z=(-0.480,0.005,-0.877))
 0.15  2.05  22.0 0.130  0.55  1.00   460.1  2.08  BACKWARD ok  (+z=(-0.534,-0.023,-0.845))

20 cells checked, 0 not confirmed backward
```

## 5. The `z0` DR tail: 0.225, and why not 0.25 or 0.30

The spec asked for an operator tail to 0.30 m. Measured at the box's
worst-landing corner (`vz`=2.10, `w0`=21, `t_launch`=0.12, hold=1.0, tuck=1.0),
landing speed rises monotonically with launch height:

```
worst-landing corner of the box (vz=2.10 w0=21 launch=0.12 hold=1.0 tuck=1.0)
    z0     rot   land   apex
 0.100   453.6   2.32  0.521
 0.150   466.9   2.42  0.571
 0.200   475.7   2.51  0.621
 0.225   480.1   2.56  0.646
 0.250   484.5   2.61  0.671
 0.275   488.9   2.67  0.696
 0.300   497.7   2.80  0.721
```

0.250 m lands at **2.61 m/s** — already past the ~2.6 m/s damage threshold —
and 0.300 m at **2.80 m/s**. **0.225 m is the last height that stays under it**
(2.56 m/s, 0.04 m/s of margin), so that is where the curriculum's tail stops.
This supersedes the previous round's 0.25 m, which was a conservative reading
of a rationale rather than a measurement; the measurement now exists.

## 6. `MAX_PAID_RATE` re-checked against the new box

Peak and mean BACKWARD rotation rate while airborne, over 64 corner cells of
the new box:

```
   peak    mean     rot   land  z0    vz    w0   tuck hold  lau
  23.03   19.60   449.1   1.92  0.2 2.1 23.0 1.0 0.1 0.14
  23.03   20.13   415.2   1.71  0.1 2.1 23.0 1.0 0.1 0.14
  23.03   19.90   456.2   1.69  0.1 2.1 23.0 1.0 0.1 0.12
  23.03   15.64   483.8   2.01  0.2 2.1 23.0 1.0 0.1 0.12
  22.98   18.43   470.0   1.91  0.2 2.0 23.0 1.0 0.1 0.12
  22.98   20.03   441.8   1.68  0.1 2.0 23.0 1.0 0.1 0.12
  22.86   18.56   451.9   2.16  0.2 2.1 23.0 0.5 0.1 0.14
  22.86   19.01   424.7   1.87  0.1 2.1 23.0 0.5 0.1 0.14
...
  17.79   16.14   434.7   2.31  0.1 2.0 21.0 1.0 1.0 0.12
  17.79   16.14   466.9   2.48  0.2 2.0 21.0 1.0 1.0 0.12
  17.78   15.97   475.7   2.51  0.2 2.1 21.0 1.0 1.0 0.12
  17.78   15.99   453.6   2.32  0.1 2.1 21.0 1.0 1.0 0.12

max peak = 23.03 rad/s   max mean = 20.13 rad/s   over 64 cells
```

Peak **23.03 rad/s**, mean over a closing flip **15.6-20.1 rad/s**.
`MAX_PAID_RATE = 25.0` rad/s therefore forfeits nothing a real flip in this box
needs, while still pricing spin above the measured envelope. Unchanged.

## 7. Reproducing this section

```bash
# the measured tucked resting height
uv run python scripts/backflip_envelope.py --measure-tuck-z --bam --z0 0.15 --tuck 0.5,0.75,1.0

# settle test on the tucked pose (repeat with --z0 0.10 / 0.20, --tuck 0.5 / 1.0,
# and without --bam for the XML-PD comparison)
uv run python scripts/backflip_envelope.py --settle --bam --z0 0.15 --settle-trials 32 --tuck 0.75

# whole-box verification of the cfg's ranges from the env's actual spawn
uv run python scripts/backflip_envelope.py --box-check --bam

# a single direction check inside the box
uv run python scripts/backflip_envelope.py --bam --posture tucked_env --z0 0.10 \
    --hold 0.4 --launch 0.12 --check-direction --check-vz 2.00 --check-w0 21.0 --check-tuck 0.75

# the z0 tail scan and the peak-rate scan are two-line drivers over run_cell();
# the tables above carry their exact parameters.
```


# Flop audit, and two corrections to the tucked-hold numbers

Posture-switch re-review follow-ups. Same setup throughout: CPU MuJoCo, BAM
actuators, training sim timestep 0.005.

## 1. The flop audit (new probe mode `--flop-audit`) — and the hole it found

AGENTS.md: *"audit each positive term against every stable flop (on back /
face / side): if flopping keeps most of the stack, the policy will flop."*
Nobody had run it for `backflip_ready_stance`. Run now, it fails: the
side-lying tuck **outscored the upright one**.

The mode settles the tucked robot from six orientations, at five spawn
clearances each, and reports the WORST (highest-scoring) basin per
orientation — a flopped pose settles into different basins depending on how it
is dropped, and an audit that tries one clearance measures whichever one it
happens to hit. `pre` is `pose x height`, what the term paid before the fix;
`TOTAL` includes the reinstated upright factor.

`z0 = 0.15`:

```
# flop-audit: z0=0.15 tuck=0.75 duration=3.0s dt=0.005 bam=True
 orientation   clr    tilt   drift  trunk_z    pose  height    pre  upright   TOTAL  on?
     upright 0.005    14.1   0.009    0.189   0.948   1.000  0.948    1.000   0.948  yes
   side_left 0.015   102.8   0.016    0.191   0.997   0.994  0.991    0.000   0.000  yes
  side_right 0.015   102.8   0.016    0.191   0.997   0.994  0.991    0.000   0.000  yes
   face_down 0.015   101.4   0.019    0.184   0.783   0.972  0.762    0.000   0.000  yes
     on_back 0.015    91.0   0.002    0.202   0.982   0.829  0.814    0.000   0.000  yes
    inverted 0.015   141.7   0.025    0.165   0.986   0.531  0.524    0.000   0.000  yes
  upright: pre=0.948 TOTAL=0.948   best flop: pre=0.991 TOTAL=0.000
  WITHOUT the upright factor: a flop would pay MORE (0.991 vs 0.948)
  RESULT: PASS - upright wins (0.948 vs 0.000)
```

`z0 = 0.10` (the lowest launch height) and `z0 = 0.21` (the curriculum
ceiling), for completeness — the basins are the same, the trunk heights shift
with `z0`:

```
# flop-audit: z0=0.1 tuck=0.75 duration=3.0s dt=0.005 bam=True
 orientation   clr    tilt   drift  trunk_z    pose  height    pre  upright   TOTAL  on?
     upright 0.005    14.1   0.009    0.139   0.948   1.000  0.948    1.000   0.948  yes
   side_left 0.015   102.8   0.016    0.141   0.997   0.994  0.991    0.000   0.000  yes
  side_right 0.015   102.8   0.016    0.141   0.997   0.994  0.991    0.000   0.000  yes
   face_down 0.045   115.7   0.043    0.150   0.954   0.883  0.843    0.000   0.000  yes
     on_back 0.015    91.0   0.002    0.152   0.982   0.829  0.814    0.000   0.000  yes
    inverted 0.005   172.7   0.015    0.134   0.991   0.972  0.964    0.000   0.000  yes
  upright: pre=0.948 TOTAL=0.948   best flop: pre=0.991 TOTAL=0.000
  WITHOUT the upright factor: a flop would pay MORE (0.991 vs 0.948)
  RESULT: PASS - upright wins (0.948 vs 0.000)
```

```
# flop-audit: z0=0.21 tuck=0.75 duration=3.0s dt=0.005 bam=True
 orientation   clr    tilt   drift  trunk_z    pose  height    pre  upright   TOTAL  on?
     upright 0.005    14.1   0.009    0.249   0.948   1.000  0.948    1.000   0.948  yes
   side_left 0.015   102.8   0.016    0.251   0.997   0.994  0.991    0.000   0.000  yes
  side_right 0.015   102.8   0.016    0.251   0.997   0.994  0.991    0.000   0.000  yes
   face_down 0.015   101.4   0.019    0.244   0.783   0.972  0.762    0.000   0.000  yes
     on_back 0.015    91.0   0.002    0.262   0.982   0.829  0.814    0.000   0.000  yes
    inverted 0.015   141.7   0.025    0.225   0.985   0.531  0.523    0.000   0.000  yes
  upright: pre=0.948 TOTAL=0.948   best flop: pre=0.991 TOTAL=0.000
  WITHOUT the upright factor: a flop would pay MORE (0.991 vs 0.948)
  RESULT: PASS - upright wins (0.948 vs 0.000)
```

**What was wrong.** `pose` and `height` cannot tell an upright tuck from an
inverted one. Measured at `z0=0.15`:

| basin | settled tilt | drift @3 s | on plate? | pose | height | `pre` (old term) | TOTAL (fixed) |
|---|---|---|---|---|---|---|---|
| **upright tuck** (the spawn) | 14.1 deg | 0.009 m | yes | 0.948 | 1.000 | **0.948** | **0.948** |
| **side-lying** | 102.8 deg | 0.016 m | **yes** | **0.997** | 0.994 | **0.991** | **0.000** |
| on its back | 91.0 deg | 0.002 m | yes | 0.982 | 0.829 | 0.814 | 0.000 |
| face down | 101.4 deg | 0.019 m | yes | 0.783 | 0.972 | 0.762 | 0.000 |
| inverted | 141.7 deg | 0.025 m | yes | 0.986 | 0.531 | 0.524 | 0.000 |
| inverted (at `z0`=0.10) | 172.7 deg | 0.015 m | yes | 0.991 | 0.972 | 0.964 | 0.000 |

The side-lying tuck paid **0.991 against the upright tuck's 0.948** — a 4%
premium for flopping. And it is worse than a tie:

- it is a **passively stable on-plate basin**: 3 s at 102.8 deg with 16 mm of
  drift, and it does not fall off the plate;
- it needs **no active balancing** at all, unlike the upright tuck;
- its `pose` factor is actually **higher** (0.997 vs 0.948) because its joints
  are not load-sagged;
- it costs less `action_rate`.

And it bit hardest exactly where it matters: during discovery `flip_progress`
pays ~0 because the flip does not exist yet, so `ready_stance` is the only
paying term — and the launch that follows a side-lying hold is a SIDE flip,
the failure the cfg narrows yaw scatter to +-0.05 specifically to prevent.

**The fix.** Reinstate an upright factor as a WIDE smoothstep on trunk tilt:
1 below 40 deg, 0 above 70 deg. The original reason for dropping it (the
tuck's own equilibrium is a trunk pitched 12-15 deg, so an upright term would
fight the pose) is valid for a tight Gaussian and not for this: at 14 deg the
gate is exactly 1.000, so holding the correct pose costs **zero**, while
102.8 deg is hard-zeroed. `clamp(cos(tilt), 0)` would also close the hole but
charges 3% for doing the right thing. `tilt_full_deg` must stay above the
tuck's measured resting tilt plus a margin.

Every flop basin now scores exactly 0.000 at every `z0`, and the upright tuck
is unchanged at 0.948.

## 2. Correction: the `z0` tail's margin was 0.01 m/s, not 0.04

The previous section quoted 2.56 m/s at `z0=0.225` and called it 0.04 m/s of
margin. That number came from a **single-corner 1-D scan** — the exact method
the whole-box rule in this document rejects. Measured whole-box instead (162
cells of `vz` x `w0` x `t_launch` x hold x tuck at each height):

```
    z0  cells  min_rot  max_land  worst-landing cell
 0.200    162    430.8      2.51  vz=2.1 w0=21.0 launch=0.12 hold=1.0 tuck=1.0 (rot=475.7)
 0.205    162    430.8      2.53  vz=2.1 w0=21.0 launch=0.12 hold=0.1 tuck=0.5 (rot=471.2)
 0.210    162    436.1      2.53  vz=2.1 w0=21.0 launch=0.12 hold=0.1 tuck=0.5 (rot=471.2)
 0.215    162    436.1      2.57  vz=2.1 w0=21.0 launch=0.12 hold=1.0 tuck=0.5 (rot=478.7)
 0.220    162    436.1      2.57  vz=2.1 w0=21.0 launch=0.12 hold=1.0 tuck=0.5 (rot=478.7)
 0.225    162    436.1      2.59  vz=2.1 w0=21.0 launch=0.12 hold=0.1 tuck=0.5 (rot=476.0)
```

| z0 ceiling | whole-box worst landing | margin under 2.6 m/s |
|---|---|---|
| 0.200 | 2.51 m/s | 0.09 |
| 0.205 | 2.53 | 0.07 |
| **0.210** | **2.53** | **0.07** |
| 0.215 | 2.57 | 0.03 |
| 0.220 | 2.57 | 0.03 |
| 0.225 | **2.59** | **0.01** |

So `z0 = 0.225` has essentially no margin, and the tail is cut to
**`Z0_CURRICULUM_MAX = 0.21`** (0.07 m/s). Note what that means: the tail is
now **1 cm wide**. The spec's operator tail to 0.30 m is simply not available
at this landing limit; if the user would rather not carry a curriculum stage
for 1 cm, deleting the stage and living with `Z0_RANGE` (worst landing
2.51 m/s) is the honest alternative.

**Structural fix so this cannot recur:** `--box-check` read `Z0_RANGE` from the
cfg and therefore never saw the curriculum's widened ceiling — the state the
env actually trains into after iteration 3000. It now sweeps
`Z0_CURRICULUM_MAX` as well (648 cells instead of 486):

```
# box-check posture=tucked_env bam=True dt=0.005
#   z0 (0.1, 0.2) (curriculum ceiling 0.21) vz (2.0, 2.1) w0 (21.0, 23.0) launch (0.12, 0.14)
#   z0 grid (0.1, 0.15000000000000002, 0.2, 0.21)
#   hold (0.1, 1.0) tuck (0.5, 0.75, 1.0)
  648 cells | min rot = 393.6 deg | max landing = 2.53 m/s | short of 360: 0 | over 2.6 m/s: 0
  worst by rotation:
    rot=  393.6 land= 1.79 apex=0.381 tilt0= 13.9  z0=0.100 vz=2.000 w0=23.00 launch=0.140 hold=0.10 tuck=1.00
    rot=  398.8 land= 2.07 apex=0.424 tilt0= 15.9  z0=0.100 vz=2.000 w0=23.00 launch=0.140 hold=1.00 tuck=1.00
    rot=  400.9 land= 2.32 apex=0.458 tilt0= 15.9  z0=0.100 vz=2.000 w0=21.00 launch=0.140 hold=1.00 tuck=1.00
    rot=  403.6 land= 1.87 apex=0.395 tilt0= 13.9  z0=0.100 vz=2.000 w0=22.00 launch=0.140 hold=0.10 tuck=1.00
    rot=  404.6 land= 1.94 apex=0.404 tilt0= 13.9  z0=0.100 vz=2.000 w0=21.00 launch=0.140 hold=0.10 tuck=1.00
  worst by landing speed:
    rot=  471.2 land= 2.53 apex=0.603 tilt0= 11.4  z0=0.210 vz=2.100 w0=21.00 launch=0.120 hold=0.10 tuck=0.50
    rot=  471.5 land= 2.52 apex=0.604 tilt0= 15.9  z0=0.210 vz=2.000 w0=21.00 launch=0.120 hold=1.00 tuck=1.00
    rot=  471.9 land= 2.51 apex=0.618 tilt0= 15.9  z0=0.210 vz=2.050 w0=21.00 launch=0.120 hold=1.00 tuck=1.00
    rot=  475.7 land= 2.51 apex=0.621 tilt0= 15.9  z0=0.200 vz=2.100 w0=21.00 launch=0.120 hold=1.00 tuck=1.00
    rot=  475.7 land= 2.51 apex=0.631 tilt0= 15.9  z0=0.210 vz=2.100 w0=21.00 launch=0.120 hold=1.00 tuck=1.00
  RESULT: PASS — whole box closes 360 deg under 2.6 m/s
```

Still PASS: min rotation 393.6 deg, worst landing 2.53 m/s, both at the
extremes of the widened box.

## 3. Correction: `arrival_damping`'s ceiling encoded the STANDING hold

`height_full_max = 0.16` / `height_zero_max = 0.22` were derived when the hold
trunk sat at `z0 + PLATE_HALF_THICKNESS + STAND_Z >= 0.225` m. The TUCKED hold
trunk is `z0 + PLATE_HALF_THICKNESS + TUCK_Z`, i.e. 0.139-0.249 m, so the
damper fired **during HOLD** for every `z0 <= 0.181` — at full cost at
`z0=0.10`. The practical magnitude was small (omega ~ 0 in a held tuck, and the
weight only ramps from iteration 2000), but the number was wrong and its test
passed on `0.22 < 0.225` while the quantity it meant to bound was 0.139 — so
it asserted nothing.

Re-derived against two measured heights. Measured HOLD trunk height, 32 noisy
trials per `z0`, sampled at 0.1/0.3/0.5/0.7/1.0/1.2 s:

```
z0=0.100  nominal=0.1390  min=0.1381  mean=0.1385  max=0.1388
z0=0.150  nominal=0.1890  min=0.1881  mean=0.1885  max=0.1890
z0=0.210  nominal=0.2490  min=0.2481  mean=0.2485  max=0.2489
```

- the landed STANDING trunk (~`STAND_Z` = 0.115 m) must be INSIDE the
  full-cost band — damping the arrival is the point;
- the lowest TUCKED HOLD trunk (0.1381 m measured, 0.139 nominal) must be
  OUTSIDE it.

`height_full_max = 0.121` / `height_zero_max = 0.132` gives 6 mm of clearance
on each side. The cfg test now bounds the ceiling against
`Z0_RANGE[0] + PLATE_HALF_THICKNESS + TUCK_Z` and against the measured 0.1381,
and an mdp test asserts the gate reads 0.0 at every HOLD trunk height the env
samples while the old 0.16/0.22 pair reads 1.0 there.

## 4. Reproducing this section

```bash
# the flop audit (repeat with --z0 0.10 / 0.21)
uv run python scripts/backflip_envelope.py --flop-audit --bam --z0 0.15 --tuck 0.75

# whole box including the curriculum's z0 ceiling
uv run python scripts/backflip_envelope.py --box-check --bam

# the HOLD trunk heights the damper ceiling is derived from: run_settle() with
# posture="tucked_env" at z0 = 0.10 / 0.15 / 0.21, 32 noisy trials each, and
# take the min over the sampled times (table above).
```


# Lower and gentler — and why the plate cannot lie on the ground

The user watched the env for the first time
(`uv run play Mjlab-Backflip-Flat-MicroDuck --agent zero --num-envs 1`) and
reported three things: the plate is not under the feet, it starts up in the
air, and the robot is launched far too hard and too far. All three are
answered below. CPU MuJoCo, BAM actuators, training sim timestep throughout.

## 1. "The plate is not under the feet" — a real bug, and not the one expected

**Cause: mjlab never calls `env.reset()` before the viewer's first episode.**
`ManagerBasedRlEnv.__init__` does not reset, and `mjlab/viewer/base.py` calls
`reset_environment()` only from `_process_actions` on the RESET action. So
`uv run play` runs a whole 4 s episode on the **compiled default state** plus
`_backflip_state`'s lazy defaults, and only episode 2 onward uses the reset
events.

That state was incoherent in exactly the reported way:

| | old default | why it looks broken |
|---|---|---|
| robot trunk | 0.12 m (`robot_groundcontact.xml`: `trunk_base pos="0 0 0.12"`) | standing on the floor |
| plate | 0.15 m (`launcher.xml` / `MICRODUCK_LAUNCHER_CFG`) | 3 cm ABOVE the trunk origin, i.e. **through the robot's body** |
| `_backflip_state` lazy defaults | `t_hold=0`, `vz=0`, `w0=0`, `z0=0.15` | phase is past HOLD at t=0, so the plate hovers motionless for 0.1 s and then teleports away **without ever launching** |

"Stuck at the level of the robot's body, in the middle of the robot" is
literally the 0.15-vs-0.12 overlap.

**The competing hypothesis is REFUTED.** The robot does NOT unfold and slide
off the plate under a zero action. `play --agent zero` commands HOME (the
joint-position action's offset is the model's default pose) while the reset
folds the robot into the tuck, so it *ought* to unfold — measured, it does not
budge, because the BAM servos cannot lift it out of the fold against gravity:

```
# settle: posture=tucked_env tuck=0.75 ctrl=HOME (zero action) on plate z0=0.15 trials=16 noisy=True duration=4.0s dt=0.005 bam=True
  t[s]  tilt_mean  tilt_max  xy_mean  xy_max  z_mean  n_fallen
  0.10       12.7      13.2    0.007   0.008   0.188         0
  0.30       11.7      12.1    0.006   0.007   0.188         0
  0.50       11.5      11.7    0.006   0.006   0.189         0
  0.70       11.5      11.7    0.006   0.006   0.189         0
  1.00       11.6      11.7    0.006   0.006   0.189         0
  4.00       11.6      11.7    0.006   0.007   0.189         0
```

Trunk height pinned at 0.189 m (= the tucked rest height), tilt 11.6 deg, 6 mm
of drift, 0/16 fallen, for the full 4 s episode. So the HOLD phase IS robust
under an untrained policy; the placement complaint was entirely the pre-reset
state.

**Fixed** by making the compiled state coherent: the launcher's default height
is 0.08 m (the middle of `Z0_RANGE`), the backflip env gives its robot entity
its own tucked init state on the plate top, and the lazy launch params became a
plausible mid-box toss with a real HOLD phase. Pinned by
`test_the_pre_reset_state_is_coherent` and
`test_the_default_launch_params_are_a_plausible_toss`.

## 2. "It starts up in the air" — the plate CANNOT lie on the ground

This is the design, not a bug — and the request to put the plate on the ground
runs into a measured geometric wall.

**The kneeling tuck rests on its SHINS with its FEET HANGING ~8 cm BELOW the
surface it sits on.** Contact trace of the settled hold at four plate heights
(`on FLOOR` lists robot geoms touching the terrain):

```
--- z0=0.01 (plate top at 0.020)
  t=1.00 tilt= 40.5 z=0.086 x=+0.055 | on FLOOR: ['g49', 'left_foot_collision', 'right_foot_collision']
--- z0=0.03 (plate top at 0.040)
  t=1.00 tilt= 13.6 z=0.069 x=+0.013 | on FLOOR: ['g76', 'left_foot_collision', 'right_foot_collision']
--- z0=0.05 (plate top at 0.060)
  t=1.00 tilt= 14.2 z=0.088 x=+0.009 | on FLOOR: ['left_foot_collision', 'right_foot_collision']
--- z0=0.10 (plate top at 0.110)
  t=1.00 tilt= 13.8 z=0.139 x=+0.008 | on FLOOR: []
```

The lowest foot geom sits 0.079 m below the plate top, at x = +0.051 — off the
front edge of an 18 cm plate. Minimum plate-top height for the feet to clear
the floor, scanned at three tuck depths:

```
   top     z0    tilt       h     |x|   floor-touching steps
-- tuck 0.5
 0.020  0.010    49.1  0.0643  0.0629      261 / 260
 0.060  0.050    13.2  0.0281  0.0052      261 / 260
 0.080  0.070    12.5  0.0285  0.0064        0 / 260
 0.090  0.080    12.5  0.0285  0.0064        0 / 260
 0.100  0.090    12.5  0.0285  0.0064        0 / 260
 0.110  0.100    12.5  0.0285  0.0064        0 / 260
 0.120  0.110    12.5  0.0285  0.0064        0 / 260
 0.150  0.140    12.5  0.0285  0.0064        0 / 260
-- tuck 0.75
 0.020  0.010    40.6  0.0661  0.0555      261 / 260
 0.060  0.050    13.8  0.0283  0.0089      261 / 260
 0.080  0.070    13.9  0.0286  0.0083        0 / 260
 0.090  0.080    13.9  0.0286  0.0083        0 / 260
 0.100  0.090    13.9  0.0286  0.0083        0 / 260
 0.110  0.100    13.9  0.0286  0.0083        0 / 260
 0.120  0.110    13.9  0.0286  0.0083        0 / 260
 0.150  0.140    13.9  0.0286  0.0083        0 / 260
-- tuck 1.0
 0.020  0.010    31.0  0.0628  0.0367      260 / 260
 0.060  0.050    15.4  0.0254  0.0097        0 / 260
 0.080  0.070    15.4  0.0254  0.0097        0 / 260
 0.090  0.080    15.4  0.0254  0.0097        0 / 260
 0.100  0.090    15.4  0.0254  0.0097        0 / 260
 0.110  0.100    15.4  0.0254  0.0097        0 / 260
 0.120  0.110    15.4  0.0254  0.0097        0 / 260
 0.150  0.140    15.4  0.0254  0.0097        0 / 260
```

**0.08 m is the floor**: at a 0.06 m top the feet still touch (261/261 steps at
tuck 0.5 and 0.75); at 0.08 and above the hold is clean and identical to the
elevated case (tilt 12.5-15.4 deg, h = 0.0285, 6-9 mm of drift) at every tuck
depth.

Three ways out were measured, and two fail:

- **A wider pad does not help.** The feet are BELOW the surface, not merely
  beyond its edge. Pads of 18x18, 28x22 and 40x28 cm at a 2 cm thickness on the
  floor all leave the hold at 33-49 deg with the floor still being hit.
- **A solid ground-resting block is worse.** Body origin at the top face,
  9-11 cm tall: the dangling feet strike the block's side and the robot slides
  off — tilt 23-93 deg and up to 14 cm of drift; only tuck 0.75 with the full
  footprint stays put, and then it perches ON its own feet at h = 0.065 rather
  than kneeling. Trimming the footprint in x makes it slide off every time.
- **A feet-flat SQUAT hold DOES sit on a ground-level plate.** `HOLD_LERP` is
  HOME lerped toward the tuck per joint (roulade's mid-roll parametrisation),
  which keeps the soles down:

```
### rest height scan (noiseless, spawned 0.10 above the plate top)
    f   h@0.5   h@3.0  tilt@0.5  tilt@3.0  floorhits
 0.50  0.0559  0.0551      33.8      35.9          0
 0.55  0.0629  0.0630      15.4      15.4          0
 0.60  0.0653  0.0650       6.1       5.7          0
 0.65  0.0661  0.0657       1.5       0.3          0
 0.70  0.0665  0.0661       8.2      11.2          0
 0.75  0.0725  0.0799      28.8      60.3        479
 0.80  0.0791  0.0778      48.5      60.6        523
 0.85  0.0789  0.0766      54.4      59.5        538
 0.90  0.0827  0.0764      64.1      58.7        544

### settle from the measured rest height, 16 noisy trials
-- f=0.6 spawn_h=0.065
   t=0.10 tilt    3.5/   5.2 xy 0.010/0.012 h 0.0720 fallen=0
   t=0.30 tilt   11.1/  12.3 xy 0.021/0.024 h 0.0644 fallen=0
   t=0.50 tilt   10.4/  11.7 xy 0.021/0.024 h 0.0647 fallen=0
   t=0.70 tilt   10.2/  11.5 xy 0.020/0.023 h 0.0647 fallen=0
   t=1.00 tilt    9.8/  10.9 xy 0.020/0.023 h 0.0647 fallen=0
   t=3.00 tilt    9.6/  10.5 xy 0.020/0.022 h 0.0644 fallen=0
   floor-touching steps: 0 over 16 trials
-- f=0.65 spawn_h=0.0657
   t=0.10 tilt    3.5/   5.0 xy 0.010/0.011 h 0.0683 fallen=0
   t=0.30 tilt    5.1/   6.4 xy 0.013/0.016 h 0.0658 fallen=0
   t=0.50 tilt    3.5/   4.8 xy 0.012/0.015 h 0.0659 fallen=0
   t=0.70 tilt    3.0/   4.5 xy 0.011/0.015 h 0.0658 fallen=0
   t=1.00 tilt    2.1/   2.9 xy 0.010/0.013 h 0.0658 fallen=0
   t=3.00 tilt    0.8/   1.4 xy 0.009/0.012 h 0.0657 fallen=0
   floor-touching steps: 0 over 16 trials
-- f=0.7 spawn_h=0.0661
   t=0.10 tilt    2.5/   3.6 xy 0.007/0.009 h 0.0668 fallen=0
   t=0.30 tilt    2.0/   3.8 xy 0.005/0.008 h 0.0667 fallen=0
   t=0.50 tilt    6.7/   8.0 xy 0.002/0.004 h 0.0667 fallen=0
   t=0.70 tilt    7.9/   9.1 xy 0.002/0.004 h 0.0665 fallen=0
   t=1.00 tilt    8.7/   9.7 xy 0.002/0.004 h 0.0663 fallen=0
   t=3.00 tilt   11.0/  12.0 xy 0.004/0.006 h 0.0661 fallen=0
   floor-touching steps: 0 over 16 trials
-- f=0.75 spawn_h=0.0799
   t=0.10 tilt    0.8/   1.4 xy 0.004/0.006 h 0.0671 fallen=0
   t=0.30 tilt   12.5/  14.7 xy 0.007/0.010 h 0.0673 fallen=0
   t=0.50 tilt   22.2/  26.0 xy 0.016/0.021 h 0.0690 fallen=0
   t=0.70 tilt   37.6/  47.2 xy 0.037/0.055 h 0.0749 fallen=5
   t=1.00 tilt   49.2/  52.3 xy 0.064/0.072 h 0.0798 fallen=15
   t=3.00 tilt   53.1/  58.4 xy 0.073/0.087 h 0.0810 fallen=16
   floor-touching steps: 7143 over 16 trials
```

  `HOLD_LERP = 0.65` is a genuinely good hold on a plate lying on the floor:
  tilt 3.5 deg at 0.1 s and 0.5 s, 2.1 at 1.0 s, **0.8 at 3.0 s** (it converges
  toward upright), 9-13 mm of drift, 0/16 fallen, and **zero robot-floor
  contacts**. 0.60 and 0.70 also hold; 0.75 and deeper topple.

**But the squat does not fly.** 420 launch cells from that hold at ground
level, with the policy folding to a full tuck at the flick,
`t_launch` in [0.09, 0.13], `vz` in [2.0, 2.5], `w0` in [18, 30]:

```
420 cells scanned
cells with rot >= 340 AND land <= 2.6: 0

cells with rot >= 340: 28; softest 10 landings:
  land= 3.09 rot= 364.6 launch=0.11 vz=2.5 w0=26.0 tuck=1.0 apex=0.501
  land= 3.20 rot= 342.0 launch=0.12 vz=2.4 w0=22.0 tuck=1.0 apex=0.495
  land= 3.22 rot= 352.8 launch=0.12 vz=2.5 w0=22.0 tuck=1.0 apex=0.512
  land= 3.24 rot= 342.5 launch=0.11 vz=2.3 w0=20.0 tuck=1.0 apex=0.501
  land= 3.26 rot= 386.3 launch=0.1 vz=2.5 w0=22.0 tuck=1.0 apex=0.574
  land= 3.26 rot= 349.1 launch=0.09 vz=2.3 w0=26.0 tuck=1.0 apex=0.521
  land= 3.28 rot= 345.8 launch=0.1 vz=2.5 w0=26.0 tuck=1.0 apex=0.535
  land= 3.29 rot= 368.5 launch=0.1 vz=2.4 w0=22.0 tuck=1.0 apex=0.544
  land= 3.29 rot= 343.6 launch=0.1 vz=2.5 w0=30.0 tuck=1.0 apex=0.534
  land= 3.30 rot= 346.8 launch=0.11 vz=2.4 w0=20.0 tuck=1.0 apex=0.536
```

**Zero cells close 360 deg under the 2.6 m/s hardware limit.** The softest
360-closing cell lands at **3.09 m/s**, 19% over. The mechanism is the same one
that killed the standing hold: a 6.6 cm CoM (vs the kneeling tuck's 2.9) does
not transfer the flick, and the reduced airtime of a ground launch has to be
paid for with more `vz`, which lands harder.

So the plate stays elevated — but pushed down to the geometric floor.
`HOLD_LERP`, `HOLD_Z` and the probe's `squat_env` posture are kept so this
measurement stays reproducible, not because the env uses them.

## 3. "Launched far too hard and too far" — the retune

Over-rotation is now a defect to minimise. New whole-box-verified box, from the
lowest plate the hold pose allows:

```
# box-check posture=tucked_env bam=True dt=0.005
#   z0 (0.07, 0.09) vz (1.9, 2.0) w0 (23.0, 24.0) launch (0.12, 0.13)
#   z0 grid (0.07, 0.08, 0.09)
#   hold (0.1, 1.0) tuck (0.5, 0.75, 1.0)
  486 cells | min rot = 372.5 deg | max landing = 2.15 m/s | short of 360: 0 | over 2.6 m/s: 0 | never landed: 0
  worst by rotation:
    rot=  372.5 land= 1.83 apex=0.329 tilt0= 13.9  z0=0.070 vz=1.900 w0=24.00 launch=0.130 hold=0.10 tuck=1.00
    rot=  378.3 land= 1.81 apex=0.339 tilt0= 13.9  z0=0.080 vz=1.900 w0=24.00 launch=0.130 hold=0.10 tuck=1.00
    rot=  381.3 land= 1.72 apex=0.336 tilt0= 13.9  z0=0.070 vz=1.950 w0=24.00 launch=0.130 hold=0.10 tuck=1.00
    rot=  381.8 land= 1.85 apex=0.338 tilt0= 13.9  z0=0.070 vz=1.900 w0=23.00 launch=0.130 hold=0.10 tuck=1.00
    rot=  383.5 land= 1.79 apex=0.335 tilt0= 13.9  z0=0.070 vz=1.900 w0=23.50 launch=0.130 hold=0.10 tuck=1.00
  worst by landing speed:
    rot=  387.4 land= 2.15 apex=0.390 tilt0= 15.9  z0=0.070 vz=1.900 w0=23.00 launch=0.130 hold=1.00 tuck=1.00
    rot=  397.5 land= 2.14 apex=0.400 tilt0= 15.9  z0=0.080 vz=1.900 w0=23.00 launch=0.130 hold=1.00 tuck=1.00
    rot=  402.6 land= 2.13 apex=0.410 tilt0= 15.9  z0=0.090 vz=1.900 w0=23.00 launch=0.130 hold=1.00 tuck=1.00
    rot=  401.5 land= 2.11 apex=0.398 tilt0= 15.9  z0=0.070 vz=1.900 w0=23.00 launch=0.125 hold=1.00 tuck=1.00
    rot=  384.1 land= 2.11 apex=0.381 tilt0= 15.9  z0=0.070 vz=1.900 w0=23.50 launch=0.130 hold=1.00 tuck=1.00
  RESULT: PASS — whole box closes 360 deg under 2.6 m/s
```

| | old box | **new box** |
|---|---|---|
| `z0` | 0.10-0.20 (tail to 0.225) | **0.07-0.09** |
| plate top | 0.11-0.21 m | **0.08-0.10 m** |
| `vz` | 2.00-2.10 | **1.90-2.00** |
| `w0` | 21-23 | **23-24** |
| `t_launch` | 0.12-0.14 | **0.12-0.13** |
| rotation (whole box) | 393.6-475.7 deg | **372.5-457.3 deg** |
| apex | 0.52-0.63 m | **0.29-0.40 m** |
| worst landing | 2.53 m/s | **2.15 m/s** |
| `never landed` cells | 0 | **0** |

Rotation is down at both ends, apex is down by a third, the worst landing has
0.45 m/s of margin instead of 0.09, and the plate is less than half as high.
**The target of 360-400 deg across the whole box is not reachable**: ~85 deg of
the spread is irreducible DR (`z0` x tuck depth x hold length), so a box tight
enough to hold 360-400 everywhere has no DR width left. 457 deg is 1.27 turns.

The scan behind the choice (worst rotation and worst landing over
`z0` in {0.07, 0.08, 0.09} x tuck x hold, per `(t_launch, vz, w0)`):

```
  lau    vz    w0   minrot   maxrot  maxland  verdict
 0.12  1.70  20.0    338.1    380.3     2.68  no
 0.12  1.70  22.0    357.8    375.0     2.43  no
 0.12  1.70  24.0    351.2    376.7     2.23  no
 0.12  1.70  26.0    330.7    373.3     2.15  no
 0.12  1.70  28.0    298.7    351.5     2.26  no
 0.12  1.80  20.0    364.7    400.4     2.62  no
 0.12  1.80  22.0    364.7    393.6     2.26  ok
 0.12  1.80  24.0    380.5    402.6     2.09  ok
 0.12  1.80  26.0    349.0    402.1     1.99  no
 0.12  1.80  28.0    326.6    389.9     2.04  no
 0.12  1.90  20.0    388.2    416.7     2.54  ok
 0.12  1.90  22.0    407.2    422.3     2.22  ok
 0.12  1.90  24.0    400.6    424.1     1.98  GENTLE
 0.12  1.90  26.0    381.3    420.0     1.79  GENTLE
 0.12  1.90  28.0    347.3    414.4     1.82  no
 0.12  2.00  20.0    399.3    440.3     2.41  no
 0.12  2.00  22.0    415.2    443.0     2.14  no
 0.12  2.00  24.0    425.3    443.9     1.93  no
 0.12  2.00  26.0    410.7    437.3     1.74  no
 0.12  2.00  28.0    384.1    439.2     1.61  no
 0.14  1.70  20.0    314.8    344.6     2.67  no
 0.14  1.70  22.0    309.3    343.5     2.53  no
 0.14  1.70  24.0    288.4    332.5     2.43  no
 0.14  1.70  26.0    270.5    313.0     2.42  no
 0.14  1.70  28.0    244.8    292.6     2.49  no
 0.14  1.80  20.0    345.3    362.2     2.61  no
 0.14  1.80  22.0    337.9    362.0     2.41  no
 0.14  1.80  24.0    317.4    354.0     2.30  no
 0.14  1.80  26.0    289.2    333.3     2.31  no
 0.14  1.80  28.0    258.1    316.5     2.40  no
 0.14  1.90  20.0    368.4    383.3     2.54  ok
 0.14  1.90  22.0    360.0    387.8     2.31  no
 0.14  1.90  24.0    339.8    384.6     2.15  no
 0.14  1.90  26.0    311.6    370.6     2.14  no
 0.14  1.90  28.0    285.8    333.1     2.28  no
 0.14  2.00  20.0    389.1    402.3     2.44  ok
 0.14  2.00  22.0    387.3    408.5     2.17  ok
 0.14  2.00  24.0    363.1    408.5     2.00  GENTLE
 0.14  2.00  26.0    331.5    388.8     1.95  no
 0.14  2.00  28.0    311.4    366.9     2.05  no
```

and the whole-box evaluation of the four final candidates:

```
### A: z0(0.07, 0.09) vz(1.9, 2.0) w0(23.0, 25.0) launch(0.12, 0.13)
    486 cells | rot 357.3-457.3 | max land 2.15
    worst rot  cell: z0=0.07 vz=1.9 w0=25.0 lau=0.13 hold=0.1 tuck=1.0 land=1.89 apex=0.316
    worst land cell: z0=0.07 vz=1.9 w0=23.0 lau=0.13 hold=1.0 tuck=1.0 rot=387.4 apex=0.390
### B: z0(0.07, 0.09) vz(1.95, 2.05) w0(23.0, 25.0) launch(0.12, 0.14)
    486 cells | rot 335.1-463.3 | max land 2.16
    worst rot  cell: z0=0.07 vz=1.95 w0=25.0 lau=0.14 hold=0.1 tuck=1.0 land=2.00 apex=0.302
    worst land cell: z0=0.07 vz=1.95 w0=23.0 lau=0.14 hold=1.0 tuck=1.0 rot=373.9 apex=0.383
### C: z0(0.07, 0.09) vz(1.85, 1.95) w0(24.0, 26.0) launch(0.11, 0.13)
    486 cells | rot 332.6-466.1 | max land 2.13
    worst rot  cell: z0=0.07 vz=1.9 w0=26.0 lau=0.13 hold=0.1 tuck=1.0 land=1.99 apex=0.292
    worst land cell: z0=0.07 vz=1.85 w0=24.0 lau=0.13 hold=1.0 tuck=1.0 rot=374.9 apex=0.360
### A2: z0(0.07, 0.09) vz(1.9, 2.0) w0(23.0, 24.0) launch(0.12, 0.13)
    486 cells | rot 372.5-457.3 | max land 2.15
    worst rot  cell: z0=0.07 vz=1.9 w0=24.0 lau=0.13 hold=0.1 tuck=1.0 land=1.83 apex=0.329
    worst land cell: z0=0.07 vz=1.9 w0=23.0 lau=0.13 hold=1.0 tuck=1.0 rot=387.4 apex=0.390
### A3: z0(0.07, 0.09) vz(1.95, 2.05) w0(23.0, 24.0) launch(0.12, 0.13)
    486 cells | rot 381.3-463.7 | max land 2.08
    worst rot  cell: z0=0.07 vz=1.95 w0=24.0 lau=0.13 hold=0.1 tuck=1.0 land=1.72 apex=0.336
    worst land cell: z0=0.07 vz=1.95 w0=23.0 lau=0.13 hold=1.0 tuck=1.0 rot=402.3 apex=0.402
```

`A2` was taken: the only candidate whose minimum rotation stays above 360 deg
with a real `w0` and `t_launch` width. `w0 = 25` drops the `z0=0.07` corner to
357 deg; `t_launch = 0.14` drops it to 340.

**Direction checks: 20/20 backward** — all 16 `(z0, vz, w0, t_launch)` corners
plus the box's two worst cells and both tuck extremes:

```
   z0    vz    w0   lau  hold  tuck     rot  land  verdict
 0.07  1.90  23.0 0.120  0.40  0.75   428.8  1.83  BACKWARD ok  (+z=(-0.518,-0.010,-0.855))
 0.07  1.90  23.0 0.130  0.40  0.75   407.6  1.81  BACKWARD ok  (+z=(-0.460,-0.013,-0.888))
 0.07  1.90  24.0 0.120  0.40  0.75   426.3  1.75  BACKWARD ok  (+z=(-0.508,-0.023,-0.861))
 0.07  1.90  24.0 0.130  0.40  0.75   400.8  1.77  BACKWARD ok  (+z=(-0.376,0.009,-0.926))
 0.07  2.00  23.0 0.120  0.40  0.75   443.1  1.81  BACKWARD ok  (+z=(-0.530,-0.015,-0.848))
 0.07  2.00  23.0 0.130  0.40  0.75   428.0  1.77  BACKWARD ok  (+z=(-0.463,-0.008,-0.886))
 0.07  2.00  24.0 0.120  0.40  0.75   443.0  1.71  BACKWARD ok  (+z=(-0.518,0.006,-0.856))
 0.07  2.00  24.0 0.130  0.40  0.75   428.6  1.69  BACKWARD ok  (+z=(-0.347,-0.012,-0.938))
 0.09  1.90  23.0 0.120  0.40  0.75   439.5  1.88  BACKWARD ok  (+z=(-0.518,-0.010,-0.855))
 0.09  1.90  23.0 0.130  0.40  0.75   418.6  1.83  BACKWARD ok  (+z=(-0.460,-0.013,-0.888))
 0.09  1.90  24.0 0.120  0.40  0.75   437.3  1.79  BACKWARD ok  (+z=(-0.508,-0.023,-0.861))
 0.09  1.90  24.0 0.130  0.40  0.75   412.1  1.78  BACKWARD ok  (+z=(-0.376,0.009,-0.926))
 0.09  2.00  23.0 0.120  0.40  0.75   453.8  1.88  BACKWARD ok  (+z=(-0.530,-0.015,-0.848))
 0.09  2.00  23.0 0.130  0.40  0.75   433.4  1.80  BACKWARD ok  (+z=(-0.463,-0.008,-0.886))
 0.09  2.00  24.0 0.120  0.40  0.75   448.5  1.75  BACKWARD ok  (+z=(-0.518,0.006,-0.856))
 0.09  2.00  24.0 0.130  0.40  0.75   434.3  1.71  BACKWARD ok  (+z=(-0.347,-0.012,-0.938))
 0.07  1.90  24.0 0.130  0.10  1.00   372.5  1.83  BACKWARD ok  (+z=(-0.229,0.033,-0.973))
 0.07  1.90  23.0 0.130  1.00  1.00   387.4  2.15  BACKWARD ok  (+z=(-0.499,-0.042,-0.866))
 0.08  1.95  23.5 0.125  0.55  0.50   420.9  1.88  BACKWARD ok  (+z=(-0.443,0.006,-0.896))
 0.08  1.95  23.5 0.125  0.55  1.00   417.4  1.85  BACKWARD ok  (+z=(-0.429,0.005,-0.903))

20 cells checked, 0 not confirmed backward
```

**Settle and flop audit re-run at the new heights** (the hold trunk moved from
0.139-0.249 m to 0.109-0.129 m, and `ready_stance`'s height factor moves with
it):

```
# flop-audit: z0=0.07 tuck=0.75 duration=3.0s dt=0.005 bam=True
 orientation   clr    tilt   drift  trunk_z    pose  height    pre  upright   TOTAL  on?
     upright 0.005    14.1   0.009    0.109   0.948   1.000  0.948    1.000   0.948  yes
   side_left 0.015   102.8   0.016    0.111   0.997   0.994  0.991    0.000   0.000  yes
  side_right 0.015   102.8   0.016    0.111   0.997   0.994  0.991    0.000   0.000  yes
   face_down 0.015    79.1   0.018    0.120   0.913   0.867  0.791    0.000   0.000  yes
     on_back 0.015    91.0   0.002    0.122   0.982   0.829  0.814    0.000   0.000  yes
    inverted 0.035   175.2   0.012    0.111   0.990   0.994  0.984    0.000   0.000  yes
  upright: pre=0.948 TOTAL=0.948   best flop: pre=0.991 TOTAL=0.000
  WITHOUT the upright factor: a flop would pay MORE (0.991 vs 0.948)
  RESULT: PASS - upright wins (0.948 vs 0.000)
# flop-audit: z0=0.09 tuck=0.75 duration=3.0s dt=0.005 bam=True
 orientation   clr    tilt   drift  trunk_z    pose  height    pre  upright   TOTAL  on?
     upright 0.005    14.1   0.009    0.129   0.948   1.000  0.948    1.000   0.948  yes
   side_left 0.015   102.8   0.016    0.131   0.997   0.994  0.991    0.000   0.000  yes
  side_right 0.015   102.8   0.016    0.131   0.997   0.994  0.991    0.000   0.000  yes
   face_down 0.015    88.5   0.021    0.134   0.880   0.977  0.860    0.000   0.000  yes
     on_back 0.015    91.0   0.002    0.142   0.982   0.829  0.814    0.000   0.000  yes
    inverted 0.005   178.3   0.012    0.127   0.991   0.994  0.985    0.000   0.000  yes
  upright: pre=0.948 TOTAL=0.948   best flop: pre=0.991 TOTAL=0.000
  WITHOUT the upright factor: a flop would pay MORE (0.991 vs 0.948)
  RESULT: PASS - upright wins (0.948 vs 0.000)
# settle: posture=tucked_env tuck=0.75 ctrl=spawn pose on plate z0=0.07 trials=32 noisy=True duration=3.0s dt=0.005 bam=True
  t[s]  tilt_mean  tilt_max  xy_mean  xy_max  z_mean  n_fallen
  0.10       13.7      14.4    0.008   0.009   0.108         0
  0.30       14.3      14.6    0.008   0.009   0.109         0
  0.50       14.1      14.5    0.008   0.009   0.109         0
  0.70       14.0      14.5    0.008   0.009   0.109         0
  1.00       13.9      14.4    0.008   0.010   0.109         0
  3.00       14.0      14.5    0.008   0.011   0.109         0
# settle: posture=tucked_env tuck=0.75 ctrl=spawn pose on plate z0=0.09 trials=32 noisy=True duration=3.0s dt=0.005 bam=True
  t[s]  tilt_mean  tilt_max  xy_mean  xy_max  z_mean  n_fallen
  0.10       13.7      14.4    0.008   0.009   0.128         0
  0.30       14.3      14.6    0.008   0.009   0.129         0
  0.50       14.1      14.5    0.008   0.009   0.129         0
  0.70       14.0      14.5    0.008   0.009   0.129         0
  1.00       13.9      14.4    0.008   0.010   0.129         0
  3.00       14.0      14.5    0.008   0.011   0.129         0
```

Both unchanged: the hold still settles at 14.0 deg with 0/32 fallen at both
`z0` extremes, and every flop basin still scores exactly 0.000 against the
upright hold's 0.948.

## 4. Two consequences of the lower plate, recorded

- **The `z0` DR tail is deleted.** Upward it crosses the landing limit
  (whole-box worst landing 2.59 m/s at `z0=0.225`); downward the range is now
  2 cm wide. A curriculum stage that widens a 2 cm range is not worth its
  pacing risk.
- **`arrival_damping` now also acts during HOLD, deliberately.** The tucked
  hold trunk is 0.109-0.129 m, straddling `STAND_Z` = 0.115, so NO height
  ceiling can separate "held tuck" from "landed stand" any more — they are the
  same height. The ceiling (0.121 / 0.132) is kept for the FLIGHT, which the
  retune moved to a 0.29-0.40 m apex, comfortably outside. The accepted cost is
  ~0: a held tuck's trunk omega_xy is negligible (8-11 mm of drift over 3 s)
  and the weight is 0 until iteration 2000. A cfg test asserts the overlap so
  the next reader does not try to re-derive a number that cannot exist.
- **`ready_stance`'s height factor is a weaker discriminator now.** At
  `z0 = 0.07` a standing trunk (0.115 m) is only 6 mm from the tucked target
  (0.109), so the height Gaussian scores standing at 0.96 instead of 3e-4. The
  POSE factor still separates them (0.95 tucked vs 0.12 standing), so the
  composite prefers the tuck about 8:1 rather than 1000:1. Watch it if the
  policy stands up during HOLD.

## 5. Reproducing this section

```bash
# the zero-action hold (what `play --agent zero` commands)
uv run python scripts/backflip_envelope.py --settle --bam --z0 0.15 \
    --settle-trials 16 --settle-ctrl-home --settle-duration 4.0

# the ground-level squat hold, and the launch sweep from it
uv run python scripts/backflip_envelope.py --settle --bam --z0 0.01 --posture squat_env \
    --settle-trials 16
uv run python scripts/backflip_envelope.py --posture squat_env --tuck-at-flick --bam \
    --z0 0.01 --vz 2.0,2.25,2.5 --w0 18,22,26,30 --tuck 1.0

# the retuned box, whole-box
uv run python scripts/backflip_envelope.py --box-check --bam

# settle + flop audit at both z0 extremes
uv run python scripts/backflip_envelope.py --settle --bam --z0 0.07 --settle-trials 32 --tuck 0.75
uv run python scripts/backflip_envelope.py --flop-audit --bam --z0 0.09 --tuck 0.75

# the minimum plate-top height (contact-trace and clearance scans are two-line
# drivers over run_settle(); the tables above carry their parameters)
```


# The feet are under the plate — every envelope on this branch is invalid

**STOP-AND-REPORT SECTION.** A reviewer reproduced the user's "the plate is
still not at the level of the feet" complaint in one command, and following it
down invalidates every launch-envelope table above. No fix is shipped here;
this section is the evidence and the options.

## 1. What the spawn actually writes

At the env's own spawn (`Z0_RANGE` midpoint 0.08 → plate slab spanning
z 0.070-0.090; robot trunk at `z0 + PLATE_HALF_THICKNESS + TUCK_Z` = 0.119,
joints at `TUCK_OVERRIDES x TUCK_FACTOR`, orientation from `reset_base`, i.e.
near identity):

```
plate center z 0.08 -> top 0.09
trunk z 0.119
left_foot_collision  center z 0.0657  x 0.0574  -> relative to plate top: -0.0243
right_foot_collision center z 0.0657  x 0.0574  -> relative to plate top: -0.0243
ncon 8
    plate_geom <-> 9                    dist -0.01038  (x2)
    plate_geom <-> 26                   dist -0.00602
    plate_geom <-> left_foot_collision  dist -0.02198  (x2)
    plate_geom <-> 76                   dist -0.00602
    plate_geom <-> right_foot_collision dist -0.02198  (x2)
```

Six to nine simultaneously penetrating contacts, the deepest **20-22 mm**. The
feet are at x = 0.057 against a plate half-width of 0.09 — **inside** the
footprint, not overhanging its edge.

## 2. Where the feet actually are: UNDER the slab

`geom_xpos` z of the foot geoms against the slab's own extent, at every
configuration this branch has used:

```
plate slab: z in [0.070, 0.090], footprint |x|<0.09, |y|<0.09

A. the ORIGINAL probe spawn (SPAWN_OFFSET = 0.02 above the plate top):
  at t=0, before any stepping
      left_foot_collision x=+0.0574 y=+0.0455 z=+0.0541  footprint=IN  UNDER the slab
     right_foot_collision x=+0.0574 y=-0.0455 z=+0.0541  footprint=IN  UNDER the slab

B. the env's spawn (TUCK_Z = 0.029, level):
  at t=0
      left_foot_collision x=+0.0574 y=+0.0455 z=+0.0631  footprint=IN  UNDER the slab
     right_foot_collision x=+0.0574 y=-0.0455 z=+0.0631  footprint=IN  UNDER the slab

C. the env's spawn WITH the equilibrium pitch (14 deg):
  at t=0
      left_foot_collision x=+0.0422 y=+0.0455 z=+0.0509  footprint=IN  UNDER the slab
     right_foot_collision x=+0.0422 y=-0.0455 z=+0.0509  footprint=IN  UNDER the slab

D. the settled 'working' equilibrium (what every envelope table used):
  after 2 s, trunk-top=+0.0286 tilt=14.1
      left_foot_collision x=+0.0512 y=+0.0459 z=+0.0469  footprint=IN  UNDER the slab
     right_foot_collision x=+0.0510 y=-0.0460 z=+0.0469  footprint=IN  UNDER the slab

E. the geometrically VALID rest (clearance height 0.0792, level):
  at t=0 (zero penetrating contacts)
      left_foot_collision x=+0.0574 y=+0.0455 z=+0.1133  footprint=IN  above the top
     right_foot_collision x=+0.0574 y=-0.0455 z=+0.1133  footprint=IN  above the top
  after 2 s, trunk-top=+0.0654 tilt=13.0
      left_foot_collision x=+0.0541 y=+0.0483 z=+0.1101  footprint=IN  above the top
     right_foot_collision x=+0.0540 y=-0.0487 z=+0.1100  footprint=IN  above the top
```

The feet are **under the plate slab, inside its footprint**, in the original
Task 3 probe spawn (A — at t=0, before a single step), in the env's spawn (B),
with the equilibrium pitch added (C), and in the settled "working" equilibrium
that every envelope table was measured from (D). Only the geometrically valid
rest (E) has them above the plate top.

**This is the root cause, and it is original.** The first probe's
`SPAWN_OFFSET = 0.02` was chosen by watching the trunk settle, and at that
offset the feet were already through the slab at t=0. Every measurement since
inherited it: the tucked tables, the standing comparison, the "whole-box
verified" boxes, all of it. The `--measure-tuck-z` mode reported a settled
TRUNK height and never once looked at a contact distance, which is why it
produced `TUCK_Z = 0.029` — a height at which the feet are 2.6 cm below the
surface they are supposed to be resting on.

## 3. Why the trunk height and the orientation disagree

Two separate errors compound:

- **Mixed configurations.** `TUCK_Z` is the settled trunk height of a pose
  pitched 14 deg forward, but the spawn writes it at `reset_base`'s near-level
  orientation. Spawn penetration vs pitch, at that height:

```
 pitch       h  spawn worst  npen/  n  peak|az|  end tilt    end h  end worst
   0.0   0.029     -0.02035     6/  6       3.7      14.1  +0.0286   -0.00441
   0.0   0.033     -0.02435     6/  6       8.4      13.8  +0.0286   -0.00433
   0.0   0.037     -0.02835     6/  6      17.6      14.1  +0.0286   -0.00443
   0.0   0.045     -0.03422     4/  4      43.7      14.1  +0.0264   -0.01571
   8.0   0.029     -0.01136     8/  8       3.0      14.1  +0.0286   -0.00445
   8.0   0.033     -0.01536     6/  6      12.4      14.0  +0.0286   -0.00439
   8.0   0.037     -0.01936     4/  4      27.6      14.1  +0.0286   -0.00439
   8.0   0.045     -0.02736     4/  4      42.7      14.0  +0.0286   -0.00440
  12.0   0.029     -0.00713     9/  9       2.6      13.9  +0.0286   -0.00445
  12.0   0.033     -0.01113     6/  6      15.3      14.3  +0.0285   -0.00458
  12.0   0.037     -0.01513     4/  4      26.1      14.4  +0.0286   -0.00455
  12.0   0.045     -0.02313     4/  4      50.3      14.0  +0.0286   -0.00443
  14.0   0.029     -0.00509     8/  8       3.8      13.8  +0.0286   -0.00437
  14.0   0.033     -0.00909     6/  6      16.9      14.1  +0.0286   -0.00446
  14.0   0.037     -0.01309     4/  4      30.5      13.8  +0.0286   -0.00438
  14.0   0.045     -0.02109     4/  4      47.7      13.9  +0.0286   -0.00441
  16.0   0.029     -0.00490     8/  8       4.3      13.8  +0.0287   -0.00435
  16.0   0.033     -0.00710     6/  6      14.8      13.8  +0.0286   -0.00440
  16.0   0.037     -0.01110     4/  4      30.8      14.0  +0.0286   -0.00446
  16.0   0.045     -0.01910     4/  4      48.0      14.0  +0.0286   -0.00442
  20.0   0.029     -0.00593     4/  4       3.4      13.8  +0.0286   -0.00438
  20.0   0.033     -0.00694     6/  6      13.9      13.8  +0.0286   -0.00436
  20.0   0.037     -0.01094     4/  4      31.6      13.8  +0.0286   -0.00436
  20.0   0.045     -0.01894     4/  4      48.6      13.8  +0.0286   -0.00441

best penetration-free spawn: None
```

  Adding the equilibrium pitch takes the deepest penetration from **20.4 mm to
  5.1 mm** — essentially the equilibrium's own 4.4 mm of loaded soft-contact
  compression. **But it does not lift the feet out from under the slab** (C
  above), so it is not a fix.

- **No penetration-free rest exists at that height.** Bisecting for the
  clearance height at the target pose gives **0.0792 m** above the plate top,
  5 cm higher than `TUCK_Z`, with the shins and feet co-planar at the surface —
  a proper flat kneel. Raising the trunk part-way is worse than either end
  (−31 mm at +11 mm of height, and a 48 m/s² settling transient) because the
  dangling feet close on the pad's underside on the way up.

**One thing the penetration is NOT: the cause of the violent departure.**
Resolving the 20 mm overlap peaks at |a_z| = 3.7 m/s² and |v_z| = 0.02 m/s —
below gravity. The "part beaucoup trop loin et fort" is the launch itself, not
an ejection.

## 4. The valid configuration does not fly

540 cells from the geometrically valid rest (trunk 0.0654-0.0792 m above the
plate top, zero penetrating contacts at spawn), `vz` in [1.5, 3.5],
`w0` in [6, 38], `t_launch` in [0.10, 0.15], tuck depth 0.5/1.0, with and
without folding at the flick:

```
540 cells from the corrected spawn (trunk 6.5 cm above the plate top)
  cells rotating BACKWARD at all (rot > 0): 98
  cells closing >= 360 deg:                 4
  ... AND landing <= 2.6 m/s:               0
  best (max) rotation: 390.6 deg

top 12 by rotation:
  rot=  390.6 land= 4.13 apex=0.942 vz=3.5 w0= 18.0 lau=0.15 tuck=0.5 tuck_at_flick=False
  rot=  390.6 land= 4.13 apex=0.942 vz=3.5 w0= 18.0 lau=0.15 tuck=0.5 tuck_at_flick=True
  rot=  374.8 land= 3.21 apex=0.507 vz=3.0 w0= 38.0 lau=0.15 tuck=0.5 tuck_at_flick=False
  rot=  374.8 land= 3.21 apex=0.507 vz=3.0 w0= 38.0 lau=0.15 tuck=0.5 tuck_at_flick=True
  rot=  345.1 land= 4.41 apex=1.030 vz=3.5 w0= 14.0 lau=0.15 tuck=0.5 tuck_at_flick=False
  rot=  345.1 land= 4.41 apex=1.030 vz=3.5 w0= 14.0 lau=0.15 tuck=0.5 tuck_at_flick=True
  rot=  324.5 land= 3.86 apex=0.752 vz=3.0 w0= 18.0 lau=0.15 tuck=0.5 tuck_at_flick=False
  rot=  324.5 land= 3.86 apex=0.752 vz=3.0 w0= 18.0 lau=0.15 tuck=0.5 tuck_at_flick=True
  rot=  290.8 land= 4.14 apex=0.863 vz=3.5 w0= 22.0 lau=0.15 tuck=0.5 tuck_at_flick=False
  rot=  290.8 land= 4.14 apex=0.863 vz=3.5 w0= 22.0 lau=0.15 tuck=0.5 tuck_at_flick=True
  rot=  290.6 land= 4.45 apex=1.011 vz=3.5 w0= 14.0 lau=0.12 tuck=0.5 tuck_at_flick=False
  rot=  290.6 land= 4.45 apex=1.011 vz=3.5 w0= 14.0 lau=0.12 tuck=0.5 tuck_at_flick=True
```

**4 cells close 360 deg; ZERO land under 2.6 m/s.** Best rotation 390.6 deg at
4.13 m/s.

Folding DEEPER at the flick (hold at 0.75, fold to 1.0 or 1.15) helps rotation
and not landing — 80 more cells:

```
80 cells: hold at 0.75, fold deeper at the flick
  backward at all: 69
  closing >= 360:  11
  ... under 2.6:   0
  best rotation:   487.7 deg

top 8 by rotation:
  rot=  487.7 land= 3.38 apex=0.828 vz=3.0 w0= 22.0 lau=0.12 flick_depth=1.0
  rot=  458.7 land= 3.87 apex=0.910 vz=3.0 w0= 18.0 lau=0.12 flick_depth=1.15
  rot=  448.3 land= 3.60 apex=0.828 vz=3.0 w0= 18.0 lau=0.15 flick_depth=1.15
  rot=  433.2 land= 3.43 apex=0.742 vz=3.0 w0= 26.0 lau=0.12 flick_depth=1.0
  rot=  426.1 land= 3.45 apex=0.849 vz=3.0 w0= 22.0 lau=0.12 flick_depth=1.15
  rot=  416.1 land= 3.58 apex=0.775 vz=3.0 w0= 26.0 lau=0.12 flick_depth=1.15
  rot=  383.7 land= 4.09 apex=0.833 vz=3.0 w0= 18.0 lau=0.15 flick_depth=1.0
  rot=  373.2 land= 3.89 apex=0.753 vz=3.0 w0= 22.0 lau=0.15 flick_depth=1.15
```

11 cells close; **zero under 2.6 m/s**; every closing cell needs `vz` = 3.0 and
lands at 3.38-4.09 m/s.

And the whole retuned box, re-run from the valid spawn, inverts completely:

```
# box-check posture=tucked_env bam=True dt=0.005
#   z0 (0.07, 0.09) vz (1.9, 2.0) w0 (23.0, 24.0) launch (0.12, 0.13)
#   z0 grid (0.07, 0.08, 0.09)
#   hold (0.1, 1.0) tuck (0.5, 0.75, 1.0)
  486 cells | min rot = -521.6 deg | max landing = 3.86 m/s | short of 360: 486 | over 2.6 m/s: 229 | never landed: 0
  worst by rotation:
    rot= -521.6 land= 3.80 apex=0.615 tilt0=  6.7  z0=0.090 vz=2.000 w0=23.50 launch=0.120 hold=0.10 tuck=1.00
    rot= -517.4 land= 3.78 apex=0.605 tilt0=  6.7  z0=0.080 vz=2.000 w0=23.50 launch=0.120 hold=0.10 tuck=1.00
    rot= -516.8 land= 3.78 apex=0.613 tilt0=  6.7  z0=0.090 vz=2.000 w0=24.00 launch=0.120 hold=0.10 tuck=1.00
    rot= -516.8 land= 3.78 apex=0.603 tilt0=  6.7  z0=0.080 vz=2.000 w0=24.00 launch=0.120 hold=0.10 tuck=1.00
    rot= -514.1 land= 3.86 apex=0.619 tilt0=  6.7  z0=0.090 vz=2.000 w0=23.00 launch=0.120 hold=0.10 tuck=1.00
  worst by landing speed:
    rot= -514.1 land= 3.86 apex=0.619 tilt0=  6.7  z0=0.090 vz=2.000 w0=23.00 launch=0.120 hold=0.10 tuck=1.00
    rot= -507.2 land= 3.84 apex=0.594 tilt0=  6.7  z0=0.080 vz=1.950 w0=23.00 launch=0.120 hold=0.10 tuck=1.00
    rot= -507.2 land= 3.84 apex=0.604 tilt0=  6.7  z0=0.090 vz=1.950 w0=23.00 launch=0.120 hold=0.10 tuck=1.00
    rot= -509.9 land= 3.83 apex=0.609 tilt0=  6.7  z0=0.080 vz=2.000 w0=23.00 launch=0.120 hold=0.10 tuck=1.00
    rot= -509.9 land= 3.83 apex=0.599 tilt0=  6.7  z0=0.070 vz=2.000 w0=23.00 launch=0.120 hold=0.10 tuck=1.00
  RESULT: FAIL — whole box does NOT close 360 deg under 2.6 m/s
```

All 486 cells rotate FORWARD (−514 to −522 deg), 229 of them over the landing
limit.

## 5. Every hold posture measured, one table

CoM height is the trunk above the surface it rests on.

| hold posture | trunk above support | best 360-closing landing | safe under 2.6 m/s? |
|---|---|---|---|
| standing (HOME at `STAND_Z`) | 0.115 m | 3.43 m/s | **no** (0 of ~700) |
| feet-flat squat, `HOLD_LERP` 0.65 | 0.066 m | 3.09 m/s | **no** (0 of 420) |
| tuck, correctly resting on the plate | 0.065-0.079 m | 3.38 m/s | **no** (0 of 620) |
| tuck, feet TUNNELLED under the plate | 0.029 m | 1.72-2.15 m/s | yes — **and it is not a physical configuration** |

The only posture that flies is the one whose feet are inside the launcher.
Every posture that rests validly on the plate has a trunk 6.5-11.5 cm above it,
and at that CoM the flick overdrives the sole contact and the robot comes out
forward — the same mechanism that killed the standing hold two waves ago.

## 6. What was left in the repo

No fix, deliberately. The user has already rejected two designs from the
viewer and a third guess is worse than this table.

- The cfg module docstring opens with a **DO NOT TRAIN** warning carrying the
  measurement.
- Three acceptance tests in `tests/test_backflip_cfg.py`
  (`test_the_spawn_is_not_jammed_into_the_plate`,
  `test_the_spawn_is_clean_at_every_sampled_launch_height`,
  `test_the_feet_are_not_inside_the_plate_footprint_and_below_its_top`) build
  the spawn state in CPU MuJoCo and measure every plate-robot contact. They are
  **strict xfail** against the current design, so they turn RED the moment a
  spawn becomes valid and must be unmarked then. A fourth
  (`test_pitching_the_spawn_reduces_but_does_not_remove_the_penetration`)
  passes today and pins the 20 mm → 5 mm pitch effect.
- `--box-check` now reports the rotation RANGE and a `never landed` counter.

## 7. The four directions, with what is already measured about each

1. **Change the tuck so the feet are genuinely the lowest point** (a deeper
   ankle fold), keeping the compactness. Not measured. The requirement is a
   pose whose lowest geoms are the FEET, whose trunk is under ~4 cm above them,
   and which is a stable equilibrium. Nothing in the current tuck family
   satisfies the first two together — that is the search.
2. **Shrink the pad so the feet overhang the edge while the shins are
   supported.** Partially measured and it looks hard: at the target pose the
   feet occupy x ≈ 0.02-0.09 and the supporting shin contacts sit at
   x ≈ -0.036 to +0.02, so the pad's front edge must fall inside a ~4 cm
   window, and trimming the footprint that far made the robot slide off in
   every configuration tried (wave 4, "solid block" scan).
3. **Accept a 6.5 cm trunk and give up the 2.6 m/s limit.** The valid tuck
   closes 360 deg at 3.38 m/s (≈58 cm of free fall). This is the user's call,
   not a simulation call.
4. **Give up the plate-flick launch for this pose.** Every measurement says
   the flick only transfers at a ~3 cm CoM, and no valid resting pose puts the
   duck that low on a 2 cm plate.

## 8. Reproducing this section

```bash
# the spawn state and its contacts (the reviewer's repro)
uv run python scripts/backflip_envelope.py --measure-tuck-z --bam --z0 0.08 --tuck 0.75
uv run --with pytest pytest tests/test_backflip_cfg.py -q -rx   # the xfail reasons

# the valid-rest sweeps and the pitch scan are short drivers over
# backflip_envelope.run_cell()/run_settle(); the tables above carry their
# exact parameters.
```


# Back to standing — the env's current, valid box

The user's call, on the previous section's numbers: the tucked hold was adopted
because it flew at 1.5-2.2 m/s where standing needed 3.4, and that advantage
turned out to come from a spawn with the feet tunnelled under the plate. From a
valid rest the tuck lands at 3.38 m/s and standing at 3.43 — equivalent — so
the tuck bought nothing and cost the whole geometric problem. **This section is
the standing hold, re-measured from scratch. It is the env's current box; every
box above it is history.**

## 1. The box

`--posture standing --tuck-at-flick --bam --box-check`, corners and midpoints of
`z0` / `vz` / `w0` / `t_launch`, crossed with the hold extremes and midpoint:

```
# box-check posture=standing bam=True dt=0.005
#   z0 (0.07, 0.09) vz (2.8, 2.9) w0 (18.5, 18.5) launch (0.155, 0.16)
#   z0 grid (0.07, 0.08, 0.09)
#   hold (0.1, 0.2, 0.5) tuck (1.0,)
  243 cells | rot 363.3-457.8 deg | max landing = 3.93 m/s | short of 360: 0 | over 2.6 m/s: 243 | never landed: 0
  worst by rotation:
    rot=  363.3 land= 3.48 apex=0.627 tilt0=  1.2  z0=0.080 vz=2.800 w0=18.50 launch=0.158 hold=0.10 tuck=1.00
    rot=  363.3 land= 3.48 apex=0.627 tilt0=  1.2  z0=0.080 vz=2.800 w0=18.50 launch=0.158 hold=0.10 tuck=1.00
    rot=  363.3 land= 3.48 apex=0.627 tilt0=  1.2  z0=0.080 vz=2.800 w0=18.50 launch=0.158 hold=0.10 tuck=1.00
    rot=  363.3 land= 3.48 apex=0.617 tilt0=  1.2  z0=0.070 vz=2.800 w0=18.50 launch=0.158 hold=0.10 tuck=1.00
    rot=  363.3 land= 3.48 apex=0.617 tilt0=  1.2  z0=0.070 vz=2.800 w0=18.50 launch=0.158 hold=0.10 tuck=1.00
  worst by landing speed:
    rot=  383.0 land= 3.93 apex=0.732 tilt0=  8.9  z0=0.080 vz=2.850 w0=18.50 launch=0.158 hold=0.50 tuck=1.00
    rot=  383.0 land= 3.93 apex=0.732 tilt0=  8.9  z0=0.080 vz=2.850 w0=18.50 launch=0.158 hold=0.50 tuck=1.00
    rot=  383.0 land= 3.93 apex=0.732 tilt0=  8.9  z0=0.080 vz=2.850 w0=18.50 launch=0.158 hold=0.50 tuck=1.00
    rot=  383.0 land= 3.93 apex=0.742 tilt0=  8.9  z0=0.090 vz=2.850 w0=18.50 launch=0.158 hold=0.50 tuck=1.00
    rot=  383.0 land= 3.93 apex=0.742 tilt0=  8.9  z0=0.090 vz=2.850 w0=18.50 launch=0.158 hold=0.50 tuck=1.00
  CLOSURE: PASS — 0 cells short of 360 deg, 0 that never land
  LANDING: 3.07-3.93 m/s (ABOVE the 2.6 m/s operator comfort threshold; 243/243 cells over)
```

| | value |
|---|---|
| `z0` | **0.07-0.09** m (plate top 0.08-0.10) |
| `vz` | **2.80-2.90** m/s |
| `w0` | **18.5** rad/s — no width, see below |
| `t_launch` | **0.155-0.16** s |
| `HOLD_RANGE` | 0.1-0.3 s, curriculum to 0.5 |
| rotation across the box | **363.3-457.8 deg** |
| landing across the box | **3.07-3.93 m/s** |
| cells short of 360 / never landing | **0 / 0** (243 cells) |
| direction checks | **33/33 backward** |

Direction traces, all 16 `(z0, vz, w0, t_launch)` corners x both hold extremes
plus the box centre:

```
   z0    vz    w0   lau  hold  tuck     rot  land  verdict
 0.07  2.80  18.5 0.155  0.10  1.00   374.8  3.50  BACKWARD ok  (+z=(-0.984,0.132,-0.117))
 0.07  2.80  18.5 0.155  0.50  1.00   406.5  3.72  BACKWARD ok  (+z=(-1.000,0.006,0.014))
 0.07  2.80  18.5 0.160  0.10  1.00   392.4  3.46  BACKWARD ok  (+z=(-0.986,-0.076,-0.151))
 0.07  2.80  18.5 0.160  0.50  1.00   405.1  3.68  BACKWARD ok  (+z=(-0.999,-0.044,-0.003))
 0.07  2.80  18.5 0.155  0.10  1.00   374.8  3.50  BACKWARD ok  (+z=(-0.984,0.132,-0.117))
 0.07  2.80  18.5 0.155  0.50  1.00   406.5  3.72  BACKWARD ok  (+z=(-1.000,0.006,0.014))
 0.07  2.80  18.5 0.160  0.10  1.00   392.4  3.46  BACKWARD ok  (+z=(-0.986,-0.076,-0.151))
 0.07  2.80  18.5 0.160  0.50  1.00   405.1  3.68  BACKWARD ok  (+z=(-0.999,-0.044,-0.003))
 0.07  2.90  18.5 0.155  0.10  1.00   389.5  3.66  BACKWARD ok  (+z=(-0.989,0.010,-0.150))
 0.07  2.90  18.5 0.155  0.50  1.00   373.7  3.85  BACKWARD ok  (+z=(-0.994,0.099,0.051))
 0.07  2.90  18.5 0.160  0.10  1.00   406.7  3.42  BACKWARD ok  (+z=(-0.983,-0.079,-0.168))
 0.07  2.90  18.5 0.160  0.50  1.00   391.8  3.81  BACKWARD ok  (+z=(-0.994,-0.104,-0.024))
 0.07  2.90  18.5 0.155  0.10  1.00   389.5  3.66  BACKWARD ok  (+z=(-0.989,0.010,-0.150))
 0.07  2.90  18.5 0.155  0.50  1.00   373.7  3.85  BACKWARD ok  (+z=(-0.994,0.099,0.051))
 0.07  2.90  18.5 0.160  0.10  1.00   406.7  3.42  BACKWARD ok  (+z=(-0.983,-0.079,-0.168))
 0.07  2.90  18.5 0.160  0.50  1.00   391.8  3.81  BACKWARD ok  (+z=(-0.994,-0.104,-0.024))
 0.09  2.80  18.5 0.155  0.10  1.00   381.1  3.56  BACKWARD ok  (+z=(-0.984,0.132,-0.117))
 0.09  2.80  18.5 0.155  0.50  1.00   409.8  3.75  BACKWARD ok  (+z=(-1.000,0.006,0.014))
 0.09  2.80  18.5 0.160  0.10  1.00   395.8  3.49  BACKWARD ok  (+z=(-0.986,-0.076,-0.151))
 0.09  2.80  18.5 0.160  0.50  1.00   408.5  3.71  BACKWARD ok  (+z=(-0.999,-0.044,-0.003))
 0.09  2.80  18.5 0.155  0.10  1.00   381.1  3.56  BACKWARD ok  (+z=(-0.984,0.132,-0.117))
 0.09  2.80  18.5 0.155  0.50  1.00   409.8  3.75  BACKWARD ok  (+z=(-1.000,0.006,0.014))
 0.09  2.80  18.5 0.160  0.10  1.00   395.8  3.49  BACKWARD ok  (+z=(-0.986,-0.076,-0.151))
 0.09  2.80  18.5 0.160  0.50  1.00   408.5  3.71  BACKWARD ok  (+z=(-0.999,-0.044,-0.003))
 0.09  2.90  18.5 0.155  0.10  1.00   392.8  3.69  BACKWARD ok  (+z=(-0.989,0.010,-0.150))
 0.09  2.90  18.5 0.155  0.50  1.00   376.6  3.89  BACKWARD ok  (+z=(-0.994,0.099,0.051))
 0.09  2.90  18.5 0.160  0.10  1.00   410.1  3.44  BACKWARD ok  (+z=(-0.983,-0.079,-0.168))
 0.09  2.90  18.5 0.160  0.50  1.00   394.9  3.84  BACKWARD ok  (+z=(-0.994,-0.104,-0.024))
 0.09  2.90  18.5 0.155  0.10  1.00   392.8  3.69  BACKWARD ok  (+z=(-0.989,0.010,-0.150))
 0.09  2.90  18.5 0.155  0.50  1.00   376.6  3.89  BACKWARD ok  (+z=(-0.994,0.099,0.051))
 0.09  2.90  18.5 0.160  0.10  1.00   410.1  3.44  BACKWARD ok  (+z=(-0.983,-0.079,-0.168))
 0.09  2.90  18.5 0.160  0.50  1.00   394.9  3.84  BACKWARD ok  (+z=(-0.994,-0.104,-0.024))
 0.08  2.85  18.5 0.158  0.30  1.00   375.0  3.72  BACKWARD ok  (+z=(-0.990,0.116,-0.074))

33 cells checked, 0 not confirmed backward
```

## 2. Two things to be honest about

**The landing is above the operator's comfort threshold.** 3.1-3.9 m/s against
the ~2.6 m/s named. Accepted knowingly. `--box-check` now reports CLOSURE and
LANDING as separate verdicts rather than one fused PASS/FAIL, because closure
is a correctness property of the box while the landing speed is a trade-off the
user owns, and fusing them hid which was failing.

**THE PROBE MEASURES THE LAUNCH, NOT THE SKILL.** It holds a fixed pose and
folds once, instantaneously, at the flick. A trained policy tucks to spin
faster — needing less altitude, so a lower apex and a slower touchdown — and
extends to brake before landing. Whether that closes the 1.3 m/s gap is a
training question and nothing here predicts it either way.

**The launch is knife-edge in `w0`.** Whole-box minimum rotation against `w0`
at `t_launch` = 0.16, `vz` = 2.85: 379 deg at 17.5, **342 at 18.0**, 366 at
18.5, **248 at 19.0**. A 100-300 deg swing between neighbouring values, so
`W0_RANGE` carries no DR width at all. A 3x wider search
(`w0` in [15, 24] x `vz` in [2.3, 3.0] x `t_launch` in [0.12, 0.17]) found no
rectangle with more width that closes whole-box:

```
  lau    vz    w0   minrot   maxrot  maxland  never  ok
 0.15  2.60  18.0    342.2    405.7     3.83      0  no
 0.15  2.60  21.0    221.7    397.0     3.44      0  no
 0.15  2.80  18.0    360.4    457.5     3.92      0  YES
 0.15  2.80  21.0    284.7    439.1     3.46      0  no
 0.15  3.00  18.0    285.0    409.1     3.93      0  no
 0.15  3.00  21.0    332.7    498.8     3.89      0  no
 0.16  2.60  18.0    267.6    415.4     3.50      0  no
 0.16  2.60  21.0    140.9    358.2     3.46      0  no
 0.16  2.80  18.0    365.3    413.7     3.71      0  YES
 0.16  2.80  21.0    148.2    377.7     3.57      0  no
 0.16  3.00  18.0    307.0    403.1     3.84      0  no
 0.16  3.00  21.0    131.3    429.8     3.43      0  no
 0.17  2.60  18.0    356.6    367.5     3.51      0  no
 0.17  2.60  21.0    183.0    391.6     3.38      0  no
 0.17  2.80  18.0    354.2    453.2     3.81      0  no
 0.17  2.80  21.0    192.9    371.8     3.46      0  no
 0.17  3.00  18.0    180.5    491.5     3.47      0  no
 0.17  3.00  21.0    342.2    448.2     3.75      0  no

rectangles with min rot >= 360 and no never-landing cell:
  cells= 2 launch=[0.15, 0.16] vz=[2.8] w0=[18.0] -> rot 360.4-457.5, land <= 3.92
  cells= 1 launch=[0.16] vz=[2.8] w0=[18.0] -> rot 365.3-413.7, land <= 3.71
  cells= 1 launch=[0.15] vz=[2.8] w0=[18.0] -> rot 360.4-457.5, land <= 3.92
```

That is a real sim2real risk — a human's flick does not repeat to +-0.5 rad/s —
and it is the honest state of the measurement, not a tuning failure.

**The fold depth is the policy's action, not DR.** At `t_launch` = 0.15,
`vz` = 2.8, `w0` = 18 a 0.75 fold gives 209-254 deg where a full fold gives
357-458. `z0` and `t_hold` shift rotation by under 10 deg over the same cells.
The box is therefore verified at the full fold — the way the task requires it
to be flown — and the fold depth is what the policy has to learn:

```
### launch=0.15 vz=2.8 w0=18.0
     z0  hold  fold      rot   land   apex  tilt0
   0.07  0.10  0.75    250.8   3.53  0.639    1.2
   0.07  0.10  1.00    360.4   3.71  0.636    1.2
   0.07  0.15  0.75    225.5   3.28  0.619    2.1
   0.07  0.15  1.00    356.9   3.66  0.625    2.1
   0.07  0.20  0.75    208.7   3.01  0.615    3.0
   0.07  0.20  1.00    450.0   3.21  0.680    3.0
   0.08  0.10  0.75    252.4   3.59  0.649    1.2
   0.08  0.10  1.00    360.4   3.71  0.646    1.2
   0.08  0.15  0.75    225.5   3.28  0.629    2.1
   0.08  0.15  1.00    360.0   3.71  0.635    2.1
   0.08  0.20  0.75    208.7   3.01  0.625    3.0
   0.08  0.20  1.00    453.7   3.23  0.690    3.0
   0.09  0.10  0.75    253.9   3.65  0.659    1.2
   0.09  0.10  1.00    363.5   3.75  0.656    1.2
   0.09  0.15  0.75    227.0   3.34  0.639    2.1
   0.09  0.15  1.00    360.0   3.71  0.645    2.1
   0.09  0.20  0.75    210.5   3.06  0.635    3.0
   0.09  0.20  1.00    457.5   3.25  0.700    3.0
### launch=0.16 vz=2.8 w0=18.0
     z0  hold  fold      rot   land   apex  tilt0
   0.07  0.10  0.75    303.1   3.65  0.662    1.2
   0.07  0.10  1.00    368.5   3.55  0.625    1.2
   0.07  0.15  0.75    325.9   3.80  0.691    2.1
   0.07  0.15  1.00    389.5   3.52  0.636    2.1
   0.07  0.20  0.75    334.5   3.80  0.699    3.0
   0.07  0.20  1.00    365.3   3.67  0.652    3.0
   0.08  0.10  0.75    305.6   3.71  0.672    1.2
   0.08  0.10  1.00    371.7   3.58  0.635    1.2
   0.08  0.15  0.75    325.9   3.80  0.701    2.1
   0.08  0.15  1.00    392.8   3.55  0.646    2.1
   0.08  0.20  0.75    334.5   3.80  0.709    3.0
   0.08  0.20  1.00    368.3   3.71  0.662    3.0
   0.09  0.10  0.75    305.6   3.71  0.682    1.2
   0.09  0.10  1.00    374.8   3.62  0.645    1.2
   0.09  0.15  0.75    328.5   3.85  0.711    2.1
   0.09  0.15  1.00    396.2   3.58  0.656    2.1
   0.09  0.20  0.75    337.2   3.85  0.719    3.0
   0.09  0.20  1.00    368.3   3.71  0.672    3.0
### launch=0.15 vz=2.6 w0=18.0
     z0  hold  fold      rot   land   apex  tilt0
   0.07  0.10  0.75    280.2   3.33  0.601    1.2
   0.07  0.10  1.00    402.1   3.38  0.602    1.2
   0.07  0.15  0.75    298.0   3.47  0.614    2.1
   0.07  0.15  1.00    318.4   3.52  0.590    2.1
   0.07  0.20  0.75    289.9   3.43  0.622    3.0
   0.07  0.20  1.00    342.2   3.51  0.586    3.0
   0.08  0.10  0.75    280.3   3.33  0.611    1.2
   0.08  0.10  1.00    402.1   3.38  0.612    1.2
   0.08  0.15  0.75    298.0   3.47  0.624    2.1
   0.08  0.15  1.00    321.1   3.57  0.600    2.1
   0.08  0.20  0.75    289.9   3.43  0.632    3.0
   0.08  0.20  1.00    345.0   3.55  0.596    3.0
   0.09  0.10  0.75    282.8   3.39  0.621    1.2
   0.09  0.10  1.00    405.7   3.41  0.622    1.2
   0.09  0.15  0.75    300.5   3.53  0.634    2.1
   0.09  0.15  1.00    323.7   3.63  0.610    2.1
   0.09  0.20  0.75    292.4   3.49  0.642    3.0
```

## 3. `HOLD_RANGE`, capped from the drift measurement

Open-loop standing drift on the plate under BAM, 32 noisy trials: **3.5 deg of
tilt at 0.3 s, 7.3 at 0.5 s, 11.6 at 0.7 s, 23.2 (max 42.4) at 1.0 s.**
`HOLD_RANGE` is 0.1-0.3 s and the curriculum widens it to 0.5 and no further —
the last point the pose holds itself unaided. This is a floor on what is safe,
not a claim about the limit: a trained policy balances, and this robot's
walking and stand-up policies hold far longer than a second.

## 4. Flop audit, standing

Run over the HOLD window (0.5 s) rather than 3 s, because that is the window
`ready_stance` pays in — standing is not a passive basin and topples by 3 s,
which is why the hold is capped above.

```
# flop-audit: z0=0.08 tuck=0.75 duration=0.5s dt=0.005 bam=True
 orientation   clr    tilt   drift  trunk_z    pose  height    pre  upright   TOTAL  on?
     upright 0.045     8.4   0.009    0.200   1.000   0.975  0.975    1.000   0.975  yes
   side_left 0.045   126.1   0.018    0.115   1.000   0.000  0.000    0.000   0.000  yes
  side_right 0.045   126.1   0.018    0.115   1.000   0.000  0.000    0.000   0.000  yes
   face_down 0.035    99.4   0.002    0.112   1.000   0.000  0.000    0.000   0.000  yes
     on_back 0.035   120.2   0.025    0.142   1.000   0.012  0.012    0.000   0.000  yes
    inverted 0.005   115.9   0.049    0.157   1.000   0.079  0.079    0.000   0.000  yes
  upright: pre=0.975 TOTAL=0.975   best flop: pre=0.079 TOTAL=0.000
  WITHOUT the upright factor: upright would win (0.079 vs 0.975)
  RESULT: PASS - upright wins (0.975 vs 0.000)
```

| basin | settled tilt | trunk z | height | upright | **TOTAL** |
|---|---|---|---|---|---|
| **upright** (the spawn) | 8.4 deg | 0.200 | 0.975 | 1.000 | **0.975** |
| side_left | 126.1 deg | 0.115 | 0.000 | 0.000 | **0.000** |
| side_right | 126.1 deg | 0.115 | 0.000 | 0.000 | **0.000** |
| face_down | 99.4 deg | 0.112 | 0.000 | 0.000 | **0.000** |
| on_back | 120.2 deg | 0.142 | 0.012 | 0.000 | **0.000** |
| inverted | 115.9 deg | 0.157 | 0.079 | 0.000 | **0.000** |

Every flop scores exactly 0.000 against the upright hold's 0.975. Note that for
a STANDING hold the height factor alone would already have won (0.975 vs
0.079), because a fallen robot's trunk is much lower — unlike the tucked hold,
where height could not tell upright from inverted and the side basin outscored
the intended pose. The upright factor is kept anyway: it costs exactly zero at
0 deg of tilt, and it is the guard whose absence cost a whole wave.

## 5. The spawn-validity tests now PASS

The three tests left as strict xfail in the previous section are ordinary
passing tests again, and a fourth was added:

- `test_the_spawn_is_not_jammed_into_the_plate` — deepest plate penetration at
  the spawn is within loaded-contact tolerance (6 mm).
- `test_the_spawn_is_clean_at_every_sampled_launch_height` — and it is
  `z0`-invariant, so the two heights cannot drift apart again.
- `test_no_robot_geom_is_below_the_plate_top_inside_its_footprint` — the exact
  geometry the tuck produced.
- `test_the_feet_are_the_lowest_geoms_so_the_spawn_cannot_tunnel` — **why
  standing is structurally safe rather than merely fixed.** The tuck kneels on
  its shins, so placing it by trunk height put the feet through the slab.
  Standing's lowest geoms are the feet, so "trunk at STAND_Z above the surface"
  puts the soles on it by construction.

Full suite: **298 passed, 1 skipped, 0 xfailed.**

## 6. Reproducing this section

```bash
uv run python scripts/backflip_envelope.py --box-check --bam
uv run python scripts/backflip_envelope.py --flop-audit --bam --posture standing \
    --z0 0.08 --settle-duration 0.5
uv run python scripts/backflip_envelope.py --settle --bam --posture standing \
    --z0 0.08 --settle-trials 32
uv run python scripts/backflip_envelope.py --bam --posture standing --tuck-at-flick \
    --z0 0.08 --hold 0.3 --launch 0.158 --check-direction \
    --check-vz 2.85 --check-w0 18.5 --check-tuck 1.0
uv run --with pytest pytest tests/test_backflip_cfg.py -q
```


# v0 — a plausible human throw, and closure as information

Change of direction from the user: the acceptance bar was over-engineered.
**Whole-box open-loop closure is dropped as a criterion.** It demanded that a
robot with NO skill complete every throw, when compensating for an imperfect
throw is precisely the policy's job — and it is what squeezed `w0` to the
single knife-edge value 18.5 rad/s that no human hand reproduces.

The launch ranges are now the spread a person's hands plausibly deliver,
centred on what was already measured. No new sweep was run to justify them.

| | v0 range | previous |
|---|---|---|
| `z0` | 0.07-0.09 m | 0.07-0.09 |
| `vz` | **2.50-3.50** m/s | 2.80-2.90 |
| `w0` | **15.0-24.0** rad/s | 18.5 (single value) |
| `t_launch` | **0.12-0.16** s | 0.155-0.16 |
| `HOLD_RANGE` | 0.1-0.3 s, curriculum to 0.5 | unchanged |

**The one hard limit is DIRECTION.** Above roughly `w0` 24-27 rad/s from a
standing hold the flick overdrives the sole contact and the robot comes out
FORWARD, face-down. The ceiling stays inside that measured boundary.

Verified cheaply, corners and midpoints of all four ranges crossed with the
hold extremes and midpoint (243 cells):

```
# box-check posture=standing bam=True dt=0.005
#   z0 (0.07, 0.09) vz (2.5, 3.5) w0 (15.0, 24.0) launch (0.12, 0.16)
#   z0 grid (0.07, 0.08, 0.09)
#   hold (0.1, 0.2, 0.5) tuck (1.0,)
  243 cells | rot 40.5-572.3 deg | max landing = 4.41 m/s | short of 360: 154 | over 2.6 m/s: 232 | never landed: 0
  worst by rotation:
    rot=   40.5 land= 2.84 apex=0.473 tilt0=  3.0  z0=0.070 vz=2.500 w0=15.00 launch=0.120 hold=0.20 tuck=1.00
    rot=   40.8 land= 2.89 apex=0.483 tilt0=  3.0  z0=0.080 vz=2.500 w0=15.00 launch=0.120 hold=0.20 tuck=1.00
    rot=   40.8 land= 2.89 apex=0.493 tilt0=  3.0  z0=0.090 vz=2.500 w0=15.00 launch=0.120 hold=0.20 tuck=1.00
    rot=   43.5 land= 2.38 apex=0.426 tilt0=  1.2  z0=0.070 vz=2.500 w0=15.00 launch=0.120 hold=0.10 tuck=1.00
    rot=   44.5 land= 2.43 apex=0.436 tilt0=  1.2  z0=0.080 vz=2.500 w0=15.00 launch=0.120 hold=0.10 tuck=1.00
  worst by landing speed:
    rot=  370.6 land= 4.41 apex=0.993 tilt0=  8.9  z0=0.080 vz=3.500 w0=15.00 launch=0.160 hold=0.50 tuck=1.00
    rot=  370.6 land= 4.41 apex=1.003 tilt0=  8.9  z0=0.090 vz=3.500 w0=15.00 launch=0.160 hold=0.50 tuck=1.00
    rot=  350.8 land= 4.38 apex=0.933 tilt0=  3.0  z0=0.090 vz=3.500 w0=15.00 launch=0.140 hold=0.20 tuck=1.00
    rot=  350.8 land= 4.38 apex=0.923 tilt0=  3.0  z0=0.080 vz=3.500 w0=15.00 launch=0.140 hold=0.20 tuck=1.00
    rot=  368.6 land= 4.38 apex=0.983 tilt0=  8.9  z0=0.070 vz=3.500 w0=15.00 launch=0.160 hold=0.50 tuck=1.00
  DIRECTION: PASS — 0 cells rotate FORWARD (min rot 40.5 deg)
  OPEN-LOOP CLOSURE (information, not a gate): 89/243 cells reach 360 deg = 37%; 0 never land
  LANDING: 2.38-4.41 m/s (operator comfort threshold 2.6 m/s; 232/243 cells over)
```

- **DIRECTION: PASS — 0 of 243 cells rotate forward** (minimum rotation
  +40.5 deg).
- **Open-loop closure: 89/243 = 37%.** This is INFORMATION about starting
  difficulty, not a gate. A fixed-pose robot completes about a third of the
  throws it will be handed; the rest are what the policy has to learn to
  rescue. A low fraction here is expected and fine.
- Landing 2.38-4.41 m/s open-loop, against the operator's ~2.6 m/s preference.
  A trained policy that tucks to spin faster and extends to brake before
  contact may do better; nothing measured open-loop predicts that.

`--box-check` now prints DIRECTION as the only PASS/FAIL and reports closure as
a percentage, so the tool cannot re-impose the bar that was just removed.


# After the first training run — long hold, plate on the ground

First real run: 4096 envs, 279 iterations, `flip_progress +1.97`,
`landing +1.72`, every penalty negative, the 300 deg landing gate opening. The
env produces real backflips. The user then watched the video and found three
things, all correct.

## 1. It collapsed before the impulse

`ready_stance` logged **+0.036** against a weight of 1.0. The constraint the
user asked about does exist — `ready_stance` is a height Gaussian times a wide
upright gate — it was simply too weak to matter: the hold lasted 0.1-0.3 s, so
the term's entire episode-sum mass was 0.1-0.3 against 8.0 for the flip and up
to 8.0 for the landing. Collapsing was almost free.

## 2. Hold 1-5 s, episode 7.5 s

`HOLD_RANGE` is now **1.0-5.0 s**. Episode length is derived, not guessed:
5.0 s hold + 0.16 s launch + the MEASURED worst-case flight leaves the settle
window. Airborne duration across the box, GONE to first terrain contact:

```
### flight duration at the CURRENT box's most energetic corner
  z0=0.09 vz=2.5 w0=15 lau=0.16: rot= 285.1 land=3.48 apex=0.598 flight=0.565 s
  z0=0.09 vz=3.0 w0=15 lau=0.16: rot= 325.5 land=4.04 apex=0.771 flight=0.665 s
  z0=0.09 vz=3.5 w0=15 lau=0.16: rot= 445.1 land=4.17 apex=0.976 flight=0.750 s

### the same launch from a plate resting on the FLOOR (z0 = 0.01)
     vz    w0   lau     rot   land   apex  flight
   2.50  15.0  0.12    29.3   2.59  0.413   0.455
   2.50  15.0  0.16   273.6   3.13  0.517   0.535
   2.50  19.5  0.12   321.9   3.19  0.483   0.525
   2.50  19.5  0.16   333.8   3.16  0.458   0.475
   2.50  24.0  0.12   179.7   2.30  0.393   0.460
   2.50  24.0  0.16   264.0   2.58  0.379   0.415
   3.00  15.0  0.12   213.0   2.91  0.585   0.585
   3.00  15.0  0.16   289.7   3.64  0.665   0.625
   3.00  19.5  0.12   309.8   3.50  0.598   0.590
   3.00  19.5  0.16   404.3   3.35  0.619   0.565  <== closes
   3.00  24.0  0.12   430.3   3.32  0.656   0.605  <== closes
   3.00  24.0  0.16   363.3   3.16  0.493   0.475  <== closes
   3.50  15.0  0.12   257.1   3.56  0.775   0.695
   3.50  15.0  0.16   429.4   4.04  0.901   0.730  <== closes
   3.50  19.5  0.12   265.7   3.85  0.726   0.675
   3.50  19.5  0.16   476.7   3.12  0.776   0.650  <== closes
   3.50  24.0  0.12   300.3   3.42  0.730   0.645
   3.50  24.0  0.16   489.0   2.39  0.575   0.510  <== closes
   4.00  15.0  0.12   140.7   4.24  0.926   0.765
   4.00  15.0  0.16   361.1   4.93  1.172   0.875  <== closes
   4.00  19.5  0.12   551.7   3.66  1.008   0.780  <== closes
   4.00  19.5  0.16   652.7   4.49  0.947   0.740  <== closes
   4.00  24.0  0.12   547.3   3.51  0.930   0.740  <== closes
   4.00  24.0  0.16   527.5   3.05  0.789   0.635  <== closes
```

0.42-0.88 s, worst case at `vz` = 4.0. So 7.5 s leaves
7.5 - 5.0 - 0.16 - 0.88 = **1.46 s** to settle in the worst case and over 6 s
at a short hold. `EPISODE_LENGTH_S = 7.5`.

The hold is reached by curriculum, not immediately: at a 5 s hold in a 7.5 s
episode roughly 70% of collected experience is standing still, which would slow
the flip's discovery badly. Stages (steps = iteration x 24):

| step | hold_range |
|---|---|
| 0 | 0.1-0.3 s |
| 1000 | 0.3-1.0 |
| 2000 | 0.5-2.0 |
| 3000 | 1.0-3.5 |
| 4000 | **1.0-5.0** |

The END state is the requirement; the ramp only keeps early discovery cheap.

## 3. The plate rests on the ground

`Z0_RANGE` is now **0.01-0.03 m** (plate CENTRE; 0.01 = `PLATE_HALF_THICKNESS`,
so the slab sits exactly on the floor). The old 0.07 floor came from the
KNEELING tuck, whose feet hung ~8 cm below the slab — that posture is gone, and
standing's lowest geoms ARE its feet, so nothing can tunnel. The four
spawn-penetration tests pass unchanged at the new height.

**The honest cost is altitude.** A floor-level launch has less airtime: at
`vz` = 2.5 nothing closes any more (best 334 deg), so `VZ_RANGE` came up from
2.50-3.50 to **3.00-4.00**, and the landing got worse — 2.4-4.9 m/s where it
was 2.4-4.4.

## Launch box re-check

`--box-check`, corners and midpoints of all four ranges (243 cells):

```
# box-check posture=standing bam=True dt=0.005
#   z0 (0.01, 0.03) vz (3.0, 4.0) w0 (15.0, 24.0) launch (0.12, 0.16)
#   z0 grid (0.01, 0.02, 0.03)
#   hold (0.1, 0.2, 0.3) (NOT the env's (1.0, 5.0) — see BOX_CHECK_HOLDS) tuck (1.0,)
  243 cells | rot 48.4-652.7 deg | max landing = 4.93 m/s | short of 360: 93 | over 2.6 m/s: 238 | never landed: 0
  worst by rotation:
    rot=   48.4 land= 2.98 apex=0.521 tilt0=  1.2  z0=0.010 vz=3.000 w0=15.00 launch=0.120 hold=0.10 tuck=1.00
    rot=   70.0 land= 3.01 apex=0.532 tilt0=  1.2  z0=0.020 vz=3.000 w0=15.00 launch=0.120 hold=0.10 tuck=1.00
    rot=   70.6 land= 3.06 apex=0.542 tilt0=  1.2  z0=0.030 vz=3.000 w0=15.00 launch=0.120 hold=0.10 tuck=1.00
    rot=   99.7 land= 3.11 apex=0.554 tilt0=  3.0  z0=0.020 vz=3.000 w0=15.00 launch=0.120 hold=0.20 tuck=1.00
    rot=   99.7 land= 3.11 apex=0.564 tilt0=  3.0  z0=0.030 vz=3.000 w0=15.00 launch=0.120 hold=0.20 tuck=1.00
  worst by landing speed:
    rot=  361.1 land= 4.93 apex=1.172 tilt0=  3.0  z0=0.010 vz=4.000 w0=15.00 launch=0.160 hold=0.20 tuck=1.00
    rot=  176.5 land= 4.77 apex=1.101 tilt0=  3.0  z0=0.030 vz=4.000 w0=15.00 launch=0.140 hold=0.20 tuck=1.00
    rot=  286.6 land= 4.73 apex=1.091 tilt0=  4.9  z0=0.030 vz=4.000 w0=15.00 launch=0.120 hold=0.30 tuck=1.00
    rot=  176.9 land= 4.73 apex=1.091 tilt0=  3.0  z0=0.020 vz=4.000 w0=15.00 launch=0.140 hold=0.20 tuck=1.00
    rot=  285.9 land= 4.68 apex=1.081 tilt0=  4.9  z0=0.020 vz=4.000 w0=15.00 launch=0.120 hold=0.30 tuck=1.00
  DIRECTION: PASS — 0 cells rotate FORWARD (min rot 48.4 deg)
  OPEN-LOOP CLOSURE (information, not a gate): 150/243 cells reach 360 deg = 62%; 0 never land
  LANDING: 2.39-4.93 m/s (operator comfort threshold 2.6 m/s; 238/243 cells over)
```

- **DIRECTION: PASS — 0 of 243 cells rotate forward** (minimum +48.4 deg). That
  stays the one hard gate.
- **Open-loop closure 150/243 = 62%**, up from 37% because `vz` rose. Still
  information, not a gate.
- Landing 2.39-4.93 m/s.

**A note on the hold used for this check.** `--box-check` sweeps
`BOX_CHECK_HOLDS = (0.1, 0.2, 0.3)`, deliberately NOT the env's 1-5 s. The
probe has no balance controller — it freezes the HOME command, and open-loop
standing drifts 23 deg by 1.0 s — so sweeping the env's real hold measures a
TOPPLING robot being flicked: 27 deg of tilt at the flick, rotations of
+-1000 deg, 42 forward cells and 177 that never land. That says nothing about
the launch. The env's long hold exists so the POLICY learns to stand, and a
policy that has learned it presents an upright robot at the flick whatever the
hold lasted. This is the sharpest form of the standing caveat: **the probe
measures the launch, not the skill.**

## Reward mass, re-derived

Episode sums, dt-scaled. With hold H, launch ~0.14 s and a 0.65 s typical
flight, the post-landing settle is S ~ 6.7 - H seconds:

| term | weight | value/step | mass H=1 | H=3 | H=5 |
|---|---|---|---|---|---|
| `flip_progress` | 8.0 | potential | 8.0 | 8.0 | 8.0 |
| `landing` | 4.0 | ~1.0 x S | 22.8 | 14.8 | 5.8 |
| `ready_stance` | **1.0** | ~0.975 x H | **1.0** | **3.0** | **5.0** |

The stance weight STAYS 1.0 and the hold does the work: its mass rises 5-25x
from the hold change alone (0.1-0.3 -> 1.0-5.0), which is the fix for the
observed collapse. Raising the weight on top would have pushed the stance PAST
the landing annuity at the long end — at H=5 the annuity is only 5.8 — and
`landing` has to stay the dominant attractor. Never flipping caps the episode
at 5.0; flipping and landing adds 13.8-30.8 on top.

**Known consequence:** the landing mass swings 3.9x across the hold DR (5.8 at
H=5, 22.8 at H=1), which is noisy credit assignment. If that shows up as
instability, cap the annuity's paying window rather than reaching for weights.

## Flop audit at the new plate height

```
# flop-audit: z0=0.02 tuck=0.75 duration=0.5s dt=0.005 bam=True
 orientation   clr    tilt   drift  trunk_z    pose  height    pre  upright   TOTAL  on?
     upright 0.045     8.4   0.009    0.140   1.000   0.975  0.975    1.000   0.975  yes
   side_left 0.015   100.0   0.012    0.062   1.000   0.000  0.000    0.000   0.000  yes
  side_right 0.015   100.0   0.012    0.062   1.000   0.000  0.000    0.000   0.000  yes
   face_down 0.045    80.9   0.006    0.058   1.000   0.000  0.000    0.000   0.000  yes
     on_back 0.045   102.4   0.011    0.077   1.000   0.006  0.006    0.000   0.000  yes
    inverted 0.005    69.1   0.171    0.056   1.000   0.000  0.000    0.003   0.000  OFF
  upright: pre=0.975 TOTAL=0.975   best flop: pre=0.006 TOTAL=0.000
  WITHOUT the upright factor: upright would win (0.006 vs 0.975)
  RESULT: PASS - upright wins (0.975 vs 0.000)
```

Upright 0.975, every flop 0.000. Unchanged in substance; the trunk heights move
with the plate.


# Second video — the invisible hold and the pre-flick crouch

Two findings from the current policy's video. Both are about pricing, not
physics, so this section carries arithmetic rather than sweeps.

## 1. The 1-5 s hold was invisible in `play`

`play` builds a fresh env with `common_step_counter == 0`, so EVERY curriculum
term evaluates at stage ZERO no matter which checkpoint is loaded. The viewer
showed a 0.1-0.3 s hold while the policy had been trained on 1-5 s — and the
same was true of every other range or weight a curriculum moves (action_rate
-0.05 instead of -0.2, arrival_damping 0.0 instead of -0.05, gentle_landing
0.004 instead of 0.0125, the CoM DR at a fifth of its trained range).

`make_microduck_backflip_env_cfg(play=True)` now fast-forwards every curriculum
to its LAST stage and DELETES the terms — deleting matters, because the
curriculum manager runs every step and would write stage 0 straight back.
Training is untouched. Verified:

| | training (stage 0) | play |
|---|---|---|
| hold | 1.0-2.0 s | **1.0-5.0 s** |
| `action_rate_l2` | -0.05 | **-0.2** |
| `arrival_damping` | 0.0 | **-0.05** |
| `gentle_landing` | 0.004 | **0.0125** |
| CoM DR | +-0.003 | **+-0.015** |
| curriculum terms | 7 | **0** |

Six tests pin it, including one that walks EVERY curriculum term and fails on a
stage-list shape `_apply_final_curriculum` does not understand — so a new
curriculum cannot silently keep its stage-0 value in play.

## 2. The pre-flick crouch: a pricing error at stage 0

The robot crouched before the impulse. The instinct that it was chasing a lower
CoM is right, and the mechanism is ours: a compact body rotates much further at
the same flick — measured directly when the tucked hold "flew" at 1.5-2.2 m/s
and standing did not. So crouching is the policy rationally buying rotation.

**The error was in the mass reasoning.** The previous revision priced
`ready_stance` at H=1, 3 and 5 s — but the crouch is LEARNED at curriculum
stage 0, which was 0.1-0.3 s, where the term's episode-sum mass is 0.1-0.3
against ~30 for flip+landing. The term was sized for the END of the curriculum
and the behaviour is acquired at its START. **Curriculum stages have to be
priced at every stage.**

Both levers were used.

**The hold curriculum now starts at 1.0-2.0 s**, so there is no cheap-crouch
window at any stage:

| step | hold_range |
|---|---|
| 0 | **1.0-2.0 s** (was 0.1-0.3) |
| 1500 | 1.0-3.5 |
| 3000 | **1.0-5.0** |

Affordable now: the first real run reached `landing +1.72` by iteration 279 and
open-loop closure is 62%, so flip discovery is not fragile. What the ramp still
buys is range WIDTH — a narrower early window keeps each episode's launch at a
similar time so the policy can learn what the flick feels like before it has to
handle 5 s of timing uncertainty.

**`ready_stance` weight 1.0 -> 1.15**, bounded at BOTH ends of the curriculum.
Episode sums, dt-scaled, with S = 7.5 - H - 0.16 - 0.88 = 6.46 - H (the
worst-case 0.88 s flight, so the bounds hold everywhere):

| stage | hold H | `ready_stance` | `landing` | `flip` | never-flip cap |
|---|---|---|---|---|---|
| 0 | 1.0-2.0 s | 1.12 - 2.24 | 17.8 - 21.8 | 8.0 | 2.24 |
| 1 | 1.0-3.5 s | 1.12 - 3.92 | 11.8 - 21.8 | 8.0 | 3.92 |
| 2 | 1.0-5.0 s | 1.12 - 5.61 | 5.84 - 21.8 | 8.0 | 5.61 |

The ceiling comes from the LONG end: at H=5 the settle window is shortest and
the annuity is only 5.84, so `0.975 * w * 5 <= 5.84` gives `w <= 1.19`. The
floor comes from the SHORT end, and it is why the hold had to move: at a
0.1-0.3 s hold NO admissible weight makes the term matter (it caps at
0.3 x 1.19 = 0.36).

**The acceptance statement, checked at every stage:**

1. **Collapsing costs more than the rotation it buys.** It forfeits the whole
   stance mass — at worst 1.12, at best 5.61, against 0.1-0.3 before. And the
   rotation a PRE-FLICK crouch buys over tucking AT the flick is bounded: the
   policy can fold at the flick for free (`ready_stance` dies at launch), and
   `flip_progress` is capped at one turn so rotation beyond 360 deg pays
   nothing.
2. **Flipping and landing still beats never flipping**, at every stage: the
   never-flip cap is the stance mass alone (2.24 / 3.92 / 5.61) while flipping
   adds at least 8.0 + 5.84 = 13.84 on top.
3. **`landing` stays the dominant attractor** at every stage: its minimum
   (5.84) exceeds `ready_stance`'s maximum (5.61).

The plate height was NOT touched — it went to the floor in the previous commit
and the video was an older checkpoint.


# Third video — hold as plain DR, and a hold-independent annuity

Two coupled changes, both about pricing rather than physics. No new sweeps.

## 1. The hold curriculum is deleted

`HOLD_RANGE = (1.0, 5.0)` sampled uniformly from step 0. The ramp
(0.1-0.3 -> 1-2 -> 1-3.5 -> 1-5) existed only to keep early episodes from
spending most of their time standing still. It bought little — the first real
run reached `landing +1.72` by iteration 279, so flip discovery was never
fragile — and it cost the ability to see the real behaviour: `play` starts
`common_step_counter` at 0, so a curriculum on the hold meant the viewer showed
0.1-0.3 s whatever checkpoint was loaded.

`EPISODE_LENGTH_S = 7.5` and its derivation are unchanged: 5.0 s hold + 0.16 s
flick + the measured 0.88 s worst-case flight + 1.46 s to settle. The other
curricula (regularizer ramps, CoM DR) stay.

## 2. The landing annuity now pays over a FIXED window

`LANDING_WINDOW_S = 1.4` s after the first terrain contact, latched per env in
`backflip_landing` and cleared on reset like every other buffer.

**The problem it fixes.** The annuity paid per step for "all the time remaining
after touchdown", so its episode mass was:

| hold H | settle window | landing mass (old) |
|---|---|---|
| 1.0 s | 5.46 s | **21.8** |
| 3.0 s | 3.46 s | 13.8 |
| 5.0 s | 1.46 s | **5.84** |

A ~4x different payout for an IDENTICAL backflip, decided by a draw the policy
neither controls nor observes — noise injected straight into the main
attractor's credit assignment — and it made a long-hold episode worth less than
a short one for the same skill. With a fixed window the mass is **5.6 at every
hold**, and 1.4 s is what the worst case can always afford (1.46 s).

Anything past the window pays zero, and the latch does not move once set, so
bouncing back into the air and touching down again cannot restart the annuity.

## The re-derived mass table

| hold H | `ready_stance` | `landing` | `flip` | never-flip cap |
|---|---|---|---|---|
| 1.0 s | 1.07 | 5.6 | 8.0 | 1.07 |
| 3.0 s | 3.21 | 5.6 | 8.0 | 3.21 |
| 5.0 s | 5.36 | 5.6 | 8.0 | 5.36 |

(`ready_stance` = 0.975 x 1.10 x H; `landing` = 4.0 x 1.4.)

**The ceiling did NOT rise.** The hope was that a hold-independent annuity
would lift its floor and free stance headroom. It does not: the binding case
was always the longest hold, and pinning the annuity to the worst case's
affordance pins the ceiling with it. `0.975 * w * 5.0 <= 5.6` gives
**w <= 1.148**, so the weight went 1.15 -> **1.10** for margin — slightly DOWN,
not up. (Lengthening `EPISODE_LENGTH_S` is what would raise it: at 8.0 s the
window could be 1.9 s and w could reach 1.56.)

Acceptance statement, checked at every hold draw: collapsing forfeits the whole
stance mass (1.07-5.36 against 0.1-0.3 under the old first curriculum stage);
flipping and landing adds a hold-independent 13.6 on top, so it always beats
never flipping; and `landing` (5.6) stays above `ready_stance`'s maximum (5.36).

## 3. The lean the mass table could not fix

The user also reported the robot launching **leaning ~35 deg back**. That is
not a mass problem: `ready_stance`'s tilt gate was full-credit below 40 deg, so
a 35 deg lean cost **nothing**. The 40/70 width had been sized for the
short-lived TUCKED hold, whose own equilibrium is pitched 14 deg, and it
survived the revert to standing unchanged.

The gate is now **10/45 deg**, measured through the real function:

| trunk tilt | gate |
|---|---|
| 0 deg | 1.000 |
| 7.3 deg (the measured open-loop drift) | 1.000 |
| 20 deg | ~0.80 |
| 35 deg | ~0.20 |
| 45 deg and beyond | 0.000 |

So a genuinely upright stance still costs nothing, a lean is priced in
proportion, and every measured flop basin (80-126 deg) is still hard-zeroed.
**This is a shape fix, not a mass fix** — which is the only lever left, since
the mass ceiling is set by the annuity and has no headroom.


# Gentler ejection — the vz retune

The user watched the video and said the ejection was too strong. One
measurement pass with `--box-check` over eight candidate ranges; landing speed
is the number they actually care about, so it is the column to read.

| candidate | vz | w0 | t_launch | rotation | **landing m/s** | closure | apex m | direction |
|---|---|---|---|---|---|---|---|---|
| previous | 3.0-4.0 | 15-24 | 0.12-0.16 | 48-653 | **2.39-4.93** | 62% | 0.47-1.19 | PASS |
| A | 2.5-3.2 | 15-24 | 0.12-0.16 | 4-485 | 2.17-4.08 | 28% | 0.34-0.83 | PASS |
| B | 2.2-2.8 | 15-24 | 0.12-0.16 | -110-423 | 1.78-3.75 | 12% | 0.28-0.65 | **FAIL** (2 fwd) |
| C | 2.2-2.8 | 18-27 | 0.12-0.16 | -83-420 | 1.40-3.65 | 12% | 0.24-0.63 | **FAIL** (5 fwd) |
| D | 2.0-2.5 | 18-27 | 0.12-0.16 | -83-395 | 1.45-3.37 | 3% | 0.23-0.53 | **FAIL** (11 fwd) |
| E | 2.3-2.9 | 18-24 | 0.13-0.16 | 137-466 | 1.92-3.69 | 27% | 0.31-0.66 | PASS |
| F | 2.4-3.0 | 15-24 | 0.13-0.16 | 108-483 | 2.00-3.95 | 28% | 0.33-0.74 | PASS |
| **G (chosen)** | **2.2-2.8** | **18-24** | **0.14-0.16** | **98-442** | **1.78-3.65** | **27%** | **0.28-0.64** | **PASS** |
| H | 2.0-2.6 | 18-24 | 0.14-0.16 | 97-403 | 1.54-3.50 | 10% | 0.26-0.58 | PASS |
| I | 2.1-2.7 | 18-24 | 0.15-0.16 | 120-440 | 1.59-3.57 | 15% | 0.24-0.60 | PASS |

## The direction gate binds from BOTH sides

This is the finding that shaped the answer. It was already known that
`w0` above ~24-27 rad/s overdrives the sole contact and comes out FORWARD. It
turns out a **low `vz` with a low `w0` and a SHORT flick does the same**: B, C
and D all fail the gate, and their forward cells are all at the
`vz` 2.0-2.2 / `w0` 15-27 / `t_launch` 0.12 corner (vz 2.2, w0 15,
t_launch 0.12 measures **-110 deg**).

So simply lowering `vz` does not work. Trimming `t_launch`'s low end from 0.12
to 0.14 and `w0`'s from 15 to 18 is what let `vz` come down to 2.2 while
keeping the gate — compare B (FAIL at 0.12-0.16) with G (PASS at 0.14-0.16) on
the same `vz`.

## Chosen: G

`vz` 2.20-2.80, `w0` 18-24, `t_launch` 0.14-0.16, `z0` unchanged at 0.01-0.03.

| | before | after |
|---|---|---|
| landing | 2.39-4.93 m/s | **1.78-3.65 m/s** |
| cells over the 2.6 m/s comfort threshold | 238/243 | **191/243** |
| apex | 0.47-1.19 m | **0.28-0.64 m** |
| rotation | 48-653 deg | **98-442 deg** |
| open-loop closure | 62% | **27%** |
| direction | PASS | **PASS** (min rot +98 deg) |

**How far down does this get the landing speed?** The maximum falls from 4.93
to **3.65 m/s** and the minimum from 2.39 to **1.78** — the gentlest cells are
now well under the ~2.6 m/s threshold, where before none were. Nearly a fifth
of the box is now under it (52 of 243 cells) against 5 of 243. The apex halves,
which is the part that shows up in a video as "too strong".

**Why G and not H.** H is gentler still (1.54-3.50) but drops open-loop closure
to 10%, i.e. 25 of 243 cells. That is a much bigger leap from what has already
trained successfully than G's 27% — the same fraction that was accepted at 37%
earlier and reached `landing +1.72` by iteration 279. G is the gentlest range
that keeps both the direction gate and a closure fraction in the region already
shown to train. **H and I are measured and available** if the user wants to go
further; the ladder is in the cfg docstring so the next step needs no new
measurement.

Nothing else changed: hold sampling (1-5 s uniform), the annuity window
(1.4 s), the tilt gate (10/45 deg) and all weights stay as they were.


# The 2026-09-08 NaN crash — diagnosis

Run `2026-09-08_16-26-30_backflip` (wandb `thd7grux`) died ~2 min in with
rsl_rl's `check_nan` reporting NaN in the **critic** group, while
`Episode_Termination/nan_state` read exactly **0.0000** — the env's own guard
never fired. Two hypotheses were tested.

## Hypothesis A: contact overflow. REFUTED, measured.

`nconmax = 50`. Peak simultaneous contacts on CPU MuJoCo, full-collision robot
with the plate in the scene, 5 s per case:

| situation | peak ncon | settled mean / max |
|---|---|---|
| standing on the plate (HOLD) | **16** | 11.3 / 16 |
| fallen on its back | 5 | 5.0 / 5 |
| fallen face down | 10 | 7.1 / 9 |
| fallen on its side | 8 | 5.0 / 5 |
| inverted | 6 | 3.5 / 4 |
| dropped flat from 0.4 m | 10 | 6.5 / 9 |
| fallen on back, tuck commanded | 8 | 5.0 / 5 |

Plus 4 persistent plate-floor contacts once `z0` puts the slab on the ground,
so **~20 worst case against a budget of 50** — 2.5x headroom. The robot has
only 13 collision geoms, so it cannot generate 50 contacts. Not the mechanism.

A test now pins `nconmax >= 2x` the measured worst case, so a future collision
or prop change cannot quietly eat the margin.

## The ordering hypothesis: REFUTED, by reading mjlab.

`ManagerBasedRlEnv.step` computes observations **after** `_reset_idx`
(`manager_based_rl_env.py`: terminations -> rewards -> `_reset_idx` ->
`sim.forward()` -> `observation_manager.compute()`). So a firing `nan_state`
resets the env and the obs handed to rsl_rl comes from the clean reset state.

That is what makes the `nan_state = 0.0000` clue decisive rather than
confusing: **anything `robot_state_is_nan` checks would have terminated and
returned clean obs.** The NaN necessarily arrived through a quantity it does
not check.

## Hypothesis B: an unprotected critic obs path. CONFIRMED as the only
## remaining mechanism, with a documented precedent.

`robot_state_is_nan` checks the robot's `joint_pos`, `joint_vel`, root
`pos`/`quat`/`lin_vel`/`ang_vel`, **and contact forces — but only for the
sensors named in its `sensor_names` parameter.** The backflip cfg passed
**none**:

```
nan_state params: {}          # before
```

Meanwhile its critic group contained three sensor-derived terms straight from
mjlab's unsanitized observation module:

```
  foot_air_time          mjlab.tasks.velocity.mdp.observations.foot_air_time
  foot_contact           mjlab.tasks.velocity.mdp.observations.foot_contact
  foot_contact_forces    mjlab.tasks.velocity.mdp.observations.foot_contact_forces
```

`foot_contact_forces` is `sign(F) * log1p(|F|)` over a contact-sensor force.
MuJoCo resolves a degenerate contact into an inf/NaN impulse **a step before**
the integrated state goes bad, so the force is non-finite while
`joint_pos`/root state are still clean: the guard reads 0, and the NaN goes
straight into the critic group and into `check_nan`.

**This is a recurrence, not a new failure mode.** The microduck velocity cfg
has both passed `sensor_names` and swapped these three terms for `_safe`
variants since its own crash of 2026-08-21 (Velocity2-Rough-Backlash), with
the mechanism written in its comments. The backflip env builds on **mjlab's**
`make_velocity_env_cfg`, not the microduck one, so it inherited neither
protection. Every other env in the family was covered; this one was not.

Two secondary paths in the backflip's own critic terms, both real and both now
closed: `ball_pos_in_base` / `ball_vel_in_base` rotate into the robot's base
frame, so a degenerate robot quaternion propagates through; and the phase mask
was applied as a MULTIPLICATION, so `0 * inf` would have *created* a NaN once
the plate went GONE.

## What changed

1. `nan_state` now receives **all three** contact sensors
   (`feet_ground_contact`, `self_collision`, `robot_ground_contact`), so a
   force blow-up terminates the env.
2. The three sensor-derived critic terms are swapped for the microduck `_safe`
   variants — the same loop the velocity cfg runs. This is the load-bearing
   fix: because obs is computed after the reset, terminating is not enough on
   its own for a value that is non-finite *in the step it is read*.
3. `backflip_plate_pos_obs` / `_vel_obs` sanitize **before** the mask
   multiplies (killing the `0 * inf` path), and `backflip_phase_obs`
   sanitizes too.
4. `nconmax` keeps 50, now with the measurement in the comment rather than an
   analogy to the ball-kick env.

Six tests, two of which were verified to fail when the fix is reverted: one
asserts `nan_state` watches **every** sensor the scene defines, so adding a
sensor cannot silently reopen the hole; one asserts **no** critic term comes
from mjlab's unsanitized sensor module (with `foot_contact` whitelisted as a
0/1 flag); one feeds `inf` and `NaN` plate states through both plate obs terms
in both phases and requires finite, zeroed output.

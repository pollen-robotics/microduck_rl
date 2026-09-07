# Microduck Backflip — Design

Date: 2026-09-03
Branch: `feat/backflip`
Status: approved design, ready for implementation planning

> **AMENDMENT 2026-09-07 — the hold posture is TUCKED, not standing.**
> This design specified that the robot *stands* on the plate through HOLD, and
> the env was built that way. It is wrong, and the measurement is what says so:
> re-measured from a standing spawn under BAM actuators (~700
> direction-checked cells over `vz` ∈ [2, 4] × `ω0` ∈ [3, 36] × `t_launch` ∈
> [0.08, 0.15], with and without tucking at the flick), **no launch setting
> closes 360° below the ~2.6 m/s hardware landing limit**, and the DR box that
> had been configured rotated a standing robot **forward** (to −275°,
> face-down, orientation-verified). A standing robot has a ~11 cm CoM and a
> large pitch inertia, so the flick overdrives the sole contact instead of
> tipping it over its heels; tucked (~3 cm CoM) the same class of box closes
> 393–484° at 1.68–2.51 m/s. The launch envelope in the original document had
> been measured from a *tucked* probe spawn all along, so the spec, the spawn
> and the reward had simply never agreed on one posture.
>
> Everything else in this document stands. The concrete deltas are marked
> **[AMENDED]** below: the robot spawns folded, `ready_stance` pays for holding
> the tuck rather than for standing, the launch box is narrower and whole-box
> verified, and the `z0` DR tail stops at 0.225 m. The landing target is
> unchanged — the duck still lands on its feet, standing.
> Evidence: `docs/backflip_envelope_results.md`, section "Tucked hold".

> **AMENDMENT 2 — 2026-09-07 — the launch is LOWER and GENTLER, and the plate
> still cannot lie on the ground.**
> The user watched the env for the first time and reported it "launched far too
> hard and too far", and asked for the plate to sit ON the ground. The retune
> is done; the ground-level request is measured to be impossible for this hold
> pose, and that is a design answer rather than a refusal:
> - **Retuned [AMENDED]:** `z0` 0.10-0.20 → **0.07-0.09**, `vz` 2.00-2.10 →
>   **1.90-2.00**, `ω0` 21-23 → **23-24**, `t_launch` 0.12-0.14 →
>   **0.12-0.13**. Whole-box: rotation 372-457° (was 393-476), apex 0.29-0.40 m
>   (was 0.52-0.63), worst landing 2.15 m/s (was 2.53). The plate is less than
>   half as high. Over-rotation is now a defect to minimise; ~85° of the
>   remaining spread is irreducible DR.
> - **The `z0` DR tail is deleted [AMENDED]:** upward it crosses the landing
>   limit, downward the range is only 2 cm wide.
> - **The plate cannot rest on the ground with this hold [AMENDED]:** the
>   kneeling tuck's feet hang ~8 cm below the surface it sits on, so a plate
>   top under 0.08 m puts them through the floor (41° tilt, 5.5 cm slide). A
>   solid ground-resting block is worse; a wider pad does not help. A
>   feet-flat squat DOES sit on a ground-level plate but does not fly — 420
>   cells, none closing 360° under 2.6 m/s, softest 3.09 m/s.
> - **The "plate not under the feet" complaint was a real bug [AMENDED]:**
>   mjlab never resets before the viewer's first episode, so `play` ran a whole
>   episode on the compiled default — robot at 0.12 m, plate at 0.15 m through
>   its body. Entity init states and the lazy launch params are now coherent.

## Problem

Microduck (~800 g, ~25 cm, 14 XL330 servos) cannot generate the vertical
impulse a 360° backflip needs. The trick is therefore a *hand-off*: a human
holds the robot, tosses it upward with a backward flick, and the robot's own
job is the airborne half — tuck to speed the rotation up, extend to slow it,
and land on its feet without breaking itself.

The environment models the hands as a **launcher plate**: a prop the robot
waits on, which lifts and flicks it and then leaves the scene, so the robot
lands on bare floor. **[AMENDED]** the robot waits on it *tucked* (folded, chin
in), not standing — see the amendment at the top.

## Success criterion

An episode succeeds when the robot accumulates ≥ ~330° of backward pitch while
airborne and then settles on its feet, upright, near standing trunk height,
with low residual angular rate — without a landing impact that would damage
hardware.

## Non-goals

- No standing (unassisted) backflip. The plate supplies the lift.
- No front flip / no rotation-direction command. Backward only.
- No training run in this branch. The deliverable is env + tests +
  measurements + a green smoke test.

## Decisions taken (with the alternatives that were rejected)

| Decision | Chosen | Rejected because |
|---|---|---|
| Ejector model | Real contact body the robot stands on | A velocity teleport or `xfrc` shove gives the robot nothing to push against, so it never learns a push-off and the launch ignores what the legs do |
| Launch timing | Variable hold (0.1–1.0 s), no trigger bit — reactive | Firing at t=0 requires the operator to press and throw in the same instant; a trigger bit adds runtime plumbing and a real-time sync the hands must honor |
| Spin source | Plate supplies lift *and* backward flick | Pure vertical ejection makes the robot generate all angular momentum in a very short push window — the part most likely never to converge |
| Launch height | 0.10–0.20 m, 0.30 m as a DR tail | Higher launches buy airtime, but the landing impact is the hardware risk |
| **[AMENDED]** Hold posture | **Tucked** (folded, chin in), at the measured 0.029 m resting trunk height on the plate top | Standing was the original choice; measured, it cannot close a safe flip at any launch setting, and it rotates the robot forward inside the configured box |
| **[AMENDED]** Launch height, revised | 0.10–0.20 m, **0.225 m** as the DR tail | Measured landing speed at the box's worst corner: 2.51 m/s at `z0`=0.20, 2.56 at 0.225, 2.61 at 0.250, **2.80 at 0.300** — the original 0.30 m tail is over the damage threshold |
| **[AMENDED]** Flick duration | 0.12–0.14 s, not 0.08–0.15 | A *short* flick is the violent one: `t_launch`=0.08 lands at up to 3.97 m/s. The operator's flick duration is not free DR |

## 1. Launcher

New prop `src/mjlab_microduck/robot/microduck/launcher.xml`, registered as a
second scene entity `plate` alongside `robot` (the `ball` in the BallKick env
is the precedent for a non-articulated prop entity).

Geometry: box plate ≈ 0.18 × 0.18 × 0.02 m. Two joints: a vertical `slide` and
a lateral-axis (`0 1 0`) `hinge` for the pitch flick. Joint names carry the
`passive_` prefix for consistency with the repo convention; they belong to a
separate entity, so no robot-scoped selector can reach them, but new
`passive_*` regexes in robot selectors must stay narrow regardless.

**The plate is kinematically prescribed, not dynamic.** An `EventTermCfg` with
mjlab's `mode="step"` (fires every env step, on all envs) rewrites its root
pose and velocity every control step from the sampled launch parameters. The
consequences are all wanted:

- it does not sag under the robot's weight during the hold,
- it does not recoil when the robot pushes off (a hand is effectively
  infinitely heavier than an 800 g duck), so the push-off transfers full
  reaction,
- the launch is exactly reproducible from the sampled parameters.

### Phase machine (per env)

| Phase | Plate | Robot |
|---|---|---|
| HOLD, duration `t_hold` | parked at `z0`, zero velocity | **[AMENDED]** waits on it *tucked*; no cue that launch is coming |
| LAUNCH, `t_launch` ≈ 0.08–0.15 s | prescribed constant *acceleration* — linear velocity ramps 0 → `vz`, pitch rate 0 → `ω0` | rides it; may add energy by extending legs |
| GONE | teleported to `BACKFLIP_GONE_POS`, velocities zeroed | ballistic; lands on bare floor |

**[AMENDED]** the GONE row originally said "teleported to z = −3". That parks
the plate *inside* MuJoCo's infinite floor half-space, i.e. at maximal
penetration, which measurably burned 4 contacts per env for most of every
episode and had the solver ejecting the 50 kg plate at 59 m/s within a control
step. It is now parked **above and beside** the floor at `BACKFLIP_GONE_POS`
= (5, 5, 5) m; see that constant's comment in `mdp.py` for the measurement.

The launch prescribes an acceleration ramp, not a step change in velocity. A
plate that jumped straight to `vz` would drive the contact solver to accelerate
the robot to launch speed within a single step — an impulsive, effectively
infinite-jerk kick through the legs, and a |a_z| spike that has nothing to do
with the hand it is meant to model. The ramp is what a hand does, and
`a = vz / t_launch` keeps contact continuous throughout.

Release (LAUNCH → GONE) is primarily **time-based**: the plate is removed the
step the ramp completes. Removal cannot wait for contact loss, because a plate
coasting at constant `vz` while the robot decelerates under gravity keeps
pushing indefinitely. Loss of foot↔plate contact is the *secondary* trigger,
covering the case where the robot pushes off hard enough to leave the plate
before the ramp ends.

State lives in lazily-created per-env buffers on `env`, updated under a step
guard — the pattern `_roulade_state` / `_update_roulade_accum` already use.
All buffers are reset per-env on episode reset: phase state must not accumulate
across resets, the same rule DR follows.

### Randomized per episode

`z0` ∈ [0.10, 0.20] m (**[AMENDED]** DR tail to 0.225, not 0.30), `t_hold`,
`t_launch`, `vz`, `ω0`, plus
small lateral and yaw asymmetry so the policy cannot assume a perfectly clean
toss.

### Flip envelope — measured before training

A 360° needs airtime × spin rate, and those trade directly against landing
impact. Illustrative free-flight numbers from a 0.15 m launch: `vz` = 3 m/s
gives ~0.65 s of air, a ~0.6 m apex and a ~3.4 m/s landing; `vz` = 2 m/s gives
~0.43 s of air and requires ~14 rad/s of sustained spin. Neither is a number to
build on, so implementation step 1 is a headless sweep of `vz` × `ω0` × tuck
(**[AMENDED]** and the sweep must run from the env's ACTUAL spawn posture, and
be accepted WHOLE-BOX — every corner and midpoint closing, not the best cell.
The measured box is `vz` ∈ [2.00, 2.10], `ω0` ∈ [21, 23], `t_launch` ∈
[0.12, 0.14], worst cell 393.6° at 2.51 m/s.)
depth on the actual model, recording rotation achieved and landing speed. The
measured envelope sets the DR ranges. Guessing heights across model revisions
is exactly the failure the standup env logged.

## 2. Episode structure and curriculum

Episodic policy, triggered by a policy switch like roulade/sitstand. Episode
length 4.0 s: hold + ~0.6 s flight + ≥ 1.5 s landing settle.

Two spawn buckets (roulade's reverse curriculum, inverted for a flip):

- **On-plate** (**[AMENDED]**: tucked, not standing) — the whole task, start
  to finish.
- **Mid-flight** — spawned already airborne at 90°–330° through the flip with
  matching backward ω and downward velocity, plate already gone, rotation
  accumulator pre-set to the spawn angle. Without this, the frontier of the
  task — the landing — never receives on-policy data.

The ratio starts at ≈ 50/50 and shifts toward on-plate as the landing
consolidates.

Curriculum stages, each phase-aligned with what the policy has actually
learned:

1. Generous `vz`/`ω0` (the hands do most of the work), short hold,
   regularizers ≈ 0.
2. Launch ranges widen and lower once the flip closes.
3. Hold window widens 0.1 s → 1.0 s once reactive detection works.
4. Smoothness and impact taxes ramp in only after the flip exists.

## 3. Rewards

- **`flip_progress`** — the single dense signal. Paid increments of the
  max-so-far accumulated *backward* pitch: potential-based, so a full flip pays
  2π worth in total and camping pays zero per step. Gated on **airborne** (no
  terrain contact) — the mirror of roulade's support gate, and what forecloses
  flopping onto the back and rolling. The paid rate is capped, so spinning
  harder than the envelope requires buys nothing.
- **`landing`** — annuity gated on the rotation frontier reaching ~330° AND
  feet-only contact. Multiplicative composite (upright × height ≈ STAND_Z ×
  low |ω|), paid per step, slew-limited so arriving early is not a jackpot.
- **`impact`** — |a_z| penalty, weighted harder than roulade: the landing is
  the hardware risk. This is what buys knee/ankle absorption; a bespoke "bend
  your knees" term would be gamed.
- **`ready_stance`** — **[AMENDED]** small `pose × height` term active only
  during HOLD, paying the robot to hold the TUCK on the hands instead of
  squirming off before the flick (originally: upright + standing height). No
  upright factor: the tuck's own equilibrium is a trunk pitched 12–15°
  (measured), so an upright term would fight the pose the term exists to hold.
  It pays a crouched robot per step, which is the shape AGENTS.md warns about;
  it is legitimate here because the tuck is the *good* state and the paying
  window is closed by the plate's prescribed schedule rather than by anything
  the policy does, so it cannot be camped.
- **Regularizers** — `action_rate` / `joint_torque_rate` from ≈ 0, ramped in
  after the flip exists; motion-blockers (body angular velocity, angular
  momentum) near zero throughout, since a backflip *is* a large
  angular-velocity event; leg/neck limit-proximity penalty.

Sign convention is audited per term against the two penalty styles in `mdp.py`
(mjlab-base cost → negative weight; self-negating `*_penalty` → positive
weight). Per-run check: every `Episode_Reward/<penalty>` ≤ 0.

## 4. Observations, DR, registration

- **Actor obs stays 61D and plate-blind.** The real robot has no launcher
  sensing; it feels the toss through its IMU, which is the point of the
  reactive design. The command block is zero-padded with tiny non-zero sampling
  ranges so its input weights stay alive.
- **Critic** sees plate state and phase (asymmetric actor-critic, as the
  ball-blind kick env does).
- **DR stack mirrored from standup/roulade** for sim2real parity:
  `expand_bam_friction_fields`, encoder bias, IMU misalignment, CoM / mass /
  armature / joint-friction DR, NaN-guard termination with sensor names,
  `_safe` critic terms. Velocity pushes off (a shove mid-flip is incoherent).
- **Robot cfg**: `MICRODUCK_STANDUP_ROBOT_CFG` (full collisions).
- **Symmetry mirror loss ON** — the flip is sagittally symmetric, and the
  mirror loss fights sideways collapse.
- **Registration**: `Mjlab-Backflip-Flat-MicroDuck` plus the `-Backlash-`
  variant on the matching allcollisions model, with its own
  `MicroduckBackflipRlCfg` and `experiment_name`.

## 5. Testing

`tests/test_backflip_cfg.py` (CPU, no GPU):

- joint indices resolve on the actual model (via the `_servo_joint_*` helpers,
  never hardcoded),
- reward weights carry the intended sign per term,
- gates open and close where expected (airborne gate, landing frontier gate,
  HOLD-only stance term),
- actor obs is 61D and contains no plate term,
- the `plate` entity is present in the scene cfg,
- phase state machine unit tests: HOLD → LAUNCH → GONE ordering, release on
  contact loss, timeout backstop, and per-env reset with no cross-episode
  accumulation.

Then: a 64-env / 5-iteration smoke test, and an ONNX export through
`scripts/export.py` (normalizer baked in — never a hand conversion).

## 6. Risks

- **Reactive launch detection is the riskiest assumption.** If the policy
  cannot reliably distinguish "the hand fired" from IMU alone, the fallback is
  a trigger bit carried in an existing command slot. That is a decision point
  to bring back to the user, not a silent change.
- **The envelope may not close safely.** If the sweep shows 360° only completes
  above a landing speed that risks the hardware, report the numbers and stop —
  do not train something that breaks the robot.
- ~~The prescribed-kinematic plate depends on a per-step event.~~ **Resolved
  during spec review**: mjlab's `EventTermCfg` supports `mode="step"`
  ("every environment step, unconditionally on all envs"), which is exactly the
  hook the phase machine needs. No interval-timer workaround required.

The plate carries a free joint (like the `ball` prop) rather than
slide + hinge joints, because the per-step prescription only needs
`write_root_link_pose_to_sim` / `write_root_link_velocity_to_sim` — the same
two calls the ball reset already uses — and a free joint makes the teleport to
z = −3 trivial. Its mass is set large enough that any intra-step deviation from
the prescribed motion, before the next step rewrites it, is negligible.

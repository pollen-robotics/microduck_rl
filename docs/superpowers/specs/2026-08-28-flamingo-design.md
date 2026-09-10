# Flamingo — balance on one foot (design)

Date: 2026-08-28. Status: stage 1 (balance-only feasibility) approved; stage 2 (full
stand → lift → hold → stand cycle) designed, not yet built.

## Goal

A policy that stands on the right foot (left foot ≥ 3 cm off the ground), holds a validated
one-foot pose for several seconds under training-style pushes, and — stage 2 — gets there from
a two-foot stand and returns to it. Left-foot variant later by mirroring.

## What we know (measured in `notes/tools/duck_pose.py`, 2026-08-28)

- With the XL330 at firmware kP 200 (BAM), joint stiffness ≈ 0.56 N·m/rad < gravity's
  ≈ 1.07 N·m/rad about the ankle: **no pose is passively stable, not even STAND**. Balance is
  always a control task; the goal pose only needs to be a geometric equilibrium.
- Lateral CoM authority is the bottleneck (no ankle roll; hip roll ±22°). With a **3° margin on
  every joint** the best pose has the CoM **1.21 cm** inside the stance sole's contact hull and
  holds 4 s from a standstill with an ideal actuator:

  | joint | rad | | joint | rad |
  |---|---|---|---|---|
  | left_hip_yaw | 0.000 | | right_hip_yaw | +0.386 |
  | left_hip_roll | +0.300 | | right_hip_roll | −0.334 |
  | left_hip_pitch | +1.200 | | right_hip_pitch | +0.258 |
  | left_knee | −0.800 | | right_knee | +0.005 |
  | left_ankle | +0.800 | | right_ankle | −0.253 |
  | neck_pitch | +0.349 | head_pitch +0.350 | head_yaw −1.500 | head_roll 0.000 |

  Placed with the right sole flat: base roll ≈ +22.3°, pitch ≈ 0, trunk tilt 24°, trunk z ≈ 0.124 m,
  swing foot 13.6 cm up. (`notes/poses/flamingo_right.json`.)
- The transition cannot be scripted open-loop; in double support the CoM only moves toward the
  stance foot with the *opposite* stance hip roll ("scissor"). The policy must discover the
  dynamic weight transfer (stage 2).
- Falling toward the swing side is a soft failure (the free foot lands); falling toward the
  stance side is a hard one (no catch). Rewards encode this asymmetry.

## Decisions

| question | decision | why |
|---|---|---|
| One policy or several | one env, one policy; staging via spawn buckets and curricula | repo pattern (standup, sitstand); runtime hot-swaps whole tricks only |
| Stage 1 scope | spawn **in** the pose (noised), all-zero 13-D command, hold ≥ 5 s under pushes | cheapest test that the pose is stabilisable with BAM servos + delays (~$1) |
| Stage 2 scope | phase clock in the twist slots (GroundPick idiom): stand 1.0 s → shift+lift 1.0 s → hold H s → lower 1.0 s → stand rest; in-pose spawn bucket kept ≥ 20 % forever; H curriculum 1 → 4 s | "learns the start, never the last mile" fix; symmetric up/down reward |
| Robot model | `robot_allcollisions.xml` (standup robot cfg) | falls must be physical |
| Command slots | stage 1 all zeros; stage 2 `[cos φ, sin φ, 0]` in twist, head/body zero-padded | daemon contract, tiny non-zero sampling keeps weights alive |
| Joint limits | `joint_pos_limit_proximity` penalty, target pose already ≥ 3° from limits | AGENTS.md: stock `dof_pos_limits` is toothless; sim2real at the limits |
| Terminations | `nan_state`; hard fall = trunk tilt > 60° for > 0.5 s **or** trunk/head ground contact; swing-foot touchdown is NOT a termination | soft vs hard failure asymmetry; wobbles must play out |
| Episode | 6 s (stage 1), 10 s (stage 2) | hold ≥ 5 s is the success criterion |
| Pushes | `push_by_setting_velocity` x/y, interval 3–6 s, magnitude 0 → 0.08 (it 300) → 0.15 (it 600) → 0.25 (it 1000) | standup's ramp; do not disturb discovery |
| DR | inherit the velocity/standup block (CoM ±3→15 mm, head CoM, mass, armature, BAM friction, foot friction 0.7–1.3, IMU misalignment 6°, encoder bias, obs noise/delays, BAM voltage 6.5–8.2 V, delay 3–6) | sim2real recipe that transferred |

## Stage 1 env (`Mjlab-Flamingo-Flat-MicroDuck`)

**Spawn** (`mode="reset"`, new event `set_flamingo_state`): root at the placed pose (z 0.124 + U(0, 0.01),
roll 22.3° ± 3°, pitch 0 ± 3°, random yaw), joints = FLAMINGO_POSE + N(0, 0.05 rad) on servo joints,
qvel = 0. Optional `standing_prob` bucket (STAND pose, z 0.125) for stage 2; 0 in stage 1.

**Rewards** (weights are starting points; task mass ≈ 10, regularisers phased in):

| term | fn | weight | notes |
|---|---|---|---|
| `com_over_stance_foot` | `com_over_support_foot` on `right_foot` site, std 0.02, ungated | +3.0 | the balance signal; target biased 3 mm toward the swing side (safe failure direction) |
| `stance_foot_grounded` | `single_foot_grounded_reward` (right) | +1.0 | pin the stance foot (anti-hop) |
| `swing_foot_clear` | new `foot_clearance_target`: exp(−((z_left − 0.05)/0.03)²), gated on stance contact | +1.5 | keep the free foot up, but not by kicking |
| `swing_foot_touch` | new `foot_contact_penalty` (left foot contact) | −0.5 (self-negating → +0.5 if implemented as ≤0) | soft failure: cheap but not free |
| `pose_flamingo` | `pose_target_match` (target = FLAMINGO_POSE, legs + head, std 0.5) | +1.5 | stay near the equilibrium, generous std |
| `gravity_flamingo` | new `projected_gravity_target`: exp(−‖g_b − g_target‖²/0.15²) | +2.0 | the trunk lean *is* the pose; plain upright would fight it |
| `stillness` | `posture_stillness`-style joint-vel Gaussian, gated on stance contact | +1.0 | quiet hold |
| `stance_side_tilt` | new `lateral_tilt_penalty`: −max(0, g_b,y·sign − 0.15)² (roll toward stance side beyond the pose) | −4.0 | hard-failure direction taxed early and steeply |
| `action_rate_l2` | mjlab | −0.1 → −0.5 (it 400) → −1.0 (it 800) | smoothness after discovery |
| `joint_torque_rate_l2` | | 0 → −1e-3 (it 600) | |
| `joint_limit_proximity` | `joint_pos_limit_proximity` (margin 0.1 rad) | −1.0 | off the limits |
| `body_ang_vel` | | −0.05 | low: balance needs motion |
| `self_collisions` | | −1.0 | |

Sign convention per AGENTS.md: penalty fns that already return ≤ 0 get **positive** weights;
every `Episode_Reward/<penalty>` must be ≤ 0 in wandb.

**Success metric** (logged): fraction of episodes with single support (right contact, left clear)
for ≥ 5 s and no hard fall. Target for stage 1: > 70 % at push 0.15 m/s.

**Curricula** (steps = it × 24): push magnitude (above), action-rate weight, com std 0.03 → 0.02
at it 500.

**PPO**: `(512,256,128)` elu, obs-norm on, lr 1e-3 adaptive, 24 steps/env, 4096 envs,
`max_iterations` 500 (stage 1), `experiment_name="flamingo"`, wandb project `mjlab_microduck`.

**Tests** (CPU): pose within joint ranges with ≥ 0.05 rad margin; every reward term present with
the intended sign; `fell_over` replaced by the hard-fall termination; obs layout 61 with
`head_command` (4) and `body_command` (6) zero-padded; spawn event writes z/roll within ranges.

**Plan B if stage 1 plateaus**: (1) reduce hold target: swing foot 2 cm instead of 5; (2) widen
com std / remove stance-side tilt tax during the first 200 iterations; (3) allow the pose to drift:
drop `pose_flamingo` weight to 0.5; (4) if nothing balances at push 0, the pose margin is too small
for the servo delay — revisit the 3° joint margin.

## Stage 2 (after stage 1 succeeds)

Add `FlamingoPhaseCommand` (twist slots `[cos φ, sin φ, 0]`, `randomize_phase=False` for the
standing bucket, φ set to the hold segment for the in-pose bucket), segment envelope
`stand → shift/lift → hold(H) → lower → stand`, rewards gated by segment (`kick_engagement`
pattern: single-support terms only in lift/hold, two-foot `feet_grounded` + HOME pose in stand
segments, symmetric lift/lower shaping so "down" pays like "up"), spawn mix curriculum
70/30 → 20/80 (in-pose / standing), H 1 → 4 s, `recovery_success`-style one-shot bounty for a
verified hold ≥ H followed by a clean two-foot stand.

## Out of scope

Left-foot stance (mirror later), roller model, backlash twin, real-robot deployment.

## Stage 2 as built (2026-08-29, `Mjlab-FlamingoCycle-Flat-MicroDuck`)

Built as a **controller-driven posture command** instead of the fixed phase clock above, because
the user wants to choose when to lift and when to come back (a button on a controller) and wants
the robot to give up gracefully when a push is too hard.

| item | decision |
|---|---|
| command | twist slots `[flag, side, 0]`: flag 0 = two feet, 1 = one foot; side +1 = right stance (left lifted), −1 = left stance. Obs = raw command. Runtime writes flag/side. |
| both feet | one policy, side in the command; left pose = exact mirror of the right one (swap legs, negate leg joints, negate head yaw/roll — verified 1.20 cm CoM margin both sides, `notes/poses/flamingo_left.json`) |
| internal blend | `FlamingoCommand` slews α ∈ [0,1] toward the flag at 1/1.5 s (SitStand idiom). Shaping targets are blends: CoM target midpoint(feet) → stance foot, pose HOME → FLAMINGO(side), gravity upright → pose lean |
| sequencing | by α-gates: swing-foot contact +1 for α<0.4, −1 for α>0.9; swing-foot clearance asked for α>0.6; stillness only when |flag−α|<0.02. Same in reverse when lowering |
| dwell | flag resampled every 2.5–5 s (episode 10 s); flamingo prob 0.6; side re-drawn only from STAND |
| spawn | in the pose of a side drawn by the event (`in_pose_prob` 0.6 → 0.4 (it 600) → 0.3 (it 1200)) or standing at HOME; flag drawn independently → hold / lower / lift / stand all trained. Reset events run BEFORE the command reset in mjlab, so the event pins the side into the command term |
| fallback | swing-foot touchdown is a −0.5/step tax, never terminal; staying upright is worth far more → step down, then re-lift or wait for the flag to drop |
| pushes | 0 → 0.08 (it 500) → 0.15 (it 1000) → 0.25 (it 1500) |
| eval | `duck-result <run> --cycle`: right/left cycles, pushes during the hold (swing side, stance side, forward), a 0.3 m/s too-hard push, lower from a static hold; reports `hold_single_support`, `hold_swing_clear_frac`, `stand_two_feet`, `final_upright` per rollout |

Review fixes before the first cycle run (2026-08-29 01:10): the CoM rewards now use the
whole-robot `subtree_com` (mjlab's `root_com_pos_w` is the trunk body's own CoM, centimetres
off with the head and a lifted leg — stage 1 trained on that proxy and still balanced); the
rewards use a latched ±1 stance side while the OBSERVED side slot is zeroed half the time in
STAND so the runtime's `[0,0,0]` idle command is trained; swing-foot clearance target 0.10 ± 0.06
(0.05 ± 0.03 contradicted the pose's 0.17 m foot); the success diagnostic carries weight 1e-3
(mjlab skips weight-0 terms entirely).

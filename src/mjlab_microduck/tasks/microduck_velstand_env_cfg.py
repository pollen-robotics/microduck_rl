"""Microduck VelStand environment: walking + protective fall + recovery, one policy.

CURRENT POLICY (Hub v7, 2026-10-05): wandb j4i6yoq2 @ model_1250, trained from this
file at commit 53fb7d1 — warm-started from fhathosb@3750 with ENABLE_BODY_CONTROL.
Lineage, launch commands and eval numbers: docs/velstand_policy.md.

BODY CONTROL (2026-10, ENABLE_BODY_CONTROL): body_pose roll/pitch (± z) commanded
only while standing (StandingGatedPoseCommand: exact zero while walking, as
deployed). Tilt-only frames are distilled from alpha_stand, z frames left to PPO,
both taken off the walk anchor (distill.py body routing); standing-gated tracking
reward + command-relative upright + relaxed leg-pose stds. Roll/pitch ≈ ±8-10° for
±10° on the robot; z (crouch) is NOT learned. ENABLE_YAW_FIX is a documented
failed experiment (OFF) — see its comment.

PROTECTIVE-FALL REBUILD (2026-09, branch protective_fall). Motivation: the real
robots keep breaking XL330 gearboxes. The daemon's fall-detect limp (kp→50)
helped but is imperfect, and the limp→standup hand-off produces "convulsions".
Goal: ONE policy that walks, falls in a way that protects the servos, and gets
up gently — so the daemon limp can eventually be switched off.

What changed vs the 2026-07 design (kept below, still valid):
  - WARM START from the deployed walk (wandb 441tzs6d @ model_3750 — the exact
    checkpoint behind alpha_walking.onnx). Same 61D obs / 512-256-128 actor /
    76D critic, so `--agent.resume True --wandb-run-path ... --wandb-checkpoint-name
    model_3750.pt` loads everything. Consequence: the walk exists at iter 0, so
    every velocity-recipe curriculum (action_rate ramp, standing-env fraction,
    head/body command ranges, CoM DR, head_pose_bias weight) is COLLAPSED to
    its final stage (WARM_START) — re-running them from stage 0 would re-teach
    the walk under easier conditions and drift it. Only velstand's own phases
    ramp, and they are ~2× shorter than the from-scratch schedule.
    LAUNCH:  MICRODUCK_WARM_START=1 uv run train Mjlab-VelStand-Flat-MicroDuck \
               --env.scene.num-envs 4096 --agent.resume True \
               --wandb-run-path pollen-robotics/mjlab_microduck/441tzs6d \
               --wandb-checkpoint-name model_3750.pt
    The env var matters: mjlab's runner restores common_step_counter (90 000)
    and the iteration (3750) from the checkpoint, which would jump every
    step-based curriculum below to its final stage in iteration 1 (seen in the
    first smoke test). MICRODUCK_WARM_START=1 (tasks/mdp.py Patch 5) restarts
    both at 0 after loading weights / normalizers / optimizer, so the
    curricula below are relative to THIS run's start. Do NOT warm-start from the stand
    expert (69u48n8l): its obs normalizer has twist std 0.005 — walking
    commands through it are a 40σ input.
  - ROBOT = robot_allcollisions.xml (true full-collision export; identical
    bodies/masses/ranges to groundcontact, 70 collision geoms vs 11), with the
    15 XL330 housing geoms NAMED (name_servo_collision_geoms) so a contact
    sensor can tell "a servo hit the floor" from "the shell hit the floor".
  - SERVO-PROTECTION COSTS, all mjlab-style (>= 0, negative weight), all ≈ 0
    during clean walking so they price only falls and bad recoveries:
      servo_impact    — housing contact force above 2 N (the event that strips
                        gears: a servo case taking the landing)
      head_impact     — head-subtree floor force above 15 N (≈2× body weight:
                        pushing off the head to get up is quasi-static ~7 N and
                        stays FREE — the 2026-07 lesson that an ungated head
                        penalty taxed the recovery strategy; a face-plant is a
                        60 N spike, measured)
      trunk_impact    — trunk shell floor force above 20 N (same logic)
      servo_acc_spike — servo joint |accel| above 300 rad/s² (impact
                        back-driving loads the gear train through the reflected
                        rotor inertia; CPU falls peak 450–550 regardless of
                        limp strategy — the policy's lever is HOW it lands)
      servo_stall     — servos pushing > 0.4 N·m at < 0.5 rad/s (a limb pinned
                        under the body while the servo fights it: the
                        convulsion / gear-strip / overheat case)
      gentle_rise     — trunk |a_z| (self-negating, POSITIVE weight)
    Ramped 25% → 100% after the recovery economics kick in, so the recovery
    discovery window is not taxed by its own failed attempts.
  - TOPPLE PUSHES: a second push event (topple_push) with velocity kicks far
    beyond the walking DR (ramped ±0.6 → ±1.2 m/s; a 1 m/s Δv topples this
    robot ~100% of the time in the CPU baseline), so falls are frequent
    on-policy data rather than rare accidents.
  - Backlash twin: robot_allcollisions_backlash.xml (add_backlash.py, 2° total)
    → Mjlab-VelStand-*-Backlash-MicroDuck mirror the base model as required.
  - NOT modelled on purpose (user decision 2026-09-09): the daemon limp. If the
    policy finds a gentler strategy with full authority, better let it.

Run-1 lesson (wandb 4otqmkb4, killed @1098, headless eval of ckpt 750):
  The walk transferred (iter 100 = source-run tracking/upright) and the topple
  ramp produced falls, but 0 recoveries ever: 73% of eval env-steps fallen,
  fallen action rate 1/5 of upright, torque 1/3 → the policy learned to LIE
  STILL. Two causes, both fixed here:
  (1) ATTEMPT TAX. The warm-start collapse pinned action_rate_l2 at -1.0 from
      step 0 (-1.5..-2/step, the largest cost in the stack). With a flat -0.5
      fallen tax, thrashing costs more than waiting → "do nothing" wins (the
      AGENTS.md smoothness-after-discovery rule, hit from the other side).
      Fix: action_rate / torque_rate are ×FALLEN_SMOOTHNESS_SCALE (0.1) while
      tilt > 40° (action_rate_l2_fallen_scaled) — the walk keeps its full tax.
  (2) SPAWN HOLE. 79% of falls end ON THE SIDE, 20% face-up, 0% face-down; the
      prone init only knew face-down/face-up. Fix: side_prob in the prone slice.
  Also: recovery ramps pulled earlier (walk needs no bootstrap window) and the
  crouch slice enlarged.

Run-2 lesson (wandb 1bqctpkq, killed @1085, per-spawn battery of ckpt 1000):
  standing 98% / crouch→stand 97% / face-down, face-up, side 0% — still lies
  still (fallen |Δa| a quarter of upright). Relieving the attempt tax was not
  enough: the warm-started walk has action std ≈ 0.2, random flailing from
  lying never yields a partial rise, so the potential-based terms never pay →
  no gradient at all (the from-scratch velstand runs discovered recovery with
  std 1.0 flailing, at the cost of the walk). The deployed stand expert
  (69u48n8l@9750) run through the same battery: 100/100/95% (side never
  trained!) within ~1 s. Fix: PpoWithExpertBc — fallen-gated behavior cloning
  toward that frozen expert after each PPO update (distill.py). RL keeps
  shaping the fall itself and the last mile.

Run-3 lesson (wandb 4lflk7ii, @1060): IT STANDS UP from front and back — and
  the walk died at the first BC pass (iteration 8: 20 Adam steps at 1e-3 on a
  few hundred fallen frames; fall terminations 2 → 72 by iteration 12, then a
  falls→fallen-frames→more-BC loop; yaw tracking error 0.9 → 3.4 rad/s).
  Nothing constrained the shared weights on upright frames. Fix (distill.py):
  the warm-start WALK checkpoint is a second frozen teacher anchoring frames
  with tilt < 25°; stand expert on tilt > 35°; PPO alone in between and for
  everything the reward adds on top. BC lr 3e-4, min mini-batch 512. 40-iter
  check at 64 envs with 50% prone spawns + 1 m/s pushes: fallen frac 0.44 →
  0.33 while upright lin-vel error stayed 0.22 m/s (run 3's recipe destroyed
  the walk in 12 iterations under milder conditions). Impact costs were confirmed real but weak (face-plant
  spikes 129 N at 0.5% of steps ≈ -0.3 per fall vs ~7/step walking) — to be
  raised ×5 once recoveries exist, not before.

Run-4 (wandb 6op8a8u8) WORKS, on the robot too → published reference. Sim
  benchmark vs the deployed walk→limp→stand pipeline (3 seeds): recovers 88-95%
  vs 83-86% of push falls, 2.7 s vs 3.3 s median, zero servo-housing contact vs
  50 ms/fall, trunk impact 0 vs 14 N, stall halved; forward speed equal to the
  walk expert; head impact EQUAL (~32 N) — the open target. Turn-in-place yaw
  rate drifted 0.34 → 0.12 rad/s over training (parked, not critical).
  Robot feedback: (a) convulsions rising from the BACK after a real fall — in sim
  face-up LANDINGS after a push recover 58-77% vs 100% from a face-up SPAWN; the
  stand expert has the same hole (71%): a real fall leaves the legs anywhere,
  HOME-pose spawns never show that. (b) occasional backward overshoot after a
  front stand-up — NOT reproduced in sim (0/192 re-falls).

Runs 5-6 (axr82ws4, padquu42; resumed from 5999) — ROLLED BACK 2026-09-13:
  tried post-fall-like prone spawns (randomize_servo_joints_uniform, kept in
  mdp.py, OFF here), servo_stall ×3, then fallen smoothness 0.4 / gentle_rise ×4
  / BC coef 0.3. Lessons: servo_stall ×3 made the rise BALLISTIC ("high torque at
  low velocity" IS a slow careful push-up) — robot overshoot + fall back; the
  stand expert is only 83% on randomized-joint face-up spawns and the student
  plateaued there (teacher ceiling); run 6 gave a mild push-recovery gain but
  face-up stayed at 85% and front stand-ups re-fell 9/190 (5999: 0/192), cause
  unknown. Not better than 5999 on the robot → back to this recipe.

Rough+backlash run 1 (wy8gcaus) — CATASTROPHIC, root cause FIXED in mdp.py:
  every fallen-spawn function wrote an ABSOLUTE trunk z (0.05-0.09 m); rough env
  origins span 0-0.21 m (slope pyramids) → robots spawned inside the terrain
  from iter 700 (prone spawns on) → value loss 0.6 → 42, entropy 5 → 33. Spawn z
  is now origin-relative (_env_origin_z). The flat run with the same constants
  was stable throughout. Prod recipe = THIS file on
  Mjlab-VelStand-Rough-Backlash-MicroDuck, warm start from 6op8a8u8@5999.

REBASED (2026-07, audit follow-up) on the velocity recipe — the proven
walker — instead of the abandoned older recipe the old velstand used.
The 2026-07 audit found the old design starved the walk: only ~25% of
experience was clean commanded walking (2/3 prone resets + fallen envs farming
recovery reward for full 20 s episodes), the recovery rewards taxed the gait
(always-on posture double-counting, a bounce incentive from com_upward_velocity
below walk height), and the prone init dropped the robot from 0.20–0.25 m
(function defaults — a violent uncontrolled impact opening most episodes).

Design now:
  - Walk layer  = make_microduck_velocity_env_cfg, verbatim. Everything the
    good walker has (tracking weights, air_time, turn-in-place bucket, fixed
    command ranges, DR/noise/obs) flows in by construction.
  - Robot       = all-collision standup XML (body can physically lie down).
  - Recovery    = a small reward layer GATED on actually-being-fallen
    (trunk z < 0.10 m OR tilt > 40°): contributes exactly zero during clean
    walking, steers only when down. upright_linear gives an orientation
    gradient everywhere; com_upward_velocity pays for rising. (The old
    com_height_recovery was dropped: flat/no-gradient inside its band and
    redundant with the two above — audit finding 3.)
  - Impact penalties (trunk/head) discourage hard landings, ungated.
  - joint_torque_rate_l2 (standup's proven anti-jitter) for transfer
    smoothness — penalizes torque CHANGE, never blocks the recovery flip.

Run-5 lesson (crouch endpoint): recoveries walked nicely but parked in a deep
crouch just past the 40° gates — every dense recovery term stops paying there,
and the recovery_success bounty demanded z > 0.105, above the policy's real
standing envelope (0.084–0.096), so it never fired. Fixes: (1) shared
"recovery complete" definition (tilt < 25° AND z > 0.09 — reachable) for the
bounty and (2) a fallen_tax hysteresis that keeps taxing after a fall until
that definition is met, and (3) height_progress — a potential-based Δz term
giving the crouch→stand last mile the dense gradient nothing else provides.

Run-6 lesson (still parked at 4k): fixing the economics wasn't enough — the
bounty fired (rising recovery_success curve) but stayed exploration-rare,
because the last mile got almost no on-policy DATA: a prone episode spends
most of its 5 s fallen budget getting TO the crouch, then fallen_too_long
recycles it right at the frontier. The old velstand learned recovery fast
precisely because 2/3 prone resets + 20 s episodes made fallen-state data
abundant (at the cost of the walk). Run-6 recovers that data density without
the starvation: (1) crouch_prob reverse-curriculum slice — reset directly
into random mid-recovery crouches, dense last-mile data from step 0; (2)
fallen timeout 5 → 8 s; (3) economics at 800 (walk is stable by ~750) and
the whole prone ramp pulled ~500 iters earlier.

Run-7 lesson (headless eval of run 6 vs run 5 @4k, 2026-07-21): the crouch
slice WORKED — run 6 stands truly vertical (tilt ≈1°, z ≈0.117) and recovers
94–97% from crouch inits — but prone recovery collapsed to 0% (run 5: gets up
from prone but parks at ~30°). Cause: run 6 turned on tax + bounty + prone +
crouch ALL at iter 800, deleting the tax-free natural-fall window (500→1200 in
run 5) where prone-flip exploration was cheap and the dense progress terms
alone taught it — run 5's recovery_success was already firing the moment its
weight turned on at 1200. With the tax live from 800 and hopeless prone
episodes bleeding -0.5/step for the full 8 s timeout, the run-3 avoidance/
freeze mechanism re-emerged for prone states while PPO capacity went to the
easy crouch-slice reward. Run-7: keep the crouch slice (validated) + 8 s
timeout, restore econ to 1200 and prone to the run-5 ramp (1500+), crouch
slice alone from 800 (harmless pre-econ: it just adds stand-tall data).

Phases (as before, but with a recovery backstop):
  Phase 1 (0 → 500 iters): `fell_over` termination active (70°) → clean
    walking first.
  Phase 2 (500+): fell_over disabled (limit → π) so falls become recovery
    opportunities — but `fallen_too_long` (5 s continuously down) recycles
    failed recoveries instead of letting them farm the full 20 s episode.
  Phase 3 (1500+): prone-init ramp: face-down first (easier), face-up mixed
    in later, capped at 45% prone so the walking data share stays ≥ ~55%
    (was 2/3 prone → ~25% walking share).
"""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlModelCfg,
)

from mjlab.envs.mdp import push_by_setting_velocity
from mjlab.sensor import ContactMatch, ContactSensorCfg

from mjlab_microduck.robot.microduck_constants import (
    MICRODUCK_ALLCOLLISIONS_ROBOT_CFG,
    SERVO_GEOM_SUFFIX,
)
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    HEAD_BODY_NAMES,
    make_microduck_velocity_env_cfg,
)
from mjlab_microduck.tasks.distill import PpoWithExpertBcCfg, default_bc_cfg
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg

NUM_STEPS_PER_ENV = 24

# Warm start from the deployed walk (see module docstring). Collapses every
# velocity-recipe curriculum to its final stage so the loaded walk is trained
# under the conditions it was trained under. Set False to train from scratch
# with the original from-scratch schedule (phase constants below scale up).
WARM_START = True
WARM_START_RUN = "pollen-robotics/mjlab_microduck/441tzs6d"   # alpha_walking.onnx
WARM_START_CHECKPOINT = "model_3750.pt"

# Phase boundaries (PPO iterations; env step counter scales by num_steps_per_env=24).
# Warm start: the walk exists at iter 0, so fell_over only needs a short
# adaptation window to the all-collisions model before falls become data.
FELL_OVER_DISABLE_ITER = 100 if WARM_START else 500

# Fallen gates. LESSON (first rebase training run): the recovery REWARDS must
# gate on TILT ONLY. Gating them on low height too made SITTING (z≈0.07, trunk
# upright) open the gate → the policy learned to sit and farm upright_linear
# while bobbing for com_upward_velocity and shaking its legs through the
# air_time window. Gating a positive reward on a bad state rewards entering
# the state. Tilt>40° can't be farmed from a comfortable pose — you're
# genuinely toppled. The TERMINATION keeps the z-condition so sitters and
# stuck-low envs get recycled (terminated) rather than paid.
REWARD_GATE_TILT_DEG = 40.0   # recovery rewards: fallen = tilt > 40° ONLY
# TERM z-gate at 0.08, NOT 0.10 (run-3 lesson): a normally wobbling upright
# robot dips to z=0.084-0.096 — 0.10 sits inside the early-learning envelope
# and recycled crouch-walking explorers every 5 s. 0.08 still catches sitting
# (z≈0.07) and prone (z≈0.05).
TERM_GATE_Z = 0.08            # fallen_too_long: z < 0.08 OR tilt > 40°
TERM_GATE_TILT_DEG = 40.0

# "Recovery COMPLETE" definition — shared by the recovery_success bounty and
# the fallen_tax release (run-5 crouch-endpoint lesson). z threshold must sit
# INSIDE the policy's real standing envelope: run 3 measured a normally
# wobbling upright robot at z ≈ 0.084–0.096, and the full STAND keyframe
# settles at ≈ 0.117. The old up_z=0.105 demanded standing TALLER than the
# policy ever is in practice → the bounty never fired → recoveries converged
# to a deep crouch just past the 40° gates (where every dense recovery term
# stops paying) instead of finishing the stand. 0.09 is reachable every stand
# yet still 2 cm above sitting (z ≈ 0.07) and 4 cm above prone (z ≈ 0.05).
RECOVERED_UP_TILT_DEG = 25.0
RECOVERED_UP_Z = 0.09

# The tax and bounty exist FOR THE RECOVERY PHASE. Run-3 lesson: fallen_tax
# active from step 0 (dense, -0.5) taught "avoid tilt at all costs" within ~25
# iters → crouch-freeze local optimum before walking could bootstrap (ep_len
# pinned at the 5 s recycle, air_time never grew). Run-6 tried 800 ("walk is
# stable by ~750") and prone recovery never bootstrapped — 1200 was never
# about the walk; it bought a TAX-FREE window (fell_over off at 500 → econ on
# at 1200) where natural-fall get-up attempts cost nothing and the dense
# progress terms alone could teach them. Run-7 restores it.
RECOVERY_ECON_KICKIN_ITER = 600 if WARM_START else 1200

# Run-2 fix (1bqctpkq, 0 recoveries @1085 even with the attempt tax relieved):
# distill the deployed stand expert into the fallen frames (see distill.py).
# The expert recovers 100% face-down/up and 95% side on this model in ~1 s.
ENABLE_EXPERT_BC = True
EXPERT_BC_COEF = 1.0
EXPERT_BC_GATE_TILT_DEG = 35.0

# Body control (2026-10, branch improve_velstand2). The deployed velstand ignores
# body commands entirely (measured: 0 mm / 0° response — the inherited velocity
# recipe keeps body_pose at weight 0 with ±5 mm/±0.05 alive ranges, and the walk
# anchor pins standing to alpha_walking, which never learned it). The old pipeline's
# alpha_stand tracks roll/pitch (9.2-9.5° for 10°) but not z (+1 cm → +3.7 mm,
# crouch → 0). So:
#   - command only while STANDING (StandingGatedPoseCommand: exact zero while
#     walking = deployment), 30 % all-zero bucket so plain standing stays trained;
#   - tilt-only frames (z == 0, BODY_Z_ZERO_PROB of them) → stand-expert BC;
#     z frames → PPO only (alpha_stand cannot crouch); both off the walk anchor
#     (distill.py body routing — separable: the command is in the obs);
#   - standing-gated z/roll/pitch tracking reward + upright made command-relative
#     (identical for walking / zero command).
ENABLE_BODY_CONTROL = True
BODY_NOMINAL_Z = 0.116             # measured: standing trunk z on allcollisions (fhathosb 0.1158, alpha_stand 0.1157)
BODY_CMD_MAX_ANGLE = math.radians(12)   # ≈ alpha_stand's trained range (normalizer std 0.10 rad)
BODY_CMD_Z_RANGE = (-0.025, 0.010)      # crouch room below HOME, ~1 cm extension above
BODY_ALIVE_XY = 0.002              # x/y/yaw untracked: tiny ranges inside alpha_stand's normalizer
BODY_ALIVE_YAW = 0.02
BODY_ZERO_CMD_PROB = 0.3
BODY_Z_ZERO_PROB = 0.5             # share of body commands that are tilt-only (→ teachable)
BODY_TRACK_WEIGHT = 2.0
BODY_RANGE_STAGES_ITERS = (0, 300)  # half range → full range (warm start: iters of THIS run)
# Leg pose stds while a body command is active (standing). The standing stds (hip_roll
# 0.05, knee/hip_pitch 0.15, ankle 0.1) priced a 10° commanded roll at -0.85/step.
BODY_POSE_STD = {
    r".*hip_yaw.*": 0.1,
    r".*hip_roll.*": 0.25,
    r".*hip_pitch.*": 0.4,
    r".*knee.*": 0.4,
    r".*ankle.*": 0.3,
}

# Yaw dead zone fix (2026-10-01). Robot: "still very bad at turning". j4i6yoq2@1250
# in sim: in-place 0.3 / 0.6 rad/s → 0.00 / 0.07, 1.0 → 0.45. Counterfactual (same
# policy fed a bigger yaw cmd so it steps + turns, reward at the TRUE cmd): standing
# still WAS the reward optimum at 0.3 and turning won by only 4 % at 0.6 — the stock
# yaw std 0.71 is loose at small errors and mjlab's term mixes in the roll/pitch rates
# stepping creates. Fixes:
#   - ADD a yaw-only fine term (std 0.35, weight 2) next to the unchanged stock term.
#     Run sape62zb REPLACED the stock term with the sharp one instead: turning got WORSE
#     (1.0 rad/s 0.42 → 0.16 @1500) — no gradient at the policy's 0.6 rad/s error.
#     Additive margins: 0.6 rad/s +0.33 → +0.97/step, 1.0 +0.62 → +1.05;
#   - turn-in-place bucket samples |wz| from 0.1 (was 0.4: in-place yaw < 0.4 never trained);
#   - turn-in-place frames off the walk anchor (alpha_walking has this exact dead zone).
# OFF — both runs with it LOST turning (sape62zb: replaced term; brkpcnn8: additive term;
# in-place ±1.0 rad/s 0.39/−0.36 → ~0.2-0.3 within 250 iters, 0.3/0.6 still ≈0). Common
# factor = the unanchored turn frames: the walk anchor is the turning floor and PPO does
# not discover slow turning even with positive reward margins. What actually works is a
# COMMAND REMAP: the policy turns monotonically when fed a larger yaw (fed 1.0/1.5/2.0 →
# 0.40/0.80/1.09 rad/s in place, no falls); runtime fed = sign(c)·(0.33 + |c|/0.6) makes
# v7 (j4i6yoq2@1250) track 0.5-1.0 rad/s in sim (see docs/velstand_policy.md). Kept OFF
# so this file reproduces the published v7 recipe (commit 53fb7d1).
ENABLE_YAW_FIX = False
TRACK_YAW_FINE_STD = 0.35
TRACK_YAW_FINE_WEIGHT = 2.0
TURN_IN_PLACE_FRACTION_VELSTAND = 0.2    # velocity recipe: 0.15
TURN_IN_PLACE_MIN_FRAC = 0.1             # velocity recipe: 0.4

# Teacher-side yaw remap (2026-10-12). Robot (qz5g3bwi@1750, good stand-up): "still
# pretty bad at yaw turning". Sim (velstand_yaw_probe.py, backlash): in place 0.3 / 0.6
# → 0.00 / ±0.05 rad/s, 1.0 → 0.34 / −0.42; walking 0.2 m/s ≈ half the command — and the
# WALK ANCHOR has the same curve (in place 0.6 → 0.01, 1.0 → 0.35 / −0.19), which BC
# coef 1.0 copies. Both turn monotonically when fed a bigger yaw (no falls up to 2.4).
# Unanchoring turn frames lost turning twice (ENABLE_YAW_FIX runs above), so instead the
# anchor is fed the command that makes IT turn at the student's commanded rate: inverse
# of its measured response (claude_experiments/calibrate_walk_yaw.py --backlash, 441tzs6d
# @3750 as distill loads it, 8 envs per cell, 0 falls). Rows = |v_xy| knots, columns =
# target |wz| knots, values = fed |wz|; "pos"/"neg" per turn direction (the anchor is
# asymmetric). Standing and straight walking (wz = 0) are unchanged, fallen frames use the
# stand expert → the stand-up is untouched. Plus the fine yaw-only term (reward agrees
# with the labels; the stock term preferred standing still at 0.3 in place) and a
# turn-in-place bucket reaching down to 0.2 rad/s.
ENABLE_ANCHOR_YAW_REMAP = True
ANCHOR_YAW_REMAP = {
    "lin": (0.0, 0.1, 0.2, 0.3, 0.4),
    "targets": (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    "pos": (
        (0.75, 0.89, 1.00, 1.06, 1.11, 1.17, 1.25, 1.38, 1.67, 1.95),
        (0.80, 0.85, 0.89, 0.94, 0.99, 1.12, 1.27, 1.39, 1.67, 1.87),
        (0.28, 0.45, 0.55, 0.68, 0.86, 1.06, 1.26, 1.44, 1.55, 1.85),
        (0.21, 0.32, 0.47, 0.68, 0.86, 1.05, 1.23, 1.39, 1.51, 1.77),
        (0.19, 0.39, 0.62, 0.72, 0.82, 0.97, 1.25, 1.37, 1.51, 1.72),
    ),
    "neg": (
        (0.90, 1.01, 1.07, 1.13, 1.19, 1.31, 1.46, 1.70, 1.94, 2.16),
        (0.71, 0.81, 0.89, 0.96, 1.08, 1.29, 1.49, 1.67, 1.88, 2.08),
        (0.25, 0.49, 0.67, 0.80, 1.04, 1.17, 1.41, 1.58, 1.81, 1.90),
        (0.23, 0.38, 0.60, 0.77, 0.92, 1.10, 1.33, 1.49, 1.62, 1.83),
        (0.17, 0.30, 0.44, 0.64, 0.85, 1.02, 1.15, 1.42, 1.60, 1.73),
    ),
}
YAW_REMAP_TURN_IN_PLACE_FRACTION = 0.2   # velocity recipe: 0.15
YAW_REMAP_TURN_IN_PLACE_MIN_FRAC = 0.2   # velocity recipe: 0.4 (in-place |wz| < 0.4 never trained)

# Stand-up coverage (2026-10-07, branch improve_standup_velstand). Robot: get-ups
# sometimes fail, and the robot sometimes stays stuck without trying again.
# Measured on v7 (j4i6yoq2@1250, claude_experiments/velstand_rise_battery.py,
# backlash, 12 s, 256 starts per family): 100 % from the ±90° HOME-joint spawns
# training uses, but face-up with random joints 74 % (26 % STUCK = not trying),
# side 86 %, random orientation 89 %, sitting 88 %. Every stuck v7 run on the back
# ends in ONE attractor that the policy actively holds: tilt ~96°, head_yaw
# commanded to −1.56 rad (always negative), left hip_roll driven into its limit.
# A face-up start with only head_yaw at −1.0..−1.7 freezes 17 % (+yaw: 0 %).
# alpha_stand has the same attractor (50-70 % of its own stuck runs) and the
# student copies it at BC coef 1.0 (runs 5/6 plateaued at the teacher's ceiling
# on post-fall spawns for the same reason). Fixes:
#   - JOINT-MASKED TEACHER: on frames tilted beyond TEACHER_MASK_TILT_DEG the stand
#     expert is fed the obs with EVERY joint pos / vel / last action zeroed (= "you
#     are at HOME"), so its label only depends on ang_vel + gravity there: the
#     HOME-pose rise sequence it does reliably, whatever the actual joints are.
#     Gate at 75°, NOT the 35° BC gate (the expert needs its joints for the end of
#     the rise). A pure function of the obs, so BC labels stay consistent.
#     Emulated closed-loop on alpha_stand (backlash, 256 starts/family): ±90° HOME
#     spawns 97-100 %, random-joint face-up 89 → 93 %, random orientation 91 → 95 %,
#     limp power-on 91 → 96 %, sitting 87 → 95 %, AND gentler: rise peak |ω| from the
#     back 10.7 → 6.7 rad/s, t_rec p90 4.1 → 2.3 s.
#     HEAD-ONLY MASK FAILED (run s4jdgkux = head cols masked above 60°, resumed from
#     j4i6@1250): its own emulation looked fine (rj face-up 90 → 95 %), but the
#     student froze MORE (rj face-up 74 → 55 %, ±90° side 100 → 89 %) in the same
#     leg attractor with head_yaw moved −1.66 → −0.96. The head-masked teacher
#     agrees with the student there (label gap 0.03 rad) and, handed control,
#     rescues nobody; the all-joint-masked teacher at 75° rescues 98-99 % of those
#     states. The attractor lives in the teacher's response to the LEG config.
#     FAILED FIRST TRY (local 400 iters from j4i6@1250): dropping the teacher label
#     on envs down > 3 s (PPO alone there) made it WORSE everywhere (rj face-up
#     74 → 62 %, even ±90° side 100 → 88 %): with no label, PPO's cheapest answer in
#     a hard state is to lie still (the fallen tax is paid either way) and that
#     spread to nearby states — the run-1/2 lesson again;
#   - WIDER SPAWNS in the same 45 % prone slice: random servo joints (head
#     included) on half of it, a quarter of it at a uniformly random orientation,
#     plus a small sitting slice — the states where the freeze happens.
ENABLE_JOINT_MASKED_TEACHER = True
TEACHER_MASK_TILT_DEG = 75.0
# Flat 61D actor obs: ang_vel 0:3, gravity 3:6, joint_pos 6:20, joint_vel 20:34, last action 34:48.
TEACHER_MASK_COLS = tuple(range(6, 48))
ENABLE_WIDE_SPAWNS = True
WIDE_JOINT_RANDOM_PROB = 0.5    # share of the prone slice with random servo joints
WIDE_JOINT_RANGE_FRAC = 0.9
WIDE_SO3_PROB = 0.25            # share of the prone slice at a uniformly random orientation (always random joints)
WIDE_SIT_PROB = 0.05            # exclusive sitting slice (of all resets)
SIT_JOINT_OVERRIDES = {         # sitstand's stable SIT keyframe (servo indices)
    1: 0.0, 2: -0.4079, 3: 1.35, 4: 0.0,
    10: 0.0, 11: 0.4079, 12: -1.35, 13: 0.0,
}

# Run-1 fix (1): smoothness taxes scaled down while fallen so get-up attempts
# are affordable; full weight while upright (the walk's smoothness is untouched).
FALLEN_SMOOTHNESS_SCALE = 0.1

# Servo-protection costs ramp (see docstring). 25% from step 0 keeps the
# gradient alive; full weight after the recovery economics are in place.
PROTECT_FULL_ITER = RECOVERY_ECON_KICKIN_ITER + 400
PROTECT_STAGE0_FRAC = 0.25
SERVO_IMPACT_WEIGHT = -0.02     # per N above 2 N on servo housings, per step
HEAD_IMPACT_WEIGHT = -0.01      # per N above 15 N on the head subtree
TRUNK_IMPACT_WEIGHT = -0.01     # per N above 20 N on the trunk shell
SERVO_ACC_SPIKE_WEIGHT = -1e-3  # per rad/s² above 300 (summed over servos)
SERVO_STALL_WEIGHT = -0.05      # per stalled servo per step
GENTLE_RISE_WEIGHT = 0.005      # POSITIVE: trunk_vertical_accel_penalty is self-negating
SERVO_IMPACT_THRESH_N = 2.0
HEAD_IMPACT_THRESH_N = 15.0
TRUNK_IMPACT_THRESH_N = 20.0
SERVO_ACC_THRESH = 300.0
SERVO_STALL_TORQUE = 0.4
SERVO_STALL_VEL = 0.5

# Gear protection (2026-10-30, branch gear_protection). v8 still breaks knees, hip
# pitch and neck pitch on the robots. The XL330 bench (one fresh motor per setting
# driven into a hard stop until the first tooth skip) says what kills a gearbox: the
# load on the teeth, with life ∝ (load − 0.40 N·m)^−2.4 and a plateau near 0.40, and
# above all the HOLD after the impact (×3.6 life when the servo stops pushing). The
# protection costs above miss it in two ways: servo_stall is a count at 0.4 N·m (no
# gradient where the curve is steepest), and servo_acc_spike differences velocity at
# 50 Hz, blind to a few-millisecond contact. Measured on v8 in this sim
# (scripts/gear_wear_eval.py): lying down, neck_pitch sits at FULL torque ~30 % of the
# time (pushing off the head); walking, the knees pass 0.4 N·m on ~2 % of steps, just.
#   - gear_hold:   Σ (|τ| − 0.3)₊² on servos moving < 0.5 rad/s — the graded stall.
#   - gear_impact: Σ (peak gear load − 0.4)₊² with the gear load at PHYSICS rate,
#                  τ_motor − J_rotor·q̈ (FrictionDRBamActuator): arrival speed into the
#                  floor and back-driven landings, which the 50 Hz term cannot see.
# Both ≈ 0 while walking (reward mass measured there, not guessed). The run-5/6
# lesson (servo_stall ×3 → BALLISTIC rise) is why the hold is never priced alone:
# a ballistic rise is a fast arrival, which gear_impact prices.
#   - label_hold_clamp (distill.py): the stand expert's labels on fallen frames keep
#                  their direction but never ask a stalled knee / hip pitch / neck pitch
#                  for more than LABEL_HOLD_MAX_ERR past where it is. BC coef 1.0
#                  dominates the fallen frames, so without this the rewards above only
#                  nudge a rise the teacher dictates.
# Fine-tune from v8 by PLAIN resume (--init-checkpoint the v8 bundle): curricula stay
# final, and these ramp from GEAR_BASE_ITER, v8's last iteration.
ENABLE_GEAR_PROTECTION = True
GEAR_BASE_ITER = 2750
GEAR_RAMP_ITERS = 300
GEAR_STAGE0_FRAC = 0.25
GEAR_HOLD_WEIGHT = -3.0
GEAR_HOLD_TORQUE = 0.3
GEAR_HOLD_VEL = 0.5
GEAR_IMPACT_WEIGHT = -1.0
GEAR_IMPACT_THRESH = 0.4
ENABLE_LABEL_HOLD_CLAMP = True
LABEL_HOLD_JOINTS = (2, 3, 5, 11, 12)   # hip_pitch, knee (both legs), neck_pitch
# Stall torque per rad of error at P gain 200: BAM 0.56 N·m/rad, the bench motors ≈ 0.9
# (1.05 A/rad × 0.80-0.91 N·m/A), so 0.35 rad ≈ 0.2-0.3 N·m — under the 0.4 plateau.
LABEL_HOLD_MAX_ERR = 0.35

# Topple pushes: velocity kicks big enough to knock the robot over, so falling
# (and therefore protective landing + recovery) gets dense on-policy data.
# Ramp starts once fell_over is off (a topple before that is just a reset).
TOPPLE_PUSH_INTERVAL_S = (4.0, 8.0)
TOPPLE_PUSH_STAGES = [
    {"step": 0,                                          "velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}},
    {"step": FELL_OVER_DISABLE_ITER * NUM_STEPS_PER_ENV, "velocity_range": {"x": (-0.6, 0.6), "y": (-0.6, 0.6)}},
    {"step": 400 * NUM_STEPS_PER_ENV,                    "velocity_range": {"x": (-0.9, 0.9), "y": (-0.9, 0.9)}},
    {"step": 800 * NUM_STEPS_PER_ENV,                    "velocity_range": {"x": (-1.2, 1.2), "y": (-1.2, 1.2)}},
]

# Failed-recovery backstop: continuously fallen this long → terminate/reset.
# Run-6: 5 s → 8 s. At 5 s a face-down recovery spent most of its budget
# getting TO the deep crouch and was recycled right at the frontier — almost
# no on-policy data for the crouch→stand last mile.
FALLEN_TIMEOUT_S = 8.0

# Prone + crouch init ramp (phase 3). Prone capped at 45% (was 2/3 — starved
# the walk); face-down first (easier recovery), face-up mixed in later.
# Run-6: crouch_prob adds a REVERSE-CURRICULUM slice — envs reset directly
# into random mid-recovery crouches (see set_random_crouch_state) so the
# last mile gets dense data instead of only being reached at the tail of rare
# good rollouts. Run-7: back to the run-5 prone schedule (prone AFTER econ,
# which is AFTER a tax-free natural-fall window — see econ note above); run 6
# started prone+econ together at 800 and prone recovery never bootstrapped.
# Crouch slice alone starts at 800: near-upright states, tax-free until econ,
# and it doubles as full-stand posture data (run 6 stood truly vertical).
# Warm start: same shape, sooner (the walk needs no bootstrap window). Run-1
# fix (2): side_prob puts that fraction of the prone slice ON A SIDE (the
# dominant natural fall end-state, 79% in eval); the rest splits by face_down_prob.
# Crouch slice 0.15 → 0.20 and from iter 150: it doubles as stand-tall data.
_PRONE_ITERS = (150, 700, 1000, 1400) if WARM_START else (800, 1500, 2000, 2500)
PRONE_SIDE_PROB = 0.5
PRONE_RAMP_STAGES = [
    {"step": 0,                                   "params": {"prone_prob": 0.00, "face_down_prob": 1.0,  "side_prob": PRONE_SIDE_PROB, "crouch_prob": 0.00}},
    {"step": _PRONE_ITERS[0] * NUM_STEPS_PER_ENV, "params": {"prone_prob": 0.00, "face_down_prob": 1.0,  "side_prob": PRONE_SIDE_PROB, "crouch_prob": 0.20}},
    {"step": _PRONE_ITERS[1] * NUM_STEPS_PER_ENV, "params": {"prone_prob": 0.15, "face_down_prob": 0.80, "side_prob": PRONE_SIDE_PROB, "crouch_prob": 0.20}},
    {"step": _PRONE_ITERS[2] * NUM_STEPS_PER_ENV, "params": {"prone_prob": 0.30, "face_down_prob": 0.65, "side_prob": PRONE_SIDE_PROB, "crouch_prob": 0.20}},
    {"step": _PRONE_ITERS[3] * NUM_STEPS_PER_ENV, "params": {"prone_prob": 0.45, "face_down_prob": 0.50, "side_prob": PRONE_SIDE_PROB, "crouch_prob": 0.20}},
]


def _collapse_curricula_to_final(cfg: ManagerBasedRlEnvCfg) -> None:
    """Warm start: pin every inherited velocity curriculum at its final stage.

    Every velocity-recipe curriculum is a list of ``{"step": ..., ...}`` stage
    dicts under some ``*_stages`` param (weight_stages / standing_stages /
    range_stages / param_stages / push_stages). Keep only the last stage, at
    step 0, and — for reward_weight terms — also write the final weight into
    the reward cfg so step 0 itself already runs at the final value.
    """
    for name, term in cfg.curriculum.items():
        for key, val in list(term.params.items()):
            if isinstance(val, list) and val and all(isinstance(v, dict) and "step" in v for v in val):
                final = {**val[-1], "step": 0}
                term.params[key] = [final]
                if key == "weight_stages" and term.params.get("reward_name") in cfg.rewards:
                    cfg.rewards[term.params["reward_name"]].weight = final["weight"]


def _body_ranges(frac: float) -> tuple[tuple[float, float], ...]:
    a = BODY_CMD_MAX_ANGLE * frac
    return (
        (-BODY_ALIVE_XY, BODY_ALIVE_XY),
        (-BODY_ALIVE_XY, BODY_ALIVE_XY),
        (BODY_CMD_Z_RANGE[0] * frac, BODY_CMD_Z_RANGE[1] * frac),
        (-a, a),
        (-a, a),
        (-BODY_ALIVE_YAW, BODY_ALIVE_YAW),
    )


def _add_body_control(cfg: ManagerBasedRlEnvCfg, play: bool) -> None:
    """Standing-only body pose control (see ENABLE_BODY_CONTROL). Call AFTER the
    warm-start collapse: it replaces the inherited body_pose command/curriculum."""
    old = cfg.commands["body_pose"]
    cfg.commands["body_pose"] = microduck_mdp.StandingGatedPoseCommandCfg(
        resampling_time_range=old.resampling_time_range,
        ranges=_body_ranges(1.0 if play else 0.5),
        zero_command_prob=BODY_ZERO_CMD_PROB,
        axis_zero_prob=(0.0, 0.0, BODY_Z_ZERO_PROB, 0.0, 0.0, 0.0),
        twist_command_name="twist",
    )
    assert list(cfg.commands).index("twist") < list(cfg.commands).index("body_pose"), "twist must be computed first"
    cfg.curriculum.pop("body_pose_range", None)
    if not play:
        cfg.curriculum["body_pose_range"] = CurriculumTermCfg(
            func=microduck_mdp.pose_command_range_curriculum,
            params={
                "command_name": "body_pose",
                "range_stages": [
                    {"step": it * NUM_STEPS_PER_ENV, "ranges": _body_ranges(f)}
                    for it, f in zip(BODY_RANGE_STAGES_ITERS, (0.5, 1.0))
                ],
            },
        )
    cfg.rewards["body_pose_tracking"] = RewardTermCfg(
        func=microduck_mdp.body_pose_tracking_locomotion,
        weight=BODY_TRACK_WEIGHT,
        params={
            "command_name": "body_pose",
            "nominal_height": BODY_NOMINAL_Z,
            "z_std": 0.01,
            "angle_std": math.radians(5),
            "axis_weights": (0.0, 0.0, 1.0, 1.0, 1.0, 0.0),
            "standing_gate_command_name": "twist",
        },
    )
    pose = cfg.rewards["pose"]
    cfg.rewards["pose"] = RewardTermCfg(
        func=microduck_mdp.variable_posture_body_relaxed,
        weight=pose.weight,
        params={**pose.params, "std_body": BODY_POSE_STD, "body_command_name": "body_pose"},
    )
    up = cfg.rewards["upright"]
    cfg.rewards["upright"] = RewardTermCfg(
        func=microduck_mdp.upright_body_cmd_relative,
        weight=up.weight,
        params={"std": up.params["std"], "command_name": "body_pose", "asset_cfg": up.params["asset_cfg"]},
    )


def make_microduck_velstand_env_cfg(play: bool = False, rough: bool = False) -> ManagerBasedRlEnvCfg:
    # Walk layer: the PROVEN velocity recipe, verbatim.
    cfg = make_microduck_velocity_env_cfg(play=play, rough=rough)

    # In play mode the curriculum doesn't run, so the fall-termination disable
    # below never fires — just delete the termination outright.
    if play:
        cfg.terminations.pop("fell_over", None)

    # Warm start: pin the inherited velocity curricula at their final stage
    # BEFORE adding velstand's own (which must still ramp).
    if WARM_START and not play:
        _collapse_curricula_to_final(cfg)

    if ENABLE_BODY_CONTROL:
        _add_body_control(cfg, play)
    if ENABLE_YAW_FIX:
        tw = cfg.commands["twist"]
        tw.rel_turn_in_place_envs = TURN_IN_PLACE_FRACTION_VELSTAND
        tw.turn_in_place_min_frac = TURN_IN_PLACE_MIN_FRAC
        cfg.rewards["track_yaw_fine"] = RewardTermCfg(
            func=microduck_mdp.track_yaw_rate_fine,
            weight=TRACK_YAW_FINE_WEIGHT,
            params={"std": TRACK_YAW_FINE_STD, "command_name": cfg.rewards["track_angular_velocity"].params["command_name"]},
        )
    if ENABLE_ANCHOR_YAW_REMAP:
        assert not ENABLE_YAW_FIX, "the remap needs the turn frames ON the walk anchor"
        tw = cfg.commands["twist"]
        tw.rel_turn_in_place_envs = YAW_REMAP_TURN_IN_PLACE_FRACTION
        tw.turn_in_place_min_frac = YAW_REMAP_TURN_IN_PLACE_MIN_FRAC
        cfg.rewards["track_yaw_fine"] = RewardTermCfg(
            func=microduck_mdp.track_yaw_rate_fine,
            weight=TRACK_YAW_FINE_WEIGHT,
            params={"std": TRACK_YAW_FINE_STD, "command_name": cfg.rewards["track_angular_velocity"].params["command_name"]},
        )

    # True full-collision model: the robot can lie on / push off any part, and
    # the servo housings are named so the impact sensor below can single them
    # out. 70 collision geoms (vs 11) → more simultaneous contacts in a pile-up;
    # the rough-terrain nconmax (200) is the right budget here on flat too.
    cfg.scene.entities = {"robot": MICRODUCK_ALLCOLLISIONS_ROBOT_CFG}
    cfg.sim.nconmax = max(cfg.sim.nconmax or 0, 200)

    servo_ground_cfg = ContactSensorCfg(
        name="servo_ground_contact",
        primary=ContactMatch(mode="geom", pattern=rf"^.*{SERVO_GEOM_SUFFIX}$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("force",),
        reduce="netforce",
        num_slots=1,
    )
    head_ground_cfg = ContactSensorCfg(
        name="head_ground_contact",
        primary=ContactMatch(mode="body", pattern=rf"^({'|'.join(HEAD_BODY_NAMES)})$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("force",),
        reduce="netforce",
        num_slots=1,
    )
    trunk_ground_cfg = ContactSensorCfg(
        name="trunk_ground_contact",
        primary=ContactMatch(mode="body", pattern="^trunk_base$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("force",),
        reduce="netforce",
        num_slots=1,
    )
    cfg.scene.sensors = tuple(cfg.scene.sensors) + (servo_ground_cfg, head_ground_cfg, trunk_ground_cfg)

    # velocity env's head_pose_bias flows in UNGATED (fine on a walk-only env —
    # fell_over terminates fallen episodes there). Velstand episodes SURVIVE
    # falls, so the ungated EMA would charge head "droop" all through the
    # ground phase — a flat tax on being fallen that the recovery economics
    # (runs 1-7) never priced in. Add the upright gate: error stops feeding the
    # EMA below z=0.09 / beyond 40° tilt (matching REWARD_GATE_TILT_DEG), so
    # the term prices exactly what it does in the velocity env — sustained droop while
    # actually standing/walking — and nothing during recovery.
    cfg.rewards["head_pose_bias"].params.update({
        "gate_height_low":    0.09,
        "gate_height_high":   0.11,
        "gate_tilt_full_deg": 20.0,
        "gate_tilt_zero_deg": REWARD_GATE_TILT_DEG,
    })

    # ── Recovery reward layer ─────────────────────────────────────────────────
    # LESSON (runs 1/2/4 — sitting, lying, head-tripod): ANY positive reward for
    # BEING in a fallen-ish state gets farmed from some comfortable pose. The
    # orientation reward is therefore POTENTIAL-BASED (Δcos tilt): rising pays,
    # falling costs, holding anything pays zero. Unfarmable, ungated, and also
    # rewards catching a stumble while walking. (Run 4 specifically: removing
    # the head-impact penalty unlocked a head-tripod at ~55° farming the gated
    # +2·cos(tilt) — run 2 had only been protected from it by that penalty.)
    cfg.rewards["upright_progress"] = RewardTermCfg(
        func=microduck_mdp.upright_progress,
        weight=5.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )
    # z-axis companion to upright_progress (run-5 crouch-endpoint lesson): the
    # crouch→stand last mile is mostly a HEIGHT change at modest tilt — where
    # Δcos(tilt) is tiny and the Gaussian upright/pose rewards are flat. Same
    # potential-based construction: unfarmable (holding/bobbing nets zero),
    # ungated, charges falls symmetrically. Full prone→stand rise (0.05 →
    # 0.115 m) collects Δ≈+0.065 × 30 ≈ +2; the crouch→stand mile ≈ +1.
    cfg.rewards["height_progress"] = RewardTermCfg(
        func=microduck_mdp.height_progress,
        weight=30.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
            "ceiling": 0.115,
        },
    )
    cfg.rewards["com_upward_velocity"] = RewardTermCfg(
        func=microduck_mdp.com_upward_velocity,
        weight=0.0,  # recovery term — ramped in at RECOVERY_ECON_KICKIN_ITER
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
            # Height gate slightly above standing (standup uses 0.125) so the
            # rising reward keeps paying until fully up; the fallen gate is
            # what prevents gait-bounce farming, not this ceiling.
            "max_height": 0.125,
            # tilt-only gate: z=0.0 never triggers (see LESSON above)
            "gate_z_below": 0.0,
            "gate_tilt_above_deg": REWARD_GATE_TILT_DEG,
        },
    )
    # Standup's proven anti-jitter term: penalizes torque CHANGE (not magnitude
    # or rotation) → smooths transfer without blocking the recovery flip.
    # Run-1 fix (1): both smoothness taxes ×FALLEN_SMOOTHNESS_SCALE while fallen.
    cfg.rewards["joint_torque_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torque_rate_l2_fallen_scaled,
        weight=-2e-3,
        params={"fallen_scale": FALLEN_SMOOTHNESS_SCALE, "gate_tilt_above_deg": REWARD_GATE_TILT_DEG},
    )
    ar = cfg.rewards["action_rate_l2"]
    cfg.rewards["action_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.action_rate_l2_fallen_scaled,
        weight=ar.weight,
        params={"fallen_scale": FALLEN_SMOOTHNESS_SCALE, "gate_tilt_above_deg": REWARD_GATE_TILT_DEG},
    )

    # ── Servo-protection costs (see module docstring) ─────────────────────────
    # 2026-07 lesson kept: the head penalty at -1.0 @ 2 N taxed the push-off-
    # with-the-head recovery. These thresholds sit ABOVE quasi-static push-off
    # loads (≈ body weight, 7 N) so only impact spikes are billed. All start at
    # PROTECT_STAGE0_FRAC and ramp to full at PROTECT_FULL_ITER.
    cfg.rewards["servo_impact"] = RewardTermCfg(
        func=microduck_mdp.body_impact_cost,
        weight=SERVO_IMPACT_WEIGHT * PROTECT_STAGE0_FRAC,
        params={"sensor_name": servo_ground_cfg.name, "threshold": SERVO_IMPACT_THRESH_N},
    )
    cfg.rewards["head_impact"] = RewardTermCfg(
        func=microduck_mdp.body_impact_cost,
        weight=HEAD_IMPACT_WEIGHT * PROTECT_STAGE0_FRAC,
        params={"sensor_name": head_ground_cfg.name, "threshold": HEAD_IMPACT_THRESH_N},
    )
    cfg.rewards["trunk_impact"] = RewardTermCfg(
        func=microduck_mdp.body_impact_cost,
        weight=TRUNK_IMPACT_WEIGHT * PROTECT_STAGE0_FRAC,
        params={"sensor_name": trunk_ground_cfg.name, "threshold": TRUNK_IMPACT_THRESH_N},
    )
    cfg.rewards["servo_acc_spike"] = RewardTermCfg(
        func=microduck_mdp.servo_acc_spike_penalty,
        weight=SERVO_ACC_SPIKE_WEIGHT * PROTECT_STAGE0_FRAC,
        params={"acc_thresh": SERVO_ACC_THRESH},
    )
    cfg.rewards["servo_stall"] = RewardTermCfg(
        func=microduck_mdp.servo_stall_penalty,
        weight=SERVO_STALL_WEIGHT * PROTECT_STAGE0_FRAC,
        params={"torque_thresh": SERVO_STALL_TORQUE, "vel_thresh": SERVO_STALL_VEL},
    )
    # ⚠️ POSITIVE weight: trunk_vertical_accel_penalty returns -|a_z| already.
    cfg.rewards["gentle_rise"] = RewardTermCfg(
        func=microduck_mdp.trunk_vertical_accel_penalty,
        weight=GENTLE_RISE_WEIGHT * PROTECT_STAGE0_FRAC,
    )

    # ── Recovery economics (first-run lessons #3-#5) ──────────────────────────
    # air_time zeroed while fallen: a robot lying on its trunk can rhythmically
    # tap its feet through the swing window — the observed "shaking a leg" farm.
    at = cfg.rewards["air_time"]
    at_params = dict(at.params)
    cfg.rewards["air_time"] = RewardTermCfg(
        func=microduck_mdp.feet_air_time_upright,
        weight=at.weight,
        params={**at_params, "gate_tilt_above_deg": REWARD_GATE_TILT_DEG},
    )
    # Flat tax while fallen: lying still must be strictly worse than trying.
    # (Without it, waiting 5 s for the fallen_too_long recycle was rational —
    # recovery attempts cost action-rate/torque penalties, waiting cost 0.)
    cfg.rewards["fallen_tax"] = RewardTermCfg(
        func=microduck_mdp.fallen_state_penalty,
        weight=0.0,  # ramped to -0.5 at RECOVERY_ECON_KICKIN_ITER (see curriculum)
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
            "gate_tilt_above_deg": REWARD_GATE_TILT_DEG,
            # Hysteresis (run-5 crouch-endpoint lesson): recoveries parked in a
            # deep crouch just under the 40° gate — past every recovery term's
            # gate, but short of standing. With release conditions matching the
            # recovery_success bounty (below), a fall keeps taxing until the
            # stand is actually FINISHED; the sub-40° crouch is no longer a
            # zero-cost rest state. Arms only on tilt > 40°, so normal gait is
            # never taxed.
            "release_tilt_below_deg": RECOVERED_UP_TILT_DEG,
            "release_z_above": RECOVERED_UP_Z,
        },
    )
    # One-shot bounty on a COMPLETED recovery (fallen ≥0.5 s → genuinely up),
    # with hysteresis so gate-oscillation pays nothing. The strong endpoint
    # signal the dense gated terms lack.
    cfg.rewards["recovery_success"] = RewardTermCfg(
        func=microduck_mdp.recovery_success,
        weight=0.0,  # ramped to +10 at RECOVERY_ECON_KICKIN_ITER (see curriculum)
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
            "fallen_tilt_deg": REWARD_GATE_TILT_DEG,
            "min_fallen_s": 0.5,
            "up_tilt_deg": RECOVERED_UP_TILT_DEG,
            "up_z": RECOVERED_UP_Z,  # was 0.105 — unreachable, see constant note
        },
    )

    # ── Events: prone init ────────────────────────────────────────────────────
    # z fix (audit BUG): the function defaults were 0.20–0.25 m — a 15–20 cm
    # free-fall opening every prone episode. Face-down trunk rests at ~0.044 m;
    # spawn just above the ground instead.
    cfg.events["random_prone_init"] = EventTermCfg(
        func=microduck_mdp.maybe_set_random_prone_orientation,
        mode="reset",
        params={
            "prone_prob": 0.0,        # ramped by the prone_init_prob curriculum
            "face_down_prob": 1.0,
            "side_prob": PRONE_SIDE_PROB,
            "prone_z_min": 0.05,
            "prone_z_max": 0.09,
            "crouch_prob": 0.0,       # ramped by the prone_init_prob curriculum
            # Stand-up coverage (ENABLE_WIDE_SPAWNS). Not in the curriculum stages, so
            # event_param_curriculum's shallow merge leaves them as set here.
            **({
                "joint_random_prob": WIDE_JOINT_RANDOM_PROB,
                "joint_range_frac": WIDE_JOINT_RANGE_FRAC,
                "so3_prob": WIDE_SO3_PROB,
                "sit_prob": WIDE_SIT_PROB,
                "sit_params": {
                    "sitting_joint_overrides": SIT_JOINT_OVERRIDES,
                    "sitting_joint_noise_std": 0.15,
                    "sitting_tilt_max": math.radians(20),
                    "sitting_z_min": 0.06,
                    "sitting_z_max": 0.09,
                },
            } if ENABLE_WIDE_SPAWNS else {}),
        },
    )

    # Topple pushes (docstring): the velocity env's push_robot (±0.3 m/s) trains
    # stumble recovery; this one trains FALLING. Range ramped by topple_push_range.
    cfg.events["topple_push"] = EventTermCfg(
        func=push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 2.0) if play else TOPPLE_PUSH_INTERVAL_S,
        params={
            "velocity_range": TOPPLE_PUSH_STAGES[-1 if play else 0]["velocity_range"],
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # ── Terminations ──────────────────────────────────────────────────────────
    # Failed-recovery backstop (see module docstring, Phase 2).
    cfg.terminations["fallen_too_long"] = TerminationTermCfg(
        func=microduck_mdp.fallen_too_long,
        time_out=False,
        params={
            "gate_z_below": TERM_GATE_Z,
            "gate_tilt_above_deg": TERM_GATE_TILT_DEG,
            "max_duration_s": FALLEN_TIMEOUT_S,
        },
    )

    # ── Curricula ─────────────────────────────────────────────────────────────
    # Phase 1 → 2: disable fell_over at iter 500 (limit 70° → 180°) so falls
    # become recovery training instead of episode ends.
    if not play:
        cfg.curriculum["fell_over_disable"] = CurriculumTermCfg(
            func=microduck_mdp.termination_param_curriculum,
            params={
                "term_name": "fell_over",
                "param_stages": [
                    {"step": 0,
                     "params": {"limit_angle": math.radians(70.0)}},
                    {"step": FELL_OVER_DISABLE_ITER * NUM_STEPS_PER_ENV,
                     "params": {"limit_angle": math.pi}},
                ],
            },
        )

    # Phase 3: prone-init ramp (face-down first, face-up later, capped 45%).
    cfg.curriculum["prone_init_prob"] = CurriculumTermCfg(
        func=microduck_mdp.event_param_curriculum,
        params={
            "event_name": "random_prone_init",
            "param_stages": PRONE_RAMP_STAGES,
        },
    )

    # Recovery economics ramp: tax + bounty OFF until the walk is established
    # (see RECOVERY_ECON_KICKIN_ITER note above — run-3 crouch-freeze lesson).
    cfg.curriculum["fallen_tax_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "fallen_tax",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": RECOVERY_ECON_KICKIN_ITER * NUM_STEPS_PER_ENV, "weight": -0.5},
            ],
        },
    )
    cfg.curriculum["recovery_success_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "recovery_success",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": RECOVERY_ECON_KICKIN_ITER * NUM_STEPS_PER_ENV, "weight": 10.0},
            ],
        },
    )
    if not play:
        cfg.curriculum["topple_push_range"] = CurriculumTermCfg(
            func=microduck_mdp.push_curriculum,
            params={"event_name": "topple_push", "push_stages": TOPPLE_PUSH_STAGES},
        )
    for name, full in (
        ("servo_impact", SERVO_IMPACT_WEIGHT),
        ("head_impact", HEAD_IMPACT_WEIGHT),
        ("trunk_impact", TRUNK_IMPACT_WEIGHT),
        ("servo_acc_spike", SERVO_ACC_SPIKE_WEIGHT),
        ("servo_stall", SERVO_STALL_WEIGHT),
        ("gentle_rise", GENTLE_RISE_WEIGHT),
    ):
        cfg.curriculum[f"{name}_weight"] = CurriculumTermCfg(
            func=microduck_mdp.reward_weight,
            params={
                "reward_name": name,
                "weight_stages": [
                    {"step": 0, "weight": full * PROTECT_STAGE0_FRAC},
                    {"step": PROTECT_FULL_ITER * NUM_STEPS_PER_ENV, "weight": full},
                ],
            },
        )
    if ENABLE_GEAR_PROTECTION:
        cfg.rewards["gear_hold"] = RewardTermCfg(
            func=microduck_mdp.gear_hold_cost,
            weight=GEAR_HOLD_WEIGHT * GEAR_STAGE0_FRAC,
            params={"threshold": GEAR_HOLD_TORQUE, "vel_thresh": GEAR_HOLD_VEL},
        )
        cfg.rewards["gear_impact"] = RewardTermCfg(
            func=microduck_mdp.gear_impact_cost,
            weight=GEAR_IMPACT_WEIGHT * GEAR_STAGE0_FRAC,
            params={"threshold": GEAR_IMPACT_THRESH, "power": 2.0},
        )
        if not play:
            for name, full in (("gear_hold", GEAR_HOLD_WEIGHT), ("gear_impact", GEAR_IMPACT_WEIGHT)):
                cfg.curriculum[f"{name}_weight"] = CurriculumTermCfg(
                    func=microduck_mdp.reward_weight,
                    params={
                        "reward_name": name,
                        "weight_stages": [
                            {"step": 0, "weight": full * GEAR_STAGE0_FRAC},
                            {"step": (GEAR_BASE_ITER + GEAR_RAMP_ITERS // 2) * NUM_STEPS_PER_ENV, "weight": full * 0.5},
                            {"step": (GEAR_BASE_ITER + GEAR_RAMP_ITERS) * NUM_STEPS_PER_ENV, "weight": full},
                        ],
                    },
                )

    cfg.curriculum["com_upward_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "com_upward_velocity",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": RECOVERY_ECON_KICKIN_ITER * NUM_STEPS_PER_ENV, "weight": 2.0},
            ],
        },
    )

    return cfg


MicroduckVelStandRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=PpoWithExpertBcCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=None,
        bc_cfg={
            **default_bc_cfg(), "coef": EXPERT_BC_COEF, "gate_tilt_deg": EXPERT_BC_GATE_TILT_DEG,
            **({"body_slice": (55, 61)} if ENABLE_BODY_CONTROL else {}),
            **({"unanchor_turn_in_place": True} if ENABLE_YAW_FIX else {}),
            **({"anchor_yaw_remap": ANCHOR_YAW_REMAP} if ENABLE_ANCHOR_YAW_REMAP else {}),
            **({"expert_mask_cols": TEACHER_MASK_COLS, "expert_mask_tilt_deg": TEACHER_MASK_TILT_DEG} if ENABLE_JOINT_MASKED_TEACHER else {}),
            **({"label_hold_clamp": {"joints": LABEL_HOLD_JOINTS, "max_err": LABEL_HOLD_MAX_ERR, "vel_thresh": GEAR_HOLD_VEL}}
               if ENABLE_LABEL_HOLD_CLAMP else {}),
        } if ENABLE_EXPERT_BC else None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="velstand",
    run_name="velstand",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=6_000,
)

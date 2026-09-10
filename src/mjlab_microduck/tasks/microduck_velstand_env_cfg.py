"""Microduck VelStand environment: walking + protective fall + recovery, one policy.

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
  crouch slice enlarged. Impact costs were confirmed real but weak (face-plant
  spikes 129 N at 0.5% of steps ≈ -0.3 per fall vs ~7/step walking) — to be
  raised ×5 once recoveries exist, not before.

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
    algorithm=PpoWithSymmetryCfg(
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
    ),
    wandb_project="mjlab_microduck",
    experiment_name="velstand",
    run_name="velstand",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=6_000,
)

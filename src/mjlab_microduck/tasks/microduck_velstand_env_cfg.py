"""Microduck VelStand environment: walking + fall recovery, one policy.

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

from mjlab_microduck.robot.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
)
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg

# Phase boundaries (PPO iterations; env step counter scales by num_steps_per_env=24)
FELL_OVER_DISABLE_ITER = 500
NUM_STEPS_PER_ENV = 24

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
RECOVERY_ECON_KICKIN_ITER = 1200

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
PRONE_RAMP_STAGES = [
    {"step": 0,                        "params": {"prone_prob": 0.00, "face_down_prob": 1.0,  "crouch_prob": 0.00}},
    {"step": 800 * NUM_STEPS_PER_ENV,  "params": {"prone_prob": 0.00, "face_down_prob": 1.0,  "crouch_prob": 0.15}},
    {"step": 1500 * NUM_STEPS_PER_ENV, "params": {"prone_prob": 0.15, "face_down_prob": 0.80, "crouch_prob": 0.15}},
    {"step": 2000 * NUM_STEPS_PER_ENV, "params": {"prone_prob": 0.30, "face_down_prob": 0.65, "crouch_prob": 0.15}},
    {"step": 2500 * NUM_STEPS_PER_ENV, "params": {"prone_prob": 0.45, "face_down_prob": 0.50, "crouch_prob": 0.15}},
]


def make_microduck_velstand_env_cfg(play: bool = False, rough: bool = False) -> ManagerBasedRlEnvCfg:
    # Walk layer: the PROVEN velocity recipe, verbatim.
    cfg = make_microduck_velocity_env_cfg(play=play, rough=rough)

    # In play mode the curriculum doesn't run, so the fall-termination disable
    # below never fires — just delete the termination outright.
    if play:
        cfg.terminations.pop("fell_over", None)

    # Full-collision standup XML: trunk/head shells keep their contacts so the
    # robot can physically lie on the ground and push off it.
    cfg.scene.entities = {"robot": MICRODUCK_STANDUP_ROBOT_CFG}

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
    # NO impact penalties (first run lesson #2): the standup SPECIALIST has
    # none — the duck's recovery pushes off with head/trunk, and the head
    # penalty (-1.0 @ 2 N) taxed exactly that strategy. Falls stayed cheaper
    # than getting up. joint_torque_rate_l2 below covers landing harshness.
    # Standup's proven anti-jitter term: penalizes torque CHANGE (not magnitude
    # or rotation) → smooths transfer without blocking the recovery flip.
    cfg.rewards["joint_torque_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torque_rate_l2,
        weight=-2e-3,
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
            "prone_z_min": 0.05,
            "prone_z_max": 0.09,
            "crouch_prob": 0.0,       # ramped by the prone_init_prob curriculum
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
    max_iterations=20_000,
)

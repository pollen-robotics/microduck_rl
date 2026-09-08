# AGENTS.md

RL training environments for Microduck — a ~800 g, ~25 cm tall bipedal
robot with 14 Dynamixel XL330 servos — built on [mjlab](https://github.com/mujocolab/mjlab)
(MuJoCo Warp) with PPO (rsl_rl). Policies are trained here at 50 Hz, exported to
ONNX, and deployed by the runtime in the `pollen-robotics/microduck` repo on
the real robot. Sim2real transfer
is the whole point: every convention below exists because breaking it produced a
policy that worked in the viewer and failed on hardware.

## Commands

```bash
uv run list-envs                                    # live task registry
uv run train <TASK_ID> --env.scene.num-envs 4096    # train (add --hf-jobs for Hugging Face Jobs)
uv run train <TASK_ID> --env.scene.num-envs 64 --agent.max_iterations 5   # SMOKE TEST — always run first
uv run play <TASK_ID> --wandb-run-path <entity/project/run_id>
uv run scripts/export.py <TASK_ID> --wandb-run-path <...>   # → ONNX (bakes obs normalizer — mandatory path)
uv run publish --task <TASK_ID> --wandb-run-path <...> --checkpoint N --repo <user>/microduck-<name> --kind episodic --duration-s 4.0
                                                    # → HF Hub repo (policy.onnx + schema-2 manifest.json + README) the daemon loads via `robotctl policy add`
uv run scripts/infer_policy.py --walking out.onnx   # CPU MuJoCo deployment rehearsal (BAM M6 actuators as in training; --no-bam = XML PD)
uv run --with pytest pytest tests/
```

A 5-iteration smoke test at 64 envs catches ~95% of config errors for cents.
Never launch a long run without one.

## Repo map

- `src/mjlab_microduck/tasks/mdp.py` — ALL custom MDP functions (rewards, events,
  observations, commands, curricula). Add new functions here, grouped by task.
- `src/mjlab_microduck/tasks/microduck_*_env_cfg.py` — one cfg module per task
  family. `microduck_velocity_env_cfg.py` is the main walking recipe AND the
  shared base (robot, DR, obs, commands) other envs build on or mirror.
- `src/mjlab_microduck/tasks/__init__.py` — task registration (base + `-Backlash-` variants).
- `src/mjlab_microduck/tasks/backlash.py` — wraps any env cfg into its backlash twin.
- `src/mjlab_microduck/robot/microduck_constants.py` — robot cfgs, HOME frame, BAM actuator cfg.
- `src/mjlab_microduck/robot/microduck/` — MJCF exports from Onshape
  (onshape-to-robot, one `config_mjcf_*.json` per model) + scenes + `add_backlash.py`.
- `src/mjlab_microduck/actuator/friction_dr_bam.py` — BAM actuator + friction DR + backlash encoder.
- `src/mjlab_microduck/export.py` — the ONNX export (normalizer baked in); `scripts/export.py` wraps it.
- `src/mjlab_microduck/publish/` — `uv run publish`: schema-2 manifest builder + ONNX shape/smoke
  gate + Hub upload. Contract = `docs/policy-manifest.md` in the `microduck` repo; only
  constant-command episodic/perpetual policies are publishable (phase/posture-flag are the set's).
- `scripts/` — export wrapper, infer, sim2real comparison, wandb helpers.
- `tests/` — cfg-invariant and mdp-function regression tests (CPU, no GPU needed).

## Invariants — do not break these

- **Obs layout is 61D (actor) and shared across the whole policy family** so
  policies are hot-swappable in the runtime: 48 base proprioception +
  13D command block `[twist(3), head_pose(4), body_pose(6)]`, in that order.
  An env that doesn't use a command slot ZERO-PADS it (keep the obs term,
  sample tiny ranges) — never delete a slot.
- **Joint layout** (14 servos, ctrl idx = joint idx on walk/groundcontact
  models): 0–4 left leg (hip_yaw, hip_roll, hip_pitch, knee, ankle), 5–8
  neck/head (neck_pitch, head_pitch, head_yaw, head_roll), 9–13 right leg.
  On roller/backlash models, passive joints INTERLEAVE — never hardcode joint
  indices in mdp functions; use the `_servo_joint_ids` / `_servo_joint_pos`
  helpers in mdp.py (identity on plain models, correct everywhere else).
- **Unactuated joints are all named `passive_*`** (wheels, backlash hinges).
  Every actuator/obs/reward selector uses `^(?!passive_).*` — keep the prefix
  convention when adding joints, and new `passive_` regexes must not
  accidentally match backlash joints (`^passive_.*wheel`, not `^passive_.*`).
- **Actuators are BAM** (voltage-controlled XL330 model, friction computed by
  the actuator). Two consequences: any STANDALONE env cfg must register the
  `expand_bam_friction_fields` startup event, and joint-friction DR must scale
  the actuator's `friction_scale` — `dof_frictionloss` is zeroed under BAM, so
  randomizing it is a silent no-op.
- **Obs normalization is ON** → the normalizer must be baked into the ONNX.
  `scripts/export.py` does this; in-sim play hides the bug (it applies the
  normalizer anyway), so never hand-convert a checkpoint.
- **Policies are UNFILTERED** (no action low-pass in training). Don't add EMA
  filtering without a matched runtime flag and a transfer test — trained-with /
  deployed-without (either direction) breaks transfer.
- **Domain randomization must not accumulate across resets.** mjlab 1.3.0's
  `dr.*` ops with `operation="add"/"scale"` are natively non-accumulating (they
  re-read compile-time defaults); custom DR functions must restore-then-apply.
  An accumulating CoM randomizer once degraded every long run for months.
- If an obs is remapped to a sensor view (backlash encoder, bias), any tracking
  REWARD on the same quantity must measure the same view — otherwise the policy
  is punished for correcting what it sees.
- `-Backlash-` task variants must mirror their base task's robot model
  (walk / groundcontact / rollers) so backlash A/B comparisons are unconfounded.

## Building a new env — the workflow

1. **Pick the closest template** and build on it, don't start from scratch:
   locomotion → the velocity recipe; episodic trick ending in a pose →
   standup; commanded two-state → sitstand; dynamic maneuver → roulade
   (read its cfg docstring — it encodes a 5-run lesson arc). Building on
   `make_microduck_velocity*_env_cfg` keeps DR / obs / noise / delays in sync
   for free; if you build standalone from mjlab's base template, you must port
   the whole DR + obs-noise + NaN-guard stack yourself (grep for what velocity
   wires: `_safe` critic obs terms, `nan_state` termination with sensor_names,
   `expand_bam_friction_fields`, encoder bias, IMU misalignment).
2. **Verify physics assumptions in sim BEFORE training** — this is the single
   biggest time-saver:
   - A target/rest pose must be a stable equilibrium: hold its ctrl for 3 s
     from noisy inits and check TILT, not just height (a settle test that only
     records z reports fallen states as "resting fine").
   - Measure target heights off the actual robot in sim (e.g. trunk z under a
     standing policy), never carry them across model revisions. A 5 mm-wrong
     STAND_Z once turned the goal into an impossible target for days.
3. **Config conventions**: `ENABLE_*` toggles + tuned constants at the top of
   the cfg file; factory `make_..._env_cfg(play: bool, rough: bool)`; register
   in `tasks/__init__.py` (+ the `_BACKLASH_TASKS` table if applicable); own
   `RslRl...RunnerCfg` with a distinct `experiment_name`. Symmetry mirror-loss
   is available (61D table in `symmetry.py`) — OFF by default, never for
   asymmetric tasks.
4. **Write cfg tests** (see `tests/test_*_cfg.py`): joint indices resolve on
   the actual model, reward weights have the intended sign, gates open/closed
   where expected. These run on CPU and lock in the invariants.
5. **Smoke test** (64 envs, 5 iters): builds, steps NaN-free, obs is 61D,
   every reward term computes, ONNX exports.
6. Train, watch the log (below), and expect 2–5 iterations of reward-hacking
   whack-a-mole — that's normal, the lessons below shortcut most of it.

## Reward design — rules that were each learned the hard way

- **Sign convention (bit four envs):** mdp.py has two penalty styles. mjlab-base
  cost functions return ≥ 0 → negative weight. Self-negating microduck functions
  (`*_penalty`, `*_l1` returning ≤ 0) → POSITIVE weight. A negative weight on a
  self-negating penalty double-negates into a reward for the violation, and the
  policy will farm it (butt-hopping, crash-sits). **The infallible check: on
  every run, every `Episode_Reward/<penalty>` in wandb must be ≤ 0.**
- **RL optimizes the letter of the reward.** Every under-specified degree of
  freedom will be exploited (ballistic whip instead of a roll, shoulder-roll
  instead of sagittal, head-tripod instead of standing). Encode what counts as
  the maneuver in hard state-based gates (support contact, orientation-axis
  checks, latches), not in small penalty nudges.
- **No jackpots:** any "reach X" reward must be rate-limited or slewed.
  Arriving early at a goal state that then pays per-step is a jackpot that
  buys arbitrary violence. For commanded transitions, track a slewed internal
  target (constant-rate blend) — being ahead of the ramp pays zero, so slow IS
  the argmax. Speed-cap penalties alone integrate to a bounded cost and lose.
- **Never gate a positive reward on being in a bad state** (fallen, low) — the
  policy parks in the cheapest qualifying pose and farms it. Use
  potential-based shaping instead (pay Δprogress, e.g. Δcos(tilt): rising pays,
  holding pays zero, unfarmable). For rest tasks, audit each positive term
  against every stable flop (on back / face / side): if flopping keeps most of
  the stack, the policy will flop.
- **Episodic pose-landing tasks:** single fixed target from t=0 (Gaussian + L1
  on joints and height, generous std) + |a_z| impact penalty + two-layer
  upright — NOT keyframe/waypoint trajectories (the policy camps at
  waypoints). The path is what RL is supposed to discover.
- **Regularizers come in two kinds.** Motion-blockers (body_ang_vel,
  angular_momentum, pose std) penalize what a dynamic motion physically
  requires — keep them LOW for dynamic tasks. Smoothness (action_rate,
  joint_torque_rate) damps jitter without blocking slow big motions — safe to
  weight, but introduce it AFTER skill discovery (curriculum from ~0): any
  attempt-tax active while a hard skill is being explored makes "do nothing"
  win. Slow careful tasks (reaching) want heavier smoothness than walking.
- **Compare reward mass, not weights, when copying regularizers between envs.**
  PPO sees relative advantage: the same action_rate weight is 4× weaker under a
  4×-larger positive task stack.
- **Tracking Gaussian std:** ≈ the error you still care about, not the max
  error — too loose has no gradient at small errors. BUT before tightening,
  ask whether the error is escapable by the policy or inherent to the behavior
  you want (a 38%-of-body-mass head MUST oscillate while walking; a tight
  instantaneous head-tracking std taxed walking so hard the policy stood
  still). Price only the escapable part — e.g. L1 on a 1 s EMA charges DC bias
  and lets oscillation cancel.
- **Multiplicative composites beat additive sums at goal states:** when an
  additive stack has a compromise basin (80% of every term via a lean), a
  product of Gaussians collapses on any single deficient factor — but pick stds
  wide enough that the CURRENT policy scores visibly, or the gradient is
  invisible and nothing changes.
- **Joints parking on hard limits:** fix with a qpos-side limit-proximity
  penalty on the offending joints; the stock `dof_pos_limits` only fires in the
  last ~7.5% of range, and command-side penalties don't work (wide ctrlrange is
  intentional — low-kp servos need overshoot).

## Commands, observations, dead weights

- **A command input that is never non-zero has dead weights forever.** Every
  command slot keeps a small non-zero sampling range from step 0 (even at
  reward weight 0) so its input neurons stay alive for later curricula.
- **Zero-command behavior must be explicitly trained** (`zero_command_prob`-style
  exact-zero sampling): uniform sampling essentially never produces the all-zero
  command, which is exactly the deployment idle state.
- Rare-but-important command regions need explicit buckets — e.g. turn-in-place
  (`rel_turn_in_place_envs`): independent uniform sampling made spinning ~2% of
  experience and it never trained.

## Curricula

- Steps are env steps: `iteration × 24` (`NUM_STEPS_PER_ENV = 24`).
- Use the proven split: `microduck_mdp.reward_weight` for weight schedules, a
  dedicated params-curriculum for command/event ranges. `mdp.reward_weight` is
  a step function, not an interpolation — discretize ramps into stages.
- Mutate term cfgs via the managers (`env.event_manager.get_term_cfg(...)`),
  never `env.cfg.events[...]` — managers deepcopy their cfg at init, so writes
  to `env.cfg` are silent no-ops (this also bites eval scripts that force
  spawn states).
- **Phase-align every stage with what the policy has actually learned**: don't
  harden spawn mixes before the current slice consolidates; don't introduce
  taxes before the skill exists. When a wandb metric steps DOWN exactly at
  curriculum stage boundaries, the pacing is wrong — stretch stages or move
  the introduction later, never earlier.
- Reverse-curriculum spawns (starting episodes partway through the maneuver,
  including nearly-done) are the reliable fix for "learns the start, never the
  last mile" — the frontier otherwise gets no on-policy data.

## Training ops & reading a run

- wandb project `mjlab_microduck`; logs in `logs/<experiment_name>/`; resume
  with `--agent.load-checkpoint model_XXXX.pt --agent.resume True`.
- Watch per-iteration: mean reward rising AND episode length behaving as the
  task demands; every penalty term ≤ 0; the MAIN task term actually growing
  (total reward can rise purely on regularizers while the trick never happens).
  `Episode_Reward/<term>` logs the WEIGHTED value — a term at weight 0 reads 0
  regardless of behavior, so interpret against the weight schedule.
- Budgets: simple episodic tricks ≈ 1000 iters at 4096 envs; gaits and
  curriculum-heavy recovery need 4000–6000.
- **Measure before theorizing.** When a run "fails", run a headless eval of the
  actual checkpoint (per-spawn-type batteries, end-state clusters, angular-rate
  profiles) before changing rewards: past "failures" turned out to be early
  checkpoints, a success criterion splitting one behavior cluster in half, and
  a pay cap fighting measured physics. Sim metrics can pass while the video
  fails the human eye — watch the video AND check which geom/axis touches.
- Report what rollouts actually show ("rolls but face-plants 1 in 3"), not
  "it works!". The user decides when it's good enough.

## Sim2real footguns (cost real debugging weeks)

- A fresh `uv sync` is the ground truth (HF Jobs run one): anything that only
  works via manually-installed local packages will die remotely. Keep
  `pyproject.toml` honest.
- **Wheels are per-architecture.** On linux-`aarch64` (DGX Spark / GB10) PyPI's
  torch wheel is CPU-ONLY (`2.9.1+cpu`, `torch.version.cuda is None`), so
  `torch.cuda.device_count() == 0` and mjlab's `select_gpus()` indexes an empty
  list → `IndexError` before iteration 0. `[tool.uv.sources]` routes torch to
  the cu129 index for `aarch64` only (cu129 matches the CUDA toolkit warp
  bundles; x86_64/HF Jobs stay on PyPI). Two silent break points, both locked
  by `tests/test_aarch64_cuda_torch.py`: torch must stay a DIRECT dependency
  (uv applies `[tool.uv.sources]` to direct deps only — deleting the
  redundant-looking `torch==` pin makes the routing a no-op), and the pin must
  stay `==`, since the CUDA index carries newer builds than PyPI (a `>=`
  silently dragged torch 2.9.1 → 2.13.0).
- Physics-aligned limits: a 25 cm robot tumbles at 3.5–5.5 rad/s NATURALLY —
  don't impose human-scale speed intuitions via caps; put anti-violence
  pressure on impacts and thrash (|a_z|, action_rate, support gates), not on
  rotation speed.
- IMU DR is zero-centered — it trains tolerance to misalignment magnitude, and
  CANNOT compensate a systematic mounting bias (that's a runtime calibration).
- Real deployments hot-swap ONNX policies (walk / stand / trick) with a shared
  obs contract — rehearse in `scripts/infer_policy.py` before touching the
  robot, with the correct command-slot writes (a posture flag lives in the
  twist vx slot; feeding all-zeros means "stand", which looks like "policy
  ignores the button").

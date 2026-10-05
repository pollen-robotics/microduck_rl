# Reward parity between the two microduck stacks

There are two implementations of the same robot's reward vocabulary:

| stack | reward lives in | runs on |
|---|---|---|
| `microduck-lab` (CPU) | `microduck_local/behaviors/core.py` + one module per behaviour | MuJoCo CPU, SB3 PPO |
| `microduck_rl` (GPU) | `mjlab_microduck/tasks/*_env_cfg.py` + `mdp.py` | MJLab (MuJoCo Warp + rsl_rl) |

Nothing is shared between them. `microduck_local/port/` gives the CPU
vocabulary a second home, and **its** parity is exact and asserted
(`231/231` term-slots, `max|diff| = 0.00e+00`). That work deliberately does NOT
cover the MJLab configs, which are independent code with their own numbers.

So "does jumping train the same way on both stacks?" had no checkable answer.
This document is the answer for the one case where it is measurably **no**,
plus the rule that keeps the next divergence from hiding.

## Finding: `flat_feet` means the opposite thing on the two stacks

Same robot, same pose, same term name, opposite verdict.

Evidence (`BehaviorEnv(behavior_id="jump")` at the STAND keyframe — the pose
the term is documented to call *flat*):

```
lab  _flat_feet                      -> 1.000000   (1.0 = perfectly flat)
mjlab rule (xy^2 of unit gravity)    -> 0.999999   (0.0 = perfectly flat, so this is MAXIMUM tilt)

foot-frame Z axis in world           -> [0.999992 0.003900 0.001062]
```

The foot frame's Z axis points along **world +X**, not world up — this duck's
feet are near-vertical blades (that is what the roller variant rolls on). At
the flat stand the two rules therefore disagree by essentially their whole
range: **≈2.0 per foot**, so ≈4.0 total, on a term weighted `-1.0` in
`microduck_jump_env_cfg.py`.

Mechanism:

* CPU (`core.py::_flat_feet`) compares the foot-frame gravity vector to a
  reference **measured off the STAND keyframe at env init**
  (`env.foot_flat_ref`, set in `behaviors/env.py`). Self-calibrating, so it
  holds whatever the foot's rest orientation happens to be. It is a *relative*
  term: "is this foot where a flat stand puts it?"
* MJLab (`mdp.py::feet_flat_penalty`) projects a **unit world gravity vector**
  into the foot site frame and charges `sum(proj[:, :2] ** 2)`. Zero only when
  the foot's local Z axis is world-vertical. It is an *absolute* term: "is this
  foot lying sole-down on the floor like a shoe?"

Both are reasonable in isolation; only one can be right for this robot, and the
one that is wrong is wrong in the expensive direction — MJLab charges a
standing duck the maximum, so the policy's cheapest escape is to stand on the
blade edges, which is the behaviour the term was added to prevent.

### The decisive evidence, straight from the model

The MJLab term runs on the **site** frames (`_ROLLER_FEET_SITE_CFG`,
`site_names=("left_foot", "right_foot")`), and the XML gives their rest
orientation as a literal quaternion. At the rest pose the site frames are 90°
off what the docstring assumes:

```
robot_allcollisions.xml:
  <site name="left_foot"  quat="0 0 0.707107 0.707107"/>
  <site name="right_foot" quat="0.707107 -0.707107 0 0"/>

both sites, at rest:  local Z axis in world = [0, 1, 0]
                      ->  sum(proj[:2] ** 2) with unit gravity = 1.000000
```

So it is not a near-miss or a tuning offset: at the pose that *should* score a
clean zero, the term scores its exact maximum, deterministically, from the
asset. (`feet_ground_contact` and the site quaternions are consistent — this is
the model, not a transcription slip.)

Affected configs (6): `jump`, `spin`, `ground_pick`, `roller_crouch`,
`roller_slope`, `velocity_rollers`. Every one of them uses the CPU vocabulary's
term name for a different rule.

`mdp.py`'s own docstring is honest about the assumption ("The foot site frame
has Z+ pointing up when flat") — it was simply never checked against the model.
The bug note above it, about `torch.norm` without `dim`, shows the function was
debugged hard for *shape* errors while this *semantic* one sat underneath.

## Why this is invisible

* `robotd` and the ONNX contract check width, not meaning. A policy trained
  under either rule exports and loads identically.
* Both stacks train to completion and produce green numbers. A reward that is
  merely *different* trains a policy that is merely *different* — which looks
  like a tuning problem, not a bug. This is the same class as the `1/N red`
  trap in the port work: nothing crashes, so nothing gets investigated.
* The two repos have no shared test that could fail.

## The rule

**Any reward term implemented on both stacks must have its geometric reference
either taken from the model or asserted against it — never assumed.**

Concretely, when porting or writing a term that depends on a frame:

1. Name the physical quantity, not the sign convention ("tilt of the foot away
   from its rest orientation", not "xy² of gravity").
2. If the reference is a model constant (`stand_z`, foot rest orientation),
   measure it from the model at runtime — as the CPU side does.
3. If it must be a literal, add a test that the literal still holds.
4. Then compare the two implementations at a shared pose. A single pose is
   enough: `flat_feet` would have been caught at STAND in one line.

## Status: fixed on the MJLab side, guarded from both sides

`mdp.py::feet_flat_penalty` now reads its reference from the model
(`_foot_rest_gravity_ref`), matching the CPU side's `foot_flat_ref` approach, and
charges the squared distance from that reference. The docstring keeps the old
formula in a "history (do not regress)" note.

`tests/test_reward_parity.py` (4 tests) guards it, verified by **actually
reverting the fix and watching them fail** rather than assuming they would:

| test | what it does | fails on revert? |
|---|---|---|
| `..._agrees_with_the_cpu_one_at_the_flat_stand` | CPU 1.0 vs MJLab 0.0 at STAND | yes |
| `..._agree_across_a_sweep_of_foot_orientations` | 20 tilted poses, rankings compared | yes |
| `..._confined_to_terms_that_assume_a_frame` | AST census: no other term assumes an axis | yes |
| `..._fixed_rule_would_have_caught_the_original_bug` | negative control on the old formula | n/a (it asserts the control) |

Two things that made writing this harder than it looks, both worth knowing:

* **The docstring is a trap.** The history note quotes `sum(proj[:, :2] ** 2)`,
  so text-searching `mdp.py` finds the explanation of the bug before the fixed
  code and reports a false alarm. Twice. Every check here now comes from
  `_feet_flat_code`, which parses the AST and strips the docstring.
* **A sweep that reuses the fix to compute its expectation is tautological.**
  The first version of the sweep called the same `_foot_rest_gravity_ref` the
  implementation does, so it passed even with the bug reintroduced.
  `_mjlab_flat_rule_from_source` now reads the formula *shape* out of the code
  under test, which is what gives the sweep its teeth.

### What still needs a CUDA box

The guard is static + geometric; it proves the *rule* agrees, not that training
behaves. Unchanged and still required: that every `.params[...]` key is real,
that the weights produce a hop rather than a squat-and-rise, and that the 61-D
obs layout matches the runtime slot (see `runs/JUMP_TASK.md`). The weights were
NOT retuned for this fix — `flat_feet` reads ~1.0 lower per foot while
standing, so the effective penalty changed in 6 configs and the training runs
that produced the current numbers did so under the wrong sign. Treat existing
perf figures for those 6 as suspect until re-run.

Do not "fix" this by aligning the CPU side to MJLab — the CPU rule is the one
with the postmortem behind it (`_stance_flat`'s `std 0.45` note: at `0.15` a
mildly rolled foot scored ~0 and finished runs earned 0.01 of a 0.8 max).

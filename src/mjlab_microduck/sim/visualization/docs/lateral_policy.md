# Pure lateral locomotion: reproducible diagnostics and training

This work changes the simulated actuator contract and provides an independent
policy training/evaluation path. It does **not** establish that the published v8
policy or any locally trained candidate can reliably strafe. There is no browser
heading-hold controller or extra yaw command.

The implementation is isolated under `src/mjlab_microduck/sim/visualization`:
`bam.py` owns the CPU motors, `physics.py` owns the optional world/body adapter,
and `lateral/` owns task configuration, registration, fine-tuning and validation.
All added modules, command entry points, scene XML, documentation and tests live
in that directory. There are no compatibility files in `sim/`, `tasks/` or
`scripts/`. The scene references the upstream robot XML/meshes without copying
them; Studio includes the extension scene directory in its scene catalog.

## Physics contract

The legacy `duck-body` default is unchanged. Select the training-compatible CPU
path explicitly:

```sh
uv run duck-body --headless --keyframe HOME \
  --scene src/mjlab_microduck/sim/visualization/scenes/scene_velstand.xml \
  --actuator-model bam
```

This scene uses `robot_allcollisions_backlash.xml`. The BAM adapter applies the
v8 M6 motor, armature, friction constraint and 3–6 physics-step command delay.
It preserves BAM's default 1.75A firmware current limiter, including its physical
PWM/back-EMF constraints, rather than replacing it with an unlimited motor.
Position feedback and observed joint velocity include serial backlash; back-EMF
uses motor velocity. Velocity observation history advances on simulation ticks,
so additional telemetry readers do not change it. The solver uses 5ms
`implicitfast`, Newton, pyramidal contacts, 30 iterations and 50 line-search
iterations, with a forward refresh after each four-step control period.

The pinned upstream CPU BAM implementation compares DOF-friction constraint
IDs with joint IDs. These differ for the floating root/backlash model. The local
NumPy adapter scatters those constraints by DOF, and its M6 friction budget is
tested against the training Torch implementation. It imports neither Torch nor
mjlab at runtime. These correctness fixes alone did not pass the lateral gate.

For a matching daemon simulation profile, use action scale 1.0 and disable target
low-pass filtering (`legs_lowpass = head_lowpass = 1.0`). This is a simulation
profile, not a recommendation to change real-robot tuning. Studio owns Docker
dependencies and profile selection.

## Separate training task

`Mjlab-Lateral-Flat-Backlash-MicroDuck` retains the 61-observation/14-action
actor interface. It explicitly samples rest, pure lateral movement, stop and
reverse, while retaining longitudinal and turning families. Published v8 task
definitions and cached models are not overwritten.

This task alone expands the training critic from 76 to 85 observations. Its
nine privileged fields describe segment-relative yaw (sine/cosine), XY
displacement, mean yaw rate, active phase, remaining phase fraction, command
family and initial direction. They let the value function distinguish identical
local robot states with different accumulated penalties or upcoming commands.
These fields never enter the exported actor or the runtime protocol. Lateral
checkpoints saved before this critic change cannot be resumed directly; an ONNX
actor export can still warm-start a new critic.

The lateral rewards require actual progress along the segment's original side
axis. Accumulated turning and forward displacement are penalized in that same
frame; otherwise a body-frame velocity reward can pay for an arc. These are
training rewards, not heading targets supplied to the deployed actor. Slow yaw
drift and alternating step wobble have separate penalties. Optional frozen-base
action anchoring protects non-lateral examples, and optional mirror loss acts
only during training. Both optional losses join each PPO minibatch's gradient
update; they do not run a separate optimizer step or transform inference actions.

```sh
MPLCONFIGDIR=/tmp/microduck-mpl \
WARP_CACHE_PATH=/tmp/microduck-warp \
uv run python -m mjlab_microduck.sim.visualization.lateral.finetune \
  --policy /path/to/official/velstand.onnx --output /path/to/new-run \
  --device cpu --num-envs 16 --iterations 100 --seed 0
```

The same entry point accepts a future Linux `cuda:0` device. CPU reset, stepping
and a bounded PPO update have been exercised on the current Mac; this does not
promise fast convergence. A fresh ONNX warm start restores the actor and frozen
normalization exactly, but starts a new critic, optimizer and exploration
distribution. `--resume` restores a local runner checkpoint while resetting
environment trajectories and RNG. Every new invocation saves its actual dirty
source/configuration snapshot and hashes. Checkpoints and ONNX exports remain
explicitly marked unvalidated candidates.

## Acceptance before deployment

```sh
uv run python -m mjlab_microduck.sim.visualization.lateral.validate \
  --policy /path/to/candidate.onnx --output /path/to/report.json \
  --seeds 0 1 2
```

This gate uses actual MuJoCo trunk displacement and unwrapped yaw under pure
`[vx=0, vy=+/-0.3, vyaw=0]` commands. Each moving segment lasts 10s; stops and a
second leftward segment expose restart/direction-change failures. Default
limits require at least 0.2m signed lateral progress, at most 10 degrees maximum
yaw excursion, forward drift at most 25% of lateral progress, and upright
posture throughout. Small yaw with no movement is a failure.
During a stop, the entire trajectory must stay within 0.05m of its start and
within the same yaw limit; moving away and returning does not count as stopping.
Reports include policy/validator hashes and the MuJoCo/NumPy versions.

Passing this deterministic CPU gate is necessary but insufficient. Keep
independent seeds, test forward/back/turn/get-up regressions, then exercise the
real `robotd` socket and Studio controls with rendering enabled. Record control
frequency, requested/applied commands, safety state and final stops. Do not
replace the official cache or change the default policy based on training
reward, one successful run, or a partially passing report.

## Observed baseline, 2026-10-25

The official v8 `velstand.onnx` (SHA256
`2fc7f8ee6acd7d0d3f66a99927b5d5a56589aeed02ec714de329b252d5ccf492`)
also showed inconsistent lateral motion in a local nominal evaluation built
on mjlab/MuJoCo Warp. This used a locally modified diagnostic task, not an
untouched upstream task: flat terrain, all-collisions/backlash geometry, an
upright initial pose, zero head/body commands, fixed 7.4V supply/drop gain 0.1,
and disabled pushes, observation corruption, IMU mounting error and model DR.
The local reward/critic/termination settings were also changed; no training or
critic inference was performed and no resets occurred during the probe.

With one seed (0) and three parallel environments, deterministic actor outputs
and zero commanded yaw, none of the nine 10-second lateral segments passed the
local gate above. These are local acceptance limits, not official criteria.
Two initial left commands moved only about 6mm; rightward segments drifted by
roughly 10–17 degrees. All rest stages passed, with no falls. This has not been
validated on physical hardware or confirmed by the upstream maintainers.

The NumPy BAM world similarly reproduced deadpoints and curved travel. Restoring
the current limiter did not change that baseline: observed motor torque stayed
below the limiter's onset. These results reproduce the behavior outside the
browser, but cannot distinguish limitations of the released policy from a
mismatch with the local evaluation configuration. They do not establish a
confirmed upstream source-code defect, convergence of the new lateral recipe,
or justification for deploying its candidates.

A subsequent bounded CPU run exercised the 85D critic and combined PPO/action
preservation gradients, using 16 environments, 64 steps and 64 updates (65,536
rollout samples), with initial exploration std 0.03. Its four saved candidates
started lateral motion but each failed all nine moving-segment checks because
of drift. None was deployed. This short run is not evidence that the task has
converged; passing unit tests or increasing training reward cannot replace the
motion gate.

# MicroDuck tricks and stairs

This document records the first simulation setup for three new behaviors:

| Task | What it teaches now | Current status |
|---|---|---|
| `Mjlab-Stairs-MicroDuck` | Walking over flat terrain mixed with dedicated pyramid stairs | Ready for a real training run |
| `Mjlab-Stairs-Standard-MicroDuck` | Five 170 mm risers, 280 mm treads, forward command, and top-landing success | Visible full-size challenge, train after low-rise progression |
| `Mjlab-Headstand-Flat-MicroDuck` | A head-supported freeze with no foot contact | Experimental scaffold, smoke-tested |
| `Mjlab-Backflip-Flat-MicroDuck` | Backward rotation frontier plus a feet-first landing gate | Experimental staged scaffold, smoke-tested |

The 29 August 2026 smoke captures are diagnostic only. They prove that the
tasks build, step on the RTX 4090, write checkpoints, and record video. They do
not claim that the robot has learned the behaviors yet.

## What the hardware makes plausible

MicroDuck is approximately 25 cm tall and 800 g, with 14 Dynamixel XL330
servos. The official project already demonstrates walking, rolling, grasping,
and stand-up behavior. The actuator model and the 50 Hz shared observation
contract are preserved here because sim-to-real fidelity matters more than a
large simulated skill list.

The practical order is:

1. **Stairs first.** Train 5 mm, 10 mm, then 15 mm risers in
   `Mjlab-Stairs-MicroDuck`, then transfer the policy to
   `Mjlab-Stairs-Standard-MicroDuck`. The latter is a representative full-size
   challenge with 170 mm risers and 280 mm treads, not a jurisdiction-specific
   building-code certification. Its route starts beside the robot and ends on
   a flat top landing. A forward-only command range and potential-based route
   and height shaping give learning a useful walking signal, while a latched
   upright arrival term defines success without paying forever on the landing.
2. **Headstand as a freeze.** The task starts in the measured tucked pose with
   the flat head top down, rewards head contact and stillness, and rejects foot
   contact. The next curriculum stage should learn stand to tuck to headstand,
   then add small pose and friction randomization.
3. **Backflip as a staged experiment.** First learn a crouch, a small hop, and
   a tucked backward hop. Then add a motion-reference or trajectory-optimization
   stage, followed by residual PPO and domain randomization. A pure from-stand
   PPO run is not a credible first bet for a full aerial 360 degree flip on
   these small servos.

## Research used

- [MicroDuck official project](https://github.com/pollen-robotics/microduck)
  documents the robot scale, servos, runtime, and existing behaviors.
- [MicroDuck RL repository](https://github.com/pollen-robotics/microduck_rl)
  is the upstream mjlab and PPO baseline used by this project.
- [OPT-Mimic](https://www.cs.ubc.ca/~van/papers/2022-opt-mimic/index.html)
  and its [trajectory optimization code](https://github.com/yunifuchioka/opt-mimic-traj-opt)
  show why a physically optimized reference can make a Solo8 backflip
  learnable when a kinematic sketch does not. They also highlight underactuated
  dynamics, upper-body inertia, and avoiding velocity-target feedforward for
  sim-to-real transfer.
- [Stage-Wise CMORL](https://arxiv.org/abs/2409.15755) uses staged curricula for
  dynamic quadruped maneuvers, which motivates the reverse-curriculum late
  landing states in the backflip task.
- [Robot Parkour](https://proceedings.mlr.press/v229/zhuang23a/zhuang23a.pdf)
  and [Extreme Parkour](https://extreme-parkour.github.io/) support explicit
  terrain curricula, contact-aware rewards, and randomized obstacle geometry.
- [Blind Bipedal Stair Traversal](https://arxiv.org/abs/2105.08328) is a useful
  biped reference for proprioceptive stair policies, even though Cassie is
  much larger than MicroDuck.
- The official [Isaac Lab terrain API](https://isaac-sim.github.io/IsaacLab/develop/source/api/lab/isaaclab.terrains.html)
  documents the pyramid-stair construction mirrored by the dedicated task.
- The official [MuJoCo sample programs](https://github.com/google-deepmind/mujoco/tree/main/sample)
  show the low-level pattern used by the native viewer: advance with `mj_step`,
  update the scene, render with `mjr_render`, and read pixels for recording.
  MuJoCo's samples are intentionally single-model examples. The four-square
  view comes from mjlab's vectorized environments, where independent worlds
  share one model and are placed at separate terrain origins.
- The [ICC A117.1 stair draft](https://www.iccsafe.org/wp-content/uploads/asc_a117_1/A117.1_2023-legislative-draft-for-final-draft-2025-9-1.pdf)
  is a reference for conventional stair proportions. The values used here
  are deliberately chosen as a clear simulation challenge and still need
  physical validation before any real-robot attempt.

## Training commands

Run the stair curriculum on the local GPU after the 64 environment smoke test:

```powershell
uv run train Mjlab-Stairs-MicroDuck --env.scene.num-envs 4096 --agent.max-iterations 4000
```

After the low-rise policy has a useful checkpoint, train the full-size route
with the same 61D policy interface:

```powershell
uv run train Mjlab-Stairs-Standard-MicroDuck --env.scene.num-envs 4096 --agent.max-iterations 10000
```

To make the transition explicit, the supported handoff helper stages a flat
walking `model_*.pt` into the stair experiment and launches PPO with the stair
configuration:

```powershell
uv run python scripts/train_stair_from_walking.py `
  --walking-checkpoint logs/rsl_rl/velocity/<run>/model_XXXX.pt `
  --num-envs 4096 --stair-iterations 10000
```

The flat walking checkpoint must be a local PPO checkpoint, not an ONNX export.
The official simulator's ONNX walking files are useful for playback, but they
do not contain the PPO optimizer and critic state needed for a direct fine-tune.
The local ONNX playback path preserves the XML actuator limits used by the
official simulator. A firmware-style `--current-limit 1.75` clamp is available
for explicit sim-to-real stress tests, but it is disabled by default because it
prevents the manufacturer walking export from producing forward motion here.

For a GPU-friendly four-robot visible check, use `play` with a checkpoint from
either stair run:

```powershell
uv run play Mjlab-Stairs-Standard-MicroDuck `
  --checkpoint-file logs/rsl_rl/microduck_standard_stairs/<run>/model_XXXX.pt `
  --num-envs 4 --viewer native --video True --video-length 300 `
  --video-width 1280 --video-height 720
```

For a first reproducible local capture with offline TensorBoard logging:

```powershell
$env:WANDB_MODE = "offline"
uv run train Mjlab-Stairs-MicroDuck --env.scene.num-envs 64 `
  --agent.max-iterations 250 --agent.logger tensorboard `
  --agent.save-interval 50 --agent.upload-model False `
  --video True --video-length 200 --video-interval 100
```

The two experimental tasks use the same pattern:

```powershell
uv run train Mjlab-Headstand-Flat-MicroDuck --env.scene.num-envs 4096 --agent.max-iterations 3000
uv run train Mjlab-Backflip-Flat-MicroDuck --env.scene.num-envs 4096 --agent.max-iterations 10000
```

Use the dashboard while these run:

```powershell
uv run python scripts/serve_dashboard.py --host 0.0.0.0 --port 9999
```

Then open `http://localhost:9999` or `http://<tailscale-ip>:9999`. The server
is local-only apart from the interface binding and has no upload path.

## Files added

- `src/mjlab_microduck/tasks/microduck_stairs_env_cfg.py` isolates low-rise
  stairs from the mixed rough-terrain task.
- `src/mjlab_microduck/tasks/microduck_standard_stairs_env_cfg.py` builds the
  visible full-size staircase and its forward route/top-landing objective.
- `src/mjlab_microduck/tasks/microduck_headstand_env_cfg.py` defines the
  head-only contact and stillness experiment.
- `src/mjlab_microduck/tasks/microduck_backflip_env_cfg.py` defines the staged
  rotation and landing experiment.
- `scripts/train_stair_from_walking.py` performs the flat-walking checkpoint
  handoff before the standard-stair fine-tune.
- `src/mjlab_microduck/tasks/mdp.py` contains the contact gates, reverse reset,
  rotation frontier, and landing reward.

Do not deploy a trick policy to hardware from a short smoke run. Before any
hardware test, evaluate several seeds, inspect the MP4 clips, check peak
contact forces and actuator torques, export with `scripts/export.py`, and use a
physical safety tether and a low-energy progression.

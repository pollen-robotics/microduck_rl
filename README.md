# Microduck RL

<img width="2215" height="884" alt="image" src="https://github.com/user-attachments/assets/5db7cc83-b3ce-4f7c-83f0-0572a63baed7" />


RL training environments for [Microduck](https://github.com/pollen-robotics/microduck) —
a ~800 g, ~25 cm tall bipedal robot — built on
[mjlab](https://github.com/mujocolab/mjlab) (MuJoCo Warp) with PPO.
Policies are trained here at 50 Hz, exported to ONNX, and deployed on the real
robot by the runtime in [pollen-robotics/microduck](https://github.com/pollen-robotics/microduck).

<!-- HERO VIDEO — real robot montage: walking, standup, roulade, roller skating.
     Keep it short (~30 s) and real-robot-first: this is the "why should I care" shot. -->

https://github.com/user-attachments/assets/50c3d537-8db2-4005-9d9c-3472faeec4d0

The repo encodes the full sim2real recipe: [BAM](https://github.com/Rhoban/bam)
actuator physics, domain randomization, backlash simulation, and the
reward-design lessons that made it work
(see [AGENTS.md](AGENTS.md) for the distilled playbook).

## Quickstart

Requires a CUDA GPU (training runs through MuJoCo Warp) and [uv](https://docs.astral.sh/uv/).

> **On ARM boxes (DGX Spark / GB10, Jetson):** `uv sync` pulls ~2 GB of CUDA
> wheels on first run and uv's default 30 s HTTP timeout can abort mid-download.
> Export `UV_HTTP_TIMEOUT=600` for the first sync. 

```bash
git clone https://github.com/pollen-robotics/microduck_rl
cd microduck_rl

# train the walking policy (uses your GPU; ~1-2 h for a usable gait at 4096 envs)
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096

# watch a trained policy in the viewer
uv run play Mjlab-Velocity-Flat-MicroDuck --wandb-run-path <entity/project/run_id>

# export to ONNX for deployment
uv run scripts/export.py Mjlab-Velocity-Flat-MicroDuck --wandb-run-path <...>
uv run publish --onnx output.onnx --repo <user>/microduck-<name> --kind episodic --duration-s 4.0   # share it (see "Publishing a policy")

# drive the exported policy in CPU MuJoCo with the keyboard
uv run scripts/infer_policy.py --walking output.onnx
```

Resume from a checkpoint:

```bash
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096 \
    --agent.run-name resume --agent.load-checkpoint model_29999.pt --agent.resume True
```

No GPU? Add `--hf-jobs` to any train command to run it on Hugging Face Jobs
instead of locally (see [scripts/hf/README.md](scripts/hf/README.md)).

## Tasks

`uv run list-envs` prints the live registry. Flat/Rough variants exist where noted.

<!-- SHOWCASE GRID — one short GIF per task family (sim or real), 3 per row.
     Priority order if you only record a few: Velocity, VelStand (fall+recover),
     Roulade, SitStand, Rollers/Swizzle, BallKick. -->

| Task id | Terrain | Description |
|---|---|---|
| `Mjlab-Velocity-{Flat,Rough}-MicroDuck` | flat/rough | **The main task**: walking with velocity commands + head-pose commands |
| `Mjlab-VelStand-{Flat,Rough}-MicroDuck` | flat/rough | Walking + fall recovery in one policy |
| `Mjlab-StandUp-{Flat,Rough}-MicroDuck` | flat/rough | Stand up from face-down/face-up/sitting, then hold the stand + body-pose control |
| `Mjlab-SitStand-{Flat,Rough}-MicroDuck` | flat/rough | Commanded sit ↔ stand in one policy, gently, head commandable |
| `Mjlab-GroundPick-{Flat,Rough}-MicroDuck` | flat/rough | Crouch and touch the ground with the mouth tip, return to stand |
| `Mjlab-BallKick-Flat-MicroDuck` | flat | Kick a 70 mm / 15 g ball forward (actor is ball-blind) |
| `Mjlab-Roulade-Flat-MicroDuck` | flat | Forward roll over the head, land back on the feet |
| `Mjlab-Velocity-Flat-MicroDuck-Rollers` | flat | Roller-skate velocity tracking (passive wheels under the feet) |
| `Mjlab-Velocity-Swizzle-MicroDuck` | flat | Classic symmetric swizzle skating |
| `Mjlab-RollerCrouch-Flat-MicroDuck` | flat | Crouch while gliding on rollers |
| `Mjlab-RollerSlope-Flat-MicroDuck` | slope | Glide down slopes on rollers |
| `Mjlab-RollerStandUp-Flat-MicroDuck` | flat | Stand up from the ground onto the wheels |
| `Mjlab-Spin-Flat-MicroDuck` | flat | Fast spin in place on rollers |

At deployment the runtime hot-swaps these policies (walk / recover / trick)
behind a shared 61-dimensional observation contract, so any of them can take
over the robot at any moment. `scripts/infer_policy.py` rehearses exactly that:

```bash
uv run scripts/infer_policy.py --walking walk.onnx --standing stand.onnx \
    --sitstand sitstand.onnx --roulade roulade.onnx --new-cmd-obs
```

Keyboard-driven (velocity commands, `G` ground pick, `Y` sit/stand, `R` roulade,
`K`/`L` kicks); `--debug`, `--save-csv`, `--record` support sim2real comparisons.
The servos are simulated with the same BAM M6 XL330 model the policies are
trained against (voltage control + load-dependent friction, via
`bam.mujoco.MujocoController`); `--vin` / `--vin-drop-gain` / `--kp-fw` pin the
training DR ranges to one value, `--no-bam` falls back to the XML PD actuators.

### Backlash variants

Every main task has a **Backlash** twin that trains on a model with ±1° of gear
play (2° total) in series with each of the 14 servo joints: insert `-Backlash`
before `MicroDuck` in the task id, e.g. `Mjlab-Velocity-Flat-Backlash-MicroDuck`.

The backlash is modeled properly for sim2real: each servo gets an unactuated
`passive_<joint>_backlash` hinge, and because the real encoder sits on the
output side of the play, both the firmware PD emulation
(`BacklashEncoderBamActuator`) and the `joint_pos`/`joint_vel` observations
read *through* the backlash (`qpos[servo] + qpos[backlash]`). Observation and
action dims are unchanged, so ONNX export and the runtime need no changes.
See `src/mjlab_microduck/tasks/backlash.py`.

## Actuator model

All tasks use the [BAM](https://github.com/Rhoban/bam) M6 actuator model for
the Dynamixel XL330 (voltage control law, back-EMF, Coulomb/Stribeck/load-dependent
friction), with per-env domain randomization on battery voltage, voltage sag
under load, command delay, and friction magnitude
(`FrictionDRBamActuator` in `src/mjlab_microduck/actuator/`).

At this scale — tiny servos driving a ~800 g biped — actuator fidelity is most
of the sim2real gap, which is why the actuator is modeled down to its voltage
control law instead of an ideal PD.

## Robot models

MJCF models live in `src/mjlab_microduck/robot/microduck/` and are exported
from Onshape with [onshape-to-robot](https://github.com/Rhoban/onshape-to-robot),
one `config_mjcf_*.json` per model:

| XML | Used by |
|---|---|
| `robot_walk.xml` | Velocity (stripped trunk/head contacts — falling is cheap) |
| `robot_groundcontact.xml` | VelStand, StandUp, SitStand, GroundPick, BallKick, Roulade (curated collision set for the parts that touch the floor — body can physically lie on the ground; formerly `robot_allcollisions.xml`) |
| `robot_groundcontact_rollers.xml` | Roller tasks (passive wheels) |
| `robot_allcollisions.xml` | True full-collision model — every part has a collision geom. No task uses it yet |
| `robot_*_backlash.xml` | Backlash task variants (generated by `add_backlash.py`) |

`scene*.xml` files wrap the robots with a floor + keyframes (STAND/SIT/FOLD)
for quick viewing and for `infer_policy.py`.

<!-- IMAGE — side-by-side render: walk model vs rollers model (or a collision-geom
     visualization). One image here makes the model-variant story instant. -->

## Project structure

```
src/mjlab_microduck/
├── robot/
│   ├── microduck/                    # MJCF exports, export configs, scenes, add_backlash.py
│   └── microduck_constants.py        # robot cfgs, HOME frame, BAM actuator cfg
├── actuator/friction_dr_bam.py       # BAM + friction DR + backlash encoder feedback
├── tasks/
│   ├── __init__.py                   # task registration (base + backlash variants)
│   ├── mdp.py                        # rewards, events, observations, custom classes
│   ├── backlash.py                   # make_backlash_variant() env-cfg wrapper
│   └── microduck_*_env_cfg.py        # one cfg module per task family
├── train_cli.py                      # `train` script (identical to mjlab's)
├── train_hook.py                     # intercepts `train ... --hf-jobs`
└── hf_jobs.py                        # Hugging Face Jobs submission
```

Conventions worth knowing:

- The observation layout is shared across every policy (61-dim actor obs:
  48 proprioception + commands `[twist(3), head_pose(4), body_pose(6)]`), which
  is what makes runtime policy hot-swapping possible. Envs that don't use a
  command slot zero-pad it rather than dropping it.
- Unactuated joints are all named `passive_*` (roller wheels, backlash
  hinges); actuators, joint observations and pose rewards select servo joints
  with `^(?!passive_).*`.
- Domain-randomization toggles are `ENABLE_*` booleans at the top of each
  env cfg file.
- Joint layout (14 servos): 0–4 left leg (hip_yaw, hip_roll, hip_pitch, knee,
  ankle), 5–8 neck/head (neck_pitch, head_pitch, head_yaw, head_roll),
  9–13 right leg.
- The exporter bakes the observation normalizer into the ONNX graph — always
  deploy ONNX produced by `scripts/export.py`, never a hand-converted
  checkpoint, or the policy sees unnormalized observations at runtime.

[AGENTS.md](AGENTS.md) documents the env-building workflow and the reward-design
rules learned across the project (also aimed at AI coding agents working in
this repo).

## Publishing a policy

`uv run publish` puts a policy on the Hugging Face Hub in the shape the robot's
daemon loads: one `policy.onnx` with the observation normalizer baked in, a
`manifest.json` following schema 2 of the
[microduck policy manifest](https://github.com/pollen-robotics/microduck/blob/main/docs/policy-manifest.md),
and a README saying how to run it. Anyone with a microduck can then install it
with one command, no daemon release needed.

```bash
# From a wandb run — exports through the one safe path, then uploads
uv run publish --task Mjlab-PoliteBow-Flat-MicroDuck \
    --wandb-run-path <entity/project/run_id> --checkpoint 3000 \
    --repo <user>/microduck-polite-bow --kind episodic --duration-s 4.0 \
    --description "Bows from a two-foot stand and comes back up."

# From an ONNX you already exported (validated, not re-exported)
uv run publish --onnx output.onnx --repo <user>/microduck-flamingo \
    --kind perpetual --unwind-s 1.5 --twist-help "[flag, side, 0]"

# A new gait for a slot
uv run publish --onnx output.onnx --repo <user>/microduck-my-walk --kind perpetual --slot walk

# See what would be uploaded without touching the Hub
uv run publish --onnx output.onnx --repo <user>/microduck-bow --kind episodic --duration-s 4.0 --dry-run
```

Then on a robot:

```bash
sudo robotctl policy add polite-bow <user>/microduck-polite-bow   # episodic: length comes from the manifest
sudo robotctl policy add flamingo <user>/microduck-flamingo --hold 5   # held pose: you pick how long
sudo robotctl policy load walk <user>/microduck-my-walk                # gait: into the walk slot
robotctl robot do polite-bow
```

What `--kind` means, and what each needs:

- **episodic** — runs for `--duration-s` and returns itself to a standing pose
  (kicks, roulade, a bow). Add `--chain` if holding the button should repeat it.
- **perpetual** — runs until told otherwise. Two shapes:
  - a **gait** (a new walk or stand): add `--slot walk` (or `stand`) and
    nothing else; the owner installs it with `robotctl policy load walk <repo>`.
  - a **held pose** (the flamingo): give `--unwind-s`, how long the daemon
    drives the idle twist (`--idle`, zeros by default) before handing back to
    the gait, so the robot is not let go of on one foot. The owner runs it as a
    one-shot with `policy add ... --hold <seconds>`.

Before anything is uploaded, `publish` checks the graph is `[1,61] -> [1,14]`
(a 51-D legacy policy is refused with a message), runs it on plausible inputs
and refuses NaNs or a constant output, fills the `training` block from git and
wandb (task, commit, branch, dirty flag, run, checkpoint), and refuses to
overwrite an existing `.onnx` in the repo without `--force`. Repos are created
private; `--no-private` for public, `--tag v1` to tag the revision.

Only constant-command policies are publishable this way. Phase-driven moves
(the ground pick) and the posture-flag sit↔stand are driven by the daemon
itself and live in the official set, `pollen-robotics/microduck-policies`.

## Tests

```bash
uv run --with pytest pytest tests/
```

CPU-only config-invariant and reward-function regression tests — they lock in
joint-index mappings, reward sign conventions, and NaN guards.

## Related projects

- [microduck](https://github.com/pollen-robotics/microduck) — the Microduck project home, including the onboard runtime that runs the exported policies
- [mjlab](https://github.com/mujocolab/mjlab) — the training framework (MuJoCo Warp + rsl_rl)
- [BAM](https://github.com/Rhoban/bam) — better actuator models, by Rhoban

## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.
3D model files are licensed under Creative Commons BY-SA-NC.

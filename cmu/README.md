# cmu/: the CMU robot layer

Everything that is specific to our MicroDuck, in one directory on top of Pollen's unchanged training repository.
Start with [ROBOT.md](ROBOT.md) (the robot as built and the runtime) and [SIM2REAL.md](SIM2REAL.md) (the gap and
the open problems); the quickstarts are in the [top-level README](../README.md).

The physical build (parts list, printed parts, wiring, assembly guide, bring-up and the runtime as it stood on 2026-10-23) is published separately at https://github.com/qilinxiaoxiang/cmu-microduck; project page https://shawnxiang.com/microduck.

## Layout

- `robot-condition.json`: the current measured facts the sim needs (servo rail, IMU rate, added delay, dropped
  reads, deployed policy, sole) with verification dates. `simcheck/sim_suite.py` derives its "real" case flags from it.
- `model/`: the two mass variants (`variants/plugged.json`, `variants/battery.json` as deltas; `manifests/` as full
  part lists), `apply_masses.py` (writes the parts into Pollen's MJCF as explicit inertials; CLI exports a scene XML),
  `make_variants.py` (regenerates the manifests and prints mass and centre of mass), `sole/` (sole sweep and the 1:1
  wrap template). Measured and estimated numbers: `model/README.md`.
- `mac/`: the MacBook runtime that drives the robot (`walk_wired.py`, `walk_brain.py`), the CPU sim check
  (`brain_sim_check.py`), the pickup chain (`pickup.py`, `pickup_sim.py`, `target_ground.py`, `orange_target.py`,
  `duck_cam.py`, `jaw_ctl.py`), the jitter study scripts (`jitter-20261018/`) and their fake-link tests.
- `pi/`: head-board side: the IMU streamer, the stand pose, joint sign and offset conventions, jaw limits, battery
  guard, and `microduck_rt/` with the community runtime's constants.
- `whole/`: whole-robot studies the sim check depends on: `dynamics/` (as-built manifest and the earlier v19 one,
  the real-gap study with its scripts), `delay-repair-20260912/` (model loader with BAM, observation compensation),
  `electrical/` (the NP-F970 battery entry), `runtime-overlay/` (the deployed policies `walk61`, `stand61`,
  `groundpick61` with their source hashes, and the ft3 candidate in `models-ft3-it999/`).
- `simcheck/`: the eight-case acceptance suite and its reference rows for walk61 and ft3.
- `real-logs/`: the 2026-10-18 robot tick logs and the real-versus-sim jitter numbers.
- `finetune/`: the walk61 fine-tune pipeline (`modal/`, with the reward audit gate, tracking guard and cost caps) and
  its Kaggle launcher (`kaggle/`); `modal/FINETUNE-DELTA.md` is the run history.
- `aim_pick/`: the aiming crouch (`Mjlab-AimPick-Flat-MicroDuck[-CMU-Plugged|-CMU-Battery]`, groundpick61's crouch
  with an object the beak has to reach): its reward audit (`audit.py`, `reward-audit.json`), the literature recipe,
  the cost estimate and what a run still needs (`README.md`).
- `hardware/`: the battery backpack design materials (`battery-backpack/`: STL files, mating parts, envelopes, checks,
  masses, wiring page, parts list, CAD source) and the print pipeline for the RoboClub printers (`printing/`: profiles,
  slicer commands, orientation and brim rules, the ready V1.20 plate). `hardware/README.md` is the index.
- `simulator/`: the browser simulator and the Microduck macOS app: the game with our duck as built (legs only), the
  ground and table pickup pages, the real-robot monitor and remote control (`app/robot-live.html`), the desktop
  launcher (`desktop/`) and the jaw PPO research (`rl/`). Run it with `cd cmu/simulator/app && npm ci && npx vite`;
  checks: `npm run test:physics` and `npm run test:ground`. `simulator/desktop/README.md` covers the app and the robot link.

## Three ways in

- Simulate and judge. `uv run cmu/mac/brain_sim_check.py --model asbuilt --volt 7.0 --drop 0.04 --vx 0.25` walks the
  as-built model with the robot's control logic; `uv run cmu/simcheck/sim_suite.py <onnx-dir> out.jsonl` scores any
  policy directory against the reference rows (`--variant plugged|battery` runs it on the other mass models). Flags worth knowing: `--com-x 5` (shift the centre of mass in mm),
  `--imu-pitch 4` (IMU misalignment), `--imu-hz 35` (hold the IMU below 50 Hz), `--drop 0.08`, `--kp-rl`,
  `--tick-log` (a real-log-shaped trace), `--head-yaw/--head-pitch`, `--gp-at/--gp-phase` (ground pick),
  `--manifest cmu/model/manifests/battery.json` (another mass model).
- Train. `Mjlab-Velocity-{Flat,Rough}-MicroDuck-CMU-{Plugged,Battery}` are Pollen's velocity task on our masses;
  `uv run train <task> --env.scene.num-envs 4096` on a CUDA GPU, export with `uv run scripts/export.py`. The exported
  ONNX drops into `walk_wired.py --models <dir>` unchanged as long as the 61-input layout is kept. To fine-tune the
  deployed walk61 instead of training from scratch, use `finetune/` (it rebuilds the actor from the ONNX, warms up a
  critic, applies a registered delta, and refuses to launch without a reward audit pass).
- Close the perception loop. `uv run cmu/mac/pickup_sim.py --obj 0.35 -0.60 --volt 7.0 --pitch-offset 4.75` renders
  the head camera in MuJoCo, detects the orange object, projects it to the floor, walks, crouches, grips and rises
  with the same controller that runs on the robot. A learned replacement for any piece (a slower walk, a crouch
  policy with the object in view, a grasp policy) plugs in at the mode boundary in `walk_brain.py`.

## Measured, estimated, missing

- Measured: everything in `robot-condition.json`; the as-built part masses of 2026-10-15 (printed parts, boards,
  screws, cables); the sole sweep; the jitter and servo statistics in `SIM2REAL.md`; the reference suite rows.
- Estimated (replace when weighed): the A7Z head board (18 g, bounded 15 to 20 g), its antenna (1.5 g), the battery
  backpack parts (plan masses from `bp_mass.json`), the fitted camera board (4 g entry of the previous module).
- Missing: a position trajectory of the real robot (only IMU and encoders are logged); a weighed total of the robot in
  each form; the A7Z camera path.

## How this directory is refreshed

`mac/`, `pi/`, `whole/`, `real-logs/`, `model/sole/`, `simulator/` and the copied files under `hardware/` (everything there but
the `README.md` pages, `hardware/battery-backpack/stl/` and `hardware/battery-redesign/`) are generated copies of the robotics working tree on the robot's MacBook. `sync_from_robotics.py` copies them with a short list of rewrite rules (path hops between the two
layouts, no private venv re-exec, relative manifest paths, private addresses replaced). Each rule must match or the
sync stops, so a drifted source is noticed. `simulator/` is the committed content of the microduck-simulator checkout
(Pollen's Hugging Face Space plus our commits), without its 1.5 GB of September result archives. The workflow on that machine:

```bash
python3 cmu/sync_from_robotics.py --check   # what would change
python3 cmu/sync_from_robotics.py           # copy
uv run --with pytest pytest tests/ cmu/mac cmu/pi
git add -A cmu && git commit && git push origin main
```

Do not edit the copied files in this fork: fix the source and resync. `model/`, `simcheck/`, `finetune/`,
`robot-condition.json`, the three Markdown pages and the `hardware/**/README.md` pages, `hardware/battery-backpack/stl/` and `hardware/battery-redesign/` are owned here. Pollen's updates arrive through the `develop`
mirror (see the top-level README).

## Conventions

- Model radians everywhere in logs and the sim; the servo tick conventions (signs, knee offsets, the ID 5 hip-yaw check) are
  applied only at the wire by `walk_brain.py` and `pi/joint_signs.py`.
- Joint order in `q`, `dq`, `action`: 0 to 4 left leg (hip yaw, hip roll, hip pitch, knee, ankle), 5 to 8 neck pitch,
  head pitch, head yaw, head roll, 9 to 13 right leg.
- 61 observation inputs, 14 outputs, action scale 1, gain 200: any policy that keeps this contract runs on the robot
  without a runtime change.
- "Real" means the robot's measured condition from `robot-condition.json`; "stress" means harsher, report only.
- Current facts come from `robot-condition.json`; the Markdown pages explain, they do not override.

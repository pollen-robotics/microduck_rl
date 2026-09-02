# move-away — step aside from an approaching person, without losing sight of them

A **CPU MuJoCo behavior demo**, not robot autonomy. It shows how far a stock
exported walking policy can be pushed by a small scripted supervisor, and it
doubles as a rehearsal harness for the velocity-command interface the runtime
uses.

No new policy is trained. An exported walking ONNX is driven by a velocity
command produced by a state machine:

```
IDLE -> RETREAT -> TURN -> CLEAR -> DONE
```

On top of that, an **independent kinematic gaze layer** aims head yaw/pitch at
the person for the whole sequence, so the person stays centred in the head
camera even while the robot turns away from them.

## Running it

The policy is a required argument — no policy is committed to this repository:

```bash
# headless, metrics only
python scripts/behavior_demos/move_away/run_demo.py --policy path/to/walking.onnx

# machine-readable summary
python scripts/behavior_demos/move_away/run_demo.py --policy path/to/walking.onnx --json

# render annotated frames (adds imageio + Pillow)
python scripts/behavior_demos/move_away/run_demo.py \
    --policy path/to/walking.onnx --render-dir /tmp/move_away_frames --fps 50
ffmpeg -framerate 50 -i /tmp/move_away_frames/f%05d.png \
    -c:v libx264 -pix_fmt yuv420p -movflags +faststart move_away.mp4
```

The demo uses `src/mjlab_microduck/robot/microduck/scene_move_away.xml`, which is
`scene.xml` plus a scripted mocap "person" prop.

## Measured behavior

With the stock `alpha_walking` policy, defaults, 22 s:

| metric | value |
| --- | --- |
| final state | `DONE`, upright |
| person visible | 1100 / 1100 control steps (`lost_steps=0`) |
| max off-axis angle | 2.3° from the camera centre |
| heading change | +82.8° (90° commanded) |
| trunk height | z = 0.116 (nominal), min 0.114 |

Also checked: `--turn-sign -1` mirrors the maneuver (−96.5°), a faster person
(`--person-speed 0.20`) keeps 1100/1100 visibility, and a longer run
(`--seconds 28`) stays upright at 1400/1400.

## MEASURED constants — do not change blindly

These come from sweeps against this scene and the stock walking policy. They are
observations, not cosmetic tuning.

- **The backward gait does not engage continuously.** Commanding `vx > -0.30`
  makes the policy step in place (net displacement < 5 mm over 8 s). Backward
  walking engages from about `vx = -0.32` and stays upright through at least
  `-0.45`. `VX_RETREAT = -0.36` sits inside that band.
- **Heading must be closed-loop in every walking state.** Unopposed, the gait
  drifts hard: at `vx = -0.36` with `wz = 0` the robot ends 8 s at yaw ≈ −55°.
  With `YAW_KP = 2.0`, `WZ_MAX = 0.8`, the same command holds heading to ~4–6°.
- **The `wz` sign is normal** — positive `wz` yields a positive (left) yaw rate —
  when the policy is fed the real `imu_ang_vel` gyro. See the note below.
- **The turn converges, so it is tracked rather than cut early.** Commanding a
  +90° heading offset reaches +84° at 12 s and +87° at 20 s without winding up,
  so `TURN` exits on a tolerance (`TURN_TOL = 15°`) with `TURN_MAX = 8 s` as a
  safety fallback.
- `CMD_TAU = 0.08` — command low-pass. 0.25 is too slow to start the gait.
- `RETREAT_D = 1.15 m` with the person starting at 1.60 m gives ~1.7 s of margin.

### Read the gyro sensor by name, and check the name

`mujoco.mj_name2id` returns `-1` for an unknown sensor, and `model.sensor_adr[-1]`
is a **valid index** — the last sensor. So a mistyped sensor name does not raise:
it silently feeds a completely different quantity into the policy's
`base_ang_vel` slot.

An earlier version of this demo looked up `imu_gyro`, which does not exist in
these models, and therefore fed `root_angmom` (subtree angular momentum, kg·m²/s)
into a slot the policy expects to hold rad/s. The robot still walked and the
resulting video looked plausible, but every "measured" gait constant derived from
it described that corrupted observation rather than the robot. `runtime.py`
therefore resolves the sensor through `sensor_address()`, which raises on an
unknown name, and `tests/test_move_away_demo.py` locks the behavior in.

## Perception

Perception is a **ray cast** from the head camera to the person, not
segmentation. Segmentation is unusable here: the camera sits INSIDE the robot's
own jaw geometry, so every pixel/ray hits `jaw_soft` first and the person is
never visible. The ray cast skips the robot's own bodies and answers the real
question — is the person inside the field of view and unoccluded?

**Optical-frame correction.** The `head_camera` quaternion exported in the MJCF
is `[0 0 -1 0]`, which points the optical `-Z` axis backwards into the robot's
own CAD while the robot walks towards `+X`; rendering from it shows internal
geometry. The demo applies a −90° local-Z rotation to the in-memory `MjModel`
only, so optical `-Z` follows the head's forward axis and `+Y` follows world up.
This affects rendering and perception only, never physics, and the committed
MJCF is left untouched.

## Why the gaze layer is kinematic

The head is a large fraction of the robot's mass. Driving it physically while the
stock walking policy runs makes the robot fall — that policy was never trained to
compensate an externally imposed head trajectory. So the demo:

1. advances locomotion physics and policy inference in the primary `MjData`,
   unchanged;
2. poses head yaw/pitch in a separate `MjData` copy used only for perception and
   rendering;
3. never feeds that pose back into the locomotion dynamics.

This mirrors the gaze/locomotion split a real robot would use, and it makes
explicit that the demo claims no physical head-control stability.

## Limitations

- **The "person" is scripted, not perceived.** It is a mocap prop moving on a
  fixed straight path, and its position is read from simulator state. There is no
  detector, no tracker and no real sensing; the visibility test is geometric.
- **Gaze is kinematic only.** Head pose is applied in an isolated `MjData` copy.
  This demo makes **no claim** about physical head-control stability while
  walking, and the numbers above do not support one.
- **Sim only, CPU only.** Nothing here has been validated on hardware.
- The constants were measured against one exported walking policy in this scene;
  another policy will need its own sweep.

## Files

- `controller.py` — the state machine and velocity-command law. Pure Python, no
  MuJoCo/ONNX, fully unit-tested.
- `gaze.py` — camera-frame maths and the head visual servo.
- `perception.py` — field-of-view + ray-cast occlusion test.
- `runtime.py` — scene path, default pose, observation layout, safe sensor lookup.
- `run_demo.py` — CLI entry point, simulation loop and optional rendering.

Tests live in `tests/test_move_away_demo.py` and run on CPU without a policy.

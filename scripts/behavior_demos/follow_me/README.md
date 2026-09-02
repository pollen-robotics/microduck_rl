# follow-me — behavior demo

A CPU MuJoCo demo in which the Microduck follows a walking leader's **actual
path**, turning where the leader turned instead of cutting the corner.

This is a **behavior layer over a stock exported walking policy**. It does not
train, fine-tune, or replace a locomotion network: it only chooses the 3-DoF
twist command `(vx, vy, wz)` written into the policy's existing command slots.

> **Scope.** This is a simulation demo, not robot autonomy. The leader is a
> scripted kinematic mocap body and its pose is read directly from the
> simulator — there is no person detector, and this has not been validated on
> hardware. See [Limitations](#limitations).

## Run it

Requires an exported walking policy (the 61D observation layout, e.g.
`alpha_walking.onnx`). No policy is bundled; pass one explicitly:

```bash
# metrics only, no rendering
uv run python scripts/behavior_demos/follow_me/run_follow_me.py \
    --policy /path/to/alpha_walking.onnx --no-render

# with frames + metrics
uv run python scripts/behavior_demos/follow_me/run_follow_me.py \
    --policy /path/to/alpha_walking.onnx \
    --out /tmp/follow-me-frames --metrics /tmp/follow-me-metrics.json
```

Encode the frames with ffmpeg:

```bash
ffmpeg -framerate 50 -i /tmp/follow-me-frames/f%05d.png \
    -c:v libx264 -crf 18 -pix_fmt yuv420p -movflags +faststart follow-me.mp4
```

## Following footsteps, not poses

An earlier iteration selected the duck's command from the leader's **current**
pose. When the leader began turning, the duck turned immediately from its own
coordinates and cut across the inside of the corner instead of arriving at it.

This version follows a queue of world-space footsteps. At 50 Hz the behavior
records the leader's accumulated path in world coordinates, and the duck
pursues the interpolated point the leader walked `0.65 m` of **path length**
earlier:

```text
leader position → append world-space sample → subtract 0.65 m of path length
                                               ↓
                                  queued footprint + stored motion phase
                                               ↓
                                      stock walking policy command
```

Because the gap is a path length rather than a time delay, a corner stays in
the queue: the duck keeps walking straight and turns at the recorded corner,
a measured **5.44 s delayed turn** rather than an immediate mirrored one.

## True opposite turns

Two directional errors were corrected while building this demo:

1. the phase labelled `LEFT TURN` used negative world yaw, so it was physically
   a right-hand curve;
2. the later `RIGHT` phase was a diagonal strafe, not an opposite turn.

The route now uses conventional actor-relative directions. Facing `+X`, a left
turn increases world yaw by `+90°`; the later right turn decreases it by `−90°`
and returns the leader to its original heading. `test_turns_are_genuinely_opposite`
locks this in.

## Route

| Phase | Time | Description |
|---|---:|---|
| READY | `0–2 s` | both stand |
| FORWARD | `2–7 s` | straight approach |
| LEFT TURN | `7–15 s` | a genuine `+90°` left arc |
| STOP | `15–18 s` | both stop; the path queue freezes |
| RIGHT TURN | `18–35 s` | a genuine `−90°` right arc, then a straight exit |
| BACKWARD | `35–41 s` | both reverse |
| DONE | `41–44 s` | stable final stand |

Both turns occur at the same queued locations after the leader:

| Turn | Leader starts | Duck starts | Spatial delay | Leader yaw Δ | Duck yaw Δ |
|---|---:|---:|---:|---:|---:|
| Left | `7.00 s` | `12.44 s` | `5.44 s` | `+90.0°` | `+86.4°` |
| Right | `18.00 s` | `23.44 s` | `5.44 s` | `−90.0°` | `−84.0°` |

The signs are measured from the simulated trunk orientation, not inferred from
labels or camera projection.

## Controller detail

The stock walking policy has strongly **asymmetric turning authority** — a
single mirrored command does not produce mirrored body motion. The controller
therefore closes the loop on the delayed footprint yaw using separately
measured limits:

- positive-yaw / left correction: `wz = +0.60 … +1.00`
- negative-yaw / right correction: `wz = −0.18 … −0.32`
- deadband: `3°`
- forward command during turns: `vx = +0.24`

The policy has a sharp gait-onset threshold, so tiny continuous velocity
corrections are counterproductive; inside the deadband the command is exactly
zero. Actions use the shipped `0.9` action scale.

The angular-velocity sensor is resolved **by name** (`imu_ang_vel` and
aliases). Silently reading the last sensor by index picked up a different
quantity during development and produced unstable, falsely-good results, so it
is rejected explicitly.

**Reverse is a deliberate safety exception.** When the leader backs toward the
follower, the duck backs away immediately instead of waiting for the leader's
reverse footprint to arrive, so the leader cannot walk into it. This makes the
queued-footprint error grow during the final reverse — that is expected, and
`test_reverse_is_immediate_and_overrides_the_queue` pins the behavior.
Cornering and lateral motion still come exclusively from the recorded trail.

## Gaze and camera

Gaze and camera stabilization run entirely in a **separate rendering `MjData`**
copied from the physical state each step. They never feed back into the walking
dynamics — the physical locomotion state stays authoritative.

Person visibility in the picture-in-picture view is computed **geometrically**
(frustum test plus an occlusion ray against the known mocap pose), not from
image content.

## Measured validation

A 44 s / 2,200-step rollout with the stock walking policy:

- both leader turns have opposite signed `+90.0° / −90.0°` yaw changes;
- both duck turns have opposite signed `+86.4° / −84.0°` changes;
- both delayed turn starts occur `5.44 s` after the leader, at the queued
  footprint;
- all seven phases complete over `44.0 s` / `2,200` control steps;
- `fallen_steps = 0`, minimum trunk height `0.114 m`, final height `0.116 m`;
- leader range mean `0.857 m` (`0.519–1.363 m`);
- leader visible in the stabilized view for `2,200 / 2,200` steps.

`run_follow_me.py` writes all of these to the `--metrics` JSON so any run can be
re-checked.

## Tests

`tests/test_follow_me_demo.py` covers the choreography, the world-space trail,
the asymmetric controller, the reverse exception, and the scene/model
invariants. Everything runs on CPU and **needs no policy file, GPU, or
renderer**:

```bash
uv run --with pytest pytest tests/test_follow_me_demo.py
```

## Files

- `scene_follow_me.xml` — the official `robot_walk.xml` collision model plus the
  animated leader, footprint marker and stabilized camera rig.
- `follow_motion.py` — leader route, world-space trail, asymmetric turn
  controller. Free of MuJoCo/ONNX imports so it is unit-testable.
- `camera_tracking.py` — isolated gaze and geometric visibility measurement.
- `video_overlay.py` — HUD, picture-in-picture and timeline (rendering only).
- `run_follow_me.py` — ONNX rollout and metrics.

### Scene asset resolution

`scene_follow_me.xml` lives outside the robot directory, so it restates
`<compiler meshdir=...>` pointing back at the robot assets. That tag **must stay
after** the `<include>`: MuJoCo resolves `meshdir` against the top-level file
and applies the *last* `<compiler>` it parses. Placing it before the include
lets the robot's own `meshdir` win and the meshes fail to load.
`test_demo_scene_compiles_and_matches_the_reference_walk_model` compares the
compiled model against `scene_walk.xml` so a silent unit or asset change is
caught.

## Limitations

- **The leader is scripted.** It follows a fixed semantic choreography and its
  pose is read from the simulator. There is no person detector, no tracking
  from images, and no real perception of any kind.
- **Simulation only.** No hardware validation; the numbers above are from
  MuJoCo, not from a real robot.
- **Gaze is isolated render state.** It exists to make the picture-in-picture
  watchable and never influences locomotion.
- **The policy is unmodified.** All following behavior comes from command-slot
  steering of a stock walking policy, so its turning authority — including the
  left/right asymmetry compensated above — is a fixed constraint here.

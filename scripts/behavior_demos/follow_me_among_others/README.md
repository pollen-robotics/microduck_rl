# follow-me-among-others

A **CPU MuJoCo behavior demo**: the robot repeatedly searches a moving crowd
through its head camera, acquires a requested shirt color, follows that
person's queued footsteps, and stops.

```text
BLUE -> GREEN -> RED -> BLUE
SEARCH -> FOUND -> FOLLOW -> STOP   (once per selection)
```

This is **not** a trained policy, a new task, or robot autonomy. It drives the
*stock exported walking policy* with a twist command produced by a geometric
controller, so it needs no GPU, no training run and no new weights. Nothing in
`src/` changes except one added scene.

## Run it

Bring your own walking policy — there is no default weights path:

```bash
uv run --with imageio --with pillow \
    scripts/behavior_demos/follow_me_among_others/run_demo.py \
    --policy /path/to/walking.onnx --no-render --metrics /tmp/fmao.json
```

`--no-render` needs neither `imageio` nor `pillow`; both are only imported when
frames are actually written. To also render the HUD:

```bash
uv run --with imageio --with pillow \
    scripts/behavior_demos/follow_me_among_others/run_demo.py \
    --policy /path/to/walking.onnx --out /tmp/fmao-frames --fps 50
```

The process exits non-zero if any acceptance gate fails, so it works as a
regression check and not just a video generator.

Useful flags: `--targets BLUE,GREEN` (any order/length of BLUE/GREEN/RED),
`--seconds`, `--xml`, `--quiet`.

## What is real, and what is a proxy

Being precise about this matters more than the demo itself.

**Genuinely simulated.** The physics and the camera geometry. The robot walks
with the real policy at 50 Hz through the real BAM actuators. The head camera
sits where the model puts it, the search sweep really rotates the view, and
acquisition requires the target to be inside the camera frustum and within 8°
of the crosshair.

**A proxy.** Color recognition. The simulator supplies each actor's identity and
world pose, and `camera.py` tests known shirt/head sample points against the
frustum. There is **no RGB classifier**: swapping the identity lookup for
onboard vision is separate work. The picture-in-picture view is rendered for
inspection only — no pixel of it is fed back into any decision.

**Frustum-only visibility.** The test does not ray-cast for mutual occlusion,
so a person standing directly behind another still counts as visible. This is
the gate the published numbers were measured with; adding occlusion changes
acquisition timing and would require re-measuring the whole sequence.

## Layout

| File | Role |
|---|---|
| `crowd.py` | pedestrian routes, footstep queues, state machine, controller |
| `camera.py` | color-selective search sweep and tracking (needs `mujoco`) |
| `metrics.py` | acceptance gates and the metrics summary |
| `overlay.py` | HUD/PiP composition (needs `pillow`) |
| `run_demo.py` | rollout entry point |

`crowd.py` and `metrics.py` import no `mujoco`, which is why the state machine,
the color gate and every acceptance threshold are unit-tested on CPU with no
model, policy or renderer:

```bash
uv run --with pytest pytest tests/test_follow_me_among_others_*.py
```

The scene lives at
`src/mjlab_microduck/robot/microduck/scene_follow_me_among_others.xml` and
includes the official `robot_walk.xml`. The five pedestrians are mocap bodies
with `contype="0" conaffinity="0"` geoms: they are scenery, so they can never
push the robot, and following is not an artifact of being bumped.

## Behavior details

- **Five people, all moving.** Blue, green and red are selectable; yellow and
  purple are distractors. Each walks an independent elliptical lane with its own
  direction, speed, phase and unsynchronised wobble. Nobody freezes or teleports
  to make acquisition easier — a unit test asserts this over the full rollout.
- **Following means trailing, not mirroring.** During `FOLLOW` the robot walks
  toward the selected person's queued world-space footprint 0.55 m back along
  *that person's own path*, interpolated by arc length.
- **Zero locomotion outside FOLLOW.** In `SEARCH`, `FOUND`, `STOP` and `DONE`
  the command is exactly `(0, 0, 0)`, and the run reports the maximum.
- **Gait onset is a real threshold.** The controller holds `vx = 0.24` while
  turning. A smaller command produces a valid ONNX action but no visible
  locomotion — that is how the original RED hand-off failed, and the
  "did it actually move" gates exist to catch it.

## Measured validation

60 s, 3000 control steps at 50 Hz, CPU only, stock walking policy.

| Selection | Target | Search | Found | Follow | Stop | Search time |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Blue | `0.00 s` | `2.44 s` | `3.44 s` | `12.44 s` | `2.44 s` |
| 2 | Green | `13.94 s` | `16.50 s` | `17.50 s` | `26.50 s` | `2.56 s` |
| 3 | Red | `28.00 s` | `29.34 s` | `30.34 s` | `39.34 s` | `1.34 s` |
| 4 | Blue | `40.84 s` | `41.92 s` | `42.92 s` | `51.92 s` | `1.08 s` |

Acceptance gates, all enforced in `metrics.check_gates`:

- completed targets exactly `BLUE -> GREEN -> RED -> BLUE`, 4/4 cycles;
- every `FOUND` occurs while the requested color is visible;
- `wrong_color_locks = 0`;
- target visible for **100%** of control steps during `FOLLOW`;
- every follow segment travels ≥ `0.40 m`, displaces ≥ `0.30 m`, and measurably
  approaches its target;
- maximum stationary-state command `0.0`;
- `fallen_steps = 0`, minimum trunk height `0.1135 m` (above the `0.09 m`
  fallen threshold), final `0.1163 m` back at nominal.

Mean selected-person range during following is `0.929 m`. Queued-footprint RMSE
is `1.028 m` — this includes the deliberately long first Blue approach rather
than only the settled tail of each interval. In the Red segment the robot
travels `1.249 m`, displaces `0.718 m`, and reduces its queued-footprint error
from `0.397 m` to a minimum of `0.202 m`.

Because the demo is deterministic and headless, the whole 60 s rollout
reproduces in about two seconds on a laptop CPU.

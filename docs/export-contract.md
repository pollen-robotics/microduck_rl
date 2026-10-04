# The policy export contract

A microduck policy is one `.onnx` file produced by `run_export`, behind `uv run
scripts/export.py <TASK_ID>`. The robot's daemon
([`pollen-robotics/microduck`](https://github.com/pollen-robotics/microduck)) runs it against the
facts below, and reads the `microduck` metadata key to check them.

This repository owns these facts. The daemon's `docs/policy-manifest.md` owns `manifest.json`:
how the robot *drives* a policy (`kind`, `duration_s`, `command.encoding`). What the policy *was
trained against* travels inside the `.onnx`, because the weights and those facts cannot be
separated: a set holds ten files and one manifest, and a file copied out of its repo keeps its
metadata and loses its manifest.

## The graph

| | |
| --- | --- |
| input | `obs`, float32, `[1, 61]` |
| output | `actions`, float32, `[1, 14]` |
| opset | 18, exported with `dynamo=False` |
| batch | fixed at 1; no dynamic axes |
| normalizer | baked in: the graph is `actor(normalizer(obs))`, so it takes raw observations |

Every task trains a feed-forward MLP actor. A recurrent export (`h_in`/`c_in` beside `obs`) is the
daemon's `docs/recurrent-policies.md`, and needs `model_api: 2`.

## The observation

61 floats, in this order. Each is a term of the task's `actor` observation group, and the order is
the group's term order.

| index | width | term | contents |
| --- | --- | --- | --- |
| 0..3 | 3 | `base_ang_vel` | gyro, trunk frame, rad/s |
| 3..6 | 3 | `projected_gravity` | gravity in the IMU frame, unit vector |
| 6..20 | 14 | `joint_pos` | joint position minus `default_joint_pos`, rad |
| 20..34 | 14 | `joint_vel` | joint velocity, rad/s, one control step old |
| 34..48 | 14 | `actions` | the previous tick's raw network output |
| 48..51 | 3 | `command` | twist: `vx`, `vy` (m/s), `vyaw` (rad/s) |
| 51..55 | 4 | `head_command` | `neck_pitch`, `head_pitch`, `head_yaw`, `head_roll`, rad |
| 55..61 | 6 | `body_command` | body `x`, `y`, `z` (m), `roll`, `pitch`, `yaw` (rad) |

`joint_vel` is lagged one step in training because the Dynamixel firmware computes present
velocity over the previous sample window; the robot feeds the value it reads.

A skill trained on no command sees zeros in 48..61. A ground pick writes a phase and a sit↔stand
a posture flag into the twist slots; the manifest's `command.encoding` says which.

## The action

14 floats, one per joint in `joint_names` order. The joint target is

```text
target = default_joint_pos + action_scale × action
```

with no clipping. The mouth is not a policy joint: the robot's 15th servo is driven outside the
network.

## Joints

`joint_names`, the order of every 14-wide block above and of the action:

```text
left_hip_yaw  left_hip_roll  left_hip_pitch  left_knee  left_ankle
neck_pitch    head_pitch     head_yaw        head_roll
right_hip_yaw right_hip_roll right_hip_pitch right_knee right_ankle
```

`default_joint_pos` is `HOME_FRAME` in `robot/microduck_constants.py`: the pose `joint_pos` is
measured from and the action is added to.

## The actuator

Training simulates an XL330 with BAM's M6 model, which reproduces the servo's firmware position
loop. `kp_fw` is that loop's P gain **in Dynamixel register units** — the value the robot writes
to Position P Gain, with I and D at zero. The control loop runs at 50 Hz: a 5 ms physics step,
decimation 4.

## The `microduck` metadata key

One `metadata_props` entry, key `microduck`, value a JSON object:

```json
{
  "schema_version": 1,
  "obs_len": 61,
  "action_len": 14,
  "control_hz": 50,
  "joint_names": ["left_hip_yaw", "…", "right_ankle"],
  "default_joint_pos": [0.0, -0.0873, -0.4579, -0.0049, 0.453, "…"],
  "action_scale": 1.0,
  "kp_fw": 200,
  "observation": [
    ["base_ang_vel", 3], ["projected_gravity", 3], ["joint_pos", 14], ["joint_vel", 14],
    ["actions", 14], ["command", 3], ["head_command", 4], ["body_command", 6]
  ]
}
```

| field | read from | the robot |
| --- | --- | --- |
| `schema_version` | this document | nothing gates on it; a new field is additive |
| `obs_len`, `action_len` | the graph | refuses a mismatch |
| `control_hz` | `1 / (timestep × decimation)` | refuses a mismatch |
| `joint_names` | the entity's actuated joints | refuses a different order |
| `default_joint_pos` | the entity's default pose, full float precision | refuses a pose that differs beyond a small tolerance |
| `action_scale` | the `joint_pos` action term | uses it unless `[policy]` overrides |
| `kp_fw` | the BAM actuator config | uses it unless `[policy]` overrides |
| `observation` | the `actor` group's terms and widths | refuses a different layout |

Every value is read from the environment the checkpoint was trained in, never written by hand.
A field that is absent is not evidence: the robot falls back to its compiled values and logs it.
Only a field that is present and wrong refuses.

## Other metadata

mjlab's exporter also writes its generic keys — `joint_names`, `joint_stiffness`,
`joint_damping`, `default_joint_pos`, `action_scale`, `command_names`, `observation_names`,
`run_path` — as CSV at three decimals. They are not part of this contract. Under BAM the MuJoCo
actuators are torque motors, so `joint_stiffness` reads 1.0 and `joint_damping` is stale XML
data; neither is a gain.

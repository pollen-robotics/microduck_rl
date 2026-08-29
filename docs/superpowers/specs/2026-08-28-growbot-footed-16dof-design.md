# Growbot Footed 16-DOF Simulation Design

## Objective

Create a physics-first MuJoCo/mjlab variant of Microduck for a 25 cm Growbot
prototype. The robot uses the proven footed Microduck lower body, a lightweight
custom upper body without a decorative chest plate, the existing egg-shaped
head dome, and one actuated elbow per arm. The simulator must determine whether
lower-cost servos can balance, move, fall safely, self-right, and perform the
forward roulade before hardware is purchased.

## Confirmed Physical Configuration

- Overall height target: approximately 0.25 m in the standing home pose.
- Lower body: stock footed Microduck hip, knee, ankle, and sole geometry.
- Wheels and rollerblade geometry: excluded.
- Torso: structural Microduck trunk frame with no decorative chest plate.
- Head: existing egg-shaped dome, modeled as a very thin reinforced shell.
- Head print intent: 0.9 mm nominal PETG-CF wall, 1.35 mm along the forward-roll
  contact band, and local internal ribs around the visor and mounts.
- Arms: fixed shoulder attachment, one pitch hinge at each elbow, simplified
  collision-enabled upper arm, forearm, and three-finger hand geometry.
- Actuation: 10 leg servos, 4 neck/head servos, and 2 elbow servos; 16 total.
- Feet and hands may contact the floor and participate in learned recovery.

## Asset Architecture

The Growbot model is a new MJCF asset rather than a mutation of
`robot_allcollisions.xml`. It reuses the existing footed lower-body hierarchy,
joint names, actuator directions, sensors, and collision conventions. New arm
bodies and elbow hinges attach beneath `trunk_base`. Visual meshes may be
exported from `/home/wes/robot.blend`, but simulation collisions use simple
capsules and rounded boxes so contact is stable and inexpensive.

The head keeps the existing smooth collision envelope because head-ground
contact is load-bearing in the roulade reward. Its mass and inertia are changed
to represent the thin dome, visor, display, camera, and internal electronics;
visual wall thickness is not used as collision thickness.

The new asset and task use `growbot` names and do not replace or silently alter
the stock Microduck assets. Existing 14-servo checkpoints remain loadable in
their original tasks.

## Joint and Policy Contract

The original 14 servo order remains unchanged:

1. left hip yaw, roll, pitch, knee, ankle
2. neck pitch, head pitch, head yaw, head roll
3. right hip yaw, roll, pitch, knee, ankle

The elbow joints are appended as indices 14 and 15:

14. left elbow pitch
15. right elbow pitch

The actor observation grows from 61D to 67D because joint position, joint
velocity, and previous action each gain two elbow values. The 13D command block
retains its existing layout. The Growbot policy is therefore a deliberately
separate policy family and cannot be hot-swapped into a stock Microduck 61D
runtime slot. Its ONNX metadata must declare 16 actions and 67 observations.

## Configurable Actuator Model

Servo selection is an output of simulation, not a hard-coded purchasing
assumption. The Growbot configuration exposes per-joint-class parameters for:

- servo mass;
- stall torque and continuous torque proxy;
- no-load speed;
- command delay;
- position-loop stiffness and damping;
- friction and backlash;
- supply voltage and voltage sag.

At minimum, the feasibility sweep evaluates:

- an SC09-class low-cost configuration on all joints;
- a hybrid configuration with stronger leg servos and SC09-class head/elbow
  servos;
- the existing Microduck BAM actuator envelope as a reference ceiling.

The sweep records configurations by physical parameters, not vendor names, so
new candidates can be evaluated without editing the task.

## Mass and Inertia Model

Initial total-mass exploration spans 0.50-0.80 kg. Domain randomization is
centered on the measured or estimated component masses rather than scaling the
entire robot uniformly. Separate ranges cover the trunk electronics, battery,
head electronics, thin dome, each arm, and each servo class.

The battery and dense electronics are positioned as low and close to the trunk
center as packaging permits. The head center-of-mass range remains independently
randomized because it dominates balance and roll behavior. Final training must
wait until weighed printed parts replace estimates.

## Task and Reward Adaptation

The first task is `Mjlab-Growbot-Roulade-Flat`, derived from the existing
footed Roulade environment. It keeps the state-based over-the-head progress
latch, head-ground pivot signal, upright landing composite, impact penalty,
NaN guard, and reverse-curriculum spawn strategy.

Arm additions are limited to:

- elbow actions and observations;
- joint limits and neutral elbow pose;
- arm/hand self-collision and terrain contact;
- a small elbow-limit penalty;
- existing action-rate and torque-rate smoothing applied to all 16 actions.

There is no positive reward for hand contact. This prevents the policy from
parking on its arms, while still allowing it to discover bracing and pushing
during recovery.

## Feasibility Workflow

Long training is gated by progressively more expensive checks:

1. Compile the MJCF and resolve exactly 16 actuated joints with no unintended
   passive-joint matches.
2. Run CPU configuration tests for joint order, action and observation shapes,
   collision names, reward signs, and ONNX metadata.
3. Hold the standing home pose for 3 seconds across the mass/torque sweep and
   reject configurations that fall, exceed limits, or saturate continuously.
4. Run the mandatory 64-environment, 5-iteration smoke test and require finite
   state, reward, loss, and action statistics.
5. Run short discovery trials at modest environment counts for the surviving
   actuator configurations.
6. Render evaluation videos from identical spawn batteries and compare success,
   impact, action acceleration, servo saturation, and recovery time.
7. Start a full run only after one configuration passes the physics gates and
   visibly performs the intended maneuver.

## Verification and Acceptance Criteria

The simulation is ready for actuator comparison when:

- the stock Microduck test suite remains green;
- the Growbot asset compiles with 16 and only 16 actuated joints;
- actor observations are 67D and actions are 16D in training and ONNX export;
- arm, hand, foot, head, and terrain contacts resolve without NaNs;
- the home-pose settle battery reports tilt as well as height;
- all penalty reward contributions are non-positive;
- a 5-iteration smoke train completes and exports a loadable ONNX policy;
- a recorded rollout clearly shows the custom footed 16-DOF model rather than
  the stock Microduck or rollerblade asset.

The actuator design is considered physically promising only when it completes
standing, recovery, and roulade evaluation without persistent torque saturation.
Passing in simulation does not authorize hardware purchase until printed-part
and electronics masses are entered into the model.

## Out of Scope

- Powered or passive rollerblade locomotion.
- Motorized shoulders, wrists, or fingers.
- Final printable CAD reinforcement and fastener design.
- ESP32 firmware and the server intelligence protocol.
- Replacing stock Microduck policy contracts or checkpoints.

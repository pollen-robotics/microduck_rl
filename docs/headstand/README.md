# Microduck headstand routine

This project adds six training tasks for an acrobatic headstand routine on Microduck.

The duck folds forward from standing, kicks up to a headstand with its legs together, holds, and rolls back onto its feet. It then enters a second headstand with a split leg entrance, switches which leg leads while inverted, and continues over in the split to return to standing.

The tasks, rewards, handover data, and tests are self-contained in this repository; training and testing them requires no companion checkout. The verified evaluation completed the routine in 87 of 96 simulated attempts. The optional [companion repository](https://github.com/zachgarner/microduck-headstand/tree/a5dad724d124fe50eee9f78942361bb794950ff2) holds videos, evaluation scripts, raw logs, and training jobs. All results so far are from simulation; the routine has not been tested on a physical robot.

## How the routine works

Each movement has its own policy. The fold brings the duck into a pike, with its head and both feet on the floor. From there, a kick-up policy lifts the legs into a headstand and balances. Separate policies handle the leg switch and the two exits.

The routine uses these six policies alongside Pollen's existing standing policy, `alpha_stand.onnx`, from `pollen-robotics/microduck-policies`. The standing policy steadies the duck between the first roll and the next fold, and after the final exit.

The next policy takes over once the duck has held the required position. The intended sequence holds the pike for 0.3 seconds before starting a kick-up, each headstand for 2 seconds, and the opposite split for 1 second before the final exit. The standing policy then holds for 1 second.

| Movement | Starting position | Target | Task |
| --- | --- | --- | --- |
| Fold | Standing | Pike, with the trunk about 76° from upright | `Mjlab-HeadstandFold-Flat-MicroDuck` |
| Kick-up, legs together | Pike | Headstand with the legs together | `Mjlab-HeadstandKickupLegsTogether-Flat-MicroDuck` |
| Kick-up, split | Pike | Headstand with the left leg forward | `Mjlab-HeadstandKickup-Flat-MicroDuck` |
| Split switch | Split headstand | The opposite leg leading | `Mjlab-HeadstandSplitSwitch-Flat-MicroDuck` |
| Back roll | Headstand with the legs together | Standing | `Mjlab-HeadstandBackrollLegsTogether-Flat-MicroDuck` |
| Split over | Either split headstand | Standing, with the legs split and straight until the lead foot lands | `Mjlab-HeadstandSplitOver-Flat-MicroDuck` |

The policies share the repository's 61-value observation layout and control 14 joints. The split-switch policy also reads a command flag: `0` requests the original split and `1` requests its mirror. The flag uses the forward-velocity command slot, following the sit-stand task's convention.

## Training the movements

The fold learns to place the head on the floor and settle into the pike. The kick-up learns to lift the legs and balance from there. Separating the two gives each policy a specific movement to learn.

The kick-up tasks train on 512 states recorded from the fold policy, including the robot's position and velocity. These handovers let the kick-up practice from the positions the preceding policy actually produces.

The recordings are stored in `src/mjlab_microduck/tasks/handover_pike_from_fold.npz`, along with the source task, checkpoint path, and robot XML hash. The loader checks the hash to ensure the recordings match the current robot XML.

Some episodes start partway through the kick-up or already in the headstand, giving the policy practice balancing before it can reliably complete the entrance. Others start in a reference pike established by a three-second settling test. Training gradually shifts toward pike starts.

With the default handover weight, the starting positions are distributed as follows. Percentages are rounded.

| Starting position | Iteration 0 | Iteration 3,000 |
| --- | ---: | ---: |
| Recorded fold handover | 38% | 43% |
| Partway through the kick-up | 23% | 17% |
| Already in the headstand | 23% | 13% |
| Reference pike | 15% | 28% |

The simulation uses the `allcollisions` robot model, including floor contact for the thighs, shins, and trunk. Contact sensors distinguish head support, foot support, and contact elsewhere on the body.

## Rewards

The kick-up rewards both reaching the headstand and holding it. During the entrance, the policy earns a progress reward when the trunk moves closer to upside down than it has previously reached in that episode. Repeating an earlier part of the movement earns no additional progress. This reward requires floor contact and favors forward rotation with little sideways tilt.

Once inverted, the duck earns a balance reward based on its height, orientation, and joint pose. The head must touch the floor while the feet and other monitored body parts remain clear. The target pose specifies whether the legs should be together or split.

A recorded head force above 12 N disables the balance reward for the rest of the episode. This discourages a forceful landing followed by a long hold. The first 0.3 seconds are excluded to allow the starting position to settle.

The fold uses the same approach with the pike as its target. It rewards progress toward the target trunk angle, then rewards holding the position with the head and both feet on the floor. The same 12 N sampled-force threshold applies to its hold reward.

A kick-up episode ends if another monitored body part touches the floor after a 0.5-second settling period. Earlier versions kept failed attempts running and applied a penalty at every step. The policy learned to avoid that penalty by staying in the pike. The current approach ends the failed attempt, while a separate penalty discourages waiting without completing the kick-up.

Rotation above `omega_max` incurs a speed penalty. Separate penalties discourage abrupt changes in actions and joint torque, and continued rotation near the headstand. These smoothness penalties increase during training on schedules controlled by `polish_at`.

The split switch moves the target pose gradually toward the opposite split over `switch_ramp_s` seconds. The split-over exit rewards roll progress with the legs split and straight during the configured rotation window, and penalizes closing the split or bending the knees in that window.

## Simulation results

The routine completed **87 of 96 attempts** across three random seeds, with 32 attempts per seed. Successful routines took a median of **12.3 seconds**. Each of the six trained policies also passed 32 of 32 attempts when evaluated separately at seed 0.

These measurements were made on September 22, 2026, using the corrected evaluator. It disables automatic resets, checks leg shape during the headstand holds, and requires the duck to finish standing on both feet after completing every stage. The companion reports and raw logs in `results/verified/` include seeds, checkpoint hashes, software versions, and per-attempt results.

| Evaluation | Checkpoint iteration | Success | Median time to goal | Sampled head force, median / maximum |
| --- | ---: | ---: | --- | --- |
| Fold | 1,000 | 32/32 | 0.40 s | 11.3 N / 34.8 N |
| Kick-up, legs together | 1,499 | 32/32 | 0.40 s | 9.0 N / 17.7 N |
| Kick-up, split | 1,499 | 32/32 | 0.42 s | 9.1 N / 17.9 N |
| Split switch | 500 | 32/32 | 0.32 s after the command | — |
| Back roll | 1,999 | 32/32 | 0.52 s | — |
| Split over | 1,499 | 32/32 | 0.54 s | — |
| Complete routine, seeds 0 / 1 / 2 | — | 28/32, 29/32, 30/32 | 12.3 s including holds | — |

Individual policy times measure the first qualifying goal state among attempts that also succeeded at the end. The routine time includes every required hold. The kick-ups start from recorded fold handovers; the split-over check uses a mixture of the original and mirrored split starts.

The nine incomplete routines stopped at four stages: three at the legs-together kick-up, four at the back roll, one at the split switch, and one during the final standing hold. One of these attempts was upright at the end but had not completed every hold, so it was counted as a failure. Success in isolation does not establish that a policy can handle every position and velocity left by the preceding movement.

The evaluator uses these criteria:

| Position | Criterion |
| --- | --- |
| Pike | Head and both feet touching, no other monitored body contact, nose down, and trunk 60–95° from upright. |
| Headstand, legs together | Trunk within 35° of inverted, head-only support, both knees within 0.3 rad of straight, and hip separation at most 0.2 rad. |
| Split headstand | The same support, orientation, and knee requirements, with hip separation at least 1.0 rad. |
| Opposite split | The split-headstand criterion, with the joints closer to the mirrored target than the original target. |
| Standing | Trunk within 30° of upright, both feet touching, and no head or other monitored body contact. |

Hip separation is `abs(left_hip_pitch + right_hip_pitch)` under the model's joint sign convention. The routine must complete every stage and meet the standing criterion at the final frame of its eighteen-second rollout. The individual exit counts assess final standing; they do not certify leg shape throughout the exit trajectory.

Head forces are the largest values sampled at the policy's 50 Hz control rate, including initial settling. Physics runs at 200 Hz, so shorter impacts can go unrecorded. The 12 N training threshold uses the same sampling but excludes the first 0.3 seconds. It does not bound every physical impact. The simulated duck weighs 7.2 N. A dash indicates a measurement that was not collected; the exit tasks' head sensors report contact only.

All 512 historical handover states are retained because they are the training data used by the evaluated checkpoints. A separate CPU contact audit found head-and-both-feet contact in 511 of its 512 states, and head-and-one-foot contact in one. The current collector requires both feet. The audit checks saved poses rather than replaying the original Warp contact history.

The kick-up and split-switch checkpoints continued training from earlier policies with `polish_at=0`, so their smoothness penalties were fully active from the start of those runs. Checkpoint numbers count iterations in the final run; the companion repository records the earlier training.

The six evaluated checkpoints and normalized ONNX exports are public on Hugging Face under `ZachGarner`. Replaying these checkpoint evaluations is optional and uses the companion repository's setup instructions. Its downloader retrieves pinned revisions of the original checkpoints and Pollen's standing policy, checks their hashes, and requires no login. From that checkout's `microduck_rl/` directory, run:

```bash
uv run ../tools/download_policies.py
uv run ../tools/run_verification.py --routine-seeds 0 1 2 --individual --workers 2
uv run ../tools/summarize_verification.py
```

## Configuration reference

Task variants are registered in `src/mjlab_microduck/tasks/__init__.py`. The headstand factory accepts `style="fold"`, `"split"`, or `"legs_together"`; adding `switch=True` to the split style enables the switch command. The back-roll factory accepts `style="legs_together"` or `"splitover"`.

These are the current factory defaults. Historical runs used different settings in some cases: the split-switch training record lists a handover weight of `0.3`, compared with the current default of `0.5`. Configure new variants through the factory arguments; the former `HEADSTAND_*` environment variables no longer set these values. The optional companion training wrapper and examples in `anyscale/README.md` pass these arguments explicitly.

**`make_microduck_headstand_env_cfg`**

| Argument | Default | Purpose |
| --- | --- | --- |
| `park_cost_w` | `-0.5` | Initial penalty weight for remaining short of inversion. Increases in magnitude during training. |
| `progress_w` | `10` | Weight of the kick-up progress reward. |
| `omega_max` | `2.0` | Rotation rate in rad/s above which the speed penalty applies. |
| `overspeed_w` | `-0.5` | Weight of that speed penalty. |
| `handover_prob` | `0.5` | Relative weight of recorded handovers in the starting-state mix, equivalent to about 38% of initial episodes with the other defaults. |
| `polish_at` | `2500` | Shifts the smoothness schedules. Their final stages occur up to 1,000 iterations later. Set to `0` for full weights from the start. |
| `switch_ramp_s` | `0.3` | Time in seconds for the switch target to move to the opposite split. |

**`make_microduck_backroll_env_cfg`**, when `style="splitover"`

| Argument | Default | Purpose |
| --- | --- | --- |
| `progress_w` | `8` | Weight of the roll-progress reward. |
| `rate_cap` | `5` | Maximum rotation rate credited by the progress reward, in rad/s. |
| `tuck_w` | `-2` | Penalty weight for losing the straight, split leg position during the exit window. |
| `window_deg` | `(170, 330)` | Accumulated roll angles over which the split-leg requirement applies. |
| `omega_max` | `7` | Rotation rate in rad/s above which the speed penalty applies. |
| `overspeed_w` | `-0.1` | Weight of that speed penalty. |

## Code and tests

Paths below are relative to the repository root.

| File | Contents |
| --- | --- |
| `src/mjlab_microduck/tasks/microduck_headstand_env_cfg.py` | Fold, kick-up, and split-switch configurations. |
| `src/mjlab_microduck/tasks/microduck_backroll_env_cfg.py` | Back-roll and split-over configurations, based on the existing roulade task. |
| `src/mjlab_microduck/tasks/mdp.py` | Reward functions, starting-state logic, and the split-switch command. |
| `scripts/headstand/collect_handover.py` | Collection of states for training the next policy. |
| `tests/test_headstand_cfg.py` | CPU tests for task configuration, rewards, contacts, starting states, observation layout, and pose stability. |

Run the headstand tests from the repository root:

```bash
uv run --with pytest pytest tests/test_headstand_cfg.py
```

# A78 iteration 1000 race leader

Checkpoint 1000 is the first A78 policy to beat the retained parent on the primary valid-roll-linked 10 m race objective while keeping all four robots inside the shared road.

## Identity

| Field | Value |
|---|---|
| Policy ID | `a78-iter1000-76e2799e3e3d` |
| Checkpoint | `artifacts/training/roll-sprint-saved-policies/a78-iter1000-76e2799e3e3d/model_1000.pt` |
| SHA-256 | `76e2799e3e3d39e562fc2b95f1c72cc8bf7abfd6095fc0284eb146be148fd71c` |
| Source run | `2026-08-31_21-00-47_a78_rearm2_champion_8192x4000_seed57` |
| Policy code commit | `4599dfd153a1636ed497e7c66cf84d0b870c7c58` |
| Evaluation | `artifacts/training/roll-sprint-saved-policies/a78-iter1000-76e2799e3e3d/evaluation-v8.json` |
| 15-second video | `artifacts/training/roll-sprint-saved-policies/a78-iter1000-76e2799e3e3d/race-preview-15s.mp4` |

## Verified race improvement

| Metric | Checkpoint 300 | Checkpoint 1000 |
|---|---:|---:|
| Valid 10 m finishers | 0 / 4 | 3 / 4 |
| Mean credited frontier | 3.666 m | 10.373 m |
| Best credited frontier | 8.774 m | 11.239 m |
| Valid rolls | 24 | 69 |
| Recovered rerolls | 21 | 65 |
| Shared-road exits | 3 robots | 0 robots |
| NaN / out-of-bounds | 0 / 0 | 0 / 0 |

All four checkpoint-1000 robots traveled about 18.8 to 19.4 m in raw projected distance during the 40-second evaluation. Three extended their valid roll-linked frontier beyond 10 m. Robot 4 reached 9.464 m of credited frontier, so the four-robot gate remains open.

## Remaining failure modes

- No robot ended launch-ready and standing on the road.
- Robot 4 missed the 10 m valid-frontier threshold by 0.536 m.
- The deterministic recovery battery regressed from 16 of 16 at checkpoint 300 to 9 of 16 at checkpoint 1000.
- Only 4 of the 9 recovered battery cases completed a subsequent reroll.

Checkpoint 1000 is therefore preserved as the **race leader**, not the formal all-gates champion. Training continues from the existing A78 run toward 4 of 4 finishers while the user-selected checkpoint-300 video remains preserved separately.

## Video settings

The dedicated race-leader video is 15.033 seconds, 1280 by 720, 60 fps, with no overlay label. Future five-minute sampler videos use 1500 simulation steps, frame stride 3, 60 fps output, and 2x playback, producing the same 15-second presentation.

# Behavior-brain training

`scripts/train_brain.py` trains the small neural affect selector consumed by MicroDuck 2.0's
`robotd/src/brain.rs`. Its reviewed JSONL input contains exactly `stimulus` and `affect`, using
the labels documented in the script. It learns a 5→hidden→5 ReLU network and emits schema-1 JSON.

```sh
python scripts/train_brain.py reviewed-feedback.jsonl --out brain.json
```

This is offline supervised feedback learning, not online physical trial-and-error. Do trial and
error only in the simulator/PPO workflow; a generated model selects an affect only and cannot
command a motor, enable a skill, or replace fall/human-control safety rules.

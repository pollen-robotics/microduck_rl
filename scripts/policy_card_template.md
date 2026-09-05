---
library_name: onnx
pipeline_tag: reinforcement-learning
tags: [{{TAGS}}]
license: apache-2.0
---

# {{NAME}}

{{DESCRIPTION}}

{{VIDEO}}

**Command** — twist slots of the 13-D command block (head / body slots: {{HEAD}}, {{BODY}}); idle = `{{IDLE}}`

| slot | meaning |
|---|---|
{{TWIST_ROWS}}

{{EXTRA}}

**Contract** `obs[1,61] f32 → actions[1,14] f32`, normalizer baked in, 50 Hz, targets around HOME × {{ACTION_SCALE}}. Kind {{KIND}}, entry pose {{ENTRY_POSE}}.
**Provenance** `{{TASK_ID}}` — {{TRAIN_REPO}} @ `{{COMMIT}}`, run `{{RUN}}`; exported with `scripts/export.py`.
**Files** `policy.onnx` · `manifest.json` (all of the above, machine-readable){{MEDIA_LINE}}{{EXTRA_FILES}}
Format: [Microduck policy sharing](https://github.com/pollen-robotics/microduck_rl/blob/main/docs/sharing-policies.md).

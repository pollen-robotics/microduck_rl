# Optional Studio extensions

Keep the fork's visualization, simulation adapters and lateral-policy tools in
this directory. `body_server.py` owns the authoritative world, protocol and
world lock; it delegates optional behavior here through explicit hooks.

| Module | Responsibility |
|---|---|
| `service.py` | Render CLI options, request dispatch and stream lifecycle |
| `stream.py` | Read-only MuJoCo snapshots, camera controls and cached JPEG/PNG frames |
| `bam.py` | NumPy motor, friction, voltage/current limits and command delay |
| `physics.py` | BAM scene/solver setup and body/world lifecycle adapters |
| `lateral/config.py` | Public side-step task configuration entry point |
| `lateral/register.py` | Register the extra task without editing published task recipes |
| `lateral/finetune.py` | ONNX actor warm start, local PPO and source snapshots |
| `lateral/validate.py` | Pure lateral trajectory and physical-stop acceptance |
| `scenes/` | Extension scene XML and resource resolution |
| `docs/` | Lateral recipe and evaluation limitations |
| `tests/` | Regression tests for these extensions |

Rendering stays read-only. BAM is a separate, explicit physics choice through
`--actuator-model bam`; XML remains the default. Importing BAM or the body
server does not import Torch, mjlab or the lateral training tools. Expensive
rendering/encoding happens outside the world lock. Physics adapters run inside
the caller's existing lock and must not acquire it again.

Use `mjlab_microduck.sim.visualization.bam` for motor imports. The old fork-only
files (`sim/bam.py`, `tasks/microduck_lateral_env_cfg.py`,
`scripts/finetune_lateral.py`, `scripts/validate_lateral.py`) have been removed;
there are no compatibility wrappers outside this directory. Run the tools as
modules:

```sh
uv run python -m mjlab_microduck.sim.visualization.lateral.finetune --help
uv run python -m mjlab_microduck.sim.visualization.lateral.validate --help
uv run pytest
```

`scene_velstand.xml` lives in `scenes/` and references the upstream robot XML
and meshes in their original location. Studio resolves scene names from both
the upstream and extension directories. Original upstream files retain only
the integration hooks. Default pytest discovery includes both the upstream
`tests/` and this directory's `tests/`. The full recipe is in
[docs/lateral_policy.md](docs/lateral_policy.md).

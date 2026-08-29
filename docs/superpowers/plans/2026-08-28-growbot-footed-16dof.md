# Growbot Footed 16-DOF Simulation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a runnable, testable 25 cm footed Growbot Roulade simulation with two actuated, collision-enabled elbows and a separate 16-action/67-observation policy contract.

**Architecture:** Build the Growbot MJCF in Python by loading the proven footed Microduck `MjSpec` and attaching lightweight capsule/box arm bodies to `trunk_base`. A dedicated environment factory derives the existing Roulade task, swaps in the Growbot entity, initializes elbows, and registers a separate task and experiment without altering stock Microduck behavior.

**Tech Stack:** Python 3.12, MuJoCo `MjSpec`, mjlab, RSL-RL, pytest, MuJoCo Warp

**Spec:** `docs/superpowers/specs/2026-08-28-growbot-footed-16dof-design.md`

## Global Constraints

- Standing height remains approximately 0.25 m.
- Use flat feet; include no wheel or rollerblade joints or geoms.
- Keep stock Microduck joint indices 0-13 unchanged and append elbow joints at 14-15.
- Use 16 actions, 67D actor observations, and 80D critic observations.
- Preserve all existing stock Microduck tasks, assets, and checkpoint contracts.
- Shoulders are fixed; only left and right elbow pitch are added.
- Arms and hands participate in terrain and self collision but receive no positive contact reward.
- Run a 64-environment, 5-iteration smoke test before any long training.

---

### Task 1: Build and validate the 16-joint Growbot asset

**Files:**
- Create: `src/mjlab_microduck/robot/growbot_constants.py`
- Create: `tests/test_growbot_asset.py`

**Interfaces:**
- Produces: `get_growbot_spec() -> mujoco.MjSpec`
- Produces: `GROWBOT_ROBOT_CFG: EntityCfg`
- Produces: `GROWBOT_HOME_FRAME: EntityCfg.InitialStateCfg`
- Consumes: `MICRODUCK_ALLCOLLISIONS_XML` and the canonical BAM actuator configuration.

- [ ] **Step 1: Write the failing asset-contract tests**

```python
def test_growbot_joint_order_and_count():
    model = get_growbot_spec().compile()
    names = articulated_joint_names(model)
    assert names[:14] == EXPECTED_MICRODUCK_JOINTS
    assert names[14:] == ["left_elbow_pitch", "right_elbow_pitch"]

def test_growbot_is_footed_and_has_arm_collisions():
    model = get_growbot_spec().compile()
    assert model.njnt == 17  # free joint + 16 hinges
    assert all("wheel" not in name for name in joint_names(model))
    for name in ("left_hand_collision", "right_hand_collision"):
        assert model.geom(name).id >= 0
```

- [ ] **Step 2: Run the tests and verify import failure**

Run: `uv run --with pytest pytest tests/test_growbot_asset.py -q`

Expected: FAIL because `growbot_constants` does not exist.

- [ ] **Step 3: Implement the programmatic `MjSpec` builder**

Load `robot_allcollisions.xml`, lower the explicit trunk/head mass while scaling
their inertia tensors by the same ratio, and add symmetric arms:

```python
def _add_arm(spec: mujoco.MjSpec, side: str, y: float) -> None:
    trunk = spec.body("trunk_base")
    upper = trunk.add_body(name=f"{side}_upper_arm", pos=(0.0, y, 0.018))
    upper.add_geom(
        name=f"{side}_upper_arm_collision",
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        fromto=(0, 0, 0, 0, 0, -0.040),
        size=(0.010,), mass=0.018, group=3,
    )
    forearm = upper.add_body(name=f"{side}_forearm", pos=(0, 0, -0.040))
    forearm.add_joint(
        name=f"{side}_elbow_pitch", type=mujoco.mjtJoint.mjJNT_HINGE,
        axis=(0, 1, 0), limited=True, range=(0.0, 2.35),
        damping=0.041, frictionloss=0.0048, armature=0.0018,
    )
    forearm.add_geom(
        name=f"{side}_forearm_collision",
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        fromto=(0, 0, 0, 0, 0, -0.038),
        size=(0.009,), mass=0.016, group=3,
    )
    forearm.add_geom(
        name=f"{side}_hand_collision",
        type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
        pos=(0, 0, -0.043), size=(0.013, 0.009, 0.016),
        mass=0.010, group=3,
    )
    spec.add_actuator(
        default=spec.find_default("chosen_actuator"),
        name=f"{side}_elbow_pitch", target=f"{side}_elbow_pitch",
    )
```

Create a Home frame that prepends the exact elbow rules before the catch-all
velocity rule, and create an `EntityCfg` using the same BAM actuator and full
collision conventions as `MICRODUCK_STANDUP_ROBOT_CFG`.

- [ ] **Step 4: Run the focused asset tests**

Run: `uv run --with pytest pytest tests/test_growbot_asset.py -q`

Expected: PASS; compiled model has one free joint, 16 hinges, and no wheels.

- [ ] **Step 5: Commit the asset**

```bash
git add src/mjlab_microduck/robot/growbot_constants.py tests/test_growbot_asset.py
git commit -m "feat: add 16-joint Growbot asset"
```

### Task 2: Add the dedicated Growbot Roulade environment

**Files:**
- Create: `src/mjlab_microduck/tasks/growbot_roulade_env_cfg.py`
- Create: `tests/test_growbot_roulade_cfg.py`
- Modify: `src/mjlab_microduck/tasks/__init__.py`

**Interfaces:**
- Consumes: `GROWBOT_ROBOT_CFG`, `make_microduck_roulade_env_cfg()`.
- Produces: `make_growbot_roulade_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg`.
- Produces: `GrowbotRouladeRlCfg: RslRlOnPolicyRunnerCfg`.
- Registers: `Mjlab-Growbot-Roulade-Flat`.

- [ ] **Step 1: Write failing environment tests**

```python
def test_growbot_task_contract():
    cfg = make_growbot_roulade_env_cfg()
    spec = cfg.scene.entities["robot"].spec_fn().compile()
    assert spec.nu == 16
    assert cfg.actions["joint_pos"].scale == 1.0
    assert cfg.scene.terrain.terrain_type == "plane"

def test_growbot_task_registration():
    import mjlab_microduck.tasks  # noqa: F401
    assert "Mjlab-Growbot-Roulade-Flat" in list_tasks()
```

- [ ] **Step 2: Run the focused test and verify failure**

Run: `uv run --with pytest pytest tests/test_growbot_roulade_cfg.py -q`

Expected: FAIL because the factory and registration do not exist.

- [ ] **Step 3: Implement the derived environment**

```python
def make_growbot_roulade_env_cfg(play: bool = False):
    cfg = make_microduck_roulade_env_cfg(play=play)
    cfg.scene.entities = {"robot": GROWBOT_ROBOT_CFG}
    return cfg

GrowbotRouladeRlCfg = deepcopy(MicroduckRouladeRlCfg)
GrowbotRouladeRlCfg.experiment_name = "growbot_roulade"
GrowbotRouladeRlCfg.run_name = "growbot_footed_16dof"
```

Register the task with `MicroduckOnPolicyRunner`. Do not add it to the stock
backlash table or change any Microduck task ID.

- [ ] **Step 4: Verify configuration and registration**

Run: `uv run --with pytest pytest tests/test_growbot_roulade_cfg.py tests/test_growbot_asset.py -q`

Expected: PASS.

- [ ] **Step 5: Commit the environment**

```bash
git add src/mjlab_microduck/tasks/growbot_roulade_env_cfg.py src/mjlab_microduck/tasks/__init__.py tests/test_growbot_roulade_cfg.py
git commit -m "feat: register Growbot Roulade environment"
```

### Task 3: Lock the separate observation/action contract

**Files:**
- Modify: `tests/test_growbot_roulade_cfg.py`
- Modify: `src/mjlab_microduck/tasks/growbot_roulade_env_cfg.py`

**Interfaces:**
- Consumes: mjlab observation and action managers instantiated from the Growbot cfg.
- Produces: verified actor shape 67, critic shape 80, action shape 16.

- [ ] **Step 1: Add a manager-level shape test**

```python
def test_growbot_dimensions_compile_on_cpu():
    cfg = make_growbot_roulade_env_cfg(play=True)
    cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
        assert env.observation_manager.group_obs_dim["actor"] == (67,)
        assert env.observation_manager.group_obs_dim["critic"] == (80,)
        assert env.action_manager.total_action_dim == 16
    finally:
        env.close()
```

- [ ] **Step 2: Run the test and capture any selector or shape mismatch**

Run: `uv run --with pytest pytest tests/test_growbot_roulade_cfg.py -q`

Expected: either PASS automatically through regex joint selection or a focused
failure naming the selector that excludes elbows.

- [ ] **Step 3: Correct only the failing selectors**

Ensure joint observations, actions, reset rules, regularizers, and NaN guards
select `^(?!passive_).*`, while leg-only and head-only rewards keep their
existing named subsets. Do not change the 13D command block.

- [ ] **Step 4: Re-run the contract tests**

Run: `uv run --with pytest pytest tests/test_growbot_roulade_cfg.py -q`

Expected: PASS with 67/80/16.

- [ ] **Step 5: Commit the contract**

```bash
git add src/mjlab_microduck/tasks/growbot_roulade_env_cfg.py tests/test_growbot_roulade_cfg.py
git commit -m "test: lock Growbot policy dimensions"
```

### Task 4: Verify the full repository and run the mandatory smoke train

**Files:**
- Modify only if verification exposes a Growbot-specific defect.

**Interfaces:**
- Consumes: registered `Mjlab-Growbot-Roulade-Flat` task.
- Produces: finite 5-iteration smoke checkpoint and ONNX export.

- [ ] **Step 1: Run the CPU test suite**

Run: `uv run --with pytest pytest tests/ -q`

Expected: all existing Microduck tests and new Growbot tests PASS.

- [ ] **Step 2: Confirm task discovery**

Run: `uv run list-envs | rg 'Mjlab-Growbot-Roulade-Flat'`

Expected: one matching registered task.

- [ ] **Step 3: Run the required smoke train**

Run: `uv run train Mjlab-Growbot-Roulade-Flat --env.scene.num-envs 64 --agent.max-iterations 5 --agent.logger tensorboard`

Expected: iterations 0-4 finish without NaN, OOM, or action-shape errors and save
a checkpoint under `logs/rsl_rl/growbot_roulade/`.

- [ ] **Step 4: Export and record a visual proof**

Run:

```bash
GROWBOT_CKPT=$(find logs/rsl_rl/growbot_roulade -type f -name 'model_*.pt' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
uv run play Mjlab-Growbot-Roulade-Flat --checkpoint-file "$GROWBOT_CKPT" --num-envs 1 --video True --video-length 200 --video-height 720 --video-width 960 --viewer native
```

Expected: an H.264 MP4 showing flat feet and both arms; no rollerblade geometry.

- [ ] **Step 5: Report the verified boundary**

Report compile dimensions, test count, smoke reward/NaN status, checkpoint path,
and video path. Label the 5-iteration policy as an untrained smoke policy, not a
successful roulade.

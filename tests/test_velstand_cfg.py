"""Cfg invariants for the protective-fall VelStand rebuild (2026-09).

CPU only: builds cfgs and compiles the MuJoCo models, never steps an env.
"""
import math

import mujoco
import pytest
import torch

import mjlab_microduck.tasks  # noqa: F401  (registers tasks)
from mjlab_microduck.robot.microduck_constants import (
    SERVO_GEOM_SUFFIX,
    get_allcollisions_backlash_spec,
    get_allcollisions_spec,
    get_standup_spec,
)
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks import microduck_velstand_env_cfg as vs
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
)

PROTECTION_COSTS = ("servo_impact", "head_impact", "trunk_impact", "servo_acc_spike", "servo_stall")


def _geom_names(model):
    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(model.ngeom)]


# ── Model ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("spec_fn", [get_allcollisions_spec, get_allcollisions_backlash_spec])
def test_allcollisions_servo_geoms_are_named(spec_fn):
    model = spec_fn().compile()
    servo = [n for n in _geom_names(model) if n and n.endswith(SERVO_GEOM_SUFFIX)]
    # 14 servos; one body carries a 15th xl330 housing mesh in the export.
    assert len(servo) >= 14
    for n in servo:
        g = model.geom(n)
        assert g.contype[0] or g.conaffinity[0], f"{n} is not a collision geom"
        assert mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, int(g.dataid[0])) == "xl330"


def test_allcollisions_matches_groundcontact_kinematics():
    a = get_allcollisions_spec().compile()
    g = get_standup_spec().compile()
    assert a.njnt == g.njnt
    assert [mujoco.mj_id2name(a, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(a.njnt)] == \
           [mujoco.mj_id2name(g, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(g.njnt)]
    assert (a.jnt_range == g.jnt_range).all()
    assert abs(a.body_subtreemass[0] - g.body_subtreemass[0]) < 1e-9
    assert a.ngeom > g.ngeom  # it really is the fuller collision set


def test_backlash_twin_has_14_backlash_joints():
    m = get_allcollisions_backlash_spec().compile()
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
    assert sum(n.startswith("passive_") and n.endswith("_backlash") for n in names) == 14


# ── Cfg ──────────────────────────────────────────────────────────────────────

def test_env_builds_train_and_play():
    assert vs.make_microduck_velstand_env_cfg() is not None
    assert vs.make_microduck_velstand_env_cfg(play=True) is not None
    assert vs.make_microduck_velstand_env_cfg(rough=True) is not None


def test_uses_allcollisions_robot_and_servo_sensor():
    cfg = vs.make_microduck_velstand_env_cfg()
    assert cfg.scene.entities["robot"].spec_fn is get_allcollisions_spec
    names = {s.name for s in cfg.scene.sensors}
    assert {"servo_ground_contact", "head_ground_contact", "trunk_ground_contact"} <= names
    assert cfg.sim.nconmax >= 200
    servo_sensor = next(s for s in cfg.scene.sensors if s.name == "servo_ground_contact")
    model = get_allcollisions_spec().compile()
    import re
    pat = re.compile(servo_sensor.primary.pattern)
    assert sum(bool(n and pat.match(n)) for n in _geom_names(model)) >= 14


def test_backlash_variant_mirrors_base_model():
    from mjlab_microduck.tasks import _BACKLASH_TASKS
    from mjlab_microduck.robot.microduck_constants import MICRODUCK_ALLCOLLISIONS_BACKLASH_ROBOT_CFG
    rows = [r for r in _BACKLASH_TASKS if r[0].startswith("Mjlab-VelStand-")]
    assert len(rows) == 2
    for row in rows:
        assert row[4] is MICRODUCK_ALLCOLLISIONS_BACKLASH_ROBOT_CFG


def test_penalty_weight_signs():
    cfg = vs.make_microduck_velstand_env_cfg()
    # mjlab-style costs return >= 0 → negative weight
    for name in PROTECTION_COSTS:
        assert cfg.rewards[name].weight < 0, name
    # self-negating microduck penalty → POSITIVE weight
    assert cfg.rewards["gentle_rise"].func is microduck_mdp.trunk_vertical_accel_penalty
    assert cfg.rewards["gentle_rise"].weight > 0
    assert cfg.rewards["joint_torque_rate_l2"].weight < 0
    # protection ramp: stage 0 is the reduced weight, final is the full weight, same sign
    for name in PROTECTION_COSTS + ("gentle_rise",):
        stages = cfg.curriculum[f"{name}_weight"].params["weight_stages"]
        assert stages[0]["step"] == 0
        assert math.isclose(stages[0]["weight"], cfg.rewards[name].weight)
        assert abs(stages[-1]["weight"]) > abs(stages[0]["weight"])
        assert math.copysign(1, stages[-1]["weight"]) == math.copysign(1, stages[0]["weight"])


def test_impact_thresholds_spare_quasi_static_pushoff():
    # Body weight ≈ 0.74 kg × 9.81 ≈ 7.3 N: pushing off the head/trunk to get up
    # must stay free; only impact spikes are billed (2026-07 lesson).
    cfg = vs.make_microduck_velstand_env_cfg()
    assert cfg.rewards["head_impact"].params["threshold"] >= 2 * 7.3
    assert cfg.rewards["trunk_impact"].params["threshold"] >= 2 * 7.3


def test_warm_start_collapses_inherited_curricula_only():
    if not vs.WARM_START:
        pytest.skip("warm start disabled")
    base = make_microduck_velocity_env_cfg()
    cfg = vs.make_microduck_velstand_env_cfg()
    for name, term in base.curriculum.items():
        for key, val in cfg.curriculum[name].params.items():
            if key.endswith("_stages"):
                assert len(val) == 1 and val[0]["step"] == 0, (name, key)
                final = {k: v for k, v in term.params[key][-1].items() if k != "step"}
                assert {k: v for k, v in val[0].items() if k != "step"} == final, (name, key)
    # the loaded walk's action_rate weight is the FINAL one from step 0
    assert cfg.rewards["action_rate_l2"].weight == base.curriculum["action_rate_weight"].params["weight_stages"][-1]["weight"]
    # velstand's own curricula still ramp
    for name in ("fell_over_disable", "prone_init_prob", "fallen_tax_weight", "recovery_success_weight", "topple_push_range"):
        stages = next(v for k, v in cfg.curriculum[name].params.items() if k.endswith("_stages"))
        assert len(stages) > 1, name
    # play cfg keeps the velocity curricula untouched (they don't run in play anyway)
    play = vs.make_microduck_velstand_env_cfg(play=True)
    assert len(play.curriculum["action_rate_weight"].params["weight_stages"]) > 1


def test_phase_order():
    # fell_over off → tax-free window → economics → protection full; prone spawns after econ
    assert vs.FELL_OVER_DISABLE_ITER < vs.RECOVERY_ECON_KICKIN_ITER < vs.PROTECT_FULL_ITER
    prone_start = next(s["step"] for s in vs.PRONE_RAMP_STAGES if s["params"]["prone_prob"] > 0)
    assert prone_start >= vs.RECOVERY_ECON_KICKIN_ITER * vs.NUM_STEPS_PER_ENV
    # topple ramp only grows once fell_over is off
    assert vs.TOPPLE_PUSH_STAGES[1]["step"] == vs.FELL_OVER_DISABLE_ITER * vs.NUM_STEPS_PER_ENV
    assert vs.TOPPLE_PUSH_STAGES[-1]["velocity_range"]["x"][1] >= 1.0  # ≥ the Δv that topples in the CPU baseline


def test_topple_push_event_and_play_range():
    cfg = vs.make_microduck_velstand_env_cfg()
    ev = cfg.events["topple_push"]
    assert ev.mode == "interval"
    assert ev.params["velocity_range"] == vs.TOPPLE_PUSH_STAGES[0]["velocity_range"]
    play = vs.make_microduck_velstand_env_cfg(play=True)
    assert play.events["topple_push"].params["velocity_range"] == vs.TOPPLE_PUSH_STAGES[-1]["velocity_range"]
    assert "topple_push_range" not in play.curriculum


# ── mdp functions (pure torch, fake env) ─────────────────────────────────────

class _Data:
    pass


class _Asset:
    def __init__(self, torque, vel):
        self.data = _Data()
        self.data.actuator_force = torque
        self.data.joint_vel = vel

    def find_joints(self, pattern):
        n = self.data.joint_vel.shape[1]
        return list(range(n)), [f"j{i}" for i in range(n)]


class _Env:
    def __init__(self, torque, vel):
        self.scene = {"robot": _Asset(torque, vel)}
        self.num_envs = torque.shape[0]
        self.device = "cpu"
        self.step_dt = 0.02
        self.episode_length_buf = torch.full((self.num_envs,), 10)


def test_servo_stall_penalty_counts_only_pinned_servos():
    torque = torch.zeros(2, 14)
    vel = torch.zeros(2, 14)
    torque[0, :3] = 0.9          # env 0: three servos at high torque, not moving → stalled
    torque[1, :3] = 0.9
    vel[1, :3] = 3.0             # env 1: same torque but moving → not stalled
    out = microduck_mdp.servo_stall_penalty(_Env(torque, vel))
    assert out.tolist() == [3.0, 0.0]
    assert (out >= 0).all()


def test_servo_acc_spike_penalty_bills_only_above_threshold_and_not_at_reset():
    vel0 = torch.zeros(2, 14)
    env = _Env(torch.zeros(2, 14), vel0)
    microduck_mdp.servo_acc_spike_penalty(env, acc_thresh=300.0)  # prime prev
    vel1 = torch.zeros(2, 14)
    vel1[0, 0] = 10.0            # 10 rad/s in 0.02 s → 500 rad/s² → 200 above threshold
    vel1[1, 0] = 10.0
    env.scene["robot"].data.joint_vel = vel1
    env.episode_length_buf[1] = 0  # env 1 just reset → no bill
    out = microduck_mdp.servo_acc_spike_penalty(env, acc_thresh=300.0)
    assert math.isclose(out[0].item(), 200.0, rel_tol=1e-5)
    assert out[1].item() == 0.0


# ── Patch 5: warm start ──────────────────────────────────────────────────────

def test_warm_start_patch_resets_counters(monkeypatch):
    """MICRODUCK_WARM_START=1 zeroes the env step counter and the iteration
    after a checkpoint load; unset, the restored values are kept (plain resume)."""
    from mjlab.rl.runner import MjlabOnPolicyRunner

    class _Unwrapped:
        common_step_counter = 90024

    class _Env:
        unwrapped = _Unwrapped()

    class _Runner:
        env = _Env()
        current_learning_iteration = 3750

    monkeypatch.setattr(microduck_mdp, "_orig_runner_load", lambda self, path, *a, **k: {"env_state": {"common_step_counter": 90024}})
    monkeypatch.setenv(microduck_mdp.WARM_START_ENV, "1")
    r = _Runner()
    MjlabOnPolicyRunner.load(r, "model_3750.pt")
    assert r.env.unwrapped.common_step_counter == 0
    assert r.current_learning_iteration == 0

    monkeypatch.delenv(microduck_mdp.WARM_START_ENV)
    r = _Runner(); r.env = _Env(); r.env.unwrapped = _Unwrapped()
    MjlabOnPolicyRunner.load(r, "model_3750.pt")
    assert r.env.unwrapped.common_step_counter == 90024
    assert r.current_learning_iteration == 3750


# ── Run-1 fixes: fallen-scaled smoothness + side spawns ──────────────────────

def test_smoothness_costs_are_fallen_scaled():
    cfg = vs.make_microduck_velstand_env_cfg()
    for name in ("action_rate_l2", "joint_torque_rate_l2"):
        term = cfg.rewards[name]
        assert term.func.__name__.endswith("_fallen_scaled"), name
        assert term.weight < 0
        assert 0.0 < term.params["fallen_scale"] < 0.5
        assert term.params["gate_tilt_above_deg"] == vs.REWARD_GATE_TILT_DEG
    # the walk's full action_rate weight is still what the loaded policy was trained with
    assert cfg.rewards["action_rate_l2"].weight == -1.0


def test_prone_init_has_side_spawns():
    cfg = vs.make_microduck_velstand_env_cfg()
    assert cfg.events["random_prone_init"].params["side_prob"] == vs.PRONE_SIDE_PROB > 0
    for st in vs.PRONE_RAMP_STAGES:
        assert st["params"]["side_prob"] == vs.PRONE_SIDE_PROB


class _SimData:
    def __init__(self, n):
        self.qpos = torch.zeros(n, 21); self.qpos[:, 3] = 1.0
        self.qvel = torch.zeros(n, 20)


class _Sim:
    def __init__(self, n): self.data = _SimData(n)


class _SpawnEnv:
    def __init__(self, n):
        self.num_envs = n; self.device = "cpu"; self.sim = _Sim(n); self.scene = {"robot": object()}


def _tilt_and_axes(q):
    w, x, y, z = q.unbind(1)
    R22 = 1 - 2 * (x * x + y * y)                 # body z · world z
    xz = 2 * (x * z - w * y)                      # body x · world z (nose up/down)
    yz = 2 * (y * z + w * x)                      # body y · world z (side)
    return torch.rad2deg(torch.acos(R22.clamp(-1, 1))), xz, yz


def test_side_spawn_quaternions_lie_on_a_side():
    torch.manual_seed(0)
    env = _SpawnEnv(256)
    microduck_mdp.set_random_prone_orientation(env, torch.arange(256), face_down_prob=0.5, side_prob=1.0)
    q = env.sim.data.qpos[:, 3:7]
    assert torch.allclose(q.norm(dim=1), torch.ones(256), atol=1e-5)
    tilt, xz, yz = _tilt_and_axes(q)
    assert (tilt > 89).all() and (tilt < 91).all()
    assert (xz.abs() < 1e-4).all()                # nose horizontal → not face down/up
    assert ((yz - 1).abs() < 1e-4).sum() > 80 and ((yz + 1).abs() < 1e-4).sum() > 80  # both sides used


def test_side_prob_zero_keeps_face_spawns():
    torch.manual_seed(0)
    env = _SpawnEnv(128)
    microduck_mdp.set_random_prone_orientation(env, torch.arange(128), face_down_prob=1.0, side_prob=0.0)
    tilt, xz, yz = _tilt_and_axes(env.sim.data.qpos[:, 3:7])
    assert (tilt > 89).all() and (xz < -0.99).all()   # face-down: nose points down


def test_fallen_scaled_action_rate(monkeypatch):
    class _AM:
        action = torch.tensor([[1.0] * 14, [1.0] * 14]); prev_action = torch.zeros(2, 14)

    class _E:
        num_envs = 2; device = "cpu"; action_manager = _AM(); scene = {"robot": object()}

    calls = {}
    def fake_mask(env, asset, z_below, tilt_above):
        calls["tilt"] = tilt_above; return torch.tensor([True, False])
    monkeypatch.setattr(microduck_mdp, "_fallen_mask", fake_mask)
    out = microduck_mdp.action_rate_l2_fallen_scaled(_E(), fallen_scale=0.1, gate_tilt_above_deg=40.0)
    assert calls["tilt"] == 40.0
    assert torch.allclose(out, torch.tensor([1.4, 14.0]))


# ── Run-2 fix: fallen-gated expert BC ────────────────────────────────────────

def test_runner_cfg_uses_expert_bc():
    from rsl_rl.utils import resolve_callable
    from mjlab_microduck.tasks import distill
    alg = vs.MicroduckVelStandRlCfg.algorithm
    assert resolve_callable(alg.class_name) is distill.PpoWithExpertBc
    if vs.ENABLE_EXPERT_BC:
        assert alg.bc_cfg["coef"] > 0
        assert alg.bc_cfg["gate_tilt_deg"] <= vs.REWARD_GATE_TILT_DEG
        assert tuple(alg.bc_cfg["gravity_slice"]) == (3, 6)
        assert tuple(alg.bc_cfg["twist_slice"]) == (48, 51)
        assert alg.bc_cfg["wandb_run_path"].endswith("69u48n8l")  # alpha_stand.onnx's run


def test_fallen_mask_from_obs():
    from mjlab_microduck.tasks.distill import fallen_mask_from_obs, expert_input
    obs = torch.zeros(4, 61)
    obs[0, 3:6] = torch.tensor([0.0, 0.0, -1.0])                       # upright
    obs[1, 3:6] = torch.tensor([math.sin(math.radians(30)), 0, -math.cos(math.radians(30))])   # 30°
    obs[2, 3:6] = torch.tensor([math.sin(math.radians(60)), 0, -math.cos(math.radians(60))])   # 60°
    obs[3, 3:6] = torch.tensor([0.0, 1.0, 0.0])                        # on the side
    m = fallen_mask_from_obs(obs, (3, 6), 35.0)
    assert m.tolist() == [False, False, True, True]
    obs[:, 48:51] = 1.0
    e = expert_input(obs, (48, 51))
    assert (e[:, 48:51] == 0).all() and (obs[:, 48:51] == 1).all()   # copy, not in place
    assert torch.equal(e[:, :48], obs[:, :48])


def test_load_expert_is_frozen_copy():
    from mjlab_microduck.tasks.distill import load_expert_from
    actor = torch.nn.Sequential(torch.nn.Linear(3, 2))
    sd = {"0.weight": torch.ones(2, 3), "0.bias": torch.zeros(2)}
    exp = load_expert_from(actor, sd)
    assert exp is not actor and not exp.training
    assert all(not p.requires_grad for p in exp.parameters())
    assert torch.equal(exp[0].weight, torch.ones(2, 3)) and not torch.equal(actor[0].weight, torch.ones(2, 3))


# ── Run-4 fix: post-fall-like prone spawns + stall weight ───────────────────

def test_prone_init_randomizes_joints_for_post_fall_like_spawns():
    cfg = vs.make_microduck_velstand_env_cfg()
    p = cfg.events["random_prone_init"].params
    assert 0.5 <= p["joint_random_prob"] < 1.0          # most prone spawns post-fall-like, some keep HOME
    assert 0.5 <= p["joint_range_frac"] <= 0.9
    assert cfg.rewards["servo_stall"].weight <= -0.15 * vs.PROTECT_STAGE0_FRAC
    assert cfg.curriculum["servo_stall_weight"].params["weight_stages"][-1]["weight"] == vs.SERVO_STALL_WEIGHT <= -0.15


def test_randomize_servo_joints_uniform_respects_limits(monkeypatch):
    lo = torch.tensor([-1.0] * 14); hi = torch.tensor([1.0] * 14); hi[7] = 3.0; lo[7] = -3.0
    written = {}

    class _D:
        joint_pos_limits = torch.stack([lo, hi], dim=-1).unsqueeze(0).repeat(4, 1, 1)
        joint_pos = torch.zeros(4, 14)

    class _A:
        data = _D()
        def find_joints(self, pattern): return list(range(14)), [f"j{i}" for i in range(14)]
        def write_joint_position_to_sim(self, pos, joint_ids=None, env_ids=None): assert joint_ids is None and env_ids is not None; written["pos"] = pos.clone()
        def write_joint_velocity_to_sim(self, vel, joint_ids=None, env_ids=None): assert env_ids is not None; written["vel"] = vel.clone()

    class _E:
        device = "cpu"; scene = {"robot": _A()}

    torch.manual_seed(0)
    microduck_mdp.randomize_servo_joints_uniform(_E(), torch.arange(4), range_frac=0.8)
    pos = written["pos"]
    assert pos.shape == (4, 14) and (written["vel"] == 0).all()
    assert (pos[:, :7].abs() <= 0.8 + 1e-6).all() and (pos[:, 7].abs() <= 2.4 + 1e-6).all()
    assert pos.std() > 0.3  # actually randomized, not HOME

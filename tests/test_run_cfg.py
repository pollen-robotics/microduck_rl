"""Cfg invariants for the Run task (fast alternating-leg running). CPU only."""
import mjlab_microduck.tasks  # noqa: F401  (registers tasks)
from mjlab.tasks.registry import list_tasks
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks import microduck_run_env_cfg as run
from mjlab_microduck.tasks.microduck_velocity_env_cfg import make_microduck_velocity_env_cfg

RUN_REWARDS = ("run_flight", "run_alternating_step", "run_foot_reach", "run_pushoff")


def test_registered_with_backlash_twin():
    ids = set(list_tasks())
    for t in ("Mjlab-Run-Flat-MicroDuck", "Mjlab-Run-Flat-Backlash-MicroDuck"):
        assert t in ids


def test_obs_layout_matches_velocity_family():
    cfg, vel = run.make_microduck_run_env_cfg(), make_microduck_velocity_env_cfg()
    for group in ("actor", "critic"):
        assert list(cfg.observations[group].terms) == list(vel.observations[group].terms)
    assert "head_command" in cfg.observations["actor"].terms
    assert "body_command" in cfg.observations["actor"].terms


def test_run_reward_signs():
    cfg = run.make_microduck_run_env_cfg()
    for name in RUN_REWARDS:
        assert cfg.rewards[name].weight > 0, name
    # self-negating penalty (<= 0) -> weight must be >= 0 (starts at 0, ramped)
    assert cfg.rewards["run_double_support"].func is microduck_mdp.run_double_support_penalty
    assert cfg.rewards["run_double_support"].weight == 0.0
    assert cfg.rewards["run_head_sway"].weight == 0.0  # late curriculum


def test_late_terms_curricula_start_at_initial_weight():
    cfg = run.make_microduck_run_env_cfg()
    for term in ("run_double_support", "run_head_sway"):
        stages = cfg.curriculum[f"{term}_weight"].params["weight_stages"]
        assert stages[0]["step"] == 0 and stages[0]["weight"] == cfg.rewards[term].weight
        assert stages[-1]["weight"] > 0


def test_command_is_forward_heavy_and_slots_alive():
    cfg = run.make_microduck_run_env_cfg()
    r = cfg.commands["twist"].ranges
    assert r.lin_vel_x[1] > 0.4 > -r.lin_vel_x[0]
    assert cfg.commands["twist"].rel_forward_envs > 0
    # head / body slots keep non-zero ranges (live input neurons)
    assert cfg.commands["head_pose"].ranges[0][1] > 0
    assert cfg.commands["body_pose"].ranges[0][1] > 0
    stages = [s["lin_vel_x"][1] for s in run.RUN_SPEED_STAGES]
    assert stages == sorted(stages) and stages[-1] > 1.0


def test_gait_targets_more_aggressive_than_walk():
    cfg, vel = run.make_microduck_run_env_cfg(), make_microduck_velocity_env_cfg()
    assert cfg.rewards["foot_clearance"].params["target_height"] > vel.rewards["foot_clearance"].params["target_height"]
    assert cfg.rewards["pose"].params["std_running"][r".*knee.*"] > vel.rewards["pose"].params["std_walking"][r".*knee.*"]
    assert abs(cfg.rewards["body_ang_vel"].weight) < abs(vel.rewards["body_ang_vel"].weight)
    # velocity cfg is not mutated by building the run cfg (shared-state guard)
    assert vel.rewards["foot_clearance"].params["target_height"] == 0.02


def test_foot_reach_resolves_both_feet():
    cfg = run.make_microduck_run_env_cfg()
    assert tuple(cfg.rewards["run_foot_reach"].params["asset_cfg"].site_names) == ("left_foot", "right_foot")

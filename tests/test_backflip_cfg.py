"""Config invariants for the Backflip env (CPU-only, no GPU / no sim step).

These lock the things that fail SILENTLY at training time: a sensor name that
no longer matches what the mdp gate reads (``_sensor_any_contact`` fails OPEN —
a typo turns the anti-farming gate into a no-op with no error), the 61D
actor-obs layout the whole policy family shares, entity order (base reset
events write robot root state at ``qpos[:, 0:7]``), reward signs, and the
reset-event ordering that puts the robot on the plate at the sampled ``z0``.
"""

import math

import pytest
import torch

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_backflip_env_cfg import (
    MicroduckBackflipRlCfg,
    PLATE_HALF_THICKNESS,
    STAND_Z,
    make_microduck_backflip_env_cfg,
)
from mjlab_microduck.tasks.microduck_roulade_env_cfg import (
    make_microduck_roulade_env_cfg,
)


@pytest.fixture(scope="module")
def cfg():
    return make_microduck_backflip_env_cfg()


# ── Scene ────────────────────────────────────────────────────────────────────


def test_robot_stays_the_first_entity_and_the_plate_is_present(cfg):
    # Base reset events write robot root state at qpos[:, 0:7]: robot MUST be
    # the first entity or every reset writes into the wrong body.
    names = list(cfg.scene.entities.keys())
    assert names[0] == "robot"
    assert "plate" in names


def test_contact_headroom_is_raised_for_the_plate(cfg):
    assert cfg.sim.nconmax >= 50


# ── Sensor wiring (fails OPEN if wrong — assert it here) ─────────────────────


def test_airborne_gate_sensor_exists_under_the_name_the_mdp_reads(cfg):
    # _update_backflip_accum looks the sensor up by this exact name; a miss
    # returns None and the airborne gate silently stops gating.
    names = {s.name for s in cfg.scene.sensors}
    assert microduck_mdp._BACKFLIP_GROUND_SENSOR in names


def test_landing_term_names_a_sensor_that_actually_exists(cfg):
    names = {s.name for s in cfg.scene.sensors}
    assert cfg.rewards["landing"].params["sensor_name"] in names


# ── Observations ─────────────────────────────────────────────────────────────


def test_actor_obs_matches_the_shared_layout_and_is_plate_blind(cfg):
    # The 61D actor layout is shared across the whole policy family so ONNX
    # policies are hot-swappable on the robot. Roulade is the template this env
    # was built from: the actor term SET must be identical, term for term.
    # Term ORDER is what fixes the 61D layout (terms are concatenated in
    # insertion order), so compare the list, not the set.
    actor = list(cfg.observations["actor"].terms.keys())
    roulade_actor = list(
        make_microduck_roulade_env_cfg().observations["actor"].terms.keys()
    )
    assert actor == roulade_actor
    # 48 proprioception + [twist(3), head_pose(4), body_pose(6)]:
    #   base_ang_vel 3 + projected_gravity 3 + joint_pos 14 + joint_vel 14
    #   + actions 14 = 48, then command 3 + head_command 4 + body_command 6.
    assert actor == [
        "base_ang_vel",
        "projected_gravity",
        "joint_pos",
        "joint_vel",
        "actions",
        "command",
        "head_command",
        "body_command",
    ]
    # The 14 servo joints only — a passive_* joint slipping into the obs would
    # silently change the actor width on the backlash / roller models.
    for term in ("joint_pos", "joint_vel"):
        names = cfg.observations["actor"].terms[term].params["asset_cfg"].joint_names
        assert names == (r"^(?!passive_).*",)
    # The real robot has no launcher sensing: it feels the toss through its IMU.
    assert not any("plate" in n or "backflip" in n for n in actor)


def test_critic_sees_the_plate_but_the_actor_does_not(cfg):
    critic = set(cfg.observations["critic"].terms.keys())
    assert any("plate" in n or "backflip" in n for n in critic)
    assert not any(
        "plate" in n or "backflip" in n for n in cfg.observations["actor"].terms
    )


def test_command_slots_stay_zero_padded_not_deleted(cfg):
    for group in ("actor", "critic"):
        terms = cfg.observations[group].terms
        assert terms["head_command"].params["dim"] == 4
        assert terms["body_command"].params["dim"] == 6


def test_backflip_phase_obs_is_a_column_of_phase_ids():
    class _Env:
        num_envs = 3
        device = "cpu"
        step_dt = 0.02
        episode_length_buf = torch.zeros(3, dtype=torch.long)

    env = _Env()
    out = microduck_mdp.backflip_phase_obs(env)
    assert out.shape == (3, 1)
    assert out.dtype == torch.float32
    assert torch.equal(out.squeeze(-1), microduck_mdp.backflip_phase(env).float())


# ── Events ───────────────────────────────────────────────────────────────────


def test_plate_event_runs_every_step(cfg):
    term = cfg.events["backflip_plate"]
    assert term.mode == "step"
    assert term.func is microduck_mdp.backflip_plate_step
    assert term.params["asset_name"] == "plate"


def test_launch_params_are_sampled_on_reset_within_the_operator_range(cfg):
    term = cfg.events["backflip_launch_params"]
    assert term.mode == "reset"
    assert term.func is microduck_mdp.reset_backflip_launch_params
    lo, hi = term.params["z0_range"]
    assert lo >= 0.10 and hi <= 0.30  # the operator's hands, per the spec


def test_launch_envelope_is_the_measured_box_not_the_placeholders(cfg):
    # docs/backflip_envelope_results.md: vz has a CLIFF below 2.0 (1.80 fails
    # at 347deg) and w0 above ~36-39 reverses the rotation direction.
    p = cfg.events["backflip_launch_params"].params
    vz_lo, vz_hi = p["vz_range"]
    w0_lo, w0_hi = p["w0_range"]
    assert vz_lo >= 2.0
    assert w0_hi <= 30.0
    assert w0_lo >= 24.0 and vz_hi <= 2.25


def test_robot_is_placed_on_the_plate_after_z0_is_sampled(cfg):
    # The spawn height derives from the sampled z0, so the ordering is
    # load-bearing (events run in dict insertion order).
    order = list(cfg.events.keys())
    assert order.index("reset_base") < order.index("backflip_launch_params")
    assert order.index("backflip_launch_params") < order.index("backflip_spawn")
    spawn = cfg.events["backflip_spawn"]
    assert spawn.mode == "reset"
    assert spawn.func is microduck_mdp.reset_backflip_robot_on_plate


def test_the_robot_spawns_over_the_plate_not_half_a_metre_away(cfg):
    # The plate is 18x18cm and sits at the env origin; the base template's
    # ±0.5 m spawn scatter would drop the robot on the floor beside it.
    # Yaw must stay near zero too: the flick axis is world +y, so a random
    # heading turns the backflip into a side flip.
    pose = cfg.events["reset_base"].params["pose_range"]
    for axis in ("x", "y"):
        lo, hi = pose[axis]
        assert abs(lo) <= 0.02 and abs(hi) <= 0.02
    lo, hi = pose["yaw"]
    assert abs(lo) <= 0.2 and abs(hi) <= 0.2


def test_the_plate_is_also_parked_during_the_reset_itself(cfg):
    # ManagerBasedRlEnv.reset() does not run step-mode events, so without a
    # reset-mode placement the first physics step after construction runs with
    # the plate at its XML default while the robot stands at the sampled z0.
    term = cfg.events["backflip_plate_reset"]
    assert term.mode == "reset"
    assert term.func is microduck_mdp.backflip_plate_step
    order = list(cfg.events.keys())
    assert order.index("backflip_launch_params") < order.index("backflip_plate_reset")


def test_no_mid_flip_pushes(cfg):
    assert "push_robot" not in cfg.events


# ── Rewards ──────────────────────────────────────────────────────────────────


def test_progress_is_the_dominant_positive_term(cfg):
    assert cfg.rewards["flip_progress"].weight > 0.0
    assert cfg.rewards["landing"].weight > 0.0
    assert cfg.rewards["ready_stance"].weight > 0.0
    # the stance term must not compete with the flip
    assert cfg.rewards["ready_stance"].weight < cfg.rewards["flip_progress"].weight


def test_ready_stance_target_height_accounts_for_the_plate_thickness(cfg):
    # backflip_ready_stance measures trunk z against (origin + z0 + stand_z),
    # but the robot stands on the plate TOP, one half-thickness above z0.
    assert cfg.rewards["ready_stance"].params["stand_z"] == pytest.approx(
        STAND_Z + PLATE_HALF_THICKNESS
    )
    # The landing is measured against the ground, with no plate under it.
    assert cfg.rewards["landing"].params["stand_z"] == pytest.approx(STAND_Z)


def test_progress_pay_cap_is_the_measured_one(cfg):
    # 14 rad/s (the pre-measurement placeholder) would forfeit rotation during
    # a perfectly good flip; the envelope needs ~13-16 rad/s on average.
    assert cfg.rewards["flip_progress"].params["max_paid_rate"] >= 20.0
    assert cfg.rewards["flip_progress"].params["target_angle"] == pytest.approx(
        2 * math.pi
    )


# The sign convention that has bitten four envs. It CANNOT be checked from the
# function name: mjlab's own body_angular_velocity_penalty returns >= 0 (a cost,
# negative weight) while microduck's trunk_vertical_accel_penalty returns <= 0
# (self-negating, POSITIVE weight) — same suffix, opposite sign. So the sign of
# every term's function is stated here explicitly, and the test also fails when
# a new reward term is added without classifying it. A wrong weight sign turns a
# penalty into a bounty for the violation, which the policy WILL farm.
_TERM_SIGNS = {
    # task terms: the function returns >= 0 and we want it → positive weight
    "flip_progress":        "bonus",
    "landing":              "bonus",
    "ready_stance":         "bonus",
    # costs: the function returns >= 0 → NEGATIVE weight (may be 0 pre-curriculum)
    "body_ang_vel":         "cost",
    "angular_momentum":     "cost",
    "dof_pos_limits":       "cost",
    "action_rate_l2":       "cost",
    "joint_torque_rate_l2": "cost",
    "arrival_damping":      "cost",
    "self_collisions":      "cost",
    # self-negating: the function returns <= 0 → POSITIVE weight
    "gentle_landing":       "self_negating",
}


def test_every_reward_term_is_classified(cfg):
    assert set(cfg.rewards.keys()) == set(_TERM_SIGNS)


def test_every_penalty_term_carries_the_sign_that_makes_it_a_cost(cfg):
    for name, term in cfg.rewards.items():
        kind = _TERM_SIGNS[name]
        if kind == "cost":
            assert term.weight <= 0.0, f"{name}: a >=0 cost needs a negative weight"
        else:
            assert term.weight > 0.0, f"{name}: {kind} term needs a positive weight"


def test_the_self_negating_term_really_is_self_negating():
    # Pins the classification above to the actual function, so the table can't
    # drift away from the code it is asserting about.
    import inspect

    src = inspect.getsource(microduck_mdp.trunk_vertical_accel_penalty)
    assert "-torch.abs" in src or "return -" in src


def test_motion_blockers_stay_low_for_this_dynamic_task(cfg):
    # A backflip IS a large angular-velocity event: taxing it blocks discovery.
    for name in ("body_ang_vel", "angular_momentum"):
        if name in cfg.rewards:
            assert abs(cfg.rewards[name].weight) <= 0.05, name


def test_no_always_on_upright_term_opposes_the_flip(cfg):
    assert "upright" not in cfg.rewards


def test_walking_terms_are_gone(cfg):
    for name in (
        "track_linear_velocity",
        "track_angular_velocity",
        "air_time",
        "foot_slip",
    ):
        assert name not in cfg.rewards


# ── Curriculum ───────────────────────────────────────────────────────────────


def test_smoothness_is_introduced_by_curriculum_not_at_full_strength(cfg):
    stages = cfg.curriculum["torque_rate_weight"].params["weight_stages"]
    assert stages[0]["step"] == 0 and stages[0]["weight"] == 0.0
    assert stages[-1]["step"] > 0
    # every stage of a mjlab-base cost stays <= 0
    assert all(s["weight"] <= 0.0 for s in stages)


def test_impact_penalty_ramps_up_and_keeps_the_self_negating_sign(cfg):
    stages = cfg.curriculum["gentle_landing_weight"].params["weight_stages"]
    assert all(s["weight"] > 0.0 for s in stages)  # self-negating func
    assert stages[-1]["weight"] > stages[0]["weight"]  # ramps UP


def test_launch_dr_widens_over_training(cfg):
    hold = cfg.curriculum["backflip_hold_range"].params["param_stages"]
    assert hold[0]["params"]["hold_range"] == (0.1, 0.4)
    assert hold[-1]["params"]["hold_range"][1] > 0.4
    z0 = cfg.curriculum["backflip_z0_range"].params["param_stages"]
    assert z0[-1]["params"]["z0_range"][1] <= 0.30


# ── Runner cfg / registration ────────────────────────────────────────────────


def test_symmetry_mirror_loss_is_enabled():
    assert MicroduckBackflipRlCfg.algorithm.symmetry_cfg is not None


def test_task_is_registered():
    from mjlab.tasks.registry import list_tasks

    import mjlab_microduck.tasks  # noqa: F401

    tasks = list_tasks()
    assert "Mjlab-Backflip-Flat-MicroDuck" in tasks
    assert "Mjlab-Backflip-Flat-Backlash-MicroDuck" in tasks


def test_backlash_variant_keeps_the_plate_and_the_entity_order():
    from mjlab_microduck.robot.microduck_constants import MICRODUCK_BACKLASH_ROBOT_CFG
    from mjlab_microduck.tasks.backlash import make_backlash_variant

    bl = make_backlash_variant(
        make_microduck_backflip_env_cfg(), MICRODUCK_BACKLASH_ROBOT_CFG
    )
    names = list(bl.scene.entities.keys())
    assert names[0] == "robot"
    assert "plate" in names
    assert bl.scene.entities["robot"] is MICRODUCK_BACKLASH_ROBOT_CFG

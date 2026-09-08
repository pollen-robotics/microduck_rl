"""Config invariants for the Backflip env (CPU-only, no GPU / no sim step).

These lock the things that fail SILENTLY at training time: a sensor name that
no longer matches what the mdp gate reads (``_sensor_any_contact`` fails OPEN —
a typo turns the anti-farming gate into a no-op with no error), the 61D
actor-obs layout the whole policy family shares, entity order (base reset
events write robot root state at ``qpos[:, 0:7]``), reward signs, and the
reset-event ordering that puts the robot on the plate at the sampled ``z0``.
"""

import math

import mujoco
import pytest
import torch

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_backflip_env_cfg import (
    HOLD_RANGE,
    LANDING_WINDOW_S,
    LAUNCH_RANGE,
    VZ_RANGE,
    W0_RANGE,
    MicroduckBackflipRlCfg,
    PLATE_HALF_THICKNESS,
    STAND_Z,
    TUCK_FACTOR,
    TUCK_OVERRIDES,
    TUCK_Z,
    Z0_RANGE,
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
    # The slab rests ON the floor at the low end (z0 = PLATE_HALF_THICKNESS);
    # the spread above it is operator variation.
    assert lo >= PLATE_HALF_THICKNESS and hi <= 0.05


def test_launch_envelope_is_the_measured_box_not_the_placeholders(cfg):
    # The box is asserted in full by test_the_standing_box_is_the_measured_one;
    # this pins that the mdp DEFAULTS say the same thing, so an env built
    # without the cfg (or a copy-paste into a new task) cannot inherit a stale
    # box -- and every historical box on this branch is stale.
    import inspect

    defaults = inspect.signature(
        microduck_mdp.reset_backflip_launch_params
    ).parameters
    assert defaults["vz_range"].default == VZ_RANGE
    assert defaults["w0_range"].default == W0_RANGE
    assert defaults["launch_range"].default == LAUNCH_RANGE
    assert defaults["z0_range"].default == Z0_RANGE
    assert defaults["hold_range"].default == HOLD_RANGE


def test_robot_is_placed_on_the_plate_after_z0_is_sampled(cfg):
    # The spawn height derives from the sampled z0, so the ordering is
    # load-bearing (events run in dict insertion order).
    order = list(cfg.events.keys())
    assert order.index("reset_base") < order.index("backflip_launch_params")
    assert order.index("backflip_launch_params") < order.index("backflip_spawn")
    # The spawn SHIFTS the joints reset_robot_joints scattered (rather than
    # overwriting them), so it must run after that event too. This held only by
    # luck of insertion order until it was pinned here.
    assert order.index("reset_robot_joints") < order.index("backflip_spawn")
    spawn = cfg.events["backflip_spawn"]
    assert spawn.mode == "reset"
    assert spawn.func is microduck_mdp.reset_backflip_robot_on_plate


def test_the_spawn_height_and_the_stance_target_are_the_same_number(cfg):
    # THE invariant between the spawn and the reward that scores it. The spawn
    # puts the trunk at z0 + plate_half_thickness + stand_z; ready_stance
    # scores it against z0 + its own stand_z. Edit either alone and the robot
    # spawns at a height its own hold reward calls wrong.
    spawn = cfg.events["backflip_spawn"].params
    stance = cfg.rewards["ready_stance"].params
    assert spawn["stand_z"] + spawn["plate_half_thickness"] == pytest.approx(
        stance["stand_z"]
    )
    assert spawn["stand_z"] == pytest.approx(STAND_Z)
    assert spawn["plate_half_thickness"] == pytest.approx(PLATE_HALF_THICKNESS)


def test_the_hold_posture_is_standing():
    # The tucked hold was tried and reverted: its 1.5-2.2 m/s advantage came
    # from a spawn whose FEET were tunnelled under the plate slab, and from a
    # valid rest the two postures land the same (3.38 vs 3.43 m/s). Standing
    # puts the feet on the plate BY CONSTRUCTION, so no spawn can tunnel.
    import inspect

    params = inspect.signature(
        microduck_mdp.reset_backflip_robot_on_plate
    ).parameters
    assert "stand_z" in params
    assert "tuck_overrides" not in params and "tuck_z" not in params
    stance = inspect.signature(microduck_mdp.backflip_ready_stance).parameters
    assert "stand_z" in stance
    assert "tuck_overrides" not in stance


def test_ready_stance_keeps_its_upright_factor():
    # Dropped once, during the tucked-hold experiment, and the side-lying basin
    # immediately outscored the intended pose (0.991 vs 0.948) because height
    # alone cannot tell upright from inverted. The gate must be WIDE so that
    # standing costs nothing.
    import inspect

    stance = inspect.signature(microduck_mdp.backflip_ready_stance).parameters
    assert "tilt_full_deg" in stance and "tilt_zero_deg" in stance
    params = cfg_tilt = cfg = None
    del params, cfg_tilt, cfg


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
    # and the upright gate is present, tight enough to price a lean but wide
    # enough not to tax the pose's own drift (see
    # test_the_lean_penalty_has_the_shape_the_docstring_claims)
    assert 5.0 <= cfg.rewards["ready_stance"].params["tilt_full_deg"] <= 15.0
    assert cfg.rewards["ready_stance"].params["tilt_zero_deg"] <= 70.0
    # The landing is measured against the ground, with no plate under it, and
    # against the STANDING height: the duck lands on its feet.
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


# --- Fix-wave additions. -----------------------------------------------------


def test_arrival_damping_curriculum_keeps_the_cost_sign_at_every_stage(cfg):
    # The one staged table whose signs were unchecked while the other two were.
    # body_ang_vel_at_height is an mjlab-style POSITIVE cost, so every stage
    # must be <= 0; a positive stage would pay for trunk thrash at the landing.
    stages = cfg.curriculum["arrival_damping_weight"].params["weight_stages"]
    assert stages[0]["step"] == 0 and stages[0]["weight"] == 0.0
    assert all(s["weight"] <= 0.0 for s in stages)
    assert stages[-1]["weight"] < stages[0]["weight"]   # ramps DOWN (stronger)
    assert stages[-1]["step"] > 0                       # after skill discovery


def test_arrival_damping_is_a_height_window_not_a_floor(cfg):
    # body_ang_vel_at_height's height_low/height_high pair is a FLOOR: without
    # an upper edge every airborne step of the flip pays full cost, and the
    # tilt gate alone lets a rotating robot through twice per revolution.
    params = cfg.rewards["arrival_damping"].params
    assert params["height_high"] < params["height_full_max"] < params["height_zero_max"]

    # the landed STANDING trunk must be inside the full-cost band
    assert params["height_full_max"] > STAND_Z

    # ... and the flight must be outside it. Measured apex in the retuned box
    # is 0.29-0.40 m, so anything at or above 0.2 m is comfortably clear.
    assert params["height_zero_max"] < 0.2

    # The HOLD is separable by height again, and the ceiling must exclude it:
    # standing on a floor-resting plate puts the trunk at
    # z0 + PLATE_HALF_THICKNESS + STAND_Z = 0.135-0.155 m, above STAND_Z, so
    # the damper must switch off before it. (It could NOT be separated when the
    # hold was a tuck at z0 0.07-0.09 -- that trunk straddled STAND_Z -- which
    # is why this assertion is worth keeping explicit.)
    hold_lo = Z0_RANGE[0] + PLATE_HALF_THICKNESS + STAND_Z
    assert hold_lo > STAND_Z
    assert params["height_zero_max"] < hold_lo, (
        "arrival_damping fires during the HOLD; re-derive its ceiling against "
        f"the standing hold trunk height ({hold_lo:.3f} m)"
    )


def test_plate_reset_event_passes_t_zero_explicitly(cfg):
    # episode_length_buf is zeroed AFTER reset events, so without an explicit
    # t=0 this event reads the terminal episode's time, the phase comes out
    # GONE, and it parks the plate 5 m away instead of placing it at z0.
    # (The placement itself is measured in test_backflip_mdp.py.)
    assert cfg.events["backflip_plate_reset"].params["t_override"] == 0.0


def test_critic_plate_terms_are_the_gone_masked_ones(cfg):
    # The parked plate sits ~8.7 m away for ~85% of every episode's steps; an
    # unmasked plate_position normalizer converges to std ~3 m and squashes the
    # informative 0-0.3 m HOLD/LAUNCH range into noise. The mask itself is
    # measured in test_backflip_mdp.py.
    critic = cfg.observations["critic"].terms
    assert critic["plate_position"].func is microduck_mdp.backflip_plate_pos_obs
    assert critic["plate_velocity"].func is microduck_mdp.backflip_plate_vel_obs
    # The actor must still carry no plate term at all, masked or not.
    assert not any("plate" in name for name in cfg.observations["actor"].terms)


def test_there_is_no_z0_dr_tail(cfg):
    # The tail was removed, and both ends of the argument are measured
    # (docs "Lower and gentler"): upward, whole-box landing speed crosses the
    # ~2.6 m/s hardware limit between z0=0.21 and 0.225; downward, Z0_RANGE is
    # floored at 0.07 by the hold pose's own geometry (the kneeling tuck's feet
    # hang ~8 cm below the surface it rests on). A 2 cm range does not need a
    # curriculum stage. If the hold posture changes, re-measure first.
    assert "backflip_z0_range" not in cfg.curriculum
    assert cfg.events["backflip_launch_params"].params["z0_range"] == Z0_RANGE
    # the slab rests ON the floor at the low end (z0 = PLATE_HALF_THICKNESS)
    assert Z0_RANGE == (0.01, 0.03)
    assert Z0_RANGE[0] == pytest.approx(PLATE_HALF_THICKNESS)


def test_the_standing_box_is_a_plausible_human_throw(cfg):
    # v0 ranges: the spread a person's hands plausibly deliver, NOT ranges
    # tuned so an unskilled robot completes every throw. Requiring whole-box
    # open-loop closure once squeezed w0 to the single value 18.5 rad/s, which
    # no hand reproduces, and it is the wrong bar -- compensating for an
    # imperfect throw is the policy's job. Measured over 243 cells from the
    # floor-resting plate: 0 rotate forward, 27% close 360 deg open-loop,
    # rotation 98-442 deg, landing 1.78-3.65 m/s, apex 0.28-0.64 m -- the
    # gentler retune after the user said the ejection was too strong.
    p = cfg.events["backflip_launch_params"].params
    assert p["vz_range"] == VZ_RANGE == (2.20, 2.80)
    assert p["w0_range"] == W0_RANGE == (18.0, 24.0)
    assert p["launch_range"] == LAUNCH_RANGE == (0.14, 0.16)
    assert p["z0_range"] == Z0_RANGE == (0.01, 0.03)


def test_the_flick_stays_inside_the_direction_reversal_boundary(cfg):
    # THE one hard limit. Above roughly w0 24-27 rad/s from a standing hold the
    # flick overdrives the sole contact and the robot comes out FORWARD,
    # face-down -- a different maneuver the accumulator would have to be
    # re-signed to score. Every corner of the box is direction-checked backward;
    # widening w0 past this needs a fresh --box-check.
    assert cfg.events["backflip_launch_params"].params["w0_range"][1] <= 24.0
    # ... and the gate binds from BELOW too: a low vz with a low w0 and a short
    # flick also comes out forward (vz 2.2, w0 15, t_launch 0.12 = -110 deg),
    # which is what sets these two floors.
    assert cfg.events["backflip_launch_params"].params["w0_range"][0] >= 18.0
    assert cfg.events["backflip_launch_params"].params["launch_range"][0] >= 0.14


def test_the_pre_reset_state_is_coherent(cfg):
    # mjlab never calls env.reset() before the viewer's first episode
    # (ManagerBasedRlEnv.__init__ does not reset; mjlab/viewer/base.py calls
    # reset only on the RESET action), so `uv run play` runs a whole 4 s
    # episode on the COMPILED default state. It used to be nonsense: the robot
    # standing at its compiled 0.12 m trunk height with the plate hovering at
    # 0.15 m, i.e. through its body -- reported by a user as "the plate is
    # stuck in the middle of the robot, not under its feet". Both entities must
    # therefore compile to a coherent pose, not just reset to one.
    robot = cfg.scene.entities["robot"]
    plate = cfg.scene.entities["plate"]
    z0_mid = 0.5 * (Z0_RANGE[0] + Z0_RANGE[1])

    # the plate's default height is inside the sampled range
    assert Z0_RANGE[0] <= plate.init_state.pos[2] <= Z0_RANGE[1]
    # the robot's default trunk sits on the plate top, not on the floor
    assert robot.init_state.pos[2] == pytest.approx(
        z0_mid + PLATE_HALF_THICKNESS + STAND_Z
    )
    # and the plate is BELOW the trunk, never through it
    assert plate.init_state.pos[2] + PLATE_HALF_THICKNESS < robot.init_state.pos[2]

    # standup/roulade must keep their own HOME init: the tuck is a deepcopy
    from mjlab_microduck.robot.microduck_constants import (
        MICRODUCK_STANDUP_ROBOT_CFG,
    )

    assert robot is not MICRODUCK_STANDUP_ROBOT_CFG
    assert MICRODUCK_STANDUP_ROBOT_CFG.init_state.pos[2] != robot.init_state.pos[2]


def test_the_default_launch_params_are_a_plausible_toss(cfg):
    # Same reason: the lazy _backflip_state defaults are what the un-reset
    # first episode flies. They must be a real toss inside the box, not zeros
    # (which put the phase past HOLD at t=0 and made the plate vanish without
    # ever launching).
    del cfg
    import torch

    class _Env:
        """Minimal stand-in: _backflip_state only needs these three."""

        num_envs = 2
        device = "cpu"
        step_dt = 0.02

        def __init__(self):
            self.episode_length_buf = torch.zeros(2, dtype=torch.long)

    env = _Env()
    microduck_mdp._backflip_state(env)
    assert Z0_RANGE[0] <= float(env._backflip_z0[0]) <= Z0_RANGE[1]
    assert VZ_RANGE[0] <= float(env._backflip_vz[0]) <= VZ_RANGE[1]
    assert W0_RANGE[0] <= float(env._backflip_w0[0]) <= W0_RANGE[1]
    assert LAUNCH_RANGE[0] <= float(env._backflip_t_launch[0]) <= LAUNCH_RANGE[1]
    # a HOLD phase must actually exist at t=0, or the plate leaves immediately
    assert float(env._backflip_t_hold[0]) > 0.0
    env.episode_length_buf[:] = 0
    assert int(microduck_mdp.backflip_phase(env)[0]) == (
        microduck_mdp.BACKFLIP_PHASE_HOLD
    )
    assert torch.all(env._backflip_accum == 0.0)


def test_the_named_tuck_matches_the_indexed_one(cfg):
    # _TUCK_BY_NAME feeds the entity init_state (mjlab matches joint_pos by
    # regex) while TUCK_OVERRIDES feeds the reset event and the reward (servo
    # index). A divergence would spawn the pre-reset pose differently from
    # every reset pose, silently.
    del cfg
    from mjlab_microduck.tasks.microduck_backflip_env_cfg import _TUCK_BY_NAME

    index_to_name = {
        2: "left_hip_pitch", 3: "left_knee", 4: "left_ankle",
        5: "neck_pitch", 6: "head_pitch",
        11: "right_hip_pitch", 12: "right_knee", 13: "right_ankle",
    }
    assert set(index_to_name) == set(TUCK_OVERRIDES)
    by_name = {k.strip("^$"): v for k, v in _TUCK_BY_NAME.items()}
    assert set(by_name) == set(index_to_name.values())
    for idx, name in index_to_name.items():
        assert by_name[name] == pytest.approx(TUCK_OVERRIDES[idx])


# --- The spawn must sit ON the plate, not jammed INTO it. -------------------
#
# THE TEST WHOSE ABSENCE LET IT THROUGH. During the tucked-hold experiment the
# spawn wrote a trunk height whose FEET were tunnelled UNDER the launcher plate
# — the tuck kneels on its shins, so its feet are not its lowest point and a
# spawn placed by trunk height put them through the 2 cm slab. Six to nine
# simultaneously penetrating contacts, 22 mm deep, and every launch-envelope
# table on this branch was measured from it before anyone noticed. Nothing
# checked. These build the exact spawn state in CPU MuJoCo and measure it.
#
# Standing makes this structurally safe rather than merely fixed: the feet ARE
# the lowest geoms, so placing the trunk at STAND_Z above the plate top puts
# the soles on the surface by construction and no spawn can tunnel.

_SCENE = "src/mjlab_microduck/robot/microduck/scene_backflip.xml"

# A robot resting under load compresses MuJoCo's contact constraint, so the bar
# is not literally zero. This is the loaded-compression tolerance; the tucked
# spawn was 22 mm, i.e. 4x outside it.
_RESTING_PENETRATION_M = 0.006


def _spawn_state(z0=None):
    """Build the env's spawn state in plain MuJoCo.

    Mirrors reset_backflip_robot_on_plate: trunk at
    z0 + PLATE_HALF_THICKNESS + STAND_Z, HOME joints (what reset_robot_joints
    leaves), level, plate prescribed at z0.
    """
    if z0 is None:
        z0 = 0.5 * (Z0_RANGE[0] + Z0_RANGE[1])
    model = mujoco.MjModel.from_xml_path(_SCENE)
    data = mujoco.MjData(model)
    data.qpos[0:3] = [0.0, 0.0, z0 + PLATE_HALF_THICKNESS + STAND_Z]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    for i, angle in enumerate(_HOME_POSE):
        data.qpos[7 + i] = angle
    pj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "plate_free")
    qa = model.jnt_qposadr[pj]
    data.qpos[qa : qa + 3] = [0.0, 0.0, z0]
    data.qpos[qa + 3 : qa + 7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)
    return model, data


# HOME_FRAME in servo order; the spawn leaves the joints to reset_robot_joints,
# which puts them here.
_HOME_POSE = (
    0.0, -0.0873, -0.4579, -0.0049, 0.4530,
    0.3491, 0.3491, 0.0, 0.0,
    0.0, 0.0873, 0.4579, 0.0049, -0.4530,
)


def _plate_penetration(model, data):
    """(deepest penetration as a POSITIVE depth, number of penetrating pairs)."""
    plate_g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "plate_geom")
    depths = [
        -float(data.contact.dist[i])
        for i in range(data.ncon)
        if plate_g in (int(data.contact.geom1[i]), int(data.contact.geom2[i]))
        and float(data.contact.dist[i]) < 0.0
    ]
    return (max(depths) if depths else 0.0), len(depths)


def test_the_spawn_is_not_jammed_into_the_plate():
    model, data = _spawn_state()
    depth, n = _plate_penetration(model, data)
    assert depth <= _RESTING_PENETRATION_M, (
        f"spawn penetrates the plate by {depth * 1000:.1f} mm on {n} contacts. "
        "Either the spawn height and the hold pose disagree, or the plate "
        "geometry changed -- re-measure, do not raise this bound."
    )


def test_the_spawn_is_clean_at_every_sampled_launch_height():
    # z0 shifts the plate AND the robot together, so the penetration must be
    # z0-invariant; a z0-dependent spawn means the two heights have drifted.
    depths = []
    for z0 in (Z0_RANGE[0], 0.5 * sum(Z0_RANGE), Z0_RANGE[1]):
        depth, _ = _plate_penetration(*_spawn_state(z0=z0))
        depths.append(depth)
        assert depth <= _RESTING_PENETRATION_M
    assert max(depths) - min(depths) < 1e-6


def test_no_robot_geom_is_below_the_plate_top_inside_its_footprint():
    # The exact geometry the user reported and the tucked hold produced: a geom
    # under the plate's top surface while still inside its footprint has passed
    # THROUGH the slab. Standing has no such geom, by construction.
    model, data = _spawn_state()
    plate_g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "plate_geom")
    plate_b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "plate")
    half_x, half_y, half_z = model.geom_size[plate_g]
    z0 = 0.5 * (Z0_RANGE[0] + Z0_RANGE[1])
    plate_top = z0 + half_z
    offenders = []
    for g in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"g{g}"
        if model.geom_bodyid[g] == plate_b or name == "floor":
            continue
        if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
            continue
        x, y, z = (float(v) for v in data.geom_xpos[g])
        if abs(x) < half_x and abs(y) < half_y and z < plate_top:
            offenders.append((name, x, y, z))
    assert not offenders, (
        "geoms below the plate top and inside its footprint (i.e. through the "
        f"slab): {offenders}"
    )


def test_the_feet_are_the_lowest_geoms_so_the_spawn_cannot_tunnel():
    # WHY standing is structurally safe and the tuck was not. The tuck kneels
    # on its shins, so placing it by trunk height put the FEET through the
    # plate. Standing's lowest geoms are the feet, so "trunk at STAND_Z above
    # the surface" puts the soles ON it.
    model, data = _spawn_state()
    plate_b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "plate")
    lowest = []
    for g in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"g{g}"
        if model.geom_bodyid[g] == plate_b or name == "floor":
            continue
        if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
            continue
        lowest.append((float(data.geom_xpos[g][2]) - float(model.geom_rbound[g]),
                       name))
    lowest.sort()
    assert "foot" in lowest[0][1], (
        f"the lowest collision geom is {lowest[0][1]}, not a foot -- a spawn "
        "placed by trunk height can then tunnel whatever hangs below it"
    )


# --- The play cfg must show the env the policy was TRAINED on. --------------


@pytest.fixture(scope="module")
def play_cfg():
    return make_microduck_backflip_env_cfg(play=True)


def test_play_cfg_carries_every_curriculum_to_its_last_stage(play_cfg, cfg):
    # Not just the hold: every range or weight a curriculum moves over training
    # must be at its trained value in play, or the viewer is showing a
    # different env from the one the checkpoint knows.
    for name, term in cfg.curriculum.items():
        params = term.params
        if "param_stages" in params:
            last = params["param_stages"][-1]["params"]
            got = play_cfg.events[params["event_name"]].params
            for key, value in last.items():
                assert got[key] == value, f"{name}: {key}"
        elif "weight_stages" in params:
            last = params["weight_stages"][-1]["weight"]
            assert play_cfg.rewards[params["reward_name"]].weight == last, name
        elif "range_stages" in params:
            last = params["range_stages"][-1]["range"]
            assert play_cfg.events[params["event_name"]].params["ranges"] == (
                -last, last
            ), name
        else:
            raise AssertionError(
                f"{name} has an unrecognised stage list; _apply_final_curriculum "
                "must learn it or play silently runs at stage 0"
            )


def test_play_cfg_drops_the_curriculum_terms(play_cfg):
    # Pre-applying is not enough: the curriculum manager runs every step and
    # would write stage 0 straight back over the final values.
    assert play_cfg.curriculum == {}

def test_there_is_no_hold_curriculum(cfg):
    # The hold is sampled uniformly 1-5 s from step 0. The ramp it replaced
    # existed only to keep early episodes from standing still, it bought little
    # (the first run reached landing +1.72 by iteration 279), and it made the
    # long hold invisible in `play`, which evaluates every curriculum at stage
    # zero.
    assert "backflip_hold_range" not in cfg.curriculum
    assert cfg.events["backflip_launch_params"].params["hold_range"] == HOLD_RANGE
    assert HOLD_RANGE == (1.0, 5.0)


def test_the_episode_fits_the_worst_case_hold(cfg):
    # 5.0 s hold + 0.16 s flick + the MEASURED 0.88 s worst-case flight leaves
    # 1.46 s to settle, which is what LANDING_WINDOW_S is sized against.
    assert cfg.episode_length_s == pytest.approx(7.5)
    settle = cfg.episode_length_s - HOLD_RANGE[1] - 0.16 - 0.88
    assert settle >= LANDING_WINDOW_S
    assert settle - LANDING_WINDOW_S < 0.2, (
        "the payout window should use nearly all of the worst case's "
        "affordance; a large gap is wasted annuity"
    )


def test_the_landing_annuity_is_hold_independent(cfg):
    # THE POINT OF THE FIXED WINDOW. Paying for "all the time remaining after
    # touchdown" gave an identical backflip 21.8 in episode-sum at a 1 s hold
    # against 5.84 at a 5 s hold -- a ~4x swing in the MAIN attractor decided
    # by a draw the policy neither controls nor observes.
    assert cfg.rewards["landing"].params["window_s"] == LANDING_WINDOW_S
    landing_w = cfg.rewards["landing"].weight
    masses = [
        landing_w * LANDING_WINDOW_S           # same at every hold, by design
        for _ in (HOLD_RANGE[0], HOLD_RANGE[1])
    ]
    assert masses[0] == masses[-1]
    # ... and it must still fit the shortest available settle window
    assert LANDING_WINDOW_S <= (
        cfg.episode_length_s - HOLD_RANGE[1] - 0.16 - 0.88
    )


def test_the_stance_never_outweighs_the_landing_at_any_hold(cfg):
    # The ceiling on the stance weight comes from the LONGEST hold, where the
    # stance is largest and the (now fixed) annuity is unchanged.
    stance_w = cfg.rewards["ready_stance"].weight
    landing_mass = cfg.rewards["landing"].weight * LANDING_WINDOW_S
    flip_w = cfg.rewards["flip_progress"].weight
    for hold in (HOLD_RANGE[0], 0.5 * sum(HOLD_RANGE), HOLD_RANGE[1]):
        stance_mass = 0.975 * stance_w * hold
        assert landing_mass >= stance_mass, (
            f"at a {hold}s hold the stance ({stance_mass:.2f}) outweighs the "
            f"landing annuity ({landing_mass:.2f})"
        )
        # never flipping must never beat flipping and landing
        assert stance_mass < flip_w + landing_mass
    # and the hold must be long enough that the term is worth defending at all
    assert 0.975 * stance_w * HOLD_RANGE[0] >= 1.0


def test_the_stance_prices_a_LEAN_not_just_a_topple(cfg):
    # The user saw the robot launch leaning ~35 deg back and the term charged
    # nothing, because the gate was full-credit below 40 deg -- a width sized
    # for the short-lived tucked hold, whose own equilibrium is pitched 14 deg.
    params = cfg.rewards["ready_stance"].params
    assert params["tilt_full_deg"] <= 15.0, (
        "a lean must cost something; standing's own drift is only 3.5-7.3 deg"
    )
    # ... while every measured flop basin (80-126 deg) is still hard-zeroed
    assert params["tilt_zero_deg"] <= 70.0
    assert params["tilt_full_deg"] < params["tilt_zero_deg"]


def test_the_lean_penalty_has_the_shape_the_docstring_claims(cfg):
    # Measured through the real gate, at the cfg's own numbers.
    import torch

    p = cfg.rewards["ready_stance"].params

    def gate(tilt_deg):
        half = math.radians(tilt_deg) * 0.5
        quat = torch.tensor([[math.cos(half), 0.0, math.sin(half), 0.0]])
        return float(
            microduck_mdp._trunk_tilt_smoothstep(
                quat, p["tilt_full_deg"], p["tilt_zero_deg"]
            )[0]
        )

    assert gate(0.0) == 1.0
    assert gate(7.3) == 1.0            # the measured open-loop drift is free
    assert gate(20.0) < 0.85           # a 20 deg lean costs ~20%
    assert gate(35.0) < 0.25           # a 35 deg lean costs ~80%
    assert gate(45.0) == 0.0
    assert gate(90.0) == 0.0


# --- The NaN path that killed run 2026-09-08_16-26-30_backflip. -------------
#
# rsl_rl's check_nan found NaN in the CRITIC group while
# Episode_Termination/nan_state read exactly 0.0000. mjlab computes
# observations AFTER _reset_idx (manager_based_rl_env.step), so anything
# `robot_state_is_nan` checks would have reset the env and returned clean obs.
# The NaN therefore arrived through a quantity it does NOT check -- and with no
# `sensor_names` it did not check contact FORCES, which is what the critic's
# `foot_contact_forces` obs is built from.


def test_nan_guard_watches_every_contact_sensor_in_the_scene(cfg):
    # THE OMISSION. Contact forces blow up to inf/NaN a step before the
    # integrated state does; `robot_state_is_nan` only sees them if it is told
    # which sensors to read. Every sensor the scene defines must be listed, so
    # adding a sensor cannot silently reopen the hole.
    named = set(cfg.terminations["nan_state"].params.get("sensor_names", ()))
    in_scene = {s.name for s in cfg.scene.sensors}
    assert named == in_scene, (
        f"nan_state watches {sorted(named)} but the scene has "
        f"{sorted(in_scene)}; contact forces on the unwatched ones reach the "
        "critic group unchecked"
    )


def test_every_sensor_derived_critic_term_is_nan_safe(cfg):
    # The critic group is the one obs path the NaN guard cannot protect in
    # time. mjlab's own sensor observations are NOT sanitized, so any term
    # sourced from that module must be swapped for a `_safe` variant -- which
    # this env failed to do, because it builds on mjlab's base velocity
    # template rather than the microduck one that already does the swap.
    unsafe_module = "mjlab.tasks.velocity.mdp.observations"
    allowed_raw = {
        # boolean/found flags: 0-1 by construction, cannot be non-finite
        "foot_contact",
    }
    offenders = []
    for name, term in cfg.observations["critic"].terms.items():
        if term.func.__module__ == unsafe_module and name not in allowed_raw:
            offenders.append((name, term.func.__name__))
    assert not offenders, (
        f"unsanitized sensor-derived critic terms: {offenders}. Swap them for "
        "the microduck _safe variants (see the cfg's critic block)."
    )


def test_the_backflip_critic_terms_are_the_sanitizing_ones(cfg):
    critic = cfg.observations["critic"].terms
    assert critic["foot_contact_forces"].func is microduck_mdp.foot_contact_forces_safe
    assert critic["foot_air_time"].func is microduck_mdp.foot_air_time_safe
    assert critic["plate_position"].func is microduck_mdp.backflip_plate_pos_obs
    assert critic["plate_velocity"].func is microduck_mdp.backflip_plate_vel_obs
    assert critic["plate_phase"].func is microduck_mdp.backflip_phase_obs


def test_nconmax_clears_the_measured_worst_case_contact_count(cfg):
    # AGENTS.md: contact overflow presents as sudden NaN, so this was the first
    # suspect for the 2026-09-08 crash. MEASURED on CPU MuJoCo with the
    # full-collision robot and the plate in the scene: peak 16 standing on the
    # plate, 3-10 sprawled on the floor for 5 s in every orientation a flip can
    # end in, plus 4 persistent plate-floor contacts once the slab rests on the
    # ground -- ~20 worst case. The hypothesis was RULED OUT; this pins the
    # margin so a future collision or prop change cannot quietly eat it.
    measured_worst_case = 20
    assert cfg.sim.nconmax >= 2 * measured_worst_case, (
        f"nconmax={cfg.sim.nconmax} leaves less than 2x the measured worst "
        f"case ({measured_worst_case}); re-measure before lowering it"
    )

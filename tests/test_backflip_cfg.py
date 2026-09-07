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
    HOLD_RANGE,
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
    # 0.07 is the geometric floor for the kneeling hold (its feet hang ~8 cm
    # below the surface it rests on); the retune pushed the top down from 0.20.
    assert lo >= 0.07 and hi <= 0.09


def test_launch_envelope_is_the_measured_box_not_the_placeholders(cfg):
    # The WHOLE-BOX-verified tucked-spawn envelope, retuned for ONE clean turn
    # from the lowest plate the pose allows (docs "Lower and gentler"): every
    # corner and midpoint of these four ranges closes >= 360 deg backward at
    # <= 2.6 m/s. Measured across the box: 372.5-457.3 deg, worst landing
    # 2.15 m/s. The bounds each mark a measured failure just outside them:
    #   z0 < 0.07     -> the tuck's dangling feet reach the ground (41 deg tilt)
    #   t_launch 0.14 -> under-rotates at this vz (339.8 deg at the z0=0.07 corner)
    #   t_launch < 0.12 -> landings up to 3.97 m/s (a short flick is violent)
    #   w0 > 25       -> 357 deg at the z0=0.07 corner
    p = cfg.events["backflip_launch_params"].params
    assert p["vz_range"] == VZ_RANGE == (1.90, 2.00)
    assert p["w0_range"] == W0_RANGE == (23.0, 24.0)
    assert p["launch_range"] == LAUNCH_RANGE == (0.12, 0.13)
    assert p["z0_range"] == Z0_RANGE == (0.07, 0.09)
    # ... and the mdp defaults say the same thing, so an env built without the
    # cfg (or a copy-paste into a new task) does not inherit a stale box.
    import inspect

    defaults = inspect.signature(
        microduck_mdp.reset_backflip_launch_params
    ).parameters
    assert defaults["vz_range"].default == VZ_RANGE
    assert defaults["w0_range"].default == W0_RANGE
    assert defaults["launch_range"].default == LAUNCH_RANGE
    assert defaults["z0_range"].default == Z0_RANGE


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


def test_the_spawn_pose_and_the_stance_target_are_the_same_numbers(cfg):
    # THE invariant between the spawn and the reward that scores it. The spawn
    # folds the robot to tuck_overrides x tuck_factor and puts the trunk at
    # z0 + plate_half_thickness + tuck_z; backflip_ready_stance scores the same
    # joint target and the same height. Edit either alone and the robot spawns
    # in a pose its own hold reward calls wrong — with every other test in this
    # file still green.
    spawn = cfg.events["backflip_spawn"].params
    stance = cfg.rewards["ready_stance"].params

    assert spawn["tuck_overrides"] == stance["tuck_overrides"] == TUCK_OVERRIDES
    assert spawn["tuck_factor"] == stance["tuck_factor"] == TUCK_FACTOR
    assert spawn["tuck_z"] + spawn["plate_half_thickness"] == pytest.approx(
        stance["tuck_z"]
    )
    # and each half is the constant this module exports, not a stray number
    assert spawn["tuck_z"] == pytest.approx(TUCK_Z)
    assert spawn["plate_half_thickness"] == pytest.approx(PLATE_HALF_THICKNESS)


def test_the_hold_posture_is_tucked_not_standing(cfg):
    # The defect this whole branch turned on: the spawn, the reward that pays
    # for the hold, and the posture the envelope was measured from must be ONE
    # posture. STAND_Z belongs to the LANDING only.
    spawn = cfg.events["backflip_spawn"].params
    stance = cfg.rewards["ready_stance"].params
    assert "stand_z" not in spawn and "stand_z" not in stance
    assert cfg.rewards["ready_stance"].func is microduck_mdp.backflip_ready_stance
    # A tucked trunk sits far below a standing one — if these ever converge,
    # someone has carried STAND_Z into the tuck.
    assert TUCK_Z < STAND_Z - 0.05
    assert cfg.rewards["landing"].params["stand_z"] == pytest.approx(STAND_Z)


def test_the_tuck_map_matches_roulades(cfg):
    # TUCK_OVERRIDES is duplicated from the roulade env rather than imported,
    # so that each task can retune its own tuck. Pin them equal anyway: a
    # silent divergence would make the roulade lesson stop applying here.
    from mjlab_microduck.tasks.microduck_roulade_env_cfg import (
        TUCK_OVERRIDES as ROULADE_TUCK,
    )

    assert TUCK_OVERRIDES == ROULADE_TUCK


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
    # backflip_ready_stance measures trunk z against (origin + z0 + tuck_z),
    # but the robot rests on the plate TOP, one half-thickness above z0.
    assert cfg.rewards["ready_stance"].params["tuck_z"] == pytest.approx(
        TUCK_Z + PLATE_HALF_THICKNESS
    )
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


def test_launch_dr_widens_over_training(cfg):
    # Only the HOLD window widens now; the z0 tail was removed (see
    # test_there_is_no_z0_dr_tail).
    hold = cfg.curriculum["backflip_hold_range"].params["param_stages"]
    assert hold[0]["params"]["hold_range"] == HOLD_RANGE
    assert hold[-1]["params"]["hold_range"][1] > HOLD_RANGE[1]


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

    # The HOLD is NOT separable by height any more, and that is deliberate:
    # with the plate at z0 = 0.07-0.09 the tucked hold trunk is 0.109-0.129 m,
    # straddling STAND_Z (0.115). This assertion documents the overlap so the
    # next reader does not "fix" the ceiling into a number that cannot exist.
    hold_lo = Z0_RANGE[0] + PLATE_HALF_THICKNESS + TUCK_Z
    hold_hi = Z0_RANGE[1] + PLATE_HALF_THICKNESS + TUCK_Z
    assert hold_lo < STAND_Z < hold_hi, (
        "the tucked hold no longer straddles standing height - re-derive the "
        "arrival_damping ceiling instead of accepting HOLD-phase damping"
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
    assert Z0_RANGE == (0.07, 0.09)


def test_the_launch_is_retuned_for_one_clean_turn(cfg):
    # The user watched the env and reported it "launched far too hard and too
    # far". Measured across the new box: rotation 372.5-457.3 deg (was
    # 393.6-475.7), apex 0.29-0.40 m (was 0.52-0.63), worst landing 2.15 m/s
    # (was 2.53). Over-rotation is a defect to minimise now, not headroom.
    p = cfg.events["backflip_launch_params"].params
    assert p["vz_range"][1] <= 2.00        # was 2.10
    assert p["z0_range"][1] <= 0.09        # was 0.20, with a tail to 0.225
    assert p["launch_range"] == (0.12, 0.13)


# --- The first-episode state, and the tuck-by-name init pose. ---------------


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
    # the robot's default trunk sits on the plate top in the tuck, not on the
    # floor at standing height
    assert robot.init_state.pos[2] == pytest.approx(
        z0_mid + PLATE_HALF_THICKNESS + TUCK_Z
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

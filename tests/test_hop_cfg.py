"""Config-level assertions for the hop variant transform."""

import math

import pytest

from mjlab_microduck.robot.sprung_foot import H_ADD, PAD_MASS
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.hop import (
    AIRBORNE_WEIGHT,
    BODY_HEIGHT_WEIGHT,
    BODY_WEIGHT_N,
    HOP_ARM_SUFFIX,
    HOP_COM_HEIGHT_MAX,
    HOP_HEIGHT_GAIN,
    HOP_HEIGHT_STD,
    HOP_MAX_LAUNCH_VEL,
    HOP_PERIOD,
    LOAD_FORCE_MAX_RATIO,
    LOAD_FORCE_WEIGHT,
    MIN_RISE,
    SENSOR_NAME,
    UNLOADED_RIGID_HEIGHT,
    UPWARD_VELOCITY_WEIGHT,
    make_hop_variant,
)
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
)


@pytest.fixture
def hop_cfg():
    return make_hop_variant(make_microduck_velocity_env_cfg())


def test_command_is_the_cyclic_phase_command(hop_cfg):
    term = hop_cfg.commands["twist"]
    assert term.class_type is microduck_mdp.GroundPickPhaseCommand
    assert term.period == pytest.approx(HOP_PERIOD)


def test_height_target_is_a_rise_and_carries_no_standing_height_datum(hop_cfg):
    """The target is the GAIN itself, with no standing height added to it.

    This replaces two tests that asserted the old absolute target was shifted by
    h_add. That shift existed only because the target was an absolute base
    height, and it was the source of two datum errors (the spawn height, then
    the settled-vs-unloaded sag). Measured as rise above takeoff, there is
    nothing to shift: the sprung robot stands 30 mm taller and its takeoff
    height is 30 mm higher, so the rise it must achieve is identical.
    """
    params = hop_cfg.rewards["hop_body_height"].params
    assert params["target_rise"] == pytest.approx(HOP_HEIGHT_GAIN)
    assert "target_height" not in params, "the absolute-height target is gone"
    # The datum is genuinely absent, not merely renamed: the target must not
    # contain a standing height on any arm.
    assert params["target_rise"] < UNLOADED_RIGID_HEIGHT


def test_make_hop_variant_no_longer_takes_h_add():
    """The hop rewards are invariant to how tall the boot makes the robot, so
    the transform has no use for h_add and must not silently accept one -- a
    caller passing it would reasonably expect it to do something."""
    import inspect

    assert "h_add" not in inspect.signature(make_hop_variant).parameters


def test_all_three_hop_rewards_registered_with_positive_weight(hop_cfg):
    for name, func in (
        ("hop_both_feet_airborne", microduck_mdp.hop_both_feet_airborne),
        ("hop_upward_velocity", microduck_mdp.hop_upward_velocity),
        ("hop_body_height", microduck_mdp.hop_body_height),
    ):
        term = hop_cfg.rewards[name]
        assert term.func is func
        assert term.weight > 0.0


def test_energy_monitor_registered_with_non_zero_weight(hop_cfg):
    # RewardManager.compute skips weight==0.0 terms before calling them.
    term = hop_cfg.rewards["hop_energy_monitor"]
    assert term.func is microduck_mdp.hop_energy_monitor
    assert term.weight != 0.0


def test_energy_monitor_stiffness_threads_through_the_variant():
    """make_hop_variant's default (3900.0) is only correct for the k3900 arm.
    Task 4 also registers a k2500 arm; without this parameter threading through,
    hop_energy_monitor would report that arm's stored spring energy 56% high."""
    cfg = make_hop_variant(make_microduck_velocity_env_cfg(), stiffness=2500.0)
    assert cfg.rewards["hop_energy_monitor"].params["stiffness"] == 2500.0


def test_command_construction_preserves_base_fields():
    """The command-construction step filters vars(command) down to the fields
    GroundPickPhaseCommandCfg declares, then rebuilds it. A regression that
    dropped or mangled behaviour-carrying fields (ranges, rel_standing_envs,
    viz) would not show up as a TypeError -- it would silently ship a policy
    that ignores its velocity-sampling ranges. Assert the carry-over directly."""
    rigid = make_microduck_velocity_env_cfg()
    original = rigid.commands["twist"]
    hop_cfg = make_hop_variant(rigid)
    rebuilt = hop_cfg.commands["twist"]

    assert rebuilt.ranges == original.ranges
    assert rebuilt.rel_standing_envs == original.rel_standing_envs
    assert rebuilt.rel_heading_envs == original.rel_heading_envs
    assert rebuilt.viz.z_offset == original.viz.z_offset
    assert rebuilt.entity_name == original.entity_name


def test_forward_locomotion_rewards_removed(hop_cfg):
    """This is a hop in place. Leaving velocity tracking in would reward the
    policy for running away instead of hopping, and the phase command has
    overwritten the twist command those terms read."""
    for name in ("track_linear_velocity", "track_angular_velocity"):
        assert name not in hop_cfg.rewards


def test_walking_air_time_reward_removed(hop_cfg):
    """`air_time` (mjlab's feet_air_time, weight 5.0) pays a PER-FOOT indicator
    over 0.10-0.25 s of flight, i.e. it pays continuously for alternating
    single-foot stepping. Integrated over a cycle, marching in place earns
    ~3.0/step against ~1.0/step for a 1.0 s hop, and at most ~1.1/step is
    available from all three hop terms combined — so leaving it in makes a bob in
    place strictly outscore hopping, on all three arms equally, and the campaign
    would conclude "compliance does not help" from three runs that never hopped.

    Its command gate (||cmd[:2]|| + |cmd[2]| > 0.01) is also permanently latched
    on: the phase command's magnitude is identically 1.0."""
    assert "air_time" not in hop_cfg.rewards


def test_air_time_is_present_in_the_walking_env_this_is_removed_from():
    """Guard the guard: if mjlab renamed or dropped the term upstream, the
    assertion above would pass vacuously and stop protecting anything."""
    assert "air_time" in make_microduck_velocity_env_cfg().rewards


def test_hop_body_height_is_gated_on_the_contact_sensor(hop_cfg):
    """The height reward must be gated on both feet airborne, or it pays ~0.57 of
    peak for straightening the legs while planted. The gate reads the same sensor
    as hop_both_feet_airborne, so the two terms cannot disagree about what a hop
    is; make_hop_variant threads it explicitly rather than leaning on the default."""
    height = hop_cfg.rewards["hop_body_height"]
    airborne = hop_cfg.rewards["hop_both_feet_airborne"]
    assert height.params["sensor_name"] == SENSOR_NAME
    assert height.params["sensor_name"] == airborne.params["sensor_name"]


def test_action_rate_curriculum_untouched(hop_cfg):
    rigid = make_microduck_velocity_env_cfg()
    expected = [dict(s) for s in rigid.curriculum["action_rate_weight"].params["weight_stages"]]
    actual = [dict(s) for s in hop_cfg.curriculum["action_rate_weight"].params["weight_stages"]]
    assert actual == expected


def test_hop_period_is_above_the_spring_mass_period():
    """At k=3900 and 0.877 kg the spring-mass period is ~94 ms. A hop period at
    or below that would drive the spring faster than it can cycle."""
    assert HOP_PERIOD > 0.094


def test_hop_arms_are_locked_plus_the_stiffness_set():
    from mjlab_microduck.robot.sprung_foot import K_MEASURED
    from mjlab_microduck.tasks.hop import HOP_ARMS

    labels = [a[0] for a in HOP_ARMS]
    assert "locked" in labels, "the locked arm is the geometric control"
    stiffnesses = {a[1] for a in HOP_ARMS if a[0] != "locked"}
    # k2500/k3900 were a BRACKET (they gave 23 and 27 mm of rise); K_MEASURED is
    # the real prototype, measured on the gripper bench, and is the arm to train.
    assert stiffnesses == {2500.0, K_MEASURED, 3900.0}
    assert 2500.0 < K_MEASURED < 3900.0, "the bracket must straddle the measurement"
    # The Locked control must wear the SAME geometry as the sprung arms, so it
    # carries K_MEASURED as a nominal stiffness with travel 0 -- the spring joint
    # is then omitted entirely and the value is unused.
    locked = next(a for a in HOP_ARMS if a[0] == "locked")
    assert locked[2] == 0.0
    for label, _k, travel, pad in HOP_ARMS:
        # DELTA of fitting a spring boot: 69 g boot - 18 g standard pad it
        # replaces. The common 16.5 g interface is in both configs and cancels.
        assert pad == pytest.approx(0.051), "delta mass, measured 2026-09-03"
        if label == "locked":
            assert travel == 0.0
        else:
            assert travel > 0.0


def test_hop_task_ids_registered():
    import mjlab_microduck.tasks  # noqa: F401
    from mjlab.tasks.registry import list_tasks

    tasks = list_tasks()
    for tid in (
        "Mjlab-Hop-Flat-Sprung-Locked-MicroDuck",
        "Mjlab-Hop-Flat-Sprung-K2500-MicroDuck",
        "Mjlab-Hop-Flat-Sprung-K3900-MicroDuck",
    ):
        assert tid in tasks, f"{tid} not registered"


def test_hop_arms_have_distinct_wandb_identities():
    import mjlab_microduck.tasks  # noqa: F401
    from mjlab.tasks.registry import load_rl_cfg

    names = {
        load_rl_cfg(f"Mjlab-Hop-Flat-Sprung-{s}-MicroDuck").run_name
        for s in ("Locked", "K2500", "K3900")
    }
    assert len(names) == 3


def test_registered_hop_cfgs_carry_the_rise_target():
    """End-to-end: the registered task, not just the transform in isolation."""
    import mjlab_microduck.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg

    cfg = load_env_cfg("Mjlab-Hop-Flat-Sprung-K3900-MicroDuck")
    assert cfg.rewards["hop_body_height"].params["target_rise"] == pytest.approx(
        HOP_HEIGHT_GAIN
    )


def test_unloaded_rigid_height_is_pinned_to_the_compiled_locked_arm_geometry():
    """Pin UNLOADED_RIGID_HEIGHT to the Locked arm's actual kinematics.

    NOT a reward datum any more -- both hop height rewards measure RISE ABOVE
    TAKEOFF, so no standing height enters them. It survives because the CoM band
    still does care how tall the robot stands: `HOP_COM_HEIGHT_MAX` has to clear
    the absolute height a successful hop reaches, and that is computed from this
    constant in `test_com_band_ceiling_is_above_the_hop_apex`.

    The companion RIGID_STAND_HEIGHT (settled, 0.1095) was DELETED along with the
    absolute-height rewards: with no reward reading a standing height, nothing
    referenced it, and a constant pinned only by the test that pins it is a
    tautology. Its measurement is preserved in `hop.py`'s header comment.

    Trips if H_ADD, ANKLE_TO_SOLE, the pad box or any leg link changes — at which
    point re-run `.superpowers/sdd/2026-08-24-sprung-hop/measure_stand_height.py`.
    """
    import re

    import mujoco
    import numpy as np

    import mjlab_microduck.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg
    from mjlab_microduck.robot.microduck_constants import HOME_FRAME

    robot = load_env_cfg("Mjlab-Hop-Flat-Sprung-Locked-MicroDuck").scene.entities["robot"]
    m = robot.spec_fn().compile()
    d = mujoco.MjData(m)
    for i in range(m.njnt):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
        if nm is None or m.jnt_type[i] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        for pat, val in HOME_FRAME.joint_pos.items():
            if re.search(pat.strip("^$").replace(".*", ""), nm):
                d.qpos[m.jnt_qposadr[i]] = val
                break
    d.qpos[:3] = 0.0
    d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(m, d)

    # Lowest corner of either contact pad, with the base frame at z = 0.
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    lowest = min(
        (d.geom_xpos[g] + (corners * m.geom_size[g]) @ d.geom_xmat[g].reshape(3, 3).T)[:, 2].min()
        for g in (
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{s}_foot_collision")
            for s in ("left", "right")
        )
    )
    kinematic_rigid = -lowest - H_ADD
    assert kinematic_rigid == pytest.approx(0.12114, abs=0.002)
    # UNLOADED_RIGID_HEIGHT *is* this sag-free kinematic height, so pin it to the
    # geometry rather than to a copied literal.
    assert UNLOADED_RIGID_HEIGHT == pytest.approx(kinematic_rigid, abs=0.002)


def test_registered_hop_cfgs_carry_their_own_arm_stiffness():
    """End-to-end, through the registered task -- not the transform in
    isolation. The registration loop must thread stiffness=_k into
    make_hop_variant for every arm, not only into make_sprung_variant.
    make_hop_variant's default stiffness (3900.0) is only correct for the
    k3900/locked arms -- if the k2500 arm's registration omitted it,
    hop_energy_monitor would report that arm's stored spring energy 56% high,
    and the spec requires reading the spring instruments
    (hop_spring_energy_*, spring_bottomed_fraction) BEFORE any hop-height
    number, so a wrong energy metric would corrupt the primary result this
    whole phase exists to produce. Loads all three registered tasks so a
    regression in any one arm's call site is caught."""
    import mjlab_microduck.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg
    from mjlab_microduck.tasks.hop import HOP_ARM_SUFFIX, HOP_ARMS

    for label, k, _travel, _pad in HOP_ARMS:
        tid = f"Mjlab-Hop-Flat-Sprung-{HOP_ARM_SUFFIX[label]}-MicroDuck"
        cfg = load_env_cfg(tid)
        assert cfg.rewards["hop_energy_monitor"].params["stiffness"] == k, tid


# ---------------------------------------------------------------------------
# The reward ceiling. Four independent mechanisms capped the height reward at
# roughly 15-20 mm of gain while the drop-rig evidence spans 5 mm (Locked) to
# 33 mm (k3900), so all three arms would have sat at the ceiling and the
# arm-to-arm comparison -- which IS the experiment -- would have returned an
# uninformative null. The tests below pin all four, plus the discrimination
# property that motivates them.
# ---------------------------------------------------------------------------


def _hop_task_id(label: str) -> str:
    return f"Mjlab-Hop-Flat-Sprung-{HOP_ARM_SUFFIX[label]}-MicroDuck"


def _registered(label: str):
    import mjlab_microduck.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg

    return load_env_cfg(_hop_task_id(label))


def test_registered_arms_carry_the_widened_rise_gaussian():
    """End-to-end through the registered tasks, not the transform in isolation.

    The old (gain 0.015, std 0.008) pair put the entire 5-33 mm evidence band at
    ~0.000 reward. Assert the registered params, and assert the peak still sits
    ABOVE the ~27 mm energetic ceiling estimated for k=3900 so the whole band
    stays on the Gaussian's RISING limb instead of straddling its peak.

    Two whole tests were DELETED here rather than ported, because measuring rise
    made them tautologies: `test_height_target_references_the_unloaded_not_the_
    settled_height` existed only to pick between two standing-height datums, and
    the datum-revert guard below it defended a subtraction that no longer
    happens. There is no datum left to get wrong -- which was the point.
    """
    for label in HOP_ARM_SUFFIX:
        cfg = _registered(label)
        params = cfg.rewards["hop_body_height"].params
        tid = _hop_task_id(label)
        assert params["target_rise"] == pytest.approx(HOP_HEIGHT_GAIN), tid
        assert params["std"] == pytest.approx(HOP_HEIGHT_STD), tid
        # The peak must sit above the ~27 mm energetic ceiling estimated for
        # k=3900, so the 5-33 mm evidence band stays on the rising limb.
        assert params["target_rise"] > 0.033, tid
        # And it must remain a RISE, not an absolute height: anything at or above
        # the robot's own standing height is a datum that crept back in.
        assert params["target_rise"] < UNLOADED_RIGID_HEIGHT, tid


def test_upward_velocity_does_not_saturate_below_the_height_target():
    """`hop_upward_velocity` clamps vel_z/max_vel to [0, 1]. A ballistic launch at
    v rises v**2/(2*g), so max_vel caps the rise the term is willing to pay for.
    At the old 0.5 m/s that cap was 12.7 mm -- below the ENTIRE 5-33 mm band, so
    the velocity term stopped paying long before the height term peaked, and the
    two terms fought each other.

    The second assertion is the real content and is written as a RELATIONSHIP:
    whatever the two constants become, the velocity term must not saturate before
    the height term peaks.
    """
    for label in HOP_ARM_SUFFIX:
        cfg = _registered(label)
        max_vel = cfg.rewards["hop_upward_velocity"].params["max_vel"]
        tid = _hop_task_id(label)
        # The relationship first, so it is the assertion that actually fails when
        # either constant regresses rather than being shadowed by the literal.
        saturating_rise = max_vel**2 / (2 * 9.81)
        assert saturating_rise > HOP_HEIGHT_GAIN, (
            f"{tid}: hop_upward_velocity saturates at {saturating_rise * 1e3:.1f} mm of "
            f"rise, at or below the {HOP_HEIGHT_GAIN * 1e3:.1f} mm the height reward "
            "peaks at -- the velocity term stops paying before the height term does"
        )
        assert max_vel == pytest.approx(HOP_MAX_LAUNCH_VEL), tid
        assert max_vel == pytest.approx(1.0), tid


def test_com_band_ceiling_is_above_the_hop_apex():
    """`com_height_target` pays a flat +1 in band and -(z - max)**2 above it, so
    crossing the top forfeits the whole +1 as a STEP, times its weight of 1.2.
    With the old 0.14 rigid top (0.17 sprung) that step landed at 23 mm of gain --
    inside the range the experiment needs explored, penalising exactly the hops we
    are trying to measure.

    Written as a relationship so it cannot silently regress if HOP_HEIGHT_GAIN,
    UNLOADED_RIGID_HEIGHT or H_ADD moves. Checked on a sprung arm AND on the
    Locked control, which wears the same boot and so gets the same shift.

    The apex is computed HERE, not read from the reward, because the reward no
    longer names an absolute height -- it shapes rise. The CoM band still has to
    clear the absolute height a successful hop reaches, so this is the one place
    the standing height and the gain are added together, and it is why
    UNLOADED_RIGID_HEIGHT survives at all.
    """
    apex = UNLOADED_RIGID_HEIGHT + H_ADD + HOP_HEIGHT_GAIN
    for label in ("k3900", "locked"):
        params = _registered(label).rewards["com_height_target"].params
        tid = _hop_task_id(label)
        assert params["target_height_max"] > apex, (
            f"{tid}: CoM band top {params['target_height_max']:.4f} is at or below the "
            f"target apex {apex:.4f} -- reaching the commanded hop height forfeits the "
            "band's +1 as a step penalty"
        )


def test_com_band_floor_is_untouched_by_the_hop_variant():
    """Only the UPPER edge moves. `target_height_min` still pays for not
    collapsing during stance, and the Phase-2 h_add translation in
    make_sprung_variant (explicitly out of scope) must still be the only thing
    acting on it.
    """
    base_min = make_microduck_velocity_env_cfg().rewards["com_height_target"].params[
        "target_height_min"
    ]
    for label in HOP_ARM_SUFFIX:
        params = _registered(label).rewards["com_height_target"].params
        assert params["target_height_min"] == pytest.approx(base_min + H_ADD), _hop_task_id(label)


def test_height_gaussian_discriminates_locked_from_sprung():
    """The property the whole change exists for, and the test that would have
    caught the original ceiling.

    The drop-rig probe rebounded 5 mm on the Locked arm and 33 mm at k=3900. If
    the reward cannot tell those two apart, all three arms score the same and the
    campaign returns an uninformative null after hours of GPU time per arm. Under
    the old (0.015, 0.008) params both read ~1e-8 and the ratio was ~1.0.

    The Gaussian is recomputed here from the REGISTERED params -- exp(-((rise -
    target_rise)/std)**2), matching microduck_mdp.hop_body_height -- rather than
    from module constants, so a registration that fails to thread them through
    fails this test too.

    Simpler than it used to be: the term's argument IS the gain now, so there is
    no reference height to reconstruct and no chance of reconstructing it wrong.
    """
    params = _registered("k3900").rewards["hop_body_height"].params
    target, std = params["target_rise"], params["std"]

    def reward(gain_m: float) -> float:
        return math.exp(-(((gain_m - target) / std) ** 2))

    locked_like = reward(0.005)
    sprung_like = reward(0.033)
    assert sprung_like > locked_like, "the reward must increase across the band"
    assert sprung_like / locked_like >= 10.0, (
        f"5 mm scores {locked_like:.4f} and 33 mm scores {sprung_like:.4f} -- only "
        f"{sprung_like / locked_like:.1f}x apart. The arms would be indistinguishable."
    )
    # Monotone across the band, so a taller hop is never worth less.
    samples = [reward(g / 1000.0) for g in range(5, 34)]
    assert samples == sorted(samples)


def test_height_gaussian_pays_from_a_standing_start():
    """The >=10x ratio test above is satisfiable by a cliff, not just a slope.

    Reverting HOP_HEIGHT_STD to its old 0.008 while leaving HOP_HEIGHT_GAIN at
    0.040 still passes `sprung_like / locked_like >= 10.0` above (0.465 / 5e-9
    is a huge, monotone ratio) yet pays approximately zero reward below 25 mm of
    gain -- a cliff the policy cannot climb from a standing start (0 mm), which
    is exactly the risk the widening from (0.015, 0.008) to (0.040, 0.020) was
    meant to remove. The ratio test never checks the low end in absolute terms,
    so it cannot see this.

    Two guards:
      1. an absolute floor -- the registered Gaussian evaluated at a 5 mm gain
         (the Locked-arm drop-rig datum) must be worth at least 0.03, so there
         is always a live gradient from a standing start;
      2. the relationship that makes (1) robust to future retuning: std must be
         at least 40% of gain, written as a relationship between the two
         constants rather than as two more hardcoded numbers.
    """
    params = _registered("k3900").rewards["hop_body_height"].params
    target, std = params["target_rise"], params["std"]

    def reward(gain_m: float) -> float:
        return math.exp(-(((gain_m - target) / std) ** 2))

    assert reward(0.005) >= 0.03, (
        f"5 mm of gain scores {reward(0.005):.4f}, below the 0.03 floor -- "
        "there is no usable gradient from a standing start"
    )
    assert HOP_HEIGHT_STD >= 0.4 * HOP_HEIGHT_GAIN, (
        f"HOP_HEIGHT_STD ({HOP_HEIGHT_STD}) is less than 40% of HOP_HEIGHT_GAIN "
        f"({HOP_HEIGHT_GAIN}) -- the Gaussian can be narrowed back into a cliff "
        "without failing the discrimination ratio test above"
    )


# ---------------------------------------------------------------------------
# The reward BUDGET. The first Phase 4 sweep -- three arms, 8000 iterations each
# -- returned a null: all three converged to standing perfectly still, both feet
# airborne 0.07-0.3% of the time. The cause was arithmetic, not tuning, and the
# tests below are the ones that would have caught it before the GPU time was
# spent.
# ---------------------------------------------------------------------------

# Mean of the launch gate clamp(sin(2*pi*phi), 0) over one cycle. This factor --
# not the weight -- is what a hop term is actually worth per step, and it is the
# whole reason the weights are what they are.
_LAUNCH_GATE_MEAN = 1.0 / math.pi


def test_hop_budget_beats_standing_still():
    """WEIGHT-ARITHMETIC REGRESSION GUARD. Not a model of realised behaviour.

    Read that literally: neither side of this inequality is what a trained policy
    actually earns, and the test must not be cited as if it were.

      * The hop side assumes all three shaped factors pinned at 1.0 for the
        ENTIRE launch half. A real 33 mm hop earns roughly 2.86/step, not 8.91 --
        flight is ~16% of the cycle rather than 50%, and `hop_body_height` is
        both airborne-gated AND Gaussian, so it is never pinned at 1.
      * The standing side is an upper bound too, though a much tighter one: all
        three terms in it really are paid every step by a robot doing nothing.

    What it DOES catch, and the only thing it is for, is the class of failure
    that produced the first Phase 4 sweep's null: weights that look reasonable
    term by term and lose in aggregate. `clamp(sin(2*pi*phi), 0)` averages
    1/pi = 0.318 over a cycle, so a hop term's per-step ceiling is `weight *
    0.318`, not `weight`. At the original 3.0/2.0/2.0 that was
    0.955 + 0.637 + ~0.3 = ~1.9/step for a PHYSICALLY IMPOSSIBLE perfect hopper,
    against `com_height_target` 1.2 + `upright` 1.0 = 2.2/step for standing
    perfectly still, at lower `action_rate_l2` cost and near-zero fall risk. A
    perfect hop lost to standing before any risk was counted. The policy did not
    fail to find hopping; it correctly found hopping was worse, and all three
    arms -- Locked, k2500, k3900, 8000 iterations each -- learned to stand.

    Both sides are computed from the REGISTERED weights on a REGISTERED task, so
    the test does the aggregation itself rather than trusting a hardcoded total.

    Two corrections to the standing side, both made after review:

      * `com_height_target` is no longer flat: it is now recovery-gated through
        `com_height_target_recovery_only`, so its per-step maximum is
        `weight * 1/pi`, exactly like a hop term -- not `weight`.
      * `pose` (`variable_posture`, weight 2.0) belongs here. It is the
        SECOND-LARGEST standing payout, larger than `upright`, and it is one a
        hop substantially forfeits. Omitting it flattered the hop side.
    """
    for label in HOP_ARM_SUFFIX:
        cfg = _registered(label)
        tid = _hop_task_id(label)
        rewards = cfg.rewards

        # Hop side: the three launch-gated terms at their per-step ceiling.
        hop_ceiling = _LAUNCH_GATE_MEAN * sum(
            rewards[name].weight
            for name in (
                "hop_both_feet_airborne",
                "hop_upward_velocity",
                "hop_body_height",
            )
        )

        # Standing side. `upright` and `pose` are flat-+1-shaped and ungated, so
        # they pay their full weight every step. `com_height_target` is now
        # recovery-gated, so it pays weight * 1/pi like a hop term.
        standing = (
            rewards["com_height_target"].weight * _LAUNCH_GATE_MEAN
            + rewards["upright"].weight * 1.0
            + rewards["pose"].weight * 1.0
        )

        assert hop_ceiling >= 2.0 * standing, (
            f"{tid}: a perfect hopper's ceiling is {hop_ceiling:.2f}/step against "
            f"{standing:.2f}/step for standing perfectly still -- only "
            f"{hop_ceiling / standing:.2f}x. The first Phase 4 sweep ran at "
            f"{1.9 / 2.2:.2f}x and all three arms learned to stand."
        )


def test_hop_budget_terms_all_exist_on_both_sides():
    """Guard the guard. The budget test sums named terms; if one were renamed or
    dropped the sum would silently shrink (hop side) or grow (standing side) and
    the inequality could pass for the wrong reason."""
    for label in HOP_ARM_SUFFIX:
        rewards = _registered(label).rewards
        for name in (
            "hop_both_feet_airborne",
            "hop_upward_velocity",
            "hop_body_height",
            "com_height_target",
            "upright",
            "pose",
        ):
            assert name in rewards, f"{_hop_task_id(label)}: {name} missing"
            assert rewards[name].weight > 0.0, f"{_hop_task_id(label)}: {name} weight <= 0"


def test_budget_standing_side_tracks_the_com_height_gating():
    """The standing side must apply 1/pi to `com_height_target` if and only if
    that term is actually gated. If a future change reverts the func swap back to
    the flat `com_height_target`, the budget test would keep discounting a term
    that is once again paid every step -- understating standing by 0.82/step and
    passing for the wrong reason. Tie the two together explicitly."""
    for label in HOP_ARM_SUFFIX:
        term = _registered(label).rewards["com_height_target"]
        assert term.func is microduck_mdp.com_height_target_recovery_only, (
            f"{_hop_task_id(label)}: com_height_target is not recovery-gated, so "
            "test_hop_budget_beats_standing_still must stop applying 1/pi to it"
        )


def test_registered_hop_weights_are_the_rebalanced_ones():
    """Pin the 4x through the registered tasks. The budget test above is written
    as a relationship so it survives retuning; this one catches a silent revert of
    the specific numbers the sweep will be relaunched with."""
    for label in HOP_ARM_SUFFIX:
        rewards = _registered(label).rewards
        tid = _hop_task_id(label)
        assert rewards["hop_both_feet_airborne"].weight == pytest.approx(AIRBORNE_WEIGHT), tid
        assert rewards["hop_upward_velocity"].weight == pytest.approx(UPWARD_VELOCITY_WEIGHT), tid
        assert rewards["hop_body_height"].weight == pytest.approx(BODY_HEIGHT_WEIGHT), tid
    assert (AIRBORNE_WEIGHT, UPWARD_VELOCITY_WEIGHT, BODY_HEIGHT_WEIGHT) == (12.0, 8.0, 8.0)


def test_action_rate_weight_is_untouched_by_the_rebalance():
    """Explicitly out of scope: the hop weights went up 4x, the action-rate cost
    did not move. Raising both would cancel the rebalance."""
    rigid = make_microduck_velocity_env_cfg()
    for label in HOP_ARM_SUFFIX:
        assert _registered(label).rewards["action_rate_l2"].weight == pytest.approx(
            rigid.rewards["action_rate_l2"].weight
        ), _hop_task_id(label)


# --- the load-phase term ----------------------------------------------------


def test_load_force_registered_on_every_arm():
    """Nothing rewarded the load half: all three hop terms gate on sin > 0.
    Without an actuator countermovement the spring cannot be charged (static sag
    under body weight alone is 0.48 mm at k=3900, ~0.45 mJ, worth 0.1 mm of
    lift), so the spring needs a hop to charge and the hop needs a charged
    spring. This term is what breaks that circularity."""
    for label in HOP_ARM_SUFFIX:
        term = _registered(label).rewards["hop_load_force"]
        tid = _hop_task_id(label)
        assert term.func is microduck_mdp.hop_load_force, tid
        assert term.weight == pytest.approx(LOAD_FORCE_WEIGHT), tid
        assert term.params["sensor_name"] == SENSOR_NAME, tid
        assert term.params["command_name"] == "twist", tid
        # NOT the BODY_WEIGHT_N constant: apply_hop_corrections replaces it with
        # the arm's real compiled mass, which now differs per arm (8.380 N sprung,
        # 7.380 N standard). See
        # test_body_weight_is_taken_from_the_ACTUAL_arm_not_a_constant.
        assert 7.0 < term.params["body_weight_n"] < 9.0, tid
        assert term.params["max_ratio"] == pytest.approx(LOAD_FORCE_MAX_RATIO), tid


def test_load_force_max_ratio_spans_the_springs_working_range():
    """At k = 3900 N/m each mm of pad travel costs 3.9 N, so `max_ratio` fixes
    how much of the pad's 12 mm travel the reward has any gradient across:

        travel_mm = max_ratio * BODY_WEIGHT_N / (2 * k) * 1000

    At the original 2.0 that was 2.2 mm -- 18% of travel, leaving ~82% of the
    spring's working range flat, which is exactly the range that stores energy
    and exactly where the three arms differ. Written as a relationship against
    TRAVEL so it cannot regress silently if the spring or the mass changes."""
    from mjlab_microduck.robot.sprung_foot import TRAVEL

    k = 3900.0
    saturating_travel = LOAD_FORCE_MAX_RATIO * BODY_WEIGHT_N / (2 * k)
    assert saturating_travel >= 0.4 * TRAVEL, (
        f"hop_load_force saturates at {saturating_travel * 1e3:.1f} mm of the pad's "
        f"{TRAVEL * 1e3:.0f} mm travel -- under 40% of the range, so most of the "
        "spring's working band gets no gradient"
    )
    # ...and not so high the knee-limited actuators (52.4 N/foot available)
    # could never reach it, which would flatten the top of the range instead.
    assert saturating_travel <= 0.75 * TRAVEL


def test_load_force_is_identical_on_the_locked_control_arm():
    """It is a FORCE reward, not a compression reward, precisely so the Locked
    control gets the same signal. Both arms can press down; only the sprung arms
    convert that press into stored energy. A compression reward would read
    identically zero on Locked -- it has no spring joint -- and destroy the
    controlled comparison the whole experiment rests on."""
    locked = _registered("locked").rewards["hop_load_force"]
    sprung = _registered("k3900").rewards["hop_load_force"]
    assert locked.func is sprung.func
    assert locked.weight == sprung.weight
    assert locked.params == sprung.params


def test_body_weight_is_taken_from_the_ACTUAL_arm_not_a_constant():
    """The arms no longer share a mass, so a single constant is wrong.

    893.0 g with spring boots (791.0 g robot + 2 x 51 g delta) against 791.0 g
    standard. hop_load_force normalises by body weight, so hardcoding one value
    made the lighter Standard arm's load term read 12.9% high -- an advantage to
    the control arm. apply_hop_corrections now compiles each robot and uses its
    real weight.

    The datum is the WHOLE ROBOT ON A SCALE, battery and boots included: 908 g
    measured 2026-09-04, less the 15 g CPU radiator that has since been removed
    (it shorted the electronics), so 893 g. `UNMODELLED_TRUNK_MASS` carries the
    53.8 g the spec is still missing. If someone changes that constant, this
    test is where they find out what else assumed the old number.
    """
    sprung = _registered("k3344").rewards["hop_load_force"].params["body_weight_n"]
    assert sprung == pytest.approx(0.8930 * 9.81, rel=0.01)

    from mjlab.tasks.registry import load_env_cfg
    std = load_env_cfg("Mjlab-Hop-Flat-Standard-MicroDuck")
    std_w = std.rewards["hop_load_force"].params["body_weight_n"]
    assert std_w == pytest.approx(0.7910 * 9.81, rel=0.01)
    assert std_w < sprung, "the standard arm is lighter; it must not share a constant"
    assert PAD_MASS == pytest.approx(0.051)


def test_both_force_terms_share_one_body_weight():
    """hop_symmetric_push is built to saturate on the same push hop_load_force
    does -- each foot at body weight against the pair at max_ratio 2.0 -- and
    that only holds if the two read the SAME body weight.

    This is a regression test for a real break: apply_hop_corrections patched
    hop_load_force from the compiled mass and left hop_symmetric_push on the
    8.60 N literal, putting them 3.5% apart.
    """
    from mjlab.tasks.registry import load_env_cfg
    cfg = load_env_cfg("Mjlab-Hop-InPlaceSym-K3344-MicroDuck")
    lf = cfg.rewards["hop_load_force"].params["body_weight_n"]
    sp = cfg.rewards["hop_symmetric_push"].params["body_weight_n"]
    assert sp == pytest.approx(lf, rel=1e-9), (
        f"symmetric push normalises against {sp} N but load force uses {lf} N; "
        "the two terms must saturate on the same push"
    )
    # And it must be the REAL mass, not the fallback literal.
    assert lf == pytest.approx(0.8930 * 9.81, rel=0.01)


def test_load_force_stays_below_the_launch_terms():
    """The countermovement is the enabler, not the objective. If pressing down
    paid more than leaving the ground, the policy's best move would be to squat
    hard and never jump -- a new way to reach the same null."""
    for label in HOP_ARM_SUFFIX:
        rewards = _registered(label).rewards
        load_ceiling = _LAUNCH_GATE_MEAN * rewards["hop_load_force"].weight
        launch_ceiling = _LAUNCH_GATE_MEAN * sum(
            rewards[n].weight
            for n in ("hop_both_feet_airborne", "hop_upward_velocity", "hop_body_height")
        )
        assert load_ceiling < launch_ceiling, _hop_task_id(label)


# --- the anti-tuck rise gate on hop_both_feet_airborne -----------------------


def test_airborne_rise_gate_registered_on_every_arm():
    """Without it, `hop_both_feet_airborne` at weight 12.0 is farmable by a TUCK.
    Only two collision geoms exist on the whole robot (the two pads; every other
    geom is contype=0), so the contact predicate says nothing about the 877 g
    body: retracting both 70 g feet satisfies it at ~50% duty inside the launch
    half for 12*(1/pi)*0.5 = 1.9/step, matching a genuine 33 mm hop's airborne
    payout with a HIGHER `pose` reward and no fall risk. `foot_swing_height`'s
    target (0.02 m) is exactly the flutter amplitude, so it is free, and
    `foot_clearance` reads xy velocity only."""
    for label in HOP_ARM_SUFFIX:
        params = _registered(label).rewards["hop_both_feet_airborne"].params
        tid = _hop_task_id(label)
        assert "min_rise" in params, f"{tid}: airborne term has no rise gate"
        assert params["min_rise"] == pytest.approx(MIN_RISE), tid
        assert "min_height" not in params, f"{tid}: absolute-height gate came back"


def test_both_airborne_terms_are_stateful_rise_trackers():
    """Both must inherit the takeoff latch. A stateless reimplementation of
    either would silently go back to reading absolute height."""
    for label in HOP_ARM_SUFFIX:
        rewards = _registered(label).rewards
        tid = _hop_task_id(label)
        for name in ("hop_both_feet_airborne", "hop_body_height"):
            func = rewards[name].func
            assert isinstance(func, type), f"{tid}: {name} is not a class term"
            assert issubclass(func, microduck_mdp._HopRiseTracker), f"{tid}: {name}"
            assert callable(getattr(func, "reset", None)), (
                f"{tid}: {name} has no reset(), so RewardManager will never clear "
                "its latch on episode reset"
            )


def test_every_hop_terms_params_match_its_signature():
    """`RewardManager.compute` calls `func(env, **term_cfg.params)`, so a params
    key the callable does not accept is a TypeError on the FIRST STEP of an 8000
    iteration run -- after the job has been queued and the GPU allocated.

    This became a live risk when the two airborne terms turned into classes: the
    params travel to `__call__`, not to `__init__`, and renaming
    `target_height` -> `target_rise` or `min_height` -> `min_rise` in one place
    and not the other would not otherwise be caught by anything static.

    Checks the class terms against `__call__` (which is what
    `ManagerBase._resolve_common_term_cfg` leaves behind after instantiating
    them) and plain functions against themselves.
    """
    import inspect

    for label in HOP_ARM_SUFFIX:
        rewards = _registered(label).rewards
        tid = _hop_task_id(label)
        for name in (
            "hop_both_feet_airborne",
            "hop_upward_velocity",
            "hop_body_height",
            "hop_load_force",
            "hop_energy_monitor",
            "com_height_target",
        ):
            term = rewards[name]
            target = term.func.__call__ if isinstance(term.func, type) else term.func
            sig = inspect.signature(target)
            accepted = set(sig.parameters)
            unknown = set(term.params) - accepted
            assert not unknown, (
                f"{tid}: {name} is registered with param(s) {sorted(unknown)} that "
                f"{target.__qualname__} does not accept -- this would raise TypeError "
                "on the first step of training"
            )


def test_min_rise_admits_the_locked_arm_but_not_a_tuck():
    """The gate's two-sided constraint, and the reason an absolute threshold
    could not simply be raised to fix the tall-tuck.

    Upper bound: it must admit the Locked control arm's expected ~5 mm hop. A
    gate above that scores Locked at zero airborne reward and makes the
    arm-to-arm comparison -- which IS the experiment -- meaningless. That is
    exactly why the previous absolute threshold was stuck: the robot has ~14.2 mm
    of posture headroom to tuck from, but a 14 mm threshold would have gated out
    Locked entirely.

    Lower bound: it must exceed the trunk dip a tuck produces (~1 mm over the
    step, ~4 mm over a 51 ms tuck).
    """
    assert MIN_RISE < 0.005, "MIN_RISE must admit the Locked arm's ~5 mm hop"
    assert MIN_RISE > 0.0, "a zero gate admits a tuck"
    # And it must stay a small fraction of the gain being shaped, so it prices
    # out a tuck without also shaping height (that is hop_body_height's job).
    assert MIN_RISE <= 0.25 * HOP_HEIGHT_GAIN


def test_upward_velocity_stays_ungated_as_the_discovery_gradient():
    """`hop_upward_velocity` deliberately keeps NO height or contact gate. It is
    the dense signal that bootstraps liftoff from a standing start; gating it
    the way the airborne term is gated would leave nothing to climb from zero."""
    for label in HOP_ARM_SUFFIX:
        params = _registered(label).rewards["hop_upward_velocity"].params
        tid = _hop_task_id(label)
        assert "min_rise" not in params, tid
        assert "min_height" not in params, tid
        assert "sensor_name" not in params, tid
        assert not isinstance(
            _registered(label).rewards["hop_upward_velocity"].func, type
        ), f"{tid}: hop_upward_velocity must stay a plain stateless function"


# --- the launch-half silencing of com_height_target --------------------------


def test_com_height_target_is_silenced_during_launch():
    """`com_height_target`'s flat +1-in-band (x1.2) was the single largest reward
    for standing perfectly still. During the launch half we want the robot LEAVING
    the band, so the registered term must be the recovery-gated wrapper."""
    for label in HOP_ARM_SUFFIX:
        term = _registered(label).rewards["com_height_target"]
        tid = _hop_task_id(label)
        assert term.func is microduck_mdp.com_height_target_recovery_only, tid
        assert term.params["command_name"] == "twist", tid


def test_com_height_target_swap_preserves_the_sprung_band_shift():
    """The swap MUST keep the term's key and its two band params, because
    `make_sprung_variant` runs AFTER `make_hop_variant`, looks the term up by the
    key "com_height_target", and shifts `target_height_min`/`target_height_max` by
    h_add in place. Renaming the key or dropping either param would break the band
    on every sprung arm silently -- no error, just a robot penalised for its own
    geometry."""
    base = make_microduck_velocity_env_cfg().rewards["com_height_target"].params
    for label in HOP_ARM_SUFFIX:
        params = _registered(label).rewards["com_height_target"].params
        tid = _hop_task_id(label)
        # Both edges present, and both carry the h_add shift applied afterwards.
        assert params["target_height_min"] == pytest.approx(
            base["target_height_min"] + H_ADD
        ), tid
        assert params["target_height_max"] == pytest.approx(HOP_COM_HEIGHT_MAX + H_ADD), tid


def test_upright_is_deliberately_not_gated():
    """`upright` genuinely pays for not tipping. Suppressing it during launch
    would encourage tipping at exactly the moment of takeoff, so it keeps the
    velocity env's func and weight."""
    rigid = make_microduck_velocity_env_cfg().rewards["upright"]
    for label in HOP_ARM_SUFFIX:
        term = _registered(label).rewards["upright"]
        tid = _hop_task_id(label)
        assert term.func is rigid.func, tid
        assert term.weight == pytest.approx(rigid.weight), tid

from mjlab_microduck.train_hook import maybe_submit_to_hf_jobs

# `train <task> ... --hf-jobs` submits to HF Jobs and exits here, before any
# of the cfg imports below: this module is what mjlab's plugin loader pulls
# in, and it is the only train path no install order can take from us (see
# train_hook.py). A no-op without the flag.
maybe_submit_to_hf_jobs()

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner


class MicroduckOnPolicyRunner(VelocityOnPolicyRunner):
    def __init__(self, env, train_cfg: dict, log_dir=None, device="cpu", **kwargs):
        super().__init__(env, train_cfg, log_dir, device, **kwargs)
        # resolve_symmetry_config injects _env into train_cfg["algorithm"]["symmetry_cfg"]
        # in-place, sharing the same dict object with self.alg.symmetry.  Replace the
        # train_cfg reference with a copy that omits _env so dump_yaml can serialize the
        # config (MjSpec is not picklable), without touching the PPO's internal reference.
        alg = train_cfg.get("algorithm", {})
        sym = alg.get("symmetry_cfg") if isinstance(alg, dict) else None
        if isinstance(sym, dict) and "_env" in sym:
            alg["symmetry_cfg"] = {k: v for k, v in sym.items() if k != "_env"}


from .microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
    MicroduckRlCfg,
)
from .microduck_standup_env_cfg import (
    make_microduck_standup_env_cfg,
    MicroduckStandUpRlCfg,
)
from .microduck_velstand_env_cfg import (
    make_microduck_velstand_env_cfg,
    MicroduckVelStandRlCfg,
)
from .microduck_ground_pick_env_cfg import (
    make_microduck_ground_pick_env_cfg,
    MicroduckGroundPickRlCfg,
)
from .microduck_ball_kick_env_cfg import (
    make_microduck_ball_kick_env_cfg,
    MicroduckBallKickRlCfg,
)
from .microduck_sitstand_env_cfg import (
    make_microduck_sitstand_env_cfg,
    MicroduckSitStandRlCfg,
)
from .microduck_velocity_rollers_env_cfg import (
    make_microduck_velocity_rollers_env_cfg,
    MicroduckRollersRlCfg,
)
from .microduck_velocity_swizzle_env_cfg import (
    make_microduck_velocity_swizzle_env_cfg,
    MicroduckSwizzleRlCfg,
)
from .microduck_roller_crouch_env_cfg import (
    make_microduck_roller_crouch_env_cfg,
    MicroduckRollerCrouchRlCfg,
)
from .microduck_roller_slope_env_cfg import (
    make_microduck_roller_slope_env_cfg,
    MicroduckRollerSlopeRlCfg,
)
from .microduck_roller_standup_env_cfg import (
    make_microduck_roller_standup_env_cfg,
    MicroduckRollerStandUpRlCfg,
)
from .microduck_spin_env_cfg import (
    make_microduck_spin_env_cfg,
    MicroduckSpinRlCfg,
)
from .microduck_roulade_env_cfg import (
    make_microduck_roulade_env_cfg,
    MicroduckRouladeRlCfg,
)
from .backlash import make_backlash_variant

# Standard velocity task
register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-MicroDuck",
    env_cfg=make_microduck_velocity_env_cfg(),
    play_env_cfg=make_microduck_velocity_env_cfg(play=True),
    rl_cfg=MicroduckRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Rough-MicroDuck",
    env_cfg=make_microduck_velocity_env_cfg(rough=True),
    play_env_cfg=make_microduck_velocity_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# VelStand — walking + fall recovery + body pose control in one policy.
register_mjlab_task(
    task_id="Mjlab-VelStand-Flat-MicroDuck",
    env_cfg=make_microduck_velstand_env_cfg(),
    play_env_cfg=make_microduck_velstand_env_cfg(play=True),
    rl_cfg=MicroduckVelStandRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-VelStand-Rough-MicroDuck",
    env_cfg=make_microduck_velstand_env_cfg(rough=True),
    play_env_cfg=make_microduck_velstand_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckVelStandRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Stand-up task — robot starts inverted (lying on back) and must stand up
register_mjlab_task(
    task_id="Mjlab-StandUp-Flat-MicroDuck",
    env_cfg=make_microduck_standup_env_cfg(),
    play_env_cfg=make_microduck_standup_env_cfg(play=True),
    rl_cfg=MicroduckStandUpRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-StandUp-Rough-MicroDuck",
    env_cfg=make_microduck_standup_env_cfg(rough=True),
    play_env_cfg=make_microduck_standup_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckStandUpRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# SitStand task — commanded sit ↔ stand in one policy, gently, head commandable
register_mjlab_task(
    task_id="Mjlab-SitStand-Flat-MicroDuck",
    env_cfg=make_microduck_sitstand_env_cfg(),
    play_env_cfg=make_microduck_sitstand_env_cfg(play=True),
    rl_cfg=MicroduckSitStandRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-SitStand-Rough-MicroDuck",
    env_cfg=make_microduck_sitstand_env_cfg(rough=True),
    play_env_cfg=make_microduck_sitstand_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckSitStandRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Ground-pick task — crouch, touch the ground with the mouth tip, return to stand
register_mjlab_task(
    task_id="Mjlab-GroundPick-Flat-MicroDuck",
    env_cfg=make_microduck_ground_pick_env_cfg(),
    play_env_cfg=make_microduck_ground_pick_env_cfg(play=True),
    rl_cfg=MicroduckGroundPickRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# BallKick task — kick a 70mm/15g ball forward hard with the right foot from a
# standing start (flat terrain only — a ball on rough terrain is another task).
register_mjlab_task(
    task_id="Mjlab-BallKick-Flat-MicroDuck",
    env_cfg=make_microduck_ball_kick_env_cfg(),
    play_env_cfg=make_microduck_ball_kick_env_cfg(play=True),
    rl_cfg=MicroduckBallKickRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-GroundPick-Rough-MicroDuck",
    env_cfg=make_microduck_ground_pick_env_cfg(rough=True),
    play_env_cfg=make_microduck_ground_pick_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckGroundPickRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Roller skate velocity task (passive-wheel model; historical task id kept)
register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-MicroDuck-Rollers",
    env_cfg=make_microduck_velocity_rollers_env_cfg(),
    play_env_cfg=make_microduck_velocity_rollers_env_cfg(play=True),
    rl_cfg=MicroduckRollersRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Roller SWIZZLE task — clean classic swizzle (symmetric, feet grounded).
register_mjlab_task(
    task_id="Mjlab-Velocity-Swizzle-MicroDuck",
    env_cfg=make_microduck_velocity_swizzle_env_cfg(),
    play_env_cfg=make_microduck_velocity_swizzle_env_cfg(play=True),
    rl_cfg=MicroduckSwizzleRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-RollerCrouch-Flat-MicroDuck",
    env_cfg=make_microduck_roller_crouch_env_cfg(),
    play_env_cfg=make_microduck_roller_crouch_env_cfg(play=True),
    rl_cfg=MicroduckRollerCrouchRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-RollerSlope-Flat-MicroDuck",
    env_cfg=make_microduck_roller_slope_env_cfg(),
    play_env_cfg=make_microduck_roller_slope_env_cfg(play=True),
    rl_cfg=MicroduckRollerSlopeRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Roller STANDUP — se relever sur rollers (policy dédiée, départ au sol).
register_mjlab_task(
    task_id="Mjlab-RollerStandUp-Flat-MicroDuck",
    env_cfg=make_microduck_roller_standup_env_cfg(),
    play_env_cfg=make_microduck_roller_standup_env_cfg(play=True),
    rl_cfg=MicroduckRollerStandUpRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Spin task — rotation rapide sur place, sur rollers (slot ground-pick).
register_mjlab_task(
    task_id="Mjlab-Spin-Flat-MicroDuck",
    env_cfg=make_microduck_spin_env_cfg(),
    play_env_cfg=make_microduck_spin_env_cfg(play=True),
    rl_cfg=MicroduckSpinRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Roulade — forward roll over the flat head top, land back on the feet.
register_mjlab_task(
    task_id="Mjlab-Roulade-Flat-MicroDuck",
    env_cfg=make_microduck_roulade_env_cfg(),
    play_env_cfg=make_microduck_roulade_env_cfg(play=True),
    rl_cfg=MicroduckRouladeRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Backlash variants — ±1° serial gear play per servo + encoder-through-backlash
# actuator feedback and joint obs (see tasks/backlash.py). Each family keeps its
# base task's collision model: Velocity → robot_walk_backlash.xml,
# VelStand → robot_allcollisions_backlash.xml, StandUp → robot_groundcontact_backlash.xml. Obs/action dims are
# unchanged vs the base tasks.
from mjlab_microduck.robot.microduck_constants import (
    MICRODUCK_ALLCOLLISIONS_BACKLASH_ROBOT_CFG,
    MICRODUCK_BACKLASH_ROBOT_CFG,
    MICRODUCK_ROLLERS_BACKLASH_ROBOT_CFG,
    MICRODUCK_WALK_BACKLASH_ROBOT_CFG,
)

# (task_id, make_fn, make_kwargs, rl_cfg, backlash robot cfg). Task ids mirror
# the base ids with "-Backlash" inserted. Walk-model tasks get the walk
# backlash robot, roller tasks the wheels+backlash robot, the rest the
# groundcontact backlash robot — same model as their base task in each case.
_BL_GROUNDCONTACT = MICRODUCK_BACKLASH_ROBOT_CFG
_BL_WALK = MICRODUCK_WALK_BACKLASH_ROBOT_CFG
_BL_ROLLERS = MICRODUCK_ROLLERS_BACKLASH_ROBOT_CFG
_BL_ALLCOLLISIONS = MICRODUCK_ALLCOLLISIONS_BACKLASH_ROBOT_CFG
_BACKLASH_TASKS = (
    ("Mjlab-Velocity-Flat-Backlash-MicroDuck", make_microduck_velocity_env_cfg, {}, MicroduckRlCfg, _BL_WALK),
    ("Mjlab-Velocity-Rough-Backlash-MicroDuck", make_microduck_velocity_env_cfg, {"rough": True}, MicroduckRlCfg, _BL_WALK),
    ("Mjlab-VelStand-Flat-Backlash-MicroDuck", make_microduck_velstand_env_cfg, {}, MicroduckVelStandRlCfg, _BL_ALLCOLLISIONS),
    ("Mjlab-VelStand-Rough-Backlash-MicroDuck", make_microduck_velstand_env_cfg, {"rough": True}, MicroduckVelStandRlCfg, _BL_ALLCOLLISIONS),
    ("Mjlab-StandUp-Flat-Backlash-MicroDuck", make_microduck_standup_env_cfg, {}, MicroduckStandUpRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-StandUp-Rough-Backlash-MicroDuck", make_microduck_standup_env_cfg, {"rough": True}, MicroduckStandUpRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-SitStand-Flat-Backlash-MicroDuck", make_microduck_sitstand_env_cfg, {}, MicroduckSitStandRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-SitStand-Rough-Backlash-MicroDuck", make_microduck_sitstand_env_cfg, {"rough": True}, MicroduckSitStandRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-GroundPick-Flat-Backlash-MicroDuck", make_microduck_ground_pick_env_cfg, {}, MicroduckGroundPickRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-GroundPick-Rough-Backlash-MicroDuck", make_microduck_ground_pick_env_cfg, {"rough": True}, MicroduckGroundPickRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-BallKick-Flat-Backlash-MicroDuck", make_microduck_ball_kick_env_cfg, {}, MicroduckBallKickRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-Velocity-Flat-Backlash-MicroDuck-Rollers", make_microduck_velocity_rollers_env_cfg, {}, MicroduckRollersRlCfg, _BL_ROLLERS),
    ("Mjlab-Velocity-Swizzle-Backlash-MicroDuck", make_microduck_velocity_swizzle_env_cfg, {}, MicroduckSwizzleRlCfg, _BL_ROLLERS),
    ("Mjlab-RollerCrouch-Flat-Backlash-MicroDuck", make_microduck_roller_crouch_env_cfg, {}, MicroduckRollerCrouchRlCfg, _BL_ROLLERS),
    ("Mjlab-RollerSlope-Flat-Backlash-MicroDuck", make_microduck_roller_slope_env_cfg, {}, MicroduckRollerSlopeRlCfg, _BL_ROLLERS),
)
for _task_id, _make_cfg, _kw, _rl_cfg, _robot_cfg in _BACKLASH_TASKS:
    register_mjlab_task(
        task_id=_task_id,
        env_cfg=make_backlash_variant(_make_cfg(**_kw), _robot_cfg),
        play_env_cfg=make_backlash_variant(_make_cfg(play=True, **_kw), _robot_cfg),
        rl_cfg=_rl_cfg,
        runner_cls=MicroduckOnPolicyRunner,
    )


# ── Ground grasp (2026-10-10): the ground-pick slot with an opening beak + a real object ──
from pathlib import Path as _Path

from .microduck_ground_grasp_env_cfg import (
    make_microduck_ground_grasp_env_cfg,
    MicroduckGroundGraspRlCfg,
)


class GroundGraspFromPickRunner(MicroduckOnPolicyRunner):
    """Warm start from TODAY's ground-pick move (Rémi's suggestion): the actor MLP and its observation normalizer are
    copied from the deployed ONNX (same 61 -> 512 -> 256 -> 128 -> 14 ELU network; the exporter bakes
    (obs - mean) / std into it). The critic stays fresh (its observations differ: object, mouth, contacts), the
    exploration std starts at INIT_STD so the inherited move is not drowned in noise, and the normalizer count is huge
    so the inherited statistics stay put. A real resume (--agent.resume) loads over this as usual."""

    INIT_ONNX = _Path(__file__).parent / "init_policies" / "alpha_ground_pick.onnx"
    INIT_STD = 0.25
    # A FRESH Adam moves every weight by ~lr on its first steps (m/sqrt(v) ~ sign(g)): at the cfg's 1e-3 that shifted
    # the inherited actions by ~1 rad in ONE update (measured on the smoke run). Start low; the adaptive KL schedule
    # raises it (x1.5 per update while KL < desired/2) once the moments are warm.
    INIT_LR = 1e-4

    def __init__(self, env, train_cfg: dict, log_dir=None, device="cpu", **kwargs):
        super().__init__(env, train_cfg, log_dir, device, **kwargs)
        self._init_actor_from_onnx()

    def _init_actor_from_onnx(self):
        import onnx
        import torch
        from onnx import numpy_helper

        init = {i.name: numpy_helper.to_array(i) for i in onnx.load(str(self.INIT_ONNX)).graph.initializer}
        actor = self.alg.actor
        sd = actor.state_dict()
        std_key = next(k for k in init if k.startswith("onnx::Div"))
        std = init[std_key] - actor.obs_normalizer.eps      # the exporter bakes (obs - mean) / (std + eps)
        new = {
            "obs_normalizer._mean": init["obs_normalizer._mean"],
            "obs_normalizer._std": std,
            "obs_normalizer._var": std ** 2,
        }
        for k in ("mlp.0.weight", "mlp.0.bias", "mlp.2.weight", "mlp.2.bias", "mlp.4.weight", "mlp.4.bias",
                  "mlp.6.weight", "mlp.6.bias"):
            new[k] = init[k]
        for k, v in new.items():
            t = torch.as_tensor(v, dtype=sd[k].dtype, device=sd[k].device).reshape(sd[k].shape)
            sd[k].copy_(t)
        sd["obs_normalizer.count"].fill_(1e9)
        sd["distribution.std_param"].fill_(self.INIT_STD)
        actor.load_state_dict(sd)
        self.alg.learning_rate = self.INIT_LR
        for g in self.alg.optimizer.param_groups:
            g["lr"] = self.INIT_LR
        print(f"[ground_grasp] actor initialised from {self.INIT_ONNX.name} (std {self.INIT_STD}, lr {self.INIT_LR})")


register_mjlab_task(
    task_id="Mjlab-GroundGrasp-Flat-MicroDuck",
    env_cfg=make_microduck_ground_grasp_env_cfg(),
    play_env_cfg=make_microduck_ground_grasp_env_cfg(play=True),
    rl_cfg=MicroduckGroundGraspRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-GroundGrasp-FromPick-MicroDuck",
    env_cfg=make_microduck_ground_grasp_env_cfg(),
    play_env_cfg=make_microduck_ground_grasp_env_cfg(play=True),
    rl_cfg=MicroduckGroundGraspRlCfg,
    runner_cls=GroundGraspFromPickRunner,
)

# Variant: same, plus a Gaussian pull of the beak pitch toward 60° in the grasp window (the static pose study's sweet
# spot: both lips < 1 cm from the floor) — for when the task alone keeps today's straight-down beak.
register_mjlab_task(
    task_id="Mjlab-GroundGrasp-FromPickPitch60-MicroDuck",
    env_cfg=make_microduck_ground_grasp_env_cfg(beak_pitch_pref_weight=3.0, beak_pitch_target=60.0),
    play_env_cfg=make_microduck_ground_grasp_env_cfg(play=True, beak_pitch_pref_weight=3.0, beak_pitch_target=60.0),
    rl_cfg=MicroduckGroundGraspRlCfg,
    runner_cls=GroundGraspFromPickRunner,
)

# Tall objects standing up (small toys, 3-5 cm): the beak takes the top slice. Fine-tuned from a floor-grasp checkpoint.
register_mjlab_task(
    task_id="Mjlab-GroundGrasp-Tall-MicroDuck",
    env_cfg=make_microduck_ground_grasp_env_cfg(tall=True),
    play_env_cfg=make_microduck_ground_grasp_env_cfg(play=True, tall=True),
    rl_cfg=MicroduckGroundGraspRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Flat objects (2x2-Lego-like, 0.8-1.2 cm tall): no bisector preference, RL picks the beak angle.
register_mjlab_task(
    task_id="Mjlab-GroundGrasp-Flat-Objects-MicroDuck",
    env_cfg=make_microduck_ground_grasp_env_cfg(flat=True),
    play_env_cfg=make_microduck_ground_grasp_env_cfg(play=True, flat=True),
    rl_cfg=MicroduckGroundGraspRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)


# ── Expressive body motions (2026-10-28): one phase-commanded one-shot per motion, feet planted, grippy feet ──
from .expressive_motions import MOTIONS as _EXPRESSIVE_MOTIONS
from .microduck_expressive_env_cfg import make_expressive_rl_cfg, make_microduck_expressive_env_cfg


class ExpressiveFromPickRunner(GroundGraspFromPickRunner):
    """Warm start from alpha_ground_pick.onnx, exactly like the ground grasp: the actor already reads the phase
    command and balances on planted feet; the critic (reference / error / feet-offset obs) starts fresh."""


for _name in _EXPRESSIVE_MOTIONS:
    register_mjlab_task(
        task_id=f"Mjlab-Expressive-{_name}-MicroDuck",
        env_cfg=make_microduck_expressive_env_cfg(_name),
        play_env_cfg=make_microduck_expressive_env_cfg(_name, play=True),
        rl_cfg=make_expressive_rl_cfg(_name),
        runner_cls=ExpressiveFromPickRunner,
    )

# From-scratch twin of the priority motion (hedge: the inherited crouch could be a hard habit to unlearn).
register_mjlab_task(
    task_id="Mjlab-Expressive-Sassy-Scratch-MicroDuck",
    env_cfg=make_microduck_expressive_env_cfg("Sassy"),
    play_env_cfg=make_microduck_expressive_env_cfg("Sassy", play=True),
    rl_cfg=make_expressive_rl_cfg("Sassy"),
    runner_cls=MicroduckOnPolicyRunner,
)

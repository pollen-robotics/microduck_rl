from pathlib import Path

import torch
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


class MicroduckFrozenActorNormRunner(MicroduckOnPolicyRunner):
    """Preserve the transferred walking policy's observation coordinates.

    Contact-rich reverse starts have a very different observation distribution
    from the manufacturer's runway gait. Updating the actor normalizer shifts
    even unchanged runway observations and destroys walking within a few PPO
    iterations. The critic normalizer remains adaptive; only the actor's
    transferred input transform is frozen.
    """

    def __init__(self, env, train_cfg: dict, log_dir=None, device="cpu", **kwargs):
        super().__init__(env, train_cfg, log_dir, device, **kwargs)
        normalizer = getattr(self.alg.actor, "obs_normalizer", None)
        if hasattr(normalizer, "until"):
            normalizer.until = 0


class MicroduckStairSpecialistRunner(MicroduckOnPolicyRunner):
    """Seed the specialist actor without importing walking PPO state."""

    def __init__(self, env, train_cfg: dict, log_dir=None, device="cpu", **kwargs):
        distribution_cfg = train_cfg.get("actor", {}).get("distribution_cfg") or {}
        self._bootstrap_actor_std = float(distribution_cfg.get("init_std", 1.0))
        super().__init__(env, train_cfg, log_dir, device, **kwargs)

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict:
        bootstrap_actor = (
            load_cfg is None and Path(path).parent.name == ".bootstrap-walking"
        )
        if bootstrap_actor:
            load_cfg = {
                "actor": True,
                "critic": False,
                "optimizer": False,
                "iteration": False,
                "rnd": False,
            }
        infos = super().load(path, load_cfg, strict, map_location)
        if bootstrap_actor:
            distribution = self.alg.actor.distribution
            with torch.no_grad():
                if hasattr(distribution, "std_param"):
                    distribution.std_param.fill_(self._bootstrap_actor_std)
                elif hasattr(distribution, "log_std_param"):
                    distribution.log_std_param.fill_(
                        torch.log(
                            torch.tensor(
                                self._bootstrap_actor_std,
                                device=distribution.log_std_param.device,
                            )
                        )
                    )
        return infos


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
from .microduck_stairs_env_cfg import (
    make_microduck_stairs_env_cfg,
    MicroduckStairsRlCfg,
)
from .microduck_standard_stairs_env_cfg import (
    make_microduck_assisted_stair_specialist_env_cfg,
    make_microduck_stair_apex_mantle_env_cfg,
    make_microduck_stair_roulade_bank_env_cfg,
    make_microduck_stair_phase_balanced_rsi_env_cfg,
    make_microduck_stair_curriculum_rsi_env_cfg,
    make_microduck_stair_contact_mantle_rsi_env_cfg,
    make_microduck_stair_contact_release_rsi_env_cfg,
    make_microduck_stair_lip_commitment_rsi_env_cfg,
    make_microduck_stair_lip_checkpoint_rsi_env_cfg,
    make_microduck_stair_frontier_collocation_rsi_env_cfg,
    make_microduck_stair_terminal_position_rsi_env_cfg,
    make_microduck_stair_frontier_tier_rsi_env_cfg,
    make_microduck_stair_forward_propagation_rsi_env_cfg,
    make_microduck_stair_virtual_lip_transfer_rsi_env_cfg,
    make_microduck_stair_contact_stage_rsi_env_cfg,
    make_microduck_stair_stage15_reverse_rsi_env_cfg,
    make_microduck_stair_stage2_reverse_rsi_env_cfg,
    make_microduck_stair_near_shell_reverse_rsi_env_cfg,
    make_microduck_stair_stratified_shell_reverse_rsi_env_cfg,
    make_microduck_stair_soft_dynamics_rsi_env_cfg,
    make_microduck_stair_medium_dynamics_rsi_env_cfg,
    make_microduck_stair_foot_anchor_vault_env_cfg,
    make_microduck_stair_ordered_vault_env_cfg,
    make_microduck_stair_tread_contact_bank_env_cfg,
    make_microduck_stair_bridge_specialist_env_cfg,
    make_microduck_stair_launch_bank_env_cfg,
    make_microduck_stair_walker_bank_env_cfg,
    make_microduck_route_stairs_env_cfg,
    make_microduck_stair_specialist_env_cfg,
    make_microduck_standard_stairs_env_cfg,
    MicroduckAssistedStairSpecialistRlCfg,
    MicroduckStairApexMantleRlCfg,
    MicroduckStairRouladeBankRlCfg,
    MicroduckStairPhaseBalancedRsiRlCfg,
    MicroduckStairCurriculumRsiRlCfg,
    MicroduckStairContactMantleRsiRlCfg,
    MicroduckStairContactReleaseRsiRlCfg,
    MicroduckStairLipCommitmentRsiRlCfg,
    MicroduckStairLipCheckpointRsiRlCfg,
    MicroduckStairFrontierCollocationRsiRlCfg,
    MicroduckStairTerminalPositionRsiRlCfg,
    MicroduckStairFrontierTierRsiRlCfg,
    MicroduckStairForwardPropagationRsiRlCfg,
    MicroduckStairVirtualLipTransferRsiRlCfg,
    MicroduckStairContactStageRsiRlCfg,
    MicroduckStairStage15ReverseRsiRlCfg,
    MicroduckStairStage2ReverseRsiRlCfg,
    MicroduckStairNearShellReverseRsiRlCfg,
    MicroduckStairStratifiedShellReverseRsiRlCfg,
    MicroduckStairSoftDynamicsRsiRlCfg,
    MicroduckStairMediumDynamicsRsiRlCfg,
    MicroduckStairFootAnchorVaultRlCfg,
    MicroduckStairOrderedVaultRlCfg,
    MicroduckStairTreadContactBankRlCfg,
    MicroduckStairBridgeSpecialistRlCfg,
    MicroduckStairLaunchBankRlCfg,
    MicroduckStairWalkerBankRlCfg,
    MicroduckRouteStairsRlCfg,
    MicroduckStairSpecialistRlCfg,
    MicroduckStandardStairsRlCfg,
)
from .microduck_headstand_env_cfg import (
    make_microduck_headstand_env_cfg,
    MicroduckHeadstandRlCfg,
)
from .microduck_backflip_env_cfg import (
    make_microduck_backflip_env_cfg,
    MicroduckBackflipRlCfg,
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

# Experimental stair and acrobatics tasks.  They keep the shared observation
# contract, but are intentionally separate policies with task-specific resets
# and rewards.
register_mjlab_task(
    task_id="Mjlab-Stairs-MicroDuck",
    env_cfg=make_microduck_stairs_env_cfg(),
    play_env_cfg=make_microduck_stairs_env_cfg(play=True),
    rl_cfg=MicroduckStairsRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Stairs-Standard-MicroDuck",
    env_cfg=make_microduck_standard_stairs_env_cfg(),
    play_env_cfg=make_microduck_standard_stairs_env_cfg(play=True),
    rl_cfg=MicroduckStandardStairsRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Stairs-Route-MicroDuck",
    env_cfg=make_microduck_route_stairs_env_cfg(),
    play_env_cfg=make_microduck_route_stairs_env_cfg(play=True),
    rl_cfg=MicroduckRouteStairsRlCfg,
    runner_cls=MicroduckFrozenActorNormRunner,
)

register_mjlab_task(
    task_id="Mjlab-Stairs-Specialist-MicroDuck",
    env_cfg=make_microduck_stair_specialist_env_cfg(),
    play_env_cfg=make_microduck_stair_specialist_env_cfg(play=True),
    rl_cfg=MicroduckStairSpecialistRlCfg,
    runner_cls=MicroduckStairSpecialistRunner,
)

register_mjlab_task(
    task_id="Mjlab-Stairs-Assisted-Specialist-MicroDuck",
    env_cfg=make_microduck_assisted_stair_specialist_env_cfg(),
    play_env_cfg=make_microduck_assisted_stair_specialist_env_cfg(play=True),
    rl_cfg=MicroduckAssistedStairSpecialistRlCfg,
    runner_cls=MicroduckStairSpecialistRunner,
)

register_mjlab_task(
    task_id="Mjlab-Stairs-Bridge-Specialist-MicroDuck",
    env_cfg=make_microduck_stair_bridge_specialist_env_cfg(),
    play_env_cfg=make_microduck_stair_bridge_specialist_env_cfg(play=True),
    rl_cfg=MicroduckStairBridgeSpecialistRlCfg,
    runner_cls=MicroduckStairSpecialistRunner,
)

register_mjlab_task(
    task_id="Mjlab-Stairs-Walker-Bank-Specialist-MicroDuck",
    env_cfg=make_microduck_stair_walker_bank_env_cfg(),
    play_env_cfg=make_microduck_stair_walker_bank_env_cfg(play=True),
    rl_cfg=MicroduckStairWalkerBankRlCfg,
    runner_cls=MicroduckStairSpecialistRunner,
)

register_mjlab_task(
    task_id="Mjlab-Stairs-Launch-Bank-Specialist-MicroDuck",
    env_cfg=make_microduck_stair_launch_bank_env_cfg(),
    play_env_cfg=make_microduck_stair_launch_bank_env_cfg(play=True),
    rl_cfg=MicroduckStairLaunchBankRlCfg,
    runner_cls=MicroduckStairSpecialistRunner,
)

register_mjlab_task(
    task_id="Mjlab-Stairs-Apex-Mantle-Specialist-MicroDuck",
    env_cfg=make_microduck_stair_apex_mantle_env_cfg(),
    play_env_cfg=make_microduck_stair_apex_mantle_env_cfg(play=True),
    rl_cfg=MicroduckStairApexMantleRlCfg,
    runner_cls=MicroduckStairSpecialistRunner,
)

register_mjlab_task(
    task_id="Mjlab-Stairs-Roulade-Bank-Specialist-MicroDuck",
    env_cfg=make_microduck_stair_roulade_bank_env_cfg(),
    play_env_cfg=make_microduck_stair_roulade_bank_env_cfg(play=True),
    rl_cfg=MicroduckStairRouladeBankRlCfg,
    runner_cls=MicroduckStairSpecialistRunner,
)

register_mjlab_task(
    task_id="Mjlab-Stairs-Phase-Balanced-RSI-Specialist-MicroDuck",
    env_cfg=make_microduck_stair_phase_balanced_rsi_env_cfg(),
    play_env_cfg=make_microduck_stair_phase_balanced_rsi_env_cfg(play=True),
    rl_cfg=MicroduckStairPhaseBalancedRsiRlCfg,
    runner_cls=MicroduckStairSpecialistRunner,
)

register_mjlab_task(
    task_id="Mjlab-Stairs-Curriculum-RSI-Specialist-MicroDuck",
    env_cfg=make_microduck_stair_curriculum_rsi_env_cfg(),
    play_env_cfg=make_microduck_stair_curriculum_rsi_env_cfg(play=True),
    rl_cfg=MicroduckStairCurriculumRsiRlCfg,
    runner_cls=MicroduckStairSpecialistRunner,
)

register_mjlab_task(
    task_id="Mjlab-Stairs-Contact-Mantle-RSI-Specialist-MicroDuck",
    env_cfg=make_microduck_stair_contact_mantle_rsi_env_cfg(),
    play_env_cfg=make_microduck_stair_contact_mantle_rsi_env_cfg(play=True),
    rl_cfg=MicroduckStairContactMantleRsiRlCfg,
    runner_cls=MicroduckStairSpecialistRunner,
)

register_mjlab_task(
    task_id="Mjlab-Stairs-Soft-Dynamics-RSI-Specialist-MicroDuck",
    env_cfg=make_microduck_stair_soft_dynamics_rsi_env_cfg(),
    play_env_cfg=make_microduck_stair_soft_dynamics_rsi_env_cfg(play=True),
    rl_cfg=MicroduckStairSoftDynamicsRsiRlCfg,
    runner_cls=MicroduckStairSpecialistRunner,
)

register_mjlab_task(
    task_id="Mjlab-Stairs-Medium-Dynamics-RSI-Specialist-MicroDuck",
    env_cfg=make_microduck_stair_medium_dynamics_rsi_env_cfg(),
    play_env_cfg=make_microduck_stair_medium_dynamics_rsi_env_cfg(play=True),
    rl_cfg=MicroduckStairMediumDynamicsRsiRlCfg,
    runner_cls=MicroduckStairSpecialistRunner,
)

register_mjlab_task(
    task_id="Mjlab-Stairs-Contact-Release-RSI-Specialist-MicroDuck",
    env_cfg=make_microduck_stair_contact_release_rsi_env_cfg(),
    play_env_cfg=make_microduck_stair_contact_release_rsi_env_cfg(play=True),
    rl_cfg=MicroduckStairContactReleaseRsiRlCfg,
    runner_cls=MicroduckStairSpecialistRunner,
)

register_mjlab_task(
    task_id="Mjlab-Stairs-Lip-Commitment-RSI-Specialist-MicroDuck",
    env_cfg=make_microduck_stair_lip_commitment_rsi_env_cfg(),
    play_env_cfg=make_microduck_stair_lip_commitment_rsi_env_cfg(play=True),
    rl_cfg=MicroduckStairLipCommitmentRsiRlCfg,
    runner_cls=MicroduckStairSpecialistRunner,
)

register_mjlab_task(
    task_id="Mjlab-Stairs-Lip-Checkpoint-RSI-Specialist-MicroDuck",
    env_cfg=make_microduck_stair_lip_checkpoint_rsi_env_cfg(),
    play_env_cfg=make_microduck_stair_lip_checkpoint_rsi_env_cfg(play=True),
    rl_cfg=MicroduckStairLipCheckpointRsiRlCfg,
    runner_cls=MicroduckStairSpecialistRunner,
)

register_mjlab_task(
    task_id="Mjlab-Stairs-Frontier-Collocation-RSI-Specialist-MicroDuck",
    env_cfg=make_microduck_stair_frontier_collocation_rsi_env_cfg(),
    play_env_cfg=make_microduck_stair_frontier_collocation_rsi_env_cfg(play=True),
    rl_cfg=MicroduckStairFrontierCollocationRsiRlCfg,
    runner_cls=MicroduckStairSpecialistRunner,
)

register_mjlab_task(
    task_id="Mjlab-Stairs-Terminal-Position-RSI-Specialist-MicroDuck",
    env_cfg=make_microduck_stair_terminal_position_rsi_env_cfg(),
    play_env_cfg=make_microduck_stair_terminal_position_rsi_env_cfg(play=True),
    rl_cfg=MicroduckStairTerminalPositionRsiRlCfg,
    runner_cls=MicroduckStairSpecialistRunner,
)

register_mjlab_task(
    task_id="Mjlab-Stairs-Frontier-Tier-RSI-Specialist-MicroDuck",
    env_cfg=make_microduck_stair_frontier_tier_rsi_env_cfg(),
    play_env_cfg=make_microduck_stair_frontier_tier_rsi_env_cfg(play=True),
    rl_cfg=MicroduckStairFrontierTierRsiRlCfg,
    runner_cls=MicroduckStairSpecialistRunner,
)

register_mjlab_task(
    task_id="Mjlab-Stairs-Forward-Propagation-RSI-Specialist-MicroDuck",
    env_cfg=make_microduck_stair_forward_propagation_rsi_env_cfg(),
    play_env_cfg=make_microduck_stair_forward_propagation_rsi_env_cfg(play=True),
    rl_cfg=MicroduckStairForwardPropagationRsiRlCfg,
    runner_cls=MicroduckStairSpecialistRunner,
)

register_mjlab_task(
    task_id="Mjlab-Stairs-Virtual-Lip-Transfer-RSI-Specialist-MicroDuck",
    env_cfg=make_microduck_stair_virtual_lip_transfer_rsi_env_cfg(),
    play_env_cfg=make_microduck_stair_virtual_lip_transfer_rsi_env_cfg(play=True),
    rl_cfg=MicroduckStairVirtualLipTransferRsiRlCfg,
    runner_cls=MicroduckStairSpecialistRunner,
)

register_mjlab_task(
    task_id="Mjlab-Stairs-Contact-Stage-RSI-Specialist-MicroDuck",
    env_cfg=make_microduck_stair_contact_stage_rsi_env_cfg(),
    play_env_cfg=make_microduck_stair_contact_stage_rsi_env_cfg(play=True),
    rl_cfg=MicroduckStairContactStageRsiRlCfg,
    runner_cls=MicroduckStairSpecialistRunner,
)

register_mjlab_task(
    task_id="Mjlab-Stairs-Stage15-Reverse-RSI-Specialist-MicroDuck",
    env_cfg=make_microduck_stair_stage15_reverse_rsi_env_cfg(),
    play_env_cfg=make_microduck_stair_stage15_reverse_rsi_env_cfg(play=True),
    rl_cfg=MicroduckStairStage15ReverseRsiRlCfg,
    runner_cls=MicroduckStairSpecialistRunner,
)

register_mjlab_task(
    task_id="Mjlab-Stairs-Stage2-Reverse-RSI-Specialist-MicroDuck",
    env_cfg=make_microduck_stair_stage2_reverse_rsi_env_cfg(),
    play_env_cfg=make_microduck_stair_stage2_reverse_rsi_env_cfg(play=True),
    rl_cfg=MicroduckStairStage2ReverseRsiRlCfg,
    runner_cls=MicroduckStairSpecialistRunner,
)

register_mjlab_task(
    task_id="Mjlab-Stairs-Near-Shell-Reverse-RSI-Specialist-MicroDuck",
    env_cfg=make_microduck_stair_near_shell_reverse_rsi_env_cfg(),
    play_env_cfg=make_microduck_stair_near_shell_reverse_rsi_env_cfg(play=True),
    rl_cfg=MicroduckStairNearShellReverseRsiRlCfg,
    runner_cls=MicroduckStairSpecialistRunner,
)

register_mjlab_task(
    task_id="Mjlab-Stairs-Stratified-Shell-Reverse-RSI-Specialist-MicroDuck",
    env_cfg=make_microduck_stair_stratified_shell_reverse_rsi_env_cfg(),
    play_env_cfg=make_microduck_stair_stratified_shell_reverse_rsi_env_cfg(
        play=True
    ),
    rl_cfg=MicroduckStairStratifiedShellReverseRsiRlCfg,
    runner_cls=MicroduckStairSpecialistRunner,
)

register_mjlab_task(
    task_id="Mjlab-Stairs-Tread-Contact-Bank-Specialist-MicroDuck",
    env_cfg=make_microduck_stair_tread_contact_bank_env_cfg(),
    play_env_cfg=make_microduck_stair_tread_contact_bank_env_cfg(play=True),
    rl_cfg=MicroduckStairTreadContactBankRlCfg,
    runner_cls=MicroduckStairSpecialistRunner,
)

register_mjlab_task(
    task_id="Mjlab-Stairs-Foot-Anchor-Vault-Specialist-MicroDuck",
    env_cfg=make_microduck_stair_foot_anchor_vault_env_cfg(),
    play_env_cfg=make_microduck_stair_foot_anchor_vault_env_cfg(play=True),
    rl_cfg=MicroduckStairFootAnchorVaultRlCfg,
    runner_cls=MicroduckStairSpecialistRunner,
)

register_mjlab_task(
    task_id="Mjlab-Stairs-Ordered-Vault-Specialist-MicroDuck",
    env_cfg=make_microduck_stair_ordered_vault_env_cfg(),
    play_env_cfg=make_microduck_stair_ordered_vault_env_cfg(play=True),
    rl_cfg=MicroduckStairOrderedVaultRlCfg,
    runner_cls=MicroduckStairSpecialistRunner,
)

register_mjlab_task(
    task_id="Mjlab-Headstand-Flat-MicroDuck",
    env_cfg=make_microduck_headstand_env_cfg(),
    play_env_cfg=make_microduck_headstand_env_cfg(play=True),
    rl_cfg=MicroduckHeadstandRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Backflip-Flat-MicroDuck",
    env_cfg=make_microduck_backflip_env_cfg(),
    play_env_cfg=make_microduck_backflip_env_cfg(play=True),
    rl_cfg=MicroduckBackflipRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Backlash variants — ±1° serial gear play per servo + encoder-through-backlash
# actuator feedback and joint obs (see tasks/backlash.py). Each family keeps its
# base task's collision model: Velocity → robot_walk_backlash.xml,
# VelStand/StandUp → robot_allcollisions_backlash.xml. Obs/action dims are
# unchanged vs the base tasks.
from mjlab_microduck.robot.microduck_constants import (
    MICRODUCK_BACKLASH_ROBOT_CFG,
    MICRODUCK_ROLLERS_BACKLASH_ROBOT_CFG,
    MICRODUCK_WALK_BACKLASH_ROBOT_CFG,
)

# (task_id, make_fn, make_kwargs, rl_cfg, backlash robot cfg). Task ids mirror
# the base ids with "-Backlash" inserted. Walk-model tasks get the walk
# backlash robot, roller tasks the wheels+backlash robot, the rest the
# allcollisions backlash robot — same model as their base task in each case.
_BL_ALLCOL = MICRODUCK_BACKLASH_ROBOT_CFG
_BL_WALK = MICRODUCK_WALK_BACKLASH_ROBOT_CFG
_BL_ROLLERS = MICRODUCK_ROLLERS_BACKLASH_ROBOT_CFG
_BACKLASH_TASKS = (
    ("Mjlab-Velocity-Flat-Backlash-MicroDuck", make_microduck_velocity_env_cfg, {}, MicroduckRlCfg, _BL_WALK),
    ("Mjlab-Velocity-Rough-Backlash-MicroDuck", make_microduck_velocity_env_cfg, {"rough": True}, MicroduckRlCfg, _BL_WALK),
    ("Mjlab-VelStand-Flat-Backlash-MicroDuck", make_microduck_velstand_env_cfg, {}, MicroduckVelStandRlCfg, _BL_ALLCOL),
    ("Mjlab-VelStand-Rough-Backlash-MicroDuck", make_microduck_velstand_env_cfg, {"rough": True}, MicroduckVelStandRlCfg, _BL_ALLCOL),
    ("Mjlab-StandUp-Flat-Backlash-MicroDuck", make_microduck_standup_env_cfg, {}, MicroduckStandUpRlCfg, _BL_ALLCOL),
    ("Mjlab-StandUp-Rough-Backlash-MicroDuck", make_microduck_standup_env_cfg, {"rough": True}, MicroduckStandUpRlCfg, _BL_ALLCOL),
    ("Mjlab-SitStand-Flat-Backlash-MicroDuck", make_microduck_sitstand_env_cfg, {}, MicroduckSitStandRlCfg, _BL_ALLCOL),
    ("Mjlab-SitStand-Rough-Backlash-MicroDuck", make_microduck_sitstand_env_cfg, {"rough": True}, MicroduckSitStandRlCfg, _BL_ALLCOL),
    ("Mjlab-GroundPick-Flat-Backlash-MicroDuck", make_microduck_ground_pick_env_cfg, {}, MicroduckGroundPickRlCfg, _BL_ALLCOL),
    ("Mjlab-GroundPick-Rough-Backlash-MicroDuck", make_microduck_ground_pick_env_cfg, {"rough": True}, MicroduckGroundPickRlCfg, _BL_ALLCOL),
    ("Mjlab-BallKick-Flat-Backlash-MicroDuck", make_microduck_ball_kick_env_cfg, {}, MicroduckBallKickRlCfg, _BL_ALLCOL),
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

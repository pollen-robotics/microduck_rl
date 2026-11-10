"""Independent, unvalidated fine-tuning recipe for genuine lateral locomotion.

This task changes training examples and rewards, never the commanded yaw at
deployment. Existing v8 tasks and published weights are untouched. A trained
candidate must still pass repeated, push-free side-step and regression probes;
the existence of this configuration does not mean lateral walking is fixed.
"""

from copy import deepcopy
from dataclasses import dataclass

import torch
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import ObservationTermCfg, RewardTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from mjlab_microduck.robot.microduck_constants import (
    MICRODUCK_ALLCOLLISIONS_BACKLASH_ROBOT_CFG,
)
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.backlash import make_backlash_variant
from mjlab_microduck.tasks.microduck_velstand_env_cfg import (
    make_microduck_velstand_env_cfg,
)
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg


def command_families(
    samples: torch.Tensor,
    lateral_fraction: float = 0.7,
    longitudinal_fraction: float = 0.15,
) -> torch.Tensor:
    """Map uniform samples to lateral, longitudinal, and turning families."""
    if not 0 <= lateral_fraction <= 1 or not 0 <= longitudinal_fraction <= 1:
        raise ValueError("command fractions must be between zero and one")
    if lateral_fraction + longitudinal_fraction > 1:
        raise ValueError("command fractions exceed one")
    return (samples >= lateral_fraction).long() + (
        samples >= lateral_fraction + longitudinal_fraction
    ).long()


def phase_commands(
    family: torch.Tensor,
    sign: torch.Tensor,
    phase: torch.Tensor,
    lateral_speed: float = 0.3,
    longitudinal_speed: float = 0.3,
    turn_rate: float = 0.6,
) -> torch.Tensor:
    """Stand -> move -> stop -> reverse, with exactly one commanded axis."""
    phase = phase.remainder(4)
    moving = (phase == 1) | (phase == 3)
    direction = sign * torch.where(phase == 3, -1.0, 1.0) * moving
    result = torch.zeros(family.numel(), 3, device=family.device)
    result[:, 0] = direction * (family == 1) * longitudinal_speed
    result[:, 1] = direction * (family == 0) * lateral_speed
    result[:, 2] = direction * (family == 2) * turn_rate
    return result


class LateralVelocityCommand(microduck_mdp.VelocityCommandCommandOnly):
    """Command-only episodes with rest starts and both directions in every cycle."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.family = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.first_sign = torch.ones(self.num_envs, device=self.device)
        self.segment_yaw = torch.zeros(self.num_envs, device=self.device)
        self.segment_position = torch.zeros(self.num_envs, 3, device=self.device)
        self.mean_yaw_rate = torch.zeros(self.num_envs, device=self.device)

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        fresh = env_ids[self.command_counter[env_ids] == 0]
        if fresh.numel():
            if self.cfg.deterministic:
                sample = (fresh.remainder(100).float() + 0.5) / 100
                sign = torch.where(fresh.remainder(2) == 0, 1.0, -1.0)
            else:
                sample = torch.rand(fresh.numel(), device=self.device)
                sign = torch.where(
                    torch.rand(fresh.numel(), device=self.device) < 0.5, 1.0, -1.0
                )
            self.family[fresh] = command_families(
                sample, self.cfg.lateral_fraction, self.cfg.longitudinal_fraction
            )
            self.first_sign[fresh] = sign

        phase = self.command_counter[env_ids].remainder(4)
        # This reference is privileged TRAINING state, never a deployment command or
        # an observation yaw offset. Reward net travel along the original side axis;
        # body-frame velocity alone pays an arc that keeps turning with the duck.
        self.segment_yaw[env_ids] = quaternion_yaw(
            self.robot.data.root_link_quat_w[env_ids]
        )
        self.segment_position[env_ids] = self.robot.data.root_link_pos_w[env_ids]
        self.mean_yaw_rate[env_ids] = 0.0
        self.vel_command_b[env_ids] = phase_commands(
            self.family[env_ids],
            self.first_sign[env_ids],
            phase,
            self.cfg.lateral_speed,
            self.cfg.longitudinal_speed,
            self.cfg.turn_rate,
        )
        self.vel_command_w[env_ids] = self.vel_command_b[env_ids]
        self.is_standing_env[env_ids] = (phase == 0) | (phase == 2)
        self.is_heading_env[env_ids] = False
        self.is_world_env[env_ids] = False
        self.is_forward_env[env_ids] = False
        self.time_left[env_ids] = torch.where(
            self.is_standing_env[env_ids], self.cfg.rest_seconds, self.cfg.move_seconds
        )

    def _update_command(self) -> None:
        # No heading controller, world-frame rotation, or artificial root velocity.
        # In particular a pure lateral sample always keeps vx == vyaw == 0.
        # Training-only measure of slow turning; do not punish the alternating yaw
        # wobble of otherwise straight steps as strongly as persistent rotation.
        alpha = min(1.0, self._env.step_dt / 0.25)
        self.mean_yaw_rate.lerp_(self.robot.data.root_link_ang_vel_b[:, 2], alpha)


@dataclass(kw_only=True)
class LateralVelocityCommandCfg(microduck_mdp.VelocityCommandCommandOnlyCfg):
    lateral_fraction: float = 0.7
    longitudinal_fraction: float = 0.15
    lateral_speed: float = 0.3
    longitudinal_speed: float = 0.3
    turn_rate: float = 0.6
    rest_seconds: float = 2.0
    move_seconds: float = 10.0
    deterministic: bool = False

    def build(self, env) -> LateralVelocityCommand:
        if self.rest_seconds <= 0 or self.move_seconds <= 0:
            raise ValueError("command phase durations must be positive")
        command_families(
            torch.empty(0), self.lateral_fraction, self.longitudinal_fraction
        )
        return LateralVelocityCommand(self, env)


def pure_lateral_mask(command: torch.Tensor) -> torch.Tensor:
    return (
        (command[:, 0].abs() < 1e-6)
        & (command[:, 1].abs() >= 0.05)
        & (command[:, 2].abs() < 1e-6)
    )


def lateral_progress_values(
    command: torch.Tensor, velocity: torch.Tensor, gravity: torch.Tensor
) -> torch.Tensor:
    """Signed achieved progress: stillness earns zero, wrong-way motion costs."""
    progress = (
        command[:, 1].sign() * velocity[:, 1] / command[:, 1].abs().clamp_min(0.05)
    ).clamp(-1.0, 1.0)
    upright = gravity[:, 2] < -0.85
    return progress * pure_lateral_mask(command) * upright


def lateral_velocity_progress(env, command_name: str = "twist") -> torch.Tensor:
    asset = env.scene["robot"]
    term = env.command_manager.get_term(command_name)
    velocity = segment_components(asset.data.root_link_lin_vel_w, term.segment_yaw)
    return lateral_progress_values(
        env.command_manager.get_command(command_name),
        velocity,
        asset.data.projected_gravity_b,
    )


def quaternion_yaw(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))


def segment_components(vector: torch.Tensor, heading: torch.Tensor) -> torch.Tensor:
    result = vector.clone()
    c, s = heading.cos(), heading.sin()
    result[:, 0] = c * vector[:, 0] + s * vector[:, 1]
    result[:, 1] = -s * vector[:, 0] + c * vector[:, 1]
    return result


def lateral_critic_state(env, command_name: str = "twist") -> torch.Tensor:
    """Expose reward/schedule memory only to the training critic (nine fields).

    Local pose and velocity alone cannot identify the accumulated segment error:
    two otherwise identical states may have different heading/path penalties.
    The remaining phase time, command family and initial sign also distinguish
    identical zero-command rest states with different upcoming commands. The
    deployed 61D actor receives none of these privileged quantities.
    """
    term = env.command_manager.get_term(command_name)
    data = env.scene["robot"].data
    angle = quaternion_yaw(data.root_link_quat_w) - term.segment_yaw
    displacement = segment_components(
        data.root_link_pos_w - term.segment_position, term.segment_yaw
    )
    # CommandTerm increments its counter AFTER _resample_command: counter 1
    # denotes the first resting phase, rather than the first moving phase.
    phase = (term.command_counter - 1).clamp_min(0).remainder(4)
    duration = torch.where(
        (phase == 0) | (phase == 2), term.cfg.rest_seconds, term.cfg.move_seconds
    )
    return torch.stack(
        (
            angle.sin(),
            angle.cos(),
            displacement[:, 0],
            displacement[:, 1],
            term.mean_yaw_rate,
            phase.float() / 3.0,
            (term.time_left / duration).clamp(0.0, 1.0),
            term.family.float() / 2.0,
            term.first_sign,
        ),
        dim=-1,
    )


def lateral_heading_drift_l2(
    env, command_name: str = "twist", scale: float = 0.1745329252
) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    target = env.command_manager.get_term(command_name).segment_yaw
    current = quaternion_yaw(env.scene["robot"].data.root_link_quat_w)
    error = torch.atan2(torch.sin(current - target), torch.cos(current - target))
    return pure_lateral_mask(command) * (error / scale).square().clamp_max(100)


def lateral_path_drift_l2(
    env, command_name: str = "twist", scale: float = 0.05
) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    term = env.command_manager.get_term(command_name)
    offset = env.scene["robot"].data.root_link_pos_w - term.segment_position
    displacement = segment_components(offset, term.segment_yaw)
    return pure_lateral_mask(command) * (displacement[:, 0] / scale).square().clamp_max(
        100
    )


def lateral_forward_drift_l2(
    env, command_name: str = "twist", scale: float = 0.1
) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    velocity = env.scene["robot"].data.root_link_lin_vel_b
    return pure_lateral_mask(command) * (velocity[:, 0] / scale).square().clamp_max(9)


def lateral_yaw_rate_l2(
    env, command_name: str = "twist", scale: float = 0.1
) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    yaw_rate = env.scene["robot"].data.root_link_ang_vel_b[:, 2]
    return pure_lateral_mask(command) * (yaw_rate / scale).square().clamp_max(9)


def lateral_mean_yaw_rate_l2(
    env, command_name: str = "twist", scale: float = 0.1
) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    rate = env.command_manager.get_term(command_name).mean_yaw_rate
    return pure_lateral_mask(command) * (rate / scale).square().clamp_max(100)


def make_lateral_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Keep v8's 61/14 actor and BAM/backlash, isolate side-step practice."""
    cfg = deepcopy(
        make_backlash_variant(
            make_microduck_velstand_env_cfg(play=play, rough=False),
            MICRODUCK_ALLCOLLISIONS_BACKLASH_ROBOT_CFG,
        )
    )
    cfg.commands["twist"] = LateralVelocityCommandCfg(
        entity_name="robot",
        resampling_time_range=(1.0, 4.0),
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.3, 0.3), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-0.6, 0.6)
        ),
        heading_command=False,
        rel_standing_envs=0,
        rel_heading_envs=0,
        rel_forward_envs=0,
        rel_world_envs=0,
        init_velocity_prob=0,
        deterministic=play,
    )
    # Only this independent task expands its critic from 76D to 85D. Original
    # v8 checkpoints/tasks and the exported actor's observation contract stay
    # unchanged. Old lateral critic checkpoints need a fresh critic after this
    # change; their actor weights can still be restored from an ONNX export.
    cfg.observations["critic"].terms["lateral_segment_state"] = ObservationTermCfg(
        func=lateral_critic_state
    )
    for name in ("head_pose", "body_pose"):
        term = cfg.commands[name]
        term.ranges = tuple((0.0, 0.0) for _ in term.ranges)
        term.zero_command_prob = 1.0

    # Rest-to-motion tests must not be bootstrapped by pushes or recovery spawns.
    for name in ("random_prone_init", "topple_push", "push_robot"):
        cfg.events.pop(name, None)
    cfg.events["reset_base"].params["pose_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.125, 0.125),
        "yaw": (0.0, 0.0),
    }
    cfg.events["reset_base"].params["velocity_range"] = {}
    cfg.terminations.pop("fallen_too_long", None)
    # A bounded locomotion fine-tune is not a get-up retrain; recycle failed starts.
    from mjlab.envs import mdp
    from mjlab.managers import TerminationTermCfg

    cfg.terminations["fell_over"] = TerminationTermCfg(
        func=mdp.bad_orientation, params={"limit_angle": 1.0471975511965976}
    )
    cfg.curriculum.clear()
    cfg.episode_length_s = 24.0
    # Match the v8 Rough checkpoint's solver while using flat terrain.
    cfg.sim.mujoco.iterations = 30
    cfg.sim.mujoco.ls_iterations = 50
    # Retain balance/contact terms, but do not pay a recovery/height bounty or
    # charge the fully tightened action-rate tax while exploring lateral onset.
    for name in (
        "upright_progress",
        "height_progress",
        "com_upward_velocity",
        "fallen_tax",
        "recovery_success",
        "gentle_rise",
    ):
        cfg.rewards[name].weight = 0.0
    cfg.rewards["action_rate_l2"].weight = -0.1
    cfg.rewards["head_pose_bias"].weight = 0.5
    cfg.rewards["lateral_progress"] = RewardTermCfg(
        func=lateral_velocity_progress, weight=3.0
    )
    cfg.rewards["lateral_forward_drift"] = RewardTermCfg(
        func=lateral_forward_drift_l2, weight=-0.2
    )
    cfg.rewards["lateral_yaw_rate"] = RewardTermCfg(
        func=lateral_yaw_rate_l2, weight=-0.05
    )
    cfg.rewards["lateral_mean_yaw_rate"] = RewardTermCfg(
        func=lateral_mean_yaw_rate_l2, weight=-0.5
    )
    cfg.rewards["lateral_heading_drift"] = RewardTermCfg(
        func=lateral_heading_drift_l2, weight=-0.5
    )
    cfg.rewards["lateral_path_drift"] = RewardTermCfg(
        func=lateral_path_drift_l2, weight=-0.15
    )
    if play:
        # Fix the command schedule and remove extra observation/model DR for
        # acceptance. Physical command/sensor delays still use the environment
        # seed; repeated evaluations must record that seed.
        cfg.observations["actor"].enable_corruption = False
        # The mounting transform is inside the observation functions, not their noise
        # wrappers. Disable it explicitly for nominal CPU-vs-Warp acceptance.
        cfg.observations["actor"].terms["base_ang_vel"].func = mdp.base_ang_vel
        cfg.observations["actor"].terms["base_ang_vel"].params = {}
        cfg.observations["actor"].terms[
            "projected_gravity"
        ].func = mdp.projected_gravity
        cfg.observations["actor"].terms["projected_gravity"].params = {}
        actuator = cfg.scene.entities["robot"].articulation.actuators[0]
        actuator.vin_range = (7.4, 7.4)
        actuator.vin_drop_gain_range = (0.1, 0.1)
        for name in tuple(cfg.events):
            if name.startswith("randomize_") or name in {
                "foot_friction",
                "encoder_bias",
                "base_com",
            }:
                cfg.events.pop(name)
    return cfg


MicroduckLateralRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 0.05,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True
    ),
    algorithm=PpoWithSymmetryCfg(
        learning_rate=1e-4,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        schedule="adaptive",
        desired_kl=0.01,
        symmetry_cfg=None,
    ),
    experiment_name="lateral",
    run_name="lateral_v8_warm_start",
    logger="tensorboard",
    upload_model=False,
    save_interval=25,
    num_steps_per_env=24,
    max_iterations=300,
)

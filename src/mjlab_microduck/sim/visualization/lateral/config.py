"""Public recipe facade with safe ordering for mjlab's task plugin loader."""

# Bootstrap the original task package before the recipe imports mjlab. During
# plugin initialization its hook imports env_cfg directly, avoiding this facade.
import mjlab_microduck.tasks  # noqa: F401 - initializes the task plugin

from .env_cfg import (
    LateralVelocityCommand,
    LateralVelocityCommandCfg,
    MicroduckLateralRlCfg,
    command_families,
    lateral_critic_state,
    lateral_forward_drift_l2,
    lateral_heading_drift_l2,
    lateral_mean_yaw_rate_l2,
    lateral_path_drift_l2,
    lateral_progress_values,
    lateral_velocity_progress,
    lateral_yaw_rate_l2,
    make_lateral_cfg,
    phase_commands,
    pure_lateral_mask,
    quaternion_yaw,
    segment_components,
)

__all__ = [
    "LateralVelocityCommand",
    "LateralVelocityCommandCfg",
    "MicroduckLateralRlCfg",
    "command_families",
    "lateral_critic_state",
    "lateral_forward_drift_l2",
    "lateral_heading_drift_l2",
    "lateral_mean_yaw_rate_l2",
    "lateral_path_drift_l2",
    "lateral_progress_values",
    "lateral_velocity_progress",
    "lateral_yaw_rate_l2",
    "make_lateral_cfg",
    "phase_commands",
    "pure_lateral_mask",
    "quaternion_yaw",
    "segment_components",
]

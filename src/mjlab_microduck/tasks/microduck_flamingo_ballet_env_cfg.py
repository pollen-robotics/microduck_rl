"""Microduck *flamingo ballet* — the cycle env with a choice of one-foot poses.

    twist command slots = [flag, side, pose]
        flag  ∈ {0, 1}      0 = two feet, 1 = one foot
        side  ∈ {-1, +1}    +1 = right stance (left leg lifted), -1 = left stance
        pose  ∈ {-1, 0, +1} -1 = arabesque (leg extended back, head up),
                             0 = passé (knee bent, foot lifted forward — the stage-1/2 pose),
                            +1 = développé devant (leg extended forward)

Poses come from the 2026-08-28 kinematic sweep (notes/poses/kine_right.json,
3° joint-limit margin, whole-robot CoM inside the stance sole's contact hull):
passé 1.21 cm margin / foot 13.6 cm up, développé 1.05 cm / 9.4 cm, arabesque
0.56 cm / 5.4 cm. Same stance leg in all three → same trunk-lean target. The
command term slews a pose blend u ∈ [-1, 1] (POSE_RAMP_S per unit) that the
joint / clearance targets interpolate over, so a pose switch during a hold is
a smooth transition the policy tracks; everything else is the cycle env.
"""

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import EventTermCfg, RewardTermCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_flamingo_cycle_env_cfg import (
    NUM_STEPS_PER_ENV,
    FLAMINGO_POSE_LEFT,
    FLAMINGO_POSE_RIGHT,
    MicroduckFlamingoCycleRlCfg,
    make_microduck_flamingo_cycle_env_cfg,
    mirror_pose,
)

# Right-stance poses (14-servo order). Swing leg = left (joints 0-4).
# Stance leg / neck / head_yaw identical to FLAMINGO_POSE; head_pitch differs.
DEVELOPPE_POSE_RIGHT = [
    0.000,  0.300,  1.200,  0.000,  0.800,     # left leg: extended forward, foot flat
    0.349,  0.350, -1.500,  0.000,
    0.386, -0.334,  0.258,  0.005, -0.253,
]
ARABESQUE_POSE_RIGHT = [
    0.000, -0.300, -1.200,  0.000, -0.800,     # left leg: extended back, toe pointed
    0.349, -0.500, -1.500,  0.000,             # head up (counterweight + line)
    0.386, -0.334,  0.258,  0.005, -0.253,
]
PASSE_POSE_RIGHT = list(FLAMINGO_POSE_RIGHT)

# pose id -1, 0, +1
POSES_RIGHT = [ARABESQUE_POSE_RIGHT, PASSE_POSE_RIGHT, DEVELOPPE_POSE_RIGHT]
POSES_LEFT = [mirror_pose(ARABESQUE_POSE_RIGHT), list(FLAMINGO_POSE_LEFT), mirror_pose(DEVELOPPE_POSE_RIGHT)]
CLEAR_TARGETS = (0.05, 0.10, 0.08)     # swing-foot height targets per pose (m)
CLEAR_TARGETS_V2 = (0.06, 0.10, 0.09)  # v2: the développé foot must clearly leave the floor
POSE_PROBS = (0.25, 0.5, 0.25)          # P(arabesque), P(passé), P(développé) per resample
POSE_RAMP_S = 1.5                       # seconds per unit of pose blend


def make_microduck_flamingo_ballet_env_cfg(play: bool = False, hard: bool = False, v2: bool = False) -> ManagerBasedRlEnvCfg:
    """``v2`` (after ballet run 1, 2026-08-29 05:20): run 1 held the arabesque but
    parked the développé foot ON the floor (tripod) and lifted every pose only
    ~4 cm — pose tracking beat clearance. v2 taxes the lifted-foot contact 3×,
    doubles the clearance reward with per-pose targets ≥ 6 cm, tightens the
    pose std a little, and compresses the curricula to a 2200-it run."""
    cfg = make_microduck_flamingo_cycle_env_cfg(play=play, hard=hard)

    cfg.rewards["pose_track"] = RewardTermCfg(
        func=microduck_mdp.flb_pose_track, weight=1.5,
        params={"command_name": "twist", "poses_right": POSES_RIGHT, "poses_left": POSES_LEFT,
                "std": 0.5, "mid_attenuation": 0.75},
    )
    clear = cfg.rewards["swing_foot_clear"]
    cfg.rewards["swing_foot_clear"] = RewardTermCfg(
        func=microduck_mdp.flb_swing_foot_clear, weight=clear.weight,
        params={"command_name": "twist", "left_cfg": clear.params["left_cfg"], "right_cfg": clear.params["right_cfg"],
                "sensor_name": clear.params["sensor_name"], "targets": CLEAR_TARGETS, "std": clear.params["std"],
                "lo": clear.params["lo"], "hi": clear.params["hi"]},
    )

    command = cfg.commands["twist"]
    cfg.commands["twist"] = microduck_mdp.FlamingoBalletCommandCfg(
        **{**vars(command), "pose_ramp_s": POSE_RAMP_S, "pose_probs": POSE_PROBS}
    )

    ev = cfg.events.pop("set_flamingo_cycle_state")
    params = {k: v for k, v in ev.params.items() if k not in ("pose_right", "pose_left")}
    cfg.events["set_flamingo_ballet_state"] = EventTermCfg(
        func=microduck_mdp.set_flamingo_ballet_state, mode="reset",
        params={**params, "poses_right": POSES_RIGHT, "poses_left": POSES_LEFT, "pose_probs": POSE_PROBS},
    )
    cfg.curriculum["in_pose_prob"].params["event_name"] = "set_flamingo_ballet_state"
    # Gentler pacing than the cycle runs (both cycle runs stepped DOWN in reward /
    # episode length at the in-pose 0.4 → 0.3 step and at the 0.25 m/s push stage):
    # three poses to learn → slower spawn-mix ramp, later pushes, no 0.25 stage.
    cfg.curriculum["in_pose_prob"].params["param_stages"] = [
        {"step": 0,                                   "params": {"in_pose_prob": 0.6}},
        {"step": 700 * NUM_STEPS_PER_ENV,             "params": {"in_pose_prob": 0.5}},
        {"step": 1400 * NUM_STEPS_PER_ENV,            "params": {"in_pose_prob": 0.4}},
        {"step": 2100 * NUM_STEPS_PER_ENV,            "params": {"in_pose_prob": 0.3}},
    ]
    if not hard:
        cfg.curriculum["push_magnitude"].params["push_stages"] = [
            {"step": 0,                               "velocity_range": {"x": (0.0, 0.0),    "y": (0.0, 0.0)}},
            {"step": 600 * NUM_STEPS_PER_ENV,         "velocity_range": {"x": (-0.08, 0.08), "y": (-0.08, 0.08)}},
            {"step": 1200 * NUM_STEPS_PER_ENV,        "velocity_range": {"x": (-0.15, 0.15), "y": (-0.15, 0.15)}},
        ]
    if v2:
        cfg.rewards["swing_foot_contact"].weight = 1.5
        cfg.rewards["swing_foot_clear"].weight = 3.0
        cfg.rewards["swing_foot_clear"].params["targets"] = CLEAR_TARGETS_V2
        cfg.rewards["swing_foot_clear"].params["std"] = 0.05
        cfg.rewards["pose_track"].params["std"] = 0.4
        cfg.curriculum["in_pose_prob"].params["param_stages"] = [
            {"step": 0,                               "params": {"in_pose_prob": 0.6}},
            {"step": 500 * NUM_STEPS_PER_ENV,         "params": {"in_pose_prob": 0.5}},
            {"step": 1000 * NUM_STEPS_PER_ENV,        "params": {"in_pose_prob": 0.4}},
            {"step": 1500 * NUM_STEPS_PER_ENV,        "params": {"in_pose_prob": 0.3}},
        ]
        cfg.curriculum["push_magnitude"].params["push_stages"] = [
            {"step": 0,                               "velocity_range": {"x": (0.0, 0.0),    "y": (0.0, 0.0)}},
            {"step": 500 * NUM_STEPS_PER_ENV,         "velocity_range": {"x": (-0.08, 0.08), "y": (-0.08, 0.08)}},
            {"step": 1000 * NUM_STEPS_PER_ENV,        "velocity_range": {"x": (-0.15, 0.15), "y": (-0.15, 0.15)}},
        ]
        cfg.curriculum["action_rate_weight"].params["weight_stages"] = [
            {"step": 0,                               "weight": -0.1},
            {"step": 500 * NUM_STEPS_PER_ENV,         "weight": -0.5},
        ]
    return cfg


MicroduckFlamingoBalletRlCfg = deepcopy(MicroduckFlamingoCycleRlCfg)
MicroduckFlamingoBalletRlCfg.experiment_name = "flamingo_ballet"
MicroduckFlamingoBalletRlCfg.run_name = "flamingo_ballet"

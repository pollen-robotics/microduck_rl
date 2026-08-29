"""Visible standard-height staircase scene for MicroDuck.

The low-rise task is the learnable progression for this 25 cm robot.  This
scene is intentionally a separate challenge: it builds a conventional straight
flight with uniform 170 mm risers and 280 mm treads, a flat approach, and a top
landing.  The staircase is in front of each robot's spawn point, so the native
viewer shows what the robot must climb instead of hiding the stairs under it.
"""

from copy import deepcopy
from dataclasses import dataclass

import mujoco
import numpy as np
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.terrains.terrain_generator import (
    SubTerrainCfg,
    TerrainGeneratorCfg,
    TerrainGeometry,
    TerrainOutput,
)

from mjlab_microduck.robot.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    MicroduckRlCfg,
    _soften_terrain_contacts,
    make_microduck_velocity_env_cfg,
)
from mjlab_microduck.tasks.stair_action import with_stair_history_seed
from mjlab_microduck.tasks.stair_walk_state_bank import WalkerStateBankReset

STANDARD_RISER_HEIGHT = 0.17
STANDARD_TREAD_DEPTH = 0.28
STANDARD_STAIR_WIDTH = 0.90
STANDARD_NUM_STEPS = 5
STANDARD_APPROACH_LENGTH = 1.20
STANDARD_TOP_PLATFORM_LENGTH = 0.90
STANDARD_SPAWN_X = STANDARD_APPROACH_LENGTH * 0.45
STANDARD_STAIR_START_DISTANCE = STANDARD_APPROACH_LENGTH - STANDARD_SPAWN_X
STANDARD_GOAL_X = (
    STANDARD_APPROACH_LENGTH
    + STANDARD_NUM_STEPS * STANDARD_TREAD_DEPTH
    + STANDARD_TOP_PLATFORM_LENGTH
)
STANDARD_GOAL_DISTANCE = STANDARD_GOAL_X - STANDARD_SPAWN_X
# Measured standing trunk height for this robot, added to the top landing
# height so the goal describes the robot standing on the landing.
STANDARD_STANDING_ROOT_HEIGHT = 0.115
STANDARD_TOP_ROOT_HEIGHT = (
    STANDARD_NUM_STEPS * STANDARD_RISER_HEIGHT + STANDARD_STANDING_ROOT_HEIGHT
)
ROUTE_MIN_RISER_HEIGHT = 0.010
ROUTE_CURRICULUM_LEVELS = 16
STAIR_MECHANISM_MIN_RISER_HEIGHT = 0.10
STAIR_MECHANISM_CURRICULUM_LEVELS = 8


@dataclass(kw_only=True)
class BoxStandardStaircaseTerrainCfg(SubTerrainCfg):
    """A straight staircase with a visible flat run-up and top landing."""

    riser_height: float = STANDARD_RISER_HEIGHT
    tread_depth: float = STANDARD_TREAD_DEPTH
    stair_width: float = STANDARD_STAIR_WIDTH
    num_steps: int = STANDARD_NUM_STEPS
    approach_length: float = STANDARD_APPROACH_LENGTH
    top_platform_length: float = STANDARD_TOP_PLATFORM_LENGTH
    floor_thickness: float = 0.10
    riser_height_range: tuple[float, float] | None = None
    difficulty_levels: int | None = None

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        del rng
        body = spec.body("terrain")
        size_x, size_y = self.size
        center_y = size_y / 2.0
        riser_height = self.riser_height
        if self.riser_height_range is not None:
            low, high = self.riser_height_range
            normalized_difficulty = float(np.clip(difficulty, 0.0, 1.0))
            if self.difficulty_levels is not None and self.difficulty_levels > 1:
                level = min(
                    int(normalized_difficulty * self.difficulty_levels),
                    self.difficulty_levels - 1,
                )
                normalized_difficulty = level / (self.difficulty_levels - 1)
            riser_height = low + normalized_difficulty * (high - low)
        total_height = self.num_steps * riser_height
        boxes: list[mujoco.MjsGeom] = []
        colors: list[tuple[float, float, float, float]] = []

        # A continuous floor leaves a clearly visible flat approach and side
        # area around the stairs, while the step boxes provide uniform risers.
        floor = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(size_x / 2.0, size_y / 2.0, self.floor_thickness / 2.0),
            pos=(size_x / 2.0, center_y, -self.floor_thickness / 2.0),
        )
        boxes.append(floor)
        colors.append((0.18, 0.22, 0.29, 1.0))

        for step_index in range(self.num_steps):
            height = (step_index + 1) * riser_height
            x_start = self.approach_length + step_index * self.tread_depth
            step = body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=(
                    self.tread_depth / 2.0,
                    self.stair_width / 2.0,
                    height / 2.0,
                ),
                pos=(x_start + self.tread_depth / 2.0, center_y, height / 2.0),
            )
            boxes.append(step)
            colors.append(
                (
                    0.20 + 0.045 * step_index,
                    0.42 + 0.035 * step_index,
                    0.78 - 0.035 * step_index,
                    1.0,
                )
            )

        landing = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(self.top_platform_length / 2.0, self.stair_width / 2.0, total_height / 2.0),
            pos=(
                self.approach_length
                + self.num_steps * self.tread_depth
                + self.top_platform_length / 2.0,
                center_y,
                total_height / 2.0,
            ),
        )
        boxes.append(landing)
        colors.append((0.78, 0.57, 0.18, 1.0))

        # Spawn on the approach, centered with the flight.  The staircase is
        # therefore in front of the robot in the viewer and in the task.
        origin = np.array((self.approach_length * 0.45, center_y, 0.0))
        geometries = [
            TerrainGeometry(geom=box, color=color)
            for box, color in zip(boxes, colors, strict=True)
        ]
        return TerrainOutput(origin=origin, geometries=geometries)


STANDARD_STAIR_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(5.0, 3.0),
    border_width=0.15,
    num_rows=2,
    num_cols=2,
    curriculum=True,
    sub_terrains={
        # Two columns keep the generated viewer scene broad enough to place
        # four visible environments in a 2x2 arrangement.
        "standard_stairs_a": BoxStandardStaircaseTerrainCfg(proportion=0.5),
        "standard_stairs_b": BoxStandardStaircaseTerrainCfg(proportion=0.5),
    },
    add_lights=True,
)


ROUTE_STAIR_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(5.0, 3.0),
    border_width=0.15,
    num_rows=ROUTE_CURRICULUM_LEVELS,
    num_cols=1,
    curriculum=True,
    sub_terrains={
        "route_stairs": BoxStandardStaircaseTerrainCfg(
            proportion=1.0,
            riser_height_range=(ROUTE_MIN_RISER_HEIGHT, STANDARD_RISER_HEIGHT),
            difficulty_levels=ROUTE_CURRICULUM_LEVELS,
        ),
    },
    add_lights=True,
)


def make_microduck_standard_stairs_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create the standard staircase challenge and optional viewer layout."""
    cfg = make_microduck_velocity_env_cfg(rough=True)
    cfg.scene.terrain.terrain_generator = deepcopy(STANDARD_STAIR_TERRAINS_CFG)
    cfg.scene.spec_fn = _soften_terrain_contacts
    cfg.episode_length_s = 12.0
    cfg.sim.nconmax = 300
    cfg.sim.mujoco.iterations = 35
    cfg.sim.mujoco.ls_iterations = 60
    cfg.scene.terrain.max_init_terrain_level = 0 if play else None

    # A fixed forward command makes the objective legible: walk from the
    # approach to the top landing.  Turning and standing buckets are disabled
    # for this dedicated challenge, while the 61D command observation layout
    # remains unchanged.
    command = deepcopy(cfg.commands["twist"])
    command.rel_standing_envs = 0.0
    command.rel_turn_in_place_envs = 0.0
    command.rel_heading_envs = 0.0
    command.heading_command = False
    command.ranges.heading = None
    command.ranges.lin_vel_x = (0.10, 0.30)
    command.ranges.lin_vel_y = (-0.04, 0.04)
    command.ranges.ang_vel_z = (-0.20, 0.20)
    cfg.commands["twist"] = microduck_mdp.VelocityCommandCommandOnlyCfg(
        **vars(command)
    )

    # The main objective is reaching the top, not merely walking near the
    # staircase.  Progress and climb shaping are potentials; the success term
    # is latched and pays once when the robot arrives upright on the landing.
    cfg.rewards["stair_goal_progress"] = RewardTermCfg(
        func=microduck_mdp.stair_goal_progress,
        weight=2.5,
        params={
            "goal_distance": STANDARD_GOAL_DISTANCE,
            "corridor_half_width": STANDARD_STAIR_WIDTH * 0.44,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["stair_top_approach"] = RewardTermCfg(
        func=microduck_mdp.stair_top_approach,
        weight=2.0,
        params={
            "goal_distance": STANDARD_GOAL_DISTANCE,
            "goal_height": STANDARD_TOP_ROOT_HEIGHT,
            "start_height": STANDARD_STANDING_ROOT_HEIGHT,
            "corridor_half_width": STANDARD_STAIR_WIDTH * 0.44,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["stair_top_goal"] = RewardTermCfg(
        func=microduck_mdp.stair_top_goal,
        weight=12.0,
        params={
            "goal_distance": STANDARD_GOAL_DISTANCE,
            "goal_height": STANDARD_TOP_ROOT_HEIGHT,
            "x_tolerance": 0.40,
            "z_tolerance": 0.13,
            "upright_threshold": 0.78,
            "corridor_half_width": STANDARD_STAIR_WIDTH * 0.44,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # Keep the pretrained flat walking behavior useful on the approach, then
    # smoothly release velocity/turn tracking at the first riser.  This is the
    # policy-level switch: the same actor walks to the obstacle, and the stair
    # terms become dominant exactly where creative contact strategies matter.
    for reward_name, function in (
        ("track_linear_velocity", microduck_mdp.stair_approach_linear_tracking),
        ("track_angular_velocity", microduck_mdp.stair_approach_angular_tracking),
    ):
        cfg.rewards[reward_name].func = function
        cfg.rewards[reward_name].params["stair_start_distance"] = (
            STANDARD_STAIR_START_DISTANCE
        )
        cfg.rewards[reward_name].params["fade_distance"] = 0.20

    # Remove random lateral/yaw starts so the route is reproducible in the
    # viewer and the policy receives a clean forward-first curriculum.
    pose_range = cfg.events["reset_base"].params["pose_range"]
    pose_range["x"] = (0.0, 0.0)
    pose_range["y"] = (0.0, 0.0)
    pose_range["yaw"] = (0.0, 0.0)
    pose_range["z"] = (0.12, 0.12)
    if play:
        # Procedural terrain is already tiled by rows (x) and columns (y).
        # The startup event below pins four envs to (row, col) =
        # (0,0), (1,0), (0,1), (1,1), avoiding the default random reuse of a
        # patch that can make a four-robot viewer look stacked or overlapping.
        cfg.scene.terrain.terrain_generator.num_rows = 2
        cfg.scene.terrain.terrain_generator.num_cols = 2
        cfg.scene.terrain.max_init_terrain_level = 1
        cfg.curriculum.pop("terrain_levels", None)
        cfg.events["stair_viewer_grid"] = EventTermCfg(
            func=microduck_mdp.configure_stair_viewer_grid,
            mode="startup",
            params={
                "terrain_levels": (0, 1, 0, 1),
                "terrain_types": (0, 0, 1, 1),
            },
        )
    return cfg


def make_microduck_route_stairs_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create a flat-runway-to-stairs curriculum for one shared actor."""
    cfg = make_microduck_standard_stairs_env_cfg(play=False)
    # The manufacturer's walking model collides only at the feet. Contact-rich
    # ascent needs the full model so the head, shell, hips, and legs can push
    # against a riser instead of passing through it.
    cfg.scene.entities = {"robot": MICRODUCK_STANDUP_ROBOT_CFG}
    head_contact = ContactSensorCfg(
        name="head_ground_contact",
        primary=ContactMatch(mode="body", pattern="jaw_soft", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force", "pos", "normal", "tangent"),
        reduce="maxforce",
        num_slots=4,
        global_frame=True,
    )
    robot_contact = ContactSensorCfg(
        name="robot_ground_contact",
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force", "pos", "normal", "tangent"),
        reduce="maxforce",
        num_slots=4,
        global_frame=True,
    )
    trunk_contact = ContactSensorCfg(
        name="trunk_ground_contact",
        primary=ContactMatch(mode="body", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force", "pos", "normal", "tangent"),
        reduce="maxforce",
        num_slots=4,
        global_frame=True,
    )
    leg_contact = ContactSensorCfg(
        name="legs_ground_contact",
        primary=ContactMatch(
            mode="body",
            pattern=r"^(hip_l|hip_l_2|leg|leg_2)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force", "pos", "normal", "tangent"),
        reduce="maxforce",
        num_slots=2,
        global_frame=True,
    )
    feet_stair_contact = ContactSensorCfg(
        name="feet_stair_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=r"^(left_foot_collision|right_foot_collision)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force", "pos", "normal", "tangent"),
        reduce="maxforce",
        num_slots=2,
        global_frame=True,
    )
    cfg.scene.sensors = tuple(cfg.scene.sensors) + (
        head_contact,
        robot_contact,
        trunk_contact,
        leg_contact,
        feet_stair_contact,
    )
    cfg.scene.terrain.terrain_generator = deepcopy(ROUTE_STAIR_TERRAINS_CFG)
    cfg.scene.terrain.max_init_terrain_level = 0
    cfg.episode_length_s = 30.0

    command = cfg.commands["twist"]
    command.ranges.lin_vel_x = (0.20, 0.28)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (0.0, 0.0)

    dynamic_height = {
        "goal_height_range": (
            STANDARD_STANDING_ROOT_HEIGHT + STANDARD_NUM_STEPS * ROUTE_MIN_RISER_HEIGHT,
            STANDARD_TOP_ROOT_HEIGHT,
        ),
        "num_terrain_levels": ROUTE_CURRICULUM_LEVELS,
    }
    approach_params = cfg.rewards["stair_top_approach"].params
    approach_params.pop("goal_height", None)
    approach_params.update(dynamic_height)
    # Height and route progress should credit a jump, mantle, knee/chest push,
    # or head-supported lever. Upright posture is required only at the final
    # landing, not during the maneuver itself.
    approach_params["upright_power"] = 0.0
    goal_params = cfg.rewards["stair_top_goal"].params
    goal_params.pop("goal_height", None)
    goal_params.update(dynamic_height)
    goal_params["z_tolerance"] = 0.08
    # RewardManager scales one-shot terms by the 0.02 s control step. This
    # produces the intended 12-point terminal breakthrough instead of 0.24.
    cfg.rewards["stair_top_goal"].weight = 600.0
    cfg.rewards["stair_first_riser_clearance"] = RewardTermCfg(
        func=microduck_mdp.stair_first_riser_clearance,
        # Likewise, 150 * 0.02 is a true 3-point first-riser milestone.
        weight=150.0,
        params={
            "stair_start_distance": STANDARD_STAIR_START_DISTANCE,
            "min_riser_height": ROUTE_MIN_RISER_HEIGHT,
            "max_riser_height": STANDARD_RISER_HEIGHT,
            "num_terrain_levels": ROUTE_CURRICULUM_LEVELS,
            "corridor_half_width": STANDARD_STAIR_WIDTH * 0.40,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # Preserve the manufacturer's gait on the runway, then leave room for
    # contact-rich motion at the obstacle. These stock locomotion terms would
    # otherwise directly suppress the high feet, rotation, airtime, and head
    # articulation needed to clear a 170 mm riser.
    cfg.rewards["upright"].func = microduck_mdp.stair_approach_upright
    cfg.rewards["upright"].params["stair_start_distance"] = (
        STANDARD_STAIR_START_DISTANCE
    )
    cfg.rewards["upright"].params["fade_distance"] = 0.20
    cfg.rewards["upright"].weight = 2.0
    cfg.rewards["pose"].weight = 0.0
    cfg.rewards["body_ang_vel"].weight = 0.0
    cfg.rewards["angular_momentum"].weight = 0.0
    cfg.rewards["dof_pos_limits"].weight = 0.0
    cfg.rewards["action_rate_l2"].weight = -0.001
    cfg.rewards["air_time"].weight = 0.0
    cfg.rewards["foot_clearance"].weight = 0.0
    cfg.rewards["foot_swing_height"].weight = 0.0
    cfg.rewards["foot_slip"].weight = 0.0
    cfg.rewards["self_collisions"].weight = 0.0
    cfg.rewards["head_pose_tracking"].weight = 0.0
    cfg.rewards["head_pose_bias"].weight = 0.0

    # A forward fall can be part of the solution. Timeout, bounds, and NaN
    # guards remain; the final success reward still demands upright recovery.
    cfg.terminations.pop("fell_over", None)

    # Freeze command-side pose targets and robustness randomization during
    # strategy discovery. Once a standard-height solution exists, those can be
    # restored for sim-to-real hardening without changing the objective.
    cfg.commands["head_pose"].ranges = ((0.0, 0.0),) * 4
    cfg.commands["body_pose"].ranges = ((0.0, 0.0),) * 6
    route_cue_params = {
        "stair_start_distance": STANDARD_STAIR_START_DISTANCE,
        "goal_distance": STANDARD_GOAL_DISTANCE,
        "min_riser_height": ROUTE_MIN_RISER_HEIGHT,
        "max_riser_height": STANDARD_RISER_HEIGHT,
        "num_terrain_levels": ROUTE_CURRICULUM_LEVELS,
        "tread_depth": STANDARD_TREAD_DEPTH,
        "num_steps": STANDARD_NUM_STEPS,
        # Give the post-walk specialist roughly 260 mm of physical runway to
        # preload and launch. The immutable walker still receives zeroes in
        # this command slice, so its manufacturer gait is unchanged.
        "cue_distance": 0.30,
    }
    for observation_group in ("actor", "critic"):
        cfg.observations[observation_group].terms["body_command"].func = (
            microduck_mdp.stair_route_cues
        )
        cfg.observations[observation_group].terms["body_command"].params = (
            route_cue_params
        )
    for curriculum_name in (
        "action_rate_weight",
        "standing_envs",
        "head_pose_range",
        "body_pose_range",
        "com_range",
        "head_com_range",
        "head_pose_bias_weight",
    ):
        cfg.curriculum.pop(curriculum_name, None)
    for event_name in (
        "randomize_com",
        "randomize_head_com",
        "randomize_joint_friction",
        "randomize_armature",
        "randomize_mass_inertia",
    ):
        cfg.events.pop(event_name, None)

    # Most environments advance through the learnable height curriculum, but
    # a standing fraction always faces the true 170 mm target. This prevents a
    # locally successful low-step gait from becoming the entire objective.
    challenge_event = EventTermCfg(
        func=microduck_mdp.seed_route_challenge_levels,
        mode="reset",
        params={"standard_fraction": 1.0 if play else 0.35},
    )
    cfg.events = {"route_challenge_levels": challenge_event, **cfg.events}
    cfg.events["route_state_curriculum"] = EventTermCfg(
        func=microduck_mdp.reset_route_learning_states,
        mode="reset",
        params={
            "stair_start_distance": STANDARD_STAIR_START_DISTANCE,
            "min_riser_height": ROUTE_MIN_RISER_HEIGHT,
            "max_riser_height": STANDARD_RISER_HEIGHT,
            "num_terrain_levels": ROUTE_CURRICULUM_LEVELS,
            "tread_depth": STANDARD_TREAD_DEPTH,
            "num_steps": STANDARD_NUM_STEPS,
            "standing_root_height": STANDARD_STANDING_ROOT_HEIGHT,
            "near_face_fraction": 0.0 if play else 0.20,
            "partial_mantle_fraction": 0.0 if play else 0.20,
            "on_tread_fraction": 0.0 if play else 0.10,
        },
    )
    cfg.curriculum["terrain_levels"].func = microduck_mdp.route_terrain_levels
    cfg.curriculum["terrain_levels"].params = {}

    # Focus the first curriculum on the route itself. Random pushes can be
    # restored after the robot reliably reaches the landing.
    cfg.events.pop("push_robot", None)
    if play:
        cfg.curriculum.pop("terrain_levels", None)
    return cfg


def make_microduck_stair_specialist_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Train only the post-handoff specialist on the full home staircase."""
    cfg = make_microduck_route_stairs_env_cfg(play=False)
    cfg.episode_length_s = 8.0
    cfg.actions["joint_pos"] = with_stair_history_seed(cfg.actions["joint_pos"])

    # Keep the procedural row layout used by route cues, but make every row
    # the same physical five-step, 170 mm staircase. There is no miniature
    # terrain and no height curriculum in this specialist experiment.
    terrain_generator = cfg.scene.terrain.terrain_generator
    for terrain_cfg in terrain_generator.sub_terrains.values():
        terrain_cfg.riser_height = STANDARD_RISER_HEIGHT
        terrain_cfg.riser_height_range = None
        terrain_cfg.difficulty_levels = None
    cfg.events["route_challenge_levels"].params["standard_fraction"] = 1.0
    cfg.curriculum.pop("terrain_levels", None)

    # Every episode starts in the specialist's domain: a real handoff window,
    # a partial shell/head mantle, or recovery on the first full-height tread.
    # Distant runway starts belong exclusively to the immutable walker.
    reset_params = cfg.events["route_state_curriculum"].params
    reset_params.update(
        {
            "near_face_fraction": 0.50,
            "partial_mantle_fraction": 0.30,
            "on_tread_fraction": 0.20,
            "min_tread_step": 1,
            "max_tread_step": 1,
        }
    )

    # Discovery is driven by an unfarmable first-riser frontier and a durable
    # one-shot clearance. The distant five-step height potential is silent
    # until the first 170 mm transition is actually learned.
    cfg.rewards["track_linear_velocity"].weight = 0.0
    cfg.rewards["track_angular_velocity"].weight = 0.0
    cfg.rewards["upright"].weight = 0.0
    cfg.rewards["action_rate_l2"].weight = 0.0
    cfg.rewards["stair_goal_progress"].weight = 0.5
    cfg.rewards["stair_top_approach"].weight = 0.0
    cfg.rewards["stair_first_riser_clearance"].weight = 200.0
    cfg.rewards["stair_first_riser_frontier"] = RewardTermCfg(
        func=microduck_mdp.stair_first_riser_frontier,
        weight=4.0,
        params={
            "stair_start_distance": STANDARD_STAIR_START_DISTANCE,
            "standing_root_height": STANDARD_STANDING_ROOT_HEIGHT,
            "riser_height": STANDARD_RISER_HEIGHT,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    if play:
        # Specialist-only play begins at the real handoff window. Full-route
        # evaluation uses the separate hard walker-to-specialist dispatcher.
        reset_params.update(
            {
                "near_face_fraction": 1.0,
                "partial_mantle_fraction": 0.0,
                "on_tread_fraction": 0.0,
            }
        )
    return cfg


def make_microduck_assisted_stair_specialist_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Stage A: discover a real 170 mm mantle from full-height contact states."""
    cfg = make_microduck_stair_specialist_env_cfg(play=False)
    cfg.episode_length_s = 4.0
    cfg.events.pop("route_challenge_levels", None)
    cfg.events["route_state_curriculum"] = EventTermCfg(
        func=microduck_mdp.reset_assisted_stair_states,
        mode="reset",
        params={
            "stair_start_distance": STANDARD_STAIR_START_DISTANCE,
            "standing_root_height": STANDARD_STANDING_ROOT_HEIGHT,
            "lip_release_fraction": 0.50,
            "shell_brace_fraction": 0.25,
            "tread_recovery_fraction": 0.15,
            "real_handoff_fraction": 0.10,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # The terrain and all reward thresholds are literal full-size dimensions.
    # No terrain-level state can silently turn this into a low-stair task.
    for observation_group in ("actor", "critic"):
        route_cues = (
            cfg.observations[observation_group].terms["body_command"].params
        )
        route_cues["min_riser_height"] = STANDARD_RISER_HEIGHT
        route_cues["max_riser_height"] = STANDARD_RISER_HEIGHT

    for reward in cfg.rewards.values():
        reward.weight = 0.0
    cfg.rewards["stair_assisted_approach"] = RewardTermCfg(
        func=microduck_mdp.stair_assisted_approach_frontier,
        weight=0.5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    cfg.rewards["stair_assisted_lift"] = RewardTermCfg(
        func=microduck_mdp.stair_assisted_lift_frontier,
        weight=3.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    cfg.rewards["stair_assisted_crossing"] = RewardTermCfg(
        func=microduck_mdp.stair_assisted_crossing_frontier,
        weight=5.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    clearance = cfg.rewards["stair_first_riser_clearance"]
    clearance.weight = 400.0
    clearance.params.update(
        {
            "riser_height": STANDARD_RISER_HEIGHT,
            "x_margin": 0.04,
            "z_margin": 0.020,
            "max_vertical_speed": 0.60,
            "hold_time_s": 0.08,
        }
    )
    cfg.rewards["stair_first_tread_stable"] = RewardTermCfg(
        func=microduck_mdp.stair_first_tread_stable,
        weight=200.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    del play
    return cfg


def make_microduck_stair_bridge_specialist_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Stage A2: connect lip clearance to a secured full-height tread landing."""
    cfg = make_microduck_assisted_stair_specialist_env_cfg(play=False)
    reset = cfg.events["route_state_curriculum"].params
    reset.update(
        {
            "lip_release_fraction": 0.30,
            "shell_brace_fraction": 0.40,
            "tread_recovery_fraction": 0.25,
            "real_handoff_fraction": 0.05,
        }
    )
    clearance = cfg.rewards["stair_first_riser_clearance"]
    clearance.weight = 300.0
    clearance.params.update(
        {
            "z_margin": 0.025,
            "max_vertical_speed": 0.45,
            "hold_time_s": 0.12,
        }
    )
    cfg.rewards["stair_assisted_lift"].weight = 4.0
    cfg.rewards["stair_assisted_crossing"].weight = 6.0
    cfg.rewards["stair_first_tread_stable"].weight = 50.0
    cfg.rewards["stair_first_tread_secured"] = RewardTermCfg(
        func=microduck_mdp.stair_first_tread_secured,
        weight=300.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    cfg.rewards["stair_first_tread_settle_quality"] = RewardTermCfg(
        func=microduck_mdp.stair_first_tread_settle_quality,
        weight=2.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    del play
    return cfg


def make_microduck_stair_walker_bank_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Stage A3: train from exact immutable-walker states at the real stair face."""

    cfg = make_microduck_stair_bridge_specialist_env_cfg(play=False)
    reset = cfg.events["route_state_curriculum"].params
    reset.update(
        {
            "lip_release_fraction": 0.20,
            "shell_brace_fraction": 0.25,
            "tread_recovery_fraction": 0.25,
            "real_handoff_fraction": 0.30,
        }
    )
    # Appending this class event is order-sensitive. It replaces only mode 3
    # after the assisted reset selects a family, then its reset() hook restores
    # action, command, and observation history after mjlab clears the managers.
    cfg.events["walker_state_bank"] = EventTermCfg(
        func=WalkerStateBankReset,
        mode="reset",
        params={"bank_path": ".tmp/codex/full170-walker-state-bank.pt"},
    )
    del play
    return cfg


def make_microduck_stair_launch_bank_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Stage A4: learn preload and takeoff from an earlier real walker state."""

    cfg = make_microduck_stair_walker_bank_env_cfg(play=False)
    cfg.episode_length_s = 6.0
    reset = cfg.events["route_state_curriculum"].params
    reset.update(
        {
            "lip_release_fraction": 0.15,
            "shell_brace_fraction": 0.15,
            "tread_recovery_fraction": 0.10,
            "real_handoff_fraction": 0.60,
        }
    )
    cfg.events["walker_state_bank"] = EventTermCfg(
        func=WalkerStateBankReset,
        mode="reset",
        params={"bank_path": ".tmp/codex/full170-walker-launch-state-bank.pt"},
    )

    # Direct-collocation-inspired milestones: first load the legs, then create
    # upward impulse, then let the existing unconstrained lift/cross/support
    # gates judge any jump, roll, shell mantle, or head-lever solution.
    cfg.rewards["stair_assisted_approach"].params.update(
        {"start_x": 0.40, "end_x": 0.58}
    )
    cfg.rewards["stair_preload_frontier"] = RewardTermCfg(
        func=microduck_mdp.stair_preload_frontier,
        weight=2.0,
        params={
            "start_x": 0.40,
            "end_x": 0.60,
            "standing_root_height": STANDARD_STANDING_ROOT_HEIGHT,
            "target_root_height": 0.092,
            "corridor_half_width": STANDARD_STAIR_WIDTH * 0.40,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["stair_launch_sequence"] = RewardTermCfg(
        func=microduck_mdp.stair_launch_sequence,
        weight=50.0,
        params={
            "min_x": 0.46,
            "max_x": STANDARD_STAIR_START_DISTANCE + 0.02,
            "preload_root_height": 0.098,
            "min_upward_speed": 0.30,
            "corridor_half_width": STANDARD_STAIR_WIDTH * 0.40,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["stair_takeoff_frontier"] = RewardTermCfg(
        func=microduck_mdp.stair_takeoff_frontier,
        weight=3.0,
        params={
            "min_x": 0.46,
            "max_x": STANDARD_STAIR_START_DISTANCE + 0.02,
            "target_vertical_speed": 1.20,
            "corridor_half_width": STANDARD_STAIR_WIDTH * 0.40,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["stair_assisted_lift"].params.update(
        {"start_height": 0.095, "clearance_height": 0.205, "x_gate": 0.52}
    )
    cfg.rewards["stair_assisted_crossing"].params.update(
        {"start_x": 0.60, "end_x": 0.72, "clearance_height": 0.190}
    )
    cfg.rewards["stair_first_tread_stable"].weight = 100.0
    cfg.rewards["stair_first_tread_secured"].weight = 250.0
    del play
    return cfg


def make_microduck_stair_apex_mantle_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Stage A5: discover a jump or head/shell mantle from a real handoff."""

    cfg = make_microduck_stair_launch_bank_env_cfg(play=False)
    reset = cfg.events["route_state_curriculum"].params
    reset.update(
        {
            # Mode 0 is now an imperfect launch release, mode 1 is a
            # non-penetrating head-lever seed, and mode 3 remains the only
            # unassisted state family that can validate transfer.
            "lip_release_fraction": 0.25,
            "shell_brace_fraction": 0.15,
            "tread_recovery_fraction": 0.0,
            "real_handoff_fraction": 0.60,
            "lip_local_x_range": (0.515, 0.555),
            "lip_root_height_range": (0.105, 0.125),
            "lip_pitch_deg_range": (-5.0, 15.0),
            "lip_forward_speed_range": (0.20, 0.35),
            "lip_vertical_speed_range": (0.45, 0.90),
            "lip_pitch_rate_range": (0.0, 1.5),
            "shell_local_x_range": (0.540, 0.590),
            "shell_root_height_range": (0.110, 0.145),
            "shell_pitch_deg_range": (8.0, 30.0),
            "shell_forward_speed_range": (0.08, 0.18),
            "shell_vertical_speed_range": (0.0, 0.10),
            "shell_pitch_rate_range": (0.3, 1.2),
        }
    )

    # The failed A4 run proved that prescribing a crouch did not transfer to
    # the immutable walker. Pay new physical capability instead: enough
    # predicted apex to clear the lip, or real head/shell contact that advances
    # height and crossing. The remaining gates still require actual clearance.
    cfg.rewards["stair_preload_frontier"].weight = 0.0
    cfg.rewards["stair_launch_sequence"].weight = 0.0
    cfg.rewards["stair_takeoff_frontier"].weight = 0.0
    cfg.rewards["stair_apex_or_mantle_frontier"] = RewardTermCfg(
        func=microduck_mdp.stair_apex_or_mantle_frontier,
        weight=6.0,
        params={
            "approach_start_x": 0.40,
            "stair_face_x": STANDARD_STAIR_START_DISTANCE,
            "crossing_end_x": 0.70,
            "standing_root_height": STANDARD_STANDING_ROOT_HEIGHT,
            "clearance_root_height": 0.195,
            "corridor_half_width": STANDARD_STAIR_WIDTH * 0.40,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["stair_assisted_approach"].params.update(
        {"start_x": 0.40, "end_x": 0.56}
    )
    cfg.rewards["stair_assisted_lift"].weight = 2.0
    cfg.rewards["stair_assisted_lift"].params.update(
        {"start_height": 0.105, "clearance_height": 0.195, "x_gate": 0.50}
    )
    cfg.rewards["stair_assisted_crossing"].weight = 4.0
    cfg.rewards["stair_assisted_crossing"].params.update(
        {"start_x": 0.62, "end_x": 0.70, "clearance_height": 0.165}
    )
    cfg.rewards["stair_first_riser_clearance"].params.update(
        {"max_vertical_speed": 0.60, "hold_time_s": 0.08}
    )
    cfg.rewards["stair_first_tread_stable"].weight = 0.0
    cfg.rewards["stair_first_tread_secured"].weight = 100.0
    cfg.rewards["stair_first_tread_settle_quality"].weight = 0.0
    del play
    return cfg


def make_microduck_stair_roulade_bank_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Stage A7: adapt exact manufacturer roll phases into a tread mantle."""

    cfg = make_microduck_stair_apex_mantle_env_cfg(play=False)
    reset = cfg.events["route_state_curriculum"].params
    reset.update(
        {
            "lip_release_fraction": 0.0,
            "shell_brace_fraction": 0.0,
            "tread_recovery_fraction": 0.0,
            "real_handoff_fraction": 1.0,
        }
    )
    cfg.events["walker_state_bank"] = EventTermCfg(
        func=WalkerStateBankReset,
        mode="reset",
        params={
            "bank_path": ".tmp/codex/full170-roulade-state-bank.pt",
            "canonicalize_heading": False,
            "local_x_range": (0.48, 0.58),
            "local_y_range": (-0.08, 0.08),
            "zero_missing_pose_commands": True,
            "source_episode_step_range": (15, 60),
            "min_forward_speed": 0.20,
            "min_vertical_speed": -0.25,
            "min_root_height": 0.08,
        },
    )

    # A transplanted phase is useful only if the real collision converts the
    # manufacturer's roll into support on top of the tread. Hard-gate crossing
    # by height and corridor, then reward contact-supported x/z progress. The
    # strict clearance and secured-tread latches remain the curriculum gates.
    cfg.rewards["stair_apex_or_mantle_frontier"].weight = 4.0
    cfg.rewards["stair_riser_face_contact"] = RewardTermCfg(
        func=microduck_mdp.stair_riser_face_contact,
        weight=6.0,
        params={
            "stair_face_x": STANDARD_STAIR_START_DISTANCE,
            "riser_height": STANDARD_RISER_HEIGHT,
            "tread_depth": STANDARD_TREAD_DEPTH,
            "corridor_half_width": STANDARD_STAIR_WIDTH * 0.40,
        },
    )
    cfg.rewards["stair_first_tread_contact"] = RewardTermCfg(
        func=microduck_mdp.stair_first_tread_contact,
        weight=150.0,
        params={
            "stair_face_x": STANDARD_STAIR_START_DISTANCE,
            "riser_height": STANDARD_RISER_HEIGHT,
            "tread_depth": STANDARD_TREAD_DEPTH,
            "corridor_half_width": STANDARD_STAIR_WIDTH * 0.40,
        },
    )
    cfg.rewards["stair_assisted_approach"].weight = 0.5
    cfg.rewards["stair_assisted_lift"].weight = 3.0
    cfg.rewards["stair_assisted_crossing"].weight = 8.0
    cfg.rewards["stair_assisted_crossing"].params.update(
        {
            "start_x": STANDARD_STAIR_START_DISTANCE,
            "end_x": STANDARD_STAIR_START_DISTANCE + 0.08,
            "clearance_height": STANDARD_RISER_HEIGHT,
            "corridor_half_width": STANDARD_STAIR_WIDTH * 0.40,
            "hard_height_gate": True,
        }
    )
    cfg.rewards["stair_tread_support_frontier"] = RewardTermCfg(
        func=microduck_mdp.stair_tread_support_frontier,
        weight=15.0,
        params={
            "stair_face_x": STANDARD_STAIR_START_DISTANCE,
            "target_x": STANDARD_STAIR_START_DISTANCE + 0.08,
            "riser_height": STANDARD_RISER_HEIGHT,
            "target_root_height": 0.205,
            "corridor_half_width": STANDARD_STAIR_WIDTH * 0.40,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["stair_first_riser_clearance"].weight = 400.0
    cfg.rewards["stair_first_tread_secured"].weight = 300.0
    cfg.rewards["stair_first_tread_settle_quality"].weight = 1.0
    del play
    return cfg


def make_microduck_stair_phase_balanced_rsi_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Stage A18: replay every phase of the exact manufacturer roll evenly."""

    cfg = make_microduck_stair_roulade_bank_env_cfg(play=False)
    bank = cfg.events["walker_state_bank"].params
    bank.update(
        {
            "phase_balanced": True,
            "phase_bucket_count": 4,
        }
    )
    # Four temporal buckets explicitly expose preload, contact, apex, and
    # release. This is reference-state initialization, not a scripted action
    # trajectory: the policy still controls every motor after reset.
    cfg.episode_length_s = 8.0
    cfg.events["walker_state_bank"] = EventTermCfg(
        func=WalkerStateBankReset,
        mode="reset",
        params=bank,
    )
    cfg.rewards["stair_first_tread_secured"].weight = 400.0
    del play
    return cfg


def make_microduck_stair_curriculum_rsi_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Stage A20: adapt aligned manufacturer roll phases from 100 to 170 mm."""

    cfg = make_microduck_stair_phase_balanced_rsi_env_cfg(play=False)
    cfg.episode_length_s = 6.0

    # Eight discrete heights teach the contact mechanism without changing the
    # final objective. A standing quarter of each training batch is pinned to
    # the literal 170 mm home stair, while successful lower rows advance only
    # after a durable first-riser clearance.
    terrain_generator = cfg.scene.terrain.terrain_generator
    terrain_generator.num_rows = STAIR_MECHANISM_CURRICULUM_LEVELS
    for terrain_cfg in terrain_generator.sub_terrains.values():
        terrain_cfg.riser_height_range = (
            STAIR_MECHANISM_MIN_RISER_HEIGHT,
            STANDARD_RISER_HEIGHT,
        )
        terrain_cfg.difficulty_levels = STAIR_MECHANISM_CURRICULUM_LEVELS
    cfg.scene.terrain.max_init_terrain_level = (
        STAIR_MECHANISM_CURRICULUM_LEVELS - 1 if play else 2
    )
    challenge_event = EventTermCfg(
        func=microduck_mdp.seed_route_challenge_levels,
        mode="reset",
        params={"standard_fraction": 1.0 if play else 0.25},
    )
    cfg.events = {"route_challenge_levels": challenge_event, **cfg.events}
    if play:
        cfg.curriculum.pop("terrain_levels", None)
    else:
        cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
            func=microduck_mdp.route_terrain_levels,
            params={},
        )

    # Keep each reference phase at the spatial point where it can perform its
    # intended job. The old uniform 0.48-0.58 m override placed preload,
    # contact, apex, and release in the same band and erased roll timing.
    bank = cfg.events["walker_state_bank"].params
    bank.pop("local_x_range", None)
    bank.update(
        {
            "phase_aligned_local_x_range": (0.46, 0.62),
            "phase_aligned_x_jitter": 0.01,
        }
    )
    cfg.events["walker_state_bank"] = EventTermCfg(
        func=WalkerStateBankReset,
        mode="reset",
        params=bank,
    )

    route_cue_params = {
        "stair_start_distance": STANDARD_STAIR_START_DISTANCE,
        "goal_distance": STANDARD_GOAL_DISTANCE,
        "min_riser_height": STAIR_MECHANISM_MIN_RISER_HEIGHT,
        "max_riser_height": STANDARD_RISER_HEIGHT,
        "num_terrain_levels": STAIR_MECHANISM_CURRICULUM_LEVELS,
        "tread_depth": STANDARD_TREAD_DEPTH,
        "num_steps": STANDARD_NUM_STEPS,
        "cue_distance": 0.30,
    }
    for observation_group in ("actor", "critic"):
        cfg.observations[observation_group].terms["body_command"].params = (
            route_cue_params
        )
    # The actor remains the manufacturer's 61D deployment contract. Only the
    # critic sees exact obstacle, contact, curriculum, and reference-phase
    # state, matching the asymmetric training pattern used by parkour systems.
    cfg.observations["critic"].terms["stair_privileged_state"] = (
        ObservationTermCfg(
            func=microduck_mdp.stair_critic_privileged_state,
            params={
                "stair_start_distance": STANDARD_STAIR_START_DISTANCE,
                "min_riser_height": STAIR_MECHANISM_MIN_RISER_HEIGHT,
                "max_riser_height": STANDARD_RISER_HEIGHT,
                "num_terrain_levels": STAIR_MECHANISM_CURRICULUM_LEVELS,
                "source_episode_step_range": (15, 60),
                "corridor_half_width": STANDARD_STAIR_WIDTH * 0.40,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
    )

    # Fixed-170 mm contact proxies are invalid on lower rows. Retain only a
    # terrain-aware, non-farmable frontier and the same durable clearance gate
    # at every row. Full-height promotion is evaluated separately and strictly.
    for reward_name in (
        "stair_apex_or_mantle_frontier",
        "stair_riser_face_contact",
        "stair_first_tread_contact",
        "stair_assisted_lift",
        "stair_assisted_crossing",
        "stair_tread_support_frontier",
        "stair_first_tread_settle_quality",
    ):
        cfg.rewards[reward_name].weight = 0.0
    # This term keeps its literal 170 mm contact geometry. It is intentionally
    # impossible to collect from the lower curriculum rows and provides the
    # exact supported-tread acceptance target to the full-height challenge set.
    cfg.rewards["stair_first_tread_secured"].weight = 600.0
    cfg.rewards["stair_curriculum_mantle_frontier"] = RewardTermCfg(
        func=microduck_mdp.stair_curriculum_mantle_frontier,
        weight=12.0,
        params={
            "stair_start_distance": STANDARD_STAIR_START_DISTANCE,
            "min_riser_height": STAIR_MECHANISM_MIN_RISER_HEIGHT,
            "max_riser_height": STANDARD_RISER_HEIGHT,
            "num_terrain_levels": STAIR_MECHANISM_CURRICULUM_LEVELS,
            "standing_root_height": STANDARD_STANDING_ROOT_HEIGHT,
            "x_margin": 0.04,
            "z_margin": 0.025,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    clearance = cfg.rewards["stair_first_riser_clearance"]
    clearance.weight = 500.0
    clearance.params.pop("riser_height", None)
    clearance.params.update(
        {
            "min_riser_height": STAIR_MECHANISM_MIN_RISER_HEIGHT,
            "max_riser_height": STANDARD_RISER_HEIGHT,
            "num_terrain_levels": STAIR_MECHANISM_CURRICULUM_LEVELS,
            "x_margin": 0.04,
            "z_margin": 0.025,
            "max_vertical_speed": 0.45,
            "hold_time_s": 0.10,
        }
    )
    return cfg


def make_microduck_stair_contact_mantle_rsi_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Stage A21: convert real stair contact into last-mile mantle progress."""

    cfg = make_microduck_stair_curriculum_rsi_env_cfg(play=play)
    cfg.rewards["stair_curriculum_mantle_frontier"].weight = 0.0
    cfg.rewards["stair_curriculum_contact_mantle_frontier"] = RewardTermCfg(
        func=microduck_mdp.stair_curriculum_contact_mantle_frontier,
        weight=12.0,
        params={
            "stair_start_distance": STANDARD_STAIR_START_DISTANCE,
            "min_riser_height": STAIR_MECHANISM_MIN_RISER_HEIGHT,
            "max_riser_height": STANDARD_RISER_HEIGHT,
            "num_terrain_levels": STAIR_MECHANISM_CURRICULUM_LEVELS,
            "tread_depth": STANDARD_TREAD_DEPTH,
            "corridor_half_width": STANDARD_STAIR_WIDTH * 0.40,
            "x_start_margin": 0.005,
            "x_target_margin": 0.040,
            "z_start_margin": 0.005,
            "z_target_margin": 0.025,
            "support_sensor_name": "robot_ground_contact",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    return cfg


def make_microduck_stair_tread_contact_bank_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Stage A10: learn the pull-up from exact learned tread-contact states."""

    cfg = make_microduck_stair_roulade_bank_env_cfg(play=False)
    cfg.events["walker_state_bank"] = EventTermCfg(
        func=WalkerStateBankReset,
        mode="reset",
        params={"bank_path": ".tmp/codex/full170-tread-contact-state-bank.pt"},
    )

    # The reset already contains a genuine horizontal-tread contact. Do not pay
    # for replaying that initial state. Pay only newly achieved root lift and
    # advance, then the unchanged physical clearance and secured-tread gates.
    cfg.rewards["stair_riser_face_contact"].weight = 0.0
    cfg.rewards["stair_first_tread_contact"].weight = 1.0e-6
    cfg.rewards["stair_tread_support_frontier"].weight = 0.0
    cfg.rewards["stair_apex_or_mantle_frontier"].weight = 1.0
    cfg.rewards["stair_assisted_lift"].weight = 2.0
    cfg.rewards["stair_assisted_crossing"].weight = 10.0
    cfg.rewards["stair_tread_pullup_frontier"] = RewardTermCfg(
        func=microduck_mdp.stair_tread_pullup_frontier,
        weight=20.0,
        params={
            "start_x": 0.56,
            "target_x": STANDARD_STAIR_START_DISTANCE + 0.06,
            "start_height": 0.10,
            "target_height": 0.205,
            "corridor_half_width": STANDARD_STAIR_WIDTH * 0.40,
            "stair_face_x": STANDARD_STAIR_START_DISTANCE,
            "riser_height": STANDARD_RISER_HEIGHT,
            "tread_depth": STANDARD_TREAD_DEPTH,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["stair_first_tread_secured"].weight = 600.0
    cfg.rewards["stair_first_tread_settle_quality"].weight = 2.0
    del play
    return cfg


def make_microduck_stair_foot_anchor_vault_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Stage A11: turn a loaded first-tread foot plant into a vault."""

    cfg = make_microduck_stair_tread_contact_bank_env_cfg(play=False)
    cfg.events["walker_state_bank"] = EventTermCfg(
        func=WalkerStateBankReset,
        mode="reset",
        params={
            "bank_path": ".tmp/codex/full170-loaded-foot-anchor-state-bank.pt",
            "min_vault_momentum": 0.12,
            "vault_lever_arm": 0.06,
        },
    )
    # Remove the A10 scalar root proxy. The new potential can advance only
    # after a loaded foot contact performs positive sagittal mechanical work.
    cfg.rewards["stair_tread_pullup_frontier"].weight = 0.0
    cfg.rewards["stair_apex_or_mantle_frontier"].weight = 0.0
    cfg.rewards["stair_assisted_lift"].weight = 0.0
    cfg.rewards["stair_foot_anchor_vault_frontier"] = RewardTermCfg(
        func=microduck_mdp.stair_foot_anchor_vault_frontier,
        weight=20.0,
        params={
            "start_x": 0.60,
            "target_x": STANDARD_STAIR_START_DISTANCE + 0.06,
            "start_height": 0.12,
            "target_height": 0.205,
            "min_normal_force": 0.40,
            "min_positive_power": 0.01,
            "target_positive_power": 0.25,
            "release_window_s": 0.25,
            "support_sensor_name": "feet_stair_contact",
            "corridor_half_width": STANDARD_STAIR_WIDTH * 0.40,
            "stair_face_x": STANDARD_STAIR_START_DISTANCE,
            "riser_height": STANDARD_RISER_HEIGHT,
            "tread_depth": STANDARD_TREAD_DEPTH,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    del play
    return cfg


def make_microduck_stair_ordered_vault_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Stage A12: require loaded work, new lift, then height-gated crossing."""

    cfg = make_microduck_stair_foot_anchor_vault_env_cfg(play=False)
    cfg.episode_length_s = 2.5
    cfg.events["walker_state_bank"] = EventTermCfg(
        func=WalkerStateBankReset,
        mode="reset",
        params={
            "bank_path": ".tmp/codex/full170-loaded-foot-anchor-state-bank.pt",
            "min_vertical_speed": -0.35,
            "min_root_height": 0.09,
            "min_vault_momentum": 0.12,
            "vault_lever_arm": 0.06,
            "max_abs_local_y": 0.08,
            "max_abs_lateral_speed": 0.20,
            "max_abs_yaw_rate": 4.0,
        },
    )
    # Positive contact power is now an unrewarded eligibility event. A static
    # anchor receives zero: only root motion after the arm event can advance
    # the ordered frontier, and crossing remains gated by absolute lip height.
    cfg.rewards["stair_foot_anchor_vault_frontier"].weight = 0.0
    cfg.rewards["stair_ordered_foot_vault_frontier"] = RewardTermCfg(
        func=microduck_mdp.stair_ordered_foot_vault_frontier,
        weight=25.0,
        params={
            "target_x": STANDARD_STAIR_START_DISTANCE + 0.06,
            "target_height": 0.205,
            "required_lift": 0.04,
            "lip_gate_height": 0.165,
            "min_normal_force": 0.40,
            "min_positive_power": 0.02,
            "min_positive_work_j": 0.004,
            "min_control_steps": 3,
            "support_sensor_name": "feet_stair_contact",
            "corridor_half_width": 0.20,
            "stair_face_x": STANDARD_STAIR_START_DISTANCE,
            "riser_height": STANDARD_RISER_HEIGHT,
            "tread_depth": STANDARD_TREAD_DEPTH,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["stair_assisted_crossing"].weight = 10.0
    cfg.rewards["stair_assisted_crossing"].params.update(
        {"clearance_height": 0.198, "hard_height_gate": True}
    )
    cfg.rewards["stair_first_riser_clearance"].weight = 400.0
    cfg.rewards["stair_first_riser_clearance"].params.update(
        {"z_margin": 0.028, "hold_time_s": 0.08}
    )
    cfg.rewards["stair_first_tread_secured"].weight = 600.0
    cfg.rewards["stair_first_tread_secured"].params["min_height"] = 0.198
    cfg.rewards["stair_first_tread_settle_quality"].weight = 0.0
    cfg.rewards["action_rate_l2"].weight = 0.0
    del play
    return cfg


MicroduckStandardStairsRlCfg = deepcopy(MicroduckRlCfg)
MicroduckStandardStairsRlCfg.experiment_name = "microduck_standard_stairs"
MicroduckStandardStairsRlCfg.run_name = "microduck_standard_stairs"
MicroduckStandardStairsRlCfg.max_iterations = 10_000

MicroduckRouteStairsRlCfg = deepcopy(MicroduckRlCfg)
MicroduckRouteStairsRlCfg.experiment_name = "microduck_stair_route"
MicroduckRouteStairsRlCfg.run_name = "microduck_stair_route"
MicroduckRouteStairsRlCfg.max_iterations = 10_000
MicroduckRouteStairsRlCfg.save_interval = 50

MicroduckStairSpecialistRlCfg = deepcopy(MicroduckRouteStairsRlCfg)
MicroduckStairSpecialistRlCfg.experiment_name = "microduck_stair_specialist"
MicroduckStairSpecialistRlCfg.run_name = "microduck_stair_specialist"
MicroduckStairSpecialistRlCfg.max_iterations = 800
MicroduckStairSpecialistRlCfg.save_interval = 50
MicroduckStairSpecialistRlCfg.algorithm.learning_rate = 1.0e-4
MicroduckStairSpecialistRlCfg.algorithm.schedule = "fixed"
MicroduckStairSpecialistRlCfg.algorithm.clip_param = 0.1
MicroduckStairSpecialistRlCfg.algorithm.entropy_coef = 0.003
MicroduckStairSpecialistRlCfg.algorithm.num_learning_epochs = 3
MicroduckStairSpecialistRlCfg.algorithm.num_mini_batches = 4

MicroduckAssistedStairSpecialistRlCfg = deepcopy(MicroduckStairSpecialistRlCfg)
MicroduckAssistedStairSpecialistRlCfg.experiment_name = (
    "microduck_stair_assisted_specialist"
)
MicroduckAssistedStairSpecialistRlCfg.run_name = (
    "microduck_stair_assisted_specialist"
)
MicroduckAssistedStairSpecialistRlCfg.max_iterations = 100
MicroduckAssistedStairSpecialistRlCfg.save_interval = 25
MicroduckAssistedStairSpecialistRlCfg.actor.distribution_cfg["init_std"] = 0.45
MicroduckAssistedStairSpecialistRlCfg.algorithm.learning_rate = 5.0e-5
MicroduckAssistedStairSpecialistRlCfg.algorithm.schedule = "fixed"
MicroduckAssistedStairSpecialistRlCfg.algorithm.clip_param = 0.10
MicroduckAssistedStairSpecialistRlCfg.algorithm.entropy_coef = 0.0005
MicroduckAssistedStairSpecialistRlCfg.algorithm.num_learning_epochs = 5
MicroduckAssistedStairSpecialistRlCfg.algorithm.num_mini_batches = 4
MicroduckAssistedStairSpecialistRlCfg.algorithm.gamma = 0.99
MicroduckAssistedStairSpecialistRlCfg.algorithm.lam = 0.95
MicroduckAssistedStairSpecialistRlCfg.algorithm.max_grad_norm = 0.5

MicroduckStairBridgeSpecialistRlCfg = deepcopy(MicroduckAssistedStairSpecialistRlCfg)
MicroduckStairBridgeSpecialistRlCfg.experiment_name = (
    "microduck_stair_bridge_specialist"
)
MicroduckStairBridgeSpecialistRlCfg.run_name = "microduck_stair_bridge_specialist"
MicroduckStairBridgeSpecialistRlCfg.max_iterations = 200
MicroduckStairBridgeSpecialistRlCfg.actor.distribution_cfg["init_std"] = 0.35
MicroduckStairBridgeSpecialistRlCfg.algorithm.learning_rate = 3.0e-5
MicroduckStairBridgeSpecialistRlCfg.algorithm.entropy_coef = 0.0002

MicroduckStairWalkerBankRlCfg = deepcopy(MicroduckStairBridgeSpecialistRlCfg)
MicroduckStairWalkerBankRlCfg.experiment_name = (
    "microduck_stair_walker_bank_specialist"
)
MicroduckStairWalkerBankRlCfg.run_name = "microduck_stair_walker_bank_specialist"
MicroduckStairWalkerBankRlCfg.max_iterations = 100
MicroduckStairWalkerBankRlCfg.save_interval = 25
MicroduckStairWalkerBankRlCfg.actor.distribution_cfg["init_std"] = 0.30
MicroduckStairWalkerBankRlCfg.algorithm.learning_rate = 2.0e-5
MicroduckStairWalkerBankRlCfg.algorithm.entropy_coef = 0.0001

MicroduckStairLaunchBankRlCfg = deepcopy(MicroduckStairWalkerBankRlCfg)
MicroduckStairLaunchBankRlCfg.experiment_name = (
    "microduck_stair_launch_bank_specialist"
)
MicroduckStairLaunchBankRlCfg.run_name = "microduck_stair_launch_bank_specialist"
MicroduckStairLaunchBankRlCfg.max_iterations = 200
MicroduckStairLaunchBankRlCfg.save_interval = 25
MicroduckStairLaunchBankRlCfg.actor.distribution_cfg["init_std"] = 0.34
MicroduckStairLaunchBankRlCfg.algorithm.learning_rate = 2.5e-5
MicroduckStairLaunchBankRlCfg.algorithm.entropy_coef = 0.0002

MicroduckStairApexMantleRlCfg = deepcopy(MicroduckStairLaunchBankRlCfg)
MicroduckStairApexMantleRlCfg.experiment_name = (
    "microduck_stair_apex_mantle_specialist"
)
MicroduckStairApexMantleRlCfg.run_name = "microduck_stair_apex_mantle_specialist"
MicroduckStairApexMantleRlCfg.max_iterations = 100
MicroduckStairApexMantleRlCfg.save_interval = 25
MicroduckStairApexMantleRlCfg.actor.distribution_cfg["init_std"] = 0.35
MicroduckStairApexMantleRlCfg.algorithm.learning_rate = 3.0e-5
MicroduckStairApexMantleRlCfg.algorithm.entropy_coef = 0.0005

MicroduckStairRouladeBankRlCfg = deepcopy(MicroduckStairApexMantleRlCfg)
MicroduckStairRouladeBankRlCfg.experiment_name = (
    "microduck_stair_roulade_bank_specialist"
)
MicroduckStairRouladeBankRlCfg.run_name = "microduck_stair_roulade_bank_specialist"
MicroduckStairRouladeBankRlCfg.max_iterations = 150
MicroduckStairRouladeBankRlCfg.save_interval = 25
MicroduckStairRouladeBankRlCfg.actor.distribution_cfg["init_std"] = 0.28
MicroduckStairRouladeBankRlCfg.algorithm.learning_rate = 2.0e-5
MicroduckStairRouladeBankRlCfg.algorithm.entropy_coef = 0.0002

MicroduckStairPhaseBalancedRsiRlCfg = deepcopy(MicroduckStairRouladeBankRlCfg)
MicroduckStairPhaseBalancedRsiRlCfg.experiment_name = (
    "microduck_stair_phase_balanced_rsi"
)
MicroduckStairPhaseBalancedRsiRlCfg.run_name = "microduck_stair_phase_balanced_rsi"
MicroduckStairPhaseBalancedRsiRlCfg.max_iterations = 400
MicroduckStairPhaseBalancedRsiRlCfg.save_interval = 25
MicroduckStairPhaseBalancedRsiRlCfg.actor.distribution_cfg["init_std"] = 0.30
MicroduckStairPhaseBalancedRsiRlCfg.algorithm.learning_rate = 2.0e-5

MicroduckStairCurriculumRsiRlCfg = deepcopy(MicroduckStairPhaseBalancedRsiRlCfg)
MicroduckStairCurriculumRsiRlCfg.experiment_name = (
    "microduck_stair_curriculum_rsi_specialist"
)
MicroduckStairCurriculumRsiRlCfg.run_name = (
    "microduck_stair_curriculum_rsi_specialist"
)
MicroduckStairCurriculumRsiRlCfg.max_iterations = 600
MicroduckStairCurriculumRsiRlCfg.save_interval = 25
MicroduckStairCurriculumRsiRlCfg.actor.distribution_cfg["init_std"] = 0.30
MicroduckStairCurriculumRsiRlCfg.algorithm.learning_rate = 2.0e-5

MicroduckStairContactMantleRsiRlCfg = deepcopy(MicroduckStairCurriculumRsiRlCfg)
MicroduckStairContactMantleRsiRlCfg.experiment_name = (
    "microduck_stair_contact_mantle_rsi_specialist"
)
MicroduckStairContactMantleRsiRlCfg.run_name = (
    "microduck_stair_contact_mantle_rsi_specialist"
)
MicroduckStairContactMantleRsiRlCfg.max_iterations = 300
MicroduckStairContactMantleRsiRlCfg.save_interval = 25

MicroduckStairTreadContactBankRlCfg = deepcopy(MicroduckStairRouladeBankRlCfg)
MicroduckStairTreadContactBankRlCfg.experiment_name = (
    "microduck_stair_tread_contact_bank_specialist"
)
MicroduckStairTreadContactBankRlCfg.run_name = (
    "microduck_stair_tread_contact_bank_specialist"
)
MicroduckStairTreadContactBankRlCfg.max_iterations = 150
MicroduckStairTreadContactBankRlCfg.save_interval = 25
MicroduckStairTreadContactBankRlCfg.actor.distribution_cfg["init_std"] = 0.22
MicroduckStairTreadContactBankRlCfg.algorithm.learning_rate = 1.0e-5
MicroduckStairTreadContactBankRlCfg.algorithm.entropy_coef = 0.0001

MicroduckStairFootAnchorVaultRlCfg = deepcopy(MicroduckStairTreadContactBankRlCfg)
MicroduckStairFootAnchorVaultRlCfg.experiment_name = (
    "microduck_stair_foot_anchor_vault_specialist"
)
MicroduckStairFootAnchorVaultRlCfg.run_name = (
    "microduck_stair_foot_anchor_vault_specialist"
)
MicroduckStairFootAnchorVaultRlCfg.max_iterations = 100
MicroduckStairFootAnchorVaultRlCfg.save_interval = 25
MicroduckStairFootAnchorVaultRlCfg.actor.distribution_cfg["init_std"] = 0.24
MicroduckStairFootAnchorVaultRlCfg.algorithm.learning_rate = 2.0e-5
MicroduckStairFootAnchorVaultRlCfg.algorithm.entropy_coef = 0.0002

MicroduckStairOrderedVaultRlCfg = deepcopy(MicroduckStairFootAnchorVaultRlCfg)
MicroduckStairOrderedVaultRlCfg.experiment_name = (
    "microduck_stair_ordered_vault_specialist"
)
MicroduckStairOrderedVaultRlCfg.run_name = "microduck_stair_ordered_vault_specialist"
MicroduckStairOrderedVaultRlCfg.max_iterations = 100
MicroduckStairOrderedVaultRlCfg.save_interval = 25
MicroduckStairOrderedVaultRlCfg.actor.distribution_cfg["init_std"] = 0.28
MicroduckStairOrderedVaultRlCfg.algorithm.learning_rate = 2.0e-5
MicroduckStairOrderedVaultRlCfg.algorithm.entropy_coef = 0.0002

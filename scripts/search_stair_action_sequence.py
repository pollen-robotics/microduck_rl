#!/usr/bin/env python3
"""Bounded A13 CEM search for a full-height first-stair action sequence.

The manufacturer-derived A9 actor remains frozen. CEM searches only a short,
piecewise-constant residual action sequence from one exact A12 loaded-foot bank
state. Candidate fitness is the best single state on its own trajectory. It is
never assembled from independent maxima, and a lateral bypass invalidates the
candidate.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "Mjlab-Stairs-Ordered-Vault-Specialist-MicroDuck"
DEFAULT_BASELINE_CHECKPOINT = REPO_ROOT / (
    "logs/rsl_rl/microduck_stair_roulade_bank_specialist/"
    "2026-08-29_13-46-10_full170_physical_contact_stage_a9_256/model_25.pt"
)
STANDARD_RISER_HEIGHT_M = 0.170
STANDARD_TREAD_DEPTH_M = 0.280
STAIR_FACE_X_M = 0.660
TRUNK_SHELL_GEOM_NAME = "robot/trunk_shell_collision"
TRUNK_SHELL_HALF_EXTENTS_M = np.array((0.034, 0.031, 0.022), dtype=np.float64)
TRUNK_SHELL_LOCAL_CENTER_M = np.array((-0.006, 0.0, -0.006), dtype=np.float64)
SIDE_BYPASS_PENALTY = 1_000_000.0
SAGITTAL_ACTION_DIM = 7


@dataclass(frozen=True)
class StrictScoreConfig:
    """Physical thresholds and same-state score weights."""

    corridor_half_width_m: float = 0.20
    stair_face_x_m: float = STAIR_FACE_X_M
    root_progress_start_x_m: float = 0.58
    root_cross_x_m: float = 0.70
    root_progress_start_z_m: float = 0.115
    root_cross_z_m: float = 0.198
    shell_progress_start_x_m: float = 0.58
    shell_progress_start_z_m: float = 0.10
    shell_clear_x_m: float = STAIR_FACE_X_M
    shell_clear_z_m: float = STANDARD_RISER_HEIGHT_M
    root_height_weight: float = 2.0
    face_crossing_weight: float = 2.0
    shell_height_weight: float = 2.0
    shell_crossing_weight: float = 2.0
    root_coupled_progress_weight: float = 6.0
    shell_coupled_progress_weight: float = 10.0
    exact_shell_clearance_bonus: float = 20.0
    root_and_shell_crossing_bonus: float = 40.0
    secured_tread_bonus: float = 500.0
    side_bypass_penalty: float = SIDE_BYPASS_PENALTY


@dataclass(frozen=True)
class CandidateScore:
    score: float
    success: bool
    best_step: int
    side_bypass: bool
    exact_shell_clearance: bool
    root_and_shell_crossing: bool
    secured_tread: bool
    best_root_x_m: float
    best_root_y_m: float
    best_root_z_m: float
    best_shell_min_x_m: float
    best_shell_min_z_m: float


def expand_action_knots(knots: np.ndarray, horizon_steps: int) -> np.ndarray:
    """Expand ``(..., knots, actions)`` into balanced constant time segments."""

    values = np.asarray(knots)
    if values.ndim < 2:
        raise ValueError("knots must have shape (..., num_knots, action_dim)")
    if values.shape[-2] < 1 or values.shape[-1] < 1:
        raise ValueError("num_knots and action_dim must be positive")
    if horizon_steps < 1:
        raise ValueError("horizon_steps must be positive")
    knot_indices = (
        np.arange(horizon_steps, dtype=np.int64) * values.shape[-2]
    ) // horizon_steps
    return np.take(values, knot_indices, axis=-2)


def sample_cem_population(
    rng: np.random.Generator,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    candidates: int,
    residual_limit: float,
    frozen_prefix_knots: int = 0,
) -> np.ndarray:
    """Sample CEM candidates while preserving an exact proven prefix."""

    if mean.shape != std.shape or mean.ndim != 2:
        raise ValueError("mean and std must share shape (knots, actions)")
    if candidates < 1 or residual_limit <= 0.0:
        raise ValueError("candidates and residual_limit must be positive")
    if not 0 <= frozen_prefix_knots <= mean.shape[0]:
        raise ValueError("frozen_prefix_knots is outside the knot range")
    population = rng.normal(
        loc=mean,
        scale=std,
        size=(candidates, *mean.shape),
    )
    population = np.clip(population, -residual_limit, residual_limit)
    population[0] = np.clip(mean, -residual_limit, residual_limit)
    if frozen_prefix_knots:
        population[:, :frozen_prefix_knots] = mean[:frozen_prefix_knots]
    return population


def expand_sagittal_actions(actions: np.ndarray) -> np.ndarray:
    """Expand seven sagittal controls into the mirrored 14-joint convention."""

    compact = np.asarray(actions)
    if compact.shape[-1:] != (SAGITTAL_ACTION_DIM,):
        raise ValueError("sagittal actions must end in shape (7,)")
    expanded = np.zeros((*compact.shape[:-1], 14), dtype=compact.dtype)
    expanded[..., :5] = compact[..., :5]
    expanded[..., 5:7] = compact[..., 5:7]
    expanded[..., 9:14] = -compact[..., :5]
    return expanded


def project_sagittal_actions(actions: np.ndarray) -> np.ndarray:
    """Project full residual actions onto the sagittal mirror-invariant subspace."""

    full = np.asarray(actions)
    if full.shape[-1:] != (14,):
        raise ValueError("full actions must end in shape (14,)")
    compact = np.empty((*full.shape[:-1], SAGITTAL_ACTION_DIM), dtype=full.dtype)
    compact[..., :5] = 0.5 * (full[..., :5] - full[..., 9:14])
    compact[..., 5:7] = full[..., 5:7]
    return compact


def box_corners_world(
    centers: np.ndarray,
    rotations: np.ndarray,
    half_extents: np.ndarray,
) -> np.ndarray:
    """Return all eight corners of oriented boxes in world coordinates."""

    center_values = np.asarray(centers, dtype=np.float64)
    rotation_values = np.asarray(rotations, dtype=np.float64)
    extents = np.asarray(half_extents, dtype=np.float64)
    if center_values.shape[-1:] != (3,):
        raise ValueError("centers must end in shape (3,)")
    if rotation_values.shape[-2:] != (3, 3):
        raise ValueError("rotations must end in shape (3, 3)")
    if center_values.shape[:-1] != rotation_values.shape[:-2]:
        raise ValueError("center and rotation batch shapes must match")
    if extents.shape != (3,) or np.any(extents <= 0.0):
        raise ValueError("half_extents must be three positive values")
    signs = np.array(
        [
            (sx, sy, sz)
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ],
        dtype=np.float64,
    )
    local_corners = signs * extents
    return center_values[..., None, :] + np.einsum(
        "...ij,kj->...ki", rotation_values, local_corners
    )


def _unit_progress(values: np.ndarray, start: float, target: float) -> np.ndarray:
    if target <= start:
        raise ValueError("progress target must exceed start")
    return np.clip((values - start) / (target - start), 0.0, 1.0)


def score_strict_trajectory(
    root_positions: np.ndarray,
    shell_corners: np.ndarray,
    secured_tread: np.ndarray,
    valid_steps: np.ndarray | None = None,
    config: StrictScoreConfig | None = None,
) -> CandidateScore:
    """Score one trajectory without combining evidence from different states.

    Every positive component in ``step_scores`` is evaluated at the same time
    index. Strict success additionally requires corridor compliance, root-over-
    lip, exact clearance by every trunk-shell corner, and the environment's
    physical secured-tread latch at that same index.
    """

    if config is None:
        config = StrictScoreConfig()
    roots = np.asarray(root_positions, dtype=np.float64)
    corners = np.asarray(shell_corners, dtype=np.float64)
    secured = np.asarray(secured_tread, dtype=bool)
    if roots.ndim != 2 or roots.shape[1] != 3 or roots.shape[0] < 1:
        raise ValueError("root_positions must have shape (steps, 3)")
    if corners.shape != (roots.shape[0], 8, 3):
        raise ValueError("shell_corners must have shape (steps, 8, 3)")
    if secured.shape != (roots.shape[0],):
        raise ValueError("secured_tread must have shape (steps,)")
    if valid_steps is None:
        valid = np.ones(roots.shape[0], dtype=bool)
    else:
        valid = np.asarray(valid_steps, dtype=bool)
        if valid.shape != (roots.shape[0],):
            raise ValueError("valid_steps must have shape (steps,)")

    finite = np.isfinite(roots).all(axis=1) & np.isfinite(corners).all(axis=(1, 2))
    valid &= finite
    if not valid.any():
        return CandidateScore(
            score=-config.side_bypass_penalty,
            success=False,
            best_step=-1,
            side_bypass=False,
            exact_shell_clearance=False,
            root_and_shell_crossing=False,
            secured_tread=False,
            best_root_x_m=float("nan"),
            best_root_y_m=float("nan"),
            best_root_z_m=float("nan"),
            best_shell_min_x_m=float("nan"),
            best_shell_min_z_m=float("nan"),
        )

    x = roots[:, 0]
    abs_y = np.abs(roots[:, 1])
    z = roots[:, 2]
    shell_min_x = corners[:, :, 0].min(axis=1)
    shell_min_z = corners[:, :, 2].min(axis=1)
    in_corridor = abs_y <= config.corridor_half_width_m
    side_bypass_steps = (
        valid & (x >= config.stair_face_x_m) & ~in_corridor
    )
    side_bypass = bool(side_bypass_steps.any())

    root_height_progress = _unit_progress(
        z, config.root_progress_start_z_m, config.root_cross_z_m
    )
    face_crossing_progress = _unit_progress(
        x, config.root_progress_start_x_m, config.root_cross_x_m
    )
    shell_height_progress = _unit_progress(
        shell_min_z, config.shell_progress_start_z_m, config.shell_clear_z_m
    )
    shell_crossing_progress = _unit_progress(
        shell_min_x, config.shell_progress_start_x_m, config.shell_clear_x_m
    )

    exact_shell_clear = (
        (shell_min_x >= config.shell_clear_x_m)
        & (shell_min_z >= config.shell_clear_z_m)
    )
    root_cross = (x >= config.root_cross_x_m) & (z >= config.root_cross_z_m)
    strict_cross = valid & in_corridor & exact_shell_clear & root_cross
    strict_secured = strict_cross & secured

    step_scores = (
        config.root_height_weight * root_height_progress
        + config.face_crossing_weight * face_crossing_progress
        + config.shell_height_weight * shell_height_progress
        + config.shell_crossing_weight * shell_crossing_progress
        + config.root_coupled_progress_weight
        * root_height_progress
        * face_crossing_progress
        + config.shell_coupled_progress_weight
        * shell_height_progress
        * shell_crossing_progress
        + config.exact_shell_clearance_bonus * exact_shell_clear.astype(np.float64)
        + config.root_and_shell_crossing_bonus * strict_cross.astype(np.float64)
        + config.secured_tread_bonus * strict_secured.astype(np.float64)
    )
    step_scores = np.where(valid & in_corridor, step_scores, -np.inf)
    best_step = int(np.argmax(step_scores))
    best_score = float(step_scores[best_step])
    if not np.isfinite(best_score):
        best_step = int(np.flatnonzero(valid)[0])
        best_score = 0.0
    if side_bypass:
        best_score -= config.side_bypass_penalty

    return CandidateScore(
        score=best_score,
        success=bool(strict_secured.any() and not side_bypass),
        best_step=best_step,
        side_bypass=side_bypass,
        exact_shell_clearance=bool((valid & in_corridor & exact_shell_clear).any()),
        root_and_shell_crossing=bool(strict_cross.any()),
        secured_tread=bool(strict_secured.any() and not side_bypass),
        best_root_x_m=float(x[best_step]),
        best_root_y_m=float(roots[best_step, 1]),
        best_root_z_m=float(z[best_step]),
        best_shell_min_x_m=float(shell_min_x[best_step]),
        best_shell_min_z_m=float(shell_min_z[best_step]),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-checkpoint", type=Path, default=DEFAULT_BASELINE_CHECKPOINT
    )
    parser.add_argument("--candidates", type=int, default=64)
    parser.add_argument("--elite-fraction", type=float, default=0.125)
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument(
        "--batch-size",
        "--env-count",
        dest="batch_size",
        type=int,
        default=16,
        help="Number of candidates simulated concurrently.",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--bank-row", type=int)
    parser.add_argument("--horizon-steps", type=int, default=40)
    parser.add_argument("--knots", type=int, default=5)
    parser.add_argument("--residual-std", type=float, default=0.25)
    parser.add_argument("--residual-limit", type=float, default=0.60)
    parser.add_argument("--action-limit", type=float, default=10.0)
    parser.add_argument(
        "--sagittal-symmetry",
        action="store_true",
        help="Search seven mirror-invariant sagittal controls instead of 14 joints.",
    )
    parser.add_argument("--cem-alpha", type=float, default=0.70)
    parser.add_argument("--min-std", type=float, default=0.02)
    parser.add_argument(
        "--initial-npz",
        type=Path,
        help="Warm-start the leading knots from a previous A13 NPZ result.",
    )
    parser.add_argument(
        "--freeze-prefix-knots",
        type=int,
        default=0,
        help="Keep this many warm-start knots exact and search only the continuation.",
    )
    parser.add_argument("--max-wall-seconds", type=float, default=900.0)
    parser.add_argument(
        "--gpu-headroom-gb",
        type=float,
        default=4.0,
        help="Abort before another batch if free CUDA memory falls below this.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output-json", type=Path, default=REPO_ROOT / ".tmp/codex/a13-cem.json"
    )
    parser.add_argument(
        "--output-npz", type=Path, default=REPO_ROOT / ".tmp/codex/a13-cem.npz"
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if not 2 <= args.candidates <= 4096:
        raise SystemExit("--candidates must be in [2, 4096]")
    if not 0.0 < args.elite_fraction <= 0.5:
        raise SystemExit("--elite-fraction must be in (0, 0.5]")
    if not 1 <= args.generations <= 100:
        raise SystemExit("--generations must be in [1, 100]")
    if not 1 <= args.batch_size <= args.candidates:
        raise SystemExit("--batch-size/--env-count must be in [1, candidates]")
    if not 1 <= args.horizon_steps <= 250 or not 1 <= args.knots <= 50:
        raise SystemExit("--horizon-steps must be <= 250 and --knots <= 50")
    if args.knots > args.horizon_steps:
        raise SystemExit("--knots cannot exceed --horizon-steps")
    if not 0 <= args.freeze_prefix_knots < args.knots:
        raise SystemExit("--freeze-prefix-knots must be in [0, knots)")
    if args.freeze_prefix_knots and args.initial_npz is None:
        raise SystemExit("--freeze-prefix-knots requires --initial-npz")
    for name in ("residual_std", "residual_limit", "action_limit", "min_std"):
        if getattr(args, name) <= 0.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if not 0.0 < args.cem_alpha <= 1.0:
        raise SystemExit("--cem-alpha must be in (0, 1]")
    if args.max_wall_seconds <= 0.0 or args.gpu_headroom_gb < 0.0:
        raise SystemExit("wall-time must be positive and GPU headroom nonnegative")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w", encoding="utf-8", suffix=".tmp", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _write_npz_atomic(path: Path, **arrays: Any) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(suffix=".npz", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_initial_knots(
    path: Path | None,
    *,
    target_knots: int,
    action_dim: int,
) -> np.ndarray:
    """Load a previous best sequence into the leading knots of a longer search."""

    mean = np.zeros((target_knots, action_dim), dtype=np.float64)
    if path is None:
        return mean
    source_path = path.expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Initial A13 NPZ not found: {source_path}")
    with np.load(source_path, allow_pickle=False) as archive:
        if "best_knots" not in archive:
            raise ValueError(f"Initial A13 NPZ has no best_knots array: {source_path}")
        source = np.asarray(archive["best_knots"], dtype=np.float64)
    if source.ndim == 2 and source.shape[1] == 14 and action_dim == SAGITTAL_ACTION_DIM:
        source = project_sagittal_actions(source)
    if source.ndim != 2 or source.shape[1] != action_dim:
        raise ValueError(
            f"Initial best_knots must have shape (knots, {action_dim}), got {source.shape}"
        )
    if source.shape[0] > target_knots:
        raise ValueError(
            f"Initial sequence has {source.shape[0]} knots, target has only {target_knots}"
        )
    mean[: source.shape[0]] = source
    return mean


def _check_gpu_headroom(torch: Any, device: str, minimum_gb: float) -> None:
    if not str(device).startswith("cuda"):
        return
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")
    free_bytes, _ = torch.cuda.mem_get_info(torch.device(device))
    required = minimum_gb * (1024**3)
    if free_bytes < required:
        raise RuntimeError(
            f"CUDA headroom gate failed: {free_bytes / 1024**3:.2f} GiB free, "
            f"{minimum_gb:.2f} GiB required"
        )


def _validate_fixed_stair(env_cfg: Any) -> None:
    generator = env_cfg.scene.terrain.terrain_generator
    terrain_cfgs = list(generator.sub_terrains.values())
    checked = 0
    for terrain_cfg in terrain_cfgs:
        if not hasattr(terrain_cfg, "riser_height"):
            continue
        riser = float(terrain_cfg.riser_height)
        tread = float(terrain_cfg.tread_depth)
        if not math.isclose(riser, STANDARD_RISER_HEIGHT_M, abs_tol=1e-9):
            raise RuntimeError(f"A13 refuses non-170 mm stair riser: {riser}")
        if not math.isclose(tread, STANDARD_TREAD_DEPTH_M, abs_tol=1e-9):
            raise RuntimeError(f"A13 refuses non-280 mm stair tread: {tread}")
        checked += 1
    if checked == 0:
        raise RuntimeError("Could not verify a fixed standard-stair terrain")


def _pin_loaded_foot_bank_row(base_env: Any, requested_row: int | None, torch: Any) -> int:
    term_cfg = base_env.event_manager.get_term_cfg("walker_state_bank")
    bank_reset = term_cfg.func
    if not hasattr(bank_reset, "_eligible_rows"):
        raise RuntimeError("walker_state_bank does not expose eligible bank rows")
    eligible = bank_reset._eligible_rows.detach().cpu().to(torch.long)
    if eligible.numel() == 0:
        raise RuntimeError("A12 loaded-foot bank has no eligible rows")
    if requested_row is None:
        selected = int(eligible[0].item())
    else:
        selected = int(requested_row)
        if not bool((eligible == selected).any().item()):
            raise RuntimeError(
                f"Requested --bank-row {selected} is not among the A12 eligible rows"
            )
    bank_reset._eligible_rows = torch.tensor([selected], dtype=torch.long)
    return selected


def _locate_and_validate_shell(base_env: Any) -> int:
    model = base_env.sim.mj_model
    try:
        geom = model.geom(TRUNK_SHELL_GEOM_NAME)
    except KeyError as error:
        raise RuntimeError(
            f"MuJoCo model has no {TRUNK_SHELL_GEOM_NAME!r} geom"
        ) from error
    if not np.allclose(np.asarray(geom.size), TRUNK_SHELL_HALF_EXTENTS_M, atol=1e-9):
        raise RuntimeError(f"Unexpected trunk-shell half extents: {geom.size}")
    if not np.allclose(np.asarray(geom.pos), TRUNK_SHELL_LOCAL_CENTER_M, atol=1e-9):
        raise RuntimeError(f"Unexpected trunk-shell local center: {geom.pos}")
    return int(geom.id)


def _evaluate_candidates(
    candidates: np.ndarray,
    *,
    env: Any,
    base_env: Any,
    baseline_actor: Any,
    shell_geom_id: int,
    horizon_steps: int,
    batch_size: int,
    action_limit: float,
    score_config: StrictScoreConfig,
    sagittal_symmetry: bool,
    deadline: float,
    gpu_headroom_gb: float,
    torch: Any,
) -> list[CandidateScore]:
    expanded = expand_action_knots(candidates, horizon_steps)
    if sagittal_symmetry:
        expanded = expand_sagittal_actions(expanded)
    expanded = expanded.astype(np.float32)
    results: list[CandidateScore] = []
    signs = torch.tensor(
        [
            (sx, sy, sz)
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ],
        dtype=torch.float32,
        device=base_env.device,
    )
    extents = torch.as_tensor(
        TRUNK_SHELL_HALF_EXTENTS_M, dtype=torch.float32, device=base_env.device
    )
    local_corners = signs * extents

    for start in range(0, len(expanded), batch_size):
        if time.monotonic() >= deadline:
            raise TimeoutError("A13 wall-time limit reached before the next batch")
        _check_gpu_headroom(torch, str(base_env.device), gpu_headroom_gb)
        count = min(batch_size, len(expanded) - start)
        residual = np.zeros(
            (batch_size, horizon_steps, expanded.shape[-1]), dtype=np.float32
        )
        residual[:count] = expanded[start : start + count]
        residual_actions = torch.as_tensor(residual, device=base_env.device)

        with torch.inference_mode():
            observations, _ = env.reset()
        alive = torch.ones(batch_size, dtype=torch.bool, device=base_env.device)
        previous_secured = getattr(
            base_env,
            "_stair_first_tread_secured_latched",
            torch.zeros_like(alive),
        ).clone()
        roots_by_step: list[Any] = []
        corners_by_step: list[Any] = []
        secured_by_step: list[Any] = []
        valid_by_step: list[Any] = []

        for step in range(horizon_steps):
            robot = base_env.scene["robot"]
            origins = base_env.scene.terrain.env_origins
            roots_by_step.append((robot.data.root_link_pos_w - origins).clone())
            shell_centers = (
                base_env.sim.data.geom_xpos[:, shell_geom_id] - origins
            )
            shell_rotations = base_env.sim.data.geom_xmat[:, shell_geom_id]
            corners_by_step.append(
                shell_centers[:, None, :]
                + torch.einsum("bij,kj->bki", shell_rotations, local_corners)
            )
            secured_latched = getattr(
                base_env,
                "_stair_first_tread_secured_latched",
                torch.zeros_like(alive),
            )
            secured_by_step.append((secured_latched & ~previous_secured).clone())
            previous_secured = secured_latched.clone()
            valid_by_step.append(alive.clone())

            with torch.inference_mode():
                actions = baseline_actor(observations) + residual_actions[:, step]
                actions = torch.clamp(actions, -action_limit, action_limit)
                observations, _, dones, _ = env.step(actions)
            alive &= ~dones.bool()

        root_trajectories = torch.stack(roots_by_step, dim=1).cpu().numpy()
        shell_trajectories = torch.stack(corners_by_step, dim=1).cpu().numpy()
        secured_trajectories = torch.stack(secured_by_step, dim=1).cpu().numpy()
        valid_trajectories = torch.stack(valid_by_step, dim=1).cpu().numpy()
        for index in range(count):
            results.append(
                score_strict_trajectory(
                    root_trajectories[index],
                    shell_trajectories[index],
                    secured_trajectories[index],
                    valid_trajectories[index],
                    score_config,
                )
            )
    return results


def run_search(args: argparse.Namespace) -> dict[str, Any]:
    """Create A12 once, run bounded CEM, and persist the best residual sequence."""

    import mjlab.tasks  # noqa: F401  # Populate registry.
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.utils.torch import configure_torch_backends

    import mjlab_microduck.tasks  # noqa: F401  # Register MicroDuck tasks.
    from mjlab_microduck.policies.stair_handoff import load_frozen_actor

    checkpoint = args.baseline_checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"A9 baseline checkpoint not found: {checkpoint}")
    _check_gpu_headroom(torch, args.device, args.gpu_headroom_gb)
    configure_torch_backends()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env_cfg = load_env_cfg(TASK_ID, play=True)
    agent_cfg = load_rl_cfg(TASK_ID)
    _validate_fixed_stair(env_cfg)
    env_cfg.scene.num_envs = args.batch_size
    env_cfg.seed = args.seed
    base_env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
    try:
        selected_bank_row = _pin_loaded_foot_bank_row(base_env, args.bank_row, torch)
        shell_geom_id = _locate_and_validate_shell(base_env)
        runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
        runner = runner_cls(env, asdict(agent_cfg), device=args.device)
        baseline_actor = load_frozen_actor(runner, checkpoint, device=args.device)
        env_action_dim = int(env.num_actions)
        if env_action_dim != 14:
            raise RuntimeError(f"Expected 14 MicroDuck actions, got {env_action_dim}")
        search_action_dim = SAGITTAL_ACTION_DIM if args.sagittal_symmetry else env_action_dim

        rng = np.random.default_rng(args.seed)
        mean = load_initial_knots(
            args.initial_npz,
            target_knots=args.knots,
            action_dim=search_action_dim,
        )
        std = np.full_like(mean, args.residual_std)
        if args.freeze_prefix_knots:
            std[: args.freeze_prefix_knots] = 0.0
        elite_count = max(1, math.ceil(args.candidates * args.elite_fraction))
        score_config = StrictScoreConfig()
        generation_summaries: list[dict[str, Any]] = []
        generation_score_rows: list[np.ndarray] = []
        best_knots: np.ndarray | None = None
        best_result: CandidateScore | None = None
        started = time.monotonic()
        deadline = started + args.max_wall_seconds

        for generation in range(args.generations):
            if time.monotonic() >= deadline:
                break
            _check_gpu_headroom(torch, args.device, args.gpu_headroom_gb)
            population = sample_cem_population(
                rng,
                mean,
                std,
                candidates=args.candidates,
                residual_limit=args.residual_limit,
                frozen_prefix_knots=args.freeze_prefix_knots,
            )
            try:
                results = _evaluate_candidates(
                    population,
                    env=env,
                    base_env=base_env,
                    baseline_actor=baseline_actor,
                    shell_geom_id=shell_geom_id,
                    horizon_steps=args.horizon_steps,
                    batch_size=args.batch_size,
                    action_limit=args.action_limit,
                    score_config=score_config,
                    sagittal_symmetry=args.sagittal_symmetry,
                    deadline=deadline,
                    gpu_headroom_gb=args.gpu_headroom_gb,
                    torch=torch,
                )
            except TimeoutError:
                break
            scores = np.asarray([result.score for result in results], dtype=np.float64)
            elite_indices = np.argsort(scores)[-elite_count:]
            elite_population = population[elite_indices]
            elite_mean = elite_population.mean(axis=0)
            elite_std = elite_population.std(axis=0)
            mean = (1.0 - args.cem_alpha) * mean + args.cem_alpha * elite_mean
            std = np.maximum(
                args.min_std,
                (1.0 - args.cem_alpha) * std + args.cem_alpha * elite_std,
            )
            if args.freeze_prefix_knots:
                std[: args.freeze_prefix_knots] = 0.0

            generation_best_index = int(np.argmax(scores))
            generation_best = results[generation_best_index]
            if best_result is None or generation_best.score > best_result.score:
                best_result = generation_best
                best_knots = population[generation_best_index].copy()
            generation_score_rows.append(scores)
            generation_summaries.append(
                {
                    "generation": generation,
                    "best": asdict(generation_best),
                    "mean_score": float(scores.mean()),
                    "median_score": float(np.median(scores)),
                    "strict_successes": int(sum(result.success for result in results)),
                    "side_bypasses": int(sum(result.side_bypass for result in results)),
                }
            )
            print(
                f"generation={generation} best={generation_best.score:.3f} "
                f"strict_successes={generation_summaries[-1]['strict_successes']} "
                f"bypasses={generation_summaries[-1]['side_bypasses']}",
                flush=True,
            )

        if best_result is None or best_knots is None:
            raise RuntimeError(
                "No complete CEM generation fit within the wall-time/headroom bounds"
            )
        elapsed = time.monotonic() - started
        payload: dict[str, Any] = {
            "schema_version": 1,
            "task": TASK_ID,
            "baseline_checkpoint": str(checkpoint),
            "loaded_foot_bank_row": selected_bank_row,
            "fixed_stair": {
                "riser_height_m": STANDARD_RISER_HEIGHT_M,
                "tread_depth_m": STANDARD_TREAD_DEPTH_M,
                "face_x_m": STAIR_FACE_X_M,
            },
            "search": {
                "candidates": args.candidates,
                "elite_fraction": args.elite_fraction,
                "elite_count": elite_count,
                "requested_generations": args.generations,
                "completed_generations": len(generation_summaries),
                "batch_size": args.batch_size,
                "seed": args.seed,
                "horizon_steps": args.horizon_steps,
                "knots": args.knots,
                "residual_std": args.residual_std,
                "residual_limit": args.residual_limit,
                "cem_alpha": args.cem_alpha,
                "min_std": args.min_std,
                "sagittal_symmetry": args.sagittal_symmetry,
                "initial_npz": (
                    str(args.initial_npz.expanduser().resolve())
                    if args.initial_npz is not None
                    else None
                ),
                "freeze_prefix_knots": args.freeze_prefix_knots,
                "gpu_headroom_gb": args.gpu_headroom_gb,
                "max_wall_seconds": args.max_wall_seconds,
                "elapsed_seconds": elapsed,
            },
            "strict_score_config": asdict(score_config),
            "best": asdict(best_result),
            "success_claim_rule": (
                "same-state corridor + root crossing + every trunk-shell corner "
                "above/behind the lip + secured-tread latch, with no side bypass"
            ),
            "generations": generation_summaries,
        }
        _write_json_atomic(args.output_json, payload)
        _write_npz_atomic(
            args.output_npz,
            best_knots=best_knots.astype(np.float32),
            best_expanded=(
                expand_sagittal_actions(
                    expand_action_knots(best_knots, args.horizon_steps)
                )
                if args.sagittal_symmetry
                else expand_action_knots(best_knots, args.horizon_steps)
            ).astype(np.float32),
            final_mean=mean.astype(np.float32),
            final_std=std.astype(np.float32),
            generation_scores=np.stack(generation_score_rows).astype(np.float32),
            config_json=np.asarray(json.dumps(payload["search"], sort_keys=True)),
        )
        return payload
    finally:
        env.close()


def main() -> int:
    args = _parse_args()
    _validate_args(args)
    payload = run_search(args)
    best = payload["best"]
    print(
        f"best_score={best['score']:.3f} strict_success={best['success']} "
        f"json={args.output_json.resolve()} npz={args.output_npz.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

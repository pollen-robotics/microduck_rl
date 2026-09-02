"""Capture exact A36 family-3 states at the first genuine Stage 2 edge."""

from __future__ import annotations

import argparse
import hashlib
import os
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import mjlab.tasks  # noqa: F401  # Populate the task registry.
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

from mjlab_microduck.policies import load_frozen_actor
from mjlab_microduck.tasks.stair_walk_state_bank import (
    BANK_SCHEMA_VERSION,
    STANDARD_NUM_STEPS,
    STANDARD_RISER_HEIGHT_M,
    STANDARD_TREAD_DEPTH_M,
    capture_walk_state_rows,
    concatenate_walk_state_rows,
    load_walk_state_bank,
    walk_state_count,
)

TASK_ID = "Mjlab-Stairs-Option-Frontier-Forward-RSI-Specialist-MicroDuck"
DEFAULT_CHECKPOINT = Path(
    "logs/rsl_rl/microduck_stair_option_frontier_forward_rsi_specialist/"
    "2026-08-30_09-30-50_a36_round1_option_frontier_1024x25/model_33.pt"
)
DEFAULT_INPUT_BANK = Path(".tmp/codex/stair-option-composition-frontier-bank.pt")
DEFAULT_OUTPUT = Path(".tmp/codex/a36-model33-stage2-forward-state-bank.pt")

ACTOR_OBSERVATION_DIM = 61
ACTION_DIM = 14
SOURCE_RESET_FAMILY = 3
SOURCE_RESET_MODE = 3
HARD_TERRAIN_LEVEL = 2
MIN_POLICY_STEPS = 3
INPUT_PROVENANCE_FIELDS = (
    "frontier_control_step",
    "frontier_launch_phase_step",
    "sweep_candidate_index",
    "sweep_trial_index",
    "source_guarded_bank_row",
)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--input-bank", type=Path, default=DEFAULT_INPUT_BANK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-states", type=int, default=128)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=4_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_bank_atomic(path: Path, bank: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(suffix=".tmp", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(bank, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _select_nested_rows(value: Any, rows: torch.Tensor) -> Any:
    if isinstance(value, dict):
        return {key: _select_nested_rows(item, rows) for key, item in value.items()}
    if not isinstance(value, torch.Tensor):
        raise TypeError("Walker-state leaves must be tensors")
    return value[rows].clone()


def _finite_state_rows(states: dict[str, Any]) -> torch.Tensor:
    count = walk_state_count(states)
    finite = torch.ones(count, dtype=torch.bool)

    def visit(value: Any) -> None:
        nonlocal finite
        if isinstance(value, dict):
            for item in value.values():
                visit(item)
            return
        if not isinstance(value, torch.Tensor):
            raise TypeError("Walker-state leaves must be tensors")
        if value.shape[0] != count:
            raise ValueError("Walker-state leaves have inconsistent row counts")
        if value.is_floating_point() or value.is_complex():
            finite &= torch.isfinite(value).reshape(count, -1).all(dim=-1)

    visit(states)
    return finite


def _exact_state_row_digest(states: dict[str, Any], row: int) -> str:
    digest = hashlib.sha256()

    def visit(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key in sorted(value):
                visit(value[key], f"{path}/{key}")
            return
        if not isinstance(value, torch.Tensor):
            raise TypeError("Walker-state leaves must be tensors")
        item = value[row].detach().cpu().contiguous()
        digest.update(path.encode("utf-8"))
        digest.update(str(item.dtype).encode("ascii"))
        digest.update(repr(tuple(item.shape)).encode("ascii"))
        digest.update(item.reshape(-1).view(torch.uint8).numpy().tobytes())

    visit(states, "states")
    return digest.hexdigest()


def _required_env_tensor(env: ManagerBasedRlEnv, name: str) -> torch.Tensor:
    value = getattr(env, name, None)
    if value is None or not isinstance(value, torch.Tensor):
        raise RuntimeError(f"A36 tracking tensor is unavailable: {name}")
    return value


def _genuine_stage2_edges(
    *,
    stage1: torch.Tensor,
    stage15: torch.Tensor,
    stage2: torch.Tensor,
    previous_stage2: torch.Tensor,
    episode_steps: torch.Tensor,
    dones: torch.Tensor,
    nan_termination: torch.Tensor,
    already_captured: torch.Tensor,
    reset_family: torch.Tensor,
    reset_mode: torch.Tensor,
    source_rows: torch.Tensor,
) -> torch.Tensor:
    """Select only ordered, post-reset Stage2 false-to-true transitions."""

    return (
        stage2
        & ~previous_stage2
        & stage1
        & stage15
        & (episode_steps >= MIN_POLICY_STEPS)
        & ~dones.bool()
        & ~nan_termination.bool()
        & ~already_captured
        & (reset_family == SOURCE_RESET_FAMILY)
        & (reset_mode == SOURCE_RESET_MODE)
        & (source_rows >= 0)
    )


def _configure_family3_option_frontier(
    env_cfg: Any, *, input_bank: Path, num_envs: int
) -> None:
    family = env_cfg.events.get("state_bank_family")
    frontier = env_cfg.events.get("option_frontier_state_bank")
    hard_viewer = env_cfg.events.get("a30_hard_viewer")
    if family is None or frontier is None or hard_viewer is None:
        raise RuntimeError("A36 play cfg is missing family-3 reset infrastructure")
    family.params["forced_family"] = SOURCE_RESET_FAMILY
    frontier.params["bank_path"] = str(input_bank)
    frontier.params["reset_family"] = SOURCE_RESET_FAMILY
    hard_viewer.params["terrain_levels"] = (HARD_TERRAIN_LEVEL,) * num_envs
    hard_viewer.params["terrain_types"] = (0,) * num_envs

    hard_startup = env_cfg.events.get("a34_hard_terrain")
    if hard_startup is not None:
        hard_startup.params["terrain_level"] = HARD_TERRAIN_LEVEL
        hard_startup.params["terrain_type"] = 0


def _validate_live_source(
    env: ManagerBasedRlEnv, *, input_state_count: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    family = _required_env_tensor(env, "_stair_state_bank_family")
    source_rows = _required_env_tensor(env, "_stair_walker_bank_row")
    reset_mode = _required_env_tensor(env, "_stair_assisted_reset_mode")
    if not torch.all(family == SOURCE_RESET_FAMILY):
        raise RuntimeError("A36 collector escaped forced reset family 3")
    if not torch.all(reset_mode == SOURCE_RESET_MODE):
        raise RuntimeError("A36 family-3 replay did not use assisted reset mode 3")
    if not torch.all((source_rows >= 0) & (source_rows < input_state_count)):
        raise RuntimeError("A36 family-3 replay did not write a valid input-bank row")
    if not torch.all(env.scene.terrain.terrain_levels == HARD_TERRAIN_LEVEL):
        raise RuntimeError("A36 collector escaped the hard terrain level")
    return family, source_rows, reset_mode


def _validate_actor_observations(observations: Any, num_envs: int) -> torch.Tensor:
    actor = observations["actor"]
    expected_shape = (num_envs, ACTOR_OBSERVATION_DIM)
    if tuple(actor.shape) != expected_shape:
        raise RuntimeError(
            f"A36 actor observation shape is {tuple(actor.shape)}, expected {expected_shape}"
        )
    if not torch.isfinite(actor).all():
        raise RuntimeError("A36 actor observations contain non-finite values")
    return actor


def _validate_actions(actions: torch.Tensor, num_envs: int) -> None:
    expected_shape = (num_envs, ACTION_DIM)
    if tuple(actions.shape) != expected_shape:
        raise RuntimeError(
            f"A36 policy action shape is {tuple(actions.shape)}, expected {expected_shape}"
        )
    if not torch.isfinite(actions).all():
        raise RuntimeError("A36 policy actions contain non-finite values")


def _source_bank_provenance(
    source_states: dict[str, Any], source_rows: torch.Tensor
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for field in INPUT_PROVENANCE_FIELDS:
        value = source_states.get(field)
        if isinstance(value, torch.Tensor):
            result[f"source_input_bank_{field}"] = value[source_rows].clone()
    return result


def main() -> int:
    args = _build_arg_parser().parse_args()
    if args.target_states < 1 or args.num_envs < 1 or args.max_steps < 1:
        raise SystemExit("Target states, environment count, and max steps must be positive")
    if args.target_states > args.num_envs:
        raise SystemExit(
            "Target states cannot exceed num envs because each environment contributes "
            "only its first genuine Stage2 edge"
        )

    checkpoint = args.checkpoint.expanduser().resolve()
    input_bank_path = args.input_bank.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"A36 checkpoint not found: {checkpoint}")
    source_bank = load_walk_state_bank(input_bank_path)
    source_states = source_bank["states"]
    input_state_count = walk_state_count(source_states)
    if not isinstance(source_states.get("frontier_control_step"), torch.Tensor):
        raise SystemExit(
            "Input option-frontier bank is missing frontier_control_step provenance"
        )

    configure_torch_backends()
    env_cfg = load_env_cfg(TASK_ID, play=True)
    agent_cfg = load_rl_cfg(TASK_ID)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    _configure_family3_option_frontier(
        env_cfg, input_bank=input_bank_path, num_envs=args.num_envs
    )

    base_env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
    env.reset()
    _validate_live_source(base_env, input_state_count=input_state_count)

    runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=args.device)
    actor = load_frozen_actor(runner, checkpoint, device=args.device)

    captured_environment = torch.zeros(
        args.num_envs, dtype=torch.bool, device=args.device
    )
    previous_stage2 = torch.zeros_like(captured_environment)
    episode_index = torch.zeros(
        args.num_envs, dtype=torch.long, device=args.device
    )
    reset_audited = torch.zeros_like(captured_environment)
    seen_state_digests: set[str] = set()
    duplicate_states_rejected = 0
    chunks: list[dict[str, Any]] = []
    observations = env.get_observations()
    steps_run = 0

    runtime_metadata = {
        "joint_names": list(base_env.scene["robot"].joint_names),
        "physics_dt": base_env.physics_dt,
        "step_dt": base_env.step_dt,
        "decimation": base_env.cfg.decimation,
    }

    try:
        for steps_run in range(1, args.max_steps + 1):
            _validate_actor_observations(observations, args.num_envs)
            with torch.inference_mode():
                actions = actor(observations)
                _validate_actions(actions, args.num_envs)
                observations, _, dones, _ = env.step(actions)

            family, source_rows, reset_mode = _validate_live_source(
                base_env, input_state_count=input_state_count
            )
            stage1 = _required_env_tensor(
                base_env, "_stair_contact_transfer_stage1_policy_achieved"
            ).bool()
            stage15 = _required_env_tensor(
                base_env, "_stair_contact_transfer_stage15_policy_achieved"
            ).bool()
            stage2 = _required_env_tensor(
                base_env, "_stair_contact_transfer_stage2_policy_achieved"
            ).bool()
            shell = _required_env_tensor(
                base_env, "_stair_true_shell_clearance_policy_achieved"
            ).bool()
            nan_termination = base_env.termination_manager.get_term("nan_state").bool()
            done_mask = dones.bool()

            if torch.any(nan_termination):
                raise RuntimeError("A36 collector encountered a NaN termination")
            finite_sim = (
                torch.isfinite(base_env.sim.data.qpos)
                .reshape(args.num_envs, -1)
                .all(dim=-1)
                & torch.isfinite(base_env.sim.data.qvel)
                .reshape(args.num_envs, -1)
                .all(dim=-1)
            )
            if not torch.all(finite_sim):
                raise RuntimeError("A36 simulator state contains non-finite values")

            fresh_to_audit = (
                (base_env.episode_length_buf <= 1) & ~done_mask & ~reset_audited
            )
            if torch.any(fresh_to_audit):
                if not torch.all(stage1[fresh_to_audit] & stage15[fresh_to_audit]):
                    raise RuntimeError(
                        "A36 family 3 did not seed the Stage1/Stage1.5 prefix"
                    )
                if torch.any(stage2[fresh_to_audit]):
                    raise RuntimeError(
                        "A36 family 3 leaked Stage2 credit into the reset window"
                    )
                reset_audited[fresh_to_audit] = True

            raw_stage2_edge = stage2 & ~previous_stage2
            order_violation = raw_stage2_edge & ~(stage1 & stage15)
            if torch.any(order_violation):
                raise RuntimeError("A36 Stage2 edge violated Stage1.5 ordering")

            eligible = _genuine_stage2_edges(
                stage1=stage1,
                stage15=stage15,
                stage2=stage2,
                previous_stage2=previous_stage2,
                episode_steps=base_env.episode_length_buf,
                dones=done_mask,
                nan_termination=nan_termination,
                already_captured=captured_environment,
                reset_family=family,
                reset_mode=reset_mode,
                source_rows=source_rows,
            )
            candidate_ids = eligible.nonzero(as_tuple=False).squeeze(-1)
            captured_environment[candidate_ids] = True

            remaining = args.target_states - sum(walk_state_count(chunk) for chunk in chunks)
            if len(candidate_ids) > 0 and remaining > 0:
                candidate = capture_walk_state_rows(base_env, candidate_ids)
                if not torch.all(_finite_state_rows(candidate)):
                    raise RuntimeError("Captured A36 Stage2 state is non-finite")

                keep_rows: list[int] = []
                for row in range(len(candidate_ids)):
                    state_digest = _exact_state_row_digest(candidate, row)
                    if state_digest in seen_state_digests:
                        duplicate_states_rejected += 1
                        continue
                    seen_state_digests.add(state_digest)
                    keep_rows.append(row)
                    if len(keep_rows) >= remaining:
                        break

                if keep_rows:
                    keep = torch.tensor(keep_rows, dtype=torch.long)
                    ids = candidate_ids[keep.to(args.device)]
                    chunk = _select_nested_rows(candidate, keep)
                    kept_source_rows = source_rows[ids].detach().cpu().long()
                    chunk.update(
                        {
                            "source_env_index": ids.detach().cpu().clone(),
                            "source_episode_index": episode_index[ids]
                            .detach()
                            .cpu()
                            .clone(),
                            "source_episode_step": base_env.episode_length_buf[ids]
                            .detach()
                            .cpu()
                            .clone(),
                            "source_collector_control_step": torch.full(
                                (len(ids),), steps_run, dtype=torch.long
                            ),
                            "source_state_bank_row": kept_source_rows.clone(),
                            "source_state_bank_source_step": _required_env_tensor(
                                base_env, "_stair_walker_bank_source_step"
                            )[ids]
                            .detach()
                            .cpu()
                            .clone(),
                            "source_reset_family": family[ids]
                            .detach()
                            .cpu()
                            .clone(),
                            "source_reset_mode": reset_mode[ids]
                            .detach()
                            .cpu()
                            .clone(),
                            "source_terrain_level": base_env.scene.terrain.terrain_levels[
                                ids
                            ]
                            .detach()
                            .cpu()
                            .clone(),
                            "captured_stage1_policy_achieved": stage1[ids]
                            .detach()
                            .cpu()
                            .clone(),
                            "captured_stage15_policy_achieved": stage15[ids]
                            .detach()
                            .cpu()
                            .clone(),
                            "captured_stage2_policy_achieved": stage2[ids]
                            .detach()
                            .cpu()
                            .clone(),
                            "captured_stage2_policy_edge": torch.ones(
                                len(ids), dtype=torch.bool
                            ),
                            "captured_true_shell_clearance": shell[ids]
                            .detach()
                            .cpu()
                            .clone(),
                        }
                    )
                    chunk.update(
                        _source_bank_provenance(source_states, kept_source_rows)
                    )
                    chunks.append(chunk)

            previous_stage2.copy_(stage2)
            previous_stage2[done_mask] = False
            reset_audited[done_mask] = False
            episode_index[done_mask] += 1

            collected = sum(walk_state_count(chunk) for chunk in chunks)
            if steps_run % 100 == 0 or collected >= args.target_states:
                print(
                    f"[a36-stage2-bank] step={steps_run} "
                    f"states={collected}/{args.target_states} "
                    f"first_edges={int(captured_environment.sum().item())} "
                    f"duplicates={duplicate_states_rejected}"
                )
            if collected >= args.target_states:
                break
    finally:
        env.close()

    if not chunks:
        raise SystemExit(
            f"No genuine A36 family-3 Stage2 edges in {steps_run} control steps"
        )
    states = concatenate_walk_state_rows(chunks)
    count = walk_state_count(states)
    if count < args.target_states:
        raise SystemExit(
            f"Collected only {count}/{args.target_states} unique first-edge states "
            f"in {steps_run} control steps"
        )

    bank: dict[str, object] = {
        "schema_version": BANK_SCHEMA_VERSION,
        "metadata": {
            "created_at": datetime.now(UTC).isoformat(),
            "task": TASK_ID,
            "source": "a36_family3_policy_created_stage2_forward",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "input_bank": str(input_bank_path),
            "input_bank_sha256": _sha256(input_bank_path),
            "input_bank_source": source_bank.get("metadata", {}).get("source"),
            "input_bank_num_states": input_state_count,
            "riser_height_m": STANDARD_RISER_HEIGHT_M,
            "tread_depth_m": STANDARD_TREAD_DEPTH_M,
            "num_steps": STANDARD_NUM_STEPS,
            "num_states": count,
            "num_envs": args.num_envs,
            "steps_run": steps_run,
            "seed": args.seed,
            "joint_names": runtime_metadata["joint_names"],
            "physics_dt": runtime_metadata["physics_dt"],
            "step_dt": runtime_metadata["step_dt"],
            "decimation": runtime_metadata["decimation"],
            "mjlab_version": version("mjlab"),
            "actor_observation_dim": ACTOR_OBSERVATION_DIM,
            "action_dim": ACTION_DIM,
            "source_reset_family": SOURCE_RESET_FAMILY,
            "source_reset_mode": SOURCE_RESET_MODE,
            "source_terrain_level": HARD_TERRAIN_LEVEL,
            "minimum_policy_steps": MIN_POLICY_STEPS,
            "first_policy_created_stage2_edge_per_environment": True,
            "reset_credit_suppressed": True,
            "ordered_stage1_stage15_required": True,
            "exact_post_physics_capture": True,
            "exact_full_state_deduplication": True,
            "duplicate_states_rejected": duplicate_states_rejected,
            "nan_termination_events": 0,
            "command_observations_preserved": True,
        },
        "states": states,
    }
    _write_bank_atomic(output, bank)
    print(f"[a36-stage2-bank] wrote {count} states to {output}")
    print(f"[a36-stage2-bank] checkpoint_sha256={bank['metadata']['checkpoint_sha256']}")
    print(f"[a36-stage2-bank] input_bank_sha256={bank['metadata']['input_bank_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Headless left/right evaluation for a single-leg-stand checkpoint."""

import argparse
import contextlib
import json
import sys
from dataclasses import asdict

TASK = "Mjlab-SingleLegStand-Flat-MicroDuck"
PERCENTILES = (0, 10, 25, 50, 75, 90, 100)


def summarize(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    result = {"mean": sum(ordered) / len(ordered)}
    for percentile in PERCENTILES:
        position = (len(ordered) - 1) * percentile / 100
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        fraction = position - lower
        result[f"p{percentile}"] = (
            ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
        )
    return result


def evaluate(
    checkpoint: str,
    side: int,
    num_envs: int,
    episodes: int,
    episode_seconds: float,
    seed: int,
    device: str,
    strict: bool,
) -> dict:
    import torch
    from rsl_rl.runners import OnPolicyRunner

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.utils.lab_api.math import matrix_from_quat

    env_cfg = load_env_cfg(TASK, play=True)
    env_cfg.seed = seed
    env_cfg.scene.num_envs = num_envs
    env_cfg.episode_length_s = episode_seconds
    env_cfg.commands["twist"].fixed_side = side
    env_cfg.commands["twist"].resampling_time_range = (
        episode_seconds,
        episode_seconds,
    )
    agent_cfg = load_rl_cfg(TASK)

    env = RslRlVecEnvWrapper(
        ManagerBasedRlEnv(cfg=env_cfg, device=device),
        clip_actions=agent_cfg.clip_actions,
    )
    runner_cls = load_runner_cls(TASK) or OnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
        checkpoint,
        load_cfg={"actor": True},
        strict=True,
        map_location=device,
    )
    policy = runner.get_inference_policy(device=device)
    obs = env.get_observations()

    achieved = torch.zeros(num_envs, dtype=torch.bool, device=device)
    first_success_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    episode_step = torch.zeros(num_envs, dtype=torch.long, device=device)
    completed = successes = survived = 0
    success_times: list[float] = []
    gate_totals = {
        name: torch.zeros((), device=device)
        for name in (
            "support_contact",
            "swing_clearance",
            "swing_no_contact",
            "com_inside",
            "tilt",
            "quiet",
            "nonfoot_clear",
            "all_instant",
        )
    }
    eligible_total = torch.zeros((), device=device)
    clearance_total = torch.zeros((), device=device)
    no_contact_run_s = torch.zeros(num_envs, device=device)
    valid_run_s = torch.zeros(num_envs, device=device)
    episode_max_no_contact_s = torch.zeros(num_envs, device=device)
    episode_max_valid_s = torch.zeros(num_envs, device=device)
    contact_transitions = torch.zeros(num_envs, device=device)
    eligible_steps = torch.zeros(num_envs, device=device)
    previous_contact = torch.zeros(num_envs, dtype=torch.bool, device=device)
    previous_contact_valid = torch.zeros(num_envs, dtype=torch.bool, device=device)
    max_no_contact_times: list[float] = []
    max_valid_times: list[float] = []
    transition_rates: list[float] = []

    with torch.inference_mode():
        while completed < episodes:
            obs, _, dones, extras = env.step(policy(obs))
            episode_step += 1
            raw = env.unwrapped
            time_outs = extras.get("time_outs", raw.reset_time_outs)
            alpha = raw.command_manager.get_term("twist").alpha
            eligible = alpha >= 0.99
            support = 0 if side == -1 else 1
            swing = 1 - support
            contacts = (
                raw.scene.sensors["feet_ground_contact"].data.found.reshape(
                    num_envs, -1
                )[:, :2]
                > 0.0
            )
            asset = raw.scene["robot"]
            feet_ids = asset.find_sites(("left_foot", "right_foot"))[0]
            feet = asset.data.site_pos_w[:, feet_ids]
            clearance = feet[:, swing, 2] - feet[:, support, 2]
            delta_w = asset.data.root_com_pos_w - feet[:, support]
            delta_b = torch.bmm(
                matrix_from_quat(asset.data.root_link_quat_w).transpose(1, 2),
                delta_w.unsqueeze(-1),
            ).squeeze(-1)
            com_inside = (
                torch.square(delta_b[:, 0] / 0.025)
                + torch.square(delta_b[:, 1] / 0.018)
            ) <= 1.0
            quat = asset.data.root_link_quat_w
            cos_tilt = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
            quiet = (
                torch.linalg.vector_norm(asset.data.root_link_lin_vel_w, dim=-1) < 0.08
            ) & (torch.linalg.vector_norm(asset.data.root_link_ang_vel_w, dim=-1) < 0.8)
            nonfoot_clear = (
                raw.scene.sensors["nonfoot_ground_contact"]
                .data.found.reshape(num_envs, -1)
                .sum(dim=-1)
                == 0.0
            )
            clearance_threshold = 0.003 if strict else 0.008
            gates = {
                "support_contact": contacts[:, support],
                "swing_clearance": clearance >= clearance_threshold,
                "swing_no_contact": ~contacts[:, swing],
                "com_inside": com_inside,
                "tilt": cos_tilt
                >= torch.cos(torch.tensor(0.6108652382, device=device)),
                "quiet": quiet,
                "nonfoot_clear": nonfoot_clear,
            }
            required = [
                gate
                for name, gate in gates.items()
                if not strict or name != "com_inside"
            ]
            gates["all_instant"] = torch.stack(required).all(dim=0)
            swing_contact = contacts[:, swing]
            no_contact_run_s = torch.where(
                eligible & ~swing_contact,
                no_contact_run_s + env.unwrapped.step_dt,
                torch.zeros_like(no_contact_run_s),
            )
            valid_run_s = torch.where(
                eligible & gates["all_instant"],
                valid_run_s + env.unwrapped.step_dt,
                torch.zeros_like(valid_run_s),
            )
            episode_max_no_contact_s = torch.maximum(
                episode_max_no_contact_s, no_contact_run_s
            )
            episode_max_valid_s = torch.maximum(episode_max_valid_s, valid_run_s)
            contact_transitions += (
                eligible
                & previous_contact_valid
                & (swing_contact != previous_contact)
            )
            eligible_steps += eligible
            previous_contact = swing_contact
            previous_contact_valid = eligible
            eligible_total += eligible.sum()
            clearance_total += torch.where(eligible, clearance, 0.0).sum()
            for name, gate in gates.items():
                gate_totals[name] += (gate & eligible).sum()

            current = valid_run_s >= 1.0
            first = current & ~achieved
            first_success_step[first] = episode_step[first]
            achieved |= current

            done_ids = torch.nonzero(dones, as_tuple=False).flatten()
            for idx in done_ids.tolist():
                if completed >= episodes:
                    break
                completed += 1
                survived += int(bool(time_outs[idx]))
                if bool(achieved[idx]):
                    successes += 1
                    success_times.append(
                        float(first_success_step[idx]) * env.unwrapped.step_dt
                    )
                max_no_contact_times.append(float(episode_max_no_contact_s[idx]))
                max_valid_times.append(float(episode_max_valid_s[idx]))
                eligible_s = float(eligible_steps[idx]) * env.unwrapped.step_dt
                transition_rates.append(
                    float(contact_transitions[idx]) / max(eligible_s, 1e-6)
                )
            achieved[done_ids] = False
            first_success_step[done_ids] = -1
            episode_step[done_ids] = 0
            no_contact_run_s[done_ids] = 0.0
            valid_run_s[done_ids] = 0.0
            episode_max_no_contact_s[done_ids] = 0.0
            episode_max_valid_s[done_ids] = 0.0
            contact_transitions[done_ids] = 0.0
            eligible_steps[done_ids] = 0.0
            previous_contact_valid[done_ids] = False

    env.close()
    eligible_count = float(eligible_total)

    def eligible_rate(value: torch.Tensor) -> float | None:
        return float(value) / eligible_count if eligible_count > 0.0 else None

    swing_no_contact_rate = eligible_rate(gate_totals["swing_no_contact"])
    nonfoot_clear_rate = eligible_rate(gate_totals["nonfoot_clear"])
    return {
        "side": "left" if side == -1 else "right",
        "episodes": completed,
        "episode_seconds": episode_seconds,
        "seed": seed,
        "strict": strict,
        "successes": successes,
        "success_rate": successes / completed,
        "survivals": survived,
        "survival_rate": survived / completed,
        "mean_time_to_success_s": (
            sum(success_times) / len(success_times) if success_times else None
        ),
        "mean_swing_clearance_m": (
            float(clearance_total) / eligible_count if eligible_count > 0.0 else None
        ),
        "swing_foot_contact_fraction": (
            1.0 - swing_no_contact_rate
            if swing_no_contact_rate is not None
            else None
        ),
        "nonfoot_contact_fraction": (
            1.0 - nonfoot_clear_rate if nonfoot_clear_rate is not None else None
        ),
        "gate_rates": {
            name: eligible_rate(value) for name, value in gate_totals.items()
        },
        "continuity": {
            "longest_swing_no_contact_s": summarize(max_no_contact_times),
            "longest_all_gates_s": summarize(max_valid_times),
            "mean_contact_transitions_per_s": sum(transition_rates)
            / len(transition_rates),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--episode-seconds", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    if args.num_envs <= 0:
        parser.error("--num-envs must be positive")
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.episode_seconds <= 0.0:
        parser.error("--episode-seconds must be positive")

    with contextlib.redirect_stdout(sys.stderr):
        results = [
            evaluate(
                args.checkpoint,
                side,
                args.num_envs,
                args.episodes,
                args.episode_seconds,
                args.seed,
                args.device,
                args.strict,
            )
            for side in (-1, 1)
        ]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

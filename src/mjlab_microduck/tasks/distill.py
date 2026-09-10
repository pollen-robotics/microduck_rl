"""PPO + fallen-gated behavior cloning from a frozen expert policy.

Why (protective-fall velstand, 2026-09-10): two warm-started runs (4otqmkb4,
1bqctpkq) fell on command and NEVER got up — 0 recoveries in ~1000 iterations,
fallen action rate 1/4 of upright. Removing the attempt tax did not help: with
the warm-started walk's low action noise (std ≈ 0.2), random flailing from
lying never produces a partial rise, so the potential-based recovery terms
never pay and there is no gradient at all. Meanwhile the deployed stand expert
(wandb 69u48n8l @ model_9750 = alpha_stand.onnx) recovers 100 % from
face-down / face-up and 95 % from the side on the SAME 61D obs and the same
all-collisions model, within ~1 s. The skill exists; the student cannot
discover it. So we distill it.

Mechanism (DAgger-flavoured): after every PPO update, run a behavior-cloning
pass over the observations the STUDENT just collected, restricted to fallen
frames (tilt > gate, read from the projected-gravity block of the actor obs),
regressing the student's deterministic action onto the expert's. States come
from the student's own distribution, labels from the expert, so the imitation
covers exactly the states the student visits. Upright frames are untouched:
the walk is never pulled toward the stand expert.

The expert is instantiated as a deep copy of the student actor (identical
architecture across the policy family) and loaded from its rsl_rl checkpoint,
so its OWN obs normalizer is used. Its twist command slot is zeroed on input:
the stand expert never saw a non-zero twist (normalizer std 0.005) and would
receive a 40σ input otherwise.

Obs-contract dependencies (61D actor obs, see AGENTS.md): projected gravity at
[3:6], twist command at [48:51]. Both are cfg parameters, defaulted here.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

import torch
from rsl_rl.algorithms import PPO
from tensordict import TensorDict

from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg

EXPERT_CACHE_DIR = Path("logs/rsl_rl/expert_cache")


def default_bc_cfg() -> dict:
    return {
        # Expert checkpoint: either a local ``checkpoint_path`` or a wandb run.
        "wandb_run_path": "pollen-robotics/mjlab_microduck/69u48n8l",
        "checkpoint_name": "model_9750.pt",
        "checkpoint_path": None,
        "coef": 1.0,               # MSE weight (actions in rad)
        "learning_rate": 1e-3,     # dedicated Adam (PPO's adaptive-KL LR must not throttle the BC)
        "gate_tilt_deg": 35.0,     # BC only where tilt > this (fallen frames)
        "epochs": 5,               # × mini_batches = 20 BC steps/iter, matching PPO's 5×4
        "mini_batches": 4,
        "min_samples": 64,         # skip the pass when fewer fallen frames were collected
        "gravity_slice": (3, 6),   # projected gravity in the actor obs
        "twist_slice": (48, 51),   # twist command slot → zeroed for the expert
    }


@dataclass
class PpoWithExpertBcCfg(PpoWithSymmetryCfg):
    """PpoWithSymmetryCfg + fallen-gated expert behavior cloning."""

    bc_cfg: dict | None = field(default_factory=default_bc_cfg)
    class_name: str = "mjlab_microduck.tasks.distill.PpoWithExpertBc"


def fallen_mask_from_obs(obs: torch.Tensor, gravity_slice: tuple[int, int], gate_tilt_deg: float) -> torch.Tensor:
    """tilt > gate from the projected-gravity block: upright → g = (0, 0, -1), cos(tilt) = -g_z."""
    g = obs[:, gravity_slice[0]:gravity_slice[1]]
    g = g / g.norm(dim=1, keepdim=True).clamp_min(1e-6)  # obs noise / IMU DR: renormalize
    cos_tilt = -g[:, 2]
    return cos_tilt < torch.cos(torch.deg2rad(torch.tensor(gate_tilt_deg, device=obs.device)))


def expert_input(obs: torch.Tensor, twist_slice: tuple[int, int]) -> torch.Tensor:
    out = obs.clone()
    out[:, twist_slice[0]:twist_slice[1]] = 0.0
    return out


def load_expert_from(actor: torch.nn.Module, state_dict: dict) -> torch.nn.Module:
    expert = copy.deepcopy(actor)
    expert.load_state_dict(state_dict, strict=True)
    expert.eval()
    for p in expert.parameters():
        p.requires_grad_(False)
    return expert


def _resolve_expert_checkpoint(bc_cfg: dict) -> Path:
    if bc_cfg.get("checkpoint_path"):
        return Path(bc_cfg["checkpoint_path"])
    from mjlab.utils.os import get_wandb_checkpoint_path

    path, cached = get_wandb_checkpoint_path(
        EXPERT_CACHE_DIR, Path(bc_cfg["wandb_run_path"]), bc_cfg.get("checkpoint_name")
    )
    print(f"[distill] expert checkpoint {path} ({'cached' if cached else 'downloaded'})")
    return path


class PpoWithExpertBc(PPO):
    def __init__(self, actor, critic, storage, *args, bc_cfg: dict | None = None, **kwargs) -> None:
        super().__init__(actor, critic, storage, *args, **kwargs)
        self.bc_cfg = bc_cfg
        self.expert = None
        if bc_cfg:
            ckpt = torch.load(_resolve_expert_checkpoint(bc_cfg), map_location=self.device, weights_only=False)
            self.expert = load_expert_from(self.actor, ckpt["actor_state_dict"]).to(self.device)
            # Own optimizer: PPO's adaptive-KL schedule and shared Adam moments
            # would otherwise throttle/mix the BC step (first path check: 4 BC
            # steps through the PPO optimizer lost to 20 PPO steps per iteration).
            self.bc_optimizer = torch.optim.Adam(self.actor.parameters(), lr=bc_cfg["learning_rate"])
            print(
                f"[distill] fallen-gated expert BC ON: coef={bc_cfg['coef']} gate>{bc_cfg['gate_tilt_deg']}° "
                f"epochs={bc_cfg['epochs']} mini_batches={bc_cfg['mini_batches']} (expert iter {ckpt.get('iter')})"
            )

    # ── BC pass ───────────────────────────────────────────────────────────────

    def _bc_update(self) -> tuple[float, float]:
        cfg = self.bc_cfg
        obs_td: TensorDict = self.storage.observations.flatten(0, 1)  # (T*N, ...)
        flat = torch.cat([obs_td[g] for g in self.actor.obs_groups], dim=-1)
        mask = fallen_mask_from_obs(flat, tuple(cfg["gravity_slice"]), cfg["gate_tilt_deg"])
        frac = mask.float().mean().item()
        idx = mask.nonzero().flatten()
        if idx.numel() < cfg["min_samples"]:
            return 0.0, frac
        obs_td = obs_td[idx]
        with torch.no_grad():
            exp_td = obs_td.clone()
            exp_flat = expert_input(torch.cat([exp_td[g] for g in self.actor.obs_groups], dim=-1), tuple(cfg["twist_slice"]))
            # rebuild the TensorDict groups from the masked flat obs
            off = 0
            for g in self.actor.obs_groups:
                d = exp_td[g].shape[-1]
                exp_td[g] = exp_flat[:, off:off + d]
                off += d
            target = self.expert(exp_td)
        n = idx.numel()
        mb = max(1, n // cfg["mini_batches"])
        total = 0.0
        steps = 0
        for _ in range(cfg["epochs"]):
            perm = torch.randperm(n, device=self.device)
            for s in range(0, n, mb):
                sel = perm[s:s + mb]
                pred = self.actor(obs_td[sel])
                loss = cfg["coef"] * torch.nn.functional.mse_loss(pred, target[sel])
                self.bc_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.bc_optimizer.step()
                total += loss.item()
                steps += 1
        return total / max(steps, 1), frac

    def update(self) -> dict[str, float]:
        loss_dict = super().update()
        if self.expert is not None:
            bc_loss, frac = self._bc_update()
            loss_dict["expert_bc"] = bc_loss
            loss_dict["expert_bc_fallen_frac"] = frac
        return loss_dict

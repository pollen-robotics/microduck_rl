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

Run-3 lesson (wandb 4lflk7ii, 2026-09-10): the stand-up was learned within
~100 iterations — and the WALK died at the very first BC pass. Iterations 0-7
had too few fallen frames for a pass; iteration 8 ran 20 Adam steps at 1e-3 on
a few hundred fallen frames and by iteration 12 fall terminations went 2 → 72.
Nothing constrained the shared weights on upright frames, so the imitation
rewrote the walk, which produced more falls, more fallen frames, more BC. Fix:
a second frozen teacher — the warm-start WALK checkpoint itself — anchors the
upright frames (tilt < anchor gate) to the behavior the student started with,
while the stand expert teaches the fallen frames (tilt > gate). The band in
between belongs to PPO alone, as does everything the reward shapes on top
(protective landing, impact costs, the last mile of the rise). The pass is also
gentler: lr 3e-4 and a minimum mini-batch so a first pass on a handful of
frames cannot take 20 full steps.

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
        # Expert checkpoint, first that resolves: a local ``checkpoint_path``, a Hub training
        # bundle ``hf_checkpoint`` (hf://..., checkpoints.py — no wandb access needed), a wandb run.
        "wandb_run_path": "pollen-robotics/mjlab_microduck/69u48n8l",
        "checkpoint_name": "model_9750.pt",
        "checkpoint_path": None,
        "hf_checkpoint": "hf://pollen-robotics/microduck-training/bundles/microduck_stand/69u48n8l_9750",
        "coef": 1.0,               # MSE weight (actions in rad)
        "learning_rate": 3e-4,     # dedicated Adam (PPO's adaptive-KL LR must not throttle the BC)
        "gate_tilt_deg": 35.0,     # stand expert teaches frames with tilt > this
        "epochs": 5,               # × mini_batches = up to 20 BC steps/iter
        "mini_batches": 4,
        "min_mini_batch": 512,     # never take a full 20-step pass on a handful of frames
        "min_samples": 64,         # skip the fallen term when fewer fallen frames were collected
        "gravity_slice": (3, 6),   # projected gravity in the actor obs
        "twist_slice": (48, 51),   # twist command slot → zeroed for the stand expert
        # Walk anchor (run-3 lesson): the warm-start walk teaches frames with tilt < anchor_tilt_deg.
        "anchor_wandb_run_path": "pollen-robotics/mjlab_microduck/441tzs6d",
        "anchor_checkpoint_name": "model_3750.pt",
        "anchor_checkpoint_path": None,
        "anchor_hf_checkpoint": "hf://pollen-robotics/microduck-training/bundles/velocity2/441tzs6d_3750",
        "anchor_coef": 1.0,
        "anchor_tilt_deg": 25.0,
        # Body-control routing (2026-10, branch improve_velstand2). When body_slice is set,
        # STANDING frames (twist obs exactly 0) with a non-zero body command on
        # body_active_axes are taken off the walk anchor (the walk expert ignores body
        # commands and would pin them to a level stance). Of those, frames whose
        # body_no_teacher_axes are all zero (tilt-only) are taught by the stand expert,
        # which tracks roll/pitch (9.2-9.5° for 10° commanded); frames commanding any of
        # body_no_teacher_axes (z: the stand expert does not crouch) are left to PPO.
        # The command is in the obs, so these frames are separable — unlike the
        # rise-catch / push-stumble split that leaked in run rqunethx.
        "body_slice": None,
        # Turn-in-place frames (linear command exactly 0, |yaw command| > 0) are taken
        # off the walk anchor: alpha_walking stands still below ~0.6 rad/s (the dead zone
        # being trained out), and coef-1.0 BC pinned every earlier yaw fix. Separable:
        # the command is in the obs.
        "unanchor_turn_in_place": False,
        # Teacher-side yaw remap (2026-10-12, branch improve_standup_velstand). The walk
        # anchor has a yaw dead zone (in place: fed 0.6 → 0.0 rad/s, 1.0 → ~0.25; walking
        # ≈ 0.6 × the command) and coef-1.0 BC copies it into every student. Unanchoring
        # turn frames lost turning twice (sape62zb, brkpcnn8: PPO does not discover slow
        # turning), but the anchor turns monotonically when fed a bigger yaw. So the anchor
        # is fed the yaw command that makes IT achieve the student's commanded rate,
        # looked up in a table calibrated on the anchor itself
        # (claude_experiments/calibrate_walk_yaw.py): {"lin": |v_xy| knots, "targets":
        # |wz| knots, "pos"/"neg": fed |wz| per (lin, target)}. A pure function of the obs,
        # so labels stay consistent; zero yaw (standing, straight walking) is unchanged.
        "anchor_yaw_remap": None,
        "body_active_axes": (2, 3, 4),
        "body_no_teacher_axes": (2,),
        # Flat actor-obs columns ZEROED in the stand expert's input on fallen frames (tilt >
        # gate_tilt_deg; never on body-control frames). Zero = HOME for joint_pos_rel / zero
        # velocity / zero last action. 2026-10 (branch improve_standup_velstand): alpha_stand
        # freezes on its back in a fixed point it reads off its own joints, and the student copies
        # it. Masking only the head moved the freeze (run s4jdgkux); masking every joint col
        # (6:48) above 75° makes the label the HOME-pose rise, which gets the teacher up from
        # random-joint / random-orientation / sitting starts and rescues 98-99 % of the student's
        # stuck states (emulated). A pure function of the obs, so labels stay consistent. Masking
        # from the 35° BC gate broke the END of the rise (the expert needs its joints there).
        "expert_mask_cols": (),
        "expert_mask_tilt_deg": 75.0,
        # Bounded holds in the stand expert's labels (2026-10-30, branch gear_protection). The
        # XL330 bench found the teeth skip during the HOLD after an impact, and life went ×3.6
        # when the servo stopped pushing hard against the obstacle; the stall torque of a
        # position servo is set by its error, so a label that asks a joint which is not moving
        # to go far past where it is asks for a full-torque hold. When set, on fallen frames the
        # expert's target for each listed joint whose |velocity| < vel_thresh is clamped to
        # within max_err (rad) of its current position: the label keeps the direction, the
        # student learns a bounded push. {"joints": servo indices, "max_err": rad,
        # "vel_thresh": rad/s, "pos_slice": joint_pos_rel cols, "vel_slice": joint_vel cols}.
        # Actions and joint_pos_rel share HOME as their origin (action scale 1.0).
        "label_hold_clamp": None,
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


def turn_in_place_mask(obs: torch.Tensor, twist_slice: tuple[int, int], eps: float = 1e-6) -> torch.Tensor:
    """Linear command exactly 0 and a non-zero yaw command (command obs carry no noise)."""
    tw = obs[:, twist_slice[0]:twist_slice[1]]
    return (tw[:, :2].abs().amax(dim=1) <= eps) & (tw[:, 2].abs() > eps)


def body_frame_masks(
    obs: torch.Tensor, twist_slice: tuple[int, int], body_slice: tuple[int, int],
    active_axes: tuple[int, ...], no_teacher_axes: tuple[int, ...], eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(body frames → off the walk anchor, teach frames → stand expert). Command obs carry
    no noise, so "twist == 0" and "axis == 0" are exact tests."""
    standing = obs[:, twist_slice[0]:twist_slice[1]].abs().amax(dim=1) <= eps
    body = obs[:, body_slice[0]:body_slice[1]]
    active = body[:, list(active_axes)].abs().amax(dim=1) > eps
    frames = standing & active
    if no_teacher_axes:
        frames_teach = frames & (body[:, list(no_teacher_axes)].abs().amax(dim=1) <= eps)
    else:
        frames_teach = frames
    return frames, frames_teach


def expert_input(
    obs: torch.Tensor, twist_slice: tuple[int, int],
    mask_cols: tuple[int, ...] = (), mask_rows: torch.Tensor | None = None,
) -> torch.Tensor:
    out = obs.clone()
    out[:, twist_slice[0]:twist_slice[1]] = 0.0
    if mask_cols and mask_rows is not None and mask_rows.any():
        cols = torch.as_tensor(mask_cols, device=obs.device, dtype=torch.long)
        rows = mask_rows.nonzero().flatten()
        out[rows.unsqueeze(1), cols.unsqueeze(0)] = 0.0
    return out


def clamp_label_holds(flat_obs: torch.Tensor, labels: torch.Tensor, cfg: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """(labels with stalled joints' targets held within max_err of their position, mask of clamped entries)."""
    cols = torch.as_tensor(cfg["joints"], device=labels.device, dtype=torch.long)
    p0, p1 = cfg.get("pos_slice", (6, 20))
    v0, v1 = cfg.get("vel_slice", (20, 34))
    q = flat_obs[:, p0:p1][:, cols]
    qd = flat_obs[:, v0:v1][:, cols]
    lab = labels[:, cols]
    err = lab - q
    e = float(cfg["max_err"])
    clamp = (qd.abs() < cfg.get("vel_thresh", 0.5)) & (err.abs() > e)
    out = labels.clone()
    out[:, cols] = torch.where(clamp, q + err.clamp(-e, e), lab)
    return out, clamp


def _interp_rows(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """Piecewise-linear interpolation of x (n,) on knots xp (K,) for per-row values fp (n, K); x clamped to the knots."""
    x = x.clamp(xp[0], xp[-1])
    hi = torch.searchsorted(xp, x, right=True).clamp(1, xp.numel() - 1)
    lo = hi - 1
    w = (x - xp[lo]) / (xp[hi] - xp[lo])
    return fp.gather(1, lo.unsqueeze(1)).squeeze(1) * (1 - w) + fp.gather(1, hi.unsqueeze(1)).squeeze(1) * w


def remap_anchor_yaw(obs: torch.Tensor, twist_slice: tuple[int, int], table: dict) -> torch.Tensor:
    """Copy of the flat obs with the yaw command replaced by the one the walk anchor needs to
    actually turn at that rate (bilinear in |v_xy| and |wz|, sign kept, 0 → 0)."""
    out = obs.clone()
    tw = obs[:, twist_slice[0]:twist_slice[1]]
    wz = tw[:, 2]
    lin = tw[:, :2].norm(dim=1)
    dev = obs.device
    lk = torch.as_tensor(table["lin"], dtype=obs.dtype, device=dev)
    tk = torch.as_tensor([0.0, *table["targets"]], dtype=obs.dtype, device=dev)
    fed = torch.zeros_like(wz)
    for key, sign in (("pos", 1.0), ("neg", -1.0)):
        rows = (wz * sign) > 0
        if not rows.any():
            continue
        grid = torch.as_tensor([[0.0, *r] for r in table[key]], dtype=obs.dtype, device=dev)  # (L, K)
        c = wz[rows].abs()
        n = c.numel()
        per_lin = torch.stack([_interp_rows(c, tk, grid[i].expand(n, -1)) for i in range(lk.numel())], dim=1)  # (n, L)
        fed[rows] = sign * _interp_rows(lin[rows], lk, per_lin)
    out[:, twist_slice[0] + 2] = fed
    return out


def _with_flat(td: TensorDict, groups: list[str], flat: torch.Tensor) -> TensorDict:
    """Rebuild the obs groups of ``td`` from a modified flat obs."""
    off = 0
    for g in groups:
        d = td[g].shape[-1]
        td[g] = flat[:, off:off + d]
        off += d
    return td


def load_expert_from(actor: torch.nn.Module, state_dict: dict) -> torch.nn.Module:
    expert = copy.deepcopy(actor)
    expert.load_state_dict(state_dict, strict=True)
    expert.eval()
    for p in expert.parameters():
        p.requires_grad_(False)
    return expert


def _resolve_checkpoint(bc_cfg: dict, prefix: str = "") -> Path | None:
    """Local ``<prefix>checkpoint_path``, else the ``<prefix>hf_checkpoint`` bundle, else ``<prefix>wandb_run_path``.

    A failing Hub bundle (private repo, no token, offline) falls through to wandb when one is set.
    """
    if bc_cfg.get(f"{prefix}checkpoint_path"):
        return Path(bc_cfg[f"{prefix}checkpoint_path"])
    run = bc_cfg.get(f"{prefix}wandb_run_path")
    if bc_cfg.get(f"{prefix}hf_checkpoint"):
        from mjlab_microduck.checkpoints import resolve

        try:
            return resolve(bc_cfg[f"{prefix}hf_checkpoint"])
        except Exception as e:
            if not run:
                raise
            print(f"[distill] {prefix}hf_checkpoint unavailable ({e!r}); falling back to wandb {run}")
    if not run:
        return None
    from mjlab.utils.os import get_wandb_checkpoint_path

    path, cached = get_wandb_checkpoint_path(EXPERT_CACHE_DIR, Path(run), bc_cfg.get(f"{prefix}checkpoint_name"))
    print(f"[distill] {prefix or 'expert_'}checkpoint {path} ({'cached' if cached else 'downloaded'})")
    return path


class PpoWithExpertBc(PPO):
    def __init__(self, actor, critic, storage, *args, bc_cfg: dict | None = None, **kwargs) -> None:
        super().__init__(actor, critic, storage, *args, **kwargs)
        self.bc_cfg = bc_cfg
        self.expert = None
        self.anchor = None
        if bc_cfg:
            ckpt = torch.load(_resolve_checkpoint(bc_cfg), map_location=self.device, weights_only=False)
            self.expert = load_expert_from(self.actor, ckpt["actor_state_dict"]).to(self.device)
            anchor_path = _resolve_checkpoint(bc_cfg, "anchor_") if bc_cfg.get("anchor_coef", 0.0) > 0 else None
            if anchor_path is not None:
                ack = torch.load(anchor_path, map_location=self.device, weights_only=False)
                self.anchor = load_expert_from(self.actor, ack["actor_state_dict"]).to(self.device)
            # Own optimizer: PPO's adaptive-KL schedule and shared Adam moments
            # would otherwise throttle/mix the BC step (first path check: 4 BC
            # steps through the PPO optimizer lost to 20 PPO steps per iteration).
            self.bc_optimizer = torch.optim.Adam(self.actor.parameters(), lr=bc_cfg["learning_rate"])
            print(
                f"[distill] expert BC ON: stand expert (iter {ckpt.get('iter')}) on tilt>{bc_cfg['gate_tilt_deg']}° coef={bc_cfg['coef']}; "
                f"walk anchor {'ON on tilt<' + str(bc_cfg['anchor_tilt_deg']) + '° coef=' + str(bc_cfg['anchor_coef']) if self.anchor is not None else 'OFF'}; "
                f"lr={bc_cfg['learning_rate']} epochs={bc_cfg['epochs']} mini_batches={bc_cfg['mini_batches']} min_mb={bc_cfg.get('min_mini_batch', 1)}"
            )

    # ── BC pass ───────────────────────────────────────────────────────────────

    def _bc_update(self) -> dict[str, float]:
        cfg = self.bc_cfg
        obs_td: TensorDict = self.storage.observations.flatten(0, 1)  # (T*N, ...)
        groups = list(self.actor.obs_groups)
        flat = torch.cat([obs_td[g] for g in groups], dim=-1)
        gsl = tuple(cfg["gravity_slice"])
        fallen = fallen_mask_from_obs(flat, gsl, cfg["gate_tilt_deg"])
        upright = ~fallen_mask_from_obs(flat, gsl, cfg["anchor_tilt_deg"]) if self.anchor is not None else torch.zeros_like(fallen)
        body_teach = torch.zeros_like(fallen)
        if cfg.get("body_slice") is not None:
            body_frames, body_teach = body_frame_masks(flat, tuple(cfg["twist_slice"]), tuple(cfg["body_slice"]),
                                                       tuple(cfg["body_active_axes"]), tuple(cfg["body_no_teacher_axes"]))
            upright = upright & ~body_frames
            body_teach = body_teach & ~fallen
        turn_free = torch.zeros_like(fallen)
        if cfg.get("unanchor_turn_in_place"):
            turn_free = turn_in_place_mask(flat, tuple(cfg["twist_slice"]))
            upright = upright & ~turn_free
        stats = {"expert_bc_fallen_frac": fallen.float().mean().item(), "expert_bc_anchor_frac": upright.float().mean().item(),
                 "expert_bc_body_teach_frac": body_teach.float().mean().item(),
                 "expert_bc_turn_free_frac": turn_free.float().mean().item()}
        tilted = fallen_mask_from_obs(flat, gsl, cfg.get("expert_mask_tilt_deg", 75.0)) & fallen  # expert_mask_cols rows
        fallen = fallen | body_teach
        if fallen.sum().item() < cfg["min_samples"]:
            fallen = torch.zeros_like(fallen)  # too few fallen frames: anchor-only pass (or nothing)
        use = fallen | upright
        idx = use.nonzero().flatten()
        if idx.numel() == 0:
            stats["expert_bc"] = 0.0
            return stats
        obs_td = obs_td[idx]
        fallen = fallen[idx]
        tilted = tilted[idx]
        weight = torch.where(fallen, torch.full_like(fallen, cfg["coef"], dtype=torch.float),
                             torch.full_like(fallen, cfg.get("anchor_coef", 0.0), dtype=torch.float))
        with torch.no_grad():
            target = torch.zeros(idx.numel(), self.storage.actions.shape[-1], device=self.device)
            if fallen.any():
                exp_td = obs_td[fallen].clone()
                exp_flat = expert_input(torch.cat([exp_td[g] for g in groups], dim=-1), tuple(cfg["twist_slice"]),
                                        tuple(cfg.get("expert_mask_cols", ())), tilted[fallen])
                target[fallen] = self.expert(_with_flat(exp_td, groups, exp_flat))
                if cfg.get("label_hold_clamp"):
                    # Against the student's TRUE obs (exp_flat may have its joints masked).
                    own = torch.cat([obs_td[fallen][g] for g in groups], dim=-1)
                    target[fallen], clamped = clamp_label_holds(own, target[fallen], cfg["label_hold_clamp"])
                    stats["expert_bc_label_hold_clamped"] = clamped.float().mean().item()
            if (~fallen).any():
                anc_td = obs_td[~fallen]
                if cfg.get("anchor_yaw_remap"):
                    anc_td = anc_td.clone()
                    anc_flat = remap_anchor_yaw(torch.cat([anc_td[g] for g in groups], dim=-1), tuple(cfg["twist_slice"]),
                                                cfg["anchor_yaw_remap"])
                    anc_td = _with_flat(anc_td, groups, anc_flat)
                target[~fallen] = self.anchor(anc_td)
        n = idx.numel()
        mb = max(cfg.get("min_mini_batch", 1), n // cfg["mini_batches"])
        total = 0.0
        steps = 0
        for _ in range(cfg["epochs"]):
            perm = torch.randperm(n, device=self.device)
            for s in range(0, n, mb):
                sel = perm[s:s + mb]
                pred = self.actor(obs_td[sel])
                loss = (weight[sel] * (pred - target[sel]).pow(2).mean(dim=1)).mean()
                self.bc_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.bc_optimizer.step()
                total += loss.item()
                steps += 1
        stats["expert_bc"] = total / max(steps, 1)
        return stats

    def update(self) -> dict[str, float]:
        loss_dict = super().update()
        if self.expert is not None:
            loss_dict.update(self._bc_update())
        return loss_dict

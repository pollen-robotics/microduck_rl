"""Expressive body motions (2026-10-28): smooth phase-indexed references for the Expressive task family.

Each motion is a one-shot of duration T seconds, driven like the ground pick by a phase φ = t / T ∈ [0, 1)
(command = [cos 2πφ, sin 2πφ, 0]); it starts from and ends in the normal standing pose (every reference channel is 0
at φ = 0 and back to 0 before φ = 1, leaving a short settle at the end).

Reference channels (REF_CHANNELS, offsets from the standing pose):
    roll, pitch      trunk orientation (ZYX Euler of trunk_base, rad). pitch > 0 = nose DOWN (bow), < 0 = chest out.
                     roll > 0 = right side down.
    yaw              trunk yaw relative to the heading at the start of the motion (rad), > 0 = turn LEFT.
    dz               trunk height offset (m): root z - mean foot-site z, minus its HOME value (STAND_HEIGHT).
                     HOME is near full leg extension: only ~+3 mm is reachable upward, a squat goes to ~-15 mm.
    neck_pitch       > 0 = neck back / head high      (HOME 0.349, range [-1.571, 1.047])
    head_pitch       > 0 = nod down (chin down)        (HOME 0.349, range [-1.571, 1.571])
    head_yaw         > 0 = look LEFT                   (HOME 0, range ±2.967)
    head_roll        tilt                              (HOME 0, range ±0.436)

A channel is a sum of components  amp · E(t) · W(t):
    E = cosine-eased trapezoid envelope (0 before t0, rises to 1 at t1, holds to t2, falls to 0 at t3),
    W = 1 (static offset) or sin(2π f (t - t0) + ph) (oscillation), so every reference is C1 (no jumps).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

REF_CHANNELS = ("roll", "pitch", "yaw", "dz", "neck_pitch", "head_pitch", "head_yaw", "head_roll")
CH = {n: i for i, n in enumerate(REF_CHANNELS)}
HEAD_JOINTS = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")
# root z - mean(left_foot, right_foot site z) when standing on HOME targets: FK gives 0.1171, the BAM servos sag
# ~2 mm under the weight (measured in the env, 2026-10-28), so the zero of the dz channel is the settled stand.
STAND_HEIGHT = 0.1152


@dataclass(frozen=True)
class Comp:
    ch: str
    amp: float
    env: tuple            # (t0, t1, t2, t3) seconds
    freq: float = 0.0     # Hz; 0 = static offset
    ph: float = 0.0       # oscillation phase at t0 (rad)


@dataclass(frozen=True)
class Motion:
    name: str
    period: float                       # T (s)
    comps: tuple
    note: str = ""
    settle_from: float = 0.88           # φ from which the legs are pulled back to HOME
    extra: dict = field(default_factory=dict)


def _E(t: torch.Tensor, t0, t1, t2, t3) -> torch.Tensor:
    up = 0.5 - 0.5 * torch.cos(math.pi * torch.clamp((t - t0) / max(t1 - t0, 1e-6), 0.0, 1.0))
    dn = 0.5 + 0.5 * torch.cos(math.pi * torch.clamp((t - t2) / max(t3 - t2, 1e-6), 0.0, 1.0))
    return up * dn


def reference(motion: Motion, phi: torch.Tensor) -> torch.Tensor:
    """(N,) phase -> (N, 8) reference offsets in REF_CHANNELS order."""
    t = phi * motion.period
    out = torch.zeros(phi.shape[0], len(REF_CHANNELS), device=phi.device, dtype=phi.dtype)
    for c in motion.comps:
        e = _E(t, *c.env)
        w = 1.0 if c.freq == 0.0 else torch.sin(2 * math.pi * c.freq * (t - c.env[0]) + c.ph)
        out[:, CH[c.ch]] += c.amp * e * w
    return out


def bound(motion: Motion, ch: str) -> float:
    """Upper bound of |reference| on a channel (sum of |amp|)."""
    return sum(abs(c.amp) for c in motion.comps if c.ch == ch)


# ── The motions ──────────────────────────────────────────────────────────────
PI2 = math.pi / 2

# 1. SASSY (priority): bow a little to show the tail, then wiggle it: trunk yaw ±0.22 coupled IN PHASE with a hip-level
# roll ±0.07 at 2.5 Hz (tail swings right while the right side dips -> reads as a booty wiggle from behind and 3/4),
# over a small crouch so one leg can shorten. The head looks back over the LEFT shoulder (yaw +0.6) tilted (roll +0.3)
# for the whole shake; then a sassy hold: chin up, chest up, head still tilted; back to stand.
SASSY = Motion("Sassy", 3.5, (
    # body: bow + crouch for the shake, then chest-up for the final pose
    Comp("pitch", 0.08, (0.10, 0.45, 2.30, 2.60)),
    Comp("pitch", -0.05, (2.40, 2.65, 2.95, 3.20)),
    Comp("dz", -0.006, (0.10, 0.45, 2.30, 2.60)),
    Comp("yaw", 0.22, (0.30, 0.65, 2.15, 2.50), freq=2.5),
    Comp("roll", 0.07, (0.30, 0.65, 2.15, 2.50), freq=2.5),
    # head: looking back over the shoulder, tilted, the whole time; chin up in the final hold
    Comp("head_yaw", 0.6, (0.10, 0.50, 2.90, 3.20)),
    Comp("head_roll", 0.3, (0.10, 0.50, 2.90, 3.20)),
    Comp("head_pitch", -0.25, (2.30, 2.60, 2.95, 3.20)),
    Comp("neck_pitch", 0.12, (2.30, 2.60, 2.95, 3.20)),
), note="booty wiggle 2.5 Hz (yaw+roll in phase) over a bow, head over the left shoulder; chin-up hold",
   settle_from=0.92)

# 2. PROUD / puff-up: chest out (pitch back), the little height there is, head high and chin up; hold; a small settle nod.
PROUD = Motion("Proud", 3.0, (
    Comp("pitch", -0.14, (0.15, 0.85, 2.00, 2.55)),
    Comp("pitch", 0.04, (1.95, 2.10, 2.10, 2.35)),            # settle: a small forward dip as the chest comes down
    Comp("dz", 0.002, (0.15, 0.85, 2.00, 2.55)),
    Comp("neck_pitch", 0.25, (0.15, 0.85, 2.00, 2.55)),
    Comp("head_pitch", -0.30, (0.25, 0.95, 2.00, 2.55)),
    Comp("head_pitch", 0.08, (1.95, 2.10, 2.10, 2.40)),       # settle nod
), note="chest out, head high, chin up; hold; small settle")

# 3. SHY / bashful: small crouch, hunch forward, slow body twist side to side (kid twisting, 0.8 Hz), head down and
# turned away. Head-forward shallow (head mass forward makes the real duck step).
SHY = Motion("Shy", 3.5, (
    Comp("dz", -0.008, (0.15, 0.80, 2.75, 3.20)),
    Comp("pitch", 0.07, (0.15, 0.80, 2.75, 3.20)),
    Comp("yaw", 0.15, (0.50, 0.90, 2.45, 2.95), freq=0.8),
    Comp("head_pitch", 0.30, (0.15, 0.80, 2.75, 3.20)),
    Comp("neck_pitch", -0.10, (0.15, 0.80, 2.75, 3.20)),
    Comp("head_yaw", -0.50, (0.20, 0.90, 2.75, 3.20)),
    Comp("head_roll", 0.15, (0.20, 0.90, 2.75, 3.20)),
), note="crouch, hunch, slow 0.8 Hz twist, head down and turned away", settle_from=0.92)

# 4. EXCITED bounce: small squats at 3 Hz (dz from 0 to -12 mm), feet planted, head bobbing in sync, looking up.
EXCITED = Motion("Excited", 3.0, (
    Comp("dz", -0.006, (0.15, 0.45, 2.25, 2.60)),
    Comp("dz", 0.006, (0.15, 0.45, 2.25, 2.60), freq=3.0, ph=PI2),   # -0.006 + 0.006 cos: 0 .. -12 mm
    Comp("head_pitch", -0.10, (0.15, 0.45, 2.25, 2.60)),
    Comp("head_pitch", 0.10, (0.15, 0.45, 2.25, 2.60), freq=3.0, ph=-PI2),  # nod down at the squat bottom
    Comp("neck_pitch", 0.10, (0.15, 0.45, 2.25, 2.60)),
), note="3 Hz squats 0..-12 mm, head bobbing in sync, looking up")

# 5. GIGGLE: belly laugh, a fast small trunk shake (pitch ±0.04, roll ±0.03 at 5 Hz, 90° apart: a little circle)
# while leaning back with the head tossed back and bobbing.
GIGGLE = Motion("Giggle", 2.5, (
    Comp("pitch", -0.05, (0.10, 0.40, 1.70, 2.00)),
    Comp("pitch", 0.04, (0.25, 0.45, 1.60, 1.90), freq=5.0),
    Comp("roll", 0.03, (0.25, 0.45, 1.60, 1.90), freq=5.0, ph=PI2),
    Comp("head_pitch", -0.25, (0.10, 0.40, 1.75, 2.05)),
    Comp("head_pitch", 0.06, (0.25, 0.45, 1.60, 1.90), freq=5.0),
    Comp("neck_pitch", 0.15, (0.10, 0.40, 1.75, 2.05)),
), note="5 Hz pitch/roll shake, leaning back, head tossed back")

# 6. HMPH turn-away (my pick): snap the trunk away on planted feet (yaw -0.35: the hip yaws allow ~0.43, a true
# quarter turn needs a step), nose in the air, head turned further away, a small chin flick; hold; turn back.
HMPH = Motion("Hmph", 3.5, (
    Comp("yaw", -0.35, (0.15, 0.55, 2.40, 3.00)),
    Comp("roll", -0.04, (0.15, 0.55, 2.40, 3.00)),
    Comp("pitch", -0.05, (0.15, 0.55, 2.40, 3.00)),
    Comp("head_yaw", -0.50, (0.20, 0.55, 2.40, 3.00)),
    Comp("head_pitch", -0.25, (0.20, 0.55, 2.40, 3.00)),
    Comp("head_pitch", -0.10, (0.95, 1.10, 1.10, 1.40)),       # "hmph" chin flick
    Comp("neck_pitch", 0.15, (0.20, 0.55, 2.40, 3.00)),
    Comp("head_roll", -0.15, (0.20, 0.55, 2.40, 3.00)),
), note="trunk turned away on planted feet, nose up, chin flick, back")

MOTIONS = {m.name: m for m in (SASSY, PROUD, SHY, EXCITED, GIGGLE, HMPH)}


# ── Round 2 (2026-10-28 night, notes/expressive/LEDGER.md "Round 2") ────────────────────────────────────────────────
# Round 1 eval: the oscillations barely appeared (Sassy wiggle x0.37 and out of phase, Giggle shake x0.19, Excited
# sagged instead of bouncing), Hmph parked the left hip yaw on its stop, and the stand net stepped at the hand-back.
# Round 2 references: slower / feasible oscillations, Hmph turns -18°, and EVERY envelope closes (value AND
# velocity 0, the cosine envelope has zero slope at t3) by φ ≤ R2_END_PHI, then a still stand until the hand-back.
# Registered as Mjlab-Expressive-<Name>-R2-MicroDuck (motion key "<Name>-R2"), trained from scratch.
R2_END_PHI = 0.87

SASSY_R2 = Motion("Sassy", 3.5, (
    Comp("pitch", 0.08, (0.10, 0.45, 2.15, 2.45)),
    Comp("pitch", -0.05, (2.30, 2.50, 2.75, 3.00)),
    Comp("dz", -0.006, (0.10, 0.45, 2.15, 2.45)),
    Comp("yaw", 0.22, (0.30, 0.65, 1.95, 2.30), freq=2.0),      # 2 Hz: 4 swings of the tail
    Comp("roll", 0.07, (0.30, 0.65, 1.95, 2.30), freq=2.0),     # in phase with the yaw
    Comp("head_yaw", 0.6, (0.10, 0.50, 2.70, 3.00)),
    Comp("head_roll", 0.3, (0.10, 0.50, 2.70, 3.00)),
    Comp("head_pitch", -0.25, (2.25, 2.50, 2.75, 3.00)),
    Comp("neck_pitch", 0.12, (2.25, 2.50, 2.75, 3.00)),
), note="r2: booty wiggle 2.0 Hz (yaw ±0.22 + roll ±0.07 in phase) over a bow, head over the left shoulder; chin-up hold",
   settle_from=0.88)

PROUD_R2 = Motion("Proud", 3.0, PROUD.comps, note="r2: " + PROUD.note, settle_from=0.88)   # ends at φ 0.85 already

SHY_R2 = Motion("Shy", 3.5, (
    Comp("dz", -0.008, (0.15, 0.80, 2.55, 3.00)),
    Comp("pitch", 0.07, (0.15, 0.80, 2.55, 3.00)),
    Comp("yaw", 0.15, (0.50, 0.90, 2.25, 2.75), freq=0.8),
    Comp("head_pitch", 0.30, (0.15, 0.80, 2.55, 3.00)),
    Comp("neck_pitch", -0.10, (0.15, 0.80, 2.55, 3.00)),
    Comp("head_yaw", -0.50, (0.20, 0.90, 2.55, 3.00)),
    Comp("head_roll", 0.15, (0.20, 0.90, 2.55, 3.00)),
), note="r2: crouch, hunch, slow 0.8 Hz twist, head down and turned away; ends at φ 0.86", settle_from=0.88)

EXCITED_R2 = Motion("Excited", 3.0, (
    Comp("dz", -0.007, (0.15, 0.45, 2.15, 2.55)),
    Comp("dz", 0.007, (0.15, 0.45, 2.15, 2.55), freq=2.5, ph=PI2),    # -0.007 + 0.007 cos: 0 .. -14 mm, 6 bounces
    Comp("head_pitch", -0.10, (0.15, 0.45, 2.15, 2.55)),
    Comp("head_pitch", 0.10, (0.15, 0.45, 2.15, 2.55), freq=2.5, ph=-PI2),
    Comp("neck_pitch", 0.10, (0.15, 0.45, 2.15, 2.55)),
), note="r2: 2.5 Hz squats 0..-14 mm (height-rate tracked), head bobbing in sync, looking up", settle_from=0.88)

GIGGLE_R2 = Motion("Giggle", 2.5, (
    Comp("pitch", -0.05, (0.10, 0.40, 1.70, 2.00)),
    Comp("roll", 0.08, (0.25, 0.45, 1.55, 1.85), freq=4.0),            # the shake: ±0.08 rad at 4 Hz
    Comp("head_pitch", -0.25, (0.10, 0.40, 1.75, 2.05)),
    Comp("head_pitch", 0.06, (0.25, 0.45, 1.55, 1.85), freq=4.0),
    Comp("neck_pitch", 0.15, (0.10, 0.40, 1.75, 2.05)),
), note="r2: 4 Hz roll shake ±0.08 (roll-rate tracked), leaning back, head tossed back", settle_from=0.88)

HMPH_R2 = Motion("Hmph", 3.5, (
    Comp("yaw", -math.radians(18.0), (0.15, 0.55, 2.40, 3.00)),       # -18°: off the hip-yaw stops
    Comp("roll", -0.04, (0.15, 0.55, 2.40, 3.00)),
    Comp("pitch", -0.05, (0.15, 0.55, 2.40, 3.00)),
    Comp("head_yaw", -0.50, (0.20, 0.55, 2.40, 3.00)),
    Comp("head_pitch", -0.25, (0.20, 0.55, 2.40, 3.00)),
    Comp("head_pitch", -0.10, (0.95, 1.10, 1.10, 1.40)),
    Comp("neck_pitch", 0.15, (0.20, 0.55, 2.40, 3.00)),
    Comp("head_roll", -0.15, (0.20, 0.55, 2.40, 3.00)),
), note="r2: trunk turned away -18° on planted feet, nose up, chin flick, back", settle_from=0.88)

MOTIONS_R2 = {m.name: m for m in (SASSY_R2, PROUD_R2, SHY_R2, EXCITED_R2, GIGGLE_R2, HMPH_R2)}
# Every motion by key: "Sassy" (round 1), "Sassy-R2" (round 2). The reward / obs functions look motions up here.
ALL_MOTIONS = {**MOTIONS, **{f"{k}-R2": v for k, v in MOTIONS_R2.items()}}
TRUNK_CHANNELS = ("roll", "pitch", "yaw", "dz")


def get_motion(key: str) -> Motion:
    return ALL_MOTIONS[key]


def reference_rate(motion: Motion, phi: torch.Tensor, h: float = 0.002) -> torch.Tensor:
    """(N,) phase -> (N, 8) time derivative of the reference (units / s), central difference over ±h seconds."""
    dphi = h / motion.period
    return (reference(motion, phi + dphi) - reference(motion, phi - dphi)) / (2.0 * h)


def osc_channels(motion: Motion) -> tuple:
    """Trunk channels with an oscillating component (velocity-tracked in round 2)."""
    return tuple(ch for ch in TRUNK_CHANNELS if any(c.ch == ch and c.freq > 0.0 for c in motion.comps))


def peak_rate(motion: Motion, ch: str, n: int = 4001) -> float:
    phi = torch.linspace(0.0, 1.0, n, dtype=torch.float64)
    return float(reference_rate(motion, phi)[:, CH[ch]].abs().max())

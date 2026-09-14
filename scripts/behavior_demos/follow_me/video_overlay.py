"""HUD and head-camera picture-in-picture composition for the demo video.

Rendering-only. Nothing here is read back by the behavior or the policy; if
Pillow is unavailable the demo still runs with ``--no-render`` and produces the
full metrics JSON.
"""

from __future__ import annotations

import math

from follow_motion import (
    BACKWARD_END,
    FORWARD_END,
    LEFT_TURN_END,
    READY_END,
    RIGHT_EXIT_END,
    STOP_END,
)
from PIL import Image, ImageDraw, ImageFont

PIP_W, PIP_H = 225, 165

PHASES = ["READY", "FORWARD", "LEFT TURN", "STOP", "RIGHT TURN", "BACKWARD", "DONE"]

# Phase boundaries are derived from the choreography constants rather than
# retyped, so the timeline drawn on the video can never drift from the route
# actually simulated.
PHASE_BOUNDARIES = [0.0, READY_END, FORWARD_END, LEFT_TURN_END, STOP_END,
                    RIGHT_EXIT_END, BACKWARD_END]

PHASE_COLORS = {
    "READY": (180, 210, 255),
    "FORWARD": (100, 235, 145),
    "LEFT TURN": (255, 205, 80),
    "STOP": (255, 120, 120),
    "RIGHT TURN": (120, 215, 255),
    "BACKWARD": (215, 155, 255),
    "DONE": (180, 255, 180),
    "STOPPED": (255, 120, 120),
}

# Trunk height below which the run is reported as fallen.
FALLEN_Z = 0.09

# First readable monospace font found wins; the bitmap default is the fallback
# so the HUD renders on a bare CI box with no system fonts installed.
_FONT_CANDIDATES = (
    "/System/Library/Fonts/SFNSMono.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
    "DejaVuSansMono.ttf",
)


def _font(size=14):
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def compose(main_rgb, pip_rgb, *, t, total_seconds, person, duck_pos,
            duck_yaw, follow, command, camera, yaw_rate, min_height):
    image = Image.fromarray(main_rgb)
    pip = Image.fromarray(pip_rgb)
    draw = ImageDraw.Draw(image)
    font = _font(14)
    small = _font(12)
    width, height = image.size
    phase_color = PHASE_COLORS[person.phase]

    draw.rectangle([0, 0, width, 126], fill=(4, 8, 14))
    replay = follow["replay_phase"] if person.moving else "STOPPED"
    replay_color = PHASE_COLORS.get(replay, (235, 240, 245))
    draw.text((12, 8), f"FOLLOW-ME · TRUE LEFT / RIGHT   t={t:05.2f}s",
              fill=phase_color, font=font)
    draw.text((12, 31), f"LEADER: {person.phase:<9}   DUCK REPLAYS: {replay:<9}",
              fill=replay_color, font=small)
    draw.text((12, 51),
              f"world-path lag={follow['spatial_lag']:.3f} m   "
              f"trail error={follow['error']:.3f} m   "
              f"person range={follow['person_range']:.3f} m",
              fill=(235, 240, 245), font=small)
    draw.text((12, 71),
              f"command vx={command[0]:+.3f} vy={command[1]:+.3f} wz={command[2]:+.3f}   "
              f"heading error={math.degrees(follow['yaw_error']):+5.1f} deg",
              fill=(235, 240, 245), font=small)
    stable = duck_pos[2] >= FALLEN_Z
    draw.text((12, 91),
              f"trunk z={duck_pos[2]:.3f} m   min={min_height:.3f} m   "
              f"{'UPRIGHT' if stable else 'FALLEN'}",
              fill=(145, 255, 165) if stable else (255, 90, 90), font=small)

    visible = camera["visible"]
    draw.text((12, 111),
              f"head camera: {'PERSON VISIBLE' if visible else 'TARGET LOST'}   "
              f"off-axis={math.degrees(camera['off_axis']):.1f} deg",
              fill=(120, 255, 140) if visible else (255, 110, 110), font=small)

    px0, py0 = width - PIP_W - 12, 138
    border = (100, 255, 130) if visible else (255, 90, 90)
    draw.rectangle([px0 - 4, py0 - 22, px0 + PIP_W + 4, py0 + PIP_H + 4],
                   fill=(2, 5, 8))
    draw.text((px0, py0 - 19), "DUCK VIEW · STABILIZED HEAD CAM",
              fill=border, font=small)
    image.paste(pip, (px0, py0))
    draw = ImageDraw.Draw(image)
    draw.rectangle([px0 - 1, py0 - 1, px0 + PIP_W, py0 + PIP_H],
                   outline=border, width=3)
    cx, cy = px0 + PIP_W // 2, py0 + PIP_H // 2
    draw.line([cx - 8, cy, cx + 8, cy], fill=(255, 235, 80), width=1)
    draw.line([cx, cy - 8, cx, cy + 8], fill=(255, 235, 80), width=1)

    # Timeline makes the choreography readable at a glance. Boundaries are
    # clamped to the rendered duration so a shortened run (--seconds, used for
    # smoke tests) still draws a valid, monotonically increasing bar.
    bar_y = height - 34
    x0, x1 = 12, width - 12
    draw.rectangle([x0, bar_y, x1, bar_y + 12], fill=(25, 34, 46))
    boundaries = [min(b, total_seconds) for b in (*PHASE_BOUNDARIES, total_seconds)]
    for index, phase in enumerate(PHASES):
        left = x0 + (x1 - x0) * boundaries[index] / total_seconds
        right = x0 + (x1 - x0) * boundaries[index + 1] / total_seconds
        if right <= left:
            continue
        color = PHASE_COLORS[phase]
        draw.rectangle([left, bar_y, right, bar_y + 12], fill=color)
        if right - left > 55:
            draw.text((left + 3, bar_y - 15), phase, fill=color, font=small)
    marker = x0 + (x1 - x0) * min(t / total_seconds, 1.0)
    draw.line([marker, bar_y - 4, marker, bar_y + 17], fill=(255, 255, 255), width=2)
    return image

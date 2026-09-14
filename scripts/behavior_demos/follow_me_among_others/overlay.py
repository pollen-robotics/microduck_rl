"""HUD overlay: camera PiP, active target, state pipeline and progress badges.

Imported lazily by the runner, so ``--no-render`` never requires Pillow.
"""

import math

from PIL import Image, ImageDraw, ImageFont

PIP_W, PIP_H = 260, 190

COLOR_RGB = {
    "BLUE": (55, 115, 255),
    "GREEN": (45, 220, 100),
    "RED": (255, 70, 75),
    "YELLOW": (255, 210, 55),
    "PURPLE": (190, 90, 245),
}
STATE_RGB = {
    "SEARCH": (255, 210, 75),
    "FOUND": (75, 255, 150),
    "FOLLOW": (90, 190, 255),
    "STOP": (255, 105, 110),
    "DONE": (180, 255, 180),
}
PIPELINE = ("SEARCH", "FOUND", "FOLLOW", "STOP")

# Candidate monospace faces, tried in order. Pillow's default bitmap font is a
# fine fallback: the overlay is diagnostic, so it must never be the reason a
# headless machine cannot render.
_FONT_CANDIDATES = (
    "/System/Library/Fonts/SFNSMono.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
)


def _font(size: int = 14):
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def compose(
    main_rgb,
    pip_rgb,
    *,
    t,
    total_seconds,
    state,
    state_elapsed,
    selection,
    target,
    sequence,
    duck_pos,
    follow,
    command,
    camera,
    min_height,
    completed_cycles,
):
    image = Image.fromarray(main_rgb)
    pip = Image.fromarray(pip_rgb)
    draw = ImageDraw.Draw(image)
    font, small, tiny = _font(14), _font(12), _font(11)
    width, height = image.size
    accent = COLOR_RGB[target]
    state_color = STATE_RGB[state]
    total = len(sequence)

    draw.rectangle([0, 0, width, 132], fill=(4, 8, 14))
    draw.text(
        (12, 8),
        f"FOLLOW-ME - AMONG OTHERS   t={t:05.2f}s",
        fill=accent,
        font=font,
    )
    draw.text(
        (12, 31),
        f"SELECTION {selection}/{total}   TARGET: {target:<5}   STATE: {state}",
        fill=state_color,
        font=small,
    )
    visible = ", ".join(camera["visible_colors"]) or "NONE"
    draw.text(
        (12, 52),
        f"camera sees: {visible:<25} target off-axis="
        f"{math.degrees(camera['target_off_axis']):5.1f} deg",
        fill=(230, 238, 248),
        font=small,
    )
    draw.text(
        (12, 73),
        f"queued-footprint error={follow['error']:.3f} m   "
        f"target range={camera['target_distance']:.3f} m   "
        f"state time={state_elapsed:.2f} s",
        fill=(230, 238, 248),
        font=small,
    )
    draw.text(
        (12, 94),
        f"command vx={command[0]:+.3f} vy={command[1]:+.3f} "
        f"wz={command[2]:+.3f}   yaw error="
        f"{math.degrees(follow['yaw_error']):+5.1f} deg",
        fill=(230, 238, 248),
        font=small,
    )
    stable = duck_pos[2] >= 0.09
    draw.text(
        (12, 115),
        f"trunk z={duck_pos[2]:.3f} m   min={min_height:.3f} m   "
        f"{'UPRIGHT' if stable else 'FALLEN'}",
        fill=(145, 255, 165) if stable else (255, 90, 90),
        font=small,
    )

    px0, py0 = width - PIP_W - 12, 143
    border = accent if camera["target_visible"] else (255, 190, 60)
    draw.rectangle(
        [px0 - 4, py0 - 24, px0 + PIP_W + 4, py0 + PIP_H + 4], fill=(2, 5, 8)
    )
    label = f"SEARCHING {target}" if state == "SEARCH" else f"TRACKING {target}"
    draw.text((px0, py0 - 21), f"DUCK CAMERA - {label}", fill=border, font=small)
    image.paste(pip, (px0, py0))
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        [px0 - 1, py0 - 1, px0 + PIP_W, py0 + PIP_H], outline=border, width=3
    )
    cx, cy = px0 + PIP_W // 2, py0 + PIP_H // 2
    draw.line([cx - 9, cy, cx + 9, cy], fill=(255, 245, 95), width=1)
    draw.line([cx, cy - 9, cx, cy + 9], fill=(255, 245, 95), width=1)
    if camera["target_visible"]:
        status = "VISIBLE" if state == "SEARCH" else "LOCKED"
        draw.text((px0 + 8, py0 + 8), f"{target} {status}", fill=accent, font=small)

    # Make the repeated control pattern explicit for a viewer who never reads
    # the metrics file.
    pipe_x, pipe_y = 16, height - 82
    for idx, name in enumerate(PIPELINE):
        x = pipe_x + idx * 132
        active = name == state
        fill = STATE_RGB[name] if active else (34, 44, 58)
        draw.rounded_rectangle(
            [x, pipe_y, x + 108, pipe_y + 30],
            radius=7,
            fill=fill,
            outline=STATE_RGB[name],
            width=2,
        )
        draw.text(
            (x + 9, pipe_y + 7),
            name,
            fill=(5, 10, 16) if active else STATE_RGB[name],
            font=small,
        )
        if idx < len(PIPELINE) - 1:
            draw.text((x + 112, pipe_y + 7), ">", fill=(170, 185, 205), font=font)

    # Target badges persist so a repeated color is unmistakable.
    badge_y = height - 36
    slot = max(1, (width - 32) // max(total, 1))
    for idx, color in enumerate(sequence):
        x = 16 + idx * slot
        finished = idx < completed_cycles
        active = idx == selection - 1 and state != "DONE"
        outline = COLOR_RGB[color]
        fill = outline if active else ((44, 54, 68) if not finished else (24, 90, 55))
        draw.rounded_rectangle(
            [x, badge_y, x + min(124, slot - 8), badge_y + 23],
            radius=6,
            fill=fill,
            outline=outline,
            width=2,
        )
        prefix = "OK " if finished else ""
        draw.text(
            (x + 8, badge_y + 4),
            f"{prefix}{idx + 1} {color}",
            fill=(5, 10, 16) if active else outline,
            font=tiny,
        )

    progress_x = 16 + int((width - 32) * min(t / total_seconds, 1.0))
    draw.line(
        [progress_x, height - 7, progress_x, height - 2],
        fill=(255, 255, 255),
        width=2,
    )
    return image

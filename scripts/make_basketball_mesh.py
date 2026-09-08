#!/usr/bin/env python3
"""Generate the visual UV-sphere mesh for the BallWalk basketball.

  uv run python scripts/make_basketball_mesh.py

MuJoCo's built-in sphere geom does NOT use an equirectangular UV mapping, so
the 8-panel seam texture (assets/basketball.png, from Vottivott/microduck-
playground `scripts/make_basketball_assets.py`) smears when applied directly to
the collision sphere. This writes a plain lat-long sphere OBJ whose texcoords
follow that generator's convention (u mirrored, v flipped, constant texcoord on
the pole fans), used as a visual-only geom in robot/microduck/basketball.xml.
No pebble/seam displacement (their 15 MB mesh) — a smooth sphere is enough.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "src" / "mjlab_microduck" / "robot" / "microduck" / "assets" / "basketball.obj"


def make_mesh(radius: float, segments: int, rings: int) -> str:
    us = np.arange(segments + 1) / segments
    vs = np.arange(rings + 1) / rings
    U, V = np.meshgrid(us, vs)  # (rings+1, segments+1)
    lat = (V - 0.5) * math.pi
    lon = (U - 0.5) * 2 * math.pi
    n = np.stack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)), axis=-1).reshape(-1, 3)
    verts = n * radius
    lines = [f"# basketball visual sphere r={radius} ({segments}x{rings}), equirect UVs"]
    lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in verts]
    lines += [f"vt {1.0 - u:.6f} {1.0 - v:.6f}" for u, v in zip(U.ravel(), V.ravel())]
    lines += [f"vn {x:.6f} {y:.6f} {z:.6f}" for x, y, z in n]
    n_vt = len(U.ravel())
    lines.append("vt 0.500000 1.000000")  # south cap
    lines.append("vt 0.500000 0.000000")  # north cap
    vt_s, vt_n = n_vt + 1, n_vt + 2
    for i in range(rings):
        for j in range(segments):
            a = i * (segments + 1) + j + 1
            b = a + 1
            c = a + segments + 1
            d = c + 1
            if i > 0:
                t = vt_n if i == rings - 1 else None
                lines.append(f"f {a}/{t or a}/{a} {b}/{t or b}/{b} {d}/{t or d}/{d}")
            if i < rings - 1:
                t = vt_s if i == 0 else None
                lines.append(f"f {a}/{t or a}/{a} {d}/{t or d}/{d} {c}/{t or c}/{c}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=float, default=0.1205, help="visual radius; +0.5 mm over the 0.12 collision sphere hides the polygon gap under the feet")
    ap.add_argument("--segments", type=int, default=96)
    ap.add_argument("--rings", type=int, default=48)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    args.out.write_text(make_mesh(args.radius, args.segments, args.rings))
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()

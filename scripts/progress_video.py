"""Tile checkpoints from one training run into a single progress-grid mp4.

Each tile is the same task rolled out from the same seed with a different
checkpoint, labeled with its iteration, all playing in sync — so "did training
actually improve the behavior" is answerable by watching one video.

Usage:
  # auto: discover model_*.pt in a run dir, pick 4 evenly spaced (first..last)
  uv run scripts/progress_video.py --run-dir logs/rsl_rl/ground_pick/<run> \
      --out /tmp/progress.mp4

  # explicit: hand-pick checkpoints (order = reading order, row-major)
  uv run scripts/progress_video.py \
      --checkpoint model_500.pt --checkpoint model_2000.pt \
      --checkpoint model_8000.pt --checkpoint model_19999.pt \
      --out /tmp/progress.mp4

Requires a GPU (MuJoCo Warp) and ffmpeg on PATH (or imageio-ffmpeg's bundled
binary, which uv resolves automatically).
"""

import argparse
import math
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path

# Headless GPU rendering needs EGL, and it must be selected before mujoco is
# imported (via mjlab below) or rendering dies with "an OpenGL platform
# library has not been loaded". Skip when a real X display is available.
if "DISPLAY" not in os.environ:
  os.environ.setdefault("MUJOCO_GL", "egl")
  os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
import torch
from PIL import Image, ImageDraw

import mjlab.tasks  # noqa: F401
import mjlab_microduck.tasks  # noqa: F401

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from mjlab.viewer.viewer_config import ViewerConfig

DEFAULT_TASK = "Mjlab-GroundPick-Flat-MicroDuck"
VIEWS = {"front": 270.0, "45deg": 315.0, "side": 0.0}
VIEW_W, VIEW_H = 640, 480
# Keeps tile height (480 + 32) a multiple of 16 so h264 needs no padding.
LABEL_H = 32


def find_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
  """All `model_<iter>.pt` under run_dir, sorted by iteration."""
  out = []
  for p in run_dir.glob("model_*.pt"):
    m = re.fullmatch(r"model_(\d+)\.pt", p.name)
    if m:
      out.append((int(m.group(1)), p))
  return sorted(out)


def pick_evenly(ckpts: list[tuple[int, Path]], k: int) -> list[tuple[int, Path]]:
  """Up to k checkpoints, evenly spaced, always including first and last."""
  if len(ckpts) <= k:
    return ckpts
  idx = sorted({round(i * (len(ckpts) - 1) / (k - 1)) for i in range(k)})
  return [ckpts[i] for i in idx]


def label_bar(text: str, width: int) -> np.ndarray:
  img = Image.new("RGB", (width, LABEL_H), (15, 20, 27))
  ImageDraw.Draw(img).text((10, 9), text, fill=(255, 206, 61))
  return np.asarray(img)


def stack_tiles(
  tile_paths: list[Path], out: Path, cols: int, rows: int, w: int, h: int,
  fps: int, seconds: float,
) -> None:
  """ffmpeg-xstack the per-checkpoint tiles into one grid video."""
  if len(tile_paths) == 1:
    shutil.copy(tile_paths[0], out)
    return
  ffmpeg = shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()
  cmd = [ffmpeg, "-y"]
  for t in tile_paths:
    cmd += ["-i", str(t)]
  n_inputs = len(tile_paths)
  # fill any empty grid cells with black so xstack sees cols*rows inputs
  for _ in range(cols * rows - n_inputs):
    cmd += ["-f", "lavfi", "-i", f"color=c=black:s={w}x{h}:r={fps}:d={seconds}"]
    n_inputs += 1
  layout = "|".join(
    f"{(i % cols) * w}_{(i // cols) * h}" for i in range(cols * rows)
  )
  fc = "".join(f"[{i}:v]" for i in range(n_inputs))
  fc += f"xstack=inputs={n_inputs}:layout={layout}[v]"
  cmd += [
    "-filter_complex", fc, "-map", "[v]",
    "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", str(out),
  ]
  subprocess.run(cmd, check=True, capture_output=True)


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  src = p.add_mutually_exclusive_group(required=True)
  src.add_argument("--run-dir", type=Path,
                   help="directory containing model_<iter>.pt checkpoints")
  src.add_argument("--checkpoint", action="append", type=Path,
                   help="explicit checkpoint; repeat for each tile")
  p.add_argument("--task", default=DEFAULT_TASK)
  p.add_argument("--out", type=Path, required=True)
  p.add_argument("--tiles", type=int, default=4,
                 help="max tiles in auto mode (grid is ceil(sqrt)-wide)")
  p.add_argument("--seconds", type=float, default=6.0, help="per tile")
  p.add_argument("--view", choices=list(VIEWS), default="45deg")
  p.add_argument("--three-views", action="store_true",
                 help="front/45deg/side strip per tile (3x wider video)")
  p.add_argument("--distance", type=float, default=0.8)
  p.add_argument("--elevation", type=float, default=-10.0)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--seed", type=int, default=0,
                 help="same seed for every tile so luck is shared")
  args = p.parse_args()

  if args.run_dir:
    ckpts = pick_evenly(find_checkpoints(args.run_dir), args.tiles)
    if not ckpts:
      raise SystemExit(f"no model_<iter>.pt checkpoints in {args.run_dir}")
  else:
    ckpts = []
    for c in args.checkpoint:
      m = re.search(r"model_(\d+)\.pt$", c.name)
      ckpts.append((int(m.group(1)) if m else -1, c))

  azimuths = list(VIEWS.values()) if args.three_views else [VIEWS[args.view]]
  tile_w, tile_h = VIEW_W * len(azimuths), VIEW_H + LABEL_H
  print(f"[INFO] {len(ckpts)} tiles: "
        + ", ".join(f"iter {i}" for i, _ in ckpts))

  configure_torch_backends()
  torch.manual_seed(args.seed)

  env_cfg = load_env_cfg(args.task, play=True)
  agent_cfg = load_rl_cfg(args.task)
  env_cfg.scene.num_envs = 1
  # Pin the command cycle to phase 0 at every reset so all tiles start aligned
  # (matches runtime behavior; only tasks with a phase-randomized twist have it).
  try:
    twist = env_cfg.commands["twist"]
  except (KeyError, TypeError):
    twist = None
  if twist is not None and hasattr(twist, "randomize_phase"):
    twist.randomize_phase = False
  env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)

  # Single renderer; azimuth is swapped per view each frame (multiple
  # OffscreenRenderers in one process can trip over EGL contexts).
  cam_cfg = ViewerConfig(
    origin_type=ViewerConfig.OriginType.ASSET_BODY,
    entity_name="robot",
    body_name="trunk_base",
    env_idx=0,
    azimuth=azimuths[0],
    elevation=args.elevation,
    distance=args.distance,
    height=VIEW_H,
    width=VIEW_W,
    enable_reflections=False,
  )
  renderer = OffscreenRenderer(model=env.sim.mj_model, cfg=cam_cfg, scene=env.scene)
  renderer.initialize()

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=args.device)

  fps = round(1.0 / env.unwrapped.step_dt)
  n_steps = int(args.seconds * fps)

  with tempfile.TemporaryDirectory() as tmp:
    tile_paths = []
    for iteration, ckpt in ckpts:
      print(f"[INFO] tile iter {iteration}: {ckpt}")
      runner.load(
        str(ckpt), load_cfg={"actor": True}, strict=True,
        map_location=args.device,
      )
      policy = runner.get_inference_policy(device=args.device)

      tile_path = Path(tmp) / f"tile_{iteration}.mp4"
      writer = imageio.get_writer(
        tile_path, fps=fps, codec="libx264", quality=8, macro_block_size=16
      )
      bar = label_bar(
        f"iter {iteration}" if iteration >= 0 else ckpt.name, tile_w
      )
      torch.manual_seed(args.seed)  # same episode luck for every tile
      env.reset()
      for i in range(n_steps):
        obs = env.get_observations()
        with torch.no_grad():
          actions = policy(obs)
        env.step(actions)
        frames = []
        for azimuth in azimuths:
          renderer._cam.azimuth = azimuth
          renderer.update(env.unwrapped.sim.data)
          frames.append(renderer.render())
        frame = np.concatenate(frames, axis=1)
        writer.append_data(np.concatenate([bar, frame], axis=0))
        if (i + 1) % fps == 0:
          print(f"  {(i + 1) // fps}/{args.seconds:.0f}s", flush=True)
      writer.close()
      tile_paths.append(tile_path)

    cols = math.ceil(math.sqrt(len(tile_paths)))
    rows = math.ceil(len(tile_paths) / cols)
    print(f"[INFO] stacking {len(tile_paths)} tiles ({cols}x{rows}) -> {args.out}")
    stack_tiles(
      tile_paths, args.out, cols, rows, tile_w, tile_h, fps, args.seconds
    )

  env.close()
  print(f"Wrote {args.out}")


if __name__ == "__main__":
  main()

"""Bake one checkpoint rollout into the interactive rig HTML page.

Wraps dump_rig_signals.dump_signals() + rig_template.html into a single
self-contained file: animated side-view rig, per-joint pos/vel/action traces,
live 61D actor obs vector, and the XL330→policy→servo signal-path map.
Open the output in any browser (no server needed) — use it to debug *why* a
policy misbehaves, not whether.

Usage:
  uv run scripts/make_rig_page.py --checkpoint logs/.../model_12250.pt \
      --out /tmp/rig.html --seconds 12
"""

import argparse
import json
import re
from pathlib import Path

from dump_rig_signals import DEFAULT_TASK, dump_signals


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--task", default=DEFAULT_TASK)
  p.add_argument("--checkpoint", required=True)
  p.add_argument("--out", required=True)
  p.add_argument("--seconds", type=float, default=12.0)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--seed", type=int, default=0)
  args = p.parse_args()

  data = dump_signals(
    args.task, args.checkpoint, args.seconds, args.device, seed=args.seed
  )

  ckpt = Path(args.checkpoint)
  m = re.search(r"model_(\d+)\.pt$", ckpt.name)
  label = f"iter {m.group(1)}" if m else ckpt.name
  sub = (
    f"{args.task} <b>{ckpt.parent.name}</b> {label} "
    f"· {data['fps']} Hz · {len(data['joint_names'])}× Dynamixel XL330"
  )

  template = Path(__file__).with_name("rig_template.html").read_text()
  payload = json.dumps(data).replace("</", "<\\/")  # keep </script> out of JSON
  html = template.replace("/*__DATA__*/", payload).replace("__SUB__", sub)
  Path(args.out).write_text(html)
  print(f"Wrote {args.out} — open in a browser")


if __name__ == "__main__":
  main()

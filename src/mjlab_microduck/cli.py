"""`microduck <command>` — the training side's tools under one name.

    microduck train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096
    microduck play  Mjlab-Velocity-Flat-MicroDuck --wandb-run-path <...>
    microduck list-envs
    microduck export  Mjlab-Velocity-Flat-MicroDuck --checkpoint-file model_3000.pt
    microduck publish --run logs/rsl_rl/sprint/<run> --repo <user>/microduck-sprint
    microduck check   --onnx output.onnx
    microduck infer   --walking output.onnx

Each one IS the entry point it names — `uv run train`, `uv run publish`… keep working and
behave identically. The only thing done here is rewriting argv[0] to the command's name before
the target is imported: mjlab's import runs our plugin hooks (`--hf-jobs`, provenance.json),
and both decide on `Path(sys.argv[0]).name == "train"`.
"""

from __future__ import annotations

import importlib
import sys

COMMANDS: dict[str, tuple[str, str]] = {
    "train": ("mjlab.scripts.train", "main"),
    "play": ("mjlab.scripts.play", "main"),
    "list-envs": ("mjlab.scripts.list_envs", "main"),
    "export": ("mjlab_microduck.export", "main"),
    "publish": ("mjlab_microduck.publish.cli", "main"),
    "check": ("mjlab_microduck.publish.check", "main"),
    "infer": ("mjlab_microduck.infer", "main"),
}


def _usage() -> str:
    lines = ["usage: microduck <command> [args...]", "", "commands:"]
    lines += [f"  {name}" for name in COMMANDS]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print(_usage())
        return 0
    name, rest = args[0], args[1:]
    if name not in COMMANDS:
        print(f"microduck: unknown command {name!r}\n{_usage()}", file=sys.stderr)
        return 2
    # Before any mjlab import: the hooks read argv[0] when mjlab loads its plugins.
    sys.argv = [name, *rest]
    module, attr = COMMANDS[name]
    code = getattr(importlib.import_module(module), attr)()
    return int(code or 0)


if __name__ == "__main__":
    sys.exit(main())

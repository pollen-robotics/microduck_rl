"""`train` entry point: mjlab's trainer, plus `--hf-jobs` remote submission.

This project's [project.scripts] `train` shadows mjlab's so the everyday
command grows one flag:

    uv run train Mjlab-Kick-Flat-MicroDuck --env.scene.num-envs 4096 \
        --agent.max_iterations 4000              # local, exactly as before
    uv run train Mjlab-Kick-Flat-MicroDuck --env.scene.num-envs 4096 \
        --agent.max_iterations 4000 --hf-jobs    # same run, on HF Jobs

Without --hf-jobs, argv goes to mjlab.scripts.train after device resolution
(see _resolve_device: --device cpu/cuda translation and CPU fallback on
CUDA-less machines). With --hf-jobs, the submission flags (--flavor,
--namespace, --detach, ... see hf_jobs.py) are consumed here and everything
else is forwarded to `uv run train` inside the job.
"""

from __future__ import annotations

import sys


def _resolve_device(argv: list[str]) -> list[str]:
    """Map --device {cuda,cpu} onto mjlab's --gpu-ids, defaulting to CPU when
    no CUDA GPU exists.

    mjlab's train already runs the whole stack (Warp sim + torch) on CPU via
    `--gpu-ids None`, but its default `--gpu-ids 0` crashes with an IndexError
    in select_gpus() on CUDA-less machines (macOS, CPU-only Linux). This shim
    translates before argv reaches mjlab; on a CUDA machine with no flags it
    changes nothing.
    """
    if "--device" in argv:
        i = argv.index("--device")
        device = argv[i + 1] if i + 1 < len(argv) else "<missing>"
        argv = argv[:i] + argv[i + 2 :]
        if device == "cpu":
            return [*argv, "--gpu-ids", "None"]
        if device == "cuda":
            return argv
        sys.exit(
            f"--device {device} is not supported: mjlab passes a single device "
            "string to both Warp (sim) and torch, and Warp has no MPS/other "
            "backend. Use --device cpu or --device cuda."
        )
    if "--gpu-ids" not in argv:
        import torch

        if not torch.cuda.is_available():
            print(
                "[WARN] No CUDA GPU detected - training on CPU "
                "(slow; fine for smoke tests, use --hf-jobs for real runs)."
            )
            return [*argv, "--gpu-ids", "None"]
    return argv


def main() -> int | None:
    argv = sys.argv[1:]
    if "--hf-jobs" in argv:
        from mjlab_microduck.hf_jobs import submit

        return submit([a for a in argv if a != "--hf-jobs"])

    sys.argv[1:] = _resolve_device(argv)

    from mjlab.scripts.train import main as mjlab_train_main

    return mjlab_train_main()


if __name__ == "__main__":
    sys.exit(main())

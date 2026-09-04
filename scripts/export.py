"""Export a trained checkpoint to ONNX — thin wrapper over `mjlab_microduck.export`.

    uv run scripts/export.py <TASK_ID> --wandb-run-path <entity/project/run> [--checkpoint 3000]

The logic lives in the package so `uv run publish` shares it; see that module's docstring.
"""

from mjlab_microduck.export import main

if __name__ == "__main__":
    main()

"""Export the frozen sentence encoder to ONNX — thin wrapper.

    uv run scripts/export_encoder.py --out encoder.onnx

The logic lives in the package (``mjlab_microduck.vla.export_encoder``) so the
publish/export paths share it. Requires ``uv sync --extra vla``.
"""

from mjlab_microduck.vla.export_encoder import main

if __name__ == "__main__":
    raise SystemExit(main())

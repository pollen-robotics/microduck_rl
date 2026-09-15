# Jetson: `import mjlab` pre-loads torch's undeclared libraries (tasks hook);
# tests that import torch first would miss it. No-op elsewhere.
import mjlab  # noqa: F401

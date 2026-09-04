# Jetson: the SBSA torch wheel's undeclared libraries are pre-loaded by the
# mjlab.tasks plugin hook, i.e. inside `import mjlab`. Tests that import torch
# first would miss it, so import mjlab once before collection. No-op elsewhere.
import mjlab  # noqa: F401

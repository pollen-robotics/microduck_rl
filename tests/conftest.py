# Jetson (Thor): an editable checkout has no interpreter-start hook (uv_build's
# editable wheel carries no data files, so mjlab_microduck_jetson.pth only
# lands on a non-editable install), and the pre-load of the SBSA torch wheel's
# undeclared NVPL/cuDSS libraries runs on the first import of this package.
# Tests that `import torch` before anything imports mjlab_microduck would die
# with `libnvpl_lapack_lp64_gomp.so.0: cannot open shared object file`, so
# import it here, once, before collection. A no-op off Jetson.
import mjlab_microduck  # noqa: F401

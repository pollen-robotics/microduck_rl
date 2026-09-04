# Jetson (Thor / JetPack 7): pre-load the libraries the SBSA torch wheel links
# but does not declare, BEFORE anything below imports torch. No-op elsewhere.
from mjlab_microduck._jetson import preload_jetson_libs as _preload_jetson_libs

_preload_jetson_libs()

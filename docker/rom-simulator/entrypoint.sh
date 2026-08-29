#!/bin/sh
set -eu

bundle_dir=${MICRODUCK_ROM_BUNDLE_DIR:-/bundle}
state_db=${MICRODUCK_ROM_STATE_DB:-/state/tasks.sqlite3}

if [ ! -r "$bundle_dir/microduck-policy-bundle.json" ]; then
    echo "container startup failed: /bundle must contain a readable manifest" >&2
    exit 64
fi
if [ -z "${MICRODUCK_ROM_BEARER_TOKEN:-}" ]; then
    echo "container startup failed: MICRODUCK_ROM_BEARER_TOKEN is required" >&2
    exit 64
fi
state_dir=${state_db%/*}
if [ ! -d "$state_dir" ] || [ ! -w "$state_dir" ]; then
    echo "container startup failed: /state must be a writable directory" >&2
    exit 64
fi

exec python -m mjlab_microduck.rom.main "$@"

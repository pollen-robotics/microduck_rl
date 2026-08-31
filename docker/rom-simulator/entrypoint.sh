#!/bin/sh
set -eu

bundle_dir=${MICRODUCK_ROM_BUNDLE_DIR:-/bundle}
state_db=${MICRODUCK_ROM_STATE_DB:-/state/tasks.sqlite3}
secret_path=/run/secrets/microduck_rom_bearer_token

if [ "${MICRODUCK_ROM_BEARER_TOKEN+x}" = x ]; then
    echo "container startup failed: direct bearer environment input is forbidden" >&2
    exit 64
fi
if [ "${MICRODUCK_ROM_BEARER_TOKEN_FILE:-}" != "$secret_path" ]; then
    echo "container startup failed: fixed bearer secret file is required" >&2
    exit 64
fi
if [ ! -f "$secret_path" ] || [ -L "$secret_path" ] || [ ! -r "$secret_path" ]; then
    echo "container startup failed: bearer secret file is invalid" >&2
    exit 64
fi
if [ ! -r "$bundle_dir/microduck-policy-bundle.json" ]; then
    echo "container startup failed: /bundle must contain a readable manifest" >&2
    exit 64
fi
state_dir=${state_db%/*}
if [ ! -d "$state_dir" ] || [ ! -w "$state_dir" ]; then
    echo "container startup failed: /state must be a writable directory" >&2
    exit 64
fi

exec python -P /usr/local/libexec/microduck-rom-pid1.py "$@"

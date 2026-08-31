FROM ghcr.io/astral-sh/uv:0.8.22@sha256:9874eb7afe5ca16c363fe80b294fe700e460df29a55532bbfea234a0f12eddb1 AS uv

FROM python:3.12.11-slim-bookworm@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7 AS builder

ENV UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    UV_HTTP_TIMEOUT=600

COPY --from=uv /uv /uvx /bin/
WORKDIR /app
COPY pyproject.toml uv.lock LICENSE ./
RUN uv sync --frozen --only-group rom --no-install-project

FROM python:3.12.11-slim-bookworm@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/app/.venv/bin:/usr/local/bin:/usr/bin:/bin \
    XDG_CACHE_HOME=/state/cache \
    MICRODUCK_ROM_BUNDLE_DIR=/bundle \
    MICRODUCK_ROM_STATE_DB=/state/tasks.sqlite3 \
    MICRODUCK_ROM_BEARER_TOKEN_FILE=/run/secrets/microduck_rom_bearer_token \
    MICRODUCK_ROM_HOST=0.0.0.0 \
    MICRODUCK_ROM_ALLOW_WILDCARD_BIND=true \
    MICRODUCK_ROM_PORT=8000

RUN mkdir -p /bundle /state \
    && install -d -o 0 -g 0 -m 0755 /usr/local/libexec \
    && chown 10001:10001 /state

COPY --from=builder /app/.venv /app/.venv
COPY docker/rom-simulator/mjlab_microduck_rom.pth \
    /app/.venv/lib/python3.12/site-packages/mjlab_microduck_rom.pth
COPY --from=builder /app/pyproject.toml /app/pyproject.toml
COPY --from=builder /app/LICENSE /app/LICENSE
COPY src/mjlab_microduck/__init__.py /app/src/mjlab_microduck/__init__.py
COPY src/mjlab_microduck/rom/__init__.py \
    src/mjlab_microduck/rom/action_catalog.py \
    src/mjlab_microduck/rom/action_specs.py \
    src/mjlab_microduck/rom/api.py \
    src/mjlab_microduck/rom/bundle.py \
    src/mjlab_microduck/rom/contracts.py \
    src/mjlab_microduck/rom/main.py \
    src/mjlab_microduck/rom/mirroring.py \
    src/mjlab_microduck/rom/model_semantics.py \
    src/mjlab_microduck/rom/mujoco_runtime.py \
    src/mjlab_microduck/rom/observation.py \
    src/mjlab_microduck/rom/onnx_policy.py \
    src/mjlab_microduck/rom/parent_death.py \
    src/mjlab_microduck/rom/process_protocol.py \
    src/mjlab_microduck/rom/process_service.py \
    src/mjlab_microduck/rom/process_supervisor.py \
    src/mjlab_microduck/rom/qualification.py \
    src/mjlab_microduck/rom/runtime.py \
    src/mjlab_microduck/rom/runtime_child.py \
    src/mjlab_microduck/rom/runtime_identity.py \
    src/mjlab_microduck/rom/secret_file.py \
    src/mjlab_microduck/rom/service.py \
    src/mjlab_microduck/rom/store.py \
    src/mjlab_microduck/rom/supervisor_state.py \
    /app/src/mjlab_microduck/rom/
COPY schemas/microduck-policy-bundle-v1.schema.json \
    schemas/microduck-simulator-api-v1.openapi.yaml \
    schemas/microduck-v1-portability-fixtures.json \
    /app/schemas/
COPY --chmod=0755 docker/rom-simulator/entrypoint.sh /usr/local/bin/rom-entrypoint
COPY --chmod=0644 docker/rom-simulator/pid1_bootstrap.py \
    /usr/local/libexec/microduck-rom-pid1.py

STOPSIGNAL SIGTERM
USER 10001:10001
WORKDIR /state
EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/rom-entrypoint"]

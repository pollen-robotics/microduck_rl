FROM ghcr.io/astral-sh/uv:0.8.22@sha256:9874eb7afe5ca16c363fe80b294fe700e460df29a55532bbfea234a0f12eddb1 AS uv

FROM python:3.12.11-slim-bookworm@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7 AS builder

ENV UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    UV_HTTP_TIMEOUT=600

COPY --from=uv /uv /uvx /bin/
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --only-group rom --no-install-project

FROM python:3.12.11-slim-bookworm@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/app/.venv/bin:/usr/local/bin:/usr/bin:/bin \
    PYTHONPATH=/app/src \
    XDG_CACHE_HOME=/state/cache \
    MICRODUCK_ROM_BUNDLE_DIR=/bundle \
    MICRODUCK_ROM_STATE_DB=/state/tasks.sqlite3 \
    MICRODUCK_ROM_HOST=0.0.0.0 \
    MICRODUCK_ROM_ALLOW_WILDCARD_BIND=true \
    MICRODUCK_ROM_PORT=8000

RUN mkdir -p /bundle /state \
    && chown 10001:10001 /state

COPY --from=builder /app/.venv /app/.venv
COPY src/mjlab_microduck/__init__.py /app/src/mjlab_microduck/__init__.py
COPY src/mjlab_microduck/rom /app/src/mjlab_microduck/rom
COPY --chmod=0755 docker/rom-simulator/entrypoint.sh /usr/local/bin/rom-entrypoint

USER 10001:10001
WORKDIR /state
EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/rom-entrypoint"]

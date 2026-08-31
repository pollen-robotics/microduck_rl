"""Fail-closed loading for the ROM bearer-token secret file."""

from __future__ import annotations

import os
import stat
import unicodedata
from pathlib import Path

MAX_SECRET_BYTES = 4096
_UNAVAILABLE_MESSAGE = "bearer token file is unavailable"


def read_secret_file(path_value: str) -> str:
    """Read one bounded token from an owner-only regular file."""
    path = Path(path_value)
    if not path.is_absolute():
        raise ValueError("bearer token file path must be absolute")

    try:
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError:
        raise ValueError(_UNAVAILABLE_MESSAGE) from None

    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("bearer token source must be a regular file")
        if metadata.st_mode & 0o077:
            raise ValueError("bearer token file must be owner-only")
        raw = os.read(fd, MAX_SECRET_BYTES + 1)
    except OSError:
        raise ValueError(_UNAVAILABLE_MESSAGE) from None
    finally:
        try:
            os.close(fd)
        except OSError:
            raise ValueError(_UNAVAILABLE_MESSAGE) from None

    if len(raw) > MAX_SECRET_BYTES:
        raise ValueError("bearer token file exceeds 4096 bytes")
    if raw.endswith(b"\n"):
        raw = raw[:-1]
    try:
        token = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise ValueError("bearer token file must contain valid UTF-8") from None
    if not token:
        raise ValueError("bearer token must not be empty")
    if any(unicodedata.category(character).startswith("C") for character in token):
        raise ValueError("bearer token must not contain control characters")
    return token

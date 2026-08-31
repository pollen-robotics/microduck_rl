"""Fail-closed loading for the ROM bearer-token secret file."""

from __future__ import annotations

import os
import re
import stat
import unicodedata
from pathlib import Path

MAX_SECRET_BYTES = 4096
PRODUCTION_SECRET_PATH = "/run/secrets/microduck_rom_bearer_token"
_UNAVAILABLE_MESSAGE = "bearer token file is unavailable"
_MOUNTINFO_PATH = Path("/proc/self/mountinfo")
_MOUNT_ESCAPE = re.compile(r"\\(040|011|012|134)")
_MOUNT_ESCAPE_VALUES = {"040": " ", "011": "\t", "012": "\n", "134": "\\"}


def _decode_mount_field(value: str) -> str:
    return _MOUNT_ESCAPE.sub(
        lambda matched: _MOUNT_ESCAPE_VALUES[matched.group(1)], value
    )


def _is_read_only_bind_mount(
    path: Path,
    *,
    device: int,
    mountinfo_path: Path,
) -> bool:
    try:
        mount_lines = mountinfo_path.read_text().splitlines()
    except (OSError, UnicodeError):
        return False
    expected_device = f"{os.major(device)}:{os.minor(device)}"
    for line in mount_lines:
        mount_fields = line.partition(" - ")[0].split()
        if len(mount_fields) < 6:
            continue
        root = _decode_mount_field(mount_fields[3])
        mountpoint = _decode_mount_field(mount_fields[4])
        options = mount_fields[5].split(",")
        if (
            mount_fields[2] == expected_device
            and root != "/"
            and mountpoint == str(path)
        ):
            return "ro" in options
    return False


def read_secret_file(
    path_value: str,
    *,
    require_read_only_mount: bool = False,
    mountinfo_path: Path = _MOUNTINFO_PATH,
) -> str:
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
        if metadata.st_uid != os.geteuid() or (
            require_read_only_mount and metadata.st_gid != os.getegid()
        ):
            raise ValueError(
                "bearer token file ownership must match process identity"
            )
        if metadata.st_mode & 0o077:
            raise ValueError("bearer token file must be owner-only")
        if require_read_only_mount and not _is_read_only_bind_mount(
            path,
            device=metadata.st_dev,
            mountinfo_path=mountinfo_path,
        ):
            raise ValueError("bearer token file must be a read-only bind mount")
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

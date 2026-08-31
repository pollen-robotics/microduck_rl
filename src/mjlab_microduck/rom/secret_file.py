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
_FDINFO_ROOT = Path("/proc/self/fdinfo")
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
    mount_id: int,
    mountinfo_path: Path,
) -> bool:
    try:
        mount_lines = mountinfo_path.read_text().splitlines()
    except (OSError, UnicodeError):
        return False
    matching_fields: list[list[str]] = []
    for line in mount_lines:
        mount_record, separator, _filesystem = line.partition(" - ")
        mount_fields = mount_record.split()
        if separator and mount_fields and mount_fields[0] == str(mount_id):
            matching_fields.append(mount_fields)
    if len(matching_fields) != 1 or len(matching_fields[0]) < 6:
        return False
    mount_fields = matching_fields[0]
    expected_device = f"{os.major(device)}:{os.minor(device)}"
    root = _decode_mount_field(mount_fields[3])
    mountpoint = _decode_mount_field(mount_fields[4])
    options = mount_fields[5].split(",")
    return (
        mount_fields[2] == expected_device
        and root.startswith("/")
        and root != "/"
        and mountpoint == str(path)
        and "ro" in options
    )


def _descriptor_mount_id(fd: int, fdinfo_path: Path | None) -> int | None:
    source = fdinfo_path if fdinfo_path is not None else _FDINFO_ROOT / str(fd)
    try:
        lines = source.read_text().splitlines()
    except (OSError, UnicodeError):
        return None
    values = []
    for line in lines:
        key, separator, raw_value = line.partition(":")
        value = raw_value.strip()
        if key == "mnt_id" and separator and value.isdecimal():
            values.append(int(value))
    return values[0] if len(values) == 1 and values[0] > 0 else None


def read_secret_file(
    path_value: str,
    *,
    require_read_only_mount: bool = False,
    mountinfo_path: Path = _MOUNTINFO_PATH,
    fdinfo_path: Path | None = None,
) -> str:
    """Read one bounded token from an owner-only regular file."""
    path = Path(path_value)
    if not path.is_absolute():
        raise ValueError("bearer token file path must be absolute")

    try:
        fd = os.open(
            path,
            os.O_RDONLY
            | os.O_CLOEXEC
            | os.O_NOFOLLOW
            | getattr(os, "O_NONBLOCK", 0),
        )
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
        if require_read_only_mount:
            mount_id = _descriptor_mount_id(fd, fdinfo_path)
            if mount_id is None or not _is_read_only_bind_mount(
                path,
                device=metadata.st_dev,
                mount_id=mount_id,
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

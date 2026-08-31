"""Security-boundary tests for the ROM bearer-token file reader."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from mjlab_microduck.rom.secret_file import MAX_SECRET_BYTES, read_secret_file


def _write_owner_only(path: Path, content: bytes) -> Path:
    path.write_bytes(content)
    path.chmod(0o400)
    return path


@pytest.mark.parametrize("trailing_lf", [b"", b"\n"])
def test_owner_only_regular_file_loads_one_line_token(
    tmp_path: Path, trailing_lf: bytes
) -> None:
    secret = _write_owner_only(tmp_path / "bearer", b"one-line-token" + trailing_lf)

    assert read_secret_file(str(secret)) == "one-line-token"


def test_relative_secret_path_is_rejected_without_filesystem_access() -> None:
    with pytest.raises(
        ValueError, match=r"^bearer token file path must be absolute$"
    ):
        read_secret_file("relative/bearer")


@pytest.mark.parametrize("source_kind", ["missing", "symlink"])
def test_unopenable_secret_source_has_one_non_leaking_error(
    tmp_path: Path, source_kind: str
) -> None:
    target = tmp_path / "token-value-must-not-leak"
    source = tmp_path / "configured-path-must-not-leak"
    if source_kind == "symlink":
        _write_owner_only(target, b"token-value-must-not-leak")
        source.symlink_to(target)

    with pytest.raises(ValueError) as caught:
        read_secret_file(str(source))

    assert str(caught.value) == "bearer token file is unavailable"
    assert str(source) not in str(caught.value)
    assert "token-value-must-not-leak" not in str(caught.value)


def test_directory_secret_source_is_rejected_as_non_regular(tmp_path: Path) -> None:
    source = tmp_path / "secret-directory"
    source.mkdir()

    with pytest.raises(
        ValueError, match=r"^bearer token source must be a regular file$"
    ):
        read_secret_file(str(source))


def test_fifo_secret_source_is_rejected_before_any_blocking_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "secret-fifo"
    os.mkfifo(source, 0o400)
    read_fd, write_fd = os.pipe()

    monkeypatch.setattr(os, "open", lambda _path, _flags: read_fd)

    def fail_if_read(_fd: int, _amount: int) -> bytes:
        pytest.fail("non-regular source reached os.read")

    monkeypatch.setattr(os, "read", fail_if_read)
    try:
        with pytest.raises(
            ValueError, match=r"^bearer token source must be a regular file$"
        ):
            read_secret_file(str(source))
    finally:
        os.close(write_fd)


@pytest.mark.parametrize("mode", [0o440, 0o404])
def test_group_or_other_file_permissions_are_rejected(
    tmp_path: Path, mode: int
) -> None:
    source = tmp_path / f"bearer-{mode:o}"
    source.write_bytes(b"secret")
    source.chmod(mode)

    with pytest.raises(
        ValueError, match=r"^bearer token file must be owner-only$"
    ):
        read_secret_file(str(source))


@pytest.mark.parametrize("identity", ("uid", "gid"))
def test_descriptor_owner_must_match_running_process_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    identity: str,
) -> None:
    """Trusting path/mode alone would accept a credential owned by another identity."""
    source = _write_owner_only(tmp_path / "bearer", b"secret")
    real_fstat = os.fstat

    def wrong_owner(fd: int) -> SimpleNamespace:
        metadata = real_fstat(fd)
        return SimpleNamespace(
            st_mode=metadata.st_mode,
            st_size=metadata.st_size,
            st_dev=metadata.st_dev,
            st_uid=metadata.st_uid + (1 if identity == "uid" else 0),
            st_gid=metadata.st_gid + (1 if identity == "gid" else 0),
        )

    monkeypatch.setattr(os, "fstat", wrong_owner)

    with pytest.raises(
        ValueError,
        match=r"^bearer token file ownership must match process identity$",
    ):
        read_secret_file(str(source), require_read_only_mount=True)


def test_noncontainer_owner_only_file_allows_inaccessible_different_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-container compatibility depends on owner UID, not its unused group."""
    source = _write_owner_only(tmp_path / "bearer", b"secret")
    real_fstat = os.fstat

    def different_group(fd: int) -> SimpleNamespace:
        metadata = real_fstat(fd)
        return SimpleNamespace(
            st_mode=metadata.st_mode,
            st_size=metadata.st_size,
            st_dev=metadata.st_dev,
            st_uid=metadata.st_uid,
            st_gid=metadata.st_gid + 1,
        )

    monkeypatch.setattr(os, "fstat", different_group)

    assert read_secret_file(str(source)) == "secret"


@pytest.mark.parametrize("mount_options", ("rw,nosuid", "ro,nosuid"))
def test_required_read_only_mount_uses_exact_decoded_mountinfo_entry(
    tmp_path: Path,
    mount_options: str,
) -> None:
    """A writable or merely ancestor-mounted secret must not satisfy production."""
    source = _write_owner_only(tmp_path / "bearer secret", b"secret")
    metadata = source.stat()
    encoded_mountpoint = str(source).replace(" ", r"\040")
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text(
        f"91 42 {os.major(metadata.st_dev)}:{os.minor(metadata.st_dev)} "
        f"/host/secret {encoded_mountpoint} {mount_options} - ext4 /dev/root ro\n"
    )

    if mount_options.startswith("ro"):
        assert (
            read_secret_file(
                str(source),
                require_read_only_mount=True,
                mountinfo_path=mountinfo,
            )
            == "secret"
        )
    else:
        with pytest.raises(
            ValueError,
            match=r"^bearer token file must be a read-only bind mount$",
        ):
            read_secret_file(
                str(source),
                require_read_only_mount=True,
                mountinfo_path=mountinfo,
            )


def test_required_read_only_mount_rejects_read_only_ancestor(
    tmp_path: Path,
) -> None:
    """A read-only ancestor is not proof that the secret file is its own bind mount."""
    source = _write_owner_only(tmp_path / "bearer", b"secret")
    metadata = source.stat()
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text(
        f"91 42 {os.major(metadata.st_dev)}:{os.minor(metadata.st_dev)} "
        f"/host/secret {tmp_path} ro,nosuid - ext4 /dev/root ro\n"
    )

    with pytest.raises(
        ValueError,
        match=r"^bearer token file must be a read-only bind mount$",
    ):
        read_secret_file(
            str(source),
            require_read_only_mount=True,
            mountinfo_path=mountinfo,
        )


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"", "bearer token must not be empty"),
        (b"x" * (MAX_SECRET_BYTES + 1), "bearer token file exceeds 4096 bytes"),
        (b"invalid-utf8-\xff", "bearer token file must contain valid UTF-8"),
        (b"embedded-\x00-nul", "bearer token must not contain control characters"),
        (b"embedded-\t-tab", "bearer token must not contain control characters"),
        (
            "embedded-\u200b-format".encode(),
            "bearer token must not contain control characters",
        ),
        (b"two-newlines\n\n", "bearer token must not contain control characters"),
    ],
)
def test_malformed_secret_contents_are_rejected_without_disclosure(
    tmp_path: Path, content: bytes, message: str
) -> None:
    source = _write_owner_only(tmp_path / "configured-path-must-not-leak", content)

    with pytest.raises(ValueError) as caught:
        read_secret_file(str(source))

    assert str(caught.value) == message
    assert str(source) not in str(caught.value)
    decoded_content = content.decode("utf-8", errors="ignore")
    if decoded_content:
        assert decoded_content not in str(caught.value)


def test_size_limit_uses_bounded_read_when_fstat_size_lies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_owner_only(tmp_path / "bearer", b"x" * (MAX_SECRET_BYTES + 1))
    real_fstat = os.fstat

    def lying_fstat(fd: int) -> SimpleNamespace:
        metadata = real_fstat(fd)
        return SimpleNamespace(
            st_mode=metadata.st_mode,
            st_size=0,
            st_dev=metadata.st_dev,
            st_uid=metadata.st_uid,
            st_gid=metadata.st_gid,
        )

    monkeypatch.setattr(os, "fstat", lying_fstat)

    with pytest.raises(
        ValueError, match=r"^bearer token file exceeds 4096 bytes$"
    ):
        read_secret_file(str(source))


def test_open_flags_and_single_read_are_hardened_and_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_owner_only(tmp_path / "bearer", b"secret")
    real_open = os.open
    real_read = os.read
    opened_flags: list[int] = []
    read_amounts: list[int] = []

    def recording_open(path: Path, flags: int) -> int:
        opened_flags.append(flags)
        return real_open(path, flags)

    def recording_read(fd: int, amount: int) -> bytes:
        read_amounts.append(amount)
        return real_read(fd, amount)

    monkeypatch.setattr(os, "open", recording_open)
    monkeypatch.setattr(os, "read", recording_read)

    assert read_secret_file(str(source)) == "secret"
    assert opened_flags == [os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW]
    assert read_amounts == [MAX_SECRET_BYTES + 1]


@pytest.mark.parametrize("operation", ["fstat", "read"])
def test_descriptor_errors_are_converted_without_os_or_path_detail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    source = _write_owner_only(tmp_path / "configured-path-must-not-leak", b"secret")

    def fail(*_args: object) -> object:
        raise OSError("os-detail-and-secret-must-not-leak")

    monkeypatch.setattr(os, operation, fail)

    with pytest.raises(ValueError) as caught:
        read_secret_file(str(source))

    assert str(caught.value) == "bearer token file is unavailable"
    assert str(source) not in str(caught.value)
    assert "os-detail-and-secret-must-not-leak" not in str(caught.value)

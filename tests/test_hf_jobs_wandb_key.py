"""The wandb key lookup must find the file `wandb login` actually wrote.

`wandb login` follows the platform's netrc convention: `~/.netrc` on POSIX,
`~/_netrc` on Windows (it even prints the path it chose). Looking only at
`.netrc` therefore reported

    [wandb] x no API key found (checked $WANDB_API_KEY and ~/.netrc).

immediately after a successful `wandb login` on Windows, and the submission
refused to forward a key that was sitting right there (2026-09-02).
"""

import pytest

from mjlab_microduck.hf_jobs import _wandb_api_key

_KEY = "b" * 40
_NETRC = f"machine api.wandb.ai\n  login user\n  password {_KEY}\n"


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Point `Path.home()` at a scratch dir on both platforms."""
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))  # posix
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # windows
    return tmp_path


@pytest.mark.parametrize("name", [".netrc", "_netrc"])
def test_key_is_found_under_either_netrc_spelling(home, name):
    (home / name).write_text(_NETRC)
    assert _wandb_api_key() == _KEY


def test_env_var_still_wins(home, monkeypatch):
    (home / ".netrc").write_text(_NETRC)
    monkeypatch.setenv("WANDB_API_KEY", "from-env")
    assert _wandb_api_key() == "from-env"


def test_missing_netrc_is_not_an_error(home):
    """No key is a supported state — submission just warns and suggests --no-wandb."""
    assert _wandb_api_key() is None


def test_unrelated_machine_is_ignored(home):
    home.joinpath(".netrc").write_text("machine example.com\n  login u\n  password p\n")
    assert _wandb_api_key() is None

"""The credential guard is the last thing standing between a build-time patch
and a public repository, so it gets the same scrutiny as shipped code."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_GUARD = Path(__file__).resolve().parents[1] / "scripts" / "check_no_secrets.py"

spec = importlib.util.spec_from_file_location("check_no_secrets", _GUARD)
guard = importlib.util.module_from_spec(spec)
sys.modules["check_no_secrets"] = guard
spec.loader.exec_module(guard)

# Assembled at runtime: a literal here would trip the guard on this very file.
REAL_SECRET = "GOCSPX-" + "aB3dEfGh1jKlMn0pQrStUvWxYz12"
REAL_CLIENT_ID = "123456789012-" + "a1b2c3d4e5f6g7h8i9j0" + ".apps.googleusercontent.com"


def test_a_real_client_secret_is_caught():
    found = guard._scan("some/file.py", f'SECRET = "{REAL_SECRET}"')
    assert found, "a real client secret was not detected"


def test_a_real_client_id_is_caught():
    found = guard._scan("some/file.py", f'CLIENT = "{REAL_CLIENT_ID}"')
    assert found, "a real client id was not detected"


def test_a_secret_is_caught_anywhere_not_only_in_auth_py():
    """A credential pasted into a script or a note is just as public."""
    assert guard._scan("scripts/build.ps1", REAL_SECRET)
    assert guard._scan("notes.txt", REAL_SECRET)


@pytest.mark.parametrize("fixture", [
    'monkeypatch.setattr(auth, "DEFAULT_CLIENT_SECRET", "GOCSPX-secret")',
    'monkeypatch.setenv(auth.CLIENT_SECRET_ENV_VAR, "GOCSPX-override")',
    'monkeypatch.setattr(auth, "DEFAULT_CLIENT_ID", "1-abc.apps.googleusercontent.com")',
    'form = {"client_id": client_id()}',
])
def test_test_fixtures_are_not_flagged(fixture):
    """A guard that cries wolf gets bypassed, which is worse than no guard."""
    assert guard._scan("tests/test_google_drive.py", fixture) == []


def test_a_baked_constant_is_caught_even_in_an_unexpected_shape():
    """The length-based regexes cannot see a short or oddly formatted value."""
    source = 'DEFAULT_CLIENT_SECRET = "shhh"\n'
    assert guard._scan("src/monitor/gdrive/auth.py", source)


def test_empty_constants_are_accepted():
    source = 'DEFAULT_CLIENT_ID = ""\nDEFAULT_CLIENT_SECRET = ""\n'
    assert guard._scan("src/monitor/gdrive/auth.py", source) == []


def test_the_constant_check_only_applies_to_auth_py():
    source = 'DEFAULT_CLIENT_SECRET = "shhh"\n'
    assert guard._scan("tests/test_something.py", source) == []


def test_the_real_auth_module_passes():
    """The committed source must satisfy the guard at all times."""
    path = "src/monitor/gdrive/auth.py"
    text = (Path(__file__).resolve().parents[1] / path).read_text(encoding="utf-8")
    assert guard._scan(path, text) == []


def test_the_whole_checkout_passes():
    """Equivalent to the CI sweep, so a leak fails the suite too."""
    assert guard.main.__module__ == "check_no_secrets"
    sys.argv = ["check_no_secrets.py", "--all"]
    assert guard.main() == 0

"""
Refresh-token storage.

Backed by the Windows Credential Manager through ``keyring`` (DPAPI-encrypted,
bound to the Windows user *and* machine).

There is deliberately **no plaintext fallback**: if the OS keystore is not
available the Drive feature disables itself. A silent downgrade to a file on
disk would be worse than not having the feature, and the user would have no way
to tell the difference.

The interface is intentionally narrow so it can be swapped for a DPoP /
TPM-bound implementation later without touching call sites.
"""

from __future__ import annotations

import logging
from typing import Optional

log = logging.getLogger(__name__)

#: Credential Manager "service" name. Never change without a migration.
SERVICE_NAME = "child-monitor-analyzer:google-oauth"

#: Google's documented maximum refresh-token length.
MAX_TOKEN_BYTES = 512


class CredentialStoreUnavailable(RuntimeError):
    """The OS keystore could not be used; the feature must stay disabled."""


def _keyring():
    """Import keyring and verify a real backend resolved.

    Raises:
        CredentialStoreUnavailable: If keyring is missing or resolved to a
            backend that does not actually persist anything.
    """
    try:
        import keyring
        from keyring.backends import fail as _fail_backend
    except Exception as exc:  # noqa: BLE001 - any import problem disables the feature
        raise CredentialStoreUnavailable(
            f"keyring is not available: {exc}"
        ) from exc

    try:
        backend = keyring.get_keyring()
    except Exception as exc:  # noqa: BLE001
        raise CredentialStoreUnavailable(
            f"keyring backend could not be resolved: {exc}"
        ) from exc

    if isinstance(backend, _fail_backend.Keyring):
        raise CredentialStoreUnavailable(
            "keyring resolved to the null backend — no OS keystore is usable."
        )
    return keyring


def backend_name() -> str:
    """Return the resolved keyring backend name, for the startup self-test."""
    keyring = _keyring()
    backend = keyring.get_keyring()
    return type(backend).__name__


def is_available() -> bool:
    """Return True if a usable OS keystore is present."""
    try:
        name = backend_name()
    except CredentialStoreUnavailable as exc:
        log.warning("Credential store unavailable: %s", exc)
        return False
    log.info("Credential store backend: %s", name)
    return True


def save_refresh_token(account_id: str, refresh_token: str) -> None:
    """Persist *refresh_token* for *account_id*.

    Args:
        account_id: Google ``sub`` claim — a stable, opaque account identifier.
        refresh_token: The token to store.

    Raises:
        CredentialStoreUnavailable: If the keystore is unusable.
        ValueError: If the inputs are empty or implausibly large.
    """
    if not account_id or not refresh_token:
        raise ValueError("account_id and refresh_token are required")
    if len(refresh_token.encode("utf-8")) > MAX_TOKEN_BYTES:
        raise ValueError("refresh token is larger than Google's documented limit")

    keyring = _keyring()
    try:
        keyring.set_password(SERVICE_NAME, account_id, refresh_token)
    except Exception as exc:  # noqa: BLE001
        raise CredentialStoreUnavailable(f"could not write credential: {exc}") from exc
    # Never log the value, and never log account_id at INFO in full.
    log.info("Stored Google refresh token for account %s.", _mask(account_id))


def load_refresh_token(account_id: str) -> Optional[str]:
    """Return the stored refresh token for *account_id*, or None."""
    if not account_id:
        return None
    keyring = _keyring()
    try:
        return keyring.get_password(SERVICE_NAME, account_id)
    except Exception as exc:  # noqa: BLE001
        raise CredentialStoreUnavailable(f"could not read credential: {exc}") from exc


def delete_refresh_token(account_id: str) -> bool:
    """Delete the stored token for *account_id*.

    Returns:
        True if something was deleted, False if there was nothing to delete.
    """
    if not account_id:
        return False
    keyring = _keyring()
    try:
        keyring.delete_password(SERVICE_NAME, account_id)
    except Exception as exc:  # noqa: BLE001 - keyring raises PasswordDeleteError
        log.info("No stored credential to delete for %s (%s).", _mask(account_id), exc)
        return False
    log.info("Deleted Google refresh token for account %s.", _mask(account_id))
    return True


def _mask(account_id: str) -> str:
    """Return a short, non-reversible-enough form of *account_id* for logs."""
    return f"{account_id[:6]}..." if len(account_id) > 6 else "<short>"

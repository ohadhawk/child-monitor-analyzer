"""
Google Drive transcript upload — public façade.

Everything in this package is headless (no PySide6 import) so the flow can be
unit-tested without a display and the GUI layer owns all threading.

Nothing secret is stored here: the refresh token lives in the OS keystore
(:mod:`monitor.gdrive.store`), the access token lives in memory only, and the
non-secret state below (account id, email, folder ids) is a plain JSON file.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

from . import auth, client, store
from .auth import AuthCancelled, AuthError, GoogleCredentials, InvalidGrant
from .client import DriveError, DriveNotFound
from .naming import build_transcript_base_name, sanitize_display_name

log = logging.getLogger(__name__)

__all__ = [
    "AuthCancelled", "AuthError", "DriveError", "DriveNotFound", "InvalidGrant",
    "GoogleDriveSession", "UploadResult", "build_transcript_base_name",
    "sanitize_display_name", "is_configured", "is_available",
]

#: Non-secret state. Deliberately not QSettings: that is the Windows Registry,
#: and keeping this package free of PySide6 keeps it unit-testable.
STATE_PATH = Path.home() / ".child-monitor-analyzer" / "gdrive.json"

DRIVE_FOLDER_NAME = "Transcriptions"

_STATE_VERSION = 1


def is_configured() -> bool:
    """Return True if this build has an OAuth client id."""
    return auth.is_configured()


def is_available() -> bool:
    """Return True if the feature can run at all (client id + OS keystore)."""
    return auth.is_configured() and store.is_available()


class UploadResult:
    """Outcome of a successful upload."""

    def __init__(self, file_id: str, name: str, web_view_link: Optional[str],
                 updated: bool) -> None:
        self.file_id = file_id
        self.name = name
        self.web_view_link = web_view_link
        self.updated = updated

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (
            f"UploadResult(file_id={self.file_id!r}, name={self.name!r}, "
            f"updated={self.updated})"
        )


# ===========================
# NON-SECRET STATE
# ===========================

def _load_state(path: Path = STATE_PATH) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        log.warning("Could not read the Drive state file (%s); starting fresh.", exc)
        return {}
    return data if isinstance(data, dict) else {}


def _save_state(state: dict, path: Path = STATE_PATH) -> None:
    """Write *state* atomically so a crash cannot leave a truncated file."""
    state["version"] = _STATE_VERSION
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(state, fh, ensure_ascii=False, indent=2)
            os.replace(tmp_name, path)
        except BaseException:
            _unlink_quietly(tmp_name)
            raise
    except OSError as exc:
        log.warning("Could not save the Drive state file: %s", exc)


def _unlink_quietly(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


# ===========================
# SIDECAR (idempotency)
# ===========================

def sidecar_path(artifact_dir: Path, model_key: str) -> Path:
    """Return the per-transcript upload record path."""
    safe_key = "".join(ch for ch in (model_key or "none") if ch.isalnum() or ch == "_")
    return Path(artifact_dir) / f"gdrive_{safe_key or 'none'}.json"


def read_sidecar(artifact_dir: Path, model_key: str) -> dict:
    """Return the previous upload record, or ``{}``. Contains no secrets."""
    try:
        with open(sidecar_path(artifact_dir, model_key), "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (FileNotFoundError, OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def write_sidecar(artifact_dir: Path, model_key: str, record: dict) -> None:
    """Persist the upload record. Failure here must never fail the upload."""
    try:
        path = sidecar_path(artifact_dir, model_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(record, fh, ensure_ascii=False, indent=2)
    except OSError as exc:
        log.warning("Could not write the Drive upload record: %s", exc)


# ===========================
# SESSION
# ===========================

class GoogleDriveSession:
    """Holds the linked account and performs uploads.

    One instance per application session, owned by the GUI. Every method here
    is blocking and must be called from a worker thread.
    """

    def __init__(self, state_path: Path = STATE_PATH) -> None:
        self._state_path = Path(state_path)
        self._state = _load_state(self._state_path)
        self._credentials: Optional[GoogleCredentials] = None
        self._lock = threading.RLock()
        self._linking = False
        self._linked_cache: Optional[bool] = None

    # --- account state -----------------------------------------------

    @property
    def account_id(self) -> str:
        return str(self._state.get("account_id") or "")

    @property
    def email(self) -> str:
        return str(self._state.get("email") or "")

    def is_linked(self) -> bool:
        """Return True if an account is linked and its token is retrievable.

        The answer is cached: the UI asks on every repaint, and each miss pulls
        the refresh token out of the OS vault and into this process's memory.
        The cache is invalidated wherever the credential can change.
        """
        if not self.account_id:
            return False
        if self._linked_cache is not None:
            return self._linked_cache
        try:
            linked = bool(store.load_refresh_token(self.account_id))
        except store.CredentialStoreUnavailable as exc:
            log.warning("Cannot read the stored Google credential: %s", exc)
            return False
        self._linked_cache = linked
        return linked

    def ask_every_time(self) -> bool:
        return bool(self._state.get("ask_every_time", True))

    def set_ask_every_time(self, value: bool) -> None:
        self._state["ask_every_time"] = bool(value)
        _save_state(self._state, self._state_path)

    # --- linking ------------------------------------------------------

    def link(self, *, open_browser=None, cancel_event=None) -> GoogleCredentials:
        """Run the interactive OAuth flow and persist the refresh token.

        Raises:
            AuthError / AuthCancelled: On failure. Nothing is persisted on any
                failure path — the session stays exactly as it was.
        """
        if not store.is_available():
            raise AuthError(
                "Windows Credential Manager is not available, so the Google "
                "sign-in cannot be stored securely."
            )
        with self._lock:
            if self._linking:
                raise AuthError("A Google sign-in is already in progress.")
            self._linking = True
        try:
            log.info("Starting the Google account link flow.")
            credentials = auth.start_link_flow(
                open_browser=open_browser, cancel_event=cancel_event,
            )
            if not credentials.refresh_token:
                raise AuthError(
                    "Google did not return a refresh token; try removing the app "
                    "at myaccount.google.com/permissions and signing in again."
                )
            store.save_refresh_token(credentials.account_id, credentials.refresh_token)
            with self._lock:
                self._credentials = credentials
                self._linked_cache = True
                previous_account = self.account_id
                self._state["account_id"] = credentials.account_id
                self._state["email"] = credentials.email
                if previous_account and previous_account != credentials.account_id:
                    # Folder ids are keyed by account, so the previous
                    # account's entry can never be targeted by the new one.
                    log.info("A different Google account is now linked.")
                self._state.setdefault("folders", {})
                _save_state(self._state, self._state_path)
            log.info("Google account linked (%s).", _mask_email(credentials.email))
            return credentials
        finally:
            with self._lock:
                self._linking = False

    def unlink(self) -> bool:
        """Revoke server-side, then purge everything local.

        Revocation comes first: deleting the local copy while leaving a live
        grant on Google's side would tell the user they revoked access when
        they did not.

        Returns:
            True if Google confirmed the revocation.
        """
        account_id = self.account_id
        refresh_token = ""
        if account_id:
            try:
                refresh_token = store.load_refresh_token(account_id) or ""
            except store.CredentialStoreUnavailable as exc:
                log.warning("Could not read the credential for revocation: %s", exc)

        revoked = auth.revoke(refresh_token) if refresh_token else True

        if account_id:
            try:
                store.delete_refresh_token(account_id)
            except store.CredentialStoreUnavailable as exc:
                log.warning("Could not delete the stored credential: %s", exc)

        with self._lock:
            self._credentials = None
            self._linked_cache = False
            folders = self._state.get("folders")
            if isinstance(folders, dict):
                folders.pop(account_id, None)
            self._state.pop("account_id", None)
            self._state.pop("email", None)
            _save_state(self._state, self._state_path)
        log.info("Google account unlinked (server-side revocation: %s).", revoked)
        return revoked

    # --- tokens -------------------------------------------------------

    def _access_token(self) -> str:
        """Return a valid access token, refreshing if needed.

        Raises:
            InvalidGrant: The stored grant is dead; the caller must re-link.
        """
        with self._lock:
            credentials = self._credentials
            if credentials is not None and credentials.access_token_valid():
                return credentials.access_token

            account_id = self.account_id
            if not account_id:
                raise InvalidGrant("No Google account is linked.")
            refresh_token = credentials.refresh_token if credentials else ""
            if not refresh_token:
                refresh_token = store.load_refresh_token(account_id) or ""
            if not refresh_token:
                raise InvalidGrant("No stored Google credential was found.")

            base = GoogleCredentials(account_id=account_id, email=self.email,
                                     refresh_token=refresh_token)
            try:
                refreshed = auth.refresh_access_token(base)
            except InvalidGrant:
                log.warning("The stored Google grant was rejected; purging it.")
                self._purge_locked()
                raise
            self._credentials = refreshed
            return refreshed.access_token

    def _purge_locked(self) -> None:
        """Drop the dead credential. Caller must hold ``self._lock``."""
        account_id = self.account_id
        self._credentials = None
        self._linked_cache = False
        if account_id:
            try:
                store.delete_refresh_token(account_id)
            except store.CredentialStoreUnavailable:
                log.debug("Credential purge skipped: keystore unavailable.")
        self._state.pop("account_id", None)
        self._state.pop("email", None)
        _save_state(self._state, self._state_path)

    # --- folder -------------------------------------------------------

    def _folder_id(self, drive: client.DriveClient) -> str:
        """Return the id of the app's Transcriptions folder, creating it if needed.

        ``drive.file`` cannot search the user's Drive for a folder it did not
        create, so a lost or trashed id means creating a fresh folder. That is
        an accepted, deliberate cost of the least-privilege scope.
        """
        account_id = self.account_id
        with self._lock:
            folders = self._state.setdefault("folders", {})
            cached = folders.get(account_id)
        if cached and drive.folder_exists(cached):
            return cached
        if cached:
            log.info("The cached Drive folder is gone; creating a new one.")
        folder_id = drive.create_folder(DRIVE_FOLDER_NAME)
        with self._lock:
            self._state.setdefault("folders", {})[account_id] = folder_id
            _save_state(self._state, self._state_path)
        return folder_id

    def folder_link(self) -> Optional[str]:
        """Return a browser link to the Transcriptions folder, if known."""
        folder_id = (self._state.get("folders") or {}).get(self.account_id)
        if not folder_id:
            return None
        return client.validate_web_view_link(
            f"https://drive.google.com/drive/folders/{folder_id}"
        )

    # --- upload -------------------------------------------------------

    def upload_transcript(
        self,
        *,
        content: bytes,
        content_type: str,
        display_name: str,
        artifact_dir: Optional[Path] = None,
        model_key: str = "none",
        replace_existing: bool = False,
    ) -> UploadResult:
        """Upload a transcript as a native Google Doc.

        Args:
            content: Encoded ``.txt`` or ``.docx`` bytes.
            content_type: :data:`client.TXT_MIME` or :data:`client.DOCX_MIME`.
            display_name: Final document name; sanitised again here because it
                is derived from an untrusted audio filename.
            artifact_dir: Where the idempotency sidecar lives. None disables it.
            model_key: Distinguishes the fast/thorough transcript sidecars.
            replace_existing: Update the previously-uploaded document instead
                of creating a second one.

        Raises:
            InvalidGrant, AuthError, DriveError.
        """
        name = sanitize_display_name(display_name)
        drive = client.DriveClient(self._access_token)
        folder_id = self._folder_id(drive)

        previous = read_sidecar(artifact_dir, model_key) if artifact_dir else {}
        previous_id = previous.get("file_id")
        same_account = previous.get("account_id") == self.account_id

        payload = None
        updated = False
        if replace_existing and previous_id and same_account:
            log.info("Updating the existing Drive document %s.", previous_id)
            try:
                payload = drive.update_document(
                    str(previous_id), name=name, content=content,
                    content_type=content_type,
                )
                updated = True
            except DriveNotFound:
                log.info("The previous document is gone; creating a new one.")
                payload = None

        if payload is None:
            payload = drive.create_document(
                name=name, content=content, content_type=content_type,
                parent=folder_id,
            )

        file_id = str(payload.get("id") or "")
        link = client.validate_web_view_link(payload.get("webViewLink"))
        if payload.get("webViewLink") and not link:
            log.warning("Discarded an untrusted webViewLink from the Drive response.")

        if artifact_dir:
            write_sidecar(artifact_dir, model_key, {
                "account_id": self.account_id,
                "file_id": file_id,
                "name": name,
                "uploaded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            })

        log.info("Transcript uploaded to Drive (id %s, updated=%s).", file_id, updated)
        return UploadResult(file_id, name, link, updated)


def _mask_email(email: str) -> str:
    """Return a partially-masked email for logs."""
    if "@" not in email:
        return "<unknown>"
    local, _, domain = email.partition("@")
    return f"{local[:2]}***@{domain}"

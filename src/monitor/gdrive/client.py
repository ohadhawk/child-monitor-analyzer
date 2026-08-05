"""
Minimal Google Drive v3 client built on :mod:`urllib.request`.

Deliberately not using ``google-api-python-client``: the whole surface we need
is four calls, and every dependency added to a package that handles OAuth
credentials widens the supply-chain attack surface. This also matches the
existing convention in :mod:`monitor.model_cache` and
:mod:`monitor.model_updates`, which already avoid ``requests``.

Headless — no PySide6 import. All calls are blocking and must be run from a
worker thread.
"""

from __future__ import annotations

import json
import logging
import random
import secrets
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable, Optional

log = logging.getLogger(__name__)

DRIVE_FILES_URL = "https://www.googleapis.com/drive/v3/files"
DRIVE_UPLOAD_URL = "https://www.googleapis.com/upload/drive/v3/files"

FOLDER_MIME = "application/vnd.google-apps.folder"
GOOGLE_DOC_MIME = "application/vnd.google-apps.document"
DOCX_MIME = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
)
TXT_MIME = "text/plain"

#: Drive metadata responses are small.
MAX_RESPONSE_BYTES = 256 * 1024

#: Transcripts are kilobytes; this guards against a runaway caller, and keeps
#: ``uploadType=multipart`` (rather than resumable) the correct choice.
MAX_UPLOAD_BYTES = 8 * 1024 * 1024

HTTP_TIMEOUT_SECONDS = 60

_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
MAX_ATTEMPTS = 4
BACKOFF_BASE_SECONDS = 0.5
MAX_TOTAL_BACKOFF_SECONDS = 30.0


class DriveError(RuntimeError):
    """A Drive API call failed. The message is safe to show to the user."""

    def __init__(self, message: str, *, status: Optional[int] = None) -> None:
        super().__init__(message)
        self.status = status


class DriveNotFound(DriveError):
    """The referenced file or folder no longer exists (or was never ours)."""


class DriveOffline(DriveError):
    """Google could not be reached at all.

    Distinguished from an API error so the GUI can say "no internet" instead of
    showing a transport exception the user cannot act on.
    """


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Drive should not redirect us; following one could leak the Bearer token."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise DriveError(f"Unexpected redirect from {req.full_url} to {newurl}")


def _opener() -> urllib.request.OpenerDirector:
    context = ssl.create_default_context()
    return urllib.request.build_opener(
        urllib.request.HTTPSHandler(context=context), _NoRedirectHandler(),
    )


def build_multipart_body(metadata: dict, content: bytes,
                         content_type: str) -> tuple[bytes, str]:
    """Assemble a ``multipart/related`` body.

    The boundary is random per request. With a fixed boundary, a transcript
    that happened to contain it could terminate the body part early and inject
    its own metadata part — a real injection vector, since the transcript is
    derived from third-party audio.

    Returns:
        ``(body, content_type_header)``

    Raises:
        DriveError: If the payload is oversized, or (astronomically unlikely)
            contains the generated boundary.
    """
    if len(content) > MAX_UPLOAD_BYTES:
        raise DriveError("The transcript is too large to upload.")

    metadata_bytes = json.dumps(metadata, ensure_ascii=False).encode("utf-8")

    for _ in range(4):
        boundary = secrets.token_hex(16)
        marker = f"--{boundary}".encode("ascii")
        if marker not in content and marker not in metadata_bytes:
            break
    else:  # pragma: no cover - requires 4 consecutive 128-bit collisions
        raise DriveError("Could not generate a safe multipart boundary.")

    parts = [
        f"--{boundary}\r\n".encode("ascii"),
        b"Content-Type: application/json; charset=UTF-8\r\n\r\n",
        metadata_bytes,
        f"\r\n--{boundary}\r\n".encode("ascii"),
        f"Content-Type: {content_type}\r\n\r\n".encode("ascii"),
        content,
        f"\r\n--{boundary}--\r\n".encode("ascii"),
    ]
    return b"".join(parts), f"multipart/related; boundary={boundary}"


def _backoff_delay(attempt: int) -> float:
    """Exponential backoff with full jitter, per Google's documented guidance.

    Args:
        attempt: 1-based attempt number that just failed.
    """
    ceiling = min(BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)), 8.0)
    return random.uniform(0, ceiling)  # noqa: S311 - jitter, not cryptographic


def validate_web_view_link(link: object) -> Optional[str]:
    """Return *link* if it is a URL we are willing to hand to the browser.

    An API response is untrusted input; opening it unconditionally would let a
    malicious or compromised response drive the user's browser anywhere.
    """
    if not isinstance(link, str) or not link:
        return None
    try:
        parsed = urllib.parse.urlsplit(link)
    except ValueError:
        return None
    if parsed.scheme != "https":
        return None
    host = (parsed.hostname or "").lower()
    if host != "google.com" and not host.endswith(".google.com"):
        return None
    return link


class DriveClient:
    """Thin Drive v3 wrapper.

    Args:
        access_token_provider: Callable returning a currently-valid access
            token. Called immediately before each attempt, so a token that
            expires mid-retry is transparently replaced.
    """

    def __init__(self, access_token_provider: Callable[[], str]) -> None:
        self._token_provider = access_token_provider

    # --- transport ---------------------------------------------------

    def _request(self, method: str, url: str, *, body: Optional[bytes] = None,
                 content_type: Optional[str] = None) -> dict:
        """Perform a request with retry/backoff and return the parsed JSON."""
        if not url.startswith("https://"):
            raise DriveError(f"Refusing to send a token to {url!r}")

        total_backoff = 0.0
        last_error: Optional[DriveError] = None

        for attempt in range(1, MAX_ATTEMPTS + 1):
            headers = {
                # Never as an ?access_token= query parameter: URLs land in logs.
                "Authorization": f"Bearer {self._token_provider()}",
                "Accept": "application/json",
            }
            if content_type:
                headers["Content-Type"] = content_type

            request = urllib.request.Request(
                url, data=body, method=method, headers=headers,
            )
            t0 = time.perf_counter()
            try:
                with _opener().open(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
                    raw = response.read(MAX_RESPONSE_BYTES + 1)
                    status = response.status
            except urllib.error.HTTPError as exc:
                raw = exc.read(MAX_RESPONSE_BYTES + 1)
                status = exc.code
            except (urllib.error.URLError, OSError) as exc:
                last_error = DriveOffline(f"Could not reach Google Drive: {exc}")
                status = None
                raw = b""

            elapsed = time.perf_counter() - t0
            log.info(
                "Drive %s %s -> %s in %.2fs (attempt %d/%d)",
                method, urllib.parse.urlsplit(url).path, status, elapsed,
                attempt, MAX_ATTEMPTS,
            )

            if status is not None:
                if len(raw) > MAX_RESPONSE_BYTES:
                    raise DriveError("Google Drive returned an oversized response.")
                if status < 300:
                    return self._parse_json(raw)
                if status == 404:
                    raise DriveNotFound(
                        "The file or folder no longer exists in Google Drive.",
                        status=404,
                    )
                if status not in _RETRY_STATUSES:
                    raise DriveError(self._error_message(raw, status), status=status)
                last_error = DriveError(self._error_message(raw, status), status=status)

            if attempt == MAX_ATTEMPTS:
                break
            delay = _backoff_delay(attempt)
            if total_backoff + delay > MAX_TOTAL_BACKOFF_SECONDS:
                log.warning("Drive retry budget exhausted after %.1fs.", total_backoff)
                break
            total_backoff += delay
            log.info("Retrying the Drive request in %.2fs.", delay)
            time.sleep(delay)

        raise last_error or DriveError("The Google Drive request failed.")

    @staticmethod
    def _parse_json(raw: bytes) -> dict:
        if not raw:
            return {}
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise DriveError("Google Drive returned a malformed response.") from exc
        if not isinstance(payload, dict):
            raise DriveError("Google Drive returned an unexpected response.")
        return payload

    @staticmethod
    def _error_message(raw: bytes, status: int) -> str:
        try:
            payload = json.loads(raw.decode("utf-8"))
            message = payload["error"]["message"]
        except Exception:  # noqa: BLE001 - error bodies are not guaranteed JSON
            message = ""
        return f"Google Drive error {status}{': ' + message if message else ''}"

    # --- operations --------------------------------------------------

    def create_folder(self, name: str, *, parent: Optional[str] = None) -> str:
        """Create a folder and return its id."""
        metadata: dict = {"name": name, "mimeType": FOLDER_MIME}
        if parent:
            metadata["parents"] = [parent]
        body = json.dumps(metadata, ensure_ascii=False).encode("utf-8")
        url = f"{DRIVE_FILES_URL}?{urllib.parse.urlencode({'fields': 'id'})}"
        payload = self._request(
            "POST", url, body=body, content_type="application/json; charset=UTF-8",
        )
        folder_id = payload.get("id")
        if not isinstance(folder_id, str) or not folder_id:
            raise DriveError("Google Drive did not return a folder id.")
        log.info("Created the Drive folder %r (id %s).", name, folder_id)
        return folder_id

    def folder_exists(self, folder_id: str) -> bool:
        """Return True if *folder_id* is still a live, untrashed folder.

        ``drive.file`` cannot search the user's Drive, so a stored id is the
        only handle we have; this is how we detect that it went stale.
        """
        if not folder_id:
            return False
        query = urllib.parse.urlencode({"fields": "id,trashed,mimeType"})
        try:
            payload = self._request(
                "GET", f"{DRIVE_FILES_URL}/{urllib.parse.quote(folder_id)}?{query}",
            )
        except DriveNotFound:
            return False
        return bool(
            payload.get("id")
            and not payload.get("trashed")
            and payload.get("mimeType") == FOLDER_MIME
        )

    def create_document(self, *, name: str, content: bytes, content_type: str,
                        parent: Optional[str]) -> dict:
        """Upload *content* as a native Google Doc and return ``{id, webViewLink}``."""
        metadata: dict = {"name": name, "mimeType": GOOGLE_DOC_MIME}
        if parent:
            metadata["parents"] = [parent]
        body, multipart_type = build_multipart_body(metadata, content, content_type)
        query = urllib.parse.urlencode({
            "uploadType": "multipart", "fields": "id,webViewLink,name",
        })
        payload = self._request(
            "POST", f"{DRIVE_UPLOAD_URL}?{query}",
            body=body, content_type=multipart_type,
        )
        if not payload.get("id"):
            raise DriveError("Google Drive did not return a file id.")
        log.info("Created the Drive document %s.", payload.get("id"))
        return payload

    def update_document(self, file_id: str, *, name: str, content: bytes,
                        content_type: str) -> dict:
        """Replace the content of an existing document.

        Raises:
            DriveNotFound: The user deleted it in Drive; the caller should fall
                back to :meth:`create_document`.
        """
        # "parents" is not accepted on update; changing it needs addParents.
        metadata = {"name": name, "mimeType": GOOGLE_DOC_MIME}
        body, multipart_type = build_multipart_body(metadata, content, content_type)
        query = urllib.parse.urlencode({
            "uploadType": "multipart", "fields": "id,webViewLink,name",
        })
        payload = self._request(
            "PATCH",
            f"{DRIVE_UPLOAD_URL}/{urllib.parse.quote(file_id)}?{query}",
            body=body, content_type=multipart_type,
        )
        log.info("Updated the Drive document %s.", file_id)
        return payload

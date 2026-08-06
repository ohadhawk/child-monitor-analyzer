"""
Google OAuth 2.0 for a native app (RFC 8252) with PKCE (RFC 7636).

Design constraints, in order of priority:

* **Public client.** PKCE carries the real security. Google nevertheless issues
  a ``client_secret`` for Desktop clients and its token endpoint rejects the
  exchange without one, so we send it when configured -- but it is not treated
  as a secret: Google itself states installed apps "cannot keep secrets".
* **System browser + loopback redirect.** Embedded webviews are rejected by
  Google (``disallowed_useragent``) and would put us in the password path.
* **Least privilege**: ``drive.file`` only — the app can only ever see files it
  created itself.
* **The access token never touches disk.** Only the refresh token is persisted,
  and only in the OS keystore (:mod:`monitor.gdrive.store`).

Everything here is headless: no PySide6 import, so it is unit-testable and the
GUI layer owns all threading.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import http.server
import json
import logging
import os
import secrets
import socket
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)

# ===========================
# CONFIGURATION
# ===========================

AUTH_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
REVOKE_ENDPOINT = "https://oauth2.googleapis.com/revoke"

#: Least-privilege scope set. ``openid email`` are both non-sensitive and are
#: only used to show *which* account is linked.
SCOPES = ("https://www.googleapis.com/auth/drive.file", "openid", "email")

#: Environment override, so the OAuth client can be supplied without a rebuild.
CLIENT_ID_ENV_VAR = "MONITOR_GOOGLE_CLIENT_ID"
CLIENT_SECRET_ENV_VAR = "MONITOR_GOOGLE_CLIENT_SECRET"

#: Baked-in client id. Empty until a Google Cloud "Desktop app" client exists;
#: while empty the whole Drive feature reports itself as unconfigured.
DEFAULT_CLIENT_ID = ""

#: Companion to :data:`DEFAULT_CLIENT_ID`. Not a true secret (see module docs),
#: but still never logged.
DEFAULT_CLIENT_SECRET = ""

REDIRECT_PATH = "/oauth2/callback"
LOOPBACK_HOST = "127.0.0.1"

#: How long to wait for the user to finish the consent screen.
AUTH_TIMEOUT_SECONDS = 120.0

#: Poll slice while waiting for the callback, so a cancel is acted on quickly.
_CANCEL_POLL_SECONDS = 0.1

#: Refresh this many seconds before the access token actually expires.
REFRESH_SKEW_SECONDS = 60

#: Token endpoint responses are small; anything larger is not ours.
MAX_RESPONSE_BYTES = 64 * 1024

HTTP_TIMEOUT_SECONDS = 30


class AuthError(RuntimeError):
    """OAuth flow failed. The message is safe to show to the user."""


class AuthCancelled(AuthError):
    """The user abandoned the sign-in before it completed."""


class AuthTimeout(AuthCancelled):
    """The consent screen was not completed within the timeout.

    Subclasses :class:`AuthCancelled` because both mean "no credentials, and
    the user is not surprised" - only the message differs.
    """


class AuthDenied(AuthError):
    """The user actively declined the consent screen."""


class InvalidGrant(AuthError):
    """The refresh token was rejected; the account must be re-linked."""


class MissingScope(AuthError):
    """Sign-in succeeded but Drive access was not granted.

    Google's granular consent screen lists each requested permission as its own
    checkbox, unticked by default, and "Continue" stays enabled whether or not
    the user ticks it. Skipping the Drive box therefore yields a perfectly
    valid token that simply cannot reach Drive.
    """


def client_id() -> str:
    """Return the configured OAuth client id (env var wins)."""
    return os.environ.get(CLIENT_ID_ENV_VAR, "").strip() or DEFAULT_CLIENT_ID


def client_secret() -> str:
    """Return the configured OAuth client secret, or "" if there is none."""
    return os.environ.get(CLIENT_SECRET_ENV_VAR, "").strip() or DEFAULT_CLIENT_SECRET


def _client_credentials() -> dict[str, str]:
    """Client authentication fields for a token request."""
    form = {"client_id": client_id()}
    secret = client_secret()
    if secret:
        form["client_secret"] = secret
    return form


def is_configured() -> bool:
    """Return True if an OAuth client id is available."""
    return bool(client_id())


# ===========================
# CREDENTIALS
# ===========================

@dataclass
class GoogleCredentials:
    """A linked Google account.

    ``access_token`` is memory-only and is never serialised by this class.
    """

    account_id: str                      # Google "sub" claim - opaque, stable
    email: str = ""
    refresh_token: str = field(default="", repr=False)
    access_token: str = field(default="", repr=False)
    access_expires_at: float = 0.0

    def __repr__(self) -> str:  # pragma: no cover - defensive
        return (
            f"GoogleCredentials(account_id={self.account_id[:6]}..., "
            f"email={self.email!r}, has_refresh={bool(self.refresh_token)}, "
            f"access_valid={self.access_token_valid()})"
        )

    def access_token_valid(self, *, now: Optional[float] = None) -> bool:
        """Return True if the access token is present and not near expiry."""
        if not self.access_token:
            return False
        current = time.time() if now is None else now
        return current < self.access_expires_at - REFRESH_SKEW_SECONDS


# ===========================
# PKCE
# ===========================

def generate_pkce() -> tuple[str, str]:
    """Return ``(code_verifier, code_challenge)`` for ``S256``.

    The verifier is 86 characters, comfortably inside RFC 7636's 43-128 range.
    """
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


# ===========================
# LOOPBACK LISTENER
# ===========================

_SUCCESS_HTML = (
    "<!doctype html><html><head><meta charset='utf-8'>"
    "<title>Child Monitor Analyzer</title></head>"
    "<body style='font-family:sans-serif;text-align:center;padding-top:4em'>"
    "<h2>ההתחברות הושלמה</h2><p>אפשר לסגור את החלון ולחזור לאפליקציה.</p>"
    "</body></html>"
)

_FAILURE_HTML = (
    "<!doctype html><html><head><meta charset='utf-8'>"
    "<title>Child Monitor Analyzer</title></head>"
    "<body style='font-family:sans-serif;text-align:center;padding-top:4em'>"
    "<h2>ההתחברות נכשלה</h2><p>אפשר לסגור את החלון ולנסות שוב.</p>"
    "</body></html>"
)


class _CallbackResult:
    """Thread-safe slot for the single value the listener produces.

    Strictly write-once: the listener is threaded, and the window between the
    waiter waking up and the server shutting down must not let a second
    request replace the code that is about to be redeemed.
    """

    def __init__(self) -> None:
        self.code: Optional[str] = None
        self.error: Optional[str] = None
        self.done = threading.Event()
        self._lock = threading.Lock()

    def resolve(self, *, code: Optional[str] = None,
                error: Optional[str] = None) -> bool:
        """Record the outcome. Returns False if it was already recorded."""
        with self._lock:
            if self.done.is_set():
                return False
            self.code = code
            self.error = error
            self.done.set()
            return True


def _build_handler(expected_state: str, result: _CallbackResult):
    """Build a one-shot request handler bound to *expected_state*."""

    class _Handler(http.server.BaseHTTPRequestHandler):
        # BaseHTTPRequestHandler logs the full request line to stderr by
        # default - which would write the authorization code into the log.
        def log_message(self, fmt, *args) -> None:  # noqa: A003
            return

        protocol_version = "HTTP/1.0"

        # A connection that opens and then says nothing would otherwise pin a
        # handler thread for the whole flow. Any local process can open one.
        timeout = 10

        def handle_one_request(self) -> None:
            try:
                super().handle_one_request()
            except (TimeoutError, socket.timeout, OSError):
                self.close_connection = True

        def _respond(self, status: int, body: str) -> None:
            payload = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            # The request URL holds the auth code: keep it out of caches and
            # out of any Referer header, and forbid all subresource loads.
            self.send_header("Cache-Control", "no-store")
            self.send_header("Pragma", "no-cache")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'none'")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
            parsed = urllib.parse.urlsplit(self.path)
            if parsed.path != REDIRECT_PATH:
                self.send_error(404)
                return

            params = urllib.parse.parse_qs(parsed.query)
            state = (params.get("state") or [""])[0]
            # Constant-time, and *before* looking at the code at all.
            if not hmac.compare_digest(state, expected_state):
                log.warning("OAuth callback rejected: state mismatch.")
                self._respond(400, _FAILURE_HTML)
                result.resolve(error="state mismatch")
                return

            if "error" in params:
                error = (params.get("error") or ["unknown"])[0]
                log.info("OAuth callback reported an error: %s", error)
                self._respond(200, _FAILURE_HTML)
                result.resolve(error=error)
                return

            code = (params.get("code") or [""])[0]
            if not code:
                self._respond(400, _FAILURE_HTML)
                result.resolve(error="no authorization code in callback")
                return

            if result.resolve(code=code):
                self._respond(200, _SUCCESS_HTML)
            else:
                self._respond(400, _FAILURE_HTML)

    return _Handler


class _OneShotServer(http.server.ThreadingHTTPServer):
    """Loopback callback listener.

    Threaded so one stalled connection cannot starve the real callback; the
    threads are daemons so a hung client can never keep the process alive.
    The result object is what makes the flow single-shot, not the server.
    """

    # Refuse to share the port with anything else.
    allow_reuse_address = False
    address_family = socket.AF_INET
    daemon_threads = True


def _start_listener(expected_state: str) -> tuple[_OneShotServer, _CallbackResult, int]:
    """Bind an ephemeral loopback port and start serving in a daemon thread.

    Binding to ``127.0.0.1`` (never ``0.0.0.0``, never the name ``localhost``,
    which can resolve to ``::1`` or be poisoned) keeps the callback off the LAN.
    Port 0 lets the OS pick, so no other local process can squat a fixed port.
    """
    result = _CallbackResult()
    server = _OneShotServer((LOOPBACK_HOST, 0), _build_handler(expected_state, result))
    port = server.server_address[1]
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.2},
        name="oauth-loopback", daemon=True,
    )
    thread.start()
    log.info("OAuth loopback listener bound to %s:%d.", LOOPBACK_HOST, port)
    return server, result, port


# ===========================
# HTTP
# ===========================

class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Token/revoke endpoints must never redirect."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise AuthError(f"Unexpected redirect from {req.full_url} to {newurl}")


def _opener() -> urllib.request.OpenerDirector:
    context = ssl.create_default_context()
    return urllib.request.build_opener(
        urllib.request.HTTPSHandler(context=context), _NoRedirectHandler(),
    )


def _post_form(url: str, form: dict[str, str]) -> dict:
    """POST *form* as ``application/x-www-form-urlencoded`` and parse JSON.

    Raises:
        AuthError: On transport failure or a non-JSON / oversized response.
        InvalidGrant: When Google reports ``invalid_grant``.
    """
    if not url.startswith("https://"):
        raise AuthError(f"Refusing to send credentials over {url!r}")

    body = urllib.parse.urlencode(form).encode("ascii")
    request = urllib.request.Request(
        url, data=body, method="POST",
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        },
    )
    t0 = time.perf_counter()
    try:
        with _opener().open(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
            status = response.status
    except urllib.error.HTTPError as exc:
        raw = exc.read(MAX_RESPONSE_BYTES + 1)
        status = exc.code
    except urllib.error.URLError as exc:
        raise AuthError(f"Could not reach Google: {exc.reason}") from exc
    except OSError as exc:
        raise AuthError(f"Could not reach Google: {exc}") from exc

    if len(raw) > MAX_RESPONSE_BYTES:
        raise AuthError("Google returned an implausibly large response.")

    log.debug(
        "POST %s -> HTTP %d in %.2fs (%d bytes)",
        urllib.parse.urlsplit(url).path, status, time.perf_counter() - t0, len(raw),
    )

    try:
        payload = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise AuthError(f"Google returned a malformed response (HTTP {status}).") from exc
    if not isinstance(payload, dict):
        raise AuthError(f"Google returned an unexpected response (HTTP {status}).")

    if status >= 400:
        error = str(payload.get("error", "unknown_error"))
        description = str(payload.get("error_description", ""))
        if error == "invalid_grant":
            raise InvalidGrant("The Google authorization has expired or was revoked.")
        raise AuthError(f"Google rejected the request: {error} {description}".strip())

    return payload


# ===========================
# ID TOKEN (display only)
# ===========================

def decode_id_token_claims(id_token: str, *, expected_aud: str,
                           now: Optional[float] = None) -> dict:
    """Return the validated claims of *id_token*.

    The signature is deliberately **not** verified: the token arrived over TLS
    directly from Google's token endpoint in response to our own PKCE-bound
    request, so there is no untrusted transport to defend against, and shipping
    a JWKS client would add a dependency for no gain. ``iss``/``aud``/``exp``
    are still checked so a mismatched or stale token is not displayed, and the
    ``email`` claim is used for display only - never for an authorization
    decision.

    Raises:
        AuthError: If the token is malformed or the claims do not check out.
    """
    parts = id_token.split(".")
    if len(parts) != 3:
        raise AuthError("Malformed id_token.")
    try:
        padded = parts[1] + "=" * (-len(parts[1]) % 4)
        claims = json.loads(base64.urlsafe_b64decode(padded).decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise AuthError("Malformed id_token payload.") from exc
    if not isinstance(claims, dict):
        raise AuthError("Malformed id_token payload.")

    issuer = claims.get("iss")
    if issuer not in ("https://accounts.google.com", "accounts.google.com"):
        raise AuthError("id_token has an unexpected issuer.")

    audience = claims.get("aud")
    # An empty expected audience would make compare_digest succeed against an
    # empty claim, so refuse to validate at all in that case.
    if not expected_aud:
        raise AuthError("No OAuth client id to validate the id_token against.")
    if not isinstance(audience, str) or not hmac.compare_digest(audience, expected_aud):
        raise AuthError("id_token was not issued for this application.")

    current = time.time() if now is None else now
    try:
        expiry = float(claims.get("exp", 0))
    except (TypeError, ValueError):
        expiry = 0.0
    if expiry <= current:
        raise AuthError("id_token has expired.")

    return claims


# ===========================
# FLOWS
# ===========================

def build_authorization_url(*, challenge: str, state: str, redirect_uri: str) -> str:
    """Build the consent-screen URL."""
    params = {
        "client_id": client_id(),
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": " ".join(SCOPES),
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": state,
        "access_type": "offline",
        "prompt": "consent",
    }
    return f"{AUTH_ENDPOINT}?{urllib.parse.urlencode(params)}"


def _wait_for_callback(
    result: "_CallbackResult",
    *,
    timeout: float,
    cancel_event: Optional[threading.Event],
) -> None:
    """Block until the callback arrives, the user cancels, or time runs out.

    Polled rather than a single ``wait(timeout)`` so a cancel is noticed
    promptly; the slice is long enough that the wait stays effectively idle.
    """
    deadline = time.monotonic() + timeout
    while not result.done.wait(_CANCEL_POLL_SECONDS):
        if cancel_event is not None and cancel_event.is_set():
            raise AuthCancelled("The Google sign-in was cancelled.")
        if time.monotonic() >= deadline:
            raise AuthTimeout("Timed out waiting for the Google consent screen.")


#: The one scope the feature cannot work without.
DRIVE_SCOPE = "https://www.googleapis.com/auth/drive.file"


def _check_granted_scopes(payload: dict) -> None:
    """Reject a token that cannot actually reach Drive.

    Google returns 200 with a reduced scope set when the user leaves the Drive
    checkbox unticked on the granular consent screen; without this the failure
    would only appear as a 403 on the first upload, far from its cause.
    """
    granted = payload.get("scope")
    if not isinstance(granted, str) or not granted:
        return  # Nothing to check against; the upload will report any problem.
    if DRIVE_SCOPE not in granted.split():
        # Scope names carry no user data, and knowing what *was* granted is the
        # difference between a five-minute and a five-hour diagnosis.
        log.warning("Google granted only these scopes: %s", granted)
        raise MissingScope(
            "Google did not grant Drive access. On the Google consent screen, "
            "tick the box for 'See, edit, create and delete only the specific "
            "Google Drive files that you use with this app' before pressing "
            "Continue - it is not ticked for you."
        )


def start_link_flow(
    *,
    open_browser=None,
    timeout: float = AUTH_TIMEOUT_SECONDS,
    cancel_event: Optional[threading.Event] = None,
) -> GoogleCredentials:
    """Run the full interactive authorization and return fresh credentials.

    Blocking; the GUI must call this from a worker thread.

    Args:
        open_browser: Callable receiving the authorization URL. Defaults to
            :func:`webbrowser.open`. Injected for tests.
        timeout: Seconds to wait for the user to finish.
        cancel_event: Set this to abandon the flow early. It is honoured only
            while waiting for the callback - once the code has been redeemed,
            aborting would leave Google holding a grant we then discarded.

    Raises:
        AuthError: Configuration, transport or protocol failure.
        AuthDenied: The user declined the consent screen.
        AuthCancelled: The user cancelled locally.
        AuthTimeout: The timeout elapsed.
        MissingScope: Sign-in succeeded without Drive access.
    """
    if not is_configured():
        raise AuthError(
            "No Google OAuth client is configured for this build "
            f"(set {CLIENT_ID_ENV_VAR})."
        )

    verifier, challenge = generate_pkce()
    state = secrets.token_urlsafe(32)

    server, result, port = _start_listener(state)
    redirect_uri = f"http://{LOOPBACK_HOST}:{port}{REDIRECT_PATH}"
    try:
        url = build_authorization_url(
            challenge=challenge, state=state, redirect_uri=redirect_uri,
        )
        if open_browser is None:
            import webbrowser
            open_browser = webbrowser.open
        log.info("Opening the system browser for Google consent.")
        open_browser(url)

        _wait_for_callback(result, timeout=timeout, cancel_event=cancel_event)
        if result.error:
            if result.error == "access_denied":
                raise AuthDenied("Access to Google Drive was declined.")
            raise AuthError(f"Google sign-in was not completed: {result.error}")
        if not result.code:
            raise AuthError("No authorization code was received.")
        code = result.code
    finally:
        # Always tear the listener down, on every path.
        try:
            server.shutdown()
        except Exception:  # noqa: BLE001
            log.debug("Loopback listener shutdown failed", exc_info=True)
        server.server_close()
        log.info("OAuth loopback listener closed.")

    payload = _post_form(TOKEN_ENDPOINT, {
        **_client_credentials(),
        "code": code,
        "code_verifier": verifier,
        "grant_type": "authorization_code",
        "redirect_uri": redirect_uri,
    })
    _check_granted_scopes(payload)
    return _credentials_from_token_response(payload, previous=None)


def refresh_access_token(credentials: GoogleCredentials) -> GoogleCredentials:
    """Exchange the refresh token for a new access token.

    Raises:
        InvalidGrant: The grant is gone; the caller must purge and re-link.
    """
    if not credentials.refresh_token:
        raise InvalidGrant("No refresh token is stored for this account.")
    payload = _post_form(TOKEN_ENDPOINT, {
        **_client_credentials(),
        "refresh_token": credentials.refresh_token,
        "grant_type": "refresh_token",
    })
    log.info("Refreshed the Google access token.")
    return _credentials_from_token_response(payload, previous=credentials)


def revoke(refresh_token: str) -> bool:
    """Revoke *refresh_token* server-side.

    Returns:
        True if Google accepted the revocation. A False result still means the
        caller must purge locally, but the user should be told the grant may
        still be live.
    """
    if not refresh_token:
        return True
    try:
        _post_form(REVOKE_ENDPOINT, {"token": refresh_token})
    except InvalidGrant:
        return True  # already revoked or expired - the desired end state
    except AuthError as exc:
        log.warning("Google token revocation failed: %s", exc)
        return False
    log.info("Google authorization revoked server-side.")
    return True


def _credentials_from_token_response(
    payload: dict, *, previous: Optional[GoogleCredentials],
) -> GoogleCredentials:
    """Build credentials from a token endpoint response."""
    access_token = payload.get("access_token")
    if not isinstance(access_token, str) or not access_token:
        raise AuthError("Google did not return an access token.")

    try:
        expires_in = float(payload.get("expires_in", 0))
    except (TypeError, ValueError):
        expires_in = 0.0
    if expires_in <= 0:
        expires_in = 300.0  # conservative: forces an early refresh

    # Google only returns a refresh token on the first authorization.
    refresh_token = payload.get("refresh_token")
    if not isinstance(refresh_token, str) or not refresh_token:
        refresh_token = previous.refresh_token if previous else ""

    account_id = previous.account_id if previous else ""
    email = previous.email if previous else ""
    id_token = payload.get("id_token")
    if isinstance(id_token, str) and id_token:
        try:
            claims = decode_id_token_claims(id_token, expected_aud=client_id())
        except AuthError as exc:
            log.warning("Ignoring the id_token: %s", exc)
        else:
            account_id = str(claims.get("sub") or account_id)
            email = str(claims.get("email") or email)

    if not account_id:
        raise AuthError("Google did not identify the account.")

    return GoogleCredentials(
        account_id=account_id,
        email=email,
        refresh_token=refresh_token,
        access_token=access_token,
        access_expires_at=time.time() + expires_in,
    )

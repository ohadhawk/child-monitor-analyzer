"""
Security regression tests for the Google Drive integration.

No network and no Qt. Everything that touches HTTP is stubbed, so these tests
assert the *protocol and security invariants*, which is what actually matters:
PKCE derivation, ``state`` verification, callback hardening, multipart safety,
retry bounds, untrusted-response handling and credential-store fail-safety.
"""

from __future__ import annotations

import ast
import gc
import base64
import hashlib
import json
import logging
import re
import threading
import time
import urllib.error
import urllib.parse
from pathlib import Path

import pytest

from monitor.gdrive import auth, client, store
import monitor.gdrive as gdrive


# ===========================
# CONFIGURATION
# ===========================

def test_scope_is_least_privilege():
    # drive.file is non-sensitive and confines a stolen token to files this
    # app itself created. Anything broader would require a CASA assessment.
    assert "https://www.googleapis.com/auth/drive.file" in auth.SCOPES
    for scope in auth.SCOPES:
        assert not scope.endswith("/auth/drive")
        assert "drive.readonly" not in scope
        assert "drive.metadata" not in scope
    assert set(auth.SCOPES) == {
        "https://www.googleapis.com/auth/drive.file", "openid", "email",
    }


def test_no_client_secret_value_is_shipped():
    """Google requires a client_secret field, but no value may be committed.

    Checked against the parsed AST, not the text: ``client_secret`` is a
    legitimate form-field name, so only assignments to the baked-in constant
    and secret-shaped literals are of interest.
    """
    tree = ast.parse(Path(auth.__file__).read_text(encoding="utf-8"))
    literals = {
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        and not _is_docstring(tree, node)
    }
    assert not any(text.startswith("GOCSPX-") for text in literals)
    assert auth.DEFAULT_CLIENT_SECRET == ""


def _is_docstring(tree, node) -> bool:
    for parent in ast.walk(tree):
        body = getattr(parent, "body", None)
        if isinstance(body, list) and body:
            first = body[0]
            if isinstance(first, ast.Expr) and first.value is node:
                return True
    return False


def test_token_request_carries_no_client_secret(monkeypatch):
    monkeypatch.setenv(auth.CLIENT_ID_ENV_VAR, "cid")
    captured = {}

    def _fake_post(url, form):
        captured["url"] = url
        captured["form"] = form
        return {"access_token": "ya29.A", "expires_in": 3600}

    monkeypatch.setattr(auth, "_post_form", _fake_post)
    auth.refresh_access_token(auth.GoogleCredentials(
        account_id="SUB1", refresh_token="1//OLD"))
    assert "client_secret" not in captured["form"]
    assert captured["url"] == auth.TOKEN_ENDPOINT


def test_endpoints_are_https():
    for url in (auth.AUTH_ENDPOINT, auth.TOKEN_ENDPOINT, auth.REVOKE_ENDPOINT,
                client.DRIVE_FILES_URL, client.DRIVE_UPLOAD_URL):
        assert url.startswith("https://")


def test_client_id_can_be_supplied_by_environment(monkeypatch):
    monkeypatch.setenv(auth.CLIENT_ID_ENV_VAR, "test-client.apps.googleusercontent.com")
    assert auth.client_id() == "test-client.apps.googleusercontent.com"


def test_both_halves_are_needed_before_the_feature_is_offered(monkeypatch):
    """An id alone reaches Google and is rejected there, after the consent screen.

    Observed in the field: the log said configured=True, the user worked through
    the whole browser flow, and Google answered "invalid_request client_secret
    is missing". The readiness check has to cover what the exchange requires.
    """
    monkeypatch.setattr(auth, "DEFAULT_CLIENT_ID", "")
    monkeypatch.setattr(auth, "DEFAULT_CLIENT_SECRET", "")
    monkeypatch.setenv(auth.CLIENT_ID_ENV_VAR, "test-client.apps.googleusercontent.com")
    monkeypatch.delenv(auth.CLIENT_SECRET_ENV_VAR, raising=False)
    assert not auth.is_configured(), "an id without a secret was accepted"

    monkeypatch.setenv(auth.CLIENT_SECRET_ENV_VAR, "GOCSPX-secret")
    assert auth.is_configured()


def test_the_missing_half_is_named(monkeypatch):
    """"Not configured" sends people to the Cloud Console for the wrong thing."""
    monkeypatch.setattr(auth, "DEFAULT_CLIENT_ID", "")
    monkeypatch.setattr(auth, "DEFAULT_CLIENT_SECRET", "")
    monkeypatch.setenv(auth.CLIENT_ID_ENV_VAR, "test-client.apps.googleusercontent.com")
    monkeypatch.delenv(auth.CLIENT_SECRET_ENV_VAR, raising=False)

    with pytest.raises(auth.AuthError) as caught:
        auth.start_link_flow(open_browser=lambda url: None)
    assert auth.CLIENT_SECRET_ENV_VAR in str(caught.value)

    monkeypatch.delenv(auth.CLIENT_ID_ENV_VAR, raising=False)
    with pytest.raises(auth.AuthError) as caught:
        auth.start_link_flow(open_browser=lambda url: None)
    assert auth.CLIENT_ID_ENV_VAR in str(caught.value)


def test_feature_is_off_when_unconfigured(monkeypatch):
    monkeypatch.setenv(auth.CLIENT_ID_ENV_VAR, "")
    monkeypatch.setattr(auth, "DEFAULT_CLIENT_ID", "")
    assert not auth.is_configured()
    with pytest.raises(auth.AuthError):
        auth.start_link_flow(open_browser=lambda url: None)


# ===========================
# PKCE
# ===========================

def test_pkce_challenge_matches_rfc7636_test_vector():
    verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    expected = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    assert expected == "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"


def test_generated_verifier_meets_rfc7636_length_and_alphabet():
    verifier, challenge = auth.generate_pkce()
    assert 43 <= len(verifier) <= 128
    assert all(c.isalnum() or c in "-._~" for c in verifier)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    assert challenge == base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    assert "=" not in challenge


def test_each_flow_gets_a_fresh_verifier():
    assert auth.generate_pkce()[0] != auth.generate_pkce()[0]


def test_authorization_url_uses_s256_and_carries_the_state(monkeypatch):
    monkeypatch.setenv(auth.CLIENT_ID_ENV_VAR, "cid")
    url = auth.build_authorization_url(
        challenge="CHAL", state="STATE",
        redirect_uri="http://127.0.0.1:1234/oauth2/callback",
    )
    params = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
    assert params["code_challenge_method"] == ["S256"]
    assert params["code_challenge"] == ["CHAL"]
    assert params["state"] == ["STATE"]
    assert params["response_type"] == ["code"]
    assert "client_secret" not in params


# ===========================
# LOOPBACK LISTENER
# ===========================

def _get(port: str | int, path: str) -> tuple[int, bytes, dict]:
    """Issue a plain GET to the loopback listener."""
    import http.client
    conn = http.client.HTTPConnection("127.0.0.1", int(port), timeout=5)
    conn.request("GET", path)
    response = conn.getresponse()
    body = response.read()
    headers = dict(response.getheaders())
    conn.close()
    return response.status, body, headers


@pytest.fixture(autouse=True)
def _no_ambient_oauth_config(monkeypatch):
    """Ignore any real OAuth client configured on the developer's machine.

    The env vars outrank the module defaults, so without this a machine with
    Drive set up would exercise a different code path than CI.
    """
    monkeypatch.delenv(auth.CLIENT_ID_ENV_VAR, raising=False)
    monkeypatch.delenv(auth.CLIENT_SECRET_ENV_VAR, raising=False)


@pytest.fixture
def listener():
    server, result, port = auth._start_listener("EXPECTED_STATE")
    try:
        yield server, result, port
    finally:
        server.shutdown()
        server.server_close()


def test_listener_binds_loopback_only_on_an_ephemeral_port(listener):
    server, _result, port = listener
    assert server.server_address[0] == "127.0.0.1"
    assert port > 0
    assert server.allow_reuse_address is False


def test_valid_callback_captures_the_code(listener):
    _server, result, port = listener
    status, _body, headers = _get(port, "/oauth2/callback?state=EXPECTED_STATE&code=AUTH1")
    assert status == 200
    assert result.done.wait(2)
    assert result.code == "AUTH1"
    assert result.error is None
    # The URL contains the auth code; it must not be cached or leak via Referer.
    assert headers.get("Cache-Control") == "no-store"
    assert headers.get("Referrer-Policy") == "no-referrer"
    assert headers.get("Content-Security-Policy") == "default-src 'none'"


def test_state_mismatch_is_rejected_and_the_code_is_discarded(listener):
    _server, result, port = listener
    status, _body, _headers = _get(port, "/oauth2/callback?state=WRONG&code=AUTH1")
    assert status == 400
    assert result.done.wait(2)
    assert result.code is None
    assert result.error == "state mismatch"


def test_missing_state_is_rejected(listener):
    _server, result, port = listener
    _get(port, "/oauth2/callback?code=AUTH1")
    assert result.done.wait(2)
    assert result.code is None


def test_other_paths_are_404(listener):
    _server, result, port = listener
    status, _body, _headers = _get(port, "/")
    assert status == 404
    assert not result.done.is_set()


def test_provider_error_is_reported_without_a_code(listener):
    _server, result, port = listener
    _get(port, "/oauth2/callback?state=EXPECTED_STATE&error=access_denied")
    assert result.done.wait(2)
    assert result.error == "access_denied"
    assert result.code is None


def test_success_page_has_no_external_resources(listener):
    _server, _result, port = listener
    _status, body, _headers = _get(port, "/oauth2/callback?state=EXPECTED_STATE&code=A")
    text = body.decode("utf-8")
    for marker in ("http://", "https://", "<script", "<img", "<link"):
        assert marker not in text.lower()


def test_request_logging_is_disabled():
    # BaseHTTPRequestHandler logs the full request line - including the auth
    # code - to stderr by default. This is the single easiest leak to miss.
    result = auth._CallbackResult()
    handler = auth._build_handler("S", result)
    assert handler.log_message is not auth.http.server.BaseHTTPRequestHandler.log_message
    assert handler.log_message(object(), "%s", "anything") is None


def test_flow_times_out_and_tears_the_listener_down(monkeypatch):
    monkeypatch.setenv(auth.CLIENT_ID_ENV_VAR, "cid")
    monkeypatch.setenv(auth.CLIENT_SECRET_ENV_VAR, "GOCSPX-test")
    opened: list[str] = []
    with pytest.raises(auth.AuthCancelled):
        auth.start_link_flow(open_browser=opened.append, timeout=0.05)
    assert opened, "the browser should have been opened"
    port = int(urllib.parse.urlsplit(opened[0]).query.split("127.0.0.1%3A")[0] or 0) \
        if False else None  # port is inside redirect_uri; see below
    redirect = urllib.parse.parse_qs(
        urllib.parse.urlsplit(opened[0]).query)["redirect_uri"][0]
    assert redirect.startswith("http://127.0.0.1:")
    assert redirect.endswith("/oauth2/callback")
    # The socket must be gone.
    import socket
    port = int(urllib.parse.urlsplit(redirect).port)
    probe = socket.socket()
    probe.settimeout(1)
    with pytest.raises(OSError):
        probe.connect(("127.0.0.1", port))
    probe.close()


# ===========================
# TOKEN HANDLING
# ===========================

def _id_token(claims: dict) -> str:
    payload = base64.urlsafe_b64encode(
        json.dumps(claims).encode("utf-8")).rstrip(b"=").decode("ascii")
    return f"header.{payload}.signature"


def test_id_token_claims_are_validated(monkeypatch):
    claims = {"iss": "https://accounts.google.com", "aud": "cid",
              "exp": time.time() + 600, "sub": "SUB1", "email": "a@b.com"}
    decoded = auth.decode_id_token_claims(_id_token(claims), expected_aud="cid")
    assert decoded["email"] == "a@b.com"


@pytest.mark.parametrize("claims", [
    {"iss": "https://evil.example", "aud": "cid", "exp": time.time() + 600},
    {"iss": "https://accounts.google.com", "aud": "other", "exp": time.time() + 600},
    {"iss": "https://accounts.google.com", "aud": "cid", "exp": time.time() - 10},
])
def test_bad_id_tokens_are_rejected(claims):
    with pytest.raises(auth.AuthError):
        auth.decode_id_token_claims(_id_token(claims), expected_aud="cid")


def test_malformed_id_token_is_rejected():
    with pytest.raises(auth.AuthError):
        auth.decode_id_token_claims("not-a-jwt", expected_aud="cid")


def test_credentials_never_expose_secrets_in_repr():
    creds = auth.GoogleCredentials(
        account_id="SUB1", email="a@b.com",
        refresh_token="1//SECRETREFRESH", access_token="ya29.SECRETACCESS",
        access_expires_at=time.time() + 3600,
    )
    text = repr(creds)
    assert "SECRETREFRESH" not in text
    assert "SECRETACCESS" not in text


def test_access_token_validity_uses_a_refresh_skew():
    now = 1000.0
    creds = auth.GoogleCredentials(
        account_id="S", access_token="t", access_expires_at=now + 30,
    )
    # Still nominally valid, but inside the skew - must be treated as stale.
    assert not creds.access_token_valid(now=now)
    creds.access_expires_at = now + auth.REFRESH_SKEW_SECONDS + 10
    assert creds.access_token_valid(now=now)


def test_invalid_grant_is_raised_as_its_own_error(monkeypatch):
    monkeypatch.setattr(auth, "_post_form", lambda *a, **k: (_ for _ in ()).throw(
        auth.InvalidGrant("gone")))
    creds = auth.GoogleCredentials(account_id="S", refresh_token="r")
    with pytest.raises(auth.InvalidGrant):
        auth.refresh_access_token(creds)


def test_refresh_without_a_token_fails_closed():
    with pytest.raises(auth.InvalidGrant):
        auth.refresh_access_token(auth.GoogleCredentials(account_id="S"))


def test_refresh_response_without_a_refresh_token_keeps_the_old_one(monkeypatch):
    monkeypatch.setenv(auth.CLIENT_ID_ENV_VAR, "cid")
    monkeypatch.setattr(auth, "_post_form", lambda *a, **k: {
        "access_token": "ya29.NEW", "expires_in": 3600,
    })
    previous = auth.GoogleCredentials(
        account_id="SUB1", email="a@b.com", refresh_token="1//OLD")
    refreshed = auth.refresh_access_token(previous)
    assert refreshed.refresh_token == "1//OLD"
    assert refreshed.account_id == "SUB1"


def test_token_response_without_an_access_token_is_rejected():
    with pytest.raises(auth.AuthError):
        auth._credentials_from_token_response({}, previous=None)


def test_post_form_refuses_plain_http():
    with pytest.raises(auth.AuthError):
        auth._post_form("http://oauth2.googleapis.com/token", {"a": "b"})


def test_token_endpoint_never_follows_redirects():
    handler = auth._NoRedirectHandler()

    class _Req:
        full_url = "https://oauth2.googleapis.com/token"

    with pytest.raises(auth.AuthError):
        handler.redirect_request(_Req(), None, 302, "Found", {}, "https://evil.example")


def test_revoke_treats_an_already_dead_grant_as_success(monkeypatch):
    monkeypatch.setattr(auth, "_post_form", lambda *a, **k: (_ for _ in ()).throw(
        auth.InvalidGrant("already revoked")))
    assert auth.revoke("1//OLD") is True


def test_revoke_reports_failure_so_the_user_can_be_warned(monkeypatch):
    monkeypatch.setattr(auth, "_post_form", lambda *a, **k: (_ for _ in ()).throw(
        auth.AuthError("network down")))
    assert auth.revoke("1//OLD") is False


# ===========================
# GRANULAR CONSENT
# ===========================

def test_a_partial_grant_is_caught_at_sign_in():
    """Google returns 200 with a reduced scope set when the Drive box is left
    unticked, and the failure would otherwise surface as a 403 much later."""
    with pytest.raises(auth.MissingScope):
        auth._check_granted_scopes({"scope": "openid email"})


def test_a_full_grant_passes():
    auth._check_granted_scopes({"scope": f"openid email {auth.DRIVE_SCOPE}"})


@pytest.mark.parametrize("payload", [{}, {"scope": ""}, {"scope": None}])
def test_an_unstated_scope_set_is_not_second_guessed(payload):
    """Nothing to check against; the first upload will report any problem."""
    auth._check_granted_scopes(payload)


def test_a_scope_that_merely_starts_the_same_is_not_accepted():
    auth._check_granted_scopes({"scope": auth.DRIVE_SCOPE})
    with pytest.raises(auth.MissingScope):
        auth._check_granted_scopes({"scope": auth.DRIVE_SCOPE + ".readonly"})


def test_the_partial_grant_is_logged_with_what_was_granted(caplog):
    """The scope list is the whole diagnosis, and carries no user data."""
    with caplog.at_level(logging.WARNING, logger="monitor.gdrive.auth"):
        with pytest.raises(auth.MissingScope):
            auth._check_granted_scopes({"scope": "openid email"})
    assert "openid email" in caplog.text


def test_the_partial_grant_blames_the_checkbox_not_the_console():
    """The console is configured -- the user simply did not tick the box.

    Sending them to Cloud Console instead wastes their time on a setting that
    is already correct.
    """
    with pytest.raises(auth.MissingScope) as excinfo:
        auth._check_granted_scopes({"scope": "openid email"})
    message = str(excinfo.value)
    assert "tick" in message.lower()
    assert "consent screen" in message.lower()
    assert "APIs & Services" not in message


def test_the_user_is_warned_about_the_checkbox_before_the_browser_opens():
    """Prevention: the hint shown next to the sign-in button must mention it."""
    from monitor.gui.strings import Lang, S, _STRINGS

    assert "תיבת הסימון" in _STRINGS[(S.GOOGLE_BROWSER_HINT, Lang.HE)]
    assert "checkbox" in _STRINGS[(S.GOOGLE_BROWSER_HINT, Lang.EN)]


def test_the_failure_message_tells_the_user_to_tick_the_box():
    from monitor.gui.strings import Lang, S, _STRINGS

    assert "לסמן" in _STRINGS[(S.GOOGLE_MISSING_SCOPE, Lang.HE)]
    assert "Tick it" in _STRINGS[(S.GOOGLE_MISSING_SCOPE, Lang.EN)]


# ===========================
# DRIVE CLIENT
# ===========================

def test_multipart_boundary_is_random_per_request():
    _b1, t1 = client.build_multipart_body({"name": "a"}, b"x", "text/plain")
    _b2, t2 = client.build_multipart_body({"name": "a"}, b"x", "text/plain")
    assert t1 != t2


def test_multipart_body_is_well_formed():
    body, content_type = client.build_multipart_body(
        {"name": "שם", "mimeType": client.GOOGLE_DOC_MIME}, b"hello", client.TXT_MIME,
    )
    boundary = content_type.split("boundary=")[1]
    assert body.startswith(f"--{boundary}\r\n".encode())
    assert body.endswith(f"\r\n--{boundary}--\r\n".encode())
    assert b"hello" in body
    assert "שם".encode("utf-8") in body


def test_metadata_is_json_encoded_not_concatenated():
    body, _ct = client.build_multipart_body(
        {"name": 'evil" , "parents": ["OTHER"]'}, b"x", client.TXT_MIME,
    )
    metadata = json.loads(body.split(b"\r\n\r\n", 1)[1].split(b"\r\n--", 1)[0])
    assert metadata["name"] == 'evil" , "parents": ["OTHER"]'
    assert "parents" not in metadata


def test_boundary_collision_is_detected(monkeypatch):
    # Force every generated boundary to collide with the payload.
    monkeypatch.setattr(client.secrets, "token_hex", lambda n: "deadbeef")
    with pytest.raises(client.DriveError):
        client.build_multipart_body({"name": "a"}, b"--deadbeef inside", "text/plain")


def test_oversized_upload_is_refused():
    with pytest.raises(client.DriveError):
        client.build_multipart_body(
            {"name": "a"}, b"x" * (client.MAX_UPLOAD_BYTES + 1), "text/plain")


def test_backoff_is_bounded_and_jittered():
    for attempt in range(1, client.MAX_ATTEMPTS + 1):
        for _ in range(50):
            delay = client._backoff_delay(attempt)
            assert 0.0 <= delay <= 8.0
    # Monotonically non-decreasing ceiling.
    assert client.BACKOFF_BASE_SECONDS > 0
    assert client.MAX_TOTAL_BACKOFF_SECONDS <= 30.0


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_retryable_statuses_are_retried(monkeypatch, status):
    attempts = []

    def _fake_open(self, request, timeout=None):
        attempts.append(request)
        raise urllib.error.HTTPError(
            request.full_url, status, "err", {}, _BytesIO(b'{"error":{"message":"x"}}'))

    monkeypatch.setattr(client, "_opener", lambda: _FakeOpener(_fake_open))
    monkeypatch.setattr(client.time, "sleep", lambda s: None)
    drive = client.DriveClient(lambda: "token")
    with pytest.raises(client.DriveError):
        drive.create_folder("Transcriptions")
    assert len(attempts) == client.MAX_ATTEMPTS


@pytest.mark.parametrize("status", [400, 401, 403])
def test_non_retryable_statuses_fail_immediately(monkeypatch, status):
    attempts = []

    def _fake_open(self, request, timeout=None):
        attempts.append(request)
        raise urllib.error.HTTPError(
            request.full_url, status, "err", {}, _BytesIO(b"{}"))

    monkeypatch.setattr(client, "_opener", lambda: _FakeOpener(_fake_open))
    drive = client.DriveClient(lambda: "token")
    with pytest.raises(client.DriveError):
        drive.create_folder("Transcriptions")
    assert len(attempts) == 1


def test_404_raises_drive_not_found(monkeypatch):
    def _fake_open(self, request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, 404, "gone", {}, _BytesIO(b"{}"))

    monkeypatch.setattr(client, "_opener", lambda: _FakeOpener(_fake_open))
    drive = client.DriveClient(lambda: "token")
    assert drive.folder_exists("FOLDER") is False


def test_the_token_goes_in_the_header_never_the_query(monkeypatch):
    seen = {}

    def _fake_open(self, request, timeout=None):
        seen["url"] = request.full_url
        seen["headers"] = dict(request.header_items())
        return _FakeResponse(200, b'{"id":"F1"}')

    monkeypatch.setattr(client, "_opener", lambda: _FakeOpener(_fake_open))
    drive = client.DriveClient(lambda: "ya29.SECRET")
    drive.create_folder("Transcriptions")
    assert "ya29.SECRET" not in seen["url"]
    assert "access_token" not in seen["url"]
    assert seen["headers"]["Authorization"] == "Bearer ya29.SECRET"


def test_client_refuses_plain_http():
    drive = client.DriveClient(lambda: "token")
    with pytest.raises(client.DriveError):
        drive._request("GET", "http://www.googleapis.com/drive/v3/files")


def test_drive_never_follows_redirects():
    handler = client._NoRedirectHandler()

    class _Req:
        full_url = "https://www.googleapis.com/drive/v3/files"

    with pytest.raises(client.DriveError):
        handler.redirect_request(_Req(), None, 302, "Found", {}, "https://evil.example")


@pytest.mark.parametrize("link", [
    "http://drive.google.com/file/d/1",           # not https
    "https://evil.example/file",                  # wrong host
    "https://google.com.evil.example/file",       # suffix spoofing
    "javascript:alert(1)",
    "",
    None,
    12345,
])
def test_untrusted_web_view_links_are_rejected(link):
    assert client.validate_web_view_link(link) is None


@pytest.mark.parametrize("link", [
    "https://drive.google.com/file/d/1/view",
    "https://docs.google.com/document/d/1/edit",
    "https://google.com/x",
])
def test_legitimate_web_view_links_are_accepted(link):
    assert client.validate_web_view_link(link) == link


# ===========================
# CREDENTIAL STORE
# ===========================

def test_missing_keyring_disables_the_feature(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def _blocked(name, *args, **kwargs):
        if name == "keyring" or name.startswith("keyring."):
            raise ImportError("no keyring")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    with pytest.raises(store.CredentialStoreUnavailable):
        store._keyring()
    assert store.is_available() is False


def test_null_backend_is_treated_as_unavailable(monkeypatch):
    import keyring
    from keyring.backends import fail as fail_backend
    monkeypatch.setattr(keyring, "get_keyring", lambda: fail_backend.Keyring())
    with pytest.raises(store.CredentialStoreUnavailable):
        store._keyring()
    assert store.is_available() is False


def test_there_is_no_plaintext_fallback():
    # A silent downgrade to a file on disk would be worse than no feature.
    source = Path(store.__file__).read_text(encoding="utf-8")
    assert "open(" not in source
    assert "json.dump" not in source


def test_oversized_token_is_refused(monkeypatch):
    monkeypatch.setattr(store, "_keyring", lambda: _FakeKeyring())
    with pytest.raises(ValueError):
        store.save_refresh_token("SUB1", "x" * (store.MAX_TOKEN_BYTES + 1))


def test_empty_inputs_are_refused(monkeypatch):
    monkeypatch.setattr(store, "_keyring", lambda: _FakeKeyring())
    with pytest.raises(ValueError):
        store.save_refresh_token("", "token")
    with pytest.raises(ValueError):
        store.save_refresh_token("SUB1", "")


def test_round_trip_and_delete(monkeypatch):
    fake = _FakeKeyring()
    monkeypatch.setattr(store, "_keyring", lambda: fake)
    store.save_refresh_token("SUB1", "1//TOKEN")
    assert store.load_refresh_token("SUB1") == "1//TOKEN"
    assert store.delete_refresh_token("SUB1") is True
    assert store.load_refresh_token("SUB1") is None
    assert store.delete_refresh_token("SUB1") is False


def test_service_name_is_stable():
    assert store.SERVICE_NAME == "child-monitor-analyzer:google-oauth"


# ===========================
# SESSION
# ===========================

def _session(tmp_path) -> gdrive.GoogleDriveSession:
    return gdrive.GoogleDriveSession(state_path=tmp_path / "gdrive.json")


def test_state_file_never_holds_a_secret(tmp_path, monkeypatch):
    fake = _FakeKeyring()
    monkeypatch.setattr(store, "_keyring", lambda: fake)
    monkeypatch.setattr(store, "is_available", lambda: True)
    monkeypatch.setattr(auth, "start_link_flow", lambda **kw: auth.GoogleCredentials(
        account_id="SUB1", email="a@b.com", refresh_token="1//SECRETREFRESH",
        access_token="ya29.SECRETACCESS", access_expires_at=time.time() + 3600,
    ))
    session = _session(tmp_path)
    session.link()
    text = (tmp_path / "gdrive.json").read_text(encoding="utf-8")
    assert "SECRETREFRESH" not in text
    assert "SECRETACCESS" not in text
    assert "SUB1" in text and "a@b.com" in text


# ===========================
# CLIENT AUTHENTICATION
# ===========================

def _capture_token_post(monkeypatch, payload: dict):
    """Record the form posted to the token endpoint."""
    sent: dict[str, str] = {}

    def _fake_open(opener, request, timeout=None):
        sent.update(urllib.parse.parse_qsl(request.data.decode("ascii")))
        return _FakeResponse(200, json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr(auth, "_opener", lambda: _FakeOpener(_fake_open))
    return sent


def test_the_client_secret_is_sent_when_configured(monkeypatch):
    """Google rejects the exchange without it, even for PKCE desktop clients."""
    monkeypatch.setattr(auth, "DEFAULT_CLIENT_ID", "1-abc.apps.googleusercontent.com")
    monkeypatch.setattr(auth, "DEFAULT_CLIENT_SECRET", "GOCSPX-secret")
    monkeypatch.delenv(auth.CLIENT_SECRET_ENV_VAR, raising=False)
    sent = _capture_token_post(monkeypatch, {
        "access_token": "ya29.x", "refresh_token": "1//new",
        "expires_in": 3600,
    })

    auth.refresh_access_token(auth.GoogleCredentials(
        account_id="SUB", email="a@b.com", refresh_token="1//old",
        access_token="", access_expires_at=0.0,
    ))

    assert sent["client_secret"] == "GOCSPX-secret"
    assert sent["client_id"] == "1-abc.apps.googleusercontent.com"


def test_no_client_secret_field_when_none_is_configured(monkeypatch):
    """Sending an empty client_secret is its own error; omit the field."""
    monkeypatch.setattr(auth, "DEFAULT_CLIENT_ID", "1-abc.apps.googleusercontent.com")
    monkeypatch.setattr(auth, "DEFAULT_CLIENT_SECRET", "")
    monkeypatch.delenv(auth.CLIENT_SECRET_ENV_VAR, raising=False)
    sent = _capture_token_post(monkeypatch, {
        "access_token": "ya29.x", "refresh_token": "1//new", "expires_in": 3600,
    })

    auth.refresh_access_token(auth.GoogleCredentials(
        account_id="SUB", email="a@b.com", refresh_token="1//old",
        access_token="", access_expires_at=0.0,
    ))

    assert "client_secret" not in sent


def test_the_client_secret_env_var_wins(monkeypatch):
    monkeypatch.setattr(auth, "DEFAULT_CLIENT_SECRET", "GOCSPX-baked")
    monkeypatch.setenv(auth.CLIENT_SECRET_ENV_VAR, "GOCSPX-override")
    assert auth.client_secret() == "GOCSPX-override"


def test_the_client_secret_is_never_logged(caplog, monkeypatch):
    monkeypatch.setattr(auth, "DEFAULT_CLIENT_ID", "1-abc.apps.googleusercontent.com")
    monkeypatch.setattr(auth, "DEFAULT_CLIENT_SECRET", "GOCSPX-topsecret")
    monkeypatch.delenv(auth.CLIENT_SECRET_ENV_VAR, raising=False)
    _capture_token_post(monkeypatch, {
        "access_token": "ya29.x", "refresh_token": "1//new", "expires_in": 3600,
    })

    with caplog.at_level(logging.DEBUG):
        auth.refresh_access_token(auth.GoogleCredentials(
            account_id="SUB", email="a@b.com", refresh_token="1//old",
            access_token="", access_expires_at=0.0,
        ))

    assert "GOCSPX-topsecret" not in caplog.text


def test_the_setup_script_can_still_find_the_secret_constant():
    """setup-google-drive.ps1 -PatchSource rewrites this line by regex."""
    auth_source = (
        Path(__file__).resolve().parents[1]
        / "src" / "monitor" / "gdrive" / "auth.py"
    ).read_text(encoding="utf-8")
    match = re.search(r'(?m)^DEFAULT_CLIENT_SECRET = "(.*)"', auth_source)
    assert match, "DEFAULT_CLIENT_SECRET is no longer a plain string literal"
    assert match.group(1) == "", "a client secret has been committed to source"


# ===========================
# GUI PLUMBING
# ===========================

@pytest.fixture
def window(monkeypatch):
    """A MainWindow on a throwaway QSettings scope.

    Without the override this would read and write the developer's real
    preferences.
    """
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from monitor.gui import main_window as mw

    monkeypatch.setattr(mw, "SETTINGS_ORG", "ChildMonitorAnalyzerTests")
    monkeypatch.setattr(mw, "SETTINGS_APP", f"gdrive-{id(monkeypatch)}")
    # The startup prompts are modal; nothing may schedule one during a test.
    monkeypatch.setattr(mw.QTimer, "singleShot", lambda *a, **k: None)

    app = QApplication.instance() or QApplication([])
    # Retire the previous window's C++ objects before allocating new ones:
    # PySide6 otherwise hands back an invalidated wrapper for a reused address.
    gc.collect()
    app.processEvents()
    win = mw.MainWindow()
    try:
        yield win
    finally:
        win.close()
        app.processEvents()


def _spin(app, predicate, limit: float = 10.0) -> bool:
    """Drain the event loop until *predicate* holds, or *limit* elapses.

    Bounded on purpose: the suite has no timeout plugin, so a test that waits
    on a condition that never comes would wedge the whole run.
    """
    deadline = time.monotonic() + limit
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    app.processEvents()
    return predicate()


def _stalled_worker():
    """A Drive worker that reports back only once it is cancelled."""
    from PySide6.QtCore import QObject, Signal, Slot

    class _Stalled(QObject):
        finished = Signal(object)
        failed = Signal(str)

        def __init__(self) -> None:
            super().__init__()
            self.cancelled = threading.Event()

        def cancel(self) -> None:
            self.cancelled.set()

        @Slot()
        def run(self) -> None:
            self.cancelled.wait(30.0)
            self.finished.emit(None)

    return _Stalled()


def test_the_worker_cancel_reaches_the_auth_layer(tmp_path, monkeypatch):
    """cancel() must interrupt a run() that is already blocked."""
    from PySide6.QtCore import Qt

    from monitor.gui.google_account import LinkWorker

    entered = threading.Event()

    class _BlockingSession:
        def link(self, *, open_browser=None, cancel_event=None):
            entered.set()
            assert cancel_event is not None, "the worker did not pass a cancel token"
            if not cancel_event.wait(10.0):
                raise AssertionError("cancel() never reached the auth layer")
            raise auth.AuthCancelled("The Google sign-in was cancelled.")

    worker = LinkWorker(_BlockingSession())
    messages: list[str] = []
    # Direct, because this test has no event loop to drain a queued emission.
    worker.failed.connect(messages.append, Qt.ConnectionType.DirectConnection)

    runner = threading.Thread(target=worker.run, daemon=True)
    runner.start()
    assert entered.wait(5.0), "the worker never started"
    worker.cancel()
    runner.join(10.0)

    assert not runner.is_alive()
    # Empty message == deliberate cancellation; the UI stays silent.
    assert messages == [""]


def _main_window_source() -> str:
    return (
        Path(__file__).resolve().parents[1]
        / "src" / "monitor" / "gui" / "main_window.py"
    ).read_text(encoding="utf-8")


def _function_source(name: str) -> str:
    tree = ast.parse(_main_window_source())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.unparse(node)
    raise AssertionError(f"{name}() is gone from main_window.py")


def test_the_cancel_button_never_reaches_across_threads():
    """Cancel must not be wired straight to the worker.

    The worker lives on the background thread, so such a connection crosses
    threads, and tearing one down needs both objects' Qt locks. The worker's
    destructor holds its own lock while waiting for the GIL, which the GUI
    thread is holding while it waits for that lock -- a permanent deadlock,
    observed in the field. Routing through a window slot keeps sender and
    receiver on the GUI thread; the actual cancel is a plain call from there.
    """
    tree = ast.parse(_main_window_source())

    connects = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "connect"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "canceled"
    ]
    assert connects, "the sign-in dialog no longer connects a cancel handler"
    for call in connects:
        target = ast.unparse(call.args[0]) if call.args else ""
        assert target == "self._cancel_gdrive_worker", (
            f"canceled.connect() at line {call.lineno} targets {target!r}; it must "
            "go to a GUI-thread slot on the window, never to the worker."
        )


def test_the_progress_dialog_is_silenced_rather_than_disconnected():
    """disconnect() has to lock the far end; blockSignals touches only itself.

    This is the exact line the GUI thread deadlocked on.
    """
    source = _function_source("_close_gdrive_progress")
    assert "disconnect" not in source, (
        "_close_gdrive_progress disconnects again; that is the deadlock."
    )
    assert "blockSignals(True)" in source, (
        "nothing stops close() from emitting canceled()."
    )


def test_the_cancel_slot_reaches_the_running_worker(window):
    """The indirection must not have quietly broken the Cancel button."""
    worker = _stalled_worker()
    window._gdrive_worker = worker
    window._cancel_gdrive_worker()
    assert worker.cancelled.is_set()


def test_cancelling_without_a_worker_is_harmless(window):
    window._gdrive_worker = None
    window._cancel_gdrive_worker()  # must not raise


def test_a_stalled_operation_is_abandoned_and_the_ui_freed(window, monkeypatch):
    """A worker that never reports must not leave the UI waiting forever."""
    from PySide6.QtWidgets import QApplication, QMessageBox

    from monitor.gui import main_window as mw

    warned: list = []
    monkeypatch.setattr(mw, "GDRIVE_WATCHDOG_MS", 50)
    monkeypatch.setattr(
        QMessageBox, "warning", lambda *a, **k: warned.append(a[-1]),
    )

    results: list = []
    worker = _stalled_worker()
    assert window._start_gdrive_worker(worker, results.append, results.append)

    _spin(QApplication.instance(), lambda: window._gdrive_thread is None)

    assert window._gdrive_thread is None, "the watchdog never released the UI"
    assert worker.cancelled.is_set(), "the abandoned worker was not told to stop"
    # The slot is free again, so the user can simply try once more.
    assert window._start_gdrive_worker(_stalled_worker(), results.append,
                                       results.append)
    window._on_gdrive_timeout()
    assert warned, "the user was never told the operation was given up on"


def test_the_worker_reports_to_a_gui_thread_object():
    """A plain function connected to a worker signal runs on the worker thread.

    The handlers open dialogs and touch widgets, so they must be reached
    through a QObject that lives on the GUI thread and lets Qt queue the
    emission across the boundary.
    """
    source = _function_source("_start_gdrive_worker")
    for signal in ("finished", "failed"):
        assert f"worker.{signal}.connect(relay.on_{signal})" in source, (
            f"worker.{signal} no longer reports through the GUI-thread relay."
        )
    assert "lambda" not in source, (
        "a lambda connected to a worker signal would run on the worker thread."
    )


def test_a_result_that_arrives_after_the_watchdog_is_dropped(window, monkeypatch):
    """A late success must not reopen dialogs the user has moved on from."""
    from PySide6.QtWidgets import QApplication, QMessageBox

    from monitor.gui import main_window as mw

    monkeypatch.setattr(mw, "GDRIVE_WATCHDOG_MS", 50)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)

    results: list = []
    worker = _stalled_worker()
    assert window._start_gdrive_worker(worker, results.append, results.append)

    app = QApplication.instance()
    _spin(app, lambda: window._gdrive_thread is None)
    assert worker.cancelled.is_set()

    # cancel() lets the worker fall through to emit finished(); it is too late.
    _spin(app, lambda: bool(results), limit=2.0)
    assert results == [], f"a superseded result was acted on: {results!r}"


def test_a_normal_result_stops_the_watchdog(window, monkeypatch):
    """The timer must not fire over an operation that already succeeded."""
    from PySide6.QtWidgets import QApplication, QMessageBox

    from monitor.gui import main_window as mw

    timed_out: list = []
    monkeypatch.setattr(mw, "GDRIVE_WATCHDOG_MS", 400)
    monkeypatch.setattr(QMessageBox, "warning",
                        lambda *a, **k: timed_out.append(a))

    results: list = []
    worker = _stalled_worker()
    worker.cancel()  # run() returns at once and emits finished
    assert window._start_gdrive_worker(worker, results.append, results.append)

    app = QApplication.instance()
    _spin(app, lambda: bool(results))
    assert results, "the successful result never reached the handler"
    assert not window._gdrive_watchdog.isActive(), "the watchdog is still armed"

    _spin(app, lambda: bool(timed_out), limit=1.0)
    assert timed_out == [], "the watchdog fired over a completed operation"


def test_an_abandoned_thread_does_not_clear_a_newer_one(window, monkeypatch):
    """Two operations in flight must not have their bookkeeping crossed."""
    from PySide6.QtWidgets import QApplication, QMessageBox

    from monitor.gui import main_window as mw

    monkeypatch.setattr(mw, "GDRIVE_WATCHDOG_MS", 50)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)

    app = QApplication.instance()
    stalled = _stalled_worker()
    assert window._start_gdrive_worker(stalled, lambda _r: None, lambda _r: None)
    _spin(app, lambda: window._gdrive_thread is None)

    second = _stalled_worker()
    assert window._start_gdrive_worker(second, lambda _r: None, lambda _r: None)
    live = window._gdrive_thread
    # Let the abandoned thread finish underneath the new one.
    _spin(app, lambda: not window._gdrive_abandoned)
    assert window._gdrive_thread is live, (
        "an abandoned thread cleared the running operation's bookkeeping"
    )
    second.cancel()
    _spin(app, lambda: window._gdrive_thread is None)


def test_shutdown_cancels_before_waiting():
    """quit() alone cannot interrupt the OAuth wait, so the wait would expire."""
    source = _function_source("closeEvent")
    assert "_cancel_gdrive_worker()" in source, (
        "closeEvent waits on the Drive thread without asking it to stop first."
    )


def test_link_fails_closed_without_a_keystore(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "is_available", lambda: False)
    with pytest.raises(auth.AuthError):
        _session(tmp_path).link()


def test_link_without_a_refresh_token_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "_keyring", lambda: _FakeKeyring())
    monkeypatch.setattr(store, "is_available", lambda: True)
    monkeypatch.setattr(auth, "start_link_flow", lambda **kw: auth.GoogleCredentials(
        account_id="SUB1", email="a@b.com", refresh_token="",
    ))
    session = _session(tmp_path)
    with pytest.raises(auth.AuthError):
        session.link()
    assert session.account_id == ""  # nothing persisted on the failure path


def test_unlink_revokes_before_purging(tmp_path, monkeypatch):
    order: list[str] = []
    fake = _FakeKeyring()
    fake.store[(store.SERVICE_NAME, "SUB1")] = "1//TOKEN"
    monkeypatch.setattr(store, "_keyring", lambda: fake)
    monkeypatch.setattr(auth, "revoke", lambda token: order.append("revoke") or True)
    fake.on_delete = lambda: order.append("delete")

    session = _session(tmp_path)
    session._state["account_id"] = "SUB1"
    assert session.unlink() is True
    assert order == ["revoke", "delete"]
    assert session.account_id == ""


def test_dead_grant_is_purged_on_refresh(tmp_path, monkeypatch):
    fake = _FakeKeyring()
    fake.store[(store.SERVICE_NAME, "SUB1")] = "1//TOKEN"
    monkeypatch.setattr(store, "_keyring", lambda: fake)
    monkeypatch.setattr(auth, "refresh_access_token", lambda c: (_ for _ in ()).throw(
        auth.InvalidGrant("revoked")))
    session = _session(tmp_path)
    session._state["account_id"] = "SUB1"
    with pytest.raises(auth.InvalidGrant):
        session._access_token()
    assert session.account_id == ""
    assert (store.SERVICE_NAME, "SUB1") not in fake.store


def test_access_token_is_cached_until_it_nears_expiry(tmp_path, monkeypatch):
    fake = _FakeKeyring()
    fake.store[(store.SERVICE_NAME, "SUB1")] = "1//TOKEN"
    monkeypatch.setattr(store, "_keyring", lambda: fake)
    calls = []

    def _refresh(creds):
        calls.append(1)
        return auth.GoogleCredentials(
            account_id="SUB1", refresh_token="1//TOKEN",
            access_token="ya29.A", access_expires_at=time.time() + 3600,
        )

    monkeypatch.setattr(auth, "refresh_access_token", _refresh)
    session = _session(tmp_path)
    session._state["account_id"] = "SUB1"
    assert session._access_token() == "ya29.A"
    assert session._access_token() == "ya29.A"
    assert len(calls) == 1


def test_folder_id_is_keyed_by_account(tmp_path, monkeypatch):
    session = _session(tmp_path)
    session._state["account_id"] = "SUB1"
    drive = _FakeDrive()
    assert session._folder_id(drive) == "FOLDER-1"
    assert session._folder_id(drive) == "FOLDER-1"      # cached
    assert drive.created == ["Transcriptions"]

    session._state["account_id"] = "SUB2"
    assert session._folder_id(drive) == "FOLDER-2"      # never reuses SUB1's
    assert session._state["folders"] == {"SUB1": "FOLDER-1", "SUB2": "FOLDER-2"}


def test_a_trashed_folder_is_recreated(tmp_path):
    session = _session(tmp_path)
    session._state["account_id"] = "SUB1"
    drive = _FakeDrive()
    session._folder_id(drive)
    drive.exists = False
    assert session._folder_id(drive) == "FOLDER-2"


def test_sidecar_round_trip_holds_no_secrets(tmp_path):
    gdrive.write_sidecar(tmp_path, "thorough", {
        "account_id": "SUB1", "file_id": "F1", "name": "n", "uploaded_at": "t",
    })
    record = gdrive.read_sidecar(tmp_path, "thorough")
    assert record["file_id"] == "F1"
    text = gdrive.sidecar_path(tmp_path, "thorough").read_text(encoding="utf-8")
    assert "token" not in text.lower()


def test_sidecar_path_cannot_escape_the_artifact_folder(tmp_path):
    path = gdrive.sidecar_path(tmp_path, "../../evil")
    assert path.parent == tmp_path


def test_corrupt_sidecar_is_ignored(tmp_path):
    gdrive.sidecar_path(tmp_path, "fast").write_text("{not json", encoding="utf-8")
    assert gdrive.read_sidecar(tmp_path, "fast") == {}


def test_upload_creates_a_new_document(tmp_path, monkeypatch):
    session, drive = _linked_session(tmp_path, monkeypatch)
    result = session.upload_transcript(
        content=b"hello", content_type=client.TXT_MIME,
        display_name="2026-01-01 12-00 rec - תמלול יסודי",
        artifact_dir=tmp_path, model_key="thorough",
    )
    assert result.updated is False
    assert result.file_id == "FILE-1"
    assert drive.created_documents[0]["parent"] == "FOLDER-1"
    assert gdrive.read_sidecar(tmp_path, "thorough")["file_id"] == "FILE-1"


def test_reupload_can_update_the_same_document(tmp_path, monkeypatch):
    session, drive = _linked_session(tmp_path, monkeypatch)
    session.upload_transcript(
        content=b"a", content_type=client.TXT_MIME, display_name="n",
        artifact_dir=tmp_path, model_key="thorough",
    )
    result = session.upload_transcript(
        content=b"b", content_type=client.TXT_MIME, display_name="n",
        artifact_dir=tmp_path, model_key="thorough", replace_existing=True,
    )
    assert result.updated is True
    assert result.file_id == "FILE-1"
    assert len(drive.created_documents) == 1


def test_update_falls_back_to_create_when_the_document_is_gone(tmp_path, monkeypatch):
    session, drive = _linked_session(tmp_path, monkeypatch)
    session.upload_transcript(
        content=b"a", content_type=client.TXT_MIME, display_name="n",
        artifact_dir=tmp_path, model_key="thorough",
    )
    drive.update_raises_not_found = True
    result = session.upload_transcript(
        content=b"b", content_type=client.TXT_MIME, display_name="n",
        artifact_dir=tmp_path, model_key="thorough", replace_existing=True,
    )
    assert result.updated is False
    assert len(drive.created_documents) == 2


def test_a_sidecar_from_another_account_is_not_reused(tmp_path, monkeypatch):
    session, drive = _linked_session(tmp_path, monkeypatch)
    gdrive.write_sidecar(tmp_path, "thorough", {
        "account_id": "OTHER", "file_id": "FOREIGN",
    })
    result = session.upload_transcript(
        content=b"a", content_type=client.TXT_MIME, display_name="n",
        artifact_dir=tmp_path, model_key="thorough", replace_existing=True,
    )
    assert result.updated is False
    assert result.file_id == "FILE-1"


def test_upload_sanitises_the_display_name(tmp_path, monkeypatch):
    session, drive = _linked_session(tmp_path, monkeypatch)
    result = session.upload_transcript(
        content=b"a", content_type=client.TXT_MIME,
        display_name="evil\u202egpj.exe/../x",
        artifact_dir=None, model_key="none",
    )
    assert "\u202e" not in result.name
    assert "/" not in result.name


def test_an_untrusted_web_view_link_is_discarded(tmp_path, monkeypatch):
    session, drive = _linked_session(tmp_path, monkeypatch)
    drive.web_view_link = "https://evil.example/steal"
    result = session.upload_transcript(
        content=b"a", content_type=client.TXT_MIME, display_name="n",
        artifact_dir=None, model_key="none",
    )
    assert result.web_view_link is None


# ===========================
# SELF-REVIEW REGRESSIONS
# ===========================

# ===========================
# CANCELLATION & OUTCOME SEMANTICS
# ===========================

def _configured(monkeypatch):
    monkeypatch.setattr(auth, "DEFAULT_CLIENT_ID", "1-abc.apps.googleusercontent.com")
    # Google's token endpoint needs both halves, so is_configured() does too.
    monkeypatch.setattr(auth, "DEFAULT_CLIENT_SECRET", "GOCSPX-test")
    monkeypatch.delenv(auth.CLIENT_ID_ENV_VAR, raising=False)
    monkeypatch.delenv(auth.CLIENT_SECRET_ENV_VAR, raising=False)


def test_cancelling_aborts_the_wait_promptly(monkeypatch):
    """Without this the app looks hung for the whole consent timeout."""
    _configured(monkeypatch)
    cancel = threading.Event()

    def open_browser(url):
        cancel.set()

    started = time.monotonic()
    with pytest.raises(auth.AuthCancelled) as excinfo:
        auth.start_link_flow(
            open_browser=open_browser, timeout=30.0, cancel_event=cancel,
        )
    elapsed = time.monotonic() - started

    assert elapsed < 5.0, f"cancel took {elapsed:.1f}s; it must be near-immediate"
    assert not isinstance(excinfo.value, auth.AuthTimeout)


def test_cancelling_closes_the_loopback_port(monkeypatch):
    """A cancelled flow must not leave a listening socket behind."""
    _configured(monkeypatch)
    cancel = threading.Event()
    seen = {}

    def open_browser(url):
        seen["port"] = int(urllib.parse.urlsplit(
            urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)["redirect_uri"][0]
        ).port)
        cancel.set()

    with pytest.raises(auth.AuthCancelled):
        auth.start_link_flow(
            open_browser=open_browser, timeout=30.0, cancel_event=cancel,
        )

    import socket as _socket
    probe = _socket.socket()
    probe.settimeout(2)
    try:
        connected = probe.connect_ex(("127.0.0.1", seen["port"])) == 0
    finally:
        probe.close()
    assert not connected, "the loopback listener survived a cancelled flow"


def test_a_timeout_is_not_reported_as_a_cancel(monkeypatch):
    _configured(monkeypatch)
    with pytest.raises(auth.AuthTimeout):
        auth.start_link_flow(open_browser=lambda url: None, timeout=0.05)


def test_timeout_is_still_a_cancellation():
    """Existing handlers catching AuthCancelled must keep working."""
    assert issubclass(auth.AuthTimeout, auth.AuthCancelled)


def test_declining_consent_is_distinct_from_a_timeout(monkeypatch):
    """A user who clicks Deny must not be told the flow timed out."""
    _configured(monkeypatch)

    def open_browser(url):
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
        port = urllib.parse.urlsplit(query["redirect_uri"][0]).port
        _get(port, f"/oauth2/callback?state={query['state'][0]}&error=access_denied")

    with pytest.raises(auth.AuthDenied):
        auth.start_link_flow(open_browser=open_browser, timeout=10.0)


def test_denial_is_not_swallowed_as_a_cancellation():
    """AuthDenied must surface its own message, so it is not an AuthCancelled."""
    assert not issubclass(auth.AuthDenied, auth.AuthCancelled)


def test_a_provider_error_is_not_reported_as_denial(monkeypatch):
    """Only access_denied is a denial; anything else is a real failure."""
    _configured(monkeypatch)

    def open_browser(url):
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
        port = urllib.parse.urlsplit(query["redirect_uri"][0]).port
        _get(port, f"/oauth2/callback?state={query['state'][0]}&error=server_error")

    with pytest.raises(auth.AuthError) as excinfo:
        auth.start_link_flow(open_browser=open_browser, timeout=10.0)
    assert not isinstance(excinfo.value, auth.AuthDenied)
    assert not isinstance(excinfo.value, auth.AuthCancelled)


def test_cancelling_persists_nothing(tmp_path, monkeypatch):
    """A cancelled sign-in must leave the session exactly as it was."""
    _configured(monkeypatch)
    fake = _FakeKeyring()
    monkeypatch.setattr(store, "_keyring", lambda: fake)
    monkeypatch.setattr(store, "is_available", lambda: True)
    cancel = threading.Event()
    cancel.set()

    session = _session(tmp_path)
    with pytest.raises(auth.AuthCancelled):
        session.link(open_browser=lambda url: None, cancel_event=cancel)

    assert session.account_id == ""
    assert session.is_linked() is False
    assert fake.store == {}


def test_the_callback_result_is_write_once():
    result = auth._CallbackResult()
    assert result.resolve(code="FIRST") is True
    assert result.resolve(code="SECOND") is False
    assert result.resolve(error="late error") is False
    assert result.code == "FIRST"
    assert result.error is None


def test_a_second_callback_is_rejected(listener):
    """The listener stays up until shutdown, so replays must be refused."""
    _server, result, port = listener
    first, _b, _h = _get(port, "/oauth2/callback?state=EXPECTED_STATE&code=FIRST")
    second, _b2, _h2 = _get(port, "/oauth2/callback?state=EXPECTED_STATE&code=SECOND")
    assert first == 200
    assert second == 400
    assert result.code == "FIRST"


def test_an_empty_expected_audience_never_validates(monkeypatch):
    """An unconfigured client id must not make ``compare_digest`` succeed."""
    token = _id_token({"iss": "https://accounts.google.com", "aud": "",
                       "sub": "SUB1", "exp": time.time() + 600})
    with pytest.raises(auth.AuthError):
        auth.decode_id_token_claims(token, expected_aud="")


def test_a_transport_failure_is_reported_as_offline(monkeypatch):
    """No internet must be distinguishable from an API rejection."""
    def boom(opener, request, timeout=None):
        raise urllib.error.URLError("getaddrinfo failed")

    monkeypatch.setattr(client, "_opener", lambda: _FakeOpener(boom))
    monkeypatch.setattr(client, "_backoff_delay", lambda attempt: 0.0)
    drive = client.DriveClient(lambda: "ya29.A")
    with pytest.raises(client.DriveOffline):
        drive.folder_exists("FOLDER-1")


def test_offline_is_still_a_drive_error():
    """Existing handlers that catch DriveError must keep working."""
    assert issubclass(client.DriveOffline, client.DriveError)


def test_linked_state_is_not_reread_from_the_vault_every_time(tmp_path, monkeypatch):
    """Each miss pulls the refresh token into memory, so it must be cached."""
    fake = _FakeKeyring()
    fake.store[(store.SERVICE_NAME, "SUB1")] = "1//TOKEN"
    monkeypatch.setattr(store, "_keyring", lambda: fake)
    session = _session(tmp_path)
    session._state["account_id"] = "SUB1"

    assert session.is_linked() is True
    reads_after_first = fake.reads
    for _ in range(10):
        assert session.is_linked() is True
    assert fake.reads == reads_after_first


def test_unlinking_invalidates_the_linked_cache(tmp_path, monkeypatch):
    session, _drive = _linked_session(tmp_path, monkeypatch)
    monkeypatch.setattr(auth, "revoke", lambda token: True)
    assert session.is_linked() is True
    session.unlink()
    assert session.is_linked() is False


def test_a_dead_grant_invalidates_the_linked_cache(tmp_path, monkeypatch):
    session, drive = _linked_session(tmp_path, monkeypatch)
    assert session.is_linked() is True

    def dead(credentials):
        raise auth.InvalidGrant("invalid_grant")

    monkeypatch.setattr(auth, "refresh_access_token", dead)
    with pytest.raises(auth.InvalidGrant):
        session._access_token()
    assert session.is_linked() is False


# ===========================
# CHIP INDICATOR
# ===========================

def _image(icon, size=64):
    from PySide6.QtCore import QSize

    return icon.pixmap(QSize(size, size)).toImage()


def _cloud_pixels(size=64):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from monitor.gui.player_icons import icon_cloud_upload

    QApplication.instance() or QApplication([])
    return _image(icon_cloud_upload(), size)


def _drive_icon():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from monitor.gui.player_icons import icon_google_drive

    QApplication.instance() or QApplication([])
    return icon_google_drive()


def test_the_cloud_lobes_do_not_cancel_each_other_out():
    """The lobes overlap, so an odd-even fill would punch holes in the cloud.

    Samples inside the left lobe's overlap with the top one, on the 32-unit
    design grid scaled to 64 px, and clear of the punched-out arrow.
    """
    image = _cloud_pixels()
    assert image.pixelColor(20, 28).alpha() == 255   # (10, 14) on the grid
    assert image.pixelColor(44, 28).alpha() == 255   # (22, 14), right lobe


def test_the_arrow_is_punched_through():
    """Grid (16, 20): the middle of the arrow shaft."""
    assert _cloud_pixels().pixelColor(32, 40).alpha() == 0


def test_the_glyph_fills_its_canvas():
    """The chip carries no text, so the cloud gets the whole button."""
    image = _cloud_pixels()
    opaque = [(x, y) for y in range(64) for x in range(64)
              if image.pixelColor(x, y).alpha() > 0]
    xs = [x for x, _ in opaque]
    ys = [y for _, y in opaque]
    assert min(xs) <= 1 and max(xs) >= 62, "the cloud must span the full width"
    assert max(ys) - min(ys) >= 44, "and most of the height"


def test_the_signed_out_glyph_is_not_a_google_mark():
    """The signed-out state is muted, which the guidelines forbid for a mark.

    So the glyph used there has to stay generic, drawn by us.
    """
    from monitor.gui import player_icons

    source = Path(player_icons.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    body = next(
        ast.get_source_segment(source, node)
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_draw_cloud_upload"
    )
    assert "drawText" not in body, "no lettering, and certainly no 'G'"
    assert "google_drive" not in body, "and no trace of the real mark"


# ===========================
# THE GOOGLE DRIVE MARK
# ===========================

#: The asset as published. A mismatch means it was edited or swapped, either of
#: which would breach the "reproduce verbatim" condition recorded in NOTICE.txt.
DRIVE_SVG_SHA256 = (
    "60a6496d705d8bf010ce126469be89889db331622b14d8ae1645a38c2aaa23f5"
)


def _drive_svg_path():
    from monitor.gui import player_icons

    return Path(player_icons.__file__).resolve().parent / "assets" / "google_drive.svg"


def test_the_shipped_mark_is_byte_for_byte_as_published():
    raw = _drive_svg_path().read_bytes()
    assert hashlib.sha256(raw).hexdigest() == DRIVE_SVG_SHA256


def test_the_mark_travels_with_its_trademark_notice():
    notice = _drive_svg_path().with_name("NOTICE.txt").read_text(encoding="utf-8")
    assert "trademark of Google Inc" in notice, "Google prescribe this wording"
    assert "not affiliated" in notice
    assert DRIVE_SVG_SHA256 in notice.lower(), "the notice must pin the same file"


def test_the_readme_attributes_the_trademark():
    """The guidelines ask for attribution in the app's title or description."""
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(
        encoding="utf-8")
    assert "Google Drive is a trademark of Google Inc" in readme


def test_every_mention_of_drive_uses_its_full_name():
    """"Don't abbreviate the term 'Google Drive'", say the guidelines.

    Catches a bare "Drive", and the Hebrew transliteration "דרייב", standing in
    for the product. "Google account" is left alone: that is Google's own term.
    Reads the table rather than the source, so strings split over several lines
    are checked too.
    """
    from monitor.gui.strings import _STRINGS

    for (key, lang), text in _STRINGS.items():
        where = f"{key} ({lang})"
        assert not re.search(r"(?<!Google )Drive", text), (
            f"abbreviated product name in {where}: {text!r}"
        )
        assert "דרייב" not in text, (
            f"transliterated product name in {where}: {text!r}"
        )


def test_the_mark_is_self_contained():
    """A logo that fetched a remote resource would phone home on every repaint."""
    markup = _drive_svg_path().read_text(encoding="utf-8")
    assert "http://" not in markup.replace("http://www.w3.org/2000/svg", "")
    assert "https://" not in markup
    assert "<script" not in markup and "<image" not in markup


def test_the_mark_renders_at_full_strength():
    """Qt masks by luminance, so the file's dark mask fill dims the whole logo.

    Without the workaround every pixel comes back at roughly a third alpha,
    which is a washed-out mark: an alteration, and an ugly one.
    """
    image = _image(_drive_icon())
    for x, y in ((32, 18), (22, 44), (44, 44)):   # green, blue and yellow lobes
        assert image.pixelColor(x, y).alpha() == 255


def test_the_mark_keeps_its_colours():
    image = _image(_drive_icon())
    green = image.pixelColor(32, 18)
    blue = image.pixelColor(22, 44)
    yellow = image.pixelColor(44, 44)
    assert green.green() > green.red() and green.green() > green.blue()
    assert blue.blue() > blue.red() and blue.blue() > blue.green()
    assert yellow.red() > 200 and yellow.green() > 150 and yellow.blue() < 80


def test_qt_can_never_grey_the_mark():
    """Qt synthesises a washed-out pixmap for disabled buttons otherwise."""
    from PySide6.QtGui import QIcon
    from PySide6.QtCore import QSize

    icon = _drive_icon()
    normal = icon.pixmap(QSize(64, 64), QIcon.Mode.Normal).toImage()
    disabled = icon.pixmap(QSize(64, 64), QIcon.Mode.Disabled).toImage()
    assert disabled == normal


def test_the_mark_is_shown_only_while_an_account_is_connected():
    """Linked uses the mark; every other state uses the neutral cloud.

    Reads the source rather than building a window, because the states depend
    on a keyring, a session and a running worker thread.
    """
    from monitor.gui import main_window

    source = Path(main_window.__file__).read_text(encoding="utf-8")
    body = source.split("def _refresh_google_chip")[1].split("\n    def ")[0]
    assert "icon_google_drive() if linked and not busy else icon_cloud_upload()" in body


def test_the_button_carrying_the_mark_names_its_action():
    """The guidelines require a tooltip saying what the button does with Drive."""
    from monitor.gui import main_window
    from monitor.gui.strings import Lang, S, _STRINGS

    source = Path(main_window.__file__).read_text(encoding="utf-8")
    body = source.split("def _refresh_google_chip")[1].split("\n    def ")[0]
    linked_branch = body.split("elif linked:")[1]
    assert "S.GOOGLE_CHIP_ACTION" in linked_branch
    for lang in Lang:
        assert "Google Drive" in _STRINGS[(S.GOOGLE_CHIP_ACTION, lang)]


def test_the_frozen_build_ships_the_mark_and_the_notice():
    spec = (Path(__file__).resolve().parents[1] / "monitor-gui.spec").read_text(
        encoding="utf-8")
    assert '"monitor/gui/assets"' in spec, "the asset folder must be bundled"
    assert '"PySide6.QtSvg"' in spec, "and the renderer that reads it"


def test_an_installed_wheel_ships_the_mark_and_the_notice():
    pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(
        encoding="utf-8")
    assert 'assets/*.svg' in pyproject
    assert 'assets/NOTICE.txt' in pyproject


# ===========================
# HELPERS
# ===========================

class _BytesIO:
    """Minimal file-like object for HTTPError bodies."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self, size=-1) -> bytes:
        data, self._data = self._data, b""
        return data

    def close(self) -> None:
        self._data = b""


class _FakeResponse:
    def __init__(self, status: int, body: bytes) -> None:
        self.status = status
        self._body = body

    def read(self, size=-1) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeOpener:
    def __init__(self, open_func) -> None:
        self._open = open_func

    def open(self, request, timeout=None):
        return self._open(self, request, timeout)


class _FakeKeyring:
    """In-memory stand-in for the Windows Credential Manager."""

    def __init__(self) -> None:
        self.store: dict[tuple[str, str], str] = {}
        self.reads = 0
        self.on_delete = lambda: None

    def set_password(self, service, username, password):
        self.store[(service, username)] = password

    def get_password(self, service, username):
        self.reads += 1
        return self.store.get((service, username))

    def delete_password(self, service, username):
        if (service, username) not in self.store:
            raise KeyError("no such credential")
        self.on_delete()
        del self.store[(service, username)]


class _FakeDrive:
    """Stand-in for DriveClient with no network."""

    def __init__(self) -> None:
        self.created: list[str] = []
        self.created_documents: list[dict] = []
        self.updated_documents: list[dict] = []
        self.exists = True
        self.update_raises_not_found = False
        self.web_view_link = "https://docs.google.com/document/d/FILE-1/edit"
        self._counter = 0

    def folder_exists(self, folder_id):
        return self.exists

    def create_folder(self, name, *, parent=None):
        self.created.append(name)
        self._counter += 1
        return f"FOLDER-{self._counter}"

    def create_document(self, *, name, content, content_type, parent):
        self.created_documents.append({
            "name": name, "content": content, "parent": parent,
        })
        return {"id": "FILE-1", "webViewLink": self.web_view_link, "name": name}

    def update_document(self, file_id, *, name, content, content_type):
        if self.update_raises_not_found:
            raise client.DriveNotFound("gone", status=404)
        self.updated_documents.append({"id": file_id, "name": name})
        return {"id": file_id, "webViewLink": self.web_view_link, "name": name}


def _linked_session(tmp_path, monkeypatch):
    """Return a session with a linked account and a stubbed Drive client."""
    fake_keyring = _FakeKeyring()
    fake_keyring.store[(store.SERVICE_NAME, "SUB1")] = "1//TOKEN"
    monkeypatch.setattr(store, "_keyring", lambda: fake_keyring)
    monkeypatch.setattr(auth, "refresh_access_token", lambda c: auth.GoogleCredentials(
        account_id="SUB1", refresh_token="1//TOKEN", access_token="ya29.A",
        access_expires_at=time.time() + 3600,
    ))
    drive = _FakeDrive()
    monkeypatch.setattr(client, "DriveClient", lambda provider: drive)
    session = _session(tmp_path)
    session._state["account_id"] = "SUB1"
    session._state["email"] = "a@b.com"
    return session, drive


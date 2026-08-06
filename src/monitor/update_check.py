"""
Detect a newer released version of the application.

**Metadata only.** This module never downloads, verifies or executes a payload;
it compares version strings and nothing else. That is a deliberate boundary --
an updater that installs code is an remote-code-execution channel and would
need signed, expiring metadata to be safe. Reporting that a newer version
exists needs none of that, so the threat model stays small:

* The releases page URL is a **constant**. A URL taken from the response could
  be ``file://``, a UNC path (which leaks an NTLM hash on Windows) or any
  registered URI handler, because ``webbrowser.open`` ends up at ShellExecute.
  :func:`releases_url` is the only way to obtain it, and it is validated even
  though it is hard-coded, so a careless edit cannot widen it.
* The response is size-capped, so a hostile endpoint cannot exhaust memory.
* Redirects are refused, and only ``https`` is accepted.
* Only a *strictly greater* version is reported, so a rolled-back or forged
  "latest" release cannot advertise a downgrade.

An attacker who can block the network can suppress notifications indefinitely
(a freeze attack). That is unpreventable without signed, expiring metadata and
is low severity here, because nothing installs automatically.
"""

from __future__ import annotations

import json
import logging
import re
import ssl
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional

from . import __version__

log = logging.getLogger(__name__)

GITHUB_OWNER = "ohadhawk"
GITHUB_REPO = "child-monitor-analyzer"

LATEST_RELEASE_API_URL = (
    f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/releases/latest"
)

#: Never read from a network response. See the module docstring.
RELEASES_PAGE_URL = f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/releases"

#: Release metadata is a few kilobytes; this bounds a hostile response.
MAX_RESPONSE_BYTES = 128 * 1024

HTTP_TIMEOUT_SECONDS = 10

#: GitHub allows 60 unauthenticated requests an hour per IP. Checking once a
#: day stays far inside that and limits how often the app phones home.
CHECK_INTERVAL_SECONDS = 24 * 60 * 60

NEVER = "never"
DAILY = "daily"
WEEKLY = "weekly"
MONTHLY = "monthly"

#: Menu order. ``never`` has no interval, which is how it disables a check.
CHECK_INTERVALS: dict[str, Optional[float]] = {
    NEVER: None,
    DAILY: CHECK_INTERVAL_SECONDS,
    WEEKLY: 7 * CHECK_INTERVAL_SECONDS,
    MONTHLY: 30 * CHECK_INTERVAL_SECONDS,
}

_USER_AGENT = f"child-monitor-analyzer/{__version__}"

_VERSION_RE = re.compile(r"^v?(\d+(?:\.\d+)*)")


class UpdateCheckError(RuntimeError):
    """The check could not be completed. Never shown unless the user asked."""


class UpdateRateLimited(UpdateCheckError):
    """GitHub is throttling this IP.

    The anonymous limit is 60 requests an hour *per address*, so anyone behind
    a shared or NAT'd connection can be blocked by other people's traffic. It
    says nothing about the app, hence a message of its own.
    """


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """A redirect would move us off the pinned API host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise UpdateCheckError(f"Unexpected redirect to {newurl}")


def _opener() -> urllib.request.OpenerDirector:
    context = ssl.create_default_context()
    return urllib.request.build_opener(
        urllib.request.HTTPSHandler(context=context), _NoRedirectHandler(),
    )


def parse_version(text: object) -> Optional[tuple[int, ...]]:
    """Return *text* as a comparable tuple, or None if it is not a version.

    Accepts an optional ``v`` prefix and ignores any pre-release suffix, so
    ``"v1.2.3-beta"`` parses as ``(1, 2, 3)``.
    """
    if not isinstance(text, str):
        return None
    match = _VERSION_RE.match(text.strip())
    if not match:
        return None
    try:
        return tuple(int(part) for part in match.group(1).split("."))
    except ValueError:  # pragma: no cover - the regex already excludes this
        return None


def is_newer(candidate: object, current: object) -> bool:
    """Return True only if *candidate* is strictly newer than *current*.

    Unparseable input is treated as "no update" rather than an error: a
    malformed tag must never be able to advertise a downgrade, and must never
    interrupt the user.
    """
    new = parse_version(candidate)
    old = parse_version(current)
    if new is None or old is None:
        return False
    # Pad so 1.2 and 1.2.0 compare equal rather than by length.
    width = max(len(new), len(old))
    return new + (0,) * (width - len(new)) > old + (0,) * (width - len(old))


def is_due(last_checked: float, now: float,
           interval: float = CHECK_INTERVAL_SECONDS) -> bool:
    """Return True if enough time has passed since *last_checked*.

    A *last_checked* in the future (a clock that has since been corrected
    backwards) counts as due, otherwise checking could stall for years.
    """
    if last_checked <= 0 or last_checked > now:
        return True
    return (now - last_checked) >= interval


def interval_for(frequency: object) -> Optional[float]:
    """Return the interval for *frequency*, or None if it must not run.

    An unrecognised value -- a hand-edited setting, or one written by a newer
    version -- means "never", so a bad value can never cause more traffic.
    """
    if not isinstance(frequency, str):
        return None
    return CHECK_INTERVALS.get(frequency.strip().lower())


def releases_url() -> str:
    """Return the releases page URL, validated before it reaches a browser."""
    validated = validate_release_url(RELEASES_PAGE_URL)
    if validated is None:  # pragma: no cover - guards against a careless edit
        raise UpdateCheckError("The releases URL constant is not safe to open.")
    return validated


def validate_release_url(url: object) -> Optional[str]:
    """Return *url* only if it is an https GitHub URL."""
    if not isinstance(url, str) or not url:
        return None
    try:
        parsed = urllib.parse.urlsplit(url)
    except ValueError:
        return None
    if parsed.scheme != "https":
        return None
    host = (parsed.hostname or "").lower()
    if host != "github.com" and not host.endswith(".github.com"):
        return None
    return url


def _is_rate_limited(exc: urllib.error.HTTPError) -> bool:
    """Whether a 403/429 is GitHub's quota rather than a real refusal."""
    headers = getattr(exc, "headers", None)
    if headers is None:
        return False
    return str(headers.get("x-ratelimit-remaining", "")).strip() == "0"


def fetch_latest_version() -> Optional[str]:
    """Return the newest published release tag, or None if there is none.

    Raises:
        UpdateCheckError: The endpoint could not be reached or understood.
    """
    request = urllib.request.Request(LATEST_RELEASE_API_URL, headers={
        "Accept": "application/vnd.github+json",
        # GitHub rejects requests without one; it carries no machine identity.
        "User-Agent": _USER_AGENT,
    })
    try:
        with _opener().open(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None  # No release published yet: not an error.
        if exc.code in (403, 429) and _is_rate_limited(exc):
            raise UpdateRateLimited("GitHub is rate-limiting this address.") from exc
        raise UpdateCheckError(f"GitHub returned {exc.code}.") from exc
    except (urllib.error.URLError, OSError, ssl.SSLError) as exc:
        raise UpdateCheckError(f"Could not reach GitHub: {exc}") from exc

    if len(raw) > MAX_RESPONSE_BYTES:
        raise UpdateCheckError("The release metadata was unexpectedly large.")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise UpdateCheckError("The release metadata was not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise UpdateCheckError("The release metadata had an unexpected shape.")
    if payload.get("draft") or payload.get("prerelease"):
        return None

    tag = payload.get("tag_name")
    return tag if isinstance(tag, str) else None


def check_for_update(current: str = __version__) -> Optional[str]:
    """Return the newer version, normalised, or None if *current* is up to date.

    The returned string is rebuilt from the parsed numbers rather than passed
    through from the response, so only digits and dots can ever reach the UI.
    ``tag_name`` is matched by prefix, so the raw value could otherwise carry a
    megabyte of trailing junk into a dialog.

    Raises:
        UpdateCheckError: The check could not be completed.
    """
    latest = fetch_latest_version()
    if latest is None:
        return None
    if not is_newer(latest, current):
        log.info("Update check: %s is current (latest is %.40s).", current, latest)
        return None
    normalised = ".".join(str(part) for part in parse_version(latest))
    log.info("Update check: %s is available (running %s).", normalised, current)
    return normalised

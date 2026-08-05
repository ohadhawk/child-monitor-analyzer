"""
Redaction of secrets in log records.

Users attach ``~/.child-monitor-analyzer/logs/`` files to bug reports, so any
OAuth material that reaches a log record has effectively been disclosed. This
filter is the last line of defence: the ``monitor.gdrive`` code is written not
to log secrets in the first place, but third-party libraries and tracebacks
are not under our control.

Install once, as early as possible::

    from monitor.log_redaction import install_redaction
    install_redaction()
"""

from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)

REDACTED = "[REDACTED]"

# Ordered most-specific first. Each pattern keeps the identifying prefix so a
# redacted log is still diagnosable ("we had a refresh token here").
_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # Google access tokens and refresh tokens have well-known shapes.
    (re.compile(r"ya29\.[A-Za-z0-9._\-]+"), f"ya29.{REDACTED}"),
    (re.compile(r"\b1//[A-Za-z0-9._\-]+"), f"1//{REDACTED}"),
    # JWTs (id_token) - three base64url segments.
    (re.compile(r"\beyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+"),
     f"eyJ{REDACTED}"),
    # Greedy to end of line: a header value has no reliable terminator, and
    # over-redacting a log line is always preferable to leaking a token.
    (re.compile(r"(?i)\b(authorization|proxy-authorization)\s*:\s*[^\r\n]+"),
     rf"\1: {REDACTED}"),
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-~+/]+=*"), f"Bearer {REDACTED}"),
    # Query-string and form-encoded parameters.
    (re.compile(
        r"(?i)\b(code|code_verifier|access_token|refresh_token|id_token|"
        r"client_secret|state|assertion)=[^\s&\"']+"),
     rf"\1={REDACTED}"),
    # JSON fields.
    (re.compile(
        r'(?i)"(code|code_verifier|access_token|refresh_token|id_token|'
        r'client_secret|state)"\s*:\s*"[^"]*"'),
     rf'"\1": "{REDACTED}"'),
)


def redact(text: str) -> str:
    """Return *text* with known secret shapes replaced."""
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    return text


class SecretRedactingFilter(logging.Filter):
    """Rewrites log records in place so secrets never reach a handler.

    Attached to the ``monitor`` logger, it also covers propagated records from
    child loggers. It rewrites ``record.msg`` and ``record.args`` rather than
    the formatted output so that every handler benefits, including ones added
    later.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            if isinstance(record.msg, str):
                record.msg = redact(record.msg)
            if record.args:
                if isinstance(record.args, dict):
                    record.args = {
                        key: redact(value) if isinstance(value, str) else value
                        for key, value in record.args.items()
                    }
                elif isinstance(record.args, tuple):
                    record.args = tuple(
                        redact(value) if isinstance(value, str) else value
                        for value in record.args
                    )
            if record.exc_text:
                record.exc_text = redact(record.exc_text)
        except Exception:  # noqa: BLE001 - logging must never raise
            return True
        return True


def install_redaction(logger_name: str = "monitor") -> SecretRedactingFilter:
    """Attach the redaction filter to *logger_name* and its handlers.

    Idempotent, and safe to call again after new handlers are added — which is
    required, because a logger's filters are *not* applied to records that
    propagate up from child loggers; only the handlers' filters are.

    Returns:
        The installed (or already-installed) filter instance.
    """
    target = logging.getLogger(logger_name)
    redactor = next(
        (f for f in target.filters if isinstance(f, SecretRedactingFilter)), None,
    )
    if redactor is None:
        redactor = SecretRedactingFilter()
        target.addFilter(redactor)
    for handler in target.handlers:
        if not any(isinstance(f, SecretRedactingFilter) for f in handler.filters):
            handler.addFilter(redactor)
    return redactor

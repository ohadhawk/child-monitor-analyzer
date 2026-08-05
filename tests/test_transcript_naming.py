"""
Regression tests for transcript naming and secret redaction.

Both are security controls:

* ``sanitize_display_name`` is the choke point for an untrusted audio filename
  that flows into a filesystem path *and* into a Drive API request body.
* ``SecretRedactingFilter`` is the last line of defence against an OAuth secret
  reaching a log file that a user then attaches to a bug report.
"""

from __future__ import annotations

import logging
import os

import pytest

from monitor.gdrive.naming import (
    FALLBACK_NAME,
    MAX_STEM_CHARS,
    build_transcript_base_name,
    sanitize_display_name,
)
from monitor.log_redaction import REDACTED, SecretRedactingFilter, install_redaction, redact


# ===========================
# sanitize_display_name
# ===========================

def test_plain_hebrew_name_is_unchanged():
    assert sanitize_display_name("אריאל 1") == "אריאל 1"


@pytest.mark.parametrize("control", [
    "\u202e",  # RTLO - the classic filename-spoofing character
    "\u202d", "\u202a", "\u202b", "\u202c",
    "\u2066", "\u2067", "\u2068", "\u2069",
    "\u200e", "\u200f",
])
def test_bidi_controls_are_stripped(control):
    result = sanitize_display_name(f"invoice{control}fdp.exe")
    assert control not in result


@pytest.mark.parametrize("char", list('/\\:*?"<>|'))
def test_windows_illegal_characters_are_stripped(char):
    assert char not in sanitize_display_name(f"rec{char}ording")


def test_ntfs_alternate_data_stream_separator_is_removed():
    # "name:stream" would create a hidden ADS rather than a normal file.
    assert ":" not in sanitize_display_name("report:hidden")


def test_path_traversal_cannot_survive():
    result = sanitize_display_name("../../windows/system32/evil")
    assert "/" not in result and "\\" not in result


def test_control_characters_and_newlines_are_stripped():
    result = sanitize_display_name("line1\r\nline2\x00\x07end")
    assert "\n" not in result and "\r" not in result and "\x00" not in result


@pytest.mark.parametrize("reserved", ["CON", "con", "PRN", "NUL", "COM1", "LPT9"])
def test_windows_reserved_device_names_are_prefixed(reserved):
    assert sanitize_display_name(reserved).startswith("_")


def test_reserved_name_with_extension_is_also_prefixed():
    assert sanitize_display_name("CON.txt").startswith("_")


def test_name_that_merely_starts_with_a_reserved_word_is_untouched():
    assert sanitize_display_name("CONcert recording") == "CONcert recording"


def test_over_length_names_are_truncated():
    result = sanitize_display_name("א" * 500)
    assert len(result) <= MAX_STEM_CHARS


def test_empty_and_whitespace_names_fall_back():
    assert sanitize_display_name("") == FALLBACK_NAME
    assert sanitize_display_name("   ") == FALLBACK_NAME
    assert sanitize_display_name("...") == FALLBACK_NAME


def test_name_of_only_illegal_characters_falls_back():
    assert sanitize_display_name('<<>>::||') == FALLBACK_NAME


def test_trailing_dots_and_spaces_are_removed():
    # Windows silently drops these, desynchronising the intended and real name.
    result = sanitize_display_name("recording 1 . ")
    assert not result.endswith((" ", "."))


def test_result_is_never_empty_after_truncation():
    assert sanitize_display_name("." * 200) == FALLBACK_NAME


# ===========================
# build_transcript_base_name
# ===========================

def _make_audio(tmp_path, name="אריאל 1.wav", mtime=1_700_000_000):
    audio = tmp_path / name
    audio.write_bytes(b"x")
    os.utime(audio, (mtime, mtime))
    return audio


def test_full_name_is_stem_then_label(tmp_path):
    audio = _make_audio(tmp_path)
    name = build_transcript_base_name(audio, "thorough", model_label="תמלול יסודי")
    assert name == "אריאל 1 - תמלול יסודי"


def test_no_date_is_prepended(tmp_path):
    # The recording's mtime reflects when the file was last copied, not
    # recorded, so it is not a trustworthy date and must not appear here.
    audio = _make_audio(tmp_path)
    name = build_transcript_base_name(audio, "fast", model_label="תמלול מהיר")
    assert name.startswith("אריאל 1")


def test_model_key_none_omits_the_label_and_its_separator(tmp_path):
    audio = _make_audio(tmp_path)
    name = build_transcript_base_name(audio, "none", model_label="תמלול יסודי")
    assert " - " not in name
    assert name.endswith("אריאל 1")


def test_missing_label_omits_the_suffix(tmp_path):
    audio = _make_audio(tmp_path)
    name = build_transcript_base_name(audio, "thorough")
    assert " - " not in name


def test_unreadable_mtime_does_not_raise(tmp_path):
    missing = tmp_path / "nested" / "gone.wav"
    name = build_transcript_base_name(missing, "thorough", model_label="תמלול יסודי")
    assert name == "gone - תמלול יסודי"


def test_no_audio_path_falls_back(tmp_path):
    assert build_transcript_base_name(None, "thorough") == FALLBACK_NAME


def test_rtlo_in_the_audio_stem_does_not_reach_the_name(tmp_path):
    audio = _make_audio(tmp_path, name="rec\u202egpj.exe.wav")
    name = build_transcript_base_name(audio, "none")
    assert "\u202e" not in name


def test_a_malicious_label_cannot_inject_a_separator(tmp_path):
    audio = _make_audio(tmp_path)
    name = build_transcript_base_name(audio, "fast", model_label="a/../b")
    assert "/" not in name


def test_name_stays_within_a_sane_length(tmp_path):
    # No file on disk: an over-long stem must be truncated even when the
    # filesystem would have rejected it outright.
    audio = tmp_path / ("א" * 300 + ".wav")
    name = build_transcript_base_name(audio, "thorough", model_label="תמלול יסודי")
    assert len(name) <= MAX_STEM_CHARS + 64


# ===========================
# log redaction
# ===========================

@pytest.mark.parametrize("secret", [
    "ya29.a0AfH6SMBexampleTOKENvalue",
    "1//0gExampleRefreshTokenValue",
])
def test_google_token_shapes_are_redacted(secret):
    assert secret not in redact(f"got token {secret} ok")


def test_bearer_header_is_redacted():
    assert "abc123" not in redact("Authorization: Bearer abc123")


def test_jwt_is_redacted():
    jwt = "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiIxIn0.signaturepart"
    assert "signaturepart" not in redact(f"id_token={jwt}")


@pytest.mark.parametrize("param", [
    "code", "code_verifier", "access_token", "refresh_token", "id_token",
    "client_secret", "state",
])
def test_query_parameters_are_redacted(param):
    text = f"https://example/callback?{param}=SUPERSECRET&other=1"
    result = redact(text)
    assert "SUPERSECRET" not in result
    assert REDACTED in result
    assert "other=1" in result  # non-secrets survive, so logs stay useful


@pytest.mark.parametrize("field", [
    "code", "access_token", "refresh_token", "id_token", "client_secret",
])
def test_json_fields_are_redacted(field):
    assert "SUPERSECRET" not in redact(f'{{"{field}": "SUPERSECRET"}}')


def test_ordinary_text_is_untouched():
    text = "Analysis finished in 12.3s with 4 detections."
    assert redact(text) == text


def test_filter_redacts_the_message_and_the_args():
    record = logging.LogRecord(
        "monitor.gdrive", logging.INFO, __file__, 1,
        "token=%s and %s", ("ya29.SECRETVALUE", "plain"), None,
    )
    SecretRedactingFilter().filter(record)
    assert "SECRETVALUE" not in record.getMessage()
    assert "plain" in record.getMessage()


def test_filter_redacts_dict_args():
    record = logging.LogRecord(
        "monitor.gdrive", logging.INFO, __file__, 1,
        "%(t)s", ({"t": "ya29.SECRETVALUE"},), None,
    )
    SecretRedactingFilter().filter(record)
    assert "SECRETVALUE" not in record.getMessage()


def test_filter_never_drops_records():
    record = logging.LogRecord(
        "monitor", logging.INFO, __file__, 1, "hello", None, None,
    )
    assert SecretRedactingFilter().filter(record) is True


def test_install_is_idempotent():
    logger = logging.getLogger("monitor.test_redaction_install")
    handler = logging.NullHandler()
    logger.addHandler(handler)
    try:
        first = install_redaction(logger.name)
        second = install_redaction(logger.name)
        assert first is second
        assert sum(isinstance(f, SecretRedactingFilter) for f in logger.filters) == 1
        assert sum(isinstance(f, SecretRedactingFilter) for f in handler.filters) == 1
    finally:
        logger.removeHandler(handler)
        logger.filters.clear()


def test_end_to_end_a_secret_never_reaches_a_handler():
    logger = logging.getLogger("monitor.test_redaction_e2e")
    logger.setLevel(logging.INFO)
    captured: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record):
            captured.append(self.format(record))

    handler = _Capture()
    logger.addHandler(handler)
    try:
        install_redaction(logger.name)
        logger.info("exchanging code=AUTHCODE123 for a token")
        assert captured and "AUTHCODE123" not in captured[0]
    finally:
        logger.removeHandler(handler)
        logger.filters.clear()

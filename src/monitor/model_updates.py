"""
Check the ivrit-ai HuggingFace organisation for new Hebrew STT models.

The app ships with a fixed set of Hebrew faster-whisper (CTranslate2) models.
This module queries the public HuggingFace API and reports any *newer* Hebrew
CT2 speech-to-text models that the app does not already know about, so the user
can decide whether to adopt them.

Yiddish models (ivrit-ai's ``yi-whisper`` line) are deliberately excluded — they
are not suitable for Hebrew transcription even though some CT2 conversions carry
a stray ``he`` tag.

Usage:
    from monitor.model_updates import fetch_new_hebrew_models
    new = fetch_new_hebrew_models()  # list[dict]; raises on network error
"""

from __future__ import annotations

import json
import logging
import urllib.request
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger(__name__)

# HuggingFace API: all ASR models published by the ivrit-ai organisation.
_HF_API_URL = (
    "https://huggingface.co/api/models"
    "?author=ivrit-ai&pipeline_tag=automatic-speech-recognition&limit=1000"
)

# Models the app already ships / offers in the selector.
KNOWN_MODEL_IDS = frozenset({
    "ivrit-ai/whisper-large-v3-ct2",
    "ivrit-ai/whisper-large-v3-turbo-ct2",
})

# Only surface models created after the newest model we already ship
# (whisper-large-v3-ct2, created 2025-03-06). Avoids flagging older models.
_BASELINE_DATE = datetime(2025, 3, 6, tzinfo=timezone.utc)


def _parse_created(entry: dict) -> Optional[datetime]:
    """Parse the ``createdAt`` ISO timestamp from a model entry, if present."""
    raw = entry.get("createdAt")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None


def _is_hebrew_ct2(entry: dict) -> bool:
    """Return True if *entry* is a Hebrew faster-whisper (CT2) model.

    Excludes Yiddish (``yi-whisper`` / ``yi`` tag) models, and requires the
    model to be in CTranslate2 format so it can actually load in the pipeline.
    """
    model_id = (entry.get("id") or entry.get("modelId") or "").lower()
    tags = [str(t).lower() for t in entry.get("tags", [])]

    # Exclude Yiddish models (named yi-whisper / yiddish, or tagged "yi").
    if "yi-whisper" in model_id or "yiddish" in model_id or "yi" in tags:
        return False

    # Require an explicit Hebrew tag.
    if "he" not in tags:
        return False

    # Require faster-whisper / CTranslate2 compatibility.
    return (
        "ctranslate2" in tags
        or model_id.endswith("-ct2")
        or "faster-whisper" in model_id
    )


def fetch_new_hebrew_models(timeout: float = 10.0) -> list[dict]:
    """Query HuggingFace for Hebrew CT2 ASR models unknown to the app.

    Args:
        timeout: Network timeout in seconds.

    Returns:
        A list of ``{"id": str, "created": "YYYY-MM-DD"}`` dicts, sorted
        newest-first. Empty if no new models are found.

    Raises:
        urllib.error.URLError, OSError, ValueError: on network/parse failure.
    """
    req = urllib.request.Request(
        _HF_API_URL,
        headers={"User-Agent": "child-monitor-analyzer"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    results: list[dict] = []
    for entry in data:
        model_id = entry.get("id") or entry.get("modelId")
        if not model_id or model_id in KNOWN_MODEL_IDS:
            continue
        if not _is_hebrew_ct2(entry):
            continue
        created = _parse_created(entry)
        if created is not None and created <= _BASELINE_DATE:
            continue
        results.append({
            "id": model_id,
            "created": created.strftime("%Y-%m-%d") if created else "?",
        })

    results.sort(key=lambda m: m["created"], reverse=True)
    log.info("Model update check: %d new Hebrew CT2 model(s) found.", len(results))
    return results

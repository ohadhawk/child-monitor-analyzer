"""
Canonical transcript file naming.

One definition of the exported file name, shared by the local ``.txt``, the
local ``.docx`` and (later) the Google Drive upload, so the three destinations
can never drift apart.

Format::

    <folder name> - <תמלול יסודי | תמלול מהיר>

No date prefix: the recording's file mtime reflects when it was last copied,
not when it was recorded, so it is not trustworthy enough to put in the name.

The audio stem is untrusted input: recordings are routinely handed over by
third parties, and the stem flows straight into a filesystem path and into an
API request body. :func:`sanitize_display_name` is the choke point for that.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path
from typing import Optional

from ..models import sanitize_artifact_stem

log = logging.getLogger(__name__)

# Bidirectional formatting controls. U+202E (RTLO) is the classic filename
# spoofing vector: "invoice\u202Efdp.exe" renders as "invoiceexe.pdf".
_BIDI_CONTROLS = (
    "\u200e\u200f"          # LRM, RLM
    "\u202a\u202b\u202c\u202d\u202e"  # LRE, RLE, PDF, LRO, RLO
    "\u2066\u2067\u2068\u2069"        # LRI, RLI, FSI, PDI
)

# Illegal in Windows filenames; ":" additionally opens an NTFS alternate data
# stream, and "/" would be read as a path separator by the Drive API.
_ILLEGAL_CHARS = '/\\:*?"<>|'

# Windows reserved device names, matched case-insensitively without extension.
_RESERVED_NAMES = frozenset(
    ["CON", "PRN", "AUX", "NUL"]
    + [f"COM{i}" for i in range(1, 10)]
    + [f"LPT{i}" for i in range(1, 10)]
)

#: Keeps the full name well inside MAX_PATH and Drive's own 32 KiB name limit.
MAX_STEM_CHARS = 120

#: Used when sanitising leaves nothing usable.
FALLBACK_NAME = "transcript"


def sanitize_display_name(name: str, *, max_chars: int = MAX_STEM_CHARS) -> str:
    """Return *name* made safe for use as a filename and as an API name field.

    Args:
        name: Untrusted display name.
        max_chars: Maximum length of the result.

    Returns:
        A non-empty, single-line name with no path separators, no bidi
        overrides, and no Windows reserved device name.
    """
    if not name:
        return FALLBACK_NAME

    # NFC first: decomposed sequences can otherwise smuggle a separator past
    # the character filter on systems that normalise later.
    cleaned = unicodedata.normalize("NFC", name)

    out_chars = []
    for ch in cleaned:
        if ch in _BIDI_CONTROLS or ch in _ILLEGAL_CHARS:
            continue
        # Drops C0/C1 controls (Cc), unassigned (Cn), surrogates (Cs) and the
        # remaining format characters (Cf) such as zero-width joiners.
        if unicodedata.category(ch) in ("Cc", "Cf", "Cn", "Cs"):
            continue
        out_chars.append(ch)
    cleaned = "".join(out_chars)

    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    # Windows silently drops trailing dots/spaces, which desynchronises the
    # name we think we wrote from the one on disk.
    cleaned = cleaned.strip(" .")

    if not cleaned:
        return FALLBACK_NAME

    if cleaned.split(".", 1)[0].upper() in _RESERVED_NAMES:
        cleaned = f"_{cleaned}"

    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].strip(" .") or FALLBACK_NAME

    return cleaned


def build_transcript_base_name(
    audio_path: Optional[str | Path],
    model_key: Optional[str],
    *,
    model_label: Optional[str] = None,
) -> str:
    """Build the canonical transcript base name (no extension).

    Args:
        audio_path: Full path to the source recording, or None.
        model_key: ``"thorough"`` / ``"fast"`` / ``"none"`` / None. The label
            is omitted for ``"none"`` (events-only) and unknown keys.
        model_label: Localised label for *model_key*. The GUI passes the
            translated string; headless callers may omit it.

    Returns:
        A sanitised, non-empty base name.
    """
    if not audio_path:
        return FALLBACK_NAME

    stem = sanitize_artifact_stem(Path(audio_path).stem.strip())
    name = sanitize_display_name(stem)

    if model_label and model_key in ("thorough", "fast"):
        name = f"{name} - {sanitize_display_name(model_label, max_chars=40)}"

    # Re-sanitise the assembled name: the label is translated text and the
    # join could in principle reintroduce a trailing separator.
    return sanitize_display_name(name, max_chars=MAX_STEM_CHARS + 64)

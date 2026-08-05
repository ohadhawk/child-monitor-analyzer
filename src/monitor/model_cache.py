"""
Portable model cache -- keeps all downloaded models inside the program directory.

Resolves the models directory relative to the project root (or PyInstaller
bundle root) and sets environment variables so that HuggingFace and PANNs
download into the local ``models/`` folder rather than the user's home.

Layout:
    <project_root>/
        models/
            huggingface/     <- HF_HOME (Whisper STT, toxicity model, etc.)
            panns/           <- PANNs checkpoint + labels CSV

Usage:
    from monitor.model_cache import get_models_dir, ensure_panns_ready
    import os
    os.environ["HF_HOME"] = str(get_models_dir() / "huggingface")
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# ===========================
# CONSTANTS
# ===========================

# PANNs checkpoint URL (Zenodo) and expected file size (~312 MB).
_PANNS_CHECKPOINT_URL = (
    "https://zenodo.org/record/3987831/files/"
    "Cnn14_DecisionLevelMax_mAP%3D0.385.pth?download=1"
)
_PANNS_CHECKPOINT_NAME = "Cnn14_DecisionLevelMax.pth"
_PANNS_CHECKPOINT_SIZE = 327_428_481  # bytes, exact

# Pinned SHA-256 of the PANNs checkpoint.
#
# PROVENANCE: this digest was taken from a local copy whose MD5 and byte size
# were cross-checked against the authoritative Zenodo record 3987831 metadata
# API, which publishes ``md5:70539c43c18b6a289b3199c503a82c5a`` and
# ``size: 327428481`` for ``Cnn14_DecisionLevelMax_mAP=0.385.pth``.
#
# WHY THIS MATTERS: the checkpoint is a PyTorch pickle. Even though it is now
# loaded with ``weights_only=True`` (see monitor.vendor.panns.inference), a
# substituted checkpoint would silently change every detection this program
# makes. Verifying the digest before use is defence in depth against both a
# compromised mirror and a man-in-the-middle on the download.
_PANNS_CHECKPOINT_SHA256 = (
    "dd3b4043a87d4ec13df8082c0fcfee3fb5084151808e47e060987a95eabdd142"
)

# AudioSet labels CSV URL.
#
# NOTE: upstream PANNs uses an ``http://`` URL for this file. Plain HTTP allows
# an on-path attacker to substitute the class labels, which would silently
# remap every detection type. HTTPS is required here; the label file is
# additionally validated structurally (exactly 527 rows) when parsed.
_PANNS_LABELS_URL = (
    "https://storage.googleapis.com/us_audioset/youtube_corpus/"
    "v1/csv/class_labels_indices.csv"
)
_PANNS_LABELS_NAME = "class_labels_indices.csv"

# Only these URL schemes may be fetched. Model downloads must never fall back
# to plaintext, and ``file://`` would let a crafted config read local paths.
_ALLOWED_URL_SCHEMES = ("https",)

# ===========================
# DIRECTORY RESOLUTION
# ===========================


def get_project_root() -> Path:
    """Return the project root directory.

    Inside a PyInstaller bundle, returns the bundle's temp directory.
    Otherwise, walks up from this file to find the directory containing
    ``pyproject.toml`` (development mode) or falls back to the executable's
    parent directory.

    Returns:
        Path to the project root.
    """
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys.executable).parent

    # Development mode: walk up from src/monitor/model_cache.py to project root.
    candidate = Path(__file__).resolve().parent.parent.parent
    if (candidate / "pyproject.toml").exists():
        return candidate

    return Path.cwd()


def get_models_dir() -> Path:
    """Return the portable models directory, creating it if needed.

    Returns:
        Path to ``<project_root>/models/``.
    """
    models_dir = get_project_root() / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def get_hf_home() -> Path:
    """Return the HuggingFace cache directory inside the models folder.

    Returns:
        Path to ``<project_root>/models/huggingface/``.
    """
    hf_dir = get_models_dir() / "huggingface"
    hf_dir.mkdir(parents=True, exist_ok=True)
    return hf_dir


def get_panns_dir() -> Path:
    """Return the PANNs data directory inside the models folder.

    Returns:
        Path to ``<project_root>/models/panns/``.
    """
    panns_dir = get_models_dir() / "panns"
    panns_dir.mkdir(parents=True, exist_ok=True)
    return panns_dir


# ===========================
# ENVIRONMENT SETUP
# ===========================


def setup_model_environment() -> None:
    """Set environment variables so all model downloads go to the local models dir.

    Must be called early, before importing faster_whisper or transformers.
    """
    hf_home = str(get_hf_home())
    os.environ["HF_HOME"] = hf_home
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(get_hf_home() / "hub")
    log.info("HF_HOME set to %s", hf_home)


# ===========================
# PANNS DOWNLOAD HELPERS
# ===========================


def _require_https(url: str) -> None:
    """Reject any URL that is not plain HTTPS.

    Args:
        url: URL about to be fetched.

    Raises:
        ValueError: If the scheme is not in :data:`_ALLOWED_URL_SCHEMES`.
    """
    scheme = urllib.parse.urlsplit(url).scheme.lower()
    if scheme not in _ALLOWED_URL_SCHEMES:
        raise ValueError(
            f"Refusing to download over {scheme!r}; only "
            f"{'/'.join(_ALLOWED_URL_SCHEMES)} is allowed: {url}"
        )


class _HttpsOnlyRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Redirect handler that refuses to follow a downgrade to plain HTTP.

    The checkpoint host redirects to a CDN, so redirects must be followed --
    but ``urllib`` would happily follow ``https -> http``, which would silently
    drop transport security for a 312 MB binary we are about to load.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _require_https(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _build_opener() -> urllib.request.OpenerDirector:
    """Return an opener that enforces HTTPS across the whole redirect chain."""
    return urllib.request.build_opener(_HttpsOnlyRedirectHandler)


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the lowercase hex SHA-256 digest of *path*.

    Args:
        path: File to hash.
        chunk_size: Read size in bytes; the file is streamed, not slurped.

    Returns:
        Hex digest string.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file_digest(path: Path, expected_sha256: str, label: str) -> bool:
    """Check *path* against *expected_sha256*, logging the outcome.

    Args:
        path: File to verify.
        expected_sha256: Pinned lowercase hex digest.
        label: Human-readable name for log messages.

    Returns:
        True if the digest matches, False otherwise (including read errors).
    """
    try:
        actual = file_sha256(path)
    except OSError as exc:
        log.error("Could not read %s for verification: %s", label, exc)
        return False

    # Constant-time compare is not strictly required for a public digest, but
    # it costs nothing and keeps the habit consistent across the codebase.
    if hmac.compare_digest(actual, expected_sha256.lower()):
        log.info("%s integrity verified (sha256=%s...).", label, actual[:16])
        return True

    log.error(
        "%s FAILED integrity check at %s: expected sha256=%s, got %s",
        label, path, expected_sha256, actual,
    )
    return False


def _download_file(
    url: str,
    dest: Path,
    label: str,
    on_progress: Optional[callable] = None,
) -> None:
    """Download a file with resume support and progress reporting.

    If a partial ``.tmp`` file exists from a previous interrupted download,
    the download resumes from where it left off using an HTTP Range header.

    Args:
        url: Source URL. Must be HTTPS.
        dest: Destination file path.
        label: Human-readable description for log messages.
        on_progress: Optional callback(bytes_downloaded, total_bytes, label).

    Raises:
        ValueError: If *url* is not HTTPS.
    """
    _require_https(url)
    tmp = dest.with_suffix(".tmp")
    existing_size = 0

    if tmp.exists():
        existing_size = tmp.stat().st_size
        log.info(
            "Resuming %s download from %d MB ...",
            label, existing_size // (1024 * 1024),
        )

    req = urllib.request.Request(url)
    if existing_size > 0:
        req.add_header("Range", f"bytes={existing_size}-")

    log.info("Downloading %s from %s ...", label, url)

    with _build_opener().open(req, timeout=60) as response:
        # If the server supports Range, it returns 206 with Content-Range.
        # Otherwise it returns 200 and we must restart from scratch.
        status = getattr(response, "status", 200)
        if status == 206:
            # Validate the server honoured the requested range.
            content_range = response.headers.get("Content-Range", "")
            if not content_range.startswith(f"bytes {existing_size}-"):
                log.info("Server returned unexpected range %r; restarting.", content_range)
                existing_size = 0
                tmp.unlink(missing_ok=True)
        elif existing_size > 0:
            # Server does not support resume; restart.
            log.info("Server does not support resume; restarting download.")
            existing_size = 0

        try:
            content_length = int(response.headers.get("Content-Length", 0))
        except (ValueError, TypeError):
            content_length = 0
        total = existing_size + content_length if content_length else 0
        downloaded = existing_size
        chunk_size = 1024 * 1024  # 1 MB
        last_logged_pct = -1

        mode = "ab" if existing_size > 0 and status == 206 else "wb"
        with open(tmp, mode) as fp:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                fp.write(chunk)
                downloaded += len(chunk)
                if on_progress and total > 0:
                    on_progress(downloaded, total, label)
                if total > 0:
                    pct = downloaded * 100 // total
                    # Log every 10% to avoid log spam.
                    if pct // 10 > last_logged_pct // 10:
                        log.info(
                            "  %s: %d / %d MB (%d%%)",
                            label,
                            downloaded // (1024 * 1024),
                            total // (1024 * 1024),
                            pct,
                        )
                        last_logged_pct = pct

    # replace() rather than rename(): on Windows rename() fails if the target
    # already exists (e.g. a previous partial run).
    tmp.replace(dest)
    log.info(
        "Download complete: %s (%d MB)",
        dest.name, dest.stat().st_size // (1024 * 1024),
    )


def ensure_panns_labels(panns_dir: Optional[Path] = None) -> Path:
    """Ensure a *valid* AudioSet labels CSV exists in the PANNs directory.

    Downloads it over HTTPS from Google Storage if missing, then validates it
    structurally (exactly 527 rows, three columns). An invalid file is deleted
    and fetched once more.

    The labels file is not digest-pinned, because upstream republishes it and
    a hard pin would break first-run installs. Structural validation is the
    substitute: a truncated or substituted file of a different length fails
    closed rather than silently mislabelling every detection.

    Args:
        panns_dir: Override PANNs directory (default: auto-resolved).

    Returns:
        Path to the validated labels CSV file.

    Raises:
        RuntimeError: If the downloaded CSV is still invalid on the retry.
    """
    panns_dir = panns_dir or get_panns_dir()
    labels_csv = panns_dir / _PANNS_LABELS_NAME

    # Imported lazily: labels.py is dependency-free, but keeping the import
    # local avoids pulling the vendor package in for callers that only need
    # the directory helpers.
    from .vendor.panns.labels import load_labels_cached

    for attempt in (1, 2):
        if not labels_csv.exists():
            _download_file(_PANNS_LABELS_URL, labels_csv, "AudioSet labels CSV")

        try:
            # Shares the memo with the detection hot path, so the file is
            # parsed once per process rather than on every call.
            load_labels_cached(str(labels_csv))
        except (ValueError, OSError, UnicodeDecodeError) as exc:
            log.error("AudioSet labels CSV at %s is invalid: %s", labels_csv, exc)
            labels_csv.unlink(missing_ok=True)
            if attempt == 1:
                log.warning("Re-downloading the AudioSet labels CSV.")
            continue

        log.debug("AudioSet labels CSV validated: %s", labels_csv)
        return labels_csv

    raise RuntimeError(
        f"AudioSet labels CSV downloaded from {_PANNS_LABELS_URL} is not a "
        "valid 527-class label file. Refusing to run detection with unknown "
        "class names."
    )


def ensure_panns_checkpoint(
    panns_dir: Optional[Path] = None,
    on_progress: Optional[callable] = None,
) -> Path:
    """Ensure a *verified* PANNs SED checkpoint exists in the PANNs directory.

    Downloads it from Zenodo (~312 MB) if missing, then verifies the file
    against the pinned SHA-256 digest. A file that fails verification is
    quarantined (renamed to ``*.untrusted``) and re-downloaded once; a second
    failure is fatal.

    Args:
        panns_dir: Override PANNs directory (default: auto-resolved).
        on_progress: Optional callback(done_bytes, total_bytes, label).

    Returns:
        Path to the verified checkpoint .pth file.

    Raises:
        RuntimeError: If the downloaded checkpoint fails the digest check.
    """
    panns_dir = panns_dir or get_panns_dir()
    checkpoint = panns_dir / _PANNS_CHECKPOINT_NAME
    label = "PANNs SED checkpoint (~312 MB)"

    for attempt in (1, 2):
        if not checkpoint.exists():
            _download_file(
                _PANNS_CHECKPOINT_URL, checkpoint, label, on_progress=on_progress,
            )

        # Cheap structural check before hashing 312 MB.
        actual_size = checkpoint.stat().st_size
        size_ok = actual_size == _PANNS_CHECKPOINT_SIZE
        if not size_ok:
            log.error(
                "PANNs checkpoint has size %d, expected %d.",
                actual_size, _PANNS_CHECKPOINT_SIZE,
            )

        if size_ok and verify_file_digest(
            checkpoint, _PANNS_CHECKPOINT_SHA256, "PANNs checkpoint",
        ):
            return checkpoint

        # Quarantine rather than delete: keep the artefact for inspection, but
        # move it out of the way so it can never be loaded.
        quarantine = checkpoint.with_suffix(".untrusted")
        try:
            quarantine.unlink(missing_ok=True)
            checkpoint.rename(quarantine)
            log.error("Quarantined unverified checkpoint as %s", quarantine)
        except OSError as exc:
            log.error("Could not quarantine bad checkpoint: %s", exc)
            checkpoint.unlink(missing_ok=True)

        if attempt == 1:
            log.warning("Re-downloading PANNs checkpoint after failed verification.")

    raise RuntimeError(
        "PANNs checkpoint failed SHA-256 verification twice. The download may "
        "be corrupted, or the file may have been tampered with. Expected "
        f"sha256={_PANNS_CHECKPOINT_SHA256}. Refusing to load it."
    )


def ensure_panns_ready(
    panns_dir: Optional[Path] = None,
    on_progress: Optional[callable] = None,
) -> Path:
    """Ensure all PANNs files are downloaded and integrity-checked.

    Args:
        panns_dir: Override PANNs directory (default: auto-resolved).
        on_progress: Optional callback(done_bytes, total_bytes, label).

    Returns:
        Path to the verified PANNs checkpoint file.
    """
    panns_dir = panns_dir or get_panns_dir()
    ensure_panns_labels(panns_dir)
    return ensure_panns_checkpoint(panns_dir, on_progress=on_progress)

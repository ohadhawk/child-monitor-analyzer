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

import logging
import os
import sys
import urllib.request
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# ===========================
# CONSTANTS
# ===========================

# PANNs checkpoint URL (Zenodo) and expected file size (~300 MB).
_PANNS_CHECKPOINT_URL = (
    "https://zenodo.org/record/3987831/files/"
    "Cnn14_DecisionLevelMax_mAP%3D0.385.pth?download=1"
)
_PANNS_CHECKPOINT_NAME = "Cnn14_DecisionLevelMax.pth"
_PANNS_CHECKPOINT_MIN_SIZE = 300_000_000  # bytes

# AudioSet labels CSV URL.
_PANNS_LABELS_URL = (
    "http://storage.googleapis.com/us_audioset/youtube_corpus/"
    "v1/csv/class_labels_indices.csv"
)
_PANNS_LABELS_NAME = "class_labels_indices.csv"

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

    Must be called early, before importing faster_whisper or panns_inference.
    """
    hf_home = str(get_hf_home())
    os.environ["HF_HOME"] = hf_home
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(get_hf_home() / "hub")
    log.info("HF_HOME set to %s", hf_home)


# ===========================
# PANNS DOWNLOAD HELPERS
# ===========================


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
        url: Source URL.
        dest: Destination file path.
        label: Human-readable description for log messages.
        on_progress: Optional callback(bytes_downloaded, total_bytes, label).
    """
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

    with urllib.request.urlopen(req, timeout=60) as response:
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

    tmp.rename(dest)
    log.info(
        "Download complete: %s (%d MB)",
        dest.name, dest.stat().st_size // (1024 * 1024),
    )


def ensure_panns_labels(panns_dir: Optional[Path] = None) -> Path:
    """Ensure the AudioSet labels CSV exists in the PANNs directory.

    Downloads it from Google Storage if missing.

    Args:
        panns_dir: Override PANNs directory (default: auto-resolved).

    Returns:
        Path to the labels CSV file.
    """
    panns_dir = panns_dir or get_panns_dir()
    labels_csv = panns_dir / _PANNS_LABELS_NAME
    if labels_csv.exists():
        log.debug("PANNs labels CSV already present: %s", labels_csv)
        return labels_csv

    _download_file(_PANNS_LABELS_URL, labels_csv, "AudioSet labels CSV")
    return labels_csv


def ensure_panns_checkpoint(
    panns_dir: Optional[Path] = None,
    on_progress: Optional[callable] = None,
) -> Path:
    """Ensure the PANNs SED checkpoint exists in the PANNs directory.

    Downloads it from Zenodo (~300 MB) if missing or truncated.

    Args:
        panns_dir: Override PANNs directory (default: auto-resolved).
        on_progress: Optional callback(done_bytes, total_bytes, label).

    Returns:
        Path to the checkpoint .pth file.
    """
    panns_dir = panns_dir or get_panns_dir()
    checkpoint = panns_dir / _PANNS_CHECKPOINT_NAME
    if checkpoint.exists() and checkpoint.stat().st_size >= _PANNS_CHECKPOINT_MIN_SIZE:
        log.debug("PANNs checkpoint already present: %s", checkpoint)
        return checkpoint

    _download_file(
        _PANNS_CHECKPOINT_URL, checkpoint,
        "PANNs SED checkpoint (~300 MB)", on_progress=on_progress,
    )
    return checkpoint


def ensure_panns_ready(
    panns_dir: Optional[Path] = None,
    on_progress: Optional[callable] = None,
) -> Path:
    """Ensure all PANNs files are downloaded and importable.

    Downloads the labels CSV and checkpoint if missing, then places a copy
    of the labels CSV at the legacy ``~/panns_data/`` location so that
    ``panns_inference.config`` can import without failing at module level.

    Args:
        panns_dir: Override PANNs directory (default: auto-resolved).
        on_progress: Optional callback(done_bytes, total_bytes, label).

    Returns:
        Path to the PANNs checkpoint file.
    """
    panns_dir = panns_dir or get_panns_dir()
    labels_csv = ensure_panns_labels(panns_dir)
    checkpoint = ensure_panns_checkpoint(panns_dir, on_progress=on_progress)

    # panns_inference.config reads ~/panns_data/class_labels_indices.csv at
    # import time. Place a copy there so the import does not fail or trigger
    # a wget call (which does not exist on Windows).
    _ensure_legacy_panns_labels(labels_csv)

    return checkpoint


def _ensure_legacy_panns_labels(local_labels: Path) -> None:
    """Copy the labels CSV to the legacy ~/panns_data/ location.

    This is required because panns_inference.config opens the file at
    module-level during ``import panns_inference``.

    Args:
        local_labels: Path to our portable labels CSV.
    """
    import shutil

    legacy_dir = Path.home() / "panns_data"
    legacy_csv = legacy_dir / _PANNS_LABELS_NAME
    try:
        if legacy_csv.exists():
            return
        legacy_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_labels, legacy_csv)
        log.debug("Copied labels CSV to legacy path: %s", legacy_csv)
    except OSError as exc:
        log.warning("Could not copy labels to legacy path: %s", exc)

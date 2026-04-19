"""
Hebrew profanity detection -- word-list matching with prefix stripping + AI classification.

Implements a dual-layer approach:
  Layer 1: Exact word matching against curated Hebrew profanity word lists
           (hard + soft tiers), with morphological prefix stripping for
           common Hebrew prefixes.
  Layer 2: Sentence-level AI classification using textdetox
           multilingual toxicity classifier for context-aware detection.

Usage:
    from monitor.profanity import ProfanityDetector

    detector = ProfanityDetector()
    detections = detector.detect(transcribed_segments)
"""

from __future__ import annotations

import logging
import re
import threading
import time
from pathlib import Path
from typing import Callable, List, Optional, Set

from .models import Detection, DetectionType, TranscribedSegment

log = logging.getLogger(__name__)

# ===========================
# CONSTANTS
# ===========================

# Hebrew prefixes that can be attached to words (ordered longest first).
# In Hebrew, common prepositions and conjunctions attach directly to the
# following word: "beshimuk" = "be" + "shimuk".
# Example: "בשמוק" (with prefix ב) -> base form "שמוק"
_HEBREW_PREFIXES = (
    "שה", "מה", "וה", "וש", "ומ", "וב", "ול", "וכ",
    "ב", "ל", "מ", "כ", "ה", "ו", "ש",
)

# Path to the data directory containing word list files.
# When running inside a PyInstaller bundle, sys._MEIPASS points to the
# temporary extraction folder; otherwise use the source tree layout.
import sys as _sys

if getattr(_sys, "frozen", False) and hasattr(_sys, "_MEIPASS"):
    _DATA_DIR = Path(_sys._MEIPASS) / "data"
else:
    _DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

# Pre-compiled regex: keeps only Hebrew characters (Unicode block 0590-05FF).
# Example: "שמוק!" -> "שמוק"
_HEBREW_ONLY_RE = re.compile(r"[^\u0590-\u05FF]")

# ===========================
# HELPER FUNCTIONS
# ===========================


def _strip_hebrew_prefixes(word: str) -> List[str]:
    """Return possible base forms after stripping common Hebrew prefixes.

    Args:
        word: A Hebrew word (may include a prefix).

    Returns:
        List of candidate base forms, always including the original word.

    Example:
        _strip_hebrew_prefixes("בשמוק") -> ["בשמוק", "שמוק"]
    """
    forms = [word]
    for prefix in _HEBREW_PREFIXES:
        # Only strip if the remaining base has at least 2 characters.
        if word.startswith(prefix) and len(word) > len(prefix) + 1:
            forms.append(word[len(prefix):])
    return forms


def _load_word_list(path: Path) -> Set[str]:
    """Load a newline-separated word list, ignoring blank lines and comments.

    Args:
        path: Path to the word list file.

    Returns:
        Set of normalised word strings.
    """
    if not path.exists():
        log.warning("Word list not found: %s", path)
        return set()

    words: Set[str] = set()
    # Context manager (with) -- ensures file is closed even on exceptions.
    with open(path, "r", encoding="utf-8") as word_file:
        for line in word_file:
            cleaned = line.strip()
            # Lines starting with '#' are comments.
            if cleaned and not cleaned.startswith("#"):
                words.add(cleaned)
    return words


# ===========================
# CORE CLASS
# ===========================


class ProfanityDetector:
    """Detect profanity in transcribed Hebrew text.

    Layer 1: Word-list matching with Hebrew morphological prefix stripping.
    Layer 2: AI classification using textdetox multilingual toxicity model.

    Attributes:
        _hard_words: Set of severe profanity words.
        _soft_words: Set of milder / child-sensitive words.
        _all_words: Union of hard and soft sets for matching.
        _use_ai: Whether to attempt AI classification.
        _ai_threshold: Minimum AI probability to flag a segment.
    """

    def __init__(
        self,
        hard_list_path: Optional[str | Path] = None,
        soft_list_path: Optional[str | Path] = None,
        use_ai: bool = True,
        ai_threshold: float = 0.5,
    ) -> None:
        self._hard_words = _load_word_list(
            Path(hard_list_path) if hard_list_path else _DATA_DIR / "he_profanity.txt"
        )
        self._soft_words = _load_word_list(
            Path(soft_list_path) if soft_list_path else _DATA_DIR / "he_profanity_soft.txt"
        )
        self._all_words = self._hard_words | self._soft_words
        self._use_ai = use_ai
        self._ai_threshold = ai_threshold
        self._ai_pipeline = None
        self._ai_load_lock = threading.Lock()

        log.info(
            "Profanity word lists loaded: %d hard + %d soft = %d total.",
            len(self._hard_words),
            len(self._soft_words),
            len(self._all_words),
        )

    # ------------------------------------------------------------------
    # Private helpers -- AI model
    # ------------------------------------------------------------------

    # Sub-progress callback type: (bytes_done, bytes_total, label).
    _SubProgressCallback = Callable[[int, int, str], None]

    _AI_MODEL_TIMEOUT = 120  # seconds to wait for model download/load

    def preload_ai_model(
        self,
        on_sub_progress: Optional[_SubProgressCallback] = None,
    ) -> None:
        """Public entry point to eagerly load the AI model.

        Safe to call from any thread; guarded by _ai_load_lock.

        Args:
            on_sub_progress: Optional callback for download progress
                (bytes_done, bytes_total, label).
        """
        self._load_ai_model(on_sub_progress)

    def _load_ai_model(
        self,
        on_sub_progress: Optional[_SubProgressCallback] = None,
    ) -> None:
        """Load the toxicity AI pipeline if not already loaded.

        Tries local cache first (instant). If not cached, attempts download
        with a thread-based timeout to avoid hanging on network issues.
        Thread-safe: guarded by _ai_load_lock.
        """
        with self._ai_load_lock:
            self._load_ai_model_unlocked(on_sub_progress)

    def _load_ai_model_unlocked(
        self,
        on_sub_progress: Optional[_SubProgressCallback] = None,
    ) -> None:
        """Inner model loading logic. Caller must hold _ai_load_lock."""
        if self._ai_pipeline is not None:
            return
        try:
            from transformers import pipeline as hf_pipeline

            log.info("Loading textdetox toxicity AI model...")
            # Try local cache first (no network).
            # Patch huggingface_hub offline flag directly — the env var is
            # cached at import time, so os.environ alone won't help.
            import huggingface_hub.constants as _hf_consts
            _prev_offline = _hf_consts.HF_HUB_OFFLINE
            _hf_consts.HF_HUB_OFFLINE = True
            try:
                self._ai_pipeline = hf_pipeline(
                    "text-classification",
                    model="textdetox/bert-multilingual-toxicity-classifier",
                    truncation=True,
                    max_length=512,
                    local_files_only=True,
                )
                log.info("AI toxicity model loaded from local cache.")
                return
            except OSError:
                log.info("AI model not in local cache; attempting download...")
            finally:
                _hf_consts.HF_HUB_OFFLINE = _prev_offline

            # Build a tqdm class that forwards download progress to the GUI.
            model_kwargs: dict = {}
            if on_sub_progress is not None:
                from .gui.strings import tr, S
                try:
                    from tqdm.auto import tqdm as _tqdm_base
                except ImportError:
                    _tqdm_base = None

                if _tqdm_base is not None:
                    _MIN_TOTAL = 1_000_000  # 1 MB — skip tiny config files
                    _REPORT_INTERVAL = 0.25  # seconds between GUI updates
                    _cb = on_sub_progress
                    _label = tr(S.TOXICITY_DOWNLOADING)

                    class _ProgressTqdm(_tqdm_base):
                        """Forwards tqdm byte-progress to the GUI."""

                        def __init__(self, *args, **kwargs):
                            super().__init__(*args, **kwargs)
                            self._lock = threading.Lock()
                            self._actual_downloaded = 0
                            self._actual_total = 0
                            self._last_report_time = 0.0

                        def update(self, n=1):
                            super().update(n)
                            if getattr(self, "unit", None) != "B":
                                return
                            with self._lock:
                                self._actual_downloaded += n
                                self._actual_total = self.total or 0
                                total = self._actual_total
                                if total < _MIN_TOTAL:
                                    return
                                downloaded = min(self._actual_downloaded, total)
                                now = time.perf_counter()
                                is_done = downloaded >= total
                                if not is_done and (now - self._last_report_time) < _REPORT_INTERVAL:
                                    return
                                self._last_report_time = now
                            _cb(downloaded, total, _label)

                        def close(self):
                            # Do NOT call _cb here — GC may call close()
                            # after the pipeline has hidden the bar.
                            super().close()

                    model_kwargs["tqdm_class"] = _ProgressTqdm

            # Download with timeout.
            result: list = []
            error: list = []

            def _download() -> None:
                try:
                    result.append(hf_pipeline(
                        "text-classification",
                        model="textdetox/bert-multilingual-toxicity-classifier",
                        truncation=True,
                        max_length=512,
                        model_kwargs=model_kwargs if model_kwargs else None,
                    ))
                except Exception as exc:
                    error.append(exc)

            t = threading.Thread(target=_download, daemon=True)
            t.start()
            t.join(timeout=self._AI_MODEL_TIMEOUT)
            if t.is_alive():
                log.warning(
                    "Toxicity model download timed out after %ds; "
                    "AI classification disabled.",
                    self._AI_MODEL_TIMEOUT,
                )
                self._use_ai = False
                return
            if error:
                raise error[0]
            if result:
                self._ai_pipeline = result[0]
                log.info("AI toxicity model downloaded and loaded.")
            else:
                raise RuntimeError("AI model load returned no result")
        except Exception:
            log.exception("Failed to load toxicity model; AI classification disabled.")
            self._use_ai = False

    @property
    def ai_available(self) -> bool:
        """Whether the AI profanity model is loaded and active."""
        return self._ai_pipeline is not None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, segments: List[TranscribedSegment]) -> List[Detection]:
        """Scan transcribed segments for profanity.

        Args:
            segments: List of transcribed speech segments with word-level data.

        Returns:
            List of Detection objects for segments containing profanity.
        """
        detections: List[Detection] = []

        for segment in segments:
            matched_words = self._match_words(segment)
            ai_score = self._classify_ai(segment.text) if self._use_ai else None

            is_profane_by_words = len(matched_words) > 0
            is_profane_by_ai = ai_score is not None and ai_score >= self._ai_threshold

            if not (is_profane_by_words or is_profane_by_ai):
                continue

            # --- Determine time span ---
            start, end = self._get_detection_span(segment, matched_words)

            details = {"words": sorted(matched_words), "sentence": segment.text}
            if ai_score is not None:
                details["ai_score"] = round(ai_score, 3)

            # max() picks the higher confidence between word-match (0.9) and AI.
            confidence = max(
                ai_score if ai_score is not None else 0.0,
                0.9 if is_profane_by_words else 0.0,
            )

            detections.append(
                Detection(
                    type=DetectionType.PROFANITY,
                    start=start,
                    end=end,
                    confidence=confidence,
                    details=details,
                )
            )

        return detections

    # ------------------------------------------------------------------
    # Private helpers -- matching
    # ------------------------------------------------------------------

    def _get_detection_span(
        self,
        segment: TranscribedSegment,
        matched_words: Set[str],
    ) -> tuple[float, float]:
        """Determine the time span for a profanity detection.

        Uses word-level timestamps when available, otherwise falls back
        to the segment boundaries.

        Args:
            segment: The transcribed segment.
            matched_words: Set of matched profanity words.

        Returns:
            Tuple of (start_seconds, end_seconds).
        """
        if matched_words and segment.words:
            word_objects = [
                word for word in segment.words
                if word.word in matched_words
                or any(
                    stripped in self._all_words
                    for stripped in _strip_hebrew_prefixes(word.word)
                )
            ]
            if word_objects:
                return (
                    min(word.start for word in word_objects),
                    max(word.end for word in word_objects),
                )
        return segment.start, segment.end

    def _match_words(self, segment: TranscribedSegment) -> Set[str]:
        """Match words in a segment against the profanity word lists.

        Args:
            segment: A transcribed speech segment.

        Returns:
            Set of matched Hebrew word strings.
        """
        if not self._all_words:
            return set()

        matched: Set[str] = set()
        # Use individual word objects when available; otherwise tokenise text.
        # re.split(r"\\s+", text) splits on any whitespace.
        raw_words = (
            [word.word for word in segment.words]
            if segment.words
            else re.split(r"\s+", segment.text)
        )
        for raw_word in raw_words:
            # Remove non-Hebrew characters (punctuation, digits, Latin).
            # Example: "שמוק!" -> "שמוק"
            clean = _HEBREW_ONLY_RE.sub("", raw_word)
            if not clean:
                continue
            for form in _strip_hebrew_prefixes(clean):
                if form in self._all_words:
                    matched.add(clean)
                    break

        return matched

    def _classify_ai(self, text: str) -> Optional[float]:
        """Run AI classification on a text segment.

        Args:
            text: Hebrew text string to classify.

        Returns:
            Profanity probability (0.0-1.0), or None on failure / when disabled.
        """
        if not self._use_ai:
            return None
        self._load_ai_model()
        if self._ai_pipeline is None:
            return None
        try:
            result = self._ai_pipeline(text)[0]
            label = result["label"]
            score = result["score"]
            # The model may return labels like "LABEL_1" (profane) / "LABEL_0" (clean).
            if label in ("LABEL_1", "profane", "toxic", "1"):
                return score
            return 1.0 - score
        except Exception:
            log.exception("AI classification failed for text: %.60s", text)
            return None

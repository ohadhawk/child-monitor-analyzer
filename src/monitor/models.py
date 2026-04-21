"""
Data models for detection results and analysis reports.

Defines the core data structures used throughout the Child Monitor Analyzer:
DetectionType enum, transcription dataclasses, Detection events, and
the top-level AnalysisReport that aggregates all results.

Usage:
    from monitor.models import Detection, DetectionType, AnalysisReport
"""

from __future__ import annotations

import enum
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# ===========================
# ENUMS
# ===========================


class DetectionType(enum.Enum):
    """Types of audio events that can be detected.

    Each value corresponds to a specific audio pattern the pipeline
    recognises (speech-based profanity, acoustic events, or volume anomalies).
    """

    PROFANITY = "profanity"
    SHOUT = "shout"
    SCREAM = "scream"
    CRY = "cry"
    WAIL = "wail"
    BABY_CRY = "baby_cry"
    LAUGHTER = "laughter"
    VOLUME_SPIKE = "volume_spike"


# ===========================
# CONSTANTS
# ===========================

# Hebrew display labels for each detection type.
# Used in the CLI summary table and the GUI report.
DETECTION_LABELS_HE: Dict[DetectionType, str] = {
    DetectionType.PROFANITY: "ניבול פה",
    DetectionType.SHOUT: "צעקה",
    DetectionType.SCREAM: "צרחה",
    DetectionType.CRY: "בכי",
    DetectionType.WAIL: "יללה",
    DetectionType.BABY_CRY: "בכי תינוק",
    DetectionType.LAUGHTER: "צחוק",
    DetectionType.VOLUME_SPIKE: "עוצמה חריגה",
}

# AudioSet class names (as produced by PANNs) mapped to our detection types.
# PANNs uses the Google AudioSet ontology with 527 classes; we only care about
# the subset that represents distress/emotional audio events.
AUDIOSET_CLASS_MAP: Dict[str, DetectionType] = {
    "Shout": DetectionType.SHOUT,
    "Yell": DetectionType.SHOUT,
    "Children shouting": DetectionType.SHOUT,
    "Screaming": DetectionType.SCREAM,
    "Crying, sobbing": DetectionType.CRY,
    "Wail, moan": DetectionType.WAIL,
    "Baby cry, infant cry": DetectionType.BABY_CRY,
    "Laughter": DetectionType.LAUGHTER,
    "Baby laughter": DetectionType.LAUGHTER,
}

# ===========================
# DATACLASSES
# ===========================


@dataclass
class TranscribedWord:
    """A single word from speech-to-text with timestamps.

    Attributes:
        word: The transcribed word text.
        start: Start time in seconds from beginning of audio.
        end: End time in seconds.
        confidence: STT confidence score, 0.0 to 1.0.
    """

    word: str
    start: float
    end: float
    confidence: float = 1.0


@dataclass
class TranscribedSegment:
    """A segment (sentence or phrase) from speech-to-text.

    Attributes:
        text: Full text of the segment.
        start: Start time in seconds from beginning of audio.
        end: End time in seconds.
        words: Individual words with their own timestamps.
    """

    text: str
    start: float
    end: float
    # field(default_factory=list) -- avoids the mutable-default-argument
    # pitfall where all instances would share the SAME list object.
    words: List[TranscribedWord] = field(default_factory=list)


@dataclass
class Detection:
    """A single detected event in the audio.

    Attributes:
        type: Category of the detection (profanity, shout, cry, etc.).
        start: Start time in seconds from beginning of audio.
        end: End time in seconds.
        confidence: Detection confidence score, 0.0 to 1.0.
        details: Extra information (e.g. matched profanity words, AudioSet class).
    """

    type: DetectionType
    start: float
    end: float
    confidence: float = 1.0
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def label_he(self) -> str:
        """Return the Hebrew display label for this detection type.

        Returns:
            Hebrew string such as "ניבול פה" or "צעקה".
        """
        # dict.get() -- returns default if key missing (no KeyError).
        # Example: DETECTION_LABELS_HE.get(PROFANITY, "profanity") -> "ניבול פה"
        return DETECTION_LABELS_HE.get(self.type, self.type.value)

    @property
    def time_display(self) -> str:
        """Format start time as HH:MM:SS.

        Returns:
            Time string, e.g. "01:02:35".
        """
        total = int(self.start)
        hours, remainder = divmod(total, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    @property
    def profanity_words(self) -> List[str]:
        """Return list of profanity words found (only meaningful for PROFANITY type).

        Returns:
            List of matched Hebrew words, or empty list.
        """
        return self.details.get("words", [])


@dataclass
class AnalysisReport:
    """Full analysis report for an audio file.

    Aggregates transcription segments and detected events into a single
    structure that can be displayed in the GUI or exported to JSON.

    Attributes:
        audio_path: Path to the analysed audio file.
        duration_seconds: Total audio duration in seconds.
        segments: Transcribed speech segments with word-level timestamps.
        detections: All detected events (profanity, shout, cry, etc.).
    """

    audio_path: str
    duration_seconds: float = 0.0
    segments: List[TranscribedSegment] = field(default_factory=list)
    detections: List[Detection] = field(default_factory=list)
    stt_model_key: str = "thorough"

    @property
    def full_transcription(self) -> str:
        """Concatenate all segment texts into a single transcription string.

        Returns:
            Full transcription with segments separated by spaces.
        """
        # str.join() -- combines list elements into single string.
        # Example: ["Hello", "world"] -> "Hello world"
        return " ".join(seg.text for seg in self.segments)

    def detections_sorted(self) -> List[Detection]:
        """Return detections sorted by start time.

        Returns:
            New list of Detection objects ordered chronologically.
        """
        return sorted(self.detections, key=lambda detection: detection.start)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the report to a JSON-compatible dictionary.

        Returns:
            Dictionary suitable for ``json.dumps()``.
        """
        return {
            "audio_path": self.audio_path,
            "duration_seconds": self.duration_seconds,
            "stt_model_key": self.stt_model_key,
            "transcription": self.full_transcription,
            "segments": [
                {
                    "text": segment.text,
                    "start": round(segment.start, 2),
                    "end": round(segment.end, 2),
                    "words": [
                        {
                            "word": word.word,
                            "start": round(word.start, 2),
                            "end": round(word.end, 2),
                            "confidence": round(word.confidence, 3),
                        }
                        for word in segment.words
                    ],
                }
                for segment in self.segments
            ],
            "detections": [
                {
                    "type": detection.type.value,
                    "label_he": detection.label_he,
                    "start": round(detection.start, 2),
                    "end": round(detection.end, 2),
                    "confidence": round(detection.confidence, 3),
                    "details": detection.details,
                }
                for detection in self.detections_sorted()
            ],
        }

    # ------------------------------------------------------------------
    # Serialization / deserialization
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AnalysisReport:
        """Deserialize a report from a dictionary (as produced by ``to_dict``).

        Args:
            data: Dictionary with keys matching ``to_dict`` output.

        Returns:
            Reconstructed AnalysisReport.
        """
        segments = [
            TranscribedSegment(
                text=seg["text"],
                start=seg["start"],
                end=seg["end"],
                words=[
                    TranscribedWord(
                        word=w["word"],
                        start=w["start"],
                        end=w["end"],
                        confidence=w.get("confidence", 1.0),
                    )
                    for w in seg.get("words", [])
                ],
            )
            for seg in data.get("segments", [])
        ]
        detections = [
            Detection(
                type=DetectionType(det["type"]),
                start=det["start"],
                end=det["end"],
                confidence=det.get("confidence", 1.0),
                details=det.get("details", {}),
            )
            for det in data.get("detections", [])
        ]
        return cls(
            audio_path=data.get("audio_path", ""),
            duration_seconds=data.get("duration_seconds", 0.0),
            segments=segments,
            detections=detections,
            stt_model_key=data.get("stt_model_key", "thorough"),
        )

    # ------------------------------------------------------------------
    # Cache file helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _artifact_dir(audio_path: str | Path) -> Path:
        """Return the artifact directory for an audio file.

        Creates the directory if it does not exist.

        Args:
            audio_path: Path to the audio file.

        Returns:
            Path like ``/dir/recording/`` (stem of audio filename).
        """
        d = Path(audio_path).parent / Path(audio_path).stem
        d.mkdir(parents=True, exist_ok=True)
        return d

    @staticmethod
    def _old_cache_path(audio_path: str | Path) -> Path:
        """Legacy cache path (for backwards compatibility)."""
        return Path(audio_path).with_suffix(
            Path(audio_path).suffix + ".analysis.json"
        )

    @staticmethod
    def get_cache_path(audio_path: str | Path, stt_model_key: str = "thorough") -> Path:
        """Return the cache file path for a given audio file and model.

        Cache is stored inside an artifact folder named after the audio
        file (without extension).

        Args:
            audio_path: Path to the audio file.
            stt_model_key: Model key for per-model cache separation.

        Returns:
            Path like ``/dir/recording/analysis_thorough.json``.
        """
        d = Path(audio_path).parent / Path(audio_path).stem
        return d / f"analysis_{stt_model_key}.json"

    def save_cache(self) -> Path:
        """Save this report as a JSON cache file in the artifact folder.

        Uses write-to-temp-then-rename for crash safety.

        Returns:
            Path to the written cache file.
        """
        cache_path = self.get_cache_path(self.audio_path, self.stt_model_key)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(".tmp")
        tmp_path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp_path.replace(cache_path)
        log.info("Analysis cached to %s", cache_path)
        return cache_path

    @classmethod
    def load_cache(
        cls,
        audio_path: str | Path,
        stt_model_key: str = "thorough",
    ) -> Optional[AnalysisReport]:
        """Load a cached analysis report for the given audio file and model.

        Checks the new artifact folder first, then falls back to the
        legacy location.  If found at the legacy path, migrates the file
        to the new folder.

        Returns None if no cache exists or if the cache is older than the
        audio file (i.e. the audio was modified since last analysis).

        Args:
            audio_path: Path to the audio file.
            stt_model_key: Model key for per-model cache separation.

        Returns:
            AnalysisReport if a valid cache exists, otherwise None.
        """
        audio_path = Path(audio_path)
        cache_path = cls.get_cache_path(audio_path, stt_model_key)

        # Fall back to legacy path and migrate if needed (only for default model).
        if not cache_path.exists() and stt_model_key == "thorough":
            # Also check model-agnostic name from before per-model caching.
            old_generic = audio_path.parent / audio_path.stem / "analysis.json"
            if old_generic.exists():
                try:
                    old_generic.rename(cache_path)
                    log.info("Migrated analysis cache %s → %s", old_generic, cache_path)
                except OSError:
                    cache_path = old_generic
            else:
                old_path = cls._old_cache_path(audio_path)
                if old_path.exists():
                    try:
                        cache_path.parent.mkdir(parents=True, exist_ok=True)
                        old_path.rename(cache_path)
                        log.info("Migrated cache %s → %s", old_path, cache_path)
                    except OSError as exc:
                        log.warning("Cache migration failed: %s", exc)
                        cache_path = old_path  # read from old location

        if not cache_path.exists():
            return None

        # Invalidate if audio is newer than cache.
        if audio_path.stat().st_mtime > cache_path.stat().st_mtime:
            log.info("Cache stale (audio modified); will re-analyse.")
            return None

        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            report = cls.from_dict(data)
            log.info(
                "Loaded cached analysis: %d detections from %s",
                len(report.detections), cache_path,
            )
            return report
        except FileNotFoundError:
            log.info("Cache file disappeared before reading.")
            return None
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            log.warning("Failed to load cache %s: %s", cache_path, exc)
            return None

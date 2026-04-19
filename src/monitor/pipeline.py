"""
Pipeline orchestrator -- runs all analysis stages and merges results.

Coordinates the three analysis stages:
  1. Hebrew STT (speech-to-text) via faster-whisper.
  2. Audio event detection (PANNs + RMS volume).
  3. Profanity detection on the transcription output.

Stages 1 and 2 run in parallel; stage 3 depends on stage 1's output.

Usage:
    from monitor.pipeline import AnalysisPipeline

    pipeline = AnalysisPipeline()
    report = pipeline.analyze("recording.wav")
"""

from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, List, Optional

from .audio_events import AudioEventDetector
from .model_cache import setup_model_environment
from .models import AnalysisReport, Detection, DetectionType, TranscribedSegment
from .profanity import ProfanityDetector
from .stt import HebrewSTT
from .gui.strings import tr, S

# Redirect all model downloads to the portable models/ directory.
setup_model_environment()

log = logging.getLogger(__name__)

# ===========================
# TYPE ALIASES
# ===========================

# Progress callback: (percent 0-100, status_message_hebrew).
ProgressCallback = Callable[[int, str], None]

# Sub-progress callback: (bytes_done, bytes_total, label).
# bytes_total == 0 means indeterminate; bytes_done == -1 means hide.
SubProgressCallback = Callable[[int, int, str], None]

# Task progress callback: (task_id, percent 0-100, label).
# task_id 0 = STT, task_id 1 = audio events.
# percent == -1 means hide that task bar.
TaskProgressCallback = Callable[[int, int, str], None]

# Partial-result callbacks for incremental live display.
# on_partial_stt: receives a list of new TranscribedSegment dicts.
# on_partial_events: receives a list of new Detection dicts.
PartialSttCallback = Callable[[list], None]
PartialEventsCallback = Callable[[list], None]

# Warning callback: (warning_key: str) -- string key from gui.strings.S.
WarningCallback = Callable[[str], None]

# ===========================
# CORE CLASS
# ===========================


class AnalysisPipeline:
    """Orchestrate Hebrew audio analysis: STT + audio events + profanity.

    Attributes:
        _stt: Hebrew speech-to-text engine.
        _audio_events: Audio event detector (PANNs + volume).
        _profanity: Profanity detector (word-list + AI).
    """

    def __init__(
        self,
        *,
        stt: Optional[HebrewSTT] = None,
        audio_events: Optional[AudioEventDetector] = None,
        profanity: Optional[ProfanityDetector] = None,
        use_ai_profanity: bool = True,
    ) -> None:
        self._stt = stt or HebrewSTT()
        self._audio_events = audio_events or AudioEventDetector()
        self._profanity = profanity or ProfanityDetector(use_ai=use_ai_profanity)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        audio_path: str | Path,
        on_progress: Optional[ProgressCallback] = None,
        on_sub_progress: Optional[SubProgressCallback] = None,
        on_sub_progress2: Optional[SubProgressCallback] = None,
        on_sub_progress3: Optional[SubProgressCallback] = None,
        on_task_progress: Optional[TaskProgressCallback] = None,
        on_partial_stt: Optional[PartialSttCallback] = None,
        on_partial_events: Optional[PartialEventsCallback] = None,
        on_warning: Optional[WarningCallback] = None,
    ) -> AnalysisReport:
        """Run full analysis on an audio file.

        Args:
            audio_path: Path to audio file.
            on_progress: Optional callback receiving (percent, status_message).
            on_sub_progress: Optional callback for slot 0 sub-operation progress
                (bytes_done, bytes_total, label). total=0 for indeterminate.
                Used for the STT model download.
            on_sub_progress2: Optional callback for slot 1 sub-operation progress
                (bytes_done, bytes_total, label). Used for the PANNs model
                download so both can be shown concurrently.
            on_sub_progress3: Optional callback for slot 2 sub-operation progress
                (bytes_done, bytes_total, label). Used for the toxicity model
                download.
            on_task_progress: Optional callback for parallel task progress
                (task_id, percent, label). task_id: 0=STT, 1=audio_events.
            on_partial_stt: Optional callback for incremental STT segments.
            on_partial_events: Optional callback for incremental event detections.
            on_warning: Optional callback for non-fatal warnings (string key).

        Returns:
            AnalysisReport with transcription and all detections.

        Raises:
            FileNotFoundError: If *audio_path* does not exist.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(
                f"Audio file not found: {audio_path}\n"
                f"Absolute path: {audio_path.resolve()}"
            )

        return self._analyze_impl(
            audio_path, on_progress, on_sub_progress,
            on_sub_progress2, on_sub_progress3, on_task_progress,
            on_partial_stt, on_partial_events, on_warning,
        )

    def _analyze_impl(
        self,
        audio_path: Path,
        on_progress: Optional[ProgressCallback],
        on_sub_progress: Optional[SubProgressCallback],
        on_sub_progress2: Optional[SubProgressCallback],
        on_sub_progress3: Optional[SubProgressCallback],
        on_task_progress: Optional[TaskProgressCallback],
        on_partial_stt: Optional[PartialSttCallback],
        on_partial_events: Optional[PartialEventsCallback],
        on_warning: Optional[WarningCallback],
    ) -> AnalysisReport:
        """Inner analyze body."""

        def _progress(pct: int, msg: str) -> None:
            if on_progress:
                on_progress(pct, msg)
            log.info("[%d%%] %s", pct, msg)

        def _sub(done: int, total: int, label: str) -> None:
            # Diagnostic guard: catch impossible values that indicate a bug
            # in the producer (e.g. progress accounting going off the rails).
            if done > total > 0 or (total > 0 and done * 100 // total > 100):
                log.warning(
                    "sub-progress slot 0 inconsistent: done=%d total=%d label=%r",
                    done, total, label,
                )
            if on_sub_progress:
                on_sub_progress(done, total, label)

        def _sub2(done: int, total: int, label: str) -> None:
            if done > total > 0 or (total > 0 and done * 100 // total > 100):
                log.warning(
                    "sub-progress slot 1 inconsistent: done=%d total=%d label=%r",
                    done, total, label,
                )
            if on_sub_progress2:
                on_sub_progress2(done, total, label)

        def _sub3(done: int, total: int, label: str) -> None:
            if done > total > 0 or (total > 0 and done * 100 // total > 100):
                log.warning(
                    "sub-progress slot 2 inconsistent: done=%d total=%d label=%r",
                    done, total, label,
                )
            if on_sub_progress3:
                on_sub_progress3(done, total, label)

        def _task(task_id: int, pct: int, label: str) -> None:
            if on_task_progress:
                on_task_progress(task_id, pct, label)

        # --- Check for cached analysis ---
        cached = AnalysisReport.load_cache(audio_path)
        if cached is not None:
            _progress(100, "נטען מקובץ מטמון!")
            return cached

        _progress(0, "מתחיל ניתוח...")

        # --- Phase 0.5: Load all models in parallel (with dual progress) ---
        _progress(2, tr(S.PIPE_LOADING_STT))
        phase_start = time.perf_counter()
        log.debug("phase=load_models start (parallel STT + SED + profanity)")

        # Start profanity AI model preload on a standalone daemon thread.
        # This runs concurrently and doesn't block Phase 0.5 completion.
        # Phase 2 will find the model already loaded (or wait via the lock
        # if the download is still in progress).
        _prof_thread = threading.Thread(
            target=self._profanity.preload_ai_model,
            args=(_sub3,),
            name="profanity-preload",
            daemon=True,
        )
        _prof_thread.start()

        with ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="model-load",
        ) as pool:
            f_stt = pool.submit(self._stt._load_model, on_sub_progress=_sub)
            f_sed = pool.submit(self._audio_events._load_sed, on_sub_progress=_sub2)
            # Surface any exceptions raised in the worker threads.
            f_stt.result()
            f_sed.result()
        log.debug(
            "phase=load_models done in %.2fs", time.perf_counter() - phase_start,
        )
        _sub(-1, -1, "")   # hide slot-0 sub-bar
        _sub2(-1, -1, "")  # hide slot-1 sub-bar
        _progress(25, tr(S.PIPE_ALL_MODELS_LOADED))

        # --- Phase 1: STT + audio events in parallel ---
        segments, audio_detections = self._run_parallel_phase(
            audio_path, _progress, _task,
            on_partial_stt, on_partial_events,
        )
        # Hide task bars after parallel phase.
        _task(0, -1, "")
        _task(1, -1, "")

        # NOTE: Intermediate caches (.stt_cache.json / .events_cache.json) are
        # intentionally kept so interrupted runs can resume.  They are only
        # removed when the user explicitly re-starts processing.

        # --- Phase 2: Profanity detection on transcription ---
        _progress(70, tr(S.PIPE_PROFANITY_SEARCH))
        profanity_detections = self._profanity.detect(segments)
        _progress(85, f"{tr(S.PIPE_PROFANITY_SEARCH)} -- {len(profanity_detections)}")

        # Warn the GUI if AI profanity model was not available.
        if not self._profanity.ai_available and on_warning:
            try:
                on_warning(S.AI_PROFANITY_UNAVAILABLE)
            except Exception:
                log.debug("on_warning callback failed", exc_info=True)

        # --- Phase 3: Merge and deduplicate ---
        all_detections = audio_detections + profanity_detections
        all_detections = _deduplicate(all_detections)

        duration = self._compute_duration(audio_path, segments, all_detections)

        _progress(100, tr(S.PIPE_DONE))

        report = AnalysisReport(
            audio_path=str(audio_path),
            duration_seconds=duration,
            segments=segments,
            detections=all_detections,
        )

        # Persist analysis results alongside the audio file.
        try:
            report.save_cache()
        except OSError as exc:
            log.warning("Could not save analysis cache: %s", exc)

        # Export transcript as plain text.
        try:
            _save_transcript_txt(audio_path, segments, all_detections)
        except OSError as exc:
            log.warning("Could not save transcript.txt: %s", exc)

        return report

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_parallel_phase(
        self,
        audio_path: Path,
        progress_fn: Callable[[int, str], None],
        task_fn: Callable[[int, int, str], None],
        on_partial_stt: Optional[PartialSttCallback] = None,
        on_partial_events: Optional[PartialEventsCallback] = None,
    ) -> tuple[List[TranscribedSegment], List[Detection]]:
        """Run STT and audio event detection in parallel threads.

        Args:
            audio_path: Path to the audio file.
            progress_fn: Overall progress callback.
            task_fn: Per-task progress callback(task_id, pct, label).
            on_partial_stt: Callback for incremental STT segments.
            on_partial_events: Callback for incremental event detections.

        Returns:
            Tuple of (segments, audio_event_detections).
        """
        progress_fn(30, tr(S.PIPE_PARALLEL))

        # Check for intermediate caches from a previous interrupted run.
        cached_stt = _load_intermediate_stt(audio_path)
        cached_events = _load_intermediate_events(audio_path)

        if cached_stt is not None:
            log.info("Loaded cached STT: %d segments from previous run.", len(cached_stt))
            task_fn(0, 100, f"{tr(S.TASK_STT)}: {len(cached_stt)}")
            # Send cached STT to GUI immediately so partial UI shows progress.
            if on_partial_stt is not None:
                try:
                    on_partial_stt([_seg_to_dict(s) for s in cached_stt])
                except Exception:
                    log.debug("on_partial_stt callback failed for cached STT", exc_info=True)
        else:
            task_fn(0, 1, tr(S.PIPE_STT_START))

        if cached_events is not None:
            log.info("Loaded cached events: %d detections from previous run.", len(cached_events))
            task_fn(1, 100, f"{tr(S.TASK_AUDIO_EVENTS)}: {len(cached_events)}")
            # Send cached events to GUI immediately so partial UI shows progress.
            if on_partial_events is not None:
                try:
                    on_partial_events([
                        {
                            "type": d.type.value,
                            "start": round(d.start, 2),
                            "end": round(d.end, 2),
                            "confidence": round(d.confidence, 3),
                            "details": d.details,
                        }
                        for d in cached_events
                    ])
                except Exception:
                    log.debug("on_partial_events callback failed for cached events", exc_info=True)
        else:
            task_fn(1, 1, tr(S.PIPE_EVENTS_START))

        # If both are cached, skip the parallel phase entirely.
        if cached_stt is not None and cached_events is not None:
            return cached_stt, cached_events

        # Progress callbacks for each task, forwarded to task_fn.
        def _stt_progress(pct: int, label: str) -> None:
            task_fn(0, max(1, pct), label)

        def _events_progress(pct: int, label: str) -> None:
            task_fn(1, max(1, pct), label)

        # --- Time-based batching for incremental STT segments ---
        # Accumulate segments and flush to callback every _STT_FLUSH_INTERVAL
        # seconds to avoid flooding the IPC queue.
        _STT_FLUSH_INTERVAL = 5.0  # seconds between flushes
        _stt_batch: List[TranscribedSegment] = []
        _stt_all_segments: List[TranscribedSegment] = []  # all segments received so far
        _stt_last_flush = [time.perf_counter()]  # mutable for closure
        _stt_lock = threading.Lock()

        def _on_stt_segment(seg: TranscribedSegment) -> None:
            """Accumulate STT segments and flush periodically."""
            with _stt_lock:
                _stt_all_segments.append(seg)
                _stt_batch.append(seg)
                now = time.perf_counter()
                if now - _stt_last_flush[0] >= _STT_FLUSH_INTERVAL:
                    _flush_stt_batch_locked()

        def _flush_stt_batch() -> None:
            """Acquire lock and flush."""
            with _stt_lock:
                _flush_stt_batch_locked()

        def _flush_stt_batch_locked() -> None:
            """Send accumulated STT segments via partial callback and save
            intermediate cache. Caller holds _stt_lock."""
            if not _stt_batch:
                return
            # Save all segments received so far to intermediate cache.
            _save_intermediate_stt(audio_path, list(_stt_all_segments))
            batch = [_seg_to_dict(s) for s in _stt_batch]
            _stt_batch.clear()
            _stt_last_flush[0] = time.perf_counter()
            if on_partial_stt is not None:
                try:
                    on_partial_stt(batch)
                except Exception:
                    log.debug("on_partial_stt callback failed", exc_info=True)

        # --- Incremental saving for PANNs chunk detections ---
        _EVENTS_SAVE_INTERVAL = 5.0  # seconds between saves
        _events_all_detections: List[Detection] = []
        _events_last_save = [time.perf_counter()]

        def _on_chunk_dets(dets: list) -> None:
            _events_all_detections.extend(dets)
            # Periodically save to intermediate cache.
            now = time.perf_counter()
            if now - _events_last_save[0] >= _EVENTS_SAVE_INTERVAL:
                _save_intermediate_events(audio_path, list(_events_all_detections))
                _events_last_save[0] = now
            if on_partial_events is None:
                return
            try:
                on_partial_events([
                    {
                        "type": d.type.value,
                        "start": round(d.start, 2),
                        "end": round(d.end, 2),
                        "confidence": round(d.confidence, 3),
                        "details": d.details,
                    }
                    for d in dets
                ])
            except Exception:
                log.debug("on_partial_events callback failed", exc_info=True)

        segments: List[TranscribedSegment] = []
        audio_detections: List[Detection] = []

        # ThreadPoolExecutor runs both tasks concurrently. STT is CPU/GPU-
        # bound; PANNs is also CPU-bound but on a different model.
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {}
            if cached_stt is None:
                futures[pool.submit(
                    self._stt.transcribe, audio_path,
                    on_progress=_stt_progress,
                    on_segment=_on_stt_segment,
                )] = "stt"
            if cached_events is None:
                futures[pool.submit(
                    self._audio_events.detect, audio_path,
                    on_progress=_events_progress,
                    on_chunk_detections=_on_chunk_dets,
                )] = "events"

            for future in as_completed(futures):
                task_name = futures[future]
                if task_name == "stt":
                    segments = future.result()
                    # Flush any remaining buffered segments.
                    _flush_stt_batch()
                    _save_intermediate_stt(audio_path, segments, completed=True)
                    task_fn(0, 100, f"{tr(S.TASK_STT)}: {len(segments)}")
                    progress_fn(50, f"{tr(S.TASK_STT)}: {len(segments)}")
                else:
                    audio_detections = future.result()
                    _save_intermediate_events(audio_path, audio_detections)
                    task_fn(1, 100, f"{tr(S.TASK_AUDIO_EVENTS)}: {len(audio_detections)}")
                    progress_fn(
                        60,
                        f"{tr(S.TASK_AUDIO_EVENTS)}: {len(audio_detections)}",
                    )

        # Merge cached + freshly computed.
        if cached_stt is not None:
            segments = cached_stt
        if cached_events is not None:
            audio_detections = cached_events

        return segments, audio_detections

    @staticmethod
    def _compute_duration(
        audio_path: Path,
        segments: List[TranscribedSegment],
        detections: List[Detection],
    ) -> float:
        """Determine audio duration.

        Tries to read the actual duration from the WAV header first.
        Falls back to estimating from the latest segment/detection end
        time if the header cannot be read.

        Args:
            audio_path: Path to the audio file.
            segments: Transcribed segments.
            detections: All detections.

        Returns:
            Duration in seconds.
        """
        # Try reading actual duration from WAV header.
        try:
            import wave
            with wave.open(str(audio_path), "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                if rate > 0:
                    return frames / rate
        except Exception:
            log.debug("Could not read WAV header for duration; estimating from content.")

        # Fallback: estimate from content.
        duration = 0.0
        if segments:
            duration = max(segment.end for segment in segments)
        if detections:
            duration = max(duration, max(detection.end for detection in detections))
        return duration


# ===========================
# MODULE-LEVEL HELPERS
# ===========================


def _seg_to_dict(seg: TranscribedSegment) -> dict:
    """Convert a TranscribedSegment to a plain dict for IPC serialisation."""
    return {
        "text": seg.text,
        "start": round(seg.start, 2),
        "end": round(seg.end, 2),
        "words": [
            {
                "word": w.word,
                "start": round(w.start, 2),
                "end": round(w.end, 2),
                "confidence": round(w.confidence, 3),
            }
            for w in seg.words
        ],
    }


def _deduplicate(detections: List[Detection]) -> List[Detection]:
    """Remove overlapping detections of the same type, keeping higher confidence.

    Two detections of the same DetectionType whose time ranges overlap are
    merged into a single detection spanning the union of both ranges.

    Args:
        detections: Unsorted list of Detection objects.

    Returns:
        Deduplicated list sorted by start time.
    """
    if not detections:
        return detections

    detections.sort(key=lambda detection: (detection.type.value, detection.start))
    result: List[Detection] = []

    for detection in detections:
        merged = False
        # Walk backwards through result to find a same-type overlap.
        for index in range(len(result) - 1, -1, -1):
            previous = result[index]
            if previous.type != detection.type:
                continue
            # Check for time overlap.
            if detection.start < previous.end and detection.end > previous.start:
                result[index] = Detection(
                    type=previous.type,
                    start=min(previous.start, detection.start),
                    end=max(previous.end, detection.end),
                    confidence=max(previous.confidence, detection.confidence),
                    details={**previous.details, **detection.details},
                )
                merged = True
                break
        if not merged:
            result.append(detection)

    result.sort(key=lambda detection: detection.start)
    return result


# ===========================
# INTERMEDIATE CACHE HELPERS
# ===========================


def _artifact_dir(audio_path: Path) -> Path:
    """Return the artifact directory for an audio file, creating it if needed."""
    d = audio_path.parent / audio_path.stem
    d.mkdir(parents=True, exist_ok=True)
    return d


def _stt_cache_path(audio_path: Path) -> Path:
    """Return the intermediate STT cache file path (new folder layout)."""
    return _artifact_dir(audio_path) / "stt_cache.json"


def _old_stt_cache_path(audio_path: Path) -> Path:
    """Legacy STT cache path."""
    return audio_path.with_suffix(audio_path.suffix + ".stt_cache.json")


def _events_cache_path(audio_path: Path) -> Path:
    """Return the intermediate audio events cache file path (new folder layout)."""
    return _artifact_dir(audio_path) / "events_cache.json"


def _old_events_cache_path(audio_path: Path) -> Path:
    """Legacy events cache path."""
    return audio_path.with_suffix(audio_path.suffix + ".events_cache.json")


def _save_intermediate_stt(
    audio_path: Path,
    segments: List[TranscribedSegment],
    *,
    completed: bool = False,
) -> None:
    """Save STT results to an intermediate cache file.

    Args:
        audio_path: Path to the audio file.
        segments: Transcribed segments so far.
        completed: True only when STT finished the entire file.
    """
    cache_path = _stt_cache_path(audio_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = {
            "completed": completed,
            "segments": [
                {
                    "text": seg.text,
                    "start": round(seg.start, 2),
                    "end": round(seg.end, 2),
                    "words": [
                        {
                            "word": w.word,
                            "start": round(w.start, 2),
                            "end": round(w.end, 2),
                            "confidence": round(w.confidence, 3),
                        }
                        for w in seg.words
                    ],
                }
                for seg in segments
            ],
        }
        tmp = cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(cache_path)
        log.info("Intermediate STT cache saved: %d segments (completed=%s) -> %s",
                 len(segments), completed, cache_path.name)
    except OSError as exc:
        log.warning("Could not save STT cache: %s", exc)


def _load_intermediate_stt(audio_path: Path) -> Optional[List[TranscribedSegment]]:
    """Load previously cached STT results, or None if not available.

    Only returns segments when the cache is marked as *completed* — i.e.
    the STT pass finished the entire file.  Incomplete (interrupted) caches
    are discarded so the pipeline re-runs STT from scratch.
    """
    from .models import TranscribedWord

    cache_path = _stt_cache_path(audio_path)
    # Fall back to legacy path and migrate.
    if not cache_path.exists():
        old = _old_stt_cache_path(audio_path)
        if old.exists():
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                old.rename(cache_path)
                log.info("Migrated STT cache %s → %s", old, cache_path)
            except OSError:
                cache_path = old
    if not cache_path.exists():
        return None
    # Invalidate if audio is newer than cache.
    try:
        if audio_path.stat().st_mtime > cache_path.stat().st_mtime:
            log.info("STT cache stale; will re-run STT.")
            return None
    except OSError:
        return None
    try:
        raw = json.loads(cache_path.read_text(encoding="utf-8"))

        # Support both old format (bare list) and new format (dict with
        # "completed" flag and "segments" list).
        if isinstance(raw, list):
            # Legacy format — no completeness info; treat as incomplete.
            log.info("STT cache is legacy format (no completed flag); will re-run STT.")
            return None
        elif isinstance(raw, dict):
            if not raw.get("completed", False):
                log.info("STT cache exists but is incomplete (%d segments); will re-run STT.",
                         len(raw.get("segments", [])))
                return None
            seg_list = raw.get("segments", [])
        else:
            return None

        segments = [
            TranscribedSegment(
                text=seg["text"],
                start=seg["start"],
                end=seg["end"],
                words=[
                    TranscribedWord(
                        word=w["word"], start=w["start"],
                        end=w["end"], confidence=w.get("confidence", 1.0),
                    )
                    for w in seg.get("words", [])
                ],
            )
            for seg in seg_list
        ]
        log.info("Loaded completed STT cache: %d segments.", len(segments))
        return segments
    except (json.JSONDecodeError, KeyError, OSError) as exc:
        log.warning("Failed to load STT cache: %s", exc)
        return None


def _save_intermediate_events(audio_path: Path, detections: List[Detection]) -> None:
    """Save audio event detections to an intermediate cache file."""
    cache_path = _events_cache_path(audio_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = [
            {
                "type": det.type.value,
                "start": round(det.start, 2),
                "end": round(det.end, 2),
                "confidence": round(det.confidence, 3),
                "details": det.details,
            }
            for det in detections
        ]
        tmp = cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(cache_path)
        log.info("Intermediate events cache saved: %d detections -> %s", len(detections), cache_path.name)
    except OSError as exc:
        log.warning("Could not save events cache: %s", exc)


def _load_intermediate_events(audio_path: Path) -> Optional[List[Detection]]:
    """Load previously cached audio event detections, or None if not available."""
    cache_path = _events_cache_path(audio_path)
    # Fall back to legacy path and migrate.
    if not cache_path.exists():
        old = _old_events_cache_path(audio_path)
        if old.exists():
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                old.rename(cache_path)
                log.info("Migrated events cache %s → %s", old, cache_path)
            except OSError:
                cache_path = old
    if not cache_path.exists():
        return None
    try:
        if audio_path.stat().st_mtime > cache_path.stat().st_mtime:
            log.info("Events cache stale; will re-run event detection.")
            return None
    except OSError:
        return None
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        detections = [
            Detection(
                type=DetectionType(det["type"]),
                start=det["start"],
                end=det["end"],
                confidence=det.get("confidence", 1.0),
                details=det.get("details", {}),
            )
            for det in data
        ]
        log.info("Loaded intermediate events cache: %d detections.", len(detections))
        return detections
    except (json.JSONDecodeError, KeyError, ValueError, OSError) as exc:
        log.warning("Failed to load events cache: %s", exc)
        return None


def _remove_intermediate_caches(audio_path: Path) -> None:
    """Remove intermediate cache files (both new and legacy locations)."""
    for path in [_stt_cache_path(audio_path), _events_cache_path(audio_path),
                 _old_stt_cache_path(audio_path), _old_events_cache_path(audio_path)]:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


# Emoji markers for detection types in transcript exports.
from .models import DETECTION_LABELS_HE

_DETECTION_EMOJI = {
    DetectionType.PROFANITY: "🤬",
    DetectionType.SHOUT: "🗣️",
    DetectionType.SCREAM: "😱",
    DetectionType.CRY: "😢",
    DetectionType.WAIL: "😭",
    DetectionType.BABY_CRY: "👶",
    DetectionType.LAUGHTER: "😂",
    DetectionType.VOLUME_SPIKE: "🔊",
}


def _format_hhmmss(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def _save_transcript_txt(
    audio_path: Path,
    segments: List[TranscribedSegment],
    detections: List[Detection],
) -> None:
    """Export a combined transcript + event markers as plain text.

    All detection types are included regardless of the current GUI
    filter state.
    """
    # Merge segments and detections by start time.
    items: list[tuple[float, str, str]] = []  # (start, kind, text)
    for seg in segments:
        items.append((seg.start, "seg", seg.text.strip()))
    for det in detections:
        emoji = _DETECTION_EMOJI.get(det.type, "⚠️")
        label = DETECTION_LABELS_HE.get(det.type, det.type.value)
        items.append((det.start, "det", f"[{emoji} {label}]"))

    items.sort(key=lambda x: x[0])

    lines: list[str] = []
    for start, kind, text in items:
        ts = _format_hhmmss(start)
        lines.append(f"[{ts}] {text}")

    out_path = _artifact_dir(audio_path) / "transcript.txt"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Transcript exported to %s (%d lines).", out_path, len(lines))

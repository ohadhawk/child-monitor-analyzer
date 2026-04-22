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
        stt_model_key: str = "thorough",
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
            stt_model_key,
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
        stt_model_key: str = "thorough",
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
        cached = AnalysisReport.load_cache(audio_path, stt_model_key)
        if cached is not None and is_gap_fill_complete(audio_path, stt_model_key):
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
            sub_fn=_sub,
            on_partial_stt=on_partial_stt,
            on_partial_events=on_partial_events,
            stt_model_key=stt_model_key,
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
            stt_model_key=stt_model_key,
        )

        # Persist analysis results alongside the audio file.
        try:
            report.save_cache()
        except OSError as exc:
            log.warning("Could not save analysis cache: %s", exc)

        # Export transcript as plain text.
        try:
            _save_transcript_txt(audio_path, segments, all_detections, stt_model_key)
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
        sub_fn: Optional[SubProgressCallback] = None,
        on_partial_stt: Optional[PartialSttCallback] = None,
        on_partial_events: Optional[PartialEventsCallback] = None,
        stt_model_key: str = "thorough",
    ) -> tuple[List[TranscribedSegment], List[Detection]]:
        """Run STT and audio event detection in parallel threads.

        Args:
            audio_path: Path to the audio file.
            progress_fn: Overall progress callback.
            task_fn: Per-task progress callback(task_id, pct, label).
            sub_fn: Sub-progress callback for gap-fill bar (slot 0).
            on_partial_stt: Callback for incremental STT segments.
            on_partial_events: Callback for incremental event detections.

        Returns:
            Tuple of (segments, audio_event_detections).
        """
        progress_fn(30, tr(S.PIPE_PARALLEL))

        # Check for intermediate caches from a previous interrupted run.
        cached_stt_result = _load_intermediate_stt(audio_path, stt_model_key)
        cached_events = _load_intermediate_events(audio_path)

        # Unpack STT cache result.
        cached_stt: Optional[List[TranscribedSegment]] = None
        cached_vad_completed = False
        cached_gap_fill_completed = False
        cached_gap_fill_boundaries: List[list] = []
        cached_gap_fill_done: List[list] = []
        cached_vad_timestamps: Optional[List[dict]] = None
        need_gap_fill_only = False
        # Segments from an incomplete (interrupted) VAD run to pass to
        # transcribe() so it can reuse them instead of re-transcribing.
        incomplete_cached_segments: Optional[List[TranscribedSegment]] = None

        if cached_stt_result is not None:
            cached_stt = cached_stt_result["segments"]
            cached_vad_completed = cached_stt_result.get("vad_completed", True)
            cached_gap_fill_completed = cached_stt_result["gap_fill_completed"]
            cached_gap_fill_boundaries = cached_stt_result.get("gap_fill_boundaries", [])
            cached_gap_fill_done = cached_stt_result.get("gap_fill_done", [])
            cached_vad_timestamps = cached_stt_result.get("vad_timestamps")

            if not cached_vad_completed:
                # VAD was interrupted — we have partial segments. Re-run
                # transcribe() but pass cached segments to skip re-processing.
                incomplete_cached_segments = list(cached_stt)
                cached_stt = None  # treat as needing full re-run
                log.info("Loaded incomplete STT cache: %d segments from interrupted VAD. "
                         "Will resume transcription reusing cached segments.",
                         len(incomplete_cached_segments))
                task_fn(0, 1, f"{tr(S.PIPE_STT_START)} ({len(incomplete_cached_segments)})")
            elif cached_gap_fill_completed:
                # Full STT + gap-fill done — use as-is.
                log.info("Loaded cached STT: %d segments (gap-fill complete).", len(cached_stt))
                task_fn(0, 100, f"{tr(S.TASK_STT)}: {len(cached_stt)}")
            elif not cached_gap_fill_boundaries:
                # VAD done but no gap boundaries stored — nothing to gap-fill.
                # Treat as complete and persist so we don't re-run next time.
                cached_gap_fill_completed = True
                _save_intermediate_stt(
                    audio_path, list(cached_stt), completed=True,
                    gap_fill_completed=True,
                    gap_fill_boundaries=[], gap_fill_done=[],
                    stt_model_key=stt_model_key,
                    vad_timestamps=cached_vad_timestamps,
                )
                log.info("Loaded cached STT: %d segments (no gap boundaries — marked complete).",
                         len(cached_stt))
                task_fn(0, 100, f"{tr(S.TASK_STT)}: {len(cached_stt)}")
            else:
                # VAD done but gap-fill incomplete — need to run gap-fill only.
                need_gap_fill_only = True
                log.info("Loaded cached STT: %d segments (gap-fill incomplete: %d/%d gaps done). "
                         "Will resume gap-fill.",
                         len(cached_stt), len(cached_gap_fill_done),
                         len(cached_gap_fill_boundaries))
                task_fn(0, 90, f"{tr(S.TASK_STT)}: {len(cached_stt)}")

            # Send cached STT to GUI immediately so partial UI shows progress.
            # For incomplete VAD, the send happens later (after pre-populating
            # _stt_all_segments).  For gap-fill-only, it's sent from the
            # need_gap_fill_only block below.
            if cached_stt is not None and not need_gap_fill_only and on_partial_stt is not None:
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

        # If both are fully cached, skip the parallel phase entirely.
        if cached_stt is not None and not need_gap_fill_only and cached_events is not None:
            return cached_stt, cached_events

        # Progress callbacks for each task, forwarded to task_fn.
        # Also update the main progress bar proportionally during the
        # parallel phase (30-50% for STT, 50-60% for events).
        def _stt_progress(pct: int, label: str) -> None:
            task_fn(0, max(1, pct), label)
            # Interpolate main bar: 30% (start) + pct/90 * 20% = 30-50%
            main_pct = 30 + int(min(pct, 90) / 90 * 20)
            progress_fn(main_pct, label)

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
        _vad_completed = [False]  # set True after VAD pass; keeps saves as completed
        _gap_fill_completed = [False]  # set True when gap-fill finishes (or no gaps)

        # When resuming gap-fill only, pre-populate with cached segments
        # so intermediate saves don't overwrite the completed cache.
        # Also populate _cached_start_times so _on_stt_segment skips
        # segments that Whisper re-discovers during its VAD pass.
        _cached_start_times: set = set()
        if need_gap_fill_only and cached_stt:
            _stt_all_segments.extend(cached_stt)
            _vad_completed[0] = True
            _cached_start_times = {round(s.start, 2) for s in cached_stt}
            # Send cached segments to GUI so transcript is visible while
            # gap-fill runs.
            if on_partial_stt is not None:
                try:
                    on_partial_stt([_seg_to_dict(s) for s in cached_stt])
                except Exception:
                    log.debug("on_partial_stt callback failed for gap-fill cached STT", exc_info=True)

        # When resuming from an incomplete VAD run, pre-populate with
        # cached segments so they're preserved in intermediate saves and
        # shown immediately.  Track their start times to deduplicate when
        # Whisper re-yields segments covering the same time range.
        if incomplete_cached_segments:
            _stt_all_segments.extend(incomplete_cached_segments)
            _cached_start_times = {round(s.start, 2) for s in incomplete_cached_segments}
            log.info("Pre-populated %d cached segments for resume; "
                     "new segments with matching start times will be skipped.",
                     len(incomplete_cached_segments))
            # Send cached segments to GUI immediately.
            if on_partial_stt is not None:
                try:
                    on_partial_stt([_seg_to_dict(s) for s in incomplete_cached_segments])
                except Exception:
                    log.debug("on_partial_stt callback failed for incomplete cached STT", exc_info=True)

        # Gap-fill tracking (mutable closures for thread-safe access).
        _gap_fill_boundaries: List[list] = list(cached_gap_fill_boundaries)
        _gap_fill_done: List[list] = list(cached_gap_fill_done)

        def _on_stt_segment(seg: TranscribedSegment) -> None:
            """Accumulate STT segments and flush periodically."""
            with _stt_lock:
                # Skip segments already covered by cached data from a prior run.
                if round(seg.start, 2) in _cached_start_times:
                    return
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
            # Once the VAD pass is done, mark saves as completed so that
            # an interrupted gap-fill still has usable VAD results.
            _save_intermediate_stt(
                audio_path, list(_stt_all_segments),
                completed=_vad_completed[0],
                gap_fill_completed=_gap_fill_completed[0],
                gap_fill_boundaries=_gap_fill_boundaries if _vad_completed[0] else None,
                gap_fill_done=_gap_fill_done if _vad_completed[0] else None,
                stt_model_key=stt_model_key,
                vad_timestamps=_cached_vad_ts or None,
            )
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

        # --- Gap-fill callbacks ---
        _cached_vad_ts: List[dict] = list(cached_vad_timestamps) if cached_vad_timestamps else []

        def _on_vad_timestamps(timestamps: list) -> None:
            """Cache raw VAD timestamps for reuse on resume."""
            _cached_vad_ts.clear()
            _cached_vad_ts.extend(timestamps)
            log.info("VAD timestamps received: %d speech regions.", len(timestamps))

        def _on_gap_fill_progress(done_s: int, total_s: int, label: str) -> None:
            """Forward gap-fill progress to the sub-progress bar (slot 0)."""
            if sub_fn:
                sub_fn(done_s, total_s, label)

        def _on_boundaries_computed(boundaries: list) -> None:
            """Update gap-fill boundaries with the full set computed by
            _fill_gaps (includes trailing gap that _on_vad_done cannot see).
            Persist immediately so a crash mid-gap-fill preserves the full
            boundary list for correct resume."""
            with _stt_lock:
                _gap_fill_boundaries.clear()
                _gap_fill_boundaries.extend(
                    [round(s, 2), round(e, 2)] for s, e in boundaries
                )
                _save_intermediate_stt(
                    audio_path, list(_stt_all_segments),
                    completed=True,
                    gap_fill_completed=False,
                    gap_fill_boundaries=_gap_fill_boundaries,
                    gap_fill_done=list(_gap_fill_done),
                    stt_model_key=stt_model_key,
                    vad_timestamps=_cached_vad_ts or None,
                )
            log.info("Gap-fill boundaries updated and persisted: %d (including trailing gap).",
                     len(_gap_fill_boundaries))

        def _on_gap_done(gap_start: float, gap_end: float) -> None:
            """Persist gap-fill progress after each completed gap."""
            with _stt_lock:
                _gap_fill_done.append([round(gap_start, 2), round(gap_end, 2)])
                # Mark complete if all known gaps are done.
                if len(_gap_fill_done) >= len(_gap_fill_boundaries) and _gap_fill_boundaries:
                    _gap_fill_completed[0] = True
                _save_intermediate_stt(
                    audio_path, list(_stt_all_segments),
                    completed=True,
                    gap_fill_completed=_gap_fill_completed[0],
                    gap_fill_boundaries=_gap_fill_boundaries,
                    gap_fill_done=list(_gap_fill_done),
                    stt_model_key=stt_model_key,
                    vad_timestamps=_cached_vad_ts or None,
                )
            log.info("Gap %s–%s complete; saved progress (%d/%d gaps done).",
                     _format_hhmmss(gap_start), _format_hhmmss(gap_end),
                     len(_gap_fill_done), len(_gap_fill_boundaries))

        # When resuming gap-fill, extract previously-completed gap-fill
        # segments from the cache so they can be merged back after the
        # fresh transcribe() call (which only produces VAD + new gap-fill).
        _cached_gf_segments: List[TranscribedSegment] = []
        if need_gap_fill_only and cached_stt and cached_gap_fill_done:
            done_set = {(round(s, 2), round(e, 2))
                        for s, e in cached_gap_fill_done}
            for seg in cached_stt:
                for gs, ge in done_set:
                    if gs <= seg.start < ge:
                        _cached_gf_segments.append(seg)
                        break
            log.info("Extracted %d cached gap-fill segments for merge.",
                     len(_cached_gf_segments))

        segments: List[TranscribedSegment] = []
        audio_detections: List[Detection] = []

        # ThreadPoolExecutor runs both tasks concurrently. STT is CPU/GPU-
        # bound; PANNs is also CPU-bound but on a different model.
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {}
            def _on_vad_done(segs: list) -> None:
                """Persist a completed snapshot after the VAD pass so that
                an interrupted gap-fill does not lose the VAD results.
                Also compute and store gap-fill boundaries for restart."""
                _vad_completed[0] = True

                # When resuming gap-fill only, boundaries are already loaded
                # from cache — don't recompute (would lose the trailing gap
                # and could mismatch previously completed gaps).
                if not need_gap_fill_only:
                    # Compute inter-segment gap boundaries from the segment list.
                    # This is a preliminary set — _fill_gaps() will later call
                    # _on_boundaries_computed with the full set (including
                    # trailing gap that we can't compute here without total_duration).
                    _GAP_MIN = 5 * 60  # must match stt._GAP_FILL_MIN_SECONDS
                    sorted_segs = sorted(segs, key=lambda s: s.start) if segs else []
                    boundaries: List[list] = []
                    if sorted_segs:
                        if sorted_segs[0].start > _GAP_MIN:
                            boundaries.append([round(0.0, 2), round(sorted_segs[0].start, 2)])
                        for i in range(1, len(sorted_segs)):
                            gap = sorted_segs[i].start - sorted_segs[i - 1].end
                            if gap >= _GAP_MIN:
                                boundaries.append([round(sorted_segs[i - 1].end, 2),
                                                   round(sorted_segs[i].start, 2)])
                    with _stt_lock:
                        _gap_fill_boundaries.clear()
                        _gap_fill_boundaries.extend(boundaries)

                _flush_stt_batch()
                # Save _stt_all_segments (not segs) so the flush includes
                # everything accumulated via _on_stt_segment.
                with _stt_lock:
                    all_segs = list(_stt_all_segments)

                # Don't mark gap-fill complete here — _fill_gaps() may discover
                # a trailing gap not captured in _gap_fill_boundaries.  The
                # final save after transcribe() returns will set it True.
                _save_intermediate_stt(audio_path, all_segs, completed=True,
                                      gap_fill_completed=_gap_fill_completed[0],
                                      gap_fill_boundaries=_gap_fill_boundaries,
                                      gap_fill_done=list(_gap_fill_done),
                                      stt_model_key=stt_model_key,
                                      vad_timestamps=_cached_vad_ts or None)
                log.info("VAD pass complete — saved %d segments, %d inter-segment gap boundaries.",
                         len(all_segs), len(_gap_fill_boundaries))

            skip_gaps_for_stt = [tuple(g) for g in cached_gap_fill_done] if need_gap_fill_only else None

            # For gap-fill-only mode, pass the cached segments so transcribe()
            # can skip re-processing them during the VAD pass.
            _cached_for_transcribe = list(cached_stt) if (need_gap_fill_only and cached_stt) else incomplete_cached_segments

            if cached_stt is None or need_gap_fill_only:
                futures[pool.submit(
                    self._stt.transcribe, audio_path,
                    on_progress=_stt_progress,
                    on_segment=_on_stt_segment,
                    on_vad_done=_on_vad_done,
                    on_gap_fill_progress=_on_gap_fill_progress,
                    on_gap_done=_on_gap_done,
                    on_boundaries_computed=_on_boundaries_computed,
                    skip_gaps=skip_gaps_for_stt,
                    cached_segments=_cached_for_transcribe,
                    gap_fill_only=need_gap_fill_only,
                    on_vad_timestamps=_on_vad_timestamps,
                    vad_timestamps=cached_vad_timestamps,
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
                    # Merge in previously-completed gap-fill segments
                    # from the cache (not re-discovered by the fresh run).
                    if _cached_gf_segments:
                        existing_starts = {round(s.start, 2) for s in segments}
                        merged = [s for s in _cached_gf_segments
                                  if round(s.start, 2) not in existing_starts]
                        if merged:
                            segments = sorted(segments + merged,
                                              key=lambda s: s.start)
                            log.info("Merged %d cached gap-fill segments.", len(merged))
                    # Flush any remaining buffered segments.
                    _flush_stt_batch()
                    _save_intermediate_stt(
                        audio_path, segments, completed=True,
                        gap_fill_completed=True,
                        gap_fill_boundaries=_gap_fill_boundaries,
                        gap_fill_done=_gap_fill_boundaries,  # all done
                        stt_model_key=stt_model_key,
                        vad_timestamps=_cached_vad_ts or None,
                    )
                    # Hide gap-fill sub-progress bar.
                    if sub_fn:
                        sub_fn(-1, -1, "")
                    task_fn(0, 100, f"{tr(S.TASK_STT)}: {len(segments)}")
                    progress_fn(50, f"{tr(S.TASK_STT)}: {len(segments)}")
                else:
                    audio_detections = future.result()
                    _save_intermediate_events(audio_path, audio_detections)
                    # Send the full detection list (including volume spikes
                    # that weren't streamed via on_chunk_detections) to the GUI.
                    if on_partial_events:
                        try:
                            on_partial_events([
                                {
                                    "type": d.type.value,
                                    "start": round(d.start, 2),
                                    "end": round(d.end, 2),
                                    "confidence": round(d.confidence, 3),
                                    "details": d.details,
                                }
                                for d in audio_detections
                            ])
                        except Exception:
                            log.debug("on_partial_events (final) callback failed", exc_info=True)
                    task_fn(1, 100, f"{tr(S.TASK_AUDIO_EVENTS)}: {len(audio_detections)}")
                    progress_fn(
                        60,
                        f"{tr(S.TASK_AUDIO_EVENTS)}: {len(audio_detections)}",
                    )

        # Merge cached + freshly computed.
        if cached_stt is not None and not need_gap_fill_only:
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


def _stt_cache_path(audio_path: Path, stt_model_key: str = "thorough") -> Path:
    """Return the intermediate STT cache file path (per-model)."""
    return _artifact_dir(audio_path) / f"stt_cache_{stt_model_key}.json"


def has_intermediate_stt_cache(audio_path: Path, stt_model_key: str = "thorough") -> bool:
    """Return True if an intermediate STT cache file exists for this audio/model."""
    cache = audio_path.parent / audio_path.stem / f"stt_cache_{stt_model_key}.json"
    return cache.exists()


def is_gap_fill_complete(audio_path: Path, stt_model_key: str = "thorough") -> bool:
    """Return True if the STT cache marks gap-fill as completed.

    Used by the GUI to decide whether to start analysis for gap-fill
    even when a full analysis cache already exists.
    """
    cache_file = _stt_cache_path(audio_path, stt_model_key)
    if not cache_file.exists():
        # Also check for un-suffixed legacy cache (stt_cache.json) in the
        # artifact dir — old code wrote there before per-model naming.
        legacy = _artifact_dir(audio_path) / "stt_cache.json"
        if legacy.exists():
            return False  # legacy cache has no gap-fill tracking → needs it
        # No STT cache at all.  If an analysis cache exists, it was produced
        # by old code that didn't write a separate STT cache — gap-fill was
        # never tracked, so we must re-run.
        from .models import AnalysisReport
        if AnalysisReport.load_cache(audio_path, stt_model_key) is not None:
            return False
        return True  # genuinely no prior work — nothing to gap-fill
    try:
        import json
        raw = json.loads(cache_file.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            return False  # bare-list legacy format — no gap-fill tracking
        if not raw.get("completed", False):
            return False  # VAD not done yet — needs full re-run
        return raw.get("gap_fill_completed", False)
    except Exception:
        log.warning("STT cache corrupt or unreadable; will re-run analysis.")
        return False  # corrupt cache — re-run to fix


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
    gap_fill_completed: bool = False,
    gap_fill_boundaries: Optional[List[list]] = None,
    gap_fill_done: Optional[List[list]] = None,
    stt_model_key: str = "thorough",
    vad_timestamps: Optional[List[dict]] = None,
) -> None:
    """Save STT results to an intermediate cache file.

    Args:
        audio_path: Path to the audio file.
        segments: Transcribed segments so far.
        completed: True when the VAD pass finished the entire file.
        gap_fill_completed: True when gap-fill also finished.
        gap_fill_boundaries: Original gap boundaries found after VAD.
        gap_fill_done: Subset of boundaries already scanned.
        stt_model_key: Model key for per-model cache separation.
        vad_timestamps: Raw Silero-VAD speech timestamps for reuse on resume.
    """
    cache_path = _stt_cache_path(audio_path, stt_model_key)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        data: dict = {
            "completed": completed,
            "gap_fill_completed": gap_fill_completed,
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
        if gap_fill_boundaries is not None:
            data["gap_fill_boundaries"] = gap_fill_boundaries
        if gap_fill_done is not None:
            data["gap_fill_done"] = gap_fill_done
        if vad_timestamps is not None:
            data["vad_timestamps"] = vad_timestamps
        tmp = cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(cache_path)
        log.info("Intermediate STT cache saved: %d segments (completed=%s, gf_completed=%s) -> %s",
                 len(segments), completed, gap_fill_completed, cache_path.name)
    except OSError as exc:
        log.warning("Could not save STT cache: %s", exc)


def _load_intermediate_stt(
    audio_path: Path,
    stt_model_key: str = "thorough",
) -> Optional[dict]:
    """Load previously cached STT results, or None if not available.

    Returns a dict with keys:
        "segments": List[TranscribedSegment]
        "vad_completed": bool  -- True if the VAD pass finished
        "gap_fill_completed": bool
        "gap_fill_boundaries": list of [start, end]
        "gap_fill_done": list of [start, end] already scanned

    Returns even incomplete (interrupted during VAD) caches so that
    segments from a prior run can be reused instead of re-transcribed.
    """
    from .models import TranscribedWord

    cache_path = _stt_cache_path(audio_path, stt_model_key)
    # Fall back to legacy path and migrate (only for the default model).
    if not cache_path.exists() and stt_model_key == "thorough":
        old = _old_stt_cache_path(audio_path)
        if old.exists():
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                old.rename(cache_path)
                log.info("Migrated STT cache %s → %s", old, cache_path)
            except OSError:
                cache_path = old
        else:
            # Also check un-suffixed legacy cache in artifact dir.
            unsuffixed = _artifact_dir(audio_path) / "stt_cache.json"
            if unsuffixed.exists():
                try:
                    unsuffixed.rename(cache_path)
                    log.info("Migrated STT cache %s → %s", unsuffixed, cache_path)
                except OSError:
                    cache_path = unsuffixed
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
        vad_completed = False
        if isinstance(raw, list):
            # Legacy format — treat segments as from a completed VAD pass.
            # Convert to dict format so they are reusable.
            seg_list = raw
            vad_completed = True
            log.info("STT cache is legacy format (bare list, %d segments); "
                     "treating as completed VAD.", len(seg_list))
            # Persist the conversion so next load doesn't hit legacy path.
            try:
                new_dict = {"completed": True, "gap_fill_completed": False,
                            "segments": seg_list}
                tmp = cache_path.with_suffix(".tmp")
                tmp.write_text(json.dumps(new_dict, ensure_ascii=False), encoding="utf-8")
                tmp.replace(cache_path)
                log.info("Converted legacy STT cache to dict format on disk.")
            except OSError as e:
                log.warning("Failed to persist legacy cache conversion: %s", e)
        elif isinstance(raw, dict):
            vad_completed = raw.get("completed", False)
            seg_list = raw.get("segments", [])
            if not seg_list:
                if vad_completed:
                    log.info("STT cache is completed but empty; will re-run STT.")
                return None
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

        gap_fill_completed = raw.get("gap_fill_completed", False) if isinstance(raw, dict) else False
        gap_fill_boundaries = raw.get("gap_fill_boundaries", []) if isinstance(raw, dict) else []
        gap_fill_done = raw.get("gap_fill_done", []) if isinstance(raw, dict) else []

        # Legacy caches without gap_fill_completed key: treat as needing
        # gap-fill re-analysis (the old code didn't track gap-fill separately,
        # so we can't know if it actually completed).
        # gap_fill_completed stays False from the default above.

        log.info("Loaded STT cache: %d segments (vad_completed=%s, gap_fill_completed=%s, gaps_done=%d/%d).",
                 len(segments), vad_completed, gap_fill_completed,
                 len(gap_fill_done), len(gap_fill_boundaries))
        return {
            "segments": segments,
            "vad_completed": vad_completed,
            "gap_fill_completed": gap_fill_completed,
            "gap_fill_boundaries": gap_fill_boundaries,
            "gap_fill_done": gap_fill_done,
            "vad_timestamps": raw.get("vad_timestamps") if isinstance(raw, dict) else None,
        }
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


def _remove_intermediate_caches(
    audio_path: Path,
    stt_model_key: str = "thorough",
    *,
    remove_events: bool = True,
) -> None:
    """Remove intermediate cache files for the given model.

    Only removes the STT cache for *stt_model_key* (so the other model's
    cache is preserved). Events cache is model-independent and only
    removed when *remove_events* is True (e.g. full re-analysis).
    """
    paths = [
        _stt_cache_path(audio_path, stt_model_key),
        _old_stt_cache_path(audio_path),
    ]
    if remove_events:
        paths.extend([
            _events_cache_path(audio_path),
            _old_events_cache_path(audio_path),
        ])
    for path in paths:
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
    stt_model_key: str = "thorough",
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

    out_path = _artifact_dir(audio_path) / f"transcript_{stt_model_key}.txt"
    # Remove legacy unkeyed transcript.txt if present.
    legacy = _artifact_dir(audio_path) / "transcript.txt"
    if legacy.exists():
        try:
            legacy.unlink()
        except OSError:
            pass
    out_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Transcript exported to %s (%d lines).", out_path, len(lines))

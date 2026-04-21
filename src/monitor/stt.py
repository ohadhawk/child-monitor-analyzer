"""
Hebrew speech-to-text using faster-whisper with the ivrit-ai model.

Wraps the faster-whisper CTranslate2 engine with the ivrit-ai Hebrew-
fine-tuned Whisper model to produce word-level and segment-level
transcriptions with timestamps.

Usage:
    from monitor.stt import HebrewSTT

    stt = HebrewSTT()
    segments = stt.transcribe("recording.wav")
"""

from __future__ import annotations

import logging
import os
import threading
import time
import wave
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

from .models import TranscribedSegment, TranscribedWord
from .gui.strings import tr, S

log = logging.getLogger(__name__)

# ===========================
# CONSTANTS
# ===========================

# Default model -- Hebrew fine-tuned Whisper from ivrit-ai (Apache-2.0).
# Trained on ~5 000+ hours of Hebrew data (Knesset transcripts, crowd-sourced).
# The non-turbo model has 32 decoder layers (vs 8 in turbo) for better accuracy.
DEFAULT_MODEL = "ivrit-ai/whisper-large-v3-ct2"

# Fast (turbo) model -- 8-decoder-layer distilled Whisper for speed.
TURBO_MODEL = "ivrit-ai/whisper-large-v3-turbo-ct2"

# Mapping: HuggingFace model ID → local subdirectory under models/.
# These are the canonical download locations used by _pre_download_model.
_STT_LOCAL_DIRS: dict[str, str] = {
    DEFAULT_MODEL: "stt",
    TURBO_MODEL: "stt_turbo",
}

# Legacy directory names from earlier installations (checked as fallback).
_STT_LEGACY_DIRS: dict[str, str] = {
    TURBO_MODEL: "stt_turbo_backup",
}


def _format_hhmmss(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    s = int(seconds)
    h, remainder = divmod(s, 3600)
    m, sec = divmod(remainder, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"

# ===========================
# CORE CLASS
# ===========================


class HebrewSTT:
    """Wrapper around faster-whisper for Hebrew transcription with timestamps.

    Lazily loads the model on first call to ``transcribe()`` so that import-
    time stays fast and GPU memory is only allocated when needed.

    Attributes:
        _model_name: HuggingFace model identifier or local path.
        _device: Compute device ("cpu", "cuda", or "auto").
        _compute_type: Quantisation format ("int8", "float16", or "auto").
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "auto",
        compute_type: str = "auto",
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._compute_type = compute_type
        self._model = None
        self._batched_model = None
        self._load_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(
        self,
        on_sub_progress: Optional[Callable[[int, int, str], None]] = None,
    ) -> None:
        """Load the Whisper model if not already loaded.

        Args:
            on_sub_progress: Optional callback(done, total, label) for download
                progress. total=0 means indeterminate.
        """
        with self._load_lock:
            self._load_model_locked(on_sub_progress)

    def _load_model_locked(
        self,
        on_sub_progress: Optional[Callable[[int, int, str], None]] = None,
    ) -> None:
        """Inner model load, called while holding _load_lock."""
        if self._model is not None:
            log.debug("STT model already loaded; skipping.")
            return

        t0 = time.perf_counter()
        # Import here to avoid heavy module-load time when model is not needed.
        import faster_whisper

        log.info(
            "Loading STT model %s (device=%s) -- first run downloads ~1.5 GB...",
            self._model_name,
            self._device,
        )

        device = self._device
        compute_type = self._compute_type

        if device == "auto":
            import torch

            # CTranslate2 (used by faster-whisper) supports CUDA but not XPU.
            # Intel Arc / XPU is not usable for STT; only NVIDIA CUDA is.
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        if compute_type == "auto":
            # float16 is faster on GPU; int8 is the lightest option for CPU.
            compute_type = "float16" if device == "cuda" else "int8"

        # Pre-download the model with progress tracking before WhisperModel
        # loads it. snapshot_download returns the local cache path.
        local_dir = self._pre_download_model(on_sub_progress)
        model_source = local_dir if local_dir else self._model_name

        self._model = faster_whisper.WhisperModel(
            model_source,
            device=device,
            compute_type=compute_type,
            cpu_threads=os.cpu_count() or 4,
            num_workers=1,
        )
        # BatchedInferencePipeline processes multiple VAD segments in
        # parallel, giving a significant speedup on multi-core CPUs.
        self._batched_model = faster_whisper.BatchedInferencePipeline(
            model=self._model,
        )
        elapsed = time.perf_counter() - t0
        log.info("STT model loaded (device=%s, compute=%s, cpu_threads=%d) in %.2fs.",
                 device, compute_type, os.cpu_count() or 4, elapsed)

    def _pre_download_model(
        self,
        on_sub_progress: Optional[Callable[[int, int, str], None]] = None,
    ) -> Optional[str]:
        """Pre-download the HuggingFace model with progress reporting.

        Files are placed directly into ``<models_dir>/stt/`` using
        ``local_dir``, which bypasses the HuggingFace symlink-based cache
        entirely.  This avoids WinError 1314 on Windows systems without
        Developer Mode.

        Args:
            on_sub_progress: Progress callback(done, total, label).

        Returns:
            Local directory path if downloaded/already present, else None.
        """
        try:
            from huggingface_hub import snapshot_download
            from .model_cache import get_models_dir

            models_dir = get_models_dir()
            dir_name = _STT_LOCAL_DIRS.get(
                self._model_name,
                self._model_name.split("/")[-1],  # safe fallback from HF repo name
            )
            stt_dir = models_dir / dir_name
            stt_dir.mkdir(parents=True, exist_ok=True)

            # If the model files are already present (config.json is the
            # canonical marker file for CTranslate2 models), skip download.
            if (stt_dir / "config.json").exists():
                log.info("STT model already present at %s; skipping download.", stt_dir)
                return str(stt_dir)

            # Check legacy location (e.g. stt_turbo_backup from earlier installs).
            legacy_name = _STT_LEGACY_DIRS.get(self._model_name)
            if legacy_name:
                legacy_dir = models_dir / legacy_name
                if (legacy_dir / "config.json").exists():
                    log.info(
                        "Using legacy model directory %s for %s; "
                        "consider renaming it to %s.",
                        legacy_dir, self._model_name, stt_dir,
                    )
                    return str(legacy_dir)

            # Build custom tqdm class that forwards byte progress to GUI.
            # snapshot_download creates a single tqdm for all files.
            # HF Hub reuses ONE tqdm instance, mutating self.total as
            # it learns each file's Content-Length via HTTP headers.
            # The final self.total reflects the real total (~1.5 GB for
            # the CT2 model). We simply forward self.n / self.total to
            # the GUI, skipping updates until the total looks plausible
            # (> 1 MB) to avoid a brief junk percentage during the tiny
            # config/tokenizer files.
            tqdm_cls = None
            if on_sub_progress:
                on_sub_progress(0, 0, tr(S.STT_DOWNLOADING))

                import threading as _threading
                from tqdm import tqdm as _tqdm_base

                # Minimum total before we report progress, to skip the
                # brief phase where total=357 (a tiny config file).
                _MIN_TOTAL = 1_000_000  # 1 MB
                # Throttle GUI updates to avoid signal flood.
                _REPORT_INTERVAL = 0.25  # seconds between GUI updates

                class _ProgressTqdm(_tqdm_base):
                    """Forwards tqdm progress to the GUI sub-progress bar.

                    HF Hub creates ONE bytes_progress bar (this class)
                    with total=0, unit='B'.  Per-file _AggregatedTqdm
                    instances call bytes_progress.update(n) from
                    MULTIPLE download threads concurrently.

                    tqdm's self.n += n is NOT thread-safe (4 bytecodes,
                    GIL can preempt between them), so we use our own
                    lock-protected counter to track actual progress.

                    IMPORTANT: close() must NOT call on_sub_progress
                    because Python GC may call it AFTER the pipeline
                    has already hidden the bar via _sub(-1,-1,""),
                    which would re-show the bar during transcription.
                    """

                    def __init__(self, *args, **kwargs):
                        super().__init__(*args, **kwargs)
                        self._lock = _threading.Lock()
                        self._actual_downloaded = 0
                        self._actual_total = 0
                        self._last_report_time = 0.0
                        self._last_reported_downloaded = -1
                        self._last_reported_total = -1
                        self._update_count = 0
                        self._reported_count = 0
                        self._skipped_count = 0
                        log.info(
                            "tqdm.__init__ id=%d unit=%r total=%r desc=%r",
                            id(self), getattr(self, "unit", None),
                            self.total, getattr(self, "desc", None),
                        )

                    def update(self, n=1):
                        super().update(n)
                        if getattr(self, "unit", None) != "B":
                            return
                        with self._lock:
                            self._update_count += 1
                            # Accumulate our own counter — immune to
                            # the tqdm self.n += n race condition.
                            self._actual_downloaded += n
                            # self.total is also mutated from multiple
                            # threads (_AggregatedTqdm.__init__ does
                            # bytes_progress.total += file_size), so
                            # snapshot it under the lock too.
                            self._actual_total = self.total or 0
                            total = self._actual_total
                            if total < _MIN_TOTAL:
                                self._skipped_count += 1
                                if self._skipped_count <= 3:
                                    log.info(
                                        "tqdm SKIP #%d: total=%d < %d "
                                        "(downloaded=%d, chunk=%d)",
                                        self._skipped_count, total,
                                        _MIN_TOTAL, self._actual_downloaded, n,
                                    )
                                return
                            downloaded = min(self._actual_downloaded, total)
                            if (downloaded == self._last_reported_downloaded
                                    and total == self._last_reported_total):
                                return
                            is_done = downloaded >= total
                            now = time.perf_counter()
                            if not is_done and (now - self._last_report_time) < _REPORT_INTERVAL:
                                return
                            self._last_report_time = now
                            self._last_reported_downloaded = downloaded
                            self._last_reported_total = total
                            self._reported_count += 1
                            pct = downloaded * 100 // total if total else 0
                            if self._reported_count <= 5 or self._reported_count % 20 == 0 or is_done:
                                log.info(
                                    "tqdm REPORT #%d: %d/%d (%d%%) "
                                    "[updates=%d skips=%d]",
                                    self._reported_count, downloaded, total,
                                    pct, self._update_count, self._skipped_count,
                                )
                        # Emit outside lock — on_sub_progress is a Qt
                        # signal emit which is already thread-safe.
                        on_sub_progress(downloaded, total, tr(S.STT_DOWNLOADING))

                    def close(self):
                        # Log only — do NOT call on_sub_progress here.
                        # GC may call close() long after the pipeline
                        # has hidden the bar, which would re-show it.
                        with self._lock:
                            total = self._actual_total
                            downloaded = min(self._actual_downloaded, total) if total else self._actual_downloaded
                            log.info(
                                "tqdm.close: %d/%d [updates=%d reports=%d skips=%d]",
                                downloaded, total,
                                self._update_count, self._reported_count,
                                self._skipped_count,
                            )
                        super().close()

                tqdm_cls = _ProgressTqdm

            # local_dir copies files directly — no symlinks, no HF cache.
            # Retry with exponential backoff for transient HF rate-limit
            # errors (WinError 10054 / ConnectError).
            _MAX_RETRIES = 3
            for attempt in range(1, _MAX_RETRIES + 1):
                try:
                    snapshot_download(
                        self._model_name,
                        local_dir=str(stt_dir),
                        tqdm_class=tqdm_cls,
                    )
                    log.info("STT model downloaded to %s", stt_dir)
                    return str(stt_dir)
                except Exception as exc:
                    log.warning(
                        "snapshot_download attempt %d/%d failed: %s",
                        attempt, _MAX_RETRIES, exc,
                    )
                    if attempt < _MAX_RETRIES:
                        wait = 2 ** attempt  # 2, 4 seconds
                        log.info(
                            "Retrying STT download in %ds...", wait,
                        )
                        if on_sub_progress:
                            on_sub_progress(
                                0, 0,
                                f"{tr(S.STT_DOWNLOADING)} (retry {attempt}/{_MAX_RETRIES}...)",
                            )
                        time.sleep(wait)
                    else:
                        log.warning(
                            "All %d download attempts failed; "
                            "falling back to model name (no progress bar).",
                            _MAX_RETRIES,
                        )
                        if on_sub_progress:
                            on_sub_progress(
                                0, 0,
                                f"{tr(S.STT_DOWNLOADING)} (fallback)...",
                            )
                        return None
        except Exception as exc:
            log.warning("Pre-download failed (%s); falling back to model name.", exc)
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio_path: str | Path,
        *,
        vad_filter: bool = True,
        on_progress: Optional[Callable[[int, str], None]] = None,
        on_segment: Optional[Callable[["TranscribedSegment"], None]] = None,
        on_vad_done: Optional[Callable[[List["TranscribedSegment"]], None]] = None,
        on_gap_fill_progress: Optional[Callable[[int, int, str], None]] = None,
        on_gap_done: Optional[Callable[[float, float], None]] = None,
        on_boundaries_computed: Optional[Callable[[List[tuple]], None]] = None,
        skip_gaps: Optional[List[tuple]] = None,
        cached_segments: Optional[List["TranscribedSegment"]] = None,
        gap_fill_only: bool = False,
        on_vad_timestamps: Optional[Callable[[List[dict]], None]] = None,
        vad_timestamps: Optional[List[dict]] = None,
    ) -> List[TranscribedSegment]:
        """Transcribe an audio file and return segments with word-level timestamps.

        Args:
            audio_path: Path to audio file (WAV, MP3, M4A, FLAC, etc.).
            vad_filter: Use Silero-VAD to skip silent regions (default True).
            on_progress: Optional callback(percent, label) for progress updates.
            on_segment: Optional callback(TranscribedSegment) fired after each
                segment is transcribed, for incremental live display.
            on_vad_done: Optional callback(segments) fired after the VAD pass
                completes but before gap-fill starts.  Used by the pipeline
                to persist a completed snapshot so an interrupted gap-fill
                does not discard the VAD results.
            on_gap_fill_progress: Optional callback(done_s, total_s, label)
                for gap-fill sub-progress bar (seconds done / total).
            on_gap_done: Optional callback(gap_start, gap_end) fired after
                each gap is fully scanned, so the caller can persist progress.
            skip_gaps: List of (start, end) gap boundaries already completed
                in a previous run.  These gaps will be skipped.
            cached_segments: Previously transcribed segments from an
                interrupted run.  Segments whose time range overlaps a
                cached segment are skipped (the cached version is used).
            gap_fill_only: If True, skip VAD+STT entirely and jump straight
                to gap-fill using *cached_segments*.  Requires *cached_segments*
                to be provided.  Avoids redundant Silero-VAD scan when
                resuming from cache with only gap-fill remaining.
            on_vad_timestamps: Optional callback(timestamps) fired after
                Silero-VAD finishes with the raw speech timestamps
                (list of {"start": samples, "end": samples} dicts).
                Used by the pipeline to cache VAD results for resume.
            vad_timestamps: Pre-computed Silero-VAD timestamps from a
                previous run (list of {"start": samples, "end": samples}).
                When provided, skips the Silero-VAD scan and passes
                these as clip_timestamps to faster-whisper.

        Returns:
            List of TranscribedSegment, each containing words with timestamps.

        Raises:
            FileNotFoundError: If *audio_path* does not exist.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(
                f"Audio file not found: {audio_path}\n"
                f"Absolute path: {audio_path.resolve()}"
            )

        self._load_model()
        t0 = time.perf_counter()
        log.info("STT.transcribe: start file=%s vad=%s", audio_path, vad_filter)

        # --- Gap-fill-only fast path ---
        # When all VAD+STT segments are already cached and only gap-fill
        # remains, skip the expensive Silero-VAD scan entirely.
        if gap_fill_only and not cached_segments:
            log.warning("gap_fill_only=True but no cached_segments; falling back to full VAD+STT.")
        if gap_fill_only and cached_segments:
            log.info("Gap-fill-only mode: skipping VAD, using %d cached segments.",
                     len(cached_segments))
            segments = list(cached_segments)
            # Compute total_duration from WAV header instead of info.duration.
            try:
                with wave.open(str(audio_path), "rb") as wf:
                    total_duration = wf.getnframes() / wf.getframerate()
            except Exception:
                total_duration = segments[-1].end if segments else 0
            if on_progress:
                on_progress(30, tr(S.STT_STARTING))
            if on_vad_done:
                try:
                    on_vad_done(list(segments))
                except Exception:
                    log.debug("on_vad_done callback failed", exc_info=True)
            segments = self._fill_gaps(
                audio_path, segments, total_duration, on_progress, on_segment,
                on_gap_fill_progress=on_gap_fill_progress,
                on_gap_done=on_gap_done,
                on_boundaries_computed=on_boundaries_computed,
                skip_gaps=skip_gaps,
            )
            if on_progress:
                on_progress(100, f"{tr(S.PIPE_DONE)} — {len(segments)}")
            elapsed = time.perf_counter() - t0
            log.info("STT.transcribe (gap-fill-only): done in %.2fs — %d segments.",
                     elapsed, len(segments))
            return segments

        # Show scanning message before the blocking VAD scan.
        if on_progress:
            if vad_timestamps:
                on_progress(5, tr(S.STT_STARTING))
            else:
                on_progress(1, tr(S.PIPE_STT_START))

        # Use BatchedInferencePipeline for parallel decoding of VAD chunks.
        batch_size = max(1, (os.cpu_count() or 4) // 2)

        # When we have cached VAD timestamps from a previous run, skip the
        # expensive Silero-VAD scan and pass them as clip_timestamps.
        # On first run, let BatchedInferencePipeline handle VAD internally
        # (it uses collect_chunks + restore_speech_timestamps for optimal
        # batching and timestamp accuracy).
        _clip_ts: Optional[list] = None
        _using_cached_vad = False
        if vad_timestamps and vad_filter:  # non-empty list + vad enabled
            _sr = 16000  # Silero-VAD always uses 16kHz
            _clip_ts = [
                {"start": s["start"] / _sr, "end": s["end"] / _sr}
                for s in vad_timestamps
            ]
            _using_cached_vad = True
            log.info("Using %d cached VAD timestamps; skipping Silero-VAD scan.",
                     len(vad_timestamps))

        raw_segments, info = self._batched_model.transcribe(
            str(audio_path),
            language="he",
            word_timestamps=True,
            vad_filter=vad_filter if not _using_cached_vad else False,
            vad_parameters={"threshold": 0.2, "min_silence_duration_ms": 500, "max_speech_duration_s": 30},
            beam_size=5,
            batch_size=batch_size,
            no_speech_threshold=0.4,
            condition_on_previous_text=False,
            clip_timestamps=_clip_ts,
        )

        total_duration = getattr(info, "duration", 0) or 0

        # Build a lookup of cached segment time ranges so we can skip
        # re-transcribing segments that were already done in a prior run.
        _cached_by_start: dict[float, "TranscribedSegment"] = {}
        if cached_segments:
            for cs in cached_segments:
                _cached_by_start[round(cs.start, 2)] = cs
            _max_cached_end = max(cs.end for cs in cached_segments)
            log.info("Resuming with %d cached segments (up to %.1fs).",
                     len(cached_segments), _max_cached_end)
            # Set initial progress reflecting cached coverage so the bar
            # doesn't reset to 1% while the blocking VAD scan runs.
            if on_progress and total_duration > 0:
                cached_pct = min(90, int(_max_cached_end / total_duration * 90))
                cur = _format_hhmmss(_max_cached_end)
                tot = _format_hhmmss(total_duration)
                on_progress(max(5, cached_pct),
                            f"{tr(S.STT_TRANSCRIBING)} {cur} / {tot} ({tr(S.PIPE_STT_START)})")
            elif on_progress:
                on_progress(5, tr(S.STT_STARTING))
        elif on_progress:
            on_progress(5, tr(S.STT_STARTING))

        segments: List[TranscribedSegment] = []
        _cached_used = 0
        for segment in raw_segments:
            # Check if we already have this segment from a previous run.
            seg_start_key = round(segment.start, 2)
            cached_seg = _cached_by_start.pop(seg_start_key, None)
            if cached_seg is not None:
                # Use the cached version — skip Whisper's output for this range.
                segments.append(cached_seg)
                _cached_used += 1
            else:
                words = [
                    TranscribedWord(
                        word=word.word.strip(),
                        start=word.start,
                        end=word.end,
                        confidence=word.probability,
                    )
                    for word in (segment.words or [])
                    if word.word.strip()
                ]
                segments.append(
                    TranscribedSegment(
                        text=segment.text.strip(),
                        start=segment.start,
                        end=segment.end,
                        words=words,
                    )
                )
            # Fire incremental callback for live display.
            # Skip for cached segments — the pipeline already has them
            # pre-populated and would discard the duplicate anyway.
            if on_segment and cached_seg is None:
                try:
                    on_segment(segments[-1])
                except Exception:
                    log.debug("on_segment callback failed", exc_info=True)
            # Report progress based on how far into the audio we've transcribed.
            if on_progress and total_duration > 0:
                pct = min(90, int(segment.end / total_duration * 90))
                cur = _format_hhmmss(segment.end)
                tot = _format_hhmmss(total_duration)
                on_progress(pct, f"{tr(S.STT_TRANSCRIBING)} {cur} / {tot}")

        if _cached_used > 0:
            log.info("Reused %d/%d segments from cache; %d newly transcribed.",
                     _cached_used, len(segments), len(segments) - _cached_used)

        # On first run (no cached VAD), capture VAD timestamps now for caching.
        # The library ran VAD internally during transcribe(), but doesn't
        # expose the raw timestamps.  We re-run Silero-VAD here (fast, ~40s
        # for a 5h file) so future resumes can skip it entirely.
        if vad_filter and not _using_cached_vad and on_vad_timestamps:
            try:
                from faster_whisper.vad import get_speech_timestamps as _gst, VadOptions as _VO
                from faster_whisper.audio import decode_audio as _da
                t_cap = time.perf_counter()
                _audio_np = _da(str(audio_path), sampling_rate=16000)
                _captured = _gst(_audio_np, _VO(
                    threshold=0.2,
                    min_silence_duration_ms=500,
                    max_speech_duration_s=30,
                ))
                del _audio_np
                log.info("Captured %d VAD timestamps for caching in %.1fs.",
                         len(_captured), time.perf_counter() - t_cap)
                on_vad_timestamps(_captured)
            except Exception:
                log.debug("VAD timestamp capture failed", exc_info=True)

        # --- Gap-fill pass ---
        # Silero VAD can miss real speech in recordings with unusual acoustics
        # (distant mic, outdoor, background noise). For any gap > GAP_THRESHOLD
        # seconds, re-scan that window without VAD using RMS gating to block
        # Whisper hallucinations in truly silent chunks.
        if vad_filter:
            # Notify the caller that the VAD pass is complete so it can
            # persist a "completed" snapshot.  If gap-fill is interrupted
            # later, the pipeline will reload these VAD results instead of
            # re-running from scratch.
            if on_vad_done:
                try:
                    on_vad_done(list(segments))
                except Exception:
                    log.debug("on_vad_done callback failed", exc_info=True)
            segments = self._fill_gaps(
                audio_path, segments, total_duration, on_progress, on_segment,
                on_gap_fill_progress=on_gap_fill_progress,
                on_gap_done=on_gap_done,
                on_boundaries_computed=on_boundaries_computed,
                skip_gaps=skip_gaps,
            )

        if on_progress:
            on_progress(100, f"{tr(S.PIPE_DONE)} — {len(segments)}")

        elapsed = time.perf_counter() - t0
        total_words = sum(len(s.words) for s in segments)
        log.info(
            "STT.transcribe: done in %.2fs — %d segments, %d words, lang=%s (prob=%.2f).",
            elapsed,
            len(segments),
            total_words,
            info.language,
            info.language_probability,
        )
        return segments

    # ------------------------------------------------------------------
    # Gap-fill helpers
    # ------------------------------------------------------------------

    # Minimum gap duration to trigger a no-VAD re-scan (seconds).
    _GAP_FILL_MIN_SECONDS = 5 * 60  # 5 minutes

    # Chunk size for the no-VAD scan (seconds). Matches Whisper's native window.
    _GAP_FILL_CHUNK_S = 30

    # RMS threshold below which a chunk is considered silence and skipped.
    # -40 dB ≈ 0.01.  Chunks at -46 dB (0.005) are genuinely inaudible and
    # produce "תודה רבה" hallucinations when passed to Whisper.
    _GAP_FILL_RMS_MIN = 0.01

    def _fill_gaps(
        self,
        audio_path: Path,
        segments: List[TranscribedSegment],
        total_duration: float,
        on_progress: Optional[Callable[[int, str], None]],
        on_segment: Optional[Callable[["TranscribedSegment"], None]],
        *,
        on_gap_fill_progress: Optional[Callable[[int, int, str], None]] = None,
        on_gap_done: Optional[Callable[[float, float], None]] = None,
        on_boundaries_computed: Optional[Callable[[List[tuple]], None]] = None,
        skip_gaps: Optional[List[tuple]] = None,
    ) -> List[TranscribedSegment]:
        """Re-scan large transcript gaps without VAD to catch missed speech.

        Silero VAD can miss real speech when the recording has unusual
        acoustics (distant microphone, outdoor, heavy background noise).
        This method identifies gaps > GAP_FILL_MIN_SECONDS in the
        existing segments list, then re-transcribes those windows chunk
        by chunk (30 s) WITHOUT VAD, skipping chunks whose audio RMS
        falls below RMS_MIN to avoid Whisper hallucinations in silence.
        """
        # Build list of gaps: (gap_start, gap_end) in seconds.
        # Prepend/append sentinels for the file start and end.
        boundaries: List[Tuple[float, float]] = []
        if not segments:
            if total_duration > 0:
                boundaries = [(0.0, total_duration)]
        else:
            if segments[0].start > self._GAP_FILL_MIN_SECONDS:
                boundaries.append((0.0, segments[0].start))
            for i in range(1, len(segments)):
                gap = segments[i].start - segments[i - 1].end
                if gap >= self._GAP_FILL_MIN_SECONDS:
                    boundaries.append((segments[i - 1].end, segments[i].start))
            # Also check the trailing gap from the last segment to end of file.
            if total_duration > 0 and (total_duration - segments[-1].end) >= self._GAP_FILL_MIN_SECONDS:
                boundaries.append((segments[-1].end, total_duration))

        if not boundaries:
            return segments

        # Report all computed boundaries (including trailing gap) so the
        # pipeline can track them for crash-safe completion marking.
        if on_boundaries_computed:
            try:
                on_boundaries_computed(list(boundaries))
            except Exception:
                log.debug("on_boundaries_computed callback failed", exc_info=True)

        # Filter out gaps already completed in a previous run.
        # Use overlap matching: a recomputed gap is skipped if it falls
        # entirely within any previously-completed gap range.  Boundary
        # recomputation can shift gap start/end when cached segments fill
        # part of an old gap, so exact-coordinate matching would miss.
        _skip_ranges = [(round(s, 2), round(e, 2)) for s, e in (skip_gaps or [])]
        if _skip_ranges:
            original_count = len(boundaries)
            boundaries = [
                (s, e) for s, e in boundaries
                if not any(ds <= round(s, 2) and round(e, 2) <= de
                           for ds, de in _skip_ranges)
            ]
            skipped = original_count - len(boundaries)
            if skipped:
                log.info("Gap-fill: skipping %d already-completed gap(s).", skipped)
            if not boundaries:
                log.info("Gap-fill: all gaps already completed.")
                return segments

        log.info(
            "Gap-fill: %d gap(s) > %.0f min — rescanning without VAD.",
            len(boundaries),
            self._GAP_FILL_MIN_SECONDS / 60,
        )

        # Read audio file once.
        try:
            with wave.open(str(audio_path), "rb") as wf:
                sr = wf.getframerate()
                n_channels = wf.getnchannels()
                wf.setpos(0)
                raw = wf.readframes(wf.getnframes())
            dtype = np.int16 if wf.getsampwidth() == 2 else np.int32
            audio_all = np.frombuffer(raw, dtype=dtype).astype(np.float32)
            if n_channels > 1:
                audio_all = audio_all.reshape(-1, n_channels).mean(axis=1)
            audio_all /= np.iinfo(dtype).max
        except Exception as exc:
            log.warning("Gap-fill: failed to read audio (%s); skipping.", exc)
            return segments

        new_segments: List[TranscribedSegment] = []
        chunk_s = self._GAP_FILL_CHUNK_S

        # Compute total gap duration for progress reporting.
        total_gap_seconds = sum(ge - gs for gs, ge in boundaries)
        gap_base_seconds = 0.0  # seconds processed in prior gaps

        for gap_start, gap_end in boundaries:
            log.info(
                "Gap-fill: scanning %s – %s",
                _format_hhmmss(gap_start),
                _format_hhmmss(gap_end),
            )
            pos = gap_start
            while pos < gap_end:
                chunk_end = min(pos + chunk_s, gap_end)
                s_idx = int(pos * sr)
                e_idx = int(chunk_end * sr)
                chunk = audio_all[s_idx:e_idx]

                # Report gap-fill progress (occupies 90-99% range,
                # continuing from the VAD pass which uses 0-90%).
                if total_gap_seconds > 0:
                    done = gap_base_seconds + (pos - gap_start)
                    done_int = int(done)
                    total_int = int(total_gap_seconds)
                    cur = _format_hhmmss(done)
                    tot = _format_hhmmss(total_gap_seconds)
                    label = f"{tr(S.STT_GAP_FILL)} {cur} / {tot}"
                    if on_progress:
                        gf_pct = 90 + min(9, int(done / total_gap_seconds * 9))
                        on_progress(gf_pct, label)
                    if on_gap_fill_progress:
                        try:
                            on_gap_fill_progress(done_int, total_int, label)
                        except Exception:
                            pass

                rms = float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) else 0.0
                if rms < self._GAP_FILL_RMS_MIN:
                    pos = chunk_end
                    continue  # genuinely silent — skip to avoid hallucinations

                try:
                    raw_segs, _ = self._model.transcribe(
                        chunk,
                        language="he",
                        word_timestamps=True,
                        vad_filter=False,
                        beam_size=5,
                        no_speech_threshold=0.6,
                        condition_on_previous_text=False,
                    )
                    for seg in raw_segs:
                        if seg.no_speech_prob >= 0.6:
                            continue
                        words = [
                            TranscribedWord(
                                word=w.word.strip(),
                                start=pos + w.start,
                                end=pos + w.end,
                                confidence=w.probability,
                            )
                            for w in (seg.words or [])
                            if w.word.strip()
                        ]
                        ts = TranscribedSegment(
                            text=seg.text.strip(),
                            start=pos + seg.start,
                            end=pos + seg.end,
                            words=words,
                        )
                        new_segments.append(ts)
                        if on_segment:
                            try:
                                on_segment(ts)
                            except Exception:
                                pass
                        log.debug(
                            "Gap-fill segment: [%s] %s",
                            _format_hhmmss(ts.start),
                            ts.text[:60],
                        )
                except Exception as exc:
                    log.warning("Gap-fill chunk error at %.1fs: %s", pos, exc)

                pos = chunk_end
            gap_base_seconds += gap_end - gap_start
            # Notify caller this gap is done so it can persist progress.
            if on_gap_done:
                try:
                    on_gap_done(gap_start, gap_end)
                except Exception:
                    log.debug("on_gap_done callback failed", exc_info=True)

        if not new_segments:
            return segments

        log.info("Gap-fill: added %d segment(s) from gap regions.", len(new_segments))
        combined = sorted(segments + new_segments, key=lambda s: s.start)
        return combined


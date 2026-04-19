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
from pathlib import Path
from typing import Callable, List, Optional

from .models import TranscribedSegment, TranscribedWord
from .gui.strings import tr, S

log = logging.getLogger(__name__)

# ===========================
# CONSTANTS
# ===========================

# Default model -- Hebrew fine-tuned Whisper from ivrit-ai (Apache-2.0).
# Trained on ~5 000+ hours of Hebrew data (Knesset transcripts, crowd-sourced).
DEFAULT_MODEL = "ivrit-ai/whisper-large-v3-turbo-ct2"


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

            stt_dir = get_models_dir() / "stt"
            stt_dir.mkdir(parents=True, exist_ok=True)

            # If the model files are already present (config.json is the
            # canonical marker file for CTranslate2 models), skip download.
            if (stt_dir / "config.json").exists():
                log.info("STT model already present at %s; skipping download.", stt_dir)
                return str(stt_dir)

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
    ) -> List[TranscribedSegment]:
        """Transcribe an audio file and return segments with word-level timestamps.

        Args:
            audio_path: Path to audio file (WAV, MP3, M4A, FLAC, etc.).
            vad_filter: Use Silero-VAD to skip silent regions (default True).
            on_progress: Optional callback(percent, label) for progress updates.
            on_segment: Optional callback(TranscribedSegment) fired after each
                segment is transcribed, for incremental live display.

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

        # Show scanning message before the blocking VAD scan.
        if on_progress:
            on_progress(1, tr(S.PIPE_STT_START))

        # Use BatchedInferencePipeline for parallel decoding of VAD chunks.
        batch_size = max(1, (os.cpu_count() or 4) // 2)
        raw_segments, info = self._batched_model.transcribe(
            str(audio_path),
            language="he",
            word_timestamps=True,
            vad_filter=vad_filter,
            beam_size=5,
            batch_size=batch_size,
        )

        total_duration = getattr(info, "duration", 0) or 0
        if on_progress:
            on_progress(5, tr(S.STT_STARTING))

        segments: List[TranscribedSegment] = []
        for segment in raw_segments:
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
            if on_segment:
                try:
                    on_segment(segments[-1])
                except Exception:
                    log.debug("on_segment callback failed", exc_info=True)
            # Report progress based on how far into the audio we've transcribed.
            if on_progress and total_duration > 0:
                pct = min(95, int(segment.end / total_duration * 95))
                cur = _format_hhmmss(segment.end)
                tot = _format_hhmmss(total_duration)
                on_progress(pct, f"{tr(S.STT_TRANSCRIBING)} {cur} / {tot}")

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

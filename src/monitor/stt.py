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
        if self._model is not None:
            log.debug("STT model already loaded; skipping.")
            return

        t0 = time.perf_counter()
        # Import here to avoid heavy module-load time when model is not needed.
        import faster_whisper

        log.info(
            "Loading STT model %s (device=%s) -- first run downloads ~3 GB...",
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
            # snapshot_download creates one tqdm instance per file (7 files),
            # so we need a shared accumulator to give a smooth aggregate bar.
            tqdm_cls = None
            if on_sub_progress:
                on_sub_progress(0, 0, tr(S.STT_DOWNLOADING))

                from tqdm import tqdm as _tqdm_base
                import threading as _threading

                # Shared mutable state across all tqdm instances spawned by
                # snapshot_download for this batch.
                # file_totals/file_downloaded are keyed by id(tqdm_instance).
                # Only byte-unit tqdms are tracked; the outer "Fetching N
                # files" tqdm (unit='it') is ignored entirely.
                # huggingface_hub uses a worker pool, so concurrent updates
                # from multiple threads must be serialized with a lock.
                _shared: dict = {"file_totals": {}, "file_downloaded": {}}
                _shared_lock = _threading.Lock()

                class _ProgressTqdm(_tqdm_base):
                    """Aggregates per-file byte progress into a single bar.

                    Tracks each byte-unit tqdm independently so that tqdms
                    whose total size is unknown at creation time (HF Hub
                    reports Content-Length later) don't corrupt the aggregated
                    percentage with a near-zero denominator.
                    """

                    def __init__(self, *args, **kwargs):
                        super().__init__(*args, **kwargs)
                        log.debug(
                            "tqdm.__init__ id=%d unit=%r total=%r desc=%r",
                            id(self), getattr(self, "unit", None),
                            self.total, getattr(self, "desc", None),
                        )
                        # Register early only when total is already known.
                        if getattr(self, "unit", None) == "B" and self.total and self.total > 0:
                            with _shared_lock:
                                _shared["file_totals"][id(self)] = self.total
                                _shared["file_downloaded"][id(self)] = 0

                    def update(self, n=1):
                        super().update(n)  # increments self.n
                        if not n or n <= 0:
                            return
                        # Ignore non-byte tqdms (e.g. "Fetching N files" unit='it').
                        if getattr(self, "unit", None) != "B":
                            return
                        myid = id(self)
                        with _shared_lock:
                            # huggingface_hub reuses a single tqdm instance
                            # across multiple files, mutating self.total when
                            # it learns each file's Content-Length. We must
                            # therefore re-sync our cached total whenever it
                            # changes, and reset our per-id downloaded counter
                            # to self.n (the cumulative byte count tqdm tracks
                            # for the *current* file).
                            cur_total = self.total if (self.total and self.total > 0) else 0
                            cached_total = _shared["file_totals"].get(myid)
                            if cur_total and cur_total != cached_total:
                                # New file (or first known total). Promote the
                                # already-finished previous file's count into
                                # a stable archive bucket so it isn't lost
                                # when we overwrite myid's slot.
                                if cached_total:
                                    archive_key = (myid, cached_total)
                                    _shared["file_totals"][archive_key] = cached_total
                                    _shared["file_downloaded"][archive_key] = (
                                        _shared["file_downloaded"].get(myid, cached_total)
                                    )
                                _shared["file_totals"][myid] = cur_total
                                _shared["file_downloaded"][myid] = self.n
                            elif cur_total:
                                # Same file, accumulate.
                                _shared["file_downloaded"][myid] = self.n
                            else:
                                # Total still unknown -- skip.
                                return
                            total = sum(_shared["file_totals"].values())
                            downloaded = sum(_shared["file_downloaded"].values())
                        # Sanity check.
                        if downloaded > total:
                            log.warning(
                                "STT progress overshoot: downloaded=%d > total=%d "
                                "(file id=%d unit=%r self.total=%r self.n=%d n=%d)",
                                downloaded, total, id(self),
                                getattr(self, "unit", None), self.total, self.n, n,
                            )
                        on_sub_progress(downloaded, total, tr(S.STT_DOWNLOADING))

                tqdm_cls = _ProgressTqdm

            # local_dir copies files directly — no symlinks, no HF cache.
            snapshot_download(
                self._model_name,
                local_dir=str(stt_dir),
                tqdm_class=tqdm_cls,
            )
            log.info("STT model downloaded to %s", stt_dir)
            return str(stt_dir)
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
    ) -> List[TranscribedSegment]:
        """Transcribe an audio file and return segments with word-level timestamps.

        Args:
            audio_path: Path to audio file (WAV, MP3, M4A, FLAC, etc.).
            vad_filter: Use Silero-VAD to skip silent regions (default True).
            on_progress: Optional callback(percent, label) for progress updates.

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
            on_progress(5, tr(S.STT_TRANSCRIBING))

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

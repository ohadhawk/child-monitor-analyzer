"""
Audio event detection -- shouting, crying, screaming via PANNs + volume spikes via librosa.

Uses the PANNs (Pre-trained Audio Neural Networks) SoundEventDetection model
trained on Google AudioSet (527 classes) and librosa RMS energy analysis to
identify non-speech events and abnormal volume levels in audio files.

The PANNs inference code is vendored in ``monitor.vendor.panns`` rather than
installed from PyPI -- see that package's ``LICENSE-third-party.txt`` for the
rationale and the list of security fixes applied.

Usage:
    from monitor.audio_events import AudioEventDetector

    detector = AudioEventDetector()
    detections = detector.detect("recording.wav")
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from .cancellation import raise_if_cancelled
from .model_cache import ensure_panns_labels, ensure_panns_ready, get_panns_dir
from .models import AUDIOSET_CLASS_MAP, Detection, DetectionType
from .gui.strings import tr, S

log = logging.getLogger(__name__)

# ===========================
# DEVICE DETECTION
# ===========================


def _get_best_device() -> str:
    """Return the best available torch device string.

    Prefers XPU (Intel Arc), then CUDA, then CPU.
    """
    try:
        import torch
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            name = torch.xpu.get_device_name(0)
            log.info("Using XPU device: %s", name)
            return "xpu"
        if torch.cuda.is_available():
            log.info("Using CUDA device.")
            return "cuda"
    except Exception:
        pass
    log.info("Using CPU device.")
    return "cpu"


def _describe_audio_file(audio_path: str) -> str:
    """Return a one-line description of *audio_path* for failure diagnostics."""
    import os
    parts = []
    try:
        st = os.stat(audio_path)
        parts.append(f"size={st.st_size}")
        attrs = getattr(st, "st_file_attributes", None)
        if attrs is not None:
            # Cloud placeholders (Google Drive / OneDrive) stream data on read.
            flags = {0x1000: "OFFLINE", 0x40000: "RECALL_ON_OPEN",
                     0x400000: "RECALL_ON_DATA_ACCESS", 0x400: "REPARSE_POINT"}
            parts.append(f"attrs=0x{attrs:x}[{','.join(n for b, n in flags.items() if attrs & b) or '-'}]")
    except OSError as exc:
        parts.append(f"stat_failed={exc!r}")
    try:
        with open(audio_path, "rb") as fh:
            parts.append(f"head={fh.read(16).hex()}")
    except OSError as exc:
        parts.append(f"open_failed={exc!r}")
    try:
        import soundfile
        parts.append(f"libsndfile={soundfile.__libsndfile_version__}")
        info = soundfile.info(audio_path)
        parts.append(
            f"sf_format={info.format}/{info.subtype} sr={info.samplerate} "
            f"ch={info.channels} frames={info.frames} dur={info.duration:.1f}s"
        )
    except Exception as exc:
        parts.append(f"sf_info_failed={type(exc).__name__}: {exc}")
    try:
        import audioread
        backends = [b.__name__ for b in audioread.available_backends()]
        parts.append(f"audioread_backends={backends or 'NONE'}")
    except Exception as exc:
        parts.append(f"audioread_check_failed={type(exc).__name__}: {exc}")
    return " ".join(parts)


def _load_mono(audio_path: str, sr: int) -> tuple[np.ndarray, int]:
    """Load *audio_path* as mono float32 at *sr* Hz.

    librosa uses libsndfile, whose MP3 decoder rejects some files, and its
    audioread fallback has no backend in the frozen build.  PyAV (bundled for
    faster-whisper) decodes those files, so use it as the fallback.
    """
    import librosa
    t0 = time.perf_counter()
    try:
        audio, out_sr = librosa.load(audio_path, sr=sr, mono=True)
        log.debug("librosa decoded %s: %d samples @ %d Hz in %.1fs",
                  audio_path, len(audio), out_sr, time.perf_counter() - t0)
        return audio, out_sr
    except Exception as exc:
        chain = []
        e: BaseException | None = exc
        while e is not None:
            chain.append(f"{type(e).__module__}.{type(e).__name__}: {e}")
            e = e.__cause__ or e.__context__
        log.warning(
            "librosa could not decode %s after %.1fs; falling back to PyAV.\n"
            "  exception chain: %s\n  file: %s",
            audio_path, time.perf_counter() - t0, " <- ".join(chain),
            _describe_audio_file(audio_path),
            exc_info=True,
        )
    from faster_whisper.audio import decode_audio
    t1 = time.perf_counter()
    try:
        audio = decode_audio(audio_path, sampling_rate=sr)
    except Exception:
        log.exception("PyAV also failed to decode %s.", audio_path)
        raise
    log.info("PyAV decoded %s: %d samples @ %d Hz (%.1fs) in %.1fs",
             audio_path, len(audio), sr, len(audio) / sr, time.perf_counter() - t1)
    return audio, sr

# ===========================
# CONSTANTS
# ===========================

# Confidence threshold for PANNs frame-level detections.
DEFAULT_CONFIDENCE_THRESHOLD = 0.15

# Minimum contiguous duration (seconds) for a detection to be reported.
# Filters out very brief acoustic events that are likely false positives.
MIN_EVENT_DURATION = 0.3

# RMS energy percentile above which a volume spike is reported.
VOLUME_SPIKE_PERCENTILE = 95

# Minimum RMS value to consider (avoid silence artifacts).
VOLUME_SPIKE_MIN_RMS = 0.05

# PANNs native sample rate; volume-spike detection reuses the same samples.
PANNS_SAMPLE_RATE = 32000


def _format_hhmmss(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    s = int(seconds)
    h, remainder = divmod(s, 3600)
    m, sec = divmod(remainder, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


# ===========================
# CORE CLASS
# ===========================


class AudioEventDetector:
    """Detect non-speech audio events (shout, cry, scream) and volume spikes.

    Combines two detection strategies:
    1. PANNs SoundEventDetection for acoustic event classification.
    2. librosa RMS energy analysis for volume anomalies.

    Attributes:
        _confidence_threshold: Minimum probability for PANNs detections.
    """

    def __init__(self, confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD) -> None:
        self._confidence_threshold = confidence_threshold
        self._sed_model = None
        self._load_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Private helpers -- model loading
    # ------------------------------------------------------------------

    def _load_sed(
        self,
        on_sub_progress: Optional[Callable[[int, int, str], None]] = None,
    ) -> None:
        """Load the PANNs SoundEventDetection model if not already loaded.

        Args:
            on_sub_progress: Optional callback(done, total, label) for download
                progress. Threaded to model_cache download helpers.
        """
        with self._load_lock:
            self._load_sed_locked(on_sub_progress)

    def _load_sed_locked(
        self,
        on_sub_progress: Optional[Callable[[int, int, str], None]] = None,
    ) -> None:
        """Inner SED model load, called while holding _load_lock."""
        if self._sed_model is not None:
            log.debug("PANNs SED model already loaded; skipping.")
            return

        t0 = time.perf_counter()
        # Ensure PANNs files exist in the portable models directory. The
        # checkpoint's SHA-256 is verified there before we ever load it.
        checkpoint_path = ensure_panns_ready(on_progress=on_sub_progress)
        log.info(
            "Loading PANNs SoundEventDetection model from %s ...",
            checkpoint_path,
        )

        if on_sub_progress:
            # Keep the bar at 100% (deterministic, filled) while the model
            # loads into memory — avoids reverting to indeterminate/pulsing.
            checkpoint_bytes = checkpoint_path.stat().st_size if checkpoint_path.exists() else 1
            on_sub_progress(checkpoint_bytes, checkpoint_bytes, tr(S.AE_LOADING_MODEL))

        from .vendor.panns import SoundEventDetection

        # The vendored wrapper handles XPU/CUDA/CPU directly and falls back to
        # CPU when the requested accelerator is unavailable.
        best_device = _get_best_device()

        self._sed_model = SoundEventDetection(
            checkpoint_path=checkpoint_path, device=best_device,
        )
        self._device = self._sed_model.device

        elapsed = time.perf_counter() - t0
        log.info(
            "PANNs SoundEventDetection model loaded on %s in %.2fs.",
            self._device, elapsed,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        audio_path: str | Path,
        on_progress: Optional[Callable[[int, str], None]] = None,
        on_chunk_detections: Optional[Callable[[List["Detection"]], None]] = None,
    ) -> List[Detection]:
        """Run all audio event detectors on the given file.

        Args:
            audio_path: Path to an audio file (WAV, MP3, FLAC, etc.).
            on_progress: Optional callback(percent, label) for progress updates.
            on_chunk_detections: Optional callback(List[Detection]) fired after
                each PANNs chunk with preliminary detections for live display.

        Returns:
            List of Detection objects sorted by start time.

        Raises:
            FileNotFoundError: If *audio_path* does not exist.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(
                f"Audio file not found: {audio_path}\n"
                f"Absolute path: {audio_path.resolve()}"
            )

        audio_path_str = str(audio_path)
        log.info("AudioEventDetector.detect: start file=%s", audio_path)
        t0 = time.perf_counter()
        detections: List[Detection] = []

        if on_progress:
            on_progress(10, tr(S.AE_LOADING_AUDIO))

        raise_if_cancelled()
        t_load = time.perf_counter()
        audio, sample_rate = _load_mono(audio_path_str, PANNS_SAMPLE_RATE)
        log.info(
            "Audio events: decoded %.1fs (%d samples @ %d Hz) in %.1fs",
            len(audio) / sample_rate, len(audio), sample_rate,
            time.perf_counter() - t_load,
        )
        raise_if_cancelled()

        detections.extend(self._detect_events_panns(
            audio, sample_rate, on_progress, on_chunk_detections,
        ))

        if on_progress:
            on_progress(80, tr(S.AE_CHECKING_VOLUME))

        raise_if_cancelled()
        detections.extend(self._detect_volume_spikes(audio, sample_rate))
        del audio

        # sorted() creates a NEW list -- original is unchanged.
        detections.sort(key=lambda detection: detection.start)
        elapsed = time.perf_counter() - t0
        log.info(
            "AudioEventDetector.detect: done in %.2fs — %d detections.",
            elapsed, len(detections),
        )
        if on_progress:
            on_progress(100, f"{tr(S.TASK_AUDIO_EVENTS)}: {len(detections)}")
        return detections

    # ------------------------------------------------------------------
    # PANNs-based event detection
    # ------------------------------------------------------------------

    def _detect_events_panns(
        self,
        audio: np.ndarray,
        sample_rate: int,
        on_progress: Optional[Callable[[int, str], None]] = None,
        on_chunk_detections: Optional[Callable[[List["Detection"]], None]] = None,
    ) -> List[Detection]:
        """Classify audio events using PANNs SoundEventDetection.

        For long audio files, processes in chunks of CHUNK_SECONDS to
        avoid memory issues and provide real progress reporting.

        Args:
            audio: Mono float32 samples at PANNS_SAMPLE_RATE.
            sample_rate: Sample rate of *audio*.
            on_progress: Optional callback(percent, label) for progress updates.
            on_chunk_detections: Optional callback fired after each chunk with
                preliminary detections (timestamps offset to absolute position).

        Returns:
            List of Detection objects for recognised AudioSet classes.
        """
        self._load_sed()
        from .vendor.panns import load_labels_cached

        # AudioSet display names, ordered by class index. Loaded from the
        # portable models directory -- no network access, no writes to $HOME.
        panns_labels = load_labels_cached(str(ensure_panns_labels()))

        CHUNK_SECONDS = 300  # 5 minutes per chunk
        PANNS_SR = sample_rate

        t0 = time.perf_counter()
        total_duration = len(audio) / sample_rate

        chunk_samples = CHUNK_SECONDS * PANNS_SR
        total_samples = len(audio)
        num_chunks = max(1, (total_samples + chunk_samples - 1) // chunk_samples)

        all_probabilities: List[np.ndarray] = []
        total_frames = 0

        # Minimum chunk length required by PANNs CNN layers (pooling
        # reduces the time dimension through several stages; anything
        # shorter than ~1 s at 32 kHz causes "output size too small").
        MIN_CHUNK_SAMPLES = PANNS_SR  # 1 second

        for chunk_idx in range(num_chunks):
            raise_if_cancelled()
            start_sample = chunk_idx * chunk_samples
            end_sample = min(start_sample + chunk_samples, total_samples)
            chunk = audio[start_sample:end_sample]

            # Pad a short tail chunk with zeros so the CNN layers don't
            # choke on a tiny input.
            if len(chunk) < MIN_CHUNK_SAMPLES:
                chunk = np.pad(chunk, (0, MIN_CHUNK_SAMPLES - len(chunk)))

            chunk_input = chunk[np.newaxis, :]  # (1, samples)
            framewise_output = self._sed_model.inference(chunk_input)
            # framewise_output shape: (1, num_frames, 527)
            all_probabilities.append(framewise_output[0])
            total_frames += framewise_output[0].shape[0]

            # Progress: 20% for load, 20-75% for inference chunks.
            if on_progress:
                pct = 20 + int((chunk_idx + 1) / num_chunks * 55)
                elapsed = time.perf_counter() - t0
                chunk_end_sec = end_sample / sample_rate
                on_progress(
                    pct,
                    f"PANNs {_format_hhmmss(chunk_end_sec)}/{_format_hhmmss(total_duration)} ({elapsed:.0f}s)",
                )
            log.debug(
                "PANNs: chunk %d/%d done (%.0fs-%.0fs)",
                chunk_idx + 1, num_chunks,
                start_sample / sample_rate, end_sample / sample_rate,
            )

            # Fire incremental callback with per-chunk detections for live display.
            # Skip for padded short chunks (<1s real audio) where timestamp
            # mapping would be inaccurate due to zero-padding.
            real_chunk_samples = end_sample - start_sample
            if on_chunk_detections and real_chunk_samples >= MIN_CHUNK_SAMPLES:
                try:
                    chunk_probs = framewise_output[0]  # (num_frames, 527)
                    chunk_frames = chunk_probs.shape[0]
                    chunk_duration = (end_sample - start_sample) / sample_rate
                    chunk_frame_dur = chunk_duration / chunk_frames if chunk_frames > 0 else 0
                    chunk_offset = start_sample / sample_rate
                    chunk_dets: List[Detection] = []
                    for class_name, detection_type in AUDIOSET_CLASS_MAP.items():
                        try:
                            class_index = panns_labels.index(class_name)
                        except ValueError:
                            continue
                        dets = self._frames_to_detections(
                            chunk_probs[:, class_index],
                            chunk_frame_dur, detection_type, class_name,
                        )
                        # Offset timestamps from chunk-local to absolute.
                        for d in dets:
                            chunk_dets.append(Detection(
                                type=d.type,
                                start=round(d.start + chunk_offset, 2),
                                end=round(d.end + chunk_offset, 2),
                                confidence=d.confidence,
                                details=d.details,
                            ))
                    if chunk_dets:
                        on_chunk_detections(chunk_dets)
                except Exception:
                    log.debug("on_chunk_detections callback failed", exc_info=True)

        # Concatenate all chunk probabilities.
        probabilities = np.concatenate(all_probabilities, axis=0)  # (total_frames, 527)
        t_inf = time.perf_counter() - t0
        log.info("PANNs: all %d chunks done in %.2fs, %d total frames.", num_chunks, t_inf, total_frames)

        if on_progress:
            on_progress(78, tr(S.AE_ANALYSING_PANNS))

        num_frames = probabilities.shape[0]
        frame_duration = total_duration / num_frames

        detections: List[Detection] = []

        # Check each AudioSet class we care about (defined in AUDIOSET_CLASS_MAP).
        for class_name, detection_type in AUDIOSET_CLASS_MAP.items():
            try:
                class_index = panns_labels.index(class_name)
            except ValueError:
                log.warning("AudioSet class %r not found in PANNs labels.", class_name)
                continue

            class_probabilities = probabilities[:, class_index]
            max_prob = float(class_probabilities.max())
            if max_prob >= self._confidence_threshold:
                log.debug(
                    "PANNs: class %r peak=%.3f (threshold=%.3f)",
                    class_name, max_prob, self._confidence_threshold,
                )
            detections.extend(
                self._frames_to_detections(
                    class_probabilities, frame_duration, detection_type, class_name
                )
            )

        return detections

    def _frames_to_detections(
        self,
        probabilities: np.ndarray,
        frame_duration: float,
        detection_type: DetectionType,
        class_name: str,
    ) -> List[Detection]:
        """Convert frame-level probabilities into contiguous Detection spans.

        Args:
            probabilities: 1-D array of per-frame confidence values.
            frame_duration: Duration of each frame in seconds.
            detection_type: The DetectionType to assign.
            class_name: Original AudioSet class name for the details dict.

        Returns:
            List of Detection objects for contiguous spans above threshold.
        """
        above_threshold = probabilities >= self._confidence_threshold
        detections: List[Detection] = []
        start_frame: Optional[int] = None

        for frame_index, is_active in enumerate(above_threshold):
            if is_active and start_frame is None:
                start_frame = frame_index
            elif not is_active and start_frame is not None:
                detection = self._make_span(
                    probabilities, start_frame, frame_index,
                    frame_duration, detection_type, class_name,
                )
                if detection is not None:
                    detections.append(detection)
                start_frame = None

        # Close trailing span if audio ends during an active detection.
        if start_frame is not None:
            detection = self._make_span(
                probabilities, start_frame, len(probabilities),
                frame_duration, detection_type, class_name,
            )
            if detection is not None:
                detections.append(detection)

        return detections

    def _make_span(
        self,
        probabilities: np.ndarray,
        start_frame: int,
        end_frame: int,
        frame_duration: float,
        detection_type: DetectionType,
        class_name: str,
    ) -> Optional[Detection]:
        """Create a Detection from a contiguous frame range, or None if too short.

        Args:
            probabilities: Full 1-D probability array.
            start_frame: First frame of the span (inclusive).
            end_frame: Last frame of the span (exclusive).
            frame_duration: Duration of each frame in seconds.
            detection_type: Category to assign.
            class_name: AudioSet class name for details.

        Returns:
            A Detection object, or None if the span is shorter than MIN_EVENT_DURATION.
        """
        start_time = start_frame * frame_duration
        end_time = end_frame * frame_duration
        if (end_time - start_time) < MIN_EVENT_DURATION:
            return None
        average_confidence = float(np.mean(probabilities[start_frame:end_frame]))
        return Detection(
            type=detection_type,
            start=start_time,
            end=end_time,
            confidence=average_confidence,
            details={"audioset_class": class_name},
        )

    # ------------------------------------------------------------------
    # Volume spike detection via RMS energy
    # ------------------------------------------------------------------

    def _detect_volume_spikes(self, audio: np.ndarray, sample_rate: int) -> List[Detection]:
        """Detect abnormally loud sections using RMS energy analysis.

        Args:
            audio: Mono float32 samples.
            sample_rate: Sample rate of *audio*.

        Returns:
            List of VOLUME_SPIKE Detection objects.
        """
        import librosa

        # Scale the 512/2048 window tuned at 22.05 kHz so frame timing is unchanged.
        hop_length = max(1, round(512 * sample_rate / 22050))
        # librosa.feature.rms returns shape (1, num_frames); take [0] for 1-D array.
        rms_values = librosa.feature.rms(
            y=audio, frame_length=4 * hop_length, hop_length=hop_length,
        )[0]

        if rms_values.max() < VOLUME_SPIKE_MIN_RMS:
            log.debug("Volume: max RMS %.4f below minimum %.4f; no spikes.", rms_values.max(), VOLUME_SPIKE_MIN_RMS)
            return []

        # np.percentile -- find the 95th-percentile RMS value as the spike threshold.
        threshold = float(np.percentile(rms_values, VOLUME_SPIKE_PERCENTILE))
        if threshold < VOLUME_SPIKE_MIN_RMS:
            log.debug("Volume: threshold %.4f below minimum; no spikes.", threshold)
            return []

        log.debug("Volume: threshold=%.4f, max=%.4f", threshold, float(rms_values.max()))

        frame_duration = hop_length / sample_rate
        above_threshold = rms_values >= threshold

        detections: List[Detection] = []
        start_frame: Optional[int] = None

        for frame_index, is_active in enumerate(above_threshold):
            if is_active and start_frame is None:
                start_frame = frame_index
            elif not is_active and start_frame is not None:
                detection = self._make_volume_detection(
                    rms_values, start_frame, frame_index, frame_duration, threshold
                )
                if detection is not None:
                    detections.append(detection)
                start_frame = None

        # Close trailing span.
        if start_frame is not None:
            detection = self._make_volume_detection(
                rms_values, start_frame, len(rms_values), frame_duration, threshold
            )
            if detection is not None:
                detections.append(detection)

        return detections

    def _make_volume_detection(
        self,
        rms_values: np.ndarray,
        start_frame: int,
        end_frame: int,
        frame_duration: float,
        threshold: float,
    ) -> Optional[Detection]:
        """Create a VOLUME_SPIKE Detection from a contiguous loud span.

        Args:
            rms_values: Full RMS array.
            start_frame: First frame (inclusive).
            end_frame: Last frame (exclusive).
            frame_duration: Duration of each frame in seconds.
            threshold: RMS threshold that triggered this span.

        Returns:
            Detection or None if the span is shorter than MIN_EVENT_DURATION.
        """
        start_time = start_frame * frame_duration
        end_time = end_frame * frame_duration
        if (end_time - start_time) < MIN_EVENT_DURATION:
            return None
        average_rms = float(np.mean(rms_values[start_frame:end_frame]))
        return Detection(
            type=DetectionType.VOLUME_SPIKE,
            start=start_time,
            end=end_time,
            # min() caps confidence at 1.0.
            confidence=min(average_rms / threshold, 1.0),
            details={"rms": round(average_rms, 4)},
        )

"""
Audio event detection -- shouting, crying, screaming via PANNs + volume spikes via librosa.

Uses the PANNs (Pre-trained Audio Neural Networks) SoundEventDetection model
trained on Google AudioSet (527 classes) and librosa RMS energy analysis to
identify non-speech events and abnormal volume levels in audio files.

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

from .model_cache import ensure_panns_ready, get_panns_dir
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
        # Ensure PANNs files exist in the portable models directory and
        # patch panns_inference.config before importing the class.
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

        from panns_inference import SoundEventDetection

        # panns_inference only supports "cuda" or "cpu" natively.
        # For XPU we load on CPU then move the model to XPU manually.
        best_device = _get_best_device()
        self._device = best_device

        self._sed_model = SoundEventDetection(
            checkpoint_path=str(checkpoint_path), device="cpu",
        )

        if best_device == "xpu":
            import torch
            self._sed_model.model = self._sed_model.model.to(torch.device("xpu"))
            self._sed_model.device = "xpu"
            log.info("PANNs model moved to XPU.")

        elapsed = time.perf_counter() - t0
        log.info("PANNs SoundEventDetection model loaded on %s in %.2fs.", best_device, elapsed)

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

        detections.extend(self._detect_events_panns(
            audio_path_str, on_progress, on_chunk_detections,
        ))

        if on_progress:
            on_progress(80, tr(S.AE_CHECKING_VOLUME))

        detections.extend(self._detect_volume_spikes(audio_path_str))

        # sorted() creates a NEW list -- original is unchanged.
        detections.sort(key=lambda detection: detection.start)
        elapsed = time.perf_counter() - t0
        log.info(
            "AudioEventDetector.detect: done in %.2fs — %d detections.",
            elapsed, len(detections),
        )
        if on_progress:
            on_progress(100, f"{tr(S.PIPE_DONE)} — {len(detections)}")
        return detections

    # ------------------------------------------------------------------
    # PANNs-based event detection
    # ------------------------------------------------------------------

    def _detect_events_panns(
        self,
        audio_path: str,
        on_progress: Optional[Callable[[int, str], None]] = None,
        on_chunk_detections: Optional[Callable[[List["Detection"]], None]] = None,
    ) -> List[Detection]:
        """Classify audio events using PANNs SoundEventDetection.

        For long audio files, processes in chunks of CHUNK_SECONDS to
        avoid memory issues and provide real progress reporting.

        Args:
            audio_path: Path string to the audio file.
            on_progress: Optional callback(percent, label) for progress updates.
            on_chunk_detections: Optional callback fired after each chunk with
                preliminary detections (timestamps offset to absolute position).

        Returns:
            List of Detection objects for recognised AudioSet classes.
        """
        self._load_sed()
        import librosa
        from panns_inference import labels as panns_labels

        CHUNK_SECONDS = 300  # 5 minutes per chunk
        PANNS_SR = 32000

        t0 = time.perf_counter()
        if on_progress:
            on_progress(15, tr(S.AE_LOADING_AUDIO))

        # Load audio at 32 kHz (PANNs native sample rate), mono channel.
        audio, sample_rate = librosa.load(audio_path, sr=PANNS_SR, mono=True)
        total_duration = len(audio) / sample_rate
        log.info(
            "PANNs: loaded audio %.1fs (%d samples @ %d Hz) in %.1fs",
            total_duration, len(audio), sample_rate, time.perf_counter() - t0,
        )

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

    def _detect_volume_spikes(self, audio_path: str) -> List[Detection]:
        """Detect abnormally loud sections using RMS energy analysis.

        Args:
            audio_path: Path string to the audio file.

        Returns:
            List of VOLUME_SPIKE Detection objects.
        """
        import librosa

        t0 = time.perf_counter()
        audio, sample_rate = librosa.load(audio_path, sr=22050, mono=True)
        hop_length = 512
        # librosa.feature.rms returns shape (1, num_frames); take [0] for 1-D array.
        rms_values = librosa.feature.rms(y=audio, hop_length=hop_length)[0]

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

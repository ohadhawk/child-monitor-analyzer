"""Failure handling in the analysis flow: one broken stage must not sink the rest."""

from __future__ import annotations

import numpy as np
import pytest

from monitor.cancellation import AnalysisCancelled
from monitor.gui.strings import S
from monitor.models import AnalysisReport, TranscribedSegment


class _FakeSTT:
    def _load_model(self, on_sub_progress=None):
        pass

    def transcribe(self, audio_path, **kwargs):
        return [TranscribedSegment(text="שלום", start=1.0, end=2.0, words=[])]


class _FailingEvents:
    def __init__(self, exc):
        self._exc = exc

    def _load_sed(self, on_sub_progress=None):
        pass

    def detect(self, audio_path, **kwargs):
        raise self._exc


class _FakeProfanity:
    ai_available = True

    def preload_ai_model(self, on_sub_progress=None):
        pass

    def detect(self, segments):
        return []


def _pipeline(events, stt=None):
    from monitor.pipeline import AnalysisPipeline

    return AnalysisPipeline(stt=stt, audio_events=events, profanity=_FakeProfanity())


@pytest.fixture
def audio_file(tmp_path):
    path = tmp_path / "rec.mp3"
    path.write_bytes(b"not really audio")
    return path


def test_events_failure_keeps_the_transcript(audio_file):
    warnings: list = []
    pipeline = _pipeline(_FailingEvents(RuntimeError("decoder broke")), stt=_FakeSTT())

    report = pipeline.analyze(audio_file, on_warning=warnings.append, stt_model_key="fast")

    assert [s.text for s in report.segments] == ["שלום"]
    assert S.AUDIO_EVENTS_FAILED in warnings
    # No final cache, so reopening the file retries event detection.
    assert AnalysisReport.load_cache(audio_file, "fast") is None


def test_events_failure_still_marks_stt_complete(audio_file):
    from monitor.pipeline import _load_intermediate_stt

    _pipeline(_FailingEvents(RuntimeError("x")), stt=_FakeSTT()).analyze(
        audio_file, stt_model_key="fast",
    )
    cached = _load_intermediate_stt(audio_file, "fast")
    assert cached is not None
    assert cached["gap_fill_completed"] is True


def test_events_failure_is_fatal_in_events_only_mode(audio_file):
    with pytest.raises(RuntimeError, match="decoder broke"):
        _pipeline(_FailingEvents(RuntimeError("decoder broke"))).analyze(
            audio_file, stt_model_key="none",
        )


def test_events_cancellation_is_not_swallowed(audio_file):
    with pytest.raises(AnalysisCancelled):
        _pipeline(_FailingEvents(AnalysisCancelled()), stt=_FakeSTT()).analyze(
            audio_file, stt_model_key="fast",
        )


# ----------------------------------------------------------------------
# Audio events: single shared decode
# ----------------------------------------------------------------------

def test_volume_spikes_keep_frame_timing_at_32k():
    from monitor.audio_events import AudioEventDetector

    sr = 32000
    audio = np.full(sr * 10, 0.01, dtype=np.float32)
    audio[sr * 4: sr * 5] = 0.9  # one loud second
    spikes = AudioEventDetector()._detect_volume_spikes(audio, sr)

    assert len(spikes) == 1
    assert spikes[0].start == pytest.approx(4.0, abs=0.1)
    assert spikes[0].end == pytest.approx(5.0, abs=0.1)


def test_detect_decodes_the_file_once(monkeypatch, tmp_path):
    import monitor.audio_events as ae

    calls: list = []

    def fake_load(path, sr):
        calls.append(sr)
        return np.zeros(sr, dtype=np.float32), sr

    monkeypatch.setattr(ae, "_load_mono", fake_load)
    monkeypatch.setattr(ae.AudioEventDetector, "_detect_events_panns",
                        lambda self, audio, sr, *a: [])
    path = tmp_path / "a.wav"
    path.write_bytes(b"x")

    ae.AudioEventDetector().detect(path)
    assert calls == [ae.PANNS_SAMPLE_RATE]


def test_load_mono_falls_back_to_pyav(monkeypatch, tmp_path):
    import librosa
    import faster_whisper.audio as fwa
    from monitor.audio_events import _load_mono

    def broken(*a, **k):
        raise RuntimeError("libsndfile says no")

    monkeypatch.setattr(librosa, "load", broken)
    monkeypatch.setattr(fwa, "decode_audio",
                        lambda path, sampling_rate: np.ones(sampling_rate, dtype=np.float32))
    path = tmp_path / "a.mp3"
    path.write_bytes(b"x")

    audio, sr = _load_mono(str(path), 32000)
    assert sr == 32000 and len(audio) == 32000


# ----------------------------------------------------------------------
# STT gap-fill works on any format (it used to read WAV only)
# ----------------------------------------------------------------------

def test_eta_waits_for_a_stable_rate_then_extrapolates():
    from monitor.stt import _eta_seconds

    assert _eta_seconds(30, 3000, 10) is None        # too early to tell
    assert _eta_seconds(3000, 3000, 900) is None     # finished
    # 600 s of audio in 600 s -> 1x real time -> 2400 s left.
    assert _eta_seconds(600, 3000, 600) == pytest.approx(2400)

def test_gap_fill_scans_decoded_audio_for_non_wav(monkeypatch):
    from monitor.stt import HebrewSTT, _STT_SAMPLE_RATE

    stt = HebrewSTT.__new__(HebrewSTT)
    seen: list = []

    class _Model:
        def transcribe(self, chunk, **kwargs):
            seen.append(len(chunk))
            return iter(()), None

    stt._model = _Model()
    gap = HebrewSTT._GAP_FILL_MIN_SECONDS + 30
    audio = np.full(_STT_SAMPLE_RATE * gap, 0.2, dtype=np.float32)

    stt._fill_gaps(audio, [], float(gap), None, None)

    assert seen, "gap-fill never ran"
    assert seen[0] == HebrewSTT._GAP_FILL_CHUNK_S * _STT_SAMPLE_RATE

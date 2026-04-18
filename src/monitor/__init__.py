"""
Child Monitor Analyzer -- analyse Hebrew audio for profanity, shouting,
crying, and other events with precise timestamps.

This package provides:
  - Speech-to-text transcription (faster-whisper + ivrit-ai Hebrew model)
  - Audio event detection (PANNs for shout/cry/scream, librosa for volume)
  - Profanity detection (word-list matching + OpenCensor-Hebrew AI)
  - Pipeline orchestration and CLI / GUI interfaces
"""

__version__ = "0.1.0"

"""
Vendored PANNs sound-event-detection inference.

Replaces the ``panns-inference`` and ``torchlibrosa`` PyPI packages, both of
which were last released in early 2023 by a single maintainer.  Only the
inference path actually used by this application is kept.

Removing them also removes ``matplotlib`` (and therefore ``Pillow``) from the
runtime and from the PyInstaller bundle, since the only reason those were
present was an unused module-level ``import matplotlib.pyplot`` in upstream
``panns_inference.models``.

Usage:
    from monitor.vendor.panns import SoundEventDetection, load_labels

    sed = SoundEventDetection(checkpoint_path=verified_path, device="cpu")
    framewise = sed.inference(audio_batch)   # (batch, frames, 527)

See ``LICENSE-third-party.txt`` for attribution and the full list of changes.
"""

from __future__ import annotations

from ._models import (
    CLASSES_NUM,
    FMAX,
    FMIN,
    HOP_SIZE,
    MEL_BINS,
    SAMPLE_RATE,
    WINDOW_SIZE,
    Cnn14_DecisionLevelMax,
)
from .inference import SoundEventDetection
from .labels import load_labels, load_labels_cached

__all__ = [
    "CLASSES_NUM",
    "FMAX",
    "FMIN",
    "HOP_SIZE",
    "MEL_BINS",
    "SAMPLE_RATE",
    "WINDOW_SIZE",
    "Cnn14_DecisionLevelMax",
    "SoundEventDetection",
    "load_labels",
    "load_labels_cached",
]

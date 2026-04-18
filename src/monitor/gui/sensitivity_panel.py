"""
Sensitivity adjustment panel — per-detection-type confidence sliders.

Shows a collapsible panel with one slider per acoustic detection type.
Slider position maps inversely to a confidence threshold: left = low
sensitivity (high threshold = fewer detections), right = high
sensitivity (low threshold = more detections).

Slider values persist across sessions via QSettings.

Usage:
    from monitor.gui.sensitivity_panel import SensitivityPanel

    panel = SensitivityPanel()
    panel.thresholds_changed.connect(on_thresholds_changed)
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from PySide6.QtCore import QSettings, Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ..models import DetectionType
from .strings import tr, S

log = logging.getLogger(__name__)

# QSettings keys.
_SETTINGS_ORG = "ChildMonitorAnalyzer"
_SETTINGS_APP = "monitor-gui"
_SETTINGS_SENSITIVITY_KEY = "sensitivity_thresholds"

# Detection types that have confidence-based thresholds.
# PROFANITY is excluded (word-list based, not threshold-based).
_ADJUSTABLE_TYPES = [
    DetectionType.SHOUT,
    DetectionType.SCREAM,
    DetectionType.CRY,
    DetectionType.WAIL,
    DetectionType.BABY_CRY,
    DetectionType.LAUGHTER,
    DetectionType.VOLUME_SPIKE,
]

# Slider range: 0..100.  Maps inversely to threshold:
#   slider=0   → threshold=0.50  (least sensitive)
#   slider=70  → threshold=0.15  (current default)
#   slider=100 → threshold=0.02  (most sensitive)
_SLIDER_MIN = 0
_SLIDER_MAX = 100
_THRESHOLD_AT_MIN = 0.50   # slider all the way left
_THRESHOLD_AT_MAX = 0.02   # slider all the way right
_DEFAULT_SLIDER = 70       # ≈ 0.15 threshold


def _slider_to_threshold(value: int) -> float:
    """Convert slider position (0-100) to confidence threshold."""
    t = value / _SLIDER_MAX
    return _THRESHOLD_AT_MIN + t * (_THRESHOLD_AT_MAX - _THRESHOLD_AT_MIN)


def _threshold_to_slider(threshold: float) -> int:
    """Convert confidence threshold to slider position."""
    t = (threshold - _THRESHOLD_AT_MIN) / (_THRESHOLD_AT_MAX - _THRESHOLD_AT_MIN)
    return max(_SLIDER_MIN, min(_SLIDER_MAX, round(t * _SLIDER_MAX)))


# Map string key per detection type for the label.
_DT_STRING_KEY = {
    DetectionType.SHOUT: S.DT_SHOUT,
    DetectionType.SCREAM: S.DT_SCREAM,
    DetectionType.CRY: S.DT_CRY,
    DetectionType.WAIL: S.DT_WAIL,
    DetectionType.BABY_CRY: S.DT_BABY_CRY,
    DetectionType.LAUGHTER: S.DT_LAUGHTER,
    DetectionType.VOLUME_SPIKE: S.DT_VOLUME_SPIKE,
}


class SensitivityDialog(QDialog):
    """Dialog with per-detection-type sensitivity sliders.

    Opened from a toolbar button. Sliders resize to fill up to half
    the screen width so they are easy to adjust.
    """

    thresholds_changed = Signal(object)  # Dict[DetectionType, float]

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr(S.SENSITIVITY_TITLE))
        self._settings = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        self._sliders: Dict[DetectionType, QSlider] = {}
        self._build_ui()
        self._load_settings()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(8)

        # Header row with low/high labels.
        header = QHBoxLayout()
        header.addSpacing(100)  # align with slider column
        header.addWidget(QLabel(tr(S.SENSITIVITY_LOW)))
        header.addStretch()
        header.addWidget(QLabel(tr(S.SENSITIVITY_HIGH)))
        layout.addLayout(header)

        # One slider per detection type in a grid.
        grid = QGridLayout()
        grid.setSpacing(8)

        for row, dt in enumerate(_ADJUSTABLE_TYPES):
            label = QLabel(tr(_DT_STRING_KEY[dt]))
            label.setFixedWidth(90)
            grid.addWidget(label, row, 0)

            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(_SLIDER_MIN, _SLIDER_MAX)
            slider.setValue(_DEFAULT_SLIDER)
            slider.setTickPosition(QSlider.TickPosition.TicksBelow)
            slider.setTickInterval(10)
            slider.setMinimumWidth(200)
            slider.valueChanged.connect(self._on_slider_changed)
            self._sliders[dt] = slider
            grid.addWidget(slider, row, 1)

        layout.addLayout(grid)

    def showEvent(self, event) -> None:
        """Size the dialog to ~half the screen on first show."""
        super().showEvent(event)
        screen = self.screen()
        if screen is not None:
            avail = screen.availableGeometry()
            w = avail.width() // 2
            h = min(self.sizeHint().height() + 20, avail.height() // 2)
            self.resize(w, h)
            # Centre on screen.
            self.move(
                avail.x() + (avail.width() - w) // 2,
                avail.y() + (avail.height() - h) // 2,
            )

    def _on_slider_changed(self) -> None:
        self._save_settings()
        self.thresholds_changed.emit(self.get_thresholds())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_thresholds(self) -> Dict[DetectionType, float]:
        """Return current threshold per detection type."""
        return {
            dt: _slider_to_threshold(slider.value())
            for dt, slider in self._sliders.items()
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_settings(self) -> None:
        import json
        data = {dt.value: slider.value() for dt, slider in self._sliders.items()}
        self._settings.setValue(_SETTINGS_SENSITIVITY_KEY, json.dumps(data))

    def _load_settings(self) -> None:
        import json
        raw = self._settings.value(_SETTINGS_SENSITIVITY_KEY, "")
        if not raw:
            return
        try:
            data = json.loads(raw) if isinstance(raw, str) else {}
            for dt, slider in self._sliders.items():
                val = data.get(dt.value)
                if val is not None:
                    slider.blockSignals(True)
                    slider.setValue(int(val))
                    slider.blockSignals(False)
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

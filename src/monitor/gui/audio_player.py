"""
Integrated audio player widget with play/pause, +/-10s skip, seek, and volume.

Provides a PySide6-based audio playback control panel that can be embedded
in the main application window. Emits ``position_changed`` signals so the
report table can highlight the current detection as playback progresses.

Usage:
    from monitor.gui.audio_player import AudioPlayerWidget

    player = AudioPlayerWidget()
    player.load("recording.wav")
    player.seek_to(42.5)  # jumps to 42.5 seconds and auto-plays
"""

from __future__ import annotations

import bisect
import logging
from typing import List, Optional

from PySide6.QtCore import Qt, QUrl, Signal, Slot
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QStyle,
    QStyleOptionSlider,
    QVBoxLayout,
    QWidget,
)

from .strings import tr, S
from .player_icons import icon_play, icon_pause, icon_skip_back, icon_skip_forward, icon_volume

log = logging.getLogger(__name__)

# ===========================
# CONSTANTS
# ===========================

# Skip duration in milliseconds (10 seconds).
SKIP_DURATION_MS = 10_000

# Seconds of context to include before an event when jumping to it.
EVENT_CONTEXT_SECONDS = 2.0

# Tolerance (seconds) around the current position when deciding which event
# counts as "previous" — prevents clicking "prev" landing on the current event.
EVENT_NEAR_TOLERANCE = 0.5

# ===========================
# HELPER FUNCTIONS
# ===========================


def _format_time(milliseconds: int) -> str:
    """Format milliseconds as MM:SS.

    Args:
        milliseconds: Time value in milliseconds.

    Returns:
        Formatted string, e.g. "02:35".

    Example:
        _format_time(155000) -> "02:35"
    """
    # divmod(155, 60) -> (2, 35)
    total_seconds = max(0, milliseconds) // 1000
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes:02d}:{seconds:02d}"


# ===========================
# CLICKABLE SLIDER
# ===========================


class _ClickableSlider(QSlider):
    """QSlider that jumps directly to the clicked position.

    The default QSlider only moves by a page step when the user clicks
    on the groove. This subclass makes the handle jump to the exact
    click location, which is the expected behaviour for a seek bar.
    """

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            opt = QStyleOptionSlider()
            self.initStyleOption(opt)
            groove = self.style().subControlRect(
                QStyle.ComplexControl.CC_Slider, opt,
                QStyle.SubControl.SC_SliderGroove, self,
            )
            if self.orientation() == Qt.Orientation.Horizontal:
                pos = event.position().x()
                val = QStyle.sliderValueFromPosition(
                    self.minimum(), self.maximum(),
                    int(pos - groove.x()), groove.width(),
                )
            else:
                pos = event.position().y()
                val = QStyle.sliderValueFromPosition(
                    self.minimum(), self.maximum(),
                    int(pos - groove.y()), groove.height(),
                    upsideDown=True,
                )
            self.setValue(val)
            self.sliderMoved.emit(val)
            event.accept()
        super().mousePressEvent(event)


# ===========================
# CORE WIDGET
# ===========================


class AudioPlayerWidget(QWidget):
    """Audio player with play/pause, +/-10s skip, seek slider, and volume.

    Signals:
        position_changed: Emitted with current playback position in seconds.
    """

    position_changed = Signal(float)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        # --- Media backend ---
        self._player = QMediaPlayer(self)
        self._audio_output = QAudioOutput(self)
        self._player.setAudioOutput(self._audio_output)
        self._audio_output.setVolume(0.7)

        # Sorted list of event timestamps (seconds) for prev/next navigation.
        self._event_times: List[float] = []

        # --- Build UI ---
        self._build_controls()
        self._build_layout()
        self._connect_signals()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_controls(self) -> None:
        """Create all UI control widgets."""
        self._btn_back = QPushButton()
        self._btn_back.setIcon(icon_skip_back())
        self._btn_play = QPushButton()
        self._btn_play.setIcon(icon_play())
        self._btn_forward = QPushButton()
        self._btn_forward.setIcon(icon_skip_forward())

        for button in (self._btn_back, self._btn_play, self._btn_forward):
            button.setFixedHeight(32)
            button.setFixedWidth(40)

        # --- Event navigation buttons (prev / next detection) ---
        self._btn_prev_event = QPushButton(tr(S.PLAYER_PREV_EVENT))
        self._btn_next_event = QPushButton(tr(S.PLAYER_NEXT_EVENT))
        for button in (self._btn_prev_event, self._btn_next_event):
            button.setFixedHeight(32)
            button.setEnabled(False)  # enabled once events are loaded

        self._seek_slider = _ClickableSlider(Qt.Orientation.Horizontal)
        self._seek_slider.setRange(0, 0)
        self._seek_slider.setLayoutDirection(Qt.LayoutDirection.LeftToRight)

        self._lbl_time = QLabel("00:00 / 00:00")
        self._lbl_time.setFixedWidth(100)

        self._volume_slider = QSlider(Qt.Orientation.Horizontal)
        self._volume_slider.setRange(0, 100)
        self._volume_slider.setValue(70)
        self._volume_slider.setFixedWidth(80)
        self._lbl_volume = QLabel()
        self._lbl_volume.setPixmap(icon_volume().pixmap(20, 20))

    def _build_layout(self) -> None:
        """Arrange controls into the widget layout."""
        controls_row = QHBoxLayout()
        controls_row.addWidget(self._btn_forward)
        controls_row.addWidget(self._btn_play)
        controls_row.addWidget(self._btn_back)
        controls_row.addStretch()
        # Event-navigation buttons centred over the seek slider below.
        # The layout direction is RTL, so widgets added first appear on the
        # RIGHT. To show "prev" on the left of "next", add "next" first.
        controls_row.addWidget(self._btn_next_event)
        controls_row.addWidget(self._btn_prev_event)
        controls_row.addStretch()
        controls_row.addWidget(self._lbl_volume)
        controls_row.addWidget(self._volume_slider)

        seek_row = QHBoxLayout()
        seek_row.addWidget(self._seek_slider)
        seek_row.addWidget(self._lbl_time)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addLayout(controls_row)
        layout.addLayout(seek_row)

    def _connect_signals(self) -> None:
        """Wire up button clicks and media player signals."""
        self._btn_play.clicked.connect(self._toggle_play)
        self._btn_back.clicked.connect(self._skip_back)
        self._btn_forward.clicked.connect(self._skip_forward)
        self._btn_prev_event.clicked.connect(self._jump_prev_event)
        self._btn_next_event.clicked.connect(self._jump_next_event)
        self._seek_slider.sliderMoved.connect(self._seek)
        self._volume_slider.valueChanged.connect(self._set_volume)

        self._player.positionChanged.connect(self._on_position_changed)
        self._player.durationChanged.connect(self._on_duration_changed)
        self._player.playbackStateChanged.connect(self._on_state_changed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, file_path: str) -> None:
        """Load an audio file for playback.

        Args:
            file_path: Absolute path to the audio file.
        """
        log.info("Loading audio file: %s", file_path)
        self._player.setSource(QUrl.fromLocalFile(file_path))

    def seek_to(self, seconds: float) -> None:
        """Seek to a position in seconds and start playing.

        Args:
            seconds: Target position in seconds.
        """
        self._player.setPosition(int(seconds * 1000))
        self._player.play()

    def set_event_times(self, times: List[float]) -> None:
        """Update the list of event timestamps for prev/next navigation.

        Args:
            times: Event start times (seconds). Will be sorted internally.
        """
        self._event_times = sorted(float(t) for t in times)
        has_events = bool(self._event_times)
        self._btn_prev_event.setEnabled(has_events)
        self._btn_next_event.setEnabled(has_events)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    @Slot()
    def _toggle_play(self) -> None:
        """Toggle between play and pause states."""
        if self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._player.pause()
        else:
            self._player.play()

    @Slot()
    def _skip_back(self) -> None:
        """Jump backward by SKIP_DURATION_MS milliseconds."""
        # max() prevents seeking to a negative position.
        position = max(0, self._player.position() - SKIP_DURATION_MS)
        self._player.setPosition(position)

    @Slot()
    def _skip_forward(self) -> None:
        """Jump forward by SKIP_DURATION_MS milliseconds."""
        # min() prevents seeking past the end of the audio.
        position = min(self._player.duration(), self._player.position() + SKIP_DURATION_MS)
        self._player.setPosition(position)

    @Slot()
    def _jump_prev_event(self) -> None:
        """Jump to the previous event on the timeline and start playing."""
        if not self._event_times:
            return
        current_sec = self._player.position() / 1000.0
        # Find the largest event time strictly before (current - tolerance).
        cutoff = current_sec - EVENT_NEAR_TOLERANCE
        idx = bisect.bisect_left(self._event_times, cutoff)
        if idx == 0:
            target = self._event_times[0]
        else:
            target = self._event_times[idx - 1]
        self.seek_to(max(0.0, target - EVENT_CONTEXT_SECONDS))

    @Slot()
    def _jump_next_event(self) -> None:
        """Jump to the next event on the timeline and start playing."""
        if not self._event_times:
            return
        current_sec = self._player.position() / 1000.0
        # Find the smallest event time strictly after (current + tolerance).
        cutoff = current_sec + EVENT_NEAR_TOLERANCE
        idx = bisect.bisect_right(self._event_times, cutoff)
        if idx >= len(self._event_times):
            target = self._event_times[-1]
        else:
            target = self._event_times[idx]
        self.seek_to(max(0.0, target - EVENT_CONTEXT_SECONDS))

    @Slot(int)
    def _seek(self, position_ms: int) -> None:
        """Seek to the position indicated by the slider.

        Args:
            position_ms: Target position in milliseconds.
        """
        self._player.setPosition(position_ms)

    @Slot(int)
    def _set_volume(self, value: int) -> None:
        """Update audio output volume from the volume slider.

        Args:
            value: Volume level, 0-100.
        """
        self._audio_output.setVolume(value / 100.0)

    @Slot(int)
    def _on_position_changed(self, position_ms: int) -> None:
        """Handle media player position updates.

        Args:
            position_ms: Current playback position in milliseconds.
        """
        # Only update slider programmatically when user is not dragging it.
        if not self._seek_slider.isSliderDown():
            self._seek_slider.setValue(position_ms)
        self._lbl_time.setText(
            f"{_format_time(position_ms)} / {_format_time(self._player.duration())}"
        )
        self.position_changed.emit(position_ms / 1000.0)

    @Slot(int)
    def _on_duration_changed(self, duration_ms: int) -> None:
        """Update seek slider range when media duration becomes known.

        Args:
            duration_ms: Total media duration in milliseconds.
        """
        self._seek_slider.setRange(0, duration_ms)

    @Slot(QMediaPlayer.PlaybackState)
    def _on_state_changed(self, state: QMediaPlayer.PlaybackState) -> None:
        """Update play button text to reflect current playback state.

        Args:
            state: New playback state.
        """
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self._btn_play.setIcon(icon_pause())
        else:
            self._btn_play.setIcon(icon_play())

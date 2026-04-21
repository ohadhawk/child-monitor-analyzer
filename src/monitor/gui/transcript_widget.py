"""
Transcript viewer widget — displays STT segments with timestamps.

Shows each transcribed segment as a clickable row with the timestamp
on the left and the Hebrew text on the right.  Clicking a segment
seeks the audio player to that position.  The current playback
position is highlighted automatically.

Usage:
    from monitor.gui.transcript_widget import TranscriptWidget

    transcript = TranscriptWidget()
    transcript.load_segments(report.segments)
    transcript.play_requested.connect(audio_player.seek_to)
"""

from __future__ import annotations

import logging
from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QKeyEvent, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..models import Detection, DetectionType, TranscribedSegment, DETECTION_LABELS_HE
from .strings import tr, S

log = logging.getLogger(__name__)

# Seconds of context before the segment start when clicking.
_PLAY_CONTEXT_SECONDS = 1.0

# How close (seconds) the playback must be to highlight a segment.
_HIGHLIGHT_MAX_DISTANCE = 3.0

# Emoji per detection type for marker lines.
_DETECTION_EMOJI = {
    DetectionType.PROFANITY: "🤬",
    DetectionType.SHOUT: "🗣️",
    DetectionType.SCREAM: "😱",
    DetectionType.CRY: "😢",
    DetectionType.WAIL: "😭",
    DetectionType.BABY_CRY: "👶",
    DetectionType.LAUGHTER: "😂",
    DetectionType.VOLUME_SPIKE: "🔊",
}

# Colours for detection marker blocks (same palette as report table).
_MARKER_COLORS = {
    DetectionType.PROFANITY: QColor(255, 200, 200),
    DetectionType.SHOUT: QColor(255, 225, 180),
    DetectionType.SCREAM: QColor(255, 210, 170),
    DetectionType.CRY: QColor(200, 220, 255),
    DetectionType.WAIL: QColor(200, 220, 255),
    DetectionType.BABY_CRY: QColor(210, 200, 255),
    DetectionType.LAUGHTER: QColor(220, 255, 220),
    DetectionType.VOLUME_SPIKE: QColor(230, 230, 230),
}


def _format_time(seconds: float) -> str:
    total = int(seconds)
    h, remainder = divmod(total, 3600)
    m, s = divmod(remainder, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class TranscriptWidget(QWidget):
    """Scrollable transcript panel with timestamps and click-to-seek."""

    play_requested = Signal(float)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._segments: List[TranscribedSegment] = []
        self._detections: List[Detection] = []
        self._block_to_segment: dict[int, int] = {}  # block number -> segment index
        self._block_to_detection: dict[int, int] = {}  # block number -> detection index
        self._highlighted_block: int = -1
        self._visible_types: Optional[set] = None  # None = show all

        # Search state — stores QTextCursor objects with native Qt positions.
        self._search_cursors: List[QTextCursor] = []
        self._current_match: int = -1

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        title = QLabel(tr(S.TRANSCRIPT_TITLE))
        title.setStyleSheet(
            "font-weight: bold; font-size: 13px; padding: 4px;"
        )
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # --- Search bar ---
        search_row = QHBoxLayout()
        search_row.setContentsMargins(4, 0, 4, 0)
        search_row.setSpacing(2)

        self._search_input = QLineEdit()
        self._search_input.setPlaceholderText(tr(S.TRANSCRIPT_SEARCH))
        self._search_input.setClearButtonEnabled(True)
        self._search_input.textChanged.connect(self._do_search)
        self._search_input.installEventFilter(self)
        search_row.addWidget(self._search_input, stretch=1)

        self._btn_prev = QPushButton("▲")
        self._btn_prev.setFixedWidth(28)
        self._btn_prev.clicked.connect(self._go_prev)
        search_row.addWidget(self._btn_prev)

        self._btn_next = QPushButton("▼")
        self._btn_next.setFixedWidth(28)
        self._btn_next.clicked.connect(self._go_next)
        search_row.addWidget(self._btn_next)

        self._lbl_match_count = QLabel("0/0")
        self._lbl_match_count.setFixedWidth(48)
        self._lbl_match_count.setAlignment(Qt.AlignmentFlag.AlignCenter)
        search_row.addWidget(self._lbl_match_count)

        layout.addLayout(search_row)

        self._text_edit = QTextEdit()
        self._text_edit.setReadOnly(True)
        self._text_edit.setFont(QFont("Segoe UI", 11))
        self._text_edit.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self._text_edit.setStyleSheet(
            "QTextEdit { border: 1px solid #ddd; padding: 4px; }"
        )
        # Enable click handling.
        self._text_edit.mousePressEvent = self._on_click
        layout.addWidget(self._text_edit, stretch=1)

    def load_segments(
        self,
        segments: List[TranscribedSegment],
        detections: Optional[List[Detection]] = None,
    ) -> None:
        self._segments = segments
        self._detections = sorted(detections or [], key=lambda d: d.start)
        self._block_to_segment.clear()
        self._block_to_detection.clear()
        self._highlighted_block = -1
        self._text_edit.clear()

        if not segments:
            self._text_edit.setPlaceholderText(tr(S.TRANSCRIPT_EMPTY))
            return

        # Build a merged timeline of (time, kind, index) entries.
        # kind: 's' = segment, 'd' = detection
        timeline: list = []
        for i, seg in enumerate(segments):
            timeline.append((seg.start, "s", i))
        for i, det in enumerate(self._detections):
            # Skip hidden detection types.
            if self._visible_types is not None:
                label = DETECTION_LABELS_HE.get(det.type, det.type.value)
                if label not in self._visible_types:
                    continue
            timeline.append((det.start, "d", i))
        timeline.sort(key=lambda x: (x[0], 0 if x[1] == "d" else 1))

        cursor = self._text_edit.textCursor()

        # Timestamp format.
        ts_fmt = QTextCharFormat()
        ts_fmt.setForeground(QColor(100, 100, 100))
        ts_fmt.setFontWeight(QFont.Weight.Bold)

        # Text format.
        text_fmt = QTextCharFormat()
        text_fmt.setForeground(QColor(0, 0, 0))

        first = True
        for _time, kind, idx in timeline:
            if not first:
                cursor.insertBlock()
            first = False

            block_num = cursor.blockNumber()

            if kind == "s":
                self._block_to_segment[block_num] = idx
                seg = segments[idx]
                ts = _format_time(seg.start)
                cursor.insertText(f"[{ts}]  ", ts_fmt)
                cursor.insertText(seg.text.strip(), text_fmt)
            else:
                self._block_to_detection[block_num] = idx
                det = self._detections[idx]
                emoji = _DETECTION_EMOJI.get(det.type, "⚠️")
                label = DETECTION_LABELS_HE.get(det.type, det.type.value)
                ts = _format_time(det.start)

                marker_fmt = QTextCharFormat()
                marker_fmt.setBackground(
                    _MARKER_COLORS.get(det.type, QColor(240, 240, 240))
                )
                marker_fmt.setFontWeight(QFont.Weight.Bold)
                marker_fmt.setForeground(QColor(80, 80, 80))
                cursor.insertText(f"[{ts}] {emoji} {label}", marker_fmt)

        # Scroll to top.
        self._text_edit.moveCursor(QTextCursor.MoveOperation.Start)
        log.info("Transcript loaded: %d segments, %d detection markers.",
                 len(segments), len(self._block_to_detection))

    def set_visible_types(self, visible_types: Optional[set] = None) -> None:
        """Update which detection types show markers and reload."""
        self._visible_types = visible_types
        self.load_segments(self._segments, self._detections)

    def highlight_time(self, current_seconds: float) -> None:
        if not self._segments and not self._block_to_detection:
            return

        # Consider every block that has a known start time (segment OR
        # detection marker). Previously detection markers were skipped, so
        # audio-only events like crying/screaming never highlighted.
        # candidates: list of (distance, block_num)
        candidates: list = []
        for block_num, seg_idx in self._block_to_segment.items():
            if 0 <= seg_idx < len(self._segments):
                candidates.append(
                    (abs(self._segments[seg_idx].start - current_seconds), block_num)
                )
        for block_num, det_idx in self._block_to_detection.items():
            if 0 <= det_idx < len(self._detections):
                candidates.append(
                    (abs(self._detections[det_idx].start - current_seconds), block_num)
                )
        if not candidates:
            return

        best_dist, target_block = min(candidates, key=lambda x: x[0])
        if best_dist > _HIGHLIGHT_MAX_DISTANCE:
            return
        if target_block == self._highlighted_block:
            return

        self._highlighted_block = target_block
        self._apply_extra_selections()

        # Scroll to the highlighted block.
        doc = self._text_edit.document()
        block = doc.findBlockByNumber(target_block)
        if block.isValid():
            cursor = QTextCursor(block)
            self._text_edit.setTextCursor(cursor)
            self._text_edit.ensureCursorVisible()

    def _on_click(self, event) -> None:
        # Get the block under the click.
        pos = event.pos()
        cursor = self._text_edit.cursorForPosition(pos)
        block_num = cursor.blockNumber()

        seg_idx = self._block_to_segment.get(block_num)
        if seg_idx is not None and seg_idx < len(self._segments):
            seg = self._segments[seg_idx]
            seek_pos = max(0.0, seg.start - _PLAY_CONTEXT_SECONDS)
            self.play_requested.emit(seek_pos)
        else:
            det_idx = self._block_to_detection.get(block_num)
            if det_idx is not None and det_idx < len(self._detections):
                self.play_requested.emit(self._detections[det_idx].start)

        # Let QTextEdit handle the event for scrolling etc.
        QTextEdit.mousePressEvent(self._text_edit, event)

    # ------------------------------------------------------------------
    # Extra-selections (non-destructive overlay highlights)
    # ------------------------------------------------------------------

    def _apply_extra_selections(self) -> None:
        """Merge playback + search highlights into setExtraSelections."""
        selections: list = []

        # Playback highlight (strong amber bar across the full width).
        if self._highlighted_block >= 0:
            doc = self._text_edit.document()
            block = doc.findBlockByNumber(self._highlighted_block)
            if block.isValid():
                cursor = QTextCursor(block)
                cursor.movePosition(
                    QTextCursor.MoveOperation.EndOfBlock,
                    QTextCursor.MoveMode.KeepAnchor,
                )
                sel = QTextEdit.ExtraSelection()
                sel.format.setBackground(QColor(255, 213, 74))   # strong amber
                sel.format.setForeground(QColor(0, 0, 0))
                sel.format.setFontWeight(QFont.Weight.Bold)
                sel.format.setProperty(
                    QTextCharFormat.Property.FullWidthSelection, True
                )
                sel.cursor = cursor
                selections.append(sel)

        # Search highlight (all matches light orange, current deep orange).
        for i, cur in enumerate(self._search_cursors):
            sel = QTextEdit.ExtraSelection()
            if i == self._current_match:
                sel.format.setBackground(QColor(255, 140, 0))   # deep orange
            else:
                sel.format.setBackground(QColor(255, 200, 100))  # light orange
            sel.cursor = cur
            selections.append(sel)

        self._text_edit.setExtraSelections(selections)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def eventFilter(self, obj, event) -> bool:
        """Handle Enter/Shift+Enter in the search input."""
        if obj is self._search_input and isinstance(event, QKeyEvent):
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                    self._go_prev()
                else:
                    self._go_next()
                return True
        return super().eventFilter(obj, event)

    def _do_search(self, text: str) -> None:
        """Find all occurrences of *text* using QTextDocument.find()."""
        self._search_cursors.clear()
        self._current_match = -1

        if not text:
            self._lbl_match_count.setText("0/0")
            self._apply_extra_selections()
            return

        doc = self._text_edit.document()
        cursor = QTextCursor(doc)  # positioned at start

        while True:
            found = doc.find(text, cursor)
            if found.isNull():
                break
            self._search_cursors.append(QTextCursor(found))
            cursor = found  # continue searching from the match

        if not self._search_cursors:
            self._lbl_match_count.setText("0/0")
            self._apply_extra_selections()
            return

        self._current_match = 0
        self._apply_extra_selections()
        self._scroll_to_current_match()

    def _scroll_to_current_match(self) -> None:
        """Scroll to the current search match and update counter."""
        if self._current_match < 0 or self._current_match >= len(self._search_cursors):
            return
        cur = self._search_cursors[self._current_match]
        self._text_edit.setTextCursor(cur)
        self._text_edit.ensureCursorVisible()
        self._lbl_match_count.setText(
            f"{self._current_match + 1}/{len(self._search_cursors)}"
        )

    def _seek_to_current_match(self) -> None:
        """Seek audio to the timestamp of the current search match."""
        if self._current_match < 0 or self._current_match >= len(self._search_cursors):
            return
        block_num = self._search_cursors[self._current_match].blockNumber()
        seg_idx = self._block_to_segment.get(block_num)
        if seg_idx is not None and seg_idx < len(self._segments):
            seek_pos = max(0.0, self._segments[seg_idx].start - _PLAY_CONTEXT_SECONDS)
            self.play_requested.emit(seek_pos)
        else:
            det_idx = self._block_to_detection.get(block_num)
            if det_idx is not None and det_idx < len(self._detections):
                self.play_requested.emit(self._detections[det_idx].start)

    def _go_next(self) -> None:
        if not self._search_cursors:
            return
        self._current_match = (self._current_match + 1) % len(self._search_cursors)
        self._apply_extra_selections()
        self._scroll_to_current_match()
        self._seek_to_current_match()

    def _go_prev(self) -> None:
        if not self._search_cursors:
            return
        self._current_match = (self._current_match - 1) % len(self._search_cursors)
        self._apply_extra_selections()
        self._scroll_to_current_match()
        self._seek_to_current_match()

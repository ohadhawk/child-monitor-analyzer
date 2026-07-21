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
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import Qt, QSettings, Signal
from PySide6.QtGui import QColor, QFont, QKeyEvent, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
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


# QSettings location (shared app-wide store) and export-preference keys.
_SETTINGS_ORG = "ChildMonitorAnalyzer"
_SETTINGS_APP = "monitor-gui"
_EXPORT_FORMAT_KEY = "export_format"
_EXPORT_TIMESTAMPS_KEY = "export_include_timestamps"
_EXPORT_EVENTS_KEY = "export_include_events"


class TranscriptExportDialog(QDialog):
    """Ask the user for transcript export format and content options.

    Remembers the last choices via QSettings.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr(S.EXPORT_DIALOG_TITLE))
        self.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        self._settings = QSettings(_SETTINGS_ORG, _SETTINGS_APP)

        layout = QVBoxLayout(self)

        # --- Format selection ---
        fmt_row = QHBoxLayout()
        fmt_row.addWidget(QLabel(tr(S.EXPORT_FORMAT_LABEL)))
        self._cmb_format = QComboBox()
        self._cmb_format.addItem(tr(S.EXPORT_FORMAT_TXT), "txt")
        self._cmb_format.addItem(tr(S.EXPORT_FORMAT_DOCX), "docx")
        fmt_row.addWidget(self._cmb_format, stretch=1)
        layout.addLayout(fmt_row)

        # --- Content options ---
        self._chk_timestamps = QCheckBox(tr(S.EXPORT_INCLUDE_TIMESTAMPS))
        self._chk_events = QCheckBox(tr(S.EXPORT_INCLUDE_EVENTS))
        layout.addWidget(self._chk_timestamps)
        layout.addWidget(self._chk_events)

        # --- Buttons ---
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Save).setText(tr(S.EXPORT_OK))
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText(tr(S.EXPORT_CANCEL))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._load_prefs()

    def _load_prefs(self) -> None:
        """Populate the widgets from the last saved choices."""
        fmt = self._settings.value(_EXPORT_FORMAT_KEY, "txt", type=str)
        if fmt not in ("txt", "docx"):
            fmt = "txt"
        idx = self._cmb_format.findData(fmt)
        self._cmb_format.setCurrentIndex(max(0, idx))
        self._chk_timestamps.setChecked(
            self._settings.value(_EXPORT_TIMESTAMPS_KEY, True, type=bool)
        )
        self._chk_events.setChecked(
            self._settings.value(_EXPORT_EVENTS_KEY, False, type=bool)
        )

    def accept(self) -> None:
        """Persist the current choices before closing."""
        self._settings.setValue(_EXPORT_FORMAT_KEY, self.selected_format())
        self._settings.setValue(_EXPORT_TIMESTAMPS_KEY, self.include_timestamps())
        self._settings.setValue(_EXPORT_EVENTS_KEY, self.include_events())
        super().accept()

    def selected_format(self) -> str:
        return self._cmb_format.currentData() or "txt"

    def include_timestamps(self) -> bool:
        return self._chk_timestamps.isChecked()

    def include_events(self) -> bool:
        return self._chk_events.isChecked()


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

        # Source context used to build the default export filename.
        self._audio_stem: Optional[str] = None
        self._model_key: Optional[str] = None

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

        self._btn_download = QPushButton("↓")
        self._btn_download.setFixedWidth(28)
        self._btn_download.setToolTip(tr(S.TRANSCRIPT_DOWNLOAD))
        self._btn_download.clicked.connect(self._download_transcript)
        search_row.addWidget(self._btn_download)

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
        if detections is not None:
            self._detections = sorted(detections, key=lambda d: d.start)
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

    def set_detections(self, detections: List[Detection]) -> None:
        """Replace the visible detections and rebuild, keeping segments.

        Driven by the report table's filter state so that the transcript
        shows exactly the same events as the report (type + details +
        sensitivity filters all applied).
        """
        self.load_segments(self._segments, detections)

    def _build_transcript_lines(
        self, include_timestamps: bool, include_events: bool
    ) -> list[str]:
        """Build transcript lines from current segments (+ optional events)."""
        items: list[tuple[float, str]] = []  # (start, text)
        for seg in self._segments:
            items.append((seg.start, seg.text.strip()))
        if include_events:
            for det in self._detections:
                emoji = _DETECTION_EMOJI.get(det.type, "⚠️")
                label = DETECTION_LABELS_HE.get(det.type, det.type.value)
                items.append((det.start, f"[{emoji} {label}]"))
        items.sort(key=lambda x: x[0])
        if include_timestamps:
            return [f"[{_format_time(start)}] {text}" for start, text in items]
        return [text for _start, text in items]

    def set_export_source(
        self, audio_path: Optional[str], model_key: Optional[str],
    ) -> None:
        """Record the audio file and STT model used for the default export name."""
        self._audio_stem = Path(audio_path).stem.strip() if audio_path else None
        self._model_key = model_key

    def _default_export_name(self) -> str:
        """Build the default export base name: '<audio-file> <תמלול …>'."""
        model_label = {
            "thorough": tr(S.EXPORT_NAME_THOROUGH),
            "fast": tr(S.EXPORT_NAME_FAST),
        }.get(self._model_key or "")
        if self._audio_stem and model_label:
            return f"{self._audio_stem} {model_label}"
        if self._audio_stem:
            return self._audio_stem
        return "transcript"

    def _download_transcript(self) -> None:
        """Prompt for options + a path and save the currently-shown transcript."""
        if not self._segments and not self._detections:
            QMessageBox.information(
                self, tr(S.EXPORT_DIALOG_TITLE),
                tr(S.TRANSCRIPT_DOWNLOAD_EMPTY),
            )
            return

        dialog = TranscriptExportDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        fmt = dialog.selected_format()
        include_ts = dialog.include_timestamps()
        include_ev = dialog.include_events()

        suffix = ".docx" if fmt == "docx" else ".txt"
        file_filter = (
            "Word Document (*.docx)" if fmt == "docx" else "Text files (*.txt)"
        )
        default_name = self._default_export_name() + suffix
        path, _ = QFileDialog.getSaveFileName(
            self, tr(S.EXPORT_DIALOG_TITLE), default_name, file_filter,
        )
        if not path:
            return
        # The chosen format is authoritative — force the correct suffix even
        # if the user typed a different/no extension.
        path = str(Path(path).with_suffix(suffix))

        lines = self._build_transcript_lines(include_ts, include_ev)
        try:
            if fmt == "docx":
                self._write_docx(path, lines)
            else:
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write("\n".join(lines))
            log.info("Transcript saved to %s", path)
            QMessageBox.information(
                self, tr(S.EXPORT_DIALOG_TITLE), tr(S.EXPORT_SAVED),
            )
        except Exception as exc:  # OSError, ImportError (docx), etc.
            log.exception("Failed to save transcript to %s", path)
            QMessageBox.warning(
                self, tr(S.EXPORT_DIALOG_TITLE),
                f"{tr(S.EXPORT_FAILED)} {exc}",
            )

    @staticmethod
    def _write_docx(path: str, lines: list[str]) -> None:
        """Write *lines* to a .docx with right-to-left, right-aligned text."""
        from docx import Document
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.oxml import OxmlElement
        from docx.oxml.ns import qn

        doc = Document()
        # Make the default 'Normal' style right-to-left so every paragraph
        # (including empty ones) renders as RTL.
        normal_ppr = doc.styles["Normal"].element.get_or_add_pPr()
        if normal_ppr.find(qn("w:bidi")) is None:
            normal_ppr.append(OxmlElement("w:bidi"))

        for line in lines:
            para = doc.add_paragraph()
            para.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            # Paragraph-level RTL (belt-and-suspenders alongside the style).
            p_pr = para._p.get_or_add_pPr()
            if p_pr.find(qn("w:bidi")) is None:
                p_pr.append(OxmlElement("w:bidi"))
            run = para.add_run(line)
            # Run-level RTL so mixed neutrals (brackets, timestamps) order
            # correctly within the Hebrew text.
            r_pr = run._r.get_or_add_rPr()
            rtl = OxmlElement("w:rtl")
            r_pr.append(rtl)

        doc.save(path)


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

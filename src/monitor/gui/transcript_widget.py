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

import io
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

from ..gdrive.naming import build_transcript_base_name
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


#: Elements that must follow ``w:bidi`` inside ``w:pPr`` (ECMA-376 CT_PPr).
#: Word tolerates the wrong order; the Google Docs importer discards the
#: misplaced element, which silently loses right-to-left paragraph direction.
_AFTER_BIDI = (
    "w:adjustRightInd", "w:snapToGrid", "w:spacing", "w:ind",
    "w:contextualSpacing", "w:mirrorIndents", "w:suppressOverlap", "w:jc",
    "w:textDirection", "w:textAlignment", "w:textboxTightWrap", "w:outlineLvl",
    "w:divId", "w:cnfStyle", "w:rPr", "w:sectPr", "w:pPrChange",
)


def _set_bidi(p_pr) -> None:
    """Mark a ``w:pPr`` right-to-left, in schema order."""
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    if p_pr.find(qn("w:bidi")) is not None:
        return
    p_pr.insert_element_before(OxmlElement("w:bidi"), *_AFTER_BIDI)


#: Font for the exported .docx. A humanist sans renders Hebrew far better than
#: the serif default python-docx inherits from its template.
EXPORT_FONT = "Arial"


def _set_font(r_pr, name: str) -> None:
    """Apply *name* to Latin **and** complex-script runs.

    Hebrew is a complex script, so it takes ``w:cs``; setting only the Latin
    attributes would leave the Hebrew on the template's default font.
    """
    from docx.oxml.ns import qn

    fonts = r_pr.get_or_add_rFonts()
    for attribute in ("w:ascii", "w:hAnsi", "w:cs"):
        fonts.set(qn(attribute), name)


# QSettings location (shared app-wide store) and export-preference keys.
_SETTINGS_ORG = "ChildMonitorAnalyzer"
_SETTINGS_APP = "monitor-gui"
_EXPORT_FORMAT_KEY = "export_format"
#: Upload keeps its own format preference: txt converts to a Google Doc with no
#: paragraph alignment, so a local txt export must not silently downgrade it.
_UPLOAD_FORMAT_KEY = "upload_format"
_EXPORT_TIMESTAMPS_KEY = "export_include_timestamps"
_EXPORT_EVENTS_KEY = "export_include_events"


class TranscriptExportDialog(QDialog):
    """Ask the user for transcript export format and content options.

    One dialog serves both destinations. Forking it into a separate upload
    dialog would let the two option sets drift apart, and the requirement is
    that the same choices apply whether the transcript is saved locally or
    uploaded.

    Remembers the last choices via QSettings.
    """

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        *,
        mode: str = "download",
        target_description: str = "",
        default_name: str = "",
    ) -> None:
        super().__init__(parent)
        if mode not in ("download", "upload"):
            raise ValueError(f"unknown export dialog mode: {mode!r}")
        self._mode = mode
        self.setWindowTitle(
            tr(S.UPLOAD_DIALOG_TITLE) if mode == "upload"
            else tr(S.EXPORT_DIALOG_TITLE)
        )
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

        # --- Destination (upload only) ---
        # Informed consent: the user must see exactly where the transcript is
        # going, and under what name, *before* confirming.
        self._edit_name: Optional[QLineEdit] = None
        if mode == "upload":
            name_row = QHBoxLayout()
            name_row.addWidget(QLabel(tr(S.UPLOAD_NAME_LABEL)))
            self._edit_name = QLineEdit(default_name)
            self._edit_name.setClearButtonEnabled(True)
            # Transcript names are long; show the whole one without scrolling.
            fitted = self._edit_name.fontMetrics().horizontalAdvance(default_name)
            self._edit_name.setMinimumWidth(min(620, max(320, fitted + 48)))
            self._edit_name.setCursorPosition(0)
            name_row.addWidget(self._edit_name, stretch=1)
            layout.addLayout(name_row)
        if mode == "upload" and target_description:
            target = QLabel(f"{tr(S.UPLOAD_TARGET_LABEL)} {target_description}")
            target.setWordWrap(True)
            target.setStyleSheet("color: #555; padding-top: 6px;")
            layout.addWidget(target)

        # --- Buttons ---
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Save).setText(
            tr(S.UPLOAD_OK) if mode == "upload" else tr(S.EXPORT_OK)
        )
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText(tr(S.EXPORT_CANCEL))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._load_prefs()

    def _load_prefs(self) -> None:
        """Populate the widgets from the last saved choices."""
        if self._mode == "upload":
            # docx round-trips right-to-left formatting into a Google Doc.
            fmt = self._settings.value(_UPLOAD_FORMAT_KEY, "docx", type=str)
            if fmt not in ("txt", "docx"):
                fmt = "docx"
        else:
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
        key = _UPLOAD_FORMAT_KEY if self._mode == "upload" else _EXPORT_FORMAT_KEY
        self._settings.setValue(key, self.selected_format())
        self._settings.setValue(_EXPORT_TIMESTAMPS_KEY, self.include_timestamps())
        self._settings.setValue(_EXPORT_EVENTS_KEY, self.include_events())
        super().accept()

    def selected_format(self) -> str:
        return self._cmb_format.currentData() or "txt"

    def selected_name(self) -> str:
        """The chosen upload name, sanitised, or ``""`` if left blank.

        A hand-typed name is as untrusted as the audio stem it defaults to, so
        it goes through the same choke point rather than reaching Drive raw.
        Blank means "keep the default" -- sanitising it would yield the generic
        fallback name instead.
        """
        from ..gdrive.naming import sanitize_display_name

        if self._edit_name is None or not self._edit_name.text().strip():
            return ""
        return sanitize_display_name(self._edit_name.text())

    def include_timestamps(self) -> bool:
        return self._chk_timestamps.isChecked()

    def include_events(self) -> bool:
        return self._chk_events.isChecked()


class TranscriptWidget(QWidget):
    """Scrollable transcript panel with timestamps and click-to-seek."""

    play_requested = Signal(float)
    upload_requested = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._segments: List[TranscribedSegment] = []
        self._detections: List[Detection] = []
        self._block_to_segment: dict[int, int] = {}  # block number -> segment index
        self._block_to_detection: dict[int, int] = {}  # block number -> detection index
        self._highlighted_block: int = -1
        self._visible_types: Optional[set] = None  # None = show all

        # Source context used to build the default export filename.
        self._audio_path: Optional[str] = None
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

        self._btn_upload = QPushButton("↑")
        self._btn_upload.setFixedWidth(28)
        self._btn_upload.setToolTip(tr(S.TRANSCRIPT_UPLOAD_DRIVE))
        self._btn_upload.clicked.connect(self.upload_requested.emit)
        search_row.addWidget(self._btn_upload)

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
        self._audio_path = audio_path
        self._audio_stem = Path(audio_path).stem.strip() if audio_path else None
        self._model_key = model_key

    def export_source(self) -> tuple[Optional[str], Optional[str]]:
        """Return ``(audio_path, model_key)`` for the shown transcript."""
        return self._audio_path, self._model_key

    def has_content(self) -> bool:
        """Return True if there is anything to export."""
        return bool(self._segments or self._detections)

    def default_export_name(self) -> str:
        """Return the canonical transcript base name (no extension)."""
        return self._default_export_name()

    def _default_export_name(self) -> str:
        """Build the canonical export base name.

        Delegates to :mod:`monitor.gdrive.naming` so the local ``.txt``, the
        local ``.docx`` and the Google Doc all get byte-identical names.
        """
        model_label = {
            "thorough": tr(S.EXPORT_NAME_THOROUGH),
            "fast": tr(S.EXPORT_NAME_FAST),
        }.get(self._model_key or "")
        return build_transcript_base_name(
            self._audio_path, self._model_key, model_label=model_label,
        )

    def build_export_bytes(
        self, fmt: str, include_timestamps: bool, include_events: bool,
    ) -> bytes:
        """Return the transcript encoded in *fmt* (``"txt"`` or ``"docx"``).

        Uses the same line builder as the local save, so an uploaded document
        is byte-identical to a downloaded one for the same options.
        """
        lines = self._build_transcript_lines(include_timestamps, include_events)
        if fmt == "docx":
            buffer = io.BytesIO()
            self._write_docx(buffer, lines)
            return buffer.getvalue()
        return "\n".join(lines).encode("utf-8")

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
    def _write_docx(path, lines: list[str]) -> None:
        """Write *lines* to a .docx with right-to-left, right-aligned text.

        Args:
            path: Destination path, or any binary file-like object (used by
                the upload path, which needs the bytes rather than a file).
            lines: Transcript lines.
        """
        from docx import Document
        from docx.oxml import OxmlElement
        from docx.oxml.ns import qn

        doc = Document()
        # Make the default 'Normal' style right-to-left so every paragraph
        # (including empty ones) renders as RTL.
        normal = doc.styles["Normal"].element
        _set_bidi(normal.get_or_add_pPr())
        _set_font(normal.get_or_add_rPr(), EXPORT_FONT)

        for line in lines:
            para = doc.add_paragraph()
            # No w:jc: it is *logical*, so "right" means the end edge, which is
            # the left one in an RTL paragraph. The w:bidi default already
            # aligns to the start edge -- verified against a Google Docs render.
            _set_bidi(para._p.get_or_add_pPr())
            run = para.add_run(line)
            # Run-level RTL so mixed neutrals (brackets, timestamps) order
            # correctly within the Hebrew text.
            r_pr = run._r.get_or_add_rPr()
            if r_pr.find(qn("w:rtl")) is None:
                r_pr.append(OxmlElement("w:rtl"))

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

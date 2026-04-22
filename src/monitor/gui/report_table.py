"""
Interactive report table widget with color-coded rows, play buttons, and
Excel-style column header filter dropdowns with persistent settings.

Displays analysis detections in a sortable QTableWidget where each row
is colour-coded by detection type and includes a play button that
triggers audio playback from that detection's timestamp.

Clicking the Type or Details column header opens an Excel-style filter
popup with sorted checkboxes, search box, and בחר הכל / נקה בחירה
controls.  Filter state is persisted to QSettings.

Usage:
    from monitor.gui.report_table import ReportTableWidget

    table = ReportTableWidget()
    table.load_report(analysis_report)
    table.play_requested.connect(audio_player.seek_to)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

from PySide6.QtCore import Qt, Signal, QPoint, QRect, QSize
from PySide6.QtGui import QColor, QFont, QMouseEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..models import (
    AnalysisReport,
    Detection,
    DetectionType,
    DETECTION_LABELS_HE,
)
from .player_icons import icon_play
from .strings import tr, S

log = logging.getLogger(__name__)

# ===========================
# CHECKBOX INDICATOR ICONS
# ===========================

import tempfile
import os

_INDICATOR_SIZE = 18
_indicator_dir: Optional[str] = None


def _ensure_checkbox_icons() -> str:
    """Create checked/unchecked indicator PNGs in a temp dir (once)."""
    global _indicator_dir
    if _indicator_dir is not None:
        return _indicator_dir
    _indicator_dir = tempfile.mkdtemp(prefix="monitor_icons_")

    # Unchecked: empty box.
    pm = QPixmap(_INDICATOR_SIZE, _INDICATOR_SIZE)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(QPen(QColor(100, 100, 100), 2))
    p.drawRoundedRect(2, 2, _INDICATOR_SIZE - 4, _INDICATOR_SIZE - 4, 3, 3)
    p.end()
    pm.save(os.path.join(_indicator_dir, "unchecked.png"))

    # Checked: dark box with white checkmark.
    pm2 = QPixmap(_INDICATOR_SIZE, _INDICATOR_SIZE)
    pm2.fill(Qt.GlobalColor.transparent)
    p2 = QPainter(pm2)
    p2.setRenderHint(QPainter.RenderHint.Antialiasing)
    p2.setBrush(QColor(50, 50, 50))
    p2.setPen(QPen(QColor(50, 50, 50), 2))
    p2.drawRoundedRect(2, 2, _INDICATOR_SIZE - 4, _INDICATOR_SIZE - 4, 3, 3)
    # Checkmark stroke.
    p2.setPen(QPen(QColor(255, 255, 255), 2.5, Qt.PenStyle.SolidLine,
                   Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    p2.drawLine(5, 9, 8, 13)
    p2.drawLine(8, 13, 14, 5)
    p2.end()
    pm2.save(os.path.join(_indicator_dir, "checked.png"))

    return _indicator_dir

# ===========================
# CONSTANTS
# ===========================

# Config file for filter persistence (program-level, not per-audio-file).
_CONFIG_DIR = Path.home() / ".child-monitor-analyzer"
_CONFIG_FILE = _CONFIG_DIR / "config.json"

# Row background colours by detection type.
ROW_COLORS: Dict[DetectionType, QColor] = {
    DetectionType.PROFANITY: QColor(255, 200, 200),      # red-ish
    DetectionType.SHOUT: QColor(255, 225, 180),           # orange
    DetectionType.SCREAM: QColor(255, 210, 170),          # darker orange
    DetectionType.CRY: QColor(200, 220, 255),             # blue-ish
    DetectionType.WAIL: QColor(200, 220, 255),            # blue-ish
    DetectionType.BABY_CRY: QColor(210, 200, 255),        # purple-ish
    DetectionType.LAUGHTER: QColor(220, 255, 220),        # green-ish
    DetectionType.VOLUME_SPIKE: QColor(230, 230, 230),    # grey
}

# Column headers — resolved at table build time via tr().
def _get_headers() -> list:
    return [tr(S.COL_TIME), tr(S.COL_TYPE), tr(S.COL_DETAILS),
            tr(S.COL_CONFIDENCE), tr(S.COL_PLAY)]

# Indices of columns that have filter dropdowns.
_FILTERABLE_COLUMNS = {1, 2}  # Type, Details

# Number of seconds of context before the detection start when "Play" is pressed.
PLAY_CONTEXT_SECONDS = 2.0

# Maximum time distance (seconds) for row highlighting during playback.
HIGHLIGHT_MAX_DISTANCE = 5.0


# ===========================
# FILTER POPUP WIDGET
# ===========================


class FilterPopup(QWidget):
    """Excel-style dropdown filter popup with search, select-all, and
    per-value checkboxes.

    Shown as a Qt.Popup anchored below a column header.  Emits
    ``selection_changed`` whenever checkboxes are toggled.
    """

    selection_changed = Signal(object)  # emits the set of checked value strings
    popup_closed = Signal()  # emitted when this popup is hidden/closed

    def __init__(
        self,
        values: List[str],
        checked: Optional[Set[str]],
        color_map: Optional[Dict[str, QColor]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent, Qt.WindowType.Popup)
        self._all_values = sorted(values)
        self._checked = set(self._all_values) if checked is None else set(checked)
        self._color_map = color_map or {}
        self._checkboxes: Dict[str, QCheckBox] = {}
        self._build_ui()

    def hideEvent(self, event) -> None:
        super().hideEvent(event)
        self.popup_closed.emit()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # --- Search box ---
        self._search = QLineEdit()
        self._search.setPlaceholderText(tr(S.FILTER_SEARCH))
        self._search.setClearButtonEnabled(True)
        self._search.textChanged.connect(self._on_search)
        layout.addWidget(self._search)

        # --- Select-all / Clear-all row ---
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(8)

        self._cb_select_all = QCheckBox(tr(S.FILTER_SELECT_ALL))
        self._cb_select_all.setStyleSheet("font-weight: bold;")
        self._cb_select_all.toggled.connect(self._on_select_all)
        ctrl_row.addWidget(self._cb_select_all)

        self._cb_clear_all = QCheckBox(tr(S.FILTER_CLEAR_ALL))
        self._cb_clear_all.setStyleSheet("font-weight: bold;")
        self._cb_clear_all.toggled.connect(self._on_clear_all)
        ctrl_row.addWidget(self._cb_clear_all)

        ctrl_row.addStretch()
        layout.addLayout(ctrl_row)

        # --- Scrollable checkbox list ---
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        scroll.setMaximumHeight(300)

        list_container = QWidget()
        self._list_layout = QVBoxLayout(list_container)
        self._list_layout.setContentsMargins(4, 2, 4, 2)
        self._list_layout.setSpacing(2)

        for value in self._all_values:
            cb = QCheckBox(value)
            cb.setChecked(value in self._checked)
            color = self._color_map.get(value)
            bg = color.name() if color else "#ffffff"
            icons = _ensure_checkbox_icons()
            # Use forward slashes — Qt stylesheets require them even on Windows.
            unc = os.path.join(icons, "unchecked.png").replace("\\", "/")
            chk = os.path.join(icons, "checked.png").replace("\\", "/")
            cb.setStyleSheet(
                f"QCheckBox {{ padding: 2px 4px; border-radius: 2px; "
                f"background-color: {bg}; color: black; }}"
                f"QCheckBox::indicator {{ width: {_INDICATOR_SIZE}px; height: {_INDICATOR_SIZE}px; }}"
                f"QCheckBox::indicator:unchecked {{ image: url({unc}); }}"
                f"QCheckBox::indicator:checked {{ image: url({chk}); }}"
            )
            cb.toggled.connect(self._on_item_toggled)
            self._checkboxes[value] = cb
            self._list_layout.addWidget(cb)

        self._list_layout.addStretch()
        scroll.setWidget(list_container)
        layout.addWidget(scroll, stretch=1)

        # Set a reasonable minimum size.
        self.setMinimumWidth(220)
        self._update_select_all_state()

    # --- Callbacks ---

    def _on_search(self, text: str) -> None:
        text_lower = text.lower()
        for value, cb in self._checkboxes.items():
            cb.setVisible(text_lower in value.lower())

    def _on_select_all(self, checked: bool) -> None:
        if not checked:
            return
        # Uncheck the "clear" checkbox silently.
        self._cb_clear_all.blockSignals(True)
        self._cb_clear_all.setChecked(False)
        self._cb_clear_all.blockSignals(False)
        # Check all visible items.
        for cb in self._checkboxes.values():
            if cb.isVisible():
                cb.blockSignals(True)
                cb.setChecked(True)
                cb.blockSignals(False)
        self._emit_selection()

    def _on_clear_all(self, checked: bool) -> None:
        if not checked:
            return
        # Uncheck the "select all" checkbox silently.
        self._cb_select_all.blockSignals(True)
        self._cb_select_all.setChecked(False)
        self._cb_select_all.blockSignals(False)
        # Uncheck all visible items.
        for cb in self._checkboxes.values():
            if cb.isVisible():
                cb.blockSignals(True)
                cb.setChecked(False)
                cb.blockSignals(False)
        self._emit_selection()

    def _on_item_toggled(self) -> None:
        self._update_select_all_state()
        self._emit_selection()

    def _update_select_all_state(self) -> None:
        visible = [cb for cb in self._checkboxes.values() if cb.isVisible()]
        if not visible:
            return
        all_checked = all(cb.isChecked() for cb in visible)
        none_checked = not any(cb.isChecked() for cb in visible)
        self._cb_select_all.blockSignals(True)
        self._cb_select_all.setChecked(all_checked)
        self._cb_select_all.blockSignals(False)
        self._cb_clear_all.blockSignals(True)
        self._cb_clear_all.setChecked(none_checked)
        self._cb_clear_all.blockSignals(False)

    def _emit_selection(self) -> None:
        self._checked = {v for v, cb in self._checkboxes.items() if cb.isChecked()}
        self.selection_changed.emit(self._checked)


# ===========================
# CORE WIDGET
# ===========================

# Header indicator — always filled.
_INDICATOR = "\u25bc"  # ▼
_INDICATOR_FILTERED_COLOR = QColor(30, 100, 200)  # blue when filter is active


class _FilterHeaderView(QHeaderView):
    """Custom horizontal header that draws a filter indicator on the far left
    of filterable columns and keeps the label text centred.

    Intercepts mouse clicks on filterable columns to open filter popups
    instead of triggering a sort.
    """

    filter_clicked = Signal(int)  # logical column index

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(Qt.Orientation.Horizontal, parent)
        # logical col -> True if filter is active, False if unfiltered,
        # absent if column has no filter at all.
        self._filter_active: Dict[int, bool] = {}

    def set_filter_state(self, logical_index: int, active: bool) -> None:
        self._filter_active[logical_index] = active
        self.viewport().update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        logical = self.logicalIndexAt(event.pos())
        if logical in self._filter_active:
            # Intercept: open filter popup instead of sorting.
            self.filter_clicked.emit(logical)
            return  # Do NOT call super() — prevents sort.
        super().mousePressEvent(event)

    def paintSection(
        self, painter: QPainter, rect: QRect, logical_index: int
    ) -> None:
        if logical_index not in self._filter_active:
            # Not a filterable column — standard drawing.
            super().paintSection(painter, rect, logical_index)
            return

        # For filterable columns: draw everything ourselves to avoid
        # double-text artifacts from calling super().
        painter.save()

        palette = self.palette()
        painter.fillRect(rect, palette.button())

        # Section border.
        border_color = palette.mid().color()
        painter.setPen(border_color)
        painter.drawLine(rect.right(), rect.top(), rect.right(), rect.bottom())
        painter.drawLine(rect.left(), rect.bottom(), rect.right(), rect.bottom())

        # Get label text.
        model = self.model()
        label = ""
        if model:
            data = model.headerData(
                logical_index, Qt.Orientation.Horizontal,
                Qt.ItemDataRole.DisplayRole,
            )
            if data:
                label = str(data)

        is_filtered = self._filter_active.get(logical_index, False)

        # Draw indicator ▼ on the far left of the cell.
        indicator_color = (
            _INDICATOR_FILTERED_COLOR if is_filtered
            else palette.buttonText().color()
        )
        painter.setPen(indicator_color)
        ind_rect = QRect(rect.left() + 4, rect.top(), 16, rect.height())
        painter.drawText(
            ind_rect,
            int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
            _INDICATOR,
        )

        # Draw label centred.
        painter.setPen(palette.buttonText().color())
        painter.drawText(
            rect,
            int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignHCenter),
            label,
        )

        painter.restore()


class ReportTableWidget(QWidget):
    """Composite widget: detection table with Excel-style column header
    filter dropdowns for Type and Details columns.

    Clicking the Type or Details header opens a popup with sorted
    checkboxes, search, and select-all / clear controls.

    Signals:
        play_requested: Emitted with a start-time (seconds) when a row's
            play button is clicked.
    """

    play_requested = Signal(float)
    filter_changed = Signal(object)  # emits Set[str] of visible type labels
    events_changed = Signal(object)  # emits List[float] of visible event start times

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self._detections: List[Detection] = []

        # Filter state: stored as *excluded* sets in config file.
        self._visible_types: Set[str] = self._load_filter_settings()
        # Details exclusion list (program-wide). Actual _visible_details
        # is computed per-file in load_report() from this.
        self._excluded_details: Set[str] = self._load_excluded_details()
        self._visible_details: Optional[Set[str]] = None

        self._active_popup: Optional[FilterPopup] = None
        self._active_popup_column: int = -1  # which column's popup is open
        self._highlighted_row: int = -1  # row bolded by highlight_time

        # Toggle cooldown: tracks when a popup was last auto-closed so
        # a header click within the cooldown window doesn't reopen it.
        self._popup_closed_column: int = -1
        self._popup_closed_time: float = 0.0

        self._play_icon = icon_play()

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # --- Table ---
        self._table = QTableWidget()
        headers = _get_headers()
        self._table.setColumnCount(len(headers))
        self._table.setHorizontalHeaderLabels(headers)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setCursor(Qt.CursorShape.PointingHandCursor)
        self._table.cellClicked.connect(self._on_cell_clicked)
        self._table.setSortingEnabled(True)
        self._table.verticalHeader().setVisible(False)
        self._table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        # Vivid selection colour so the auto-highlighted playback row is
        # obvious regardless of OS theme. Also applies to the scrollbar
        # marker so long tables remain readable.
        self._table.setStyleSheet(
            "QTableWidget::item:selected {"
            "  background-color: #ffd54a;"   # strong amber
            "  color: #000000;"
            "  font-weight: bold;"
            "}"
            "QTableWidget::item:selected:!active {"
            "  background-color: #ffd54a;"   # keep colour when focus is elsewhere
            "  color: #000000;"
            "}"
        )

        # Use custom header for filter indicators.
        self._header = _FilterHeaderView(self._table)
        self._table.setHorizontalHeader(self._header)
        # setSortingEnabled(True) was called before the header was replaced,
        # so the clickable flag was set on the old (now discarded) header.
        # Re-enable it so non-filterable columns (e.g. Time) trigger sorting.
        self._header.setSectionsClickable(True)

        self._header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self._header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self._header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self._header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self._header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self._header.setDefaultAlignment(
            Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter
        )

        # Connect header clicks for filter popups (custom signal, not sectionClicked).
        self._header.filter_clicked.connect(self._on_header_clicked)

        self._update_header_indicators()

        # Default sort: time ascending.
        self._table.sortByColumn(0, Qt.SortOrder.AscendingOrder)

        layout.addWidget(self._table, stretch=1)

    def _update_header_indicators(self) -> None:
        all_type_labels = {DETECTION_LABELS_HE.get(dt, dt.value) for dt in DetectionType}
        self._header.set_filter_state(1, self._visible_types != all_type_labels)
        self._header.set_filter_state(2, self._visible_details is not None)

    # ------------------------------------------------------------------
    # Filter persistence (JSON config file in program folder)
    # ------------------------------------------------------------------

    def _read_config(self) -> dict:
        """Read the program-level config file."""
        try:
            return json.loads(_CONFIG_FILE.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, ValueError):
            return {}

    def _write_config(self, cfg: dict) -> None:
        """Write the program-level config file."""
        _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        _CONFIG_FILE.write_text(
            json.dumps(cfg, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _load_filter_settings(self) -> Set[str]:
        """Load saved type filter. Returns set of visible Hebrew label strings."""
        all_he_labels = {DETECTION_LABELS_HE.get(dt, dt.value) for dt in DetectionType}
        cfg = self._read_config()
        excluded = set(cfg.get("excluded_types", []))
        return all_he_labels - (excluded & all_he_labels)

    def _save_filter_settings(self) -> None:
        all_he_labels = {DETECTION_LABELS_HE.get(dt, dt.value) for dt in DetectionType}
        excluded = sorted(all_he_labels - self._visible_types)
        cfg = self._read_config()
        cfg["excluded_types"] = excluded
        self._write_config(cfg)

    def _load_excluded_details(self) -> Set[str]:
        """Load the set of detail strings the user explicitly unchecked."""
        cfg = self._read_config()
        return set(cfg.get("excluded_details", []))

    def _save_excluded_details(self) -> None:
        """Persist the excluded-details set to config."""
        cfg = self._read_config()
        cfg["excluded_details"] = sorted(self._excluded_details)
        self._write_config(cfg)

    def _apply_details_exclusion(self) -> None:
        """Compute _visible_details from current detections and exclusion list.

        Default is *all checked*. Only terms in _excluded_details are unchecked.
        If nothing is excluded (or no exclusions match), _visible_details = None
        (meaning show all).
        """
        all_details = {
            _format_details(d)
            for d in self._detections
            if d.label_he in self._visible_types and _format_details(d)
        }
        actually_excluded = self._excluded_details & all_details
        if not actually_excluded:
            self._visible_details = None
        else:
            self._visible_details = all_details - actually_excluded

    # ------------------------------------------------------------------
    # Header click → filter popup
    # ------------------------------------------------------------------

    # Clicks within this window after a popup auto-close are treated as
    # toggle-close (don't reopen).
    _TOGGLE_COOLDOWN_SEC = 0.3

    def _on_header_clicked(self, logical_index: int) -> None:
        if logical_index not in _FILTERABLE_COLUMNS:
            return

        # Toggle cooldown: if the popup for this column was just auto-
        # closed by Qt (because the user clicked the header while the
        # popup was open), don't reopen it.
        if (logical_index == self._popup_closed_column
                and time.monotonic() - self._popup_closed_time < self._TOGGLE_COOLDOWN_SEC):
            self._popup_closed_column = -1
            return

        # If clicking the same column that has an open popup, just close it.
        if self._active_popup is not None and self._active_popup_column == logical_index:
            try:
                self._active_popup.close()
            except RuntimeError:
                pass
            self._active_popup = None
            self._active_popup_column = -1
            return

        # Close any existing popup for a different column.
        if self._active_popup is not None:
            try:
                self._active_popup.close()
            except RuntimeError:
                pass
            self._active_popup = None
            self._active_popup_column = -1

        if logical_index == 1:
            self._show_type_filter(self._header, logical_index)
        elif logical_index == 2:
            self._show_details_filter(self._header, logical_index)

    def _on_popup_closed(self) -> None:
        # Record which column was closed and when, so the toggle
        # cooldown in _on_header_clicked can detect re-clicks.
        self._popup_closed_column = self._active_popup_column
        self._popup_closed_time = time.monotonic()
        if self._active_popup is not None:
            self._active_popup.deleteLater()
        self._active_popup = None
        self._active_popup_column = -1

    def _popup_position(self, header: QHeaderView, logical_index: int) -> QPoint:
        x = header.sectionViewportPosition(logical_index)
        y = header.height()
        return header.mapToGlobal(QPoint(x, y))

    def _show_type_filter(self, header: QHeaderView, col: int) -> None:
        # Build list of all type labels with colour mapping.
        all_labels: List[str] = []
        color_map: Dict[str, QColor] = {}
        for dt in DetectionType:
            label = DETECTION_LABELS_HE.get(dt, dt.value)
            all_labels.append(label)
            color = ROW_COLORS.get(dt)
            if color:
                color_map[label] = color

        popup = FilterPopup(
            values=all_labels,
            checked=self._visible_types,
            color_map=color_map,
            parent=self,
        )
        popup.selection_changed.connect(self._on_type_selection_changed)
        popup.popup_closed.connect(self._on_popup_closed)
        popup.move(self._popup_position(header, col))
        popup.show()
        self._active_popup = popup
        self._active_popup_column = col

    def _show_details_filter(self, header: QHeaderView, col: int) -> None:
        # Collect detail values only from rows that pass the TYPE filter.
        visible_details: List[str] = sorted({
            _format_details(d)
            for d in self._detections
            if d.label_he in self._visible_types and _format_details(d)
        })
        if not visible_details:
            return

        popup = FilterPopup(
            values=visible_details,
            checked=self._visible_details,
            parent=self,
        )
        popup.selection_changed.connect(self._on_details_selection_changed)
        popup.popup_closed.connect(self._on_popup_closed)
        popup.move(self._popup_position(header, col))
        popup.show()
        self._active_popup = popup
        self._active_popup_column = col

    def _on_type_selection_changed(self, selected: set) -> None:
        self._visible_types = selected
        self._save_filter_settings()
        # Recompute details visibility from the exclusion list.
        self._apply_details_exclusion()
        self._update_header_indicators()
        self._refresh_table()
        self.filter_changed.emit(self._visible_types)

    def _on_details_selection_changed(self, selected: set) -> None:
        # Compare against all available details (type-filtered).
        all_details = {
            _format_details(d)
            for d in self._detections
            if d.label_he in self._visible_types and _format_details(d)
        }
        # Update the global exclusion list: newly unchecked items are added,
        # re-checked items are removed.
        newly_excluded = all_details - selected
        newly_included = selected & self._excluded_details
        self._excluded_details = (self._excluded_details - newly_included) | newly_excluded
        self._save_excluded_details()

        if selected == all_details:
            self._visible_details = None
        else:
            self._visible_details = selected
        self._update_header_indicators()
        self._refresh_table()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_report(self, report: AnalysisReport) -> None:
        self._detections = report.detections_sorted()
        # Recompute visible details from the program-wide exclusion list.
        self._apply_details_exclusion()
        self._refresh_table()

    def highlight_time(self, current_seconds: float) -> None:
        best_row = -1
        best_distance = float("inf")

        for row_index in range(self._table.rowCount()):
            if self._table.isRowHidden(row_index):
                continue
            item = self._table.item(row_index, 0)
            if item is None:
                continue
            start_time = item.data(Qt.ItemDataRole.UserRole)
            if start_time is None:
                continue
            distance = abs(start_time - current_seconds)
            if distance < best_distance:
                best_distance = distance
                best_row = row_index

        if best_row >= 0 and best_distance < HIGHLIGHT_MAX_DISTANCE:
            # Clear bold from the previously highlighted row.
            if self._highlighted_row >= 0 and self._highlighted_row != best_row:
                self._set_row_bold(self._highlighted_row, False)
            # Bold the new row and remember it.
            self._set_row_bold(best_row, True)
            self._highlighted_row = best_row

            self._table.selectRow(best_row)
            # Scroll the selected row into view so the user can see the
            # currently-playing event without manual scrolling.
            item = self._table.item(best_row, 0)
            if item is not None:
                self._table.scrollToItem(
                    item, QAbstractItemView.ScrollHint.PositionAtCenter,
                )

    def _set_row_bold(self, row: int, bold: bool) -> None:
        """Set or clear bold font on every cell in *row*."""
        weight = QFont.Weight.Bold if bold else QFont.Weight.Normal
        for col in range(self._table.columnCount()):
            item = self._table.item(row, col)
            if item is not None:
                font = item.font()
                font.setWeight(weight)
                item.setFont(font)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _refresh_table(self) -> None:
        self._highlighted_row = -1
        self._table.setSortingEnabled(False)
        self._table.setRowCount(0)

        filtered = [
            d for d in self._detections
            if d.label_he in self._visible_types
            and (self._visible_details is None or _format_details(d) in self._visible_details)
        ]
        self._table.setRowCount(len(filtered))

        for row_index, detection in enumerate(filtered):
            self._populate_row(row_index, detection)

        self._table.setSortingEnabled(True)
        self._update_header_indicators()
        log.info(
            "Report table refreshed: %d/%d detections shown.",
            len(filtered), len(self._detections),
        )
        self.events_changed.emit([d.start for d in filtered])

    def _populate_row(self, row_index: int, detection: Detection) -> None:
        background_color = ROW_COLORS.get(detection.type, QColor(255, 255, 255))

        # Column 0: Time (HH:MM:SS).
        time_item = QTableWidgetItem(detection.time_display)
        time_item.setData(Qt.ItemDataRole.UserRole, detection.start)
        time_item.setBackground(background_color)
        self._table.setItem(row_index, 0, time_item)

        # Column 1: Type (Hebrew label).
        type_item = QTableWidgetItem(detection.label_he)
        type_item.setBackground(background_color)
        self._table.setItem(row_index, 1, type_item)

        # Column 2: Details.
        details_text = _format_details(detection)
        details_item = QTableWidgetItem(details_text)
        details_item.setBackground(background_color)
        self._table.setItem(row_index, 2, details_item)

        # Column 3: Confidence percentage.
        confidence_item = QTableWidgetItem(f"{detection.confidence:.0%}")
        confidence_item.setData(Qt.ItemDataRole.UserRole, detection.confidence)
        confidence_item.setBackground(background_color)
        self._table.setItem(row_index, 3, confidence_item)

        # Column 4: Play button (icon matches the media player play button).
        play_button = QPushButton()
        play_button.setIcon(self._play_icon)
        play_button.setIconSize(QSize(18, 18))
        play_button.setFixedWidth(32)
        start_time = detection.start
        play_button.clicked.connect(
            lambda checked=False, time=start_time: self._on_play(time)
        )
        self._table.setCellWidget(row_index, 4, play_button)

    def _on_cell_clicked(self, row: int, col: int) -> None:
        """Any cell click triggers playback for that row."""
        item = self._table.item(row, 0)
        if item is None:
            return
        start_time = item.data(Qt.ItemDataRole.UserRole)
        if start_time is None:
            return
        self._on_play(start_time)

    def _on_play(self, start_time: float) -> None:
        seek_position = max(0.0, start_time - PLAY_CONTEXT_SECONDS)
        self.play_requested.emit(seek_position)


# ===========================
# MODULE-LEVEL HELPERS
# ===========================


def _format_details(detection: Detection) -> str:
    if detection.profanity_words:
        return ", ".join(detection.profanity_words)
    if "sentence" in detection.details:
        return detection.details["sentence"]
    if "audioset_class" in detection.details:
        return detection.details["audioset_class"]
    if "rms" in detection.details:
        return f"RMS={detection.details['rms']}"
    return ""

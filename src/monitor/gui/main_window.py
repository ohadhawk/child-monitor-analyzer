"""
Main application window -- Hebrew RTL GUI for Child Monitor Analyzer.

Provides the top-level PySide6 window with:
  - File picker and analyse button (top bar)
  - Drag-and-drop audio file support
  - Recent files history (up to 5 entries, persisted via QSettings)
  - Progress bar during analysis
  - Interactive detection report table (middle)
  - Integrated audio player (bottom)

Analysis runs in a background subprocess so the GUI stays responsive
and analysis can be cancelled instantly when switching files.

Usage:
    from monitor.gui.main_window import run_gui
    run_gui()
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import queue as _queue_mod  # for queue.Empty
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import (
    QSettings, Qt, QTimer, QUrl, Slot,
    QtMsgType, qInstallMessageHandler,
)
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..analysis_worker import (
    run_analysis,
    MSG_PROGRESS, MSG_SUB_PROGRESS, MSG_SUB_PROGRESS2,
    MSG_TASK_PROGRESS, MSG_FINISHED, MSG_ERROR,
)
from ..model_cache import setup_model_environment
from ..models import AnalysisReport
from .audio_player import AudioPlayerWidget
from .report_table import ReportTableWidget
from .sensitivity_panel import SensitivityDialog
from .transcript_widget import TranscriptWidget
from .strings import tr, S

# Ensure model env is set before any ML imports happen.
setup_model_environment()

log = logging.getLogger(__name__)

# ===========================
# CONSTANTS
# ===========================

# Supported audio file extensions for the file picker filter.
AUDIO_FILE_FILTER = (
    "Audio files (*.wav *.mp3 *.m4a *.flac *.ogg *.wma *.aac);;All files (*)"
)

# Set of extensions accepted by drag-and-drop (lower-case, with leading dot).
SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".wma", ".aac"}

WINDOW_TITLE = "Child Monitor Analyzer"
WINDOW_MIN_WIDTH = 800
WINDOW_MIN_HEIGHT = 600
WINDOW_DEFAULT_WIDTH = 1000
WINDOW_DEFAULT_HEIGHT = 700

# Maximum number of entries in the recent-files list.
MAX_RECENT_FILES = 5

# Default log directory (inside the project/user directory).
LOG_DIR = Path.home() / ".child-monitor-analyzer" / "logs"

# QSettings keys.
SETTINGS_ORG = "ChildMonitorAnalyzer"
SETTINGS_APP = "monitor-gui"
SETTINGS_RECENT_KEY = "recent_files"

# ===========================
# MAIN WINDOW
# ===========================


class MainWindow(QMainWindow):
    """Main application window with Hebrew RTL layout.

    Supports drag-and-drop of audio files and maintains a recent-files
    history (up to MAX_RECENT_FILES entries) persisted across sessions.

    Attributes:
        _worker_process: Subprocess running the analysis (if active).
        _worker_queue: Queue for receiving progress/results from subprocess.
        _current_audio: Path to the currently loaded audio file.
        _recent_files: Ordered list of recently opened file paths (newest first).
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(tr(S.WINDOW_TITLE))
        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)
        self.resize(WINDOW_DEFAULT_WIDTH, WINDOW_DEFAULT_HEIGHT)

        self._worker_process: Optional[multiprocessing.Process] = None
        self._worker_queue: Optional[multiprocessing.Queue] = None
        self._current_audio: Optional[str] = None
        self._current_report: Optional[AnalysisReport] = None

        # Timer to poll the subprocess queue for progress/results.
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(50)  # 50 ms
        self._poll_timer.timeout.connect(self._poll_worker_queue)

        # Enable drag-and-drop on the main window.
        self.setAcceptDrops(True)

        # Load recent files from persistent settings.
        self._settings = QSettings(SETTINGS_ORG, SETTINGS_APP)
        self._recent_files: List[str] = self._load_recent_files()

        self._build_ui()
        self._update_recent_menu()

        # Auto-load the most recent file after the event loop starts.
        if self._recent_files:
            QTimer.singleShot(0, self._auto_load_last_file)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """Construct the main window layout and all child widgets."""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # --- Top bar: file picker + recent + analyse button ---
        layout.addLayout(self._build_top_bar())

        # --- Drop hint label (shown when no file loaded) ---
        self._lbl_drop_hint = QLabel(tr(S.DROP_HINT))
        self._lbl_drop_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_drop_hint.setStyleSheet(
            "QLabel { color: #888; font-size: 14px; padding: 40px; "
            "border: 2px dashed #ccc; border-radius: 8px; margin: 8px; }"
        )
        layout.addWidget(self._lbl_drop_hint)

        # --- Status label (shows current operation) ---
        self._lbl_status = QLabel()
        self._lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_status.setStyleSheet(
            "QLabel { color: #2196F3; font-size: 13px; font-weight: bold; padding: 4px; }"
        )
        self._lbl_status.setVisible(False)
        layout.addWidget(self._lbl_status)

        # --- Progress bar ---
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(False)
        self._progress_bar.setTextVisible(True)
        layout.addWidget(self._progress_bar)

        # --- Sub-progress bar (download / sub-operation progress) ---
        self._sub_progress_bar = QProgressBar()
        self._sub_progress_bar.setRange(0, 100)
        self._sub_progress_bar.setValue(0)
        self._sub_progress_bar.setVisible(False)
        self._sub_progress_bar.setTextVisible(True)
        self._sub_progress_bar.setFixedHeight(18)
        self._sub_progress_bar.setStyleSheet(
            "QProgressBar { font-size: 11px; }"
        )
        layout.addWidget(self._sub_progress_bar)

        # --- Second sub-progress bar (shown when two downloads run concurrently) ---
        self._sub_progress_bar2 = QProgressBar()
        self._sub_progress_bar2.setRange(0, 100)
        self._sub_progress_bar2.setValue(0)
        self._sub_progress_bar2.setVisible(False)
        self._sub_progress_bar2.setTextVisible(True)
        self._sub_progress_bar2.setFixedHeight(18)
        self._sub_progress_bar2.setStyleSheet(
            "QProgressBar { font-size: 11px; }"
        )
        layout.addWidget(self._sub_progress_bar2)

        # --- Task progress bars (parallel STT + audio events) ---
        self._task_bars: List[QProgressBar] = []
        task_labels = [tr(S.TASK_STT), tr(S.TASK_AUDIO_EVENTS)]
        for label in task_labels:
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setVisible(False)
            bar.setTextVisible(True)
            bar.setFormat(f"{label}: 0%")
            self._task_bars.append(bar)
            layout.addWidget(bar)

        # --- Sensitivity dialog (created once, shown on button click) ---
        self._sensitivity_dialog = SensitivityDialog(self)
        self._sensitivity_dialog.thresholds_changed.connect(self._on_sensitivity_changed)

        # --- Report table + Transcript in a splitter ---
        self._splitter = QSplitter(Qt.Orientation.Horizontal)

        self._report_table = ReportTableWidget()
        self._splitter.addWidget(self._report_table)

        self._transcript = TranscriptWidget()
        self._splitter.addWidget(self._transcript)

        # Default proportions: report table gets 65%, transcript 35%.
        self._splitter.setStretchFactor(0, 65)
        self._splitter.setStretchFactor(1, 35)

        layout.addWidget(self._splitter, stretch=1)

        # --- Audio player ---
        self._audio_player = AudioPlayerWidget()
        layout.addWidget(self._audio_player)

        # --- Wire cross-widget signals ---
        self._report_table.play_requested.connect(self._audio_player.seek_to)
        self._report_table.play_requested.connect(self._transcript.highlight_time)
        self._report_table.filter_changed.connect(self._transcript.set_visible_types)
        self._report_table.events_changed.connect(self._audio_player.set_event_times)
        self._transcript.play_requested.connect(self._audio_player.seek_to)
        self._audio_player.position_changed.connect(self._report_table.highlight_time)
        self._audio_player.position_changed.connect(self._transcript.highlight_time)

    def _build_top_bar(self) -> QHBoxLayout:
        """Create the top bar with file picker, recent button, and analyse button.

        Returns:
            QHBoxLayout containing the top bar widgets.
        """
        self._btn_open = QPushButton(tr(S.OPEN_FILE))
        self._btn_open.setFixedHeight(36)
        self._btn_open.clicked.connect(self._open_file)

        # Recent-files dropdown button.
        self._btn_recent = QToolButton()
        self._btn_recent.setText(tr(S.RECENT))
        self._btn_recent.setFixedHeight(36)
        self._btn_recent.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._recent_menu = QMenu(self)
        self._btn_recent.setMenu(self._recent_menu)

        # Sensitivity button.
        self._btn_sensitivity = QPushButton(tr(S.SENSITIVITY_TITLE))
        self._btn_sensitivity.setFixedHeight(36)
        self._btn_sensitivity.clicked.connect(self._open_sensitivity_dialog)

        self._lbl_file = QLabel(tr(S.NO_FILE_SELECTED))
        self._lbl_file.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )

        self._btn_analyze = QPushButton(tr(S.ANALYSE))
        self._btn_analyze.setFixedHeight(36)
        self._btn_analyze.setEnabled(False)
        self._btn_analyze.clicked.connect(lambda: self._start_analysis(force_restart=True))

        top_bar = QHBoxLayout()
        top_bar.addWidget(self._btn_analyze)
        top_bar.addWidget(self._lbl_file, stretch=1)
        top_bar.addWidget(self._btn_sensitivity)
        top_bar.addWidget(self._btn_recent)
        top_bar.addWidget(self._btn_open)
        return top_bar

    # ------------------------------------------------------------------
    # Drag-and-drop
    # ------------------------------------------------------------------

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        """Accept drag events that contain file URLs with supported audio extensions.

        Args:
            event: The drag-enter event.
        """
        if event.mimeData().hasUrls():
            # Accept if at least one URL has a supported audio extension.
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    suffix = Path(url.toLocalFile()).suffix.lower()
                    if suffix in SUPPORTED_EXTENSIONS:
                        event.acceptProposedAction()
                        return
        event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        """Handle dropped audio files -- load the first valid one.

        Args:
            event: The drop event.
        """
        for url in event.mimeData().urls():
            if not url.isLocalFile():
                continue
            file_path = url.toLocalFile()
            suffix = Path(file_path).suffix.lower()
            if suffix in SUPPORTED_EXTENSIONS:
                self._load_audio_file(file_path)
                event.acceptProposedAction()
                return
        event.ignore()

    # ------------------------------------------------------------------
    # Window close
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        """Clean up the analysis subprocess before closing."""
        self._stop_previous_analysis()
        event.accept()

    # ------------------------------------------------------------------
    # Recent files
    # ------------------------------------------------------------------

    def _load_recent_files(self) -> List[str]:
        """Load the recent files list from QSettings.

        Returns:
            List of file path strings (newest first), max MAX_RECENT_FILES.
        """
        raw = self._settings.value(SETTINGS_RECENT_KEY, "[]")
        try:
            # QSettings stores as string; parse as JSON list.
            paths = json.loads(raw) if isinstance(raw, str) else list(raw)
        except (json.JSONDecodeError, TypeError):
            paths = []
        # Filter out files that no longer exist.
        return [p for p in paths if Path(p).exists()][:MAX_RECENT_FILES]

    def _save_recent_files(self) -> None:
        """Persist the recent files list to QSettings."""
        self._settings.setValue(SETTINGS_RECENT_KEY, json.dumps(self._recent_files))

    def _add_to_recent(self, file_path: str) -> None:
        """Add a file path to the top of the recent files list.

        Removes duplicates and trims to MAX_RECENT_FILES entries.

        Args:
            file_path: Absolute path to the audio file.
        """
        # Normalise the path for consistent duplicate detection.
        normalised = str(Path(file_path).resolve())
        # Remove if already present (will be re-added at front).
        self._recent_files = [
            p for p in self._recent_files if str(Path(p).resolve()) != normalised
        ]
        self._recent_files.insert(0, file_path)
        self._recent_files = self._recent_files[:MAX_RECENT_FILES]
        self._save_recent_files()
        self._update_recent_menu()

    def _update_recent_menu(self) -> None:
        """Rebuild the recent-files dropdown menu from the current list."""
        self._recent_menu.clear()
        if not self._recent_files:
            action = self._recent_menu.addAction(tr(S.NO_RECENT_FILES))
            action.setEnabled(False)
            return
        for file_path in self._recent_files:
            display_name = Path(file_path).name
            # lambda with default arg captures file_path by value.
            action = self._recent_menu.addAction(display_name)
            action.triggered.connect(
                lambda checked=False, path=file_path: self._load_audio_file(path)
            )

    # ------------------------------------------------------------------
    # Auto-load last file on startup
    # ------------------------------------------------------------------

    def _auto_load_last_file(self) -> None:
        """Automatically reload the most recent file and resume analysis if needed."""
        if not self._recent_files:
            return
        last_file = self._recent_files[0]
        if not Path(last_file).exists():
            return
        log.info("Auto-loading last file: %s", last_file)
        self._load_audio_file(last_file)

    # ------------------------------------------------------------------
    # File loading (shared by open, drag-drop, and recent)
    # ------------------------------------------------------------------

    def _load_audio_file(self, file_path: str) -> None:
        """Load an audio file from any source (picker, drag-drop, recent).

        If a cached analysis exists alongside the audio file, it is loaded
        directly into the report table without re-running the pipeline.
        Otherwise analysis starts automatically.

        Args:
            file_path: Absolute path to the audio file.
        """
        if not Path(file_path).exists():
            QMessageBox.warning(self, tr(S.FILE_NOT_FOUND),
                                tr(S.FILE_NOT_FOUND_MSG).format(path=file_path))
            return
        self._stop_previous_analysis()
        self._current_audio = file_path
        self._lbl_file.setText(str(Path(file_path)))
        self._btn_analyze.setEnabled(True)
        self._audio_player.load(file_path)
        self._add_to_recent(file_path)
        # Hide the drop-hint once a file is loaded.
        self._lbl_drop_hint.setVisible(False)
        # Clear all progress bars from a previous incomplete analysis.
        self._progress_bar.setVisible(False)
        self._progress_bar.setValue(0)
        self._sub_progress_bar.setVisible(False)
        self._sub_progress_bar.setValue(0)
        self._sub_progress_bar2.setVisible(False)
        self._sub_progress_bar2.setValue(0)
        for bar in self._task_bars:
            bar.setVisible(False)
            bar.setValue(0)
        self._lbl_status.setVisible(False)
        log.info("Audio file loaded: %s", file_path)

        # Try to load a cached analysis report.
        cached = AnalysisReport.load_cache(file_path)
        if cached is not None:
            self._current_report = cached
            self._apply_sensitivity_filter()
            self._transcript.load_segments(cached.segments, cached.detections)
            self._lbl_status.setText(tr(S.LOADED_CACHED))
            self._lbl_status.setVisible(True)
            log.info("Loaded cached analysis for %s", file_path)
        else:
            # No cache -- clear old data and start analysis immediately.
            self._current_report = None
            self._report_table.load_report(AnalysisReport(audio_path=file_path))
            self._transcript.load_segments([])
            log.info("No cache for %s; auto-starting analysis.", file_path)
            self._start_analysis()

    # ------------------------------------------------------------------
    # File picker
    # ------------------------------------------------------------------

    @Slot()
    def _open_file(self) -> None:
        """Open a file dialog and load the selected audio file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            tr(S.SELECT_AUDIO_FILE),
            "",
            AUDIO_FILE_FILTER,
        )
        if file_path:
            self._load_audio_file(file_path)

    # ------------------------------------------------------------------
    # Sensitivity dialog
    # ------------------------------------------------------------------

    @Slot()
    def _open_sensitivity_dialog(self) -> None:
        """Show the sensitivity adjustment dialog."""
        self._sensitivity_dialog.show()
        self._sensitivity_dialog.raise_()
        self._sensitivity_dialog.activateWindow()

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def _stop_previous_analysis(self) -> None:
        """Terminate any running analysis subprocess."""
        self._poll_timer.stop()
        if self._worker_process is not None and self._worker_process.is_alive():
            pid = self._worker_process.pid
            log.info("Terminating analysis subprocess (PID %s)...", pid)
            try:
                self._worker_process.terminate()
                self._worker_process.join(timeout=3)
                # On Windows, terminate() calls TerminateProcess which is
                # always immediate.  kill() is only meaningful on POSIX.
                if self._worker_process.is_alive():
                    log.warning("Subprocess still alive after terminate; killing (PID %s).", pid)
                    self._worker_process.kill()
                    self._worker_process.join(timeout=2)
            except Exception:
                log.exception("Error terminating analysis subprocess (PID %s).", pid)
            log.info("Analysis subprocess terminated (PID %s).", pid)

        # Clean up the queue.  cancel_join_thread() prevents a
        # deadlock: join_thread() would block if the feeder thread is
        # stuck flushing to a broken pipe after terminate().
        if self._worker_queue is not None:
            try:
                self._worker_queue.close()
                self._worker_queue.cancel_join_thread()
            except Exception:
                pass  # Queue may already be broken after terminate

        self._worker_process = None
        self._worker_queue = None

    @Slot()
    def _start_analysis(self, *, force_restart: bool = False) -> None:
        """Launch the analysis pipeline in a subprocess.

        Args:
            force_restart: If True, clears intermediate caches to force
                a full re-analysis from scratch.
        """
        if not self._current_audio:
            return

        self._stop_previous_analysis()

        # When the user explicitly clicks "Analyse" (force_restart=True),
        # clear intermediate caches so the analysis starts fresh.
        if force_restart and self._current_audio:
            from ..pipeline import _remove_intermediate_caches
            _remove_intermediate_caches(Path(self._current_audio))
            log.info("Cleared intermediate caches for fresh analysis.")

        self._btn_analyze.setEnabled(False)
        self._btn_open.setEnabled(False)
        self._progress_bar.setVisible(True)
        self._progress_bar.setValue(0)

        try:
            self._worker_queue = multiprocessing.Queue()
            self._worker_process = multiprocessing.Process(
                target=run_analysis,
                args=(self._current_audio, self._worker_queue),
                daemon=True,
            )
            self._worker_process.start()
            self._poll_timer.start()
            log.info(
                "Analysis subprocess started (PID %s) for %s",
                self._worker_process.pid, self._current_audio,
            )
        except Exception as exc:
            log.exception("Failed to start analysis subprocess.")
            # Clean up partial state so buttons are re-enabled.
            self._worker_process = None
            self._worker_queue = None
            self._btn_analyze.setEnabled(True)
            self._btn_open.setEnabled(True)
            self._progress_bar.setVisible(False)
            self._on_error(f"Failed to start analysis: {exc}")

    # ------------------------------------------------------------------
    # Subprocess queue polling
    # ------------------------------------------------------------------

    @Slot()
    def _poll_worker_queue(self) -> None:
        """Drain pending messages from the subprocess queue.

        Called every 50 ms by _poll_timer.  Dispatches each message to
        the appropriate handler and detects subprocess crashes.
        """
        if self._worker_queue is None or self._worker_process is None:
            self._poll_timer.stop()
            return

        # Drain all available messages.
        try:
            while True:
                try:
                    msg = self._worker_queue.get_nowait()
                except _queue_mod.Empty:
                    break
                self._handle_worker_message(msg)
                # Stop polling after terminal messages.
                if msg.get("type") in (MSG_FINISHED, MSG_ERROR):
                    return
        except Exception:
            log.exception("Error reading from worker queue.")

        # Detect subprocess crash (died without sending finished/error).
        if self._worker_process is not None and not self._worker_process.is_alive():
            exitcode = self._worker_process.exitcode
            self._poll_timer.stop()
            log.error(
                "Analysis subprocess died unexpectedly (PID %s, exit code %s).",
                self._worker_process.pid, exitcode,
            )
            self._on_error(
                f"Analysis process crashed unexpectedly (exit code {exitcode})."
            )

    def _handle_worker_message(self, msg: dict) -> None:
        """Dispatch a single message from the subprocess queue."""
        msg_type = msg.get("type")
        try:
            if msg_type == MSG_PROGRESS:
                self._on_progress(msg["pct"], msg["msg"])
            elif msg_type == MSG_SUB_PROGRESS:
                self._on_sub_progress(msg["done"], msg["total"], msg["label"])
            elif msg_type == MSG_SUB_PROGRESS2:
                self._on_sub_progress2(msg["done"], msg["total"], msg["label"])
            elif msg_type == MSG_TASK_PROGRESS:
                self._on_task_progress(msg["task_id"], msg["pct"], msg["label"])
            elif msg_type == MSG_FINISHED:
                report = AnalysisReport.from_dict(msg["report"])
                self._poll_timer.stop()
                self._on_finished(report)
            elif msg_type == MSG_ERROR:
                tb = msg.get("traceback", "")
                if tb:
                    log.error("Subprocess traceback:\n%s", tb)
                self._poll_timer.stop()
                self._on_error(msg["msg"])
            else:
                log.warning("Unknown worker message type: %r", msg_type)
        except Exception:
            log.exception("Error handling worker message: %r", msg)

    @Slot(int, str)
    def _on_progress(self, pct: int, msg: str) -> None:
        """Update the progress bar and status label during analysis.

        Args:
            pct: Completion percentage (0-100).
            msg: Hebrew status message.
        """
        self._progress_bar.setValue(pct)
        self._progress_bar.setFormat(f"{pct}%  |  {msg}")
        self._lbl_status.setText(f"[{pct}%] {msg}")
        self._lbl_status.setVisible(True)

    @Slot(object, object, str)
    def _on_sub_progress(self, done: int, total: int, label: str) -> None:
        """Update the sub-progress bar for download / sub-operation progress.

        Args:
            done: Bytes downloaded (or -1 to hide).
            total: Total bytes (or 0 for indeterminate).
            label: Description of current sub-operation.
        """
        self._update_sub_bar(self._sub_progress_bar, done, total, label)

    @Slot(object, object, str)
    def _on_sub_progress2(self, done: int, total: int, label: str) -> None:
        """Update the second sub-progress bar (used for parallel downloads)."""
        self._update_sub_bar(self._sub_progress_bar2, done, total, label)

    @staticmethod
    def _update_sub_bar(bar: QProgressBar, done: int, total: int, label: str) -> None:
        """Shared helper to render a byte-level download progress bar."""
        if done == -1:
            bar.setVisible(False)
            return

        bar.setVisible(True)

        if total <= 0:
            # Indeterminate mode (pulsing bar).
            bar.setRange(0, 0)
            bar.setFormat(label)
        else:
            # Diagnostic guard: clamp absurd values so the UI never shows
            # multi-thousand percentages even if a producer misbehaves. The
            # warning in pipeline._sub/_sub2 will already have logged the bug.
            if done > total:
                log.warning(
                    "sub-bar clamped: done=%d > total=%d label=%r",
                    done, total, label,
                )
                done = total
            # Determinate mode with MB display.
            bar.setRange(0, total)
            bar.setValue(done)
            done_mb = done / (1024 * 1024)
            total_mb = total / (1024 * 1024)
            pct = done * 100 // total
            bar.setFormat(
                f"{label}  {done_mb:.0f} / {total_mb:.0f} MB ({pct}%)"
            )

    @Slot(int, int, str)
    def _on_task_progress(self, task_id: int, pct: int, label: str) -> None:
        """Update a parallel-task progress bar.

        Args:
            task_id: 0 for STT, 1 for audio events.
            pct: Percentage (0-100), 0 for indeterminate, -1 to hide.
            label: Description text.
        """
        if task_id < 0 or task_id >= len(self._task_bars):
            return
        bar = self._task_bars[task_id]
        task_names = [tr(S.TASK_STT), tr(S.TASK_AUDIO_EVENTS)]
        name = task_names[task_id] if task_id < len(task_names) else f"Task {task_id}"

        if pct == -1:
            bar.setVisible(False)
            return

        bar.setVisible(True)
        if pct == 0:
            # Indeterminate (pulsing).
            bar.setRange(0, 0)
            bar.setFormat(f"{name}: {label}")
        else:
            bar.setRange(0, 100)
            bar.setValue(pct)
            bar.setFormat(f"{name}: {pct}% {label}")

    @Slot(object)
    def _on_finished(self, report: AnalysisReport) -> None:
        """Handle successful analysis completion.

        Args:
            report: The completed AnalysisReport.
        """
        # Guard against stale messages processed after subprocess was stopped.
        if self._worker_process is None:
            log.info("Ignoring stale finished message (process already stopped).")
            return
        self._progress_bar.setVisible(False)
        self._sub_progress_bar.setVisible(False)
        self._sub_progress_bar2.setVisible(False)
        for bar in self._task_bars:
            bar.setVisible(False)
        self._lbl_status.setVisible(False)
        self._btn_analyze.setEnabled(True)
        self._btn_open.setEnabled(True)
        # Clean up process/queue references now that analysis is done.
        self._worker_process = None
        self._worker_queue = None
        self._current_report = report
        self._apply_sensitivity_filter()
        self._transcript.load_segments(report.segments, report.detections)
        log.info(
            "Analysis complete: %d detections found.", len(report.detections)
        )

    @Slot(str)
    def _on_error(self, error_msg: str) -> None:
        """Handle analysis failure.

        Args:
            error_msg: Description of the error.
        """
        if self._worker_process is None:
            log.info("Ignoring stale error message (process already stopped).")
            return
        self._progress_bar.setVisible(False)
        self._sub_progress_bar.setVisible(False)
        self._sub_progress_bar2.setVisible(False)
        for bar in self._task_bars:
            bar.setVisible(False)
        self._lbl_status.setText(f"{tr(S.ERROR)}: {error_msg}")
        self._btn_analyze.setEnabled(True)
        self._btn_open.setEnabled(True)
        # Clean up process/queue references now that analysis is done.
        self._worker_process = None
        self._worker_queue = None
        QMessageBox.critical(self, tr(S.ERROR),
                             tr(S.ANALYSIS_FAILED).format(msg=error_msg))
        log.error("Analysis failed: %s", error_msg)

    # ------------------------------------------------------------------
    # Sensitivity re-filtering
    # ------------------------------------------------------------------

    @Slot(object)
    def _on_sensitivity_changed(self, _thresholds: object) -> None:
        """Re-filter detections when sensitivity sliders change."""
        self._apply_sensitivity_filter()

    def _apply_sensitivity_filter(self) -> None:
        """Apply current sensitivity thresholds to the loaded report."""
        if self._current_report is None:
            return
        from ..models import AnalysisReport, DetectionType
        thresholds = self._sensitivity_dialog.get_thresholds()
        filtered = [
            d for d in self._current_report.detections
            if d.type == DetectionType.PROFANITY
            or d.confidence >= thresholds.get(d.type, 0.0)
        ]
        filtered_report = AnalysisReport(
            audio_path=self._current_report.audio_path,
            duration_seconds=self._current_report.duration_seconds,
            segments=self._current_report.segments,
            detections=filtered,
        )
        self._report_table.load_report(filtered_report)
        self._transcript.load_segments(
            self._current_report.segments, filtered,
        )


# ===========================
# ENTRY POINT
# ===========================


def _parse_gui_args() -> argparse.Namespace:
    """Parse command-line arguments for the GUI application.

    Returns:
        Parsed namespace with *debug* flag.
    """
    parser = argparse.ArgumentParser(
        prog="monitor-gui",
        description="Child Monitor Analyzer -- GUI",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode: show console window and verbose logging.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=LOG_DIR,
        help=f"Directory for log files (default: {LOG_DIR}).",
    )
    return parser.parse_args()


def _setup_logging(debug: bool, log_dir: Path) -> Path:
    """Configure logging with both console and file handlers.

    Args:
        debug: If True, set log level to DEBUG; otherwise INFO.
        log_dir: Directory to store log files.

    Returns:
        Path to the log file.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"monitor_{timestamp}.log"

    fmt = "%(asctime)s [%(levelname)s] [%(threadName)s] %(pathname)s: %(message)s"

    # Root logger: keep at WARNING to silence noisy third-party libs.
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)

    # Our monitor.* logger: DEBUG in debug mode, INFO otherwise.
    monitor_logger = logging.getLogger("monitor")
    monitor_logger.setLevel(logging.DEBUG if debug else logging.INFO)

    # File handler -- added to monitor logger so it captures our logs.
    file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt))
    monitor_logger.addHandler(file_handler)

    # Console handler -- shows monitor.* logs at the appropriate level.
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    console_handler.setFormatter(logging.Formatter(fmt))
    monitor_logger.addHandler(console_handler)

    return log_file


def _allocate_console() -> None:
    """Allocate a visible console window on Windows for debug output."""
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.AllocConsole()
        # Reopen std streams to the new console.
        sys.stdout = open("CONOUT$", "w", encoding="utf-8")
        sys.stderr = open("CONOUT$", "w", encoding="utf-8")


def _install_crash_handlers() -> None:
    """Install global excepthook + Qt message handler so silent crashes log.

    Without this, an uncaught exception on the main thread (or in a Qt slot)
    can terminate the GUI with no log output. Qt warnings (FFmpeg, etc.) also
    bypass Python's logging by default. This routes everything through
    monitor's logger so post-mortem analysis from the log file is possible.
    """
    crash_log = logging.getLogger("monitor.crash")

    def _excepthook(exc_type, exc_value, exc_tb) -> None:
        # Don't swallow Ctrl+C -- let the default handler run.
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        crash_log.critical(
            "Uncaught exception", exc_info=(exc_type, exc_value, exc_tb),
        )

    sys.excepthook = _excepthook

    # threading excepthook (Python 3.8+) -- catches errors in worker threads.
    import threading

    def _thread_excepthook(args: "threading.ExceptHookArgs") -> None:
        crash_log.critical(
            "Uncaught exception in thread %s",
            args.thread.name if args.thread else "<unknown>",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    threading.excepthook = _thread_excepthook

    # Qt message handler -- routes Qt warnings to monitor.qt logger.
    qt_log = logging.getLogger("monitor.qt")
    level_map = {
        QtMsgType.QtDebugMsg: logging.DEBUG,
        QtMsgType.QtInfoMsg: logging.INFO,
        QtMsgType.QtWarningMsg: logging.WARNING,
        QtMsgType.QtCriticalMsg: logging.ERROR,
        QtMsgType.QtFatalMsg: logging.CRITICAL,
    }

    def _qt_handler(msg_type: "QtMsgType", context, message: str) -> None:
        qt_log.log(level_map.get(msg_type, logging.INFO), "%s", message)

    qInstallMessageHandler(_qt_handler)


def run_gui() -> None:
    """Launch the GUI application."""
    # Required for multiprocessing on Windows (especially PyInstaller).
    multiprocessing.freeze_support()

    args = _parse_gui_args()

    # In debug mode, allocate a console window (useful for PyInstaller --windowed).
    if args.debug:
        _allocate_console()

    log_file = _setup_logging(args.debug, args.log_dir)

    log.info("Starting Child Monitor Analyzer GUI (debug=%s)", args.debug)
    log.info("Log file: %s", log_file)

    app = QApplication(sys.argv)
    # Install crash handlers AFTER QApplication so qInstallMessageHandler
    # routes Qt's own diagnostic output through our logger.
    _install_crash_handlers()
    app.setLayoutDirection(Qt.LayoutDirection.RightToLeft)

    window = MainWindow()
    window.setWindowTitle(
        f"{tr(S.WINDOW_TITLE)} {'[DEBUG]' if args.debug else ''}"
    )
    window.show()
    sys.exit(app.exec())

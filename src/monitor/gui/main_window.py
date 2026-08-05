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
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import (
    QObject, QSettings, Qt, QThread, QTimer, QUrl, Signal, Slot,
    QtMsgType, qInstallMessageHandler,
)
from PySide6.QtGui import QDesktopServices, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressDialog,
    QProgressBar,
    QPushButton,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..analysis_worker import (
    run_worker,
    JOB_ANALYZE, JOB_SHUTDOWN,
    MSG_PROGRESS, MSG_SUB_PROGRESS, MSG_SUB_PROGRESS2, MSG_SUB_PROGRESS3,
    MSG_TASK_PROGRESS, MSG_PARTIAL_STT, MSG_PARTIAL_EVENTS,
    MSG_WARNING, MSG_FINISHED, MSG_ERROR, MSG_CANCELLED, MSG_READY,
)
from ..model_cache import setup_model_environment
from ..model_updates import fetch_new_hebrew_models
from ..models import (
    AnalysisReport, Detection, DetectionType,
    TranscribedSegment, TranscribedWord, sanitize_artifact_stem,
)
from .audio_player import AudioPlayerWidget
from .google_account import (
    GoogleAccountDialog, LinkWorker, UnlinkWorker, UploadWorker, isolate,
)
from .report_table import ReportTableWidget
from .sensitivity_panel import SensitivityDialog
from .transcript_widget import TranscriptExportDialog, TranscriptWidget
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
SETTINGS_STT_MODEL_KEY = "stt_model"

# STT model identifiers (must match values stored in QSettings).
STT_MODEL_THOROUGH = "thorough"  # ivrit-ai/whisper-large-v3-ct2
STT_MODEL_FAST = "fast"          # ivrit-ai/whisper-large-v3-turbo-ct2
STT_MODEL_NONE = "none"          # events-only, no transcription

# ===========================
# BACKGROUND WORKERS
# ===========================


def _log_gdrive_self_test() -> None:
    """Log the Drive feature's readiness at startup.

    ``keyring`` discovers its backends through entry points, which PyInstaller
    cannot see; a frozen build can silently resolve to a null backend. Logging
    the resolved backend name is the only cheap way to diagnose that from a
    user's log file.
    """
    try:
        from ..gdrive import auth, store
    except Exception:  # noqa: BLE001 - never block startup
        log.warning("Google Drive support is unavailable (import failed).",
                    exc_info=True)
        return
    log.info("Google Drive: OAuth client configured=%s", auth.is_configured())
    try:
        log.info("Google Drive: credential backend=%s", store.backend_name())
    except store.CredentialStoreUnavailable as exc:
        log.warning("Google Drive: no usable credential store (%s).", exc)


def _artifact_dir_for(audio_path: Optional[str]) -> Optional[Path]:
    """Return the artifact folder for *audio_path*, or None.

    Mirrors ``pipeline._artifact_dir`` but never creates the folder — the
    upload sidecar is written next to artifacts that already exist.
    """
    if not audio_path:
        return None
    path = Path(audio_path)
    return path.parent / sanitize_artifact_stem(path.stem)


class _ModelCheckWorker(QObject):
    """Runs the HuggingFace new-model query off the GUI thread.

    Emits ``finished(list)`` with the result on success, or ``failed(str)``
    with an error message on failure. Never raises.
    """

    finished = Signal(object)
    failed = Signal(str)

    @Slot()
    def run(self) -> None:
        try:
            models = fetch_new_hebrew_models()
            self.finished.emit(models)
        except Exception as exc:  # noqa: BLE001 - reported to the user
            log.warning("New-model check failed: %s", exc)
            self.failed.emit(str(exc))


# ===========================
# MAIN WINDOW
# ===========================


class MainWindow(QMainWindow):
    """Main application window with Hebrew RTL layout.

    Supports drag-and-drop of audio files and maintains a recent-files
    history (up to MAX_RECENT_FILES entries) persisted across sessions.

    Attributes:
        _worker_process: Long-lived analysis subprocess (started on demand).
        _job_queue: Queue for sending analysis jobs to the subprocess.
        _result_queue: Queue for receiving progress/results from subprocess.
        _current_audio: Path to the currently loaded audio file.
        _recent_files: Ordered list of recently opened file paths (newest first).
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(tr(S.WINDOW_TITLE))
        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)
        self.resize(WINDOW_DEFAULT_WIDTH, WINDOW_DEFAULT_HEIGHT)

        # One worker process serves every file, so models are loaded once
        # per session instead of once per file.
        self._worker_process: Optional[multiprocessing.Process] = None
        self._job_queue: Optional[multiprocessing.Queue] = None
        self._result_queue: Optional[multiprocessing.Queue] = None
        self._cancel_upto = None  # multiprocessing.Value: highest cancelled id
        self._job_counter = 0
        self._active_job_id: Optional[int] = None
        self._current_audio: Optional[str] = None
        self._current_report: Optional[AnalysisReport] = None

        # Accumulated partial results for incremental display.
        self._partial_segments: List["TranscribedSegment"] = []
        self._partial_detections: List["Detection"] = []
        self._partial_seg_starts: set = set()  # dedup by round(start, 2)
        self._partial_event_keys: set = set()  # dedup by (start, end, type)
        self._partial_dirty = False  # True when partials need UI refresh

        # Timer to poll the subprocess queue for progress/results.
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(50)  # 50 ms
        self._poll_timer.timeout.connect(self._poll_worker_queue)

        # Elapsed time tracking for analysis.
        self._analysis_start_time: Optional[float] = None
        self._analysis_elapsed_secs: int = 0
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(1000)  # 1 s
        self._elapsed_timer.timeout.connect(self._update_elapsed_label)

        # Debounce timer for refreshing UI with partial results (2s).
        self._partial_refresh_timer = QTimer(self)
        self._partial_refresh_timer.setInterval(2000)
        self._partial_refresh_timer.setSingleShot(True)
        self._partial_refresh_timer.timeout.connect(self._refresh_partial_ui)

        # Google Drive upload. The session is created lazily so a missing
        # keyring backend can never stop the app from starting.
        self._gdrive_session = None
        self._gdrive_thread: Optional[QThread] = None
        self._gdrive_worker: Optional[QObject] = None
        self._gdrive_store_ok: Optional[bool] = None
        self._gdrive_progress = None
        self._last_upload_request: Optional[dict] = None

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

        # --- Sub-progress bars for downloads (shown during model downloads) ---
        self._sub_progress_bar = QProgressBar()
        self._sub_progress_bar.setRange(0, 100)
        self._sub_progress_bar.setValue(0)
        self._sub_progress_bar.setVisible(False)
        self._sub_progress_bar.setTextVisible(True)
        self._sub_progress_bar.setFixedHeight(18)
        self._sub_progress_bar.setStyleSheet(
            "QProgressBar { font-size: 11px; }"
        )

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

        # --- Third sub-progress bar (toxicity model download) ---
        self._sub_progress_bar3 = QProgressBar()
        self._sub_progress_bar3.setRange(0, 100)
        self._sub_progress_bar3.setValue(0)
        self._sub_progress_bar3.setVisible(False)
        self._sub_progress_bar3.setTextVisible(True)
        self._sub_progress_bar3.setFixedHeight(18)
        self._sub_progress_bar3.setStyleSheet(
            "QProgressBar { font-size: 11px; }"
        )

        # Add download sub-progress bars before task bars (shown during
        # model download phase, hidden once models are loaded).
        layout.addWidget(self._sub_progress_bar2)
        layout.addWidget(self._sub_progress_bar3)

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

        # Sub-progress bar slot 0 is placed AFTER task bars because it is
        # reused for gap-fill progress (below the STT task bar).
        layout.addWidget(self._sub_progress_bar)

        # --- Sensitivity dialog (created once, shown on button click) ---
        self._sensitivity_dialog = SensitivityDialog(self)
        self._sensitivity_dialog.thresholds_changed.connect(self._on_sensitivity_changed)

        # --- Partial analysis warning banner ---
        self._lbl_partial_warning = QLabel(tr(S.PARTIAL_WARNING))
        self._lbl_partial_warning.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_partial_warning.setStyleSheet(
            "QLabel { background-color: #FFF3CD; color: #856404; "
            "font-size: 13px; font-weight: bold; padding: 6px; "
            "border: 1px solid #FFEEBA; border-radius: 4px; margin: 2px 8px; }"
        )
        self._lbl_partial_warning.setVisible(False)
        layout.addWidget(self._lbl_partial_warning)

        # --- Analysis warning banner (e.g. AI profanity model unavailable) ---
        self._lbl_analysis_warning = QLabel()
        self._lbl_analysis_warning.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_analysis_warning.setStyleSheet(
            "QLabel { background-color: #F8D7DA; color: #721C24; "
            "font-size: 13px; font-weight: bold; padding: 6px; "
            "border: 1px solid #F5C6CB; border-radius: 4px; margin: 2px 8px; }"
        )
        self._lbl_analysis_warning.setVisible(False)
        layout.addWidget(self._lbl_analysis_warning)

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
        self._report_table.detections_changed.connect(self._transcript.set_detections)
        self._report_table.events_changed.connect(self._audio_player.set_event_times)
        self._transcript.play_requested.connect(self._audio_player.seek_to)
        self._transcript.upload_requested.connect(self._upload_transcript_to_drive)
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

        # STT model selector.
        self._cmb_stt_model = QComboBox()
        self._cmb_stt_model.setFixedHeight(36)
        self._cmb_stt_model.addItem("מודל תמלול: יסודי", STT_MODEL_THOROUGH)
        self._cmb_stt_model.addItem("מודל תמלול: מהיר", STT_MODEL_FAST)
        self._cmb_stt_model.addItem("ללא תמלול", STT_MODEL_NONE)
        saved_model = self._settings.value(SETTINGS_STT_MODEL_KEY, STT_MODEL_THOROUGH)
        idx = self._cmb_stt_model.findData(saved_model)
        if idx >= 0:
            self._cmb_stt_model.setCurrentIndex(idx)
        self._cmb_stt_model.currentIndexChanged.connect(self._on_stt_model_changed)

        # Compact button to check HuggingFace for new Hebrew models.
        self._btn_check_models = QToolButton()
        self._btn_check_models.setFixedHeight(36)
        self._btn_check_models.setText(tr(S.CHECK_MODELS))
        self._btn_check_models.setToolTip(tr(S.CHECK_MODELS_TOOLTIP))
        self._btn_check_models.clicked.connect(self._check_new_models)
        self._model_check_thread: Optional[QThread] = None
        self._model_check_worker: Optional["_ModelCheckWorker"] = None

        self._lbl_file = QLineEdit(tr(S.NO_FILE_SELECTED))
        self._lbl_file.setReadOnly(True)
        self._lbl_file.setFrame(False)
        self._lbl_file.setStyleSheet(
            "QLineEdit { background: transparent; border: none; }"
        )

        self._btn_analyze = QPushButton(tr(S.ANALYSE))
        self._btn_analyze.setFixedHeight(36)
        self._btn_analyze.setEnabled(False)
        self._btn_analyze.clicked.connect(lambda: self._start_analysis(force_restart=True))

        # Elapsed time label (shown during and after analysis).
        self._lbl_elapsed = QLabel()
        self._lbl_elapsed.setStyleSheet(
            "QLabel { color: #666; font-size: 12px; padding: 0 4px; }"
        )
        self._lbl_elapsed.setVisible(False)

        # Google Drive account chip: indicator and menu in one control,
        # mirroring the recent-files button above.
        self._btn_google = QToolButton()
        self._btn_google.setFixedHeight(36)
        self._btn_google.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._google_menu = QMenu(self)
        self._google_menu.aboutToShow.connect(self._rebuild_google_menu)
        self._btn_google.setMenu(self._google_menu)
        self._refresh_google_chip()

        top_bar = QHBoxLayout()
        top_bar.addWidget(self._btn_analyze)
        top_bar.addWidget(self._lbl_elapsed)
        top_bar.addWidget(self._lbl_file, stretch=1)
        top_bar.addWidget(self._cmb_stt_model)
        top_bar.addWidget(self._btn_check_models)
        top_bar.addWidget(self._btn_sensitivity)
        top_bar.addWidget(self._btn_google)
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
        self._shutdown_worker()
        # Wait for any in-flight new-model check so the QThread is not
        # destroyed while still running.
        if self._model_check_thread is not None:
            self._model_check_thread.quit()
            self._model_check_thread.wait(3000)
        # Same for a Drive operation: destroying a running QThread aborts the
        # process. The OAuth loopback wait is the long pole, hence the timeout.
        if self._gdrive_thread is not None:
            self._gdrive_thread.quit()
            if not self._gdrive_thread.wait(5000):
                log.warning("A Google Drive operation did not finish before exit.")
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
        self._cancel_active_job()
        self._clear_partial_state()
        self._lbl_analysis_warning.setVisible(False)
        self._current_audio = file_path
        self._lbl_file.setText(str(Path(file_path)))
        self._lbl_file.setCursorPosition(0)
        self._transcript.set_export_source(file_path, self._cmb_stt_model.currentData())
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
        self._sub_progress_bar3.setVisible(False)
        self._sub_progress_bar3.setValue(0)
        for bar in self._task_bars:
            bar.setVisible(False)
            bar.setValue(0)
        self._lbl_status.setVisible(False)
        log.info("Audio file loaded: %s", file_path)

        # Try to load a cached analysis report for the selected model.
        stt_model_key = self._cmb_stt_model.currentData()
        cached = AnalysisReport.load_cache(file_path, stt_model_key)
        if cached is not None:
            self._current_report = cached
            self._apply_sensitivity_filter()
            self._lbl_status.setText(tr(S.LOADED_CACHED))
            self._lbl_status.setVisible(True)
            log.info("Loaded cached analysis for %s", file_path)

            # Check if gap-fill still needs to run for this file/model.
            from monitor.pipeline import is_gap_fill_complete
            if not is_gap_fill_complete(Path(file_path), stt_model_key):
                log.info("Gap-fill incomplete for %s; starting analysis to resume.", file_path)
                self._start_analysis()
        else:
            # No completed report -- start analysis.
            # Pre-load intermediate caches so the user sees data immediately
            # instead of waiting ~20s for the subprocess to reload models.
            self._current_report = None
            from monitor.pipeline import (has_intermediate_stt_cache,
                                          _load_intermediate_stt,
                                          _load_intermediate_events)
            stt_cache = _load_intermediate_stt(Path(file_path), stt_model_key)
            events_cache = _load_intermediate_events(Path(file_path))
            if stt_cache and stt_cache["segments"]:
                for seg in stt_cache["segments"]:
                    self._partial_seg_starts.add(round(seg.start, 2))
                    self._partial_segments.append(seg)
                log.info("Pre-loaded %d cached STT segments for immediate display.",
                         len(stt_cache["segments"]))
            if events_cache:
                self._partial_detections.extend(events_cache)
                log.info("Pre-loaded %d cached event detections for immediate display.",
                         len(events_cache))
            if self._partial_segments or self._partial_detections:
                self._schedule_partial_refresh()
            else:
                self._report_table.load_report(AnalysisReport(audio_path=file_path))
                self._transcript.load_segments([], [])
            if has_intermediate_stt_cache(Path(file_path), stt_model_key):
                log.info("No completed analysis for %s; resuming from intermediate cache.", file_path)
            else:
                log.info("No cache for %s; starting fresh analysis.", file_path)
            self._start_analysis()
            # Override progress bar to reflect pre-loaded cached state
            # instead of showing 0% while subprocess loads models.
            if self._partial_segments:
                self._progress_bar.setValue(30)
                self._progress_bar.setFormat(
                    "30%  |  \u05e0\u05d8\u05e2\u05df \u05de\u05de\u05d8\u05de\u05d5\u05df, \u05d8\u05d5\u05e2\u05df \u05de\u05d5\u05d3\u05dc\u05d9\u05dd...")

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
    # New-model check
    # ------------------------------------------------------------------

    @Slot()
    def _check_new_models(self) -> None:
        """Query HuggingFace (in a background thread) for new Hebrew models."""
        if self._model_check_thread is not None:
            return  # A check is already running.

        self._btn_check_models.setEnabled(False)
        self._lbl_status.setText(tr(S.CHECK_MODELS_CHECKING))
        self._lbl_status.setVisible(True)

        thread = QThread(self)
        worker = _ModelCheckWorker()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_model_check_finished)
        worker.failed.connect(self._on_model_check_failed)
        # Tear down the thread once the worker signals completion.
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_model_check_thread_done)
        # Keep strong references so neither object is garbage-collected
        # before the worker's run() executes on the background thread.
        self._model_check_thread = thread
        self._model_check_worker = worker
        thread.start()

    @Slot(object)
    def _on_model_check_finished(self, models: list) -> None:
        """Show the result of a successful new-model check."""
        self._lbl_status.setVisible(False)
        if models:
            listing = "\n".join(f"\u2022 {m['created']}  {m['id']}" for m in models)
            QMessageBox.information(
                self, tr(S.CHECK_MODELS_TITLE),
                tr(S.CHECK_MODELS_FOUND).format(list=listing),
            )
        else:
            QMessageBox.information(
                self, tr(S.CHECK_MODELS_TITLE), tr(S.CHECK_MODELS_NONE),
            )

    @Slot(str)
    def _on_model_check_failed(self, msg: str) -> None:
        """Show an error if the new-model check could not complete."""
        self._lbl_status.setVisible(False)
        QMessageBox.warning(
            self, tr(S.CHECK_MODELS_TITLE),
            tr(S.CHECK_MODELS_FAILED).format(msg=msg),
        )

    @Slot()
    def _on_model_check_thread_done(self) -> None:
        """Re-enable the check button once the background thread has finished."""
        self._model_check_thread = None
        self._model_check_worker = None
        self._btn_check_models.setEnabled(True)

    # ------------------------------------------------------------------
    # Google Drive
    # ------------------------------------------------------------------

    def _drive(self):
        """Return the Drive session, creating it on first use.

        Imported lazily: the Drive package pulls in ``keyring`` on demand, and
        a broken keystore must never prevent the application from starting.

        Returns:
            The ``GoogleDriveSession``, or None if the feature is unavailable.
        """
        if self._gdrive_session is not None:
            return self._gdrive_session
        try:
            from ..gdrive import GoogleDriveSession
            self._gdrive_session = GoogleDriveSession()
        except Exception:  # noqa: BLE001 - the feature simply stays off
            log.exception("Could not initialise the Google Drive session.")
            return None
        return self._gdrive_session

    def _drive_unavailable_reason(self) -> Optional[str]:
        """Return a user-facing reason the feature is off, or None if it is on.

        The credential-store probe is cached: this runs on every repaint of the
        account chip, and hitting the OS keystore that often is both slow and
        noisy in the log. Whether a keystore exists cannot change while the
        application is running.
        """
        try:
            from ..gdrive import auth, store
        except Exception:  # noqa: BLE001
            log.exception("The Google Drive package could not be imported.")
            return tr(S.GOOGLE_NOT_CONFIGURED)
        if not auth.is_configured():
            return tr(S.GOOGLE_NOT_CONFIGURED)
        if self._gdrive_store_ok is None:
            self._gdrive_store_ok = store.is_available()
        if not self._gdrive_store_ok:
            return tr(S.GOOGLE_KEYRING_UNAVAILABLE)
        return None

    def _refresh_google_chip(self) -> None:
        """Update the account chip's icon, text and tooltip."""
        from .player_icons import icon_cloud_upload

        busy = self._gdrive_thread is not None
        reason = self._drive_unavailable_reason()
        session = None if reason else self._drive()
        linked = bool(session and session.is_linked())

        self._btn_google.setIcon(icon_cloud_upload(linked))
        self._btn_google.setText("")
        # Progressive disclosure: keep the control visible but disabled with an
        # explanation, rather than hiding it and leaving the user puzzled.
        self._btn_google.setEnabled(reason is None and not busy)
        if reason:
            self._btn_google.setToolTip(reason)
        elif busy:
            self._btn_google.setToolTip(tr(S.GOOGLE_SIGNING_IN))
        elif linked:
            self._btn_google.setToolTip(
                f"{tr(S.GOOGLE_CONNECTED_AS)}{isolate(session.email)}"
            )
        else:
            self._btn_google.setToolTip(tr(S.GOOGLE_NOT_CONNECTED))

    @Slot()
    def _rebuild_google_menu(self) -> None:
        """Rebuild the account menu just before it is shown."""
        self._google_menu.clear()
        session = self._drive()
        linked = bool(session and session.is_linked())

        if linked:
            header = self._google_menu.addAction(
                f"{tr(S.GOOGLE_CONNECTED_AS)}{isolate(session.email)}"
            )
            header.setEnabled(False)
            self._google_menu.addSeparator()

        self._google_menu.addAction(
            tr(S.GOOGLE_MENU_ACCOUNT), self._open_google_account_dialog,
        )

        if linked:
            upload = self._google_menu.addAction(
                tr(S.GOOGLE_MENU_UPLOAD_CURRENT), self._upload_transcript_to_drive,
            )
            upload.setEnabled(self._transcript.has_content())
            folder_link = session.folder_link()
            open_folder = self._google_menu.addAction(
                tr(S.GOOGLE_MENU_OPEN_FOLDER),
                lambda: QDesktopServices.openUrl(QUrl(folder_link)),
            )
            open_folder.setEnabled(bool(folder_link))

            ask = self._google_menu.addAction(tr(S.GOOGLE_MENU_ASK_EVERY_TIME))
            ask.setCheckable(True)
            ask.setChecked(session.ask_every_time())
            ask.toggled.connect(session.set_ask_every_time)

            self._google_menu.addSeparator()
            self._google_menu.addAction(tr(S.GOOGLE_SIGN_OUT), self._unlink_google)
        else:
            self._google_menu.addAction(tr(S.GOOGLE_SIGN_IN), self._link_google)

    @Slot()
    def _open_google_account_dialog(self) -> None:
        """Show the account dialog and act on what the user asked for."""
        session = self._drive()
        if session is None:
            return
        dialog = GoogleAccountDialog(session, self)
        dialog.link_requested.connect(self._link_google)
        dialog.unlink_requested.connect(self._unlink_google)
        dialog.exec()

    def _start_gdrive_worker(self, worker, on_finished, on_failed) -> bool:
        """Run *worker* on a background QThread.

        Only one Drive operation runs at a time — concurrent auth flows would
        race two loopback listeners against each other.

        Returns:
            True if the worker was started.
        """
        if self._gdrive_thread is not None:
            log.info("A Google Drive operation is already running; ignoring.")
            return False

        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(on_finished)
        worker.failed.connect(on_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_gdrive_thread_done)
        # Strong references: without these the objects can be collected before
        # run() executes on the background thread.
        self._gdrive_thread = thread
        self._gdrive_worker = worker
        self._refresh_google_chip()
        thread.start()
        return True

    @Slot()
    def _on_gdrive_thread_done(self) -> None:
        self._gdrive_thread = None
        self._gdrive_worker = None
        self._refresh_google_chip()

    @Slot()
    def _link_google(self) -> None:
        """Start the interactive Google sign-in."""
        reason = self._drive_unavailable_reason()
        if reason:
            QMessageBox.information(self, tr(S.GOOGLE_ACCOUNT_TITLE), reason)
            return
        session = self._drive()
        if session is None:
            return

        worker = LinkWorker(session)
        # Without this the app looks hung for the full consent timeout if the
        # user closes the browser tab.
        progress = QProgressDialog(
            tr(S.GOOGLE_SIGNING_IN), tr(S.EXPORT_CANCEL), 0, 0, self,
        )
        progress.setWindowTitle(tr(S.GOOGLE_ACCOUNT_TITLE))
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        # DirectConnection is required, not stylistic: the worker thread is
        # blocked inside run() waiting for the callback, so it never drains its
        # event loop and a queued cancel would arrive only after the wait it is
        # meant to interrupt. cancel() just sets a threading.Event, which is
        # safe to call from the GUI thread.
        progress.canceled.connect(
            worker.cancel, Qt.ConnectionType.DirectConnection,
        )
        self._gdrive_progress = progress

        self._lbl_status.setText(tr(S.GOOGLE_SIGNING_IN))
        self._lbl_status.setVisible(True)
        if not self._start_gdrive_worker(
            worker, self._on_google_linked, self._on_google_auth_failed,
        ):
            self._close_gdrive_progress()
            self._lbl_status.setVisible(False)
            return
        progress.show()

    def _close_gdrive_progress(self) -> None:
        progress, self._gdrive_progress = self._gdrive_progress, None
        if progress is None:
            return
        # close() rejects the dialog, which emits canceled(). By now the worker
        # has finished and may already be queued for deletion, so detach first.
        try:
            progress.canceled.disconnect()
        except (RuntimeError, TypeError):
            pass
        progress.close()
        progress.deleteLater()

    @Slot(object)
    def _on_google_linked(self, _credentials) -> None:
        self._close_gdrive_progress()
        self._lbl_status.setVisible(False)
        self._refresh_google_chip()
        session = self._drive()
        QMessageBox.information(
            self, tr(S.GOOGLE_ACCOUNT_TITLE),
            f"{tr(S.GOOGLE_CONNECTED_AS)}{isolate(session.email if session else '')}",
        )

    @Slot(str)
    def _on_google_auth_failed(self, message: str) -> None:
        self._close_gdrive_progress()
        self._lbl_status.setVisible(False)
        self._refresh_google_chip()
        # An empty message means the user cancelled; do not scold them for it.
        if not message:
            return
        QMessageBox.warning(
            self, tr(S.GOOGLE_ACCOUNT_TITLE), f"{tr(S.GOOGLE_AUTH_FAILED)} {message}",
        )

    @Slot()
    def _unlink_google(self) -> None:
        """Revoke the grant server-side and purge the local credential."""
        session = self._drive()
        if session is None:
            return
        confirm = QMessageBox.question(
            self, tr(S.GOOGLE_ACCOUNT_TITLE),
            f"{tr(S.GOOGLE_SIGN_OUT)}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        self._start_gdrive_worker(
            UnlinkWorker(session), self._on_google_unlinked,
            self._on_google_auth_failed,
        )

    @Slot(object)
    def _on_google_unlinked(self, revoked: object) -> None:
        self._refresh_google_chip()
        if not revoked:
            QMessageBox.warning(
                self, tr(S.GOOGLE_ACCOUNT_TITLE),
                f"{tr(S.GOOGLE_AUTH_FAILED)} revoke",
            )

    @Slot()
    def _upload_transcript_to_drive(self) -> None:
        """Prompt for options, then upload the shown transcript to Drive."""
        if not self._transcript.has_content():
            QMessageBox.information(
                self, tr(S.UPLOAD_DIALOG_TITLE), tr(S.TRANSCRIPT_DOWNLOAD_EMPTY),
            )
            return

        reason = self._drive_unavailable_reason()
        if reason:
            QMessageBox.information(self, tr(S.UPLOAD_DIALOG_TITLE), reason)
            return
        session = self._drive()
        if session is None:
            return
        if not session.is_linked():
            # Never sign in and upload in one silent step.
            self._open_google_account_dialog()
            return

        from ..gdrive import DRIVE_FOLDER_NAME, client as drive_client, read_sidecar
        from ..gdrive.naming import sanitize_display_name

        audio_path, model_key = self._transcript.export_source()
        base_name = self._transcript.default_export_name()

        dialog = TranscriptExportDialog(
            self, mode="upload",
            default_name=base_name,
            target_description=(
                f"Google Drive › {DRIVE_FOLDER_NAME}\n"
                f"{isolate(session.email)}"
            ),
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        fmt = dialog.selected_format()
        base_name = dialog.selected_name() or base_name

        artifact_dir = _artifact_dir_for(audio_path)
        replace_existing = False
        previous = read_sidecar(artifact_dir, model_key or "none") if artifact_dir else {}
        # A renamed transcript is a different document, so there is nothing to
        # replace and nothing to ask about.
        same_name = previous.get("name") == sanitize_display_name(base_name)
        if (previous.get("file_id") and previous.get("account_id") == session.account_id
                and same_name):
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Question)
            box.setWindowTitle(tr(S.UPLOAD_EXISTING_TITLE))
            box.setText(tr(S.UPLOAD_EXISTING_QUESTION))
            # "Yes"/"No" cannot answer this question unambiguously.
            replace_button = box.addButton(
                tr(S.UPLOAD_REPLACE_EXISTING), QMessageBox.ButtonRole.AcceptRole,
            )
            box.addButton(tr(S.UPLOAD_CREATE_NEW), QMessageBox.ButtonRole.ActionRole)
            cancel_button = box.addButton(QMessageBox.StandardButton.Cancel)
            box.exec()
            if box.clickedButton() is cancel_button:
                return
            replace_existing = box.clickedButton() is replace_button

        try:
            content = self._transcript.build_export_bytes(
                fmt, dialog.include_timestamps(), dialog.include_events(),
            )
        except Exception as exc:  # noqa: BLE001 - e.g. python-docx missing
            log.exception("Could not build the transcript for upload.")
            QMessageBox.warning(
                self, tr(S.UPLOAD_DIALOG_TITLE), f"{tr(S.UPLOAD_FAILED)} {exc}",
            )
            return

        request = {
            "content": content,
            "content_type": (
                drive_client.DOCX_MIME if fmt == "docx" else drive_client.TXT_MIME
            ),
            "display_name": base_name,
            "artifact_dir": artifact_dir,
            "model_key": model_key or "none",
            "replace_existing": replace_existing,
        }
        self._last_upload_request = request
        log.info(
            "Uploading transcript to Drive: format=%s bytes=%d replace=%s",
            fmt, len(content), replace_existing,
        )
        self._lbl_status.setText(tr(S.UPLOAD_IN_PROGRESS))
        self._lbl_status.setVisible(True)
        self._start_gdrive_worker(
            UploadWorker(session, **request),
            self._on_upload_finished, self._on_upload_failed,
        )

    @Slot(object)
    def _on_upload_finished(self, result: object) -> None:
        self._lbl_status.setVisible(False)
        self._last_upload_request = None
        link = getattr(result, "web_view_link", None)
        box = QMessageBox(self)
        box.setWindowTitle(tr(S.UPLOAD_DIALOG_TITLE))
        box.setText(tr(S.UPLOAD_SUCCESS))
        box.addButton(QMessageBox.StandardButton.Ok)
        open_button = (
            box.addButton(tr(S.UPLOAD_OPEN_IN_DOCS), QMessageBox.ButtonRole.ActionRole)
            if link else None
        )
        box.exec()
        # `link` was validated (https + *.google.com) before it got here.
        if open_button is not None and box.clickedButton() is open_button:
            QDesktopServices.openUrl(QUrl(link))

    @Slot(str)
    def _on_upload_failed(self, message: str) -> None:
        self._lbl_status.setVisible(False)
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle(tr(S.UPLOAD_DIALOG_TITLE))
        box.setText(f"{tr(S.UPLOAD_FAILED)} {message}")
        box.addButton(QMessageBox.StandardButton.Close)
        retry = box.addButton(tr(S.UPLOAD_RETRY), QMessageBox.ButtonRole.ActionRole)
        box.exec()
        self._refresh_google_chip()
        if box.clickedButton() is not retry or not self._last_upload_request:
            # Do not keep a copy of the transcript alive for a retry that is
            # never going to happen.
            self._last_upload_request = None
            return
        session = self._drive()
        if session is not None:
            self._lbl_status.setText(tr(S.UPLOAD_IN_PROGRESS))
            self._lbl_status.setVisible(True)
            self._start_gdrive_worker(
                UploadWorker(session, **self._last_upload_request),
                self._on_upload_finished, self._on_upload_failed,
            )

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def _ensure_worker(self) -> None:
        """Start the analysis worker process if it is not already running.

        Raises:
            Exception: If the process could not be started.
        """
        if self._worker_process is not None and self._worker_process.is_alive():
            return

        if self._worker_process is not None:
            log.warning(
                "Analysis worker (PID %s) is gone (exit code %s); restarting it.",
                self._worker_process.pid, self._worker_process.exitcode,
            )
            self._discard_worker()

        self._job_queue = multiprocessing.Queue()
        self._result_queue = multiprocessing.Queue()
        # Highest job id the GUI has cancelled.  Monotonic, so a cancel can
        # never reach a job queued after it.
        self._cancel_upto = multiprocessing.Value("q", 0)
        self._worker_process = multiprocessing.Process(
            target=run_worker,
            args=(self._job_queue, self._result_queue, self._cancel_upto, None),
            daemon=True,
        )
        self._worker_process.start()
        log.info("Analysis worker started (PID %s).", self._worker_process.pid)

    def _discard_worker(self) -> None:
        """Drop all references to a dead worker so the next run respawns it."""
        for queue in (self._job_queue, self._result_queue):
            if queue is None:
                continue
            try:
                queue.close()
                queue.cancel_join_thread()
            except Exception:
                pass  # Queue may already be broken
        self._worker_process = None
        self._job_queue = None
        self._result_queue = None
        self._cancel_upto = None

    def _cancel_active_job(self) -> None:
        """Ask the worker to abandon the running job, keeping it warm.

        The worker checks for cancellation at phase and chunk boundaries, so
        it may keep working briefly.  Results are tagged with a job id, so
        anything still in flight is ignored by :meth:`_poll_worker_queue`.
        """
        self._poll_timer.stop()
        if self._active_job_id is None:
            return

        job_id = self._active_job_id
        self._active_job_id = None
        if self._cancel_upto is not None:
            with self._cancel_upto.get_lock():
                if self._cancel_upto.value < job_id:
                    self._cancel_upto.value = job_id
            log.info("Requested cancellation of analysis job %d.", job_id)

        # Drop anything the worker already queued for the cancelled job.
        self._drain_result_queue()

    def _drain_result_queue(self) -> None:
        """Discard every pending message without dispatching it."""
        if self._result_queue is None:
            return
        try:
            while True:
                self._result_queue.get_nowait()
        except (_queue_mod.Empty, OSError, ValueError):
            pass

    def _shutdown_worker(self, timeout: float = 5.0) -> None:
        """Stop the worker process; used when the window closes."""
        self._cancel_active_job()
        if self._worker_process is None:
            return

        pid = self._worker_process.pid
        if self._worker_process.is_alive():
            log.info("Shutting down analysis worker (PID %s)...", pid)
            try:
                if self._job_queue is not None:
                    self._job_queue.put({"type": JOB_SHUTDOWN})
                self._worker_process.join(timeout=timeout)
            except Exception:
                log.exception("Error requesting worker shutdown (PID %s).", pid)
            if self._worker_process.is_alive():
                log.warning("Worker did not exit in time; terminating (PID %s).", pid)
                try:
                    self._worker_process.terminate()
                    self._worker_process.join(timeout=3)
                except Exception:
                    log.exception("Error terminating analysis worker (PID %s).", pid)
        log.info("Analysis worker stopped (PID %s).", pid)
        self._discard_worker()

    @Slot()
    def _start_analysis(self, *, force_restart: bool = False) -> None:
        """Queue the current file for analysis on the warm worker process.

        Args:
            force_restart: If True, clears intermediate caches to force
                a full re-analysis from scratch.
        """
        if not self._current_audio:
            return

        self._cancel_active_job()
        self._clear_partial_state()

        # When the user explicitly clicks "Analyse" (force_restart=True),
        # clear intermediate caches so the analysis starts fresh.
        if force_restart and self._current_audio:
            from ..pipeline import _remove_intermediate_caches
            stt_key = self._cmb_stt_model.currentData()
            _remove_intermediate_caches(Path(self._current_audio), stt_key)
            log.info("Cleared intermediate caches for fresh analysis (model=%s).", stt_key)

        self._btn_analyze.setEnabled(False)
        self._btn_open.setEnabled(False)
        self._progress_bar.setVisible(True)
        self._progress_bar.setValue(0)

        # Start elapsed timer.
        self._analysis_start_time = time.monotonic()
        self._analysis_elapsed_secs = 0
        self._lbl_elapsed.setText(self._format_elapsed(0))
        self._lbl_elapsed.setVisible(True)
        self._elapsed_timer.start()

        try:
            self._ensure_worker()
            self._job_counter += 1
            self._active_job_id = self._job_counter
            self._job_queue.put({
                "type": JOB_ANALYZE,
                "job_id": self._active_job_id,
                "audio_path": self._current_audio,
                "stt_model_key": self._cmb_stt_model.currentData(),
            })
            self._poll_timer.start()
            log.info(
                "Analysis job %d queued (worker PID %s) for %s",
                self._active_job_id, self._worker_process.pid, self._current_audio,
            )
        except Exception as exc:
            log.exception("Failed to queue analysis job.")
            # Clean up partial state so buttons are re-enabled.
            self._active_job_id = None
            self._discard_worker()
            self._btn_analyze.setEnabled(True)
            self._btn_open.setEnabled(True)
            self._progress_bar.setVisible(False)
            self._elapsed_timer.stop()
            QMessageBox.critical(
                self, tr(S.ERROR),
                tr(S.ANALYSIS_FAILED).format(msg=f"Failed to start analysis: {exc}"),
            )

    # ------------------------------------------------------------------
    # STT model selector
    # ------------------------------------------------------------------

    @Slot(int)
    def _on_stt_model_changed(self, _index: int) -> None:
        """Persist the selected STT model; reload cached or restart analysis."""
        key = self._cmb_stt_model.currentData()
        self._settings.setValue(SETTINGS_STT_MODEL_KEY, key)
        log.info("STT model changed to: %s", key)
        self._transcript.set_export_source(self._current_audio, key)

        if not self._current_audio:
            return

        # Stop any running analysis and clear partial state.
        self._cancel_active_job()
        self._clear_partial_state()
        self._lbl_analysis_warning.setVisible(False)

        # If a completed analysis cache exists for the new model, load it.
        cached = AnalysisReport.load_cache(self._current_audio, key)
        if cached is not None:
            log.info("Loaded cached analysis for model %s.", key)
            self._current_report = cached
            self._apply_sensitivity_filter()
            self._lbl_status.setText(tr(S.LOADED_CACHED))
            self._lbl_status.setVisible(True)

            # Check if gap-fill still needs to run for this model.
            from monitor.pipeline import is_gap_fill_complete
            if not is_gap_fill_complete(Path(self._current_audio), key):
                log.info("Gap-fill incomplete for model %s; starting analysis to resume.", key)
                self._start_analysis()
                return

            # No more work to do — hide progress UI left over from prior run.
            self._progress_bar.setVisible(False)
            for bar in self._task_bars:
                bar.setVisible(False)
            self._elapsed_timer.stop()
            self._lbl_elapsed.setVisible(False)
            self._btn_analyze.setEnabled(True)
            self._btn_open.setEnabled(True)
            return

        # No completed report — pre-load intermediate caches for immediate display.
        from monitor.pipeline import (has_intermediate_stt_cache,
                                      _load_intermediate_stt,
                                      _load_intermediate_events)
        self._current_report = None
        stt_cache = _load_intermediate_stt(Path(self._current_audio), key)
        events_cache = _load_intermediate_events(Path(self._current_audio))
        if stt_cache and stt_cache["segments"]:
            for seg in stt_cache["segments"]:
                self._partial_seg_starts.add(round(seg.start, 2))
                self._partial_segments.append(seg)
            log.info("Pre-loaded %d cached STT segments for immediate display.",
                     len(stt_cache["segments"]))
        if events_cache:
            self._partial_detections.extend(events_cache)
            log.info("Pre-loaded %d cached event detections for immediate display.",
                     len(events_cache))
        if self._partial_segments or self._partial_detections:
            self._schedule_partial_refresh()
        else:
            self._report_table.load_report(AnalysisReport(audio_path=self._current_audio))
            self._transcript.load_segments([], [])
        if has_intermediate_stt_cache(Path(self._current_audio), key):
            log.info("No completed analysis for model %s; resuming from intermediate cache.", key)
        else:
            log.info("No cache for model %s; starting fresh analysis.", key)
        self._start_analysis()
        # Override progress bar to reflect pre-loaded cached state.
        if self._partial_segments:
            self._progress_bar.setValue(30)
            self._progress_bar.setFormat(
                "30%  |  \u05e0\u05d8\u05e2\u05df \u05de\u05de\u05d8\u05de\u05d5\u05df, \u05d8\u05d5\u05e2\u05df \u05de\u05d5\u05d3\u05dc\u05d9\u05dd...")

    # ------------------------------------------------------------------

    @Slot()
    def _poll_worker_queue(self) -> None:
        """Drain pending messages from the worker queue.

        Called every 50 ms by _poll_timer.  Dispatches each message to
        the appropriate handler and detects worker crashes.  Messages from
        a superseded job are discarded via their job id.
        """
        if self._result_queue is None or self._worker_process is None:
            self._poll_timer.stop()
            return

        # Drain all available messages.
        try:
            while True:
                try:
                    msg = self._result_queue.get_nowait()
                except _queue_mod.Empty:
                    break
                job_id = msg.get("job_id")
                if job_id is not None and job_id != self._active_job_id:
                    continue  # stale message from a cancelled/superseded job
                self._handle_worker_message(msg)
                # Stop polling after terminal messages.
                if msg.get("type") in (MSG_FINISHED, MSG_ERROR, MSG_CANCELLED):
                    return
        except Exception:
            log.exception("Error reading from worker queue.")

        # Detect worker crash (died without sending finished/error).
        if not self._worker_process.is_alive():
            pid, exitcode = self._worker_process.pid, self._worker_process.exitcode
            self._poll_timer.stop()
            log.error(
                "Analysis worker died unexpectedly (PID %s, exit code %s).",
                pid, exitcode,
            )
            self._discard_worker()
            self._on_error(
                f"Analysis process crashed unexpectedly (exit code {exitcode})."
            )

    def _handle_worker_message(self, msg: dict) -> None:
        """Dispatch a single message from the worker queue."""
        msg_type = msg.get("type")
        try:
            if msg_type == MSG_PROGRESS:
                self._on_progress(msg["pct"], msg["msg"])
            elif msg_type == MSG_SUB_PROGRESS:
                self._on_sub_progress(msg["done"], msg["total"], msg["label"])
            elif msg_type == MSG_SUB_PROGRESS2:
                self._on_sub_progress2(msg["done"], msg["total"], msg["label"])
            elif msg_type == MSG_SUB_PROGRESS3:
                self._on_sub_progress3(msg["done"], msg["total"], msg["label"])
            elif msg_type == MSG_TASK_PROGRESS:
                self._on_task_progress(msg["task_id"], msg["pct"], msg["label"])
            elif msg_type == MSG_PARTIAL_STT:
                self._on_partial_stt(msg["segments"])
            elif msg_type == MSG_PARTIAL_EVENTS:
                self._on_partial_events(msg["detections"])
            elif msg_type == MSG_WARNING:
                self._on_analysis_warning(msg["key"])
            elif msg_type == MSG_READY:
                log.info("Analysis worker is ready.")
            elif msg_type == MSG_CANCELLED:
                log.info("Worker confirmed job %s was cancelled.", msg.get("job_id"))
                self._poll_timer.stop()
                self._active_job_id = None
            elif msg_type == MSG_FINISHED:
                report = AnalysisReport.from_dict(msg["report"])
                self._poll_timer.stop()
                self._on_finished(report)
            elif msg_type == MSG_ERROR:
                tb = msg.get("traceback", "")
                if tb:
                    log.error("Worker traceback:\n%s", tb)
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

    @Slot(object, object, str)
    def _on_sub_progress3(self, done: int, total: int, label: str) -> None:
        """Update the third sub-progress bar (toxicity model download)."""
        self._update_sub_bar(self._sub_progress_bar3, done, total, label)

    @staticmethod
    def _update_sub_bar(bar: QProgressBar, done: int, total: int, label: str) -> None:
        """Shared helper to render a sub-operation progress bar.

        Supports two modes:
        - Byte-level download progress (large totals, shown as MB)
        - Gap-fill progress (small totals in seconds, shown as time)
        """
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
            bar.setRange(0, 1000)
            bar.setValue(int(done * 1000 / total))
            pct = done * 100 // total
            # Detect gap-fill progress (seconds) vs download progress (bytes).
            # Gap-fill totals are typically < 100_000 seconds; downloads are
            # in millions of bytes.
            if total < 100_000:
                # Gap-fill mode: label already contains the position info.
                bar.setFormat(f"{label} ({pct}%)")
            else:
                # Download mode: show MB.
                done_mb = done / (1024 * 1024)
                total_mb = total / (1024 * 1024)
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

    # ------------------------------------------------------------------
    # Incremental partial results
    # ------------------------------------------------------------------

    def _on_partial_stt(self, segment_dicts: list) -> None:
        """Accumulate partial STT segments and schedule a debounced UI refresh."""
        for d in segment_dicts:
            try:
                key = round(d["start"], 2)
                if key in self._partial_seg_starts:
                    continue
                seg = TranscribedSegment(
                    text=d["text"],
                    start=d["start"],
                    end=d["end"],
                    words=[
                        TranscribedWord(
                            word=w["word"], start=w["start"],
                            end=w["end"], confidence=w.get("confidence", 1.0),
                        )
                        for w in d.get("words", [])
                    ],
                )
                self._partial_seg_starts.add(key)
                self._partial_segments.append(seg)
            except (KeyError, TypeError):
                log.debug("Skipping malformed partial STT segment: %r", d)
        self._schedule_partial_refresh()

    def _on_partial_events(self, detection_dicts: list) -> None:
        """Accumulate partial event detections and schedule a debounced UI refresh."""
        for d in detection_dicts:
            try:
                key = (round(d["start"], 2), round(d["end"], 2), d["type"])
                if key in self._partial_event_keys:
                    continue
                det = Detection(
                    type=DetectionType(d["type"]),
                    start=d["start"],
                    end=d["end"],
                    confidence=d.get("confidence", 1.0),
                    details=d.get("details", {}),
                )
                self._partial_event_keys.add(key)
                self._partial_detections.append(det)
            except (KeyError, TypeError, ValueError):
                log.debug("Skipping malformed partial detection: %r", d)
        self._schedule_partial_refresh()

    def _schedule_partial_refresh(self) -> None:
        """Start or restart the debounce timer for partial UI refresh."""
        self._partial_dirty = True
        # Always restart the timer so the latest data is shown within 2s.
        self._partial_refresh_timer.start()
        # Show warning banner on first partial data.
        if not self._lbl_partial_warning.isVisible():
            self._lbl_partial_warning.setVisible(True)

    @Slot()
    def _refresh_partial_ui(self) -> None:
        """Rebuild the report table and transcript with current partial data."""
        if not self._partial_dirty:
            return
        self._partial_dirty = False
        audio = self._current_audio or ""
        # Apply sensitivity thresholds to partial detections.
        thresholds = self._sensitivity_dialog.get_thresholds()
        filtered_dets = [
            d for d in self._partial_detections
            if d.type == DetectionType.PROFANITY
            or d.confidence >= thresholds.get(d.type, 0.0)
        ]
        # Build a temporary partial report for the table.
        partial_report = AnalysisReport(
            audio_path=audio,
            duration_seconds=self._compute_partial_duration(),
            segments=list(self._partial_segments),
            detections=filtered_dets,
        )
        # Set transcript segments first; the report table then drives the
        # visible detections via detections_changed -> set_detections.
        self._transcript.load_segments(self._partial_segments)
        self._report_table.load_report(partial_report)
        log.debug(
            "Partial UI refresh: %d segments, %d detections (%d filtered)",
            len(self._partial_segments), len(self._partial_detections),
            len(filtered_dets),
        )

    def _compute_partial_duration(self) -> float:
        """Estimate duration from accumulated partial data."""
        dur = 0.0
        if self._partial_segments:
            dur = max(s.end for s in self._partial_segments)
        if self._partial_detections:
            dur = max(dur, max(d.end for d in self._partial_detections))
        return dur

    def _clear_partial_state(self) -> None:
        """Reset partial accumulation state."""
        self._partial_segments.clear()
        self._partial_detections.clear()
        self._partial_seg_starts.clear()
        self._partial_event_keys.clear()
        self._partial_dirty = False
        self._partial_refresh_timer.stop()
        self._lbl_partial_warning.setVisible(False)
        # Hide sub-progress bars so stale bars from a prior run don't linger.
        self._sub_progress_bar.setVisible(False)
        self._sub_progress_bar2.setVisible(False)
        self._sub_progress_bar3.setVisible(False)

    def _on_analysis_warning(self, key: str) -> None:
        """Show a non-fatal analysis warning banner."""
        text = tr(key)
        log.warning("Analysis warning: %s", text)
        self._lbl_analysis_warning.setText(text)
        self._lbl_analysis_warning.setVisible(True)

    @Slot(object)
    def _on_finished(self, report: AnalysisReport) -> None:
        """Handle successful analysis completion.

        Args:
            report: The completed AnalysisReport.
        """
        # Guard against stale messages processed after the job was cancelled.
        if self._active_job_id is None:
            log.info("Ignoring stale finished message (job already stopped).")
            return
        self._progress_bar.setVisible(False)
        self._sub_progress_bar.setVisible(False)
        self._sub_progress_bar2.setVisible(False)
        self._sub_progress_bar3.setVisible(False)
        for bar in self._task_bars:
            bar.setVisible(False)
        self._lbl_status.setVisible(False)
        self._btn_analyze.setEnabled(True)
        self._btn_open.setEnabled(True)
        # Stop elapsed timer but keep the label visible with final time.
        self._elapsed_timer.stop()
        if self._analysis_start_time is not None:
            elapsed = int(time.monotonic() - self._analysis_start_time)
            self._analysis_elapsed_secs = elapsed
            self._lbl_elapsed.setText(self._format_elapsed(elapsed))
            # Keep label visible — user can see how long it took.
        # The worker stays alive with its models loaded, ready for the next file.
        self._active_job_id = None
        # Clear partial state — final report replaces partials.
        self._clear_partial_state()
        self._current_report = report
        self._apply_sensitivity_filter()
        log.info(
            "Analysis complete: %d detections found.", len(report.detections)
        )

    @Slot(str)
    def _on_error(self, error_msg: str) -> None:
        """Handle analysis failure.

        Args:
            error_msg: Description of the error.
        """
        if self._active_job_id is None:
            log.info("Ignoring stale error message (job already stopped).")
            return
        self._progress_bar.setVisible(False)
        self._sub_progress_bar.setVisible(False)
        self._sub_progress_bar2.setVisible(False)
        self._sub_progress_bar3.setVisible(False)
        for bar in self._task_bars:
            bar.setVisible(False)
        self._lbl_status.setText(f"{tr(S.ERROR)}: {error_msg}")
        self._btn_analyze.setEnabled(True)
        self._btn_open.setEnabled(True)
        # Stop elapsed timer but keep the label visible.
        self._elapsed_timer.stop()
        if self._analysis_start_time is not None:
            elapsed = int(time.monotonic() - self._analysis_start_time)
            self._analysis_elapsed_secs = elapsed
            self._lbl_elapsed.setText(self._format_elapsed(elapsed))
        # The worker stays alive with its models loaded, ready for the next file.
        self._active_job_id = None
        self._clear_partial_state()
        QMessageBox.critical(self, tr(S.ERROR),
                             tr(S.ANALYSIS_FAILED).format(msg=error_msg))
        log.error("Analysis failed: %s", error_msg)

    # ------------------------------------------------------------------
    # Elapsed time helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_elapsed(seconds: int) -> str:
        """Format elapsed seconds as a readable string."""
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        if h:
            return f"\u23F1 {h}:{m:02d}:{s:02d}"
        return f"\u23F1 {m}:{s:02d}"

    @Slot()
    def _update_elapsed_label(self) -> None:
        """Tick the elapsed time label once per second during analysis."""
        if self._analysis_start_time is None:
            return
        elapsed = int(time.monotonic() - self._analysis_start_time)
        self._analysis_elapsed_secs = elapsed
        self._lbl_elapsed.setText(self._format_elapsed(elapsed))

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
        # Set transcript segments first; the report table then drives the
        # visible detections (type + details + sensitivity) via
        # detections_changed -> transcript.set_detections.
        self._transcript.load_segments(self._current_report.segments)
        self._report_table.load_report(filtered_report)


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
    parser.add_argument(
        "--normal-priority",
        action="store_true",
        help="Run analysis workers at normal OS priority "
             "(default: workers run at lowest/idle priority).",
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

    from ..log_redaction import install_redaction
    install_redaction()

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

    # Record the worker-priority preference so spawned analysis
    # subprocesses inherit it via the environment.  The GUI process
    # itself stays at normal priority.
    from monitor.priority import set_low_priority_env
    set_low_priority_env(not args.normal_priority)

    # In debug mode, allocate a console window (useful for PyInstaller --windowed).
    if args.debug:
        _allocate_console()

    log_file = _setup_logging(args.debug, args.log_dir)

    log.info("Starting Child Monitor Analyzer GUI (debug=%s)", args.debug)
    log.info("Log file: %s", log_file)
    _log_gdrive_self_test()

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

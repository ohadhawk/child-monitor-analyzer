"""
Subprocess entry point for running the analysis pipeline.

This module is imported by the child process spawned from the GUI.
It creates its own AnalysisPipeline, runs analyze(), and sends
progress + results back to the parent via a multiprocessing.Queue.

The child process is fully independent: it loads its own models,
configures its own logging, and can be terminated cleanly via
Process.terminate() without affecting the parent.

Usage (from main_window.py):
    process = multiprocessing.Process(
        target=analysis_worker.run_analysis,
        args=(audio_path, queue),
        daemon=True,
    )
    process.start()
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import sys
import traceback
from pathlib import Path

log = logging.getLogger(__name__)

# ===========================
# MESSAGE TYPES
# ===========================

MSG_PROGRESS = "progress"
MSG_SUB_PROGRESS = "sub_progress"
MSG_SUB_PROGRESS2 = "sub_progress2"
MSG_TASK_PROGRESS = "task_progress"
MSG_FINISHED = "finished"
MSG_ERROR = "error"


# ===========================
# CHILD-PROCESS LOGGING
# ===========================

def _setup_child_logging(log_dir: str | None = None) -> None:
    """Configure logging in the child process.

    Writes to a separate log file in the same directory as the parent's
    logs so that child-process output is always captured, even if the
    child is terminated mid-analysis.

    Args:
        log_dir: Directory for log files. Uses default if None.
    """
    from datetime import datetime

    if log_dir is None:
        log_dir = str(Path.home() / ".child-monitor-analyzer" / "logs")

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"worker_{timestamp}_{os.getpid()}.log"

    fmt = "%(asctime)s [%(levelname)s] [PID %(process)d] %(name)s: %(message)s"

    # Root logger at WARNING to silence third-party noise.
    root = logging.getLogger()
    root.setLevel(logging.WARNING)

    # Our monitor.* logger at DEBUG.
    monitor_log = logging.getLogger("monitor")
    monitor_log.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt))
    monitor_log.addHandler(file_handler)

    # Also log to stderr (captured by parent on Windows if needed).
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(fmt))
    monitor_log.addHandler(console_handler)

    log.info("Child process logging initialised: %s", log_file)


# ===========================
# QUEUE HELPERS
# ===========================

def _safe_put(queue: multiprocessing.Queue, msg: dict) -> None:
    """Put a message into the queue, swallowing errors if the queue is broken.

    After the parent calls process.terminate(), the queue pipe may be
    closed. We don't want the child to crash with BrokenPipeError in
    that case — it's about to die anyway.

    Uses a longer timeout for terminal messages (finished/error) to
    avoid silently losing the result.
    """
    try:
        if msg.get("type") in (MSG_FINISHED, MSG_ERROR):
            queue.put(msg, timeout=10)
        else:
            queue.put_nowait(msg)
    except Exception:
        pass  # Queue broken or full — child is being terminated


# ===========================
# SUBPROCESS ENTRY POINT
# ===========================

def run_analysis(
    audio_path: str,
    queue: multiprocessing.Queue,
    log_dir: str | None = None,
) -> None:
    """Entry point for the analysis subprocess.

    Creates a fresh AnalysisPipeline, runs the full analysis, and sends
    progress updates + the final result (or error) back via *queue*.

    This function never raises — all exceptions are caught and forwarded
    as error messages through the queue.

    Args:
        audio_path: Absolute path to the audio file to analyse.
        queue: multiprocessing.Queue for sending messages to the parent.
        log_dir: Optional log directory (forwarded from parent).
    """
    try:
        # --- Set up environment in child process ---
        _setup_child_logging(log_dir)
        log.info("Analysis subprocess started (PID %d) for: %s", os.getpid(), audio_path)

        from .model_cache import setup_model_environment
        setup_model_environment()

        from .pipeline import AnalysisPipeline

        # --- Build progress callbacks that forward to the queue ---
        def on_progress(pct: int, msg: str) -> None:
            _safe_put(queue, {"type": MSG_PROGRESS, "pct": pct, "msg": msg})

        def on_sub_progress(done: int, total: int, label: str) -> None:
            _safe_put(queue, {
                "type": MSG_SUB_PROGRESS,
                "done": done, "total": total, "label": label,
            })

        def on_sub_progress2(done: int, total: int, label: str) -> None:
            _safe_put(queue, {
                "type": MSG_SUB_PROGRESS2,
                "done": done, "total": total, "label": label,
            })

        def on_task_progress(task_id: int, pct: int, label: str) -> None:
            _safe_put(queue, {
                "type": MSG_TASK_PROGRESS,
                "task_id": task_id, "pct": pct, "label": label,
            })

        # --- Run the pipeline ---
        pipeline = AnalysisPipeline()
        report = pipeline.analyze(
            audio_path,
            on_progress=on_progress,
            on_sub_progress=on_sub_progress,
            on_sub_progress2=on_sub_progress2,
            on_task_progress=on_task_progress,
        )

        # Send the result as a serialised dict (safe for cross-process pickling).
        _safe_put(queue, {"type": MSG_FINISHED, "report": report.to_dict()})
        log.info("Analysis subprocess completed successfully (PID %d).", os.getpid())

    except Exception as exc:
        tb = traceback.format_exc()
        log.exception("Analysis subprocess failed (PID %d) for %s", os.getpid(), audio_path)
        _safe_put(queue, {
            "type": MSG_ERROR,
            "msg": str(exc),
            "traceback": tb,
        })

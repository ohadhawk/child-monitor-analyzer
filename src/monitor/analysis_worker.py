"""
Subprocess entry point for running the analysis pipeline.

The GUI starts one long-lived worker process and feeds it analysis jobs
through a queue.  The worker keeps its models loaded between jobs, so only
the first file of a session pays the (~20 s) model load cost.

Because the process must survive between files it can no longer be stopped
with Process.terminate(); cancellation is cooperative via ``monitor.cancellation``
and a shared ``cancel_upto`` counter (see :func:`run_worker`).

Usage (from main_window.py):
    process = multiprocessing.Process(
        target=analysis_worker.run_worker,
        args=(job_queue, result_queue, cancel_upto),
        daemon=True,
    )
    process.start()
    job_queue.put({"type": JOB_ANALYZE, "job_id": 1, "audio_path": ..., ...})
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import queue as _queue_mod
import sys
import threading
import traceback
from pathlib import Path

log = logging.getLogger(__name__)

# ===========================
# MESSAGE TYPES
# ===========================

MSG_PROGRESS = "progress"
MSG_SUB_PROGRESS = "sub_progress"
MSG_SUB_PROGRESS2 = "sub_progress2"
MSG_SUB_PROGRESS3 = "sub_progress3"
MSG_TASK_PROGRESS = "task_progress"
MSG_PARTIAL_STT = "partial_stt"
MSG_PARTIAL_EVENTS = "partial_events"
MSG_WARNING = "warning"
MSG_FINISHED = "finished"
MSG_ERROR = "error"
MSG_CANCELLED = "cancelled"
MSG_READY = "ready"

# ===========================
# JOB TYPES (parent -> worker)
# ===========================

JOB_ANALYZE = "analyze"
JOB_SHUTDOWN = "shutdown"

# How often the watchdog thread checks whether the GUI asked to cancel.
_CANCEL_POLL_SECONDS = 0.15


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

    from .log_redaction import install_redaction
    install_redaction()

    log.info("Child process logging initialised: %s", log_file)


# ===========================
# QUEUE HELPERS
# ===========================

def _safe_put(queue: multiprocessing.Queue, msg: dict) -> None:
    """Put a message into the queue, swallowing errors if the queue is broken.

    After the parent shuts the worker down, the queue pipe may be closed.
    We don't want the child to crash with BrokenPipeError in that case.

    Uses a longer timeout for terminal messages (finished/error/cancelled)
    to avoid silently losing the result.
    """
    try:
        if msg.get("type") in (MSG_FINISHED, MSG_ERROR, MSG_CANCELLED):
            queue.put(msg, timeout=10)
        else:
            queue.put_nowait(msg)
    except Exception:
        pass  # Queue broken or full — child is shutting down


# ===========================
# WARM MODEL POOL
# ===========================

class ModelPool:
    """Keeps the detectors alive between jobs so models load only once.

    The audio-event and profanity detectors are independent of the chosen
    STT model and are shared by every job.  Only one speech-to-text engine
    is kept resident at a time — holding both ``fast`` and ``thorough``
    would cost roughly 1.5 GB of extra RAM.
    """

    def __init__(self) -> None:
        self._audio_events = None
        self._profanity = None
        self._stt = None
        self._stt_model_name: str | None = None

    def pipeline_for(self, stt_model_key: str):
        """Return an AnalysisPipeline wired to the warm detectors."""
        from .pipeline import AnalysisPipeline

        if self._audio_events is None:
            from .audio_events import AudioEventDetector
            self._audio_events = AudioEventDetector()
        if self._profanity is None:
            from .profanity import ProfanityDetector
            self._profanity = ProfanityDetector(use_ai=True)

        # AnalysisPipeline itself holds no per-file state, so a fresh
        # instance per job is free and avoids stale-state surprises.
        return AnalysisPipeline(
            stt=self._stt_for(stt_model_key),
            audio_events=self._audio_events,
            profanity=self._profanity,
        )

    def _stt_for(self, stt_model_key: str):
        """Return the cached STT engine for *stt_model_key* (None = events only)."""
        if stt_model_key == "none":
            log.info("No-transcription mode: skipping STT model.")
            return None

        from .stt import DEFAULT_MODEL, TURBO_MODEL
        model_name = TURBO_MODEL if stt_model_key == "fast" else DEFAULT_MODEL
        if self._stt is not None and self._stt_model_name == model_name:
            log.info("Reusing warm STT model: %s (%s)", model_name, stt_model_key)
            return self._stt

        if self._stt is not None:
            log.info(
                "STT model changed %s -> %s; releasing the previous engine.",
                self._stt_model_name, model_name,
            )
            self._stt = None
            self._stt_model_name = None

        from .stt import HebrewSTT
        log.info("Using STT model: %s (%s)", model_name, stt_model_key)
        self._stt = HebrewSTT(model_name=model_name)
        self._stt_model_name = model_name
        return self._stt


# ===========================
# SUBPROCESS ENTRY POINT
# ===========================

def _watch_cancellation(
    cancel_upto,
    state: dict,
    stop_event: threading.Event,
) -> None:
    """Translate the parent's cancel counter into a cooperative cancel request.

    The parent never lowers *cancel_upto*, and every new job gets a strictly
    higher id, so a stale cancel can never abort a newer job.
    """
    from . import cancellation

    while not stop_event.wait(_CANCEL_POLL_SECONDS):
        job_id = state.get("job_id")
        if job_id is not None and cancel_upto.value >= job_id:
            cancellation.request_cancel()


def _run_job(
    job: dict,
    queue: multiprocessing.Queue,
    pool: ModelPool,
    cancel_upto,
    state: dict,
) -> None:
    """Run a single analysis job and report the outcome through *queue*.

    Never raises — failures are forwarded as MSG_ERROR.
    """
    from . import cancellation

    job_id = job["job_id"]
    audio_path = job["audio_path"]
    stt_model_key = job.get("stt_model_key", "thorough")

    cancellation.begin_job()
    state["job_id"] = job_id
    if cancel_upto.value >= job_id:
        # Cancelled while it was still queued.
        state["job_id"] = None
        _safe_put(queue, {"type": MSG_CANCELLED, "job_id": job_id})
        return

    def _put(msg: dict) -> None:
        msg["job_id"] = job_id
        _safe_put(queue, msg)

    def on_progress(pct: int, msg: str) -> None:
        _put({"type": MSG_PROGRESS, "pct": pct, "msg": msg})

    def on_sub_progress(done: int, total: int, label: str) -> None:
        _put({
            "type": MSG_SUB_PROGRESS,
            "done": done, "total": total, "label": label,
        })

    def on_sub_progress2(done: int, total: int, label: str) -> None:
        _put({
            "type": MSG_SUB_PROGRESS2,
            "done": done, "total": total, "label": label,
        })

    def on_sub_progress3(done: int, total: int, label: str) -> None:
        _put({
            "type": MSG_SUB_PROGRESS3,
            "done": done, "total": total, "label": label,
        })

    def on_task_progress(task_id: int, pct: int, label: str) -> None:
        _put({
            "type": MSG_TASK_PROGRESS,
            "task_id": task_id, "pct": pct, "label": label,
        })

    def on_partial_stt(segments: list) -> None:
        _put({"type": MSG_PARTIAL_STT, "segments": segments})

    def on_partial_events(detections: list) -> None:
        _put({"type": MSG_PARTIAL_EVENTS, "detections": detections})

    def on_warning(key: str) -> None:
        _put({"type": MSG_WARNING, "key": key})

    try:
        log.info("Job %d started: %s (model=%s)", job_id, audio_path, stt_model_key)
        pipeline = pool.pipeline_for(stt_model_key)
        report = pipeline.analyze(
            audio_path,
            on_progress=on_progress,
            on_sub_progress=on_sub_progress,
            on_sub_progress2=on_sub_progress2,
            on_sub_progress3=on_sub_progress3,
            on_task_progress=on_task_progress,
            on_partial_stt=on_partial_stt,
            on_partial_events=on_partial_events,
            on_warning=on_warning,
            stt_model_key=stt_model_key,
        )
        # Send the result as a serialised dict (safe for cross-process pickling).
        _put({"type": MSG_FINISHED, "report": report.to_dict()})
        log.info("Job %d completed successfully.", job_id)

    except cancellation.AnalysisCancelled:
        log.info("Job %d cancelled.", job_id)
        _put({"type": MSG_CANCELLED})

    except Exception as exc:
        tb = traceback.format_exc()
        log.exception("Job %d failed for %s", job_id, audio_path)
        _put({"type": MSG_ERROR, "msg": str(exc), "traceback": tb})

    finally:
        state["job_id"] = None
        cancellation.begin_job()


def run_worker(
    job_queue: multiprocessing.Queue,
    result_queue: multiprocessing.Queue,
    cancel_upto,
    log_dir: str | None = None,
) -> None:
    """Entry point for the long-lived analysis subprocess.

    Loads models on the first job and keeps them warm for every job that
    follows, so only the first file of a session pays the model load cost.

    This function never raises — job failures are forwarded as MSG_ERROR
    and the loop keeps running.

    Args:
        job_queue: Queue of job dicts from the parent.  Each is either
            ``{"type": JOB_ANALYZE, "job_id": int, "audio_path": str,
            "stt_model_key": str}`` or ``{"type": JOB_SHUTDOWN}``.
        result_queue: Queue for sending progress/results to the parent.
        cancel_upto: Shared ``multiprocessing.Value`` holding the highest
            job id the parent has cancelled.
        log_dir: Optional log directory (forwarded from parent).
    """
    stop_watchdog = threading.Event()
    watchdog: threading.Thread | None = None
    try:
        _setup_child_logging(log_dir)
        log.info("Analysis worker started (PID %d).", os.getpid())

        # Run this CPU-heavy worker at the lowest OS priority (unless the
        # user opted out via --normal-priority).  Guard on parent_process()
        # so an in-process call (e.g. tests) never nices the caller.
        from .priority import low_priority_enabled, set_low_priority
        if low_priority_enabled() and multiprocessing.parent_process() is not None:
            set_low_priority()

        from .model_cache import setup_model_environment
        setup_model_environment()

        state: dict = {"job_id": None}
        watchdog = threading.Thread(
            target=_watch_cancellation,
            args=(cancel_upto, state, stop_watchdog),
            name="cancel-watchdog",
            daemon=True,
        )
        watchdog.start()

        pool = ModelPool()
        _safe_put(result_queue, {"type": MSG_READY})

        while True:
            try:
                job = job_queue.get()
            except (EOFError, OSError, _queue_mod.Empty):
                log.info("Job queue closed; worker exiting.")
                break
            if job is None or job.get("type") == JOB_SHUTDOWN:
                log.info("Shutdown requested; worker exiting.")
                break
            if job.get("type") != JOB_ANALYZE:
                log.warning("Unknown job type: %r", job.get("type"))
                continue
            _run_job(job, result_queue, pool, cancel_upto, state)

    except Exception:
        # A failure out here (logging/env setup) is fatal for the worker.
        log.exception("Analysis worker crashed (PID %d).", os.getpid())
        _safe_put(result_queue, {
            "type": MSG_ERROR,
            "msg": "Analysis worker crashed during start-up.",
            "traceback": traceback.format_exc(),
        })
    finally:
        stop_watchdog.set()
        if watchdog is not None:
            watchdog.join(timeout=1)

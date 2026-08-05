"""
Cooperative cancellation for the analysis worker.

The analysis process is long-lived and keeps its models loaded between files,
so a running analysis can no longer be stopped by killing the process. Instead
the worker sets a process-wide token and the long-running loops check it at
their natural boundaries.

The token is process-global rather than passed down the call stack because the
worker runs exactly one analysis at a time, and the alternative would mean
threading a parameter through every layer of the pipeline, STT and event
detection call chains.

Typical use::

    cancellation.begin_job()          # worker, before each analysis
    ...
    cancellation.raise_if_cancelled() # inside long loops
    ...
    cancellation.request_cancel()     # watchdog thread, when the GUI asks
"""

from __future__ import annotations

import logging
import threading

log = logging.getLogger(__name__)

_cancelled = threading.Event()


class AnalysisCancelled(Exception):
    """Raised at a checkpoint when the current analysis has been cancelled."""


def begin_job() -> None:
    """Clear the token at the start of a new job."""
    _cancelled.clear()


def request_cancel() -> None:
    """Ask the current analysis to stop at its next checkpoint."""
    if not _cancelled.is_set():
        log.info("Cancellation requested for the running analysis.")
    _cancelled.set()


def is_cancelled() -> bool:
    """Return True if cancellation has been requested."""
    return _cancelled.is_set()


def raise_if_cancelled() -> None:
    """Abort the current analysis if cancellation has been requested.

    Raises:
        AnalysisCancelled: If :func:`request_cancel` was called.
    """
    if _cancelled.is_set():
        raise AnalysisCancelled("Analysis cancelled by the user.")

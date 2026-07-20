"""Process scheduling-priority helpers.

Used to run the CPU-heavy analysis worker subprocess at the lowest OS
scheduling priority so the machine stays responsive during long
transcriptions.  The main GUI process is left at normal priority.
"""

from __future__ import annotations

import logging
import os
import sys

log = logging.getLogger(__name__)

# Environment variable that carries the user's preference from the GUI
# process to the spawned worker subprocess (inherited automatically).
_ENV_FLAG = "MONITOR_LOW_PRIORITY"


def set_low_priority() -> None:
    """Best-effort: set the CURRENT process to the lowest OS priority.

    Never raises — priority tuning is a nice-to-have, not a requirement.
    Lowering priority never needs elevation on any supported platform.
    """
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            IDLE_PRIORITY_CLASS = 0x00000040
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
            k32.SetPriorityClass.restype = wintypes.BOOL
            if not k32.SetPriorityClass(k32.GetCurrentProcess(), IDLE_PRIORITY_CLASS):
                log.warning("SetPriorityClass(IDLE) failed (err=%d).",
                            ctypes.get_last_error())
        else:
            os.nice(19)  # POSIX: highest niceness = lowest priority
    except Exception:
        log.warning("Could not lower process priority.", exc_info=True)


def low_priority_enabled() -> bool:
    """Return True if low-priority mode is enabled (default: enabled)."""
    return os.environ.get(_ENV_FLAG, "1") != "0"


def set_low_priority_env(enabled: bool) -> None:
    """Record the low-priority preference so spawned children inherit it."""
    os.environ[_ENV_FLAG] = "1" if enabled else "0"

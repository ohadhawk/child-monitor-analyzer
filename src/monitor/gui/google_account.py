"""
Google Drive account UI: the toolbar chip, its menu, and the account dialog.

This module owns *all* threading for the Drive feature. Everything in
:mod:`monitor.gdrive` is blocking and headless; the workers here follow the
existing ``_ModelCheckWorker`` idiom in :mod:`monitor.gui.main_window`
(``moveToThread`` -> ``started``/``run`` -> ``finished``/``failed`` ->
``quit``/``deleteLater``, with strong references held by the caller).

The upload deliberately does **not** run in the analysis subprocess: that
process is long-lived but cancellable, and credentials must never cross the
process boundary.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .strings import tr, S

log = logging.getLogger(__name__)

#: Bidi isolates. Email addresses and URLs inside right-to-left Hebrew text
#: render scrambled without them.
_LRI = "\u2066"
_PDI = "\u2069"


def isolate(text: str) -> str:
    """Wrap left-to-right text so it renders correctly inside Hebrew."""
    return f"{_LRI}{text}{_PDI}" if text else text


# ===========================
# WORKERS
# ===========================

def _describe(exc: Exception) -> str:
    """Turn a Drive/auth exception into something a parent can act on.

    Raw ``URLError`` text ("[Errno 11001] getaddrinfo failed") tells a
    non-technical user nothing, so the cases we can recognise are mapped to
    plain Hebrew. Anything unrecognised falls through to its own message,
    which is better than a generic dead end.
    """
    from ..gdrive import auth as _auth, client as _client

    if isinstance(exc, _client.DriveOffline):
        return tr(S.UPLOAD_NO_NETWORK)
    if isinstance(exc, _auth.InvalidGrant):
        return tr(S.GOOGLE_SESSION_EXPIRED)
    if isinstance(exc, _auth.MissingScope):
        return tr(S.GOOGLE_MISSING_SCOPE)
    if isinstance(exc, _auth.AuthDenied):
        return tr(S.GOOGLE_AUTH_DENIED)
    if isinstance(exc, _auth.AuthTimeout):
        return tr(S.GOOGLE_AUTH_TIMEOUT)
    if isinstance(exc, _auth.AuthCancelled):
        # Empty message = the user did this on purpose; say nothing.
        return ""
    return str(exc)


class LinkWorker(QObject):
    """Runs the blocking OAuth link flow off the GUI thread."""

    finished = Signal(object)   # GoogleCredentials
    failed = Signal(str)

    def __init__(self, session) -> None:
        super().__init__()
        self._session = session
        self._cancel = threading.Event()

    def cancel(self) -> None:
        """Abandon the sign-in. Safe to call from the GUI thread."""
        self._cancel.set()

    @Slot()
    def run(self) -> None:
        try:
            credentials = self._session.link(cancel_event=self._cancel)
        except Exception as exc:  # noqa: BLE001 - reported to the user
            message = _describe(exc)
            # An empty description means the user cancelled on purpose; that is
            # a normal outcome, not a fault worth a WARNING in the log.
            log.log(
                logging.INFO if not message else logging.WARNING,
                "Google link flow ended: %s", exc,
            )
            self.failed.emit(message)
            return
        self.finished.emit(credentials)


class UnlinkWorker(QObject):
    """Runs revoke-then-purge off the GUI thread."""

    finished = Signal(object)   # bool: server-side revocation confirmed
    failed = Signal(str)

    def __init__(self, session) -> None:
        super().__init__()
        self._session = session

    @Slot()
    def run(self) -> None:
        try:
            revoked = self._session.unlink()
        except Exception as exc:  # noqa: BLE001
            log.warning("Google unlink failed: %s", exc)
            self.failed.emit(_describe(exc))
            return
        self.finished.emit(bool(revoked))


class UploadWorker(QObject):
    """Runs a single transcript upload off the GUI thread."""

    finished = Signal(object)   # UploadResult
    failed = Signal(str)

    def __init__(self, session, *, content: bytes, content_type: str,
                 display_name: str, artifact_dir: Optional[Path],
                 model_key: str, replace_existing: bool) -> None:
        super().__init__()
        self._session = session
        self._content = content
        self._content_type = content_type
        self._display_name = display_name
        self._artifact_dir = artifact_dir
        self._model_key = model_key
        self._replace_existing = replace_existing

    @Slot()
    def run(self) -> None:
        try:
            result = self._session.upload_transcript(
                content=self._content,
                content_type=self._content_type,
                display_name=self._display_name,
                artifact_dir=self._artifact_dir,
                model_key=self._model_key,
                replace_existing=self._replace_existing,
            )
        except Exception as exc:  # noqa: BLE001 - an upload failure must never
            # affect the analysis or the local artifacts.
            log.warning("Drive upload failed: %s", exc)
            self.failed.emit(_describe(exc))
            return
        self.finished.emit(result)


# ===========================
# DIALOG
# ===========================

class GoogleAccountDialog(QDialog):
    """Explains the requested access and offers link / unlink.

    The dialog only reports intent through :attr:`link_requested` and
    :attr:`unlink_requested`; the main window owns the worker threads.
    """

    link_requested = Signal()
    unlink_requested = Signal()

    def __init__(self, session, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._session = session
        self.setWindowTitle(tr(S.GOOGLE_ACCOUNT_TITLE))
        self.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        self.setMinimumWidth(420)

        layout = QVBoxLayout(self)

        linked = session.is_linked()

        status = QLabel(
            f"{tr(S.GOOGLE_CONNECTED_AS)}{isolate(session.email)}"
            if linked else tr(S.GOOGLE_NOT_CONNECTED)
        )
        status.setStyleSheet("font-weight: bold;")
        status.setWordWrap(True)
        layout.addWidget(status)

        # Plain-language, no jargon: the user must understand the blast radius
        # before granting anything.
        explanation = QLabel(tr(S.GOOGLE_SCOPE_EXPLANATION))
        explanation.setWordWrap(True)
        explanation.setStyleSheet("color: #444; padding: 6px 0;")
        layout.addWidget(explanation)

        if linked:
            folder_link = session.folder_link()
            if folder_link:
                link_label = QLabel(
                    f'<a href="{folder_link}">{tr(S.GOOGLE_MENU_OPEN_FOLDER)}</a>'
                )
                link_label.setOpenExternalLinks(True)
                layout.addWidget(link_label)

            self._chk_ask = QCheckBox(tr(S.GOOGLE_MENU_ASK_EVERY_TIME))
            self._chk_ask.setChecked(session.ask_every_time())
            self._chk_ask.toggled.connect(session.set_ask_every_time)
            layout.addWidget(self._chk_ask)
        else:
            hint = QLabel(tr(S.GOOGLE_BROWSER_HINT))
            hint.setWordWrap(True)
            hint.setStyleSheet("color: #666;")
            layout.addWidget(hint)

        button_row = QHBoxLayout()
        if linked:
            btn_unlink = QPushButton(tr(S.GOOGLE_SIGN_OUT))
            btn_unlink.clicked.connect(self._request_unlink)
            button_row.addWidget(btn_unlink)
        else:
            btn_link = QPushButton(tr(S.GOOGLE_SIGN_IN))
            btn_link.setDefault(True)
            btn_link.clicked.connect(self._request_link)
            button_row.addWidget(btn_link)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.StandardButton.Close).clicked.connect(self.reject)
        layout.addWidget(buttons)

    def _request_link(self) -> None:
        self.link_requested.emit()
        self.accept()

    def _request_unlink(self) -> None:
        self.unlink_requested.emit()
        self.accept()

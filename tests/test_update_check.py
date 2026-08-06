"""
Regression tests for the application update check.

The feature deliberately never downloads a payload, so these tests police the
two things that can actually hurt: the URL that reaches the browser, and the
comparison that decides whether to interrupt the user.
"""

from __future__ import annotations

import ast
import gc
import io
import json
import urllib.error
from pathlib import Path

import pytest

from monitor import update_check


# ===========================
# VERSION COMPARISON
# ===========================


@pytest.mark.parametrize("text,expected", [
    ("1.2.0", (1, 2, 0)),
    ("v1.2.0", (1, 2, 0)),
    ("  v2.0  ", (2, 0)),
    ("v1.2.3-beta.1", (1, 2, 3)),
    ("10.0.1", (10, 0, 1)),
    ("release", None),
    ("", None),
    ("v", None),
    (None, None),
    (1.2, None),
])
def test_version_parsing(text, expected):
    assert update_check.parse_version(text) == expected


@pytest.mark.parametrize("candidate,current,expected", [
    ("1.2.1", "1.2.0", True),
    ("v1.3.0", "1.2.9", True),
    ("2.0", "1.9.9", True),
    ("1.10.0", "1.9.0", True),          # Numeric, not lexicographic.
    ("1.2.0", "1.2.0", False),
    ("1.2", "1.2.0", False),            # Padded, so equal rather than shorter.
    ("1.2.0", "1.2", False),
    ("1.1.9", "1.2.0", False),
    ("garbage", "1.2.0", False),
    ("", "1.2.0", False),
    ("1.2.1", "garbage", False),
])
def test_only_a_strictly_greater_version_counts(candidate, current, expected):
    assert update_check.is_newer(candidate, current) is expected


def test_a_downgrade_is_never_advertised():
    """A rolled-back or forged 'latest' must not push the user backwards."""
    assert update_check.is_newer("0.1.0", update_check.__version__) is False


# ===========================
# URL SAFETY
# ===========================


@pytest.mark.parametrize("url", [
    "http://github.com/x/y/releases",       # Not https.
    "file:///C:/Windows/System32/calc.exe",  # ShellExecute would run this.
    "\\\\attacker\\share",                   # UNC: leaks the NTLM hash.
    "https://github.com.evil.test/x",        # Suffix confusion.
    "https://notgithub.com/x",
    "javascript:alert(1)",
    "",
    None,
    12345,
])
def test_unsafe_urls_are_rejected(url):
    assert update_check.validate_release_url(url) is None


@pytest.mark.parametrize("url", [
    "https://github.com/ohadhawk/child-monitor-analyzer/releases",
    "https://www.github.com/x",
])
def test_github_https_urls_are_accepted(url):
    assert update_check.validate_release_url(url) == url


def test_the_releases_url_constant_passes_its_own_validator():
    assert update_check.releases_url() == update_check.RELEASES_PAGE_URL


def test_the_releases_page_is_never_taken_from_the_response():
    """The page URL must be a constant, not a field of the API payload.

    ``QDesktopServices.openUrl`` reaches ShellExecute on Windows, so a
    response-supplied URL would be a remote-controlled launcher.
    """
    source = Path(update_check.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if getattr(target, "id", None) == "RELEASES_PAGE_URL":
                    # A constant or an f-string of constants -- never a lookup.
                    assert isinstance(node.value, (ast.Constant, ast.JoinedStr))


def test_the_gui_opens_only_the_validated_constant():
    """The GUI must pass ``releases_url()`` to the browser, nothing else."""
    main_window = Path(update_check.__file__).with_name("gui") / "main_window.py"
    tree = ast.parse(main_window.read_text(encoding="utf-8"))

    opened = [
        ast.unparse(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func).endswith("openUrl")
    ]
    assert any("releases_url()" in call for call in opened), opened
    for call in opened:
        if "releases_url" in call:
            assert call == "QDesktopServices.openUrl(QUrl(releases_url()))"


# ===========================
# THROTTLING
# ===========================


@pytest.mark.parametrize("last,now,expected", [
    (0.0, 1_000_000.0, True),                       # Never checked.
    (1_000_000.0, 1_000_000.0, False),              # Just checked.
    (1_000_000.0, 1_000_000.0 + 3600, False),       # An hour later.
    (1_000_000.0, 1_000_000.0 + 86_400, True),      # Exactly a day.
    (1_000_000.0, 1_000_000.0 + 90_000, True),
    (2_000_000.0, 1_000_000.0, True),               # Clock moved backwards.
])
def test_the_daily_throttle(last, now, expected):
    assert update_check.is_due(last, now) is expected


# ===========================
# TRANSPORT
# ===========================


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _patch_opener(monkeypatch, payload: bytes):
    class _Opener:
        def open(self, request, timeout=None):
            _Opener.request = request
            return _FakeResponse(payload)

    monkeypatch.setattr(update_check, "_opener", _Opener)
    return _Opener


def test_a_newer_tag_is_reported(monkeypatch):
    _patch_opener(monkeypatch, json.dumps({"tag_name": "v99.0.0"}).encode())
    assert update_check.check_for_update("1.2.0") == "99.0.0"


def test_a_hostile_tag_cannot_smuggle_text_into_the_dialog(monkeypatch):
    """Only the parsed numbers are returned, never the raw tag.

    ``tag_name`` is matched by prefix, so trailing junk -- bidi overrides, a
    fake instruction, or a megabyte of padding -- would otherwise be shown.
    """
    tag = "v99.0.0" + "\u202e evil " + "x" * 5000
    _patch_opener(monkeypatch, json.dumps({"tag_name": tag}).encode())
    assert update_check.check_for_update("1.2.0") == "99.0.0"


def test_the_same_version_reports_nothing(monkeypatch):
    _patch_opener(monkeypatch, json.dumps({"tag_name": "v1.2.0"}).encode())
    assert update_check.check_for_update("1.2.0") is None


def test_drafts_and_prereleases_are_ignored(monkeypatch):
    _patch_opener(
        monkeypatch,
        json.dumps({"tag_name": "v99.0.0", "prerelease": True}).encode(),
    )
    assert update_check.check_for_update("1.2.0") is None


def test_an_oversized_response_is_refused(monkeypatch):
    _patch_opener(monkeypatch, b"x" * (update_check.MAX_RESPONSE_BYTES + 10))
    with pytest.raises(update_check.UpdateCheckError):
        update_check.fetch_latest_version()


def test_malformed_json_is_refused(monkeypatch):
    _patch_opener(monkeypatch, b"{not json")
    with pytest.raises(update_check.UpdateCheckError):
        update_check.fetch_latest_version()


def test_a_json_array_is_refused(monkeypatch):
    """A list has no ``.get``; without the shape check this would crash."""
    _patch_opener(monkeypatch, b"[]")
    with pytest.raises(update_check.UpdateCheckError):
        update_check.fetch_latest_version()


def test_a_missing_release_is_not_an_error(monkeypatch):
    class _Opener:
        def open(self, request, timeout=None):
            raise urllib.error.HTTPError(
                update_check.LATEST_RELEASE_API_URL, 404, "Not Found", {}, None,
            )

    monkeypatch.setattr(update_check, "_opener", _Opener)
    assert update_check.check_for_update("1.2.0") is None


def _patch_http_error(monkeypatch, code, headers):
    class _Opener:
        def open(self, request, timeout=None):
            raise urllib.error.HTTPError(
                update_check.LATEST_RELEASE_API_URL, code, "Err", headers, None,
            )

    monkeypatch.setattr(update_check, "_opener", _Opener)


@pytest.mark.parametrize("code", [403, 429])
def test_rate_limiting_is_reported_separately(monkeypatch, code):
    """Observed live: a shared IP can exhaust GitHub's 60/hour anonymous quota."""
    _patch_http_error(monkeypatch, code, {"x-ratelimit-remaining": "0"})
    with pytest.raises(update_check.UpdateRateLimited):
        update_check.fetch_latest_version()


def test_a_plain_403_is_not_mistaken_for_rate_limiting(monkeypatch):
    _patch_http_error(monkeypatch, 403, {"x-ratelimit-remaining": "42"})
    with pytest.raises(update_check.UpdateCheckError) as caught:
        update_check.fetch_latest_version()
    assert not isinstance(caught.value, update_check.UpdateRateLimited)


def test_rate_limiting_is_still_a_check_error(monkeypatch):
    """Callers that only know the base class must keep working."""
    _patch_http_error(monkeypatch, 403, {"x-ratelimit-remaining": "0"})
    with pytest.raises(update_check.UpdateCheckError):
        update_check.check_for_update("1.2.0")


def test_a_network_failure_raises_our_own_error(monkeypatch):
    class _Opener:
        def open(self, request, timeout=None):
            raise urllib.error.URLError("no route to host")

    monkeypatch.setattr(update_check, "_opener", _Opener)
    with pytest.raises(update_check.UpdateCheckError):
        update_check.check_for_update("1.2.0")


def test_redirects_are_refused():
    handler = update_check._NoRedirectHandler()
    request = type("R", (), {"full_url": update_check.LATEST_RELEASE_API_URL})()
    with pytest.raises(update_check.UpdateCheckError):
        handler.redirect_request(request, None, 302, "Found", {}, "https://evil.test")


def test_the_api_url_is_https_and_pinned_to_github():
    assert update_check.LATEST_RELEASE_API_URL.startswith(
        "https://api.github.com/repos/"
    )


def test_the_user_agent_carries_no_machine_identity(monkeypatch):
    opener = _patch_opener(monkeypatch, json.dumps({"tag_name": "v1.0.0"}).encode())
    update_check.fetch_latest_version()
    agent = opener.request.get_header("User-agent")
    assert agent == f"child-monitor-analyzer/{update_check.__version__}"


def test_no_credentials_or_cookies_are_sent(monkeypatch):
    opener = _patch_opener(monkeypatch, json.dumps({"tag_name": "v1.0.0"}).encode())
    update_check.fetch_latest_version()
    headers = {k.lower() for k in opener.request.headers}
    assert not headers & {"authorization", "cookie"}


# ===========================
# GUI WIRING
# ===========================


@pytest.fixture
def window(monkeypatch):
    """A MainWindow using a throwaway QSettings scope.

    Without the scope override this would read and write the developer's real
    preferences, including the update-consent answer.
    """
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from monitor.gui import main_window as mw

    monkeypatch.setattr(mw, "SETTINGS_ORG", "ChildMonitorAnalyzerTests")
    monkeypatch.setattr(mw, "SETTINGS_APP", f"update-{id(monkeypatch)}")
    # The startup prompt is modal; nothing may schedule it during the test.
    monkeypatch.setattr(mw.QTimer, "singleShot", lambda *a, **k: None)

    app = QApplication.instance() or QApplication([])
    # Retire the previous window's C++ objects before allocating new ones:
    # PySide6 otherwise hands back an invalidated wrapper for a reused address.
    gc.collect()
    app.processEvents()
    win = mw.MainWindow()
    try:
        yield win
    finally:
        for key in (mw.SETTINGS_UPDATE_FREQUENCY_KEY,
                    mw.SETTINGS_UPDATE_LAST_CHECKED_KEY,
                    mw.SETTINGS_UPDATE_LAST_NOTIFIED_KEY,
                    mw.SETTINGS_MODEL_FREQUENCY_KEY,
                    mw.SETTINGS_MODEL_LAST_CHECKED_KEY):
            win._settings.remove(key)
        win.close()
        # No deleteLater: destroying one window invalidates the wrappers a
        # later window hands back, which breaks the following test.
        app.processEvents()


def _updates_menu(window):
    """Refresh the menu state and return its actions."""
    window._refresh_updates_menu()
    return window._updates_menu.actions()


def test_the_updates_menu_exposes_all_four_entries(window):
    from monitor.gui.strings import S, tr

    labels = [a.text() for a in _updates_menu(window)]
    assert tr(S.CHECK_MODELS) in labels
    assert tr(S.MODELS_CHECK_AUTOMATICALLY) in labels
    assert tr(S.UPDATE_MENU_CHECK_NOW) in labels
    assert tr(S.UPDATE_MENU_CHECK_AUTOMATICALLY) in labels


def test_a_submenu_outlives_its_python_wrapper(window):
    """The submenus must be owned by Qt, not by a Python wrapper.

    ``QMenu.addMenu(title)`` leaves the new menu owned by its menu-action's
    wrapper, so the entry silently stops opening once that wrapper is
    collected. Anything that walks ``actions()`` is enough to trigger it.
    """
    from monitor.gui.strings import S, tr

    title = tr(S.UPDATE_MENU_CHECK_AUTOMATICALLY)
    submenu = next(a.menu() for a in _updates_menu(window) if a.text() == title)
    gc.collect()  # Retires the temporary QAction wrappers made just above.
    assert len(submenu.actions()) == len(update_check.CHECK_INTERVALS)


def test_reopening_the_menu_does_not_grow_it(window):
    """A rebuild-on-open would leak a QMenu per automatic-check entry."""
    from PySide6.QtWidgets import QMenu

    before = len(window._updates_menu.findChildren(QMenu))
    for _ in range(5):
        window._updates_menu.aboutToShow.emit()
    assert len(window._updates_menu.findChildren(QMenu)) == before
    assert len(window._updates_menu.actions()) == 5  # 4 entries + separator


def test_there_is_no_menu_bar(window):
    """The controls live on the toolbar button, so the window has no menus."""
    assert window.menuBar().actions() == []


@pytest.mark.parametrize("title_key,settings_key", [
    ("MODELS_CHECK_AUTOMATICALLY", "SETTINGS_MODEL_FREQUENCY_KEY"),
    ("UPDATE_MENU_CHECK_AUTOMATICALLY", "SETTINGS_UPDATE_FREQUENCY_KEY"),
])
def test_each_automatic_entry_offers_the_four_frequencies(
        window, title_key, settings_key):
    from monitor.gui.strings import S, tr

    title = tr(getattr(S, title_key))
    submenu = next(a.menu() for a in _updates_menu(window) if a.text() == title)
    labels = [a.text() for a in submenu.actions()]
    assert labels == [
        tr(S.FREQ_NEVER), tr(S.FREQ_DAILY), tr(S.FREQ_WEEKLY), tr(S.FREQ_MONTHLY),
    ]
    assert all(a.isCheckable() for a in submenu.actions())
    # Exactly one tick, or the user cannot tell what is in force.
    assert sum(a.isChecked() for a in submenu.actions()) == 1


def test_choosing_a_frequency_stores_it(window):
    from monitor.gui import main_window as mw
    from monitor.gui.strings import S, tr

    title = tr(S.UPDATE_MENU_CHECK_AUTOMATICALLY)
    submenu = next(a.menu() for a in _updates_menu(window) if a.text() == title)
    next(a for a in submenu.actions() if a.text() == tr(S.FREQ_WEEKLY)).trigger()

    assert window._check_frequency(mw.SETTINGS_UPDATE_FREQUENCY_KEY) == "weekly"
    # And the tick follows on the next open.
    submenu = next(a.menu() for a in _updates_menu(window) if a.text() == title)
    checked = [a.text() for a in submenu.actions() if a.isChecked()]
    assert checked == [tr(S.FREQ_WEEKLY)]


def test_both_automatic_checks_default_to_never(window):
    from monitor.gui import main_window as mw

    for key in (mw.SETTINGS_MODEL_FREQUENCY_KEY, mw.SETTINGS_UPDATE_FREQUENCY_KEY):
        window._settings.remove(key)
        assert window._check_frequency(key) is None
        assert window._is_check_due(None, mw.SETTINGS_UPDATE_LAST_CHECKED_KEY) is False


def test_model_checks_are_never_prompted_for(window, monkeypatch):
    """Only the application check may interrupt a first run."""
    from monitor.gui import main_window as mw

    monkeypatch.setattr(
        window, "_ask_update_consent",
        lambda: pytest.fail("the model check must not prompt"),
    )
    monkeypatch.setattr(
        window, "_check_new_models",
        lambda **k: pytest.fail("no model check is due"),
    )
    window._settings.remove(mw.SETTINGS_MODEL_FREQUENCY_KEY)
    window._maybe_check_for_model_updates()


def test_a_silent_model_check_records_the_time_and_says_nothing(window, monkeypatch):
    """An automatic model check only speaks up when there is something new."""
    from monitor.gui import main_window as mw
    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setattr(
        QMessageBox, "information",
        lambda *a, **k: pytest.fail("a silent check must not open a dialog"),
    )
    window._model_announce = False
    window._settings.setValue(mw.SETTINGS_MODEL_LAST_CHECKED_KEY, 0.0)
    window._lbl_status.setText("analysing")
    window._lbl_status.setVisible(True)

    window._on_model_check_finished([])

    assert float(window._settings.value(mw.SETTINGS_MODEL_LAST_CHECKED_KEY)) > 0
    # isHidden, not isVisible: the window itself is never shown in tests.
    assert not window._lbl_status.isHidden()
    assert window._lbl_status.text() == "analysing"


def test_a_silent_model_check_still_reports_a_find(window, monkeypatch):
    """Finding a new model is the one thing worth interrupting for."""
    from PySide6.QtWidgets import QMessageBox

    shown = []
    monkeypatch.setattr(
        QMessageBox, "information", lambda *a, **k: shown.append(a[2]),
    )
    window._model_announce = False
    window._on_model_check_finished([{"id": "some/model", "created": "2026-01-01"}])

    assert shown and "some/model" in shown[0]


def test_an_unrecognised_frequency_means_never(window):
    """A hand-edited setting must not be able to cause more traffic."""
    from monitor.gui import main_window as mw

    window._settings.setValue(mw.SETTINGS_UPDATE_FREQUENCY_KEY, "hourly")
    window._settings.setValue(mw.SETTINGS_UPDATE_LAST_CHECKED_KEY, 0.0)
    assert window._is_check_due("hourly", mw.SETTINGS_UPDATE_LAST_CHECKED_KEY) is False


@pytest.mark.parametrize("frequency,elapsed,expected", [
    ("daily", 3600, False),
    ("daily", 90_000, True),
    ("weekly", 5 * 86_400, False),
    ("weekly", 8 * 86_400, True),
    ("monthly", 20 * 86_400, False),
    ("monthly", 31 * 86_400, True),
    ("never", 10_000 * 86_400, False),
])
def test_each_frequency_throttles_as_named(window, frequency, elapsed, expected):
    import time as _time

    from monitor.gui import main_window as mw

    window._settings.setValue(
        mw.SETTINGS_UPDATE_LAST_CHECKED_KEY, _time.time() - elapsed,
    )
    due = window._is_check_due(frequency, mw.SETTINGS_UPDATE_LAST_CHECKED_KEY)
    assert due is expected


def test_declining_consent_stops_the_check(window, monkeypatch):
    from monitor.gui import main_window as mw

    called = []
    monkeypatch.setattr(window, "_ask_update_consent", lambda: "never")
    monkeypatch.setattr(window, "_start_update_check", lambda **k: called.append(k))
    window._settings.remove(mw.SETTINGS_UPDATE_FREQUENCY_KEY)
    window._maybe_check_for_updates()
    assert called == []


def test_a_second_check_within_the_interval_makes_no_request(window, monkeypatch):
    import time as _time

    from monitor.gui import main_window as mw

    called = []
    monkeypatch.setattr(window, "_start_update_check", lambda **k: called.append(k))
    window._settings.setValue(mw.SETTINGS_UPDATE_FREQUENCY_KEY, "daily")

    window._settings.setValue(mw.SETTINGS_UPDATE_LAST_CHECKED_KEY, 0.0)
    window._maybe_check_for_updates()
    assert len(called) == 1

    window._settings.setValue(mw.SETTINGS_UPDATE_LAST_CHECKED_KEY, _time.time())
    window._maybe_check_for_updates()
    assert len(called) == 1


def test_an_automatic_check_stays_silent_when_up_to_date(window, monkeypatch):
    """A background check must never pop a dialog saying nothing happened."""
    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setattr(
        QMessageBox, "information",
        lambda *a, **k: pytest.fail("an automatic check showed a dialog"),
    )
    monkeypatch.setattr(
        QMessageBox, "warning",
        lambda *a, **k: pytest.fail("an automatic check showed a dialog"),
    )
    window._update_announce = False
    window._on_update_check_finished("")
    window._on_update_check_failed("no route to host")


def test_a_silent_check_leaves_the_status_label_alone(window):
    """A background check must not clear another operation's status message."""
    window._lbl_status.setText("analysing")
    window._lbl_status.setVisible(True)
    window._update_announce = False

    # isHidden, not isVisible: the window itself is never shown in tests.
    window._on_update_check_finished("")
    assert not window._lbl_status.isHidden()
    assert window._lbl_status.text() == "analysing"

    window._on_update_check_failed("no route to host")
    assert not window._lbl_status.isHidden()
    assert window._lbl_status.text() == "analysing"


def test_a_successful_check_records_the_time(window):
    window._update_announce = False
    window._settings.setValue("update_last_checked", 0.0)
    window._on_update_check_finished("")
    assert float(window._settings.value("update_last_checked")) > 0


def _answer_modal(app, role, seen, timeout_ms=5000):
    """Click the button with *role* on the next modal dialog.

    ``exec()`` spins a nested event loop, so the dialog can only be driven from
    a timer. The deadline closes whatever is open rather than hanging the run.
    """
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QMessageBox

    timer = QTimer()
    elapsed = {"ms": 0}

    def tick():
        elapsed["ms"] += 50
        modal = app.activeModalWidget()
        if isinstance(modal, QMessageBox):
            seen["title"] = modal.windowTitle()
            seen["text"] = modal.text()
            seen["buttons"] = [b.text() for b in modal.buttons()]
            for button in modal.buttons():
                if modal.buttonRole(button) == role:
                    timer.stop()
                    button.click()
                    return
        if elapsed["ms"] >= timeout_ms:
            timer.stop()
            if modal is not None:
                modal.close()

    timer.timeout.connect(tick)
    timer.start(50)
    return timer


def test_the_consent_prompt_is_shown_once_and_remembered(window, monkeypatch):
    """End-to-end through the real modal: declining must stick.

    Covers the whole first-run path, which the unit tests stub out.
    """
    from PySide6.QtWidgets import QApplication, QMessageBox

    from monitor.gui import main_window as mw
    from monitor.gui.strings import S, tr

    started = []
    monkeypatch.setattr(window, "_start_update_check", lambda **k: started.append(k))
    window._settings.remove(mw.SETTINGS_UPDATE_FREQUENCY_KEY)

    app = QApplication.instance()
    seen = {}
    _answer_modal(app, QMessageBox.ButtonRole.NoRole, seen)
    window._maybe_check_for_updates()

    assert seen.get("title") == tr(S.UPDATE_TITLE)
    assert tr(S.UPDATE_CONSENT_NO) in seen.get("buttons", [])
    assert window._check_frequency(mw.SETTINGS_UPDATE_FREQUENCY_KEY) == "never"
    assert started == [], "declining consent must not reach the network"

    # Asked once only: a second startup must go straight through.
    seen.clear()
    window._maybe_check_for_updates()
    assert seen == {}


def test_accepting_consent_starts_a_check(window, monkeypatch):
    from PySide6.QtWidgets import QApplication, QMessageBox

    from monitor.gui import main_window as mw

    started = []
    monkeypatch.setattr(window, "_start_update_check", lambda **k: started.append(k))
    window._settings.remove(mw.SETTINGS_UPDATE_FREQUENCY_KEY)
    window._settings.setValue(mw.SETTINGS_UPDATE_LAST_CHECKED_KEY, 0.0)

    _answer_modal(QApplication.instance(), QMessageBox.ButtonRole.YesRole, {})
    window._maybe_check_for_updates()

    # Yes means daily; the menu is where a different interval is chosen.
    assert window._check_frequency(mw.SETTINGS_UPDATE_FREQUENCY_KEY) == "daily"
    # Automatic checks must not announce; only a menu-driven one does.
    assert started == [{"announce": False}]

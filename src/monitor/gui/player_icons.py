"""
Programmatically drawn player control icons.

All icons use the same geometric style, stroke width, colour and size
so they look consistent across play, pause, skip-back, skip-forward
and volume controls.

Usage:
    from monitor.gui.player_icons import icon_play, icon_pause, ...
    button.setIcon(icon_play())
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from PySide6.QtCore import QByteArray, QPointF, QRectF, Qt
from PySide6.QtGui import (
    QColor,
    QIcon,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
)
from PySide6.QtSvg import QSvgRenderer

log = logging.getLogger(__name__)

_SIZE = 32  # px – icon canvas size
_COLOR = QColor(50, 50, 50)
_PEN_WIDTH = 2.4


def _new_pixmap() -> tuple[QPixmap, QPainter]:
    pm = QPixmap(_SIZE, _SIZE)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(QPen(_COLOR, _PEN_WIDTH, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    p.setBrush(_COLOR)
    return pm, p


def _finish(pm: QPixmap, p: QPainter) -> QIcon:
    p.end()
    return QIcon(pm)


# ── Play: right-pointing triangle ──────────────────────────────
def icon_play() -> QIcon:
    pm, p = _new_pixmap()
    tri = QPolygonF([
        QPointF(10, 6),
        QPointF(26, 16),
        QPointF(10, 26),
    ])
    p.drawPolygon(tri)
    return _finish(pm, p)


# ── Pause: two vertical bars ──────────────────────────────────
def icon_pause() -> QIcon:
    pm, p = _new_pixmap()
    bar_w = 4.0
    p.drawRoundedRect(QRectF(9, 7, bar_w, 18), 1, 1)
    p.drawRoundedRect(QRectF(19, 7, bar_w, 18), 1, 1)
    return _finish(pm, p)


# ── Skip back: left triangle + vertical bar ───────────────────
def icon_skip_back() -> QIcon:
    pm, p = _new_pixmap()
    # Vertical bar on the left.
    p.drawRoundedRect(QRectF(6, 8, 3, 16), 1, 1)
    # Left-pointing triangle.
    tri = QPolygonF([
        QPointF(26, 8),
        QPointF(12, 16),
        QPointF(26, 24),
    ])
    p.drawPolygon(tri)
    return _finish(pm, p)


# ── Skip forward: vertical bar + right triangle ───────────────
def icon_skip_forward() -> QIcon:
    pm, p = _new_pixmap()
    # Right-pointing triangle.
    tri = QPolygonF([
        QPointF(6, 8),
        QPointF(20, 16),
        QPointF(6, 24),
    ])
    p.drawPolygon(tri)
    # Vertical bar on the right.
    p.drawRoundedRect(QRectF(23, 8, 3, 16), 1, 1)
    return _finish(pm, p)


# ── Volume: speaker cone + waves ──────────────────────────────
def icon_volume() -> QIcon:
    pm, p = _new_pixmap()
    p.setPen(Qt.PenStyle.NoPen)
    # Speaker body (trapezoid-ish polygon).
    body = QPolygonF([
        QPointF(5, 13),
        QPointF(10, 13),
        QPointF(16, 7),
        QPointF(16, 25),
        QPointF(10, 19),
        QPointF(5, 19),
    ])
    p.drawPolygon(body)
    # Sound waves (arcs).
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.setPen(QPen(_COLOR, _PEN_WIDTH, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    path1 = QPainterPath()
    path1.arcMoveTo(QRectF(17, 10, 8, 12), -45)
    path1.arcTo(QRectF(17, 10, 8, 12), -45, 90)
    p.drawPath(path1)
    path2 = QPainterPath()
    path2.arcMoveTo(QRectF(20, 7, 12, 18), -45)
    path2.arcTo(QRectF(20, 7, 12, 18), -45, 90)
    p.drawPath(path2)
    return _finish(pm, p)


# ── Cloud upload: neutral stand-in for the signed-out state ────
#
# Google's guidelines forbid altering or recolouring their marks, and a
# greyed-out logo is exactly such an alteration. So the Drive mark below is
# used only while an account is connected, and this neutral glyph carries the
# meaning everywhere else. The word "Google" appears as plain text in the UI
# (nominative use).
_INACTIVE_COLOR = QColor(150, 150, 150)


def icon_cloud_upload() -> QIcon:
    """Return the neutral cloud used whenever no account is connected."""
    icon = QIcon()
    # Several sizes because a QIcon will not scale a pixmap up, and the chip
    # asks for one larger than the 32-unit design grid.
    for size in (16, 24, 32, 48, 64, 128):
        icon.addPixmap(_draw_cloud_upload(_INACTIVE_COLOR, size))
    return icon


def _draw_cloud_upload(colour: QColor, size: int) -> QPixmap:
    """Render the cloud-upload glyph at *size* px from the 32-unit grid."""
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.scale(size / _SIZE, size / _SIZE)

    # Drawn edge to edge, unlike the player glyphs: this one sits alone on a
    # chip rather than in a row, so the surrounding button supplies the margin.
    cloud = QPainterPath()
    # Winding, not the default odd-even: the lobes overlap, and odd-even would
    # punch the overlaps back out.
    cloud.setFillRule(Qt.FillRule.WindingFill)
    cloud.addEllipse(QRectF(0.0, 9.0, 16.0, 16.0))
    cloud.addEllipse(QRectF(5.0, 2.0, 22.0, 22.0))
    cloud.addEllipse(QRectF(16.0, 9.0, 16.0, 16.0))
    cloud.addRect(QRectF(8.0, 17.0, 16.0, 8.0))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(colour)
    p.drawPath(cloud.simplified())

    # Up arrow punched out of the cloud, kept clear of its edges so the
    # silhouette stays a cloud rather than breaking into fragments.
    arrow = QPolygonF([
        QPointF(16.0, 9.0),
        QPointF(21.5, 15.0),
        QPointF(18.5, 15.0),
        QPointF(18.5, 22.0),
        QPointF(13.5, 22.0),
        QPointF(13.5, 15.0),
        QPointF(10.5, 15.0),
    ])
    p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
    p.setBrush(Qt.GlobalColor.transparent)
    p.drawPolygon(arrow)

    p.end()
    return pm


# ── Google Drive: the official mark, used only for the linked state ────
#
# Shipped verbatim (see assets/NOTICE.txt for provenance) and never
# recoloured, greyed or overlaid: Google's brand guidelines forbid altering
# their marks, which is why the signed-out state uses the neutral cloud
# above instead of a muted Drive logo.
_ASSETS_DIR = (
    Path(sys._MEIPASS) / "monitor" / "gui" / "assets"
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")
    else Path(__file__).resolve().parent / "assets"
)
GOOGLE_DRIVE_SVG = _ASSETS_DIR / "google_drive.svg"


def icon_google_drive() -> QIcon:
    """Return the Google Drive mark, or an empty icon if it cannot be read."""
    try:
        markup = GOOGLE_DRIVE_SVG.read_bytes()
    except OSError:
        log.warning("The Google Drive mark is missing from %s.", _ASSETS_DIR)
        return QIcon()

    renderer = QSvgRenderer(QByteArray(_qt_maskable(markup)))
    if not renderer.isValid():
        log.warning("The Google Drive mark could not be parsed.")
        return QIcon()

    icon = QIcon()
    for size in (16, 24, 32, 48, 64, 128):
        pm = QPixmap(size, size)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        renderer.render(p)
        p.end()
        icon.addPixmap(pm)
        # Registered for Disabled too, so Qt cannot synthesise a greyed mark.
        icon.addPixmap(pm, QIcon.Mode.Disabled)
    return icon


def _qt_maskable(markup: bytes) -> bytes:
    """Make the artwork's alpha mask survive Qt's luminance-only masking.

    Qt treats every SVG mask as a luminance mask, so the file's dark mask fill
    dims the whole logo to about a third of its opacity. Substituting white
    restores the alpha the file asks for; the drawn result is the mark exactly
    as published, so nothing about its appearance is altered.
    """
    return markup.replace(b'fill="#b43333"', b'fill="#ffffff"')


def icon_upload_arrow() -> QIcon:
    """Return a plain upward arrow for the transcript toolbar."""
    pm, p = _new_pixmap()
    p.setPen(Qt.PenStyle.NoPen)
    p.drawPolygon(QPolygonF([
        QPointF(16.0, 6.0),
        QPointF(25.0, 16.0),
        QPointF(20.0, 16.0),
        QPointF(20.0, 26.0),
        QPointF(12.0, 26.0),
        QPointF(12.0, 16.0),
        QPointF(7.0, 16.0),
    ]))
    return _finish(pm, p)

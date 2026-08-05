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

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QColor,
    QIcon,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
)

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


# ── Cloud upload: neutral cloud with an up arrow ───────────────
#
# Deliberately NOT a Google "G" or the Drive triangle: Google's branding
# guidelines forbid altering or recolouring their marks, and a greyed-out
# logo is exactly such an alteration. A neutral glyph carries the same
# meaning with no trademark exposure. The word "Google" appears only as
# plain text elsewhere in the UI (nominative use).
_INACTIVE_COLOR = QColor(150, 150, 150)
_ACTIVE_COLOR = QColor(26, 115, 232)


def icon_cloud_upload(active: bool = False) -> QIcon:
    """Return the Drive account indicator glyph.

    Args:
        active: True when an account is linked (full colour); False draws the
            muted, signed-out state.
    """
    colour = _ACTIVE_COLOR if active else _INACTIVE_COLOR
    pm = QPixmap(_SIZE, _SIZE)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)

    # Cloud outline: three overlapping circles on a flat base.
    cloud = QPainterPath()
    cloud.addEllipse(QRectF(6.0, 12.0, 11.0, 11.0))
    cloud.addEllipse(QRectF(12.0, 8.0, 13.0, 13.0))
    cloud.addEllipse(QRectF(18.0, 13.0, 9.0, 9.0))
    cloud.addRect(QRectF(9.0, 17.0, 15.0, 6.0))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(colour)
    p.drawPath(cloud.simplified())

    # Up arrow punched through the cloud in the background colour.
    p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
    p.setBrush(Qt.GlobalColor.transparent)
    arrow = QPolygonF([
        QPointF(16.0, 10.0),
        QPointF(21.0, 16.0),
        QPointF(18.0, 16.0),
        QPointF(18.0, 22.0),
        QPointF(14.0, 22.0),
        QPointF(14.0, 16.0),
        QPointF(11.0, 16.0),
    ])
    p.drawPolygon(arrow)
    p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
    return _finish(pm, p)


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

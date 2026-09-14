"""Hint rendering: geometry is pure and testable; painting needs PySide6.

``placement_cell_rects`` converts a solver move into pixel rectangles and is
usable headless. ``draw_hint`` performs the actual QPainter work and is only
callable when PySide6 is installed (macOS).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..solver.search import Move
from ..vision.grid import HINT_FILL_OPACITY

try:  # pragma: no cover - depends on platform
    from PySide6.QtCore import QRectF
    from PySide6.QtGui import QColor, QPainter, QPen

    HAVE_QT = True
except ImportError:  # pragma: no cover
    HAVE_QT = False


@dataclass(frozen=True)
class HintStyle:
    """Visual style of the placement hint.

    What is painted here comes back round: the overlay sits over the game
    and the next capture contains it, so ``vision.grid`` has to recognize
    this paint again to keep it out of the board reading (see
    ``vision.grid._own_paint_layer``). ``fill_opacity`` is therefore
    imported rather than written twice — the painter and the reader must
    agree on the composite or the reader names nothing.
    """

    color: str = "#00e5ff"  # any Qt-parsable color string
    outline_width: float = 3.0
    fill_opacity: float = HINT_FILL_OPACITY  # 0..1 subtle fill inside each cell
    inset: float = 1.5  # pixels each cell rectangle is shrunk on every side
    show_rotation_badge: bool = True


def placement_cell_rects(
    move: Move, cell_width: float, cell_height: float, inset: float = 0.0
) -> list[tuple[float, float, float, float]]:
    """Pixel rectangles (x, y, w, h) of the move's four cells.

    Coordinates are relative to the top-left of the board region; the caller
    supplies the per-cell pixel size of the overlay window.
    """
    rects = []
    for row, col in move.cells:
        x = col * cell_width + inset
        y = row * cell_height + inset
        rects.append((x, y, cell_width - 2 * inset, cell_height - 2 * inset))
    return rects


def draw_hint(
    painter: QPainter,
    move: Move,
    cell_width: float,
    cell_height: float,
    style: HintStyle | None = None,
) -> None:
    """Paint the placement hint onto the (transparent) overlay window."""
    if not HAVE_QT:  # pragma: no cover - macOS only
        raise RuntimeError("draw_hint requires PySide6 (macOS overlay runtime)")
    style = style or HintStyle()
    color = QColor(style.color)
    fill = QColor(color)
    fill.setAlphaF(style.fill_opacity)

    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    pen = QPen(color)
    pen.setWidthF(style.outline_width)
    painter.setPen(pen)
    painter.setBrush(fill)
    for x, y, w, h in placement_cell_rects(move, cell_width, cell_height, style.inset):
        painter.drawRect(QRectF(x, y, w, h))

    if style.show_rotation_badge and move.rotation.index:
        _draw_rotation_badge(painter, move, cell_width, cell_height, color)


def _draw_rotation_badge(
    painter: QPainter,
    move: Move,
    cell_width: float,
    cell_height: float,
    color: Any,
) -> None:  # pragma: no cover - macOS only
    """Small badge with the number of clockwise rotations needed."""
    top_row = min(r for r, _ in move.cells)
    left_col = min(c for _, c in move.cells)
    x = left_col * cell_width
    y = top_row * cell_height - cell_height * 0.6
    badge = QRectF(x, max(0.0, y), cell_width * 0.6, cell_height * 0.55)
    painter.setBrush(color)
    painter.setPen(QPen(QColor(0, 0, 0, 0)))
    painter.drawEllipse(badge)
    painter.setPen(QPen(QColor("black")))
    painter.drawText(badge, 0x84, str(move.rotation.index))  # AlignCenter

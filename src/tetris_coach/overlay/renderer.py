"""Hint rendering: geometry is pure and testable; painting needs PySide6.

``placement_cell_rects`` converts a solver move into pixel rectangles and is
usable headless. ``draw_hint`` performs the actual QPainter work and is only
callable when PySide6 is installed (macOS).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..solver.search import Move
from ..vision.colour_palette import CELL_MARGIN
from ..vision.grid import HINT_FILL_OPACITY

# Size of the rotation badge, as a share of one cell. The height has to
# stay inside the cell's top margin -- the band above the patch the
# readers sample, CELL_MARGIN deep -- with room for the antialiased edge,
# or the badge is back to deleting the cell it sits in. See
# :func:`rotation_badge_rect`.
BADGE_WIDTH = 0.5
BADGE_HEIGHT = CELL_MARGIN * 0.8

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


def rotation_badge_rect(
    move: Move, cell_width: float, cell_height: float
) -> tuple[float, float, float, float]:
    """Pixel rectangle (x, y, w, h) of the rotation badge.

    THE BADGE IS OPAQUE, so wherever it lands it deletes what the game
    drew underneath: it is the one thing this tool paints that cannot be
    composited back out, and the readers recognize it as ours and skip the
    cell. It used to be hung in the cell ABOVE the hint's top-left corner
    -- which is a cell the falling piece passes through on its way to the
    target. Measured: a vertical I in column 3 with the badge over (5, 3)
    is read as three cells at (6,3), (7,3), (8,3) -- a fragment no shape
    can name -- so an unnamed colour gets no hint, the hint comes down,
    the badge goes with it, the piece is named again and the hint comes
    back. A 15 fps loop, driven by the coach's own drawing.

    So it goes INSIDE the hint's own top-left cell, in that cell's top
    margin: the band above the central patch both readers sample
    (:data:`~tetris_coach.vision.colour_palette.CELL_MARGIN` of the cell
    on every side). There it changes no reading at all -- the cell is
    already painted, its patch still carries the translucent fill the
    reader un-composites exactly, and no OTHER cell is touched, which
    matters because hint-coloured pixels in an unpainted cell would be
    un-composited too and arrive as content the game never drew.

    The badge is smaller for it. That is the trade: a digit at a fifth of
    a cell high against a hint that flickers at 15 fps and a piece that
    cannot be named.
    """
    top_row = min(r for r, _ in move.cells)
    left_col = min(c for _, c in move.cells)
    return (
        left_col * cell_width,
        top_row * cell_height,
        cell_width * BADGE_WIDTH,
        cell_height * BADGE_HEIGHT,
    )


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
    badge = QRectF(*rotation_badge_rect(move, cell_width, cell_height))
    painter.setBrush(color)
    painter.setPen(QPen(QColor(0, 0, 0, 0)))
    painter.drawEllipse(badge)
    painter.setPen(QPen(QColor("black")))
    painter.drawText(badge, 0x84, str(move.rotation.index))  # AlignCenter

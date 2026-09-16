"""Hint rendering: geometry is pure and testable; painting needs PySide6.

Everything this tool draws is drawn IN THE OUTER BAND OF A CELL. Both
readers name a cell from its central patch, inset
:data:`~tetris_coach.vision.colour_palette.CELL_MARGIN` on every side, so
paint that stays outside every patch cannot reach a reading at all -- not
because a rule takes it back out, but because the reader never looks
there. That is the whole design, and it is the reason the hints can afford
to be BOLD: the visual weight moved out of a translucent fill (which the
reader does look at, and which has cost this project three user-visible
bugs) and into a band a quarter of a cell deep, ~12 px on a real capture.

The geometry is pure and is where every decision lives:
:func:`hint_paint_rects` returns the exact pixel rectangles a hint covers.
:func:`draw_hint` fills those rectangles through QPainter (macOS only) and
:func:`paint_hint` fills the same ones into a numpy frame, which is how
``tests/test_hint_invisibility.py`` MEASURES the property above over every
committed capture window -- Qt cannot be driven headlessly, so the thing
under test is the geometry both painters share.

Two hints are drawn, and they are told apart by STROKE rather than by
transparency: the current piece gets a solid outline, the next piece a
dashed one in a second colour. A fainter second mark would sit closer to
the board background, which is the crowded band where ghosts and pale
pieces already collide; a dash reads at a glance over any of them.

THE BOARD IS NOT THE ONLY THING CAPTURED. There is a second grab every
tick -- the NEXT-piece box -- and in the games this tool was built for
that box floats INSIDE the board rectangle, so the overlay window, which
is sized to the whole board, covers it. The invisibility argument above
does not reach that reader at all: it is about the outer band of a BOARD
cell, and the preview reader samples its own crop at pixel level, where a
full-opacity stroke is not a translucent fill it can recognize but four
solid bars through the middle of the piece it is trying to name. So the
painters take a KEEP-OUT rectangle -- the next-piece capture, in
board-relative pixels -- and no rectangle here is filled inside it (see
:func:`keep_out_rect` and :data:`KEEP_OUT_SLACK`). The property is the
same one and it is kept the same way: not by a rule that takes the paint
back out, but by the paint never being there.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..solver.search import Move
from ..vision.colour_palette import CELL_MARGIN
from ..vision.grid import (
    DEFAULT_HINT_COLOR,
    DEFAULT_NEXT_HINT_COLOR,
    HINT_FILL_OPACITY,
    OwnPaint,
)

#: A pixel rectangle: (x, y, width, height), relative to the board region's
#: top-left corner.
Rect = tuple[float, float, float, float]

#: A rectangle in board-relative FRACTIONS: (x0, y0, x1, y1), each in
#: [0, 1]. Retina-agnostic, which is why the keep-out crosses module
#: boundaries in this form: the overlay window measures itself in points
#: and a captured frame may be twice that, and only ratios survive both.
Fractions = tuple[float, float, float, float]

#: Pixels the keep-out is grown by on every side. The painters snap their
#: rectangles to whole pixels and fill them opaquely, so nothing spills on
#: its own; this is for the rounding on the way IN -- a fraction of the
#: board turned back into pixels against a raster whose size is not the
#: one it was measured on. Over-stating the region the overlay must not
#: touch is the safe direction, and one pixel of a hint is not a feature.
KEEP_OUT_SLACK = 1

# The band, as a share of a cell, that is INVISIBLE to both readers: the
# part of the cell outside the central patch they sample. Anything drawn
# here changes no cell's mean colour, and anything drawn outside it does.
HINT_BAND = CELL_MARGIN

# How much of that band is left unpainted, as a share of a cell, between
# the outline's inner edge and the patch. It absorbs the two ways a
# rectangle can arrive a fraction of a pixel wider than it was computed:
# the readers round each patch edge to a whole pixel (which can move it
# half a pixel outward) and an antialiased edge can spill one more. At the
# ~48 px cells of every committed window this is 2.4 px of slack against
# 1.5 px of worst case, and ``tests/test_hint_invisibility.py`` pins it by
# dilating every painted rectangle a pixel before checking.
PATCH_CLEARANCE = 0.05

#: Depth of the outline, as a share of the cell: the whole invisible band
#: bar the clearance. ~9.6 px on a real capture, against the 3 px pen this
#: replaced -- the hint got bolder by giving up the fill, not by taking
#: more room.
OUTLINE_DEPTH = HINT_BAND - PATCH_CLEARANCE

#: A dashed side is cut into this many equal segments with every other one
#: painted. Odd on purpose: a side then starts AND ends on a dash, so the
#: corners of a dashed hint are closed and the tetromino keeps its shape.
DASH_SEGMENTS = 5

# Size of the rotation badge, as a share of one cell. The height is the
# outline's own depth, which is what keeps the badge honest: it is drawn
# inside the band its cell's outline already covers, so it adds no painted
# pixel outside the band and the invisibility argument needs no second
# case for it. See :func:`rotation_badge_rect`.
BADGE_WIDTH = 0.5
BADGE_HEIGHT = OUTLINE_DEPTH

try:  # pragma: no cover - depends on platform
    from PySide6.QtCore import QRectF
    from PySide6.QtGui import QColor, QPainter, QPen

    HAVE_QT = True
except ImportError:  # pragma: no cover
    HAVE_QT = False


@dataclass(frozen=True)
class HintStyle:
    """Visual style of one placement hint.

    ``dashed`` is the whole distinction between the two hints on screen.
    It is a stroke STYLE and not an opacity because the second hint has to
    stay readable over a pale piece and a ghost, and anything faint sits
    down among those; a dash is legible against every one of them and
    against the other hint.

    ``fill_opacity`` is the one thing here the readers can see. It is 0 by
    default -- the hint is an outline, drawn where no reader looks -- and
    turning it up RE-OPENS the path by which this tool reads its own
    drawing back as board content (:func:`~tetris_coach.vision.grid._own_paint_layer`
    is what then has to take it out again). It is kept because the
    recognition is kept: a session that wants the old look can have it,
    and the reader it is handed to is told the same number
    (``app.CoachConfig.hint_fill_opacity``), so painter and reader cannot
    drift apart about the composite.
    """

    color: str = DEFAULT_HINT_COLOR  # any Qt-parsable color string
    dashed: bool = False
    fill_opacity: float = 0.0  # 0..1; see the note above before raising it
    inset: float = 0.0  # pixels the fill rectangle is shrunk on every side
    show_rotation_badge: bool = True


def placement_cell_rects(
    move: Move, cell_width: float, cell_height: float, inset: float = 0.0
) -> list[Rect]:
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


def _snap(x0: float, y0: float, x1: float, y1: float) -> Rect:
    """A rectangle on whole pixels, so every painter covers the same ones.

    Fractional edges are the one place the Qt painter and the numpy one
    could disagree about a pixel, and a disagreement there would make the
    measured invisibility property a measurement of the wrong thing. Both
    are handed integers instead, and the clearance above absorbs the half
    pixel the rounding can move an edge by.
    """
    left, top = round(x0), round(y0)
    right, bottom = round(x1), round(y1)
    return (float(left), float(top), float(max(1, right - left)), float(max(1, bottom - top)))


def _snap_out(rect: Rect) -> Rect:
    """``rect`` grown to the whole pixels it touches.

    For the badge, which is the one thing here drawn as a shape rather
    than as a rectangle: Qt renders the ellipse antialiased, so it tints
    the pixels its bounding box lands a fraction of the way into, and
    rounding that box to the nearest pixel would under-state the paint by
    one. Over-stating it is the safe direction -- everything that reads
    these rectangles is asking whether the paint can reach somewhere it
    must not.
    """
    x, y, w, h = rect
    left, top = math.floor(x), math.floor(y)
    right, bottom = math.ceil(x + w), math.ceil(y + h)
    return (float(left), float(top), float(max(1, right - left)), float(max(1, bottom - top)))


def keep_out_rect(
    keep_out: Fractions | None, board_width: float, board_height: float
) -> Rect | None:
    """The pixel rectangle the overlay may not paint in, on this raster.

    ``keep_out`` is the NEXT-piece capture as a share of the board region
    (:func:`~tetris_coach.app.preview_keep_out`), and this is that share
    against a concrete board size: the overlay window's own, in points, or
    a captured frame's, in pixels. ``None`` in, ``None`` out, which is the
    ordinary case -- a next box drawn somewhere else on screen is not
    under the overlay at all and nothing here applies.

    Rounded OUTWARD and then grown by :data:`KEEP_OUT_SLACK`, because what
    is being answered is "could a painted pixel land inside the second
    grab", and the honest answer to that has to round the way that cannot
    be wrong.
    """
    if keep_out is None:
        return None
    fx0, fy0, fx1, fy1 = keep_out
    left = math.floor(fx0 * board_width) - KEEP_OUT_SLACK
    top = math.floor(fy0 * board_height) - KEEP_OUT_SLACK
    right = math.ceil(fx1 * board_width) + KEEP_OUT_SLACK
    bottom = math.ceil(fy1 * board_height) + KEEP_OUT_SLACK
    if right <= left or bottom <= top:
        return None
    return (float(left), float(top), float(right - left), float(bottom - top))


def _overlaps(rect: Rect, other: Rect) -> bool:
    x, y, w, h = rect
    ox, oy, ow, oh = other
    return x < ox + ow and x + w > ox and y < oy + oh and y + h > oy


def _subtract(rect: Rect, keep_out: Rect) -> list[Rect]:
    """``rect`` minus ``keep_out``, as up to four rectangles.

    Subtraction rather than "drop the whole rectangle if it touches": an
    outline side runs the full width of a cell, and a keep-out that
    clipped whole sides would erase a hint two cells before the panel
    actually reaches it. What is left is drawn; what is inside is not
    drawn at all, in either painter, which is what makes the next crop
    provably free of this tool's paint.
    """
    x, y, w, h = rect
    x1, y1 = x + w, y + h
    kx, ky, kw, kh = keep_out
    kx1, ky1 = kx + kw, ky + kh
    if not _overlaps(rect, keep_out):
        return [rect]
    parts: list[Rect] = []
    if ky > y:
        parts.append((x, y, w, ky - y))
    if ky1 < y1:
        parts.append((x, ky1, w, y1 - ky1))
    top, bottom = max(y, ky), min(y1, ky1)
    if kx > x:
        parts.append((x, top, kx - x, bottom - top))
    if kx1 < x1:
        parts.append((kx1, top, x1 - kx1, bottom - top))
    return [part for part in parts if part[2] > 0 and part[3] > 0]


def _clip(rects: list[Rect], keep_out: Rect | None) -> list[Rect]:
    """``rects`` with everything inside ``keep_out`` removed."""
    if keep_out is None:
        return rects
    out: list[Rect] = []
    for rect in rects:
        out.extend(_subtract(rect, keep_out))
    return out


def _badge_rects(
    move: Move, cell_width: float, cell_height: float, style: HintStyle, keep_out: Rect | None
) -> list[Rect]:
    """The badge's rectangle, or none at all.

    The badge is the one thing here that is not a rectangle -- a disc with
    a digit in it -- so it is dropped whole rather than clipped when it
    reaches the keep-out. Half a disc with half a digit in it is not a
    rotation count anybody can read, and an all-or-nothing rule is one
    both painters can follow identically without Qt needing a clip region.
    """
    if not (style.show_rotation_badge and move.rotation.index):
        return []
    rect = _snap_out(rotation_badge_rect(move, cell_width, cell_height))
    if keep_out is not None and _overlaps(rect, keep_out):
        return []
    return [rect]


def _side_rects(
    x0: float, y0: float, x1: float, y1: float, depth_x: float, depth_y: float, dashed: bool
) -> list[Rect]:
    """The four sides of one cell's outline, solid or cut into dashes."""
    sides = (
        (x0, y0, x1, y0 + depth_y, True),  # top
        (x0, y1 - depth_y, x1, y1, True),  # bottom
        (x0, y0, x0 + depth_x, y1, False),  # left
        (x1 - depth_x, y0, x1, y1, False),  # right
    )
    out: list[Rect] = []
    for sx0, sy0, sx1, sy1, horizontal in sides:
        if not dashed:
            out.append(_snap(sx0, sy0, sx1, sy1))
            continue
        low, high = (sx0, sx1) if horizontal else (sy0, sy1)
        step = (high - low) / DASH_SEGMENTS
        for index in range(0, DASH_SEGMENTS, 2):
            a, b = low + index * step, low + (index + 1) * step
            out.append(_snap(a, sy0, b, sy1) if horizontal else _snap(sx0, a, sx1, b))
    return out


def outline_rects(
    move: Move, cell_width: float, cell_height: float, style: HintStyle | None = None
) -> list[Rect]:
    """Every pixel rectangle the hint's outline covers.

    One outline per CELL rather than one around the tetromino's silhouette:
    the player is counting cells, and a per-cell outline is also what makes
    the invisibility argument local -- every rectangle here lies in the
    outer band of a cell the hint occupies, so it is enough to know that
    the hint's own cells are painted to know that no patch anywhere is.
    """
    style = style or HintStyle()
    depth_x, depth_y = OUTLINE_DEPTH * cell_width, OUTLINE_DEPTH * cell_height
    rects: list[Rect] = []
    for row, col in move.cells:
        x0, y0 = col * cell_width, row * cell_height
        rects.extend(
            _side_rects(x0, y0, x0 + cell_width, y0 + cell_height, depth_x, depth_y, style.dashed)
        )
    return rects


def rotation_badge_rect(move: Move, cell_width: float, cell_height: float) -> Rect:
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
    on every side). There it changes no reading at all -- and since that
    band is now exactly what the outline paints (:data:`BADGE_HEIGHT` is
    :data:`OUTLINE_DEPTH`), the badge adds no painted pixel the outline
    had not already put there. No OTHER cell is touched, which matters
    because hint-coloured pixels in an unpainted cell would be read as
    content the game never drew.

    THE CELL HAS TO BE ONE THE HINT ACTUALLY PAINTS, and ``min(row)`` with
    ``min(col)`` is the corner of the BOUNDING BOX, which for 6 of the 19
    rotations is a cell the piece leaves empty: both T verticals, S and Z
    in one orientation each, and one each of J and L. The badge landed on
    bare board there -- the case the paragraph above says must not happen
    -- and ``tests/fixtures/hint_stutter`` is 51 captured frames of what
    it cost: a phantom one-cell piece alternating with the real J at the
    top edge, and the hint blanking every other frame. The reader no
    longer un-composites it
    (:func:`~tetris_coach.vision.colour_palette.own_paint_states`), but
    OURS still means the cell is unreadable, so putting the badge on a
    cell the piece may fall through is still deleting a cell for nothing.

    The top-left OCCUPIED cell is the leftmost cell of the top row, which
    every rotation has by construction.

    WITH TWO HINTS ON SCREEN there are two badges, and the reasoning above
    is per hint: each badge sits in a cell ITS OWN hint paints. That a
    badge cannot land in a cell belonging to the other hint is not a
    property of this function but of the pair -- the second hint is solved
    on a board where the first hint's cells are already filled, so no
    placement can occupy one of them. :attr:`~tetris_coach.app.CoachEngine.second_hint`
    refuses a second hint that overlaps the first rather than trusting
    that argument.
    """
    top_row = min(r for r, _ in move.cells)
    left_col = min(c for r, c in move.cells if r == top_row)
    return (
        left_col * cell_width,
        top_row * cell_height,
        cell_width * BADGE_WIDTH,
        cell_height * BADGE_HEIGHT,
    )


def _fill_rects(move: Move, cell_width: float, cell_height: float, style: HintStyle) -> list[Rect]:
    """The hint's fill rectangles, on whole pixels; empty when there is no fill."""
    if style.fill_opacity <= 0.0:
        return []
    return [
        _snap(x, y, x + w, y + h)
        for x, y, w, h in placement_cell_rects(move, cell_width, cell_height, style.inset)
    ]


def hint_paint_rects(
    move: Move,
    cell_width: float,
    cell_height: float,
    style: HintStyle | None = None,
    keep_out: Rect | None = None,
) -> list[Rect]:
    """Every pixel rectangle this hint puts paint in, opaque or not.

    The single answer to "where does the overlay draw?", which is the
    question the invisibility property is about. The badge is included as
    its bounding rectangle rather than as the disc actually drawn, which
    over-states the paint and is the safe direction to be wrong in.

    ``keep_out`` is the region the overlay must leave alone -- the
    next-piece capture, which in the games this was built for lies under
    the overlay window (see this module's docstring). Nothing inside it is
    returned, so the answer stays the single one: what is here is what is
    painted.
    """
    style = style or HintStyle()
    rects = outline_rects(move, cell_width, cell_height, style)
    rects.extend(_fill_rects(move, cell_width, cell_height, style))
    return _clip(rects, keep_out) + _badge_rects(move, cell_width, cell_height, style, keep_out)


def paint_hint(
    frame: NDArray[np.uint8],
    move: Move,
    cell_width: float,
    cell_height: float,
    style: HintStyle | None = None,
    keep_out: Rect | None = None,
) -> NDArray[np.uint8]:
    """``frame`` with the hint painted on it: the headless mirror of :func:`draw_hint`.

    Qt is macOS-only and cannot be driven in a headless test run, so the
    property the whole design rests on -- that what this tool paints
    changes nothing a reader reads -- is measured through this instead,
    over the same :func:`hint_paint_rects` geometry ``draw_hint`` fills.
    What could drift between the two is therefore only the FILL of a
    rectangle, not which rectangles there are -- and it is measured not to,
    by ``tests/test_qt_painter_geometry.py``: driven offscreen through a
    real QPainter over all 19 rotations at four cell sizes in both styles,
    the pixels ``draw_hint`` touches are a SUBSET of the ones this paints,
    differing only where the badge's ellipse and its clipped digit do not
    reach the corners of the badge's own bounding box. A superset is the
    direction that keeps the invisibility measurement honest. That file
    skips where PySide6 is absent, so the bridge is checked wherever the
    overlay can actually run and a headless run stays green.

    ``keep_out`` is the region the overlay must not paint in, in this
    frame's own pixels; see :func:`keep_out_rect`.

    ``frame`` is BGR, the order the capture pipeline produces, and is not
    modified: a copy comes back.
    """
    style = style or HintStyle()
    paint = OwnPaint.for_hint_color(style.color)
    if paint is None:
        raise ValueError(f"paint_hint needs a hex colour, not {style.color!r}")
    out = np.array(frame, dtype=np.uint8, copy=True)
    colour = np.asarray(paint.color, dtype=np.float64)[: out.shape[2]]
    for rect in _clip(_fill_rects(move, cell_width, cell_height, style), keep_out):
        _blend(out, rect, colour, style.fill_opacity)
    for rect in _clip(outline_rects(move, cell_width, cell_height, style), keep_out):
        _blend(out, rect, colour, 1.0)
    for rect in _badge_rects(move, cell_width, cell_height, style, keep_out):
        _blend(out, rect, colour, 1.0)
    return out


def _blend(frame: NDArray[np.uint8], rect: Rect, colour: NDArray[np.float64], alpha: float) -> None:
    """Composite ``colour`` at ``alpha`` over one rectangle of ``frame``."""
    x, y, w, h = (round(v) for v in rect)
    height, width = int(frame.shape[0]), int(frame.shape[1])
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(width, x + w), min(height, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    patch = frame[y0:y1, x0:x1].astype(np.float64)
    frame[y0:y1, x0:x1] = np.round(patch + alpha * (colour - patch)).astype(np.uint8)


def draw_hint(
    painter: QPainter,
    move: Move,
    cell_width: float,
    cell_height: float,
    style: HintStyle | None = None,
    keep_out: Rect | None = None,
) -> None:
    """Paint one placement hint onto the (transparent) overlay window.

    Thin on purpose: the rectangles come from :func:`hint_paint_rects`'s
    pure geometry, which is what the headless tests measure, and nothing
    is decided here. Antialiasing is off for them so a filled rectangle
    covers exactly the whole pixels it was computed on; the badge, which
    is a disc with a digit in it, turns it back on for itself.

    ``keep_out`` is subtracted from the rectangles in that same pure
    geometry rather than applied here as a Qt clip region, so the two
    painters cannot come to different answers about which pixels the
    keep-out saves -- the one thing this function is not allowed to
    decide for itself.
    """
    if not HAVE_QT:  # pragma: no cover - macOS only
        raise RuntimeError("draw_hint requires PySide6 (macOS overlay runtime)")
    style = style or HintStyle()
    color = QColor(style.color)

    painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
    if style.fill_opacity > 0.0:
        fill = QColor(color)
        fill.setAlphaF(style.fill_opacity)
        for rect in _clip(_fill_rects(move, cell_width, cell_height, style), keep_out):
            painter.fillRect(QRectF(*rect), fill)
    for x, y, w, h in _clip(outline_rects(move, cell_width, cell_height, style), keep_out):
        painter.fillRect(QRectF(x, y, w, h), color)

    if _badge_rects(move, cell_width, cell_height, style, keep_out):
        _draw_rotation_badge(painter, move, cell_width, cell_height, color)


def _draw_rotation_badge(
    painter: QPainter,
    move: Move,
    cell_width: float,
    cell_height: float,
    color: Any,
) -> None:  # pragma: no cover - macOS only
    """Small badge with the number of clockwise rotations needed."""
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    badge = QRectF(*rotation_badge_rect(move, cell_width, cell_height))
    painter.setBrush(color)
    painter.setPen(QPen(QColor(0, 0, 0, 0)))
    painter.drawEllipse(badge)
    painter.setPen(QPen(QColor("black")))
    painter.drawText(badge, 0x84, str(move.rotation.index))  # AlignCenter


__all__ = [
    "BADGE_HEIGHT",
    "BADGE_WIDTH",
    "DASH_SEGMENTS",
    "DEFAULT_HINT_COLOR",
    "DEFAULT_NEXT_HINT_COLOR",
    "HINT_BAND",
    "HINT_FILL_OPACITY",
    "KEEP_OUT_SLACK",
    "OUTLINE_DEPTH",
    "PATCH_CLEARANCE",
    "Fractions",
    "HintStyle",
    "Rect",
    "draw_hint",
    "hint_paint_rects",
    "keep_out_rect",
    "outline_rects",
    "paint_hint",
    "placement_cell_rects",
    "rotation_badge_rect",
]

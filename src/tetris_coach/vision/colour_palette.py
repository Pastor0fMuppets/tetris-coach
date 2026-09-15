"""Colour segmentation and a colour->piece palette learned during the session.

PARALLEL PROTOTYPE. Nothing here is wired into ``app.py``/``cli.py``; the
shipped reading path (``grid.py`` -> ``pieces_vision.py`` -> ``state.py``)
is untouched. See :mod:`.colour_tracker` for the tracker built on this.

The shipped pipeline reduces every cell to one bit — filled or empty — and
must then name the falling piece by its SHAPE. A piece entering from above
shows 1-3 cells, which genuinely fit several tetrominoes, so it cannot be
named until it has descended; and in occupancy a ghost, a UI panel, this
tool's own hint paint and a real piece are all just "filled", so each needs
its own structural rule and the rules interact. This module keeps the
colour instead.

What the colour is
------------------

A cell's identity is the DIRECTION of its colour away from the board
background, not the colour itself::

    v = cell_colour - background        (BGR, uint8 units)
    identity = v / |v|                  which piece
    |v|                                 how opaque

Direction is invariant to alpha compositing over the background, which is
how every translucent thing in a Tetris rendering is drawn. Measured on
this repo's fixtures, the NEXT box draws each piece at ~87% opacity over
the same ground as the board — the board's I is (215, 46, 45) and the box's
is (220, 77, 76), 44 uint8 units apart, but the two directions differ by
0.3 degrees. Distance alone would have to be told that those are the same
piece; a direction already knows. The same identity covers a landing-preview
ghost (a weak blend, same direction) and this tool's own hint fill.

Matching is by PERPENDICULAR distance to a class's ray from the background
(:data:`LINE_TOL`), which is |v| * sin(angle): a faint colour gets the loose
angular tolerance its noise deserves, a vivid one a tight one. Measured on
the committed fixtures, the five rendered piece colours are 70+ units apart
and a piece's own cells have ZERO spread (every cell of a piece samples to
the same value to the unit), so the tolerance has two orders of magnitude of
headroom; the nearest pair of DIRECTIONS is the pale periwinkle T and the
blue I at 15 degrees, which puts the T 13.7 units off the I's ray.

Three classes of cell are not board content
-------------------------------------------

- **Near the background** (:data:`EMPTY_DIST`): empty board.
- **Our own hint paint**: this tool composites ``#00e5ff`` over the game and
  then captures the screen again, so its own overlay is in its next input.
  It is the one thing on screen whose exact appearance is known in advance
  rather than guessed at, and it signs itself: the hint is a translucent
  fill inside a THREE-PIXEL OUTLINE of the pure hint colour, so a painted
  cell carries a ring of exactly ``#00e5ff`` (measured: 18-23% of the cell
  rect on every painted cell in the fixtures, and 0.0% on every unpainted
  one). That ring says which cells are painted; the fill is then undone
  exactly, because its opacity is known too, recovering whatever the game
  drew underneath. See :func:`own_paint_states`.
- **Translucent along a known ray** (:data:`TRANSLUCENT` of the class's peak
  magnitude): a landing-preview ghost. One comparison, and it also says
  which piece the ghost belongs to. NOTE: no fixture in this repo actually
  contains a game-drawn ghost — every "ghost" in the committed windows is
  this tool's own hint paint (see :mod:`.colour_tracker`) — so this rule is
  exercised only by a synthetic test.

Everything else is content, named when the palette knows its colour and
unnamed (but still tracked) when it does not.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from ..core.pieces import ROTATIONS
from .grid import HINT_PAINT, OwnPaint
from .grid import _cell_colors as cell_colors  # same sampler => same geometry

# uint8 units from the background below which a cell is the empty board.
# The faintest real piece in any committed fixture (the pale periwinkle T)
# sits at 52.9; the largest stray reading on an observable empty cell is
# under 1.0, because ``cell_colors`` insets 25% and never samples a gridline.
EMPTY_DIST = 12.0

# Perpendicular distance, in uint8 units, from a colour class's ray.
LINE_TOL = 10.0

# Fraction of a class's peak magnitude below which a cell is a translucent
# preview of that piece (a ghost) rather than the piece.
TRANSLUCENT = 0.5

# How near a pixel must sit to the hint colour to BE the hint's own outline,
# and the share of a cell rect that ring must cover for the cell to be under
# our paint. The thinnest ring a 3 px outline can leave is a tenth of a very
# large cell; the measured range on the fixtures is 0.18-0.23.
PAINT_PIXEL_TOL = 24.0
PAINT_RING_SHARE = 0.03

# Share of the SAMPLED patch (not the whole cell) in hint colour above which
# the cell is our own drawing rather than a painted piece: the rotation badge
# is a filled disc, which lands in the patch (measured 0.37) where the fill's
# outline never does (0.00). A cell in hint colour edge to edge is a game
# piece that happens to be that colour, not our paint.
PAINT_PATCH_SHARE = 0.02
PAINT_PATCH_SOLID = 0.9

# Fraction of each cell inset before sampling (skips gridlines and borders).
CELL_MARGIN = 0.25

EMPTY = -1  # cell label: no board content here

CLEAN, PAINTED, OURS = 0, 1, 2  # per-cell paint states

_SHAPE_PIECES: dict[tuple[tuple[int, int], ...], str] = {
    tuple(sorted(rotation.cells)): piece
    for piece, rotations in ROTATIONS.items()
    for rotation in rotations
}


def piece_from_cells(cells: frozenset[tuple[int, int]]) -> str | None:
    """The tetromino those cells spell, or ``None`` if they spell none.

    Cells are normalized to their own bounding box, so this names a piece
    only from a COMPLETE four-cell sighting. Every rotation shape belongs
    to exactly one piece, so a hit is unambiguous.
    """
    if len(cells) != 4:
        return None
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    key = tuple(sorted((r - min_r, c - min_c) for r, c in cells))
    return _SHAPE_PIECES.get(key)


@dataclass
class ColourClass:
    """One rendered colour, held as a ray away from the board background."""

    unit: NDArray[np.float64]  # unit vector, background -> colour
    peak: float  # largest magnitude seen on this ray
    piece: str | None = None  # learned name, when something has named it


class Palette:
    """The board's background, its colour classes, and their piece names.

    Stateful across a session: colours are interned as they are first seen
    and named later, from the NEXT box or from an unambiguous four-cell
    sighting on the board. Nothing here decides what is falling — this only
    answers "what colour is this cell, and whose colour is that".
    """

    def __init__(self, paint: OwnPaint | None = HINT_PAINT) -> None:
        self.background: NDArray[np.float64] | None = None
        self.classes: list[ColourClass] = []
        self._paint = paint

    # -- background ---------------------------------------------------

    def update_background(
        self, colours: NDArray[np.float32], observable: NDArray[np.bool_]
    ) -> None:
        """Re-estimate the board background from this frame.

        Bootstrap is the median of every observable cell, which holds while
        content is a minority of the board. Thereafter only cells already
        reading EMPTY re-measure it, so a filling board cannot drag the
        estimate onto a piece colour (the failure that inverts a whole
        reading), while a theme that fades or animates is still tracked.
        """
        flat = colours[observable]
        if flat.size == 0:
            return
        if self.background is None:
            self.background = np.median(flat, axis=0).astype(np.float64)
            return
        empty = flat[np.linalg.norm(flat - self.background, axis=1) < EMPTY_DIST]
        if empty.shape[0] * 5 >= flat.shape[0]:  # a fifth of the board is ground
            self.background = np.median(empty, axis=0).astype(np.float64)

    # -- colour classes -----------------------------------------------

    def _perpendicular(self, vec: NDArray[np.float64], unit: NDArray[np.float64]) -> float:
        along = float(vec @ unit)
        return float(np.sqrt(max(0.0, float(vec @ vec) - along * along)))

    def match(self, vec: NDArray[np.float64]) -> int | None:
        """Index of the class whose ray ``vec`` lies on, or ``None``."""
        best, best_dist = None, LINE_TOL
        for index, klass in enumerate(self.classes):
            dist = self._perpendicular(vec, klass.unit)
            if dist < best_dist:
                best, best_dist = index, dist
        return best

    def intern(self, vec: NDArray[np.float64]) -> int:
        """Index of ``vec``'s class, creating and growing it as needed."""
        magnitude = float(np.linalg.norm(vec))
        index = self.match(vec)
        if index is None:
            self.classes.append(ColourClass(unit=vec / magnitude, peak=magnitude))
            return len(self.classes) - 1
        klass = self.classes[index]
        if magnitude > klass.peak:
            # The most opaque sighting defines the ray: a ghost seen before
            # its piece must not be what the piece is later measured against.
            klass.unit = vec / magnitude
            klass.peak = magnitude
        return index

    def name(self, index: int, piece: str) -> None:
        """Record that class ``index`` is ``piece``'s colour."""
        self.classes[index].piece = piece

    def piece_of(self, index: int) -> str | None:
        return self.classes[index].piece if 0 <= index < len(self.classes) else None

    # -- segmentation -------------------------------------------------

    def _unpainted(self, vec: NDArray[np.float64]) -> NDArray[np.float64]:
        """``vec`` with this tool's own hint fill composited back out.

        The fill is ``opacity`` of the hint colour over whatever the game
        drew, so the game's own colour is recoverable exactly rather than
        approximately — the one piece of furniture on the board this tool
        does not have to guess at, because it painted it.
        """
        if self._paint is None or self.background is None:
            return vec
        colour = np.asarray(self._paint.color, dtype=np.float64)
        if colour.shape != self.background.shape:
            return vec
        alpha = self._paint.opacity
        return (vec - alpha * (colour - self.background)) / (1.0 - alpha)

    def classify(
        self,
        colours: NDArray[np.float32],
        observable: NDArray[np.bool_],
        paint_states: NDArray[np.int8] | None = None,
    ) -> NDArray[np.int16]:
        """Per-cell colour class: :data:`EMPTY` or an index into ``classes``.

        EMPTY covers all four ways a cell holds no board content: it is the
        background, it is unobservable, it is our own drawing, or it is a
        translucent ghost of a piece that is somewhere else.
        """
        if self.background is None:
            raise ValueError("background not estimated yet")
        rows, cols = colours.shape[0], colours.shape[1]
        labels = np.full((rows, cols), EMPTY, dtype=np.int16)
        for r in range(rows):
            for c in range(cols):
                state = CLEAN if paint_states is None else int(paint_states[r, c])
                if not observable[r, c] or state == OURS:
                    continue
                vec = colours[r, c].astype(np.float64) - self.background
                if state == PAINTED:
                    vec = self._unpainted(vec)
                magnitude = float(np.linalg.norm(vec))
                if magnitude < EMPTY_DIST:
                    continue
                index = self.intern(vec)
                if magnitude >= TRANSLUCENT * self.classes[index].peak:
                    labels[r, c] = index
        return labels


def board_colours(image: NDArray[np.uint8], rows: int, cols: int) -> NDArray[np.float32]:
    """Mean colour of every cell's central patch: ``(rows, cols, 3)``."""
    return cell_colors(image, rows, cols, CELL_MARGIN)


def own_paint_states(
    image: NDArray[np.uint8], rows: int, cols: int, paint: OwnPaint | None = HINT_PAINT
) -> NDArray[np.int8]:
    """Which cells carry this tool's own hint, per :data:`CLEAN`/etc.

    The hint is drawn as a translucent fill inside a solid outline of the
    pure hint colour, so a painted cell is signed by a ring of that exact
    colour around a patch that holds none of it (:data:`PAINT_RING_SHARE`,
    :data:`PAINT_PATCH_SHARE`). Reading the signature rather than the
    composite is what lets the fill be undone over ANY background — the
    shipped rule can only recognize the composite over the board's own
    ground, and a hint drawn over a piece reads there as a colour the game
    never rendered.
    """
    states = np.zeros((rows, cols), dtype=np.int8)
    img = np.asarray(image)
    if paint is None or img.ndim != 3 or len(paint.color) != int(img.shape[2]):
        return states
    colour = np.asarray(paint.color, dtype=np.float64)
    low = np.clip(colour - PAINT_PIXEL_TOL, 0, 255).astype(np.uint8)
    high = np.clip(colour + PAINT_PIXEL_TOL, 0, 255).astype(np.uint8)
    # Integral image of the hint-coloured pixels: every cell rect and every
    # patch rect is then four lookups rather than a slice and a mean.
    summed = cv2.integral(cv2.inRange(np.ascontiguousarray(img), low, high) // 255)
    ys = np.linspace(0, img.shape[0], rows + 1).round().astype(int)
    xs = np.linspace(0, img.shape[1], cols + 1).round().astype(int)

    def share(y0: int, y1: int, x0: int, x1: int) -> float:
        area = (y1 - y0) * (x1 - x0)
        if area <= 0:
            return 0.0
        total = summed[y1, x1] - summed[y0, x1] - summed[y1, x0] + summed[y0, x0]
        return float(total) / area

    for r in range(rows):
        y0, y1 = int(ys[r]), int(ys[r + 1])
        iy0, iy1 = _inset(y0, y1)
        for c in range(cols):
            x0, x1 = int(xs[c]), int(xs[c + 1])
            if share(y0, y1, x0, x1) < PAINT_RING_SHARE:
                continue
            ix0, ix1 = _inset(x0, x1)
            patch = share(iy0, iy1, ix0, ix1)
            if patch < PAINT_PATCH_SHARE:
                states[r, c] = PAINTED  # our fill, over whatever is beneath
            elif patch < PAINT_PATCH_SOLID:
                states[r, c] = OURS  # our own drawing: the rotation badge
    return states


def _inset(low: int, high: int) -> tuple[int, int]:
    span = high - low
    return low + round(span * CELL_MARGIN), max(low + 1, high - round(span * CELL_MARGIN))


def observable_mask(
    rows: int, cols: int, unobservable: frozenset[tuple[int, int]] | None
) -> NDArray[np.bool_]:
    """``(rows, cols)`` True where the capture can actually see the board."""
    mask = np.ones((rows, cols), dtype=bool)
    for r, c in unobservable or ():
        if 0 <= r < rows and 0 <= c < cols:
            mask[r, c] = False
    return mask

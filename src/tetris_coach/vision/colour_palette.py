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
Everything else is content, named when the palette knows its colour and
unnamed (but still tracked) when it does not.

What about the landing-preview ghost?
-------------------------------------

This module used to hold a third rule: a cell below half a colour class's
brightest sighting was a translucent preview of that piece, and therefore
empty. It has been removed, because it did not do what it said and it did
do something bad.

The game DOES draw a landing preview — SPEC.md and two fixture READMEs say
no committed fixture holds one, and they are wrong. On ``spawn_latency``
00134 the oracle reports a ghost at (10,4), (10,5), (11,4), (11,5) and the
raw pixels bear it out: the board ground is RGB (251,252,252), the falling
O is (217,239,112), and those cells carry (242,249,213) — the O's own
colour at 26% alpha — over 9-12% of each cell rect.

But it is an OUTLINE. The centre of the cell is pure background over 84% of
the sampled patch, so a centre-patch sampler cannot see it at all, and
neither this tracker nor the shipped one ever had to. The translucency rule
therefore never once fired on a real landing preview. What it did fire on,
over the six committed windows, was 26 cells of the line-clear flash on
frames the oracle abstains from, and 332 cells of a browser page that had
replaced the game.

Against that it erased whole pieces. Two shades of one hue are the same
class by construction — same direction, different magnitude — so a piece
drawn dimmer than the brightest sighting of its own colour read as a ghost
of itself and vanished: neither falling nor stack, simply absent from the
board handed to the solver. Measured, a T at 45% of its class's peak
returns ``falling=None`` and an empty stack.

So the floor is now :data:`EMPTY_DIST` alone, in absolute units, and a
filled ghost would be read as its piece. Doing better needs the structural
fact that a preview sits BELOW its piece in its piece's columns, and no
frame in this repo contains a filled one to check such a rule against. A
threshold that erases real pieces is not worth keeping on speculation.
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

# What "the board is on screen" means to a reader that assumes flat cells.
# A cell is FLAT when this share of its sampled patch is within this many
# uint8 units of the patch's own median, and the board is readable when
# this share of observable cells is flat. Measured over the six committed
# windows: every frame of the real board scores 0.89 or better (the worst
# is spawn_latency's line-clear flash, and the median frame is 0.97-1.00),
# while the fourteen frames where a web page has replaced the game score
# 0.51. The threshold sits in the gap, a fifth of the range from either
# side. This does not test "is this Tetris" -- it tests the premise this
# whole module rests on, that the thing being read is drawn as flat
# rectangles on a grid.
UNIFORM_TOL = 6.0
UNIFORM_SHARE = 0.98
BOARD_FLAT_SHARE = 0.70

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
    ambiguous: bool = False  # two complete sightings spelled it differently

    def copy(self) -> ColourClass:
        return ColourClass(self.unit.copy(), self.peak, self.piece, self.ambiguous)


@dataclass(frozen=True)
class PaletteState:
    """A palette as it stood before a frame was read: see :meth:`Palette.checkpoint`."""

    background: NDArray[np.float64] | None
    classes: tuple[ColourClass, ...]


class Palette:
    """The board's background, its colour classes, and their piece names.

    Stateful across a session: colours are interned as they are first seen
    and named later, from the NEXT box or from an unambiguous four-cell
    sighting on the board. Nothing here decides what is falling — this only
    answers "what colour is this cell, and whose colour is that".

    This is the one GLOBAL memory in the colour-first design, and it is
    worth being plain about, because :meth:`classify` mutates it once per
    content cell per frame and three of its changes are one-way: a class is
    never removed, :attr:`ColourClass.peak` never falls, and a name retired
    by :meth:`witness` never returns. The background is the exception and
    moves freely. So is a frame the tracker REFUSES, which is put back
    exactly as it stood (:meth:`checkpoint`) rather than left interned --
    the one way anything here goes backwards, and it never reaches a state
    the palette did not really hold.

    What keeps that from being the shipped design's problem in new clothes
    is that nothing is DERIVED from the palette. It labels this frame's
    cells; the board handed to the solver is then read off this frame. A
    wrong class makes a cell the wrong colour, not the board the wrong
    board. The peak in particular no longer gates anything (see the module
    docstring on the ghost rule) — it only chooses which sighting aims a
    class's ray.
    """

    def __init__(self, paint: OwnPaint | None = HINT_PAINT) -> None:
        self.background: NDArray[np.float64] | None = None
        self.classes: list[ColourClass] = []
        self._paint = paint

    # -- undo ---------------------------------------------------------

    def checkpoint(self) -> PaletteState:
        """The palette as it stands now, so a frame can be taken back.

        Reading a frame WRITES here — :meth:`classify` interns a class per
        unrecognized content cell and :meth:`intern` raises a peak — and
        two of those writes are one-way. A caller that only discovers the
        frame was not a board AFTER classifying it (the board's own shape
        is not visible in the pixels until the cells are labelled) can
        therefore not simply discard the labels: it has to put this back.
        Cheap by construction, since a session's palette is a handful of
        classes, not a per-cell structure.
        """
        return PaletteState(
            background=None if self.background is None else self.background.copy(),
            classes=tuple(klass.copy() for klass in self.classes),
        )

    def restore(self, state: PaletteState) -> None:
        """Undo every change since ``state`` was taken."""
        self.background = None if state.background is None else state.background.copy()
        self.classes = [klass.copy() for klass in state.classes]

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
        """Offer a name for class ``index``, from weak evidence.

        The NEXT box's reading. Weak because a box is small, redrawn during
        its own animation, and shares the board's ground only by
        assumption, so it may name a class but never RENAME one: a class
        that already has a name keeps it, and only a complete board
        sighting (:meth:`witness`) is allowed to disagree.
        """
        klass = self.classes[index]
        if klass.piece is None and not klass.ambiguous:
            klass.piece = piece

    def witness(self, index: int, piece: str) -> None:
        """Record a COMPLETE four-cell sighting of class ``index``.

        The strongest naming evidence there is, and the only evidence that
        can contradict the palette. A contradiction is not a tie to break
        by preferring one sighting: it is proof that this colour does not
        determine the piece -- a monochrome theme, two tetrominoes the game
        renders alike, a piece recoloured by level. So the class stops
        naming anything at all and shape takes over for it, which is where
        a colour-blind tracker always was.

        This is a one-way door, and deliberately a narrow one: what it
        costs when it fires wrongly is naming by shape, and what it costs
        when it does not fire is every later piece of that colour named
        wrong for the rest of the session.
        """
        klass = self.classes[index]
        if klass.ambiguous:
            return
        if klass.piece is not None and klass.piece != piece:
            klass.ambiguous = True
            klass.piece = None
            return
        klass.piece = piece

    def piece_of(self, index: int) -> str | None:
        """The piece this class names, or ``None`` if it names none."""
        if not 0 <= index < len(self.classes):
            return None
        klass = self.classes[index]
        return None if klass.ambiguous else klass.piece

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

        EMPTY covers the three ways a cell holds no board content: it is
        within :data:`EMPTY_DIST` of the background, it is unobservable, or
        it is our own drawing. There is deliberately no fourth way -- see
        the module docstring on why the ghost rule was removed.
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
                labels[r, c] = self.intern(vec)
        return labels


def board_colours(image: NDArray[np.uint8], rows: int, cols: int) -> NDArray[np.float32]:
    """Mean colour of every cell's central patch: ``(rows, cols, 3)``."""
    return cell_colors(image, rows, cols, CELL_MARGIN)


def flat_cells(image: NDArray[np.uint8], rows: int, cols: int) -> NDArray[np.bool_]:
    """True where a cell's sampled patch is one flat colour.

    The mean this module reads off each cell is only meaningful if the cell
    IS one colour; over a photograph, a web page or a paragraph of text the
    same mean is an average of unrelated things and every rule downstream
    is reading noise with confidence. This is the cheap check that the
    premise holds, cell by cell.
    """
    img = np.asarray(image)
    ys = np.linspace(0, img.shape[0], rows + 1).round().astype(int)
    xs = np.linspace(0, img.shape[1], cols + 1).round().astype(int)
    out = np.zeros((rows, cols), dtype=bool)
    for r in range(rows):
        y0, y1 = _inset(int(ys[r]), int(ys[r + 1]))
        for c in range(cols):
            x0, x1 = _inset(int(xs[c]), int(xs[c + 1]))
            patch = img[y0:y1, x0:x1].astype(np.float64)
            if patch.size == 0:
                continue
            median = np.median(patch.reshape(-1, patch.shape[-1]), axis=0)
            spread = np.max(np.abs(patch - median), axis=-1)
            out[r, c] = float(np.mean(spread <= UNIFORM_TOL)) >= UNIFORM_SHARE
    return out


def board_readable(
    image: NDArray[np.uint8], rows: int, cols: int, observable: NDArray[np.bool_]
) -> bool:
    """Is enough of this frame drawn as flat cells to be read as a board?"""
    flat = flat_cells(image, rows, cols)[observable]
    return bool(flat.size) and float(np.mean(flat)) >= BOARD_FLAT_SHARE


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

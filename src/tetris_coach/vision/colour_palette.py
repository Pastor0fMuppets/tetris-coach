"""Colour segmentation and a colour->piece palette learned during the session.

The reading the DEFAULT tracker is built on (see :mod:`.colour_tracker`);
the older path (``grid.py`` -> ``pieces_vision.py`` -> ``state.py``) is
untouched and still reachable with ``--tracker shape``.

That path reduces every cell to one bit — filled or empty — and
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

Matching is by ANGLE to a class's ray from the background
(:data:`ANGLE_TOL`), capped in absolute units by the PERPENDICULAR distance
to it (:data:`LINE_TOL`), and both have to hold.

The angle is the one that decides a pale colour, and the perpendicular
distance on its own gets that backwards. Perpendicular distance is
|v| * sin(angle), so a tolerance in units is an angular gate that opens
WIDER the fainter the colour -- and two of the five colours this game
deals are pale (a lavender and a mint, measured on a live session). An
earlier version of this docstring called the margin "two orders of
magnitude" of headroom; that was the separation between the colours, which
is not the quantity the gate compares. The quantity the gate compares was
measured over every cell of the six committed windows: the worst angle any
cell sits off its own class's ray is 1.15 degrees (0.90 units
perpendicular), while the closest two DIFFERENT colours -- the pale
periwinkle T and the blue I -- are 14.5 degrees apart, which puts that T
13.2 units off the I's ray. Ten units of perpendicular tolerance left 3.2
units of margin there; five degrees of angular tolerance leaves 9.5, and
still admits the NEXT box's rendering of a piece, which is 0.3 degrees off
the board's and 44 units away in magnitude.

A colour on the far side of the background is never a match however small
its perpendicular distance: the ray is a RAY.

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

import math
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

# uint8 units above which a cell is no longer the board's OWN ground. The
# empty board does not read "close to" the background, it reads AS the
# background: measured over every accepted frame of the six committed
# windows outside a line-clear animation, not one observable unpainted cell
# sits even 1.0 away from it, the landing-preview ghost included (it is an
# outline, and the sampled patch is pure background). So the band between
# here and :data:`EMPTY_DIST` is empty in real play, and a cell found in it
# is something DRAWN OVER the board in a colour near its ground -- which is
# what this game's "ROW CLEARED" card is, at a measured 7 units
# (``truth/oracle.py``). Four times the largest stray ever measured, and
# three units clear of the content floor. See :func:`off_ground`.
GROUND_TOL = 4.0

# A cell joins a colour class when it is within BOTH of these of the class's
# ray: an angle in degrees, and a perpendicular distance in uint8 units.
# See the module docstring for what each is for and what the measured
# margins are.
LINE_TOL = 10.0
ANGLE_TOL = 5.0

# How far off a class's ray a COMPLETE four-cell sighting has to sit before
# a contradiction is read as two colours this tolerance merged rather than
# one colour the game draws two pieces in. Above the 0.90 units of spread a
# single rendered colour shows anywhere in the corpus, and far below the
# 13.2 the nearest genuinely different pair sits at. See
# :meth:`Palette.witness`.
SAME_COLOUR_TOL = 2.0

# tan(ANGLE_TOL): the angular gate, as the perpendicular distance allowed
# per unit ALONG the ray, which is the form :meth:`Palette.match` needs.
_ANGLE_SLOPE = math.tan(math.radians(ANGLE_TOL))

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

    def _offset(self, vec: NDArray[np.float64], unit: NDArray[np.float64]) -> tuple[float, float]:
        """``vec`` resolved along ``unit`` and perpendicular to it."""
        along = float(vec @ unit)
        return along, float(np.sqrt(max(0.0, float(vec @ vec) - along * along)))

    def _perpendicular(self, vec: NDArray[np.float64], unit: NDArray[np.float64]) -> float:
        return self._offset(vec, unit)[1]

    def match(self, vec: NDArray[np.float64]) -> int | None:
        """Index of the class whose ray ``vec`` lies on, or ``None``.

        Near the ray in ANGLE and within :data:`LINE_TOL` of it in units,
        and on the ray's own side of the background. A colour that fails
        either becomes a class of its own, which is the safe direction to
        err in: a class too many costs a cold start on one colour, and a
        class too few names every later piece of a colour after the first
        piece that ever wore it.
        """
        best, best_dist = None, LINE_TOL
        for index, klass in enumerate(self.classes):
            along, perp = self._offset(vec, klass.unit)
            if along <= 0.0:
                continue  # the far side of the background is a different colour
            if perp < best_dist and perp <= along * _ANGLE_SLOPE:
                best, best_dist = index, perp
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

    def witness(self, index: int, piece: str, vec: NDArray[np.float64] | None = None) -> int:
        """Record a COMPLETE four-cell sighting of class ``index``.

        The strongest naming evidence there is, and the only evidence that
        can contradict the palette. Returns the class that owns the
        sighting afterwards, which is ``index`` unless the contradiction
        was resolved by splitting.

        A contradiction has two possible causes and they want opposite
        answers. Either this colour really does not determine the piece --
        a monochrome theme, two tetrominoes the game renders alike, a piece
        recoloured by level -- and the class must stop naming anything, or
        two DIFFERENT colours were merged by :meth:`match`'s tolerance and
        the class must come apart.

        ``vec``, the colour actually sighted, is what tells them apart. A
        colour the game really uses for two pieces lands on the class's ray
        exactly: a single rendered colour's spread over the whole corpus is
        0.90 units. One that merely passed the gate sits measurably off it,
        and splitting is then strictly better than retiring -- retiring
        costs BOTH colours their names for the rest of the session, which
        is the shipped tracker's spawn latency twice over, and it is the
        mirror pairs (S/Z, J/L) that a merge is likeliest to hit, which is
        exactly where shape cannot help an entering piece.

        Retiring is still a one-way door, and deliberately a narrow one:
        what it costs when it fires wrongly is naming by shape, and what it
        costs when it does not fire is every later piece of that colour
        named wrong for the rest of the session.
        """
        klass = self.classes[index]
        if klass.ambiguous:
            return index
        if klass.piece is not None and klass.piece != piece:
            if vec is not None and self._perpendicular(vec, klass.unit) > SAME_COLOUR_TOL:
                magnitude = float(np.linalg.norm(vec))
                self.classes.append(ColourClass(unit=vec / magnitude, peak=magnitude, piece=piece))
                return len(self.classes) - 1
            klass.ambiguous = True
            klass.piece = None
            return index
        klass.piece = piece
        return index

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
                vec = self.vector(colours[r, c], state)
                if float(np.linalg.norm(vec)) < EMPTY_DIST:
                    continue
                labels[r, c] = self.intern(vec)
        return labels

    def vector(self, colour: NDArray[np.float32], state: int = CLEAN) -> NDArray[np.float64]:
        """One cell's colour as a vector away from the background.

        What :meth:`classify` compares, exposed because a caller that wants
        to know WHICH colour a cell was (rather than which class it fell
        into) must ask the same question the same way -- see
        :meth:`witness`, which decides whether a contradiction is one
        colour or two.
        """
        if self.background is None:
            raise ValueError("background not estimated yet")
        vec = colour.astype(np.float64) - self.background
        return self._unpainted(vec) if state == PAINTED else vec


def board_colours(image: NDArray[np.uint8], rows: int, cols: int) -> NDArray[np.float32]:
    """Mean colour of every cell's central patch: ``(rows, cols, 3)``."""
    return cell_colors(image, rows, cols, CELL_MARGIN)


def off_ground(
    colours: NDArray[np.float32],
    background: NDArray[np.float64],
    mask: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    """Cells too far from the background to BE it, too near to be content.

    Neither of :meth:`~.colour_tracker.ColourTracker._blind`'s other
    premises can see a cover drawn in a colour NEAR the board's ground,
    and that is the one this game actually draws. A cell the card covers
    is not "content resting on nothing" -- it is nothing, so the airborne
    premise has nothing to count, and the card is flat, so the flatness
    premise is happy. What gives it away is that the board's own empty
    cells do not read like that: they read as the background exactly.

    Measured end to end before this existed, ROAS Stacker's "ROW CLEARED"
    card over a 16-cell stack gave ``accepted=True`` and a stack of ZERO,
    and the coach went on drawing a placement computed on a board it
    believed was empty -- with the frame accepted, the stale-frame counter
    reset every frame and the 45-frame withdrawal that exists for exactly
    this popup never fired.

    ``mask`` should be the observable cells this tool has not painted:
    our own fill shifts a cell about 55 units and would land half the
    board in the band.
    """
    dist = np.linalg.norm(colours.astype(np.float64) - background, axis=2)
    return mask & (dist >= GROUND_TOL) & (dist < EMPTY_DIST)


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

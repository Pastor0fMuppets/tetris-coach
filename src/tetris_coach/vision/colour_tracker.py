"""A falling-piece tracker with per-cell state and no global commit.

THE DEFAULT READER since the race in :mod:`tetris_coach.race`: ``app.py``
reads frames with this unless ``--tracker shape`` asks for the other one,
which is still there, still tested, and still answers the same questions
from a different representation. What the two put on screen through the
real engine, over the eight committed windows and 474 frames, is pinned in
``tests/test_engine_race.py``: no hint for the wrong piece against seven, no
wait from a piece appearing to its hint against one to three frames, no
target moving mid-flight against three, 4 blank frames against 32.

The older design keeps ONE authoritative memory — the committed stack —
and derives the falling piece as a set difference against it. Every error is
therefore permanent until something explains it away, which is what produced
the absorbed-piece loop, the phantom locks and the 25-frame freeze the user
reported, and what the resync/reset/heal/retract machinery exists to paper
over. This one keeps no such memory. Each frame is read on its own:

    content   = cells whose colour is a real rendered colour (colour_palette)
    falling   = a component of one colour, at most a tetromino, that is
                FLOATING (no chain of content joins it to the floor) or has
                CHANGED in the last few frames
    stack     = all the other content
    lock      = the piece that was falling last frame is still there and
                something else has become the better candidate
    clear     = the stack lost a row's worth of cells at once

What memory there is, stated exactly, because the claim this design is sold
on is about memory and the first version of this docstring overstated it:

* Per cell: the colour class it showed last frame and how many frames it has
  shown it. Bounded and self-healing — a cell that is misread is read again
  next frame and its age restarts.
* Last frame's report (the piece, the stack and its cell count), used ONLY to
  name transitions: lock, spawn, clear. Nothing is derived from it.
* The board background, re-estimated every frame from the cells currently
  reading empty. It moves in both directions and forgets.
* The colour->piece palette. THIS ONE IS GLOBAL AND IT IS ONE-WAY across
  the frames it accepts: a class is never removed, its peak magnitude never
  falls, and a name retired as ambiguous never comes back. The only thing
  that ever moves backwards is a REFUSED frame, which is rolled back whole
  (:meth:`Palette.checkpoint`) to a state the palette really held.

So the reading is memoryless where it matters — the board handed to the
solver is derived from this frame's pixels alone, never from a committed
stack, which is why no error survives into the next frame and there is no
state a reset would have to rebuild. But "a misread frame costs exactly
that frame" is not true of the palette, and the palette's teeth have been
pulled rather than its memory removed: peak no longer decides whether a
cell is content (see :mod:`.colour_palette` on the ghost rule it used to
serve), and a frame that is not a board interns nothing that lasts — either
it is refused before it is read at all, or, where saying so needs the cells
labelled first, the whole read is rolled back (see
:meth:`ColourTracker._blind`). What remains is that a stray colour on a real
board frame becomes a class for the rest of the session. It costs a class,
not a piece.

Ordering is by evidence, not by history: a floating component is the falling
piece however long it has hovered (this game has no gravity — a piece sits
at the top edge until the player drags it), and a piece that has come to
rest stays the piece in flight until another candidate takes its place,
which is the frame the lock is reported. Time alone does not settle a piece,
because in a game with no gravity a piece parked on the stack is one the
player is still holding. Naming is
by colour from the FIRST VISIBLE CELL, so a piece clipped by the top edge is
named the frame it appears. Shape is consulted on every COMPLETE four-cell
sighting: it names a colour the palette has not seen, and it is allowed to
contradict a colour the palette has named. A contradiction means one of two
things and the sighting's own colour says which. Dead on the class's ray, it
is one colour the game draws two pieces in — a monochrome theme, two
tetrominoes rendered alike — and the class is retired from naming, so the
reading degrades to naming by shape. Measurably off it, it is two colours
the matching tolerance merged, and the class comes apart instead: retiring
there would cost BOTH colours their names for the rest of the session.

What NEITHER signal covers, said plainly because it is the one hole in the
naming: a piece whose colour is unknown AND whose sighting is incomplete.
Shape cannot name 1-3 cells (they fit several tetrominoes — that is the
whole reason this design exists), so such a piece is tracked and reported
with ``piece=None`` until it is whole or the NEXT box names its colour. It
costs a cold start per colour and nothing after it: measured over the eight
committed windows, 15 frames in all carry a piece in flight with no name —
14 of them absorbed_piece's opening T, unnamed until it is whole enough to
spell itself because the NEXT box never shows it in that window, and 1
ghost_session's first frame. Every other frame of every window names its
piece, the J and the Z the two newest windows brought included.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
from numpy.typing import NDArray

from ..core.board import DEFAULT_HEIGHT, WIDTH
from ..core.pieces import ROTATIONS
from .colour_palette import (
    CLEAN,
    EMPTY,
    EMPTY_DIST,
    OURS,
    Palette,
    board_colours,
    board_readable,
    observable_mask,
    off_ground,
    own_paint_states,
    piece_from_cells,
)
from .colour_preview import identify_preview
from .grid import HINT_PAINT, OwnPaint

Cell = tuple[int, int]

# Frames a cell must hold its colour before it counts as settled board.
SETTLE_FRAMES = 3

# Consecutive frames the piece in flight may be believed on its HISTORY
# rather than on what this frame shows -- see :meth:`ColourTracker._rank`
# on the exemption and :data:`PIECE_CELLS` on what it is an exemption
# from. One, because the thing it excuses is one frame long: a cell that
# samples the wrong side of a colour boundary reads right again next
# frame, and a cell that reads background twice running is not flickering.
# Measured over all nine committed windows, no frame needs even one.
LOAN_FRAMES = 1

# A tetromino. Also the budget for any ONE thing hanging over the void: a
# Tetris board has one piece in flight and everything else is held up by
# what is under it. See :meth:`ColourTracker._blind`.
PIECE_CELLS = 4

# ...and the budget for everything airborne added together, which is looser
# on purpose. A frame is not a board when it shows a SLAB resting on
# nothing, and that is what the per-component budget above is for; refusing
# a frame because one stray cell sits beside a real piece is a cliff with
# nothing at the bottom of it. It had one: measured on the 8 committed live
# ROAS Stacker frames, a board rectangle that leaves the NEXT panel
# covering two cells the mask does not name reads 6 airborne on every
# single frame against a budget of exactly 4, so every frame of the session
# was refused and the coach never spoke, silently. The largest airborne
# reading anywhere in the committed corpus is 4, in one component.
AIRBORNE_CELLS = 2 * PIECE_CELLS


class Event(Enum):
    """What changed on the board this frame."""

    PIECE_SPAWNED = auto()
    PIECE_LOCKED = auto()
    LINES_CLEARED = auto()


@dataclass(frozen=True)
class FallingPiece:
    """The piece in flight, as observed this frame."""

    piece: str | None  # None while the palette has no name for this colour
    cells: frozenset[Cell]  # only the cells the capture can SEE
    colour_class: int
    floating: bool  # nothing joins it to the floor: certainly in flight

    @property
    def complete(self) -> bool:
        """Are all four cells of the piece on screen?"""
        return len(self.cells) == 4


@dataclass(frozen=True)
class FrameReport:
    """Everything the tracker has to say about one frame."""

    falling: FallingPiece | None
    stack_rows: tuple[int, ...]  # one bitmask per row, bit c = column c
    next_piece: str | None
    events: tuple[Event, ...]
    cleared_rows: int
    board_visible: bool = True  # False when the frame is not a board at all
    #: Why the frame was refused, in one phrase, or ``None`` when it was
    #: read. A refusal is silence, and silence is the hardest thing to
    #: diagnose in a live session -- the user sees a coach that has stopped
    #: talking and nothing that says why.
    refused_because: str | None = None


def _components(labels: NDArray[np.int16]) -> list[tuple[int, frozenset[Cell]]]:
    """Four-connected runs of one colour class."""
    rows, cols = labels.shape
    seen = np.zeros((rows, cols), dtype=bool)
    out: list[tuple[int, frozenset[Cell]]] = []
    for r in range(rows):
        for c in range(cols):
            if seen[r, c] or labels[r, c] == EMPTY:
                continue
            label = int(labels[r, c])
            queue = deque([(r, c)])
            seen[r, c] = True
            cells: list[Cell] = []
            while queue:
                cr, cc = queue.popleft()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    inside = 0 <= nr < rows and 0 <= nc < cols
                    if inside and not seen[nr, nc] and labels[nr, nc] == label:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            out.append((label, frozenset(cells)))
    return out


def _connected(cells: frozenset[Cell]) -> list[frozenset[Cell]]:
    """Split a set of cells into its four-connected parts."""
    remaining = set(cells)
    parts: list[frozenset[Cell]] = []
    while remaining:
        queue = deque([remaining.pop()])
        part = []
        while queue:
            cr, cc = queue.popleft()
            part.append((cr, cc))
            for step in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                if step in remaining:
                    remaining.discard(step)
                    queue.append(step)
        parts.append(frozenset(part))
    return parts


def _completed_row(labels: NDArray[np.int16]) -> int | None:
    """The first row that is content edge to edge, or ``None``.

    A completed row is never a resting state of a Tetris board: the game
    takes it away. Seeing one means the clear is mid-animation, and what
    the animation does to the pixels (this game recolours the row, then
    flashes it) is not the board. A row holding a cell the capture cannot
    see can never read as complete, which is the honest answer there --
    the panel that hides it could be hiding a gap.

    The same fact the shipped classifier's ``_claims_a_completed_row``
    rests on (``grid.py``) and the same one the oracle abstains on
    (``truth/oracle.py``); this reader was the only one of the three not
    using it.
    """
    full = np.flatnonzero(np.all(labels != EMPTY, axis=1))
    return int(full[0]) if full.size else None


def _grounded(labels: NDArray[np.int16]) -> NDArray[np.bool_]:
    """Content cells joined to the floor by a chain of content of ANY colour.

    Colour-blind on purpose: what holds a cell up is whatever is under it,
    not whatever matches it. Sideways links are part of the chain because a
    real stack hangs over its own holes — after a clear, settled cells sit
    above empty ones, and a per-column test would call them airborne.
    """
    rows, cols = labels.shape
    content = labels != EMPTY
    grounded = np.zeros((rows, cols), dtype=bool)
    queue: deque[Cell] = deque()
    for c in range(cols):
        if content[rows - 1, c]:
            grounded[rows - 1, c] = True
            queue.append((rows - 1, c))
    while queue:
        cr, cc = queue.popleft()
        for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
            if 0 <= nr < rows and 0 <= nc < cols and content[nr, nc] and not grounded[nr, nc]:
                grounded[nr, nc] = True
                queue.append((nr, nc))
    return grounded


class ColourTracker:
    """Frames in, one :class:`FrameReport` out per frame."""

    def __init__(
        self,
        rows: int = DEFAULT_HEIGHT,
        cols: int = WIDTH,
        unobservable_cells: frozenset[Cell] | None = None,
        settle_frames: int = SETTLE_FRAMES,
        paint: OwnPaint | None = HINT_PAINT,
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.settle_frames = settle_frames
        self._paint = paint
        self.palette = Palette(paint=paint)
        self._observable = observable_mask(rows, cols, unobservable_cells)
        self._labels: NDArray[np.int16] | None = None
        # Frames each cell has held its current colour. A cell starts
        # SETTLED: on the first frame there is no motion to have seen, and
        # what is on the board has been there for all of the history we have.
        self._age = np.full((rows, cols), settle_frames, dtype=np.int32)
        self._falling: FallingPiece | None = None
        # Consecutive frames the piece in flight has been believed on its
        # history rather than on its own sighting (:data:`LOAN_FRAMES`).
        self._on_loan = 0
        self._stack_rows: tuple[int, ...] = (0,) * rows
        self._stack_count = 0
        self._next_piece: str | None = None

    # -- the frame ----------------------------------------------------

    def update(
        self, board: NDArray[np.uint8], next_crop: NDArray[np.uint8] | None = None
    ) -> FrameReport:
        """Read one captured board (BGR) and its NEXT box crop."""
        if not board_readable(board, self.rows, self.cols, self._observable):
            return self._blind("the frame is not drawn as flat cells")
        # Reading the frame writes to the palette, and the premises below
        # cannot be tested until the cells are labelled, so the whole read
        # is taken back if it turns out not to have been a board.
        mark = self.palette.checkpoint()
        colours = board_colours(board, self.rows, self.cols)
        painted = own_paint_states(board, self.rows, self.cols, self._paint)
        clean = self._observable & (painted == CLEAN)
        self.palette.update_background(colours, clean)
        background = self.palette.background
        if background is not None:
            band = int(np.count_nonzero(off_ground(colours, background, clean)))
            if band > PIECE_CELLS:
                self.palette.restore(mark)
                return self._blind("cells are neither the board's ground nor content")
        labels = self.palette.classify(colours, self._observable, painted)
        grounded = _grounded(labels)
        airborne = np.where(grounded, EMPTY, labels).astype(np.int16)
        if int(np.count_nonzero(airborne != EMPTY)) > AIRBORNE_CELLS or any(
            len(cells) > PIECE_CELLS for _, cells in _components(airborne)
        ):
            self.palette.restore(mark)
            return self._blind("more than a tetromino is resting on nothing")
        if _completed_row(labels) is not None:
            self.palette.restore(mark)
            return self._blind("a completed row is still on screen")
        # The box is read before the board's cells are NAMED, so a colour
        # the box names this frame already names the piece entering on it.
        # After the gate, so that a frame the gate refuses writes nothing
        # here either -- the box is a different region of the screen, but a
        # reading kept off a frame the palette was rolled back from would
        # name a colour class that no longer exists.
        if next_crop is not None:
            self._read_preview(next_crop)

        if self._labels is not None:
            self._age = np.where(labels != self._labels, 0, self._age + 1)
        self._labels = labels

        # Cells nothing can be read in: off the capture (the NEXT panel,
        # named once at construction) and under our own opaque mark, which
        # moves with the hint and so is this frame's alone.
        covered = ~self._observable | (painted == OURS)
        falling = self._pick_falling(labels, grounded, covered)
        # What the next frame's exemption is allowed to rest on: a piece
        # read off this frame alone repays the loan, one read on credit
        # extends it, and no piece at all ends it.
        self._on_loan = (
            self._on_loan + 1
            if falling is not None and not self._explicable(falling.cells, covered)
            else 0
        )
        if falling is not None:
            falling = self._named(falling, colours, painted)

        stack_rows, stack_count = self._stack(labels, falling)
        events, cleared = self._events(falling, stack_rows, stack_count)

        self._falling = falling
        self._stack_rows = stack_rows
        self._stack_count = stack_count
        return FrameReport(
            falling=falling,
            stack_rows=stack_rows,
            next_piece=self._next_piece,
            events=events,
            cleared_rows=cleared,
        )

    def _blind(self, reason: str) -> FrameReport:
        """The frame cannot be read: take nothing from it, and say so.

        Without this the tracker read whatever was on screen. Measured, the
        fourteen frames of ``pale_piece`` where a web page has replaced the
        game were reported as a stack of 0b11111111 / 0b1000 / 0b11111100
        -- 57 cells of furniture -- and interned six junk colour classes
        into the palette PERMANENTLY: a palette class never shrinks, so one
        occluded frame is a lasting corruption of the one memory this
        design does keep.

        FOUR PREMISES have to hold, and they fail on different frames.

        (1) The board is drawn as flat cells (:func:`board_readable`). That
        is this module's own premise rather than anything about Tetris --
        every rule here reads a cell's MEAN, which means nothing over a
        photograph or a paragraph of text. It refuses the web page (0.51 of
        cells flat against 0.89 or better on every real frame of all six
        windows) and it is blind by construction to a cover that IS flat.

        (2) Nothing bigger than a tetromino hangs OVER THE VOID, and not
        much more than one does in total: everything else on a Tetris board
        is held up by what is under it, sideways links included
        (:func:`_grounded`), and there is only ever one piece in flight.
        This is the premise the shipped engine's confidence gate rests on
        too (SPEC.md, grid.py), and it catches what (1) cannot: a flat
        panel floating over the playfield -- a leaderboard, any modal drawn
        in one colour -- arrives as dozens of cells resting on nothing.
        Measured, it costs the corpus NOTHING: over all accepted board
        frames of the eight windows the largest airborne reading is 4
        cells, exactly one tetromino, never once more; the web page, fed
        past (1), shows 58 in a single component.

        The two budgets are separate because they answer different
        questions and a single one cannot do both. What says "not a board"
        is a SLAB resting on nothing, which :data:`PIECE_CELLS` catches per
        component. What a total of exactly 4 also caught was one stray cell
        beside a real piece -- and a couple of stray cells is what a board
        rectangle a few pixels off produces, which made a slightly wrong
        selection refuse every frame of the session in silence rather than
        cost a little accuracy (see
        :func:`~tetris_coach.app.compute_overlap_mask`, which is where that
        is properly fixed). :data:`AIRBORNE_CELLS` is the slack that turns
        the cliff into a slope.

        (3) Every cell is either the board's own ground or content
        (:func:`off_ground`). The two premises above are both blind to a
        cover drawn in a colour NEAR the background, and that is the one
        this game actually draws: the oracle measures its "ROW CLEARED"
        card at 7 units off the ground against a content floor of 12, so
        the card is not content resting on nothing, it is nothing.
        Measured end to end before this premise existed, that card over a
        16-cell stack was ACCEPTED with a stack of zero, and because the
        frame was accepted the stale-frame counter was reset on every one
        of them and the 45-frame withdrawal added for exactly this popup
        never fired. What gives the card away is that the empty board does
        not read "near" the background, it reads AS the background: over
        the whole corpus, not one unpainted observable cell sits even 1.0
        away from it.

        (4) No row is COMPLETE. A finished row does not stay on a Tetris
        board -- it is taken away -- so a full row on screen means the
        clear is still playing, and this game recolours the row while it
        does. Nothing read off those frames is the board: the recoloured
        cells count as having just changed, which is the evidence this
        tracker names a piece from, so 1-2 cells of the row outranked
        everything and became "the piece". Measured on ``spawn_latency``
        before this premise existed, the coach drew ``I`` over four of the
        seven flash frames while the player held an O, and raised
        PIECE_SPAWNED/PIECE_LOCKED into the bargain. This is the oracle's
        own rule (``truth/oracle.py``, "line-clear-animation") and over the
        whole corpus it fires on exactly the seven frames the oracle
        abstains from for that reason and on no others. A banner drawn
        across the board's full width is refused by the same test, which is
        the one thing that catches a flat card that reaches the floor.

        What is NOT covered, stated so nobody has to find out live: a
        cover that is flat, drawn well clear of the background, narrower
        than the board, and grounded -- one that reaches the floor or
        touches the stack -- passes all four, and reads as a slab of
        phantom stack. Measured on the same card lifted just past the
        content floor, at 12.1 units: 14 phantom cells and the hint moves
        three rows up the board. What saves the real card is that it spans
        the board's full width and (4) refuses it; a narrower one would
        need pixels this corpus does not have to close properly.

        Nothing is read and nothing is written: not the palette, not the
        background, not the NEXT box. (1) refuses the frame before any of
        it; (3) needs the background, and (2) and (4) the labelled cells,
        so the caller takes a :meth:`Palette.checkpoint` first and restores
        it here. The stack count IS carried across, so the frame the board
        comes back does not look like a line clear to :meth:`_events`.

        The per-cell history is FORGOTTEN rather than carried, for the
        reason the constructor starts every cell settled: a cell's age is
        evidence of MOTION, and across a gap there was no motion to have
        seen. Carrying it made every cell that changed behind the cover
        read as having just changed, and the stack that grew while the
        board was hidden then outranked the real piece -- measured on a
        cover held for twenty frames, the frame the board came back
        reported three cells of settled stack as the piece in flight, with
        a PIECE_SPAWNED to go with it, and handed the solver a board those
        three cells were missing from. Forgetting costs the opposite and
        smaller error: a piece RESTING on the stack when the board comes
        back reads as stack until the player moves it (a floating one is
        picked up at once, since floating needs no history), which is
        exactly what a mid-session attach costs and for the same reason.

        What is reported is silence, not the last hint. A tracker that
        holds its advice over a board it cannot see is the frozen coach
        this design exists to avoid, so the honest output for a frame that
        says nothing is nothing -- and ``board_visible`` lets a caller that
        would rather hold make that choice itself.
        """
        self._falling = None
        self._on_loan = 0
        self._labels = None
        self._age[:] = self.settle_frames
        return FrameReport(
            falling=None,
            stack_rows=self._stack_rows,
            next_piece=self._next_piece,
            events=(),
            cleared_rows=0,
            board_visible=False,
            refused_because=reason,
        )

    # -- pieces -------------------------------------------------------

    def _named(
        self,
        falling: FallingPiece,
        colours: NDArray[np.float32],
        painted: NDArray[np.int8],
    ) -> FallingPiece:
        """Attach a name, letting a complete sighting overrule the palette.

        Shape is consulted on EVERY complete four-cell sighting, not only
        while the colour is anonymous. A complete sighting is the one piece
        of evidence that can contradict the palette, and a palette that
        cannot be contradicted is a palette that is right forever after the
        first piece -- which on a monochrome theme means naming every later
        piece after the first one.

        The sighting's own colour goes with it, because what the palette
        does about a contradiction depends on it: a colour the game really
        draws two pieces in is retired from naming and shape takes over,
        but two colours this class merged are split apart instead (see
        :meth:`Palette.witness`). Either way the class that comes back is
        the one that owns this sighting, and it is the one the name is read
        off.
        """
        shape = piece_from_cells(falling.cells)
        label = falling.colour_class
        if shape is not None:
            label = self.palette.witness(label, shape, self._sighted(falling, colours, painted))
        named = self.palette.piece_of(label)
        if named is None:
            named = shape  # the colour says nothing; this frame still can
        if named == falling.piece:
            return falling
        return FallingPiece(named, falling.cells, falling.colour_class, falling.floating)

    def _sighted(
        self,
        falling: FallingPiece,
        colours: NDArray[np.float32],
        painted: NDArray[np.int8],
    ) -> NDArray[np.float64] | None:
        """The colour this piece was actually drawn in, as one vector.

        The mean over its cells, which is the cells' common value: a piece
        of this game is drawn in one flat colour and its cells sample to
        the same value to the unit. Cells under this tool's own paint are
        un-composited first, like everywhere else.
        """
        vectors = [self.palette.vector(colours[r, c], int(painted[r, c])) for r, c in falling.cells]
        if not vectors:
            return None
        return np.mean(np.stack(vectors), axis=0)

    def _read_preview(self, crop: NDArray[np.uint8]) -> None:
        """Name the NEXT piece, and teach the palette its colour.

        The box is read with THIS session's paint, not the default one:
        the NEXT panel floats over the top corner of the playfield in the
        game this is used on, so a hint drawn in that corner is drawn over
        the box, and a reader that does not know what the paint looks like
        reads a tetromino of our own colour as the piece being dealt.
        """
        reading = identify_preview(crop, self._paint)
        if reading is None:
            return  # an unreadable box says nothing; the last reading stands
        self._next_piece = reading.piece
        background = self.palette.background
        if background is None:
            return
        if float(np.linalg.norm(reading.background - background)) >= EMPTY_DIST:
            return  # the box is not painted on the board's ground: no link
        vector = reading.colour - background
        if float(np.linalg.norm(vector)) < EMPTY_DIST:
            return
        self.palette.name(self.palette.intern(vector), reading.piece)

    def _candidates(self, label: int, cells: frozenset[Cell]) -> list[frozenset[Cell]]:
        """The cell-sets within one colour component that could be a piece.

        Usually the component itself. But a piece touching settled cells of
        its OWN colour is one component with the stack it landed on, and
        colour cannot separate them because it is the same colour -- which
        is the absorbed-piece failure this design was meant to leave
        behind, arriving by a different road. Verified before the fix: an I
        landing beside a settled I was reported as no piece at all and
        silently swallowed into the stack, with no event raised, and an I
        descending onto one vanished on the frame it came to rest.

        What separates them is not colour but TIME, which this tracker
        already keeps per cell: the arriving piece changed in the last few
        frames and what it landed on did not. So an oversized component is
        split on age, and each young part is offered on its own. When the
        piece has held still for ``settle_frames`` its cells age out, no
        young part remains, and the whole thing is stack -- which is the
        settling this design already promised, now reached by pieces that
        happen to match what is under them.
        """
        if len(cells) <= 4:
            return [cells]
        young = frozenset(
            cell for cell in cells if int(self._age[cell[0], cell[1]]) < self.settle_frames
        )
        parts = [part for part in _connected(young) if len(part) <= 4]
        # The piece the player is holding still, inside a component it has
        # aged into: young says nothing about it any more, and
        # :meth:`_rank` is where it is decided on. What of it this frame
        # shows, rather than all of it, for :meth:`_is_held`'s reason --
        # one cell flickering at a colour boundary must not cost the
        # player the piece they are holding.
        held = self._falling  # what was in flight last frame
        if held is not None and held.colour_class == label:
            still = held.cells & cells
            if len(held.cells - still) <= 1 and still and still not in parts:
                parts.append(still)
        return parts

    def _is_held(self, label: int, cells: frozenset[Cell]) -> bool:
        """Is this exactly the piece that was in flight last frame, unmoved?

        THIS GAME HAS NO GRAVITY. A piece the player has dragged onto the
        stack sits where it was put until they drop it, and holding it over
        the landing spot while reading the advice is how the game is
        played. To the pixels that is indistinguishable from a piece that
        has locked -- both are content, resting, and not changing -- so the
        tracker has to choose which to believe, and it used to choose lock
        after ``settle_frames``. Measured through the engine at 15 fps, an
        O dragged onto the floor kept its hint for 3 frames and then went
        blank from the 4th (0.27 s), which is the moment the player is
        looking at it.

        So a piece that is already in flight KEEPS being in flight while
        nothing about it changes. The evidence that it has locked is not
        time passing, it is another piece: a candidate that floats, or one
        that has just changed, outranks a held one by the ordinary rules in
        :meth:`_rank`, and the frame it does is the frame the lock is
        reported. That is the real signal in a game that deals the next
        piece the moment you drop this one.

        What it costs when the next piece is slow to appear: the held piece
        is left out of the settled board for those frames, so the solver is
        handed a board without it and the hint on screen is the one for the
        piece that has just landed. Both are what the player sees anyway
        while they hold it, which is the case this is for; the tracker
        cannot tell the two apart and this is the side worth erring on.

        ONE CELL OF SLACK, and only of the right kind. ``==`` made the
        identity of a parked piece depend on all four of its cells being
        read the same way twice running, so a single cell flickering at a
        colour boundary -- an antialiased edge sampling a shade either
        side of :data:`EMPTY_DIST`, which is the most ordinary noise there
        is -- reclassified a piece the player was still holding as settled
        board. On this game that path is hit constantly, because parking a
        piece over the landing spot while reading the advice is how it is
        played.

        So a sighting keeps the identity when it differs from last
        frame's by at most one cell, GAINED or LOST but never moved: one
        set contains the other. That is what a flickering cell looks like
        and it is not what motion looks like -- every translation of every
        tetromino both adds and removes cells, an I stepped one column
        included, so no moved piece can inherit an identity through this.
        A piece that grows a cell is not at risk of being called settled
        anyway: the new cell's age is 0, and :meth:`_rank` only consults
        this once every cell has aged past ``settle_frames``.
        """
        was = self._falling
        if was is None or was.colour_class != label:
            return False
        return (was.cells <= cells or cells <= was.cells) and len(was.cells ^ cells) <= 1

    def _continues(self, label: int, cells: frozenset[Cell]) -> bool:
        """Is this the piece that was in flight last frame, moved or not?

        The looser of the two continuity questions, and the one the
        admission test in :meth:`_rank` is excused by. :meth:`_is_held`
        asks whether a piece has STOOD STILL, which is what decides
        whether it is board or still in hand; this asks only whether it is
        the same piece, which is what decides whether a cell it is not
        showing this frame is news.

        The difference is the player's hand. A cell dropping to noise for
        one frame was made fatal for a piece being DRAGGED -- it fails
        :meth:`_is_held` because its cells moved, then fails
        :meth:`_explicable` because three cells in open board have nothing
        hiding the fourth -- so the candidate was dropped entirely: the
        three cells it did show went into the settled stack, the lock and
        spawn fired, and the hint came off the screen and re-solved. That
        is the user's reported stutter, one frame at a time, and a
        flickering cell is no rarer during a drag than while parked.

        Same colour class, and at least one cell in common with last
        frame's sighting. Overlap is what says "this one and not another":
        this game is drag-to-drop and a piece moves a cell or two between
        captures, which always leaves cells in common, while the stray
        that stole the hint at (8, 8) shared none with the J at the top
        edge. A piece that jumps clear of where it was gets no continuity
        from this and is read on its own merits, which is right -- across
        a jump we did not see, what is at the new place is a new sighting.
        """
        was = self._falling
        if was is None or was.colour_class != label:
            return False
        return bool(was.cells & cells)

    def _hidden(self, r: int, c: int, covered: NDArray[np.bool_]) -> bool:
        """Is this cell one the capture cannot see a piece in?

        Three places. ABOVE ROW 0: this game deals a piece at the ceiling
        and the player drags it down, so a piece entering shows its bottom
        row or two and the rest is off the top of the board region. UNDER
        THE NEXT PANEL: the panel overlaps the board rectangle's top-right
        corner in every session captured here, and those cells are named
        at construction. UNDER OUR OWN OPAQUE PAINT: the coach draws the
        hint over the board it is reading, and where that drawing is
        opaque rather than a translucent fill it is not un-composited but
        SKIPPED (:func:`~.colour_palette.own_paint_states`), so the cell
        arrives here as EMPTY whatever the game drew in it.

        The third one is this tool's own doing, which is why it was missed:
        the first two are properties of the capture and are fixed for a
        session, and ``covered`` is this frame's, because where we paint
        moves with the hint. Leaving it out made a piece cell under our own
        mark indistinguishable from a cell that is not there -- measured,
        an O dragged under the rotation badge lost its fourth cell, failed
        the admission test, went into the settled stack and raised
        PIECE_LOCKED, and the badge follows the hint so the next frame did
        it again. That is the loop ``tests/fixtures/hint_stutter``
        documents (coach paints, reader misreads, hint moves, paint
        moves), pointing the other way: it deletes the piece rather than
        inventing one.

        Off the sides and below the floor are NOT hidden: the well has
        walls and a floor, and a piece is never partly through them.
        """
        if r < 0:
            return True
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            return False
        return bool(covered[r, c])

    def _explicable(self, cells: frozenset[Cell], covered: NDArray[np.bool_]) -> bool:
        """Is this sighting one a whole tetromino could be behind?

        A piece has four cells. Seeing fewer is only explicable where
        something HIDES the rest, and the places anything can hide are
        :meth:`_hidden`'s. So a sub-tetromino candidate is admissible when
        some placement of some rotation covers it and puts every cell it
        does not account for in one of those places -- and nowhere else. A
        lone cell in open board is not a piece with three cells missing;
        it is a cell.

        This is the rule the shipped shape tracker enforced and the
        rewrite dropped, and it is restored here as a BOUND rather than as
        a fix: measured over all nine committed windows, every one of the
        170 sub-tetromino sightings the tracker picks is explicable, so
        this changes not one frame of the corpus today. What it changes is
        what a stray cell can cost. The coach's own rotation badge, read
        as content by a reader that has since been fixed
        (:func:`~tetris_coach.vision.colour_palette.own_paint_states`),
        was a single cell at row 8 of open board, and it took the hint off
        the real J on every other frame for 44 frames; a noise source
        nobody has met yet would have done the same. Two independent rules
        now have to fail together for that to happen again.

        WHICH WAY IT ERRS: toward refusing a partial sighting. A piece
        genuinely hidden by something this tracker does not model -- a
        board rectangle cutting the playfield short, a game's own overlay
        that is not the NEXT panel -- is read as stack rather than as a
        piece, so the coach hints on a board with an extra blob in it
        instead of hinting for an invented piece. Both are wrong; the
        second is wrong in a way that moves the hint every frame.
        """
        missing = PIECE_CELLS - len(cells)
        if missing <= 0:
            return True
        anchor = min(cells)
        for rotations in ROTATIONS.values():
            for rotation in rotations:
                for ar, ac in rotation.cells:
                    dr, dc = anchor[0] - ar, anchor[1] - ac
                    placed = {(r + dr, c + dc) for r, c in rotation.cells}
                    if not cells <= placed:
                        continue
                    if all(self._hidden(r, c, covered) for r, c in placed - cells):
                        return True
        return False

    def _pick_falling(
        self,
        labels: NDArray[np.int16],
        grounded: NDArray[np.bool_],
        covered: NDArray[np.bool_],
    ) -> FallingPiece | None:
        """The one component that is in flight, if any."""
        best: tuple[tuple[int, int, int, int], FallingPiece] | None = None
        for label, component in _components(labels):
            for cells in self._candidates(label, component):
                best = self._rank(best, label, cells, grounded, covered)
        return None if best is None else best[1]

    def _rank(
        self,
        best: tuple[tuple[int, int, int, int], FallingPiece] | None,
        label: int,
        cells: frozenset[Cell],
        grounded: NDArray[np.bool_],
        covered: NDArray[np.bool_],
    ) -> tuple[tuple[int, int, int, int], FallingPiece] | None:
        """Keep whichever of ``best`` and ``cells`` is the better candidate.

        Anything beats a LONE CELL; then floating beats resting, younger
        beats older, higher beats lower -- among candidates that could be
        a piece at all (:meth:`_explicable`).

        A SINGLE CELL IS THE LEAST EVIDENCE THERE IS, and it must not
        displace more. :meth:`_explicable` bounds where a lone cell may be
        read as a piece at all, but the places it leaves are the whole of
        row 0 and the edge of the panel -- 11 of this session's 120 cells
        -- because a vertical I entering really does show exactly one
        cell, and that is precisely where this game deals its pieces and
        where the panel's own boundary bleeds. A stray up there is
        floating, and floating sorted first, so one noise cell at (0, 7)
        outranked a whole O the player had parked: measured on synthetic
        frames, falling went O -> the stray -> O with the O's four cells
        dropping into the stack and a PIECE_LOCKED and PIECE_SPAWNED on
        every other frame -- the reported flashing, at 15 fps, from one
        cell.

        Demoting it costs nothing where a lone cell is the only candidate:
        the piece entering behind the panel in ``stray_after_clear`` is
        one cell for three frames (00578-00580) and is still reported on
        all three. What it costs is a frame or two of a lock, where a new
        piece shows one cell while the last one is still parked -- and
        that is the hint staying on the piece the player is holding, which
        is what they are looking at.

        The piece already in flight is exempt from that admission test, and
        the exemption is a LOAN OF ONE FRAME (:data:`LOAN_FRAMES`).
        :meth:`_explicable` asks what could be hiding the cells a sighting
        does not show, and for a piece we watched arrive the answer is on
        record: it was four cells a moment ago and one of them is
        misreading now. Applying the test to it would undo
        :meth:`_is_held` exactly where it is needed, since a parked piece
        that drops a cell to noise is a three-cell sighting in open board.

        WHAT IS EXEMPT IS THE PIECE, NOT THE POSE (:meth:`_continues`, not
        :meth:`_is_held`). A piece the player is DRAGGING has moved, so
        the stricter question answers no, and the same one-cell dropout
        then took the whole candidate away: the cells it did show went
        into the stack, PIECE_LOCKED fired, the next frame raised
        PIECE_SPAWNED, and the coach withdrew the hint and re-solved --
        the reported symptom exactly, in the one case parking does not
        cover.

        WHY THE LOAN HAS TO BE REPAID. "A candidate has to have been
        believed on an earlier frame to be exempt on this one" does not
        bound anything by itself, because the belief is handed on:
        :meth:`_is_held` asks only about the PREVIOUS frame and tolerates
        a cell gained or lost, so an identity admitted once where
        :meth:`_explicable` is generous -- row 0, where a lone cell is
        excusable -- keeps the exemption while it grows and shrinks by a
        cell a frame, and walks into open board where no admission test
        would have it. Measured on a one-cell stray stepping
        (0,7) -> (0,7)+(1,7) -> (1,7) -> (1,7)+(2,7) -> ... it was
        reported as the piece in flight at (4, 7), four rows below the
        last row it could have been admitted at, and it would have kept
        going. Every frame that rests on the exemption now counts against
        :data:`LOAN_FRAMES`, and only a sighting that passes
        :meth:`_explicable` on its own merits resets the count -- so the
        drift is bounded at one row rather than at the floor.
        """
        held = self._is_held(label, cells)
        excused = self._on_loan < LOAN_FRAMES and self._continues(label, cells)
        if not excused and not self._explicable(cells, covered):
            return best  # fewer than four cells and nothing hiding the rest
        floating = not any(grounded[r, c] for r, c in cells)
        youth = min(int(self._age[r, c]) for r, c in cells)
        if not floating and youth >= self.settle_frames and not held:
            return best  # resting and old: settled board, not a piece
        rank = (
            0 if len(cells) > 1 else 1,
            0 if floating else 1,
            youth,
            min(r for r, _ in cells),
        )
        if best is not None and rank >= best[0]:
            return best
        return (
            rank,
            FallingPiece(
                piece=self.palette.piece_of(label),
                cells=cells,
                colour_class=label,
                floating=floating,
            ),
        )

    def _stack(
        self, labels: NDArray[np.int16], falling: FallingPiece | None
    ) -> tuple[tuple[int, ...], int]:
        """Settled board: all content except the piece in flight."""
        in_flight = falling.cells if falling is not None else frozenset()
        rows = []
        count = 0
        for r in range(self.rows):
            mask = 0
            for c in range(self.cols):
                if labels[r, c] != EMPTY and (r, c) not in in_flight:
                    mask |= 1 << c
                    count += 1
            rows.append(mask)
        return tuple(rows), count

    # -- transitions --------------------------------------------------

    def _events(
        self,
        falling: FallingPiece | None,
        stack_rows: tuple[int, ...],
        stack_count: int,
    ) -> tuple[tuple[Event, ...], int]:
        events: list[Event] = []
        cleared = 0
        was = self._falling

        # A clear is the only thing that takes a row's worth of cells off the
        # board at once; a lock can only ever add four.
        lost = self._stack_count + 4 - stack_count
        if lost >= self.cols:
            cleared = max(1, min(4, round(lost / self.cols)))
            events.append(Event.LINES_CLEARED)

        locked = False
        if was is not None:
            still_flying = (
                falling is not None
                and falling.colour_class == was.colour_class
                and bool(falling.cells & was.cells)
            )
            # The piece stopped being in flight, and the board gained what it
            # was made of: either in place (it settled where it stood) or
            # somewhere else (it was hard-dropped between two captures).
            settled_here = all(bit_set(stack_rows, r, c) for r, c in was.cells)
            grew = stack_count > self._stack_count
            if not still_flying and (cleared or settled_here or grew):
                locked = True
                events.append(Event.PIECE_LOCKED)

        if falling is not None:
            fresh = was is None or locked or falling.colour_class != was.colour_class
            if fresh:
                events.append(Event.PIECE_SPAWNED)
        return tuple(events), cleared


def bit_set(rows: tuple[int, ...], r: int, c: int) -> bool:
    return 0 <= r < len(rows) and bool(rows[r] >> c & 1)

"""A falling-piece tracker with per-cell state and no global commit.

THE DEFAULT READER since the race in :mod:`tetris_coach.race`: ``app.py``
reads frames with this unless ``--tracker shape`` asks for the other one,
which is still there, still tested, and still answers the same questions
from a different representation. What the two put on screen through the
real engine, over the six committed windows and 422 frames, is pinned in
``tests/test_engine_race.py``: no hint for the wrong piece against six, no
wait from a piece appearing to its hint against one to three frames, no
target moving mid-flight against three, 7 blank frames against 25.

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
    lock      = the piece that was falling last frame is still there and is
                no longer a falling candidate
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
rest becomes stack once it has held still for ``settle_frames``. Naming is
by colour from the FIRST VISIBLE CELL, so a piece clipped by the top edge is
named the frame it appears. Shape is consulted on every COMPLETE four-cell
sighting: it names a colour the palette has not seen, and where it
contradicts a colour the palette has named, that colour is retired from
naming altogether and shape takes over for it — a monochrome theme, or two
tetrominoes a game renders alike, then degrade to naming by shape rather
than naming every later piece after the first.

What NEITHER signal covers, said plainly because it is the one hole in the
naming: a piece whose colour is unknown AND whose sighting is incomplete.
Shape cannot name 1-3 cells (they fit several tetrominoes — that is the
whole reason this design exists), so such a piece is tracked and reported
with ``piece=None`` until it is whole or the NEXT box names its colour. It
costs a cold start per colour and nothing after it: measured over the six
committed windows, 15 frames in all carry a piece in flight with no name —
14 of them absorbed_piece's opening T, unnamed until it is whole enough to
spell itself because the NEXT box never shows it in that window, and 1
ghost_session's first frame. Every other frame of every window names its
piece.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
from numpy.typing import NDArray

from ..core.board import DEFAULT_HEIGHT, WIDTH
from .colour_palette import (
    CLEAN,
    EMPTY,
    EMPTY_DIST,
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

# A tetromino. Also the whole budget for content hanging OVER THE VOID: a
# Tetris board has one piece in flight and everything else is held up by
# what is under it, so a frame showing more than this is not a board. See
# :meth:`ColourTracker._blind`.
PIECE_CELLS = 4


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
        if int(np.count_nonzero((labels != EMPTY) & ~grounded)) > PIECE_CELLS:
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

        falling = self._pick_falling(labels, grounded)
        if falling is not None:
            falling = self._named(falling)

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

        (2) At most a tetromino of content hangs OVER THE VOID: everything
        else on a Tetris board is held up by what is under it, sideways
        links included (:func:`_grounded`), and there is only ever one
        piece in flight. This is the premise the shipped engine's
        confidence gate rests on too (SPEC.md, grid.py), and it catches
        what (1) cannot: a flat panel floating over the playfield -- ROAS
        Stacker's own "ROW CLEARED" card, a leaderboard, any modal drawn in
        one colour -- arrives as dozens of cells resting on nothing.
        Measured, it costs the corpus NOTHING: over all 528 accepted board
        frames of the six windows the largest airborne reading is 4 cells,
        exactly one tetromino, never once more; the web page, fed past (1),
        shows 58 in a single component.

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

    def _named(self, falling: FallingPiece) -> FallingPiece:
        """Attach a name, letting a complete sighting overrule the palette.

        Shape is consulted on EVERY complete four-cell sighting, not only
        while the colour is anonymous. A complete sighting is the one piece
        of evidence that can contradict the palette, and a palette that
        cannot be contradicted is a palette that is right forever after the
        first piece -- which on a monochrome theme means naming every later
        piece after the first one. When shape and colour disagree the class
        is retired from naming (:meth:`Palette.witness`) and this tracker
        degrades to naming by shape, exactly where a colour-blind one lives.
        """
        shape = piece_from_cells(falling.cells)
        if shape is not None:
            self.palette.witness(falling.colour_class, shape)
        named = self.palette.piece_of(falling.colour_class)
        if named is None:
            named = shape  # the colour says nothing; this frame still can
        if named == falling.piece:
            return falling
        return FallingPiece(named, falling.cells, falling.colour_class, falling.floating)

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
        return [part for part in _connected(young) if len(part) <= 4]

    def _pick_falling(
        self, labels: NDArray[np.int16], grounded: NDArray[np.bool_]
    ) -> FallingPiece | None:
        """The one component that is in flight, if any."""
        best: tuple[tuple[int, int, int], FallingPiece] | None = None
        for label, component in _components(labels):
            for cells in self._candidates(label, component):
                best = self._rank(best, label, cells, grounded)
        return None if best is None else best[1]

    def _rank(
        self,
        best: tuple[tuple[int, int, int], FallingPiece] | None,
        label: int,
        cells: frozenset[Cell],
        grounded: NDArray[np.bool_],
    ) -> tuple[tuple[int, int, int], FallingPiece] | None:
        """Keep whichever of ``best`` and ``cells`` is the better candidate.

        Floating beats resting, younger beats older, higher beats lower.
        """
        floating = not any(grounded[r, c] for r, c in cells)
        youth = min(int(self._age[r, c]) for r, c in cells)
        if not floating and youth >= self.settle_frames:
            return best  # resting and old: settled board, not a piece
        rank = (0 if floating else 1, youth, min(r for r, _ in cells))
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

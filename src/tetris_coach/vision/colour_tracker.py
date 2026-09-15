"""A falling-piece tracker with per-cell state and no global commit.

PARALLEL PROTOTYPE. Nothing here is wired into ``app.py``/``cli.py``; the
shipped tracker (``pieces_vision.py`` + ``state.py``, ~2200 lines with the
board reading it needs) is untouched and answers the same questions from a
different representation.

The shipped design keeps ONE authoritative memory — the committed stack —
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

The only memory is per cell — the colour it shows and how many frames it has
shown it — plus the colour->piece palette, which only ever grows. A misread
frame therefore costs exactly that frame: the next one is read from scratch,
so nothing can wedge, and there is no state a reset would have to rebuild.

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
    own_paint_states,
    piece_from_cells,
)
from .colour_preview import identify_preview
from .grid import HINT_PAINT, OwnPaint

Cell = tuple[int, int]

# Frames a cell must hold its colour before it counts as settled board.
SETTLE_FRAMES = 3


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
            return self._blind()
        colours = board_colours(board, self.rows, self.cols)
        painted = own_paint_states(board, self.rows, self.cols, self._paint)
        self.palette.update_background(colours, self._observable & (painted == CLEAN))
        # The box is read before the board is segmented, so a colour the box
        # names this frame already names the piece entering on it.
        if next_crop is not None:
            self._read_preview(next_crop)
        labels = self.palette.classify(colours, self._observable, painted)

        if self._labels is not None:
            self._age = np.where(labels != self._labels, 0, self._age + 1)
        self._labels = labels

        falling = self._pick_falling(labels)
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

    def _blind(self) -> FrameReport:
        """The frame is not a board: read nothing from it, and say so.

        Without this the tracker read whatever was on screen. Measured, the
        fourteen frames of ``pale_piece`` where a web page has replaced the
        game were reported as a stack of 0b11111111 / 0b1000 / 0b11111100
        and interned six junk colour classes into the palette PERMANENTLY
        -- and a palette class never shrinks, so one occluded frame is a
        lasting corruption of the one memory this design does keep.

        Nothing is read and nothing is written: not the palette, not the
        background, not the cell ages, not the NEXT box. The stack count IS
        carried across, so the frame the board comes back does not look
        like a line clear to :meth:`_events`.

        What is reported is silence, not the last hint. A tracker that
        holds its advice over a board it cannot see is the frozen coach
        this design exists to avoid, so the honest output for a frame that
        says nothing is nothing -- and ``board_visible`` lets a caller that
        would rather hold make that choice itself.
        """
        self._falling = None
        return FrameReport(
            falling=None,
            stack_rows=self._stack_rows,
            next_piece=self._next_piece,
            events=(),
            cleared_rows=0,
            board_visible=False,
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
        """Name the NEXT piece, and teach the palette its colour."""
        reading = identify_preview(crop)
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

    def _pick_falling(self, labels: NDArray[np.int16]) -> FallingPiece | None:
        """The one component that is in flight, if any."""
        grounded = _grounded(labels)
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
